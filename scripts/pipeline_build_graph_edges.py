#!/usr/bin/env python3
"""Generate edge lists (team/opponent/pass) for each play graph."""

import argparse
import concurrent.futures
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

BASE_EDGE_DTYPE = np.dtype([
    ("frame_idx", "<i4"),
    ("src", "<i2"),
    ("dst", "<i2"),
    ("weight", "<f4"),
    ("edge_type", "<i1"),
])

PASS_EDGE_DTYPE = np.dtype([
    ("start_frame", "<i4"),
    ("end_frame", "<i4"),
    ("src", "<i2"),
    ("dst", "<i2"),
    ("assist_flag", "<i1"),
    ("turnover_flag", "<i1"),
    ("air_frames", "<i2"),
    ("ball_max_z", "<f4"),
])

TURNOVER_ADJUST_FRAMES = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-arrays-dir", type=Path, default=Path("processed/graph_arrays"))
    parser.add_argument("--play-metadata-dir", type=Path, default=Path("processed/play_segments_shots"))
    parser.add_argument("--passes-dir", type=Path, default=Path("processed/play_passes"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/graph_edges"))
    parser.add_argument("--same-team-threshold", type=float, default=70.0)
    parser.add_argument("--opp-team-threshold", type=float, default=50.0)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--offset-games", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_passes(passes_path: Path) -> pd.DataFrame:
    if not passes_path.exists():
        columns = ["play_id", "passer_id", "receiver_id", "start_frame", "end_frame", "air_frames", "ball_max_z", "is_turnover_pass"]
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(passes_path)


def build_base_edges(arrays: Dict[str, np.ndarray], same_threshold: float, opp_threshold: float) -> np.ndarray:
    positions = arrays["positions"]  # (T,N,3)
    node_team_ids = arrays["node_team_ids"]
    T, N, _ = positions.shape

    coords_xy = positions[..., :2]
    diffs = coords_xy[:, :, None, :] - coords_xy[:, None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)

    team_ids = node_team_ids
    valid_players = team_ids >= 0
    same_mask = (team_ids[:, None] == team_ids[None, :]) & valid_players[:, None] & valid_players[None, :]
    opp_mask = valid_players[:, None] & valid_players[None, :] & (team_ids[:, None] != team_ids[None, :])
    np.fill_diagonal(same_mask, False)
    np.fill_diagonal(opp_mask, False)

    edges_list = []
    for edge_type, mask, threshold in (
        (0, same_mask, same_threshold),
        (1, opp_mask, opp_threshold),
    ):
        if not mask.any():
            continue
        mask3d = mask[None, :, :]
        valid = (dist <= threshold) & mask3d
        t_idx, src_idx, dst_idx = np.where(valid)
        if len(t_idx) == 0:
            continue
        weights = np.ones(len(t_idx), dtype=np.float32)
        edge_block = np.empty(len(t_idx), dtype=BASE_EDGE_DTYPE)
        edge_block["frame_idx"] = arrays["frame_ids"][t_idx]
        edge_block["src"] = src_idx.astype("<i2")
        edge_block["dst"] = dst_idx.astype("<i2")
        edge_block["weight"] = weights
        edge_block["edge_type"] = edge_type
        edges_list.append(edge_block)
    if edges_list:
        return np.concatenate(edges_list, axis=0)
    return np.empty(0, dtype=BASE_EDGE_DTYPE)


def normalize_meta(meta_row: Optional[pd.Series]):
    if meta_row is None:
        return None, None, None
    result_type = meta_row.get("result_type")
    shot_frame = meta_row.get("end_frame")
    if isinstance(shot_frame, float) and math.isnan(shot_frame):
        shot_frame = None
    elif shot_frame is not None:
        shot_frame = int(shot_frame)
    shooter_team = meta_row.get("shot_team_id")
    if isinstance(shooter_team, float) and math.isnan(shooter_team):
        shooter_team = meta_row.get("offense_team_id")
    if isinstance(shooter_team, float) and math.isnan(shooter_team):
        shooter_team = None
    elif shooter_team is not None:
        shooter_team = int(shooter_team)
    return shot_frame, shooter_team, result_type


def adjust_pass_turnovers(
    pass_edges: np.ndarray,
    node_team_ids: np.ndarray,
    shot_frame: Optional[int],
    shooter_team: Optional[int],
    result_type: Optional[str],
) -> np.ndarray:
    if pass_edges.size == 0:
        return pass_edges

    def team_for(node_idx: int) -> Optional[int]:
        if node_idx is None:
            return None
        if node_idx < 0 or node_idx >= len(node_team_ids):
            return None
        team = int(node_team_ids[node_idx])
        return team if team >= 0 else None

    for idx in range(len(pass_edges)):
        if not pass_edges["turnover_flag"][idx]:
            continue
        passer_team = team_for(int(pass_edges["src"][idx]))
        if passer_team is None:
            continue
        for next_idx in range(idx + 1, len(pass_edges)):
            frame_gap = int(pass_edges["start_frame"][next_idx]) - int(pass_edges["end_frame"][idx])
            if frame_gap > TURNOVER_ADJUST_FRAMES:
                break
            receiver_team = team_for(int(pass_edges["dst"][next_idx]))
            if receiver_team == passer_team:
                pass_edges["turnover_flag"][idx] = 0
                break
        if not pass_edges["turnover_flag"][idx]:
            continue
        if (
            shooter_team is not None
            and shot_frame is not None
            and shooter_team == passer_team
            and 0 <= shot_frame - int(pass_edges["end_frame"][idx]) <= TURNOVER_ADJUST_FRAMES
        ):
            pass_edges["turnover_flag"][idx] = 0
            continue
        if result_type:
            normalized = str(result_type).lower()
            if normalized in {"made_shot", "miss_shot", "foul", "foul_committed"}:
                pass_edges["turnover_flag"][idx] = 0
    return pass_edges


def build_pass_edges(arrays: Dict[str, np.ndarray], pass_rows: pd.DataFrame, meta_row: Optional[pd.Series]) -> np.ndarray:
    if pass_rows.empty:
        return np.empty(0, dtype=PASS_EDGE_DTYPE)
    frame_ids = arrays["frame_ids"]
    player_ids = arrays["node_player_ids"]
    player_to_idx = {int(pid): idx for idx, pid in enumerate(player_ids) if pid >= 0}
    pass_rows = pass_rows.sort_values("start_frame")
    records = []
    for row in pass_rows.itertuples(index=False, name="Pass"):
        row_dict = row._asdict()
        passer_idx = player_to_idx.get(int(row_dict["passer_id"]))
        receiver_idx = player_to_idx.get(int(row_dict["receiver_id"]))
        if passer_idx is None or receiver_idx is None:
            continue
        start_idx = np.searchsorted(frame_ids, int(row_dict["start_frame"]))
        if start_idx >= len(frame_ids) or frame_ids[start_idx] != int(row_dict["start_frame"]):
            start_idx = max(0, min(len(frame_ids) - 1, start_idx))
        end_idx = np.searchsorted(frame_ids, int(row_dict["end_frame"]))
        if end_idx >= len(frame_ids) or frame_ids[end_idx] != int(row_dict["end_frame"]):
            end_idx = max(0, min(len(frame_ids) - 1, end_idx))
        rec = np.zeros(1, dtype=PASS_EDGE_DTYPE)
        rec["start_frame"] = frame_ids[start_idx]
        rec["end_frame"] = frame_ids[end_idx]
        rec["src"] = passer_idx
        rec["dst"] = receiver_idx
        rec["assist_flag"] = 1 if row_dict.get("assist_flag") else 0
        rec["turnover_flag"] = 1 if row_dict.get("is_turnover_pass") else 0
        rec["air_frames"] = int(row_dict["air_frames"])
        rec["ball_max_z"] = float(row_dict["ball_max_z"])
        records.append(rec)
    if not records:
        return np.empty(0, dtype=PASS_EDGE_DTYPE)
    pass_edges = np.concatenate(records, axis=0)
    order = np.argsort(pass_edges["start_frame"])
    pass_edges = pass_edges[order]
    shot_frame, shooter_team, result_type = normalize_meta(meta_row)
    pass_edges = adjust_pass_turnovers(pass_edges, arrays["node_team_ids"], shot_frame, shooter_team, result_type)
    return pass_edges


def process_game(game_npz: Path, args) -> Tuple[str, int]:
    game_id = game_npz.stem
    out_path = args.output_dir / f"{game_id}.npz"
    if out_path.exists() and args.skip_existing:
        return game_id, -1

    meta_path = args.play_m_metadata_dir / f"{game_id}.parquet"
    if not meta_path.exists():
        meta_df = pd.DataFrame()
    else:
        meta_df = pd.read_parquet(
            meta_path,
            columns=[
                "play_id",
                "start_frame",
                "end_frame",
                "result_type",
                "shot_team_id",
                "offense_team_id",
            ],
        )

    passes_df = load_passes(args.passes_dir / f"{game_id}.parquet")

    payload = {"play_keys": []}
    count = 0
    with np.load(game_npz, allow_pickle=True) as data:
        play_keys = data["play_keys"]
        for key in play_keys:
            arrays = {name.split(f"{key}_", 1)[1]: data[name] for name in data.files if name.startswith(f"{key}_")}
            arrays["frame_ids"] = data[f"{key}_frame_ids"]
            arrays["node_player_ids"] = data[f"{key}_node_player_ids"]
            arrays["node_team_ids"] = data[f"{key}_node_team_ids"]
            arrays["positions"] = data[f"{key}_positions"]
            base_edges = build_base_edges(arrays, args.same_team_threshold, args.opp_team_threshold)
            play_id = int(key.split("_")[1])
            pass_rows = passes_df[passes_df["play_id"] == play_id] if not passes_df.empty else pd.DataFrame()
            meta_row = None
            if not meta_df.empty:
                subset = meta_df[meta_df["play_id"] == play_id]
                if not subset.empty:
                    meta_row = subset.iloc[0]
            pass_edges = build_pass_edges(arrays, pass_rows, meta_row)
            payload["play_keys"].append(key)
            payload[f"{key}_base_edges"] = base_edges
            payload[f"{key}_pass_edges"] = pass_edges
            count += 1
    payload["play_keys"] = np.array(payload["play_keys"], dtype="<U16")
    np.savez_compressed(out_path, **payload)
    return game_id, count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.play_m_metadata_dir = args.play_metadata_dir  # alias for convenience
    games = sorted(args.graph_arrays_dir.glob("002*.npz"))
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_game, game_npz, args): game_npz.stem
            for game_npz in games
        }
        for future in concurrent.futures.as_completed(futures):
            game_id, count = future.result()
            if count == -1:
                print(f"[SKIP] {game_id} edges exist")
            else:
                print(f"{game_id}: edges saved ({count} plays)")


if __name__ == "__main__":
    main()
