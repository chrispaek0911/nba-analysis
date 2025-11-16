#!/usr/bin/env python3
"""Detect passes from graph_arrays using the shared pass detection logic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pass_detection import HandlerRow, PlayContext, detect_passes, suppress_shot_turnovers

REQUIRED_ARRAY_FIELDS = [
    "positions",
    "frame_ids",
    "node_player_ids",
    "node_team_ids",
    "game_clocks",
    "shot_clocks",
]

OUTPUT_COLUMNS = [
    "game_id",
    "play_id",
    "pass_id",
    "passer_id",
    "receiver_id",
    "passer_team_id",
    "receiver_team_id",
    "start_frame",
    "end_frame",
    "air_frames",
    "ball_max_z",
    "is_turnover_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-arrays-dir", type=Path, default=Path("processed/graph_arrays"))
    parser.add_argument("--play-labels-dir", type=Path, default=Path("processed/play_segments_shots"))
    parser.add_argument("--pbp-dir", type=Path, default=Path("processed/pbp_with_frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/play_passes"))
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--offset-games", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--distance-threshold", type=float, default=3.0)
    parser.add_argument("--max-control-z", type=float, default=9.0)
    parser.add_argument("--max-control-speed", type=float, default=2.0)
    parser.add_argument("--offense-control-frames", type=int, default=3)
    parser.add_argument("--defense-control-frames", type=int, default=5)
    parser.add_argument("--defense-max-control-z", type=float, default=8.5)
    parser.add_argument("--defense-z-velocity-threshold", type=float, default=0.7)
    parser.add_argument("--min-air-frames", type=int, default=2)
    parser.add_argument("--initial-skip-frames", type=int, default=5)
    parser.add_argument("--deflection-confirm-frames", type=int, default=2)
    parser.add_argument("--shot-guard-frames", type=int, default=12)
    parser.add_argument("--shot-z-threshold", type=float, default=10.5)
    parser.add_argument("--shot-speed-threshold", type=float, default=3.0)
    parser.add_argument("--shot-ball-height-threshold", type=float, default=10.5)
    parser.add_argument("--shot-turnover-window", type=int, default=150)
    parser.add_argument("--jump-ball-guard-frames", type=int, default=10)
    parser.add_argument("--made-shot-cooldown-boost", type=int, default=8)
    parser.add_argument("--made-shot-endline-threshold", type=float, default=4.0)
    parser.add_argument("--made-shot-floor-z", type=float, default=4.0)
    parser.add_argument("--court-length", type=float, default=94.0)
    return parser.parse_args()


def load_play_metadata(game_id: str, labels_dir: Path) -> Dict[int, Dict]:
    path = labels_dir / f"{game_id}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        for k, v in list(record.items()):
            if pd.isna(v):
                record[k] = None
        out[int(row["play_id"])] = record
    return out


def load_pbp_events(game_id: str, pbp_dir: Path) -> Dict[int, int]:
    path = pbp_dir / f"{game_id}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["EVENTNUM", "EVENTMSGTYPE"])
    return {int(row.EVENTNUM): int(row.EVENTMSGTYPE) for row in df.itertuples()}


def load_game_arrays(game_id: str, graph_arrays_dir: Path, play_ids: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    npz_path = graph_arrays_dir / f"{game_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing graph arrays for {game_id}: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    plays: Dict[int, Dict[str, np.ndarray]] = {}
    for pid in play_ids:
        prefix = f"play_{pid:04d}"
        if any(f"{prefix}_{field}" not in data for field in REQUIRED_ARRAY_FIELDS):
            continue
        plays[pid] = {field: data[f"{prefix}_{field}"] for field in REQUIRED_ARRAY_FIELDS}
    return plays


def compute_handler_rows(arrays: Dict[str, np.ndarray], args: argparse.Namespace) -> List[HandlerRow]:
    positions = arrays["positions"]
    frame_ids = arrays["frame_ids"].astype(int)
    node_players = arrays["node_player_ids"].astype(int)
    node_teams = arrays["node_team_ids"].astype(int)
    ball_idx = int(np.where(node_teams == -1)[0][0])
    player_indices = np.where(node_teams >= 0)[0]
    ball_xy = positions[:, ball_idx, :2]
    ball_z = positions[:, ball_idx, 2]
    ball_speed = np.zeros(len(frame_ids), dtype=float)
    ball_dz = np.zeros(len(frame_ids), dtype=float)
    if len(frame_ids) > 1:
        diffs = np.diff(ball_xy, axis=0)
        ball_speed[1:] = np.linalg.norm(diffs, axis=1)
        ball_dz[1:] = np.diff(ball_z)
    rows: List[HandlerRow] = []
    for t in range(len(frame_ids)):
        player_xy = positions[t, player_indices, :2]
        dists = np.linalg.norm(player_xy - ball_xy[t], axis=1)
        closest = int(np.argmin(dists))
        node_idx = player_indices[closest]
        has_control = (
            dists[closest] <= args.distance_threshold
            and ball_z[t] <= args.max_control_z
            and ball_speed[t] <= args.max_control_speed
        )
        rows.append(
            HandlerRow(
                array_idx=t,
                frame_idx=int(frame_ids[t]),
                player_id=int(node_players[node_idx]),
                team_id=int(node_teams[node_idx]),
                distance=float(dists[closest]),
                ball_z=float(ball_z[t]),
                ball_dz=float(ball_dz[t]),
                ball_speed=float(ball_speed[t]),
                has_control=bool(has_control),
                ball_x=float(ball_xy[t, 0]),
                ball_y=float(ball_xy[t, 1]),
                event_id=None,
                timestamp_ms=None,
                game_clock=float(arrays["game_clocks"][t]) if "game_clocks" in arrays else None,
                shot_clock=float(arrays["shot_clocks"][t]) if "shot_clocks" in arrays else None,
            )
        )
    return rows


def convert_pass_records(pass_records, arrays: Dict[str, np.ndarray], game_id: str, play_id: int, offset: int) -> List[Dict]:
    frame_ids = arrays["frame_ids"].astype(int)
    output: List[Dict] = []
    for idx, rec in enumerate(pass_records, start=1):
        output.append(
            {
                "game_id": game_id,
                "play_id": play_id,
                "pass_id": offset + idx,
                "passer_id": rec.passer_id,
                "receiver_id": rec.receiver_id,
                "passer_team_id": rec.passer_team,
                "receiver_team_id": rec.receiver_team,
                "start_frame": int(frame_ids[rec.start_idx]),
                "end_frame": int(frame_ids[rec.end_idx]),
                "air_frames": rec.air_frames,
                "ball_max_z": rec.ball_max_z,
                "is_turnover_pass": rec.is_turnover,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = sorted(p.stem for p in args.graph_arrays_dir.glob("002*.npz"))
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]

    for game_id in games:
        out_path = args.output_dir / f"{game_id}.parquet"
        if out_path.exists() and args.skip_existing:
            print(f"[SKIP] {game_id} passes already computed")
            continue

        metadata = load_play_metadata(game_id, args.play_labels_dir)
        if not metadata:
            print(f"[WARN] no play metadata for {game_id}")
            continue
        pbp_events = load_pbp_events(game_id, args.pbp_dir)
        play_ids = sorted(metadata.keys())
        arrays_by_play = load_game_arrays(game_id, args.graph_arrays_dir, play_ids)
        pass_rows: List[Dict] = []
        global_pass_idx = 0
        for pid in play_ids:
            arrays = arrays_by_play.get(pid)
            if arrays is None:
                continue
            rows = compute_handler_rows(arrays, args)
            if not rows:
                continue
            play_meta = metadata.get(pid, {})
            start_event = play_meta.get("start_event")
            try:
                start_event = int(start_event) if start_event is not None else None
            except (TypeError, ValueError):
                start_event = None
            context = PlayContext(
                start_event_msg_type=pbp_events.get(start_event),
                jump_ball_guard_frames=args.jump_ball_guard_frames,
            )
            pass_records = detect_passes(rows, play_meta, args, context)
            pass_records = suppress_shot_turnovers(pass_records, rows, play_meta, args)
            converted = convert_pass_records(pass_records, arrays, game_id, pid, global_pass_idx)
            pass_rows.extend(converted)
            global_pass_idx += len(pass_records)
        df = pd.DataFrame(pass_rows, columns=OUTPUT_COLUMNS) if pass_rows else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df.to_parquet(out_path, index=False)
        print(f"{game_id}: wrote {len(pass_rows)} passes -> {out_path}")


if __name__ == "__main__":
    main()
