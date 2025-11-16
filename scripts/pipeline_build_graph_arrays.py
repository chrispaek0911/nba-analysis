#!/usr/bin/env python3
"""Convert play frame tables into dense node-time arrays for graph construction."""

import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

FEATURES = ["x", "y", "z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--play-frames-dir",
        type=Path,
        default=Path("processed/play_frames"),
        help="Directory with per-play frame NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/graph_arrays"),
        help="Destination for dense arrays",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional limit on number of games",
    )
    parser.add_argument(
        "--offset-games",
        type=int,
        default=0,
        help="Skip this many games before processing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip games already processed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="ProcessPool workers for games",
    )
    return parser.parse_args()


def load_play(npz_path: Path) -> Dict:
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data["frames"]
        meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) else data["meta"]
    return {"frames": frames, "meta": meta}


def node_order(frames_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    # Ball node first
    nodes = [(-1, -1)]
    players = {(int(row.team_id), int(row.player_id)) for row in frames_df.itertuples() if row.team_id != -1}
    nodes.extend(sorted(players))
    team_ids = np.array([t for t, _ in nodes], dtype=np.int32)
    player_ids = np.array([p for _, p in nodes], dtype=np.int64)
    index_map = {node: idx for idx, node in enumerate(nodes)}
    return player_ids, team_ids, index_map


def build_arrays(frames: np.ndarray) -> Dict[str, np.ndarray]:
    order = np.argsort(frames["frame_idx"], kind="mergesort")
    frames_sorted = frames[order]
    frame_ids, frame_inverse = np.unique(frames_sorted["frame_idx"], return_inverse=True)

    df_for_nodes = pd.DataFrame(frames_sorted)
    player_ids, team_ids, index_map = node_order(df_for_nodes)
    T = len(frame_ids)
    N = len(player_ids)
    positions = np.zeros((T, N, 3), dtype=np.float32)
    event_ids = np.zeros(T, dtype=np.int32)
    timestamps = np.zeros(T, dtype=np.int64)
    game_clocks = np.zeros(T, dtype=np.float32)
    shot_clocks = np.zeros(T, dtype=np.float32)
    filled = np.zeros(T, dtype=bool)

    for idx, row in enumerate(frames_sorted):
        t_idx = frame_inverse[idx]
        key = (int(row["team_id"]), int(row["player_id"]))
        node_idx = index_map[key]
        positions[t_idx, node_idx] = [row["x"], row["y"], row["z"]]
        if not filled[t_idx]:
            event_ids[t_idx] = int(row["event_id"])
            timestamps[t_idx] = int(row["timestamp_ms"])
            game_clocks[t_idx] = float(row["game_clock"])
            shot_clocks[t_idx] = float(row["shot_clock"])
            filled[t_idx] = True

    return {
        "frame_ids": frame_ids.astype(np.int32),
        "event_ids": event_ids,
        "timestamps": timestamps,
        "game_clocks": game_clocks,
        "shot_clocks": shot_clocks,
        "node_player_ids": player_ids,
        "node_team_ids": team_ids,
        "positions": positions,
    }


def process_game(game_dir: Path, out_root: Path, skip_existing: bool) -> Tuple[str, int]:
    if not game_dir.is_dir():
        return game_dir.name, 0
    game_id = game_dir.name
    out_path = out_root / f"{game_id}.npz"
    if out_path.exists() and skip_existing:
        return game_id, -1
    payload = {"play_keys": []}
    count = 0
    for npz_path in sorted(game_dir.glob("play_*.npz")):
        play = load_play(npz_path)
        arrays = build_arrays(play["frames"])
        play_key = f"play_{int(play['meta'].get('play_id', count+1)):04d}"
        payload["play_keys"].append(play_key)
        arrays["result_type"] = np.array(play["meta"].get("result_type") or "", dtype="<U32")
        arrays["start_frame"] = np.array(play["meta"].get("start_frame") or -1, dtype=np.int32)
        arrays["end_frame"] = np.array(play["meta"].get("end_frame") or -1, dtype=np.int32)
        for name, value in arrays.items():
            payload[f"{play_key}_{name}"] = value
        count += 1
    payload["play_keys"] = np.array(payload["play_keys"], dtype="<U16")
    np.savez_compressed(out_path, **payload)
    return game_id, count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games = sorted(args.play_frames_dir.iterdir())
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_game, game_dir, args.output_dir, args.skip_existing): game_dir.name
            for game_dir in games
        }
        for future in concurrent.futures.as_completed(futures):
            game_id, count = future.result()
            if count == -1:
                print(f"[SKIP] {game_id} graph arrays exist")
            else:
                print(f"{game_id}: graph arrays saved ({count} plays)")


if __name__ == "__main__":
    main()
