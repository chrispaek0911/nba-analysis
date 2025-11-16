#!/usr/bin/env python3
"""Estimate ball handler per frame and save diagnostics for pass detection."""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

FIELDS = [
    "frame_idx",
    "event_id",
    "timestamp_ms",
    "game_clock",
    "shot_clock",
    "team_id",
    "player_id",
    "x",
    "y",
    "z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--play-dir",
        type=Path,
        default=Path("processed/play_frames"),
        help="Directory containing per-play frame NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/play_ball_handlers"),
        help="Directory to store ball handler csv files",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional limit on number of games to process",
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
        help="Skip games that already have handler CSVs",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=3.0,
        help="Maximum distance to consider player controlling the ball (feet)",
    )
    return parser.parse_args()


def load_play(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data["frames"]
        meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) else data["meta"]
    return {"frames": frames, "meta": meta}


def compute_handlers(play_data: Dict[str, np.ndarray], distance_threshold: float) -> pd.DataFrame:
    frames = play_data["frames"]
    meta = play_data["meta"]
    df = pd.DataFrame(frames)[FIELDS]
    ball = df[df["team_id"] == -1]
    players = df[df["team_id"] != -1]
    merged = players.merge(ball, on="frame_idx", suffixes=("", "_ball"))
    merged["dist"] = np.sqrt((merged["x"] - merged["x_ball"]) ** 2 + (merged["y"] - merged["y_ball"]) ** 2)
    merged = merged.sort_values(["frame_idx", "dist"])
    closest = merged.groupby("frame_idx").first().reset_index()
    closest["has_control"] = closest["dist"] <= distance_threshold
    closest["play_id"] = meta["play_id"]
    closest["game_id"] = meta["game_id"]
    closest["result_type"] = meta.get("result_type")
    return closest[[
        "game_id",
        "play_id",
        "frame_idx",
        "event_id",
        "timestamp_ms",
        "game_clock",
        "shot_clock",
        "player_id",
        "team_id",
        "dist",
        "has_control",
        "x_ball",
        "y_ball",
        "z_ball",
        "x",
        "y",
        "z",
        "result_type",
    ]]


def main() -> None:
    args = parse_args()
    play_dir = args.play_dir
    games = sorted(play_dir.iterdir())
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for game_path in games:
        if not game_path.is_dir():
            continue
        game_id = game_path.name
        out_dir = args.output_dir / game_id
        if out_dir.exists() and args.skip_existing:
            print(f"[SKIP] {game_id} handlers already exist")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_files = sorted(game_path.glob("play_*.npz"))
        for npz_file in npz_files:
            play = load_play(npz_file)
            handler_df = compute_handlers(play, args.distance_threshold)
            out_path = out_dir / f"{npz_file.stem}.csv"
            handler_df.to_csv(out_path, index=False)
        print(f"Computed handlers for {game_id}")


if __name__ == "__main__":
    main()
