#!/usr/bin/env python3
"""Slice tracking frames per play segment."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PLAY_COLS = [
    "play_id",
    "game_id",
    "start_frame",
    "end_frame",
    "start_event",
    "end_event",
    "result_type",
    "offense_team_id",
]

FRAME_COLS = [
    "frame_idx",
    "event_id",
    "period",
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
        "--segments-dir",
        type=Path,
        default=Path("processed/play_segments"),
        help="Directory with play metadata parquet files",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("processed/tracking/frames"),
        help="Directory with per-game tracking frame CSV.GZ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/play_frames"),
        help="Directory to store per-play frame arrays",
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
        help="Skip games that already have extracted frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    segment_files = sorted(args.segments_dir.glob("002*.parquet"))
    if args.offset_games:
        segment_files = segment_files[args.offset_games :]
    if args.limit_games is not None:
        segment_files = segment_files[: args.limit_games]

    for seg_path in segment_files:
        plays = pd.read_parquet(seg_path, columns=PLAY_COLS)
        if plays.empty:
            continue
        game_id = seg_path.stem
        frame_path = args.frames_dir / f"{game_id}.csv.gz"
        if not frame_path.exists():
            print(f"[WARN] Missing frames for {game_id}")
            continue
        frame_df = pd.read_csv(frame_path, usecols=FRAME_COLS)
        game_out_dir = args.output_dir / game_id
        if game_out_dir.exists() and args.skip_existing:
            print(f"[SKIP] {game_id} already extracted")
            continue
        game_out_dir.mkdir(parents=True, exist_ok=True)

        for _, play in plays.iterrows():
            start_frame = play["start_frame"]
            end_frame = play["end_frame"]
            if pd.isna(start_frame) or pd.isna(end_frame):
                continue
            mask = (frame_df["frame_idx"] >= start_frame) & (frame_df["frame_idx"] <= end_frame)
            slice_df = frame_df.loc[mask]
            if slice_df.empty:
                continue
            arr = slice_df.to_records(index=False)
            meta = play.to_dict()
            out_path = game_out_dir / f"play_{int(play['play_id']):04d}.npz"
            np.savez_compressed(out_path, frames=arr, meta=meta)
        print(f"Saved frames for {game_id}")


if __name__ == "__main__":
    main()
