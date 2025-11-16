#!/usr/bin/env python3
"""Reconstruct event_index.csv from processed tracking frame CSVs."""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

OUTPUT_COLUMNS = [
    "game_id",
    "event_id",
    "frame_start",
    "frame_end",
    "start_period",
    "start_game_clock",
    "start_shot_clock",
    "end_period",
    "end_game_clock",
    "end_shot_clock",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("processed/tracking/frames"),
        help="Directory containing per-game frame CSV.GZ files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/tracking/event_index.csv"),
        help="Destination CSV path for rebuilt event index",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("processed/tracking/event_index_summary.csv"),
        help="Optional summary CSV path",
    )
    return parser.parse_args()


def process_game(path: Path) -> pd.DataFrame:
    game_id = Path(path.stem).stem  # remove trailing .csv from .csv.gz
    usecols = ["frame_idx", "event_id", "period", "game_clock", "shot_clock"]
    df = pd.read_csv(path, usecols=usecols)
    # collapse to one row per frame_idx per event
    frame_meta = df.drop_duplicates(subset=["event_id", "frame_idx"])
    group = frame_meta.groupby("event_id", sort=True)
    frame_start = group["frame_idx"].min().rename("frame_start")
    frame_end = group["frame_idx"].max().rename("frame_end")

    idx_start = group["frame_idx"].idxmin()
    start_info = frame_meta.loc[idx_start, ["period", "game_clock", "shot_clock"]]
    start_info = start_info.rename(
        columns={
            "period": "start_period",
            "game_clock": "start_game_clock",
            "shot_clock": "start_shot_clock",
        }
    )

    idx_end = group["frame_idx"].idxmax()
    end_info = frame_meta.loc[idx_end, ["period", "game_clock", "shot_clock"]]
    end_info = end_info.rename(
        columns={
            "period": "end_period",
            "game_clock": "end_game_clock",
            "shot_clock": "end_shot_clock",
        }
    )

    merged = (
        pd.concat([frame_start, frame_end], axis=1)
        .join(start_info)
        .join(end_info)
        .reset_index()
    )
    merged.insert(0, "game_id", game_id)
    return merged


def main() -> None:
    args = parse_args()
    paths = sorted(args.frames_dir.glob("002*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No frame files found in {args.frames_dir}")

    rows: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    for path in paths:
        try:
            game_df = process_game(path)
        except Exception as exc:
            print(f"[WARN] Failed to process {path.name}: {exc}")
            continue
        rows.append(game_df)
        game_id = Path(path.stem).stem
        summary_rows.append(
            {
                "game_id": game_id,
                "events": len(game_df),
                "frame_min": int(game_df["frame_start"].min()),
                "frame_max": int(game_df["frame_end"].max()),
            }
        )
        print(f"Processed {path.stem}: {len(game_df)} events")

    if not rows:
        raise RuntimeError("No games processed; event index not created")

    result = pd.concat(rows, ignore_index=True)
    result = result[OUTPUT_COLUMNS]
    result = result.sort_values(["game_id", "event_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
