#!/usr/bin/env python3
"""Merge NBA play-by-play CSVs with tracking event frame indices."""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

REQUIRED_EVENT_INDEX_COLUMNS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbp-dir",
        type=Path,
        default=Path("data/events"),
        help="Directory containing per-game PBP CSV files",
    )
    parser.add_argument(
        "--event-index",
        type=Path,
        default=Path("processed/tracking/event_index.csv"),
        help="CSV produced from tracking preprocessing (event -> frame mapping)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/pbp_with_frames"),
        help="Directory to store merged PBP files (per game, parquet)",
    )
    return parser.parse_args()


def load_event_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event index file not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_EVENT_INDEX_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Event index missing columns: {missing}")
    # Normalize dtypes
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str).str.replace(".0", "", regex=False)
    df["game_id"] = df["game_id"].str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    for col in ("frame_start", "frame_end"):
        df[col] = df[col].astype(int)
    return df


def merge_game(pbp_path: Path, event_index: pd.DataFrame) -> pd.DataFrame:
    pbp = pd.read_csv(pbp_path)
    if "GAME_ID" not in pbp.columns or "EVENTNUM" not in pbp.columns:
        raise ValueError(f"PBP file {pbp_path} missing GAME_ID/EVENTNUM")
    pbp = pbp.copy()
    pbp["GAME_ID"] = pbp["GAME_ID"].astype(str).str.replace(".0", "", regex=False)
    pbp["GAME_ID"] = pbp["GAME_ID"].str.zfill(10)
    pbp["EVENTNUM"] = pbp["EVENTNUM"].astype(int)
    game_id = pbp["GAME_ID"].iloc[0]
    idx = event_index[event_index["game_id"] == game_id]
    if idx.empty:
        raise ValueError(f"Event index has no entries for {game_id}")
    merged = pbp.merge(
        idx,
        how="left",
        left_on=["GAME_ID", "EVENTNUM"],
        right_on=["game_id", "event_id"],
    )
    merged = merged.drop(columns=["game_id", "event_id"])
    merged = merged.sort_values("EVENTNUM").reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    event_index = load_event_index(args.event_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, str]] = []

    game_ids = sorted(event_index["game_id"].unique())
    if not game_ids:
        raise RuntimeError("Event index has no game ids")

    for game_id in game_ids:
        pbp_path = args.pbp_dir / f"{game_id}.csv"
        if not pbp_path.exists():
            print(f"[WARN] Missing PBP file for {game_id} at {pbp_path}")
            continue
        try:
            merged = merge_game(pbp_path, event_index)
        except Exception as exc:
            print(f"[WARN] Skipping {pbp_path.name}: {exc}")
            continue
        game_id = merged["GAME_ID"].iloc[0]
        out_path = args.output_dir / f"{game_id}.parquet"
        merged.to_parquet(out_path, index=False)
        matched = merged["frame_start"].notna().sum()
        summaries.append(
            {
                "game_id": game_id,
                "events_total": str(len(merged)),
                "events_with_frames": str(matched),
                "output_path": str(out_path),
            }
        )
        print(f"Processed {game_id}: {matched}/{len(merged)} events matched")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            args.output_dir / "summary.csv", index=False
        )


if __name__ == "__main__":
    main()
