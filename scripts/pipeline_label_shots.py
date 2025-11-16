#!/usr/bin/env python3
"""Attach shotdetail labels to play segments."""

import argparse
from pathlib import Path
from typing import Dict, Set

import pandas as pd

SEGMENT_COLS = [
    "play_id",
    "game_id",
    "start_event",
    "end_event",
    "result_type",
    "offense_team_id",
]

SHOT_COLUMNS = [
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "EVENT_TYPE",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_MADE_FLAG",
]

OUTPUT_COLUMNS = None  # keep original + new shot columns

SHOT_FIELD_MAP = {
    "GAME_EVENT_ID": "shot_event_id",
    "PLAYER_ID": "shot_player_id",
    "PLAYER_NAME": "shot_player_name",
    "TEAM_ID": "shot_team_id",
    "EVENT_TYPE": "shot_event_type",
    "ACTION_TYPE": "shot_action_type",
    "SHOT_TYPE": "shot_type",
    "SHOT_ZONE_BASIC": "shot_zone_basic",
    "SHOT_ZONE_AREA": "shot_zone_area",
    "SHOT_ZONE_RANGE": "shot_zone_range",
    "SHOT_DISTANCE": "shot_distance",
    "LOC_X": "shot_loc_x",
    "LOC_Y": "shot_loc_y",
    "SHOT_MADE_FLAG": "shot_made_flag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=Path("processed/play_segments"),
        help="Directory containing base play segments parquet files",
    )
    parser.add_argument(
        "--shotdetail",
        type=Path,
        required=True,
        help="CSV file with shotdetail data (e.g., nbadata/shotdetail_2016.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/play_segments_shots"),
        help="Directory to store annotated play segments",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional number of games to process",
    )
    return parser.parse_args()


def load_shots(path: Path, game_ids: Set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=SHOT_COLUMNS)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].isin(game_ids)]
    return df


def annotate_game(seg_path: Path, shots_df: pd.DataFrame, output_dir: Path) -> None:
    segments = pd.read_parquet(seg_path)
    if segments.empty:
        return
    game_id = seg_path.stem
    game_shots = shots_df[shots_df["GAME_ID"] == game_id]
    if game_shots.empty:
        out_path = output_dir / seg_path.name
        segments.to_parquet(out_path, index=False)
        print(f"{game_id}: no shots found")
        return

    for target in SHOT_FIELD_MAP.values():
        segments[target] = pd.NA

    for _, shot in game_shots.iterrows():
        event_id = int(shot["GAME_EVENT_ID"])
        mask = (segments["start_event"] <= event_id) & (segments["end_event"] >= event_id)
        matched = segments.index[mask]
        if len(matched) == 0:
            print(f"[WARN] {game_id}: shot event {event_id} unmatched")
            continue
        if len(matched) > 1:
            print(f"[WARN] {game_id}: shot event {event_id} matched multiple plays")
        idx = matched[0]
        for src, dst in SHOT_FIELD_MAP.items():
            segments.at[idx, dst] = shot[src]

    out_path = output_dir / seg_path.name
    segments.to_parquet(out_path, index=False)
    print(f"Saved shot labels for {game_id}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    segment_files = sorted(args.segments_dir.glob("002*.parquet"))
    if args.limit_games is not None:
        segment_files = segment_files[: args.limit_games]

    game_ids = {p.stem for p in segment_files}
    shots_df = load_shots(args.shotdetail, game_ids)

    for seg_path in segment_files:
        annotate_game(seg_path, shots_df, args.output_dir)


if __name__ == "__main__":
    main()
