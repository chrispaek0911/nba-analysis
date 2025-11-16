#!/usr/bin/env python3
"""Annotate play segments with rebound/turnover/block/foul/free-throw labels."""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REB_RE = re.compile(r"Off:(\d+) Def:(\d+)", re.IGNORECASE)

PBP_COLS = [
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "PLAYER1_ID",
    "PLAYER2_ID",
    "PLAYER3_ID",
]

PLAY_COLS = [
    "play_id",
    "game_id",
    "start_event",
    "end_event",
]

NEW_COLUMNS = {
    "rebound_type": pd.NA,
    "rebound_player_id": pd.NA,
    "turnover_type": pd.NA,
    "turnover_player_id": pd.NA,
    "steal_player_id": pd.NA,
    "block_player_id": pd.NA,
    "foul_type": pd.NA,
    "foul_player_id": pd.NA,
    "free_throw_attempts": 0,
    "free_throw_made": 0,
}


def parse_rebound(desc: str) -> Optional[str]:
    if not desc:
        return None
    match = REB_RE.search(desc)
    if not match:
        return None
    off = int(match.group(1))
    deff = int(match.group(2))
    if deff > 0:
        return "def"
    if off > 0:
        return "off"
    return None


def is_miss(description: str) -> bool:
    if not description:
        return False
    return "MISS" in description.upper()


DESC_COLS = ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]


def annotate_play(events: pd.DataFrame) -> Dict[str, object]:
    labels: Dict[str, object] = {k: (v if not isinstance(v, pd.Series) else v.copy()) for k, v in NEW_COLUMNS.items()}
    rebounds = events[events["EVENTMSGTYPE"] == 4]
    if not rebounds.empty:
        last = rebounds.iloc[-1]
        desc = last["HOMEDESCRIPTION"] or last["VISITORDESCRIPTION"] or last["NEUTRALDESCRIPTION"]
        r_type = parse_rebound(desc or "")
        if r_type:
            labels["rebound_type"] = r_type
        labels["rebound_player_id"] = int(last["PLAYER1_ID"]) if pd.notna(last["PLAYER1_ID"]) else pd.NA

    turnovers = events[events["EVENTMSGTYPE"] == 5]
    if not turnovers.empty:
        last = turnovers.iloc[-1]
        labels["turnover_type"] = int(last["EVENTMSGACTIONTYPE"])
        labels["turnover_player_id"] = int(last["PLAYER1_ID"]) if pd.notna(last["PLAYER1_ID"]) else pd.NA
        labels["steal_player_id"] = int(last["PLAYER2_ID"]) if pd.notna(last["PLAYER2_ID"]) else pd.NA

    blocks = events[(events["EVENTMSGTYPE"] == 2)]
    if not blocks.empty:
        block_mask = False
        for col in DESC_COLS:
            block_mask = block_mask | blocks[col].fillna("").astype(str).str.contains("BLOCK", case=False)
        blocks = blocks[block_mask]
    if not blocks.empty:
        first = blocks.iloc[0]
        labels["block_player_id"] = int(first["PLAYER3_ID"]) if pd.notna(first["PLAYER3_ID"]) else pd.NA

    fouls = events[events["EVENTMSGTYPE"] == 6]
    if not fouls.empty:
        first = fouls.iloc[0]
        labels["foul_type"] = int(first["EVENTMSGACTIONTYPE"])
        labels["foul_player_id"] = int(first["PLAYER1_ID"]) if pd.notna(first["PLAYER1_ID"]) else pd.NA

    ft = events[events["EVENTMSGTYPE"] == 3]
    if not ft.empty:
        labels["free_throw_attempts"] = int(len(ft))
        made = 0
        for _, row in ft.iterrows():
            desc = row["HOMEDESCRIPTION"] or row["VISITORDESCRIPTION"] or row["NEUTRALDESCRIPTION"]
            if desc and "MISS" not in desc.upper():
                made += 1
        labels["free_throw_made"] = made

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbp-dir",
        type=Path,
        default=Path("processed/pbp_with_frames"),
        help="Directory with pbp parquet files",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=Path("processed/play_segments_shots"),
        help="Directory with play segments (will be overwritten)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional separate output directory (defaults to overwrite segments)",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional number of games",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else args.segments_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    segment_files = sorted(args.segments_dir.glob("002*.parquet"))
    if args.limit_games is not None:
        segment_files = segment_files[: args.limit_games]

    pbp_cache = {}

    for seg_path in segment_files:
        game_id = seg_path.stem
        pbp_path = args.pbp_dir / f"{game_id}.parquet"
        if not pbp_path.exists():
            print(f"[WARN] Missing PBP for {game_id}")
            continue
        if game_id not in pbp_cache:
            pbp_cache[game_id] = pd.read_parquet(pbp_path, columns=PBP_COLS)
        pbp = pbp_cache[game_id]
        plays = pd.read_parquet(seg_path)
        if plays.empty:
            continue
        for col, default in NEW_COLUMNS.items():
            if col not in plays.columns:
                plays[col] = default

        for idx, play in plays[PLAY_COLS].iterrows():
            mask = (pbp["EVENTNUM"] >= play["start_event"]) & (pbp["EVENTNUM"] <= play["end_event"])
            events = pbp.loc[mask]
            if events.empty:
                continue
            labels = annotate_play(events)
            for col, value in labels.items():
                plays.at[idx, col] = value

        out_path = (out_dir / seg_path.name) if out_dir != args.segments_dir else seg_path
        plays.to_parquet(out_path, index=False)
        print(f"Annotated actions for {game_id}")


if __name__ == "__main__":
    main()
