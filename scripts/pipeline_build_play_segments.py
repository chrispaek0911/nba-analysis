#!/usr/bin/env python3
"""Create play-level segments from PBP-with-frame data."""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

PBP_COLUMNS = [
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "PLAYER1_TEAM_ID",
    "PLAYER2_TEAM_ID",
    "PLAYER3_TEAM_ID",
    "frame_start",
    "frame_end",
]

REBOUND_RE = re.compile(r"Off:(\d+) Def:(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbp-dir",
        type=Path,
        default=Path("processed/pbp_with_frames"),
        help="Directory containing per-game parquet files with frame indices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/play_segments"),
        help="Directory to save play metadata parquet files",
    )
    return parser.parse_args()


def infer_team_id(row: pd.Series) -> Optional[int]:
    for col in ("PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"):
        if col in row and pd.notna(row[col]):
            try:
                return int(row[col])
            except Exception:
                continue
    return None


def opponent_team(team_catalog: List[int], team_id: Optional[int]) -> Optional[int]:
    if team_id is None:
        return None
    others = [t for t in team_catalog if t != team_id]
    if len(others) == 1:
        return others[0]
    if len(others) >= 2:
        return others[0]
    return None


def parse_rebound(desc: str) -> Optional[Tuple[int, int]]:
    if not desc:
        return None
    m = REBOUND_RE.search(desc)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def possession_change(row: pd.Series, team_id: Optional[int], team_catalog: List[int]) -> Tuple[bool, Optional[int], str]:
    msg_type = int(row["EVENTMSGTYPE"])
    description = row.get("HOMEDESCRIPTION") or row.get("VISITORDESCRIPTION") or row.get("NEUTRALDESCRIPTION")
    if msg_type == 1:  # made shot
        return True, opponent_team(team_catalog, team_id), "made_shot"
    if msg_type == 5:  # turnover
        return True, opponent_team(team_catalog, team_id), "turnover"
    if msg_type == 4:  # rebound
        numbers = parse_rebound(description or "")
        if numbers and numbers[1] > 0:  # defensive rebound
            return True, team_id, "def_rebound"
        return False, None, "off_rebound"
    if msg_type == 10:  # jump ball
        return True, team_id, "jump_ball"
    if msg_type == 12:  # start period
        return True, team_id, "period_start"
    return False, None, "continue"


def finalize_play(play: dict, result: str) -> dict:
    play = play.copy()
    play["result_type"] = result
    play.setdefault("end_event", play["start_event"])
    play.setdefault("end_index", play["start_index"])
    play.setdefault("end_period", play["start_period"])
    play.setdefault("end_game_clock", play["start_game_clock"])
    play.setdefault("end_frame", play.get("start_frame"))
    play["events_in_play"] = len(play.get("event_nums", []))
    play.pop("event_nums", None)
    return play


def segment_game(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=PBP_COLUMNS)
    df = df.sort_values("EVENTNUM").reset_index(drop=True)

    plays: List[dict] = []
    team_catalog: List[int] = []
    current_play = None
    current_team: Optional[int] = None
    play_id = 1

    for idx, row in df.iterrows():
        team_id = infer_team_id(row)
        if team_id is not None and team_id not in team_catalog:
            team_catalog.append(team_id)
        if current_play is None:
            if current_team is None and team_id is not None:
                current_team = team_id
            if current_team is None:
                continue
            current_play = {
                "play_id": play_id,
                "game_id": row["GAME_ID"],
                "start_index": idx,
                "start_event": int(row["EVENTNUM"]),
                "start_period": int(row["PERIOD"]),
                "start_game_clock": row["PCTIMESTRING"],
                "offense_team_id": current_team,
                "start_frame": int(row["frame_start"]) if pd.notna(row["frame_start"]) else None,
                "end_frame": int(row["frame_end"]) if pd.notna(row["frame_end"]) else None,
                "end_event": int(row["EVENTNUM"]),
                "end_index": idx,
                "end_period": int(row["PERIOD"]),
                "end_game_clock": row["PCTIMESTRING"],
                "event_nums": [int(row["EVENTNUM"])],
            }
        else:
            current_play["event_nums"].append(int(row["EVENTNUM"]))
            current_play["end_event"] = int(row["EVENTNUM"])
            current_play["end_index"] = idx
            current_play["end_period"] = int(row["PERIOD"])
            current_play["end_game_clock"] = row["PCTIMESTRING"]
            if current_play.get("start_frame") is None and pd.notna(row["frame_start"]):
                current_play["start_frame"] = int(row["frame_start"])
        # update frame end if available
        if pd.notna(row["frame_end"]):
            current_play["end_frame"] = int(row["frame_end"])

        change, next_team, result = possession_change(row, team_id, team_catalog)
        if change and current_play is not None:
            plays.append(finalize_play(current_play, result))
            play_id += 1
            current_play = None
            if next_team is not None:
                current_team = next_team
            # otherwise wait for next event with team info

    if current_play is not None:
        plays.append(finalize_play(current_play, "unfinished"))

    if not plays:
        return pd.DataFrame()

    return pd.DataFrame(plays)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pbp_files = sorted(args.pbp_dir.glob("002*.parquet"))
    if not pbp_files:
        raise FileNotFoundError(f"No parquet files in {args.pbp_dir}")

    for path in pbp_files:
        df = segment_game(path)
        if df.empty:
            print(f"[WARN] No plays generated for {path.name}")
            continue
        out_path = args.output_dir / f"{path.stem}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} plays for {path.stem}")


if __name__ == "__main__":
    main()
