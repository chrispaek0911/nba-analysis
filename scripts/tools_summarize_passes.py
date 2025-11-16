#!/usr/bin/env python3
"""Compare detected passes against PBP assists/turnovers per game."""

import argparse
from pathlib import Path

import pandas as pd

PASS_COL = "pass_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbp-dir",
        type=Path,
        default=Path("processed/pbp_with_frames"),
        help="Directory with per-game PBP parquet files",
    )
    parser.add_argument(
        "--passes-dir",
        type=Path,
        default=Path("processed/play_passes"),
        help="Directory with detected pass parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/pass_summary.csv"),
        help="CSV path for summary",
    )
    return parser.parse_args()


def summarize_game(pbp_path: Path, passes_path: Path) -> dict:
    game_id = pbp_path.stem
    pbp = pd.read_parquet(pbp_path)
    passes_df = pd.read_parquet(passes_path)

    passes = len(passes_df)
    made_shots = (pbp["EVENTMSGTYPE"] == 1).sum()
    assists = ((pbp["EVENTMSGTYPE"] == 1) & pbp["PLAYER2_ID"].notna()).sum()
    turnovers = (pbp["EVENTMSGTYPE"] == 5).sum()

    return {
        "game_id": game_id,
        "passes": passes,
        "made_shots": int(made_shots),
        "assists": int(assists),
        "turnovers": int(turnovers),
        "passes_per_made_shot": passes / made_shots if made_shots else None,
        "passes_per_turnover": passes / turnovers if turnovers else None,
    }


def main() -> None:
    args = parse_args()
    pbp_files = sorted(args.pbp_dir.glob("002*.parquet"))
    rows = []
    for pbp_path in pbp_files:
        game_id = pbp_path.stem
        passes_path = args.passes_dir / f"{game_id}.parquet"
        if not passes_path.exists():
            print(f"[WARN] Missing passes for {game_id}")
            continue
        rows.append(summarize_game(pbp_path, passes_path))
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote summary for {len(df)} games to {args.output}")


if __name__ == "__main__":
    main()
