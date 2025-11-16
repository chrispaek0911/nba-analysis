"""Context utilities for per-play score/time/foul state tracking."""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def clock_to_seconds(clock_str: Optional[str]) -> float:
    if not clock_str or isinstance(clock_str, float):
        return float(clock_str or 0.0)
    try:
        minutes, seconds = clock_str.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return 0.0


@dataclass
class ScoreContextBuilder:
    pbp_df: Optional[pd.DataFrame]

    def __post_init__(self) -> None:
        self.available = self.pbp_df is not None and not self.pbp_df.empty
        self.home_team_id: Optional[int] = None
        self.away_team_id: Optional[int] = None
        self.event_numbers = []
        self.score_records: Dict[int, Dict[str, float]] = {}
        if self.available:
            self._infer_team_ids()
            self._build_score_records()

    def _infer_team_ids(self) -> None:
        assert self.pbp_df is not None
        df = self.pbp_df
        home_row = df[(df["HOMEDESCRIPTION"].notna()) & (df["PLAYER1_TEAM_ID"].notna())]
        away_row = df[(df["VISITORDESCRIPTION"].notna()) & (df["PLAYER1_TEAM_ID"].notna())]
        if not home_row.empty:
            self.home_team_id = int(home_row["PLAYER1_TEAM_ID"].iloc[0])
        if not away_row.empty:
            self.away_team_id = int(away_row["PLAYER1_TEAM_ID"].iloc[0])
        if self.home_team_id is None or self.away_team_id is None:
            teams = df["PLAYER1_TEAM_ID"].dropna().unique()
            if len(teams) >= 2:
                self.home_team_id = int(teams[0])
                self.away_team_id = int(teams[1])

    def _build_score_records(self) -> None:
        assert self.pbp_df is not None
        df = self.pbp_df.sort_values("EVENTNUM")
        score_home = score_away = 0
        margin = 0
        for row in df.itertuples():
            if isinstance(row.SCORE, str) and " - " in row.SCORE:
                away_str, home_str = row.SCORE.split(" - ")
                try:
                    score_away = int(away_str)
                    score_home = int(home_str)
                except ValueError:
                    pass
            margin_val = row.SCOREMARGIN
            if isinstance(margin_val, str):
                margin_val = margin_val.strip()
                if margin_val.upper() == "TIE":
                    margin = 0
                elif margin_val:
                    try:
                        margin = int(float(margin_val))
                    except ValueError:
                        pass
            elif margin_val is not None:
                if isinstance(margin_val, float) and np.isnan(margin_val):
                    pass
                else:
                    try:
                        margin = int(margin_val)
                    except (TypeError, ValueError):
                        pass
            self.event_numbers.append(row.EVENTNUM)
            self.score_records[row.EVENTNUM] = {
                "score_home": float(score_home),
                "score_away": float(score_away),
                "score_margin_home": float(margin),
            }
        if not self.event_numbers:
            self.event_numbers = [0]
            self.score_records[0] = {"score_home": 0.0, "score_away": 0.0, "score_margin_home": 0.0}

    def _lookup(self, event_num: int) -> Dict[str, float]:
        if not self.event_numbers:
            return {"score_home": 0.0, "score_away": 0.0, "score_margin_home": 0.0}
        idx = bisect.bisect_right(self.event_numbers, event_num) - 1
        idx = max(idx, 0)
        event_key = self.event_numbers[idx]
        return self.score_records.get(event_key) or {"score_home": 0.0, "score_away": 0.0, "score_margin_home": 0.0}

    def score_for(self, event_num: int) -> Dict[str, float]:
        return self._lookup(event_num)

    def relative_margin(self, offense_team_id: Optional[int], event_num: int) -> float:
        record = self._lookup(event_num)
        margin = record.get("score_margin_home", 0.0)
        if offense_team_id is None or self.home_team_id is None:
            return 0.0
        return margin if offense_team_id == self.home_team_id else -margin

    def opponent_of(self, team_id: Optional[int]) -> Optional[int]:
        if team_id is None:
            return None
        if self.home_team_id is None or self.away_team_id is None:
            return None
        return self.away_team_id if team_id == self.home_team_id else self.home_team_id


@dataclass
class PlayContextManager:
    labels_df: pd.DataFrame
    score_builder: ScoreContextBuilder

    def __post_init__(self) -> None:
        self.sorted_df = self.labels_df.sort_values("start_index").reset_index(drop=True)
        self.period_fouls = defaultdict(lambda: defaultdict(int))

    def context_for(self, row: pd.Series, offense_team_id: Optional[int]) -> Dict:
        period = int(row.get("start_period") or row.get("end_period") or 1)
        seconds_period = clock_to_seconds(row.get("start_game_clock")) or 720.0
        event_num = int(row.get("start_event") or 0)
        score_record = self.score_builder.score_for(event_num)
        relative_margin = self.score_builder.relative_margin(offense_team_id, event_num)
        defense_team_id = self.score_builder.opponent_of(offense_team_id)
        offense_fouls = self.period_fouls[period].get(offense_team_id, 0)
        defense_fouls = self.period_fouls[period].get(defense_team_id, 0)
        return {
            "period": period,
            "start_event": event_num,
            "start_index": int(row.get("start_index") or 0),
            "offense_team_id": offense_team_id,
            "defense_team_id": defense_team_id,
            "seconds_remaining_period": seconds_period,
            "score_home": score_record.get("score_home", 0.0),
            "score_away": score_record.get("score_away", 0.0),
            "score_margin_home": score_record.get("score_margin_home", 0.0),
            "relative_score_margin": relative_margin,
            "is_offense_trailing": 1 if relative_margin < 0 else 0,
            "offense_fouls_in_period": offense_fouls,
            "defense_fouls_in_period": defense_fouls,
            "offense_in_bonus": 1 if offense_fouls >= 5 else 0,
            "defense_in_bonus": 1 if defense_fouls >= 5 else 0,
        }

    def register_foul(self, period: int, team_id: Optional[int]) -> None:
        if team_id is None:
            return
        self.period_fouls[period][team_id] += 1
