"""Stage-based global feature selector.

Parquet rows from play_feature_embeddings_new have columns like:
  feat_<name>, label_<name>, aux_<name>

This helper picks only the allowed feat_* columns per stage to avoid leakage.

Stages:
  1: 최소 컨텍스트(시간/스코어/보너스)
  2: 1단계 + 슛/파울/턴오버 타입 + 공간/속도 통계
  3: 2단계 + 팀 ID/이벤트 길이 + 패스 물리량(순수 통계)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import math


STAGE1_KEYS = [
    "period",
    "seconds_remaining_period",
    "shot_clock_start",
    "score_home",
    "score_away",
    "relative_score_margin",
    "is_offense_trailing",
    "offense_fouls_in_period",
    "defense_fouls_in_period",
    "offense_in_bonus",
    "defense_in_bonus",
]

STAGE2_EXTRA = [
    # shot atoms
    "shot_action_type_id",
    "shot_event_type_id",
    "shot_type_id",
    "shot_zone_basic_id",
    "shot_zone_area_id",
    "shot_zone_range_id",
    "shot_distance_ft",
    "shot_distance_norm",
    "shot_loc_x_norm",
    "shot_loc_y_norm",
    # foul/turnover types
    "foul_type_id",
    "foul_is_shooting",
    "foul_is_block",
    "foul_is_offensive",
    "turnover_type_id",
    "rebound_value",
    # spatial/kinematic stats (pure)
    "offense_hull_area_mean",
    "defense_hull_area_mean",
    "offense_spacing_mean",
    "offense_spacing_min",
    "defense_spacing_mean",
    "defense_spacing_min",
    "offense_paint_occupancy_mean",
    "defense_paint_occupancy_mean",
    "offense_nearest_defender_mean",
    "defense_nearest_offender_mean",
    "ball_speed_mean",
    "ball_acceleration_mean",
    "player_speed_mean",
]

STAGE3_EXTRA = [
    # ids / context length
    "offense_team_id",
    "defense_team_id",
    "events_in_play",
    # pass physics (pure stats)
    "pass_count",
    "pass_mean_air_frames",
    "pass_max_air_frames",
    "pass_air_frames_std",
    "pass_mean_ball_max_z",
    "pass_max_ball_max_z",
    "pass_ball_max_z_std",
    "pass_high_arc_ratio",
    "pass_ground_ratio",
    "pass_transition_flag",
    "pass_paint_target_flag",
    "pass_gap_short_flag",
    "pass_risk_unreliable_flag",
]


def _iter_keys_for_stage(stage: int) -> Iterable[str]:
    if stage <= 1:
        return STAGE1_KEYS
    if stage == 2:
        return list(STAGE1_KEYS) + STAGE2_EXTRA
    # stage >= 3
    return list(STAGE1_KEYS) + STAGE2_EXTRA + STAGE3_EXTRA


def _get_feat(row: Any, key: str):
    """Return value for key or feat_key; None if missing or NaN."""
    for candidate in (key, f"feat_{key}"):
        if isinstance(row, dict):
            val = row.get(candidate)
        else:
            # pandas Series / obj with getattr
            val = row[candidate] if candidate in row else getattr(row, candidate, None)
        if val is None:
            continue
        try:
            if isinstance(val, float) and math.isnan(val):
                continue
        except Exception:
            pass
        return val
    return None


def build_global_features(row: Any, stage: int = 1) -> Dict[str, float]:
    """Pick allowed feat_* columns for the given stage (leakage-safe)."""
    feats: Dict[str, float] = {}
    for key in _iter_keys_for_stage(stage):
        val = _get_feat(row, key)
        if val is None:
            continue
        try:
            feats[key] = float(val)
        except Exception:
            # skip non-convertible
            continue
    return feats

