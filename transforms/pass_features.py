
"""Pass embedding utilities for per-play graph features."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

FRAME_RATE = 25.0  # SportVU Hz
SHOT_WINDOW_FRAMES = 100  # allow ~4 seconds between pass and shot
PASS_AIR_MAX = 25.0
SHORT_GAP_SECONDS = 1.5
PAINT_DISTANCE_FT = 8.0
TRANSITION_DURATION_SEC = 5.0

STATE_FEATURE_KEYS = {
    "pass_count",
    "pass_mean_air_frames",
    "pass_max_air_frames",
    "pass_air_frames_std",
    "pass_mean_ball_max_z",
    "pass_max_ball_max_z",
    "pass_ball_max_z_std",
    "pass_high_arc_ratio",
    "pass_ground_ratio",
    "pass_turnover_airtime_mean",
    "pass_safe_airtime_mean",
    "pass_turnover_height_mean",
    "pass_safe_height_mean",
    "pass_transition_flag",
    "pass_paint_target_flag",
    "pass_gap_short_flag",
    "pass_risk_unreliable_flag",
}

LABEL_KEYS = {
    "pass_has_turnover_flag",
    "pass_turnover_count",
    "pass_turnover_ratio",
    "pass_safe_count",
    "pass_safe_ratio",
    "pass_led_to_shot_flag",
    "pass_led_to_shot_ratio",
    "pass_safe_before_shot_count",
    "pass_safe_after_shot_count",
    "pass_dead_count",
    "pass_dead_ratio",
    "pass_sequence_has_shot_flag",
    "pass_sequence_dead_flag",
    "pass_sequence_unknown_shot_flag",
    "pass_last_turnover_flag",
    "pass_first_turnover_flag",
    "pass_final_to_shot_gap_frames",
    "pass_final_to_shot_gap_seconds",
    "pass_sequence_duration_seconds",
}

AUX_TARGET_KEYS = {"pass_risk_score"}


def _base_features(shot_present: bool) -> Dict[str, float]:
    return {
        "pass_count": 0.0,
        "pass_has_turnover_flag": 0.0,
        "pass_turnover_count": 0.0,
        "pass_turnover_ratio": 0.0,
        "pass_safe_count": 0.0,
        "pass_safe_ratio": 0.0,
        "pass_led_to_shot_flag": 0.0,
        "pass_led_to_shot_ratio": 0.0,
        "pass_safe_before_shot_count": 0.0,
        "pass_safe_after_shot_count": 0.0,
        "pass_dead_count": 0.0,
        "pass_dead_ratio": 0.0,
        "pass_sequence_has_shot_flag": float(shot_present),
        "pass_sequence_dead_flag": float(not shot_present),
        "pass_sequence_unknown_shot_flag": 0.0,
        "pass_last_turnover_flag": 0.0,
        "pass_first_turnover_flag": 0.0,
        "pass_mean_air_frames": 0.0,
        "pass_max_air_frames": 0.0,
        "pass_air_frames_std": 0.0,
        "pass_mean_ball_max_z": 0.0,
        "pass_max_ball_max_z": 0.0,
        "pass_ball_max_z_std": 0.0,
        "pass_high_arc_ratio": 0.0,
        "pass_ground_ratio": 0.0,
        "pass_risk_score": 0.0,
        "pass_turnover_airtime_mean": 0.0,
        "pass_safe_airtime_mean": 0.0,
        "pass_turnover_height_mean": 0.0,
        "pass_safe_height_mean": 0.0,
        "pass_transition_flag": 0.0,
        "pass_paint_target_flag": 0.0,
        "pass_gap_short_flag": 0.0,
        "pass_risk_unreliable_flag": 0.0,
        "pass_final_to_shot_gap_frames": -1.0,
        "pass_final_to_shot_gap_seconds": -1.0,
        "pass_sequence_duration_seconds": 0.0,
    }


def _finalize_outputs(values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    features = {k: values[k] for k in STATE_FEATURE_KEYS if k in values}
    labels = {k: values[k] for k in LABEL_KEYS if k in values}
    aux = {k: values[k] for k in AUX_TARGET_KEYS if k in values}
    return {"features": features, "labels": labels, "aux_targets": aux}


def build_pass_features(
    pass_rows: Optional[pd.DataFrame],
    shot_frame: Optional[int] = None,
    shot_present: bool = False,
    shot_distance: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-play pass descriptors for downstream models."""

    features = _base_features(shot_present)
    if pass_rows is None or pass_rows.empty:
        if shot_present and shot_frame is None:
            features["pass_sequence_unknown_shot_flag"] = 1.0
            features["pass_risk_unreliable_flag"] = 1.0
        return _finalize_outputs(features)

    df = pass_rows.sort_values("end_frame").reset_index(drop=True)
    total = len(df)
    features["pass_count"] = float(total)

    air = df["air_frames"].to_numpy(dtype=float)
    height = df["ball_max_z"].to_numpy(dtype=float)
    turnover_mask = df["is_turnover_pass"].fillna(False).to_numpy(dtype=bool)
    safe_mask = ~turnover_mask

    turnover_count = int(turnover_mask.sum())
    safe_count = int(safe_mask.sum())
    features["pass_turnover_count"] = float(turnover_count)
    features["pass_safe_count"] = float(safe_count)
    features["pass_has_turnover_flag"] = float(turnover_count > 0)
    features["pass_turnover_ratio"] = turnover_count / total
    features["pass_safe_ratio"] = safe_count / total
    features["pass_last_turnover_flag"] = float(bool(total) and turnover_mask[-1])
    features["pass_first_turnover_flag"] = float(bool(total) and turnover_mask[0])

    air_clipped = np.clip(np.nan_to_num(air, nan=0.0), 0.0, PASS_AIR_MAX)
    features["pass_mean_air_frames"] = float(np.mean(air_clipped))
    features["pass_max_air_frames"] = float(np.max(air_clipped))
    features["pass_air_frames_std"] = float(np.std(air_clipped))
    height_clean = np.nan_to_num(height, nan=0.0)
    features["pass_mean_ball_max_z"] = float(np.mean(height_clean))
    features["pass_max_ball_max_z"] = float(np.max(height_clean))
    features["pass_ball_max_z_std"] = float(np.std(height_clean))
    features["pass_high_arc_ratio"] = float(np.mean(height_clean >= 10.0))
    features["pass_ground_ratio"] = float(np.mean(height_clean <= 4.0))
    features["pass_turnover_airtime_mean"] = float(np.mean(air_clipped[turnover_mask])) if turnover_count else 0.0
    features["pass_safe_airtime_mean"] = float(np.mean(air_clipped[safe_mask])) if safe_count else 0.0
    features["pass_turnover_height_mean"] = float(np.mean(height_clean[turnover_mask])) if turnover_count else 0.0
    features["pass_safe_height_mean"] = float(np.mean(height_clean[safe_mask])) if safe_count else 0.0

    air_norm = min(1.0, features["pass_mean_air_frames"] / PASS_AIR_MAX)
    alpha, beta = 0.7, 0.3
    features["pass_risk_score"] = alpha * features["pass_turnover_ratio"] + beta * air_norm

    start_frame = float(df["start_frame"].min()) if "start_frame" in df else float(df["start_event"].min() or 0)
    end_frame = float(df["end_frame"].max())
    features["pass_sequence_duration_seconds"] = max(0.0, (end_frame - start_frame) / FRAME_RATE)
    features["pass_transition_flag"] = float(features["pass_sequence_duration_seconds"] <= TRANSITION_DURATION_SEC)

    led_to_shot = 0
    safe_before = 0
    safe_after = float(safe_count)
    if safe_count and shot_present:
        if shot_frame is None:
            features["pass_sequence_unknown_shot_flag"] = 1.0
        else:
            safe_df = df[safe_mask]
            before_shot = safe_df[safe_df["end_frame"] <= shot_frame]
            safe_before = float(len(before_shot))
            safe_after = float(safe_count) - safe_before
            if not before_shot.empty:
                led_to_shot = 1
                final_pass = before_shot.iloc[-1]
                gap_frames = int(shot_frame - int(final_pass["end_frame"]))
                if gap_frames <= SHOT_WINDOW_FRAMES:
                    features["pass_final_to_shot_gap_frames"] = float(gap_frames)
                    features["pass_final_to_shot_gap_seconds"] = gap_frames / FRAME_RATE
                else:
                    led_to_shot = 0
    features["pass_led_to_shot_flag"] = float(led_to_shot)
    features["pass_led_to_shot_ratio"] = led_to_shot / total
    features["pass_safe_before_shot_count"] = safe_before
    features["pass_safe_after_shot_count"] = max(0.0, safe_after)

    dead_count = float(max(0, safe_count - led_to_shot)) if safe_count else 0.0
    features["pass_dead_count"] = dead_count
    features["pass_dead_ratio"] = dead_count / total if total else 0.0

    gap_seconds = features["pass_final_to_shot_gap_seconds"]
    if gap_seconds >= 0 and gap_seconds <= SHORT_GAP_SECONDS:
        features["pass_gap_short_flag"] = 1.0

    if shot_present and shot_distance is not None and shot_distance <= PAINT_DISTANCE_FT:
        features["pass_paint_target_flag"] = 1.0

    risk_unreliable = total < 2 or (shot_present and shot_frame is None)
    if np.isnan(air).any() or np.isnan(height).any():
        risk_unreliable = True
    features["pass_risk_unreliable_flag"] = float(risk_unreliable)

    return _finalize_outputs(features)
