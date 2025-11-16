"""Feature builders for spatial/temporal basketball graph data."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
HOOP_LEFT = np.array([5.25, 25.0], dtype=np.float32)
HOOP_RIGHT = np.array([COURT_LENGTH_FT - 5.25, 25.0], dtype=np.float32)
PAINT_RADIUS_FT = 8.0
FRAME_HZ = 25.0
FRAME_DT = 1.0 / FRAME_HZ


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    norm = positions.copy().astype(np.float32)
    norm[..., 0] = (norm[..., 0] / COURT_LENGTH_FT) * 2 - 1  # x in [-1, 1]
    norm[..., 1] = (norm[..., 1] / COURT_WIDTH_FT) * 2 - 1
    norm[..., 2] = norm[..., 2] / 10.0  # rim height baseline
    return norm


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
    return hull


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _mean_hull_area(team_positions: np.ndarray) -> float:
    if team_positions.size == 0:
        return 0.0
    areas = []
    for frame in team_positions:
        if frame.size == 0:
            continue
        hull = _convex_hull(frame)
        if len(hull) >= 3:
            areas.append(_polygon_area(hull))
    return float(np.mean(areas)) if areas else 0.0


def _pairwise_stats(points: np.ndarray) -> Tuple[float, float]:
    if points.shape[0] < 2:
        return 0.0, 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(points.shape[0], 1)
    values = dist[iu]
    return float(np.mean(values)), float(np.min(values))


def _mean_spacing(team_positions: np.ndarray) -> Tuple[float, float]:
    if team_positions.size == 0:
        return 0.0, 0.0
    means, mins = [], []
    for frame in team_positions:
        if frame.size == 0:
            continue
        mean_d, min_d = _pairwise_stats(frame)
        means.append(mean_d)
        mins.append(min_d)
    return float(np.mean(means)) if means else 0.0, float(np.mean(mins)) if mins else 0.0


def _mean_nearest_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return 0.0
    dists = []
    for frame_a, frame_b in zip(points_a, points_b):
        if frame_a.size == 0 or frame_b.size == 0:
            continue
        diff = frame_a[:, None, :] - frame_b[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dists.append(np.mean(np.min(dist, axis=1)))
    return float(np.mean(dists)) if dists else 0.0


def _choose_hoop(offense_positions: np.ndarray) -> np.ndarray:
    if offense_positions.size == 0:
        return HOOP_LEFT
    centroid = np.nanmean(offense_positions[0], axis=0)
    if np.isnan(centroid).any():
        centroid = np.array([COURT_LENGTH_FT / 2, COURT_WIDTH_FT / 2])
    return HOOP_RIGHT if centroid[0] > COURT_LENGTH_FT / 2 else HOOP_LEFT


def _paint_occupancy(team_positions: np.ndarray, hoop: np.ndarray) -> float:
    if team_positions.size == 0:
        return 0.0
    counts = []
    for frame in team_positions:
        if frame.size == 0:
            continue
        dists = np.linalg.norm(frame - hoop, axis=-1)
        counts.append(np.mean(dists <= PAINT_RADIUS_FT))
    return float(np.mean(counts)) if counts else 0.0


def _compute_speed(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.0
    vel = np.diff(positions, axis=0) / FRAME_DT
    speed = np.linalg.norm(vel[..., :2], axis=-1)
    return float(np.mean(speed))


def _compute_accel(positions: np.ndarray) -> float:
    if positions.shape[0] < 3:
        return 0.0
    vel = np.diff(positions, axis=0) / FRAME_DT
    acc = np.diff(vel, axis=0) / FRAME_DT
    accel = np.linalg.norm(acc[..., :2], axis=-1)
    return float(np.mean(accel))


def build_spatial_features(
    positions: np.ndarray,
    node_team_ids: np.ndarray,
    context: Dict,
) -> Dict[str, float]:
    offense_team = int(context.get("offense_team_id")) if context.get("offense_team_id") else None
    defense_team = int(context.get("defense_team_id")) if context.get("defense_team_id") else None
    team_ids = node_team_ids.astype(int)
    ball_indices = np.where(team_ids == -1)[0]
    ball_idx = int(ball_indices[0]) if ball_indices.size else 0
    player_mask = team_ids != -1

    def _select(team_id: Optional[int]) -> np.ndarray:
        if team_id is None:
            return np.empty((0, 0, 2))
        indices = np.where((team_ids == team_id) & player_mask)[0]
        return positions[:, indices, :2] if indices.size else np.empty((0, 0, 2))

    offense_positions = _select(offense_team)
    defense_positions = _select(defense_team)
    hoop = _choose_hoop(offense_positions)

    offense_hull = _mean_hull_area(offense_positions)
    defense_hull = _mean_hull_area(defense_positions)
    offense_spacing_mean, offense_spacing_min = _mean_spacing(offense_positions)
    defense_spacing_mean, defense_spacing_min = _mean_spacing(defense_positions)
    offense_paint = _paint_occupancy(offense_positions, hoop)
    defense_paint = _paint_occupancy(defense_positions, hoop)
    offense_vs_def = _mean_nearest_distance(offense_positions, defense_positions)
    defense_vs_off = _mean_nearest_distance(defense_positions, offense_positions)

    ball_positions = positions[:, ball_idx, :]
    ball_speed = _compute_speed(ball_positions)
    ball_accel = _compute_accel(ball_positions)

    player_positions = positions[:, player_mask, :]
    player_speed = _compute_speed(player_positions) if player_positions.size else 0.0

    return {
        "offense_hull_area_mean": offense_hull,
        "defense_hull_area_mean": defense_hull,
        "offense_spacing_mean": offense_spacing_mean,
        "offense_spacing_min": offense_spacing_min,
        "defense_spacing_mean": defense_spacing_mean,
        "defense_spacing_min": defense_spacing_min,
        "offense_paint_occupancy_mean": offense_paint,
        "defense_paint_occupancy_mean": defense_paint,
        "offense_nearest_defender_mean": offense_vs_def,
        "defense_nearest_offender_mean": defense_vs_off,
        "ball_speed_mean": ball_speed,
        "ball_acceleration_mean": ball_accel,
        "player_speed_mean": player_speed,
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_behavior_features(context: Dict, spatial_features: Dict[str, float]) -> Dict[str, float]:
    relative_margin = float(context.get("relative_score_margin", 0))
    shot_clock = float(context.get("shot_clock_start", 24.0) or 24.0)
    seconds_period = float(context.get("seconds_remaining_period", 720.0) or 720.0)
    is_trailing = float(context.get("is_offense_trailing", 0))
    offense_fouls = float(context.get("offense_fouls_in_period", 0))
    defense_fouls = float(context.get("defense_fouls_in_period", 0))

    shot_clock_term = 1 - min(1.0, shot_clock / 24.0)
    clock_term = 1 - min(1.0, seconds_period / 720.0)
    margin_term = -relative_margin / 10.0
    aggression_score = _sigmoid(1.2 * margin_term + 0.8 * shot_clock_term + 0.6 * clock_term + 0.5 * is_trailing)

    paint_delta = spatial_features.get("defense_paint_occupancy_mean", 0) - spatial_features.get(
        "offense_paint_occupancy_mean", 0
    )
    foul_pressure = min(1.0, defense_fouls / 5.0)
    defense_score = _sigmoid(0.8 * (relative_margin / 10.0) + 0.6 * paint_delta + 0.5 * (1 - foul_pressure))

    return {
        "aggression_score": aggression_score,
        "defense_intensity_score": defense_score,
        "offense_pressure_index": (shot_clock_term + clock_term + is_trailing + max(0.0, margin_term)) / 4.0,
        "defense_pressure_index": (paint_delta + (1 - foul_pressure) + max(0.0, relative_margin / 10.0)) / 3.0,
        "offense_foul_pressure": offense_fouls / 5.0,
        "defense_foul_pressure": defense_fouls / 5.0,
    }


def build_features(
    positions: np.ndarray,
    node_team_ids: np.ndarray,
    context: Dict,
) -> Dict[str, Dict[str, float]]:
    """Return spatial features and auxiliary behavior targets."""

    spatial = build_spatial_features(positions, node_team_ids, context)
    behavior = build_behavior_features(context, spatial)
    return {"features": spatial, "aux_targets": behavior}
