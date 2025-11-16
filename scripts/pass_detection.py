#!/usr/bin/env python3
"""Shared pass detection utilities used by visualization and pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class HandlerRow:
    array_idx: int
    frame_idx: int
    player_id: int
    team_id: int
    distance: float
    ball_z: float
    ball_dz: float
    ball_speed: float
    has_control: bool
    ball_x: float
    ball_y: float
    event_id: Optional[int] = None
    timestamp_ms: Optional[int] = None
    game_clock: Optional[float] = None
    shot_clock: Optional[float] = None


@dataclass
class PassRecord:
    start_idx: int
    end_idx: int
    passer_id: int
    receiver_id: int
    passer_team: int
    receiver_team: int
    air_frames: int
    ball_max_z: float
    is_turnover: bool


@dataclass
class PlayContext:
    start_event_msg_type: Optional[int] = None
    jump_ball_guard_frames: int = 0


def infer_metadata_shot_window(rows: Sequence[HandlerRow], meta: Dict, args) -> Tuple[Optional[int], Optional[int]]:
    shooter = meta.get("shot_player_id")
    if shooter is None:
        return None, None
    try:
        shooter = int(shooter)
    except (TypeError, ValueError):
        return None, None
    min_height = args.shot_ball_height_threshold
    lookahead_limit = max(args.shot_guard_frames * 2, 20)
    candidate_idx: Optional[int] = None
    remaining = 0
    for idx, row in enumerate(rows):
        if row.player_id == shooter and row.has_control:
            candidate_idx = idx
            remaining = lookahead_limit
            continue
        if candidate_idx is None:
            continue
        if row.ball_z >= min_height:
            start_idx = rows[candidate_idx].array_idx
            end_pos = min(len(rows) - 1, idx + args.shot_guard_frames)
            end_idx = rows[end_pos].array_idx
            return start_idx, end_idx
        remaining -= 1
        if remaining <= 0:
            candidate_idx = None
    return None, None


def detect_passes(
    rows: List[HandlerRow],
    meta: Dict,
    args,
    play_context: Optional[PlayContext] = None,
) -> List[PassRecord]:
    if not rows:
        return []

    def to_int(value: Optional[object]) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    shot_frame = to_int(meta.get("shot_frame"))
    if shot_frame is None and meta.get("shot_player_id") is not None:
        shot_frame = to_int(meta.get("end_frame"))
    meta_shot_start, meta_shot_end = infer_metadata_shot_window(rows, meta, args)
    result_flag = (meta.get("result_type") or "").lower()
    made_shot_boost = getattr(args, "made_shot_cooldown_boost", 0) if result_flag == "made_shot" else 0
    court_length = getattr(args, "court_length", 94.0)
    endline_threshold = getattr(args, "made_shot_endline_threshold", 4.0)
    made_shot_floor_z = getattr(args, "made_shot_floor_z", 4.0)

    def confirm(idx: int, player_id: int) -> bool:
        needed = args.deflection_confirm_frames
        if needed <= 0:
            return True
        success = 0
        for row in rows[idx + 1 : idx + 1 + needed]:
            if row.player_id == player_id and row.has_control:
                success += 1
            else:
                break
        return success >= needed

    def defense_blocked(row: HandlerRow, prev_team: Optional[int]) -> bool:
        if prev_team is None or row.team_id == prev_team:
            return False
        if row.ball_z > args.defense_max_control_z:
            return True
        if abs(row.ball_dz) > args.defense_z_velocity_threshold:
            return True
        if shot_frame is not None and args.shot_guard_frames > 0:
            if abs(row.frame_idx - shot_frame) <= args.shot_guard_frames:
                return True
        return False

    jump_ball_cooldown = 0
    if play_context and play_context.start_event_msg_type == 10:
        jump_ball_cooldown = play_context.jump_ball_guard_frames or args.jump_ball_guard_frames

    passes: List[PassRecord] = []
    prev_row: Optional[HandlerRow] = None
    prev_controller: Optional[int] = None
    prev_team: Optional[int] = None
    air_rows: List[HandlerRow] = []
    candidate_controller: Optional[int] = None
    candidate_frames = 0
    start_frame = rows[0].frame_idx
    shot_cooldown = jump_ball_cooldown
    meta_shot_triggered = False
    waiting_for_endline = False

    for idx, row in enumerate(rows):
        in_shot_arc = row.ball_z >= args.shot_z_threshold and row.ball_speed >= args.shot_speed_threshold
        if row.frame_idx - start_frame < args.initial_skip_frames:
            if row.has_control:
                prev_controller = row.player_id
                prev_team = row.team_id
                prev_row = row
            else:
                prev_controller = None
                prev_team = None
                prev_row = None
            air_rows.clear()
            candidate_controller = None
            candidate_frames = 0
            shot_cooldown = max(shot_cooldown, jump_ball_cooldown)
            continue
        if (
            not meta_shot_triggered
            and meta_shot_start is not None
            and row.array_idx >= meta_shot_start
        ):
            window_end_idx = meta_shot_end
            if window_end_idx is None:
                pos = min(len(rows) - 1, idx + args.shot_guard_frames)
                window_end_idx = rows[pos].array_idx
            remaining = max(1, window_end_idx - row.array_idx + 1)
            if made_shot_boost:
                remaining += made_shot_boost
            shot_cooldown = max(shot_cooldown, remaining)
            meta_shot_triggered = True
            if result_flag == "made_shot":
                waiting_for_endline = True
            prev_controller = None
            prev_team = None
            prev_row = None
            air_rows.clear()
            candidate_controller = None
            candidate_frames = 0
            continue

        if waiting_for_endline:
            near_endline = (
                row.ball_z <= made_shot_floor_z
                and (
                    row.ball_x <= endline_threshold
                    or row.ball_x >= court_length - endline_threshold
                )
            )
            if near_endline:
                waiting_for_endline = False
            else:
                continue

        if in_shot_arc:
            prev_controller = None
            prev_team = None
            prev_row = None
            air_rows.clear()
            candidate_controller = None
            candidate_frames = 0
            shot_cooldown = max(shot_cooldown, args.shot_guard_frames)
            continue
        if shot_cooldown > 0:
            if row.has_control:
                prev_controller = row.player_id
                prev_team = row.team_id
                prev_row = row
                air_rows.clear()
                candidate_controller = None
                candidate_frames = 0
                shot_cooldown = 0
            else:
                shot_cooldown -= 1
                continue
        if not row.has_control:
            if prev_controller is not None:
                air_rows.append(row)
            candidate_controller = None
            candidate_frames = 0
            continue
        if prev_controller is None:
            prev_controller = row.player_id
            prev_team = row.team_id
            prev_row = row
            air_rows.clear()
            continue
        if row.player_id == prev_controller:
            prev_row = row
            air_rows.clear()
            candidate_controller = None
            candidate_frames = 0
            continue
        if defense_blocked(row, prev_team):
            candidate_controller = None
            candidate_frames = 0
            if prev_controller is not None:
                air_rows.append(row)
            continue
        required = args.offense_control_frames if row.team_id == prev_team else args.defense_control_frames
        if candidate_controller != row.player_id:
            candidate_controller = row.player_id
            candidate_frames = 1
        else:
            candidate_frames += 1
        if candidate_frames >= required and prev_row is not None and confirm(idx, row.player_id):
            if len(air_rows) >= args.min_air_frames:
                passes.append(
                    PassRecord(
                        start_idx=prev_row.array_idx,
                        end_idx=row.array_idx,
                        passer_id=prev_controller,
                        receiver_id=row.player_id,
                        passer_team=prev_team,
                        receiver_team=row.team_id,
                        air_frames=len(air_rows),
                        ball_max_z=max(r.ball_z for r in air_rows) if air_rows else row.ball_z,
                        is_turnover=prev_team != row.team_id,
                    )
                )
            prev_controller = row.player_id
            prev_team = row.team_id
            prev_row = row
            air_rows.clear()
            candidate_controller = None
            candidate_frames = 0
        else:
            air_rows.append(row)
    return passes


def suppress_shot_turnovers(pass_rows: List[PassRecord], rows: Sequence[HandlerRow], meta: Dict, args) -> List[PassRecord]:
    result = (meta.get("result_type") or "").lower()
    if result not in {"made_shot", "miss_shot", "foul", "foul_committed"}:
        return pass_rows
    shot_player = meta.get("shot_player_id")
    offense_team = meta.get("offense_team_id")
    end_frame = meta.get("end_frame")
    try:
        end_frame = int(end_frame)
    except (TypeError, ValueError):
        end_frame = None
    frame_lookup = {row.array_idx: row.frame_idx for row in rows}
    for rec in pass_rows:
        if not rec.is_turnover:
            continue
        if shot_player is not None and rec.passer_id != int(shot_player):
            continue
        if offense_team is not None and rec.passer_team != int(offense_team):
            continue
        if rec.ball_max_z < args.shot_ball_height_threshold:
            if end_frame is None or end_frame - frame_lookup.get(rec.end_idx, end_frame) > args.shot_turnover_window:
                continue
        rec.is_turnover = False
    return pass_rows
