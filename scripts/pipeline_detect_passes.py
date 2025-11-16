#!/usr/bin/env python3
"""Detect passes from per-play frame slices using the shared visualization logic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pass_detection import HandlerRow, PlayContext, PassRecord, detect_passes, suppress_shot_turnovers

FRAME_FIELDS = [
    "frame_idx",
    "event_id",
    "timestamp_ms",
    "game_clock",
    "shot_clock",
    "team_id",
    "player_id",
    "x",
    "y",
    "z",
]

OUTPUT_COLUMNS = [
    "game_id",
    "play_id",
    "pass_id",
    "passer_id",
    "receiver_id",
    "passer_team_id",
    "receiver_team_id",
    "start_frame",
    "end_frame",
    "start_event",
    "end_event",
    "start_timestamp_ms",
    "end_timestamp_ms",
    "start_game_clock",
    "end_game_clock",
    "air_frames",
    "ball_max_z",
    "is_turnover_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play-frames-dir", type=Path, default=Path("processed/play_frames"))
    parser.add_argument("--play-labels-dir", type=Path, default=Path("processed/play_segments_shots"))
    parser.add_argument("--pbp-dir", type=Path, default=Path("processed/pbp_with_frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/play_passes"))
    parser.add_argument("--distance-threshold", type=float, default=3.0)
    parser.add_argument("--max-control-z", type=float, default=9.0)
    parser.add_argument("--max-control-speed", type=float, default=2.0)
    parser.add_argument("--offense-control-frames", type=int, default=3)
    parser.add_argument("--defense-control-frames", type=int, default=5)
    parser.add_argument("--defense-max-control-z", type=float, default=8.5)
    parser.add_argument("--defense-z-velocity-threshold", type=float, default=0.7)
    parser.add_argument("--min-air-frames", type=int, default=2)
    parser.add_argument("--initial-skip-frames", type=int, default=5)
    parser.add_argument("--deflection-confirm-frames", type=int, default=2)
    parser.add_argument("--shot-guard-frames", type=int, default=12)
    parser.add_argument("--shot-z-threshold", type=float, default=10.5)
    parser.add_argument("--shot-speed-threshold", type=float, default=3.0)
    parser.add_argument("--shot-ball-height-threshold", type=float, default=10.5)
    parser.add_argument("--shot-turnover-window", type=int, default=150)
    parser.add_argument("--jump-ball-guard-frames", type=int, default=10)
    parser.add_argument("--made-shot-cooldown-boost", type=int, default=8)
    parser.add_argument("--made-shot-endline-threshold", type=float, default=4.0)
    parser.add_argument("--made-shot-floor-z", type=float, default=4.0)
    parser.add_argument("--court-length", type=float, default=94.0)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--offset-games", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_play(npz_path: Path) -> Dict:
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data["frames"]
        meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) else data["meta"]
    return {"frames": frames, "meta": meta}


def load_play_metadata(game_id: str, labels_dir: Path) -> Dict[int, Dict]:
    path = labels_dir / f"{game_id}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        for k, v in list(record.items()):
            if pd.isna(v):
                record[k] = None
        out[int(row["play_id"])] = record
    return out


def load_pbp_events(game_id: str, pbp_dir: Path) -> Dict[int, int]:
    path = pbp_dir / f"{game_id}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["EVENTNUM", "EVENTMSGTYPE"])
    return {int(row.EVENTNUM): int(row.EVENTMSGTYPE) for row in df.itertuples()}


def compute_handler_rows_from_frames(frames: np.ndarray, args: argparse.Namespace) -> List[HandlerRow]:
    df = pd.DataFrame(frames)[FRAME_FIELDS]
    df = df.sort_values("frame_idx")
    ball_df = df[df["team_id"] == -1].sort_values("frame_idx").reset_index(drop=True)
    if ball_df.empty:
        return []
    players_df = df[df["team_id"] != -1]
    player_groups = {int(frame): group for frame, group in players_df.groupby("frame_idx")}
    ball_xy = ball_df[["x", "y"]].to_numpy()
    ball_z = ball_df["z"].to_numpy()
    ball_speed = np.zeros(len(ball_df), dtype=float)
    ball_dz = np.zeros(len(ball_df), dtype=float)
    if len(ball_df) > 1:
        diffs = np.diff(ball_xy, axis=0)
        ball_speed[1:] = np.linalg.norm(diffs, axis=1)
        ball_dz[1:] = np.diff(ball_z)
    rows: List[HandlerRow] = []
    for idx in range(len(ball_df)):
        frame = int(ball_df.at[idx, "frame_idx"])
        group = player_groups.get(frame)
        if group is None or group.empty:
            continue
        coords = group[["x", "y"]].to_numpy()
        dists = np.linalg.norm(coords - ball_xy[idx], axis=1)
        closest_idx = int(np.argmin(dists))
        closest = group.iloc[closest_idx]
        has_control = (
            dists[closest_idx] <= args.distance_threshold
            and ball_z[idx] <= args.max_control_z
            and ball_speed[idx] <= args.max_control_speed
        )
        rows.append(
            HandlerRow(
                array_idx=idx,
                frame_idx=frame,
                player_id=int(closest["player_id"]),
                team_id=int(closest["team_id"]),
                distance=float(dists[closest_idx]),
                ball_z=float(ball_z[idx]),
                ball_dz=float(ball_dz[idx]),
                ball_speed=float(ball_speed[idx]),
                has_control=bool(has_control),
                ball_x=float(ball_df.at[idx, "x"]),
                ball_y=float(ball_df.at[idx, "y"]),
                event_id=int(ball_df.at[idx, "event_id"]),
                timestamp_ms=int(ball_df.at[idx, "timestamp_ms"]),
                game_clock=float(ball_df.at[idx, "game_clock"]),
                shot_clock=None if pd.isna(ball_df.at[idx, "shot_clock"]) else float(ball_df.at[idx, "shot_clock"]),
            )
        )
    return rows


def pass_records_to_rows(
    pass_records: List[PassRecord],
    rows: List[HandlerRow],
    play_meta: Dict,
    pass_offset: int,
) -> List[Dict]:
    output: List[Dict] = []
    row_lookup = {row.array_idx: row for row in rows}
    for idx, rec in enumerate(pass_records, start=1):
        start_row = row_lookup.get(rec.start_idx)
        end_row = row_lookup.get(rec.end_idx)
        if start_row is None or end_row is None:
            continue
        output.append(
            {
                "game_id": play_meta.get("game_id"),
                "play_id": play_meta.get("play_id"),
                "pass_id": pass_offset + idx,
                "passer_id": rec.passer_id,
                "receiver_id": rec.receiver_id,
                "passer_team_id": rec.passer_team,
                "receiver_team_id": rec.receiver_team,
                "start_frame": start_row.frame_idx,
                "end_frame": end_row.frame_idx,
                "start_event": start_row.event_id,
                "end_event": end_row.event_id,
                "start_timestamp_ms": start_row.timestamp_ms,
                "end_timestamp_ms": end_row.timestamp_ms,
                "start_game_clock": start_row.game_clock,
                "end_game_clock": end_row.game_clock,
                "air_frames": rec.air_frames,
                "ball_max_z": rec.ball_max_z,
                "is_turnover_pass": rec.is_turnover,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games = sorted(p for p in args.play_frames_dir.iterdir() if p.is_dir())
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]

    for game_dir in games:
        game_id = game_dir.name
        out_path = args.output_dir / f"{game_id}.parquet"
        if out_path.exists() and args.skip_existing:
            print(f"[SKIP] {game_id} passes already computed")
            continue
        metadata = load_play_metadata(game_id, args.play_labels_dir)
        pbp_events = load_pbp_events(game_id, args.pbp_dir)
        pass_rows: List[Dict] = []
        pass_offset = 0
        for npz_path in sorted(game_dir.glob("play_*.npz")):
            play = load_play(npz_path)
            play_meta = dict(play.get("meta", {}))
            play_id = int(play_meta.get("play_id", 0))
            play_meta.setdefault("game_id", game_id)
            play_meta.setdefault("play_id", play_id)
            if play_id in metadata:
                play_meta = {**play_meta, **metadata[play_id]}
            rows = compute_handler_rows_from_frames(play["frames"], args)
            if not rows:
                continue
            start_event = play_meta.get("start_event")
            try:
                start_event = int(start_event) if start_event is not None else None
            except (TypeError, ValueError):
                start_event = None
            context = PlayContext(
                start_event_msg_type=pbp_events.get(start_event),
                jump_ball_guard_frames=args.jump_ball_guard_frames,
            )
            pass_records = detect_passes(rows, play_meta, args, context)
            pass_records = suppress_shot_turnovers(pass_records, rows, play_meta, args)
            converted = pass_records_to_rows(pass_records, rows, play_meta, pass_offset)
            pass_rows.extend(converted)
            pass_offset += len(pass_records)

        if pass_rows:
            pd.DataFrame(pass_rows)[OUTPUT_COLUMNS].to_parquet(out_path, index=False)
            print(f"{game_id}: detected {len(pass_rows)} passes")
        else:
            print(f"{game_id}: no passes detected")


if __name__ == "__main__":
    main()
