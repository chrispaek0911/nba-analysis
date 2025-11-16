#!/usr/bin/env python3
"""Visualize plays with inline pass detection, filtering, and rich HUD."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd

from pass_detection import (
    HandlerRow,
    PassRecord,
    PlayContext,
    detect_passes,
    suppress_shot_turnovers,
)

COURT_LENGTH = 94
COURT_WIDTH = 50
BALL_COLOR = "#FF595E"
OFFENSE_COLOR = "#FFB703"
DEFENSE_COLOR = "#4CC9F0"
PASS_COLOR = "#F4F1DE"
TURNOVER_PASS_COLOR = "#FA8072"
BACKGROUND_COLOR = "#07142b"
HUD_BG_COLOR = "#0b172e"
HUD_TEXT_COLOR = "#E2E8F0"
TRAIL_COLOR = "#FFFFFF"
FFMPEG_EXTRA_ARGS = ["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]


@dataclass
class FrameInfo:
    play_id: int
    raw_frame: int
    game_clock: float
    shot_clock: float
    offense_team_id: Optional[int]


@dataclass
class PassSegment:
    start_idx: int
    end_idx: int
    src: int
    dst: int
    turnover: bool
    play_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--play-id", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ball-trail", type=int, default=40)
    parser.add_argument("--graph-arrays-dir", type=Path, default=Path("processed/graph_arrays"))
    parser.add_argument("--play-labels-dir", type=Path, default=Path("processed/play_segments_shots"))
    parser.add_argument("--pbp-dir", type=Path, default=Path("processed/pbp_with_frames"))
    parser.add_argument("--play-passes-dir", type=Path, default=Path("processed/play_passes"))
    parser.add_argument("--use-existing-passes", action="store_true", help="Load passes from parquet instead of re-detecting.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--show", action="store_true")

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
    parser.add_argument("--passes-output-dir", type=Path, default=None)
    return parser.parse_args()


def load_play_metadata(game_id: str, play_ids: Sequence[int], labels_dir: Path) -> Dict[int, Dict]:
    labels_path = labels_dir / f"{game_id}.parquet"
    if not labels_path.exists():
        return {pid: {} for pid in play_ids}
    df = pd.read_parquet(labels_path)
    df = df[df["play_id"].isin(play_ids)]
    out: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        for k, v in list(record.items()):
            if pd.isna(v):
                record[k] = None
        out[int(row["play_id"])] = record
    for pid in play_ids:
        out.setdefault(pid, {})
    return out


def load_pbp_events(game_id: str, pbp_dir: Path) -> Dict[int, int]:
    path = pbp_dir / f"{game_id}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["EVENTNUM", "EVENTMSGTYPE"])
    return {int(row.EVENTNUM): int(row.EVENTMSGTYPE) for row in df.itertuples()}


def load_game_arrays(game_id: str, arrays_dir: Path, play_ids: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    path = arrays_dir / f"{game_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing arrays file: {path}")
    data = np.load(path, allow_pickle=True)
    plays: Dict[int, Dict[str, np.ndarray]] = {}
    required = ["positions", "frame_ids", "node_player_ids", "node_team_ids", "game_clocks", "shot_clocks", "timestamps"]
    for pid in play_ids:
        key = f"play_{pid:04d}"
        if any(f"{key}_{attr}" not in data for attr in required):
            raise KeyError(f"{key} arrays missing in {path}")
        plays[pid] = {attr: data[f"{key}_{attr}"] for attr in required}
    return plays


def ensure_consistent_nodes(play_arrays: Dict[int, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    base_players = None
    base_teams = None
    for payload in play_arrays.values():
        players = payload["node_player_ids"]
        teams = payload["node_team_ids"]
        if base_players is None:
            base_players = players
            base_teams = teams
            continue
        if not np.array_equal(base_players, players) or not np.array_equal(base_teams, teams):
            raise ValueError("Node ordering differs between plays; visualize separately.")
    if base_players is None or base_teams is None:
        raise ValueError("No plays loaded")
    return base_players, base_teams


def compute_handler_rows(arrays: Dict[str, np.ndarray], args: argparse.Namespace) -> List[HandlerRow]:
    positions = arrays["positions"]
    frame_ids = arrays["frame_ids"].astype(int)
    node_players = arrays["node_player_ids"].astype(int)
    node_teams = arrays["node_team_ids"].astype(int)
    ball_idx = int(np.where(node_teams == -1)[0][0])
    player_indices = np.where(node_teams >= 0)[0]
    ball_xy = positions[:, ball_idx, :2]
    ball_z = positions[:, ball_idx, 2]
    ball_speed = np.zeros(len(frame_ids), dtype=float)
    ball_dz = np.zeros(len(frame_ids), dtype=float)
    if len(frame_ids) > 1:
        diffs = np.diff(ball_xy, axis=0)
        ball_speed[1:] = np.linalg.norm(diffs, axis=1)
        ball_dz[1:] = np.diff(ball_z)
    rows: List[HandlerRow] = []
    for t in range(len(frame_ids)):
        player_xy = positions[t, player_indices, :2]
        dists = np.linalg.norm(player_xy - ball_xy[t], axis=1)
        closest = int(np.argmin(dists))
        node_idx = player_indices[closest]
        has_control = (
            dists[closest] <= args.distance_threshold
            and ball_z[t] <= args.max_control_z
            and ball_speed[t] <= args.max_control_speed
        )
        rows.append(
            HandlerRow(
                array_idx=t,
                frame_idx=int(frame_ids[t]),
                player_id=int(node_players[node_idx]),
                team_id=int(node_teams[node_idx]),
                distance=float(dists[closest]),
                ball_z=float(ball_z[t]),
                ball_dz=float(ball_dz[t]),
                ball_speed=float(ball_speed[t]),
                has_control=bool(has_control),
                ball_x=float(ball_xy[t, 0]),
                ball_y=float(ball_xy[t, 1]),
            )
        )
    return rows



def convert_passes_to_segments(pass_rows: List[PassRecord], arrays: Dict[str, np.ndarray], play_id: int) -> List[PassSegment]:
    segments: List[PassSegment] = []
    players = arrays["node_player_ids"].astype(int)
    player_to_idx = {int(pid): idx for idx, pid in enumerate(players)}
    for rec in pass_rows:
        src = player_to_idx.get(rec.passer_id)
        dst = player_to_idx.get(rec.receiver_id)
        if None in (src, dst):
            continue
        segments.append(
            PassSegment(
                start_idx=rec.start_idx,
                end_idx=rec.end_idx,
                src=src,
                dst=dst,
                turnover=rec.is_turnover,
                play_id=play_id,
            )
        )
    return segments

def load_existing_pass_segments(
    game_id: str,
    play_ids: Sequence[int],
    play_arrays: Dict[int, Dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> Dict[int, List[PassSegment]]:
    path = args.play_passes_dir / f"{game_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing pass parquet: {path}")
    df = pd.read_parquet(path)
    df = df[df['play_id'].isin(play_ids)]
    segments: Dict[int, List[PassSegment]] = {pid: [] for pid in play_ids}
    for pid in play_ids:
        arrays = play_arrays[pid]
        frame_ids = arrays['frame_ids'].astype(int)
        frame_to_idx = {int(frame): idx for idx, frame in enumerate(frame_ids)}
        players = arrays['node_player_ids'].astype(int)
        player_to_idx = {int(pid_): idx for idx, pid_ in enumerate(players)}
        subset = df[df['play_id'] == pid]
        for _, row in subset.iterrows():
            src = player_to_idx.get(int(row['passer_id']))
            dst = player_to_idx.get(int(row['receiver_id']))
            start_idx = frame_to_idx.get(int(row['start_frame']))
            end_idx = frame_to_idx.get(int(row['end_frame']))
            if None in (src, dst, start_idx, end_idx):
                continue
            segments[pid].append(
                PassSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    src=src,
                    dst=dst,
                    turnover=bool(row['is_turnover_pass']),
                    play_id=pid,
                )
            )
    return segments


def build_sequence(
    play_ids: Sequence[int],
    play_arrays: Dict[int, Dict[str, np.ndarray]],
    pass_segments: Dict[int, List[PassSegment]],
    metadata: Dict[int, Dict],
) -> Tuple[np.ndarray, List[FrameInfo], List[PassSegment]]:
    base_players, base_teams = ensure_consistent_nodes(play_arrays)
    positions_blocks = []
    frame_infos: List[FrameInfo] = []
    concat_segments: List[PassSegment] = []
    offset = 0
    for pid in play_ids:
        arrays = play_arrays[pid]
        positions = arrays["positions"]
        frames = arrays["frame_ids"].astype(int)
        clocks = arrays["game_clocks"].astype(float)
        shot_clocks = arrays["shot_clocks"].astype(float)
        positions_blocks.append(positions)
        offense_team = metadata.get(pid, {}).get("offense_team_id")
        for idx, raw in enumerate(frames):
            frame_infos.append(
                FrameInfo(
                    play_id=pid,
                    raw_frame=int(raw),
                    game_clock=float(clocks[idx]),
                    shot_clock=float(shot_clocks[idx]),
                    offense_team_id=int(offense_team) if offense_team is not None else None,
                )
            )
        for seg in pass_segments.get(pid, []):
            concat_segments.append(
                PassSegment(
                    start_idx=seg.start_idx + offset,
                    end_idx=seg.end_idx + offset,
                    src=seg.src,
                    dst=seg.dst,
                    turnover=seg.turnover,
                    play_id=pid,
                )
            )
        offset += len(frames)
    positions = np.concatenate(positions_blocks, axis=0)
    play_arrays["_node_player_ids"] = base_players
    play_arrays["_node_team_ids"] = base_teams
    return positions, frame_infos, concat_segments


def draw_court(ax: plt.Axes) -> None:
    ax.set_xlim(-1, COURT_LENGTH + 1)
    ax.set_ylim(-1, COURT_WIDTH + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND_COLOR)
    outer = patches.Rectangle((0, 0), COURT_LENGTH, COURT_WIDTH, linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(outer)
    ax.plot([COURT_LENGTH / 2, COURT_LENGTH / 2], [0, COURT_WIDTH], color="white", linewidth=1.5)
    center_circle = patches.Circle((COURT_LENGTH / 2, COURT_WIDTH / 2), 6, linewidth=1.5, edgecolor="white", facecolor="none")
    ax.add_patch(center_circle)
    for is_left in (True, False):
        lane_origin = 0 if is_left else COURT_LENGTH - 19
        hoop_x = 5.25 if is_left else COURT_LENGTH - 5.25
        theta = (-90, 90) if is_left else (90, 270)
        lane = patches.Rectangle((lane_origin, (COURT_WIDTH - 16) / 2), 19, 16, linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(lane)
        rim = patches.Circle((hoop_x, COURT_WIDTH / 2), 0.75, linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(rim)
        arc = patches.Arc((hoop_x, COURT_WIDTH / 2), 47.5, 47.5, theta1=theta[0], theta2=theta[1], linewidth=1.5, color="white")
        ax.add_patch(arc)


def format_player_label(pid: int) -> str:
    return "BALL" if pid < 0 else str(pid)


def describe_passes(pass_segments: Sequence[PassSegment], frame_infos: Sequence[FrameInfo], node_player_ids: np.ndarray) -> None:
    if not pass_segments:
        print("No pass segments detected for selected plays.")
        return
    print("Pass segments (ordered by start frame):")
    for idx, seg in enumerate(sorted(pass_segments, key=lambda s: s.start_idx), start=1):
        start_info = frame_infos[seg.start_idx]
        end_info = frame_infos[min(seg.end_idx, len(frame_infos) - 1)]
        passer = format_player_label(int(node_player_ids[seg.src]))
        receiver = format_player_label(int(node_player_ids[seg.dst]))
        flag = "TURNOVER" if seg.turnover else "COMPLETE"
        print(
            f"  #{idx:02d} | play {seg.play_id:04d} | frames {start_info.raw_frame}->{end_info.raw_frame} "
            f"| {passer} -> {receiver} | {flag}"
        )


def hud_text(metadata: Dict[int, Dict], play_id: int, pass_segments: Sequence[PassSegment]) -> str:
    meta = metadata.get(play_id, {})
    shot_desc = meta.get("shot_action_type") or meta.get("shot_event_type") or "N/A"
    result = meta.get("result_type") or "unknown"
    dist = meta.get("shot_distance")
    dist_str = f"{float(dist):.1f} ft" if dist is not None else "N/A"
    total = len(pass_segments)
    turnovers = sum(seg.turnover for seg in pass_segments)
    risk = (turnovers / total) if total else 0.0
    agg = 0.5 if total == 0 else min(0.99, 0.4 + total / 20)
    defense = 0.5 if total == 0 else min(0.99, 0.4 + turnovers / max(1, total))
    return (
        f"Shot: {shot_desc}\n"
        f"Result: {result}\n"
        f"Dist: {dist_str}\n"
        f"Pass risk: {risk:.2f}\n"
        f"Turnover passes: {turnovers}\n"
        f"Agg {agg:.2f} / Def {defense:.2f}"
    )


def build_animation(
    args: argparse.Namespace,
    positions: np.ndarray,
    frame_infos: Sequence[FrameInfo],
    pass_segments: Sequence[PassSegment],
    node_player_ids: np.ndarray,
    node_team_ids: np.ndarray,
    metadata: Dict[int, Dict],
) -> FuncAnimation:
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    plt.subplots_adjust(top=0.93, bottom=0.07)
    draw_court(ax)
    title_text = fig.text(0.5, 0.965, "", color="white", fontsize=18, ha="center", va="top", weight="bold")
    info_text = fig.text(0.015, 0.035, "", color="white", fontsize=11, family="monospace", ha="left", va="bottom")
    hud_label = fig.text(0.97, 0.885, "", color=HUD_TEXT_COLOR, fontsize=11, family="monospace", ha="right", va="center")

    offense_scatter = ax.scatter([], [], s=85, c=OFFENSE_COLOR, edgecolors="black", linewidths=0.6, zorder=4)
    defense_scatter = ax.scatter([], [], s=85, c=DEFENSE_COLOR, edgecolors="black", linewidths=0.6, zorder=4)
    ball_scatter = ax.scatter([], [], s=130, c=BALL_COLOR, edgecolors="white", linewidths=0.9, zorder=5)
    trail_line, = ax.plot([], [], color=TRAIL_COLOR, linewidth=1.2, alpha=0.8, zorder=3)
    pass_line, = ax.plot([], [], color=PASS_COLOR, linewidth=3.0, zorder=5)

    ball_idx = int(np.where(node_team_ids == -1)[0][0])
    player_labels = []
    for idx, pid in enumerate(node_player_ids):
        if idx == ball_idx:
            continue
        text = ax.text(0, 0, "", color="white", fontsize=8, ha="center", va="center", zorder=6)
        player_labels.append((idx, text))

    def update(frame_idx: int):
        pos2d = positions[frame_idx, :, :2]
        frame_info = frame_infos[frame_idx]
        offense_team = frame_info.offense_team_id
        ball_pos = pos2d[ball_idx]
        ball_scatter.set_offsets(ball_pos.reshape(1, -1))
        trail_start = max(0, frame_idx - args.ball_trail)
        trail_positions = positions[trail_start : frame_idx + 1, ball_idx, :2]
        trail_line.set_data(trail_positions[:, 0], trail_positions[:, 1])
        offense_mask = (node_team_ids == offense_team) if offense_team is not None else (node_team_ids >= 0)
        defense_mask = (node_team_ids >= 0) & (~offense_mask)
        offense_scatter.set_offsets(pos2d[offense_mask] if offense_mask.any() else np.empty((0, 2)))
        defense_scatter.set_offsets(pos2d[defense_mask] if defense_mask.any() else np.empty((0, 2)))
        for idx, text in player_labels:
            label_pos = pos2d[idx]
            text.set_position((label_pos[0], label_pos[1] + 0.9))
            text.set_text(format_player_label(int(node_player_ids[idx])))
        active_pass = None
        for seg in pass_segments:
            if seg.start_idx <= frame_idx <= seg.end_idx:
                active_pass = seg
                break
        if active_pass:
            src_pos = pos2d[active_pass.src]
            dst_pos = pos2d[active_pass.dst]
            pass_line.set_data([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]])
            pass_line.set_color(TURNOVER_PASS_COLOR if active_pass.turnover else PASS_COLOR)
            passer = format_player_label(int(node_player_ids[active_pass.src]))
            receiver = format_player_label(int(node_player_ids[active_pass.dst]))
            status = "TURNOVER" if active_pass.turnover else "COMPLETE"
            hud_label.set_text(f"Pass: {passer} → {receiver}\nStatus: {status}")
        else:
            pass_line.set_data([], [])
            hud_label.set_text("")
        info_lines = [
            f"Frame {frame_idx + 1}/{len(frame_infos)} (raw {frame_info.raw_frame})",
            f"Play {frame_info.play_id:04d} | Clock {frame_info.game_clock:6.2f}s | Shot {frame_info.shot_clock:5.2f}s",
            f"Offense team: {frame_info.offense_team_id or 'Unknown'}",
        ]
        info_text.set_text("\n".join(info_lines))
        title_text.set_text(args.title or f"{args.game_id} • Play {frame_info.play_id:04d}")
        return (
            offense_scatter,
            defense_scatter,
            ball_scatter,
            trail_line,
            pass_line,
            info_text,
            hud_label,
        )

    return FuncAnimation(fig, update, frames=len(frame_infos), interval=1000 / args.fps, blit=False)


def main() -> None:
    args = parse_args()
    play_ids = sorted(set(args.play_id))
    metadata = load_play_metadata(args.game_id, play_ids, args.play_labels_dir)
    arrays = load_game_arrays(args.game_id, args.graph_arrays_dir, play_ids)
    pbp_events = load_pbp_events(args.game_id, args.pbp_dir)
    pass_segments_by_play: Dict[int, List[PassSegment]] = {}
    if args.use_existing_passes:
        pass_segments_by_play = load_existing_pass_segments(args.game_id, play_ids, arrays, args)
        if args.passes_output_dir:
            print("[visualize_play] --passes-output-dir ignored when --use-existing-passes is set.")
    else:
        for pid in play_ids:
            play_meta = metadata.get(pid, {})
            rows = compute_handler_rows(arrays[pid], args)
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
            segments = convert_passes_to_segments(pass_records, arrays[pid], pid)
            pass_segments_by_play[pid] = segments
            if args.passes_output_dir:
                args.passes_output_dir.mkdir(parents=True, exist_ok=True)
                records = [
                    {
                        "game_id": args.game_id,
                        "play_id": pid,
                        "passer_id": rec.passer_id,
                        "receiver_id": rec.receiver_id,
                        "start_frame": int(arrays[pid]["frame_ids"][rec.start_idx]),
                        "end_frame": int(arrays[pid]["frame_ids"][rec.end_idx]),
                        "air_frames": rec.air_frames,
                        "ball_max_z": rec.ball_max_z,
                        "is_turnover_pass": rec.is_turnover,
                    }
                    for rec in pass_records
                ]
                columns = [
                    "game_id",
                    "play_id",
                    "passer_id",
                    "receiver_id",
                    "start_frame",
                    "end_frame",
                    "air_frames",
                    "ball_max_z",
                    "is_turnover_pass",
                ]
                df = pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)
                df.to_parquet(args.passes_output_dir / f"{args.game_id}_play{pid:04d}.parquet", index=False)

    positions, frame_infos, pass_segments = build_sequence(play_ids, arrays, pass_segments_by_play, metadata)
    node_players = arrays["_node_player_ids"]
    node_teams = arrays["_node_team_ids"]
    describe_passes(pass_segments, frame_infos, node_players)
    anim = build_animation(args, positions, frame_infos, pass_segments, node_players, node_teams, metadata)

    if args.output and not args.show:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".gif":
            anim.save(args.output, writer="pillow", dpi=args.dpi)
        else:
            writer = FFMpegWriter(fps=args.fps, metadata={"title": args.title or "play_viz"}, extra_args=FFMPEG_EXTRA_ARGS)
            anim.save(args.output, writer=writer, dpi=args.dpi)
        print(f"Saved animation to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
