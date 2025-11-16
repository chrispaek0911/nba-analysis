#!/usr/bin/env python3
"""Embed per-play arrays with label, spatial, pass, and context features."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transforms.context_features import PlayContextManager, ScoreContextBuilder
from transforms.feature_builder import build_features as build_spatial_behavior_features
from transforms.label_encoder import PlayLabelEncoder
from transforms.pass_features import build_pass_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-data-dir", type=Path, default=Path("/Users/chrispaek/graph_data_new_local"))
    parser.add_argument("--play-labels-dir", type=Path, default=Path("processed/play_segments_shots"))
    parser.add_argument("--pbp-dir", type=Path, default=Path("processed/pbp_with_frames"))
    parser.add_argument("--play-passes-dir", type=Path, default=Path("processed/play_passes"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/play_feature_embeddings"))
    parser.add_argument("--vocab-dir", type=Path, default=Path("mappings/vocabs"))
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--offset-games", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def _first_valid_shot_clock(values: Optional[np.ndarray]) -> float:
    if values is None:
        return 24.0
    arr = np.asarray(values, dtype=float)
    valid = arr[arr > 0]
    return float(valid[0]) if valid.size else 24.0


def _infer_defense_team_id(node_team_ids: np.ndarray, offense_team_id: Optional[int]) -> Optional[int]:
    teams = sorted({int(t) for t in node_team_ids if int(t) != -1})
    for team in teams:
        if offense_team_id is not None and team != offense_team_id:
            return team
    return None


def _load_labels_json(play_npz: np.lib.npyio.NpzFile) -> Dict:
    raw = play_npz["labels_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return json.loads(str(raw))


def _player_team_map(node_player_ids: np.ndarray, node_team_ids: np.ndarray) -> Dict[int, int]:
    mapping = {}
    for pid, tid in zip(node_player_ids, node_team_ids):
        pid = int(pid)
        tid = int(tid)
        if pid == -1:
            continue
        mapping[pid] = tid
    return mapping


def _build_event_frame_lookup(pbp_df: Optional[pd.DataFrame]) -> Dict[int, int]:
    if pbp_df is None or pbp_df.empty:
        return {}
    normalized = {col.lower(): col for col in pbp_df.columns}
    frame_col = normalized.get("frame_start")
    if frame_col is None:
        return {}
    lookup: Dict[int, int] = {}
    for row in pbp_df.itertuples(index=False):
        event_num = getattr(row, "EVENTNUM", None)
        frame_start = getattr(row, frame_col)
        if event_num is None or frame_start is None:
            continue
        try:
            lookup[int(event_num)] = int(frame_start)
        except (TypeError, ValueError):
            continue
    return lookup


def _resolve_foul_team(labels: Dict, player_team: Dict[int, int], context: Dict) -> Optional[int]:
    foul_player = labels.get("foul_player_id")
    foul_team_id = player_team.get(int(foul_player)) if foul_player is not None else None
    foul_type = labels.get("foul_type")
    if isinstance(foul_type, float):
        foul_type = None
    if foul_team_id is None and foul_type:
        token = str(foul_type).lower()
        if "offensive" in token:
            foul_team_id = context.get("offense_team_id")
        elif any(key in token for key in ["shoot", "block", "defensive", "loose"]):
            foul_team_id = context.get("defense_team_id")
    return foul_team_id


def process_game(
    game_id: str,
    args: argparse.Namespace,
    encoder: PlayLabelEncoder,
    encoder_lock: Optional[Lock] = None,
) -> None:
    labels_path = args.play_labels_dir / f"{game_id}.parquet"
    if not labels_path.exists():
        print(f"[WARN] missing labels for {game_id}")
        return
    out_path = args.output_dir / f"{game_id}.parquet"
    if out_path.exists() and args.skip_existing:
        print(f"[SKIP] {game_id} features exist")
        return
    labels_df = pd.read_parquet(labels_path)
    pbp_path = args.pbp_dir / f"{game_id}.parquet"
    pbp_df = pd.read_parquet(pbp_path) if pbp_path.exists() else None
    score_builder = ScoreContextBuilder(pbp_df)
    context_manager = PlayContextManager(labels_df, score_builder)
    event_frames = _build_event_frame_lookup(pbp_df)
    passes_path = args.play_passes_dir / f"{game_id}.parquet"
    passes_df = pd.read_parquet(passes_path) if passes_path.exists() else None
    pass_indices = passes_df.groupby("play_id").indices if passes_df is not None and not passes_df.empty else {}
    graph_dir = args.graph_data_dir / game_id
    if not graph_dir.exists():
        print(f"[WARN] missing graph data for {game_id}")
        return

    rows = []
    for row in context_manager.sorted_df.itertuples(index=False):
        play_id = int(row.play_id)
        play_key = f"play_{play_id:04d}"
        npz_path = graph_dir / f"{play_key}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            labels = _load_labels_json(data)
            positions = data["positions"]
            node_player_ids = data["node_player_ids"]
            node_team_ids = data["node_team_ids"]
            shot_clocks = data["shot_clocks"] if "shot_clocks" in data.files else None
        offense_team_id = int(row.offense_team_id) if row.offense_team_id else None
        row_series = pd.Series(row._asdict())
        context = context_manager.context_for(row_series, offense_team_id)
        defense_team_id = context.get("defense_team_id") or _infer_defense_team_id(node_team_ids, offense_team_id)
        context["defense_team_id"] = defense_team_id
        context["shot_clock_start"] = _first_valid_shot_clock(shot_clocks)
        player_map = _player_team_map(node_player_ids, node_team_ids)

        if encoder_lock:
            with encoder_lock:
                label_payload = encoder.encode(labels, context)
        else:
            label_payload = encoder.encode(labels, context)
        spatial_payload = build_spatial_behavior_features(positions, node_team_ids, context)

        pass_rows = None
        idx = pass_indices.get(play_id)
        if idx is not None:
            pass_rows = passes_df.iloc[idx]
        shot_event_id = row_series.get("shot_event_id")
        shot_present = bool(pd.notna(shot_event_id))
        shot_frame = event_frames.get(int(shot_event_id)) if shot_present and shot_event_id is not None else None
        shot_distance = labels.get("shot_distance")
        pass_payload = build_pass_features(pass_rows, shot_frame, shot_present, shot_distance)

        row_features: Dict[str, float] = {
            "offense_team_id": float(offense_team_id or -1),
            "defense_team_id": float(defense_team_id or -1),
            "period": float(context.get("period", 0)),
            "seconds_remaining_period": float(context.get("seconds_remaining_period", 0.0)),
            "score_home": float(context.get("score_home", 0.0)),
            "score_away": float(context.get("score_away", 0.0)),
            "relative_score_margin": float(context.get("relative_score_margin", 0.0)),
            "is_offense_trailing": float(context.get("is_offense_trailing", 0)),
            "shot_clock_start": float(context.get("shot_clock_start", 24.0)),
        }
        row_labels: Dict[str, float] = {}
        row_aux: Dict[str, float] = {}

        row_features.update(label_payload["features"])
        row_labels.update(label_payload["labels"])
        row_aux.update(label_payload["aux_targets"])

        row_features.update(spatial_payload["features"])
        row_aux.update(spatial_payload["aux_targets"])

        row_features.update(pass_payload.get("features", {}))
        row_labels.update(pass_payload.get("labels", {}))
        row_aux.update(pass_payload.get("aux_targets", {}))

        metadata_row = {
            "game_id": game_id,
            "play_id": play_id,
        }
        metadata_row.update({f"feat_{k}": v for k, v in row_features.items()})
        metadata_row.update({f"label_{k}": v for k, v in row_labels.items()})
        metadata_row.update({f"aux_{k}": v for k, v in row_aux.items()})
        rows.append(metadata_row)

        foul_team_id = _resolve_foul_team(labels, player_map, context)
        context_manager.register_foul(context.get("period", 0), foul_team_id)

    if not rows:
        print(f"[WARN] {game_id} produced zero rows")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"{game_id}: saved {len(rows)} feature rows")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = PlayLabelEncoder(args.vocab_dir)
    encoder_lock = Lock() if args.num_workers and args.num_workers > 1 else None
    games = sorted(p.name for p in args.graph_data_dir.iterdir() if p.is_dir())
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]
    if args.num_workers and args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_game, game_id, args, encoder, encoder_lock): game_id
                for game_id in games
            }
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[ERROR] {game_id} failed: {exc}")
    else:
        for game_id in games:
            process_game(game_id, args, encoder)
    encoder.save()


if __name__ == "__main__":
    main()
