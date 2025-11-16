#!/usr/bin/env python3
"""Assemble final per-play graph data packages combining arrays, edges, and labels."""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ARRAY_ATTRS = [
    "frame_ids",
    "event_ids",
    "timestamps",
    "game_clocks",
    "shot_clocks",
    "node_player_ids",
    "node_team_ids",
    "positions",
    "result_type",
    "start_frame",
    "end_frame",
]

EDGE_TYPES = ["base_edges", "pass_edges"]

LABEL_COLS = [
    "play_id",
    "result_type",
    "events_in_play",
    "shot_event_id",
    "shot_player_id",
    "shot_player_name",
    "shot_team_id",
    "shot_event_type",
    "shot_action_type",
    "shot_type",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "shot_distance",
    "shot_loc_x",
    "shot_loc_y",
    "shot_made_flag",
    "rebound_type",
    "rebound_player_id",
    "turnover_type",
    "turnover_player_id",
    "steal_player_id",
    "block_player_id",
    "foul_type",
    "foul_player_id",
    "free_throw_attempts",
    "free_throw_made",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-arrays-dir", type=Path, default=Path("processed/graph_arrays"))
    parser.add_argument("--graph-edges-dir", type=Path, default=Path("processed/graph_edges"))
    parser.add_argument("--play-labels-dir", type=Path, default=Path("processed/play_segments_shots"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/graph_data"))
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--offset-games", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for games")
    return parser.parse_args()


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame(columns=LABEL_COLS)
    df = pd.read_parquet(labels_path)
    missing = [col for col in LABEL_COLS if col not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return df[LABEL_COLS]


def get_play_arrays(arr_npz: np.lib.npyio.NpzFile, play_key: str) -> dict:
    arrays = {}
    for attr in ARRAY_ATTRS:
        key = f"{play_key}_{attr}"
        if key in arr_npz.files:
            arrays[attr] = arr_npz[key]
        else:
            arrays[attr] = np.empty(0)
    return arrays


def get_play_edges(edge_npz: np.lib.npyio.NpzFile, play_key: str) -> dict:
    edges = {}
    for attr in EDGE_TYPES:
        key = f"{play_key}_{attr}"
        if edge_npz and key in edge_npz.files:
            edges[attr] = edge_npz[key]
        else:
            edges[attr] = np.empty(0)
    return edges


def labels_to_json(labels_df: pd.DataFrame, play_id: int) -> str:
    if labels_df.empty:
        return "{}"
    row = labels_df[labels_df["play_id"] == play_id]
    if row.empty:
        return "{}"
    record = row.iloc[0].to_dict()
    for k, v in record.items():
        if pd.isna(v):
            record[k] = None
        elif isinstance(v, (np.generic,)):
            record[k] = v.item()
    return json.dumps(record)


def save_play(output_dir: Path, play_key: str, arrays: dict, edges: dict, labels_json: str) -> None:
    out_path = output_dir / f"{play_key}.npz"
    payload = {**arrays, **edges, "labels_json": np.array(labels_json, dtype=object)}
    np.savez_compressed(out_path, **payload)


def process_game(game_id: str, args) -> None:
    out_dir = args.output_dir / game_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and any(out_dir.iterdir()):
        print(f"[SKIP] {game_id} graph data exists")
        return

    arrays_path = args.graph_arrays_dir / f"{game_id}.npz"
    edges_path = args.graph_edges_dir / f"{game_id}.npz"
    labels_path = args.play_labels_dir / f"{game_id}.parquet"
    if not arrays_path.exists():
        print(f"[WARN] Missing arrays for {game_id}")
        return
    arr_npz = np.load(arrays_path, allow_pickle=True)
    edge_npz = np.load(edges_path, allow_pickle=True) if edges_path.exists() else None
    labels_df = load_labels(labels_path)

    for play_key in arr_npz["play_keys"]:
        arrays = get_play_arrays(arr_npz, play_key)
        edges = get_play_edges(edge_npz, play_key) if edge_npz else {attr: np.empty(0) for attr in EDGE_TYPES}
        play_id = int(play_key.split("_")[1])
        labels_json = labels_to_json(labels_df, play_id)
        save_play(out_dir, play_key, arrays, edges, labels_json)

    print(f"{game_id}: graph data saved ({len(arr_npz['play_keys'])} plays)")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games = sorted(args.graph_arrays_dir.glob("002*.npz"))
    if args.offset_games:
        games = games[args.offset_games :]
    if args.limit_games is not None:
        games = games[: args.limit_games]
    if not games:
        print("No games found")
        return
    if args.num_workers <= 1:
        for npz_path in games:
            process_game(npz_path.stem, args)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_game, path.stem, args): path.stem for path in games}
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[ERROR] {game_id}: {exc}")


if __name__ == "__main__":
    main()
