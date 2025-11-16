"""Utilities to turn graph_data npz into a PyG Batch sequence with richer node features.

This builds per-frame Batches with:
- node features: positions (x,y,z) + velocity (vx,vy,vz) + team one-hot (ball/off/def) + is_ball flag.
- edges: per-frame base_edges filtered by frame_idx; edge_attr = [weight, edge_type].

Usage:
    seq, node_feat_dim = build_graph_sequence_from_npz(npz_path, offense_team_id=<optional>)
    # seq is a list of Batch, length = num_frames, ready for StateEncoder
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data


def _safe_offense_team_id(labels_json: Optional[str], node_team_ids: np.ndarray) -> Optional[int]:
    """Try to infer offense_team_id; fall back to first non -1 team if missing."""
    if labels_json:
        try:
            obj = json.loads(str(labels_json))
            off = obj.get("offense_team_id")
            if off not in (None, "", -1):
                return int(off)
        except Exception:
            pass
    teams = [int(t) for t in node_team_ids if int(t) != -1]
    return teams[0] if teams else None


def build_graph_sequence_from_npz(
    npz_path: Union[str, Path], offense_team_id: Optional[int] = None
) -> Tuple[List[Batch], int]:
    """
    Convert a single play npz into a list of PyG Batch objects (one per frame) with expanded node features.

    Node features per node:
        positions (x,y,z) +
        velocity (vx,vy,vz) +
        team one-hot (ball/offense/defense) +
        is_ball flag
    """
    data = np.load(Path(npz_path), allow_pickle=True)
    positions = data["positions"]  # (T, 11, 3)
    node_team_ids = data["node_team_ids"]  # (11,)
    labels_json = data.get("labels_json")
    off_id = offense_team_id or _safe_offense_team_id(labels_json, node_team_ids)

    T, N, _ = positions.shape
    # velocity: simple frame difference, pad last with previous
    velocity = np.zeros_like(positions)
    if T > 1:
        velocity[1:] = positions[1:] - positions[:-1]
        velocity[-1] = velocity[-2]

    # team one-hot: ball/off/def
    team_onehot = np.zeros((N, 3), dtype=np.float32)
    for i, tid in enumerate(node_team_ids):
        tid = int(tid)
        if tid == -1:  # ball
            team_onehot[i, 0] = 1.0
        elif off_id is not None and tid == off_id:
            team_onehot[i, 1] = 1.0
        else:
            team_onehot[i, 2] = 1.0
    team_onehot = np.broadcast_to(team_onehot, (T, N, 3))

    # is_ball flag
    is_ball = (node_team_ids == -1).astype(np.float32)
    is_ball = np.broadcast_to(is_ball, (T, N))[..., None]

    # final node features: (T, N, 3 + 3 + 3 + 1) = (T, N, 10)
    x = np.concatenate([positions, velocity, team_onehot, is_ball], axis=-1)
    node_feat_dim = x.shape[-1]

    base_edges = data["base_edges"]
    seq: List[Batch] = []
    for t in range(T):
        xt = torch.tensor(x[t], dtype=torch.float)  # (N, D)
        mask = base_edges["frame_idx"] == t
        edges_t = base_edges[mask]
        if edges_t.size == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor([edges_t["src"], edges_t["dst"]], dtype=torch.long)
            edge_attr = torch.tensor(
                np.stack([edges_t["weight"], edges_t["edge_type"]], axis=1), dtype=torch.float
            )
        data_t = Data(x=xt, edge_index=edge_index, edge_attr=edge_attr)
        seq.append(Batch.from_data_list([data_t]))

    return seq, node_feat_dim

