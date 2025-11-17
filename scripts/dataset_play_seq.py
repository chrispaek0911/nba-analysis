from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from models.graph_sequence_builder import build_graph_sequence_from_npz
from models.global_features import build_global_features


class PlaySeqDataset(Dataset):
    """Dataset yielding (graph_seq, global_feats, label, npz_path)."""

    def __init__(
        self,
        graph_root: Path,
        parquet_root: Path,
        seq_len: int = 16,
        stage: int = 2,
        label_key: str = "result_type_id",
    ) -> None:
        super().__init__()
        self.graph_root = Path(graph_root)
        self.parquet_root = Path(parquet_root)
        self.seq_len = seq_len
        self.stage = stage
        self.label_key = label_key

        # index all npz files
        self.files: List[Path] = sorted(self.graph_root.glob("*/*.npz"))
        if not self.files:
            raise ValueError(f"No npz files under {self.graph_root}")

        # cache per-game parquet frames
        self.parquet_cache: Dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load_parquet_row(self, npz_path: Path) -> Optional[pd.Series]:
        game_id = npz_path.parent.name
        parquet_path = self.parquet_root / f"{game_id}.parquet"
        if not parquet_path.exists():
            return None
        if game_id not in self.parquet_cache:
            self.parquet_cache[game_id] = pd.read_parquet(parquet_path)
        df = self.parquet_cache[game_id]
        play_key = npz_path.stem  # play_XXXX
        try:
            play_id = int(play_key.split("_")[1])
        except Exception:
            return None
        row = df[df["play_id"] == play_id]
        if row.empty:
            return None
        # take first match
        return row.iloc[0]

    def __getitem__(self, idx: int):
        npz_path = self.files[idx]
        # graph sequence
        seq, node_feat_dim = build_graph_sequence_from_npz(npz_path, seq_len=self.seq_len)

        # global feats + label from parquet
        row = self._load_parquet_row(npz_path)
        global_feats: Dict[str, float] = {}
        label_val: Optional[float] = None
        if row is not None:
            global_feats = build_global_features(row, stage=self.stage)
            if self.label_key in row:
                try:
                    label_val = float(row[self.label_key])
                except Exception:
                    label_val = None
            elif f"label_{self.label_key}" in row:
                try:
                    label_val = float(row[f"label_{self.label_key}"])
                except Exception:
                    label_val = None
        return seq, node_feat_dim, global_feats, label_val, str(npz_path)


def to_device(seq: List[torch.Tensor], device: torch.device):
    return [b.to(device) for b in seq]

