from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.state_encoder import StateEncoder
from scripts.dataset_play_seq import PlaySeqDataset, to_device


def infer_label_mapping(parquet_root: Path, label_key: str) -> Dict[int, int]:
    """Scan parquet files to build label -> index mapping (consecutive IDs)."""
    vals = set()
    for p in ParquetIterator(parquet_root):
        # prefer label_<key> then raw key
        key = f"label_{label_key}" if f"label_{label_key}" in p.columns else label_key
        if key not in p.columns:
            continue
        try:
            vals.update(int(v) for v in p[key].dropna().unique())
        except Exception:
            continue
    mapping = {v: i for i, v in enumerate(sorted(vals))}
    return mapping


class ParquetIterator:
    """Lightweight iterator over parquet files to avoid repeated read calls."""

    def __init__(self, root: Path):
        self.files = sorted(Path(root).glob("*.parquet"))

    def __iter__(self):
        for f in self.files:
            try:
                yield pd.read_parquet(f)
            except Exception:
                continue


def dict_to_tensor(feats: Dict[str, float], keys: List[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([feats.get(k, 0.0) for k in keys], dtype=torch.float, device=device).unsqueeze(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-root", type=Path, required=True, help="processed/graph_data_new")
    p.add_argument("--parquet-root", type=Path, required=True, help="processed/play_feature_embeddings")
    p.add_argument("--stage", type=int, default=2, help="global feature stage (1/2/3)")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--label-key", type=str, default="result_type_id")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt", type=Path, default=None, help="checkpoint path to save/load")
    p.add_argument("--save-every", type=int, default=100, help="save checkpoint every N steps")
    p.add_argument("--no-shuffle", action="store_true", help="disable shuffling (for step resume)")
    return p.parse_args()


def _collate(batch):
    # batch_size=1 assumed for simplicity; just unwrap
    return batch[0]


def load_ckpt(ckpt_path: Optional[Path], encoder, head, optim, device):
    if ckpt_path is None or not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    if head is not None and "head" in ckpt and isinstance(head, nn.Module):
        head.load_state_dict(ckpt["head"])
    if optim is not None and "optim" in ckpt and isinstance(optim, torch.optim.Optimizer):
        optim.load_state_dict(ckpt["optim"])
    return ckpt


def save_ckpt(ckpt_path: Optional[Path], state: Dict) -> None:
    if ckpt_path is None:
        return
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    label_mapping = infer_label_mapping(args.parquet_root, args.label_key)
    num_classes = max(label_mapping.values()) + 1 if label_mapping else 1

    dataset = PlaySeqDataset(
        graph_root=args.graph_root,
        parquet_root=args.parquet_root,
        seq_len=args.seq_len,
        stage=args.stage,
        label_key=args.label_key,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.no_shuffle, collate_fn=_collate)

    encoder = StateEncoder(node_feat_dim=11, hidden_dim=128).to(device)
    # head input: encoder z (128) + global feats (len(feat_keys) determined lazily)
    # initialize with encoder dim only; we will rebuild once feat_keys known
    head = None
    criterion = nn.CrossEntropyLoss()
    optim: Optional[torch.optim.Optimizer] = None
    feat_keys: Optional[List[str]] = None
    start_epoch = 0
    start_idx = 0

    # optional resume
    ckpt = load_ckpt(args.ckpt, encoder, head, optim, device)
    if ckpt:
        start_epoch = ckpt.get("epoch", 0)
        start_idx = ckpt.get("idx", 0)
        feat_keys = ckpt.get("feat_keys")
        num_classes = ckpt.get("num_classes", num_classes)
        if feat_keys is not None:
            head = nn.Linear(128 + len(feat_keys), num_classes).to(device)
            optim = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
            if "head" in ckpt and ckpt["head"]:
                head.load_state_dict(ckpt["head"])
            if "optim" in ckpt and ckpt["optim"]:
                optim.load_state_dict(ckpt["optim"])

    total_steps = start_idx

    for epoch in range(start_epoch, args.epochs):
        for step, (seq, node_feat_dim, global_feats, label_val, npz_path) in enumerate(loader, 1):
            # if resuming mid-epoch, skip processed steps
            if epoch == start_epoch and step <= start_idx:
                continue
            if label_val is None:
                continue  # skip if label missing
            seq = to_device(seq, device)
            if label_mapping and int(label_val) in label_mapping:
                label_idx = label_mapping[int(label_val)]
            else:
                continue  # skip unknown label
            label = torch.tensor([label_idx], device=device)

            out = encoder(seq)
            z = out.z  # (batch, 128)

            # global feats -> tensor (ensure consistent key order)
            if feat_keys is None:
                feat_keys = sorted(global_feats.keys())
                # (re)build head/optimizer now that we know global feat dim
                head = nn.Linear(128 + len(feat_keys), num_classes).to(device)
                optim = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
            g = dict_to_tensor(global_feats, feat_keys, device)

            logits = head(torch.cat([z, g], dim=-1))

            loss = criterion(logits, label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_steps += 1
            if step % 50 == 0:
                print(f"[epoch {epoch+1} step {step} global_step {total_steps}] loss={loss.item():.4f} path={npz_path}")
            if args.save_every and total_steps % args.save_every == 0:
                save_ckpt(
                    args.ckpt,
                    {
                        "epoch": epoch,
                        "idx": step,
                        "global_step": total_steps,
                        "feat_keys": feat_keys,
                        "num_classes": num_classes,
                        "encoder": encoder.state_dict(),
                        "head": head.state_dict() if head else None,
                        "optim": optim.state_dict() if optim else None,
                    },
                )

        # reset start_idx after the first resumed epoch
        start_idx = 0
        print(f"epoch {epoch+1} done")


if __name__ == "__main__":
    main()
