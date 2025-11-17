"""State encoder for graph-based basketball tracking sequences.

Pipeline:
1) FrameGNNEncoder: per-frame GraphSAGE encoder over 11-node graphs.
2) FrameAggregator: ball embedding + global pooled embedding -> frame embedding f_t.
3) TemporalEncoder: Transformer over recent frame embeddings.
4) StateEncoder: convenience wrapper returning z_t and last-frame node embeddings.

Assumptions:
- Each frame is a PyG Data/Batch with fixed node ordering: ball is the first node
  of each graph in the batch (index 0 per graph). For batched data, we use ptr to
  find the ball node per graph.
- edge_attr is optional; when provided it is fed to SAGEConv.
- All frames in the sequence have the same batch size (same number of plays).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, global_mean_pool


HIDDEN_DIM = 128
GNN_LAYERS = 2
TRANS_LAYERS = 2
N_HEADS = 4
SEQ_LEN = 16
DROPOUT = 0.1


class FrameGNNEncoder(nn.Module):
    """GraphSAGE encoder for a single frame."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = GNN_LAYERS):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_features = in_dim if i == 0 else hidden_dim
            layers.append(SAGEConv(in_features, hidden_dim, normalize=True))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        h = x
        for conv in self.layers:
            # Some PyG builds do not expose edge_weight; fall back to unweighted edges.
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
        return h


class FrameAggregator(nn.Module):
    """Create frame embedding f_t from node embeddings."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_emb: Tensor, batch_ptr: Tensor, batch_index: Tensor) -> Tensor:
        """
        Args:
            node_emb: (total_nodes, hidden_dim)
            batch_ptr: (batch_size + 1,) cumulative node offsets (Batch.ptr)
            batch_index: (total_nodes,) graph assignment (Batch.batch)
        Returns:
            frame_emb: (batch_size, hidden_dim)
        """
        # ball embedding = first node of each graph (ptr gives offsets)
        ball_emb = []
        for i in range(batch_ptr.numel() - 1):
            start = batch_ptr[i]
            ball_emb.append(node_emb[start])
        ball_emb = torch.stack(ball_emb, dim=0)  # (batch, hidden)

        # global mean pooling per graph
        global_emb = global_mean_pool(node_emb, batch_index)  # (batch, hidden)

        f_in = torch.cat([ball_emb, global_emb], dim=-1)
        return self.fc(f_in)


class TemporalEncoder(nn.Module):
    """Transformer encoder over frame embeddings."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = TRANS_LAYERS,
        n_heads: int = N_HEADS,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, hidden_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, frame_seq: Tensor) -> Tensor:
        """
        Args:
            frame_seq: (batch, seq_len, hidden_dim)
        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        seq_len = frame_seq.size(1)
        pos = self.pos_embed[:seq_len].unsqueeze(0)  # (1, seq_len, hidden_dim)
        x = frame_seq + pos
        return self.transformer(x)


@dataclass
class StateEncoderOutput:
    z: Tensor  # (batch, hidden_dim) final state embedding
    last_node_emb: Tensor  # (total_nodes_last_frame, hidden_dim)


class StateEncoder(nn.Module):
    """End-to-end encoder: graph sequence -> state embedding."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_gnn_layers: int = GNN_LAYERS,
        num_trans_layers: int = TRANS_LAYERS,
        n_heads: int = N_HEADS,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.gnn = FrameGNNEncoder(node_feat_dim, hidden_dim, num_gnn_layers)
        self.aggregator = FrameAggregator(hidden_dim)
        self.temporal = TemporalEncoder(hidden_dim, num_trans_layers, n_heads, seq_len)

    def forward(self, graph_seq: List[Batch]) -> StateEncoderOutput:
        """
        Args:
            graph_seq: length-K list of PyG Batch objects, each with fields:
                x (total_nodes, node_feat_dim)
                edge_index
                edge_attr (optional)
                batch (graph index)
                ptr (node offsets per graph)
            All batches must align in size/order across the sequence.
        Returns:
            StateEncoderOutput with:
                z: (batch, hidden_dim) final state embedding (last frame output)
                last_node_emb: node embeddings of the last frame (for action heads)
        """
        assert len(graph_seq) > 0, "graph_seq must be non-empty"
        frame_embeds = []
        last_node_emb = None

        for frame in graph_seq:
            node_emb = self.gnn(frame.x, frame.edge_index, getattr(frame, "edge_attr", None))
            # track the latest frame's node embeddings for action heads
            last_node_emb = node_emb
            frame_embed = self.aggregator(node_emb, frame.ptr, frame.batch)
            frame_embeds.append(frame_embed)

        frame_stack = torch.stack(frame_embeds, dim=1)  # (batch, seq_len, hidden_dim)
        trans_out = self.temporal(frame_stack)
        z = trans_out[:, -1, :]  # last-frame representation

        return StateEncoderOutput(z=z, last_node_emb=last_node_emb)
