from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGAT(nn.Module):
    def __init__(self, *, in_dims: dict[str, int], hidden_dim: int, heads: int, dropout: float):
        super().__init__()

        import dgl.nn as dglnn

        self.dropout = dropout

        # Project each node type to the same hidden dim.
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })

        def make_gat():
            return dglnn.GATConv(
                in_feats=hidden_dim,
                out_feats=hidden_dim // heads,
                num_heads=heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
                allow_zero_in_degree=True,
            )

        self.conv1 = dglnn.HeteroGraphConv(
            {
                "uses_device": make_gat(),
                "used_by": make_gat(),
                "uses_ip": make_gat(),
                "uses_phone": make_gat(),
            },
            aggregate="mean",
        )

        self.conv2 = dglnn.HeteroGraphConv(
            {
                "uses_device": make_gat(),
                "used_by": make_gat(),
                "uses_ip": make_gat(),
                "uses_phone": make_gat(),
            },
            aggregate="mean",
        )

        self.user_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, g, x_dict: dict[str, torch.Tensor]):
        h = {}
        for ntype, x in x_dict.items():
            h[ntype] = F.dropout(self.proj[ntype](x), p=self.dropout, training=self.training)

        h = self._apply_conv(g, self.conv1, h)
        h = self._apply_conv(g, self.conv2, h)

        logits = self.user_head(h["user"])
        return logits

    def _apply_conv(self, g, conv, h):
        out = conv(g, h)
        merged = {}
        for ntype, v in out.items():
            # GATConv returns [N, heads, out_dim]; flatten to [N, hidden_dim]
            merged[ntype] = v.flatten(1)
        return merged
