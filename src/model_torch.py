from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum `src` rows into `out[index]` using only core PyTorch."""
    out = torch.zeros((dim_size, src.shape[1]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class TorchHeteroMPNN(nn.Module):
    """A minimal heterogeneous message passing network (PyTorch-only).

    Design goals:
      - No external graph libs (no DGL/PyG)
      - Readable and stable on CPU
      - Good enough for a demo of relational fraud rings

    It runs message passing over a small set of typed relations and outputs
    user-level fraud logits.
    """

    def __init__(self, *, in_dims: dict[str, int], hidden_dim: int, dropout: float):
        super().__init__()

        self.dropout = float(dropout)

        # Project each node type into a shared hidden space.
        self.proj = nn.ModuleDict({ntype: nn.Linear(d, hidden_dim) for ntype, d in in_dims.items()})

        # Relation-specific transforms (same hidden size on all types).
        # Keys match canonical edge types: (src_type, rel, dst_type)
        self.rel_linear = nn.ModuleDict(
            {
                "user:uses_device:device": nn.Linear(hidden_dim, hidden_dim),
                "device:used_by:user": nn.Linear(hidden_dim, hidden_dim),
                "user:uses_ip:ip": nn.Linear(hidden_dim, hidden_dim),
                "ip:used_by:user": nn.Linear(hidden_dim, hidden_dim),
                "user:uses_phone:phone": nn.Linear(hidden_dim, hidden_dim),
                "phone:used_by:user": nn.Linear(hidden_dim, hidden_dim),
            }
        )

        # Node-type update (post-aggregation).
        self.update = nn.ModuleDict({ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in in_dims.keys()})

        self.user_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _ekey(src_type: str, rel: str, dst_type: str) -> str:
        return f"{src_type}:{rel}:{dst_type}"

    def forward(
        self,
        *,
        x: dict[str, torch.Tensor],
        num_nodes: dict[str, int],
        edges: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Return logits for user nodes (shape: [n_user, 2])."""

        h = {}
        for ntype, feat in x.items():
            z = self.proj[ntype](feat)
            z = F.relu(z)
            h[ntype] = F.dropout(z, p=self.dropout, training=self.training)

        # Two message-passing steps.
        for _ in range(2):
            agg: dict[str, torch.Tensor] = {ntype: torch.zeros((num_nodes[ntype], h[ntype].shape[1]), device=h[ntype].device) for ntype in h.keys()}

            for (src_type, rel, dst_type), (src_idx, dst_idx) in edges.items():
                if src_idx.numel() == 0:
                    continue
                msg = self.rel_linear[self._ekey(src_type, rel, dst_type)](h[src_type][src_idx])
                msg = F.relu(msg)
                agg[dst_type] = agg[dst_type] + _scatter_add(msg, dst_idx, dim_size=num_nodes[dst_type])

            # Update each node type.
            for ntype in h.keys():
                upd = self.update[ntype](h[ntype] + agg[ntype])
                upd = F.relu(upd)
                h[ntype] = F.dropout(upd, p=self.dropout, training=self.training)

        return self.user_head(h["user"])
