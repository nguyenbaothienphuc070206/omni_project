from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphArtifacts:
    graph: object  # DGLHeteroGraph
    node_maps: dict[str, dict[int, int]]
    user_table: pd.DataFrame


def _make_id_map(values: pd.Series) -> dict[int, int]:
    uniq = sorted(set(int(v) for v in values.tolist()))
    return {raw: i for i, raw in enumerate(uniq)}


def build_hetero_graph_dgl(df: pd.DataFrame, user_table: pd.DataFrame) -> GraphArtifacts:
    """Build a DGL heterograph from transactions.

    Node types: user, device, ip, phone
    Relations:
      - (user, uses_device, device) and reverse
      - (user, uses_ip, ip) and reverse
      - (user, uses_phone, phone) and reverse

    Features:
      - user: [prior_anomaly_score, tx_count, avg_amount, max_velocity]
      - others: [degree]
    """

    import dgl
    import torch

    user_map = _make_id_map(df["user_id"])
    device_map = _make_id_map(df["device_id"])
    ip_map = _make_id_map(df["ip_id"])
    phone_map = _make_id_map(df["phone_id"])

    u = df["user_id"].map(user_map).to_numpy()
    d = df["device_id"].map(device_map).to_numpy()
    ip = df["ip_id"].map(ip_map).to_numpy()
    ph = df["phone_id"].map(phone_map).to_numpy()

    data_dict = {
        ("user", "uses_device", "device"): (u, d),
        ("device", "used_by", "user"): (d, u),
        ("user", "uses_ip", "ip"): (u, ip),
        ("ip", "used_by", "user"): (ip, u),
        ("user", "uses_phone", "phone"): (u, ph),
        ("phone", "used_by", "user"): (ph, u),
    }

    g = dgl.heterograph(data_dict)

    # User features table aligned to user_map order
    user_feats = user_table.copy()
    user_feats = user_feats[user_feats["user_id"].isin(user_map.keys())].copy()
    user_feats["user_idx"] = user_feats["user_id"].map(user_map)
    user_feats = user_feats.sort_values("user_idx")

    x_user = user_feats[["prior_anomaly_score", "tx_count", "avg_amount", "max_velocity"]].to_numpy(dtype=np.float32)
    x_user[:, 1] = np.log1p(x_user[:, 1])
    x_user[:, 2] = np.log1p(x_user[:, 2])
    x_user[:, 3] = np.log1p(x_user[:, 3])

    g.nodes["user"].data["x"] = torch.tensor(x_user, dtype=torch.float32)
    g.nodes["user"].data["y"] = torch.tensor(user_feats["user_label"].to_numpy(dtype=np.int64))

    # Simple degree features for other node types
    for ntype in ["device", "ip", "phone"]:
        deg = g.in_degrees(etype=(ntype, "used_by", "user")).to(torch.float32).unsqueeze(1)
        g.nodes[ntype].data["x"] = deg

    return GraphArtifacts(
        graph=g,
        node_maps={"user": user_map, "device": device_map, "ip": ip_map, "phone": phone_map},
        user_table=user_feats,
    )
