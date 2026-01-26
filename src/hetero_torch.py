from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TorchHeteroGraph:
    """A minimal hetero-graph representation usable without DGL/PyG.

    - Node types: user, device, ip, phone
    - Edge list per relation as integer index arrays

    This is intentionally tiny and readable.
    """

    num_nodes: dict[str, int]
    node_maps: dict[str, dict[int, int]]
    x: dict[str, "np.ndarray"]
    y_user: "np.ndarray"

    # Each edge is stored as (src_idx, dst_idx) arrays.
    edges: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]


def _make_id_map(values: pd.Series) -> dict[int, int]:
    uniq = sorted({int(v) for v in values.tolist()})
    return {raw: i for i, raw in enumerate(uniq)}


def build_hetero_graph_torch(df: pd.DataFrame, user_table: pd.DataFrame) -> TorchHeteroGraph:
    """Build a hetero graph from transactions using only numpy.

    Relations:
      - (user, uses_device, device)
      - (user, uses_ip, ip)
      - (user, uses_phone, phone)
      - plus reverse edges for message passing

    Features:
      - user: [prior_anomaly_score, log1p(tx_count), log1p(avg_amount), log1p(max_velocity)]
      - others: [log1p(degree)]
    """

    user_map = _make_id_map(df["user_id"])
    device_map = _make_id_map(df["device_id"])
    ip_map = _make_id_map(df["ip_id"])
    phone_map = _make_id_map(df["phone_id"])

    u = df["user_id"].map(user_map).to_numpy(dtype=np.int64)
    d = df["device_id"].map(device_map).to_numpy(dtype=np.int64)
    ip = df["ip_id"].map(ip_map).to_numpy(dtype=np.int64)
    ph = df["phone_id"].map(phone_map).to_numpy(dtype=np.int64)

    edges: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {
        ("user", "uses_device", "device"): (u, d),
        ("device", "used_by", "user"): (d, u),
        ("user", "uses_ip", "ip"): (u, ip),
        ("ip", "used_by", "user"): (ip, u),
        ("user", "uses_phone", "phone"): (u, ph),
        ("phone", "used_by", "user"): (ph, u),
    }

    # Align user_table to the user_map order
    user_feats = user_table.copy()
    user_feats = user_feats[user_feats["user_id"].isin(user_map.keys())].copy()
    user_feats["user_idx"] = user_feats["user_id"].map(user_map)
    user_feats = user_feats.sort_values("user_idx")

    x_user = user_feats[["prior_anomaly_score", "tx_count", "avg_amount", "max_velocity"]].to_numpy(dtype=np.float32)
    x_user[:, 1] = np.log1p(x_user[:, 1])
    x_user[:, 2] = np.log1p(x_user[:, 2])
    x_user[:, 3] = np.log1p(x_user[:, 3])

    y_user = user_feats["user_label"].to_numpy(dtype=np.int64)

    x: dict[str, np.ndarray] = {"user": x_user}

    # Degree features for non-user nodes
    type_sizes = {"device": len(device_map), "ip": len(ip_map), "phone": len(phone_map)}
    for ntype, rel in [
        ("device", ("device", "used_by", "user")),
        ("ip", ("ip", "used_by", "user")),
        ("phone", ("phone", "used_by", "user")),
    ]:
        src_idx, _ = edges[rel]
        deg = np.bincount(src_idx, minlength=type_sizes[ntype]).astype(np.float32)
        x[ntype] = np.log1p(deg).reshape(-1, 1)

    return TorchHeteroGraph(
        num_nodes={"user": len(user_map), "device": len(device_map), "ip": len(ip_map), "phone": len(phone_map)},
        node_maps={"user": user_map, "device": device_map, "ip": ip_map, "phone": phone_map},
        x=x,
        y_user=y_user,
        edges=edges,
    )
