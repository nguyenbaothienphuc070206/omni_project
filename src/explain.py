from __future__ import annotations

from collections import deque

import networkx as nx


def _node_key(ntype: str, idx: int) -> str:
    return f"{ntype}:{idx}"


def explain_user_via_edges(
    *,
    edges: dict[tuple[str, str, str], tuple[list[int] | "object", list[int] | "object"]],
    user_idx: int,
    known_bad_user_indices: set[int],
    max_hops: int = 4,
) -> str:
    """Backend-agnostic business-logic explanation.

    Input `edges` is a dict mapping canonical edge types to (src_idx, dst_idx).
    Indices can be Python lists, numpy arrays, or torch tensors.
    """

    G = nx.DiGraph()

    for (src_type, rel, dst_type), (src, dst) in edges.items():
        # Support lists / numpy / torch tensors.
        src_list = src.tolist() if hasattr(src, "tolist") else list(src)
        dst_list = dst.tolist() if hasattr(dst, "tolist") else list(dst)
        for s, d in zip(src_list, dst_list):
            G.add_edge(_node_key(src_type, int(s)), _node_key(dst_type, int(d)), rel=rel)

    start = _node_key("user", int(user_idx))
    targets = {_node_key("user", int(u)) for u in known_bad_user_indices}

    q = deque([(start, [start])])
    seen = {start}

    while q:
        node, path = q.popleft()
        if node in targets and node != start:
            return _format_path(G, path)
        if len(path) - 1 >= max_hops:
            continue
        for nxt in G.successors(node):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [nxt]))

    return "No short relational path found (increase max_hops or change anchors)."


def explain_user_via_shortest_path(
    *,
    g,
    user_idx: int,
    known_bad_user_indices: set[int],
    max_hops: int = 4,
) -> str:
    """Compatibility wrapper for DGL graphs used by the optional DGL backend."""

    edges = {}
    for etype in g.canonical_etypes:
        src_type, rel, dst_type = etype
        src, dst = g.edges(etype=etype)
        edges[(src_type, rel, dst_type)] = (src, dst)

    return explain_user_via_edges(
        edges=edges,
        user_idx=user_idx,
        known_bad_user_indices=known_bad_user_indices,
        max_hops=max_hops,
    )


def _format_path(G: nx.DiGraph, path: list[str]) -> str:
    parts = [path[0]]
    for a, b in zip(path, path[1:]):
        rel = G.edges[a, b].get("rel", "rel")
        parts.append(f"-({rel})->")
        parts.append(b)
    return " ".join(parts)
