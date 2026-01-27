from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from .config import DataConfig, StatsConfig, TrainConfig
from .data import generate_synthetic_transactions, iter_synthetic_transactions
from .stats_layer import run_statistical_filter, stream_user_priors_from_transactions, user_priors_from_transactions


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _stratified_split_indices(y: "np.ndarray", *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple stratified split for binary labels.

    Returns train/val/test indices.
    """

    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(arr)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        return arr[:n_train], arr[n_train : n_train + n_val], arr[n_train + n_val :]

    pos_tr, pos_va, pos_te = split(pos)
    neg_tr, neg_va, neg_te = split(neg)

    train = np.concatenate([pos_tr, neg_tr])
    val = np.concatenate([pos_va, neg_va])
    test = np.concatenate([pos_te, neg_te])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _pick_threshold(y_true: "np.ndarray", p: "np.ndarray") -> float:
    """Pick a threshold on validation to maximize F1 for the fraud class."""

    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p)

    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cc = _confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec1 = tp / max(tp + fp, 1)
    rec1 = tp / max(tp + fn, 1)
    f1 = (2 * prec1 * rec1 / max(prec1 + rec1, 1e-12))
    fpr = fp / max(fp + tn, 1)
    return {"accuracy": float(acc), "precision_fraud": float(prec1), "recall_fraud": float(rec1), "f1_fraud": float(f1), "fpr": float(fpr)}


def _apply_hard_mode(df, *, seed: int):
    """Make the synthetic problem harder and more realistic.

    Goals:
      - Add false-positive traps (benign shared hubs)
      - Add adversarial noise (fraud not always sharing the same IDs)
      - Keep code simple (no new dependencies)
    """

    rng = np.random.default_rng(seed)
    out = df.copy()

    # 1) Benign hubs: some legit traffic shares a few devices/IPs (e.g., corporate NAT / kiosks)
    legit_mask = out["label"].to_numpy() == 0
    legit_idx = np.where(legit_mask)[0]
    if legit_idx.size > 0:
        hub_devices = rng.choice(out["device_id"].unique(), size=max(2, int(0.01 * out["device_id"].nunique())), replace=False)
        hub_ips = rng.choice(out["ip_id"].unique(), size=max(2, int(0.01 * out["ip_id"].nunique())), replace=False)

        # Rewire 12% of legit transactions into hubs.
        k = int(0.12 * legit_idx.size)
        pick = rng.choice(legit_idx, size=max(1, k), replace=False)
        out.loc[pick, "device_id"] = rng.choice(hub_devices, size=len(pick), replace=True)
        out.loc[pick, "ip_id"] = rng.choice(hub_ips, size=len(pick), replace=True)

    # 2) Fraud camouflage: some fraud transactions use random IDs to reduce easy ring signatures.
    fraud_mask = out["label"].to_numpy() == 1
    fraud_idx = np.where(fraud_mask)[0]
    if fraud_idx.size > 0:
        k2 = int(0.20 * fraud_idx.size)
        pick2 = rng.choice(fraud_idx, size=max(1, k2), replace=False)
        out.loc[pick2, "device_id"] = rng.choice(out["device_id"].unique(), size=len(pick2), replace=True)
        out.loc[pick2, "ip_id"] = rng.choice(out["ip_id"].unique(), size=len(pick2), replace=True)
        out.loc[pick2, "phone_id"] = rng.choice(out["phone_id"].unique(), size=len(pick2), replace=True)

    return out


def _time_split_transactions(df, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transaction index split by time (train early, test later)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(df["ts"].to_numpy())
    n = len(order)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    tr = order[:n_train]
    va = order[n_train : n_train + n_val]
    te = order[n_train + n_val :]
    # Add tiny shuffle inside each block so batches are not too ordered.
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def run_once(
    *,
    seed: int,
    hard_mode: bool,
    time_split: bool,
    stream: bool = False,
    phase1_only: bool = False,
    n_transactions: int | None = None,
    n_users: int | None = None,
    n_devices: int | None = None,
    n_ips: int | None = None,
    n_phones: int | None = None,
    epochs: int | None = None,
) -> dict[str, float]:
    data_cfg = DataConfig()
    stats_cfg = StatsConfig()
    train_cfg = TrainConfig()

    n_users_eff = int(n_users) if n_users is not None else int(data_cfg.n_users)
    n_devices_eff = int(n_devices) if n_devices is not None else int(data_cfg.n_devices)
    n_ips_eff = int(n_ips) if n_ips is not None else int(data_cfg.n_ips)
    n_phones_eff = int(n_phones) if n_phones is not None else int(data_cfg.n_phones)
    n_tx_eff = int(n_transactions) if n_transactions is not None else int(data_cfg.n_transactions)

    # Auto-stream for huge transaction counts to avoid DataFrame blowups.
    use_stream = bool(stream) or (n_tx_eff >= 2_000_000)

    if use_stream:
        tx_iter = iter_synthetic_transactions(
            seed=seed,
            n_users=n_users_eff,
            n_devices=n_devices_eff,
            n_ips=n_ips_eff,
            n_phones=n_phones_eff,
            n_transactions=n_tx_eff,
            base_fraud_rate=data_cfg.base_fraud_rate,
            hard_mode=bool(hard_mode),
        )

        user_table, counts, _split_users = stream_user_priors_from_transactions(
            tx_iter,
            n_users=n_users_eff,
            amount_z_threshold=stats_cfg.amount_z_threshold,
            velocity_window_seconds=stats_cfg.velocity_window_seconds,
            velocity_threshold=stats_cfg.velocity_threshold,
            clear_fraud_score=stats_cfg.clear_fraud_score,
            clear_legit_score=stats_cfg.clear_legit_score,
            time_split=bool(time_split),
        )

        print("\n[Phase 1] Statistical filter decisions:")
        print(json.dumps(counts, indent=2))

        # Phase-1-only mode is the intended path for very large N.
        if phase1_only or n_tx_eff >= 2_000_000:
            y_true = user_table["user_label"].to_numpy(dtype=int)
            # Baseline: only CLEAR_FRAUD => fraud; GRAY treated as legit (fail-open).
            y_pred = (user_table["user_stats_decision"].to_numpy() == "CLEAR_FRAUD").astype(int)
            summary = _metrics_summary(y_true, y_pred)
            cc = _confusion_counts(y_true, y_pred)
            print("\n[Phase 1 Only] User-level metrics (CLEAR_FRAUD vs rest):")
            print(
                f"acc={summary['accuracy']:.3f} prec_fraud={summary['precision_fraud']:.3f} rec_fraud={summary['recall_fraud']:.3f} "
                f"f1_fraud={summary['f1_fraud']:.3f} fpr={summary['fpr']:.3f} tp={cc['tp']} fp={cc['fp']} tn={cc['tn']} fn={cc['fn']}"
            )
            return summary

        # If someone explicitly disables phase1_only, fall back to the small-data path below.
        # (Keeping code simple: full Phase 2/3 needs transaction-level tables.)

    df = generate_synthetic_transactions(
        seed=seed,
        n_users=n_users_eff,
        n_devices=n_devices_eff,
        n_ips=n_ips_eff,
        n_phones=n_phones_eff,
        n_transactions=n_tx_eff,
        base_fraud_rate=data_cfg.base_fraud_rate,
    )

    if hard_mode:
        df = _apply_hard_mode(df, seed=seed)

    df_stats = run_statistical_filter(
        df,
        amount_z_threshold=stats_cfg.amount_z_threshold,
        velocity_window_seconds=stats_cfg.velocity_window_seconds,
        velocity_threshold=stats_cfg.velocity_threshold,
        clear_fraud_score=stats_cfg.clear_fraud_score,
        clear_legit_score=stats_cfg.clear_legit_score,
    )

    counts = df_stats["stats_decision"].value_counts().to_dict()
    print("\n[Phase 1] Statistical filter decisions:")
    print(json.dumps(counts, indent=2))

    user_table = user_priors_from_transactions(df_stats)

    # User-level cascade decision derived from transaction decisions.
    # - CLEAR_FRAUD if any of the user's txns are CLEAR_FRAUD
    # - CLEAR_LEGIT if all are CLEAR_LEGIT
    # - otherwise GRAY
    user_dec = (
        df_stats.groupby("user_id")["stats_decision"]
        .apply(lambda s: "CLEAR_FRAUD" if (s == "CLEAR_FRAUD").any() else ("CLEAR_LEGIT" if (s == "CLEAR_LEGIT").all() else "GRAY"))
        .reset_index()
        .rename(columns={"stats_decision": "user_stats_decision"})
    )
    user_table = user_table.merge(user_dec, on="user_id", how="left")

    # Decide which users to include in the graph training set.
    # - If gray_only_training=True: train only on users in GRAY.
    # - Else: train on all users and report both model-only + cascade metrics.
    if train_cfg.gray_only_training:
        train_user_ids = set(user_table.loc[user_table["user_stats_decision"] == "GRAY", "user_id"].astype(int).tolist())
        user_table = user_table[user_table["user_id"].isin(train_user_ids)].reset_index(drop=True)
        df_for_graph = df_stats[df_stats["user_id"].isin(train_user_ids)].reset_index(drop=True)
    else:
        train_user_ids = set(user_table["user_id"].astype(int).tolist())
        df_for_graph = df_stats

    try:
        import torch

        # Prefer the torch-only backend for maximum install success.
        from .hetero_torch import build_hetero_graph_torch
        from .model_torch import TorchHeteroMPNN
        from .explain import explain_user_via_edges

        artifacts = build_hetero_graph_torch(df_for_graph, user_table)

        device = torch.device(train_cfg.device)

        x = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in artifacts.x.items()}
        y = torch.tensor(artifacts.y_user, dtype=torch.long, device=device)
        edges = {
            etype: (
                torch.tensor(src, dtype=torch.long, device=device),
                torch.tensor(dst, dtype=torch.long, device=device),
            )
            for etype, (src, dst) in artifacts.edges.items()
        }

        if time_split:
            # Time split at transaction level; map to user indices that appear.
            tr_tx, va_tx, te_tx = _time_split_transactions(df_for_graph, seed=seed)
            tr_users = set(df_for_graph.iloc[tr_tx]["user_id"].astype(int).tolist())
            va_users = set(df_for_graph.iloc[va_tx]["user_id"].astype(int).tolist())
            te_users = set(df_for_graph.iloc[te_tx]["user_id"].astype(int).tolist())

            # Make splits disjoint at user level (production-like: evaluate on unseen users).
            va_users = va_users - tr_users
            te_users = te_users - tr_users - va_users
            # Map raw user_id -> user_idx in graph.
            uid_to_idx = artifacts.node_maps["user"]
            tr = np.array([uid_to_idx[u] for u in tr_users if u in uid_to_idx], dtype=np.int64)
            va = np.array([uid_to_idx[u] for u in va_users if u in uid_to_idx], dtype=np.int64)
            te = np.array([uid_to_idx[u] for u in te_users if u in uid_to_idx], dtype=np.int64)
            # Fallback if split becomes too small.
            if len(tr) < 10 or len(va) < 5 or len(te) < 5:
                tr, va, te = _stratified_split_indices(artifacts.y_user, seed=seed)
        else:
            tr, va, te = _stratified_split_indices(artifacts.y_user, seed=seed)

        train_idx = torch.tensor(tr, dtype=torch.long, device=device)
        val_idx = torch.tensor(va, dtype=torch.long, device=device)
        test_idx = torch.tensor(te, dtype=torch.long, device=device)

        in_dims = {ntype: x[ntype].shape[1] for ntype in x.keys()}
        model = TorchHeteroMPNN(in_dims=in_dims, hidden_dim=train_cfg.hidden_dim, dropout=train_cfg.dropout).to(device)

        # Class weights for stability on imbalanced data.
        y_train = y[train_idx]
        n_pos = int((y_train == 1).sum().item())
        n_neg = int((y_train == 0).sum().item())
        w_pos = (n_neg / max(n_pos, 1))
        class_w = torch.tensor([1.0, float(w_pos)], device=device)

        opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

        print("\n[Phase 2] Training torch-only hetero MPNN on user nodes...")
        n_epochs = int(epochs) if epochs is not None else train_cfg.epochs
        for epoch in range(1, n_epochs + 1):
            model.train()
            logits = model(x=x, num_nodes=artifacts.num_nodes, edges=edges)
            loss = loss_fn(logits[train_idx], y[train_idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
                model.eval()
                with torch.no_grad():
                    val_logits = model(x=x, num_nodes=artifacts.num_nodes, edges=edges)
                    val_pred = val_logits[val_idx].argmax(dim=1)
                    val_acc = (val_pred == y[val_idx]).float().mean().item()
                print(f"epoch={epoch:02d} loss={loss.item():.4f} val_acc={val_acc:.3f}")

        model.eval()
        with torch.no_grad():
            logits = model(x=x, num_nodes=artifacts.num_nodes, edges=edges)
            probs = torch.softmax(logits, dim=1)[:, 1]
            val_thr = _pick_threshold(
                y[val_idx].detach().cpu().numpy(),
                probs[val_idx].detach().cpu().numpy(),
            )
            pred = (probs >= val_thr).long()

        print("\n[Test] User-level report (gray-subset if enabled):")
        print(
            classification_report(
                y[test_idx].cpu().numpy(),
                pred[test_idx].cpu().numpy(),
                digits=3,
                zero_division=0,
            )
        )
        print(f"Chosen validation threshold: {val_thr:.2f}")

        # Save artifacts
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ARTIFACTS_DIR / "torch_hetero_mpnn.pt")

        # Phase 3: explanation on strongest flagged user.
        flagged = torch.where((pred == 1) & (probs >= 0.7))[0].tolist()
        known_bad = set(torch.where(y == 1)[0].tolist())
        if flagged:
            target = int(flagged[0])
            print("\n[Phase 3] Example explanation:")
            print(f"Target user node index: {target}")
            path = explain_user_via_edges(edges=artifacts.edges, user_idx=target, known_bad_user_indices=known_bad, max_hops=4)
            print(path)
        else:
            print("\n[Phase 3] No strongly flagged users to explain in this run.")

        # Phase 2 -> Cascade: combine deterministic decisions with GNN output.
        if not train_cfg.gray_only_training:
            # Rebuild a full user table aligned to raw user ids.
            full_user_table = user_priors_from_transactions(df_stats)
            full_user_dec = (
                df_stats.groupby("user_id")["stats_decision"]
                .apply(lambda s: "CLEAR_FRAUD" if (s == "CLEAR_FRAUD").any() else ("CLEAR_LEGIT" if (s == "CLEAR_LEGIT").all() else "GRAY"))
                .reset_index()
                .rename(columns={"stats_decision": "user_stats_decision"})
            )
            full_user_table = full_user_table.merge(full_user_dec, on="user_id", how="left")

            # Map raw user_id -> user_idx in the current graph.
            user_id_to_idx = {raw: idx for raw, idx in artifacts.node_maps["user"].items()}
            full_user_table["user_idx"] = full_user_table["user_id"].map(user_id_to_idx)

            # For cascade: use stats for CLEAR users; use GNN for GRAY.
            test_set = set(test_idx.detach().cpu().numpy().tolist())

            cascade_pred = []
            cascade_y = []
            for _, row in full_user_table.dropna(subset=["user_idx"]).iterrows():
                ui = int(row["user_idx"])
                if ui not in test_set:
                    continue
                decision = str(row["user_stats_decision"])
                true_y = int(row["user_label"])

                if decision == "CLEAR_FRAUD":
                    p = 1
                elif decision == "CLEAR_LEGIT":
                    p = 0
                else:
                    p = int((probs[ui] >= val_thr).item())

                cascade_pred.append(p)
                cascade_y.append(true_y)

            print("\n[Cascade] Combined stats + GNN report (user-level, test split):")
            print(classification_report(np.array(cascade_y), np.array(cascade_pred), digits=3, zero_division=0))

            summary = _metrics_summary(np.array(cascade_y), np.array(cascade_pred))
            cc = _confusion_counts(np.array(cascade_y), np.array(cascade_pred))
            print(f"[Cascade Metrics] acc={summary['accuracy']:.3f} prec_fraud={summary['precision_fraud']:.3f} rec_fraud={summary['recall_fraud']:.3f} f1_fraud={summary['f1_fraud']:.3f} fpr={summary['fpr']:.3f} tp={cc['tp']} fp={cc['fp']} tn={cc['tn']} fn={cc['fn']}")
            return summary

        # If cascade isn't computed, return model-only metrics on test.
        y_test = y[test_idx].detach().cpu().numpy()
        pred_test = pred[test_idx].detach().cpu().numpy()
        summary = _metrics_summary(y_test, pred_test)
        return summary

    except ModuleNotFoundError as e:
        print("\n[Phase 2] Skipping GNN training because a dependency is missing:")
        print(str(e))
        print("Install PyTorch + DGL to enable Phase 2 and 3.")
        return {"accuracy": float("nan"), "precision_fraud": float("nan"), "recall_fraud": float("nan"), "f1_fraud": float("nan"), "fpr": float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid-Recursive Fraud Defense demo runner")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds (default: 1)")
    parser.add_argument("--seed", type=int, default=7, help="Base seed (default: 7)")
    parser.add_argument("--hard", action="store_true", help="Enable harder synthetic mode (benign hubs + fraud camouflage)")
    parser.add_argument("--time-split", action="store_true", help="Use a time-based split (rough production-like check)")
    parser.add_argument("--stream", action="store_true", help="Stream transactions (enables huge --n-transactions without DataFrame)")
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1 (recommended for very large --n-transactions)")
    parser.add_argument("--n-transactions", type=int, default=None, help="Override number of transactions (cases)")
    parser.add_argument("--n-users", type=int, default=None, help="Override number of users")
    parser.add_argument("--n-devices", type=int, default=None, help="Override number of devices")
    parser.add_argument("--n-ips", type=int, default=None, help="Override number of IPs")
    parser.add_argument("--n-phones", type=int, default=None, help="Override number of phones")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    summaries: list[dict[str, float]] = []
    for i in range(int(args.runs)):
        run_seed = int(args.seed) + i
        print(f"\n================ RUN {i+1}/{int(args.runs)} seed={run_seed} hard={bool(args.hard)} time_split={bool(args.time_split)} ================")
        summaries.append(
            run_once(
                seed=run_seed,
                hard_mode=bool(args.hard),
                time_split=bool(args.time_split),
                stream=bool(args.stream),
                phase1_only=bool(args.phase1_only),
                n_transactions=args.n_transactions,
                n_users=args.n_users,
                n_devices=args.n_devices,
                n_ips=args.n_ips,
                n_phones=args.n_phones,
                epochs=args.epochs,
            )
        )

    if len(summaries) > 1:
        keys = ["accuracy", "precision_fraud", "recall_fraud", "f1_fraud", "fpr"]
        arr = {k: np.array([s[k] for s in summaries], dtype=float) for k in keys}
        print("\n[Multi-run Summary] mean ± std")
        for k in keys:
            vals = arr[k]
            print(f"{k}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")


if __name__ == "__main__":
    main()
