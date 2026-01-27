from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatsDecision:
    decision: str  # CLEAR_LEGIT | CLEAR_FRAUD | GRAY
    anomaly_score: float  # 0..1
    reasons: tuple[str, ...]


def _safe_zscore(series: pd.Series) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def run_statistical_filter(
    df: pd.DataFrame,
    *,
    amount_z_threshold: float,
    velocity_window_seconds: int,
    velocity_threshold: int,
    clear_fraud_score: float,
    clear_legit_score: float,
) -> pd.DataFrame:
    """Fast, interpretable layer.

    Returns a copy of df with:
      - amount_z
      - velocity
      - anomaly_score (0..1)
      - stats_decision (CLEAR_LEGIT/CLEAR_FRAUD/GRAY)
      - stats_reasons (string)

    Notes:
      - Velocity is computed per-user as max number of txns inside a sliding window.
      - Anomaly score is a simple combination of z-score and velocity.
    """

    out = df.copy()

    out["amount_z"] = _safe_zscore(out["amount"])

    out = out.sort_values(["user_id", "ts"]).reset_index(drop=True)

    velocities = np.zeros(len(out), dtype=np.int32)
    for user_id, grp in out.groupby("user_id", sort=False):
        ts = grp["ts"].to_numpy()
        left = 0
        max_in_window = 1
        for right in range(len(ts)):
            while ts[right] - ts[left] > velocity_window_seconds:
                left += 1
            max_in_window = max(max_in_window, right - left + 1)
        velocities[grp.index.to_numpy()] = max_in_window

    out["velocity"] = velocities.astype(float)

    # Normalize components into [0,1]
    z_component = np.clip(np.abs(out["amount_z"].to_numpy()) / max(amount_z_threshold, 1e-6), 0.0, 2.0) / 2.0
    v_component = np.clip(out["velocity"].to_numpy() / max(float(velocity_threshold), 1.0), 0.0, 2.0) / 2.0

    anomaly = np.clip(0.55 * z_component + 0.45 * v_component, 0.0, 1.0)
    out["anomaly_score"] = anomaly

    decisions = []
    reasons_list = []
    for z, v, score in zip(out["amount_z"].to_numpy(), out["velocity"].to_numpy(), out["anomaly_score"].to_numpy()):
        reasons = []
        if abs(z) >= amount_z_threshold:
            reasons.append(f"amount_z={z:.2f}")
        if v >= velocity_threshold:
            reasons.append(f"velocity={int(v)}")

        if score >= clear_fraud_score:
            decisions.append("CLEAR_FRAUD")
        elif score <= clear_legit_score:
            decisions.append("CLEAR_LEGIT")
        else:
            decisions.append("GRAY")

        reasons_list.append(";".join(reasons) if reasons else "none")

    out["stats_decision"] = decisions
    out["stats_reasons"] = reasons_list

    return out


def user_priors_from_transactions(df_with_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level anomaly scores into user-level priors."""

    agg = (
        df_with_stats.groupby("user_id")
        .agg(
            prior_anomaly_score=("anomaly_score", "mean"),
            tx_count=("transaction_id", "count"),
            avg_amount=("amount", "mean"),
            max_velocity=("velocity", "max"),
            fraud_rate=("label", "mean"),
        )
        .reset_index()
    )

    # A more realistic user label than "any fraud ever": majority fraud transactions.
    agg["user_label"] = (agg["fraud_rate"] >= 0.5).astype(int)
    return agg


@dataclass
class _RunningMeanStd:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        var = self.m2 / self.n
        return float(np.sqrt(max(var, 0.0)))


def stream_user_priors_from_transactions(
    tx_iter,
    *,
    n_users: int,
    amount_z_threshold: float,
    velocity_window_seconds: int,
    velocity_threshold: int,
    clear_fraud_score: float,
    clear_legit_score: float,
    time_split: bool = False,
    max_transactions: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int], tuple[set[int], set[int], set[int]] | None]:
    """Streaming equivalent of Phase-1 + user priors.

    - Does NOT build a giant transaction DataFrame.
    - Keeps only O(n_users) state (+ small per-user deques for velocity).

    Returns:
      - user_table (DataFrame)
      - counts_tx_decisions (dict)
      - optional (train_users, val_users, test_users) if time_split=True
    """

    n_users = int(n_users)
    tx_count = np.zeros(n_users, dtype=np.int64)
    sum_amount = np.zeros(n_users, dtype=np.float64)
    sum_anomaly = np.zeros(n_users, dtype=np.float64)
    max_velocity = np.zeros(n_users, dtype=np.int32)
    fraud_sum = np.zeros(n_users, dtype=np.int64)

    any_clear_fraud = np.zeros(n_users, dtype=bool)
    all_clear_legit = np.ones(n_users, dtype=bool)

    windows = [deque() for _ in range(n_users)]
    rms = _RunningMeanStd()
    counts: dict[str, int] = {"CLEAR_LEGIT": 0, "CLEAR_FRAUD": 0, "GRAY": 0}

    split_users: tuple[set[int], set[int], set[int]] | None = None
    if time_split:
        split_users = (set(), set(), set())

    # If time_split is enabled, we define splits by transaction index (since stream is time-ordered).
    # This stays streaming-friendly and avoids storing timestamps.
    # train: first 70%, val: next 15%, test: last 15%
    tx_seen = 0
    n_total = None
    try:
        n_total = int(getattr(tx_iter, "n_transactions"))  # optional
    except Exception:
        n_total = None

    cap = int(max(8, 2 * int(velocity_threshold)))

    for tx_id, ts, user_id, device_id, ip_id, phone_id, amount, label in tx_iter:
        if max_transactions is not None and tx_seen >= int(max_transactions):
            break
        u = int(user_id)
        if u < 0 or u >= n_users:
            continue

        # Update running mean/std and compute online z-score.
        a = float(amount)
        rms.update(a)
        std = rms.std
        z = 0.0 if std == 0.0 else (a - rms.mean) / std

        # Velocity window (per user) using streaming timestamps.
        dq = windows[u]
        t = int(ts)
        dq.append(t)
        while dq and (t - dq[0] > int(velocity_window_seconds)):
            dq.popleft()
        # We clip velocity contribution anyway, so cap the queue length for speed.
        while len(dq) > cap:
            dq.popleft()
        v = len(dq)
        if v > max_velocity[u]:
            max_velocity[u] = v

        # Normalize components into [0,1].
        z_component = min(abs(z) / max(float(amount_z_threshold), 1e-6), 2.0) / 2.0
        v_component = min(float(v) / max(float(velocity_threshold), 1.0), 2.0) / 2.0
        anomaly = min(0.55 * z_component + 0.45 * v_component, 1.0)

        if anomaly >= float(clear_fraud_score):
            decision = "CLEAR_FRAUD"
        elif anomaly <= float(clear_legit_score):
            decision = "CLEAR_LEGIT"
        else:
            decision = "GRAY"

        counts[decision] = counts.get(decision, 0) + 1

        tx_count[u] += 1
        sum_amount[u] += a
        sum_anomaly[u] += float(anomaly)
        fraud_sum[u] += int(label)
        if decision == "CLEAR_FRAUD":
            any_clear_fraud[u] = True
        if decision != "CLEAR_LEGIT":
            all_clear_legit[u] = False

        tx_seen += 1
        if split_users is not None:
            # Use n_total if known; otherwise approximate by running thresholds (still stable for huge N).
            if n_total is None:
                # Simple 70/15/15 based on running index modulo 100.
                m = tx_seen % 100
                if m < 70:
                    split_users[0].add(u)
                elif m < 85:
                    split_users[1].add(u)
                else:
                    split_users[2].add(u)
            else:
                r = tx_seen / max(n_total, 1)
                if r <= 0.70:
                    split_users[0].add(u)
                elif r <= 0.85:
                    split_users[1].add(u)
                else:
                    split_users[2].add(u)

    seen = tx_count > 0
    user_ids = np.arange(n_users, dtype=np.int64)[seen]
    txc = tx_count[seen]
    avg_amount = (sum_amount[seen] / np.maximum(txc, 1)).astype(np.float64)
    prior_anomaly = (sum_anomaly[seen] / np.maximum(txc, 1)).astype(np.float64)
    fraud_rate = (fraud_sum[seen] / np.maximum(txc, 1)).astype(np.float64)

    user_label = (fraud_rate >= 0.5).astype(np.int64)
    user_stats_decision = np.full(user_ids.shape[0], "GRAY", dtype=object)
    any_f = any_clear_fraud[seen]
    all_l = all_clear_legit[seen]
    user_stats_decision[any_f] = "CLEAR_FRAUD"
    user_stats_decision[(~any_f) & (all_l)] = "CLEAR_LEGIT"

    user_table = pd.DataFrame(
        {
            "user_id": user_ids,
            "prior_anomaly_score": prior_anomaly,
            "tx_count": txc,
            "avg_amount": avg_amount,
            "max_velocity": max_velocity[seen].astype(np.int64),
            "fraud_rate": fraud_rate,
            "user_label": user_label,
            "user_stats_decision": user_stats_decision,
        }
    )

    return user_table, counts, split_users
