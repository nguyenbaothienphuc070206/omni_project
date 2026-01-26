from __future__ import annotations

from dataclasses import dataclass

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
