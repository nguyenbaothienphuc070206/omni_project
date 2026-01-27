from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CreditDataConfig:
    n_borrowers: int = 8000
    underbanked_rate: float = 0.35


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_credit_data(
    *,
    seed: int,
    n_borrowers: int,
    underbanked_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Synthetic creditworthiness dataset.

    Design goal: demonstrate why relying only on traditional fields biases
    underbanked borrowers (missing credit history), while alternative behavior
    features help.

    Returns: X, y_default, is_underbanked, feature_names
    """

    rng = np.random.default_rng(seed)

    # Traditional-ish fields (some missing for underbanked)
    income = rng.lognormal(mean=np.log(650), sigma=0.55, size=n_borrowers)  # monthly income (USD)
    credit_hist_months = rng.integers(0, 240, size=n_borrowers).astype(float)
    debt_to_income = np.clip(rng.normal(loc=0.35, scale=0.18, size=n_borrowers), 0.0, 1.5)

    # Alternative behavior fields (available for everyone)
    tx_volume_30d = rng.poisson(lam=35, size=n_borrowers).astype(float)
    avg_balance_30d = rng.lognormal(mean=np.log(350), sigma=0.8, size=n_borrowers)
    balance_volatility = np.clip(rng.normal(loc=0.28, scale=0.16, size=n_borrowers), 0.0, 1.0)
    merchant_diversity = np.clip(rng.normal(loc=0.45, scale=0.2, size=n_borrowers), 0.0, 1.0)
    late_payment_signals = rng.poisson(lam=0.25, size=n_borrowers).astype(float)

    is_underbanked = (rng.random(n_borrowers) < underbanked_rate).astype(int)

    # Underbanked: often have thin-file credit history (missing/zero).
    # We mark it as NaN to simulate unavailable bureau data.
    credit_hist_months[is_underbanked == 1] = np.nan

    # Ground-truth default risk model (simple + interpretable):
    # - More late signals, higher DTI, higher volatility increase risk
    # - More balance and higher income reduce risk
    # - Longer credit history reduces risk (when known)
    # Add noise so it's not perfectly separable.
    credit_hist_known = np.nan_to_num(credit_hist_months, nan=0.0)

    z_income = (np.log(income) - np.log(650.0))
    z_balance = (np.log(avg_balance_30d + 1.0) - np.log(350.0))

    logit = (
        0.9 * debt_to_income
        + 1.2 * balance_volatility
        + 0.8 * (late_payment_signals > 0).astype(float)
        + 0.4 * late_payment_signals
        - 0.55 * z_income
        - 0.55 * z_balance
        - 0.004 * credit_hist_known
        - 0.08 * merchant_diversity
        - 0.01 * tx_volume_30d
        + rng.normal(0.0, 0.35, size=n_borrowers)
    )

    p_default = _sigmoid(logit)
    y_default = (rng.random(n_borrowers) < p_default).astype(int)

    feature_names = [
        "income_monthly",
        "credit_history_months",
        "debt_to_income",
        "tx_volume_30d",
        "avg_balance_30d",
        "balance_volatility",
        "merchant_diversity",
        "late_payment_signals",
        "is_underbanked",
    ]

    X = np.column_stack(
        [
            income,
            credit_hist_months,
            debt_to_income,
            tx_volume_30d,
            avg_balance_30d,
            balance_volatility,
            merchant_diversity,
            late_payment_signals,
            is_underbanked.astype(float),
        ]
    ).astype(float)

    return X, y_default, is_underbanked, feature_names


def _split_indices(n: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cc = _confusion(y_true, y_pred)
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "fpr": float(fpr)}


def _find_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (p >= thr).astype(int)
        f1 = _metrics(y_true, pred)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _group_report(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, *, group_name: str) -> None:
    for g in [0, 1]:
        mask = group.astype(int) == g
        m = _metrics(y_true[mask], y_pred[mask])
        label = "underbanked" if g == 1 else "banked"
        print(
            f"[{group_name}:{label}] acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f} fpr={m['fpr']:.3f} n={int(mask.sum())}"
        )


def _top_coefficients(coef: np.ndarray, feature_names: list[str], *, k: int = 5) -> None:
    coef = coef.reshape(-1)
    order_pos = np.argsort(-coef)
    order_neg = np.argsort(coef)

    print("\n[Explain] Top risk-increasing features (positive weights):")
    for i in order_pos[:k]:
        print(f"  + {feature_names[int(i)]}: {coef[int(i)]:.3f}")

    print("[Explain] Top risk-reducing features (negative weights):")
    for i in order_neg[:k]:
        print(f"  - {feature_names[int(i)]}: {coef[int(i)]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit scoring + creditworthiness demo (simple + scalable baseline)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=CreditDataConfig.n_borrowers, help="Number of borrowers")
    parser.add_argument("--underbanked-rate", type=float, default=CreditDataConfig.underbanked_rate)
    parser.add_argument("--show-explain", action="store_true", help="Print top model coefficients")
    args = parser.parse_args()

    # Lazy import heavy deps only when needed.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, group, feature_names = generate_synthetic_credit_data(
        seed=int(args.seed),
        n_borrowers=int(args.n),
        underbanked_rate=float(args.underbanked_rate),
    )

    tr, va, te = _split_indices(len(y), seed=int(args.seed))

    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X[tr], y[tr])

    # Threshold tuned on validation for default-class F1.
    p_val = model.predict_proba(X[va])[:, 1]
    thr = _find_threshold(y[va], p_val)

    p_test = model.predict_proba(X[te])[:, 1]
    pred_test = (p_test >= thr).astype(int)

    m = _metrics(y[te], pred_test)
    cc = _confusion(y[te], pred_test)

    print(f"\n[Credit] n={len(y)} underbanked_rate={float(group.mean()):.2f}")
    print(f"[Credit] chosen_threshold={thr:.2f} (tuned on val)")
    print(
        f"[Credit] test acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f} fpr={m['fpr']:.3f} tp={cc['tp']} fp={cc['fp']} tn={cc['tn']} fn={cc['fn']}"
    )

    print("\n[Credit] subgroup metrics:")
    _group_report(y[te], pred_test, group[te], group_name="group")

    if bool(args.show_explain):
        coef = model.named_steps["clf"].coef_[0]
        _top_coefficients(coef, feature_names, k=5)


if __name__ == "__main__":
    main()
