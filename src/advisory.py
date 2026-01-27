from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdvisoryConfig:
    n_customers: int = 2000


def generate_customer_profiles(*, seed: int, n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, size=n)
    income_monthly = rng.lognormal(mean=np.log(750), sigma=0.6, size=n)
    savings = rng.lognormal(mean=np.log(1200), sigma=1.0, size=n)
    debt = rng.lognormal(mean=np.log(800), sigma=1.0, size=n)

    risk_tolerance = rng.integers(1, 6, size=n)  # 1 (low) .. 5 (high)
    horizon_years = rng.integers(1, 31, size=n)

    goal_amount = rng.lognormal(mean=np.log(8000), sigma=0.8, size=n)
    goal_years = np.minimum(horizon_years, rng.integers(1, 21, size=n))

    emergency_months = rng.integers(1, 7, size=n)

    return {
        "age": age.astype(int),
        "income_monthly": income_monthly.astype(float),
        "savings": savings.astype(float),
        "debt": debt.astype(float),
        "risk_tolerance": risk_tolerance.astype(int),
        "horizon_years": horizon_years.astype(int),
        "goal_amount": goal_amount.astype(float),
        "goal_years": goal_years.astype(int),
        "emergency_months": emergency_months.astype(int),
    }


def risk_score(*, risk_tolerance: int, horizon_years: int, debt_to_income: float, market_stress: float) -> float:
    """Return risk score in [0,1].

    Senior reasoning:
    - Risk tolerance and horizon dominate.
    - Debt pressure reduces risk capacity.
    - Market stress reduces recommended risk temporarily.
    """

    tol = (risk_tolerance - 1) / 4.0
    horizon = np.clip(horizon_years / 20.0, 0.0, 1.0)
    debt_penalty = np.clip(debt_to_income / 1.2, 0.0, 1.0)

    base = 0.55 * tol + 0.45 * horizon
    adjusted = base * (1.0 - 0.35 * market_stress) * (1.0 - 0.45 * debt_penalty)
    return float(np.clip(adjusted, 0.0, 1.0))


def allocation_from_risk(score: float) -> dict[str, float]:
    """Simple 3-bucket allocation (cash/bonds/equity)."""

    equity = 0.15 + 0.75 * score
    bonds = 0.70 - 0.50 * score
    cash = 1.0 - equity - bonds

    # Clamp + renormalize to keep it clean.
    w = np.array([cash, bonds, equity])
    w = np.clip(w, 0.0, 1.0)
    w = w / max(w.sum(), 1e-12)
    return {"cash": float(w[0]), "bonds": float(w[1]), "equity": float(w[2])}


def monthly_savings_needed(*, goal_amount: float, current_savings: float, years: int) -> float:
    """Goal-based plan: required monthly contribution (no return assumption).

    We keep it intentionally simple and conservative for a demo.
    """

    months = max(int(years) * 12, 1)
    remaining = max(goal_amount - current_savings, 0.0)
    return float(remaining / months)


def advise_one(profile: dict[str, float | int], *, market_stress: float) -> dict[str, object]:
    income = float(profile["income_monthly"])
    savings = float(profile["savings"])
    debt = float(profile["debt"])

    dti = debt / max(income * 12.0, 1.0)  # annualized
    score = risk_score(
        risk_tolerance=int(profile["risk_tolerance"]),
        horizon_years=int(profile["horizon_years"]),
        debt_to_income=float(dti),
        market_stress=float(market_stress),
    )

    alloc = allocation_from_risk(score)

    emergency_target = float(profile["emergency_months"]) * income
    emergency_gap = max(emergency_target - savings, 0.0)

    monthly_goal = monthly_savings_needed(
        goal_amount=float(profile["goal_amount"]),
        current_savings=savings,
        years=int(profile["goal_years"]),
    )

    # Simple prioritized action list.
    actions: list[str] = []
    if emergency_gap > 0:
        actions.append(f"Build emergency fund: need ~${emergency_gap:,.0f} more")
    if dti > 0.6:
        actions.append("Reduce high debt load before increasing risk")
    actions.append(f"Save ~${monthly_goal:,.0f}/month to hit goal")
    actions.append(
        f"Allocation now: cash {alloc['cash']:.0%}, bonds {alloc['bonds']:.0%}, equity {alloc['equity']:.0%} (stress={market_stress:.2f})"
    )

    return {
        "risk_score": score,
        "allocation": alloc,
        "monthly_goal_savings": monthly_goal,
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Personalized financial advisory demo (goal + risk based, scalable rules)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=AdvisoryConfig.n_customers)
    parser.add_argument("--market-stress", type=float, default=0.0, help="0=normal, 1=high stress")
    parser.add_argument("--show", type=int, default=5, help="Print N sample customers")
    args = parser.parse_args()

    prof = generate_customer_profiles(seed=int(args.seed), n=int(args.n))

    stress = float(np.clip(args.market_stress, 0.0, 1.0))

    # Summaries for a quick screenshot.
    risk_scores = np.zeros(int(args.n), dtype=float)
    equity_weights = np.zeros(int(args.n), dtype=float)
    monthly_need = np.zeros(int(args.n), dtype=float)

    for i in range(int(args.n)):
        p = {k: prof[k][i] for k in prof}
        out = advise_one(p, market_stress=stress)
        risk_scores[i] = float(out["risk_score"])
        equity_weights[i] = float(out["allocation"]["equity"])
        monthly_need[i] = float(out["monthly_goal_savings"])

    def pct(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q))

    print(f"\n[Advisory] n={int(args.n)} seed={int(args.seed)} market_stress={stress:.2f}")
    print(
        "[Advisory] risk_score: "
        f"p10={pct(risk_scores,10):.2f} p50={pct(risk_scores,50):.2f} p90={pct(risk_scores,90):.2f}"
    )
    print(
        "[Advisory] equity_weight: "
        f"p10={pct(equity_weights,10):.0%} p50={pct(equity_weights,50):.0%} p90={pct(equity_weights,90):.0%}"
    )
    print(
        "[Advisory] monthly_goal_savings: "
        f"p10=${pct(monthly_need,10):,.0f} p50=${pct(monthly_need,50):,.0f} p90=${pct(monthly_need,90):,.0f}"
    )

    show = max(0, int(args.show))
    if show:
        print("\n[Advisory] sample recommendations:")
        for i in range(min(show, int(args.n))):
            p = {k: prof[k][i] for k in prof}
            out = advise_one(p, market_stress=stress)
            print(
                f"- customer#{i} age={int(p['age'])} income=${float(p['income_monthly']):,.0f} savings=${float(p['savings']):,.0f} debt=${float(p['debt']):,.0f} "
                f"risk_tol={int(p['risk_tolerance'])} horizon={int(p['horizon_years'])}y goal=${float(p['goal_amount']):,.0f} in {int(p['goal_years'])}y"
            )
            for a in out["actions"]:
                print(f"    * {a}")


if __name__ == "__main__":
    main()
