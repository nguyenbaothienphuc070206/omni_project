from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class VortexConfig:
    n_trades: int = 1_000_000
    base_price: float = 1.0

    # Hyperbolic liquidity scale: larger => lower slippage for same trade size
    liquidity_scale: float = 50_000.0

    # Base curvature (slope factor). Higher => more slippage.
    base_alpha: float = 0.35

    # Volatility adaptation strength: alpha = base_alpha * (1 + beta * vol_rms)
    beta: float = 6.0

    # EWMA for volatility of log-returns
    vol_ewma: float = 0.02

    alpha_min: float = 0.10
    alpha_max: float = 2.00


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _price(*, q: float, base_price: float, alpha: float, L: float) -> float:
    """Non-Euclidean (hyperbolic) bonding curve.

    We map inventory q into hyperbolic space via asinh(q/L).
    Price grows (or shrinks) exponentially with hyperbolic distance.

      p(q) = p0 * exp(alpha * asinh(q/L))

    This is smooth, monotone, and stable for large |q|.
    """

    return base_price * math.exp(alpha * math.asinh(q / L))


def _iter_synthetic_trades(*, seed: int, n: int, avg_size: float) -> tuple[int, int, float]:
    """Fast synthetic trade generator.

    Yields (i, side, size) where side=+1 buy, -1 sell.
    """

    import numpy as np

    rng = np.random.default_rng(seed)

    # Regime switching to create volatility bursts.
    p_buy = 0.52
    burst_every = 30_000

    for i in range(int(n)):
        if burst_every and (i % burst_every == 0) and i > 0:
            # flip bias occasionally
            p_buy = 0.48 if p_buy > 0.5 else 0.52

        side = 1 if rng.random() < p_buy else -1
        # Lognormal sizes; keep bounded for stability.
        size = float(rng.lognormal(mean=math.log(avg_size), sigma=0.7))
        if size > 5.0 * avg_size:
            size = 5.0 * avg_size

        yield i, side, size


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 demo: The Liquidity Vortex (hyperbolic bonding curve AMM)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-trades", type=int, default=VortexConfig.n_trades)
    parser.add_argument(
        "--benchmark-trades",
        type=int,
        default=None,
        help="Process only first N trades and estimate full runtime (recommended for 100M+)",
    )
    parser.add_argument("--avg-size", type=float, default=150.0)
    parser.add_argument("--liquidity-scale", type=float, default=VortexConfig.liquidity_scale)
    parser.add_argument("--base-alpha", type=float, default=VortexConfig.base_alpha)
    parser.add_argument("--beta", type=float, default=VortexConfig.beta)
    parser.add_argument("--vol-ewma", type=float, default=VortexConfig.vol_ewma)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    n_requested = int(args.n_trades)
    n_process = int(args.benchmark_trades) if args.benchmark_trades is not None else n_requested
    n_process = min(n_process, n_requested)

    if args.benchmark_trades is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} trades (estimate full runtime).")

    p0 = float(VortexConfig.base_price)
    L = float(args.liquidity_scale)
    base_alpha = float(args.base_alpha)
    beta = float(args.beta)
    lam = _clamp(float(args.vol_ewma), 0.0, 1.0)

    q = 0.0

    # EWMA of squared log-returns
    vol2 = 0.0

    # Streaming stats (no arrays)
    k = 0
    sl_sum = 0.0
    sl_max = 0.0

    # Approximate p95 with a small fixed histogram (fast, O(1))
    # bins are in bps, [0, 200] + overflow
    bin_max = 200
    bins = [0] * (bin_max + 2)

    a_sum = 0.0
    a_min = float("inf")
    a_max = 0.0

    examples: list[str] = []

    t0 = time.perf_counter()
    for i, side, size in _iter_synthetic_trades(seed=int(args.seed), n=n_process, avg_size=float(args.avg_size)):
        vol_rms = math.sqrt(vol2)
        alpha = _clamp(base_alpha * (1.0 + beta * vol_rms), VortexConfig.alpha_min, VortexConfig.alpha_max)

        p_before = _price(q=q, base_price=p0, alpha=alpha, L=L)

        dq = float(side) * float(size)

        # Midpoint execution approximation (fast + stable)
        p_exec = _price(q=q + 0.5 * dq, base_price=p0, alpha=alpha, L=L)

        if dq >= 0:
            sl_bps = (p_exec / p_before - 1.0) * 1e4
        else:
            sl_bps = (p_before / p_exec - 1.0) * 1e4

        if sl_bps < 0:
            sl_bps = 0.0

        q += dq
        p_after = _price(q=q, base_price=p0, alpha=alpha, L=L)

        # update volatility estimator on log-return (stable)
        r = math.log(max(p_after, 1e-12) / max(p_before, 1e-12))
        vol2 = (1.0 - lam) * vol2 + lam * (r * r)

        k += 1
        sl_sum += sl_bps
        sl_max = sl_bps if sl_bps > sl_max else sl_max

        bi = int(sl_bps)
        if bi < 0:
            bi = 0
        if bi > bin_max:
            bi = bin_max + 1
        bins[bi] += 1

        a_sum += alpha
        a_min = alpha if alpha < a_min else a_min
        a_max = alpha if alpha > a_max else a_max

        if len(examples) < int(args.show):
            examples.append(
                f"- i={i} side={'BUY' if side>0 else 'SELL'} size={size:,.1f} alpha={alpha:.3f} vol_rms={vol_rms:.4f} price={p_before:.6f} slippage_bps={sl_bps:.2f}"
            )

    dt = max(time.perf_counter() - t0, 1e-9)
    speed = n_process / dt
    est_full_sec = (n_requested / speed) if speed > 0 else float("inf")

    # p95 from histogram
    target = int(math.ceil(0.95 * k))
    c = 0
    p95 = 0
    for b, cnt in enumerate(bins):
        c += cnt
        if c >= target:
            p95 = b if b <= bin_max else bin_max
            break

    mean = sl_sum / max(k, 1)
    a_mean = a_sum / max(k, 1)

    scope = "full" if n_process == n_requested else "sample"
    print(
        f"\n[Vortex] trades={n_requested:,} processed={n_process:,} scope={scope} elapsed={dt:.2f}s "
        f"speed={speed:,.0f} trades/s est_full={est_full_sec/60:.1f} min"
    )
    print(f"[Vortex] slippage_bps mean={mean:.2f} p95~{p95:.0f} max={sl_max:.2f}")
    print(f"[Vortex] alpha mean={a_mean:.3f} min={a_min:.3f} max={a_max:.3f} (vol-adaptive)")

    if examples:
        print("\n[Vortex] examples:")
        for e in examples:
            print(e)

    print("\n[Vortex] notes:")
    print("- Demo-only AMM simulator: hyperbolic bonding curve + volatility-adaptive slope.")
    print("- Not a production DEX, does not claim to eliminate impermanent loss.")


if __name__ == "__main__":
    main()
