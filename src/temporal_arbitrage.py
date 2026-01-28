from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase5Config:
    n_events: int = 5_000_000
    
    # Target service budget (microseconds) for a single batch.
    batch_budget_us: float = 250.0

    # Base per-event cost (microseconds) for a tight loop.
    base_cost_us: float = 0.18

    # Extra cost per event when contention is high.
    contention_cost_us: float = 0.08

    # EWMA smoothing for predicted latency.
    ewma: float = 0.05


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _iter_stream(*, seed: int, n: int) -> tuple[int, int, float]:
    """Synthetic market events stream.

    Yields (i, venue_id, size).

    This is a *compliance-safe* demo: it models ingest + atomic processing.
    It does NOT implement or encourage exploiting other venues.
    """

    import numpy as np

    rng = np.random.default_rng(seed)
    for i in range(int(n)):
        venue = int(rng.integers(0, 6))
        # sizes vary, but bounded
        size = float(rng.lognormal(mean=4.3, sigma=0.6))
        if size > 300.0:
            size = 300.0
        yield i, venue, size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 demo: Temporal Execution Engine (streaming + atomic batch processing, compliance-safe)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-events", type=int, default=Phase5Config.n_events)
    parser.add_argument(
        "--benchmark-events",
        type=int,
        default=None,
        help="Process only first N events and estimate full runtime (recommended for 100M+)",
    )
    parser.add_argument("--batch-budget-us", type=float, default=Phase5Config.batch_budget_us)
    parser.add_argument("--base-cost-us", type=float, default=Phase5Config.base_cost_us)
    parser.add_argument("--contention-cost-us", type=float, default=Phase5Config.contention_cost_us)
    parser.add_argument("--ewma", type=float, default=Phase5Config.ewma)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    n_requested = int(args.n_events)
    n_process = int(args.benchmark_events) if args.benchmark_events is not None else n_requested
    n_process = min(n_process, n_requested)

    if args.benchmark_events is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} events (estimate full runtime).")

    # State per venue (constant size)
    venue_load = [0.0] * 6

    # Latency predictor (EWMA)
    lam = _clamp(float(args.ewma), 0.0, 1.0)
    pred_us = float(args.base_cost_us)

    budget = float(args.batch_budget_us)
    base = float(args.base_cost_us)
    cont = float(args.contention_cost_us)

    # Atomic batches (fair FIFO): we never reorder for advantage; we only batch for throughput.
    batch_n = 0
    batches = 0
    max_batch = 0

    # Minimal quality signals
    fairness_violations = 0

    examples: list[str] = []

    t0 = time.perf_counter()
    last_i = -1

    for i, venue, size in _iter_stream(seed=int(args.seed), n=int(n_process)):
        # FIFO check (should never violate; defensive signal)
        if i <= last_i:
            fairness_violations += 1
        last_i = i

        # Simulate venue contention as an EWMA load
        venue_load[venue] = 0.98 * venue_load[venue] + 0.02 * (size / 100.0)
        contention = sum(venue_load) / len(venue_load)

        # Predicted cost for next event
        cost_us = base + cont * contention
        pred_us = (1.0 - lam) * pred_us + lam * cost_us

        # Greedy batching to stay under budget
        if (batch_n + 1) * pred_us <= budget:
            batch_n += 1
        else:
            batches += 1
            if batch_n > max_batch:
                max_batch = batch_n
            batch_n = 1

        if len(examples) < int(args.show) and i % max(1, n_process // max(1, int(args.show))) == 0:
            examples.append(
                f"- i={i} venue={venue} size={size:.1f} pred_cost_us={pred_us:.3f} contention={contention:.3f} batch_n={batch_n}"
            )

    if batch_n > 0:
        batches += 1
        if batch_n > max_batch:
            max_batch = batch_n

    dt = max(time.perf_counter() - t0, 1e-9)
    speed = n_process / dt
    est_full_sec = (n_requested / speed) if speed > 0 else float("inf")

    scope = "full" if n_process == n_requested else "sample"
    avg_batch = n_process / max(batches, 1)

    print(
        f"\n[Phase5] events={n_requested:,} processed={n_process:,} scope={scope} elapsed={dt:.2f}s "
        f"speed={speed:,.0f} ev/s est_full={est_full_sec/60:.1f} min"
    )
    print(
        f"[Phase5] atomic_batches={batches:,} avg_batch={avg_batch:.1f} max_batch={max_batch} "
        f"batch_budget_us={budget:.0f} pred_cost_us={pred_us:.3f} fairness_violations={fairness_violations}"
    )

    if examples:
        print("\n[Phase5] examples:")
        for e in examples:
            print(e)

    print("\n[Phase5] notes:")
    print("- Compliance-safe demo: focuses on streaming, atomic batching, and fairness (FIFO).")
    print("- Does NOT implement front-running, venue exploitation, or adversarial network routing.")


if __name__ == "__main__":
    main()
