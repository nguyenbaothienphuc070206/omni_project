from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase8Config:
    n_signals: int = 2_000_000
    quorum: float = 0.60
    human_veto: bool = True


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _iter_signals(*, seed: int, n: int) -> tuple[int, float, float]:
    """Synthetic governance signals.

    Yields (i, sentiment, risk).
    sentiment in [-1,1], risk in [0,1].
    """

    import numpy as np

    rng = np.random.default_rng(seed)
    for i in range(int(n)):
        s = float(rng.normal(loc=0.15, scale=0.45))
        s = _clamp(s, -1.0, 1.0)
        r = float(rng.beta(a=2.0, b=6.0))
        yield i, s, r


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 demo: Neuromorphic Governance (AI-assisted proposals + quorum + human veto, compliance-safe)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-signals", type=int, default=Phase8Config.n_signals)
    parser.add_argument(
        "--benchmark-signals",
        type=int,
        default=None,
        help="Process only first N signals and estimate full runtime",
    )
    parser.add_argument("--quorum", type=float, default=Phase8Config.quorum)
    parser.add_argument("--no-human-veto", action="store_true", help="Disable human veto (demo only; not recommended)")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    n_requested = int(args.n_signals)
    n_process = int(args.benchmark_signals) if args.benchmark_signals is not None else n_requested
    n_process = min(n_process, n_requested)

    if args.benchmark_signals is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} signals (estimate full runtime).")

    quorum = _clamp(float(args.quorum), 0.0, 1.0)
    human_veto = not bool(args.no_human_veto)

    # Streaming aggregates
    s_sum = 0.0
    r_sum = 0.0
    approve = 0

    # Simple "collective intelligence" score; bounded and explainable.
    # score = sigmoid( 3.0*sentiment_mean - 2.0*risk_mean )
    examples: list[str] = []

    t0 = time.perf_counter()
    for i, sentiment, risk in _iter_signals(seed=int(args.seed), n=int(n_process)):
        s_sum += sentiment
        r_sum += risk

        s_mean = s_sum / (i + 1)
        r_mean = r_sum / (i + 1)
        score = 1.0 / (1.0 + math.exp(-(3.0 * s_mean - 2.0 * r_mean)))

        # approval proxy: if score above 0.5, count as approve
        if score >= 0.5:
            approve += 1

        if len(examples) < int(args.show) and i % max(1, n_process // max(1, int(args.show))) == 0:
            examples.append(f"- i={i} s_mean={s_mean:+.3f} r_mean={r_mean:.3f} score={score:.3f}")

    dt = max(time.perf_counter() - t0, 1e-9)
    speed = n_process / dt
    est_full_sec = (n_requested / speed) if speed > 0 else float("inf")

    approval_rate = approve / max(n_process, 1)
    passes_quorum = approval_rate >= quorum

    # Human veto gate: always available in this safe demo.
    executed = passes_quorum and (not human_veto)

    scope = "full" if n_process == n_requested else "sample"
    print(
        f"\n[Phase8] signals={n_requested:,} processed={n_process:,} scope={scope} elapsed={dt:.2f}s "
        f"speed={speed:,.0f} sig/s est_full={est_full_sec/60:.1f} min"
    )
    print(
        f"[Phase8] approval_rate={approval_rate:.3f} quorum={quorum:.3f} passes_quorum={passes_quorum} "
        f"human_veto_enabled={human_veto} executed={executed}"
    )

    if examples:
        print("\n[Phase8] examples:")
        for e in examples:
            print(e)

    print("\n[Phase8] notes:")
    print("- Compliance-safe demo: AI assists scoring; execution remains gated (quorum + optional human veto).")
    print("- Does NOT remove humans by default; governance remains controllable and auditable.")


if __name__ == "__main__":
    main()
