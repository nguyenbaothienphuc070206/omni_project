from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return max(time.perf_counter() - t0, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all hackathon tracks (1,2,3,4) with consistent, screenshot-friendly output (includes Ghost+Sentinel phases)"
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="*",
        default=[100_000_000, 123_456_789],
        help="Transaction/event counts for the big-scale tracks (default: 100000000 123456789)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard", action="store_true", help="Enable hard mode for fraud track")

    # Benchmarks keep runs practical; full is allowed but can take a long time.
    parser.add_argument("--full", action="store_true", help="Run full N for large tracks (can take a long time)")
    parser.add_argument("--benchmark", type=int, default=2_000_000, help="Benchmark slice size (default: 2000000)")

    parser.add_argument("--credit-n", type=int, default=8000)
    parser.add_argument("--advisory-n", type=int, default=2000)
    parser.add_argument("--cyber-users", type=int, default=2000)

    parser.add_argument("--no-ghost", action="store_true", help="Skip Phase 2: Ghost Protocol demo")
    parser.add_argument("--no-sentinel", action="store_true", help="Skip Phase 3: Neural Sentinel demo")

    args = parser.parse_args()

    py = sys.executable

    print("\n================ TRACK 1: CREDIT SCORING ================")
    dt = _run([py, "-m", "src.credit_scoring", "--seed", str(args.seed), "--n", str(args.credit_n), "--show-explain"])
    print(f"[Track 1] elapsed={dt:.2f}s")

    print("\n================ TRACK 2: PERSONALIZED ADVISORY ================")
    dt = _run([py, "-m", "src.advisory", "--seed", str(args.seed), "--n", str(args.advisory_n), "--market-stress", "0.00", "--show", "2"])
    dt2 = _run([py, "-m", "src.advisory", "--seed", str(args.seed), "--n", str(args.advisory_n), "--market-stress", "0.75", "--show", "2"])
    print(f"[Track 2] elapsed={dt+dt2:.2f}s")

    for n in list(args.targets):
        print(f"\n================ TRACK 3: FRAUD (Phase1-only) N={n:,} ================")
        cmd = [py, "-m", "src.train", "--stream", "--phase1-only", "--seed", str(args.seed)]
        if bool(args.hard):
            cmd.append("--hard")
        cmd += ["--n-transactions", str(int(n))]
        if not bool(args.full):
            cmd += ["--benchmark-transactions", str(int(args.benchmark))]
        dt = _run(cmd)
        print(f"[Track 3] elapsed={dt:.2f}s")

        if not bool(args.no_ghost):
            print(f"\n================ PHASE 2: GHOST PROTOCOL (privacy/compliance) N={n:,} ================")
            cmd = [py, "-m", "src.privacy", "--seed", str(args.seed), "--n", str(int(n)), "--nodes", "5", "--show", "3"]
            if not bool(args.full):
                cmd += ["--benchmark", str(int(args.benchmark))]
            dt = _run(cmd)
            print(f"[Phase 2] elapsed={dt:.2f}s")

        if not bool(args.no_sentinel):
            print(f"\n================ PHASE 3: NEURAL SENTINEL (autonomous defense) N={n:,} ================")
            cmd = [py, "-m", "src.sentinel", "--seed", str(args.seed), "--n-transactions", str(int(n)), "--benchmark-transactions", str(int(args.benchmark))]
            if bool(args.hard):
                cmd.append("--hard")
            dt = _run(cmd)
            print(f"[Phase 3] elapsed={dt:.2f}s")

        print(f"\n================ TRACK 4: CYBER (streaming) N={n:,} ================")
        cmd = [
            py,
            "-m",
            "src.cyber_threat",
            "--seed",
            str(args.seed),
            "--n-events",
            str(int(n)),
            "--n-users",
            str(int(args.cyber_users)),
            "--show",
            "5",
        ]
        if not bool(args.full):
            cmd += ["--benchmark-events", str(int(args.benchmark))]
        dt = _run(cmd)
        print(f"[Track 4] elapsed={dt:.2f}s")


if __name__ == "__main__":
    main()
