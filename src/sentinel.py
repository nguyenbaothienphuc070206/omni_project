from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SentinelPolicy:
    """Minimal policy for autonomous defense.

    We do NOT modify code at runtime.
    We only change safe knobs (thresholds / settings) and rerun self-checks.
    """

    min_f1: float = 0.60
    max_fpr: float = 0.10


def _run_capture(cmd: list[str]) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = max(time.perf_counter() - t0, 1e-9)
    return int(p.returncode), dt, p.stdout


def _extract_last_metrics(text: str) -> dict[str, float] | None:
    """Parse last metrics line from `src.train` streaming Phase1-only output.

    Example line:
      acc=1.000 prec_fraud=1.000 rec_fraud=1.000 f1_fraud=1.000 fpr=0.000 ...
    """

    last = None
    for line in text.splitlines():
        # We target the summary metrics line printed by src.train.
        # Example tokens: acc=..., prec_fraud=..., rec_fraud=..., f1_fraud=..., fpr=...
        if "acc=" in line and "f1_fraud=" in line and "fpr=" in line:
            last = line.strip()

    if not last:
        return None

    out: dict[str, float] = {}
    toks = last.split()
    for tok in toks:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip().strip(",")
        try:
            out[k] = float(v)
        except ValueError:
            continue

    # Normalize expected keys
    if "f1_fraud" in out and "fpr" in out:
        return {
            "acc": float(out.get("acc", 0.0)),
            "prec": float(out.get("prec_fraud", 0.0)),
            "rec": float(out.get("rec_fraud", 0.0)),
            "f1": float(out.get("f1_fraud", 0.0)),
            "fpr": float(out.get("fpr", 1.0)),
        }

    return None


def _grid_thresholds() -> list[float]:
    return [round(x, 2) for x in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 demo: Neural Sentinel (autonomous defense, safe + simple)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--n-transactions", type=int, default=1_000_000)
    parser.add_argument("--benchmark-transactions", type=int, default=500_000)
    parser.add_argument("--dry-run", action="store_true", help="Print proposed mitigation without running it")

    parser.add_argument("--min-f1", type=float, default=SentinelPolicy.min_f1)
    parser.add_argument("--max-fpr", type=float, default=SentinelPolicy.max_fpr)
    args = parser.parse_args()

    py = sys.executable
    policy = SentinelPolicy(min_f1=float(args.min_f1), max_fpr=float(args.max_fpr))

    base_cmd = [
        py,
        "-m",
        "src.train",
        "--stream",
        "--phase1-only",
        "--seed",
        str(int(args.seed)),
        "--n-transactions",
        str(int(args.n_transactions)),
        "--benchmark-transactions",
        str(int(args.benchmark_transactions)),
    ]
    if bool(args.hard):
        base_cmd.append("--hard")

    print("\n[Sentinel] self-check: running baseline")
    rc, dt, out = _run_capture(base_cmd)
    print(f"[Sentinel] baseline rc={rc} elapsed={dt:.2f}s")

    m = _extract_last_metrics(out)
    if not m:
        print("[Sentinel] Could not parse metrics from output")
        print(out)
        raise SystemExit(2)

    print(f"[Sentinel] baseline metrics: {json.dumps(m)}")

    if m["f1"] >= policy.min_f1 and m["fpr"] <= policy.max_fpr:
        print("[Sentinel] OK: metrics within policy")
        return

    # Mitigation: tune Phase1-only threshold (safe knob) via small grid.
    print("\n[Sentinel] ALERT: metrics out of policy")
    print(f"[Sentinel] policy: min_f1={policy.min_f1:.3f} max_fpr={policy.max_fpr:.3f}")
    print("[Sentinel] mitigation: search a safer user threshold (prior_anomaly_score)")

    best = None
    for thr in _grid_thresholds():
        cmd = base_cmd + ["--phase1-threshold", str(thr)]
        rc2, dt2, out2 = _run_capture(cmd)
        mm = _extract_last_metrics(out2)
        if rc2 != 0 or not mm:
            continue

        ok = (mm["f1"] >= policy.min_f1) and (mm["fpr"] <= policy.max_fpr)
        score = (mm["f1"], -mm["fpr"])  # maximize f1, minimize fpr
        if best is None:
            best = (ok, score, thr, mm, dt2)
        else:
            if ok and not best[0]:
                best = (ok, score, thr, mm, dt2)
            elif ok == best[0] and score > best[1]:
                best = (ok, score, thr, mm, dt2)

    if not best:
        print("[Sentinel] mitigation failed: no threshold produced metrics")
        return

    ok, _score, thr, mm, dt2 = best
    print(f"[Sentinel] best_threshold={thr:.2f} elapsed={dt2:.2f}s metrics={json.dumps(mm)} within_policy={ok}")

    if bool(args.dry_run):
        print("[Sentinel] dry-run: not applying changes")
        return

    # "Apply" in demo sense: print the recommended operational knob.
    # In production, this would update a config store and trigger a safe rollout.
    print("\n[Sentinel] APPLY: set operational threshold")
    print(f"[Sentinel] export PHASE1_THRESHOLD={thr:.2f}")


if __name__ == "__main__":
    main()
