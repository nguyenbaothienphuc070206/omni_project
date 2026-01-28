from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run_capture(cmd: list[str], *, echo: bool = True) -> tuple[float, str]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = max(time.perf_counter() - t0, 1e-9)
    out = p.stdout or ""
    if echo and out:
        # Print after process finishes; keeps output identical for screenshots.
        print(out, end="" if out.endswith("\n") else "\n")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=out)
    return dt, out


def _parse_after(prefix: str, text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.strip()
    return None


def _parse_train_phase1(text: str) -> dict[str, float] | None:
    # Streaming line example:
    # [Streaming] processed=200,000 elapsed=16.78s speed=11,921 tx/s est_full=1.4 min for 1,000,000 tx
    streaming = _parse_after("[Streaming]", text)
    metrics_line = None
    for line in text.splitlines():
        if line.strip().startswith("acc=") and "f1_fraud=" in line and "fpr=" in line:
            metrics_line = line.strip()

    if not streaming or not metrics_line:
        return None

    def grab_number(after_key: str) -> float | None:
        i = streaming.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        # read until whitespace
        buf = []
        while j < len(streaming) and streaming[j] not in " \t":
            buf.append(streaming[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")

    # Metrics tokens: acc=... prec_fraud=... rec_fraud=... f1_fraud=... fpr=...
    out: dict[str, float] = {}
    for tok in metrics_line.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue

    if speed is None or est_full_min is None:
        return None

    return {
        "phase1_speed_tx_s": float(speed),
        "phase1_est_full_min": float(est_full_min),
        "acc": float(out.get("acc", 0.0)),
        "prec_fraud": float(out.get("prec_fraud", 0.0)),
        "rec_fraud": float(out.get("rec_fraud", 0.0)),
        "f1_fraud": float(out.get("f1_fraud", 0.0)),
        "fpr": float(out.get("fpr", 1.0)),
    }


def _parse_ghost(text: str) -> dict[str, float] | None:
    # Example:
    # [Ghost] tx=1,000,000 processed=200,000 scope=sample nodes=5 elapsed=25.36s speed=7,885 tx/s est_full=2.1 min
    head = _parse_after("[Ghost] tx=", text)
    pass_line = _parse_after("[Ghost] private_settlement_verified=", text)
    if not head or not pass_line:
        return None

    def grab_number(after_key: str) -> float | None:
        i = head.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        buf = []
        while j < len(head) and head[j] not in " \t":
            buf.append(head[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")

    pass_rate = None
    for tok in pass_line.split():
        if tok.startswith("pass_rate="):
            try:
                pass_rate = float(tok.split("=", 1)[1])
            except ValueError:
                pass

    if speed is None or est_full_min is None or pass_rate is None:
        return None

    return {"ghost_speed_tx_s": float(speed), "ghost_est_full_min": float(est_full_min), "ghost_pass_rate": float(pass_rate)}


def _parse_sentinel(text: str) -> dict[str, float] | None:
    # Example:
    # [Sentinel] baseline metrics: {"acc": 1.0, "prec": 1.0, "rec": 1.0, "f1": 1.0, "fpr": 0.0}
    for line in text.splitlines():
        if "[Sentinel] baseline metrics:" in line:
            raw = line.split(":", 1)[1].strip()
            try:
                import json

                obj = json.loads(raw)
                return {
                    "sentinel_acc": float(obj.get("acc", 0.0)),
                    "sentinel_prec": float(obj.get("prec", 0.0)),
                    "sentinel_rec": float(obj.get("rec", 0.0)),
                    "sentinel_f1": float(obj.get("f1", 0.0)),
                    "sentinel_fpr": float(obj.get("fpr", 1.0)),
                }
            except Exception:
                return None
    return None


def _parse_vortex(text: str) -> dict[str, float] | None:
    # Example:
    # [Vortex] trades=1,000,000 processed=200,000 scope=sample elapsed=2.40s speed=83,238 trades/s est_full=0.2 min
    head = _parse_after("[Vortex] trades=", text)
    if not head:
        return None

    def grab_number(after_key: str) -> float | None:
        i = head.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        buf = []
        while j < len(head) and head[j] not in " \t":
            buf.append(head[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")
    if speed is None or est_full_min is None:
        return None

    # Optional stats lines:
    # [Vortex] slippage_bps mean=3.73 p95~10 max=26.57
    slip = _parse_after("[Vortex] slippage_bps", text) or ""
    alpha = _parse_after("[Vortex] alpha", text) or ""

    def grab_from_line(line: str, key: str) -> float | None:
        i = line.find(key)
        if i < 0:
            return None
        j = i + len(key)
        buf = []
        while j < len(line) and line[j] not in " \t":
            buf.append(line[j])
            j += 1
        try:
            return float("".join(buf))
        except ValueError:
            return None

    return {
        "vortex_speed_trades_s": float(speed),
        "vortex_est_full_min": float(est_full_min),
        "vortex_slippage_mean_bps": float(grab_from_line(slip, "mean=") or 0.0),
        "vortex_slippage_p95_bps": float(grab_from_line(slip, "p95~") or 0.0),
        "vortex_slippage_max_bps": float(grab_from_line(slip, "max=") or 0.0),
        "vortex_alpha_mean": float(grab_from_line(alpha, "mean=") or 0.0),
        "vortex_alpha_min": float(grab_from_line(alpha, "min=") or 0.0),
        "vortex_alpha_max": float(grab_from_line(alpha, "max=") or 0.0),
    }


def _parse_phase5(text: str) -> dict[str, float] | None:
    # Example:
    # [Phase5] events=100,000,000 processed=200,000 scope=sample elapsed=2.40s speed=83,238 ev/s est_full=40.1 min
    head = _parse_after("[Phase5] events=", text)
    tail = _parse_after("[Phase5] atomic_batches=", text)
    if not head:
        return None

    def grab_number(after_key: str) -> float | None:
        i = head.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        buf = []
        while j < len(head) and head[j] not in " \t":
            buf.append(head[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")
    if speed is None or est_full_min is None:
        return None

    fairness = 0.0
    if tail:
        i = tail.find("fairness_violations=")
        if i >= 0:
            j = i + len("fairness_violations=")
            buf = []
            while j < len(tail) and tail[j] not in " \t":
                buf.append(tail[j])
                j += 1
            try:
                fairness = float("".join(buf).replace(",", ""))
            except ValueError:
                fairness = 0.0

    return {"phase5_speed_ev_s": float(speed), "phase5_est_full_min": float(est_full_min), "phase5_fairness_violations": float(fairness)}


def _parse_phase6(text: str) -> dict[str, float] | None:
    # Example:
    # [Phase6] calls=100,000,000 processed=200,000 scope=sample elapsed=0.91s speed=219,000 calls/s est_full=7.6 min
    head = _parse_after("[Phase6] calls=", text)
    tail = _parse_after("[Phase6] verified=", text)
    if not head or not tail:
        return None

    def grab_number(after_key: str) -> float | None:
        i = head.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        buf = []
        while j < len(head) and head[j] not in " \t":
            buf.append(head[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")
    if speed is None or est_full_min is None:
        return None

    pass_rate = None
    for tok in tail.split():
        if tok.startswith("pass_rate="):
            try:
                pass_rate = float(tok.split("=", 1)[1])
            except ValueError:
                pass
    if pass_rate is None:
        return None

    return {"phase6_speed_calls_s": float(speed), "phase6_est_full_min": float(est_full_min), "phase6_pass_rate": float(pass_rate)}


def _parse_phase7(text: str) -> dict[str, float] | None:
    # Example:
    # [Phase7] blocks=100,000,000 processed=200,000 scope=sample elapsed=1.24s speed=161,000 blocks/s est_full=10.3 min
    head = _parse_after("[Phase7] blocks=", text)
    tail = _parse_after("[Phase7] verified=", text)
    if not head or not tail:
        return None

    def grab_number(after_key: str) -> float | None:
        i = head.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        buf = []
        while j < len(head) and head[j] not in " \t":
            buf.append(head[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")
    if speed is None or est_full_min is None:
        return None

    pass_rate = None
    for tok in tail.split():
        if tok.startswith("pass_rate="):
            try:
                pass_rate = float(tok.split("=", 1)[1])
            except ValueError:
                pass
    if pass_rate is None:
        return None

    return {"phase7_speed_blocks_s": float(speed), "phase7_est_full_min": float(est_full_min), "phase7_pass_rate": float(pass_rate)}


def _parse_phase8(text: str) -> dict[str, float] | None:
    # Example:
    # [Phase8] signals=100,000,000 processed=200,000 scope=sample elapsed=0.56s speed=359,000 sig/s est_full=4.6 min
    head = _parse_after("[Phase8] signals=", text)
    tail = _parse_after("[Phase8] approval_rate=", text)
    if not head or not tail:
        return None

    def grab_number(after_key: str) -> float | None:
        i = head.find(after_key)
        if i < 0:
            return None
        j = i + len(after_key)
        buf = []
        while j < len(head) and head[j] not in " \t":
            buf.append(head[j])
            j += 1
        s = "".join(buf).replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    speed = grab_number("speed=")
    est_full_min = grab_number("est_full=")
    if speed is None or est_full_min is None:
        return None

    approval_rate = None
    passes_quorum = None
    executed = None
    for tok in tail.split():
        if tok.startswith("approval_rate="):
            try:
                approval_rate = float(tok.split("=", 1)[1])
            except ValueError:
                pass
        if tok.startswith("passes_quorum="):
            passes_quorum = 1.0 if tok.split("=", 1)[1].lower() == "true" else 0.0
        if tok.startswith("executed="):
            executed = 1.0 if tok.split("=", 1)[1].lower() == "true" else 0.0

    if approval_rate is None:
        return None

    return {
        "phase8_speed_sig_s": float(speed),
        "phase8_est_full_min": float(est_full_min),
        "phase8_approval_rate": float(approval_rate),
        "phase8_passes_quorum": float(passes_quorum or 0.0),
        "phase8_executed": float(executed or 0.0),
    }


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

    parser.add_argument(
        "--three-phase-only",
        action="store_true",
        help="Only run Track 3 Phase 1 + Phase 2 Ghost + Phase 3 Sentinel (skip Tracks 1/2/4)",
    )

    parser.add_argument("--no-ghost", action="store_true", help="Skip Phase 2: Ghost Protocol demo")
    parser.add_argument("--no-sentinel", action="store_true", help="Skip Phase 3: Neural Sentinel demo")
    parser.add_argument("--no-vortex", action="store_true", help="Skip Phase 4: Liquidity Vortex demo")
    parser.add_argument("--no-phase5", action="store_true", help="Skip Phase 5: Temporal Execution Engine demo")
    parser.add_argument("--no-phase6", action="store_true", help="Skip Phase 6: Versioned ABI + rotation demo")
    parser.add_argument("--no-phase7", action="store_true", help="Skip Phase 7: Quantum-ready ledger demo")
    parser.add_argument("--no-phase8", action="store_true", help="Skip Phase 8: Neuromorphic governance demo")

    args = parser.parse_args()

    py = sys.executable

    if not bool(args.three_phase_only):
        print("\n================ TRACK 1: CREDIT SCORING ================")
        dt, _ = _run_capture([py, "-m", "src.credit_scoring", "--seed", str(args.seed), "--n", str(args.credit_n), "--show-explain"])
        print(f"[Track 1] elapsed={dt:.2f}s")

        print("\n================ TRACK 2: PERSONALIZED ADVISORY ================")
        dt_a, _ = _run_capture([
            py,
            "-m",
            "src.advisory",
            "--seed",
            str(args.seed),
            "--n",
            str(args.advisory_n),
            "--market-stress",
            "0.00",
            "--show",
            "2",
        ])
        dt_b, _ = _run_capture([
            py,
            "-m",
            "src.advisory",
            "--seed",
            str(args.seed),
            "--n",
            str(args.advisory_n),
            "--market-stress",
            "0.75",
            "--show",
            "2",
        ])
        print(f"[Track 2] elapsed={dt_a+dt_b:.2f}s")

    for n in list(args.targets):
        print(f"\n================ TRACK 3: FRAUD (Phase1-only) N={n:,} ================")
        cmd = [py, "-m", "src.train", "--stream", "--phase1-only", "--seed", str(args.seed)]
        if bool(args.hard):
            cmd.append("--hard")
        cmd += ["--n-transactions", str(int(n))]
        if not bool(args.full):
            cmd += ["--benchmark-transactions", str(int(args.benchmark))]
        dt, out_train = _run_capture(cmd)
        print(f"[Track 3] elapsed={dt:.2f}s")

        if not bool(args.no_ghost):
            print(f"\n================ PHASE 2: GHOST PROTOCOL (privacy/compliance) N={n:,} ================")
            cmd = [py, "-m", "src.privacy", "--seed", str(args.seed), "--n", str(int(n)), "--nodes", "5", "--show", "3"]
            if not bool(args.full):
                cmd += ["--benchmark", str(int(args.benchmark))]
            dt, out_ghost = _run_capture(cmd)
            print(f"[Phase 2] elapsed={dt:.2f}s")
        else:
            out_ghost = ""

        if not bool(args.no_sentinel):
            print(f"\n================ PHASE 3: NEURAL SENTINEL (autonomous defense) N={n:,} ================")
            cmd = [py, "-m", "src.sentinel", "--seed", str(args.seed), "--n-transactions", str(int(n)), "--benchmark-transactions", str(int(args.benchmark))]
            if bool(args.hard):
                cmd.append("--hard")
            dt_sentinel, out_sentinel = _run_capture(cmd)
            print(f"[Phase 3] elapsed={dt_sentinel:.2f}s")
        else:
            dt_sentinel = 0.0
            out_sentinel = ""

        # Phase 4 is optional; if user asks for "three-phase-only", stop at Phase 3.
        if (not bool(args.three_phase_only)) and (not bool(args.no_vortex)):
            print(f"\n================ PHASE 4: LIQUIDITY VORTEX (hyperbolic AMM) N={n:,} ================")
            cmd = [
                py,
                "-m",
                "src.liquidity_vortex",
                "--seed",
                str(args.seed),
                "--n-trades",
                str(int(n)),
                "--avg-size",
                "150",
                "--show",
                "3",
            ]
            if not bool(args.full):
                cmd += ["--benchmark-trades", str(int(args.benchmark))]
            dt, out_vortex = _run_capture(cmd)
            print(f"[Phase 4] elapsed={dt:.2f}s")
        else:
            out_vortex = ""

        if (not bool(args.three_phase_only)) and (not bool(args.no_phase5)):
            print(f"\n================ PHASE 5: TEMPORAL EXECUTION (streaming atomic batches) N={n:,} ================")
            cmd = [py, "-m", "src.temporal_arbitrage", "--seed", str(args.seed), "--n-events", str(int(n)), "--show", "3"]
            if not bool(args.full):
                cmd += ["--benchmark-events", str(int(args.benchmark))]
            dt, out_p5 = _run_capture(cmd)
            print(f"[Phase 5] elapsed={dt:.2f}s")
        else:
            out_p5 = ""

        if (not bool(args.three_phase_only)) and (not bool(args.no_phase6)):
            print(f"\n================ PHASE 6: MORPHING CONTRACTS (versioned ABI) N={n:,} ================")
            cmd = [py, "-m", "src.morphing_contracts", "--seed", str(args.seed), "--n-calls", str(int(n)), "--show", "3"]
            if not bool(args.full):
                cmd += ["--benchmark-calls", str(int(args.benchmark))]
            dt, out_p6 = _run_capture(cmd)
            print(f"[Phase 6] elapsed={dt:.2f}s")
        else:
            out_p6 = ""

        if (not bool(args.three_phase_only)) and (not bool(args.no_phase7)):
            print(f"\n================ PHASE 7: QUANTUM-READY LEDGER (crypto agility) N={n:,} ================")
            cmd = [py, "-m", "src.quantum_ledger", "--seed", str(args.seed), "--n-blocks", str(int(n)), "--show", "2"]
            if not bool(args.full):
                cmd += ["--benchmark-blocks", str(int(args.benchmark))]
            dt, out_p7 = _run_capture(cmd)
            print(f"[Phase 7] elapsed={dt:.2f}s")
        else:
            out_p7 = ""

        if (not bool(args.three_phase_only)) and (not bool(args.no_phase8)):
            print(f"\n================ PHASE 8: NEUROMORPHIC GOVERNANCE (AI-assisted) N={n:,} ================")
            cmd = [py, "-m", "src.neuromorphic_governance", "--seed", str(args.seed), "--n-signals", str(int(n)), "--show", "2"]
            if not bool(args.full):
                cmd += ["--benchmark-signals", str(int(args.benchmark))]
            dt, out_p8 = _run_capture(cmd)
            print(f"[Phase 8] elapsed={dt:.2f}s")
        else:
            out_p8 = ""

        # Pipeline summary for this target (rates + estimated full time)
        p1 = _parse_train_phase1(out_train) or {}
        gp = _parse_ghost(out_ghost) or {}
        sn = _parse_sentinel(out_sentinel) or {}
        vx = _parse_vortex(out_vortex) or {}
        p5 = _parse_phase5(out_p5) or {}
        p6 = _parse_phase6(out_p6) or {}
        p7 = _parse_phase7(out_p7) or {}
        p8 = _parse_phase8(out_p8) or {}

        # Sentinel timing: it benchmarks via src.train but doesn't print est_full.
        # We estimate full time by scaling baseline elapsed by (n / benchmark).
        sentinel_est_full_min = 0.0
        if not bool(args.no_sentinel):
            bench = float(max(int(args.benchmark), 1))
            sentinel_est_full_min = (float(dt_sentinel) / 60.0) * (float(n) / bench)

        est_total = 0.0
        est_total += float(p1.get("phase1_est_full_min", 0.0))
        est_total += float(gp.get("ghost_est_full_min", 0.0))
        est_total += float(vx.get("vortex_est_full_min", 0.0))
        est_total += float(p5.get("phase5_est_full_min", 0.0))
        est_total += float(p6.get("phase6_est_full_min", 0.0))
        est_total += float(p7.get("phase7_est_full_min", 0.0))
        est_total += float(p8.get("phase8_est_full_min", 0.0))
        est_total += float(sentinel_est_full_min)

        phase_count = 0
        for present in [bool(p1), bool(gp), (not bool(args.no_sentinel)), bool(vx), bool(p5), bool(p6), bool(p7), bool(p8)]:
            if present:
                phase_count += 1
        label = f"{phase_count}-PHASE"
        print(f"\n---------------- {label} SUMMARY N={n:,} ----------------")
        if p1:
            print(
                f"[Phase 1] speed={p1['phase1_speed_tx_s']:,.0f} tx/s est_full={p1['phase1_est_full_min']:.1f} min "
                f"acc={p1['acc']:.3f} prec_fraud={p1['prec_fraud']:.3f} rec_fraud={p1['rec_fraud']:.3f} f1_fraud={p1['f1_fraud']:.3f} fpr={p1['fpr']:.3f}"
            )
        if gp:
            print(
                f"[Phase 2] speed={gp['ghost_speed_tx_s']:,.0f} tx/s est_full={gp['ghost_est_full_min']:.1f} min pass_rate={gp['ghost_pass_rate']:.3f}"
            )
        if sn:
            print(
                f"[Phase 3] acc={sn['sentinel_acc']:.3f} prec={sn['sentinel_prec']:.3f} rec={sn['sentinel_rec']:.3f} "
                f"f1={sn['sentinel_f1']:.3f} fpr={sn['sentinel_fpr']:.3f} est_full~{sentinel_est_full_min:.1f} min"
            )
        if vx:
            print(
                f"[Phase 4] speed={vx['vortex_speed_trades_s']:,.0f} trades/s est_full={vx['vortex_est_full_min']:.1f} min "
                f"slip_mean={vx['vortex_slippage_mean_bps']:.2f}bps p95~{vx['vortex_slippage_p95_bps']:.0f} max={vx['vortex_slippage_max_bps']:.2f} "
                f"alpha_mean={vx['vortex_alpha_mean']:.3f}"
            )
        if p5:
            print(
                f"[Phase 5] speed={p5['phase5_speed_ev_s']:,.0f} ev/s est_full={p5['phase5_est_full_min']:.1f} min "
                f"fairness_violations={p5['phase5_fairness_violations']:,.0f}"
            )
        if p6:
            print(
                f"[Phase 6] speed={p6['phase6_speed_calls_s']:,.0f} calls/s est_full={p6['phase6_est_full_min']:.1f} min pass_rate={p6['phase6_pass_rate']:.3f}"
            )
        if p7:
            print(
                f"[Phase 7] speed={p7['phase7_speed_blocks_s']:,.0f} blocks/s est_full={p7['phase7_est_full_min']:.1f} min pass_rate={p7['phase7_pass_rate']:.3f}"
            )
        if p8:
            print(
                f"[Phase 8] speed={p8['phase8_speed_sig_s']:,.0f} sig/s est_full={p8['phase8_est_full_min']:.1f} min "
                f"approval_rate={p8['phase8_approval_rate']:.3f} passes_quorum={bool(p8['phase8_passes_quorum'])} executed={bool(p8['phase8_executed'])}"
            )

        print(f"[TOTAL] estimated_full_runtime~{est_total:.1f} min\n")

        if not bool(args.three_phase_only):
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
            dt, _ = _run_capture(cmd)
            print(f"[Track 4] elapsed={dt:.2f}s")


if __name__ == "__main__":
    main()
