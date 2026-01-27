from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class CyberConfig:
    n_events: int = 200_000
    n_users: int = 2000
    n_ips: int = 5000
    n_devices: int = 3500
    attack_rate: float = 0.02


Event = tuple[int, int, int, int, int, float, float, float, int]
# (ts, user_id, ip_id, device_id, failed_logins, success_login, bytes_out, bytes_in, geo_km, is_attack)


def iter_synthetic_security_events(
    *,
    seed: int,
    n_events: int,
    n_users: int,
    n_ips: int,
    n_devices: int,
    attack_rate: float,
) -> Iterator[Event]:
    """Stream events one-by-one (O(1) memory w.r.t. n_events)."""

    rng = np.random.default_rng(seed)
    ts = 0
    for _ in range(int(n_events)):
        ts += int(rng.integers(0, 3))

        user_id = int(rng.integers(0, n_users))
        ip_id = int(rng.integers(0, n_ips))
        device_id = int(rng.integers(0, n_devices))

        failed_logins = int(rng.poisson(lam=0.08))
        success_login = int(rng.random() < 0.35)

        bytes_out = float(rng.lognormal(mean=float(np.log(60_000)), sigma=0.8))
        bytes_in = float(rng.lognormal(mean=float(np.log(120_000)), sigma=0.8))
        geo_km = float(rng.exponential(scale=30.0))

        is_attack = int(rng.random() < float(attack_rate))
        if is_attack:
            attack_type = int(rng.integers(1, 4))
            if attack_type == 1:
                failed_logins += int(rng.integers(4, 14))
                success_login = int(rng.random() < 0.05)
                geo_km += float(rng.exponential(scale=150.0))
            elif attack_type == 2:
                success_login = 1
                failed_logins += int(rng.integers(1, 4))
                geo_km += float(rng.exponential(scale=400.0))
                device_id = int(rng.integers(0, n_devices))
            else:
                bytes_out *= float(rng.uniform(20.0, 120.0))
                bytes_in *= float(rng.uniform(0.2, 0.8))

        yield (ts, user_id, ip_id, device_id, failed_logins, success_login, bytes_out, bytes_in, geo_km, is_attack)


def _metrics_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {
        "accuracy": float(acc),
        "precision_attack": float(prec),
        "recall_attack": float(rec),
        "f1_attack": float(f1),
        "fpr": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


class OnlineUserState:
    __slots__ = ("mean_out", "var_out", "count", "last_device", "last_ts", "fail_ewma")

    def __init__(self) -> None:
        self.mean_out = 0.0
        self.var_out = 1.0
        self.count = 0
        self.last_device = -1
        self.last_ts = -1
        self.fail_ewma = 0.0


def detect_threats_streaming(
    event_iter: Iterable[Event],
    *,
    n_users: int,
    z_out_threshold: float = 4.0,
    geo_far_km: float = 300.0,
    fail_burst_ewma: float = 2.5,
    ewma_alpha: float = 0.05,
    keep_examples: int = 0,
) -> tuple[dict[str, float], list[str]]:
    """Streaming threat detection with explainable reasons.

    Output:
      - y_pred per event
      - reasons per event (list of short strings)

    Senior reasoning:
    - Use bounded per-user state (O(n_users) memory).
    - Keep detectors interpretable and fast (no heavy models needed for the demo).
    """

    states = [OnlineUserState() for _ in range(int(n_users))]

    tp = tn = fp = fn = 0
    flagged_examples: list[str] = []

    idx = 0
    for (ts, uid, _ip, dev, failed, success, outb, _inb, geo, is_attack) in event_iter:
        st = states[int(uid)]

        # Update EWMA for failed logins (burst detection)
        st.fail_ewma = (1.0 - ewma_alpha) * st.fail_ewma + ewma_alpha * float(failed)

        # Online mean/variance (Welford) on bytes_out per user
        st.count += 1
        delta = outb - st.mean_out
        st.mean_out += delta / max(st.count, 1)
        delta2 = outb - st.mean_out
        st.var_out += delta * delta2
        std = float(np.sqrt(max(st.var_out / max(st.count, 2), 1e-9)))
        z = (outb - st.mean_out) / max(std, 1e-9)

        score = 0
        why: list[str] = []

        if st.fail_ewma >= float(fail_burst_ewma) and failed >= 3:
            score += 2
            why.append("failed_login_burst")

        # ATO-ish: success from far geo and device changed
        if success == 1 and geo >= float(geo_far_km) and st.last_device != -1 and dev != st.last_device:
            score += 2
            why.append("new_device_far_geo")

        # Exfil-ish: bytes_out spike compared to user baseline
        if st.count >= 25 and z >= float(z_out_threshold):
            score += 2
            why.append("bytes_out_spike")

        # Mild heuristic: far geo without known context
        if geo >= float(geo_far_km) and success == 1:
            score += 1
            why.append("far_geo_success")

        # Final decision: small integer score threshold
        pred = 1 if score >= 3 else 0
        truth = int(is_attack)
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 0:
            tn += 1
        else:
            fn += 1

        if keep_examples and pred == 1 and len(flagged_examples) < int(keep_examples):
            why_str = ",".join(why) if why else "(no_reason)"
            flagged_examples.append(
                f"- idx={idx} user={int(uid)} truth={truth} success={int(success)} failed={int(failed)} geo_km={float(geo):.0f} bytes_out={float(outb):,.0f} reasons=[{why_str}]"
            )

        st.last_device = dev
        st.last_ts = ts
        idx += 1

    return _metrics_counts(tp, tn, fp, fn), flagged_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Cybersecurity threat detection demo (technical layer, streaming + explainable)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-events", type=int, default=CyberConfig.n_events)
    parser.add_argument("--n-users", type=int, default=CyberConfig.n_users)
    parser.add_argument("--n-ips", type=int, default=CyberConfig.n_ips)
    parser.add_argument("--n-devices", type=int, default=CyberConfig.n_devices)
    parser.add_argument("--attack-rate", type=float, default=CyberConfig.attack_rate)
    parser.add_argument("--show", type=int, default=8, help="Print N flagged examples")
    parser.add_argument(
        "--benchmark-events",
        type=int,
        default=None,
        help="Process only the first N events and estimate full runtime (recommended for 100M+)",
    )
    args = parser.parse_args()

    n_requested = int(args.n_events)
    n_process = int(args.benchmark_events) if args.benchmark_events is not None else n_requested
    n_process = min(n_process, n_requested)

    if args.benchmark_events is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} events (estimate full runtime).")

    event_iter = iter_synthetic_security_events(
        seed=int(args.seed),
        n_events=n_process,
        n_users=int(args.n_users),
        n_ips=int(args.n_ips),
        n_devices=int(args.n_devices),
        attack_rate=float(args.attack_rate),
    )

    t0 = time.perf_counter()
    m, examples = detect_threats_streaming(event_iter, n_users=int(args.n_users), keep_examples=int(args.show))
    dt = max(time.perf_counter() - t0, 1e-9)

    ev_per_sec = n_process / dt
    est_full_sec = (n_requested / ev_per_sec) if ev_per_sec > 0 else float("inf")
    print(
        f"\n[Cyber] processed={n_process:,} elapsed={dt:.2f}s speed={ev_per_sec:,.0f} ev/s "
        f"est_full={est_full_sec/60:.1f} min for {n_requested:,} events"
    )
    print(
        f"[Cyber] acc={m['accuracy']:.3f} prec_attack={m['precision_attack']:.3f} rec_attack={m['recall_attack']:.3f} "
        f"f1_attack={m['f1_attack']:.3f} fpr={m['fpr']:.3f} tp={int(m['tp'])} fp={int(m['fp'])} tn={int(m['tn'])} fn={int(m['fn'])}"
    )

    show = max(0, int(args.show))
    if show:
        print(f"\n[Cyber] flagged_examples={len(examples)}/{show}")
        for line in examples:
            print(line)


if __name__ == "__main__":
    main()
