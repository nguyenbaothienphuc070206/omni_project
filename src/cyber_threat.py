from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CyberConfig:
    n_events: int = 200_000
    n_users: int = 2000
    n_ips: int = 5000
    n_devices: int = 3500
    attack_rate: float = 0.02


def generate_synthetic_security_events(
    *,
    seed: int,
    n_events: int,
    n_users: int,
    n_ips: int,
    n_devices: int,
    attack_rate: float,
) -> dict[str, np.ndarray]:
    """Generate auth + network-like events.

    Each event is a single row with a binary label `is_attack`.
    We inject several attack patterns:
      - Credential stuffing: bursts of failed logins
      - Account takeover: new device + far geo + success
      - Data exfiltration: sudden bytes_out spike
    """

    rng = np.random.default_rng(seed)

    user_id = rng.integers(0, n_users, size=n_events)
    ip_id = rng.integers(0, n_ips, size=n_events)
    device_id = rng.integers(0, n_devices, size=n_events)

    # Relative timestamps (seconds). Not strictly used for modeling, but for "impossible travel" style features.
    ts = np.cumsum(rng.integers(0, 3, size=n_events)).astype(int)

    # Basic traffic/auth signals
    failed_logins = rng.poisson(lam=0.08, size=n_events).astype(int)
    success_login = (rng.random(n_events) < 0.35).astype(int)

    bytes_out = rng.lognormal(mean=np.log(60_000), sigma=0.8, size=n_events)
    bytes_in = rng.lognormal(mean=np.log(120_000), sigma=0.8, size=n_events)

    # "Distance" from user home geo (km): most are near, some are far.
    geo_km = rng.exponential(scale=30.0, size=n_events)

    # Attack injection
    is_attack = (rng.random(n_events) < attack_rate).astype(int)

    attack_type = np.zeros(n_events, dtype=int)
    # 1=stuffing, 2=takeover, 3=exfil
    pick = np.where(is_attack == 1)[0]
    if pick.size:
        attack_type[pick] = rng.integers(1, 4, size=pick.size)

        # Credential stuffing: many failed, mostly no success.
        stuffing = pick[attack_type[pick] == 1]
        if stuffing.size:
            failed_logins[stuffing] += rng.integers(4, 14, size=stuffing.size)
            success_login[stuffing] = (rng.random(stuffing.size) < 0.05).astype(int)
            geo_km[stuffing] += rng.exponential(scale=150.0, size=stuffing.size)

        # Account takeover: new device + far geo + success.
        takeover = pick[attack_type[pick] == 2]
        if takeover.size:
            success_login[takeover] = 1
            failed_logins[takeover] += rng.integers(1, 4, size=takeover.size)
            geo_km[takeover] += rng.exponential(scale=400.0, size=takeover.size)
            # Force device changes by randomizing devices more.
            device_id[takeover] = rng.integers(0, n_devices, size=takeover.size)

        # Data exfiltration: huge bytes_out.
        exfil = pick[attack_type[pick] == 3]
        if exfil.size:
            bytes_out[exfil] *= rng.uniform(20.0, 120.0, size=exfil.size)
            bytes_in[exfil] *= rng.uniform(0.2, 0.8, size=exfil.size)

    return {
        "ts": ts,
        "user_id": user_id,
        "ip_id": ip_id,
        "device_id": device_id,
        "failed_logins": failed_logins,
        "success_login": success_login,
        "bytes_out": bytes_out.astype(float),
        "bytes_in": bytes_in.astype(float),
        "geo_km": geo_km.astype(float),
        "is_attack": is_attack,
        "attack_type": attack_type,
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

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
    events: dict[str, np.ndarray],
    *,
    n_users: int,
    z_out_threshold: float = 4.0,
    geo_far_km: float = 300.0,
    fail_burst_ewma: float = 2.5,
    ewma_alpha: float = 0.05,
) -> tuple[np.ndarray, list[list[str]]]:
    """Streaming threat detection with explainable reasons.

    Output:
      - y_pred per event
      - reasons per event (list of short strings)

    Senior reasoning:
    - Use bounded per-user state (O(n_users) memory).
    - Keep detectors interpretable and fast (no heavy models needed for the demo).
    """

    y_pred = np.zeros(len(events["ts"]), dtype=int)
    reasons: list[list[str]] = [[] for _ in range(len(y_pred))]

    states = [OnlineUserState() for _ in range(int(n_users))]

    for i in range(len(y_pred)):
        uid = int(events["user_id"][i])
        st = states[uid]

        ts = int(events["ts"][i])
        dev = int(events["device_id"][i])
        failed = int(events["failed_logins"][i])
        success = int(events["success_login"][i])
        geo = float(events["geo_km"][i])
        outb = float(events["bytes_out"][i])

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

        if st.fail_ewma >= float(fail_burst_ewma) and failed >= 3:
            score += 2
            reasons[i].append("failed_login_burst")

        # ATO-ish: success from far geo and device changed
        if success == 1 and geo >= float(geo_far_km) and st.last_device != -1 and dev != st.last_device:
            score += 2
            reasons[i].append("new_device_far_geo")

        # Exfil-ish: bytes_out spike compared to user baseline
        if st.count >= 25 and z >= float(z_out_threshold):
            score += 2
            reasons[i].append("bytes_out_spike")

        # Mild heuristic: far geo without known context
        if geo >= float(geo_far_km) and success == 1:
            score += 1
            reasons[i].append("far_geo_success")

        # Final decision: small integer score threshold
        if score >= 3:
            y_pred[i] = 1

        st.last_device = dev
        st.last_ts = ts

    return y_pred, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Cybersecurity threat detection demo (technical layer, streaming + explainable)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-events", type=int, default=CyberConfig.n_events)
    parser.add_argument("--n-users", type=int, default=CyberConfig.n_users)
    parser.add_argument("--n-ips", type=int, default=CyberConfig.n_ips)
    parser.add_argument("--n-devices", type=int, default=CyberConfig.n_devices)
    parser.add_argument("--attack-rate", type=float, default=CyberConfig.attack_rate)
    parser.add_argument("--show", type=int, default=8, help="Print N flagged examples")
    args = parser.parse_args()

    events = generate_synthetic_security_events(
        seed=int(args.seed),
        n_events=int(args.n_events),
        n_users=int(args.n_users),
        n_ips=int(args.n_ips),
        n_devices=int(args.n_devices),
        attack_rate=float(args.attack_rate),
    )

    t0 = time.perf_counter()
    y_pred, reasons = detect_threats_streaming(events, n_users=int(args.n_users))
    dt = max(time.perf_counter() - t0, 1e-9)

    y_true = events["is_attack"].astype(int)
    m = _metrics(y_true, y_pred)

    print(
        f"\n[Cyber] events={int(args.n_events):,} users={int(args.n_users):,} attack_rate={float(args.attack_rate):.3f} "
        f"speed={int(args.n_events)/dt:,.0f} ev/s elapsed={dt:.2f}s"
    )
    print(
        f"[Cyber] acc={m['accuracy']:.3f} prec_attack={m['precision_attack']:.3f} rec_attack={m['recall_attack']:.3f} "
        f"f1_attack={m['f1_attack']:.3f} fpr={m['fpr']:.3f} tp={int(m['tp'])} fp={int(m['fp'])} tn={int(m['tn'])} fn={int(m['fn'])}"
    )

    show = max(0, int(args.show))
    if show:
        flagged = np.where(y_pred == 1)[0]
        print(f"\n[Cyber] flagged_examples={min(show, int(flagged.size))}/{int(flagged.size)}")
        for j in flagged[:show]:
            uid = int(events["user_id"][j])
            geo = float(events["geo_km"][j])
            outb = float(events["bytes_out"][j])
            failed = int(events["failed_logins"][j])
            success = int(events["success_login"][j])
            truth = int(events["is_attack"][j])
            why = ",".join(reasons[int(j)]) if reasons[int(j)] else "(no_reason)"
            print(
                f"- idx={int(j)} user={uid} truth={truth} success={success} failed={failed} geo_km={geo:.0f} bytes_out={outb:,.0f} reasons=[{why}]"
            )


if __name__ == "__main__":
    main()
