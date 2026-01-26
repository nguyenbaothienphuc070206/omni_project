from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticRingConfig:
    n_rings: int = 4
    ring_size: int = 10
    shared_devices_per_ring: int = 2
    shared_ips_per_ring: int = 2
    shared_phones_per_ring: int = 1


def generate_synthetic_transactions(
    *,
    seed: int,
    n_users: int,
    n_devices: int,
    n_ips: int,
    n_phones: int,
    n_transactions: int,
    base_fraud_rate: float,
    ring_cfg: SyntheticRingConfig | None = None,
) -> pd.DataFrame:
    """Generate a readable synthetic dataset.

    Columns:
      - transaction_id, ts, user_id, device_id, ip_id, phone_id, amount, label
    label = 1 means fraud, 0 means legit.

    Fraud pattern:
      - some users belong to collusive rings sharing devices/IPs/phones
      - ring transactions have higher amounts and bursty timestamps
    """

    rng = np.random.default_rng(seed)
    ring_cfg = ring_cfg or SyntheticRingConfig()

    users = np.arange(n_users)
    devices = np.arange(n_devices)
    ips = np.arange(n_ips)
    phones = np.arange(n_phones)

    ring_users: list[int] = []
    ring_shared_devices: list[list[int]] = []
    ring_shared_ips: list[list[int]] = []
    ring_shared_phones: list[list[int]] = []

    available_users = users.copy()
    rng.shuffle(available_users)

    cursor = 0
    for _ in range(ring_cfg.n_rings):
        if cursor + ring_cfg.ring_size > len(available_users):
            break
        ring = available_users[cursor : cursor + ring_cfg.ring_size].tolist()
        cursor += ring_cfg.ring_size

        ring_users.extend(ring)
        ring_shared_devices.append(rng.choice(devices, size=ring_cfg.shared_devices_per_ring, replace=False).tolist())
        ring_shared_ips.append(rng.choice(ips, size=ring_cfg.shared_ips_per_ring, replace=False).tolist())
        ring_shared_phones.append(rng.choice(phones, size=ring_cfg.shared_phones_per_ring, replace=False).tolist())

    ring_users_set = set(ring_users)

    base_ts = 1_700_000_000

    rows = []
    for tx_id in range(n_transactions):
        is_ring_tx = rng.random() < (base_fraud_rate * 1.6)
        if is_ring_tx and ring_users:
            ring_idx = int(rng.integers(0, len(ring_shared_devices)))
            user_id = int(rng.choice(np.array(list(ring_users_set))))
            device_id = int(rng.choice(ring_shared_devices[ring_idx]))
            ip_id = int(rng.choice(ring_shared_ips[ring_idx]))
            phone_id = int(rng.choice(ring_shared_phones[ring_idx]))

            # Bursty timestamps
            ts = base_ts + int(rng.integers(0, 24 * 60 * 60))
            if rng.random() < 0.5:
                ts = base_ts + int(rng.integers(0, 30 * 60))

            amount = float(rng.lognormal(mean=math.log(220.0), sigma=0.65))
            label = 1
        else:
            user_id = int(rng.choice(users))
            device_id = int(rng.choice(devices))
            ip_id = int(rng.choice(ips))
            phone_id = int(rng.choice(phones))

            ts = base_ts + int(rng.integers(0, 24 * 60 * 60))
            amount = float(rng.lognormal(mean=math.log(45.0), sigma=0.55))
            label = 1 if rng.random() < base_fraud_rate else 0

        rows.append(
            {
                "transaction_id": tx_id,
                "ts": ts,
                "user_id": user_id,
                "device_id": device_id,
                "ip_id": ip_id,
                "phone_id": phone_id,
                "amount": amount,
                "label": label,
            }
        )

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    # Make ring users more consistently fraudulent.
    if ring_users:
        df.loc[df["user_id"].isin(list(ring_users_set)), "label"] = (
            rng.random(df["user_id"].isin(list(ring_users_set)).sum()) < 0.75
        ).astype(int)

    return df
