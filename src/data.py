from __future__ import annotations

import math
from dataclasses import dataclass

from collections.abc import Iterator

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


def iter_synthetic_transactions(
    *,
    seed: int,
    n_users: int,
    n_devices: int,
    n_ips: int,
    n_phones: int,
    n_transactions: int,
    base_fraud_rate: float,
    ring_cfg: SyntheticRingConfig | None = None,
    hard_mode: bool = False,
) -> Iterator[tuple[int, int, int, int, int, int, float, int]]:
    """Stream transactions in (mostly) chronological order.

    Yields tuples:
      (transaction_id, ts, user_id, device_id, ip_id, phone_id, amount, label)

    This avoids constructing a huge list/DataFrame, so it can scale to
    very large `n_transactions` without RAM blowups.
    """

    rng = np.random.default_rng(seed)
    ring_cfg = ring_cfg or SyntheticRingConfig()

    users = np.arange(int(n_users))
    devices = np.arange(int(n_devices))
    ips = np.arange(int(n_ips))
    phones = np.arange(int(n_phones))

    # Build collusive rings once.
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

    ring_users_arr = np.array(ring_users, dtype=np.int64)
    ring_user_mask = bytearray(int(n_users))
    if ring_users_arr.size > 0:
        for u in ring_users_arr.tolist():
            ring_user_mask[int(u)] = 1

    # Hard-mode knobs (applied online).
    hub_devices: np.ndarray | None = None
    hub_ips: np.ndarray | None = None
    if hard_mode and n_devices > 0 and n_ips > 0:
        hub_devices = rng.choice(devices, size=max(2, int(0.01 * len(devices))), replace=False)
        hub_ips = rng.choice(ips, size=max(2, int(0.01 * len(ips))), replace=False)

    base_ts = 1_700_000_000
    ts = base_ts

    # Convert ring shared IDs to dense numpy arrays to avoid rng.choice(list) overhead.
    ring_shared_devices_arr = np.asarray(ring_shared_devices, dtype=np.int64) if ring_shared_devices else None
    ring_shared_ips_arr = np.asarray(ring_shared_ips, dtype=np.int64) if ring_shared_ips else None
    ring_shared_phones_arr = np.asarray(ring_shared_phones, dtype=np.int64) if ring_shared_phones else None

    # Hot-loop locals
    rand = rng.random
    integers = rng.integers
    lognormal = rng.lognormal

    n_users_i = int(n_users)
    n_devices_i = int(n_devices)
    n_ips_i = int(n_ips)
    n_phones_i = int(n_phones)
    ring_ok = ring_users_arr.size > 0

    for tx_id in range(int(n_transactions)):
        # Keep monotonic timestamps so streaming velocity is meaningful.
        ts += int(integers(1, 4))  # +1..+3 seconds

        is_ring_tx = (rand() < (base_fraud_rate * 1.6)) and ring_ok
        if is_ring_tx:
            ring_idx = int(integers(0, len(ring_shared_devices)))
            user_id = int(ring_users_arr[int(integers(0, ring_users_arr.size))])

            # Shared IDs are small fixed arrays; index by randint is faster than rng.choice.
            device_id = int(ring_shared_devices_arr[ring_idx, int(integers(0, ring_shared_devices_arr.shape[1]))])
            ip_id = int(ring_shared_ips_arr[ring_idx, int(integers(0, ring_shared_ips_arr.shape[1]))])
            phone_id = int(ring_shared_phones_arr[ring_idx, int(integers(0, ring_shared_phones_arr.shape[1]))])

            amount = float(lognormal(mean=math.log(220.0), sigma=0.65))
            label = 1
        else:
            user_id = int(integers(0, n_users_i))
            device_id = int(integers(0, n_devices_i))
            ip_id = int(integers(0, n_ips_i))
            phone_id = int(integers(0, n_phones_i))

            amount = float(lognormal(mean=math.log(45.0), sigma=0.55))

            # Ring users are more consistently fraudulent (no post-processing pass needed).
            if ring_user_mask[user_id]:
                label = 1 if rand() < 0.75 else 0
            else:
                label = 1 if rand() < base_fraud_rate else 0

        if hard_mode:
            # Benign hubs: rewire some legit traffic into shared device/IP hubs.
            if label == 0 and hub_devices is not None and hub_ips is not None:
                if rand() < 0.12:
                    device_id = int(hub_devices[int(integers(0, hub_devices.size))])
                    ip_id = int(hub_ips[int(integers(0, hub_ips.size))])

            # Fraud camouflage: some fraud uses random IDs to weaken ring signatures.
            if label == 1 and rand() < 0.20:
                device_id = int(integers(0, n_devices_i))
                ip_id = int(integers(0, n_ips_i))
                phone_id = int(integers(0, n_phones_i))

        yield (tx_id, ts, user_id, device_id, ip_id, phone_id, amount, int(label))
