from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatsConfig:
    amount_z_threshold: float = 3.0
    velocity_window_seconds: int = 10 * 60
    velocity_threshold: int = 6

    clear_fraud_score: float = 0.90
    clear_legit_score: float = 0.10


@dataclass(frozen=True)
class DataConfig:
    seed: int = 7
    n_users: int = 300
    n_devices: int = 120
    n_ips: int = 160
    n_phones: int = 140
    n_transactions: int = 6000
    base_fraud_rate: float = 0.08


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    heads: int = 2
    dropout: float = 0.2

    gray_only_training: bool = False
    device: str = "cpu"
