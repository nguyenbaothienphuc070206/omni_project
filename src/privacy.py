from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Iterable


def _hash_bytes(data: bytes) -> bytes:
    # Faster than SHA-256 in CPython stdlib for many small messages.
    return hashlib.blake2s(data, digest_size=32).digest()


def _hash_hex(data: bytes) -> str:
    return _hash_bytes(data).hex()


def _h_hex(*parts: str) -> str:
    return _hash_hex("|".join(parts).encode("utf-8"))


def _h_bytes(*parts: str) -> bytes:
    return _hash_bytes("|".join(parts).encode("utf-8"))


class _MerkleAccumulator:
    """Streaming Merkle accumulator.

    Keeps O(log n) state and avoids storing all leaves.
    Leaves are raw digest bytes.
    """

    def __init__(self) -> None:
        self._levels: list[bytes | None] = []
        self._count = 0

    def add(self, leaf_digest: bytes) -> None:
        h = leaf_digest
        lvl = 0
        while True:
            if lvl >= len(self._levels):
                self._levels.append(h)
                break
            cur = self._levels[lvl]
            if cur is None:
                self._levels[lvl] = h
                break
            self._levels[lvl] = None
            h = _hash_bytes(cur + h)
            lvl += 1
        self._count += 1

    def root_hex(self) -> str:
        if self._count <= 0:
            return _hash_hex(b"")

        # Fold remaining nodes upward. To mimic the "duplicate last" behavior
        # for odd counts, any unpaired node at a level is duplicated.
        node: bytes | None = None
        for lvl in range(len(self._levels)):
            cur = self._levels[lvl]
            if cur is None:
                continue
            if node is None:
                node = cur
            else:
                node = _hash_bytes(cur + node)

        if node is None:
            return _hash_hex(b"")

        # Duplicate upward for remaining height (standard padding).
        for _ in range(len(self._levels)):
            node = _hash_bytes(node + node)

        return node.hex()


def _encode_metadata_bytes(metadata: dict[str, str]) -> bytes:
    # Faster than json.dumps(sort_keys=True) and still deterministic.
    # Keys are fixed in iter_synthetic_ghost_transactions.
    return (
        "channel="
        + metadata.get("channel", "")
        + "|purpose="
        + metadata.get("purpose", "")
        + "|device="
        + metadata.get("device", "")
        + "|ip="
        + metadata.get("ip", "")
    ).encode("utf-8")


@dataclass(frozen=True)
class GhostConfig:
    n_transactions: int = 50_000
    n_nodes: int = 5


@dataclass(frozen=True)
class PrivateSettlementProof:
    """Demo-only proof artifact.

    This is NOT a real ZK proof.

    It models the API shape:
      - commitment binds the (hidden) balance
      - statement binds the transfer request
      - proof is a verifiable tag over the statement
    """

    commitment: str
    statement: str
    proof: str


class PrivateSettlementEngine:
    """Private settlement demo.

    Goal: show how a system can verify *sufficient balance* without exposing raw balances.

    Important: this implementation uses hash commitments and a server-held secret.
    It is a compliance/privacy demo, NOT cryptographic ZK.
    """

    def __init__(self, *, server_secret: bytes) -> None:
        self._secret = server_secret

    def commit_balance(self, *, account_id: str, balance_cents: int, salt: str) -> str:
        return _h_hex("bal", account_id, str(balance_cents), salt)

    def prove_sufficient_balance(
        self,
        *,
        account_id: str,
        balance_cents: int,
        amount_cents: int,
        salt: str,
        tx_id: str,
    ) -> PrivateSettlementProof:
        ok = balance_cents >= amount_cents
        commitment = self.commit_balance(account_id=account_id, balance_cents=balance_cents, salt=salt)
        statement = _h_hex("stmt", tx_id, account_id, str(amount_cents), commitment)
        proof = _hash_hex(self._secret + statement.encode("utf-8") + (b"1" if ok else b"0"))
        return PrivateSettlementProof(commitment=commitment, statement=statement, proof=proof)

    def verify(self, proof: PrivateSettlementProof) -> bool:
        # Demo verifier: the server can verify the tag.
        # A real ZK verifier would not need the secret.
        expected_prefix = _hash_hex(self._secret + proof.statement.encode("utf-8") + b"1")
        return proof.proof == expected_prefix


class ShadowMetadataStore:
    """Shadow metadata: hash + shard across nodes.

    We never store plaintext metadata; we store only hash shards.
    """

    def __init__(self, *, n_nodes: int) -> None:
        self.n_nodes = int(n_nodes)
        self._nodes: list[dict[str, str]] = [dict() for _ in range(self.n_nodes)]

    def shards_for_digest(self, *, digest_hex: str) -> list[str]:
        shard_len = max(len(digest_hex) // self.n_nodes, 1)
        return [digest_hex[i * shard_len : (i + 1) * shard_len] for i in range(self.n_nodes)]

    def put(self, *, tx_id: str, metadata: dict[str, str], store: bool) -> str:
        digest = _hash_hex(_encode_metadata_bytes(metadata))

        # Storing all shards for huge N is slow and memory-heavy.
        # We only persist shards when we intend to show examples.
        if store:
            shards = self.shards_for_digest(digest_hex=digest)
            nodes = self._nodes
            for i in range(self.n_nodes):
                nodes[i][tx_id] = shards[i]

        return digest

    def get_shards(self, *, tx_id: str) -> list[str]:
        return [node.get(tx_id, "") for node in self._nodes]


def iter_synthetic_ghost_transactions(*, seed: int, n: int) -> Iterable[tuple[str, str, str, int, dict[str, str]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    for i in range(int(n)):
        tx_id = f"tx_{i:08d}"
        sender = f"u{int(rng.integers(0, 2000))}"
        receiver = f"u{int(rng.integers(0, 2000))}"
        amount_cents = int(rng.lognormal(mean=8.5, sigma=0.7) * 10)
        meta = {
            "channel": str(rng.choice(["web", "mobile", "api"])),
            "purpose": str(rng.choice(["p2p", "invoice", "payroll", "merchant"])),
            "device": f"d{int(rng.integers(0, 8000))}",
            "ip": f"ip{int(rng.integers(0, 12000))}",
        }
        yield tx_id, sender, receiver, amount_cents, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 demo: Ghost Protocol (privacy & compliance, simplified)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=GhostConfig.n_transactions)
    parser.add_argument(
        "--benchmark",
        type=int,
        default=None,
        help="Process only the first N tx and estimate full runtime for large targets",
    )
    parser.add_argument("--nodes", type=int, default=GhostConfig.n_nodes)
    parser.add_argument("--show", type=int, default=5, help="Print N sample tx proofs + metadata shards")
    args = parser.parse_args()

    # Server-held secret simulates compliance gatekeeper.
    engine = PrivateSettlementEngine(server_secret=os.urandom(32))
    store = ShadowMetadataStore(n_nodes=int(args.nodes))

    # Demo balances: stored only as commitments per tx (no persistent ledger here).
    # We simulate a sender balance fresh per tx for a simple demo.
    ok = 0
    bad = 0
    merkle = _MerkleAccumulator()

    n_requested = int(args.n)
    n_process = int(args.benchmark) if args.benchmark is not None else n_requested
    n_process = min(n_process, n_requested)
    if args.benchmark is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} Ghost tx (estimate full runtime).")

    t0 = time.perf_counter()
    examples: list[str] = []

    hash_hex = _hash_hex
    h_bytes = _h_bytes

    show_n = int(args.show)
    for idx, (tx_id, sender, receiver, amount_cents, meta) in enumerate(
        iter_synthetic_ghost_transactions(seed=int(args.seed), n=int(n_process))
    ):
        # Synthetic "hidden" balance; most should pass.
        balance_cents = amount_cents + (50_000 if (idx % 10 != 0) else -5_000)
        salt = hash_hex((sender + ":" + tx_id).encode("utf-8"))[:16]

        proof = engine.prove_sufficient_balance(
            account_id=sender,
            balance_cents=balance_cents,
            amount_cents=amount_cents,
            salt=salt,
            tx_id=tx_id,
        )
        verified = engine.verify(proof)

        # Shadow metadata storage (hashed + sharded)
        want_store = (show_n > 0) and (len(examples) < show_n)
        meta_hash = store.put(tx_id=tx_id, metadata=meta, store=want_store)

        if verified:
            ok += 1
        else:
            bad += 1

        merkle.add(h_bytes("tx", tx_id, sender, receiver, str(amount_cents), proof.commitment, meta_hash))

        if len(examples) < show_n:
            shards = store.get_shards(tx_id=tx_id) if want_store else store.shards_for_digest(digest_hex=meta_hash)
            examples.append(
                f"- {tx_id} sender={sender} receiver={receiver} amount_cents={amount_cents} verified={verified} meta_hash={meta_hash[:12]}.. shards={shards}"
            )

    dt = max(time.perf_counter() - t0, 1e-9)
    tx_per_sec = n_process / dt
    est_full_sec = (n_requested / tx_per_sec) if tx_per_sec > 0 else float("inf")

    root = merkle.root_hex()

    scope = "full" if n_process == n_requested else "sample"
    print(
        f"\n[Ghost] tx={n_requested:,} processed={n_process:,} scope={scope} nodes={int(args.nodes)} "
        f"elapsed={dt:.2f}s speed={tx_per_sec:,.0f} tx/s est_full={est_full_sec/60:.1f} min"
    )
    print(f"[Ghost] private_settlement_verified={ok} rejected={bad} pass_rate={ok/max(ok+bad,1):.3f}")
    if scope == "full":
        print(f"[Ghost] blind_audit_anchor (merkle_root)={root}")
    else:
        print(f"[Ghost] blind_audit_anchor_sample (merkle_root)={root}")

    if examples:
        print("\n[Ghost] examples:")
        for e in examples:
            print(e)

    print("\n[Ghost] notes:")
    print("- This is a privacy/compliance demo API; not real ZK-SNARKs or homomorphic encryption.")
    print("- In production, replace commitments/proofs with audited cryptographic libraries and formal verification.")


if __name__ == "__main__":
    main()
