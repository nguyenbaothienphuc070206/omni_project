from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class Phase7Config:
    n_blocks: int = 200_000


class HashSig:
    """Demo-only hash-based signature-like primitive.

    This is NOT real post-quantum crypto.

    Goal: show crypto-agility plumbing (swap algorithm, keep same ledger API).
    """

    def __init__(self, *, secret: bytes) -> None:
        self._secret = secret

    def sign(self, msg: bytes) -> str:
        return _sha256(self._secret + msg)

    def verify(self, msg: bytes, sig: str) -> bool:
        return sig == _sha256(self._secret + msg)


class QuantumReadyLedger:
    """Minimal append-only ledger with pluggable signing scheme."""

    def __init__(self, *, signer: HashSig) -> None:
        self._signer = signer
        self._prev = _sha256(b"")

    def append(self, *, height: int, payload: bytes) -> tuple[str, str]:
        header = f"h={height}|prev={self._prev}".encode("utf-8")
        msg = header + b"|" + payload
        sig = self._signer.sign(msg)
        block_id = _sha256(msg + sig.encode("utf-8"))
        self._prev = block_id
        return block_id, sig

    def verify_block(self, *, height: int, payload: bytes, prev: str, block_id: str, sig: str) -> bool:
        header = f"h={height}|prev={prev}".encode("utf-8")
        msg = header + b"|" + payload
        if not self._signer.verify(msg, sig):
            return False
        return block_id == _sha256(msg + sig.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 demo: Quantum-Resistant Ledger (crypto-agility plumbing, demo-only)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-blocks", type=int, default=Phase7Config.n_blocks)
    parser.add_argument(
        "--benchmark-blocks",
        type=int,
        default=None,
        help="Process only first N blocks and estimate full runtime",
    )
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    n_requested = int(args.n_blocks)
    n_process = int(args.benchmark_blocks) if args.benchmark_blocks is not None else n_requested
    n_process = min(n_process, n_requested)

    if args.benchmark_blocks is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} blocks (estimate full runtime).")

    signer = HashSig(secret=_sha256(b"phase7" + str(args.seed).encode("utf-8")).encode("utf-8"))
    led = QuantumReadyLedger(signer=signer)

    ok = 0
    bad = 0
    examples: list[str] = []

    t0 = time.perf_counter()
    prev = _sha256(b"")
    for h in range(int(n_process)):
        payload = f"tx_root={_sha256(f'{args.seed}:{h}'.encode('utf-8'))}".encode("utf-8")
        block_id, sig = led.append(height=h, payload=payload)

        if led.verify_block(height=h, payload=payload, prev=prev, block_id=block_id, sig=sig):
            ok += 1
        else:
            bad += 1

        if len(examples) < int(args.show):
            examples.append(f"- h={h} block={block_id[:16]}.. sig={sig[:12]}..")

        prev = block_id

    dt = max(time.perf_counter() - t0, 1e-9)
    speed = n_process / dt
    est_full_sec = (n_requested / speed) if speed > 0 else float("inf")

    scope = "full" if n_process == n_requested else "sample"
    print(
        f"\n[Phase7] blocks={n_requested:,} processed={n_process:,} scope={scope} elapsed={dt:.2f}s "
        f"speed={speed:,.0f} blocks/s est_full={est_full_sec/60:.1f} min"
    )
    print(f"[Phase7] verified={ok} rejected={bad} pass_rate={ok/max(ok+bad,1):.3f}")

    # Expose a ledger anchor that can be used like an audit root.
    print(f"[Phase7] ledger_anchor={prev}")

    if examples:
        print("\n[Phase7] examples:")
        for e in examples:
            print(e)

    print("\n[Phase7] notes:")
    print("- Demo-only: this is crypto-agility plumbing using hash-based tags.")
    print("- For real post-quantum security, replace with audited PQ signatures (e.g., Dilithium) and formal review.")


if __name__ == "__main__":
    main()
