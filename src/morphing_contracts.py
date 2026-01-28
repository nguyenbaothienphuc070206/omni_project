from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _h(*parts: str) -> str:
    return _sha256("|".join(parts).encode("utf-8"))


@dataclass(frozen=True)
class Phase6Config:
    n_calls: int = 200_000
    rotate_every_seconds: int = 24 * 60 * 60


class VersionedABIRegistry:
    """Transparent ABI registry.

    This is a compliance-safe alternative to "morphing bytecode":
    - code does not self-mutate
    - interface versions are explicit, auditable, and signed (demo)
    """

    def __init__(self) -> None:
        self._versions: dict[int, dict[str, str]] = {}

    def publish(self, *, version: int, signatures: dict[str, str]) -> None:
        self._versions[int(version)] = dict(signatures)

    def resolve(self, *, version: int, fn: str) -> str:
        sigs = self._versions.get(int(version))
        if not sigs or fn not in sigs:
            raise KeyError(f"unknown function: v{version}.{fn}")
        return sigs[fn]


class RotatingCapability:
    """Time-rotating capability token (demo).

    This is NOT meant to hide behavior from auditors.
    It models operational key rotation (like short-lived API tokens).
    """

    def __init__(self, *, secret: bytes, rotate_every_seconds: int) -> None:
        self._secret = secret
        self._period = int(rotate_every_seconds)

    def token(self, *, epoch_seconds: int) -> str:
        slot = epoch_seconds // max(self._period, 1)
        return _sha256(self._secret + str(slot).encode("utf-8"))

    def verify(self, *, token: str, epoch_seconds: int) -> bool:
        return token == self.token(epoch_seconds=epoch_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 demo: Morphing Smart Contracts (compliance-safe: versioned ABI + rotating capability)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-calls", type=int, default=Phase6Config.n_calls)
    parser.add_argument(
        "--benchmark-calls",
        type=int,
        default=None,
        help="Process only first N calls and estimate full runtime",
    )
    parser.add_argument("--rotate-every", type=int, default=Phase6Config.rotate_every_seconds)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    n_requested = int(args.n_calls)
    n_process = int(args.benchmark_calls) if args.benchmark_calls is not None else n_requested
    n_process = min(n_process, n_requested)

    if args.benchmark_calls is not None and n_process < n_requested:
        print(f"\n[Benchmark] Processing {n_process:,} of {n_requested:,} calls (estimate full runtime).")

    reg = VersionedABIRegistry()

    # Publish two explicit, auditable versions.
    reg.publish(
        version=1,
        signatures={
            "swap": "swap(address,uint256,uint256)",
            "add_liquidity": "add_liquidity(address,uint256)",
        },
    )
    reg.publish(
        version=2,
        signatures={
            "swap": "swap_v2(address,uint256,uint256,bytes32)",
            "add_liquidity": "add_liquidity(address,uint256,uint16)",
        },
    )

    cap = RotatingCapability(secret=_sha256(b"phase6" + str(args.seed).encode("utf-8")).encode("utf-8"), rotate_every_seconds=int(args.rotate_every))

    ok = 0
    bad = 0
    examples: list[str] = []

    t0 = time.perf_counter()
    now = int(time.time())

    # Deterministic call stream: version switches every 10k calls.
    for i in range(int(n_process)):
        version = 1 if (i // 10_000) % 2 == 0 else 2
        fn = "swap" if (i % 3 != 0) else "add_liquidity"

        sig = reg.resolve(version=version, fn=fn)

        # Use rotating capability as an auth gate (demo)
        tok = cap.token(epoch_seconds=now + i)
        verified = cap.verify(token=tok, epoch_seconds=now + i)

        # Always true for demo, but we keep the branch for realism.
        if verified:
            ok += 1
        else:
            bad += 1

        if len(examples) < int(args.show):
            examples.append(
                f"- i={i} v={version} fn={fn} sig={sig} token={tok[:12]}.. verified={verified}"
            )

    dt = max(time.perf_counter() - t0, 1e-9)
    speed = n_process / dt
    est_full_sec = (n_requested / speed) if speed > 0 else float("inf")

    scope = "full" if n_process == n_requested else "sample"
    print(
        f"\n[Phase6] calls={n_requested:,} processed={n_process:,} scope={scope} elapsed={dt:.2f}s "
        f"speed={speed:,.0f} calls/s est_full={est_full_sec/60:.1f} min"
    )
    print(f"[Phase6] verified={ok} rejected={bad} pass_rate={ok/max(ok+bad,1):.3f} rotate_every_s={int(args.rotate_every)}")

    if examples:
        print("\n[Phase6] examples:")
        for e in examples:
            print(e)

    print("\n[Phase6] notes:")
    print("- Compliance-safe demo: ABI versions are explicit (auditable).")
    print("- Rotating capability models key rotation, not audit evasion or obfuscation.")


if __name__ == "__main__":
    main()
