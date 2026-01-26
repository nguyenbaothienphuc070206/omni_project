from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path


PATTERNS: list[str] = [
    "Shared-device fraud ring",
    "Shared-IP proxy cluster",
    "Shared-phone mule network",
    "Device reset / emulator farm",
    "Many accounts on one device",
    "One account on many devices",
    "Credential stuffing + low amounts",
    "Chargeback farming",
    "Promo abuse ring",
    "Referral abuse graph",
    "Merchant collusion",
    "Synthetic identity buildup",
    "Account takeover (ATO)",
    "Cash-out after warmup",
    "New-user burst signup",
    "Benign power-user (false positive risk)",
    "Family/shared household devices",
    "Corporate NAT (many users one IP)",
    "Bot-like periodic behavior",
    "Mixed ring + legit traffic camouflage",
]

TWISTS: list[str] = [
    "Amounts look normal (needs relational signals)",
    "Velocity is high but amounts are small",
    "IP rotates frequently (VPN/proxy)",
    "Device ID spoofing attempt",
    "Phone number recycled",
    "Time-of-day anomaly (night burst)",
    "Geo impossible travel pattern (simulated)",
    "Label noise / delayed chargebacks",
    "Cold-start user with little history",
    "Sparse graph (few edges)",
]

CONTEXTS: list[str] = [
    "Marketplace seller payouts",
    "E-commerce checkout",
    "Wallet top-up",
    "P2P transfer",
    "Loan application",
]


@dataclass(frozen=True)
class Case:
    id: int
    pattern: str
    twist: str
    context: str
    expected_route: str
    expected_explanation: str


@dataclass(frozen=True)
class ScoredCase:
    case: Case
    score: float


def _variant_suffix(case_id: int) -> str:
    """Deterministic light variation to create many unique cases.

    This keeps the base pattern/twist/context readable while enabling
    millions of distinct scenarios.
    """

    # 64-way stable variation
    v = (case_id * 2654435761) % 64
    if v == 0:
        return ""
    return f" (variant {v:02d})"


def iter_cases(n: int = 1000):
    """Yield cases without building a huge in-memory list."""

    n = int(n)
    for case_id in range(1, n + 1):
        p = PATTERNS[(case_id - 1) % len(PATTERNS)]
        t = TWISTS[((case_id - 1) // len(PATTERNS)) % len(TWISTS)]
        c = CONTEXTS[((case_id - 1) // (len(PATTERNS) * len(TWISTS))) % len(CONTEXTS)]

        # Make the text unique at scale without complicating the taxonomy.
        t2 = t + _variant_suffix(case_id)

        expected_route = "Stats: GRAY -> Graph: SCORE -> Decision"
        if "Benign" in p or "Family" in p or "Corporate NAT" in p:
            expected_route = "Stats: GRAY -> Graph: likely LEGIT -> Decision"

        expected_explanation = "Shortest path to risky anchor via shared device/ip/phone"
        if "Corporate NAT" in p:
            expected_explanation = "Explanation should avoid IP-only anchors (high false positives)"
        if "Family" in p:
            expected_explanation = "Explanation should show multi-signal evidence (not device-only)"

        yield Case(
            id=case_id,
            pattern=p,
            twist=t2,
            context=c,
            expected_route=expected_route,
            expected_explanation=expected_explanation,
        )


def generate_cases(n: int = 1000) -> list[Case]:
    """Generate a large, hackathon-ready case library.

    Notes:
    - These are *scenario descriptions* (not training data).
    - They help you communicate coverage, edge cases, and business logic.
    - Output is deterministic and intentionally concise.
    """

    # Keep a small helper for existing code paths; for large N, prefer iter_cases.
    return list(iter_cases(n))


def _score_case(c: Case) -> float:
    """Heuristic scoring for hackathon "strongest" cases.

    Strongest = high-signal fraud patterns + realistic adversarial twists + high-impact contexts,
    while still keeping some false-positive traps (important for credibility).
    """

    pattern_w = {
        "Shared-device fraud ring": 10,
        "Mixed ring + legit traffic camouflage": 10,
        "Account takeover (ATO)": 10,
        "Cash-out after warmup": 10,
        "Shared-IP proxy cluster": 9,
        "Shared-phone mule network": 9,
        "Promo abuse ring": 8,
        "Referral abuse graph": 8,
        "Synthetic identity buildup": 8,
        "Credential stuffing + low amounts": 7,
        "Chargeback farming": 7,
        "Device reset / emulator farm": 7,
        "Many accounts on one device": 7,
        "One account on many devices": 7,
        "Merchant collusion": 7,
        "Bot-like periodic behavior": 6,
        # False-positive traps (important to show maturity)
        "Family/shared household devices": 6,
        "Corporate NAT (many users one IP)": 6,
        "Benign power-user (false positive risk)": 6,
        "New-user burst signup": 6,
    }

    twist_w = {
        "Amounts look normal (needs relational signals)": 4,
        "IP rotates frequently (VPN/proxy)": 4,
        "Device ID spoofing attempt": 4,
        "Geo impossible travel pattern (simulated)": 4,
        "Velocity is high but amounts are small": 3,
        "Label noise / delayed chargebacks": 3,
        "Cold-start user with little history": 3,
        "Sparse graph (few edges)": 3,
        "Time-of-day anomaly (night burst)": 2,
        "Phone number recycled": 2,
    }

    context_w = {
        "P2P transfer": 4,
        "Wallet top-up": 4,
        "Marketplace seller payouts": 4,
        "Loan application": 3,
        "E-commerce checkout": 2,
    }

    score = 0.0
    score += pattern_w.get(c.pattern, 5)
    score += twist_w.get(c.twist, 1)
    score += context_w.get(c.context, 1)

    # Bonus: cases that explicitly demand "relational" reasoning are strong for this project.
    if "relational" in c.twist.lower() or "ring" in c.pattern.lower() or "graph" in c.pattern.lower():
        score += 1.0

    # Bonus: hard cases where stats alone is insufficient.
    if "Amounts look normal" in c.twist or "Sparse graph" in c.twist or "Cold-start" in c.twist:
        score += 0.5

    return score


def strongest_cases(*, k: int = 500) -> list[ScoredCase]:
    """Return top-k cases by heuristic score (deterministic)."""

    all_cases = generate_cases(n=1000)
    scored = [ScoredCase(case=c, score=_score_case(c)) for c in all_cases]
    scored.sort(key=lambda sc: (-sc.score, sc.case.id))
    return scored[:k]


def render_markdown(cases: list[Case], *, title: str | None = None) -> str:
    lines: list[str] = []
    title = title or f"Scenario Library ({len(cases)} cases)"
    lines.append(f"# {title}\n")
    lines.append(
        "These are concise *test/pitch scenarios* for hackathons. "
        "They describe what the system should do (route, decision, explanation), "
        "not ground-truth training labels.\n"
    )
    lines.append("## Cases\n")
    lines.append("| ID | Pattern | Twist | Context | Expected Route | Expected Explanation |")
    lines.append("|---:|---|---|---|---|---|")

    for c in cases:
        lines.append(
            f"| {c.id:04d} | {c.pattern} | {c.twist} | {c.context} | {c.expected_route} | {c.expected_explanation} |"
        )

    return "\n".join(lines) + "\n"


def render_scored_markdown(scored: list[ScoredCase], *, title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        "Top scenarios chosen by a simple, deterministic heuristic score. "
        "Use this list for pitching, judging, and focused demo testing.\n"
    )
    lines.append("## Cases\n")
    lines.append("| Rank | ID | Score | Pattern | Twist | Context | Expected Route | Expected Explanation |")
    lines.append("|---:|---:|---:|---|---|---|---|---|")

    for rank, sc in enumerate(scored, start=1):
        c = sc.case
        lines.append(
            f"| {rank:03d} | {c.id:04d} | {sc.score:.1f} | {c.pattern} | {c.twist} | {c.context} | {c.expected_route} | {c.expected_explanation} |"
        )

    return "\n".join(lines) + "\n"


def write_jsonl(out_path: Path, cases_iter) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for c in cases_iter:
            f.write(
                json.dumps(
                    {
                        "id": c.id,
                        "pattern": c.pattern,
                        "twist": c.twist,
                        "context": c.context,
                        "expected_route": c.expected_route,
                        "expected_explanation": c.expected_explanation,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def write_jsonl_gz(out_path: Path, cases_iter) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        for c in cases_iter:
            f.write(
                json.dumps(
                    {
                        "id": c.id,
                        "pattern": c.pattern,
                        "twist": c.twist,
                        "context": c.context,
                        "expected_route": c.expected_route,
                        "expected_explanation": c.expected_explanation,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def write_csv(out_path: Path, cases_iter) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pattern", "twist", "context", "expected_route", "expected_explanation"])
        for c in cases_iter:
            w.writerow([c.id, c.pattern, c.twist, c.context, c.expected_route, c.expected_explanation])
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hackathon scenario library.")
    parser.add_argument("--n", type=int, default=1000, help="Number of cases to generate (default: 1000)")
    parser.add_argument("--out", type=str, default="docs/cases.md", help="Output markdown path")
    parser.add_argument(
        "--mode",
        choices=["all", "strongest"],
        default="all",
        help="Generate all cases or the strongest subset (default: all)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of strongest cases when --mode=strongest (default: 500)",
    )
    parser.add_argument(
        "--format",
        choices=["md", "csv", "jsonl", "jsonl.gz"],
        default="md",
        help="Output format for --mode=all (default: md). For huge N, prefer jsonl.gz.",
    )
    args = parser.parse_args()

    if args.mode == "strongest":
        scored = strongest_cases(k=int(args.k))
        md = render_scored_markdown(scored, title=f"Strongest Scenarios ({len(scored)} cases)")
    else:
        n_total = int(args.n)
        out_path = Path(args.out)

        # For very large N, avoid Markdown tables (huge file, slow editors).
        if args.format == "md" and n_total > 50_000:
            raise SystemExit("For --n > 50000, use --format csv or jsonl.gz (Markdown would be too large).")

        if args.format == "md":
            cases = generate_cases(n=n_total)
            md = render_markdown(cases, title=f"Scenario Library ({len(cases)} cases)")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            n_written = len(cases)
            print(f"Wrote {n_written} cases to {out_path.as_posix()}")
            return

        cases_iter = iter_cases(n_total)
        if args.format == "csv":
            n_written = write_csv(out_path, cases_iter)
        elif args.format == "jsonl":
            n_written = write_jsonl(out_path, cases_iter)
        else:
            n_written = write_jsonl_gz(out_path, cases_iter)

        print(f"Wrote {n_written} cases to {out_path.as_posix()}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    n_written = len(scored)
    print(f"Wrote {n_written} cases to {out_path.as_posix()}")


if __name__ == "__main__":
    main()
