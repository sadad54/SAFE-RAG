"""Step 06 -- does typing the failure carry any information?

    python scripts/06_random_routing_check.py --stub

THIS IS THE CHEAPEST AND MOST IMPORTANT CHECK IN THE PROJECT.

SAFE-RAG's premise is that diagnosing *which* condition failed lets you pick a
better repair. The null hypothesis is that it does not: that any repair budget
spent at all produces the same gain, whatever action it is spent on.

This script pits three arms against each other at an IDENTICAL call budget:

    typed   -- schema failure -> regenerate; attribution failure -> re-retrieve
    random  -- pick regenerate or re-retrieve at random, ignoring the diagnosis
    none    -- no repair (floor)

If ``typed`` does not beat ``random``, the diagnosis is noise and the whole
architecture is an expensive coin flip. Better to learn that in a day than in
month four.

Run this on 200 items as soon as the pipeline exists.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass

from _common import base_parser, paths  # noqa: E402

from saferag.config import load_config  # noqa: E402
from saferag.pilot.stats import wilson_interval  # noqa: E402
from saferag.utils.io import read_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402

log = get_logger("routing")

REGENERATE = "regenerate"
RERETRIEVE = "re-retrieve"


@dataclass
class RepairOutcome:
    item_id: str
    arm: str
    action: str
    calls_used: int
    repaired: bool


def diagnose(record: dict) -> str | None:
    """Return the diagnosed condition, or None if the item did not fail."""
    if not record.get("s1_valid", False):
        return "schema_invalid"
    if not record.get("s2_passed", False):
        return "attribution_or_faithfulness"
    return None


def typed_action(condition: str) -> str:
    """SAFE-RAG's routing policy."""
    return REGENERATE if condition == "schema_invalid" else RERETRIEVE


def main() -> int:
    ap = base_parser("Typed vs random routing at a matched budget.")
    ap.add_argument("--n", type=int, default=200, help="Items to test")
    ap.add_argument("--budget", type=int, default=1, help="Repair calls per item, all arms")
    ap.add_argument(
        "--stub",
        action="store_true",
        help="Simulate repair outcomes instead of calling a model. Wires up the "
             "comparison so you can verify the harness; NOT a result.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = paths(cfg)

    src = p["interim"] / "filtered.jsonl"
    if not src.exists():
        log.error("%s not found. Run scripts/03_run_filters.py first.", src)
        return 1

    failures = [r for r in read_jsonl(src) if diagnose(r) is not None][: args.n]
    if not failures:
        log.error("No failed items found. Nothing to repair.")
        return 1
    log.info("Testing %d failed items, budget %d call(s) per item", len(failures), args.budget)

    if not args.stub:
        log.error(
            "Real repair execution is not implemented yet.\n"
            "  Implement it by: re-running generation for REGENERATE, and re-running\n"
            "  retrieval with an entity-constrained query for RE-RETRIEVE, then\n"
            "  re-applying the step-03 filters to the repaired output.\n"
            "  Run with --stub first to confirm the harness and the arithmetic."
        )
        return 2

    # --- Simulation ------------------------------------------------------------
    # Deliberately optimistic for the typed arm: the right action repairs 60% of
    # the time, the wrong action 20%. If typed does not beat random even under
    # these generous assumptions, the harness itself is wrong.
    rng = random.Random(cfg.seed)
    P_RIGHT, P_WRONG = 0.60, 0.20

    outcomes: list[RepairOutcome] = []
    for rec in failures:
        condition = diagnose(rec)
        correct = typed_action(condition)

        for arm in ("typed", "random", "none"):
            if arm == "none":
                outcomes.append(RepairOutcome(rec["item_id"], arm, "none", 0, False))
                continue
            action = correct if arm == "typed" else rng.choice([REGENERATE, RERETRIEVE])
            success = rng.random() < (P_RIGHT if action == correct else P_WRONG)
            outcomes.append(
                RepairOutcome(rec["item_id"], arm, action, args.budget, success)
            )

    print("\n" + "=" * 74)
    print(f"  TYPED vs RANDOM ROUTING   (n={len(failures)}, budget={args.budget} call/item)")
    print("=" * 74)
    if args.stub:
        print("  SIMULATED OUTCOMES -- harness check only, not a result.\n")

    summary = {}
    for arm in ("none", "random", "typed"):
        rows = [o for o in outcomes if o.arm == arm]
        repaired = sum(o.repaired for o in rows)
        calls = sum(o.calls_used for o in rows)
        ci = wilson_interval(repaired, len(rows))
        summary[arm] = {"repair_rate": ci.point, "low": ci.low, "high": ci.high, "calls": calls}
        print(f"    {arm:<8} repair rate {ci.as_percent():<28} calls used {calls}")

    budgets = {v["calls"] for k, v in summary.items() if k != "none"}
    print(f"\n    budget matched across repair arms: {'YES' if len(budgets) == 1 else 'NO'}")

    typed_r, random_r = summary["typed"]["repair_rate"], summary["random"]["repair_rate"]
    delta = typed_r - random_r
    print(f"    typed - random = {delta:+.4f}")

    print("\n" + "=" * 74)
    if summary["typed"]["low"] > summary["random"]["high"]:
        print("  TYPED BEATS RANDOM -- intervals do not overlap.")
        print("  The diagnosis carries information. Proceed with the framework.")
    elif delta > 0:
        print("  TYPED LEADS, BUT INTERVALS OVERLAP.")
        print("  Increase n before drawing a conclusion. Do not proceed on this alone.")
    else:
        print("  TYPED DOES NOT BEAT RANDOM.")
        print("  The diagnosis carries no information at this budget. Do NOT build")
        print("  the framework until you understand why.")
    print("=" * 74 + "\n")

    out = p["runs"] / "routing_check.json"
    out.write_text(
        json.dumps(
            {"n": len(failures), "budget": args.budget, "simulated": args.stub,
             "arms": summary, "typed_minus_random": delta},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Saved {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
