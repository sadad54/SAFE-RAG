"""Step 05c -- RQ-a: does the deceptive-grounding rate differ by question type?

    python scripts/05c_rqa_breakdown.py --annotator adnan

Implements PREREGISTRATION.md Section 2 (RQ-a) under the operational definition
declared in Section 11, Amendment 2: cross-reference = more than one gold
passage; single-reference = exactly one. Declared BEFORE this script was run
against real labels, specifically to keep the category boundary from being
chosen by whichever split makes the numbers look best.

This is a DESCRIPTIVE secondary cut over the same 150 primary labels used for
the headline `r` -- unweighted (not re-stratified by S3 pool, which would
fragment n=150 into cells too small to interpret). It does not change the
primary stratified estimate in scripts/05_compute_results.py.

Output: runs/<run>/rqa_breakdown.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import base_parser, load_cfg, paths  # noqa: E402

from saferag.pilot.stats import wilson_interval  # noqa: E402
from saferag.utils.io import read_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402

log = get_logger("rqa_breakdown")

CROSS_REFERENCE = "cross_reference"
SINGLE_REFERENCE = "single_reference"


def load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    return {r["item_id"]: r["label"] for r in read_jsonl(path)}


def main() -> int:
    ap = base_parser("RQ-a: deceptive-grounding rate by question type (cross-reference vs single).")
    ap.add_argument("--annotator", required=True, help="Primary annotator name")
    args = ap.parse_args()

    cfg = load_cfg(args)
    p = paths(cfg)

    filtered_path = p["interim"] / "filtered.jsonl"
    if not filtered_path.exists():
        log.error("%s not found. Run scripts/03_run_filters.py first.", filtered_path)
        return 1

    # gold_passage_ids is already carried on every survivor record (used for the
    # S3 screen) -- no new data engineering needed, just a different cut of what
    # scripts/03_run_filters.py already wrote.
    n_gold: dict[str, int] = {}
    for r in read_jsonl(filtered_path):
        if "item_id" in r:
            n_gold[r["item_id"]] = len(r.get("gold_passage_ids", []))

    try:
        labels = load_labels(p["annotation"] / "labels_{}_batch_01.jsonl".format(args.annotator))
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    groups: dict[str, list[str]] = {CROSS_REFERENCE: [], SINGLE_REFERENCE: []}
    missing_gold = 0
    for item_id, label in labels.items():
        if label == "NA":
            continue
        gold_count = n_gold.get(item_id)
        if gold_count is None:
            missing_gold += 1
            continue
        group = CROSS_REFERENCE if gold_count > 1 else SINGLE_REFERENCE
        groups[group].append(label)

    if missing_gold:
        log.warning(
            "%d labelled item(s) not found in filtered.jsonl (stale batch vs. "
            "re-run pipeline?); excluded from the breakdown.", missing_gold
        )

    print("\n" + "=" * 74)
    print("  RQ-a -- DECEPTIVE GROUNDING RATE BY QUESTION TYPE")
    print("  (PREREGISTRATION.md Section 2 / Section 11 Amendment 2)")
    print("=" * 74)

    results: dict[str, dict] = {}
    for name, labs in groups.items():
        n = len(labs)
        b = sum(1 for lab in labs if lab == "B")
        interval = wilson_interval(b, n) if n else None
        results[name] = {
            "n": n,
            "n_B": b,
            "rate": None if interval is None else interval.point,
            "low": None if interval is None else interval.low,
            "high": None if interval is None else interval.high,
        }
        label = "cross-reference (>1 gold passage)" if name == CROSS_REFERENCE else "single-reference (1 gold passage)"
        if interval is None:
            print(f"    {label:<38} n=0, no items in this group")
        else:
            print(f"    {label:<38} n={n:<4} B={b:<4} rate={interval.as_percent()}")

    print(
        "\n  Descriptive secondary cut, unweighted across S3 pools -- does NOT "
        "replace the primary stratified r in runs/*/results.json.\n"
    )

    out = p["runs"] / "rqa_breakdown.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "definition": "cross_reference = len(gold_passage_ids) > 1 (Amendment 2)",
                "annotator": args.annotator,
                "n_missing_gold_lookup": missing_gold,
                "groups": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Saved {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
