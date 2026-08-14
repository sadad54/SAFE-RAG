"""Step 05b -- recompute kappa on the revision-round batch (PREREGISTRATION.md
Section 8, Deviation 2).

    python scripts/05b_compute_revision_kappa.py --annotator adnan --second riyad

Both names must have complete label files for data/annotation/batch_02.jsonl
(python -m saferag.pilot.annotate --batch data/annotation/batch_02.jsonl
--annotator <name>), produced independently.

This does NOT touch the primary 150-item base rate -- that is still
scripts/05_compute_results.py, unchanged, over batch_01. This script's only
job is the kappa check the revision round exists to answer, plus the
disagreement detail that becomes the paper's qualitative material either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import base_parser, load_cfg, paths  # noqa: E402

from saferag.pilot.stats import bootstrap_kappa_interval, cohens_kappa, interpret_kappa  # noqa: E402
from saferag.utils.io import read_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402

log = get_logger("revision_kappa")


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    return {r["item_id"]: r for r in read_jsonl(path)}


def main() -> int:
    ap = base_parser("Recompute kappa on the 30-item revision batch.")
    ap.add_argument("--annotator", required=True, help="First annotator name")
    ap.add_argument("--second", required=True, help="Second annotator name")
    ap.add_argument(
        "--batch-stem", default="batch_02",
        help="Batch file stem (default batch_02, the revision round)",
    )
    args = ap.parse_args()

    cfg = load_cfg(args)
    p = paths(cfg)

    key_path = p["annotation"] / f"{args.batch_stem}_key.jsonl"
    if not key_path.exists():
        log.error("%s not found. Run scripts/04b_make_revision_batch.py first.", key_path)
        return 1
    pools = {r["item_id"]: r["pool"] for r in read_jsonl(key_path)}

    try:
        first = load_labels(p["annotation"] / f"labels_{args.annotator}_{args.batch_stem}.jsonl")
        second = load_labels(p["annotation"] / f"labels_{args.second}_{args.batch_stem}.jsonl")
    except FileNotFoundError as exc:
        log.error(str(exc))
        log.error(
            "Both annotators must finish before kappa can be computed. Check "
            "with the other annotator, or resume with `python -m saferag.pilot.annotate "
            "--batch data/annotation/%s.jsonl --annotator YOUR_NAME`.", args.batch_stem
        )
        return 1

    missing_first = set(pools) - set(first)
    missing_second = set(pools) - set(second)
    if missing_first:
        log.warning("%s: %d of %d item(s) still unlabelled.", args.annotator, len(missing_first), len(pools))
    if missing_second:
        log.warning("%s: %d of %d item(s) still unlabelled.", args.second, len(missing_second), len(pools))

    shared = sorted(set(first) & set(second))
    if len(shared) < 2:
        log.error("Only %d overlapping item(s); kappa needs at least 2.", len(shared))
        return 1

    labels_a = [first[i]["label"] for i in shared]
    labels_b = [second[i]["label"] for i in shared]
    kappa = cohens_kappa(labels_a, labels_b)
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / len(shared)

    # How much would kappa move on a different draw of the same 30 items? Bare
    # point estimates on a small revision batch invite exactly the "is this
    # solid" question -- answer it with a number instead of reassurance.
    kappa_ci = bootstrap_kappa_interval(labels_a, labels_b, seed=cfg.seed)

    print("\n" + "=" * 74)
    print("  REVISION-ROUND INTER-ANNOTATOR AGREEMENT")
    print("=" * 74)
    print(f"    batch                  {args.batch_stem}")
    print(f"    overlapping items      {len(shared)} of {len(pools)}")
    print(f"    observed agreement     {observed:.4f}")
    if kappa_ci.low == kappa_ci.low:  # not nan
        print(f"    Cohen's kappa          {kappa:.4f}   95% bootstrap CI [{kappa_ci.low:.4f}, {kappa_ci.high:.4f}]")
    else:
        print(f"    Cohen's kappa          {kappa:.4f}   (bootstrap CI not computable -- too many "
              "degenerate resamples at this n; treat the point estimate cautiously)")
    print(f"    reading                {interpret_kappa(kappa)}")

    disagreements = [i for i in shared if first[i]["label"] != second[i]["label"]]
    print(f"\n  DISAGREEMENTS ({len(disagreements)} of {len(shared)})")
    for i in disagreements:
        a_rec, b_rec = first[i], second[i]
        print(f"\n    item_id     {i}")
        print(f"    pool        {pools.get(i, '?')}")
        print(f"    {args.annotator:<10}  {a_rec['label']:<3}  note: {a_rec.get('note') or '(none)'}")
        print(f"    {args.second:<10}  {b_rec['label']:<3}  note: {b_rec.get('note') or '(none)'}")

    out = p["runs"] / "revision_kappa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "batch": args.batch_stem,
                "n_shared": len(shared),
                "n_total": len(pools),
                "observed_agreement": observed,
                "cohens_kappa": kappa,
                "cohens_kappa_ci_low": None if kappa_ci.low != kappa_ci.low else kappa_ci.low,
                "cohens_kappa_ci_high": None if kappa_ci.high != kappa_ci.high else kappa_ci.high,
                "cohens_kappa_ci_confidence": kappa_ci.confidence,
                "reading": interpret_kappa(kappa),
                "disagreements": [
                    {
                        "item_id": i,
                        "pool": pools.get(i),
                        args.annotator: {"label": first[i]["label"], "note": first[i].get("note")},
                        args.second: {"label": second[i]["label"], "note": second[i].get("note")},
                    }
                    for i in disagreements
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Saved {out}")

    if kappa >= 0.50:
        print("\n  >= 0.50: per PREREGISTRATION.md Section 8, the construct is now "
              "reliably distinguishable (or usable -- check which band). The base "
              "rate can return to being reported as a primary result. Update "
              "PROJECT_STATE.md and PREREGISTRATION.md Section 11 with this outcome.")
    else:
        print("\n  < 0.50: per PREREGISTRATION.md Section 8, the one permitted revision "
              "did not clear the threshold. The natural-occurrence measurement is "
              "abandoned per the pre-registered rule; move to Pivot A. Record this "
              "outcome in Section 11 before doing anything else.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
