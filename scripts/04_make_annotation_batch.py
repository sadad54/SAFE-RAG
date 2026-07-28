"""Step 04 -- draw the stratified annotation batch.

    python scripts/04_make_annotation_batch.py

Writes:
  data/annotation/batch_01.jsonl          the 150 items to label (pool-blinded)
  data/annotation/batch_01_double.jsonl   the 50 for the second annotator
  data/annotation/batch_01_key.jsonl      pool membership, for analysis only
"""

from __future__ import annotations

import sys

from _common import base_parser, paths  # noqa: E402

from saferag.config import load_config  # noqa: E402
from saferag.pilot.sample import double_annotation_subset, stratified_sample  # noqa: E402
from saferag.utils.io import read_jsonl, write_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402
from saferag.utils.provenance import stamp  # noqa: E402

log = get_logger("batch")


def main() -> int:
    ap = base_parser("Draw the stratified annotation batch.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = paths(cfg)

    src = p["interim"] / "filtered.jsonl"
    if not src.exists():
        log.error("%s not found. Run scripts/03_run_filters.py first.", src)
        return 1

    survivors = [r for r in read_jsonl(src) if r.get("stage") == "survivor"]
    log.info("Survivors available: %d", len(survivors))
    if not survivors:
        log.error("No survivors to sample. Check the funnel from step 03.")
        return 1

    batch, report = stratified_sample(
        survivors,
        n_candidate=cfg.sampling.n_candidate,
        n_control=cfg.sampling.n_control,
        seed=cfg.seed,
    )
    double = double_annotation_subset(batch, n=cfg.sampling.n_double_annotated, seed=cfg.seed)

    prov = stamp(
        script="04_make_annotation_batch.py",
        config=cfg,
        config_path=str(args.config),
        seed=cfg.seed,
        sampling_report=report,
    )

    # What the annotator sees: no pool, no gold ids.
    visible_keys = {"item_id", "question", "answer", "cited_passages"}
    blinded = [{k: v for k, v in item.items() if k in visible_keys} for item in batch]
    double_ids = {d["item_id"] for d in double}
    blinded_double = [item for item in blinded if item["item_id"] in double_ids]

    # The key, for analysis after annotation is complete.
    key = [{"item_id": item["item_id"], "pool": item["_pool"]} for item in batch]

    write_jsonl(p["annotation"] / "batch_01.jsonl", blinded, provenance=prov)
    write_jsonl(p["annotation"] / "batch_01_double.jsonl", blinded_double, provenance=prov)
    write_jsonl(p["annotation"] / "batch_01_key.jsonl", key, provenance=prov)

    print("\n  SAMPLING")
    for k, v in report.items():
        print(f"    {k:<28} {v}")
    if report["shortfall_candidate"] or report["shortfall_control"]:
        print("\n    NOTE: a pool was smaller than its registered allocation.")
        print("    This is permitted; record it in PREREGISTRATION.md Section 11.")

    print(f"\n  Second annotator subset: {len(blinded_double)} items")
    print(f"\n  Wrote batches to {p['annotation']}")
    print("\n  Now annotate. Read docs/ANNOTATION_GUIDELINES.md first, then:")
    print("    python -m saferag.pilot.annotate --batch data/annotation/batch_01.jsonl "
          "--annotator YOUR_NAME")
    print("\n  Second annotator, independently, no discussion until both are done:")
    print("    python -m saferag.pilot.annotate --batch data/annotation/batch_01_double.jsonl "
          "--annotator SECOND_NAME\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
