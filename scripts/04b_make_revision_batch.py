"""Step 04b -- draw the revision-round batch (PREREGISTRATION.md Section 8,
Deviation 2).

    python scripts/04b_make_revision_batch.py

Draws 30 items FRESH -- disjoint from every item already sampled into a
previous batch -- stratified proportionally to the registered allocation
(16 candidate_recoverable / 4 candidate_unrecoverable / 10 control, i.e. 30/150
of 80/20/50). Both annotators label ALL 30 of these, independently and in
parallel; this is a full double-annotation of the revision batch, not a split,
because Cohen's kappa requires the same items judged twice.

Writes:
  data/annotation/batch_02.jsonl       the 30 items to label (pool-blinded)
  data/annotation/batch_02_key.jsonl   pool membership, for analysis only

batch_02.jsonl embeds ADGM passage text and is gitignored on purpose (see
.gitignore's comment on data/annotation/batch_*.jsonl) -- it never goes through
git, exactly like batch_01.jsonl before it. Send it to the second annotator by
whatever direct channel got them batch_01_double.jsonl the first time. Only
batch_02_key.jsonl (ids + pool, no corpus text) is safe to commit.

Then both annotators run, at the same time, on their own copies of batch_02.jsonl:
  python -m saferag.pilot.annotate --batch data/annotation/batch_02.jsonl --annotator adnan
  python -m saferag.pilot.annotate --batch data/annotation/batch_02.jsonl --annotator riyad

Then: python scripts/05b_compute_revision_kappa.py --annotator adnan --second riyad
"""

from __future__ import annotations

import sys

from _common import base_parser, load_cfg, paths  # noqa: E402

from saferag.pilot.sample import scale_allocation, stratified_sample  # noqa: E402
from saferag.utils.io import read_jsonl, write_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402
from saferag.utils.provenance import stamp  # noqa: E402

log = get_logger("revision_batch")


def _already_sampled_ids(annotation_dir) -> set[str]:  # noqa: ANN001
    """Every item_id that has appeared in any previous batch's key file.

    Globs rather than hardcoding batch_01_key.jsonl so a second revision round,
    if one were ever needed, would still draw fresh items against everything
    that came before it -- though Section 8 permits at most one.
    """
    ids: set[str] = set()
    for key_path in sorted(annotation_dir.glob("batch_*_key.jsonl")):
        for r in read_jsonl(key_path):
            ids.add(r["item_id"])
    return ids


def main() -> int:
    ap = base_parser("Draw the revision-round batch (30 fresh items, both annotators label all).")
    ap.add_argument(
        "--n", type=int, default=30,
        help=(
            "Revision batch size. Default 30 is the number PREREGISTRATION.md "
            "Section 8 commits to -- changing this is itself a further deviation "
            "and should get its own dated entry in Section 11 if you do."
        ),
    )
    ap.add_argument(
        "--stem", default="batch_02",
        help=(
            "Output filename stem. Default batch_02 (the registered revision "
            "round). Change this when drawing a FURTHER batch on top of an "
            "already-completed one -- e.g. --stem batch_03 for a size "
            "extension -- so the new draw does not overwrite the prior "
            "batch's files. _already_sampled_ids() excludes every item in "
            "every existing batch_*_key.jsonl regardless of --stem, so the "
            "new batch is always disjoint from all of them."
        ),
    )
    args = ap.parse_args()

    cfg = load_cfg(args)
    p = paths(cfg)

    src = p["interim"] / "filtered.jsonl"
    if not src.exists():
        log.error("%s not found. Run scripts/03_run_filters.py first.", src)
        return 1

    survivors = [r for r in read_jsonl(src) if r.get("stage") == "survivor"]
    log.info("Survivors in %s: %d", src, len(survivors))

    already = _already_sampled_ids(p["annotation"])
    log.info("Already sampled in a previous batch: %d item(s), excluded.", len(already))

    fresh = [r for r in survivors if r["item_id"] not in already]
    log.info("Fresh survivors available: %d", len(fresh))
    if not fresh:
        log.error("No fresh survivors left to sample. Check the funnel from step 03.")
        return 1

    if args.n != 30:
        log.warning(
            "--n %d does not match the registered revision size of 30. Record "
            "this deviation in PREREGISTRATION.md Section 11 if you keep it.",
            args.n,
        )

    allocation = scale_allocation(dict(cfg.sampling.allocation), args.n)
    batch, report = stratified_sample(fresh, allocation=allocation, seed=cfg.seed)

    prov = stamp(
        script="04b_make_revision_batch.py",
        config=cfg,
        config_path=str(args.config),
        seed=cfg.seed,
        sampling_report=report,
        excluded_ids=len(already),
        revision_round=True,
    )

    # Same visible fields as the primary batch -- no pool, no gold ids.
    visible_keys = {"item_id", "question", "answer", "cited_passages"}
    blinded = [{k: v for k, v in item.items() if k in visible_keys} for item in batch]
    key = [{"item_id": item["item_id"], "pool": item["_pool"]} for item in batch]

    stem = args.stem
    write_jsonl(p["annotation"] / f"{stem}.jsonl", blinded, provenance=prov)
    write_jsonl(p["annotation"] / f"{stem}_key.jsonl", key, provenance=prov)

    print(f"\n  REVISION BATCH ({stem}, {args.n} fresh items)")
    print(f"    allocation                 {allocation}")
    for name, st in report["strata"].items():
        print(f"    {name:<26} available={st['available']:<6} "
              f"sampled={st['sampled']:<5} shortfall={st['shortfall']}")
    print(f"    {'total':<26} {report['total']}")
    if report["any_shortfall"]:
        print("\n    NOTE: a pool was smaller than its allocation within the fresh pool.")
        print("    Record this in PREREGISTRATION.md Section 11.")

    print(f"\n  Wrote batches to {p['annotation']}")
    print(f"\n  {stem}.jsonl embeds ADGM passage text and is gitignored on purpose --")
    print("  same as batch_01.jsonl was. It does NOT go through git. Send it to the")
    print("  second annotator the same way batch_01_double.jsonl got to them the first")
    print(f"  time (direct transfer). Only commit {stem}_key.jsonl (ids + pool, no")
    print("  corpus text):")
    print(f"    git add data/annotation/{stem}_key.jsonl")
    print(f"    git commit -m \"Revision batch {stem}: key for the {args.n} fresh items\"")
    print("    git push")
    print(f"\n  Both annotators label ALL {args.n} items, independently, at the same time:")
    print(f"    python -m saferag.pilot.annotate --batch data/annotation/{stem}.jsonl "
          "--annotator adnan")
    print(f"    python -m saferag.pilot.annotate --batch data/annotation/{stem}.jsonl "
          "--annotator riyad")
    print("\n  Do not discuss any item with each other until both files are complete.")
    print(f"  Then: python scripts/05b_compute_revision_kappa.py --annotator adnan "
          f"--second riyad --batch-stem {stem}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
