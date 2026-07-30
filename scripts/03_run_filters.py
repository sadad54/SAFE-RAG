"""Step 03 -- apply S1 (schema), S2 (faithfulness) and S3 (attribution screen).

    python scripts/03_run_filters.py --stub-nli     # laptop, no models
    python scripts/03_run_filters.py                # cluster

Output: data/interim/filtered.jsonl plus a printed funnel summary.
"""

from __future__ import annotations

import sys

from _common import base_parser, paths  # noqa: E402
from tqdm import tqdm  # noqa: E402

from saferag.checks.attribution import STRATA, screen  # noqa: E402
from saferag.checks.faithfulness import (  # noqa: E402
    HFNLIScorer,
    RuleDecomposer,
    StubNLIScorer,
    check_faithfulness,
)
from saferag.config import load_config  # noqa: E402
from saferag.generation.schema import check_schema  # noqa: E402
from saferag.utils.io import read_jsonl, write_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402
from saferag.utils.provenance import stamp  # noqa: E402

log = get_logger("filters")


def main() -> int:
    """Apply S1 (schema), S2 (faithfulness) and S3 (attribution screen)."""
    ap = base_parser("Apply the S1/S2/S3 filters.")
    ap.add_argument("--stub-nli", action="store_true", help="Fake NLI scorer (testing only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = paths(cfg)

    src = p["interim"] / "answers.jsonl"
    if not src.exists():
        log.error("%s not found. Run scripts/02_run_rag.py first.", src)
        return 1

    records = list(read_jsonl(src))
    log.info("Loaded %d answers", len(records))

    if args.stub_nli:
        log.warning("Using the stub NLI scorer. Results are NOT valid for the study.")
        scorer = StubNLIScorer()
    else:
        log.info("Loading NLI model: %s", cfg.checks.nli.model)
        scorer = HFNLIScorer(cfg.checks.nli.model, batch_size=cfg.checks.nli.batch_size)

    decomposer = RuleDecomposer()
    passage_text = {
        pp["id"]: pp["text"] for r in records for pp in r.get("retrieved_passages", [])
    }

    out_records = []
    n_s1 = n_s2 = 0
    pool_counts: dict[str, int] = dict.fromkeys(STRATA, 0)

    for rec in tqdm(records, desc="filter"):
        row = {
            "item_id": rec["item_id"],
            "question": rec["question"],
            "gold_passage_ids": rec.get("gold_passage_ids", []),
            "retrieved_passage_ids": rec.get("retrieved_passage_ids", []),
        }

        # --- S1 -------------------------------------------------------------
        s1 = check_schema(rec.get("raw_output", ""))
        row["s1_valid"] = s1.valid
        row["s1_error"] = s1.error
        if not s1.valid:
            row["stage"] = "failed_s1"
            out_records.append(row)
            continue
        n_s1 += 1

        parsed = s1.parsed or {}
        cited_ids = parsed.get("cited_passage_ids", [])
        row["answer"] = parsed.get("answer", "")
        row["cited_passage_ids"] = cited_ids
        row["obligations"] = parsed.get("obligations", [])
        row["cited_passages"] = [
            {"id": pid, "text": passage_text.get(pid, "")} for pid in cited_ids
        ]

        # --- S2 -------------------------------------------------------------
        cited_text = "\n\n".join(passage_text.get(pid, "") for pid in cited_ids)
        s2 = check_faithfulness(
            row["answer"], cited_text, decomposer, scorer,
            entailment_threshold=cfg.checks.faithfulness.entailment_threshold,
            contradiction_threshold=cfg.checks.faithfulness.contradiction_threshold,
        )
        row["s2_passed"] = s2.passed
        row["s2_reason"] = s2.reason
        row["s2_min_entailment"] = s2.min_entailment
        row["s2_max_contradiction"] = s2.max_contradiction
        row["s2_n_claims"] = len(s2.claims)
        if not s2.passed:
            row["stage"] = "failed_s2"
            out_records.append(row)
            continue
        n_s2 += 1

        # --- S3 (screen, not a criterion) -----------------------------------
        sc = screen(
            cited_ids,
            row["gold_passage_ids"],
            row["retrieved_passage_ids"],
            threshold=cfg.checks.attribution.jaccard_threshold,
        )
        row["pool"] = sc.pool
        row["s3_jaccard"] = sc.jaccard
        row["s3_n_overlap"] = sc.n_overlap
        row["gold_in_context"] = sc.gold_in_context
        row["stage"] = "survivor"
        pool_counts[sc.pool] += 1
        out_records.append(row)

    total = len(records)
    p1 = n_s1 / total if total else 0.0
    p2 = n_s2 / n_s1 if n_s1 else 0.0
    weights = {k: (v / n_s2 if n_s2 else 0.0) for k, v in pool_counts.items()}

    out = p["interim"] / "filtered.jsonl"
    prov = stamp(
        script="03_run_filters.py",
        config=cfg,
        config_path=str(args.config),
        seed=cfg.seed,
        models={"nli": "stub" if args.stub_nli else cfg.checks.nli.model},
        funnel={
            "n_total": total,
            "n_s1_pass": n_s1,
            "n_s2_pass": n_s2,
            "pool_counts": pool_counts,
            "p_schema_valid": p1,
            "p_faithful_given_schema": p2,
            "stratum_weights": weights,
        },
    )
    write_jsonl(out, out_records, provenance=prov)

    print("\n  FUNNEL")
    print(f"    generated answers            {total:>6}")
    print(f"    S1 schema-valid              {n_s1:>6}   ({p1:.1%})")
    print(f"    S2 faithfulness-passing      {n_s2:>6}   ({p2:.1%} of S1)")
    for name in STRATA:
        print(f"      -> {name:<24} {pool_counts[name]:>6}   (w = {weights[name]:.4f})")
    print(f"\n  Wrote {out}")
    print("  Next: python scripts/04_make_annotation_batch.py\n")

    if n_s2 < 150:
        log.warning(
            "Only %d survivors. The registered sample is 150; you will hit the "
            "shortfall path. Consider running more questions.", n_s2
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
