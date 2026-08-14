"""Step 05 -- compute the headline result.

    python scripts/05_compute_results.py --annotator adnan --second siti

Produces the base rate with a bootstrap interval, Cohen's kappa, and the
pre-registered verdict. Everything here follows PREREGISTRATION.md Sections 7-9;
no analytic choice is made at this point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import base_parser, load_cfg, paths  # noqa: E402

from saferag.pilot.stats import (  # noqa: E402
    cohens_kappa,
    decision,
    estimate_base_rate,
    interpret_kappa,
)
from saferag.utils.io import read_jsonl, read_provenance  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402

log = get_logger("results")


def load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    return {r["item_id"]: r["label"] for r in read_jsonl(path)}


def main() -> int:
    ap = base_parser("Compute the deceptive grounding base rate.")
    ap.add_argument("--annotator", required=True, help="Primary annotator name")
    ap.add_argument("--second", default=None, help="Second annotator name (for kappa)")
    args = ap.parse_args()

    cfg = load_cfg(args)
    p = paths(cfg)

    key_path = p["annotation"] / "batch_01_key.jsonl"
    if not key_path.exists():
        log.error("%s not found. Run scripts/04_make_annotation_batch.py first.", key_path)
        return 1
    pools = {r["item_id"]: r["pool"] for r in read_jsonl(key_path)}

    labels = load_labels(p["annotation"] / f"labels_{args.annotator}_batch_01.jsonl")
    log.info("Primary labels: %d", len(labels))

    missing = set(pools) - set(labels)
    if missing:
        log.warning("%d item(s) unlabelled; they are excluded.", len(missing))

    prov = read_provenance(p["interim"] / "filtered.jsonl") or {}
    funnel = prov.get("extra", {}).get("funnel", {})
    weights = funnel.get("stratum_weights")
    if not weights:
        log.error(
            "Could not read stratum_weights from the step-03 provenance header. "
            "Rerun step 03 rather than supplying weights by hand."
        )
        return 1

    strata = {
        name: [labels[i] for i in pools if pools[i] == name and i in labels]
        for name in weights
    }
    empty = [n for n, labs in strata.items() if not labs]
    if empty:
        log.warning("No annotated items in stratum/strata: %s", ", ".join(empty))

    result = estimate_base_rate(
        strata=strata,
        weights=weights,
        p_schema_valid=funnel.get("p_schema_valid"),
        p_faithful=funnel.get("p_faithful_given_schema"),
        n_resamples=cfg.analysis.bootstrap_resamples,
        confidence=cfg.analysis.confidence_level,
        seed=cfg.seed,
    )

    print("\n" + "=" * 74)
    print("  DECEPTIVE GROUNDING BASE RATE")
    print("=" * 74)
    print(result.summary())

    # The operative reliability figure is the revision round if one has been run
    # (PREREGISTRATION.md Section 8 / Deviation 2) -- NOT the original 50-item
    # pairing, which is why the revision round existed in the first place.
    # scripts/05b_compute_revision_kappa.py writes runs/<run>/revision_kappa.json;
    # prefer it, but still surface the original round's kappa as history rather
    # than silently dropping it.
    kappa = float("nan")
    kappa_source = "none"
    kappa_ci_low = None
    kappa_ci_high = None
    original_kappa = None

    revision_path = p["runs"] / "revision_kappa.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text(encoding="utf-8"))
        kappa = revision["cohens_kappa"]
        kappa_source = f"revision round ({revision['n_shared']} items, {revision['batch']})"
        kappa_ci_low = revision.get("cohens_kappa_ci_low")
        kappa_ci_high = revision.get("cohens_kappa_ci_high")
    elif args.second:
        second = load_labels(p["annotation"] / f"labels_{args.second}_batch_01_double.jsonl")
        shared = sorted(set(labels) & set(second))
        if len(shared) >= 2:
            kappa = cohens_kappa([labels[i] for i in shared], [second[i] for i in shared])
            kappa_source = f"original round ({len(shared)} items, batch_01_double)"

    # Original round's kappa, for historical context, whenever it's available --
    # regardless of which source is operative above.
    try:
        second_original = load_labels(
            p["annotation"] / f"labels_{args.second}_batch_01_double.jsonl"
        ) if args.second else {}
        shared_original = sorted(set(labels) & set(second_original))
        if len(shared_original) >= 2:
            original_kappa = cohens_kappa(
                [labels[i] for i in shared_original], [second_original[i] for i in shared_original]
            )
    except FileNotFoundError:
        pass

    print("\n  INTER-ANNOTATOR AGREEMENT")
    if kappa_source == "none":
        print("    not computed -- pass --second, and/or run scripts/04b_make_revision_batch.py")
        print("    + scripts/05b_compute_revision_kappa.py. A result without kappa is not")
        print("    publishable.")
    else:
        if kappa_ci_low is not None and kappa_ci_high is not None:
            print(f"    operative kappa     {kappa:.4f}   95% CI [{kappa_ci_low:.4f}, {kappa_ci_high:.4f}]   ({kappa_source})")
        else:
            print(f"    operative kappa     {kappa:.4f}   ({kappa_source})")
        print(f"    reading             {interpret_kappa(kappa)}")
        if original_kappa is not None and "revision" in kappa_source:
            print(f"    original round      {original_kappa:.4f}   (superseded, kept for history)")

    verdict, action = decision(result.conditional_rate.point)
    print("\n" + "=" * 74)
    print(f"  PRE-REGISTERED VERDICT: {verdict}")
    print("=" * 74)
    for line in action.split(". "):
        if line.strip():
            print(f"    {line.strip().rstrip('.')}.")
    print()

    out = p["runs"] / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "conditional_rate": {
                    "point": result.conditional_rate.point,
                    "low": result.conditional_rate.low,
                    "high": result.conditional_rate.high,
                },
                "unconditional_rate": (
                    None if result.unconditional_rate is None
                    else {
                        "point": result.unconditional_rate.point,
                        "low": result.unconditional_rate.low,
                        "high": result.unconditional_rate.high,
                    }
                ),
                "per_stratum": {k: v.point for k, v in result.per_stratum.items()},
                "stratum_weights": result.weights,
                "n_annotated": result.n_annotated,
                "n_excluded_na": result.n_excluded_na,
                "cohens_kappa": None if kappa != kappa else kappa,
                "cohens_kappa_ci_low": kappa_ci_low,
                "cohens_kappa_ci_high": kappa_ci_high,
                "cohens_kappa_source": kappa_source,
                "cohens_kappa_original_round": original_kappa,
                "verdict": verdict,
                "funnel": funnel,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Saved {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
