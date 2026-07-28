"""Stratified sampling for annotation (PREREGISTRATION.md Section 6).

100 items from the candidate pool, 50 from the control pool, seed 20260728.
Records are shuffled after sampling so the annotator cannot infer an item's pool
from its position in the batch -- pool membership is a strong hint about the
expected label and would bias annotation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from saferag.checks.attribution import CANDIDATE, CONTROL


def stratified_sample(
    records: Sequence[dict[str, Any]],
    n_candidate: int = 100,
    n_control: int = 50,
    seed: int = 20260728,
    pool_key: str = "pool",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Draw the annotation batch.

    Returns the shuffled batch and a report of pool sizes and any shortfall.

    Shortfall is not an error: if a pool is smaller than its allocation the whole
    pool is taken and the shortfall recorded, as the pre-registration specifies.
    """
    rng = random.Random(seed)

    candidates = [r for r in records if r.get(pool_key) == CANDIDATE]
    controls = [r for r in records if r.get(pool_key) == CONTROL]

    take_cand = min(n_candidate, len(candidates))
    take_ctrl = min(n_control, len(controls))

    picked = rng.sample(candidates, take_cand) + rng.sample(controls, take_ctrl)

    # Blind the annotator: strip pool and gold-set fields, then shuffle.
    batch: list[dict[str, Any]] = []
    for rec in picked:
        item = {k: v for k, v in rec.items() if k not in {pool_key, "gold_passage_ids"}}
        item["_pool"] = rec[pool_key]  # retained for analysis, ignored by the CLI
        batch.append(item)
    rng.shuffle(batch)

    report = {
        "pool_candidate_available": len(candidates),
        "pool_control_available": len(controls),
        "sampled_candidate": take_cand,
        "sampled_control": take_ctrl,
        "shortfall_candidate": n_candidate - take_cand,
        "shortfall_control": n_control - take_ctrl,
        "total": len(batch),
    }
    return batch, report


def double_annotation_subset(
    batch: Sequence[dict[str, Any]],
    n: int = 50,
    seed: int = 20260728,
) -> list[dict[str, Any]]:
    """Draw the subset for the second annotator, proportional across pools.

    Proportional allocation keeps kappa from being dominated by one pool.
    """
    rng = random.Random(seed)
    cands = [r for r in batch if r.get("_pool") == CANDIDATE]
    ctrls = [r for r in batch if r.get("_pool") == CONTROL]
    total = len(cands) + len(ctrls)
    if total == 0:
        return []

    n = min(n, total)
    take_cand = min(len(cands), round(n * len(cands) / total))
    take_ctrl = min(len(ctrls), n - take_cand)
    # Repair any rounding shortfall from whichever pool still has items.
    while take_cand + take_ctrl < n and take_cand < len(cands):
        take_cand += 1

    subset = rng.sample(cands, take_cand) + rng.sample(ctrls, take_ctrl)
    rng.shuffle(subset)
    return subset
