"""Stratified sampling for annotation (PREREGISTRATION.md Section 6, as amended).

80 items from candidate_recoverable, 20 from candidate_unrecoverable, 50 from
control. Seed 20260728.

Records are shuffled after sampling so the annotator cannot infer an item's
stratum from its position in the batch -- stratum membership is a strong hint
about the expected label and would bias annotation.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from saferag.checks.attribution import (
    CANDIDATE_RECOVERABLE,
    CANDIDATE_UNRECOVERABLE,
    CONTROL,
)

DEFAULT_ALLOCATION: dict[str, int] = {
    CANDIDATE_RECOVERABLE: 80,
    CANDIDATE_UNRECOVERABLE: 20,
    CONTROL: 50,
}

# Never shown to the annotator: these would give the answer away.
_HIDDEN = {"pool", "gold_passage_ids", "retrieved_passage_ids", "s3_jaccard", "s3_n_overlap"}


def stratified_sample(
    records: Sequence[Mapping[str, Any]],
    allocation: Mapping[str, int] | None = None,
    seed: int = 20260728,
    pool_key: str = "pool",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw the annotation batch.

    Returns the shuffled batch and a report of stratum sizes and any shortfall.

    Shortfall is not an error: if a stratum is smaller than its allocation the
    whole stratum is taken and the shortfall recorded, as the pre-registration
    specifies.
    """
    allocation = dict(allocation or DEFAULT_ALLOCATION)
    rng = random.Random(seed)

    by_pool: dict[str, list[Mapping[str, Any]]] = {name: [] for name in allocation}
    for rec in records:
        pool = rec.get(pool_key)
        if pool in by_pool:
            by_pool[pool].append(rec)

    picked: list[Mapping[str, Any]] = []
    report: dict[str, Any] = {"allocation": allocation, "strata": {}}
    for name, want in allocation.items():
        available = by_pool[name]
        take = min(want, len(available))
        picked.extend(rng.sample(available, take))
        report["strata"][name] = {
            "available": len(available),
            "sampled": take,
            "shortfall": want - take,
        }

    batch: list[dict[str, Any]] = []
    for rec in picked:
        item = {k: v for k, v in rec.items() if k not in _HIDDEN}
        item["_pool"] = rec[pool_key]  # kept for analysis, stripped before annotation
        batch.append(item)
    rng.shuffle(batch)

    report["total"] = len(batch)
    report["any_shortfall"] = any(s["shortfall"] for s in report["strata"].values())
    return batch, report


def scale_allocation(allocation: Mapping[str, int], n: int) -> dict[str, int]:
    """Scale a stratum allocation to a new total, keeping proportions and an
    exact sum.

    Used to derive the revision-round allocation (30 items) from the registered
    150-item allocation (80/20/50) without hand-picking numbers: 30/150 of each
    stratum, rounded, with any rounding drift repaired against whichever stratum
    still has room -- same technique as ``double_annotation_subset``.

    >>> scale_allocation({"a": 80, "b": 20, "c": 50}, 30)
    {'a': 16, 'b': 4, 'c': 10}
    """
    total = sum(allocation.values())
    if total <= 0:
        raise ValueError(f"allocation must sum to a positive number, got {total}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    scaled = {name: round(n * count / total) for name, count in allocation.items()}
    while sum(scaled.values()) != n:
        delta = 1 if sum(scaled.values()) < n else -1
        for name in allocation:
            if scaled[name] + delta >= 0:
                scaled[name] += delta
                break
        else:
            break
    return scaled


def double_annotation_subset(
    batch: Sequence[Mapping[str, Any]],
    n: int = 50,
    seed: int = 20260728,
) -> list[dict[str, Any]]:
    """Draw the subset for the second annotator, proportional across strata.

    Proportional allocation keeps kappa from being dominated by one stratum.
    """
    rng = random.Random(seed)
    by_pool: dict[str, list[Mapping[str, Any]]] = {}
    for rec in batch:
        by_pool.setdefault(rec.get("_pool", "?"), []).append(rec)

    total = len(batch)
    if total == 0:
        return []
    n = min(n, total)

    take = {name: min(len(items), round(n * len(items) / total)) for name, items in by_pool.items()}
    # Repair rounding drift against whichever stratum still has items.
    while sum(take.values()) != n:
        delta = 1 if sum(take.values()) < n else -1
        for name, items in by_pool.items():
            if 0 <= take[name] + delta <= len(items):
                take[name] += delta
                break
        else:
            break

    subset = [r for name, items in by_pool.items() for r in rng.sample(items, take[name])]
    rng.shuffle(subset)
    return [dict(r) for r in subset]
