"""Statistics for the deceptive-grounding pilot.

Implements exactly what docs/PREREGISTRATION.md Section 7 and Section 8 specify:

* Wilson score intervals for the component proportions
* Cohen's kappa for inter-annotator agreement
* The stratified base-rate estimator with a stratified bootstrap interval

Nothing here depends on a model, a GPU, or the dataset. It is pure arithmetic and
it is the part of the study that must be correct, so it is the part with tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Two-sided normal quantiles for the confidence levels we actually use.
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z_for(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    raise ValueError(
        f"Unsupported confidence level {confidence!r}. Use one of {sorted(_Z)}, "
        "or add the quantile explicitly rather than approximating it."
    )


@dataclass(frozen=True)
class Interval:
    """A point estimate with an interval."""

    point: float
    low: float
    high: float
    confidence: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"

    def as_percent(self, dp: int = 1) -> str:
        return (
            f"{self.point * 100:.{dp}f}% "
            f"[{self.low * 100:.{dp}f}%, {self.high * 100:.{dp}f}%]"
        )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it behaves sensibly at small
    n and near 0 or 1 -- both of which this study will hit, since the whole
    question is whether a rate is small.

    >>> ci = wilson_interval(0, 50)
    >>> ci.point
    0.0
    >>> ci.low
    0.0
    >>> round(ci.high, 4)
    0.0713
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0 <= successes <= n:
        raise ValueError(f"successes ({successes}) must lie in [0, n] with n={n}")
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, confidence)

    z = _z_for(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low, high = centre - half, centre + half

    # At p = 0 the exact lower bound is 0; at p = 1 the exact upper bound is 1.
    # Both come out a few ulps off under floating point, so set them exactly.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return Interval(p, max(0.0, low), min(1.0, high), confidence)


def cohens_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    exclude: Sequence[str] = ("NA",),
) -> float:
    """Cohen's kappa between two annotators.

    Items where *either* annotator used an excluded label (by default ``NA``) are
    dropped pairwise, as specified in the pre-registration.

    Returns ``nan`` if fewer than two items remain, or if expected agreement is
    exactly 1 while observed agreement is not (kappa is undefined there).

    >>> round(cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]), 4)
    1.0
    >>> round(cohens_kappa(["A", "A", "B", "B"], ["B", "B", "A", "A"]), 4)
    -1.0
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Annotator label lists differ in length: {len(labels_a)} vs {len(labels_b)}"
        )
    excluded = set(exclude)
    pairs = [
        (a, b) for a, b in zip(labels_a, labels_b, strict=True)
        if a not in excluded and b not in excluded
    ]
    n = len(pairs)
    if n < 2:
        return float("nan")

    categories = sorted({lab for pair in pairs for lab in pair})
    observed = sum(1 for a, b in pairs if a == b) / n

    count_a = {c: 0 for c in categories}
    count_b = {c: 0 for c in categories}
    for a, b in pairs:
        count_a[a] += 1
        count_b[b] += 1
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else float("nan")
    return (observed - expected) / (1.0 - expected)


def interpret_kappa(kappa: float) -> str:
    """The pre-registered reading of kappa (PREREGISTRATION.md Section 8)."""
    if math.isnan(kappa):
        return "undefined (too few comparable items)"
    if kappa >= 0.70:
        return "reliable -- proceed"
    if kappa >= 0.50:
        return "usable -- proceed, report prominently, analyse disagreements"
    return "NOT RELIABLE -- stop, revise guidelines once, re-annotate 30 items"


@dataclass(frozen=True)
class BaseRateResult:
    """Output of the stratified base-rate estimate."""

    conditional_rate: Interval  # r -- among schema-valid, faithfulness-passing answers
    unconditional_rate: Interval | None  # R -- among all generated answers
    per_stratum: dict[str, Interval]
    weights: dict[str, float]
    n_annotated: dict[str, int]
    n_excluded_na: int

    def summary(self) -> str:
        lines = [f"  Conditional rate  r = {self.conditional_rate.as_percent()}"]
        if self.unconditional_rate is not None:
            lines.append(f"  Unconditional     R = {self.unconditional_rate.as_percent()}")
        lines.append("")
        width = max(len(s) for s in self.per_stratum) if self.per_stratum else 10
        for name, interval in self.per_stratum.items():
            lines.append(
                f"    B | {name:<{width}}  {interval.as_percent():<26} "
                f"n={self.n_annotated[name]:<4} weight={self.weights[name]:.4f}"
            )
        lines.append(f"\n  Excluded as NA      = {self.n_excluded_na}")
        return "\n".join(lines)


def estimate_base_rate(
    strata: dict[str, Sequence[str]],
    weights: dict[str, float],
    p_schema_valid: float | None = None,
    p_faithful: float | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260728,
) -> BaseRateResult:
    """Stratified estimate of the deceptive-grounding rate.

    Implements PREREGISTRATION.md Section 7, generalised to any number of strata::

        r = sum over strata of  w_s * b_s
        R = p1 * p2 * r

    Args:
        strata: stratum name -> A/B/C/NA labels for the items annotated in it.
        weights: stratum name -> that stratum's share of all S1-and-S2 survivors.
            Measured on the full survivor set, not on the annotated sample. Must
            cover the same names as ``strata`` and sum to 1.
        p_schema_valid: Optional S1 pass rate, for the unconditional rate.
        p_faithful: Optional S2 pass rate given S1, for the unconditional rate.
        n_resamples: Bootstrap resamples. Pre-registered at 10,000.
        confidence: Pre-registered at 0.95.
        seed: Pre-registered at 20260728.

    The interval on ``r`` comes from a stratified non-parametric bootstrap: each
    stratum is resampled with replacement at its own observed size, and the
    weights are held fixed (they are estimated on the full survivor set, where n
    is large enough that their contribution to the variance of r is negligible
    relative to the 150 annotated items).
    """
    if set(strata) != set(weights):
        raise ValueError(
            f"strata and weights must cover the same names: "
            f"{sorted(strata)} vs {sorted(weights)}"
        )
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"weights must be non-negative, got {weights}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"weights must sum to 1, got {total}")

    clean = {name: [lab for lab in labs if lab != "NA"] for name, labs in strata.items()}
    n_na = sum(len(strata[name]) - len(clean[name]) for name in strata)
    if not any(clean.values()):
        raise ValueError("No annotated items remain after excluding NA.")

    arrays = {
        name: np.array([1 if lab == "B" else 0 for lab in labs], dtype=np.int8)
        for name, labs in clean.items()
    }
    per_stratum = {
        name: wilson_interval(int(arr.sum()), len(arr), confidence)
        for name, arr in arrays.items()
    }
    means = {name: (float(arr.mean()) if len(arr) else 0.0) for name, arr in arrays.items()}
    r_point = sum(weights[name] * means[name] for name in arrays)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        draws[i] = sum(
            weights[name]
            * (rng.choice(arr, size=len(arr), replace=True).mean() if len(arr) else 0.0)
            for name, arr in arrays.items()
        )

    alpha = 1.0 - confidence
    lo, hi = np.quantile(draws, [alpha / 2, 1.0 - alpha / 2])
    r_interval = Interval(r_point, float(lo), float(hi), confidence)

    uncond = None
    if p_schema_valid is not None and p_faithful is not None:
        scale = p_schema_valid * p_faithful
        uncond = Interval(r_point * scale, float(lo) * scale, float(hi) * scale, confidence)

    return BaseRateResult(
        conditional_rate=r_interval,
        unconditional_rate=uncond,
        per_stratum=per_stratum,
        weights=dict(weights),
        n_annotated={name: len(arr) for name, arr in arrays.items()},
        n_excluded_na=n_na,
    )


def decision(rate: float) -> tuple[str, str]:
    """The pre-registered decision rule (PREREGISTRATION.md Section 9).

    Returns (verdict, action). Called on the conditional rate r.
    """
    if rate > 0.40:
        return (
            "SUSPECT DEFECT",
            "Do not report. Hand-inspect 20 items. Check entailment thresholds, "
            "the gold-passage join, and passage-ID matching.",
        )
    if rate >= 0.10:
        return ("STRONG", "Headline this. Attribution branch proceeds as designed.")
    if rate >= 0.05:
        return (
            "THIN BUT REAL",
            "Report aggregate plus the breakdown by question type (RQ-a). "
            "Claim becomes the conditional structure, not the magnitude.",
        )
    return (
        "TOO RARE",
        "Execute pivots in order: Pivot 0 (check FinDER, ~2 days), then Pivot A "
        "(construct an entity-confusable diagnostic set).",
    )
