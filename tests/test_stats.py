"""Tests for the pilot statistics.

These are the numbers the paper reports, so they get real tests: known closed-form
values, boundary conditions, and agreement with an independent computation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from saferag.pilot.stats import (
    cohens_kappa,
    decision,
    estimate_base_rate,
    interpret_kappa,
    wilson_interval,
)

# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------


def test_wilson_known_value():
    """Textbook check: 0/50 successes, 95% Wilson upper bound is ~0.0713."""
    ci = wilson_interval(0, 50)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert ci.high == pytest.approx(0.0713, abs=1e-4)


def test_wilson_symmetric_case():
    """At p = 0.5 the interval is symmetric about 0.5."""
    ci = wilson_interval(50, 100)
    assert ci.point == 0.5
    assert (ci.low + ci.high) / 2 == pytest.approx(0.5, abs=1e-12)


def test_wilson_all_successes_bounded():
    ci = wilson_interval(30, 30)
    assert ci.point == 1.0
    assert ci.high == 1.0
    assert 0.0 < ci.low < 1.0


def test_wilson_interval_narrows_with_n():
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_wilson_contains_point_estimate():
    for k, n in [(1, 10), (7, 20), (99, 100), (0, 3)]:
        ci = wilson_interval(k, n)
        assert ci.low <= ci.point <= ci.high


def test_wilson_higher_confidence_is_wider():
    c95 = wilson_interval(20, 100, confidence=0.95)
    c99 = wilson_interval(20, 100, confidence=0.99)
    assert (c99.high - c99.low) > (c95.high - c95.low)


def test_wilson_zero_n_is_nan_point():
    ci = wilson_interval(0, 0)
    assert math.isnan(ci.point)
    assert (ci.low, ci.high) == (0.0, 1.0)


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, 10, confidence=0.975)


# --------------------------------------------------------------------------
# Cohen's kappa
# --------------------------------------------------------------------------


def test_kappa_perfect_agreement():
    labels = ["A", "B", "C", "A", "B", "C"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_complete_disagreement_two_categories():
    assert cohens_kappa(["A", "A", "B", "B"], ["B", "B", "A", "A"]) == pytest.approx(-1.0)


def test_kappa_known_worked_example():
    """Standard 2x2 example: a=20, b=5, c=10, d=15, n=50.

    po = 35/50 = 0.70
    pe = (25/50)(30/50) + (25/50)(20/50) = 0.30 + 0.20 = 0.50
    kappa = (0.70 - 0.50) / 0.50 = 0.40
    """
    a = ["Y"] * 20 + ["Y"] * 5 + ["N"] * 10 + ["N"] * 15
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.40, abs=1e-9)


def test_kappa_chance_agreement_is_near_zero():
    """Independent random labellers should give kappa near 0."""
    rng = np.random.default_rng(0)
    n = 4000
    a = list(rng.choice(["A", "B", "C"], size=n))
    b = list(rng.choice(["A", "B", "C"], size=n))
    assert abs(cohens_kappa(a, b)) < 0.05


def test_kappa_excludes_na_pairwise():
    a = ["A", "B", "NA", "A"]
    b = ["A", "B", "A", "A"]
    # The NA pair drops out, leaving 3 items in perfect agreement.
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_kappa_all_na_is_nan():
    assert math.isnan(cohens_kappa(["NA", "NA"], ["NA", "NA"]))


def test_kappa_single_category_both_annotators():
    """Expected agreement is 1; observed is 1. Degenerate but not a disagreement."""
    assert cohens_kappa(["A"] * 10, ["A"] * 10) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["A", "B"], ["A"])


def test_interpret_kappa_thresholds():
    assert "reliable" in interpret_kappa(0.75)
    assert "usable" in interpret_kappa(0.60)
    assert "NOT RELIABLE" in interpret_kappa(0.30)
    assert "undefined" in interpret_kappa(float("nan"))


# --------------------------------------------------------------------------
# Stratified base-rate estimator
# --------------------------------------------------------------------------


def _labels(n_b: int, n_other: int) -> list[str]:
    return ["B"] * n_b + ["A"] * n_other


def test_base_rate_matches_hand_computation():
    """q=0.3, b_cand=0.40, b_ctrl=0.10  ->  r = 0.3*0.4 + 0.7*0.1 = 0.19"""
    res = estimate_base_rate(
        candidate_labels=_labels(40, 60),
        control_labels=_labels(5, 45),
        q_candidate_share=0.3,
        n_resamples=200,
    )
    assert res.b_candidate.point == pytest.approx(0.40)
    assert res.b_control.point == pytest.approx(0.10)
    assert res.conditional_rate.point == pytest.approx(0.19, abs=1e-12)


def test_base_rate_interval_brackets_point():
    res = estimate_base_rate(_labels(30, 70), _labels(5, 45), 0.4, n_resamples=2000)
    assert res.conditional_rate.low <= res.conditional_rate.point <= res.conditional_rate.high


def test_base_rate_is_deterministic_given_seed():
    kwargs = dict(
        candidate_labels=_labels(25, 75),
        control_labels=_labels(3, 47),
        q_candidate_share=0.35,
        n_resamples=1000,
        seed=20260728,
    )
    a = estimate_base_rate(**kwargs)
    b = estimate_base_rate(**kwargs)
    assert a.conditional_rate.low == b.conditional_rate.low
    assert a.conditional_rate.high == b.conditional_rate.high


def test_unconditional_rate_scales_correctly():
    res = estimate_base_rate(
        _labels(40, 60), _labels(10, 40), 0.5,
        p_schema_valid=0.85, p_faithful=0.70, n_resamples=200,
    )
    expected = res.conditional_rate.point * 0.85 * 0.70
    assert res.unconditional_rate is not None
    assert res.unconditional_rate.point == pytest.approx(expected)
    assert res.unconditional_rate.point < res.conditional_rate.point


def test_na_items_are_excluded_and_counted():
    res = estimate_base_rate(
        candidate_labels=["B"] * 10 + ["A"] * 10 + ["NA"] * 5,
        control_labels=["A"] * 20 + ["NA"] * 2,
        q_candidate_share=0.5,
        n_resamples=200,
    )
    assert res.n_excluded_na == 7
    assert res.n_candidate == 20
    assert res.n_control == 20
    assert res.b_candidate.point == pytest.approx(0.5)


def test_zero_rate_everywhere():
    res = estimate_base_rate(["A"] * 100, ["A"] * 50, 0.3, n_resamples=500)
    assert res.conditional_rate.point == 0.0
    assert res.conditional_rate.low == 0.0


def test_q_out_of_range_raises():
    with pytest.raises(ValueError):
        estimate_base_rate(["B"], ["A"], q_candidate_share=1.5)


def test_all_na_raises():
    with pytest.raises(ValueError):
        estimate_base_rate(["NA"], ["NA"], 0.5)


def test_bootstrap_approximates_wilson_when_one_stratum_dominates():
    """With q=1 the estimator collapses to the candidate proportion, so the
    bootstrap interval should sit close to the Wilson interval for that stratum."""
    res = estimate_base_rate(_labels(30, 70), _labels(0, 50), 1.0, n_resamples=8000)
    wil = wilson_interval(30, 100)
    assert res.conditional_rate.low == pytest.approx(wil.low, abs=0.04)
    assert res.conditional_rate.high == pytest.approx(wil.high, abs=0.04)


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate,verdict",
    [
        (0.55, "SUSPECT DEFECT"),
        (0.40, "STRONG"),
        (0.15, "STRONG"),
        (0.10, "STRONG"),
        (0.099, "THIN BUT REAL"),
        (0.05, "THIN BUT REAL"),
        (0.049, "TOO RARE"),
        (0.0, "TOO RARE"),
    ],
)
def test_decision_rule_boundaries(rate, verdict):
    assert decision(rate)[0] == verdict
