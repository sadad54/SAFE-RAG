"""The target answer schema (S1).

Deliberately minimal. This is not the schema SAFE-RAG will eventually use; it is
the smallest schema on which deceptive grounding can be observed, which is all the
pilot needs. Registered in docs/PREREGISTRATION.md Section 4.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class RegulatoryAnswer(BaseModel):
    """A structured answer to a regulatory question."""

    answer: str = Field(..., min_length=1, description="The answer in prose.")
    cited_passage_ids: list[str] = Field(
        ..., description="IDs of the passages the answer relies on. Must be non-empty."
    )
    obligations: list[str] = Field(
        default_factory=list,
        description="Discrete obligations extracted from the answer, if any.",
    )

    @field_validator("cited_passage_ids")
    @classmethod
    def _citations_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("cited_passage_ids must contain at least one non-empty id")
        return cleaned

    model_config = {"extra": "forbid"}


# The model is asked for bare JSON, but instruction-tuned models frequently wrap it
# in a markdown fence anyway. Stripping the fence is normalisation, not repair --
# it does not change whether the JSON itself is well formed.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Strip a surrounding markdown code fence, if present."""
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def _normalise_id(passage_id: str) -> str:
    """Fold away the trailing punctuation models drop when copying an id.

    ObliQA PassageIDs end in ')' or '.' -- '19::100)', '13::4.13.3.Guidance.5.'.
    Models reproduce the digits and lose the punctuation, so a literal string
    match rejects a citation that is unambiguously correct.
    """
    return re.sub(r"[\s.)\]]+$", "", passage_id.strip()).lower()


class CitationResolution(BaseModel):
    """Outcome of matching model-emitted ids against the ids actually offered."""

    resolved: list[str] = []
    unresolved: list[str] = []
    n_exact: int = 0
    n_normalised: int = 0
    n_ambiguous: int = 0


def resolve_citation_ids(cited: list[str], offered: list[str]) -> CitationResolution:
    """Match cited ids onto offered ids, tolerating dropped trailing punctuation.

    Exact matches win. Otherwise a citation resolves only if normalisation maps it
    onto exactly ONE offered id -- if it maps onto several, the model's intent is
    genuinely ambiguous and we refuse rather than guess. A bare DocumentID such as
    '17' matches nothing and stays unresolved, which is correct: it names a
    document, not a passage.

    >>> r = resolve_citation_ids(["19::100"], ["19::100)", "19::25)"])
    >>> r.resolved, r.n_normalised
    (['19::100)'], 1)
    >>> resolve_citation_ids(["17"], ["17::Part 2.5A.(1)"]).unresolved
    ['17']
    """
    exact = set(offered)
    by_norm: dict[str, list[str]] = {}
    for o in offered:
        by_norm.setdefault(_normalise_id(o), []).append(o)

    out = CitationResolution()
    seen: set[str] = set()
    for c in cited:
        if c in exact:
            hit, kind = c, "exact"
        else:
            candidates = by_norm.get(_normalise_id(c), [])
            if len(candidates) == 1:
                hit, kind = candidates[0], "normalised"
            else:
                out.unresolved.append(c)
                if len(candidates) > 1:
                    out.n_ambiguous += 1
                continue
        if hit not in seen:
            seen.add(hit)
            out.resolved.append(hit)
        out.n_exact += kind == "exact"
        out.n_normalised += kind == "normalised"
    return out


class SchemaCheckResult(BaseModel):
    """Outcome of the S1 check."""

    valid: bool
    error: str | None = None
    parsed: dict[str, Any] | None = None


def check_schema(raw_output: str) -> SchemaCheckResult:
    """Run the S1 check on a raw generator output.

    >>> check_schema('{"answer": "Yes.", "cited_passage_ids": ["p1"]}').valid
    True
    >>> check_schema('{"answer": "Yes.", "cited_passage_ids": []}').valid
    False
    >>> check_schema("not json at all").valid
    False
    """
    text = extract_json_block(raw_output)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return SchemaCheckResult(valid=False, error=f"json_decode_error: {exc.msg}")

    if not isinstance(payload, dict):
        return SchemaCheckResult(valid=False, error="top_level_not_object")

    try:
        model = RegulatoryAnswer(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        return SchemaCheckResult(valid=False, error=f"validation_error: {loc}: {first['msg']}")

    return SchemaCheckResult(valid=True, parsed=model.model_dump())
