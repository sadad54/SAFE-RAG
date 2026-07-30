"""S2 -- the faithfulness check.

Decompose the answer into atomic claims, then score each claim for entailment
against the concatenated text of the passages the answer cited. The item passes
if every claim is entailed above threshold and none is contradicted above
threshold.

Two constraints from the pre-registration, both load-bearing:

1. The NLI model MUST come from a different family than the generator. Using one
   family to both write and judge produces a self-referential measurement.
2. The thresholds (0.5 / 0.5) are registered and are NOT to be tuned. If they turn
   out to be badly calibrated, that is a finding to report, not a knob to turn.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Claim decomposition
# ---------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


class ClaimDecomposer(ABC):
    @abstractmethod
    def decompose(self, answer: str) -> list[str]:
        """Split an answer into atomic claims."""


class RuleDecomposer(ClaimDecomposer):
    """Sentence splitting.

    A weak but transparent baseline: it under-splits compound sentences, which
    makes S2 slightly permissive. Being permissive is the safe direction here --
    it lets more items through to human annotation rather than filtering out
    items that might have been deceptive grounding.

    >>> RuleDecomposer().decompose("A firm must report. It must do so in 30 days.")
    ['A firm must report.', 'It must do so in 30 days.']
    """

    def decompose(self, answer: str) -> list[str]:
        parts = [s.strip() for s in _SENT.split(answer.strip()) if s.strip()]
        return [p for p in parts if len(p) > 3]


class LLMDecomposer(ClaimDecomposer):
    """LLM-based atomic claim extraction. Better on legal prose, needs a model."""

    PROMPT = """Break the following statement into atomic factual claims.
Each claim must stand alone and assert exactly one thing.
Return one claim per line, with no numbering and no commentary.

STATEMENT
{answer}

CLAIMS:"""

    def __init__(self, generator) -> None:  # noqa: ANN001 - any Generator
        self.generator = generator

    def decompose(self, answer: str) -> list[str]:
        out = self.generator.generate([self.PROMPT.format(answer=answer)])[0]
        claims = [
            re.sub(r"^\s*[-*\d.)\]]+\s*", "", line).strip()
            for line in out.splitlines()
            if line.strip()
        ]
        return [c for c in claims if len(c) > 3]


# ---------------------------------------------------------------------------
# Entailment scoring
# ---------------------------------------------------------------------------


@dataclass
class ClaimScore:
    claim: str
    entailment: float
    contradiction: float
    neutral: float

    @property
    def supported(self) -> bool:
        return self.entailment >= 0.5 and self.contradiction < 0.5


@dataclass
class FaithfulnessResult:
    passed: bool
    claims: list[ClaimScore] = field(default_factory=list)
    min_entailment: float = 0.0
    max_contradiction: float = 0.0
    reason: str | None = None


class NLIScorer(ABC):
    @abstractmethod
    def score(self, premise: str, hypotheses: Sequence[str]) -> list[ClaimScore]:
        """Return (entailment, contradiction, neutral) per hypothesis."""


class StubNLIScorer(NLIScorer):
    """Deterministic fake scorer for pipeline tests. Never used for results."""

    def score(self, premise: str, hypotheses: Sequence[str]) -> list[ClaimScore]:
        out = []
        for h in hypotheses:
            overlap = len(set(h.lower().split()) & set(premise.lower().split()))
            ent = min(0.99, 0.3 + 0.1 * overlap)
            out.append(ClaimScore(h, ent, 1.0 - ent - 0.05, 0.05))
        return out


class HFNLIScorer(NLIScorer):
    """transformers sequence-classification NLI model.

    Label order differs between checkpoints, so it is read from the model config
    rather than assumed. Assuming it is a classic and silent source of inverted
    faithfulness scores.
    """

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", batch_size: int = 32):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install -e '.[models]'") from exc
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # Without an explicit device the model stays on CPU, and DeBERTa-large over
        # thousands of 512-token pairs then takes hours with no visible symptom.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp32 deliberately. DeBERTa v1's disentangled attention mixes fp32
        # buffers with activations, so fp16 dies with "expected scalar type Float
        # but found Half". fp32 on GPU is still orders of magnitude faster than
        # the CPU path this replaced, and NLI is not the bottleneck.
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.batch_size = batch_size
        self._torch = torch
        log = logging.getLogger("saferag.nli")
        log.info("NLI model %s on %s (%s)", model_name, self.device, self.model.dtype)

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx = {}
        for want in ("entailment", "contradiction", "neutral"):
            match = [i for i, lab in id2label.items() if want in lab]
            if not match:
                raise ValueError(
                    f"NLI model {model_name} has no '{want}' label. Labels: {id2label}"
                )
            self.idx[want] = match[0]

        self._self_check(log)

    def _self_check(self, log: logging.Logger) -> None:
        """Score one obvious entailment and one obvious contradiction at load time.

        Catches two failures in about a second, both of which otherwise surface
        only after a long run or not at all:

        * dtype/device mismatches, which raise on the first real forward pass;
        * an inverted label mapping, which raises nothing at all and silently
          turns the faithfulness filter upside down.
        """
        premise = "An Authorised Person must retain transaction records for six years."
        entailed, contradicted = self.score(
            premise,
            [
                "Records must be kept for six years.",
                "Records may be destroyed immediately.",
            ],
        )
        log.info(
            "self-check: entailed=%.3f contradicted-hypothesis-entailment=%.3f",
            entailed.entailment, contradicted.entailment,
        )
        if entailed.entailment <= contradicted.entailment:
            raise RuntimeError(
                "NLI self-check failed: an obvious entailment did not score higher "
                f"than an obvious contradiction ({entailed.entailment:.3f} vs "
                f"{contradicted.entailment:.3f}). The entailment/contradiction "
                f"label indices are probably inverted. Resolved mapping: {self.idx}. "
                "Every faithfulness score downstream would be meaningless."
            )

    def score(self, premise: str, hypotheses: Sequence[str]) -> list[ClaimScore]:
        results: list[ClaimScore] = []
        for i in range(0, len(hypotheses), self.batch_size):
            chunk = list(hypotheses[i : i + self.batch_size])
            enc = self.tok(
                [premise] * len(chunk), chunk,
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(self.device)
            with self._torch.no_grad():
                probs = self.model(**enc).logits.softmax(dim=-1)
            for h, row in zip(chunk, probs, strict=True):
                results.append(
                    ClaimScore(
                        claim=h,
                        entailment=float(row[self.idx["entailment"]]),
                        contradiction=float(row[self.idx["contradiction"]]),
                        neutral=float(row[self.idx["neutral"]]),
                    )
                )
        return results


def check_faithfulness(
    answer: str,
    cited_text: str,
    decomposer: ClaimDecomposer,
    scorer: NLIScorer,
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
) -> FaithfulnessResult:
    """Run S2. Registered thresholds are 0.5 / 0.5; do not tune them."""
    claims = decomposer.decompose(answer)
    if not claims:
        return FaithfulnessResult(passed=False, reason="no_claims_extracted")
    if not cited_text.strip():
        return FaithfulnessResult(passed=False, reason="no_cited_text")

    scores = scorer.score(cited_text, claims)
    min_ent = min(s.entailment for s in scores)
    max_con = max(s.contradiction for s in scores)
    passed = min_ent >= entailment_threshold and max_con < contradiction_threshold

    return FaithfulnessResult(
        passed=passed,
        claims=scores,
        min_entailment=min_ent,
        max_contradiction=max_con,
        reason=None if passed else "claim_unsupported_or_contradicted",
    )
