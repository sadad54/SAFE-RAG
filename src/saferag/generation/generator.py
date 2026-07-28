"""Generator backends.

Three backends:

* ``vllm``  -- fast batched local inference. Linux + CUDA. Use this on the cluster.
* ``hf``    -- transformers fallback. Slow, but runs anywhere torch runs.
* ``stub``  -- deterministic fake output. No models, no GPU. Used by the tests and
               useful for exercising the whole pipeline end to end on a laptop
               before committing cluster time.

Temperature is pinned to 0 and the seed is recorded, because the pilot's numbers
must be reproducible from the manifest.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

PROMPT_V1 = """You answer questions about financial regulation using only the passages provided.

Return ONLY a JSON object, with no commentary and no markdown fence, in exactly this form:
{{"answer": "<your answer>", "cited_passage_ids": ["<id>", ...], "obligations": ["<obligation>", ...]}}

Rules:
- Use only the passages given below. Do not use outside knowledge.
- cited_passage_ids must list the ids of the passages you actually relied on, and must not be empty.
- obligations may be an empty list if the question does not concern obligations.

PASSAGES
{passages}

QUESTION
{question}

JSON:"""


def render_prompt(question: str, passages: Sequence[tuple[str, str]]) -> str:
    """Build the generation prompt from (passage_id, text) pairs."""
    block = "\n\n".join(f"[id: {pid}]\n{text}" for pid, text in passages)
    return PROMPT_V1.format(passages=block, question=question)


def prompt_fingerprint() -> str:
    """Hash of the prompt template, recorded in provenance."""
    return hashlib.sha256(PROMPT_V1.encode()).hexdigest()[:12]


class Generator(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, prompts: Sequence[str]) -> list[str]:
        """Return one completion per prompt, in order."""


class StubGenerator(Generator):
    """Deterministic fake generator. No models required.

    Cites the first passage id it finds in the prompt and echoes that passage's
    first sentence as the answer. Echoing means the output is trivially entailed
    by what it cites, so it survives S2 and the whole chain -- filters, sampling,
    annotation, routing -- can be smoke-tested on a laptop with no models.

    The answers are meaningless. This exists to test plumbing, never to produce
    results.
    """

    name = "stub"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        out = []
        for prompt in prompts:
            first_id: str | None = None
            first_text: list[str] = []
            capturing = False
            for line in prompt.splitlines():
                if line.startswith("[id: "):
                    if first_id is not None:
                        break
                    first_id = line.split("[id: ", 1)[1].rstrip("]").strip()
                    capturing = True
                    continue
                if capturing and line.strip():
                    first_text.append(line.strip())

            body = " ".join(first_text)
            answer = (body.split(". ", 1)[0].strip() + ".") if body else "No passage available."
            payload = {
                "answer": answer,
                "cited_passage_ids": [first_id] if first_id else ["UNKNOWN"],
                "obligations": [],
            }
            out.append(json.dumps(payload))
        return out


class HFGenerator(Generator):
    """transformers backend. Correct but slow; fine for a few hundred items."""

    name = "hf"

    def __init__(self, model: str, max_new_tokens: int = 512, seed: int = 20260728) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install -e '.[models]'") from exc
        set_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map="auto"
        )
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def generate(self, prompts: Sequence[str]) -> list[str]:
        outputs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tok(text, return_tensors="pt").to(self.model.device)
            with self._torch.no_grad():
                ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            outputs.append(
                self.tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            )
        return outputs


class VLLMGenerator(Generator):
    """vLLM backend. Use this on the cluster; it is the only one fast enough for 2,000 items."""

    name = "vllm"

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        seed: int = 20260728,
        temperature: float = 0.0,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "vLLM is Linux + CUDA only: pip install -e '.[serve]' on the cluster."
            ) from exc
        self.llm = LLM(model=model, seed=seed)
        self.params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=seed)

    def generate(self, prompts: Sequence[str]) -> list[str]:
        results = self.llm.generate(list(prompts), self.params)
        return [r.outputs[0].text for r in results]


def build_generator(backend: str, model: str, **kwargs) -> Generator:
    """Factory. ``backend`` is one of vllm | hf | stub."""
    backend = backend.lower()
    if backend == "stub":
        return StubGenerator()
    if backend == "hf":
        return HFGenerator(model, **kwargs)
    if backend == "vllm":
        return VLLMGenerator(model, **kwargs)
    raise ValueError(f"Unknown generation backend {backend!r}. Use vllm, hf, or stub.")


def write_prompt_template(path: str | Path) -> None:
    """Persist the prompt template alongside a run, for the record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROMPT_V1, encoding="utf-8")
