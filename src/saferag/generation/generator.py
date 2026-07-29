"""Generator backends.

Three backends:

* ``vllm``  -- fast batched local inference. Linux + CUDA.
* ``hf``    -- transformers. Runs anywhere torch runs, including Windows. Batched.
* ``stub``  -- deterministic fake output. No models, no GPU. Used by the tests and
               for exercising the whole pipeline end to end before committing real
               compute.

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


def render_prompt(
    question: str,
    passages: Sequence[tuple[str, str]],
    max_passage_chars: int = 4000,
    max_total_chars: int = 32000,
) -> str:
    """Build the generation prompt from (passage_id, text) pairs.

    Two length caps, both needed for different failure modes.

    ``max_passage_chars`` handles single monstrous records: the ADGM corpus has a
    152,000-character "passage" that is really the defined-terms glossary, against
    a median of 262. Uncapped it produces a ~38k-token prompt whose attention
    matrix will not fit on a 16 GB card. No question in ObliQA has a passage over
    8k characters in its gold set, so these giants are never the right answer and
    truncating them cannot make a question unanswerable.

    ``max_total_chars`` handles the other direction: ten individually-legal
    passages that together overflow. Passages are kept in rank order, so anything
    dropped is what the retriever ranked last.

    Truncation is marked in the text so it is visible to a human reading the
    prompt, and so it cannot be mistaken for the passage ending there.
    """
    kept: list[str] = []
    used = 0
    for pid, text in passages:
        if len(text) > max_passage_chars:
            text = text[:max_passage_chars] + " ...[truncated]"
        entry = f"[id: {pid}]\n{text}"
        if used + len(entry) > max_total_chars:
            break
        kept.append(entry)
        used += len(entry)
    return PROMPT_V1.format(passages="\n\n".join(kept), question=question)


def prompt_hash(prompt: str) -> str:
    """Short content hash, used to invalidate cached generations when the prompt
    that produced them no longer matches the prompt we would send today."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def prompt_fingerprint() -> str:
    """Hash of the prompt template, recorded in provenance."""
    return hashlib.sha256(PROMPT_V1.encode()).hexdigest()[:12]


def gpu_report() -> dict[str, object]:
    """What hardware is available, and what model size it can hold.

    Returned rather than printed so callers can put it in a provenance header.
    Degrades to a CPU verdict when torch is missing.
    """
    try:
        import torch
    except ImportError:
        return {"torch": False, "cuda": False, "advice": "torch not installed: pip install -e '.[models]'"}

    if not torch.cuda.is_available():
        # Distinguish "no GPU present" from "CPU-only wheel installed". Same
        # symptom, completely different fix, and `pip install torch` on Windows
        # silently gives you the CPU wheel.
        cpu_only_build = torch.version.cuda is None
        return {
            "torch": True,
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda or "NONE (CPU-only wheel)",
            "cuda": False,
            "advice": (
                "You installed the CPU-ONLY torch wheel. If this machine has an "
                "NVIDIA GPU, run `nvidia-smi` to confirm, then reinstall:\n"
                "    pip uninstall -y torch\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "If `nvidia-smi` is not found, there is no usable GPU here -- see "
                "docs/RUNNING_WITHOUT_A_GPU.md."
                if cpu_only_build else
                "torch has CUDA support but no device is visible. Check the driver, "
                "and that no other process is holding the GPU."
            ),
        }

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    if vram_gb >= 20:
        advice = "Fits a 7B model in fp16 comfortably."
    elif vram_gb >= 14:
        advice = "7B in fp16 is tight but usually fits. Reduce --batch-size if you OOM."
    elif vram_gb >= 8:
        advice = "Too small for 7B in fp16. Use a 3B model, e.g. Qwen/Qwen2.5-3B-Instruct."
    else:
        advice = "Use a 1.5B model, e.g. Qwen/Qwen2.5-1.5B-Instruct, and expect weaker output."

    return {
        "torch": True,
        "cuda": True,
        "device": props.name,
        "vram_gb": round(vram_gb, 1),
        "advice": advice,
    }


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
    """transformers backend, batched. Works on Windows.

    Decoder-only models must be LEFT-padded for batched generation: right padding
    puts pad tokens between the prompt and the first generated token, and the
    model continues from padding instead of from the prompt. This is the single
    most common cause of silently garbage batched output.
    """

    name = "hf"

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        seed: int = 20260728,
        dtype: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install -e '.[models]'") from exc

        set_seed(seed)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        resolved_dtype = (
            torch.float16 if dtype == "auto" and torch.cuda.is_available()
            else torch.float32 if dtype == "auto"
            else getattr(torch, dtype)
        )
        # transformers renamed torch_dtype -> dtype in v5. Try the new name, fall
        # back to the old one, so the same code runs on 4.x and 5.x.
        kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=resolved_dtype, **kwargs
            ).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=resolved_dtype, **kwargs
            ).eval()
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.max_new_tokens = max_new_tokens

    def _apply_template(self, prompt: str) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _generate_once(self, prompts: Sequence[str]) -> list[str]:
        texts = [self._apply_template(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=False).to(
            self.model.device
        )
        with self._torch.no_grad():
            ids = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [self.tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in ids]

    def generate(self, prompts: Sequence[str]) -> list[str]:
        """Generate, halving the batch on CUDA OOM rather than losing the run.

        Peak memory is driven by the LONGEST prompt in the batch, since everything
        pads up to it. Splitting on OOM is what keeps one outlier prompt from
        destroying hours of completed work.
        """
        if not prompts:
            return []
        try:
            return self._generate_once(prompts)
        except (RuntimeError, self._torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if len(prompts) == 1:
                raise RuntimeError(
                    "CUDA out of memory on a SINGLE prompt -- the batch size is not "
                    "the problem. This prompt is too long for the model on this GPU.\n"
                    "  Lower retrieval.top_k in the config, or use a smaller model."
                ) from exc
            self._torch.cuda.empty_cache()
            half = len(prompts) // 2
            # ponytail: recursive halving, so worst case is len(prompts) forward
            # passes at batch 1. Fine here because OOM is rare once prompts are
            # length-sorted; if it stops being rare, bucket by token count instead.
            return self.generate(prompts[:half]) + self.generate(prompts[half:])


class VLLMGenerator(Generator):
    """vLLM backend. Linux + CUDA, and the only one fast enough for the full split."""

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
