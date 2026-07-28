"""Step 02 -- run the plain RAG pipeline and generate structured answers.

    # laptop smoke test, no models needed
    python scripts/02_run_rag.py --backend stub --limit 20 --no-dense

    # cluster
    python scripts/02_run_rag.py

COMMIT docs/PREREGISTRATION.md BEFORE RUNNING THIS FOR THE FIRST TIME.
The study design must be fixed before any data is observed.

Output: data/interim/answers.jsonl, with a provenance header.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import ROOT, base_parser, get_corpus, paths  # noqa: E402
from tqdm import tqdm  # noqa: E402

from saferag.config import load_config  # noqa: E402
from saferag.data.obliqa import load_questions  # noqa: E402
from saferag.generation.generator import (  # noqa: E402
    build_generator,
    prompt_fingerprint,
    render_prompt,
    write_prompt_template,
)
from saferag.retrieval.hybrid import HybridRetriever  # noqa: E402
from saferag.utils.io import write_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402
from saferag.utils.provenance import stamp  # noqa: E402

log = get_logger("run_rag")


def main() -> int:
    ap = base_parser("Generate structured answers over ObliQA.")
    ap.add_argument("--backend", default=None, help="Override generation backend (vllm|hf|stub)")
    ap.add_argument("--limit", type=int, default=None, help="Override number of questions")
    ap.add_argument("--no-dense", action="store_true", help="BM25 only (skips torch entirely)")
    ap.add_argument("--batch-size", type=int, default=32, help="Generation batch size")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = paths(cfg)

    backend = args.backend or cfg.generation.backend
    limit = args.limit or cfg.data.n_questions

    files = sorted(list(p["raw"].glob("*.json")) + list(p["raw"].glob("*.jsonl")))
    if not files:
        log.error("No ObliQA files in %s. Run scripts/00_fetch_data.py first.", p["raw"])
        return 1

    log.info("Loading questions from %s", files[0].name)
    questions = list(load_questions(files[0], limit=limit))
    if not questions:
        log.error("No questions loaded.")
        return 1
    log.info("Loaded %d questions", len(questions))

    corpus = get_corpus(cfg, questions)
    log.info("Passage corpus: %d unique passages", len(corpus))
    if len(corpus) < 50:
        log.warning(
            "Corpus is very small (%d passages). Retrieval results will not be "
            "meaningful; this is fine for a smoke test only.", len(corpus)
        )

    log.info("Building retriever (dense=%s)", not args.no_dense)
    retriever = HybridRetriever(
        corpus,
        use_dense=not args.no_dense,
        dense_model=cfg.retrieval.dense.model,
        bm25_params={"k1": cfg.retrieval.bm25.k1, "b": cfg.retrieval.bm25.b},
        rrf_k=cfg.retrieval.fusion.rrf_k,
    )

    log.info("Building generator: backend=%s model=%s", backend, cfg.generation.model)
    gen_kwargs = {}
    if backend != "stub":
        gen_kwargs = {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "seed": cfg.generation.seed,
        }
        if backend == "vllm":
            gen_kwargs["temperature"] = cfg.generation.temperature
    generator = build_generator(backend, cfg.generation.model, **gen_kwargs)

    # Retrieve first, so generation can be batched.
    log.info("Retrieving top-%d", cfg.retrieval.top_k)
    contexts = []
    for q in tqdm(questions, desc="retrieve"):
        hits = retriever.search(q.question, top_k=cfg.retrieval.top_k)
        contexts.append([(h.passage_id, corpus.get(h.passage_id, "")) for h in hits])

    prompts = [render_prompt(q.question, ctx) for q, ctx in zip(questions, contexts, strict=True)]

    log.info("Generating %d answers", len(prompts))
    raw_outputs: list[str] = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="generate"):
        raw_outputs.extend(generator.generate(prompts[i : i + args.batch_size]))

    records = []
    for q, ctx, raw in zip(questions, contexts, raw_outputs, strict=True):
        records.append(
            {
                "item_id": q.question_id,
                "question": q.question,
                "gold_passage_ids": q.gold_passage_ids,
                "retrieved_passage_ids": [pid for pid, _ in ctx],
                "retrieved_passages": [{"id": pid, "text": txt} for pid, txt in ctx],
                "raw_output": raw,
            }
        )

    out = p["interim"] / "answers.jsonl"
    prov = stamp(
        script="02_run_rag.py",
        config=cfg,
        config_path=str(args.config),
        seed=cfg.seed,
        models={
            "generator": cfg.generation.model if backend != "stub" else "stub",
            "dense_retriever": "none" if args.no_dense else cfg.retrieval.dense.model,
        },
        backend=backend,
        n_questions=len(questions),
        top_k=cfg.retrieval.top_k,
        prompt_fingerprint=prompt_fingerprint(),
    )
    n = write_jsonl(out, records, provenance=prov)
    write_prompt_template(ROOT / "prompts" / "answer_v1.txt")

    log.info("Wrote %d records to %s", n, out)
    log.info("Next: python scripts/03_run_filters.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
