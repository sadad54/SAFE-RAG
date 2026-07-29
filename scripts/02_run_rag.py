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

import json
import os
import sys
from pathlib import Path

from _common import ROOT, base_parser, get_corpus, paths  # noqa: E402
from tqdm import tqdm  # noqa: E402

from saferag.config import load_config  # noqa: E402
from saferag.data.obliqa import load_questions  # noqa: E402
from saferag.generation.generator import (  # noqa: E402
    build_generator,
    gpu_report,
    prompt_fingerprint,
    prompt_hash,
    render_prompt,
    write_prompt_template,
)
from saferag.generation.schema import check_schema  # noqa: E402
from saferag.retrieval.hybrid import HybridRetriever  # noqa: E402
from saferag.utils.io import read_jsonl, write_jsonl  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402
from saferag.utils.provenance import stamp  # noqa: E402

# Set before CUDA initialises. Reduces the fragmentation that turns a
# survivable allocation into an OOM on a nearly-full card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

log = get_logger("run_rag")


def _selftest(generator, prompts: list[str]) -> int:
    """Batched and unbatched generation must agree.

    Decoder-only models need LEFT padding for batched generation. Get it wrong and
    the model continues from pad tokens instead of from the prompt -- output stays
    fluent and plausible while being wrong, which is the worst kind of bug. This
    catches it in about a minute on the real model rather than after a long run.
    """
    if len(prompts) < 2:
        log.error("Need at least 2 prompts for the self-test.")
        return 1

    log.info("Generating %d prompts one at a time...", len(prompts))
    solo = [generator.generate([p])[0] for p in prompts]
    log.info("Generating the same %d as one batch...", len(prompts))
    batched = generator.generate(prompts)

    mismatches = [i for i, (a, b) in enumerate(zip(solo, batched, strict=True)) if a != b]

    print("\n" + "=" * 70)
    print("  BATCHING SELF-TEST")
    print("=" * 70)
    for i, (a, b) in enumerate(zip(solo, batched, strict=True)):
        print(f"    prompt {i}: {'MISMATCH' if a != b else 'ok'}")
        if a != b:
            print(f"      unbatched: {a[:110]!r}")
            print(f"      batched  : {b[:110]!r}")

    print()
    if mismatches:
        print("  FAILED. Batched output differs from unbatched.")
        print("  Almost always left-padding: check tokenizer.padding_side == 'left'")
        print("  and that pad_token is set. Use --batch-size 1 until it is fixed;")
        print("  results from a mismatching batch configuration are not trustworthy.")
        print("=" * 70 + "\n")
        return 1

    n_valid = sum(1 for o in batched if check_schema(o).valid)
    print("  PASSED. Batching is safe at this size.")
    print(f"  Schema-valid outputs: {n_valid}/{len(batched)}")
    if n_valid < len(batched):
        print("  Some output did not parse. Inspect before the full run:")
        for o in batched:
            if not check_schema(o).valid:
                print(f"    {o[:200]!r}")
                break
    print("=" * 70 + "\n")
    return 0


def main() -> int:
    ap = base_parser("Generate structured answers over ObliQA.")
    ap.add_argument("--backend", default=None, help="Override generation backend (vllm|hf|stub)")
    ap.add_argument("--limit", type=int, default=None, help="Override number of questions")
    ap.add_argument("--no-dense", action="store_true", help="BM25 only (skips torch entirely)")
    ap.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    ap.add_argument("--model", default=None, help="Override the generation model")
    ap.add_argument(
        "--preflight", action="store_true",
        help="Report GPU/VRAM and the recommended model size, then exit",
    )
    ap.add_argument(
        "--selftest", action="store_true",
        help="Generate 4 prompts unbatched and batched, check they agree, then exit",
    )
    args = ap.parse_args()

    if args.preflight:
        print("\n  PREFLIGHT")
        for k, v in gpu_report().items():
            print(f"    {k:<10} {v}")
        print()
        return 0

    cfg = load_config(args.config)
    p = paths(cfg)

    backend = args.backend or cfg.generation.backend
    model_name = args.model or cfg.generation.model
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

    log.info("Building generator: backend=%s model=%s", backend, model_name)
    if backend != "stub":
        for k, v in gpu_report().items():
            log.info("  %s: %s", k, v)
    gen_kwargs = {}
    if backend != "stub":
        gen_kwargs = {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "seed": cfg.generation.seed,
        }
        if backend == "vllm":
            gen_kwargs["temperature"] = cfg.generation.temperature
    generator = build_generator(backend, model_name, **gen_kwargs)

    # Retrieve first, so generation can be batched.
    log.info("Retrieving top-%d", cfg.retrieval.top_k)
    contexts = []
    for q in tqdm(questions, desc="retrieve"):
        hits = retriever.search(q.question, top_k=cfg.retrieval.top_k)
        contexts.append([(h.passage_id, corpus.get(h.passage_id, "")) for h in hits])

    prompts = [
        render_prompt(
            q.question, ctx,
            max_passage_chars=cfg.generation.max_passage_chars,
            max_total_chars=cfg.generation.max_total_prompt_chars,
        )
        for q, ctx in zip(questions, contexts, strict=True)
    ]

    if args.selftest:
        return _selftest(generator, prompts[:4])

    # Resume: generations are cached by question id as they complete, so a
    # disconnect or a crash costs the current batch, not the whole run.
    cache_path = p["interim"] / "_generations.jsonl"
    hashes = {q.question_id: prompt_hash(pr) for q, pr in zip(questions, prompts, strict=True)}
    # Entries written before prompt hashing existed carry no hash. They cannot be
    # trusted blindly: the length caps were introduced after them, so any prompt
    # the caps would change was produced under a different construction. For those
    # we compare against the uncapped render -- if capping makes no difference to
    # that prompt, the two constructions are byte-identical and the entry stands.
    uncapped = {
        q.question_id: prompt_hash(render_prompt(q.question, ctx, 10**9, 10**9))
        for q, ctx in zip(questions, contexts, strict=True)
    }

    cached: dict[str, str] = {}
    if cache_path.exists():
        stale = legacy_ok = 0
        for r in read_jsonl(cache_path):
            qid = r["item_id"]
            want = hashes.get(qid)
            got = r.get("prompt_hash")
            if got is None:
                # Legacy entry: keep only where capping is a no-op for this prompt.
                if uncapped.get(qid) == want:
                    legacy_ok += 1
                else:
                    stale += 1
                    continue
            elif got != want:
                stale += 1
                continue
            cached[qid] = r["raw_output"]
        log.info(
            "Resuming: %d cached (%d legacy verified identical), %d invalidated "
            "by prompt change", len(cached), legacy_ok, stale
        )

    todo = [i for i, q in enumerate(questions) if q.question_id not in cached]
    # Sort by prompt length. Everything in a batch pads up to the longest member,
    # so mixing a 1.8k-token prompt with an 8k one wastes most of the compute and
    # drives the peak attention allocation. Sorting makes batches near-uniform.
    todo.sort(key=lambda i: len(prompts[i]))

    log.info("Generating %d answers (%d cached)", len(todo), len(cached))
    with cache_path.open("a", encoding="utf-8") as cache_fh:
        for start in tqdm(range(0, len(todo), args.batch_size), desc="generate"):
            idx = todo[start : start + args.batch_size]
            outs = generator.generate([prompts[i] for i in idx])
            for i, out_text in zip(idx, outs, strict=True):
                qid = questions[i].question_id
                cached[qid] = out_text
                cache_fh.write(
                    json.dumps(
                        {"item_id": qid, "prompt_hash": hashes[qid], "raw_output": out_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            cache_fh.flush()

    raw_outputs = [cached[q.question_id] for q in questions]

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
            "generator": model_name if backend != "stub" else "stub",
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
