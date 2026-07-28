"""Step 01 -- build the passage corpus and report retrieval sanity.

    python scripts/01_build_index.py --no-dense

Builds the corpus, runs BM25 over a sample of questions, and reports recall@k
against the gold passages. This is the reproduction check: if recall@10 is
implausibly low, retrieval is broken and nothing downstream means anything.
Find that out here, not after generating 2,000 answers.
"""

from __future__ import annotations

import json
import sys

from _common import base_parser, paths  # noqa: E402
from tqdm import tqdm  # noqa: E402

from saferag.config import load_config  # noqa: E402
from saferag.data.obliqa import build_passage_corpus, load_questions  # noqa: E402
from saferag.retrieval.hybrid import HybridRetriever  # noqa: E402
from saferag.utils.logging import get_logger  # noqa: E402

log = get_logger("index")


def main() -> int:
    ap = base_parser("Build the corpus and check retrieval quality.")
    ap.add_argument("--no-dense", action="store_true", help="BM25 only")
    ap.add_argument("--sample", type=int, default=200, help="Questions for the recall check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = paths(cfg)

    files = sorted(list(p["raw"].glob("*.json")) + list(p["raw"].glob("*.jsonl")))
    if not files:
        log.error("No ObliQA files in %s. Run scripts/00_fetch_data.py first.", p["raw"])
        return 1

    questions = list(load_questions(files[0], limit=cfg.data.n_questions))
    corpus = build_passage_corpus(questions)
    log.info("%d questions, %d unique passages", len(questions), len(corpus))

    lengths = [len(t.split()) for t in corpus.values()]
    if lengths:
        lengths.sort()
        log.info(
            "Passage length (words): min=%d median=%d p95=%d max=%d",
            lengths[0], lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.95)], lengths[-1],
        )

    retriever = HybridRetriever(
        corpus,
        use_dense=not args.no_dense,
        dense_model=cfg.retrieval.dense.model,
        bm25_params={"k1": cfg.retrieval.bm25.k1, "b": cfg.retrieval.bm25.b},
        rrf_k=cfg.retrieval.fusion.rrf_k,
    )

    sample = questions[: args.sample]
    ks = (1, 5, 10)
    hits = {k: 0 for k in ks}
    evaluated = 0

    for q in tqdm(sample, desc="recall check"):
        if not q.gold_passage_ids:
            continue
        evaluated += 1
        gold = set(q.gold_passage_ids)
        results = retriever.search(q.question, top_k=max(ks))
        ids = [h.passage_id for h in results]
        for k in ks:
            if gold & set(ids[:k]):
                hits[k] += 1

    print("\n  RETRIEVAL SANITY")
    if not evaluated:
        print("    No questions had gold passages. The S3 screen cannot work.")
        return 1
    for k in ks:
        print(f"    recall@{k:<3} {hits[k] / evaluated:.3f}   ({hits[k]}/{evaluated})")

    r10 = hits[10] / evaluated
    print()
    if r10 < 0.30:
        print("    WARNING: recall@10 below 0.30. Retrieval is probably broken.")
        print("    Check passage-id consistency and tokenisation before continuing.")
    elif r10 > 0.98:
        print("    WARNING: recall@10 above 0.98. Suspiciously high -- the index may")
        print("    contain only each question's own gold passages, which would make")
        print("    retrieval trivial and the whole pilot meaningless. Verify the corpus.")
    else:
        print("    Plausible. Compare against the published ObliQA baseline before trusting it.")

    stats = {
        "n_questions": len(questions),
        "n_passages": len(corpus),
        "recall": {f"@{k}": hits[k] / evaluated for k in ks},
        "n_evaluated": evaluated,
        "dense": not args.no_dense,
    }
    (p["runs"] / "retrieval_check.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n  Saved {p['runs'] / 'retrieval_check.json'}")
    print("  Next: python scripts/02_run_rag.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
