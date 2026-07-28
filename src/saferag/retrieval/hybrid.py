"""Retrieval: BM25, dense, and reciprocal rank fusion.

BM25 works out of the box on CPU. Dense retrieval requires the ``models`` extra
and is imported lazily so that the pilot's CPU-only paths -- and the test suite --
do not need torch installed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation.

    >>> tokenize("Category 2 Authorised Person's capital!")
    ['category', '2', 'authorised', 'person', 's', 'capital']
    """
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    passage_id: str
    score: float
    rank: int


class BM25Retriever:
    """Thin wrapper over rank_bm25 that keeps passage ids alongside the index."""

    def __init__(self, corpus: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. pip install rank-bm25"
            ) from exc
        if not corpus:
            raise ValueError("Cannot build a BM25 index over an empty corpus.")
        self.passage_ids = list(corpus)
        self.texts = [corpus[pid] for pid in self.passage_ids]
        self._bm25 = BM25Okapi([tokenize(t) for t in self.texts], k1=k1, b=b)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            Hit(passage_id=self.passage_ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


class DenseRetriever:
    """sentence-transformers bi-encoder. Requires the ``models`` extra."""

    def __init__(
        self,
        corpus: dict[str, str],
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Dense retrieval needs the models extra: pip install -e '.[models]'"
            ) from exc
        self.passage_ids = list(corpus)
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.embeddings = self.model.encode(
            [corpus[pid] for pid in self.passage_ids],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        import numpy as np

        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=self.normalize
        )[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(passage_id=self.passage_ids[int(i)], score=float(scores[int(i)]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


def reciprocal_rank_fusion(
    run_lists: Sequence[Sequence[Hit]], k: int = 60, top_k: int = 10
) -> list[Hit]:
    """Reciprocal rank fusion: score(d) = sum over runs of 1 / (k + rank(d)).

    Rank-based, so it needs no score normalisation between BM25 and cosine
    similarity -- which is exactly why it is used here.

    >>> a = [Hit("p1", 9.0, 1), Hit("p2", 8.0, 2)]
    >>> b = [Hit("p2", 0.9, 1), Hit("p3", 0.8, 2)]
    >>> [h.passage_id for h in reciprocal_rank_fusion([a, b], top_k=3)]
    ['p2', 'p1', 'p3']
    """
    fused: dict[str, float] = {}
    for run in run_lists:
        for hit in run:
            fused[hit.passage_id] = fused.get(hit.passage_id, 0.0) + 1.0 / (k + hit.rank)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [
        Hit(passage_id=pid, score=score, rank=rank)
        for rank, (pid, score) in enumerate(ordered, start=1)
    ]


class HybridRetriever:
    """BM25 + dense, fused by RRF. Falls back to BM25 alone if dense is disabled."""

    def __init__(
        self,
        corpus: dict[str, str],
        use_dense: bool = True,
        dense_model: str = "BAAI/bge-base-en-v1.5",
        bm25_params: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = BM25Retriever(corpus, **(bm25_params or {}))
        self.dense = DenseRetriever(corpus, model_name=dense_model) if use_dense else None
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        runs = [self.bm25.search(query, top_k=top_k * 2)]
        if self.dense is not None:
            runs.append(self.dense.search(query, top_k=top_k * 2))
        if len(runs) == 1:
            return runs[0][:top_k]
        return reciprocal_rank_fusion(runs, k=self.rrf_k, top_k=top_k)
