"""ObliQA loading and validation.

ObliQA (RIRAG, arXiv:2409.05677) is derived from Abu Dhabi Global Markets
regulatory text. Source: https://github.com/RegNLP/ObliQADataset

FIELD NAMES ARE NOT ASSUMED. The upstream JSON layout has changed between
releases, and silently guessing at it is how a pipeline ends up measuring
nothing. ``adapt_record`` below tries a list of known aliases and raises a loud,
specific error if none matches, telling you exactly which keys were present.

If it raises: open one record from the download, read the keys, and add them to
the alias lists. That is a two-minute fix and it is deliberately your decision,
not a guess made on your behalf.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Known aliases, most likely first. Extend after inspecting the real download.
_QID_KEYS = ("QuestionID", "question_id", "qid", "id")
_QUESTION_KEYS = ("Question", "question", "query", "text")
_PASSAGES_KEYS = ("Passages", "passages", "relevant_passages", "positives", "contexts")
_PID_KEYS = ("PassageID", "passage_id", "pid", "id")
_DOCID_KEYS = ("DocumentID", "document_id", "doc_id")
_PTEXT_KEYS = ("Passage", "passage", "text", "content", "body")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return None


class ObliQAFormatError(RuntimeError):
    """Raised when a record does not match any known ObliQA layout."""


@dataclass
class Passage:
    passage_id: str
    text: str


@dataclass
class Question:
    question_id: str
    question: str
    gold_passage_ids: list[str] = field(default_factory=list)
    gold_passages: list[Passage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "gold_passage_ids": self.gold_passage_ids,
        }


def adapt_record(record: dict[str, Any]) -> Question:
    """Map one raw ObliQA record onto the internal representation."""
    qid = _first(record, _QID_KEYS)
    question = _first(record, _QUESTION_KEYS)

    if qid is None or question is None:
        raise ObliQAFormatError(
            "Could not find question id / question text in an ObliQA record.\n"
            f"  Keys present : {sorted(record)}\n"
            f"  Tried for id : {_QID_KEYS}\n"
            f"  Tried for text: {_QUESTION_KEYS}\n"
            "Open one record from the download, read the real key names, and add "
            "them to the alias tuples at the top of src/saferag/data/obliqa.py."
        )

    passages: list[Passage] = []
    raw_passages = _first(record, _PASSAGES_KEYS) or []
    if isinstance(raw_passages, dict):
        raw_passages = [raw_passages]

    for i, p in enumerate(raw_passages):
        if isinstance(p, str):
            passages.append(Passage(passage_id=f"{qid}::p{i}", text=p))
            continue
        if not isinstance(p, dict):
            continue
        # ObliQA's PassageID is only unique WITHIN a document ("3)", "1.1.2"),
        # so the key must be composite. Joining on PassageID alone collides
        # across the 40 ADGM documents and silently corrupts the S3 screen.
        pid = _first(p, _PID_KEYS) or f"p{i}"
        doc = _first(p, _DOCID_KEYS)
        full_id = f"{doc}::{pid}" if doc is not None else f"{qid}::{pid}"
        ptext = _first(p, _PTEXT_KEYS) or ""
        passages.append(Passage(passage_id=full_id, text=str(ptext)))

    return Question(
        question_id=str(qid),
        question=str(question),
        gold_passage_ids=[p.passage_id for p in passages],
        gold_passages=passages,
    )


def load_questions(path: str | Path, limit: int | None = None) -> Iterator[Question]:
    """Load ObliQA questions from a JSON array or a JSONL file.

    Raises ObliQAFormatError with a readable message if the layout is unknown.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ObliQA file not found: {path}\n"
            "Run scripts/00_fetch_data.py, or download from "
            "https://github.com/RegNLP/ObliQADataset and place it there."
        )

    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    for i, record in enumerate(records):
        if limit is not None and i >= limit:
            return
        yield adapt_record(record)


def load_corpus_documents(directory: str | Path) -> dict[str, str]:
    """Load the full ADGM corpus from StructuredRegulatoryDocuments.

    Records are flat: ``{"ID", "DocumentID", "PassageID", "Passage"}``.

    Prefer this over build_passage_corpus. Indexing only the passages that some
    question cites makes retrieval artificially easy -- every passage in the index
    is somebody's gold answer, so there are no true distractors and the measured
    deceptive-grounding rate is not meaningful.
    """
    directory = Path(directory)
    files = sorted(directory.rglob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No structured document JSON under {directory}.\n"
            "Download StructuredRegulatoryDocuments.zip from "
            "https://github.com/RegNLP/ObliQADataset and unzip it there."
        )
    corpus: dict[str, str] = {}
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        for rec in records:
            doc, pid = _first(rec, _DOCID_KEYS), _first(rec, _PID_KEYS)
            text = _first(rec, _PTEXT_KEYS)
            if doc is None or pid is None or not text:
                continue
            corpus[f"{doc}::{pid}"] = str(text)
    if not corpus:
        raise ObliQAFormatError(
            f"Found {len(files)} JSON file(s) under {directory} but no record had "
            "DocumentID + PassageID + Passage. Check the unzipped layout."
        )
    return corpus


def build_passage_corpus(questions: list[Question]) -> dict[str, str]:
    """Collect the unique passage id -> text map across all questions.

    ObliQA ships passages attached to questions rather than as a standalone
    corpus, so the retrieval index is built from their union. Duplicate ids are
    checked for text conflicts, which would indicate an id collision.
    """
    corpus: dict[str, str] = {}
    conflicts = 0
    for q in questions:
        for p in q.gold_passages:
            if p.passage_id in corpus and corpus[p.passage_id] != p.text:
                conflicts += 1
            corpus[p.passage_id] = p.text
    if conflicts:
        print(
            f"WARNING: {conflicts} passage id(s) mapped to conflicting text. "
            "Passage ids may not be unique in this release; check before trusting "
            "the S3 screen, which joins on exactly these ids."
        )
    return corpus
