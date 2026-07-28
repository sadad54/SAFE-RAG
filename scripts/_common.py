"""Shared argument parsing and path setup for the numbered scripts.

Scripts are runnable directly (``python scripts/03_run_filters.py``) without the
package being installed, by putting ``src`` on sys.path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pilot.yaml",
        help="Path to the run config (default: configs/pilot.yaml)",
    )
    return ap


def get_corpus(cfg, questions) -> dict[str, str]:  # noqa: ANN001
    """Full ADGM corpus if it has been downloaded, otherwise the gold-passage union.

    The fallback exists so the pipeline runs for a smoke test, but it is NOT valid
    for results: an index built only from cited passages has no true distractors,
    so retrieval is artificially easy and the measured rate means nothing.
    """
    from saferag.data.obliqa import build_passage_corpus, load_corpus_documents

    corpus_dir = ROOT / "data" / "raw" / "adgm"
    if corpus_dir.exists() and any(corpus_dir.rglob("*.json")):
        corpus = load_corpus_documents(corpus_dir)
        print(f"  corpus: {len(corpus)} passages from the full ADGM documents")
        return corpus

    corpus = build_passage_corpus(questions)
    print(
        f"  corpus: {len(corpus)} passages from cited gold passages only.\n"
        "  WARNING: no true distractors -- smoke testing only, not valid for results.\n"
        "  Unzip StructuredRegulatoryDocuments.zip into data/raw/adgm/ for a real run."
    )
    return corpus


def paths(cfg) -> dict[str, Path]:  # noqa: ANN001
    p = cfg.paths
    out = {
        "raw": ROOT / cfg.data.raw_dir,
        "index": ROOT / p.index_dir,
        "interim": ROOT / p.interim_dir,
        "annotation": ROOT / p.annotation_dir,
        "runs": ROOT / p.runs_dir,
    }
    for value in out.values():
        value.mkdir(parents=True, exist_ok=True)
    return out
