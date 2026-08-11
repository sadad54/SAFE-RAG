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
    ap.add_argument(
        "--run-name",
        default=None,
        help=(
            "Override config.run_name. interim/ and runs/ output is nested under "
            "this name so concurrent runs (e.g. the four ID-format ablation "
            "configurations) never overwrite each other's cache or results."
        ),
    )
    return ap


def load_cfg(args) -> "Config":  # noqa: ANN001, F821
    """Load the config for ``args.config`` and apply ``args.run_name`` if given.

    Every numbered script should use this instead of calling ``load_config``
    directly, so ``--run-name`` behaves identically everywhere.
    """
    from saferag.config import load_config

    cfg = load_config(args.config)
    if getattr(args, "run_name", None):
        cfg.run_name = args.run_name
    return cfg


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
    # interim/ and runs/ are nested under run_name: several configurations (e.g.
    # the four ID-format ablation runs) can now execute against the same repo
    # checkout without one run's generation cache or results overwriting
    # another's. index/ and annotation/ stay flat -- retrieval is unaffected by
    # the ablation, and annotation is not produced for it at all.
    run_name = cfg.get("run_name", "default")
    out = {
        "raw": ROOT / cfg.data.raw_dir,
        "index": ROOT / p.index_dir,
        "interim": ROOT / p.interim_dir / run_name,
        "annotation": ROOT / p.annotation_dir,
        "runs": ROOT / p.runs_dir / run_name,
    }
    for value in out.values():
        value.mkdir(parents=True, exist_ok=True)
    # Print the resolved root. A nested clone (SAFE-RAG/SAFE-RAG) makes one script
    # write where another cannot find it, and the only visible symptom is a
    # "not found" two steps later.
    print(f"  repo root: {ROOT}")
    return out
