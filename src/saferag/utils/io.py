"""JSONL read/write. All intermediate artefacts in this project are JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, skipping the provenance header if present.

    The header is a single line whose only key is ``_provenance``.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if isinstance(record, dict) and set(record) == {"_provenance"}:
                continue
            yield record


def read_provenance(path: str | Path) -> dict[str, Any] | None:
    """Return the provenance header of a JSONL file, or None if absent."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and set(record) == {"_provenance"}:
                return record["_provenance"]
            return None
    return None


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> int:
    """Write records to JSONL, optionally preceded by a provenance header.

    Returns the number of data records written (excluding the header).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        if provenance is not None:
            fh.write(json.dumps({"_provenance": provenance}, ensure_ascii=False) + "\n")
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
