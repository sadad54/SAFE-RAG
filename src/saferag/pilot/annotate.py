"""Terminal annotation tool.

    python -m saferag.pilot.annotate --batch data/annotation/batch_01.jsonl --annotator adnan

Shows one item at a time. Saves after every keystroke, so quitting and resuming
loses nothing. Labels are written to a separate file per annotator, which keeps
the second annotator's labels genuinely independent.

Read docs/ANNOTATION_GUIDELINES.md before using this.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from saferag.utils.io import read_jsonl

VALID = {"a": "A", "b": "B", "c": "C", "n": "NA"}

BANNER = """
+--------------------------------------------------------------------------+
|  GROUNDING ANNOTATION                                                     |
|                                                                           |
|  Judge ONE thing: does the cited passage govern what the question asked?  |
|                                                                           |
|   [a] A  - correct passage, right scope                                   |
|   [b] B  - on topic, but a DIFFERENT entity / category / clause           |
|   [c] C  - plainly irrelevant passage                                     |
|   [n] NA - cannot judge (malformed item)                                  |
|                                                                           |
|   [s] skip   [u] undo last   [?] help   [q] save and quit                 |
+--------------------------------------------------------------------------+
"""

HELP = """
B vs C -- the distinction the whole study rests on:

  The EFFORT test
    Spotted instantly, no domain knowledge needed .............. C
    Had to compare the question's scope to the passage's ....... B

  The PLAUSIBILITY test
    A busy professional would never accept this ................ C
    A busy professional might well accept it ................... B

  Several passages cited, at least one correctly scoped ........ A
  Passage correct but the answer misreads it ................... A
  Question itself is vague about scope ......................... A
  Question unanswerable / item malformed ....................... NA

Full guidelines: docs/ANNOTATION_GUIDELINES.md
"""


def _wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    out = []
    for para in str(text).split("\n"):
        out.append(
            textwrap.fill(para, width=width, initial_indent=indent, subsequent_indent=indent)
            if para.strip()
            else ""
        )
    return "\n".join(out)


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _render(item: dict[str, Any], idx: int, total: int, done: int) -> None:
    _clear()
    print(BANNER)
    print(f"  Item {idx + 1} of {total}   |   labelled so far: {done}")
    print("=" * 90)
    print("\n  QUESTION\n")
    print(_wrap(item.get("question", "<missing>")))
    print("\n  SYSTEM ANSWER\n")
    print(_wrap(item.get("answer", "<missing>")))
    print("\n  CITED PASSAGE(S)\n")
    passages = item.get("cited_passages") or []
    if not passages:
        print("    <no passage text available>")
    for i, p in enumerate(passages, 1):
        pid = p.get("id", "?") if isinstance(p, dict) else "?"
        body = p.get("text", str(p)) if isinstance(p, dict) else str(p)
        print(f"    [{i}] id={pid}")
        print(_wrap(body, indent="        "))
        print()
    print("=" * 90)


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels[rec["item_id"]] = rec
    return labels


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite(path: Path, labels: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in labels.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Annotate grounding correctness.")
    ap.add_argument("--batch", required=True, type=Path, help="Batch JSONL from step 04")
    ap.add_argument("--annotator", required=True, help="Your name or initials")
    ap.add_argument("--out", type=Path, default=None, help="Label output path")
    args = ap.parse_args(argv)

    if not args.batch.exists():
        print(f"Batch file not found: {args.batch}", file=sys.stderr)
        return 1

    items = list(read_jsonl(args.batch))
    if not items:
        print("Batch is empty.", file=sys.stderr)
        return 1

    out_path = args.out or args.batch.parent / f"labels_{args.annotator}_{args.batch.stem}.jsonl"
    labels = _load_existing(out_path)

    if labels:
        print(f"Resuming: {len(labels)} of {len(items)} already labelled.")
        input("Press Enter to continue... ")

    order = [it for it in items if it.get("item_id") not in labels]
    history: list[str] = []
    i = 0

    while i < len(order):
        item = order[i]
        item_id = item.get("item_id")
        if item_id is None:
            print("Item missing 'item_id'; batch file is malformed.", file=sys.stderr)
            return 1

        _render(item, len(labels), len(items), len(labels))
        choice = input("  label > ").strip().lower()

        if choice == "q":
            break
        if choice == "?":
            _clear()
            print(HELP)
            input("\nPress Enter to return... ")
            continue
        if choice == "s":
            i += 1
            continue
        if choice == "u":
            if not history:
                input("  Nothing to undo. Press Enter... ")
                continue
            last = history.pop()
            labels.pop(last, None)
            _rewrite(out_path, labels)
            i = max(0, i - 1)
            continue
        if choice not in VALID:
            input("  Not a valid key. Press Enter... ")
            continue

        note = input("  note (optional, Enter to skip) > ").strip()
        record = {
            "item_id": item_id,
            "label": VALID[choice],
            "note": note,
            "annotator": args.annotator,
            "labelled_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        labels[item_id] = record
        _append(out_path, record)
        history.append(item_id)
        i += 1

    _clear()
    counts: dict[str, int] = {}
    for rec in labels.values():
        counts[rec["label"]] = counts.get(rec["label"], 0) + 1

    print(f"\n  Saved {len(labels)} of {len(items)} labels to {out_path}\n")
    for lab in ("A", "B", "C", "NA"):
        if lab in counts:
            print(f"    {lab:<3} {counts[lab]:>4}")
    remaining = len(items) - len(labels)
    if remaining:
        print(f"\n  {remaining} remaining. Rerun the same command to resume.\n")
    else:
        print("\n  Batch complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
