"""Step 00 -- fetch and validate ObliQA.

Does not download automatically. ObliQA is distributed from a GitHub repository
whose layout changes between releases, and a silent download of an unexpected
layout is worse than no download. This script tells you what to fetch, then
validates whatever you placed in data/raw/obliqa and reports exactly what it found.

    python scripts/00_fetch_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import ROOT, base_parser, paths  # noqa: E402

from saferag.config import load_config  # noqa: E402
from saferag.data.obliqa import ObliQAFormatError, adapt_record  # noqa: E402

INSTRUCTIONS = """
ObliQA was not found.

  1. Download from:  https://github.com/RegNLP/ObliQADataset
  2. Place the split files under:  {target}
  3. Rerun this script.

The paper describing ObliQA is RIRAG (arXiv:2409.05677). Cite it, and check its
licence terms before redistributing any of the corpus text.
"""


def main() -> int:
    ap = base_parser("Validate the ObliQA download.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = paths(cfg)

    files = sorted(list(p["raw"].glob("*.json")) + list(p["raw"].glob("*.jsonl")))
    if not files:
        print(INSTRUCTIONS.format(target=p["raw"]))
        return 1

    print(f"Found {len(files)} file(s) in {p['raw']}\n")
    ok = True

    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        try:
            records = json.loads(text) if text.startswith("[") else [
                json.loads(line) for line in text.splitlines() if line.strip()
            ]
        except json.JSONDecodeError as exc:
            print(f"  [FAIL] {path.name}: not valid JSON -- {exc}")
            ok = False
            continue

        if not records:
            print(f"  [WARN] {path.name}: empty")
            continue

        print(f"  {path.name}: {len(records)} records")
        print(f"    keys on first record: {sorted(records[0])}")

        try:
            q = adapt_record(records[0])
        except ObliQAFormatError as exc:
            print(f"    [FAIL] adapter did not recognise this layout:\n{exc}")
            ok = False
            continue

        print(f"    parsed -> id={q.question_id!r}, {len(q.gold_passage_ids)} gold passage(s)")
        print(f"    question: {q.question[:100]}...")
        if not q.gold_passage_ids:
            print("    [WARN] no gold passages on the first record. The S3 screen "
                  "joins on gold passage ids and will not work without them.")

    print()
    if ok:
        print("Validation passed. Next: python scripts/01_build_index.py")
    else:
        print("Validation FAILED. Fix the adapter aliases in src/saferag/data/obliqa.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
