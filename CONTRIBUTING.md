# Working on this repository

## Before you run anything that produces a number

1. `pytest` must be green.
2. `docs/PREREGISTRATION.md` must be committed.
3. The working tree must be clean. Every artefact records the commit SHA that
   produced it; a dirty tree makes that record a lie, and the scripts will warn
   you.

## Smoke-testing without models or a GPU

`tests/fixtures_synthetic_obliqa.json` is 300 fabricated ADGM-style records. It
is synthetic and is for exercising the pipeline only — never for results.

```bash
mkdir -p data/raw/obliqa
cp tests/fixtures_synthetic_obliqa.json data/raw/obliqa/test.json

python scripts/00_fetch_data.py
python scripts/01_build_index.py --no-dense --sample 100
python scripts/02_run_rag.py --backend stub --no-dense --limit 300
python scripts/03_run_filters.py --stub-nli
python scripts/04_make_annotation_batch.py
```

Anything using `--stub`, `--stub-nli` or `--backend stub` produces fake numbers by
construction. They exist to prove the plumbing works, nothing more.

## Registered values

Anything marked `[REGISTERED]` in `configs/pilot.yaml` is fixed by the
pre-registration. Changing one after step 02 has run requires an entry in
Section 11 of `docs/PREREGISTRATION.md`, with a date and a reason.

## Style

`ruff check . && ruff format .` before committing. Line length 100.
