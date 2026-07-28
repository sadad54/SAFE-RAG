# SAFE-RAG

**Schema-Grounded, Faithfulness-Aware Self-Correction for Retrieval-Augmented Generation in Financial Compliance Question Answering**

Adnan Mashrur Sadad · Malaysia-Japan International Institute of Technology (MJIIT), Universiti Teknologi Malaysia
Supervisor: Dr Siti Nur Khadijah Aishah Ibrahim

---

## What this repository is, right now

This repository is currently a **measurement study**, not a framework implementation.

Before building SAFE-RAG, one number has to exist: how often does *deceptive grounding* actually occur in retrieval-augmented QA over financial regulation? A deceptively grounded answer is one that parses against the required schema, is fully entailed by the passage it cites, and is still wrong — because the cited passage governs a different regulated entity, licence category, or clause than the question asked about.

The SAFE-RAG framework devotes an entire repair branch to this failure. If it is rare, that branch is unmotivated. Nobody has measured it in this domain.

**The pilot measures it.** The framework comes after.

Read, in order:

1. [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) — the study design, fixed in advance. Operational definitions, sampling plan, estimator, and the pre-committed decision rule and pivots.
2. [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md) — how items are labelled A / B / C, with worked examples.

## The pilot in one picture

```
2,000 ObliQA questions
        |
        v
  plain RAG (BM25 + dense -> generate structured answer)
        |
        v
  [S1] does it parse against the schema?          -- automatic
        |
        v
  [S2] is every claim entailed by what it cites?  -- automatic
        |
        v
  [S3] three-way screen                           -- automatic
        |
        +-- cites a gold passage ............ control ............. 50 --.
        +-- no gold cited, gold WAS in context  candidate_recoverable  80 -+
        +-- no gold cited, gold NOT retrieved   candidate_unrecoverable 20 -+
                                                                           |
                                                                           v
                                                        label A / B / C by hand
                                              (2nd annotator on 50 -> Cohen's kappa)
                                                                           |
                                                                           v
                                            r = sum of w_s * b_s = the rate
```

**A** = cited passage governs what the question asked about.
**B** = topically adjacent, but a different entity / category / clause. *This is the phenomenon.*
**C** = plainly irrelevant passage. Ordinary retrieval failure, not interesting.

`candidate_recoverable` is where deceptive grounding lives: the model had the right
passage in front of it and cited a neighbour anyway. On ObliQA, BM25 top-10 recall
is ~0.82, so ~16% of questions never see their gold passage at all — those land in
`candidate_unrecoverable` and are label C by construction.

## Install

Requires **Python 3.11**.

```bash
git clone <your-repo-url> SAFE-RAG
cd SAFE-RAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"          # core + test tooling, CPU only
pip install -e ".[dev,models]"   # adds torch, sentence-transformers, transformers
```

On the GPU cluster (Linux) add fast batched generation:

```bash
pip install -e ".[models,serve]"   # adds vLLM -- Linux + CUDA only
```

`vllm` will not install on Windows. That is expected. Develop and test on your laptop with the core install; run generation on the cluster.

Verify the install:

```bash
pytest
```

All tests must pass before you run anything. They cover the statistics, which is the part of this study that has to be right.

## Running the pilot

Each script is standalone and writes a provenance header (git SHA, config hash, seeds, model revisions) into its output.

```bash
python scripts/00_fetch_data.py            # fetch + validate ObliQA
python scripts/01_build_index.py           # BM25 + dense index
python scripts/02_run_rag.py               # generate answers        [GPU]
python scripts/03_run_filters.py           # S1, S2, S3
python scripts/04_make_annotation_batch.py # stratified sample -> batch files
# ... annotate by hand ...
python -m saferag.pilot.annotate --batch data/annotation/batch_01.jsonl --annotator adnan
python scripts/05_compute_results.py       # base rate, CI, kappa
python scripts/06_random_routing_check.py  # typed vs random routing at equal budget
```

**Commit `docs/PREREGISTRATION.md` before running `02_run_rag.py` for the first time.** Its value is entirely in having been fixed before observation.

### Step 06 is not optional

`06_random_routing_check.py` compares routing repairs by diagnosed type against routing them **at random**, at an identical call budget. If typed routing does not beat random routing, the diagnosis carries no information and the SAFE-RAG architecture is an expensive coin flip. This costs one day and de-risks four months. Run it before building anything.

## Layout

```
docs/          pre-registration, annotation guidelines, proposals
configs/       YAML run configs (pilot.yaml is the registered one)
src/saferag/
  data/        ObliQA loading + schema validation
  retrieval/   BM25, dense, hybrid fusion
  generation/  answer schema, generator backends
  checks/      S1 schema, S2 faithfulness, S3 attribution
  pilot/       sampling, annotation CLI, statistics
  utils/       provenance, logging, io
scripts/       numbered runners, 00 -> 06
tests/         pytest suite
data/          gitignored except annotation labels
```

## Status

| Component | State |
|---|---|
| Pre-registration | Done |
| Annotation guidelines | Done |
| Statistics (Wilson, kappa, stratified estimator, bootstrap) | Done, tested |
| Answer schema + S1 check | Done, tested |
| Stratified sampler | Done, tested |
| Annotation CLI | Done |
| Provenance logging | Done |
| BM25 retrieval | Done. recall@10 = 0.82 on the full ADGM corpus |
| ObliQA loader | Verified against the real download (2,786 test questions, 13,016 passages) |
| Dense retrieval | Scaffolded, needs GPU run |
| Generator | Scaffolded, needs model choice + GPU run |
| S2 faithfulness (claim decomposition + NLI) | Scaffolded, needs model choice |
| SAFE-RAG framework itself | Not started, and should not be until the pilot reports |

## Licence

MIT. See [`LICENSE`](LICENSE).
