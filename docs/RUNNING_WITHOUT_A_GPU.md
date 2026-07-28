# Running step 02 without a local GPU

Everything in this pipeline runs on CPU **except** step 02 (generation). Retrieval,
filtering, sampling, annotation and analysis are all fine on a laptop.

Work out which situation you are in first:

```bash
nvidia-smi
```

| Result | Situation | Go to |
|---|---|---|
| Prints a GPU table | You have a GPU, wrong torch wheel | **A** |
| "not recognized" / not found | No NVIDIA GPU on this machine | **B**, **C** or **D** |

---

## A. You have an NVIDIA GPU but the CPU-only wheel

`pip install torch` on Windows gives you the CPU build by default. Reinstall:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
python scripts/02_run_rag.py --preflight
```

`torch_cuda_build` should now show a version rather than `NONE`.

---

## B. Free hosted GPU (recommended if you have no local card)

Kaggle Notebooks give ~30 GPU-hours a week free; Colab's free tier gives a T4 with
shorter sessions. Either finishes the full 2,786-question run in well under an hour.

Only step 02 needs to run there. The plan:

1. Push your repo (already done).
2. In the notebook:

```python
!git clone https://github.com/sadad54/SAFE-RAG.git
%cd SAFE-RAG
!pip install -q -e ".[models]"

# ObliQA is not in the repo -- fetch it into data/raw/
!git clone -q https://github.com/RegNLP/ObliQADataset.git /tmp/obliqa
!mkdir -p data/raw/obliqa data/raw/adgm
!cp /tmp/obliqa/ObliQA_test.json data/raw/obliqa/
!cp /tmp/obliqa/StructuredRegulatoryDocuments/*.json data/raw/adgm/

!python scripts/02_run_rag.py --preflight
```

Read the preflight VRAM figure, then pick a model. On a free T4 (16 GB) a 7B model
in fp16 is ~15 GB of weights before any KV cache, so it will OOM at a useful batch
size. Either drop to 3B, or keep 7B at `--batch-size 2` and accept it is slower.

```python
MODEL = "Qwen/Qwen2.5-3B-Instruct"   # 7B needs --batch-size 2 on a 16 GB card

!python scripts/02_run_rag.py --selftest --limit 10 --model $MODEL
!python scripts/02_run_rag.py --model $MODEL --batch-size 8
```

Run the self-test first and do not skip it. It generates four prompts singly and
then as a batch and checks they match; a left-padding fault produces fluent,
plausible, wrong output that you would not catch by eye.

Then retrieve the one file you need:

```python
!ls -lh data/interim/answers.jsonl
```

It will be roughly 20 MB (each record carries the ten retrieved passages in full).
Download it and put it at `data/interim/answers.jsonl` locally.

3. Download `data/interim/answers.jsonl` and drop it into the same path locally.
4. Continue on your laptop from step 03 — the NLI model for S2 is small enough for
   CPU, though it will take a while.

The provenance header records the git SHA, config hash, seed and model revision,
so a run done in a notebook is exactly as reproducible as one done locally. Note
in the write-up where it ran.

---

## C. Your university cluster

The right long-term answer, and it is what `backend: vllm` in the config is for.
Worth setting up regardless, because the SAFE-RAG experiments after this pilot will
need far more compute than the pilot does.

---

## D. CPU only, no alternative

Possible but painful, and it constrains model choice hard.

**Measure before committing.** Do not guess at throughput:

```bash
python scripts/02_run_rag.py --selftest --limit 10 --model Qwen/Qwen2.5-1.5B-Instruct
```

Time it. The self-test generates 4 prompts singly plus 4 batched, so 8 generations.
Divide by 8 for a per-question figure and multiply by 2,786.

Rough expectation on a modern laptop CPU: 20–60 s per question for a 1.5B model at
these prompt lengths, i.e. **15–45 hours** for the full split. A 7B model is
several times worse and not worth attempting.

Two things to be aware of before going down this road:

- **A 1.5B model will fail S1 often.** Small models are unreliable at emitting
  strict JSON. A low schema-validity rate is a legitimate descriptive finding, but
  if it drops far enough there will not be 150 survivors to annotate.
- **Do not silently shrink the run to make it finish.** The pre-registration fixes
  n = 2,000. Running fewer is a deviation and belongs in Section 11 with a reason.

If the CPU route is the only option, run overnight in two halves with `--limit`,
and check the S1 pass rate on the first half before committing to the second.
