# Getting the repo running on the university GPU cluster

You have AnyDesk access to the cluster (obtained 2026-08-03, see
`docs/PROJECT_STATE.md`) but nothing is cloned there yet. This is the one-time
setup, then how to run the ID-format ablation once it's done.

Do this over AnyDesk, in a terminal on the machine AnyDesk connects you to.

---

## 0. Figure out what you actually have

AnyDesk to a login-node desktop can mean two different things, and it changes
everything below:

- **A dedicated box** (assigned to you, or your lab) — you can run things
  directly in a terminal.
- **A shared login node in front of a Slurm cluster** — the login node is for
  editing files and submitting jobs, NOT for running GPU workloads. Running
  `python scripts/02_run_rag.py` directly on a shared login node with a real
  model will either be killed by a process reaper or is against the cluster's
  acceptable-use policy.

Check which one you have:

```bash
sinfo          # Slurm cluster info -- if this exists, you're on a login node
squeue         # shows the job queue if Slurm is present
nvidia-smi     # is there a GPU attached to THIS machine right now?
```

If `sinfo`/`squeue` work, you're on a shared cluster and need to submit jobs
with `sbatch` rather than running `02_run_rag.py` directly — ask your
supervisor or the cluster docs for the GPU partition name and a template job
script, then wrap the commands in Step 5 in an `sbatch` script instead of
running them inline. If there's no Slurm and `nvidia-smi` shows a GPU, you're
on a dedicated box and everything below runs directly.

---

## 1. Open a terminal, check Python

The project requires **Python ≥3.11** (`pyproject.toml`). Check what's there:

```bash
python3 --version
```

If it's older, or if your account can't touch the system interpreter, use a
module system (common on university clusters) or a self-contained install:

```bash
module avail python          # if the cluster uses environment modules
module load python/3.11      # exact name varies by cluster

# if there's no module system, install one yourself, no root needed:
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.11
```

## 2. Get the repo onto the cluster

You need either an SSH key or a GitHub personal access token, generated on
this machine (keys don't transfer from your laptop for security reasons).

```bash
ssh-keygen -t ed25519 -C "adnan-cluster"
cat ~/.ssh/id_ed25519.pub
```

Copy that output to GitHub → Settings → SSH and GPG keys → New SSH key. Then:

```bash
cd ~                          # or wherever you want the checkout
git clone git@github.com:<your-username>/SAFE-RAG.git
cd SAFE-RAG
```

(If GitHub SSH is blocked by the cluster's firewall, use HTTPS with a
[personal access token](https://github.com/settings/tokens) instead:
`git clone https://<token>@github.com/<your-username>/SAFE-RAG.git`.)

## 3. Set up the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# core + dense retrieval/NLI + vLLM (Linux+CUDA, this is why you're here)
pip install -e '.[models,serve,dev]'
```

`vllm` is a large install (pulls in a matching CUDA-built torch). If it fails
on a version conflict, install torch first with the cluster's CUDA version,
then `pip install -e '.[models,serve,dev]'` again — the [PyTorch install
matrix](https://pytorch.org/get-started/locally/) will tell you the right
`--index-url` once you know `nvidia-smi`'s reported CUDA version.

## 4. Bring the data over

`data/` is gitignored on purpose (the pipeline warns loudly rather than commit
corpus text). Nothing under it comes with `git clone`. You need, at minimum:

- `data/raw/obliqa/ObliQA_test.json` — the question set
- `data/raw/adgm/` — the unzipped `StructuredRegulatoryDocuments.zip` (the full
  40-document ADGM corpus; **required**, the ablation is meaningless against
  the gold-passage-only fallback corpus — see the warning
  `scripts/_common.py` prints if this is missing)

From your laptop, over a *separate* terminal (not the AnyDesk session — AnyDesk
doesn't give you a normal file-transfer shell):

```bash
# adjust host/user to however you SSH into the same machine AnyDesk shows you
scp -r data/raw/obliqa youruser@cluster-host:~/SAFE-RAG/data/raw/
scp -r data/raw/adgm   youruser@cluster-host:~/SAFE-RAG/data/raw/
```

If direct SSH to the cluster isn't available from your laptop (some
university clusters only allow AnyDesk in, no outbound SSH), use AnyDesk's
built-in file transfer feature to drop a zip onto the remote desktop, then
unzip it into place with the terminal there.

## 5. Verify before running anything expensive

```bash
cd ~/SAFE-RAG
source .venv/bin/activate

python scripts/02_run_rag.py --preflight     # confirms GPU + recommends model size
pytest                                        # the 96+ tests should all pass here too
python scripts/02_run_rag.py --backend stub --limit 20 --no-dense --run-name smoke
python scripts/03_run_filters.py --stub-nli --run-name smoke
```

If all four of those are clean, the environment is sound and you're ready to
run the real thing. Delete `data/interim/smoke/` and `runs/smoke/` afterwards
(they're gitignored scratch, harmless to leave, but no reason to keep them).

## 6. Run the ID-format ablation

Long enough to survive an AnyDesk disconnect — use `tmux` or `screen` so the
run keeps going if the connection drops:

```bash
tmux new -s ablation
```

Then, per `configs/ablation.yaml` (see the comment block at the top of that
file for the full rationale):

```bash
# composite x 3B is already done (the registered pilot, 36.3% failure) --
# no need to rerun it.

python scripts/02_run_rag.py --config configs/ablation.yaml \
    --model Qwen/Qwen2.5-3B-Instruct --id-style ordinal \
    --run-name ablation_3b_ordinal --backend vllm
python scripts/03_run_filters.py --config configs/ablation.yaml \
    --run-name ablation_3b_ordinal

python scripts/02_run_rag.py --config configs/ablation.yaml \
    --model Qwen/Qwen2.5-7B-Instruct --id-style composite \
    --run-name ablation_7b_composite --backend vllm
python scripts/03_run_filters.py --config configs/ablation.yaml \
    --run-name ablation_7b_composite

python scripts/02_run_rag.py --config configs/ablation.yaml \
    --model Qwen/Qwen2.5-7B-Instruct --id-style ordinal \
    --run-name ablation_7b_ordinal --backend vllm
python scripts/03_run_filters.py --config configs/ablation.yaml \
    --run-name ablation_7b_ordinal
```

Detach with `Ctrl-b d` (tmux) and it keeps running if AnyDesk drops.
Reconnect with `tmux attach -t ablation`.

Each run writes to its own `data/interim/<run-name>/` and `runs/<run-name>/`
(see `scripts/_common.py`), so the three of these plus the existing pilot
result can't clobber each other. The citation-resolution numbers you want are
printed at the end of each `03_run_filters.py` call, and also land in the
`_provenance.extra.funnel.citation_resolution` header of each
`filtered.jsonl` for the comparison table (`docs/PROJECT_STATE.md`, "Then").

## Notes specific to this project

- `git status` before anything destructive — same rule as always,
  **never `git add -A`** (see `docs/PROJECT_STATE.md`, repo conventions).
- The tree will show as dirty on the cluster until you commit there or pull
  the local commits across; every artefact this pipeline writes warns loudly
  about a dirty tree in its provenance header, which is intentional, not a bug.
- `backend: vllm` is Linux+CUDA only by design (`generation/generator.py`) —
  this is the whole reason the ablation needs the cluster instead of your
  laptop's Colab/HF path.
