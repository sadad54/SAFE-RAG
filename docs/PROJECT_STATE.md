# Project state — read this first

**Last updated:** 2026-08-03
**Current plan: PATH B.** No further annotation. See "The decision" below.

A handoff document. If you are picking this project up cold — a new conversation,
a collaborator, or yourself in three weeks — read this, then `PREREGISTRATION.md`.

---

## Where things stand in one paragraph

The pilot ran end to end and produced a result. 2,000 ObliQA questions through a
plain RAG pipeline, 150 items annotated by the PI, 50 double-annotated. Cohen's
kappa came in at **0.3663**, below the registered 0.50 threshold. Rather than run
the registered revision round, the project has moved to **Path B**: the base rate
becomes a preliminary secondary result reported with its kappa, and the paper's
primary contributions become the automated funnel measurements — above all the
**citation-ID resolution failure**, which needs no human labels. The immediate
next task is a GPU-cluster ablation isolating the cause of that failure.

## The decision (2026-08-03)

Three options were weighed: (A) run the registered revision round, ~15 more hours
of annotation; (B) reframe around the automated findings, ~4 hours; (C) abandon.
**B was chosen** because the annotation burden was the binding constraint and the
automated findings stand on their own. Recorded as **Deviation 1** in
`PREREGISTRATION.md` §11 with the full diagnostic.

## Results so far

Generator Qwen2.5-3B-Instruct, NLI microsoft/deberta-large-mnli, seed 20260728.
Generation ran on Colab; retrieval and filtering are CPU-reproducible.

| Stage | Result |
|---|---|
| Retrieval recall@10 (BM25, 13,016-passage ADGM corpus) | 0.82 |
| Generated answers | 2,000 |
| S1 schema-valid | 1,881 (94.0%) |
| S2 faithfulness-passing | 750 (39.9% of S1) |
| candidate_recoverable / unrecoverable / control | 142 / 40 / 568 |
| stratum weights | .1893 / .0533 / .7573 |

**Citation resolution (the headline finding).** Of 3,343 cited ids: **63.6% exact,
28.0% recovered only by normalising trailing punctuation, 8.3% unresolvable, 0
ambiguous.** 36.3% of schema-valid answers contain at least one unresolvable id.
126 answers cite nothing resolvable.

**Deceptive grounding (preliminary, low reliability).**
`r = 5.9% [2.2%, 10.7%]`, unconditional `R = 2.2% [0.8%, 4.0%]`.
Per stratum: recoverable 13.8%, unrecoverable 5.0%, control 4.0%.
**kappa = 0.3663** on 50 double-annotated items. Observed agreement 0.80, expected
0.684, PABAK 0.60. Disagreements are directional: 7 items A(PI)/B(second), 2 the
reverse, 1 B/C. The second annotator's notes on all seven flips cite scope
mismatch. **5.9% is therefore likely an under-estimate.**

## Next task: the ID-format ablation

**Not yet built.** This is the immediate priority and the thing that makes Path B
a good paper rather than a thin one.

The 36.3% resolution failure cannot currently be attributed. Passage ids in the
prompt use the composite form `DocumentID::PassageID` (`19::100)`,
`13::4.13.3.Guidance.5.`), which models demonstrably struggle to reproduce — they
drop trailing punctuation, and sometimes emit a bare DocumentID. That format was
an implementation choice, not a finding about models.

**The experiment**, run on the university GPU cluster (access obtained 2026-08-03,
via AnyDesk), no annotation required:

| | composite ids | ordinal ids `[1]`…`[10]` |
|---|---|---|
| Qwen2.5-3B-Instruct | done — 36.3% failure | to run |
| Qwen2.5-7B-Instruct | to run | to run |

If failure collapses under ordinal ids, apparent citation hallucination in
structured-output RAG is largely an artefact of identifier design — a concrete,
actionable result. If it does not collapse, the 8.3% residual is genuine model
behaviour. Either outcome is publishable; that is what makes it worth running.

**What needs building first:**

1. `render_prompt` gains `id_style: composite | ordinal`. Ordinal mode labels
   passages `[1]`…`[10]` and keeps a per-item map back to real passage ids.
2. `resolve_citation_ids` handles ordinal citations.
3. Config knob under `generation:`; provenance records the style.
4. Tests for both paths. The prompt hash changes automatically, so the generation
   cache invalidates cleanly — no stale mixing.
5. Per-run output directories so the four runs do not overwrite each other.

## Then

1. Run all four configurations. vLLM on the cluster (`backend: vllm`) — roughly
   10× the Colab transformers path; the full 2,786 questions in minutes.
2. `scripts/03_run_filters.py` on each; build the comparison table.
3. Write. **ALTA 2026, deadline 11 September**, archival, ACL Anthology.

**Paper structure (short paper):**
1. Funnel evaluation of structured-output RAG on ObliQA
2. **Citation-ID resolution failure and its dependence on identifier format** ← headline
3. Preliminary deceptive-grounding estimate, kappa = 0.37, honestly caveated
4. Released pipeline and pre-registration

## Optional, cheap, high value

Adjudicating the 10 annotator disagreements with Riyad is ~45 minutes of
discussion, not annotation. It would yield three or four worked examples of what
deceptive grounding looks like in ADGM text — the qualitative material that makes
the results section readable. Not required for Path B.

## Decisions already made and why

- **Deviation 1** (§11): kappa 0.37 below threshold; revision round not run; base
  rate demoted to preliminary secondary, direction of bias reported.
- **Amendment 1** (§11): candidate pool split by whether a gold passage was
  retrieved. Made before generation, because recall@10 = 0.82 means ~16% of
  questions are unanswerable from context.
- **Correction 1** (§11): cited ids resolved by normalised match. Moved S2 from
  31.7% to 39.9%. Implementation fix, not a design change.
- **Corpus**: the full 40-document ADGM set, not the union of gold passages. An
  index containing only correct answers has no distractors.
- **Passage ids** are `DocumentID::PassageID` because ObliQA's PassageID is unique
  only within a document. This is exactly what the ablation now tests.

## Known limitations for the write-up

- **S2 requires every atomic claim to clear 0.5**, so longer answers are penalised
  geometrically. Registered, defensible, must be stated.
- **Claim decomposition is sentence-splitting** (`RuleDecomposer`). Under-splits
  compound legal sentences. Permissive, which is the safe direction.
- **S2 disproportionately removes the unrecoverable stratum** — 18% of questions
  lack gold in context but only 5.3% of survivors do.
- **The annotation reliability failure**, reported in full rather than smoothed.
- Generation ran on Colab, not the cluster. Provenance records it.

## Publication and scholarship strategy

Verified from primary sources unless marked.

- **UQ international scholarship round closes 19 October 2026**; EOI opened 20
  August; outcomes from 22 February 2027. UQ's rubric scores *"quality of the
  proposed advisory team"* — the supervisor's engagement is part of your score.
  Melbourne ~31 October (aggregator-sourced, verify).
- **ALTA 2026: 11 September deadline**, conference in Melbourne 30 Nov – 2 Dec.
- **Supervisor targets.** Guido Zuccon (UQ, ielab) is the closest topical match —
  2025 work on RAG hallucination detection and source attribution in RAG, noting
  existing approaches link only at document level. Damiano Spina and Falk Scholer
  (RMIT, ADM+S). Ehsan Shareghi and Reza Haffari (Monash). Aditya Joshi (UNSW).
- **Sequencing.** Contact supervisors in parallel with the work, not after
  acceptance. "First-author paper under review, here is the preprint and repo" is
  nearly as strong and arrives in time for the cycle.

## Novelty position

The original proposal claimed typed error routing had not been applied to RAG.
**That claim is false and was withdrawn.** Doctor-RAG (arXiv:2604.00865) and
Skill-RAG (arXiv:2604.15771) did it during 2026; arXiv:2606.29377 added a
budget-matched evaluation; the EACL 2026 RAG error taxonomy (arXiv:2510.13975)
supplies a diagnosis vocabulary; RefWalk (arXiv:2605.29742) couples schema and
attribution in regulatory QA. Both revised proposals are in `docs/proposals/`.
**All arXiv ids above came from search results and must be re-verified before
submission.**

## Standing constraint on annotation

**An LLM cannot supply the ground truth.** The finding is that automated methods
miss this failure; establishing it with an automated method is circular, and §4
defines S4 as human. An LLM may annotate *independently and in addition*, reported
openly. It may not stand in for the human pass, and two LLM passes are not two
annotators.

## Repo conventions

- `[REGISTERED]` in `configs/pilot.yaml` means fixed by the pre-registration.
  Changing one after generation needs a §11 entry.
- Every artefact carries a provenance header: git SHA, config hash, seed, models.
  Scripts warn on a dirty tree and print the resolved repo root.
- Data is gitignored. Annotation **labels** are committed; **batches** are not.
- `labels_adnan_batch_01RIP.jsonl` is a voided first pass, kept for transparency.
  It is not loaded by any script.
- `pytest` before anything that produces a number. 96 tests.
- **Do not use `git add -A` in this repo.** It has twice swept in files that
  should not be tracked.
