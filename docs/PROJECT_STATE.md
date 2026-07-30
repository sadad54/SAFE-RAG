# Project state — read this first

**Last updated:** 2026-07-30

A handoff document. If you are picking this project up cold — a new conversation,
a collaborator, or yourself in three weeks — read this, then the pre-registration.

---

## Where things stand in one paragraph

The deceptive-grounding pilot has run end to end. 2,000 ObliQA questions through a
plain RAG pipeline, filtered, and a 150-item annotation batch drawn to the
registered allocation with zero shortfall. **The only outstanding step is human
annotation.** Nothing else blocks the result.

## The one-line research question

How often does an answer that is schema-valid and passes an entailment-based
faithfulness check nonetheless cite a passage governing a *different* entity,
clause, or category than the question asked about?

That failure is called **deceptive grounding**. It has been demonstrated in
clinical RAG and never measured in a regulatory corpus. SAFE-RAG devotes a whole
repair branch to it, so the rate determines whether that branch is motivated.

## Pipeline results as of 2026-07-30

Generator Qwen2.5-3B-Instruct, NLI microsoft/deberta-large-mnli, seed 20260728.

| Stage | Result |
|---|---|
| Retrieval recall@10 (BM25, 13,016-passage ADGM corpus) | 0.82 |
| Generated answers | 2,000 |
| S1 schema-valid | 1,881 (94.0%) |
| S2 faithfulness-passing | 750 (39.9% of S1) |
| candidate_recoverable / unrecoverable / control | 142 / 40 / 568 |
| stratum weights | .1893 / .0533 / .7573 |
| Annotation batch drawn | 80 / 20 / 50 = 150, no shortfall |

Citation resolution: 63.6% exact, 28.0% recovered by normalising trailing
punctuation, 8.3% unresolvable, 0 ambiguous. 126 answers cite nothing resolvable.

**Three findings already exist independent of annotation:** 94% schema validity for
a 3B model; 60% of schema-valid answers failing entailment; a 6.7% answer-level
citation-resolution failure.

## Next actions, in order

1. **Annotate 150 items** by hand. `docs/ANNOTATION_GUIDELINES.md`, then
   `python -m saferag.pilot.annotate --batch data/annotation/batch_01.jsonl --annotator NAME`.
   3–5 min/item, blocks of 25. Progress saves per keystroke.
2. **Second annotator on the 50-item double batch.** Independent, guidelines only,
   no discussion until both are done. Dr Siti is the natural ask.
   **Without kappa the finding is not publishable.** This is the main open risk.
3. `scripts/05_compute_results.py --annotator X --second Y` → base rate, CI, kappa,
   and the pre-registered verdict.
4. `scripts/06_random_routing_check.py` — typed vs random routing at matched budget.
   Currently simulation-only; real repair execution is unimplemented. Run before
   building any of SAFE-RAG proper.

## Standing constraint on annotation

**An LLM cannot supply the ground truth.** The finding is that automated methods
miss this failure; establishing it with an automated method is circular, and
Section 4 defines S4 as human. An LLM may annotate *independently and in addition*,
reported openly as a secondary result answering RQ-b — that is useful, because high
agreement would justify LLM-assisted scaling for the main study. It may not stand
in for the human pass, and two LLM passes are not two annotators.

## Decisions already made and why

- **Amendment 1** (pre-registration §11): candidate pool split by whether a gold
  passage was retrieved. Made before generation, because recall@10 = 0.82 means
  ~16% of questions are unanswerable from context and would have burned a third of
  the annotation budget confirming what the logs already said.
- **Correction 1** (§11): cited passage ids resolved by normalised match. Models
  drop the trailing `)` / `.` in ObliQA ids. Implementation fix, not a design
  change; estimand untouched. Moved S2 from 31.7% to 39.9%.
- **Corpus**: the full 40-document ADGM set, not the union of gold passages. An
  index containing only correct answers has no distractors and makes the whole
  measurement meaningless.
- **Passage ids** are `DocumentID::PassageID` because ObliQA's PassageID is unique
  only within a document.

## Known limitations to carry into the write-up

- **The composite id format is the wrong design.** A main study should label
  passages `[1]`…`[10]` in the prompt and map back internally. Most of the 36%
  resolution-failure rate is attributable to the format, so the residual must not
  be reported as pure model hallucination.
- **S2 requires every atomic claim to clear 0.5**, so longer answers are penalised
  geometrically. Registered, defensible, must be stated.
- **Claim decomposition is sentence-splitting** (`RuleDecomposer`). Under-splits
  compound legal sentences. Permissive, which is the safe direction.
- **S2 disproportionately removes the unrecoverable stratum** — 18% of questions
  lack gold in context but only 5.3% of survivors do. Worth a sentence.
- Generation ran on Colab, not the cluster. Provenance records it; note where.

## Publication and scholarship strategy

Verified from primary sources unless marked.

- **UQ international scholarship round closes 19 October 2026**; EOI opens 20
  August; outcomes from 22 February 2027. UQ's rubric scores *"quality of the
  proposed advisory team"* — the supervisor's engagement is scored as part of the
  application. Melbourne ~31 October (aggregator-sourced, verify).
- **ALTA 2026: 11 September deadline**, archival, ACL Anthology, conference in
  Melbourne 30 Nov – 2 Dec. Puts you in front of the right people before UQ
  outcomes. FinNLP @ EMNLP was 11 August. RegNLP has no 2026 edition found.
- **Supervisor targets.** Guido Zuccon (UQ, leads ielab) is the closest topical
  match in Australia — 2025 work on RAG hallucination detection and *source
  attribution in RAG*, noting existing approaches link only at document level.
  Damiano Spina and Falk Scholer (RMIT, ADM+S) for the evaluation framing. Also
  Ehsan Shareghi and Reza Haffari (Monash), Aditya Joshi (UNSW).
- **Sequencing.** Contact supervisors in parallel with the work, not after
  acceptance. "First-author paper under review, here is the preprint and repo" is
  nearly as strong as acceptance and arrives in time for the cycle.

## Novelty position

The original proposal claimed typed error routing had not been applied to RAG.
**That claim is false and was withdrawn.** Doctor-RAG (arXiv:2604.00865) and
Skill-RAG (arXiv:2604.15771) did it during 2026; arXiv:2606.29377 added a
budget-matched evaluation; the EACL 2026 RAG error taxonomy (arXiv:2510.13975)
supplies a diagnosis vocabulary; RefWalk (arXiv:2605.29742) couples schema and
attribution in regulatory QA.

**What survives:** every existing RAG diagnosis vocabulary is trajectory-oriented —
retrieval insufficiency, reasoning error, malformed tool call. None diagnoses
schema validity against a target schema, and none does entity-level attribution
checking. A deceptive-grounding failure is by construction a trajectory where
retrieval succeeded, reasoning was sound, and every claim is entailed, so it is
invisible to a sufficiency-based diagnosis model for the same reason it is
invisible to a faithfulness metric.

Both revised proposals are in `docs/proposals/`. All arXiv ids above came from
search results and **should be re-verified against arXiv before submission.**

## Repo conventions

- `[REGISTERED]` in `configs/pilot.yaml` means fixed by the pre-registration.
  Changing one after generation needs a §11 entry.
- Every artefact carries a provenance header: git SHA, config hash, seed, models.
  Scripts warn on a dirty tree.
- Data is gitignored. Annotation **labels** are committed; **batches** are not
  (derived, and they embed ADGM text).
- `pytest` before anything that produces a number. 96 tests.
- Do not use `git add -A` in this repo. It has twice swept in files that should
  not be tracked — the vendored ObliQA dataset and the annotation batches.
