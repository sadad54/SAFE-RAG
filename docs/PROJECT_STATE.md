# Project state — read this first

**Last updated:** 2026-08-12
**Current plan: base rate reinstated as a primary result.** The revision round
succeeded — κ = 0.7860, "reliable" band. See "The revision round result
(2026-08-12)" below.

A handoff document. If you are picking this project up cold — a new conversation,
a collaborator, or yourself in three weeks — read this, then `PREREGISTRATION.md`.

---

## Where things stand in one paragraph

The pilot ran end to end and produced a result. 2,000 ObliQA questions through a
plain RAG pipeline, 150 items annotated by the PI. The first double-annotation
round (50 items) scored kappa 0.3663 — below threshold — which on 2026-08-03 led
to Path B (demote the base rate, lead with the automated funnel findings
instead). **Reversed 2026-08-10:** Riyad asked for another attempt at the B/C
boundary. The one registered revision round (Section 8) was run — guidelines
sharpened to v1.1, 30 fresh items double-annotated by both Adnan and Riyad in
parallel — and **cleared the threshold convincingly: κ = 0.7860, "reliable,"
27/30 exact agreement.** The base rate `r = 5.9% [2.2%, 10.7%]` is a primary
result again. The ID-format ablation is still fully built and still a good
secondary result; it's queued behind writing up the revision-round outcome and
one open scope decision (below).

## The revision round result (2026-08-12)

**κ = 0.7860** on the 30-item revision batch (observed agreement 0.9000, 27/30).
Clears not just the 0.50 "usable" bar but the 0.70 "reliable" bar. Full detail:
`runs/pilot_v1/revision_kappa.json`, recorded in `PREREGISTRATION.md` §11 as the
resolution of Deviation 2.

Three disagreements, no longer one-directional the way the original 9/50 were:
one NA/C (a bare-heading passage, genuinely ambiguous), one A/B (the PI missed a
scope mismatch the second annotator caught — same direction as the original
bias, but 1 case in 30 vs. 7 in 50), one B/C (both agreed it was wrongly
grounded, disagreed only on how obviously). Reduced, not eliminated, which is
the honest reading, not zero.

**Open decision — not yet made, needs your and Riyad's call, not just mine.**
The original 150 primary labels were made under guidelines v1.0; the revision
batch that just passed was labelled under v1.1. A passing κ under v1.1
establishes the construct is reliably distinguishable *under v1.1* — it doesn't
by construction certify the specific v1.0-era labels sitting in the primary
sample. The evidence is reassuring (PI's A/B error rate dropped from 14% to
~3.3% under the same guidelines fix) but 30 items is too few to bound that
tightly, and the primary sample is 150. Two live options:

1. **Report as-is, caveat it.** Cheapest. State plainly in limitations that the
   primary 150 predate the v1.1 clarification, cite the error-rate evidence
   above as reassurance, move on.
2. **Targeted audit.** Re-check just the PI's own **A**-labelled items in
   `candidate_recoverable` + `candidate_unrecoverable` (86 of the 150 — the
   strata deceptive grounding actually lives in; see `docs/PREREGISTRATION.md`
   §4 S3) against the v1.1 scope-clause test. Not blind re-annotation — you
   already know what you called it, you're checking whether the sharpened Step
   3 changes any of the 86 calls. Rough budget: 1–2 minutes each, well under the
   3–5 min/item cold-annotation rate, so 1.5–3 hours, once, by the PI alone
   (this isn't a kappa exercise, no second annotator needed).

Neither is registered — either is now a further §11 entry regardless of which
you pick. This is flagged rather than decided because it trades rigor against
the time budget you were explicit about, and that's your and Riyad's call.

## The 2026-08-10 decision — Path B reversed

Recorded as **Deviation 2** in `PREREGISTRATION.md` §11, which supersedes
Deviation 1 without deleting it (the pre-registration is append-only).

**What's happening:** `docs/ANNOTATION_GUIDELINES.md` moves to v1.1, fixing the
three things the Deviation-1 disagreement analysis actually found (incompleteness
read as B, no explicit scope-clause test, thin worked examples on that failure
mode). 30 fresh items — disjoint from the original 150 — get drawn and **both
annotators label all 30, independently, in parallel.** `scripts/04b_make_revision_batch.py`
builds the batch; `scripts/05b_compute_revision_kappa.py` recomputes kappa once
both label files exist. See "Next task" below for the exact commands.

**What does NOT get redone:** the original 150-item primary labels stand as-is if
kappa clears this time — Section 8 commits to re-annotating 30 fresh items, not
to re-labelling the primary sample. Caveat carried into the pre-registration: the
original 150 were labelled under v1.0 guidelines, so a passing kappa on the
revision batch establishes reliability under v1.1, not strictly under the wording
the primary labels were made with. Small gap, not zero. Worth a sentence in
limitations if the base rate goes back to being the headline.

## The 2026-08-03 decision (superseded)

Three options were weighed: (A) run the registered revision round, ~15 more hours
of annotation; (B) reframe around the automated findings, ~4 hours; (C) abandon.
**B was chosen** because the annotation burden was the binding constraint and the
automated findings stand on their own. Recorded as **Deviation 1** in
`PREREGISTRATION.md` §11 with the full diagnostic. **Reversed 2026-08-10 — see
above.**

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

**Deceptive grounding — primary result, revision round complete.**
`r = 5.9% [2.2%, 10.7%]`, unconditional `R = 2.2% [0.8%, 4.0%]`.
Per stratum: recoverable 13.8%, unrecoverable 5.0%, control 4.0%.
**Operative kappa = 0.7860** (30-item revision batch, 2026-08-12, "reliable"
band). The original round's kappa = 0.3663 (50 items, 2026-08-03) is retained as
history — the reason the revision round ran — not as the current reliability
figure. See "The revision round result" above for the open v1.0/v1.1 audit
decision.

**RQ-a (2026-08-12).** Operational definition declared first as Amendment 2
(`PREREGISTRATION.md` §11): cross-reference = >1 gold passage, single-reference
= 1. Result over the 150 primary labels: cross-reference **8.6% [3.0%, 22.4%]**
(n=35), single-reference **9.6% [5.4%, 16.3%]** (n=115). Intervals overlap
heavily — **no detectable difference by this cut**, and the cross-reference
group is small enough (n=35) that this doesn't rule much out either way. Report
honestly as a null secondary result, not a finding. `scripts/05c_rqa_breakdown.py`,
`runs/pilot_v1/rqa_breakdown.json`.

## Next task: commit labels, finalize the primary result, decide on the audit

**Revision-round annotation is done** — both `labels_adnan_batch_02.jsonl` and
`labels_riyad_batch_02.jsonl` exist and κ = 0.7860 has been computed. What's
left:

1. Commit and push both label files (they ARE meant to be committed — labels
   are the research output; only batches with corpus text are excluded):
   ```
   git add data/annotation/labels_adnan_batch_02.jsonl data/annotation/labels_riyad_batch_02.jsonl data/annotation/batch_02_key.jsonl
   git commit -m "Revision round: 30-item double annotation, kappa 0.786"
   git push
   ```
2. Run `python scripts/05_compute_results.py --annotator adnan --second riyad`
   to regenerate `runs/pilot_v1/results.json` with the operative kappa folded
   in (the script now prefers `revision_kappa.json` when present — see the
   fix below, already applied).
3. **Decide on the audit** (the open decision above) with Riyad — 45 minutes of
   discussion, not more, to pick option 1 or 2 and record whichever in
   `PREREGISTRATION.md` §11.
4. Then: write up the revision round in the paper (structure below), and only
   after that circle back to the ID-format ablation.

## Then: the ID-format ablation (secondary, already built)

Not the current priority, but fully built and ready whenever there's cluster
time — see `docs/CLUSTER_SETUP.md`. Kept here so it isn't lost track of.

The 36.3% resolution failure cannot currently be attributed. Passage ids in the
prompt use the composite form `DocumentID::PassageID` (`19::100)`,
`13::4.13.3.Guidance.5.`), which models demonstrably struggle to reproduce — they
drop trailing punctuation, and sometimes emit a bare DocumentID. That format was
an implementation choice, not a finding about models.

**The experiment**, run on the lab desktop (16GB Turing card; the university
cluster access obtained 2026-08-03 was never needed in the end -- see
`docs/CLUSTER_SETUP.md`), no annotation required:

| | composite ids | ordinal ids `[1]`…`[10]` |
|---|---|---|
| Qwen2.5-3B-Instruct | done — 36.3% failure | done 2026-08-12 |
| Qwen2.5-7B-Instruct-AWQ | done 2026-08-13 | running 2026-08-13 |

**2026-08-13 incident, and the resulting model-id change for the 7B row.**
An attempt to run the 7B cells locally on the lab desktop used vLLM's
`cpu_offload_gb` to make the fp16 weights fit, and took the whole machine
down mid-run — a hard power loss, not a CUDA OOM (kernel log shows nothing:
no OOM-killer, no thermal event, the log just stops). Offloading keeps CPU
and GPU both under sustained heavy load streaming weights over PCIe every
forward pass; the combined draw is the likely trigger on this desktop's PSU.
The 7B row now runs on `Qwen/Qwen2.5-7B-Instruct-AWQ` (Qwen's own 4-bit
checkpoint) instead of the fp16 identifier — no offload needed, ~5GiB of
weights. Same base weights and instruction tune as the 3B row's family, just
quantized; not a change to `PREREGISTRATION.md` (generation model precision
was never registered, only that one open-weight model is used and its
identifier recorded — see §5), but the identifier differs from what's named
above and in `configs/ablation.yaml`'s original header, so record
`Qwen2.5-7B-Instruct-AWQ` explicitly in the write-up rather than shortening it
to "Qwen2.5-7B-Instruct". Details: `src/saferag/generation/generator.py`'s
`VLLMGenerator` docstring.

The crash also surfaced a second, independent bug: `02_run_rag.py`'s resume
cache (`data/interim/<run_name>/_generations.jsonl`) invalidates entries by
prompt hash only, not by which model produced them. The pre-incident fp16
attempt left a cache under `ablation_7b_composite` that the AWQ rerun would
have silently resumed from — reusing fp16-generated answers inside what's
supposed to be a pure-AWQ cell — had the file not also been corrupted
mid-write by the power loss, which is what actually surfaced it (crashed on
the corrupted line instead of silently mixing models). Worked around by
moving the stale file aside (`_generations.jsonl.pre-incident-fp16-corrupt.bak`)
before rerunning. Not fixed at the code level yet: the cache key should
include the model identifier. Low urgency day-to-day (`run_name` is
conventionally one model per cell) but worth doing before the next time a
model gets swapped under an existing `run_name`.

If failure collapses under ordinal ids, apparent citation hallucination in
structured-output RAG is largely an artefact of identifier design — a concrete,
actionable result. If it does not collapse, the 8.3% residual is genuine model
behaviour. Either outcome is publishable; that is what makes it worth running.

**What needs building first — done 2026-08-10:**

1. ~~`render_prompt` gains `id_style: composite | ordinal`.~~ Done. Ordinal mode
   labels passages `[1]`…`[10]`; `ordinal_id_map` (same args) recovers the
   per-item map back to real passage ids.
2. ~~`resolve_citation_ids` handles ordinal citations.~~ Done — takes an
   optional `id_map`, tolerates `[1]`/`(1)`/`1.` decoration the way it already
   tolerated composite punctuation. No-op (identical to before) when `id_map`
   is empty/omitted.
3. ~~Config knob under `generation:`~~. Done — `generation.id_style`, overridable
   with `--id-style`; `configs/ablation.yaml` added. Provenance records
   `id_style` and `run_name`.
4. ~~Tests for both paths.~~ Done — 12 new tests in `tests/test_pipeline.py`,
   all 107 tests pass. Verified end-to-end with the stub backend that ordinal
   citations resolve through the whole 02→03 pipeline.
5. ~~Per-run output directories~~. Done — `interim/` and `runs/` nest under
   `run_name` (`scripts/_common.py paths()`); existing pilot_v1 artefacts
   migrated into `data/interim/pilot_v1/` and `runs/pilot_v1/` (gitignored,
   no history impact).

**Status 2026-08-13.** Three of four cells generated (3B×composite, 3B×ordinal,
7B×composite); 7B×ordinal generating now on the lab desktop, see the incident
note above. The cluster was never needed in the end.

**Once the ablation has run:**

1. Run all four ablation configurations. Done or in progress -- see the table
   above.
2. `scripts/03_run_filters.py` on each; build the comparison table. Done for
   3B×ordinal already; still needed for both 7B cells once generated. Run
   sequentially, not alongside a `02_run_rag.py` generation job -- S2's
   `LLMDecomposer` loads its own generator onto the same GPU.
3. Write. **ALTA 2026, deadline 11 September**, archival, ACL Anthology.

**Paper structure (short paper) — settled 2026-08-12, kappa cleared:**

1. Funnel evaluation of structured-output RAG on ObliQA
2. Deceptive-grounding base rate (`r = 5.9%`), κ = 0.7860 after the one
   permitted revision round, both rounds narrated honestly (0.37 → guideline
   fix → 0.79, not just the final number)
3. **Citation-ID resolution failure and its dependence on identifier format**
   (from the ablation, once run)
4. Released pipeline and pre-registration

## Optional, cheap, high value

Adjudicating the original 10 annotator disagreements (from the 50-item double
batch) with Riyad is still ~45 minutes of discussion, not annotation, and still
yields three or four worked examples for the results section. It does not
substitute for the revision round above — the pre-registration commits to fresh
items, not re-litigating old ones — but it's good qualitative material either
way and can happen any time, before or after the revision batch.

## Decisions already made and why

- **Deviation 2** (§11): Deviation 1 reversed, 2026-08-10; resolved 2026-08-12.
  Guidelines to v1.1, 30 fresh items double-annotated by both annotators, κ =
  0.7860 ("reliable"). Base rate reinstated as primary. See "Next task" above.
- **Deviation 1** (§11, superseded by Deviation 2): kappa 0.37 below threshold;
  revision round not run; base rate demoted to preliminary secondary, direction
  of bias reported.
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
- **The annotation reliability history**, reported in full: κ = 0.3663 (original
  50), revision round run, κ = 0.7860 (revision 30). Not smoothed into a single
  final number.
- **v1.0/v1.1 label consistency**, decided 2026-08-12: the primary 150 stand as
  labelled under guidelines v1.0; the passing κ = 0.7860 was measured under
  v1.1. Reported as a limitation rather than closed by audit — see
  `PREREGISTRATION.md` §11 for the reasoning and the audit's cost if a reviewer
  asks for it.
- Generation ran on Colab, not the cluster. Provenance records it.

## Publication and scholarship strategy

Verified from primary sources unless marked.

- **UQ international scholarship round closes 19 October 2026**; EOI opened 20
  August; outcomes from 22 February 2027. UQ's rubric scores *"quality of the
  proposed advisory team"* — the supervisor's engagement is part of your score.
  Melbourne ~31 October (aggregator-sourced, verify).
- **ALTA 2026 — verified 2026-08-12 from alta2026.alta.asn.au directly** (not
  aggregator-sourced): archival submission deadline **Friday 11 September 2026,
  11:59pm Anywhere-on-Earth**. Short papers: **4 pages + unlimited
  references/appendices** (appendices not guaranteed to be reviewed), two-column
  ACL format, LaTeX or Word template from `github.com/acl-org/acl-style-files`.
  **Double-blind — the submission must be anonymised**: no author identity, no
  personal-site URLs, and the GitHub repo must be anonymised too if linked (do
  not link `github.com/sadad54/SAFE-RAG` directly in the anonymised draft —
  either omit the link and say "code released on acceptance," or use an
  anonymous mirror). Preprints ARE allowed to be public during review (post-2024
  ARR policy, no anonymity period), so posting to arXiv is fine even while the
  ALTA copy stays anonymous. Submission via OpenReview
  (`openreview.net/group?id=ALTA.asn.au/2026/Archival`) — **create the
  OpenReview profile now with an MJIIT/UTM institutional email**; profiles made
  with a non-institutional email face up to a two-week moderation delay, which
  would eat a meaningful chunk of the runway to 11 September. ALTA is CORE 2026
  rank **Australasian C** — a real, respectable regional venue, appropriately
  scoped for a rigorous pilot study rather than requiring a flagship-conference
  level result.
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

**Re-verified 2026-08-12 — all five confirmed real, independently of the
original search that surfaced them.** Cross-checked via a second search plus
direct arXiv fetch: Doctor-RAG (HIT + Macquarie + UNSW + CSIRO Data61, trajectory-
level failure diagnosis + tool-conditioned repair), Skill-RAG (hidden-state
probing + skill routing, four repair skills), arXiv:2606.29377 (budget-
constrained diagnose-and-repair), the EACL 2026 taxonomy paper
(`github.com/layer6ai-labs/rag-error-classification`, real and well-formed),
and RefWalk (Korea University, `RegOps-Bench`, cross-document citation
traversal + per-rule attribution). None are placeholders or misremembered ids.
The withdrawal of the original novelty claim stands — these are genuinely
close prior art and the paper needs to position against them, not merely cite
them — but the citations themselves are now safe to use.

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
