# Pre-Registration — Deceptive Grounding Base Rate in Regulatory RAG

**Study title.** How often does deceptive grounding occur in retrieval-augmented question answering over financial regulation?

**Principal investigator.** Adnan Mashrur Sadad, Malaysia-Japan International Institute of Technology (MJIIT), Universiti Teknologi Malaysia.

**Supervisor.** Dr Siti Nur Khadijah Aishah Ibrahim.

**Version.** 1.0
**Date registered.** 2026-07-28
**Status.** Registered before any data was generated, any model was run, and any item was inspected.

> **How to use this file.** Commit it before running `scripts/02_run_rag.py` for the first time. Do not edit Sections 2–9 after that point. All changes go in Section 11 (Deviations), appended with a date and a reason. The value of this document comes entirely from the fact that it was fixed in advance; editing it silently destroys that value.

---

## 1. Motivation

The SAFE-RAG framework proposes to route retrieval-augmented generation failures to different corrective actions depending on failure type. One of its two diagnosed conditions is *evidence misattribution*: an answer that is schema-valid, is fully entailed by the passage it cites, and is nonetheless wrong because the cited passage governs a different entity, clause, or category than the question asked about. Prior work terms this **deceptive grounding** and demonstrates it in clinical retrieval-augmented generation.

No published measurement of this phenomenon exists in the financial or regulatory domain. The SAFE-RAG framework devotes an entire branch — entity-constrained query reformulation and re-retrieval — to repairing it. If the phenomenon is rare in regulatory corpora, that branch addresses a problem that does not occur at a rate justifying the engineering, and the framework should be rescoped.

This study measures the rate. It is a measurement study. It proposes no system and evaluates no method.

## 2. Research question

**RQ.** In retrieval-augmented question answering over the ObliQA regulatory corpus, what proportion of answers that are schema-valid and pass a standard entailment-based faithfulness check are nonetheless grounded in a passage governing a different regulated entity, clause, or category than the question concerns?

Two secondary questions, answered from the same data at no additional cost:

**RQ-a.** Does the rate differ by question type — in particular, are cross-reference and multi-obligation questions affected more than single-obligation questions?

**RQ-b.** How well does an automated entailment model agree with human judgement on these items? (This bounds the reliability of any diagnosis signal built on it.)

## 3. Primary outcome measure

The **deceptive grounding base rate**: the estimated proportion of all generated answers that satisfy conditions S1, S2 and S4 defined in Section 4.

Reported as a point estimate with a 95% interval. The estimator and interval method are fixed in Section 7.

## 4. Operational definitions

An item is a **deceptive grounding instance** if and only if all four conditions hold.

### S1 — Schema-valid *(automatic)*

The generated output parses against the declared target schema without error. The schema is fixed in advance as:

```json
{
  "answer": "string",
  "cited_passage_ids": ["string", "..."],
  "obligations": ["string", "..."]
}
```

`cited_passage_ids` must be non-empty. An output that parses but cites nothing fails S1.

The schema is deliberately minimal. It is not the schema SAFE-RAG will eventually use; it is the smallest schema on which the phenomenon can be observed.

### S2 — Faithfulness-passing *(automatic)*

The `answer` field is decomposed into atomic claims. Every atomic claim is scored for entailment against the concatenated text of the passages named in `cited_passage_ids`. The item passes S2 if:

- every atomic claim has entailment probability ≥ **0.5**, and
- no atomic claim has contradiction probability ≥ **0.5**.

Both thresholds are fixed here and will not be tuned. If they are later found to be poorly calibrated, that finding is reported as a result and the original thresholds are retained for the headline number.

### S3 — Attribution mismatch *(automatic screen, not a criterion)*

*Amended 2026-07-28, before any generation was run. See Section 11, Amendment 1.*

The set `cited_passage_ids` is compared against the ObliQA gold relevant-passage set for that question, and against the set of passages the retriever actually placed in the model's context. Items are partitioned into three strata:

| Stratum | Condition |
|---|---|
| **control** | Jaccard overlap between cited and gold sets exceeds 0.0 — the answer cites at least one gold passage. |
| **candidate_recoverable** | No overlap, **but a gold passage was present in the retrieved context.** The model had the correct passage available and cited a neighbour instead. |
| **candidate_unrecoverable** | No overlap, **and no gold passage was retrieved at all.** The model could not have cited correctly. |

**S3 is a sampling device, not part of the definition of the phenomenon.** The rate of S3 alone is a retrieval-error statistic and will not be reported as the headline result. Reporting it as such would be measuring ordinary retrieval failure and renaming it.

### S4 — Human confirmation *(manual, decisive)*

A human annotator, following `docs/ANNOTATION_GUIDELINES.md`, assigns one of:

- **A** — the cited passage governs the entity, clause, or category the question asks about.
- **B** — the cited passage is topically adjacent but governs a *different* entity, clause, section, or category. **This is deceptive grounding.**
- **C** — the cited passage is off-topic or irrelevant; a non-expert reader would identify it as wrong. This is ordinary retrieval failure.
- **NA** — the item cannot be judged (malformed question, corpus defect, gold label appears wrong). Excluded from the denominator and reported as a count.

An item is a deceptive grounding instance iff S1 ∧ S2 ∧ (S4 = B).

The distinction between **B** and **C** carries the entire study. If annotators cannot make it reliably, the construct is not well defined and the finding does not exist. Section 8 specifies how this is tested.

## 5. Materials

- **Corpus and questions.** ObliQA (RIRAG, arXiv:2409.05677), derived from Abu Dhabi Global Markets regulatory text. The public test split is used. Gold relevant-passage annotations from the dataset are used for the S3 screen only.
- **Retrieval.** BM25 (lexical) fused with one off-the-shelf dense retriever. Top-10 passages passed to the generator. No retriever is fine-tuned.
- **Generation.** One open-weight instruction-tuned model, served locally, temperature 0, fixed seed. The exact model identifier and revision hash are recorded in the run manifest.
- **Entailment.** A dedicated natural language inference model, **from a different model family than the generator**. This separation is required, not optional: using the same family to generate and to judge produces a self-referential measurement.

All model identifiers, revisions, prompts, sampling parameters, seeds and the git commit SHA are written into every output file by `saferag.utils.provenance`.

## 6. Sampling plan

1. Run the pipeline over **2,000** ObliQA test questions. If the test split contains fewer than 2,000, use all of it and record the actual number.
2. Apply S1 and S2 automatically. Record the pass rate at each stage; these are reported as descriptive statistics.
3. Partition the S1 ∧ S2 survivors by S3 into the three strata defined above, and record each stratum's share `w_s` of the full survivor set.
4. Draw a stratified sample for annotation, using the fixed seed **20260728**:
   - **80** items from **candidate_recoverable**
   - **20** items from **candidate_unrecoverable**
   - **50** items from **control**

   all sampled uniformly at random without replacement. Total 150, unchanged.

The 50 control items are **not optional**. Without them the study estimates a precision conditional on S3 and cannot estimate a base rate, because deceptive grounding can occur in items whose cited set overlaps the gold set (for example, one correct and one wrong-entity citation).

The 20 candidate_unrecoverable items are likewise **not optional**, even though these items are expected to be almost entirely label C. They are what converts "these are retrieval failures by construction" from an assumption into a measurement. Assuming that stratum's B-rate is zero without checking would be exactly the kind of unverified analytic shortcut this document exists to prevent.

If any stratum contains fewer items than its allocation, take the whole stratum and record the shortfall.

## 7. Analysis plan

Let:

- `p1` = proportion of generated answers passing S1
- `p2` = proportion of S1-passers also passing S2
- `w_s` = proportion of S1 ∧ S2 survivors falling in stratum `s`, measured on the **full** survivor set, not on the annotated sample
- `b_s` = proportion labelled **B** among annotated items in stratum `s`

The **conditional rate** (among schema-valid, faithfulness-passing answers) is:

```
r = Σ_s  w_s · b_s        over s ∈ {candidate_recoverable, candidate_unrecoverable, control}
```

The **unconditional base rate** (among all generated answers) is:

```
R = p1 · p2 · r
```

The estimand is unchanged by Amendment 1. Splitting the candidate pool in two adds a term to the sum; it does not alter what `r` measures.

**Headline number is `r`**, the conditional rate, because it is the quantity SAFE-RAG's attribution branch acts on. `R` is reported alongside it.

**Interval estimation.** 95% interval on `r` by non-parametric bootstrap over the annotated items, stratified, 10,000 resamples, percentile method. Wilson score intervals are reported for each `b_s` individually. The bootstrap is used for `r` because it is a function of two independent binomials plus an estimated weight, for which no clean closed form exists.

NA-labelled items are removed from both numerator and denominator of their stratum, and their count is reported.

**No significance test is planned**, because no comparison between conditions is being made. This is an estimation study.

## 8. Inter-annotator agreement

**50 of the 150 annotated items** are independently labelled by a second annotator who has read only `docs/ANNOTATION_GUIDELINES.md` and has not seen the first annotator's labels. The 50 are drawn with seed 20260728 and span both pools proportionally.

**Cohen's κ** is computed over the three substantive labels (A/B/C), with NA items excluded pairwise. κ is reported in the paper regardless of its value.

Pre-committed interpretation:

| κ | Interpretation | Action |
|---|---|---|
| ≥ 0.70 | The construct is reliably distinguishable. | Proceed. Report κ. |
| 0.50 – 0.69 | Usable but imperfect. | Proceed, report κ prominently, and include a qualitative analysis of the disagreement cases in the paper. |
| < 0.50 | **The B/C boundary is not well defined.** | Stop. Examine all disagreements, sharpen the guidelines (expected sharpening: an explicit test on whether the passage's scope or application clause names a different entity type), re-annotate 30 fresh items, recompute κ. **At most one such revision round is permitted.** If κ remains below 0.50 after one revision, the natural-occurrence measurement is abandoned and the study moves to Pivot A (Section 10). |

The one-revision limit is pre-committed specifically to prevent guideline tuning until the desired agreement appears.

## 9. Decision rule

Fixed before any data is seen. `r` is the conditional rate from Section 7.

| Measured `r` | Pre-committed action |
|---|---|
| **≥ 0.10** | Primary finding stands. Report as headline. SAFE-RAG's attribution branch proceeds as designed. |
| **0.05 – 0.099** | Finding is real but thin. Report the aggregate **and** the RQ-a breakdown by question type. The paper's claim becomes the conditional structure of the rate rather than its magnitude. |
| **< 0.05** | The phenomenon is too rare on ObliQA to motivate the branch on frequency grounds. Execute the pivot sequence in Section 10. |
| **> 0.40** | **Treat as a suspected defect, not a finding.** Hand-inspect 20 items immediately. A rate this high most likely indicates a mis-specified entailment threshold, a broken gold-passage join, or a passage-ID mismatch. Do not report until the pipeline has been audited. |

## 10. Pre-specified pivots

Written in advance so that a low result is a branch in the plan rather than a project failure. Attempt in order; stop at the first that succeeds.

**Pivot 0 — check the other corpus first.** Before concluding the phenomenon is rare, run a 200-item version of the same pipeline on FinDER. Financial filings contain far more entity confusability than a single regulator's rulebook: the same line item across fiscal years, the same metric across subsidiaries and reporting segments. Cost: approximately two days. If the rate on FinDER clears 0.10, the study is reframed as a cross-domain comparison, which is a stronger paper than either corpus alone.

**Pivot A — construct the phenomenon instead of observing it.** Build an entity-confusable diagnostic set: take questions with known gold passages and perturb the governed entity, category, or licence class while holding surface wording nearly constant. Measure how often the system grounds in the unperturbed passage. This converts the contribution from a base-rate observation into a **diagnostic benchmark** on which existing faithfulness metrics can be shown to be blind by construction. Labels are correct by construction, so κ ceases to be the bottleneck.

**Pivot B — reframe as metric blindness.** Even at a low rate, if standard faithfulness scoring assigns passing scores to items human annotators judge wrongly grounded, that is an evaluation finding in its own right: the metric cannot see the failure *in principle*, not merely rarely.

**Pivot C — change domain.** Move the primary evaluation to FinDER or FinanceBench-derived filings and retain ObliQA as the low-rate contrast case.

## 11. Deviations from pre-registration

Any departure from Sections 2–10 is recorded here with date, description, and reason. This section is append-only.

### Amendment 1 — split the candidate pool by retrievability

**Date.** 2026-07-28
**Sections affected.** 4 (S3), 6 (sampling), 7 (analysis)
**Status when made.** Before `scripts/02_run_rag.py` had been run. No generated answer, no annotation label, and no outcome data of any kind existed. Retrieval had been run; the amendment is motivated by retrieval statistics only.

**What changed.** The single candidate pool is split into `candidate_recoverable` (no cited passage in the gold set, but a gold passage *was* in the retrieved context) and `candidate_unrecoverable` (no gold passage retrieved at all). Annotation allocation changes from 100 candidate / 50 control to 80 recoverable / 20 unrecoverable / 50 control. Total annotation burden is unchanged at 150 items.

**Why.** A retrieval check on the ObliQA test split (2,786 questions, 13,016-passage ADGM corpus, BM25 top-10) measured recall@10 at 0.82–0.84. Roughly one question in six therefore has no gold passage anywhere in the model's context. For those items the model *cannot* cite correctly; whatever it cites necessarily lands in the candidate pool and is a retrieval failure — label C — by construction. Under the original design an estimated third of the 100 candidate annotations would have been spent re-confirming a fact already visible in the retrieval logs, leaving the quantity of interest underpowered.

Deceptive grounding, as defined in Section 1, requires that the correct evidence was *available* and the model grounded in something adjacent instead. That is precisely the `candidate_recoverable` stratum. Concentrating annotation there measures the phenomenon the study is about.

**Effect on the estimand.** None. `r` is still the proportion of schema-valid, faithfulness-passing answers that are deceptively grounded. The estimator gains a third term in an already-stratified weighted sum.

**Effect on power.** Expected to increase substantially for the quantity of interest, because annotation moves from a stratum where the outcome is near-certain to one where it is uncertain.

**Who decided.** Proposed on the basis of the retrieval statistics above and approved by the PI before generation was run.

### Deviation 1 — annotation halted at kappa 0.37; base rate demoted to secondary

**Date.** 2026-08-03
**Sections affected.** 8 (inter-annotator agreement), 9 (decision rule)
**Status when made.** After both annotation passes were complete and kappa computed.

**What happened.** 150 items annotated by the PI; 50 double-annotated by a second
annotator. Cohen's kappa = 0.3663, below the 0.50 threshold in Section 8. The
registered response is to stop, revise the guidelines once, and re-annotate 30
fresh items.

**What was decided instead.** The revision round is NOT being run. The base rate
is therefore demoted from primary outcome to a **preliminary, low-reliability
secondary result**, reported with kappa stated in the abstract and the failed
threshold narrated in full. The paper's primary contributions become the
automated funnel measurements, which do not depend on human labels.

**Why this is a deviation and is reported as one.** The registered rule was not
followed. Stating that plainly is the alternative to quietly reporting kappa
without its consequence.

**Diagnostic detail, reported in the paper.** Observed agreement 40/50 = 0.80;
expected agreement 0.684; PABAK 0.60. The low kappa is partly the skewed-marginal
artefact, but not only that: disagreements are systematic and directional. Seven
items were A for the PI and B for the second annotator, two the reverse, one B/C.
The second annotator's notes on all seven A-to-B flips cite scope mismatch
(different regulated entity, licence category, or time period), which is the B
definition. Two of the PI's B labels appear to penalise incompleteness, which
Section 7 of the guidelines assigns to A.

**Direction of the bias.** The second annotator flagged B on 11 of 50 items
against the PI's 7. If the stricter reading is correct, the reported 5.9% is an
UNDER-estimate. The paper states this rather than presenting 5.9% as unbiased.

**What would resolve it.** One guideline revision sharpening (a) incompleteness is
A, (b) an explicit scope test against the passage's application clause, and (c)
the general-provision-covering-a-specific-case rule; then 30 fresh double-annotated
items. Left as future work.

### Correction 1 — resolve cited passage ids by normalised match

**Date.** 2026-07-30
**Sections affected.** None. This is an implementation correction, not a design change.
**Status when made.** After generation, before any annotation. No label existed.

**What was wrong.** S2 and S3 both depend on looking up the passages named in
`cited_passage_ids`. That lookup was an exact string match against the ids offered
in the prompt. ObliQA PassageIDs carry trailing punctuation (`19::100)`,
`13::4.13.3.Guidance.5.`) which models routinely drop when copying, so citations
that named the correct passage failed to resolve.

**Measured impact on the first run.** Of 3,343 cited ids: 2,127 (63.6%) matched
exactly, 937 (28.0%) matched only after normalising trailing punctuation, 279
(8.3%) did not resolve at all. 683 of 1,881 schema-valid answers (36.3%) contained
at least one unresolvable id. Where *all* ids failed, S2 recorded `no_cited_text`
(522 answers). Where only *some* failed, the premise was silently truncated and
the answer then scored as unsupported — so the corruption reached beyond the
visible failure count.

**Why this is a correction and not an amendment.** `19::100` and `19::100)`
denote the same passage. Resolving the citation the model plainly made is correct
implementation of the registered definition, not a change to it. The estimand,
the thresholds, the strata and the allocation are all untouched.

**What is deliberately NOT resolved.** A citation resolves only if normalisation
maps it onto exactly one offered id. Bare DocumentIDs (`17`) name a document
rather than a passage and stay unresolved. Ids that normalise onto two offered
ids are refused rather than guessed. Genuinely invented ids stay unresolved and
are reported as such.

**Measured effect of the correction.** Re-running S1-S3 on the same generations,
with no regeneration:

| | before | after |
|---|---|---|
| S2 passing | 597 (31.7% of S1) | 750 (39.9% of S1) |
| answers citing nothing resolvable | 522 | 126 |
| candidate_recoverable / unrecoverable / control | 110 / 27 / 460 | 142 / 40 / 568 |
| stratum weights | .1843 / .0452 / .7705 | .1893 / .0533 / .7573 |

Of 3,343 cited ids: 63.6% exact, 28.0% normalised, 8.3% unresolved, **0 ambiguous**
— every recovery mapped onto exactly one offered passage, so none required a
guess. Stratum weights moved by under one percentage point, indicating the
correction recovered items roughly uniformly rather than reshaping the sample.
S2 still rejects 60% of schema-valid answers, so the majority of faithfulness
failures are genuine unsupported claims rather than artefacts of id matching.

**Reported as a result.** Citation resolution outcomes are now a funnel stage in
their own right. The residual unresolvable rate is a finding about structured-output
regulatory RAG, and the composite `DocumentID::PassageID` format used in the prompt
is a contributing cause that belongs in the limitations rather than being presented
as pure model hallucination.

### Deviation 2 — Deviation 1 reversed; the registered revision round is now being run

**Date.** 2026-08-10
**Sections affected.** 8 (inter-annotator agreement), 9 (decision rule)
**Status when made.** After Deviation 1. No new outcome data existed at the point
this was decided — this reverses a *process* decision, not a measurement.

**What changed.** Deviation 1 (2026-08-03) chose not to run the revision round
Section 8 permits and demoted the base rate to a secondary result. That choice
is reversed. The second annotator (Riyad) asked to make another attempt at the
B/C boundary before the base rate is given up on as primary. The registered
revision round — Section 8's "sharpen the guidelines once, re-annotate 30 fresh
items, recompute κ" — is now being executed. This is the one revision the
pre-registration permits; if κ remains below 0.50 after it, Section 8's
original instruction stands: the natural-occurrence measurement is abandoned
and the study moves to Pivot A.

**Guideline changes.** `docs/ANNOTATION_GUIDELINES.md` moves to v1.1, making
explicit the three fixes the Deviation-1 disagreement analysis identified:
(a) incompleteness is A, stated as its own edge case rather than left implicit
in the framing paragraph; (b) an explicit scope/application-clause comparison
procedure, rather than the general "check whether the passage applies to that
scope" wording in Step 3; (c) worked examples built from the actual failure
pattern (7 of 7 A→B flips cited scope mismatch; two of the PI's B labels
penalised incompleteness). No change to the A/B/C/NA definitions themselves —
Section 4 is untouched. This is a clarification of an already-registered
distinction, not a new criterion.

**Revision batch.** 30 fresh items, disjoint from the original 150 (drawn from
`data/interim/pilot_v1/filtered.jsonl` survivors excluding every `item_id` in
`batch_01_key.jsonl`), stratified proportionally to the registered allocation
(16 candidate_recoverable / 4 candidate_unrecoverable / 10 control — exactly
1/5 of 80/20/50), seed 20260728. **Both annotators label all 30, independently
and in parallel** — this is a full double-annotation of the revision batch, not
a split; κ requires the same items judged twice. `scripts/04b_make_revision_batch.py`
and `scripts/05b_compute_revision_kappa.py`.

**What happens to the original 150-item primary labels.** They stand as-is if
κ on the revision batch clears the threshold. Section 8 commits only to
re-annotating 30 fresh items and recomputing κ on those — it does not commit to
re-labelling the primary sample. **Caveat, stated here rather than glossed
over:** the original 150 were labelled under guidelines v1.0; if κ on the v1.1
guidelines clears 0.50 on the revision batch, that establishes reliability
*under v1.1*, not under the v1.0 wording the primary labels were actually made
with. The gap is judged small — v1.1 clarifies existing Section 4 language, it
does not change what A/B/C mean — but it is a real gap and belongs in the
paper's limitations if the primary base rate is reported as-is. Re-labelling
all 150 under v1.1 would close it properly; that is not being done now, in the
interest of the time both annotators asked to conserve. Reconsider this
trade-off explicitly before drafting the results section, not silently at that
point.

**Who decided.** Adnan and Riyad, jointly, 2026-08-10.

**Resolution — 2026-08-12.** Both annotators labelled all 30 revision items
independently. Cohen's κ = **0.7860**, observed agreement 27/30 = 0.9000. This
clears both bands in Section 8: it is not merely "usable" (≥0.50) but
**"reliable" (≥0.70)**. Per the pre-committed table, the action is *proceed;
report κ*. The revision round succeeds; only one was permitted and it worked,
so no further guideline changes are made.

Three disagreements out of 30, no longer one-directional the way the original
round's were. One NA/C (a genuinely borderline heading-only passage). One A/B —
the PI called A, the second annotator caught a scope mismatch the PI missed
(same *direction* as the original bias, but 1 case in 30 rather than 7 in 50 —
reduced, not eliminated). One B/C — both annotators agreed the passage was
wrongly grounded, disagreed only on how obviously so, which is a milder
disagreement than a full A/B flip. Full disagreement detail with both notes:
`runs/pilot_v1/revision_kappa.json`.

**Decision.** The base rate returns to being reported as a primary result.
`r = 5.9% [2.2%, 10.7%]` stands, computed from the original 150 PI-only labels,
now reported alongside **κ = 0.7860 (revision round, 30 items)** as the
operative reliability figure. κ = 0.3663 (original round, 50 items) is retained
and reported as history — the reason the revision round was run — not
presented as the current reliability estimate.

**Addendum — 2026-08-13, bootstrap interval on κ.** Raised by Riyad as a fair
question: is 30 items enough to trust 0.7860 as settled? It wasn't previously
possible to answer that quantitatively — κ had only ever been reported as a
bare point estimate, here and in the original round. `bootstrap_kappa_interval`
was added to `src/saferag/pilot/stats.py` (paired non-parametric bootstrap,
10,000 resamples, same seed convention as the rest of the study) and run on the
real revision-round labels: **κ = 0.7860, 95% CI [0.4563, 1.0000].** The
interval is genuinely wide at n=30, and its lower bound sits in "moderate," not
"reliable," territory — so the honest statement is not "reliability is
settled" but "the point estimate clears the bar, and the data cannot rule out,
at 95% confidence, that the true reliability is only moderate." Report the
interval alongside the point estimate in the paper, not the point estimate
alone. This does not reopen Section 8's decision rule, which is defined on the
point estimate and was satisfied — but it is real information the rule itself
doesn't capture, and hiding it would be exactly the kind of smoothing this
document exists to prevent.

**The v1.0-vs-v1.1 consistency caveat, resolved.** Flagged when Deviation 2 was
opened: the original 150 were labelled under v1.0, so a passing κ on v1.1 does
not by itself certify those specific labels. Evidence bearing on this, from the
revision round: the PI's residual A/B error rate under v1.1 (1/30 ≈ 3.3%) is
markedly lower than under v1.0 (7/50 = 14%), a rate reduction consistent with
the guidelines fix working rather than merely being present, though 30 items is
too few to bound it tightly. The full 150 are not being re-labelled — that
remains future work if a reviewer requires it — but a targeted check of the PI's
own A-labelled items in `candidate_recoverable`/`candidate_unrecoverable` (the
strata deceptive grounding actually occurs in, 86 of the 100 items there) against
the v1.1 scope-clause test is a live option, cheaper than a full re-annotation.
Whether to run it is recorded as a decision for the write-up stage, not settled
here.

**v1.0/v1.1 consistency decision — 2026-08-12.** The primary 150 are reported
as-is, not re-audited or re-labelled. Decided by Adnan. Basis: the reassurance
evidence in the resolution above (PI's A/B error rate 14% → ~3.3% under the same
guidelines fix, same direction of residual bias but an order of magnitude
smaller) is judged sufficient, and the targeted-audit alternative (1.5–3 hours
against the PI's 86 A-labelled items in the two candidate strata) was not taken
up. This is stated here as a limitation for the paper: **the primary 150-item
base rate was labelled under guidelines v1.0; the reliability figure reported
alongside it (κ = 0.7860) was measured under v1.1.** If a reviewer requires it,
the targeted audit remains available and is documented above with its cost.

### Amendment 2 — operational definition for RQ-a's question-type breakdown

**Date.** 2026-08-12
**Sections affected.** 2 (RQ-a), none of 4-9 (does not touch S1-S4, the
sampling plan, the estimator, or the decision rule).
**Status when made.** After `r` and κ were both known (the primary base rate
and the revision-round reliability are already computed). **Before** the RQ-a
breakdown itself has been computed — this declares the split rule first, so the
category boundary cannot be chosen by looking at which split makes the
subgroup rates come out favourably. That ordering is what makes this an
amendment rather than post-hoc slicing, even though it comes later than
Amendment 1 did.

**Why this was needed.** RQ-a (Section 2) asks whether cross-reference and
multi-obligation questions are affected more than single-obligation ones, but
Section 4 never operationalised "cross-reference" or "multi-obligation" — the
terms were named, not defined, and `src/saferag/data/obliqa.py` does not even
retain ObliQA's `Group` field through the pipeline. Verified by inspection
2026-08-12: this is a genuine gap in the original pre-registration, not an
oversight found and quietly patched.

**What was chosen.** **Cross-reference = a question with more than one gold
passage** (`len(gold_passage_ids) > 1`), **single-reference = exactly one**.
This is a structural property of the question available from the dataset
before generation, retrieval, or annotation — it does not depend on the model's
output, the retriever's behaviour, or any human label, so it carries none of
the circularity risk a definition built from e.g. the model's own `obligations`
field would. "Multi-obligation" is not separately operationalised: the only
available proxy (the number of obligations the *system* extracted) is a
downstream generation artefact, not a property of the question, and conflating
it with cross-reference would muddy which factor is doing the explaining. RQ-a
is answered here only for the cross-reference / single-reference split;
multi-obligation is left as a stated limitation rather than defined badly to
force an answer.

**Analysis.** Unweighted (not re-stratified by S3 pool) Wilson interval on the
B-rate within each question-type group, computed over the same 150 primary
labels used for the headline `r` — a descriptive secondary cut, not a change to
the primary stratified estimator in Section 7. Re-stratifying by both S3 pool
and question type would fragment n=150 into cells too small to interpret; kept
as a limitation of this specific breakdown rather than hidden. `scripts/05c_rqa_breakdown.py`.

**Who decided.** Adnan, 2026-08-12, on the recommendation above.

| Date | Section | Deviation | Reason |
|---|---|---|---|
| 2026-08-12 | 2 | Amendment 2 (above) | RQ-a's "cross-reference" was never operationally defined in Section 4. Defined as >1 gold passage, declared before the breakdown was computed. |
| 2026-08-12 | 8, 9 | Deviation 2 resolution (above) | κ = 0.7860 on the 30-item revision batch, "reliable" band. Base rate reinstated as primary result. v1.0/v1.1 label-consistency gap reported as a limitation, not audited. |
| 2026-08-10 | 8, 9 | Deviation 2 (above) | Deviation 1 reversed. Registered revision round now executing: guidelines sharpened to v1.1, 30 fresh items double-annotated by both annotators in parallel, κ to be recomputed. |
| 2026-08-03 | 8, 9 | Deviation 1 (above) | kappa 0.3663 below the registered 0.50 threshold. Revision round not run; base rate demoted to preliminary secondary result with kappa and direction of bias reported. **Superseded by Deviation 2.** |
| 2026-07-30 | none | Correction 1 (above) | Citation ids resolved by normalised match. Implementation fix to the registered definition; estimand unchanged. |
| 2026-07-28 | 4, 6, 7 | Amendment 1 (above) | Retrieval recall@10 ≈ 0.82 implies ~16% of questions are unanswerable from context; splitting the candidate pool concentrates annotation on the stratum where deceptive grounding can actually occur. Made before any outcome data existed. |

## 12. Availability

Code, configuration, prompts, annotation guidelines, anonymised annotation labels and the analysis notebook will be released in the project repository. Corpus text is not redistributed; the loader fetches ObliQA from its original source.

---

*This pre-registration follows the spirit of the OSF pre-registration format, adapted for a computational measurement study. It is not lodged with a registry; its function is to fix analytic decisions before observation and to make any subsequent deviation visible.*
