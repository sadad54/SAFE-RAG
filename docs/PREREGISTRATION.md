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

| Date | Section | Deviation | Reason |
|---|---|---|---|
| 2026-07-30 | none | Correction 1 (above) | Citation ids resolved by normalised match. Implementation fix to the registered definition; estimand unchanged. |
| 2026-07-28 | 4, 6, 7 | Amendment 1 (above) | Retrieval recall@10 ≈ 0.82 implies ~16% of questions are unanswerable from context; splitting the candidate pool concentrates annotation on the stratum where deceptive grounding can actually occur. Made before any outcome data existed. |

## 12. Availability

Code, configuration, prompts, annotation guidelines, anonymised annotation labels and the analysis notebook will be released in the project repository. Corpus text is not redistributed; the loader fetches ObliQA from its original source.

---

*This pre-registration follows the spirit of the OSF pre-registration format, adapted for a computational measurement study. It is not lodged with a registry; its function is to fix analytic decisions before observation and to make any subsequent deviation visible.*
