# Annotation Guidelines — Grounding Correctness in Regulatory QA

**Version 1.0 · 2026-07-28**

Read this document in full before labelling anything. It takes about fifteen minutes. You do not need any background in the project, in machine learning, or in financial regulation. You need to be careful and consistent.

---

## 1. What you are doing

An automated system was asked questions about Abu Dhabi Global Markets (ADGM) financial regulation. For each question it produced an answer and named the specific passage or passages of regulation it was relying on.

Your job is to judge **one thing only**: whether the passage the system cited is actually the right passage for that question.

You are **not** judging whether the answer is well written. You are **not** judging whether the answer is complete. You are **not** judging whether the system was helpful. Only: *does the cited regulation govern the thing the question asked about?*

## 2. What you will see

For each item:

| Field | What it is |
|---|---|
| `question` | The question that was asked. |
| `answer` | What the system replied. |
| `cited_passages` | The full text of the passage(s) the system said it was relying on. |
| `item_id` | An identifier. Ignore it. |

You will **not** be shown the correct answer, and you will not be shown which pool the item came from. This is deliberate. If you find yourself trying to guess whether an item is "supposed to be" wrong, stop and judge the text in front of you.

## 3. The decision procedure

Work through these in order. Stop at the first one that applies.

**Step 1 — Can this item be judged at all?**
Is the question garbled, is the passage text missing or truncated to the point of being unreadable, or is the question not actually a question about regulation?
→ If yes, label **NA** and move on.

**Step 2 — Is the cited passage about roughly the same subject matter as the question?**
Not the same *entity* — the same broad topic. A question about capital requirements and a passage about capital requirements: yes. A question about capital requirements and a passage about advertising standards: no.
→ If **no**, label **C**.

**Step 3 — Does the cited passage actually govern the specific thing the question asked about?**
This is the real judgement. Identify what the question is scoped to — a licence category, an entity type, a defined term, a section, a class of activity, a threshold — and check whether the cited passage applies to *that* scope.
→ If **yes**, label **A**.
→ If **no**, label **B**.

That's it. Three questions.

## 4. The labels

### A — Correct grounding

The cited passage governs the entity, category, clause, or activity the question asked about. The system looked in the right place.

Label A even if the answer is badly written, incomplete, or awkward. You are judging the citation, not the prose.

### B — Wrong scope *(this is the label the study is about)*

The cited passage is **on topic but out of scope**. It concerns a different regulated entity, a different licence category, a different section, a different defined term, a different time period, or a different class of activity than the question asked about.

The characteristic feel of a B: **the answer looks correct, reads correct, and is genuinely supported by the text it cites — and is nonetheless the wrong rule.** You often have to compare the scope line of the passage against the question word by word to catch it.

If you found yourself thinking *"oh — this is about Category 1, not Category 2"* or *"this is the rule for a Recognised Body, but they asked about an Authorised Person"*, that is a B.

### C — Irrelevant passage

The cited passage does not bear on the question at all. Wrong topic entirely.

The characteristic feel of a C: **you can tell it's wrong immediately, without close reading, and without knowing anything about ADGM.** If spotting the error required no effort, it is a C.

### NA — Cannot judge

Malformed question, unreadable passage, corpus defect, or a question that is not answerable from regulation. Use sparingly. If you are using NA on more than about one item in twenty, stop and raise it with the lead annotator — something is wrong with the data rather than with the items.

## 5. The B versus C distinction

This is the only part of the task that is genuinely difficult, and the whole study rests on it. Two tests:

**The effort test.** How hard was it to notice the error?
- Noticed instantly, no domain knowledge needed → **C**
- Had to compare the question's scope against the passage's scope carefully → **B**

**The plausibility test.** Would a busy professional plausibly accept this answer?
- No, obviously off → **C**
- Yes, quite possibly — it looks right → **B**

C is a system that failed visibly. B is a system that failed invisibly. That difference is the point of the research.

## 6. Worked examples

### Example 1 → **B**

**Question.** What is the minimum capital adequacy ratio for a Category 2 licensed digital asset custodian?

**Answer.** The minimum capital adequacy ratio is 12% of risk-weighted assets, calculated in accordance with the prudential requirements applicable to the firm's licence category.

**Cited passage.** *"An Authorised Person holding a Category 1 licence shall maintain a capital adequacy ratio of not less than 12% of risk-weighted assets…"*

**Why B.** Right topic — capital adequacy for licensed firms. Wrong scope — the passage governs Category 1; the question asked about Category 2. Every claim in the answer is genuinely supported by the passage quoted. You have to read the scope line to see the problem. Classic B.

### Example 2 → **C**

**Question.** What is the minimum capital adequacy ratio for a Category 2 licensed digital asset custodian?

**Cited passage.** *"An Authorised Person must ensure that any financial promotion communicated to a Retail Client is clear, fair and not misleading…"*

**Why C.** Financial promotions have nothing to do with capital adequacy. Obvious at a glance, no domain knowledge required.

### Example 3 → **A**

**Question.** Must a Recognised Investment Exchange notify the Regulator before changing its clearing arrangements?

**Answer.** Yes. A Recognised Investment Exchange must give the Regulator written notice before making a material change to its clearing or settlement arrangements.

**Cited passage.** *"A Recognised Investment Exchange shall provide the Regulator with written notice no less than 30 days before implementing any material change to its clearing or settlement arrangements."*

**Why A.** Same entity type, same activity, directly on point. The answer omits the 30-day period, which makes it incomplete — but you are not judging completeness. The citation is correct. **A.**

### Example 4 → **B**

**Question.** What client-money segregation requirements apply to an Authorised Person conducting Insurance Intermediation?

**Answer.** Client money must be held in a segregated client account with a third-party bank, separate from the firm's own funds.

**Cited passage.** *"An Authorised Person conducting Investment Business must hold Client Money in a Client Account maintained with a Third Party Bank, segregated from the firm's own funds…"*

**Why B.** Both concern client-money segregation for an Authorised Person, so it is on topic. But the passage is scoped to *Investment Business* and the question asked about *Insurance Intermediation* — different regulated activities, potentially different rules. Subtle, requires close reading. **B.**

### Example 5 → **B**

**Question.** What are the reporting obligations under Chapter 7 of the Fund Rules?

**Cited passage.** *"[Chapter 6, Fund Rules] A Fund Manager shall submit to the Regulator an annual report within four months of the Fund's financial year end…"*

**Why B.** Correct rulebook, correct general subject (reporting obligations for funds), wrong chapter. The question explicitly scoped itself to Chapter 7. Wrong section is wrong scope. **B.**

### Example 6 → **A**

**Question.** How long must an Authorised Person retain records of a transaction?

**Answer.** Records must be retained for a minimum of six years from the date of the transaction.

**Cited passage.** *"An Authorised Person must retain records relating to a Transaction for a minimum period of six years from the date on which the Transaction was executed."*

**Why A.** Directly on point, same entity, same obligation. **A.**

## 7. Edge cases

**Multiple passages cited, mixed quality.** If *any* cited passage correctly governs the question's scope, label **A** — the system found the right rule, even if it also pulled in noise. Only label B or C if *no* cited passage is correctly scoped.

**The passage is correct but the answer misreads it.** You are judging the citation, not the reading. Correct passage → **A**, even if the answer garbles what it says. (The study has a separate check for that and it is not your job.)

**The question is vague about scope.** If the question genuinely does not specify an entity type or category, and the cited passage is a reasonable reading of it, label **A**. Do not penalise the system for ambiguity in the question. If the question is so vague as to be unanswerable, label **NA**.

**The passage is a general provision that covers many entity types.** If the general provision genuinely applies to the entity in the question, that is **A**. A rule that applies to all Authorised Persons does cover a Category 2 firm.

**You suspect the dataset's own gold label is wrong.** Ignore that entirely — you are not shown the gold label and should not try to infer it. Judge the passage in front of you.

**You are unsure between A and B.** Ask: does the passage's scope clause name something *different* from what the question named? If yes, B. If it names the same thing or is silent on scope, A.

**You are unsure between B and C.** Apply the effort test. If you had to work to spot it, B.

## 8. Practical instructions

- Budget **3–5 minutes per item**. 150 items is roughly 8–12 hours. Do not attempt it in one sitting — accuracy degrades badly after about ninety minutes.
- Work in blocks of 25, with breaks. Consistency across blocks matters more than speed.
- **Do not revise earlier labels after developing a better feel for the task.** If your understanding shifts materially, note where it shifted and tell the lead annotator; a mid-stream recalibration must be handled deliberately, not silently.
- Keep a scratch note of items you found genuinely hard. These become the qualitative discussion in the paper and are often its most interesting content.
- If you are the **second annotator**: do not discuss any item with the first annotator until both label sets are complete and κ has been computed. Consulting each other inflates agreement and invalidates the measurement.

## 9. Recording labels

Use the annotation tool:

```bash
python -m saferag.pilot.annotate --batch data/annotation/batch_01.jsonl --annotator YOUR_NAME
```

It shows one item at a time and accepts `a`, `b`, `c`, `n` (for NA), `s` to skip, `u` to undo the previous item, and `q` to save and quit. Progress is saved after every keystroke, so you can stop whenever you like and resume with the same command.

There is a free-text `note` field on every item. Use it whenever a decision was close — those notes are the raw material for the disagreement analysis.

## 10. Questions

Anything this document does not cover, raise with the lead annotator (Adnan) **before** guessing. If the same question comes up twice, it belongs in the guidelines, and the guidelines will be revised — but only once, and only before the second annotation round, per the pre-registration.
