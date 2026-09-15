# Research evidence and claim boundaries

This index reconciles the committed project narrative; it does not report a new
experiment or change registered labels. See PROJECT_STATE.md and PREREGISTRATION.md.

| Claim | Supported scope |
|---|---|
| Deceptive grounding base rate | Current conditional estimate r = 8.6% [4.5%, 13.5%], n=200; report the interval and conditioning, not just the point estimate |
| Annotation reliability | Revised-guideline 30-item round κ=0.7860; 50-item extension κ=0.6678; original κ=0.3663 is separate historical evidence |
| Citation-format ablation | Documented 3B and 7B model runs; no substantiated 14B result |
| Pilot size | Original 2,786-question run and 2,000-question ablation subset are different samples |
| Publication status | Repository writing plans do not establish submission or acceptance; verify against a submission receipt before making that claim |

Citation-ID resolution is a format/attribution check, not evidence that a claim
is entailed by its citation. Deceptive grounding concerns apparently grounded
answers whose cited material does not support the claim, including scope errors.
Do not equate improved ID resolution with improved factual faithfulness.

The batch_03 fold-in is complete. Earlier n=150 estimates and original-round
reliability remain historical context, not alternative current headline results.
The implemented pilot/evaluation pipeline should be distinguished from the
proposed full SAFE-RAG correction framework. Committed documentation of runs
is not a fresh rerun; source corpus and model access are required to reproduce
the experiments.
