# Product Requirements Document (PRD)

## 1. Product identity

- **Product:** NimbusPay Support Ticket Triage PEFT Research Prototype.
- **Role:** VIMA3YA Machine Learning Intern — LLM Fine-Tuning take-home.
- **Delivery interface:** a single open-weight chat model with a PEFT adapter that emits one eight-key JSON object.
- **User:** evaluator submitting a fixed system and raw user ticket; downstream fictional support queue consuming the JSON.
- **Deadline:** 48 hours after actual pack receipt. Do not infer an exact timestamp; read email headers.
- **Effort target:** 6–8 focused hours, with fresh T4 reproducibility under three hours.

## 2. Problem

Map noisy, multilingual, sometimes adversarial support text to a canonical triage record that enforces category precedence, priority thresholds, monetary/date/ID normalization, source-specific context and escalation rules. Training labels contain multiple forms of corruption. The final model must internalize these rules without relying on inference-time rules or postprocessing.

## 3. Product principles

1. **Truthful contract adherence:** score raw text; never fix model outputs.
2. **Data provenance:** immutable raw files, deterministic transformations and an exhaustive action log.
3. **Scientific controls:** comparable paired experiments changing one variable at a time.
4. **Operational feasibility:** a free 16 GB-class T4, compact adapter, predictable run time.
5. **Human explainability:** every decision, error class and metric is traceable in the notebook.
6. **No speculative success claims:** negative results remain visible; only measured gains are claimed.

## 4. Target user journeys

- **UJ-01:** Given an email, IVR, social or chat ticket, the trained model emits only a JSON object with eight exact keys.
- **UJ-02:** An auditor can identify every transformed/dropped train row from `cleaning_log.csv` and reproduce counts.
- **UJ-03:** An interviewer can rerun the notebook top-to-bottom on a fresh T4 and reproduce the final adapter and test outputs within three hours.
- **UJ-04:** A reviewer can compare base-fixed and base-with-rules baselines and four controlled PEFT comparisons on the same dev scorer.
- **UJ-05:** A research reviewer can inspect at least ten real errors with causal hypotheses and suggested controlled next tests.

## 5. Functional features, ordered by priority

### P0 — Nonnegotiable

| ID | Feature | Acceptance |
|---|---|---|
| P0-01 | Eight-field raw JSON output | Only `category,priority,amount_paise,txn_date,txn_id,channel,language,needs_human`, correct types |
| P0-02 | Category precedence | Unauthorized/unrecognized transaction always `fraud`; other categories follow schema |
| P0-03 | Priority precedence | P1 fraud/claim >=₹50k/legal threat; P2 failed/refund >=₹5k or account access unless P1; else P3 |
| P0-04 | Exact numeric handling | Decimal-safe INR→paise conversion, `k`, `lakh`, distractor amounts ignored |
| P0-05 | Dates and references | Relative dates anchored to Received; year rollover; quoted-history dates excluded; canonical NP+10 digits |
| P0-06 | Multilingual and escalation | Hinglish detection; human escalation if P1 or prior support contact |
| P0-07 | Strict inference | Only fixed system/user messages, model chat template, greedy, 200 tokens, whitespace trim only |
| P0-08 | Model limits | Base <=4B; trainable LoRA <=50M; no full fine-tuning; free Colab T4 |
| P0-09 | Traceable cleaning | Drop/fix decisions logged with id/action/reason; train-only input to cleaning decisions |
| P0-10 | Reproducibility | PEFT adapter load, exact generation settings and predictions; notebook <3h fresh T4 |

### P1 — Scoring-critical

- Two baselines: no fine-tuning fixed prompt vs same base with rules in prompt **on dev only**.
- Four controlled comparisons (raw/clean, LoRA/QLoRA, rank or LR or epochs, original idea).
- Table of dev exact match, mean-field accuracy, peak GPU memory and training minutes.
- Per-field/category analysis with ten concrete failures.
- A <=3-page report, including limitations and AI assistance disclosure.

### P2 — Research-quality improvements if time allows

- Deterministic boundary-preserving augmentation from train plus schema only; isolated as an experiment.
- Stress and metamorphic evaluations separate from official scoring.
- Stratified confidence intervals and multi-seed repeats *after* the required experiments; never imply reliability from one seed.
- Model probing with read-only logits, label masks and error correlations; no use in test-time predictions.

## 6. Explicit exclusions

- Production service, distributed serving, retrieval, multi-model ensemble, hidden rule engine, tool-calling, grammar decoding or schema-correction pipeline.
- Large automated searches over hundreds of settings under restricted GPU quota.
- Claims of real financial-risk deployment readiness.
- Publishing raw supplied training/dev/test tickets or base weights.
- LLM-judge-generated labels, pseudo-labels or test predictions.

## 7. Metrics and business risk

**Official**: `score.py` dev exact match (primary), mean eight-field accuracy, JSON parse rate, per-field/category scores. Additional field-accurate rows with wrong extra keys do not count as exact.

**Supplementary**: overlap-free dev slice, failure taxonomy, boundary-case error rate, false negative fraud escalation, input-source and language breakdown, generation token length, GPU max allocated/reserved, training and inference wall time. All supplementary slicing must be disclosed and use no dev for training.

**Risk:** Actual fraud/legal escalation errors have asymmetric consequences. The official scorer does not weight such errors differently. Report both its metric and this limitation; never quietly change the official metric.

## 8. Primary experiment contract

Use one base model, one seed, one tokenizer/template, one effective batch and fixed optimizer-step budget. The reference experiment is cleaned 4-bit QLoRA rank 8, uniform sampling. Variants: raw data; FP16 LoRA; rank 16; hard-case sampling. If a variant changes effective number of tokens/steps, explicitly report the confound and, where feasible, normalize the budget.

No table may include invented performance numbers. For each experiment persist a manifest and raw generated dev predictions.

## 9. Acceptance and release

Deliver exactly the required repo files (README, adapter, output-bearing notebook, <=3-page report, cleaning log, 400-row test JSONL). Keep private through deadline, freeze last commit before it, turn public within an hour afterward, email link. Run independent release validation before the freeze. If any gate fails, state it plainly and prioritize compliant submission over cosmetic complexity.

## 10. Unresolved product decisions

- **PD-01:** Employer interpretation of dev-ticket overlap versus train-only data cleaning; seek clarification, otherwise disclose both official and overlap-free scores.
- **PD-02:** Actual tokenizer length quantiles and T4 memory ceiling; must measure before final max length.
- **PD-03:** Whether FP16 LoRA fits chosen context length on free T4; prove with a pilot, else reduce model/context with documented tradeoff.
- **PD-04:** Evaluator's adapter loading precision; reproduce exact recorded inference and document base revision and quantization.

