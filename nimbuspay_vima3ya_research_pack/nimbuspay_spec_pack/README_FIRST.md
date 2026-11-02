# NimbusPay Triage — High-Assurance Engineering and Research Pack

Prepared for Shoryakumar Mishra | 2 November 2026

## Read this first

This is an **agent-ready planning, architecture, validation, and research specification**, not a claim that a trained model or final submission already exists. It is grounded in the supplied `training_assessment.zip`, its `ASSIGNMENT.md`, `SCHEMA.md`, `score.py`, and a preliminary independent data audit.

### Two tracks, deliberately separated

1. **TRACK A — assignment submission**: one open-weight model with <=4B parameters, a PEFT LoRA/QLoRA adapter with <=50M trainable parameters, training on a free Colab T4, exactly the supplied two-message prompt at test time, greedy `model.generate(max_new_tokens=200)`, no output repair, no multi-model prediction, and a private-to-public deadline workflow.
2. **TRACK B — research extension**: after the assignment, investigate reliability, selective data curation, counterfactual invariance, and structured output robustness with broader datasets and research-grade methodology. Do not introduce TRACK B components into TRACK A when they violate the assessment contract or its compute/time budget.

### Critical clarification about Qwen

`Qwen/Qwen2.5-1.5B-Instruct` is an eligible *candidate base model* (about 1.54B parameters). The prohibition on "another model or API" applies to **using an additional model/API to create training labels or final predictions**, not to training an allowed open-weight model. The supplied task expressly demands a trained model. Using the *same untouched Qwen base* for the two mandated dev baselines is also allowed. Qwen selection is a hypothesis until its T4, accuracy, and reproducibility behavior are measured.

### Absolutely forbidden at final inference

- No rules, extra examples, external retriever, chain-of-thought, or schema embedded in the fixed system/user messages.
- No ensemble, router, cascade, second model, specialized external tool, validator-repair loop, constrained decoding, JSON grammar, regex field replacement, retry-with-new-prompt, or fallback label generator.
- No rewriting the model's raw output apart from stripping whitespace.
- No training on dev/test labels or test inputs; no other model-generated labels/predictions.

**Allowed reliability layers** operate *offline*: data integrity checks, deterministic train-only repairs justified by `SCHEMA.md`, deduplication, versioned transformations, curriculum/augmentation experiments, training-time diagnostics, independent validators for evaluating raw outputs, adapter reload tests, and honest error analysis. Diagnostic validators never mutate final predictions.

### Preliminary findings — not a finished audit

| Finding | Observed count | Confidence / interpretation |
|---|---:|---|
| Malformed training label JSON | 25 | Direct parse failure |
| Legacy `prio` schema instead of `priority` | 70 | Direct key inspection |
| Duplicate training entries beyond first occurrence | 60 | Exact duplicate ticket and label |
| Noncanonical transaction IDs in parseable current-schema labels | 20 | Format check |
| Amount labels apparently retaining rupees rather than paise | 25 | Numerical cross-check; manually confirm before repair |
| Meaningless training ticket bodies | 10 | `test`, `asdf asdf`, blank, etc. |
| `P1` with `needs_human=false` | 30 | Definite cross-field contradiction where schema current |
| Candidate priority inconsistencies | 65 | Regex-based scan; each needs semantic verification |
| Exact train/dev ticket overlaps | 30 | Exact input and label match; 15% of dev |
| Long quoted-email train/dev/test samples | 40 / 3 / 11 | Difficult inputs, not automatically corrupt |

These groups overlap; do **not** sum them. There were 1,445 train, 200 dev, and 400 unlabeled test rows. Preliminary scans found 270 distinct train rows flagged across internal checks (including heuristic candidates and duplicates), excluding cross-split overlaps; this is *not* a certified count of incorrect labels. The complete audit, decisions, and cleaning log remain to be implemented.

### A real specification tension: train/dev overlap

The assignment says train-data cleaning can use only train plus the schema, and also says never train on dev rows. Thirty exact dev tickets occur in train. **Do not quietly use dev labels to correct training rows.** Ask the employer how to handle cross-split duplicate suppression if possible. In all cases publish both the official 200-row dev score and an overlap-excluded 170-row diagnostic score; disclose how the exclusion view was formed and any limitations. If you remove training rows based on dev fingerprints, disclose that as split-hygiene only and seek explicit permission because of the train-only cleaning restriction.

### Files in this pack

- `PRD.md`: objectives, scope, priorities, user stories, metrics, delivery and decision criteria.
- `SRS.md`: precise functional/nonfunctional requirements, contracts, invariants, traceability.
- `ARCHITECTURE.md`: layered design, data/control flow, error classes, idempotency, provenance, failure containment.
- `IMPLEMENTATION_BLUEPRINT.md`: codebase layout, APIs, CLI workflows, config and runtime behavior.
- `TEST_MATRIX.md`: unit, metamorphic, adversarial, property, integration, reproducibility, and release tests.
- `RESEARCH_PLAN.md`: verified scholarly sources, hypotheses, proposed ablations and paper-quality limitations.
- `AGENTS.md`: instructions to a coding/research agent implementing the pack.
- `MASTER_PROMPT.md`: standalone agent master prompt with phased gates and deliverable acceptance.
- `SUBMISSION_GATES.md`: exact packaging, evaluation, deadline, and live-defense checklist.

### Priority order

**Legality and assignment compliance > correctness > empirical evidence > reproducibility > measured accuracy > architectural sophistication.** Reject elegant approaches that violate the inference contract. Never claim "zero errors", a specific benchmark result, guaranteed employer acceptance, or A* publishability without evidence.

