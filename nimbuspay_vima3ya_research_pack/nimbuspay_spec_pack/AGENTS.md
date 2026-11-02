# AGENTS.md — Rules for Coding / Research Agents

## Mission

Implement and independently validate the VIMA3YA NimbusPay small-model ticket triage assignment using `PRD.md`, `SRS.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_BLUEPRINT.md`, `TEST_MATRIX.md`, `RESEARCH_PLAN.md` and `SUBMISSION_GATES.md`. Read `ASSIGNMENT.md`, `SCHEMA.md`, `score.py` first; those are the normative authorities.

## Nonnegotiable policy

1. **Do not fabricate success.** No metric, experiment, training run, paper result or reproducibility claim without raw artifacts and validation.
2. **Do not optimize against hidden test or dev labels.** The training dataset transformation takes only training rows and the schema; no dev/test in augmentation or labels. Split-overlap diagnostic is disclosed, not used to inject information.
3. **Do not use another model/API for labels or predictions.** AI assistants may write/review code and discuss reasoning but must not generate or correct training targets or final test outputs.
4. **Single final model.** One eligible <=4B open-weight base + PEFT adapter <=50M trainable parameters; no runtime ensembles, agents, validators as repairs, grammar decoding, or tool calls.
5. **Exact fixed inference.** Native chat template, supplied system+user messages only, greedy `max_new_tokens=200`, whitespace trim only. Never add rules/examples in final system/user content.
6. **No full fine-tuning.** Compare FP16 LoRA with 4-bit QLoRA. All runs must fit free T4.
7. **Data logging.** Every fixed/dropped training row is in `cleaning_log.csv(id,action,reason)`. Do not silently alter.
8. **Scoring integrity.** Use supplied `score.py` unchanged on raw predictions. Keep both official 200-row and disclosed overlap-free 170-row diagnostic dev metrics.
9. **Deadline and secrecy.** Private repo before deadline; no public release or GitHub posting of supplied raw training data/base weights; no pushes after deadline; visibility public and email within hour after deadline, under user-authorized workflow.
10. **Provenance and idempotency.** Hash source, config, code, selected checkpoint; atomic writes; resume only verified stages.

## Working order and stop gates

1. **Inspect** all source requirements, scorer and repo. Pin current SHA/checksums. Stop if sources missing or ambiguous.
2. **Audit** all 1,445 train rows independently. Treat previously identified counts as hypotheses to reproduce, not as a substitute for code audit. Reconcile overlap, duplicates and label corruption. Gate: every action traceable.
3. **Baselines** both forms on all dev rows. Gate: exact input/prompt behavior and score output.
4. **Tokenization/training correctness**. Gate: token-mask unit checks, supervised tokens survive truncation, parameter budget and GPU pilot.
5. **Controlled runs**: cleaned QLoRA r8 reference + raw, FP16, rank16, own idea. Gate: one variable changed, required measured columns complete.
6. **Error analysis**: ten actual dev failures, per-field/category and custom slices, residual risks.
7. **Reproduce** adapter from fresh T4 and re-generate test JSONL. Gate: no repair, parity tested and reload validated.
8. **Release** verify README/notebook outputs/report/adapter/log/predictions, commit timestamp, privacy and user approval for external actions.

## Quality controls

- Use precise typed interfaces and small, testable functions over heavyweight abstractions.
- Separate `evidence.py` and data cleaning from `generation.py`. It is an architectural violation if inference imports train-label repair modules.
- Code comments and docs explain **why** a decision exists, not trivial syntax.
- Use `Decimal` rather than float for currency. Preserve relative-date anchoring and explicit no-channel cases.
- Make status machine-readable (`RUNNING/PASSED/FAILED/BLOCKED`) and include exact failing ID/command/error, not generic assurances.
- Exclude notebook temporary outputs, raw data, model cache, tokens and secrets from Git. Avoid exposing tickets in logs more than necessary.
- Review subprocess return codes; don't hide exception traces by generic retry loops.
- Do not apply destructive Git or filesystem operations to unrelated user data.
- If GPU unavailable, implement CPU-verifiable pieces and mark training/metrics as unverified. Never mock a GPU run or fabricate notebook output.
- If the assignment has a genuine contradiction, seek clarification or record the strict interpretation and limitations; do not invent permission.

## How to report progress

Report (a) verified facts, (b) unverified hypotheses, (c) artifacts changed, (d) commands and observed results, (e) unresolved blockers, (f) next authorized action. Include numbers only with traceable machine logs. Don't call an output "perfect", "zero bugs" or "A* ready" simply because tests pass.

## Review rubric

- **Data layer:** independent corruption checks, exhaustive row action logging, no dangerous repairs, no leakage.
- **Training layer:** FP16/4bit memory truth, LoRA parameter count, assistant-only mask, context coverage.
- **Experiment layer:** correct one-factor design, equal compute treatment, full measured table, authentic negative results.
- **Inference layer:** unmodified raw strings, exactly 400 ids, prompt+generation parity, reloaded PEFT format.
- **Research layer:** prior art accurately cited, novelty described as hypothesis, statistical and generalization limitations.
- **Submission layer:** correct file structure, under-three-hour notebook, <=3-page report, Git visibility/timing.

## Document precedence

`ASSIGNMENT.md` governs allowed methods and delivery. `SCHEMA.md` governs target semantics. `score.py` governs official scoring. Every derived local spec is lower priority. When source instructions actually conflict, stop, disclose and seek clarification.

