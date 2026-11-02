# Submission Gates, 100-Point Traceability and Live Defense

## 1. Points to artifacts

| Assessment part | Points | Required traceable artifact / evidence |
|---|---:|---|
| A: training-data audit | 25 | Audit summary, before/after counts, code/row evidence, exhaustive `cleaning_log.csv` |
| B: two base baselines | 10 | Fixed prompt and rule prompt results on dev using supplied `score.py` |
| C: fine-tuning pipeline | 20 | Native template, supervised target mask, token-length evidence, FP16/QLoRA configs and rationale |
| D: controlled experiments | 20 | >=4 one-factor comparisons, table exact/mean fields/peak GPU/minutes, code and logs |
| E: error analysis | 15 | Weakest categories/fields, >=10 specific raw prediction failures, one-week roadmap |
| Hidden test | 10 | 400 raw, unmodified predictions in correct envelope |

The 30-minute live defense is said to count as much as the written submission, but the pack gives no separate numeric formula. Prepare for code changes live; do not claim full score based only on submitted files.

## 2. Hard release gates

- [ ] Model base <=4B open weights; model ID and revision pinned; license checked.
- [ ] Adapter trainable <=50M; only LoRA/QLoRA; trainable modules counted programmatically.
- [ ] Free Colab T4 actually used for measured training runs.
- [ ] Native chat template exactly maps supplied system and user messages.
- [ ] Training loss only assistant JSON+end-of-turn, with actual label mask screenshots/log tests.
- [ ] Token length distribution and sequence choice measured, no silent target truncation.
- [ ] All train fixes/drops appear once in `id,action,reason` cleaning log and reconcile exactly.
- [ ] No dev/test input or labels used to train, clean or augment; dev/train overlap handling disclosed.
- [ ] Both base baselines scored with original scorer.
- [ ] Four one-factor comparisons measured and reporting all metrics, including negative results.
- [ ] Final selected checkpoint loads from only `adapter_config.json` + PEFT weights, correct pinned base.
- [ ] Generation uses no rules prompt, no postprocessing beyond `.strip()`, no grammar, second model or repair loop.
- [ ] `predictions_test.jsonl` exactly 400 unique `ts-` ids, output strings raw, no duplicated rows.
- [ ] New-process adapter reload on representative dev/test inputs matches recorded outputs under declared environment.
- [ ] Fresh-T4 notebook top-to-bottom <3h, with saved outputs from actual execution and no missing data path assumptions.
- [ ] Three-page-or-shorter report includes audit, experiments, ten errors, one-week plan, AI tool disclosure.
- [ ] Staged Git files exclude training data, base weights, private tokens, model cache and redundant large artifacts.
- [ ] GitHub repo private before deadline; last commit is before exact receipt+48h deadline; no push afterward.
- [ ] Visibility public and email link within one hour after deadline, with explicit authority to perform external actions.

## 3. `README.md` minimum

- Exact base model id, revision, quantization mode, PEFT adapter type and trainable parameter count.
- One-line best measured dev exact/field result, **with real score and overlap disclosure**.
- Setup and fresh-Colab run instructions, input archive placement, dependency pinning.
- Exact inference command/config, reproduce predictions, test SHA/count, memory/time evidence.
- Link/name of report and experiment provenance; confidentiality and original data attribution.

## 4. `report.pdf` three-page editorial plan

- **Page 1:** Dataset audit taxonomy with known counts, decision logic, log reference, train/dev overlap disclosure and limitations.
- **Page 2:** Base model/LoRA details, exact mask/context rationale, base baselines and all controlled experiment measurements; honest comparisons.
- **Page 3:** Per-field/category weakness, ten compact failure cases with IDs/reasons, one-week scientifically sound follow-up plan, AI tools and boundaries.

Do not pretend plain-text `report.md` is automatically three pages. Render/check the final report if choosing PDF.

## 5. Checklist for live 30-minute defense

Prepare to explain and modify: sample repairing a rupee→paise mismatch with dependent priority, assistant-only target masking, why Qwen is allowed but second prediction model is not, actual peak T4 VRAM, rank ablation, train/dev leakage consequences, false positives on legal threats, source/quoted dates, raw JSON generation, checkpoint replay, and why a negative experiment result is informative.

No screen recording or synthetic logs should masquerade as actual execution. Practice a small change in the real notebook environment and be prepared to answer unplanned questions.

## 6. What cannot be certified ahead of execution

- Zero wrong outputs or perfect field accuracy.
- Which model/rank/precision wins.
- All time/memory targets on a future free T4 session.
- Reproducibility under a mismatched evaluator environment.
- Employer acceptance, internship offer or research-publication acceptance.

This gate list is intentionally evidence-based: if a gate is unmet, mark it with a precise explanation rather than asserting a perfect submission.

