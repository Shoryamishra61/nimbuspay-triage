# Implementation Blueprint — Repo Layout, APIs, Commands, Runtime

## 1. Proposed repository layout

```text
nimbuspay-triage/
  README.md
  notebook.ipynb                    # must run top-to-bottom on a fresh T4 <3h
  report.pdf                       # <=3 pages
  requirements.txt                # pinned after testing, no unverified latest claims
  cleaning_log.csv                 # id,action,reason
  predictions_test.jsonl          # raw output strings, exactly 400 unique ids
  adapter/
    adapter_config.json
    adapter_model.safetensors
  src/triage/
    contracts.py                   # constants + schema/invariants
    source_io.py                   # JSONL reader, raw checksums, split manifests
    audit.py                       # independent structural/semantic checks
    clean.py                       # train-only deterministic repairs/drop decisions
    evidence.py                    # Decimal amounts, dates, NP refs; offline ONLY
    tokenizer_pipeline.py          # chat template, supervised token masks
    training.py                    # LoRA/QLoRA setup, seeds, memory, checkpoints
    generation.py                  # minimal exact inference; NO repairs
    scoring.py                     # wrapper around supplied score.py; never edit scorer
    experiments.py                 # one-factor configs, metrics, manifests
    probes.py                      # read-only diagnostics and slices
    release.py                     # envelope/schema/adapter/file/commit gates
    provenance.py                  # hashes, atomic staging, run manifest
  configs/
    reference.yaml
    raw.yaml
    fp16.yaml
    rank16.yaml
    hardcase.yaml
  tests/
    test_contracts.py
    test_clean_idempotency.py
    test_evidence.py
    test_token_masks.py
    test_generation_contract.py
    test_score_parity.py
    test_release_gate.py
    test_metamorphic.py
  experiment_logs/
    summary.csv
    manifests/                    # small artifacts; NOT training/test data
  .gitignore
```

This is a blueprint, not a claim those code files have been implemented. Training data live outside the Git repo, e.g. a mounted Drive folder, referenced via configurable paths.

## 2. Module boundary contracts

### `source_io.py`

- `load_jsonl(path, expected_split) -> list[dict]`: UTF-8 strict; error includes line number; no silent skipped rows.
- `source_manifest(paths) -> {sha256, bytes, rows, ids}`.
- `validate_roles(rows, split)`: training/dev have system,user,assistant; test only system,user. Validate stable ids.
- `scan_split_overlap(train,dev,test) -> diagnostic`: fingerprints for **reporting**, not training labels.

### `contracts.py`

- `validate_record(obj, original_ticket=None) -> list[ValidationIssue]` with strict keys/types/values and certain cross-field invariants.
- `classify_priority(category, amount_paise, has_legal_threat) -> P1/P2/P3` strictly ordered.
- `expected_needs_human(priority, prior_support_contact) -> bool`.
- Enums and schema defined once from source spec, tests cover all 7 categories, 3 priorities, 4 channels, 2 languages.

### `evidence.py` (OFFLINE ONLY)

- `find_amount_candidates(text) -> list[Evidence(value_paise, span, cue, confidence)]` using `decimal.Decimal`, never floats.
- `find_date_candidates(text, received_date) -> list[Evidence(iso_date,...)]`; exclude quote-start lines for date.
- `find_np_identifiers(text)` with explicit transaction/reference evidence and NP+10-digit canonicalization.
- `has_legal_threat`, `has_prior_support_contact`, `is_hinglish_heuristic`: *candidate detectors*, never authoritative on ambiguous sentences.
- No method in `generation.py` may import or invoke these functions.

### `audit.py` / `clean.py`

- `audit_train(train, schema) -> AuditResult`: error classes, counts, row ids, evidence spans, confidence; no write.
- `propose_changes(train, audit_result) -> list[Decision]` with explicit drop/fix reason and proposed complete replacement JSON for certifiable cases.
- `apply_changes(train, decisions) -> cleaned_rows, cleaning_log` deterministic, atomic, stable id order. Log every changed/dropped row once with compound reason as needed.
- `check_changed_ids(raw,clean,log)` asserts equality; no use of dev or test in cleaning functions or fixtures.
- For 70 legacy rows, map old `prio` *only after recomputing current priority rules*; date format migration does not prove all other fields correct.

### `tokenizer_pipeline.py`

- `make_prompt_completion(row)` returns two-message prompt and one assistant completion.
- `encode_and_mask(row,tokenizer,max_length)` validates that prompt labels are `-100`, assistant JSON and EOS supervised, no duplicate control tokens, target nonempty and not truncated.
- `length_profile(rows,tokenizer)` logs prompt/target/full token quantiles and coverage, including 40 long-quoted train cases.
- If TRL `assistant_only_loss=True` is used, verify generated assistant mask is non-empty and correct. Do not assume Qwen2.5's unmodified chat template supports generation markers.

### `training.py`

- `load_base(model_id,revision,precision)` with optional NF4 quantization and `torch.float16` compute on T4.
- `build_lora(base, config)` with frozen base; list trainable modules; `sum(p.numel() for p in model.parameters() if p.requires_grad) <= 50_000_000` assertion.
- `run_train(config,tokenized_train,output_dir)` logs GPU name, package versions, seed, config, source SHA, trainable ratio, step/epoch/token counts, peaks, minutes, checkpoint and status.
- Use gradient checkpointing, effective batch explicitly, consistent optimizer, no premature `torch.cuda.empty_cache()` in memory measurement logic.
- `save_pretrained` to a staged PEFT adapter directory and verify reloading.

### `generation.py` — strict minimal function

Pseudo-code only; use tested API arguments for pinned Transformers version:

```python
messages = row["messages"][:2]            # exactly supplied system and user
prompt = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt"
)
input_ids = prompt.to(model.device)
generated = model.generate(
    input_ids=input_ids, do_sample=False, num_beams=1,
    max_new_tokens=200
)
new_tokens = generated[0, input_ids.shape[1]:]
raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
# Store raw_output directly. DO NOT parse and rewrite, retry, regex-fix or fill.
```

Use native EOS handling, attention masks and pinned config consistently. `skip_special_tokens` during decode is normal token decoding, not rewriting parsed fields. Verify the exact evaluator-compatible behavior by a fresh reload.

### `experiments.py` / `scoring.py`

- Two dev baselines with same untouched model: fixed prompt, rule-added prompt. Rules may appear **only** in reference baseline.
- Five train runs for four controlled comparisons; keep all other fields fixed and same update budget wherever feasible.
- Run original supplied scorer on full 200 dev labels; store raw predictions + JSON summary. Calculate 170 overlap-free diagnostic separately and disclose the 30 overlap.
- Record model revision, data hash, prompt variant, bits, rank, learning rate, epochs, optimizer, peak allocated/reserved GiB, elapsed minutes, exact, field accuracy, validity, all per-field/category metrics.

### `release.py`

- `validate_prediction_envelope(predictions,expected_test_ids)`: exactly 400 unique IDs, every `output` a string, not rewritten.
- `validate_adapter(adapter_dir,expected_model_id,trainable_params)`: PEFT files exist, size, load, fresh generation parity sample.
- `validate_repo()`: no training data/base weights/secret tokens, report <=3 pages, notebook output present, file permissions sane, latest commit recorded before deadline.
- `run_release_gate` is read-only except report generation; it must fail instead of auto-editing predictions.

## 3. Configuration discipline

One canonical reference config; variants are deep copies with exactly one changed factor. Example conceptual YAML:

```yaml
model_id: Qwen/Qwen2.5-1.5B-Instruct
revision: PIN_A_REAL_VERIFIED_COMMIT
seed: 42
precision: qlora4_nf4_fp16
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 0.0002
epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_seq_length: MEASURE_AND_SELECT
mask_assistant_only: true
training_data: cleaned
sampling: uniform
generate_max_new_tokens: 200
generate_do_sample: false
```

Do not run this example verbatim: `revision` and `max_seq_length` are intentionally unset until evidence exists, and actual library argument names/versions must be verified. If `target_modules` exceeds memory, profile and adjust with a new documented config rather than silently reducing modules.

## 4. Local CLI plan

```bash
python -m triage.audit --train /private/data/train.jsonl --schema /private/SCHEMA.md --out artifacts/audit.json
python -m triage.clean --train /private/data/train.jsonl --schema /private/SCHEMA.md --out /private/cleaned.jsonl --log cleaning_log.csv
python -m triage.tokenizer_pipeline --train /private/cleaned.jsonl --config configs/reference.yaml
python -m triage.experiments --config configs/reference.yaml --train /private/cleaned.jsonl --dev /private/data/dev.jsonl
python -m triage.generation --adapter adapter --test /private/data/test_inputs.jsonl --out predictions_test.jsonl
python -m triage.release --repo . --test-ids /private/data/test_inputs.jsonl
```

These are intended entry points for an implementing agent, **not commands known to work before coding/testing**.

## 5. Notebook narrative, top to bottom

1. Hardware/runtime and dependencies; immutable hashes; seed.
2. Data inspection and independent audit; notable examples, no raw corpus dump.
3. Cleaning/log generation and verification; precise mutation counts.
4. Tokenizer length profile, rendered chat template and supervised-mask assertions.
5. Both baseline predictions/scorer results (saved verified outputs if runtime cost demands).
6. Primary training and recorded metrics; save adapter.
7. Controlled experiment code and saved original logs (can be opt-in; must be real runs elsewhere).
8. Final model selected by documented dev criteria; fresh adapter reload.
9. Dev error analysis and official scores, 10 specific failures.
10. Generate raw test predictions and enforce submission envelope.

The **default fresh top-to-bottom path** must reproduce the selected final model and test JSONL within three hours, even if extra optional experiment reruns are disabled by default. Document why/how prior experiment outputs were obtained and retain their code and logs.

## 6. Engineering decisions to verify rather than assume

- Qwen tokenizer/TRL assistant loss mask support.
- Model tensor dtype on T4, 4-bit bitsandbytes version and kernel compatibility.
- Trainable parameter count for chosen target modules and rank.
- Token length and max-new-token sufficiency for eight-key JSON responses.
- FP16 LoRA T4 peak memory and speed at selected sequence limit.
- Exact adapter reloading and precision parity.
- Score using **raw** generation, not processed JSON.
- GitHub PEFT safetensors under 100 MB or validated LFS setup.

