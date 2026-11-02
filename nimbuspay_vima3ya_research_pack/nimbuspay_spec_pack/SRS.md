# Software Requirements Specification (SRS)

## 1. Normative sources and precedence

1. `candidate_pack/ASSIGNMENT.md`: limits, disallowed methods, deliverables, deadline.
2. `candidate_pack/SCHEMA.md`: sole normative labeling specification (v2).
3. `candidate_pack/score.py`: official dev scoring implementation.
4. This specification: derived operational requirements; it cannot override the three files above.

If this document conflicts with an assessment source, follow the assessment and record a decision. The 30 observed train/dev duplicate inputs create a real interpretation ambiguity; see `README_FIRST.md`.

## 2. Canonical input and output

Input messages: `[{"role":"system","content":"You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},{"role":"user","content":"Received: YYYY-MM-DD (weekday)\nSource: ...\n\n<ticket>"}]`.

The model's own chat template formats exactly these messages. No additional system text, schema, examples, injected metadata or normalized text at inference.

Output must be one raw JSON object with **exactly** these keys and types:

```json
{"category":"refund","priority":"P2","amount_paise":2936300,"txn_date":"2026-09-13","txn_id":"NP5972758064","channel":"netbanking","language":"en","needs_human":false}
```

| Key | Type | Values |
|---|---|---|
| category | string | payment_failed/refund/fraud/account_access/kyc/offers/other |
| priority | string | P1/P2/P3 |
| amount_paise | integer or null | exact paise; no float and no numeric string |
| txn_date | ISO date or null | YYYY-MM-DD |
| txn_id | string or null | `NP` + exactly 10 digits |
| channel | string or null | upi/card/netbanking/wallet |
| language | string | en/hi-en |
| needs_human | boolean | true/false |

## 3. Functional invariants — all mandatory

### SRS-F01 Category

`fraud` wins if the customer reports a transaction they did not make/approve/recognize. Otherwise map failed or stuck intended payment to `payment_failed`; money owed back or duplicate charge to `refund`; access/reset/OTP/block to `account_access`; identity/Aadhaar/PAN to `kyc`; offers/rewards/referrals to `offers`; other requests to `other`. When intent evidence is ambiguous, a training-data auditor flags the row for review, rather than inventing a categorical fix.

### SRS-F02 Priority

Ordered precedence: P1 when fraud OR disputed amount >= 5,000,000 paise OR legal threat (RBI, ombudsman, consumer court, lawyer, legal notice/action). Else P2 when (`refund`/`payment_failed` and amount >=500,000 paise) OR `account_access`. Else P3. `URGENT!!` by itself is not a legal threat and does not set P1.

### SRS-F03 Transaction nullability

For `account_access`, `kyc`, and `other`, `amount_paise`, `txn_date`, `txn_id`, `channel` are `null` even if distractor mentions occur. For other categories, extract only the specific relevant transaction or claim.

### SRS-F04 Amount

Use exact decimal arithmetic when repairing train labels. E.g. ₹6,500 => 650,000 paise; ₹49,999.99 => 4,999,999; ₹50,000 => 5,000,000; 1.3k => ₹1,300 => 130,000 paise; 1.5 lakh => ₹150,000 => 15,000,000 paise. Exclude balances/fees/limits. If the ticket supplies no disputed amount, use null. If multiple plausible amounts cannot be resolved confidently, do not silently rewrite a training label.

### SRS-F05 Transaction dates

Anchor relative dates to `Received`, not execution date. Yesterday/`kal` in a past-transaction context => -1 day; day before yesterday/`parso` => -2 days; `N days ago`/`N din pehle` => -N. For yearless month/day, choose the most recent such date on/before receipt, including previous year. Ignore earlier email quoted lines beginning with `>` as sources of transaction date. Keep source text unchanged for test inference.

### SRS-F06 Transaction ID

Normalize only a NimbusPay identifier to `NP` followed by ten digits. In training repairs, uppercase and remove separators (`np 12345 67890` => `NP1234567890`) where evidence is certain. Do not reinterpret bank refs, ticket IDs or order IDs as a transaction ID.

### SRS-F07 Channel

Only explicit UPI, card/credit/debit/Visa/Mastercard, netbanking/internet banking, or wallet gives a non-null channel. Device type, ticket `Source`, and bank account mention do not imply payment channel.

### SRS-F08 Language

`hi-en` when customer content is Hinglish in Latin script, potentially mixed with English. Else `en`. Do not infer Hindi from sender name, region or quoted boilerplate alone.

### SRS-F09 Human escalation

`needs_human=true` for all P1 records or explicit prior contact on same issue (third time, repeatedly, pehle bhi complaint, baar baar, etc.). Prior ticket number alone is not always proof of a previous complaint about the *same issue*; follow semantic evidence. Mere urgency without contact/history/P1 is false.

### SRS-F10 Output invariants

- JSON top level is object, keys exactly eight, no markdown/analysis/prefix/suffix.
- Types match the scorer exactly (`true` is not `1`; numeric strings fail).
- `null` is JSON null; booleans lowercase.
- Decoder output is not modified except whitespace strip.
- All model inference steps use only a single allowed base+adapter; no post-hoc repair.

## 4. Data pipeline requirements

| ID | Requirement | Test evidence |
|---|---|---|
| SRS-D01 | Raw archive and JSONL remain immutable | Source checksum unchanged |
| SRS-D02 | Each row has stable unique id and correct message roles | Validation report and tests |
| SRS-D03 | Every parsed label is schema-checked | Per-error counts, row ids |
| SRS-D04 | All train corrections use train text and schema only | Provenance audit; no dev labels in cleaner |
| SRS-D05 | For each dropped/fixed row, log `id,action,reason` | Set equality: changed IDs vs log IDs |
| SRS-D06 | Duplicate handling deterministic | Ordered occurrence, idempotence |
| SRS-D07 | No ambiguous label auto-repair | Quarantine list with reasons |
| SRS-D08 | No labels/predictions from other models or APIs | Dependency and call audit |
| SRS-D09 | Train/dev/test roles strictly separated | Split manifests and tests |
| SRS-D10 | Cleaned outputs stay local and excluded from GitHub | `.gitignore`, staged-files scan |

## 5. Training requirements

| ID | Requirement | Evidence |
|---|---|---|
| SRS-T01 | Open-weight model <=4B | Model card + actual parameter count |
| SRS-T02 | Only LoRA or QLoRA; <=50M trainable | Programmatic count and list of trainable modules |
| SRS-T03 | T4 compatible | Runtime GPU report and measured pilot |
| SRS-T04 | Correct model-native chat template | Rendered sample and token checks |
| SRS-T05 | Loss only on assistant response and end token | Mask audit, nonzero targets, no prompt leakage |
| SRS-T06 | Sequence limit justified by actual tokenizer quantiles | Histogram, coverage, truncation assertions |
| SRS-T07 | Main hyperparameters and tokenizer/revision pinned | Config + manifest |
| SRS-T08 | Four controlled one-factor comparisons | Side-by-side table with all required metrics |
| SRS-T09 | Baselines correctly isolate prompt rules | Exact original/fixed prompt and rule-prompt baseline |
| SRS-T10 | Final adapter in standard PEFT format | Fresh load in new process; safetensors |

## 6. Inference and scoring requirements

`generate` contract: `do_sample=False, num_beams=1, max_new_tokens=200`; decode **newly generated** token IDs only. Native EOS handling may terminate early. Only `.strip()` on text. The model and adapter must be reloaded in the documented precision; no temperature sampling, reranking, self-correction or retry strategy.

Dev predicted file: one line `{"id":"dv-00001","output":"<raw model text>"}` per dev row. Test file: 400 lines with corresponding `ts-` IDs. No duplicates, unknown IDs, missing IDs or order mismatch. Validate the file's envelope without modifying `output`.

Official scorer results must be generated unmodified from supplied `score.py`. Additional analyses cannot replace the official score. Record the overlap-excluded diagnostic view separately.

## 7. Nonfunctional requirements

| ID | Requirement |
|---|---|
| SRS-N01 | Fresh Colab T4 notebook top-to-bottom <3h including final training and test output |
| SRS-N02 | Memory-safe: reset/record GPU peaks per run, catch OOM without corrupting prior artifacts |
| SRS-N03 | Deterministic: config/seed/input checksum/model revision/versions/run ID recorded |
| SRS-N04 | Idempotent preprocessing: reruns yield byte-equivalent cleaned data and logs |
| SRS-N05 | Atomic checkpoint + prediction writes; no partial file considered successful |
| SRS-N06 | Safe restart: an interrupted run never mislabels a stale artifact as complete |
| SRS-N07 | No secrets in notebook, logs, console or Git |
| SRS-N08 | Clean report <=3 pages; all measured claims traceable to logs |
| SRS-N09 | Adapter artifact remains below GitHub file limit or uses Git LFS properly |
| SRS-N10 | Repo private until deadline, no post-deadline pushes, public within 1h after |

## 8. Change control

Any modification to the user-facing model prompt, decoder, model architecture or output format needs re-evaluation against the assignment's strict contract. Changes to offline audit rules need new cleaning log, checksum, regression run and report update. Never "fix" a hidden-test defect by special-casing a test row.

