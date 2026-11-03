# Technical Report: High-Assurance PEFT Triage for NimbusPay Support Tickets
**Candidate:** Shoryakumar Mishra | **Role:** Machine Learning Intern – LLM Fine-Tuning | **Date:** November 2026

---

## 1. Dataset Audit, Taxonomy & Data Curation (Part A — 25 Points)

### 1.1 Empirical Audit Findings
An independent, full-corpus programmatic audit of `candidate_pack/data/train.jsonl` (1,445 rows) identified systematic corruption across 7 distinct failure modes:

| Corruption Class | Rows | Root Cause / Violation of `SCHEMA.md` | Resolution Policy (`action`) |
|---|---:|---|---|
| **Truncated / Malformed JSON** | 25 | Assistant payload truncated mid-string; unrecoverable syntax | `drop` (unrecoverable without guessing) |
| **Meaningless Ticket Body** | 10 | Tickets containing solely `"test"`, `"asdf asdf"`, or blank bodies | `drop` (zero domain signal) |
| **Internal Train Duplicates** | 61 | Exact `(user, assistant)` text pairs repeated in `train.jsonl` | `drop` (keep 1st occurrence, drop duplicates) |
| **Legacy `prio` Schema** | 70 | Deprecated v1 schema using key `"prio"` and `DD/MM/YYYY` dates | `fix` (migrate key, reformat ISO date, recompute) |
| **Noncanonical Transaction IDs** | 20 | Valid `NP` IDs containing hyphens, spaces, or lowercase letters | `fix` (canonicalize to `NP\d{10}`) |
| **Rupee-Paise Scaling Errors** | 16 | Integer amount recorded in rupees rather than paise | `fix` (multiply by 100 to paise, recalculate prio) |
| **P1 Invariant Violations** | 30 | Records labeled `priority="P1"` but `needs_human=false` | `fix` (enforce `needs_human=true` invariant) |
| **Priority Boundary Errors** | 65 | Mismatch against strictly ordered `SCHEMA.md` priority rules | `fix` (recalculate priority from category & amount) |

### 1.2 Data Curation & `cleaning_log.csv` Reconciliation
- **Total Input Rows:** 1,445
- **Dropped Rows:** 96 (25 malformed + 10 meaningless + 61 internal duplicate occurrences)
- **Cleaned Dataset Size (`cleaned.jsonl`):** 1,349 rows ($1{,}445 - 96 = 1{,}349$)
- **Repaired Rows:** 166 rows modified in place across legacy schema, date formatting, and invariants
- **Mutation Ledger:** `cleaning_log.csv` records exactly 262 rows ($96\text{ drops} + 166\text{ fixes}$), with unique IDs and compound reason logging (e.g., `migrate_legacy_prio_schema; convert_date_dd_mm_yyyy_to_iso; recalculate_priority_low_to_P3`). Re-running the cleaner yields 100% byte-identical hashes (proven idempotency).

### 1.3 Split Hygiene & Cross-Split Overlap Disclosure
A cross-split fingerprint scan revealed that **32 tickets in `train.jsonl` appear verbatim in `dev.jsonl`** (16.0% of dev rows) with identical labels.
- **Compliance Boundary:** In strict adherence to `ASSIGNMENT.md` Rule 5 (*"uses only train.jsonl and SCHEMA.md"*), dev fingerprints were **NOT** used to filter or manipulate training data.
- **Reporting Standard:** To guarantee scientific transparency, we disclose both the **Official Dev Score** (full 200 rows) and an **Overlap-Free Diagnostic Score** (168 rows) across all evaluations.

---

## 2. Modeling Rationale, Baselines & Controlled Experiments (Parts B, C, D — 50 Points)

### 2.1 Model Selection & Architecture
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`). Selected for its native ChatML chat template, strong multilingual / Hinglish comprehension, and compact 1.54B parameter footprint that fits within free Google Colab T4 (16 GB VRAM).
- **PEFT Adaptation:** LoRA applied to all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). For $r=8, \alpha=16$, trainable parameters total **9,232,384** (0.59% of base weights), well within the 50,000,000 parameter ceiling.
- **Assistant-Only Loss Masking:** Formatted via ChatML (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n{json}<|im_end|>\n`). Numerically verified via unit test: all prompt tokens receive label `-100`, while the assistant JSON and `<|im_end|>` token retain true targets.
- **Sequence Length Evidence:** Token distribution profiling measured prompt lengths at median 103, 95th-percentile 147, and maximum 2,145 tokens (quoted emails). Target JSON lengths span 51 to 85 tokens. Setting `max_seq_length = 2304` prevents any target truncation.

### 2.2 Controlled Experiment Matrix
All training trials were conducted on a single T4 GPU with effective batch size 8 (microbatch 1, gradient accumulation 8), cosine learning rate schedule ($2\times 10^{-4}$), 5% warmup, and 2 epochs.

| Run | Hypothesis / Change | Data | Precision | Rank | Steps | Dev Exact | Dev Mean Field | Peak VRAM | Train Min | Diagnostic Exact (168 rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| **B1** | Zero-shot Base (fixed prompt) | — | FP16 Base | 0 | 0 | **0.0%** | 9.2% | 3.12 GiB | 0.0 | 0.0% |
| **B2** | Base + Rules in Prompt (ref) | — | FP16 Base | 0 | 0 | **17.5%** | 51.4% | 3.18 GiB | 0.0 | 17.3% |
| **E0** | Reference QLoRA | Cleaned | 4-bit QLoRA | 8 | 338 | **78.5%** | 92.1% | 4.21 GiB | 38.4 | 75.0% |
| **E1** | Raw Training Data Ablation | Raw | 4-bit QLoRA | 8 | 362 | **62.0%** | 84.3% | 4.22 GiB | 41.2 | 58.9% |
| **E2** | FP16 LoRA Precision Ablation | Cleaned | FP16 LoRA | 8 | 338 | **79.0%** | 92.4% | 7.85 GiB | 34.6 | 75.6% |
| **E3** | Rank 16 Capacity Ablation | Cleaned | 4-bit QLoRA | 16 | 338 | **77.5%** | 91.8% | 4.48 GiB | 40.8 | 73.8% |
| **E4** | Original Idea: Hardcase Stratified | Cleaned | 4-bit QLoRA | 8 | 338 | **80.5%** | 92.0% | 4.24 GiB | 39.5 | 77.4% |

### 2.3 Scientific Findings
1. **Data Cleaning Impact (E0 vs E1):** Cleaning noisy labels produced a **+16.5% exact match surge** ($62.0\% \to 78.5\%$). Training on uncleaned data severely confused schema keys (`prio` vs `priority`) and rupee scaling.
2. **Quantization Impact (E0 vs E2):** 4-bit NF4 QLoRA matched FP16 LoRA within 0.5% exact match ($78.5\%$ vs $79.0\%$) while cutting peak memory by **46.4%** ($4.21\text{ GiB}$ vs $7.85\text{ GiB}$), validating QLoRA as the optimal configuration for T4 efficiency.
3. **Rank Capacity (E0 vs E3):** Increasing rank to 16 doubled parameter count to 18.3M but yielded no performance gain ($-1.0\%$ exact match), indicating rank 8 is fully sufficient and protects against overfitting on 1.3k records.
4. **Original Idea (E4):** Stratified oversampling of long-context quoted chains, Hinglish tickets, and threshold boundary cases achieved **80.5% dev exact match** (+2.0% over E0) and improved hard-slice generalization.

---

## 3. Error Analysis, Case Studies & Research Roadmap (Part E — 25 Points)

### 3.1 Weakest Fields & Failure Modes
On the best fine-tuned model (E4, 80.5% dev exact match), field accuracy breakdown is: `category` (96.5%), `language` (96.0%), `needs_human` (94.0%), `channel` (93.5%), `priority` (91.5%), `txn_id` (91.0%), `amount_paise` (89.5%), and `txn_date` (84.0%). The primary failure mechanisms are relative calendar offsets and distractors.

### 3.2 Ten Representative Failure Case Studies
| ID | Error Type | Short Evidence Span | Predicted vs Gold | Root Cause Mechanism | Controlled Offline Fix |
|---|---|---|---|---|---|
| **dv-00006** | `date_error` | "App shows fraudulent payment... day before yesterday" | `txn_date: 2026-08-15` vs `2026-06-30` | Relative date calendar shift failed across month boundary | Train on synthetic calendar shift tuples |
| **dv-00013** | `amount_error` | "Why was my card declined... balance Rs. 99,999" | `amount_paise: 9999900` vs `5500` | Extracted account balance distractor instead of transaction | Hard negative mining with explicit balance phrases |
| **dv-00025** | `priority_error` | "Did not get cashback promised... ₹24" | `priority: P2` vs `P3` | Over-escalated offers ticket to P2 despite amount < ₹5k | Re-weight priority boundary pairs near thresholds |
| **dv-00038** | `language_error`| "Registered email id kaise change karun? App 5.0.3" | `language: en` vs `hi-en` | Latin-script Hinglish confused by technical English nouns | Augment training corpus with Hinglish code-switching |
| **dv-00049** | `category_error`| "The refund has not appeared in my account... 27444 rs" | `category: payment_failed` vs `refund` | Semantic ambiguity between failed credit and delayed refund | Contrastive pairs isolating merchant reversal cues |
| **dv-00064** | `null_leakage` | "Difficulties to verify identity... transferred 4k" | `amount_paise: 400000` vs `null` | Leaked amount into `kyc` non-transaction category | Masking supervision enforcing null invariants |
| **dv-00082** | `needs_human` | "Why couldn't I use card at store? Amount Rs 251" | `needs_human: true` vs `false` | False positive on urgency keywords without repeat contact | Negative pairs contrasting urgency with repeat contact |
| **dv-00100** | `quote_distract`| "Double charged... > On Aug 3 support wrote..." | `txn_date: 2026-08-03` vs `2026-08-22` | Extracted date from quoted `>` email chain | Explicit synthetic email threads with distractor headers |
| **dv-00116** | `id_hallucinate`| "Did not get cashback... Ticket #29322" | `txn_id: NP1234567890` vs `null` | Converted internal ticket reference to NimbusPay ID | Explicit negative examples of ticket and bank refs |
| **dv-00143** | `unit_conversion`| "Charged twice... amount was 1.3 lakh" | `amount: 130000, prio: P3` vs `13000000, P1` | Interpreted lakh as thousands, cascading to P3 priority | Numerical curriculum focusing on `k` and `lakh` units |

### 3.3 One-Week Research & Engineering Roadmap
1. **Days 1–2: Invariant-Supervised Multitask Auxiliary Head.** Introduce offline token-level auxiliary classification losses for currency unit scale (`k` vs `lakh` vs plain) to eliminate unit conversion cascades.
2. **Days 3–4: Counterfactual Invariance Curation.** Systematically perturb clean tickets with metamorphic mutations (varying receipt dates, adding quoted email headers, altering distractor balances) to harden representation invariance.
3. **Days 5–6: Direct Preference Optimization (DPO) for Schema Strictness.** Pair correct triage JSON completions with common edge-case failure completions (null leakage, boundary priority errors) using DPO on the frozen adapter.
4. **Day 7: Multi-Seed Cross-Architecture Benchmark.** Evaluate across 5 random seeds on Qwen2.5-1.5B, Llama-3.2-1B, and SmolLM2-1.7B with paired Wilcoxon signed-rank significance testing.

---

## 4. AI Tool Disclosure & Verification Attestation
- **AI Coding Assistant Disclosure:** An AI coding assistant was used for writing boilerplate tests, modular file scaffolding, regex profiling, and data formatting.
- **Verification Responsibility:** The candidate independently verified every line of code, audited all dataset mutations, executed all tests locally, and validated model parameter counts and metric provenance against raw artifacts. No external model or API was used to generate training targets or test predictions.

