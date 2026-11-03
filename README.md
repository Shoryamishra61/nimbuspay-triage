# NimbusPay Ticket Triage — High-Assurance PEFT Pipeline

> **VIMA3YA Machine Learning Intern — LLM Fine-Tuning Take-Home Assessment**  
> **Candidate:** Shoryakumar Mishra | **Date:** November 2026

---

## Executive Summary & Best Result
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)
- **Quantization:** 4-bit NF4 (`bitsandbytes`) with FP16 compute
- **Adapter:** PEFT LoRA ($r=8, \alpha=16$, dropout $0.05$) on all 7 linear projections
- **Trainable Parameters:** **9,232,384** (0.59% of base weights; strictly $\le 50{,}000{,}000$)
- **Best Measured Result:** **80.5% Dev Exact Match** (93.2% Mean Field Accuracy) with 4-bit QLoRA r=8 and hard-case stratified sampling on cleaned data (**77.4% on 168-row Overlap-Free Diagnostic Set**).
- **Peak T4 VRAM:** **4.21 GiB** (comfortably fits on free Colab T4)
- **Top-to-Bottom Execution Time:** **< 45 minutes** for full training and inference

---

## Key Repository Deliverables
```text
nimbuspay-triage/
├── README.md                      # Model ID, quantization, best score, setup guide
├── notebook.ipynb                 # Fully executed Colab notebook (<3h on fresh T4)
├── report.md                      # 3-page comprehensive technical report
├── cleaning_log.csv               # 262 mutation rows (id,action,reason)
├── predictions_test.jsonl         # 400 raw test predictions (ts-00001 to ts-00400)
├── requirements.txt               # Pinned dependencies
├── adapter/
│   ├── adapter_config.json        # Standard PEFT configuration
│   └── adapter_model.safetensors  # Pinned LoRA weights (17.6 MB)
├── src/triage/                    # Modular, production-grade implementation
│   ├── contracts.py               # Canonical 8-key schema & priority invariants
│   ├── source_io.py               # Strict JSONL parsing, SHA256, split manifests
│   ├── evidence.py                # Offline Decimal arithmetic & regex extractors
│   ├── audit.py                   # 1,445-row independent data auditor
│   ├── clean.py                   # Deterministic idempotent data cleaner
│   ├── tokenizer_pipeline.py      # ChatML formatter & assistant-only loss masking
│   ├── training.py                # LoRA/QLoRA trainer & memory profiler
│   ├── generation.py              # Strict minimal inference (greedy, 200 tokens)
│   ├── scoring.py                 # Wrapper for official score.py
│   ├── experiments.py             # Controlled experiment matrix (E0-E4)
│   ├── probes.py                  # Read-only failure diagnostics & slice metrics
│   ├── release.py                 # 18 hard release verification gates
│   └── provenance.py              # Run manifests and runtime environment capture
├── configs/                       # Declarative experiment configs (E0-E4)
└── tests/                         # 28 automated unit, contract, and metamorphic tests
```

---

## 1. Data Audit & Curation Summary (Part A — 25 Points)
An exhaustive audit of `train.jsonl` (1,445 rows) uncovered systematic noise:
- **Dropped Rows (96 total):**
  - 25 malformed, truncated JSON labels (unrecoverable mid-string)
  - 10 meaningless ticket bodies (`"test"`, `"asdf asdf"`, empty)
  - 61 internal duplicate tickets (retaining first occurrence)
- **Repaired Rows (166 total):**
  - 70 legacy `prio` schema migrations with recalculated priority
  - 30 legacy `DD/MM/YYYY` dates converted to ISO `YYYY-MM-DD`
  - 20 unnormalized transaction IDs canonicalized to `NP\d{10}`
  - 16 rupee-paise scaling errors converted to integer paise
  - 30 `P1` invariant violations fixed to `needs_human=true`
  - 65 priority boundary corrections aligned to `SCHEMA.md`
- **Output:** `artifacts/cleaned.jsonl` (1,349 clean rows) and `cleaning_log.csv` (262 rows, unique IDs, compound reasons).

> **Cross-Split Overlap Disclosure:** 32 tickets in `train.jsonl` appear identically in `dev.jsonl`. Following `ASSIGNMENT.md` Rule 5, dev fingerprints were **NOT** used to filter training data. We publish both the 200-row official dev score and the 168-row overlap-free diagnostic score.

---

## 2. Controlled Experiment Matrix (Part D — 20 Points)
Measured with `candidate_pack/score.py` on dev:

| Run | Description | Data | Precision | Rank | Steps | Dev Exact | Dev Mean Field | Peak VRAM | Train Min | Diagnostic Exact |
|---|---|---|---|---|---|---|---|---|---|---|
| **B1** | Base model zero-shot (fixed prompt) | — | FP16 Base | 0 | 0 | **0.0%** | 9.2% | 3.12 GiB | 0.0 | 0.0% |
| **B2** | Base + Rules in Prompt (ref baseline) | — | FP16 Base | 0 | 0 | **17.5%** | 51.4% | 3.18 GiB | 0.0 | 17.3% |
| **E0** | Reference QLoRA | Cleaned | 4-bit QLoRA | 8 | 338 | **78.5%** | 92.1% | 4.21 GiB | 38.4 | 75.0% |
| **E1** | Raw Training Data Ablation | Raw | 4-bit QLoRA | 8 | 362 | **62.0%** | 84.3% | 4.22 GiB | 41.2 | 58.9% |
| **E2** | FP16 LoRA Precision Ablation | Cleaned | FP16 LoRA | 8 | 338 | **79.0%** | 92.4% | 7.85 GiB | 34.6 | 75.6% |
| **E3** | Rank 16 Capacity Ablation | Cleaned | 4-bit QLoRA | 16 | 338 | **77.5%** | 91.8% | 4.48 GiB | 40.8 | 73.8% |
| **E4** | Original Idea: Hardcase Stratified | Cleaned | 4-bit QLoRA | 8 | 338 | **80.5%** | 93.2% | 4.24 GiB | 39.5 | 77.4% |

**Key Takeaways:**
1. Data cleaning delivers a massive **+16.5% exact match surge** ($62.0\% \to 78.5\%$).
2. 4-bit QLoRA matches FP16 LoRA within 0.5% while reducing peak memory by **46.4%** ($4.21\text{ GiB}$ vs $7.85\text{ GiB}$).
3. Rank 8 is optimal; rank 16 doubles parameter footprint with no accuracy gain.
4. Hard-case stratified sampling (E4) achieves the highest exact match (**80.5%**).

---

## 3. Strict Inference Contract & Zero Post-Processing
In strict compliance with `ASSIGNMENT.md` Rule 9:
- **Prompt:** Exactly the supplied system and user messages rendered via model's native ChatML template.
- **Decoding:** Plain `model.generate(do_sample=False, num_beams=1, max_new_tokens=200)`.
- **Post-Processing:** Direct `.strip()` on newly decoded tokens.
- **Strictly Prohibited & Verified Absent:** Zero regex post-processing, zero JSON repair libraries, zero grammar/constrained decoding, zero retry prompts, and zero multi-model ensembles.

---

## 4. Reproducing on Google Colab (Free T4)
1. Open Google Colab and set Runtime to **Python 3 with T4 GPU**.
2. Upload this repository or open `notebook.ipynb`.
3. Run all cells top to bottom (`Runtime -> Run all`).
   - Total measured execution time: **~42 minutes** (well under the 3-hour limit).
   - Automatically executes data audit, cleaning, reference baseline, PEFT training, dev scoring, error analysis, and test prediction generation.

---

## 5. Local Automated Testing
To run the full suite of 28 unit, contract, metamorphic, and release tests:
```bash
python -m pytest -o pythonpath=. tests/ -v
```

---

## 6. Pre-Submission Release Gates
Verified via `src/triage/release.py`:
- [x] All 7 required files present and verified authentic
- [x] Pinned base model `Qwen/Qwen2.5-1.5B-Instruct` ($\le 4\text{B}$)
- [x] PEFT LoRA adapter $\le 50\text{M}$ trainable parameters (9,232,384 params)
- [x] `predictions_test.jsonl` contains exactly 400 lines matching unique `ts-` IDs
- [x] `cleaning_log.csv` contains exactly 262 entries reconciling $1{,}445 = 1{,}349 + 96$
- [x] Free T4 compatible peak memory (4.21 GiB) and runtime (<45 min)
- [x] Zero training data, base model weights, or private tokens committed to Git

