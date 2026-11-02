"""Controlled experiment definitions and result management module.
Defines the 5-run matrix (E0-E4) and records verified metrics for the
single consolidated experiment table.
"""

from typing import Dict, Any, List, Optional
import os
import json
import csv
from dataclasses import dataclass, asdict


@dataclass
class ExperimentResult:
    run_id: str
    description: str
    data_source: str  # 'cleaned' or 'raw'
    precision: str    # '4-bit QLoRA' or 'FP16 LoRA'
    lora_rank: int
    seed: int
    optimizer_steps: int
    dev_exact_match: float
    dev_mean_field: float
    gpu_peak_gib: float
    training_minutes: float
    status: str       # 'COMPLETED', 'FAILED', 'BLOCKED'
    notes: str
    diag_exact_match: Optional[float] = None
    diag_mean_field: Optional[float] = None


EXPERIMENT_CONFIGS = {
    "baseline_fixed": {
        "run_id": "B1",
        "description": "Base model zero-shot (fixed prompt, untouched)",
        "data_source": "none",
        "precision": "FP16 Base",
        "lora_rank": 0,
        "seed": 42,
    },
    "baseline_rules": {
        "run_id": "B2",
        "description": "Base model with full rules in prompt (reference point)",
        "data_source": "none",
        "precision": "FP16 Base",
        "lora_rank": 0,
        "seed": 42,
    },
    "E0_reference": {
        "run_id": "E0",
        "description": "Reference Run: Cleaned data, 4-bit QLoRA, rank 8",
        "data_source": "cleaned",
        "precision": "4-bit QLoRA",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 2,
        "seed": 42,
        "sampling": "uniform",
    },
    "E1_raw": {
        "run_id": "E1",
        "description": "Raw Data: Same QLoRA r8 on uncleaned noisy train.jsonl",
        "data_source": "raw",
        "precision": "4-bit QLoRA",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 2,
        "seed": 42,
        "sampling": "uniform",
    },
    "E2_fp16": {
        "run_id": "E2",
        "description": "FP16 Precision: Cleaned data, FP16 LoRA, rank 8",
        "data_source": "cleaned",
        "precision": "FP16 LoRA",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 2,
        "seed": 42,
        "sampling": "uniform",
    },
    "E3_rank16": {
        "run_id": "E3",
        "description": "Rank Ablation: Cleaned data, 4-bit QLoRA, rank 16",
        "data_source": "cleaned",
        "precision": "4-bit QLoRA",
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 2,
        "seed": 42,
        "sampling": "uniform",
    },
    "E4_hardcase": {
        "run_id": "E4",
        "description": "Original Idea: Cleaned data, 4-bit QLoRA r8, hard-case oversampling",
        "data_source": "cleaned",
        "precision": "4-bit QLoRA",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 2,
        "seed": 42,
        "sampling": "hardcase_stratified",
    },
}


def save_experiment_table_csv(results: List[ExperimentResult], out_path: str):
    """Save experiment table to CSV format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "description",
            "data_source",
            "precision",
            "lora_rank",
            "seed",
            "optimizer_steps",
            "dev_exact_match",
            "dev_mean_field",
            "gpu_peak_gib",
            "training_minutes",
            "status",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    os.replace(tmp_path, out_path)


def format_markdown_table(results: List[ExperimentResult]) -> str:
    """Format experiment results into a GitHub-style markdown table."""
    lines = [
        "| Run | Description | Data | Precision | Rank | Steps | Dev Exact | Dev Mean Field | Peak VRAM | Train Min | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        exact_str = f"{r.dev_exact_match:.1%}" if r.dev_exact_match is not None else "N/A"
        field_str = f"{r.dev_mean_field:.1%}" if r.dev_mean_field is not None else "N/A"
        vram_str = f"{r.gpu_peak_gib:.2f} GiB" if r.gpu_peak_gib > 0 else "N/A"
        min_str = f"{r.training_minutes:.1f}" if r.training_minutes > 0 else "0.0"
        lines.append(
            f"| **{r.run_id}** | {r.description} | {r.data_source} | {r.precision} | {r.lora_rank} | {r.optimizer_steps} | {exact_str} | {field_str} | {vram_str} | {min_str} | {r.notes} |"
        )
    return "\n".join(lines)
