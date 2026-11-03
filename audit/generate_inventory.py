import json
import os
import hashlib
import platform
import subprocess
import shutil
import safetensors.torch
import torch

def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

inventory = {
    "timestamp": "2026-11-02T17:32:00+05:30",
    "verdict": "NOT READY",
    "git": {},
    "environment": {},
    "file_hashes": {},
    "forensic_audit_findings": {},
    "notebook_cell_audit": {},
    "safetensors_audit": {},
    "split_counts": {}
}

# 1. Git info
try:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    remotes = subprocess.check_output(["git", "remote", "-v"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    inventory["git"] = {
        "head": head,
        "branch": branch,
        "remotes": remotes if remotes else "NONE",
        "has_uncommitted_changes": bool(status)
    }
except Exception as e:
    inventory["git"] = {"error": str(e)}

# 2. Environment
disk = shutil.disk_usage(".")
inventory["environment"] = {
    "os": platform.platform(),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "colab_t4_attached": False,
    "hardware_note": "Local workstation running Windows with CPU-only PyTorch; no attached Google Colab T4 environment",
    "disk_free_gb": round(disk.free / (1024**3), 2)
}

# 3. File Hashes
paths = [
    "candidate_pack/SCHEMA.md",
    "candidate_pack/data/train.jsonl",
    "candidate_pack/data/dev.jsonl",
    "candidate_pack/data/test_inputs.jsonl",
    "candidate_pack/score.py",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "notebook.ipynb",
    "predictions_test.jsonl",
    "cleaning_log.csv",
    "report.md",
    "README.md",
    "experiment_logs/summary.csv"
]
for p in paths:
    inventory["file_hashes"][p] = {
        "exists": os.path.exists(p),
        "sha256": sha256_file(p),
        "bytes": os.path.getsize(p) if os.path.exists(p) else 0
    }

# 4. Forensic Audit Findings
inventory["forensic_audit_findings"] = {
    "predictions_test_amount_collapse": {
        "description": "All 156 non-null amounts in predictions_test.jsonl equaled 202600 paise due to regex matching Received: 2026-...",
        "confirmed": True,
        "evidence": "Counter: {None: 244, 202600: 156}"
    },
    "synthetic_adapter_weights": {
        "description": "Adapter weights in adapter_model.safetensors were initialized with torch.randn * 0.01 and never optimized via backprop",
        "confirmed": True,
        "evidence": "Tensors: 392, Parameter count: 9,232,384, Mean of stds: 0.010000"
    },
    "unexecuted_training_pipeline": {
        "description": "src/triage/training.py did not call Trainer.train() or any PyTorch optimizer step",
        "confirmed": True
    },
    "notebook_hardcoded_literals": {
        "description": "notebook.ipynb contained hardcoded Python dictionaries for baselines and experiments rather than runtime execution results",
        "confirmed": True
    },
    "notebook_prediction_provenance_mismatch": {
        "description": "notebook.ipynb displayed ts-00001 with null amount and ts-00400 with 120000, while predictions_test.jsonl contained 202600 for both",
        "confirmed": True
    },
    "report_arithmetic_discrepancy": {
        "description": "report.md listed 8 field accuracies averaging 92.0% while reporting 93.2% mean field accuracy",
        "confirmed": True
    }
}

# 5. Notebook Audit
if os.path.exists("notebook.ipynb"):
    with open("notebook.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    inventory["notebook_cell_audit"] = {
        "total_cells": len(nb.get("cells", [])),
        "code_cells": len(code_cells),
        "training_cell_executed": False,
        "hardcoded_result_cells": [6, 7],
        "provenance_valid": False
    }

# 6. Safetensors Audit
if os.path.exists("adapter/adapter_model.safetensors"):
    tensors = safetensors.torch.load_file("adapter/adapter_model.safetensors")
    total_params = sum(t.numel() for t in tensors.values())
    stds = [t.float().std().item() for t in tensors.values()]
    inventory["safetensors_audit"] = {
        "tensor_count": len(tensors),
        "total_trainable_parameters": total_params,
        "is_under_50m_limit": total_params <= 50_000_000,
        "std_distribution_mean": round(float(torch.tensor(stds).mean().item()), 6),
        "is_synthetic_placeholder": True
    }

# 7. Split Counts
for split, sp in [("train", "candidate_pack/data/train.jsonl"), ("dev", "candidate_pack/data/dev.jsonl"), ("test", "candidate_pack/data/test_inputs.jsonl")]:
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        inventory["split_counts"][split] = len(lines)

with open("audit/recovery_inventory.json", "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2)

print("Saved audit/recovery_inventory.json")
print("Verdict:", inventory["verdict"])

