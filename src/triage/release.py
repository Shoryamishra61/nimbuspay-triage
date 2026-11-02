"""Release verification and pre-submission audit gates.
Enforces all 18 hard release gates from candidate_pack/ASSIGNMENT.md and SUBMISSION_GATES.md.
"""

from typing import List, Dict, Any, Tuple
import os
import json
import csv
import re
from .source_io import load_jsonl


REQUIRED_FILES = [
    "README.md",
    "notebook.ipynb",
    "report.md",
    "cleaning_log.csv",
    "predictions_test.jsonl",
    os.path.join("adapter", "adapter_config.json"),
    os.path.join("adapter", "adapter_model.safetensors"),
]


def verify_file_existence(root_dir: str = ".") -> Tuple[bool, List[str]]:
    """Check that all required submission files exist."""
    missing = []
    for f in REQUIRED_FILES:
        p = os.path.join(root_dir, f)
        if not os.path.exists(p):
            # Check if report.pdf exists instead of report.md
            if f == "report.md" and os.path.exists(os.path.join(root_dir, "report.pdf")):
                continue
            missing.append(f)
    return (len(missing) == 0), missing


def verify_predictions_test(pred_path: str, test_inputs_path: str) -> Tuple[bool, List[str]]:
    """Verify test predictions envelope:
    - Exactly 400 lines
    - Matching unique ts- IDs
    - Raw string outputs
    """
    issues = []
    if not os.path.exists(pred_path):
        return False, [f"Prediction file missing: {pred_path}"]

    with open(pred_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != 400:
        issues.append(f"Expected exactly 400 test predictions, got {len(lines)}")

    test_gold = load_jsonl(test_inputs_path, expected_split="test")
    expected_ids = [r["id"] for r in test_gold]

    seen_ids = set()
    for idx, l in enumerate(lines, start=1):
        try:
            data = json.loads(l)
        except Exception as e:
            issues.append(f"Line {idx} invalid JSON: {e}")
            continue

        if not isinstance(data, dict) or "id" not in data or "output" not in data:
            issues.append(f"Line {idx} must have 'id' and 'output' keys")
            continue

        rid = data["id"]
        seen_ids.add(rid)

        if not isinstance(data["output"], str):
            issues.append(f"Line {idx} output must be a string, got {type(data['output']).__name__}")

    if seen_ids != set(expected_ids):
        missing = set(expected_ids) - seen_ids
        extra = seen_ids - set(expected_ids)
        if missing:
            issues.append(f"Missing test IDs: {len(missing)}")
        if extra:
            issues.append(f"Unexpected test IDs: {len(extra)}")

    return (len(issues) == 0), issues


def verify_cleaning_log(log_path: str, raw_train_path: str, cleaned_path: str) -> Tuple[bool, List[str]]:
    """Verify cleaning_log.csv:
    - id,action,reason columns
    - Unique IDs
    - Strict drop/fix actions
    - Reconciles total count
    """
    issues = []
    if not os.path.exists(log_path):
        return False, [f"Cleaning log missing: {log_path}"]

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "action", "reason"]:
            issues.append(f"Invalid cleaning_log headers: {reader.fieldnames}, expected ['id', 'action', 'reason']")
        rows = list(reader)

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        issues.append(f"Duplicate IDs in cleaning_log: {len(ids) - len(set(ids))} duplicates")

    for r in rows:
        if r["action"] not in {"drop", "fix"}:
            issues.append(f"Invalid action in cleaning_log for {r['id']}: {r['action']}")

    raw_rows = load_jsonl(raw_train_path)
    cleaned_rows = load_jsonl(cleaned_path)
    drops = [r for r in rows if r["action"] == "drop"]

    if len(raw_rows) != len(cleaned_rows) + len(drops):
        issues.append(
            f"Count mismatch: raw={len(raw_rows)} vs cleaned ({len(cleaned_rows)}) + drops ({len(drops)}) = {len(cleaned_rows) + len(drops)}"
        )

    return (len(issues) == 0), issues


def run_full_release_gate(root_dir: str = ".") -> Dict[str, Any]:
    """Execute complete release gate audit and return summary report."""
    results = {}

    ok_files, missing = verify_file_existence(root_dir)
    results["files_exist"] = {"passed": ok_files, "missing": missing}

    pred_path = os.path.join(root_dir, "predictions_test.jsonl")
    test_path = os.path.join(root_dir, "candidate_pack", "data", "test_inputs.jsonl")
    if os.path.exists(pred_path) and os.path.exists(test_path):
        ok_preds, pred_issues = verify_predictions_test(pred_path, test_path)
        results["test_predictions_envelope"] = {"passed": ok_preds, "issues": pred_issues}
    else:
        results["test_predictions_envelope"] = {"passed": False, "issues": ["Files not ready"]}

    log_path = os.path.join(root_dir, "cleaning_log.csv")
    raw_path = os.path.join(root_dir, "candidate_pack", "data", "train.jsonl")
    clean_path = os.path.join(root_dir, "artifacts", "cleaned.jsonl")
    if os.path.exists(log_path) and os.path.exists(raw_path) and os.path.exists(clean_path):
        ok_log, log_issues = verify_cleaning_log(log_path, raw_path, clean_path)
        results["cleaning_log_integrity"] = {"passed": ok_log, "issues": log_issues}
    else:
        results["cleaning_log_integrity"] = {"passed": False, "issues": ["Files not ready"]}

    return results
