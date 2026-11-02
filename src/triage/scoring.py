"""Scoring module wrapping candidate_pack/score.py strictly and unmodified.
Computes both official 200-row dev score and 168-row overlap-free diagnostic score.
"""

from typing import List, Dict, Any, Optional
import json
import os
import subprocess
import sys

# Import scoring functions directly from official scorer
scorer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "candidate_pack", "score.py"))


def run_official_score(gold_path: str, pred_path: str) -> Dict[str, Any]:
    """Run candidate_pack/score.py as a subprocess with --json flag."""
    cmd = [sys.executable, scorer_path, "--gold", gold_path, "--pred", pred_path, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def evaluate_predictions(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    overlap_train_inputs: Optional[set] = None,
) -> Dict[str, Any]:
    """Evaluate predictions against gold rows with full official metrics,
    plus split diagnostic (official full set vs overlap-free diagnostic set).
    """
    from collections import defaultdict

    KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

    def parse(text):
        try:
            obj = json.loads(text.strip())
        except (json.JSONDecodeError, AttributeError):
            return None
        return obj if isinstance(obj, dict) else None

    def same(a, b):
        return type(a) is type(b) and a == b

    preds = {p["id"]: p.get("output") for p in pred_rows}

    # Evaluate all gold rows
    total_n = len(gold_rows)
    valid_cnt = 0
    exact_cnt = 0
    field_hits = defaultdict(int)

    # Overlap-free subset tracking
    diag_n = 0
    diag_valid = 0
    diag_exact = 0
    diag_field_hits = defaultdict(int)

    for g in gold_rows:
        gold = json.loads(g["messages"][-1]["content"])
        raw_pred = preds.get(g["id"])
        pred = parse(raw_pred)

        is_valid = (pred is not None)
        valid_cnt += int(is_valid)

        hits = {k: is_valid and k in pred and same(pred[k], gold[k]) for k in KEYS}
        exact = is_valid and all(hits.values()) and set(pred) == set(KEYS)
        exact_cnt += int(exact)

        for k in KEYS:
            field_hits[k] += int(hits[k])

        # Check overlap diagnostic
        u_text = g["messages"][1]["content"].strip()
        is_overlap = (overlap_train_inputs is not None and u_text in overlap_train_inputs)

        if not is_overlap:
            diag_n += 1
            diag_valid += int(is_valid)
            diag_exact += int(exact)
            for k in KEYS:
                diag_field_hits[k] += int(hits[k])

    official_fields = {k: field_hits[k] / total_n for k in KEYS}
    official_mean_field = sum(official_fields.values()) / len(KEYS)

    res = {
        "official": {
            "n": total_n,
            "json_valid": valid_cnt / total_n,
            "exact_match": exact_cnt / total_n,
            "mean_field_acc": official_mean_field,
            "fields": official_fields,
        }
    }

    if diag_n > 0:
        diag_fields = {k: diag_field_hits[k] / diag_n for k in KEYS}
        diag_mean_field = sum(diag_fields.values()) / len(KEYS)
        res["overlap_free_diagnostic"] = {
            "n": diag_n,
            "json_valid": diag_valid / diag_n,
            "exact_match": diag_exact / diag_n,
            "mean_field_acc": diag_mean_field,
            "fields": diag_fields,
        }

    return res
