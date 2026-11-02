"""Read-only model probes, slice diagnostics, and error analysis module.
OBSERVER ONLY — NEVER MODIFIES OUTPUT STRINGS.

Analyzes raw model outputs against gold dev labels to identify:
- Weakest fields and categories
- Confusion matrices
- Hard slice performance (long context, Hinglish, P1 fraud/disputes)
- Formats 10 specific failure case studies for report and notebook.
"""

from typing import List, Dict, Any, Optional
import json
from collections import defaultdict


def analyze_slice_performance(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculate accuracy metrics across critical operational slices."""
    preds = {p["id"]: p.get("output") for p in pred_rows}
    KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

    def parse(t):
        try:
            return json.loads(t.strip()) if t else None
        except:
            return None

    slices = {
        "overall": [],
        "hinglish": [],
        "long_quoted": [],
        "p1_critical": [],
        "non_txn_categories": [],
    }

    for g in gold_rows:
        rid = g["id"]
        gold = json.loads(g["messages"][-1]["content"])
        u_text = g["messages"][1]["content"]

        slices["overall"].append(g)
        if gold.get("language") == "hi-en":
            slices["hinglish"].append(g)
        if any(line.strip().startswith(">") for line in u_text.splitlines()) or len(u_text) > 1000:
            slices["long_quoted"].append(g)
        if gold.get("priority") == "P1":
            slices["p1_critical"].append(g)
        if gold.get("category") in {"account_access", "kyc", "other"}:
            slices["non_txn_categories"].append(g)

    slice_metrics = {}
    for s_name, s_rows in slices.items():
        if not s_rows:
            continue
        n = len(s_rows)
        valid = 0
        exact = 0
        field_hits = defaultdict(int)

        for g in s_rows:
            gold = json.loads(g["messages"][-1]["content"])
            pred = parse(preds.get(g["id"]))
            if pred and isinstance(pred, dict):
                valid += 1
                hits = {k: k in pred and type(pred[k]) is type(gold[k]) and pred[k] == gold[k] for k in KEYS}
                if all(hits.values()) and set(pred) == set(KEYS):
                    exact += 1
                for k in KEYS:
                    field_hits[k] += int(hits[k])

        slice_metrics[s_name] = {
            "n": n,
            "json_valid": valid / n,
            "exact_match": exact / n,
            "mean_field": sum(field_hits.values()) / (n * len(KEYS)),
        }

    return slice_metrics


def extract_top_failure_cases(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    max_cases: int = 10,
) -> List[Dict[str, Any]]:
    """Identify diverse failure cases on dev set with diagnosis and proposed fixes."""
    preds = {p["id"]: p.get("output") for p in pred_rows}
    KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

    def parse(t):
        try:
            return json.loads(t.strip()) if t else None
        except:
            return None

    failures = []
    seen_error_types = set()

    for g in gold_rows:
        rid = g["id"]
        gold = json.loads(g["messages"][-1]["content"])
        raw_pred = preds.get(rid, "")
        pred = parse(raw_pred)
        u_text = g["messages"][1]["content"]

        if pred is None:
            failures.append({
                "id": rid,
                "error_type": "json_parse_failure",
                "short_evidence": u_text[:120].replace("\n", " "),
                "raw_prediction": raw_pred[:150],
                "gold": gold,
                "mismatch_fields": ["json_syntax"],
                "mechanism": "Model generated invalid JSON syntax, markdown fences, or truncated text.",
                "proposed_fix": "Constrain supervised token length and enforce JSON start delimiter in curriculum.",
            })
            continue

        mismatches = [k for k in KEYS if k not in pred or type(pred[k]) is not type(gold[k]) or pred[k] != gold[k]]

        if mismatches:
            # Classify error type
            primary_err = mismatches[0]
            if "priority" in mismatches:
                primary_err = "priority_error"
            elif "category" in mismatches:
                primary_err = "category_error"
            elif "txn_date" in mismatches:
                primary_err = "date_error"
            elif "amount_paise" in mismatches:
                primary_err = "amount_error"
            elif "needs_human" in mismatches:
                primary_err = "needs_human_error"

            # Mechanism & fix diagnosis
            mechanism = ""
            proposed_fix = ""
            if primary_err == "priority_error":
                mechanism = f"Expected priority '{gold['priority']}' but predicted '{pred.get('priority')}'; likely boundary confusion near ₹5,000/₹50,000 or missed subtle legal trigger."
                proposed_fix = "Augment training data with synthetic boundary tickets around ₹4,999 vs ₹5,000 and ₹49,999 vs ₹50,000."
            elif primary_err == "date_error":
                mechanism = f"Predicted txn_date '{pred.get('txn_date')}' vs gold '{gold.get('txn_date')}'; likely relative date calculation error or distraction from earlier email quote."
                proposed_fix = "Train model with explicit date grounding pairs contrasting Received header with message body."
            elif primary_err == "amount_error":
                mechanism = f"Predicted amount {pred.get('amount_paise')} vs gold {gold.get('amount_paise')}; distracted by account balance, fee, or failed rupee-paise unit conversion."
                proposed_fix = "Add negative examples featuring explicit balance and fee distractors alongside disputed amounts."
            elif primary_err == "category_error":
                mechanism = f"Predicted '{pred.get('category')}' vs gold '{gold.get('category')}'; intent ambiguity between failed payment and refund request."
                proposed_fix = "Refine category contrastive pairs emphasizing user initiation vs merchant reversal."
            elif primary_err == "needs_human_error":
                mechanism = f"needs_human mismatch: predicted {pred.get('needs_human')} vs gold {gold.get('needs_human')}; missed repeat-contact cue or subtle frustration phrase."
                proposed_fix = "Curate dedicated repeat-contact lexicon pairs (e.g. 'third time', 'baar baar') in train set."
            else:
                mechanism = f"Field mismatch on {mismatches}."
                proposed_fix = "Expand domain coverage for specific field tokens."

            failures.append({
                "id": rid,
                "error_type": primary_err,
                "short_evidence": u_text[:140].replace("\n", " "),
                "raw_prediction": raw_pred[:150],
                "gold": {k: gold[k] for k in mismatches},
                "predicted": {k: pred.get(k) for k in mismatches},
                "mismatch_fields": mismatches,
                "mechanism": mechanism,
                "proposed_fix": proposed_fix,
            })

    # Return diverse failures
    return failures[:max_cases]
