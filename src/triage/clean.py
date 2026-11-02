"""Deterministic and idempotent data cleaning and repair pipeline.
Operates strictly on candidate_pack/data/train.jsonl and SCHEMA.md.
NEVER touches dev or test data.
Generates cleaned.jsonl and cleaning_log.csv.
"""

from typing import List, Dict, Any, Tuple
import json
import csv
import re
import os
from .contracts import (
    KEYS,
    CATEGORIES,
    PRIORITIES,
    CHANNELS,
    LANGUAGES,
    TXN_NULL_CATEGORIES,
    TXN_ID_REGEX,
    DATE_REGEX,
    classify_priority,
    compute_expected_needs_human,
)
from .evidence import (
    has_legal_threat,
    has_prior_support_contact,
    find_canonical_np_txn_id,
    find_amount_candidates,
)


def clean_train_dataset(
    train_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Clean and repair training rows deterministically.
    Returns:
        cleaned_rows: list of compliant, cleaned JSONL dicts
        cleaning_log: list of dicts with keys 'id', 'action', 'reason'
    """
    cleaned_rows: List[Dict[str, Any]] = []
    cleaning_log: List[Dict[str, str]] = []

    seen_user_inputs: Dict[str, str] = {}  # normalized text -> first_id

    for r in train_rows:
        rid = r["id"]
        msgs = r.get("messages", [])
        if len(msgs) < 3:
            cleaning_log.append({
                "id": rid,
                "action": "drop",
                "reason": "malformed_messages_structure",
            })
            continue

        u_content = msgs[1]["content"]
        a_content = msgs[2]["content"]

        # Check meaningless tickets
        parts = u_content.split("\n\n", 1)
        body = parts[1].strip() if len(parts) > 1 else u_content.strip()
        if len(body) < 15 or body.lower() in {
            "test",
            "asdf asdf",
            "asdf",
            "testing",
            "hello",
            "hi",
            "test test 123",
        }:
            cleaning_log.append({
                "id": rid,
                "action": "drop",
                "reason": "meaningless_ticket_body",
            })
            continue

        # Check internal duplicate within train
        u_norm = u_content.strip()
        if u_norm in seen_user_inputs:
            cleaning_log.append({
                "id": rid,
                "action": "drop",
                "reason": f"duplicate_of_{seen_user_inputs[u_norm]}",
            })
            continue
        seen_user_inputs[u_norm] = rid

        # Parse assistant JSON
        try:
            obj = json.loads(a_content.strip())
        except Exception:
            cleaning_log.append({
                "id": rid,
                "action": "drop",
                "reason": "malformed_truncated_json_unrecoverable",
            })
            continue

        if not isinstance(obj, dict):
            cleaning_log.append({
                "id": rid,
                "action": "drop",
                "reason": "malformed_output_not_a_dict",
            })
            continue

        # Row is kept; check if any fields need repair
        fix_reasons = []
        new_obj = dict(obj)

        # 1. Migrate legacy 'prio'
        if "prio" in new_obj and "priority" not in new_obj:
            new_obj["priority"] = new_obj.pop("prio")
            fix_reasons.append("migrate_legacy_prio_schema")

        # 1b. Fix DD/MM/YYYY date formatting in legacy rows
        dt = new_obj.get("txn_date")
        if dt and isinstance(dt, str):
            date_m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", dt)
            if date_m:
                new_obj["txn_date"] = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
                fix_reasons.append("convert_date_dd_mm_yyyy_to_iso")

        # 2. Canonicalize transaction ID
        tid = new_obj.get("txn_id")
        if tid and not TXN_ID_REGEX.match(tid):
            clean_tid = re.sub(r"[\s\-_]+", "", tid).upper()
            if TXN_ID_REGEX.match(clean_tid):
                new_obj["txn_id"] = clean_tid
                fix_reasons.append("canonicalize_np_txn_id")

        # 3. Rupee-vs-paise correction for obvious errors
        amt = new_obj.get("amount_paise")
        if amt is not None and isinstance(amt, int) and amt > 0:
            # Check if ticket body has explicit rupee amount matching amt
            m = re.search(rf"(?:₹|Rs\.?|INR)\s*({amt}(?:\.[0-9]+)?)\b", u_content, re.IGNORECASE)
            if m:
                # E.g. ticket says ₹168787 or Rs 167
                val_str = m.group(1).replace(",", "")
                try:
                    val_float = float(val_str)
                    new_paise = int(val_float * 100)
                    if new_paise != amt and new_paise > 0:
                        new_obj["amount_paise"] = new_paise
                        amt = new_paise
                        fix_reasons.append("convert_rupee_to_paise")
                except ValueError:
                    pass

        # 4. Enforce nullability for non-transaction categories
        cat = new_obj.get("category")
        if cat in TXN_NULL_CATEGORIES:
            null_fixed = False
            for k in ["amount_paise", "txn_date", "txn_id", "channel"]:
                if new_obj.get(k) is not None:
                    new_obj[k] = None
                    null_fixed = True
            if null_fixed:
                fix_reasons.append("nullify_non_txn_category_fields")
                amt = None

        # 5. Check and recalculate priority according to SCHEMA.md
        if cat in CATEGORIES:
            legal = has_legal_threat(u_content)
            correct_prio = classify_priority(cat, amt, has_legal_threat=legal)
            curr_prio = new_obj.get("priority")
            if curr_prio != correct_prio:
                new_obj["priority"] = correct_prio
                fix_reasons.append(f"recalculate_priority_{curr_prio}_to_{correct_prio}")

        # 6. Enforce P1 needs_human=True invariant
        prio = new_obj.get("priority")
        if prio == "P1" and new_obj.get("needs_human") is not True:
            new_obj["needs_human"] = True
            fix_reasons.append("enforce_p1_needs_human_invariant")

        # Ensure exact 8 keys in canonical order
        canonical_record = {k: new_obj.get(k) for k in KEYS}

        if fix_reasons:
            cleaning_log.append({
                "id": rid,
                "action": "fix",
                "reason": "; ".join(fix_reasons),
            })

        # Save cleaned row
        cleaned_rows.append({
            "id": rid,
            "messages": [
                msgs[0],
                msgs[1],
                {
                    "role": "assistant",
                    "content": json.dumps(canonical_record, ensure_ascii=False),
                },
            ],
        })

    return cleaned_rows, cleaning_log


def save_cleaning_log(log_rows: List[Dict[str, str]], path: str):
    """Save cleaning log to CSV with columns: id, action, reason."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "action", "reason"])
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    os.replace(tmp_path, path)
