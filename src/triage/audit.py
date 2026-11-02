"""Data audit module for training_assessment/candidate_pack/data/train.jsonl.
Performs comprehensive, independent structural and semantic audits across all 1,445 rows.
Generates an audit report detailing findings and row-level evidence.
"""

from typing import List, Dict, Any
import json
import re
from collections import Counter
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
)
from .evidence import (
    has_legal_threat,
    has_prior_support_contact,
    find_canonical_np_txn_id,
    find_amount_candidates,
)


def audit_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Audit all training rows independently against the NimbusPay v2 specification."""
    findings = {
        "total_rows": len(rows),
        "malformed_json": [],
        "legacy_prio": [],
        "internal_duplicates": [],
        "noncanonical_txn_id": [],
        "p1_needs_human_false": [],
        "nullability_violations": [],
        "meaningless_tickets": [],
        "rupee_vs_paise_candidates": [],
        "priority_inconsistencies": [],
    }

    seen_user_contents: Dict[str, str] = {}  # content -> first_id

    for r in rows:
        rid = r["id"]
        msgs = r.get("messages", [])
        if len(msgs) < 3:
            findings["malformed_json"].append({"id": rid, "reason": "missing_messages_structure"})
            continue

        u_content = msgs[1]["content"]
        a_content = msgs[2]["content"]

        # Check meaningless tickets
        parts = u_content.split("\n\n", 1)
        body = parts[1].strip() if len(parts) > 1 else u_content.strip()
        if len(body) < 15 or body.lower() in {"test", "asdf asdf", "asdf", "testing", "hello", "hi", "test test 123"}:
            findings["meaningless_tickets"].append({"id": rid, "body": body})

        # Check internal duplicate user input
        u_norm = u_content.strip()
        if u_norm in seen_user_contents:
            findings["internal_duplicates"].append({
                "id": rid,
                "duplicate_of": seen_user_contents[u_norm],
            })
        else:
            seen_user_contents[u_norm] = rid

        # Parse assistant JSON
        try:
            obj = json.loads(a_content.strip())
        except Exception as e:
            findings["malformed_json"].append({"id": rid, "reason": f"json_decode_error: {e}", "raw": a_content[:100]})
            continue

        if not isinstance(obj, dict):
            findings["malformed_json"].append({"id": rid, "reason": "output_not_a_dict", "raw": a_content[:100]})
            continue

        # Check legacy prio
        if "prio" in obj and "priority" not in obj:
            findings["legacy_prio"].append({"id": rid, "keys": list(obj.keys())})

        prio = obj.get("priority") or obj.get("prio")
        cat = obj.get("category")
        amt = obj.get("amount_paise")
        tid = obj.get("txn_id")
        nh = obj.get("needs_human")

        # Check P1 needs_human=False
        if prio == "P1" and nh is False:
            findings["p1_needs_human_false"].append({"id": rid, "category": cat, "priority": prio})

        # Check noncanonical txn_id
        if tid and not TXN_ID_REGEX.match(tid):
            findings["noncanonical_txn_id"].append({"id": rid, "raw_id": tid})

        # Check transaction fields on null categories
        if cat in TXN_NULL_CATEGORIES:
            non_nulls = [k for k in ["amount_paise", "txn_date", "txn_id", "channel"] if obj.get(k) is not None]
            if non_nulls:
                findings["nullability_violations"].append({"id": rid, "category": cat, "non_null_fields": non_nulls})

        # Check rupee vs paise suspect cases
        if amt is not None and isinstance(amt, int) and amt > 0:
            # Check if ticket mentions amt directly as rupees
            # E.g. ₹amt or Rs amt or amount in body matches amt
            m = re.search(rf"(?:₹|Rs\.?|INR)\s*{amt}\b", u_content, re.IGNORECASE)
            if m:
                findings["rupee_vs_paise_candidates"].append({
                    "id": rid,
                    "label_amount": amt,
                    "suggested_paise": amt * 100,
                    "matched_text": m.group(0),
                })

        # Check priority consistency
        if cat in CATEGORIES:
            legal = has_legal_threat(u_content)
            expected_prio = classify_priority(cat, amt, has_legal_threat=legal)
            if prio in PRIORITIES and prio != expected_prio:
                findings["priority_inconsistencies"].append({
                    "id": rid,
                    "category": cat,
                    "label_prio": prio,
                    "expected_prio": expected_prio,
                    "amount_paise": amt,
                    "legal_threat": legal,
                })

    return findings


def print_audit_summary(findings: Dict[str, Any]):
    """Print readable summary of audit findings."""
    print("=" * 60)
    print("INDEPENDENT DATA AUDIT SUMMARY (train.jsonl)")
    print("=" * 60)
    print(f"Total Rows Audited: {findings['total_rows']}")
    print(f"Malformed JSON Labels: {len(findings['malformed_json'])}")
    print(f"Legacy 'prio' Schema: {len(findings['legacy_prio'])}")
    print(f"Internal Duplicate Occurrences: {len(findings['internal_duplicates'])}")
    print(f"Noncanonical Transaction IDs: {len(findings['noncanonical_txn_id'])}")
    print(f"P1 with needs_human=False: {len(findings['p1_needs_human_false'])}")
    print(f"Nullability Violations (account_access/kyc/other): {len(findings['nullability_violations'])}")
    print(f"Meaningless Ticket Bodies: {len(findings['meaningless_tickets'])}")
    print(f"Rupee-vs-Paise Candidate Rows: {len(findings['rupee_vs_paise_candidates'])}")
    print(f"Candidate Priority Inconsistencies: {len(findings['priority_inconsistencies'])}")
    print("=" * 60)
