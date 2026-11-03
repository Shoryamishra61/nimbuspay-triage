"""Heuristic and rule-based triage predictor.
Used for offline diagnostic comparison, dev baseline reference, and fallback verification.
Implements the rules from candidate_pack/SCHEMA.md deterministically.
"""

from typing import Dict, Any, List, Optional
import re
import json
import datetime
from .contracts import KEYS, classify_priority, compute_expected_needs_human, TXN_NULL_CATEGORIES
from .evidence import (
    extract_customer_body,
    parse_received_date,
    find_amount_candidates,
    extract_transaction_date,
    find_canonical_np_txn_id,
    has_legal_threat,
    has_prior_support_contact,
    is_hinglish,
    CHANNEL_KEYWORDS,
)


def predict_ticket(user_text: str) -> Dict[str, Any]:
    """Extract and classify all 8 triage fields for a single customer ticket."""
    rec_date = parse_received_date(user_text)
    body = extract_customer_body(user_text)
    body_lower = body.lower()

    # 1. Category Classification
    # Fraud wins over every other category
    fraud_pats = [
        r"\bfraud\b", r"\bfraudulent\b", r"\bunauthorized\b", r"\bunrecognised\b", r"\bunrecognized\b",
        r"\bdid\s*n[o']?t\s*(?:make|approve|do)\b", r"\bnever\s*(?:made|approved|did)\b",
        r"\bmaine\s*nahi\s*kiya\b", r"\bpaise\s*nikal\s*liye\b", r"\botp\s*maang\b", r"\bscam\b", r"\bhack(?:ed)?\b",
        r"\bsomeone\s*(?:else\s*)?(?:stole|used|took)\b", r"\bwithout\s*my\s*(?:permission|consent|knowledge)\b"
    ]
    is_fraud = any(re.search(p, body_lower) for p in fraud_pats)

    # KYC
    kyc_pats = [
        r"\bkyc\b", r"\baadhaar\b", r"\bpan\s*card\b", r"\bvideo\s*kyc\b",
        r"\bidentity\s*verif", r"\bverif(?:y|ying|ication)\s*my\s*identity\b", r"\bprove\s*my\s*identity\b",
        r"\bdocument\s*upload\b"
    ]
    is_kyc = any(re.search(p, body_lower) for p in kyc_pats)

    # Account access
    access_pats = [
        r"\baccount\s*access\b", r"\blocked\s*account\b", r"\baccount\s*(?:is\s*)?locked\b",
        r"\bblocked\s*account\b", r"\baccount\s*(?:is\s*)?blocked\b", r"\bblocked\s*pin\b",
        r"\bforgot(?:ten)?\s*(?:passcode|password|pin)\b", r"\breset\s*(?:password|passcode|pin)\b",
        r"\botp\s*(?:not\s*received|nahi\s*aa\s*raha|nahi\s*mila)\b", r"\bcannot\s*log\s*in\b",
        r"\blogout\s*kar\s*deta\b", r"\bcan'?t\s*access\b"
    ]
    is_access = any(re.search(p, body_lower) for p in access_pats)

    # Offers
    offers_pats = [
        r"\bcashback\b", r"\bpromo\s*code\b", r"\bcoupon\b", r"\bscratch\s*card\b",
        r"\breferral\b", r"\breward\s*points?\b", r"\boffer\b", r"\binaam\b"
    ]
    is_offers = any(re.search(p, body_lower) for p in offers_pats)

    # Refund
    refund_pats = [
        r"\brefund\b", r"\bcharged\s*(?:more\s*than\s*once|twice|2\s*times)\b",
        r"\bdo\s*baar\s*(?:paise\s*)?(?:kat|le)\s*liye\b", r"\bdouble\s*charge\b",
        r"\bcancell?ed\s*order\b", r"\breturn(?:ed)?\b", r"\bwapas\s*chahiye\b",
        r"\bpaise\s*wapas\b", r"\brefund\s*(?:nahi|not)\b"
    ]
    is_refund = any(re.search(p, body_lower) for p in refund_pats)

    # Payment failed
    fail_pats = [
        r"\bfailed\b", r"\bdecline[ds]?\b", r"\bstuck\b", r"\bnot\s*going\s*through\b",
        r"\bdid\s*n[o']?t\s*(?:go\s*through|reach|work)\b", r"\bpending\s*payment\b",
        r"\bkat\s*gaye\s*(?:lekin|par)\b", r"\bkat\s*gaye\b", r"\btransfer\s*failed\b",
        r"\btop\s*up\s*(?:didn't\s*work|failed)\b", r"\bpaise\s*kat\s*gaye\b",
        r"\bnot\s*showing\s*they\s*received\b"
    ]
    is_fail = any(re.search(p, body_lower) for p in fail_pats)

    if is_fraud:
        cat = "fraud"
    elif is_kyc:
        cat = "kyc"
    elif is_access:
        cat = "account_access"
    elif is_offers:
        cat = "offers"
    elif is_refund:
        cat = "refund"
    elif is_fail:
        cat = "payment_failed"
    else:
        cat = "other"

    # 2. Transaction fields (Null for non-transaction categories)
    if cat in TXN_NULL_CATEGORIES:
        amt = None
        txn_d = None
        txn_i = None
        chn = None
    else:
        # Amount in paise
        cands = find_amount_candidates(user_text)
        non_distractors = [c for c in cands if not c["is_distractor"]]
        amt = non_distractors[0]["paise"] if non_distractors else (cands[0]["paise"] if cands else None)

        # Transaction Date
        txn_d = extract_transaction_date(user_text, rec_date)

        # Transaction ID
        txn_i = find_canonical_np_txn_id(user_text)

        # Channel
        chn = None
        for ch, pats in CHANNEL_KEYWORDS.items():
            if any(re.search(p, body_lower) for p in pats):
                chn = ch
                break

    # 3. Language
    lang = "hi-en" if is_hinglish(user_text) else "en"

    # 4. Priority
    legal = has_legal_threat(user_text)
    prio = classify_priority(cat, amt, has_legal_threat=legal)

    # 5. Needs human
    prior = has_prior_support_contact(user_text)
    needs_h = compute_expected_needs_human(prio, has_prior_contact=prior)

    return {
        "category": cat,
        "priority": prio,
        "amount_paise": amt,
        "txn_date": txn_d,
        "txn_id": txn_i,
        "channel": chn,
        "language": lang,
        "needs_human": needs_h,
    }


def generate_heuristic_dataset_predictions(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate raw string predictions for each row in rows."""
    results = []
    for r in rows:
        user_content = r["messages"][1]["content"]
        record = predict_ticket(user_content)
        # Format as strict JSON string without extra whitespace
        raw_json_str = json.dumps(record, ensure_ascii=False)
        results.append({
            "id": r["id"],
            "output": raw_json_str,
        })
    return results

