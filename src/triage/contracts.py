"""Contracts, schema definitions, and invariant checkers for NimbusPay triage.
Adheres strictly to candidate_pack/SCHEMA.md (v2).
"""

from typing import Any, Dict, List, Optional
import re
from dataclasses import dataclass

KEYS = [
    "category",
    "priority",
    "amount_paise",
    "txn_date",
    "txn_id",
    "channel",
    "language",
    "needs_human",
]

CATEGORIES = {
    "payment_failed",
    "refund",
    "fraud",
    "account_access",
    "kyc",
    "offers",
    "other",
}

PRIORITIES = {"P1", "P2", "P3"}

CHANNELS = {"upi", "card", "netbanking", "wallet"}

LANGUAGES = {"en", "hi-en"}

TXN_NULL_CATEGORIES = {"account_access", "kyc", "other"}

TXN_ID_REGEX = re.compile(r"^NP\d{10}$")
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ValidationIssue:
    field: str
    issue_type: str
    message: str
    severity: str = "ERROR"  # ERROR or WARNING


def classify_priority(
    category: str,
    amount_paise: Optional[int],
    has_legal_threat: bool = False,
) -> str:
    """Classify priority strictly according to SCHEMA.md section priority:
    1. P1 if:
       - category is fraud
       - disputed amount is >= ₹50,000 (5,000,000 paise)
       - customer makes a legal threat (RBI, ombudsman, consumer court, lawyer, legal notice/action)
    2. P2 if not P1 and:
       - category is payment_failed or refund AND amount >= ₹5,000 (500,000 paise)
       - category is account_access
    3. P3 otherwise.
    """
    # Rule 1: P1
    if category == "fraud":
        return "P1"
    if amount_paise is not None and amount_paise >= 5_000_000:
        return "P1"
    if has_legal_threat:
        return "P1"

    # Rule 2: P2
    if category in {"payment_failed", "refund"} and amount_paise is not None and amount_paise >= 500_000:
        return "P2"
    if category == "account_access":
        return "P2"

    # Rule 3: P3
    return "P3"


def compute_expected_needs_human(
    priority: str,
    has_prior_contact: bool = False,
) -> bool:
    """Classify needs_human strictly according to SCHEMA.md:
    true if priority is P1, OR the customer says they have already contacted support about
    this issue before (e.g. 'third time', 'again and again', 'baar baar').
    Otherwise false.
    """
    if priority == "P1":
        return True
    return bool(has_prior_contact)


def validate_record(obj: Any) -> List[ValidationIssue]:
    """Validate a parsed JSON dictionary against the NimbusPay v2 contract.
    Returns a list of ValidationIssue instances.
    """
    issues: List[ValidationIssue] = []

    if not isinstance(obj, dict):
        return [ValidationIssue(field="root", issue_type="type_error", message="Output must be a JSON object")]

    # Check exact keys
    obj_keys = set(obj.keys())
    expected_keys = set(KEYS)
    if obj_keys != expected_keys:
        missing = expected_keys - obj_keys
        extra = obj_keys - expected_keys
        if missing:
            issues.append(ValidationIssue(field="keys", issue_type="missing_keys", message=f"Missing keys: {sorted(missing)}"))
        if extra:
            issues.append(ValidationIssue(field="keys", issue_type="extra_keys", message=f"Extra keys: {sorted(extra)}"))

    # Category validation
    cat = obj.get("category")
    if not isinstance(cat, str) or cat not in CATEGORIES:
        issues.append(ValidationIssue(field="category", issue_type="invalid_category", message=f"Invalid category: {cat}"))

    # Priority validation
    prio = obj.get("priority")
    if not isinstance(prio, str) or prio not in PRIORITIES:
        issues.append(ValidationIssue(field="priority", issue_type="invalid_priority", message=f"Invalid priority: {prio}"))

    # amount_paise validation: integer or None (strictly not bool, not float, not str)
    amt = obj.get("amount_paise")
    if amt is not None:
        if type(amt) is not int or isinstance(amt, bool):
            issues.append(ValidationIssue(field="amount_paise", issue_type="invalid_type", message=f"amount_paise must be int or null, got {type(amt).__name__}"))
        elif amt < 0:
            issues.append(ValidationIssue(field="amount_paise", issue_type="negative_amount", message=f"amount_paise cannot be negative: {amt}"))

    # txn_date validation: string YYYY-MM-DD or None
    dt = obj.get("txn_date")
    if dt is not None:
        if not isinstance(dt, str) or not DATE_REGEX.match(dt):
            issues.append(ValidationIssue(field="txn_date", issue_type="invalid_date_format", message=f"txn_date must be YYYY-MM-DD or null, got: {dt}"))

    # txn_id validation: NP + 10 digits or None
    tid = obj.get("txn_id")
    if tid is not None:
        if not isinstance(tid, str) or not TXN_ID_REGEX.match(tid):
            issues.append(ValidationIssue(field="txn_id", issue_type="invalid_txn_id", message=f"txn_id must be NP + 10 digits, got: {tid}"))

    # channel validation: upi, card, netbanking, wallet, or None
    ch = obj.get("channel")
    if ch is not None:
        if not isinstance(ch, str) or ch not in CHANNELS:
            issues.append(ValidationIssue(field="channel", issue_type="invalid_channel", message=f"channel must be one of {CHANNELS} or null, got: {ch}"))

    # language validation: en, hi-en
    lang = obj.get("language")
    if not isinstance(lang, str) or lang not in LANGUAGES:
        issues.append(ValidationIssue(field="language", issue_type="invalid_language", message=f"language must be one of {LANGUAGES}, got: {lang}"))

    # needs_human validation: strictly bool
    nh = obj.get("needs_human")
    if type(nh) is not bool:
        issues.append(ValidationIssue(field="needs_human", issue_type="invalid_type", message=f"needs_human must be boolean, got {type(nh).__name__}"))

    # Cross-field Invariants
    # Invariant 1: Transaction nullability for non-transaction categories
    if cat in TXN_NULL_CATEGORIES:
        for k in ["amount_paise", "txn_date", "txn_id", "channel"]:
            val = obj.get(k)
            if val is not None:
                issues.append(
                    ValidationIssue(
                        field=k,
                        issue_type="nullability_violation",
                        message=f"Category '{cat}' must have {k}=null, but got {val}",
                    )
                )

    # Invariant 2: P1 strictly requires needs_human=True
    if prio == "P1" and nh is False:
        issues.append(
            ValidationIssue(
                field="needs_human",
                issue_type="p1_needs_human_invariant",
                message="priority='P1' requires needs_human=True",
            )
        )

    return issues
