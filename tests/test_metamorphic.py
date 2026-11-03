"""Metamorphic and property-based invariance tests from TEST_MATRIX.md (Section F)."""

import datetime
from src.triage.contracts import classify_priority, compute_expected_needs_human
from src.triage.evidence import (
    find_canonical_np_txn_id,
    resolve_relative_date,
    has_legal_threat,
    has_prior_support_contact,
    parse_amount_to_paise,
)


def test_metamorphic_priority_boundaries():
    # Boundary at ₹5,000 (500,000 paise) for payment_failed / refund
    assert classify_priority("payment_failed", 499_999) == "P3"
    assert classify_priority("payment_failed", 500_000) == "P2"
    assert classify_priority("refund", 499_999) == "P3"
    assert classify_priority("refund", 500_000) == "P2"

    # Boundary at ₹50,000 (5,000,000 paise)
    assert classify_priority("payment_failed", 4_999_999) == "P2"
    assert classify_priority("payment_failed", 5_000_000) == "P1"
    assert classify_priority("refund", 4_999_999) == "P2"
    assert classify_priority("refund", 5_000_000) == "P1"


def test_metamorphic_fraud_dominance():
    # Fraud is always P1 regardless of amount or absence of amount
    amounts = [None, 0, 100, 500_000, 5_000_000, 100_000_000]
    for amt in amounts:
        assert classify_priority("fraud", amt) == "P1"


def test_metamorphic_legal_threat_override():
    # Legal threat forces P1 regardless of category or amount
    categories = ["payment_failed", "refund", "account_access", "kyc", "offers", "other"]
    for cat in categories:
        assert classify_priority(cat, 100, has_legal_threat=True) == "P1"


def test_metamorphic_needs_human_escalation():
    # P1 always requires human
    assert compute_expected_needs_human("P1", False) is True
    assert compute_expected_needs_human("P1", True) is True

    # P2/P3 depends on prior contact
    assert compute_expected_needs_human("P2", False) is False
    assert compute_expected_needs_human("P2", True) is True
    assert compute_expected_needs_human("P3", False) is False
    assert compute_expected_needs_human("P3", True) is True


def test_metamorphic_receipt_date_shift():
    # Changing receipt date shifts relative dates by exact delta
    rec1 = datetime.date(2026, 8, 14)
    rec2 = datetime.date(2026, 8, 20)
    delta = rec2 - rec1

    d1 = resolve_relative_date("yesterday", rec1)
    d2 = resolve_relative_date("yesterday", rec2)
    assert (d2 - d1) == delta

    p1 = resolve_relative_date("parso", rec1)
    p2 = resolve_relative_date("parso", rec2)
    assert (p2 - p1) == delta


def test_metamorphic_np_casing_and_separators():
    variants = [
        "np 12345 67890",
        "NP 12345 67890",
        "np-12345-67890",
        "NP-12345-67890",
        "np1234567890",
        "NP1234567890",
    ]
    for v in variants:
        assert find_canonical_np_txn_id(f"Transaction ID {v}") == "NP1234567890"

