"""Unit tests for NimbusPay v2 contracts, priority resolution, and schema invariants."""

import pytest
from src.triage.contracts import (
    KEYS,
    CATEGORIES,
    PRIORITIES,
    CHANNELS,
    LANGUAGES,
    classify_priority,
    compute_expected_needs_human,
    validate_record,
)


def test_valid_record():
    valid = {
        "category": "refund",
        "priority": "P2",
        "amount_paise": 2936300,
        "txn_date": "2026-09-13",
        "txn_id": "NP5972758064",
        "channel": "netbanking",
        "language": "en",
        "needs_human": False,
    }
    assert validate_record(valid) == []


def test_missing_and_extra_keys():
    # Missing priority
    record_missing = {
        "category": "refund",
        "amount_paise": 2936300,
        "txn_date": "2026-09-13",
        "txn_id": "NP5972758064",
        "channel": "netbanking",
        "language": "en",
        "needs_human": False,
    }
    issues = validate_record(record_missing)
    assert any(i.issue_type == "missing_keys" for i in issues)

    # Extra key
    record_extra = dict(record_missing)
    record_extra["priority"] = "P2"
    record_extra["explanation"] = "some text"
    issues = validate_record(record_extra)
    assert any(i.issue_type == "extra_keys" for i in issues)


def test_type_strictness():
    # amount_paise as string
    r1 = {
        "category": "refund", "priority": "P2", "amount_paise": "650000",
        "txn_date": "2026-09-13", "txn_id": "NP5972758064", "channel": "netbanking",
        "language": "en", "needs_human": False
    }
    issues = validate_record(r1)
    assert any(i.field == "amount_paise" and i.issue_type == "invalid_type" for i in issues)

    # needs_human as integer 1
    r2 = {
        "category": "refund", "priority": "P2", "amount_paise": 650000,
        "txn_date": "2026-09-13", "txn_id": "NP5972758064", "channel": "netbanking",
        "language": "en", "needs_human": 1
    }
    issues = validate_record(r2)
    assert any(i.field == "needs_human" and i.issue_type == "invalid_type" for i in issues)


def test_priority_rules():
    # Rule 1: Fraud -> P1 regardless of amount
    assert classify_priority("fraud", None) == "P1"
    assert classify_priority("fraud", 10000) == "P1"

    # Rule 1: Amount >= ₹50,000 (5,000,000 paise) -> P1
    assert classify_priority("payment_failed", 5_000_000) == "P1"
    assert classify_priority("refund", 5_000_001) == "P1"

    # Rule 1: Legal threat -> P1
    assert classify_priority("offers", None, has_legal_threat=True) == "P1"
    assert classify_priority("kyc", None, has_legal_threat=True) == "P1"

    # Rule 2: payment_failed / refund >= ₹5,000 (500,000 paise) -> P2
    assert classify_priority("payment_failed", 500_000) == "P2"
    assert classify_priority("refund", 4_999_999) == "P2"

    # Rule 2: account_access -> P2
    assert classify_priority("account_access", None) == "P2"

    # Rule 3: P3 otherwise
    assert classify_priority("payment_failed", 499_999) == "P3"
    assert classify_priority("refund", 10000) == "P3"
    assert classify_priority("offers", 25000) == "P3"
    assert classify_priority("other", None) == "P3"
    assert classify_priority("kyc", None) == "P3"


def test_p1_needs_human_invariant():
    r = {
        "category": "fraud", "priority": "P1", "amount_paise": 500000,
        "txn_date": "2026-09-13", "txn_id": "NP5972758064", "channel": "card",
        "language": "en", "needs_human": False
    }
    issues = validate_record(r)
    assert any(i.issue_type == "p1_needs_human_invariant" for i in issues)


def test_non_transaction_categories_nullability():
    for cat in ["account_access", "kyc", "other"]:
        r = {
            "category": cat, "priority": "P2" if cat == "account_access" else "P3",
            "amount_paise": 100000, "txn_date": "2026-09-13", "txn_id": "NP5972758064",
            "channel": "upi", "language": "en", "needs_human": False
        }
        issues = validate_record(r)
        assert any(i.issue_type == "nullability_violation" for i in issues)
