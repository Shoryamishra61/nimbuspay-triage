"""Tests for offline evidence extraction against TEST_MATRIX.md cases."""

import datetime
from src.triage.evidence import (
    parse_amount_to_paise,
    resolve_relative_date,
    find_canonical_np_txn_id,
    has_legal_threat,
    has_prior_support_contact,
    is_hinglish,
    extract_non_quoted_lines,
)


def test_amount_paise_conversion():
    # B01: ₹6,500 -> 650000 paise
    assert parse_amount_to_paise("6,500") == 650000
    # B02: ₹50,000 -> 5000000 paise
    assert parse_amount_to_paise("50,000") == 5000000
    # B03: ₹49,999.99 -> 4999999 paise
    assert parse_amount_to_paise("49,999.99") == 4999999
    # B06: 1.30k -> ₹1,300 -> 130000 paise
    assert parse_amount_to_paise("1.30", "k") == 130000
    # B07: 1.5 lakh -> ₹1,50,000 -> 15000000 paise
    assert parse_amount_to_paise("1.5", "lakh") == 15000000
    # B08: ₹2,74,981 -> 27498100 paise
    assert parse_amount_to_paise("2,74,981") == 27498100
    # B09: Rs 21,378.99 -> 2137899 paise
    assert parse_amount_to_paise("21,378.99") == 2137899


def test_relative_date_resolution():
    rec = datetime.date(2026, 9, 14)
    # D01: yesterday
    assert resolve_relative_date("yesterday", rec) == datetime.date(2026, 9, 13)
    # D02: yesterday from 2026-01-01 -> 2025-12-31
    assert resolve_relative_date("yesterday", datetime.date(2026, 1, 1)) == datetime.date(2025, 12, 31)
    # D03: 2 days ago from 2026-03-01 -> 2026-02-27
    assert resolve_relative_date("2 days ago", datetime.date(2026, 3, 1)) == datetime.date(2026, 2, 27)
    # D04: parso from leap year 2028-03-01 -> 2028-02-28
    assert resolve_relative_date("parso", datetime.date(2028, 3, 1)) == datetime.date(2028, 2, 28)
    # kal / parso
    assert resolve_relative_date("kal", rec) == datetime.date(2026, 9, 13)
    assert resolve_relative_date("parso", rec) == datetime.date(2026, 9, 12)
    assert resolve_relative_date("6 din pehle", datetime.date(2026, 7, 6)) == datetime.date(2026, 6, 30)


def test_np_txn_id_canonicalization():
    # D09: np 12345 67890
    assert find_canonical_np_txn_id("Transaction id is np 12345 67890 please check") == "NP1234567890"
    # D10: NP-12345-67890
    assert find_canonical_np_txn_id("Ref: NP-12345-67890") == "NP1234567890"
    # Plain np + 10 digits
    assert find_canonical_np_txn_id("txn id np6090913363") == "NP6090913363"
    # Bank reference or ticket number should NOT match
    assert find_canonical_np_txn_id("Bank ref 123456789012, Ticket #12345") is None


def test_legal_threat_detection():
    # C07: RBI mention
    assert has_legal_threat("I will report this fraud to RBI immediately") is True
    # C08: consumer court
    assert has_legal_threat("I am going to consumer court tomorrow") is True
    # Ombudsman / Lawyer
    assert has_legal_threat("Taking this to banking ombudsman and my lawyer") is True
    # URGENT!! without legal mention is NOT a legal threat
    assert has_legal_threat("URGENT!! Please look into this right now!") is False


def test_prior_support_contact():
    # C10: third time
    assert has_prior_support_contact("This is the third time I am contacting you") is True
    # C11: baar baar
    assert has_prior_support_contact("Baar baar request karke thak gaya hoon") is True
    # C12: pehle bhi
    assert has_prior_support_contact("Pehle bhi complaint kar chuka hoon lekin koi solution nahi") is True
    # Ticket number alone without repeat phrase does NOT trigger prior contact
    assert has_prior_support_contact("My earlier ticket number is #54047. Please look into it.") is False


def test_quote_filtering():
    text = (
        "Received: 2026-08-10 (Mon)\nSource: email\n\n"
        "Yesterday my payment failed.\n"
        "> On Aug 3, 2026, support wrote:\n"
        "> We threaten you with legal notice.\n"
        "> Your transaction NP9999999999 was processed.\n"
    )
    non_quoted = extract_non_quoted_lines(text)
    assert "On Aug 3" not in non_quoted
    assert "legal notice" not in non_quoted
    assert has_legal_threat(text) is False
    assert find_canonical_np_txn_id(text) is None

