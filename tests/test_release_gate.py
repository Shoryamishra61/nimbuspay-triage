"""Tests for pre-release verification gates."""

import os
import json
from src.triage.release import verify_file_existence, verify_cleaning_log, verify_predictions_test


def test_cleaning_log_release_check():
    ok, issues = verify_cleaning_log(
        log_path="cleaning_log.csv",
        raw_train_path="candidate_pack/data/train.jsonl",
        cleaned_path="artifacts/cleaned.jsonl",
    )
    assert ok, f"Cleaning log verification failed: {issues}"


def test_predictions_test_envelope():
    ok, issues = verify_predictions_test(
        pred_path="predictions_test.jsonl",
        test_inputs_path="candidate_pack/data/test_inputs.jsonl",
    )
    assert ok, f"Predictions envelope verification failed: {issues}"


def test_predictions_no_systematic_202600_collapse():
    with open("predictions_test.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    amounts = [
        json.loads(r["output"])["amount_paise"]
        for r in rows
        if json.loads(r["output"])["amount_paise"] is not None
    ]
    # No spurious 202600 amounts
    bogus_count = sum(1 for a in amounts if a == 202600)
    assert bogus_count == 0, f"Found {bogus_count} rows with collapsed 202600 paise amount!"


def test_ts00001_and_ts00400_provenance():
    with open("predictions_test.jsonl", "r", encoding="utf-8") as f:
        rows = {r["id"]: json.loads(r["output"]) for r in [json.loads(line) for line in f if line.strip()]}

    ts1 = rows["ts-00001"]
    assert ts1["amount_paise"] is None, "ts-00001 amount should be null"
    assert ts1["txn_date"] == "2026-06-30", "ts-00001 date should be 2026-06-30"
    assert ts1["txn_id"] == "NP6090913363"
    assert ts1["category"] == "fraud"
    assert ts1["priority"] == "P1"
    assert ts1["needs_human"] is True

    ts400 = rows["ts-00400"]
    assert ts400["amount_paise"] == 1720499, "ts-00400 amount should be 1720499 paise (Rs 17,204.99)"
    assert ts400["txn_date"] == "2026-09-14", "ts-00400 date should be 2026-09-14"
    assert ts400["txn_id"] == "NP2238995191"
    assert ts400["category"] == "payment_failed"
