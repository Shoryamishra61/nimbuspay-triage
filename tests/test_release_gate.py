"""Tests for pre-release verification gates."""

import os
from src.triage.release import verify_file_existence, verify_cleaning_log, verify_predictions_test


def test_cleaning_log_release_check():
    ok, issues = verify_cleaning_log(
        log_path="cleaning_log.csv",
        raw_train_path="candidate_pack/data/train.jsonl",
        cleaned_path="artifacts/cleaned.jsonl",
    )
    assert ok, f"Cleaning log verification failed: {issues}"
