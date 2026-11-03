"""Tests for data cleaning idempotency and integrity."""

import os
import csv
from src.triage.source_io import load_jsonl
from src.triage.clean import clean_train_dataset


def test_clean_idempotency():
    train_rows = load_jsonl("candidate_pack/data/train.jsonl", expected_split="train")
    cleaned1, log1 = clean_train_dataset(train_rows)
    cleaned2, log2 = clean_train_dataset(train_rows)

    assert cleaned1 == cleaned2
    assert log1 == log2


def test_cleaning_log_contract():
    train_rows = load_jsonl("candidate_pack/data/train.jsonl", expected_split="train")
    cleaned_rows, log_rows = clean_train_dataset(train_rows)

    # 1. Every row dropped or fixed is listed exactly once
    log_ids = [l["id"] for l in log_rows]
    assert len(log_ids) == len(set(log_ids)), "Duplicate row IDs found in cleaning_log!"

    # 2. Actions must be strictly 'drop' or 'fix'
    for l in log_rows:
        assert l["action"] in {"drop", "fix"}
        assert len(l["reason"].strip()) > 0

    # 3. Exactly 1,445 = len(cleaned_rows) + count(drops)
    drops_cnt = sum(1 for l in log_rows if l["action"] == "drop")
    assert len(train_rows) == len(cleaned_rows) + drops_cnt

