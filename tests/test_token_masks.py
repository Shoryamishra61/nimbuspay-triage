"""Tests for tokenization, chat template formatting, and supervised target loss masking."""

from transformers import AutoTokenizer
from src.triage.tokenizer_pipeline import NimbusChatFormatter


def test_token_mask_numerical_proof():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    formatter = NimbusChatFormatter(tok, max_seq_length=2304)

    row = {
        "id": "test-001",
        "messages": [
            {
                "role": "system",
                "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON.",
            },
            {
                "role": "user",
                "content": "Received: 2026-08-14 (Fri)\nSource: email\n\nPayment failed for ₹6,500.",
            },
            {
                "role": "assistant",
                "content": '{"category": "payment_failed", "priority": "P2", "amount_paise": 650000, "txn_date": null, "txn_id": null, "channel": null, "language": "en", "needs_human": false}',
            },
        ],
    }

    formatted = formatter.format_row(row, is_training=True)

    input_ids = formatted["input_ids"]
    labels = formatted["labels"]
    p_len = formatted["prompt_length"]
    t_len = formatted["target_length"]

    assert len(input_ids) == len(labels)
    assert len(input_ids) == p_len + t_len

    # Invariant: prompt tokens MUST have label -100
    for idx in range(p_len):
        assert labels[idx] == -100, f"Prompt token at {idx} leaked into loss computation!"

    # Invariant: target tokens MUST have valid non-negative token IDs
    for idx in range(p_len, len(labels)):
        assert labels[idx] >= 0, f"Target token at {idx} erroneously masked!"

    # Invariant: decoded target must end with EOS token
    target_ids = labels[p_len:]
    decoded_target = tok.decode(target_ids)
    assert "<|im_end|>" in decoded_target

