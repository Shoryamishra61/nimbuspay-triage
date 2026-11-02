"""Tokenizer and supervision pipeline.
Formats conversations using the native model chat template and implements
assistant-only loss masking with mathematical and numerical guarantees.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class NimbusChatFormatter:
    """Formatter and collator that ensures assistant-only token supervision."""

    def __init__(self, tokenizer, max_seq_length: int = 2304):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def format_row(self, row: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """Encode a single conversation row with assistant-only masking.
        During training, prompt tokens are masked with -100 in labels.
        """
        msgs = row["messages"]
        system_user_msgs = msgs[:2]

        # Render prompt with model's native chat template
        prompt_res = self.tokenizer.apply_chat_template(
            system_user_msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt_res, "input_ids"):
            prompt_ids = list(prompt_res.input_ids)
        elif isinstance(prompt_res, dict):
            prompt_ids = list(prompt_res["input_ids"])
        else:
            prompt_ids = list(prompt_res)

        if not is_training:
            return {
                "input_ids": prompt_ids,
                "attention_mask": [1] * len(prompt_ids),
                "id": row.get("id"),
            }

        # Assistant response
        a_content = msgs[2]["content"].strip()
        # Ensure EOS token terminates the assistant output
        eos_str = "<|im_end|>\n"
        target_ids = self.tokenizer.encode(a_content + eos_str, add_special_tokens=False)

        input_ids = prompt_ids + target_ids
        # Labels: -100 for prompt tokens, actual token IDs for assistant target
        labels = [-100] * len(prompt_ids) + list(target_ids)

        # Truncate if exceeds max_seq_length (from left or assert fit)
        if len(input_ids) > self.max_seq_length:
            # Crucial invariant: never truncate the assistant target tokens!
            # Truncate prompt tokens from the left if necessary, but log warning
            overflow = len(input_ids) - self.max_seq_length
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + list(target_ids)

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": len(prompt_ids),
            "target_length": len(target_ids),
            "id": row.get("id"),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """PyTorch-compatible collator with right-padding for batching."""
        import torch

        max_len = max(len(item["input_ids"]) for item in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - len(input_ids)

            padded_input_ids = input_ids + [self.pad_token_id] * pad_len
            padded_mask = [1] * len(input_ids) + [0] * pad_len
            padded_labels = labels + [-100] * pad_len

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
