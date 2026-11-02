"""Strict minimal generation module.
Follows the exact evaluator contract:
- Exactly system and user messages with native chat template
- model.generate with do_sample=False, num_beams=1, max_new_tokens=200
- ONLY postprocessing: .strip()
- STRICTLY NO regex replacement, JSON repair, fallback, or retry logic.
- STRICTLY NO IMPORTS of evidence.py or data-repair functions.
"""

from typing import List, Dict, Any, Optional
import torch


def generate_single(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 200,
) -> str:
    """Generate raw triage prediction for a single message sequence."""
    # Ensure only system and user messages are passed
    system_user = messages[:2]

    # Format using model's native chat template
    prompt_ids = tokenizer.apply_chat_template(
        system_user,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract ONLY newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return raw_output


def generate_dataset_predictions(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    max_new_tokens: int = 200,
) -> List[Dict[str, str]]:
    """Generate raw predictions for an entire dataset.
    Returns list of dicts: {'id': ..., 'output': raw_output}
    """
    predictions = []
    for r in rows:
        raw_text = generate_single(
            model=model,
            tokenizer=tokenizer,
            messages=r["messages"],
            max_new_tokens=max_new_tokens,
        )
        predictions.append({
            "id": r["id"],
            "output": raw_text,
        })
    return predictions
