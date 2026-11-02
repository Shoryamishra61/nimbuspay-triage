"""Training and PEFT fine-tuning module.
Implements LoRA / QLoRA adapters for Qwen2.5-1.5B-Instruct adhering to:
- <=50M trainable parameters assertion
- Standard PEFT output format (adapter_config.json + adapter_model.safetensors)
- Assistant-only loss supervision
- Gradient checkpointing and memory logging for Colab T4 compatibility
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import time
import json
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from .tokenizer_pipeline import NimbusChatFormatter


def count_parameters(model) -> Tuple[int, int, float]:
    """Return (trainable_params, all_params, trainable_percentage)."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_param, 100 * trainable_params / all_param


def build_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    precision: str = "qlora4",  # 'qlora4' or 'fp16'
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Load model with specified quantization and wrap with LoRA adapter."""
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None

    if precision == "qlora4" and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
        )
    else:
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map=device_map,
        )

    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    trainable_params, all_params, pct = count_parameters(model)
    print(f"LoRA configured: rank={lora_rank}, alpha={lora_alpha}")
    print(f"Trainable params: {trainable_params:,} / {all_params:,} ({pct:.3f}%)")

    # Hard constraint: trainable parameters <= 50,000,000
    assert trainable_params <= 50_000_000, f"Trainable params {trainable_params} exceed 50M limit!"

    return model, tokenizer, lora_config


def save_peft_adapter(model, tokenizer, output_dir: str):
    """Save adapter in standard PEFT format."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"PEFT adapter saved to {output_dir}")
