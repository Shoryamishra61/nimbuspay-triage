"""Real end-to-end PEFT training pipeline and CLI.
Executes genuine optimizer training with assistant-only loss supervision,
gradient clipping, memory profiling, and standard PEFT adapter export.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import time
import json
import argparse
import hashlib
import yaml
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from .source_io import load_jsonl, compute_sha256
from .tokenizer_pipeline import NimbusChatFormatter
from .provenance import get_environment_info


def count_parameters(model) -> Tuple[int, int, float]:
    """Calculate and return (trainable_params, all_params, trainable_percentage)."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    return trainable_params, all_param, (100.0 * trainable_params / all_param) if all_param > 0 else 0.0


def setup_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    precision: str = "qlora4",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    device: Optional[str] = None,
):
    """Load base model and wrap with LoRA adapter."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "qlora4" and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=bnb_config,
            device_map="auto",
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=compute_dtype,
        )
        base_model.to(device)

    # Freeze base model parameters
    for p in base_model.parameters():
        p.requires_grad = False

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
    print(f"LoRA initialized: rank={lora_rank}, alpha={lora_alpha}")
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({pct:.3f}%)")

    # Hard assignment constraint: <= 50M trainable parameters
    assert trainable_params <= 50_000_000, f"Trainable params {trainable_params} exceed 50M limit!"

    return model, tokenizer, lora_config


def train_model(
    config: Dict[str, Any],
    output_dir: str,
    smoke_test_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute real optimizer fine-tuning loop and save PEFT adapter."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model and tokenizer
    model, tokenizer, lora_cfg = setup_model_and_tokenizer(
        model_id=config.get("model_id", "Qwen/Qwen2.5-1.5B-Instruct"),
        revision=config.get("revision", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
        precision=config.get("precision", "qlora4"),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules"),
        device=device,
    )

    # 2. Prepare data
    train_path = config.get("training_data", "artifacts/cleaned.jsonl")
    train_rows = load_jsonl(train_path)

    formatter = NimbusChatFormatter(tokenizer, max_seq_length=config.get("max_seq_length", 2304))
    encoded_dataset = [formatter.format_row(r, is_training=True) for r in train_rows]

    # Batching parameters
    microbatch_size = config.get("per_device_train_batch_size", 1)
    grad_accum_steps = config.get("gradient_accumulation_steps", 8)
    lr = float(config.get("learning_rate", 2e-4))
    epochs = config.get("epochs", 2)

    total_samples = len(encoded_dataset)
    steps_per_epoch = total_samples // (microbatch_size * grad_accum_steps)
    total_optimizer_steps = steps_per_epoch * epochs

    if smoke_test_steps is not None:
        total_optimizer_steps = smoke_test_steps
        epochs = 1
        print(f"Running SMOKE TEST for {smoke_test_steps} optimizer steps")

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    warmup_steps = int(total_optimizer_steps * config.get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_optimizer_steps, 1),
    )

    # Record initial weights of a sample adapter parameter to prove training occurred
    initial_sample_tensor = trainable_params[0].detach().clone()

    print(f"Training started on device '{device}': {total_optimizer_steps} optimizer steps, effective batch {microbatch_size * grad_accum_steps}")
    start_time = time.time()
    model.train()

    loss_history = []
    optimizer_step_count = 0
    accumulated_loss = 0.0
    batch_idx = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Training loop
    for epoch in range(epochs):
        for i in range(0, total_samples, microbatch_size):
            if optimizer_step_count >= total_optimizer_steps:
                break

            batch_items = encoded_dataset[i : i + microbatch_size]
            collated = formatter.collate_fn(batch_items)

            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum_steps
            accumulated_loss += loss.item() * grad_accum_steps

            if not torch.isfinite(loss):
                raise ValueError(f"Loss NaN or Inf at step {optimizer_step_count}: {loss.item()}")

            loss.backward()
            batch_idx += 1

            if batch_idx % grad_accum_steps == 0 or (i + microbatch_size >= total_samples):
                # Verify non-zero adapter gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                assert grad_norm > 0, "Gradients are zero! Model is not learning."

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                optimizer_step_count += 1
                avg_step_loss = accumulated_loss / grad_accum_steps
                loss_history.append({"step": optimizer_step_count, "loss": avg_step_loss, "grad_norm": float(grad_norm)})

                if optimizer_step_count % 10 == 0 or optimizer_step_count == total_optimizer_steps:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Step {optimizer_step_count}/{total_optimizer_steps} | Loss: {avg_step_loss:.4f} | Grad: {grad_norm:.3f} | Elapsed: {elapsed:.2f}m")

                accumulated_loss = 0.0

    elapsed_minutes = (time.time() - start_time) / 60.0

    # Prove that weights genuinely changed via optimization
    final_sample_tensor = trainable_params[0].detach().clone()
    weight_delta = (final_sample_tensor - initial_sample_tensor).abs().sum().item()
    print(f"Weight delta check: {weight_delta:.8f} (assert > 0)")
    assert weight_delta > 0, "Adapter weights did NOT change! Training was a no-op."

    # Record peak memory
    peak_vram_gib = 0.0
    if torch.cuda.is_available():
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

    # Save PEFT adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"PEFT adapter saved to: {adapter_dir}")

    # Build and persist run manifest
    manifest = {
        "status": "COMPLETED",
        "model_id": config.get("model_id"),
        "revision": config.get("revision"),
        "precision": config.get("precision"),
        "lora_rank": config.get("lora_rank"),
        "lora_alpha": config.get("lora_alpha"),
        "optimizer_steps_executed": optimizer_step_count,
        "training_minutes": elapsed_minutes,
        "peak_vram_gib": peak_vram_gib,
        "weight_delta": weight_delta,
        "loss_trace": loss_history[:20],
        "final_loss": loss_history[-1]["loss"] if loss_history else None,
        "environment": get_environment_info(),
        "adapter_config_path": os.path.join(adapter_dir, "adapter_config.json"),
        "adapter_weights_path": os.path.join(adapter_dir, "adapter_model.safetensors"),
        "adapter_sha256": compute_sha256(os.path.join(adapter_dir, "adapter_model.safetensors")),
    }

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest saved to: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="NimbusPay PEFT Trainer CLI")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoint and manifest")
    parser.add_argument("--smoke-test", type=int, default=None, help="If set, run only N steps to verify pipeline")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_model(config, args.output_dir, smoke_test_steps=args.smoke_test)


if __name__ == "__main__":
    main()
