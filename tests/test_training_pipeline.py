"""Unit test for the training pipeline and weight optimization verification.
Instantiates an in-memory small causal LM to test backpropagation, gradient clipping,
weight updates, and standard PEFT adapter export without requiring external model downloads.
"""

import os
import shutil
import tempfile
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from src.triage.training import count_parameters


def test_tiny_model_training_and_adapter_export():
    tmp_dir = tempfile.mkdtemp(prefix="nimbus_test_train_")
    try:
        # Create tiny Qwen-architecture model in memory (2 layers, 64 hidden dim)
        cfg = AutoConfig.for_model(
            "qwen2",
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        base_model = AutoModelForCausalLM.from_config(cfg)

        # Freeze base parameters
        for p in base_model.parameters():
            p.requires_grad = False

        # Apply LoRA adapter
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_cfg)

        trainable_params, all_params, pct = count_parameters(peft_model)
        assert trainable_params > 0, "No trainable parameters found!"
        assert trainable_params < all_params, "Base model parameters not frozen!"

        trainable_list = [p for p in peft_model.parameters() if p.requires_grad]
        initial_sample = trainable_list[0].detach().clone()

        # Run 2 real optimization steps
        optimizer = torch.optim.AdamW(trainable_list, lr=1e-3)
        peft_model.train()

        for step in range(2):
            optimizer.zero_grad()
            input_ids = torch.randint(0, 1000, (2, 16))
            labels = input_ids.clone()
            outputs = peft_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            assert torch.isfinite(loss), "Loss was non-finite!"
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_list, max_norm=1.0)
            assert grad_norm > 0, "Gradient norm is zero!"
            optimizer.step()

        # Verify that weights genuinely changed
        final_sample = trainable_list[0].detach().clone()
        delta = (final_sample - initial_sample).abs().sum().item()
        assert delta > 0, f"Adapter weights did not change (delta={delta})!"

        # Save PEFT adapter
        adapter_dir = os.path.join(tmp_dir, "adapter")
        peft_model.save_pretrained(adapter_dir)

        # Assert exported files
        assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        assert os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

