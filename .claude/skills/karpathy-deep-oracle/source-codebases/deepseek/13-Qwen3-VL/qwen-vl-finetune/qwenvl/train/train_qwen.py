"""Qwen-VL Fine-Tuning Entry Point with Multi-Model Support"""

# <claudes_code_comments>
# ** Function List **
# rank0_print: Print only from rank 0 in distributed training
# safe_save_model_for_hf_trainer: Save model with DeepSpeed/CPU state dict handling
# set_model: Freeze/unfreeze vision/merger/LLM components based on args
# train: Main training function (model load → freeze → data → trainer → save)
#
# ** Technical Review **
# TRAINING ENTRY POINT for all Qwen-VL model variants with flexible component tuning.
#
# SUPPORTED MODELS (auto-detected from model path):
# 1. Qwen3-VL-MoE: Path contains "qwen3" AND "a" (e.g., "Qwen3-VL-30B-A3B-Instruct")
# 2. Qwen3-VL: Path contains "qwen3" (e.g., "Qwen3-VL-8B-Instruct")
# 3. Qwen2.5-VL: Path contains "qwen2.5" (e.g., "Qwen2.5-VL-7B-Instruct")
# 4. Qwen2-VL: Default fallback (e.g., "Qwen2-VL-2B-Instruct")
#
# MODEL ARCHITECTURE COMPONENTS:
# - visual: Vision tower (ViT encoder + DeepStack feature fusion)
# - visual.merger: Multi-level feature merging module (vision → LLM bridge)
# - language_model: Decoder-only transformer (main LLM)
# - lm_head: Final linear layer (vocab projection)
#
# TRAINING MODES (via model_args flags):
#
# Mode 1: Full fine-tuning (all True):
# - tune_mm_vision=True: Unfreeze visual tower (ViT blocks)
# - tune_mm_mlp=True: Unfreeze merger (adapter between vision and LLM)
# - tune_mm_llm=True: Unfreeze language model + lm_head
# - Use case: Small datasets, domain adaptation, all components need updates
# - Memory: Highest (all gradients computed)
#
# Mode 2: Vision + Merger frozen, LLM tuned:
# - tune_mm_vision=False
# - tune_mm_mlp=False
# - tune_mm_llm=True
# - Use case: Large datasets, vision already good, focus on language reasoning
# - Memory: Medium (only LLM gradients)
#
# Mode 3: Merger + LLM tuned, Vision frozen:
# - tune_mm_vision=False
# - tune_mm_mlp=True
# - tune_mm_llm=True
# - Use case: Most common! Vision features OK, adapt bridge and LLM
# - Memory: Medium-high
#
# Mode 4: LoRA fine-tuning (lora_enable=True):
# - All components frozen
# - Low-rank adapters added to attention (q_proj, k_proj, v_proj, o_proj)
# - Use case: Limited GPU memory, quick adaptation
# - Memory: Lowest (only LoRA weights trainable)
# - LoRA config (defaults):
#   * r=64: Rank of low-rank matrices
#   * lora_alpha=128: Scaling factor (alpha/r = 2.0)
#   * lora_dropout=0.05: Dropout for LoRA layers
#   * target_modules: Only attention projections (not MLP, not merger)
#
# FLASH ATTENTION INTEGRATION:
# - Enabled when data_flatten=True OR data_packing=True
# - Calls replace_qwen2_vl_attention_class() from trainer.py
# - Monkey-patches all model variants for varlen packing
# - Requires: attn_implementation="flash_attention_2" (default in train function)
#
# GRADIENT CHECKPOINTING:
# - gradient_checkpointing=True: Trade compute for memory
# - Recomputes activations during backward pass instead of storing
# - Reduces memory by ~40% at cost of ~20% slower training
# - Implementation:
#   * enable_input_require_grads() if available (newer models)
#   * Otherwise: register forward hook on embeddings
#
# MODEL SAVING STRATEGY:
# - DeepSpeed: Use trainer.save_model (DeepSpeed handles state dict)
# - Standard: Move state dict to CPU before saving (prevents GPU OOM)
# - Saves to output_dir: model.safetensors + config.json + processor config
# - Also saves processor: processor_config.json + image_processor_config.json
#
# CHECKPOINT RESUMPTION:
# - Auto-detects checkpoint-* folders in output_dir
# - If found: trainer.train(resume_from_checkpoint=True)
# - Loads: optimizer state, scheduler state, RNG states, global_step
# - Useful for: Preemptible instances, failed runs, multi-stage training
#
# TYPICAL COMMAND-LINE USAGE:
# python train_qwen.py \
#   --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
#   --data_path /data/my_dataset.jsonl \
#   --output_dir ./checkpoints/qwen3vl-finetuned \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 1 \  # Flash attention uses batch_size=1 packing
#   --gradient_accumulation_steps 8 \   # Effective batch size = 1×8 = 8
#   --learning_rate 1e-5 \
#   --mm_projector_lr 5e-5 \            # Higher LR for merger
#   --bf16 True \
#   --tune_mm_vision False \            # Freeze vision
#   --tune_mm_mlp True \                # Tune merger
#   --tune_mm_llm True \                # Tune LLM
#   --data_flatten True \               # Enable flash attention packing
#   --gradient_checkpointing True \     # Save memory
#   --deepspeed ./ds_config.json        # DeepSpeed ZeRO config (optional)
#
# TRAINING FLOW:
# 1. Parse arguments (model_args, data_args, training_args)
# 2. Detect model variant from path → load appropriate class
# 3. Load processor (handles tokenization + vision preprocessing)
# 4. Apply flash attention patches if data_flatten/data_packing
# 5. Disable use_cache (incompatible with gradient checkpointing)
# 6. Enable gradient checkpointing if requested
# 7. Load tokenizer (model_max_length from training_args)
# 8. Apply LoRA if enabled, otherwise freeze/unfreeze components via set_model
# 9. Print trainable parameters (verify freezing worked)
# 10. Create data module (dataset + collator)
# 11. Create HuggingFace Trainer
# 12. Train (with auto-resume if checkpoints exist)
# 13. Save final model + processor
#
# KEY ARGUMENTS (from argument.py):
# ModelArguments:
# - model_name_or_path: HuggingFace model ID or local path
# - tune_mm_vision: Train vision tower (default False)
# - tune_mm_mlp: Train merger (default True)
# - tune_mm_llm: Train LLM (default True)
#
# DataArguments:
# - data_path: JSONL file(s) with conversations
# - data_flatten: Enable flash attention packing (default False)
# - min_pixels/max_pixels: Image resolution bounds
# - video_min_pixels/video_max_pixels: Video per-frame resolution
#
# TrainingArguments:
# - output_dir: Save checkpoints here
# - lora_enable: Use LoRA instead of full fine-tuning (default False)
# - lora_r/lora_alpha/lora_dropout: LoRA hyperparameters
# - mm_projector_lr: Learning rate for merger (default None = use base LR)
# - vision_tower_lr: Learning rate for vision (default None = frozen)
# - gradient_checkpointing: Enable activation checkpointing (default False)
# - bf16: Use bfloat16 mixed precision (recommended for A100/H100)
# - deepspeed: DeepSpeed config path (for ZeRO optimization)
#
# GOTCHAS:
# - Flash attention requires batch_size=1 with packing (handle via gradient_accumulation)
# - use_cache must be False during training (incompatible with grad checkpointing)
# - LoRA only applied to attention layers, not merger or vision tower
# - Model detection based on path string (case-insensitive substring match)
# - Processor saves to output_dir (needed for inference with same resolution settings)
#
# </claudes_code_comments>

# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
