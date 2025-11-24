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

"""Qwen3-VL Training Entry Point - Multi-Model Support with LoRA/Full Fine-Tuning"""

# <claudes_code_comments>
# ** Function List **
# rank0_print: Print only on rank 0 for distributed training
# safe_save_model_for_hf_trainer: Save model with DeepSpeed compatibility
# set_model: Configure trainable parameters (vision/merger/LLM)
# train: Main training function with model selection and training loop
#
# ** Technical Review **
# MAIN TRAINING SCRIPT supporting 4 Qwen-VL variants with flexible fine-tuning modes.
#
# SUPPORTED MODELS:
# 1. Qwen2-VL (original): Qwen2VLForConditionalGeneration
# 2. Qwen2.5-VL (improved): Qwen2_5_VLForConditionalGeneration
# 3. Qwen3-VL (latest): Qwen3VLForConditionalGeneration
# 4. Qwen3-VL-MoE (sparse): Qwen3VLMoeForConditionalGeneration
#
# MODEL SELECTION LOGIC:
# - Detection by model_name_or_path string matching:
#   * "qwen3" + "a" in filename → Qwen3-VL-MoE (e.g., "Qwen3-VL-235B-A22B-Instruct")
#   * "qwen3" → Qwen3-VL
#   * "qwen2.5" → Qwen2.5-VL
#   * else → Qwen2-VL
# - Sets data_args.model_type for correct RoPE implementation
#
# FINE-TUNING MODES:
# 1. Full Fine-Tuning (default):
#    - set_model() configures trainable components via model_args:
#      * tune_mm_vision: Train vision encoder blocks
#      * tune_mm_mlp: Train merger (MLP projector)
#      * tune_mm_llm: Train language model + lm_head
#    - Typical: tune_mm_mlp=True, others=False (only train projector)
#
# 2. LoRA Fine-Tuning (lora_enable=True):
#    - Freezes all parameters, adds LoRA adapters to attention
#    - Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
#    - Config: r=64, alpha=128, dropout=0.05 (default)
#    - Uses PEFT library (get_peft_model)
#    - Much lower memory: ~4GB for 7B model vs ~28GB full fine-tuning
#
# FLASH ATTENTION 2 INTEGRATION:
# - attn_implementation="flash_attention_2" for all models
# - Requires CUDA, NVIDIA GPUs with compute capability >= 8.0 (A100/H100)
# - If data_flatten or data_packing enabled:
#   * Calls replace_qwen2_vl_attention_class() to enable varlen FA2
#   * Standard FA2 wastes memory on padding, varlen packs sequences
#
# GRADIENT CHECKPOINTING:
# - Enabled via training_args.gradient_checkpointing
# - Trades compute for memory: recomputes activations during backward
# - Essential for large models (>7B) on consumer GPUs
# - Registers forward hook: make_inputs_require_grad for embedding layer
#
# TRAINING ARGUMENTS:
# - ModelArguments: model_name_or_path, tune flags
# - DataArguments: dataset_use, pixel budgets, packing config
# - TrainingArguments: Standard HF args + custom:
#   * mm_projector_lr: Learning rate for merger (typically 10× base LR)
#   * vision_tower_lr: Learning rate for vision encoder (typically 0.5× base LR)
#   * lora_r, lora_alpha, lora_dropout: LoRA hyperparameters
#
# CHECKPOINT RESUMPTION:
# - Scans output_dir for checkpoint-* folders
# - If found: trainer.train(resume_from_checkpoint=True)
# - Automatically continues from last saved state
#
# SAVING STRATEGY:
# - safe_save_model_for_hf_trainer handles DeepSpeed zero3 state_dict collection
# - Saves model weights to output_dir
# - Saves processor (tokenizer + image/video processors) alongside
# - Sets model.config.use_cache=False during training, True after
#
# DISTRIBUTED TRAINING:
# - Uses transformers.Trainer with native DDP/FSDP/DeepSpeed support
# - local_rank from training_args for process identification
# - rank0_print for clean logging
# - print_trainable_parameters() on rank 0 to show parameter status
#
# DATA FLOW:
# 1. Load model + processor
# 2. Configure trainable parameters (full/LoRA)
# 3. make_supervised_data_module creates dataset + collator
# 4. Trainer handles optimization, gradient accumulation, mixed precision
# 5. Save final model + processor
#
# INTEGRATION WITH OTHER FILES:
# - trainer.py: Provides create_optimizer, print_trainable_parameters
# - data_processor.py: Provides make_supervised_data_module
# - argument.py: Defines all argument dataclasses
# - rope2d.py: Position ID generation for M-RoPE
#
# TYPICAL COMMAND:
# torchrun --nproc_per_node=8 train_qwen.py \
#   --model_name_or_path Qwen/Qwen3-VL-7B-Instruct \
#   --dataset_use caption,vqa \
#   --output_dir ./outputs \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 2e-5 \
#   --mm_projector_lr 2e-4 \
#   --tune_mm_mlp True \
#   --bf16 True \
#   --data_packing True \
#   --gradient_checkpointing True
# </claudes_code_comments>

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
