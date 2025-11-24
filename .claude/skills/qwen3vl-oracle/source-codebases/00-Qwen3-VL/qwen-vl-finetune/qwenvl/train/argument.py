"""Qwen3-VL Training Arguments - Configuration Dataclasses"""

# <claudes_code_comments>
# ** Class List **
# ModelArguments: Model selection and component trainability flags
# DataArguments: Dataset configuration and dynamic resolution settings
# TrainingArguments: Extended HF TrainingArguments with multi-LR and LoRA support
#
# ** Technical Review **
# CONFIGURATION CLASSES for Qwen3-VL fine-tuning with three distinct scopes:
#
# 1. MODEL ARGUMENTS (ModelArguments):
#    - model_name_or_path: HuggingFace model ID or local path
#      * Examples: "Qwen/Qwen3-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"
#      * Supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE variants
#    - Trainability flags (all False by default = freeze everything):
#      * tune_mm_llm: Train language model layers and lm_head
#      * tune_mm_mlp: Train merger (MLP projector) between vision and LLM
#      * tune_mm_vision: Train vision encoder (ViT) blocks
#    - Typical configurations:
#      * Projector-only: tune_mm_mlp=True (fastest, lowest memory)
#      * Full fine-tuning: All three=True (highest quality, most memory)
#      * Vision + Projector: tune_mm_vision=True, tune_mm_mlp=True
#
# 2. DATA ARGUMENTS (DataArguments):
#    a) Dataset Configuration:
#       - dataset_use: Comma-separated dataset names (e.g., "caption,vqa,ocr")
#       - Maps to dataset configs via data_list() function
#
#    b) Sequence Packing (Critical for efficiency):
#       - data_flatten: Enable flattened sequences (required for varlen FA2)
#       - data_packing: Enable multi-sample packing (reduces padding waste)
#       - When True: FlattenedDataCollatorForSupervisedDataset + cu_seqlens attention
#       - Memory savings: 25-40% by eliminating padding
#
#    c) Dynamic Resolution for Images:
#       - max_pixels: 28×28×576 = 451,584 pixels (default, ~672×672)
#       - min_pixels: 28×28×16 = 12,544 pixels (~112×112)
#       - Factor of 28: patch_size(14) × spatial_merge(2)
#       - Controls smart_resize in vision_process.py
#       - Higher max_pixels → more detail, more tokens, more memory
#
#    d) Video Configuration:
#       - video_max_frames: 8 frames (default)
#       - video_min_frames: 4 frames
#       - video_max_pixels: 1024×28×28 = 802,816 pixels per frame
#       - video_min_pixels: 256×28×28 = 200,704 pixels per frame
#       - video_fps: 2.0 (sample 2 frames per second)
#       - base_interval: 2 (temporal grouping stride)
#       - Total video budget shared across all frames!
#
#    e) ARR-COC Integration Point:
#       - max_pixels/min_pixels are UNIFORM budgets currently
#       - ARR-COC enhancement: Make these query-aware and patch-specific
#       - High-relevance patches: Increase local max_pixels
#       - Low-relevance patches: Decrease local max_pixels
#
# 3. TRAINING ARGUMENTS (Extended transformers.TrainingArguments):
#    a) Standard HF Arguments (inherited):
#       - learning_rate, num_train_epochs, per_device_train_batch_size
#       - gradient_accumulation_steps, warmup_steps, weight_decay
#       - fp16, bf16, gradient_checkpointing, deepspeed
#
#    b) Custom Extensions:
#       - cache_dir: Cache for downloaded models
#       - optim: "adamw_torch" (default), or "adamw_8bit", "adafactor"
#       - model_max_length: 512 (default), increase for long contexts
#
#    c) Multi-Learning-Rate Support:
#       - mm_projector_lr: LR for merger (typically 10× base LR)
#       - vision_tower_lr: LR for vision encoder (typically 0.5× base LR)
#       - None = use base learning_rate for all components
#       - Implemented in trainer.py create_optimizer()
#
#    d) LoRA Configuration:
#       - lora_enable: False (default = full fine-tuning)
#       - lora_r: 64 (rank of low-rank matrices)
#       - lora_alpha: 128 (scaling factor, typically 2× rank)
#       - lora_dropout: 0.0 (dropout in LoRA layers)
#       - Applied to ["q_proj", "k_proj", "v_proj", "o_proj"]
#
# TYPICAL CONFIGURATIONS:
#
# Projector-Only (Fastest):
# ModelArguments(tune_mm_mlp=True)
# - Memory: ~16GB for 7B model
# - Quality: Good for task adaptation
#
# Full Fine-Tuning (Best Quality):
# ModelArguments(tune_mm_llm=True, tune_mm_mlp=True, tune_mm_vision=True)
# - Memory: ~32GB for 7B model with gradient_checkpointing
# - Quality: Best for domain shift
#
# LoRA (Most Efficient):
# TrainingArguments(lora_enable=True, lora_r=64, lora_alpha=128)
# - Memory: ~12GB for 7B model
# - Quality: Good compromise
#
# High-Resolution Images:
# DataArguments(max_pixels=28*28*1024)  # ~1024×1024
# - More visual detail but more memory
#
# Sequence Packing (Recommended):
# DataArguments(data_packing=True, data_flatten=True)
# - 25-40% memory savings from eliminating padding
# - Requires Flash Attention 2 varlen
#
# Multi-LR Optimization:
# TrainingArguments(
#     learning_rate=2e-5,        # Base LR for LLM
#     mm_projector_lr=2e-4,      # 10× for merger
#     vision_tower_lr=1e-5       # 0.5× for vision
# )
# - Faster convergence by matching component learning needs
# </claudes_code_comments>

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
