# Fine-Tuning Overview: Training Qwen3-VL

**Complete guide to fine-tuning Qwen3-VL for custom vision-language tasks**

## Philosophy: When to Fine-Tune vs Pre-Train

From the codebase README:
> "We provide the [code](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune) for fine-tuning Qwen2-VL and Qwen2.5-VL."

**Fine-Tuning** (what we provide):
- Adapt pre-trained model to specific tasks
- Requires 100s-1000s examples
- Trains LoRA adapters or full model
- Days-weeks on multi-GPU setup

**Pre-Training** (not covered):
- Build vision-language understanding from scratch
- Requires millions of image-text pairs
- Months of training on thousands of GPUs
- Not practical for most users

**Use fine-tuning when:**
- You have task-specific data (medical images, industrial inspection, custom documents)
- You want to specialize behavior (formatting, output style, domain terminology)
- You need to improve performance on specific benchmarks

## What Can You Train?

Qwen3-VL's architecture has **three trainable components**:

### 1. Vision Encoder (ViT)
**Location**: Vision tower extracting image features

**When to train**:
- ✅ New visual domain (medical scans, satellite imagery)
- ✅ Different image quality/style than pre-training
- ❌ General images (pre-trained weights already excellent)

**Training cost**: **High** (largest component, many layers)

### 2. MLP Projection Layers
**Location**: Maps vision features → LLM embedding space

**When to train**:
- ✅ Always (lightweight, high impact)
- ✅ Adapting to new vision encoder
- ✅ Fine-tuning visual-text alignment

**Training cost**: **Low** (small layer, few parameters)

### 3. Language Model (Qwen3)
**Location**: Text generation backbone

**When to train**:
- ✅ Domain-specific terminology (legal, medical, technical)
- ✅ Output format requirements (JSON, markdown, custom templates)
- ❌ General language (pre-trained LLM is very capable)

**Training cost**: **Very High** (billions of parameters)

## Training Strategies

### Strategy 1: LoRA (Low-Rank Adaptation)
**Best for**: Limited compute, quick experiments

```bash
# Example from scripts/sft_30a3b_lora.sh
torchrun --nproc_per_node=8 train_qwen.py \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct \
    --use_lora True \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --train_modules q_proj v_proj \  # Which layers to adapt
    ...
```

**Advantages**:
- 10-100x fewer parameters to update
- Fits on smaller GPUs (24GB can train 7B model)
- Faster training (2-3x speedup)
- Easy to merge/switch adapters

**Disadvantages**:
- Slightly lower final quality than full fine-tuning
- Can't radically change model behavior

### Strategy 2: Full Fine-Tuning
**Best for**: Maximum quality, sufficient compute

```bash
# Example from scripts/sft_7b.sh
torchrun --nproc_per_node=8 train_qwen.py \
    --model_name_or_path Qwen/Qwen3-VL-7B-Instruct \
    --use_lora False \  # Full parameter training
    --freeze_vision False \  # Train vision encoder
    --freeze_llm False \      # Train language model
    ...
```

**Advantages**:
- Maximum adaptation capacity
- Can fundamentally alter behavior
- Better final benchmarks

**Disadvantages**:
- Requires large GPU clusters (8× A100/H100)
- Slow training (days-weeks)
- Higher risk of overfitting

### Strategy 3: Mixed Approach (Recommended)
**Best for**: Balancing quality and efficiency

```python
# Freeze vision encoder, train MLP + LLM with LoRA
--freeze_vision True \       # Keep pre-trained ViT
--freeze_llm False \         # Train LLM
--use_lora True \            # LoRA for LLM
--train_mlp True \           # Full training for projection
```

**Rationale**:
- Vision encoder rarely needs changes (pre-training is excellent)
- MLP is cheap to train fully (high ROI)
- LoRA on LLM gives 80% of full fine-tuning quality at 20% cost

## Resolution Strategies

Qwen3-VL supports **dynamic resolution** during training:

### Strategy A: Fixed Resolution
```python
# Simple, predictable memory usage
min_pixels = 256 * 28 * 28  # ~200k pixels
max_pixels = 256 * 28 * 28  # Same as min
```

**Pros**: Consistent batch sizes, easy to debug
**Cons**: Doesn't match variable-resolution inference

### Strategy B: Variable Resolution
```python
# More realistic, better generalization
min_pixels = 64 * 28 * 28   # ~50k pixels (low res)
max_pixels = 1024 * 28 * 28 # ~800k pixels (high res)
```

**Pros**: Trains model to handle diverse inputs
**Cons**: Variable memory usage, longer training

**Best practice**: Start with fixed, switch to variable once stable

## Multi-Modal Data Packing

Qwen3-VL implements **efficient data packing**:

```python
# Without packing: Each sample = separate batch
[image1 + query1], [image2 + query2], ...
→ GPU utilization: ~60% (padding waste)

# With packing: Multiple samples per batch
[image1 + query1 + image2 + query2 + image3 + query3]
→ GPU utilization: ~95% (minimal padding)
```

**Configuration**:
```bash
--packing True \            # Enable packing
--max_seq_length 8192 \     # Total sequence length
--pack_dataset_break_mode no_break_longest_only  # Packing strategy
```

**Impact**: 30-50% faster training, same final quality

## Training Hyperparameters

**Critical settings** (from `scripts/sft_7b.sh`):

```bash
# Learning rate
--learning_rate 5e-5 \              # LoRA: 1e-4, Full: 5e-5
--lr_scheduler_type cosine \         # Smooth decay
--warmup_ratio 0.03 \                # 3% warmup steps

# Batch size (effective = per_device × gradient_accum × num_gpus)
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
# → Effective batch size = 1 × 8 × 8 GPUs = 64

# Optimization
--bf16 True \                        # Use bfloat16 precision
--optim adamw_torch \                # Optimizer choice
--weight_decay 0.01 \                # Regularization

# Vision-specific
--freeze_vision False \              # Train vision encoder?
--train_mlp True \                   # Train projection?
```

**Key trade-offs**:
- **Higher LR** → Faster convergence, risk instability
- **Larger batch** → More stable, requires more memory
- **More epochs** → Better fit, risk overfitting

## Hardware Requirements

**Minimum** (LoRA on 7B model):
- 1× A100 (80GB) or 4× A6000 (48GB each)
- ~200GB disk space
- Training time: 1-2 days for 10k examples

**Recommended** (Full fine-tuning on 7B):
- 8× A100 (80GB each)
- ~500GB disk space
- Training time: 3-5 days for 50k examples

**Large-scale** (30B+ models):
- 16-32× H100 (80GB each)
- ~2TB disk space
- Training time: 1-2 weeks

## DeepSpeed Integration

All training scripts use **DeepSpeed** for distributed training:

```bash
# ZeRO Stage 2: Optimizer state sharding
--deepspeed scripts/zero2.json

# ZeRO Stage 3: Full parameter sharding (largest models)
--deepspeed scripts/zero3.json

# ZeRO Stage 3 + CPU offload: Extreme memory savings
--deepspeed scripts/zero3_offload.json
```

**ZeRO configurations**:
- **Stage 2**: Good for 7B-30B models, ~30% memory reduction
- **Stage 3**: Required for 70B+ models, ~70% memory reduction
- **Stage 3 + Offload**: Can train 70B on 8× 40GB GPUs

## Common Issues and Solutions

### 1. Out of Memory (OOM)
**Symptoms**: CUDA OOM error during training

**Solutions**:
```bash
# Reduce batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \  # Increase to maintain effective batch

# Use gradient checkpointing
--gradient_checkpointing True \

# Switch to ZeRO Stage 3
--deepspeed scripts/zero3_offload.json
```

### 2. NaN Loss
**Symptoms**: Loss suddenly becomes NaN

**Solutions**:
```bash
# Lower learning rate
--learning_rate 1e-5 \  # Was 5e-5

# Use fp32 for layer norm
--bf16_full_eval False \

# Check data quality (corrupted images, invalid text)
```

### 3. Slow Convergence
**Symptoms**: Loss decreases very slowly

**Solutions**:
```bash
# Unfreeze more components
--freeze_vision False \  # Was True
--freeze_llm False \     # Was True

# Increase learning rate (carefully)
--learning_rate 1e-4 \  # Was 5e-5

# Check data diversity (too homogeneous?)
```

## Evaluation During Training

**Built-in logging**:
```bash
--logging_steps 10 \              # Log every 10 steps
--eval_steps 500 \                # Evaluate every 500 steps
--save_steps 500 \                # Save checkpoint every 500 steps
```

**Monitored metrics**:
- **Training loss**: Should decrease smoothly
- **Evaluation loss**: Should track training loss (not diverge)
- **Perplexity**: Lower = better language modeling
- **GPU utilization**: Should stay >80%

## Post-Training: Merging LoRA Weights

After LoRA training, merge adapters into base model:

```python
from transformers import AutoModel
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained("Qwen/Qwen3-VL-7B-Instruct")

# Load LoRA adapter
lora_model = PeftModel.from_pretrained(base_model, "path/to/lora/checkpoint")

# Merge and save
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("path/to/merged/model")
```

## Related Documentation

- [01-dataset-format.md](01-dataset-format.md) - Data preparation
- [02-training-script.md](02-training-script.md) - Detailed `train_qwen.py` guide
- [../usage/01-inference.md](../usage/01-inference.md) - Using fine-tuned models
- [../codebase/03-data-processor.md](../codebase/03-data-processor.md) - Data loading internals

## Quick Reference

**Start here**:
1. Prepare data in JSON format ([01-dataset-format.md](01-dataset-format.md))
2. Choose strategy (LoRA vs Full)
3. Select script template (`scripts/sft_7b.sh` or `scripts/sft_30a3b_lora.sh`)
4. Adjust hyperparameters for your hardware
5. Run training with DeepSpeed
6. Evaluate on held-out set
7. Merge LoRA (if applicable)

**Key decision tree**:
```
Do you have 8× A100 (80GB)?
├─ Yes → Full fine-tuning (best quality)
└─ No → LoRA (80% quality, 20% cost)

Is vision domain very different from pre-training?
├─ Yes → Unfreeze vision encoder
└─ No → Freeze vision (default)

Need fast turnaround?
├─ Yes → LoRA + frozen vision
└─ No → Full training
```

---

**Last Updated**: 2025-10-28
**Status**: Complete practical guide
**Importance**: ⭐⭐⭐⭐⭐ (Critical for customization)
