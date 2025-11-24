# VLM Fine-Tuning Best Practices 2024-2025

Comprehensive guide to fine-tuning vision-language models in 2024-2025, covering parameter-efficient methods (LoRA, QLoRA, Spectrum), optimization strategies, architecture choices, and practical training recipes.

## Overview

VLM fine-tuning in 2024-2025 focuses on **efficiency and optimization** while maintaining high performance. Key developments include parameter-efficient fine-tuning (PEFT) methods, advanced optimizations (Flash Attention, Liger Kernels), and distributed training strategies.

**Key Trends 2024-2025:**
- **Smaller, more powerful models**: Llama 3.1 1B outperforms Llama 2 13B
- **PEFT dominance**: QLoRA and Spectrum for resource-efficient training
- **Optimization stack**: Flash Attention + Liger Kernels for 2-3x speedup
- **Distributed training**: DeepSpeed ZeRO3 for multi-GPU scaling
- **Synthetic data**: LLM-generated datasets using frameworks like Distilabel

## Architecture Design Choices

### Vision Encoding Strategies

**1. Pretrained Vision Encoder (Most Common)**
```
CLIP ViT → Projection/Resampler → LLM
```
- **Advantages**: Pre-aligned with text, faster training, proven performance
- **Popular choices**: CLIP ViT-L/14, SigLIP, EVA-CLIP
- **Used by**: LLaVA, Qwen-VL, Idefics 2, DeepSeek-VL

**2. Raw Image Patches (End-to-End)**
```
Raw patches → Linear projection → LLM
```
- **Advantages**: No information loss, variable resolution support
- **Disadvantages**: Requires more compute, longer training
- **Used by**: Fuyu-8B

**Trade-off**: Pretrained encoders win for most use cases due to training efficiency.

### Vision-Language Alignment Strategies

**1. Projection (Minimalistic)**
- **Method**: Linear layer or MLP maps vision → text embedding space
- **Token budget**: Fixed (e.g., 256 tokens from ViT-L/14)
- **Used by**: LLaVA (linear → MLP in v1.5/v1.6)
- **Best for**: Single images, high-detail tasks

**2. Resampling (Compression)**
- **Method**: Learnable queries + attention → fixed token budget
- **Variants**: Perceiver Resampler (Flamingo), Q-Former (BLIP-2)
- **Token budget**: Configurable (32-144 tokens typical)
- **Used by**: Qwen-VL, Kosmos-2, Idefics
- **Best for**: Video, multiple images, token budget constraints

**3. Text-Conditioned Resampling (Context-Aware)**
- **Method**: Resampling conditioned on input query
- **Used by**: BLIP-2 Q-Former, VideoChat
- **Best for**: Video QA, visual reasoning with specific queries

**Performance comparison (from Cha et al 2023):**
| Method | Tokens | VQAv2 | GQA | TextVQA |
|--------|--------|-------|-----|---------|
| Projection (MLP) | 256 | 78.5 | 62.0 | 58.3 |
| Perceiver Resampler | 64 | 76.3 | 60.1 | 54.2 |
| Q-Former | 32 | 78.9 | 61.9 | 56.7 |

### Multimodal Fusion Strategies

**1. Interleaved Vision-Language Tokens (Dominant 2024)**
```
[CLS] <img> <img> ... </img> User query <assistant>
```
- Process vision embeddings as text tokens
- No architecture modification needed
- **Used by**: LLaVA, Qwen-VL, DeepSeek-VL, Yi-VL

**2. Modality Experts**
- Vision/text processed by different experts in model
- **Used by**: CogVLM, BeiT-3
- **Trade-off**: More parameters, better modality-specific processing

**3. Cross-Attention (Deprecated 2024)**
- Vision tokens accessed via cross-attention between transformer blocks
- **Used by**: Flamingo, Idefics 1
- **Why deprecated**: Too many new parameters, interleaved tokens work better

## Parameter-Efficient Fine-Tuning (PEFT) Methods

### QLoRA (2024 Standard)

**What it is**: 4-bit quantization + LoRA adapters (Low-Rank Adaptation)

**Recipe:**
```yaml
# Quantization
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

# LoRA
lora_r: 16              # Rank (8-64 typical)
lora_alpha: 16          # Scaling factor (usually = r)
lora_dropout: 0.05      # Regularization
lora_target_modules: "all-linear"  # or specific layers
lora_modules_to_save: ["lm_head", "embed_tokens"]  # Train special tokens
```

**Performance (Llama 3.1 8B on GSM8K):**
- **Accuracy**: 54% (1 epoch, 10k samples)
- **Memory**: Fits on 1x L4 24GB
- **Training time**: 135 min (1 GPU), 18 min (8 GPUs with DeepSpeed)

**When to use**: Limited GPU memory, quick experimentation, adapter reuse

### Spectrum (2024 Advanced)

**What it is**: Selective layer fine-tuning using Signal-to-Noise Ratio (SNR) analysis

**Method**:
1. Analyze SNR of all layers on small dataset sample
2. Select top N% most informative layers (30% typical)
3. Fine-tune only those layers (no quantization)

**Recipe:**
```bash
# Generate SNR layer selection
python spectrum.py --model-name meta-llama/Llama-3.1-8B --top-percent 30
# Output: snr_results_meta-llama-Meta-Llama-3.1-8B_unfrozenparameters_30percent.yaml

# Train with Spectrum config
accelerate launch --config deepspeed_zero3.yaml --num_processes 8 \
  run_sft.py --config llama-3-1-8b-spectrum.yaml
```

**Performance (Llama 3.1 8B on GSM8K):**
- **Accuracy**: 58% (1 epoch) → 60% (3 epochs)
- **Memory**: 30-50GB single GPU, scales better with multi-GPU
- **Training time**: 21 min (8x L4 GPUs)
- **Improvement over QLoRA**: +4% accuracy

**When to use**: Production models, multi-GPU setups, want best accuracy

### Full Fine-Tuning (Baseline)

**When to use**:
- Have compute budget
- Need absolute best performance
- Small models (<7B parameters)

**Trade-offs**: 10-100x more memory, no adapter reuse, longer training

### Method Comparison

| Method | Memory (8B model) | Training Speed | Accuracy | Adapter Reuse |
|--------|-------------------|----------------|----------|---------------|
| Full FT | ~80GB | 1x | Baseline | ❌ |
| QLoRA | ~24GB | 0.8x | -2 to -5% | ✅ |
| Spectrum 30% | ~40GB (1 GPU), efficient multi-GPU | 0.9x | -1 to -2% | ❌ |

## Optimization Strategies

### Flash Attention 2 (Mandatory 2024)

**What**: Memory-efficient exact attention with fused kernels

**Impact**: 2-3x faster training, 50% memory reduction

```python
attn_implementation: "flash_attention_2"
```

**Requirements**: CUDA-capable GPU, `pip install flash-attn`

### Liger Kernels (2024 Addition)

**What**: Fused kernels for common LLM operations (RMSNorm, SwiGLU, cross-entropy)

**Impact**: +30-50% speedup, 20% memory reduction

```python
use_liger: true
```

**Source**: https://github.com/linkedin/Liger-Kernel

### Gradient Checkpointing

**What**: Trade compute for memory (recompute activations during backward pass)

```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false  # Better with Flash Attention
```

**Impact**: 40% memory reduction, 20% slower training

### Packing (Data Efficiency)

**What**: Concatenate multiple samples into single sequence up to max_seq_length

```yaml
packing: true
max_seq_length: 1024
```

**Impact**: 2-3x fewer optimization steps (better GPU utilization)

### Mixed Precision Training

```yaml
bf16: true  # bfloat16 (preferred on Ampere+)
tf32: true  # TensorFloat32 (automatic on Ampere+)
```

**Impact**: 2x faster, 50% memory reduction, minimal accuracy loss

## Training Recipes

### LLaVA-Style Recipe (Single Image QA)

```yaml
# Model
model: Meta-Llama/Meta-Llama-3.1-8B
tokenizer: Meta-Llama/Meta-Llama-3.1-8B-Instruct
vision_encoder: openai/clip-vit-large-patch14-336
alignment: mlp  # Multi-layer perceptron projection

# PEFT
use_peft: true
load_in_4bit: true
lora_r: 16
lora_alpha: 16
lora_target_modules: "all-linear"
lora_modules_to_save: ["lm_head", "embed_tokens"]

# Training
num_train_epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 2e-4
lr_scheduler_type: constant
warmup_ratio: 0.1
max_seq_length: 1024
packing: true

# Optimization
attn_implementation: flash_attention_2
use_liger: true
bf16: true
gradient_checkpointing: true
```

### Video Understanding Recipe (Multi-Frame)

```yaml
# Model
vision_encoder: openai/clip-vit-large-patch14-336
alignment: perceiver_resampler
num_query_tokens: 64  # Compress 256*N_frames → 64 tokens

# Handle multiple frames
max_num_frames: 8
frame_sampling: uniform  # or "fps"

# Larger context for video
max_seq_length: 2048
per_device_train_batch_size: 4  # Reduce due to longer sequences

# Rest similar to single image recipe
```

### Document Understanding Recipe (OCR-Free)

```yaml
# High-resolution vision
vision_encoder: google/siglip-so400m-patch14-384  # 384x384
image_resolution: 768  # Upsample for details
patch_pooling: adaptive  # Handle varying text densities

# Document-specific data augmentation
augmentation:
  - rotation: [-5, 5]  # Slight rotations
  - perspective: 0.1
  - jpeg_quality: [80, 100]

# Longer sequences for documents
max_seq_length: 4096
```

## Dataset Preparation

### Modern Dataset Creation (2024)

**1. Synthetic Generation with LLMs (Most Common)**
```python
from distilabel.llm import OpenAILLM
from distilabel.tasks import TextGenerationTask

# Generate high-quality synthetic data at scale
llm = OpenAILLM(model="gpt-4", max_new_tokens=512)
task = TextGenerationTask()

# Frameworks: Distilabel, Argilla, LLM-as-Judge
```

**2. Public Datasets**
- **General VQA**: COCO Captions, VQAv2, GQA, TextVQA
- **Document**: DocVQA, InfographicVQA, ChartQA
- **Video**: MSVD, ActivityNet Captions
- **Instruction Following**: LLaVA-Instruct-150K, ShareGPT4V

**3. Format Conversion**
```python
# Standard conversational format
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "<image>\nWhat is in this image?"},
    {"role": "assistant", "content": "The image shows..."}
  ],
  "images": ["image_url.jpg"]  # or base64
}
```

### Data Quality Best Practices

1. **Diverse image sources**: Natural photos, documents, diagrams, charts
2. **Instruction variety**: Questions, descriptions, reasoning, editing
3. **Response quality**: Detailed but concise, factually accurate
4. **Length balance**: Mix short and long responses
5. **Safety filtering**: Remove toxic, biased, or inappropriate content

## Distributed Training

### DeepSpeed ZeRO3 (8 GPU Example)

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: deepspeed_zero3.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 8  # Number of GPUs
```

```json
// deepspeed_zero3.json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5
  },
  "gradient_accumulation_steps": 2,
  "train_micro_batch_size_per_gpu": 8
}
```

**Launch:**
```bash
accelerate launch --config_file accelerate_config.yaml \
  --num_processes 8 \
  run_sft.py --config llama-3-1-8b-qlora.yaml
```

**Scaling efficiency (Llama 3.1 8B, 10k samples):**
- 1x L4: 135 min
- 4x L4: 33 min (4.1x speedup)
- 8x L4: 18 min (7.5x speedup)

## Training Stages

### Stage 1: Vision-Language Alignment (Mandatory)

**Goal**: Align vision encoder outputs with LLM token space

**Data**: Image-caption pairs (COCO, LAION-400M)

**Training**:
- Freeze vision encoder and LLM
- Train only projection/resampler
- ~600k image-caption pairs
- 1 epoch, learning rate 1e-3

**Duration**: 6-12 hours (8 GPUs)

### Stage 2: Instruction Tuning (Mandatory)

**Goal**: Teach model to follow instructions on visual tasks

**Data**: Instruction-following datasets (LLaVA-Instruct, ShareGPT4V)

**Training**:
- Option A: QLoRA (freeze LLM base, train adapters + vision alignment)
- Option B: Full FT (train everything)
- ~150k-600k instruction samples
- 1-3 epochs, learning rate 2e-4 (QLoRA) or 2e-5 (Full FT)

**Duration**: 1-3 days (depending on method/GPUs)

### Stage 3: Task-Specific Fine-Tuning (Optional)

**Goal**: Specialize for specific downstream task

**Data**: Task-specific dataset (DocVQA, GQA, ChartQA, etc.)

**Training**:
- QLoRA on task data
- 10k-50k samples
- 1-2 epochs, learning rate 1e-4

**Duration**: Hours to 1 day

## Evaluation

### Standard Benchmarks (2024)

**General VQA:**
- VQAv2: General visual question answering
- GQA: Compositional reasoning
- TextVQA: OCR-dependent QA

**Specialized:**
- POPE: Object hallucination
- MMBench: Multi-dimensional evaluation
- SEED-Bench: Generative comprehension
- MM-Vet: Complex reasoning
- DocVQA: Document understanding

### Evaluation Tools

**1. LM Evaluation Harness**
```bash
lm_eval --model local-chat-completions \
  --tasks vqa2,gqa,textvqa \
  --model_args model=my-vlm,base_url=http://localhost:8080/v1 \
  --apply_chat_template
```

**2. VLMEvalKit**
```bash
python run.py --data VQAv2 --model llava-v1.6-7b
```

**3. Custom Evaluation**
- Human evaluation (preferred for safety, quality)
- LLM-as-judge (GPT-4V for ranking outputs)
- Task-specific metrics (BLEU for captioning, accuracy for classification)

## Common Issues and Solutions

### Issue 1: Model Hallucinates Objects

**Causes**: Insufficient negative examples, imbalanced data

**Solutions**:
- Add "object does not exist" examples
- Use POPE (Polling-based Object Probing Evaluation) during training
- Lower temperature during inference (0.2-0.5)

### Issue 2: Poor OCR/Text Recognition

**Causes**: Low resolution, encoder not trained on text-heavy images

**Solutions**:
- Use higher resolution vision encoder (384+ pixels)
- Add document/text-rich data to training mix
- Use text-specific encoders (SigLIP outperforms CLIP for text)

### Issue 3: Out of Memory (OOM)

**Immediate fixes**:
- Enable gradient checkpointing
- Reduce batch size, increase gradient accumulation
- Enable QLoRA instead of full fine-tuning
- Reduce max_seq_length

**Long-term fixes**:
- Use more GPUs with DeepSpeed ZeRO3
- Use smaller base model (7B → 3B)
- Implement packing more efficiently

### Issue 4: Training Too Slow

**Optimizations (in order of impact)**:
1. Enable Flash Attention 2 (2-3x speedup)
2. Add Liger Kernels (+30-50% speedup)
3. Enable packing (2-3x fewer steps)
4. Use bfloat16 + tf32 (2x speedup)
5. Multi-GPU with DeepSpeed (near-linear scaling)

### Issue 5: Poor Multi-Image Reasoning

**Causes**: Single-image training data, no cross-image attention

**Solutions**:
- Add multi-image training data
- Increase context window (2048+ tokens)
- Use resampling to compress per-image tokens
- Add position embeddings for image ordering

## Cost and Compute Budgets

### Training Cost Estimates (Llama 3.1 8B)

| Setup | Method | GPUs | Duration | Cost (AWS) |
|-------|--------|------|----------|------------|
| Minimal | QLoRA | 1x L4 24GB | 135 min | $2 |
| Standard | QLoRA | 4x L4 24GB | 33 min | $2 |
| Production | Spectrum | 8x L4 24GB | 21 min | $3 |
| Full FT | None | 8x A100 80GB | ~6 hours | $50 |

**10k samples, 1 epoch. Costs vary by cloud provider and spot instance availability.*

### GPU Memory Requirements

| Model Size | QLoRA | Spectrum | Full FT |
|------------|-------|----------|---------|
| 3B | 16GB | 24GB | 32GB |
| 7-8B | 24GB | 40GB | 80GB |
| 13B | 40GB | 80GB | 160GB |
| 34B | 80GB | 160GB+ | 320GB+ |

## 2025 Trends and Future Directions

1. **Native high-resolution**: Moving away from fixed 336x336 to native resolutions
2. **Video as first-class**: Efficient temporal modeling for hour-long videos
3. **Mixture-of-Experts (MoE)**: Conditional compute for efficiency
4. **Unified modalities**: Audio, 3D, sensor data alongside vision/text
5. **Longer contexts**: 100k+ token windows for multi-document understanding
6. **On-device inference**: Quantized 1-3B models for mobile/edge deployment

## Further Resources

**Training Frameworks:**
- Hugging Face TRL: https://github.com/huggingface/trl
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl
- LLaVA Training: https://github.com/haotian-liu/LLaVA

**Papers:**
- QLoRA: https://arxiv.org/abs/2305.14314
- Spectrum: https://arxiv.org/abs/2406.06623
- MM1 (Meta design study): https://arxiv.org/abs/2403.09611
- Visual Instruction Tuning: https://arxiv.org/abs/2304.08485

**Benchmarks & Evaluation:**
- OpenVLM Leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- Vision Arena: https://huggingface.co/spaces/WildVision/vision-arena
- LM Eval Harness: https://github.com/EleutherAI/lm-evaluation-harness

---

**Sources:**
- Schmid, P. (2024). "How to fine-tune open LLMs in 2025 with Hugging Face"
- Gigant, T. (2024). "Design choices for Vision Language Models in 2024"
- Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
- Li, M. et al. (2024). "Spectrum: Targeted Training on Signal to Noise Ratio"
- Liu, H. et al. (2024). "Improved Baselines with Visual Instruction Tuning"
