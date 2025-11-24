# HuggingFace PEFT Library: LoRA, QLoRA, and Adapter Management

## Overview

The HuggingFace PEFT (Parameter-Efficient Fine-Tuning) library provides production-ready implementations of LoRA, QLoRA, and other adapter-based fine-tuning methods. This comprehensive guide covers practical usage, configuration strategies, quantization integration, and deployment patterns for vision-language models and large language models.

**Key Features:**
- Unified API for LoRA, QLoRA, Prefix Tuning, P-Tuning v2, and more
- Seamless integration with Transformers and Trainer
- 4-bit quantization with bitsandbytes (QLoRA)
- Adapter merging and switching for multi-task deployment
- Memory-efficient training (train 65B models on 48GB GPUs)

From [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft) (accessed 2025-11-16):
> "PEFT approaches enable you to get performance comparable to full fine-tuning while only having a small number of trainable parameters. This makes it easier to train and store large language models on consumer hardware."

## Section 1: PEFT Library Architecture and Supported Methods

### Core Components

The PEFT library provides a unified interface for multiple parameter-efficient fine-tuning methods:

**Available Methods:**
1. **LoRA (Low-Rank Adaptation)** - Decomposes weight updates into low-rank matrices
2. **QLoRA** - LoRA with 4-bit quantized base model
3. **Prefix Tuning** - Trainable prefix vectors at each layer
4. **P-Tuning v2** - Deep prompt tuning without reparameterization
5. **Prompt Tuning** - Soft prompts at input layer only
6. **AdaLoRA** - Adaptive rank allocation for LoRA
7. **IA³** - Inhibited and Amplified Inner Activations
8. **DoRA** - Weight-Decomposed LoRA (magnitude + direction)

From [PEFT GitHub Repository](https://github.com/huggingface/peft) (accessed 2025-11-16):

**Unified Workflow:**
```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Configure PEFT method
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

# 3. Wrap model with PEFT
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

### Task Types

PEFT supports task-specific configurations through `TaskType` enum:

```python
from peft import TaskType

# Language modeling
TaskType.CAUSAL_LM          # Autoregressive generation (GPT-style)
TaskType.SEQ_2_SEQ_LM       # Encoder-decoder (T5, BART)

# Classification
TaskType.SEQ_CLS            # Sequence classification
TaskType.TOKEN_CLS          # Token classification (NER, POS)

# Question answering
TaskType.QUESTION_ANS       # Extractive QA

# Feature extraction
TaskType.FEATURE_EXTRACTION # Embedding models
```

### Model Compatibility

**Compatible Base Models:**
- **Language Models**: GPT-2, GPT-J, LLaMA, Mistral, Falcon, Qwen
- **Vision-Language Models**: LLaVA, BLIP-2, Flamingo, InstructBLIP
- **Encoder Models**: BERT, RoBERTa, DeBERTa
- **Encoder-Decoder**: T5, BART, mBART

From [PEFT Blog Post](https://huggingface.co/blog/peft) (accessed 2025-11-16):
> "PEFT methods work with any model from the Transformers library. You can use the same code for different model architectures by simply changing the model checkpoint."

## Section 2: LoRA Configuration and Rank Selection

### LoRAConfig Parameters

Complete configuration options for LoRA:

```python
from peft import LoraConfig

config = LoraConfig(
    # Core parameters
    r=16,                              # Rank of decomposition
    lora_alpha=32,                     # Scaling factor
    target_modules=["q_proj", "v_proj"], # Which layers to adapt

    # Regularization
    lora_dropout=0.05,                 # Dropout for LoRA layers

    # Advanced options
    bias="none",                       # Train bias: "none", "all", "lora_only"
    task_type=TaskType.CAUSAL_LM,      # Task type

    # Initialization
    init_lora_weights=True,            # Initialize A with Kaiming, B with zeros
    use_rslora=False,                  # Rank-stabilized LoRA (α/sqrt(r))
    use_dora=False,                    # Weight-Decomposed LoRA

    # Module selection
    modules_to_save=None,              # Additional modules to train
    layers_to_transform=None,          # Specific layer indices
    layers_pattern=None,               # Regex pattern for layers
)
```

### Rank Selection Guidelines

From practical experience and [HuggingFace PEFT tutorials](https://huggingface.co/blog/peft) (accessed 2025-11-16):

**Rank Selection Table:**

| Task Complexity | Recommended Rank | Trainable % | Use Case |
|----------------|------------------|-------------|----------|
| Simple classification | r = 4-8 | ~0.05% | Sentiment, topic classification |
| Moderate tasks | r = 8-16 | ~0.1% | QA, summarization |
| Complex reasoning | r = 16-32 | ~0.2-0.5% | Code generation, complex QA |
| Vision-language | r = 32-64 | ~0.5-1% | VQA, image captioning |
| Multi-task | r = 64+ | ~1-2% | Multiple diverse tasks |

**Factors Influencing Rank Choice:**

1. **Model Size**: Larger models benefit from higher ranks
   - 1B-7B params: r = 8-16
   - 7B-13B params: r = 16-32
   - 13B+ params: r = 32-64

2. **Dataset Size**: More data supports higher ranks
   - <10K examples: r = 4-8
   - 10K-100K examples: r = 8-16
   - >100K examples: r = 16-32

3. **Domain Shift**: Larger shift from pre-training → higher rank
   - Similar domain: r = 8
   - Different domain: r = 16-32
   - Specialized domain: r = 32-64

From [Reddit r/MachineLearning discussion](https://www.reddit.com/r/MachineLearning/comments/17z82pc/p_practical_tips_for_finetuning_llms_using_lora/) (accessed 2025-11-16):
> "For VLMs, I found that vision encoder needs higher rank (r=32-64) than language decoder (r=8-16). Vision features are high-dimensional and benefit from more capacity."

### Alpha and Scaling

**lora_alpha parameter controls the magnitude of LoRA updates:**

```
Effective scaling = lora_alpha / r
```

**Common patterns:**
- `lora_alpha = r`: Scaling factor of 1 (standard)
- `lora_alpha = 2 * r`: More aggressive updates
- `lora_alpha = r / 2`: Conservative updates

**Rank-Stabilized LoRA (rsLoRA):**

From [rsLoRA paper](https://arxiv.org/abs/2312.03732) (accessed 2025-11-16):

```python
config = LoraConfig(
    r=64,
    lora_alpha=64,
    use_rslora=True,  # Changes scaling to alpha/sqrt(r)
)
```

**Benefit**: Stabilizes training at higher ranks by reducing effective learning rate as rank increases.

### Target Module Selection

**Attention Projections (Most Common):**
```python
# Query and Value only (QLoRA style - minimal parameters)
target_modules=["q_proj", "v_proj"]

# All attention (comprehensive)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Including MLP Layers:**
```python
# Attention + MLP (maximum adaptation)
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # MLP (for LLaMA-style)
]

# Or use regex for all linear layers
target_modules="all-linear"
```

**Vision-Language Models:**
```python
# Language decoder only (common for VLMs)
target_modules=[
    r"language_model\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
]

# Vision encoder only
target_modules=[
    r"vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
]

# Both vision and language
target_modules=[
    r".*\.(q_proj|v_proj|o_proj)"  # All attention in all components
]
```

From [Fine-Tuning Vision-Language Models using LoRA](https://gautam75.medium.com/fine-tuning-vision-language-models-using-lora-b640c9af8b3c) (accessed 2025-11-16):
> "For VLMs, applying LoRA to language decoder only is often sufficient. Vision encoders are usually well-pretrained and benefit less from adaptation."

## Section 3: QLoRA - 4-bit Quantized Fine-Tuning

### QLoRA Overview

From [QLoRA Paper](https://arxiv.org/abs/2305.14314) (accessed 2025-11-16):

QLoRA enables fine-tuning of extremely large models on consumer hardware by combining:
1. **4-bit NormalFloat (NF4) quantization** for base model weights
2. **Double quantization** of quantization constants
3. **Paged optimizers** for memory spike management
4. **16-bit LoRA adapters** trained on top of frozen 4-bit base

**Memory Savings Example (65B Model):**
- Full FP16 fine-tuning: 780 GB (impossible on single GPU)
- LoRA FP16: 120 GB (still too large)
- QLoRA 4-bit: 48 GB (fits on single A100!)

### BitsAndBytesConfig Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",              # Use NormalFloat4 (optimal for normally distributed weights)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 for stability
    bnb_4bit_use_double_quant=True,         # Double quantization for extra compression
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                      # Automatic device placement
    torch_dtype=torch.bfloat16,
)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,504,607,232 || trainable%: 0.12%
```

From [HuggingFace 4-bit Transformers Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) (accessed 2025-11-16):
> "QLoRA makes it possible to fine-tune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance."

### NF4 vs FP4 Quantization Types

**NormalFloat4 (NF4) - Recommended:**
- Information-theoretically optimal for normally distributed weights
- Neural network weights follow ~N(0, σ²) distribution
- Quantization bins chosen to minimize expected quantization error

**Float4 (FP4) - Standard floating point:**
- Uniform quantization levels
- Not optimized for weight distributions
- Generally inferior to NF4 for neural networks

```python
# NF4 (recommended for most models)
BitsAndBytesConfig(bnb_4bit_quant_type="nf4")

# FP4 (only if weights don't follow normal distribution)
BitsAndBytesConfig(bnb_4bit_quant_type="fp4")
```

From [Mastering QLoRA Deep Dive](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html) (accessed 2025-11-16):
> "NF4 achieves better quantization by placing more bins near zero (where most weights concentrate) and fewer bins at extremes. This matches the natural distribution of neural network parameters."

### Double Quantization

**First Quantization:** Base model weights → 4-bit NF4
**Second Quantization:** Quantization constants (scale factors) → 8-bit

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Compress scale factors
)
```

**Memory savings:**
- Without double quantization: 32-bit floats for scale factors
- With double quantization: 8-bit quantized scale factors
- Typical reduction: 0.4-0.8 GB for 7B model

### Compute Dtype Considerations

```python
# BFloat16 (recommended for most cases)
BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16)
# Pros: Stable training, wider range than FP16
# Cons: Requires modern GPUs (Ampere+)

# Float16 (if BF16 not available)
BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float16)
# Pros: Widely supported
# Cons: Narrower range, can have overflow issues

# Float32 (fallback)
BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float32)
# Pros: Maximum precision
# Cons: Slower, more memory for activations
```

From [QLoRA GitHub Discussion](https://github.com/ggerganov/llama.cpp/discussions/1595) (accessed 2025-11-16):
> "Use BFloat16 compute dtype when possible. It provides the stability of FP32 dynamic range with the memory efficiency of FP16."

### QLoRA Training Considerations

**Learning Rate:** QLoRA often needs higher learning rates than full fine-tuning
```python
TrainingArguments(
    learning_rate=2e-4,  # vs 5e-5 for full fine-tuning
    warmup_steps=100,
)
```

**Training Speed:** QLoRA is 2-3× slower than FP16 LoRA due to:
- Quantization/dequantization overhead
- Mixed precision compute
- Still much faster than full fine-tuning!

**Memory Usage Breakdown (7B model with QLoRA):**
- Base model (4-bit): ~3.5 GB
- LoRA adapters (FP16): ~100 MB
- Optimizer states: ~200 MB
- Gradients: ~100 MB
- Activations: ~2-4 GB (depends on batch size)
- **Total**: ~6-8 GB (fits on RTX 3090!)

## Section 4: Advanced LoRA Variants

### DoRA: Weight-Decomposed LoRA

From [DoRA Paper](https://arxiv.org/abs/2402.09353) (accessed 2025-11-16):

DoRA decomposes weight updates into magnitude and direction components:

```python
config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,  # Enable DoRA
    target_modules=["q_proj", "v_proj"],
)
```

**Advantages:**
- Better performance especially at low ranks (r=4-8)
- More stable training dynamics
- Improved learning of directional vs magnitude changes

**Trade-offs:**
- ~10-15% slower training
- Slightly more memory (stores magnitude separately)
- Worth it for low-rank scenarios (r < 16)

### AdaLoRA: Adaptive Rank Allocation

AdaLoRA dynamically allocates rank budget across layers:

```python
from peft import AdaLoraConfig

config = AdaLoraConfig(
    r=12,                      # Average rank
    target_r=8,                # Target rank after pruning
    init_r=12,                 # Initial rank
    tinit=0,                   # Start pruning at step 0
    tfinal=1000,               # Finish pruning at step 1000
    deltaT=10,                 # Pruning interval
    target_modules=["q_proj", "v_proj"],
)
```

**How it works:**
1. Start with rank r for all modules
2. Compute importance scores during training
3. Gradually reduce rank of less important modules
4. Redistribute budget to important modules

**Use case:** When you have a fixed parameter budget and want to automatically allocate it optimally.

### LoRA Layer Replication (SOLAR Pattern)

From [PEFT Documentation](https://huggingface.co/docs/peft) (accessed 2025-11-16):

Expand model by duplicating layers with shared weights:

```python
config = LoraConfig(
    layer_replication=[[0, 4], [2, 5]],  # Replicate layer 0→4 and layer 2→5
    r=16,
)
# Original layers: [0, 1, 2, 3, 4]
# After replication: [0, 1, 2, 3, 2, 3, 4]
```

**Memory efficiency:** Replicated layers share base weights, only LoRA adapters are duplicated.

**Use case:** Expanding smaller models (7B → 10B) without full training from scratch.

## Section 5: Training with PEFT and Trainer

### Basic Training Setup

```python
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

# 1. Prepare model with PEFT
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_config)

# 2. Configure training
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch size = 16
    learning_rate=2e-4,                 # Higher LR for PEFT
    fp16=True,                          # Mixed precision
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    warmup_steps=100,
    optim="adamw_torch",
)

# 3. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# 4. Save adapters only (small!)
model.save_pretrained("./lora-adapters")
```

### Training with QLoRA

```python
from transformers import BitsAndBytesConfig

# 1. Load quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Prepare model for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# 3. Add LoRA
peft_config = LoraConfig(...)
model = get_peft_model(model, peft_config)

# 4. Train normally with Trainer
# (rest is identical to standard LoRA training)
```

From [MLflow QLoRA Tutorial](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/) (accessed 2025-11-16):
> "prepare_model_for_kbit_training() freezes all base model layers and configures them for gradient checkpointing, enabling training of quantized models."

### Hyperparameter Recommendations

**Learning Rates:**
```python
# PEFT methods need higher LR than full fine-tuning
{
    "LoRA (FP16)": 1e-4 to 3e-4,
    "QLoRA (4-bit)": 2e-4 to 5e-4,
    "Full fine-tuning": 5e-6 to 2e-5,
}
```

**Batch Sizes:**
```python
# PEFT allows larger batches due to reduced memory
{
    "Full fine-tuning (7B)": 1-2 per GPU,
    "LoRA (7B)": 4-8 per GPU,
    "QLoRA (7B)": 8-16 per GPU,
}
```

**Training Duration:**
- PEFT converges faster: 3-10 epochs typical
- Full fine-tuning: 10-30 epochs
- Use early stopping to prevent overfitting

## Section 6: Adapter Merging and Deployment

### Merging Adapters with Base Model

From [PEFT Model Merging Guide](https://huggingface.co/docs/peft/en/developer_guides/model_merging) (accessed 2025-11-16):

```python
from peft import PeftModel

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora-adapters")

# Merge and unload adapters
merged_model = model.merge_and_unload()

# Save as standard Transformers model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Now it's a regular model (no PEFT dependency)
# Can be loaded with:
# model = AutoModelForCausalLM.from_pretrained("./merged-model")
```

**Benefits of Merging:**
- No inference latency overhead
- Smaller deployment footprint (no adapter files)
- Standard Transformers compatibility
- Easier distribution and deployment

**When NOT to merge:**
- Multi-task deployment (keep adapters separate)
- A/B testing different adapters
- Frequent adapter updates

### Multi-Adapter Loading and Switching

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load first adapter
model = PeftModel.from_pretrained(base_model, "./adapter-task1")

# Load additional adapters
model.load_adapter("./adapter-task2", adapter_name="task2")
model.load_adapter("./adapter-task3", adapter_name="task3")

# Switch between adapters
model.set_adapter("task1")  # Use task1 adapter
output1 = model.generate(...)

model.set_adapter("task2")  # Switch to task2
output2 = model.generate(...)

# Use multiple adapters in same batch
model.set_adapter(["task1", "task2"])  # Advanced: mix adapters
```

**Use case:** Single base model serving multiple tasks with task-specific adapters.

**Memory efficiency:** Base model loaded once, adapters are small (MBs).

From [PEFT GitHub Issues](https://github.com/huggingface/peft/issues/868) (accessed 2025-11-16):
> "merge_and_unload() merges LoRA adapters into the base model and removes PEFT layers, returning a standard Transformers model. This is the recommended deployment pattern."

### Adapter Merging for Quantized Models

**Important:** Cannot directly merge adapters into 4-bit quantized base!

From [PEFT GitHub Discussion](https://github.com/huggingface/peft/issues/2105) (accessed 2025-11-16):

**Correct workflow:**
```python
# 1. Load 4-bit model and adapter
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
peft_model = PeftModel.from_pretrained(base_model, "./qlora-adapters")

# 2. CANNOT merge directly - must dequantize first
# merged_model = peft_model.merge_and_unload()  # ERROR!

# 3. Instead: Load FP16 base and merge
base_fp16 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
)
peft_fp16 = PeftModel.from_pretrained(base_fp16, "./qlora-adapters")
merged = peft_fp16.merge_and_unload()

# 4. Save merged FP16 model
merged.save_pretrained("./merged-fp16")

# 5. For deployment: Quantize merged model
# (using same BitsAndBytesConfig or export to GGUF/AWQ)
```

**Alternative:** Keep adapters separate for 4-bit inference (small overhead, no merging needed).

### Inference Optimization Strategies

**Strategy 1: Merged Model (No Adapter Overhead)**
```python
# Best for single-task deployment
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged")
```

**Strategy 2: Adapter Switching (Multi-Task)**
```python
# Best for serving multiple tasks
base_model = AutoModelForCausalLM.from_pretrained("model")
model = PeftModel.from_pretrained(base_model, "adapter-1")
model.load_adapter("adapter-2", adapter_name="task2")
# Switch as needed
```

**Strategy 3: Quantized Base + Adapter**
```python
# Best for memory-constrained deployment
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
base_model = AutoModelForCausalLM.from_pretrained(
    "model",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base_model, "adapter")
# Small adapter overhead (~1-2% latency)
```

From [Kaggle Merging LoRA Adapters](https://www.kaggle.com/code/ebinbt007/merging-lora-adapters-with-base-model) (accessed 2025-11-16):
> "For production deployment, merge adapters to eliminate runtime overhead. For multi-task serving or frequent updates, keep adapters separate and use adapter switching."

## Section 7: Vision-Language Model PEFT Patterns

### VLM-Specific LoRA Configuration

Vision-language models require careful consideration of which components to adapt:

```python
from peft import LoraConfig

# Strategy 1: Language decoder only (most common)
config_language_only = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        # LLaVA-style: target language model only
        r"language_model\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj|o_proj)",
        r"language_model\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Strategy 2: Vision encoder only
config_vision_only = LoraConfig(
    r=32,  # Higher rank for vision (high-dimensional features)
    lora_alpha=64,
    target_modules=[
        r"vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
    ],
    lora_dropout=0.1,
)

# Strategy 3: Both vision and language (maximum adaptation)
config_both = LoraConfig(
    r=16,  # Can use different ranks via layer-specific configs
    target_modules=[
        r".*\.(q_proj|v_proj|o_proj)",  # All attention layers
    ],
)
```

From [Low-Rank Few-Shot Adaptation of VLMs](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.pdf) (accessed 2025-11-16):
> "Applying LoRA to both vision and text encoders improves performance over single-modality adaptation, but the vision encoder typically requires higher rank (r=32-64) than the text encoder (r=8-16)."

### Rank Recommendations for VLM Components

| Component | Recommended Rank | Reasoning |
|-----------|-----------------|-----------|
| Vision encoder (ViT) | r = 32-64 | High-dimensional visual features |
| Cross-attention | r = 16-32 | Critical fusion point |
| Language decoder | r = 8-16 | Text is lower-dimensional |
| Vision-language connector | Full fine-tune | Small parameter count |

### VLM Training Example

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import get_peft_model, LoraConfig

# 1. Load VLM
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 2. Configure LoRA for language model only
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "language_model.model.layers.*.self_attn.q_proj",
        "language_model.model.layers.*.self_attn.v_proj",
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 3. Apply PEFT
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Train with vision-language data
# (Trainer setup identical to language models)
```

### Freezing Strategy Best Practices

From [LoRA VLM Fine-Tuning Guide](https://medium.com/@clturner23/fine-tuning-hugging-face-llms-with-qlora-31be90a49a41) (accessed 2025-11-16):

**Conservative (recommended for most cases):**
- Freeze: Vision encoder, connector
- Train: LoRA on language decoder
- Reason: Vision features usually well-pretrained

**Moderate:**
- Freeze: Vision encoder
- Train: Connector (full) + LoRA on language decoder
- Reason: Adapt fusion while preserving vision backbone

**Aggressive:**
- Train: LoRA on vision encoder + LoRA on language decoder
- Reason: Maximum adaptation for domain-specific tasks

## Section 8: arr-coc-0-1 LoRA Fine-Tuning Integration

### arr-coc-0-1 Architecture Components

The arr-coc-0-1 VLM combines:
1. **Vision encoder**: Processes multi-resolution patches with variable LOD
2. **ARR-COC layers**: Knowing, Balancing, Attending modules
3. **Language decoder**: Qwen-VL or similar VLM backbone

### Recommended LoRA Strategy for arr-coc-0-1

From [arr-coc-0-1 Training Configuration](../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/config.py):

```python
from peft import LoraConfig, get_peft_model

# ARR-COC specific LoRA configuration
lora_config = LoraConfig(
    r=32,  # Higher rank for complex multimodal reasoning
    lora_alpha=64,

    # Target attention AND MLP in language decoder
    target_modules=[
        # Attention layers
        "language_model.*.self_attn.q_proj",
        "language_model.*.self_attn.v_proj",
        "language_model.*.self_attn.o_proj",

        # MLP layers for opponent processing adaptation
        "language_model.*.mlp.gate_proj",
        "language_model.*.mlp.up_proj",
        "language_model.*.mlp.down_proj",
    ],

    # Keep ARR-COC components trainable (not LoRA)
    modules_to_save=[
        "knowing",      # Relevance scorers
        "balancing",    # Tension navigation
        "attending",    # LOD allocation
    ],

    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Rationale for MLP targeting:**

From [Reddit r/MachineLearning LoRA Discussion](https://www.reddit.com/r/MachineLearning/comments/17z82pc/p_practical_tips_for_finetuning_llms_using_lora/) (accessed 2025-11-16):
> "FFNs (MLP layers) have much more capacity than attention. For complex reasoning tasks, targeting both attention and MLP yields better results than attention-only LoRA."

### Training Configuration for arr-coc-0-1

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Basic settings
    output_dir="./arr-coc-lora",
    num_train_epochs=3,

    # Batch sizes (optimized for A100 40GB)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16

    # Learning rates
    learning_rate=2e-4,  # Higher for LoRA
    warmup_steps=100,
    lr_scheduler_type="cosine",

    # Memory optimization
    fp16=False,
    bf16=True,           # BFloat16 for stability
    gradient_checkpointing=True,

    # Logging and saving
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",

    # Optimizer
    optim="adamw_torch",
    weight_decay=0.01,
)
```

### QLoRA for arr-coc-0-1 (Memory-Constrained Training)

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base VLM with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "arr-coc-base-vlm",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# Add LoRA (same config as above)
model = get_peft_model(model, lora_config)

# Train normally
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

**Memory Usage (7B VLM with QLoRA):**
- Base model (4-bit): ~4 GB
- ARR-COC modules (FP16): ~500 MB
- LoRA adapters (FP16): ~200 MB
- Optimizer states: ~400 MB
- Activations (batch=2): ~4 GB
- **Total**: ~9 GB (fits on RTX 4090!)

### ARR-COC Component Training Strategy

**Critical distinction:** ARR-COC cognitive modules should NOT use LoRA:

```python
# DON'T apply LoRA to ARR-COC components
# (they need full parameter updates for opponent processing)

# Instead: Freeze base VLM, train ARR-COC + LoRA
for name, param in model.named_parameters():
    if "knowing" in name or "balancing" in name or "attending" in name:
        param.requires_grad = True  # Train fully
    elif "lora" in name:
        param.requires_grad = True  # Train LoRA adapters
    else:
        param.requires_grad = False  # Freeze base VLM
```

**Why full training for ARR-COC modules:**
- Opponent processing requires learning balanced tension navigation
- LOD allocation depends on precise relevance scoring
- Low-rank approximation may not capture complex non-linear dynamics

### Deployment Pattern for arr-coc-0-1

```python
# After training: Merge LoRA for production deployment
from peft import PeftModel

# 1. Load trained model
base_model = AutoModelForCausalLM.from_pretrained("arr-coc-base-vlm")
peft_model = PeftModel.from_pretrained(base_model, "./arr-coc-lora")

# 2. Merge LoRA adapters
merged_model = peft_model.merge_and_unload()

# 3. Save complete model
merged_model.save_pretrained("./arr-coc-deployed")

# Now deployable as standard VLM with integrated ARR-COC
# No PEFT dependency in production
```

From arr-coc-0-1 project documentation:
> "ARR-COC's relevance realization benefits from full fine-tuning of cognitive modules (knowing, balancing, attending) while using LoRA for language decoder adaptation. This hybrid approach balances parameter efficiency with the need for precise opponent processing dynamics."

## Sources

### Source Documents
- [karpathy/practical-implementation/47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md) - Lines 1-668 (LoRA fundamentals, QLoRA, rank selection)
- [karpathy/practical-implementation/48-prefix-prompt-tuning-comparison.md](../karpathy/practical-implementation/48-prefix-prompt-tuning-comparison.md) - Lines 1-637 (PEFT methods comparison)

### Web Research

**HuggingFace Documentation:**
- [PEFT Library Documentation](https://huggingface.co/docs/peft) - Official PEFT documentation (accessed 2025-11-16)
- [PEFT GitHub Repository](https://github.com/huggingface/peft) - Source code and examples (accessed 2025-11-16)
- [Parameter-Efficient Fine-Tuning Blog](https://huggingface.co/blog/peft) - Introduction to PEFT (accessed 2025-11-16)
- [4-bit Transformers with bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes) - QLoRA announcement (accessed 2025-11-16)
- [PEFT Model Merging Guide](https://huggingface.co/docs/peft/en/developer_guides/model_merging) - Adapter merging documentation (accessed 2025-11-16)
- [LoRA Developer Guide](https://huggingface.co/docs/peft/en/developer_guides/lora) - Advanced LoRA usage (accessed 2025-11-16)

**Academic Papers:**
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023 (accessed 2025-11-16)
- [Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732) - rsLoRA paper (accessed 2025-11-16)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) - DoRA paper (accessed 2025-11-16)
- [Low-Rank Few-Shot Adaptation of Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.pdf) - CVPR 2024 (accessed 2025-11-16)

**Tutorials & Guides:**
- [Fine-Tuning Llama 2 with PEFT LoRA](https://obot.ai/resources/learning-center/fine-tuning-llama-2/) - Obot AI tutorial (accessed 2025-11-16)
- [Practical Guide to Fine-tune LLMs with LoRA](https://medium.com/@manindersingh120996/practical-guide-to-fine-tune-llms-with-lora-c835a99d7593) - Medium tutorial (accessed 2025-11-16)
- [Mastering QLoRA Deep Dive](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html) - Technical breakdown (accessed 2025-11-16)
- [Fine-Tuning with QLoRA](https://medium.com/@clturner23/fine-tuning-hugging-face-llms-with-qlora-31be90a49a41) - Medium tutorial (accessed 2025-11-16)
- [MLflow QLoRA Tutorial](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/) - Production workflow (accessed 2025-11-16)

**Community Resources:**
- [Reddit r/MachineLearning - LoRA Tips](https://www.reddit.com/r/MachineLearning/comments/17z82pc/p_practical_tips_for_finetuning_llms_using_lora/) - Practical advice (accessed 2025-11-16)
- [Reddit r/LocalLLaMA - LoRA Discussion](https://www.reddit.com/r/LocalLLaMA/comments/17z91wk/practical_tips_for_finetuning_llms_using_lora/) - Community tips (accessed 2025-11-16)
- [GitHub PEFT Issues](https://github.com/huggingface/peft/issues) - merge_and_unload discussions (accessed 2025-11-16)
- [Kaggle Merging LoRA Adapters](https://www.kaggle.com/code/ebinbt007/merging-lora-adapters-with-base-model) - Deployment notebook (accessed 2025-11-16)

**Implementation Guides:**
- [Hands-on LLM Fine-Tuning with LoRA](https://apxml.com/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/hands-on-fine-tuning-with-lora) - ApX ML course (accessed 2025-11-16)
- [Merging PEFT Adapters](https://apxml.com/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/merging-peft-adapters) - Deployment guide (accessed 2025-11-16)
- [Fine-Tuning Vision-Language Models using LoRA](https://gautam75.medium.com/fine-tuning-vision-language-models-using-lora-b640c9af8b3c) - VLM-specific patterns (accessed 2025-11-16)

### Additional References
- [arr-coc-0-1 Project](../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/) - Adaptive Relevance Realization implementation
- [Qwen-VL Fine-Tuning Scripts](../source-codebases/deepseek/13-Qwen3-VL/qwen-vl-finetune/) - VLM LoRA implementation examples
