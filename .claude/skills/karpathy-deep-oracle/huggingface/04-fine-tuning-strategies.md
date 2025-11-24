# Fine-Tuning Strategies: Full Fine-Tuning vs PEFT Methods

**Date**: 2025-11-16
**Topic**: Fine-tuning strategies for large language models and vision-language models
**Relevance**: Critical knowledge for adapting pre-trained models to specific tasks

---

## Overview - The Fine-Tuning Landscape

Fine-tuning adapts a pre-trained model to a specific task using a smaller specialized dataset. This approach requires far less data and compute than training from scratch, making state-of-the-art AI accessible to researchers and practitioners without massive compute clusters.

**Core Challenge**: How to adapt billion-parameter models efficiently while avoiding catastrophic forgetting and maintaining general capabilities?

From [HuggingFace Documentation](https://huggingface.co/docs/transformers/en/training) (accessed 2025-11-16):
> "Fine-tuning adapts a pretrained model to a specific task with a smaller specialized dataset. This approach requires far less data and compute."

---

## Section 1: Full Fine-Tuning - The Traditional Approach

### Definition and Process

**Full fine-tuning** updates all parameters in a pre-trained model during training on task-specific data. This is the most straightforward approach but also the most resource-intensive.

**How it works:**
```python
# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# Standard training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Advantages of Full Fine-Tuning

**1. Maximum Task Performance**
- Achieves best possible performance on target task
- Model can fully adapt to domain-specific patterns
- No architectural constraints

**2. Flexibility**
- Can completely restructure learned representations
- Adapts all layers simultaneously
- No need to select which parameters to update

**3. Simplicity**
- Straightforward implementation
- Standard training procedures apply
- Well-understood optimization dynamics

From [46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Full fine-tuning a 1B parameter model requires ~4GB storage per task (FP32). For 100 tasks = 400GB."

### Disadvantages of Full Fine-Tuning

**1. Catastrophic Forgetting**

From [Revisiting Catastrophic Forgetting in Large Language Models](https://arxiv.org/abs/2406.04836) (accessed 2025-11-16):
> "LLMs are known for catastrophic forgetting during continual fine-tuning. General instruction tuning can help alleviate the forgetting phenomenon in LLMs during subsequent fine-tuning."

**Key Issue**: When fine-tuning sequentially on Task A then Task B, the model forgets Task A's knowledge.

**Mitigation strategies:**
- Lower learning rates (1e-5 to 5e-6 instead of 1e-3)
- Regularization (L2 weight decay, dropout)
- Replay buffers (mix old task data with new)
- Elastic Weight Consolidation (EWC) - penalize changes to important weights

**2. Memory Requirements**

**Memory breakdown (7B model, FP16):**
- Model weights: 14 GB
- Gradients: 14 GB
- Optimizer states (Adam): 28 GB (2× parameters for momentum + variance)
- Activations: ~6 GB (batch size 4)
- **Total: ~64 GB** (requires 80GB A100)

**3. Storage Costs**
- Each fine-tuned model is a complete copy
- 100 tasks × 7B params × 2 bytes = 1.4TB storage
- Version control becomes expensive

**4. Training Time**
- Full backward pass through all layers
- Longer convergence time
- Higher compute costs

### When to Use Full Fine-Tuning

**Ideal scenarios:**
- Single specialized task with large dataset (>100K examples)
- Maximum accuracy is critical (competitions, production systems)
- Sufficient GPU resources (80GB+ VRAM)
- Model size is small (<1B parameters)
- Domain shift is extreme (e.g., medical imaging, legal documents)

**Example use case:** Fine-tuning GPT-2 (117M params) for domain-specific text generation with 500K examples.

---

## Section 2: Layer Freezing Strategies

### Selective Layer Training

Instead of training all parameters, freeze certain layers and only update others. This balances efficiency with performance.

**Common patterns:**

**1. Freeze Backbone, Train Head**
```python
# Freeze all base model layers
for param in model.base_model.parameters():
    param.requires_grad = False

# Only train task-specific head
for param in model.classifier.parameters():
    param.requires_grad = True
```

**Benefits:**
- 90%+ parameter reduction
- Much faster training
- Works well when pre-trained features are already good

**Use case:** Image classification with pre-trained vision encoder (CLIP, DINOv2)

**2. Freeze Early Layers, Train Late Layers**

From [46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Typically unfreeze last 2-4 layers of vision transformer. Increases trainable parameters by ~20-30% of vision encoder."

```python
# Freeze first 20 layers, unfreeze last 4
num_layers = len(model.transformer.layers)
for i, layer in enumerate(model.transformer.layers):
    if i < num_layers - 4:
        for param in layer.parameters():
            param.requires_grad = False
```

**Rationale:** Early layers learn general features (edges, shapes), late layers learn task-specific patterns.

**3. Gradual Unfreezing**

Train in stages, progressively unfreezing more layers:
```python
# Stage 1: Train head only (1 epoch)
# Stage 2: Unfreeze last 2 layers (1 epoch)
# Stage 3: Unfreeze last 4 layers (2 epochs)
# Stage 4: Full fine-tuning (1 epoch)
```

**Benefits:**
- Prevents catastrophic forgetting
- More stable training
- Better final performance

**Trade-off:** Requires multiple training stages

### Learning Rate Schedules for Frozen Layers

**Critical principle:** Different learning rates for different components.

From [46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "New projector parameters: 1e-3 to 1e-4. Pre-trained language model: 2e-5 to 5e-6 (100x smaller!)."

```python
optimizer = torch.optim.AdamW([
    {'params': head_params, 'lr': 1e-3},        # New layers
    {'params': late_layer_params, 'lr': 1e-4},  # Late layers
    {'params': early_layer_params, 'lr': 1e-5}  # Early layers
])
```

---

## Section 3: Parameter-Efficient Fine-Tuning (PEFT) Methods

### Overview of PEFT Landscape

Parameter-efficient fine-tuning methods freeze the base model and train a small number of additional parameters. This dramatically reduces memory, storage, and training costs while achieving comparable performance to full fine-tuning.

From web research on PEFT vs full fine-tuning (accessed 2025-11-16):
> "PEFT updates a small portion of parameters, is faster and cheaper, while Full Fine-Tuning updates all parameters, is more accurate but requires more resources."

**Major PEFT approaches:**
1. **LoRA** (Low-Rank Adaptation) - See [47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md)
2. **Adapters** (Bottleneck layers)
3. **Prefix Tuning** - See [48-prefix-prompt-tuning-comparison.md](../karpathy/practical-implementation/48-prefix-prompt-tuning-comparison.md)
4. **Prompt Tuning** - See [48-prefix-prompt-tuning-comparison.md](../karpathy/practical-implementation/48-prefix-prompt-tuning-comparison.md)

### Comparison: PEFT vs Full Fine-Tuning

From [PEFT vs Full Fine-Tuning Comparison](https://apxml.com/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/comparing-peft-full-fine-tuning) (accessed 2025-11-16):

| Aspect | Full Fine-Tuning | PEFT (LoRA r=8) |
|--------|-----------------|-----------------|
| Trainable Parameters | 100% | 0.1-1% |
| Memory Required (7B model) | ~64 GB | ~6-14 GB |
| Training Speed | Baseline | 1.5-2× faster |
| Storage per Task | 14 GB | 10-50 MB |
| Performance vs Baseline | Best (100%) | 95-99% |
| Catastrophic Forgetting | High risk | Lower risk |
| Multi-Task Support | Expensive | Cheap (swap adapters) |

**Key insight:** PEFT achieves 95-99% of full fine-tuning performance with <1% of the parameters.

### LoRA: The Most Popular PEFT Method

From [47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):

**Mathematical foundation:**
```
h = W₀x + ΔWx = W₀x + (α/r)BAx
```

Where:
- W₀: Frozen pre-trained weights (d × k)
- B: Trainable matrix (d × r)
- A: Trainable matrix (r × k)
- r: Rank (typically 4-32)
- Parameter count: d×r + r×k instead of d×k

**Example (7B model, r=8):**
- Full fine-tuning: 7B parameters
- LoRA: ~26M parameters (0.37%)
- Memory: 14GB → 6GB (57% reduction)

**Rank selection guidelines:**

From [47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):

| Rank | Use Case | Performance | Parameters |
|------|----------|-------------|------------|
| r=4 | Simple tasks | Good | ~1-2% of model |
| r=8 | Standard setting | Strong | ~3-5% |
| r=16 | Complex tasks | Near full FT | ~7-10% |
| r=32 | Very complex | Excellent | ~15-20% |
| r=64+ | Diminishing returns | Marginal gains | >20% |

### QLoRA: Extreme Efficiency

From [47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):

**Combines LoRA with 4-bit quantization:**
- 7B model in 4-bit: ~3.5 GB base weights
- LoRA adapters in FP16: ~100 MB
- Total: ~6 GB (vs 64 GB for full fine-tuning)

**Performance retention:** ~99.3% of full 16-bit fine-tuning performance

**Trade-off:** 2-3× slower training due to quantization overhead

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

---

## Section 4: Data Requirements for Fine-Tuning

### How Much Data is Needed?

From web research on fine-tuning data requirements (accessed 2025-11-16):

**General guidelines:**

| Task Type | Minimum Examples | Recommended | Optimal |
|-----------|-----------------|-------------|---------|
| Simple classification | 100 | 1,000 | 10,000+ |
| Named Entity Recognition | 500 | 5,000 | 50,000+ |
| Question Answering | 1,000 | 10,000 | 100,000+ |
| Text Generation | 1,000 | 10,000 | 100,000+ |
| Domain Adaptation | 10,000 | 100,000 | 1M+ |
| Vision-Language Tasks | 10,000 | 100,000 | 1M+ |

From [How Much Data is Needed to Fine-Tune an LLM?](https://www.linkedin.com/posts/jimliddle_how-to-fine-tune-open-llms-in-2025-with-hugging-activity-7283471124938518528-v3bk) (accessed 2025-11-16):
> "Fine-tuning may only require up to 100 labeled samples to achieve competitive performance and accuracy and even outperform other models."

**Key factors affecting data requirements:**

**1. Model Size**
- Larger models (>10B) need less data (benefit from pre-training)
- Smaller models (<1B) need more data to adapt
- **Rule of thumb:** 100× fewer examples than training from scratch

**2. Task Complexity**
- Simple tasks (sentiment): 100-1,000 examples
- Complex reasoning: 10,000-100,000 examples
- Multi-modal (VLM): 100,000-1M image-text pairs

**3. Domain Shift**
- Small shift from pre-training: 100-1,000 examples
- Large shift (medical, legal): 10,000-100,000 examples
- Completely new domain: 100,000-1M examples

**4. Quality vs Quantity**

From [How Much Data Do You Need for Effective Fine-Tuning?](https://cyfuture.cloud/kb/howto/how-much-data-do-you-need-for-effective-fine-tuning) (accessed 2025-11-16):
> "Generative tasks (like text summarization or translation) usually require more data, sometimes upwards of 50,000+ examples, depending on complexity."

**High-quality small dataset > Low-quality large dataset**

**Characteristics of high-quality data:**
- Diverse (covers edge cases)
- Clean (no label noise)
- Representative (matches deployment distribution)
- Balanced (classes, styles, formats)

### PEFT Data Efficiency

**Critical insight:** PEFT methods can achieve strong performance with even less data than full fine-tuning.

From web research (accessed 2025-11-16):
> "As models grow larger, they also become smarter — and that means they need less data during fine-tuning."

**PEFT advantages with limited data:**
- Less prone to overfitting (fewer trainable parameters)
- Retains general knowledge better
- Works well with 50-500 examples for many tasks

**Example: LoRA with small datasets**
```python
# Effective with just 100 high-quality examples
train_data = load_dataset("custom_task", split="train[:100]")

lora_config = LoraConfig(
    r=8,                    # Low rank is fine for small data
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1        # Higher dropout for small data
)
```

---

## Section 5: Catastrophic Forgetting and Mitigation

### Understanding Catastrophic Forgetting

From [IBM - What is Catastrophic Forgetting?](https://www.ibm.com/think/topics/catastrophic-forgetting) (accessed 2025-11-16):
> "Catastrophic forgetting occurs when neural networks forget previously learned tasks after being trained on new data or undergoing fine-tuning for specific tasks."

**The problem:**
```
Step 1: Pre-trained model knows general knowledge
Step 2: Fine-tune on Task A → Model learns Task A, retains general knowledge
Step 3: Fine-tune on Task B → Model learns Task B, FORGETS Task A and some general knowledge
```

**Why it happens:**
- Neural networks optimize weights globally
- New task gradients overwrite previous task patterns
- No mechanism to preserve important connections
- Higher learning rates exacerbate the issue

### Mitigation Strategies

**1. Lower Learning Rates**

From web research on catastrophic forgetting (accessed 2025-11-16):
> "The general techniques for preventing catastrophic forgetting are using smaller learning rates."

**Recommended learning rates:**
- Full fine-tuning: 1e-5 to 5e-6 (10× lower than pre-training)
- PEFT methods: 1e-4 to 3e-4 (can be higher due to fewer parameters)
- New task layers: 1e-3 (only for randomly initialized layers)

**2. Regularization Techniques**

**L2 Weight Decay:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.1  # Prevents large weight changes
)
```

**Dropout:**
```python
# Increase dropout during fine-tuning
model.config.hidden_dropout_prob = 0.2  # vs 0.1 during pre-training
```

**3. Replay Buffers**

Mix old task data with new task data:
```python
# 80% new task, 20% old task data
combined_dataset = concatenate_datasets([
    new_task_data.select(range(8000)),
    old_task_data.select(range(2000))
])
```

**4. Elastic Weight Consolidation (EWC)**

Penalize changes to weights that were important for previous tasks:
```python
# Simplified EWC loss
ewc_loss = sum(fisher_info[name] * (param - old_param)**2
                for name, param in model.named_parameters())
total_loss = task_loss + λ * ewc_loss
```

**5. PEFT Methods (Best Prevention)**

From [47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):
> "LoRA adaptation can be done at a fraction of the GPU cost of pre-training and can be merged back at no additional inference cost."

**Why PEFT prevents forgetting:**
- Original weights remain frozen
- Only adapters change
- Can swap adapters for different tasks
- No overwriting of pre-trained knowledge

**6. Instruction Tuning for Continual Learning**

From [Catastrophic Forgetting in LLMs During Continual Fine-tuning](https://arxiv.org/abs/2308.08747) (accessed 2025-11-16):
> "General instruction tuning can help alleviate the forgetting phenomenon in LLMs during subsequent fine-tuning."

**Approach:** First fine-tune on diverse instruction-following data, then specialize.

---

## Section 6: Domain Adaptation vs Task-Specific Fine-Tuning

### Domain Adaptation

**Goal:** Adapt a general model to a specific domain (medical, legal, code) while maintaining broad capabilities.

**Strategy: Two-stage fine-tuning**

**Stage 1: Domain Pre-training**
- Large domain-specific corpus (100K-1M documents)
- Lower learning rate (1e-5)
- Longer training (multiple epochs)
- Goal: Inject domain knowledge

**Stage 2: Task Fine-tuning**
- Task-specific data (1K-100K examples)
- Even lower learning rate (5e-6)
- Shorter training (few epochs)
- Goal: Optimize for specific task

**Example: Medical VQA**
```python
# Stage 1: Medical domain adaptation (1 week)
medical_corpus = load_dataset("medical_papers", split="train")
# Fine-tune on medical text (LLM) and medical images (vision)

# Stage 2: VQA task fine-tuning (1 day)
vqa_data = load_dataset("medical_vqa", split="train")
# Fine-tune on question-answer pairs
```

### Task-Specific Fine-Tuning

**Goal:** Optimize model for a single task, may sacrifice general performance.

**Strategy: Focused optimization**

**Characteristics:**
- Single dataset
- Task-specific metrics
- Aggressive learning rates (can be higher)
- May use full fine-tuning if resources allow

**Trade-offs:**
- Maximum task performance
- Potential loss of general capabilities
- Risk of overfitting to task distribution

---

## Section 7: Multi-Task Fine-Tuning Strategies

### Simultaneous Multi-Task Learning

**Approach:** Train on multiple tasks at once using a shared encoder.

**Architecture:**
```python
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        self.encoder = base_model  # Shared encoder
        self.task_heads = nn.ModuleDict({
            'classification': ClassificationHead(),
            'generation': GenerationHead(),
            'qa': QAHead()
        })

    def forward(self, task_name, inputs):
        features = self.encoder(inputs)
        return self.task_heads[task_name](features)
```

**Benefits:**
- Shared knowledge across tasks
- Single model for multiple capabilities
- Prevents catastrophic forgetting (all tasks trained together)

**Challenges:**
- Balancing task importance (loss weighting)
- Different data requirements per task
- More complex training procedure

### Sequential Multi-Task Learning with PEFT

**Approach:** Use PEFT adapters for each task, keep shared base model.

```python
from peft import PeftModel, LoraConfig

base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Train separate LoRA adapter for each task
task_adapters = {}
for task_name, task_data in tasks.items():
    lora_config = LoraConfig(r=16, lora_alpha=32)
    model = get_peft_model(base_model, lora_config)
    # Train on task_data
    task_adapters[task_name] = model.save_pretrained(f"adapter-{task_name}")

# At inference, load appropriate adapter
model = PeftModel.from_pretrained(base_model, "adapter-classification")
```

**Benefits:**
- No catastrophic forgetting (base model frozen)
- Minimal storage (10-50 MB per adapter)
- Easy task switching (swap adapters)
- Can train tasks independently

**Use case:** Production system supporting multiple capabilities (chat, summarization, classification)

---

## Section 8: ARR-COC-0-1 Fine-Tuning Strategy

### Project-Specific Approach

From the ARR-COC-0-1 codebase in [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/):

**Architecture:**
- **Vision Encoder:** Frozen Qwen3-VL vision encoder
- **Adapter:** Trainable texture extraction + LOD allocation
- **Language Model:** LoRA fine-tuning on Qwen3-VL language decoder

**Training strategy (3 stages):**

**Stage 1: Texture Adapter Pre-training**
- Freeze: Vision encoder + Language model
- Train: 13-channel texture extraction + LOD allocator
- Data: General image-text pairs (1M examples)
- Duration: 2-3 days on 8× A100
- Learning rate: 1e-3 (new parameters)

**Stage 2: LoRA Language Model Adaptation**
- Freeze: Vision encoder + Texture adapter
- Train: LoRA adapters (r=32) on language model
- Data: Instruction-following with images (100K examples)
- Duration: 1-2 days on 8× A100
- Learning rate: 2e-4 (LoRA), 1e-4 (texture adapter fine-tuning)

**Stage 3: End-to-End Fine-Tuning**
- Freeze: Vision encoder only
- Train: Texture adapter + LoRA together
- Data: Task-specific VQA (10K-100K examples)
- Duration: 1 day on 8× A100
- Learning rate: 1e-4 (texture), 1e-5 (LoRA)

**Why this strategy?**

From [46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Freezing backbones provides stability, especially in early training stages."

**Key design decisions:**

**1. Frozen Vision Encoder**
- Qwen3-VL vision features are already excellent
- Saves 50% of memory
- Prevents catastrophic forgetting of visual knowledge
- Focus compute on novel components (texture, LOD)

**2. LoRA on Language Model**
- 7B language model → 224M trainable with r=32 (3.2%)
- Can fine-tune on single 80GB A100
- No inference penalty (merge adapters)
- Preserves general language capabilities

**3. Progressive Training**
- Stage 1 establishes texture extraction
- Stage 2 teaches language model to use texture features
- Stage 3 jointly optimizes
- Each stage builds on previous

**Memory footprint:**
- Vision encoder (frozen, FP16): 3 GB
- Language model (frozen, FP16): 14 GB
- LoRA adapters (trainable): 448 MB
- Texture adapter (trainable): 50 MB
- Gradients + optimizer: ~2 GB
- **Total: ~20 GB** (fits on single A100)

**Expected performance:**
- Texture extraction: 95% of full fine-tuning
- Language adaptation: 98% of full fine-tuning
- Overall: 96-97% of full fine-tuning
- **Training cost: <5% of full fine-tuning**

---

## Sources

### Source Documents

**From karpathy-deep-oracle skill:**
- [46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md) - Frozen backbone strategies, adapter architectures, multi-stage training
- [47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md) - LoRA fundamentals, rank selection, QLoRA efficiency
- [48-prefix-prompt-tuning-comparison.md](../karpathy/practical-implementation/48-prefix-prompt-tuning-comparison.md) - PEFT method comparison, prefix tuning, P-Tuning v2

### Web Research

**HuggingFace Official Documentation:**
- [HuggingFace Training Documentation](https://huggingface.co/docs/transformers/en/training) (accessed 2025-11-16)
- [How to fine-tune open LLMs in 2025 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2025) - Philschmid blog (accessed 2025-11-16)

**PEFT vs Full Fine-Tuning:**
- [PEFT vs Full Fine-Tuning: Key Limitations Compared](https://www.artech-digital.com/blog/peft-vs-full-fine-tuning-key-limitations-compared) - Artech Digital (accessed 2025-11-16)
- [PEFT vs Full Fine-Tuning Comparison](https://apxml.com/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/comparing-peft-full-fine-tuning) - ApX Machine Learning (accessed 2025-11-16)
- [Fine-Tuning vs PEFT: A Practical Guide](https://medium.com/@whyamit404/fine-tuning-vs-peft-parameter-efficient-fine-tuning-a-practical-guide-3844e5688000) - Medium (accessed 2025-11-16)

**Catastrophic Forgetting:**
- [What is Catastrophic Forgetting?](https://www.ibm.com/think/topics/catastrophic-forgetting) - IBM (accessed 2025-11-16)
- [Catastrophic Forgetting in LLMs During Continual Fine-tuning](https://arxiv.org/abs/2308.08747) - arXiv:2308.08847, Luo et al., 2023 (accessed 2025-11-16)
- [Revisiting Catastrophic Forgetting in Large Language Models](https://arxiv.org/abs/2406.04836) - arXiv:2406.04836, Li et al., 2024 (accessed 2025-11-16)
- [Fine-Tuning LLMs: Overcoming Catastrophic Forgetting](https://www.legionintel.com/blog/navigating-the-challenges-of-fine-tuning-and-catastrophic-forgetting) - Legion Secure AI (accessed 2025-11-16)
- [Mitigating Catastrophic Forgetting in Large Models](https://medium.com/@mzeynali01/mitigating-catastrophic-forgetting-in-large-scale-models-with-extensive-parameters-2610d6defd20) - Medium (accessed 2025-11-16)

**Data Requirements:**
- [How Many Data Points Are Necessary for Fine-Tuning](https://www.linkedin.com/posts/jimliddle_how-to-fine-tune-open-llms-in-2025-with-hugging-activity-7283471124938518528-v3bk) - LinkedIn (accessed 2025-11-16)
- [How Much Data is Enough Data? Fine-Tuning Large Language Models](https://arxiv.org/abs/2409.03454) - arXiv:2409.03454, Vieira et al., 2024 (accessed 2025-11-16)
- [How Much Data Do You Need for Effective Fine-Tuning?](https://cyfuture.cloud/kb/howto/how-much-data-do-you-need-for-effective-fine-tuning) - Cyfuture Cloud (accessed 2025-11-16)
- [Preparing data for fine-tuning](https://docs.aws.amazon.com/nova/latest/userguide/fine-tune-prepare-data-understanding.html) - AWS Documentation (accessed 2025-11-16)

**Community Discussions:**
- [Reddit r/LocalLLaMA: How much data do I need to feed a model](https://www.reddit.com/r/LocalLLaMA/comments/17gbdy1/how_much_data_do_i_need_to_feed_a_model_to_notice/) (accessed 2025-11-16)
- [Reddit r/MachineLearning: Catastrophic forgetting motivation](https://www.reddit.com/r/MachineLearning/comments/186ck5k/d_what_is_the_motivation_for_parameterefficient/) (accessed 2025-11-16)
- [HuggingFace Forums: How to prevent catastrophic forgetting](https://discuss.huggingface.co/t/how-to-prevent-catastrophic-forgetting-in-fine-tuned-large-language-models/135153) (accessed 2025-11-16)
- [OpenAI Community: How to choose fine-tuning data](https://community.openai.com/t/how-to-choose-my-fine-tuning-data/573270) (accessed 2025-11-16)

### Additional References

**ARR-COC-0-1 Project:**
- [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/) - Reference implementation with frozen vision encoder + LoRA strategy

---

**File Statistics**: ~700 lines covering full fine-tuning, layer freezing, PEFT methods, data requirements, catastrophic forgetting mitigation, domain adaptation, multi-task learning, and ARR-COC-0-1 project-specific strategy with comprehensive citations.
