# VLM Fine-tuning & Instruction Tuning

## Overview

Instruction tuning adapts vision-language models (VLMs) to follow natural language instructions paired with images, enabling conversational AI capabilities and task generalization. This supervised fine-tuning stage transforms pre-trained VLMs into instruction-following assistants capable of visual question answering, image captioning, reasoning, and multi-turn dialogue.

**Key Concepts:**
- **Instruction format**: User provides image + text instruction, model generates text response
- **Visual instruction data**: Curated datasets of (image, instruction, response) triplets
- **Multi-task learning**: Single model handles diverse vision-language tasks through instructions
- **Parameter-efficient methods**: LoRA, adapters reduce trainable parameters by 90%+

From [karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Multi-stage training typically occurs due to limited availability of high-quality data, memory constraints for efficient training, and stability issues. Standard approach: Projector pre-training (frozen backbones) → SFT (unfreeze LLM) → Alignment (DPO/RLHF)."

---

## Section 1: Instruction Tuning Fundamentals for VLMs

### What is Visual Instruction Tuning?

Visual instruction tuning is supervised fine-tuning on vision-language instruction-following data, teaching VLMs to:
1. Understand natural language instructions about images
2. Generate helpful, detailed, context-aware responses
3. Handle diverse tasks through instruction variations
4. Engage in multi-turn visual conversations

From [Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) (accessed 2025-11-16):
> "We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant."

**Instruction format example:**
```
Input:
  Image: [photo of a cat on a couch]
  Instruction: "What is the cat doing in this image?"

Output:
  "The cat is lying comfortably on a gray couch, appearing to be resting or sleeping."
```

### Differences from Pre-training

| Aspect | Pre-training | Instruction Tuning |
|--------|-------------|-------------------|
| **Data** | Image-text pairs (captions) | (Image, instruction, response) triplets |
| **Objective** | Align vision/language features | Follow natural language instructions |
| **Data scale** | Millions of pairs | 100K-1M instruction examples |
| **Model behavior** | General features | Task-specific responses |
| **Frozen components** | Vision encoder + LLM | Vision encoder only |

From [karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Stage 1: Projector Pre-training - Align vision and language representations with image-caption pairs. Stage 2: Supervised Fine-tuning (SFT) - Learn to follow instructions, handle diverse tasks with instruction-following datasets."

### Task Taxonomy for VLMs

Instruction tuning enables diverse vision-language capabilities:

**1. Visual Question Answering (VQA)**
- Open-ended questions: "What color is the car?"
- Counting: "How many people are in the image?"
- Spatial reasoning: "Is the laptop to the left or right of the cup?"

**2. Image Captioning**
- Detailed descriptions: "Describe this image in detail."
- Specific aspects: "What is the main subject doing?"
- Creative captions: "Write a poetic caption for this scene."

**3. Visual Reasoning**
- Compositional: "Is there a red car AND a blue bike?"
- Relational: "What is the relationship between the dog and the cat?"
- Multi-hop: "If the person picks up the ball, what will happen next?"

**4. Conversational AI**
- Multi-turn dialogue: Follow-up questions about the same image
- Context retention: Remember previous questions in conversation
- Clarification: Ask for more details when needed

From [karpathy-deep-oracle/practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
> "VQA task taxonomy: counting, spatial, compositional, reasoning. Models must handle diverse question types with query-aware relevance realization."

---

## Section 2: Visual Instruction Datasets

### LLaVA-Instruct: GPT-4 Generated Data

From [Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) (accessed 2025-11-16):

LLaVA pioneered using GPT-4 to generate visual instruction data:
- **Method**: Provide GPT-4 with image captions and bounding boxes, prompt it to generate question-answer pairs
- **Scale**: 158K unique instruction-following samples
- **Types**: Conversations (58K), detailed descriptions (23K), complex reasoning (77K)

**Data generation pipeline:**
1. Start with COCO images + captions
2. Use GPT-4 to generate: "Based on this caption, create diverse questions and answers"
3. Filter for quality and diversity
4. Format as instruction-response pairs

Example prompt to GPT-4:
```
Caption: "A person riding a bike on a city street"
Generate 3 question-answer pairs about this image.
```

**Quality characteristics:**
- Natural conversational style (GPT-4's strength)
- Detailed, informative responses
- Diverse question types (what, why, how, count, describe)

### ShareGPT-4V: Real Multi-Modal Conversations

From web research (accessed 2025-11-16):

ShareGPT-4V curates high-quality vision-language conversations:
- **Source**: Real GPT-4V conversations shared by users
- **Scale**: 100K+ diverse multi-turn dialogues
- **Strength**: Natural language variety, creative instructions
- **Format**: Multi-turn conversations with images

Example conversation:
```
Turn 1:
  User: [Image of a sunset] "What time of day is this?"
  Assistant: "This is sunset, based on the warm orange glow..."

Turn 2:
  User: "What makes you say that?"
  Assistant: "The position of the sun low on the horizon and the characteristic orange/red hues..."
```

### Visual Instruction Dataset Creation Pipeline

From [SVIT: Scaling up Visual Instruction Tuning](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning) (accessed 2025-11-16):

**Synthetic generation approach:**
1. **Select source images**: COCO, Visual Genome, web scrapes
2. **Extract visual features**: Captions, object detections, scene graphs
3. **Generate instructions**: Use LLMs (GPT-4, Claude) to create question-answer pairs
4. **Quality filtering**: Remove low-quality, repetitive, or irrelevant samples
5. **Diversity balancing**: Ensure coverage of task types and difficulty levels

**Instruction templates:**
```python
templates = [
    "What is {object} doing in this image?",
    "Describe the relationship between {object1} and {object2}.",
    "How many {objects} are visible?",
    "What would happen if {hypothetical}?",
    "Explain why {reasoning_question}."
]
```

### Domain-Specific Instruction Data

Specialized datasets for targeted applications:

**Medical VQA:**
- **VQA-RAD**: Radiology image questions
- **PathVQA**: Pathology slide questions
- Example: "Is there evidence of pneumonia in this chest X-ray?"

**Document Understanding:**
- **DocVQA**: Questions about document images
- **ChartQA**: Questions about charts and graphs
- Example: "What is the revenue in Q3 according to this chart?"

**Robotics:**
- **Visual instruction for manipulation**: "Pick up the red cube"
- **Navigation instructions**: "Go to the room with the blue chair"

From [karpathy-deep-oracle/practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
> "Dataset curation: web scraping, filtering, deduplication. Image augmentation: random crop, flip, color jitter, RandAugment. Text augmentation: paraphrasing, back-translation, template-based."

---

## Section 3: Full Fine-tuning vs PEFT for VLMs

### Full Fine-tuning Approach

**What gets updated:**
- Vision encoder: Often kept frozen (preserve pre-trained features)
- Vision-language projector: Trained
- Language model: Fully fine-tuned (all parameters)

**Memory requirements:**
- Example (7B parameter VLM):
  - Model weights (FP16): ~14 GB
  - Gradients: ~14 GB
  - Optimizer states (Adam): ~28 GB
  - **Total**: ~56-64 GB (requires A100 80GB)

**Advantages:**
- Maximum model flexibility
- Best performance on complex tasks
- Full adaptation to instruction distribution

**Disadvantages:**
- High GPU memory requirements
- Slower training
- Risk of catastrophic forgetting

From [karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Full fine-tuning: Most capable but memory intensive. Requires large GPU memory (80GB A100 for 7B model). Best performance if resources available."

### LoRA for Vision-Language Models

From [karpathy-deep-oracle/practical-implementation/47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):

**LoRA application strategies for VLMs:**

**Strategy 1: LoRA on Language Decoder Only**
- Freeze vision encoder entirely
- Apply LoRA to LLM attention layers (q_proj, v_proj)
- Minimal parameters (~0.05-0.1% of total)
- Fast training, preserves vision features

**Strategy 2: LoRA on Vision Encoder Only**
- Freeze language model
- Apply LoRA to vision transformer attention
- Adapts visual representations to new domains
- Useful for medical images, satellite imagery

**Strategy 3: LoRA on Both Components**
- Apply LoRA to vision encoder AND language decoder
- Maximum flexibility (~0.1-0.5% trainable)
- Best for significant domain shift

**Recommended rank values:**
- Vision encoder: r=32-64 (rich visual features, high-dimensional)
- Cross-attention: r=16-32 (critical fusion point)
- Language decoder: r=8-16 (text is lower dimensional)

**Example configuration for LLaVA-style VLM:**
```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        # Language model
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Memory savings (7B VLM):**
- Full fine-tuning: ~64 GB
- LoRA (r=16): ~14 GB
- **Reduction**: 78% memory savings

### QLoRA for Extreme Efficiency

From [karpathy-deep-oracle/practical-implementation/47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):

QLoRA combines 4-bit quantization + LoRA:
- **Base model weights**: 4-bit NormalFloat (NF4)
- **LoRA adapters**: FP16
- **Memory**: 7B VLM in ~6 GB (vs 64 GB full fine-tuning)

**QLoRA setup for VLM:**
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load vision-language model with quantization
model = VLMModel.from_pretrained(
    "llava-1.5-7b",
    quantization_config=bnb_config,
    device_map="auto",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

**Performance retention:**
- QLoRA achieves ~99.3% of full fine-tuning performance
- 2-3× slower training (quantization overhead)
- Enables training on consumer GPUs (24GB)

---

## Section 4: Multi-Task Fine-tuning Strategies

### Single-Task vs Multi-Task Instruction Tuning

From [Multi-task instruction fine-tuning](https://discuss.huggingface.co/t/multi-task-instruction-fine-tuning/50310) (accessed 2025-11-16):

**Single-task approach:**
- Train VLM on one task type (e.g., only VQA)
- Simpler data curation
- Risk: Model forgets other capabilities

**Multi-task approach:**
- Train on diverse tasks simultaneously (VQA + captioning + reasoning + OCR)
- Preserves general capabilities
- Better transfer learning

**Multi-task training benefits:**
1. **Prevents catastrophic forgetting**: Model retains pre-trained skills
2. **Improves generalization**: Cross-task knowledge transfer
3. **Single model for multiple tasks**: No task-specific models needed
4. **Better instruction following**: Learns general instruction format

**Dataset mixing strategy:**
```python
# Example multi-task dataset composition
training_data = {
    "vqa": 40_000,        # 40% - Visual question answering
    "caption": 30_000,    # 30% - Image captioning
    "reasoning": 20_000,  # 20% - Visual reasoning
    "ocr": 10_000,        # 10% - Text recognition
}
```

### Task Weighting and Balancing

From web research (accessed 2025-11-16):

**Challenge**: Different tasks have varying difficulty and data availability

**Balancing strategies:**

**1. Equal sampling per task:**
```python
# Sample equally from each task per batch
for batch in dataloader:
    vqa_samples = sample(vqa_dataset, batch_size // 4)
    caption_samples = sample(caption_dataset, batch_size // 4)
    reasoning_samples = sample(reasoning_dataset, batch_size // 4)
    ocr_samples = sample(ocr_dataset, batch_size // 4)
    batch = concat([vqa_samples, caption_samples, reasoning_samples, ocr_samples])
```

**2. Proportional to dataset size:**
```python
# Sample proportionally to dataset size
weights = {
    "vqa": 0.4,      # Largest dataset
    "caption": 0.3,
    "reasoning": 0.2,
    "ocr": 0.1,      # Smallest dataset
}
```

**3. Temperature-based sampling:**
```python
# Use temperature to smooth distribution
def sample_with_temperature(dataset_sizes, temperature=0.5):
    probs = [size ** (1/temperature) for size in dataset_sizes]
    probs = [p / sum(probs) for p in probs]  # Normalize
    return probs
```

### Multi-Task Loss Functions

**Simple approach: Sum of task losses**
```python
def multi_task_loss(model_outputs, targets):
    vqa_loss = cross_entropy(outputs['vqa'], targets['vqa'])
    caption_loss = cross_entropy(outputs['caption'], targets['caption'])
    reasoning_loss = cross_entropy(outputs['reasoning'], targets['reasoning'])

    total_loss = vqa_loss + caption_loss + reasoning_loss
    return total_loss
```

**Weighted loss approach:**
```python
def weighted_multi_task_loss(model_outputs, targets, weights):
    vqa_loss = cross_entropy(outputs['vqa'], targets['vqa'])
    caption_loss = cross_entropy(outputs['caption'], targets['caption'])
    reasoning_loss = cross_entropy(outputs['reasoning'], targets['reasoning'])

    total_loss = (weights['vqa'] * vqa_loss +
                  weights['caption'] * caption_loss +
                  weights['reasoning'] * reasoning_loss)
    return total_loss
```

**Adaptive weighting (uncertainty weighting):**
```python
# Learn task weights during training
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        # Uncertainty-based weighting
        precision = torch.exp(-self.log_vars)
        loss = sum(precision[i] * losses[i] + self.log_vars[i]
                   for i in range(len(losses)))
        return loss
```

---

## Section 5: Continual Learning and Catastrophic Forgetting

### The Catastrophic Forgetting Problem

**What happens:**
When fine-tuning VLMs on narrow instruction distributions, models can forget:
- General vision understanding
- Language generation quality
- Pre-trained task capabilities
- Cross-domain knowledge

**Example failure:**
```
Before instruction tuning:
  Q: "Describe this medical X-ray"
  A: "I see a chest X-ray showing..."

After instruction tuning on only object detection:
  Q: "Describe this medical X-ray"
  A: "Person, bed, medical equipment" (lost detailed description ability)
```

### Mitigation Strategies

**1. Replay Buffer with Pre-training Data**
```python
# Mix instruction data with original pre-training data
training_data = {
    "instruction_data": 100_000,    # New instruction examples
    "pretrain_replay": 20_000,      # 20% replay from pre-training
}
```

**2. Regularization Techniques**

**Elastic Weight Consolidation (EWC):**
- Penalize changes to important pre-trained parameters
- Identify important weights from pre-training phase
```python
# Simplified EWC loss
ewc_loss = sum(fisher_info[param] * (param - old_param)**2
               for param in model.parameters())
total_loss = task_loss + lambda_ewc * ewc_loss
```

**Knowledge Distillation:**
- Keep original pre-trained model as "teacher"
- Distill knowledge during fine-tuning
```python
# Distillation loss
teacher_logits = teacher_model(inputs)  # Frozen pre-trained model
student_logits = student_model(inputs)  # Fine-tuning model
distill_loss = KL_divergence(student_logits, teacher_logits)

total_loss = task_loss + alpha * distill_loss
```

**3. Multi-Task Training (Best Practice)**

From web research (accessed 2025-11-16):
> "It's actually a good idea to fine tune a model with several different tasks, so the model doesn't lose the skills it had before."

Mix diverse tasks to maintain capabilities:
```python
# Diverse task mixture preserves general capabilities
tasks = [
    "vqa",           # Visual QA
    "caption",       # Captioning
    "reasoning",     # Visual reasoning
    "ocr",           # Text recognition
    "general_chat",  # General conversation (from pre-training)
]
```

---

## Section 6: Domain Adaptation for VLMs

### When Domain Adaptation is Needed

VLMs pre-trained on natural images (COCO, ImageNet) may underperform on:
- **Medical images**: X-rays, CT scans, pathology slides
- **Satellite imagery**: Remote sensing, aerial views
- **Document images**: Receipts, forms, charts
- **Robotic vision**: First-person manipulation views
- **Industrial inspection**: Manufacturing defects, quality control

**Domain shift characteristics:**
- Different visual distributions (grayscale vs color, specialized lighting)
- Domain-specific terminology and concepts
- Different task requirements (diagnosis vs object detection)

### Domain-Specific Instruction Tuning

**Approach 1: LoRA on Vision Encoder Only**

From [karpathy-deep-oracle/practical-implementation/47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md):
> "Strategy 2: LoRA on Vision Encoder Only. Use when: Domain shift from pre-training data (e.g., medical images, satellite imagery). Adapts visual representations to new domains."

**Example: Medical VLM adaptation**
```python
# Apply LoRA to vision encoder for medical images
vision_lora_config = LoraConfig(
    r=32,  # Higher rank for rich visual features
    target_modules=["vision_model.encoder.layers.*.self_attn.q_proj",
                    "vision_model.encoder.layers.*.self_attn.v_proj"],
)

# Freeze language model, adapt vision encoder
model.language_model.requires_grad_(False)
model = get_peft_model(model, vision_lora_config)
```

**Approach 2: Domain-Specific Instruction Data**

Create specialized instruction datasets:
```python
# Medical VQA instruction examples
medical_instructions = [
    {
        "image": "chest_xray_001.jpg",
        "instruction": "Is there evidence of pneumonia in this chest X-ray?",
        "response": "Yes, there are bilateral infiltrates consistent with pneumonia..."
    },
    {
        "image": "pathology_slide_042.jpg",
        "instruction": "Describe the cellular abnormalities in this tissue sample.",
        "response": "The sample shows pleomorphic cells with increased nuclear-to-cytoplasmic ratio..."
    }
]
```

### Incremental Domain Adaptation

**Progressive fine-tuning strategy:**
1. **Stage 1**: Fine-tune on general instruction data (preserve broad capabilities)
2. **Stage 2**: Introduce domain-specific instructions gradually
3. **Stage 3**: Full domain-specific fine-tuning

**Data mixing schedule:**
```python
# Gradually increase domain-specific data proportion
epoch_1_4:   general 80%, domain 20%
epoch_5_8:   general 60%, domain 40%
epoch_9_12:  general 40%, domain 60%
epoch_13+:   general 20%, domain 80%
```

---

## Section 7: Hyperparameter Tuning for Instruction Fine-tuning

### Learning Rate Selection

From [karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):

**Different learning rates for different components:**

**New parameters (projectors/adapters):**
- Learning rate: 1e-3 to 1e-4
- Reason: Randomly initialized, need larger updates

**Pre-trained language model (if unfrozen):**
- Learning rate: 1e-5 to 5e-6
- Reason: Already well-trained, avoid catastrophic forgetting

**Pre-trained vision encoder (if unfrozen):**
- Learning rate: 1e-5 to 1e-6
- Reason: Visual features are precious, update gently

**Example multi-component optimizer:**
```python
optimizer = torch.optim.AdamW([
    {
        'params': [p for n, p in model.named_parameters()
                   if 'projector' in n and p.requires_grad],
        'lr': 1e-3,
        'weight_decay': 0.0
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if 'language_model' in n and p.requires_grad],
        'lr': 2e-5,
        'weight_decay': 0.1
    }
])
```

### Warmup Strategies

From [karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Warmup prevents early overfitting. New projector parameters are randomly initialized. Large initial gradients can destabilize frozen pre-trained models. Linear warmup over 500-2000 steps."

**Warmup schedule:**
```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,   # Gradually increase LR
    num_training_steps=10000
)
```

**Why warmup matters for VLMs:**
1. Vision-language projector starts from random initialization
2. Large initial gradients could corrupt frozen vision/language features
3. Allows projector to "catch up" to frozen components

### Batch Size Considerations

**Effective batch size:**
- Recommended: 64-256 for instruction tuning
- Constraint: GPU memory limits physical batch size to 4-8

**Gradient accumulation solution:**
```python
accumulation_steps = 32  # Effective batch = 8 * 32 = 256

for i, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

From [karpathy-deep-oracle/practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
> "Gradient accumulation: Physical batch size 4-8, accumulation steps 8-32, effective batch size 64-256. Trade compute time for memory."

### Training Duration

**General guidelines:**
- **Quick adaptation**: 1-3 epochs (sufficient for in-distribution tasks)
- **Domain shift**: 5-10 epochs (medical, satellite imagery)
- **Complex reasoning**: 10-20 epochs (mathematical, multi-hop reasoning)

**Early stopping:**
```python
# Stop if validation performance plateaus
patience = 3  # Number of epochs without improvement
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        save_checkpoint(model)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered")
        break
```

---

## Section 8: ARR-COC-0-1 Instruction Fine-tuning Strategy

### Relevance-Aware Instruction Tuning

ARR-COC-0-1 uses Vervaekean relevance realization during instruction fine-tuning:

**Approach:**
1. **Query-aware visual processing**: Instructions guide relevance allocation
2. **Dynamic token budgets**: 64-400 tokens per patch based on instruction relevance
3. **Multi-task learning**: VQA + captioning + reasoning preserve general capabilities

**Fine-tuning configuration:**
```python
# ARR-COC-0-1 instruction tuning setup
config = {
    "vision_encoder": "frozen",           # Preserve CLIP features
    "relevance_allocator": "trainable",   # Adapt relevance realization
    "language_model": "lora_r16",         # Efficient LLM adaptation
    "learning_rates": {
        "relevance_allocator": 1e-3,
        "language_lora": 2e-5,
    },
    "multi_task_mixture": {
        "vqa": 0.4,
        "caption": 0.3,
        "reasoning": 0.3,
    }
}
```

### Instruction-Guided Relevance Allocation

**How instructions modulate relevance:**
```python
# Example: Instruction changes relevance allocation
instruction_1 = "What color is the car?"
# → Focus relevance on car region, allocate more tokens to car patch

instruction_2 = "Describe the entire scene in detail."
# → Distribute relevance broadly, allocate tokens across all patches

instruction_3 = "Count the number of people."
# → Focus relevance on person regions, allocate tokens to person patches
```

**Opponent processing during instruction tuning:**
- **Compress ↔ Particularize**: Specific questions focus tokens, broad questions distribute
- **Exploit ↔ Explore**: Familiar tasks exploit, novel reasoning explores
- **Focus ↔ Diversify**: Count tasks focus, description tasks diversify

### Training Protocol for ARR-COC-0-1

**Stage 1: Relevance Allocator Pre-training (Frozen Backbones)**
- Duration: 1-2 days on 8 GPUs
- Data: COCO captions (80K images)
- Goal: Learn basic query-aware token allocation
- Frozen: Vision encoder + LLM
- Trainable: Relevance allocator only

**Stage 2: Multi-Task Instruction Tuning (LoRA on LLM)**
- Duration: 3-5 days on 8 GPUs
- Data: VQA (40K) + Captions (30K) + Reasoning (30K) = 100K total
- Goal: Follow diverse instructions with relevance-aware processing
- Frozen: Vision encoder
- Trainable: Relevance allocator + LLM (LoRA r=16)

From [karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md):
> "Using LoRA to adapt the parameters of the unimodal backbones while using standard fine-tuning for the new parameters yields more stable training runs."

**Evaluation metrics:**
- VQA accuracy on VQAv2 validation set
- CIDEr score on COCO Captions
- Visual reasoning accuracy on custom benchmark
- Token efficiency: Average tokens per image vs baseline

From [karpathy-deep-oracle/practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
> "VQA accuracy formula: min(# humans that said answer / 3, 1.0). At least 3 humans must agree for 100% accuracy. Partial credit for 1-2 agreements."

**Expected performance:**
- VQAv2 accuracy: 65-70% (competitive with LLaVA-1.5)
- COCO CIDEr: 100-110 (strong captioning)
- Token efficiency: 30-50% reduction vs fixed 576 tokens
- Inference latency: <200ms per image (relevance allocation cached)

---

## Sources

### Source Documents

- [karpathy/practical-implementation/46-frozen-backbone-adapter-training.md](../karpathy/practical-implementation/46-frozen-backbone-adapter-training.md) - Multi-stage training, LoRA strategies, learning rates
- [karpathy/practical-implementation/47-lora-low-rank-adaptation.md](../karpathy/practical-implementation/47-lora-low-rank-adaptation.md) - LoRA for VLMs, QLoRA, rank selection
- [karpathy/practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md) - VQA training, evaluation metrics

### Web Research

**Academic Papers:**
- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - arXiv:2304.08485, Liu et al., 2023 (accessed 2025-11-16) - LLaVA, GPT-4 generated instruction data
- [Improved Baselines with Visual Instruction Tuning](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf) - Liu et al., CVPR 2024 (accessed 2025-11-16) - LLaVA-1.5 improvements
- [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/abs/2308.10792) - arXiv:2308.10792, Zhang et al., 2023 (accessed 2025-11-16) - Comprehensive instruction tuning survey

**Code Repositories:**
- [SVIT: Scaling up Visual Instruction Tuning](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning) - GitHub (accessed 2025-11-16) - 4.2M visual instruction dataset
- [LLaVA Official Repository](https://github.com/haotian-liu/LLaVA) - GitHub (accessed 2025-11-16) - Visual instruction tuning implementation
- [LLaMA-Factory: Unified Efficient Fine-Tuning](https://github.com/hiyouga/LLaMA-Factory) - GitHub (accessed 2025-11-16) - 100+ LLMs & VLMs fine-tuning framework

**Tutorials & Documentation:**
- [Multi-task instruction fine-tuning](https://discuss.huggingface.co/t/multi-task-instruction-fine-tuning/50310) - HuggingFace Forums (accessed 2025-11-16) - Multi-task training best practices
- [How to Fine-Tune Multimodal Models or VLMs](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl) - Philschmid (accessed 2025-11-16) - Practical VLM fine-tuning guide
- [Visual Instruction Tuning Explained](https://zilliz.com/blog/llava-visual-instruction-training) - Zilliz (accessed 2025-11-16) - LLaVA architecture and training

**Additional Resources:**
- [LLaVA Official Website](https://llava-vl.github.io/) (accessed 2025-11-16) - Demos and model information
- [Vision-Flan Dataset](https://vision-flan.github.io/) (accessed 2025-11-16) - 200+ diverse vision-language tasks

---

*Last updated: 2025-11-16*
*Knowledge domain: Vision-language model fine-tuning, instruction tuning, multi-task learning, PEFT methods*
