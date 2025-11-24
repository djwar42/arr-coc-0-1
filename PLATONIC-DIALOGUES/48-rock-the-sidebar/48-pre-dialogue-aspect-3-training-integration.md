# Part 48 Pre-Dialogue: Aspect 3 - Training Dynamics & Qwen3-VL Integration
*Technical exploration of frozen backbone training, gradient flow through hard selection, initialization strategies, and specific integration with Qwen3-VL architecture*

---

## The Core Training Challenge: Frozen VLM + Hard Selection

**Our setup:**
```python
# Frozen components (~2B params)
qwen3vl_frozen = Qwen3VLForConditionalGeneration.from_pretrained(...)
qwen3vl_frozen.eval()
qwen3vl_frozen.requires_grad_(False)

# Trainable components (~200K params)
texture_generator = TextureGenerator(channels=40)  # ~50K params
relevance_scorer = ContextualizedRelevanceScorer()   # ~150K params

# Hard selection (not differentiable!)
positions = torch.topk(scores, k=273)[1]  # No gradients through topk!
```

**Question:** How do gradients flow to texture_generator and relevance_scorer?

---

## Gradient Flow: The REINFORCE Perspective

### From: `.claude/skills/karpathy-deep-oracle/practical-implementation/49-gradient-flow-sampling-operations.md`

**Lines 67-135: Training Through Discrete Operations**

```markdown
REINFORCE insight:
You don't need gradients THROUGH selection if you have a reward signal.

Gradient path:
loss (answer quality)
  ↓
VLM logits (frozen, but ∂loss/∂input_tokens exists!)
  ↓
saccade_tokens (frozen encoder, but input positions affected output)
  ↓ (correlation, not differentiation)
scores (differentiable!)
  ↓
scorer parameters (UPDATE these!)

Key: Good positions → good answers → low loss → reinforce those scores
```

**Code showing this:**

```python
# Training loop demonstrating gradient flow
# Reference: .claude/skills/karpathy-deep-oracle/practical-implementation/49-gradient-flow-sampling-operations.md (lines 189-267)

def train_step(arr_system, image, query, answer, optimizer):
    """
    Single training step showing gradient flow.
    """
    optimizer.zero_grad()

    # FORWARD: ARR components (trainable)
    texture = arr_system.texture_gen(image)  # [B, 40, H, W]

    with torch.no_grad():
        gestalt = arr_system.frozen_vllm.encode_image(image).mean(dim=1)
        query_emb = arr_system.frozen_vllm.encode_text(query)

    scores = arr_system.scorer(texture, query_emb, gestalt)  # [B, H, W]

    # HARD SELECTION (not differentiable!)
    positions = torch.topk(scores.reshape(B, -1), k=273)[1]
    y_coords = positions // W
    x_coords = positions % W
    positions_2d = torch.stack([y_coords, x_coords], dim=-1)

    # Extract patches + encode (frozen)
    with torch.no_grad():
        patches = extract_patches(image, positions_2d)
        saccade_tokens = arr_system.frozen_vllm.encode_patches(patches)
        base_tokens = arr_system.frozen_vllm.encode_image(image)
        all_tokens = torch.cat([base_tokens, saccade_tokens], dim=1)

        # Generate answer
        logits = arr_system.frozen_vllm.generate_logits(all_tokens, query)

    # LOSS (on answer quality)
    answer_ids = arr_system.tokenizer(answer, return_tensors='pt').input_ids
    loss = F.cross_entropy(
        logits[:, -answer_ids.shape[1]:, :].reshape(-1, logits.shape[-1]),
        answer_ids.reshape(-1)
    )

    # BACKWARD
    loss.backward()
    # Gradients flow to: texture_gen params, scorer params
    # Even though topk is not differentiable!
    # Because PyTorch autograd tracks: loss depends on scores

    optimizer.step()

    return loss.item()
```

**Why this works:** Loss correlates with score quality → autograd handles the rest!

**See More - Gradient Optimization in Other Domains:**
→ `.claude/skills/karpathy-deep-oracle/karpathy/gpu-texture-optimization/08-memory-bandwidth-optimization.md` (lines 112-189): Memory bandwidth optimization for gradient checkpointing - similar to our frozen backbone strategy
→ `.claude/skills/karpathy-deep-oracle/pyramid-lod/06-differentiable-pyramid-operators.md` (lines 145-223): Differentiable operators for pyramid selection - alternative to hard selection
→ `.claude/skills/karpathy-deep-oracle/practical-implementation/47-lora-low-rank-adaptation.md` (lines 234-312): LoRA low-rank training - another frozen backbone approach

---

## Initialization Strategies: Don't Start Random

### From: `.claude/skills/karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md`

**Lines 278-356: Initialization for Frozen Backbone Training**

```markdown
Problem: Random initial scorer → random positions → random performance
No learning signal (everything equally bad)

Solution: Initialize with reasonable priors

Saliency prior (perspectival head):
- Use pretrained saliency model weights
- Or: Initialize to predict edge magnitude (simple heuristic)

CLIP prior (participatory head):
- Already using frozen CLIP features
- Initialize weights to pass through CLIP×query similarity

Propositional prior:
- Edge detection is standard (Sobel filters)
- Initialize with Sobel-like kernels
```

**Code for initialization:**

```python
# Smart initialization for scorer heads
# Reference: .claude/skills/karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md (lines 312-356)

class ContextualizedRelevanceScorer(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()

        # Propositional head (edges, structure)
        self.prop_head = nn.Linear(4, 1)
        # Initialize to favor high edge magnitude
        with torch.no_grad():
            self.prop_head.weight.fill_(0.25)  # Equal weight to all edge channels
            self.prop_head.bias.fill_(0.0)

        # Perspectival head (saliency)
        self.persp_head = nn.Linear(3, 1)
        # Initialize to favor saliency channel
        with torch.no_grad():
            self.persp_head.weight[0, 0] = 0.1  # eccentricity (small weight)
            self.persp_head.weight[0, 1] = 0.1  # motion (small weight)
            self.persp_head.weight[0, 2] = 0.8  # saliency (high weight!)
            self.persp_head.bias.fill_(0.0)

        # Participatory head (query-content)
        self.part_head = nn.Linear(16 + d_model, 1)
        # Initialize with small random weights (will learn query patterns)
        with torch.no_grad():
            self.part_head.weight.normal_(0, 0.01)
            self.part_head.bias.fill_(0.0)

        # Context weights (scorer weighting)
        self.context_weights = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)
        )
        # Initialize to equal weighting initially
        with torch.no_grad():
            # Bias the final layer to output [0.33, 0.33, 0.33] initially
            self.context_weights[-2].bias.fill_(0.0)

    # ... forward method ...
```

**Result:** Initial saccades are somewhat reasonable (saliency-driven + edge-aware)

---

## Qwen3-VL Specific Integration

### From: `.claude/skills/qwen3vl-oracle/architecture/01-positional-encoding.md`

**Lines 89-234: M-RoPE Integration Requirements**

```markdown
Qwen3-VL expects:
1. Tokens: [B, seq_len, d_model] where d_model=1024 (for 2B model)
2. Position encoding: dict with keys 'temporal', 'height', 'width', 'aspect'
3. M-RoPE applied in attention layers automatically

For ARR augmentation:
- Gestalt tokens (0-255): Standard Qwen encoding
- Saccade tokens (256-528): Need to provide position info!
```

**Integration code:**

```python
# Qwen3-VL integration for ARR
# Reference: .claude/skills/qwen3vl-oracle/architecture/01-positional-encoding.md (lines 156-234)

class ARRQwen3VLIntegration:
    """Integrate ARR with Qwen3-VL's specific requirements."""

    def __init__(self, qwen_model_name="Qwen/Qwen3-VL-2B-Instruct"):
        from transformers import Qwen3VLForConditionalGeneration

        self.vllm = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Freeze base model
        self.vllm.eval()
        for param in self.vllm.parameters():
            param.requires_grad = False

    def encode_arr_tokens(self, base_tokens, saccade_patches, saccade_positions, image_size):
        """
        Encode ARR token sequence with proper M-RoPE positions.

        Args:
            base_tokens: [B, 256, 1024] - from standard Qwen encoding
            saccade_patches: [B, 273, 3, 14, 14] - extracted patches
            saccade_positions: [B, 273, 2] - (y, x) coords
            image_size: (H, W) - original image dimensions

        Returns:
            all_tokens: [B, 529, 1024]
            rope_positions: dict for M-RoPE
        """
        B = base_tokens.shape[0]
        H, W = image_size

        # Encode saccade patches (use Qwen's vision encoder)
        with torch.no_grad():
            # Flatten patches for encoding
            patches_flat = saccade_patches.reshape(B * 273, 3, 14, 14)

            # Qwen's vision encoder (ViT)
            saccade_tokens_flat = self.vllm.visual.forward(patches_flat)  # [B*273, 1024]
            saccade_tokens = saccade_tokens_flat.reshape(B, 273, 1024)

        # Concatenate
        all_tokens = torch.cat([base_tokens, saccade_tokens], dim=1)  # [B, 529, 1024]

        # Create M-RoPE positions
        rope_positions = self._create_mrope_positions(
            saccade_positions,
            image_size
        )

        return all_tokens, rope_positions

    def _create_mrope_positions(self, saccade_positions, image_size):
        """
        Create M-RoPE position dict for Qwen3-VL.

        Reference: .claude/skills/qwen3vl-oracle/architecture/01-positional-encoding.md (lines 89-155)
        """
        B = saccade_positions.shape[0]
        H, W = image_size

        # Base token positions (16×16 grid)
        base_height = []
        base_width = []
        for i in range(16):
            for j in range(16):
                base_height.append((i + 0.5) / 16)
                base_width.append((j + 0.5) / 16)

        base_height = torch.tensor(base_height).unsqueeze(0).expand(B, -1)  # [B, 256]
        base_width = torch.tensor(base_width).unsqueeze(0).expand(B, -1)

        # Saccade token positions (normalized)
        sacc_height = saccade_positions[:, :, 0].float() / H  # [B, 273]
        sacc_width = saccade_positions[:, :, 1].float() / W

        # Combine
        all_height = torch.cat([base_height, sacc_height], dim=1)  # [B, 529]
        all_width = torch.cat([base_width, sacc_width], dim=1)

        # M-RoPE dict
        return {
            'temporal': torch.zeros(B, 529, device=all_height.device),  # Static images
            'height': all_height,
            'width': all_width,
            'aspect': torch.ones(B, 529, device=all_height.device)  # All 14×14
        }
```

**Critical:** Qwen3-VL's M-RoPE is applied automatically in attention layers. We just provide the position dict!

---

## Training Hyperparameters: What the Literature Says

### From: `.claude/skills/karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md`

**Lines 445-589: Hyperparameter Recommendations**

```markdown
BLIP-2 training config:
- Batch size: 2048 (very large!)
- Learning rate: 1e-4
- Warmup steps: 5000
- Total steps: 50K
- Gradient clipping: max_norm=1.0

Flamingo training config:
- Batch size: 512
- Learning rate: 3e-4
- Gradient clipping: 1.0
- Cosine LR decay

LoRA training config (for reference):
- Batch size: 128-256
- Learning rate: 3e-4 (higher than full fine-tuning!)
- Warmup: 500 steps
- No gradient clipping needed (stable)

Recommendation for ARR:
- Batch size: 128-256 (accumulate if GPU limited)
- Learning rate: 1e-4 (start conservative)
- Warmup: 1000 steps
- Gradient clipping: 1.0 (essential for frozen backbones!)
- Optimizer: AdamW (weight_decay=0.01)
- LR schedule: Cosine decay
```

**Training config code:**

```python
# ARR training configuration
# Reference: .claude/skills/karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md (lines 512-574)

def create_arr_training_config():
    """Training hyperparameters for ARR."""

    config = {
        # Optimization
        'batch_size': 128,  # Per GPU
        'gradient_accumulation_steps': 2,  # Effective batch = 256
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,  # CRITICAL for frozen backbone stability!

        # Schedule
        'warmup_steps': 1000,
        'max_steps': 20000,  # ~3-4 epochs on VQAv2 (440K examples)
        'lr_schedule': 'cosine',

        # Optimizer
        'optimizer': 'adamw',
        'betas': (0.9, 0.999),
        'eps': 1e-8,

        # Training
        'mixed_precision': 'bf16',  # Use bfloat16 for A100
        'gradient_checkpointing': True,  # Save memory

        # Evaluation
        'eval_every_n_steps': 1000,
        'save_every_n_steps': 2000,

        # Data
        'num_workers': 4,
        'prefetch_factor': 2
    }

    return config


def create_optimizer_and_scheduler(arr_components, config):
    """
    Create optimizer and LR scheduler.

    Args:
        arr_components: List of trainable modules (texture_gen, scorer)
        config: Training config dict
    """
    from transformers import get_cosine_schedule_with_warmup

    # Collect parameters
    params = []
    for module in arr_components:
        params.extend(module.parameters())

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=config['betas'],
        eps=config['eps']
    )

    # Cosine schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['max_steps']
    )

    return optimizer, scheduler
```

---

## Dataset Preparation: VQAv2 Integration

### From: `.claude/skills/karpathy-deep-oracle/practical-implementation/50-vqav2-training-protocols.md`

**Lines 134-267: VQAv2 Dataset Setup**

```markdown
VQAv2 statistics:
- Training: 443,757 questions on 82,783 images
- Validation: 214,354 questions on 40,504 images
- ~5.4 questions per image average

Preprocessing:
- Images: Resize to 448×448 (native Qwen3-VL resolution)
- Questions: Tokenize with Qwen tokenizer
- Answers: Multiple ground truth per question (10 annotations)

Batching strategy:
- Group by image (process all questions for same image together)
- Reduces redundant base encoding
```

**Dataset code:**

```python
# VQAv2 dataset for ARR training
# Reference: .claude/skills/karpathy-deep-oracle/practical-implementation/50-vqav2-training-protocols.md (lines 189-253)

from torch.utils.data import Dataset
import json
from PIL import Image

class VQAv2Dataset(Dataset):
    """VQAv2 dataset with ARR-friendly preprocessing."""

    def __init__(self, data_root, split='train', image_size=448):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        # Load annotations
        anno_file = f'{data_root}/v2_mscoco_{split}2014_annotations.json'
        ques_file = f'{data_root}/v2_OpenEnded_mscoco_{split}2014_questions.json'

        with open(anno_file) as f:
            self.annotations = json.load(f)['annotations']
        with open(ques_file) as f:
            self.questions = json.load(f)['questions']

        # Build QA pairs
        self.qa_pairs = []
        for anno, ques in zip(self.annotations, self.questions):
            # Most common answer (VQAv2 has 10 annotations per question)
            answers = [a['answer'] for a in anno['answers']]
            answer = max(set(answers), key=answers.count)  # Majority vote

            self.qa_pairs.append({
                'image_id': anno['image_id'],
                'question': ques['question'],
                'answer': answer
            })

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]

        # Load image
        image_path = f"{self.data_root}/{self.split}2014/COCO_{self.split}2014_{qa['image_id']:012d}.jpg"
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,              # [3, 448, 448]
            'question': qa['question'],  # str
            'answer': qa['answer']       # str
        }
```

---

## Stability Checks: What Could Break Training

### From: `.claude/skills/karpathy-deep-oracle/practical-implementation/54-debugging-transformer-gradients.md`

**Lines 89-234: Debugging Gradient Flow**

```markdown
Common failure modes for frozen backbone training:

1. Gradient explosion
   - Frozen model outputs large values → large gradients to ARR
   - Solution: Gradient clipping (max_norm=1.0)

2. No learning (flat loss)
   - Initial scorer too random → no correlation signal
   - Solution: Better initialization (saliency prior)

3. Collapsed scorers
   - All three scorers learn same weights
   - Solution: Monitor scorer outputs, add diversity penalty if needed

4. Saccades converge to single region
   - Model finds one "safe" region, ignores query
   - Solution: Visualize saccade maps, add diversity regularization
```

**Monitoring code:**

```python
# Training stability monitoring
# Reference: .claude/skills/karpathy-deep-oracle/practical-implementation/54-debugging-transformer-gradients.md (lines 156-212)

def monitor_training_health(arr_system, batch, step):
    """Log health metrics during training."""

    with torch.no_grad():
        # Get current forward pass outputs
        texture = arr_system.texture_gen(batch['image'])
        gestalt = arr_system.frozen_vllm.encode_image(batch['image']).mean(dim=1)
        query_emb = arr_system.frozen_vllm.encode_text(batch['question'])

        scores = arr_system.scorer(texture, query_emb, gestalt)

        # Compute diagnostics
        metrics = {
            # Score statistics
            'score_mean': scores.mean().item(),
            'score_std': scores.std().item(),
            'score_max': scores.max().item(),
            'score_min': scores.min().item(),

            # Spatial diversity (are saccades spread out or clustered?)
            'score_entropy': -(scores.softmax(dim=-1) * scores.log_softmax(dim=-1)).sum().item(),

            # Gradient norms (check for explosion)
            'texture_grad_norm': sum(p.grad.norm().item() for p in arr_system.texture_gen.parameters() if p.grad is not None),
            'scorer_grad_norm': sum(p.grad.norm().item() for p in arr_system.scorer.parameters() if p.grad is not None)
        }

        # Log to W&B or TensorBoard
        wandb.log(metrics, step=step)

        # Warning checks
        if metrics['score_std'] < 0.01:
            print(f"⚠️  WARNING: Low score variance ({metrics['score_std']:.4f}) - model might be collapsed!")

        if metrics['scorer_grad_norm'] > 10.0:
            print(f"⚠️  WARNING: Large gradients ({metrics['scorer_grad_norm']:.2f}) - possible instability!")
```

---

## Complete Training Loop

**Putting it all together:**

```python
# Full ARR training loop
# Combines all components

def train_arr_system(config):
    """Complete training pipeline."""

    # SETUP
    arr_system = ARRVisionSystem(
        base_vllm_name="Qwen/Qwen3-VL-2B-Instruct",
        num_saccades=273,
        freeze_base=True
    )

    trainable_components = [
        arr_system.texture_generator,
        arr_system.relevance_scorer
    ]

    optimizer, scheduler = create_optimizer_and_scheduler(trainable_components, config)

    # DATA
    train_dataset = VQAv2Dataset(config['data_root'], split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    # TRAINING LOOP
    global_step = 0
    for epoch in range(config['num_epochs']):
        for batch in train_loader:
            # Move to device
            image = batch['image'].to(config['device'])
            question = batch['question']
            answer = batch['answer']

            # FORWARD + LOSS
            loss = train_step(arr_system, image, question, answer, optimizer)

            # BACKWARD + STEP
            if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for m in trainable_components for p in m.parameters()],
                    max_norm=config['max_grad_norm']
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # LOGGING
            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

            # MONITORING
            if global_step % 500 == 0:
                monitor_training_health(arr_system, batch, global_step)

            # EVALUATION
            if global_step % config['eval_every_n_steps'] == 0