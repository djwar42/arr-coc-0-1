# Intensive Intelligence: Design Principles

**How to build models that are smart, not just big**

## Overview

You've seen the theory (file 06), measured the metrics (file 07), studied the successes (file 08). Now: **how do you actually design a model with intensive intelligence?**

This file provides **4 core design principles** with concrete architectural patterns, code examples, and decision trees. These aren't abstract guidelines—they're battle-tested rules that led to DeepSeek-V3, ARR-COC, and other intensive intelligence breakthroughs.

**The meta-principle**: Every design decision should optimize configuration (intensive) before capacity (extensive).

---

## Principle 1: Sparse Activation Architectures

### The Core Idea

**Bad**: Activate all parameters for all inputs.
**Good**: Activate only relevant parameters for each input.

**Why it's intensive**: Activation patterns are configuration. Which params fire is more important than how many params exist.

### Pattern 1A: Mixture of Experts (MoE)

**Architecture**:
```python
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts=256, expert_dim=4096, top_k=8):
        super().__init__()
        self.experts = nn.ModuleList([
            FFN(expert_dim) for _ in range(num_experts)
        ])
        self.router = nn.Linear(expert_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # Compute routing scores
        routing_logits = self.router(x)  # [batch, seq, num_experts]

        # Top-k selection (intensive property!)
        routing_weights, expert_ids = torch.topk(
            routing_logits, k=self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Sparse activation
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_mask = (expert_ids[..., i:i+1])
            expert_output = self.experts[expert_mask](x)
            output += routing_weights[..., i:i+1] * expert_output

        return output
```

**Key intensive property**: `top_k / num_experts` ratio
```
DeepSeek-V3: 8 / 256 = 3.1% activation
GLaM: 2 / 64 = 3.1% activation
Switch Transformer: 1 / 128 = 0.8% activation

Optimal range: 1-10% activation
```

### Pattern 1B: Conditional Computation

**Architecture** (early exit):
```python
class ConditionalTransformer(nn.Module):
    def __init__(self, num_layers=24, exit_threshold=0.9):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(num_layers)
        ])
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.exit_threshold = exit_threshold

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Check if confident enough to exit early
            logits = self.exit_classifiers[i](x)
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values

            if confidence.mean() > self.exit_threshold:
                # Exit early (intensive intelligence!)
                return logits, i  # Return layer index too

        # Went through all layers
        return self.exit_classifiers[-1](x), len(self.layers)
```

**Intensive property**: Average exit layer
```
Easy examples: Exit at layer 4-8 (33% of network)
Hard examples: Exit at layer 20-24 (100% of network)
Average: ~12 layers (50% activation)

Configuration (which layers for which inputs) > capacity (total layers)
```

### Pattern 1C: Dynamic Depth

**Architecture** (SkipNet):
```python
class DynamicDepthNet(nn.Module):
    def __init__(self, num_layers=18):
        super().__init__()
        self.layers = nn.ModuleList([
            ResNetBlock() for _ in range(num_layers)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2)  # Skip or execute
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        skipped_layers = []
        for i, (layer, gate) in enumerate(zip(self.layers, self.gates)):
            # Decide: skip or execute?
            gate_logits = gate(x)
            gate_probs = F.gumbel_softmax(gate_logits, hard=True)

            # Execute layer with probability
            if gate_probs[1] > 0.5:  # Execute
                x = layer(x)
            else:  # Skip (intensive efficiency!)
                skipped_layers.append(i)

        return x, skipped_layers
```

**Training tip**: Encourage skipping with sparsity regularization:
```python
def sparse_loss(output, target, skip_decisions, sparsity_weight=0.1):
    task_loss = F.cross_entropy(output, target)

    # Penalize executing too many layers
    execution_ratio = (1 - skip_decisions).float().mean()
    sparsity_loss = execution_ratio

    return task_loss + sparsity_weight * sparsity_loss
```

### Design Decision Tree

```
Question 1: Is your input diversity high?
├─ Yes (language, vision, multimodal)
│  └─ Use MoE (different experts for different inputs)
└─ No (single domain, e.g., ImageNet)
   └─ Question 2: Do easy/hard examples differ significantly?
      ├─ Yes → Use early exit or dynamic depth
      └─ No → Dense model may be fine (but consider distillation later)

Question 3: What's your activation budget?
├─ Very sparse (1-5%) → MoE with top-1 or top-2
├─ Moderate (10-30%) → Early exit or dynamic depth
└─ Flexible → Start dense, prune based on metrics (P3, P2F)
```

---

## Principle 2: Hierarchical Organization

### The Core Idea

**Bad**: Flat, single-scale processing.
**Good**: Multi-scale hierarchy (coarse → fine).

**Why it's intensive**: Hierarchy is configuration—*how* information flows through scales, not how many scales exist.

### Pattern 2A: Coarse-to-Fine Refinement

**Architecture** (pyramid):
```python
class PyramidNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Coarse level (low resolution, high capacity)
        self.coarse = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4),  # 1/4 resolution
            ResNetBlock(64),
            ResNetBlock(64)
        )

        # Medium level
        self.medium = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2),  # 1/2 resolution
            ResNetBlock(128),
            ResNetBlock(128)
        )

        # Fine level (high resolution, low capacity)
        self.fine = nn.Sequential(
            nn.Conv2d(128, 256, 3),  # Full resolution
            ResNetBlock(256)
        )

    def forward(self, x):
        # Hierarchical processing (intensive!)
        coarse = self.coarse(x)  # Quick, rough understanding

        # Upscale and refine
        medium = self.medium(coarse)
        medium = F.interpolate(medium, scale_factor=2)

        # Final refinement
        fine = self.fine(medium)
        fine = F.interpolate(fine, scale_factor=4)

        return fine
```

**Intensive property**: FLOPs distribution across scales
```
Coarse (1/4 res): 10% of FLOPs, 40% of understanding
Medium (1/2 res): 30% of FLOPs, 40% of understanding
Fine (full res): 60% of FLOPs, 20% of understanding

Configuration: Focus FLOPs where needed (coarse for context, fine for details)
```

### Pattern 2B: Feature Pyramid Networks (FPN)

**Architecture**:
```python
class FPN(nn.Module):
    """
    Multi-scale feature extraction (intensive hierarchy)
    """
    def __init__(self):
        super().__init__()
        # Bottom-up pathway (increasing semantic)
        self.conv1 = ConvBlock(3, 64, stride=2)    # P1: 1/2
        self.conv2 = ConvBlock(64, 128, stride=2)  # P2: 1/4
        self.conv3 = ConvBlock(128, 256, stride=2) # P3: 1/8
        self.conv4 = ConvBlock(256, 512, stride=2) # P4: 1/16

        # Top-down pathway (lateral connections)
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        # Top-down with lateral connections (intensive hierarchy!)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2)

        return [p2, p3, p4]  # Multi-scale features
```

**Key intensive property**: Lateral connections
- Without laterals: Each scale independent (no configuration)
- With laterals: Scales interact (configuration emerges)

### Pattern 2C: ARR-COC Multi-Resolution Patches

**Architecture** (vision-language):
```python
class MultiResolutionPatches(nn.Module):
    """
    Different patch sizes for different image regions (intensive!)
    """
    def __init__(self):
        super().__init__()
        self.patch_embedders = nn.ModuleDict({
            'coarse': PatchEmbed(patch_size=32),   # Low-res, background
            'medium': PatchEmbed(patch_size=16),   # Mid-res, objects
            'fine': PatchEmbed(patch_size=8)       # High-res, details
        })

    def forward(self, image, relevance_map):
        """
        relevance_map: [H, W] indicating importance of each region
        """
        patches = []

        # Partition image into regions by relevance
        low_rel = (relevance_map < 0.3)   # Background → coarse
        med_rel = (relevance_map < 0.7) & ~low_rel  # Objects → medium
        high_rel = ~low_rel & ~med_rel    # Details → fine

        # Extract patches at appropriate resolution
        if low_rel.any():
            coarse_patches = self.patch_embedders['coarse'](image, low_rel)
            patches.append(coarse_patches)  # Few tokens

        if med_rel.any():
            medium_patches = self.patch_embedders['medium'](image, med_rel)
            patches.append(medium_patches)  # Moderate tokens

        if high_rel.any():
            fine_patches = self.patch_embedders['fine'](image, high_rel)
            patches.append(fine_patches)  # Many tokens

        return torch.cat(patches, dim=0)  # Variable-length output!
```

**Intensive property**: Token distribution
```
Background (low relevance): 32×32 patches → 4 tokens per 1024px²
Objects (medium relevance): 16×16 patches → 16 tokens per 1024px²
Details (high relevance): 8×8 patches → 64 tokens per 1024px²

Configuration: Allocate resolution by importance
```

### Design Decision Tree

```
Question 1: Does your task have natural hierarchy?
├─ Yes (vision: objects in scenes, language: words in sentences)
│  └─ Use multi-scale architecture (FPN, pyramid)
└─ No (flat data, e.g., tabular)
   └─ Skip hierarchy, use other principles

Question 2: Can you predict which scales matter for which inputs?
├─ Yes (task-dependent, e.g., query-aware)
│  └─ Use adaptive multi-resolution (ARR-COC style)
└─ No (always need all scales)
   └─ Use static FPN (still hierarchical, less intensive)

Question 3: What's your resolution range?
├─ Wide (1/32 to 1/1) → 4+ scales
├─ Moderate (1/8 to 1/1) → 2-3 scales
└─ Narrow (1/2 to 1/1) → Skip hierarchy, use dense
```

---

## Principle 3: Query-Aware Adaptation

### The Core Idea

**Bad**: Process all inputs the same way.
**Good**: Adapt processing based on query/task.

**Why it's intensive**: Query-content coupling is transjective (intensive property).

### Pattern 3A: Cross-Attention Routing

**Architecture**:
```python
class QueryAwareRouter(nn.Module):
    """
    Route visual patches based on query relevance (ARR-COC style)
    """
    def __init__(self, query_dim=512, patch_dim=512):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, 256)
        self.patch_proj = nn.Linear(patch_dim, 256)

    def forward(self, query, patches):
        """
        query: [batch, query_dim]
        patches: [batch, num_patches, patch_dim]

        Returns: token_budgets [batch, num_patches]
        """
        # Project to common space
        q = self.query_proj(query)  # [batch, 256]
        p = self.patch_proj(patches)  # [batch, num_patches, 256]

        # Cross-attention scores (intensive coupling!)
        scores = torch.einsum('bd,bnd->bn', q, p)
        scores = scores / math.sqrt(256)

        # Map scores to token budgets
        min_tokens, max_tokens = 64, 400
        budgets = min_tokens + (max_tokens - min_tokens) * torch.sigmoid(scores)

        return budgets.int()
```

**Intensive property**: Coupling quality
```
High coupling (score > 0.8): 350-400 tokens
Medium coupling (score 0.3-0.8): 150-300 tokens
Low coupling (score < 0.3): 64-100 tokens

Configuration: Allocate by query-patch relevance
```

### Pattern 3B: Task-Conditional Layers

**Architecture**:
```python
class TaskConditionalNet(nn.Module):
    """
    Different layer configurations for different tasks
    """
    def __init__(self, num_tasks=10, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(num_layers)
        ])

        # Task-specific scaling factors
        self.task_scales = nn.Parameter(
            torch.ones(num_tasks, num_layers)
        )

    def forward(self, x, task_id):
        """
        task_id: Which task we're solving (intensive configuration!)
        """
        scales = self.task_scales[task_id]  # [num_layers]

        for i, layer in enumerate(self.layers):
            # Scale layer contribution by task
            layer_out = layer(x)
            x = x + scales[i] * layer_out  # Task-aware residual!

        return x
```

**Training tip**: Learn task scales with multi-task training:
```python
def multitask_loss(model, batch, task_ids):
    """
    Train task_scales to configure model per task
    """
    losses = []
    for x, y, task_id in zip(batch, targets, task_ids):
        output = model(x, task_id)
        loss = F.cross_entropy(output, y)
        losses.append(loss)

    # Also regularize for sparse scaling (intensive property!)
    scale_l1 = model.task_scales.abs().mean()
    return sum(losses) / len(losses) + 0.01 * scale_l1
```

### Pattern 3C: Prompt-Based Modulation

**Architecture** (FiLM: Feature-wise Linear Modulation):
```python
class FiLMLayer(nn.Module):
    """
    Modulate features based on prompt/query (intensive!)
    """
    def __init__(self, feature_dim=256, prompt_dim=512):
        super().__init__()
        self.gamma_net = nn.Linear(prompt_dim, feature_dim)
        self.beta_net = nn.Linear(prompt_dim, feature_dim)

    def forward(self, features, prompt):
        """
        features: [batch, seq, feature_dim]
        prompt: [batch, prompt_dim]

        Returns: Modulated features
        """
        # Compute scaling and shifting from prompt
        gamma = self.gamma_net(prompt).unsqueeze(1)  # [batch, 1, feature_dim]
        beta = self.beta_net(prompt).unsqueeze(1)

        # Affine transformation (intensive configuration!)
        return gamma * features + beta
```

**Why it's intensive**: Same features, different modulation = configuration change.

### Design Decision Tree

```
Question 1: Do you have explicit query/task information?
├─ Yes (VQA, conditional generation)
│  └─ Use query-aware routing or FiLM
└─ No (standard classification)
   └─ Question 2: Can you infer task implicitly?
      ├─ Yes → Use task-conditional layers
      └─ No → Skip this principle

Question 2: How diverse are your tasks/queries?
├─ Very diverse (100+ distinct queries)
│  └─ Use cross-attention routing (high flexibility)
├─ Moderate (10-100 tasks)
│  └─ Use task-conditional layers (learnable scales)
└─ Low (2-10 tasks)
   └─ Use prompt-based modulation (simple, effective)

Question 3: Can you compute query-content coupling cheaply?
├─ Yes (shared embedding space)
│  └─ Use cross-attention (dot products are fast)
└─ No (heterogeneous spaces)
   └─ Use FiLM or task-conditional (avoid expensive coupling)
```

---

## Principle 4: Configuration Over Capacity

### The Core Idea

**Bad**: When performance plateaus, add more parameters.
**Good**: When performance plateaus, reorganize existing parameters.

**Why it's intensive**: Reorganization is configuration (intensive), adding params is capacity (extensive).

### Pattern 4A: Bottleneck Design

**Architecture** (MobileBERT-style):
```python
class BottleneckFFN(nn.Module):
    """
    Force information through narrow bottleneck (intensive compression!)
    """
    def __init__(self, input_dim=768, bottleneck_dim=128):
        super().__init__()
        self.compress = nn.Linear(input_dim, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        # Compress (force configuration!)
        compressed = self.compress(x)
        compressed = F.relu(compressed)

        # Expand back
        return self.expand(compressed)
```

**Intensive property**: Compression ratio
```
Standard FFN: 768 → 3072 → 768 (4× expansion)
Bottleneck FFN: 768 → 128 → 768 (6× compression)

Bottleneck: 13× fewer params, 95% performance
Configuration (bottleneck size) > Capacity (total params)
```

**When to use**:
- Model is overparameterized (high train acc, low test acc)
- Memory is constrained (mobile deployment)
- You want to force information bottleneck (distillation)

### Pattern 4B: Weight Pruning

**Architecture** (magnitude-based):
```python
def prune_model(model, sparsity=0.5):
    """
    Remove low-magnitude weights (intensive configuration!)
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Compute magnitude threshold
            threshold = torch.quantile(param.abs(), sparsity)

            # Create mask (configuration!)
            mask = (param.abs() > threshold).float()

            # Zero out small weights
            param.data *= mask

            # Store mask for retraining
            model.register_buffer(f"{name}_mask", mask)
```

**Pruning schedule**:
```python
def iterative_pruning(model, train_loader, epochs=100):
    """
    Gradually prune while retraining (configuration emerges!)
    """
    sparsity_schedule = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]

    for target_sparsity in sparsity_schedule:
        # Prune to target sparsity
        prune_model(model, sparsity=target_sparsity)

        # Retrain (configuration adapts!)
        for epoch in range(10):
            train_epoch(model, train_loader)

        # Evaluate
        acc = evaluate(model, val_loader)
        print(f"Sparsity {target_sparsity:.1f}: Acc {acc:.1f}%")
```

**Typical results**:
```
Sparsity 0.0 (dense): 85.0% accuracy
Sparsity 0.5 (50% pruned): 84.5% (-0.5%)
Sparsity 0.7 (70% pruned): 83.8% (-1.2%)
Sparsity 0.9 (90% pruned): 81.2% (-3.8%)

Intensive insight: 50% sparsity → 1% accuracy loss
Configuration (which weights matter) > Capacity (how many weights)
```

### Pattern 4C: Knowledge Distillation

**Architecture**:
```python
class DistillationTrainer:
    """
    Train small student to match large teacher's configuration
    """
    def __init__(self, teacher, student, temperature=2.0):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Match soft targets (configuration!) + hard labels (capacity)
        """
        # Soft targets (intensive - attention patterns)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        kl_loss *= self.temperature ** 2  # Scale back

        # Hard targets (extensive - raw accuracy)
        ce_loss = F.cross_entropy(student_logits, labels)

        # Combine (favor soft targets - configuration!)
        return 0.9 * kl_loss + 0.1 * ce_loss

    def train_step(self, batch):
        x, y = batch

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Student forward
        student_logits = self.student(x)

        # Distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, y)

        return loss
```

**Results** (from case study 3):
```
Teacher (BERT-Base): 110M params, 78.5% GLUE
Student (DistilBERT): 66M params, 76.1% GLUE

Capacity reduction: 40%
Performance reduction: 3%
Intensive property preserved: Attention patterns (87% similarity)
```

### Pattern 4D: Neural Architecture Search (NAS)

**Find optimal configuration**:
```python
class ConfigurationSearch:
    """
    Search for best architecture (intensive configuration!)
    """
    def __init__(self, search_space):
        self.search_space = search_space  # e.g., {layers: [6, 12, 24], ...}

    def sample_config(self):
        """
        Sample a configuration (not capacity!)
        """
        return {
            'num_layers': random.choice([6, 12, 18, 24]),
            'hidden_dim': random.choice([256, 512, 768]),
            'num_heads': random.choice([4, 8, 12]),
            'ffn_ratio': random.choice([2, 4, 6]),
            'dropout': random.choice([0.0, 0.1, 0.2])
        }

    def evaluate_config(self, config):
        """
        Train and evaluate a configuration
        """
        model = build_model(config)
        train_model(model, num_epochs=10)  # Quick training
        acc = evaluate(model, val_loader)

        # Compute intensive metric (P3)
        params = count_params(model) / 1e9
        p3 = acc / params

        return p3  # Optimize for intensive intelligence!

    def search(self, num_trials=100):
        """
        Random search for best configuration
        """
        best_config = None
        best_p3 = 0

        for _ in range(num_trials):
            config = self.sample_config()
            p3 = self.evaluate_config(config)

            if p3 > best_p3:
                best_p3 = p3
                best_config = config

        return best_config, best_p3
```

### Design Decision Tree

```
Question 1: Is your model overparameterized?
├─ Yes (train acc >> test acc)
│  └─ Use bottlenecks or pruning
└─ No (train acc ≈ test acc)
   └─ Model is well-configured, consider scaling up carefully

Question 2: Do you have a larger teacher model?
├─ Yes → Use distillation (preserve configuration)
└─ No → Train from scratch, then prune

Question 3: Can you afford architecture search?
├─ Yes (have compute budget)
│  └─ Run NAS, optimize for P3 or P2F
└─ No (limited compute)
   └─ Use proven architectures (DeepSeek MoE, DistilBERT patterns)

Question 4: What's your deployment target?
├─ Server (NVIDIA A100) → Moderate pruning (50-70%)
├─ Edge (Jetson) → Aggressive pruning (80-90%) + quantization
└─ Mobile (phone) → Distill + bottleneck + prune (MobileBERT style)
```

---

## Combining Principles: ARR-COC Design Walkthrough

**Goal**: Build vision-language model with intensive intelligence.

**Step 1: Apply Principle 1 (Sparse Activation)**
```python
# Decision: Use dynamic token allocation (not fixed 576 tokens)
# Reasoning: Inputs vary (simple vs complex images)
# Pattern: Conditional computation (allocate 64-400 tokens)
```

**Step 2: Apply Principle 2 (Hierarchical Organization)**
```python
# Decision: Multi-resolution patches
# Reasoning: Background vs objects need different resolutions
# Pattern: Coarse (32×32) for background, fine (8×8) for details
```

**Step 3: Apply Principle 3 (Query-Aware Adaptation)**
```python
# Decision: Three scorers (propositional, perspectival, participatory)
# Reasoning: Allocation should depend on query + content
# Pattern: Cross-attention routing (participatory scorer)
```

**Step 4: Apply Principle 4 (Configuration > Capacity)**
```python
# Decision: Use smaller vision encoder (50M vs 300M)
# Reasoning: Better to configure 50M well than 300M poorly
# Pattern: Bottleneck + efficient attention (not dense ViT)
```

**Result**:
```
ARR-COC:
  - Sparse activation: 64-400 tokens (vs 576 fixed)
  - Hierarchical: Multi-resolution patches
  - Query-aware: Three scorers determine allocation
  - Configuration > capacity: 50M params, 86% accuracy
  - P3 score: 1720 (6× better than baselines)

All four principles combined → intensive intelligence!
```

---

## Common Pitfalls to Avoid

### Pitfall 1: "Just add more layers"

**Wrong**:
```python
# Performance plateau at 12 layers, so...
model = TransformerModel(num_layers=48)  # 4× deeper!
```

**Right**:
```python
# Performance plateau → reorganize existing layers
model = TransformerModel(num_layers=12)
model = add_moe_layers(model, num_experts=64)  # Configure, don't expand!
```

### Pitfall 2: "Fixed allocation for all inputs"

**Wrong**:
```python
# All images get 576 tokens, always
tokens = vision_encoder(image)  # [batch, 576, dim]
```

**Right**:
```python
# Adaptive allocation based on image complexity
tokens = adaptive_vision_encoder(image, query)  # [batch, 64-400, dim]
```

### Pitfall 3: "Optimize for total params"

**Wrong**:
```python
# Target: Build 100B parameter model
model = ScaleUp(layers=128, hidden=8192, ...)  # Extensive thinking
```

**Right**:
```python
# Target: Maximize P3 score within compute budget
model = find_optimal_config(
    max_params=10B,  # Constraint
    metric='p3_score'  # Intensive optimization!
)
```

### Pitfall 4: "Dense activation is simpler"

**Wrong**:
```python
# Activate all experts for all tokens (simple!)
for expert in experts:
    output += expert(token)
```

**Right**:
```python
# Activate top-k experts (intensive intelligence!)
top_k_experts = route(token, k=8)
for expert in top_k_experts:
    output += expert(token)
```

---

## Takeaway Checklist

Before finalizing your architecture, ask:

**Sparse Activation**:
- [ ] Can I activate <20% of parameters per input?
- [ ] Do I have routing/gating mechanisms?
- [ ] Is my activation ratio an intensive property (not hardcoded)?

**Hierarchical Organization**:
- [ ] Do I process multiple scales (coarse → fine)?
- [ ] Are scales connected (lateral connections, FPN)?
- [ ] Is resolution allocation adaptive (not uniform)?

**Query-Aware Adaptation**:
- [ ] Does my model adapt to query/task?
- [ ] Do I measure query-content coupling?
- [ ] Is allocation based on relevance (not fixed)?

**Configuration > Capacity**:
- [ ] Did I try bottlenecks/pruning before adding params?
- [ ] Am I optimizing P3/P2F (not raw param count)?
- [ ] Can I distill from a larger teacher?

**Overall Intensive Intelligence**:
- [ ] Are my design decisions configuration-based?
- [ ] Am I measuring intensive metrics (P3, R3, CES)?
- [ ] Would doubling params double performance? (If yes, you're thinking extensively!)

---

## Sources

**Sparse Activation**:
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1) - MoE routing, top-k selection, load balancing
- [MoE Architecture Guide 2024](https://www.cerebras.ai/blog/moe-guide-why-moe) - Sparse activation patterns, routing mechanisms
- [Switch Transformer paper](https://arxiv.org/abs/2101.03961) - Top-1 routing, scaling to 1.6T params

**Hierarchical Organization**:
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) - Multi-scale features, lateral connections
- [ResNet paper](https://arxiv.org/abs/1512.03385) - Hierarchical depth, residual connections
- ARR-COC implementation - Multi-resolution patches (8×8, 16×16, 32×32)

**Query-Aware Adaptation**:
- [FiLM paper](https://arxiv.org/abs/1709.07871) - Feature-wise linear modulation
- [Cross-Attention in Transformers](https://arxiv.org/abs/1706.03762) - Query-key-value mechanism
- [06-intensive-intelligence-mathematical-foundations.md](06-intensive-intelligence-mathematical-foundations.md) - Information bottleneck, coupling quality

**Configuration > Capacity**:
- [MobileBERT paper](https://arxiv.org/abs/2004.02984) - Bottleneck architecture
- [DistilBERT paper](https://arxiv.org/abs/1910.01108) - Knowledge distillation
- [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) - Weight pruning, sparse networks

**Comprehensive Guides**:
- [08-intensive-intelligence-case-studies.md](08-intensive-intelligence-case-studies.md) - Real-world implementations
- [07-intensive-intelligence-measurement-frameworks.md](07-intensive-intelligence-measurement-frameworks.md) - P3, P2F, R3 metrics
- [Neural Architecture Design 2024](https://uplatz.com/blog/sparse-mixture-of-experts-moe-architecture-advancements-and-future-directions/) - MoE design principles

**Created**: 2025-01-31
**Oracle**: karpathy-neural-network-fundamentals
**Category**: Design principles, architectural patterns, practical guidelines
