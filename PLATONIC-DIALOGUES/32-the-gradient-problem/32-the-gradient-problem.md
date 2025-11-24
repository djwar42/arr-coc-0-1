# Part 32: The Gradient Problem - Learning to Realize Relevance
*Wherein the oracles confront the training challenge: how do you teach a system to allocate attention when position selection is discrete, filters are non-differentiable, and the VLM is frozen?*

---

## Opening: The Unasked Question

**KARPATHY:**
Alright. We've designed the system. 40-channel textures, Vervaekean framework, Qwen3-VL integration.

**THEAETETUS:**
The architecture is complete.

**KARPATHY:**
But we haven't asked the most important question.

**LOD ORACLE:**
Which is?

**KARPATHY:**
**How the hell do we train this thing?**

*Long pause*

**THEAETETUS:**
Ah. The gradient flow problem.

**KARPATHY:**
Exactly. We have all these learned parameters:
- Tension balances (compress/exploit/focus) - 3 parameters
- Allocation steepness - 1 parameter
- Temporal decay, motion boost - 2 parameters
- MLP combiner weights in TensionBalancer - ~20K parameters

But the signal comes from the END—VLM answer quality. How do gradients flow BACKWARDS through:
1. Qwen3-VL (frozen? fine-tuned?)
2. Feature extraction (depends on position selection)
3. Token allocation (discrete decision: select these 273 positions)
4. Relevance scoring (continuous)
5. Texture array generation (mostly non-differentiable filters)

**LOD ORACLE:**
This is not a simple backprop problem.

---

## Act I: The Gradient Flow Diagram

**KARPATHY:**
Let me draw out where gradients CAN flow and where they CAN'T.

```
╔═══════════════════════════════════════════════════════════
║ FORWARD PASS (what happens)
╠═══════════════════════════════════════════════════════════
║
║ Image → Texture Array [40, H, W]
║   ├─ RGB (channels 0-2) ✓ differentiable (pixels)
║   ├─ Position (3-5) ✓ differentiable (coordinates)
║   ├─ Edges (6-7) ✗ non-differentiable (Sobel filter)
║   ├─ Highpass/Lowpass (8-9) ✗ non-differentiable (FFT)
║   ├─ Motion (10) ✗ non-differentiable (optical flow)
║   ├─ Saliency (11) ✗ non-differentiable (OpenCV)
║   ├─ Distance (12) ✗ non-differentiable (scipy)
║   ├─ Clusters (13-16) ✗ non-differentiable (SLIC/SAM)
║   └─ CLIP embeddings (17-32) ✓ differentiable (neural net)
║
║ ↓ Sample 500 candidates (cluster-first cascade)
║
║ Texture → Features [500, 40]
║   ✓ differentiable (just indexing)
║
║ ↓ Three ways of knowing
║
║ Features → Scores [500, 3]
║   ├─ InformationScorer(features) ✓ differentiable
║   ├─ PerspectivalScorer(features) ✓ differentiable
║   └─ ParticipatoryScorer(features, query) ✓ differentiable
║
║ ↓ Opponent processing
║
║ Scores → Balanced [500]
║   TensionBalancer(info, persp, partic) ✓ differentiable
║
║ ↓ Token allocation
║
║ Balanced → Select top 273 positions ✗✗✗ NON-DIFFERENTIABLE
║   torch.topk(balanced, k=273) → discrete selection
║
║ ↓ Feature extraction
║
║ Positions → RGB patches → Qwen3-VL encode → Tokens
║   ✓ differentiable (if Qwen3-VL unfrozen)
║
║ ↓ VLM inference
║
║ Tokens + Query → Answer
║   ✓ differentiable
║
║ ↓ Loss
║
║ CrossEntropy(answer, ground_truth)
║   ✓ differentiable
║
╚═══════════════════════════════════════════════════════════

╔═══════════════════════════════════════════════════════════
║ BACKWARD PASS (where gradients die)
╠═══════════════════════════════════════════════════════════
║
║ ∂Loss/∂Answer ← ✓ flows normally
║ ∂Loss/∂Tokens ← ✓ flows if Qwen3-VL unfrozen
║ ∂Loss/∂Positions ← ✗✗✗ DEAD END (discrete selection)
║
║ The top-273 selection is DISCRETE:
║   balanced = [0.9, 0.7, 0.85, 0.3, ...]
║   selected = topk(balanced, k=273)
║            = [indices: 0, 2, 5, 7, ...]
║
║ Gradient of topk with respect to balanced? UNDEFINED.
║
║ Even if balanced[i] changes from 0.85 → 0.86,
║ the selected indices might not change at all.
║
║ No gradient = no learning.
║
╚═══════════════════════════════════════════════════════════
```

**THEAETETUS:**
So the position selection acts as a gradient barrier?

**KARPATHY:**
Exactly. Everything AFTER selection is differentiable (if Qwen3-VL is trainable). Everything BEFORE selection can't learn from the final loss.

**LOD ORACLE:**
This is a fundamental problem in discrete optimization. How do you backprop through a discrete decision?

---

## Act II: Three Approaches to the Gradient Problem

**KARPATHY:**
There are three ways to handle this. Let me walk through each.

### Approach 1: Gumbel-Softmax (Continuous Relaxation)

**THEAETETUS:**
Replace discrete selection with soft selection?

**KARPATHY:**
Exactly.

**Traditional (non-differentiable):**
```python
# Hard selection
balanced = balancer(info, persp, partic)  # [500]
selected_indices = torch.topk(balanced, k=273).indices
selected_positions = positions[selected_indices]  # [273, 2]

# Extract features at selected positions only
features = extract_at_positions(image, selected_positions)
```

**Gumbel-Softmax (differentiable):**
```python
# Soft selection with temperature
balanced = balancer(info, persp, partic)  # [500]

# Compute soft weights using Gumbel-Softmax
temperature = 0.5  # Lower = closer to discrete
weights = F.gumbel_softmax(balanced / temperature, hard=False)  # [500]

# Weighted combination of ALL positions (not just top-273)
features = torch.zeros(273, feature_dim)
for i in range(273):
    # Each output feature is weighted sum of all candidates
    features[i] = (weights.unsqueeze(-1) * all_features).sum(dim=0)
```

**LOD ORACLE:**
So instead of SELECTING 273 positions, you compute a WEIGHTED AVERAGE of all 500 candidates?

**KARPATHY:**
Right. And Gumbel-Softmax makes the weights ALMOST discrete (close to one-hot) but still differentiable.

**Pros:**
- Fully differentiable
- Gradients flow cleanly
- Standard technique (used in neural architecture search)

**Cons:**
- EXPENSIVE: Must encode features for all 500 candidates, not just 273
- Doesn't actually select discrete positions (soft weighting instead)
- At inference time, need to switch back to hard selection (train/test mismatch)

**THEAETETUS:**
So we compute features for 500 positions instead of 273? That defeats the efficiency purpose.

**KARPATHY:**
Yeah. You save compute on SELECTION (500 instead of 4096) but lose the savings from sparse sampling (273 vs 500).

Net speedup: 4096 → 500 = 8× (not 4096 → 273 = 15×)

### Approach 2: Straight-Through Estimator (Pretend It's Differentiable)

**KARPATHY:**
This is a hack, but it works surprisingly well.

**The idea:** During FORWARD pass, use discrete selection. During BACKWARD pass, PRETEND the gradient flows through.

```python
class TopKStraightThrough(torch.autograd.Function):
    """
    Forward: Discrete top-k selection
    Backward: Gradient flows as if selection was continuous
    """

    @staticmethod
    def forward(ctx, balanced_scores, k):
        # Hard selection (discrete)
        top_indices = torch.topk(balanced_scores, k=k).indices
        ctx.save_for_backward(balanced_scores, top_indices)
        return top_indices

    @staticmethod
    def backward(ctx, grad_output):
        balanced_scores, top_indices = ctx.saved_tensors

        # Pretend gradient: distribute grad_output to top-k scores
        grad_balanced = torch.zeros_like(balanced_scores)
        grad_balanced[top_indices] = grad_output

        return grad_balanced, None

# Usage
balanced = balancer(info, persp, partic)
selected_indices = TopKStraightThrough.apply(balanced, 273)
```

**LOD ORACLE:**
You're lying to the gradient?

**KARPATHY:**
lol yeah. We tell the backward pass: "Hey, those top-273 positions you selected? Pretend their scores directly determined the loss."

**Pros:**
- Simple to implement
- Maintains efficiency (only 273 positions encoded)
- Works in practice (used in binarized neural networks)

**Cons:**
- Mathematically dishonest (gradient is wrong)
- Can be unstable (gradient doesn't reflect actual selection dynamics)
- No guarantee of convergence

**THEAETETUS:**
But it works?

**KARPATHY:**
Often, yeah. Because the DIRECTION of the gradient is roughly correct: "increase scores of good positions, decrease scores of bad positions."

Even if the MAGNITUDE is wrong, SGD can compensate.

### Approach 3: Reinforcement Learning (Treat Selection as Action)

**KARPATHY:**
The nuclear option.

**Frame it as RL problem:**
- State: Texture array [40, H, W]
- Action: Select 273 positions
- Reward: VQA accuracy (or BLEU, or other metric)
- Policy: Your TensionBalancer + TokenAllocator

```python
class RelevanceRealizationPolicy(nn.Module):
    """
    Policy that selects positions to maximize VQA reward.
    """

    def __init__(self):
        self.balancer = TensionBalancer()
        self.allocator = TokenAllocator()

    def forward(self, texture, query):
        # Compute scores
        features = sample_candidates(texture)
        info, persp, partic = compute_scores(features, query)
        balanced = self.balancer(info, persp, partic)

        # Sample positions according to scores (stochastic policy)
        # Higher balanced score = higher probability of selection
        probs = F.softmax(balanced, dim=0)
        selected_indices = torch.multinomial(probs, num_samples=273, replacement=False)

        return selected_indices, probs

# Training with REINFORCE
policy = RelevanceRealizationPolicy()

for image, query, answer in dataset:
    texture = generate_texture(image)

    # Sample action (select positions)
    selected_indices, probs = policy(texture, query)
    positions = candidates[selected_indices]

    # Execute action (extract features, run VLM)
    features = extract_features(image, positions)
    prediction = vlm(features, query)

    # Compute reward
    reward = compute_reward(prediction, answer)  # 1.0 if correct, 0.0 if wrong

    # REINFORCE gradient
    log_probs = torch.log(probs[selected_indices])
    loss = -(log_probs.mean() * reward)  # Policy gradient

    loss.backward()
    optimizer.step()
```

**LOD ORACLE:**
You're training it like a game-playing agent?

**KARPATHY:**
Exactly. The "game" is: select 273 positions to maximize VQA accuracy.

**Pros:**
- Theoretically correct (RL is designed for discrete actions)
- No gradient hacks
- Can optimize non-differentiable rewards (accuracy, F1, BLEU)

**Cons:**
- HIGH VARIANCE (REINFORCE is notoriously unstable)
- SLOW (need many samples to estimate gradient)
- Complex (need baselines, variance reduction, etc.)

**THEAETETUS:**
Which approach do we choose?

---

## Act III: The Pragmatic Hybrid

**KARPATHY:**
None of them. We use a FOURTH approach.

**LOD ORACLE:**
Which is?

**KARPATHY:**
**Don't backprop through position selection at all.**

**The insight:** Your system has TWO types of parameters:

**Type 1: Scoring parameters (CONTINUOUS influence)**
- TensionBalancer MLP weights
- Tension balance parameters (compress/exploit/focus)
- Allocation steepness

These determine SCORES, which are continuous. We CAN backprop through these.

**Type 2: Selection parameters (DISCRETE influence)**
- Number of positions (273)
- Cluster filtering threshold
- Top-k selection

These determine which positions are chosen. We CAN'T backprop through these.

**Solution:** Only train Type 1. Fix Type 2 as hyperparameters.

```python
# Training loop
for image, query, answer in dataset:
    texture = generate_texture(image)

    # Score candidates (differentiable)
    features = sample_candidates(texture)  # Fixed 500 candidates
    info = info_scorer(features)  # ← learnable
    persp = persp_scorer(features)  # ← learnable
    partic = partic_scorer(features, query)  # ← learnable
    balanced = balancer(info, persp, partic)  # ← learnable

    # Select top-273 (non-differentiable, but that's OK)
    selected_indices = torch.topk(balanced, k=273).indices.detach()

    # Extract features (differentiable)
    positions = candidates[selected_indices]
    features = extract_features(image, positions)

    # VLM inference (differentiable if VLM unfrozen)
    prediction = vlm(features, query)

    # Loss (differentiable)
    loss = cross_entropy(prediction, answer)

    # Backward (gradients flow through scoring, not selection)
    loss.backward()
    optimizer.step()
```

**THEAETETUS:**
But gradients don't flow through the selection step. How does the system learn which positions to select?

**KARPATHY:**
**It learns to SCORE positions correctly.**

If position A should be selected but isn't, the gradient increases its score.
If position B shouldn't be selected but is, the gradient decreases its score.

Eventually, top-273 scores correspond to the 273 most relevant positions.

**LOD ORACLE:**
So the discrete selection is just a CONSEQUENCE of learned continuous scores?

**KARPATHY:**
Exactly. You're not learning "select these exact positions." You're learning "score relevance correctly, and top-k will pick the right positions."

**Analogy:**
```
You don't train a classifier to output "class 3"
You train it to output scores [0.1, 0.05, 0.8, 0.05]
And argmax gives you class 3 as a side effect
```

Same here:
```
You don't train to select positions [42, 108, 215, ...]
You train to output scores [0.9, 0.3, 0.85, ...]
And top-273 gives you those positions as a side effect
```

---

## Act IV: The Frozen VLM Problem

**THEAETETUS:**
But we haven't addressed the elephant in the room. Is Qwen3-VL frozen or trainable?

**KARPATHY:**
Good question. Three options:

**Option A: Fully frozen Qwen3-VL**
```python
qwen3vl = load_pretrained_qwen3vl()
qwen3vl.eval()  # Freeze all parameters

# Only train ARR-COC components
optimizer = Adam([
    balancer.parameters(),
    info_scorer.parameters(),
    persp_scorer.parameters(),
    partic_scorer.parameters(),
])
```

**Pros:**
- Fast (no VLM backward pass)
- Stable (pretrained VLM is proven)
- Low memory (don't store VLM activations)

**Cons:**
- No gradient signal from VLM output
- Can't adapt VLM to your token format
- Limited learning (only scoring parameters update)

**LOD ORACLE:**
How do you get gradient signal if VLM is frozen?

**KARPATHY:**
**Proxy losses.** Instead of end-to-end loss, use intermediate losses:

```python
# Ground truth: bounding boxes of relevant objects
gt_boxes = get_ground_truth_boxes(image, query)

# Your selection
selected_positions = allocate_tokens(texture, query)

# Loss: How much do selected positions overlap with ground truth?
overlap = compute_iou(selected_positions, gt_boxes)
loss = 1.0 - overlap  # Higher overlap = lower loss

# This loss doesn't require VLM at all!
loss.backward()
```

**THEAETETUS:**
So you supervise the SELECTION directly, not the final answer?

**KARPATHY:**
Right. If you have bounding box annotations, you can train "select positions that overlap with relevant objects."

But this requires expensive annotations.

**Option B: Adapter layers (LoRA)**
```python
# Freeze most of Qwen3-VL, add small trainable adapters
qwen3vl = load_pretrained_qwen3vl()

# Add LoRA adapters to attention layers
for layer in qwen3vl.layers:
    layer.attention = add_lora_adapter(layer.attention, rank=16)

# Train: ARR-COC components + LoRA adapters
optimizer = Adam([
    balancer.parameters(),
    info_scorer.parameters(),
    persp_scorer.parameters(),
    partic_scorer.parameters(),
    qwen3vl.lora_parameters(),  # Only ~1% of VLM params
])
```

**Pros:**
- End-to-end gradient flow
- VLM can adapt to your tokens
- Low memory (LoRA is efficient)

**Cons:**
- More complex (need to implement LoRA)
- Slower than frozen (still some VLM backprop)
- Risk of overfitting adapters

**Option C: Full fine-tuning**
```python
# Train EVERYTHING
optimizer = Adam([
    balancer.parameters(),
    info_scorer.parameters(),
    persp_scorer.parameters(),
    partic_scorer.parameters(),
    qwen3vl.parameters(),  # ALL parameters
])
```

**Pros:**
- Maximum flexibility
- VLM fully adapts to your system

**Cons:**
- EXPENSIVE (4B+ parameters, 80GB VRAM)
- SLOW (hours per epoch)
- Risk of catastrophic forgetting (VLM forgets pretraining)

**KARPATHY:**
**My recommendation: Start with Option A (frozen), add Option B (LoRA) if needed.**

Frozen VLM + proxy losses lets you validate your allocation strategy without expensive fine-tuning.

If proxy loss optimization works (positions overlap with relevant objects), THEN add LoRA to improve end-to-end.

---

## Act V: The Curriculum Strategy

**LOD ORACLE:**
Assuming we go with frozen VLM + proxy losses, what's the training curriculum?

**KARPATHY:**
Three stages, like we outlined in Part 29, but with explicit loss functions:

**Stage 1: Position Supervision (Proxy Loss)**
```python
# Dataset: Images with bounding box annotations
# Example: COCO (80 object categories, boxes provided)

for image, query, gt_boxes in annotated_dataset:
    texture = generate_texture(image)
    positions = allocate_tokens(texture, query)

    # Proxy loss: IoU with ground truth boxes
    loss = 1.0 - compute_iou(positions, gt_boxes)
    loss.backward()
    optimizer.step()

# Goal: Learn to allocate tokens to relevant objects
# 10 epochs, ~100K images
```

**Stage 2: VQA End-to-End (If using LoRA)**
```python
# Dataset: VQA with answer annotations
# Example: VQAv2 (200K questions)

for image, query, answer in vqa_dataset:
    texture = generate_texture(image)
    positions = allocate_tokens(texture, query)
    features = extract_features(image, positions)

    # End-to-end loss through VLM
    prediction = qwen3vl(features, query)  # LoRA adapters active
    loss = cross_entropy(prediction, answer)
    loss.backward()
    optimizer.step()

# Goal: Adapt allocation + VLM together
# 5 epochs, ~200K images
```

**Stage 3: Adversarial Hardening**
```python
# Dataset: Curated hard examples
# - Small text (formulas, captions)
# - Low contrast objects
# - Multiple distractors

for image, query, answer in hard_examples:
    texture = generate_texture(image)
    positions = allocate_tokens(texture, query)
    features = extract_features(image, positions)

    # High loss weight for hard examples
    prediction = qwen3vl(features, query)
    loss = 2.0 * cross_entropy(prediction, answer)
    loss.backward()
    optimizer.step()

# Goal: Handle edge cases
# 3 epochs, ~10K hard examples
```

**THEAETETUS:**
So Stage 1 doesn't require VLM at all? Just bounding box annotations?

**KARPATHY:**
Right. And bounding boxes are cheaper to annotate than VQA answers.

You can even use pseudo-labels:
- Run object detector (YOLO, etc.) to get boxes automatically
- Use those as proxy ground truth
- Noisy, but cheap to generate at scale

---

## Act VI: What Actually Needs to Learn?

**LOD ORACLE:**
Let's be concrete. What parameters are we actually training?

**KARPATHY:**
Let me list them:

**TensionBalancer (~20K parameters):**
```python
self.compress_vs_particularize = Parameter(0.5)  # 1 param
self.exploit_vs_explore = Parameter(0.5)         # 1 param
self.focus_vs_diversify = Parameter(0.5)         # 1 param

self.combiner = Sequential(
    Linear(3, 128),    # 3×128 = 384 params
    ReLU(),
    Linear(128, 128),  # 128×128 = 16,384 params
    ReLU(),
    Linear(128, 1)     # 128×1 = 128 params
)
# Total: ~17K params
```

**TokenAllocator (~10 parameters):**
```python
self.allocation_steepness = Parameter(2.0)  # 1 param
self.allocation_offset = Parameter(0.0)     # 1 param
# (Plus hyperparameters: min_tokens, max_tokens, total_budget)
```

**Scorers (depends on implementation):**
```python
# If using simple feature combiners:
class InformationScorer:
    def __init__(self):
        self.edge_weight = Parameter(0.4)
        self.highpass_weight = Parameter(0.3)
        self.distance_weight = Parameter(0.3)
    # Total: 3 params

# If using MLPs:
class InformationScorer(nn.Module):
    def __init__(self):
        self.mlp = Sequential(
            Linear(40, 64),   # 40 texture channels → 64 hidden
            ReLU(),
            Linear(64, 1)     # 64 → 1 score
        )
    # Total: ~2.5K params
```

**THEAETETUS:**
So we're talking about 20-50K trainable parameters total?

**KARPATHY:**
If scorers are simple, yes. If MLPs, maybe 20-100K.

Compare to Qwen3-VL: 4 BILLION parameters.

**LOD ORACLE:**
So we're training 0.0025% of the total system?

**KARPATHY:**
Yep. And that's GOOD. Means:
- Fast training (only optimize 50K params)
- Low risk of overfitting
- Frozen VLM retains pretrained knowledge

**Your contribution is the ATTENTION MECHANISM, not the encoding/generation.**

---

## Act VII: The Unexpected Insight

**THEAETETUS:**
We began this dialogue worried about gradient flow. But the solution is... don't try to backprop through everything?

**KARPATHY:**
Exactly. **Embrace the non-differentiability.**

Modern ML has this obsession with end-to-end differentiability. "Everything must be backpropable!"

But sometimes, discrete decisions are FINE. You just need to:
1. Learn continuous scores (differentiable)
2. Make discrete selections based on scores (non-differentiable, but that's OK)
3. Extract features from selections (differentiable again)

The discrete step is just a PROJECTION from continuous space to action space.

**LOD ORACLE:**
And we've seen this pattern before.

**KARPATHY:**
Where?

**LOD ORACLE:**
**Classification with softmax.**

```python
# Learn continuous logits (differentiable)
logits = model(input)  # [batch, num_classes]

# Project to discrete prediction (non-differentiable)
predicted_class = torch.argmax(logits, dim=-1)

# But we don't backprop through argmax!
# We backprop through logits instead.
```

Same thing here. We don't backprop through top-k. We backprop through the scores that FEED INTO top-k.

**KARPATHY:**
Huh. I never thought about it that way. Classification has always had this gradient barrier at argmax, but we just... ignore it.

**THEAETETUS:**
Because the loss is computed on LOGITS (before argmax), not on predicted class (after argmax).

```python
# Standard classification loss
loss = cross_entropy(logits, ground_truth)  # logits are continuous

# NOT this:
predicted_class = argmax(logits)  # discrete
loss = (predicted_class != ground_truth)  # can't backprop
```

**KARPATHY:**
So the trick for ARR-COC is the same: **compute proxy losses on continuous scores, not on discrete selections.**

```python
# Good: Loss on scores (continuous)
balanced_scores = balancer(info, persp, partic)
gt_relevance = compute_ground_truth_relevance(positions, query)
loss = mse_loss(balanced_scores, gt_relevance)

# Bad: Loss on selected positions (discrete)
selected_positions = topk(balanced_scores, k=273)
loss = position_match_loss(selected_positions, gt_positions)  # can't backprop
```

**LOD ORACLE:**
And if you have ground truth relevance scores (from human annotations or object detectors), you can train directly without ever selecting positions during training?

**KARPATHY:**
Exactly! That's even better.

```python
# Training: Never do discrete selection
for image, query, gt_relevance_map in dataset:
    texture = generate_texture(image)
    predicted_relevance = score_all_positions(texture, query)

    # Loss on continuous relevance map
    loss = mse_loss(predicted_relevance, gt_relevance_map)
    loss.backward()

# Inference: Use discrete selection
predicted_relevance = score_all_positions(texture, query)
selected_positions = topk(predicted_relevance, k=273)  # Now discrete is OK
```

**THEAETETUS:**
Train on continuous task (predict relevance), deploy with discrete task (select positions).

**KARPATHY:**
And this is mathematically sound. If your relevance predictions are accurate (continuous task), then top-k selection will pick the right positions (discrete task).

---

## Closing: The Path Forward

**SOCRATES:**
*Who has been listening*

We feared the gradient problem would derail the system. But we've discovered: the problem is not as severe as imagined.

**KARPATHY:**
Three key insights:

1. **Don't backprop through discrete selection.** Learn continuous scores instead.

2. **Use proxy losses.** Supervise relevance prediction, not position selection.

3. **Keep VLM frozen initially.** Add LoRA only if proxy loss optimization succeeds.

**LOD ORACLE:**
The training strategy:

**Stage 1 (Frozen VLM + Proxy Loss):**
- Dataset: COCO with bounding boxes (or pseudo-labels from object detector)
- Loss: IoU between selected positions and ground truth boxes
- Parameters: Only ARR-COC components (~50K params)
- Duration: 10 epochs, ~2 days on single GPU

**Stage 2 (LoRA Fine-tuning):**
- Dataset: VQAv2 with question-answer pairs
- Loss: Cross-entropy on VQA answers
- Parameters: ARR-COC + LoRA adapters (~100K params)
- Duration: 5 epochs, ~1 week on 8×A100

**Stage 3 (Adversarial Hardening):**
- Dataset: Curated hard examples (small text, low contrast)
- Loss: Weighted cross-entropy (2× weight for hard examples)
- Parameters: All components
- Duration: 3 epochs, ~2 days

**THEAETETUS:**
Total training time?

**KARPATHY:**
~10 days if you have 8×A100s. Maybe 3-4 weeks on weaker hardware.

**But Stage 1 can run on single GPU in 2 days.** So you can validate the approach quickly before scaling up.

**SOCRATES:**
And the learned system—what has it discovered?

**KARPATHY:**
After training, your TensionBalancer parameters might look like:

```python
compress_vs_particularize: 0.65  # Learned: slight bias toward compression
exploit_vs_explore: 0.42          # Learned: bias toward exploration
focus_vs_diversify: 0.71          # Learned: strong bias toward focus

allocation_steepness: 3.2         # Learned: steep curve (high relevance → many tokens)
```

**These aren't hand-coded.** They're DISCOVERED from data.

The model learned:
- Query-specific regions deserve detail (particularize when query demands it)
- Explore unknown regions (don't just exploit salient areas)
- Focus attention (concentrated allocation, not uniform spread)

**LOD ORACLE:**
Relevance realized through learning.

**KARPATHY:**
Not just implemented—**emerged.**

---

**END OF PART 32**

∿◇∿

---

## Appendix: Training Recipes

### Recipe 1: Minimal Training (Frozen VLM)

```python
# Pseudo-code for simplest training loop

model = ARR_COC_VIS(qwen3vl_frozen)
optimizer = Adam(model.arr_coc_parameters(), lr=1e-4)

for epoch in range(10):
    for image, query, gt_boxes in coco_dataset:
        # Forward
        texture = generate_texture(image)
        positions = model.allocate(texture, query)

        # Proxy loss: IoU with ground truth
        iou = compute_iou(positions, gt_boxes)
        loss = 1.0 - iou.mean()

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Result: Model learns to allocate tokens to relevant objects
# No VLM fine-tuning required
```

### Recipe 2: End-to-End Training (LoRA)

```python
# Pseudo-code for end-to-end with LoRA adapters

model = ARR_COC_VIS(qwen3vl_with_lora)
optimizer = Adam([
    {'params': model.arr_coc_parameters(), 'lr': 1e-4},
    {'params': model.qwen3vl.lora_parameters(), 'lr': 1e-5}
])

for epoch in range(5):
    for image, query, answer in vqav2_dataset:
        # Forward
        texture = generate_texture(image)
        positions = model.allocate(texture, query)
        features = extract_features(image, positions)
        prediction = model.qwen3vl(features, query)

        # End-to-end loss
        loss = cross_entropy(prediction, answer)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Result: Model learns allocation + VLM adapts to your tokens
```

### Recipe 3: Relevance Map Supervision

```python
# Pseudo-code for continuous supervision

model = ARR_COC_VIS(qwen3vl_frozen)
optimizer = Adam(model.arr_coc_parameters(), lr=1e-4)

for epoch in range(10):
    for image, query, gt_relevance_map in annotated_dataset:
        # Forward: Predict relevance for all positions
        texture = generate_texture(image)
        candidates = sample_candidates(texture)  # [500, 2]

        predicted_relevance = model.score_positions(
            texture, query, candidates
        )  # [500]

        # Ground truth relevance at candidate positions
        gt_relevance = gt_relevance_map[candidates[:, 0], candidates[:, 1]]

        # Continuous loss (no discrete selection during training!)
        loss = mse_loss(predicted_relevance, gt_relevance)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Inference: Use discrete selection
predicted_relevance = model.score_positions(texture, query, candidates)
selected = torch.topk(predicted_relevance, k=273).indices
```

---

**KEY INSIGHT:** The gradient problem is solved by NOT trying to differentiate through discrete selection. Learn continuous relevance scores, make discrete selections as a post-processing step.

**PARTICIPANTS:**
- Socrates (philosophical insight)
- Theaetetus (technical questions)
- Karpathy Oracle (gradient flow analysis)
- LOD Oracle (training strategy)

**NEXT DIALOGUE:** Part 33 - First Prototype (building the minimal viable integration)
