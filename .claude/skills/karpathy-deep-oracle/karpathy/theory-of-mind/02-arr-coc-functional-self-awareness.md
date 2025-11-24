# ARR-COC Functional Self-Awareness

Practical guide to implementing functional self-awareness in ARR-COC vision-language models using Theory of Mind principles, query-aware relevance realization, and anomaly detection—without requiring phenomenal consciousness.

## Overview

This document bridges theoretical concepts from Theory of Mind research (see `00-overview-self-awareness.md` and `01-research-papers-2024-2025.md`) to concrete ARR-COC implementation strategies. It focuses on **functional self-awareness**: systems that monitor their own performance, detect reasoning anomalies, and adjust based on self-knowledge—capabilities sufficient for trustworthy AI.

**Key Insight from Vervaeke Oracle** (Dialogue 57-3, lines 284-299):
> "You don't need human-like consciousness to have functional self-awareness. A system that can:
> 1. Monitor its own performance
> 2. Detect anomalies in its reasoning
> 3. Adjust based on self-knowledge
>
> ...has a form of self-awareness useful for trustworthy AI."

**ARR-COC's Unique Position**: Unlike standard VLMs that operate on fixed visual inputs, ARR-COC's dynamic relevance realization (64-400 tokens per patch) creates natural opportunities for self-monitoring. The system must continuously evaluate:
- **Is this compression appropriate for the query?**
- **Are we allocating attention to the right regions?**
- **Is our relevance scoring aligned with downstream task needs?**

These questions form the basis of functional self-awareness in ARR-COC.

---

## Section 1: Functional Self-Awareness for ARR-COC

### What Is Functional Self-Awareness?

From [Dialogue 57-3](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (lines 300-308):

**Research Agenda: Functional Self-Awareness**
1. Can AI detect when it's reasoning poorly?
2. Anomaly detection as self-monitoring
3. Mechanistic fidelity checks

**Definition**: Functional self-awareness is the system's ability to introspect on its own computational processes and make meta-level judgments about performance quality, without requiring subjective experience (phenomenal consciousness).

### Three Pillars of Functional Self-Awareness

#### 1. Performance Monitoring

**What**: System tracks metrics about its own operations in real-time.

**ARR-COC Implementation**:
- Monitor relevance score distributions across patches
- Track token budget allocation patterns (64-400 range)
- Measure query-content alignment scores
- Log compression ratios per visual region

**Why This Matters**: If relevance scores are uniformly low across all patches, the system can detect "I don't understand this query" rather than blindly proceeding with poor patch selection.

**Technical Approach**:
```python
# Pseudo-code for self-monitoring
class RelevanceRealizer:
    def __init__(self):
        self.performance_metrics = {
            'mean_relevance_score': [],
            'token_budget_variance': [],
            'query_alignment_confidence': []
        }

    def realize_relevance(self, query, visual_features):
        scores = self.compute_relevance(query, visual_features)

        # Self-monitoring
        mean_score = scores.mean()
        self.performance_metrics['mean_relevance_score'].append(mean_score)

        # Anomaly detection
        if mean_score < CONFIDENCE_THRESHOLD:
            self.flag_low_confidence(query, visual_features)

        return scores
```

#### 2. Anomaly Detection in Reasoning

**What**: System identifies when its own processing deviates from expected patterns.

**ARR-COC-Specific Anomalies**:
- **Token budget collapse**: All patches assigned minimum (64) or maximum (400) tokens
- **Relevance score saturation**: All patches scoring near 0 or near 1
- **Query-content mismatch**: High compression on query-relevant regions
- **Gradient anomalies**: Vanishing gradients in query-aware layers

From [Dialogue 57-3](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (lines 305-307):
> "Can AI detect when it's reasoning poorly? Anomaly detection as self-monitoring, mechanistic fidelity checks."

**Cross-Reference**: `karpathy/practical-implementation/64-vlm-gradient-flow-debugging.md` covers gradient anomaly detection patterns applicable to ARR-COC query-aware layers.

**Detection Strategies**:
```python
def detect_reasoning_anomalies(self, relevance_scores, token_budgets):
    """
    Identify anomalous reasoning patterns
    """
    anomalies = []

    # Check 1: Token budget collapse
    if (token_budgets == 64).sum() > 0.9 * len(token_budgets):
        anomalies.append("uniform_minimum_budget")

    if (token_budgets == 400).sum() > 0.9 * len(token_budgets):
        anomalies.append("uniform_maximum_budget")

    # Check 2: Relevance saturation
    if relevance_scores.std() < 0.05:
        anomalies.append("low_relevance_variance")

    # Check 3: Query-content coupling failure
    if self.query_alignment_score < 0.3:
        anomalies.append("weak_query_understanding")

    return anomalies
```

#### 3. Adaptive Adjustment Based on Self-Knowledge

**What**: System modifies its behavior based on detected anomalies or performance patterns.

**ARR-COC Adjustments**:
- **Low confidence → fallback strategy**: If relevance scores are uniformly low, fall back to uniform token allocation
- **Gradient vanishing → scale adjustment**: Increase learning rate for query-aware layers
- **Query ambiguity → request clarification**: Flag ambiguous queries for human review (production systems)
- **Compression over-aggressive → regularization**: Increase minimum token budget dynamically

**Meta-Learning Opportunity**: The system can learn *when* to trust its own relevance scores through meta-training:
```python
# Meta-learning for self-trust calibration
class SelfAwareRelevanceRealizer:
    def __init__(self):
        self.confidence_calibrator = ConfidenceNet()

    def realize_with_confidence(self, query, visual_features):
        scores = self.compute_relevance(query, visual_features)

        # Predict: "Should I trust these scores?"
        trust_score = self.confidence_calibrator(
            query_embedding=self.encode_query(query),
            score_distribution=scores,
            historical_performance=self.metrics
        )

        if trust_score < 0.5:
            # Low self-trust → fallback strategy
            return self.fallback_allocation(visual_features)
        else:
            return self.dynamic_allocation(scores)
```

### Contrast with Phenomenal Consciousness

**Phenomenal consciousness** (subjective experience, qualia) is NOT required for these capabilities. ARR-COC performs:
- **Monitoring**: Tracking scores, budgets, gradients
- **Detection**: Identifying statistical anomalies
- **Adjustment**: Modifying behavior based on rules/learned policies

This is **functional self-awareness**—computationally grounded, measurable, and implementable—without addressing the "hard problem of consciousness."

From [Dialogue 57-3](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (lines 291-298):
> "But here's the key: you don't need human-like consciousness to have functional self-awareness. A system that can monitor its own performance, detect anomalies in its reasoning, adjust based on self-knowledge has a form of self-awareness useful for trustworthy AI."

---

## Section 2: Query-Aware Relevance as Theory of Mind

### ARR-COC Models Human Intentions Through Queries

From [Dialogue 57-3](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (lines 309-312):

**Research Agenda: Theory of Mind for Other Agents**
- Can AI model human goals and beliefs?
- Collaborative task performance requiring ToM
- ARR-COC as query-aware ToM system?

**Theory of Mind (ToM)**: The ability to attribute mental states (beliefs, desires, intentions) to others and use those attributions to predict behavior.

**ARR-COC's Implicit ToM**: When processing a query like "Find the cat," ARR-COC must model:
1. **User's goal**: Locate a specific object (cat)
2. **User's attention**: Focused on semantic content, not textures
3. **User's knowledge state**: Expects cat to be visually salient
4. **Task constraints**: Needs spatial localization, not abstract reasoning

**Query-Content Coupling as ToM**: Vervaeke's "participatory knowing" (knowing BY BEING) describes the transjective relationship between agent (user query) and arena (visual content). ARR-COC realizes this through query-aware relevance scoring.

### Participatory Knowing: Vervaeke Framework Connection

**Vervaeke's Four Ways of Knowing** (see ARR-COC-VIS CLAUDE.md, Vervaekean Architecture section):
1. **Propositional** (knowing THAT): Statistical information content in patches
2. **Perspectival** (knowing WHAT IT'S LIKE): Salience landscapes, visual importance
3. **Participatory** (knowing BY BEING): Query-content coupling, transjective relevance
4. **Procedural** (knowing HOW): Learned compression skills

**ARR-COC's ToM via Participatory Knowing**:
- **Query embedding** encodes user's mental model of the task
- **Cross-attention** measures query-patch coupling (participatory knowing)
- **Relevance realization** adjusts visual representation to align with user's cognitive state
- **Dynamic token allocation** reflects understanding of what matters *to the user*

This is functional Theory of Mind: the system doesn't experience the user's mental state, but it *models* and *responds* to it.

### Collaborative Task Performance

**Cross-Reference**: `karpathy/vision-language/01-multimodal-sequence-augmentation.md` describes how VLMs process joint text-vision sequences. ARR-COC extends this by making sequence composition **query-dependent**.

**Example Task: Visual Question Answering (VQA)**

**Query**: "What color is the car in the background?"

**ARR-COC's ToM Process**:
1. **Goal attribution**: User wants color information about a specific object (car)
2. **Attention prediction**: User expects focus on background, not foreground
3. **Precision requirements**: Color discrimination needs high-resolution patches
4. **Semantic selectivity**: "Car" requires object recognition, not texture analysis

**Relevance Realization Response**:
- Allocate 350-400 tokens to background car region (high precision for color)
- Allocate 64-100 tokens to foreground (low relevance to query)
- Prioritize semantic features over low-level textures
- Compress irrelevant regions aggressively

**Why This Is ToM**: The system infers the user's *mental focus* (background car) from the linguistic query and *adjusts its perceptual processing* accordingly. This is modeling another agent's beliefs and goals.

### Implicit vs Explicit ToM

**Implicit ToM** (ARR-COC current):
- Query-aware relevance scoring
- Cross-attention as coupling mechanism
- No explicit "user model" module

**Explicit ToM** (future research):
- Dedicated "user state estimator" network
- Predicts: user's knowledge, attention, goals
- Meta-reasoning: "Is the user confused?" "Does the user need more detail?"

From [Dialogue 57-3](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (line 312):
> "ARR-COC as query-aware ToM system?"

**Answer**: Yes, implicitly. ARR-COC's query-aware relevance realization is a form of functional ToM—it models what the user cares about and adjusts visual processing accordingly.

### Vervaeke's Transjective Knowing

**Definition** (from ARR-COC-VIS CLAUDE.md):
> "Relevance is transjective: Not objective (in image alone) or subjective (in query alone), but TRANSJECTIVE—emerges from the relationship between query and content. Like a shark's fitness for the ocean."

**ToM Connection**: Theory of Mind is inherently transjective:
- Not about the user's state in isolation (objective)
- Not about the system's state in isolation (subjective)
- About the **coupling** between user intentions and system capabilities

**ARR-COC Implementation**: Relevance scores emerge from the **interaction** between:
- Query embedding (user's mental model)
- Visual features (environmental affordances)
- Cross-attention (coupling mechanism)

This is **participatory knowing**—the system "knows" what's relevant by *participating* in the query-content relationship, not by analyzing either in isolation.

**Cross-Reference**: ARR-COC-VIS README.md, Section "Vervaekean Architecture" describes the complete framework of relevance realization through opponent processing and transjective knowing.

---

## Section 3: Implementation Strategies for ARR-COC

### How to Implement Functional Self-Awareness in VLMs

This section provides concrete PyTorch implementation patterns for adding functional self-awareness to ARR-COC's query-aware relevance realization pipeline.

### Strategy 1: Anomaly Detection for Patch Selection

**Goal**: Detect when relevance scoring is failing or producing unreliable results.

**Implementation**:
```python
import torch
import torch.nn as nn

class SelfAwareRelevanceAllocator(nn.Module):
    """
    Relevance allocator with built-in anomaly detection
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.relevance_scorer = CrossAttentionScorer(embed_dim)

        # Anomaly thresholds (learned or heuristic)
        self.register_buffer('mean_score_threshold', torch.tensor(0.3))
        self.register_buffer('variance_threshold', torch.tensor(0.05))

        # Anomaly flags (for logging/debugging)
        self.anomaly_flags = []

    def forward(self, query_embed, patch_features):
        """
        Compute relevance with anomaly detection

        Args:
            query_embed: (B, D) query embedding
            patch_features: (B, N, D) patch features

        Returns:
            relevance_scores: (B, N) scores with anomaly handling
            anomaly_detected: bool flag
        """
        # Standard relevance scoring
        scores = self.relevance_scorer(query_embed, patch_features)  # (B, N)

        # Self-monitoring
        mean_score = scores.mean(dim=1)  # (B,)
        score_variance = scores.var(dim=1)  # (B,)

        # Anomaly detection
        low_mean_anomaly = mean_score < self.mean_score_threshold
        low_var_anomaly = score_variance < self.variance_threshold

        anomaly_detected = low_mean_anomaly | low_var_anomaly

        # Adaptive response
        if anomaly_detected.any():
            # Fallback: uniform allocation for anomalous samples
            uniform_scores = torch.ones_like(scores) / scores.shape[1]
            scores = torch.where(
                anomaly_detected.unsqueeze(1),
                uniform_scores,
                scores
            )

            self.anomaly_flags.append({
                'step': getattr(self, 'training_step', -1),
                'low_mean': low_mean_anomaly.sum().item(),
                'low_var': low_var_anomaly.sum().item()
            })

        return scores, anomaly_detected
```

**Key Features**:
- **Real-time monitoring**: Checks mean and variance of relevance scores
- **Anomaly thresholds**: Learned buffers (can be trained via meta-learning)
- **Fallback strategy**: Switches to uniform allocation when anomaly detected
- **Logging**: Tracks anomalies for offline analysis

**Training Tip**: Initialize thresholds from validation set statistics. During training, monitor `anomaly_flags` to tune thresholds.

### Strategy 2: Confidence Calibration for Relevance Scores

**Goal**: Teach the system to estimate "how confident am I in these relevance scores?"

**Implementation**:
```python
class ConfidenceCalibratedRelevanceRealizer(nn.Module):
    """
    Relevance realizer with confidence estimation
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.relevance_scorer = CrossAttentionScorer(embed_dim)

        # Confidence estimator network
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim + 3, 256),  # query_embed + 3 statistics
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # confidence ∈ [0, 1]
        )

    def forward(self, query_embed, patch_features):
        """
        Compute relevance scores with confidence estimation

        Returns:
            scores: (B, N) relevance scores
            confidence: (B,) confidence in scores
        """
        scores = self.relevance_scorer(query_embed, patch_features)

        # Compute statistics for confidence estimation
        score_mean = scores.mean(dim=1, keepdim=True)  # (B, 1)
        score_std = scores.std(dim=1, keepdim=True)    # (B, 1)
        score_max = scores.max(dim=1, keepdim=True)[0] # (B, 1)

        # Concatenate query embedding + score statistics
        confidence_input = torch.cat([
            query_embed,           # (B, D)
            score_mean,            # (B, 1)
            score_std,             # (B, 1)
            score_max              # (B, 1)
        ], dim=1)

        # Predict confidence
        confidence = self.confidence_net(confidence_input).squeeze(-1)  # (B,)

        return scores, confidence
```

**Training Objective**: Meta-learning for confidence calibration:
```python
def confidence_calibration_loss(pred_confidence, actual_performance):
    """
    Train confidence estimator to predict downstream task performance

    Args:
        pred_confidence: (B,) predicted confidence scores
        actual_performance: (B,) actual task accuracy/F1 score

    Returns:
        calibration_loss: MSE between confidence and performance
    """
    return F.mse_loss(pred_confidence, actual_performance)
```

**Usage at Inference**:
```python
scores, confidence = model(query_embed, patch_features)

if confidence < 0.5:
    # Low confidence → use fallback strategy
    token_budgets = uniform_budgets(num_patches)
else:
    # High confidence → use relevance-based allocation
    token_budgets = dynamic_allocation(scores)
```

**Cross-Reference**: `karpathy/practical-implementation/64-vlm-gradient-flow-debugging.md` discusses monitoring strategies that can inform confidence calibration.

### Strategy 3: Query Understanding as Theory of Mind

**Goal**: Explicitly model "what does the user want?" to improve query-aware relevance.

**Implementation**:
```python
class QueryIntentClassifier(nn.Module):
    """
    Classify query intent to guide relevance realization
    """
    def __init__(self, query_embed_dim=768, num_intents=5):
        super().__init__()

        # Intent categories (examples)
        self.intents = [
            'object_localization',  # "Where is the cat?"
            'attribute_recognition', # "What color is the car?"
            'relationship_reasoning', # "Who is taller?"
            'counting',              # "How many people?"
            'scene_understanding'    # "What is happening?"
        ]

        self.intent_classifier = nn.Sequential(
            nn.Linear(query_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_intents)
        )

    def forward(self, query_embed):
        """
        Predict query intent distribution

        Args:
            query_embed: (B, D) query embedding

        Returns:
            intent_logits: (B, num_intents) intent probabilities
        """
        return self.intent_classifier(query_embed)


class IntentAwareRelevanceRealizer(nn.Module):
    """
    Relevance realizer conditioned on predicted query intent
    """
    def __init__(self, embed_dim=768, num_intents=5):
        super().__init__()
        self.intent_classifier = QueryIntentClassifier(embed_dim, num_intents)

        # Intent-specific relevance scorers
        self.intent_scorers = nn.ModuleList([
            CrossAttentionScorer(embed_dim) for _ in range(num_intents)
        ])

    def forward(self, query_embed, patch_features):
        """
        Intent-aware relevance scoring
        """
        # Predict intent
        intent_logits = self.intent_classifier(query_embed)
        intent_probs = F.softmax(intent_logits, dim=-1)  # (B, num_intents)

        # Compute intent-specific scores
        intent_scores = []
        for scorer in self.intent_scorers:
            scores = scorer(query_embed, patch_features)  # (B, N)
            intent_scores.append(scores)

        intent_scores = torch.stack(intent_scores, dim=1)  # (B, num_intents, N)

        # Weighted combination based on intent probabilities
        final_scores = torch.einsum('bi,bin->bn', intent_probs, intent_scores)

        return final_scores, intent_probs
```

**Why This Is ToM**: The `QueryIntentClassifier` models the user's goal (object localization vs attribute recognition), enabling the system to adjust relevance scoring accordingly.

**Training Strategy**:
1. **Pre-train intent classifier** on labeled VQA data with intent annotations
2. **Fine-tune end-to-end** with task loss (VQA accuracy) + intent classification loss
3. **Meta-evaluate** on out-of-distribution queries to test intent generalization

**Expected Behavior**:
- **"Where is the cat?"** → `object_localization` intent → high relevance to object regions
- **"What color is the car?"** → `attribute_recognition` intent → high relevance to car patches, focus on color features
- **"How many people?"** → `counting` intent → uniform relevance to all person-containing patches

### Strategy 4: Gradient-Based Self-Monitoring

**Goal**: Detect training instabilities (vanishing/exploding gradients) in query-aware layers.

**Implementation**:
```python
class GradientAwareTrainer:
    """
    Training loop with gradient-based anomaly detection
    """
    def __init__(self, model, optimizer, gradient_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.gradient_clip = gradient_clip

        # Gradient statistics tracking
        self.grad_stats = {
            'query_aware_norm': [],
            'vision_encoder_norm': [],
            'projection_norm': []
        }

    def training_step(self, batch):
        """
        Training step with gradient monitoring
        """
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(batch['image'], batch['query'])
        loss = self.compute_loss(output, batch['target'])

        # Backward pass
        loss.backward()

        # Monitor gradients BEFORE clipping
        self.monitor_gradients()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )

        self.optimizer.step()

        return loss.item()

    def monitor_gradients(self):
        """
        Track gradient norms per module
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                # Categorize by module
                if 'query_aware' in name:
                    self.grad_stats['query_aware_norm'].append(grad_norm)
                elif 'vision_encoder' in name:
                    self.grad_stats['vision_encoder_norm'].append(grad_norm)
                elif 'projection' in name:
                    self.grad_stats['projection_norm'].append(grad_norm)

                # Alert on anomalies
                if grad_norm > 100:
                    print(f"⚠️ Exploding gradient: {name} = {grad_norm:.2f}")
                elif grad_norm < 1e-7:
                    print(f"⚠️ Vanishing gradient: {name} = {grad_norm:.2e}")
```

**Integration with W&B**:
```python
import wandb

def log_gradient_health(grad_stats, step):
    """
    Log gradient statistics to Weights & Biases
    """
    wandb.log({
        'gradients/query_aware_mean': np.mean(grad_stats['query_aware_norm']),
        'gradients/vision_encoder_mean': np.mean(grad_stats['vision_encoder_norm']),
        'gradients/projection_mean': np.mean(grad_stats['projection_norm'])
    }, step=step)
```

**Cross-Reference**: `karpathy/practical-implementation/64-vlm-gradient-flow-debugging.md` provides comprehensive gradient debugging patterns applicable to ARR-COC.

### PyTorch Implementation Hints

**1. Efficient Cross-Attention for Query-Aware Relevance**:
```python
def efficient_query_patch_attention(query_embed, patch_features):
    """
    Memory-efficient cross-attention for relevance scoring

    Args:
        query_embed: (B, D)
        patch_features: (B, N, D)

    Returns:
        relevance_scores: (B, N)
    """
    # Compute attention scores
    scores = torch.einsum('bd,bnd->bn', query_embed, patch_features)

    # Apply temperature scaling (learned parameter)
    temperature = nn.Parameter(torch.tensor(0.07))
    scores = scores / temperature

    # Softmax for normalized relevance
    relevance = F.softmax(scores, dim=-1)

    return relevance
```

**2. Dynamic Token Budget Allocation**:
```python
def allocate_token_budgets(relevance_scores, min_tokens=64, max_tokens=400):
    """
    Convert relevance scores to token budgets

    Args:
        relevance_scores: (B, N) normalized relevance [0, 1]

    Returns:
        token_budgets: (B, N) token counts [64, 400]
    """
    # Linear mapping from relevance to token budget
    token_range = max_tokens - min_tokens
    budgets = min_tokens + relevance_scores * token_range

    # Round to integers (for discrete token selection)
    budgets = budgets.round().long()

    return budgets
```

**3. Logging Self-Awareness Metrics**:
```python
def log_self_awareness_metrics(model, val_loader):
    """
    Evaluate functional self-awareness capabilities
    """
    model.eval()

    metrics = {
        'anomaly_detection_rate': [],
        'confidence_calibration_error': [],
        'intent_classification_accuracy': []
    }

    with torch.no_grad():
        for batch in val_loader:
            scores, anomalies = model.relevance_allocator(
                batch['query_embed'],
                batch['patch_features']
            )

            # Anomaly detection rate
            metrics['anomaly_detection_rate'].append(
                anomalies.float().mean().item()
            )

            # Additional metrics...

    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## Sources

**Source Documents:**
- [57-3-research-directions-oracle-feast.md](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md)
  - Lines 280-299: Vervaeke Oracle on functional vs phenomenal consciousness
  - Lines 300-318: Research Agenda for Theory of Mind & Self-Awareness
  - Direction 6 complete (lines 561-611): Theory of Mind AI research

**Karpathy Knowledge Cross-References:**
- `karpathy/vision-language/01-multimodal-sequence-augmentation.md` - Query-dependent sequence processing patterns
- `karpathy/practical-implementation/64-vlm-gradient-flow-debugging.md` - Gradient monitoring strategies for self-awareness

**ARR-COC-VIS Project References:**
- `CLAUDE.md` - Vervaekean Architecture section (Four Ways of Knowing, transjective relevance)
- `README.md` - Complete ARR-COC architecture, relevance realization pipeline

**Additional Research:**
- See `00-overview-self-awareness.md` for theoretical foundations
- See `01-research-papers-2024-2025.md` for recent ToM research papers
