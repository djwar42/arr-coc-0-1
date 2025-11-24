# Cascade Attention & Early Exit Mechanisms

## Overview

Cascade attention and early exit mechanisms enable **adaptive computation** in transformers - allowing models to dynamically adjust computational resources based on input difficulty. Instead of processing every token through all layers with full attention, these techniques create "exit ramps" where confident predictions can terminate early, and attention patterns can be progressively refined.

**Core principle**: Not all inputs require the same computational depth. Simple examples (e.g., "The movie was great") can exit after shallow layers, while complex examples (e.g., sarcastic or ambiguous text) proceed through deeper processing.

From [Efficient Adaptive Transformer (EAT)](https://arxiv.org/html/2510.12856v1) (accessed 2025-01-31):
> "EAT unifies three orthogonal efficiency levers—token pruning, sparse attention, and early exits—inside a standard encoder. The result is an input-adaptive classifier that preserves capacity for hard inputs and saves compute on easy ones."

**Key benefits**:
- 2-3x inference speedup on easy examples
- Maintains accuracy on difficult examples
- Reduces energy consumption and latency
- Enables deployment on resource-constrained devices

## Architecture Components

### 1. Multi-Stage Cascade Design

Cascade attention processes inputs through progressive stages, with each stage refining representations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CascadeTransformerLayer(nn.Module):
    """Single stage in cascade with optional early exit"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention with residual
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class CascadeTransformer(nn.Module):
    """Multi-stage cascade transformer with progressive refinement"""

    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 num_classes, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Create cascade stages
        self.layers = nn.ModuleList([
            CascadeTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Classifier heads at each stage (for early exit)
        self.classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None, key_padding_mask=None,
                return_all_logits=False):
        """
        Args:
            x: Input embeddings (seq_len, batch, d_model)
            return_all_logits: If True, return predictions from all stages

        Returns:
            If return_all_logits: List of logits from each stage
            Else: Final stage logits only
        """
        all_logits = []

        for layer_idx, (layer, classifier) in enumerate(
            zip(self.layers, self.classifiers)
        ):
            # Process through cascade stage
            x, _ = layer(x, attn_mask, key_padding_mask)

            # Get prediction at this stage (use CLS token or mean pooling)
            pooled = x[0]  # CLS token (assuming x[0] is CLS)
            logits = classifier(pooled)
            all_logits.append(logits)

        if return_all_logits:
            return all_logits
        return all_logits[-1]  # Final stage prediction
```

### 2. Early Exit with Confidence Thresholds

Early exit mechanisms allow samples to "exit" the network when predictions reach sufficient confidence:

```python
class EarlyExitTransformer(nn.Module):
    """Transformer with confidence-based early exiting"""

    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 num_classes, exit_threshold=0.9, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.exit_threshold = exit_threshold

        self.layers = nn.ModuleList([
            CascadeTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Exit classifiers (lightweight heads)
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None, key_padding_mask=None,
                enable_early_exit=True):
        """
        Forward pass with optional early exiting

        Returns:
            logits: Final predictions
            exit_layer: Layer where prediction was made (if early exit)
            confidence: Maximum confidence score
        """
        for layer_idx, (layer, classifier) in enumerate(
            zip(self.layers, self.exit_classifiers)
        ):
            # Process through layer
            x, _ = layer(x, attn_mask, key_padding_mask)

            # Check for early exit
            if enable_early_exit or layer_idx == self.num_layers - 1:
                pooled = x[0]  # CLS token
                logits = classifier(pooled)
                probs = F.softmax(logits, dim=-1)
                max_prob, _ = probs.max(dim=-1)

                # Exit if confident or at final layer
                if max_prob >= self.exit_threshold or \
                   layer_idx == self.num_layers - 1:
                    return logits, layer_idx, max_prob

        # Should never reach here
        return logits, self.num_layers - 1, max_prob


    def compute_exit_statistics(self, dataloader, device='cuda'):
        """Analyze exit layer distribution on a dataset"""
        self.eval()
        exit_counts = [0] * self.num_layers

        with torch.no_grad():
            for batch in dataloader:
                x = batch['input'].to(device)
                _, exit_layer, _ = self.forward(x, enable_early_exit=True)
                exit_counts[exit_layer] += 1

        total = sum(exit_counts)
        exit_percentages = [count / total * 100 for count in exit_counts]

        return {
            'exit_counts': exit_counts,
            'exit_percentages': exit_percentages,
            'avg_exit_layer': sum(i * p for i, p in
                                 enumerate(exit_percentages)) / 100
        }
```

### 3. Patience-Based Early Exiting

From [BERxiT paper](https://towardsdatascience.com/berxit-early-exiting-for-bert-6f76b2f561c5) (accessed 2025-01-31):
> "PABEE (Patient Early Stopping) looks at the output class if it were to exit. We exit if the intermediate outputs are the same over multiple layers."

```python
class PatienceEarlyExit(nn.Module):
    """Early exit requiring consistent predictions across layers"""

    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 num_classes, patience=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.patience = patience  # Require N consistent predictions

        self.layers = nn.ModuleList([
            CascadeTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.exit_classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None, key_padding_mask=None,
                enable_early_exit=True):
        """Forward with patience-based early exit"""

        prediction_history = []

        for layer_idx, (layer, classifier) in enumerate(
            zip(self.layers, self.exit_classifiers)
        ):
            # Process layer
            x, _ = layer(x, attn_mask, key_padding_mask)

            # Get prediction
            pooled = x[0]  # CLS token
            logits = classifier(pooled)
            pred_class = logits.argmax(dim=-1)
            prediction_history.append(pred_class)

            # Check for patience threshold
            if enable_early_exit and len(prediction_history) >= self.patience:
                # Check if last N predictions are identical
                recent_preds = prediction_history[-self.patience:]
                if all(p == recent_preds[0] for p in recent_preds):
                    return logits, layer_idx, None

            # Always return final layer
            if layer_idx == self.num_layers - 1:
                return logits, layer_idx, None
```

## Training Strategy: Alternating Fine-Tuning

From [BERxiT: Early Exiting for BERT](https://aclanthology.org/2021.eacl-main.8/) (accessed 2025-01-31):

Key challenge: Balance between optimizing early classifiers and maintaining final classifier performance.

```python
class AlternatingTrainer:
    """Alternating fine-tuning strategy for early exit models"""

    def __init__(self, model, optimizer, num_classes):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes

    def train_epoch(self, dataloader, device='cuda', epoch=0):
        """
        Alternating training:
        - Odd iterations: Train final classifier + backbone
        - Even iterations: Train early exit classifiers + backbone
        """
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            x = batch['input'].to(device)
            labels = batch['labels'].to(device)

            self.optimizer.zero_grad()

            # Get all predictions
            all_logits = self.model(x, return_all_logits=True)

            # Alternating objective
            if batch_idx % 2 == 0:
                # Even iteration: Train early exit heads
                loss = self._compute_early_exit_loss(all_logits[:-1], labels)
            else:
                # Odd iteration: Train final classifier
                loss = F.cross_entropy(all_logits[-1], labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _compute_early_exit_loss(self, early_logits, labels):
        """Weighted sum of losses from early exit heads"""
        total_loss = 0
        num_exits = len(early_logits)

        for i, logits in enumerate(early_logits):
            # Weight decreases for earlier exits
            weight = (i + 1) / num_exits
            loss = F.cross_entropy(logits, labels)
            total_loss += weight * loss

        return total_loss / num_exits


class JointTrainer:
    """Joint training strategy (simpler but suboptimal)"""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_epoch(self, dataloader, device='cuda'):
        """Train all classifiers jointly with equal weight"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            x = batch['input'].to(device)
            labels = batch['labels'].to(device)

            self.optimizer.zero_grad()

            # Get all predictions
            all_logits = self.model(x, return_all_logits=True)

            # Sum losses from all stages
            loss = sum(
                F.cross_entropy(logits, labels)
                for logits in all_logits
            ) / len(all_logits)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
```

## Auxiliary Loss Functions

From [Efficient Adaptive Transformer](https://arxiv.org/html/2510.12856v1) (accessed 2025-01-31):

```python
def compute_cascade_loss(all_logits, labels, layer_weights=None):
    """
    Multi-stage loss for cascade attention

    Args:
        all_logits: List of predictions from each stage
        labels: Ground truth labels
        layer_weights: Optional weights for each layer (default: [0.3, ..., 1.0])

    Returns:
        Weighted sum of cross-entropy losses
    """
    if layer_weights is None:
        # Default: Final layer has weight 1.0, others have 0.3
        layer_weights = [0.3] * (len(all_logits) - 1) + [1.0]

    total_loss = 0
    for logits, weight in zip(all_logits, layer_weights):
        loss = F.cross_entropy(logits, labels)
        total_loss += weight * loss

    return total_loss


def compute_confidence_calibration_loss(logits, labels, temperature=2.0):
    """
    Temperature-scaled loss for better confidence calibration

    Better calibrated confidences improve early exit decisions
    """
    # Scale logits by temperature
    scaled_logits = logits / temperature
    loss = F.cross_entropy(scaled_logits, labels)

    # Scale loss back
    return loss * (temperature ** 2)


def compute_distillation_loss(student_logits, teacher_logits,
                              temperature=2.0, alpha=0.5):
    """
    Knowledge distillation loss for training student exit heads

    Args:
        student_logits: Predictions from early exit head
        teacher_logits: Predictions from final (teacher) head
        temperature: Softening temperature
        alpha: Balance between distillation and task loss
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)

    distill_loss = F.kl_div(
        soft_predictions, soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)

    return distill_loss
```

## Inference with Dynamic Exit

```python
class DynamicInference:
    """Efficient inference with early exiting"""

    def __init__(self, model, confidence_threshold=0.9,
                 use_patience=False, patience=2):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.use_patience = use_patience
        self.patience = patience

    @torch.no_grad()
    def predict(self, x, return_stats=False):
        """
        Predict with early exit

        Returns:
            prediction: Final prediction
            exit_layer: Layer where exit occurred
            confidence: Confidence score
        """
        self.model.eval()

        logits, exit_layer, confidence = self.model(
            x,
            enable_early_exit=True
        )

        prediction = logits.argmax(dim=-1)

        if return_stats:
            return {
                'prediction': prediction,
                'exit_layer': exit_layer,
                'confidence': confidence,
                'layers_saved': self.model.num_layers - exit_layer - 1
            }

        return prediction

    def benchmark_efficiency(self, dataloader, device='cuda'):
        """Measure speedup from early exiting"""
        import time

        # Benchmark with early exit
        start = time.time()
        exit_layers = []

        for batch in dataloader:
            x = batch['input'].to(device)
            _, exit_layer, _ = self.model(x, enable_early_exit=True)
            exit_layers.append(exit_layer)

        time_with_exit = time.time() - start

        # Benchmark without early exit
        start = time.time()
        for batch in dataloader:
            x = batch['input'].to(device)
            _ = self.model(x, enable_early_exit=False)

        time_without_exit = time.time() - start

        speedup = time_without_exit / time_with_exit
        avg_exit_layer = sum(exit_layers) / len(exit_layers)

        return {
            'speedup': speedup,
            'avg_exit_layer': avg_exit_layer,
            'time_with_exit': time_with_exit,
            'time_without_exit': time_without_exit
        }
```

## Complete Example: Training & Evaluation

```python
def train_cascade_early_exit_model():
    """Complete training pipeline for cascade + early exit"""

    # Model configuration
    config = {
        'num_layers': 6,
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'num_classes': 2,  # Binary classification
        'exit_threshold': 0.9,
        'dropout': 0.1
    }

    # Initialize model
    model = EarlyExitTransformer(**config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Load data (example)
    train_loader = get_train_dataloader()  # Your data loading
    val_loader = get_val_dataloader()

    # Training with alternating strategy
    trainer = AlternatingTrainer(model, optimizer, config['num_classes'])

    best_val_acc = 0
    for epoch in range(3):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch=epoch)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        # Validate
        val_acc, exit_stats = evaluate_with_early_exit(
            model, val_loader, device='cuda'
        )
        print(f"Val Accuracy = {val_acc:.4f}")
        print(f"Avg Exit Layer = {exit_stats['avg_exit_layer']:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

    return model


@torch.no_grad()
def evaluate_with_early_exit(model, dataloader, device='cuda'):
    """Evaluate model with early exit statistics"""
    model.eval()

    correct = 0
    total = 0
    exit_layers = []

    for batch in dataloader:
        x = batch['input'].to(device)
        labels = batch['labels'].to(device)

        logits, exit_layer, _ = model(x, enable_early_exit=True)
        predictions = logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        exit_layers.append(exit_layer)

    accuracy = correct / total
    avg_exit = sum(exit_layers) / len(exit_layers)

    exit_stats = {
        'avg_exit_layer': avg_exit,
        'layers_saved_pct': (1 - avg_exit / model.num_layers) * 100
    }

    return accuracy, exit_stats
```

## Advanced: Learning to Exit (for Regression)

From [BERxiT paper](https://aclanthology.org/2021.eacl-main.8/) (accessed 2025-01-31):
> "Learning-To-Exit (LTE) component... takes as input a hidden state of some layer and outputs a confidence score for the prediction in this layer."

```python
class LearnedExitModule(nn.Module):
    """Learned confidence scorer for regression tasks"""

    def __init__(self, d_model):
        super().__init__()
        self.confidence_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )

    def forward(self, hidden_state):
        """Output confidence score for current representation"""
        return self.confidence_scorer(hidden_state)


class RegressionEarlyExit(nn.Module):
    """Early exit for regression tasks using LTE"""

    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 exit_threshold=0.9, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.exit_threshold = exit_threshold

        self.layers = nn.ModuleList([
            CascadeTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Regression heads
        self.regressors = nn.ModuleList([
            nn.Linear(d_model, 1)  # Single output value
            for _ in range(num_layers)
        ])

        # Shared LTE module
        self.lte = LearnedExitModule(d_model)

    def forward(self, x, enable_early_exit=True):
        """Forward pass with learned exit decisions"""

        for layer_idx, (layer, regressor) in enumerate(
            zip(self.layers, self.regressors)
        ):
            # Process layer
            x, _ = layer(x)

            # Get prediction and confidence
            pooled = x[0]  # CLS token
            prediction = regressor(pooled)
            confidence = self.lte(pooled)

            # Exit decision based on learned confidence
            if enable_early_exit and confidence >= self.exit_threshold:
                return prediction, layer_idx, confidence

            if layer_idx == self.num_layers - 1:
                return prediction, layer_idx, confidence


def train_lte_module(model, dataloader, optimizer, device='cuda'):
    """Train LTE module with ground truth confidence"""
    model.train()

    for batch in dataloader:
        x = batch['input'].to(device)
        y_true = batch['target'].to(device)

        optimizer.zero_grad()

        # Get predictions at all layers
        all_predictions = []
        all_confidences = []

        for layer in model.layers:
            x, _ = layer(x)
            pooled = x[0]
            pred = model.regressors[len(all_predictions)](pooled)
            conf = model.lte(pooled)

            all_predictions.append(pred)
            all_confidences.append(conf)

        # Compute LTE loss: confidence should correlate with error
        lte_loss = 0
        for pred, conf in zip(all_predictions[:-1], all_confidences[:-1]):
            error = torch.abs(pred - y_true)
            # Ground truth confidence: 1 - tanh(error)
            target_conf = 1 - torch.tanh(error)
            lte_loss += F.mse_loss(conf, target_conf)

        # Regression loss on final prediction
        pred_loss = F.mse_loss(all_predictions[-1], y_true)

        # Combined loss
        loss = pred_loss + 0.3 * lte_loss
        loss.backward()
        optimizer.step()
```

## Theoretical Analysis

From [Efficient Adaptive Transformer](https://arxiv.org/html/2510.12856v1) (accessed 2025-01-31):

**Computational complexity reduction:**

Dense transformer: O(L × T²) where L = layers, T = sequence length

With early exit:
- Average depth: p̄ × L (where p̄ < 1 is avg exit probability)
- Effective compute: O(p̄L × T²)

**Speedup factor:**
```
Speedup = 1 / p̄

If 50% of samples exit at layer 3 of 12:
p̄ = 0.5 × (3/12) + 0.5 × 1.0 = 0.625
Speedup ≈ 1.6x
```

**Trade-off analysis:**
- Lower threshold → More early exits → Faster but lower accuracy
- Higher threshold → Fewer early exits → Slower but higher accuracy

Optimal threshold depends on:
1. Input distribution difficulty
2. Calibration quality of confidence scores
3. Acceptable accuracy degradation

## Practical Considerations

### 1. Confidence Calibration

From research (accessed 2025-01-31):
> "Exit confidence must be calibrated for robust threshold sweeps. We estimate Expected Calibration Error (ECE) and apply temperature scaling if ECE > 2%."

```python
def compute_ece(predictions, confidences, labels, n_bins=15):
    """Expected Calibration Error"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calibrate_temperature(model, val_loader, device='cuda'):
    """Find optimal temperature for confidence calibration"""
    from scipy.optimize import minimize_scalar

    # Collect predictions and confidences
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch['input'].to(device)
            labels = batch['labels'].to(device)
            logits, _, _ = model(x, enable_early_exit=False)

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Find temperature that minimizes NLL
    def nll(temperature):
        scaled_logits = all_logits / temperature
        loss = F.cross_entropy(scaled_logits, all_labels)
        return loss.item()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    optimal_temp = result.x

    return optimal_temp
```

### 2. Threshold Selection

```python
def select_optimal_threshold(model, val_loader, target_accuracy=0.95,
                             device='cuda'):
    """
    Select threshold that maintains target accuracy

    Returns threshold and corresponding speedup
    """
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    results = []

    for threshold in thresholds:
        model.exit_threshold = threshold

        accuracy, exit_stats = evaluate_with_early_exit(
            model, val_loader, device
        )

        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'avg_exit_layer': exit_stats['avg_exit_layer'],
            'speedup': model.num_layers / (exit_stats['avg_exit_layer'] + 1)
        })

    # Find highest speedup that meets accuracy target
    valid_results = [r for r in results if r['accuracy'] >= target_accuracy]

    if not valid_results:
        return results[-1]  # Return most conservative

    return max(valid_results, key=lambda r: r['speedup'])
```

### 3. Monitoring in Production

```python
class ProductionMonitor:
    """Monitor early exit behavior in deployment"""

    def __init__(self):
        self.exit_counts = None
        self.confidence_scores = []
        self.predictions = []

    def log_inference(self, exit_layer, confidence, prediction):
        """Log each inference for monitoring"""
        if self.exit_counts is None:
            self.exit_counts = {}

        self.exit_counts[exit_layer] = self.exit_counts.get(exit_layer, 0) + 1
        self.confidence_scores.append(confidence)
        self.predictions.append(prediction)

    def get_statistics(self):
        """Get deployment statistics"""
        total = sum(self.exit_counts.values())

        return {
            'exit_distribution': {
                layer: count / total
                for layer, count in self.exit_counts.items()
            },
            'avg_confidence': sum(self.confidence_scores) / len(self.confidence_scores),
            'total_inferences': total
        }

    def detect_distribution_shift(self, baseline_exit_dist, threshold=0.1):
        """Detect if exit pattern has shifted significantly"""
        current_dist = self.get_statistics()['exit_distribution']

        # Compare distributions
        shift = sum(
            abs(current_dist.get(layer, 0) - baseline_exit_dist.get(layer, 0))
            for layer in set(current_dist.keys()) | set(baseline_exit_dist.keys())
        )

        if shift > threshold:
            return True, shift
        return False, shift
```

## Combining with Other Efficiency Techniques

Cascade attention and early exit can be combined with:

1. **Token pruning**: Remove unimportant tokens before attention
2. **Sparse attention**: Use local windows + global tokens
3. **Quantization**: Reduce precision for early exit heads

```python
class HybridEfficientTransformer(nn.Module):
    """Combines early exit + token pruning + sparse attention"""

    def __init__(self, num_layers, d_model, nhead, dim_feedforward,
                 num_classes, prune_ratio=0.3, window_size=32):
        super().__init__()
        # Implementation combining all techniques
        # See EAT paper for complete architecture
        pass
```

## Connection to ARR-COC-VIS

Early exit mechanisms align with Vervaeke's relevance realization:
- **Propositional knowing**: Statistical information content determines exit confidence
- **Procedural knowing**: Model learns which inputs need deeper processing
- **Dynamic allocation**: Similar to ARR-COC's adaptive token budgets (64-400)

Both approaches recognize: **Not all inputs merit equal computation.**

## Sources

**Research Papers:**
- [Efficient Adaptive Transformer (EAT)](https://arxiv.org/html/2510.12856v1) - arXiv:2510.12856 (accessed 2025-01-31)
- [BERxiT: Early Exiting for BERT](https://aclanthology.org/2021.eacl-main.8/) - Xin et al., EACL 2021 (accessed 2025-01-31)
- [DeeBERT: Dynamic Early Exiting](https://arxiv.org/abs/2004.12993) - Xin et al., ACL 2020
- [PABEE: BERT Loses Patience](https://arxiv.org/abs/2006.04152) - Zhou et al., 2020

**Web Resources:**
- [BERxiT Explained](https://towardsdatascience.com/berxit-early-exiting-for-bert-6f76b2f561c5) - Towards Data Science (accessed 2025-01-31)
- [Awesome Adaptive Computation](https://github.com/koayon/awesome-adaptive-computation) - Community resource (accessed 2025-01-31)

**Additional References:**
- [PonderNet](https://arxiv.org/abs/2107.05407) - DeepMind, probabilistic halting
- [Universal Transformer](https://arxiv.org/abs/1807.03819) - Recurrent transformer with ACT
- [FastBERT](https://arxiv.org/abs/2004.02178) - Self-distilling BERT with adaptive inference
