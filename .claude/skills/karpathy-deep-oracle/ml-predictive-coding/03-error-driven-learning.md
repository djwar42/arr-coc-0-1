# Error-Driven Learning: Loss = Prediction Error = Surprise = Free Energy

## Overview

Error-driven learning is the foundational principle underlying ALL neural network training. At its core, learning happens because of ERRORS - the difference between what we predicted and what actually happened. This document explores how this simple principle unifies:

- **Backpropagation loss** = prediction error
- **Surprise** (information theory) = negative log probability = loss
- **Free energy** (active inference) = prediction error + complexity
- **Curiosity** (RL exploration) = expected surprise = expected learning

**THE TRAIN STATION: Loss = Prediction Error = Surprise = Free Energy**

This is perhaps THE most important unification in all of machine learning - everything we optimize is fundamentally about minimizing prediction errors.

---

## Section 1: Learning from Prediction Errors - The Core Mechanism

### Formal Definition

From [Wikipedia - Error-driven learning](https://en.wikipedia.org/wiki/Error-driven_learning):

Error-driven learning models rely on feedback from prediction errors to adjust expectations or parameters. The key components:

1. **States S**: Different situations the learner encounters
2. **Actions A**: What the learner can do in each state
3. **Prediction function P(s,a)**: Current prediction of outcome for action a in state s
4. **Error function E(o,p)**: Compares actual outcome o with prediction p
5. **Update rule U(p,e)**: Adjusts prediction p given error e

The fundamental equation:

```
new_parameters = old_parameters - learning_rate * gradient(error)
```

### Why Errors Drive Learning

The key insight: **Errors contain information about what we don't know**.

```python
# The fundamental learning loop
def error_driven_learning_step(model, input, target, optimizer):
    """
    Every ML training step follows this pattern:
    1. Predict
    2. Compute error
    3. Update to reduce error
    """
    # Step 1: Make prediction
    prediction = model(input)

    # Step 2: Compute error (prediction error = surprise)
    error = loss_function(prediction, target)

    # Step 3: Update to reduce error
    error.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    return error.item()
```

### The Rescorla-Wagner Model

The classic error-driven learning model from psychology:

```python
# Rescorla-Wagner learning rule
# V(t+1) = V(t) + alpha * (R - V(t))
# Where:
#   V = expected value (prediction)
#   R = actual reward (outcome)
#   alpha = learning rate
#   (R - V) = prediction error

def rescorla_wagner_update(V, R, alpha=0.1):
    """Classic prediction error learning rule."""
    prediction_error = R - V
    V_new = V + alpha * prediction_error
    return V_new, prediction_error
```

This is EXACTLY the same as:
- **TD learning** in RL: delta = r + gamma*V(s') - V(s)
- **Backprop**: delta = target - prediction
- **Active inference**: free energy minimization

---

## Section 2: Surprise as Learning Signal - Information-Theoretic View

### Surprise = Negative Log Probability

From [Achiam & Sastry, 2017 - Surprise-Based Intrinsic Motivation](https://arxiv.org/abs/1703.01732):

```python
import torch
import torch.nn.functional as F

def compute_surprise(prediction_probs, actual_outcome):
    """
    Surprise = -log P(outcome | prediction)

    High surprise = unexpected outcome = large learning signal
    Low surprise = expected outcome = small learning signal
    """
    # Get probability assigned to actual outcome
    prob = prediction_probs[actual_outcome]

    # Surprise = negative log probability
    surprise = -torch.log(prob + 1e-8)

    return surprise
```

### Cross-Entropy Loss IS Surprise

The standard classification loss is actually measuring surprise:

```python
def cross_entropy_as_surprise(logits, targets):
    """
    Cross-entropy loss = average surprise across batch

    CE = -sum(y_true * log(y_pred))
       = -log(p(correct_class))
       = surprise at correct answer
    """
    # This is what PyTorch's CrossEntropyLoss computes
    probs = F.softmax(logits, dim=-1)

    # Get probability of true class
    batch_size = logits.shape[0]
    true_class_probs = probs[range(batch_size), targets]

    # Surprise = -log(prob)
    surprises = -torch.log(true_class_probs + 1e-8)

    # Average surprise across batch
    return surprises.mean()
```

### KL-Divergence as Expected Surprise Difference

```python
def kl_divergence(p, q):
    """
    KL(P || Q) = sum(P * log(P/Q))
               = sum(P * log(P)) - sum(P * log(Q))
               = -H(P) + cross_entropy(P, Q)
               = expected surprise under Q minus entropy of P

    This measures how much MORE surprised we are using Q
    instead of the true distribution P.
    """
    return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum()
```

### Intrinsic Motivation via Surprise

From the surprise-based RL paper:

```python
class SurpriseBasedExploration:
    """
    Use prediction error as intrinsic reward for exploration.
    Agent seeks surprising states = states where it can learn!
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        # Dynamics model predicts next state
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Intrinsic reward = prediction error = surprise

        r_intrinsic = ||next_state - predicted_next_state||^2
        """
        # Predict next state
        state_action = torch.cat([state, action], dim=-1)
        predicted_next = self.dynamics_model(state_action)

        # Prediction error = surprise = intrinsic reward
        surprise = F.mse_loss(predicted_next, next_state)

        return surprise

    def update_dynamics_model(self, state, action, next_state):
        """Learn to predict transitions (reduce surprise over time)."""
        state_action = torch.cat([state, action], dim=-1)
        predicted = self.dynamics_model(state_action)
        loss = F.mse_loss(predicted, next_state)
        return loss
```

---

## Section 3: Curriculum by Prediction Difficulty

### The Core Idea: Learn What You Don't Know

Optimal learning focuses on examples at the right difficulty level:
- **Too easy**: Low prediction error, nothing to learn
- **Too hard**: High prediction error but noisy, hard to extract signal
- **Just right**: Moderate prediction error, maximum learning signal

### Curriculum Learning via Prediction Error

```python
class PredictionErrorCurriculum:
    """
    Order training examples by prediction difficulty.
    Start with easy (low error) and gradually increase.
    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.difficulty_scores = None

    def compute_difficulty_scores(self):
        """Score each example by model's prediction error."""
        self.model.eval()
        scores = []

        with torch.no_grad():
            for x, y in self.dataset:
                pred = self.model(x.unsqueeze(0))
                error = F.cross_entropy(pred, y.unsqueeze(0))
                scores.append(error.item())

        self.difficulty_scores = scores
        return scores

    def get_curriculum_batch(self, epoch, total_epochs, batch_size):
        """
        Sample batch with difficulty proportional to training progress.
        Early epochs: easy examples (low error)
        Later epochs: hard examples (high error)
        """
        # What fraction of training complete?
        progress = epoch / total_epochs

        # Max difficulty to include (increases over training)
        max_difficulty_percentile = 0.3 + 0.7 * progress

        # Get threshold
        threshold = np.percentile(self.difficulty_scores,
                                   max_difficulty_percentile * 100)

        # Get indices of examples within difficulty range
        valid_indices = [i for i, score in enumerate(self.difficulty_scores)
                        if score <= threshold]

        # Sample batch
        batch_indices = np.random.choice(valid_indices,
                                         size=min(batch_size, len(valid_indices)),
                                         replace=False)

        return [self.dataset[i] for i in batch_indices]
```

### Self-Paced Learning

```python
class SelfPacedLearning:
    """
    Dynamically weight examples by inverse prediction error.
    Focus on what's currently learnable.
    """

    def __init__(self, initial_pace=1.0, pace_increase=0.1):
        self.pace = initial_pace  # Controls difficulty range
        self.pace_increase = pace_increase

    def compute_sample_weights(self, losses):
        """
        Weight = 1 if loss < threshold, else decay

        Soft version: weight = exp(-loss / pace)
        """
        # Hard threshold
        weights = (losses < self.pace).float()

        # Or soft weighting
        # weights = torch.exp(-losses / self.pace)

        return weights

    def weighted_loss(self, individual_losses):
        """Compute weighted average loss."""
        weights = self.compute_sample_weights(individual_losses)
        weighted_loss = (weights * individual_losses).sum() / weights.sum()
        return weighted_loss

    def increase_pace(self):
        """Allow harder examples after training progress."""
        self.pace += self.pace_increase
```

### Active Learning via Uncertainty

```python
class UncertaintySampling:
    """
    Sample examples where model is most uncertain.
    Uncertainty = expected prediction error = expected surprise
    """

    def __init__(self, model):
        self.model = model

    def compute_uncertainty(self, x):
        """
        Uncertainty measures (all related to prediction error):
        1. Entropy of prediction
        2. Least confidence
        3. Margin between top two
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

        # Entropy (expected surprise)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Least confidence
        confidence = probs.max(dim=-1)[0]
        uncertainty = 1 - confidence

        # Margin
        sorted_probs = probs.sort(dim=-1, descending=True)[0]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        margin_uncertainty = 1 - margin

        return {
            'entropy': entropy,
            'confidence_uncertainty': uncertainty,
            'margin_uncertainty': margin_uncertainty
        }

    def select_samples(self, pool, n_samples, method='entropy'):
        """Select n_samples with highest uncertainty."""
        uncertainties = self.compute_uncertainty(pool)[method]
        top_indices = uncertainties.topk(n_samples)[1]
        return top_indices
```

---

## Section 4: Complete Error-Driven Training Implementation

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple

class ErrorDrivenTrainer:
    """
    Complete error-driven training loop with:
    - Prediction error tracking
    - Surprise-based sample weighting
    - Curriculum learning
    - Intrinsic motivation for exploration
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        use_curriculum: bool = True,
        use_self_paced: bool = False,
        track_surprise: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.use_curriculum = use_curriculum
        self.use_self_paced = use_self_paced
        self.track_surprise = track_surprise

        # Tracking
        self.epoch_losses = []
        self.sample_difficulties = {}
        self.learning_progress = []

        # Self-paced learning
        if use_self_paced:
            self.pace = 1.0
            self.pace_growth = 0.1

    def compute_prediction_errors(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute prediction errors (loss per sample).

        This IS the surprise signal that drives learning!
        """
        if outputs.dim() > 1 and outputs.shape[-1] > 1:
            # Classification: cross-entropy = surprise
            errors = F.cross_entropy(outputs, targets, reduction=reduction)
        else:
            # Regression: MSE = squared prediction error
            errors = F.mse_loss(outputs, targets, reduction=reduction)

        return errors

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with error-driven learning.

        Returns metrics including:
        - Average loss (prediction error)
        - Average surprise
        - Learning progress (error reduction)
        """
        self.model.train()

        total_loss = 0.0
        total_surprise = 0.0
        num_batches = 0

        all_errors = []

        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass: make predictions
            outputs = self.model(data)

            # Compute prediction errors (per sample)
            errors = self.compute_prediction_errors(outputs, targets, reduction='none')

            # Track individual sample difficulties
            if self.track_surprise:
                all_errors.extend(errors.detach().cpu().tolist())

            # Apply sample weighting if using self-paced learning
            if self.use_self_paced:
                weights = self._compute_self_paced_weights(errors)
                loss = (weights * errors).sum() / weights.sum()
            else:
                loss = errors.mean()

            # Backward pass: compute gradients from prediction error
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update parameters to reduce prediction error
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_surprise += (-torch.log(F.softmax(outputs, dim=-1).max(dim=-1)[0] + 1e-8)).mean().item()
            num_batches += 1

        # Update self-paced learning threshold
        if self.use_self_paced:
            self.pace += self.pace_growth

        # Compute learning progress (error reduction from last epoch)
        avg_loss = total_loss / num_batches
        if len(self.epoch_losses) > 0:
            learning_progress = self.epoch_losses[-1] - avg_loss
        else:
            learning_progress = 0.0

        self.epoch_losses.append(avg_loss)
        self.learning_progress.append(learning_progress)

        return {
            'loss': avg_loss,
            'surprise': total_surprise / num_batches,
            'learning_progress': learning_progress,
            'error_std': np.std(all_errors) if all_errors else 0.0
        }

    def _compute_self_paced_weights(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Compute sample weights based on prediction error.
        Focus on learnable examples (moderate difficulty).
        """
        # Soft weighting: lower weight for very high errors
        weights = torch.exp(-errors / self.pace)

        # Normalize
        weights = weights / weights.sum() * len(weights)

        return weights

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model and track prediction errors."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_errors = []

        for data, targets in dataloader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(data)
            errors = self.compute_prediction_errors(outputs, targets, reduction='none')

            total_loss += errors.mean().item()
            all_errors.extend(errors.cpu().tolist())

            # Accuracy for classification
            if outputs.dim() > 1 and outputs.shape[-1] > 1:
                pred = outputs.argmax(dim=-1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0.0,
            'error_mean': np.mean(all_errors),
            'error_std': np.std(all_errors),
            'error_max': np.max(all_errors),
            'error_min': np.min(all_errors)
        }


class SurpriseAwareModel(nn.Module):
    """
    Neural network that explicitly tracks its own surprise/uncertainty.
    Outputs both predictions and confidence estimates.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Prediction head
        self.predictor = nn.Linear(hidden_dim, output_dim)

        # Uncertainty head (log variance)
        self.uncertainty = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            predictions: Model's predictions
            log_var: Log variance (uncertainty estimate)
        """
        features = self.backbone(x)
        predictions = self.predictor(features)
        log_var = self.uncertainty(features)

        return predictions, log_var

    def loss_with_uncertainty(
        self,
        x: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Heteroscedastic loss: weight errors by predicted uncertainty.

        Loss = 0.5 * exp(-log_var) * (pred - target)^2 + 0.5 * log_var

        This teaches the model to:
        - Predict well where it's confident
        - Admit uncertainty where it can't predict
        """
        predictions, log_var = self.forward(x)

        # Precision = inverse variance
        precision = torch.exp(-log_var)

        # Weighted squared error + complexity penalty
        sq_error = (predictions - targets) ** 2
        loss = 0.5 * precision * sq_error + 0.5 * log_var

        return loss.mean()


def train_with_error_driven_learning():
    """Example training loop demonstrating error-driven learning."""

    # Create synthetic dataset
    X = torch.randn(1000, 10)
    y = (X.sum(dim=1) > 0).long()  # Binary classification
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Error-driven trainer
    trainer = ErrorDrivenTrainer(
        model=model,
        optimizer=optimizer,
        device='cpu',
        use_curriculum=True,
        track_surprise=True
    )

    # Training loop
    for epoch in range(50):
        metrics = trainer.train_epoch(dataloader, epoch, 50)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                  f"Surprise={metrics['surprise']:.4f}, "
                  f"Progress={metrics['learning_progress']:.4f}")

    # Final evaluation
    eval_metrics = trainer.evaluate(dataloader)
    print(f"\nFinal: Accuracy={eval_metrics['accuracy']:.4f}, "
          f"Error std={eval_metrics['error_std']:.4f}")

    return trainer


# Demonstration
if __name__ == "__main__":
    trainer = train_with_error_driven_learning()
```

### Performance Notes

**Memory considerations:**
- Tracking per-sample errors: O(N) memory for N samples
- Self-paced weighting adds minimal overhead
- Gradient computation is the main cost

**GPU optimization:**
```python
# Enable mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(data)
    loss = F.cross_entropy(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Batch size effects:**
- Larger batches: smoother gradient estimates, less noise
- Smaller batches: noisier gradients can help escape local minima
- Error variance decreases with sqrt(batch_size)

---

## Section 5: TRAIN STATION - Loss = Prediction Error = Surprise = Free Energy

### The Grand Unification

**This is THE train station where all these concepts meet:**

```
LOSS FUNCTION
     |
     |  (is defined as)
     v
PREDICTION ERROR = |prediction - actual|
     |
     |  (in probability terms)
     v
SURPRISE = -log P(actual | prediction)
     |
     |  (plus complexity term)
     v
FREE ENERGY = Surprise + KL(approximate || true)
     |
     |  (expected value)
     v
EXPECTED FREE ENERGY = Future surprise = Uncertainty
```

### Mathematical Equivalences

**1. MSE Loss = Gaussian Surprise**

```python
def mse_is_gaussian_surprise():
    """
    Under Gaussian assumption:
    P(y|x) = N(y; f(x), sigma^2)

    -log P(y|x) = 0.5 * (y - f(x))^2 / sigma^2 + const

    MSE is negative log likelihood = surprise!
    """
    # Model predicts mean of Gaussian
    prediction = model(x)  # f(x)

    # MSE loss
    mse = 0.5 * (y - prediction) ** 2

    # This IS the surprise under Gaussian assumption
    # (ignoring constant terms)
    return mse
```

**2. Cross-Entropy = Categorical Surprise**

```python
def cross_entropy_is_categorical_surprise():
    """
    P(class=k|x) = softmax(f(x))_k

    -log P(correct_class|x) = cross_entropy

    Cross-entropy is surprise at correct answer!
    """
    logits = model(x)
    probs = F.softmax(logits, dim=-1)

    # Surprise at correct class
    surprise = -torch.log(probs[correct_class])

    # This IS cross-entropy
    return surprise
```

**3. Free Energy = Prediction Error + Complexity**

```python
def free_energy_decomposition(model, x, y):
    """
    Free Energy F = E_q[-log P(y,z|x)] + KL(q(z|x) || P(z))

    Under predictive coding:
    F = Prediction_Error + Complexity

    Where:
    - Prediction Error = surprise at observation
    - Complexity = how much beliefs deviate from prior
    """
    # Get model predictions and uncertainty
    pred_mean, pred_logvar = model(x)

    # Prediction error (surprise)
    prediction_error = 0.5 * torch.exp(-pred_logvar) * (y - pred_mean)**2

    # Complexity (KL from prior)
    # Assuming standard normal prior
    complexity = 0.5 * (pred_mean**2 + torch.exp(pred_logvar) - pred_logvar - 1)

    # Free energy
    free_energy = prediction_error + complexity

    return free_energy, prediction_error, complexity
```

**4. VAE Loss = Free Energy**

```python
def vae_loss_is_free_energy(x, recon_x, mu, logvar):
    """
    VAE Loss = Reconstruction + KL
             = Prediction Error + Complexity
             = Free Energy!
    """
    # Reconstruction error (prediction error / surprise)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence (complexity)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

    # Total = Free Energy
    free_energy = recon_loss + kl_loss

    return free_energy
```

### Why This Unification Matters

**For understanding:**
- All learning is prediction error minimization
- Different loss functions = different assumptions about noise
- Free energy adds principled complexity penalty

**For practice:**
- Choose loss based on data distribution (Gaussian vs categorical vs...)
- Regularization = complexity penalty = prior
- Uncertainty estimation comes free from probabilistic view

**For research:**
- Active inference: actions minimize expected free energy
- Curiosity: seek high expected surprise (learning potential)
- Curriculum: match difficulty to current prediction ability

---

## Section 6: ARR-COC-0-1 Connection - Relevance-Weighted Errors

### Core Insight: Not All Errors Are Equal

In ARR-COC (Attention-based Relevance Reranking), we care about prediction errors on RELEVANT content more than irrelevant content. This connects directly to precision-weighted prediction errors from active inference.

### Relevance-Weighted Loss

```python
class RelevanceWeightedLoss(nn.Module):
    """
    Weight prediction errors by relevance scores.

    High relevance = high precision = error matters more
    Low relevance = low precision = error matters less

    This is EXACTLY precision-weighted prediction error!
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        relevance_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [batch, ...]
            targets: Ground truth [batch, ...]
            relevance_scores: Relevance weights [batch] in [0, 1]

        Returns:
            Relevance-weighted loss
        """
        # Compute per-sample errors
        errors = F.mse_loss(predictions, targets, reduction='none')

        # Average over non-batch dimensions
        if errors.dim() > 1:
            errors = errors.mean(dim=list(range(1, errors.dim())))

        # Weight by relevance (precision)
        # Higher relevance = higher weight on error
        weighted_errors = relevance_scores * errors

        # Return mean weighted error
        return weighted_errors.mean()
```

### Surprise-Based Token Allocation

```python
class SurpriseBasedTokenAllocator:
    """
    Allocate more compute tokens to high-surprise (unexpected) regions.

    This is adaptive precision:
    - High surprise = need more processing = more tokens
    - Low surprise = already understood = fewer tokens
    """

    def __init__(self, predictor_model: nn.Module):
        self.predictor = predictor_model

    def compute_region_surprises(
        self,
        image_regions: torch.Tensor  # [batch, num_regions, dim]
    ) -> torch.Tensor:
        """Compute surprise for each region based on prediction error."""
        # Predict each region from context
        batch_size, num_regions, dim = image_regions.shape

        surprises = []
        for i in range(num_regions):
            # Context = all other regions
            context = torch.cat([
                image_regions[:, :i, :],
                image_regions[:, i+1:, :]
            ], dim=1)

            # Predict region i from context
            predicted = self.predictor(context)
            actual = image_regions[:, i, :]

            # Surprise = prediction error
            surprise = F.mse_loss(predicted, actual, reduction='none').mean(dim=-1)
            surprises.append(surprise)

        return torch.stack(surprises, dim=1)  # [batch, num_regions]

    def allocate_tokens(
        self,
        image_regions: torch.Tensor,
        total_tokens: int
    ) -> torch.Tensor:
        """
        Allocate tokens proportional to surprise.
        High-surprise regions get more tokens.
        """
        surprises = self.compute_region_surprises(image_regions)

        # Normalize to get allocation proportions
        allocations = F.softmax(surprises, dim=1)

        # Scale to total tokens
        token_counts = (allocations * total_tokens).long()

        return token_counts
```

### Curriculum via Relevance-Adjusted Difficulty

```python
class RelevanceCurriculum:
    """
    Curriculum that considers both difficulty AND relevance.

    Priority = Relevance * (1 - Mastery)

    Where Mastery = 1 - normalized_error

    Focus on relevant content we haven't mastered yet.
    """

    def __init__(self, model: nn.Module, relevance_model: nn.Module):
        self.model = model
        self.relevance_model = relevance_model

    def compute_priorities(
        self,
        data: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute learning priority for each sample."""
        self.model.eval()
        self.relevance_model.eval()

        with torch.no_grad():
            # Compute prediction errors (difficulty)
            predictions = self.model(data)
            errors = F.mse_loss(predictions, targets, reduction='none').mean(dim=-1)

            # Normalize errors to [0, 1]
            errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

            # Compute relevance
            relevance = self.relevance_model(data).squeeze()

            # Priority = relevance * (1 - mastery) = relevance * error
            # High relevance + high error = high priority
            priorities = relevance * errors_norm

        return priorities

    def sample_batch(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch weighted by priority."""
        priorities = self.compute_priorities(data, targets)

        # Sample proportional to priority
        probs = F.softmax(priorities, dim=0)
        indices = torch.multinomial(probs, batch_size, replacement=False)

        return data[indices], targets[indices]
```

### Connection to Active Inference

```python
class ActiveRelevanceInference:
    """
    Active inference for relevance estimation.

    The model maintains beliefs about relevance and updates them
    based on prediction errors, then takes actions (allocates attention)
    to minimize expected free energy.
    """

    def __init__(self, belief_dim: int, action_dim: int):
        # Beliefs about relevance (encoded as parameters)
        self.beliefs = nn.Parameter(torch.zeros(belief_dim))

        # Generative model: beliefs -> observations
        self.generative_model = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def perception_step(
        self,
        observation: torch.Tensor
    ) -> torch.Tensor:
        """
        Update beliefs based on prediction error.

        This is variational inference / free energy minimization.
        """
        # Generate prediction from current beliefs
        prediction = self.generative_model(self.beliefs)

        # Prediction error
        error = observation - prediction

        # Update beliefs to reduce error
        # (In full implementation, this would be gradient descent on free energy)
        belief_update = error.mean() * 0.1  # Simplified
        self.beliefs.data += belief_update

        return error

    def action_selection(self) -> torch.Tensor:
        """
        Select action (attention allocation) to minimize expected free energy.

        EFE = Expected surprise + epistemic value + pragmatic value
        """
        # Generate action from beliefs
        action = self.generative_model(self.beliefs)

        # This action allocates attention/tokens based on
        # what we expect to be relevant (minimize future surprise)
        return torch.softmax(action, dim=-1)
```

### Performance Implications

**Memory efficiency:**
- Store relevance scores alongside data
- Weighted sampling is O(N) for N samples

**Training dynamics:**
- Relevance weighting focuses gradients on important examples
- Prevents catastrophic forgetting of rare but relevant cases

**Inference optimization:**
- Surprise-based allocation is adaptive compute
- More processing for uncertain regions, less for easy ones

---

## Sources

### Primary References

**Web Research (accessed 2025-11-23):**

- [Wikipedia - Error-driven learning](https://en.wikipedia.org/wiki/Error-driven_learning) - Formal definition and applications
- [Achiam & Sastry, 2017 - Surprise-Based Intrinsic Motivation](https://arxiv.org/abs/1703.01732) - Surprise for RL exploration
- [Friston, 2009 - Predictive coding under the free-energy principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) - Free energy and prediction error
- [Hoppe et al., 2022 - Error-driven learning in simple two-layer networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC9579095/) - Discriminative learning perspective

**Key Papers:**

- O'Reilly, R.C. (1996). Biologically Plausible Error-Driven Learning Using Local Activation Differences. Neural Computation.
- Bengio, Y. (2009). Learning deep architectures for AI. Foundations and Trends in ML.
- White, A. (2014). Surprise and Curiosity for Big Data Robotics. AAAI Workshop.

### Implementation References

**PyTorch Documentation:**
- [torch.nn.functional](https://pytorch.org/docs/stable/nn.functional.html) - Loss functions
- [torch.optim](https://pytorch.org/docs/stable/optim.html) - Optimizers

**Related Concepts:**
- Temporal Difference Learning
- Rescorla-Wagner Model
- Predictive Coding Networks
- Active Inference

---

## Summary: The Error-Driven Universe

**Everything in ML is error-driven learning:**

1. **Supervised learning**: Minimize prediction error on labels
2. **Unsupervised learning**: Minimize reconstruction error
3. **Reinforcement learning**: Minimize TD error (value prediction error)
4. **Active inference**: Minimize free energy = prediction error + complexity
5. **Exploration**: Seek high expected error (surprise) for learning

**The train station unification:**

```
Loss = Prediction Error = Surprise = Free Energy

All roads lead to the same destination:
Making better predictions about the world.
```

**For ARR-COC:**
- Relevance = precision weighting on errors
- Important content = high-precision = errors matter more
- Token allocation = adaptive precision = more compute for uncertain regions

**The free energy principle gives us a principled way to:**
- Weight errors by relevance (precision)
- Trade off accuracy vs complexity
- Guide exploration by expected information gain
- Unify perception, learning, and action

This is why "Loss = Prediction Error = Surprise = Free Energy" is the GRAND TRAIN STATION of machine learning - it's where all the concepts meet and become one.
