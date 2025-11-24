# W&B Quick Validation: Smoke Tests & Rapid Model Verification

## Overview

Quick validation methodology focuses on catching bugs and confirming model capability before investing in full-scale training. This approach, championed by Karpathy and the Full Stack Deep Learning team, treats model debugging as a structured process rather than trial-and-error experimentation.

**Core Philosophy**: "Don't tell me it works, show me the loss curve dropping to zero on 2 examples."

From [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (Karpathy, 2019):
> "A 'fast and furious' approach to training neural networks does not work and only leads to suffering. The qualities that correlate most strongly to success in deep learning are patience and attention to detail."

## Section 1: Smoke Tests (~120 lines)

### What is a Smoke Test?

From [Smoke testing for machine learning](https://link.springer.com/article/10.1007/s10664-021-10073-7) (Herbold et al., 2022):
> "Smoke tests are simple tests to assert that basic functions can be executed without crashing. They catch major issues or errors that could prevent the model from running at all."

**Purpose**: Verify that your model pipeline runs without crashing before investing time in full training.

**Key Insight**: Most deep learning bugs fail silently - the code runs, but produces wrong results. Smoke tests catch the noisy failures first.

### 1.1 Forward Pass Test

**Goal**: Ensure data flows through your model without shape errors.

```python
import wandb

# Initialize W&B run for smoke testing
wandb.init(
    project="arr-coc-smoke-tests",
    name="forward-pass-test",
    config={"test_type": "forward_pass"}
)

# Create dummy input matching expected shape
dummy_batch = {
    'images': torch.randn(2, 3, 224, 224),  # 2 samples, RGB, 224x224
    'queries': torch.randn(2, 512)           # 2 query embeddings
}

try:
    # Forward pass
    outputs = model(dummy_batch)

    # Log success
    wandb.log({
        "smoke_test/forward_pass": "PASS",
        "smoke_test/output_shape": str(outputs.shape),
        "smoke_test/output_dtype": str(outputs.dtype)
    })
    print("✓ Forward pass successful")

except Exception as e:
    # Log failure
    wandb.log({
        "smoke_test/forward_pass": "FAIL",
        "smoke_test/error": str(e)
    })
    print(f"✗ Forward pass failed: {e}")
    raise

wandb.finish()
```

**What to check**:
- Model accepts input without shape mismatch
- Output shape matches expected dimensions
- No NaN or Inf values in output

### 1.2 Backward Pass Test

**Goal**: Verify gradients flow correctly through the model.

From [Full Stack Deep Learning Lecture 7](https://fullstackdeeplearning.com/spring2021/lecture-7/):
> "Use backprop to chart dependencies. Set the loss to be something trivial like the sum of all outputs and ensure you get non-zero gradients only where expected."

```python
wandb.init(
    project="arr-coc-smoke-tests",
    name="backward-pass-test",
    config={"test_type": "backward_pass"}
)

# Forward pass
outputs = model(dummy_batch)
loss = outputs.sum()  # Trivial loss

# Backward pass
loss.backward()

# Check gradients
gradient_stats = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        gradient_stats[f"grad/{name}/mean"] = param.grad.mean().item()
        gradient_stats[f"grad/{name}/std"] = param.grad.std().item()
        gradient_stats[f"grad/{name}/max"] = param.grad.max().item()

        # Check for NaN/Inf
        has_nan = torch.isnan(param.grad).any().item()
        has_inf = torch.isinf(param.grad).any().item()

        if has_nan or has_inf:
            wandb.log({
                "smoke_test/backward_pass": "FAIL",
                f"smoke_test/gradient_issue/{name}": "NaN" if has_nan else "Inf"
            })
            raise ValueError(f"Gradient issue in {name}")

# Log gradient statistics
wandb.log(gradient_stats)
wandb.log({"smoke_test/backward_pass": "PASS"})

wandb.finish()
```

**Red flags**:
- All gradients are zero (dead network)
- NaN or Inf gradients (numerical instability)
- Gradients only non-zero in unexpected layers (architecture bug)

### 1.3 Overfit One Batch Test

**The most powerful debugging technique.**

From Karpathy's recipe:
> "Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model and verify that we can reach the lowest achievable loss (e.g. zero). If they do not align perfectly once we reach minimum loss, there is a bug somewhere."

```python
wandb.init(
    project="arr-coc-smoke-tests",
    name="overfit-one-batch",
    config={
        "test_type": "overfit_one_batch",
        "batch_size": 2,
        "max_steps": 500,
        "target_loss": 0.01  # Near-zero for classification
    }
)

# Get single batch
single_batch = next(iter(train_dataloader))
single_batch = {k: v[:2] for k, v in single_batch.items()}  # Just 2 examples

# Increase model capacity if needed
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for step in range(500):
    optimizer.zero_grad()

    outputs = model(single_batch)
    loss = criterion(outputs, single_batch['labels'])

    loss.backward()
    optimizer.step()

    # Log every step
    wandb.log({
        "overfit_test/loss": loss.item(),
        "overfit_test/step": step
    })

    # Check if we've reached target
    if loss.item() < 0.01:
        wandb.log({
            "smoke_test/overfit_one_batch": "PASS",
            "smoke_test/steps_to_converge": step
        })
        print(f"✓ Overfitting successful in {step} steps")
        break
else:
    # Failed to overfit
    wandb.log({"smoke_test/overfit_one_batch": "FAIL"})
    print("✗ Failed to overfit single batch - model cannot learn")

wandb.finish()
```

**What this catches**:
- Flipped signs in loss function
- Data preprocessing bugs (e.g., labels not matching images)
- Model capacity issues
- Learning rate problems
- Off-by-one errors in sequence models

### 1.4 Shape Validation Test

**Goal**: Verify tensor shapes throughout the network.

```python
wandb.init(
    project="arr-coc-smoke-tests",
    name="shape-validation",
    config={"test_type": "shape_validation"}
)

# Register hooks to capture intermediate shapes
activation_shapes = {}

def get_activation(name):
    def hook(model, input, output):
        activation_shapes[name] = {
            "input_shape": str(input[0].shape) if isinstance(input, tuple) else str(input.shape),
            "output_shape": str(output.shape)
        }
    return hook

# Attach hooks to all modules
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Leaf modules only
        module.register_forward_hook(get_activation(name))

# Forward pass
outputs = model(dummy_batch)

# Log all shapes
for layer_name, shapes in activation_shapes.items():
    wandb.log({
        f"shapes/{layer_name}/input": shapes["input_shape"],
        f"shapes/{layer_name}/output": shapes["output_shape"]
    })

wandb.finish()
```

**Common shape bugs** (from Full Stack DL):
- Silent broadcasting in automatic differentiation
- Transpose vs. permute confusion
- Batch dimension accidentally mixed across examples

## Section 2: Quick Validation (~130 lines)

### The 100/10 Pattern

**Philosophy**: Validate on tiny dataset before full training.

From [Full Stack Deep Learning Lecture 7](https://fullstackdeeplearning.com/spring2021/lecture-7/):
> "Simplify the problem itself. Work with a small training set around 10,000 examples. Use a fixed number of objects, classes, input size. Your iteration speed will increase."

**ARR-COC Context**: From `VALIDATION-FOR-PLATONIC-CODING-CODEBASES.md`:
> "100 examples, 10 epochs - watch it overfit. This is your smoke test. If it doesn't overfit, something is fundamentally broken."

### 2.1 Quick Validation Setup

```python
import wandb
from torch.utils.data import Subset

# Initialize W&B for quick validation
wandb.init(
    project="arr-coc-validation",
    name="quick-validation-100-examples",
    config={
        "validation_type": "quick",
        "num_examples": 100,
        "num_epochs": 10,
        "expected_outcome": "overfit"
    }
)

# Create tiny dataset (100 examples)
train_dataset_small = Subset(train_dataset, indices=range(100))
train_loader_small = DataLoader(
    train_dataset_small,
    batch_size=16,
    shuffle=True
)

# Use same 100 examples for validation (should overfit)
val_loader_small = DataLoader(
    train_dataset_small,
    batch_size=16,
    shuffle=False
)
```

### 2.2 Metrics to Track

**Core metrics for quick validation**:

```python
def run_quick_validation(model, train_loader, val_loader, epochs=10):
    """
    Quick validation loop with comprehensive W&B logging.

    Success criteria:
    - Training loss should decrease monotonically
    - Validation loss should decrease (we're using train set)
    - Final loss should be near zero (overfitting)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase (on same data - should overfit)
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                loss = criterion(outputs, batch['labels'])
                val_losses.append(loss.item())

                # Accuracy
                _, predicted = outputs.max(1)
                total += batch['labels'].size(0)
                correct += predicted.eq(batch['labels']).sum().item()

        # Log to W&B
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        accuracy = 100. * correct / total

        wandb.log({
            "quick_val/epoch": epoch,
            "quick_val/train_loss": avg_train_loss,
            "quick_val/val_loss": avg_val_loss,
            "quick_val/accuracy": accuracy,
            "quick_val/train_val_gap": avg_train_loss - avg_val_loss
        })

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Acc={accuracy:.2f}%")

    # Final validation
    final_train_loss = avg_train_loss
    final_accuracy = accuracy

    # Success criteria
    success = final_train_loss < 0.5 and final_accuracy > 90.0

    wandb.log({
        "quick_val/final_train_loss": final_train_loss,
        "quick_val/final_accuracy": final_accuracy,
        "quick_val/success": success
    })

    return success
```

### 2.3 Success Criteria

**What "success" looks like in quick validation**:

From Karpathy's recipe:
> "Verify decreasing training loss. At this stage you will hopefully be underfitting because you're working with a toy model. Try to increase its capacity. Did your training loss go down as it should?"

**Quick validation checklist**:

```python
def evaluate_quick_validation_results(run_id):
    """
    Analyze W&B run to determine if quick validation succeeded.

    Returns detailed diagnostics and recommendations.
    """
    api = wandb.Api()
    run = api.run(f"your-entity/arr-coc-validation/{run_id}")

    history = run.history()

    # Extract metrics
    final_train_loss = history['quick_val/train_loss'].iloc[-1]
    final_val_loss = history['quick_val/val_loss'].iloc[-1]
    final_accuracy = history['quick_val/accuracy'].iloc[-1]

    # Check for decreasing trend
    train_loss_decreasing = history['quick_val/train_loss'].is_monotonic_decreasing

    # Diagnostics
    diagnostics = {
        "train_loss_decreased": train_loss_decreasing,
        "final_train_loss_low": final_train_loss < 0.5,
        "final_accuracy_high": final_accuracy > 90.0,
        "overfitting_achieved": final_train_loss < 0.1 and final_val_loss < 0.2
    }

    # Recommendations
    recommendations = []

    if not train_loss_decreasing:
        recommendations.append(
            "⚠️ Training loss not decreasing - check learning rate or model capacity"
        )

    if final_train_loss > 0.5:
        recommendations.append(
            "⚠️ Final train loss too high - model cannot fit data. "
            "Increase capacity or check data pipeline."
        )

    if final_accuracy < 90.0:
        recommendations.append(
            "⚠️ Accuracy too low on 100 examples - fundamental issue with model/data"
        )

    if diagnostics["overfitting_achieved"]:
        recommendations.append(
            "✓ Successfully overfit 100 examples - model can learn. "
            "Ready for full training."
        )

    # Log diagnostics to W&B
    wandb.log({
        "diagnostics/train_loss_decreased": diagnostics["train_loss_decreased"],
        "diagnostics/final_train_loss_low": diagnostics["final_train_loss_low"],
        "diagnostics/final_accuracy_high": diagnostics["final_accuracy_high"],
        "diagnostics/overfitting_achieved": diagnostics["overfitting_achieved"]
    })

    return diagnostics, recommendations
```

### 2.4 ARR-COC Specific Quick Validation

**Testing relevance realization on small dataset**:

```python
def arr_coc_quick_validation():
    """
    Quick validation specifically for ARR-COC model.
    Tests relevance scoring and token allocation on 100 examples.
    """

    wandb.init(
        project="arr-coc-validation",
        name="arr-coc-quick-val",
        config={
            "num_examples": 100,
            "num_epochs": 10,
            "test_relevance_scores": True
        }
    )

    # Create 100-example dataset
    small_dataset = create_small_dataset(num_examples=100)

    for epoch in range(10):
        for batch in small_dataset:
            # Forward pass through relevance realization pipeline
            relevance_scores = model.knowing(batch['images'], batch['queries'])

            # Log relevance score statistics
            wandb.log({
                "relevance/propositional_mean": relevance_scores['propositional'].mean().item(),
                "relevance/perspectival_mean": relevance_scores['perspectival'].mean().item(),
                "relevance/participatory_mean": relevance_scores['participatory'].mean().item(),
                "relevance/propositional_std": relevance_scores['propositional'].std().item(),
                "relevance/perspectival_std": relevance_scores['perspectival'].std().item(),
                "relevance/participatory_std": relevance_scores['participatory'].std().item()
            })

            # Check token allocation
            token_budgets = model.attending(relevance_scores)

            wandb.log({
                "tokens/min_budget": token_budgets.min().item(),
                "tokens/max_budget": token_budgets.max().item(),
                "tokens/mean_budget": token_budgets.mean().item(),
                "tokens/total_allocated": token_budgets.sum().item()
            })

            # Verify budgets in valid range [64, 400]
            assert token_budgets.min() >= 64, "Token budget too low"
            assert token_budgets.max() <= 400, "Token budget too high"

    wandb.finish()
```

## Section 3: Debugging with W&B (~100 lines)

### 3.1 Red Flags During Training

From [Full Stack Deep Learning Lecture 7](https://fullstackdeeplearning.com/spring2021/lecture-7/):
> "There are a few things that can happen when you try to overfit a single batch and it fails:
> - Error goes up: Commonly due to a flip sign in loss function/gradient
> - Error explodes: Usually a numerical issue or high learning rate
> - Error oscillates: Lower learning rate, inspect data for shuffled labels
> - Error plateaus: Increase learning rate, inspect loss function and data pipeline"

**Red flag detection with W&B**:

```python
def detect_training_red_flags(run):
    """
    Analyze W&B run for common failure patterns.
    """

    history = run.history()

    red_flags = []

    # 1. Loss going up
    if history['loss'].iloc[-1] > history['loss'].iloc[0]:
        red_flags.append({
            "type": "LOSS_INCREASING",
            "severity": "CRITICAL",
            "message": "Loss increasing over time - likely sign flip in gradient",
            "recommendation": "Check loss function implementation"
        })

    # 2. Loss exploding (NaN or Inf)
    if history['loss'].isna().any() or (history['loss'] > 1e6).any():
        red_flags.append({
            "type": "LOSS_EXPLODING",
            "severity": "CRITICAL",
            "message": "Loss exploded to NaN/Inf - numerical instability",
            "recommendation": "Reduce learning rate, check for division by zero, add gradient clipping"
        })

    # 3. Loss oscillating
    loss_std = history['loss'].std()
    loss_mean = history['loss'].mean()
    if loss_std / loss_mean > 0.5:  # High relative variance
        red_flags.append({
            "type": "LOSS_OSCILLATING",
            "severity": "WARNING",
            "message": "Loss oscillating significantly",
            "recommendation": "Reduce learning rate, check batch shuffling, inspect labels"
        })

    # 4. Loss plateauing early
    recent_losses = history['loss'].iloc[-10:]
    if recent_losses.std() < 0.01 and recent_losses.mean() > 1.0:
        red_flags.append({
            "type": "LOSS_PLATEAUING",
            "severity": "WARNING",
            "message": "Loss plateaued at high value",
            "recommendation": "Increase learning rate, remove regularization, check data pipeline"
        })

    # Log red flags to W&B
    for flag in red_flags:
        wandb.log({
            f"red_flags/{flag['type']}": True,
            f"red_flags/{flag['type']}_severity": flag['severity']
        })

    return red_flags
```

### 3.2 Gradient Monitoring

**Karpathy's advice on using gradients for debugging**:

From [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/):
> "Gradients give you information about what depends on what in your network, which can be useful for debugging. Ensure you get a non-zero gradient only where expected."

```python
def monitor_gradients_with_wandb(model, loss):
    """
    Monitor gradient statistics during training.
    Catches dead neurons, exploding gradients, vanishing gradients.
    """

    loss.backward()

    gradient_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad

            # Basic statistics
            gradient_stats[f"gradients/{name}/mean"] = grad.mean().item()
            gradient_stats[f"gradients/{name}/std"] = grad.std().item()
            gradient_stats[f"gradients/{name}/max"] = grad.max().item()
            gradient_stats[f"gradients/{name}/min"] = grad.min().item()

            # Gradient norm
            grad_norm = grad.norm(2).item()
            gradient_stats[f"gradients/{name}/norm"] = grad_norm

            # Check for issues
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            is_zero = (grad.abs() < 1e-10).all().item()

            if has_nan or has_inf:
                gradient_stats[f"gradients/{name}/issue"] = "NaN/Inf"
            elif is_zero:
                gradient_stats[f"gradients/{name}/issue"] = "All zeros (dead)"
            elif grad_norm > 100:
                gradient_stats[f"gradients/{name}/issue"] = "Exploding"
            elif grad_norm < 1e-7:
                gradient_stats[f"gradients/{name}/issue"] = "Vanishing"

    wandb.log(gradient_stats)
```

### 3.3 Learning Rate Finding

**Quick experiment to find good learning rate**:

```python
def find_learning_rate_with_wandb(model, train_loader, min_lr=1e-7, max_lr=1e-1, num_steps=100):
    """
    Learning rate range test (fastai-style).

    Gradually increase LR and plot loss to find optimal range.
    """

    wandb.init(
        project="arr-coc-lr-finder",
        name="lr-range-test",
        config={
            "min_lr": min_lr,
            "max_lr": max_lr,
            "num_steps": num_steps
        }
    )

    # Exponentially increase LR
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)

    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)
    criterion = nn.CrossEntropyLoss()

    lrs = []
    losses = []

    model.train()
    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break

        # Current learning rate
        current_lr = min_lr * (lr_mult ** i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training step
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()

        # Record
        lrs.append(current_lr)
        losses.append(loss.item())

        # Log to W&B
        wandb.log({
            "lr_finder/lr": current_lr,
            "lr_finder/loss": loss.item(),
            "lr_finder/step": i
        })

    # Find optimal LR (steepest descent point)
    smoothed_losses = pd.Series(losses).rolling(5).mean()
    gradients = np.gradient(smoothed_losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]

    wandb.log({
        "lr_finder/optimal_lr": optimal_lr,
        "lr_finder/optimal_loss": losses[optimal_idx]
    })

    print(f"Optimal learning rate: {optimal_lr:.2e}")

    wandb.finish()

    return optimal_lr
```

### 3.4 Visualizing Training Dynamics

**Karpathy's advice on visualization**:

From [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/):
> "Visualize model predictions on a fixed test batch during training. The 'dynamics' of how these predictions move will give you incredibly good intuition for how training progresses. It's possible to feel the network 'struggle' to fit data if it wiggles too much."

```python
def log_prediction_dynamics(model, fixed_batch, step):
    """
    Visualize how predictions change over training on fixed examples.
    """

    model.eval()
    with torch.no_grad():
        outputs = model(fixed_batch)
        predictions = outputs.argmax(dim=1)
        confidences = torch.softmax(outputs, dim=1).max(dim=1)[0]

    # Log predictions as table
    table_data = []
    for i in range(len(fixed_batch['labels'])):
        table_data.append([
            i,
            fixed_batch['labels'][i].item(),
            predictions[i].item(),
            confidences[i].item()
        ])

    wandb.log({
        f"predictions/step_{step}": wandb.Table(
            columns=["example_id", "true_label", "predicted_label", "confidence"],
            data=table_data
        )
    })

    # Log confidence distribution
    wandb.log({
        "dynamics/mean_confidence": confidences.mean().item(),
        "dynamics/min_confidence": confidences.min().item(),
        "dynamics/max_confidence": confidences.max().item(),
        "dynamics/step": step
    })

    model.train()
```

## Sources

**Source Documents:**
- [VALIDATION-FOR-PLATONIC-CODING-CODEBASES.md](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/VALIDATION-FOR-PLATONIC-CODING-CODEBASES.md) - ARR-COC validation requirements (100 examples, 10 epochs pattern)

**Web Research:**
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) - Andrej Karpathy blog (accessed 2025-01-31)
  - Overfit one batch technique
  - Visualization of prediction dynamics
  - Gradient debugging strategies
  - Step-by-step troubleshooting process

- [Full Stack Deep Learning Lecture 7: Troubleshooting Deep Neural Networks](https://fullstackdeeplearning.com/spring2021/lecture-7/) (accessed 2025-01-31)
  - Bias-variance decomposition for debugging
  - Red flag patterns (loss increasing, exploding, oscillating, plateauing)
  - Quick validation on small datasets
  - Gradient monitoring techniques

**Additional References:**
- Herbold et al., 2022 - "Smoke testing for machine learning: simple tests to assert that basic functions can be executed"
- Full Stack Deep Learning course materials on systematic debugging approaches
- Karpathy's Twitter thread on common neural network mistakes

---

**Quick Start Checklist**:

1. ✓ Run forward pass test (shapes correct?)
2. ✓ Run backward pass test (gradients flowing?)
3. ✓ Overfit 2 examples (loss → 0?)
4. ✓ Quick validation on 100 examples (10 epochs, overfit?)
5. ✓ Monitor gradients (no NaN/Inf/vanishing?)
6. ✓ Check for red flags (loss behavior normal?)
7. ✓ Find good learning rate (LR range test)
8. ✓ Visualize prediction dynamics (model learning?)

If all pass → ready for full training with confidence.
If any fail → debug before scaling up.