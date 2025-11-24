# Vertex AI Vizier: Neural Architecture Search & Hyperparameter Tuning

## Overview

Vertex AI Vizier is Google's black-box optimization service for hyperparameter tuning and neural architecture search (NAS). Built on Google's internal Vizier system (used across Google since 2010), it automates the search for optimal model configurations without requiring manual trial-and-error.

**Core Value Proposition:**
- Reduces training cost by finding better hyperparameters faster
- Adapts to unknown convergence behavior (no need to model learning curves)
- Supports multi-objective optimization (accuracy + latency, accuracy + model size)
- Integrates seamlessly with Vertex AI Custom Training Jobs

**When to Use Vizier:**
- Hyperparameter tuning (learning rate, batch size, layer dimensions)
- Neural architecture search (number of layers, kernel sizes, activation functions)
- Multi-objective optimization (balancing accuracy, latency, memory, cost)
- Any black-box optimization problem with expensive evaluations

From [Vertex AI Vizier overview](https://cloud.google.com/vertex-ai/docs/vizier/overview) (accessed 2025-11-16):
> "Vertex AI Vizier is a black-box optimization service that helps you tune hyperparameters in complex machine learning models. When ML models have many different hyperparameters, it can be difficult and time consuming to tune them manually. Vertex AI Vizier optimizes your model's output by tuning the hyperparameters for you."

## Search Algorithms

Vizier supports multiple search strategies, each optimized for different scenarios:

### 1. Grid Search

**How it works:** Exhaustively evaluates all combinations in a discrete grid.

```python
# Grid search configuration
{
    "algorithm": "ALGORITHM_UNSPECIFIED",  # Defaults to grid for discrete params
    "parameters": [
        {
            "parameter_id": "learning_rate",
            "discrete_value_spec": {
                "values": [0.001, 0.01, 0.1]
            }
        },
        {
            "parameter_id": "batch_size",
            "discrete_value_spec": {
                "values": [32, 64, 128, 256]
            }
        }
    ]
}
# Total trials: 3 * 4 = 12
```

**Pros:**
- Guaranteed to find best combination in search space
- Reproducible and interpretable
- No algorithmic complexity

**Cons:**
- Exponential growth with dimensionality (curse of dimensionality)
- Wasteful for continuous parameters
- No early stopping

**Best for:**
- Small discrete search spaces (< 100 combinations)
- Debugging/validation (reproducible results)
- When you need exhaustive coverage

### 2. Random Search

**How it works:** Samples hyperparameter configurations uniformly at random.

```python
{
    "algorithm": "RANDOM_SEARCH",
    "parameters": [
        {
            "parameter_id": "learning_rate",
            "double_value_spec": {
                "min_value": 0.0001,
                "max_value": 0.1
            },
            "scale_type": "UNIT_LOG_SCALE"  # Log-uniform sampling
        },
        {
            "parameter_id": "dropout",
            "double_value_spec": {
                "min_value": 0.1,
                "max_value": 0.5
            }
        }
    ],
    "metrics": [
        {
            "metric_id": "accuracy",
            "goal": "MAXIMIZE"
        }
    ]
}
```

**Surprising effectiveness:** Research shows random search often matches Bayesian optimization on many tasks, especially when:
- Search space has low intrinsic dimensionality
- Many hyperparameters don't significantly affect performance
- Evaluation budget is limited

From [Hyperband research](https://homes.cs.washington.edu/~jamieson/hyperband.html) (Jamieson et al., 2016):
> "Random search appears to be soundly beat by state-of-the-art Bayesian optimization methods... However, running random search for twice as long yields superior results."

**Best for:**
- High-dimensional spaces (> 10 parameters)
- Baseline comparison
- When Bayesian overhead isn't justified

### 3. Bayesian Optimization (Default)

**How it works:** Builds a probabilistic model (Gaussian Process) of the objective function and uses acquisition functions to intelligently select next trials.

**Acquisition Functions:**
- **Expected Improvement (EI):** Balance exploration vs exploitation
- **Upper Confidence Bound (UCB):** Optimistic sampling with confidence intervals
- **Probability of Improvement (PI):** Conservative improvement seeking

```python
{
    "algorithm": "ALGORITHM_UNSPECIFIED",  # Default = Bayesian for continuous params
    "parameters": [
        {
            "parameter_id": "learning_rate",
            "double_value_spec": {
                "min_value": 1e-5,
                "max_value": 1e-1
            },
            "scale_type": "UNIT_LOG_SCALE"
        },
        {
            "parameter_id": "num_layers",
            "integer_value_spec": {
                "min_value": 2,
                "max_value": 10
            }
        }
    ],
    "metrics": [{"metric_id": "val_loss", "goal": "MINIMIZE"}]
}
```

**Gaussian Process Basics:**
- Maintains uncertainty estimates for unexplored regions
- Updates posterior distribution after each trial
- Computationally expensive for > 100 trials (O(n³) complexity)

From [Google Vizier paper](https://research.google.com/pubs/archive/46180.pdf) (Golovin et al., 2017):
> "Typically, the model for f is a Gaussian Process. Hence this literature goes under the name Bayesian Optimization."

**Best for:**
- Expensive evaluations (training takes hours/days)
- Low-to-medium dimensionality (< 20 parameters)
- Smooth objective functions
- When you want sample efficiency

### 4. Hyperband (Adaptive Early Stopping)

**Revolutionary insight:** Most hyperparameter configurations reveal their quality early in training. Allocate more resources to promising candidates.

**Algorithm Overview:**

```python
# Hyperband parameters
max_iter = 81        # Maximum epochs per trial
eta = 3              # Downsampling rate (default)
s_max = 4            # Number of brackets (log_eta(max_iter))
B = (s_max + 1) * max_iter  # Total budget per bracket

# Successive Halving rounds
for s in reversed(range(s_max + 1)):
    n = initial_configs(s)  # More configs for aggressive brackets
    r = initial_budget(s)   # Fewer epochs for aggressive brackets

    # Run n configs for r epochs, keep top 1/eta
    configs = sample_random_configs(n)
    for i in range(s + 1):
        n_i = n * eta**(-i)
        r_i = r * eta**(i)

        # Train and evaluate
        results = [train(config, epochs=r_i) for config in configs]

        # Keep top performers
        configs = top_k(results, k=int(n_i / eta))
```

**Bracket Strategy (max_iter=81, eta=3):**

| Bracket s | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
|-----------|---------|---------|---------|---------|---------|
| s=4 (aggressive) | 81 configs × 1 epoch | 27 × 3 | 9 × 9 | 3 × 27 | 1 × 81 |
| s=3 (balanced) | 27 configs × 3 epochs | 9 × 9 | 3 × 27 | 1 × 81 | - |
| s=2 (moderate) | 9 configs × 9 epochs | 3 × 27 | 1 × 81 | - | - |
| s=1 (conservative) | 6 configs × 27 epochs | 2 × 81 | - | - | - |
| s=0 (baseline) | 5 configs × 81 epochs | - | - | - | - |

**Key Properties:**
- **Adaptive:** No need to model learning curves (adapts to unknown convergence)
- **Parameter-free:** Only requires max_iter (no tuning of the tuner)
- **Provably near-optimal:** Within log factors of best possible early stopping

From [Hyperband demo](https://homes.cs.washington.edu/~jamieson/hyperband.html):
> "Hyperband is parameter-free, has strong theoretical guarantees for correctness and sample complexity. The rate of convergence does not need to be known in advance and our algorithm adapts to it."

**Vertex AI Implementation:**

```python
from google.cloud import aiplatform

# Hyperband study configuration
study = aiplatform.HyperparameterTuningJob(
    display_name="arr-coc-hyperband-tuning",
    custom_job=custom_training_job,
    metric_spec={
        "val_accuracy": "maximize"
    },
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-1, scale="log"),
        "token_budget": hpt.IntegerParameterSpec(min=64, max=400, scale="linear"),
    },
    search_algorithm="ALGORITHM_UNSPECIFIED",  # Uses Vizier's default
    max_trial_count=100,
    parallel_trial_count=10,
    # Early stopping configuration
    measurement_selection_type="BEST_MEASUREMENT",
    early_stopping_algorithm="EARLY_STOPPING_TYPE_UNSPECIFIED"  # Automated
)

study.run()
```

**Best for:**
- Deep learning (expensive training)
- Unknown convergence behavior
- Large hyperparameter spaces
- When parallel trials are available

## Trial Configuration

### Parameter Specifications

**Continuous Parameters (DoubleParameterSpec):**

```python
{
    "parameter_id": "learning_rate",
    "double_value_spec": {
        "min_value": 1e-5,
        "max_value": 1e-1
    },
    "scale_type": "UNIT_LOG_SCALE"  # Log-uniform sampling
}

# Scale types:
# - UNIT_LINEAR_SCALE: Uniform sampling [min, max]
# - UNIT_LOG_SCALE: Log-uniform exp(uniform(log(min), log(max)))
# - UNIT_REVERSE_LOG_SCALE: Reverse log for parameters near 1.0
```

**Integer Parameters (IntegerParameterSpec):**

```python
{
    "parameter_id": "num_layers",
    "integer_value_spec": {
        "min_value": 2,
        "max_value": 10
    },
    "scale_type": "UNIT_LINEAR_SCALE"
}
```

**Categorical Parameters (CategoricalValueSpec):**

```python
{
    "parameter_id": "optimizer",
    "categorical_value_spec": {
        "values": ["adam", "sgd", "rmsprop", "adamw"]
    }
}
```

**Discrete Parameters (DiscreteValueSpec):**

```python
{
    "parameter_id": "batch_size",
    "discrete_value_spec": {
        "values": [16, 32, 64, 128, 256, 512]
    }
}
```

### Conditional Parameters

Some parameters only make sense given values of other parameters:

```python
{
    "parameters": [
        {
            "parameter_id": "optimizer",
            "categorical_value_spec": {"values": ["adam", "sgd"]}
        },
        {
            "parameter_id": "momentum",
            "double_value_spec": {"min_value": 0.0, "max_value": 0.99},
            "conditional_parameter_specs": [
                {
                    "parent_categorical_values": {
                        "values": ["sgd"]  # Only tune momentum for SGD
                    }
                }
            ]
        }
    ]
}
```

### Metric Specification

**Single Objective:**

```python
{
    "metrics": [
        {
            "metric_id": "val_accuracy",
            "goal": "MAXIMIZE"  # or "MINIMIZE"
        }
    ]
}
```

**Reporting Metrics from Training:**

```python
# Inside training script
from google.cloud import aiplatform

# Report intermediate metrics
aiplatform.start_run("trial-123")
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_acc = validate()

    # Report to Vizier
    aiplatform.log_metrics({
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "epoch": epoch
    })

# Final metric for optimization
aiplatform.log_metrics({"val_accuracy": final_val_acc})
aiplatform.end_run()
```

## Early Stopping Strategies

Early stopping accelerates hyperparameter tuning by terminating unpromising trials before completion.

### 1. Median Stopping Rule

**How it works:** Stop trial if performance at step t is worse than the median of all trials at step t.

```python
# Median stopping configuration
{
    "measurement_selection_type": "LAST_MEASUREMENT",
    "early_stopping_algorithm": "AUTOMATED_EARLY_STOPPING",
    "automated_early_stopping_spec": {
        "type": "MEDIAN_AUTOMATED_STOPPING",
        "min_measurement_count": 5,  # Don't stop before 5 measurements
        "max_measurement_count": 50   # Evaluate stopping every 50 measurements
    }
}
```

**Example:**
- Trial A: epochs 1-10 → accuracy = [0.5, 0.6, 0.7, 0.75, 0.78, ...]
- Trial B: epochs 1-10 → accuracy = [0.3, 0.4, 0.45, 0.48, 0.50, ...]
- Median at epoch 5: 0.65

If Trial B's accuracy (0.50) < median (0.65), stop Trial B early.

**Pros:**
- Simple, interpretable
- No modeling required
- Works well when many trials run in parallel

**Cons:**
- Can be too aggressive (stops trials that recover later)
- Requires many parallel trials for good median estimate
- Not adaptive to convergence rate

### 2. Performance Curve Prediction

**How it works:** Fit learning curve models to predict final performance from early measurements.

**Common Models:**
- **Power law:** `accuracy(t) = a - b * t^(-c)`
- **Exponential:** `accuracy(t) = a - b * exp(-c * t)`
- **Log-linear:** `accuracy(t) = a + b * log(t)`

```python
# Predictive early stopping (Vertex AI automated)
{
    "early_stopping_algorithm": "AUTOMATED_EARLY_STOPPING",
    "automated_early_stopping_spec": {
        "type": "CURVE_FITTING",
        "min_measurement_count": 10,  # Need data to fit curves
        "prediction_horizon": 50       # Predict performance at epoch 50
    }
}
```

**Algorithm:**
1. Collect measurements for epochs 1-10
2. Fit power law curve to data
3. Predict performance at epoch 50
4. If predicted performance < current best × threshold, stop trial

**Pros:**
- More sophisticated than median stopping
- Can identify slow starters (high learning rate causing divergence early)
- Adapts to individual trial behavior

**Cons:**
- Requires assumption about learning curve shape
- Can fail if assumption is wrong (e.g., learning rate schedules)
- More computationally expensive

From [Vertex AI hyperparameter tuning overview](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview):
> "Hyperparameter tuning works by running multiple trials of your training application with values for your chosen hyperparameters, set within limits you specify. The Vertex AI Vizier service manages the trials and makes recommendations on hyperparameter assignments for subsequent trials."

### 3. Hyperband-Style Successive Halving

**How it works:** Allocate budget geometrically - run many configs for few epochs, keep top performers, run longer.

```python
# Successive Halving configuration
max_resource = 81  # Maximum epochs
eta = 3            # Keep top 1/3 each round

# Round 1: 81 configs × 1 epoch → keep 27 best
# Round 2: 27 configs × 3 epochs → keep 9 best
# Round 3: 9 configs × 9 epochs → keep 3 best
# Round 4: 3 configs × 27 epochs → keep 1 best
# Round 5: 1 config × 81 epochs → final winner
```

**Pros:**
- Provably near-optimal resource allocation
- No modeling assumptions
- Adapts to unknown convergence rates

**Cons:**
- Requires ability to resume training from checkpoints
- Fixed resource budget (can't add more trials mid-run)
- May be aggressive for slow-starting configurations

## Multi-Objective Optimization

Real-world ML systems often optimize multiple competing objectives simultaneously.

### Common Trade-offs

1. **Accuracy vs. Latency**
   - Higher accuracy often requires larger models (slower inference)
   - Use case: Mobile deployment with strict latency budgets

2. **Accuracy vs. Model Size**
   - Larger models → better accuracy but more storage/memory
   - Use case: Edge devices with memory constraints

3. **Accuracy vs. Training Cost**
   - More training epochs → better accuracy but higher cost
   - Use case: Budget-constrained research

### Pareto Frontier

Multi-objective optimization finds the **Pareto frontier**: set of solutions where improving one objective requires degrading another.

```
Accuracy
   ▲
   │         ● ← Pareto optimal (90% acc, 50ms latency)
   │       ●   ● ← Pareto optimal points
   │     ●
   │   ●           ○ ← Dominated (worse on both)
   │ ●
   └─────────────────► Latency (lower is better)
```

### Vertex AI Multi-Objective Configuration

```python
{
    "metrics": [
        {
            "metric_id": "val_accuracy",
            "goal": "MAXIMIZE"
        },
        {
            "metric_id": "inference_latency_ms",
            "goal": "MINIMIZE"
        }
    ],
    "parameters": [
        {
            "parameter_id": "num_layers",
            "integer_value_spec": {"min_value": 2, "max_value": 12}
        },
        {
            "parameter_id": "hidden_dim",
            "discrete_value_spec": {"values": [128, 256, 512, 1024]}
        }
    ]
}
```

**Vizier's Approach:**
- Uses **scalarization** to convert multi-objective to single objective
- Explores different weight combinations: `w1 * accuracy - w2 * latency`
- Returns Pareto frontier for human selection

### Selecting from Pareto Frontier

After tuning, Vizier returns multiple Pareto-optimal configurations:

```python
# Retrieve Pareto frontier
study = aiplatform.HyperparameterTuningJob.get("study-id")
trials = study.trials

# Filter to Pareto optimal
pareto_trials = [t for t in trials if t.pareto_optimal]

# Example results:
# Trial 1: 92% acc, 120ms latency, 50M params
# Trial 2: 90% acc, 80ms latency, 30M params  ← Choose this for production
# Trial 3: 88% acc, 50ms latency, 15M params  ← Choose for mobile
```

**Selection Criteria:**
- Production server: Optimize for accuracy (Trial 1)
- Mobile deployment: Optimize for latency (Trial 3)
- Balanced use case: Mid-point (Trial 2)

## Cost Optimization

Hyperparameter tuning can be expensive. Strategies to reduce cost:

### 1. Preemptible Trials

Use preemptible VMs for trials - 80% cost reduction with restart overhead.

```python
custom_job = aiplatform.CustomJob(
    display_name="training-job",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1
            },
            "replica_count": 1,
            "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
            # Enable preemptible instances
            "scheduling": {
                "restart_job_on_worker_restart": True,  # Auto-restart if preempted
                "timeout": "3600s"
            }
        }
    ],
    base_output_dir="gs://bucket/tuning-output/"
)

tuning_job = aiplatform.HyperparameterTuningJob(
    display_name="cost-optimized-tuning",
    custom_job=custom_job,
    max_trial_count=50,
    parallel_trial_count=5  # Limit parallel to manage preemption restarts
)
```

**Checkpointing for Preemptible Trials:**

```python
# Inside training script
import os
import tensorflow as tf

checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR", "./checkpoints")

# Save checkpoints frequently
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{checkpoint_dir}/model_{{epoch:02d}}.h5",
    save_freq="epoch"
)

# Resume from latest checkpoint if exists
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)
    print(f"Resumed from checkpoint: {latest}")

model.fit(train_data, epochs=100, callbacks=[checkpoint_callback])
```

### 2. Parallel Execution Limits

More parallel trials = faster results but higher instantaneous cost.

**Trade-off Analysis:**

| Parallel Trials | Total Cost | Wall Clock Time | Cost per Hour |
|-----------------|------------|-----------------|---------------|
| 1 trial | $100 | 50 hours | $2/hour |
| 5 trials | $100 | 10 hours | $10/hour |
| 10 trials | $100 | 5 hours | $20/hour |

```python
# Conservative: Low parallel count
tuning_job = aiplatform.HyperparameterTuningJob(
    max_trial_count=50,
    parallel_trial_count=2,  # 2 trials at a time (low hourly cost)
    # Wall clock: ~25x longer but spreads cost over time
)

# Aggressive: High parallel count
tuning_job = aiplatform.HyperparameterTuningJob(
    max_trial_count=50,
    parallel_trial_count=20,  # 20 trials at a time (high hourly cost)
    # Wall clock: ~2.5x faster but concentrated cost burst
)
```

**Bayesian optimization caveat:** High parallelism reduces Bayesian advantage (later trials can't learn from earlier trials running in parallel).

### 3. Warm Starting

Reuse knowledge from previous tuning jobs:

```python
# First tuning job (coarse search)
initial_study = aiplatform.HyperparameterTuningJob(
    display_name="coarse-search",
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-1, scale="log"),
        "batch_size": hpt.IntegerParameterSpec(min=16, max=256, scale="linear")
    },
    max_trial_count=20,
    parallel_trial_count=5
)
initial_study.run()

# Get best trial
best_trial = max(initial_study.trials, key=lambda t: t.final_measurement.metrics[0].value)

# Second tuning job (fine search around best)
refined_study = aiplatform.HyperparameterTuningJob(
    display_name="fine-search",
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(
            min=best_trial.parameters["learning_rate"] * 0.5,
            max=best_trial.parameters["learning_rate"] * 2.0,
            scale="log"
        ),
        "batch_size": hpt.IntegerParameterSpec(
            min=max(16, best_trial.parameters["batch_size"] - 32),
            max=min(256, best_trial.parameters["batch_size"] + 32),
            scale="linear"
        )
    },
    max_trial_count=10,  # Fewer trials for refinement
    parallel_trial_count=3
)
```

### 4. Subset Data for Early Trials

Train on fraction of data for initial exploration, full data for final trials:

```python
# Inside training script
import os

# Get trial ID from environment
trial_id = int(os.environ.get("CLOUD_ML_TRIAL_ID", "0"))

# Use 20% data for trials 0-29, 100% for trials 30+
if trial_id < 30:
    train_data = full_train_data.take(int(len(full_train_data) * 0.2))
    print("Using 20% data for exploration")
else:
    train_data = full_train_data
    print("Using 100% data for exploitation")
```

**Cost Savings:**
- 20% data → ~5x faster training
- First 30 trials cost: 30 × (training_cost / 5) = 6 × training_cost
- Last 20 trials cost: 20 × training_cost
- Total: 26 × training_cost (vs 50 × training_cost for all full data)

## ARR-COC-0-1 Hyperparameter Search

Practical hyperparameter tuning for the arr-coc-0-1 vision-language model.

### Search Space Design

**Core Hyperparameters:**

```python
# arr-coc-0-1 specific parameters
{
    "parameters": [
        # Token allocation
        {
            "parameter_id": "token_budget",
            "integer_value_spec": {"min_value": 64, "max_value": 400},
            "scale_type": "UNIT_LINEAR_SCALE"
        },
        {
            "parameter_id": "min_lod",
            "integer_value_spec": {"min_value": 64, "max_value": 128}
        },
        {
            "parameter_id": "max_lod",
            "integer_value_spec": {"min_value": 256, "max_value": 400}
        },

        # Relevance realization weights
        {
            "parameter_id": "propositional_weight",
            "double_value_spec": {"min_value": 0.1, "max_value": 1.0}
        },
        {
            "parameter_id": "perspectival_weight",
            "double_value_spec": {"min_value": 0.1, "max_value": 1.0}
        },
        {
            "parameter_id": "participatory_weight",
            "double_value_spec": {"min_value": 0.1, "max_value": 1.0}
        },

        # Opponent processing tensions
        {
            "parameter_id": "compression_vs_particularize",
            "double_value_spec": {"min_value": 0.3, "max_value": 0.7}
        },
        {
            "parameter_id": "exploit_vs_explore",
            "double_value_spec": {"min_value": 0.4, "max_value": 0.6}
        },

        # Training
        {
            "parameter_id": "learning_rate",
            "double_value_spec": {"min_value": 1e-5, "max_value": 1e-3},
            "scale_type": "UNIT_LOG_SCALE"
        },
        {
            "parameter_id": "adapter_learning_rate",
            "double_value_spec": {"min_value": 1e-4, "max_value": 1e-2},
            "scale_type": "UNIT_LOG_SCALE"
        },
        {
            "parameter_id": "warmup_steps",
            "integer_value_spec": {"min_value": 100, "max_value": 1000}
        }
    ]
}
```

### Multi-Objective: Accuracy + Efficiency

```python
{
    "metrics": [
        {
            "metric_id": "vqa_accuracy",
            "goal": "MAXIMIZE"
        },
        {
            "metric_id": "avg_tokens_per_image",
            "goal": "MINIMIZE"  # Efficiency: fewer tokens is better
        },
        {
            "metric_id": "inference_time_ms",
            "goal": "MINIMIZE"
        }
    ]
}
```

**Objective:** Find configurations on Pareto frontier:
- High accuracy + many tokens (for offline analysis)
- Medium accuracy + medium tokens (for production API)
- Lower accuracy + few tokens (for mobile/edge deployment)

### Complete Tuning Job

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Training container
custom_job = aiplatform.CustomJob(
    display_name="arr-coc-0-1-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-16",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 2
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-west2-docker.pkg.dev/arr-coc-0-1/training/arr-coc:latest",
            "command": ["python", "training/train.py"],
            "args": [
                "--data_path=gs://arr-coc-data/vqa/",
                "--output_dir=gs://arr-coc-models/tuning/",
                "--epochs=10"
            ]
        },
        "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200}
    }]
)

# Hyperparameter tuning job
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name="arr-coc-vizier-tuning",
    custom_job=custom_job,
    metric_spec={
        "vqa_accuracy": "maximize",
        "avg_tokens_per_image": "minimize",
        "inference_time_ms": "minimize"
    },
    parameter_spec={
        # Token allocation
        "token_budget": hpt.IntegerParameterSpec(min=64, max=400, scale="linear"),
        "min_lod": hpt.IntegerParameterSpec(min=64, max=128, scale="linear"),
        "max_lod": hpt.IntegerParameterSpec(min=256, max=400, scale="linear"),

        # Relevance weights (will be normalized in code)
        "propositional_weight": hpt.DoubleParameterSpec(min=0.1, max=1.0, scale="linear"),
        "perspectival_weight": hpt.DoubleParameterSpec(min=0.1, max=1.0, scale="linear"),
        "participatory_weight": hpt.DoubleParameterSpec(min=0.1, max=1.0, scale="linear"),

        # Opponent processing
        "compression_particularize_balance": hpt.DoubleParameterSpec(min=0.3, max=0.7, scale="linear"),
        "exploit_explore_balance": hpt.DoubleParameterSpec(min=0.4, max=0.6, scale="linear"),

        # Learning rates
        "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
        "adapter_lr": hpt.DoubleParameterSpec(min=1e-4, max=1e-2, scale="log"),
        "warmup_steps": hpt.IntegerParameterSpec(min=100, max=1000, scale="linear")
    },
    search_algorithm="ALGORITHM_UNSPECIFIED",  # Bayesian for continuous params
    max_trial_count=100,
    parallel_trial_count=8,
    max_failed_trial_count=10,

    # Early stopping
    measurement_selection_type="BEST_MEASUREMENT",
    early_stopping_algorithm="AUTOMATED_EARLY_STOPPING"
)

# Run tuning
tuning_job.run(
    service_account="arr-coc-training@project.iam.gserviceaccount.com",
    network="projects/123/global/networks/arr-coc-vpc",
    sync=True
)

# Retrieve Pareto frontier
pareto_trials = [t for t in tuning_job.trials if hasattr(t, 'pareto_optimal') and t.pareto_optimal]

print(f"Found {len(pareto_trials)} Pareto-optimal configurations:")
for i, trial in enumerate(pareto_trials):
    metrics = {m.metric_id: m.value for m in trial.final_measurement.metrics}
    print(f"\nConfig {i+1}:")
    print(f"  VQA Accuracy: {metrics['vqa_accuracy']:.2%}")
    print(f"  Avg Tokens: {metrics['avg_tokens_per_image']:.1f}")
    print(f"  Latency: {metrics['inference_time_ms']:.0f}ms")
    print(f"  Hyperparameters: {trial.parameters}")
```

### Training Script Integration

```python
# training/train.py
import os
import argparse
from google.cloud import aiplatform

def parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters from Vizier (passed as args)
    parser.add_argument("--token_budget", type=int, default=200)
    parser.add_argument("--min_lod", type=int, default=64)
    parser.add_argument("--max_lod", type=int, default=400)
    parser.add_argument("--propositional_weight", type=float, default=0.33)
    parser.add_argument("--perspectival_weight", type=float, default=0.33)
    parser.add_argument("--participatory_weight", type=float, default=0.34)
    parser.add_argument("--compression_particularize_balance", type=float, default=0.5)
    parser.add_argument("--exploit_explore_balance", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Fixed arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Vertex AI for metric reporting
    aiplatform.init(
        project=os.environ.get("CLOUD_ML_PROJECT_ID"),
        location=os.environ.get("CLOUD_ML_REGION", "us-west2")
    )

    # Start experiment run
    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "local")
    aiplatform.start_run(f"trial-{trial_id}")

    # Normalize relevance weights
    total_weight = (args.propositional_weight +
                   args.perspectival_weight +
                   args.participatory_weight)
    relevance_weights = {
        "propositional": args.propositional_weight / total_weight,
        "perspectival": args.perspectival_weight / total_weight,
        "participatory": args.participatory_weight / total_weight
    }

    # Build model with hyperparameters
    model = build_arr_coc_model(
        token_budget=args.token_budget,
        min_lod=args.min_lod,
        max_lod=args.max_lod,
        relevance_weights=relevance_weights,
        opponent_balances={
            "compression_particularize": args.compression_particularize_balance,
            "exploit_explore": args.exploit_explore_balance
        }
    )

    # Training loop
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, args.learning_rate)
        val_metrics = validate(model, val_loader)

        # Report to Vizier
        aiplatform.log_metrics({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "vqa_accuracy": val_metrics["accuracy"],
            "avg_tokens_per_image": val_metrics["avg_tokens"],
            "inference_time_ms": val_metrics["latency_ms"]
        })

        print(f"Epoch {epoch}: VQA Acc={val_metrics['accuracy']:.2%}, "
              f"Tokens={val_metrics['avg_tokens']:.1f}, "
              f"Latency={val_metrics['latency_ms']:.0f}ms")

    # Final metrics (used by Vizier for optimization)
    final_metrics = validate(model, test_loader)
    aiplatform.log_metrics({
        "vqa_accuracy": final_metrics["accuracy"],
        "avg_tokens_per_image": final_metrics["avg_tokens"],
        "inference_time_ms": final_metrics["latency_ms"]
    })

    # Save model
    model.save(f"{args.output_dir}/trial-{trial_id}/")
    aiplatform.end_run()

if __name__ == "__main__":
    main()
```

### Interpreting Results

After tuning completes, analyze the Pareto frontier:

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract metrics from Pareto-optimal trials
pareto_trials = [t for t in tuning_job.trials if t.pareto_optimal]

accuracies = []
token_counts = []
latencies = []
configs = []

for trial in pareto_trials:
    metrics = {m.metric_id: m.value for m in trial.final_measurement.metrics}
    accuracies.append(metrics["vqa_accuracy"])
    token_counts.append(metrics["avg_tokens_per_image"])
    latencies.append(metrics["inference_time_ms"])
    configs.append(trial.parameters)

# Visualize Pareto frontier
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy vs Tokens
ax1.scatter(token_counts, accuracies, s=100, alpha=0.6)
ax1.set_xlabel("Avg Tokens per Image")
ax1.set_ylabel("VQA Accuracy")
ax1.set_title("Accuracy-Efficiency Pareto Frontier")
for i, (x, y) in enumerate(zip(token_counts, accuracies)):
    ax1.annotate(f"Config {i+1}", (x, y), textcoords="offset points", xytext=(5,5))

# Accuracy vs Latency
ax2.scatter(latencies, accuracies, s=100, alpha=0.6, color='orange')
ax2.set_xlabel("Inference Latency (ms)")
ax2.set_ylabel("VQA Accuracy")
ax2.set_title("Accuracy-Latency Pareto Frontier")

plt.tight_layout()
plt.savefig("pareto_frontier.png")

# Select deployment configs
production_config = configs[np.argmax(accuracies)]  # Best accuracy
mobile_config = configs[np.argmin(latencies)]        # Lowest latency
balanced_config = configs[len(configs) // 2]         # Middle ground

print("Deployment Recommendations:")
print(f"\n1. Production (maximize accuracy):")
print(f"   Accuracy: {max(accuracies):.2%}")
print(f"   Config: {production_config}")

print(f"\n2. Mobile (minimize latency):")
print(f"   Latency: {min(latencies):.0f}ms")
print(f"   Config: {mobile_config}")

print(f"\n3. Balanced:")
print(f"   Accuracy: {accuracies[len(configs)//2]:.2%}")
print(f"   Tokens: {token_counts[len(configs)//2]:.1f}")
print(f"   Config: {balanced_config}")
```

## Sources

**Google Cloud Documentation:**
- [Vertex AI Vizier overview](https://cloud.google.com/vertex-ai/docs/vizier/overview) - Vizier service overview and algorithms (accessed 2025-11-16)
- [Overview of hyperparameter tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview) - Hyperparameter tuning integration with Custom Training Jobs (accessed 2025-11-16)

**Research Papers:**
- [Google Vizier: A Service for Black-Box Optimization](https://research.google.com/pubs/archive/46180.pdf) - Golovin et al., 2017 - Foundational Vizier paper describing Bayesian optimization approach
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) - Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, 2016 - Hyperband algorithm and theoretical guarantees

**Web Resources:**
- [Hyperband demo by kgjamieson](https://homes.cs.washington.edu/~jamieson/hyperband.html) - Kevin Jamieson's interactive Hyperband explanation (accessed 2025-11-16)
- [Neural Architecture Search: Insights from 1000 Papers](https://arxiv.org/abs/2301.08727) - White et al., 2023 - Survey of NAS techniques and empirical comparisons

**Additional References:**
- [How to use GCP Vertex AI Vizier Hyperparameter Optimization](https://medium.com/google-cloud/how-to-use-gcp-vertex-ai-vizier-hyperparameter-optimization-78a8ed7b6253) - Nicholas Ord, Medium, 2024 - Practical tutorial (accessed 2025-11-16)
- [Neural Architecture Search: A Survey](https://www.jmlr.org/papers/volume20/18-598/18-598.pdf) - Elsken, Metzen, Hutter, 2019 - Comprehensive NAS survey covering search spaces, optimization strategies, and performance estimation
