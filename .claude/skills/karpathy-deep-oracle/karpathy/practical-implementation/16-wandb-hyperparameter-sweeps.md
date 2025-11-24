# W&B Hyperparameter Sweeps

**Automate hyperparameter optimization with Weights & Biases Sweeps**

## Overview

Finding optimal hyperparameters is tedious manual work. W&B Sweeps automate this process by systematically exploring hyperparameter combinations and identifying which settings work best for your model.

**Core concept**: Define what to search → Initialize controller → Run agents → Analyze results

**Three search strategies**:
- **Grid**: Exhaustive search through all combinations
- **Random**: Random sampling from distributions
- **Bayesian**: Intelligent search using probabilistic models

From [W&B Sweeps Documentation](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) (accessed 2025-01-31):
> A W&B Sweep combines a strategy for exploring hyperparameter values with the code that evaluates them. The strategy can be as simple as trying every option or as complex as Bayesian Optimization and Hyperband.

## Sweep Basics

### YAML Configuration Structure

Sweeps are defined with YAML configs or Python dictionaries. YAML is preferred for CLI-based workflows.

**Basic sweep config**:

```yaml
program: train.py
name: my-sweep
method: bayes
metric:
  goal: minimize
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  epochs:
    values: [5, 10, 15]
  optimizer:
    values: ["adam", "sgd"]
```

From [W&B Sweep Config Documentation](https://docs.wandb.ai/models/sweeps/sweep-config-keys) (accessed 2025-01-31):
> Use top-level keys within your sweep configuration to define qualities of your sweep search such as the parameters to search through (`parameter` key), the methodology to search the parameter space (`method` key), and more.

### Python Dictionary Alternative

For notebook-based workflows:

```python
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.1
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}
```

## Search Methods Comparison

### Grid Search

**When to use**: Small, discrete hyperparameter spaces where you want to try every combination.

**Pros**:
- Guaranteed to find best combination in search space
- Deterministic and reproducible
- Simple to understand

**Cons**:
- Computationally expensive (exponential growth)
- Runs forever with continuous parameters
- Wastes resources on bad regions

```yaml
method: grid
parameters:
  learning_rate:
    values: [0.001, 0.01, 0.1]
  batch_size:
    values: [32, 64, 128]
  # Total runs: 3 × 3 = 9
```

### Random Search

**When to use**: Large search spaces, continuous parameters, exploratory phase.

**Pros**:
- Better than grid for high-dimensional spaces
- Can run indefinitely (stop when satisfied)
- Good for initial exploration

**Cons**:
- No guarantee of finding optimum
- Uninformed decisions
- May waste time in bad regions

```yaml
method: random
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  dropout:
    distribution: uniform
    min: 0.3
    max: 0.5
```

From [W&B Sweeps Tutorial](https://docs.wandb.ai/models/tutorials/sweeps) (accessed 2025-01-31):
> For a `random` search, all the `values` of a parameter are equally likely to be chosen on a given run. Alternatively, you can specify a named `distribution`, plus its parameters, like the mean `mu` and standard deviation `sigma` of a `normal` distribution.

### Bayesian Search

**When to use**: Expensive training runs where you want intelligent exploration.

**Pros**:
- Makes informed decisions using past results
- Efficient for expensive evaluations
- Balances exploration vs exploitation

**Cons**:
- Scales poorly to many parameters (>20)
- Requires more setup
- Can get stuck in local optima

```yaml
method: bayes
metric:
  goal: minimize
  name: val_loss
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  hidden_size:
    values: [128, 256, 512]
```

**How Bayesian search works**: Uses a probabilistic model (Gaussian Process) to predict which hyperparameters will improve your metric, then tests the most promising combinations.

## Parameter Configuration

### Fixed Values

Lock parameters you don't want to vary:

```yaml
parameters:
  epochs:
    value: 10  # Always use 10 epochs
  model_type:
    value: "resnet50"
```

### Discrete Choices

List specific values to try:

```yaml
parameters:
  optimizer:
    values: ["adam", "sgd", "rmsprop"]
  batch_size:
    values: [16, 32, 64, 128]
```

### Continuous Distributions

From [W&B Sweep Config Documentation](https://docs.wandb.ai/models/sweeps/sweep-config-keys) (accessed 2025-01-31), supported distributions:

**Uniform distributions**:
```yaml
learning_rate:
  distribution: uniform
  min: 0.0
  max: 0.1
```

**Log-uniform (for learning rates)**:
```yaml
learning_rate:
  distribution: log_uniform_values
  min: 0.0001
  max: 0.1
```

**Quantized distributions**:
```yaml
batch_size:
  distribution: q_log_uniform_values
  min: 32
  max: 256
  q: 8  # Rounds to multiples of 8
```

**Normal distributions**:
```yaml
dropout:
  distribution: normal
  mu: 0.5
  sigma: 0.1
```

### Nested Parameters

Support for hierarchical configs:

```yaml
parameters:
  model:
    parameters:
      hidden_size:
        values: [128, 256]
      num_layers:
        values: [2, 4, 6]
  optimizer:
    parameters:
      learning_rate:
        min: 0.0001
        max: 0.1
```

**Important**: Nested parameters defined in sweep config overwrite values in `wandb.init(config={...})`.

## Running Sweeps

### Step 1: Define Sweep Config

Create `sweep.yaml`:

```yaml
program: train.py
method: bayes
metric:
  goal: minimize
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
```

### Step 2: Initialize Sweep

**CLI approach**:
```bash
wandb sweep sweep.yaml
# Returns: Created sweep with ID: abc123xyz
```

**Python approach**:
```python
import wandb

sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="my-project"
)
```

### Step 3: Modify Training Code

Key integration points:

```python
import wandb

def train():
    # Initialize run - sweep controller provides config
    with wandb.init() as run:
        config = run.config  # Get sweep hyperparameters

        # Use config values
        model = build_model(config.hidden_size)
        optimizer = build_optimizer(
            model,
            config.optimizer,
            config.learning_rate
        )

        # Training loop
        for epoch in range(config.epochs):
            loss = train_epoch(model, optimizer)

            # Log metric specified in sweep config
            run.log({
                "validation_loss": loss,
                "epoch": epoch
            })
```

From [W&B Sweeps Tutorial](https://docs.wandb.ai/models/tutorials/sweeps) (accessed 2025-01-31):
> The key to integrating W&B Sweeps into your training code is to ensure that, for each training experiment, that your training logic can access the hyperparameter values you defined in your sweep configuration.

### Step 4: Start Agent(s)

**Single agent**:
```bash
wandb agent <sweep_id>
```

**Limited runs**:
```bash
wandb agent <sweep_id> --count 10
```

**Python**:
```python
wandb.agent(sweep_id, function=train, count=5)
```

**Parallel agents** (same sweep_id, different machines):
```bash
# Machine 1
wandb agent <sweep_id>

# Machine 2
wandb agent <sweep_id>

# Machine 3
wandb agent <sweep_id>
```

The sweep controller coordinates all agents automatically.

## Early Termination

Stop poorly performing runs early to save compute.

### Hyperband Stopping

From [W&B Sweep Config Documentation](https://docs.wandb.ai/models/sweeps/sweep-config-keys) (accessed 2025-01-31):
> Hyperband hyperparameter optimization evaluates if a program should stop or if it should continue at one or more pre-set iteration counts, called _brackets_.

**With max iterations**:
```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
  eta: 3
```

Brackets: `[3, 9, 27, 81]` where `3 = max_iter / (eta^s)`

**With min iterations**:
```yaml
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 3
```

**How it works**:
1. Run reaches bracket checkpoint (e.g., epoch 9)
2. Compare metric to all other runs at same checkpoint
3. Terminate if metric is in bottom percentage
4. Continue if metric is promising

**Parameters**:
- `eta`: Bracket multiplier (default: 3)
- `s`: Number of brackets
- `max_iter` or `min_iter`: Defines bracket schedule
- `strict`: More aggressive pruning (default: false)

## Advanced Configuration

### Command Macros

Customize how sweep runs your training script:

```yaml
program: run.py
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--batch_size=${batch_size}"
  - "--optimizer=${optimizer}"
  - "--wandb_project=my-project"
```

From [W&B Sweep Config Documentation](https://docs.wandb.ai/models/sweeps/sweep-config-keys) (accessed 2025-01-31), supported macros:

- `${env}`: `/usr/bin/env` on Unix
- `${interpreter}`: Expands to `python`
- `${program}`: Script filename
- `${args}`: All params as `--param=value`
- `${args_no_boolean_flags}`: Booleans as flags
- `${args_json}`: Parameters as JSON
- `${args_json_file}`: Path to JSON file with params

### Boolean Arguments

From [W&B Sweep Config Documentation](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) (accessed 2025-01-31):

The `argparse` module doesn't support booleans by default. Use custom function:

```python
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', '1')

parser.add_argument('--use_dropout', type=str2bool)
```

Or use action flags:
```python
parser.add_argument('--use_dropout', action='store_true')
```

### Run Caps

Limit total runs in sweep:

```yaml
run_cap: 100  # Stop after 100 runs
```

Useful for budgeting compute or when you've found good results.

## Analyzing Results

### Parallel Coordinates Plot

From [W&B Sweeps Tutorial](https://docs.wandb.ai/models/tutorials/sweeps) (accessed 2025-01-31):
> This plot maps hyperparameter values to model metrics. It's useful for honing in on combinations of hyperparameters that led to the best model performance.

**How to read**:
- Each line = one run
- Each vertical axis = one hyperparameter or metric
- Color = metric value (e.g., loss)
- Select regions to filter runs

**Use case**: Identify parameter ranges that consistently produce good results.

### Hyperparameter Importance Plot

From [W&B Sweeps Tutorial](https://docs.wandb.ai/models/tutorials/sweeps) (accessed 2025-01-31):
> The hyperparameter importance plot surfaces which hyperparameters were the best predictors of your metrics. We report feature importance (from a random forest model) and correlation (implicitly a linear model).

**Two metrics**:
1. **Feature importance**: Random forest model (nonlinear relationships)
2. **Correlation**: Linear model (linear relationships)

**Use case**: Focus future sweeps on parameters that matter most.

## Practical Patterns

### Quick Exploration Then Refinement

**Phase 1**: Random search, wide ranges
```yaml
method: random
parameters:
  learning_rate:
    min: 0.00001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128, 256]
```

**Phase 2**: Bayesian search, narrow ranges based on Phase 1
```yaml
method: bayes
parameters:
  learning_rate:
    min: 0.001  # Narrowed from Phase 1
    max: 0.01
  batch_size:
    values: [32, 64]  # Best from Phase 1
```

### Overfit One Batch + Sweep

Validate your code works before sweeping:

```python
# Sanity check: overfit 1 batch
config = {'learning_rate': 0.001, 'batch_size': 32}
train_one_batch(config)  # Should reach ~0 loss

# Once validated, run sweep
wandb.agent(sweep_id, function=train)
```

### Multi-Stage Sweeps

**Stage 1**: Optimizer and learning rate
**Stage 2**: Architecture (given best optimizer/LR)
**Stage 3**: Regularization (given best architecture)

Avoids combinatorial explosion while still exploring thoroughly.

## Common Issues

### Sweep Runs Forever

**Problem**: Random/Bayesian sweeps don't auto-stop.

**Solutions**:
1. Use `--count` flag: `wandb agent sweep_id --count 20`
2. Set `run_cap` in config: `run_cap: 100`
3. Stop manually from W&B UI or `wandb sweep --stop <sweep_id>`

### Parameters Not Being Used

**Problem**: Training code doesn't see sweep parameters.

**Checklist**:
```python
# ✓ Initialize run inside function
with wandb.init() as run:
    config = run.config  # Get sweep params

    # ✓ Use config values
    model = Model(config.hidden_size)

    # ✗ Don't hardcode
    # model = Model(128)  # Wrong!
```

### Grid Search Takes Forever

**Problem**: Grid search with continuous parameters or too many combinations.

**Solutions**:
1. Switch to `method: random` or `method: bayes`
2. Reduce parameter values: `values: [64, 128, 256]` not `[32, 48, 64, 80, 96, ...]`
3. Use early termination

## Integration with HuggingFace Trainer

Sweeps work seamlessly with HuggingFace:

```python
def train():
    with wandb.init() as run:
        config = run.config

        training_args = TrainingArguments(
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            report_to="wandb",  # Auto-logs to W&B
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

# Sweep config
sweep_config = {
    'method': 'bayes',
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-3},
        'batch_size': {'values': [8, 16, 32]},
        'epochs': {'values': [3, 5, 10]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="hf-sweep")
wandb.agent(sweep_id, function=train, count=10)
```

## Best Practices

### 1. Start Small

Run 3-5 manual experiments first to validate code and get baseline.

### 2. Log the Metric You Care About

```yaml
metric:
  goal: minimize
  name: validation_loss  # Match exactly what you log
```

```python
run.log({"validation_loss": val_loss})  # Name must match
```

### 3. Use Scientific Notation

```yaml
learning_rate:
  min: !!float 1e-5  # YAML needs !!float for scientific notation
  max: !!float 1e-2
```

### 4. Parallelize When Possible

```bash
# Launch 4 agents on different GPUs
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_id &
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_id &
CUDA_VISIBLE_DEVICES=2 wandb agent sweep_id &
CUDA_VISIBLE_DEVICES=3 wandb agent sweep_id &
```

### 5. Monitor and Adjust

Don't set-and-forget. Check results after 10-20 runs:
- Are parameters being explored properly?
- Is the metric improving?
- Should you narrow the search space?

## Sources

**W&B Official Documentation**:
- [Define Sweep Configuration](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) (accessed 2025-01-31)
- [Sweep Configuration Options](https://docs.wandb.ai/models/sweeps/sweep-config-keys) (accessed 2025-01-31)
- [Tune Hyperparameters Tutorial](https://docs.wandb.ai/models/tutorials/sweeps) (accessed 2025-01-31)

**Web Research**:
- Google Search: "wandb sweeps tutorial 2024"
- Google Search: "hyperparameter optimization wandb bayesian"
- Google Search: "wandb sweep yaml configuration"
