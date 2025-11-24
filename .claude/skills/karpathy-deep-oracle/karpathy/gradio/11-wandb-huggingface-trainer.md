# HuggingFace Trainer + Weights & Biases Integration

## Overview

The HuggingFace Transformers `Trainer` class provides seamless integration with Weights & Biases (W&B) through the built-in `WandbCallback`. This integration enables automatic logging of training metrics, model checkpoints, and custom visualizations with minimal code changes.

**Key benefit**: Turn on enterprise-grade experiment tracking by adding a single parameter to `TrainingArguments`.

From [HuggingFace Transformers - W&B Documentation](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- The W&B integration adds "rich, flexible experiment tracking and model versioning to interactive centralized dashboards without compromising ease of use"
- Built-in support via `WandbCallback` in the Transformers library

---

## Basic Integration (~100 lines)

### Minimal Setup (3 Lines)

The absolute minimum to enable W&B logging:

```python
from transformers import TrainingArguments, Trainer

# Set project name via environment variable
os.environ["WANDB_PROJECT"] = "my-amazing-project"

# Enable W&B in training arguments
args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # This is the key line!
    # ... other arguments
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- `report_to="wandb"` is "the most important step" for enabling W&B logging
- By default, `report_to` is set to `"all"`, which includes W&B if installed

### Environment Variables for Basic Configuration

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):

| Variable | Purpose | Default |
|----------|---------|---------|
| `WANDB_PROJECT` | Project name for organizing runs | `"huggingface"` |
| `WANDB_DISABLED` | Disable W&B entirely | `False` |
| `WANDB_QUIET` | Limit output to critical statements | `False` |
| `WANDB_SILENT` | Silence all W&B output | `False` |

```python
# Example: Configure via environment
os.environ["WANDB_PROJECT"] = "sentiment-analysis"
os.environ["WANDB_QUIET"] = "true"  # Less verbose output
```

### What Gets Logged Automatically

The `WandbCallback` automatically logs:
- **Training loss** (every `logging_steps`)
- **Evaluation metrics** (every `eval_steps`)
- **Learning rate** (tracked throughout training)
- **Training progress** (epoch, global step, samples/second)
- **System metrics** (GPU utilization if available)

From [HuggingFace Trainer Callbacks Documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) (accessed 2025-01-31):
- The `logging_steps` argument in `TrainingArguments` controls how often metrics are pushed to W&B

### Naming Your Run

```python
args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    run_name="bert-base-high-lr",  # Descriptive run name
    logging_steps=100,  # Log every 100 steps
)
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- Use `run_name` argument to give your W&B run a descriptive name
- Without `run_name`, W&B auto-generates a random name

---

## Custom Callbacks (~120 lines)

### Understanding WandbCallback Customization

The `Trainer` uses `WandbCallback` internally. You can customize logging by:
1. Subclassing `WandbCallback`
2. Adding custom functionality using Trainer callback methods
3. Registering your custom callback with the Trainer

From [HuggingFace Callbacks Documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) (accessed 2025-01-31):
- "If you need to customize your Hugging Face logging you can modify this callback by subclassing `WandbCallback` and adding additional functionality"

### Custom Callback Pattern

```python
from transformers.integrations import WandbCallback
import wandb

class MyCustomWandbCallback(WandbCallback):
    """Custom callback extending WandbCallback"""

    def __init__(self, trainer, custom_param):
        super().__init__()
        self.trainer = trainer
        self.custom_param = custom_param

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Call parent logging first
        super().on_log(args, state, control, logs=logs, **kwargs)

        # Add custom logging
        if logs is not None:
            custom_metrics = self._compute_custom_metrics(logs)
            self._wandb.log(custom_metrics)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        # Custom evaluation logging here
        pass

    def _compute_custom_metrics(self, logs):
        # Your custom metric computation
        return {"custom_metric": 42}

# Register the custom callback
trainer = Trainer(model=model, args=args)
custom_callback = MyCustomWandbCallback(trainer, custom_param="value")
trainer.add_callback(custom_callback)
```

From [HuggingFace Callbacks Documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) (accessed 2025-01-31):
- Custom callbacks must be added AFTER the Trainer is instantiated
- Cannot pass custom callback instances during Trainer initialization

### Logging Custom Metrics in Training Loop

```python
class MetricsCallback(WandbCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)

        if logs is not None and state.global_step > 0:
            # Compute gradient norm
            total_norm = 0
            for p in self.trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Log custom metric
            self._wandb.log({
                "gradient_norm": total_norm,
                "step": state.global_step
            })
```

### Logging Images and Visualizations

```python
import pandas as pd
import wandb

class VisualizationCallback(WandbCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)

        # Log confusion matrix as image
        if metrics is not None:
            cm_image = self._create_confusion_matrix(metrics)
            self._wandb.log({
                "confusion_matrix": wandb.Image(cm_image),
                "epoch": state.epoch
            })

    def _create_confusion_matrix(self, metrics):
        # Your visualization code here
        # Return PIL Image or matplotlib figure
        pass
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- WandbCallback supports logging images via `wandb.Image()`
- Can log matplotlib figures, PIL images, or numpy arrays

### Logging Predictions During Training

Complete example of logging model predictions to W&B Table:

```python
from transformers.integrations import WandbCallback
import pandas as pd
import wandb

def decode_predictions(tokenizer, predictions):
    """Decode model predictions and labels"""
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}

class WandbPredictionProgressCallback(WandbCallback):
    """Log model predictions to W&B Table during training"""

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """
        Args:
            trainer: HuggingFace Trainer instance
            tokenizer: Tokenizer for decoding predictions
            val_dataset: Validation dataset
            num_samples: Number of samples to log
            freq: Log every N epochs
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        # Log predictions every freq epochs
        if state.epoch % self.freq == 0:
            # Generate predictions
            predictions = self.trainer.predict(self.sample_dataset)

            # Decode predictions and labels
            decoded = decode_predictions(self.tokenizer, predictions)

            # Create DataFrame and add epoch
            predictions_df = pd.DataFrame(decoded)
            predictions_df["epoch"] = state.epoch

            # Log to W&B Table
            records_table = self._wandb.Table(dataframe=predictions_df)
            self._wandb.log({"sample_predictions": records_table})

# Usage
trainer = Trainer(model=model, args=args, ...)
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=val_dataset,
    num_samples=10,
    freq=2
)
trainer.add_callback(progress_callback)
trainer.train()
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- Callback logs "model predictions and labels to a wandb.Table at each logging step"
- Allows visualization of model predictions as training progresses
- Full example available in [W&B Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)

---

## Advanced Patterns (~130 lines)

### Model Checkpointing with W&B Artifacts

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):

```python
# Enable checkpoint logging via environment variable
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # or "end" or "false"

args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    save_steps=500,  # Save every 500 steps
    save_total_limit=3,  # Keep only 3 checkpoints
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="eval_loss",
)
```

**`WANDB_LOG_MODEL` options**:
- `"checkpoint"`: Upload checkpoint every `args.save_steps`
- `"end"`: Upload final model at end of training (requires `load_best_model_at_end=True`)
- `"false"`: Don't upload model

**Checkpoint naming**:
- Default: `checkpoint-{run_id}` or `model-{run_id}`
- Custom: `checkpoint-{run_name}` if `run_name` is provided in TrainingArguments

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- Checkpoints are viewable in W&B Artifacts UI with full model lineage
- Can store up to 100GB of models/datasets for free

### Loading Models from W&B Artifacts

```python
import wandb
from transformers import AutoModelForSequenceClassification

# Start new run and download artifact
with wandb.init(project="sentiment-analysis") as run:
    # Reference the artifact (name:version or name:alias)
    artifact_name = "model-bert-base-high-lr:latest"
    model_artifact = run.use_artifact(artifact_name)

    # Download to local directory
    model_dir = model_artifact.download()

    # Load into HuggingFace model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels
    )

    # Continue training or run inference
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- Use `run.use_artifact()` to reference saved models
- Artifacts support versioning (`:v0`, `:v1`, `:latest`, `:best`)
- Downloaded models work with standard HuggingFace `.from_pretrained()`

### Resuming Training from Checkpoint

```python
import wandb
from transformers import Trainer, TrainingArguments

last_run_id = "abc123xyz"  # Previous run ID

# Resume W&B run
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",  # Must resume existing run
) as run:
    # Download checkpoint artifact
    checkpoint_name = f"checkpoint-{last_run_id}:latest"
    checkpoint_artifact = run.use_artifact(checkpoint_name)
    checkpoint_dir = checkpoint_artifact.download()

    # Reinitialize model and trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="./results",
        report_to="wandb",
        # ... other args
    )

    trainer = Trainer(model=model, args=training_args)

    # Resume training from checkpoint
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- Pass `resume="must"` to `wandb.init()` to continue existing run
- Use checkpoint directory as `resume_from_checkpoint` parameter
- Requires `WANDB_LOG_MODEL='checkpoint'` was set during original training

### Hyperparameter Sweeps Integration

While W&B Sweeps are typically configured via YAML, you can integrate with Trainer:

```python
import wandb
from transformers import Trainer, TrainingArguments

def train():
    """Training function called by W&B sweep agent"""
    # Initialize W&B run (sweep agent handles this)
    run = wandb.init()

    # Access sweep parameters
    config = wandb.config

    # Use sweep config in TrainingArguments
    args = TrainingArguments(
        output_dir="./results",
        report_to="wandb",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Best metric is automatically logged to sweep
    run.finish()

# Sweep configuration (typically in YAML file)
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 5e-5},
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [3, 5, 10]},
        'weight_decay': {'min': 0.0, 'max': 0.1},
    }
}

# Initialize sweep and run agent
sweep_id = wandb.sweep(sweep_config, project="sentiment-analysis")
wandb.agent(sweep_id, function=train, count=10)
```

From search results on [W&B Community Forum](https://community.wandb.ai/t/what-is-the-official-way-to-run-a-wandb-sweep-with-hugging-face-hf-transformers/4668) (accessed 2025-01-31):
- Sweep agent automatically calls `wandb.init()` for each run
- Access sweep parameters via `wandb.config`
- Trainer metrics automatically reported to sweep dashboard

### Multi-Run Experiments and Comparison

```python
# Experiment 1: Low learning rate
os.environ["WANDB_PROJECT"] = "bert-experiments"
args1 = TrainingArguments(
    output_dir="./results_low_lr",
    report_to="wandb",
    run_name="bert-low-lr",
    learning_rate=1e-5,
    # ... other args
)
trainer1 = Trainer(model=model1, args=args1)
trainer1.train()

# Experiment 2: High learning rate
args2 = TrainingArguments(
    output_dir="./results_high_lr",
    report_to="wandb",
    run_name="bert-high-lr",
    learning_rate=5e-5,
    # ... other args
)
trainer2 = Trainer(model=model2, args=args2)
trainer2.train()

# Both runs appear in same project for easy comparison
```

**Best practices for multi-run experiments**:
- Use consistent `WANDB_PROJECT` for related experiments
- Use descriptive `run_name` values
- Tag runs with `tags` parameter in `wandb.init()` for grouping
- Group runs with `group` parameter for related experiments

### Dataset Logging with W&B Artifacts

```python
import wandb

# Initialize run
with wandb.init(project="sentiment-analysis") as run:
    # Log dataset as artifact
    dataset_artifact = wandb.Artifact(
        name="imdb-train-dataset",
        type="dataset",
        description="IMDB sentiment training data"
    )

    # Add files to artifact
    dataset_artifact.add_file("./data/train.csv")
    dataset_artifact.add_file("./data/val.csv")

    # Log artifact
    run.log_artifact(dataset_artifact)

    # Link dataset to model training run
    run.use_artifact("imdb-train-dataset:latest")

    # Normal training proceeds
    trainer = Trainer(...)
    trainer.train()
```

From [W&B HuggingFace Integration Docs](https://docs.wandb.ai/models/integrations/huggingface) (accessed 2025-01-31):
- Artifacts track datasets, models, and evaluation results
- Full lineage tracking shows which datasets were used to train which models
- Can link models to W&B Registry for deployment tracking

---

## Sources

**Web Research (accessed 2025-01-31):**
- [Hugging Face Transformers - W&B Documentation](https://docs.wandb.ai/models/integrations/huggingface) - Official W&B integration guide
- [HuggingFace Callbacks Documentation](https://huggingface.co/docs/transformers/en/main_classes/callback) - TrainerCallback API reference
- [W&B Community Forum: Sweeps with HuggingFace](https://community.wandb.ai/t/what-is-the-official-way-to-run-a-wandb-sweep-with-hugging-face-hf-transformers/4668) - Sweep integration patterns

**Key GitHub Resources:**
- [W&B Examples: HuggingFace Custom Callback](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) - Complete custom callback example

**Additional References:**
- [HuggingFace Trainer Documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer) - Trainer class reference
- [W&B Artifacts Documentation](https://docs.wandb.ai/models/artifacts) - Model versioning guide
