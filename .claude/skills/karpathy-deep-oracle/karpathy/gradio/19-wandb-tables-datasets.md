# W&B Tables: Dataset Exploration and Prediction Logging

**Purpose**: Interactive tabular data visualization and analysis for datasets, model predictions, and training monitoring

**Context**: W&B Tables provide a powerful interface for exploring datasets, logging predictions with ground truth, analyzing model failures, and tracking training progress through structured tabular data with rich media support.

---

## Section 1: Tables Basics (Creating, Logging, Data Types)

### What is a W&B Table?

A W&B Table is a two-dimensional grid of data where:
- Each column has a single data type
- Each row represents one or more data points logged to a W&B run
- Supports primitive types, nested structures, and rich media
- Tables are specialized data types logged as artifact objects

From [W&B Tables Documentation](https://docs.wandb.ai/models/tables) (accessed 2025-01-31):
> "Use W&B Tables to visualize and query tabular data. For example:
> - Compare how different models perform on the same test set
> - Identify patterns in your data
> - Look at sample model predictions visually
> - Query to find commonly misclassified examples"

### Creating Tables

**Basic table creation:**

```python
import wandb

with wandb.init(project="table-demo") as run:
    # Create table with columns and data
    my_table = wandb.Table(
        columns=["a", "b"],
        data=[["a1", "b1"], ["a2", "b2"]],
        log_mode="IMMUTABLE"  # Default mode
    )

    # Log to W&B
    run.log({"Table Name": my_table})
```

**From Pandas DataFrame:**

```python
import pandas as pd
import wandb

df = pd.DataFrame({
    'image_id': [1, 2, 3],
    'class': ['cat', 'dog', 'bird'],
    'confidence': [0.95, 0.87, 0.92]
})

with wandb.init(project="dataframe-demo") as run:
    # Convert DataFrame to Table
    table = wandb.Table(dataframe=df)
    run.log({"predictions": table})
```

### Supported Data Types

From [W&B Tables Log Documentation](https://docs.wandb.ai/models/tables/log_tables) (accessed 2025-01-31):

**Primitive types:**
- Strings, integers, floats, booleans
- Nested lists and dictionaries

**Rich media types:**
- `wandb.Image()` - images with optional masks/boxes
- `wandb.Audio()` - audio files
- `wandb.Video()` - video files
- `wandb.Html()` - custom HTML content
- `wandb.Object3D()` - 3D objects

**Example with rich media:**

```python
import wandb

with wandb.init(project="media-table") as run:
    table = wandb.Table(columns=["image", "label", "prediction"])

    # Add row with image
    table.add_data(
        wandb.Image("cat.jpg"),
        "cat",
        "cat"
    )

    run.log({"predictions": table})
```

### Logging Modes

From [W&B Tables Log Documentation](https://docs.wandb.ai/models/tables/log_tables) (accessed 2025-01-31):

| Mode | Definition | Use Cases | Benefits |
|------|-----------|-----------|----------|
| `IMMUTABLE` | Cannot modify after logging | End-of-run analysis | Minimal overhead, all rows rendered |
| `MUTABLE` | Overwrite with new table | Add columns/rows, enrich results | Capture mutations, all rows rendered |
| `INCREMENTAL` | Add batches during training | Long-running jobs, large datasets | View updates during training, step through increments |

**IMMUTABLE (default):**
```python
# Log once, cannot modify
table = wandb.Table(
    columns=["input", "output"],
    data=[[1, 2], [3, 4]],
    log_mode="IMMUTABLE"
)
run.log({"final_results": table})
```

**MUTABLE (for enrichment):**
```python
# Can update with new columns
table = wandb.Table(
    columns=["input", "label", "prediction"],
    log_mode="MUTABLE"
)

# Initial log
for inp, label, pred in zip(inputs, labels, predictions):
    table.add_data(inp, label, pred)
run.log({"eval_table": table})

# Add confidence scores
confidences = np.max(predictions, axis=1)
table.add_column("confidence", confidences)
run.log({"eval_table": table})  # Overwrites previous
```

**INCREMENTAL (for streaming):**
```python
# Add rows during training
table = wandb.Table(
    columns=["step", "input", "label", "prediction"],
    log_mode="INCREMENTAL"
)

for step in range(num_batches):
    inputs, labels = get_batch(step)
    predictions = model.predict(inputs)

    for inp, label, pred in zip(inputs, labels, predictions):
        table.add_data(step, inp, label, pred)

    # Log incrementally (efficient)
    run.log({"training_table": table}, step=step)
```

---

## Section 2: Prediction Logging Patterns

### Ground Truth vs Prediction Comparison

From [W&B Tables Tutorial](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY) (accessed 2025-01-31):

**Basic prediction logging:**

```python
import wandb

with wandb.init(project="predictions") as run:
    table = wandb.Table(columns=[
        "id",
        "image",
        "ground_truth",
        "prediction",
        "confidence",
        "correct"
    ])

    for idx, (img, gt, pred, conf) in enumerate(test_data):
        table.add_data(
            idx,
            wandb.Image(img),
            gt,
            pred,
            conf,
            gt == pred  # Boolean for filtering
        )

    run.log({"predictions": table})
```

### Multi-Class Prediction Logging

```python
import numpy as np
import wandb

with wandb.init(project="multiclass") as run:
    table = wandb.Table(columns=[
        "image",
        "true_class",
        "pred_class",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
        "top_3_classes"
    ])

    for img, true_label, probs in test_data:
        pred_class = np.argmax(probs)
        top_3 = np.argsort(probs)[-3:][::-1].tolist()

        table.add_data(
            wandb.Image(img),
            true_label,
            pred_class,
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
            top_3
        )

    run.log({"multiclass_predictions": table})
```

### Failure Case Analysis

From [W&B Tables Walkthrough](https://docs.wandb.ai/models/tables/tables-walkthrough) (accessed 2025-01-31):

**Pattern for analyzing errors:**

```python
import wandb

with wandb.init(project="failure-analysis") as run:
    table = wandb.Table(columns=[
        "image",
        "true_label",
        "predicted_label",
        "confidence",
        "error_type",  # "false_positive", "false_negative", "correct"
        "loss"
    ])

    for img, true_label, pred_label, conf, loss in results:
        # Classify error type
        if true_label == pred_label:
            error_type = "correct"
        elif pred_label == 1 and true_label == 0:
            error_type = "false_positive"
        else:
            error_type = "false_negative"

        table.add_data(
            wandb.Image(img),
            true_label,
            pred_label,
            conf,
            error_type,
            loss
        )

    run.log({"error_analysis": table})
```

**Filter in UI to find:**
- High-confidence errors: `confidence > 0.9 AND error_type != "correct"`
- False positives: `error_type == "false_positive"`
- Worst losses: Sort by `loss` descending

### Confidence Score Visualization

```python
import wandb
import numpy as np

with wandb.init(project="confidence-analysis") as run:
    table = wandb.Table(columns=[
        "image",
        "prediction",
        "max_confidence",      # Max probability
        "entropy",             # Prediction uncertainty
        "margin",              # Difference between top 2 probs
        "calibrated_conf"      # Post-calibration confidence
    ])

    for img, probs in predictions:
        sorted_probs = np.sort(probs)[::-1]

        max_conf = sorted_probs[0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        margin = sorted_probs[0] - sorted_probs[1]

        # Hypothetical calibration
        calibrated_conf = calibrate(max_conf, model)

        table.add_data(
            wandb.Image(img),
            np.argmax(probs),
            float(max_conf),
            float(entropy),
            float(margin),
            float(calibrated_conf)
        )

    run.log({"confidence_table": table})
```

### VQA and Multi-Modal Logging

**Visual Question Answering pattern:**

```python
import wandb

with wandb.init(project="vqa") as run:
    table = wandb.Table(columns=[
        "image",
        "question",
        "ground_truth_answer",
        "predicted_answer",
        "answer_confidence",
        "question_type",  # e.g., "counting", "color", "location"
        "correct"
    ])

    for img, question, gt_answer, pred_answer, conf, q_type in vqa_results:
        table.add_data(
            wandb.Image(img),
            question,
            gt_answer,
            pred_answer,
            conf,
            q_type,
            gt_answer.lower() == pred_answer.lower()
        )

    run.log({"vqa_predictions": table})
```

**Filter examples:**
- Counting errors: `question_type == "counting" AND correct == False`
- Low confidence correct: `correct == True AND answer_confidence < 0.5`

---

## Section 3: Interactive Filtering and Analysis

### Filtering and Querying

From [W&B Tables Visualization Documentation](https://docs.wandb.ai/models/tables/visualize-tables) (accessed 2025-01-31):

**Built-in UI operations:**
- **Filter**: Query rows with expressions (`confidence > 0.8 AND label == "cat"`)
- **Sort**: Order by any column (ascending/descending)
- **Group**: Aggregate by categorical columns
- **Custom columns**: Create computed fields

**Filter examples:**

```python
# In W&B UI, use filter expressions:

# Find high-confidence errors
"confidence > 0.9 AND prediction != ground_truth"

# Find specific class errors
"ground_truth == 'dog' AND prediction == 'cat'"

# Find low-confidence predictions
"confidence < 0.5"

# Complex boolean logic
"(loss > 2.0 OR confidence < 0.3) AND error_type == 'false_positive'"
```

### Cross-Run Table Comparison

From [W&B Tables Visualization Documentation](https://docs.wandb.ai/models/tables/visualize-tables) (accessed 2025-01-31):

**Merged view (compare predictions across models):**

1. Navigate to project workspace
2. Select multiple runs
3. Tables with same key are automatically merged
4. Each run's data gets a colored highlight (blue=run 0, yellow=run 1)
5. Use join key (e.g., `image_id`) to align rows

**Example: Compare two model versions:**

```python
# Model A run
with wandb.init(project="model-comparison", name="baseline") as run:
    table = wandb.Table(columns=["image_id", "prediction", "confidence"])
    for img_id, pred, conf in model_a_predictions:
        table.add_data(img_id, pred, conf)
    run.log({"predictions": table})

# Model B run
with wandb.init(project="model-comparison", name="improved") as run:
    table = wandb.Table(columns=["image_id", "prediction", "confidence"])
    for img_id, pred, conf in model_b_predictions:
        table.add_data(img_id, pred, conf)
    run.log({"predictions": table})
```

**In UI:**
- Merged view shows both predictions side-by-side
- Filter to disagreements: `0.prediction != 1.prediction`
- Find confidence changes: Histogram of `1.confidence - 0.confidence`

### Artifact Version Comparison

From [W&B Tables Visualization Documentation](https://docs.wandb.ai/models/tables/visualize-tables) (accessed 2025-01-31):

**Compare tables across training time:**

```python
import wandb

# Log predictions at different epochs
for epoch in range(num_epochs):
    with wandb.init(project="training-progress") as run:
        # Train model...

        # Log predictions as artifact
        table = wandb.Table(columns=["image", "prediction", "confidence"])
        for img, pred, conf in validate():
            table.add_data(wandb.Image(img), pred, conf)

        # Log as versioned artifact
        artifact = wandb.Artifact(f"predictions", type="predictions")
        artifact.add(table, "predictions")
        run.log_artifact(artifact)
```

**Compare in UI:**
1. Go to Artifacts tab
2. Select "predictions" artifact
3. Choose two versions (e.g., epoch 1 vs epoch 5)
4. Click "Compare" on second version
5. View side-by-side or merged view

### Custom Columns and Transforms

```python
# Tables support computed columns in UI
# Example: Add column in UI for error margin

# In code, pre-compute useful metrics:
import wandb

with wandb.init(project="custom-metrics") as run:
    table = wandb.Table(columns=[
        "image",
        "true_label",
        "pred_label",
        "confidence",
        "loss",
        "error_margin",  # Custom metric
        "prediction_bucket"  # Categorical grouping
    ])

    for img, true_label, pred_label, conf, loss in results:
        # Compute custom metrics
        error_margin = abs(conf - (1.0 if true_label == pred_label else 0.0))

        # Bucket predictions
        if conf > 0.9:
            bucket = "high_confidence"
        elif conf > 0.5:
            bucket = "medium_confidence"
        else:
            bucket = "low_confidence"

        table.add_data(
            wandb.Image(img),
            true_label,
            pred_label,
            conf,
            loss,
            error_margin,
            bucket
        )

    run.log({"analysis_table": table})
```

### Exporting Table Data

From [W&B Tables Documentation](https://docs.wandb.ai/models/tables/tables-download) (accessed 2025-01-31):

**Export to Pandas for analysis:**

```python
import wandb

# Fetch table from run
api = wandb.Api()
run = api.run("entity/project/run_id")
table_artifact = run.logged_artifacts()[0]  # Get first artifact

# Download and convert to DataFrame
table = table_artifact.get("table_name")
df = pd.DataFrame(data=table.data, columns=table.columns)

# Now use pandas for analysis
error_df = df[df['prediction'] != df['ground_truth']]
print(f"Error rate: {len(error_df) / len(df):.2%}")
```

### Performance Tips for Large Tables

From [W&B Tables Documentation](https://docs.wandb.ai/models/tables/log_tables) (accessed 2025-01-31):

**Use INCREMENTAL + IMMUTABLE pattern for large datasets:**

```python
import wandb

with wandb.init(project="large-dataset") as run:
    # Incremental table for monitoring during training
    incr_table = wandb.Table(
        columns=["step", "input", "prediction", "label"],
        log_mode="INCREMENTAL"
    )

    # Training loop
    for step in range(num_batches):
        inputs, labels = get_batch(step)
        predictions = model.predict(inputs)

        for inp, pred, label in zip(inputs, predictions, labels):
            incr_table.add_data(step, inp, pred, label)

        # Log incrementally (efficient, but limited to 100 increments in UI)
        run.log({"table-incr": incr_table}, step=step)

    # At end, create complete immutable table with all data
    final_table = wandb.Table(
        columns=incr_table.columns,
        data=incr_table.data,
        log_mode="IMMUTABLE"
    )
    run.log({"table": final_table})
```

**Key points:**
- INCREMENTAL mode: View progress during training (max 100 increments shown in UI)
- IMMUTABLE mode: All rows rendered for final analysis
- Use separate keys (`table-incr` vs `table`) to distinguish

---

## Sources

**W&B Documentation:**
- [W&B Tables Overview](https://docs.wandb.ai/models/tables) (accessed 2025-01-31)
- [W&B Tables Walkthrough](https://docs.wandb.ai/models/tables/tables-walkthrough) (accessed 2025-01-31)
- [W&B Tables Log Documentation](https://docs.wandb.ai/models/tables/log_tables) (accessed 2025-01-31)
- [W&B Tables Visualization](https://docs.wandb.ai/models/tables/visualize-tables) (accessed 2025-01-31)

**W&B Reports and Examples:**
- [Tables Tutorial: Text Data & Predictions](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY) (accessed 2025-01-31)
- [Guide to W&B Tables | MNIST](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk) (accessed 2025-01-31)

**Additional Resources:**
- [W&B Tables Gallery Examples](https://docs.wandb.ai/models/tables/tables-gallery) (accessed 2025-01-31)
- [Better Data Understanding with W&B Tables](https://www.kaggle.com/code/ayuraj/better-data-understanding-with-w-b-tables) - Kaggle notebook (accessed 2025-01-31)
