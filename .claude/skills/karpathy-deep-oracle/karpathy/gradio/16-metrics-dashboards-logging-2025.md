# Gradio Metrics Dashboards and Logging (2025)

## Overview

Gradio provides powerful tools for creating real-time metrics dashboards and integrating with experiment tracking platforms like Weights & Biases (W&B) and MLflow. This guide covers building research-quality dashboards with live plotting, experiment tracking integration, and export capabilities.

**Key capabilities for VLM research:**
- Real-time metrics visualization during training
- Integration with W&B, MLflow, and Trackio
- Multi-panel dashboard layouts
- CSV/JSON export for offline analysis
- Embedding live metrics in research reports

## Real-Time Metrics with gr.Plot

### Basic Plot Components

Gradio provides three native plot components for different visualization needs:

```python
import gradio as gr
import pandas as pd
import numpy as np

# Create sample metrics data
df = pd.DataFrame({
    'epoch': range(100),
    'train_loss': np.linspace(1.0, 0.2, 100) + np.random.randn(100) * 0.05,
    'val_loss': np.linspace(1.0, 0.25, 100) + np.random.randn(100) * 0.05,
    'train_acc': np.linspace(0.5, 0.95, 100) + np.random.randn(100) * 0.02
})

with gr.Blocks() as demo:
    # Line plot for losses
    gr.LinePlot(df, x="epoch", y="train_loss",
                title="Training Loss")

    # Scatter plot for accuracy
    gr.ScatterPlot(df, x="epoch", y="train_acc",
                   title="Training Accuracy")

    # Bar plot for aggregated metrics
    gr.BarPlot(df, x="epoch", y="train_loss",
               x_bin=10, y_aggregate="mean")

demo.launch()
```

### Multi-Series Plots with Color

Break out metrics by category using the `color` argument:

```python
import gradio as gr
import pandas as pd

# Metrics from multiple model variants
df = pd.DataFrame({
    'epoch': list(range(50)) * 3,
    'loss': np.concatenate([
        np.linspace(1.0, 0.2, 50),  # baseline
        np.linspace(1.0, 0.15, 50),  # improved
        np.linspace(1.0, 0.25, 50)   # variant
    ]),
    'model': ['baseline']*50 + ['improved']*50 + ['variant']*50
})

with gr.Blocks() as demo:
    gr.LinePlot(df, x="epoch", y="loss", color="model",
                title="Model Comparison")

demo.launch()
```

**Custom color mapping:**
```python
color_map = {
    'baseline': '#FF6B6B',
    'improved': '#4ECDC4',
    'variant': '#95E1D3'
}

gr.LinePlot(df, x="epoch", y="loss", color="model",
            color_map=color_map)
```

### Updating Plots During Training

Use functions to dynamically update plots:

```python
import gradio as gr
import pandas as pd
import time

# Store metrics globally
metrics_history = {'epoch': [], 'loss': [], 'accuracy': []}

def train_step(epoch):
    """Simulate training step"""
    loss = 1.0 / (epoch + 1) + np.random.randn() * 0.05
    acc = (1 - loss) * 0.9 + np.random.randn() * 0.02

    metrics_history['epoch'].append(epoch)
    metrics_history['loss'].append(loss)
    metrics_history['accuracy'].append(acc)

    df = pd.DataFrame(metrics_history)
    return df

with gr.Blocks() as demo:
    epoch_input = gr.Number(label="Epoch", value=0)
    train_btn = gr.Button("Train Step")

    loss_plot = gr.LinePlot(x="epoch", y="loss",
                            title="Loss over Time")
    acc_plot = gr.LinePlot(x="epoch", y="accuracy",
                           title="Accuracy over Time")

    train_btn.click(train_step, inputs=epoch_input,
                    outputs=[loss_plot, acc_plot])

demo.launch()
```

### Interactive Plot Features

**Selecting regions:**
```python
with gr.Blocks() as demo:
    plt = gr.LinePlot(df, x="epoch", y="loss")
    selection_info = gr.Textbox(label="Selection Info")

    def select_region(selection: gr.SelectData):
        min_x, max_x = selection.index
        selected = df[(df["epoch"] >= min_x) & (df["epoch"] <= max_x)]
        return f"Epochs {min_x}-{max_x}: Avg Loss = {selected['loss'].mean():.4f}"

    plt.select(select_region, None, selection_info)

demo.launch()
```

**Zoom functionality:**
```python
def zoom_to_selection(selection: gr.SelectData):
    min_x, max_x = selection.index
    return gr.LinePlot(x_lim=(min_x, max_x))

plt.select(zoom_to_selection, None, plt)
plt.double_click(lambda: gr.LinePlot(x_lim=None), None, plt)
```

## W&B Integration

### Basic W&B + Gradio Setup

From [Gradio and W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-10-31):

```python
import gradio as gr
import wandb
import torch
import time

# Initialize W&B
wandb.init(
    project="vlm-training",
    config={
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32
    }
)

# Training loop with logging
for epoch in range(100):
    # Simulate training
    train_loss = compute_loss()
    train_acc = compute_accuracy()

    # Log to W&B
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Log images periodically
    if epoch % 10 == 0:
        sample_output = model(sample_input)
        wandb.log({
            "sample_output": [wandb.Image(sample_output)]
        })

wandb.finish()
```

### Integrating W&B Dashboard into Gradio

**Simple one-line integration:**
```python
import gradio as gr
import wandb

def create_interface():
    demo = gr.Interface(
        fn=my_model_function,
        inputs=gr.Image(),
        outputs=gr.Image()
    )

    # Integrate with W&B
    demo.integrate(wandb=wandb)

    return demo

demo = create_interface()
demo.launch()
```

### Embedding W&B Reports in Gradio

From the official guide:

```python
import gradio as gr

def embed_wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)

with gr.Blocks() as demo:
    gr.Markdown("# Training Metrics Dashboard")

    # Embed W&B report
    report_url = 'https://wandb.ai/username/project/reports/Report-Name'
    report = embed_wandb_report(report_url)

    # Add controls below
    with gr.Row():
        epoch_slider = gr.Slider(0, 100, label="Epoch")
        metric_dropdown = gr.Dropdown(
            ["loss", "accuracy", "f1"],
            label="Metric"
        )

demo.launch()
```

### Logging Complex Metrics to W&B

```python
import wandb
import gradio as gr

# Create W&B table for detailed results
samples = []
column_names = ["Input Image", "Prediction", "Ground Truth", "Loss"]

for batch in dataloader:
    images, labels = batch
    predictions = model(images)
    loss = criterion(predictions, labels)

    # Log per-sample metrics
    for i in range(len(images)):
        samples.append([
            wandb.Image(images[i]),
            predictions[i].argmax(),
            labels[i],
            loss[i].item()
        ])

# Create and log table
table = wandb.Table(data=samples, columns=column_names)
wandb.log({"predictions": table})
```

## Trackio: Local-First Experiment Tracking

From [gradio-app/trackio](https://github.com/gradio-app/trackio) (accessed 2025-10-31):

**Trackio** is Gradio's lightweight, free experiment tracking library with W&B-compatible API.

### Basic Trackio Usage

```python
import trackio
import random
import time

runs = 3
epochs = 8

for run in range(runs):
    trackio.init(
        project="arr-coc-vlm",
        config={"epochs": epochs, "lr": 0.001, "batch_size": 64}
    )

    for epoch in range(epochs):
        train_loss = random.uniform(0.2, 1.0)
        train_acc = random.uniform(0.6, 0.95)
        val_loss = train_loss - random.uniform(0.01, 0.1)
        val_acc = train_acc + random.uniform(0.01, 0.05)

        trackio.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        time.sleep(0.2)

    trackio.finish()

# Launch dashboard
trackio.show(project="arr-coc-vlm")
```

### Drop-in W&B Replacement

```python
# Just change the import!
import trackio as wandb

# All existing W&B code works
wandb.init(project="my-project")
wandb.log({"loss": 0.5})
wandb.finish()
```

### Deploying Trackio to Hugging Face Spaces

```python
import trackio

# Automatically deploys to HF Spaces
trackio.init(
    project="arr-coc-experiments",
    space_id="username/trackio-dashboard"
)

# Logs stored in private HF Dataset
trackio.log({"metric": value})

# Dashboard hosted on HF Spaces (free!)
```

### Embedding Trackio Dashboards

Embed live dashboards with query parameters:

```html
<iframe
  src="https://username-trackio.hf.space/?project=my-project&metrics=train_loss,train_accuracy&sidebar=hidden&xmin=0&xmax=100&smoothing=5"
  style="width:1600px; height:500px; border:0;">
</iframe>
```

**Query parameters:**
- `project`: Filter to specific project
- `metrics`: Comma-separated list (e.g., `train_loss,val_loss`)
- `sidebar`: `hidden` or `collapsed`
- `xmin`/`xmax`: Set x-axis limits
- `smoothing`: 0-20 (0 = no smoothing)

## MLflow Integration

### Basic MLflow Logging

```python
import mlflow
import gradio as gr

mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)

# Log metrics
for epoch in range(100):
    train_loss = train_step()
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)

# Log artifacts
mlflow.log_artifact("model_checkpoint.pt")
mlflow.end_run()
```

### MLflow in Gradio Dashboard

```python
import mlflow
import gradio as gr
import pandas as pd

def load_mlflow_metrics(run_id):
    """Load metrics from MLflow run"""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    metrics_data = []
    for key in run.data.metrics:
        history = client.get_metric_history(run_id, key)
        for entry in history:
            metrics_data.append({
                'step': entry.step,
                'metric': key,
                'value': entry.value
            })

    return pd.DataFrame(metrics_data)

with gr.Blocks() as demo:
    run_id_input = gr.Textbox(label="MLflow Run ID")
    load_btn = gr.Button("Load Metrics")

    plot = gr.LinePlot(x="step", y="value", color="metric")

    load_btn.click(load_mlflow_metrics, inputs=run_id_input,
                   outputs=plot)

demo.launch()
```

## Dashboard Layouts for Research

### Multi-Panel Metrics Dashboard

```python
import gradio as gr
import pandas as pd

def create_research_dashboard(df):
    with gr.Blocks() as demo:
        gr.Markdown("# ARR-COC VLM Training Dashboard")

        # Top row: Key metrics
        with gr.Row():
            with gr.Column():
                gr.Number(label="Best Val Loss",
                         value=df['val_loss'].min())
            with gr.Column():
                gr.Number(label="Best Val Acc",
                         value=df['val_accuracy'].max())
            with gr.Column():
                gr.Number(label="Current Epoch",
                         value=df['epoch'].max())

        # Loss plots
        with gr.Row():
            with gr.Column():
                gr.LinePlot(df, x="epoch", y="train_loss",
                           title="Training Loss")
            with gr.Column():
                gr.LinePlot(df, x="epoch", y="val_loss",
                           title="Validation Loss")

        # Accuracy plots
        with gr.Row():
            with gr.Column():
                gr.LinePlot(df, x="epoch", y="train_accuracy",
                           title="Training Accuracy")
            with gr.Column():
                gr.LinePlot(df, x="epoch", y="val_accuracy",
                           title="Validation Accuracy")

        # Learning rate schedule
        gr.LinePlot(df, x="epoch", y="learning_rate",
                   title="Learning Rate Schedule")

        # Export button
        export_btn = gr.Button("Export Metrics")
        export_file = gr.File(label="Download")

        export_btn.click(
            lambda: df.to_csv("metrics.csv", index=False),
            outputs=export_file
        )

    return demo

demo = create_research_dashboard(metrics_df)
demo.launch()
```

### Tabbed Dashboard Layout

```python
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Training Metrics"):
            gr.LinePlot(df, x="epoch", y="train_loss")
            gr.LinePlot(df, x="epoch", y="train_accuracy")

        with gr.TabItem("Validation Metrics"):
            gr.LinePlot(df, x="epoch", y="val_loss")
            gr.LinePlot(df, x="epoch", y="val_accuracy")

        with gr.TabItem("Learning Rate"):
            gr.LinePlot(df, x="epoch", y="learning_rate")

        with gr.TabItem("GPU Metrics"):
            gr.LinePlot(df, x="epoch", y="gpu_memory_used")
            gr.LinePlot(df, x="epoch", y="gpu_utilization")

demo.launch()
```

### Interactive Filtering Dashboard

```python
import gradio as gr

def create_filtered_dashboard(df):
    with gr.Blocks() as demo:
        gr.Markdown("# Filtered Metrics Dashboard")

        # Filters
        with gr.Row():
            metric_filter = gr.Dropdown(
                ["train_loss", "val_loss", "train_accuracy"],
                label="Metric",
                value="train_loss"
            )
            epoch_range = gr.Slider(0, 100, value=100,
                                   label="Max Epoch")
            smoothing = gr.Slider(0, 20, value=0,
                                 label="Smoothing")

        def filter_and_plot(metric, max_epoch, smooth):
            filtered = df[df['epoch'] <= max_epoch]

            if smooth > 0:
                filtered[metric] = filtered[metric].rolling(
                    window=smooth, center=True
                ).mean()

            return gr.LinePlot(filtered, x="epoch", y=metric,
                              title=f"{metric} (Smoothing={smooth})")

        plot = gr.Plot()

        # Update on any change
        for component in [metric_filter, epoch_range, smoothing]:
            component.change(
                filter_and_plot,
                inputs=[metric_filter, epoch_range, smoothing],
                outputs=plot
            )

    return demo
```

## Export and Reporting Patterns

### CSV Export

```python
import gradio as gr
import pandas as pd

def export_metrics_csv(metrics_dict):
    """Export metrics to CSV"""
    df = pd.DataFrame(metrics_dict)
    csv_path = "training_metrics.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

with gr.Blocks() as demo:
    # Display current metrics
    metrics_table = gr.Dataframe(value=metrics_df)

    # Export button
    export_btn = gr.Button("Export to CSV")
    download_file = gr.File(label="Download CSV")

    export_btn.click(
        lambda: export_metrics_csv(metrics_df),
        outputs=download_file
    )

demo.launch()
```

### JSON Export for Analysis

```python
import json
import gradio as gr

def export_experiment_json(config, metrics):
    """Export full experiment data"""
    export_data = {
        'config': config,
        'metrics': metrics,
        'metadata': {
            'timestamp': time.time(),
            'best_epoch': metrics['val_loss'].argmin(),
            'best_val_loss': metrics['val_loss'].min()
        }
    }

    with open('experiment.json', 'w') as f:
        json.dump(export_data, f, indent=2)

    return 'experiment.json'

# Usage in Gradio
export_btn.click(
    lambda: export_experiment_json(config_dict, metrics_dict),
    outputs=gr.File()
)
```

### Summary Statistics Display

```python
import gradio as gr
import pandas as pd

def compute_summary_stats(df):
    """Compute summary statistics"""
    summary = {
        'Total Epochs': len(df),
        'Best Train Loss': f"{df['train_loss'].min():.4f}",
        'Best Val Loss': f"{df['val_loss'].min():.4f}",
        'Final Train Acc': f"{df['train_accuracy'].iloc[-1]:.4f}",
        'Final Val Acc': f"{df['val_accuracy'].iloc[-1]:.4f}",
        'Training Time': f"{df['time'].sum():.2f}s"
    }
    return summary

with gr.Blocks() as demo:
    gr.Markdown("## Summary Statistics")

    stats_display = gr.JSON(label="Metrics Summary")
    update_btn = gr.Button("Update Summary")

    update_btn.click(
        lambda: compute_summary_stats(current_metrics_df),
        outputs=stats_display
    )

demo.launch()
```

### Automated Report Generation

```python
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_pdf_report(df, config):
    """Generate PDF report with plots"""
    pdf_path = "training_report.pdf"

    with PdfPages(pdf_path) as pdf:
        # Page 1: Loss curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(df['epoch'], df['train_loss'], label='Train')
        axes[0].plot(df['epoch'], df['val_loss'], label='Val')
        axes[0].set_title('Loss Curves')
        axes[0].legend()

        axes[1].plot(df['epoch'], df['train_accuracy'], label='Train')
        axes[1].plot(df['epoch'], df['val_accuracy'], label='Val')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()

        pdf.savefig(fig)
        plt.close()

        # Page 2: Summary table
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')

        summary_data = [
            ['Metric', 'Value'],
            ['Best Val Loss', f"{df['val_loss'].min():.4f}"],
            ['Best Val Acc', f"{df['val_accuracy'].max():.4f}"],
            ['Total Epochs', str(len(df))],
            ['Learning Rate', str(config['learning_rate'])]
        ]

        table = ax.table(cellText=summary_data, loc='center',
                        cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        pdf.savefig(fig)
        plt.close()

    return pdf_path

# In Gradio interface
report_btn = gr.Button("Generate PDF Report")
report_file = gr.File(label="Download Report")

report_btn.click(
    lambda: generate_pdf_report(metrics_df, config),
    outputs=report_file
)
```

## Sources

**Official Gradio Documentation:**
- [Gradio and W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) - Official guide on integrating W&B with Gradio (accessed 2025-10-31)
- [Creating Plots in Gradio](https://www.gradio.app/guides/creating-plots) - Guide on gr.Plot components and interactive dashboards (accessed 2025-10-31)

**Trackio (Gradio's Experiment Tracking):**
- [gradio-app/trackio GitHub Repository](https://github.com/gradio-app/trackio) - Lightweight, local-first experiment tracking library with W&B-compatible API (accessed 2025-10-31)

**Community Resources:**
- [How Gradio and W&B Work Beautifully Together](https://wandb.ai/abidlabs/your-test-project/reports/How-Gradio-and-W-B-Work-Beautifully-Together---Vmlldzo4MTk0MzI) - Weights & Biases tutorial on integration
- Analytics Vidhya: "Building an Interactive Data Dashboard Creation With Gradio" (2025-07-06)
- Medium: "How I Built a Real-Time, Interactive Dashboard Using Python and Gradio" by Rajeev Barnwal (2025)

**Additional References:**
- MLflow Documentation for Python API
- Plotly integration patterns with Gradio
- Gradio Blocks advanced layout patterns
