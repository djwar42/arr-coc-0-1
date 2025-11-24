# Gradio Checkpoint Comparison Workflows (2025)

## Overview

Multi-checkpoint comparison is essential for VLM development, allowing researchers to evaluate model performance across training epochs, compare different configurations, and select optimal checkpoints. Gradio provides flexible patterns for building checkpoint comparison interfaces with dropdown selection, simultaneous inference, and result visualization.

**Key Use Cases:**
- Comparing checkpoints across training epochs
- A/B testing different model configurations
- Evaluating ablation study results
- Selecting best checkpoint for deployment
- Visualizing performance degradation/improvement

**Core Pattern:**
```python
import gradio as gr
import torch

def compare_checkpoints(image, query, checkpoint_a, checkpoint_b):
    """Run inference on multiple checkpoints simultaneously"""
    results = {}
    for name, ckpt_path in [("A", checkpoint_a), ("B", checkpoint_b)]:
        model = load_checkpoint(ckpt_path)
        output = model(image, query)
        results[name] = output
    return results["A"], results["B"]

with gr.Blocks() as demo:
    with gr.Row():
        ckpt_a = gr.Dropdown(choices=available_checkpoints, label="Checkpoint A")
        ckpt_b = gr.Dropdown(choices=available_checkpoints, label="Checkpoint B")

    with gr.Row():
        output_a = gr.Textbox(label="Output A")
        output_b = gr.Textbox(label="Output B")

    compare_btn.click(compare_checkpoints, [img, query, ckpt_a, ckpt_b], [output_a, output_b])
```

---

## Section 1: Checkpoint Discovery & Loading

### Auto-Discovery Pattern

Automatically detect available checkpoints from a directory:

```python
import os
import glob
from pathlib import Path
from datetime import datetime

def discover_checkpoints(checkpoint_dir, pattern="*.pth"):
    """
    Discover all checkpoint files in directory

    Returns list of tuples: (display_name, filepath)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    for ckpt_path in sorted(checkpoint_dir.glob(pattern)):
        # Extract metadata from filename
        name = ckpt_path.stem

        # Get file modification time
        mtime = os.path.getmtime(ckpt_path)
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        # Get file size
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)

        # Create display name with metadata
        display_name = f"{name} ({date_str}, {size_mb:.1f}MB)"

        checkpoints.append((display_name, str(ckpt_path)))

    return checkpoints

# Usage in Gradio
checkpoint_dir = "checkpoints/"
available_checkpoints = discover_checkpoints(checkpoint_dir)

with gr.Blocks() as demo:
    checkpoint_dropdown = gr.Dropdown(
        choices=available_checkpoints,
        label="Select Checkpoint",
        value=available_checkpoints[0][1] if available_checkpoints else None
    )
```

### Lazy Loading with LRU Cache

Avoid loading all checkpoints into memory simultaneously:

```python
from functools import lru_cache
import torch

class CheckpointManager:
    """Manage checkpoint loading with LRU cache"""

    def __init__(self, max_loaded=2):
        """
        max_loaded: Maximum number of models kept in memory
        """
        self.max_loaded = max_loaded
        self._cache = {}
        self._load_order = []

    def load_checkpoint(self, ckpt_path, device="cuda"):
        """Load checkpoint with automatic cache management"""

        # Check if already loaded
        if ckpt_path in self._cache:
            # Move to end (most recently used)
            self._load_order.remove(ckpt_path)
            self._load_order.append(ckpt_path)
            return self._cache[ckpt_path]

        # Evict least recently used if cache full
        if len(self._cache) >= self.max_loaded:
            lru_path = self._load_order.pop(0)
            lru_model = self._cache.pop(lru_path)

            # Move to CPU and clear cache
            lru_model.to("cpu")
            torch.cuda.empty_cache()
            del lru_model

        # Load new checkpoint
        model = load_model_from_checkpoint(ckpt_path)
        model = model.to(device)

        # Update cache
        self._cache[ckpt_path] = model
        self._load_order.append(ckpt_path)

        return model

    def clear_cache(self):
        """Manually clear all loaded checkpoints"""
        for model in self._cache.values():
            model.to("cpu")
            del model
        self._cache.clear()
        self._load_order.clear()
        torch.cuda.empty_cache()

# Usage in Gradio
checkpoint_manager = CheckpointManager(max_loaded=2)

def run_inference(image, query, ckpt_path):
    model = checkpoint_manager.load_checkpoint(ckpt_path)
    return model(image, query)
```

### Checkpoint Metadata Display

Extract and display training metadata from checkpoints:

```python
def extract_checkpoint_metadata(ckpt_path):
    """Extract metadata from checkpoint file"""
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    metadata = {
        "epoch": checkpoint.get("epoch", "Unknown"),
        "step": checkpoint.get("step", "Unknown"),
        "train_loss": checkpoint.get("train_loss", "Unknown"),
        "val_loss": checkpoint.get("val_loss", "Unknown"),
        "val_accuracy": checkpoint.get("val_accuracy", "Unknown"),
        "config": checkpoint.get("config", {}),
        "timestamp": checkpoint.get("timestamp", "Unknown")
    }

    return metadata

def format_metadata_display(metadata):
    """Format metadata for display in Gradio"""
    lines = [
        f"**Epoch:** {metadata['epoch']}",
        f"**Step:** {metadata['step']}",
        f"**Train Loss:** {metadata['train_loss']:.4f}" if isinstance(metadata['train_loss'], float) else f"**Train Loss:** {metadata['train_loss']}",
        f"**Val Loss:** {metadata['val_loss']:.4f}" if isinstance(metadata['val_loss'], float) else f"**Val Loss:** {metadata['val_loss']}",
        f"**Val Accuracy:** {metadata['val_accuracy']:.2%}" if isinstance(metadata['val_accuracy'], float) else f"**Val Accuracy:** {metadata['val_accuracy']}",
        f"**Timestamp:** {metadata['timestamp']}"
    ]
    return "\n\n".join(lines)

# Usage in Gradio
def on_checkpoint_select(ckpt_path):
    """Update metadata display when checkpoint selected"""
    metadata = extract_checkpoint_metadata(ckpt_path)
    return format_metadata_display(metadata)

with gr.Blocks() as demo:
    checkpoint_dropdown = gr.Dropdown(choices=available_checkpoints, label="Checkpoint")
    metadata_display = gr.Markdown(label="Checkpoint Info")

    checkpoint_dropdown.change(on_checkpoint_select, checkpoint_dropdown, metadata_display)
```

From [GitHub - Checkpoint Management Patterns](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/445) (accessed 2025-10-31):
- Common issue: Users need to switch between multiple checkpoints without reloading entire app
- Solution: Implement lazy loading and cache management for checkpoint switching
- Best practice: Display checkpoint metadata to help users make informed selections

---

## Section 2: Multi-Model Inference UI

### Dropdown Selection for Checkpoints A/B/C/D

Create flexible comparison interface with multiple checkpoint slots:

```python
import gradio as gr

def compare_multiple_checkpoints(image, query, ckpt_paths):
    """
    Run inference on multiple checkpoints simultaneously

    Args:
        image: Input image
        query: Text query
        ckpt_paths: List of checkpoint paths

    Returns:
        Dictionary mapping checkpoint name to output
    """
    results = {}

    for i, ckpt_path in enumerate(ckpt_paths):
        if ckpt_path is None:
            continue

        model = checkpoint_manager.load_checkpoint(ckpt_path)
        output = model.generate(image, query)
        results[f"Checkpoint {chr(65+i)}"] = output  # A, B, C, D

    return results

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Checkpoint Comparison")

    with gr.Row():
        image_input = gr.Image(label="Input Image")
        query_input = gr.Textbox(label="Query")

    # Checkpoint selection dropdowns
    with gr.Row():
        ckpt_a = gr.Dropdown(choices=available_checkpoints, label="Checkpoint A")
        ckpt_b = gr.Dropdown(choices=available_checkpoints, label="Checkpoint B")
        ckpt_c = gr.Dropdown(choices=available_checkpoints, label="Checkpoint C")
        ckpt_d = gr.Dropdown(choices=available_checkpoints, label="Checkpoint D")

    compare_btn = gr.Button("Compare Checkpoints")

    # Results display
    with gr.Row():
        output_a = gr.Textbox(label="Output A")
        output_b = gr.Textbox(label="Output B")

    with gr.Row():
        output_c = gr.Textbox(label="Output C")
        output_d = gr.Textbox(label="Output D")

    def run_comparison(img, query, a, b, c, d):
        results = compare_multiple_checkpoints(img, query, [a, b, c, d])
        return (
            results.get("Checkpoint A", ""),
            results.get("Checkpoint B", ""),
            results.get("Checkpoint C", ""),
            results.get("Checkpoint D", "")
        )

    compare_btn.click(
        run_comparison,
        [image_input, query_input, ckpt_a, ckpt_b, ckpt_c, ckpt_d],
        [output_a, output_b, output_c, output_d]
    )
```

### Run Same Query on Multiple Checkpoints

Batch processing pattern for consistent comparisons:

```python
def batch_compare_checkpoints(image, query, checkpoint_list):
    """
    Run same query on all selected checkpoints

    Returns list of (checkpoint_name, output, inference_time) tuples
    """
    import time

    results = []

    for ckpt_display_name, ckpt_path in checkpoint_list:
        start_time = time.time()

        model = checkpoint_manager.load_checkpoint(ckpt_path)
        output = model.generate(image, query)

        inference_time = time.time() - start_time

        results.append({
            "checkpoint": ckpt_display_name,
            "output": output,
            "inference_time_ms": inference_time * 1000
        })

    return results

# Gradio interface with batch comparison
with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(label="Input Image")
        query_input = gr.Textbox(label="Query")

    checkpoint_selector = gr.CheckboxGroup(
        choices=[name for name, _ in available_checkpoints],
        label="Select Checkpoints to Compare"
    )

    run_batch_btn = gr.Button("Run Batch Comparison")

    results_dataframe = gr.Dataframe(
        headers=["Checkpoint", "Output", "Inference Time (ms)"],
        label="Comparison Results"
    )

    def execute_batch_comparison(img, query, selected_names):
        # Get checkpoint paths for selected names
        selected_ckpts = [(name, path) for name, path in available_checkpoints
                          if name in selected_names]

        results = batch_compare_checkpoints(img, query, selected_ckpts)

        # Format for dataframe
        rows = [[r["checkpoint"], r["output"][:100] + "...",
                 f"{r['inference_time_ms']:.2f}"]
                for r in results]

        return rows

    run_batch_btn.click(
        execute_batch_comparison,
        [image_input, query_input, checkpoint_selector],
        results_dataframe
    )
```

### Gallery Display of Results

Visual comparison using Gradio Gallery:

```python
def compare_checkpoints_visual(image, query, checkpoint_list):
    """
    Generate visual outputs from multiple checkpoints

    For VLMs that generate images or visualizations
    """
    results = []

    for ckpt_name, ckpt_path in checkpoint_list:
        model = checkpoint_manager.load_checkpoint(ckpt_path)

        # Generate output visualization
        output_image = model.generate_visualization(image, query)

        # Add checkpoint name as caption
        results.append((output_image, ckpt_name))

    return results

with gr.Blocks() as demo:
    gr.Markdown("# Visual Checkpoint Comparison")

    with gr.Row():
        image_input = gr.Image(label="Input Image")
        query_input = gr.Textbox(label="Query")

    checkpoint_selector = gr.CheckboxGroup(
        choices=[name for name, _ in available_checkpoints],
        label="Select Checkpoints"
    )

    compare_btn = gr.Button("Generate Comparisons")

    gallery = gr.Gallery(
        label="Checkpoint Outputs",
        columns=2,
        height="auto"
    )

    def generate_gallery(img, query, selected_names):
        selected_ckpts = [(name, path) for name, path in available_checkpoints
                          if name in selected_names]
        return compare_checkpoints_visual(img, query, selected_ckpts)

    compare_btn.click(
        generate_gallery,
        [image_input, query_input, checkpoint_selector],
        gallery
    )
```

From [Gradio Docs - Multipage Apps](https://www.gradio.app/guides/multipage-apps) (accessed 2025-10-31):
- Multipage pattern useful for organizing checkpoint comparisons by experiment
- Each page can show different checkpoint groups
- Shared backend ensures consistent model loading across pages

---

## Section 3: Result Comparison Visualization

### Side-by-Side Output Display

Visual diff highlighting for text outputs:

```python
import difflib

def highlight_differences(text_a, text_b):
    """
    Highlight differences between two text outputs

    Returns HTML with highlighted differences
    """
    diff = difflib.ndiff(text_a.split(), text_b.split())

    html_a = []
    html_b = []

    for token in diff:
        if token.startswith('  '):  # Common
            word = token[2:]
            html_a.append(word)
            html_b.append(word)
        elif token.startswith('- '):  # Only in A
            word = token[2:]
            html_a.append(f'<mark style="background-color: #ffcccc;">{word}</mark>')
        elif token.startswith('+ '):  # Only in B
            word = token[2:]
            html_b.append(f'<mark style="background-color: #ccffcc;">{word}</mark>')

    return ' '.join(html_a), ' '.join(html_b)

def compare_with_diff(image, query, ckpt_a, ckpt_b):
    """Compare two checkpoints with difference highlighting"""
    model_a = checkpoint_manager.load_checkpoint(ckpt_a)
    model_b = checkpoint_manager.load_checkpoint(ckpt_b)

    output_a = model_a.generate(image, query)
    output_b = model_b.generate(image, query)

    # Highlight differences
    diff_a, diff_b = highlight_differences(output_a, output_b)

    return output_a, output_b, diff_a, diff_b

with gr.Blocks() as demo:
    with gr.Row():
        ckpt_a = gr.Dropdown(choices=available_checkpoints, label="Checkpoint A")
        ckpt_b = gr.Dropdown(choices=available_checkpoints, label="Checkpoint B")

    compare_btn = gr.Button("Compare with Diff")

    # Raw outputs
    with gr.Row():
        output_a = gr.Textbox(label="Output A (Raw)")
        output_b = gr.Textbox(label="Output B (Raw)")

    # Highlighted differences
    gr.Markdown("### Differences Highlighted")
    with gr.Row():
        diff_a = gr.HTML(label="Output A (Highlighted)")
        diff_b = gr.HTML(label="Output B (Highlighted)")

    compare_btn.click(
        compare_with_diff,
        [image_input, query_input, ckpt_a, ckpt_b],
        [output_a, output_b, diff_a, diff_b]
    )
```

### Metric Comparison (Accuracy, Latency)

Quantitative comparison with visualizations:

```python
import time
import numpy as np

def benchmark_checkpoints(test_images, queries, checkpoint_list):
    """
    Benchmark multiple checkpoints on test set

    Returns metrics for each checkpoint
    """
    results = {}

    for ckpt_name, ckpt_path in checkpoint_list:
        model = checkpoint_manager.load_checkpoint(ckpt_path)

        inference_times = []
        outputs = []

        for img, query in zip(test_images, queries):
            start = time.time()
            output = model.generate(img, query)
            inference_times.append(time.time() - start)
            outputs.append(output)

        results[ckpt_name] = {
            "avg_latency_ms": np.mean(inference_times) * 1000,
            "std_latency_ms": np.std(inference_times) * 1000,
            "min_latency_ms": np.min(inference_times) * 1000,
            "max_latency_ms": np.max(inference_times) * 1000,
            "outputs": outputs
        }

    return results

def create_comparison_table(benchmark_results):
    """Format benchmark results as table"""
    rows = []

    for ckpt_name, metrics in benchmark_results.items():
        rows.append([
            ckpt_name,
            f"{metrics['avg_latency_ms']:.2f}",
            f"{metrics['std_latency_ms']:.2f}",
            f"{metrics['min_latency_ms']:.2f}",
            f"{metrics['max_latency_ms']:.2f}"
        ])

    return rows

with gr.Blocks() as demo:
    gr.Markdown("# Checkpoint Benchmarking")

    test_set_upload = gr.File(label="Upload Test Images (zip)")
    queries_upload = gr.Textbox(label="Queries (one per line)", lines=5)

    checkpoint_selector = gr.CheckboxGroup(
        choices=[name for name, _ in available_checkpoints],
        label="Select Checkpoints to Benchmark"
    )

    benchmark_btn = gr.Button("Run Benchmark")

    results_table = gr.Dataframe(
        headers=["Checkpoint", "Avg Latency (ms)", "Std Dev (ms)", "Min (ms)", "Max (ms)"],
        label="Benchmark Results"
    )

    latency_plot = gr.BarPlot(
        x="Checkpoint",
        y="Avg Latency (ms)",
        title="Average Inference Latency Comparison"
    )
```

### Checkpoint Performance Over Time

Timeline visualization for training progression:

```python
import plotly.graph_objects as go

def plot_checkpoint_timeline(checkpoint_dir):
    """
    Create timeline plot showing checkpoint metrics over training
    """
    checkpoints = discover_checkpoints(checkpoint_dir)

    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    checkpoint_names = []

    for name, path in checkpoints:
        metadata = extract_checkpoint_metadata(path)

        if metadata['epoch'] != "Unknown":
            epochs.append(metadata['epoch'])
            train_losses.append(metadata.get('train_loss', 0))
            val_losses.append(metadata.get('val_loss', 0))
            val_accuracies.append(metadata.get('val_accuracy', 0))
            checkpoint_names.append(name)

    fig = go.Figure()

    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='blue')
    ))

    # Validation loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Val Loss',
        line=dict(color='red')
    ))

    # Validation accuracy (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_accuracies,
        mode='lines+markers',
        name='Val Accuracy',
        line=dict(color='green'),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Checkpoint Performance Timeline',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        yaxis2=dict(
            title='Accuracy',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )

    return fig

with gr.Blocks() as demo:
    gr.Markdown("# Training Timeline")

    refresh_btn = gr.Button("Refresh Timeline")
    timeline_plot = gr.Plot(label="Checkpoint Metrics Over Time")

    def update_timeline():
        return plot_checkpoint_timeline("checkpoints/")

    refresh_btn.click(update_timeline, None, timeline_plot)
    demo.load(update_timeline, None, timeline_plot)
```

From [Freepik Flux.1 Lite - Checkpoint Comparison](https://huggingface.co/Freepik/flux.1-lite-8B-alpha) (accessed 2025-10-31):
- Example: Flux.1 Lite used checkpoint comparison to show 23% speedup vs original
- Pattern: Display performance metrics (RAM usage, inference speed) alongside outputs
- Visual comparison essential for demonstrating model improvements

---

## Section 4: Checkpoint Metadata & History

### Display Training Epoch, Validation Score

Comprehensive metadata viewer:

```python
def create_metadata_viewer(ckpt_path):
    """Create detailed metadata display for checkpoint"""
    metadata = extract_checkpoint_metadata(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Format comprehensive info
    info_sections = []

    # Training progress
    training_info = f"""
### Training Progress
- **Epoch:** {metadata['epoch']}
- **Step:** {metadata['step']}
- **Train Loss:** {metadata['train_loss']}
- **Val Loss:** {metadata['val_loss']}
- **Val Accuracy:** {metadata['val_accuracy']}
"""
    info_sections.append(training_info)

    # Model configuration
    if metadata['config']:
        config_items = [f"- **{k}:** {v}" for k, v in metadata['config'].items()]
        config_info = "### Model Configuration\n" + "\n".join(config_items)
        info_sections.append(config_info)

    # Optimizer state
    if 'optimizer_state' in checkpoint:
        opt_state = checkpoint['optimizer_state']
        optimizer_info = f"""
### Optimizer State
- **Learning Rate:** {opt_state.get('param_groups', [{}])[0].get('lr', 'N/A')}
- **Weight Decay:** {opt_state.get('param_groups', [{}])[0].get('weight_decay', 'N/A')}
"""
        info_sections.append(optimizer_info)

    # Model architecture summary
    if 'model_state_dict' in checkpoint:
        num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        arch_info = f"""
### Architecture
- **Total Parameters:** {num_params:,}
- **Parameter Groups:** {len(checkpoint['model_state_dict'])}
"""
        info_sections.append(arch_info)

    return "\n\n".join(info_sections)

with gr.Blocks() as demo:
    gr.Markdown("# Checkpoint Inspector")

    checkpoint_dropdown = gr.Dropdown(
        choices=available_checkpoints,
        label="Select Checkpoint to Inspect"
    )

    metadata_display = gr.Markdown(label="Checkpoint Details")

    checkpoint_dropdown.change(
        create_metadata_viewer,
        checkpoint_dropdown,
        metadata_display
    )
```

### Timeline Visualization

Interactive timeline for checkpoint history:

```python
def create_interactive_timeline(checkpoint_dir):
    """
    Create interactive timeline with checkpoint selection
    """
    checkpoints = discover_checkpoints(checkpoint_dir)

    # Extract timeline data
    timeline_data = []
    for name, path in checkpoints:
        metadata = extract_checkpoint_metadata(path)
        timeline_data.append({
            "name": name,
            "path": path,
            "epoch": metadata['epoch'],
            "val_loss": metadata['val_loss'],
            "timestamp": metadata['timestamp']
        })

    # Sort by epoch
    timeline_data.sort(key=lambda x: x['epoch'] if x['epoch'] != "Unknown" else 0)

    return timeline_data

with gr.Blocks() as demo:
    gr.Markdown("# Checkpoint Timeline")

    # Timeline display with clickable checkpoints
    def render_timeline():
        timeline = create_interactive_timeline("checkpoints/")

        with gr.Column():
            for i, ckpt in enumerate(timeline):
                with gr.Row():
                    gr.Markdown(f"**Epoch {ckpt['epoch']}**")
                    select_btn = gr.Button(f"Select", size="sm")
                    gr.Markdown(f"Val Loss: {ckpt['val_loss']}")

                    # Wire up selection button
                    # This would trigger loading that checkpoint

    demo.load(render_timeline)
```

### Best Checkpoint Recommendation

Automatic selection of optimal checkpoint:

```python
def recommend_best_checkpoint(checkpoint_dir, metric="val_loss", mode="min"):
    """
    Recommend best checkpoint based on metric

    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to optimize (e.g., "val_loss", "val_accuracy")
        mode: "min" for loss, "max" for accuracy

    Returns:
        Tuple of (best_checkpoint_path, metadata, reason)
    """
    checkpoints = discover_checkpoints(checkpoint_dir)

    best_ckpt = None
    best_value = float('inf') if mode == "min" else float('-inf')

    for name, path in checkpoints:
        metadata = extract_checkpoint_metadata(path)
        value = metadata.get(metric, None)

        if value is None or value == "Unknown":
            continue

        if mode == "min" and value < best_value:
            best_value = value
            best_ckpt = (name, path, metadata)
        elif mode == "max" and value > best_value:
            best_value = value
            best_ckpt = (name, path, metadata)

    if best_ckpt is None:
        return None, None, "No valid checkpoints found"

    reason = f"Selected based on {mode}imum {metric}: {best_value:.4f}"
    return best_ckpt[1], best_ckpt[2], reason

with gr.Blocks() as demo:
    gr.Markdown("# Best Checkpoint Selector")

    with gr.Row():
        metric_selector = gr.Radio(
            choices=["val_loss", "val_accuracy", "train_loss"],
            label="Optimize for",
            value="val_loss"
        )
        mode_selector = gr.Radio(
            choices=["min", "max"],
            label="Mode",
            value="min"
        )

    find_best_btn = gr.Button("Find Best Checkpoint")

    best_ckpt_display = gr.Textbox(label="Best Checkpoint", interactive=False)
    metadata_display = gr.Markdown(label="Details")
    reason_display = gr.Textbox(label="Reason", interactive=False)

    def find_and_display_best(metric, mode):
        path, metadata, reason = recommend_best_checkpoint(
            "checkpoints/", metric, mode
        )

        if path is None:
            return "", "No checkpoints found", reason

        metadata_str = format_metadata_display(metadata)
        return path, metadata_str, reason

    find_best_btn.click(
        find_and_display_best,
        [metric_selector, mode_selector],
        [best_ckpt_display, metadata_display, reason_display]
    )
```

### Export Comparison Results

Save comparison results for later analysis:

```python
import json
import csv
from datetime import datetime

def export_comparison_results(results, format="json"):
    """
    Export checkpoint comparison results

    Args:
        results: Dictionary of comparison results
        format: "json" or "csv"

    Returns:
        File path to exported results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "json":
        filename = f"comparison_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    elif format == "csv":
        filename = f"comparison_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["Checkpoint", "Output", "Inference Time (ms)", "Metadata"])

            # Data rows
            for ckpt_name, data in results.items():
                writer.writerow([
                    ckpt_name,
                    data.get("output", ""),
                    data.get("inference_time_ms", ""),
                    json.dumps(data.get("metadata", {}))
                ])

    return filename

with gr.Blocks() as demo:
    # ... comparison interface ...

    export_format = gr.Radio(
        choices=["json", "csv"],
        label="Export Format",
        value="json"
    )

    export_btn = gr.Button("Export Results")
    export_status = gr.Textbox(label="Export Status", interactive=False)
    download_link = gr.File(label="Download Results")

    def export_and_download(format_choice):
        # Assuming results are stored in state
        filename = export_comparison_results(comparison_results, format_choice)
        return f"Exported to {filename}", filename

    export_btn.click(
        export_and_download,
        export_format,
        [export_status, download_link]
    )
```

From [GitHub Trackio - Experiment Tracking](https://github.com/gradio-app/trackio) (accessed 2025-10-31):
- Trackio provides experiment tracking with Gradio dashboard
- Pattern: Log metrics for each checkpoint, visualize in dashboard
- Integration: `trackio.init()` for tracking, `trackio.log()` for metrics, `trackio.show()` for dashboard
- Useful for comparing checkpoint performance over entire training run

---

## Sources

**Web Research:**
- [Freepik Flux.1 Lite Model](https://huggingface.co/Freepik/flux.1-lite-8B-alpha) - Hugging Face (accessed 2025-10-31)
  - Alpha 8B checkpoint comparison demo
  - Performance metrics visualization (23% faster, 7GB less RAM)
  - ComfyUI workflow for checkpoint comparison

- [Gradio Multipage Apps Guide](https://www.gradio.app/guides/multipage-apps) - Gradio Official Docs (accessed 2025-10-31)
  - Multi-page pattern for organizing checkpoint groups
  - Shared backend for consistent model loading
  - Per-page navbar configuration

- [GitHub - Trackio Experiment Tracking](https://github.com/gradio-app/trackio) - Gradio App (accessed 2025-10-31)
  - Lightweight experiment tracking with Gradio
  - Dashboard for visualizing checkpoint metrics
  - Integration patterns: `trackio.init()`, `trackio.log()`, `trackio.show()`

- [GitHub Issue - Multiple Checkpoint Loading](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/445) - Stable Diffusion WebUI Forge (accessed 2025-10-31)
  - Bug report: Unable to load multiple checkpoints simultaneously
  - Solution: Implement lazy loading and cache management
  - Best practice: Auto-evict least recently used checkpoints

**Additional References:**
- [Gradio Interface Documentation](https://www.gradio.app/docs/gradio/interface) - Interface class reference
- [Gradio Blocks Documentation](https://www.gradio.app/docs/gradio/blocks) - Blocks API for complex layouts
- [Reddit - Checkpoint Comparison Discussion](https://www.reddit.com/r/StableDiffusion/comments/163ybib/noob_question_gradio_login_to_access_different/) - Community patterns

---

**Key Takeaways:**

1. **Lazy Loading Essential**: Load checkpoints on-demand with LRU cache (max 2-3 for 16GB GPU)
2. **Metadata Display Critical**: Show epoch, validation scores, inference time for informed selection
3. **Side-by-Side Comparison**: Visual diff highlighting helps identify output differences
4. **Timeline Visualization**: Plot metrics over training to identify best checkpoint
5. **Export Results**: Save comparison data (JSON/CSV) for reproducibility
6. **Auto-Discovery**: Scan checkpoint directory, extract metadata from filenames/files
7. **Batch Comparison**: Run same query on multiple checkpoints for consistent evaluation

**ARR-COC-Specific Applications:**
- Compare relevance realization patterns across training epochs
- Visualize LOD allocation differences between checkpoints
- A/B test opponent processing weight changes
- Benchmark inference speed for different compression ratios
- Select optimal checkpoint for deployment based on speed/quality tradeoff
