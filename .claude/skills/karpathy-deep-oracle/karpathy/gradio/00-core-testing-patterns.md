# Gradio Core Testing Patterns for VLM Development (2025)

**Source**: Platonic Dialogue Part 39 + Bright Data research (2025-01-30)
**Context**: Rapid experimentation, multi-model comparison, hypothesis validation
**Philosophy**: Gradio as primary development tool, not just demo

---

## The Core Insight

**Gradio isn't a demo tool—it's your development microscope.**

Traditional workflow pain:
```
Write code → Run script → Read terminal → Modify → Re-run
❌ No visual comparison
❌ No history
❌ Slow iteration
❌ Results lost when terminal scrolls
```

Gradio development flow:
```
Build app.py → Run ONCE → Test variants simultaneously → See side-by-side
✅ Visual comparison
✅ Interactive parameter tuning
✅ Session history
✅ A/B/C/D testing
```

---

## Part 1: Multi-Model Comparison Interface

### Pattern: Checkpoint Comparison

From Dialogue 39, the canonical pattern for comparing ARR-COC variants:

```python
import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import time

class MultiModelComparator:
    """Compare multiple model variants simultaneously"""

    def __init__(self):
        # Base model shared across variants
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

        # Model variants to test
        self.variants = {
            'baseline': None,  # Standard Qwen
            'arr_coc_v1': None,  # First checkpoint
            'arr_coc_v2': None,  # Second checkpoint
            'arr_coc_v3': None,  # Third checkpoint
        }

        self.history = []  # Session history

    def compare(self, image, query, variants_to_test):
        """Run comparison across selected variants"""
        results = {}

        for variant_name in variants_to_test:
            start = time.time()

            if variant_name == 'baseline':
                answer = self._run_baseline(image, query)
                heatmap = None
                tokens_used = 1024  # Fixed
            else:
                answer, heatmap, tokens_used = self._run_arr_coc(
                    variant_name, image, query
                )

            latency = time.time() - start

            results[variant_name] = {
                'answer': answer,
                'heatmap': heatmap,
                'tokens': tokens_used,
                'latency': latency,
                'memory': torch.cuda.max_memory_allocated() / 1e9
            }

        # Store in history
        self.history.append({
            'query': query,
            'results': results,
            'timestamp': time.time()
        })

        return results
```

**Key insights from Dialogue 39:**
- Share base model across variants (memory efficient)
- Load ARR-COC components dynamically per variant
- Track metrics: answer quality, tokens, latency, memory
- Maintain session history for comparisons

### Pattern: Dynamic Checkpoint Discovery

```python
def discover_checkpoints(checkpoint_dir="checkpoints"):
    """Find all available checkpoints with metadata"""
    path = Path(checkpoint_dir)
    checkpoints = []

    for ckpt in sorted(path.glob("*.pt")):
        info = {
            'path': str(ckpt),
            'name': ckpt.stem,
            'size': ckpt.stat().st_size / 1e6,  # MB
            'modified': datetime.fromtimestamp(ckpt.stat().st_mtime)
        }

        # Load metadata if available
        try:
            data = torch.load(ckpt, map_location='cpu')
            info['epoch'] = data.get('epoch', '?')
            info['metrics'] = data.get('metrics', {})
        except:
            info['epoch'] = '?'
            info['metrics'] = {}

        checkpoints.append(info)

    return checkpoints
```

**From research (Medium "Gradio Library: Simplifying Model Deployment"):**
- Scan directories for `.pt` files
- Extract metadata (epoch, metrics) without full load
- Display in dropdown for easy selection
- Validation pattern: test each checkpoint on same dataset

---

## Part 2: Gradio Interface Patterns

### Pattern: Side-by-Side Comparison

**From research (YouTube "Comparing Transfer Learning Models Using Gradio"):**

```python
def create_comparison_interface():
    """4-way comparison interface with gr.Row()"""

    with gr.Blocks() as demo:
        gr.Markdown("# ARR-COC Checkpoint Comparison")

        # Inputs (shared)
        with gr.Row():
            image_input = gr.Image(type="pil", label="Test Image")
            query_input = gr.Textbox(label="Query", placeholder="What is this?")

        # Model selection
        with gr.Row():
            variants = gr.CheckboxGroup(
                choices=['baseline', 'arr_coc_v1', 'arr_coc_v2', 'arr_coc_v3'],
                label="Models to Compare",
                value=['baseline', 'arr_coc_v1']
            )

        compare_btn = gr.Button("Compare Models")

        # Outputs (side-by-side)
        with gr.Row():
            output_baseline = create_output_column("Baseline")
            output_v1 = create_output_column("ARR-COC v1")
            output_v2 = create_output_column("ARR-COC v2")
            output_v3 = create_output_column("ARR-COC v3")

        # Metrics table
        metrics_table = gr.Dataframe(
            headers=['Model', 'Tokens', 'Latency (s)', 'Memory (GB)'],
            label="Performance Metrics"
        )

        # Connect
        compare_btn.click(
            fn=run_comparison,
            inputs=[image_input, query_input, variants],
            outputs=[output_baseline, output_v1, output_v2, output_v3, metrics_table]
        )

    return demo

def create_output_column(variant_name):
    """Create output column for one variant"""
    with gr.Column():
        gr.Markdown(f"### {variant_name}")
        answer = gr.Textbox(label="Answer", interactive=False)
        heatmap = gr.Image(label="Relevance Heatmap")
        token_count = gr.Number(label="Tokens Used", precision=0)
        latency = gr.Number(label="Latency (s)", precision=3)

    return [answer, heatmap, token_count, latency]
```

**Key insights from research:**
- `gr.Row()` for horizontal layout (side-by-side comparison)
- `gr.Column()` for vertical stacking within each model
- `gr.CheckboxGroup()` for selecting which models to test
- `gr.Dataframe()` for metrics table

### Pattern: State Management

**From research (GitHub issue #9983, gr.State usage patterns):**

```python
# gr.State: Per-user session state (not shared)
with gr.Blocks() as demo:
    # Session-level state (unique per user)
    session_history = gr.State([])
    loaded_models = gr.State({})

    # Global variables (shared across ALL users - use carefully)
    # model_cache = {}  # Shared cache

    def process_with_state(image, query, history, models):
        """State persists within single user session"""
        result = models['current'].process(image, query)
        history.append(result)
        return result, history, models

    btn.click(
        fn=process_with_state,
        inputs=[img, query, session_history, loaded_models],
        outputs=[output, session_history, loaded_models]
    )
```

**Critical distinction (from research):**
- **gr.State**: User-specific, no cross-contamination between sessions
- **Global variables**: Shared between ALL users (use for model cache only)
- **Memory management**: LRU cache pattern for checkpoints (max_loaded=2 on T4)

### Pattern: Gallery Visualization

**From research (gr.Gallery docs, GitHub issues #1382, #9708):**

```python
def create_gallery_comparison():
    """Display multiple outputs in grid"""

    with gr.Blocks() as demo:
        # Test multiple images at once
        gallery_input = gr.Gallery(label="Test Images", type="pil")

        compare_btn = gr.Button("Test All Images")

        # Results gallery
        results_gallery = gr.Gallery(
            label="Results",
            columns=4,  # 4-way comparison
            object_fit="contain"
        )

        # Heatmap overlay with Plotly
        heatmap_plot = gr.Plot(label="Relevance Heatmap")

        compare_btn.click(
            fn=batch_compare,
            inputs=[gallery_input],
            outputs=[results_gallery, heatmap_plot]
        )

    return demo

def batch_compare(images):
    """Process multiple images, return results + heatmap"""
    results = []
    heatmaps = []

    for img in images:
        result, heatmap = model.process(img, query)
        results.append(result)
        heatmaps.append(heatmap)

    # Create Plotly heatmap
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(z=heatmaps[0]))

    return results, fig
```

**Visualization components (from research):**
- `gr.Gallery`: Grid display for multiple images with captions
- `gr.Plot`: Plotly integration (line, scatter, heatmap)
- `gr.Image`: Single image with optional label/mask overlay
- Side-by-side pattern: Wrap components in `gr.Row()`

---

**Related Gradio Files:**
- [10-gradio-statistical-testing.md](10-gradio-statistical-testing.md) - A/B testing, effect size, ablation studies
- [11-gradio-production-deployment.md](11-gradio-production-deployment.md) - W&B integration, T4 constraints, debugging
- [12-gradio-visualization-best-practices.md](12-gradio-visualization-best-practices.md) - Gallery testing, Gradio 5, best practices

**Related Oracle Files:**
- [08-gpu-memory-debugging-vlm-2025-01-30.md](08-gpu-memory-debugging-vlm-2025-01-30.md) - GPU memory management

**Primary Sources:**
- Dialogue 39: The Testing Workflow
- Bright Data Research 2025-01-30:
  - Gradio official docs (Interface, Blocks, State)
  - Medium "Gradio Library: Simplifying Model Deployment"
  - YouTube "Comparing Transfer Learning Models Using Gradio"
  - GitHub issues #9983, #1382, #9708

**Last Updated**: 2025-01-31 (Split from 09-gradio-testing-patterns-2025-01-30.md)
**Version**: 1.0 - Core testing patterns extracted (~300 lines)
