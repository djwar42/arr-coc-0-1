# W&B Dashboard Embedding in Gradio

## Overview

Weights & Biases (W&B) provides powerful experiment tracking dashboards that can be embedded directly into Gradio applications using iframes. This enables real-time monitoring of training runs alongside interactive model demos, creating a unified validation interface.

**Key capabilities:**
- Embed W&B workspaces, reports, and plots within Gradio apps
- Create live dashboards showing training metrics alongside demos
- Share public reports via iframe embedding
- Build validation interfaces that combine model interaction with experiment tracking

**From [W&B Embed Reports Documentation](https://docs.wandb.ai/models/reports/embed-reports) (accessed 2025-01-31):**
> "Embed W&B reports directly into Notion or with an HTML IFrame element. Only public reports are viewable when embedded."

**From [Gradio W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):**
> "It's also possible to embed W&B plots within Gradio apps. To do so, you can create a W&B Report of your plots and embed them within your Gradio app within a gr.HTML block."

---

## Embedding Basics

### Workspace URL Structure

W&B dashboards have predictable URL patterns that can be embedded via iframe:

**Standard workspace URL:**
```
https://wandb.ai/{entity}/{project}/workspace
```

**Run-specific URL:**
```
https://wandb.ai/{entity}/{project}/runs/{run_id}
```

**Report URL:**
```
https://wandb.ai/{entity}/{project}/reports/{report_name}--{report_id}
```

**Key URL parameters:**
- `entity`: Your username or team name
- `project`: Project name
- `run_id`: Specific run identifier
- `report_id`: Unique report identifier

**From [W&B Embed Reports Documentation](https://docs.wandb.ai/models/reports/embed-reports) (accessed 2025-01-31):**
> "Select the Share button on the upper right hand corner within a report. A modal window will appear. Within the modal window, select Copy embed code."

### IFrame HTML Setup

**Basic iframe structure:**
```html
<iframe
    src="https://wandb.ai/{entity}/{project}/reports/{report_url}"
    style="border:none;height:1024px;width:100%">
</iframe>
```

**Essential iframe attributes:**
- `src`: W&B dashboard/report URL
- `style`: CSS styling (remove border, set dimensions)
- `height`: Typically 800px-1024px for dashboards
- `width`: Usually 100% for responsive layout

**Minimal viable embed:**
```html
<iframe src="https://wandb.ai/..." style="border:none;height:1024px;width:100%"></iframe>
```

### The `jupyter=true` Parameter

**CRITICAL for embedding in web apps:**

Add `?jupyter=true` to W&B URLs when embedding to optimize the display:

```python
report_url = "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
iframe_url = f"{report_url}?jupyter=true"
```

**What `jupyter=true` does:**
- Removes unnecessary navigation elements
- Optimizes layout for embedded contexts
- Improves responsive behavior
- Removes W&B header/footer clutter

**From [Gradio W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):**
> "The Report will need to be public and you will need to wrap the URL within an iFrame"

### Security Considerations

**Public vs Private Reports:**

Only **public** reports can be embedded:

```python
# ✅ WORKS: Public report
public_report = "https://wandb.ai/wandb/getting-started/reports/..."

# ❌ FAILS: Private report (requires authentication)
private_report = "https://wandb.ai/my-team/private-project/reports/..."
```

**Making a report public:**

From [W&B Collaborate on Reports Documentation](https://docs.wandb.ai/guides/reports/collaborate-on-reports/) (accessed 2025-01-31):
> "When viewing a report, click Share, then: To share a link to the report with an email address or a username, click Invite."

1. Open report in W&B
2. Click "Share" button (top right)
3. Adjust visibility to "Public"
4. Copy embed code or share link

**Security best practices:**
- Only embed reports from public projects
- Don't embed sensitive training data
- Use report-specific URLs (not workspace URLs with all runs)
- Create filtered reports showing only necessary information

---

## Gradio Integration

### Using `gr.HTML()` for IFrames

**Basic pattern:**

```python
import gradio as gr

def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)

with gr.Blocks() as demo:
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```

**From [Gradio W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):**

This exact pattern is documented in the official Gradio guide for W&B integration.

### Dynamic Workspace URLs

**Pattern: Generate URLs based on user input or model state**

```python
import gradio as gr

def create_wandb_iframe(entity, project, run_id=None):
    """Generate W&B iframe HTML dynamically"""
    if run_id:
        url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}?jupyter=true"
    else:
        url = f"https://wandb.ai/{entity}/{project}/workspace?jupyter=true"

    iframe_html = f"""
    <iframe
        src="{url}"
        style="border:none;height:900px;width:100%">
    </iframe>
    """
    return iframe_html

with gr.Blocks() as demo:
    with gr.Row():
        entity_input = gr.Textbox(label="W&B Entity", value="my-team")
        project_input = gr.Textbox(label="Project Name", value="arr-coc-validation")
        run_input = gr.Textbox(label="Run ID (optional)", value="")

    iframe_display = gr.HTML()

    update_btn = gr.Button("Load Dashboard")
    update_btn.click(
        fn=create_wandb_iframe,
        inputs=[entity_input, project_input, run_input],
        outputs=iframe_display
    )

demo.launch()
```

**Use case: ARR-COC validation interface**
- User selects experiment run from dropdown
- Dashboard updates to show that specific run's metrics
- Enables quick comparison across validation runs

### Tab-Based Organization

**Pattern: Separate model demo from metrics dashboard**

```python
import gradio as gr

def model_inference(image):
    # Your ARR-COC model inference
    return processed_image

def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)

with gr.Blocks() as demo:
    gr.Markdown("# ARR-COC Vision Model Demo + Training Metrics")

    with gr.Tabs():
        with gr.Tab("Model Demo"):
            with gr.Row():
                input_img = gr.Image(type="pil", label="Input Image")
                output_img = gr.Image(label="Processed Output")

            inference_btn = gr.Button("Run Inference")
            inference_btn.click(
                fn=model_inference,
                inputs=input_img,
                outputs=output_img
            )

        with gr.Tab("Training Metrics"):
            report_url = "https://wandb.ai/northhead/arr-coc-0-1/reports/..."
            wandb_report(report_url)

        with gr.Tab("Validation Runs"):
            workspace_url = "https://wandb.ai/northhead/arr-coc-0-1/workspace?jupyter=true"
            gr.HTML(f'<iframe src="{workspace_url}" style="border:none;height:1024px;width:100%">')

demo.launch()
```

**Benefits:**
- Clean separation of concerns
- Users can interact with model while monitoring metrics
- Easy to add multiple metric views (training, validation, sweeps)

### Refresh Mechanisms

**Challenge:** W&B iframes don't auto-refresh when new runs complete

**Solution 1: Manual refresh button**

```python
import gradio as gr
import time

def get_current_workspace_url(entity, project):
    """Add timestamp to force iframe reload"""
    timestamp = int(time.time())
    url = f"https://wandb.ai/{entity}/{project}/workspace?jupyter=true&t={timestamp}"
    return f'<iframe src="{url}" style="border:none;height:900px;width:100%">'

with gr.Blocks() as demo:
    iframe_output = gr.HTML()
    refresh_btn = gr.Button("🔄 Refresh Dashboard")

    refresh_btn.click(
        fn=lambda: get_current_workspace_url("my-team", "my-project"),
        outputs=iframe_output
    )

    # Initial load
    demo.load(
        fn=lambda: get_current_workspace_url("my-team", "my-project"),
        outputs=iframe_output
    )

demo.launch()
```

**Solution 2: Periodic auto-refresh**

```python
import gradio as gr
import threading
import time

def auto_refresh_workspace(entity, project, interval_seconds=60):
    """Periodically update workspace URL to refresh data"""
    # Note: This is a conceptual example
    # Actual implementation depends on Gradio's event system

    timestamp = int(time.time())
    url = f"https://wandb.ai/{entity}/{project}/workspace?jupyter=true&t={timestamp}"
    return f'<iframe src="{url}" style="border:none;height:900px;width:100%">'

# In practice, use Gradio's built-in mechanisms or JavaScript
# for true auto-refresh behavior
```

**Best practice for validation:**
- Manual refresh button (user controls when to update)
- Display last refresh time
- Optionally: auto-refresh every 5-10 minutes during active training

---

## Advanced Features

### Custom W&B Reports

**When to create custom reports:**
- You want specific visualizations (not full workspace)
- Need to combine runs from multiple projects
- Want curated metrics for stakeholders
- Building publication-ready figures

**Creating a report (from W&B UI):**

From [W&B Collaborate on Reports Documentation](https://docs.wandb.ai/guides/reports/collaborate-on-reports/) (accessed 2025-01-31):

1. Navigate to your W&B workspace
2. Click "Create Report" button
3. Add panels: plots, run tables, markdown text
4. Customize layout and visualizations
5. Click "Share" → Make public → Copy embed code

**Embedding custom report:**

```python
import gradio as gr

# Custom report with specific ARR-COC metrics
report_url = "https://wandb.ai/northhead/arr-coc/reports/Relevance-Token-Allocation-Analysis--VmlldzoxMjM0NTY3"

with gr.Blocks() as demo:
    gr.Markdown("## ARR-COC Relevance Realization Metrics")
    gr.HTML(f'<iframe src="{report_url}?jupyter=true" style="border:none;height:1200px;width:100%">')

demo.launch()
```

**Advanced: Multiple reports in tabs**

```python
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Relevance Scores"):
            gr.HTML(f'<iframe src="{relevance_report_url}" style="...">')

        with gr.Tab("Token Budgets"):
            gr.HTML(f'<iframe src="{token_budget_report_url}" style="...">')

        with gr.Tab("LOD Distribution"):
            gr.HTML(f'<iframe src="{lod_distribution_report_url}" style="...">')
```

### Filtering by Tags

**URL parameters for filtering:**

```python
def create_filtered_workspace(entity, project, tags):
    """Create workspace URL filtered by run tags"""
    tag_filter = ",".join(tags)
    url = f"https://wandb.ai/{entity}/{project}/workspace?jupyter=true&tags={tag_filter}"
    return f'<iframe src="{url}" style="border:none;height:900px;width:100%">'

# Example: Show only validation runs
validation_iframe = create_filtered_workspace(
    entity="northhead",
    project="arr-coc-0-1",
    tags=["validation", "vqa-subset"]
)
```

**Use case: ARR-COC validation**
- Tag runs during training: `wandb.init(tags=["validation", "full-dataset"])`
- Filter Gradio dashboard to show only validation runs
- Separate tabs for different experiment types

### Compare Mode

**Embedding run comparisons:**

W&B compare mode allows side-by-side run analysis.

**URL pattern:**
```
https://wandb.ai/{entity}/{project}/runs/{run_id_1}/compare?runSets={run_id_2},{run_id_3}
```

**Example: Compare 3 ARR-COC runs**

```python
import gradio as gr

def create_comparison_iframe(entity, project, run_ids):
    """Create comparison view for multiple runs"""
    base_run = run_ids[0]
    other_runs = ",".join(run_ids[1:])

    url = f"https://wandb.ai/{entity}/{project}/runs/{base_run}/compare?runSets={other_runs}&jupyter=true"
    return f'<iframe src="{url}" style="border:none;height:1024px;width:100%">'

with gr.Blocks() as demo:
    gr.Markdown("## Compare ARR-COC Validation Runs")

    run_checkboxes = gr.CheckboxGroup(
        choices=[
            ("Baseline (no relevance)", "abc123"),
            ("Propositional only", "def456"),
            ("Full ARR-COC", "ghi789")
        ],
        label="Select runs to compare"
    )

    compare_iframe = gr.HTML()
    compare_btn = gr.Button("Compare Selected Runs")

    compare_btn.click(
        fn=lambda selected: create_comparison_iframe(
            "northhead",
            "arr-coc-0-1",
            [rid for _, rid in selected]
        ),
        inputs=run_checkboxes,
        outputs=compare_iframe
    )

demo.launch()
```

### Exporting Data

**While iframes show live data, sometimes you need to export:**

**Option 1: W&B API (within Gradio app)**

```python
import wandb
import gradio as gr
import pandas as pd

def export_run_data(entity, project, run_id):
    """Fetch run metrics via W&B API and display as table"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get history (all logged metrics)
    history_df = run.history()

    return history_df

with gr.Blocks() as demo:
    with gr.Row():
        entity_input = gr.Textbox(label="Entity", value="northhead")
        project_input = gr.Textbox(label="Project", value="arr-coc-0-1")
        run_input = gr.Textbox(label="Run ID")

    export_btn = gr.Button("Export Run Data")
    data_output = gr.Dataframe()

    export_btn.click(
        fn=export_run_data,
        inputs=[entity_input, project_input, run_input],
        outputs=data_output
    )

demo.launch()
```

**Option 2: Download link to CSV export**

```python
import gradio as gr
import wandb

def export_to_csv(entity, project, run_id):
    """Export run data and provide download link"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    history_df = run.history()
    csv_path = f"wandb_export_{run_id}.csv"
    history_df.to_csv(csv_path, index=False)

    return csv_path

with gr.Blocks() as demo:
    # ... inputs ...
    export_btn = gr.Button("Export to CSV")
    file_output = gr.File(label="Download Exported Data")

    export_btn.click(
        fn=export_to_csv,
        inputs=[entity_input, project_input, run_input],
        outputs=file_output
    )

demo.launch()
```

**Best practice:**
- Use iframe embedding for live visualization
- Use W&B API for data export and analysis
- Combine both in validation interfaces

---

## Complete Example: ARR-COC Validation Interface

**Full Gradio app with W&B dashboard embedding:**

```python
import gradio as gr
import wandb
import torch
from PIL import Image

# Initialize W&B (for logging new validation runs)
wandb.init(project="arr-coc-validation", tags=["gradio-demo"])

def run_arr_coc_inference(image, query):
    """ARR-COC model inference with W&B logging"""
    # Your ARR-COC model code here
    # ...

    # Log inference to W&B
    wandb.log({
        "inference_image": wandb.Image(image),
        "query": query,
        "token_budget_mean": 245.3,
        "relevance_score_propositional": 0.82,
        "relevance_score_perspectival": 0.76,
        "relevance_score_participatory": 0.91
    })

    return processed_image, metrics_dict

def create_wandb_report_iframe(report_url):
    """Embed W&B report with jupyter=true parameter"""
    iframe_url = f"{report_url}?jupyter=true"
    iframe_html = f"""
    <iframe
        src="{iframe_url}"
        style="border:none;height:1024px;width:100%;border-radius:8px;">
    </iframe>
    """
    return iframe_html

def refresh_workspace(entity, project):
    """Force workspace refresh with timestamp"""
    import time
    timestamp = int(time.time())
    url = f"https://wandb.ai/{entity}/{project}/workspace?jupyter=true&t={timestamp}"
    return f'<iframe src="{url}" style="border:none;height:900px;width:100%">'

# Build Gradio interface
with gr.Blocks(title="ARR-COC Vision Model Validation") as demo:
    gr.Markdown("# ARR-COC: Adaptive Relevance Realization - Vision Model")
    gr.Markdown("Demo + Training Metrics Dashboard")

    with gr.Tabs():
        # Tab 1: Model Demo
        with gr.Tab("🎨 Model Demo"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Input Image")
                    query_text = gr.Textbox(
                        label="Query",
                        placeholder="What is shown in this image?"
                    )
                    inference_btn = gr.Button("Run Inference", variant="primary")

                with gr.Column():
                    output_image = gr.Image(label="Processed Output")
                    metrics_display = gr.JSON(label="Relevance Metrics")

            inference_btn.click(
                fn=run_arr_coc_inference,
                inputs=[input_image, query_text],
                outputs=[output_image, metrics_display]
            )

        # Tab 2: Training Metrics Report
        with gr.Tab("📊 Training Metrics"):
            gr.Markdown("### Latest Training Run Metrics")
            report_url = "https://wandb.ai/northhead/arr-coc-0-1/reports/Training-Progress--VmlldzoxMjM0NTY3"
            report_iframe = gr.HTML(create_wandb_report_iframe(report_url))

        # Tab 3: All Validation Runs
        with gr.Tab("🔬 Validation Workspace"):
            gr.Markdown("### All Validation Runs")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Dashboard")

            workspace_iframe = gr.HTML()

            # Initial load
            demo.load(
                fn=lambda: refresh_workspace("northhead", "arr-coc-validation"),
                outputs=workspace_iframe
            )

            # Manual refresh
            refresh_btn.click(
                fn=lambda: refresh_workspace("northhead", "arr-coc-validation"),
                outputs=workspace_iframe
            )

        # Tab 4: Run Comparison
        with gr.Tab("⚖️ Compare Runs"):
            gr.Markdown("### Compare Different ARR-COC Configurations")

            run_selector = gr.CheckboxGroup(
                choices=[
                    ("Baseline (no relevance)", "run_abc123"),
                    ("Propositional only", "run_def456"),
                    ("Perspectival only", "run_ghi789"),
                    ("Full ARR-COC (all 3 ways)", "run_jkl012")
                ],
                label="Select runs to compare",
                value=["run_abc123", "run_jkl012"]
            )

            compare_btn = gr.Button("Compare Selected Runs")
            comparison_iframe = gr.HTML()

            def create_comparison(selected_runs):
                if len(selected_runs) < 2:
                    return "<p>Please select at least 2 runs to compare.</p>"

                base_run = selected_runs[0]
                other_runs = ",".join(selected_runs[1:])
                url = f"https://wandb.ai/northhead/arr-coc-0-1/runs/{base_run}/compare?runSets={other_runs}&jupyter=true"
                return f'<iframe src="{url}" style="border:none;height:1024px;width:100%">'

            compare_btn.click(
                fn=create_comparison,
                inputs=run_selector,
                outputs=comparison_iframe
            )

# Launch with W&B integration
demo.integrate(wandb=wandb)
demo.launch(share=False)
```

**Key features demonstrated:**
- Model inference tab with W&B logging
- Embedded training metrics report
- Live workspace with manual refresh
- Run comparison interface
- W&B integration via `demo.integrate(wandb=wandb)`

---

## Sources

**W&B Official Documentation:**
- [Embed Reports](https://docs.wandb.ai/models/reports/embed-reports) - W&B Docs (accessed 2025-01-31)
  - IFrame embedding instructions
  - Public report requirements
  - Embed code generation
- [Collaborate on Reports](https://docs.wandb.ai/guides/reports/collaborate-on-reports/) - W&B Docs (accessed 2025-01-31)
  - Sharing and permissions
  - Public vs private reports
  - Report collaboration features

**Gradio Integration:**
- [Gradio and W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) - Gradio Docs (accessed 2025-01-31)
  - Official Gradio guide for W&B embedding
  - `gr.HTML()` iframe pattern
  - `demo.integrate(wandb=wandb)` usage
  - Complete JoJoGAN example

**Additional References:**
- [W&B Workspace URLs](https://wandb.ai) - URL structure patterns for workspaces, runs, reports
- ARR-COC validation requirements from VALIDATION-FOR-PLATONIC-CODING-CODEBASES.md (project documentation)
