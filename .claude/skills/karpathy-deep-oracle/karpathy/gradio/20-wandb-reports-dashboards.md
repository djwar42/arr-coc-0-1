# W&B Reports and Dashboards

**Complete guide to creating, customizing, and sharing W&B Reports for ML collaboration**

---

## Overview

W&B Reports are collaborative documents that combine visualizations, markdown text, and interactive panels to communicate ML findings. They transform training metrics, experiment results, and model comparisons into shareable, professional documentation.

**Key capabilities:**
- **Interactive dashboards** - Live charts updating with new runs
- **Programmatic creation** - Python API for automated report generation
- **Rich media support** - Embed images, videos, charts, tables, LaTeX
- **Version control** - Track changes, compare report versions
- **Team collaboration** - Comments, mentions, shared editing
- **Export options** - PDF, LaTeX, embedded iframes

**Use cases:**
- Experiment documentation and notebooks
- Weekly progress reports to stakeholders
- Model comparison dashboards
- Research paper supplementary materials
- Team knowledge bases

From [W&B Reports Documentation](https://docs.wandb.ai/models/reports) (accessed 2025-01-31)

---

## Section 1: Report Basics

### Creating Reports in UI

**Three ways to create reports:**

1. **From workspace** - Click "Create report" button in project workspace
2. **From run page** - Save current visualization state as report
3. **From template** - Use pre-built report templates

**Step-by-step UI creation:**

```
1. Navigate to project workspace
2. Click "Create report" (upper right corner)
3. Modal appears - select initial charts/panels
4. Choose "Filter run sets" to lock run selection
5. Click "Create report" → Opens draft in report tab
6. Edit content, add panels, customize
7. Click "Publish to project" when ready
8. Share via "Share" button
```

**Report templates available:**
- Blank report (start from scratch)
- Experiment comparison template
- Model evaluation template
- Weekly update template
- Research findings template

From [Create a Report Documentation](https://docs.wandb.ai/models/reports/create-a-report) (accessed 2025-01-31)

### Panel Types and Configurations

**Visualization panels:**
- **Line Plot** - Time series metrics (loss, accuracy)
- **Bar Chart** - Metric comparisons across runs
- **Scatter Plot** - Correlation analysis
- **Parallel Coordinates** - Multi-dimensional hyperparameter visualization
- **Custom Charts** - Vega-based custom visualizations
- **Media Panels** - Images, audio, video, HTML
- **Tables** - W&B Tables with filtering/sorting

**Content panels:**
- **Markdown blocks** - Rich text, equations, code snippets
- **Headers** (H1, H2, H3) - Section organization
- **Table of Contents** - Auto-generated navigation
- **Images/Videos** - Embedded media from URLs
- **Code blocks** - Syntax-highlighted code
- **LaTeX equations** - Mathematical notation

**Configuration options:**
- Panel width (full, half, third, quarter)
- Run filtering (regex, tags, specific runs)
- Metric smoothing and aggregation
- Color schemes and styling
- Legend positioning
- Axis scaling (log, linear)

From [W&B Reports Documentation](https://docs.wandb.ai/models/reports) (accessed 2025-01-31)

### Run Filtering and Grouping

**Filter mechanisms:**
- **Run set filtering** - Select specific runs by name, tag, or regex
- **Dynamic filters** - Automatically include new runs matching criteria
- **Locked run sets** - Freeze run selection at report creation time
- **Cross-project filters** - Include runs from multiple projects

**Example filtering patterns:**

```python
# UI: Use regex filter
runs matching: "baseline-.*"

# UI: Filter by tags
tags: ["production", "v2"]

# UI: Filter by config values
config.learning_rate > 0.001
```

**Grouping strategies:**
- Group by hyperparameter (learning rate, batch size)
- Group by model architecture
- Group by dataset split
- Group by user/team member
- Custom grouping via tags

**Run set management:**
- Create named run sets for reuse
- Share run sets across reports
- Update run sets without republishing
- Version control for run selections

From [W&B Reports Documentation](https://docs.wandb.ai/models/reports) (accessed 2025-01-31)

### Sharing and Permissions

**Sharing options:**
- **Public link** - Anyone with link can view (no W&B account needed)
- **Team link** - Only team members can access
- **Private** - Only report creator can view
- **Embedded iframe** - Embed in external websites/docs

**Permission levels:**
- **Viewer** - Can view report, cannot edit
- **Commenter** - Can add comments and suggestions
- **Editor** - Can modify report content
- **Admin** - Full control including deletion

**Collaboration features:**
- **Comments** - Thread discussions on specific panels
- **Mentions** - Tag team members with @username
- **Suggestions** - Propose edits without changing original
- **Version history** - Track all changes, revert if needed
- **Real-time editing** - Multiple editors simultaneously

**Export formats:**
- **PDF** - Static document for sharing
- **LaTeX zip** - For academic papers
- **Markdown** - Export as .md file
- **Embedded HTML** - Copy iframe code
- **Direct link** - Share URL

From [Collaborate on Reports Documentation](https://docs.wandb.ai/models/reports/collaborate-on-reports) (accessed 2025-01-31)

---

## Section 2: Programmatic Reports

### Python API for Report Generation

**Installation and setup:**

```python
# Install wandb-workspaces library
pip install wandb-workspaces

# Import reports API
import wandb_workspaces.reports.v2 as wr

# Authenticate (uses WANDB_API_KEY env var)
# Or set explicitly: wandb.login(key="your-api-key")
```

**Basic report creation:**

```python
# Create report object
report = wr.Report(
    entity="my-team",
    project="my-project",
    title="Automated Weekly Report",
    description="Generated from latest experiment results"
)

# Add content blocks
report.blocks = [
    wr.TableOfContents(),
    wr.H1("Experiment Results"),
    wr.P("Summary of this week's training runs."),
    wr.H2("Performance Metrics"),
    wr.LinePlot(x="step", y=["train/loss", "val/loss"]),
    wr.H2("Key Findings"),
    wr.UnorderedList([
        "Validation loss decreased by 15%",
        "Best checkpoint at epoch 42",
        "New baseline established"
    ]),
]

# Save report (creates draft)
report.save()

# Publish to project
report.publish()
```

**Advanced report with run filtering:**

```python
# Create report with specific runs
report = wr.Report(
    entity="my-team",
    project="vision-models",
    title="Model Comparison Dashboard"
)

# Add panels with run filtering
report.blocks = [
    wr.H1("ResNet vs ViT Comparison"),

    # Line plot comparing specific runs
    wr.LinePlot(
        x="epoch",
        y=["accuracy"],
        title="Validation Accuracy",
        # Filter runs by regex
        runs_filter="(resnet-50|vit-base)-.*",
        groupby=["architecture"],
        smoothing=0.3
    ),

    # Bar chart aggregating across runs
    wr.BarPlot(
        metrics=["final_accuracy", "inference_time"],
        title="Final Metrics Comparison",
        runs_filter="production-.*",
        groupby_aggfunc="mean"
    ),

    # Custom HTML/markdown
    wr.MarkdownBlock("""
    ### Analysis

    ViT-base shows **12% higher accuracy** but **3x slower inference**.

    Recommendation: Use ResNet-50 for production deployment.
    """),
]

report.save()
```

From [Reports and Workspaces API Documentation](https://docs.wandb.ai/models/ref/wandb_workspaces) (accessed 2025-01-31)

### Template Creation and Reuse

**Creating reusable templates:**

```python
# Define template function
def create_training_report_template(entity, project, run_filter):
    """Reusable template for training reports"""
    return wr.Report(
        entity=entity,
        project=project,
        title=f"Training Report - {project}",
        blocks=[
            wr.TableOfContents(),

            # Header section
            wr.H1("Training Summary"),
            wr.P("Automated report generated from latest runs."),

            # Metrics section
            wr.H2("Loss Curves"),
            wr.LinePlot(
                x="step",
                y=["train/loss", "val/loss"],
                runs_filter=run_filter,
                smoothing=0.6
            ),

            # Hyperparameters section
            wr.H2("Hyperparameter Analysis"),
            wr.ParallelCoordinatesPlot(
                columns=["learning_rate", "batch_size", "val/accuracy"],
                runs_filter=run_filter
            ),

            # Best model section
            wr.H2("Best Checkpoint"),
            wr.ScalarChart(
                metric="val/accuracy",
                groupby_aggfunc="max",
                runs_filter=run_filter
            ),
        ]
    )

# Use template for different projects
report1 = create_training_report_template(
    entity="my-team",
    project="nlp-experiments",
    run_filter="bert-.*"
)
report1.save()

report2 = create_training_report_template(
    entity="my-team",
    project="vision-experiments",
    run_filter="resnet-.*"
)
report2.save()
```

**Template with parameterization:**

```python
class ExperimentReportTemplate:
    """Configurable experiment report template"""

    def __init__(self, entity, project):
        self.entity = entity
        self.project = project

    def create_report(
        self,
        title,
        runs_filter,
        metrics_to_plot,
        include_images=False
    ):
        blocks = [
            wr.TableOfContents(),
            wr.H1(title),
        ]

        # Add metric plots
        for metric in metrics_to_plot:
            blocks.extend([
                wr.H2(f"{metric} Over Time"),
                wr.LinePlot(x="step", y=[metric], runs_filter=runs_filter)
            ])

        # Optionally add image gallery
        if include_images:
            blocks.extend([
                wr.H2("Sample Predictions"),
                wr.MediaBrowser(
                    media_keys=["predictions"],
                    runs_filter=runs_filter
                )
            ])

        report = wr.Report(
            entity=self.entity,
            project=self.project,
            title=title,
            blocks=blocks
        )

        return report

# Use template
template = ExperimentReportTemplate("my-team", "my-project")

report = template.create_report(
    title="Baseline Experiments",
    runs_filter="baseline-v.*",
    metrics_to_plot=["train/loss", "val/accuracy", "val/f1"],
    include_images=True
)
report.save()
```

From [Programmatic Workspaces Tutorial](https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb) (accessed 2025-01-31)

### Dynamic Report Updates

**Automatically updating reports:**

```python
# Create report that auto-updates with new runs
report = wr.Report(
    entity="my-team",
    project="continuous-training",
    title="Live Training Dashboard"
)

# Use dynamic run filters (no locked run set)
report.blocks = [
    wr.H1("Real-time Training Monitor"),
    wr.P("This report automatically includes new runs matching the filter."),

    # Filter includes future runs
    wr.LinePlot(
        x="step",
        y=["loss"],
        title="Latest Training Runs (Last 7 Days)",
        runs_filter="train-.*",
        # Include runs from last 7 days
        created_at=">=now-7d"
    ),

    # Show only currently running jobs
    wr.Table(
        columns=["name", "state", "duration", "gpu_util"],
        runs_filter=".*",
        state="running"
    ),
]

report.save()
```

**Scheduled report generation:**

```python
import schedule
import time

def generate_weekly_report():
    """Generate weekly summary report"""
    report = wr.Report(
        entity="my-team",
        project="production-models",
        title=f"Weekly Report - {time.strftime('%Y-%m-%d')}"
    )

    report.blocks = [
        wr.H1(f"Week of {time.strftime('%B %d, %Y')}"),

        # Runs from this week
        wr.LinePlot(
            x="step",
            y=["accuracy"],
            runs_filter=".*",
            created_at=">=now-7d"
        ),

        # Summary stats
        wr.MarkdownBlock(f"""
        ### Weekly Summary
        - Total runs: {count_runs_this_week()}
        - Best accuracy: {get_best_accuracy_this_week():.3f}
        - Average training time: {get_avg_training_time():.1f} hours
        """),
    ]

    report.save()
    report.publish()

    # Send notification
    send_slack_notification(report.url)

# Schedule weekly report
schedule.every().monday.at("09:00").do(generate_weekly_report)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

**CI/CD integration:**

```python
# In CI/CD pipeline (e.g., GitHub Actions)
def create_experiment_report_on_merge():
    """Create report when PR is merged"""
    import os

    pr_number = os.environ.get("PR_NUMBER")
    branch_name = os.environ.get("BRANCH_NAME")

    report = wr.Report(
        entity="my-team",
        project="experiments",
        title=f"PR #{pr_number} - {branch_name}"
    )

    # Include runs from this branch
    report.blocks = [
        wr.H1(f"Results from PR #{pr_number}"),
        wr.P(f"Branch: {branch_name}"),

        # Compare against main branch
        wr.LinePlot(
            x="step",
            y=["val/accuracy"],
            title="Accuracy vs Main Branch",
            runs_filter=f"(main|{branch_name})-.*",
            groupby=["branch"]
        ),

        # Link to PR
        wr.MarkdownBlock(f"""
        [View Pull Request](https://github.com/org/repo/pull/{pr_number})
        """),
    ]

    report.save()

    # Comment on PR with report link
    comment_on_pr(pr_number, f"W&B Report: {report.url}")

# Run in CI
if __name__ == "__main__":
    create_experiment_report_on_merge()
```

From [W&B Reports API Quickstart](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) (accessed 2025-01-31)

### Automated Report Generation (CI/CD)

**GitHub Actions workflow:**

```yaml
# .github/workflows/wandb-report.yml
name: Generate W&B Report

on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9 AM
  workflow_dispatch:  # Manual trigger

jobs:
  generate-report:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install wandb wandb-workspaces

      - name: Generate report
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python scripts/generate_weekly_report.py

      - name: Post to Slack
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "Weekly W&B report generated: ${{ steps.report.outputs.url }}"
            }
```

**Report generation script:**

```python
# scripts/generate_weekly_report.py
import wandb_workspaces.reports.v2 as wr
import wandb
from datetime import datetime, timedelta

def get_runs_from_last_week():
    """Fetch runs from W&B API"""
    api = wandb.Api()
    runs = api.runs(
        "my-team/my-project",
        filters={
            "created_at": {"$gte": (datetime.now() - timedelta(days=7)).isoformat()}
        }
    )
    return runs

def calculate_summary_stats(runs):
    """Calculate summary statistics"""
    total_runs = len(runs)
    successful_runs = sum(1 for r in runs if r.state == "finished")
    avg_accuracy = sum(r.summary.get("val/accuracy", 0) for r in runs) / total_runs

    return {
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "avg_accuracy": avg_accuracy,
    }

def generate_weekly_report():
    """Main report generation function"""
    runs = get_runs_from_last_week()
    stats = calculate_summary_stats(runs)

    week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end = datetime.now().strftime("%Y-%m-%d")

    report = wr.Report(
        entity="my-team",
        project="my-project",
        title=f"Weekly Report: {week_start} to {week_end}",
        description="Automated weekly summary of experiment results"
    )

    report.blocks = [
        wr.TableOfContents(),

        wr.H1(f"Week of {week_start}"),

        wr.MarkdownBlock(f"""
        ## Summary Statistics

        - **Total runs**: {stats['total_runs']}
        - **Successful runs**: {stats['successful_runs']}
        - **Average validation accuracy**: {stats['avg_accuracy']:.3f}
        - **Success rate**: {stats['successful_runs'] / stats['total_runs'] * 100:.1f}%
        """),

        wr.H2("Training Progress"),
        wr.LinePlot(
            x="step",
            y=["train/loss", "val/loss"],
            title="Loss Curves",
            runs_filter=".*",
            created_at=">=now-7d"
        ),

        wr.H2("Best Models"),
        wr.Table(
            columns=["name", "val/accuracy", "val/f1", "created_at"],
            runs_filter=".*",
            created_at=">=now-7d",
            sort_by="val/accuracy",
            sort_order="desc",
            limit=10
        ),

        wr.H2("Hyperparameter Analysis"),
        wr.ParallelCoordinatesPlot(
            columns=["learning_rate", "batch_size", "dropout", "val/accuracy"],
            runs_filter=".*",
            created_at=">=now-7d"
        ),
    ]

    report.save()
    report.publish()

    print(f"Report created: {report.url}")

    # Output for GitHub Actions
    with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
        f.write(f"url={report.url}\n")

    return report

if __name__ == "__main__":
    generate_weekly_report()
```

From [W&B Programmatic Reports Documentation](https://docs.wandb.ai/models/ref/wandb_workspaces) (accessed 2025-01-31)

---

## Section 3: Advanced Dashboards

### Multi-Project Dashboards

**Cross-project reporting:**

```python
# Create report combining multiple projects
report = wr.Report(
    entity="my-team",
    project="project-a",  # Primary project
    title="Multi-Project Model Comparison"
)

report.blocks = [
    wr.H1("Cross-Project Analysis"),

    # Compare runs from multiple projects
    wr.LinePlot(
        x="step",
        y=["accuracy"],
        title="Accuracy Across All Projects",
        # Specify runs from different projects
        runs=[
            {"entity": "my-team", "project": "nlp-models", "filter": "bert-.*"},
            {"entity": "my-team", "project": "vision-models", "filter": "resnet-.*"},
            {"entity": "my-team", "project": "multimodal", "filter": "clip-.*"},
        ],
        groupby=["project"]
    ),

    # Aggregate metrics across projects
    wr.BarPlot(
        metrics=["final_accuracy", "training_time", "model_size"],
        title="Project Comparison",
        runs=[
            {"entity": "my-team", "project": "nlp-models"},
            {"entity": "my-team", "project": "vision-models"},
            {"entity": "my-team", "project": "multimodal"},
        ],
        groupby=["project"],
        groupby_aggfunc="mean"
    ),
]

report.save()
```

**Organization-wide dashboard:**

```python
def create_org_dashboard(entity, project_list):
    """Create organization-wide metrics dashboard"""
    report = wr.Report(
        entity=entity,
        project="org-dashboard",
        title="Organization Metrics Dashboard"
    )

    blocks = [
        wr.TableOfContents(),
        wr.H1("Organization-Wide ML Metrics"),
        wr.P("Summary of all active ML projects"),
    ]

    # Add section for each project
    for project_name in project_list:
        blocks.extend([
            wr.H2(f"Project: {project_name}"),

            # Recent runs table
            wr.Table(
                columns=["name", "state", "val/accuracy", "created_at"],
                runs=[{"entity": entity, "project": project_name}],
                created_at=">=now-7d",
                limit=5
            ),

            # Training progress
            wr.LinePlot(
                x="step",
                y=["loss"],
                title=f"{project_name} - Recent Training",
                runs=[{"entity": entity, "project": project_name}],
                created_at=">=now-7d"
            ),
        ])

    # Overall statistics
    blocks.append(
        wr.MarkdownBlock("""
        ## Overall Statistics

        View detailed statistics in individual project reports.
        """)
    )

    report.blocks = blocks
    report.save()
    return report

# Create dashboard for all projects
dashboard = create_org_dashboard(
    entity="my-org",
    project_list=[
        "nlp-research",
        "computer-vision",
        "reinforcement-learning",
        "multimodal-models",
    ]
)
```

From [Cross-Project Reports Documentation](https://docs.wandb.ai/models/reports/cross-project-reports) (accessed 2025-01-31)

### Custom Visualizations

**Vega-based custom charts:**

```python
# Create custom visualization with Vega spec
custom_vega_spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Custom scatter plot with regression line",
    "data": {"name": "wandb"},
    "layer": [
        {
            "mark": "point",
            "encoding": {
                "x": {"field": "learning_rate", "type": "quantitative", "scale": {"type": "log"}},
                "y": {"field": "val_accuracy", "type": "quantitative"},
                "color": {"field": "optimizer", "type": "nominal"},
                "size": {"value": 60},
            }
        },
        {
            "mark": {"type": "line", "color": "firebrick"},
            "transform": [{"regression": "val_accuracy", "on": "learning_rate"}],
            "encoding": {
                "x": {"field": "learning_rate", "type": "quantitative"},
                "y": {"field": "val_accuracy", "type": "quantitative"},
            }
        }
    ]
}

report.blocks.append(
    wr.CustomChart(
        vega_spec=custom_vega_spec,
        title="Learning Rate vs Accuracy (with trend)",
        runs_filter=".*"
    )
)
```

**HTML/JavaScript interactive panels:**

```python
# Embed custom HTML visualization
custom_html = """
<div id="custom-viz">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        // Custom D3.js visualization
        const data = %WANDB_DATA%;  // W&B injects run data

        const svg = d3.select("#custom-viz")
            .append("svg")
            .attr("width", 800)
            .attr("height", 400);

        // Custom visualization logic
        // ...
    </script>
</div>
"""

report.blocks.append(
    wr.Html(content=custom_html)
)
```

From [Custom Charts Documentation](https://wandb.ai/wandb/customizable-charts/reports/Powerful-Custom-Charts-To-Debug-Model-Performance--VmlldzoyNzY4ODI) (accessed 2025-01-31)

### Report Embedding (iframe, API)

**Embed in external website:**

```html
<!-- Embed full report -->
<iframe
  src="https://wandb.ai/username/project/reports/Report-Title--VmlldzoxMjM0NTY"
  style="border:none;width:100%;height:1024px;"
></iframe>

<!-- Embed specific panel only -->
<iframe
  src="https://wandb.ai/username/project/reports/Report-Title--VmlldzoxMjM0NTY?panel=panel-id"
  style="border:none;width:800px;height:600px;"
></iframe>
```

**Programmatic embedding:**

```python
# Get embeddable URL
report = wr.Report.from_url(
    "https://wandb.ai/my-team/my-project/reports/My-Report--VmlldzoxMjM0NTY"
)

# Get embed code
embed_code = report.get_embed_code(
    width="100%",
    height="1024px",
    panel_id=None  # Or specific panel ID
)

print(embed_code)
# <iframe src="..." style="..."></iframe>
```

**Embed in Notion/Confluence:**

```markdown
## In Notion:
1. Type /embed
2. Paste W&B report URL
3. Notion auto-embeds the report

## In Confluence:
1. Insert > Other Macros > iframe
2. Paste W&B report URL
3. Set width/height
```

**Export and embed in documentation:**

```python
# Export report to static HTML
report = wr.Report.from_url("https://wandb.ai/...")

# Download as static HTML
html_content = report.export_html()

# Save to file
with open("report.html", "w") as f:
    f.write(html_content)

# Serve in documentation site
# Use in Sphinx, MkDocs, Docusaurus, etc.
```

From [Embed Reports Documentation](https://docs.wandb.ai/models/reports/embed-reports) (accessed 2025-01-31)

### Real-Time Updating Reports

**Live dashboard configuration:**

```python
# Create dashboard that updates in real-time
live_dashboard = wr.Report(
    entity="my-team",
    project="production-monitoring",
    title="Live Training Dashboard"
)

live_dashboard.blocks = [
    wr.H1("Real-Time Training Monitor"),
    wr.P("Updates automatically as new data is logged"),

    # Show currently running jobs
    wr.Table(
        columns=["name", "state", "step", "loss", "gpu_util"],
        title="Active Training Jobs",
        runs_filter=".*",
        state="running",
        auto_refresh=30  # Refresh every 30 seconds
    ),

    # Live loss curves (last 1000 steps)
    wr.LinePlot(
        x="step",
        y=["loss"],
        title="Loss (Real-Time)",
        runs_filter=".*",
        state="running",
        x_range="last-1000",  # Only show last 1000 steps
        smoothing=0.5
    ),

    # Resource utilization
    wr.LinePlot(
        x="timestamp",
        y=["system.gpu.0.memory", "system.cpu"],
        title="Resource Utilization",
        runs_filter=".*",
        state="running",
        x_range="last-1h"  # Last hour only
    ),
]

live_dashboard.save()
```

**Streaming metrics display:**

```python
# Dashboard for long-running experiments
def create_streaming_dashboard(run_id):
    """Create dashboard for monitoring specific run"""
    report = wr.Report(
        entity="my-team",
        project="long-experiments",
        title=f"Streaming Monitor - Run {run_id}"
    )

    report.blocks = [
        wr.H1(f"Live Monitoring: {run_id}"),

        # Recent metrics (streaming view)
        wr.LinePlot(
            x="step",
            y=["train/loss", "train/accuracy"],
            title="Training Metrics (Live)",
            runs_filter=f"^{run_id}$",
            x_range="last-100",  # Rolling window
            refresh_interval=10  # Update every 10 seconds
        ),

        # System metrics
        wr.LinePlot(
            x="_timestamp",
            y=["system.gpu.0.temp", "system.gpu.0.powerWatts"],
            title="GPU Health",
            runs_filter=f"^{run_id}$",
            x_range="last-5m",
            refresh_interval=5
        ),

        # Latest console output
        wr.CodeBlock(
            content="{{ run.logs.tail(20) }}",  # Template syntax
            language="text",
            title="Recent Logs"
        ),
    ]

    return report
```

From [W&B Reports Documentation](https://docs.wandb.ai/models/reports) (accessed 2025-01-31)

### Export and Archival

**Export formats:**

```python
# Export report to PDF
report = wr.Report.from_url("https://wandb.ai/...")

# Download as PDF
pdf_bytes = report.export_pdf()
with open("report.pdf", "wb") as f:
    f.write(pdf_bytes)

# Export to LaTeX (for academic papers)
latex_zip = report.export_latex()
with open("report_latex.zip", "wb") as f:
    f.write(latex_zip)

# Export to Markdown
markdown_content = report.export_markdown()
with open("report.md", "w") as f:
    f.write(markdown_content)
```

**Archival best practices:**

```python
def archive_monthly_reports(entity, project, year, month):
    """Archive all reports from a specific month"""
    import wandb

    api = wandb.Api()

    # Get all reports from project
    reports = api.reports(f"{entity}/{project}")

    # Filter by creation date
    monthly_reports = [
        r for r in reports
        if r.created_at.year == year and r.created_at.month == month
    ]

    # Export each report
    for report in monthly_reports:
        report_obj = wr.Report.from_url(report.url)

        # Export to multiple formats
        pdf_bytes = report_obj.export_pdf()
        latex_zip = report_obj.export_latex()

        # Save to archive directory
        archive_dir = f"archive/{year}/{month:02d}"
        os.makedirs(archive_dir, exist_ok=True)

        filename_base = f"{archive_dir}/{report.name}"

        with open(f"{filename_base}.pdf", "wb") as f:
            f.write(pdf_bytes)

        with open(f"{filename_base}_latex.zip", "wb") as f:
            f.write(latex_zip)

    print(f"Archived {len(monthly_reports)} reports from {year}-{month:02d}")

# Archive reports from January 2025
archive_monthly_reports("my-team", "my-project", 2025, 1)
```

**Version control integration:**

```python
# Commit report exports to git
def version_control_report(report_url, version_tag):
    """Export and commit report to git"""
    import subprocess

    report = wr.Report.from_url(report_url)

    # Export to markdown
    md_content = report.export_markdown()

    # Save to versioned file
    filename = f"docs/reports/{report.title}_{version_tag}.md"
    with open(filename, "w") as f:
        f.write(md_content)

    # Git commit
    subprocess.run(["git", "add", filename])
    subprocess.run([
        "git", "commit", "-m",
        f"Add report: {report.title} (version {version_tag})"
    ])

    print(f"Report committed: {filename}")

# Version control weekly reports
version_control_report(
    "https://wandb.ai/team/project/reports/Weekly-Report--VmlldzoxMjM0NTY",
    version_tag="2025-01-31"
)
```

From [Clone and Export Reports Documentation](https://docs.wandb.ai/models/reports/clone-and-export-reports) (accessed 2025-01-31)

---

## Sources

**Official Documentation:**
- [W&B Reports Overview](https://docs.wandb.ai/models/reports) - Complete reports documentation (accessed 2025-01-31)
- [Create a Report](https://docs.wandb.ai/models/reports/create-a-report) - UI and programmatic creation (accessed 2025-01-31)
- [Reports and Workspaces API](https://docs.wandb.ai/models/ref/wandb_workspaces) - Python API reference (accessed 2025-01-31)
- [Collaborate on Reports](https://docs.wandb.ai/models/reports/collaborate-on-reports) - Sharing and permissions (accessed 2025-01-31)
- [Embed Reports](https://docs.wandb.ai/models/reports/embed-reports) - Embedding in websites/docs (accessed 2025-01-31)
- [Cross-Project Reports](https://docs.wandb.ai/models/reports/cross-project-reports) - Multi-project dashboards (accessed 2025-01-31)
- [Clone and Export Reports](https://docs.wandb.ai/models/reports/clone-and-export-reports) - Export formats (accessed 2025-01-31)

**Tutorials and Examples:**
- [Reports API Quickstart](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) - Interactive Colab tutorial (accessed 2025-01-31)
- [Programmatic Workspaces Tutorial](https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb) - Advanced API usage (accessed 2025-01-31)
- [Custom Charts Report](https://wandb.ai/wandb/customizable-charts/reports/Powerful-Custom-Charts-To-Debug-Model-Performance--VmlldzoyNzY4ODI) - Custom visualization examples (accessed 2025-01-31)
- [Example Reports Gallery](https://docs.wandb.ai/models/reports/reports-gallery) - Live report examples (accessed 2025-01-31)

**Additional Resources:**
- [W&B Reports Video Tutorial](https://www.youtube.com/watch?v=2xeJIv_K_eI) - Video demonstration (accessed 2025-01-31)
- [GitHub: wandb-workspaces](https://github.com/wandb/wandb-workspaces) - Source code repository (accessed 2025-01-31)
