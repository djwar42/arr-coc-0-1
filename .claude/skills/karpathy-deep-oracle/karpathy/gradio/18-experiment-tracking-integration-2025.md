# Experiment Tracking Integration with Gradio (2025)

## Overview

Integrating experiment tracking tools with Gradio applications enables comprehensive ML workflow monitoring - from development to production. This guide covers deep integrations with W&B (Weights & Biases), MLflow, TensorBoard, and custom tracking solutions, specifically focused on VLM testing and validation workflows relevant to ARR-COC development.

**Why Experiment Tracking with Gradio?**
- Track model performance across checkpoints during interactive testing
- Log user interactions and inference results for analysis
- Compare ablation studies visually while maintaining data provenance
- Build reproducible validation workflows with full experiment history
- Enable team collaboration with shared experiment dashboards

**Key Use Cases for VLM Testing:**
- A/B testing different compression strategies (ARR-COC homunculus validation)
- Logging relevance scores and patch selection decisions
- Tracking query-aware performance across diverse test cases
- Monitoring memory usage and latency for different configurations
- Comparing statistical metrics across model versions

---

## Section 1: W&B Integration Deep Dive (~130 lines)

### Why W&B with Gradio?

Weights & Biases provides the most mature Gradio integration, with first-class support for embedding demos in W&B Reports and logging from Gradio functions.

**Perfect for:**
- Research projects needing shareable dashboards
- Team collaboration with interactive demos
- Long-term experiment archival with visual demos
- Model registry integration

### Basic W&B Setup

```python
import gradio as gr
import wandb

# Initialize W&B project
wandb.init(
    project="arr-coc-vlm-testing",
    entity="your-team",
    config={
        "model_name": "arr-coc-v0.1",
        "compression_range": "64-400 tokens",
        "query_aware": True
    }
)

def inference_with_logging(image, query):
    # Your VLM inference
    result = model(image, query)

    # Log to W&B
    wandb.log({
        "inference_count": wandb.run.summary.get("inference_count", 0) + 1,
        "query_length": len(query),
        "result": result
    })

    # Log images
    wandb.log({
        "input_image": wandb.Image(image, caption=query),
        "output": result
    })

    return result

demo = gr.Interface(
    fn=inference_with_logging,
    inputs=[gr.Image(), gr.Textbox()],
    outputs=gr.Textbox()
)

# Integrate W&B with Gradio (creates embedded demo in W&B dashboard)
demo.integrate(wandb=wandb)
demo.launch()
```

### Advanced: Logging Structured Data

For VLM testing with complex outputs (relevance scores, patch selections):

```python
import wandb

def vlm_test_with_structured_logging(image, query):
    # Run inference
    output = model.generate(image, query)
    relevance_scores = model.get_patch_relevance(image, query)
    selected_patches = model.get_selected_patches(image, query)

    # Create W&B Table for detailed results
    table = wandb.Table(
        columns=["Query", "Patch_ID", "Relevance_Score", "Selected", "Token_Budget"],
        data=[
            [query, i, score, i in selected_patches, budget]
            for i, (score, budget) in enumerate(zip(relevance_scores, token_budgets))
        ]
    )

    wandb.log({
        "test_results": table,
        "total_tokens_used": sum(token_budgets),
        "num_patches_selected": len(selected_patches)
    })

    return output, relevance_scores

demo = gr.Interface(
    fn=vlm_test_with_structured_logging,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Query")],
    outputs=[gr.Textbox(label="Output"), gr.JSON(label="Relevance Scores")]
)
```

### Embedding W&B Reports in Gradio

Reverse integration - show W&B dashboards inside Gradio:

```python
import gradio as gr

def create_wandb_report_embed(report_url):
    """Embed W&B Report in Gradio using iframe"""
    iframe_html = f'''
    <iframe
        src="{report_url}"
        style="border:none;height:800px;width:100%"
    ></iframe>
    '''
    return iframe_html

with gr.Blocks() as demo:
    gr.Markdown("# ARR-COC Experiment Dashboard")

    with gr.Tab("Live Testing"):
        image_input = gr.Image()
        query_input = gr.Textbox()
        output = gr.Textbox()
        test_btn = gr.Button("Run Test")

    with gr.Tab("Experiment History"):
        # Embed W&B Report showing all experiments
        report_url = "https://wandb.ai/team/project/reports/ARR-COC-Results--VmlldzoxMjM0NTY"
        gr.HTML(create_wandb_report_embed(report_url))

    test_btn.click(fn=inference_with_logging, inputs=[image_input, query_input], outputs=output)

demo.launch()
```

### Best Practices for W&B + Gradio

**Session Management:**
```python
import gradio as gr
import wandb

# Don't reinitialize W&B on every function call
# Initialize once at app startup
wandb.init(project="vlm-testing", mode="online")

def inference(image, query, session_state):
    # Use session state to track per-user metrics
    session_state["call_count"] = session_state.get("call_count", 0) + 1

    result = model(image, query)

    # Log with session context
    wandb.log({
        "session_id": session_state.get("session_id"),
        "call_count": session_state["call_count"],
        "result": result
    })

    return result, session_state

with gr.Blocks() as demo:
    session = gr.State({"session_id": wandb.util.generate_id()})
    # ... interface components
```

**Gradio-Specific W&B Artifacts:**
- Log Gradio app version alongside experiments
- Save example inputs as W&B Artifacts for reproducibility
- Track which Gradio component triggered logging (useful for multi-interface apps)

From [Gradio W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):
- W&B + Gradio enables researchers to share interactive demos directly in experiment dashboards
- The `demo.integrate(wandb=wandb)` method automatically creates shareable demo links
- W&B Reports can embed Gradio apps using `<gradio-app>` web components

---

## Section 2: MLflow Integration (~110 lines)

### Why MLflow?

MLflow offers strong model registry and deployment features, ideal for production VLM workflows.

**Perfect for:**
- Model versioning and registry
- Production deployment tracking
- Multi-framework projects
- On-premise experiment tracking

### Basic MLflow Setup

```python
import gradio as gr
import mlflow
import mlflow.pytorch

# Start MLflow run
mlflow.set_experiment("arr-coc-vlm-testing")

def inference_with_mlflow(image, query):
    with mlflow.start_run(run_name=f"query_{len(query)}_chars"):
        # Log parameters
        mlflow.log_param("query_length", len(query))
        mlflow.log_param("model_version", "arr-coc-v0.1")

        # Run inference
        result = model(image, query)
        tokens_used = model.get_token_count()

        # Log metrics
        mlflow.log_metric("tokens_used", tokens_used)
        mlflow.log_metric("inference_time_ms", inference_time)

        # Log artifacts (images)
        image.save("temp_input.jpg")
        mlflow.log_artifact("temp_input.jpg", artifact_path="inputs")

        return result

demo = gr.Interface(
    fn=inference_with_mlflow,
    inputs=[gr.Image(type="pil"), gr.Textbox()],
    outputs=gr.Textbox()
)
demo.launch()
```

### MLflow Model Registry Integration

Track which model checkpoint is being used in Gradio:

```python
import mlflow.pyfunc

# Load registered model from MLflow
model_uri = "models:/arr-coc-vlm/Production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

def inference_from_registry(image, query):
    with mlflow.start_run():
        # Log which registered model version is being used
        model_version = mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get("mlflow.log-model.history")
        mlflow.log_param("model_source", model_uri)

        prediction = loaded_model.predict([image, query])

        return prediction

with gr.Blocks() as demo:
    gr.Markdown(f"## Using Model: `{model_uri}`")
    # ... rest of interface
```

### Multi-Run Comparison in Gradio

Compare multiple MLflow runs interactively:

```python
import mlflow
import gradio as gr
import pandas as pd

def compare_mlflow_runs(run_ids):
    """Fetch and compare metrics from multiple MLflow runs"""
    client = mlflow.tracking.MlflowClient()

    comparison_data = []
    for run_id in run_ids.split(","):
        run = client.get_run(run_id.strip())
        comparison_data.append({
            "Run ID": run_id,
            "Accuracy": run.data.metrics.get("accuracy", 0),
            "Tokens Used": run.data.metrics.get("avg_tokens", 0),
            "Model Version": run.data.params.get("model_version", "unknown")
        })

    return pd.DataFrame(comparison_data)

with gr.Blocks() as demo:
    gr.Markdown("# MLflow Run Comparison")
    run_ids_input = gr.Textbox(label="MLflow Run IDs (comma-separated)")
    compare_btn = gr.Button("Compare Runs")
    output_table = gr.Dataframe()

    compare_btn.click(
        fn=compare_mlflow_runs,
        inputs=run_ids_input,
        outputs=output_table
    )
```

### Best Practices

**Nested Runs for Complex Workflows:**
```python
with mlflow.start_run(run_name="vlm_session") as parent_run:
    # Parent run for the entire Gradio session

    def inference_nested(image, query):
        with mlflow.start_run(run_name=f"query_{idx}", nested=True):
            # Child run for each inference
            mlflow.log_param("parent_run_id", parent_run.info.run_id)
            result = model(image, query)
            mlflow.log_metric("result_length", len(result))
            return result

    demo = gr.Interface(fn=inference_nested, ...)
    demo.launch()
```

**Automatic Model Logging:**
```python
# Log model to MLflow registry when Gradio app starts
with mlflow.start_run():
    mlflow.pytorch.log_model(
        model,
        "arr-coc-model",
        registered_model_name="ARR-COC-VLM"
    )
```

---

## Section 3: TensorBoard Logging (~80 lines)

### Why TensorBoard?

TensorBoard excels at real-time metric visualization, especially for neural network debugging.

**Perfect for:**
- Real-time loss/accuracy tracking during fine-tuning
- Weight/gradient visualization
- Model graph inspection
- Local development workflows

### Basic TensorBoard Setup

```python
import gradio as gr
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/arr-coc-gradio")

global_step = 0

def inference_with_tensorboard(image, query):
    global global_step

    # Run inference
    result = model(image, query)
    relevance_scores = model.get_relevance_scores()

    # Log scalar metrics
    writer.add_scalar("inference/query_length", len(query), global_step)
    writer.add_scalar("inference/avg_relevance", relevance_scores.mean(), global_step)

    # Log images
    writer.add_image("input_image", T.ToTensor()(image), global_step)

    # Log histograms (relevance score distribution)
    writer.add_histogram("relevance_distribution", relevance_scores, global_step)

    global_step += 1
    writer.flush()  # Ensure data is written

    return result

demo = gr.Interface(
    fn=inference_with_tensorboard,
    inputs=[gr.Image(type="pil"), gr.Textbox()],
    outputs=gr.Textbox()
)

# Launch TensorBoard in parallel (user opens localhost:6006)
print("View TensorBoard at: http://localhost:6006")
demo.launch()
```

### Embedding TensorBoard in Gradio

Show TensorBoard dashboard within Gradio app:

```python
import gradio as gr
import subprocess
import time

# Start TensorBoard server in background
tensorboard_process = subprocess.Popen(
    ["tensorboard", "--logdir", "runs/", "--port", "6006"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(2)  # Give TensorBoard time to start

with gr.Blocks() as demo:
    with gr.Tab("Inference"):
        # Your inference interface
        pass

    with gr.Tab("TensorBoard"):
        gr.HTML('''
            <iframe
                src="http://localhost:6006"
                style="width:100%; height:800px; border:none;"
            ></iframe>
        ''')

demo.launch()
```

### Real-Time Plotting with TensorBoard Data

Read TensorBoard logs and display in Gradio:

```python
from tensorboard.backend.event_processing import event_accumulator
import gradio as gr
import plotly.graph_objects as go

def get_tensorboard_metrics(log_dir):
    """Extract metrics from TensorBoard logs"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get scalar data
    tags = ea.Tags()['scalars']
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data

def plot_metrics():
    metrics = get_tensorboard_metrics("runs/arr-coc-gradio")

    fig = go.Figure()
    for tag, values in metrics.items():
        steps, vals = zip(*values)
        fig.add_trace(go.Scatter(x=steps, y=vals, name=tag, mode='lines+markers'))

    fig.update_layout(title="Real-Time Metrics", xaxis_title="Step", yaxis_title="Value")
    return fig

with gr.Blocks() as demo:
    plot_output = gr.Plot()
    refresh_btn = gr.Button("Refresh Metrics")
    refresh_btn.click(fn=plot_metrics, outputs=plot_output)
```

From [TensorFlow TensorBoard Documentation](https://www.tensorflow.org/tensorboard/get_started) (accessed 2025-01-31):
- TensorBoard provides measurements and visualizations for machine learning workflows
- Best used for monitoring training progress and debugging model internals
- Can be embedded in web apps via iframe for integrated dashboards

---

## Section 4: Custom Experiment Tracking (~80 lines)

### Why Custom Tracking?

For projects needing full control, lightweight solutions, or offline tracking.

**Perfect for:**
- Offline/air-gapped environments
- Minimal dependencies
- Custom metrics not supported by standard tools
- Privacy-sensitive projects

### SQLite-Based Tracker

```python
import gradio as gr
import sqlite3
import json
from datetime import datetime

class SimpleExperimentTracker:
    def __init__(self, db_path="experiments.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                result TEXT,
                metrics TEXT,
                image_path TEXT
            )
        ''')
        self.conn.commit()

    def log_experiment(self, query, result, metrics, image_path=None):
        self.conn.execute(
            "INSERT INTO experiments (timestamp, query, result, metrics, image_path) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), query, result, json.dumps(metrics), image_path)
        )
        self.conn.commit()

    def get_all_experiments(self):
        cursor = self.conn.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
        return cursor.fetchall()

    def get_metrics_summary(self):
        experiments = self.get_all_experiments()
        metrics_list = [json.loads(exp[4]) for exp in experiments]

        summary = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            summary[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        return summary

# Initialize tracker
tracker = SimpleExperimentTracker()

def inference_with_custom_tracking(image, query):
    result = model(image, query)

    metrics = {
        "tokens_used": model.get_token_count(),
        "inference_time_ms": measure_time(),
        "query_length": len(query)
    }

    # Save image temporarily
    image_path = f"temp/{datetime.now().timestamp()}.jpg"
    image.save(image_path)

    # Log to custom tracker
    tracker.log_experiment(query, result, metrics, image_path)

    return result

with gr.Blocks() as demo:
    with gr.Tab("Inference"):
        image_input = gr.Image(type="pil")
        query_input = gr.Textbox()
        output = gr.Textbox()
        submit_btn = gr.Button("Submit")
        submit_btn.click(fn=inference_with_custom_tracking, inputs=[image_input, query_input], outputs=output)

    with gr.Tab("Experiment History"):
        def show_history():
            import pandas as pd
            experiments = tracker.get_all_experiments()
            df = pd.DataFrame(experiments, columns=["ID", "Timestamp", "Query", "Result", "Metrics", "Image"])
            return df

        history_table = gr.Dataframe()
        refresh_btn = gr.Button("Refresh History")
        refresh_btn.click(fn=show_history, outputs=history_table)

demo.launch()
```

### JSON File-Based Tracker

Simpler alternative for small projects:

```python
import json
import os
from datetime import datetime

class JSONExperimentTracker:
    def __init__(self, log_file="experiments.json"):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)

    def log(self, experiment_data):
        with open(self.log_file, 'r') as f:
            data = json.load(f)

        experiment_data['timestamp'] = datetime.now().isoformat()
        data.append(experiment_data)

        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def export_csv(self, output_path):
        import pandas as pd
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return output_path

tracker = JSONExperimentTracker()

def inference_logged(image, query):
    result = model(image, query)
    tracker.log({
        "query": query,
        "result": result,
        "tokens": model.tokens_used
    })
    return result

with gr.Blocks() as demo:
    # ... interface

    export_btn = gr.Button("Export to CSV")
    export_file = gr.File()
    export_btn.click(
        fn=lambda: tracker.export_csv("experiments_export.csv"),
        outputs=export_file
    )
```

### Best Practices for Custom Tracking

**Thread Safety:**
- Use `check_same_thread=False` for SQLite in Gradio (multi-user)
- Add locks for concurrent file writes

**Data Retention:**
```python
def cleanup_old_experiments(days=30):
    """Remove experiments older than N days"""
    cutoff_date = datetime.now() - timedelta(days=days)
    tracker.conn.execute(
        "DELETE FROM experiments WHERE timestamp < ?",
        (cutoff_date.isoformat(),)
    )
    tracker.conn.commit()
```

**Export Functionality:**
- Always provide CSV/JSON export for downstream analysis
- Include metadata (Gradio version, model version, timestamp)

---

## Comparison Table: Choosing the Right Tool

| Feature | W&B | MLflow | TensorBoard | Custom |
|---------|-----|--------|-------------|--------|
| **Ease of Setup** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Gradio Integration** | ⭐⭐⭐⭐⭐ (Native) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (Full control) |
| **Collaboration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Model Registry** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐ |
| **Real-Time Viz** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Offline Support** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | Free tier | Free (self-host) | Free | Free |

**Recommendation for ARR-COC:**
- **Development/Research**: W&B (best Gradio integration, team collaboration)
- **Production**: MLflow (model registry, deployment tracking)
- **Local debugging**: TensorBoard (real-time metrics, lightweight)
- **Offline/airgapped**: Custom SQLite tracker

---

## Sources

**Official Documentation:**
- [Gradio W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) - Official Gradio documentation (accessed 2025-01-31)
- [W&B Experiment Tracking](https://docs.wandb.ai/models/tutorials/experiments) - W&B documentation
- [TensorFlow TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started) - TensorBoard getting started (accessed 2025-01-31)
- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html) - MLflow official docs

**Web Research (2025):**
- [How Gradio and W&B Work Beautifully Together](https://wandb.ai/abidlabs/your-test-project/reports/How-Gradio-and-W-B-Work-Beautifully-Together---Vmlldzo4MTk0MzI) - W&B Report by Abubakar Abid (accessed 2025-01-31)
- [Embed W&B Reports in Gradio](https://docs.wandb.ai/models/reports/embed-reports) - W&B documentation on Gradio embedding (accessed 2025-01-31)

**Additional References:**
- [Hugging Face Model Hub Integration](https://huggingface.co/docs/huggingface_hub/en/package_reference/tensorboard) - TensorBoard logger for HF Hub
- [Serverless RAG with Gradio, MLflow Tracing](https://aibits.blog/2024/11/13/serverless-rag-with-gradio-mlflow-tracing-and-databricks/) - MLflow + Gradio tutorial (November 2024)

**Search Queries Used:**
- "Gradio Weights and Biases W&B integration tutorial 2024 2025"
- "Gradio MLflow integration experiment tracking"
- "Gradio TensorBoard logging visualization"
- "site:wandb.ai Gradio integration example code"
