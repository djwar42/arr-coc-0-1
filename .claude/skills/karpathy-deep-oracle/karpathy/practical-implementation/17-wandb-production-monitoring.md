# W&B Production Monitoring

## Overview

Production monitoring extends W&B's experiment tracking capabilities to deployed models, enabling real-time performance tracking, data drift detection, and model health monitoring. Unlike training-time tracking (which focuses on loss curves and validation metrics), production monitoring focuses on inference latency, throughput, prediction quality, and operational health.

**Key distinction**: Training monitoring asks "Is my model learning?" Production monitoring asks "Is my deployed model working correctly for real users?"

From [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):
- Production inference logging captures request/response pairs
- Real-time dashboards visualize latency and error rates
- Sampling strategies handle high-traffic scenarios

From [LLM Debugging and Monitoring Guide](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) (accessed 2025-01-31):
- W&B Weave provides specialized LLM observability
- Trace entire request chains (retrieval → generation → response)
- Monitor token usage, cost, and safety metrics

---

## Section 1: Production Inference Tracking (~150 lines)

### Logging Predictions in Production

Production inference logging captures model inputs, outputs, and metadata for every prediction (or sampled predictions in high-traffic scenarios).

**Basic Production Logging Pattern:**

```python
import wandb
from datetime import datetime

# Initialize production run (long-lived)
run = wandb.init(
    project="arr-coc-production",
    job_type="inference",
    name=f"production-{datetime.now().strftime('%Y-%m-%d')}",
    tags=["production", "v1.0.3"]
)

def predict_with_logging(image, query, model):
    """Make prediction and log to W&B"""
    start_time = datetime.now()

    # Run inference
    try:
        result = model(image, query)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)

    end_time = datetime.now()
    latency_ms = (end_time - start_time).total_seconds() * 1000

    # Log inference metrics
    wandb.log({
        "inference/latency_ms": latency_ms,
        "inference/success": 1 if success else 0,
        "inference/error": error,
        "inference/timestamp": end_time.timestamp(),
        # Token budget if applicable
        "inference/tokens_allocated": result.tokens_allocated if result else 0,
    })

    return result
```

From [Foundation Model Builder Tips](https://wandb.ai/wandb/report/reports/Foundation-Model-Builder-Tips-Best-Practices--Vmlldzo5MTA3MDI2) (accessed 2025-01-31):
- Long-running production runs stay open for continuous logging
- Use daily or weekly runs to avoid single massive runs
- Tag with deployment version for A/B testing

### Latency and Throughput Monitoring

Track performance metrics to identify bottlenecks and capacity issues.

**Latency Distribution Logging:**

```python
import numpy as np

class LatencyMonitor:
    """Track and log latency distributions"""

    def __init__(self, window_size=100):
        self.latencies = []
        self.window_size = window_size

    def record(self, latency_ms):
        self.latencies.append(latency_ms)

        # Log distribution every N requests
        if len(self.latencies) >= self.window_size:
            self._log_distribution()
            self.latencies = []  # Reset window

    def _log_distribution(self):
        wandb.log({
            "latency/p50_ms": np.percentile(self.latencies, 50),
            "latency/p95_ms": np.percentile(self.latencies, 95),
            "latency/p99_ms": np.percentile(self.latencies, 99),
            "latency/mean_ms": np.mean(self.latencies),
            "latency/std_ms": np.std(self.latencies),
            "latency/min_ms": np.min(self.latencies),
            "latency/max_ms": np.max(self.latencies),
        })

# Usage
latency_monitor = LatencyMonitor(window_size=100)

for image, query in requests:
    start = datetime.now()
    result = model(image, query)
    latency_ms = (datetime.now() - start).total_seconds() * 1000

    latency_monitor.record(latency_ms)
```

**Throughput Tracking:**

```python
from collections import deque
import time

class ThroughputMonitor:
    """Track requests per second"""

    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def record_request(self):
        now = time.time()
        self.timestamps.append(now)

        # Remove timestamps outside window
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

        # Calculate throughput
        throughput = len(self.timestamps) / self.window_seconds

        wandb.log({
            "throughput/requests_per_second": throughput,
            "throughput/active_requests": len(self.timestamps)
        })
```

### Error Rate Tracking

Monitor prediction failures and categorize errors.

**Error Tracking Pattern:**

```python
class ErrorTracker:
    """Track and categorize production errors"""

    def __init__(self):
        self.error_counts = {}
        self.total_requests = 0

    def record_request(self, success, error_type=None):
        self.total_requests += 1

        if not success:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error metrics
        error_rate = (sum(self.error_counts.values()) / self.total_requests) * 100

        wandb.log({
            "errors/total_rate_percent": error_rate,
            "errors/total_count": sum(self.error_counts.values()),
            "errors/request_count": self.total_requests,
        })

        # Log individual error types
        for error_type, count in self.error_counts.items():
            wandb.log({
                f"errors/{error_type}_count": count,
                f"errors/{error_type}_rate": (count / self.total_requests) * 100
            })

# Usage
error_tracker = ErrorTracker()

try:
    result = model(image, query)
    error_tracker.record_request(success=True)
except ValueError as e:
    error_tracker.record_request(success=False, error_type="invalid_input")
except RuntimeError as e:
    error_tracker.record_request(success=False, error_type="model_error")
except Exception as e:
    error_tracker.record_request(success=False, error_type="unknown")
```

### Request/Response Logging

Capture full request/response pairs for debugging and analysis.

**W&B Tables for Request Logging:**

```python
def log_request_batch(requests, responses, sample_rate=0.1):
    """Log sample of requests/responses to W&B Table"""

    # Sample requests (don't log everything in production)
    if np.random.random() > sample_rate:
        return

    # Build table data
    table_data = []
    for req, resp in zip(requests, responses):
        table_data.append([
            req['query'],
            wandb.Image(req['image']),
            resp['prediction'],
            resp['confidence'],
            resp['tokens_allocated'],
            resp['latency_ms'],
            datetime.now().isoformat()
        ])

    # Log table
    table = wandb.Table(
        columns=['query', 'image', 'prediction', 'confidence',
                 'tokens', 'latency_ms', 'timestamp'],
        data=table_data
    )

    wandb.log({"production/requests": table})
```

### Sampling Strategies for High Traffic

In production with thousands of requests per second, log strategically to avoid overwhelming W&B.

**Adaptive Sampling:**

```python
class AdaptiveSampler:
    """Sample logs based on error rate"""

    def __init__(self, base_rate=0.01, error_boost=10):
        self.base_rate = base_rate  # 1% baseline
        self.error_boost = error_boost  # 10x for errors
        self.recent_errors = deque(maxlen=100)

    def should_log(self, is_error=False):
        """Decide whether to log this request"""

        # Always log errors (or high probability)
        if is_error:
            self.recent_errors.append(1)
            return np.random.random() < (self.base_rate * self.error_boost)

        self.recent_errors.append(0)

        # Increase sampling if error rate is high
        error_rate = sum(self.recent_errors) / len(self.recent_errors)

        if error_rate > 0.1:  # >10% errors
            sample_rate = self.base_rate * 5
        else:
            sample_rate = self.base_rate

        return np.random.random() < sample_rate

# Usage
sampler = AdaptiveSampler(base_rate=0.01)

for image, query in requests:
    result = model(image, query)
    is_error = result is None

    if sampler.should_log(is_error):
        wandb.log({
            "inference/result": result,
            "inference/error": is_error
        })
```

From [Model Monitoring Guide](https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide) (accessed 2025-01-31):
- Sample 1-10% of successful requests
- Log 100% of errors (or high percentage)
- Use stratified sampling for balanced coverage

---

## Section 2: Model Health Monitoring (~130 lines)

### Data Drift Detection

Monitor when input distributions change from training data.

**Distribution Monitoring:**

```python
from scipy.stats import ks_2samp

class DriftDetector:
    """Detect data drift in production"""

    def __init__(self, reference_stats):
        """
        Args:
            reference_stats: Statistics from training/validation data
                e.g., {'query_length': {'mean': 50, 'std': 20}}
        """
        self.reference_stats = reference_stats
        self.production_samples = []

    def check_drift(self, query_length):
        """Check if current distribution drifts from reference"""
        self.production_samples.append(query_length)

        # Check every 1000 samples
        if len(self.production_samples) >= 1000:
            # KS test for distribution shift
            ref_mean = self.reference_stats['query_length']['mean']
            ref_std = self.reference_stats['query_length']['std']

            # Synthetic reference distribution
            reference = np.random.normal(ref_mean, ref_std, 1000)

            statistic, p_value = ks_2samp(reference, self.production_samples)

            wandb.log({
                "drift/query_length_ks_statistic": statistic,
                "drift/query_length_p_value": p_value,
                "drift/query_length_mean": np.mean(self.production_samples),
                "drift/query_length_std": np.std(self.production_samples),
                "drift/drift_detected": p_value < 0.05  # Significant drift
            })

            self.production_samples = []  # Reset
```

From [Data Drift Detection Research](https://wandb.ai/yujiewang/data-drift-detector/reports/Drift-Detection-Progress-Report--VmlldzoyODY3NDk1) (accessed 2025-01-31):
- Statistical tests: KS test, Chi-square, Population Stability Index
- Monitor feature distributions over time
- Alert when p-value < 0.05 (significant drift)

### Prediction Distribution Monitoring

Track how prediction distributions change over time.

**Confidence Score Analysis:**

```python
class PredictionMonitor:
    """Monitor prediction quality metrics"""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.confidences = []
        self.predictions = []

    def record(self, prediction, confidence):
        self.confidences.append(confidence)
        self.predictions.append(prediction)

        if len(self.confidences) >= self.window_size:
            self._analyze_distribution()
            self.confidences = []
            self.predictions = []

    def _analyze_distribution(self):
        """Analyze prediction distribution"""

        # Confidence distribution
        wandb.log({
            "predictions/confidence_mean": np.mean(self.confidences),
            "predictions/confidence_std": np.std(self.confidences),
            "predictions/confidence_p25": np.percentile(self.confidences, 25),
            "predictions/confidence_median": np.percentile(self.confidences, 50),
            "predictions/confidence_p75": np.percentile(self.confidences, 75),
        })

        # Low confidence rate (potential issue)
        low_confidence_rate = np.mean(np.array(self.confidences) < 0.5)
        wandb.log({
            "predictions/low_confidence_rate": low_confidence_rate,
            "predictions/low_confidence_alert": low_confidence_rate > 0.3  # >30% low conf
        })

        # Prediction diversity (avoid model collapse)
        unique_predictions = len(set(self.predictions))
        wandb.log({
            "predictions/unique_count": unique_predictions,
            "predictions/diversity_ratio": unique_predictions / len(self.predictions)
        })
```

### Model Degradation Alerts

Detect when model performance drops in production.

**Performance Regression Detection:**

```python
class PerformanceMonitor:
    """Detect model degradation"""

    def __init__(self, baseline_metrics):
        """
        Args:
            baseline_metrics: Expected performance from validation
                e.g., {'accuracy': 0.85, 'f1': 0.82}
        """
        self.baseline = baseline_metrics
        self.recent_metrics = []

    def check_degradation(self, accuracy, f1_score):
        """Check if performance has degraded"""

        # Store recent metrics
        self.recent_metrics.append({
            'accuracy': accuracy,
            'f1': f1_score
        })

        # Keep last 100 evaluations
        if len(self.recent_metrics) > 100:
            self.recent_metrics.pop(0)

        # Calculate current performance
        current_accuracy = np.mean([m['accuracy'] for m in self.recent_metrics])
        current_f1 = np.mean([m['f1'] for m in self.recent_metrics])

        # Degradation thresholds
        accuracy_degraded = current_accuracy < (self.baseline['accuracy'] * 0.95)  # 5% drop
        f1_degraded = current_f1 < (self.baseline['f1'] * 0.95)

        wandb.log({
            "performance/accuracy": current_accuracy,
            "performance/f1_score": current_f1,
            "performance/accuracy_degradation": accuracy_degraded,
            "performance/f1_degradation": f1_degraded,
            "alerts/degradation_detected": accuracy_degraded or f1_degraded
        })

        # Trigger alert
        if accuracy_degraded or f1_degraded:
            wandb.alert(
                title="Model Performance Degradation",
                text=f"Accuracy: {current_accuracy:.3f} (baseline: {self.baseline['accuracy']:.3f})\n"
                     f"F1: {current_f1:.3f} (baseline: {self.baseline['f1']:.3f})",
                level=wandb.AlertLevel.WARN
            )
```

From [Automated Drift Detection](https://wandb.ai/onlineinference/genai-research/reports/LLM-observability-Enhancing-AI-systems-with-W-B-Weave--VmlldzoxMjY4MjMwNQ) (accessed 2025-01-31):
- Alert when accuracy drops >5% from baseline
- Monitor prediction trends for gradual degradation
- Combine metrics: accuracy, latency, error rate

---

## Section 3: Integration Patterns (~120 lines)

### FastAPI + W&B Monitoring

Integrate W&B monitoring into FastAPI production server.

**FastAPI Middleware Pattern:**

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import wandb
import time

app = FastAPI()

# Initialize W&B run at startup
@app.on_event("startup")
async def startup_event():
    wandb.init(
        project="arr-coc-production",
        job_type="api-server",
        name=f"api-{datetime.now().strftime('%Y-%m-%d')}",
        config={
            "server_version": "1.0.3",
            "environment": "production"
        }
    )

# Monitoring middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request/response metrics"""
    start_time = time.time()

    # Process request
    try:
        response = await call_next(request)
        success = True
        status_code = response.status_code
    except Exception as e:
        success = False
        status_code = 500
        raise
    finally:
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log metrics (async, non-blocking)
        wandb.log({
            f"api/{request.url.path}/latency_ms": latency_ms,
            f"api/{request.url.path}/status": status_code,
            f"api/{request.url.path}/success": 1 if success else 0,
            "api/total_requests": 1  # Counter
        })

    return response

# Prediction endpoint
@app.post("/predict")
async def predict(image: bytes, query: str):
    """VLM prediction with monitoring"""

    # Run model
    result = model.predict(image, query)

    # Log prediction metrics
    wandb.log({
        "predictions/confidence": result.confidence,
        "predictions/tokens_allocated": result.tokens,
        "predictions/query_length": len(query)
    })

    return {
        "prediction": result.text,
        "confidence": result.confidence
    }
```

### Gradio App Monitoring

Track user interactions in Gradio demos.

**Gradio Event Logging:**

```python
import gradio as gr
import wandb

# Initialize W&B
wandb.init(project="arr-coc-demo", job_type="gradio-app")

def predict_with_logging(image, query):
    """Gradio prediction function with W&B logging"""

    start_time = datetime.now()

    # Run model
    result = model(image, query)

    latency_ms = (datetime.now() - start_time).total_seconds() * 1000

    # Log to W&B
    wandb.log({
        "demo/latency_ms": latency_ms,
        "demo/query_length": len(query),
        "demo/tokens_allocated": result.tokens_allocated,
        "demo/prediction": result.text,
        "demo/image": wandb.Image(image, caption=query)
    })

    return result.text, result.visualization

# Create Gradio interface
demo = gr.Interface(
    fn=predict_with_logging,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Query", placeholder="What do you want to know?")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Image(label="Relevance Map")
    ],
    title="ARR-COC Vision-Language Model",
    examples=[
        ["examples/image1.jpg", "What is the main object?"],
        ["examples/image2.jpg", "Describe the scene in detail"]
    ]
)

if __name__ == "__main__":
    demo.launch()
```

From [Gradio and W&B Integration](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):
- Gradio automatically logs to W&B when `wandb` module is imported
- Track user interactions and feedback
- Monitor demo usage patterns

### Streamlit Integration

Monitor Streamlit apps with W&B.

**Streamlit Session Tracking:**

```python
import streamlit as st
import wandb

# Initialize W&B (once per session)
if 'wandb_run' not in st.session_state:
    st.session_state.wandb_run = wandb.init(
        project="arr-coc-streamlit",
        job_type="streamlit-app",
        config={"user_session": st.session_state.get('session_id')}
    )

st.title("ARR-COC Vision-Language Model")

# User inputs
image = st.file_uploader("Upload Image", type=['jpg', 'png'])
query = st.text_input("Enter your question")

if st.button("Predict"):
    if image and query:
        # Run prediction
        result = model(image, query)

        # Display result
        st.write(f"**Answer:** {result.text}")
        st.image(result.relevance_map, caption="Relevance Map")

        # Log to W&B
        st.session_state.wandb_run.log({
            "streamlit/query": query,
            "streamlit/tokens": result.tokens_allocated,
            "streamlit/image": wandb.Image(image, caption=query)
        })
```

### Async Logging Patterns

Don't block production requests with logging.

**Non-Blocking W&B Logging:**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncWandbLogger:
    """Async W&B logger to avoid blocking"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.log_queue = asyncio.Queue()

    async def log_async(self, metrics):
        """Add metrics to queue"""
        await self.log_queue.put(metrics)

    async def _background_logger(self):
        """Background task to flush logs"""
        while True:
            metrics = await self.log_queue.get()

            # Log in background thread (non-blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                wandb.log,
                metrics
            )

# Usage in FastAPI
logger = AsyncWandbLogger()

@app.on_event("startup")
async def start_logger():
    asyncio.create_task(logger._background_logger())

@app.post("/predict")
async def predict(image: bytes, query: str):
    result = model(image, query)

    # Non-blocking log
    await logger.log_async({
        "inference/latency": result.latency_ms
    })

    return result
```

### Cost and Usage Tracking

Monitor API costs and resource usage.

**Token and Cost Tracking:**

```python
class CostMonitor:
    """Track inference costs"""

    # Cost per 1M tokens (example pricing)
    COST_PER_MILLION_TOKENS = {
        'input': 0.15,   # $0.15 per 1M input tokens
        'output': 0.60   # $0.60 per 1M output tokens
    }

    def __init__(self):
        self.daily_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'requests': 0
        }

    def record_request(self, input_tokens, output_tokens):
        """Record token usage"""
        self.daily_usage['input_tokens'] += input_tokens
        self.daily_usage['output_tokens'] += output_tokens
        self.daily_usage['requests'] += 1

        # Calculate costs
        input_cost = (self.daily_usage['input_tokens'] / 1e6) * self.COST_PER_MILLION_TOKENS['input']
        output_cost = (self.daily_usage['output_tokens'] / 1e6) * self.COST_PER_MILLION_TOKENS['output']
        total_cost = input_cost + output_cost

        wandb.log({
            "cost/daily_input_tokens": self.daily_usage['input_tokens'],
            "cost/daily_output_tokens": self.daily_usage['output_tokens'],
            "cost/daily_requests": self.daily_usage['requests'],
            "cost/daily_total_usd": total_cost,
            "cost/avg_cost_per_request_usd": total_cost / max(self.daily_usage['requests'], 1)
        })
```

From [LLM Cost Tracking](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) (accessed 2025-01-31):
- Track token usage for cost attribution
- Monitor per-user and per-endpoint costs
- Set budget alerts with `wandb.alert()`

---

## Sources

**Web Research:**

- [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31) - Production logging patterns
- [LLM Debugging and Monitoring](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) (accessed 2025-01-31) - Weave observability, cost tracking
- [Foundation Model Builder Tips](https://wandb.ai/wandb/report/reports/Foundation-Model-Builder-Tips-Best-Practices--Vmlldzo5MTA3MDI2) (accessed 2025-01-31) - Large-scale training best practices
- [Data Drift Detection](https://wandb.ai/yujiewang/data-drift-detector/reports/Drift-Detection-Progress-Report--VmlldzoyODY3NDk1) (accessed 2025-01-31) - Statistical drift detection methods
- [Automated Drift Detection](https://wandb.ai/onlineinference/genai-research/reports/LLM-observability-Enhancing-AI-systems-with-W-B-Weave--VmlldzoxMjY4MjMwNQ) (accessed 2025-01-31) - Performance degradation alerts
- [Gradio W&B Integration](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31) - Gradio monitoring patterns
- [Model Monitoring Guide](https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide) (accessed 2025-01-31) - Sampling strategies, operational monitoring
- [W&B Documentation - App UI](https://docs.wandb.ai/guides/app/) (accessed 2025-01-31) - Workspace settings, panel configuration

**Additional References:**

- [W&B Experiment Tracking](https://wandb.ai/site/experiment-tracking/) (accessed 2025-01-31) - Platform overview
