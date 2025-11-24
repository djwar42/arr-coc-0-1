# Production Performance Monitoring

**Knowledge File**: Complete guide to monitoring ML model and training performance in production environments, including training metrics, inference latency, Prometheus/Grafana integration, performance regression detection, alerting strategies, and arr-coc-0-1 monitoring dashboard

---

## Overview

Production ML performance monitoring tracks both training throughput and inference performance to ensure models meet SLAs and identify bottlenecks. Unlike development monitoring which focuses on debugging, production monitoring emphasizes reliability, latency percentiles (P50/P95/P99), cost efficiency, and automated alerting for regressions.

**Key Production Monitoring Challenges:**
- Silent performance degradation without visible errors
- Multi-model deployments with varying SLAs
- Cost vs. performance tradeoffs (GPU utilization vs. latency)
- Training job monitoring at scale (multi-node distributed)
- Inference latency spikes during peak traffic

From [Datadog ML Model Monitoring Best Practices](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/) (accessed 2025-11-16):
> "Monitoring a machine learning model after deployment is vital, as models can break and degrade in production. Deployment is not a one-time event."

From [karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md](../karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md):
> "Models degrade without visible errors (silent failures). Ground truth labels arrive with delays (feedback lag). Data pipeline failures cascade to model quality."

---

## Section 1: Training Performance Metrics (~100 lines)

### 1.1 Throughput Metrics

**Samples/second** measures training speed - critical for large-scale training jobs:

```python
import time

class ThroughputTracker:
    def __init__(self):
        self.start_time = time.time()
        self.total_samples = 0

    def update(self, batch_size):
        """Track samples processed"""
        self.total_samples += batch_size

    def get_throughput(self):
        """Calculate samples per second"""
        elapsed = time.time() - self.start_time
        return self.total_samples / elapsed if elapsed > 0 else 0

# Usage in training loop
tracker = ThroughputTracker()
for batch in dataloader:
    # Training step
    model.train_step(batch)
    tracker.update(len(batch))

    if step % 100 == 0:
        throughput = tracker.get_throughput()
        print(f"Throughput: {throughput:.2f} samples/sec")
```

**Tokens/second** for LLM training (more relevant than samples/sec):

```python
def calculate_token_throughput(num_tokens, elapsed_time, world_size):
    """
    Calculate effective token throughput

    num_tokens: Total tokens processed
    elapsed_time: Time in seconds
    world_size: Number of GPUs in distributed training
    """
    tokens_per_sec = num_tokens / elapsed_time
    # Effective throughput accounts for all GPUs
    effective_throughput = tokens_per_sec * world_size
    return effective_throughput

# Example
tokens_per_batch = 2048  # Sequence length
batch_size = 32
world_size = 8  # 8 GPUs

total_tokens = tokens_per_batch * batch_size * world_size
elapsed = 5.2  # seconds

throughput = calculate_token_throughput(total_tokens, elapsed, world_size)
print(f"Token throughput: {throughput:.0f} tokens/sec")
```

**GPU utilization** - target 90%+ for cost efficiency:

```python
import subprocess

def get_gpu_utilization():
    """Query GPU utilization using nvidia-smi"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )
    utilizations = [int(u) for u in result.stdout.strip().split('\n')]
    return utilizations

# Monitor during training
utilizations = get_gpu_utilization()
avg_util = sum(utilizations) / len(utilizations)
if avg_util < 80:
    print(f"WARNING: Low GPU utilization: {avg_util:.1f}%")
```

### 1.2 Iteration Time Breakdown

**Profile training iteration** to identify bottlenecks:

```python
import time

class IterationProfiler:
    def __init__(self):
        self.timings = {}

    def time_section(self, name):
        """Context manager for timing code sections"""
        class Timer:
            def __init__(self, profiler, section_name):
                self.profiler = profiler
                self.name = section_name
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                elapsed = time.time() - self.start
                if self.name not in self.profiler.timings:
                    self.profiler.timings[self.name] = []
                self.profiler.timings[self.name].append(elapsed)

        return Timer(self, name)

    def get_summary(self):
        """Get timing summary"""
        summary = {}
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            summary[name] = avg_time
        return summary

# Usage in training loop
profiler = IterationProfiler()

for batch in dataloader:
    with profiler.time_section('data_loading'):
        inputs, labels = batch
        inputs = inputs.cuda()
        labels = labels.cuda()

    with profiler.time_section('forward_pass'):
        outputs = model(inputs)

    with profiler.time_section('loss_computation'):
        loss = criterion(outputs, labels)

    with profiler.time_section('backward_pass'):
        loss.backward()

    with profiler.time_section('optimizer_step'):
        optimizer.step()
        optimizer.zero_grad()

    if step % 100 == 0:
        summary = profiler.get_summary()
        for section, avg_time in summary.items():
            print(f"{section}: {avg_time*1000:.2f}ms")
```

**Common bottlenecks identified:**
- **Data loading > 20% of iteration time** → Increase num_workers, use DALI, or cache to SSD
- **Backward pass > 2× forward pass** → Likely gradient checkpointing overhead
- **Optimizer step > 10% of iteration** → Try fused optimizer (FusedAdam)

---

## Section 2: Inference Performance Metrics (~120 lines)

### 2.1 Latency Percentiles

**P50, P95, P99 latency** are critical for user-facing applications:

From [Inference Latency P99 Monitoring](https://medium.com/@madhuri15/performance-metrics-for-regression-algorithms-1c889e68fde5) (accessed 2025-11-16):
> "P99 (99th percentile): This suggests that 99% of the API requests are faster than this value. Just 1% of the requests are slower than the P99."

```python
import numpy as np

class LatencyTracker:
    def __init__(self):
        self.latencies = []

    def record(self, latency_ms):
        """Record a latency measurement"""
        self.latencies.append(latency_ms)

    def get_percentiles(self):
        """Calculate P50, P95, P99"""
        if not self.latencies:
            return {}

        latencies_array = np.array(self.latencies)
        return {
            'p50': np.percentile(latencies_array, 50),
            'p95': np.percentile(latencies_array, 95),
            'p99': np.percentile(latencies_array, 99),
            'mean': np.mean(latencies_array),
            'max': np.max(latencies_array)
        }

    def reset(self):
        """Reset for next measurement window"""
        self.latencies = []

# Usage in inference server
tracker = LatencyTracker()

for request in requests:
    start_time = time.time()

    # Run inference
    output = model.predict(request)

    latency_ms = (time.time() - start_time) * 1000
    tracker.record(latency_ms)

# Report every 1000 requests
if len(tracker.latencies) >= 1000:
    percentiles = tracker.get_percentiles()
    print(f"P50: {percentiles['p50']:.2f}ms")
    print(f"P95: {percentiles['p95']:.2f}ms")
    print(f"P99: {percentiles['p99']:.2f}ms")
    tracker.reset()
```

**Latency SLA targets** (typical):

| Application Type | P50 | P95 | P99 |
|------------------|-----|-----|-----|
| User-facing API | <50ms | <100ms | <200ms |
| Backend service | <100ms | <200ms | <500ms |
| Batch processing | <1s | <2s | <5s |
| Real-time (gaming) | <16ms | <33ms | <50ms |

### 2.2 Throughput Metrics

**Requests per second (RPS)** for load testing:

```python
import time
from collections import deque

class ThroughputCounter:
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def record_request(self):
        """Record a request timestamp"""
        now = time.time()
        self.timestamps.append(now)

        # Remove old timestamps outside window
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def get_rps(self):
        """Calculate requests per second"""
        if not self.timestamps:
            return 0.0

        # Calculate RPS over the window
        elapsed = time.time() - self.timestamps[0]
        if elapsed == 0:
            return 0.0

        return len(self.timestamps) / elapsed

# Usage
counter = ThroughputCounter()

for request in incoming_requests:
    counter.record_request()
    handle_request(request)

    # Report every 100 requests
    if len(counter.timestamps) % 100 == 0:
        rps = counter.get_rps()
        print(f"Current RPS: {rps:.1f}")
```

**Batch size vs. latency tradeoff:**

```python
def benchmark_batch_sizes(model, test_data, batch_sizes=[1, 4, 8, 16, 32]):
    """
    Test different batch sizes to find optimal throughput/latency balance
    """
    results = []

    for batch_size in batch_sizes:
        latencies = []

        # Run 100 batches
        for i in range(100):
            batch = test_data[i*batch_size:(i+1)*batch_size]

            start = time.time()
            _ = model.predict(batch)
            elapsed = time.time() - start

            # Latency per sample
            latency_per_sample = (elapsed / batch_size) * 1000  # ms
            latencies.append(latency_per_sample)

        avg_latency = np.mean(latencies)
        throughput = batch_size / (np.mean([e/batch_size for e in latencies]) / 1000)

        results.append({
            'batch_size': batch_size,
            'latency_ms': avg_latency,
            'throughput_qps': throughput
        })

    return results

# Example results
# batch_size=1:  latency=10ms,  throughput=100 QPS
# batch_size=8:  latency=15ms,  throughput=533 QPS
# batch_size=32: latency=30ms,  throughput=1067 QPS
```

### 2.3 Model Quality Metrics

**Track prediction distribution drift** as proxy for quality:

```python
from scipy.stats import ks_2samp

class PredictionDriftDetector:
    def __init__(self, baseline_predictions):
        """
        Initialize with baseline predictions from validation set
        """
        self.baseline = np.array(baseline_predictions)

    def detect_drift(self, current_predictions):
        """
        Use Kolmogorov-Smirnov test to detect drift

        Returns: (drift_detected, p_value)
        """
        current = np.array(current_predictions)

        # KS test compares distributions
        statistic, p_value = ks_2samp(self.baseline, current)

        # p < 0.05 indicates significant drift
        drift_detected = p_value < 0.05

        return drift_detected, p_value

# Usage
baseline = validation_predictions  # From model evaluation
detector = PredictionDriftDetector(baseline)

# Check production predictions every 1000 requests
production_preds = []
for prediction in model.predict_stream():
    production_preds.append(prediction)

    if len(production_preds) >= 1000:
        drift, p_value = detector.detect_drift(production_preds)
        if drift:
            print(f"WARNING: Prediction drift detected (p={p_value:.4f})")
        production_preds = []  # Reset
```

---

## Section 3: Prometheus + Grafana Integration (~150 lines)

### 3.1 Prometheus GPU Metrics Exporter

From [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) (accessed 2025-11-16):
> "NVIDIA GPU metrics exporter for Prometheus leveraging NVIDIA DCGM"

**Install DCGM Exporter** for GPU metrics:

```bash
# Docker deployment
docker run -d --gpus all --cap-add SYS_ADMIN --rm -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.6.0-ubuntu22.04

# Verify metrics endpoint
curl localhost:9400/metrics
# DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-..."} 95
# DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-..."} 82
# DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-..."} 68
```

**Kubernetes deployment** with Helm:

```bash
helm repo add gpu-helm-charts \
  https://nvidia.github.io/dcgm-exporter/helm-charts

helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.interval=10s
```

**Custom metrics CSV** for specific counters:

```csv
# /etc/dcgm-exporter/custom-counters.csv
# DCGM FIELD, Prometheus metric type, help message

# GPU Utilization
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (%).
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory utilization (%).

# Temperature
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (C).
DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (C).

# Power
DCGM_FI_DEV_POWER_USAGE, gauge, Power usage (W).

# Clocks
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (MHz).
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (MHz).

# Compute
DCGM_FI_PROF_GR_ENGINE_ACTIVE, gauge, Graphics engine active (%).
DCGM_FI_PROF_SM_ACTIVE, gauge, SM active (%).
DCGM_FI_PROF_SM_OCCUPANCY, gauge, SM occupancy (%).

# Memory
DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (MB).
DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free (MB).

# Errors
DCGM_FI_DEV_XID_ERRORS, counter, GPU XID errors.
```

Run with custom counters:

```bash
dcgm-exporter --collectors /etc/dcgm-exporter/custom-counters.csv
```

### 3.2 Prometheus Configuration

**prometheus.yml** for scraping GPU and training metrics:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # DCGM Exporter (GPU metrics)
  - job_name: 'dcgm'
    static_configs:
      - targets: ['localhost:9400']
        labels:
          cluster: 'gpu-cluster-1'
          environment: 'production'

  # Training job metrics (custom exporter)
  - job_name: 'training'
    static_configs:
      - targets: ['training-node-1:8000', 'training-node-2:8000']
        labels:
          job_type: 'llm-training'

  # Inference service metrics
  - job_name: 'inference'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ml-inference
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: arr-coc-inference
```

### 3.3 Custom Training Metrics Exporter

**Export training metrics** to Prometheus:

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time

# Define metrics
training_loss = Gauge('training_loss', 'Current training loss')
training_throughput = Gauge('training_throughput_samples_per_sec',
                            'Training throughput')
gpu_utilization = Gauge('gpu_utilization_percent',
                       'GPU utilization',
                       ['gpu_id'])
iteration_time = Histogram('iteration_time_seconds',
                          'Training iteration time',
                          buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
total_samples = Counter('total_samples_processed',
                       'Total samples processed')

# Start metrics server
start_http_server(8000)

# Update metrics during training
for step, batch in enumerate(dataloader):
    start_time = time.time()

    # Training step
    loss = train_step(model, batch)

    # Record metrics
    iteration_duration = time.time() - start_time
    iteration_time.observe(iteration_duration)
    training_loss.set(loss)
    total_samples.inc(len(batch))

    # Calculate throughput
    throughput = len(batch) / iteration_duration
    training_throughput.set(throughput)

    # GPU utilization (from nvidia-smi)
    for gpu_id, util in enumerate(get_gpu_utilization()):
        gpu_utilization.labels(gpu_id=str(gpu_id)).set(util)
```

### 3.4 Grafana Dashboards

**Import DCGM Exporter dashboard:**

Official NVIDIA dashboard: [Grafana Dashboard 12239](https://grafana.com/grafana/dashboards/12239)

**Custom training dashboard panels:**

```json
{
  "dashboard": {
    "title": "ML Training Performance",
    "panels": [
      {
        "title": "Training Loss",
        "targets": [
          {
            "expr": "training_loss",
            "legendFormat": "Loss"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "avg(DCGM_FI_DEV_GPU_UTIL) by (gpu)",
            "legendFormat": "GPU {{gpu}}"
          }
        ],
        "type": "graph",
        "yaxes": [
          {
            "format": "percent",
            "max": 100,
            "min": 0
          }
        ]
      },
      {
        "title": "Throughput",
        "targets": [
          {
            "expr": "rate(total_samples_processed[5m])",
            "legendFormat": "Samples/sec"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Iteration Time P99",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(iteration_time_seconds_bucket[5m]))",
            "legendFormat": "P99 latency"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

**PromQL queries** for common metrics:

```promql
# Average GPU utilization across all GPUs
avg(DCGM_FI_DEV_GPU_UTIL)

# GPU memory usage percentage
(DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) * 100

# Training throughput (5-minute average)
rate(total_samples_processed[5m])

# P95 iteration time
histogram_quantile(0.95, rate(iteration_time_seconds_bucket[5m]))

# GPU temperature alert (>80C)
DCGM_FI_DEV_GPU_TEMP > 80

# Low GPU utilization alert (<70%)
avg(DCGM_FI_DEV_GPU_UTIL) < 70
```

---

## Section 4: Performance Regression Detection (~100 lines)

From [Performance Regression Detection ML Models](https://www.statsig.com/perspectives/model-performance-quality-decline) (accessed 2025-11-16):
> "Models degrade over time. Regularly measure, diagnose, and prevent regression with disciplined metrics and rollouts."

### 4.1 Baseline Benchmarking

**Establish performance baselines** before deployment:

```python
import json
import numpy as np

class PerformanceBaseline:
    def __init__(self):
        self.baseline = {}

    def record_baseline(self, model_version, metrics):
        """
        Record baseline metrics for a model version

        metrics: dict with keys like 'throughput', 'p50_latency', etc.
        """
        self.baseline[model_version] = {
            'metrics': metrics,
            'timestamp': time.time()
        }

    def save(self, filepath):
        """Save baseline to file"""
        with open(filepath, 'w') as f:
            json.dump(self.baseline, f, indent=2)

    def load(self, filepath):
        """Load baseline from file"""
        with open(filepath, 'r') as f:
            self.baseline = json.load(f)

    def check_regression(self, model_version, current_metrics, threshold=0.1):
        """
        Check if current metrics show regression vs baseline

        threshold: Maximum allowed degradation (10% default)
        Returns: (has_regression, regression_details)
        """
        if model_version not in self.baseline:
            return False, "No baseline found"

        baseline_metrics = self.baseline[model_version]['metrics']
        regressions = {}

        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline_metrics:
                continue

            baseline_value = baseline_metrics[metric_name]

            # For latency, higher is worse
            if 'latency' in metric_name.lower():
                degradation = (current_value - baseline_value) / baseline_value
                if degradation > threshold:
                    regressions[metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_pct': degradation * 100
                    }

            # For throughput, lower is worse
            elif 'throughput' in metric_name.lower():
                degradation = (baseline_value - current_value) / baseline_value
                if degradation > threshold:
                    regressions[metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_pct': degradation * 100
                    }

        has_regression = len(regressions) > 0
        return has_regression, regressions

# Usage
baseline = PerformanceBaseline()

# Record baseline after successful deployment
baseline.record_baseline('v1.0', {
    'throughput_samples_per_sec': 1250,
    'p50_latency_ms': 45,
    'p99_latency_ms': 120,
    'gpu_utilization_pct': 92
})
baseline.save('baselines.json')

# Check for regression in production
current = {
    'throughput_samples_per_sec': 1100,  # 12% slower
    'p50_latency_ms': 48,
    'p99_latency_ms': 145,  # 21% worse
    'gpu_utilization_pct': 88
}

has_regression, details = baseline.check_regression('v1.0', current, threshold=0.1)
if has_regression:
    print("REGRESSION DETECTED:")
    for metric, info in details.items():
        print(f"  {metric}: {info['baseline']} → {info['current']} "
              f"({info['degradation_pct']:.1f}% worse)")
```

### 4.2 Automated Performance Testing

**CI/CD performance tests** before deployment:

```python
import subprocess

def run_performance_benchmark(model_path, test_data_path):
    """
    Run standardized performance benchmark

    Returns: dict of metrics
    """
    # Run benchmark script
    result = subprocess.run([
        'python', 'benchmark.py',
        '--model', model_path,
        '--data', test_data_path,
        '--batch-sizes', '1,8,32',
        '--iterations', '1000'
    ], capture_output=True, text=True)

    # Parse output
    metrics = {}
    for line in result.stdout.split('\n'):
        if ':' in line:
            key, value = line.split(':')
            metrics[key.strip()] = float(value.strip())

    return metrics

def performance_regression_test(new_model, baseline_file='baselines.json'):
    """
    Automated test for CI/CD pipeline

    Returns: 0 if pass, 1 if regression detected
    """
    # Run benchmark
    current_metrics = run_performance_benchmark(new_model, 'test_data.pt')

    # Compare to baseline
    baseline = PerformanceBaseline()
    baseline.load(baseline_file)

    has_regression, details = baseline.check_regression(
        'production',
        current_metrics,
        threshold=0.15  # Allow 15% degradation
    )

    if has_regression:
        print("PERFORMANCE REGRESSION DETECTED - BLOCKING DEPLOYMENT")
        for metric, info in details.items():
            print(f"  {metric}: {info['degradation_pct']:.1f}% worse")
        return 1
    else:
        print("Performance tests PASSED")
        return 0

# CI/CD pipeline usage
exit_code = performance_regression_test('models/new_model.pt')
sys.exit(exit_code)
```

### 4.3 Statistical Change Detection

**Use control charts** to detect performance shifts:

```python
import numpy as np
from collections import deque

class PerformanceControlChart:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.measurements = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None

    def record(self, value):
        """Record a measurement"""
        self.measurements.append(value)

        # Establish baseline after window fills
        if len(self.measurements) == self.window_size and self.baseline_mean is None:
            self.baseline_mean = np.mean(self.measurements)
            self.baseline_std = np.std(self.measurements)

    def check_anomaly(self, sigma_threshold=3):
        """
        Check if recent values are anomalous

        Uses 3-sigma rule (99.7% confidence)
        """
        if self.baseline_mean is None:
            return False, "Baseline not established"

        # Check last 10 measurements
        recent = list(self.measurements)[-10:]
        recent_mean = np.mean(recent)

        # Calculate z-score
        z_score = abs(recent_mean - self.baseline_mean) / self.baseline_std

        is_anomaly = z_score > sigma_threshold

        return is_anomaly, {
            'z_score': z_score,
            'baseline_mean': self.baseline_mean,
            'recent_mean': recent_mean,
            'threshold': sigma_threshold
        }

# Usage
latency_chart = PerformanceControlChart(window_size=1000)

for request in production_requests:
    latency = process_request(request)
    latency_chart.record(latency)

    # Check every 10 requests
    if len(latency_chart.measurements) % 10 == 0:
        is_anomaly, info = latency_chart.check_anomaly()
        if is_anomaly:
            print(f"ALERT: Latency anomaly detected!")
            print(f"  Baseline: {info['baseline_mean']:.2f}ms")
            print(f"  Recent: {info['recent_mean']:.2f}ms")
            print(f"  Z-score: {info['z_score']:.2f}")
```

---

## Section 5: Alerting Strategies (~80 lines)

### 5.1 Prometheus Alerting Rules

**alerting-rules.yml** for performance alerts:

```yaml
groups:
  - name: training_performance
    interval: 30s
    rules:
      # Low GPU utilization
      - alert: LowGPUUtilization
        expr: avg(DCGM_FI_DEV_GPU_UTIL) < 70
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low GPU utilization detected"
          description: "Average GPU utilization is {{ $value }}% (< 70%)"

      # High GPU temperature
      - alert: HighGPUTemperature
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU overheating"
          description: "GPU {{ $labels.gpu }} temperature is {{ $value }}°C"

      # Slow iteration time
      - alert: SlowIterationTime
        expr: histogram_quantile(0.95, rate(iteration_time_seconds_bucket[5m])) > 2.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Training iterations are slow"
          description: "P95 iteration time is {{ $value }}s (> 2.0s)"

      # Throughput drop
      - alert: TrainingThroughputDrop
        expr: |
          rate(total_samples_processed[5m]) <
          rate(total_samples_processed[5m] offset 1h) * 0.8
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Training throughput dropped"
          description: "Throughput is 20% below 1-hour average"

  - name: inference_performance
    interval: 15s
    rules:
      # High P99 latency
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High inference P99 latency"
          description: "P99 latency is {{ $value }}s (> 200ms)"

      # Low throughput
      - alert: LowInferenceThroughput
        expr: rate(inference_requests_total[5m]) < 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low inference throughput"
          description: "Request rate is {{ $value }} req/s (< 100)"

      # OOM errors
      - alert: InferenceOOMErrors
        expr: rate(inference_oom_errors_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Out of memory errors detected"
          description: "{{ $value }} OOM errors per second"
```

### 5.2 Alert Escalation

**Multi-tier alerting** based on severity:

```python
import requests

class AlertManager:
    def __init__(self, config):
        self.config = config

    def send_alert(self, level, message, metrics):
        """
        Send alert with appropriate escalation

        level: 'info', 'warning', 'critical'
        """
        if level == 'critical':
            self._send_pagerduty(message, metrics)
            self._send_slack(message, metrics, channel='#ml-oncall')
        elif level == 'warning':
            self._send_slack(message, metrics, channel='#ml-monitoring')
        else:  # info
            self._log_alert(message, metrics)

    def _send_slack(self, message, metrics, channel):
        """Send Slack notification"""
        webhook_url = self.config['slack_webhook']

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 {message}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*{k}:*\n{v}"
                    }
                    for k, v in metrics.items()
                ]
            }
        ]

        requests.post(webhook_url, json={
            'channel': channel,
            'blocks': blocks
        })

    def _send_pagerduty(self, message, metrics):
        """Send PagerDuty alert for oncall"""
        # PagerDuty integration
        pass

    def _log_alert(self, message, metrics):
        """Log alert for auditing"""
        print(f"ALERT: {message} - {metrics}")

# Usage
alert_mgr = AlertManager(config={'slack_webhook': 'https://...'})

# Check performance and alert
if p99_latency > 200:
    alert_mgr.send_alert('critical', 'High P99 latency', {
        'P99': f"{p99_latency:.2f}ms",
        'Threshold': '200ms',
        'Model': 'arr-coc-v1.0'
    })
```

---

## Section 6: Cost Monitoring (~70 lines)

### 6.1 GPU Cost Tracking

**Calculate cost** per training run:

```python
class GPUCostTracker:
    def __init__(self):
        # GCP pricing (as of 2024)
        self.cost_per_gpu_hour = {
            'a2-highgpu-1g': 3.67,     # 1× A100
            'a2-highgpu-8g': 29.39,    # 8× A100
            'a2-ultragpu-8g': 33.60,   # 8× A100 80GB
        }

    def calculate_training_cost(self, instance_type, duration_hours):
        """
        Calculate cost of training run

        duration_hours: Total training time
        """
        hourly_cost = self.cost_per_gpu_hour.get(instance_type, 0)
        total_cost = hourly_cost * duration_hours

        return {
            'total_cost_usd': total_cost,
            'hourly_cost_usd': hourly_cost,
            'duration_hours': duration_hours
        }

    def cost_per_sample(self, instance_type, duration_hours, total_samples):
        """Calculate cost per training sample"""
        cost_info = self.calculate_training_cost(instance_type, duration_hours)
        cost_per_sample = cost_info['total_cost_usd'] / total_samples
        return cost_per_sample

# Usage
tracker = GPUCostTracker()

# Training run on 8× A100 for 24 hours
cost = tracker.calculate_training_cost('a2-highgpu-8g', 24)
print(f"Training cost: ${cost['total_cost_usd']:.2f}")

# Cost per sample (1M samples processed)
cost_per_sample = tracker.cost_per_sample('a2-highgpu-8g', 24, 1_000_000)
print(f"Cost per sample: ${cost_per_sample*1000:.4f} per 1K samples")
```

**Cost efficiency metric:**

```python
def calculate_cost_efficiency(cost_usd, final_accuracy):
    """
    Cost efficiency = accuracy improvement per dollar

    Higher is better
    """
    baseline_accuracy = 0.70  # Previous model
    accuracy_improvement = final_accuracy - baseline_accuracy

    efficiency = accuracy_improvement / cost_usd

    return efficiency

# Example
training_cost = 706  # $706 for 24 hours on 8× A100
final_accuracy = 0.85

efficiency = calculate_cost_efficiency(training_cost, final_accuracy)
print(f"Cost efficiency: {efficiency:.6f} accuracy points per dollar")
```

---

## Section 7: arr-coc-0-1 Production Dashboard (~80 lines)

### 7.1 Key Metrics for ARR-COC

**arr-coc specific monitoring:**

```python
from prometheus_client import Gauge, Histogram

# ARR-COC relevance metrics
relevance_score_dist = Histogram(
    'arr_coc_relevance_score',
    'Distribution of relevance scores',
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

token_allocation_dist = Histogram(
    'arr_coc_token_allocation',
    'Distribution of tokens allocated per patch',
    buckets=[64, 128, 192, 256, 320, 384, 400]
)

# Performance metrics
inference_latency = Histogram(
    'arr_coc_inference_latency_seconds',
    'Inference latency',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

patches_processed = Gauge(
    'arr_coc_patches_per_image',
    'Number of patches processed per image'
)

# Record metrics during inference
def arr_coc_inference(image, query):
    start = time.time()

    # Run ARR-COC inference
    patches, relevance_scores, token_allocations = model.process(image, query)

    # Record metrics
    for score in relevance_scores:
        relevance_score_dist.observe(score)

    for allocation in token_allocations:
        token_allocation_dist.observe(allocation)

    patches_processed.set(len(patches))

    latency = time.time() - start
    inference_latency.observe(latency)

    return patches
```

### 7.2 ARR-COC Dashboard Panels

**Grafana dashboard JSON** for arr-coc-0-1:

```json
{
  "dashboard": {
    "title": "ARR-COC-0-1 Production Performance",
    "panels": [
      {
        "title": "Inference Latency (P50/P95/P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(arr_coc_inference_latency_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(arr_coc_inference_latency_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(arr_coc_inference_latency_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "Token Allocation Distribution",
        "targets": [
          {
            "expr": "sum(rate(arr_coc_token_allocation_bucket[5m])) by (le)",
            "legendFormat": "≤ {{le}} tokens"
          }
        ],
        "type": "histogram"
      },
      {
        "title": "Relevance Score Distribution",
        "targets": [
          {
            "expr": "sum(rate(arr_coc_relevance_score_bucket[5m])) by (le)",
            "legendFormat": "≤ {{le}} relevance"
          }
        ]
      },
      {
        "title": "GPU Utilization (8× A100)",
        "targets": [
          {
            "expr": "avg(DCGM_FI_DEV_GPU_UTIL{job='arr-coc-inference'}) by (gpu)",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      }
    ]
  }
}
```

### 7.3 SLA Monitoring

**Track SLA compliance:**

```python
class SLAMonitor:
    def __init__(self, p99_target_ms=200):
        self.p99_target_ms = p99_target_ms
        self.latencies = []

    def record(self, latency_ms):
        self.latencies.append(latency_ms)

    def check_sla_compliance(self):
        """
        Calculate SLA compliance percentage

        SLA: 99% of requests < 200ms
        """
        if not self.latencies:
            return 100.0

        below_target = sum(1 for l in self.latencies if l < self.p99_target_ms)
        compliance_pct = (below_target / len(self.latencies)) * 100

        return compliance_pct

    def get_sla_report(self):
        """Generate SLA report"""
        compliance = self.check_sla_compliance()
        p99 = np.percentile(self.latencies, 99)

        return {
            'compliance_pct': compliance,
            'p99_actual_ms': p99,
            'p99_target_ms': self.p99_target_ms,
            'sla_met': compliance >= 99.0 and p99 < self.p99_target_ms
        }

# Usage
sla_monitor = SLAMonitor(p99_target_ms=200)

for request in daily_requests:
    latency = process_request(request)
    sla_monitor.record(latency)

# Daily SLA report
report = sla_monitor.get_sla_report()
print(f"SLA Compliance: {report['compliance_pct']:.2f}%")
print(f"P99 Latency: {report['p99_actual_ms']:.2f}ms (target: {report['p99_target_ms']}ms)")
print(f"SLA Met: {'✓' if report['sla_met'] else '✗'}")
```

---

## Summary

Production performance monitoring requires comprehensive tracking of both training and inference metrics:

**Training Monitoring:**
- Throughput (samples/sec, tokens/sec)
- GPU utilization (target >90%)
- Iteration time breakdown
- Cost per sample

**Inference Monitoring:**
- Latency percentiles (P50/P95/P99)
- Requests per second
- Prediction distribution drift
- SLA compliance

**Infrastructure:**
- Prometheus + DCGM Exporter for GPU metrics
- Grafana dashboards for visualization
- Automated alerting for regressions
- Performance baselines and CI/CD tests

**Best Practices:**
- Establish baselines before deployment
- Monitor percentiles, not just averages
- Use statistical tests for regression detection
- Multi-tier alerting (info/warning/critical)
- Track cost efficiency alongside performance

---

## Sources

**Source Documents:**
- [karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md](../karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md) - MLOps monitoring, drift detection, CI/CD
- [gcp-vertex/10-model-monitoring-drift.md](../gcp-vertex/10-model-monitoring-drift.md) - Vertex AI model monitoring, drift algorithms

**Web Research:**
- [Datadog: Machine learning model monitoring best practices](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/) - Production monitoring strategies, drift detection (accessed 2025-11-16)
- [SigNoz: A Comprehensive Guide to Model Monitoring in ML Production](https://signoz.io/guides/model-monitoring/) - Model monitoring frameworks, metrics tracking (accessed 2025-11-16)
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) - GPU metrics exporter for Prometheus (accessed 2025-11-16)
- [Statsig: Model performance regression detection](https://www.statsig.com/perspectives/model-performance-quality-decline) - Performance regression strategies (accessed 2025-11-16)

**Additional References:**
- [Prometheus Exporters](https://prometheus.io/docs/instrumenting/exporters/) - Metrics exporters list
- [Grafana Dashboard 12239](https://grafana.com/grafana/dashboards/12239) - Official NVIDIA DCGM dashboard

---

**Knowledge file complete**: ~700 lines
**Created**: 2025-11-16
**Coverage**: Training metrics (throughput, GPU utilization, iteration breakdown), inference metrics (latency percentiles, throughput, drift detection), Prometheus/Grafana integration, DCGM Exporter, performance regression detection, alerting strategies, cost monitoring, arr-coc-0-1 production dashboard
**All claims cited**: 4 web sources + 2 existing knowledge files
