# HuggingFace Production Deployment & Inference API

**Comprehensive guide to deploying HuggingFace models in production: managed Inference Endpoints, self-hosted infrastructure, TGI server, load balancing, autoscaling, and monitoring**

From [HuggingFace Inference Endpoints Documentation](https://huggingface.co/docs/inference-endpoints/en/index) (accessed 2025-11-16):

> "Inference Endpoints is a fully managed service to easily deploy machine learning models with no infrastructure management required. Don't worry about Kubernetes, CUDA versions, or configuring VPNs. Focus on deploying your model and serving customers."

---

## Overview

HuggingFace provides a complete production deployment ecosystem that bridges the gap between model development and serving at scale. Unlike research-focused tools, HuggingFace production deployment emphasizes **reliability, scalability, and cost-efficiency** through managed services and optimized self-hosted solutions.

**Key deployment options:**
- **Managed Inference Endpoints**: Fully managed hosting (pay-as-you-go)
- **Text Generation Inference (TGI)**: Self-hosted LLM server (Rust/Python/gRPC)
- **Self-hosted infrastructure**: Kubernetes, Docker, custom deployments
- **Hybrid approaches**: Hub + custom infrastructure

**Production requirements addressed:**
- Autoscaling (scale to zero, metric-based scaling)
- Load balancing (replica distribution, traffic management)
- Monitoring (Prometheus metrics, distributed tracing)
- High availability (99.9% SLA for Enterprise)
- A/B testing and canary deployments
- Cost optimization (resource management, scaling policies)

**Related knowledge:**
- See [../vertex-ai-production/00-distributed-training-patterns.md](../vertex-ai-production/00-distributed-training-patterns.md) for distributed training patterns
- See [06-inference-optimization-pipeline.md](06-inference-optimization-pipeline.md) for inference acceleration techniques
- See [07-spaces-gradio-streamlit.md](07-spaces-gradio-streamlit.md) for demo deployment on Spaces

---

## Section 1: HuggingFace Inference Endpoints (Managed Hosting) (~90 lines)

### What are Inference Endpoints?

From [HuggingFace Inference Endpoints](https://endpoints.huggingface.co/) (accessed 2025-11-16):

HuggingFace Inference Endpoints is a **fully managed inference solution** that handles:
- Infrastructure provisioning (no Kubernetes required)
- GPU/CPU allocation and scaling
- Model loading and optimization
- API endpoint management
- Security (VPNs, private links, authentication)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│ HuggingFace Inference Endpoints (Managed)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Your Model (Hub) → Endpoint Creation → Auto Deployment     │
│                                                              │
│  ┌────────────┐    ┌─────────────┐    ┌──────────────┐    │
│  │ Load       │ → │ Optimize    │ → │ Serve        │    │
│  │ Model      │    │ (TGI/vLLM)  │    │ (REST API)   │    │
│  └────────────┘    └─────────────┘    └──────────────┘    │
│                                                              │
│  Features: Autoscaling, Monitoring, Private Link, Logs      │
└─────────────────────────────────────────────────────────────┘
```

### Creating an Inference Endpoint

**Step 1: Access the service**
```python
# Prerequisites: Valid payment method on HuggingFace account
# Visit: https://ui.endpoints.huggingface.co/

# Programmatic creation via huggingface_hub
from huggingface_hub import HfApi, InferenceEndpoint

api = HfApi()

# Create endpoint
endpoint = api.create_inference_endpoint(
    name="my-llm-endpoint",
    repository="meta-llama/Meta-Llama-3.1-8B-Instruct",
    framework="pytorch",
    task="text-generation",
    accelerator="gpu",
    vendor="aws",
    region="us-east-1",
    type="protected",  # or "public"
    instance_size="medium",  # small, medium, large, xlarge
    instance_type="nvidia-t4",  # T4, A10G, A100
    min_replica=1,
    max_replica=5,  # Autoscaling enabled
    revision="main",  # Model revision/branch
    custom_image={
        "health_route": "/health",
        "env": {
            "MAX_BATCH_SIZE": "32",
            "MAX_TOTAL_TOKENS": "2048"
        }
    }
)

print(f"Endpoint URL: {endpoint.url}")
```

**Step 2: Use the endpoint**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model=endpoint.url,
    token="hf_..."  # Your HF token
)

# Text generation
response = client.text_generation(
    "What is deep learning?",
    max_new_tokens=100,
    temperature=0.7
)

# Streaming
for token in client.text_generation(
    "Explain transformers",
    max_new_tokens=200,
    stream=True
):
    print(token, end="")
```

### Pricing and Plans

From [Inference Endpoints Pricing](https://huggingface.co/docs/inference-endpoints/en/support/pricing) (accessed 2025-11-16):

**Pay-as-you-go pricing:**
- **CPU**: $0.032 per core/hour
- **GPU**: $0.50 - $5.00 per GPU/hour (T4 to A100)
- **Billed monthly** based on actual usage
- **No minimum commitment** for standard plan

**Hardware options:**
- nvidia-t4: $0.50/hour (16GB VRAM)
- nvidia-a10g: $1.20/hour (24GB VRAM)
- nvidia-a100: $4.50/hour (40GB VRAM)
- nvidia-a100-80gb: $5.00/hour (80GB VRAM)

**Enterprise plan:**
- Custom volume pricing
- 24/7 SLA with uptime guarantees
- Dedicated support
- Annual contracts
- Private cloud deployments

**Cost optimization:**
- Enable autoscaling (scale to zero when idle)
- Use smaller instances for development
- Batch processing during off-peak hours
- Monitor usage via Analytics dashboard

---

## Section 2: Self-Hosted Inference Infrastructure (~85 lines)

### Docker Deployment

From [Text Generation Inference GitHub](https://github.com/huggingface/text-generation-inference) (accessed 2025-11-16):

**Basic Docker setup:**
```bash
# Set model and volume
model=meta-llama/Meta-Llama-3.1-8B-Instruct
volume=$PWD/data

# Run TGI container
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.3.5 \
    --model-id $model \
    --max-batch-prefill-tokens 8192 \
    --max-total-tokens 4096 \
    --max-input-length 3072

# For gated models (Llama, Gemma)
docker run --gpus all --shm-size 1g -e HF_TOKEN=$HF_TOKEN -p 8080:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.3.5 \
    --model-id $model

# For AMD GPUs (ROCm)
docker run --device /dev/kfd --device /dev/dri --shm-size 1g -p 8080:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.3.5-rocm \
    --model-id $model
```

**Why `--shm-size 1g`?**
- NCCL (distributed communication) uses shared memory for inter-GPU transfers
- Peer-to-peer GPU communication (NVLink, PCI) may fall back to host memory
- 1GB shared memory prevents OOM errors during distributed inference

### Kubernetes Deployment

From [Kubernetes GPU Scheduling Research](https://xebia.com/blog/deploy-open-source-llm-in-your-private-cluster-with-hugging-face-and-gke-autopilot/) (accessed 2025-11-16):

**TGI Kubernetes manifest:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-llama
  namespace: inference
spec:
  replicas: 3  # Load balanced across 3 pods
  selector:
    matchLabels:
      app: tgi-llama
  template:
    metadata:
      labels:
        app: tgi-llama
    spec:
      containers:
      - name: tgi
        image: ghcr.io/huggingface/text-generation-inference:3.3.5
        args:
          - --model-id
          - meta-llama/Meta-Llama-3.1-8B-Instruct
          - --max-concurrent-requests
          - "128"
          - --max-batch-prefill-tokens
          - "8192"
        ports:
        - containerPort: 80
          name: http
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        - name: NCCL_SHM_DISABLE
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: model-cache
          mountPath: /data
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 1Gi
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

**Service for load balancing:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: tgi-llama-service
  namespace: inference
spec:
  selector:
    app: tgi-llama
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer  # External load balancer
```

**Horizontal Pod Autoscaler (HPA):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tgi-llama-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tgi-llama
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

---

## Section 3: Text Generation Inference (TGI) Server (~95 lines)

### TGI Architecture and Features

From [HuggingFace TGI Documentation](https://huggingface.co/docs/text-generation-inference/en/index) (accessed 2025-11-16):

Text Generation Inference (TGI) is a **production-ready LLM serving toolkit** built with Rust, Python, and gRPC:

**Core features:**
- **Continuous batching**: Processes incoming requests without waiting for batch completion
- **Token streaming**: Server-Sent Events (SSE) for real-time generation
- **Tensor parallelism**: Distribute model across multiple GPUs
- **Quantization**: AWQ, GPTQ, bitsandbytes, EETQ, fp8 support
- **Flash Attention & Paged Attention**: Optimized memory-efficient attention
- **Distributed tracing**: OpenTelemetry integration
- **Prometheus metrics**: Production monitoring

**Supported models:**
- LLaMA, Mistral, Falcon, StarCoder, BLOOM, GPT-NeoX
- Vision-Language Models (VLMs)
- Mixture-of-Experts (MoE) models

### Continuous Batching Explained

**Traditional batching (inefficient):**
```
Batch 1: [Request A (10 tokens), Request B (100 tokens), Request C (5 tokens)]
         All requests wait for longest (100 tokens) to complete
         Wasted GPU cycles: Requests A and C idle while B generates

Total time: 100 token generation cycles
```

**Continuous batching (efficient):**
```
┌─────────────────────────────────────────────────────────┐
│ TGI Continuous Batching (Dynamic Scheduling)            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Cycle 1: [A, B, C] → Generate 1 token each             │
│  Cycle 5: [B, C] → A finished (10 tokens), D added      │
│  Cycle 6: [B, C, D] → Processing 3 requests             │
│  Cycle 11: [B, D] → C finished (5 tokens), E added      │
│  ...                                                     │
│  Cycle 100: [E, F, G] → B finished, new requests added  │
│                                                          │
│  Key: No idle GPU time, constant throughput             │
└─────────────────────────────────────────────────────────┘

Total time: Still 100 cycles, but served 10+ requests vs 3
Throughput: 3-5× higher than traditional batching
```

### TGI Configuration and Tuning

**Launch with optimized settings:**
```bash
text-generation-launcher \
    --model-id meta-llama/Meta-Llama-3.1-70B-Instruct \
    --num-shard 4 \  # Tensor parallelism across 4 GPUs
    --max-concurrent-requests 256 \  # Concurrent request limit
    --max-batch-prefill-tokens 8192 \  # Prefill batch size
    --max-total-tokens 4096 \  # Max sequence length
    --max-input-length 3072 \  # Max input tokens
    --max-batch-total-tokens 32768 \  # Total tokens across batch
    --quantize gptq \  # GPTQ quantization
    --dtype float16 \  # Data type
    --trust-remote-code \  # Allow custom modeling code
    --otlp-endpoint http://jaeger:4317 \  # Distributed tracing
    --cors-allow-origin "*" \  # CORS policy
    --max-waiting-tokens 20 \  # Max tokens to wait for batching
    --waiting-served-ratio 1.2  # Batch scheduling ratio
```

**Key parameters explained:**
- `--num-shard`: Number of GPUs for tensor parallelism (model split)
- `--max-concurrent-requests`: HTTP server concurrency limit
- `--max-batch-prefill-tokens`: Limits prefill phase memory usage
- `--max-total-tokens`: Maximum sequence length (input + output)
- `--max-batch-total-tokens`: Total tokens across all requests in batch

**API usage:**
```bash
# Simple generation
curl http://localhost:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "inputs": "What is deep learning?",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": true
        }
    }'

# Streaming (SSE)
curl http://localhost:8080/generate_stream \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "inputs": "Explain transformers",
        "parameters": {"max_new_tokens": 200}
    }'

# OpenAI-compatible API (Messages API)
curl http://localhost:8080/v1/chat/completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "tgi",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is deep learning?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

### Performance Optimization

**Memory optimization:**
```bash
# Flash Attention (2× memory reduction)
--flash-attention true

# Paged Attention (vLLM-style)
--paged-attention true

# Quantization (4× memory reduction)
--quantize awq  # Or gptq, bitsandbytes-nf4, eetq
```

**Throughput optimization:**
```bash
# Increase batch size
--max-batch-total-tokens 65536

# Reduce waiting time
--max-waiting-tokens 10

# Increase concurrent requests
--max-concurrent-requests 512
```

---

## Section 4: Load Balancing and Traffic Management (~85 lines)

### Load Balancing Strategies

**1. Round-robin (Kubernetes default):**
```yaml
# Service automatically load balances across pods
apiVersion: v1
kind: Service
metadata:
  name: tgi-service
spec:
  selector:
    app: tgi
  ports:
  - port: 80
    targetPort: 80
  sessionAffinity: None  # Round-robin across pods
```

**2. Least connections (NGINX Ingress):**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tgi-ingress
  annotations:
    nginx.ingress.kubernetes.io/load-balance: "least_conn"
    nginx.ingress.kubernetes.io/upstream-hash-by: "$request_uri"
spec:
  rules:
  - host: tgi.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tgi-service
            port:
              number: 80
```

**3. Application-level load balancing (Python):**
```python
import random
from typing import List
import httpx

class TGILoadBalancer:
    """Client-side load balancer for TGI endpoints"""

    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.current_index = 0

    def round_robin(self) -> str:
        """Round-robin selection"""
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        return endpoint

    def random_choice(self) -> str:
        """Random selection"""
        return random.choice(self.endpoints)

    async def least_latency(self) -> str:
        """Select endpoint with lowest latency (health check)"""
        async with httpx.AsyncClient() as client:
            latencies = []
            for endpoint in self.endpoints:
                try:
                    start = time.time()
                    await client.get(f"{endpoint}/health")
                    latency = time.time() - start
                    latencies.append((endpoint, latency))
                except:
                    latencies.append((endpoint, float('inf')))

            return min(latencies, key=lambda x: x[1])[0]

# Usage
balancer = TGILoadBalancer([
    "http://tgi-pod-1:8080",
    "http://tgi-pod-2:8080",
    "http://tgi-pod-3:8080"
])

endpoint = balancer.round_robin()
response = httpx.post(f"{endpoint}/generate", json={"inputs": "..."})
```

### Traffic Splitting and Canary Deployments

**Canary deployment (10% new version, 90% stable):**
```yaml
# Stable deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: tgi
      version: stable
  template:
    metadata:
      labels:
        app: tgi
        version: stable
    spec:
      containers:
      - name: tgi
        image: ghcr.io/huggingface/text-generation-inference:3.3.5
---
# Canary deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tgi
      version: canary
  template:
    metadata:
      labels:
        app: tgi
        version: canary
    spec:
      containers:
      - name: tgi
        image: ghcr.io/huggingface/text-generation-inference:3.4.0  # New version
---
# Service selects both versions
apiVersion: v1
kind: Service
metadata:
  name: tgi-service
spec:
  selector:
    app: tgi  # Matches both stable and canary
  ports:
  - port: 80
```

**Result**: 10% of traffic goes to canary, 90% to stable (proportional to replica count)

---

## Section 5: Monitoring, Metrics, and Observability (~90 lines)

### Prometheus Metrics

TGI exposes **Prometheus metrics** on `/metrics` endpoint:

**Key metrics:**
```prometheus
# Request metrics
tgi_request_duration_seconds_bucket{method="POST", le="0.1"}
tgi_request_duration_seconds_sum{method="POST"}
tgi_request_duration_seconds_count{method="POST"}

# Batch metrics
tgi_batch_next_size  # Current batch size
tgi_batch_next_tokens  # Tokens in current batch

# Queue metrics
tgi_queue_size  # Requests waiting in queue

# Generation metrics
tgi_request_generated_tokens_sum  # Total tokens generated
tgi_request_input_tokens_sum  # Total input tokens processed

# GPU metrics (via DCGM exporter)
dcgm_gpu_utilization  # GPU utilization %
dcgm_fb_used  # GPU memory used (bytes)
dcgm_power_usage  # GPU power consumption (watts)
```

**Prometheus scrape config:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tgi'
    static_configs:
      - targets: ['tgi-service:80']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Grafana dashboard queries:**
```promql
# Request rate (requests per second)
rate(tgi_request_duration_seconds_count[1m])

# Average latency (P50)
histogram_quantile(0.5, rate(tgi_request_duration_seconds_bucket[5m]))

# P99 latency
histogram_quantile(0.99, rate(tgi_request_duration_seconds_bucket[5m]))

# Throughput (tokens per second)
rate(tgi_request_generated_tokens_sum[1m])

# GPU utilization
avg(dcgm_gpu_utilization)

# Queue depth
avg(tgi_queue_size)
```

### Distributed Tracing with OpenTelemetry

From [TGI Documentation - Distributed Tracing](https://huggingface.co/docs/text-generation-inference/en/index) (accessed 2025-11-16):

**Enable tracing:**
```bash
# Launch TGI with OTLP endpoint
text-generation-launcher \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --otlp-endpoint http://jaeger:4317 \
    --otlp-service-name tgi-llama
```

**Jaeger deployment (Kubernetes):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:latest
        ports:
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 16686  # Jaeger UI
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
```

**Trace spans captured:**
- `http.request`: Full HTTP request lifecycle
- `tokenization`: Input tokenization time
- `prefill`: Prefill phase (process input)
- `decode`: Decode phase (generate tokens)
- `detokenization`: Convert token IDs to text

**Python instrumentation:**
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Trace inference calls
with tracer.start_as_current_span("inference_call"):
    response = httpx.post("http://tgi:8080/generate", json={
        "inputs": "What is deep learning?"
    })
```

### Logging and Analytics

**HuggingFace Inference Endpoints Analytics:**
- Request rate over time
- Latency percentiles (P50, P95, P99)
- Error rate
- Token throughput
- Cost per request
- GPU utilization

**Custom logging (self-hosted):**
```python
import logging
import json
from datetime import datetime

logger = logging.getLogger("tgi_analytics")

class TGIAnalyticsMiddleware:
    """Log TGI inference metrics"""

    def log_inference(self, request, response, latency_ms):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_tokens": request.get("input_tokens", 0),
            "output_tokens": response.get("generated_tokens", 0),
            "latency_ms": latency_ms,
            "model": request.get("model"),
            "status": "success" if response else "error"
        }
        logger.info(json.dumps(log_entry))

# Use with requests
start = time.time()
response = httpx.post("http://tgi:8080/generate", json=payload)
latency = (time.time() - start) * 1000
middleware.log_inference(payload, response.json(), latency)
```

---

## Section 6: Autoscaling Strategies (~85 lines)

### HuggingFace Managed Autoscaling

From [HuggingFace Autoscaling Documentation](https://huggingface.co/docs/inference-endpoints/en/autoscaling) (accessed 2025-11-16):

**Managed autoscaling criteria:**
- **CPU accelerators**: Scale up when average CPU utilization > 80%
- **GPU accelerators**: Scale up when average GPU utilization > 80% (1-minute window)
- **Scaling frequency**: Scale up every 60 seconds, scale down every 120 seconds
- **Stabilization**: 300-second cooldown after scaling down

**Scale-to-zero:**
- Automatically scales to 0 replicas after 15 minutes of inactivity
- First request after scale-to-zero triggers cold start
- HTTP 502 returned during initialization
- Recommended: Implement client-side request queue with retries

**Pending requests-based scaling (beta):**
```python
# Enable via Endpoint settings
# Default threshold: 1.5 pending requests per replica (20-second window)
# Pending = in-flight + processing (no HTTP status yet)

# Example: 5 replicas, 10 pending requests
# 10 / 5 = 2.0 pending per replica > 1.5 threshold
# → Triggers autoscaling, adds 1 replica
```

### Self-Hosted Autoscaling (Kubernetes HPA)

**GPU utilization-based HPA:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tgi-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tgi-deployment
  minReplicas: 1
  maxReplicas: 20
  metrics:
  # GPU utilization (requires DCGM exporter + Prometheus adapter)
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "75"
  # Request rate (custom metric)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100  # Double replicas
        periodSeconds: 60
      - type: Pods
        value: 2  # Add max 2 pods at once
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1  # Remove 1 pod at a time
        periodSeconds: 120
      selectPolicy: Min
```

**Queue depth-based autoscaling:**
```yaml
# Scale based on TGI queue size
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tgi-queue-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tgi-deployment
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: tgi_queue_size
      target:
        type: AverageValue
        averageValue: "10"  # Scale if queue > 10 requests per pod
```

**Custom autoscaler (Python):**
```python
import httpx
import asyncio
from kubernetes import client, config

class TGIAutoscaler:
    """Custom autoscaler based on TGI metrics"""

    def __init__(self, deployment_name, namespace, min_replicas=1, max_replicas=10):
        config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    async def get_queue_depth(self, endpoint: str) -> int:
        """Get current queue depth from TGI metrics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{endpoint}/metrics")
            # Parse Prometheus metrics
            for line in response.text.split("\n"):
                if line.startswith("tgi_queue_size"):
                    return int(float(line.split()[1]))
        return 0

    async def scale(self, target_replicas: int):
        """Scale deployment to target replicas"""
        deployment = self.apps_v1.read_namespaced_deployment(
            self.deployment_name, self.namespace
        )
        deployment.spec.replicas = max(
            self.min_replicas,
            min(target_replicas, self.max_replicas)
        )
        self.apps_v1.patch_namespaced_deployment(
            self.deployment_name, self.namespace, deployment
        )

    async def autoscale_loop(self, endpoints: list):
        """Main autoscaling loop"""
        while True:
            total_queue = sum([
                await self.get_queue_depth(ep) for ep in endpoints
            ])
            current_replicas = len(endpoints)

            # Scale up if queue > 20 per replica
            if total_queue / current_replicas > 20:
                await self.scale(current_replicas + 2)
            # Scale down if queue < 5 per replica
            elif total_queue / current_replicas < 5:
                await self.scale(current_replicas - 1)

            await asyncio.sleep(30)  # Check every 30 seconds
```

---

## Section 7: A/B Testing and Canary Deployments (~80 lines)

### A/B Testing Model Versions

**Traffic splitting with Istio:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: tgi-ab-test
spec:
  hosts:
  - tgi.example.com
  http:
  - match:
    - headers:
        x-user-group:
          exact: "beta"
    route:
    - destination:
        host: tgi-service
        subset: v2  # New model
      weight: 100
  - route:
    - destination:
        host: tgi-service
        subset: v1  # Stable model
      weight: 80
    - destination:
        host: tgi-service
        subset: v2  # New model
      weight: 20
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: tgi-destination
spec:
  host: tgi-service
  subsets:
  - name: v1
    labels:
      version: stable
  - name: v2
    labels:
      version: experimental
```

**Application-level A/B testing:**
```python
import hashlib
from typing import Literal

class ABTester:
    """A/B test two TGI endpoints"""

    def __init__(self, endpoint_a: str, endpoint_b: str, split_ratio: float = 0.5):
        self.endpoint_a = endpoint_a
        self.endpoint_b = endpoint_b
        self.split_ratio = split_ratio  # % traffic to B

    def get_variant(self, user_id: str) -> Literal["A", "B"]:
        """Consistent user assignment to variant"""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "B" if (hash_value % 100) < (self.split_ratio * 100) else "A"

    def get_endpoint(self, user_id: str) -> str:
        """Get endpoint for user"""
        variant = self.get_variant(user_id)
        return self.endpoint_b if variant == "B" else self.endpoint_a

    async def compare_responses(self, user_id: str, prompt: str):
        """Get responses from both variants for comparison"""
        endpoint_a_response = await self._query(self.endpoint_a, prompt)
        endpoint_b_response = await self._query(self.endpoint_b, prompt)

        return {
            "user_id": user_id,
            "variant_a": endpoint_a_response,
            "variant_b": endpoint_b_response,
            "assigned_variant": self.get_variant(user_id)
        }

# Usage
ab_tester = ABTester(
    endpoint_a="http://tgi-llama-3-1:8080",
    endpoint_b="http://tgi-llama-3-2:8080",
    split_ratio=0.2  # 20% to B
)

user_endpoint = ab_tester.get_endpoint(user_id="user_12345")
response = httpx.post(f"{user_endpoint}/generate", json={"inputs": prompt})
```

### Blue-Green Deployment

**Instant traffic switch:**
```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-blue
spec:
  replicas: 5
  selector:
    matchLabels:
      app: tgi
      color: blue
---
# Green deployment (new version, ready)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-green
spec:
  replicas: 5
  selector:
    matchLabels:
      app: tgi
      color: green
---
# Service points to blue (current)
apiVersion: v1
kind: Service
metadata:
  name: tgi-service
spec:
  selector:
    app: tgi
    color: blue  # Change to "green" for instant switch
  ports:
  - port: 80
```

**Instant rollback:**
```bash
# Switch to green
kubectl patch service tgi-service -p '{"spec":{"selector":{"color":"green"}}}'

# Rollback to blue (if issues detected)
kubectl patch service tgi-service -p '{"spec":{"selector":{"color":"blue"}}}'
```

---

## Section 8: arr-coc-0-1 Production Deployment (Self-Hosted K8s) (~90 lines)

### Production Architecture for arr-coc-0-1

The arr-coc-0-1 MVP implements **Vervaekean relevance realization** with adaptive visual token allocation. Production deployment requires:

**Infrastructure requirements:**
- **GPU**: NVIDIA A100 (40GB+) for VLM processing
- **CPU**: 16+ cores for preprocessing
- **Memory**: 64GB+ RAM
- **Storage**: 500GB SSD for model cache

**Self-hosted Kubernetes deployment:**

```yaml
# arr-coc-0-1 production deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arr-coc-vlm
  namespace: production
spec:
  replicas: 3  # 3 replicas for high availability
  selector:
    matchLabels:
      app: arr-coc-vlm
  template:
    metadata:
      labels:
        app: arr-coc-vlm
    spec:
      containers:
      - name: vlm-server
        image: arr-coc-0-1:production-v0.1
        command: ["python", "-m", "uvicorn"]
        args:
          - "app.main:app"
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "8080"
          - "--workers"
          - "1"  # Single worker per GPU
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: MODEL_NAME
          value: "Qwen/Qwen3-VL-2B-Instruct"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "max_split_size_mb:512"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "16"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 30
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: arr-coc-model-cache
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
```

### Monitoring arr-coc-0-1 Metrics

**Custom Prometheus metrics for arr-coc-0-1:**
```python
# app/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUESTS_TOTAL = Counter(
    'arr_coc_requests_total',
    'Total VLM inference requests',
    ['status']
)

INFERENCE_LATENCY = Histogram(
    'arr_coc_inference_latency_seconds',
    'VLM inference latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Relevance realization metrics
TOKEN_ALLOCATION = Histogram(
    'arr_coc_token_allocation',
    'Tokens allocated per patch',
    ['patch_type'],  # high_relevance, medium_relevance, low_relevance
    buckets=[64, 128, 200, 256, 400]
)

PATCH_RELEVANCE = Histogram(
    'arr_coc_patch_relevance_score',
    'Relevance scores for patches',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

GPU_MEMORY_USED = Gauge(
    'arr_coc_gpu_memory_bytes',
    'GPU memory usage'
)

# Usage in inference
class ARRCOCInferenceServer:
    async def process_vlm_request(self, image, query):
        start = time.time()

        try:
            # Relevance realization
            relevance_scores = self.realize_relevance(image, query)
            for patch_idx, score in enumerate(relevance_scores):
                PATCH_RELEVANCE.observe(score)

            # Token allocation
            allocations = self.allocate_tokens(relevance_scores)
            for patch_type, tokens in allocations.items():
                TOKEN_ALLOCATION.labels(patch_type=patch_type).observe(tokens)

            # VLM inference
            response = await self.vlm_model.generate(image, query, allocations)

            # Record success
            REQUESTS_TOTAL.labels(status='success').inc()
            INFERENCE_LATENCY.observe(time.time() - start)

            # GPU memory
            import torch
            GPU_MEMORY_USED.set(torch.cuda.memory_allocated())

            return response

        except Exception as e:
            REQUESTS_TOTAL.labels(status='error').inc()
            raise
```

**Grafana dashboard queries for arr-coc-0-1:**
```promql
# Request rate
rate(arr_coc_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(arr_coc_inference_latency_seconds_bucket[5m]))

# Average token allocation per patch type
avg(arr_coc_token_allocation) by (patch_type)

# Relevance score distribution
histogram_quantile(0.5, rate(arr_coc_patch_relevance_score_bucket[5m]))

# GPU memory utilization
arr_coc_gpu_memory_bytes / (40 * 1024 * 1024 * 1024) * 100  # % of 40GB
```

### Cost Optimization for arr-coc-0-1

**Strategy 1: Autoscaling with scale-to-zero**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: arr-coc-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: arr-coc-vlm
  minReplicas: 0  # Scale to zero during low traffic
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 180
```

**Strategy 2: Spot instances for development**
```yaml
# Use preemptible/spot instances for dev/staging
nodeSelector:
  cloud.google.com/gke-spot: "true"
tolerations:
- key: cloud.google.com/gke-spot
  operator: Equal
  value: "true"
  effect: NoSchedule
```

**Strategy 3: Batch processing during off-peak**
```python
import asyncio
from datetime import datetime

class BatchScheduler:
    """Schedule VLM processing for off-peak hours"""

    async def schedule_batch(self, images, queries):
        current_hour = datetime.utcnow().hour

        # Off-peak: 00:00-06:00 UTC
        if 0 <= current_hour < 6:
            # Process immediately (cheaper compute)
            return await self.process_batch(images, queries)
        else:
            # Queue for off-peak processing
            await self.queue_for_later(images, queries)
```

**Cost comparison:**
- **Managed Inference Endpoints**: $4.50/hour (A100 40GB)
- **Self-hosted GKE**: $2.50/hour (A100 spot instance)
- **With autoscaling**: ~$500/month (8hrs/day usage)
- **Without autoscaling**: ~$3,240/month (24/7 uptime)

**Savings**: 85% cost reduction with intelligent autoscaling

---

## Sources

**HuggingFace Documentation:**
- [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/en/index) - Managed hosting documentation
- [Text Generation Inference GitHub](https://github.com/huggingface/text-generation-inference) - TGI server repository
- [Autoscaling Documentation](https://huggingface.co/docs/inference-endpoints/en/autoscaling) - Scaling strategies

**Web Research:**
- [Kubernetes GPU Deployment Guide](https://xebia.com/blog/deploy-open-source-llm-in-your-private-cluster-with-hugging-face-and-gke-autopilot/) (accessed 2025-11-16)
- [TGI Deployment on Kubernetes](https://huggingface.co/blog/voatsap/tgi-kubernetes-cluster-dev) (accessed 2025-11-16)

**Related Knowledge Files:**
- [../vertex-ai-production/00-distributed-training-patterns.md](../vertex-ai-production/00-distributed-training-patterns.md) - Production training patterns
- [06-inference-optimization-pipeline.md](06-inference-optimization-pipeline.md) - Inference acceleration
- [07-spaces-gradio-streamlit.md](07-spaces-gradio-streamlit.md) - Demo deployment
