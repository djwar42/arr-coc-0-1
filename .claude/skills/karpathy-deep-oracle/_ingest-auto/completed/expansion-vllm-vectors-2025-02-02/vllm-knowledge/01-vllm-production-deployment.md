# vLLM Production Deployment Guide

## Overview

vLLM production deployment requires careful configuration of containerization, orchestration, multi-GPU scaling, and observability. This guide covers Docker-based deployments, Kubernetes orchestration, distributed serving strategies, and production monitoring patterns for enterprise-scale LLM inference.

From [vLLM Production Stack](https://docs.vllm.ai/en/stable/deployment/integrations/production-stack.html) (accessed 2025-02-02):
- Official vLLM production reference architecture
- Docker and Kubernetes deployment patterns
- Multi-GPU tensor parallelism configuration

From [Effortless vLLM Deployment with Docker](https://medium.com/@kimdoil1211/effortless-vllm-deployment-with-docker-a-comprehensive-guide-2a23119839e2) by Doil Kim (accessed 2025-02-02):
- Single-script Docker deployment automation
- GPU passthrough configuration
- Environment variable management

---

## Section 1: Docker Deployment

### Basic Docker Setup

**Official vLLM Docker Image:**

```bash
# Pull official image
docker pull vllm/vllm-openai:latest

# Run with single GPU
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model facebook/opt-125m
```

From [vLLM Docker Documentation](https://docs.vllm.ai/en/stable/deployment/docker.html) (accessed 2025-02-02):
- Official Docker images available on Docker Hub
- Automatic GPU detection with NVIDIA runtime
- HuggingFace cache mounting for efficient model loading

**Key Docker Configuration Elements:**

1. **NVIDIA Runtime:**
   - `--runtime nvidia` enables GPU access
   - `--gpus all` exposes all available GPUs
   - `--gpus "device=0,1"` for specific GPU selection

2. **Volume Mounts:**
   - HuggingFace cache: `~/.cache/huggingface:/root/.cache/huggingface`
   - Local models: `-v /path/to/model:/models`
   - Shared memory: `--ipc=host` (critical for multi-GPU)

3. **Port Mapping:**
   - API endpoint: `-p 8000:8000`
   - Metrics endpoint: `-p 8001:8001` (if enabled)

### Production Docker Deployment Script

From [Doil Kim's Docker Guide](https://medium.com/@kimdoil1211/effortless-vllm-deployment-with-docker-a-comprehensive-guide-2a23119839e2):

**Automated Deployment Script (`run_vllm_docker.sh`):**

```bash
#!/bin/bash
# Production vLLM Docker deployment with automatic configuration

# Load environment variables
source .env

# Configuration
GPU_ID=0                          # GPU selection (0 or 0,1,2,3)
PORT=8000                         # API endpoint port
MAX_MODEL_LEN=131072             # Maximum context length
MAX_NUM_SEQS=256                 # Maximum batch size
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
SERVED_MODEL_NAME="llama-3.2-1b-instruct"
GPU_MEMORY_UTILIZATION=0.9       # GPU memory fraction (0.0-1.0)
LOG_FILE="/var/log/vllm/vllm_docker.log"
DTYPE=half                       # Precision: half, bfloat16, float

# Calculate tensor parallel size from GPU count
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_ID"
TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}

# Run vLLM container
docker run --runtime nvidia --gpus "device=$GPU_ID" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    -p $PORT:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $MODEL_PATH \
    --trust-remote-code \
    --host 0.0.0.0 \
    --served-model-name $SERVED_MODEL_NAME \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 \
    --dtype $DTYPE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    2>&1 | tee "$LOG_FILE"
```

**Environment Variables (`.env`):**

```bash
# Security: Store sensitive tokens in .env file
HUGGING_FACE_HUB_TOKEN=your_token_here
```

**Critical Configuration Parameters:**

- `--max-model-len`: Context window size (triggers chunked prefill if >32k)
- `--max-num-seqs`: Concurrent sequences (batch size)
- `--gpu-memory-utilization`: Memory fraction (0.9 = 90% of VRAM)
- `--tensor-parallel-size`: Automatically calculated from GPU count
- `--swap-space 0`: Disable CPU swapping for single-answer generation
- `--ipc=host`: Required for multi-GPU inter-process communication

### Docker Compose for Multi-Service Stacks

**Production Docker Compose Example:**

```yaml
version: '3.8'

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - MODEL_NAME=facebook/opt-125m
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    command: >
      vllm serve facebook/opt-125m
      --host 0.0.0.0
      --port 8000
      --max-model-len 4096
      --gpu-memory-utilization 0.9
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Section 2: Kubernetes Deployment

### Kubernetes Manifests for vLLM

From [vLLM Kubernetes Deployment Guide](https://docs.vllm.ai/en/stable/serving/deploying_with_k8s.html) (accessed 2025-02-02):

**Deployment Manifest:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
  namespace: vllm
spec:
  replicas: 2  # Horizontal scaling
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          requests:
            nvidia.com/gpu: 1  # GPU requirement
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        env:
        - name: MODEL_NAME
          value: "facebook/opt-125m"
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: metrics
        volumeMounts:
        - name: cache
          mountPath: /root/.cache/huggingface
        - name: shm
          mountPath: /dev/shm
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: vllm-cache-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

**Service Manifest:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: vllm
spec:
  selector:
    app: vllm-server
  type: LoadBalancer
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 8001
    targetPort: 8001
    protocol: TCP
```

**PersistentVolumeClaim for HuggingFace Cache:**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-cache-pvc
  namespace: vllm
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nfs-storage
  resources:
    requests:
      storage: 100Gi
```

### GPU Scheduling and Node Selection

**Node Affinity for GPU Nodes:**

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - Tesla-T4
          - Tesla-V100
          - NVIDIA-A100-SXM4-40GB
```

**GPU Resource Limits:**

```yaml
resources:
  limits:
    nvidia.com/gpu: 2  # Multi-GPU per pod
  requests:
    nvidia.com/gpu: 2
```

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: vllm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
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
        name: vllm_request_queue_depth
      target:
        type: AverageValue
        averageValue: "50"
```

---

## Section 3: Multi-GPU and Tensor Parallelism

From [vLLM Distributed Serving Documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) (accessed 2025-02-02):

### Tensor Parallelism Configuration

**Single-Node Multi-GPU:**

```bash
# 4 GPUs with tensor parallelism
vllm serve meta-llama/Llama-3-70B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192
```

**Tensor Parallelism calculates automatically:**
- Model sharded across GPUs
- Each GPU holds 1/N of model weights
- Communication via NCCL (NVIDIA Collective Communications Library)

**Multi-GPU Docker:**

```bash
docker run --runtime nvidia --gpus '"device=0,1,2,3"' \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3-70B \
    --tensor-parallel-size 4
```

### Pipeline Parallelism

From [vLLM Parallelism Documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html):

**Pipeline Parallel for Multi-Node:**

```bash
# 2 nodes, 4 GPUs each = 8 GPUs total
vllm serve meta-llama/Llama-3-70B \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

**Parallelism Strategy Selection:**

| Model Size | GPUs | Strategy |
|------------|------|----------|
| 7B-13B | 1 | No parallelism |
| 13B-30B | 1-2 | Single GPU or TP=2 |
| 30B-70B | 2-4 | Tensor parallel |
| 70B+ | 4-8 | TP + pipeline parallel |

### Performance Optimization

**GPU Memory Utilization:**
- Default: 0.9 (90% of VRAM)
- High throughput: 0.95
- Safety margin: 0.85
- Multi-tenant: 0.7-0.8

**Max Model Length:**
- Triggers chunked prefill if >32k tokens
- Balance context vs throughput
- Large contexts = lower batch size

**Batch Size (`max-num-seqs`):**
- Higher = better throughput
- Lower = better latency
- Start with 256, tune based on workload

---

## Section 4: Monitoring and Observability

From [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) by Himadri Talukder (accessed 2025-02-02):

### Prometheus Metrics Integration

**Prometheus Scrape Configuration:**

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'vllm-server'
    static_configs:
      - targets: ['vllm-server:8000']
    metrics_path: /metrics
```

**Key vLLM Metrics:**

From [vLLM Metrics Documentation](https://docs.vllm.ai/en/latest/design/metrics.html):

1. **Request Latency:**
   - `vllm:request_success_total`
   - `vllm:time_to_first_token_seconds`
   - `vllm:time_per_output_token_seconds`

2. **Throughput:**
   - `vllm:request_prompt_tokens_total`
   - `vllm:request_generation_tokens_total`
   - `vllm:num_requests_running`

3. **GPU Utilization:**
   - `vllm:gpu_cache_usage_perc`
   - `vllm:gpu_kv_cache_usage_perc`
   - `vllm:num_preemptions_total`

4. **Queue Depth:**
   - `vllm:num_requests_waiting`
   - `vllm:request_queue_time_seconds`

### Grafana Dashboard Configuration

**Datasource Provisioning:**

```yaml
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

**Dashboard Provisioning:**

```yaml
# grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1
providers:
  - name: 'vLLM Dashboards'
    folder: 'vLLM'
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

From [vLLM Grafana Dashboard](https://grafana.com/grafana/dashboards/23991-vllm/) (accessed 2025-02-02):
- Official vLLM dashboard template
- Pre-configured panels for key metrics
- Token throughput, latency, cache utilization

### OpenTelemetry Tracing

From [Building Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949):

**Enable OpenTelemetry Traces:**

```bash
vllm serve facebook/opt-125m \
    --otlp-traces-endpoint=grpc://jaeger:4317
```

**Docker Compose with Observability:**

```yaml
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    environment:
      - OTEL_SERVICE_NAME=vllm-server
      - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://jaeger:4317
      - OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    command: >
      vllm serve facebook/opt-125m
      --otlp-traces-endpoint=grpc://jaeger:4317

  jaeger:
    image: jaegertracing/all-in-one:1.57
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
```

---

## Section 5: Production Best Practices

### Health Checks and Readiness

**Health Check Endpoint:**

```bash
# Kubernetes livenessProbe
curl http://vllm-server:8000/health
```

**Response:**
```json
{"status": "ok"}
```

**Readiness Check:**
- Wait 60-120 seconds for model loading
- Check `/health` returns 200
- Verify `/v1/models` lists served model

### Load Balancing

**Kubernetes Service Load Balancing:**
- Default: Round-robin across pods
- Session affinity: Not recommended (stateless inference)
- Use external load balancer (NGINX, HAProxy, Envoy)

**NGINX Configuration:**

```nginx
upstream vllm_backend {
    least_conn;
    server vllm-pod-1:8000 max_fails=3 fail_timeout=30s;
    server vllm-pod-2:8000 max_fails=3 fail_timeout=30s;
    server vllm-pod-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

### Rolling Updates

**Kubernetes Rolling Update Strategy:**

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero downtime
  minReadySeconds: 30
```

**Update Process:**
1. New pod starts
2. Wait for readiness probe (60-120s)
3. Add to service endpoints
4. Terminate old pod after draining

### Security Best Practices

**1. Token Management:**
```bash
# Kubernetes Secret for HuggingFace token
kubectl create secret generic hf-token \
    --from-literal=token=YOUR_TOKEN \
    -n vllm
```

**2. Network Policies:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-network-policy
spec:
  podSelector:
    matchLabels:
      app: vllm-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8000
```

**3. Resource Limits:**
- Always set memory limits (prevent OOM)
- Set CPU requests for scheduling
- GPU limits prevent over-allocation

### Debugging and Troubleshooting

**Log Verbosity:**

```bash
# Enable debug logging
vllm serve model --vllm-logging-level DEBUG
```

**Common Issues:**

1. **OOM (Out of Memory):**
   - Reduce `--gpu-memory-utilization`
   - Decrease `--max-model-len`
   - Lower `--max-num-seqs`

2. **Slow Startup:**
   - Use persistent HuggingFace cache
   - Pre-download models to volume
   - Increase readiness probe timeout

3. **Low Throughput:**
   - Increase `--max-num-seqs`
   - Enable chunked prefill for long contexts
   - Check GPU utilization in metrics

4. **High Latency:**
   - Reduce batch size (`--max-num-seqs`)
   - Check network latency to storage
   - Monitor queue depth metrics

---

## Sources

**vLLM Official Documentation:**
- [vLLM Production Stack](https://docs.vllm.ai/en/stable/deployment/integrations/production-stack.html)
- [vLLM Docker Deployment](https://docs.vllm.ai/en/stable/deployment/docker.html)
- [vLLM Kubernetes Guide](https://docs.vllm.ai/en/stable/serving/deploying_with_k8s.html)
- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html)
- [vLLM Metrics Documentation](https://docs.vllm.ai/en/latest/design/metrics.html)

**Web Research:**
- [Effortless vLLM Deployment with Docker](https://medium.com/@kimdoil1211/effortless-vllm-deployment-with-docker-a-comprehensive-guide-2a23119839e2) by Doil Kim (accessed 2025-02-02)
- [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) by Himadri Talukder, IBM (accessed 2025-02-02)
- [vLLM Grafana Dashboard](https://grafana.com/grafana/dashboards/23991-vllm/) (accessed 2025-02-02)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) (accessed 2025-02-02)

**Additional References:**
- [Monitoring vLLM Inference Servers Guide](https://www.dataunboxed.io/blog/monitoring-vllm-inference-servers-a-quick-and-easy-guide) by ERRAJI BADR (accessed 2025-02-02)
- vLLM GitHub Repository discussions on production deployment
- Kubernetes GPU scheduling documentation
