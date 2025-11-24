# GKE GPU Inference Serving

## Overview

GPU-accelerated inference serving on Google Kubernetes Engine (GKE) enables deploying ML models at scale with optimal performance. This guide covers three major inference serving platforms: Triton Inference Server, KServe, and TorchServe, all optimized for GPU acceleration on GKE.

## Triton Inference Server on GKE

### Architecture

NVIDIA Triton Inference Server provides optimized serving for multiple frameworks with GPU acceleration.

**Key Features:**
- Multi-framework support (TensorFlow, PyTorch, ONNX, TensorRT)
- Dynamic batching for throughput optimization
- Model ensemble pipelines
- GPU memory sharing across models
- HTTP/REST and gRPC inference APIs

From [Google Cloud GKE Gemma Tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-tensortllm):
- Triton serves Gemma LLMs using TensorRT-LLM backend on GKE
- Supports deployment on L4, T4, A100 GPUs
- Integrates with GKE Autopilot for managed infrastructure

### Deployment Configuration

**Basic Triton Deployment YAML:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton-server
  template:
    metadata:
      labels:
        app: triton-server
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
        - containerPort: 8002
          name: metrics
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-repository
          mountPath: /models
      volumes:
      - name: model-repository
        persistentVolumeClaim:
          claimName: model-storage
```

**Model Repository Structure:**
```
/models/
├── model_name/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan (TensorRT)
│       └── model.onnx (ONNX)
│       └── model.pt (PyTorch)
```

### Dynamic Batching

From [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html):

**Configuration in config.pbtxt:**
```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 100
}
```

**Benefits:**
- Automatically combines individual requests into batches
- Maximizes GPU utilization
- Reduces inference latency for high-throughput scenarios
- Configurable batch size and queue delay

### Multi-Model Serving

**Concurrent Model Loading:**
```yaml
# Triton can serve multiple models simultaneously
# sharing GPU memory efficiently
args:
  - tritonserver
  - --model-repository=/models
  - --model-control-mode=explicit
  - --load-model=bert-base
  - --load-model=resnet50
  - --load-model=gpt2-small
```

**GPU Memory Sharing:**
- Models share GPU DRAM
- Dynamic model loading/unloading
- Priority-based scheduling

## KServe on GKE

### Architecture Overview

From [GKE AI Labs KServe Tutorial](https://gke-ai-labs.dev/docs/tutorials/inference-servers/kserve/):

KServe provides serverless inference serving on Kubernetes with:
- Autoscaling (scale-to-zero capability)
- Canary deployments and traffic splitting
- Standard inference protocol (V2)
- GPU resource management
- Integration with Knative and Istio

### Prerequisites Installation

**1. Install Knative Serving:**
```bash
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.15.1/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.15.1/serving-core.yaml
```

**2. Install Istio (Networking Layer):**
```bash
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update
kubectl create namespace istio-system
helm install istio-base istio/base -n istio-system --set defaultRevision=default
helm install istiod istio/istiod -n istio-system --wait
helm install istio-ingressgateway istio/gateway -n istio-system
```

**3. Install Knative-Istio Integration:**
```bash
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.15.1/net-istio.yaml
```

**4. Install Cert Manager:**
```bash
helm repo add jetstack https://charts.jetstack.io --force-update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.15.3 \
  --set crds.enabled=true
```

**5. Install KServe:**
```bash
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.14.0-rc0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.14.0-rc0/kserve-cluster-resources.yaml
```

**6. Enable GPU Node Selector:**
```bash
kubectl patch configmap/config-features \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"kubernetes.podspec-nodeselector":"enabled", "kubernetes.podspec-tolerations":"enabled"}}'
```

### InferenceService Custom Resource

From [The New Stack KServe Tutorial](https://thenewstack.io/serve-tensorflow-models-with-kserve-on-google-kubernetes-engine/):

**TensorFlow Model Serving:**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: tensorflow-model
  namespace: default
spec:
  predictor:
    tensorflow:
      storageUri: "gs://bucket-name/model-path"
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
    nodeSelector:
      cloud.google.com/gke-accelerator: nvidia-t4
```

**vLLM Backend for LLMs:**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma2-vllm
  namespace: kserve-test
spec:
  predictor:
    nodeSelector:
      cloud.google.com/gke-accelerator: nvidia-l4
      cloud.google.com/gke-accelerator-count: "1"
    model:
      modelFormat:
        name: huggingface
      args:
        - --enable_docs_url=True
        - --model_name=gemma2
        - --model_id=google/gemma-2-2b
      env:
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef:
            name: hf-secret
            key: hf_api_token
      resources:
        limits:
          cpu: "6"
          memory: 24Gi
          nvidia.com/gpu: "1"
        requests:
          cpu: "6"
          memory: 24Gi
          nvidia.com/gpu: "1"
```

### Autoscaling Configuration

**Scale-to-Zero:**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: autoscale-model
  annotations:
    autoscaling.knative.dev/target: "10"
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "5"
spec:
  predictor:
    tensorflow:
      storageUri: "gs://models/resnet50"
      resources:
        limits:
          nvidia.com/gpu: 1
```

**Concurrency-Based Scaling:**
- `target`: Requests per pod before scaling
- `minScale: "0"`: Enable scale-to-zero
- `maxScale`: Maximum replica count

### Canary Deployments

**Traffic Splitting:**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: canary-model
spec:
  predictor:
    tensorflow:
      storageUri: "gs://models/v1"
  canaryTrafficPercent: 20
```

**Progressive Rollout:**
1. Deploy canary with 10% traffic
2. Monitor metrics (latency, accuracy)
3. Increase to 50% if successful
4. Complete rollout to 100%

## TorchServe on GKE

### Architecture

From [Google Cloud TorchServe Tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/scalable-ml-models-torchserve):

TorchServe provides production-ready PyTorch model serving with:
- Native PyTorch model support
- Custom inference handlers
- Model versioning
- Metrics and logging integration
- GPU acceleration

### Deployment Configuration

**TorchServe Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve
spec:
  replicas: 1
  selector:
    matchLabels:
      app: torchserve
  template:
    metadata:
      labels:
        app: torchserve
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-t4
      containers:
      - name: torchserve
        image: pytorch/torchserve:latest-gpu
        ports:
        - containerPort: 8080
          name: inference
        - containerPort: 8081
          name: management
        - containerPort: 8082
          name: metrics
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: 4
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: 4
        volumeMounts:
        - name: model-store
          mountPath: /home/model-server/model-store
        env:
        - name: TS_NUMBER_OF_GPU
          value: "1"
        - name: TS_INSTALL_PY_DEP_PER_MODEL
          value: "true"
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: torchserve-models
```

**Config.properties:**
```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=100
default_workers_per_model=1
number_of_gpu=1
```

### Model Archive (.mar) Creation

**Creating Model Archive:**
```bash
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet50.pth \
  --handler image_classifier \
  --extra-files index_to_name.json \
  --export-path model-store/
```

**Registering Model:**
```bash
curl -X POST "http://torchserve:8081/models?url=resnet50.mar&batch_size=8&max_batch_delay=100"
```

### Autoscaling with HPA

From [Google Cloud Tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/scalable-ml-models-torchserve):

**Horizontal Pod Autoscaler:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: torchserve-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: torchserve
  minReplicas: 1
  maxReplicas: 5
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
        name: torchserve_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

**Custom Metrics:**
- Requests per second
- Queue depth
- Inference latency
- GPU utilization

## Multi-Model GPU Sharing

### GPU Time-Sharing

**NVIDIA Time-Slicing Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  tesla-t4: |-
    version: v1
    sharing:
      timeSlicing:
        replicas: 4
```

**Benefits:**
- Multiple pods share single GPU
- Cost optimization for low-utilization workloads
- Suitable for development/testing environments

**Limitations:**
- No memory isolation
- Best-effort scheduling
- Not recommended for production inference

### MIG (Multi-Instance GPU)

**A100 MIG Configuration:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mig-pod
spec:
  containers:
  - name: inference
    image: pytorch/torchserve:latest-gpu
    resources:
      limits:
        nvidia.com/mig-1g.5gb: 1
```

**MIG Profiles:**
- 1g.5gb: 1 GPU slice, 5GB memory
- 2g.10gb: 2 GPU slices, 10GB memory
- 3g.20gb: 3 GPU slices, 20GB memory
- 7g.40gb: Full GPU, 40GB memory

## Load Balancing Strategies

### L7 Load Balancing with Istio

**VirtualService Configuration:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: inference-vs
spec:
  hosts:
  - inference.example.com
  gateways:
  - inference-gateway
  http:
  - match:
    - uri:
        prefix: /v1/models
    route:
    - destination:
        host: triton-service
        port:
          number: 8000
      weight: 80
    - destination:
        host: triton-service-canary
        port:
          number: 8000
      weight: 20
```

### Internal Load Balancer

**Service Configuration:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-ilb
  annotations:
    cloud.google.com/load-balancer-type: "Internal"
spec:
  type: LoadBalancer
  selector:
    app: triton-server
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
```

## Monitoring and Observability

### Prometheus Metrics

**ServiceMonitor for Triton:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: triton-metrics
spec:
  selector:
    matchLabels:
      app: triton-server
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

**Key Metrics:**
- `nv_inference_request_success`: Successful requests
- `nv_inference_request_duration_us`: Latency
- `nv_inference_queue_duration_us`: Queue time
- `nv_gpu_utilization`: GPU usage
- `nv_gpu_memory_total_bytes`: GPU memory

### Cloud Monitoring Integration

**GPU Metrics in Cloud Monitoring:**
```bash
# Enable DCGM metrics
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/cmd/nvidia_gpu/dcgm-exporter.yaml
```

**Monitored Metrics:**
- GPU utilization (%)
- GPU memory used (bytes)
- GPU temperature (C)
- Power usage (watts)
- SM clock frequency (MHz)

### Logging Best Practices

From [NVIDIA Best Practices](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html):

**Structured Logging:**
```yaml
env:
- name: TRITON_LOG_VERBOSE
  value: "1"
- name: TRITON_LOG_INFO
  value: "1"
```

**Log Aggregation:**
- Send logs to Cloud Logging
- Parse inference latency from logs
- Alert on error rates

## Performance Optimization

### Batch Size Tuning

**Finding Optimal Batch Size:**
```python
# Test different batch sizes
for batch_size in [1, 4, 8, 16, 32]:
    latency = benchmark_inference(batch_size)
    throughput = batch_size / latency
    print(f"Batch {batch_size}: {throughput} req/sec")
```

**Trade-offs:**
- Larger batches: Higher throughput, higher latency
- Smaller batches: Lower latency, lower throughput
- Dynamic batching: Best of both worlds

### Model Optimization

**TensorRT Optimization:**
```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --workspace=4096
```

**Quantization:**
- FP16: 2x speedup, minimal accuracy loss
- INT8: 4x speedup, requires calibration
- Mixed precision: Balance speed and accuracy

### Network Optimization

**Pod Network Settings:**
```yaml
spec:
  containers:
  - name: triton
    securityContext:
      capabilities:
        add:
        - NET_ADMIN
    env:
    - name: NCCL_SOCKET_IFNAME
      value: eth0
    - name: NCCL_IB_DISABLE
      value: "1"
```

## Cost Optimization

### Spot Instances for Inference

**Node Pool Configuration:**
```bash
gcloud container node-pools create spot-gpu-pool \
  --cluster=inference-cluster \
  --spot \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5
```

**Spot Tolerations:**
```yaml
tolerations:
- key: cloud.google.com/gke-spot
  operator: Equal
  value: "true"
  effect: NoSchedule
```

**Savings:**
- 60-91% cost reduction vs on-demand
- Suitable for fault-tolerant workloads
- Combine with autoscaling for cost efficiency

### Right-Sizing GPU Selection

**GPU Selection Matrix:**
- **T4**: Cost-effective, inference-optimized (FP16, INT8)
- **L4**: Newer, better price/performance for LLMs
- **A100**: High-throughput, large batch sizes
- **H100**: Cutting-edge, FP8 support for largest models

**Cost per GPU Hour (approximate):**
- T4: $0.35/hour
- L4: $0.60/hour
- A100 40GB: $3.67/hour
- A100 80GB: $4.68/hour

## Production Deployment Patterns

### Blue-Green Deployment

**Service Switching:**
```yaml
# Blue deployment
apiVersion: v1
kind: Service
metadata:
  name: inference-prod
spec:
  selector:
    app: triton
    version: blue
  ports:
  - port: 8000

# Switch to green
kubectl patch service inference-prod -p '{"spec":{"selector":{"version":"green"}}}'
```

### Canary with Metrics-Based Promotion

**Progressive Rollout:**
1. Deploy canary with 5% traffic
2. Monitor error rate and latency for 10 minutes
3. If metrics acceptable, increase to 25%
4. Continue increasing until 100%
5. Automatic rollback if metrics degrade

**Flagger Integration:**
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: inference-canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-server
  service:
    port: 8000
  analysis:
    interval: 1m
    threshold: 10
    maxWeight: 50
    stepWeight: 5
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 500
```

### Health Checks and Readiness

**Readiness Probe:**
```yaml
readinessProbe:
  httpGet:
    path: /v2/health/ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3
```

**Liveness Probe:**
```yaml
livenessProbe:
  httpGet:
    path: /v2/health/live
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
  failureThreshold: 5
```

## Security Considerations

### Workload Identity

**Service Account Binding:**
```bash
# Create GCP service account
gcloud iam service-accounts create triton-sa

# Grant GCS access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member "serviceAccount:triton-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/storage.objectViewer"

# Bind to K8s service account
kubectl annotate serviceaccount triton-sa \
  iam.gke.io/gcp-service-account=triton-sa@PROJECT_ID.iam.gserviceaccount.com
```

### Network Policies

**Restrict Inference Traffic:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: inference-network-policy
spec:
  podSelector:
    matchLabels:
      app: triton-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: application-namespace
    ports:
    - protocol: TCP
      port: 8000
```

### Secret Management

**Model Access Credentials:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-token
type: Opaque
stringData:
  token: <huggingface-token>
---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: inference
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
```

## arr-coc-0-1 Integration

### Inference Service for arr-coc

**KServe Deployment:**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: arr-coc-inference
  namespace: arr-coc-prod
spec:
  predictor:
    pytorch:
      storageUri: "gs://arr-coc-models/latest"
      resources:
        limits:
          nvidia.com/gpu: 1
          memory: 16Gi
        requests:
          nvidia.com/gpu: 1
          memory: 16Gi
    nodeSelector:
      cloud.google.com/gke-accelerator: nvidia-l4
      cloud.google.com/gke-accelerator-count: "1"
    env:
    - name: PYTORCH_CUDA_ALLOC_CONF
      value: "max_split_size_mb:512"
```

**Custom Handler for ARR-COC:**
```python
# handler.py for arr-coc model
import torch
from ts.torch_handler.base_handler import BaseHandler

class ARRCOCHandler(BaseHandler):
    def preprocess(self, data):
        # Extract image and query
        images = [row["image"] for row in data]
        queries = [row["query"] for row in data]
        return images, queries

    def inference(self, images, queries):
        # Run arr-coc inference
        with torch.no_grad():
            outputs = self.model(images, queries)
        return outputs

    def postprocess(self, inference_output):
        # Format response
        return [{
            "compressed_features": out.tolist(),
            "token_allocation": out.shape
        } for out in inference_output]
```

## Troubleshooting

### Common Issues

**1. OOM (Out of Memory):**
```bash
# Check GPU memory
kubectl exec -it pod-name -- nvidia-smi

# Reduce batch size or model precision
# Enable gradient checkpointing for training
```

**2. Slow Cold Start:**
```bash
# Pre-pull images
kubectl create -f image-puller-daemonset.yaml

# Increase minReplicas to avoid cold starts
# Use model caching on node local SSD
```

**3. Low GPU Utilization:**
```bash
# Check batch size and queue depth
# Verify dynamic batching is enabled
# Monitor with DCGM metrics
```

## Sources

**Source Documents:**
- [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md) (reference - not yet created)
- [inference-optimization/00-tensorrt-fundamentals.md](../inference-optimization/00-tensorrt-fundamentals.md) (reference - not yet created)
- [orchestration/00-kubernetes-gpu-scheduling.md](../orchestration/00-kubernetes-gpu-scheduling.md) (reference - not yet created)

**Web Research:**
- [Google Cloud: Serve Gemma on GKE with TensorRT-LLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-tensortllm) (accessed 2025-11-16)
- [GKE AI Labs: KServe Tutorial](https://gke-ai-labs.dev/docs/tutorials/inference-servers/kserve/) (accessed 2025-11-16)
- [The New Stack: Serve TensorFlow Models with KServe on GKE](https://thenewstack.io/serve-tensorflow-models-with-kserve-on-google-kubernetes-engine/) (accessed 2025-11-16)
- [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (accessed 2025-11-16)
- [Google Cloud: Scalable ML Models with TorchServe](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/scalable-ml-models-torchserve) (accessed 2025-11-16)

**Additional References:**
- [Kubeflow KServe Introduction](https://www.kubeflow.org/docs/components/kserve/introduction/)
- [PyTorch TorchServe Documentation](https://docs.pytorch.org/serve/)
