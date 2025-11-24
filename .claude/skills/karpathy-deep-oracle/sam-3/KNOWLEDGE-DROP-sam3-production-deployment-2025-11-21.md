# SAM 3 Production Deployment Strategies

## Overview

Deploying SAM 3 (Segment Anything Model 3) in production requires careful consideration of containerization, API serving, GPU optimization, and scaling strategies. While SAM 3 was released November 2025, deployment patterns from SAM and SAM 2 provide established best practices that apply directly to SAM 3's architecture.

## Docker Containerization

### Basic Dockerfile Structure

From [Deploying SAM with FastAPI and Docker](https://medium.com/@faical.ammisaid_48405/deploying-segment-anything-model-with-fastapi-and-docker-6e9ae439c7a8) (accessed 2025-11-23):

```dockerfile
# GPU-enabled base image
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    ffmpeg \
    git \
    wget

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and application code
COPY . /app
WORKDIR /app

# Expose API port
EXPOSE 8000

# Start server
CMD ["python", "api.py"]
```

### CPU vs GPU Containers

From [segment-anything-services](https://github.com/developmentseed/segment-anything-services) (accessed 2025-11-23):

**GPU Container (Image Encoder)**:
- Used for compute-intensive image embedding
- Inference time: ~1.8 seconds on modern GPUs
- Requires CUDA toolkit and cuDNN

**CPU Container (Mask Decoder)**:
- Used for lightweight mask prediction from embeddings
- Can handle interactive prompting without GPU
- Enables cost-effective scaling

```bash
# Build GPU container for encoding
docker build -t sam3-gpu -f Dockerfile-gpu .

# Build CPU container for decoding
docker build -t sam3-cpu -f Dockerfile-cpu .
```

### Multi-Stage Builds

For production optimization:

```dockerfile
# Build stage - compile dependencies
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel AS builder
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Runtime stage - minimal footprint
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
```

## API Serving Options

### FastAPI (Recommended for Simplicity)

From [MLOps SAM Task](https://github.com/AMMISAIDFaical/mlops_sam_task) (accessed 2025-11-23):

```python
from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse
from sam3 import Sam3Processor
import io

app = FastAPI()

# Initialize model at startup
processor = None

@app.on_event("startup")
async def load_model():
    global processor
    processor = Sam3Processor.from_pretrained("facebook/sam3-hiera-large")

@app.post("/segment-image")
async def segment_image(
    file: bytes = File(...),
    text_prompt: str = None
):
    """Segment image with optional text prompt."""
    image = decode_image(file)

    if text_prompt:
        masks = processor.segment_with_text(image, text_prompt)
    else:
        masks = processor.segment_everything(image)

    result = encode_masks(masks)
    return StreamingResponse(
        content=io.BytesIO(result),
        media_type="image/png"
    )
```

### TorchServe (Production-Grade)

From [Deploying SAM with TorchServe](https://github.com/facebookresearch/segment-anything/issues/549) (accessed 2025-11-23):

**Key Architecture Decision**: Separate encoder and decoder into different services.

```python
# Handler for image encoding (GPU)
class SAM3EncoderHandler(BaseHandler):
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        self.model = build_sam3_image_model(
            checkpoint=f"{model_dir}/sam3_hiera_large.pt"
        )
        self.model.eval()

    def preprocess(self, data):
        image_bytes = data[0].get("body")
        image = Image.open(io.BytesIO(image_bytes))
        return preprocess_image(image)

    def inference(self, image_tensor):
        with torch.no_grad():
            embedding = self.model.image_encoder(image_tensor)
        return embedding

    def postprocess(self, embedding):
        # Store in Redis with UUID key
        embedding_id = str(uuid.uuid4())
        self.cache.set(embedding_id, embedding.cpu().numpy().tobytes())
        return [{"embedding_id": embedding_id}]

# Handler for mask prediction (CPU/GPU)
class SAM3DecoderHandler(BaseHandler):
    def inference(self, data):
        embedding_id = data["embedding_id"]
        prompts = data["prompts"]

        # Retrieve cached embedding
        embedding = self.cache.get(embedding_id)

        # Generate masks from prompts
        masks = self.model.predict_masks(embedding, prompts)
        return masks
```

**Package models as .mar archives**:
```bash
torch-model-archiver \
    --model-name sam3_encoder \
    --version 1.0.0 \
    --serialized-file sam3_hiera_large.pt \
    --handler handler_encode.py \
    --export-path model_store
```

### NVIDIA Triton Inference Server

For maximum performance and multi-model serving:

- Native TensorRT optimization
- Dynamic batching
- Concurrent model execution
- HTTP/gRPC protocols

## Scaling Considerations

### Horizontal Scaling with Kubernetes

From [segment-anything-services deployment](https://github.com/developmentseed/segment-anything-services) (accessed 2025-11-23):

**Helm Chart Configuration**:
```yaml
# values.yaml
replicaCount: 3

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 32Gi
  requests:
    memory: 16Gi

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-l4
```

### Serverless Deployment (Modal)

From [Modal SAM 2 Example](https://modal.com/docs/examples/segment_anything) (accessed 2025-11-23):

```python
import modal

app = modal.App("sam3-service")
cache_vol = modal.Volume.from_name("sam3-cache", create_if_missing=True)

@app.cls(
    image=modal.Image.debian_slim()
        .pip_install("torch", "sam3", "fastapi"),
    gpu="A100",
    volumes={"/cache": cache_vol},
)
class SAM3Model:
    @modal.enter()
    def load_model(self):
        from sam3 import build_sam3_video_predictor
        self.predictor = build_sam3_video_predictor("sam3_hiera_large")

    @modal.method()
    def segment(self, image_bytes, text_prompt):
        # Process request
        return masks
```

**Benefits**:
- Auto-scaling to zero when idle
- Pay-per-second billing
- Fast cold starts with pre-warmed containers
- A100 GPUs on demand

### Cloud Provider Options

**AWS**:
- ECS/EKS with GPU instances (p3, p4, g5)
- SageMaker endpoints with auto-scaling
- Lambda with container images (CPU only)

**GCP**:
- Cloud Run with GPU support
- GKE with GPU node pools
- Vertex AI endpoints

**Azure**:
- Container Apps with GPU
- AKS with GPU nodes
- Machine Learning endpoints

## Performance Optimization

### AOTInductor for Fast Cold Starts

From [PyTorch SAM 2 Acceleration](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/) (accessed 2025-11-23):

**Key Insight**: torch.compile adds significant first-call overhead. Use AOTInductor for ahead-of-time compilation.

```python
import torch
from torch.export import export

# Export model ahead of time
model = build_sam3_image_model()
example_input = torch.randn(1, 3, 1024, 1024).cuda()

# Export to AOTInductor format
exported = export(model.image_encoder, (example_input,))
torch._inductor.aot_compile(
    exported.module(),
    example_input,
    options={"aot_inductor.output_path": "sam3_encoder.so"}
)

# Load at inference time (fast!)
model_encoder = torch._inductor.load("sam3_encoder.so")
```

**Performance Results** (from PyTorch blog):
- Up to **13x improvement** in p90 execution latency
- Cold start: 10+ seconds (compile) vs ~700ms (AOTInductor)

### Precision Optimization

```python
# Use float16 for encoder (significant speedup)
model.image_encoder = model.image_encoder.half()

# Use TF32 for matrix operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# GPU preprocessing (avoid CPU bottlenecks)
image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
image_tensor = image_tensor.float() / 255.0
```

### Batched Inference

From PyTorch optimization blog (accessed 2025-11-23):

```python
# Process multiple prompts in parallel
batch_size = 1024  # For AMG task

# Instead of loop:
# for prompt in prompts:
#     mask = model.predict(prompt)

# Use batched prediction:
prompts_tensor = torch.stack(prompts).cuda()
masks = model.predict_batch(prompts_tensor)  # Single forward pass
```

**Memory vs Latency Trade-off**:
- Batch size 64: ~4GB VRAM
- Batch size 1024: ~34GB VRAM
- Latency improvement: 1.7x faster

### Embedding Caching

```python
import redis

class EmbeddingCache:
    def __init__(self):
        self.redis = redis.Redis(host='redis', port=6379)

    def cache_embedding(self, image_hash, embedding):
        """Cache embedding with TTL."""
        self.redis.setex(
            f"emb:{image_hash}",
            3600,  # 1 hour TTL
            embedding.cpu().numpy().tobytes()
        )

    def get_embedding(self, image_hash):
        """Retrieve cached embedding."""
        data = self.redis.get(f"emb:{image_hash}")
        if data:
            return torch.from_numpy(
                np.frombuffer(data, dtype=np.float32)
            ).reshape(256, 64, 64)
        return None
```

## Memory Management

### GPU Memory Optimization

```python
# Clear cache between requests
torch.cuda.empty_cache()

# Use inference mode (no gradient tracking)
with torch.inference_mode():
    embedding = model.image_encoder(image)
    masks = model.mask_decoder(embedding, prompts)

# Stream large results
def stream_masks(masks):
    for mask in masks:
        yield encode_rle(mask)
```

### Model Sharding for Large Models

```python
# For multi-GPU deployment
model = build_sam3_image_model()

# Shard across GPUs
model.image_encoder = model.image_encoder.to("cuda:0")
model.text_encoder = model.text_encoder.to("cuda:1")
model.mask_decoder = model.mask_decoder.to("cuda:0")
```

## Monitoring and Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

REQUESTS = Counter('sam3_requests_total', 'Total requests')
LATENCY = Histogram('sam3_latency_seconds', 'Request latency')
GPU_MEMORY = Gauge('sam3_gpu_memory_bytes', 'GPU memory usage')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUESTS.inc()
    with LATENCY.time():
        response = await call_next(request)
    GPU_MEMORY.set(torch.cuda.memory_allocated())
    return response
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": processor is not None,
        "memory_used": torch.cuda.memory_allocated() / 1e9
    }

@app.get("/ready")
async def readiness_check():
    # Verify model can process request
    test_image = torch.randn(1, 3, 64, 64).cuda()
    try:
        with torch.no_grad():
            _ = processor.model.image_encoder(test_image)
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}
```

## Production Checklist

### Security
- [ ] Container runs as non-root user
- [ ] Network policies restrict inter-service communication
- [ ] API authentication (JWT/API keys)
- [ ] Input validation (image size limits, format checks)
- [ ] Rate limiting

### Reliability
- [ ] Health checks configured
- [ ] Graceful shutdown handling
- [ ] Retry logic for transient failures
- [ ] Circuit breakers for downstream services
- [ ] Logging with correlation IDs

### Performance
- [ ] Model weights cached on persistent volumes
- [ ] AOTInductor export for fast cold starts
- [ ] Appropriate GPU instance type selected
- [ ] Horizontal Pod Autoscaler configured
- [ ] Resource requests/limits set

### Observability
- [ ] Prometheus metrics exported
- [ ] Distributed tracing enabled
- [ ] Structured logging (JSON)
- [ ] Alerting rules defined
- [ ] Dashboard created

## Cost Optimization

### GPU Selection Guide

| GPU | VRAM | Best For | Cost/hr (AWS) |
|-----|------|----------|---------------|
| T4 | 16GB | Development, low throughput | $0.53 |
| L4 | 24GB | Production, good value | $0.81 |
| A10G | 24GB | Production, higher throughput | $1.21 |
| A100 40GB | 40GB | High throughput, batching | $3.67 |
| H100 | 80GB | Maximum performance | $9.22 |

### Spot Instances

Use spot/preemptible instances for batch processing:

```yaml
# Kubernetes spot node pool
nodeSelector:
  cloud.google.com/gke-spot: "true"

tolerations:
- key: "cloud.google.com/gke-spot"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

### Right-Sizing

- Start with smaller GPU (T4/L4)
- Monitor actual memory usage
- Scale up only if needed
- Use auto-scaling for variable load

## Example Production Architecture

```
                    ┌─────────────┐
                    │  Load       │
                    │  Balancer   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼──┐ ┌──────▼──────┐
       │  API Pod 1  │ │ ... │ │  API Pod N  │
       │  (FastAPI)  │ │     │ │  (FastAPI)  │
       └──────┬──────┘ └─────┘ └──────┬──────┘
              │                       │
              └───────────┬───────────┘
                          │
                   ┌──────▼──────┐
                   │   Redis     │
                   │   Cache     │
                   └──────┬──────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
│ GPU Worker 1│    │ GPU Worker 2│    │ GPU Worker N│
│ (TorchServe)│    │ (TorchServe)│    │ (TorchServe)│
│     A100    │    │     A100    │    │     A100    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Sources

**Source Documents:**
- SAM 3 GitHub Repository: https://github.com/facebookresearch/sam3
- SAM 3 Blog: https://ai.meta.com/blog/segment-anything-model-3/

**Web Research (accessed 2025-11-23):**
- [Deploying SAM with FastAPI and Docker](https://medium.com/@faical.ammisaid_48405/deploying-segment-anything-model-with-fastapi-and-docker-6e9ae439c7a8)
- [segment-anything-services](https://github.com/developmentseed/segment-anything-services) - TorchServe deployment
- [Deploying SAM with TorchServe](https://github.com/facebookresearch/segment-anything/issues/549) - Architecture patterns
- [Modal SAM 2 Example](https://modal.com/docs/examples/segment_anything) - Serverless deployment
- [PyTorch SAM 2 Acceleration](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/) - AOTInductor optimization

**Additional References:**
- [TorchServe Documentation](https://pytorch.org/serve/)
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
