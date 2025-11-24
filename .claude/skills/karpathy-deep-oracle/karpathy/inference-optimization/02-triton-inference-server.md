# Triton Inference Server

**NVIDIA's Production-Grade Multi-Framework Model Serving Platform**

From [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (accessed 2025-11-13):

> Triton Inference Server is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more.

---

## Overview

### What Triton Is

Triton Inference Server is NVIDIA's **open-source inference serving platform** designed for production deployment of ML models. Unlike vLLM (LLM-specific) or TensorFlow Serving (framework-specific), Triton is **framework-agnostic** and **multimodal-capable**.

**Key Differentiator**: Triton can serve TensorFlow, PyTorch, ONNX, TensorRT, and custom models **simultaneously** from a single server instance.

From [NVIDIA Triton Architecture Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html):

**Triton Architecture Components**:
- **Model Repository**: File-system based model storage
- **HTTP/REST and GRPC APIs**: KServe-compatible protocols
- **Backend System**: Pluggable framework support (PyTorch, TensorFlow, TensorRT, ONNX, Python, vLLM)
- **Scheduler**: Dynamic batching, sequence batching, ensemble pipelines
- **C API**: In-process deployment for edge devices

### When to Use Triton

**Triton Excels At**:
1. **Multi-framework deployment** - Serve PyTorch + TensorRT + ONNX models together
2. **Multimodal serving** - Vision + language models in ensemble pipelines
3. **Production-grade features** - Dynamic batching, model versioning, A/B testing
4. **Heterogeneous hardware** - GPUs (NVIDIA), CPUs (x86/ARM), AWS Inferentia

**Triton vs Alternatives**:

From [Reddit r/mlops discussion](https://www.reddit.com/r/mlops/comments/1frcu8b/why_use_ml_server_frameworks_like_triton_inf/) (accessed 2025-11-13):

| Feature | Triton | TorchServe | vLLM |
|---------|--------|------------|------|
| Multi-framework | ✓ All major | PyTorch only | LLMs only |
| Dynamic batching | ✓ Advanced | ✓ Basic | ✓ Continuous |
| Ensemble models | ✓ Native | ✗ | ✗ |
| VLM support | ✓ Excellent | Limited | ✓ Recent |
| Production maturity | ✓✓✓ | ✓✓ | ✓✓ |
| Ease of setup | Medium | Easy | Easy |

From [Neptune.ai Model Serving Comparison](https://neptune.ai/blog/ml-model-serving-best-tools) (accessed 2025-11-13):
- **TorchServe**: "Mature and robust for PyTorch teams, easy conversion to C++"
- **Triton**: "Best for multi-framework production deployments with advanced batching"

---

## Dynamic Batching

### How Dynamic Batching Works

From [NVIDIA Triton Batcher Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) (accessed 2025-11-13):

**Dynamic Batching** = Server-side request combining to create batches dynamically.

**Key Concept**: Unlike static batching (client sends batch), Triton **delays requests briefly** to accumulate more requests into larger batches, increasing throughput.

**Configuration Example**:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

**How It Works**:
1. Request arrives at scheduler
2. Triton waits up to `max_queue_delay_microseconds` (100μs)
3. If more requests arrive, batch them together
4. Send batch of `preferred_batch_size` (4 or 8) to model
5. If delay expires, send whatever batch size exists

**Performance Impact**:

From [NVIDIA Conceptual Guide - Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html) (accessed 2025-11-13):

- **Without dynamic batching**: 10 req/s → batch size 1 → 10 inferences/s
- **With dynamic batching (batch=8)**: 10 req/s → ~1-2 batches/s → same throughput, **5-8× GPU utilization**

**When to Use**:
- High request volume (>10 req/s)
- Latency budget allows 100-1000μs delay
- Batch processing is significantly faster than single inference

**When NOT to Use**:
- Ultra-low latency requirements (<10ms)
- Low request volume (sporadic requests)
- Model doesn't benefit from batching (already optimized)

### Delayed Batching vs Continuous Batching

From [Rohan's Bytes - Batch Inference at Scale](https://www.rohan-paul.com/p/batch-inference-at-scale-processing) (accessed 2025-11-13):

**Delayed Batching (Triton)**:
- Wait up to `max_queue_delay_microseconds`
- Form batch, execute completely, return all responses
- Good for: Non-generative models (classification, embeddings)

**Continuous Batching (vLLM)**:
- Requests join/leave batch at each iteration step
- Iterative sequence support (LLMs generate token-by-token)
- Good for: Generative models (LLMs, autoregressive)

**Triton's Iterative Sequences** (NEW):

From [NVIDIA Triton Batcher Documentation - Iterative Sequences](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html#iterative-sequences) (accessed 2025-11-13):

Triton now supports **continuous/inflight batching** via iterative sequences:

```protobuf
sequence_batching {
  iterative_sequence: true
}
```

- Single request processed over multiple scheduling iterations
- Backend yields back to scheduler between steps
- Enables continuous batching like vLLM
- Supports decoupled responses (streaming)

**Use Case**: LLM inference where each request generates variable-length output.

### Priority Levels & Queue Policy

From [NVIDIA Triton Batcher Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html#priority-levels) (accessed 2025-11-13):

**Multi-Priority Queues**:
```protobuf
dynamic_batching {
  priority_levels: 3
  default_priority_level: 1

  default_queue_policy {
    max_queue_size: 100
    timeout_action: REJECT
    default_timeout_microseconds: 10000
  }

  priority_queue_policy {
    key: 0  # Highest priority
    value {
      max_queue_size: 50
      allow_timeout_override: true
    }
  }
}
```

**Priority Level Features**:
- Higher priority requests bypass lower priority requests
- Per-priority queue policies (size, timeout)
- Client sets priority in request metadata
- Useful for: Production traffic (high) vs batch jobs (low)

---

## Model Ensembles

### Ensemble Architecture

From [NVIDIA Triton Ensemble Models Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html) (accessed 2025-11-13):

**Ensemble Model** = Pipeline of multiple models with tensor connections between stages.

**Key Benefit**: Execute multi-stage pipelines **within Triton** without client-side orchestration.

**Example: VLM Ensemble (Vision + Language)**:
```
Client Request (image + text)
    ↓
[Vision Encoder] → image embeddings
    ↓
[Text Tokenizer] → token IDs
    ↓
[Multimodal Fusion] → combined features
    ↓
[Language Decoder] → generated text
    ↓
Client Response
```

**Configuration Example**:

From [Medium - Getting Started with Triton Ensemble](https://medium.com/@raghunandan.ramesha/getting-started-with-deploying-ml-models-using-triton-ensemble-mode-on-amazon-sagemaker-b55d0877eaaa) (accessed 2025-11-13):

```protobuf
name: "vlm_ensemble"
platform: "ensemble"

input [
  {
    name: "IMAGE"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  },
  {
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "GENERATED_TEXT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "vision_encoder"
      model_version: -1
      input_map {
        key: "input_image"
        value: "IMAGE"
      }
      output_map {
        key: "embeddings"
        value: "vision_features"
      }
    },
    {
      model_name: "text_tokenizer"
      model_version: -1
      input_map {
        key: "input_text"
        value: "TEXT"
      }
      output_map {
        key: "tokens"
        value: "text_tokens"
      }
    },
    {
      model_name: "multimodal_decoder"
      model_version: -1
      input_map {
        key: "vision_input"
        value: "vision_features"
      }
      input_map {
        key: "text_input"
        value: "text_tokens"
      }
      output_map {
        key: "output_text"
        value: "GENERATED_TEXT"
      }
    }
  ]
}
```

### Ensemble Use Cases

**1. Preprocessing + Model + Postprocessing**:

From [ML6 Blog - Triton Ensemble for Transformers](https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3) (accessed 2025-11-13):

```
[Tokenizer Model (Python)] → [BERT Model (TensorRT)] → [Detokenizer Model (Python)]
```

**Benefits**:
- Client sends raw text, receives raw text
- Tokenization logic server-side (no client library dependency)
- Can use different backends per stage (Python for tokenizer, TensorRT for model)

**2. A/B Testing Within Ensemble**:
```
[Feature Extractor] → [Model A (90% traffic)]
                   → [Model B (10% traffic)]
```

**3. Multimodal VLM Pipelines**:

From [NVIDIA TensorRT-LLM Backend - Multimodal Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/multimodal.html) (accessed 2025-11-13):

```
[Visual Encoder (TensorRT)] → [LLaVA Decoder (TensorRT-LLM)] → [Text Output]
```

**LLaVA on Triton**:
- Visual encoder processes images
- TensorRT-LLM backend handles LLM decoding
- Ensemble coordinates both stages
- Supports streaming responses (decoupled mode)

---

## Backend System

### Supported Backends

From [NVIDIA Triton GitHub - Backends](https://github.com/triton-inference-server/backend) (accessed 2025-11-13):

**Official Backends**:
1. **TensorRT** - NVIDIA GPU-optimized (fastest for inference)
2. **PyTorch (LibTorch)** - Native PyTorch JIT models
3. **ONNX Runtime** - Cross-platform ONNX models
4. **TensorFlow** - TensorFlow SavedModel
5. **Python Backend** - Custom Python code (preprocessing, business logic)
6. **OpenVINO** - Intel CPU/GPU optimization
7. **FIL (Forest Inference Library)** - RAPIDS GPU-accelerated tree models
8. **vLLM Backend** - LLM-specific continuous batching (NEW)
9. **TensorRT-LLM Backend** - Optimized LLM serving (NEW)

**Backend Selection Strategy**:

From [Ximilar Blog - Best Tools for ML Model Serving](https://www.ximilar.com/blog/the-best-tools-for-machine-learning-model-serving/) (accessed 2025-11-13):

| Model Type | Recommended Backend | Why |
|------------|---------------------|-----|
| Vision (ResNet, EfficientNet) | TensorRT | 2-5× faster than PyTorch |
| NLP (BERT, small transformers) | ONNX Runtime or TensorRT | Cross-platform or max speed |
| LLMs (GPT, LLaMA) | TensorRT-LLM or vLLM | Continuous batching + KV cache |
| Custom preprocessing | Python Backend | Flexibility for complex logic |
| Multimodal (CLIP, LLaVA) | Ensemble (TensorRT + Python) | Mix optimized + flexible |

### Python Backend Deep Dive

From [NVIDIA Python Backend GitHub](https://github.com/triton-inference-server/python_backend) (accessed 2025-11-13):

**Python Backend** = Serve models written in Python without C++ code.

**Use Cases**:
1. **Preprocessing/Postprocessing**: Tokenization, image augmentation, NMS
2. **Custom Models**: HuggingFace Transformers, scikit-learn, custom PyTorch
3. **Business Logic**: Database lookups, API calls, complex rules

**Example: HuggingFace BERT**:
```python
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizer, BertModel
import torch

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.model.cuda()

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input text
            text = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text_str = text.as_numpy()[0].decode('utf-8')

            # Tokenize
            inputs = self.tokenizer(text_str, return_tensors="pt").to("cuda")

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Return embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            output_tensor = pb_utils.Tensor("EMBEDDINGS", embeddings)
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses
```

**Performance Considerations**:

From [Stack Overflow - Triton Python Backend Performance](https://stackoverflow.com/questions/73829280/nvidia-triton-vs-torchserve-for-sagemaker-inference) (accessed 2025-11-13):

> "Python backend adds 5-20ms overhead vs native backends (TensorRT, ONNX). Use for preprocessing logic, but serve core model with TensorRT for best performance."

### vLLM Backend Integration

From [NVIDIA vLLM Backend GitHub](https://github.com/triton-inference-server/vllm_backend) (accessed 2025-11-13):

**vLLM Backend** = Integrate vLLM's continuous batching into Triton ecosystem.

**Why Use vLLM Backend in Triton**:
1. **Best of Both Worlds**: vLLM's LLM optimizations + Triton's production features
2. **Unified API**: Serve LLMs alongside vision models with same HTTP/GRPC interface
3. **Ensemble Capability**: Use LLM as part of multimodal pipeline

**Example: Multimodal RAG System**:
```
[Document Embedder (ONNX)] → [Vector DB Lookup (Python)] → [LLM (vLLM Backend)]
```

**Performance**:

From [Medium - Running Triton with vLLM](https://medium.com/@manishsingh7163/running-triton-inference-server-with-vllm-76309370cf50) (accessed 2025-11-13):

- vLLM backend supports **PagedAttention** (2-24× throughput vs naive)
- Dynamic batching at iteration level (continuous batching)
- Streaming responses via decoupled mode
- Compatible with Triton's model versioning and A/B testing

---

## Production Deployment Workflows

### Model Repository Structure

From [NVIDIA Triton Model Repository Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) (accessed 2025-11-13):

**Model Repository Layout**:
```
model_repository/
├── vision_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan  # TensorRT engine
├── text_tokenizer/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py  # Python backend
├── multimodal_decoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── vlm_ensemble/
    └── config.pbtxt  # Ensemble definition
```

**Version Management**:
- Each model has versioned subdirectories (1/, 2/, 3/)
- Triton loads all versions or specific versions
- Supports rolling updates (load v2, drain v1 traffic, unload v1)

**Model Configuration** (`config.pbtxt`):
```protobuf
name: "vision_encoder"
backend: "tensorrt"
max_batch_size: 32

input [
  {
    name: "input_image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]

instance_group [
  {
    count: 2  # 2 instances per GPU
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 500
}
```

### Kubernetes Deployment

From [AWS - Triton on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-models-frameworks-triton.html) (accessed 2025-11-13):

**Triton on Kubernetes**:
- Deploy as StatefulSet or Deployment
- Use persistent volumes for model repository
- Horizontal Pod Autoscaling based on request latency
- Load balancer for multi-replica serving

**Example Kubernetes Manifest**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        args:
          - tritonserver
          - --model-repository=s3://my-bucket/models
          - --log-verbose=1
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000  # HTTP
        - containerPort: 8001  # GRPC
        - containerPort: 8002  # Metrics
```

**Autoscaling**:

From [E2E Networks - Multi-Model Inference with Triton](https://www.e2enetworks.com/blog/multi-model-inference-with-triton-inference-server) (accessed 2025-11-13):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: triton_inference_request_duration_us
      target:
        type: AverageValue
        averageValue: "50000"  # 50ms target latency
```

### Monitoring & Metrics

From [NVIDIA Triton Metrics Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html) (accessed 2025-11-13):

**Prometheus Metrics Endpoint**: `http://triton:8002/metrics`

**Key Metrics**:

**Request Metrics**:
- `nv_inference_request_success` - Successful inference requests
- `nv_inference_request_failure` - Failed requests
- `nv_inference_request_duration_us` - Request latency (histogram)
- `nv_inference_queue_duration_us` - Time spent in queue

**Model Metrics**:
- `nv_inference_exec_count` - Model execution count
- `nv_inference_count` - Total inferences (batch size accounted)
- `nv_gpu_utilization` - GPU utilization per model
- `nv_gpu_memory_total_bytes` - GPU memory usage

**Batch Metrics**:
- `nv_inference_pending_request_count` - Requests waiting for batching
- `nv_inference_compute_input_duration_us` - Input preparation time
- `nv_inference_compute_infer_duration_us` - Actual model execution time
- `nv_inference_compute_output_duration_us` - Output processing time

**Grafana Dashboard Example**:
```promql
# Average latency per model
rate(nv_inference_request_duration_us_sum[5m]) /
rate(nv_inference_request_duration_us_count[5m])

# Throughput (requests/sec)
rate(nv_inference_request_success[1m])

# GPU utilization
avg(nv_gpu_utilization) by (model)
```

---

## Triton vs TorchServe vs vLLM

### Decision Framework

From [Axel Mendoza - Best Model Serving Runtimes](https://www.axelmendoza.com/posts/best-model-serving-runtimes/) (accessed 2025-11-13):

**When to Use Triton**:
- ✓ Multi-framework deployment (PyTorch + TensorFlow + ONNX)
- ✓ Multimodal models (vision + language ensembles)
- ✓ Production features needed (A/B testing, versioning, metrics)
- ✓ Hardware heterogeneity (GPUs + CPUs in same deployment)
- ✓ High-throughput requirements (dynamic batching critical)

**When to Use TorchServe**:
- ✓ PyTorch-only deployment
- ✓ Simpler setup (less configuration)
- ✓ Rapid prototyping
- ✓ Don't need multi-framework support

**When to Use vLLM**:
- ✓ LLM-only workload (GPT, LLaMA, etc.)
- ✓ Need continuous batching (variable output lengths)
- ✓ PagedAttention benefits (large KV cache)
- ✓ Don't need multimodal support

**When to Use Triton + vLLM Backend**:
- ✓ Multimodal system with LLM component
- ✓ Want vLLM's LLM optimizations + Triton's production features
- ✓ Need unified API for vision models + LLMs

### Performance Comparison

From [SqueezeBits Blog - vLLM vs TensorRT-LLM Comparison](https://blog.squeezebits.com/vllm-vs-tensorrtllm-13-visionlanguage-models-40761) (accessed 2025-11-13):

**Serving Framework Comparison (VLM Workload)**:

| Metric | vLLM Standalone | Triton + vLLM Backend | Triton + TensorRT-LLM |
|--------|----------------|----------------------|----------------------|
| Setup complexity | Easy | Medium | Hard |
| Throughput (req/s) | 45 | 43 | 67 |
| Latency P50 (ms) | 120 | 125 | 89 |
| Latency P99 (ms) | 380 | 390 | 210 |
| Multi-model support | ✗ | ✓ | ✓ |
| A/B testing | ✗ | ✓ | ✓ |

**Insights**:
- TensorRT-LLM fastest (1.5× throughput vs vLLM)
- vLLM backend adds ~5ms overhead
- Triton production features worth the small overhead

---

## ARR-COC-VIS Use Cases

### Multimodal VLM Ensemble

**ARR-COC Ensemble Pipeline**:
```
[Texture Array Generation (Python Backend)]
    ↓ 13-channel texture array
[Propositional Scorer (TensorRT)]
    ↓ information content scores
[Perspectival Scorer (TensorRT)]
    ↓ salience scores
[Participatory Scorer (TensorRT)]
    ↓ query-aware scores
[Opponent Processing (Python Backend)]
    ↓ balanced relevance map
[Qwen3-VL Decoder (vLLM Backend)]
    ↓ generated text
Client Response
```

**Benefits for ARR-COC**:
1. **Modular Scorers**: Each scorer as separate model (easy A/B testing)
2. **Dynamic Batching**: Batch relevance scoring across multiple images
3. **Mixed Backends**: TensorRT for scorers (fast), Python for opponent processing (flexible)
4. **Production Ready**: Metrics, versioning, canary deployments

### VQA Production Deployment

**Scenario**: Deploy ARR-COC VQA system to production

**Model Repository**:
```
arr_coc_vqa/
├── texture_generator/      # Python backend
├── propositional_scorer/   # TensorRT
├── perspectival_scorer/    # TensorRT
├── participatory_scorer/   # TensorRT
├── opponent_processor/     # Python backend
├── qwen3_vl_decoder/       # vLLM backend
└── arr_coc_ensemble/       # Ensemble tying all together
```

**Dynamic Batching Configuration**:
```protobuf
# For relevance scorers (fast, batch-friendly)
dynamic_batching {
  preferred_batch_size: [ 16, 32 ]
  max_queue_delay_microseconds: 1000
}

# For VLM decoder (slower, careful batching)
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 5000
}
```

**Monitoring ARR-COC Metrics**:
```promql
# Latency breakdown per stage
rate(nv_inference_compute_infer_duration_us_sum{model="propositional_scorer"}[5m]) /
rate(nv_inference_compute_infer_duration_us_count{model="propositional_scorer"}[5m])

# Token allocation distribution (custom metric via Python backend)
histogram_quantile(0.95, arr_coc_token_budget_allocated_bucket)

# Query-aware relevance correlation (custom metric)
avg(arr_coc_participatory_score_mean) by (query_type)
```

### A/B Testing Scorers

**Scenario**: Test new perspectival scorer (Jungian archetypes v2)

**Ensemble Configuration**:
```protobuf
ensemble_scheduling {
  step [
    # ... texture generation, propositional scorer ...

    {
      model_name: "perspectival_scorer"
      model_version: 1  # 90% traffic to v1
    },
    {
      model_name: "perspectival_scorer_v2"  # 10% traffic to v2
      model_version: 1
    }

    # ... merge results, continue pipeline ...
  ]
}
```

**Comparison Metrics**:
- VQA accuracy (v1 vs v2)
- Latency impact (v2 might be slower)
- Relevance distribution (are results different?)

---

## Production Best Practices

### Configuration Recommendations

From [NVIDIA Triton Best Practices](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) (accessed 2025-11-13):

**1. Batch Size Selection**:
- Start with `max_batch_size` = largest batch GPU can handle
- Use `preferred_batch_size` **only if** specific sizes have significantly better performance
- For most models, don't set `preferred_batch_size` (let Triton create largest possible batch)

**2. Delay Tuning**:
- Start with `max_queue_delay_microseconds: 0` (no delay)
- Measure baseline latency/throughput
- Increase delay to 100μs, 500μs, 1000μs
- Find sweet spot where throughput increases without violating latency SLA

**3. Instance Count**:
```protobuf
instance_group [
  {
    count: 2-4  # Multiple instances per GPU
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```
- Use 2-4 instances per GPU for concurrency
- More instances = better GPU utilization (overlap compute/memory)
- Too many instances = memory contention

**4. Model Warmup**:
```protobuf
model_warmup [
  {
    name: "typical_batch"
    batch_size: 8
    inputs {
      key: "input"
      value {
        data_type: TYPE_FP32
        dims: [ 3, 224, 224 ]
        zero_data: true
      }
    }
  }
]
```
- Warmup prevents first-request latency spikes
- Preallocates GPU memory
- Initializes TensorRT engines

### Troubleshooting Common Issues

**Issue: Low GPU Utilization**

From [NVIDIA Triton Debugging Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/debugging_guide.html) (accessed 2025-11-13):

**Symptoms**:
- GPU utilization <50%
- High request latency
- Low throughput

**Solutions**:
1. Enable dynamic batching (increase batch size)
2. Increase `instance_group count` (2-4 instances)
3. Check if model is CPU-bound (profile with Nsight Systems)
4. Reduce `max_queue_delay_microseconds` if queue is empty

**Issue: High Latency**

**Symptoms**:
- P99 latency >SLA
- Requests spending time in queue

**Solutions**:
1. Reduce `max_queue_delay_microseconds`
2. Reduce `preferred_batch_size` (smaller batches = lower latency)
3. Add more GPU instances (scale horizontally)
4. Use model optimization (quantization, TensorRT)

**Issue: Out of Memory**

**Symptoms**:
- `CUDA_ERROR_OUT_OF_MEMORY`
- Server crashes under load

**Solutions**:
1. Reduce `max_batch_size`
2. Reduce `instance_group count`
3. Enable model versioning policy to limit loaded versions
4. Use dynamic sequence batching (better memory for variable-length)

---

## Sources

**NVIDIA Official Documentation**:
- [Triton Inference Server User Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) - Architecture, features, configuration (accessed 2025-11-13)
- [Dynamic Batcher Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) - Batching strategies, configuration (accessed 2025-11-13)
- [Ensemble Models Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html) - Pipeline construction (accessed 2025-11-13)
- [Model Repository Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) - Model layout, versioning (accessed 2025-11-13)
- [Metrics Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html) - Prometheus metrics (accessed 2025-11-13)
- [TensorRT-LLM Backend - Multimodal Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/multimodal.html) - VLM deployment (accessed 2025-11-13)

**GitHub Repositories**:
- [triton-inference-server/server](https://github.com/triton-inference-server/server) - Main repository (accessed 2025-11-13)
- [triton-inference-server/backend](https://github.com/triton-inference-server/backend) - Backend API, examples (accessed 2025-11-13)
- [triton-inference-server/python_backend](https://github.com/triton-inference-server/python_backend) - Python backend (accessed 2025-11-13)
- [triton-inference-server/vllm_backend](https://github.com/triton-inference-server/vllm_backend) - vLLM integration (accessed 2025-11-13)
- [triton-inference-server/tutorials](https://github.com/triton-inference-server/tutorials) - Examples, tutorials (accessed 2025-11-13)

**Community Resources**:
- [Reddit r/mlops - Triton Discussion](https://www.reddit.com/r/mlops/comments/1frcu8b/why_use_ml_server_frameworks_like_triton_inf/) - Framework comparison (accessed 2025-11-13)
- [Medium - Triton Ensemble on SageMaker](https://medium.com/@raghunandan.ramesha/getting-started-with-deploying-ml-models-using-triton-ensemble-mode-on-amazon-sagemaker-b55d0877eaaa) - Ensemble examples (accessed 2025-11-13)
- [ML6 Blog - Triton Ensemble for Transformers](https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3) - Transformer deployment (accessed 2025-11-13)
- [Rohan's Bytes - Batch Inference at Scale](https://www.rohan-paul.com/p/batch-inference-at-scale-processing) - Batching strategies comparison (accessed 2025-11-13)

**Comparison Articles**:
- [Neptune.ai - Best Model Serving Tools](https://neptune.ai/blog/ml-model-serving-best-tools) - Triton vs TorchServe vs BentoML (accessed 2025-11-13)
- [Axel Mendoza - Best Model Serving Runtimes](https://www.axelmendoza.com/posts/best-model-serving-runtimes/) - Framework comparison (accessed 2025-11-13)
- [Ximilar Blog - Best Tools for ML Model Serving](https://www.ximilar.com/blog/the-best-tools-for-machine-learning-model-serving/) - Backend selection (accessed 2025-11-13)
- [Stack Overflow - Triton vs TorchServe](https://stackoverflow.com/questions/73829280/nvidia-triton-vs-torchserve-for-sagemaker-inference) - SageMaker deployment (accessed 2025-11-13)
- [SqueezeBits Blog - vLLM vs TensorRT-LLM](https://blog.squeezebits.com/vllm-vs-tensorrtllm-13-visionlanguage-models-40761) - VLM performance comparison (accessed 2025-11-13)

**Cloud Provider Documentation**:
- [AWS - Triton on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-models-frameworks-triton.html) - SageMaker deployment (accessed 2025-11-13)
- [E2E Networks - Multi-Model Inference with Triton](https://www.e2enetworks.com/blog/multi-model-inference-with-triton-inference-server) - Kubernetes deployment (accessed 2025-11-13)

---

**Created**: 2025-11-13
**Lines**: ~520
**Purpose**: Part of EXPANSION 19 (Production Inference Optimization) - Comprehensive Triton Inference Server knowledge for production VLM deployment, ensemble models, and dynamic batching strategies.
