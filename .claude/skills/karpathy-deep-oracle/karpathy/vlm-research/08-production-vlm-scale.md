# Production VLM Systems at Scale

## Overview

Deploying vision-language models (VLMs) at production scale requires sophisticated infrastructure, observability, and cost optimization strategies. This guide covers serving frameworks, monitoring systems, performance optimization, and real-world deployment patterns drawn from industry experience.

**Key Challenges:**
- High computational and memory requirements (70B+ parameter models need 140GB+ VRAM)
- Latency-sensitive applications requiring sub-second response times
- Cost management at scale (GPU infrastructure is expensive)
- Operational complexity (distributed systems, fault tolerance)
- Variable workload patterns (spikes, batch processing)

**Primary Serving Frameworks:**
- **vLLM**: Open-source, high-performance, hardware-agnostic
- **TGI (Text Generation Inference)**: HuggingFace production server
- **TensorRT-LLM**: NVIDIA-optimized for maximum GPU performance
- **llm-d**: Distributed orchestration layer for vLLM at scale

From [Why vLLM is the best choice for AI inference today](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today) (accessed 2025-02-02)

---

## Section 1: Serving Infrastructure (~90 lines)

### vLLM: The Open-Source Standard

vLLM has emerged as the default serving solution for production LLM/VLM deployments, with impressive community adoption and enterprise backing from Red Hat, PyTorch Foundation, and others.

**Core Advantages:**
- **PagedAttention**: Revolutionary KV-cache management treating GPU memory like OS virtual memory
- **Continuous batching**: Dynamic request handling that maximizes GPU utilization
- **Hardware flexibility**: Support for NVIDIA, AMD, Intel, Google TPU, AWS Inferentia
- **Model ecosystem**: 100+ supported architectures including multimodal models

**Architecture Highlights:**

```
Request → Router → Prefill Workers (compute-intensive)
                 ↓
          KV Cache Transfer
                 ↓
          Decode Workers (memory-bound)
                 ↓
          Response
```

**Performance Characteristics:**
- 2-3x better GPU utilization than static batching
- Near-zero memory waste from PagedAttention
- Sub-second time-to-first-token for most queries
- Linear scaling with proper parallelization

From [vLLM Production Stack Documentation](https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.2/dev_guide/peripheral/models.html) (accessed 2025-02-02)

### TGI: HuggingFace Production Integration

Text Generation Inference provides tight HuggingFace ecosystem integration with production-ready features.

**Key Features:**
- Direct HuggingFace Hub integration
- Flash Attention 2 support
- Token streaming
- Distributed tracing
- Quantization support (GPTQ, AWQ, FP8)

**When to Choose TGI:**
- Heavy HuggingFace ecosystem usage
- Need tight integration with HF models/tokenizers
- Smaller deployments (single-node focus)
- Prototyping and development

**Limitations:**
- Less mature distributed serving
- Fewer optimization options vs vLLM
- Limited hardware support (primarily NVIDIA)

### TensorRT-LLM: NVIDIA-Optimized Performance

For NVIDIA-exclusive deployments requiring maximum performance.

**Advantages:**
- Compilation-based optimization
- Best NVIDIA GPU utilization
- Advanced quantization (FP8, INT4)
- Tight CUDA kernel optimization

**Trade-offs:**
- NVIDIA hardware lock-in
- Complex build process
- Requires Triton Inference Server for production
- Longer iteration cycles (compilation overhead)

**Typical Use Case:** High-volume, NVIDIA-standardized infrastructure where every millisecond matters.

### Framework Comparison Summary

| Feature | vLLM | TGI | TensorRT-LLM |
|---------|------|-----|--------------|
| Hardware Support | NVIDIA, AMD, Intel, TPU, CPU | Primarily NVIDIA | NVIDIA Only |
| Model Count | 100+ architectures | ~40 models | ~60 models |
| Distributed Serving | llm-d native | Limited | Via Triton |
| Ease of Setup | Easy | Easy | Complex |
| Performance | Excellent | Good | Best (NVIDIA) |
| Community | Large, active | Active | NVIDIA-driven |
| Production Ready | ✓ | ✓ | ✓ (with Triton) |

From [vLLM vs TGI comparison](https://modal.com/blog/vllm-vs-tgi-article) and [Red Hat vLLM analysis](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today) (accessed 2025-02-02)

### llm-d: Distributed Orchestration at Scale

The llm-d project (Red Hat, Google Cloud, IBM Research, NVIDIA, CoreWeave) provides Kubernetes-native distributed serving built on vLLM.

**Architecture Components:**
- **vLLM**: Inference engine
- **Inference Gateway**: AI-aware routing
- **Kubernetes**: Orchestration platform

**Key Capabilities:**
- Disaggregated prefill/decode workers
- Intelligent request routing (cache-aware, load-balanced)
- Distributed prefix caching
- Heterogeneous hardware support
- Independent scaling of components

**Performance Gains:**
- 30-50% infrastructure cost reduction
- 2-3x better GPU utilization
- Maintains low-latency SLAs
- Proven at 100+ node deployments

From [Why vLLM is the best choice for AI inference today](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today) (accessed 2025-02-02)

---

## Section 2: Monitoring & Observability (~80 lines)

### Production Observability Stack

A comprehensive monitoring solution combines metrics collection, distributed tracing, and visualization.

**Core Components:**
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing
- **OpenTelemetry**: Unified telemetry framework

From [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) (accessed 2025-02-02)

### Critical VLM Metrics

**Request Metrics:**
- `vllm_num_requests_waiting`: Queue depth (autoscaling trigger)
- `vllm_num_requests_running`: Active processing
- `vllm_request_success_total`: Success rate
- Request latency percentiles (p50, p95, p99)

**Performance Metrics:**
- `vllm_time_to_first_token`: User-perceived latency
- `vllm_time_per_output_token`: Generation speed
- `vllm_request_prompt_tokens`: Input size distribution
- `vllm_request_generation_tokens`: Output size distribution
- Throughput (tokens/second)

**Resource Metrics:**
- GPU utilization percentage
- GPU memory usage
- KV cache utilization
- Batch size distribution

**Token Economics:**
- Total tokens processed
- Cost per token (by model/tier)
- Cost per request
- Daily/monthly spend tracking

### Observability Implementation

**vLLM Metrics Endpoint:**
```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'vllm-server'
    scrape_interval: 5s
    static_configs:
      - targets: ['vllm-server:8000']
```

**Grafana Dashboard Essentials:**
- Request rate and latency (time series)
- GPU utilization (gauge + heatmap)
- Queue depth and batch size (area chart)
- Error rate and success percentage
- Token throughput (counter)
- Cost tracking (calculated metric)

**Alerting Rules:**
```yaml
groups:
  - name: vllm_alerts
    rules:
      - alert: HighLatency
        expr: vllm_time_to_first_token_seconds > 2.0
        for: 5m

      - alert: QueueBacklog
        expr: vllm_num_requests_waiting > 100
        for: 2m

      - alert: LowGPUUtil
        expr: gpu_utilization_percent < 30
        for: 10m
```

From [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) (accessed 2025-02-02)

### Distributed Tracing

**OpenTelemetry Integration:**
```python
# vLLM server with OTLP tracing
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=grpc://jaeger:4317
  - OTEL_SERVICE_NAME=vllm-server
```

**Trace Visualization:** End-to-end request flow from RAG application → embedding → vector DB → LLM generation → vLLM server. Each component visible as spans.

**Key Insights from Tracing:**
- Identify bottlenecks in multi-stage pipelines
- Understand cache hit/miss patterns
- Debug slow requests
- Optimize component interactions

From [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) (accessed 2025-02-02)

---

## Section 3: Cost Optimization (~70 lines)

### Infrastructure Cost Drivers

**GPU Costs:**
- H100 (80GB): ~$3-4/hour cloud, $30K+ on-prem
- A100 (80GB): ~$2-3/hour cloud, $15K+ on-prem
- L40S (48GB): ~$1-2/hour cloud, $8K+ on-prem
- MI300X (192GB): Competitive AMD alternative

**Hidden Costs:**
- Network bandwidth (inter-node communication)
- Storage (model weights, KV cache offload)
- Idle GPU time (under-utilization)
- Over-provisioning for peak loads

### Optimization Strategies

**1. Right-Sizing Infrastructure**

Use disaggregated serving to match hardware to workload:
- **Prefill workers**: High-end GPUs (H100) for compute-intensive prompt processing
- **Decode workers**: Cost-effective GPUs (A100, L40S) for memory-bound token generation
- **CPU workers**: Ultra-low-frequency requests

**Cost Savings:** 40-60% reduction through optimal hardware allocation.

From [Why vLLM is the best choice for AI inference today](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today) (accessed 2025-02-02)

**2. Batch Processing**

Process non-real-time requests in batches:
```python
# OpenAI Batch API style
batch_job = {
    "requests": [...],  # 1000s of requests
    "max_wait_time": "24h",
    "priority": "low"
}
```

**Benefits:**
- 50% cost reduction vs real-time
- Better GPU utilization (larger batches)
- Reduced infrastructure overhead

From [Batch Processing for LLM Cost Savings](https://www.prompts.ai/en/blog/batch-processing-for-llm-cost-savings) (accessed 2025-02-02)

**3. Caching Strategies**

**Prefix Caching:** Share KV cache for common prompts
- System prompts (same across requests)
- Few-shot examples
- RAG context chunks

**Semantic Caching:** Cache similar queries
- Use embedding similarity
- TTL-based invalidation
- Works well for FAQ-style workloads

**Impact:** 30-70% cache hit rate can halve infrastructure needs.

**4. Model Optimization**

**Quantization:**
- FP16 → INT8: 2x memory reduction, minimal quality loss
- FP16 → INT4: 4x memory reduction, <5% quality degradation
- FP8: Emerging standard (H100 native support)

**Distillation:** Train smaller model to mimic larger one
- 70B → 7B: 10x cost reduction
- Preserve 90-95% task performance

**5. Autoscaling**

Dynamic scaling based on demand:
```yaml
# KEDA autoscaling config
triggers:
  - type: prometheus
    metadata:
      query: vllm_num_requests_waiting
      threshold: '50'
```

**Strategies:**
- Scale up: Queue depth > threshold
- Scale down: Low utilization for 10+ minutes
- Pre-warm instances during known peaks

**6. Spot Instances**

Use spot/preemptible VMs for batch workloads:
- 60-80% cost savings
- Acceptable for non-critical workloads
- Requires fault-tolerant architecture

From [Scale AI: Reducing Cold Start Time for LLM Inference](https://scale.com/blog/reduce-cold-start-time-llm-inference) (accessed 2025-02-02)

---

## Section 4: Latency & Performance (~70 lines)

### Latency Components

**Time-to-First-Token (TTFT):**
- **Definition:** Time from request start to first token output
- **User Impact:** Perceived responsiveness
- **Target:** <500ms for interactive apps, <2s for batch

**Breakdown:**
1. Network latency: 10-50ms
2. Queue wait time: 0-1000ms (depends on load)
3. Prefill computation: 100-500ms (prompt-dependent)
4. First token generation: 50-100ms

**Inter-Token Latency:**
- **Definition:** Time between subsequent tokens
- **User Impact:** Streaming smoothness
- **Target:** <50ms for real-time feel

**Total Latency:** TTFT + (num_tokens × inter-token_latency)

### Performance Optimization Techniques

**1. Continuous Batching**

Traditional static batching waits for all sequences to complete. vLLM's continuous batching:
- Adds new requests between token generations
- Removes completed sequences immediately
- Maintains optimal batch size dynamically

**Result:** 2-3x throughput improvement.

**2. PagedAttention**

Eliminates memory fragmentation:
- Stores KV cache in non-contiguous blocks
- Dynamic allocation as sequences grow
- Zero waste from pre-allocation

**Impact:** 2-4x higher batch sizes possible.

**3. Speculative Decoding**

Generate multiple tokens speculatively:
```
Draft model (small): Fast, parallel token generation
Verify with main model: Accept or reject draft tokens
```

**Speedup:** 2-3x for compatible tasks, especially with high token acceptance rates.

**4. Flash Attention**

Optimized attention kernel:
- Reduces memory bandwidth requirements
- Enables longer context windows
- 2-4x faster than naive attention

From [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (accessed 2025-02-02)

**5. KV Cache Management**

**Offloading:** Move inactive KV cache to CPU/SSD
- Frees GPU memory for active requests
- Swap back when needed
- Adds latency but enables longer contexts

**Compression:** Quantize KV cache (FP16 → INT8)
- 2x memory reduction
- Minimal quality impact

**6. Hardware Selection**

**Compute-Bound Workloads (Prefill):**
- Prioritize FP16/BF16 TFLOPS
- H100 > A100 > L40S

**Memory-Bound Workloads (Decode):**
- Prioritize memory bandwidth
- H100 (3.35 TB/s) > A100 (2 TB/s) > MI300X (5.3 TB/s!)

**7. Model Parallelism**

For models too large for single GPU:
- **Tensor Parallelism:** Split model weights across GPUs (requires fast interconnect)
- **Pipeline Parallelism:** Different layers on different GPUs (higher latency)
- **Hybrid:** Combine both strategies

From [Why vLLM is the best choice for AI inference today](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today) (accessed 2025-02-02)

### Latency SLAs

**Service Level Objectives:**
```
Interactive chat:  p95 TTFT < 500ms, p95 total < 5s
Document Q&A:      p95 TTFT < 1s, p95 total < 10s
Batch processing:  p95 < 60s, throughput > cost
```

**Managing SLAs:**
- Route requests by priority
- Shed load during overload
- Pre-warm capacity before peaks
- Use multiple model sizes (cascade)

---

## Section 5: Batch Processing (~60 lines)

### Batch vs Real-Time Trade-offs

**Real-Time Inference:**
- Low latency required (milliseconds to seconds)
- Higher cost per request
- Smaller batch sizes
- Continuous GPU availability

**Batch Inference:**
- Latency tolerant (minutes to hours acceptable)
- 50% lower cost per request
- Larger batch sizes (better GPU utilization)
- Can use spot instances

From [Batch Processing for LLM Cost Savings](https://www.prompts.ai/en/blog/batch-processing-for-llm-cost-savings) (accessed 2025-02-02)

### Batch Processing Patterns

**1. Scheduled Batch Jobs**

Run at fixed intervals:
```python
# Process overnight batch
schedule: "0 2 * * *"  # 2 AM daily
batch_size: 10000
priority: low
max_duration: 6h
```

**Use Cases:**
- Document summarization pipelines
- Content moderation queues
- Embedding generation for search indices

**2. Queue-Based Processing**

Accumulate requests, process when queue fills:
```python
# Ray Data batch processing
ray.data.read_parquet("requests.parquet") \
    .map_batches(vllm_inference, batch_size=32) \
    .write_parquet("results.parquet")
```

**Benefits:**
- Dynamic batch sizing
- Fault tolerance (retry failed batches)
- Progress tracking

From [Scaling LLM Batch Inference: Ray Data & vLLM](https://www.youtube.com/watch?v=_rEsLo21WvE) (accessed 2025-02-02)

**3. Continuous Batch Processing**

Hybrid approach for moderate-latency workloads:
- Accept requests continuously
- Process in micro-batches (every 100ms)
- Return results as soon as ready

**Balance:** Lower latency than pure batch, better efficiency than pure real-time.

### Implementation Best Practices

**vLLM Batch Serving:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-70b")
prompts = [...]  # 1000s of prompts

# Automatic batching
outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0.7, max_tokens=512)
)
```

**Databricks Mosaic AI:**
```python
# Batch endpoint
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict_batch(
    endpoint="llama-3-70b-batch",
    inputs={"dataframe_records": df.to_dict("records")}
)
```

From [Batch LLM Inference on Mosaic AI Model Serving](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving) (accessed 2025-02-02)

**Monitoring Batch Jobs:**
- Track progress (% complete)
- Estimate completion time
- Monitor cost accumulation
- Detect and retry failures

---

## Section 6: Production Case Studies (~40 lines)

### Case Study 1: Financial Services Document Processing

**Challenge:** Extract structured data from millions of financial documents (invoices, reports, contracts).

**Solution:**
- Qwen-2.5-VL for vision-language understanding
- vLLM for high-throughput serving
- Disaggregated prefill/decode workers

**Architecture:**
```
Document Queue → Prefill Workers (H100, 8 GPU)
               → KV Cache Transfer
               → Decode Workers (A100, 16 GPU)
               → Structured JSON Output
```

**Results:**
- 10M+ documents processed monthly
- $0.02 per document (vs $0.15 for GPT-4V API)
- 99.9% extraction accuracy
- <30s per document (batch mode)

From [Deploy an in-house Vision Language Model](https://pub.towardsai.net/deploy-an-in-house-vision-language-model-to-parse-millions-of-documents-say-goodbye-to-gemini-and-cdac6f77aff5) (accessed 2025-02-02)

### Case Study 2: Healthcare RAG System

**Challenge:** HIPAA-compliant medical question answering with multimodal inputs (text + medical images).

**Solution:**
- LLaVA-based VLM for medical Q&A
- On-premises deployment (data sovereignty)
- OpenTelemetry tracing for compliance auditing

**Infrastructure:**
- 4x A100 nodes (tensor parallelism)
- Prometheus + Grafana monitoring
- Jaeger for request tracing

**Outcomes:**
- 300+ concurrent users supported
- <2s response time (p95)
- Full audit trail for regulatory compliance
- Zero PHI exposure (all on-prem)

From [JAVIS Chat: A Seamless Open-Source Multi-LLM/VLM Framework](https://www.mdpi.com/2076-3417/15/4/1796) (accessed 2025-02-02)

### Case Study 3: Automotive Edge AI

**Challenge:** Real-time vision-language processing in autonomous vehicles (edge deployment).

**Solution:**
- Liquid AI hardware-optimized VLM
- 50% model size reduction via quantization
- 10x faster inference on existing CPUs

**Deployment:**
- Edge SDK for on-vehicle inference
- INT4 quantization
- CPU-only (no GPU requirement)

**Impact:**
- Real-time object understanding (<100ms)
- 90% cost reduction (no GPU needed)
- Runs on standard automotive compute

From [Accelerating Vision-Language Model Deployment for Automotive AI](https://www.liquid.ai/use-cases/accelerating-vision-language-model-deployment-for-automotive-ai) (accessed 2025-02-02)

### Common Success Patterns

**1. Hybrid Deployment:**
- Real-time tier: Low-latency requests on expensive GPUs
- Batch tier: High-volume processing on spot instances
- Edge tier: Local inference for privacy/latency

**2. Progressive Optimization:**
- Start: Single-node vLLM deployment
- Scale: Multi-node with llm-d orchestration
- Optimize: Disaggregated workers, caching, quantization

**3. Cost-Performance Balance:**
- Use smaller models where acceptable (cascade)
- Cache aggressively
- Batch non-critical workloads
- Right-size hardware per workload type

---

## Sources

**Serving Infrastructure:**
- [vLLM Production Stack Documentation](https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.2/dev_guide/peripheral/models.html) - Observability models and deployment
- [Why vLLM is the best choice for AI inference today](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today) - Red Hat Developer (accessed 2025-02-02)
- [vLLM vs TGI comparison](https://modal.com/blog/vllm-vs-tgi-article) - Modal (accessed 2025-02-02)

**Monitoring & Observability:**
- [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) - IBM Data Science in Practice (accessed 2025-02-02)

**Cost Optimization:**
- [Batch Processing for LLM Cost Savings](https://www.prompts.ai/en/blog/batch-processing-for-llm-cost-savings) - Prompts.ai (accessed 2025-02-02)
- [Scale AI: Reducing Cold Start Time for LLM Inference](https://scale.com/blog/reduce-cold-start-time-llm-inference) - Scale AI (accessed 2025-02-02)

**Performance:**
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - NVIDIA Developer (accessed 2025-02-02)
- [Scaling LLM Batch Inference: Ray Data & vLLM](https://www.youtube.com/watch?v=_rEsLo21WvE) - InfoQ video (accessed 2025-02-02)

**Batch Processing:**
- [Batch LLM Inference on Mosaic AI Model Serving](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving) - Databricks (accessed 2025-02-02)

**Case Studies:**
- [Deploy an in-house Vision Language Model](https://pub.towardsai.net/deploy-an-in-house-vision-language-model-to-parse-millions-of-documents-say-goodbye-to-gemini-and-cdac6f77aff5) - Towards AI (accessed 2025-02-02)
- [JAVIS Chat: A Seamless Open-Source Multi-LLM/VLM Framework](https://www.mdpi.com/2076-3417/15/4/1796) - MDPI (accessed 2025-02-02)
- [Accelerating Vision-Language Model Deployment for Automotive AI](https://www.liquid.ai/use-cases/accelerating-vision-language-model-deployment-for-automotive-ai) - Liquid AI (accessed 2025-02-02)

**Additional Resources:**
- [GitHub: vllm-project/vllm](https://github.com/vllm-project/vllm) - vLLM open source repository
- [GitHub: vllm-project/production-stack](https://github.com/vllm-project/production-stack) - Production deployment tools
- [vLLM Blog: 2024 Wrapped & 2025 Vision](https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html)
