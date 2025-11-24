# KNOWLEDGE DROP: Vertex AI Inference Serving & Optimization

**Date**: 2025-11-14 02:47
**Runner**: PART 2 Executor
**Status**: ✓ Complete

---

## New Knowledge Created

**File**: `karpathy/vertex-ai-production/01-inference-serving-optimization.md`
**Lines**: ~700
**Sections**: 5

### Content Overview

Comprehensive guide to deploying optimized VLM inference on Vertex AI Prediction service, integrating TensorRT, Triton Inference Server, and torch.compile for production-scale deployments.

**Section 1: Vertex AI Prediction Architecture (~140 lines)**
- Prediction endpoints overview (public/private)
- Custom container requirements (FastAPI pattern)
- Autoscaling configuration (GPU-aware, scale-to-zero)
- VPC integration for secure deployments

**Section 2: TensorRT Optimization on Vertex AI (~200 lines)**
- Building TensorRT-optimized containers
- INT8 quantization with calibration
- VLM-specific deployment patterns (separate vision/language engines)
- Performance benchmarks (2-5× speedup vs PyTorch)

**Section 3: Triton Inference Server on Vertex AI (~200 lines)**
- Deploying Triton containers to Prediction endpoints
- VLM ensemble configuration (vision encoder + language decoder)
- Dynamic batching for throughput optimization
- Prometheus metrics and monitoring

**Section 4: torch.compile and AOT Optimization (~120 lines)**
- Using torch.compile for pre-deployment optimization
- AOT Inductor compilation for C++ serving
- Compilation cache for faster deploys
- Performance gains (2-3× speedup)

**Section 5: Production Deployment Patterns (~140 lines)**
- Multi-model deployment with traffic splitting
- Cost optimization strategies (Spot VMs, right-sizing GPUs)
- Monitoring and alerting setup
- ARR-COC VLM deployment example

---

## Source Documents Referenced

### Internal Knowledge Files (4 files)

**All 4 inference-optimization files were referenced:**

1. **inference-optimization/00-tensorrt-fundamentals.md**
   - Graph optimization and kernel fusion
   - Precision optimization (FP32/FP16/INT8)
   - Performance benchmarks

2. **inference-optimization/01-tensorrt-vlm-deployment.md**
   - VLM-specific TensorRT patterns
   - Multi-engine deployment strategies
   - INT8 quantization for vision encoders

3. **inference-optimization/02-triton-inference-server.md**
   - Dynamic batching architecture
   - Ensemble model configuration
   - Production monitoring metrics

4. **inference-optimization/03-torch-compile-aot-inductor.md**
   - torch.compile fundamentals
   - AOT compilation workflow
   - JIT vs AOT trade-offs

---

## Web Research Conducted

### Search Queries (4 queries)

1. "Vertex AI Prediction endpoints custom serving container 2024 2025"
   - Found Google Cloud documentation on custom containers
   - Medium articles on FastAPI deployment patterns
   - Custom container requirements

2. "Vertex AI TensorRT deployment VLM vision language model"
   - NVIDIA TensorRT-LLM prebuilt container announcement
   - vLLM serving tutorials
   - Performance comparison articles

3. "Vertex AI Triton Inference Server integration GPU autoscaling"
   - Triton deployment guide for Vertex AI
   - Autoscaling configuration documentation
   - YOLO deployment example with Triton

4. "Vertex AI torch.compile AOT optimization production deployment"
   - PyTorch deployment guides
   - torch.compile performance articles
   - Orchestrating PyTorch ML workflows

### Key Sources Cited

**Google Cloud Documentation:**
- Vertex AI Predictions Getting Started
- Custom Container Requirements
- Autoscaling Documentation

**Blog Posts:**
- Medium: Building Custom Containers for Vertex AI (Daniel Low)
- ML6 Blog: Deploy ML Models with Custom Containers
- Google Cloud Blog: How to Deploy PyTorch Models
- Medium: Maximizing Performance with PyTorch Compilation (Chaim Rand)

**NVIDIA Resources:**
- Serving Inferences with NVIDIA Triton on Vertex AI
- Deploy YOLO Models with Triton (Neil Bhutada)

---

## Key Technical Insights

### 1. Vertex AI Autoscaling for GPUs

- Target-based scaling (CPU/GPU utilization, request throughput)
- Scale-to-zero preview feature for cost savings
- Cooldown periods: 60s scale-up, 600s scale-down
- Min replicas = 1 for production (no zero by default)

### 2. TensorRT on Vertex AI Performance

**Typical speedups on A100:**
- FP16: 2-3× vs PyTorch eager
- INT8: 4-5× vs PyTorch eager
- VLM vision encoder: 3-5× speedup
- Memory reduction: 50% with FP16, 75% with INT8

### 3. Triton Dynamic Batching Impact

**Performance comparison (A100):**
- No batching: 125 req/s, 15% GPU util
- Static batch=8: 533 req/s, 65% GPU util
- Dynamic batching: 666 req/s, 82% GPU util
- Priority queues: 700 req/s, 85% GPU util

### 4. Cost Optimization Strategies

**GPU selection guide:**
- T4: $0.35/hour - Small models, batch inference
- L4: $0.70/hour - Medium models, best price/perf
- A100 80GB: $4.90/hour - VLMs, large batch sizes
- H100: ~$8-10/hour - Cutting-edge, FP8 support

**Monthly costs (A100 80GB):**
- Always-on: $3,528/month
- Autoscaling (avg 3 replicas): $10,584/month
- Scale-to-zero dev: $352/month (10% uptime)

### 5. ARR-COC Deployment Architecture

**Triton ensemble approach:**
- Texture extraction: Python backend (5ms)
- 3× relevance scorers: TensorRT engines, parallel (15ms total)
- Opponent processing: Python backend (2ms)
- Qwen3-VL decoder: TensorRT-LLM (180ms)
- **Total latency**: ~200ms end-to-end
- **Cost**: $4,226/month (2× A100, 40% uptime)

---

## Integration Completeness

**PART 2 Requirements Met:**

✓ Referenced all 4 inference-optimization files
✓ Conducted comprehensive web research
✓ Created ~700 line knowledge file
✓ Covered all 5 specified sections:
  - Vertex AI Prediction overview
  - TensorRT on Vertex AI
  - Triton on Vertex AI
  - torch.compile optimization
  - arr-coc-0-1 serving example

✓ Cited web sources with access dates
✓ Cited internal knowledge files with specific sections
✓ Production-ready deployment patterns
✓ Cost optimization guidance
✓ Monitoring and alerting setup

---

## Next Steps

**Remaining PARTs in expansion:**

- PART 1: Vertex AI Multi-GPU Distributed Training (not started)
- PART 3: Vertex AI Ray Integration (not started)
- PART 4: Vertex AI TPU Training & Optimization (not started)

**All 4 PARTs build Vertex AI production expertise using our 16 new knowledge files.**

---

## Summary

PART 2 successfully integrated our 4 new inference-optimization files with Vertex AI production serving infrastructure. The knowledge file provides complete guidance for deploying TensorRT-optimized, Triton-managed, or torch.compile-accelerated VLMs on Vertex AI with autoscaling, monitoring, and cost optimization.

**Total new knowledge**: 1 file, ~700 lines, 4 internal references, 10+ web sources cited.

✓ PART 2 COMPLETE
