# KNOWLEDGE DROP: Azure ML Integration

**Timestamp**: 2025-11-14 03:00
**Runner**: Part 2 of 4
**Target**: karpathy/azure-ml/00-distributed-training-aks-serving.md

---

## What Was Created

**File**: `karpathy/azure-ml/00-distributed-training-aks-serving.md` (717 lines)

Complete technical guide covering:

1. **Azure ML Distributed Training** (~200 lines)
   - PyTorch DDP with automatic environment variable setup
   - DeepSpeed ZeRO integration with autotuning
   - TensorFlow distributed strategies
   - InfiniBand acceleration on NC/ND-series VMs

2. **AKS GPU Scheduling** (~200 lines)
   - GPU node pool creation and management
   - NVIDIA device plugin deployment
   - GPU observability with DCGM metrics
   - Cost optimization with spot instances and autoscaling

3. **Triton Inference Server** (~180 lines)
   - No-code deployment to Azure ML managed endpoints
   - Model ensemble configuration
   - Dynamic batching for throughput
   - Triton client inference patterns

4. **Cost Optimization** (~120 lines)
   - Spot instance strategies (70% savings)
   - Savings plans and reserved instances
   - Multi-cloud cost comparison (Azure vs AWS vs GCP)

5. **ARR-COC-VIS Deployment** (~100 lines)
   - Distributed training configuration
   - Triton ensemble for VLM pipeline
   - Cost analysis for training and inference

---

## Key Technical Findings

### Azure ML Distributed Training

**Automatic Environment Variables**:
- Azure ML sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `NODE_RANK`
- No launcher utility needed (no `torch.distributed.launch`)
- Clean integration with PyTorch DDP via `env://` init_method

**DeepSpeed Integration**:
- Curated environments include DeepSpeed pre-configured
- Autotuning finds optimal `ds_config.json` settings
- ZeRO-3 + CPU offload enables training beyond GPU memory

**InfiniBand VMs**:
- VM SKUs with "r" suffix have InfiniBand (e.g., NC24**rs**_v3)
- 200 Gb/s InfiniBand vs 25 GB/s Ethernet = ~8× faster
- Critical for multi-node training scalability

### AKS GPU Architecture

**NVIDIA Device Plugin Required**:
- Manually deploy DaemonSet to make GPUs schedulable
- Taints (`sku=gpu:NoSchedule`) prevent non-GPU workloads
- DCGM Exporter provides Prometheus metrics

**Cost Optimization Patterns**:
- Spot instances: 70% savings vs on-demand
- Cluster autoscaler can scale-to-zero
- T4 GPUs ($0.526/hr) more cost-effective for inference than V100/A100

### Triton on Azure ML

**No-Code Deployment**:
- Set `type: triton_model` in model YAML
- No scoring script or custom environment needed
- Azure ML handles Triton server setup automatically

**Dynamic Batching Impact**:
- Wait up to `max_queue_delay_microseconds` to form batches
- Can achieve 5-8× GPU utilization improvement
- Configurable via `config.pbtxt` per model

**Model Ensembles**:
- Server-side multi-stage pipelines
- No client-side orchestration needed
- Optimized data transfer between models

---

## Integration with Existing Knowledge

### Cross-References Created

**From existing knowledge**:
- [distributed-training/02-megatron-lm-tensor-parallelism.md](../../karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md) - Tensor parallelism concepts
- [inference-optimization/02-triton-inference-server.md](../../karpathy/inference-optimization/02-triton-inference-server.md) - Triton fundamentals
- [orchestration/00-kubernetes-gpu-scheduling.md](../../karpathy/orchestration/00-kubernetes-gpu-scheduling.md) - Kubernetes GPU concepts
- [orchestration/01-kubeflow-ml-pipelines.md](../../karpathy/orchestration/01-kubeflow-ml-pipelines.md) - Kubeflow patterns

**New Azure-specific knowledge**:
- Azure ML SDK v2 patterns for distributed training
- AKS-specific GPU node pool configuration
- Azure ML managed endpoints for Triton (vs self-hosted)
- Cost optimization strategies specific to Azure

---

## Web Research Summary

**Total sources cited**: 15+ web sources + 4 existing knowledge files

**Key sources used**:

1. **Microsoft Learn Documentation**:
   - Distributed GPU training guide
   - AKS GPU usage guide
   - Triton deployment guide
   - PyTorch training guide
   - GPU observability guide

2. **Community Resources**:
   - Medium: Data Science at Microsoft - PyTorch DDP scaling article
   - Ubuntu blog - Kubeflow on Azure deployment
   - Azure GitHub repos - Official examples

3. **Technical References**:
   - Triton model repository structure
   - NVIDIA DCGM metrics
   - Azure VM pricing and SKU details

---

## Completeness Check

✓ **Section 1**: Azure ML distributed training (PyTorch, TensorFlow, DeepSpeed, InfiniBand)
✓ **Section 2**: AKS GPU scheduling (node pools, device plugin, monitoring, cost)
✓ **Section 3**: Triton managed endpoints (no-code, ensembles, dynamic batching)
✓ **Section 4**: Cost optimization (spot instances, savings plans, multi-cloud comparison)
✓ **Section 5**: ARR-COC-VIS deployment examples
✓ **All sources cited** with access dates
✓ **Cross-references** to 4 existing knowledge files
✓ **717 lines** of comprehensive technical content

---

## Next Steps for Oracle

**This PART is complete**. The oracle should:

1. ✓ Verify file creation: `karpathy/azure-ml/00-distributed-training-aks-serving.md`
2. ✓ Confirm 717 lines of content
3. ✓ Check all citations include access dates
4. Update INDEX.md to include new azure-ml/ folder
5. Continue with remaining PARTs (1, 3, 4)

---

**KNOWLEDGE DROP END**
