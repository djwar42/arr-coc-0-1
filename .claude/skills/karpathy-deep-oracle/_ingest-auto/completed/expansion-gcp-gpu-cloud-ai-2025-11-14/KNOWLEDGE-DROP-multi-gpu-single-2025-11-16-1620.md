# KNOWLEDGE DROP: Multi-GPU Single-Node Training Patterns

**Runner**: PART 5
**File Created**: `gcp-gpu/04-multi-gpu-training-patterns.md`
**Lines**: ~720 lines
**Completed**: 2025-11-16 16:20

## Summary

Comprehensive guide to single-node multi-GPU training on GCP covering PyTorch DDP, NCCL optimization, NVLink topology, NUMA affinity, and arr-coc-0-1 specific configuration. Includes practical code examples, performance benchmarks, and GCP instance recommendations.

## Key Topics Covered

### 1. PyTorch DistributedDataParallel (DDP) Fundamentals
- DDP vs DataParallel comparison (process model, communication, scalability)
- Gradient synchronization with AllReduce overlapping
- Process group initialization with NCCL backend
- Basic DDP setup with `mp.spawn` and `DistributedSampler`

### 2. NCCL Optimization for Single-Node Training
- NCCL environment variables (`NCCL_IB_DISABLE`, `NCCL_P2P_LEVEL=NVL`)
- GCP-specific configuration for A100/H100 instances
- AllReduce bandwidth benchmarking (expected: 75,000 MB/s on NVLink)
- Performance monitoring and topology detection

### 3. GPU Topology and NVLink Architecture
- A100 8-GPU NVSwitch topology (600 GB/s bidirectional per GPU)
- `nvidia-smi topo -m` analysis (NV12 = full bandwidth connections)
- CPU-GPU NUMA affinity for optimal memory access
- GCP Compact Placement Policies for multi-node setups

### 4. Efficient Data Loading for Multi-GPU
- `DistributedSampler` configuration and `set_epoch()` requirement
- DataLoader optimization (`num_workers`, `pin_memory`, `prefetch_factor`)
- Effective batch size scaling and learning rate adjustment
- Linear warmup for large batch training

### 5. Gradient Accumulation and Mixed Precision
- Gradient accumulation for simulating larger batches
- PyTorch AMP (Automatic Mixed Precision) with GradScaler
- BF16 vs FP16 comparison (recommend BF16 on A100/H100)
- Combined AMP + gradient accumulation pattern

### 6. Monitoring and Profiling Multi-GPU Training
- `nvidia-smi` dashboard and utilization monitoring
- NCCL communication profiling with `NCCL_DEBUG=INFO`
- PyTorch Profiler integration with TensorBoard
- Expected performance metrics (GPU util >90%, memory 70-95%)

### 7. Checkpointing and Model Saving in DDP
- Save only from rank 0 to avoid corruption
- Extract model from DDP wrapper (`.module.state_dict()`)
- Collective calls warning (avoid inside rank-specific code)
- Proper checkpoint loading across all ranks

### 8. GCP-Specific Configuration for arr-coc-0-1
- Instance recommendations (a2-ultragpu-8g: 8×A100 80GB)
- Startup script with NCCL configuration
- Optimized hyperparameters (batch size, lr scaling, warmup)
- Complete training script example

## Knowledge Synthesis

**From existing CUDA knowledge** (`cuda/00-streams-concurrency-async.md`):
- GPU concurrency fundamentals applied to multi-GPU synchronization
- Understanding of asynchronous operations underlying DDP gradient overlap

**From web research:**
- PyTorch official DDP tutorial provided canonical implementation patterns
- Cerfacs blog detailed NCCL environment variables and profiling techniques
- NVIDIA A100 whitepaper confirmed NVLink bandwidth specifications (600 GB/s)
- NUMA affinity research showed 10-30% performance impact when misconfigured

**GCP Integration:**
- Mapped GCP instance types (a2-highgpu, a2-ultragpu, a3-highgpu) to training scenarios
- Provided startup scripts for automated driver/CUDA installation
- Configured NCCL for GCP's network topology (localhost for single-node)

## Practical Recommendations for arr-coc-0-1

**Recommended Setup:**
- Instance: `a2-ultragpu-8g` (8×A100 80GB) for vision-language models
- Effective batch size: 128 (16 per GPU × 8 GPUs)
- Learning rate: 8e-4 (1e-4 base × 8 GPU scaling)
- Mixed precision: BF16 (no GradScaler needed on A100)
- Gradient accumulation: 4 steps (512 effective batch size)

**Expected Performance:**
- Training throughput: 7.5-7.8× speedup vs single-GPU (near-linear scaling)
- GPU utilization: >90% with optimized data loading
- Communication overhead: <5% with NVLink (vs 20-30% over PCIe)

## Citations Quality

**Strong citations maintained:**
- Direct quotes from PyTorch documentation with access dates
- NVIDIA A100 whitepaper specifications for NVLink bandwidth
- Cerfacs blog for NCCL optimization patterns
- Massed Compute for NUMA affinity performance impact

**All web links preserved** with access dates (2025-11-16).

## Integration with Existing Knowledge

**Complements:**
- `gcp-gpu/02-nvidia-driver-cuda-management.md` (driver installation)
- `gcp-gpu/03-storage-optimization-gpu-training.md` (data loading from GCS/Local SSD)

**Prepares for:**
- Next section (PART 6): Multi-node distributed training patterns
- PART 7: Preemptible/Spot GPU strategies with checkpointing
- PART 8: Network optimization for multi-GPU communication

## Potential Improvements

**Could expand in future:**
- PyTorch Lightning integration for simplified multi-GPU training
- DeepSpeed ZeRO optimizer comparison (will be covered in separate distributed-training/ section)
- Horovod as alternative to DDP (mentioned but not detailed)
- Pipeline parallelism for very large models (covered in multi-node section)

**File complete and ready for integration into INDEX.md by oracle.**
