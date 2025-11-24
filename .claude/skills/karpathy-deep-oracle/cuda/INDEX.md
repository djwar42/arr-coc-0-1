# CUDA & PyTorch - Index

**CUDA programming and PyTorch GPU optimization**

**Total Files**: 37 knowledge files
**Coverage**: CUDA fundamentals, streams, memory, Tensor Cores, torch.compile, profiling
**Focus**: Deep learning GPU optimization for VLMs and LLMs

---

## Overview

This folder contains comprehensive CUDA and PyTorch GPU knowledge:
- CUDA programming fundamentals
- Memory hierarchy optimization
- Tensor Core utilization
- PyTorch torch.compile and autograd
- Profiling and debugging
- Multi-GPU patterns

---

## Topics Covered

### CUDA Fundamentals
- Kernel programming
- Thread hierarchy (grids, blocks, threads)
- Memory types (global, shared, registers)
- Synchronization

### Memory Optimization
- Memory hierarchy (HBM, L2, L1, registers)
- Coalesced memory access
- Shared memory usage
- Memory bandwidth optimization

### Tensor Core Acceleration
- FP16, BF16, TF32, FP8 formats
- WMMA (Warp Matrix Multiply Accumulate)
- Mixed precision training
- Performance optimization

### PyTorch Integration
- torch.compile and TorchDynamo
- Custom CUDA kernels
- Autograd and gradient computation
- DataLoader optimization

### Profiling & Debugging
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- PyTorch Profiler
- Memory profiling
- Bottleneck identification

### Multi-GPU Patterns
- DistributedDataParallel (DDP)
- FSDP (Fully Sharded Data Parallel)
- Pipeline parallelism
- NCCL communication

---

## Cross-References

**Related Folders**:
- `karpathy/llm-gpu-integration/` - FlashAttention, LLM-specific
- `karpathy/distributed-training/` - Multi-GPU training
- `implementations/` - GPU hardware implementations
- `karpathy/practical-implementation/` - Real-world training guides

**GCP Integration**:
- `gcp-gpu/` - GCP GPU quotas and management
- `gcp-vertex/` - Vertex AI GPU training

---

## Quick Navigation

**Getting started with CUDA:**
→ Start with fundamental kernel programming files

**Optimizing PyTorch:**
→ Focus on torch.compile and memory optimization files

**Multi-GPU training:**
→ See DDP/FSDP files + `karpathy/distributed-training/`

**Profiling:**
→ Check Nsight Systems/Compute files

---

**Last Updated**: 2025-11-21
