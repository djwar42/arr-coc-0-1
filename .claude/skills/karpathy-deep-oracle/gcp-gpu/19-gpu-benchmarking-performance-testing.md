# GPU Benchmarking & Performance Testing: Comprehensive Guide

## Overview

GPU benchmarking and performance testing validates training and inference performance, identifies bottlenecks, and ensures optimal hardware utilization. This guide covers MLPerf standards, NCCL communication tests, NVIDIA profiling tools (Nsight Systems, Nsight Compute), synthetic vs real workload benchmarks, and practical testing methodologies for arr-coc-0-1.

**Why Benchmarking Matters:**
- **Performance Validation**: Confirm GPU meets expected throughput/latency
- **Bottleneck Identification**: Find memory, compute, or communication limits
- **Cost Optimization**: Right-size GPU types (T4 vs L4 vs A100 vs H100)
- **Regression Testing**: Ensure code changes don't degrade performance
- **Hardware Acceptance**: Validate cloud provider GPU performance claims

**Key Benchmark Categories:**
1. **MLPerf** - Industry-standard ML training/inference benchmarks
2. **NCCL Tests** - Multi-GPU communication bandwidth/latency
3. **NVIDIA Nsight** - GPU profiling (kernel-level optimization)
4. **Synthetic Workloads** - Controlled GEMM/convolution/attention tests
5. **Real Workloads** - Actual model training/inference (ResNet, BERT, GPT)

From [NVIDIA Developer Blog: Understanding NCCL Tuning](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) (NVIDIA, accessed 2025-11-16):
> "NCCL default CTA counts and buffer sizes are carefully selected to maximize end-to-end workload performance. Increasing these values will result in better communication benchmark performance."

---

## Section 1: MLPerf Benchmarks (Training & Inference Standards)

### What is MLPerf?

**MLPerf** is an industry-standard benchmark suite from MLCommons measuring ML system performance across training and inference workloads.

**MLPerf Training Benchmarks:**
- **ResNet-50** (image classification, ImageNet)
- **BERT** (NLP, SQuAD dataset)
- **GPT-3** (LLM pretraining) - Replaced with **Llama 3.1 405B** in MLPerf Training 5.0 (2025)
- **DLRM** (recommendation systems)
- **3D U-Net** (medical image segmentation)

**MLPerf Inference Benchmarks:**
- **Datacenter** (high-throughput inference servers)
- **Edge** (low-latency edge devices)
- **Client** (consumer devices - PCs, laptops)

From [Data Center Knowledge: How MLPerf Benchmarks Guide Data Center Decisions](https://www.datacenterknowledge.com/ai-data-centers/how-mlperf-benchmarks-guide-data-center-design-decisions) (Data Center Knowledge, accessed 2025-10-22):
> "MLPerf Training 5.0 (2025) replaced the GPT-3 benchmark with a new LLM pretraining evaluation based on the Llama 3.1 405B generative AI system."

### MLPerf Training Metrics

**Convergence Time:**
- Time to train model to target quality metric
- Example: ResNet-50 to 75.9% ImageNet top-1 accuracy
- Lower time = faster convergence

**Samples per Second:**
- Training throughput (images/sec, tokens/sec)
- Measures data processing speed

**Hardware Efficiency:**
- FLOPs utilization (% of theoretical peak)
- Memory bandwidth utilization

### Running MLPerf Training Benchmarks

**Installation:**
```bash
# Clone MLPerf Training reference implementations
git clone https://github.com/mlcommons/training.git
cd training

# Install dependencies (PyTorch, CUDA, cuDNN)
pip install -r requirements.txt

# Download datasets (ImageNet, SQuAD, etc.)
# Follow dataset preparation instructions in each benchmark folder
```

**Example: ResNet-50 Training Benchmark:**
```bash
# Single-node 8×A100 GPU training
cd image_classification

# Run benchmark
python main.py \
  --data-path /data/imagenet \
  --model resnet50 \
  --batch-size 256 \
  --epochs 90 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --workers 32 \
  --device cuda \
  --distributed \
  --world-size 8
```

**Output Metrics:**
```
Training time: 42.3 minutes
Samples/sec: 12,450
Top-1 accuracy: 76.1%
GPU utilization: 94%
```

### MLPerf Inference Benchmarks

**Metrics:**
- **Latency** (p50, p90, p99 milliseconds)
- **Throughput** (queries per second, tokens per second)
- **Accuracy** (validation against quality target)

**Scenarios:**
1. **Server** - High throughput, multi-stream queries
2. **Single Stream** - Low latency, one query at a time
3. **Multi-Stream** - Fixed query rate
4. **Offline** - Batch processing (no latency constraint)

**Example: BERT Inference Benchmark:**
```bash
cd language/bert

# Run inference benchmark (TensorRT backend)
python run_inference.py \
  --backend tensorrt \
  --model bert-large-uncased \
  --scenario Server \
  --qps 1000 \
  --max-batchsize 128
```

**Output:**
```
Latency (p50): 12.3 ms
Latency (p99): 18.7 ms
Throughput: 983 QPS
Accuracy: 90.4% F1 (SQuAD)
```

From [MLCommons MLPerf Benchmark Page](https://mlcommons.org/benchmarks/training/) (MLCommons, accessed 2025-11-16):
> "The MLPerf Benchmark Suites measures how fast machine learning systems can train models to a target quality metric using v2.0 results."

### MLPerf Results Analysis

**NVIDIA H100 vs A100 (MLPerf Training 4.0):**
- **ResNet-50**: H100 2.5× faster than A100
- **BERT**: H100 2.8× faster than A100
- **GPT-3 175B**: H100 3.2× faster than A100 (multi-node)

**Cost-Performance Analysis:**
```python
# Calculate cost per training run
a100_time_hours = 0.7  # 42 minutes
h100_time_hours = 0.28  # 17 minutes

a100_cost_per_hour = 3.67  # $/hour (GCP on-demand)
h100_cost_per_hour = 5.50  # $/hour (estimate)

a100_total_cost = a100_time_hours * a100_cost_per_hour  # $2.57
h100_total_cost = h100_time_hours * h100_cost_per_hour  # $1.54

# H100 is 40% cheaper per training run despite higher hourly cost
```

From [HPCwire: NVIDIA Showcases Blackwell Ultra Performance on MLPerf Benchmark](https://www.hpcwire.com/aiwire/2025/11/14/nvidia-showcases-blackwell-ultra-performance-on-mlperf-benchmark/) (HPCwire, accessed 2025-11-14):
> "MLCommons released the latest MLPerf benchmark results this week, and you will not be surprised to learn that Nvidia GPU-equipped systems dominated the leaderboard."

### arr-coc-0-1 MLPerf Benchmark Suite

**Training Benchmark:**
```bash
# Benchmark arr-coc relevance realization training
cd arr-coc-0-1

# Run synthetic patch selection benchmark (similar to DLRM)
python training/benchmark_relevance.py \
  --batch-size 64 \
  --num-patches 196 \
  --k-patches 200 \
  --lod-range 64-400 \
  --gpus 8 \
  --iterations 1000
```

**Expected Metrics:**
- Samples/sec: ~450 (8×A100, batch 64)
- Patch selection time: 2.3 ms/batch
- LOD allocation time: 0.8 ms/batch
- End-to-end latency: 15 ms/sample

**Inference Benchmark:**
```bash
# Test inference throughput
python training/benchmark_inference.py \
  --model arr-coc-qwen3-vl \
  --batch-sizes 1,4,8,16 \
  --input-size 224 \
  --iterations 500
```

**Target Metrics:**
- Batch 1: 45 QPS, p99 latency 28 ms
- Batch 8: 180 QPS, p99 latency 52 ms
- GPU utilization > 90%

---

## Section 2: NCCL Performance Tests (AllReduce Bandwidth)

### NCCL Communication Benchmarks

**NCCL (NVIDIA Collective Communications Library)** provides optimized multi-GPU and multi-node communication primitives (AllReduce, Broadcast, Reduce, etc.).

**Key Operations:**
- **AllReduce** - Sum gradients across all GPUs (most critical for training)
- **AllGather** - Gather tensors from all GPUs
- **ReduceScatter** - Reduce and distribute results
- **Broadcast** - Send tensor from one GPU to all others

From [Together AI: A Practitioner's Guide to Testing Large GPU Clusters](https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models) (Together AI, accessed 2025-08-13):
> "Generally we are looking for the all_reduce_perf test to show bandwidth around 92% of the theoretical maximum of the fabric: so around 370 GB/s on a 400GB/s fabric."

### Installing NCCL Tests

```bash
# Clone NCCL tests repository
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# Build with MPI support (for multi-node)
make MPI=1 CUDA_HOME=/usr/local/cuda

# Executables created in build/
ls build/
# all_reduce_perf  all_gather_perf  broadcast_perf  reduce_scatter_perf
```

### Single-Node AllReduce Benchmark

**Test 8×A100 NVLink bandwidth:**
```bash
# Run AllReduce performance test
./build/all_reduce_perf \
  -b 8 \
  -e 4G \
  -f 2 \
  -g 8

# -b 8: Start at 8 bytes
# -e 4G: End at 4 GB message size
# -f 2: Multiply size by 2 each iteration
# -g 8: Use 8 GPUs
```

**Expected Output:**
```
# nThread 1 nGpus 8 minBytes 8 maxBytes 4294967296 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12345 on node1 device  0 [0x17] NVIDIA A100-SXM4-80GB
#  Rank  1 Group  0 Pid  12345 on node1 device  1 [0x1a] NVIDIA A100-SXM4-80GB
#  ...
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    32.45    0.00    0.00      0    31.89    0.00    0.00      0
          16             4     float     sum      -1    32.12    0.00    0.00      0    31.67    0.00    0.00      0
          32             8     float     sum      -1    32.34    0.00    0.00      0    31.98    0.00    0.00      0
        1024           256     float     sum      -1    33.21    0.03    0.06      0    32.76    0.03    0.06      0
      131072         32768     float     sum      -1    45.67    2.87    5.02      0    44.89    2.92    5.10      0
     4194304       1048576     float     sum      -1    89.34   46.95   82.16      0    87.12   48.15   84.26      0
   134217728      33554432     float     sum      -1   421.23  318.67  557.42      0   415.89  322.74  564.79      0
  4294967296    1073741824     float     sum      -1  7234.12  593.78 1039.12      0  7198.45  596.71 1044.25      0

# algbw: Algorithm bandwidth (actual data transferred)
# busbw: Bus bandwidth (physical bandwidth used, includes NVLink overhead)
```

**Analysis:**
- **4 GB message size**: 593.78 GB/s algorithmic bandwidth
- **Theoretical max (8×A100 NVLink)**: 600 GB/s (NVSwitch 3.0)
- **Achieved efficiency**: 98.96% (excellent!)
- **Bus bandwidth**: 1039.12 GB/s (bidirectional NVLink traffic)

### Multi-Node AllReduce Benchmark

**Test 4-node cluster (32 GPUs total):**
```bash
# Run with MPI across 4 nodes
mpirun -np 32 \
  -H node1:8,node2:8,node3:8,node4:8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_DISABLE=0 \
  -x NCCL_NET_GDR_LEVEL=5 \
  ./build/all_reduce_perf \
  -b 8 -e 4G -f 2 -g 8
```

**Expected Output (100 Gbps network):**
```
# 4 nodes × 8 GPUs = 32 GPUs
#       size         count      type   redop    root     time   algbw   busbw
  4294967296    1073741824     float     sum      -1  34521.34  124.38  241.74

# algbw: 124.38 GB/s (close to theoretical 100 Gbps = 12.5 GB/s × 10 links)
```

From [NVIDIA GitHub Issue #212: H100 AllReduce Performance is Poor](https://github.com/NVIDIA/nccl-tests/issues/212) (GitHub, accessed 2025-05-06):
> "We test the all_reduce_perf in H100, the algbw is about 250GB/S. But NV officials claim that the bandwidth of all reduce can reach 450GB/S."

**Troubleshooting Low Bandwidth:**
- Check NCCL topology detection: `NCCL_DEBUG=INFO`
- Verify NVLink connectivity: `nvidia-smi topo -m`
- Disable InfiniBand if not available: `NCCL_IB_DISABLE=1`
- Enable GPU Direct RDMA: `NCCL_NET_GDR_LEVEL=5`

### NCCL Tuning Parameters

**Environment Variables:**
```bash
# Increase parallelism (more CUDA blocks for communication)
export NCCL_NTHREADS=512        # Default: 256
export NCCL_MAX_NCHANNELS=16    # Default: 8-12

# Network optimization
export NCCL_SOCKET_IFNAME=eth0  # Specify network interface
export NCCL_IB_DISABLE=0        # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5     # GPU Direct RDMA level

# Algorithm selection
export NCCL_ALGO=Ring           # AllReduce algorithm (Ring, Tree, SHARP)
export NCCL_PROTO=Simple        # Protocol (Simple, LL, LL128)

# Debugging
export NCCL_DEBUG=INFO          # Print topology and tuning info
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV  # Specific subsystems
```

**Testing Different Configurations:**
```bash
# Baseline
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 4G -g 8

# Tune for max bandwidth
NCCL_NTHREADS=512 NCCL_MAX_NCHANNELS=16 \
  ./build/all_reduce_perf -b 8 -e 4G -g 8

# Compare results
```

### arr-coc-0-1 NCCL Benchmark

**Test multi-GPU gradient synchronization:**
```bash
cd arr-coc-0-1

# Run NCCL test for arr-coc model size
# 1.2B parameter model = 4.8 GB (FP32) or 2.4 GB (FP16)
./nccl-tests/build/all_reduce_perf \
  -b 1M \
  -e 2G \
  -f 2 \
  -g 8 \
  -c 0  # In-place operation (gradient buffers)

# Target: > 550 GB/s on 8×A100 (92% of 600 GB/s theoretical)
```

**Integration with PyTorch DDP:**
```python
import torch
import torch.distributed as dist

# Initialize NCCL backend
dist.init_process_group(backend='nccl')

# Measure AllReduce time during training
gradients = torch.randn(1_200_000_000, device='cuda')  # 1.2B params

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
end.record()

torch.cuda.synchronize()
time_ms = start.elapsed_time(end)
bandwidth_gbps = (gradients.numel() * 4 / 1e9) / (time_ms / 1000)

print(f"AllReduce time: {time_ms:.2f} ms")
print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")

# Target: < 8 ms for 4.8 GB gradient synchronization
```

---

## Section 3: Nsight Systems and Nsight Compute (NVIDIA Profilers)

### Nsight Systems (System-Wide Profiling)

**Nsight Systems** provides timeline-based profiling of GPU workloads, showing CPU/GPU activity, CUDA kernels, memory transfers, and NVTX annotations.

**Installation:**
```bash
# Download from NVIDIA Developer (requires NVIDIA account)
wget https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-systems-2024.5.1_2024.5.1.85-1_amd64.deb
sudo dpkg -i nsight-systems-2024.5.1_2024.5.1.85-1_amd64.deb

# Verify installation
nsys --version
# NVIDIA Nsight Systems version 2024.5.1.85-1
```

From [NVIDIA Docs: Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/2024.5/UserGuide/index.html) (NVIDIA, accessed 2024-07-31):
> "The Nsight Systems CLI provides a simple interface to collect on a target without using the GUI. The collected data can then be copied to any system and opened with the GUI."

**Basic Profiling:**
```bash
# Profile PyTorch training script
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=arr_coc_training \
  --force-overwrite=true \
  python training/train.py --epochs 1 --batch-size 64

# Output: arr_coc_training.nsys-rep
```

**Profile Options:**
- `--trace=cuda,nvtx,osrt`: Trace CUDA kernels, NVTX ranges, OS runtime
- `--cuda-memory-usage=true`: Track GPU memory allocations
- `--gpu-metrics-device=all`: Collect GPU metrics (SM %, memory BW)
- `--capture-range=cudaProfilerApi`: Profile only annotated code regions

**Viewing Results:**
```bash
# Open in Nsight Systems GUI (on local machine with display)
nsys-ui arr_coc_training.nsys-rep

# Generate summary report (CLI)
nsys stats arr_coc_training.nsys-rep
```

**Timeline Analysis:**
- **CUDA Kernels**: Which kernels run, duration, concurrency
- **Memory Transfers**: H2D, D2H, D2D transfers
- **CPU Activity**: Host code execution, thread utilization
- **Idle Time**: GPU idle periods (optimization opportunity)

### Nsight Compute (Kernel-Level Profiling)

**Nsight Compute** provides detailed analysis of individual CUDA kernels (register usage, memory bandwidth, warp occupancy, etc.).

**Installation:**
```bash
# Download from NVIDIA Developer
wget https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-compute-2024.5.1_2024.5.1.1-1_amd64.deb
sudo dpkg -i nsight-compute-2024.5.1_2024.5.1.1-1_amd64.deb

# Verify
ncu --version
# NVIDIA (R) Nsight Compute Command Line Profiler
# Copyright (c) 2018-2024 NVIDIA Corporation
# Version 2024.5.1.1 (build 34782007)
```

**Profile Specific Kernel:**
```bash
# Profile top 5 slowest kernels
ncu \
  --set full \
  --target-processes all \
  --kernel-name-base=function \
  --launch-skip 10 \
  --launch-count 5 \
  --export arr_coc_kernels \
  python training/train.py

# Output: arr_coc_kernels.ncu-rep
```

**Profile Options:**
- `--set full`: Collect all metrics (slow but comprehensive)
- `--set basic`: Collect basic metrics only (faster)
- `--kernel-name regex:<pattern>`: Profile kernels matching regex
- `--launch-skip N`: Skip first N kernel launches (warmup)

**Key Metrics:**
```
Kernel: fused_texture_kernel (arr_coc custom CUDA)
Duration: 2.34 ms
SM Efficiency: 87.3%
Occupancy: 75.2%
Memory Throughput: 623 GB/s (78% of peak 800 GB/s)
Compute Throughput: 124 TFLOPS (82% of peak 156 TFLOPS)

Bottleneck: Memory Bound
Recommendation: Reduce global memory transactions, use shared memory
```

### NVTX Annotations for Custom Code

**Add custom ranges to training code:**
```python
import torch.cuda.nvtx as nvtx

# Annotate training loop
for epoch in range(num_epochs):
    nvtx.range_push(f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(dataloader):
        nvtx.range_push("Data Loading")
        images = images.cuda()
        labels = labels.cuda()
        nvtx.range_pop()

        nvtx.range_push("Forward Pass")
        outputs = model(images)
        loss = criterion(outputs, labels)
        nvtx.range_pop()

        nvtx.range_push("Backward Pass")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("Optimizer Step")
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()

    nvtx.range_pop()
```

**Profile with NVTX:**
```bash
nsys profile \
  --trace=cuda,nvtx \
  --nvtx-capture=Epoch \
  --output=arr_coc_epoch \
  python training/train.py

# Timeline shows custom "Data Loading", "Forward Pass", etc. ranges
```

### arr-coc-0-1 Profiling Workflow

**Step 1: System-Wide Profile (Nsight Systems)**
```bash
cd arr-coc-0-1

# Profile full training epoch
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=arr_coc_training_profile \
  python training/train.py --epochs 1 --batch-size 64 --gpus 1

# Analyze timeline, identify slow kernels
nsys stats arr_coc_training_profile.nsys-rep --report cuda_api_sum
```

**Step 2: Kernel-Level Profile (Nsight Compute)**
```bash
# Profile top 3 slowest kernels identified in Nsys
ncu \
  --set full \
  --kernel-name regex:fused_texture|top_k_patches|opponent_balance \
  --launch-count 10 \
  --export arr_coc_kernel_profile \
  python training/train.py --epochs 1 --batch-size 8 --gpus 1

# Analyze kernel bottlenecks
ncu -i arr_coc_kernel_profile.ncu-rep --page details
```

**Step 3: Optimize and Re-Profile**
```bash
# After optimization (e.g., increase shared memory usage)
ncu \
  --set full \
  --kernel-name regex:fused_texture_optimized \
  --launch-count 10 \
  --export arr_coc_optimized \
  python training/train.py

# Compare before/after
ncu --import arr_coc_kernel_profile.ncu-rep,arr_coc_optimized.ncu-rep
```

From [ndeepak.com: System Setup for GPU Performance Measurements](https://ndeepak.com/posts/2025-03-16-gpu-profile/) (ndeepak.com, accessed 2025-03-16):
> "We will use NVIDIA Nsight for profiling. As of this writing, it requires Nsight versions 2025.1 or later."

---

## Section 4: Synthetic Workloads (GEMM, Convolution, Attention)

### Synthetic vs Real Workload Benchmarks

**Synthetic Benchmarks:**
- **Controlled environment** - Isolated operation (GEMM, convolution)
- **Raw performance** - Measure theoretical peak (TFLOPS, GB/s)
- **Reproducible** - Standardized test conditions
- **Use case**: Hardware validation, comparing GPU models

**Real Workload Benchmarks:**
- **Actual applications** - Full model training/inference (ResNet, BERT)
- **End-to-end performance** - Includes data loading, preprocessing, etc.
- **Variable** - Depends on model architecture, batch size, etc.
- **Use case**: Production performance prediction, regression testing

From [Milvus: What is the Difference Between Synthetic and Real-World Benchmarks?](https://milvus.io/ai-quick-reference/what-is-the-difference-between-synthetic-and-realworld-benchmarks) (Milvus, accessed 2025-11-16):
> "Synthetic and real-world benchmarks serve different purposes in evaluating system performance. Synthetic benchmarks are controlled tests designed to stress specific aspects of a system."

### GEMM (General Matrix Multiply) Benchmarks

**Why GEMM Matters:**
- **Core ML operation**: 80-90% of ML training/inference time
- **GPU bottleneck test**: Measures compute throughput (TFLOPS)
- **Precision comparison**: FP32, FP16, BF16, INT8, FP8 performance

**cuBLAS GEMM Benchmark:**
```bash
# Install CUDA samples
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/4_CUDA_Libraries/matrixMul

# Compile
make

# Run GEMM benchmark (4096×4096 matrices)
./matrixMul -wA=4096 -hA=4096 -wB=4096 -hB=4096

# Output:
# Processing time: 12.34 ms
# GFLOPS: 5647.23 (FP32)
# Bandwidth: 687 GB/s
```

**Custom GEMM Benchmark (Multiple Precisions):**
```python
import torch
import time

def benchmark_gemm(M, N, K, dtype, iterations=100):
    """Benchmark GEMM: C = A @ B"""
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_ms = (end - start) * 1000 / iterations
    flops = 2 * M * N * K  # 2 FLOPs per multiply-add
    tflops = (flops / 1e12) / (time_ms / 1000)

    return time_ms, tflops

# Test different precisions
sizes = [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]
dtypes = [torch.float32, torch.float16, torch.bfloat16]

for M, N, K in sizes:
    print(f"\nMatrix size: {M}×{N}×{K}")
    for dtype in dtypes:
        time_ms, tflops = benchmark_gemm(M, N, K, dtype)
        print(f"  {dtype}: {time_ms:.2f} ms, {tflops:.2f} TFLOPS")

# Example output (A100 80GB):
# Matrix size: 4096×4096×4096
#   torch.float32: 12.34 ms, 11.15 TFLOPS
#   torch.float16: 3.21 ms, 42.87 TFLOPS (Tensor Cores)
#   torch.bfloat16: 3.18 ms, 43.29 TFLOPS (Tensor Cores)
```

**A100 Theoretical Peak:**
- FP32: 19.5 TFLOPS
- FP16/BF16 (Tensor Cores): 312 TFLOPS
- INT8 (Tensor Cores): 624 TOPS

**Achieved Efficiency:**
- FP32: 11.15 / 19.5 = 57% (memory bound)
- FP16: 42.87 / 312 = 13.7% (need larger matrices for peak)

### Convolution Benchmarks

**cuDNN Convolution Benchmark:**
```python
import torch
import torch.nn as nn
import time

def benchmark_conv2d(batch, channels, height, width, out_channels, kernel, iterations=100):
    """Benchmark 2D convolution"""
    conv = nn.Conv2d(channels, out_channels, kernel, padding=kernel//2).cuda()
    x = torch.randn(batch, channels, height, width, device='cuda')

    # Warmup
    for _ in range(10):
        y = conv(x)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        y = conv(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_ms = (end - start) * 1000 / iterations
    return time_ms

# ResNet-50 convolution layers
configs = [
    (64, 64, 56, 56, 64, 3),    # Layer 2
    (64, 256, 56, 56, 64, 1),   # Layer 2 (1×1)
    (64, 128, 28, 28, 128, 3),  # Layer 3
    (64, 512, 28, 28, 128, 1),  # Layer 3 (1×1)
]

for batch, in_ch, h, w, out_ch, k in configs:
    time_ms = benchmark_conv2d(batch, in_ch, h, w, out_ch, k)
    print(f"Conv2d({in_ch}→{out_ch}, {k}×{k}, {h}×{w}): {time_ms:.2f} ms")

# Example output:
# Conv2d(64→64, 3×3, 56×56): 2.34 ms
# Conv2d(256→64, 1×1, 56×56): 0.87 ms
# Conv2d(128→128, 3×3, 28×28): 1.12 ms
# Conv2d(512→128, 1×1, 28×28): 0.91 ms
```

### Attention Benchmarks

**Flash Attention vs Standard Attention:**
```python
import torch
from flash_attn import flash_attn_func

def benchmark_attention(batch, seq_len, num_heads, head_dim, use_flash=False):
    """Benchmark multi-head attention"""
    q = torch.randn(batch, seq_len, num_heads, head_dim, device='cuda')
    k = torch.randn(batch, seq_len, num_heads, head_dim, device='cuda')
    v = torch.randn(batch, seq_len, num_heads, head_dim, device='cuda')

    if use_flash:
        # Flash Attention (fused kernel)
        func = lambda: flash_attn_func(q, k, v, causal=False)
    else:
        # Standard Attention (unfused)
        q_proj = q.transpose(1, 2)  # [B, H, S, D]
        k_proj = k.transpose(1, 2)
        v_proj = v.transpose(1, 2)

        def standard_attn():
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_proj)
            return out

        func = standard_attn

    # Warmup
    for _ in range(10):
        func()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        func()

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_ms = (end - start) * 10
    return time_ms

# Test different sequence lengths
seq_lengths = [512, 1024, 2048, 4096]
batch, num_heads, head_dim = 8, 16, 64

for seq_len in seq_lengths:
    time_standard = benchmark_attention(batch, seq_len, num_heads, head_dim, use_flash=False)
    time_flash = benchmark_attention(batch, seq_len, num_heads, head_dim, use_flash=True)
    speedup = time_standard / time_flash

    print(f"Seq {seq_len}: Standard {time_standard:.2f} ms, Flash {time_flash:.2f} ms, Speedup {speedup:.2f}×")

# Example output:
# Seq 512: Standard 3.45 ms, Flash 1.23 ms, Speedup 2.80×
# Seq 1024: Standard 12.34 ms, Flash 3.87 ms, Speedup 3.19×
# Seq 2048: Standard 47.89 ms, Flash 11.23 ms, Speedup 4.26×
# Seq 4096: Standard 189.45 ms, Flash 34.56 ms, Speedup 5.48×
```

### arr-coc-0-1 Synthetic Benchmark Suite

**Test relevance realization operations:**
```python
# test_synthetic_ops.py
import torch
from arr_coc import fused_texture, top_k_patches, opponent_balance

def benchmark_fused_texture():
    """Benchmark fused RGB→LAB+Sobel kernel"""
    rgb = torch.randn(64, 3, 224, 224, device='cuda')

    # Warmup
    for _ in range(10):
        lab, sobel_x, sobel_y = fused_texture(rgb)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        lab, sobel_x, sobel_y = fused_texture(rgb)

    torch.cuda.synchronize()
    time_ms = (time.perf_counter() - start) * 10

    # Compare to unfused PyTorch ops
    time_unfused = benchmark_unfused_texture(rgb)
    speedup = time_unfused / time_ms

    print(f"Fused texture: {time_ms:.2f} ms (Speedup {speedup:.2f}×)")

def benchmark_top_k_patches():
    """Benchmark top-K patch selection"""
    features = torch.randn(64, 196, 512, device='cuda')
    relevance = torch.randn(64, 196, device='cuda')

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        selected, indices = top_k_patches(relevance, features, k=200)

    torch.cuda.synchronize()
    time_ms = (time.perf_counter() - start) * 10

    print(f"Top-K selection: {time_ms:.2f} ms")

# Run benchmarks
benchmark_fused_texture()
benchmark_top_k_patches()
```

---

## Section 5: Real Workload Benchmarking (ResNet, BERT, GPT)

### ResNet-50 Training Benchmark

**Standard ImageNet Training:**
```bash
# Clone torchvision training scripts
git clone https://github.com/pytorch/vision.git
cd vision/references/classification

# Single-GPU training
python train.py \
  --model resnet50 \
  --batch-size 128 \
  --epochs 1 \
  --data-path /data/imagenet \
  --device cuda:0 \
  --print-freq 100

# Output:
# Epoch: [0]  [ 100/5004]  eta: 0:42:13  loss: 6.9234  acc1: 0.781  acc5: 4.688  time: 0.512  data: 0.045  max mem: 8234
# Training time: 45.3 minutes
# Samples/sec: 282
```

**Multi-GPU Training (8×A100):**
```bash
# Distributed Data Parallel
torchrun \
  --nproc_per_node=8 \
  train.py \
  --model resnet50 \
  --batch-size 128 \
  --epochs 1 \
  --data-path /data/imagenet \
  --dist-backend nccl

# Output:
# Training time: 6.2 minutes
# Samples/sec: 2,067 (7.3× speedup from 8 GPUs)
# GPU utilization: 94%
# NCCL AllReduce time: 3.4% of total
```

**Performance Metrics to Track:**
- **Throughput**: Images/second
- **GPU Utilization**: nvidia-smi dmon
- **Memory Usage**: Peak allocated memory
- **Data Loading Time**: Time spent in DataLoader
- **Gradient Sync Time**: NCCL AllReduce duration

### BERT Training Benchmark

**BERT-Large Fine-Tuning (SQuAD):**
```bash
# Clone HuggingFace transformers
git clone https://github.com/huggingface/transformers.git
cd transformers/examples/pytorch/question-answering

# Fine-tune BERT-Large on SQuAD
python run_qa.py \
  --model_name_or_path bert-large-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./bert_squad

# Output:
# Training time: 1.8 hours (single A100)
# Samples/sec: 48
# F1 score: 88.5%
```

**Multi-GPU BERT Training:**
```bash
# 8-GPU distributed training
torchrun \
  --nproc_per_node=8 \
  run_qa.py \
  --model_name_or_path bert-large-uncased \
  --dataset_name squad \
  --do_train \
  --per_device_train_batch_size 12 \
  --fp16 \
  --output_dir ./bert_squad_8gpu

# Output:
# Training time: 18 minutes (6× speedup)
# Samples/sec: 312
```

### GPT Inference Benchmark

**GPT-2 Text Generation Latency:**
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2-large').cuda()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

# Warmup
for _ in range(10):
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100)

# Benchmark
import time
num_iterations = 100

torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(num_iterations):
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, do_sample=False)

torch.cuda.synchronize()
time_ms = (time.perf_counter() - start) * 1000 / num_iterations

tokens_generated = outputs.shape[1] - input_ids.shape[1]
time_per_token = time_ms / tokens_generated

print(f"Generation time: {time_ms:.2f} ms")
print(f"Tokens generated: {tokens_generated}")
print(f"Time per token: {time_per_token:.2f} ms")
print(f"Throughput: {1000/time_per_token:.2f} tokens/sec")

# Example output (A100):
# Generation time: 487.34 ms
# Tokens generated: 93
# Time per token: 5.24 ms
# Throughput: 190.8 tokens/sec
```

### arr-coc-0-1 Real Workload Benchmark

**Full Training Pipeline:**
```bash
cd arr-coc-0-1

# Benchmark end-to-end training
python training/benchmark_training.py \
  --config configs/arr_coc_base.yaml \
  --data-path /data/coco \
  --batch-size 64 \
  --iterations 500 \
  --gpus 8

# Measure:
# - Data loading time
# - Forward pass time (texture extraction, relevance scoring, LOD allocation)
# - Backward pass time (gradient computation)
# - NCCL AllReduce time
# - Optimizer step time
# - End-to-end iteration time
```

**Expected Breakdown:**
```
Data loading: 12.3 ms/iter (8%)
Forward pass: 45.6 ms/iter (30%)
  - Texture extraction: 8.2 ms
  - Relevance scoring: 15.4 ms
  - LOD allocation: 3.1 ms
  - VLM forward: 18.9 ms
Backward pass: 67.8 ms/iter (44%)
NCCL AllReduce: 8.4 ms/iter (5%)
Optimizer step: 18.2 ms/iter (12%)
---
Total: 152.3 ms/iter
Throughput: 420 samples/sec (64 batch × 8 GPUs)
```

From [IBM Think: What is a GPU?](https://www.ibm.com/think/topics/gpu) (IBM, accessed 2025-11-16):
> "Synthetic benchmarks test a GPU's raw performance in a standardized environment. Real-world benchmarks test a GPU's performance in specific applications."

---

## Section 6: A/B Testing GPU Configurations

### Testing Methodology

**A/B Test Framework:**
1. **Baseline Configuration** (A): Current setup
2. **Variant Configuration** (B): Modified setup (new GPU, driver, tuning)
3. **Controlled Variables**: Same code, data, batch size, etc.
4. **Metrics**: Throughput, latency, cost, memory usage

**Example: A100 vs H100 Comparison**
```python
# test_gpu_comparison.py
import torch
import time
from arr_coc import ARRCOCModel

def benchmark_gpu_type(model, dataloader, gpu_name, iterations=100):
    """Benchmark model on specific GPU"""
    model = model.cuda()

    times = []
    for i, (images, labels) in enumerate(dataloader):
        if i >= iterations:
            break

        images = images.cuda()
        labels = labels.cuda()

        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model(images)
        loss = outputs.loss

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    throughput = images.shape[0] / avg_time

    print(f"{gpu_name} Results:")
    print(f"  Avg time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  p50: {sorted(times)[len(times)//2]*1000:.2f} ms")
    print(f"  p99: {sorted(times)[int(len(times)*0.99)]*1000:.2f} ms")

    return avg_time, throughput

# Run on A100
model = ARRCOCModel.from_pretrained('arr-coc-base')
dataloader = get_coco_dataloader(batch_size=32)

a100_time, a100_throughput = benchmark_gpu_type(model, dataloader, "A100 80GB")

# Run on H100 (requires H100 instance)
# h100_time, h100_throughput = benchmark_gpu_type(model, dataloader, "H100 80GB")

# Compare
# speedup = a100_time / h100_time
# print(f"\nH100 Speedup: {speedup:.2f}×")
```

### Testing GPU Driver Versions

**Compare NVIDIA Driver Performance:**
```bash
# Test driver 535.x (stable)
sudo apt-get install --reinstall nvidia-driver-535
nvidia-smi

# Run benchmark
python benchmark_training.py --iterations 100 > results_driver_535.txt

# Test driver 550.x (latest)
sudo apt-get install --reinstall nvidia-driver-550
nvidia-smi

# Run same benchmark
python benchmark_training.py --iterations 100 > results_driver_550.txt

# Compare results
diff results_driver_535.txt results_driver_550.txt
```

### Testing CUDA Toolkit Versions

**CUDA 12.1 vs 12.4:**
```bash
# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Rebuild PyTorch extension
cd arr-coc-cuda
python setup.py clean
python setup.py build_ext --inplace

# Benchmark
python benchmark_kernels.py > results_cuda_12_1.txt

# Install CUDA 12.4
# (repeat process)

# Compare kernel performance
python compare_results.py results_cuda_12_1.txt results_cuda_12_4.txt
```

### arr-coc-0-1 Configuration Matrix

**Test Matrix:**
```yaml
configurations:
  baseline:
    gpu: A100-80GB
    driver: 535.154.05
    cuda: 12.1
    batch_size: 64
    precision: fp16

  variant_1:
    gpu: A100-80GB
    driver: 550.90.07  # Latest
    cuda: 12.4
    batch_size: 64
    precision: fp16

  variant_2:
    gpu: A100-80GB
    driver: 535.154.05
    cuda: 12.1
    batch_size: 64
    precision: bf16  # Test BF16 vs FP16

  variant_3:
    gpu: L4
    driver: 535.154.05
    cuda: 12.1
    batch_size: 32  # Smaller batch for L4
    precision: fp16

metrics:
  - throughput (samples/sec)
  - latency_p50 (ms)
  - latency_p99 (ms)
  - memory_usage (GB)
  - cost_per_1000_samples ($)
```

**Automated A/B Test Runner:**
```bash
# Run all configurations
python testing/ab_test_runner.py \
  --config testing/gpu_config_matrix.yaml \
  --iterations 500 \
  --output results/ab_test_2025-11-16.json

# Generate comparison report
python testing/generate_report.py \
  --input results/ab_test_2025-11-16.json \
  --output results/ab_test_report.html
```

---

## Section 7: Regression Testing (Performance CI/CD)

### Automated Performance Testing in CI/CD

**GitHub Actions Workflow:**
```yaml
# .github/workflows/performance_tests.yml
name: GPU Performance Tests

on:
  pull_request:
    paths:
      - 'arr_coc/**.py'
      - 'arr_coc/csrc/**.cu'
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu-a100]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Run performance benchmarks
        run: |
          python testing/benchmark_suite.py \
            --output results/benchmark_${GITHUB_SHA}.json

      - name: Compare with baseline
        run: |
          python testing/compare_performance.py \
            --baseline results/baseline.json \
            --current results/benchmark_${GITHUB_SHA}.json \
            --threshold 0.05  # Fail if >5% regression

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: results/benchmark_${GITHUB_SHA}.json
```

**Benchmark Suite:**
```python
# testing/benchmark_suite.py
import torch
import json
from arr_coc import ARRCOCModel

def run_benchmark_suite():
    """Run standard benchmark tests"""
    model = ARRCOCModel.from_pretrained('arr-coc-base').cuda()

    results = {}

    # Test 1: Forward pass latency
    results['forward_latency_ms'] = benchmark_forward_pass(model)

    # Test 2: Backward pass latency
    results['backward_latency_ms'] = benchmark_backward_pass(model)

    # Test 3: Throughput
    results['throughput_samples_sec'] = benchmark_throughput(model)

    # Test 4: Memory usage
    results['peak_memory_gb'] = benchmark_memory_usage(model)

    # Test 5: Custom kernel performance
    results['fused_texture_ms'] = benchmark_fused_texture()
    results['top_k_patches_ms'] = benchmark_top_k_patches()

    return results

if __name__ == '__main__':
    results = run_benchmark_suite()

    with open('results/benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
```

**Performance Comparison:**
```python
# testing/compare_performance.py
import json
import sys

def compare_performance(baseline_path, current_path, threshold=0.05):
    """Compare current performance vs baseline"""
    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(current_path) as f:
        current = json.load(f)

    regressions = []

    for metric, baseline_value in baseline.items():
        current_value = current[metric]

        # Lower is better for latency/time metrics
        if 'latency' in metric or 'ms' in metric or 'time' in metric:
            regression = (current_value - baseline_value) / baseline_value
            if regression > threshold:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'regression_pct': regression * 100
                })

        # Higher is better for throughput metrics
        elif 'throughput' in metric or 'samples' in metric:
            regression = (baseline_value - current_value) / baseline_value
            if regression > threshold:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'regression_pct': regression * 100
                })

    if regressions:
        print("PERFORMANCE REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  {r['metric']}: {r['baseline']:.2f} → {r['current']:.2f} ({r['regression_pct']:+.1f}%)")
        sys.exit(1)
    else:
        print("No performance regressions detected.")
        sys.exit(0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--current', required=True)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()

    compare_performance(args.baseline, args.current, args.threshold)
```

### Historical Performance Tracking

**Store results in database:**
```python
# testing/store_results.py
import sqlite3
import json
from datetime import datetime

def store_benchmark_results(results, git_commit, gpu_type):
    """Store benchmark results in SQLite database"""
    conn = sqlite3.connect('performance_history.db')
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            git_commit TEXT,
            gpu_type TEXT,
            forward_latency_ms REAL,
            backward_latency_ms REAL,
            throughput_samples_sec REAL,
            peak_memory_gb REAL,
            fused_texture_ms REAL,
            top_k_patches_ms REAL
        )
    ''')

    # Insert results
    cursor.execute('''
        INSERT INTO benchmarks VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        git_commit,
        gpu_type,
        results['forward_latency_ms'],
        results['backward_latency_ms'],
        results['throughput_samples_sec'],
        results['peak_memory_gb'],
        results['fused_texture_ms'],
        results['top_k_patches_ms']
    ))

    conn.commit()
    conn.close()

# Usage
results = run_benchmark_suite()
store_benchmark_results(results, git_commit='abc123', gpu_type='A100-80GB')
```

**Generate performance trend graph:**
```python
import matplotlib.pyplot as plt
import sqlite3

def plot_performance_trends():
    """Plot performance over time"""
    conn = sqlite3.connect('performance_history.db')
    cursor = conn.cursor()

    cursor.execute('SELECT timestamp, throughput_samples_sec FROM benchmarks ORDER BY timestamp')
    data = cursor.fetchall()

    timestamps = [row[0] for row in data]
    throughput = [row[1] for row in data]

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, throughput, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('Training Throughput Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_trends.png')

plot_performance_trends()
```

---

## Section 8: arr-coc-0-1 Production Benchmark Suite

### Comprehensive Test Suite

**benchmark_suite.py:**
```python
import torch
import time
import json
from arr_coc import ARRCOCModel, fused_texture, top_k_patches

class ARRCOCBenchmarkSuite:
    """Complete benchmark suite for arr-coc-0-1"""

    def __init__(self, gpu_id=0):
        self.device = f'cuda:{gpu_id}'
        torch.cuda.set_device(self.device)
        self.results = {}

    def benchmark_mlperf_style_training(self):
        """MLPerf-style training benchmark"""
        model = ARRCOCModel.from_pretrained('arr-coc-base').to(self.device)
        dataloader = self._get_synthetic_dataloader(batch_size=64)

        # Warmup
        for _ in range(10):
            batch = next(iter(dataloader))
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for i, batch in enumerate(dataloader):
            if i >= 100:
                break
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = (100 * 64) / elapsed
        self.results['mlperf_training_throughput'] = throughput
        self.results['mlperf_training_time_sec'] = elapsed

    def benchmark_nccl_allreduce(self):
        """NCCL AllReduce bandwidth test"""
        # Simulate gradient tensor (1.2B parameters × 4 bytes)
        gradients = torch.randn(1_200_000_000, device=self.device)

        # Measure AllReduce time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        import torch.distributed as dist
        if dist.is_initialized():
            start.record()
            dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
            end.record()
            torch.cuda.synchronize()

            time_ms = start.elapsed_time(end)
            bandwidth_gbps = (gradients.numel() * 4 / 1e9) / (time_ms / 1000)

            self.results['nccl_allreduce_time_ms'] = time_ms
            self.results['nccl_allreduce_bandwidth_gbps'] = bandwidth_gbps

    def benchmark_custom_kernels(self):
        """Benchmark arr-coc custom CUDA kernels"""
        # Fused texture extraction
        rgb = torch.randn(64, 3, 224, 224, device=self.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(100):
            lab, sobel_x, sobel_y = fused_texture(rgb)

        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) * 10
        self.results['fused_texture_ms'] = time_ms

        # Top-K patch selection
        features = torch.randn(64, 196, 512, device=self.device)
        relevance = torch.randn(64, 196, device=self.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(100):
            selected, indices = top_k_patches(relevance, features, k=200)

        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) * 10
        self.results['top_k_patches_ms'] = time_ms

    def benchmark_real_workload(self):
        """End-to-end training on COCO dataset"""
        model = ARRCOCModel.from_pretrained('arr-coc-base').to(self.device)
        dataloader = self._get_coco_dataloader(batch_size=32)

        times = []
        for i, batch in enumerate(dataloader):
            if i >= 500:
                break

            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        self.results['real_workload_avg_time_ms'] = sum(times) / len(times) * 1000
        self.results['real_workload_p50_ms'] = sorted(times)[len(times)//2] * 1000
        self.results['real_workload_p99_ms'] = sorted(times)[int(len(times)*0.99)] * 1000

    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("Running MLPerf-style training benchmark...")
        self.benchmark_mlperf_style_training()

        print("Running NCCL AllReduce benchmark...")
        self.benchmark_nccl_allreduce()

        print("Running custom kernel benchmarks...")
        self.benchmark_custom_kernels()

        print("Running real workload benchmark...")
        self.benchmark_real_workload()

        return self.results

# Run benchmarks
suite = ARRCOCBenchmarkSuite(gpu_id=0)
results = suite.run_all_benchmarks()

print("\n=== Benchmark Results ===")
print(json.dumps(results, indent=2))

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**Expected Output:**
```json
{
  "mlperf_training_throughput": 428.5,
  "mlperf_training_time_sec": 14.95,
  "nccl_allreduce_time_ms": 8.23,
  "nccl_allreduce_bandwidth_gbps": 583.4,
  "fused_texture_ms": 2.34,
  "top_k_patches_ms": 3.12,
  "real_workload_avg_time_ms": 152.3,
  "real_workload_p50_ms": 148.7,
  "real_workload_p99_ms": 167.9
}
```

### Production Acceptance Criteria

**GPU Performance Checklist:**
- [ ] MLPerf training throughput > 400 samples/sec (8×A100)
- [ ] NCCL AllReduce bandwidth > 550 GB/s (92% of theoretical 600 GB/s)
- [ ] Fused texture kernel < 2.5 ms/batch
- [ ] Top-K patch selection < 3.5 ms/batch
- [ ] End-to-end training p99 latency < 170 ms
- [ ] GPU utilization > 90% during training
- [ ] No performance regression > 5% from baseline
- [ ] Memory usage < 75 GB (leaves headroom for larger batches)

---

## Sources

**Official Documentation:**
- [MLCommons MLPerf Benchmarks](https://mlcommons.org/benchmarks/training/) - MLCommons, accessed 2025-11-16
- [NVIDIA Developer: Nsight Systems](https://developer.nvidia.com/nsight-systems) - NVIDIA, accessed 2025-11-16
- [NVIDIA Developer: Nsight Compute](https://developer.nvidia.com/nsight-compute) - NVIDIA, accessed 2025-11-16
- [NVIDIA Docs: Nsight Systems User Guide 2024.5](https://docs.nvidia.com/nsight-systems/2024.5/UserGuide/index.html) - NVIDIA, accessed 2024-07-31

**Web Research:**
- [NVIDIA Developer Blog: Understanding NCCL Tuning](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) - NVIDIA, accessed 2025-07-22
- [Data Center Knowledge: How MLPerf Benchmarks Guide Data Center Decisions](https://www.datacenterknowledge.com/ai-data-centers/how-mlperf-benchmarks-guide-data-center-design-decisions) - Data Center Knowledge, accessed 2025-10-22
- [HPCwire: NVIDIA Showcases Blackwell Ultra Performance on MLPerf Benchmark](https://www.hpcwire.com/aiwire/2025/11/14/nvidia-showcases-blackwell-ultra-performance-on-mlperf-benchmark/) - HPCwire, accessed 2025-11-14
- [Together AI: A Practitioner's Guide to Testing Large GPU Clusters](https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models) - Together AI, accessed 2025-08-13
- [Milvus: What is the Difference Between Synthetic and Real-World Benchmarks?](https://milvus.io/ai-quick-reference/what-is-the-difference-between-synthetic-and-realworld-benchmarks) - Milvus, accessed 2025-11-16
- [ndeepak.com: System Setup for GPU Performance Measurements](https://ndeepak.com/posts/2025-03-16-gpu-profile/) - ndeepak.com, accessed 2025-03-16
- [IBM Think: What is a GPU?](https://www.ibm.com/think/topics/gpu) - IBM, accessed 2025-11-16

**GitHub:**
- [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) - NCCL performance benchmarks
- [NVIDIA GitHub Issue #212: H100 AllReduce Performance](https://github.com/NVIDIA/nccl-tests/issues/212) - GitHub, accessed 2025-05-06
- [mlcommons/training](https://github.com/mlcommons/training) - MLPerf Training reference implementations

**Existing Knowledge Base:**
- [cuda/04-pytorch-custom-cuda-extensions.md](cuda/04-pytorch-custom-cuda-extensions.md) - Custom CUDA kernel profiling patterns
- [practical-implementation/55-vlm-inference-latency-benchmarks.md](../practical-implementation/55-vlm-inference-latency-benchmarks.md) - VLM benchmarking methodology (if exists)
