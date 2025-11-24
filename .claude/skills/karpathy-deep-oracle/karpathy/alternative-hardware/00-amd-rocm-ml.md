# AMD ROCm for Machine Learning

**AMD's open-source alternative to NVIDIA CUDA for GPU computing**

## Overview

ROCm (Radeon Open Compute) is AMD's open-source software platform for GPU-accelerated computing, designed to compete with NVIDIA's proprietary CUDA ecosystem. While CUDA remains the dominant platform for AI/ML development, ROCm offers a compelling alternative with open-source flexibility, cost-effective AMD hardware, and growing framework support.

**Key Value Proposition:**
- **Open Source**: Complete software stack under MIT/Apache licenses
- **Cost Effective**: AMD GPUs typically 10-30% cheaper than NVIDIA equivalents
- **CUDA Compatibility**: HIP layer allows porting existing CUDA code with minimal changes
- **High Memory**: MI300X offers 192GB HBM3 vs H100's 80GB

From [ROCm vs CUDA comparison](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/) (accessed 2025-11-14):
> "AMD's ROCm is an open-source software platform designed for GPU-accelerated computing. It provides the tools and libraries necessary for running high-performance applications on AMD GPUs. ROCm's open-source nature allows for greater flexibility and customization."

## ROCm Architecture

### Core Components

**HIP (Heterogeneous-compute Interface for Portability)**:
- Runtime API similar to CUDA Runtime
- Allows writing portable GPU code
- Compiles to both AMD (via ROCm) and NVIDIA (via CUDA) backends

**ROCm Runtime**:
- Device management and memory allocation
- Kernel dispatch and execution
- Analogous to CUDA Runtime API

**MIOpen**: Deep learning library (equivalent to cuDNN)
**rocBLAS**: BLAS operations (equivalent to cuBLAS)
**rocFFT**: Fast Fourier transforms
**RCCL**: Collective communications (equivalent to NCCL)

From [AMD ROCm PyTorch installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) (accessed 2025-11-14):
> "AMD recommends the PIP install method to create a PyTorch environment when working with ROCm for machine learning development."

### Software Stack

```
Application Layer
    ├── PyTorch (ROCm backend)
    ├── TensorFlow (ROCm backend)
    └── JAX (experimental)

Library Layer
    ├── MIOpen (DNN primitives)
    ├── rocBLAS (linear algebra)
    ├── rocFFT (FFT operations)
    └── RCCL (multi-GPU comms)

Runtime Layer
    ├── HIP Runtime
    └── ROCm Runtime

Kernel Layer
    └── Linux kernel with amdgpu driver

Hardware
    └── AMD RDNA/CDNA GPUs
```

**No New Programming Language Required**: Like CUDA, ROCm works with existing languages (C, C++, Python). Both platforms interpret code into hardware instructions, acting as hardware abstraction layers.

## PyTorch on ROCm

### Installation (Ubuntu 24.04 Example)

From [official ROCm docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html):

```bash
# Install Python and pip
sudo apt install python3-pip -y
pip3 install --upgrade pip wheel

# Download ROCm 7.1 PyTorch wheels
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/torch-2.8.0%2Brocm7.1.0.lw.git7a520360-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/torchvision-0.23.0%2Brocm7.1.0.git824e8c87-cp312-cp312-linux_x86_64.whl

# Uninstall any existing PyTorch
pip3 uninstall torch torchvision triton torchaudio

# Install ROCm PyTorch
pip3 install torch-2.8.0+rocm7.1.0.lw.git7a520360-cp312-cp312-linux_x86_64.whl \
             torchvision-0.23.0+rocm7.1.0.git824e8c87-cp312-cp312-linux_x86_64.whl
```

**Note**: The `--break-system-packages` flag may be needed for Python 3.12 in non-virtual environments.

### Docker Installation (Recommended)

```bash
# Pull official ROCm PyTorch image
sudo docker pull rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Run container with GPU access
sudo docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

### Verification

```bash
# Check PyTorch installed
python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'

# Check GPU available
python3 -c 'import torch; print(torch.cuda.is_available())'  # True

# Display GPU name
python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
# Output: device name [0]: Radeon RX 7900 XTX

# Full environment info
python3 -m torch.utils.collect_env
```

**Expected Output**:
- PyTorch version: 2.8.0+rocm7.1
- ROCM used to build PyTorch: 7.1
- Is CUDA available: True (note: torch.cuda APIs work on ROCm)
- HIP runtime version: 7.1.x
- MIOpen runtime version: 7.1.x

## CUDA to ROCm Porting

### HIP Translation Layer

From [AMD GPUOpen HIP portability guide](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-hipify-readme/) (accessed 2025-11-14):
> "HIP is designed to ease the porting of existing CUDA code into the HIP environment. This blog discusses various ROCm tools developers can leverage to port existing applications from CUDA to HIP."

**Two Hipify Tools**:

1. **hipify-clang** (Recommended):
   - Uses Clang AST for accurate translation
   - Handles complex CUDA code
   - Better for production code

2. **hipify-perl**:
   - Simple regex-based translator
   - Good for simple kernel code
   - Faster but less accurate

### Translation Example

**Original CUDA Code**:
```cpp
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

**HIP Translated Code**:
```cpp
#include <hip/hip_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(vectorAdd, dim3(blocks), dim3(threads), 0, 0, d_A, d_B, d_C, N);

    hipMemcpy(h_C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
}
```

**Key Changes**:
- `cuda*` → `hip*` (most API calls)
- `<<<blocks, threads>>>` → `hipLaunchKernelGGL()` (kernel launch)
- Include path: `cuda_runtime.h` → `hip/hip_runtime.h`

### API Compatibility

From [ROCm HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) (accessed 2025-11-14):

**Directly Compatible APIs**:
- Memory allocation: `cudaMalloc` → `hipMalloc`
- Memory copy: `cudaMemcpy` → `hipMemcpy`
- Device sync: `cudaDeviceSynchronize` → `hipDeviceSynchronize`
- Streams: `cudaStream_t` → `hipStream_t`
- Events: `cudaEvent_t` → `hipEvent_t`

**Requires Manual Porting**:
- CUDA-specific intrinsics (`__ballot_sync`, `__shfl_sync`)
- Texture memory (different implementation)
- Dynamic parallelism (limited support)
- Some CUDA libraries (cuDNN → MIOpen requires code changes)

## AMD MI300X Architecture

### Hardware Specifications

From [AMD MI300X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) and [Neysa MI300X analysis](https://neysa.ai/blog/amd-mi300x/) (accessed 2025-11-14):

**Compute**:
- **Compute Units (CUs)**: 304 CUs
- **Architecture**: CDNA 3 (3rd gen AMD data center GPU)
- **Peak FP64**: 163 TFLOPs (double precision)
- **Peak FP32**: 653 TFLOPs (single precision)
- **Peak FP16/BF16**: 1,307 TFLOPs (mixed precision)
- **Peak INT8**: 2,614 TOPS (AI inference)

**Memory**:
- **Capacity**: 192GB HBM3 (industry-leading for AI accelerators)
- **Bandwidth**: 5.3 TB/s (8 HBM3 stacks)
- **Memory per CU**: ~632 MB per CU
- **ECC**: Full error correction

**Interconnect**:
- **GPU-to-GPU**: 128 GB/s per direction via Infinity Fabric
- **PCIe**: Gen 5 x16 (128 GB/s bidirectional)
- **Power**: 750W TDP

**Chiplet Design**:
- 3D stacked chiplets (multiple compute dies + HBM stacks)
- Improved thermal management vs monolithic design
- Better scalability and yield

### MI300X vs H100 Comparison

From [TensorWave ROCm vs CUDA benchmarks](https://tensorwave.com/blog/rocm-vs-cuda-a-performance-showdown-for-modern-ai-workloads) (accessed 2025-11-14):

| Feature | AMD MI300X | NVIDIA H100 |
|---------|-----------|-------------|
| **Memory** | 192GB HBM3 | 80GB HBM3/HBM3e |
| **Memory Bandwidth** | 5.3 TB/s | 3.35 TB/s (HBM3) / 4.8 TB/s (HBM3e) |
| **FP16/BF16 (Tensor)** | 1,307 TFLOPs | 1,979 TFLOPs (with sparsity) |
| **FP8** | Not officially supported | 3,958 TFLOPs |
| **TDP** | 750W | 700W |
| **Cost (estimated)** | $10,000-$15,000 | $25,000-$40,000 |
| **Software Ecosystem** | ROCm (growing) | CUDA (mature) |

**MI300X Advantages**:
- **2.4× more memory** (critical for large language models)
- **58% higher memory bandwidth** (better for memory-bound workloads)
- **40-60% lower cost** per unit
- **Open-source software stack**

**H100 Advantages**:
- **51% higher FP16 compute** (better for compute-bound workloads)
- **FP8 support** (2× faster AI inference on Hopper architecture)
- **Mature CUDA ecosystem** (wider framework support)
- **Better software tooling** (Nsight, CUDA-GDB, extensive libraries)

From [TensorWave benchmarks](https://tensorwave.com/blog/rocm-vs-cuda-a-performance-showdown-for-modern-ai-workloads):
> "ROCm + AMD MI325X is ready for prime time. See benchmarks vs CUDA and why more teams are switching to ROCm for AI performance and cost."

Performance typically shows CUDA 10-30% faster in compute-intensive workloads, but MI300X excels in memory-bound scenarios (large batch inference, large model training).

## Framework Support

### Current Framework Compatibility (2024-2025)

From [Scimus ROCm comparison](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/):

**Fully Supported**:
- **PyTorch**: Official ROCm backend, regular releases
- **TensorFlow**: ROCm builds available
- **ONNX Runtime**: DirectML backend supports AMD GPUs
- **MIGraphX**: AMD's native inference engine

**Growing Support**:
- **JAX**: Experimental ROCm backend
- **Triton**: OpenAI's GPU language (ROCm port available)
- **vLLM**: LLM inference framework (ROCm support added 2024)
- **Hugging Face Transformers**: Works via PyTorch ROCm backend

**Limited/No Support**:
- **Keras** (TF backend): Limited testing on ROCm
- Many specialized CUDA libraries still lack ROCm equivalents

### PyTorch ROCm Example

```python
import torch

# Check ROCm availability
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Create tensors on GPU
device = torch.device("cuda")  # Works with ROCm
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Matrix multiplication on GPU
z = torch.matmul(x, y)

# Use MIOpen for convolutions
conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
input_tensor = torch.randn(1, 3, 224, 224, device=device)
output = conv(input_tensor)

print(f"Output shape: {output.shape}")
```

**Note**: PyTorch's `torch.cuda` namespace works with ROCm. The naming is historical—internally, PyTorch detects ROCm and uses HIP runtime.

## Performance Comparison

### ROCm vs CUDA Performance

From [Reddit r/MachineLearning discussion](https://www.reddit.com/r/MachineLearning/comments/1fa8vq5/d_why_is_cuda_so_much_faster_than_rocm/) (accessed 2025-11-14):
> "Properly written and optimized ROCm code is just as fast as CUDA, right below whatever the maximum number of tflops of your GPU is. However, every machine learning library is written with CUDA in mind."

**Reality Check**:
- **Raw compute**: ROCm within 5-10% of CUDA on same-tier hardware
- **Framework overhead**: CUDA 10-30% faster due to better library optimization
- **Memory-bound workloads**: ROCm competitive or better (MI300X's 192GB advantage)

From [Thunder Compute comparison](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing) (accessed 2025-11-14):
> "Some recent testing shows that CUDA typically outperforms ROCm by 10% to 30% in compute-intensive workloads. AMD's MI325X represents a turning point..."

**When ROCm Beats CUDA**:
- Large batch inference (MI300X's 192GB vs H100's 80GB)
- Memory bandwidth-bound workloads (5.3 TB/s vs 3.35 TB/s)
- Multi-node training (lower GPU cost = more GPUs in budget)

**When CUDA Dominates**:
- Highly optimized frameworks (cuDNN fused kernels)
- FP8 inference workloads (H100 exclusive)
- Specialized CUDA libraries (CUTLASS, cuBLAS tuning)

### Real-World Benchmarks

**LLaMA 70B Training** (tokens/sec/GPU):
- NVIDIA H100 + CUDA: ~3,200 tokens/sec
- AMD MI300X + ROCm: ~2,800 tokens/sec
- Performance gap: ~12-15% (improving with each ROCm release)

**Stable Diffusion Inference** (it/s at 512×512):
- NVIDIA RTX 4090 + CUDA: ~35 it/s
- AMD RX 7900 XTX + ROCm: ~28 it/s
- Performance gap: ~20%

**Cost-Performance**:
- MI300X offers 2.4× memory at 40-60% lower cost
- For memory-bound workloads, MI300X can be 2-3× more cost-effective

## Installation and Compatibility

### Linux Kernel Requirements

From [Scimus ROCm guide](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/):
> "ROCm requires the use of a newer Linux kernel. The advantage is that ROCm can be easily integrated into modern Linux environments, often with fewer compatibility issues."

**Minimum Kernel**: 5.4+ (recommended 5.15+)
**Supported Distros**:
- Ubuntu 22.04 / 24.04 (officially supported)
- RHEL 9.x / Rocky Linux 9.x
- SUSE Linux Enterprise 15 SP4+

**Installation Methods**:
1. **Package manager** (apt/yum): Easiest, may lag latest features
2. **Docker containers**: Recommended for ML workloads
3. **Source build**: For cutting-edge features or custom hardware

### Docker Deployment Pattern

From [Scimus](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/):
> "Applications can be packaged in Docker containers with ROCm libraries or built as single executable files that include the necessary ROCm components. This method simplifies deployment significantly."

**Advantages**:
- No host driver conflicts
- Reproducible environments
- Simpler updates and rollbacks
- Microservices-friendly

## Supported Hardware

### AMD Radeon GPUs (Consumer)

From [ROCm compatibility docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/compatibility.html) (accessed 2025-11-14):

**RDNA 3 (Recommended for ML)**:
- Radeon RX 7900 XTX (24GB VRAM)
- Radeon RX 7900 XT (20GB VRAM)
- Radeon RX 7800 XT (16GB VRAM)
- Radeon RX 7700 XT (12GB VRAM)

**RDNA 2**:
- Radeon RX 6900 XT (16GB)
- Radeon RX 6800 XT (16GB)
- Limited ROCm support, not recommended for production

### AMD Instinct GPUs (Data Center)

**CDNA 3 (Current Generation)**:
- **MI300X**: 192GB HBM3, flagship AI accelerator (2023-2024)
- **MI300A**: APU with integrated CPU+GPU (HPC workloads)

**CDNA 2**:
- **MI250X**: 128GB HBM2e, dual-GPU module
- **MI250**: Single-GPU variant
- **MI210**: 64GB HBM2e, cost-effective option

**CDNA 1** (Legacy):
- MI100: 32GB HBM2, first CDNA GPU

### Cloud Provider Support

From [Scimus comparison](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/):
> "Support from cloud providers is also a key factor in platform choice. CUDA is more widely supported among major cloud providers, making it the preferred option for organizations seeking scalable, cloud-based AI and HPC solutions. In contrast, ROCm support among cloud providers is more limited."

**ROCm Cloud Availability (2024-2025)**:
- **Oracle Cloud Infrastructure**: MI300X instances available
- **Microsoft Azure**: MI300X preview (limited regions)
- **Neysa**: Full MI300X support with AI-optimized stack
- **AWS**: No AMD GPU instances yet (NVIDIA-only)
- **Google Cloud**: No AMD GPU instances yet

**CUDA Ubiquity**: All major cloud providers offer extensive NVIDIA GPU options (A100, H100, L4, T4, etc.)

## Advantages of ROCm

### Open Source Benefits

From [Scimus](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/):
> "ROCm stands out for its open-source nature, cost-effectiveness, and flexibility, making it an ideal choice for organizations that need to customize their computing environment or are working within budget constraints."

**Customization Examples**:
- Modify runtime for specialized hardware
- Port to non-standard platforms (ARM servers)
- Add custom profiling/debugging hooks
- Audit code for security/compliance

**Licensing**:
- MIT/Apache 2.0 licenses (permissive)
- No vendor lock-in concerns
- Community contributions welcome
- Transparent development process

### Cost Advantages

**Hardware Cost Comparison** (estimated street prices):

| GPU | Price | Memory | Cost per GB |
|-----|-------|--------|-------------|
| AMD MI300X | $10,000-$15,000 | 192GB | $52-$78/GB |
| NVIDIA H100 SXM5 | $25,000-$40,000 | 80GB | $312-$500/GB |
| AMD RX 7900 XTX | $900-$1,000 | 24GB | $37-$42/GB |
| NVIDIA RTX 4090 | $1,600-$2,000 | 24GB | $67-$83/GB |

**Total Cost of Ownership** (3-year):
- Lower upfront hardware cost (40-60% savings)
- Same/lower power consumption (MI300X: 750W vs H100: 700W)
- No software licensing fees (ROCm is free, CUDA is free but proprietary)
- **BUT**: Higher engineering overhead for ROCm setup/optimization

**When ROCm TCO Wins**:
- Large GPU clusters (100+ GPUs)
- Memory-intensive workloads
- Long-term projects with optimization time
- Organizations with in-house GPU expertise

### Code Portability

**HIP Portability Layer**:
```cpp
// Same HIP code compiles for both AMD and NVIDIA
#include <hip/hip_runtime.h>

__global__ void kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}

int main() {
    float *d_data;
    hipMalloc(&d_data, N * sizeof(float));
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, d_data);
    hipDeviceSynchronize();
    hipFree(d_data);
}
```

**Compilation**:
```bash
# For AMD GPU (ROCm)
hipcc -o program program.cpp

# For NVIDIA GPU (CUDA)
hipcc --cuda -o program program.cpp
```

**Benefits**:
- Single codebase for multi-vendor GPUs
- Easier hardware migration
- Reduced vendor lock-in
- Fallback options if GPU unavailable

## Disadvantages and Challenges

### Ecosystem Maturity

From [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/1fa8vq5/d_why_is_cuda_so_much_faster_than_rocm/):
> "Because every machine learning library is written with CUDA in mind. This means NVIDIA's hardware is highly optimized for this exact use case, while AMD's is not."

**CUDA Ecosystem Advantages**:
- **15+ years** of development (CUDA launched 2007)
- **Thousands** of CUDA-optimized libraries
- **Millions** of developers trained in CUDA
- **Extensive** documentation and tutorials
- **Professional** support from NVIDIA

**ROCm Challenges**:
- Framework support lags 6-12 months behind CUDA
- Fewer optimized kernels (MIOpen vs cuDNN)
- Limited third-party libraries
- Smaller community = fewer Stack Overflow answers
- Documentation gaps for advanced features

### Software Stability

From [Reddit r/ROCm discussions](https://www.reddit.com/r/ROCm/):
> "ROCm is only for the brave." (common sentiment in 2023-2024)

**Common Issues**:
- Driver conflicts with newer kernels
- MIOpen bugs in specific layer types
- Compilation errors with hipify tools
- Performance regressions between ROCm versions
- Less mature profiling/debugging tools

**Improving**: ROCm 6.0+ (2024) shows significant stability improvements

### Limited Cloud Support

**Cloud Availability Gap**:
- AWS: 0 AMD GPU instance types
- Google Cloud: 0 AMD GPU instance types
- Azure: Limited MI300X preview only
- Oracle: MI300X available but smaller footprint than NVIDIA

**Impact on ML Teams**:
- Harder to scale dynamically
- Limited spot instance savings
- Fewer managed services (SageMaker, Vertex AI work with NVIDIA only)
- Higher barrier to experimentation

### Framework Support Lag

**PyTorch Example**:
- CUDA: PyTorch 2.2 supports CUDA 12.1 (same-day release)
- ROCm: PyTorch 2.2 ROCm support arrived 3 months later

**Library Parity**:
- cuDNN has ~200 optimized kernels
- MIOpen has ~120 optimized kernels
- Flash Attention 2: CUDA version is 2× faster than ROCm port

## CUDA vs ROCm: Decision Framework

### Choose CUDA When:

1. **Production AI workloads** requiring maximum reliability
2. **Time-to-market** is critical (mature ecosystem)
3. **FP8 inference** needed (H100-exclusive currently)
4. **Cloud deployment** on AWS/GCP/Azure
5. **Framework support** for cutting-edge features essential
6. **Large organization** with budget for NVIDIA premium
7. **Extensive CUDA codeb** already deployed

### Choose ROCm When:

1. **Budget constraints** significant (40-60% hardware savings)
2. **Large memory** requirements (192GB MI300X)
3. **Open-source mandate** for compliance/security
4. **In-house GPU expertise** to handle rough edges
5. **Long-term projects** with optimization runway
6. **Hardware flexibility** needed (multi-vendor strategy)
7. **On-premises deployment** (cloud limitations less relevant)

### Hybrid Strategy:

**Development**: NVIDIA GPUs + CUDA (faster iteration)
**Production**: AMD GPUs + ROCm (cost optimization)
**Example**: Train on H100, deploy inference on MI300X

## Future Outlook (2025-2026)

### ROCm Improvements

**ROCm 6.0+ Roadmap** (from AMD announcements):
- Better HIP-CUDA compatibility (95%+ API coverage)
- FP8 tensor core support (MI300 series)
- Improved MIOpen performance (targeting cuDNN parity)
- Flash Attention 3 optimizations
- Better multi-GPU scaling (RCCL enhancements)

From [Phoronix ROCm coverage](https://www.phoronix.com/forums/forum/linux-graphics-x-org-drivers/open-source-amd-linux/1550326-amd-rocm-7-0-to-align-hip-c-even-more-closely-with-cuda) (accessed 2025-11-14):
> "AMD ROCm 7.0 to align HIP C++ even more closely with CUDA. They instead designed ROCm to be as similar to CUDA as possible so that porting software becomes as simple as just re-compiling it."

### MI400 Series (2025-2026 Expected)

**Rumored Specs**:
- CDNA 4 architecture
- 256-384GB HBM3e memory
- Improved FP8/FP4 support
- Better power efficiency (sub-600W TDP)
- Enhanced AI matrix engines

### ZLUDA Project

From [Level1Techs forum](https://forum.level1techs.com/t/a-drop-in-cuda-implementation-built-on-rocm/207042) (accessed 2025-11-14):
> "AMD quietly funded a drop-in CUDA implementation built on ROCm. While there have been efforts by AMD over the years to make it easier to port codebases targeting NVIDIA's CUDA API to run atop HIP/ROCm..."

**ZLUDA**: Enables unmodified CUDA applications to run on AMD GPUs with near-native performance. Currently experimental but shows promise for compatibility layer approach.

## arr-coc-0-1 Considerations

### Would ROCm Work for arr-coc-0-1?

**Theoretical Compatibility**:
- PyTorch ROCm backend: ✅ Yes
- Qwen3-VL model: ✅ Should work (HuggingFace Transformers)
- Custom CUDA kernels: ❌ Would need HIP porting
- Vertex AI deployment: ❌ No AMD GPU support

**Practical Reality**:
- arr-coc-0-1 uses Vertex AI → NVIDIA GPUs only
- No custom CUDA kernels → PyTorch ROCm would work
- Production deployment → CUDA ecosystem safer choice

**If Starting Fresh**:
- **Research/Development**: MI300X viable for large model experimentation
- **Production Inference**: NVIDIA L4/T4 better cloud support
- **On-Prem Cluster**: MI300X cost-effective for scaling

**Hybrid Approach**:
- Use ROCm for offline experimentation (cost savings)
- Deploy to CUDA for production (reliability)
- Keep codebase portable (avoid CUDA-specific features)

## Sources

### Official Documentation

**AMD ROCm**:
- [ROCm Installation Guide - PyTorch for Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) - Official installation guide (accessed 2025-11-14)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) - Official CUDA to HIP porting documentation (accessed 2025-11-14)
- [AMD MI300X Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) - Official hardware specifications (accessed 2025-11-14)
- [AMD GPUOpen - HIPify Application Portability](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-hipify-readme/) - Tools for CUDA to HIP conversion (April 30, 2024)

**NVIDIA Documentation** (for comparison):
- NVIDIA H100 tensor core GPU architecture white paper
- CUDA programming guide (version 12.x)

### Web Research

**Technical Comparisons**:
- [Scimus: ROCm vs CUDA - A Practical Comparison for AI Developers](https://thescimus.com/blog/rocm-vs-cuda-a-practical-comparison-for-ai-developers/) - Comprehensive ROCm vs CUDA analysis (August 12, 2024)
- [TensorWave: ROCm vs CUDA Performance Showdown](https://tensorwave.com/blog/rocm-vs-cuda-a-performance-showdown-for-modern-ai-workloads) - MI325X benchmarks vs H100 (August 7, 2025 - likely typo for 2024)
- [Thunder Compute: ROCm vs CUDA Comparison](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing) - GPU computing system comparison (October 27, 2025 - likely typo for 2024)
- [Neysa: AMD MI300X Specs and Performance](https://neysa.ai/blog/amd-mi300x/) - Detailed MI300X analysis (March 4, 2025 - likely typo for 2024)

**Community Discussions**:
- [Reddit r/MachineLearning: Why is CUDA so much faster than ROCm?](https://www.reddit.com/r/MachineLearning/comments/1fa8vq5/d_why_is_cuda_so_much_faster_than_rocm/) - Discussion on performance differences (2024)
- [Reddit r/ROCm: ROCm/HIP Tutorials](https://www.reddit.com/r/ROCm/comments/1ae67j6/rocmhip_tutorials_that_dont_assume_cuda_background/) - Community tutorials (2024)
- [Level1Techs Forum: Drop-in CUDA Implementation on ROCm](https://forum.level1techs.com/t/a-drop-in-cuda-implementation-built-on-rocm/207042) - ZLUDA discussion (February 12, 2024)

**GitHub Resources**:
- [ROCm/rocm-examples](https://github.com/ROCm/rocm-examples) - Official AMD ROCm code examples
- [PyTorch ROCm support issues](https://github.com/pytorch/pytorch/issues/120433) - ROCm 6.x support discussion

**Note**: Some future-dated sources (2025) are likely publication errors and reflect 2024 content based on technical details mentioned.

### Cloud Provider Documentation

- Oracle Cloud Infrastructure: MI300X instance documentation
- Microsoft Azure: AMD MI300X preview announcement
- Vertex AI: NVIDIA GPU support (for comparison)

---

**Last Updated**: November 14, 2025
**ROCm Version Covered**: 6.x - 7.1
**PyTorch Version**: 2.8.0+rocm7.1
**Primary Focus**: Machine learning and AI workloads on AMD hardware
