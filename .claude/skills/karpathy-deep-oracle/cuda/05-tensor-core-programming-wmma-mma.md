# Tensor Core Programming: WMMA API and PTX mma.sync

## Overview

Tensor Cores are specialized hardware units in NVIDIA GPUs that accelerate matrix multiply-accumulate (MMA) operations, providing groundbreaking performance for AI and HPC workloads. This document covers programming Tensor Cores using the high-level WMMA API and low-level PTX mma.sync instructions, with focus on NVIDIA Hopper (H100), Ampere (A100), and earlier architectures.

**Key Concepts:**
- WMMA (Warp Matrix Multiply-Accumulate) provides warp-level matrix operations
- PTX mma.sync instructions offer fine-grained control over Tensor Core execution
- Tensor Cores deliver 10-20× speedup over standard CUDA cores for matrix operations
- Multiple precision formats supported: FP8, FP16, BF16, TF32, FP64, INT8

From [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (accessed 2025-02-03):
- H100 fourth-generation Tensor Cores: 6× faster than A100 chip-to-chip
- Per-SM: 2× MMA rates on equivalent datatypes, 4× with new FP8
- Sparsity feature doubles performance on structured sparse networks

From [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d) (accessed 2025-02-03):
- PTX is low-level parallel thread execution virtual machine and ISA
- MMA PTX enables mixed-precision D = A×B + C operations
- Available on compute capability 7.0+ (Volta and later)

---

## Section 1: Tensor Core Architecture (~100 lines)

### What Are Tensor Cores?

Tensor Cores are specialized compute cores for matrix multiply and accumulate operations. They operate in parallel across SMs to deliver massive throughput increases compared to standard FP/INT/FMA operations.

**Matrix Operation:**
```
D = A × B + C
```
Where:
- A, B: Input matrices
- C: Accumulator (input)
- D: Output accumulator

**Key Characteristics:**
- Warp-level or warpgroup-level operation (32 or 128 threads)
- Fixed tile sizes (e.g., 16×16×16, 64×64×16)
- Support for multiple precision formats
- Automatic handling of complex data layouts

### Tensor Core Generations

From [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):

**Generation 1 (Volta, sm_70):**
- First Tensor Core implementation
- FP16 input, FP32 accumulate
- 16×16×16 tile size

**Generation 2 (Turing, sm_75):**
- T4: 65 TFLOPs FP16
- INT8, INT4, binary precision support
- Same 16×16×16 tile for FP16

**Generation 3 (Ampere, sm_80/sm_86):**
- A100: 312 TFLOPs FP16, 156 TFLOPs TF32
- New TF32 precision (8-bit exponent, 10-bit mantissa)
- BF16 support for better dynamic range
- FP64 Tensor Cores (19.5 TFLOPs on A100)
- Structured sparsity (2:4 pattern, 2× throughput)

**Generation 4 (Hopper, sm_90):**
- H100: 2000 TFLOPs FP8, 1000 TFLOPs FP16
- New FP8 formats: E4M3 (4 exp, 3 mantissa), E5M2 (5 exp, 2 mantissa)
- Warpgroup operations (128 threads vs 32)
- Asynchronous execution model
- 2× raw throughput per SM over A100 (same precision)

### Performance Benefits

From [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md):

**A100 Tensor Core Specs:**
- FP16: 312 TFLOPs → 624 TFLOPs (sparse)
- TF32: 156 TFLOPs → 312 TFLOPs (sparse)
- FP32 (non-Tensor): 19.5 TFLOPs
- Ratio: 16× faster for FP16, 8× for TF32

**H100 Tensor Core Specs:**
- FP8: 2000 TFLOPs → 4000 TFLOPs (sparse)
- FP16/BF16: 1000 TFLOPs → 2000 TFLOPs (sparse)
- TF32: 500 TFLOPs → 1000 TFLOPs (sparse)
- FP32 (non-Tensor): 60 TFLOPs
- Ratio: 33× faster for FP8, 16× for FP16

**Speedup Calculation:**
- H100 FP8 vs A100 FP16: 6.4× improvement
- H100 FP16 vs A100 FP16: 3.2× improvement
- Includes SM count increase + clock boost + architectural improvements

---

## Section 2: WMMA API Programming (~200 lines)

### WMMA API Overview

WMMA (Warp Matrix Multiply-Accumulate) provides warp-level matrix operations targeting Tensor Cores. Available in `<mma.h>` header.

**Key WMMA Operations:**
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 1. Declare fragments (register storage)
fragment<matrix_a, M, N, K, half, row_major> a_frag;
fragment<matrix_b, M, N, K, half, col_major> b_frag;
fragment<accumulator, M, N, K, float> c_frag;

// 2. Load data from memory
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);
fill_fragment(c_frag, 0.0f);

// 3. Matrix multiply-accumulate
mma_sync(c_frag, a_frag, b_frag, c_frag);

// 4. Store result
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```

### Fragment Types

From PTX documentation:

**Template Parameters:**
```cpp
template<Use, m, n, k, T, Layout>
fragment<...>;
```

**Use:**
- `matrix_a`: Left operand (M×K)
- `matrix_b`: Right operand (K×N)
- `accumulator`: Accumulator/output (M×N)

**Supported Shapes (Ampere/Hopper):**
- 16×16×16 (FP16, BF16, TF32)
- 8×8×4 (FP64)
- 16×8×8, 16×8×16 (various precisions)

**Layouts:**
- `row_major`: Rows contiguous in memory
- `col_major`: Columns contiguous in memory

### Complete WMMA GEMM Example

```cpp
// Compute C = A * B using WMMA
// A: M×K (row-major), B: K×N (col-major), C: M×N (row-major)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_kernel(
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Compute tile coordinates
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;

    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aCol = i;
        int bRow = i;

        // Bounds check
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result
    if (aRow < M && bCol < N) {
        wmma::store_matrix_sync(C + aRow * N + bCol, c_frag, N,
                                wmma::mem_row_major);
    }
}
```

### WMMA Data Layout

From [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d):

**Fragment Distribution (16×16×16):**
- Each warp (32 threads) computes one 16×16 output tile
- Each thread holds multiple elements (not one element!)
- Elements distributed in specific pattern across threads

**Thread Ownership Pattern:**
- Thread owns non-contiguous elements in output
- Enables efficient warp-level reduction
- Pattern varies by precision and architecture

**Key Insight:**
Fragments are **not** simple arrays. Layout is architecture-specific and optimized for Tensor Core hardware. Always use WMMA API methods for access.

### Mixed Precision Operations

```cpp
// FP16 input, FP32 accumulate (most common)
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;

// BF16 input, FP32 accumulate (better range)
fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;

// TF32 (Ampere+, automatic precision reduction)
// Declared as FP32, executed as TF32 on Tensor Cores
fragment<matrix_a, 16, 16, 8, float, row_major> a_frag;
fragment<matrix_b, 16, 16, 8, float, col_major> b_frag;
fragment<accumulator, 16, 16, 8, float> acc_frag;
```

---

## Section 3: PTX mma.sync Programming (~150 lines)

### PTX MMA Overview

PTX (Parallel Thread Execution) mma.sync instructions provide low-level control over Tensor Cores. Required for architectures without WMMA API support or when fine-grained control needed.

From [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d):

**PTX Instruction Format:**
```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16  d, a, b, c;
```

**Components:**
- `mma.sync`: Synchronous matrix multiply-accumulate
- `aligned`: Memory alignment requirement
- `m16n8k16`: Tile shape (M=16, N=8, K=16)
- `row.col`: Layout of A (row-major), B (col-major)
- `f16.f16.f16`: Input A, input B, accumulator C types
- `d, a, b, c`: Output, operand A, operand B, accumulator C

### PTX Fragment Layout

From [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d):

**m16n8k16 FP16 Element Distribution:**

Matrix A fragment (16×16):
```
groupID = laneid >> 2          // Warp divided into 8 groups of 4 threads
threadID_in_group = laneid % 4

row = groupID           for elements 0,1,4,5
      groupID + 8       for elements 2,3,6,7

col = (threadID_in_group * 2) + (i & 0x1)       for elements 0-3
      (threadID_in_group * 2) + (i & 0x1) + 8   for elements 4-7
```

Matrix B fragment (16×8):
```
row = (threadID_in_group * 2) + (i & 0x1)       for elements 0,1
      (threadID_in_group * 2) + (i & 0x1) + 8   for elements 2,3

col = groupID
```

Accumulator C/D fragment (16×8):
```
row = groupID           for elements 0,1
      groupID + 8       for elements 2,3

col = (threadID_in_group * 2) + (i & 0x1)
```

### PTX MMA Example

```cpp
// Matrix multiply using PTX inline assembly
__device__ void mma_m16n8k16_fp16(
    uint32_t* d,      // Output (4 registers)
    uint32_t* a,      // Matrix A (4 registers)
    uint32_t* b,      // Matrix B (2 registers)
    uint32_t* c       // Accumulator (4 registers)
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}
```

### LDMATRIX PTX Instruction

From [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d):

LDMATRIX cooperatively loads data from shared memory to registers in layout compatible with mma.sync.

**Syntax:**
```
ldmatrix.sync.aligned.m8n8.x4.shared.b16  r, [p];
```

**Key Features:**
- Warp-level operation (all 32 threads participate)
- Loads from shared memory only
- 8 threads provide 8 addresses (one per matrix row)
- Each thread loads fragments for one row
- Output layout matches mma.sync requirements

**Example:**
```cpp
__device__ void load_matrix_sync_ptx(
    uint32_t* regs,           // Output registers
    const void* smem_ptr      // Shared memory pointer
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]),
          "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr)
    );
}
```

**Thread Participation:**
- Threads 0-7: Load first 8×8 matrix
- Threads 8-15: Load second 8×8 matrix
- For .x4: All 32 threads load four 8×8 matrices

---

## Section 4: Precision Formats and Support (~150 lines)

### FP16 (Half Precision)

**Format:** 1 sign, 5 exponent, 10 mantissa bits
**Range:** 6e-5 to 65504
**Precision:** ~3 decimal digits

**Characteristics:**
- Standard half-precision IEEE 754
- Supported since Volta (Gen 1)
- Good balance of range and precision for DNNs
- Gradient underflow risk in training

**Usage:**
```cpp
// WMMA FP16
fragment<matrix_a, 16, 16, 16, half, row_major> a;
fragment<matrix_b, 16, 16, 16, half, col_major> b;
fragment<accumulator, 16, 16, 16, float> c;  // FP32 accumulator

mma_sync(c, a, b, c);
```

### BF16 (Brain Float 16)

**Format:** 1 sign, 8 exponent, 7 mantissa bits
**Range:** Same as FP32 (1.2e-38 to 3.4e38)
**Precision:** ~2 decimal digits

**Characteristics:**
- Truncated FP32 (same exponent range)
- Supported since Ampere (Gen 3)
- Better for training (no gradient scaling needed)
- Simpler conversion to/from FP32

**Usage:**
```cpp
// WMMA BF16
fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
fragment<accumulator, 16, 16, 16, float> c;

mma_sync(c, a, b, c);
```

### TF32 (TensorFloat-32)

From [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):

**Format:** 1 sign, 8 exponent, 10 mantissa bits
**Range:** Same as FP32
**Precision:** Between FP16 and FP32

**Characteristics:**
- Ampere+ only (Gen 3+)
- Automatic in matmul operations when enabled
- No code changes needed
- ~10× speedup over FP32, minimal accuracy loss

**Enabling TF32:**
```cpp
// Enable TF32 (default on Ampere+)
torch::backends::cuda::matmul::allow_tf32 = true;
torch::backends::cudnn::allow_tf32 = true;

// In CUDA
// Happens automatically for FP32 WMMA/MMA on Ampere+
```

### FP8 (8-bit Float, Hopper Only)

From [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):

**Two Formats:**

**E4M3:** 1 sign, 4 exponent, 3 mantissa bits
- More precision, less range
- Suitable for weights, activations

**E5M2:** 1 sign, 5 exponent, 2 mantissa bits
- More range, less precision
- Suitable for gradients

**Characteristics:**
- Hopper only (Gen 4, sm_90)
- 2× throughput vs FP16
- Half the memory footprint
- Requires careful scaling (Transformer Engine)

**Performance:**
```
H100 FP8: 2000 TFLOPs (4000 sparse)
H100 FP16: 1000 TFLOPs (2000 sparse)
Speedup: 2× raw, 4× effective with better occupancy
```

### FP64 (Double Precision)

From [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):

**Characteristics:**
- Ampere+ Tensor Cores support FP64
- Primarily for HPC applications
- A100: 19.5 TFLOPs FP64 Tensor Core (9.7 non-Tensor)
- H100: 60 TFLOPs FP64 Tensor Core (30 non-Tensor)

**Usage:**
```cpp
// WMMA FP64 (8x8x4 tiles on Ampere)
fragment<matrix_a, 8, 8, 4, double, row_major> a;
fragment<matrix_b, 8, 8, 4, double, col_major> b;
fragment<accumulator, 8, 8, 4, double> c;

mma_sync(c, a, b, c);
```

### INT8 Precision

**Characteristics:**
- Inference-focused (quantized models)
- Supported since Turing (Gen 2)
- A100: 624 TOPS INT8
- H100: 2000 TOPS INT8

**Use Cases:**
- Quantized neural network inference
- Lower precision CNNs
- Edge deployment optimization

---

## Section 5: Verification and Profiling (~100 lines)

### Verifying Tensor Core Usage

**Method 1: CUDA Occupancy Calculator**
```bash
# Check if Tensor Cores are active
nvcc -arch=sm_80 -Xptxas -v kernel.cu
# Look for mma.sync or wmma instructions in PTX
```

**Method 2: Nsight Compute (ncu)**

From [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md):

```bash
# Profile kernel with Tensor Core metrics
ncu --set full --metrics \
  smsp__sass_thread_inst_executed_op_hmma_,\
  smsp__sass_thread_inst_executed_op_imma_,\
  smsp__inst_executed_pipe_tensor \
  ./my_kernel

# Key metrics:
# - smsp__sass_thread_inst_executed_op_hmma_*: Half/BFloat Tensor Core ops
# - smsp__sass_thread_inst_executed_op_imma_*: Integer Tensor Core ops
# - smsp__inst_executed_pipe_tensor: Total Tensor Core instructions
```

**Method 3: Check SASS Disassembly**
```bash
# Dump SASS assembly
cuobjdump -sass kernel.cubin > kernel.sass

# Look for HMMA (half), IMMA (int), DMMA (double) instructions
# Example SASS:
# HMMA.884.F16.F16 R4, R0.ROW, R2.COL, R4;
```

### Tensor Core Utilization Metrics

**Target Metrics:**
- Tensor Core utilization: >80% ideal
- Warp execution efficiency: >90%
- Achieved occupancy: 50-75% (balance with register pressure)

**Common Bottlenecks:**
- Memory bandwidth (HBM)
- Shared memory bank conflicts
- Register pressure limiting occupancy
- Insufficient parallel work (small batch size)

### Performance Targets

From [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):

**A100 Peak Performance:**
- FP16 Tensor Core: 312 TFLOPs
- TF32 Tensor Core: 156 TFLOPs
- FP32 (non-Tensor): 19.5 TFLOPs
- Memory Bandwidth: 1.5 TB/s (40GB), 2.0 TB/s (80GB)

**H100 Peak Performance:**
- FP8 Tensor Core: 2000 TFLOPs
- FP16 Tensor Core: 1000 TFLOPs
- TF32 Tensor Core: 500 TFLOPs
- FP32 (non-Tensor): 60 TFLOPs
- Memory Bandwidth: 3.0 TB/s

**Efficiency Calculation:**
```python
# Arithmetic intensity (ops/byte)
AI = (2 * M * N * K) / ((M*K + K*N + M*N) * sizeof(dtype))

# For square matrices M=N=K=4096, FP16:
# AI = 2*4096^3 / (3*4096^2*2) = ~2731 ops/byte

# Memory-bound threshold on A100 (FP16):
# 1555 GB/s * 2731 ops/byte = 4.25 PFLOPs
# But Tensor Core peak = 0.312 PFLOPs
# This problem is compute-bound!
```

### Profiling with Nsight Systems

```bash
# Capture timeline with Tensor Core activity
nsys profile --stats=true \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  ./my_program

# Look for:
# - Tensor Core kernel execution time
# - Memory transfer overlaps
# - GPU utilization percentage
# - Kernel launch overhead
```

---

## Section 6: Hopper WGMMA Instructions (~150 lines)

### Warpgroup Matrix Multiply-Accumulate

From [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) (accessed 2025-02-03):

**Key Differences from WMMA:**
- **Warpgroup:** 128 threads (4 warps) vs 32 threads (1 warp)
- **Asynchronous:** Can overlap with other operations
- **SMEM-only:** Operand B must be in shared memory
- **Larger tiles:** 64×64×16 and larger supported

**WGMMA Instruction Format:**
```
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
  {d0, d1, ..., d15},  // 16 output registers
  desc_a,               // Matrix A descriptor (64-bit)
  desc_b,               // Matrix B descriptor (64-bit)
  scale_d;              // Scale factor for accumulator
```

### Matrix Descriptors

From [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/):

**Descriptor Components:**
- Start address: Base address in SMEM
- LBO (Leading dimension Byte Offset): Distance between K-adjacent core matrices
- SBO (Stride dimension Byte Offset): Distance between M/N-adjacent core matrices
- Swizzle mode: None, 32B, 64B, or 128B
- Matrix base offset: SMEM alignment correction

**Creating Descriptors:**
```cpp
// CUTLASS creates descriptors automatically from SMEM tensors
auto sA = cute::tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<half_t>{},
    cute::make_shape(128, 64, 3)  // M, K, stages
);

// Descriptor created by partitioning with TiledMMA
ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // Creates descriptor
```

### Core Matrices

From [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/):

**Core Matrix Definition:**
- Small matrix building block
- 8 elements in strided direction
- 16 bytes in contiguous direction

**Example: 64×16 matrix A (K-major):**
- Divided into 8×2 = 16 core matrices
- Each core matrix: 8 rows × (16 bytes / sizeof(dtype)) columns
- LBO: Distance in bytes between K-adjacent cores
- SBO: Distance in bytes between M-adjacent cores

**Swizzle Modes:**
```cpp
// No swizzle: 16-byte boundary
GMMA::Layout_K_INTER_Atom<T>{}

// 32-byte swizzle: 2 consecutive 16-byte segments
GMMA::Layout_K_SW32_Atom<T>{}

// 64-byte swizzle: 4 consecutive 16-byte segments
GMMA::Layout_K_SW64_Atom<T>{}

// 128-byte swizzle: 8 consecutive 16-byte segments
GMMA::Layout_K_SW128_Atom<T>{}
```

### Asynchronous Execution and Synchronization

From [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/):

**Synchronization Primitives:**
```cpp
// 1. Fence before WGMMA (ensures SMEM writes complete)
cute::warpgroup_arrive();

// 2. Issue WGMMA instructions
cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

// 3. Commit wgmma-group (batch all prior WGMMA)
cute::warpgroup_commit_batch();

// 4. Wait for completion (N=0 means wait for all)
cute::warpgroup_wait<0>();
```

**PTX Equivalents:**
- `warpgroup_arrive()` → `wgmma.fence.sync.aligned`
- `warpgroup_commit_batch()` → `wgmma.commit_group.sync.aligned`
- `warpgroup_wait<N>()` → `wgmma.wait_group.sync.aligned N`

**Synchronization Rules:**
1. `wgmma.fence` ensures prior RMEM/SMEM writes are visible
2. `fence.proxy.async` needed if generic proxy writes to SMEM (not needed with TMA)
3. `wgmma.commit_group` batches all uncommitted WGMMA into a group
4. `wgmma.wait_group N` waits until ≤N groups are pending

### WGMMA Example (Conceptual)

```cpp
// Hopper GEMM mainloop using WGMMA
__global__ void hopper_gemm_wgmma(
    TiledMMA tiled_mma,
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    // Shared memory for A and B tiles
    __shared__ half sA[128][64];  // With swizzle
    __shared__ half sB[128][64];  // With swizzle

    // Get thread slice (0-127 for warpgroup)
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // Partition SMEM and create descriptors
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // Descriptor
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // Descriptor

    // Allocate accumulator
    Tensor tCrC = thr_mma.make_fragment_C(...);
    clear(tCrC);

    // Loop over K
    for (int k = 0; k < K; k += 16) {
        // Load tiles into SMEM (using TMA or cp.async)
        // ...

        // Synchronize warpgroup
        cute::warpgroup_arrive();

        // Matrix multiply-accumulate
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

        // Commit and wait
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<0>();
    }

    // Store results
    // ...
}
```

---

## Section 7: ARR-COC Relevance Scorer Optimization (~100 lines)

### Current ARR-COC Architecture

From [arr-coc architecture](../../README.md):

**Relevance Scoring Pipeline:**
```python
# Three ways of knowing (Propositional, Perspectival, Participatory)
# Each computes attention-like scores over patch embeddings

def compute_relevance(query_emb, patch_embs):
    # query_emb: [B, D]
    # patch_embs: [B, N_patches, D]

    # Propositional: Information content (entropy-based)
    info_scores = compute_entropy(patch_embs)  # [B, N_patches]

    # Perspectival: Salience (learned importance)
    salience = salience_net(patch_embs)  # [B, N_patches, 1]

    # Participatory: Query-content coupling
    # THIS IS THE TENSOR CORE TARGET
    coupling = query_emb @ patch_embs.T  # [B, N_patches]

    return combine_scores(info_scores, salience, coupling)
```

### Tensor Core Optimization Opportunities

**Current Bottleneck:**
```python
# Participatory scorer: Query × Patch attention
# Shape: [B, D] @ [B, D, N] = [B, N]
# For B=32, D=1024, N=200 patches
# Operations: 32 × 1024 × 200 = 6.5M ops
# With 200 patches × 64-400 tokens = 12.8K-80K tokens
```

**Optimization 1: Batch Matrix Multiply with WMMA**
```cpp
// Use WMMA for query-patch attention scores
// Reframe as: [B, 1, D] @ [B, D, N] = [B, 1, N]

template<int D, int N>
__global__ void participatory_scorer_wmma(
    const half* query,      // [B, D]
    const half* patches,    // [B, N, D]
    float* scores,          // [B, N]
    int B
) {
    // Each block handles one batch
    int batch = blockIdx.x;

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> p_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

    wmma::fill_fragment(s_frag, 0.0f);

    // Loop over D dimension in chunks of 16
    for (int k = 0; k < D; k += 16) {
        // Load query: 1×16 broadcasted to 16×16
        wmma::load_matrix_sync(q_frag,
            query + batch * D + k, D);

        // Load patches: 16×16
        wmma::load_matrix_sync(p_frag,
            patches + batch * N * D + k * N, N);

        // Accumulate
        wmma::mma_sync(s_frag, q_frag, p_frag, s_frag);
    }

    // Store scores
    wmma::store_matrix_sync(
        scores + batch * N, s_frag, N,
        wmma::mem_row_major
    );
}
```

**Optimization 2: Fused Opponent Processing**
```cpp
// Combine multiple scorers in one Tensor Core kernel
// Compute Propositional + Perspectival + Participatory together

__global__ void fused_relevance_scorer(
    const half* query,          // [B, D]
    const half* patches,        // [B, N, D]
    const float* entropy,       // [B, N] (precomputed)
    const half* salience_w,     // [D, 1] (learned weights)
    float* combined_scores,     // [B, N]
    int B, int N, int D
) {
    // Use Tensor Cores for:
    // 1. Participatory: query @ patches.T
    // 2. Perspectival: patches @ salience_weights
    // Then combine with entropy (Propositional)

    // WMMA for participatory
    // ... (as above)

    // WMMA for perspectival
    // ...

    // Combine all three scores with opponent processing weights
    // Exploit Tensor Core for weighted combination
}
```

**Optimization 3: Variable LOD with Tensor Cores**
```cpp
// For each patch, compute at different LODs (64-400 tokens)
// Use Tensor Core's mixed precision for LOD hierarchy

__global__ void hierarchical_lod_scorer(
    const half* query,          // [B, D]
    const half* patch_64,       // [B, N, 64, D_64]
    const half* patch_128,      // [B, N, 128, D_128]
    const half* patch_256,      // [B, N, 256, D_256]
    const half* patch_400,      // [B, N, 400, D_400]
    float* scores,              // [B, N, 4] (one per LOD)
    int B, int N
) {
    // Compute attention scores for all LODs
    // Use FP16 Tensor Cores for speed
    // Accumulate in FP32 for stability

    // Then select LOD based on combined relevance score
}
```

### Expected Performance Gains

**A100 (sm_80) Performance:**
- Standard attention: ~100-200 GFLOPS (FP32 CUDA cores)
- WMMA FP16 with FP32 accumulate: ~2-3 TFLOPS
- Speedup: **10-30×**

**H100 (sm_90) Performance:**
- WGMMA FP16: ~8-10 TFLOPS
- WGMMA FP8 (if precision sufficient): ~15-20 TFLOPS
- Speedup vs A100 standard: **50-100×**
- Speedup vs A100 WMMA: **3-5×**

**Memory Bandwidth Savings:**
```
# Standard FP32: 4 bytes/element
Query: 32 × 1024 × 4 = 128 KB
Patches: 32 × 200 × 1024 × 4 = 25.6 MB
Total: ~25.7 MB

# WMMA FP16: 2 bytes/element
Query: 32 × 1024 × 2 = 64 KB
Patches: 32 × 200 × 1024 × 2 = 12.8 MB
Total: ~12.9 MB

Bandwidth saving: 2× (plus higher compute throughput!)
```

---

## Sources

### Source Documents

- [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) - Lines 36-99: Tensor Core specifications and memory hierarchy
- [karpathy/practical-implementation/73-cuda-cooperative-groups.md](../karpathy/practical-implementation/73-cuda-cooperative-groups.md) - Mentioned custom kernels with Tensor Cores

### Web Research

**Primary Sources:**

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (accessed 2025-02-03)
  - Fourth-generation Tensor Cores specifications
  - FP8 precision formats and Transformer Engine
  - Performance comparisons: A100 vs H100
  - WGMMA asynchronous execution model

- [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d) (accessed 2025-02-03)
  - PTX mma.sync instruction syntax
  - Fragment layout and thread ownership patterns
  - LDMATRIX instruction for data loading
  - Complete HGEMM example with PTX

- [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) (accessed 2025-02-03)
  - Warpgroup matrix multiply-accumulate
  - Matrix descriptors and core matrices
  - Asynchronous execution and synchronization
  - SMEM layout constraints and swizzle modes

**Additional References:**

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) - NVIDIA official PTX reference (too large to scrape, referenced in other sources)
- NVIDIA GPU Architecture whitepapers (Ampere, Hopper) - Referenced in blog posts
- CUTLASS library source code - Mentioned as implementation reference

---

## Practical Recommendations

### When to Use Tensor Cores

**✓ Use Tensor Cores when:**
- Matrix multiply dominates computation (GEMM, attention, convolution)
- Problem size allows 16×16 or larger tiles
- Mixed precision acceptable (FP16/BF16 input, FP32 accumulate)
- High arithmetic intensity (compute-bound)

**✗ Avoid Tensor Cores when:**
- Small matrices (M,N,K < 16)
- Element-wise operations only
- Requires full FP32 precision throughout
- Memory-bound workload (data movement dominates)

### WMMA vs PTX mma.sync

**Use WMMA API:**
- Portable across architectures (sm_70+)
- Simpler programming model
- Automatic layout handling
- Good for most applications

**Use PTX mma.sync:**
- Need architecture-specific optimization
- Custom memory layouts required
- Maximum performance critical
- Integrating with hand-written assembly

### Precision Selection

**FP16 (Half):**
- Standard for inference
- Good training with gradient scaling
- Supported since Volta (sm_70)

**BF16 (Brain Float):**
- Preferred for training (better range)
- No gradient scaling needed
- Ampere+ (sm_80)

**TF32 (Tensor Float):**
- Drop-in FP32 replacement
- Automatic on Ampere+ matmul
- ~10× speedup, minimal loss

**FP8 (8-bit Float):**
- Hopper only (sm_90)
- Requires Transformer Engine for scaling
- 2× throughput over FP16
- Best for large language models

### ARR-COC Integration Strategy

**Phase 1: Baseline WMMA**
- Implement participatory scorer with WMMA FP16
- Measure speedup vs FP32 baseline
- Target: 10-20× improvement on A100

**Phase 2: Fused Operations**
- Combine all three scorers in one kernel
- Reduce memory traffic
- Target: 2-3× additional speedup

**Phase 3: Hopper WGMMA**
- Port to H100 with warpgroup operations
- Exploit asynchronous execution
- Target: 3-5× over A100 WMMA

**Phase 4: FP8 Exploration**
- Evaluate FP8 precision for relevance scoring
- Implement Transformer Engine scaling
- Target: 2× over FP16 WGMMA
