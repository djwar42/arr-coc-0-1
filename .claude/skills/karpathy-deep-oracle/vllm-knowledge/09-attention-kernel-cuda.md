# CUDA Attention Kernels: Implementation and Optimization

## Overview

CUDA kernels for attention mechanisms are critical performance components in modern LLM serving systems. This guide covers CUDA programming fundamentals, memory optimization techniques, Tensor Core utilization, and specific implementations for FlashAttention and PagedAttention used in production systems like vLLM.

From [FlashAttention GitHub Repository](https://github.com/Dao-AILab/flash-attention) (accessed 2025-02-02):
- FlashAttention is fast and memory-efficient exact attention with IO-awareness
- Optimized CUDA kernels including integration with FlashAttention and FlashInfer
- Reduces memory reads/writes by avoiding materialization of large intermediate attention matrices
- 2-4x wallclock speedup through tiling and recomputation strategies

From [FlashAttention-3 Blog Post](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):
- FlashAttention-3 achieves 1.5-2.0x faster than FlashAttention-2 with FP16
- Reaches up to 740 TFLOPS (75% utilization of H100 theoretical max FLOPS)
- With FP8, FlashAttention-3 reaches close to 1.2 PFLOPS

## CUDA Kernel Basics

### Kernel Launch and Thread Organization

CUDA kernels execute in a grid of thread blocks. Understanding this hierarchy is fundamental to writing efficient kernels.

**Thread Hierarchy:**
```
Grid (entire kernel launch)
├── Block 0
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ...
├── Block 1
│   └── Warps...
└── ...
```

**Basic Kernel Structure:**
```cuda
__global__ void attention_kernel(
    float* Q,           // Query: [batch, seqlen, heads, headdim]
    float* K,           // Key: [batch, seqlen, heads, headdim]
    float* V,           // Value: [batch, seqlen, heads, headdim]
    float* output,      // Output: [batch, seqlen, heads, headdim]
    int seqlen,
    int headdim
) {
    // Thread identification
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Kernel logic here
}
```

**Launch Configuration:**
```cuda
// Calculate grid and block dimensions
dim3 block(256);  // 256 threads per block
dim3 grid(
    (seqlen + block.x - 1) / block.x,  // Sequence blocks
    num_heads,                          // Head dimension
    batch_size                          // Batch dimension
);

attention_kernel<<<grid, block>>>(Q, K, V, output, seqlen, headdim);
```

### Memory Hierarchy

From [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (accessed 2025-02-02):

**GPU Memory Types (fastest to slowest):**

1. **Registers** (per-thread, ~20 TB/s)
   - Fastest storage, limited quantity
   - Automatic variables stored here
   - Spilling to local memory if exhausted

2. **Shared Memory** (per-block, ~15 TB/s)
   - Explicitly managed cache
   - Shared among threads in a block
   - 48-164 KB per SM depending on GPU

3. **L1/L2 Cache** (hardware-managed)
   - Automatic caching of global memory
   - ~5-10 TB/s effective bandwidth

4. **Global Memory** (HBM, ~1-3 TB/s)
   - Large capacity (40-80 GB on datacenter GPUs)
   - High latency (~hundreds of cycles)
   - Main storage for inputs/outputs

**Memory Declaration:**
```cuda
__global__ void kernel() {
    // Register variables
    float reg_var = 0.0f;

    // Shared memory (static)
    __shared__ float smem_static[256];

    // Shared memory (dynamic, size specified at launch)
    extern __shared__ float smem_dynamic[];

    // Global memory access
    float global_val = global_array[idx];
}
```

## Memory Coalescing

From [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) (accessed 2025-02-02):

Memory coalescing enables hardware to combine multiple memory accesses into fewer transactions, dramatically improving bandwidth utilization.

### Coalesced Access Pattern

**Good (Coalesced):**
```cuda
// Sequential access - threads access consecutive addresses
__global__ void coalesced_read(float* data, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Thread 0 reads data[0], thread 1 reads data[1], etc.
        output[idx] = data[idx];  // Coalesced!
    }
}
```

In this pattern:
- Thread 0 accesses address 0
- Thread 1 accesses address 4 (next float)
- Thread 2 accesses address 8
- All 32 threads in warp access consecutive 128-byte region
- Hardware combines into 1-2 memory transactions

**Bad (Strided):**
```cuda
// Strided access - poor memory performance
__global__ void strided_read(float* data, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Thread 0 reads data[0], thread 1 reads data[stride], etc.
        output[idx] = data[idx * stride];  // Uncoalesced!
    }
}
```

From [Understanding CUDA Memory Usage Blog](https://medium.com/@heyamit10/understanding-cuda-memory-usage-a-practical-guide-6dbb85d4da5a) (accessed 2025-02-02):
- Memory coalescing is key to GPU speed
- Warp (32 threads) memory accesses combined when addresses are consecutive
- Strided or random access patterns cause multiple transactions
- Performance difference can be 5-10x between coalesced and uncoalesced

### Attention-Specific Coalescing

**Matrix multiplication in attention requires careful access patterns:**
```cuda
// Computing Q @ K^T with coalescing
__global__ void attention_qk(
    const float* Q,     // [batch, heads, seqlen_q, headdim]
    const float* K,     // [batch, heads, seqlen_k, headdim]
    float* scores,      // [batch, heads, seqlen_q, seqlen_k]
    int seqlen_q, int seqlen_k, int headdim
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y;

    if (q_idx < seqlen_q) {
        float sum = 0.0f;

        // Coalesced read of Q row
        for (int d = 0; d < headdim; d++) {
            float q_val = Q[q_idx * headdim + d];  // Coalesced across threads
            float k_val = K[k_idx * headdim + d];  // Same K row for all threads
            sum += q_val * k_val;
        }

        scores[q_idx * seqlen_k + k_idx] = sum;
    }
}
```

## Shared Memory Optimization

From [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) (accessed 2025-02-02):

Shared memory is a programmable cache that enables data reuse and cooperation between threads in a block.

### Basic Shared Memory Usage

```cuda
__global__ void matmul_shared(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    const int TILE_SIZE = 16;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory (coalesced)
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // Wait for all threads to load

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // Wait before loading next tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Bank Conflicts

From [CUDA Memory Coalescing Blog](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/) (accessed 2025-02-02):

Shared memory is divided into 32 banks. When multiple threads in a warp access different addresses in the same bank, a bank conflict occurs, serializing the accesses.

**Bank Organization:**
- 32 banks, 4-byte wide (one float per bank)
- Address mapping: `bank_id = (address / 4) % 32`
- Addresses 0,32,64,96... map to bank 0
- Addresses 4,36,68,100... map to bank 1

**Conflict-Free Access:**
```cuda
__shared__ float smem[32][33];  // Note: 33 columns (padding)

// Each thread accesses different bank
int tid = threadIdx.x;
float val = smem[tid][0];  // No conflict - different rows, different banks
```

**Bank Conflict Example:**
```cuda
__shared__ float smem[32][32];  // No padding

// All threads access same column - 32-way bank conflict!
int tid = threadIdx.x;
float val = smem[0][tid];  // Conflict! All threads hit different banks in same row
```

**Padding Solution:**
```cuda
// Add padding to avoid conflicts
__shared__ float smem[32][33];  // Extra column shifts bank mapping

// Now column access is conflict-free
float val = smem[0][tid];  // Each thread hits different bank
```

From [Mastering CUDA Matrix Multiplication Blog](https://medium.com/@dhanushg295/mastering-cuda-matrix-multiplication-an-introduction-to-shared-memory-tile-memory-coalescing-and-d7979499b9c5) (accessed 2025-02-02):
- Tile memory (shared memory) dramatically improves performance
- Memory coalescing ensures efficient global memory access
- Bank conflict avoidance through padding is critical
- Combined optimizations can achieve 10-50x speedup over naive implementation

## Tensor Core Programming (WMMA)

From [CUDA Programming Guide - Tensor Cores](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (accessed 2025-02-02):

Tensor Cores perform matrix multiply-accumulate (MMA) operations at very high throughput. The WMMA (Warp Matrix Multiply-Accumulate) API provides access to Tensor Cores.

### WMMA Basics

**Supported Matrix Sizes:**
- Volta/Turing: 16x16x16 (M x N x K)
- Ampere: 16x16x16, 8x8x4 (MMA instruction)
- Hopper: New WGMMA instructions with larger tiles

**Basic WMMA Code:**
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmma_matmul(
    half* A, half* B, float* C,
    int M, int N, int K
) {
    // Declare fragments (register storage for 16x16 tiles)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);

    // Each warp computes a 16x16 output tile
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) * 16;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) * 16;

    // Tile over K dimension
    for (int k = 0; k < K; k += 16) {
        // Load 16x16 tiles from A and B
        load_matrix_sync(a_frag, A + warp_row * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warp_col, N);

        // Perform 16x16x16 matrix multiply
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, mem_row_major);
}
```

From [Introduction to Tensor Cores Programming](https://0mean1sigma.com/tgemm/) (accessed 2025-02-02):
- Tensor Cores are dedicated GPU units for matrix multiply-accumulate
- Designed for half-precision input (FP16/BF16) and single-precision accumulation (FP32)
- WMMA API provides warp-level primitives for Tensor Core access
- Significantly higher throughput than CUDA cores for matrix operations

### Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate)

From [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):

Hopper GPUs introduce WGMMA instructions with much higher throughput than Ampere's mma.sync:

**Key Features:**
- Operates on warpgroups (4 warps = 128 threads) instead of single warps
- Asynchronous execution - allows overlap with other operations
- Higher peak throughput - 989 TFLOPS FP16, 1978 TFLOPS FP8 on H100
- Requires new programming model with producer/consumer warps

**WGMMA vs MMA Comparison:**
```
Ampere (mma.sync):
- Synchronous execution
- Single warp (32 threads)
- ~312 TFLOPS FP16 on A100
- Can reach ~70% peak

Hopper (WGMMA):
- Asynchronous execution
- Warpgroup (128 threads)
- 989 TFLOPS FP16 on H100
- FlashAttention-3 reaches 75% peak
```

## FlashAttention CUDA Implementation

From [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) and [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):

### Algorithm Overview

FlashAttention reorders attention computation using tiling and online softmax to minimize HBM access:

**Standard Attention (naive):**
```
1. Compute S = Q @ K^T          [seqlen_q, seqlen_k] - write to HBM
2. Compute P = softmax(S)       [seqlen_q, seqlen_k] - write to HBM
3. Compute O = P @ V            [seqlen_q, headdim]   - write to HBM

Memory: O(seqlen_q * seqlen_k) for intermediate matrices
Bandwidth: 3 full matrix writes to HBM
```

**FlashAttention (tiled):**
```
For each block of Q (size Br):
    For each block of K, V (size Bc):
        1. Load Q block, K block to SRAM
        2. Compute S_block = Q_block @ K_block^T (in SRAM)
        3. Update running max and sum for softmax (in SRAM)
        4. Load V block to SRAM
        5. Update output accumulator O (in SRAM)

    Write final O block to HBM

Memory: O(seqlen_q * headdim) for output only
Bandwidth: Only final output written to HBM
```

### Online Softmax Algorithm

The key innovation is computing softmax without materializing the full attention matrix:

**Running Statistics:**
```cuda
// For each query position, maintain:
float m_i = -INFINITY;  // Running maximum
float l_i = 0.0f;       // Running sum of exponentials
float acc_i[d];         // Accumulated weighted values

// For each key-value block:
float m_i_new = max(m_i, max(S_ij));
float l_i_new = exp(m_i - m_i_new) * l_i + sum(exp(S_ij - m_i_new));

// Rescale accumulator
for (int d = 0; d < headdim; d++) {
    acc_i[d] = acc_i[d] * exp(m_i - m_i_new);
}

// Add new contribution
for (int d = 0; d < headdim; d++) {
    acc_i[d] += sum_j(exp(S_ij - m_i_new) * V_j[d]);
}

// Update statistics
m_i = m_i_new;
l_i = l_i_new;

// Final output (after all blocks)
O_i[d] = acc_i[d] / l_i;
```

### FlashAttention-3 Optimizations

From [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):

**1. Asynchrony - Overlapping GEMM and Softmax:**

Non-matmul operations (like exponential for softmax) are much slower than matmul on modern accelerators:
- H100 GPU: 989 TFLOPS FP16 matmul, but only 3.9 TFLOPS for special functions (256x less!)
- For head dimension 128: 512x more matmul FLOPs than exponential
- Yet exponential can take 50% of time due to low throughput

**Pingpong Scheduling:**
```
Warpgroup 1: GEMM1 → Softmax1 → GEMM2 → Softmax2
Warpgroup 2:         GEMM1' → Softmax1' → GEMM2' → Softmax2'

Timeline:
|--GEMM1--|--Softmax1--|--GEMM2--|--Softmax2--|  WG1
          |--GEMM1'---|--Softmax1'--|--GEMM2'---|--Softmax2'--|  WG2

Overlapping: While WG1 does softmax, WG2 does GEMM
```

This scheduling improves FP16 forward pass from ~570 TFLOPS to ~620 TFLOPS.

**2. Intra-Warpgroup Pipelining:**

Even within one warpgroup, overlap softmax with GEMM:
```
Single warpgroup timeline:
|--GEMM0--|
          |--GEMM1--|--Softmax0--|
                    |--GEMM2--|--Softmax1--|
```

This further improves to 640-660 TFLOPS, at cost of higher register pressure.

**3. Incoherent Processing for FP8:**

From [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):

LLM activations have outliers with large magnitude, causing quantization errors in FP8. Solution: multiply Q and K with random orthogonal matrix (Hadamard transform) to spread outliers.

**Hadamard Transform:**
- Multiply each head's Q and K by random sign-flipped Hadamard matrix
- O(d log d) time instead of O(d²) for general orthogonal matrix
- Fused with rotary embedding (both memory-bound, so "free")
- Reduces quantization error by 2.6x

**Performance Results:**
- FP16: 1.5-2.0x faster than FlashAttention-2, up to 740 TFLOPS (75% of H100 peak)
- FP8: Close to 1.2 PFLOPS with 2.6x lower error than baseline FP8

## PagedAttention CUDA Implementation

From [vLLM PagedAttention Documentation](https://docs.vllm.ai/en/latest/design/paged_attention.html) and [vLLM GitHub](https://github.com/vllm-project/vllm) (accessed 2025-02-02):

### PagedAttention Concept

PagedAttention applies virtual memory paging concepts to KV cache management:

**Problem:**
- KV cache for each sequence grows dynamically during generation
- Continuous memory allocation is wasteful due to unknown final length
- Internal fragmentation (reserved but unused space) is significant

**Solution:**
- Divide KV cache into fixed-size blocks (pages)
- Allocate pages on-demand as sequence grows
- Use block table to map logical positions to physical blocks
- Share blocks between sequences for prefix sharing

**Memory Layout:**
```
Standard (continuous):
Seq 1: [K0 K1 K2 __ __ __]  - 3 tokens, reserved 6
Seq 2: [K0 K1 K2 K3 K4 __]  - 5 tokens, reserved 6
Wasted: 4 slots

PagedAttention (paged):
Block pool: [B0][B1][B2][B3][B4]...
Seq 1: blocks [0, 1]      - B0: K0,K1  B1: K2
Seq 2: blocks [0, 3, 4]   - B0: K0,K1  B3: K2,K3  B4: K4
Wasted: 1 slot (in B1) + 1 slot (in B4) = much less!
```

### CUDA Kernel Structure

**Key Challenge:** Non-contiguous memory access through block table requires special kernel implementation.

```cuda
__global__ void paged_attention_kernel(
    float* Q,                    // Query: [batch, heads, headdim]
    float* K_cache,              // KV cache: [num_blocks, block_size, heads, headdim]
    float* V_cache,
    int* block_tables,           // Block table: [batch, max_blocks]
    int* context_lens,           // Actual sequence lengths: [batch]
    float* output,               // Output: [batch, heads, headdim]
    int num_heads,
    int headdim,
    int block_size,
    int max_blocks
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    // Get actual context length for this sequence
    int context_len = context_lens[batch_idx];
    int num_blocks = (context_len + block_size - 1) / block_size;

    // Pointers to this sequence's block table
    int* block_table = block_tables + batch_idx * max_blocks;

    // Load query
    float q[headdim];
    for (int d = threadIdx.x; d < headdim; d += blockDim.x) {
        q[d] = Q[batch_idx * num_heads * headdim + head_idx * headdim + d];
    }
    __syncthreads();

    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    float acc[headdim] = {0.0f};

    // Iterate over blocks (pages)
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block = block_table[block_idx];

        // Iterate over tokens in this block
        int tokens_in_block = min(block_size, context_len - block_idx * block_size);

        for (int token_idx = 0; token_idx < tokens_in_block; token_idx++) {
            // Compute attention score Q @ K^T
            float score = 0.0f;
            for (int d = 0; d < headdim; d++) {
                int k_offset = physical_block * block_size * num_heads * headdim
                             + token_idx * num_heads * headdim
                             + head_idx * headdim + d;
                score += q[d] * K_cache[k_offset];
            }
            score /= sqrt((float)headdim);

            // Update online softmax statistics
            float old_max = max_logit;
            max_logit = max(max_logit, score);
            float exp_score = exp(score - max_logit);
            float rescale = exp(old_max - max_logit);
            sum_exp = sum_exp * rescale + exp_score;

            // Rescale and accumulate: acc = acc * rescale + exp_score * V
            for (int d = 0; d < headdim; d++) {
                acc[d] = acc[d] * rescale;
                int v_offset = physical_block * block_size * num_heads * headdim
                             + token_idx * num_heads * headdim
                             + head_idx * headdim + d;
                acc[d] += exp_score * V_cache[v_offset];
            }
        }
    }

    // Final softmax normalization and output
    for (int d = threadIdx.x; d < headdim; d += blockDim.x) {
        output[batch_idx * num_heads * headdim + head_idx * headdim + d] = acc[d] / sum_exp;
    }
}
```

From [vLLM Forums Discussion](https://discuss.vllm.ai/t/promblem-about-the-pagedattention-split-kv-cache-implimentation/509) (accessed 2025-02-02):

The implementation handles split KV cache efficiently by:
- Using block tables to indirectly access physical blocks
- Avoiding memory copies through in-place updates
- Supporting dynamic allocation as sequences grow

### Optimization Considerations

**Memory Access Patterns:**
- Block table lookups add indirection overhead
- Coalescing is harder with paged access
- Prefetching blocks can hide latency

**Performance vs Standard Attention:**
- Small overhead from indirection (~5-10%)
- Massive memory savings (up to 4x less waste)
- Enables much higher batch sizes and throughput

## Performance Optimization Techniques

### 1. Occupancy Optimization

From [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) (accessed 2025-02-02):

**Occupancy** = (Active warps per SM) / (Maximum warps per SM)

Higher occupancy helps hide memory latency:

```cuda
// Check occupancy
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// Occupancy depends on:
// - Registers per thread
// - Shared memory per block
// - Block size (threads per block)
```

**Occupancy vs Performance:**
- Higher occupancy ≠ always better performance
- Balance between hiding latency and having enough resources per thread
- Sweet spot often 50-75% occupancy for compute-intensive kernels

### 2. Warp-Level Primitives

**Warp Shuffle:**
```cuda
// Exchange data between threads in warp without shared memory
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Use Cases:**
- Reducing sums across warps
- Broadcasting values within warp
- Faster than shared memory for small data

### 3. Loop Unrolling

```cuda
// Manual unrolling for known sizes
#pragma unroll
for (int d = 0; d < 128; d++) {  // 128 known at compile time
    sum += q[d] * k[d];
}

// Compiler generates 128 multiply-add instructions
// Reduces loop overhead and enables instruction-level parallelism
```

### 4. Kernel Fusion

Combine multiple operations into single kernel to reduce memory traffic:

```cuda
// Fused kernel: softmax + scale + output
__global__ void fused_attention_output(
    float* scores,     // [seqlen_q, seqlen_k]
    float* V,          // [seqlen_k, headdim]
    float* output,     // [seqlen_q, headdim]
    float scale,
    int seqlen_q, int seqlen_k, int headdim
) {
    int q_idx = blockIdx.x;

    // Load row of scores
    float row_scores[MAX_SEQLEN];
    float max_val = -INFINITY;

    // Find max
    for (int k = 0; k < seqlen_k; k++) {
        row_scores[k] = scores[q_idx * seqlen_k + k] * scale;
        max_val = max(max_val, row_scores[k]);
    }

    // Compute softmax
    float sum_exp = 0.0f;
    for (int k = 0; k < seqlen_k; k++) {
        row_scores[k] = exp(row_scores[k] - max_val);
        sum_exp += row_scores[k];
    }

    // Normalize and compute output in same kernel
    for (int d = threadIdx.x; d < headdim; d += blockDim.x) {
        float val = 0.0f;
        for (int k = 0; k < seqlen_k; k++) {
            val += (row_scores[k] / sum_exp) * V[k * headdim + d];
        }
        output[q_idx * headdim + d] = val;
    }
}
```

## Advanced Topics

### Tensor Memory Accelerator (TMA) - Hopper

From [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):

TMA is special hardware unit on Hopper GPUs for data transfer:

**Features:**
- Accelerates global ↔ shared memory transfers
- Handles index calculation and bounds checking
- Frees up registers for other work
- Enables async data movement

**Programming Model:**
```cuda
// Traditional (manual):
__shared__ float smem[TILE_SIZE][TILE_SIZE];
smem[ty][tx] = global_mem[...complex indexing...];

// TMA (hardware-accelerated):
tma_load_2d(smem, global_mem, tile_coords);
// Hardware handles all indexing automatically
```

### Multi-GPU Attention

**Tensor Parallelism:**
```
Partition attention heads across GPUs:
GPU 0: heads 0-7
GPU 1: heads 8-15
GPU 2: heads 16-23
GPU 3: heads 24-31

Each GPU computes full attention for its heads independently
No communication during attention computation
All-reduce output at end (if needed)
```

**Sequence Parallelism:**
```
Partition sequence across GPUs:
GPU 0: tokens 0-1023
GPU 1: tokens 1024-2047
GPU 2: tokens 2048-3071

Requires communication for causal masking
More complex but enables longer sequences
```

### Debugging CUDA Kernels

**Common Issues:**

1. **Race Conditions:**
```cuda
// Missing synchronization
__shared__ float smem[256];
smem[tid] = data[tid];
// BUG: Need __syncthreads() here!
float val = smem[tid + 1];  // May read old value
```

2. **Out-of-Bounds Access:**
```cuda
// Always check bounds
if (idx < n) {
    output[idx] = input[idx];
}
```

3. **Bank Conflicts:**
```cuda
// Use padding to avoid
__shared__ float smem[32][33];  // Not [32][32]
```

**Debugging Tools:**
- `cuda-memcheck`: Detect memory errors
- `cuda-gdb`: Step through kernel code
- `nsight compute`: Profile kernel performance
- `printf` in kernel (with synchronization!)

## Code Examples Repository

Reference implementations demonstrating these concepts:

From [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) (accessed 2025-02-02):
- `csrc/flash_attn/src/flash.h`: Main attention interface
- `csrc/flash_attn/src/flash_fwd_kernel.h`: Forward pass kernels
- `csrc/flash_attn/src/flash_bwd_kernel.h`: Backward pass kernels

From [vLLM GitHub](https://github.com/vllm-project/vllm) (accessed 2025-02-02):
- `vllm/attention/backends/`: Attention backend implementations
- `csrc/`: CUDA kernel sources for PagedAttention

From [CUDA Matrix Multiplication Optimization Blog](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) (accessed 2025-02-02):
- Iterative optimization from naive to highly optimized GEMM
- Shows progression: naive → tiled → shared memory → reduced bank conflicts → WMMA
- Achieves near-cuBLAS performance with proper optimizations

## Performance Benchmarks

From [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) (accessed 2025-02-02):

**H100 GPU Performance (FP16):**
```
Sequence Length | FlashAttention-2 | FlashAttention-3 | Speedup
512             | 420 TFLOPS       | 650 TFLOPS       | 1.55x
1024            | 450 TFLOPS       | 690 TFLOPS       | 1.53x
2048            | 480 TFLOPS       | 720 TFLOPS       | 1.50x
4096            | 500 TFLOPS       | 735 TFLOPS       | 1.47x
8192            | 510 TFLOPS       | 740 TFLOPS       | 1.45x
```

**FP8 Performance:**
- FlashAttention-3 with FP8: ~1150 TFLOPS
- Near 1.2 PFLOPS peak
- 2.6x lower error than baseline FP8 (with incoherent processing)

**Memory Efficiency:**
```
Standard Attention: O(seqlen² * headdim) bytes
FlashAttention: O(seqlen * headdim) bytes

For seqlen=4096, headdim=128:
Standard: 2 GB per batch
FlashAttention: 0.5 MB per batch
Savings: 4000x less memory!
```

## Best Practices Summary

1. **Memory Access:**
   - Always aim for coalesced global memory access
   - Use shared memory for data reuse
   - Pad shared memory to avoid bank conflicts
   - Prefetch data to hide latency

2. **Computation:**
   - Use Tensor Cores (WMMA/WGMMA) for matrix ops
   - Fuse operations to reduce memory traffic
   - Unroll loops for known sizes
   - Overlap computation and memory access

3. **Attention-Specific:**
   - Use tiling to fit in SRAM
   - Implement online softmax to avoid materialization
   - Consider PagedAttention for KV cache management
   - Optimize for your target GPU architecture (Ampere vs Hopper)

4. **Debugging:**
   - Start with simple, correct implementation
   - Validate against reference (PyTorch)
   - Profile before optimizing
   - Optimize bottlenecks systematically

## Sources

**Official Documentation:**
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - NVIDIA official CUDA documentation
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - NVIDIA optimization guidelines

**FlashAttention Resources:**
- [FlashAttention GitHub Repository](https://github.com/Dao-AILab/flash-attention) - Official implementation (accessed 2025-02-02)
- [FlashAttention-3 Blog Post](https://tridao.me/blog/2024/flash3/) by Tri Dao (accessed 2025-02-02)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness - Dao et al., NeurIPS 2022
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning - Dao, ICLR 2024

**vLLM and PagedAttention:**
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) - Official vLLM implementation (accessed 2025-02-02)
- [vLLM PagedAttention Documentation](https://docs.vllm.ai/en/latest/design/paged_attention.html) (accessed 2025-02-02)
- [vLLM Forums Discussion](https://discuss.vllm.ai/t/promblem-about-the-pagedattention-split-kv-cache-implimentation/509) (accessed 2025-02-02)

**Technical Blogs and Tutorials:**
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) - NVIDIA Developer Blog (accessed 2025-02-02)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) by Lei Mao (accessed 2025-02-02)
- [CUDA Coalesced Memory Access](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/) by Lei Mao (accessed 2025-02-02)
- [Mastering CUDA Matrix Multiplication](https://medium.com/@dhanushg295/mastering-cuda-matrix-multiplication-an-introduction-to-shared-memory-tile-memory-coalescing-and-d7979499b9c5) by Dhanush (accessed 2025-02-02)
- [Understanding CUDA Memory Usage](https://medium.com/@heyamit10/understanding-cuda-memory-usage-a-practical-guide-6dbb85d4da5a) by Hey Amit (accessed 2025-02-02)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm (accessed 2025-02-02)
- [Introduction to Tensor Cores Programming](https://0mean1sigma.com/tgemm/) by 0Mean1Sigma (accessed 2025-02-02)

**Additional Web Research:**
- Stack Overflow discussions on CUDA shared memory and bank conflicts (accessed 2025-02-02)
- Reddit r/CUDA community discussions on optimization techniques (accessed 2025-02-02)
- NVIDIA Developer Forums on Tensor Core programming (accessed 2025-02-02)
