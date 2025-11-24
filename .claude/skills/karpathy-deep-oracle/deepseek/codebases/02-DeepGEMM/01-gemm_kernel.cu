/*
DeepGEMM CUDA Kernel - Optimized matrix multiplication

<karpathys_code_comments>
** This File's Role **
Hand-optimized CUDA kernel for GEMM (General Matrix Multiply). This is what actually runs on the GPU
hardware to multiply massive matrices fast.

** Function List **
gemm_kernel<<<grid, block>>>(A, B, C, M, N, K) - Main GEMM kernel
load_tile_shared(A, tile_id) - Load matrix tile into shared memory
compute_tile_product(A_shared, B_shared) - Compute partial matrix product
write_result(C, partial_sum) - Write computed values to global memory

** Technical Deep Dive **
Matrix multiplication is the heart of deep learning. C = A @ B where A is [M, K] and B is [K, N].

Standard approach: O(M*N*K) operations. The challenge: memory bandwidth, not compute, is the bottleneck.

DeepGEMM's optimizations:
1. Tiling - Load tiles into fast shared memory (100x faster than global memory)
2. Thread cooperation - Multiple threads compute one output tile together
3. Memory coalescing - Access memory in patterns that match GPU hardware

Karpathy: Writing CUDA kernels is hard. You're programming at the level of GPU warp schedulers,
cache hierarchies, and memory banks. But when you need maximum performance (like DeepSeek training
V3), hand-optimized kernels can be 2-5x faster than PyTorch's default GEMM. That's the difference
between $5M and $15M training costs.
</karpathys_code_comments>
*/

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Karpathy: Each thread block computes one tile of output matrix C.
    // Block dimensions: (TILE_SIZE, TILE_SIZE), where TILE_SIZE = 16 or 32.

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Karpathy: Accumulator for this thread's output element. Use register (fastest memory).
    float sum = 0.0f;

    // Karpathy: Loop over K dimension in tiles. Load tiles to shared memory for reuse.
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        // Load tiles from global memory to shared memory
        // Karpathy: This is the key optimization - amortize global memory latency across thread block.
        __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
        __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

        // Karpathy: Coalesced memory access - adjacent threads load adjacent addresses.
        A_tile[threadIdx.y][threadIdx.x] = A[row * K + tile + threadIdx.x];
        B_tile[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * N + col];

        __syncthreads();  // Karpathy: Wait for all threads to finish loading

        // Karpathy: Compute partial dot product using shared memory (fast!)
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();  // Karpathy: Wait before loading next tile
    }

    // Karpathy: Write final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Karpathy: That's the basic GEMM kernel. Real DeepGEMM has more optimizations:
// - Vectorized loads (load float4 instead of float)
// - Warp-level primitives (use tensor cores on modern GPUs)
// - Bank conflict avoidance in shared memory
// But the core idea is here: tile, shared memory, reuse.
