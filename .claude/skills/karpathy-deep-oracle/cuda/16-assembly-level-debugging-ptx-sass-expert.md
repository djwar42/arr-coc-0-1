# PTX/SASS Assembly-Level Debugging & Cubin Analysis Expert

## Overview

CUDA GPU code execution involves multiple compilation layers, from high-level CUDA C++ to PTX (Parallel Thread Execution) intermediate representation to SASS (Shader Assembly) machine code. Understanding assembly-level debugging techniques enables diagnosis of the hardest performance issues and deepest kernel bugs that aren't visible at higher abstraction levels.

**Key capabilities at assembly level:**
- PTX/SASS instruction-level analysis and correlation with source code
- Register pressure diagnosis and spill detection
- Instruction throughput analysis and warp scheduler behavior
- Memory access pattern verification at hardware level
- Inline assembly debugging for hand-optimized kernels
- Cubin inspection and binary analysis

This document provides **absolute deepest** assembly-level debugging expertise for production VLM training and inference workloads.

---

## Section 1: PTX/SASS Fundamentals & Tools (~150 lines)

### Understanding the Compilation Pipeline

CUDA code compilation flows through distinct stages:

```
CUDA C++ (.cu)
    ↓ [nvcc frontend]
PTX Assembly (.ptx)
    ↓ [ptxas assembler]
SASS Machine Code (.cubin)
    ↓ [driver at runtime]
GPU Execution
```

From [eunomia-bpf CUDA Tutorial: Understanding GPU Assembly with PTX](https://eunomia.dev/others/cuda-tutorial/02-ptx-assembly/) (accessed 2025-11-13):
- **PTX**: Virtual assembly language, architecture-independent, similar to LLVM IR
- **SASS**: Actual GPU machine code, architecture-specific (sm_70 to sm_121)
- **Forward compatibility**: PTX compiled to SASS at runtime enables running old code on new GPUs
- **cubin**: ELF container (EM_CUDA machine type 190) embedding compiled GPU code

### PTX vs SASS: When to Use Each

**PTX (Parallel Thread Execution):**
- Virtual ISA, machine-independent
- Human-readable intermediate representation
- Access via `nvcc -ptx` or `cuobjdump --dump-ptx`
- Use for: understanding compiler output, writing portable inline assembly

**SASS (Streaming Assembly):**
- Real hardware instructions (RISC/NISC/VLIW-like hybrid)
- Architecture-specific (Volta sm_70 to Blackwell sm_121)
- Access via `cuobjdump --dump-sass` or `nvdisasm`
- Use for: final performance analysis, hardware-level debugging

From [Reversing Nvidia GPU's SASS code - JEB Decompiler](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/) (accessed 2025-11-13):
> "SASS is the low-level, semi-documented machine code generated when compiling high-level CUDA source code with nvcc or when translating PTX intermediate code with ptxas. All Volta+ instructions are fixed size, 16-byte long."

### Core cuobjdump Commands

**Extract PTX from compiled binary:**
```bash
cuobjdump --dump-ptx mykernel.cubin > kernel.ptx
```

**Extract SASS disassembly:**
```bash
cuobjdump --dump-sass mykernel.cubin > kernel.sass
```

**List ELF sections and metadata:**
```bash
cuobjdump --list-elf mykernel.cubin
```

**Dump constant memory segments:**
```bash
cuobjdump --dump-resource-usage mykernel.cubin
```

**Extract specific function:**
```bash
cuobjdump --dump-sass --function-name="myKernel" mykernel.cubin
```

From [Tutorial: Understanding GPU Assembly with PTX](https://eunomia.dev/others/cuda-tutorial/02-ptx-assembly/):
> "cuobjdump extracts PTX from a compiled executable, useful for seeing what actually got compiled. The --dump-sass option reveals the actual machine code generated from your PTX."

### PTX ISA Fundamentals

**Virtual Registers (unlimited):**
```ptx
.reg .pred  %p<8>;      // Predicate registers (boolean flags)
.reg .b32   %r<256>;    // 32-bit integer registers
.reg .b64   %rd<64>;    // 64-bit integer registers
.reg .f32   %f<32>;     // 32-bit floating-point registers
.reg .f64   %d<16>;     // 64-bit floating-point registers
```

**Special registers (read-only built-ins):**
```ptx
%tid.x, %tid.y, %tid.z          // Thread index within block
%ntid.x, %ntid.y, %ntid.z       // Block dimensions
%ctaid.x, %ctaid.y, %ctaid.z    // Block index within grid
%nctaid.x, %nctaid.y, %nctaid.z // Grid dimensions
%clock, %clock64                 // Cycle counter
%laneid                          // Lane ID within warp (0-31)
```

From [Reversing Nvidia GPU's SASS code](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/):
> "General Registers (Rx): up to 256 32-bit registers; R255 is a zero-register (aliased RZ). Predicate Registers (Px): 8 boolean flags per thread; P7 is always true (aliased PT). Special Registers (SRx): 256 read-only registers, containing thread/block IDs, lane ID, clock values, performance counters."

**Common PTX instruction patterns:**
```ptx
// Thread index calculation
mov.u32  %r2, %ctaid.x;         // blockIdx.x → r2
mov.u32  %r3, %ntid.x;          // blockDim.x → r3
mov.u32  %r4, %tid.x;           // threadIdx.x → r4
mad.lo.s32  %r1, %r3, %r2, %r4; // r1 = r3 * r2 + r4

// Memory operations
ld.global.f32  %f1, [%rd1];     // Load from global memory
st.global.f32  [%rd2], %f1;     // Store to global memory
ld.shared.f32  %f2, [%rd3];     // Load from shared memory

// Control flow (predicated)
setp.ge.s32  %p1, %r1, %r2;     // p1 = (r1 >= r2)
@%p1 bra  BB0_2;                // Branch if p1 is true
```

### SASS Architecture-Specific Instructions

**Volta (sm_70) to Blackwell (sm_121) instruction formats:**

All SASS instructions are **16 bytes (128 bits)** with fixed encoding.

From [Reversing Nvidia GPU's SASS code](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/):
> "Classes of Instructions: Integer (IMAD, IADD3, SHF, LOP3), Floating-point (FADD, FFMA, FSET, F2F/F2I/I2F, MUFU), Load/Store (LDx/STx for each memory space), Control flow (BRA, BRX, CALL, RET, SSY, BSYNC, EXIT), Uniform ops (UIADD3, UIMAD, ULEA for uniform registers)."

**Example SASS disassembly:**
```sass
IMAD R5, R5, c[0x0][0x4], R2        // Integer multiply-add
FFMA R1, R2, R3, R4                 // Fused multiply-add
LDG.E.U8 R6, desc[UR4][R2.64]       // Global load with descriptor
STG.E [R10], R1                     // Global store
@!P0 BRA 0x240                      // Conditional branch
```

### ISA Version Differences by Architecture

**Volta (sm_70, sm_72):**
- Tensor Core generation 1 (WMMA API)
- Independent thread scheduling
- Unified memory architecture improvements

**Turing (sm_75):**
- Tensor Core generation 2
- Uniform registers (URx) introduced
- Integer and FP16 Tensor Core support

**Ampere (sm_80, sm_86):**
- Tensor Core generation 3
- TF32 and BF16 native support
- Asynchronous copy (cp.async)

**Ada (sm_89):**
- Tensor Core generation 4
- FP8 support (E4M3, E5M2)
- Improved L2 cache

**Hopper (sm_90):**
- WGMMA (warpgroup matrix multiply)
- Thread Block Clusters
- TMA (Tensor Memory Accelerator)
- Distributed shared memory

**Blackwell (sm_100-121):**
- 256 uniform registers (vs 64 in sm_75-90)
- Enhanced FP8 formats (MXFP8)
- Improved instruction scheduling

---

## Section 2: Register & Occupancy Analysis (~150 lines)

### Understanding ptxas Verbose Output

**Enable verbose compilation:**
```bash
nvcc -Xptxas -v kernel.cu
```

From [How to Improve CUDA Kernel Performance](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/) (accessed 2025-11-13):
> "The ptxas compiler provides verbose output showing register usage, shared memory usage, and spilling information - critical for occupancy analysis."

**Example ptxas output:**
```
ptxas info    : Compiling entry function '_Z6kernelPfS_i' for 'sm_80'
ptxas info    : Function properties for _Z6kernelPfS_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 46 registers, 8192 bytes smem, 360 bytes cmem[0]
```

**Interpreting register usage:**
- **Used 46 registers**: Each thread uses 46 32-bit registers
- **0 bytes spill stores/loads**: No register spilling (good!)
- **8192 bytes smem**: Shared memory usage per block
- **360 bytes cmem[0]**: Constant memory bank 0 usage

From [Stack Overflow: Interpreting ptxas verbose output](https://stackoverflow.com/questions/12388207/interpreting-the-verbose-output-of-ptxas-part-i) (accessed 2025-11-13):
> "'Used 46 registers' indicates the compiler has reserved 46 registers per thread for the compiled kernel. There is no register spilling to local memory when spill stores/loads are 0 bytes."

### Register Spilling Detection

**Register spilling** occurs when a kernel uses more registers than available per thread, forcing data to slower local memory (actually in global DRAM).

**Detect spilling in ptxas output:**
```
ptxas info    : Used 64 registers, 0+16 bytes lmem
ptxas warning : Stack size for entry function '_Z6kernelPfS_i' cannot be
                statically determined
```

The `+16 bytes lmem` indicates **local memory usage** (register spills).

**Measure spilling impact in NSight Compute:**
```bash
ncu --set full --section MemoryWorkloadAnalysis ./myapp

# Look for these metrics:
# - l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum  (local loads)
# - l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum  (local stores)
```

From [NVIDIA Developer Forums: Understanding PTXAS output](https://forums.developer.nvidia.com/t/understanding-ptxas-output/332181) (accessed 2025-11-13):
> "Scalar registers and vector registers (PTX name for types like int4 or short2) are stored separately. Register spilling detection requires examining both ptxas output and runtime profiling with NSight Compute."

**Spilling mitigation strategies:**

1. **Reduce register pressure:**
```cpp
// Add compiler hint
__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
__global__ void kernel() { ... }
```

2. **Limit max registers per thread:**
```bash
nvcc -maxrregcount=48 kernel.cu
# Or per-kernel:
__global__ void __launch_bounds__(256, 4) kernel() { ... }
```

3. **Use shared memory instead:**
```cpp
// Instead of many local arrays
extern __shared__ float smem[];  // Shared across block
```

### Occupancy Calculation from Register Usage

**Theoretical occupancy formula:**
```
Occupancy = Active Warps / Maximum Warps per SM
```

**Register-limited occupancy (Ampere A100 example):**
- 65,536 registers per SM
- Maximum 2048 threads per SM (64 warps)
- For kernel using **R** registers per thread:
  ```
  Max threads = min(2048, 65536 / R)
  Max warps = Max threads / 32
  ```

**Examples for A100 (sm_80):**
| Registers/Thread | Max Threads/SM | Max Warps/SM | Occupancy |
|------------------|----------------|--------------|-----------|
| 32               | 2048           | 64           | 100%      |
| 64               | 1024           | 32           | 50%       |
| 96               | 682 (21 warps) | 21           | 32.8%     |
| 128              | 512            | 16           | 25%       |
| 255              | 256            | 8            | 12.5%     |

From [cuda/12-kernel-debugging-internals-expert.md](../cuda/12-kernel-debugging-internals-expert.md) (existing knowledge):
> "Register pressure analysis: ptxas -v output interpretation shows registers used. Occupancy calculator integration reveals registers vs shared memory trade-offs."

### Analyzing Register Usage Per Instruction

**Use NSight Compute Source View to see per-instruction register allocation:**

```bash
ncu --set full --section SourceCounters -o profile ./myapp
# Open profile.ncu-rep in NSight Compute UI
# Navigate to Source View → see register usage per source line
```

**PTX register allocation example:**
```ptx
// PTX shows virtual registers (unlimited)
.reg .f32 %f<100>;  // Declares 100 float registers

// ptxas maps to physical registers
// If >255 physical regs needed → spilling
```

### Occupancy vs Performance Trade-offs

**Higher occupancy ≠ always better performance**

From [cuda/10-performance-debugging-profiling-expert.md](../cuda/10-performance-debugging-profiling-expert.md) (existing knowledge):
> "Expert optimization: block size tuning 128-256, warp efficiency, async 10-20%. GPU utilization debugging shows Meta case study: 9.1%→74% SM utilization."

**Decision matrix:**

| Scenario | Optimize For | Strategy |
|----------|--------------|----------|
| Memory-bound kernel | Latency hiding | Maximize occupancy |
| Compute-bound kernel | ILP (instruction-level parallelism) | Accept lower occupancy for more registers |
| Latency-sensitive | Low register pressure | Use shared memory, limit registers |
| Throughput-focused | Maximum ALU utilization | Balance ILP vs TLP (thread-level parallelism) |

---

## Section 3: Instruction-Level Optimization (~150 lines)

### Instruction Throughput Analysis with NSight Compute

**NSight Compute provides detailed SASS instruction metrics:**

```bash
ncu --metrics smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum \
./myapp
```

From [NSight Compute SASS analysis instruction throughput](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (accessed 2025-11-13):
> "Using throughput metrics ensures meaningful and actionable analysis. An assembly (SASS) instruction executed may generate multiple micro-operations depending on the data types and execution path."

**Key throughput metrics:**

| Metric | Meaning | Ideal Value |
|--------|---------|-------------|
| `inst_executed` | Total instructions executed | Minimize |
| `smsp__inst_executed_pipe_tensor` | Tensor Core instructions | Maximize for matmul |
| `smsp__pipe_fma_cycles_active.avg` | FMA pipeline utilization | Close to 100% |
| `smsp__warp_issue_stalled_*` | Stall reasons | Identify bottleneck |

### Dual-Issue and Warp Scheduler Behavior

**Modern GPUs have 4 warp schedulers per SM** (Ampere/Hopper):
- Each scheduler issues 1 instruction per warp per cycle
- **Dual-issue**: Some instruction pairs execute simultaneously

**Dual-issue eligible pairs:**
```sass
IADD3 R0, R1, R2, RZ    // Integer add
FFMA R4, R5, R6, R7     // FP multiply-add (independent)
```

If R0-R3 and R4-R7 are independent, both instructions issue in same cycle.

**Check dual-issue in NSight Compute:**
```bash
ncu --metrics smsp__inst_executed,smsp__inst_issued ./myapp

# Dual-issue ratio = inst_executed / inst_issued
# Ideal: close to 2.0 (both pipelines utilized)
```

### Latency Hiding Techniques

**Arithmetic instruction latencies (approximate, architecture-dependent):**
| Instruction | Latency (cycles) | Throughput |
|-------------|------------------|------------|
| IADD, IMAD  | 4-6              | 1/cycle    |
| FADD        | 4-6              | 1/cycle    |
| FFMA        | 4-6              | 1/cycle    |
| FMUL        | 4-6              | 1/cycle    |
| MUFU (sin)  | ~20              | 1/4 cycles |
| Global load | 200-400          | Varies     |

**Hide latency with:**

1. **Instruction-Level Parallelism (ILP):**
```cpp
// Good: independent operations
float r1 = a[i] * b[i] + c[i];
float r2 = d[i] * e[i] + f[i];  // Can execute in parallel
float r3 = g[i] * h[i] + j[i];
```

2. **Thread-Level Parallelism (TLP):**
```cpp
// More warps = more latency hiding
__global__ void __launch_bounds__(256) kernel() {
    // 256 threads/block × many blocks = high TLP
}
```

From [cuda/10-performance-debugging-profiling-expert.md](../cuda/10-performance-debugging-profiling-expert.md):
> "Latency hiding techniques: ILP vs TLP, occupancy vs resource usage. Warp efficiency and async operations provide 10-20% speedup."

### SASS Optimization Patterns

**Memory access alignment (detected in SASS):**

```sass
# Good: Aligned 128-byte coalesced access
LDG.E.128 R0, [R2]      // Load 128 bytes aligned

# Bad: Unaligned access
LDG.E.32 R0, [R2+0x1]   // Misaligned, sector waste
```

**Check coalescing in NSight Compute:**
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./myapp

# Coalescing ratio = sectors / (requests * 4)
# Ideal: 1.0 (perfect coalescing for 128-byte)
```

**Predication to avoid divergence:**

```sass
# Instead of branch divergence:
@P0 BRA SKIP
    FFMA R1, R2, R3, R4
SKIP:

# Use predication:
@P0 FFMA R1, R2, R3, R4  // Only executes if P0 true
```

**Reuse flags for register efficiency:**

```sass
# SASS may reuse register values marked with .reuse
LDG.E.STRONG.GPU R0, [R2]     // Mark for reuse
FFMA R1, R0.reuse, R3, R4     // Reuse R0 value
FADD R5, R0.reuse, R6         // Reuse again
```

From [Reversing Nvidia GPU's SASS code](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/):
> "SASS optimization patterns include load/store alignment, predication to reduce divergence, and reuse flags for register value caching."

### Analyzing Warp Divergence at SASS Level

**Detect divergence in NSight Compute:**
```bash
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./myapp

# Warp execution efficiency
# Ideal: 1.0 (all 32 threads execute same instruction)
# <1.0: divergence (some threads masked)
```

**Divergence visible in SASS:**
```sass
ISETP.GE.AND P0, PT, R1, c[0x0][0x160], PT  // Set predicate
@!P0 BRA BB0_2                               // Divergent branch
    # Threads with P0=false take branch
    # Threads with P0=true fall through
    # Both paths must execute serially (divergence)
BB0_2:
```

---

## Section 4: Advanced Assembly Debugging (~150 lines)

### Inline PTX Assembly Syntax and Constraints

**Basic inline PTX syntax:**
```cpp
asm("instruction" : output_operands : input_operands : clobbered_registers);
```

From [NVIDIA Docs: Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) (accessed 2025-11-13):
> "The reference guide for inlining PTX (parallel thread execution) assembly statements into CUDA provides detailed constraint specifications for operand mapping."

**Constraint letters for operands:**

| Constraint | Meaning | Example |
|------------|---------|---------|
| `"r"` | 32-bit register | `int` |
| `"l"` | 64-bit register | `long long`, pointers |
| `"f"` | 32-bit float register | `float` |
| `"d"` | 64-bit double register | `double` |
| `"h"` | 16-bit register | `short`, `__half` |
| `"c"` | Predicate register | N/A (rarely used directly) |
| `"="` | Write-only output | `"=r"(result)` |
| `"+"` | Read-write | `"+r"(accumulator)` |

**Example: Fused multiply-add with inline PTX:**
```cpp
__device__ float fma_inline(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)      // %0: output float register
        : "f"(a),           // %1: input float
          "f"(b),           // %2: input float
          "f"(c));          // %3: input float
    return result;
}
```

From [Tutorial: Understanding GPU Assembly with PTX](https://eunomia.dev/others/cuda-tutorial/02-ptx-assembly/) (accessed 2025-11-13):
> "The 'fma.rn.f32' instruction performs fused multiply-add with round-to-nearest mode. This single instruction is both faster and more accurate than separate multiply and add because it performs rounding only once."

**Volatile keyword for side effects:**
```cpp
__device__ void atomic_custom(int* addr, int val) {
    asm volatile("atom.global.add.s32 [%0], %1;"
        : /* no outputs */
        : "l"(addr), "r"(val)
        : "memory");  // Clobber: compiler must assume memory changed
}
```

The `volatile` keyword prevents:
- Dead code elimination
- Reordering across other volatile operations
- Optimization that assumes no side effects

From [Stack Overflow: inline PTX volatile](https://stackoverflow.com/questions/61497980/in-asm-volatile-inline-ptx-instructions-why-also-specify-memory-side-effecs) (accessed 2025-11-13):
> "To ensure that the asm is not deleted or moved during generation of PTX, you should use the volatile keyword. A non-volatile inline asm statement is treated as a pure function of its inputs."

### Advanced Inline PTX Examples

**Warp shuffle for efficient reduction:**
```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        asm("shfl.sync.down.b32 %0|%1, %2, %3, 0x1f, 0xffffffff;"
            : "=f"(val), "=r"(temp)  // Output: shuffled value + predicate
            : "f"(val), "r"(offset)   // Input: value + offset
            : );
    }
    return val;
}
```

**Custom atomic min for floats:**
```cpp
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float assumed_float = __int_as_float(assumed);
        float min_val = fminf(value, assumed_float);
        int min_int = __float_as_int(min_val);

        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;"
            : "=r"(old)
            : "l"(addr_as_int), "r"(assumed), "r"(min_int)
            : "memory");
    } while (assumed != old);
    return __int_as_float(old);
}
```

**Memory fence control for precise ordering:**
```cpp
__device__ void write_with_fence(int* ptr, int value) {
    *ptr = value;
    asm volatile("membar.gl;" ::: "memory");
    // Ensures all global memory writes complete before continuing
}
```

### Debugging Inline PTX Issues

**Common inline PTX errors:**

1. **Register constraint mismatch:**
```cpp
// WRONG: Using "r" for float
asm("add.f32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
//                             ^^^ Should be "f"

// CORRECT:
asm("add.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
```

2. **Missing volatile for side effects:**
```cpp
// WRONG: Compiler may optimize away
asm("st.global.u32 [%0], %1;" : : "l"(ptr), "r"(val));

// CORRECT: Use volatile for stores
asm volatile("st.global.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
```

3. **Incorrect operand numbering:**
```cpp
// Operands numbered from %0 in order: outputs, then inputs
asm("mad.lo.s32 %0, %1, %2, %3;"  // %0=out, %1=in1, %2=in2, %3=in3
    : "=r"(result)                 // %0
    : "r"(a), "r"(b), "r"(c));    // %1, %2, %3
```

**Verify inline PTX compilation:**
```bash
# Compile to PTX to see generated assembly
nvcc -ptx -lineinfo kernel.cu -o kernel.ptx

# Check that your inline asm appears correctly
grep -A5 "your_function_name" kernel.ptx
```

### Production Debugging Workflow for Assembly Issues

**7-step assembly-level debugging process:**

1. **Compile with line info and keep intermediates:**
```bash
nvcc -lineinfo -keep -O3 kernel.cu -o kernel
# Generates: kernel.ptx, kernel.cubin, kernel.fatbin
```

2. **Extract SASS and correlate with source:**
```bash
cuobjdump --dump-sass --function-name="myKernel" kernel.cubin > kernel.sass
# Look for @lineinfo annotations mapping SASS to source lines
```

3. **Check register usage and spilling:**
```bash
nvcc -Xptxas -v kernel.cu 2>&1 | grep "registers\|spill\|lmem"
```

4. **Profile with NSight Compute Source View:**
```bash
ncu --set full --section SourceCounters --target-processes all -o profile ./kernel
# Open profile.ncu-rep in UI
# Navigate to Source View → correlate C++ / PTX / SASS
```

5. **Analyze warp stall reasons:**
```bash
ncu --metrics smsp__warp_issue_stalled_long_scoreboard.avg,\
smsp__warp_issue_stalled_short_scoreboard.avg,\
smsp__warp_issue_stalled_wait.avg ./kernel
```

6. **Verify instruction mix and pipeline utilization:**
```bash
ncu --metrics smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_fp64.sum ./kernel
```

7. **If using inline PTX, test variants:**
```cpp
// Create A/B test version
#ifdef USE_INLINE_PTX
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" ...);
#else
    result = __fmaf_rn(a, b, c);  // Intrinsic version
#endif
```

### SASS-Level Race Condition Detection

**Use compute-sanitizer for memory races:**
```bash
compute-sanitizer --tool racecheck ./kernel

# Example output:
# Race reported between Write access at kernel.cu:42
# and Read access at kernel.cu:45 in different warps
```

**Inspect SASS for race-prone patterns:**
```sass
# Potential race: unsynchronized shared memory access
STG.E [R2], R1          // Thread 0 writes
...
LDS.U32 R5, [R2]        // Thread 1 reads (no __syncthreads!)
```

**Fix requires synchronization:**
```cpp
__shared__ float smem[256];
smem[tid] = value;
__syncthreads();  // Ensures all threads complete write
float result = smem[other_tid];
```

### Architecture-Specific SASS Debugging

**Hopper (sm_90) specific: WGMMA debugging:**
```bash
# Check if Tensor Core WGMMA instructions are used
cuobjdump --dump-sass kernel.cubin | grep WGMMA

# Example WGMMA instruction:
# WGMMA.MMA.F16.F16 {...}, {...}, {...}, P0, 1, 1, 0;
```

**Ada (sm_89) specific: FP8 verification:**
```bash
# Verify FP8 Tensor Core usage
ncu --metrics smsp__inst_executed_pipe_tensor_op_hmma.sum ./kernel

# Check for FP8 format instructions in SASS
cuobjdump --dump-sass kernel.cubin | grep "\.E4M3\|\.E5M2"
```

---

## Sources

**Source Documents:**
- [cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) - PTX mma.sync, WMMA API, LDMATRIX
- [cuda/10-performance-debugging-profiling-expert.md](../cuda/10-performance-debugging-profiling-expert.md) - NSight Compute profiling, optimization
- [cuda/12-kernel-debugging-internals-expert.md](../cuda/12-kernel-debugging-internals-expert.md) - NSight Compute Source View, cuda-gdb

**Web Research:**

- [Tutorial: Understanding GPU Assembly with PTX - eunomia-bpf](https://eunomia.dev/others/cuda-tutorial/02-ptx-assembly/) (accessed 2025-11-13)
  - PTX fundamentals, compilation pipeline, inline assembly syntax
  - PTX instruction types and memory spaces
  - Hands-on examples of PTX programming

- [Reversing Nvidia GPU's SASS code - JEB Decompiler](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/) (accessed 2025-11-13)
  - SASS instruction set architecture (sm_70 to sm_121)
  - cuobjdump detailed usage for PTX/SASS extraction
  - Register types: general, predicate, special, uniform
  - Instruction classes and execution environment

- [NVIDIA Docs: Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) (accessed 2025-11-13)
  - Inline PTX syntax and constraint letters
  - Volatile keyword usage for side effects
  - Operand mapping and register constraints

- [NSight Compute SASS analysis documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (referenced 2025-11-13)
  - Instruction throughput metrics
  - Source View PTX/SASS/C++ correlation
  - Warp execution efficiency analysis

- [How to Improve CUDA Kernel Performance with Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/) (accessed 2025-11-13)
  - CUDA 13.0 ptxas optimization for register spills
  - Shared memory as spill target (better than local memory)

- [Stack Overflow: Understanding ptxas output](https://stackoverflow.com/questions/12388207/interpreting-the-verbose-output-of-ptxas-part-i) (accessed 2025-11-13)
  - Register usage interpretation
  - Spill stores/loads detection
  - Local memory vs register usage

- [Stack Overflow: inline PTX volatile keyword](https://stackoverflow.com/questions/61497980/in-asm-volatile-inline-ptx-instructions-why-also-specify-memory-side-effecs) (accessed 2025-11-13)
  - Volatile semantics for inline assembly
  - Memory clobbering and side effects

**Additional References:**
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) - Complete PTX instruction set
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) - cuobjdump, nvdisasm documentation
- [GPU Architecture Whitepapers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) - Volta through Blackwell architectures
