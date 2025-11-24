# CUDA Kernel Debugging Internals (Ultra-Expert)

**Deep cuda-gdb workflows, device-side debugging, NSight Compute interactive profiling, advanced breakpoint techniques, and production kernel debugging**

---

## Section 1: cuda-gdb Fundamentals (~125 lines)

### Overview: Interactive CUDA Debugger

cuda-gdb is GDB extended for simultaneous CPU/GPU debugging. Supports:
- **Kernel breakpoints** - Pause execution inside running kernels
- **Thread/warp/block inspection** - Navigate GPU execution hierarchy
- **Device state inspection** - View registers, shared/local/global memory
- **Source-level debugging** - Step through CUDA C/C++ code line-by-line

From [NVIDIA cuda-gdb Documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html) (accessed 2025-11-13):
> "cuda-gdb allows the user to set breakpoints, to single-step CUDA applications, and also to inspect and modify the memory and variables of any CUDA thread running on the GPU."

### Compilation Requirements

**Always compile with `-g -G` for full debug information:**

```bash
# Correct: Generate host and device debug symbols
nvcc -g -G my_kernel.cu -o my_program

# -g  : Host debug info
# -G  : Device debug info (disables optimizations)
```

**Without `-G`, you get:**
- ❌ No source-level debugging in kernels
- ❌ No variable inspection
- ❌ Only SASS/PTX assembly debugging

### Essential cuda-gdb Commands

From [NSight Visual Studio State Inspection](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-inspect-state/index.html) (accessed 2025-11-13):

**Launch and Attach:**
```bash
# Launch application under cuda-gdb
cuda-gdb ./my_program

# Attach to running process (requires NSIGHT_CUDA_DEBUGGER=1 env var)
cuda-gdb -p <pid>
```

**Kernel Breakpoints:**
```gdb
# Set breakpoint in kernel function
(cuda-gdb) break my_kernel

# Set breakpoint at specific line in .cu file
(cuda-gdb) break my_kernel.cu:42

# Set conditional breakpoint
(cuda-gdb) break my_kernel if threadIdx.x == 0

# Set breakpoint on register value (use $R prefix)
(cuda-gdb) break my_kernel if $R9 > 100
```

**Thread/Warp/Block Navigation:**
```gdb
# Show current CUDA focus
(cuda-gdb) cuda thread
(cuda-gdb) cuda block
(cuda-gdb) cuda kernel

# Switch focus to specific thread
(cuda-gdb) cuda thread (0,0,0)

# Switch focus to specific block
(cuda-gdb) cuda block (1,0,0)

# Navigate between warps
(cuda-gdb) cuda kernel 0 block (0,0,0) warp 2 lane 5
```

**Info Commands (Query Device State):**
```gdb
# List all CUDA threads
(cuda-gdb) info cuda threads

# List all CUDA blocks
(cuda-gdb) info cuda blocks

# List all CUDA kernels
(cuda-gdb) info cuda kernels

# Show device info
(cuda-gdb) info cuda devices

# Show SMS and warps
(cuda-gdb) info cuda sms
(cuda-gdb) info cuda warps

# Show launch configuration
(cuda-gdb) info cuda launch
```

**Memory Inspection:**
```gdb
# Print variable (current thread)
(cuda-gdb) print my_var

# Print shared memory (cast required)
(cuda-gdb) print *(__shared__ float*)0x0

# Print global memory
(cuda-gdb) print *(__device__ float*)0x2001b000

# Print local memory
(cuda-gdb) print *(__local__ int*)0x0

# Examine memory range
(cuda-gdb) x/10f shared_array
```

**Step Commands:**
```gdb
# Step single thread (freezes other threads in warp)
(cuda-gdb) step

# Continue all threads
(cuda-gdb) continue

# Step over function call
(cuda-gdb) next

# Finish current function
(cuda-gdb) finish
```

### Advanced: cuda-gdb Workflow for Index Errors

**Scenario:** Kernel crashes with illegal memory access.

```bash
# 1. Compile with debug info
nvcc -g -G kernel.cu -o test

# 2. Run under cuda-gdb
cuda-gdb ./test

# 3. Set breakpoint before crash
(cuda-gdb) break kernel.cu:50
(cuda-gdb) run

# 4. When breakpoint hits, inspect indices
(cuda-gdb) print threadIdx.x
(cuda-gdb) print blockIdx.x
(cuda-gdb) print blockDim.x

# 5. Calculate actual index
(cuda-gdb) print blockIdx.x * blockDim.x + threadIdx.x

# 6. Check against array bounds
(cuda-gdb) print array_size
(cuda-gdb) print (blockIdx.x * blockDim.x + threadIdx.x) < array_size

# 7. If false, you found the bug!
```

From [NVIDIA Developer Forums: cuda-gdb debugging workflow](https://forums.developer.nvidia.com/t/cuda-gdb-debug-on-jetson-orin-nano/317222) (accessed 2025-11-13):
> "Set CUDA_LAUNCH_BLOCKING=1 to force synchronous kernel launches. This makes cuda-gdb breakpoints more reliable and error messages more accurate."

### Limitations and Gotchas

**cuda-gdb cannot:**
- ❌ Modify device memory directly (read-only in Memory window)
- ❌ Set breakpoints in optimized code (need `-G` flag)
- ❌ Debug code without debug symbols
- ❌ Attach to already-running kernels (must launch under debugger)

**Thread focus changes automatically:**
- When you hit a breakpoint, focus switches to the thread that hit it
- This can be confusing when debugging multiple threads
- Use `cuda thread` frequently to confirm current focus

---

## Section 2: Device-Side Debugging (~125 lines)

### Device-Side printf() Debugging

**printf() is the most accessible kernel debugging tool**, but has critical limitations.

From [NVIDIA CUDA Programming Guide - printf](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (accessed 2025-11-13):

**Basic Usage:**
```cuda
__global__ void my_kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Print from thread 0 only
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Block %d, Thread %d: data[%d] = %f\n",
               blockIdx.x, threadIdx.x, idx, data[idx]);
    }
}
```

**Critical: printf Buffer Size Limits**

From [NVIDIA Developer Forums: Size of printf buffer](https://forums.developer.nvidia.com/t/size-of-printf-buffer/160957) (accessed 2025-11-13):
> "The default printf buffer is 1MB. You can increase it with cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_in_bytes). However, buffer overflow silently truncates output without warning!"

```cuda
// Set larger printf buffer (before kernel launch)
size_t new_size = 8 * 1024 * 1024; // 8MB
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, new_size);

// Query current buffer size
size_t current_size;
cudaDeviceGetLimit(&current_size, cudaLimitPrintfFifoSize);
printf("Printf buffer: %zu bytes\n", current_size);
```

**Argument Limit: 32 Arguments Maximum**

From [Stack Overflow: Printf() 45 arguments in CUDA](https://stackoverflow.com/questions/79165817/printf-45-arguments-in-cuda) (accessed 2025-11-13):
> "The printf() command can accept at most 32 arguments in addition to the format string. Additional arguments beyond this will be ignored silently."

```cuda
// ❌ WRONG: More than 32 arguments (will fail silently)
printf("%d %d %d ... (35 arguments total) ...", a, b, c, ...);

// ✅ CORRECT: Split into multiple prints
printf("Part 1: %d %d %d\n", a, b, c);
printf("Part 2: %d %d %d\n", d, e, f);
```

**Performance Impact of printf()**

From [NVIDIA Developer Blog: Debugging CUDA More Efficiently](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/) (accessed 2025-11-13):

**printf() overhead:**
- **~1000 cycles per call** (extremely slow compared to normal instructions)
- Serializes warp execution (breaks parallelism)
- Flushes buffer to host (synchronous operation)

**Best practice: Conditional printf with compile flag:**
```cuda
#ifdef DEBUG_PRINT
    #define DPRINTF(...) printf(__VA_ARGS__)
#else
    #define DPRINTF(...) do {} while(0)
#endif

__global__ void kernel() {
    DPRINTF("Debug: idx=%d\n", threadIdx.x);  // No-op in release
}
```

Compile with: `nvcc -DDEBUG_PRINT kernel.cu` for debug builds.

### Device-Side Assertions

**assert() works in device code** (CUDA 4.0+), triggering immediate kernel termination.

```cuda
__global__ void kernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Assert index is valid
    assert(idx < size && "Index out of bounds!");

    data[idx] = idx * 2;
}
```

**When assertion fails:**
```
cuda_assert.cu:15: void kernel(int*, int): block: [1,0,0], thread: [128,0,0]
Assertion `idx < size` failed.
```

**Compile with `-G` to get source file/line in assertion messages.**

From [NVIDIA Developer Forums: Using assert in CUDA code](https://forums.developer.nvidia.com/t/using-assert-in-cuda-code/21816) (accessed 2025-11-13):
> "Assertions are only enabled with -G flag. In optimized builds (-O2, -O3), assert() becomes a no-op for performance."

**Custom assertions with trap():**
```cuda
#define CUDA_ASSERT(condition, msg) \
    if (!(condition)) { \
        printf("ASSERTION FAILED: %s\n", msg); \
        asm("trap;"); \
    }

__global__ void kernel() {
    CUDA_ASSERT(threadIdx.x < 256, "Thread index too large");
}
```

**`trap()` immediately terminates the warp** (useful for catching errors early).

### Device-Side Error Codes (TORCH_USE_CUDA_DSA)

From [PyTorch Forums: device-side assert triggered](https://discuss.pytorch.org/t/practical-tips-for-runtimeerror-cuda-error-device-side-assert-triggered/157167) (accessed 2025-11-13):

**Enable device-side assertions in PyTorch:**
```bash
# Set environment variable
export TORCH_USE_CUDA_DSA=1

# Now you get helpful assertion messages instead of generic "CUDA error"
```

**Before (generic error):**
```
RuntimeError: CUDA error: device-side assert triggered
```

**After (with TORCH_USE_CUDA_DSA=1):**
```
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call.
Assertion failed: index < num_classes
at line 42 in embedding.cu
```

**Set CUDA_LAUNCH_BLOCKING=1 for synchronous error reporting:**
```bash
export CUDA_LAUNCH_BLOCKING=1
```

This forces kernels to complete before returning, making stack traces accurate.

### Printf Debugging Patterns (Production-Safe)

**Pattern 1: Thread-Selective Printing**
```cuda
// Print from single thread to avoid spam
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("First thread only\n");
}

// Print from warp 0 only
if ((threadIdx.x % 32) == 0 && blockIdx.x == 0) {
    printf("Warp leader: warp_id=%d\n", threadIdx.x / 32);
}
```

**Pattern 2: Sampling (Every Nth Thread)**
```cuda
// Print from every 1000th thread
if (idx % 1000 == 0) {
    printf("Sample idx=%d, value=%f\n", idx, data[idx]);
}
```

**Pattern 3: Conditional on Error**
```cuda
float result = compute();
if (isnan(result) || isinf(result)) {
    printf("NaN/Inf at idx=%d, input=%f\n", idx, input[idx]);
}
```

**Pattern 4: Hexadecimal Dump (for bit-level debugging)**
```cuda
// Print raw bits (useful for FP precision issues)
float val = data[idx];
unsigned int bits = __float_as_uint(val);
printf("idx=%d, float=%f, bits=0x%08x\n", idx, val, bits);
```

---

## Section 3: NSight Compute Interactive Debugging (~125 lines)

### NSight Compute Overview

**NSight Compute = CUDA kernel profiler + interactive debugger** (not just metrics collector).

From [NVIDIA NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (accessed 2025-11-13):

**Key debugging features:**
- **Source View** - Correlate SASS/PTX assembly with C++ source
- **Warp Analysis** - Inspect warp divergence and predication
- **Kernel Replay** - Re-run kernel multiple times for metric collection
- **Memory Access Patterns** - Visualize bank conflicts, coalescing issues

### Source View Debugging

**Launch NSight Compute in UI mode:**
```bash
# Interactive GUI (requires X11 or WSL2 with GUI)
ncu-ui

# Or launch from command line with profile
ncu --set full --target-processes all -o profile ./my_app
```

**Navigate to kernel in Source View:**
1. Select kernel from kernel list
2. Click "Source" tab
3. See **three panes:**
   - **Source Code** (CUDA C++)
   - **PTX** (intermediate representation)
   - **SASS** (actual GPU assembly)

**Correlating source to assembly:**
- Click on source line → corresponding PTX/SASS highlights
- See **instruction mix** per source line (loads, stores, FMAs)
- Identify **expensive operations** (e.g., `__syncthreads()` shows as `BAR.SYNC`)

From [NASA HECC: NSight Compute kernel analysis](https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html) (accessed 2025-11-13):
> "Source View allows developers to see exactly which assembly instructions are generated for each source line, essential for understanding compiler optimizations and identifying performance bottlenecks."

### Warp Analysis and Divergence Detection

**Warp State Visualization:**

NSight Compute shows **per-warp execution state:**
- **Active** - Thread executing
- **Inactive** - Thread predicated out (divergence!)
- **Blocked** - Waiting on synchronization
- **Not Launched** - Thread not in grid

From [NSight Visual Studio State Inspection: Warp Info](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-inspect-state/index.html#warp-info) (accessed 2025-11-13):

**Warp divergence debugging:**
```cuda
__global__ void divergent_kernel(int *data) {
    int idx = threadIdx.x;

    // This causes divergence!
    if (idx < 16) {
        data[idx] = expensive_compute();  // Only half of warp executes
    }
}
```

**NSight Compute metrics for divergence:**
```bash
ncu --metrics smsp__sass_thread_inst_executed_op_dadd_pred_on.avg,\
              smsp__sass_thread_inst_executed_op_dadd_pred_off.avg \
    ./my_app
```

**Metrics reveal:**
- `pred_on` - Instructions with predicate enabled (active threads)
- `pred_off` - Instructions predicated off (inactive threads)
- **High pred_off ratio = severe divergence**

**Fix: Avoid branch divergence within warps:**
```cuda
// ✅ BETTER: All threads in warp take same path
__global__ void non_divergent_kernel(int *data) {
    int idx = threadIdx.x;
    int warp_id = idx / 32;

    // Process entire warps together
    if (warp_id == 0) {
        data[idx] = expensive_compute();
    }
}
```

### Kernel Replay Debugging

**NSight Compute re-runs kernels multiple times** to collect complete metric sets.

From [Forschungszentrum Jülich: CUDA Tools Profiling](https://juser.fz-juelich.de/record/1019123/files/02_cuda_tools_mhrywniak.pdf) (accessed 2025-11-13):

**Replay behavior:**
- Kernel runs **5-10 times** (varies by GPU)
- Each run collects different metric subset
- **Side effects happen multiple times!**

**Problem: printf appears multiple times:**
```cuda
__global__ void kernel() {
    printf("Hello\n");  // Prints 5-10 times under NSight!
}
```

From [Stack Overflow: CUDA kernel launched from Nsight Compute gives inconsistent results](https://stackoverflow.com/questions/74094838/cuda-kernel-launched-from-nsight-compute-gives-inconsistent-results) (accessed 2025-11-13):
> "Nsight Compute runs a kernel multiple times to collect all of its information. So things like print statements in the kernel will show up multiple times."

**Workaround: Skip replay for debugging:**
```bash
# Run kernel only once (minimal metrics)
ncu --replay-mode application ./my_app

# Default replay mode (full metrics, multiple runs)
ncu --replay-mode kernel ./my_app
```

**Race condition debugging with replay:**
- If kernel has race conditions, **results may differ between replays**
- NSight will warn: "Kernel results inconsistent across replays"
- Use `--check-exit-code` to detect non-deterministic behavior

### Memory Access Pattern Debugging

**Bank Conflict Detection:**
```bash
# Check for shared memory bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./my_app
```

**Interpretation:**
- **0 conflicts** - Perfect access pattern
- **< 10% of accesses** - Acceptable
- **> 50% of accesses** - Severe problem (fix needed)

**Common bank conflict pattern:**
```cuda
__shared__ float shared[32][32];

// ❌ WRONG: All threads in warp access same bank
shared[threadIdx.x][0] = data[idx];  // Column 0 → bank conflict!

// ✅ CORRECT: Transpose access pattern
shared[0][threadIdx.x] = data[idx];  // Row 0 → no conflict
```

**Global Memory Coalescing:**
```bash
# Check memory transaction efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_elapsed,\
              l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    ./my_app
```

**Metrics reveal:**
- **Low sectors_per_request (< 8)** - Poor coalescing
- **High sector count** - Many transactions per request (uncoalesced)

---

## Section 4: Advanced Debugging Techniques (~125 lines)

### Race Condition Detection

From [NVIDIA Developer Forums: Race condition within warp](https://forums.developer.nvidia.com/t/race-condition-within-warp/44951) (accessed 2025-11-13):

**Intra-warp race conditions are IMPOSSIBLE** (SIMT execution guarantees synchronization within warp).

**BUT:** Shared memory races occur **between warps:**

```cuda
__shared__ int counter;

__global__ void racy_kernel() {
    // ❌ RACE CONDITION: Multiple warps increment without atomics
    if (threadIdx.x == 0) {
        counter++;  // NOT SAFE!
    }
    __syncthreads();
}
```

**Fix with atomics:**
```cuda
if (threadIdx.x == 0) {
    atomicAdd(&counter, 1);  // ✅ SAFE
}
```

**Detect race with cuda-memcheck (deprecated, use compute-sanitizer):**
```bash
# Modern tool (CUDA 11.0+)
compute-sanitizer --tool racecheck ./my_app

# Old tool (CUDA < 11.0)
cuda-memcheck --tool racecheck ./my_app
```

**Example output:**
```
========= RACECHECK SUMMARY
========= Potential RAW hazard detected at 0x400b20 in block (0,0,0):
=========     Write Thread (31,0,0)
=========     Read  Thread (0,0,0)
```

From [SHARCNET: Debugging CUDA programs](https://helpwiki.sharcnet.ca/wiki/images/1/10/CUDA_debugging_webinar_2016.pdf) (accessed 2025-11-13):
> "Shared memory levels are prone to race condition bugs. Both shared and distributed memory levels can have deadlock bugs."

### Minimal Overhead Debugging (Production)

**Problem:** `-G` flag disables optimizations → unacceptable for production.

**Solution: Selective instrumentation with debug builds:**

```cmake
# CMakeLists.txt: Separate debug/release configs
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -DDEBUG_BUILD)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -DNDEBUG)
endif()
```

**Use debug macros sparingly:**
```cuda
#ifdef DEBUG_BUILD
    #define DEBUG_CHECK(cond, msg) \
        if (!(cond)) { printf("ERROR: %s\n", msg); }
#else
    #define DEBUG_CHECK(cond, msg) do {} while(0)
#endif

__global__ void kernel() {
    DEBUG_CHECK(threadIdx.x < 256, "Thread overflow");
    // ... production code (optimized in release) ...
}
```

**Minimal overhead: Use CUDA events for timing:**
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<grid, block>>>();
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);

printf("Kernel time: %f ms\n", ms);  // < 1 μs overhead
```

### Memory Corruption Debugging

**compute-sanitizer memcheck (detects illegal access):**
```bash
compute-sanitizer --tool memcheck ./my_app
```

**Common issues detected:**
- **Out-of-bounds access** - `data[idx]` where `idx >= array_size`
- **Uninitialized memory read** - Using malloc without initialization
- **Use-after-free** - Accessing freed memory
- **Double-free** - Calling cudaFree twice

**Example output:**
```
========= MEMCHECK
========= Invalid __global__ write of size 4 bytes
=========     at 0x400b20 in kernel(float*)
=========     by thread (127,0,0) in block (0,0,0)
=========     Address 0x7f8e40000400 is out of bounds
```

### Conditional Breakpoints for Specific Threads

**Break only when specific condition is true:**

```gdb
# Break when thread (5,0,0) in block (2,0,0) accesses invalid index
(cuda-gdb) break kernel.cu:42 if threadIdx.x == 5 && blockIdx.x == 2 && idx >= array_size

# Break when register R9 exceeds threshold
(cuda-gdb) break kernel.cu:50 if $R9 > 1000

# Break when shared memory pointer is null
(cuda-gdb) break kernel.cu:60 if s_ptr == 0x0
```

**Watchpoints (break on memory change):**
```gdb
# Watch global variable (breaks when modified)
(cuda-gdb) watch *(__device__ int*)0x2001b000

# Watch shared memory location
(cuda-gdb) watch *(__shared__ float*)s_data

# Conditional watch
(cuda-gdb) watch data[42] if data[42] > 100.0
```

### Advanced: Debugging Warp Shuffle Bugs

**Warp shuffles have subtle requirements** (threads must be active).

```cuda
// ❌ BUG: Shuffle with divergent warp
__global__ void buggy_shuffle() {
    int val = threadIdx.x;

    if (threadIdx.x < 16) {
        val = __shfl_xor_sync(0xffffffff, val, 1);  // WRONG: assumes full warp
    }
}
```

**Problem:** `__shfl` requires **all threads in mask to be active**.

**Fix: Use correct active mask:**
```cuda
// ✅ CORRECT: Mask matches active threads
if (threadIdx.x < 16) {
    unsigned int mask = 0x0000ffff;  // Only lower 16 threads
    val = __shfl_xor_sync(mask, val, 1);
}
```

**Debug with assertions:**
```cuda
int val = threadIdx.x;
unsigned int active_mask = __activemask();

// Assert all threads in intended mask are active
assert((active_mask & 0xffffffff) == 0xffffffff);

val = __shfl_xor_sync(0xffffffff, val, 1);
```

### Production Debugging Workflow (Real-World)

**1. Reproduce in minimal test case:**
```bash
# Extract failing kernel to standalone .cu file
# Run with small input (easier to debug)
```

**2. Run compute-sanitizer first (fast, catches 80% of bugs):**
```bash
compute-sanitizer --tool memcheck ./test
compute-sanitizer --tool racecheck ./test
compute-sanitizer --tool synccheck ./test  # Detects invalid __syncthreads()
```

**3. If sanitizer finds nothing, use cuda-gdb:**
```bash
# Compile debug build
nvcc -g -G -DDEBUG_BUILD test.cu -o test_debug

# Run under cuda-gdb
cuda-gdb ./test_debug
(cuda-gdb) break kernel
(cuda-gdb) run
```

**4. Inspect state at breakpoint:**
```gdb
# Check indices
(cuda-gdb) print threadIdx.x
(cuda-gdb) print blockIdx.x
(cuda-gdb) print idx

# Check memory values
(cuda-gdb) print data[idx]
(cuda-gdb) print *(__shared__ float*)s_data

# Check for NaN/Inf
(cuda-gdb) print isnan(result)
(cuda-gdb) print isinf(result)
```

**5. If issue only appears in optimized build:**
```bash
# Use NSight Compute to profile optimized kernel
ncu --set full -o profile ./test_release

# Compare metrics between debug/release builds
# Look for warp divergence, bank conflicts, uncoalesced access
```

---

## Sources

**Source Documents:**
- cuda/09-runtime-errors-debugging-expert.md (existing knowledge, lines 270-296)

**Web Research:**
- [NVIDIA cuda-gdb Documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html) - Complete cuda-gdb reference (accessed 2025-11-13)
- [NVIDIA NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - Interactive debugging workflows (accessed 2025-11-13)
- [NSight Visual Studio State Inspection](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-inspect-state/index.html) - Warp/thread inspection (accessed 2025-11-13)
- [NVIDIA Developer Forums: Size of printf buffer](https://forums.developer.nvidia.com/t/size-of-printf-buffer/160957) - Printf limitations (accessed 2025-11-13)
- [NVIDIA Developer Forums: Using assert in CUDA code](https://forums.developer.nvidia.com/t/using-assert-in-cuda-code/21816) - Device-side assertions (accessed 2025-11-13)
- [NVIDIA Developer Forums: Race condition within warp](https://forums.developer.nvidia.com/t/race-condition-within-warp/44951) - Warp synchronization (accessed 2025-11-13)
- [NVIDIA Developer Blog: Debugging CUDA More Efficiently](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/) - Compute Sanitizer usage (accessed 2025-11-13)
- [PyTorch Forums: device-side assert triggered](https://discuss.pytorch.org/t/practical-tips-for-runtimeerror-cuda-error-device-side-assert-triggered/157167) - PyTorch debugging (accessed 2025-11-13)
- [Stack Overflow: Printf() 45 arguments in CUDA](https://stackoverflow.com/questions/79165817/printf-45-arguments-in-cuda) - Printf argument limits (accessed 2025-11-13)
- [Stack Overflow: CUDA kernel launched from Nsight Compute](https://stackoverflow.com/questions/74094838/cuda-kernel-launched-from-nsight-compute-gives-inconsistent-results) - NSight replay behavior (accessed 2025-11-13)
- [NASA HECC: NSight Compute kernel analysis](https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html) - Source view debugging (accessed 2025-11-13)
- [Forschungszentrum Jülich: CUDA Tools Profiling](https://juser.fz-juelich.de/record/1019123/files/02_cuda_tools_mhrywniak.pdf) - Kernel replay (accessed 2025-11-13)
- [SHARCNET: Debugging CUDA programs](https://helpwiki.sharcnet.ca/wiki/images/1/10/CUDA_debugging_webinar_2016.pdf) - Race conditions (accessed 2025-11-13)

**Additional References:**
- [Carnegie Mellon cuda-gdb PDF](https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/cuda-gdb.pdf) - Academic cuda-gdb guide
- [Massed Compute: How to use CUDA Debugger](https://massedcompute.com/faq-answers/?question=How%20to%20use%20CUDA%20Debugger%20to%20step%20through%20a%20CUDA%20kernel?) - Practical debugging examples
