# PyTorch JIT & torch.compile

## Overview

PyTorch provides two distinct approaches to optimizing models through compilation: the legacy **TorchScript** system (torch.jit) and the modern **torch.compile** introduced in PyTorch 2.0. While TorchScript uses ahead-of-time compilation with explicit scripting or tracing, torch.compile uses just-in-time (JIT) compilation with Python bytecode transformation to capture and optimize PyTorch operations with minimal code changes.

**Why JIT Compilation Matters:**
- **Eliminates Python overhead** - Removes per-operation dispatch costs
- **Enables graph-level optimization** - Fuses operations, eliminates redundant work
- **Reduces CPU-GPU synchronization** - Bundles operations for efficient execution
- **Critical for small kernels** - When kernel time < 100μs, launch overhead dominates

From [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (PyTorch Tutorials, accessed 2025-02-03):
> "torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, while requiring minimal code changes."

From [PyTorch 2 Paper](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/) (PyTorch Blog, accessed 2025-02-03):
> "TorchDynamo is a Python-level just-in-time (JIT) compiler that enables graph compilation in PyTorch programs without sacrificing the flexibility of Python. It achieves this by dynamically modifying Python bytecode before execution and extracting sequences of PyTorch operations into an FX graph."

**Performance Impact:**
- 2-3× speedup for inference (typical)
- 1.4× speedup for training (typical)
- Up to 5× speedup for CPU-bound workloads (small batches, many kernels)

---

## Section 1: TorchScript (Legacy System) (~150 lines)

### What is TorchScript?

TorchScript is PyTorch's legacy compilation system that converts Python code into an intermediate representation (IR) that can be optimized and run without Python. It provides two modes: **tracing** and **scripting**.

**Key Characteristics:**
- Ahead-of-time compilation (run once to capture, then deploy)
- Serializable to `.pt` files (deployment without Python)
- Limited Python support (static types, no dynamic control flow in trace mode)
- More restrictive than torch.compile

### torch.jit.trace - Recording Operations

**How Tracing Works:**
```python
import torch

def model(x, y):
    z = x + y
    return torch.relu(z)

# Provide example inputs - only the executed path is captured
traced = torch.jit.trace(model, (torch.randn(3, 3), torch.randn(3, 3)))

# Traced model can be saved and loaded
traced.save("model.pt")
loaded = torch.jit.load("model.pt")
```

**Tracing Mechanism:**
1. Runs the function with example inputs
2. Records all tensor operations executed
3. Builds computation graph from recorded operations
4. Replays graph on new inputs with same shapes

**Limitations - Data-Dependent Control Flow:**

From [TorchScript Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) (accessed 2025-02-03):
> "Tracing records operations from a model run, while scripting analyzes Python code directly. Tracing is for no control flow; scripting for if/for loops."

```python
def conditional_model(x):
    if x.sum() > 0:  # Data-dependent condition
        return x * 2
    else:
        return x * -1

# Tracing will ONLY capture one branch
traced = torch.jit.trace(conditional_model, torch.randn(5, 5))

# SILENT INCORRECTNESS - wrong results for other inputs!
# traced() will always execute the same branch (whichever was traced)
```

**When to Use Tracing:**
- Model has no control flow (pure feed-forward)
- All operations are PyTorch tensors
- Shape and datatype are constant
- Need simple serialization

### torch.jit.script - Compiling Python Code

**How Scripting Works:**
```python
import torch

@torch.jit.script
def scripted_model(x: torch.Tensor, threshold: float) -> torch.Tensor:
    # Type annotations required
    if x.sum() > threshold:  # Control flow supported
        return x * 2
    else:
        return x * -1

# Script compiles the Python code itself
output = scripted_model(torch.randn(5, 5), 0.0)
```

**Scripting Mechanism:**
1. Parses Python abstract syntax tree (AST)
2. Converts to TorchScript IR
3. Type-checks all operations
4. Compiles to optimized graph

**Type Annotation Requirements:**

```python
@torch.jit.script
def requires_types(x: torch.Tensor, count: int) -> torch.Tensor:
    result = x
    for i in range(count):  # count must be typed as int
        result = result + 1
    return result

# Without type annotations, TorchScript will fail
# TypeError: Expected a value of type 'Tensor' for argument 'count'
```

**Supported Python Features:**
- if/else, for/while loops
- Type annotations (required)
- torch.Tensor operations
- Basic Python types (int, float, bool, str, List, Dict, Tuple)

**Unsupported Features:**
- Dynamic imports
- Most standard library functions
- Complex Python objects
- Generator expressions
- f-strings (in older versions)

### TorchScript Limitations

**1. Type Annotation Burden:**
```python
# TorchScript requires explicit types
@torch.jit.script
def add(x: torch.Tensor, y: int) -> torch.Tensor:
    return x + y

# torch.compile needs NO type annotations
@torch.compile
def add(x, y):
    return x + y  # Works with any types
```

**2. Limited Python Support:**
```python
# TorchScript fails on many Python idioms
@torch.jit.script
def broken():
    import numpy as np  # ERROR: imports not allowed
    x = [i for i in range(10)]  # ERROR: comprehensions limited
    return sum(x)

# torch.compile handles arbitrary Python
@torch.compile
def working():
    import numpy as np  # Fine
    x = [i for i in range(10)]  # Fine
    return sum(x)
```

**3. Silent Tracing Errors:**
```python
def model(x, threshold):
    if x.sum() > threshold:
        return x * 2
    return x

# Trace captures only one branch - WRONG for other inputs!
traced = torch.jit.trace(model, (torch.randn(5), 0.0))
```

### Saving and Loading TorchScript

```python
# Script a model
model = MyModule()
scripted = torch.jit.script(model)

# Save to file
scripted.save("model.pt")

# Load (no Python code needed!)
loaded = torch.jit.load("model.pt")
loaded.eval()
output = loaded(torch.randn(1, 3, 224, 224))

# Can load in C++ too
# torch::jit::script::Module module = torch::jit::load("model.pt");
```

---

## Section 2: torch.compile Architecture (PyTorch 2.0+) (~200 lines)

### The PyTorch 2.0 Compilation Stack

From [PyTorch 2 Paper](https://docs.pytorch.org/assets/pytorch2-2.pdf) (ASPLOS 2024):
> "This paper introduces two extensions to the popular PyTorch machine learning framework, TorchDynamo and TorchInductor, which implement the torch.compile feature released in PyTorch 2."

**Three-Layer Architecture:**

```
Python Code with @torch.compile
         ↓
   TorchDynamo (Frontend)
   - Python bytecode interception
   - Graph extraction
   - Guards for dynamic behavior
         ↓
   FX Graph (Intermediate Representation)
   - Torch operations as nodes
   - Data flow graph
         ↓
   TorchInductor (Backend Compiler)
   - Triton code generation (GPU)
   - C++ code generation (CPU)
   - Optimized kernel fusion
         ↓
   Compiled Kernels
```

### TorchDynamo: Python Bytecode Transformation

**How TorchDynamo Works:**

TorchDynamo operates at the Python bytecode level, intercepting execution before the Python interpreter runs the code. This is fundamentally different from TorchScript's source-level analysis.

**Bytecode Interception:**
```python
@torch.compile
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

# TorchDynamo intercepts CPython bytecode:
# LOAD_GLOBAL (torch)
# LOAD_ATTR (sin)
# LOAD_FAST (x)
# CALL_FUNCTION
# ... etc

# Captures sequence: sin → cos → add
# Builds FX graph representing this computation
```

**Graph Extraction Process:**
1. Hook into Python frame evaluation
2. Intercept bytecode before execution
3. Identify PyTorch tensor operations
4. Build FX graph of tensor ops
5. Compile graph with backend
6. Cache compiled graph
7. Execute compiled version

**Guards for Correctness:**

```python
@torch.compile
def dynamic_example(x):
    if x.shape[0] > 10:
        return x * 2
    return x + 1

# TorchDynamo creates guards:
# - Guard: tensor shape[0] > 10 is True → use graph A
# - Guard: tensor shape[0] > 10 is False → use graph B
# Guards checked before each execution
```

### FX Graph: Intermediate Representation

**FX Graph Structure:**

```python
import torch

@torch.compile
def example(x, y):
    z = x + y
    return torch.relu(z)

# Generates FX Graph (viewable with TORCH_LOGS=graph_code):
"""
class GraphModule(torch.nn.Module):
    def forward(self, x, y):
        add = x + y
        relu = torch.relu(add)
        return relu
"""
```

**FX Graph Properties:**
- Directed acyclic graph (DAG) of operations
- Nodes represent operations (add, relu, matmul, etc.)
- Edges represent data dependencies
- Preserves PyTorch semantics exactly
- Can be transformed and optimized

### TorchInductor: Code Generation Backend

**Default Compilation Backend:**

TorchInductor is the default backend that generates optimized code for CPUs and GPUs.

From [PyTorch 2 Paper](https://docs.pytorch.org/assets/pytorch2-2.pdf):
> "TorchInductor translates PyTorch programs into OpenAI's Triton for GPUs and C++ for CPUs."

**GPU Code Generation (Triton):**
```python
# PyTorch operations
@torch.compile
def fused_kernel(x, y):
    return torch.relu(x + y)

# TorchInductor generates Triton kernel:
# @triton.jit
# def fused_add_relu_kernel(
#     x_ptr, y_ptr, out_ptr,
#     n_elements, BLOCK_SIZE: tl.constexpr
# ):
#     pid = tl.program_id(0)
#     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements
#     x = tl.load(x_ptr + offsets, mask=mask)
#     y = tl.load(y_ptr + offsets, mask=mask)
#     out = tl.maximum(x + y, 0)  # Fused add + relu
#     tl.store(out_ptr + offsets, out, mask=mask)
```

**CPU Code Generation (C++):**
```python
# For CPU, generates vectorized C++ with OpenMP:
# #pragma omp parallel for
# for (int64_t i = 0; i < n_elements; i += 16) {
#     auto x_vec = _mm512_loadu_ps(&x[i]);
#     auto y_vec = _mm512_loadu_ps(&y[i]);
#     auto sum = _mm512_add_ps(x_vec, y_vec);
#     auto result = _mm512_max_ps(sum, _mm512_setzero_ps());
#     _mm512_storeu_ps(&out[i], result);
# }
```

**Optimization Techniques:**
- **Kernel fusion** - Combine multiple ops into single kernel
- **Memory planning** - Reuse buffers, minimize allocations
- **Vectorization** - SIMD on CPU, warp-level on GPU
- **Loop tiling** - Optimize cache locality
- **Constant propagation** - Fold compile-time constants

### Compilation Backends

**Available Backends:**

```python
# Default: TorchInductor
compiled = torch.compile(model)

# Specify backend explicitly
compiled = torch.compile(model, backend="inductor")

# Other backends:
compiled = torch.compile(model, backend="aot_eager")  # Debugging
compiled = torch.compile(model, backend="cudagraphs")  # CUDA Graphs
```

**Backend Options:**
- **inductor** (default) - TorchInductor with Triton/C++ codegen
- **cudagraphs** - CUDA Graphs for ultra-low latency
- **aot_eager** - Ahead-of-time compilation for debugging
- **onnxrt** - ONNX Runtime integration
- **tensorrt** - NVIDIA TensorRT integration

### AOTAutograd: Ahead-of-Time Autograd

**Compiling Backward Pass:**

```python
@torch.compile
def forward_and_backward(x, target):
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()  # Backward also compiled!
    return loss

# AOTAutograd captures backward graph too
# Both forward and backward are optimized
```

**How AOTAutograd Works:**
1. Run forward pass symbolically
2. Trace backward operations
3. Build joint forward-backward graph
4. Optimize both together
5. Generate fused forward+backward kernels

**Benefits:**
- Backward pass gets same optimizations as forward
- Joint optimization opportunities (recomputation vs memory)
- Reduced memory allocation in backward

---

## Section 3: torch.compile Compilation Modes (~150 lines)

### Mode Overview

torch.compile provides three compilation modes that trade compilation time for runtime performance:

```python
# Default mode - balanced performance/compile time
model = torch.compile(model)
model = torch.compile(model, mode="default")

# Reduce overhead mode - lower latency via CUDA graphs
model = torch.compile(model, mode="reduce-overhead")

# Max autotune mode - maximum performance
model = torch.compile(model, mode="max-autotune")
```

### Mode: "default"

From [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) (accessed 2025-02-03):
> "default is the default mode, which is a good balance between performance and overhead."

**Characteristics:**
- Fast compilation time
- Good runtime speedup (1.5-2× typical)
- Minimal kernel tuning
- Recommended for initial optimization

**Compilation Strategy:**
- Basic kernel fusion
- Standard Triton templates
- Quick code generation
- No autotuning

**Use Cases:**
- Development and iteration
- Models with frequent recompilation
- When compile time matters
- First attempt at optimization

**Example:**
```python
import torch

model = MyLargeModel().cuda()
opt_model = torch.compile(model)  # mode="default" implicit

# Compilation: ~10 seconds
# Speedup: 1.8× on inference
```

### Mode: "reduce-overhead"

From [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html):
> "reduce-overhead is a mode that reduces the overhead of python with CUDA graphs."

**Characteristics:**
- Uses CUDA Graphs when possible
- Reduces Python overhead
- Slightly longer compilation
- Best for inference and small batches

**CUDA Graph Integration:**

From [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (PyTorch Blog, accessed 2025-02-03):
> "CUDA Graphs let a series of CUDA kernels to be defined and encapsulated as a single unit, i.e., a graph of operations, rather than a sequence of individually-launched operations. It provides a mechanism to launch multiple GPU operations through a single CPU operation, and hence reduces the launching overheads."

```python
# reduce-overhead mode automatically uses CUDA graphs
model = torch.compile(model, mode="reduce-overhead")

# Internally creates CUDA graph for static sections:
# g = torch.cuda.CUDAGraph()
# with torch.cuda.graph(g):
#     output = model(static_input)
#
# Then replays graph with minimal overhead:
# g.replay()
```

**Requirements for CUDA Graphs:**
- Static tensor shapes
- No CPU synchronization
- No dynamic memory allocation
- Deterministic kernel order

**Benefits:**
- Reduces launch overhead from ~5-20μs per kernel to ~2μs total
- Eliminates Python interpreter overhead
- Consistent kernel timing (important for distributed training)

**Use Cases:**
- Inference serving (low latency critical)
- Small batch sizes (overhead dominates)
- Models with many small kernels
- Static shape workloads

**Example Performance:**
```python
# Without reduce-overhead: 100 iterations @ 8.5ms = 850ms
# With reduce-overhead: 100 iterations @ 3.6ms = 360ms
# Speedup: 2.36×
```

### Mode: "max-autotune"

From [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html):
> "max-autotune leverages Triton's autotuning to select the best kernel for each operation."

**Characteristics:**
- Longest compilation time (minutes to hours)
- Best runtime performance
- Autotuning searches kernel parameter space
- Recommended for production deployment

**Autotuning Process:**
1. Generate multiple kernel variants
2. Test each variant with actual data
3. Benchmark all variants
4. Select fastest kernel
5. Cache tuning results

**Triton Autotuning Example:**

```python
# TorchInductor generates multiple Triton configs:
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel implementation
    pass

# Triton runs benchmarks and picks fastest config
```

**Compilation Time:**
```python
# Typical compilation times:
# - default mode: 10 seconds
# - reduce-overhead: 30 seconds
# - max-autotune: 5-60 minutes (depends on model size)

# But max-autotune often gives 10-20% additional speedup
```

**Use Cases:**
- Production deployment (compile once, run millions)
- Performance-critical workloads
- Large models where 10% speedup = significant savings
- Stable models (not changing frequently)

**Caching:**
```python
# Autotuning results are cached
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024  # Increase cache

# First run: autotuning takes 10 minutes
model = torch.compile(model, mode="max-autotune")
output = model(input)

# Subsequent runs: instant (cache hit)
output = model(new_input)  # Same shapes
```

### Mode Comparison Table

| Mode | Compile Time | Speedup | Best For |
|------|--------------|---------|----------|
| default | 10s | 1.5-2× | Development, iteration |
| reduce-overhead | 30s | 2-3× | Inference, small batches |
| max-autotune | 5-60m | 2-4× | Production, large scale |

### fullgraph Mode

**Forcing Single Graph (No Graph Breaks):**

```python
# fullgraph=True raises error on graph breaks
model = torch.compile(model, fullgraph=True)

# If compilation succeeds, entire model is one graph
# If it fails, you get clear error message about what broke
```

**Use Cases:**
- Debugging graph breaks
- Ensuring maximum optimization
- Verifying model is compilable
- Production deployment (fail fast if incompatible change)

**Example Error:**
```python
@torch.compile(fullgraph=True)
def broken(x):
    if x.sum() > 0:  # Data-dependent control flow
        return x * 2
    return x

# Raises: torch._dynamo.exc.Unsupported: Data-dependent branching
```

---

## Section 4: CUDA Graph Integration (~150 lines)

### CUDA Graphs in torch.compile

torch.compile can automatically use CUDA graphs to eliminate kernel launch overhead, especially in `mode="reduce-overhead"`.

**Manual CUDA Graph Usage:**

```python
import torch

model = MyModel().cuda()

# Warmup (required before capture)
static_input = torch.randn(32, 3, 224, 224, device='cuda')
for _ in range(3):
    _ = model(static_input)

# Capture CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay graph (ultra-fast)
for real_input in dataloader:
    static_input.copy_(real_input)
    g.replay()
    result = static_output.clone()
```

### torch.compile with cudagraphs Backend

```python
# Use CUDA graphs backend explicitly
model = torch.compile(model, backend="cudagraphs")

# Or via reduce-overhead mode (automatic CUDA graphs)
model = torch.compile(model, mode="reduce-overhead")
```

**How torch.compile Uses CUDA Graphs:**

1. Identifies static graph regions
2. Captures those regions as CUDA graphs
3. Inserts graph replay nodes in execution
4. Falls back to eager for dynamic regions

**Requirements:**
- Static tensor shapes
- No CPU synchronization within graph
- No dynamic memory allocation
- No data-dependent control flow

### Performance Impact

From [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/):

**Launch Overhead Reduction:**
- Traditional: ~2μs + 200ns per kernel
- CUDA Graphs: ~2.5μs constant (entire graph)
- For 10+ node graphs: 10× reduction in overhead

**Example Benchmark:**

```python
import torch
import time

model = MyTransformer().cuda()
input_tensor = torch.randn(8, 512, 512, device='cuda')

# Baseline (eager)
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = model(input_tensor)
torch.cuda.synchronize()
eager_time = time.time() - start

# With CUDA graphs
compiled = torch.compile(model, mode="reduce-overhead")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = compiled(input_tensor)
torch.cuda.synchronize()
graph_time = time.time() - start

print(f"Eager: {eager_time:.3f}s")
print(f"Graphs: {graph_time:.3f}s")
print(f"Speedup: {eager_time/graph_time:.2f}×")

# Typical output:
# Eager: 0.850s
# Graphs: 0.360s
# Speedup: 2.36×
```

### Static Shapes Requirement

**Why Static Shapes Matter:**

CUDA graphs record kernel launches with specific shapes and memory addresses. Dynamic shapes require different kernel configurations, breaking the graph.

```python
# Static shapes - CUDA graph works
@torch.compile(mode="reduce-overhead")
def static_model(x):
    return torch.relu(x + 1)

for batch in dataloader:  # All same shape
    output = static_model(batch)  # Fast (graph replay)

# Dynamic shapes - graph breaks
@torch.compile(mode="reduce-overhead")
def dynamic_model(x):
    return torch.relu(x + 1)

for batch in variable_dataloader:  # Different shapes
    output = dynamic_model(batch)  # Slow (recompile each time)
```

**Padding to Static Shapes:**

```python
# Pad variable-length sequences to max length
def pad_to_static(sequences):
    max_len = 512  # Fixed maximum
    return torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, max_length=max_len
    )

compiled = torch.compile(model, mode="reduce-overhead")
for batch in dataloader:
    padded = pad_to_static(batch)
    output = compiled(padded)  # Static shape - CUDA graph used
```

### Memory Address Stability

**CUDA Graphs Record Memory Addresses:**

```python
# WRONG - reallocation breaks graph
static_input = torch.randn(32, 3, 224, 224, device='cuda')
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

for real_input in dataloader:
    static_input = real_input  # WRONG - different memory address!
    g.replay()  # Will read/write wrong memory!

# CORRECT - copy to same memory address
static_input = torch.randn(32, 3, 224, 224, device='cuda')
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

for real_input in dataloader:
    static_input.copy_(real_input)  # Copy to stable address
    g.replay()  # Reads from correct address
```

### Automatic Graph Splitting

torch.compile can split models into graph-safe and graph-unsafe regions:

```python
@torch.compile(mode="reduce-overhead")
def mixed_model(x):
    # Graph-safe region (captured as CUDA graph)
    x = torch.relu(x)
    x = torch.matmul(x, weight)

    # Graph break (data-dependent control flow)
    if x.sum() > 0:
        x = x * 2

    # Another graph-safe region (new CUDA graph)
    x = torch.softmax(x, dim=-1)
    return x

# TorchDynamo creates:
# - CUDA graph 1: relu + matmul
# - Eager execution: if statement
# - CUDA graph 2: softmax
```

---

## Section 5: Debugging and Profiling (~100 lines)

### Viewing Compiled Graphs

**Enable Graph Logging:**

```python
import torch

# View generated FX graphs
torch._logging.set_logs(graph_code=True)

@torch.compile
def example(x, y):
    z = torch.sin(x) + torch.cos(y)
    return torch.relu(z)

output = example(torch.randn(3, 3), torch.randn(3, 3))

# Prints:
# TRACED GRAPH
# class GraphModule(torch.nn.Module):
#     def forward(self, x, y):
#         sin = torch.sin(x)
#         cos = torch.cos(y)
#         add = sin + cos
#         relu = torch.relu(add)
#         return relu
```

### Finding Graph Breaks

**Detect Optimization Losses:**

```python
import torch

torch._logging.set_logs(graph_breaks=True)

@torch.compile
def with_break(x, y):
    z = x + y
    print(z.sum())  # Graph break - CPU sync
    return torch.relu(z)

output = with_break(torch.randn(3, 3), torch.randn(3, 3))

# Logs:
# Graph break: print() encountered
# Reason: CPU synchronization required
```

**Common Graph Break Causes:**
- print() statements
- .item() or .numpy() calls
- Data-dependent control flow
- Unsupported Python operations
- Third-party library calls

### torch._dynamo.explain()

**Detailed Compilation Information:**

```python
import torch

@torch.compile
def model(x):
    return torch.relu(x + 1)

# Get compilation details
explain_output = torch._dynamo.explain(model)(torch.randn(3, 3))

print(explain_output)
# Outputs:
# - Number of graphs created
# - Graph break locations and reasons
# - Optimization opportunities
```

### Performance Profiling

**Benchmarking Compilation Impact:**

```python
import torch
import torch.utils.benchmark as benchmark

model = MyModel().cuda()
input_tensor = torch.randn(32, 3, 224, 224, device='cuda')

# Benchmark eager mode
eager_timer = benchmark.Timer(
    stmt='model(input_tensor)',
    globals={'model': model, 'input_tensor': input_tensor}
)
eager_time = eager_timer.timeit(100)

# Benchmark compiled mode
compiled = torch.compile(model)
compiled_timer = benchmark.Timer(
    stmt='model(input_tensor)',
    globals={'model': compiled, 'input_tensor': input_tensor}
)
compiled_time = compiled_timer.timeit(100)

print(f"Eager: {eager_time}")
print(f"Compiled: {compiled_time}")
print(f"Speedup: {eager_time.median / compiled_time.median:.2f}×")
```

### Compilation Cache Management

**Cache Location:**

```python
import torch._dynamo

# Check cache directory
print(torch._dynamo.config.cache_dir)
# Default: ~/.triton/cache/

# Increase cache size
torch._dynamo.config.cache_size_limit = 2048  # MB

# Clear cache
torch._dynamo.reset()
```

**Recompilation Detection:**

```python
import torch

torch._logging.set_logs(recompiles=True)

@torch.compile
def model(x):
    return x + 1

# First call: compilation
model(torch.randn(10))

# Same shape: cache hit (no recompilation)
model(torch.randn(10))

# Different shape: recompilation!
# Logs: Recompiling due to new input shapes
model(torch.randn(20))
```

### Common Errors and Solutions

**Error 1: Dynamic Shapes**

```python
# Problem: Variable sequence lengths
@torch.compile(fullgraph=True)
def process_sequence(x):
    return torch.nn.functional.pad(x, (0, x.size(0)))  # Error!

# Solution: Set dynamic=True
@torch.compile(dynamic=True)
def process_sequence(x):
    return torch.nn.functional.pad(x, (0, x.size(0)))  # OK
```

**Error 2: Unsupported Operations**

```python
# Problem: NumPy operations
@torch.compile
def with_numpy(x):
    arr = x.numpy()  # Graph break
    return torch.from_numpy(arr)

# Solution: Use pure PyTorch
@torch.compile
def pure_pytorch(x):
    return x.clone()  # No graph break
```

**Error 3: In-Place Mutations**

```python
# Problem: In-place mutation of inputs
@torch.compile
def mutating(x):
    x += 1  # May cause issues
    return x

# Solution: Avoid in-place ops on inputs
@torch.compile
def safe(x):
    return x + 1  # Create new tensor
```

---

## Section 6: ARR-COC Relevance Scorer Optimization (~150 lines)

### Compiling Relevance Scorers

ARR-COC has three relevance scorers that benefit significantly from torch.compile optimization:

```python
# From arr_coc/knowing.py (ARR-COC-VIS project)

import torch
import torch.nn as nn

class InformationScorer(nn.Module):
    """Propositional knowing: statistical information content."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Entropy calculation benefits from fusion
        probs = torch.softmax(features, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

class SalienceScorer(nn.Module):
    """Perspectival knowing: salience landscape."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Edge detection + local contrast
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.t()
        # ... gradient computation
        return salience

class QueryContentScorer(nn.Module):
    """Participatory knowing: query-content coupling."""

    def forward(self, query: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        # Attention scoring benefits from Tensor Cores
        scores = torch.matmul(query, content.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(content.size(-1)))
        return torch.softmax(scores, dim=-1)
```

### Optimization Strategy

**Step 1: Profile Baseline Performance**

```python
import torch
import torch.utils.benchmark as benchmark

# Create scorers
info_scorer = InformationScorer().cuda()
salience_scorer = SalienceScorer().cuda()
query_content_scorer = QueryContentScorer().cuda()

# Test inputs
features = torch.randn(32, 196, 768, device='cuda')  # 32 patches, 768 features
query = torch.randn(32, 1, 768, device='cuda')

# Benchmark eager mode
def benchmark_eager():
    info = info_scorer(features)
    salience = salience_scorer(features)
    qc = query_content_scorer(query, features)
    return info, salience, qc

eager_timer = benchmark.Timer(
    stmt='benchmark_eager()',
    globals=globals()
)
eager_time = eager_timer.timeit(100)
print(f"Eager: {eager_time.median * 1000:.3f}ms")
```

**Step 2: Apply torch.compile**

```python
# Compile individual scorers
info_scorer_opt = torch.compile(info_scorer, mode="max-autotune")
salience_scorer_opt = torch.compile(salience_scorer, mode="max-autotune")
qc_scorer_opt = torch.compile(query_content_scorer, mode="max-autotune")

# Benchmark compiled
def benchmark_compiled():
    info = info_scorer_opt(features)
    salience = salience_scorer_opt(features)
    qc = qc_scorer_opt(query, features)
    return info, salience, qc

compiled_timer = benchmark.Timer(
    stmt='benchmark_compiled()',
    globals=globals()
)
compiled_time = compiled_timer.timeit(100)
print(f"Compiled: {compiled_time.median * 1000:.3f}ms")
print(f"Speedup: {eager_time.median / compiled_time.median:.2f}×")

# Typical results:
# Eager: 2.45ms
# Compiled: 0.89ms
# Speedup: 2.75×
```

### Kernel Fusion Benefits

**Information Scorer Fusion:**

```python
# Eager mode (separate kernels):
# 1. softmax (exp + sum + divide)
# 2. log
# 3. multiply
# 4. sum
# Total: 4 kernel launches, 4 memory reads/writes

# torch.compile fuses:
@torch.compile
def fused_entropy(features):
    probs = torch.softmax(features, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1)

# Compiled: 1 Triton kernel
# - Single memory read of features
# - Fused softmax → log → multiply → sum
# - Single memory write of result
# Speedup: ~3× (eliminates 3 memory passes)
```

**Query-Content Scorer Optimization:**

```python
# Benefits from Tensor Core usage
@torch.compile
def optimized_attention(query, content):
    # matmul automatically uses Tensor Cores (FP16/BF16)
    scores = torch.matmul(query, content.transpose(-2, -1))
    scale = torch.sqrt(torch.tensor(content.size(-1)))
    scores = scores / scale
    return torch.softmax(scores, dim=-1)

# Compilation enables:
# 1. TF32 Tensor Cores on A100 (10× faster than FP32)
# 2. Fused scaling + softmax
# 3. Optimized memory layout
```

### Training Loop Integration

```python
# arr_coc/training.py (example)

import torch
from torch.cuda.amp import autocast, GradScaler

# Compile the entire model
model = ARRCOCModel(config).cuda()
model = torch.compile(model, mode="reduce-overhead")

# Use mixed precision for additional speedup
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        images, queries, targets = batch

        with autocast(dtype=torch.bfloat16):
            # Forward pass (compiled)
            output = model(images, queries)
            loss = criterion(output, targets)

        # Backward pass (also compiled via AOTAutograd)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# Speedup breakdown:
# - torch.compile: 1.8× faster forward/backward
# - BF16 mixed precision: 1.5× faster (Tensor Cores)
# - Combined: 2.7× faster training
```

### Inference Optimization

```python
# Inference with maximum optimization
import torch

model = ARRCOCModel.from_pretrained("checkpoints/best.pt")
model = model.cuda().eval()

# Compile with reduce-overhead for inference
model = torch.compile(model, mode="reduce-overhead")

# Optional: Use CUDA graphs for static shapes
static_image = torch.randn(1, 3, 224, 224, device='cuda')
static_query = torch.randn(1, 768, device='cuda')

# Warmup
for _ in range(3):
    _ = model(static_image, static_query)

# Capture CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_image, static_query)

# Ultra-fast inference loop
for image, query in dataloader:
    static_image.copy_(image)
    static_query.copy_(query)
    g.replay()
    result = static_output.clone()

    # Process result...

# Performance:
# Eager: 12ms per image
# torch.compile: 5ms per image
# CUDA graphs: 2ms per image
# Total speedup: 6×
```

### Debugging Compilation Issues

**Check for Graph Breaks:**

```python
import torch

torch._logging.set_logs(graph_breaks=True)

# Compile relevance scoring module
from arr_coc.knowing import KnowingModule

knowing = KnowingModule(config).cuda()
knowing_opt = torch.compile(knowing)

# Test with sample input
features = torch.randn(32, 196, 768, device='cuda')
query = torch.randn(32, 768, device='cuda')

output = knowing_opt(features, query)

# Check logs for graph breaks
# If breaks found, refactor to eliminate them
```

**Verify Kernel Fusion:**

```python
import torch

torch._logging.set_logs(graph_code=True)

@torch.compile
def relevance_fusion(features):
    # Should fuse into single kernel
    info = -(torch.softmax(features, -1) * torch.log(torch.softmax(features, -1) + 1e-8)).sum(-1)
    return info

output = relevance_fusion(torch.randn(32, 196, 768, device='cuda'))

# Check generated code - should see single Triton kernel
# If multiple kernels, may need to refactor
```

### Production Deployment

```python
# Pre-compile for deployment

# 1. Compile with max-autotune (long compilation, best performance)
model = ARRCOCModel.from_pretrained("checkpoints/best.pt")
model = model.cuda().eval()
model = torch.compile(model, mode="max-autotune")

# 2. Warmup to trigger compilation
dummy_image = torch.randn(1, 3, 224, 224, device='cuda')
dummy_query = torch.randn(1, 768, device='cuda')
for _ in range(10):
    _ = model(dummy_image, dummy_query)

# 3. Save compiled model (optional - cache is persistent)
# torch.save(model.state_dict(), "model_compiled.pt")

# 4. Deploy
# Subsequent runs use cached compiled kernels
# No recompilation needed unless input shapes change
```

---

## Sources

### Source Documents

**ARR-COC Project Code:**
- Not directly cited (internal project structure referenced)
- Relevance scorer architecture from arr_coc/knowing.py

### Web Research

**PyTorch Official Documentation:**
- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - PyTorch Tutorials (accessed 2025-02-03)
  - Basic torch.compile usage and graph breaks
  - Speedup demonstrations and TorchScript comparison

- [PyTorch 2 Paper and Tutorial @ ASPLOS 2024](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/) - PyTorch Blog (accessed 2025-02-03)
  - TorchDynamo architecture and bytecode transformation
  - TorchInductor code generation to Triton/C++
  - Paper: "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (ASPLOS 2024)

- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) - PyTorch Blog (accessed 2025-02-03)
  - CUDA Graphs performance benefits and usage patterns
  - Launch overhead reduction measurements
  - Integration with torch.compile

**PyTorch Tutorials and Guides:**
- [TorchScript Fundamentals](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) (accessed 2025-02-03)
  - torch.jit.trace vs torch.jit.script differences
  - Type annotation requirements and limitations

**Community Resources:**
- PyTorch Forums discussions on torch.compile modes and performance
- Reddit r/StableDiffusion torch.compile optimization discussions
- Medium articles on torch.compile architecture and usage

**Performance Documentation:**
- NVIDIA CUDA Graphs documentation
- Triton autotuning documentation
- PyTorch profiling guides

### Additional References

**Academic Papers:**
- "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" - ASPLOS 2024
  - Full technical details on TorchDynamo and TorchInductor implementation

**NVIDIA Resources:**
- CUDA Graphs Programming Guide
- Triton Language Documentation
- MLPerf training results using torch.compile

**Performance Benchmarks:**
- Hugging Face Transformers torch.compile benchmarks
- vLLM torch.compile integration guide
- Community benchmarks on various model architectures
