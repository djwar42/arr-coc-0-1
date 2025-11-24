# torch.compile & AOT Inductor for Production Inference

## Overview

`torch.compile` is PyTorch 2.0+'s modern compilation system that JIT-compiles models into optimized kernels with minimal code changes. For production inference, **AOT Inductor** extends torch.compile to ahead-of-time (AOT) compilation, generating standalone shared libraries for Python-free deployment in C++ or mobile environments.

**Key Innovation**: torch.compile uses TorchDynamo to capture Python bytecode, TorchInductor to generate optimized CUDA/CPU kernels, and AOTAutograd for backwards pass compilation - achieving 2-5× inference speedup vs eager mode.

From [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (PyTorch Tutorials, accessed 2025-11-13):
> "torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, while requiring minimal code changes."

From [AOTInductor Documentation](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html) (PyTorch Docs, accessed 2025-11-13):
> "AOTInductor is a specialized version of TorchInductor, designed to process exported PyTorch models, optimize them, and produce shared libraries suitable for deployment in non-Python environments."

**Performance vs TorchScript**:
- torch.compile: 2-3× inference speedup (typical), up to 5× for CPU-bound workloads
- TorchScript: More restrictive (requires type annotations, limited Python support)
- torch.compile handles data-dependent control flow with graph breaks (vs TorchScript silent incorrectness)

**Related Knowledge**:
- See [../cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md) for complete torch.compile architecture (TorchDynamo, TorchInductor, compilation modes)
- See [../llm-gpu-integration/00-flashattention-internals.md](../llm-gpu-integration/00-flashattention-internals.md) for kernel fusion examples
- See [../cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) for precision integration with compiled models

---

## Section 1: torch.compile Fundamentals for Inference (~80 lines)

### Basic Usage

torch.compile is applied as a decorator or function wrapper:

```python
import torch

# Method 1: Decorator
@torch.compile
def inference_fn(x):
    return torch.nn.functional.relu(x + 1) * 2

# Method 2: Wrapper
model = MyModel()
compiled_model = torch.compile(model)
output = compiled_model(input_tensor)

# Method 3: Module.compile()
model = MyModel()
model.compile()
output = model(input_tensor)
```

From [PyTorch torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (accessed 2025-11-13):
> "torch.compile is applied recursively, so nested function calls within the top-level compiled function will also be compiled."

### Compilation Modes

torch.compile supports multiple optimization levels:

```python
# Default mode: balanced speed/memory
model = torch.compile(model, mode="default")

# Reduce overhead: optimize for small batch sizes
model = torch.compile(model, mode="reduce-overhead")

# Max autotune: extensive search for best kernels (slow compile)
model = torch.compile(model, mode="max-autotune")

# Max autotune with CUDA graphs (2.36× additional speedup)
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

From [../cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md):
> "reduce-overhead mode specifically optimizes for reducing Python overhead and CPU-GPU synchronization, critical for small batch inference scenarios where kernel launch overhead dominates."

**Mode Selection Guide**:
- **default**: Production inference with moderate batch sizes (8-32)
- **reduce-overhead**: Low-latency inference (batch size 1-4), real-time applications
- **max-autotune**: Offline optimization, willing to wait 10-30 minutes for best kernels
- **max-autotune-no-cudagraphs**: When CUDA graphs cause compatibility issues

### First Compilation Overhead

The first run with torch.compile takes significantly longer due to compilation:

```python
import torch
import time

model = torch.compile(MyModel())
input = torch.randn(1, 3, 224, 224).cuda()

# First run: 5-30 seconds (compilation + execution)
start = time.time()
output = model(input)
print(f"First run: {time.time() - start:.2f}s")  # e.g., 12.5s

# Subsequent runs: actual optimized speed
start = time.time()
output = model(input)
print(f"Second run: {time.time() - start:.2f}s")  # e.g., 0.015s
```

From [PyTorch torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (accessed 2025-11-13):
> "Notice that torch.compile appears to take a lot longer to complete compared to eager. This is because torch.compile takes extra time to compile the model on the first few executions."

**Compilation caching**: PyTorch caches compiled kernels (in `~/.triton/cache/`), reusing them across runs for the same model + input shapes.

---

## Section 2: TorchDynamo Capture & Graph Breaks (~100 lines)

### TorchDynamo: Python Bytecode Tracing

TorchDynamo captures PyTorch operations by dynamically modifying Python bytecode:

```python
# This code is fully captured without graph breaks
@torch.compile
def simple_model(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b  # All ops are PyTorch tensors
```

From [PyTorch 2 Paper](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/) (PyTorch Blog, accessed 2025-11-13):
> "TorchDynamo is a Python-level just-in-time (JIT) compiler that enables graph compilation in PyTorch programs without sacrificing the flexibility of Python. It achieves this by dynamically modifying Python bytecode before execution."

**TorchDynamo vs TorchScript**:

| Feature | TorchDynamo (torch.compile) | TorchScript |
|---------|----------------------------|-------------|
| Python support | Full Python 3.8+ | Limited subset |
| Control flow | Handles via graph breaks | Trace: silent fail, Script: requires type annotations |
| User code changes | Minimal (just decorator) | Moderate to extensive |
| Error handling | Graph breaks instead of errors | Errors or silent incorrectness |

### Graph Breaks: The Key Difference

When TorchDynamo encounters unsupported code (data-dependent control flow, external library calls), it creates a **graph break**:

```python
@torch.compile
def model_with_graph_break(x, y):
    # Graph 1: captured
    a = torch.sin(x)

    # Graph break: data-dependent condition
    if y.sum() < 0:  # Python boolean from tensor
        y = y * -1

    # Graph 2: captured
    return a + y
```

**Execution flow**:
1. Run compiled graph 1 → get `a`
2. Let Python evaluate `y.sum() < 0` (graph break)
3. Run compiled graph 2 or 3 (depending on branch)

From [PyTorch torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (accessed 2025-11-13):
> "Code that is difficult to trace will result a graph break, which are lost optimization opportunities, rather than errors or silent incorrectness."

**Detecting graph breaks**:

```python
torch._logging.set_logs(graph_breaks=True)

@torch.compile
def model(x, y):
    a = x + 1
    if a.sum() > 0:  # Graph break here
        return a * 2
    return a * -1

model(torch.randn(10), torch.randn(10))
# Output: "Graph break: data-dependent branching"
```

### Working Around Graph Breaks

**Method 1: Use torch.cond** (for data-dependent control flow):

```python
from torch import cond

@torch.compile(fullgraph=True)  # Force single graph
def model_fixed(x, y):
    a = x + 1

    def true_fn(y): return y * 2
    def false_fn(y): return y * -1

    # No graph break with torch.cond
    result = cond(a.sum() > 0, true_fn, false_fn, (y,))
    return a + result
```

**Method 2: Refactor to avoid data-dependent code**:

```python
# Bad: graph break on .item()
if tensor.item() > 0:
    return x * 2

# Good: no graph break
return torch.where(tensor > 0, x * 2, x)
```

From [PyTorch torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (accessed 2025-11-13):
> "We can work around this graph break by replacing the if statement with a torch.cond."

---

## Section 3: TorchInductor Optimization (~100 lines)

### TorchInductor: Kernel Generation Backend

TorchInductor takes the captured FX graph and generates optimized CUDA/CPU kernels using Triton (for CUDA) or C++ (for CPU).

**Optimization strategies**:
1. **Kernel fusion**: Combine multiple operations into single kernel
2. **Memory layout optimization**: Eliminate unnecessary transposes
3. **Auto-tuning**: Search best tile sizes, thread blocks
4. **CUDA graph integration**: Reduce kernel launch overhead

From [../cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md):
> "TorchInductor generates Triton kernels for CUDA and C++/OpenMP code for CPU. It performs automatic operator fusion, memory planning, and leverages hardware-specific optimizations."

**Example kernel fusion**:

```python
# Without torch.compile: 3 separate kernels
x = input + 1          # Kernel 1: add
y = torch.relu(x)      # Kernel 2: relu
z = y * 2              # Kernel 3: mul

# With torch.compile: 1 fused kernel
@torch.compile
def fused(input):
    return torch.relu(input + 1) * 2  # Single kernel: fused_add_relu_mul
```

**Memory access reduction**: Fusion eliminates intermediate memory writes/reads:
- Unfused: 3× DRAM writes + 3× DRAM reads (6 total)
- Fused: 1× DRAM read + 1× DRAM write (2 total) = **3× memory bandwidth reduction**

### Triton Kernel Generation

TorchInductor generates Triton kernels (GPU-optimized Python-like language):

```python
# Conceptual Triton output for fused operation
@triton.jit
def fused_add_relu_mul_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Fused: load once, compute all ops, store once
    x = tl.load(input_ptr + offsets, mask=mask)
    x = x + 1.0        # add
    x = tl.maximum(x, 0.0)  # relu
    x = x * 2.0        # mul
    tl.store(output_ptr + offsets, x, mask=mask)
```

From [PyTorch torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (accessed 2025-11-13):
> "Speedup mainly comes from reducing Python overhead and GPU read/writes."

### Auto-Tuning with max-autotune Mode

max-autotune mode exhaustively searches optimal kernel configurations:

```python
model = torch.compile(model, mode="max-autotune")

# TorchInductor will try different configurations:
# - Block sizes: 64, 128, 256, 512
# - Thread layouts: (32, 1), (16, 2), (8, 4)
# - Memory layouts: row-major, column-major
# - Warp specialization: on/off
```

**Trade-offs**:
- **Compilation time**: 10-30 minutes (vs <1 minute for default mode)
- **Runtime speedup**: +10-30% over default mode
- **Use case**: Offline optimization for production deployment

From [Ways to use torch.compile](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/) (ezyang's blog, accessed 2025-11-13):
> "max-autotune performs extensive kernel tuning, significantly increasing compilation time but producing the fastest kernels for repeated execution."

---

## Section 4: AOT Compilation Workflow (~80 lines)

### AOT Inductor: From Python to C++

AOT Inductor compiles models ahead-of-time into standalone shared libraries for deployment without Python:

```python
import torch
from torch._export import capture_pre_autograd_graph

# Step 1: Export model to static graph
model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)
exported_model = torch.export.export(model, example_inputs)

# Step 2: AOT compile to shared library
from torch._inductor import aot_compile

# Generates .so file (Linux) or .dylib (Mac) or .dll (Windows)
so_path = aot_compile(exported_model.module(), example_inputs)
print(f"Compiled library: {so_path}")
# Output: /tmp/compiled_model_abc123.so
```

From [AOTInductor Documentation](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html) (accessed 2025-11-13):
> "AOTInductor is a specialized version of TorchInductor, designed to process exported PyTorch models, optimize them, and produce shared libraries suitable for deployment in non-Python environments."

### Loading Compiled Model in C++

**C++ inference code**:

```cpp
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// Load AOT-compiled model
AOTIModelContainerHandle handle;
aoti_load_model(&handle, "/tmp/compiled_model_abc123.so");

// Create input tensor
std::vector<int64_t> shape = {1, 3, 224, 224};
torch::Tensor input = torch::randn(shape);

// Run inference
std::vector<torch::Tensor> outputs;
aoti_run(handle, {input.data_ptr()}, outputs);

// Cleanup
aoti_delete_model(handle);
```

**Key benefits**:
- No Python runtime dependency (50MB+ saved)
- No GIL (Global Interpreter Lock) = true multi-threading
- Lower memory footprint (no Python objects overhead)
- Predictable latency (no Python GC pauses)

### AOT vs JIT Compilation

| Aspect | AOT Inductor | torch.compile (JIT) |
|--------|--------------|---------------------|
| Compilation | Ahead-of-time (deploy .so) | Just-in-time (first run) |
| Python dependency | No Python needed | Requires Python |
| First inference latency | Fast (pre-compiled) | Slow (compilation + inference) |
| Deployment size | Smaller (no Python) | Larger (Python + packages) |
| Flexibility | Fixed input shapes | Dynamic shapes supported |
| Use case | Production servers, edge | Development, dynamic models |

From [torch.export AOTInductor Tutorial](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html) (PyTorch Tutorials, accessed 2025-11-13):
> "AOTInductor compiles the exported model ahead of time, creating standalone artifacts that can be loaded and executed without requiring a full Python runtime."

### torch.export for Static Graphs

AOT Inductor requires `torch.export` to create static graphs (no graph breaks allowed):

```python
import torch
from torch.export import export

model = MyModel()
example_inputs = (torch.randn(1, 3, 224, 224),)

# Export model (no graph breaks allowed)
try:
    exported = export(model, example_inputs)
    print("Export successful!")
except Exception as e:
    print(f"Export failed: {e}")
    # Must fix dynamic control flow before AOT compilation
```

From [PyTorch Deployment Recommendations](https://dev-discuss.pytorch.org/t/pytorch-2-x-inference-recommendations/2506) (PyTorch Dev Discussion, accessed 2025-11-13):
> "For C++ deployment, use torch.export + AOTInductor. This provides the best performance and smallest deployment footprint."

---

## Section 5: VLM Optimization Cases (~40 lines)

### torch.compile for VLM Inference

Vision-language models benefit significantly from torch.compile:

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load VLM
model = AutoModel.from_pretrained("model_name")
model = model.eval().cuda()

# Compile vision encoder + language decoder separately
model.vision_encoder = torch.compile(model.vision_encoder, mode="reduce-overhead")
model.language_decoder = torch.compile(model.language_decoder, mode="default")

# Inference
images = torch.randn(1, 3, 224, 224).cuda()
text_tokens = torch.randint(0, 50000, (1, 77)).cuda()

with torch.no_grad():
    output = model(images, text_tokens)
```

**VLM-specific optimizations**:
1. **Vision encoder**: Use reduce-overhead mode (small batch, high kernel count)
2. **Language decoder**: Use default mode (autoregressive, sequential)
3. **Compile separately**: Different optimization strategies for vision vs language

From [../llm-gpu-integration/03-inference-kv-cache-optimization.md](../llm-gpu-integration/03-inference-kv-cache-optimization.md):
> "Multi-stream execution allows vision encoder and language decoder to overlap computation, reducing end-to-end latency."

### ARR-COC Relevance Scoring

torch.compile can optimize ARR-COC's three relevance scorers:

```python
# Propositional scorer (information content)
@torch.compile(mode="max-autotune")
def propositional_scorer(patches):
    # Entropy calculation across patches
    return compute_information_density(patches)

# Perspectival scorer (salience landscape)
@torch.compile(mode="default")
def perspectival_scorer(patches, query):
    # Query-aware attention scores
    return compute_salience_map(patches, query)

# Participatory scorer (query-content coupling)
@torch.compile(mode="reduce-overhead")
def participatory_scorer(patches, query):
    # Cross-attention between query and patches
    return compute_coupling_scores(patches, query)
```

**Expected speedups** (based on operation types):
- Propositional (entropy): 2-3× (many small kernels → fusion)
- Perspectival (attention): 1.5-2× (FlashAttention already optimized)
- Participatory (cross-attention): 2-2.5× (fusion + memory optimization)

---

## Sources

**PyTorch Official Documentation:**
- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - PyTorch Tutorials (accessed 2025-11-13)
- [AOTInductor Documentation](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html) - PyTorch Stable Docs (accessed 2025-11-13)
- [torch.export AOTInductor Tutorial for Python runtime](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html) - PyTorch Tutorials (accessed 2025-11-13)
- [PyTorch 2 Paper and Tutorial](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/) - PyTorch Blog (accessed 2025-11-13)

**Technical Analysis:**
- [Ways to use torch.compile](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/) - ezyang's blog, November 2024 (accessed 2025-11-13)
- [PyTorch 2.x Inference Recommendations](https://dev-discuss.pytorch.org/t/pytorch-2-x-inference-recommendations/2506) - PyTorch Dev Discussion (accessed 2025-11-13)

**GitHub Implementation References:**
- [pytorch/pytorch](https://github.com/pytorch/pytorch) - Main PyTorch repository (accessed 2025-11-13)
- DeepSpeed torch.compile integration discussion: [Issue #3375](https://github.com/deepspeedai/DeepSpeed/issues/3375)
- vLLM torch.compile support: [Issue #130174](https://github.com/pytorch/pytorch/issues/130174)

**Source Documents:**
- [../cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md) - Complete torch.compile architecture (TorchDynamo, TorchInductor, compilation modes)
- [../llm-gpu-integration/00-flashattention-internals.md](../llm-gpu-integration/00-flashattention-internals.md) - Kernel fusion examples
- [../cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) - Mixed precision with compilation

**Additional References:**
- [Dissecting torch.compile: Surgical Precision in PyTorch](https://themlsurgeon.substack.com/p/dissecting-torchcompile-surgical) - The ML Surgeon, Substack (accessed 2025-11-13)
- [Simple Path to PyTorch Graphs: Dynamo & AOT Autograd](https://medium.com/@sgurwinderr/pytorch-dynamo-and-aot-autograd-enhancing-performance-and-flexibility-fa18feda5f3a) - Medium (accessed 2025-11-13)
