# GPU Debugging and Profiling for ARR-COC Pipeline

## Overview

GPU debugging and profiling are essential for optimizing the ARR-COC relevance realization pipeline. This document covers industry-standard tools (RenderDoc, NSight Graphics, WebGL/WebGPU profilers) and their application to visual token compression and query-aware optimization workflows.

From [GPU Debug Scopes](https://wunkolo.github.io/post/2024/09/gpu-debug-scopes/) (accessed 2025-10-31):
- Debug scopes provide hierarchical visualization of GPU command buffers
- RAII patterns enable automatic scope lifecycle management
- Colored labels help identify performance bottlenecks and execution flow

---

## RenderDoc: Frame-Level Graphics Debugging

RenderDoc is an open-source graphics debugger providing frame capture and analysis for graphics applications.

### Core Capabilities

From [GPU Debug Scopes - Wunk](https://wunkolo.github.io/post/2024/09/gpu-debug-scopes/):

**Command Buffer Visualization:**
- Frame captures show complete GPU command execution
- Without debug information: flat list of API calls requiring manual filtering
- With debug scopes: hierarchical structure showing logical operation groups
- Color-coded labels enable visual identification of execution phases

**Object Naming and Metadata:**
- Attach readable names to GPU objects (buffers, textures, pipelines)
- Organize command-buffer operations into labeled sections
- Query performance metrics at scope boundaries

### ARR-COC Pipeline Profiling with RenderDoc

**Token Allocation Profiling:**
- Capture frames during relevance realization process
- Isolate "knowing" phase (three-way scoring): propositional, perspectival, participatory
- Track "balancing" phase (opponent processing) tension navigation
- Monitor "attending" phase (LOD allocation) token budget assignment
- Measure "realizing" phase execution and output generation

**Visualization Points:**
- Multi-pass rendering stages (scoring passes, compression passes)
- Texture sampling patterns during query-content coupling
- Buffer writes during patch-level token allocation
- Compression artifact analysis in foveated regions

---

## NSight Graphics: NVIDIA GPU Profiling

NSight Graphics (2024.1+) provides comprehensive GPU profiling and shader analysis, especially for NVIDIA hardware.

From [NSight Graphics Documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/) (accessed 2025-10-31):

### Frame Debugging Capabilities

**Shader Analysis:**
- Per-pixel shader debugging
- Register usage and occupancy analysis
- Branch efficiency in compute shaders
- Memory access patterns during texture sampling

**Performance Metrics:**
- GPU utilization and streaming multiprocessor occupancy
- Memory bandwidth consumption
- L1/L2 cache hit rates
- Stall detection (memory, instruction issue, synchronization)

### Application to Relevance Realization

**Compute Shader Optimization:**
- Profile multi-threaded relevance scoring (Propositions, Perspectives, Participatory)
- Monitor tension balancing compute kernels
- Track attention mask generation for token allocation
- Measure occupancy of compression shaders

**GPU Trace Analysis:**
- Identify bottlenecks in query-content coupling computation
- Visualize synchronization points between scoring phases
- Analyze memory access patterns during patch LOD assignment
- Detect redundant texture samplings in perspectival knowing

---

## Comparing RenderDoc vs NSight Graphics

From [Massed Compute - RenderDoc vs NSight](https://massedcompute.com/faq-answers/) (accessed 2025-10-31):

| Aspect | RenderDoc | NSight Graphics |
|--------|-----------|-----------------|
| **Primary Use** | Frame capture and detailed API analysis | GPU performance profiling |
| **APIs Supported** | Vulkan, D3D11/12, OpenGL, Metal | D3D11/12, Vulkan, Metal (NVIDIA focus) |
| **Strengths** | Frame reconstruction, shader editing, buffer inspection | GPU metrics, occupancy analysis, NVIDIA optimization |
| **Workflow** | Offline frame analysis | Real-time or captured profiling |
| **Debug Info** | Complete command stream with state | Performance counters and hardware metrics |

**Recommendation for ARR-COC:**
- **RenderDoc**: Initial debugging, command sequence validation, texture coordinate mapping
- **NSight Graphics**: Performance optimization, GPU utilization tuning, occupancy analysis

---

## WebGL and WebGPU Profiling

### WebGL Debugging (Browser-Based)

**Chrome DevTools:**
- JavaScript performance timeline integration
- WebGL context state inspection
- Shader source analysis
- Texture and buffer inspection

From [Real-time Rendering - WebGL Tools](https://www.realtimerendering.com/blog/webgl-debugging-and-profiling-tools/) (accessed 2025-10-31):
- Frame timing analysis
- Draw call identification
- State machine validation

**SpectorJS:**
- WebGL-specific debugging extension
- Command-list capture
- State tracking
- Texture preview

### WebGPU Profiling (Modern Approach)

From [GPU Profiling for WebGPU - Windows with Chrome](https://frguthmann.github.io/posts/profiling_webgpu/) (accessed 2025-10-31):

**Challenge:** WebGPU in Chrome runs on native APIs (D3D12, Vulkan, Metal) but doesn't expose profiling data directly.

**Solution: D3D12 Shimming:**
- Download custom `d3d12.dll` shim from [d3d12_webgpu_shim](https://github.com/frguthmann/d3d12_webgpu_shim)
- Place in Chrome installation directory
- Enable PIX debug markers via `WinPixEventRuntime.dll`

**Chrome Launch Arguments:**
```bash
--no-sandbox \
--disable-gpu-sandbox \
--disable-gpu-watchdog \
--disable-direct-composition \
--enable-dawn-features=emit_hlsl_debug_symbols,disable_symbol_renaming
```

**Profiler Integration:**
- **AMD Radeon Developer Tool (RDP):** Automatic detection with shim
- **NVIDIA NSight Graphics:** Requires `--no-sandbox` flag

---

## Debug Scopes: Hierarchical GPU Profiling

From [GPU Debug Scopes - RAII Implementation](https://wunkolo.github.io/post/2024/09/gpu-debug-scopes/) (accessed 2025-10-31):

### Vulkan Implementation (VK_EXT_debug_utils)

```cpp
class DebugScope {
private:
    const VkCommandBuffer commandBuffer;

public:
    DebugScope(
        VkCommandBuffer targetCommandBuffer,
        const char* scopeName,
        std::span<const float, 4> scopeColor
    ) : commandBuffer(targetCommandBuffer) {
        VkDebugUtilsLabelEXT label = {};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = scopeName;
        std::copy_n(scopeColor.begin(), 4, label.color);
        vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &label);
    }

    ~DebugScope() {
        vkCmdEndDebugUtilsLabelEXT(commandBuffer);
    }
};
```

**ARR-COC Usage:**
```cpp
{
    DebugScope knowingScope(cmd, "Knowing Phase", {0.0f, 1.0f, 0.0f, 1.0f});

    {
        DebugScope propScope(cmd, "Propositional Scoring", {0.2f, 1.0f, 0.2f, 1.0f});
        // Shannon entropy calculation
    }

    {
        DebugScope perspScope(cmd, "Perspectival Salience", {0.2f, 1.0f, 0.5f, 1.0f});
        // Jungian archetype scoring
    }

    {
        DebugScope partScope(cmd, "Participatory Coupling", {0.2f, 0.8f, 1.0f, 1.0f});
        // Cross-attention computation
    }
}

{
    DebugScope balancingScope(cmd, "Balancing Phase", {1.0f, 1.0f, 0.0f, 1.0f});
    // Opponent processing tension navigation
}

{
    DebugScope attendingScope(cmd, "Attending Phase", {1.0f, 0.5f, 0.0f, 1.0f});
    // LOD allocation based on realized relevance
}
```

### OpenGL Implementation (GL_KHR_debug)

```cpp
class DebugScope {
    inline static GLuint GlobalScopeDepth = 0;
    const GLuint ScopeDepth;

public:
    DebugScope(std::string_view ScopeName)
        : ScopeDepth(GlobalScopeDepth++) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, ScopeDepth,
                        ScopeName.size(), ScopeName.data());
    }

    ~DebugScope() {
        glPopDebugGroup();
        GlobalScopeDepth--;
    }
};
```

### Coloration Strategies

**Depth-Based Coloring:**
- Maintain scope depth counter incrementing in constructor/destructor
- Use procedural palettes (e.g., Inigo Quilez color palettes) for visual differentiation
- Enables quick identification of nested phases

**Functional Coloring:**
- All scoring operations: Green
- Balancing operations: Yellow
- Attending (allocation): Orange
- Realizing (execution): Magenta
- Transfer operations: Cyan

**Value-Based Coloring (Advanced):**
- Color compression ratios based on actual values
- Color attention weights by magnitude
- Visualize token budgets as heatmaps

---

## ARR-COC Profiling Workflow

### Phase 1: Baseline Profiling

1. **Capture frames** with RenderDoc at different query complexities
2. **Measure baseline metrics:**
   - Total pipeline execution time
   - Per-phase overhead (knowing, balancing, attending, realizing)
   - Memory bandwidth for patch streaming
   - Texture cache hit rates

### Phase 2: Bottleneck Identification

1. **Use NSight GPU Trace** for detailed metrics:
   - SM occupancy in relevance scoring kernels
   - Memory stall analysis in texture sampling
   - Synchronization overhead between phases

2. **Analyze debug scopes** in RenderDoc:
   - Identify phases with excessive API call counts
   - Check for redundant texture binds or buffer updates

### Phase 3: Optimization Validation

1. **Implement optimizations** (e.g., query caching, tensor tiling)
2. **Re-capture frames** with same tools
3. **Compare metrics:**
   - Execution time reduction
   - Occupancy improvement
   - Cache efficiency gains

---

## Performance Optimization Strategies

### Reducing Token Allocation Overhead

**Profile with NSight:**
- Measure compute shader occupancy during LOD calculation
- Identify memory bottlenecks in attention mask generation
- Track stalls in synchronization between scoring and allocation

**Optimize:**
- Increase thread block size for better occupancy
- Batch token allocation across multiple patches
- Use shared memory for temporary relevance scores

### Query-Content Coupling Efficiency

**Profile with RenderDoc:**
- Trace cross-attention texture access patterns
- Identify redundant similarity computations
- Verify patch-query interaction caching

**Optimize:**
- Memoize query embeddings across frame sequences
- Use gather operations instead of individual lookups
- Implement early termination for low-relevance patches

### Compression Kernel Optimization

**Profile with NSight:**
- Occupancy in compression shaders (BC7, ETC2)
- Register pressure from simultaneous token scoring
- Memory access coherence in compression writes

**Optimize:**
- Fuse multiple compression formats in single pass
- Pre-sort patches by compression complexity
- Use texture arrays for unified compression output

---

## Sources

**Web Research:**
- [GPU Debug Scopes - RAII Patterns](https://wunkolo.github.io/post/2024/09/gpu-debug-scopes/) (accessed 2025-10-31)
- [NVIDIA NSight Graphics User Guide](https://docs.nvidia.com/nsight-graphics/UserGuide/) (accessed 2025-10-31)
- [RenderDoc vs NSight Graphics Comparison](https://massedcompute.com/faq-answers/) (accessed 2025-10-31)
- [GPU Profiling for WebGPU on Windows](https://frguthmann.github.io/posts/profiling_webgpu/) - GitHub shimming approach (accessed 2025-10-31)
- [Real-time Rendering - WebGL Debugging Tools](https://www.realtimerendering.com/blog/webgl-debugging-and-profiling-tools/) (accessed 2025-10-31)

**Tool Documentation:**
- [RenderDoc Official](https://renderdoc.org/)
- [NVIDIA NSight Graphics](https://developer.nvidia.com/nsight-graphics/get-started)
- [d3d12_webgpu_shim GitHub](https://github.com/frguthmann/d3d12_webgpu_shim)
- [Vulkan Debug Utils Extension](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_debug_utils.html)
- [OpenGL KHR Debug Extension](https://registry.khronos.org/OpenGL/extensions/KHR/KHR_debug.txt)
