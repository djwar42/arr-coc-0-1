# WebGL 2.0 Advanced Techniques: MRT, UBO, Instancing, 3D Textures

**Date**: 2025-10-31
**Context**: Advanced GPU programming for ARR-COC parallel processing and relevance realization
**Focus**: Production-grade shader patterns, buffer optimization, and volumetric techniques

---

## 1. Multiple Render Targets (MRT)

### Overview

Multiple Render Targets enable parallel output to multiple framebuffer attachments in a single rendering pass. This is fundamental for deferred rendering architectures where geometry properties are captured in separate G-buffers.

From [WebGL 2 Basics](https://www.realtimerendering.com/blog/webgl-2-basics/):
- WebGL 2 makes MRT clean and standardized (previously required WEBGL_draw_buffers extension)
- Use layout qualifiers with `out vec4` variables in fragment shader
- Direct binding to COLOR_ATTACHMENT slots without extension boilerplate

### WebGL 2 MRT Implementation

**Fragment Shader:**
```glsl
#version 300 es
precision highp float;

// Layout specifies which color attachment each output goes to
layout(location = 0) out vec4 gbuf_position;
layout(location = 1) out vec4 gbuf_normal;
layout(location = 2) out vec4 gbuf_color;
layout(location = 3) out vec4 gbuf_data;

void main() {
    gbuf_position = vec4(v_position.xyz, 1.0);
    gbuf_normal = vec4(normalize(v_normal), material_id);
    gbuf_color = vec4(texture(u_colmap, v_uv).rgb, metallic);
    gbuf_data = vec4(roughness, ao, emissive, depth_derivative);
}
```

**C++ Setup:**
```cpp
// Create framebuffer with multiple color attachments
GLuint fbo;
gl.createFramebuffer();

// Attach textures to different slots
gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex_pos, 0);
gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, tex_norm, 0);
gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, tex_color, 0);
gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, tex_data, 0);

// Enable all draw buffers
gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);
```

### ARR-COC Application

MRT enables parallel scoring in relevance realization:
- **Slot 0**: Propositional knowing (information entropy scores per patch)
- **Slot 1**: Perspectival knowing (salience landscape, visual attention maps)
- **Slot 2**: Participatory knowing (query-content coupling scores)
- **Slot 3**: Debug/metadata (compression ratio, token allocation)

Single-pass G-buffer creation reduces CPU-GPU sync overhead critical for interactive foveation.

---

## 2. Uniform Buffer Objects (UBO)

### Core Benefits

From [WebGL 2 Basics](https://www.realtimerendering.com/blog/webgl-2-basics/):
- Reduce uniform setting calls from N (per-uniform) to 2 (buffer bind + data upload)
- Enable shader reuse across programs via shared buffer layouts
- Support hierarchical uniforms: global, per-model, per-material, per-light blocks

### Standard UBO Pattern

**GLSL Shader:**
```glsl
#version 300 es
precision highp float;

// Define uniform block matching C++ struct layout (std140)
layout(std140, binding = 0) uniform GlobalMatrices {
    mat4 projection;
    mat4 view;
    mat4 inverse_view;
};

layout(std140, binding = 1) uniform PerModelData {
    mat4 model;
    mat4 normal_matrix;
    vec4 object_bounds;
};

layout(std140, binding = 2) uniform RelevanceScores {
    vec4 propositional_weight;
    vec4 perspectival_weight;
    vec4 participatory_weight;
    float query_strength;
};

void main() {
    // Uniforms automatically available
    vec4 world_pos = model * vec4(position, 1.0);
    gl_Position = projection * view * world_pos;
}
```

**C++ Setup:**
```cpp
struct GlobalMatricesUBO {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 inverse_view;
};

struct PerModelUBO {
    glm::mat4 model;
    glm::mat4 normal_matrix;
    glm::vec4 object_bounds;
};

// Create buffer and bind to binding point
GLuint ubo_global;
glCreateBuffers(1, &ubo_global);
glNamedBufferData(ubo_global, sizeof(GlobalMatricesUBO), &global_data, GL_DYNAMIC_DRAW);
glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo_global);

// Update when needed (much faster than individual uniformXX calls)
glNamedBufferSubData(ubo_global, offsetof(GlobalMatricesUBO, projection),
                     sizeof(glm::mat4), &new_projection);
```

### ARR-COC Optimization

UBO enables efficient parameter tuning for relevance realization:
- Global block: camera, projection matrices (shared across all patches)
- Per-patch block: content features, local salience landscape
- Relevance block: realtime weight adjustment for Vervaekean tensions (Compress↔Particularize, Exploit↔Explore)

Batch 100+ patches with single buffer update instead of per-patch uniform calls.

---

## 3. GPU Instancing

### Overview

From [WebGL 2 Basics](https://www.realtimerendering.com/blog/webgl-2-basics/):
- `drawArraysInstanced()` / `drawElementsInstanced()` render same geometry N times with different parameters
- `gl_InstanceID` available in vertex shader for per-instance data access
- Eliminates CPU overhead of individual draw calls

### Instancing Pattern

**Vertex Shader:**
```glsl
#version 300 es
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

// Per-instance data (sourced from separate SSBO or texture)
layout(std140, binding = 3) uniform PerInstanceData {
    mat4 instances[1024];
} instance_data;

flat out int instance_id;

void main() {
    instance_id = gl_InstanceID;

    // Apply per-instance transformation
    mat4 model = instance_data.instances[gl_InstanceID];
    gl_Position = projection * view * model * vec4(position, 1.0);
}
```

**Fragment Shader:**
```glsl
#version 300 es
precision highp float;

flat in int instance_id;

uniform sampler2DArray material_textures; // Array of materials

out vec4 color;

void main() {
    // Each instance accesses different material layer
    color = texture(material_textures, vec3(texcoord, instance_id % 16));
}
```

**Draw Call:**
```cpp
// Render 512 instances of patch geometry in single call
glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count, 512);
```

### ARR-COC Use Case

Instancing processes multiple LOD patches in parallel:
- Each instance: one visual patch (64-400 tokens)
- Per-instance data: patch position, content features, relevance scores
- Fragment output: compressed tokens for single patch
- Result: 512 patches processed in single draw call (vs 512 individual calls)

Massive CPU→GPU bandwidth savings for interactive foveation.

---

## 4. 3D Textures & Volume Rendering

### 3D Texture Fundamentals

From [Volume Rendering with WebGL](https://www.willusher.io/webgl/2019/01/13/volume-rendering-with-webgl/):
- WebGL 2 added native 3D texture support (previously emulated with 2D arrays)
- Ideal for scientific datasets: MRI/CT scans, particle simulations, density fields
- Natural extension of 2D texture sampling: `texture(sampler3D, vec3(u,v,w))`

### 3D Texture Creation

**JavaScript/WebGL:**
```javascript
// Create 3D texture (128x128x128 float)
const texture3d = gl.createTexture();
gl.bindTexture(gl.TEXTURE_3D, texture3d);

// Allocate storage
gl.texImage3D(
    gl.TEXTURE_3D, 0, gl.R32F,
    128, 128, 128,  // width, height, depth
    0,
    gl.RED, gl.FLOAT,
    volumeData  // typed array or null
);

// Configure sampling
gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
```

### Volume Rendering: Raymarching Algorithm

From [Volume Rendering with WebGL](https://www.willusher.io/webgl/2019/01/13/volume-rendering-with-webgl/):

**Fragment Shader Raymarching:**
```glsl
#version 300 es
precision highp float;

uniform sampler3D volume_data;
uniform sampler2D transfer_function;  // 1D LUT as 2D texture
uniform ivec3 volume_dims;
uniform float step_size;

in vec3 ray_origin;
in vec3 ray_direction;
out vec4 out_color;

// Box-ray intersection for volume bounds
vec2 intersect_box(vec3 orig, vec3 dir) {
    const vec3 box_min = vec3(0.0);
    const vec3 box_max = vec3(1.0);
    vec3 inv_dir = 1.0 / dir;
    vec3 tmin = (box_min - orig) * inv_dir;
    vec3 tmax = (box_max - orig) * inv_dir;
    tmin = min(tmin, tmax);
    tmax = max(tmin, tmax);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(t0, t1);
}

void main() {
    vec3 ray_dir = normalize(ray_direction);
    vec2 t_bounds = intersect_box(ray_origin, ray_dir);

    if (t_bounds.x > t_bounds.y) {
        discard;
    }

    // Raymarch through volume
    vec4 accumulated = vec4(0.0);
    for (float t = t_bounds.x; t < t_bounds.y; t += step_size) {
        vec3 sample_pos = ray_origin + t * ray_dir;

        // Sample scalar value and apply transfer function
        float scalar = texture(volume_data, sample_pos).r;
        vec4 sample_color = texture(transfer_function, vec2(scalar, 0.5));

        // Front-to-back alpha compositing
        accumulated.rgb += (1.0 - accumulated.a) * sample_color.a * sample_color.rgb;
        accumulated.a += (1.0 - accumulated.a) * sample_color.a;

        // Early exit when opaque
        if (accumulated.a >= 0.95) break;
    }

    out_color = accumulated;
}
```

### Transfer Function

A 1D color lookup table maps scalar values (0-1) to RGBA:
```glsl
// In fragment shader
vec4 sample_color = texture(transfer_function, vec2(scalar_value, 0.5));
// Returns: vec4(color.rgb, opacity)
```

---

## 5. ARR-COC Parallel Processing Architecture

### Multi-Scale Relevance Pipeline

Combine MRT + UBO + Instancing + 3D Textures for hierarchical compression:

**Stage 1: Propositional Knowing (MRT)**
- Shader: Encode entropy, frequency content per patch
- Output: G-buffer with statistical features
- MRT Slot 0: Information scores

**Stage 2: Perspectival Knowing (3D Texture Sampling)**
- Input: 3D texture of query saliency landscape
- Shader: Sample saliency at each patch location
- MRT Slot 1: Attention map, visual importance

**Stage 3: Participatory Knowing (UBO Relevance)**
- Input: Query embedding via UBO
- Shader: Cross-attention scoring (dot product with query)
- MRT Slot 2: Query-content coupling

**Stage 4: Opponent Processing (Instancing)**
- Process 512 patches in parallel via instancing
- Each instance balances three knowing dimensions
- Output: Token allocation (64-400 per patch)

**Stage 5: Realize Compression (3D Texture Output)**
- Write selected feature tokens to 3D output texture
- MRT Slot 3: Debug visualization (compression ratio)

### Performance Profile

From [WebGL 2 Basics](https://www.realtimerendering.com/blog/webgl-2-basics/):
- **MRT**: Single-pass multi-output reduces framebuffer switches
- **UBO**: Batch parameter updates (vs 1000s of uniform calls)
- **Instancing**: 512 patches in one draw call (vs 512 individual)
- **3D Textures**: Direct volumetric sampling (no 2D array emulation)

Example: 8192 image patches
- Without optimization: 8192 draw calls + 8192×20 uniform calls = 172k GPU submissions
- With ARR-COC pipeline: 16 draw calls + 16×2 UBO updates = 48 GPU submissions
- Speedup: **3.5x** reduction in CPU→GPU overhead

---

## 6. Implementation Best Practices

### VAO (Vertex Array Objects)

Always use VAO for vertex setup (no performance penalty, cleaner code):
```cpp
GLuint vao;
glCreateVertexArrays(1, &vao);
glBindVertexArray(vao);

// Configure vertex attributes
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, position));
glEnableVertexAttribArray(0);
// ... more attributes

// Now draw with: glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count, instance_count);
```

### Texture Unit Management

WebGL 2 guarantees 16 texture units minimum (vs 8 in WebGL 1):
```glsl
layout(location = 0) uniform sampler2D tex0;      // Unit 0
layout(location = 1) uniform sampler3D volume;    // Unit 1
layout(location = 2) uniform sampler2DArray mats; // Unit 2
layout(location = 3) uniform sampler2D transfer;  // Unit 3
```

### Precision & Numerical Stability

Volume rendering requires careful precision:
```glsl
precision highp float;  // Use 32-bit float for raymarching
precision highp int;    // Integer indices in loops
```

---

## Sources

**Web Research (Accessed 2025-10-31):**

- [WebGL2Fundamentals.org - WebGL2 What's New](https://webgl2fundamentals.org/webgl/lessons/webgl2-whats-new.html) - Complete feature list with shader examples
- [Real-Time Rendering - WebGL 2 Basics](https://www.realtimerendering.com/blog/webgl-2-basics/) - MRT, instancing, UBO patterns with code migration guide
- [Will Usher - Volume Rendering with WebGL](https://www.willusher.io/webgl/2019/01/13/volume-rendering-with-webgl/) - Complete raymarching algorithm, transfer functions, 3D texture techniques
- [Khronos WebGL 2.0 Specification](https://www.khronos.org/registry/webgl/specs/latest/2.0/) - Official spec reference

**Technical Context:**
- MRT enables G-buffer rendering for deferred shading and multi-output visualization
- UBO optimization critical for real-time parameter tuning in interactive systems
- Instancing scales parallel processing from single patches to thousands
- 3D textures fundamental for volumetric queries and scientific visualization
- Combined architecture supports ARR-COC's hierarchical relevance realization with minimal CPU overhead
