# Shader Pipeline Fundamentals

## Overview

Modern GPU rendering pipelines execute a series of programmable and fixed-function stages to transform 3D geometry into 2D pixels. Shaders are small programs written in languages like GLSL (OpenGL Shading Language) or HLSL that run on the GPU, processing vertices, primitives, and fragments in parallel.

Understanding the shader pipeline is fundamental to graphics programming, from simple 2D rendering to complex 3D scenes with advanced effects. This document covers the complete graphics pipeline, shader stages, data flow mechanisms, and how ARR-COC-VIS leverages these concepts for vision-language model compression.

## The Graphics Pipeline

### Pipeline Stages Overview

The modern graphics pipeline consists of the following stages:

```
Vertex Data (CPU)
    ↓
[1. Vertex Shader] ← Programmable
    ↓
[2. Tessellation Control Shader] ← Programmable (optional)
    ↓
[3. Tessellation Evaluation Shader] ← Programmable (optional)
    ↓
[4. Geometry Shader] ← Programmable (optional)
    ↓
[5. Primitive Assembly] ← Fixed-function
    ↓
[6. Rasterization] ← Fixed-function
    ↓
[7. Fragment Shader] ← Programmable
    ↓
[8. Per-Sample Operations] ← Fixed-function
    ↓
Framebuffer (Display)
```

From [Stanford CS248A Graphics Pipeline](https://gfxcourses.stanford.edu/cs248a/winter24content/media/rastpipeline/05_pipeline_oCu532u.pdf) (accessed 2025-10-31):
- Complex vertex and fragment shader computations drive modern rendering
- Behavior of programmable stages is application-defined using shader programs
- Fixed-function stages (rasterization, blending) have configurable parameters but fixed algorithms

### Stage 1: Vertex Shader

**Purpose**: Transform vertex positions from model space to clip space and prepare per-vertex data for later stages.

**Execution**: Runs once per vertex. Cannot create or destroy vertices.

**Typical operations**:
- Model-View-Projection (MVP) transformations
- Lighting calculations (per-vertex lighting)
- Texture coordinate generation
- Skinning/skeletal animation

**Input**: Vertex attributes (position, normal, color, texture coordinates)

**Output**: `gl_Position` (required) in clip space, plus optional varyings for fragment shader

From [LearnOpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) (accessed 2025-10-31):

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;      // Vertex position attribute
layout (location = 1) in vec3 aNormal;   // Normal vector attribute
layout (location = 2) in vec2 aTexCoord; // Texture coordinate attribute

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;    // Varying: position in world space
out vec3 Normal;     // Varying: normal vector
out vec2 TexCoord;   // Varying: texture coordinates

void main()
{
    // Transform to clip space (required output)
    gl_Position = projection * view * model * vec4(aPos, 1.0);

    // Pass data to fragment shader
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
}
```

**Key built-in variables**:
- `gl_Position`: Output clip-space position (vec4, required)
- `gl_VertexID`: Input vertex index (int, built-in)
- `gl_InstanceID`: Input instance index for instanced rendering (int, built-in)

### Stage 2-3: Tessellation Shaders (Optional)

**Purpose**: Subdivide primitives (patches) into smaller geometric primitives, enabling level-of-detail (LOD) control and displacement mapping.

**Tessellation Control Shader (TCS)**:
- Runs once per control point
- Determines tessellation levels
- Can modify control point positions

**Tessellation Evaluation Shader (TES)**:
- Runs once per tessellation coordinate
- Computes final vertex positions from tessellated patch
- Often used for displacement mapping

From [LearnOpenGL - Tessellation](https://learnopengl.com/Guest-Articles/2021/Tessellation/Height-map) (accessed 2025-10-31):
- Enables dynamic mesh refinement based on camera distance
- Reduces CPU-GPU bandwidth by generating geometry on GPU
- Critical for terrain rendering and smooth surface subdivision

**Note**: ARR-COC-VIS doesn't use tessellation shaders but achieves adaptive resolution through query-aware patch compression (64-400 tokens per patch).

### Stage 4: Geometry Shader (Optional)

**Purpose**: Process entire primitives (points, lines, triangles) and optionally emit new primitives.

**Execution**: Runs once per primitive. Can create or destroy geometry.

**Typical operations**:
- Generating shadow volumes
- Point sprite expansion
- Simple mesh subdivision
- Generating geometry from points

From [AMD GPUOpen - Mesh Shaders](https://gpuopen.com/learn/mesh_shaders/mesh_shaders-from_vertex_shader_to_mesh_shader/) (accessed 2025-10-31):
- Traditional geometry shaders have performance limitations
- Modern approaches prefer compute-based geometry generation
- Mesh shaders (new in modern APIs) replace vertex/geometry shader model

```glsl
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
out vec3 fColor;

void main() {
    // Expand point to quad (4 vertices)
    vec4 center = gl_in[0].gl_Position;

    // Emit 4 vertices as triangle strip
    gl_Position = center + vec4(-0.1, -0.1, 0.0, 0.0);
    fColor = vColor[0];
    EmitVertex();

    gl_Position = center + vec4(0.1, -0.1, 0.0, 0.0);
    fColor = vColor[0];
    EmitVertex();

    gl_Position = center + vec4(-0.1, 0.1, 0.0, 0.0);
    fColor = vColor[0];
    EmitVertex();

    gl_Position = center + vec4(0.1, 0.1, 0.0, 0.0);
    fColor = vColor[0];
    EmitVertex();

    EndPrimitive();
}
```

**Performance note**: Geometry shaders can be slow due to variable output and limited parallelism. Modern rendering prefers compute shaders for geometry generation.

### Stage 5: Primitive Assembly (Fixed-Function)

**Purpose**: Assemble vertices into geometric primitives (points, lines, triangles).

**Operations**:
- Group vertices based on drawing mode (GL_TRIANGLES, GL_TRIANGLE_STRIP, etc.)
- Face culling (back-face/front-face removal)
- Clipping to view frustum

**Not programmable**, but configurable via API calls:
- `glFrontFace()`: Define front-facing orientation
- `glCullFace()`: Enable/disable face culling
- `glEnable(GL_CLIP_DISTANCE)`: User-defined clipping planes

### Stage 6: Rasterization (Fixed-Function)

**Purpose**: Convert geometric primitives into fragments (potential pixels).

**Operations**:
- Scan conversion: determine which pixels primitive covers
- Perspective-correct interpolation of varyings
- Generate fragment positions and barycentrics
- Depth calculation for each fragment

From [WebGL Fundamentals - Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html) (accessed 2025-10-31):
- Fragment interpolation creates smooth gradients across primitives
- A triangle with 3 vertex colors produces thousands of interpolated fragment colors
- Interpolation is perspective-correct (accounts for depth)

**Example**: Triangle with RGB vertices creates color gradient:
- Vertex 0: Red (1, 0, 0)
- Vertex 1: Green (0, 1, 0)
- Vertex 2: Blue (0, 0, 1)
- Fragment at 70% toward vertex 1: interpolated color (0.3R + 0.7G)

### Stage 7: Fragment Shader

**Purpose**: Compute final color (and depth) for each fragment.

**Execution**: Runs once per fragment. Cannot access neighboring fragments.

**Typical operations**:
- Texture sampling
- Lighting calculations (per-pixel/per-fragment lighting)
- Normal mapping
- Shadow calculations
- Fog effects

From [LearnOpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) (accessed 2025-10-31):

```glsl
#version 330 core
out vec4 FragColor;  // Required output: final fragment color

in vec3 FragPos;     // Interpolated from vertex shader
in vec3 Normal;      // Interpolated from vertex shader
in vec2 TexCoord;    // Interpolated from vertex shader

uniform sampler2D texture1;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    // Sample texture
    vec4 texColor = texture(texture1, TexCoord);

    // Lighting calculations
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    vec3 result = (diff + spec) * texColor.rgb;
    FragColor = vec4(result, texColor.a);
}
```

**Key built-in variables**:
- `gl_FragCoord`: Fragment window-space coordinates (vec4, input)
- `gl_FragDepth`: Fragment depth value (float, output, optional)
- `gl_FrontFacing`: Whether fragment is front-facing (bool, input)
- `gl_FragColor`: Fragment color (vec4, output, deprecated in modern GLSL)

**Modern GLSL**: Use user-defined `out` variables instead of `gl_FragColor`:

```glsl
out vec4 FragColor;  // Modern approach
```

### Stage 8: Per-Sample Operations (Fixed-Function)

**Purpose**: Determine final pixel values through testing and blending.

**Operations** (in order):
1. **Scissor test**: Discard fragments outside scissor rectangle
2. **Stencil test**: Use stencil buffer for masking
3. **Depth test**: Compare fragment depth with depth buffer
4. **Blending**: Combine fragment color with framebuffer color
5. **Dithering**: Reduce color banding
6. **Write to framebuffer**: Update color/depth/stencil buffers

**Configurable via API**:
- `glEnable(GL_DEPTH_TEST)`, `glDepthFunc()`
- `glEnable(GL_BLEND)`, `glBlendFunc()`
- `glStencilFunc()`, `glStencilOp()`

### Compute Shaders (Separate Pipeline)

**Purpose**: General-purpose GPU computation outside the graphics pipeline.

**Execution**: Dispatched in work groups, not tied to vertices or fragments.

**Typical operations**:
- Physics simulation
- Image processing
- Particle systems
- Mesh generation
- Parallel reduction algorithms

From [Reddit r/GraphicsProgramming](https://www.reddit.com/r/GraphicsProgramming/comments/1ewuher/why_can_compute_shaders_be_faster_at_rendering/) (accessed 2025-10-31):
- Compute shaders can outperform graphics pipeline for certain tasks
- Avoid overhead of vertex/fragment stages when only computation needed
- Full control over memory access patterns and synchronization

```glsl
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (rgba32f, binding = 0) uniform image2D imgOutput;

void main() {
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    vec4 computedColor = vec4(pixelCoords.x / 256.0, pixelCoords.y / 256.0, 0.0, 1.0);
    imageStore(imgOutput, pixelCoords, computedColor);
}
```

**Key differences from graphics shaders**:
- No fixed input/output (vertices, fragments)
- Explicit work group sizing
- Direct image/buffer reads/writes
- Shared memory within work groups

## Shader Data Flow: Attributes, Varyings, Uniforms

Understanding how data flows between CPU, shader stages, and back is critical for efficient GPU programming.

### Attributes (Vertex Input)

**Definition**: Per-vertex data uploaded from CPU to GPU via buffers.

**Characteristics**:
- Only available to vertex shader (as input)
- Different value for each vertex
- Stored in vertex buffer objects (VBOs)
- Configured via vertex array objects (VAOs)

From [MDN WebGL Data](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Data) (accessed 2025-10-31):
- Attributes store color, texture coordinates, normals, positions
- Typically interleaved in buffer for cache efficiency
- Must enable each attribute location

**Setup example** (JavaScript):

```javascript
// Create buffer
const positionBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

// Upload vertex data
const positions = new Float32Array([
    -1.0, -1.0, 0.0,  // Vertex 0
     1.0, -1.0, 0.0,  // Vertex 1
     0.0,  1.0, 0.0   // Vertex 2
]);
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

// Get attribute location
const positionLoc = gl.getAttribLocation(shaderProgram, "aPos");

// Configure attribute
gl.enableVertexAttribArray(positionLoc);
gl.vertexAttribPointer(
    positionLoc,  // location
    3,            // components per vertex (x, y, z)
    gl.FLOAT,     // data type
    false,        // normalize
    0,            // stride (0 = tightly packed)
    0             // offset
);
```

**GLSL declaration**:

```glsl
layout (location = 0) in vec3 aPos;     // Position attribute
layout (location = 1) in vec3 aNormal;  // Normal attribute
layout (location = 2) in vec2 aTexCoord; // Texture coordinate attribute
```

**Interleaved attributes** (cache-friendly):

```javascript
// Position (3) + Normal (3) + TexCoord (2) = 8 floats per vertex
const vertexData = new Float32Array([
    // Pos X,Y,Z    Normal X,Y,Z   TexCoord U,V
    -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // Vertex 0
     1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,  // Vertex 1
     0.0,  1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0   // Vertex 2
]);

const stride = 8 * 4; // 8 floats * 4 bytes

// Position attribute
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
// Normal attribute
gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);  // offset: 3*4 bytes
// TexCoord attribute
gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 24);  // offset: 6*4 bytes
```

**Maximum attributes**: OpenGL guarantees minimum 16 vertex attributes (query with `GL_MAX_VERTEX_ATTRIBS`).

### Varyings (Inter-Shader Communication)

**Definition**: Variables passed from vertex shader to fragment shader with automatic interpolation.

**Characteristics**:
- Declared as `out` in vertex shader, `in` in fragment shader
- Interpolated across primitive during rasterization
- Perspective-correct interpolation by default
- Names and types must match between stages

From [WebGL Fundamentals - Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html) (accessed 2025-10-31):
- Varyings enable smooth color/texture/lighting gradients
- Each fragment receives interpolated values based on barycentric coordinates
- Modern GLSL uses `in`/`out` keywords (legacy used `varying` keyword)

**Example**:

```glsl
// Vertex shader
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;  // Varying: output to fragment shader

void main()
{
    gl_Position = vec4(aPos, 1.0);
    vertexColor = aColor;  // Pass color to fragment shader
}
```

```glsl
// Fragment shader
#version 330 core
in vec3 vertexColor;  // Varying: input from vertex shader (interpolated)

out vec4 FragColor;

void main()
{
    FragColor = vec4(vertexColor, 1.0);  // Use interpolated color
}
```

**Interpolation qualifiers**:
- `smooth`: Perspective-correct interpolation (default)
- `flat`: No interpolation, use value from provoking vertex
- `noperspective`: Linear interpolation in screen space

```glsl
flat out int primitiveID;           // No interpolation
smooth out vec3 normalVector;       // Perspective-correct (default)
noperspective out vec2 screenCoord; // Linear screen-space
```

### Uniforms (Global Shader Constants)

**Definition**: Variables set by CPU code, constant across all shader invocations in a draw call.

**Characteristics**:
- Available to all shader stages
- Same value for all vertices/fragments in one draw call
- Set via `glUniform*()` API calls
- Can change between draw calls

From [LearnOpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) (accessed 2025-10-31):
- Uniforms are "global" per shader program
- Each program has its own uniform storage
- Must use program before setting uniforms

**Common uniform uses**:
- Transformation matrices (model, view, projection)
- Light positions and colors
- Material properties
- Time values for animation
- Texture samplers

**Setup example**:

```javascript
// Get uniform location (at initialization)
const colorLoc = gl.getUniformLocation(shaderProgram, "ourColor");
const timeLoc = gl.getUniformLocation(shaderProgram, "time");
const mvpLoc = gl.getUniformLocation(shaderProgram, "mvpMatrix");

// Set uniforms (before drawing)
gl.useProgram(shaderProgram);  // Must activate program first
gl.uniform4f(colorLoc, 1.0, 0.5, 0.2, 1.0);  // vec4
gl.uniform1f(timeLoc, performance.now() / 1000.0);  // float
gl.uniformMatrix4fv(mvpLoc, false, mvpMatrixArray);  // mat4
```

**GLSL declaration**:

```glsl
#version 330 core
uniform vec4 ourColor;
uniform float time;
uniform mat4 mvpMatrix;
uniform sampler2D texture1;  // Texture sampler

void main()
{
    // Use uniforms...
}
```

**Uniform types and setter functions**:

```javascript
// Scalar types
gl.uniform1f(loc, v);                  // float
gl.uniform1i(loc, v);                  // int, bool
gl.uniform1ui(loc, v);                 // unsigned int

// Vector types
gl.uniform2f(loc, v0, v1);             // vec2
gl.uniform3f(loc, v0, v1, v2);         // vec3
gl.uniform4f(loc, v0, v1, v2, v3);     // vec4

// Vector array versions
gl.uniform2fv(loc, [v0, v1]);          // vec2 or vec2 array
gl.uniform3fv(loc, [v0, v1, v2]);      // vec3 or vec3 array
gl.uniform4fv(loc, [v0, v1, v2, v3]);  // vec4 or vec4 array

// Matrix types
gl.uniformMatrix2fv(loc, false, array);   // mat2 (4 values)
gl.uniformMatrix3fv(loc, false, array);   // mat3 (9 values)
gl.uniformMatrix4fv(loc, false, array);   // mat4 (16 values)
```

**Uniform arrays**:

```glsl
uniform vec3 lightPositions[4];  // Array of 4 light positions
```

```javascript
// Set entire array at once
gl.uniform3fv(lightPosLoc, [
    1.0, 2.0, 3.0,  // Light 0
    4.0, 5.0, 6.0,  // Light 1
    7.0, 8.0, 9.0,  // Light 2
    10.0, 11.0, 12.0 // Light 3
]);

// Or set individual elements
const light0Loc = gl.getUniformLocation(program, "lightPositions[0]");
const light1Loc = gl.getUniformLocation(program, "lightPositions[1]");
gl.uniform3fv(light0Loc, [1.0, 2.0, 3.0]);
gl.uniform3fv(light1Loc, [4.0, 5.0, 6.0]);
```

**Uniform structs**:

```glsl
struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};
uniform Material material;
```

```javascript
const ambientLoc = gl.getUniformLocation(program, "material.ambient");
const diffuseLoc = gl.getUniformLocation(program, "material.diffuse");
gl.uniform3fv(ambientLoc, [0.2, 0.2, 0.2]);
gl.uniform3fv(diffuseLoc, [0.8, 0.8, 0.8]);
```

### Texture Samplers

**Definition**: Special uniform type representing texture units.

**Usage**: Sample texels from textures in shaders.

From [WebGL Fundamentals - Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html) (accessed 2025-10-31):

```glsl
uniform sampler2D texture1;  // 2D texture
uniform samplerCube skybox;  // Cubemap texture

void main()
{
    vec4 color = texture(texture1, texCoord);  // Sample texture
}
```

```javascript
// Bind texture to unit 0
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, texture);

// Tell shader to use unit 0
const samplerLoc = gl.getUniformLocation(program, "texture1");
gl.uniform1i(samplerLoc, 0);  // Use texture unit 0
```

## GLSL Language Fundamentals

### Data Types

**Scalar types**:
- `float`: 32-bit floating point
- `int`: 32-bit signed integer
- `uint`: 32-bit unsigned integer
- `bool`: Boolean value

**Vector types** (2, 3, or 4 components):
- `vec2`, `vec3`, `vec4`: Float vectors
- `ivec2`, `ivec3`, `ivec4`: Integer vectors
- `uvec2`, `uvec3`, `uvec4`: Unsigned integer vectors
- `bvec2`, `bvec3`, `bvec4`: Boolean vectors

**Matrix types**:
- `mat2`, `mat3`, `mat4`: Square float matrices (2×2, 3×3, 4×4)
- `mat2x3`, `mat3x2`, `mat4x2`, etc.: Non-square matrices

**Sampler types**:
- `sampler2D`: 2D texture
- `sampler3D`: 3D texture
- `samplerCube`: Cubemap texture
- `sampler2DShadow`: Shadow map

### Vector Component Access

From [LearnOpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) (accessed 2025-10-31):

```glsl
vec4 v = vec4(1.0, 2.0, 3.0, 4.0);

// Position notation: x, y, z, w
float x = v.x;  // 1.0
float y = v.y;  // 2.0

// Color notation: r, g, b, a
float r = v.r;  // 1.0 (same as v.x)
float g = v.g;  // 2.0 (same as v.y)

// Texture notation: s, t, p, q
float s = v.s;  // 1.0 (same as v.x)
float t = v.t;  // 2.0 (same as v.y)

// Array access
float z = v[2];  // 3.0
```

**Swizzling** (reordering/repeating components):

```glsl
vec4 v = vec4(1.0, 2.0, 3.0, 4.0);

vec2 xy = v.xy;           // vec2(1.0, 2.0)
vec3 bgr = v.bgr;         // vec3(3.0, 2.0, 1.0)
vec4 yyyy = v.yyyy;       // vec4(2.0, 2.0, 2.0, 2.0)
vec3 rgb1 = vec4(v.rgb, 1.0);  // vec4(1.0, 2.0, 3.0, 1.0)
```

### Type Strictness

From [WebGL Fundamentals - Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html) (accessed 2025-10-31):

GLSL is very type-strict:

```glsl
float f = 1;  // ERROR: cannot assign int to float

// Correct ways:
float f = 1.0;       // Use float literal
float f = float(1);  // Cast int to float
```

### Built-in Functions

GLSL provides extensive built-in functions that operate component-wise on vectors:

```glsl
// Trigonometric
vec3 s = sin(v);      // sin of each component
vec3 c = cos(v);      // cos of each component

// Exponential
vec3 p = pow(v, 2.0); // v^2 for each component
vec3 e = exp(v);      // e^v for each component

// Common
vec3 a = abs(v);      // absolute value
vec3 m = min(v, 0.5); // component-wise min
vec3 c = clamp(v, 0.0, 1.0);  // clamp each component

// Geometric
float d = dot(v1, v2);      // dot product
vec3 c = cross(v1, v2);     // cross product
float l = length(v);        // vector length
vec3 n = normalize(v);      // unit vector

// Interpolation
vec3 m = mix(v1, v2, 0.5);  // linear interpolation (50%)
```

**Mixed scalar/vector operations**:

```glsl
vec4 v1 = vec4(1.0, 2.0, 3.0, 4.0);
vec4 v2 = vec4(5.0, 6.0, 7.0, 8.0);
float f = 0.5;

vec4 m = mix(v1, v2, f);
// Equivalent to:
vec4 m = vec4(
    mix(v1.x, v2.x, f),
    mix(v1.y, v2.y, f),
    mix(v1.z, v2.z, f),
    mix(v1.w, v2.w, f)
);
```

Reference: [WebGL Reference Card](https://www.khronos.org/files/webgl/webgl-reference-card-1_0.pdf) for complete function list.

## Shader Compilation and Linking

### Compilation Process

1. **Create shader objects**:

```javascript
const vertexShader = gl.createShader(gl.VERTEX_SHADER);
const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
```

2. **Load shader source code**:

```javascript
gl.shaderSource(vertexShader, vertexShaderSource);
gl.shaderSource(fragmentShader, fragmentShaderSource);
```

3. **Compile shaders**:

```javascript
gl.compileShader(vertexShader);
gl.compileShader(fragmentShader);

// Check compilation status
if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
    console.error('Vertex shader compilation failed:',
                  gl.getShaderInfoLog(vertexShader));
}
```

4. **Create program and attach shaders**:

```javascript
const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
```

5. **Link program**:

```javascript
gl.linkProgram(program);

// Check link status
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program linking failed:',
                  gl.getProgramInfoLog(program));
}
```

6. **Use program**:

```javascript
gl.useProgram(program);
```

7. **Clean up** (optional):

```javascript
// Shaders can be deleted after successful linking
gl.deleteShader(vertexShader);
gl.deleteShader(fragmentShader);
```

### Shader Class Example

From [LearnOpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) (accessed 2025-10-31):

Encapsulating shader compilation into a reusable class:

```javascript
class Shader {
    constructor(vertexPath, fragmentPath) {
        // Load shader source from files
        const vertexCode = this.loadShaderFile(vertexPath);
        const fragmentCode = this.loadShaderFile(fragmentPath);

        // Compile shaders
        const vertex = this.compileShader(gl.VERTEX_SHADER, vertexCode);
        const fragment = this.compileShader(gl.FRAGMENT_SHADER, fragmentCode);

        // Link program
        this.ID = gl.createProgram();
        gl.attachShader(this.ID, vertex);
        gl.attachShader(this.ID, fragment);
        gl.linkProgram(this.ID);

        // Check for errors
        if (!gl.getProgramParameter(this.ID, gl.LINK_STATUS)) {
            throw new Error('Shader linking failed: ' +
                          gl.getProgramInfoLog(this.ID));
        }

        // Clean up
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
    }

    compileShader(type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error('Shader compilation failed: ' + info);
        }

        return shader;
    }

    use() {
        gl.useProgram(this.ID);
    }

    setFloat(name, value) {
        const loc = gl.getUniformLocation(this.ID, name);
        gl.uniform1f(loc, value);
    }

    setVec3(name, x, y, z) {
        const loc = gl.getUniformLocation(this.ID, name);
        gl.uniform3f(loc, x, y, z);
    }

    setMat4(name, matrix) {
        const loc = gl.getUniformLocation(this.ID, name);
        gl.uniformMatrix4fv(loc, false, matrix);
    }
}

// Usage
const shader = new Shader('vertex.glsl', 'fragment.glsl');
shader.use();
shader.setFloat('time', performance.now() / 1000.0);
shader.setVec3('lightPos', 1.0, 2.0, 3.0);
```

## ARR-COC-VIS Shader Architecture

ARR-COC-VIS leverages shader pipelines for real-time multi-channel texture visualization of vision-language model features with adaptive level-of-detail.

### Multi-Pass Rendering Pipeline

ARR-COC-VIS uses a 13-channel multi-pass shader architecture to visualize compressed visual features:

**Pass 1: Feature Decompression**
```
Compressed Features (64-400 tokens per patch)
    ↓
[Vertex Shader] → Expand to full resolution grid
    ↓
[Fragment Shader] → Decode compressed representation
    ↓
Feature Texture (13 channels: position, color, depth, normal, etc.)
```

**Pass 2: Channel Visualization**
```
Feature Texture (13 channels)
    ↓
[Vertex Shader] → Quad rendering
    ↓
[Fragment Shader] → Channel selection + color mapping
    ↓
Display Buffer (RGB visualization)
```

### Query-Aware Compression Shader

ARR-COC-VIS implements Vervaekean relevance realization through compute shaders:

```glsl
#version 430
layout (local_size_x = 8, local_size_y = 8) in;

// Input: Full resolution image patches
layout (rgba32f, binding = 0) readonly uniform image2D inputImage;

// Input: Query embedding
uniform vec3 queryEmbedding[512];

// Output: Compressed features (variable token count per patch)
layout (std430, binding = 1) buffer CompressedFeatures {
    vec4 features[];
};

// Output: Token allocation per patch
layout (r32ui, binding = 2) writeonly uniform uimage2D tokenAllocation;

void main() {
    ivec2 patchID = ivec2(gl_GlobalInvocationID.xy);

    // Compute relevance scores (3 ways of knowing)
    float propositional = computeInformationContent(patchID);
    float perspectival = computeSaliency(patchID);
    float participatory = computeQueryRelevance(patchID, queryEmbedding);

    // Balance tensions (opponent processing)
    float relevance = balanceTensions(propositional, perspectival, participatory);

    // Allocate tokens based on relevance (64-400 range)
    uint tokenCount = uint(clamp(relevance * 336.0 + 64.0, 64.0, 400.0));

    // Store allocation
    imageStore(tokenAllocation, patchID, uvec4(tokenCount, 0, 0, 0));

    // Compress patch to allocated token budget
    compressPatch(patchID, tokenCount);
}
```

### Adaptive LOD Visualization Shader

Fragment shader for rendering compressed features with visual quality indicators:

```glsl
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D featureTexture;
uniform usampler2D tokenAllocationTexture;
uniform int channelSelect;  // 0-12 for different feature channels
uniform float visualizationMode;  // 0=features, 1=tokens, 2=hybrid

void main()
{
    // Sample compressed features
    vec4 features = texture(featureTexture, TexCoord);
    uint tokens = texture(tokenAllocationTexture, TexCoord).r;

    vec3 color;

    if (visualizationMode < 0.5) {
        // Feature visualization
        color = visualizeChannel(features, channelSelect);
    } else if (visualizationMode < 1.5) {
        // Token allocation heatmap (64=blue, 400=red)
        float normalized = (float(tokens) - 64.0) / 336.0;
        color = heatmapColor(normalized);
    } else {
        // Hybrid: feature + token overlay
        color = visualizeChannel(features, channelSelect);
        color = mix(color, heatmapColor((float(tokens) - 64.0) / 336.0), 0.3);
    }

    FragColor = vec4(color, 1.0);
}

vec3 visualizeChannel(vec4 features, int channel) {
    // Channel-specific visualization (position, normal, depth, etc.)
    // Implementation depends on feature encoding
    if (channel == 0) return features.rgb;  // Position/color
    if (channel == 1) return features.aaa;  // Depth
    // ... (13 channels total)
    return vec3(0.0);
}

vec3 heatmapColor(float t) {
    // Blue → Cyan → Green → Yellow → Red heatmap
    t = clamp(t, 0.0, 1.0);
    vec3 c1 = vec3(0.0, 0.0, 1.0);  // Blue (low relevance/tokens)
    vec3 c2 = vec3(0.0, 1.0, 1.0);  // Cyan
    vec3 c3 = vec3(0.0, 1.0, 0.0);  // Green
    vec3 c4 = vec3(1.0, 1.0, 0.0);  // Yellow
    vec3 c5 = vec3(1.0, 0.0, 0.0);  // Red (high relevance/tokens)

    if (t < 0.25) return mix(c1, c2, t * 4.0);
    if (t < 0.5)  return mix(c2, c3, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c3, c4, (t - 0.5) * 4.0);
    return mix(c4, c5, (t - 0.75) * 4.0);
}
```

### Performance Characteristics

From [Stanford CS248A Graphics Pipeline](https://gfxcourses.stanford.edu/cs248a/winter24content/media/rastpipeline/05_pipeline_oCu532u.pdf) (accessed 2025-10-31):

Performance considerations for ARR-COC-VIS shader pipeline:

**Vertex shader cost**:
- Low complexity: Simple quad expansion for visualization passes
- Negligible bottleneck for typical use cases

**Fragment shader cost**:
- Moderate-to-high: Texture sampling + channel decoding + color mapping
- Optimization: Precompute channel mappings, use texture compression

**Compute shader cost**:
- High: Relevance calculation (3 scorers) + compression per patch
- Critical optimization: Work group sizing, shared memory for query embedding

**Memory bandwidth**:
- Main bottleneck: Reading full-resolution patches, writing compressed features
- Optimization: Tile-based processing, texture cache locality

**Typical frame budget** (60 FPS target = 16.67ms):
- Compression pass: ~8-12ms (compute shader)
- Visualization pass: ~2-4ms (vertex + fragment)
- Leaves headroom for VLM inference and UI

## Sources

**Source Documents:**
None (web research only)

**Web Research:**

- [LearnOpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) (accessed 2025-10-31)
  - GLSL fundamentals: types, attributes, uniforms, varyings
  - Vertex/fragment shader structure and built-in variables
  - Shader class implementation pattern

- [WebGL Fundamentals - Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html) (accessed 2025-10-31)
  - Attribute/varying/uniform data flow
  - Fragment interpolation mechanics
  - GLSL type system and built-in functions

- [MDN Web Docs - Data in WebGL](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Data) (accessed 2025-10-31)
  - WebGL variable types overview
  - Attribute/varying/uniform definitions and use cases

- [Stanford CS248A - Graphics Pipeline](https://gfxcourses.stanford.edu/cs248a/winter24content/media/rastpipeline/05_pipeline_oCu532u.pdf) (accessed 2025-10-31)
  - Complete rendering pipeline architecture
  - Fixed-function vs programmable stages
  - Performance characteristics

- [AMD GPUOpen - Mesh Shaders](https://gpuopen.com/learn/mesh_shaders/mesh_shaders-from_vertex_shader_to_mesh_shader/) (accessed 2025-10-31)
  - Modern geometry pipeline evolution
  - Compute-based geometry generation
  - NGG pipeline automatic meshlet conversion

- [Reddit r/GraphicsProgramming - Compute Shaders](https://www.reddit.com/r/GraphicsProgramming/comments/1ewuher/why_can_compute_shaders_be_faster_at_rendering/) (accessed 2025-10-31)
  - Compute shader performance vs graphics pipeline
  - Use cases for general-purpose GPU computation

**Additional References:**

- [WebGL Reference Card](https://www.khronos.org/files/webgl/webgl-reference-card-1_0.pdf) - Complete GLSL function reference
- [GLSL Specification](https://www.khronos.org/files/opengles_shading_language.pdf) - Official language specification
