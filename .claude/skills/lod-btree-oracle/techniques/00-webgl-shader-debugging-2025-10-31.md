# WebGL Shader Debugging & Texture Inspection Tools (2025)

**Topic**: Advanced WebGL debugging tools for shader inspection and texture visualization
**Date**: 2025-10-31
**Focus**: SpectorJS, Chrome DevTools, and debugging workflows for ARR-COC texture arrays

---

## Overview

WebGL shader debugging remains challenging due to the asynchronous, GPU-based nature of rendering. Unlike CPU debugging with breakpoints and variable inspection, WebGL requires specialized tools to capture, inspect, and replay rendering commands. This document covers modern (2024-2025) debugging tools and workflows, with specific focus on texture inspection for multi-channel arrays like ARR-COC's 13-channel relevance textures.

**Key Challenge**: Traditional debuggers cannot step through shader code or inspect GPU memory. WebGL debugging tools bridge this gap through frame capture, command replay, and state inspection.

---

## Section 1: SpectorJS - Primary WebGL Debugging Tool

From [Debugging WebGL with SpectorJS](https://www.realtimerendering.com/blog/debugging-webgl-with-spectorjs/) (Real-Time Rendering, June 2017, accessed 2025-10-31):

### What is SpectorJS?

SpectorJS is an **open-source WebGL inspector** developed by the BabylonJS team, designed to capture and analyze complete WebGL frames with full WebGL2 support.

**Key Features:**
- Frame capture with visual state changes timeline
- Complete command list with arguments and call stacks
- Texture inspection (2D, 3D, cube maps, draw buffers)
- Shader source code viewer with beautification
- Memory statistics and draw call analysis
- JSON export/import for sharing captures

**Availability:**
- Browser extension: [Chrome](https://chrome.google.com/webstore/detail/spectorjs/denbgaamihkadbghdceggmchnflmhpmk), [Firefox](https://addons.mozilla.org/en-US/firefox/addon/spector-js/)
- NPM package: `npm install spectorjs`
- CDN: `https://cdn.jsdelivr.net/npm/spectorjs@0.9.30/dist/spector.bundle.js`

### Core Debugging Workflow

From [GitHub - BabylonJS/Spector.js](https://github.com/BabylonJS/Spector.js) (accessed 2025-10-31):

**1. Enable Spector on a page:**
```javascript
var spector = new SPECTOR.Spector();
spector.displayUI();  // Shows embedded UI
```

**2. Capture a frame:**
```javascript
var canvas = document.getElementById("renderCanvas");
spector.captureCanvas(canvas);
```

**3. Programmatic capture with callback:**
```javascript
spector.onCapture.add((capture) => {
    // Access complete frame data as JSON
    console.log(JSON.stringify(capture));
});
```

**4. Continuous tracking (spy mode):**
```javascript
spector.spyCanvases();  // Enables pre-capture tracking
// Provides texture inputs and memory consumption data
```

### Visual State Inspection

SpectorJS displays **all visual state changes** during frame creation:

- Left panel: Thumbnails of each render target update
- Click thumbnail → Jumps to associated WebGL command
- Supports all renderable outputs:
  - Cube textures
  - 3D textures
  - Draw buffers (MRT)
  - Render target textures
  - Renderbuffers

**Color-coded command list:**
- **Blue background**: Draw calls / Clear commands
- **Green name**: Valid commands (changing state)
- **Orange name**: Redundant commands (optimization opportunities)
- **Red name**: Deprecated WebGL commands

### Texture Inspection Features

**For each draw call, SpectorJS shows:**
- All bound textures with preview thumbnails
- Texture properties (size, format, mipmaps)
- Sampler uniform values
- Texture coordinate data
- UV mapping visualization

**Multi-channel texture viewing:**
- Individual channel inspection (R, G, B, A)
- Combined RGB view
- Texture atlas visualization
- Mipmap level inspection

### Shader Debugging

From [Real-Time Rendering blog](https://www.realtimerendering.com/blog/debugging-webgl-with-spectorjs/) (accessed 2025-10-31):

**Shader inspection workflow:**
1. Select draw call from command list
2. Navigate to Program information in right panel
3. Click "Click to open" link on shader
4. View beautified source code with:
   - Preprocessor defines resolved
   - Formatted indentation
   - Syntax highlighting

**What you can inspect:**
- Vertex shader source
- Fragment shader source
- Uniform values at draw call time
- Attribute bindings
- Varying declarations

**Limitation**: SpectorJS shows shader _source code_ and _uniform values_, but cannot step through shader execution or inspect per-fragment variables. For that level of debugging, use printf-style debugging (output to texture) or WEBGL_debug_shaders extension.

### Memory and Performance Analysis

**Frame information panel includes:**
- Total draw calls count
- Primitives drawn (triangles, points, lines)
- Texture memory consumption
- Buffer memory usage
- Command frequency statistics

**Use cases:**
- Identify redundant state changes (orange commands)
- Detect unnecessary draw calls
- Find texture memory leaks
- Optimize command batching

### Custom Metadata

From [SpectorJS GitHub README](https://github.com/BabylonJS/Spector.js) (accessed 2025-10-31):

**Annotate WebGL objects for easier debugging:**
```javascript
var cubeVerticesColorBuffer = gl.createBuffer();
cubeVerticesColorBuffer.__SPECTOR_Metadata = {
    name: "cubeVerticesColorBuffer"
};
```

Metadata appears in SpectorJS capture wherever the object is referenced, making it easy to identify buffers, textures, and framebuffers by friendly names.

**ARR-COC application:**
```javascript
relevanceTexture.__SPECTOR_Metadata = {
    name: "RelevanceTexture_13ch",
    description: "Propositional (0-2), Perspectival (3-5), Participatory (6-12)"
};
```

### Capture Sharing

**Export/import workflow:**
1. Navigate to Captures menu in SpectorJS
2. Click floppy disk icon to download JSON
3. Share JSON file with team
4. Drag and drop JSON onto SpectorJS popup to open

**Benefits:**
- Debug customer issues remotely
- Compare before/after engine changes
- Archive problematic frames for regression testing

---

## Section 2: Chrome DevTools WebGL Debugging

From [Stack Overflow - debugging webgl in chrome](https://stackoverflow.com/questions/28588388/debugging-webgl-in-chrome) (accessed 2025-10-31):

### Built-in Chrome DevTools Features

**Performance panel:**
- GPU timeline visualization
- Frame timing analysis
- Identify rendering bottlenecks

**Memory panel:**
- WebGL memory consumption
- Texture allocation tracking
- Detect memory leaks in contexts

**Console API for debugging:**
```javascript
// Enable error checking after every WebGL call
function wrapGLContext(gl) {
    function checkError() {
        const err = gl.getError();
        if (err !== gl.NO_ERROR) {
            console.error('WebGL error:', err);
        }
    }

    for (let key in gl) {
        if (typeof gl[key] === 'function') {
            const original = gl[key];
            gl[key] = function(...args) {
                const result = original.apply(gl, args);
                checkError();
                return result;
            };
        }
    }
}
```

### WEBGL_debug_shaders Extension

From [MDN - WEBGL_debug_shaders](https://developer.mozilla.org/en-US/docs/Web/API/WEBGL_debug_shaders) (accessed 2025-10-31):

**Purpose**: Exposes translated shader source from GLSL to GPU-specific assembly.

**Availability**: Only in **privileged contexts** (browser extensions, DevTools) due to privacy concerns (can reveal GPU hardware details).

```javascript
const ext = gl.getExtension('WEBGL_debug_shaders');
if (ext) {
    const translatedSource = ext.getTranslatedShaderSource(shader);
    console.log(translatedSource);
}
```

**Use cases:**
- Verify shader compilation output
- Debug driver-specific shader issues
- Understand GPU-level optimizations

**Limitation**: Not available to regular web pages (privacy protection).

### Chrome WebGL Inspector Extension

From [GitHub - benvanik/WebGL-Inspector](https://github.com/benvanik/WebGL-Inspector) (accessed 2025-10-31):

**Status**: No longer actively maintained. Use SpectorJS instead.

**Historical context**: WebGL Inspector was the original debugging tool (inspired by PIX/gDEBugger) but lacks WebGL2 support and modern extensions.

### Shader Editor Extension (Deprecated)

From [GitHub - spite/ShaderEditorExtension](https://github.com/spite/ShaderEditorExtension) (accessed 2025-10-31):

**Purpose**: Live shader editing in Chrome DevTools (similar to Firefox Shader Editor).

**Status**: Unmaintained. Modern workflow uses SpectorJS or live reload with source maps.

---

## Section 3: Firefox WebGL Debugging Tools

From [Reddit - r/webgl - What is the best way to debug a webgl program?](https://www.reddit.com/r/webgl/comments/19bemj3/what_is_the_best_way_to_debug_a_webgl_program/) (accessed 2025-10-31):

### Firefox Shader Editor (Legacy)

**Historical tool** (removed in modern Firefox versions):
- Real-time shader editing
- Live preview of shader changes
- Replaced by SpectorJS workflow

### Firefox DevTools Canvas Debugger

**Current status**: Limited WebGL support. Recommended to use SpectorJS extension instead.

---

## Section 4: Texture Inspection Workflows

### Multi-Channel Texture Debugging Strategy

**Challenge**: Inspecting 13-channel textures like ARR-COC's relevance arrays.

**Workflow with SpectorJS:**

1. **Capture frame during texture generation:**
```javascript
spector.setMarker("Generating 13-channel relevance texture");
// ARR-COC texture generation code
spector.clearMarker();
```

2. **Inspect texture in capture:**
   - Navigate to draw call using texture
   - Click texture thumbnail in Uniforms section
   - View individual channels (R, G, B, A)
   - Export texture data as image

3. **Verify channel contents:**
   - Channel 0 (R): Propositional - Statistical info
   - Channel 1 (G): Propositional - Entropy
   - Channel 2 (B): Propositional - Complexity
   - Channels 3-5 (RGB): Perspectival - Salience maps
   - Channels 6-12 (separate textures): Participatory - Query-content coupling

**Limitation**: SpectorJS shows RGBA textures natively. For >4 channels, ARR-COC uses **multiple textures** or **3D texture** with slicing.

### Texture Download and Analysis

**Export texture from SpectorJS:**
1. Right-click texture thumbnail
2. Save as PNG/data URL
3. Analyze in external tools (ImageJ, GIMP, Python)

**Programmatic texture readback:**
```javascript
// During capture, read texture data
const fb = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                         gl.TEXTURE_2D, texture, 0);

const pixels = new Float32Array(width * height * 4);
gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, pixels);
console.log('Texture channel 0 (R):', pixels.filter((_, i) => i % 4 === 0));
```

### Real-Time Texture Visualization

**False color shader for debugging:**
```glsl
// Fragment shader for visualizing specific channel
uniform sampler2D uRelevanceTexture;
uniform int uChannelToView; // 0=R, 1=G, 2=B, 3=A

void main() {
    vec4 texel = texture2D(uRelevanceTexture, vUV);
    float value = texel[uChannelToView];

    // False color: blue=low, red=high
    vec3 color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), value);
    gl_FragColor = vec4(color, 1.0);
}
```

---

## Section 5: ARR-COC Debugging Strategy

### Debugging 13-Channel Relevance Textures

**ARR-COC texture architecture:**
- 32×32 patches (1024 patches total)
- 13 channels per patch (Propositional 3, Perspectival 3, Participatory 7)
- Dynamic token allocation (64-400 tokens per patch)

**Debugging workflow:**

**1. Verify texture generation (knowing.py):**
```javascript
spector.setMarker("ARR-COC: Knowing - Generate 13 channels");
// InformationScorer, SalienceMapper, CouplingAnalyzer
spector.clearMarker();
```

**2. Verify opponent processing (balancing.py):**
```javascript
spector.setMarker("ARR-COC: Balancing - Opponent processing");
// TensionBalancer navigation
spector.clearMarker();
```

**3. Verify token allocation (attending.py):**
```javascript
spector.setMarker("ARR-COC: Attending - Salience to budgets");
// RelevanceAllocator mapping
spector.clearMarker();
```

**4. Verify final compression (realizing.py):**
```javascript
spector.setMarker("ARR-COC: Realizing - Execute compression");
// Pipeline orchestrator
spector.clearMarker();
```

### Inspecting Intermediate Render Targets

**Use SpectorJS visual state panel:**
- Each balancing step renders to intermediate texture
- Visual thumbnails show progression: Raw features → Balanced relevance → Allocated LOD
- Click thumbnail to see exact WebGL commands

**Debugging shader uniforms:**
- Verify `uQueryEmbedding` matches input query
- Check `uCompressLevel` per patch (0.0 = 400 tokens, 1.0 = 64 tokens)
- Inspect `uTensionWeights` from opponent processing

### Memory Profiling

**ARR-COC texture memory calculation:**
```
Base image: 1024×1024×3 (RGB) = 3 MB
Relevance texture: 32×32×13 channels (RGBA textures) = 4 textures
  - 3 RGBA textures (12 channels) + 1 R texture (1 channel)
  - Each 32×32 RGBA float32 = 32KB
  - Total: 128 KB

Compressed features: Variable (64-400 tokens × 1024 patches)
  - Min: 64 × 1024 × 4 bytes = 256 KB
  - Max: 400 × 1024 × 4 bytes = 1.6 MB
```

**SpectorJS memory panel validates these numbers.**

### Common Debugging Scenarios

**Scenario 1: Textures appear black**
- Check SpectorJS uniform values (are samplers bound?)
- Verify texture format matches shader expectations (RGBA vs RGB)
- Inspect framebuffer completeness status

**Scenario 2: Relevance scores seem incorrect**
- Export texture, analyze channel distributions in Python
- Verify opponent processing weights (compress↔particularize)
- Check query-content coupling math

**Scenario 3: Performance issues**
- SpectorJS command list shows redundant state changes (orange)
- Texture rebinding on every draw call
- Unnecessary framebuffer switches

---

## Section 6: Advanced Debugging Techniques

### Printf-Style Shader Debugging

**Technique**: Output debug values to specific pixels in render target.

```glsl
// Debug fragment shader
void main() {
    vec3 relevance = computeRelevance();

    // Top-left corner: output raw channel values
    if (gl_FragCoord.x < 10.0 && gl_FragCoord.y < 10.0) {
        gl_FragColor = vec4(relevance.r, relevance.g, relevance.b, 1.0);
        return;
    }

    // Normal rendering
    gl_FragColor = vec4(finalColor, 1.0);
}
```

Read top-left pixels with `readPixels()` to inspect intermediate calculations.

### Marker-Based Capture Analysis

From [Real-Time Rendering - SpectorJS APIs](https://www.realtimerendering.com/blog/debugging-webgl-with-spectorjs/) (accessed 2025-10-31):

**Use spector.setMarker() to annotate captures:**
```javascript
spector.setMarker("Shadow map rendering");
renderShadowMap();
spector.clearMarker();

spector.setMarker("Main scene rendering");
renderMainScene();
spector.clearMarker();
```

Markers appear in SpectorJS command list, creating visual sections for easier navigation.

### Remote Debugging

**Workflow for debugging production issues:**
1. User experiences visual glitch
2. Trigger programmatic capture:
```javascript
window.debugCapture = function() {
    spector.captureCanvas(canvas);
    spector.onCapture.add((capture) => {
        // Send to server
        fetch('/api/debug-capture', {
            method: 'POST',
            body: JSON.stringify(capture)
        });
    });
};
```
3. Download capture JSON from server
4. Import into local SpectorJS for analysis

---

## ARR-COC Integration Recommendations

### 1. Embed SpectorJS in Development Builds

```javascript
// Only in development mode
if (process.env.NODE_ENV === 'development') {
    const spector = new SPECTOR.Spector();
    spector.displayUI();
    window.spector = spector; // Global access
}
```

### 2. Annotate All Textures with Metadata

```javascript
class ARRCOCTextureManager {
    createRelevanceTexture() {
        const tex = gl.createTexture();
        tex.__SPECTOR_Metadata = {
            name: "RelevanceTexture",
            channels: "R=Statistical, G=Entropy, B=Complexity",
            size: "32x32",
            format: "RGBA32F"
        };
        return tex;
    }
}
```

### 3. Add Debug Markers for Pipeline Stages

```javascript
class RelevanceRealizer {
    realize(image, query) {
        spector.setMarker("KNOWING: 3 ways measurement");
        const scores = this.knowing.measure(image, query);
        spector.clearMarker();

        spector.setMarker("BALANCING: Opponent processing");
        const balanced = this.balancing.navigate(scores);
        spector.clearMarker();

        spector.setMarker("ATTENDING: Relevance to budgets");
        const budgets = this.attending.allocate(balanced);
        spector.clearMarker();

        return budgets;
    }
}
```

### 4. Create Debug Visualization Shaders

```glsl
// Debug shader: visualize specific relevance channel
uniform int uDebugChannel; // 0-12
uniform sampler2D uRelevanceTex1; // Channels 0-3
uniform sampler2D uRelevanceTex2; // Channels 4-7
uniform sampler2D uRelevanceTex3; // Channels 8-11
uniform sampler2D uRelevanceTex4; // Channel 12

void main() {
    vec4 tex1 = texture2D(uRelevanceTex1, vUV);
    vec4 tex2 = texture2D(uRelevanceTex2, vUV);
    vec4 tex3 = texture2D(uRelevanceTex3, vUV);
    float tex4 = texture2D(uRelevanceTex4, vUV).r;

    float value = 0.0;
    if (uDebugChannel < 4) value = tex1[uDebugChannel];
    else if (uDebugChannel < 8) value = tex2[uDebugChannel - 4];
    else if (uDebugChannel < 12) value = tex3[uDebugChannel - 8];
    else value = tex4;

    // False color
    gl_FragColor = vec4(value, 0.5, 1.0 - value, 1.0);
}
```

### 5. Automated Capture Testing

```javascript
// Regression test: capture and compare frames
async function testRelevanceRealization() {
    const capture = await spector.captureCanvas(canvas);

    // Verify expected draw calls
    assert(capture.commands.length > 100);

    // Verify relevance textures exist
    const relevanceTextures = capture.commands.filter(cmd =>
        cmd.name === 'texImage2D' &&
        cmd.metadata?.name?.includes('Relevance')
    );
    assert(relevanceTextures.length === 4); // 4 textures for 13 channels
}
```

---

## Sources

**Web Research:**
- [Debugging WebGL with SpectorJS](https://www.realtimerendering.com/blog/debugging-webgl-with-spectorjs/) - Real-Time Rendering blog (accessed 2025-10-31)
- [GitHub - BabylonJS/Spector.js](https://github.com/BabylonJS/Spector.js) - Official repository (accessed 2025-10-31)
- [MDN - WEBGL_debug_shaders](https://developer.mozilla.org/en-US/docs/Web/API/WEBGL_debug_shaders) - WebGL extension documentation (accessed 2025-10-31)
- [Stack Overflow - debugging webgl in chrome](https://stackoverflow.com/questions/28588388/debugging-webgl-in-chrome) (accessed 2025-10-31)
- [Reddit - r/webgl - debugging WebGL programs](https://www.reddit.com/r/webgl/comments/19bemj3/what_is_the_best_way_to_debug_a_webgl_program/) (accessed 2025-10-31)

**Additional References:**
- Chrome WebGL Inspector: https://github.com/benvanik/WebGL-Inspector (historical reference)
- SpectorJS Chrome Extension: https://chrome.google.com/webstore/detail/spectorjs/denbgaamihkadbghdceggmchnflmhpmk
- SpectorJS Firefox Extension: https://addons.mozilla.org/en-US/firefox/addon/spector-js/

---

**Knowledge Type**: Debugging techniques, tools documentation
**Application Domain**: WebGL development, VLM texture debugging
**ARR-COC Relevance**: Critical for debugging 13-channel relevance textures, verifying Vervaekean pipeline stages, optimizing rendering performance
