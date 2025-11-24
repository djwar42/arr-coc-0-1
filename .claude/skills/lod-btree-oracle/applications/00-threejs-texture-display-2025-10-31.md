# Three.js Texture Visualization & WebGL Display (2025)

**Created**: 2025-10-31
**Focus**: Interactive 3D texture visualization for multi-channel arrays using Three.js and WebGL

## Overview

Three.js provides powerful capabilities for displaying and interacting with multi-channel texture data in real-time 3D environments. This document covers practical approaches for building interactive texture viewers, particularly for ARR-COC's 13-channel texture arrays used in vision-language models.

From [Three.js Official Documentation](https://threejs.org/) (accessed 2025-10-31):
- Three.js r181 is the current release with WebGL and WebGPU support
- Provides DataTexture class for custom texture data
- ShaderMaterial enables custom GLSL shaders for channel compositing
- Built-in tools for texture inspection and interactive 3D rendering

## Three.js Texture Fundamentals

### DataTexture Class

From [Three.js DataTexture Documentation](https://threejs.org/docs/api/en/textures/DataTexture.html) (accessed 2025-10-31):

**Creating custom textures from raw data:**

```javascript
// Create 32x32 texture with RGB data
const width = 32;
const height = 32;
const size = width * height;
const data = new Uint8Array(3 * size);

// Populate with custom data
for (let i = 0; i < size; i++) {
    const stride = i * 3;
    data[stride] = 255;     // R
    data[stride + 1] = 128; // G
    data[stride + 2] = 0;   // B
}

// Create DataTexture
const texture = new THREE.DataTexture(
    data,
    width,
    height,
    THREE.RGBFormat
);
texture.needsUpdate = true;
```

**Key texture formats for multi-channel data:**
- `RGBFormat`: 3 values per texel (Red, Green, Blue)
- `RGBAFormat`: 4 values per texel (includes Alpha channel)
- `RedFormat`: Single channel data
- `RGFormat`: Two channel data

From [Three.js Forum: DataTexture RGB Channels](https://discourse.threejs.org/t/who-can-tell-me-how-to-render-data3dtexture-rgb-in-three-js/63941) (accessed 2025-10-31):
- Use floating point textures for high precision: `THREE.FloatType`
- RedFormat is commonly used for density fields (single channel)
- Multi-channel textures require careful format selection

### Texture Properties

**Essential texture settings:**

```javascript
texture.minFilter = THREE.LinearFilter;
texture.magFilter = THREE.LinearFilter;
texture.wrapS = THREE.ClampToEdgeWrapping;
texture.wrapT = THREE.ClampToEdgeWrapping;
texture.colorSpace = THREE.SRGBColorSpace; // For color textures
```

## ShaderMaterial for Custom Visualization

From [Three.js ShaderMaterial Documentation](https://threejs.org/docs/api/en/materials/ShaderMaterial.html) (accessed 2025-10-31):

ShaderMaterial allows custom GLSL shaders for complete control over texture rendering. Essential for multi-channel visualization and false color modes.

### Basic ShaderMaterial Setup

```javascript
const material = new THREE.ShaderMaterial({
    uniforms: {
        uTexture: { value: dataTexture },
        uChannelSelect: { value: 0 }, // Which channel to display
        uChannelMapping: { value: new THREE.Vector3(0, 1, 2) } // RGB mapping
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform sampler2D uTexture;
        uniform int uChannelSelect;
        uniform vec3 uChannelMapping;
        varying vec2 vUv;

        void main() {
            vec4 texColor = texture2D(uTexture, vUv);

            // Single channel display
            if (uChannelSelect == 0) {
                gl_FragColor = vec4(texColor.r, texColor.r, texColor.r, 1.0);
            }
            // Custom RGB mapping
            else {
                vec3 mapped = vec3(
                    texColor[int(uChannelMapping.x)],
                    texColor[int(uChannelMapping.y)],
                    texColor[int(uChannelMapping.z)]
                );
                gl_FragColor = vec4(mapped, 1.0);
            }
        }
    `
});
```

From [Stack Overflow: Three.js Custom Shader](https://stackoverflow.com/questions/12627422/custom-texture-shader-in-three-js) (accessed 2025-10-31):
- Fragment shaders process each pixel independently
- `texture2D()` samples texture at UV coordinates
- Uniforms allow JavaScript to control shader behavior in real-time

### Multi-Channel Compositing Shader

From [Three.js Forum: Multi-Texture ShaderMaterial](https://discourse.threejs.org/t/pass-multiple-textures-to-a-shadermaterial-from-boxbuffergeometry/13297) (accessed 2025-10-31):

**Handling multiple textures in one shader:**

```javascript
const material = new THREE.ShaderMaterial({
    uniforms: {
        uTexture1: { value: texture1 },
        uTexture2: { value: texture2 },
        uTexture3: { value: texture3 },
        uBlendMode: { value: 0 }
    },
    fragmentShader: `
        uniform sampler2D uTexture1;
        uniform sampler2D uTexture2;
        uniform sampler2D uTexture3;
        uniform int uBlendMode;
        varying vec2 vUv;

        void main() {
            vec4 color1 = texture2D(uTexture1, vUv);
            vec4 color2 = texture2D(uTexture2, vUv);
            vec4 color3 = texture2D(uTexture3, vUv);

            // Composite RGB from three separate channels
            vec3 composite = vec3(color1.r, color2.r, color3.r);
            gl_FragColor = vec4(composite, 1.0);
        }
    `
});
```

**WebGL texture unit limits:**
- Minimum 8 combined texture units per WebGL spec
- iOS devices typically meet this minimum (8 units)
- Desktop browsers often support 16+ texture units

## Interactive 3D Texture Viewer Implementation

### Basic Three.js Scene Setup

```javascript
// Scene, camera, renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// OrbitControls for interaction
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

camera.position.z = 5;
```

### Texture Display Plane

```javascript
// Create plane geometry for texture display
const geometry = new THREE.PlaneGeometry(4, 4, 32, 32);

// Apply ShaderMaterial with texture
const material = new THREE.ShaderMaterial({
    uniforms: {
        uTexture: { value: dataTexture },
        uChannelMode: { value: 0 }
    },
    vertexShader: vertexShaderCode,
    fragmentShader: fragmentShaderCode
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
```

### Interactive Channel Selector UI

From [Codrops: Grid Displacement with RGB Shift](https://tympanus.net/codrops/2024/08/27/grid-displacement-texture-with-rgb-shift-using-three-js-gpgpu-and-shaders/) (accessed 2025-10-31):

**HTML/JavaScript integration:**

```html
<div id="channel-controls">
    <label>Channel Select:</label>
    <select id="channel-dropdown">
        <option value="0">Channel 0 (Red)</option>
        <option value="1">Channel 1 (Green)</option>
        <option value="2">Channel 2 (Blue)</option>
        <!-- ... more channels ... -->
    </select>

    <label>False Color Mode:</label>
    <select id="color-mode">
        <option value="0">Grayscale</option>
        <option value="1">Heatmap</option>
        <option value="2">RGB Composite</option>
    </select>
</div>

<script>
document.getElementById('channel-dropdown').addEventListener('change', (e) => {
    material.uniforms.uChannelMode.value = parseInt(e.target.value);
});

document.getElementById('color-mode').addEventListener('change', (e) => {
    material.uniforms.uColorMode.value = parseInt(e.target.value);
});
</script>
```

### Real-Time Shader Updates

**Updating uniforms from JavaScript:**

```javascript
// Change displayed channel
function setChannel(channelIndex) {
    material.uniforms.uChannelSelect.value = channelIndex;
    // Shader updates automatically on next render
}

// Update RGB channel mapping
function setRGBMapping(r, g, b) {
    material.uniforms.uChannelMapping.value.set(r, g, b);
}

// Apply false color mode
function setFalseColor(mode) {
    material.uniforms.uFalseColorMode.value = mode;
}
```

## False Color Visualization Modes

### GLSL Heatmap Shader

```glsl
// Fragment shader for heatmap visualization
uniform sampler2D uTexture;
uniform int uChannel;
varying vec2 vUv;

vec3 heatmap(float value) {
    // Heatmap: blue -> cyan -> green -> yellow -> red
    vec3 color;
    value = clamp(value, 0.0, 1.0);

    if (value < 0.25) {
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), value * 4.0);
    } else if (value < 0.5) {
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (value - 0.25) * 4.0);
    } else if (value < 0.75) {
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (value - 0.5) * 4.0);
    } else {
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (value - 0.75) * 4.0);
    }

    return color;
}

void main() {
    vec4 texColor = texture2D(uTexture, vUv);
    float channelValue = texColor[uChannel];
    vec3 color = heatmap(channelValue);
    gl_FragColor = vec4(color, 1.0);
}
```

### Edge Detection Shader

```glsl
uniform sampler2D uTexture;
uniform vec2 uResolution;
varying vec2 vUv;

void main() {
    vec2 texel = 1.0 / uResolution;

    // Sobel edge detection
    float tl = texture2D(uTexture, vUv + vec2(-texel.x, texel.y)).r;
    float tm = texture2D(uTexture, vUv + vec2(0.0, texel.y)).r;
    float tr = texture2D(uTexture, vUv + vec2(texel.x, texel.y)).r;
    float ml = texture2D(uTexture, vUv + vec2(-texel.x, 0.0)).r;
    float mr = texture2D(uTexture, vUv + vec2(texel.x, 0.0)).r;
    float bl = texture2D(uTexture, vUv + vec2(-texel.x, -texel.y)).r;
    float bm = texture2D(uTexture, vUv + vec2(0.0, -texel.y)).r;
    float br = texture2D(uTexture, vUv + vec2(texel.x, -texel.y)).r;

    float gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br;
    float gy = tl + 2.0*tm + tr - bl - 2.0*bm - br;
    float edge = sqrt(gx*gx + gy*gy);

    gl_FragColor = vec4(vec3(edge), 1.0);
}
```

## ARR-COC Implementation Guide

### Displaying 13-Channel Texture Arrays

**Architecture for ARR-COC's multi-channel textures:**

```javascript
// ARR-COC generates 13-channel texture (32x32 patches)
// Channels: [semantic features (8), edges (2), spatial (3)]

class ARRCOCTextureViewer {
    constructor(canvas) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.controls = new OrbitControls(this.camera, canvas);

        this.channelTextures = []; // Store 13 separate textures
        this.currentChannel = 0;
        this.viewMode = 'single'; // 'single', 'rgb-composite', 'heatmap'

        this.setupScene();
    }

    loadTexture(channelData, channelIndex, width, height) {
        // channelData: Float32Array or Uint8Array
        const texture = new THREE.DataTexture(
            channelData,
            width,
            height,
            THREE.RedFormat,
            THREE.FloatType
        );
        texture.needsUpdate = true;
        this.channelTextures[channelIndex] = texture;
    }

    setChannelMapping(r, g, b) {
        // Map 3 channels to RGB for visualization
        this.material.uniforms.uTexture1.value = this.channelTextures[r];
        this.material.uniforms.uTexture2.value = this.channelTextures[g];
        this.material.uniforms.uTexture3.value = this.channelTextures[b];
    }

    setupScene() {
        // Create 32x32 grid of patches as 3D plane
        const geometry = new THREE.PlaneGeometry(10, 10, 32, 32);

        this.material = new THREE.ShaderMaterial({
            uniforms: {
                uTexture1: { value: null },
                uTexture2: { value: null },
                uTexture3: { value: null },
                uViewMode: { value: 0 },
                uHeatmapRange: { value: new THREE.Vector2(0.0, 1.0) }
            },
            vertexShader: this.getVertexShader(),
            fragmentShader: this.getFragmentShader()
        });

        const mesh = new THREE.Mesh(geometry, this.material);
        this.scene.add(mesh);

        this.camera.position.z = 15;
        this.animate();
    }

    getFragmentShader() {
        return `
            uniform sampler2D uTexture1;
            uniform sampler2D uTexture2;
            uniform sampler2D uTexture3;
            uniform int uViewMode;
            uniform vec2 uHeatmapRange;
            varying vec2 vUv;

            vec3 heatmap(float value) {
                // Implementation from above
                value = (value - uHeatmapRange.x) / (uHeatmapRange.y - uHeatmapRange.x);
                // ... heatmap color logic
                return vec3(value); // Simplified
            }

            void main() {
                vec3 color;

                if (uViewMode == 0) {
                    // Single channel grayscale
                    float val = texture2D(uTexture1, vUv).r;
                    color = vec3(val);
                }
                else if (uViewMode == 1) {
                    // RGB composite from 3 channels
                    float r = texture2D(uTexture1, vUv).r;
                    float g = texture2D(uTexture2, vUv).r;
                    float b = texture2D(uTexture3, vUv).r;
                    color = vec3(r, g, b);
                }
                else if (uViewMode == 2) {
                    // Heatmap
                    float val = texture2D(uTexture1, vUv).r;
                    color = heatmap(val);
                }

                gl_FragColor = vec4(color, 1.0);
            }
        `;
    }

    getVertexShader() {
        return `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Usage:
const canvas = document.getElementById('texture-canvas');
const viewer = new ARRCOCTextureViewer(canvas);

// Load 13 channels
for (let i = 0; i < 13; i++) {
    const channelData = getChannelDataFromARRCOC(i); // Your data source
    viewer.loadTexture(channelData, i, 32, 32);
}

// Display semantic channels as RGB
viewer.setChannelMapping(0, 1, 2); // Channels 0, 1, 2 → R, G, B
```

### Interactive Patch Inspector

**Click-to-inspect individual patches:**

```javascript
// Raycasting for patch selection
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

canvas.addEventListener('click', (event) => {
    // Convert mouse coordinates to normalized device coordinates
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, viewer.camera);
    const intersects = raycaster.intersectObject(viewer.mesh);

    if (intersects.length > 0) {
        const uv = intersects[0].uv;
        const patchX = Math.floor(uv.x * 32);
        const patchY = Math.floor(uv.y * 32);

        // Display all 13 channels for this patch
        showPatchDetails(patchX, patchY);
    }
});

function showPatchDetails(x, y) {
    const detailsDiv = document.getElementById('patch-details');
    let html = `<h3>Patch (${x}, ${y})</h3>`;

    for (let ch = 0; ch < 13; ch++) {
        const value = getChannelValue(x, y, ch);
        html += `<div>Channel ${ch}: ${value.toFixed(3)}</div>`;
    }

    detailsDiv.innerHTML = html;
}
```

## Development Tools & Debugging

From [Three.js DevTools Chrome Extension](https://chromewebstore.google.com/detail/threejs-devtools/jechbjkglifdaldbdbigibihfaclnkbo) (accessed 2025-10-31):

**Browser debugging tools:**
- Three.js DevTools: Inspect object hierarchies, materials, textures
- SpectorJS: WebGL capture and frame analysis
- Chrome DevTools: WebGL shader debugging

From [Three.js Forum: THREE.RenderTargetInspector](https://discourse.threejs.org/t/three-rendertargetinspector/13659) (accessed 2025-10-31):

**Texture inspection helper:**
```javascript
import { RenderTargetInspector } from 'three-rendertarget-inspector';

const inspector = new RenderTargetInspector();
inspector.addRenderTarget(dataTexture);
// Displays thumbnail and viewer window for texture contents
```

## Performance Optimization

From [Three.js Forum: DataTexture Performance](https://discourse.threejs.org/t/creating-datatexture-showing-weird-warnings-inside-console/74205) (accessed 2025-10-31):

**Best practices for large texture arrays:**

1. **Texture format selection:**
   - Use `FloatType` only when necessary (high precision)
   - `UnsignedByteType` is faster for 8-bit data
   - `HalfFloatType` balances precision and performance

2. **Update strategy:**
   ```javascript
   // Avoid recreating textures
   texture.needsUpdate = true; // Triggers GPU upload

   // Partial updates for large textures
   renderer.copyTextureToTexture(position, srcTexture, destTexture);
   ```

3. **Texture pooling:**
   ```javascript
   // Reuse texture objects
   const texturePool = [];
   function getTexture(width, height) {
       const texture = texturePool.pop() || new THREE.DataTexture(null, width, height);
       return texture;
   }
   ```

## Integration with Gradio (Python Backend)

**Data transfer pattern for ARR-COC:**

```python
# Python (Gradio backend)
import numpy as np
import base64

def generate_texture_data(image):
    # ARR-COC processing
    texture_array = model.generate_13_channel_texture(image)  # Shape: (13, 32, 32)

    # Convert to JSON-serializable format
    texture_data = {}
    for ch in range(13):
        channel_bytes = texture_array[ch].tobytes()
        texture_data[f'channel_{ch}'] = base64.b64encode(channel_bytes).decode('utf-8')

    return texture_data

# Gradio interface
import gradio as gr

def visualize(image):
    texture_data = generate_texture_data(image)

    # Custom HTML with Three.js viewer
    html = f"""
    <div id="viewer-container"></div>
    <script type="module">
        import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.181.0/build/three.module.js';

        const textureData = {texture_data};
        // ... Three.js viewer setup with loaded data
    </script>
    """

    return html

iface = gr.Interface(
    fn=visualize,
    inputs=gr.Image(),
    outputs=gr.HTML()
)
```

**JavaScript data decoding:**

```javascript
function decodeTextureChannel(base64String, width, height) {
    const binaryString = atob(base64String);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Convert to Float32Array if needed
    const floats = new Float32Array(bytes.buffer);

    return new THREE.DataTexture(
        floats,
        width,
        height,
        THREE.RedFormat,
        THREE.FloatType
    );
}
```

## Example: Complete ARR-COC 3D Texture Viewer

**Full implementation template:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>ARR-COC Texture Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; }
        #canvas { width: 100vw; height: 100vh; }
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 8px;
        }
        #patch-info {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>

    <div id="controls">
        <h3>ARR-COC Texture Viewer</h3>

        <label>View Mode:</label>
        <select id="view-mode">
            <option value="single">Single Channel</option>
            <option value="rgb">RGB Composite</option>
            <option value="heatmap">Heatmap</option>
        </select>

        <div id="single-controls">
            <label>Channel:</label>
            <select id="channel-select">
                <!-- Semantic channels 0-7 -->
                <option value="0">Semantic 0</option>
                <option value="1">Semantic 1</option>
                <option value="2">Semantic 2</option>
                <option value="3">Semantic 3</option>
                <option value="4">Semantic 4</option>
                <option value="5">Semantic 5</option>
                <option value="6">Semantic 6</option>
                <option value="7">Semantic 7</option>
                <!-- Edge channels 8-9 -->
                <option value="8">Edge X</option>
                <option value="9">Edge Y</option>
                <!-- Spatial channels 10-12 -->
                <option value="10">Spatial X</option>
                <option value="11">Spatial Y</option>
                <option value="12">Spatial Z</option>
            </select>
        </div>

        <div id="rgb-controls" style="display:none;">
            <label>R:</label><select id="r-channel"></select>
            <label>G:</label><select id="g-channel"></select>
            <label>B:</label><select id="b-channel"></select>
        </div>
    </div>

    <div id="patch-info">
        Click a patch for details
    </div>

    <script type="module">
        import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.181.0/build/three.module.js';
        import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.181.0/examples/jsm/controls/OrbitControls.js';

        // Initialize viewer
        const viewer = new ARRCOCTextureViewer(document.getElementById('canvas'));

        // Load texture data (from Gradio backend or local source)
        loadTextureData().then(data => {
            for (let i = 0; i < 13; i++) {
                viewer.loadTexture(data[i], i, 32, 32);
            }
            viewer.setChannelMapping(0, 1, 2);
        });

        // UI event listeners
        document.getElementById('view-mode').addEventListener('change', (e) => {
            viewer.setViewMode(e.target.value);
        });

        document.getElementById('channel-select').addEventListener('change', (e) => {
            viewer.setChannel(parseInt(e.target.value));
        });
    </script>
</body>
</html>
```

## Sources

**Official Documentation:**
- [Three.js Official Site](https://threejs.org/) - Main library documentation and examples
- [Three.js DataTexture Docs](https://threejs.org/docs/api/en/textures/DataTexture.html) - DataTexture API reference
- [Three.js ShaderMaterial Docs](https://threejs.org/docs/api/en/materials/ShaderMaterial.html) - Custom shader materials

**Web Research (accessed 2025-10-31):**
- [Three.js DevTools Chrome Extension](https://chromewebstore.google.com/detail/threejs-devtools/jechbjkglifdaldbdbigibihfaclnkbo) - Browser debugging tools
- [Codrops: Grid Displacement with RGB Shift](https://tympanus.net/codrops/2024/08/27/grid-displacement-texture-with-rgb-shift-using-three-js-gpgpu-and-shaders/) - Advanced shader techniques
- [DEV.to: Creating Custom Shaders in Three.js](https://dev.to/maniflames/creating-a-custom-shader-in-threejs-3bhi) - ShaderMaterial tutorial

**Community Forums (accessed 2025-10-31):**
- [Three.js Forum: DataTexture RGB Channels](https://discourse.threejs.org/t/who-can-tell-me-how-to-render-data3dtexture-rgb-in-three-js/63941) - Multi-channel rendering
- [Three.js Forum: Multi-Texture ShaderMaterial](https://discourse.threejs.org/t/pass-multiple-textures-to-a-shadermaterial-from-boxbuffergeometry/13297) - Multiple texture handling
- [Three.js Forum: RenderTargetInspector](https://discourse.threejs.org/t/three-rendertargetinspector/13659) - Texture inspection tools

**Stack Overflow (accessed 2025-10-31):**
- [Custom Texture Shader in Three.js](https://stackoverflow.com/questions/12627422/custom-texture-shader-in-three-js) - Basic shader patterns
- [ThreeJS Multiple Texture with ShaderMaterial](https://stackoverflow.com/questions/71575467/threejs-multiple-texture-with-shadermaterial-problem-on-ios) - iOS WebGL texture limits

**GitHub Resources:**
- GitHub repositories for Three.js texture viewers (search results included model viewers and crystallographic visualization tools)

**Additional References:**
- [Observable: 3D Volume Rendering with WebGL](https://observablehq.com/@mroehlig/3d-volume-rendering-with-webgl-three-js) - 3D texture techniques
- [Medium: Three.js WebGPURenderer Part 1](https://medium.com/@christianhelgeson/three-js-webgpurenderer-part-1-fragment-vertex-shaders-1070063447f0) - Modern shader approaches
