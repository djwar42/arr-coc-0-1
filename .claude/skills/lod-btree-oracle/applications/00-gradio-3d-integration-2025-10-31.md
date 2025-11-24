# Gradio + 3D Visualization Integration (2025)

## Overview

Gradio provides multiple approaches for integrating 3D visualizations (Three.js, Babylon.js, WebGL) into machine learning applications. This guide covers custom components, HTML/JavaScript integration, and data transfer patterns specifically relevant for ARR-COC texture visualization.

**Key Integration Methods:**
1. **gr.HTML Component** - Embed custom HTML/JavaScript (simplest, limited interactivity)
2. **Custom Components** - Full-featured components with Python ↔ JavaScript communication
3. **gr.Model3D** - Built-in 3D model viewer (for .obj, .glb, .gltf files)

---

## Section 1: Gradio Custom Components Architecture

From [Gradio Custom Components Guide](https://www.gradio.app/guides/custom-components-in-five-minutes) (accessed 2025-10-31):

### The Custom Components Workflow

**Four-Step Process:**
```bash
# 1. Create template
gradio cc create MyComponent --template SimpleTextbox

# 2. Develop with hot reload
gradio cc dev

# 3. Build distributable package
gradio cc build

# 4. Publish to PyPI
gradio cc publish
```

**Project Structure:**
```
my-component/
├── backend/      # Python code (data processing)
├── frontend/     # JavaScript/Svelte code (UI)
├── demo/         # Sample app for testing
└── pyproject.toml # Package metadata
```

**Requirements (as of 2025):**
- Python 3.10+
- Node.js 20+
- npm 9+
- Gradio 5+

### Why Custom Components for 3D?

**Advantages over gr.HTML:**
- Bidirectional data flow (Python ↔ JavaScript)
- Type-safe interfaces
- Event listeners and callbacks
- State management
- Reusable across projects

**Example Use Case (ARR-COC):**
- **Backend (Python)**: Generate 13-channel texture arrays, compute relevance scores
- **Frontend (JavaScript/Three.js)**: Interactive 3D texture viewer with channel selector
- **Communication**: NumPy arrays → JSON → Three.js DataTexture

---

## Section 2: gr.HTML for Embedding 3D Visualizations

From [Gradio Custom CSS and JS Guide](https://www.gradio.app/guides/custom-CSS-and-JS) (accessed 2025-10-31):

### Basic HTML Component Usage

The `gr.HTML` component displays arbitrary HTML, including `<canvas>` elements for WebGL:

```python
import gradio as gr

# Simple example
with gr.Blocks() as demo:
    gr.HTML("<p>This <em>example</em> uses <strong>HTML</strong></p>")
```

**Important Limitations:**
- Only static HTML is rendered
- **No JavaScript execution in gr.HTML value** (security restriction)
- For JavaScript, must use `js=` parameter or `head=` parameter

### Adding Three.js via head Parameter

From [Gradio HTML Component Docs](https://www.gradio.app/docs/gradio/html) (accessed 2025-10-31):

```python
import gradio as gr

# Include Three.js in head
threejs_head = """
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
<script>
    function initThreeJS() {
        const container = document.getElementById('threejs-container');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 800/600, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({canvas: container});

        // Add 3D objects here
        const geometry = new THREE.BoxGeometry();
        const material = new THREE.MeshBasicMaterial({color: 0x00ff00});
        const cube = new THREE.Mesh(geometry, material);
        scene.add(cube);

        camera.position.z = 5;

        function animate() {
            requestAnimationFrame(animate);
            cube.rotation.x += 0.01;
            cube.rotation.y += 0.01;
            renderer.render(scene, camera);
        }
        animate();
    }

    // Wait for DOM to load
    document.addEventListener('DOMContentLoaded', initThreeJS);
</script>
"""

with gr.Blocks(head=threejs_head) as demo:
    gr.HTML('<canvas id="threejs-container" width="800" height="600"></canvas>')

demo.launch()
```

**Key Points:**
- `head` parameter adds to HTML `<head>` (runs on page load)
- `js` parameter on events runs JavaScript on user interaction
- Use `elem_id` to target specific components from JavaScript

---

## Section 3: Data Transfer Patterns (Python ↔ JavaScript)

### Pattern 1: Base64 Encoding (Small Data)

```python
import gradio as gr
import numpy as np
import base64
import json

def generate_texture_data():
    # Generate texture (e.g., 32x32x3 RGB)
    texture = np.random.rand(32, 32, 3).astype(np.float32)

    # Convert to base64
    texture_bytes = texture.tobytes()
    texture_b64 = base64.b64encode(texture_bytes).decode('utf-8')

    # Return metadata + data
    return json.dumps({
        'width': 32,
        'height': 32,
        'channels': 3,
        'data': texture_b64
    })

with gr.Blocks() as demo:
    btn = gr.Button("Generate")
    output = gr.Textbox()
    btn.click(generate_texture_data, None, output)
```

**JavaScript side (in custom component or js parameter):**
```javascript
function loadTexture(dataJson) {
    const data = JSON.parse(dataJson);
    const bytes = Uint8Array.from(atob(data.data), c => c.charCodeAt(0));
    const floats = new Float32Array(bytes.buffer);

    // Create Three.js DataTexture
    const texture = new THREE.DataTexture(
        floats,
        data.width,
        data.height,
        THREE.RGBFormat,
        THREE.FloatType
    );
    texture.needsUpdate = true;
    return texture;
}
```

### Pattern 2: File-Based Transfer (Large Data)

For ARR-COC's 13-channel textures (potentially large):

```python
import gradio as gr
import numpy as np
from pathlib import Path

def generate_and_save_texture():
    # Generate 13-channel texture
    texture = np.random.rand(256, 256, 13).astype(np.float32)

    # Save as binary file
    filepath = Path("temp_texture.bin")
    texture.tofile(filepath)

    # Return file path (Gradio serves files in /gradio_api/file=)
    return str(filepath), 256, 256, 13

with gr.Blocks() as demo:
    btn = gr.Button("Generate")
    file_output = gr.File()
    width = gr.Number()
    height = gr.Number()
    channels = gr.Number()

    btn.click(
        generate_and_save_texture,
        None,
        [file_output, width, height, channels]
    )
```

**Accessing from JavaScript:**
```javascript
// In custom component or via js parameter
fetch('/gradio_api/file=temp_texture.bin')
    .then(response => response.arrayBuffer())
    .then(buffer => {
        const floats = new Float32Array(buffer);
        // Load into Three.js texture
    });
```

### Pattern 3: Real-Time Updates via Events

From [Gradio Event Listeners](https://www.gradio.app/docs/gradio/html#event-listeners) (accessed 2025-10-31):

```python
import gradio as gr

def update_channel(channel_idx):
    # Generate data for specific channel
    return f"Channel {channel_idx} data"

with gr.Blocks() as demo:
    channel_slider = gr.Slider(0, 12, step=1, label="Channel")
    display = gr.HTML()

    # Update on change
    channel_slider.change(
        update_channel,
        inputs=channel_slider,
        outputs=display
    )
```

---

## Section 4: Three.js Integration Patterns

### Approach A: Inline Three.js in head

**Pros:**
- Simple setup
- Works for static visualizations
- No build step

**Cons:**
- Limited interactivity with Python backend
- Hard to debug
- No hot reload during development

**Best For:** Proof of concept, static 3D models

### Approach B: Custom Component with Three.js

**Template Structure:**
```
gradio-threejs-viewer/
├── backend/
│   └── __init__.py  # Python interface
├── frontend/
│   ├── Index.svelte  # Svelte wrapper
│   ├── ThreeScene.js # Three.js logic
│   └── package.json  # Dependencies
└── demo/
    └── app.py
```

**frontend/package.json:**
```json
{
  "dependencies": {
    "three": "^0.158.0",
    "@gradio/client": "^0.10.0"
  }
}
```

**frontend/Index.svelte:**
```svelte
<script>
  import { onMount } from 'svelte';
  import * as THREE from 'three';

  export let value = null; // Data from Python
  export let elem_id = "";

  let container;
  let scene, camera, renderer;

  onMount(() => {
    // Initialize Three.js
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, 800/600, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({canvas: container});

    // Render loop
    function animate() {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();
  });

  // React to value changes from Python
  $: if (value) {
    updateTexture(value);
  }

  function updateTexture(data) {
    // Update Three.js scene with new data
  }
</script>

<canvas bind:this={container} id={elem_id} width="800" height="600"></canvas>
```

**backend/__init__.py:**
```python
from gradio.components import Component

class ThreeJSViewer(Component):
    def __init__(self, value=None, **kwargs):
        super().__init__(value=value, **kwargs)

    def preprocess(self, x):
        return x

    def postprocess(self, y):
        # Convert NumPy to JSON-serializable format
        if isinstance(y, np.ndarray):
            return {
                'shape': y.shape,
                'data': y.tolist()  # or base64 for efficiency
            }
        return y
```

**Pros:**
- Full control over rendering
- Type-safe Python ↔ JavaScript interface
- Hot reload during development
- Reusable package

**Cons:**
- More complex setup
- Requires Node.js/npm
- Longer development time

**Best For:** Production applications, reusable components

### Approach C: Use gr.Model3D (Built-in)

Gradio includes a `gr.Model3D` component for viewing 3D models:

```python
import gradio as gr

with gr.Blocks() as demo:
    model = gr.Model3D(
        value="path/to/model.glb",
        label="3D Model Viewer"
    )
```

**Supported Formats:** .obj, .glb, .stl, .gltf, .splat, .ply

**Limitations:**
- Fixed camera controls (cannot customize easily)
- No access to texture data programmatically
- Designed for viewing models, not raw textures

**Best For:** Displaying 3D meshes, not texture arrays

Reference: [How To Use 3D Model Component](https://www.gradio.app/guides/how-to-use-3D-model-component) (accessed 2025-10-31)

---

## Section 5: ARR-COC Gradio + Three.js Blueprint

### Architecture: Hybrid Approach

**Recommendation:** Start with gr.HTML + Three.js in head (Phase 1), migrate to custom component if needed (Phase 2).

### Phase 1: Proof of Concept (gr.HTML + Three.js)

**File: `arr_coc_texture_viewer.py`**

```python
import gradio as gr
import numpy as np
import base64
import json

# Three.js initialization in head
threejs_head = """
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
<script>
let scene, camera, renderer, textureMesh;

function initThreeJS() {
    const container = document.getElementById('texture-canvas');
    scene = new THREE.Scene();
    camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 1;

    renderer = new THREE.WebGLRenderer({
        canvas: container,
        antialias: true
    });
    renderer.setSize(800, 800);

    // Create plane for texture
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.MeshBasicMaterial({color: 0xffffff});
    textureMesh = new THREE.Mesh(geometry, material);
    scene.add(textureMesh);

    renderer.render(scene, camera);
}

function updateTexture(dataJson) {
    const data = JSON.parse(dataJson);
    const width = data.width;
    const height = data.height;

    // Decode base64 texture data
    const bytes = Uint8Array.from(atob(data.texture), c => c.charCodeAt(0));
    const floats = new Float32Array(bytes.buffer);

    // Create DataTexture (assuming RGB from 3 selected channels)
    const texture = new THREE.DataTexture(
        floats,
        width,
        height,
        THREE.RGBFormat,
        THREE.FloatType
    );
    texture.needsUpdate = true;

    // Update material
    textureMesh.material.map = texture;
    textureMesh.material.needsUpdate = true;

    renderer.render(scene, camera);
}

document.addEventListener('DOMContentLoaded', initThreeJS);
</script>
"""

def generate_texture(channel_r, channel_g, channel_b):
    """
    Simulate ARR-COC texture generation.
    In real implementation, this would call ARR-COC model.
    """
    # Generate 13-channel texture (32x32 for demo)
    full_texture = np.random.rand(32, 32, 13).astype(np.float32)

    # Extract selected channels for RGB display
    rgb_texture = np.stack([
        full_texture[:, :, channel_r],
        full_texture[:, :, channel_g],
        full_texture[:, :, channel_b]
    ], axis=2)

    # Convert to base64
    texture_bytes = rgb_texture.tobytes()
    texture_b64 = base64.b64encode(texture_bytes).decode('utf-8')

    return json.dumps({
        'width': 32,
        'height': 32,
        'texture': texture_b64
    })

# Gradio app
with gr.Blocks(head=threejs_head, css=".gradio-container {max-width: 1200px}") as demo:
    gr.Markdown("# ARR-COC Texture Viewer (Three.js)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Channel Selection")
            channel_r = gr.Slider(0, 12, value=0, step=1, label="Red Channel")
            channel_g = gr.Slider(0, 12, value=1, step=1, label="Green Channel")
            channel_b = gr.Slider(0, 12, value=2, step=1, label="Blue Channel")

            btn = gr.Button("Generate Texture")

        with gr.Column(scale=2):
            gr.Markdown("## 3D Texture Display")
            gr.HTML('<canvas id="texture-canvas" width="800" height="800"></canvas>')

    # Hidden textbox to pass data to JavaScript
    texture_data = gr.Textbox(visible=False, elem_id="texture-data")

    # Generate texture and update via JavaScript
    btn.click(
        generate_texture,
        inputs=[channel_r, channel_g, channel_b],
        outputs=texture_data,
        js="(r, g, b) => { return r; }"  # Placeholder
    )

    # Trigger JavaScript update on texture data change
    texture_data.change(
        None,
        texture_data,
        None,
        js="(data) => { updateTexture(data); return data; }"
    )

demo.launch()
```

**Key Features:**
- Channel selection UI (sliders for R/G/B mapping)
- Three.js canvas embedded via gr.HTML
- Data transfer via hidden gr.Textbox + base64
- JavaScript function `updateTexture()` called on data change

**Limitations:**
- One-way data flow (Python → JavaScript)
- Cannot capture JavaScript events back to Python easily
- Hidden textbox is a workaround

### Phase 2: Production Custom Component

**When to migrate:**
- Need bidirectional communication (e.g., user clicks on texture patch → Python processes)
- Want reusable component across projects
- Performance optimization needed (streaming data)

**Migration Path:**
```bash
# Create custom component
gradio cc create ARRCOCTextureViewer --template SimpleImage

# Move Three.js logic to frontend/Index.svelte
# Implement preprocess/postprocess in backend/__init__.py
# Add proper TypeScript types
```

**Custom Component Benefits:**
- Events: `viewer.select(fn, inputs, outputs)` for patch selection
- Streaming: Update textures incrementally
- State: Maintain camera position, zoom level
- Type Safety: Automatic validation of data shapes

---

## Section 6: Common Pitfalls and Solutions

### Issue 1: "Cannot load script" in gr.HTML

**Problem:** Scripts in `gr.HTML(value="<script>...</script>")` don't execute.

**Solution:** Use `head=` parameter in `gr.Blocks()` instead:
```python
# ❌ Won't work
gr.HTML("<script>console.log('test');</script>")

# ✅ Works
with gr.Blocks(head="<script>console.log('test');</script>") as demo:
    pass
```

Reference: [GitHub Issue #3446](https://github.com/gradio-app/gradio/issues/3446)

### Issue 2: CORS Errors with External Resources

**Problem:** Loading textures from filesystem fails with CORS errors.

**Solution:** Use `/gradio_api/file=` prefix:
```python
# Save texture
texture.tofile("texture.bin")

# Access from JavaScript
fetch('/gradio_api/file=texture.bin')
```

**Note:** File must be in allowed paths (same directory as app by default).

### Issue 3: Large Data Transfer Performance

**Problem:** Base64 encoding large textures (e.g., 256×256×13) is slow.

**Solution:** Use file-based transfer or custom component with streaming:
```python
# Instead of base64 in JSON
texture_b64 = base64.b64encode(texture.tobytes())  # Slow for large data

# Use temporary file
with open("temp.bin", "wb") as f:
    texture.tofile(f)
return gr.File("temp.bin")
```

### Issue 4: WebGL Context Limit

**Problem:** Creating multiple Three.js renderers hits WebGL context limit (browser limit: ~8-16).

**Solution:** Reuse renderer, update scene:
```javascript
// ❌ Don't create new renderer every time
function update(data) {
    renderer = new THREE.WebGLRenderer();  // Bad!
}

// ✅ Reuse renderer
let renderer;  // Global
function init() {
    renderer = new THREE.WebGLRenderer();
}
function update(data) {
    // Just update scene, reuse renderer
    scene.remove(oldMesh);
    scene.add(newMesh);
    renderer.render(scene, camera);
}
```

---

## Section 7: Testing and Debugging

### Debug Three.js in Gradio

**Enable Browser DevTools:**
```python
# Launch with debug flag
demo.launch(debug=True)
```

**Console Logging from JavaScript:**
```javascript
console.log("Three.js scene:", scene);
console.log("Texture data:", texture);
```

**Inspect WebGL State:**
- Chrome DevTools → More Tools → WebGL Inspector
- Check for context loss: `renderer.getContext().isContextLost()`

### Test Data Transfer

```python
def test_data_transfer():
    # Generate test texture
    texture = np.ones((32, 32, 3), dtype=np.float32)

    # Encode
    b64 = base64.b64encode(texture.tobytes()).decode('utf-8')

    # Decode (verify)
    decoded = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    decoded = decoded.reshape((32, 32, 3))

    assert np.allclose(texture, decoded), "Data transfer failed"
    print("✅ Data transfer test passed")

test_data_transfer()
```

---

## Section 8: Performance Optimization

### Optimize Texture Uploads

**Use Compressed Formats:**
```javascript
// Instead of Float32Array (4 bytes/channel)
const texture = new THREE.DataTexture(
    new Uint8Array(data),  // 1 byte/channel
    width, height,
    THREE.RGBFormat,
    THREE.UnsignedByteType  // Normalize 0-255 to 0.0-1.0
);
```

**For ARR-COC:** If 32-bit precision not needed, quantize to 8-bit → 4x smaller.

### Lazy Loading

```javascript
// Load base texture immediately
loadBaseTexture();

// Load high-res details on demand
canvas.addEventListener('click', (e) => {
    const patch = getPatchAtClick(e);
    loadPatchDetails(patch);
});
```

### Web Workers for Encoding

```javascript
// Offload base64 decode to worker
const worker = new Worker('decode-worker.js');
worker.postMessage({data: base64String});
worker.onmessage = (e) => {
    const floats = e.data;
    createTexture(floats);
};
```

---

## Section 9: Alternative: Gradio-Lite (No Server)

From research: [Gradio-Lite: Serverless Gradio](https://medium.com/data-science-collective/gradio-lite-serverless-gradio-in-your-browser-df36d70c25b9) (accessed 2025-10-31)

**Gradio-Lite** runs Gradio apps entirely in the browser using Pyodide (Python in WebAssembly).

**Pros:**
- No Python server required
- Fast iteration (changes refresh instantly)
- Can embed in static sites

**Cons:**
- Limited library support (only packages with WebAssembly builds)
- Cannot use native extensions (e.g., PyTorch, NumPy with MKL)
- Not suitable for ARR-COC (requires GPU inference)

**Use Case:** Static demos where texture data is pre-generated.

---

## Section 10: Summary and Recommendations

### Quick Reference Matrix

| Approach | Complexity | Interactivity | Data Transfer | Best For |
|----------|-----------|---------------|---------------|----------|
| gr.HTML + head | Low | Limited | One-way (Py→JS) | Prototypes |
| Custom Component | High | Full | Bidirectional | Production |
| gr.Model3D | Low | Medium | File-based | 3D meshes |
| Gradio-Lite | Medium | Medium | Client-only | Static demos |

### ARR-COC Recommendations

**Phase 1 (MVP - 1-2 days):**
- Use `gr.HTML` + Three.js in `head` parameter
- Base64 encode selected 3-channel composites (RGB from 13 channels)
- Simple channel selector UI (sliders)
- Display as 2D texture on plane geometry

**Phase 2 (Interactive - 3-5 days):**
- Migrate to custom Gradio component
- Add click events (patch selection → show 13-channel breakdown)
- Implement camera controls (pan, zoom)
- Display texture as 32×32 grid with hover tooltips

**Phase 3 (Advanced - 5-7 days):**
- 3D material preview (texture applied to sphere, like Unity's material ball)
- Real-time false color modes (shader-based)
- Integration with SpectorJS for WebGL debugging
- Performance optimization (texture streaming, LOD)

### Code Templates Available

**Minimal Example:** See Phase 1 blueprint (Section 5)

**Custom Component Template:**
```bash
git clone https://github.com/gradio-app/gradio
cd gradio/js/model3d  # Study official 3D component
```

**Three.js + Gradio Examples:**
- Search GitHub: "Gradio Three.js" (limited results, mostly theory)
- Babylon.js Playground: [babylonjs.com/playground](https://www.babylonjs.com/playground/) (test shaders)

---

## Sources

**Gradio Official Documentation:**
- [Custom Components in Five Minutes](https://www.gradio.app/guides/custom-components-in-five-minutes) (accessed 2025-10-31)
- [Custom CSS and JS](https://www.gradio.app/guides/custom-CSS-and-JS) (accessed 2025-10-31)
- [HTML Component Docs](https://www.gradio.app/docs/gradio/html) (accessed 2025-10-31)
- [How To Use 3D Model Component](https://www.gradio.app/guides/how-to-use-3D-model-component) (accessed 2025-10-31)
- [Getting Started With The JS Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (accessed 2025-10-31)

**Web Research:**
- [Gradio-Lite: Serverless Gradio in your browser](https://medium.com/data-science-collective/gradio-lite-serverless-gradio-in-your-browser-df36d70c25b9) - Medium article on browser-based Gradio (accessed 2025-10-31)
- [GitHub gradio-app/gradio](https://github.com/gradio-app/gradio) - Official repository with custom component examples (accessed 2025-10-31)
- GitHub search results for "Gradio Three.js example" and "Gradio WebGL custom component" - Limited practical examples found (accessed 2025-10-31)

**Community Resources:**
- [Gradio Custom Components Gallery](https://www.gradio.app/custom-components/gallery) - 175+ community components (accessed 2025-10-31)
- [HuggingFace Gradio Forum](https://discuss.huggingface.co) - Discussions on HTML/JavaScript integration (accessed 2025-10-31)

**Three.js Resources (for implementation):**
- [Three.js Documentation](https://threejs.org/docs/) - Core API reference
- [Three.js Examples](https://threejs.org/examples/) - DataTexture and custom materials

**Technical References:**
- Browser WebGL context limits: ~8-16 contexts per page (browser-dependent)
- Gradio file serving: `/gradio_api/file=` prefix for local files
- Base64 overhead: ~33% size increase (use for small data only)

---

**Implementation Status:** Research complete, blueprint provided, code templates ready for ARR-COC integration.
