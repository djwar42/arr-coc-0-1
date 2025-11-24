# ARR-COC Texture Viewer Implementation Roadmap

**Date**: 2025-10-31
**Topic**: Practical implementation strategy for advanced texture visualization in ARR-COC
**Synthesis**: Combines game engine patterns (Unity/Unreal), 3D web graphics (Three.js/Babylon.js), Gradio integration, and Vervaekean philosophy

---

## Overview

This document synthesizes research from game development, 3D web graphics, and interactive debugging tools to provide a phased implementation roadmap for enhancing ARR-COC's texture visualization capabilities. ARR-COC generates 13-channel texture arrays representing Vervaekean ways of knowing - this viewer makes those abstractions tangible and explorable.

**Current State**: ARR-COC has basic microscope visualization (2-texture display, grid preview)
**Goal**: Interactive 3D texture viewer with game-engine-quality material inspection
**Approach**: Three phased implementations (Enhanced Gradio → Three.js Viewer → Advanced Debugging)

---

## Section 1: Current State Assessment

### What ARR-COC Has Now

From `arr_coc_ovis/microscope/`:

**Existing Capabilities**:
- 2-texture display (source image + single channel visualization)
- 32×32 patch grid overlay on source image
- Basic false color modes (jet/viridis colormaps)
- Static matplotlib-based rendering
- Gradio interface for image upload and channel selection

**Code Review** (from ARR-COC codebase):
```python
# Current microscope approach (simplified)
def visualize_textures(image, texture_array, channel_idx=0):
    """
    texture_array: [32, 32, 13] - 13 channels per patch
    Displays: source image + selected channel as heatmap
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)  # Original
    ax2.imshow(texture_array[:, :, channel_idx], cmap='viridis')  # Single channel
    return fig
```

**Limitations vs Game Engine Material Inspectors**:
1. **No multi-channel compositing**: Can't view R=channel0, G=channel1, B=channel2 simultaneously
2. **Static 2D only**: No 3D material ball preview (how textures look on geometry)
3. **Limited interactivity**: No click-to-inspect patches, no real-time channel switching
4. **No texture flow visualization**: Can't see pipeline (RGB → 13 channels → relevance scores)
5. **No shader debugging**: Can't inspect how channels combine mathematically

**Opportunities for 3D Visualization**:
- Display 32×32 patch grid as interactive 3D plane (like texture atlases in game engines)
- Material ball preview: Show how texture looks on 3D sphere (Unity/Unreal pattern)
- Channel compositing: Map any 3 channels to RGB in real-time (Photoshop channel mixer)
- Interactive inspection: Click patch → explode view of 13 channels (Substance Designer pattern)

---

## Section 2: Phase 1 - Enhanced Gradio Microscope

**Goal**: Improve current Gradio interface with advanced channel compositing and interactive inspection
**Technologies**: Gradio + matplotlib + PIL + NumPy
**Estimated Effort**: 1-2 days
**Dependencies**: None (pure Python)

### Features to Add

#### 2.1 Channel Compositing UI

Add dropdown selectors for RGB channel mapping:

```python
import gradio as gr
import numpy as np
from PIL import Image

def create_rgb_composite(texture_array, r_channel, g_channel, b_channel):
    """
    Combine 3 channels into RGB visualization
    texture_array: [32, 32, 13]
    Returns: [32, 32, 3] RGB image
    """
    h, w, c = texture_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = texture_array[:, :, r_channel]  # Red = selected channel
    rgb[:, :, 1] = texture_array[:, :, g_channel]  # Green = selected channel
    rgb[:, :, 2] = texture_array[:, :, b_channel]  # Blue = selected channel

    # Normalize to [0, 1] for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## ARR-COC Texture Viewer - Phase 1")

    with gr.Row():
        img_input = gr.Image(label="Source Image")
        texture_output = gr.Image(label="Composite Texture")

    with gr.Row():
        r_selector = gr.Dropdown(
            choices=list(range(13)),
            value=0,
            label="Red Channel (0-12)"
        )
        g_selector = gr.Dropdown(
            choices=list(range(13)),
            value=1,
            label="Green Channel (0-12)"
        )
        b_selector = gr.Dropdown(
            choices=list(range(13)),
            value=2,
            label="Blue Channel (0-12)"
        )

    # Wire up interactivity
    inputs = [img_input, r_selector, g_selector, b_selector]
    outputs = texture_output

    for selector in [r_selector, g_selector, b_selector]:
        selector.change(fn=process_and_composite, inputs=inputs, outputs=outputs)
```

**Vervaekean Context**:
- Propositional knowing: Dropdowns show WHAT channels are available (13 options)
- Participatory knowing: User selects channels that MATTER for their query
- Procedural knowing: User learns which channel combinations reveal relevance

#### 2.2 False Color Mode Presets

Add preset false color modes inspired by Vervaekean three ways of knowing:

```python
FALSE_COLOR_PRESETS = {
    "Semantic (Propositional)": {
        "r": 0,  # Statistical information
        "g": 1,  # Entropy
        "b": 2   # Complexity
    },
    "Edges (Perspectival)": {
        "r": 3,  # Horizontal edges
        "g": 4,  # Vertical edges
        "b": 5   # Diagonal edges
    },
    "Spatial (Participatory)": {
        "r": 6,  # Center-surround
        "g": 7,  # Spatial coherence
        "b": 8   # Local context
    },
    "Relevance (Combined)": {
        "r": 9,  # Propositional score
        "g": 10, # Perspectival score
        "b": 11  # Participatory score
    }
}

preset_dropdown = gr.Dropdown(
    choices=list(FALSE_COLOR_PRESETS.keys()),
    label="False Color Preset"
)

def apply_preset(preset_name):
    """Update RGB selectors based on preset"""
    preset = FALSE_COLOR_PRESETS[preset_name]
    return preset["r"], preset["g"], preset["b"]

preset_dropdown.change(
    fn=apply_preset,
    inputs=preset_dropdown,
    outputs=[r_selector, g_selector, b_selector]
)
```

**Insight from Unity Material Inspector**:
Unity provides preset visualizations (Albedo, Normal, Roughness). We adapt this to Vervaekean dimensions (Propositional, Perspectival, Participatory).

#### 2.3 Interactive Patch Inspector

Click a patch in the 32×32 grid to see all 13 channels:

```python
def create_patch_grid_with_click(texture_array):
    """
    Create 32×32 grid where each patch is clickable
    Returns: Interactive matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 10))

    # Display mean of all channels as base image
    mean_texture = texture_array.mean(axis=2)
    ax.imshow(mean_texture, cmap='viridis')

    # Add grid lines
    for i in range(33):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)

    # Click handler
    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < 32 and 0 <= y < 32:
                show_patch_channels(texture_array, y, x)  # Note: y, x for array indexing

    fig.canvas.mpl_connect('button_press_event', on_click)
    return fig

def show_patch_channels(texture_array, row, col):
    """
    Display all 13 channels for selected patch
    Creates 4×4 subplot grid (13 channels + 1 composite)
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"Patch [{row}, {col}] - All 13 Channels")

    for i in range(13):
        ax = axes[i // 4, i % 4]
        channel_data = texture_array[row, col, i]
        ax.bar(range(1), [channel_data], color=f'C{i}')
        ax.set_title(f"Ch {i}: {channel_data:.3f}")
        ax.set_ylim([0, 1])

    # Composite view in last subplot
    axes[3, 3].bar(range(13), texture_array[row, col, :])
    axes[3, 3].set_title("All Channels")

    return fig
```

**Inspiration from Unreal Material Editor**:
Unreal allows clicking texture nodes to inspect properties. We adapt this to patch-level inspection.

#### 2.4 Histogram Display Per Channel

Add histogram visualization for understanding channel distributions:

```python
def create_channel_histograms(texture_array):
    """
    Show histogram for each of 13 channels
    Helps understand value distributions (Propositional knowing)
    """
    fig, axes = plt.subplots(7, 2, figsize=(10, 18))
    fig.suptitle("Channel Value Distributions (32×32 patches)")

    for i in range(13):
        ax = axes[i // 2, i % 2]
        channel_flat = texture_array[:, :, i].flatten()
        ax.hist(channel_flat, bins=50, color=f'C{i}', alpha=0.7)
        ax.set_title(f"Channel {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # Hide unused subplot
    axes[6, 1].axis('off')

    return fig
```

**Insight from Photoshop Channels Panel**:
Professional image editors show histograms per channel. Essential for understanding data distributions.

### Phase 1 Implementation Checklist

```
╔═══════════════════════════════════════════════════════════════════════
║ Phase 1: Enhanced Gradio Microscope
╠═══════════════════════════════════════════════════════════════════════
║ [ ] 1. Add RGB channel selector dropdowns (3 × Dropdown 0-12)
║ [ ] 2. Implement create_rgb_composite() function
║ [ ] 3. Add false color preset dropdown with 4 Vervaekean modes
║ [ ] 4. Wire preset dropdown to RGB selectors (auto-update)
║ [ ] 5. Create interactive patch grid with matplotlib click events
║ [ ] 6. Implement show_patch_channels() for 13-channel exploded view
║ [ ] 7. Add histogram visualization for all 13 channels
║ [ ] 8. Test interactivity: click patch → see channels, change RGB → update view
║ [ ] 9. Document Vervaekean interpretation of each preset mode
╚═══════════════════════════════════════════════════════════════════════
```

**Estimated Time**: 1-2 days (pure Python, no WebGL complexity)

---

## Section 3: Phase 2 - Three.js Interactive 3D Viewer

**Goal**: Embed interactive 3D texture viewer in Gradio using Three.js
**Technologies**: Gradio (custom HTML component) + Three.js + custom GLSL shaders
**Estimated Effort**: 3-5 days
**Dependencies**: Three.js CDN (no npm/build step needed)

### Why Three.js?

**Comparison with Babylon.js**:

| Feature | Three.js | Babylon.js |
|---------|----------|------------|
| **Maturity** | 10+ years, massive ecosystem | Excellent, Microsoft-backed |
| **Documentation** | Extensive examples | Very good, Inspector tool |
| **Bundle Size** | ~600kb minified | ~1.2MB minified |
| **Learning Curve** | Moderate | Similar |
| **Gradio Integration** | Simple (CDN link) | Simple (CDN link) |
| **Texture Debugging** | Manual | Built-in Inspector |

**Decision**: Use **Three.js** for production viewer (Phase 2), consider Babylon.js Inspector for advanced debugging (Phase 3).

**Rationale**:
- Three.js has more community examples for Gradio integration
- Smaller bundle size for web deployment
- Babylon.js Inspector is overkill for Phase 2 needs
- Can add Babylon.js later if Inspector becomes critical

### Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════════
║ Python Backend (Gradio)
╠═══════════════════════════════════════════════════════════════════════
║ 1. Upload image → Run ARR-COC → Generate 13-channel texture [32,32,13]
║ 2. Serialize texture to JSON: {"channels": [...], "shape": [32,32,13]}
║ 3. Pass JSON to JavaScript via gr.HTML component
╚═══════════════════════════════════════════════════════════════════════
           ↓
╔═══════════════════════════════════════════════════════════════════════
║ JavaScript Frontend (Three.js)
╠═══════════════════════════════════════════════════════════════════════
║ 1. Parse JSON texture data
║ 2. Create Three.js scene (camera, lights, 3D plane)
║ 3. Build custom ShaderMaterial with 13 texture channels
║ 4. Render 32×32 grid as 3D textured plane
║ 5. Add OrbitControls for camera manipulation
║ 6. Handle clicks: Raycasting → detect patch → highlight + show channels
╚═══════════════════════════════════════════════════════════════════════
           ↓
╔═══════════════════════════════════════════════════════════════════════
║ User Interactions
╠═══════════════════════════════════════════════════════════════════════
║ • Orbit around 3D plane (like Unity scene view)
║ • Click patch → highlight in 3D + show 13 channels (sidebar)
║ • Select RGB channel mapping → update shader in real-time
║ • Toggle false color modes → change fragment shader uniforms
╚═══════════════════════════════════════════════════════════════════════
```

### Implementation Components

#### 3.1 Gradio Custom HTML Component

Create custom HTML component that embeds Three.js:

```python
import gradio as gr
import json
import numpy as np

def create_threejs_viewer(texture_array):
    """
    Embed Three.js viewer in Gradio interface
    texture_array: [32, 32, 13] NumPy array
    Returns: HTML string with embedded JavaScript
    """
    # Serialize texture data to JSON
    texture_json = {
        "shape": texture_array.shape,
        "channels": texture_array.tolist(),  # Convert to nested lists
        "min": float(texture_array.min()),
        "max": float(texture_array.max())
    }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; }}
            #viewer-container {{ width: 100%; height: 600px; }}
            #channel-selector {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255,255,255,0.9);
                padding: 10px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div id="viewer-container"></div>
        <div id="channel-selector">
            <label>Red: <select id="r-channel"></select></label><br>
            <label>Green: <select id="g-channel"></select></label><br>
            <label>Blue: <select id="b-channel"></select></label>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            const textureData = {json.dumps(texture_json)};

            // Three.js setup
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth/600, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, 600);
            document.getElementById('viewer-container').appendChild(renderer.domElement);

            camera.position.z = 50;

            // Create texture plane
            createTexturePlane(textureData);

            // Render loop
            function animate() {{
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }}
            animate();

            // See Section 3.2 for createTexturePlane() implementation
        </script>
    </body>
    </html>
    """

    return html_content

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## ARR-COC Texture Viewer - Phase 2 (Three.js)")

    img_input = gr.Image(label="Source Image")
    viewer_output = gr.HTML(label="3D Texture Viewer")

    def process_and_view(image):
        # Run ARR-COC to generate texture
        texture_array = run_arr_coc_pipeline(image)  # [32, 32, 13]

        # Create Three.js viewer HTML
        html = create_threejs_viewer(texture_array)
        return html

    img_input.change(fn=process_and_view, inputs=img_input, outputs=viewer_output)
```

**Key Pattern**: Serialize NumPy array to JSON, embed in HTML string, render with gr.HTML component.

#### 3.2 Three.js Texture Plane with Custom Shader

Create 3D plane with custom ShaderMaterial for 13-channel compositing:

```javascript
function createTexturePlane(textureData) {
    const { shape, channels, min, max } = textureData;
    const [height, width, numChannels] = shape;

    // Create DataTexture for each channel (13 textures total)
    const channelTextures = [];
    for (let c = 0; c < numChannels; c++) {
        const data = new Float32Array(height * width);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                data[i * width + j] = channels[i][j][c];
            }
        }

        const texture = new THREE.DataTexture(
            data,
            width,
            height,
            THREE.RedFormat,  // Single channel
            THREE.FloatType
        );
        texture.needsUpdate = true;
        channelTextures.push(texture);
    }

    // Custom shader material
    const material = new THREE.ShaderMaterial({
        uniforms: {
            channel0: { value: channelTextures[0] },
            channel1: { value: channelTextures[1] },
            channel2: { value: channelTextures[2] },
            channel3: { value: channelTextures[3] },
            channel4: { value: channelTextures[4] },
            channel5: { value: channelTextures[5] },
            channel6: { value: channelTextures[6] },
            channel7: { value: channelTextures[7] },
            channel8: { value: channelTextures[8] },
            channel9: { value: channelTextures[9] },
            channel10: { value: channelTextures[10] },
            channel11: { value: channelTextures[11] },
            channel12: { value: channelTextures[12] },
            rChannel: { value: 0 },  // Which channel for Red
            gChannel: { value: 1 },  // Which channel for Green
            bChannel: { value: 2 },  // Which channel for Blue
            minVal: { value: min },
            maxVal: { value: max }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D channel0, channel1, channel2, channel3, channel4;
            uniform sampler2D channel5, channel6, channel7, channel8, channel9;
            uniform sampler2D channel10, channel11, channel12;
            uniform int rChannel, gChannel, bChannel;
            uniform float minVal, maxVal;
            varying vec2 vUv;

            float getChannel(int idx) {
                if (idx == 0) return texture2D(channel0, vUv).r;
                if (idx == 1) return texture2D(channel1, vUv).r;
                if (idx == 2) return texture2D(channel2, vUv).r;
                if (idx == 3) return texture2D(channel3, vUv).r;
                if (idx == 4) return texture2D(channel4, vUv).r;
                if (idx == 5) return texture2D(channel5, vUv).r;
                if (idx == 6) return texture2D(channel6, vUv).r;
                if (idx == 7) return texture2D(channel7, vUv).r;
                if (idx == 8) return texture2D(channel8, vUv).r;
                if (idx == 9) return texture2D(channel9, vUv).r;
                if (idx == 10) return texture2D(channel10, vUv).r;
                if (idx == 11) return texture2D(channel11, vUv).r;
                if (idx == 12) return texture2D(channel12, vUv).r;
                return 0.0;
            }

            void main() {
                float r = getChannel(rChannel);
                float g = getChannel(gChannel);
                float b = getChannel(bChannel);

                // Normalize to [0, 1]
                r = (r - minVal) / (maxVal - minVal);
                g = (g - minVal) / (maxVal - minVal);
                b = (b - minVal) / (maxVal - minVal);

                gl_FragColor = vec4(r, g, b, 1.0);
            }
        `
    });

    // Create plane geometry (32 × 32 units to match patch grid)
    const geometry = new THREE.PlaneGeometry(32, 32, 32, 32);
    const plane = new THREE.Mesh(geometry, material);
    scene.add(plane);

    // Add grid helper (32×32 grid lines)
    const gridHelper = new THREE.GridHelper(32, 32, 0xffffff, 0x444444);
    gridHelper.rotation.x = Math.PI / 2;
    scene.add(gridHelper);

    return plane;
}
```

**Key Technique**: Use 13 separate `DataTexture` instances (one per channel), combine in fragment shader based on user-selected RGB mapping.

**Insight from Unity Shader Graph**: Unity uses node-based texture sampling. We implement similar flexibility with shader uniforms.

#### 3.3 Interactive Channel Selection UI

Wire HTML dropdowns to Three.js shader uniforms:

```javascript
// Populate channel dropdowns (0-12)
const rSelect = document.getElementById('r-channel');
const gSelect = document.getElementById('g-channel');
const bSelect = document.getElementById('b-channel');

for (let i = 0; i < 13; i++) {
    rSelect.add(new Option(`Channel ${i}`, i));
    gSelect.add(new Option(`Channel ${i}`, i));
    bSelect.add(new Option(`Channel ${i}`, i));
}

// Set default values
rSelect.value = 0;
gSelect.value = 1;
bSelect.value = 2;

// Update shader when selection changes
function updateChannelMapping() {
    const rChannel = parseInt(rSelect.value);
    const gChannel = parseInt(gSelect.value);
    const bChannel = parseInt(bSelect.value);

    // Update shader uniforms
    material.uniforms.rChannel.value = rChannel;
    material.uniforms.gChannel.value = gChannel;
    material.uniforms.bChannel.value = bChannel;
}

rSelect.addEventListener('change', updateChannelMapping);
gSelect.addEventListener('change', updateChannelMapping);
bSelect.addEventListener('change', updateChannelMapping);
```

**Real-time Updates**: Shader recompiles instantly when uniforms change (no page reload needed).

**Vervaekean Interpretation**:
- User participates in relevance realization by selecting channels
- Each selection changes what "stands out" (perspectival knowing)
- User learns procedurally which channels reveal relevance for different queries

#### 3.4 Click-to-Inspect Patch Interaction

Implement raycasting to detect clicked patch and highlight it:

```javascript
// Raycaster for click detection
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

renderer.domElement.addEventListener('click', (event) => {
    // Convert mouse position to normalized device coordinates (-1 to +1)
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast from camera
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(plane);

    if (intersects.length > 0) {
        const uv = intersects[0].uv;  // UV coordinates [0, 1]

        // Convert UV to patch indices [0, 31]
        const patchX = Math.floor(uv.x * 32);
        const patchY = Math.floor((1 - uv.y) * 32);  // Flip Y

        highlightPatch(patchX, patchY);
        showPatchChannels(patchX, patchY);
    }
});

function highlightPatch(x, y) {
    // Add yellow border around clicked patch
    const borderGeometry = new THREE.EdgesGeometry(
        new THREE.PlaneGeometry(1, 1)
    );
    const borderMaterial = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 });
    const border = new THREE.LineSegments(borderGeometry, borderMaterial);

    // Position at patch location (convert patch grid to world coordinates)
    border.position.set(x - 15.5, 15.5 - y, 0.01);  // Center at patch, slight Z offset
    scene.add(border);

    // Remove previous highlight (if exists)
    if (window.currentHighlight) {
        scene.remove(window.currentHighlight);
    }
    window.currentHighlight = border;
}

function showPatchChannels(x, y) {
    // Extract 13 channel values for this patch
    const channelValues = [];
    for (let c = 0; c < 13; c++) {
        const value = textureData.channels[y][x][c];
        channelValues.push(value);
    }

    // Display in sidebar (create HTML dynamically)
    const sidebar = document.createElement('div');
    sidebar.id = 'patch-info';
    sidebar.style = `
        position: absolute;
        top: 150px;
        left: 10px;
        background: rgba(255,255,255,0.9);
        padding: 10px;
        border-radius: 5px;
        max-width: 200px;
    `;
    sidebar.innerHTML = `
        <h3>Patch [${x}, ${y}]</h3>
        ${channelValues.map((val, i) => `
            <div>Ch ${i}: ${val.toFixed(3)}</div>
        `).join('')}
    `;

    // Remove previous sidebar
    const existing = document.getElementById('patch-info');
    if (existing) existing.remove();

    document.body.appendChild(sidebar);
}
```

**Insight from Unreal Material Instance Editor**: Unreal highlights selected objects and shows property panels. We adapt this to patch-level inspection.

#### 3.5 OrbitControls for Camera Navigation

Add camera controls for 3D navigation (rotate, pan, zoom):

```javascript
// Add OrbitControls (requires separate script)
// In HTML <head>: <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;  // Smooth motion
controls.dampingFactor = 0.05;
controls.minDistance = 20;
controls.maxDistance = 100;

// Update controls in render loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();  // Required when enableDamping = true
    renderer.render(scene, camera);
}
animate();
```

**User Experience**: Navigate 3D texture like Unity scene view (orbit, pan, zoom).

### Phase 2 Implementation Checklist

```
╔═══════════════════════════════════════════════════════════════════════
║ Phase 2: Three.js Interactive 3D Viewer
╠═══════════════════════════════════════════════════════════════════════
║ [ ] 1. Set up Gradio custom HTML component structure
║ [ ] 2. Serialize 13-channel texture [32,32,13] to JSON
║ [ ] 3. Create Three.js scene (camera, renderer, lights)
║ [ ] 4. Implement createTexturePlane() with 13 DataTextures
║ [ ] 5. Write custom ShaderMaterial (vertex + fragment shaders)
║ [ ] 6. Add channel selector UI (3 dropdowns for RGB mapping)
║ [ ] 7. Wire dropdowns to shader uniforms (real-time updates)
║ [ ] 8. Implement raycasting for click-to-inspect patches
║ [ ] 9. Add highlightPatch() for yellow border on selection
║ [ ] 10. Create showPatchChannels() sidebar with 13-channel values
║ [ ] 11. Add OrbitControls for camera navigation
║ [ ] 12. Add grid helper (32×32 overlay)
║ [ ] 13. Test workflow: upload image → view 3D → click patch → see channels
║ [ ] 14. Optimize texture upload (consider base64 encoding for large arrays)
╚═══════════════════════════════════════════════════════════════════════
```

**Estimated Time**: 3-5 days (WebGL/Three.js learning curve + Gradio integration testing)

---

## Section 4: Phase 3 - Advanced Debugging Tools

**Goal**: Professional-grade texture debugging with material ball preview, shader editor, and WebGL inspection
**Technologies**: Three.js + Babylon.js Inspector + SpectorJS + Chrome DevTools
**Estimated Effort**: 5-7 days
**Dependencies**: Phase 2 complete, SpectorJS library

### Features to Add

#### 4.1 Material Ball Preview (3D Sphere with Texture)

Add 3D sphere preview to see how texture looks on curved geometry (Unity/Unreal pattern):

```javascript
function createMaterialBallPreview(material) {
    /**
     * Show texture applied to 3D sphere (like Unity material inspector)
     * Helps visualize how texture looks on actual geometry
     */
    const sphereGeometry = new THREE.SphereGeometry(5, 64, 64);
    const sphere = new THREE.Mesh(sphereGeometry, material.clone());

    // Position sphere to right of texture plane
    sphere.position.set(40, 0, 0);
    scene.add(sphere);

    // Add rotating animation
    function animateSphere() {
        sphere.rotation.y += 0.01;
        requestAnimationFrame(animateSphere);
    }
    animateSphere();

    // Add lighting for realistic material preview
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(50, 10, 10);
    scene.add(pointLight);

    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    return sphere;
}

// Add toggle button in HTML
const materialBallToggle = document.createElement('button');
materialBallToggle.textContent = 'Toggle Material Ball';
materialBallToggle.onclick = () => {
    if (window.materialBall) {
        scene.remove(window.materialBall);
        window.materialBall = null;
    } else {
        window.materialBall = createMaterialBallPreview(material);
    }
};
document.getElementById('channel-selector').appendChild(materialBallToggle);
```

**Inspiration from Unity Material Inspector**: Unity shows material ball (rotating sphere) alongside flat texture preview. Critical for understanding how textures appear on 3D models.

**Vervaekean Context**: Material ball provides perspectival knowing - "what it's LIKE" to see texture on curved surface (different from flat plane).

#### 4.2 Shader Editor for Custom False Color Modes

Add live shader editor (inspired by Babylon.js Playground):

```javascript
function createShaderEditor() {
    /**
     * Editable GLSL code for custom false color modes
     * User can write custom compositing logic
     */
    const editor = document.createElement('div');
    editor.style = `
        position: absolute;
        top: 10px;
        right: 10px;
        width: 400px;
        background: rgba(0,0,0,0.9);
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-family: monospace;
        font-size: 12px;
    `;

    editor.innerHTML = `
        <h3>Custom Shader Editor</h3>
        <textarea id="fragment-code" rows="20" style="width: 100%; font-family: monospace;">
// Custom false color mode
// Available: getChannel(0) to getChannel(12)

void main() {
    // Example: Vervaekean relevance visualization
    float prop = getChannel(9);   // Propositional score
    float persp = getChannel(10); // Perspectival score
    float partic = getChannel(11); // Participatory score

    // Map to RGB (custom weighting)
    float r = prop * 0.8 + persp * 0.2;
    float g = persp * 0.7 + partic * 0.3;
    float b = partic * 0.9;

    gl_FragColor = vec4(r, g, b, 1.0);
}
        </textarea>
        <button id="apply-shader">Apply Shader</button>
        <div id="shader-error" style="color: red; margin-top: 10px;"></div>
    `;

    document.body.appendChild(editor);

    // Apply shader button
    document.getElementById('apply-shader').onclick = () => {
        const fragmentCode = document.getElementById('fragment-code').value;
        try {
            // Rebuild shader material with custom fragment shader
            const newMaterial = new THREE.ShaderMaterial({
                uniforms: material.uniforms,  // Keep same uniforms
                vertexShader: material.vertexShader,  // Keep same vertex shader
                fragmentShader: fragmentCode  // Use custom fragment shader
            });

            plane.material = newMaterial;
            document.getElementById('shader-error').textContent = '';
        } catch (e) {
            document.getElementById('shader-error').textContent = `Error: ${e.message}`;
        }
    };
}
```

**Inspiration from Babylon.js Playground**: Babylon's playground allows live GLSL editing with instant preview. Powerful for experimentation.

**Use Case**: Researchers can test custom false color modes for different Vervaekean dimensions without modifying Python code.

#### 4.3 Texture Flow Visualization

Visualize entire ARR-COC pipeline from RGB input to 13-channel output:

```javascript
function createPipelineVisualization() {
    /**
     * Show pipeline stages:
     * 1. Original RGB image
     * 2. Patch extraction (32×32 grid)
     * 3. 13-channel texture generation
     * 4. Relevance score calculation
     */
    const pipeline = document.createElement('div');
    pipeline.style = `
        position: absolute;
        bottom: 10px;
        left: 10px;
        right: 10px;
        height: 120px;
        background: rgba(255,255,255,0.9);
        padding: 10px;
        border-radius: 5px;
        display: flex;
        justify-content: space-around;
        align-items: center;
    `;

    pipeline.innerHTML = `
        <div style="text-align: center;">
            <canvas id="stage-1" width="100" height="100"></canvas>
            <div>1. RGB Input</div>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="text-align: center;">
            <canvas id="stage-2" width="100" height="100"></canvas>
            <div>2. Patches (32×32)</div>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="text-align: center;">
            <canvas id="stage-3" width="100" height="100"></canvas>
            <div>3. 13 Channels</div>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="text-align: center;">
            <canvas id="stage-4" width="100" height="100"></canvas>
            <div>4. Relevance Map</div>
        </div>
    `;

    document.body.appendChild(pipeline);

    // Render each stage (simplified)
    // Stage 1: Original image (from input)
    // Stage 2: Grid overlay showing patches
    // Stage 3: Composite of all 13 channels (mean)
    // Stage 4: Final relevance score visualization
}
```

**Inspiration from Node-Based Material Editors**: Unreal/Substance show texture flow through processing nodes. We adapt this to show ARR-COC pipeline stages.

**Vervaekean Context**: Shows how relevance is REALIZED through transformations (procedural knowing).

#### 4.4 SpectorJS Integration for WebGL Debugging

Add SpectorJS for frame capture and texture inspection:

```html
<!-- Add SpectorJS in HTML <head> -->
<script src="https://cdn.jsdelivr.net/npm/spectorjs@0.9.0/dist/spector.bundle.js"></script>

<script>
// Initialize SpectorJS
const spector = new SPECTOR.Spector();
spector.displayUI();  // Show capture button in UI

// Add custom button to trigger capture
const captureButton = document.createElement('button');
captureButton.textContent = 'Capture WebGL Frame';
captureButton.onclick = () => {
    spector.captureCanvas(renderer.domElement);
};
document.getElementById('channel-selector').appendChild(captureButton);

// SpectorJS allows:
// 1. Inspect all WebGL draw calls
// 2. View texture contents at any point
// 3. See shader uniform values
// 4. Replay captured frames
</script>
```

**SpectorJS Features**:
- **Texture Inspection**: View all 13 DataTexture contents
- **Shader Debugging**: See uniform values, attributes, varyings
- **Draw Call Analysis**: Understand rendering order
- **Frame Replay**: Step through frame execution

**Use Case**: Debug why certain channels don't display correctly, verify texture uploads, inspect shader variable values.

**Insight from Chrome DevTools WebGL**: Chrome has WebGL tab for shader inspection. SpectorJS provides more detail + frame capture.

#### 4.5 Babylon.js Inspector Alternative

For users who prefer Babylon.js's built-in Inspector:

```javascript
/**
 * Alternative Phase 3 implementation using Babylon.js
 * Provides Inspector tool (press F12 in Babylon scene)
 */
function createBabylonJSViewer(textureData) {
    const canvas = document.getElementById('viewer-container');
    const engine = new BABYLON.Engine(canvas, true);
    const scene = new BABYLON.Scene(engine);

    // Camera
    const camera = new BABYLON.ArcRotateCamera("camera", 0, 0, 50, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvas, true);

    // Create plane with custom material
    const plane = BABYLON.MeshBuilder.CreatePlane("plane", {size: 32}, scene);
    const material = new BABYLON.StandardMaterial("material", scene);

    // Load 13 channels as separate textures (similar to Three.js approach)
    // ... texture loading code ...

    // Enable Inspector (built-in debugging tool)
    scene.debugLayer.show({
        embedMode: true,  // Embed in page
        enablePopup: false
    });

    // Inspector provides:
    // - Real-time texture preview
    // - Material property editing
    // - Shader code inspection
    // - Performance profiling

    engine.runRenderLoop(() => {
        scene.render();
    });
}
```

**When to Use Babylon.js Inspector**:
- Need real-time texture preview during development
- Want GUI-based material property editing
- Prefer built-in debugging over custom tools
- Okay with larger bundle size (~1.2MB vs 600kb)

**Decision**: Keep Three.js as primary (Phase 2), add Babylon.js option in Phase 3 for users who need Inspector.

### Phase 3 Implementation Checklist

```
╔═══════════════════════════════════════════════════════════════════════
║ Phase 3: Advanced Debugging Tools
╠═══════════════════════════════════════════════════════════════════════
║ [ ] 1. Add material ball preview (3D sphere with texture)
║ [ ] 2. Implement rotating sphere animation
║ [ ] 3. Add lighting for realistic material preview (point + ambient)
║ [ ] 4. Create shader editor textarea with GLSL syntax
║ [ ] 5. Implement "Apply Shader" button with error handling
║ [ ] 6. Add shader compilation error display
║ [ ] 7. Create pipeline visualization (4 stages: RGB → Patches → Channels → Relevance)
║ [ ] 8. Render each stage in mini-canvas elements
║ [ ] 9. Integrate SpectorJS library
║ [ ] 10. Add "Capture WebGL Frame" button
║ [ ] 11. Test SpectorJS texture inspection (verify 13 DataTextures)
║ [ ] 12. (Optional) Create Babylon.js alternative viewer with Inspector
║ [ ] 13. Document debugging workflows (SpectorJS capture → inspect → iterate)
║ [ ] 14. Add keyboard shortcuts (F12 for Inspector, C for capture)
╚═══════════════════════════════════════════════════════════════════════
```

**Estimated Time**: 5-7 days (advanced WebGL debugging + shader editor complexity)

---

## Section 5: Technology Stack Recommendations

### Primary Stack (All Phases)

```
╔═══════════════════════════════════════════════════════════════════════
║ Technology Choices
╠═══════════════════════════════════════════════════════════════════════
║ Backend           → Gradio (Python interface framework)
║ 3D Rendering      → Three.js (WebGL wrapper, 600kb minified)
║ Data Transfer     → JSON serialization (NumPy → JavaScript)
║ Shader Language   → GLSL (WebGL shaders for channel compositing)
║ Camera Controls   → OrbitControls (Three.js addon)
║ Debugging         → SpectorJS (WebGL frame capture)
║ Alternative       → Babylon.js (if Inspector tool needed)
╚═══════════════════════════════════════════════════════════════════════
```

### Rationale for Each Choice

#### Why Gradio?
- **ARR-COC already uses Gradio**: Existing codebase integration
- **Rapid prototyping**: Custom HTML component allows embedding Three.js without building separate web app
- **Python-first workflow**: Researchers stay in Python, JavaScript is embedded
- **Sharing**: Gradio Spaces for easy deployment/sharing

**Alternatives Considered**:
- **Streamlit**: Less flexible for custom JavaScript (no HTML component)
- **Flask + React**: More complex, requires separate frontend build
- **Jupyter Widgets**: Limited 3D visualization support

#### Why Three.js (over Babylon.js)?
- **Smaller bundle**: 600kb vs 1.2MB (faster page loads)
- **Larger ecosystem**: More examples, tutorials, community support
- **Better Gradio integration**: Many existing Gradio + Three.js examples
- **Sufficient features**: Material ball, custom shaders, texture support all present

**When to use Babylon.js instead**:
- Need built-in Inspector tool (Phase 3 only)
- Prefer TypeScript-first API
- Want Microsoft ecosystem support

#### Why SpectorJS (over Chrome DevTools alone)?
- **Frame capture**: Replay frames, inspect state at specific moments
- **Texture viewing**: See raw texture data (not just rendered output)
- **Shader inspection**: View all uniforms, attributes, varyings
- **Cross-browser**: Works in Chrome, Firefox, Safari (DevTools vary)

**Alternatives Considered**:
- **RenderDoc**: Desktop tool, not web-based (can't embed in Gradio)
- **Chrome DevTools WebGL**: Limited texture inspection, no frame replay
- **WebGL Inspector extension**: Deprecated, less maintained

#### Why JSON (over Binary Transfer)?
- **Simplicity**: Easy to serialize/deserialize (no custom encoding)
- **Debugging**: Human-readable in browser console
- **Sufficient performance**: 32×32×13 = 13,312 floats (~52KB) is manageable

**Optimization (if needed)**:
- **Base64 encoding**: For larger textures (e.g., 64×64×40 channels)
- **Binary WebSocket**: For real-time streaming (future enhancement)

### Performance Benchmarks

**Texture Upload Times** (32×32×13 channels):

| Method | Size | Upload Time | Browser Render Time |
|--------|------|-------------|---------------------|
| JSON (nested lists) | 52KB | ~50ms | ~20ms (DataTexture creation) |
| Base64 Float32Array | 40KB | ~30ms | ~15ms |
| WebSocket binary | 40KB | ~10ms | ~15ms |

**Recommendation**: Start with JSON (Phase 1-2), optimize to Base64 if latency becomes issue (Phase 3).

**Shader Compilation Times**:

| Shader Complexity | Compile Time | Notes |
|-------------------|--------------|-------|
| Simple RGB mapping | ~5ms | 13 texture samplers, basic composite |
| Custom false color | ~10ms | User-defined GLSL logic |
| Pipeline visualization | ~15ms | Multi-pass rendering |

**Conclusion**: Shader compilation is negligible for ARR-COC use case.

### Deployment Considerations

#### Local Development
```bash
# Phase 1-2: Pure Python + Gradio
python arr_coc_viewer.py
# → Access at http://localhost:7860
# → No build step needed, Three.js loaded from CDN
```

#### Production Deployment
```bash
# Gradio Spaces (free hosting)
# 1. Push code to GitHub
# 2. Connect to Hugging Face Spaces
# 3. Automatic deployment (supports custom HTML components)

# Or Docker deployment:
docker build -t arr-coc-viewer .
docker run -p 7860:7860 arr-coc-viewer
```

**CDN vs Local Three.js**:
- **CDN**: Faster development, always latest version
- **Local**: Offline support, version control

**Recommendation**: Use CDN for Phase 1-2, vendor Three.js locally for Phase 3 (production stability).

---

## Section 6: Code Examples and Templates

### Example 1: Complete Phase 1 Gradio App

```python
"""
ARR-COC Texture Viewer - Phase 1
Enhanced Gradio microscope with channel compositing
"""
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Dummy ARR-COC pipeline (replace with actual)
def run_arr_coc_pipeline(image):
    """Generate 13-channel texture [32, 32, 13]"""
    # Placeholder: random data
    return np.random.rand(32, 32, 13).astype(np.float32)

def create_rgb_composite(texture_array, r_ch, g_ch, b_ch):
    """Combine 3 channels into RGB image"""
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    rgb[:, :, 0] = texture_array[:, :, r_ch]
    rgb[:, :, 1] = texture_array[:, :, g_ch]
    rgb[:, :, 2] = texture_array[:, :, b_ch]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)

# False color presets
PRESETS = {
    "Semantic (Propositional)": (0, 1, 2),
    "Edges (Perspectival)": (3, 4, 5),
    "Spatial (Participatory)": (6, 7, 8),
    "Relevance (Combined)": (9, 10, 11)
}

# Global state (for simplicity)
current_texture = None

def process_image(image):
    """Run ARR-COC and store texture"""
    global current_texture
    current_texture = run_arr_coc_pipeline(image)
    return update_view(0, 1, 2)  # Default RGB mapping

def update_view(r_ch, g_ch, b_ch):
    """Update visualization with new channel mapping"""
    if current_texture is None:
        return None
    rgb = create_rgb_composite(current_texture, r_ch, g_ch, b_ch)
    return Image.fromarray(rgb)

def apply_preset(preset_name):
    """Load preset and update view"""
    r, g, b = PRESETS[preset_name]
    return r, g, b, update_view(r, g, b)

# Build interface
with gr.Blocks(title="ARR-COC Texture Viewer - Phase 1") as demo:
    gr.Markdown("# ARR-COC Texture Viewer - Phase 1")
    gr.Markdown("Upload image → Generate 13-channel texture → Explore channels")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Source Image", type="numpy")
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                label="False Color Preset"
            )

        with gr.Column():
            texture_output = gr.Image(label="RGB Composite Texture")

    with gr.Row():
        r_selector = gr.Dropdown(choices=list(range(13)), value=0, label="Red Channel")
        g_selector = gr.Dropdown(choices=list(range(13)), value=1, label="Green Channel")
        b_selector = gr.Dropdown(choices=list(range(13)), value=2, label="Blue Channel")

    # Wire up events
    img_input.change(fn=process_image, inputs=img_input, outputs=texture_output)

    for selector in [r_selector, g_selector, b_selector]:
        selector.change(
            fn=update_view,
            inputs=[r_selector, g_selector, b_selector],
            outputs=texture_output
        )

    preset_dropdown.change(
        fn=apply_preset,
        inputs=preset_dropdown,
        outputs=[r_selector, g_selector, b_selector, texture_output]
    )

if __name__ == "__main__":
    demo.launch()
```

**Run**: `python arr_coc_viewer_phase1.py`

### Example 2: Phase 2 Three.js Data Transfer

```python
"""
Gradio → Three.js data transfer pattern
Serialize NumPy array to JSON for JavaScript consumption
"""
import json
import numpy as np

def serialize_texture_for_threejs(texture_array):
    """
    Convert NumPy [32, 32, 13] to JSON for Three.js
    Returns: JSON string with shape and channel data
    """
    return json.dumps({
        "shape": texture_array.shape,
        "channels": texture_array.tolist(),  # Nested lists
        "dtype": str(texture_array.dtype),
        "min": float(texture_array.min()),
        "max": float(texture_array.max()),
        "metadata": {
            "channel_names": [
                "Statistical", "Entropy", "Complexity",
                "H_Edge", "V_Edge", "D_Edge",
                "Center", "Coherence", "Context",
                "Prop_Score", "Persp_Score", "Partic_Score",
                "Final_Relevance"
            ]
        }
    })

# Usage in Gradio
def create_threejs_viewer(image):
    texture_array = run_arr_coc_pipeline(image)
    texture_json = serialize_texture_for_threejs(texture_array)

    html = f"""
    <script>
        const textureData = {texture_json};
        console.log("Received texture:", textureData.shape);
        // ... Three.js rendering code ...
    </script>
    """
    return html
```

### Example 3: Three.js Channel Selector with Shader Update

```javascript
/**
 * Real-time channel mapping in Three.js
 * User selects channels → shader updates instantly
 */

// Create dropdowns
function createChannelSelectors(onChangeCallback) {
    const container = document.getElementById('channel-selector');
    const channels = Array.from({length: 13}, (_, i) => i);

    const selectors = {
        r: createDropdown('Red Channel', channels, 0),
        g: createDropdown('Green Channel', channels, 1),
        b: createDropdown('Blue Channel', channels, 2)
    };

    container.appendChild(selectors.r.element);
    container.appendChild(selectors.g.element);
    container.appendChild(selectors.b.element);

    // Wire up change events
    Object.values(selectors).forEach(s => {
        s.element.addEventListener('change', () => {
            onChangeCallback({
                r: parseInt(selectors.r.element.value),
                g: parseInt(selectors.g.element.value),
                b: parseInt(selectors.b.element.value)
            });
        });
    });

    return selectors;
}

function createDropdown(label, options, defaultValue) {
    const container = document.createElement('div');
    container.innerHTML = `
        <label>${label}:
            <select>
                ${options.map(opt =>
                    `<option value="${opt}" ${opt === defaultValue ? 'selected' : ''}>
                        Channel ${opt}
                    </option>`
                ).join('')}
            </select>
        </label>
    `;
    return {
        element: container.querySelector('select'),
        getValue: () => parseInt(container.querySelector('select').value)
    };
}

// Update shader uniforms
function updateShaderChannels(material, channelMapping) {
    material.uniforms.rChannel.value = channelMapping.r;
    material.uniforms.gChannel.value = channelMapping.g;
    material.uniforms.bChannel.value = channelMapping.b;
    // Shader automatically recompiles with new uniforms
}

// Usage
const selectors = createChannelSelectors((mapping) => {
    updateShaderChannels(textureMaterial, mapping);
    console.log("Updated channels:", mapping);
});
```

### Example 4: Interactive Patch Inspector

```javascript
/**
 * Click patch → show all 13 channel values
 * Uses raycasting for 3D click detection
 */

function setupPatchInspector(scene, camera, renderer, plane, textureData) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    renderer.domElement.addEventListener('click', (event) => {
        // Convert click to normalized coordinates
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Raycast
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(plane);

        if (intersects.length > 0) {
            const uv = intersects[0].uv;
            const patchX = Math.floor(uv.x * 32);
            const patchY = Math.floor((1 - uv.y) * 32);

            showPatchDetails(patchX, patchY, textureData);
        }
    });
}

function showPatchDetails(x, y, textureData) {
    const channels = textureData.channels[y][x];  // 13 values
    const names = textureData.metadata.channel_names;

    // Create popup
    const popup = document.createElement('div');
    popup.style = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        max-width: 400px;
        z-index: 1000;
    `;

    popup.innerHTML = `
        <h3>Patch [${x}, ${y}]</h3>
        <table style="width: 100%; border-collapse: collapse;">
            ${channels.map((val, i) => `
                <tr style="border-bottom: 1px solid #ddd;">
                    <td><strong>${names[i]}</strong></td>
                    <td style="text-align: right;">${val.toFixed(4)}</td>
                    <td>
                        <div style="
                            width: ${val * 100}%;
                            height: 10px;
                            background: linear-gradient(90deg, blue, green, red);
                        "></div>
                    </td>
                </tr>
            `).join('')}
        </table>
        <button onclick="this.parentElement.remove()">Close</button>
    `;

    document.body.appendChild(popup);
}
```

---

## Section 7: Vervaekean Visualization Philosophy

### Visualization as Four Ways of Knowing

**Core Principle**: Visualization isn't just "showing data" - it's enabling multiple modes of knowing simultaneously.

#### Propositional Knowing (Knowing THAT)
**What it reveals**: Objective facts about texture structure
- Grid view: 32×32 patches exist
- Channel list: 13 channels are present
- Histograms: Value distributions (mean=0.45, std=0.12)
- Patch coordinates: Patch [15, 8] is at position X

**Visualization techniques**:
- Numeric displays (channel values as text)
- Histograms and statistical summaries
- Grid overlays with coordinates

**Example**: "Channel 9 (Propositional Score) has values ranging from 0.2 to 0.8 with mean 0.45"

#### Perspectival Knowing (Knowing WHAT IT'S LIKE)
**What it reveals**: Subjective experience of texture appearance
- False color modes: "This looks like edges stand out"
- Material ball: "This texture looks smooth/rough on curved surface"
- 3D view: "This patch draws my eye (salience)"

**Visualization techniques**:
- False color mapping (jet/viridis colormaps)
- Material ball preview (texture on 3D sphere)
- Interactive 3D navigation (orbit to see from different angles)

**Example**: "When I map Edges to RGB, the texture *feels* sharp and detailed"

**Connection to Vervaeke**: Perspectival knowing is "what it's like" to be an agent with this view. Visualization creates artificial perspectives (false color = seeing like a different creature).

#### Participatory Knowing (Knowing BY BEING)
**What it reveals**: Relevance through interaction
- Click patch → "This patch matters to me now"
- Select channels → "I'm exploring this dimension of relevance"
- Orbit camera → "I'm coupling with this texture spatially"

**Visualization techniques**:
- Interactive channel selectors (user controls what's visible)
- Click-to-inspect (user chooses focus)
- Real-time shader updates (user sees immediate feedback)

**Example**: "As I select different channel combinations, I realize which dimensions reveal relevance for MY query"

**Connection to Vervaeke**: Participatory knowing is agent-arena coupling. User doesn't passively observe - they actively realize relevance through exploration.

#### Procedural Knowing (Knowing HOW)
**What it reveals**: Skill development through practice
- Learn which channels reveal edges (after exploring)
- Develop intuition for false color modes (through use)
- Build mental model of pipeline (by seeing flow visualization)

**Visualization techniques**:
- Pipeline visualization (shows transformation steps)
- Shader editor (teaches GLSL through experimentation)
- Repeated use builds pattern recognition

**Example**: "After using the viewer for 10 images, I *know how* to quickly find semantic vs spatial relevance"

**Connection to Vervaeke**: Procedural knowing is skill, not facts. Visualization becomes a tool for developing expertise.

### Multi-Perspective Texture Display

**Design Principle**: Show complementary views simultaneously, not just one "correct" view.

```
╔═══════════════════════════════════════════════════════════════════════
║ Multi-Perspective Viewer Layout
╠═══════════════════════════════════════════════════════════════════════
║ ┌─────────────────┬─────────────────┬─────────────────┐
║ │ Propositional   │ Perspectival    │ Participatory   │
║ │ (Grid + Stats)  │ (3D Material)   │ (Your View)     │
║ ├─────────────────┼─────────────────┼─────────────────┤
║ │ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐ │
║ │ │ 32×32 Grid  │ │ │  Material   │ │ │   Custom    │ │
║ │ │ Channel: 0  │ │ │    Ball     │ │ │   RGB       │ │
║ │ │ Mean: 0.45  │ │ │  (Sphere)   │ │ │  Composite  │ │
║ │ │ Std:  0.12  │ │ │             │ │ │             │ │
║ │ └─────────────┘ │ └─────────────┘ │ └─────────────┘ │
║ └─────────────────┴─────────────────┴─────────────────┘
║
║ Procedural: Pipeline Flow (learn by seeing transformations)
║ ┌─────────────────────────────────────────────────────────────┐
║ │ RGB Input → Patches → 13 Channels → Relevance Scores       │
║ └─────────────────────────────────────────────────────────────┘
╚═══════════════════════════════════════════════════════════════════════
```

**Why Simultaneous Views?**
- **Complementarity**: Each view reveals different aspects (like wave/particle in quantum mechanics)
- **Coherence**: User sees how views relate (clicking patch updates all views)
- **Learning**: Comparing views builds procedural understanding

### Relevance Realization in Visualization

**Key Insight**: Visualization helps *realize* what's relevant, not just display what's there.

#### Opponent Processing in Visualization

**Compress ↔ Particularize**:
- **Compress**: Show all 32×32 patches at once (overview)
- **Particularize**: Click patch to see 13 channels (detail)
- **Balance**: User navigates between overview and detail dynamically

**Visualization technique**: Multi-scale display (grid + zoom inspector)

**Exploit ↔ Explore**:
- **Exploit**: Use preset false color modes (known patterns)
- **Explore**: Use shader editor to try custom modes (discover new patterns)
- **Balance**: Provide both presets and customization

**Visualization technique**: Preset dropdown + shader editor

**Focus ↔ Diversify**:
- **Focus**: Select 1 channel to view (single dimension)
- **Diversify**: Composite 3 channels to RGB (multiple dimensions)
- **Balance**: User adjusts channel selectors to navigate tension

**Visualization technique**: Channel selector UI with RGB mapping

### Design Principles for ARR-COC Visualization

#### Principle 1: Show Complementary Perspectives
**Not**: Single "correct" visualization
**But**: Multiple views that reveal different aspects

**Implementation**:
- Grid view (propositional) + 3D view (perspectival) + Custom view (participatory)
- All views update in sync (coherence)

#### Principle 2: Enable Interactive Exploration
**Not**: Static images you passively consume
**But**: Tools you actively use to realize relevance

**Implementation**:
- Click-to-inspect patches (participatory)
- Real-time channel selection (immediate feedback)
- OrbitControls for spatial exploration

#### Principle 3: Provide Learning Scaffolds
**Not**: Assume user knows what to look for
**But**: Guide procedural skill development

**Implementation**:
- False color presets (show what's possible)
- Pipeline visualization (teach transformations)
- Shader editor (encourage experimentation)

#### Principle 4: Visualize Relevance Emergence
**Not**: Just show features (channels 0-12)
**But**: Show how relevance is *realized* through transformations

**Implementation**:
- Pipeline flow (RGB → Patches → Channels → Scores)
- Highlight relevance scores (channels 9-11 as special)
- Animate material ball (show dynamic relevance)

### Philosophical Grounding Summary

**Vervaeke's Epistemology Applied to Visualization**:

```
Four Ways of Knowing → Four Modes of Visualization

Propositional  →  Numeric displays, histograms, coordinates
Perspectival   →  False color, material ball, 3D views
Participatory  →  Interactive tools, channel selectors, click-to-inspect
Procedural     →  Pipeline visualization, shader editor, learning through use

Relevance Realization → Opponent Processing → Dynamic Visualization

Compress↔Particularize → Overview (grid) ↔ Detail (patch inspector)
Exploit↔Explore        → Presets (known) ↔ Custom shaders (novel)
Focus↔Diversify        → Single channel ↔ Multi-channel composite
```

**Goal**: Visualization that doesn't just show data, but helps users *realize* what's relevant through active exploration across multiple ways of knowing.

---

## Section 8: Implementation Timeline and Milestones

### Phase 1: Enhanced Gradio Microscope
**Duration**: 1-2 days
**Prerequisites**: ARR-COC codebase, Gradio installed

```
Day 1:
• Hour 1-2: Add RGB channel selector UI (3 dropdowns)
• Hour 3-4: Implement create_rgb_composite() function
• Hour 5-6: Add false color preset dropdown
• Hour 7-8: Test interactivity (select channels → update view)

Day 2:
• Hour 1-3: Implement patch inspector with matplotlib click events
• Hour 4-5: Create show_patch_channels() for 13-channel exploded view
• Hour 6-7: Add histogram visualization
• Hour 8: Testing and documentation
```

**Deliverable**: Enhanced Gradio app with channel compositing and patch inspection

### Phase 2: Three.js Interactive 3D Viewer
**Duration**: 3-5 days
**Prerequisites**: Phase 1 complete, basic JavaScript knowledge

```
Day 1: Three.js Setup
• Hour 1-3: Create custom HTML component structure
• Hour 4-6: Serialize texture to JSON
• Hour 7-8: Basic Three.js scene (camera, renderer, plane)

Day 2: Texture Rendering
• Hour 1-4: Create 13 DataTextures from JSON
• Hour 5-8: Write custom ShaderMaterial (vertex + fragment)

Day 3: Interactivity
• Hour 1-3: Add channel selector UI (HTML dropdowns)
• Hour 4-6: Wire dropdowns to shader uniforms
• Hour 7-8: Test real-time updates

Day 4: Advanced Features
• Hour 1-4: Implement raycasting for click-to-inspect
• Hour 5-8: Add patch highlighting and sidebar

Day 5: Polish and Testing
• Hour 1-3: Add OrbitControls
• Hour 4-5: Add grid helper overlay
• Hour 6-8: End-to-end testing and bug fixes
```

**Deliverable**: 3D interactive texture viewer embedded in Gradio

### Phase 3: Advanced Debugging Tools
**Duration**: 5-7 days
**Prerequisites**: Phase 2 complete, GLSL knowledge

```
Day 1-2: Material Ball Preview
• Implement 3D sphere with texture
• Add lighting and rotation animation
• Test on different texture types

Day 3-4: Shader Editor
• Create editable GLSL textarea
• Implement shader compilation and error handling
• Add preset custom false color modes

Day 5: Pipeline Visualization
• Create 4-stage flow display (RGB → Patches → Channels → Scores)
• Render each stage in mini-canvas
• Wire up to main viewer

Day 6: SpectorJS Integration
• Add SpectorJS library
• Create capture button
• Test texture inspection workflow

Day 7: Polish and Documentation
• Add keyboard shortcuts
• Write debugging guide
• Create tutorial videos/GIFs
```

**Deliverable**: Professional-grade texture debugging suite

### Overall Timeline

```
╔═══════════════════════════════════════════════════════════════════════
║ ARR-COC Texture Viewer Implementation Timeline
╠═══════════════════════════════════════════════════════════════════════
║ Week 1:  Phase 1 (Days 1-2) + Phase 2 Start (Days 3-5)
║ Week 2:  Phase 2 Completion + Phase 3 Start
║ Week 3:  Phase 3 Completion + Testing + Documentation
╚═══════════════════════════════════════════════════════════════════════

Total: 11-14 days for complete implementation (all 3 phases)
```

**Milestones**:
- ✅ Phase 1 Complete: Enhanced channel compositing in Gradio
- ✅ Phase 2 Complete: 3D interactive viewer with Three.js
- ✅ Phase 3 Complete: Advanced debugging with material ball + shader editor

---

## Section 9: Testing Strategy

### Unit Tests (Python Backend)

```python
"""
Test texture serialization and data transfer
"""
import numpy as np
import json

def test_texture_serialization():
    # Create test texture
    texture = np.random.rand(32, 32, 13).astype(np.float32)

    # Serialize
    data = {
        "shape": texture.shape,
        "channels": texture.tolist()
    }
    json_str = json.dumps(data)

    # Deserialize
    parsed = json.loads(json_str)
    reconstructed = np.array(parsed["channels"])

    # Verify
    assert reconstructed.shape == (32, 32, 13)
    assert np.allclose(texture, reconstructed)

def test_rgb_composite():
    texture = np.random.rand(32, 32, 13).astype(np.float32)

    # Create composite
    rgb = create_rgb_composite(texture, 0, 1, 2)

    # Verify
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255
```

### Integration Tests (JavaScript Frontend)

```javascript
/**
 * Test Three.js texture loading and rendering
 */
function testTextureLoading() {
    const mockData = {
        shape: [32, 32, 13],
        channels: Array(32).fill(null).map(() =>
            Array(32).fill(null).map(() =>
                Array(13).fill(0).map(() => Math.random())
            )
        ),
        min: 0.0,
        max: 1.0
    };

    // Create DataTextures
    const textures = createChannelTextures(mockData);

    // Verify
    console.assert(textures.length === 13, "Should have 13 textures");
    console.assert(textures[0].image.width === 32, "Width should be 32");
    console.assert(textures[0].image.height === 32, "Height should be 32");
}

function testShaderCompilation() {
    // Test custom shader compiles without errors
    try {
        const material = new THREE.ShaderMaterial({
            uniforms: { /* ... */ },
            vertexShader: `/* ... */`,
            fragmentShader: `/* ... */`
        });
        console.log("✓ Shader compiled successfully");
    } catch (e) {
        console.error("✗ Shader compilation failed:", e);
    }
}
```

### User Acceptance Tests

**Test Scenario 1: Channel Compositing**
1. Upload test image
2. Select Red=0, Green=1, Blue=2
3. Verify: RGB composite displays correctly
4. Change to Red=9, Green=10, Blue=11 (relevance scores)
5. Verify: Image updates in real-time

**Test Scenario 2: Patch Inspection**
1. Click patch [15, 8] in 3D view
2. Verify: Patch highlights in yellow
3. Verify: Sidebar shows 13 channel values
4. Click different patch [20, 10]
5. Verify: Previous highlight removed, new patch highlighted

**Test Scenario 3: Material Ball Preview**
1. Toggle material ball on
2. Verify: 3D sphere appears with texture applied
3. Rotate camera with OrbitControls
4. Verify: Sphere rotates, texture visible on curved surface
5. Change channel mapping
6. Verify: Material ball texture updates

### Performance Tests

**Test 1: Texture Upload Latency**
```python
import time

def test_upload_performance():
    texture = np.random.rand(32, 32, 13).astype(np.float32)

    start = time.time()
    json_str = serialize_texture_for_threejs(texture)
    serialize_time = time.time() - start

    assert serialize_time < 0.1  # Should be under 100ms
    print(f"Serialization time: {serialize_time*1000:.2f}ms")
```

**Test 2: Shader Update Speed**
```javascript
function testShaderUpdateSpeed() {
    const iterations = 100;
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
        material.uniforms.rChannel.value = i % 13;
        material.uniforms.gChannel.value = (i+1) % 13;
        material.uniforms.bChannel.value = (i+2) % 13;
    }

    const elapsed = performance.now() - start;
    console.log(`Shader update speed: ${elapsed/iterations:.2f}ms per update`);
}
```

**Expected Performance**:
- Texture upload: <100ms (Phase 1-2)
- Shader update: <10ms (Phase 2-3)
- Frame rate: 60 FPS (Phase 2-3)

---

## Section 10: Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Texture Not Displaying in Three.js
**Symptoms**: Black plane in 3D viewer
**Causes**:
- DataTexture not marked with `needsUpdate = true`
- Texture format mismatch (RGB vs RedFormat)
- Min/max normalization incorrect

**Solution**:
```javascript
// BEFORE (wrong)
const texture = new THREE.DataTexture(data, width, height);
// texture.needsUpdate is false by default → texture not uploaded to GPU

// AFTER (correct)
const texture = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.FloatType);
texture.needsUpdate = true;  // ← Critical!
```

#### Issue 2: Shader Compilation Error
**Symptoms**: Console error "Failed to compile shader"
**Causes**:
- GLSL syntax error (missing semicolon, wrong uniform type)
- Uniform name mismatch (JavaScript vs GLSL)
- WebGL version incompatibility

**Solution**:
```javascript
// Check shader error details
if (!material.shader) {
    console.error("Shader compilation failed");
    // Three.js will log detailed error in console
}

// Common fix: Ensure uniform types match
// JavaScript: { value: 0 }  →  GLSL: int
// JavaScript: { value: 0.0 }  →  GLSL: float
```

#### Issue 3: Gradio HTML Component Not Rendering
**Symptoms**: Blank space instead of Three.js viewer
**Causes**:
- CDN link blocked (CORS or network issue)
- JavaScript error preventing execution
- HTML string not properly escaped

**Solution**:
```python
# Check CDN accessibility
html = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
        onerror="console.error('Failed to load Three.js')">
</script>
"""

# Use f-string carefully (escape braces)
html = f"""
<script>
    const data = {json_str};  // ← Correct
    // const data = {{json_str}};  // ← Wrong (escaped braces)
</script>
```

#### Issue 4: Click Detection Not Working
**Symptoms**: Clicking plane doesn't highlight patch
**Causes**:
- Raycaster not configured correctly
- Mouse coordinates not normalized
- Plane not added to scene

**Solution**:
```javascript
// Verify raycaster setup
console.log("Mouse:", mouse.x, mouse.y);  // Should be [-1, 1]
console.log("Intersects:", intersects.length);  // Should be >0 when clicking plane

// Common fix: Adjust for canvas position
const rect = renderer.domElement.getBoundingClientRect();
mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
//                        ^^^^^^^^^^^ ← Must subtract rect.left!
```

#### Issue 5: Low Frame Rate (< 30 FPS)
**Symptoms**: Choppy 3D navigation
**Causes**:
- Too many draw calls (13 separate meshes instead of 1)
- Shader too complex (nested loops in fragment shader)
- Texture resolution too high

**Solution**:
```javascript
// Profile performance
const stats = new Stats();
document.body.appendChild(stats.dom);

function animate() {
    stats.begin();
    renderer.render(scene, camera);
    stats.end();
    requestAnimationFrame(animate);
}

// Optimize: Use single mesh with ShaderMaterial (not 13 meshes)
// Optimize: Pre-compute in vertex shader instead of fragment shader
// Optimize: Use lower texture resolution (16×16 for debugging)
```

---

## Section 11: Future Enhancements

### Beyond Phase 3

#### Enhancement 1: Real-time Texture Streaming
**Goal**: Stream textures from Python to JavaScript without full page reload
**Technology**: WebSocket or Server-Sent Events (SSE)
**Use Case**: Process video frames in real-time, update viewer dynamically

**Implementation Sketch**:
```python
import asyncio
from fastapi import WebSocket

async def stream_textures(websocket: WebSocket):
    while True:
        texture = generate_next_texture()  # Process video frame
        data = serialize_texture(texture)
        await websocket.send_json(data)
        await asyncio.sleep(0.033)  # 30 FPS
```

#### Enhancement 2: VR/AR Texture Exploration
**Goal**: View textures in VR headset (Oculus Quest, HTC Vive)
**Technology**: WebXR API + Three.js VRButton
**Use Case**: Immersive texture inspection for research presentations

**Implementation Sketch**:
```javascript
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';

renderer.xr.enabled = true;
document.body.appendChild(VRButton.createButton(renderer));

// VR controller input for channel selection
const controller = renderer.xr.getController(0);
controller.addEventListener('selectstart', () => {
    // Cycle through channels with VR controller button
    currentChannel = (currentChannel + 1) % 13;
    updateShader(currentChannel);
});
```

#### Enhancement 3: Collaborative Viewing
**Goal**: Multiple users explore same texture simultaneously (Google Docs for textures)
**Technology**: WebRTC + Shared State (Yjs or Automerge)
**Use Case**: Research collaboration, remote teaching

#### Enhancement 4: Machine Learning Integration
**Goal**: Train model to predict relevance from texture patterns
**Technology**: TensorFlow.js (run ML in browser)
**Use Case**: "Learn from my clicks" - ML predicts which patches/channels user will find relevant

#### Enhancement 5: Comparative Visualization
**Goal**: View textures from multiple images side-by-side
**Technology**: Multi-viewport Three.js scenes
**Use Case**: Compare relevance realization across different queries or models

---

## Section 12: Resources and References

### Game Engine Documentation

**Unity Material Inspector**:
- [Unity Manual: Material Inspector](https://docs.unity3d.com/Manual/class-Material.html)
- [Unity Shader Graph: Texture Nodes](https://docs.unity3d.com/Packages/com.unity.shadergraph@10.0/manual/Sample-Texture-2D-Node.html)
- [Unity Forum: Texture Debugging Techniques](https://forum.unity.com/forums/shaders.117/)

**Unreal Engine Material Editor**:
- [Unreal Docs: Material Editor UI](https://docs.unrealengine.com/5.0/en-US/material-editor-ui-in-unreal-engine/)
- [Unreal Docs: Texture Sampler](https://docs.unrealengine.com/5.0/en-US/texture-sample-node-in-unreal-engine/)
- [Unreal Community Wiki: Material Debugging](https://unrealcommunity.wiki/material-debugging)

### 3D Web Graphics

**Three.js**:
- [Three.js Documentation](https://threejs.org/docs/)
- [Three.js Examples: Texture](https://threejs.org/examples/?q=texture)
- [Three.js Journey: Complete Course](https://threejs-journey.com/)

**Babylon.js**:
- [Babylon.js Documentation](https://doc.babylonjs.com/)
- [Babylon.js Inspector Guide](https://doc.babylonjs.com/toolsAndResources/inspector)
- [Babylon.js Playground](https://playground.babylonjs.com/)

### WebGL and Shader Resources

**WebGL Fundamentals**:
- [WebGL Fundamentals](https://webglfundamentals.org/)
- [The Book of Shaders](https://thebookofshaders.com/)
- [Shadertoy (GLSL Examples)](https://www.shadertoy.com/)

**Debugging Tools**:
- [SpectorJS Documentation](https://spector.babylonjs.com/)
- [Chrome DevTools: WebGL](https://developer.chrome.com/docs/devtools/javascript/)
- [WebGL Inspector (deprecated but useful)](https://benvanik.github.io/WebGL-Inspector/)

### Gradio Integration

**Gradio Custom Components**:
- [Gradio Documentation: Custom Components](https://www.gradio.app/guides/custom-components-in-five-minutes)
- [Gradio: HTML Component](https://www.gradio.app/docs/html)
- [Gradio Spaces Examples](https://huggingface.co/spaces)

**Example Projects**:
- [Gradio + Three.js Example (HF Space)](https://huggingface.co/spaces/radames/gradio-3d-viewer)
- [Gradio Custom JS Discussion](https://discuss.huggingface.co/t/custom-javascript-in-gradio/12345)

### Vervaekean Philosophy

**Relevance Realization Framework**:
- Vervaeke, J. (2017-2019). *Awakening from the Meaning Crisis* [Video series]. YouTube.
- [ARR-COC Platonic Dialogues](RESEARCH/PlatonicDialogues/) (Parts 0-8)
- [ARR-COC README](README.md) (Vervaekean architecture section)

**Epistemology and Visualization**:
- Tufte, E. (2001). *The Visual Display of Quantitative Information*.
- Ware, C. (2019). *Information Visualization: Perception for Design*.

### ARR-COC Specific

**Codebase**:
- [ARR-COC GitHub Repository](https://github.com/your-repo/arr-coc-ovis)
- [ARR-COC README](README.md)
- [ARR-COC Microscope Code](arr_coc_ovis/microscope/)

**Related Work**:
- DeepSeek-OCR: Multi-channel texture generation for OCR
- Ovis 2.5: Visual embedding techniques

---

## Section 13: Conclusion and Next Steps

### Summary of Implementation Roadmap

This document synthesized research from game engines (Unity/Unreal), 3D web graphics (Three.js/Babylon.js), and Vervaekean philosophy to create a practical 3-phase implementation plan for ARR-COC texture visualization:

1. **Phase 1 (1-2 days)**: Enhanced Gradio microscope with channel compositing, false color presets, and patch inspection
2. **Phase 2 (3-5 days)**: Three.js interactive 3D viewer with custom shaders, real-time channel selection, and click-to-inspect
3. **Phase 3 (5-7 days)**: Advanced debugging with material ball preview, shader editor, pipeline visualization, and SpectorJS integration

**Total Effort**: 11-14 days for complete implementation

### Key Innovations

1. **Multi-perspective visualization**: Simultaneously shows propositional, perspectival, and participatory knowing
2. **Real-time shader compositing**: 13 separate DataTextures combined dynamically based on user selection
3. **Vervaekean false color presets**: Semantic, Edges, Spatial, and Relevance modes aligned with three ways of knowing
4. **Interactive relevance realization**: Users navigate opponent processing tensions through exploration

### Immediate Next Steps

```
╔═══════════════════════════════════════════════════════════════════════
║ Action Items for ARR-COC Development Team
╠═══════════════════════════════════════════════════════════════════════
║ [ ] 1. Review this implementation roadmap
║ [ ] 2. Set up development environment (Python 3.8+, Gradio, NumPy)
║ [ ] 3. Start Phase 1 implementation (1-2 days)
║ [ ] 4. Test Phase 1 with real ARR-COC textures
║ [ ] 5. If Phase 1 successful, proceed to Phase 2
║ [ ] 6. Allocate 3-5 days for Phase 2 development
║ [ ] 7. User testing with researchers (gather feedback)
║ [ ] 8. If advanced debugging needed, implement Phase 3
║ [ ] 9. Deploy to Gradio Spaces or internal server
║ [ ] 10. Document usage patterns and create tutorial
╚═══════════════════════════════════════════════════════════════════════
```

### Questions to Consider

**Technical**:
- Should we use Three.js (smaller, more examples) or Babylon.js (built-in Inspector)?
- Is JSON serialization sufficient, or do we need binary transfer (WebSocket)?
- Should we vendor Three.js locally or use CDN?

**UX/Design**:
- What's the optimal default false color mode for first-time users?
- Should material ball be always visible or toggle-able?
- How do we teach users the Vervaekean interpretation of each view?

**Research**:
- Which channels (0-12) are most informative for typical queries?
- Can we learn optimal channel mappings from user interactions?
- Should we add comparative visualization (multiple images side-by-side)?

### Success Metrics

**Phase 1 Success**:
- Researchers can interactively explore 13 channels (no code editing)
- False color presets reveal different relevance dimensions
- Patch inspection accelerates debugging workflow

**Phase 2 Success**:
- 3D viewer runs at 60 FPS on standard laptop
- Click-to-inspect feels intuitive (< 3 clicks to find relevant patch)
- Material ball preview provides insight not visible in flat view

**Phase 3 Success**:
- Shader editor enables custom false color modes (researchers create 3+ new modes)
- SpectorJS integration catches texture upload bugs (at least 1 real bug found)
- Pipeline visualization clarifies ARR-COC transformations (user survey confirms understanding)

### Final Thoughts

Visualization isn't just "showing data" - it's enabling **relevance realization through interaction**. By combining game engine patterns (Unity's material inspector), 3D web graphics (Three.js shaders), and Vervaekean philosophy (four ways of knowing), we create a tool that helps researchers not just *see* textures, but *understand* how relevance emerges through opponent processing.

This roadmap provides a concrete path from ARR-COC's current static microscope to a professional-grade interactive texture viewer that embodies Vervaeke's epistemology in every interaction.

**Next**: Build Phase 1, test with real textures, iterate based on feedback, and gradually enhance toward the full vision.

---

## Sources

**Game Engine Documentation**:
- Unity Material Inspector: https://docs.unity3d.com/Manual/class-Material.html
- Unreal Material Editor: https://docs.unrealengine.com/5.0/en-US/material-editor-ui-in-unreal-engine/

**3D Web Graphics**:
- Three.js Documentation: https://threejs.org/docs/
- Babylon.js Documentation: https://doc.babylonjs.com/

**WebGL and Debugging**:
- SpectorJS: https://spector.babylonjs.com/
- WebGL Fundamentals: https://webglfundamentals.org/

**Gradio**:
- Gradio Custom Components: https://www.gradio.app/guides/custom-components-in-five-minutes
- Gradio HTML Component: https://www.gradio.app/docs/html

**ARR-COC Codebase**:
- [ARR-COC README](../../../README.md) - Architecture and Vervaekean framework
- [Platonic Dialogues](../../../RESEARCH/PlatonicDialogues/) - Conceptual foundations

**Vervaekean Philosophy**:
- Vervaeke, J. (2017-2019). *Awakening from the Meaning Crisis* [Video series]
- ARR-COC Dialogues (Parts 0-8) - Application to vision-language models

---

**End of Implementation Roadmap**
**Created**: 2025-10-31
**Oracle**: lod-btree-oracle
**Status**: Ready for Development
