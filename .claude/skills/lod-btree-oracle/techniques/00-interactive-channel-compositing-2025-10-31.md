# Interactive Channel Compositing UI Patterns (2025)

**Created**: 2025-10-31
**Topic**: UI/UX design patterns for multi-channel texture visualization and compositing
**Application**: ARR-COC 13-channel texture array inspection and debugging

---

## Overview

Interactive channel compositing interfaces allow users to select, combine, and visualize individual channels from multi-channel images. Professional tools like Photoshop, Blender, and Substance Designer have refined these patterns over decades. This document analyzes their UI/UX approaches and applies them to ARR-COC's 13-channel texture arrays.

**Key Challenge**: ARR-COC generates 13-channel texture arrays (Propositional, Perspectival, Participatory dimensions). Standard RGB displays can only show 3 channels simultaneously, requiring sophisticated UI for channel exploration.

---

## Section 1: Channel Selection UI Patterns

### 1.1 Photoshop Channel Mixer Approach

From [Adobe Photoshop Channel Basics](https://helpx.adobe.com/ca/photoshop/using/channel-basics.html) (accessed 2025-10-31):

**Interface Components:**
- **Channels Panel**: Vertical list of all channels (composite + individual channels)
- **Visibility Toggles**: Eye icon (👁) in left column to show/hide channels
- **Channel Selection**: Click channel name to select for editing
- **Multi-select**: Shift-click to select multiple channels
- **Composite View**: When all channels visible → automatic composite display

**Interaction Patterns:**
```
Channel Panel Layout:
╔═══════════════════════════
║ 👁 RGB (composite)
╟──────────────────────────
║ 👁 Red
║ 👁 Green
║ 👁 Blue
╟──────────────────────────
║ 👁 Alpha 1
║   Alpha 2 (hidden)
```

**Key Features:**
1. **Visual Thumbnails**: Small preview of each channel next to name
2. **Color-coded Names**: Channels displayed in their representative color
3. **Keyboard Shortcuts**: Ctrl+1/2/3 for RGB channels
4. **Drag-and-Drop**: Reorder alpha/spot channels in panel
5. **Context Menu**: Right-click for channel operations

**Strengths:**
- Simple, well-understood interface pattern
- Clear visual hierarchy (composite → individual channels)
- Immediate visual feedback (eye icons)
- Compact vertical layout saves screen space

**Limitations for ARR-COC:**
- Designed for 4-5 channels (RGB + alpha), not 13
- No built-in false color modes for semantic visualization
- Limited batch comparison capabilities

### 1.2 Substance Designer Channel Display

From [Substance Designer Interface Overview](https://helpx.adobe.com/substance-3d-designer/using/interface-overview.html) (accessed 2025-10-31):

**2D Viewport Controls:**
```
Lower-Left Control Strip:
[R] [G] [B]  [◧] [⬚/🎨] [⊞] [📏] [📊] [🌈] [⚛]
 ↑    ↑    ↑    ↑      ↑     ↑    ↑    ↑    ↑    ↑
RGB  Transp Color Tile Size Info Hist sRGB Premul
toggles      mode
```

**Channel Toggle Buttons:**
- **Independent R/G/B Toggles**: Best used in grayscale mode
- **Visual Feedback**: Active channels highlighted
- **Combine with Output Selector**: Dropdown menu for switching output maps
- **Real-time Updates**: Changes apply immediately to viewport

**Output Selection Dropdown:**
```
[▼ Select Output     ]
├─ Base Color
├─ Roughness
├─ Metallic
├─ Normal
├─ Height
└─ Ambient Occlusion
```

**Strengths:**
- Minimal UI footprint (icon buttons)
- Direct manipulation (toggle = instant preview)
- Separate concerns: output selection vs channel display
- Optimized for material textures (multiple output maps)

**Adaptations for ARR-COC:**
- Replace "Output" dropdown with 13-channel selector
- Add false color mode presets (Semantic/Edges/Spatial)
- Retain RGB toggle pattern for channel compositing

### 1.3 Blender Compositor Node-Based Approach

From web research on Blender compositor (accessed 2025-10-31):

**Node-Based Channel Operations:**
```
[Texture Input] → [Separate RGB] → [R] [G] [B]
                                     ↓   ↓   ↓
                                   [Mix/Blend Nodes]
                                          ↓
                                   [Viewer Output]
```

**UI Patterns:**
- **Separate Channels Node**: Splits RGB into individual outputs
- **Combine Channels Node**: Merges R/G/B inputs into RGB output
- **Channel Swizzling**: Connect any channel to any output socket
- **Visual Flow**: See data flow from source → operations → output

**Strengths:**
- Extreme flexibility (arbitrary channel routing)
- Visual programming paradigm (clear data flow)
- Composable operations (stack channel operations)
- Power user friendly (complex workflows possible)

**Limitations for ARR-COC:**
- Steep learning curve for casual users
- Overkill for simple "view channel X" tasks
- Requires more screen real estate

---

## Section 2: Real-time Compositing Feedback

### 2.1 Live Preview During Channel Mixing

**Pattern**: Immediate visual update when channel selection changes

**Implementation Approaches:**

**A. Photoshop Channel Mixer Adjustment:**
```
Channel Mixer Dialog:
╔═══════════════════════════════
║ Output Channel: [Red ▼]
╟──────────────────────────────
║ Red:    +100%  ████░░░░░░
║ Green:    +0%  ░░░░░░░░░░
║ Blue:     +0%  ░░░░░░░░░░
║ Constant:  0%  ░░░░░░░░░░
╟──────────────────────────────
║ [Preview ✓]  [OK] [Cancel]
```

- **Slider Adjustments**: Drag slider → image updates in real-time
- **Preview Checkbox**: Toggle to compare before/after
- **Blend Percentages**: Mix multiple channels with weights

**B. Substance Designer Real-time Shader Updates:**
- **GPU-Accelerated Preview**: Changes render in <16ms (60 FPS)
- **Delayed Computation**: Heavy nodes compute on mouse-up (not drag)
- **Progressive Rendering**: Start with low-res preview, refine to full resolution
- **Cached Results**: Unchanged nodes use cached output

**Performance Optimization Strategies:**
1. **Debounce UI Events**: Update preview 100ms after last slider change
2. **Resolution Scaling**: Preview at 512x512, full quality on demand
3. **WebGL Shaders**: Offload channel compositing to GPU
4. **Lazy Evaluation**: Only compute visible channels

### 2.2 Split-View Comparisons

**Before/After Comparison Patterns:**

**Pattern A: Horizontal Split**
```
╔════════════════════╦════════════════════╗
║                    ║                    ║
║   BEFORE           ║   AFTER            ║
║   (Original)       ║   (Composited)     ║
║                    ║                    ║
╚════════════════════╩════════════════════╝
        [←═══╪═══→]  (Draggable divider)
```

**Pattern B: Toggle Switch**
```
[Before ○━━━● After]  ← Toggle button
  (Click to switch instantly)
```

**Pattern C: Overlay/Difference Mode**
```
[Overlay] [Difference] [Side-by-Side]
   ↑          ↑              ↑
  Blend   Show only    Standard split
  modes   differences      view
```

**ARR-COC Application:**
- **Before**: Raw RGB image
- **After**: Selected 13-channel composite (false color)
- **Use Case**: Verify relevance score visualization matches expectations

### 2.3 Histogram Display Per Channel

**Purpose**: Show pixel value distribution for each channel

**Photoshop Histogram Panel:**
```
Histogram (showing Red channel):
╔═══════════════════════════════
║ Channel: [Red ▼]
╟──────────────────────────────
║     ▁▂▃▅▇█▇▅▃▂▁              ← Visual histogram
║   ░░░░░░██████░░░░░░
║ ░░░░░░░░██████░░░░░░░░
║ 0   64   128  192  255        ← Value range
╟──────────────────────────────
║ Mean: 128.5  |  StdDev: 42.3
║ Median: 132  |  Pixels: 262K
```

**Display Options:**
- **Individual Channels**: Histogram for R, G, or B
- **RGB Composite**: Overlaid histograms (colored)
- **Luminosity**: Grayscale histogram of perceived brightness
- **Statistics Panel**: Mean, median, standard deviation, percentiles

**ARR-COC Adaptation:**
```
Channel Histogram (Propositional Channels):
╔═══════════════════════════════
║ [Entropy ▼] [Contrast ▼] [Sharp ▼]
╟──────────────────────────────
║ Entropy Distribution:
║     ▁▃▅█▇▅▃▂▁                 ← Most patches: medium entropy
║
║ Stats:  Mean: 0.65  |  Range: [0.21, 0.94]
```

**Benefits:**
- **Spot Outliers**: Identify unusual channel value distributions
- **Verify Normalization**: Ensure channels use full value range
- **Debug Issues**: Flat histogram = channel likely broken

---

## Section 3: Web Implementation Approaches

### 3.1 HTML/CSS Channel Selector UI

**Basic Channel Selector Component:**

```html
<!-- ARR-COC 13-Channel Selector -->
<div class="channel-selector">
  <!-- Propositional Channels (0-3) -->
  <div class="channel-group">
    <h3>Propositional Knowing</h3>
    <label><input type="checkbox" data-channel="0"> Entropy</label>
    <label><input type="checkbox" data-channel="1"> Contrast</label>
    <label><input type="checkbox" data-channel="2"> Sharpness</label>
    <label><input type="checkbox" data-channel="3"> Edge Density</label>
  </div>

  <!-- Perspectival Channels (4-7) -->
  <div class="channel-group">
    <h3>Perspectival Knowing</h3>
    <label><input type="checkbox" data-channel="4"> Saliency</label>
    <label><input type="checkbox" data-channel="5"> Gist</label>
    <label><input type="checkbox" data-channel="6"> Texture</label>
    <label><input type="checkbox" data-channel="7"> Color Vivid</label>
  </div>

  <!-- Participatory Channels (8-12) -->
  <div class="channel-group">
    <h3>Participatory Knowing</h3>
    <label><input type="checkbox" data-channel="8"> Query Sim</label>
    <label><input type="checkbox" data-channel="9"> Semantic</label>
    <label><input type="checkbox" data-channel="10"> Spatial</label>
    <label><input type="checkbox" data-channel="11"> Attention</label>
    <label><input type="checkbox" data-channel="12"> Coupling</label>
  </div>
</div>
```

**CSS Styling (Material Design inspired):**

```css
.channel-selector {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 8px;
}

.channel-group {
  background: white;
  padding: 12px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.channel-group h3 {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin-bottom: 12px;
  border-bottom: 2px solid #1976d2;
  padding-bottom: 4px;
}

.channel-group label {
  display: block;
  padding: 8px 4px;
  cursor: pointer;
  transition: background 0.2s;
}

.channel-group label:hover {
  background: #f0f0f0;
  border-radius: 2px;
}

.channel-group input[type="checkbox"] {
  margin-right: 8px;
  cursor: pointer;
}

.channel-group input[type="checkbox"]:checked + label {
  font-weight: 500;
  color: #1976d2;
}
```

**Responsive Mobile Layout:**

```css
@media (max-width: 768px) {
  .channel-selector {
    grid-template-columns: 1fr;  /* Stack vertically */
  }

  .channel-group {
    margin-bottom: 12px;
  }
}
```

### 3.2 JavaScript Real-time Updates

**Event-Driven Channel Compositing:**

```javascript
// ARR-COC Channel Compositor
class ChannelCompositor {
  constructor(canvasId, textureData) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.textureData = textureData;  // Float32Array: 13 channels × H × W
    this.activeChannels = [0, 1, 2];  // Default: first 3 channels → RGB
    this.falseColorMode = 'rgb';  // 'rgb', 'semantic', 'edges', 'spatial'
  }

  // Update which channels map to R, G, B
  setChannelMapping(r, g, b) {
    this.activeChannels = [r, g, b];
    this.render();
  }

  // Real-time rendering (called on channel selection change)
  render() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const imageData = this.ctx.createImageData(width, height);

    // Extract selected channels
    const rChannel = this.getChannel(this.activeChannels[0]);
    const gChannel = this.getChannel(this.activeChannels[1]);
    const bChannel = this.getChannel(this.activeChannels[2]);

    // Composite into RGB image
    for (let i = 0; i < width * height; i++) {
      const pixelIdx = i * 4;
      imageData.data[pixelIdx + 0] = rChannel[i] * 255;  // R
      imageData.data[pixelIdx + 1] = gChannel[i] * 255;  // G
      imageData.data[pixelIdx + 2] = bChannel[i] * 255;  // B
      imageData.data[pixelIdx + 3] = 255;                // A (opaque)
    }

    this.ctx.putImageData(imageData, 0, 0);
  }

  // Extract single channel from 13-channel texture
  getChannel(channelIdx) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const channelData = new Float32Array(width * height);

    const channelOffset = channelIdx * width * height;
    for (let i = 0; i < width * height; i++) {
      channelData[i] = this.textureData[channelOffset + i];
    }

    return channelData;
  }

  // Apply false color mode
  setFalseColorMode(mode) {
    this.falseColorMode = mode;
    switch(mode) {
      case 'semantic':
        this.setChannelMapping(9, 10, 11);  // Semantic, Spatial, Attention
        break;
      case 'edges':
        this.setChannelMapping(3, 2, 1);    // EdgeDensity, Sharp, Contrast
        break;
      case 'spatial':
        this.setChannelMapping(10, 10, 10); // Spatial (grayscale)
        break;
      default:  // 'rgb'
        this.setChannelMapping(0, 1, 2);
    }
  }
}

// Event Listeners (attach to checkboxes)
document.querySelectorAll('.channel-selector input[type="checkbox"]').forEach(checkbox => {
  checkbox.addEventListener('change', (e) => {
    const channelIdx = parseInt(e.target.dataset.channel);

    // Update active channels array
    if (e.target.checked) {
      compositor.activeChannels.push(channelIdx);
    } else {
      const idx = compositor.activeChannels.indexOf(channelIdx);
      if (idx > -1) compositor.activeChannels.splice(idx, 1);
    }

    // Re-render with updated channels
    compositor.render();
  });
});

// Debounced rendering for slider-based channel weights
let renderTimeout;
function debouncedRender(compositor) {
  clearTimeout(renderTimeout);
  renderTimeout = setTimeout(() => compositor.render(), 100);  // 100ms delay
}
```

**Performance: GPU-Accelerated WebGL Version:**

```javascript
// WebGL Fragment Shader for Channel Compositing
const channelCompositeShader = `
precision mediump float;

uniform sampler2D u_texture;  // 13-channel texture (as 4×RGBA textures)
uniform vec3 u_channelWeights;  // (R weight, G weight, B weight)
uniform ivec3 u_channelIndices;  // Which channels to display

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;

  // Sample selected channels (simplified, actual needs multi-texture lookup)
  float r = texture2D(u_texture, uv).r;  // Channel u_channelIndices.r
  float g = texture2D(u_texture, uv).g;  // Channel u_channelIndices.g
  float b = texture2D(u_texture, uv).b;  // Channel u_channelIndices.b

  // Apply weights and output
  gl_FragColor = vec4(r * u_channelWeights.r,
                      g * u_channelWeights.g,
                      b * u_channelWeights.b,
                      1.0);
}
`;

// Benefits: 60 FPS real-time updates, handles 4K textures smoothly
```

### 3.3 Canvas API Compositing Visualization

**Basic Canvas Pattern:**

```javascript
// Draw channel composite on HTML5 Canvas
function drawChannelComposite(canvas, channels, mapping) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Create ImageData buffer
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  // Iterate over pixels
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIdx = (y * width + x) * 4;

      // Map selected channels to RGB
      data[pixelIdx + 0] = channels[mapping.r][y][x] * 255;  // R
      data[pixelIdx + 1] = channels[mapping.g][y][x] * 255;  // G
      data[pixelIdx + 2] = channels[mapping.b][y][x] * 255;  // B
      data[pixelIdx + 3] = 255;  // Alpha (fully opaque)
    }
  }

  // Draw to canvas
  ctx.putImageData(imageData, 0, 0);
}

// Example usage
const channelMapping = { r: 9, g: 10, b: 11 };  // Semantic, Spatial, Attention
drawChannelComposite(myCanvas, arrCocChannels, channelMapping);
```

**Advanced: Color Mapping with Legends:**

```javascript
// Apply false color LUT (Look-Up Table)
function applyFalseColorLUT(value, mode) {
  switch(mode) {
    case 'heatmap':  // Blue → Red (cold to hot)
      return {
        r: Math.min(255, value * 512),
        g: Math.min(255, Math.abs((value - 0.5) * 512)),
        b: Math.min(255, (1 - value) * 512)
      };

    case 'viridis':  // Perceptually uniform colormap
      // Simplified viridis approximation
      const r = 0.267 + value * (0.993 - 0.267);
      const g = 0.005 + value * (0.906 - 0.005);
      const b = 0.329 + value * (0.144 - 0.329);
      return { r: r * 255, g: g * 255, b: b * 255 };

    default:  // Grayscale
      const gray = value * 255;
      return { r: gray, g: gray, b: gray };
  }
}

// Draw with legend
function drawWithLegend(canvas, channelData, colorMode) {
  // Main visualization
  drawChannelComposite(canvas, channelData, colorMode);

  // Draw color bar legend (vertical gradient)
  const legendX = canvas.width - 50;
  const legendY = 20;
  const legendHeight = 200;

  for (let i = 0; i < legendHeight; i++) {
    const value = i / legendHeight;
    const color = applyFalseColorLUT(value, colorMode);
    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    ctx.fillRect(legendX, legendY + legendHeight - i, 30, 1);
  }

  // Label endpoints
  ctx.fillStyle = '#000';
  ctx.font = '12px monospace';
  ctx.fillText('1.0', legendX + 35, legendY + 10);
  ctx.fillText('0.0', legendX + 35, legendY + legendHeight);
}
```

---

## Section 4: ARR-COC Channel Compositor Design

### 4.1 UI Mockup: 13-Channel Selector

**Visual Design:**

```
╔═════════════════════════════════════════════════════════════════
║ ARR-COC Texture Microscope - Channel Compositor
╟─────────────────────────────────────────────────────────────────
║
║ ┌─ Propositional Knowing ─────────────────┐
║ │ ☑ [0] Entropy        ☐ [1] Contrast    │
║ │ ☑ [2] Sharpness      ☐ [3] Edge Dens   │
║ └─────────────────────────────────────────┘
║
║ ┌─ Perspectival Knowing ──────────────────┐
║ │ ☐ [4] Saliency       ☑ [5] Gist        │
║ │ ☐ [6] Texture        ☐ [7] Vivid       │
║ └─────────────────────────────────────────┘
║
║ ┌─ Participatory Knowing ─────────────────┐
║ │ ☐ [8] Query Sim      ☑ [9] Semantic    │
║ │ ☐ [10] Spatial       ☐ [11] Attention  │
║ │ ☐ [12] Coupling                         │
║ └─────────────────────────────────────────┘
║
║ False Color Mode: [Semantic ▼]  [Custom RGB]
║
║ RGB Mapping:
║   R ← Channel: [0 ▼] (Entropy)      Weight: ████████░░ 80%
║   G ← Channel: [5 ▼] (Gist)         Weight: ██████░░░░ 60%
║   B ← Channel: [9 ▼] (Semantic)     Weight: ██████████ 100%
║
║ ┌─ Live Preview Canvas ──────────────────────────────────┐
║ │                                                         │
║ │   [Real-time rendered channel composite displays here] │
║ │                                                         │
║ │   Updates on checkbox change or slider drag            │
║ │                                                         │
║ └─────────────────────────────────────────────────────────┘
║
║ [Export Composite] [Reset to RGB] [Save Preset]
╚═════════════════════════════════════════════════════════════════
```

### 4.2 False Color Mode Dropdown

**Preset False Color Modes for ARR-COC:**

```javascript
const falseColorPresets = {
  // Preset 1: Semantic Relevance (highlights query-aware channels)
  'semantic': {
    name: 'Semantic Relevance',
    description: 'Query coupling + semantic similarity + spatial relevance',
    mapping: { r: 12, g: 9, b: 10 },  // Coupling, Semantic, Spatial
    weights: { r: 1.0, g: 0.8, b: 0.6 }
  },

  // Preset 2: Edge Detection (propositional sharpness signals)
  'edges': {
    name: 'Edge & Contrast',
    description: 'Edge density + sharpness + contrast',
    mapping: { r: 3, g: 2, b: 1 },  // EdgeDens, Sharp, Contrast
    weights: { r: 1.0, g: 0.7, b: 0.5 }
  },

  // Preset 3: Salience (what stands out perceptually)
  'salience': {
    name: 'Perceptual Salience',
    description: 'Saliency + color vividness + texture',
    mapping: { r: 4, g: 7, b: 6 },  // Saliency, Vivid, Texture
    weights: { r: 1.0, g: 0.8, b: 0.6 }
  },

  // Preset 4: Information Content (entropy-based)
  'information': {
    name: 'Information Density',
    description: 'Entropy + contrast + gist',
    mapping: { r: 0, g: 1, b: 5 },  // Entropy, Contrast, Gist
    weights: { r: 1.0, g: 0.7, b: 0.5 }
  },

  // Preset 5: Attention Flow (participatory coupling)
  'attention': {
    name: 'Attention Coupling',
    description: 'Attention + query similarity + coupling',
    mapping: { r: 11, g: 8, b: 12 },  // Attention, QuerySim, Coupling
    weights: { r: 1.0, g: 0.8, b: 1.0 }
  }
};

// Apply preset
function applyPreset(presetName) {
  const preset = falseColorPresets[presetName];
  compositor.setChannelMapping(
    preset.mapping.r,
    preset.mapping.g,
    preset.mapping.b
  );
  compositor.setWeights(preset.weights);
  compositor.render();

  // Update UI
  document.getElementById('preset-description').textContent = preset.description;
}
```

**Dropdown HTML:**

```html
<label for="false-color-mode">False Color Mode:</label>
<select id="false-color-mode" onchange="applyPreset(this.value)">
  <option value="semantic">Semantic Relevance</option>
  <option value="edges">Edge & Contrast</option>
  <option value="salience">Perceptual Salience</option>
  <option value="information">Information Density</option>
  <option value="attention">Attention Coupling</option>
  <option value="custom">Custom RGB Mapping</option>
</select>

<p id="preset-description" class="description-text">
  Query coupling + semantic similarity + spatial relevance
</p>
```

### 4.3 RGB Mapping Controls

**Slider-Based Channel Weight UI:**

```html
<!-- Channel Mapping Control (one per RGB component) -->
<div class="channel-mapping">
  <label>R ← Channel:</label>
  <select id="r-channel" onchange="updateMapping('r', this.value)">
    <option value="0">0: Entropy</option>
    <option value="1">1: Contrast</option>
    <option value="2">2: Sharpness</option>
    <!-- ... all 13 channels ... -->
    <option value="12">12: Coupling</option>
  </select>

  <label>Weight:</label>
  <input type="range" id="r-weight" min="0" max="100" value="100"
         oninput="updateWeight('r', this.value / 100)">
  <span id="r-weight-display">100%</span>

  <div class="weight-bar">
    <div id="r-weight-fill" style="width: 100%; background: #ff5555;"></div>
  </div>
</div>

<!-- Repeat for G and B channels -->
```

**CSS for Weight Visualization:**

```css
.weight-bar {
  width: 200px;
  height: 20px;
  background: #ddd;
  border-radius: 10px;
  overflow: hidden;
  margin-top: 4px;
}

.weight-bar div {
  height: 100%;
  transition: width 0.2s ease;
}

#r-weight-fill { background: linear-gradient(90deg, #ff5555, #ff0000); }
#g-weight-fill { background: linear-gradient(90deg, #55ff55, #00ff00); }
#b-weight-fill { background: linear-gradient(90deg, #5555ff, #0000ff); }
```

### 4.4 Live Preview Canvas

**Implementation: Real-time Update on Channel Change**

```javascript
class ARRCOCChannelViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    // Texture data: Float32Array[13][H][W]
    this.textureChannels = null;
    this.width = 0;
    this.height = 0;

    // Current mapping state
    this.mapping = { r: 0, g: 1, b: 2 };
    this.weights = { r: 1.0, g: 1.0, b: 1.0 };

    // Performance
    this.renderPending = false;
  }

  // Load 13-channel texture data
  loadTexture(channelData, width, height) {
    this.textureChannels = channelData;
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.render();
  }

  // Update channel mapping (called from UI)
  setMapping(component, channelIdx) {
    this.mapping[component] = channelIdx;
    this.scheduleRender();
  }

  // Update channel weight (called from sliders)
  setWeight(component, weight) {
    this.weights[component] = weight;
    this.scheduleRender();
  }

  // Debounced render (performance optimization)
  scheduleRender() {
    if (!this.renderPending) {
      this.renderPending = true;
      requestAnimationFrame(() => {
        this.render();
        this.renderPending = false;
      });
    }
  }

  // Core rendering function
  render() {
    if (!this.textureChannels) return;

    const imageData = this.ctx.createImageData(this.width, this.height);
    const data = imageData.data;

    // Extract selected channels
    const rChan = this.textureChannels[this.mapping.r];
    const gChan = this.textureChannels[this.mapping.g];
    const bChan = this.textureChannels[this.mapping.b];

    // Composite with weights
    for (let i = 0; i < this.width * this.height; i++) {
      const pixelIdx = i * 4;
      data[pixelIdx + 0] = rChan[i] * this.weights.r * 255;
      data[pixelIdx + 1] = gChan[i] * this.weights.g * 255;
      data[pixelIdx + 2] = bChan[i] * this.weights.b * 255;
      data[pixelIdx + 3] = 255;
    }

    this.ctx.putImageData(imageData, 0, 0);

    // Optional: Draw channel labels overlay
    this.drawChannelLabels();
  }

  // Overlay channel names on canvas
  drawChannelLabels() {
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(0, 0, 200, 60);

    this.ctx.fillStyle = '#ff5555';
    this.ctx.font = '14px monospace';
    this.ctx.fillText(`R: Channel ${this.mapping.r} (${this.weights.r.toFixed(2)}x)`, 5, 15);

    this.ctx.fillStyle = '#55ff55';
    this.ctx.fillText(`G: Channel ${this.mapping.g} (${this.weights.g.toFixed(2)}x)`, 5, 32);

    this.ctx.fillStyle = '#5555ff';
    this.ctx.fillText(`B: Channel ${this.mapping.b} (${this.weights.b.toFixed(2)}x)`, 5, 49);
  }
}

// Initialization
const viewer = new ARRCOCChannelViewer('channel-canvas');

// Load texture from ARR-COC model output
fetch('/api/arr-coc/texture?patch=15')
  .then(r => r.json())
  .then(data => {
    viewer.loadTexture(data.channels, data.width, data.height);
  });
```

---

## Section 5: Advanced Features

### 5.1 Patch Inspector (Click-to-Inspect)

**Interactive Patch Grid:**

```javascript
// Click on 32×32 patch grid → show 13-channel breakdown
class PatchGridInspector {
  constructor(gridCanvas, detailCanvas) {
    this.gridCanvas = gridCanvas;
    this.detailCanvas = detailCanvas;
    this.selectedPatch = null;

    // Attach click handler
    this.gridCanvas.addEventListener('click', (e) => this.handleClick(e));
  }

  handleClick(event) {
    const rect = this.gridCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Calculate which patch was clicked (assuming 32×32 grid)
    const patchX = Math.floor(x / (this.gridCanvas.width / 32));
    const patchY = Math.floor(y / (this.gridCanvas.height / 32));
    const patchIdx = patchY * 32 + patchX;

    this.selectedPatch = patchIdx;
    this.showPatchDetails(patchIdx);
    this.highlightPatch(patchX, patchY);
  }

  highlightPatch(x, y) {
    const ctx = this.gridCanvas.getContext('2d');
    const patchSize = this.gridCanvas.width / 32;

    // Clear previous highlight
    this.redrawGrid();

    // Draw highlight border
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 3;
    ctx.strokeRect(x * patchSize, y * patchSize, patchSize, patchSize);
  }

  showPatchDetails(patchIdx) {
    // Fetch patch's 13-channel data
    fetch(`/api/arr-coc/patch/${patchIdx}`)
      .then(r => r.json())
      .then(data => {
        this.renderChannelBreakdown(data.channels);
      });
  }

  renderChannelBreakdown(channels) {
    const ctx = this.detailCanvas.getContext('2d');
    const width = this.detailCanvas.width;
    const height = this.detailCanvas.height / 13;  // Stack 13 channels vertically

    for (let i = 0; i < 13; i++) {
      const channelImage = this.channelToImage(channels[i]);
      ctx.putImageData(channelImage, 0, i * height);

      // Label channel
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px monospace';
      ctx.fillText(`Ch${i}: ${this.getChannelName(i)}`, 5, i * height + 15);
    }
  }

  getChannelName(idx) {
    const names = [
      'Entropy', 'Contrast', 'Sharpness', 'EdgeDens',
      'Saliency', 'Gist', 'Texture', 'Vivid',
      'QuerySim', 'Semantic', 'Spatial', 'Attention', 'Coupling'
    ];
    return names[idx];
  }
}
```

### 5.2 Channel Difference Visualization

**Show Difference Between Two Channel States:**

```javascript
// Visualize diff: (Channel A) - (Channel B)
function visualizeDifference(channelA, channelB, canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const data = imageData.data;

  for (let i = 0; i < channelA.length; i++) {
    const diff = channelA[i] - channelB[i];

    // Red = positive diff, Blue = negative diff
    const pixelIdx = i * 4;
    if (diff > 0) {
      data[pixelIdx + 0] = Math.min(255, diff * 512);  // R
      data[pixelIdx + 1] = 0;                          // G
      data[pixelIdx + 2] = 0;                          // B
    } else {
      data[pixelIdx + 0] = 0;                          // R
      data[pixelIdx + 1] = 0;                          // G
      data[pixelIdx + 2] = Math.min(255, -diff * 512); // B
    }
    data[pixelIdx + 3] = 255;  // Alpha
  }

  ctx.putImageData(imageData, 0, 0);
}

// Use case: Compare Semantic channel before/after query change
visualizeDifference(semanticBefore, semanticAfter, diffCanvas);
```

### 5.3 Export & Save Functionality

**Export Composite as PNG:**

```javascript
function exportComposite(canvas, filename) {
  // Convert canvas to blob
  canvas.toBlob((blob) => {
    // Create download link
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || 'arr-coc-composite.png';

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Cleanup
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// Save current view
document.getElementById('export-btn').addEventListener('click', () => {
  const timestamp = new Date().toISOString().replace(/:/g, '-');
  exportComposite(viewer.canvas, `arr-coc-${timestamp}.png`);
});
```

**Save Preset Configuration:**

```javascript
// Save current channel mapping as preset
function savePreset(name) {
  const preset = {
    name: name,
    mapping: { ...viewer.mapping },
    weights: { ...viewer.weights },
    timestamp: new Date().toISOString()
  };

  // Save to localStorage
  const presets = JSON.parse(localStorage.getItem('arr-coc-presets') || '[]');
  presets.push(preset);
  localStorage.setItem('arr-coc-presets', JSON.stringify(presets));

  // Update UI
  addPresetToDropdown(preset);
}

// Load preset
function loadPreset(presetName) {
  const presets = JSON.parse(localStorage.getItem('arr-coc-presets') || '[]');
  const preset = presets.find(p => p.name === presetName);

  if (preset) {
    viewer.mapping = preset.mapping;
    viewer.weights = preset.weights;
    viewer.render();
    updateUIControls(preset);
  }
}
```

---

## Sources

### Source Documents
- None (web research based)

### Web Research
- [Adobe Photoshop Channel Basics](https://helpx.adobe.com/ca/photoshop/using/channel-basics.html) - Photoshop Channels Panel UI (accessed 2025-10-31)
- [Substance Designer Interface Overview](https://helpx.adobe.com/substance-3d-designer/using/interface-overview.html) - 2D/3D viewport controls (accessed 2025-10-31)
- [Blender Compositor Documentation](https://docs.blender.org/manual/en/latest/compositing/introduction.html) - Node-based compositing (accessed 2025-10-31)

### Additional References
- Photoshop Channel Mixer: Google search "Photoshop channel mixer UI design interface" (accessed 2025-10-31)
- Blender Compositor Nodes: Google search "Blender compositor nodes texture channels interface" (accessed 2025-10-31)
- Substance Designer Patterns: Google search "Substance Designer channel operations UI patterns" (accessed 2025-10-31)
- Web Image Editors: Google search "web-based image editor channel selector interface 2024 2025" (accessed 2025-10-31)

---

**End of Document**
