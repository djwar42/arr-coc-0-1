# Texture Atlas Visualization Techniques

**Date**: 2025-10-31
**Topic**: Interactive texture atlas visualization tools and patterns from game development
**Application**: Visualizing ARR-COC's 32×32 patch grid as an interactive texture atlas

---

## Overview

Texture atlases (sprite sheets) are fundamental to game development, combining multiple textures into a single image for efficient rendering. Game developers have built sophisticated visualization tools to inspect, debug, and navigate these atlases. This document explores these techniques and applies them to ARR-COC's vision-language model architecture.

**Key insight**: ARR-COC's 32×32 patch grid IS a texture atlas — each patch is a "sprite" containing 13-channel relevance information that can be visualized using game development patterns.

---

## Section 1: Texture Atlas Fundamentals

### What is a Texture Atlas?

From [Game Developer: Using Texture Atlases](https://www.gamedeveloper.com/programming/using-texture-atlases) (accessed 2025-10-31):

> "A texture atlas is an image that contains multiple textures. This allows multiple textures to be drawn with a single draw call, reducing overhead and increasing performance."

**Core concept**: Pack multiple images into one large image, use UV coordinates to select regions.

### Texture Atlas Structure

```
┌───────────────────────────────
│ Texture Atlas (e.g., 2048×2048)
│
│  ┌──┐  ┌────┐  ┌──┐
│  │01│  │ 02 │  │03│  ← Individual sprites/textures
│  └──┘  └────┘  └──┘
│
│  ┌────┐ ┌──┐  ┌────┐
│  │ 04 │ │05│  │ 06 │
│  └────┘ └──┘  └────┘
│
│  UV coordinates map to each region
│  Metadata file stores positions/sizes
```

**Applies to ARR-COC**: 32×32 patches = 1,024 "sprites" in one atlas, each containing 13 relevance channels.

---

## Section 2: TexturePacker — Industry Standard Visualization

From [TexturePacker by CodeAndWeb](https://www.codeandweb.com/texturepacker) (accessed 2025-10-31):

### Key Visualization Features

**1. Grid Overlay Display**
- Shows atlas with visible grid lines
- Highlights individual sprite boundaries
- Color-coded for different sprite types

**2. Interactive Inspector**
- Click on sprite → show properties (name, size, UV coords)
- Hover → highlight sprite region
- Metadata panel shows texture info

**3. Pack Preview Window**
- Real-time preview while packing sprites
- Shows how sprites are arranged
- Displays efficiency metrics (atlas usage %)

**4. Animation Preview**
- Play sprite animations directly in viewer
- Frame-by-frame scrubbing
- Pivot point visualization

From [TexturePacker Documentation](https://www.codeandweb.com/texturepacker/documentation):

> "TexturePacker creates sprite sheets from your images. Fast. Optimized. Easy to use. The Pack Preview window shows the atlas as it's being built, with interactive controls to inspect individual sprites."

### ARR-COC Application

Map TexturePacker's features → ARR-COC visualization:

| TexturePacker Feature | ARR-COC Equivalent |
|----------------------|-------------------|
| Sprite = game asset | Patch = image region (e.g., 14×14 pixels) |
| Atlas = packed sprites | 32×32 grid = 1,024 patches |
| UV coordinates | Patch position (x, y) in grid |
| Sprite metadata | 13-channel relevance scores + token count |
| Pack preview | Dynamic texture array preview |

**Visualization idea**: Click patch → show 13 channels as separate images or false-color composite.

---

## Section 3: Leshy SpriteSheet Tool — Web-Based Interactive Viewer

From [Leshy SpriteSheet Tool](https://www.leshylabs.com/apps/sstool/) (accessed 2025-10-31):

### HTML5 Interactive Features

**1. Canvas-Based Display**
- HTML5 Canvas for rendering atlas
- Drag/pan to navigate large atlases
- Zoom in/out with mouse wheel

**2. Sprite Selection UI**
- Click sprite → highlight with border
- List view shows all sprites
- Search/filter sprites by name

**3. Sprite Preview Panel**
- Selected sprite shown in preview window
- Shows sprite dimensions, position
- Export individual sprite as PNG

**4. Edit Mode**
- Move sprites within atlas
- Resize sprite regions
- Real-time updates

### Technical Implementation Pattern

```javascript
// Leshy-style texture atlas viewer (conceptual)
class TextureAtlasViewer {
  constructor(canvas, atlasImage, metadata) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.atlasImage = atlasImage;
    this.sprites = metadata.sprites; // Array of {name, x, y, width, height}
    this.selectedSprite = null;

    this.setupInteraction();
  }

  draw() {
    // Draw atlas image
    this.ctx.drawImage(this.atlasImage, 0, 0);

    // Draw grid overlay
    this.sprites.forEach(sprite => {
      this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
      this.ctx.strokeRect(sprite.x, sprite.y, sprite.width, sprite.height);
    });

    // Highlight selected sprite
    if (this.selectedSprite) {
      this.ctx.strokeStyle = 'rgba(255, 255, 0, 1)';
      this.ctx.lineWidth = 3;
      this.ctx.strokeRect(
        this.selectedSprite.x,
        this.selectedSprite.y,
        this.selectedSprite.width,
        this.selectedSprite.height
      );
    }
  }

  setupInteraction() {
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Find sprite at click position
      this.selectedSprite = this.sprites.find(sprite =>
        x >= sprite.x && x <= sprite.x + sprite.width &&
        y >= sprite.y && y <= sprite.y + sprite.height
      );

      this.draw();
      this.updateInfoPanel();
    });
  }

  updateInfoPanel() {
    if (this.selectedSprite) {
      document.getElementById('sprite-name').textContent = this.selectedSprite.name;
      document.getElementById('sprite-pos').textContent =
        `Position: (${this.selectedSprite.x}, ${this.selectedSprite.y})`;
      document.getElementById('sprite-size').textContent =
        `Size: ${this.selectedSprite.width}×${this.selectedSprite.height}`;
    }
  }
}
```

**ARR-COC Implementation**: Use Canvas to display 32×32 grid, click patch → show 13 channels in side panel.

---

## Section 4: Free Texture Packer — Batch Processing & Web Tools

From [Free Texture Packer](https://free-tex-packer.com/app/) (accessed 2025-10-31):

### Visualization Modes

**1. Atlas View**
- Shows complete packed atlas
- Color-coded regions by sprite type
- Grid lines optional

**2. List View**
- Table of all sprites
- Sortable by name, size, position
- Click row → highlight in atlas

**3. Export Formats**
- JSON (sprite positions, sizes)
- XML (alternative metadata format)
- CSS (for web sprites)

### Metadata JSON Structure

```json
{
  "frames": {
    "sprite_01.png": {
      "frame": {"x": 0, "y": 0, "w": 64, "h": 64},
      "rotated": false,
      "trimmed": false,
      "spriteSourceSize": {"x": 0, "y": 0, "w": 64, "h": 64},
      "sourceSize": {"w": 64, "h": 64}
    },
    "sprite_02.png": {
      "frame": {"x": 64, "y": 0, "w": 128, "h": 128},
      "rotated": false,
      "trimmed": true,
      "spriteSourceSize": {"x": 2, "y": 2, "w": 124, "h": 124},
      "sourceSize": {"w": 128, "h": 128}
    }
  },
  "meta": {
    "app": "Free Texture Packer",
    "version": "1.0",
    "image": "spritesheet.png",
    "format": "RGBA8888",
    "size": {"w": 2048, "h": 2048},
    "scale": "1"
  }
}
```

**ARR-COC Metadata Equivalent**:

```json
{
  "patches": {
    "patch_0_0": {
      "position": {"row": 0, "col": 0, "x": 0, "y": 0},
      "size": {"width": 14, "height": 14},
      "channels": 13,
      "token_count": 196,
      "relevance_scores": {
        "semantic": 0.85,
        "edge": 0.62,
        "spatial": 0.91
      }
    },
    "patch_0_1": {
      "position": {"row": 0, "col": 1, "x": 14, "y": 0},
      "size": {"width": 14, "height": 14},
      "channels": 13,
      "token_count": 64,
      "relevance_scores": {
        "semantic": 0.23,
        "edge": 0.15,
        "spatial": 0.48
      }
    }
  },
  "meta": {
    "model": "ARR-COC-VIS",
    "grid_size": "32×32",
    "total_patches": 1024,
    "image_size": {"width": 448, "height": 448},
    "patch_size": {"width": 14, "height": 14}
  }
}
```

---

## Section 5: Unity Sprite Atlas Inspector — Game Engine Patterns

From [Unity Sprite Atlas Documentation](https://docs.unity3d.com/Manual/class-SpriteAtlas.html) (accessed 2025-10-31):

### Unity Inspector Features

**1. Pack Preview Panel**
- Bottom of Inspector window
- Shows packed atlas texture
- Grid overlay toggle

**2. Sprite Preview**
- Click sprite → zoom to that region
- Shows sprite borders
- UV coordinate display

**3. Atlas Properties Panel**
- Compression settings
- Max atlas size
- Padding between sprites

From [Unity Discussions](https://discussions.unity.com/t/is-there-a-way-to-preview-sprite-atlases-in-editor-when-using-sprite-atlas-v2-enabled-for-builds/937614) (accessed 2025-10-31):

> "The Pack Preview button generates the atlas and shows it at the bottom of the Inspector. You can tap on it to reveal the full preview with interactive controls."

### Interaction Pattern

```
Unity Inspector Layout:

┌─────────────────────────────────
│ Sprite Atlas Inspector
├─────────────────────────────────
│ Objects for Packing:
│   [List of sprites]
│
│ Atlas Settings:
│   - Max Size: 2048×2048
│   - Padding: 4px
│   - Compression: Automatic
│
│ [Pack Preview Button]
├─────────────────────────────────
│ Pack Preview:
│ ┌───────────────────────────
│ │ [Atlas visualization here]
│ │  - Grid overlay
│ │  - Click sprite → highlight
│ │  - Zoom controls
│ └───────────────────────────
```

**ARR-COC Equivalent**: Gradio interface with similar layout — patch list, settings, preview panel.

---

## Section 6: Interactive Grid Overlay Techniques

### Grid Visualization Patterns

**Pattern 1: Canvas Grid Lines**

```javascript
function drawGrid(ctx, rows, cols, cellWidth, cellHeight) {
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;

  // Vertical lines
  for (let col = 0; col <= cols; col++) {
    const x = col * cellWidth;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rows * cellHeight);
    ctx.stroke();
  }

  // Horizontal lines
  for (let row = 0; row <= rows; row++) {
    const y = row * cellHeight;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cols * cellWidth, y);
    ctx.stroke();
  }
}
```

**Pattern 2: Hover Highlight**

```javascript
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  // Calculate which cell is hovered
  const col = Math.floor(x / cellWidth);
  const row = Math.floor(y / cellHeight);

  // Highlight hovered cell
  ctx.fillStyle = 'rgba(255, 255, 0, 0.2)';
  ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);

  // Show tooltip with patch info
  showTooltip(row, col, {
    position: `Patch (${row}, ${col})`,
    tokens: patchData[row][col].tokens,
    relevance: patchData[row][col].relevance
  });
});
```

**Pattern 3: Click Selection**

```javascript
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const col = Math.floor(x / cellWidth);
  const row = Math.floor(y / cellHeight);

  // Update selected patch
  selectedPatch = {row, col};

  // Draw selection border
  ctx.strokeStyle = 'rgba(0, 255, 255, 1)';
  ctx.lineWidth = 3;
  ctx.strokeRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);

  // Update detail panel
  updateDetailPanel(patchData[row][col]);
});
```

---

## Section 7: ARR-COC Texture Atlas Visualization Design

### Conceptual Layout

```
┌──────────────────────────────────────────────────────────────
│ ARR-COC Texture Atlas Inspector
├──────────────────────────────────────────────────────────────
│
│ ┌─────────────────────┐  ┌────────────────────────────────
│ │  32×32 Grid View    │  │ Patch Detail Panel
│ │                     │  │
│ │  [Interactive       │  │ Selected: Patch (5, 12)
│ │   Canvas showing    │  │ Position: Row 5, Col 12
│ │   448×448 image     │  │ Size: 14×14 pixels
│ │   with 32×32 grid]  │  │ Tokens: 196 (max)
│ │                     │  │
│ │  - Hover: highlight │  │ Relevance Scores:
│ │  - Click: select    │  │  Semantic:  ████████░░ 0.85
│ │  - Grid overlay     │  │  Edge:      ██████░░░░ 0.62
│ │                     │  │  Spatial:   █████████░ 0.91
│ └─────────────────────┘  │
│                          │ 13-Channel Breakdown:
│ ┌─────────────────────┐  │  [Tabs: RGB | False Color | Individual]
│ │ Channel Compositor  │  │
│ │                     │  │  ┌───┬───┬───┬───
│ │ R: [Channel 3 ▼]    │  │  │ 0 │ 1 │ 2 │ 3  ← Select channels
│ │ G: [Channel 7 ▼]    │  │  └───┴───┴───┴───
│ │ B: [Channel 11 ▼]   │  │
│ │                     │  │  [Preview of selected patch]
│ │ [Apply]             │  │  [Show all 13 channels]
│ └─────────────────────┘  └────────────────────────────────
│
│ ┌──────────────────────────────────────────────────────────
│ │ Patch List (Sortable Table)
│ │
│ │ | Patch | Position | Tokens | Semantic | Edge | Spatial |
│ │ |-------|----------|--------|----------|------|---------|
│ │ | (0,0) | Top-left |  196   |  0.85    | 0.62 |  0.91   |
│ │ | (0,1) | Top      |   64   |  0.23    | 0.15 |  0.48   |
│ │ | ...   | ...      |  ...   |  ...     | ...  |  ...    |
│ │
│ │ Click row → highlight in grid view
│ └──────────────────────────────────────────────────────────
```

### Key Features

**1. Grid View (Canvas)**
- Display 448×448 input image
- Overlay 32×32 grid lines
- Hover → highlight patch with tooltip
- Click → select patch, show details

**2. Detail Panel**
- Show selected patch info (position, tokens, relevance)
- Display 13 channels as:
  - RGB composite (select 3 channels → map to RGB)
  - False color modes (semantic/edge/spatial)
  - Individual channel view (grid of 13 images)

**3. Channel Compositor**
- Dropdown to select which channels map to R, G, B
- Live preview updates as channels change
- Presets: "Semantic", "Edges", "Spatial"

**4. Patch List Table**
- Sortable by tokens, relevance scores
- Click row → highlight in grid
- Filter/search patches

---

## Section 8: UV Coordinate Mapping for ARR-COC

### UV Coordinates in Texture Atlases

From [Lark: Texture Atlas Glossary](https://www.larksuite.com/en_us/topics/gaming-glossary/texture-atlas) (accessed 2025-10-31):

> "Texture atlas combines multiple smaller textures into a single larger texture, using UV coordinates to map regions onto 3D models."

**UV coordinate system**: (0, 0) = top-left, (1, 1) = bottom-right of atlas.

### ARR-COC UV Mapping

For a 32×32 patch grid:

```python
def patch_to_uv(row, col, grid_size=32):
    """Convert patch position to UV coordinates."""
    u_min = col / grid_size
    v_min = row / grid_size
    u_max = (col + 1) / grid_size
    v_max = (row + 1) / grid_size
    return (u_min, v_min, u_max, v_max)

# Example: Patch (5, 12)
uv = patch_to_uv(5, 12)
# Returns: (0.375, 0.15625, 0.40625, 0.1875)
```

**Use case**: When displaying patch in 3D viewer, UV coords map patch to correct region of texture.

---

## Section 9: Implementation Technologies

### Option 1: HTML5 Canvas + JavaScript

**Pros**:
- Direct pixel manipulation
- Fast rendering
- Well-documented game dev patterns

**Cons**:
- Manual event handling
- No built-in 3D support

**Best for**: 2D grid view, simple interactions

### Option 2: Three.js (WebGL)

**Pros**:
- Hardware-accelerated rendering
- 3D visualization (e.g., patches as tiles in 3D space)
- Extensive texture handling

**Cons**:
- Steeper learning curve
- Overkill for 2D grid

**Best for**: Advanced 3D visualizations, material preview

### Option 3: Gradio + Matplotlib

**Pros**:
- Easy Python integration
- Quick prototyping
- No JavaScript required

**Cons**:
- Limited interactivity
- Slower than Canvas/WebGL

**Best for**: MVP, proof-of-concept

### Option 4: Gradio + Custom HTML Component

**Pros**:
- Combines Python backend with JavaScript frontend
- Full control over UI
- Can embed Canvas or Three.js

**Cons**:
- Requires bidirectional data transfer
- More complex setup

**Best for**: Production-ready ARR-COC visualizer

---

## Section 10: ARR-COC Specific Visualization Challenges

### Challenge 1: Multi-Channel Display

**Problem**: 13 channels can't be displayed as RGB (only 3 channels).

**Solutions**:
1. **Channel Compositor**: User selects 3 channels → map to R, G, B
2. **False Color Modes**: Pre-defined composites (e.g., Semantic = channels 0-2 → RGB)
3. **Grid of 13 Images**: Show all channels simultaneously as small thumbnails

### Challenge 2: Variable Token Count (64-400)

**Problem**: Patches have different token budgets based on relevance.

**Visualization**:
- Color-code patches by token count
- Use opacity/saturation to show token budget
- Heatmap overlay: high tokens = warm colors, low tokens = cool colors

```javascript
// Color patch by token count
function getTokenColor(tokenCount, maxTokens = 400) {
  const ratio = tokenCount / maxTokens;
  // Interpolate from blue (low) to red (high)
  const r = Math.floor(255 * ratio);
  const b = Math.floor(255 * (1 - ratio));
  return `rgba(${r}, 0, ${b}, 0.7)`;
}
```

### Challenge 3: 32×32 = 1,024 Patches

**Problem**: Too many patches to show details for all at once.

**Solutions**:
1. **Hover Tooltip**: Show patch info on hover (non-intrusive)
2. **Detail-on-Demand**: Click patch → full detail panel
3. **Filtering**: Show only high-relevance patches, hide low-relevance
4. **Aggregation**: Group patches into 8×8 regions, drill down on click

---

## Sources

**Texture Atlas Tools:**
- [TexturePacker by CodeAndWeb](https://www.codeandweb.com/texturepacker) - Industry-standard sprite sheet tool (accessed 2025-10-31)
- [Leshy SpriteSheet Tool](https://www.leshylabs.com/apps/sstool/) - HTML5 interactive atlas editor (accessed 2025-10-31)
- [Free Texture Packer](https://free-tex-packer.com/app/) - Open-source web-based packer (accessed 2025-10-31)

**Game Engine Documentation:**
- [Unity Sprite Atlas Documentation](https://docs.unity3d.com/Manual/class-SpriteAtlas.html) - Official Unity docs (accessed 2025-10-31)
- [Unity Discussions: Sprite Atlas Preview](https://discussions.unity.com/t/is-there-a-way-to-preview-sprite-atlases-in-editor-when-using-sprite-atlas-v2-enabled-for-builds/937614) - Community discussion (accessed 2025-10-31)

**Texture Atlas Concepts:**
- [Game Developer: Using Texture Atlases](https://www.gamedeveloper.com/programming/using-texture-atlases) - Tutorial article (accessed 2025-10-31)
- [Lark Gaming Glossary: Texture Atlas](https://www.larksuite.com/en_us/topics/gaming-glossary/texture-atlas) - Definition and use cases (accessed 2025-10-31)

**Additional References:**
- [Live2D Cubism: Edit Texture Atlas](https://docs.live2d.com/en/cubism-editor-manual/texture-atlas-edit/) - Character animation atlas editing (accessed 2025-10-31)
- [WoWInterface: Texture Atlas Viewer](https://www.wowinterface.com/downloads/info25415-492408.html) - Developer tool for game textures (accessed 2025-10-31)

---

## Conclusion

Game developers have created sophisticated texture atlas visualization tools with:
- Interactive grid overlays
- Click/hover patch inspection
- UV coordinate mapping
- Metadata display (position, size, properties)
- Channel compositing

These patterns directly apply to ARR-COC's 32×32 patch grid:
- Each patch = sprite in atlas
- 13 channels = texture properties
- Token counts = metadata
- Grid view + detail panel = inspector UI

**Next steps**: Implement prototype using Gradio + Canvas for MVP, then upgrade to Three.js for 3D visualization if needed.
