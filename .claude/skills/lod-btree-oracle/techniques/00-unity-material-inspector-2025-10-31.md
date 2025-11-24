# Unity Material Inspector & Texture Debugging Techniques

**Created**: 2025-10-31
**Category**: Game Engine Visualization Techniques
**Application**: Multi-channel texture visualization for VLMs

---

## Overview

Unity's Material Inspector provides game developers with sophisticated tools for visualizing and debugging multi-channel textures in real-time. This document explores Unity's approach to texture visualization, channel-packed textures, and material preview systems, with specific application to ARR-COC's 13-channel texture arrays for vision-language models.

From [Unity Manual: Material Inspector](https://docs.unity3d.com/6000.2/Documentation/Manual/class-Material.html) (accessed 2025-10-31):
- Material Inspector allows viewing and editing Material assets
- Provides real-time preview of material properties
- Supports channel-packed textures (R, G, B, A used independently)
- Offers debug visualization modes for texture properties

---

## Section 1: Unity Material Inspector Architecture

### Core Inspector Features

The Material Inspector in Unity provides a centralized interface for material property manipulation with these key capabilities:

**Property Editing** (from Unity 6.2 Manual):
- Modify Material or Material Variant properties in real-time
- Identify relationships between Materials and Variants
- Copy/paste property settings across materials
- Associate materials with specific shaders or lightmap atlases

**Real-Time Preview** (from Unity Discussions on Material Preview):
- Material preview sphere (default 3D preview object)
- Customizable preview shapes (sphere, cube, plane, cylinder)
- Live material updates during property editing
- Interactive 3D rotation of preview object

**Inspector Controls Available**:
1. Select Shader - Navigate to shader asset in Project window
2. Edit Shader - Open shader source file
3. Reset - Reset all properties to shader defaults
4. Create Material Preset - Duplicate material settings

From [Unity Forum: Change Material Preview Shape](https://discussions.unity.com/t/change-material-preview-shape/543326) (accessed 2025-10-31):
- Material preview defaults to sphere in project bar
- Can be changed via Inspector settings for specific materials
- Flat images benefit from plane preview
- 3D materials benefit from sphere/complex mesh preview

---

## Section 2: Channel-Packed Texture Display

### Multi-Channel Texture Packing in URP

Unity's Universal Render Pipeline (URP) implements sophisticated channel packing for efficient texture use:

From [URP: Assign Channel-Packed Texture](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@14.0/manual/shaders-in-universalrp-channel-packed-texture.html) (accessed 2025-10-31):

**Standard Channel Packing Format**:
```
Channel    Property
-------    ----------
Red        Metallic
Green      Occlusion
Blue       [Not used]
Alpha      Smoothness
```

**Implementation Steps**:
1. Create RGBA texture with appropriate channel data
2. Import texture into Unity project
3. In Texture Inspector: **Disable sRGB (Color Texture)** (critical!)
4. Assign texture to both Metallic and Occlusion properties
5. Unity automatically uses correct channels from the texture

**Why Disable sRGB**: Normal maps and channel-packed textures store non-color data (height, roughness, masks). sRGB gamma correction would corrupt these values. Unity's texture importer provides this toggle specifically for technical textures.

### Custom Channel Swizzling

From [Unity Forum: Texture Channel Manager](https://discussions.unity.com/t/texture-channel-manager/577145) (accessed 2025-10-31):
- Unity supports custom channel mapping via Shader Graph
- Texture Channel Manager tool (community asset) enables visual channel packing
- Export individual maps and combine into single RGBA texture
- Automatically maintains correct channel assignments

From [Unity Docs: Sprite Texture Swizzle](https://docs.unity.cn/2022.1/Documentation/Manual/texture-type-sprite.html) (accessed 2025-10-31):
- **Swizzle property**: Orders texture source file color channel data
- Available in Texture Import Settings
- Allows R→G, G→B, B→A channel remapping
- Useful for platform-specific texture formats

### Channel Visualization Modes

**Single Channel View**:
- Unity Rendering Debugger provides per-channel visualization
- View R, G, B, A channels independently
- Grayscale display of individual channel data
- Useful for debugging channel-packed mask maps

From [URP Rendering Debugger](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@16.0/manual/features/rendering-debugger.html) (accessed 2025-10-31):
- **Rendering Debugger window**: Visualize lighting, rendering, Material properties
- Debug visualization modes for identifying rendering issues
- Material property overrides for testing
- Texture channel inspection modes

---

## Section 3: Texture Preview & Debugging Tools

### Material Ball Preview System

**3D Material Preview**:
- Default: Sphere preview in Material Inspector
- Real-time lighting updates during editing
- Rotatable preview for viewing from all angles
- Configurable preview environment lighting

From [Unity 3D Texture Preview Reference](https://docs.unity3d.com/6000.2/Documentation/Manual/class-Texture3D-reference.html) (accessed 2025-10-31):
- **Slice preview mode** for 3D textures
- Display 2D slice of each axis (X, Y, Z)
- Sliders to select specific slices for preview
- Useful for volumetric texture inspection

**Flat Texture Preview**:
- In Texture Import Settings window
- Preview at bottom of Inspector
- Zoom/pan controls for detailed inspection
- No 3D lighting applied (raw texture data)

### Mipmap Visualization

From [Unity Texture Mipmaps](https://docs.unity3d.com/6000.2/Documentation/Manual/texture-mipmaps-introduction.html) (accessed 2025-10-31):
- **Enable/disable mipmaps** in Texture Import Settings Inspector
- GPU samples appropriate mipmap level based on screen space
- Mipmap chain generated on texture import
- Debug views for mipmap level visualization

**Mipmap Streaming System** (Unity 2022.2+):
- Set up Mipmap Streaming on individual textures
- Select texture asset, navigate to Inspector
- Enable Mipmap Streaming toggle
- Unity dynamically loads/unloads mipmap levels based on visibility

From [Unity Forum: Texture Mipmap Bias](https://discussions.unity.com/t/texture-mipmapbias-setting-should-be-exposed/809349) (accessed 2025-10-31):
- **mipMapBias setting**: Offsets mipmap level selection
- Bias of -0.5 delays switch to lower mip level
- Improves texture sharpness at distance
- Currently requires script access (not exposed in Inspector UI)

### Normal Map Visualization

**False Color Display for Normal Maps**:

From [Unity Forum: Normal Map Looks Red](https://discussions.unity.com/t/normal-map-looks-red-in-shader-graph/831975) (accessed 2025-10-31):
- **Red color in Unity's tangent space normal maps**: Normal
- Inspector preview shows reconstructed normal (not raw texture data)
- XYZ normal data encoded as RGB color
- Unity automatically decodes during shader sampling

**Normal Map Formats**:
- Unity expects normal maps in tangent space
- Red channel: X component of normal
- Green channel: Y component of normal
- Blue channel: Z component of normal
- Inspector must mark texture as "Normal Map" for proper decoding

From [Unity: Normal Map (Bump Mapping)](https://docs.unity3d.com/2023.2/Documentation/Manual/StandardShaderMaterialParameterNormalMap.html) (accessed 2025-10-31):
- **Normal maps**: Type of bump map
- Add surface detail (bumps, grooves, scratches)
- Modify lighting calculations without extra geometry
- Inspector provides "Normal Map" checkbox in Texture Import Settings

---

## Section 4: Application to ARR-COC 13-Channel Textures

### Mapping Unity's System to ARR-COC

**ARR-COC's 13-Channel Texture Structure**:
1. Propositional channels (0-3): Statistical information content
2. Perspectival channels (4-8): Salience landscapes
3. Participatory channels (9-11): Query-content coupling
4. Procedural channel (12): Learned compression guidance

**Unity-Inspired Visualization Approaches**:

**1. Channel Packing Visualization**:
```
Strategy: Pack ARR-COC's 13 channels into 4 RGBA textures
- Texture 0 (RGBA): Channels 0-3 (Propositional)
- Texture 1 (RGBA): Channels 4-7 (Perspectival 1-4)
- Texture 2 (RGBA): Channels 8-11 (Perspectival 5 + Participatory 1-3)
- Texture 3 (RG__): Channels 12 + unused

Unity-style display:
- Material Inspector shows each texture separately
- Channel swizzle UI for custom RGB composites
- Single-channel grayscale view per texture
```

**2. Material Ball Preview for Texture Patches**:
```
Approach: Apply ARR-COC texture array to 3D preview sphere
- Each 32×32 patch becomes a material
- Sphere preview shows how texture channels affect appearance
- Rotate sphere to inspect texture from multiple angles
- Compare high-relevance (400 tokens) vs low-relevance (64 tokens) patches

UI Pattern:
- Click patch in 2D grid → Load patch texture → Display on sphere
- Channel selector buttons (0-12) → Update sphere material
- False color mode dropdown → Apply Unity-style visualization shader
```

**3. Mipmap-Style LOD Visualization**:
```
ARR-COC LOD Levels (inspired by Unity mipmaps):
- Level 0: 400 tokens (full detail patch)
- Level 1: 256 tokens (reduced detail)
- Level 2: 128 tokens (moderate compression)
- Level 3: 64 tokens (maximum compression)

Visualization:
- Display patch at multiple LOD levels side-by-side
- Slider to select LOD level (like Unity mipmap slider)
- Show token budget and relevance score per level
- Highlight which level Unity would choose at given distance
```

**4. Rendering Debugger-Style Channel Inspection**:
```
Unity Rendering Debugger for ARR-COC:
- Material Property Override: Force specific channel to 1.0
- Channel Isolation: Display only Channel 5 (Perspectival salience)
- False Color Modes:
  - Semantic (RGB = Channels 0,1,2)
  - Edges (RGB = Channels 4,5,6)
  - Spatial (RGB = Channels 9,10,11)
- Numeric Overlay: Show exact tensor values on hover
```

### Recommended Unity-Style Features for ARR-COC

**1. Texture Channel Manager Integration**:
- Import 13-channel numpy array
- Visual channel assignment to RGBA textures
- Export channel-packed textures for web display
- Preview channel combinations before export

**2. Material Preview Configuration**:
- **Sphere Preview**: Default for 3D understanding of texture application
- **Plane Preview**: Flat 2D view of texture patches
- **Custom Mesh Preview**: Apply to human face mesh for relevance visualization
- Real-time preview updates as channel weights change

**3. Inspector Debug Tools**:
```
Custom Inspector Panel for ARR-COC:
[ ] Show Propositional Channels (0-3)
[ ] Show Perspectival Channels (4-8)
[ ] Show Participatory Channels (9-11)
[ ] Show Procedural Channel (12)

False Color Mode: [Dropdown]
- Semantic Composite
- Edge Detection View
- Spatial Relationship View
- Relevance Heatmap

Channel Swizzle:
R: [Channel 0 ▼]  G: [Channel 5 ▼]  B: [Channel 9 ▼]
→ Live Preview Updates
```

**4. Mipmap Bias for Token Budget Tuning**:
- Adjust token budget "bias" to shift LOD selection
- Bias +0.5: Prefer lower token counts (compress more)
- Bias -0.5: Prefer higher token counts (preserve detail)
- Visual comparison: biased vs unbiased rendering

---

## Section 5: Practical Implementation Patterns

### Unity Workflow for Multi-Channel Debugging

**Standard Unity Developer Workflow**:
1. Import texture assets into project
2. Configure Texture Import Settings (compression, mipmaps, format)
3. Assign textures to Material properties in Inspector
4. Preview material on sphere/mesh in Inspector
5. Use Rendering Debugger to visualize specific channels
6. Iterate on texture content based on visual feedback

**Adapted Workflow for ARR-COC Development**:
1. Generate 13-channel texture array from VLM forward pass
2. Pack channels into 4 RGBA textures (use Unity channel packer pattern)
3. Create Material with custom shader supporting 4 texture inputs
4. Preview material on 3D sphere in custom Inspector panel
5. Use channel isolation to debug individual channels
6. Adjust relevance realization parameters based on visual inspection
7. Export final composite visualizations for analysis

### Code Example: Unity-Style Channel Swizzle in Python

```python
import numpy as np
from PIL import Image

def unity_style_channel_pack(channels_13: np.ndarray) -> list[np.ndarray]:
    """
    Pack 13-channel ARR-COC texture into 4 RGBA textures (Unity pattern)

    Args:
        channels_13: (H, W, 13) array of texture channels

    Returns:
        List of 4 RGBA textures as (H, W, 4) arrays
    """
    h, w, c = channels_13.shape
    assert c == 13, "Expected 13 channels"

    # Normalize to 0-255 for display
    normalized = ((channels_13 - channels_13.min()) /
                  (channels_13.max() - channels_13.min()) * 255).astype(np.uint8)

    # Pack into 4 RGBA textures (Unity URP pattern)
    texture_0 = np.stack([normalized[:,:,0],   # R: Channel 0 (Propositional)
                          normalized[:,:,1],   # G: Channel 1 (Propositional)
                          normalized[:,:,2],   # B: Channel 2 (Propositional)
                          normalized[:,:,3]],  # A: Channel 3 (Propositional)
                         axis=-1)

    texture_1 = np.stack([normalized[:,:,4],   # R: Channel 4 (Perspectival)
                          normalized[:,:,5],   # G: Channel 5 (Perspectival)
                          normalized[:,:,6],   # B: Channel 6 (Perspectival)
                          normalized[:,:,7]],  # A: Channel 7 (Perspectival)
                         axis=-1)

    texture_2 = np.stack([normalized[:,:,8],   # R: Channel 8 (Perspectival)
                          normalized[:,:,9],   # G: Channel 9 (Participatory)
                          normalized[:,:,10],  # B: Channel 10 (Participatory)
                          normalized[:,:,11]], # A: Channel 11 (Participatory)
                         axis=-1)

    texture_3 = np.stack([normalized[:,:,12],  # R: Channel 12 (Procedural)
                          np.zeros((h,w), dtype=np.uint8),  # G: Unused
                          np.zeros((h,w), dtype=np.uint8),  # B: Unused
                          np.ones((h,w), dtype=np.uint8)*255],  # A: Full alpha
                         axis=-1)

    return [texture_0, texture_1, texture_2, texture_3]


def unity_false_color_composite(channels_13: np.ndarray,
                                 mode: str = "semantic") -> np.ndarray:
    """
    Generate Unity-style false color composite for visualization

    Args:
        channels_13: (H, W, 13) array of texture channels
        mode: "semantic", "edges", "spatial", "relevance"

    Returns:
        (H, W, 3) RGB false color image
    """
    if mode == "semantic":
        # Map propositional channels to RGB
        rgb = np.stack([channels_13[:,:,0],   # R: Channel 0
                        channels_13[:,:,1],   # G: Channel 1
                        channels_13[:,:,2]],  # B: Channel 2
                       axis=-1)

    elif mode == "edges":
        # Map perspectival edge channels to RGB
        rgb = np.stack([channels_13[:,:,4],   # R: Channel 4
                        channels_13[:,:,5],   # G: Channel 5
                        channels_13[:,:,6]],  # B: Channel 6
                       axis=-1)

    elif mode == "spatial":
        # Map participatory spatial channels to RGB
        rgb = np.stack([channels_13[:,:,9],   # R: Channel 9
                        channels_13[:,:,10],  # G: Channel 10
                        channels_13[:,:,11]], # B: Channel 11
                       axis=-1)

    elif mode == "relevance":
        # Composite relevance heatmap (weight channels by importance)
        relevance = (channels_13[:,:,0] * 0.3 +    # Propositional weight
                     channels_13[:,:,4] * 0.4 +    # Perspectival weight
                     channels_13[:,:,9] * 0.3)     # Participatory weight

        # Apply false color: blue (low) → green → red (high)
        rgb = np.zeros((*relevance.shape, 3))
        rgb[:,:,2] = np.clip(1.0 - relevance * 2, 0, 1)  # Blue (low relevance)
        rgb[:,:,1] = np.clip(1.0 - abs(relevance - 0.5) * 2, 0, 1)  # Green (medium)
        rgb[:,:,0] = np.clip(relevance * 2 - 1, 0, 1)   # Red (high relevance)

    # Normalize and convert to uint8
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
    return rgb
```

### Material Inspector UI Mockup for ARR-COC

```
╔══════════════════════════════════════════════════════════
║ ARR-COC Texture Inspector
╠══════════════════════════════════════════════════════════
║
║ 📦 Texture Pack Properties
║   ├─ Texture 0 (RGBA): Propositional (Channels 0-3)
║   ├─ Texture 1 (RGBA): Perspectival 1-4 (Channels 4-7)
║   ├─ Texture 2 (RGBA): Perspectival 5 + Participatory (Channels 8-11)
║   └─ Texture 3 (R___): Procedural (Channel 12)
║
║ 🎨 Visualization Mode
║   ( ) Single Channel View → Channel: [5 ▼] (Perspectival Salience)
║   (•) False Color Composite → Mode: [Semantic ▼]
║   ( ) Channel Swizzle → R:[0▼] G:[5▼] B:[9▼]
║
║ 🖼️ Preview
║   ┌─────────────────────────────
║   │                          Sphere Preview
║   │         ░▒▓███▓▒░
║   │       ░▓████████▓░        Interactive 3D
║   │      ▒████████████▒       (drag to rotate)
║   │      ▓██████████████
║   │      ▓██████████████
║   │      ▒████████████▒
║   │       ░▓████████▓░
║   │         ░▒▓███▓▒░
║   └─────────────────────────────
║
║ 📊 Channel Statistics
║   Channel 0: min=0.02  max=0.98  mean=0.45  std=0.23
║   Channel 5: min=0.00  max=1.00  mean=0.52  std=0.31
║   Channel 9: min=0.15  max=0.87  mean=0.48  std=0.19
║
║ 🔧 Debug Options
║   [x] Show Relevance Overlay
║   [ ] Show Token Budget Labels
║   [x] Enable Mipmap LOD Visualization
║   Mipmap Bias: [-0.5] ←------•------→ [+0.5]
║
║ [Apply] [Reset] [Export Visualization]
╚══════════════════════════════════════════════════════════
```

---

## Key Takeaways for ARR-COC Implementation

### Unity's Lessons for Multi-Channel Texture Visualization

1. **Channel Packing is Standard**: Game engines routinely pack 4-8 textures into single RGBA files. ARR-COC's 13 channels naturally map to 3-4 RGBA textures.

2. **Real-Time Preview is Essential**: Unity's material ball preview provides immediate visual feedback. ARR-COC should offer similar 3D preview of texture patches.

3. **False Color Modes Enable Understanding**: Single-channel grayscale + false color composites help developers debug. ARR-COC needs semantic/edges/spatial false color modes.

4. **Inspector UI Consolidates Tools**: Unity centralizes texture/material debugging in one Inspector panel. ARR-COC visualization should provide unified interface.

5. **Mipmap System Maps to LOD**: Unity's mipmap LOD selection parallels ARR-COC's token budget allocation. Visualizing LOD levels helps understand compression tradeoffs.

### Immediate Next Steps

**MVP Implementation** (1-2 days):
1. Implement `unity_style_channel_pack()` to export 4 RGBA textures
2. Add false color composite modes (semantic, edges, spatial, relevance)
3. Create single-channel grayscale view selector
4. Display channel statistics (min/max/mean/std) per channel

**Phase 2** (3-5 days):
1. Add 3D sphere preview using Three.js (see Part 3)
2. Implement interactive channel swizzle UI
3. Add mipmap-style LOD level visualization
4. Create Unity-style Inspector panel in Gradio

---

## Sources

**Unity Official Documentation**:
- [Material Inspector Window Reference](https://docs.unity3d.com/6000.2/Documentation/Manual/class-Material.html) - Unity 6.2 Manual (accessed 2025-10-31)
- [Assign Channel-Packed Texture to URP Material](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@14.0/manual/shaders-in-universalrp-channel-packed-texture.html) - URP 14.0 Manual (accessed 2025-10-31)
- [Preview Mode Control - Shader Graph](https://docs.unity3d.com/Packages/com.unity.shadergraph@12.0/manual/Preview-Mode-Control.html) - Shader Graph 12.0 Manual (accessed 2025-10-31)
- [Normal Map (Bump Mapping)](https://docs.unity3d.com/2023.2/Documentation/Manual/StandardShaderMaterialParameterNormalMap.html) - Unity 2023.2 Manual (accessed 2025-10-31)
- [Rendering Debugger - URP](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@16.0/manual/features/rendering-debugger.html) - URP 16.0 Manual (accessed 2025-10-31)
- [Texture Mipmaps Introduction](https://docs.unity3d.com/6000.2/Documentation/Manual/texture-mipmaps-introduction.html) - Unity 6.2 Manual (accessed 2025-10-31)
- [3D Texture Preview Reference](https://docs.unity3d.com/6000.2/Documentation/Manual/class-Texture3D-reference.html) - Unity 6.2 Manual (accessed 2025-10-31)
- [Mipmap Streaming System](https://docs.unity3d.com/2022.2/Documentation/Manual/TextureStreaming.html) - Unity 2022.2 Manual (accessed 2025-10-31)
- [Sprite Texture Swizzle](https://docs.unity.cn/2022.1/Documentation/Manual/texture-type-sprite.html) - Unity China Docs 2022.1 (accessed 2025-10-31)

**Unity Community Discussions**:
- [Texture Channel Manager](https://discussions.unity.com/t/texture-channel-manager/577145) - Unity Discussions (accessed 2025-10-31)
- [Change Material Preview Shape](https://discussions.unity.com/t/change-material-preview-shape/543326) - Unity Discussions (accessed 2025-10-31)
- [Normal Map Looks Red in Shader Graph](https://discussions.unity.com/t/normal-map-looks-red-in-shader-graph/831975) - Unity Discussions (accessed 2025-10-31)
- [Texture Mipmap Bias Should Be Exposed](https://discussions.unity.com/t/texture-mipmapbias-setting-should-be-exposed/809349) - Unity Discussions (accessed 2025-10-31)
- [Efficient Way to Swizzle Texture Channels](https://discussions.unity.com/t/efficient-way-to-swizzle-texture-channels/782472) - Unity Discussions (accessed 2025-10-31)

**Web Research**:
- Web search results: "Unity material inspector texture channels 2024 2025"
- Web search results: "Unity shader graph texture preview debugging"
- Web search results: "Unity editor texture visualization multi-channel"
- Web search results: "Unity texture inspector debugging false color normal map visualization 2024"
- Web search results: "Unity material preview sphere 3D inspector texture display"
- Web search results: "Unity texture mipmap preview channel swizzling inspector tools"

**Additional References**:
- [Reddit: ChannelPacker Tool](https://www.reddit.com/r/Unity3D/comments/14v04tp/i_made_a_tool_called_channelpacker_allowing_you/) - Unity3D subreddit (accessed 2025-10-31)
- [YouTube: Unity Shader Graph Basics - Textures and UVs](https://www.youtube.com/watch?v=jVNC0Z2p9qw) - Daniel Ilett (accessed 2025-10-31)
- [YouTube: Basics of Mipmaps in Unity](https://www.youtube.com/watch?v=Vq5T4hu8NWM) - Ned Makes Games (accessed 2025-10-31)

---

**Related Knowledge Files**:
- `00-unreal-material-editor-2025-10-31.md` (PART 2) - Comparison with Unreal Engine approach
- `00-threejs-texture-display-2025-10-31.md` (PART 3) - Web-based 3D visualization
- `00-gradio-3d-integration-2025-10-31.md` (PART 4) - Integrating with Python/Gradio
- `00-arr-coc-texture-viewer-implementation-2025-10-31.md` (PART 10) - Complete implementation roadmap
