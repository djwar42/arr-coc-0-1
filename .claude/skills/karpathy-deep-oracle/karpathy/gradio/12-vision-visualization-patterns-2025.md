# Gradio Vision Visualization Patterns (2025)

**Last Updated**: 2025-10-31
**Relevance**: Critical for ARR-COC-0.1 homunculus visualization, heatmaps, and patch selection displays

## Overview

This document covers visualization patterns for vision models in Gradio, with emphasis on techniques directly applicable to ARR-COC's adaptive visual processing. Key patterns include heatmap overlays, patch selection visualization (the "homunculus" pattern), multi-view layouts, and interactive exploration tools.

**Why This Matters for ARR-COC**:
- Visualizing which image patches receive high/low token budgets (64-400 tokens)
- Displaying relevance scores as heatmaps
- Showing the "visual homunculus" - where the model focuses vs. compresses
- Interactive debugging of relevance realization decisions

## Section 1: Heatmap Overlays (~130 lines)

### 1.1 Core Heatmap Generation with Matplotlib

Heatmaps visualize scalar values (like relevance scores) across spatial regions.

**Basic Pattern - Matplotlib to PIL**:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
import io

def generate_heatmap(scores, cmap_name='viridis'):
    """
    Convert 2D numpy array of scores to colored heatmap image.

    Args:
        scores: np.ndarray of shape (H, W) with relevance scores
        cmap_name: matplotlib colormap name

    Returns:
        PIL.Image: Colored heatmap
    """
    # Normalize scores to [0, 1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Apply colormap
    cmap = colormaps.get_cmap(cmap_name)
    colored = cmap(scores_norm)[..., :3]  # RGB only, drop alpha

    # Convert to uint8
    heatmap_array = (colored * 255).astype(np.uint8)

    return Image.fromarray(heatmap_array)
```

From [DINOv3-PCA-visualization](https://huggingface.co/spaces/sayedM/DINOv3-PCA-visualization/blob/main/app.py) (accessed 2025-10-31).

**Smooth Interpolation for Patch-Based Heatmaps**:

For models that produce patch-level scores (like ARR-COC's 14×14 patches), interpolate to match image resolution:

```python
from PIL import Image

def smooth_heatmap_overlay(patch_scores, target_size=(224, 224)):
    """
    Convert patch-level scores to smooth full-resolution heatmap.

    Args:
        patch_scores: np.ndarray of shape (H_patches, W_patches)
        target_size: tuple (height, width) of original image

    Returns:
        PIL.Image: Smooth heatmap at target resolution
    """
    # Generate low-res heatmap
    heatmap_lowres = generate_heatmap(patch_scores, cmap_name='viridis')

    # Upsample with bicubic interpolation for smoothness
    heatmap_smooth = heatmap_lowres.resize(target_size, Image.Resampling.BICUBIC)

    return heatmap_smooth
```

### 1.2 Blending Heatmaps with Original Images

**Alpha Blending Pattern**:

```python
def blend_heatmap(base_image, heatmap, alpha=0.5):
    """
    Blend heatmap onto base image with transparency.

    Args:
        base_image: PIL.Image (original)
        heatmap: PIL.Image (colored relevance map)
        alpha: float in [0, 1], opacity of heatmap

    Returns:
        PIL.Image: Blended visualization
    """
    # Ensure both are RGBA
    base_rgba = base_image.convert("RGBA")
    heat_rgba = heatmap.convert("RGBA")

    # Blend
    blended = Image.blend(base_rgba, heat_rgba, alpha=alpha)

    return blended
```

**Complete Heatmap Overlay Pipeline** (ARR-COC Pattern):

```python
def visualize_relevance_overlay(
    image_pil,
    relevance_scores,  # Shape: (14, 14) for patch-level scores
    alpha=0.5,
    cmap='viridis'
):
    """
    Complete pipeline: patch scores → smooth heatmap → overlay.

    Perfect for ARR-COC homunculus visualization.
    """
    # 1. Smooth interpolation to image size
    heatmap = smooth_heatmap_overlay(relevance_scores, target_size=image_pil.size)

    # 2. Blend with original
    overlay = blend_heatmap(image_pil, heatmap, alpha=alpha)

    return overlay
```

### 1.3 Gradio Integration for Heatmap Display

**Interactive Heatmap Component**:

```python
import gradio as gr

def create_heatmap_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                alpha_slider = gr.Slider(0, 1, 0.5, label="Heatmap Opacity")
                cmap_dropdown = gr.Dropdown(
                    ['viridis', 'magma', 'inferno', 'plasma', 'jet'],
                    value='viridis',
                    label="Colormap"
                )
                generate_btn = gr.Button("Generate Heatmap Overlay")

            with gr.Column():
                output_overlay = gr.Image(label="Heatmap Overlay")
                output_heatmap = gr.Image(label="Raw Heatmap")

        def process(image, alpha, cmap):
            # Your model inference here
            relevance_scores = compute_relevance(image)  # Returns (14, 14)

            # Generate visualizations
            overlay = visualize_relevance_overlay(image, relevance_scores, alpha, cmap)
            heatmap = smooth_heatmap_overlay(relevance_scores, image.size)

            return overlay, heatmap

        generate_btn.click(
            fn=process,
            inputs=[input_image, alpha_slider, cmap_dropdown],
            outputs=[output_overlay, output_heatmap]
        )

    return demo
```

**Color Palette Choices**:
- `viridis`: Perceptually uniform, colorblind-friendly (default)
- `magma`: High contrast for saliency
- `jet`: Traditional but not colorblind-safe
- `plasma`: Good for publication figures

## Section 2: Patch Selection Visualization (~140 lines)

### 2.1 Drawing Bounding Boxes with PIL ImageDraw

The **critical pattern for ARR-COC homunculus visualization** - showing which patches get high token budgets (selected) vs. low budgets (rejected).

**Basic Bounding Box Drawing**:

```python
from PIL import ImageDraw, ImageFont

def draw_patch_grid(image, patch_size=16, color=(0, 255, 0), thickness=2):
    """
    Draw grid overlay showing patch boundaries.

    Args:
        image: PIL.Image
        patch_size: int, size of each patch in pixels
        color: tuple (R, G, B)
        thickness: int, line width

    Returns:
        PIL.Image with grid overlay
    """
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    width, height = image.size

    # Vertical lines
    for x in range(0, width, patch_size):
        draw.line([(x, 0), (x, height)], fill=color, width=thickness)

    # Horizontal lines
    for y in range(0, height, patch_size):
        draw.line([(0, y), (width, y)], fill=color, width=thickness)

    return draw_img
```

### 2.2 Highlighting Selected Patches (High Relevance)

**Pattern: Green boxes for selected patches**:

```python
def highlight_selected_patches(
    image,
    patch_indices,  # List of (row, col) tuples for selected patches
    patch_size=16,
    color=(0, 255, 0),
    thickness=3
):
    """
    Draw thick borders around selected (high-relevance) patches.

    Args:
        patch_indices: List of (i, j) tuples indicating selected patches
    """
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    for (i, j) in patch_indices:
        # Calculate pixel coordinates
        x_min = j * patch_size
        y_min = i * patch_size
        x_max = x_min + patch_size
        y_max = y_min + patch_size

        # Draw rectangle
        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline=color,
            width=thickness
        )

    return draw_img
```

### 2.3 Red Overlay for Rejected Areas (ARR-COC Homunculus)

**THE KEY PATTERN for ARR-COC**: Show which regions get compressed heavily.

```python
def overlay_rejected_regions(
    image,
    rejection_mask,  # Boolean array (H_patches, W_patches): True = rejected
    patch_size=16,
    overlay_color=(255, 0, 0),
    alpha=0.3
):
    """
    Apply semi-transparent red overlay to rejected (low-token) regions.

    This is the "visual homunculus" - clearly showing where model ignores.

    Args:
        rejection_mask: np.ndarray of shape (14, 14) with True for rejected patches
        overlay_color: RGB tuple for overlay (default red)
        alpha: transparency of overlay
    """
    # Create overlay layer
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    H_patches, W_patches = rejection_mask.shape

    for i in range(H_patches):
        for j in range(W_patches):
            if rejection_mask[i, j]:
                # Calculate pixel coordinates
                x_min = j * patch_size
                y_min = i * patch_size
                x_max = x_min + patch_size
                y_max = y_min + patch_size

                # Fill rectangle with semi-transparent color
                rgba_color = overlay_color + (int(255 * alpha),)
                draw.rectangle(
                    [x_min, y_min, x_max, y_max],
                    fill=rgba_color
                )

    # Composite onto original
    result = image.convert('RGBA')
    result = Image.alpha_composite(result, overlay)

    return result.convert('RGB')
```

### 2.4 Complete ARR-COC Homunculus Visualization

**Full pattern combining selection + rejection**:

```python
def visualize_arr_coc_homunculus(
    image_pil,
    token_allocation,  # Array (14, 14) with token counts (64-400)
    threshold_high=300,  # Patches with >= 300 tokens highlighted green
    threshold_low=100,   # Patches with <= 100 tokens get red overlay
    patch_size=16
):
    """
    Complete ARR-COC homunculus visualization:
    - Green borders: High-detail regions (>=300 tokens)
    - Red overlay: Compressed regions (<=100 tokens)
    - Neutral: Medium-detail regions

    Returns:
        PIL.Image with homunculus overlay
    """
    result = image_pil.copy()

    # 1. Apply red overlay to low-token regions
    rejection_mask = token_allocation <= threshold_low
    result = overlay_rejected_regions(result, rejection_mask, patch_size)

    # 2. Draw green borders around high-token regions
    high_relevance_indices = [
        (i, j) for i in range(token_allocation.shape[0])
               for j in range(token_allocation.shape[1])
               if token_allocation[i, j] >= threshold_high
    ]
    result = highlight_selected_patches(result, high_relevance_indices, patch_size)

    return result
```

### 2.5 Gradio Interface for Patch Selection

```python
def create_homunculus_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# ARR-COC Visual Homunculus")

        with gr.Row():
            input_image = gr.Image(type="pil", label="Input Image")
            output_homunculus = gr.Image(label="Homunculus Visualization")

        with gr.Row():
            threshold_high = gr.Slider(200, 400, 300, label="High Detail Threshold")
            threshold_low = gr.Slider(64, 200, 100, label="Compression Threshold")

        def process(image, t_high, t_low):
            # Run ARR-COC model
            token_allocation = arr_coc_model.allocate_tokens(image)  # (14, 14)

            # Visualize
            homunculus = visualize_arr_coc_homunculus(
                image, token_allocation, t_high, t_low
            )

            return homunculus

        gr.Button("Generate").click(
            fn=process,
            inputs=[input_image, threshold_high, threshold_low],
            outputs=output_homunculus
        )

    return demo
```

## Section 3: Multi-View Layouts (~120 lines)

### 3.1 Side-by-Side Comparison with gr.Row

**Pattern for comparing multiple visualizations**:

```python
def create_comparison_layout():
    with gr.Blocks() as demo:
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Original")
                process_btn = gr.Button("Process")

            # Output columns
            with gr.Column(scale=1):
                out_heatmap = gr.Image(label="Relevance Heatmap")

            with gr.Column(scale=1):
                out_homunculus = gr.Image(label="Patch Selection")

            with gr.Column(scale=1):
                out_overlay = gr.Image(label="Combined Overlay")

        def process_all(image):
            # Generate all visualizations
            heatmap = generate_heatmap_viz(image)
            homunculus = generate_homunculus_viz(image)
            overlay = generate_combined_viz(image)

            return heatmap, homunculus, overlay

        process_btn.click(
            fn=process_all,
            inputs=input_image,
            outputs=[out_heatmap, out_homunculus, out_overlay]
        )

    return demo
```

From [Gradio Controlling Layout](https://www.gradio.app/guides/controlling-layout) (accessed 2025-10-31).

### 3.2 Grid Layouts for Multiple Visualizations

**Pattern for showing many patches/views**:

```python
def create_patch_grid_layout():
    """Display all 196 patches (14x14) in grid format."""
    with gr.Blocks() as demo:
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input")
            extract_btn = gr.Button("Extract Patches")

        # Create 14x14 grid of image components
        patch_grid = []
        for i in range(14):
            with gr.Row():
                row_patches = []
                for j in range(14):
                    patch_img = gr.Image(label=f"({i},{j})", height=50, width=50)
                    row_patches.append(patch_img)
                patch_grid.append(row_patches)

        def extract_patches(image):
            patches = split_into_patches(image, patch_size=16)
            # Flatten list of lists for return
            return [p for row in patches for p in row]

        extract_btn.click(
            fn=extract_patches,
            inputs=input_image,
            outputs=[img for row in patch_grid for img in row]
        )

    return demo
```

### 3.3 Tabbed Views for Different Analysis Modes

**Pattern for organizing complex visualizations**:

```python
def create_tabbed_analysis():
    with gr.Blocks() as demo:
        input_image = gr.Image(type="pil", label="Input Image")
        analyze_btn = gr.Button("Analyze")

        with gr.Tabs():
            with gr.TabItem("Relevance Heatmap"):
                gr.Markdown("Shows spatial relevance scores")
                heatmap_output = gr.Image(label="Heatmap")
                heatmap_stats = gr.Markdown()

            with gr.TabItem("Patch Selection"):
                gr.Markdown("Homunculus visualization")
                homunculus_output = gr.Image(label="Selected Patches")
                selection_stats = gr.Markdown()

            with gr.TabItem("Token Allocation"):
                gr.Markdown("Per-patch token budgets")
                token_heatmap = gr.Image(label="Token Distribution")
                token_histogram = gr.Plot(label="Histogram")

            with gr.TabItem("Raw Data"):
                token_json = gr.JSON(label="Token Allocation Matrix")
                relevance_json = gr.JSON(label="Relevance Scores")

        def analyze(image):
            # Run model
            results = arr_coc_model.analyze(image)

            return (
                results['heatmap'],
                results['heatmap_stats'],
                results['homunculus'],
                results['selection_stats'],
                results['token_heatmap'],
                results['token_histogram'],
                results['token_json'],
                results['relevance_json']
            )

        analyze_btn.click(
            fn=analyze,
            inputs=input_image,
            outputs=[
                heatmap_output, heatmap_stats,
                homunculus_output, selection_stats,
                token_heatmap, token_histogram,
                token_json, relevance_json
            ]
        )

    return demo
```

From [DINOv3-PCA-visualization Gradio Interface](https://huggingface.co/spaces/sayedM/DINOv3-PCA-visualization/blob/main/app.py) (accessed 2025-10-31).

## Section 4: Interactive Visualization (~110 lines)

### 4.1 Click-to-Inspect Patch Details

**Pattern using Gradio SelectData**:

```python
def create_interactive_patch_inspector():
    with gr.Blocks() as demo:
        with gr.Row():
            patch_viz = gr.Image(
                label="Click a patch to inspect",
                interactive=True
            )

            with gr.Column():
                patch_info = gr.Markdown("Click a patch for details")
                patch_tokens = gr.Number(label="Token Budget")
                patch_relevance = gr.Number(label="Relevance Score")

        def inspect_patch(evt: gr.SelectData):
            # evt.index contains click coordinates
            x, y = evt.index

            # Convert pixel coords to patch indices
            patch_i = y // 16
            patch_j = x // 16

            # Lookup patch data
            tokens = token_allocation[patch_i, patch_j]
            relevance = relevance_scores[patch_i, patch_j]

            info_text = f"### Patch ({patch_i}, {patch_j})\n"
            info_text += f"- **Tokens**: {tokens}\n"
            info_text += f"- **Relevance**: {relevance:.3f}"

            return info_text, tokens, relevance

        patch_viz.select(
            fn=inspect_patch,
            outputs=[patch_info, patch_tokens, patch_relevance]
        )

    return demo
```

From [Gradio AnnotatedImage docs](https://www.gradio.app/docs/gradio/annotatedimage) (accessed 2025-10-31).

### 4.2 Interactive Bounding Box Annotation

**Using gradio_image_annotator for manual annotation**:

```python
# Requires: pip install gradio_image_annotation
from gradio_image_annotation import image_annotator

def create_annotation_interface():
    with gr.Blocks() as demo:
        annotator = image_annotator(
            label_list=["High Detail", "Medium Detail", "Low Detail"],
            label_colors=[(0, 255, 0), (255, 255, 0), (255, 0, 0)],
            box_min_size=16,  # Minimum box size = patch size
            show_legend=True
        )

        get_boxes_btn = gr.Button("Get Annotations")
        boxes_json = gr.JSON(label="Annotated Regions")

        def extract_annotations(annotations):
            return annotations['boxes']

        get_boxes_btn.click(
            fn=extract_annotations,
            inputs=annotator,
            outputs=boxes_json
        )

    return demo
```

From [gradio_image_annotator GitHub](https://github.com/edgarGracia/gradio_image_annotator) (accessed 2025-10-31).

### 4.3 Dynamic Threshold Adjustment

**Pattern for real-time visualization updates**:

```python
def create_dynamic_threshold_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            input_image = gr.Image(type="pil", label="Image")
            output_viz = gr.Image(label="Visualization")

        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=64,
                maximum=400,
                value=200,
                step=8,
                label="Token Threshold"
            )
            colormap = gr.Dropdown(
                ['viridis', 'magma', 'jet'],
                value='viridis',
                label="Colormap"
            )

        # Cache model state to avoid recomputation
        state_cache = gr.State()

        def initial_process(image):
            # Run model once
            results = arr_coc_model.analyze(image)

            # Store in state
            state = {
                'token_allocation': results['tokens'],
                'relevance_scores': results['relevance']
            }

            # Generate initial viz
            viz = visualize_with_threshold(
                image, results['tokens'], threshold=200
            )

            return viz, state

        def update_viz(image, threshold, cmap, state):
            # Use cached results, just regenerate viz
            viz = visualize_with_threshold(
                image,
                state['token_allocation'],
                threshold=threshold,
                colormap=cmap
            )
            return viz

        # Initial processing
        input_image.change(
            fn=initial_process,
            inputs=input_image,
            outputs=[output_viz, state_cache]
        )

        # Live threshold updates
        threshold_slider.change(
            fn=update_viz,
            inputs=[input_image, threshold_slider, colormap, state_cache],
            outputs=output_viz
        )

        colormap.change(
            fn=update_viz,
            inputs=[input_image, threshold_slider, colormap, state_cache],
            outputs=output_viz
        )

    return demo
```

## ARR-COC Specific Patterns

### Complete Microscope Module

**Production-ready visualization dashboard for ARR-COC development**:

```python
def create_arr_coc_microscope():
    """
    Complete development microscope for ARR-COC visualization.
    Shows all aspects of relevance realization and token allocation.
    """
    with gr.Blocks(title="ARR-COC Microscope") as demo:
        gr.Markdown("# ARR-COC Visual Microscope")
        gr.Markdown("Visualize adaptive relevance realization and token allocation")

        with gr.Row():
            # Input panel
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                query_text = gr.Textbox(
                    label="Query (for query-aware relevance)",
                    placeholder="Describe what you're looking for..."
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

                with gr.Accordion("Settings", open=False):
                    threshold_high = gr.Slider(250, 400, 320, label="High Detail")
                    threshold_low = gr.Slider(64, 150, 80, label="Compression")
                    colormap = gr.Dropdown(
                        ['viridis', 'magma', 'plasma'],
                        value='viridis',
                        label="Heatmap Colormap"
                    )

            # Visualization panel
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Homunculus"):
                        homunculus_viz = gr.Image(label="Visual Homunculus")
                        gr.Markdown("""
                        - **Green borders**: High-detail regions (>250 tokens)
                        - **Red overlay**: Compressed regions (<80 tokens)
                        """)

                    with gr.TabItem("Relevance Heatmap"):
                        relevance_heatmap = gr.Image(label="Relevance Scores")
                        relevance_overlay = gr.Image(label="Overlay on Image")

                    with gr.TabItem("Token Allocation"):
                        token_heatmap = gr.Image(label="Token Distribution")
                        token_stats = gr.Markdown()

                    with gr.TabItem("Patch Grid"):
                        patch_grid_viz = gr.Image(label="Patch Boundaries")
                        patch_inspector = gr.Image(
                            label="Click to inspect patches",
                            interactive=True
                        )

                    with gr.TabItem("Statistics"):
                        with gr.Row():
                            stats_text = gr.Markdown()
                            stats_plot = gr.Plot(label="Token Distribution")

        def analyze_image(image, query, t_high, t_low, cmap):
            # 1. Run ARR-COC model
            results = arr_coc_model.forward(image, query)

            # 2. Generate all visualizations
            homunculus = visualize_arr_coc_homunculus(
                image, results['token_allocation'], t_high, t_low
            )

            rel_heatmap = smooth_heatmap_overlay(
                results['relevance_scores'], image.size
            )
            rel_overlay = blend_heatmap(image, rel_heatmap, alpha=0.5)

            token_heatmap = smooth_heatmap_overlay(
                results['token_allocation'], image.size
            )

            patch_grid = draw_patch_grid(image, patch_size=16)

            # 3. Generate statistics
            stats = compute_statistics(results['token_allocation'])
            stats_text = format_statistics(stats)
            stats_plot = plot_token_histogram(results['token_allocation'])

            return (
                homunculus,
                rel_heatmap, rel_overlay,
                token_heatmap, stats_text,
                patch_grid, patch_grid,
                stats_text, stats_plot
            )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, query_text, threshold_high, threshold_low, colormap],
            outputs=[
                homunculus_viz,
                relevance_heatmap, relevance_overlay,
                token_heatmap, token_stats,
                patch_grid_viz, patch_inspector,
                stats_text, stats_plot
            ]
        )

    return demo
```

## Best Practices

### Color Choices
- **Green**: High relevance, selected regions (positive connotation)
- **Red**: Low relevance, compressed regions (warning connotation)
- **Yellow/Orange**: Medium relevance (neutral)
- **Colormaps**: Use perceptually uniform maps (viridis, plasma) for scientific visualization

### Performance
- Cache model outputs when only visualization parameters change
- Use PIL ImageDraw for fast patch drawing (faster than matplotlib)
- Resize heatmaps with `Image.Resampling.BICUBIC` for smooth appearance
- For 14×14 patches → 224×224 image, draw directly without upsampling first

### Debugging Tips
- Always show original image alongside visualizations
- Include numerical statistics (mean tokens, min/max, std)
- Make threshold values adjustable with sliders
- Provide downloadable outputs (gr.DownloadButton)

## Sources

**Web Research**:
- [DINOv3-PCA-visualization](https://huggingface.co/spaces/sayedM/DINOv3-PCA-visualization/blob/main/app.py) - Heatmap overlay and blending techniques (accessed 2025-10-31)
- [Gradio AnnotatedImage docs](https://www.gradio.app/docs/gradio/annotatedimage) - Bounding box and mask overlays (accessed 2025-10-31)
- [gradio_image_annotator GitHub](https://github.com/edgarGracia/gradio_image_annotator) - Interactive annotation patterns (accessed 2025-10-31)
- [Fine Tuning Vision Transformer - DebuggerCafe](https://debuggercafe.com/fine-tuning-vision-transformer/) - Attention map visualization (accessed 2025-10-31)
- [Gradio Controlling Layout](https://www.gradio.app/guides/controlling-layout) - Multi-view layouts (accessed 2025-10-31)

**Additional References**:
- [Gradio Plot Component](https://www.gradio.app/docs/gradio/plot) - Matplotlib integration
- [Gradio Image Component](https://www.gradio.app/docs/gradio/image) - Image handling patterns
