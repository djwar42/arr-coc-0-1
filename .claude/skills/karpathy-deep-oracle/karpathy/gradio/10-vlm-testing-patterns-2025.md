# Gradio VLM Testing Patterns (2024-2025)

## Overview

Gradio has emerged as the standard tool for building interactive interfaces for Vision Language Model (VLM) testing and validation. This document covers modern patterns, real-world examples, and best practices for VLM testing interfaces based on 2024-2025 implementations.

**Key Use Cases:**
- Checkpoint comparison and A/B testing
- Ablation study visualization
- Development microscopes (interactive model inspection)
- Demo deployment on HuggingFace Spaces
- Statistical validation interfaces

## Why Gradio for VLM Testing

From [HuggingFace PaliGemma 2 Release](https://huggingface.co/blog/paligemma2) (accessed 2025-10-31):

Gradio provides the fastest path from model to interactive demo:
- **Rapid prototyping**: Build interfaces in <20 lines of code
- **HuggingFace integration**: Native support for transformers models
- **Zero deployment friction**: Push to Spaces, get instant API
- **Researcher-friendly**: No web development knowledge required

From [SmolVLM Release](https://huggingface.co/blog/smolvlm) (accessed 2025-10-31):

The SmolVLM team demonstrates the development microscope pattern - using Gradio to iteratively test model capabilities during development, evaluating checkpoints every 25 optimization steps across multiple benchmarks (MMMU, DocVQA, TextVQA, MathVista).

## Interface Patterns for VLM Testing

### Pattern 1: Image + Query → Output (Basic VLM Interface)

The fundamental VLM testing pattern. From [Gradio Gallery Documentation](https://www.gradio.app/docs/gradio/gallery) (accessed 2025-10-31):

```python
import gradio as gr
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

# Load model
model_id = "google/paligemma2-10b-ft-docci-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

def caption_image(image, prompt):
    """Generate caption for uploaded image"""
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)

    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(output[0][input_len:], skip_special_tokens=True)

# Build interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(label="Prompt", value="caption en")
            submit_btn = gr.Button("Generate Caption")

        with gr.Column():
            output_text = gr.Textbox(label="Generated Caption", lines=5)

    submit_btn.click(
        fn=caption_image,
        inputs=[image_input, prompt_input],
        outputs=output_text
    )

demo.launch()
```

### Pattern 2: Gallery for Batch Testing

From [Gradio Gallery Docs](https://www.gradio.app/docs/gradio/gallery):

```python
import gradio as gr
from typing import List, Tuple

def batch_caption(images: List[Tuple[str, str]]) -> str:
    """Process multiple images and return aggregated results"""
    results = []
    for img_path, caption in images:
        # Process each image
        result = process_vlm(img_path)
        results.append(f"Image: {caption or 'Unnamed'}\nCaption: {result}\n")

    return "\n".join(results)

with gr.Blocks() as demo:
    gallery = gr.Gallery(
        label="Upload Images for Batch Testing",
        show_label=True,
        columns=4,
        object_fit="contain",
        height="auto"
    )

    process_btn = gr.Button("Process All Images")
    output = gr.Textbox(label="Results", lines=10)

    process_btn.click(
        fn=batch_caption,
        inputs=gallery,
        outputs=output
    )

demo.launch()
```

### Pattern 3: Side-by-Side Checkpoint Comparison

Real-world pattern from [PaliGemma 2 Demo](https://huggingface.co/spaces/merve/paligemma2-vqav2) (accessed 2025-10-31):

```python
import gradio as gr

def compare_checkpoints(image, query, checkpoint_a, checkpoint_b):
    """Compare two model checkpoints on same input"""
    # Load models
    model_a = load_checkpoint(checkpoint_a)
    model_b = load_checkpoint(checkpoint_b)

    # Run inference
    output_a = model_a.generate(image, query)
    output_b = model_b.generate(image, query)

    return output_a, output_b

with gr.Blocks() as demo:
    gr.Markdown("# Checkpoint Comparison Tool")

    with gr.Row():
        image = gr.Image(type="pil")
        query = gr.Textbox(label="Query")

    with gr.Row():
        ckpt_a = gr.Dropdown(
            choices=["checkpoint-100", "checkpoint-200", "checkpoint-300"],
            label="Checkpoint A"
        )
        ckpt_b = gr.Dropdown(
            choices=["checkpoint-100", "checkpoint-200", "checkpoint-300"],
            label="Checkpoint B"
        )

    compare_btn = gr.Button("Compare")

    with gr.Row():
        output_a = gr.Textbox(label="Checkpoint A Output", lines=8)
        output_b = gr.Textbox(label="Checkpoint B Output", lines=8)

    compare_btn.click(
        fn=compare_checkpoints,
        inputs=[image, query, ckpt_a, ckpt_b],
        outputs=[output_a, output_b]
    )

demo.launch()
```

### Pattern 4: Multi-View Layouts (Text + Visualization)

Combining text output with visual debugging:

```python
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def vlm_with_attention(image, query):
    """Generate caption + attention heatmap"""
    # Run model
    output, attention_weights = model.generate_with_attention(image, query)

    # Create attention heatmap
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(attention_weights, alpha=0.5, cmap='jet')
    ax.axis('off')

    return output, fig

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            query = gr.Textbox(label="Query")
            submit = gr.Button("Generate")

        with gr.Column():
            output = gr.Textbox(label="Caption")
            attention_plot = gr.Plot(label="Attention Heatmap")

    submit.click(
        fn=vlm_with_attention,
        inputs=[image, query],
        outputs=[output, attention_plot]
    )

demo.launch()
```

## Visualization Integration Patterns

### Heatmap Overlays on Images

For visualizing attention, relevance scores, or LOD allocation (ARR-COC use case):

```python
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def create_heatmap_overlay(image: Image.Image, scores: np.ndarray, alpha=0.5):
    """
    Overlay heatmap on image

    Args:
        image: PIL Image
        scores: 2D numpy array of relevance scores (H x W)
        alpha: Transparency of overlay

    Returns:
        PIL Image with heatmap overlay
    """
    # Normalize scores to 0-1
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    # Apply colormap
    colormap = cm.get_cmap('jet')
    heatmap = colormap(scores_norm)[:, :, :3]  # RGB only
    heatmap = (heatmap * 255).astype(np.uint8)

    # Convert to PIL
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize(image.size)

    # Blend with original
    result = Image.blend(image, heatmap_img, alpha)

    return result

def vlm_with_heatmap(image, query):
    """Generate caption with relevance heatmap"""
    # Get model prediction + relevance scores
    caption, relevance_map = model.predict_with_relevance(image, query)

    # Create visualization
    viz = create_heatmap_overlay(image, relevance_map, alpha=0.4)

    return caption, viz

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(type="pil", label="Input Image")
        query = gr.Textbox(label="Query")

    generate_btn = gr.Button("Generate with Heatmap")

    with gr.Row():
        caption = gr.Textbox(label="Caption")
        heatmap_viz = gr.Image(label="Relevance Heatmap")

    generate_btn.click(
        fn=vlm_with_heatmap,
        inputs=[input_img, query],
        outputs=[caption, heatmap_viz]
    )

demo.launch()
```

### Patch Selection Visualization (ARR-COC Pattern)

From ARR-COC use case - visualizing which image patches are selected/compressed:

```python
import gradio as gr
from PIL import Image, ImageDraw

def draw_patch_selection(image: Image.Image, selected_patches, grid_size=16):
    """
    Visualize patch selection on image

    Args:
        image: Input PIL Image
        selected_patches: List of (row, col) tuples for selected patches
        grid_size: Number of patches per dimension
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')

    w, h = image.size
    patch_w = w // grid_size
    patch_h = h // grid_size

    # Create set for fast lookup
    selected_set = set(selected_patches)

    # Draw grid and highlight selected patches
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = col * patch_w
            y0 = row * patch_h
            x1 = x0 + patch_w
            y1 = y0 + patch_h

            if (row, col) in selected_set:
                # Selected: green border
                draw.rectangle([x0, y0, x1, y1], outline='green', width=2)
            else:
                # Rejected: red overlay
                draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 100))

    return img_copy

def arr_coc_visualization(image, query):
    """Visualize ARR-COC patch selection"""
    # Get model prediction + selected patches
    caption, selected_patches, lod_allocation = model.predict_with_selection(
        image, query
    )

    # Visualize
    viz = draw_patch_selection(image, selected_patches, grid_size=16)

    # Format LOD info
    lod_info = f"Selected patches: {len(selected_patches)}/256\n"
    lod_info += f"Avg tokens/patch: {lod_allocation.mean():.1f}\n"
    lod_info += f"Total tokens: {lod_allocation.sum()}"

    return caption, viz, lod_info

with gr.Blocks() as demo:
    gr.Markdown("# ARR-COC Patch Selection Visualizer")

    with gr.Row():
        image = gr.Image(type="pil", label="Input")
        query = gr.Textbox(label="Query")

    run_btn = gr.Button("Run ARR-COC")

    with gr.Row():
        with gr.Column():
            caption = gr.Textbox(label="Caption", lines=5)
            lod_stats = gr.Textbox(label="LOD Statistics", lines=3)
        with gr.Column():
            patch_viz = gr.Image(label="Patch Selection")

    run_btn.click(
        fn=arr_coc_visualization,
        inputs=[image, query],
        outputs=[caption, patch_viz, lod_stats]
    )

demo.launch()
```

## Real-World HuggingFace Spaces Examples

### Example 1: PaliGemma 2 VQA Demo

From [merve/paligemma2-vqav2](https://huggingface.co/spaces/merve/paligemma2-vqav2) (accessed 2025-10-31):

**Key Features:**
- LoRA fine-tuned model on VQAv2 dataset
- Simple image + question → answer interface
- Live inference on T4 GPU
- Example inputs provided for users

**Code Pattern:**
```python
import gradio as gr
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

# Load base model + LoRA adapter
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-pt-448"
)
model = PeftModel.from_pretrained(base_model, "merve/paligemma2-3b-vqav2")
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-pt-448")

def answer_question(image, question):
    prompt = f"answer {question}"
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    examples=[
        ["example1.jpg", "What color is the car?"],
        ["example2.jpg", "How many people are in the image?"]
    ]
)

demo.launch()
```

### Example 2: SmolVLM Demo

From [HuggingFaceTB/SmolVLM](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM) (accessed 2025-10-31):

**Key Features:**
- Multi-image input support
- Chat-style interface with history
- Demonstrates transjective relevance (query-aware processing)
- Memory-efficient 2B model running on CPU

**Pattern:**
```python
import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16
)

def process_multimodal(images, text):
    """Handle multiple images + text query"""
    # Create message format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"} for _ in images
        ] + [{"type": "text", "text": text}]
    }]

    # Process
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")

    # Generate
    output = model.generate(**inputs, max_new_tokens=500)
    return processor.decode(output[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    with gr.Row():
        images = gr.Gallery(label="Upload Images", file_types=["image"])
        query = gr.Textbox(label="Question", lines=3)

    submit = gr.Button("Ask")
    output = gr.Textbox(label="Answer", lines=8)

    submit.click(
        fn=process_multimodal,
        inputs=[images, query],
        outputs=output
    )

demo.launch()
```

### Example 3: BLIP Image Captioning API

From [Medium: Image Captioning API with Gradio](https://medium.com/@younes_belkada/how-to-write-a-image-captioning-api-using-gradio-and-blip-with-few-lines-of-code-9dfb88254b0) (accessed 2025-10-31):

**Key Pattern: URL-based image input for API usage**

```python
from PIL import Image
import requests
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = BlipProcessor.from_pretrained(model_id)

def caption_from_url(image_url):
    """Caption image from URL - enables API usage"""
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=caption_from_url,
    inputs="text",  # URL as text input
    outputs="text"
)
iface.launch()
```

**API Usage:**
```python
import requests

url = "https://hf.space/embed/ybelkada/blip-api/+/api/predict/"
r = requests.post(
    url=url,
    json={"data": ["https://example.com/image.jpg"]}
)
caption = r.json()["data"][0]
```

## Common Patterns from Community

### Pattern: Example Inputs for User Guidance

From multiple HF Spaces - provide example inputs to guide users:

```python
demo = gr.Interface(
    fn=vlm_function,
    inputs=[gr.Image(type="pil"), gr.Textbox()],
    outputs=gr.Textbox(),
    examples=[
        ["examples/cat.jpg", "What animal is this?"],
        ["examples/chart.png", "Describe this chart"],
        ["examples/document.jpg", "Extract the text"]
    ],
    cache_examples=True  # Pre-compute example outputs
)
```

### Pattern: Tabs for Multiple Views

```python
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Basic Captioning"):
            # Simple interface
            image = gr.Image(type="pil")
            caption = gr.Textbox()
            gr.Button("Caption").click(fn=caption_fn, inputs=image, outputs=caption)

        with gr.Tab("Advanced Analysis"):
            # Detailed analysis with visualization
            image2 = gr.Image(type="pil")
            with gr.Row():
                caption2 = gr.Textbox()
                heatmap = gr.Image()
            gr.Button("Analyze").click(
                fn=analyze_fn,
                inputs=image2,
                outputs=[caption2, heatmap]
            )

        with gr.Tab("Batch Processing"):
            # Gallery-based batch testing
            gallery = gr.Gallery()
            results = gr.Textbox(lines=10)
            gr.Button("Process All").click(
                fn=batch_fn,
                inputs=gallery,
                outputs=results
            )

demo.launch()
```

### Pattern: Interactive Debugging (Click to Inspect)

```python
def on_select(evt: gr.SelectData):
    """Handle click on gallery image"""
    return f"Selected image {evt.index}: {evt.value}"

with gr.Blocks() as demo:
    gallery = gr.Gallery(label="Results", columns=4)
    info = gr.Textbox(label="Selected Info")

    gallery.select(on_select, None, info)

demo.launch()
```

## Best Practices for VLM Testing

### 1. Memory Management

Keep interfaces responsive with proper resource management:

```python
import torch
import gradio as gr

def vlm_inference(image, query):
    try:
        # Move to GPU
        inputs = processor(query, image, return_tensors="pt").to("cuda")

        with torch.no_grad():  # Disable gradients for inference
            output = model.generate(**inputs, max_new_tokens=200)

        result = processor.decode(output[0], skip_special_tokens=True)

        # Clear GPU cache
        del inputs, output
        torch.cuda.empty_cache()

        return result
    except torch.cuda.OutOfMemoryError:
        return "Error: GPU out of memory. Try a smaller image."

demo = gr.Interface(vlm_inference, ...)
demo.launch()
```

### 2. State Management for Sessions

Track metrics across multiple inferences:

```python
import gradio as gr

def process_with_state(image, query, state):
    """Maintain state across calls"""
    if state is None:
        state = {"count": 0, "results": []}

    # Run inference
    result = model.generate(image, query)

    # Update state
    state["count"] += 1
    state["results"].append(result)

    summary = f"Processed {state['count']} images\n"
    summary += f"Latest: {result}"

    return result, summary, state

with gr.Blocks() as demo:
    state = gr.State()  # Persistent state

    image = gr.Image(type="pil")
    query = gr.Textbox()
    submit = gr.Button("Process")

    output = gr.Textbox(label="Result")
    summary = gr.Textbox(label="Session Summary")

    submit.click(
        fn=process_with_state,
        inputs=[image, query, state],
        outputs=[output, summary, state]
    )

demo.launch()
```

### 3. Error Handling and User Feedback

Provide clear feedback for failures:

```python
import gradio as gr

def safe_inference(image, query):
    """Inference with comprehensive error handling"""
    if image is None:
        raise gr.Error("Please upload an image first!")

    if not query.strip():
        gr.Warning("Empty query - using default prompt")
        query = "caption en"

    try:
        result = model.generate(image, query)
        gr.Info("Generation complete!")
        return result
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

demo = gr.Interface(safe_inference, ...)
demo.launch()
```

## Sources

**HuggingFace Blog Posts:**
- [PaliGemma 2 Release](https://huggingface.co/blog/paligemma2) - Google's VLM, Gradio demos, fine-tuning (accessed 2025-10-31)
- [SmolVLM Release](https://huggingface.co/blog/smolvlm) - 2B VLM, development microscope pattern, checkpo comparison (accessed 2025-10-31)

**Gradio Documentation:**
- [Gallery Component](https://www.gradio.app/docs/gradio/gallery) - Official docs for image galleries, batch processing (accessed 2025-10-31)

**Community Resources:**
- [Medium: Image Captioning API with Gradio and BLIP](https://medium.com/@younes_belkada/how-to-write-a-image-captioning-api-using-gradio-and-blip-with-few-lines-of-code-9dfb88254b0) - Younes Belkada, API pattern, URL-based input (accessed 2025-10-31)

**HuggingFace Spaces (Live Demos):**
- [merve/paligemma2-vqav2](https://huggingface.co/spaces/merve/paligemma2-vqav2) - VQA fine-tuned demo
- [HuggingFaceTB/SmolVLM](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM) - Multi-image VLM demo
- [big-vision/paligemma](https://huggingface.co/spaces/big-vision/paligemma) - Original PaliGemma demo

**Related Research:**
- BLIP Paper: [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
- PaliGemma 2 Technical Report: [https://huggingface.co/papers/2412.03555](https://huggingface.co/papers/2412.03555)
