# SAM 3 Batched Inference Patterns

## Overview

SAM 3 provides robust support for batch processing multiple images with the same or different text prompts. This capability is essential for large-scale annotation pipelines, dataset processing, and production deployments requiring high throughput.

## Official Batched Inference Support

SAM 3 includes a dedicated example notebook (`sam3_image_batched_inference.ipynb`) demonstrating batched inference patterns. The API is designed to handle batching, GPU memory management, and format conversions automatically.

**Reference**: [SAM 3 GitHub - Examples](https://github.com/facebookresearch/sam3/tree/main/examples)

## Basic Batch Processing Pattern

### Processing Multiple Images with Different Prompts

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torch

# Initialize model (848M parameters)
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Process multiple images with different text prompts
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
prompts = ["person", "car", "building", "animal", "plant"] * 2

results = []
for image, prompt in zip(images, prompts):
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    results.append(output)
```

**Source**: [stable-learn.com SAM3 Tutorial](https://stable-learn.com/en/sam3-segment-anything-model-tutorial/) (accessed 2025-11-23)

## Throughput Optimization Strategies

### 1. Batch Size Management

```python
def batch_inference(images, prompts, batch_size=4):
    """
    Batch process multiple images and prompts with controlled batch sizes
    """
    results = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        batch_results = []

        with torch.no_grad():  # Disable gradient computation to save memory
            for image, prompt in zip(batch_images, batch_prompts):
                inference_state = processor.set_image(image)
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )
                batch_results.append(output)

        results.extend(batch_results)

        # Clear GPU memory between batches
        torch.cuda.empty_cache()

    return results
```

### 2. Performance Benchmarks

**SAM 3 runs at approximately 30ms per image on H200 GPU**, handling 100+ objects per image efficiently.

| Configuration | Throughput | Notes |
|--------------|------------|-------|
| H200 GPU | ~30 images/sec | Optimal performance |
| A100 GPU | ~20-25 images/sec | Production recommended |
| Consumer GPU (16GB) | ~10-15 images/sec | Requires memory optimization |

**Source**: [Roboflow Blog - What is SAM3](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23)

## Memory Management

### Memory-Efficient Inference Pattern

```python
import gc

def memory_efficient_inference(image, prompt):
    """
    Memory-optimized inference method for constrained environments
    """
    try:
        # Clear previous memory
        torch.cuda.empty_cache()
        gc.collect()

        # Execute inference
        with torch.no_grad():
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

        return output

    finally:
        # Ensure memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
```

### GPU Memory Requirements

| Model Size | Minimum VRAM | Recommended VRAM |
|-----------|--------------|------------------|
| SAM 3 Full (848M params) | 12GB | 16GB+ |
| Batch Processing | 16GB | 24GB+ |
| Large-Scale Pipeline | 24GB | 32GB+ |

### Memory Monitoring

```python
def monitor_performance():
    """
    Monitor SAM3 performance metrics during batch processing
    """
    import psutil

    # GPU memory usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)

        print(f"GPU Total Memory: {gpu_memory / 1e9:.1f} GB")
        print(f"GPU Allocated: {gpu_allocated / 1e9:.1f} GB")
        print(f"GPU Cached: {gpu_cached / 1e9:.1f} GB")

    # System memory usage
    memory = psutil.virtual_memory()
    print(f"System Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / 1e9:.1f} GB")
```

## Use Cases for Batched Inference

### 1. Large-Scale Dataset Annotation

SAM 3's data engine automatically annotated over **4 million unique concepts** using batched inference pipelines. This demonstrates the scalability of the batched processing approach.

**Key Pattern**: Process images in batches with consistent prompts across entire dataset partitions.

```python
# Example: Annotating a large dataset
def annotate_dataset(dataset_path, concepts, batch_size=8):
    """
    Annotate entire dataset with multiple concepts
    """
    import os
    from pathlib import Path

    image_paths = list(Path(dataset_path).glob("*.jpg"))
    annotations = {}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(p) for p in batch_paths]

        for img_path, image in zip(batch_paths, batch_images):
            img_annotations = {}

            for concept in concepts:
                inference_state = processor.set_image(image)
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=concept
                )

                if len(output["masks"]) > 0:
                    img_annotations[concept] = {
                        "masks": output["masks"],
                        "boxes": output["boxes"],
                        "scores": output["scores"]
                    }

            annotations[str(img_path)] = img_annotations

        # Memory cleanup
        torch.cuda.empty_cache()

        # Progress reporting
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

    return annotations
```

### 2. Production API Serving

For production deployments, the API handles batching, GPU memory management, and format conversions automatically.

```python
# Production batch endpoint pattern
class SAM3BatchEndpoint:
    def __init__(self):
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.max_batch_size = 8

    def process_batch(self, image_data_list, prompts):
        """
        Process batch request with automatic memory management
        """
        results = []

        for i in range(0, len(image_data_list), self.max_batch_size):
            batch_images = image_data_list[i:i+self.max_batch_size]
            batch_prompts = prompts[i:i+self.max_batch_size]

            with torch.no_grad():
                batch_results = self._process_batch(batch_images, batch_prompts)

            results.extend(batch_results)
            torch.cuda.empty_cache()

        return results
```

### 3. Video Frame Processing

For video processing, batched inference enables efficient processing of frame sequences:

```python
# Batch process video frames
video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
text_prompts = ["soccer player", "cyclist", "swimmer"]

results = []

for video_file, prompt in zip(video_files, text_prompts):
    print(f"Processing: {video_file} - '{prompt}'")

    # Start new session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_file,
        )
    )

    session_id = response["session_id"]

    # Add text prompt
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )

    results.append({
        "video": video_file,
        "prompt": prompt,
        "output": response["outputs"]
    })
```

### 4. Multi-Concept Simultaneous Segmentation

Process multiple concepts on the same image efficiently:

```python
def multi_concept_batch(image, concepts):
    """
    Segment multiple concepts in a single image efficiently
    """
    all_results = {}

    for concept in concepts:
        # Reset state for each concept
        inference_state = processor.set_image(image)

        output = processor.set_text_prompt(
            state=inference_state,
            prompt=concept
        )

        all_results[concept] = {
            "masks": output["masks"],
            "boxes": output["boxes"],
            "scores": output["scores"]
        }

    return all_results

# Usage
concepts = ["person", "car", "building", "tree", "animal"]
results = multi_concept_batch(image, concepts)
```

## Optimization Best Practices

### 1. Batch Size Selection

- **Small batches (2-4)**: Consumer GPUs with limited memory
- **Medium batches (4-8)**: Production A100/H100 GPUs
- **Large batches (8-16)**: High-memory GPUs with memory optimization

### 2. Gradient Disabling

Always use `torch.no_grad()` during inference:

```python
with torch.no_grad():
    # All inference operations here
    pass
```

### 3. Memory Cleanup Between Batches

```python
# After each batch
torch.cuda.empty_cache()
gc.collect()
```

### 4. Progress Tracking and Logging

```python
import time

def benchmark_sam3(test_images, test_prompts, num_runs=10):
    """
    SAM3 performance benchmark
    """
    times = []

    for i in range(num_runs):
        start_time = time.time()

        for image, prompt in zip(test_images, test_prompts):
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    throughput = len(test_images) / avg_time

    print(f"Average Processing Time: {avg_time:.3f}s")
    print(f"Processing Speed: {throughput:.1f} images/s")

    return avg_time, throughput
```

## Performance Comparison

### SAM 3 vs YOLO11

| Feature | SAM 3 | YOLO11 |
|---------|-------|--------|
| Inference Speed | ~30ms/image | ~2-3ms/image |
| Model Size | 848M params | ~12M params |
| Open Vocabulary | Yes (270K+ concepts) | No (fixed classes) |
| Accuracy (LVIS AP) | 48.5 | - |

**Source**: [Ultralytics YOLO Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23)

## Key Insights

1. **2x Performance Gain**: SAM 3 delivers a 2x gain over existing systems in both image and video Promptable Concept Segmentation (PCS)

2. **Shared Vision Encoder**: The detector and tracker share a vision encoder, improving memory efficiency during batch processing

3. **Automatic Batching**: The SAM3Processor handles batching internally when configured properly

4. **Scalability**: Successfully used to annotate 4M+ concepts, demonstrating large-scale viability

5. **Memory Efficiency**: Decoupled detector-tracker design minimizes task interference and scales efficiently with data

## Sources

**Official Resources**:
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) - Official code and examples
- [SAM 3 Batched Inference Notebook](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_batched_inference.ipynb) - Official batched inference examples
- [Meta AI SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/) - Performance benchmarks

**Web Research**:
- [stable-learn.com SAM3 Tutorial](https://stable-learn.com/en/sam3-segment-anything-model-tutorial/) - Batch inference optimization patterns (accessed 2025-11-23)
- [Roboflow Blog - What is SAM3](https://blog.roboflow.com/what-is-sam3/) - Performance benchmarks ~30ms on H200 (accessed 2025-11-23)
- [Ultralytics YOLO Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) - Comparison with YOLO (accessed 2025-11-23)
- [Medium - SAM 3 Article](https://medium.com/@harsh.vardhan7695/meta-sam-3-the-ai-that-understands-find-every-red-hat-b489d341977b) - API batching and memory management (accessed 2025-11-23)

**Additional References**:
- [HuggingFace - facebook/sam3](https://huggingface.co/facebook/sam3) - Model card and usage
- [ModelScope - SAM 3](https://modelscope.cn/models/facebook/sam3) - Chinese documentation with batch examples
