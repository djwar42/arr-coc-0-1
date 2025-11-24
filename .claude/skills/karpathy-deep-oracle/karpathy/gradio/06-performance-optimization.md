# Gradio Performance Optimization

**Current as of**: January 2025
**Target**: Gradio 5.x for high-traffic production applications

This guide covers strategies to maximize performance and minimize latency for Gradio applications handling thousands of concurrent users.

---

## Section 1: Queue Configuration for Concurrency

**Source**: [gradio.app/guides/setting-up-a-demo-for-maximum-performance](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance)

### Overview of Gradio's Queueing System

Every Gradio demo includes a built-in queuing system that scales to thousands of requests using Server-Side Events (SSE).

**Why SSE over HTTP POST?**
1. **No timeout**: Browsers timeout POST requests after ~1 minute
2. **Multiple updates**: Server can send real-time ETAs and progress
3. **Long-running**: Perfect for ML inference that takes >1 minute

**Single-Function-Single-Worker Model**

By default, Gradio assigns one worker per function type (not per request):

```
Requests:  1   2   3   4   5   6   7
Functions: A   B   A   A   C   B   A
Workers:   w1  w2  w1  w3  w4  w2  w1
```

- Worker 1 handles all "A" requests
- Worker 2 handles all "B" requests
- Worker 4 handles all "C" requests

This prevents out-of-memory errors from multiple workers loading the same ML model simultaneously.

### Parameter: `default_concurrency_limit` in `queue()`

**Default**: `1` (one worker per function)
**Purpose**: Control how many workers execute the same event simultaneously

```python
import gradio as gr

app = gr.Interface(lambda x: x, "image", "image")
app.queue(default_concurrency_limit=10)  # 10 parallel workers per function
app.launch()
```

**When to increase:**
- Function doesn't call resource-intensive operations
- Function queries external APIs (I/O bound, not CPU/GPU bound)
- App has sufficient memory to handle parallel requests

**When to keep low:**
- Function loads large ML models
- GPU memory is limited
- Risk of out-of-memory errors

**Impact**: Linearly multiplies server capacity until memory/switching overhead limits are hit.

**Recommendation**: Increase gradually while monitoring memory usage. Start at 5-10 for API-heavy apps, stay at 1-3 for GPU-intensive models.

### Parameter: `concurrency_limit` in Events

**Per-event override** of `default_concurrency_limit`:

```python
with gr.Blocks() as demo:
    # Fast text processing: high concurrency
    text_btn.click(
        fast_text_fn,
        inputs=[...],
        outputs=[...],
        concurrency_limit=20
    )

    # GPU model inference: low concurrency
    image_btn.click(
        gpu_inference_fn,
        inputs=[...],
        outputs=[...],
        concurrency_limit=2
    )

demo.queue(default_concurrency_limit=5)
demo.launch()
```

**Use case**: Mix of fast and slow functions in same app.

### Parameter: `max_threads` in `launch()`

**Default**: 40 threads
**Purpose**: Size of threadpool for non-async functions

```python
demo.launch(max_threads=80)
```

**Important**: Only affects `def` functions (not `async def`).

**Best practice**: Convert CPU-bound quick functions to `async def` when possible:

```python
# ❌ Blocks threadpool
def quick_process(text):
    return text.upper()

# ✅ Async (doesn't block threadpool)
async def quick_process(text):
    return text.upper()
```

**When to increase `max_threads`**: Non-async functions hitting the 40-thread limit.

### Parameter: `max_size` in `queue()`

**Default**: `None` (unlimited queue)
**Purpose**: Maximum number of requests in queue

```python
demo.queue(max_size=100)  # Queue holds max 100 requests
```

**Benefit**: Prevents extremely long wait times that discourage users.

**Trade-off**: Users see "Queue full, try again" error instead of waiting 10+ minutes.

**Recommendation**: Set based on acceptable wait time. If average inference is 10s, `max_size=30` means ~5min max wait (acceptable for most users).

---

## Section 2: Batch Functions for Throughput

**Source**: [gradio.app/guides/batch-functions](https://www.gradio.app/guides/batch-functions)

Batch processing allows Gradio to automatically group incoming requests and process them together, significantly speeding up inference for deep learning models.

### How Batch Functions Work

**Without batching:**
- 16 requests arrive
- Each processed sequentially (16 × 5s = 80s total)

**With batching (max_batch_size=16):**
- 16 requests batched together
- Processed in parallel (1 × 5s = 5s total)

### Implementing Batch Functions

**Non-batched function:**
```python
import time

def trim_words(word, length):
    time.sleep(5)
    return word[:int(length)]
```

**Batched function:**
```python
import time

def trim_words(words, lengths):
    """Process list of inputs, return list of outputs"""
    time.sleep(5)  # Simulate model inference on batch
    trimmed_words = []
    for w, l in zip(words, lengths):
        trimmed_words.append(w[:int(l)])
    return [trimmed_words]  # Must return list of outputs
```

### Using Batched Functions

**With `gr.Interface`:**
```python
demo = gr.Interface(
    fn=trim_words,
    inputs=["textbox", "number"],
    outputs=["output"],
    batch=True,
    max_batch_size=16
)

demo.launch()
```

**With `gr.Blocks`:**
```python
with gr.Blocks() as demo:
    with gr.Row():
        word = gr.Textbox(label="word")
        leng = gr.Number(label="leng")
        output = gr.Textbox(label="Output")
    run = gr.Button()

    run.click(
        trim_words,
        [word, leng],
        output,
        batch=True,
        max_batch_size=16
    )

demo.launch()
```

### Batch Function Requirements

1. **Input**: Function receives **lists** of inputs
2. **Output**: Function returns **lists** of outputs
3. **Synchronous**: Processes entire batch before returning
4. **Consistent**: All inputs must be processed in order

### Real-World Example: Image Generation

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

def generate_images(prompts):
    """Generate batch of images from batch of prompts"""
    images = pipe(prompts).images  # Process batch at once
    return images

demo = gr.Interface(
    fn=generate_images,
    inputs=gr.Textbox(),
    outputs=gr.Image(),
    batch=True,
    max_batch_size=8  # Generate 8 images at once
)

demo.queue().launch()
```

### Choosing `max_batch_size`

**Factors to consider:**
- **GPU memory**: Larger batches require more VRAM
- **Latency**: Larger batches wait longer to fill
- **Throughput**: Sweet spot balances wait time vs processing speed

**Typical values:**
- **Text models** (BERT, GPT): 16-32
- **Image models** (ResNet, ViT): 8-16
- **Diffusion models**: 4-8
- **Large VLMs**: 2-4

**Recommendation**: Start with batch size your model can handle in memory, then reduce if users experience long waits for batches to fill.

### Batch vs Concurrency

| Strategy | Best For | Tradeoff |
|----------|----------|----------|
| **Batching** | Deep learning models | Requires code changes, waiting for batch to fill |
| **Concurrency** | Any function | Memory overhead, potential resource contention |
| **Both** | Production ML apps | Maximum throughput, requires tuning |

**Combine strategies:**
```python
def batched_inference(inputs_batch):
    # Process batch efficiently
    return model(inputs_batch)

demo = gr.Interface(
    batched_inference,
    inputs="text",
    outputs="text",
    batch=True,
    max_batch_size=16,
    concurrency_limit=4  # 4 workers, each processing batches of 16
)

demo.queue(default_concurrency_limit=4).launch()
```

---

## Section 3: Caching Strategies

**Source**: [gradio.app/guides/setting-up-a-demo-for-maximum-performance](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance)

### Example Caching Pattern

Gradio automatically caches `examples` to speed up demo loading:

```python
demo = gr.Interface(
    fn=expensive_model,
    inputs="text",
    outputs="text",
    examples=[
        ["Example 1"],
        ["Example 2"],
        ["Example 3"]
    ],
    cache_examples=True  # Precompute examples on launch
)
```

**How it works:**
1. On app launch, Gradio runs all examples through your function
2. Results cached to disk
3. Users clicking examples get instant results (no computation)

**Use case**: Showcase model capabilities without waiting

### Custom Caching with `functools`

```python
from functools import lru_cache
import gradio as gr

@lru_cache(maxsize=128)
def expensive_computation(text):
    # Expensive operation cached in memory
    result = perform_heavy_computation(text)
    return result

demo = gr.Interface(expensive_computation, "text", "text")
demo.launch()
```

**Limitations:**
- Works for hashable inputs only (strings, numbers, tuples)
- Memory-based (cleared on restart)

### Persistent Caching with Redis

```python
import gradio as gr
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_inference(prompt):
    # Generate cache key
    cache_key = hashlib.md5(prompt.encode()).hexdigest()

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute result
    result = expensive_model(prompt)

    # Store in cache (expire after 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(result))

    return result

demo = gr.Interface(cached_inference, "text", "text")
demo.launch()
```

**Advantages:**
- Persists across restarts
- Shared across multiple workers
- Automatic expiration

---

## Section 4: Resource Cleanup & Memory Management

**Source**: [gradio.app/guides/resource-cleanup](https://www.gradio.app/guides/resource-cleanup)

### The Problem: Memory Leaks

Without cleanup, long-running Gradio apps can accumulate:
- Temporary files from uploads
- Model instances in GPU memory
- Cached data structures
- Open database connections

### Cleanup Pattern with Context Managers

```python
import gradio as gr
import tempfile
import os

def process_file(file):
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file)
        tmp_path = tmp.name

    try:
        # Process file
        result = heavy_processing(tmp_path)
        return result
    finally:
        # Always cleanup, even if error occurs
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

demo = gr.Interface(process_file, "file", "text")
demo.launch()
```

### Cleanup with `@gr.render()` Lifecycle

```python
import gradio as gr

class ModelManager:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = load_large_model()  # Expensive
        return self.model

    def cleanup(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

manager = ModelManager()

with gr.Blocks() as demo:
    state = gr.State(value=manager)

    # Load on first use
    def inference(input, state):
        model = state.load()
        result = model(input)
        return result

    # Cleanup on session end (if implemented)
    demo.unload(lambda: manager.cleanup())
```

### GPU Memory Management

```python
import torch
import gradio as gr

def gpu_inference(image):
    # Move to GPU
    image_tensor = preprocess(image).to("cuda")

    try:
        # Inference
        with torch.no_grad():  # Disable gradient computation
            result = model(image_tensor)

        # Move back to CPU immediately
        result = result.cpu()

        return result

    finally:
        # Clear GPU cache
        torch.cuda.empty_cache()

demo = gr.Interface(gpu_inference, "image", "image")
demo.launch()
```

### File Upload Cleanup

Gradio stores uploads in temporary directory. Clean them proactively:

```python
import gradio as gr
import os
from pathlib import Path

def process_video(video_path):
    try:
        result = video_processing(video_path)
        return result
    finally:
        # Cleanup uploaded file after processing
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Failed to cleanup {video_path}: {e}")

demo = gr.Interface(
    process_video,
    gr.Video(),
    gr.Video()
)

demo.launch()
```

---

## Section 5: Scaling with Ray Serve

**Source**: [docs.ray.io/en/latest/serve/tutorials/gradio-integration.html](https://docs.ray.io/en/latest/serve/tutorials/gradio-integration.html) (referenced in performance guide)

### Why Ray Serve?

For applications requiring extreme scale:
- **Horizontal scaling**: Multiple machines, not just workers
- **Autoscaling**: Automatically add/remove replicas based on traffic
- **Model composition**: Chain multiple models with different hardware requirements

### Basic Ray Serve + Gradio Pattern

```python
from ray import serve
import gradio as gr

@serve.deployment
class GradioDeployment:
    def __init__(self):
        self.model = load_model()

    def predict(self, input):
        return self.model(input)

    def __call__(self, request):
        # Ray Serve entry point
        return self.app(request)

    def build_app(self):
        self.app = gr.Interface(
            fn=self.predict,
            inputs="text",
            outputs="text"
        ).queue().launch(prevent_thread_lock=True)

        return self.app

# Deploy
deployment = GradioDeployment.bind()
serve.run(deployment)
```

### Autoscaling Configuration

```python
@serve.deployment(
    num_replicas=2,  # Start with 2 replicas
    max_concurrent_queries=10,  # Per replica
    ray_actor_options={
        "num_gpus": 1  # Each replica gets 1 GPU
    }
)
class ScalingGradioDeployment:
    # ... implementation
```

**Note**: Ray Serve is advanced. Start with Gradio's built-in queue for most applications.

---

## Best Practices Summary

### Performance Optimization Hierarchy

1. **Start with basics**: Enable queue, set reasonable `default_concurrency_limit`
2. **Use async**: Convert quick functions to `async def`
3. **Batch when possible**: Rewrite model inference to accept batches
4. **Cache intelligently**: Examples, repeated inputs
5. **Clean up resources**: Memory, GPU, temp files
6. **Monitor and tune**: Adjust limits based on real traffic
7. **Scale horizontally**: Ray Serve for extreme scale only

### Quick Win Checklist

- [ ] Enable queue: `demo.queue()`
- [ ] Set concurrency: `default_concurrency_limit=5`
- [ ] Use batching: `batch=True, max_batch_size=16`
- [ ] Cache examples: `cache_examples=True`
- [ ] Clean up GPU: `torch.cuda.empty_cache()`
- [ ] Set max queue: `max_size=100`
- [ ] Use async functions where possible
- [ ] Monitor memory usage and adjust

### Hardware Recommendations

**CPU-only apps:**
- Concurrency limit: 10-20
- Batch size: Not critical
- Focus: Async functions, caching

**GPU apps (single GPU):**
- Concurrency limit: 2-4
- Batch size: 8-16
- Focus: Batching, GPU memory cleanup

**GPU apps (multiple GPUs):**
- Concurrency limit: 2-4 per GPU
- Batch size: 8-16
- Focus: Load balancing across GPUs

**Hugging Face Spaces:**
- Free tier: concurrency_limit=1, small batches
- A10G GPU: concurrency_limit=2-3, batch_size=8-16
- A100 GPU: concurrency_limit=4-6, batch_size=16-32

---

## Summary

Gradio performance optimization involves:

1. **Queue tuning**: Adjust concurrency limits for your hardware
2. **Batch processing**: Group requests for GPU efficiency
3. **Caching**: Avoid redundant computation
4. **Resource management**: Clean up memory and files
5. **Scaling**: Horizontal scaling with Ray Serve for extreme traffic

Start conservative, monitor performance, and increase parallelism gradually. Most apps achieve 10-50x throughput improvement with proper configuration.
