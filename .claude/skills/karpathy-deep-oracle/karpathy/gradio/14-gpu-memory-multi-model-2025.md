# GPU Memory Management for Multiple Models in Gradio (2025)

## Overview

Managing GPU memory when loading multiple model checkpoints in Gradio applications is critical for ARR-COC microscope modules and VLM testing interfaces. This guide covers LRU checkpoint managers, memory optimization strategies, graceful OOM handling, and hardware-specific optimizations for T4 (16GB) and A100 (40GB) GPUs.

**Key Challenge**: When switching between multiple VLM checkpoints in Gradio, improper memory management causes memory leaks, OOM errors, and degraded performance. The root cause is often hidden references that prevent Python's garbage collector from freeing GPU memory.

## Memory Challenges with Multiple Checkpoints

### Common Scenarios

**1. Multi-Checkpoint Comparison (ARR-COC Use Case)**
- Loading checkpoints A, B, C for side-by-side validation
- Each SDXL checkpoint: ~6-7GB VRAM (fp16)
- Each Flux checkpoint: ~12-15GB VRAM (bf16)
- T4 GPU (16GB): Can fit 1-2 SDXL models max
- A100 GPU (40GB): Can fit 2-3 Flux models or 5-6 SDXL models

**2. Memory Leak Symptoms**
- VRAM usage increases after each checkpoint switch
- Memory never returns to baseline
- Eventually hits OOM and crashes
- Affects both GPU VRAM and CPU RAM with `enable_model_cpu_offload()`

**3. Root Causes**
- Pipeline loaded in global scope before Gradio context
- Hidden references in Gradio's event loop
- Components not explicitly deleted before loading new checkpoint
- Tensors not moved to CPU before deletion

From [Diffusers GitHub Discussion #10936](https://github.com/huggingface/diffusers/discussions/10936) (accessed 2025-10-31):
> "By creating the pipeline instance and loading the model before Gradio was fully initialized, the pipeline and its memory were created outside Gradio's managed environment. When switching models within Gradio event handlers, cleanup operations didn't fully work because the initial model was loaded in a different context."

## LRU Checkpoint Manager Implementation

### Design Pattern for T4 GPU

**Concept**: Automatically evict least-recently-used checkpoints when loading new ones, maintaining a maximum of N loaded models (typically 2 for T4, 3-4 for A100).

```python
import gc
import torch
from collections import OrderedDict
from diffusers import StableDiffusionXLPipeline

class CheckpointManagerLRU:
    """LRU cache-based checkpoint manager for multiple VLM models.

    Critical for ARR-COC microscope modules that compare multiple checkpoints.
    Automatically evicts least-recently-used models to stay within VRAM limits.
    """

    def __init__(self, max_loaded=2, device="cuda", torch_dtype=torch.float16):
        """
        Args:
            max_loaded: Maximum number of simultaneously loaded checkpoints
                       T4 (16GB): 2 for SDXL, 1 for Flux
                       A100 (40GB): 4 for SDXL, 2-3 for Flux
            device: Target device for models
            torch_dtype: Default dtype (float16 for SDXL, bfloat16 for Flux)
        """
        self.max_loaded = max_loaded
        self.device = device
        self.torch_dtype = torch_dtype
        self.loaded_checkpoints = OrderedDict()  # Maintains insertion order
        self.checkpoint_configs = {}

    def register_checkpoint(self, name, model_id, **pipeline_kwargs):
        """Register a checkpoint configuration without loading it."""
        self.checkpoint_configs[name] = {
            "model_id": model_id,
            "pipeline_kwargs": pipeline_kwargs
        }

    def load_checkpoint(self, name):
        """Load checkpoint with LRU eviction."""
        # If already loaded, move to end (most recently used)
        if name in self.loaded_checkpoints:
            self.loaded_checkpoints.move_to_end(name)
            return self.loaded_checkpoints[name]

        # Evict least recently used if at capacity
        if len(self.loaded_checkpoints) >= self.max_loaded:
            self._evict_lru()

        # Load new checkpoint
        config = self.checkpoint_configs[name]
        print(f"Loading checkpoint: {name}")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=self.torch_dtype,
            **config["pipeline_kwargs"]
        ).to(self.device)

        # Apply memory optimizations
        pipeline.enable_vae_tiling()
        pipeline.enable_vae_slicing()

        self.loaded_checkpoints[name] = pipeline
        return pipeline

    def _evict_lru(self):
        """Evict least recently used checkpoint."""
        if not self.loaded_checkpoints:
            return

        # Get first item (least recently used)
        lru_name, lru_pipeline = self.loaded_checkpoints.popitem(last=False)
        print(f"Evicting checkpoint: {lru_name}")

        # Critical cleanup sequence
        lru_pipeline.to("cpu")  # Move to CPU first

        # Delete all components explicitly
        for component_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
            if hasattr(lru_pipeline, component_name):
                component = getattr(lru_pipeline, component_name)
                if component is not None:
                    del component
                    setattr(lru_pipeline, component_name, None)

        del lru_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def get_memory_stats(self):
        """Get current GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "loaded_count": len(self.loaded_checkpoints),
                "loaded_names": list(self.loaded_checkpoints.keys())
            }
        return None
```

### Integration with Gradio

**Critical: Initialize INSIDE Gradio Context**

From [Diffusers GitHub Discussion #10936](https://github.com/huggingface/diffusers/discussions/10936) (accessed 2025-10-31):

```python
import gradio as gr
from checkpoint_manager import CheckpointManagerLRU

# WRONG: Global initialization before Gradio
# manager = CheckpointManagerLRU(max_loaded=2)  # DON'T DO THIS

def initialize_manager():
    """Initialize manager INSIDE Gradio context."""
    global manager
    if 'manager' not in globals():
        manager = CheckpointManagerLRU(max_loaded=2)

        # Register all checkpoints
        manager.register_checkpoint(
            "RealVisXL_v5",
            "SG161222/RealVisXL_V5.0"
        )
        manager.register_checkpoint(
            "RealVisXL_Lightning",
            "SG161222/RealVisXL_V5.0_Lightning"
        )
    return "Manager initialized"

def switch_checkpoint(checkpoint_name):
    """Switch to a different checkpoint."""
    pipeline = manager.load_checkpoint(checkpoint_name)
    stats = manager.get_memory_stats()

    return f"Loaded: {checkpoint_name}\nMemory: {stats['allocated_gb']:.2f}GB"

with gr.Blocks() as app:
    checkpoint_selector = gr.Dropdown(
        label="Select Checkpoint",
        choices=["RealVisXL_v5", "RealVisXL_Lightning"],
        value="RealVisXL_v5"
    )
    status_output = gr.Textbox(label="Status")
    switch_button = gr.Button("Switch Checkpoint")

    # Initialize on load - CRITICAL!
    app.load(
        fn=initialize_manager,
        inputs=None,
        outputs=status_output
    )

    switch_button.click(
        fn=switch_checkpoint,
        inputs=checkpoint_selector,
        outputs=status_output
    )

app.launch()
```

## Explicit Memory Management Patterns

### Component-Level Cleanup

From [Diffusers GitHub Discussion #10936](https://github.com/huggingface/diffusers/discussions/10936) (accessed 2025-10-31):

```python
def clear_pipeline_memory(pipeline):
    """Explicit cleanup for Diffusers pipelines."""
    if pipeline is None:
        return

    # Step 1: Move entire pipeline to CPU
    pipeline.to("cpu")

    # Step 2: Delete components individually
    components_to_delete = [
        "unet", "vae", "text_encoder", "text_encoder_2",
        "controlnet", "safety_checker", "feature_extractor"
    ]

    for component_name in components_to_delete:
        if hasattr(pipeline, component_name):
            component = getattr(pipeline, component_name)
            if component is not None:
                del component
                setattr(pipeline, component_name, None)

    # Step 3: Delete pipeline itself
    del pipeline

    # Step 4: Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Optional: Synchronize CUDA operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```

### Advanced Cleanup with Referrer Tracking

From [Diffusers GitHub Discussion #10936](https://github.com/huggingface/diffusers/discussions/10936) (accessed 2025-10-31):

```python
import gc

def delete_with_referrers(obj):
    """Delete object and clear all referrers.

    Useful when standard del doesn't free memory due to hidden references.
    """
    if obj is None:
        return 0, [], 0, []

    deleted_count = 0
    deleted_refs = []
    skipped_count = 0
    skipped_refs = []

    for referrer in gc.get_referrers(obj):
        if hasattr(referrer, "__dict__"):
            __dict__ = object.__getattribute__(referrer, "__dict__")
        elif isinstance(referrer, dict):
            __dict__ = referrer
        elif isinstance(referrer, list):
            for index, item in enumerate(referrer):
                if item is obj:
                    referrer[index] = None
                    deleted_refs.append(f"list.{index}")
                    deleted_count += 1
            continue
        else:
            skipped_refs.append(id(referrer))
            skipped_count += 1
            continue

        target_keys = []
        for key, value in __dict__.items():
            if value is obj:
                target_keys.append(key)
                deleted_refs.append(f"dict.{key}")
                deleted_count += 1

        for target_key in target_keys:
            __dict__.update({target_key: None})

    return deleted_count, deleted_refs, skipped_count, skipped_refs

# Usage for pipeline components
for component in pipeline.components.values():
    delete_with_referrers(component)
```

### CPU Offloading with Proper Cleanup

From [HuggingFace Diffusers Optimization Docs](https://huggingface.co/docs/diffusers/en/optimization/memory) (accessed 2025-10-31):

```python
def disable_cpu_offload(pipeline):
    """Remove CPU offloading hooks to enable proper cleanup."""
    from accelerate.hooks import remove_hook_from_module
    from accelerate.hooks import AlignDevicesHook, CpuOffload

    is_model_cpu_offload = False
    is_sequential_cpu_offload = False

    if pipeline is not None:
        for _, component in pipeline.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "_hf_hook"):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)

                # Remove hooks to allow proper deletion
                remove_hook_from_module(component, recurse=True)

    return is_model_cpu_offload, is_sequential_cpu_offload
```

## Graceful OOM Handling

### Detection and Fallback Strategy

```python
import gradio as gr
import torch

def safe_inference_wrapper(inference_fn):
    """Wrapper that catches OOM and falls back to CPU or reduced batch."""

    def wrapped_function(*args, **kwargs):
        try:
            # Try GPU inference
            return inference_fn(*args, **kwargs)

        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU OOM detected: {e}")

            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Try with CPU offloading
            try:
                if hasattr(inference_fn, '__self__'):
                    pipeline = inference_fn.__self__
                    pipeline.enable_model_cpu_offload()
                    result = inference_fn(*args, **kwargs)
                    pipeline.disable_model_cpu_offload()
                    return result
            except Exception as e2:
                print(f"CPU offload failed: {e2}")

            # Final fallback: informative error
            raise gr.Error(
                "GPU memory exhausted. Try:\n"
                "1. Reduce batch size or image resolution\n"
                "2. Close other GPU applications\n"
                "3. Select a smaller checkpoint"
            )

        except Exception as e:
            # Catch other errors gracefully
            raise gr.Error(f"Inference error: {str(e)}")

    return wrapped_function

# Usage in Gradio
@safe_inference_wrapper
def generate_image(prompt, checkpoint_name):
    pipeline = manager.load_checkpoint(checkpoint_name)
    return pipeline(prompt).images[0]
```

### Memory Monitoring in Gradio

From [Gradio Resource Cleanup Docs](https://www.gradio.app/guides/resource-cleanup) (accessed 2025-10-31):

```python
import psutil
import torch

def get_memory_usage():
    """Get comprehensive memory statistics."""
    stats = {}

    # CPU RAM
    cpu_mem = psutil.virtual_memory()
    stats["cpu_used_gb"] = cpu_mem.used / 1024**3
    stats["cpu_available_gb"] = cpu_mem.available / 1024**3
    stats["cpu_percent"] = cpu_mem.percent

    # GPU VRAM
    if torch.cuda.is_available():
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        stats["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        # Get total GPU memory
        props = torch.cuda.get_device_properties(0)
        stats["gpu_total_gb"] = props.total_memory / 1024**3

    return stats

def create_memory_dashboard():
    """Create real-time memory monitoring dashboard."""
    with gr.Blocks() as demo:
        memory_display = gr.JSON(label="Memory Stats")
        refresh_btn = gr.Button("Refresh Memory Stats")

        refresh_btn.click(
            fn=get_memory_usage,
            inputs=None,
            outputs=memory_display
        )

        # Auto-refresh every 5 seconds
        demo.load(
            fn=get_memory_usage,
            inputs=None,
            outputs=memory_display,
            every=5
        )

    return demo
```

## T4 vs A100 Optimization Strategies

### T4 GPU (16GB VRAM) Strategy

**Constraints:**
- Limited VRAM for large models
- Lower compute capability (7.5 vs 8.0)
- No fp8 support

**Optimizations:**

```python
class T4OptimizedManager(CheckpointManagerLRU):
    """Checkpoint manager optimized for T4 GPUs (16GB)."""

    def __init__(self):
        super().__init__(
            max_loaded=1,  # Only 1 large model at a time
            device="cuda",
            torch_dtype=torch.float16  # fp16 only on T4
        )

    def load_checkpoint(self, name):
        """Load with T4-specific optimizations."""
        pipeline = super().load_checkpoint(name)

        # Enable all memory-saving features
        pipeline.enable_vae_tiling()
        pipeline.enable_vae_slicing()
        pipeline.enable_model_cpu_offload()  # Offload to CPU

        # Use attention slicing for large images
        pipeline.enable_attention_slicing(slice_size="auto")

        return pipeline
```

**T4 Best Practices:**
- `max_loaded=1` for Flux/SDXL, `max_loaded=2` for smaller models
- Always use `enable_model_cpu_offload()` for large models
- Enable VAE tiling for images >768px
- Use fp16 (bfloat16 not well-supported on T4)
- Consider quantization (int8) for extreme memory savings

### A100 GPU (40GB VRAM) Strategy

**Advantages:**
- Large VRAM capacity
- Better compute capability (8.0)
- Full bfloat16 support
- Faster memory bandwidth

**Optimizations:**

```python
class A100OptimizedManager(CheckpointManagerLRU):
    """Checkpoint manager optimized for A100 GPUs (40GB)."""

    def __init__(self):
        super().__init__(
            max_loaded=3,  # Can hold 3 Flux or 5-6 SDXL models
            device="cuda",
            torch_dtype=torch.bfloat16  # bf16 for better quality
        )

    def load_checkpoint(self, name):
        """Load with A100-specific optimizations."""
        pipeline = super().load_checkpoint(name)

        # Minimal offloading - keep on GPU
        pipeline.enable_vae_tiling()  # Still useful for very large images

        # Can skip CPU offloading on A100
        # pipeline.enable_model_cpu_offload()  # Not needed

        # Use xformers for faster attention
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass  # Fall back to standard attention

        return pipeline
```

**A100 Best Practices:**
- `max_loaded=3-4` for large models
- Use bfloat16 for better numerical stability
- Skip CPU offloading unless comparing 4+ models
- Enable xformers for speed
- Can use larger batch sizes (2-4 images)

### Comparison Table

| Feature | T4 (16GB) | A100 (40GB) |
|---------|-----------|-------------|
| Max SDXL checkpoints | 2 | 5-6 |
| Max Flux checkpoints | 1 | 2-3 |
| Recommended dtype | float16 | bfloat16 |
| CPU offloading | Required | Optional |
| VAE tiling | Always | For >1024px |
| Batch size | 1 | 2-4 |
| xformers | Optional | Recommended |

## Advanced Memory Techniques

### VAE Slicing and Tiling

From [Gradio Performance Guide](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance) (accessed 2025-10-31):

```python
def configure_vae_optimizations(pipeline, image_size=(1024, 1024)):
    """Configure VAE for memory efficiency."""

    # VAE slicing: Process batch one image at a time
    # Useful when generating multiple images (batch > 1)
    pipeline.enable_vae_slicing()

    # VAE tiling: Process image in tiles
    # Critical for high-resolution images (>768px)
    if image_size[0] > 768 or image_size[1] > 768:
        pipeline.enable_vae_tiling()

    return pipeline
```

### Attention Slicing

```python
def configure_attention_slicing(pipeline, vram_gb=16):
    """Configure attention slicing based on VRAM."""

    if vram_gb <= 16:  # T4
        # Maximum slicing for T4
        pipeline.enable_attention_slicing(slice_size="auto")
    elif vram_gb <= 24:  # RTX 3090/4090
        # Moderate slicing
        pipeline.enable_attention_slicing(slice_size=1)
    else:  # A100/A6000
        # Minimal or no slicing
        pipeline.disable_attention_slicing()

    return pipeline
```

### Channels Last Memory Format

From [HuggingFace Diffusers Memory Docs](https://huggingface.co/docs/diffusers/en/optimization/memory) (accessed 2025-10-31):

```python
def optimize_memory_format(pipeline):
    """Use channels_last for better memory locality."""

    # Convert UNet to channels_last
    pipeline.unet.to(memory_format=torch.channels_last)

    # Verify conversion
    weight_stride = pipeline.unet.conv_out.state_dict()["weight"].stride()
    print(f"Weight stride: {weight_stride}")
    # Should show stride of 1 for 2nd dimension

    return pipeline
```

## Production Error Handling

### User-Friendly Error Messages

From [Gradio Resource Cleanup](https://www.gradio.app/guides/resource-cleanup) (accessed 2025-10-31):

```python
def create_robust_inference_handler():
    """Production-ready inference with comprehensive error handling."""

    def inference_handler(prompt, checkpoint_name, **kwargs):
        try:
            # Load checkpoint with LRU manager
            pipeline = manager.load_checkpoint(checkpoint_name)

            # Check memory before inference
            stats = manager.get_memory_stats()
            if stats["allocated_gb"] > 14.5:  # T4 limit
                raise gr.Error(
                    f"Memory nearly full ({stats['allocated_gb']:.1f}GB). "
                    "Please wait or restart."
                )

            # Run inference
            result = pipeline(prompt, **kwargs)

            return result.images[0]

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error(
                "⚠️ GPU Out of Memory\n\n"
                "Try:\n"
                "• Reduce image size\n"
                "• Select a smaller checkpoint\n"
                "• Reduce num_inference_steps\n"
                "• Wait for other users' jobs to complete"
            )

        except FileNotFoundError as e:
            raise gr.Error(f"Checkpoint not found: {checkpoint_name}")

        except Exception as e:
            # Log error for debugging
            print(f"Inference error: {type(e).__name__}: {e}")
            raise gr.Error(
                f"Generation failed: {str(e)}\n\n"
                "Please try again or report this issue."
            )

    return inference_handler
```

### Automatic Cache Cleanup

From [Gradio Resource Cleanup](https://www.gradio.app/guides/resource-cleanup) (accessed 2025-10-31):

```python
# Configure automatic cache cleanup
with gr.Blocks(delete_cache=(3600, 7200)) as app:
    # delete_cache=(frequency, age)
    # Every 3600 seconds (1 hour), delete files older than 7200 seconds (2 hours)

    # Your Gradio interface here
    pass

app.launch()
```

### State Cleanup with delete_callback

From [Gradio Resource Cleanup](https://www.gradio.app/guides/resource-cleanup) (accessed 2025-10-31):

```python
def cleanup_model_state(model_state):
    """Called when gr.State is deleted."""
    if model_state is not None and "pipeline" in model_state:
        clear_pipeline_memory(model_state["pipeline"])
        print("Model state cleaned up")

with gr.Blocks() as app:
    # State with automatic cleanup
    model_state = gr.State(
        value=None,
        delete_callback=cleanup_model_state,
        time_to_live=3600  # Auto-delete after 1 hour
    )

    # Interface components
    pass
```

## Complete Example: ARR-COC Checkpoint Comparison

```python
import gradio as gr
import torch
from checkpoint_manager import CheckpointManagerLRU

# Initialize manager INSIDE Gradio context
manager = None

def init_checkpoint_manager():
    """Initialize manager with ARR-COC checkpoints."""
    global manager

    if manager is None:
        # Detect GPU and configure accordingly
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_gb = gpu_props.total_memory / 1024**3

            if total_gb <= 16:  # T4
                max_loaded = 1
            elif total_gb <= 24:  # RTX 3090/4090
                max_loaded = 2
            else:  # A100
                max_loaded = 3
        else:
            max_loaded = 1

        manager = CheckpointManagerLRU(max_loaded=max_loaded)

        # Register ARR-COC evaluation checkpoints
        manager.register_checkpoint("baseline", "runwayml/stable-diffusion-v1-5")
        manager.register_checkpoint("arr_coc_epoch_10", "username/arr-coc-checkpoint-10")
        manager.register_checkpoint("arr_coc_epoch_20", "username/arr-coc-checkpoint-20")

    return "Manager initialized"

def compare_checkpoints(prompt, ckpt_a, ckpt_b):
    """Generate images from two checkpoints for comparison."""

    try:
        # Load first checkpoint
        pipe_a = manager.load_checkpoint(ckpt_a)
        img_a = pipe_a(prompt, num_inference_steps=20).images[0]

        # Load second checkpoint (may evict first if over limit)
        pipe_b = manager.load_checkpoint(ckpt_b)
        img_b = pipe_b(prompt, num_inference_steps=20).images[0]

        # Get memory stats
        stats = manager.get_memory_stats()
        status = f"Loaded: {stats['loaded_names']}\nVRAM: {stats['allocated_gb']:.2f}GB"

        return img_a, img_b, status

    except Exception as e:
        raise gr.Error(f"Comparison failed: {str(e)}")

with gr.Blocks(delete_cache=(3600, 7200)) as demo:
    gr.Markdown("# ARR-COC Checkpoint Comparison")

    with gr.Row():
        checkpoint_a = gr.Dropdown(
            choices=["baseline", "arr_coc_epoch_10", "arr_coc_epoch_20"],
            label="Checkpoint A",
            value="baseline"
        )
        checkpoint_b = gr.Dropdown(
            choices=["baseline", "arr_coc_epoch_10", "arr_coc_epoch_20"],
            label="Checkpoint B",
            value="arr_coc_epoch_10"
        )

    prompt = gr.Textbox(label="Prompt", value="A photorealistic astronaut riding a horse")
    generate_btn = gr.Button("Generate Comparison")

    with gr.Row():
        output_a = gr.Image(label="Checkpoint A")
        output_b = gr.Image(label="Checkpoint B")

    status_text = gr.Textbox(label="System Status")

    # Initialize on load
    demo.load(fn=init_checkpoint_manager, outputs=status_text)

    # Generate comparison
    generate_btn.click(
        fn=compare_checkpoints,
        inputs=[prompt, checkpoint_a, checkpoint_b],
        outputs=[output_a, output_b, status_text]
    )

demo.launch()
```

## Troubleshooting Common Issues

### Issue 1: Memory Leak Despite Cleanup

**Symptom**: VRAM increases after each checkpoint switch, never returns to baseline.

**Solution**: Initialize pipeline INSIDE Gradio context, not in global scope.

```python
# WRONG
manager = CheckpointManagerLRU()  # Global initialization
with gr.Blocks() as app:
    pass

# CORRECT
with gr.Blocks() as app:
    app.load(fn=init_checkpoint_manager)  # Initialize inside Gradio
```

### Issue 2: OOM on Second Inference

**Symptom**: First inference works, second fails with OOM.

**Solution**: Remove all hooks before cleanup, apply `torch.cuda.synchronize()`.

From [Diffusers GitHub Discussion #10936](https://github.com/huggingface/diffusers/discussions/10936) (accessed 2025-10-31):

```python
# Add synchronization between unload and load
clear_previous_model_memory()
torch.cuda.synchronize()  # Wait for all operations to complete
load_new_model()
```

### Issue 3: Slow Sequential CPU Offload

**Symptom**: Inference becomes extremely slow after enabling CPU offload.

**Solution**: Use model-level offload instead of sequential offload.

```python
# Slow: Sequential CPU offload
pipeline.enable_sequential_cpu_offload()

# Faster: Model-level offload
pipeline.enable_model_cpu_offload()
```

## Sources

**Source Documents:**
- N/A (This knowledge was acquired through web research)

**Web Research:**
- [Gradio Resource Cleanup Guide](https://www.gradio.app/guides/resource-cleanup) - Official Gradio docs on memory management (accessed 2025-10-31)
- [Diffusers GitHub Discussion #10936](https://github.com/huggingface/diffusers/discussions/10936) - "Diffusers, Gradio, and the Elusive Memory Leak" with LRU implementation examples (accessed 2025-10-31)
- [Gradio Performance Guide](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance) - Performance optimization including concurrency and memory limits (accessed 2025-10-31)
- [HuggingFace Diffusers Memory Optimization](https://huggingface.co/docs/diffusers/en/optimization/memory) - Comprehensive memory reduction techniques including VAE slicing, tiling, and offloading strategies (accessed 2025-10-31)

**Key Contributions:**
- elismasilva's debugging journey and solution for Gradio memory leaks (GitHub Discussion #10936)
- CyberVy's referrer tracking cleanup implementation (GitHub Discussion #10936)
- Official Gradio and HuggingFace documentation on resource management
