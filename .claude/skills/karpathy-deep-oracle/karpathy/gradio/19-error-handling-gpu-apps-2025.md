# Error Handling for GPU-Intensive Gradio Apps (2025)

**Focus**: Production-ready error handling patterns for GPU-intensive Gradio applications, with emphasis on OOM errors, graceful degradation, and user-friendly messaging.

---

## Section 1: gr.Error Basics

### Overview

Gradio provides three built-in alert classes for user communication:

- **`gr.Error()`** - Exception that halts execution and displays error modal
- **`gr.Warning()`** - Function that displays warning while continuing execution
- **`gr.Info()`** - Function that displays informational message while continuing

From [Gradio Error Documentation](https://www.gradio.app/docs/gradio/error) (accessed 2025-10-31):

### Key Distinction

```python
# gr.Error is an EXCEPTION - must be raised
def process(input):
    if invalid_condition:
        raise gr.Error("Cannot proceed!")  # Halts execution

# gr.Warning and gr.Info are FUNCTIONS - called directly
def process(input):
    gr.Info("Starting process...")  # Continues execution
    if edge_case:
        gr.Warning("Unusual input detected")  # Continues execution
    return result
```

### Basic Usage Pattern

```python
import gradio as gr

def divide(numerator, denominator):
    if denominator == 0:
        raise gr.Error("Cannot divide by zero!")
    return numerator / denominator

gr.Interface(divide, ["number", "number"], "number").launch()
```

### Customization Options

From [Gradio Error Documentation](https://www.gradio.app/docs/gradio/error):

```python
raise gr.Error(
    message="An error occurred!",  # HTML supported
    duration=5,                     # Seconds to display (None = until closed)
    visible=True,                   # Show in UI
    title="Error",                  # Modal title
    print_exception=True            # Print traceback to console
)
```

**Duration control**:
- `duration=None` or `duration=0` - Display until user closes
- `duration=5` - Auto-dismiss after 5 seconds
- Use longer durations for critical errors, shorter for informational

### When to Use gr.Error vs Standard Exceptions

From [GitHub Issue #6335](https://github.com/gradio-app/gradio/issues/6335) (accessed 2025-10-31):

**Problem**: Standard Python exceptions (`ValueError`, `RuntimeError`) only appear in terminal logs, not visible to end users in HuggingFace Spaces or deployed apps.

**Best Practice**:
```python
# ❌ BAD: User sees nothing, only in terminal
def preprocess_audio(audio):
    if duration < min_length:
        raise ValueError(f"Audio too short: {duration}s")

# ✅ GOOD: User sees friendly error in UI
def preprocess_audio(audio):
    if duration < min_length:
        raise gr.Error(f"Audio must be at least {min_length} seconds long. Uploaded: {duration:.1f}s")
```

**User-facing errors**: Always use `gr.Error()` for validation failures, resource constraints, or user-correctable issues.

---

## Section 2: GPU Error Handling (OOM)

### The OOM Challenge

From [PyTorch Forums Discussion](https://discuss.pytorch.org/t/cuda-out-of-memory-run-time-error-handling-fall-back-to-cpu-possibility/153749) (accessed 2025-10-31):

**Critical issue**: `torch.cuda.OutOfMemoryError` is difficult to recover from gracefully because:
- GPU memory remains fragmented even after error
- `torch.cuda.empty_cache()` helps but doesn't guarantee recovery
- Multiple OOM events can crash the application

### Detection Pattern

```python
import torch
import gradio as gr

def gpu_safe_inference(image, model):
    """Wrapper that catches OOM and provides fallback."""
    try:
        # Attempt GPU inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        result = model(image.to(device))
        return result

    except torch.cuda.OutOfMemoryError:
        # Clear cache and inform user
        torch.cuda.empty_cache()
        raise gr.Error(
            "GPU out of memory! Try:\n"
            "• Reducing image size\n"
            "• Using a smaller batch\n"
            "• Waiting a moment and trying again",
            duration=10
        )
    except Exception as e:
        # Catch other errors
        raise gr.Error(f"Inference failed: {str(e)}")
```

### Graceful Degradation: CPU Fallback

**Pattern 1: Automatic CPU Fallback**

```python
def inference_with_fallback(image, model, use_gpu=True):
    """Automatically fall back to CPU on OOM."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    try:
        model = model.to(device)
        result = model(image.to(device))
        return result, f"✓ Processed on {device.type.upper()}"

    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            # Clear cache and retry on CPU
            torch.cuda.empty_cache()
            gr.Warning("GPU memory full - falling back to CPU (slower)")
            return inference_with_fallback(image, model, use_gpu=False)
        else:
            raise gr.Error("CPU inference also failed - image may be too large")
```

**Pattern 2: User-Controlled Fallback**

```python
with gr.Blocks() as demo:
    image = gr.Image()
    use_gpu = gr.Checkbox(label="Use GPU (faster but may OOM)", value=True)
    output = gr.Textbox()

    def process(img, gpu_enabled):
        try:
            device = torch.device("cuda" if gpu_enabled else "cpu")
            return model_inference(img, device)
        except torch.cuda.OutOfMemoryError:
            raise gr.Error(
                "GPU out of memory!\n\n"
                "Try unchecking 'Use GPU' or reducing image size.",
                duration=15
            )

    gr.Button("Process").click(process, [image, use_gpu], output)
```

### Proactive Memory Management

From [Stack Overflow: Avoiding CUDA OOM](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) (accessed 2025-10-31):

```python
def clear_gpu_memory():
    """Aggressive GPU memory clearing."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def safe_model_swap(old_model, new_model_path, device):
    """Safely swap models without OOM."""
    # Move old model to CPU
    if old_model is not None:
        old_model.cpu()
        del old_model

    # Clear memory
    clear_gpu_memory()

    # Load new model
    try:
        new_model = torch.load(new_model_path, map_location='cpu')
        new_model = new_model.to(device)
        return new_model
    except torch.cuda.OutOfMemoryError:
        raise gr.Error(
            "Cannot load model - GPU memory insufficient.\n\n"
            f"Model size: {os.path.getsize(new_model_path) / 1e9:.1f} GB\n"
            "Available GPU memory may be insufficient.",
            duration=None
        )
```

### Memory Monitoring

```python
def get_gpu_memory_info():
    """Get human-readable GPU memory status."""
    if not torch.cuda.is_available():
        return "No GPU available"

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return (f"GPU Memory: {allocated:.2f}GB allocated, "
            f"{reserved:.2f}GB reserved, {total:.2f}GB total")

# Display in Gradio UI
with gr.Blocks() as demo:
    memory_display = gr.Textbox(label="GPU Status")

    def update_and_process(image):
        result = model_inference(image)
        memory_info = get_gpu_memory_info()
        return result, memory_info

    btn.click(update_and_process, inputs=image, outputs=[result, memory_display])
```

---

## Section 3: Production Error Patterns

### Pattern 1: Layered Error Handling

```python
def production_inference(image, model_path, debug_mode=False):
    """Production-ready inference with comprehensive error handling."""

    # Layer 1: Input Validation
    if image is None:
        raise gr.Error("No image provided. Please upload an image.")

    # Layer 2: Resource Availability
    if not os.path.exists(model_path):
        raise gr.Error(
            "Model not found. Please contact support.",
            print_exception=False  # Don't leak paths to users
        )

    # Layer 3: GPU Availability
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            gr.Warning("GPU not available - using CPU (slower)")
    except Exception as e:
        raise gr.Error("Failed to initialize compute device")

    # Layer 4: Model Loading
    try:
        model = torch.load(model_path, map_location=device)
    except Exception as e:
        if debug_mode:
            raise gr.Error(f"Model load failed: {str(e)}", duration=None)
        else:
            raise gr.Error("Failed to load model. Please try again.")

    # Layer 5: Inference
    try:
        with torch.no_grad():
            result = model(image.to(device))
        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error(
            "Insufficient GPU memory for this image.\n\n"
            "Suggestions:\n"
            "• Try a smaller image (current size too large)\n"
            "• Wait a moment and retry\n"
            "• Contact support if issue persists",
            duration=15
        )
    except RuntimeError as e:
        if "CUDA" in str(e):
            raise gr.Error(f"GPU error occurred. Please retry or contact support.")
        else:
            raise gr.Error(f"Processing failed: {str(e)}")
    except Exception as e:
        if debug_mode:
            import traceback
            raise gr.Error(f"Unexpected error:\n\n{traceback.format_exc()}", duration=None)
        else:
            raise gr.Error("An unexpected error occurred. Please try again.")
```

### Pattern 2: Debug Mode Toggle

From [Gradio Alerts Guide](https://www.gradio.app/guides/alerts) (accessed 2025-10-31):

```python
with gr.Blocks() as demo:
    with gr.Row():
        debug_mode = gr.Checkbox(label="Debug Mode (show detailed errors)", value=False)

    image = gr.Image()
    output = gr.Textbox()

    def process_with_debug(img, debug):
        try:
            return model_inference(img)
        except Exception as e:
            if debug:
                # Show full traceback in debug mode
                import traceback
                error_details = traceback.format_exc()
                raise gr.Error(
                    f"Debug Info:\n\n{error_details}",
                    duration=None,
                    print_exception=True
                )
            else:
                # User-friendly message in production
                raise gr.Error(
                    "Processing failed. Enable Debug Mode for details.",
                    duration=10
                )

    gr.Button("Process").click(process_with_debug, [image, debug_mode], output)
```

### Pattern 3: Error Logging

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='gradio_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def logged_inference(image, session_id=None):
    """Inference with error logging for debugging."""
    try:
        return model_inference(image)
    except torch.cuda.OutOfMemoryError as e:
        # Log for operators
        logging.error(f"OOM Error [Session: {session_id}] - Image size: {image.shape}")

        # User-friendly message
        raise gr.Error(
            "GPU memory full. Try a smaller image.",
            duration=10
        )
    except Exception as e:
        # Log unexpected errors with full context
        logging.error(
            f"Inference failed [Session: {session_id}]\n"
            f"Error: {str(e)}\n"
            f"Time: {datetime.now()}\n"
            f"Image shape: {getattr(image, 'shape', 'unknown')}"
        )

        # Generic user message
        raise gr.Error(
            f"Processing failed (Error ID: {session_id}). "
            "Please contact support with this ID.",
            duration=15
        )
```

### Pattern 4: Retry Mechanism

```python
def inference_with_retry(image, max_retries=2):
    """Automatic retry with exponential backoff for transient errors."""
    import time

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Clear memory before retry
                torch.cuda.empty_cache()
                wait_time = 2 ** attempt  # Exponential backoff
                gr.Info(f"Retry {attempt}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

            return model_inference(image)

        except torch.cuda.OutOfMemoryError:
            if attempt == max_retries:
                raise gr.Error(
                    "GPU memory insufficient after multiple retries.\n\n"
                    "Please try:\n"
                    "• Smaller image size\n"
                    "• Waiting 30 seconds\n"
                    "• Using a different GPU (if available)",
                    duration=15
                )
            else:
                continue  # Retry

        except Exception as e:
            # Don't retry non-OOM errors
            raise gr.Error(f"Processing failed: {str(e)}")
```

---

## Section 4: User-Friendly Error Messages

### Principles of Good Error Messages

From [Gradio Error Documentation](https://www.gradio.app/docs/gradio/error) and [GitHub Issue #956](https://github.com/gradio-app/gradio/issues/956) (accessed 2025-10-31):

**Good error messages are**:
1. **Actionable** - Tell users what to do
2. **Specific** - Explain what went wrong
3. **Friendly** - Avoid technical jargon
4. **Contextual** - Include relevant details

### Bad vs Good Examples

**❌ Bad: Technical jargon**
```python
raise gr.Error("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.95 GiB")
```

**✅ Good: User-actionable**
```python
raise gr.Error(
    "GPU memory full!\n\n"
    "Your image is too large to process. Try:\n"
    "• Reducing image size (currently processing very large image)\n"
    "• Processing a smaller batch\n"
    "• Waiting a moment and trying again"
)
```

**❌ Bad: No context**
```python
raise gr.Error("Invalid input")
```

**✅ Good: Specific context**
```python
raise gr.Error(
    f"Image format not supported: {image.format}\n\n"
    "Please upload: JPEG, PNG, or WebP"
)
```

**❌ Bad: No solution**
```python
raise gr.Error("Processing failed")
```

**✅ Good: Suggests action**
```python
raise gr.Error(
    "Processing failed due to corrupted image.\n\n"
    "Please try:\n"
    "• Re-uploading the image\n"
    "• Using a different image\n"
    "• Converting to PNG format"
)
```

### Message Templates

```python
# Template 1: Resource constraint
def oom_error_message(available_gb, required_gb):
    return gr.Error(
        f"Insufficient GPU memory\n\n"
        f"Available: {available_gb:.1f} GB\n"
        f"Required: {required_gb:.1f} GB\n\n"
        f"Try reducing batch size or image resolution.",
        duration=15
    )

# Template 2: Validation error
def validation_error(field, provided, expected):
    return gr.Error(
        f"Invalid {field}\n\n"
        f"Provided: {provided}\n"
        f"Expected: {expected}\n\n"
        f"Please correct and try again."
    )

# Template 3: Transient error
def transient_error_with_retry(error_type):
    return gr.Error(
        f"Temporary {error_type} error\n\n"
        f"This is usually temporary. Please:\n"
        f"• Wait 10-30 seconds\n"
        f"• Try again\n"
        f"• Contact support if it persists",
        duration=10
    )

# Template 4: Configuration error
def config_error(setting, current, recommended):
    return gr.Error(
        f"Configuration issue: {setting}\n\n"
        f"Current: {current}\n"
        f"Recommended: {recommended}\n\n"
        f"Please adjust settings and retry."
    )
```

### Progressive Error Disclosure

```python
def handle_error_levels(error, attempt=1):
    """Show increasingly detailed errors with retries."""

    if attempt == 1:
        # First error: Simple message
        raise gr.Error(
            "Processing failed. Click retry to try again.",
            duration=5
        )
    elif attempt == 2:
        # Second error: More context
        raise gr.Error(
            "Still having issues.\n\n"
            f"Error type: {type(error).__name__}\n"
            "Suggestion: Try reducing image size",
            duration=10
        )
    else:
        # Third error: Full details
        raise gr.Error(
            "Multiple failures detected.\n\n"
            f"Error: {str(error)}\n\n"
            "Please contact support with this information.",
            duration=None
        )
```

### Contextual Help

```python
def error_with_help_link(error_type):
    """Provide documentation links for common errors."""

    help_urls = {
        "oom": "https://docs.example.com/gpu-memory-issues",
        "format": "https://docs.example.com/supported-formats",
        "timeout": "https://docs.example.com/processing-times"
    }

    url = help_urls.get(error_type, "https://docs.example.com/troubleshooting")

    raise gr.Error(
        f"{error_type.upper()} Error\n\n"
        f"For detailed help, see:\n{url}",
        duration=15
    )
```

### Visual Error Indicators

```python
with gr.Blocks() as demo:
    status = gr.Markdown("Status: Ready ✓")

    def process_with_status(image):
        try:
            status.update("Status: Processing... ⏳")
            result = model_inference(image)
            status.update("Status: Complete ✓")
            return result
        except torch.cuda.OutOfMemoryError:
            status.update("Status: Error - Out of Memory ❌")
            raise gr.Error("GPU memory full")
        except Exception as e:
            status.update(f"Status: Error - {type(e).__name__} ❌")
            raise gr.Error(f"Processing failed: {str(e)}")
```

---

## Best Practices Summary

### DO:
✅ Always use `gr.Error()` for user-facing errors
✅ Catch `torch.cuda.OutOfMemoryError` explicitly
✅ Provide actionable suggestions in error messages
✅ Clear GPU memory (`torch.cuda.empty_cache()`) after OOM
✅ Implement graceful degradation (CPU fallback)
✅ Log errors server-side for debugging
✅ Test error handling under resource constraints
✅ Use appropriate error durations (critical = longer)

### DON'T:
❌ Let standard Python exceptions reach users
❌ Show technical stack traces to end users
❌ Use vague messages like "Error occurred"
❌ Assume GPU is always available
❌ Retry indefinitely without backoff
❌ Expose internal paths or configuration
❌ Ignore memory fragmentation after OOM
❌ Provide no guidance on how to resolve errors

---

## Production Checklist

**Error Handling**:
- [ ] All user-correctable errors use `gr.Error()`
- [ ] OOM errors caught and handled gracefully
- [ ] Fallback to CPU implemented for OOM
- [ ] All error messages are actionable
- [ ] Debug mode available for operators

**Resource Management**:
- [ ] GPU memory cleared after errors
- [ ] Model loading wrapped in try/except
- [ ] Memory status displayed to users
- [ ] Resource constraints documented

**User Experience**:
- [ ] Error messages tested with real users
- [ ] No technical jargon in production errors
- [ ] Appropriate error durations set
- [ ] Visual status indicators present
- [ ] Help documentation linked

**Monitoring**:
- [ ] Errors logged server-side
- [ ] Error rates tracked
- [ ] OOM frequency monitored
- [ ] User feedback collected

---

## Sources

**Gradio Official Documentation**:
- [Error Class Documentation](https://www.gradio.app/docs/gradio/error) - gr.Error usage and parameters (accessed 2025-10-31)
- [Alerts Guide](https://www.gradio.app/guides/alerts) - Error, Warning, Info patterns (accessed 2025-10-31)

**GitHub Issues & Discussions**:
- [Issue #6335: Use gr.Error for component errors](https://github.com/gradio-app/gradio/issues/6335) - Visibility problem with standard exceptions (accessed 2025-10-31)

**PyTorch Forums**:
- [CUDA OOM Runtime Error Handling](https://discuss.pytorch.org/t/cuda-out-of-memory-run-time-error-handling-fall-back-to-cpu-possibility/153749) - Graceful degradation discussion (accessed 2025-10-31)

**Stack Overflow**:
- [How to Avoid CUDA Out of Memory in PyTorch](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) - Memory management techniques (accessed 2025-10-31)

**Additional References**:
- PyTorch memory management best practices for production deployments
- Gradio error handling patterns from HuggingFace Spaces examples
- Production error messaging guidelines from ML engineering community
