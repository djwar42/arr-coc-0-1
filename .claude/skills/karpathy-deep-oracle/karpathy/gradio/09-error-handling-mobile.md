# Gradio Error Handling & Mobile Design (2025)

**Category**: Practical Implementation
**Related**: [10-gradio-testing-best-practices.md](10-gradio-testing-best-practices.md), [08-gradio-interface-and-blocks.md](../concepts/08-gradio-interface-and-blocks.md)

## Overview

Production Gradio applications require robust error handling and mobile-responsive design. This guide covers gr.Error for custom error messages, debugging with show_error, production error patterns, and mobile responsiveness features in Gradio 5.x (2025).

---

## gr.Error Class & Custom Messages

From [gradio.app/docs/gradio/error](https://www.gradio.app/docs/gradio/error):

### Basic Usage

**Raising Custom Errors:**

```python
import gradio as gr

def divide(numerator, denominator):
    if denominator == 0:
        raise gr.Error("Cannot divide by zero!")
    return numerator / denominator

demo = gr.Interface(
    fn=divide,
    inputs=["number", "number"],
    outputs="number"
)

demo.launch()
```

**Error Displays:**
- Error appears in modal overlay
- User-friendly message shown in UI
- Replaces generic stack trace
- Automatically dismissible

### Error Duration Control

From [gradio.app/docs/gradio/error](https://www.gradio.app/docs/gradio/error):

```python
# Error displayed for 5 seconds
raise gr.Error("An error occurred 💥!", duration=5)

# Error displayed until user closes it
raise gr.Error("Critical error!", duration=None)

# Default: 10 seconds
raise gr.Error("Something went wrong")
```

**Duration Behavior:**

```python
def process_with_timing(action):
    if action == "quick":
        # Auto-dismiss after 3 seconds
        raise gr.Error("Quick notification", duration=3)

    elif action == "important":
        # Stays until user closes
        raise gr.Error("⚠️ Important: Please review", duration=None)

    elif action == "warning":
        # Medium duration
        raise gr.Error("⚡ Warning message", duration=7)

    return "Success"
```

### HTML Error Messages

```python
# Rich HTML formatting in error messages
raise gr.Error(
    """
    <strong>Validation Failed</strong><br>
    <ul>
        <li>Email format invalid</li>
        <li>Password too short</li>
        <li>Username taken</li>
    </ul>
    """,
    duration=10
)
```

### Silent Errors

```python
# Error logged but not shown in UI
raise gr.Error(
    "Internal validation failed",
    visible=False,
    print_exception=True  # Still logged to console
)
```

### Custom Error Titles

From [gradio.app/docs/gradio/error](https://www.gradio.app/docs/gradio/error):

```python
# Custom title in error modal
raise gr.Error(
    "Invalid input format",
    title="Validation Error",
    duration=5
)

# Default title is "Error"
raise gr.Error("Something went wrong")  # Title: "Error"
```

---

## Debugging with show_error

From [gradio.app/docs/gradio/interface](https://www.gradio.app/docs/gradio/interface):

### Development Mode

**Enable Detailed Errors:**

```python
import gradio as gr

def buggy_function(x):
    # This will raise TypeError
    return x + "string"

demo = gr.Interface(
    fn=buggy_function,
    inputs="number",
    outputs="text"
)

# Show full stack traces in browser
demo.launch(show_error=True)
```

**What show_error Does:**

- Displays Python stack traces in UI
- Shows line numbers and file paths
- Includes variable values at error point
- Useful for debugging during development

**Production vs Development:**

```python
import os

# Conditional error display
is_production = os.getenv("ENVIRONMENT") == "production"

demo.launch(
    show_error=not is_production  # Hide errors in production
)
```

### Browser Console Logging

```python
# Errors also logged to browser console
# Open DevTools (F12) to see:
# - Full Python traceback
# - Request/response data
# - WebSocket messages
# - Network errors

demo.launch(show_error=True)
```

---

## Production Error Handling

From [datacamp.com/tutorial/llama-gradio-app](https://www.datacamp.com/tutorial/llama-gradio-app):

### Input Validation

**Comprehensive Validation:**

```python
import gradio as gr
import re

def validate_and_process(text, image, api_key):
    """Validate all inputs before processing"""

    # Text validation
    if not text or not text.strip():
        raise gr.Error(
            "Input text cannot be empty",
            duration=5
        )

    if len(text) > 1000:
        raise gr.Error(
            f"Text too long: {len(text)} characters (max 1000)",
            duration=7
        )

    # Image validation
    if image is not None:
        allowed_formats = ['.jpg', '.png', '.jpeg']
        file_ext = image.name.lower().split('.')[-1]

        if f'.{file_ext}' not in allowed_formats:
            raise gr.Error(
                f"Unsupported image format: {file_ext}. "
                f"Allowed: {', '.join(allowed_formats)}",
                duration=8
            )

        # Check file size (5MB limit)
        import os
        file_size = os.path.getsize(image.name)
        max_size = 5 * 1024 * 1024

        if file_size > max_size:
            raise gr.Error(
                f"Image too large: {file_size / 1024 / 1024:.1f}MB (max 5MB)",
                duration=8
            )

    # API key validation
    if not api_key:
        raise gr.Error(
            "API key required. Please configure in settings.",
            duration=None  # Must be addressed
        )

    api_key_pattern = r'^[A-Za-z0-9_-]{20,}$'
    if not re.match(api_key_pattern, api_key):
        raise gr.Error(
            "Invalid API key format",
            duration=10
        )

    # Process if all validations pass
    return process_llm_request(text, image, api_key)


def process_llm_request(text, image, api_key):
    """Actual processing logic"""
    try:
        # Call model API
        result = call_model(text, image, api_key)
        return result

    except Exception as e:
        # Catch and re-raise as user-friendly error
        raise gr.Error(
            f"Model API error: {str(e)}",
            duration=10
        )
```

### Try-Except Patterns

**Graceful Degradation:**

```python
import gradio as gr
import logging

logger = logging.getLogger(__name__)

def robust_inference(input_data):
    """Handle errors gracefully with fallbacks"""

    try:
        # Primary model inference
        result = primary_model.predict(input_data)
        return result

    except ConnectionError as e:
        logger.error(f"Primary model connection failed: {e}")

        # Fallback to secondary model
        try:
            result = secondary_model.predict(input_data)
            gr.Warning(
                "Using fallback model due to connection issue",
                duration=5
            )
            return result

        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {fallback_error}")
            raise gr.Error(
                "Both primary and fallback models unavailable. "
                "Please try again later.",
                duration=None
            )

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise gr.Error(
            f"Invalid input format: {str(e)}",
            duration=7
        )

    except TimeoutError:
        logger.error("Model inference timeout")
        raise gr.Error(
            "Request timeout. The model is taking too long to respond. "
            "Please try with shorter input.",
            duration=10
        )

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception("Unexpected error in inference")

        # Don't expose internal details in production
        raise gr.Error(
            "An unexpected error occurred. Our team has been notified.",
            duration=10
        )
```

### Error Recovery

**Retry with Exponential Backoff:**

```python
import time
import gradio as gr

def api_call_with_retry(input_text, max_retries=3):
    """Retry failed API calls with backoff"""

    for attempt in range(max_retries):
        try:
            # Attempt API call
            response = call_external_api(input_text)
            return response

        except ConnectionError as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)

                # Show progress to user
                gr.Info(
                    f"Connection failed, retrying... (attempt {attempt + 2})",
                    duration=3
                )
            else:
                # Final attempt failed
                raise gr.Error(
                    f"Failed after {max_retries} attempts. "
                    "Service may be unavailable.",
                    duration=15
                )

        except RateLimitError:
            raise gr.Error(
                "⏰ Rate limit exceeded. Please wait a moment before trying again.",
                duration=10
            )
```

### User-Friendly Error Messages

From [datacamp.com/tutorial/llama-gradio-app](https://www.datacamp.com/tutorial/llama-gradio-app):

```python
def format_error_for_user(exception):
    """Convert technical errors to user-friendly messages"""

    error_messages = {
        "KeyError": "Required configuration is missing. Please check settings.",
        "FileNotFoundError": "The requested file could not be found.",
        "PermissionError": "Access denied. Check file permissions.",
        "MemoryError": "Out of memory. Try processing smaller inputs.",
        "TimeoutError": "Request timed out. The operation took too long.",
    }

    error_type = type(exception).__name__

    if error_type in error_messages:
        user_message = error_messages[error_type]
    else:
        user_message = "An unexpected error occurred."

    # Add action items
    if error_type == "MemoryError":
        user_message += " Try reducing image size or text length."
    elif error_type == "TimeoutError":
        user_message += " Try again with simpler input."

    return user_message


def safe_process(input_data):
    """Wrapper that catches and formats errors"""
    try:
        return complex_processing(input_data)

    except Exception as e:
        logger.exception("Processing error")
        user_message = format_error_for_user(e)
        raise gr.Error(user_message, duration=10)
```

### Logging Best Practices

```python
import logging
import gradio as gr
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def logged_handler(input_text, request: gr.Request):
    """Handler with comprehensive logging"""

    user_id = getattr(request, 'session_hash', 'unknown')
    request_id = datetime.now().strftime('%Y%m%d-%H%M%S')

    logger.info(
        f"Request {request_id} from user {user_id}: "
        f"input_length={len(input_text)}"
    )

    try:
        result = process_input(input_text)

        logger.info(
            f"Request {request_id} completed successfully: "
            f"output_length={len(result)}"
        )

        return result

    except gr.Error as e:
        # Expected errors (user-facing)
        logger.warning(
            f"Request {request_id} failed with user error: {str(e)}"
        )
        raise  # Re-raise to show user

    except Exception as e:
        # Unexpected errors (internal)
        logger.exception(
            f"Request {request_id} failed with internal error"
        )

        # Don't expose details to user
        raise gr.Error(
            "An internal error occurred. Please try again.",
            duration=10
        )
```

---

## Status Modals: Info and Warning

From [gradio.app/docs/gradio/error](https://www.gradio.app/docs/gradio/error):

### gr.Info for Success Messages

```python
import gradio as gr

def save_settings(api_key, model_choice):
    """Save with success notification"""

    # Validate
    if not api_key:
        raise gr.Error("API key required", duration=5)

    # Save settings
    save_to_config(api_key, model_choice)

    # Success message
    gr.Info("✅ Settings saved successfully!", duration=3)

    return "Settings updated"
```

### gr.Warning for Non-Critical Issues

```python
def process_with_warnings(text, use_cache):
    """Process with non-blocking warnings"""

    if use_cache and cache_is_stale():
        gr.Warning(
            "⚠️ Cache is outdated. Consider refreshing.",
            duration=5
        )

    if len(text) > 500:
        gr.Warning(
            "Input is long. Processing may take extra time.",
            duration=7
        )

    result = process(text, use_cache)

    gr.Info("Processing complete!", duration=3)

    return result
```

### Combined Status Patterns

From [gradio.app/docs/gradio/error](https://www.gradio.app/docs/gradio/error):

```python
def comprehensive_handler(input_text):
    """Use all status types appropriately"""

    # Info: Operation started
    gr.Info("Processing started...", duration=2)

    try:
        # Warning: Non-critical issues
        if len(input_text) < 10:
            gr.Warning(
                "Input is very short. Results may be limited.",
                duration=5
            )

        result = process(input_text)

        # Info: Success
        gr.Info("✅ Processing completed successfully!", duration=3)

        return result

    except ValueError as e:
        # Error: User error
        raise gr.Error(f"Invalid input: {str(e)}", duration=7)

    except Exception as e:
        # Error: System error
        logger.exception("Processing failed")
        raise gr.Error(
            "Processing failed. Please try again.",
            duration=10
        )
```

---

## Mobile Responsive Design (2025)

From [huggingface.co/blog/why-gradio-stands-out](https://huggingface.co/blog/why-gradio-stands-out):

### Automatic Mobile Responsiveness

**Built-in Features:**

- Components automatically stack on narrow screens
- Touch-friendly button sizes
- Optimized font scaling
- Responsive images and media
- Mobile-friendly file uploads

**No Configuration Required:**

```python
import gradio as gr

# This layout is automatically mobile-responsive
with gr.Blocks() as demo:
    with gr.Row():
        # Desktop: side-by-side
        # Mobile: stacked vertically
        input_text = gr.Textbox(label="Input")
        output_text = gr.Textbox(label="Output")

    btn = gr.Button("Process")  # Touch-friendly on mobile

demo.launch()
```

### Progressive Web App (PWA) Support

From [huggingface.co/blog/why-gradio-stands-out](https://huggingface.co/blog/why-gradio-stands-out):

**Gradio apps are installable PWAs:**

```python
# No extra configuration needed
demo.launch()

# Users can:
# 1. "Add to Home Screen" on mobile
# 2. Use as standalone app
# 3. Access offline (with service worker)
# 4. Receive push notifications (if configured)
```

**PWA Manifest (auto-generated):**

- App name and description
- Icons for different sizes
- Display mode (standalone/fullscreen)
- Theme colors
- Orientation preferences

### Mobile-Optimized Components

**Best Practices for Mobile:**

```python
with gr.Blocks() as demo:
    # Use vertical layouts (natural for mobile)
    with gr.Column():
        gr.Markdown("# Mobile-Friendly App")

        # Large touch targets
        input_text = gr.Textbox(
            label="Input",
            placeholder="Type here...",
            lines=3  # Adequate size on mobile
        )

        # Radio buttons better than dropdowns on mobile
        choice = gr.Radio(
            choices=["Option A", "Option B", "Option C"],
            label="Select Option"
        )

        # Full-width buttons
        btn = gr.Button("Submit", variant="primary")

        # Results area
        output = gr.Textbox(label="Output", lines=5)

    btn.click(process, inputs=[input_text, choice], outputs=output)
```

### Responsive Image Handling

```python
with gr.Blocks() as demo:
    # Images scale automatically
    gr.Image(
        label="Input Image",
        type="filepath",
        # Automatically responsive:
        # - Scales to container width
        # - Maintains aspect ratio
        # - Touch-friendly zoom on mobile
    )

    # For custom sizing, use max dimensions
    gr.Image(
        label="Thumbnail",
        type="filepath",
        height=200,  # Max height
        # Width adjusts for mobile
    )
```

### Mobile-Friendly File Upload

From [huggingface.co/blog/why-gradio-stands-out](https://huggingface.co/blog/why-gradio-stands-out):

```python
with gr.Blocks() as demo:
    # Mobile file upload includes:
    # - Camera capture option
    # - Photo library access
    # - Drag & drop support
    # - Multiple file selection

    file_upload = gr.File(
        label="Upload Documents",
        file_count="multiple",  # Mobile-friendly picker
        file_types=[".pdf", ".txt", ".docx"]
    )

    image_upload = gr.Image(
        label="Upload Photo",
        type="filepath",
        sources=["upload", "webcam"]  # Camera on mobile
    )
```

### Viewport-Aware Layouts

```python
with gr.Blocks(fill_width=True) as demo:
    # Full width on mobile
    gr.Markdown("# Responsive Dashboard")

    # Automatically stacks on mobile
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # Sidebar: full width on mobile
            gr.Markdown("## Controls")
            slider = gr.Slider(0, 100, label="Value")

        with gr.Column(scale=2, min_width=400):
            # Main content: full width on mobile
            gr.Markdown("## Output")
            output = gr.Textbox(lines=10)
```

### Accessibility Features

From [huggingface.co/blog/why-gradio-stands-out](https://huggingface.co/blog/why-gradio-stands-out):

**Built-in Accessibility:**

- Screen reader support
- Keyboard navigation
- ARIA labels on components
- Focus management
- High contrast mode compatible

```python
# Accessible by default
with gr.Blocks() as demo:
    # Proper labels for screen readers
    input_box = gr.Textbox(
        label="Enter your question",  # Read by screen readers
        placeholder="Type here...",
        info="Ask anything about the document"  # Additional context
    )

    # Button with clear action
    submit_btn = gr.Button(
        "Submit Question",  # Descriptive text
        variant="primary"   # Visual emphasis
    )
```

---

## Testing Error Handling

### Unit Tests for Errors

```python
import pytest
import gradio as gr

def test_divide_by_zero():
    """Test that division by zero raises gr.Error"""

    def divide(a, b):
        if b == 0:
            raise gr.Error("Cannot divide by zero!")
        return a / b

    # Test error is raised
    with pytest.raises(gr.Error) as exc_info:
        divide(10, 0)

    assert "Cannot divide by zero" in str(exc_info.value)

def test_input_validation():
    """Test input validation errors"""

    def validate_email(email):
        if "@" not in email:
            raise gr.Error("Invalid email format", duration=5)
        return "Valid"

    # Test invalid input
    with pytest.raises(gr.Error) as exc_info:
        validate_email("invalid-email")

    assert "Invalid email" in str(exc_info.value)

    # Test valid input
    result = validate_email("user@example.com")
    assert result == "Valid"
```

---

## Summary

Production-ready error handling and mobile design:

**Error Handling:**
- gr.Error for user-facing messages
- show_error for development debugging
- Input validation before processing
- Try-except with fallbacks
- Logging for monitoring

**Status Communication:**
- gr.Error: Critical failures
- gr.Warning: Non-blocking issues
- gr.Info: Success messages
- Duration control for UX

**Mobile Responsiveness:**
- Automatic responsive layouts
- PWA support for installation
- Touch-friendly components
- Camera/file access
- Accessibility built-in

**Best Practices:**
- Validate all inputs
- Provide actionable error messages
- Log errors for monitoring
- Test on mobile devices
- Use appropriate status levels
- Never expose internal details

These patterns ensure robust, user-friendly applications that work seamlessly across desktop and mobile devices while providing clear feedback on errors and status.
