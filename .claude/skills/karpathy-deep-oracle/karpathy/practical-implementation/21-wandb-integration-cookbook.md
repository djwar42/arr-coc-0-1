# W&B Integration Cookbook: FastAPI, Gradio, Streamlit

Complete production patterns for integrating W&B tracking into web frameworks and ML apps.

---

## Section 1: Web Framework Integration (FastAPI) (~170 lines)

### FastAPI Middleware Pattern

**Core Pattern: Request/Response Logging**

From [Medium - FastAPI & W&B Integration](https://medium.com/@jordao.cassiano.009/deploying-your-ml-model-into-production-using-wandb-fastapi-heroku-558aac33552a) (accessed 2025-01-31):

```python
import wandb
from fastapi import FastAPI, Request
from typing import Dict, Any
import time
import asyncio

# Initialize W&B once at app startup
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize W&B on app startup"""
    wandb.init(
        project="fastapi-production",
        name="api-server",
        config={
            "framework": "fastapi",
            "environment": "production"
        }
    )

# Async logging to avoid blocking requests
async def log_to_wandb(data: Dict[str, Any]):
    """Background task for W&B logging"""
    await asyncio.to_thread(wandb.log, data)

@app.middleware("http")
async def wandb_middleware(request: Request, call_next):
    """Middleware for automatic request/response tracking"""
    start_time = time.time()

    # Execute request
    response = await call_next(request)

    # Calculate metrics
    duration = time.time() - start_time

    # Log asynchronously (non-blocking)
    asyncio.create_task(log_to_wandb({
        "endpoint": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "duration_ms": duration * 1000,
        "timestamp": time.time()
    }))

    return response
```

**Endpoint-Specific Tracking**

```python
import weave
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Initialize Weave for LLM tracking
weave.init("fastapi-llm-app")

class PredictionRequest(BaseModel):
    text: str
    model: str = "gpt-4"

@weave.op  # Automatic tracing with Weave
async def generate_completion(text: str, model: str) -> str:
    """Traced LLM call"""
    # Your LLM logic here
    return f"Generated: {text}"

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Endpoint with automatic W&B Weave tracing
    """
    # Weave automatically traces this call
    result = await generate_completion(
        request.text,
        request.model
    )

    return {"prediction": result}
```

From [W&B Tracing Quickstart](https://docs.wandb.ai/weave/guides/tracking/tracing) (accessed 2025-01-31):
- `@weave.op` decorator automatically tracks inputs/outputs
- Async functions fully supported
- Parent-child call relationships preserved

**Authentication and User Tracking**

```python
from fastapi import Depends, HTTPException, Header
import wandb

def get_current_user(authorization: str = Header(...)):
    """Extract user from auth header"""
    # Your auth logic
    return {"user_id": "user123", "tier": "premium"}

@app.post("/api/predict")
async def predict_with_user(
    request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """Log predictions with user context"""
    with weave.attributes({
        "user_id": user["user_id"],
        "user_tier": user["tier"],
        "endpoint": "/api/predict"
    }):
        result = await generate_completion(request.text)

    # Additional W&B logging
    wandb.log({
        "predictions": 1,
        f"user_{user['user_id']}_requests": 1
    })

    return {"prediction": result}
```

From [W&B Tracing Docs](https://docs.wandb.ai/weave/guides/tracking/tracing) (accessed 2025-01-31):
- `weave.attributes` adds metadata to traces
- Attributes frozen at call start
- User/session context preserved in trace tree

**Error Monitoring**

```python
from fastapi import HTTPException
import traceback

@app.exception_handler(Exception)
async def wandb_exception_handler(request: Request, exc: Exception):
    """Log exceptions to W&B"""
    error_data = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "endpoint": request.url.path,
        "traceback": traceback.format_exc()
    }

    # Log to W&B
    wandb.log({"errors": 1, **error_data})

    # Return error response
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
```

---

## Section 2: UI Framework Integration (Gradio & Streamlit) (~170 lines)

### Gradio Event Tracking

**Basic Gradio + W&B Integration**

From [Gradio W&B Integration Guide](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):

```python
import gradio as gr
import weave
from datetime import datetime

# Initialize Weave
weave.init("gradio-demo-app")

@weave.op
def process_input(text: str, model_choice: str) -> str:
    """Tracked function for Gradio interface"""
    # Your ML logic here
    return f"Processed with {model_choice}: {text}"

# Event tracking wrapper
def gradio_predict_with_logging(text, model):
    """Wrapper that logs user interactions"""
    # Get result and call object
    result, call = process_input.call(text, model)

    # Add custom attributes to trace
    call.summary.update({
        "user_interaction": True,
        "timestamp": datetime.now().isoformat(),
        "interface": "gradio"
    })

    return result

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Model Demo with W&B Tracking")

    with gr.Row():
        text_input = gr.Textbox(label="Input Text")
        model_choice = gr.Dropdown(
            ["gpt-4", "claude-3", "llama-3"],
            label="Model"
        )

    output = gr.Textbox(label="Output")
    submit_btn = gr.Button("Submit")

    # Track button clicks
    submit_btn.click(
        fn=gradio_predict_with_logging,
        inputs=[text_input, model_choice],
        outputs=output
    )

demo.launch()
```

**A/B Test UI Variants**

```python
import random
import weave

@weave.op
def run_variant_a(input_text: str) -> str:
    """Variant A: Simple processing"""
    return f"Variant A: {input_text}"

@weave.op
def run_variant_b(input_text: str) -> str:
    """Variant B: Enhanced processing"""
    return f"Variant B (enhanced): {input_text}"

def ab_test_predict(text):
    """Randomly assign variant and track"""
    variant = random.choice(["A", "B"])

    with weave.attributes({
        "ab_variant": variant,
        "experiment": "ui_optimization"
    }):
        if variant == "A":
            result = run_variant_a(text)
        else:
            result = run_variant_b(text)

    return result
```

**User Feedback Collection**

```python
import weave

def predict_with_feedback(text, model):
    """Prediction with feedback mechanism"""
    # Get prediction and call
    result, call = process_input.call(text, model)

    # Store call ID for feedback
    call_id = call.id

    # Return result with call ID
    return result, call_id

def submit_feedback(call_id: str, rating: int, comments: str):
    """Submit user feedback to W&B"""
    client = weave.init("gradio-demo-app")
    call = client.get_call(call_id)

    # Add feedback to call
    call.feedback.add_reaction("👍" if rating >= 4 else "👎")
    call.feedback.add_note(comments)

    return "Feedback submitted!"

# Gradio interface with feedback
with gr.Blocks() as demo:
    with gr.Row():
        text_input = gr.Textbox(label="Input")
        output = gr.Textbox(label="Output")

    call_id_state = gr.State()  # Hidden state for call ID

    submit_btn = gr.Button("Generate")
    submit_btn.click(
        fn=predict_with_feedback,
        inputs=[text_input],
        outputs=[output, call_id_state]
    )

    # Feedback section
    with gr.Row():
        rating = gr.Slider(1, 5, label="Rating")
        comments = gr.Textbox(label="Comments")
        feedback_btn = gr.Button("Submit Feedback")

    feedback_btn.click(
        fn=submit_feedback,
        inputs=[call_id_state, rating, comments],
        outputs=gr.Textbox(label="Status")
    )
```

### Streamlit Integration Patterns

**Basic Streamlit + W&B**

From [W&B Streamlit Report](https://wandb.ai/capecape/st30/reports/Simple-Streamlit-integration--VmlldzoxODgwMTUz) (accessed 2025-01-31):

```python
import streamlit as st
import weave

# Initialize Weave
weave.init("streamlit-app")

@weave.op
def process_data(data: str, option: str):
    """Tracked function"""
    return f"Processed: {data} with {option}"

# Streamlit UI
st.title("ML App with W&B Tracking")

data_input = st.text_input("Enter data")
option = st.selectbox("Choose option", ["A", "B", "C"])

if st.button("Process"):
    # Track with Weave
    with weave.attributes({
        "session_id": st.session_state.get("session_id", "unknown"),
        "interface": "streamlit"
    }):
        result = process_data(data_input, option)

    st.success(result)
```

**Session Tracking**

```python
import streamlit as st
import weave
import uuid

# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

@weave.op
def track_session_interaction(action: str, data: dict):
    """Track user session interactions"""
    return {"action": action, **data}

# Track page navigation
with weave.attributes({
    "session_id": st.session_state.session_id,
    "page": "main"
}):
    track_session_interaction("page_view", {"timestamp": time.time()})
```

**Embed W&B Visualizations**

From [W&B Streamlit Integration](https://wandb.ai/capecape/st30/reports/Using-Streamlit-with-W-B-reports--VmlldzoxODgwMTQ1) (accessed 2025-01-31):

```python
import streamlit as st
import streamlit.components.v1 as components

# Embed W&B run
st.title("Model Performance Dashboard")

wandb_run_url = "https://wandb.ai/username/project/runs/run_id"
components.iframe(wandb_run_url, height=600, scrolling=True)

# Or embed specific visualizations
chart_url = "https://wandb.ai/username/project/reports/..."
components.iframe(chart_url, height=400)
```

---

## Section 3: Production Patterns (~160 lines)

### Async Logging (Non-Blocking)

**Pattern: Background Task Logging**

From [W&B Performance Improvements](https://wandb.ai/wandb_fc/product-announcements-fc/reports/W-B-Models-performance-improvements-Fast-logging-immediate-results--VmlldzoxMjYyMjUzMA) (accessed 2025-01-31):

```python
import asyncio
import weave
from concurrent.futures import ThreadPoolExecutor

# Thread pool for async logging
executor = ThreadPoolExecutor(max_workers=4)

async def log_async(data: dict):
    """Non-blocking W&B logging"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, wandb.log, data)

@app.post("/predict")
async def predict_async(request: PredictionRequest):
    """Async endpoint with non-blocking logging"""
    # Make prediction
    result = await model.predict(request.text)

    # Log in background (don't await)
    asyncio.create_task(log_async({
        "predictions": 1,
        "latency_ms": result.latency
    }))

    # Return immediately
    return {"prediction": result.output}
```

**Weave's Built-in Async Support**

From [W&B Weave Tracing](https://docs.wandb.ai/weave/guides/tracking/tracing) (accessed 2025-01-31):
- W&B handles async logging automatically
- Local cache prevents data loss during network issues
- Asynchronous logging doesn't affect training performance

```python
import weave

# Weave handles async automatically
weave.init("production-app")

@weave.op
async def async_prediction(text: str):
    """Async function - Weave handles logging automatically"""
    # Weave logs in background, doesn't block
    result = await some_async_operation(text)
    return result
```

### Sampling Strategies (High Traffic)

**Request Sampling**

```python
import random
import weave

SAMPLE_RATE = 0.1  # Log 10% of requests

@app.post("/predict")
async def predict_with_sampling(request: PredictionRequest):
    """Sample high-traffic endpoint"""
    # Always execute prediction
    result = await model.predict(request.text)

    # Only log sample of requests
    if random.random() < SAMPLE_RATE:
        with weave.attributes({"sampled": True}):
            # This call will be traced
            log_prediction(request, result)

    return {"prediction": result}

@weave.op
def log_prediction(request, result):
    """Separate logging function"""
    return {
        "input": request.text,
        "output": result,
        "model": request.model
    }
```

**Adaptive Sampling**

```python
class AdaptiveSampler:
    """Sample more during errors, less during normal operation"""

    def __init__(self, base_rate=0.1, error_rate=1.0):
        self.base_rate = base_rate
        self.error_rate = error_rate
        self.recent_errors = []

    def should_sample(self, is_error: bool = False) -> bool:
        if is_error:
            self.recent_errors.append(time.time())
            return random.random() < self.error_rate

        # Clean old errors (last hour)
        cutoff = time.time() - 3600
        self.recent_errors = [t for t in self.recent_errors if t > cutoff]

        # Increase sampling if many recent errors
        if len(self.recent_errors) > 10:
            return random.random() < (self.base_rate * 2)

        return random.random() < self.base_rate

sampler = AdaptiveSampler()

@app.post("/predict")
async def predict_adaptive(request: PredictionRequest):
    """Adaptive sampling based on error rate"""
    try:
        result = await model.predict(request.text)

        if sampler.should_sample():
            log_prediction(request, result)

        return {"prediction": result}

    except Exception as e:
        # Always log errors
        if sampler.should_sample(is_error=True):
            log_error(request, e)
        raise
```

### Multi-Tenant Tracking

**Per-User Namespacing**

```python
import weave

def get_user_project(user_id: str) -> str:
    """Get user-specific project name"""
    return f"production/user-{user_id}"

@app.post("/predict")
async def predict_multi_tenant(
    request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """Multi-tenant tracking with user isolation"""
    # Initialize user-specific tracking
    user_client = weave.init(get_user_project(user["user_id"]))

    with weave.attributes({
        "tenant": user["tenant_id"],
        "user_tier": user["tier"]
    }):
        result = await model.predict(request.text)

    return {"prediction": result}
```

### Privacy and PII Handling

**Input Redaction**

From [W&B Weave Tracing](https://docs.wandb.ai/weave/guides/tracking/tracing) (accessed 2025-01-31):

```python
import re
import weave

def redact_pii(text: str) -> str:
    """Redact PII from inputs"""
    # Email redaction
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', text)
    # Phone redaction
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # SSN redaction
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return text

def postprocess_inputs(inputs: dict) -> dict:
    """Redact PII before logging"""
    if "text" in inputs:
        inputs["text"] = redact_pii(inputs["text"])
    if "email" in inputs:
        inputs["email"] = "[REDACTED]"
    return inputs

@weave.op(postprocess_inputs=postprocess_inputs)
def predict_with_pii_protection(text: str, email: str):
    """Automatically redact PII in traces"""
    # Original data used in function
    # But logged data is redacted
    return process_text(text)
```

### Cost-Effective Logging

**Conditional Detailed Logging**

```python
import weave

def should_log_detailed(request: PredictionRequest) -> bool:
    """Decide if we should log full details"""
    # Always log errors
    # Log 10% of successful requests
    # Always log expensive model calls
    return (
        request.model.startswith("gpt-4") or
        random.random() < 0.1
    )

@weave.op
async def predict_cost_optimized(request: PredictionRequest):
    """Cost-optimized logging"""
    result = await model.predict(request.text)

    # Always log basic metrics
    wandb.log({
        "requests": 1,
        "model": request.model
    })

    # Conditionally log full trace
    if should_log_detailed(request):
        # Full Weave trace with inputs/outputs
        return result
    else:
        # Skip detailed trace by using context manager
        from weave.trace.context.call_context import set_tracing_enabled
        with set_tracing_enabled(False):
            return result
```

### Complete ARR-COC Gradio + W&B Example

**Production-Ready Integration**

```python
import gradio as gr
import weave
from typing import List, Dict
import numpy as np

# Initialize Weave
weave.init("arr-coc-production")

@weave.op
def arr_coc_compress(
    image: np.ndarray,
    query: str,
    token_budget: int = 256
) -> Dict[str, any]:
    """
    ARR-COC relevance-aware compression
    Tracked with W&B Weave
    """
    # Your ARR-COC logic here
    # (knowing, balancing, attending, realizing)

    result = {
        "compressed_tokens": token_budget,
        "relevance_scores": {
            "propositional": 0.85,
            "perspectival": 0.72,
            "participatory": 0.91
        },
        "lod_allocation": "64-400 tokens/patch"
    }

    return result

def gradio_wrapper(image, query, budget):
    """Gradio interface with full W&B tracking"""
    # Track with user context
    with weave.attributes({
        "interface": "gradio",
        "model": "arr-coc-v1",
        "budget_type": "dynamic"
    }):
        # Get result and call
        result, call = arr_coc_compress.call(
            image, query, budget
        )

        # Add custom metrics to trace
        call.summary.update({
            "compression_ratio": result["compressed_tokens"] / 1024,
            "relevance_mean": np.mean(list(
                result["relevance_scores"].values()
            ))
        })

    # Format for UI display
    output_text = f"""
    **Compression Result:**
    - Tokens: {result['compressed_tokens']}
    - LOD Strategy: {result['lod_allocation']}

    **Relevance Scores:**
    - Propositional: {result['relevance_scores']['propositional']:.2f}
    - Perspectival: {result['relevance_scores']['perspectival']:.2f}
    - Participatory: {result['relevance_scores']['participatory']:.2f}
    """

    return output_text

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ARR-COC Vision Compression")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image")
            query_input = gr.Textbox(label="Query")
            budget_slider = gr.Slider(
                64, 400, value=256,
                label="Token Budget"
            )
            submit_btn = gr.Button("Compress")

        with gr.Column():
            output_display = gr.Markdown(label="Results")

    submit_btn.click(
        fn=gradio_wrapper,
        inputs=[image_input, query_input, budget_slider],
        outputs=output_display
    )

demo.launch()
```

---

## Sources

**Web Research:**

From [Medium - FastAPI & W&B](https://medium.com/@jordao.cassiano.009/deploying-your-ml-model-into-production-using-wandb-fastapi-heroku-558aac33552a) (accessed 2025-01-31):
- FastAPI middleware patterns for ML monitoring
- Production deployment with W&B tracking

From [W&B Weave Tracing Quickstart](https://docs.wandb.ai/weave/guides/tracking/tracing) (accessed 2025-01-31):
- `@weave.op` decorator for automatic tracing
- Async function support and generator tracing
- `weave.attributes` for metadata
- Call objects and feedback API
- Postprocessing inputs/outputs for PII

From [Gradio W&B Integration](https://www.gradio.app/guides/Gradio-and-Wandb-Integration) (accessed 2025-01-31):
- Gradio event tracking patterns
- User interaction logging

From [W&B Streamlit Integration](https://wandb.ai/capecape/st30/reports/Simple-Streamlit-integration--VmlldzoxODgwMTUz) (accessed 2025-01-31):
- Streamlit + W&B embedding patterns
- Converting iframe to wandb.Html

From [W&B Performance Improvements](https://wandb.ai/wandb_fc/product-announcements-fc/reports/W-B-Models-performance-improvements-Fast-logging-immediate-results--VmlldzoxMjYyMjUzMA) (accessed 2025-01-31):
- Asynchronous logging architecture
- Local cache for reliability
- Performance optimization strategies

**Additional References:**
- [W&B fastai Integration](https://docs.wandb.ai/guides/integrations/fastai/) - Framework integration patterns
- [GitHub - W&B Async Logging](https://github.com/wandb/wandb/issues/5428) - Async patterns discussion
- [W&B LLM Debugging Guide](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) - Production LLM monitoring

**ARR-COC Validation:**
Referenced from existing ARR-COC architecture for realistic integration examples.
