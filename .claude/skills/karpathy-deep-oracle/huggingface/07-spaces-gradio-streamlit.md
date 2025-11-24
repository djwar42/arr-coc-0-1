# HuggingFace Spaces: Gradio, Streamlit, and Docker Deployment

## 1. Spaces Types and SDK Selection

HuggingFace Spaces provides four SDK options for deploying ML applications:

**Gradio Spaces** - Python-based interactive ML demos
- Rapid prototyping with minimal code
- Built-in components for images, text, audio, video
- Automatic input/output handling
- Real-time updates and streaming support
- Example: `app.py` with 20-30 lines creates full demo

**Streamlit Spaces** - Data science dashboard framework
- Widget-based UI building (sliders, buttons, charts)
- Native data visualization (Plotly, Matplotlib, Altair)
- State management and caching
- Multi-page app support
- Example: Dashboard with plots, metrics, model inference

**Docker Spaces** - Custom container environments
- Full control over runtime environment
- Any language/framework (FastAPI, Flask, Go, Rust)
- Custom dependencies and system packages
- Advanced networking and ports
- Example: Production API with FastAPI + Postgres

**Static HTML Spaces** - Client-side JavaScript apps
- No backend compute required
- WASM models (TensorFlow.js, ONNX.js)
- Interactive visualizations
- Lowest latency (runs in browser)
- Example: Lightweight demo with TensorFlow.js

### SDK Selection Guide

| Use Case | Recommended SDK | Why |
|----------|----------------|-----|
| ML model demo | Gradio | Built for ML, fastest setup |
| Data visualization | Streamlit | Rich plotting, dashboard widgets |
| Production API | Docker | Custom stack, full control |
| Client-side inference | Static | No compute cost, instant load |
| Multi-model pipeline | Docker | Complex orchestration |
| Simple chat interface | Gradio | Chat component, streaming |

From [HuggingFace Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview) (accessed 2025-11-16):
- Spaces integrate with Hub's git-based workflow
- Same repository tools (git, git-lfs) work for Spaces
- Automatic rebuild on every commit push
- Free CPU tier with optional GPU upgrades

## 2. Hardware Selection and Pricing

### CPU Hardware Options

| Hardware | CPU | Memory | Disk | Price/Hour | Best For |
|----------|-----|--------|------|------------|----------|
| CPU Basic | 2 vCPU | 16 GB | 50 GB | Free | Demos, lightweight models |
| CPU Upgrade | 8 vCPU | 32 GB | 50 GB | $0.03 | Data processing, preprocessing |

### GPU Hardware Options (2024 Pricing)

**Entry Tier - Nvidia T4**
- T4 Small: 4 vCPU, 15 GB RAM, 16 GB VRAM, 50 GB disk → $0.60/hour
- T4 Medium: 8 vCPU, 30 GB RAM, 16 GB VRAM, 100 GB disk → $0.90/hour
- Use case: Small language models (7B params), image classification, stable diffusion

**Mid Tier - Nvidia A10G**
- A10G Small: 4 vCPU, 15 GB RAM, 24 GB VRAM, 110 GB disk → $1.05/hour
- A10G Large: 12 vCPU, 46 GB RAM, 24 GB VRAM, 200 GB disk → $3.15/hour
- 2x A10G Large: 24 vCPU, 92 GB RAM, 48 GB VRAM, 1000 GB disk → $5.70/hour
- 4x A10G Large: 48 vCPU, 184 GB RAM, 96 GB VRAM, 2000 GB disk → $10.80/hour
- Use case: Medium LLMs (13B params), multi-modal models, batch inference

**High Performance - Nvidia A100**
- A100 Large: 12 vCPU, 142 GB RAM, 80 GB VRAM, 1000 GB disk → $4.13/hour
- Use case: Large models (70B params), training, fine-tuning

**Latest Generation - Nvidia H100**
- H100: 23 vCPU, 240 GB RAM, 80 GB VRAM, 3000 GB disk → $4.50/hour
- 8x H100: 184 vCPU, 1920 GB RAM, 640 GB VRAM, 24 TB disk → $36.00/hour
- Use case: Cutting-edge LLMs, multi-GPU training, high-throughput inference

From [HuggingFace Pricing](https://huggingface.co/pricing) (accessed 2025-11-16):
- Free CPU tier for public Spaces
- GPU billing per minute of runtime
- Automatic sleep after inactivity (free tier: 48 hours)
- Custom sleep time for paid hardware

### Hardware Selection Strategy

**Model Size Guidelines:**
- 7B params: T4 (16 GB VRAM sufficient)
- 13B params: A10G (24 GB VRAM recommended)
- 30-40B params: A100 (80 GB VRAM)
- 70B+ params: Multiple A100s or H100

**Inference vs Training:**
- Inference: Single GPU sufficient (T4/A10G)
- Fine-tuning: A100 or multi-GPU setup
- Training from scratch: Multi-GPU (8x H100)

**Cost Optimization:**
```python
# Enable custom sleep time (paid hardware only)
# Settings → Hardware → Sleep Time
# Options: Never, 15 min, 1 hour, 3 hours, 1 day, 3 days

# Programmatic hardware management
from huggingface_hub import HfApi
api = HfApi()

# Upgrade to GPU for batch job
api.request_space_hardware("username/space-name", "a10g-small")

# Downgrade to CPU after completion
api.request_space_hardware("username/space-name", "cpu-basic")
```

## 3. Gradio Spaces Development

### Basic Gradio App Structure

```python
# app.py - Minimal Gradio Space
import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline("sentiment-analysis")

def analyze(text):
    results = classifier(text)
    return results[0]

# Create interface
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis",
    description="Analyze text sentiment using transformers"
)

if __name__ == "__main__":
    demo.launch()
```

### Advanced Gradio Features

**Multi-Input Components:**
```python
import gradio as gr
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def caption_image(image, max_length, num_beams):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    output_ids = model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams
    )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(10, 50, value=16, step=1, label="Max Length"),
        gr.Slider(1, 10, value=4, step=1, label="Num Beams")
    ],
    outputs=gr.Textbox(label="Caption"),
    examples=[
        ["examples/cat.jpg", 16, 4],
        ["examples/city.jpg", 20, 5]
    ]
)

demo.launch()
```

**Streaming Outputs:**
```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def generate_stream(prompt, max_length):
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generator for streaming
    for i in range(max_length):
        outputs = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + i + 1,
            do_sample=True,
            temperature=0.7
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        yield text

demo = gr.Interface(
    fn=generate_stream,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(10, 100, value=50, label="Max Length")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    live=False  # Set to True for real-time updates
)

demo.launch()
```

From [Gradio Documentation](https://gradio.app/docs/) (accessed 2025-11-16):
- Gradio 4.x supports custom components and themes
- Built-in support for HuggingFace model loading
- Automatic API generation for programmatic access

### requirements.txt for Gradio Spaces

```txt
gradio>=4.0.0
transformers>=4.30.0
torch>=2.0.0
pillow
```

## 4. Streamlit Spaces Development

### Basic Streamlit App

```python
# app.py - Streamlit Space
import streamlit as st
from transformers import pipeline
import pandas as pd

st.title("Text Classification Dashboard")

# Sidebar configuration
model_name = st.sidebar.selectbox(
    "Select Model",
    ["distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment"]
)

# Load model with caching
@st.cache_resource
def load_model(name):
    return pipeline("sentiment-analysis", model=name)

classifier = load_model(model_name)

# Input
text = st.text_area("Enter text for classification:")

if st.button("Classify"):
    if text:
        results = classifier(text)

        # Display results
        st.subheader("Results")
        df = pd.DataFrame(results)
        st.dataframe(df)

        # Visualization
        st.bar_chart(df.set_index('label')['score'])
    else:
        st.warning("Please enter text")

# File upload
uploaded_file = st.file_uploader("Or upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    if st.button("Classify All"):
        texts = df['text'].tolist()
        results = classifier(texts)
        df['sentiment'] = [r['label'] for r in results]
        df['confidence'] = [r['score'] for r in results]
        st.dataframe(df)
```

### Advanced Streamlit Features

**Multi-Page App:**
```python
# pages/1_Model_Inference.py
import streamlit as st

st.set_page_config(page_title="Inference", page_icon="🔮")
st.title("Model Inference")
# ... inference code

# pages/2_Model_Comparison.py
import streamlit as st

st.set_page_config(page_title="Comparison", page_icon="⚖️")
st.title("Model Comparison")
# ... comparison code
```

**State Management:**
```python
import streamlit as st

# Session state for persistent data
if 'history' not in st.session_state:
    st.session_state.history = []

text = st.text_input("Enter text:")
if st.button("Submit"):
    result = classifier(text)
    st.session_state.history.append({
        'text': text,
        'result': result[0]
    })

# Display history
st.subheader("History")
for item in st.session_state.history:
    st.write(f"{item['text']}: {item['result']['label']}")
```

### packages.txt for System Dependencies

```txt
# Streamlit Spaces system packages
ffmpeg
libsm6
libxext6
```

From [Streamlit Documentation](https://docs.streamlit.io/) (accessed 2025-11-16):
- Streamlit 1.28+ supports native theming
- Built-in caching decorators (@st.cache_resource, @st.cache_data)
- Multi-page apps organized in pages/ directory

## 5. Docker Spaces for Custom Environments

### Docker Space Setup

**README.md Header:**
```yaml
---
title: Custom Docker Space
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---
```

### Basic Dockerfile Pattern

```dockerfile
FROM python:3.10

# Set up user (required for Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy application files
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### FastAPI + Docker Space Example

```python
# main.py - FastAPI application
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

# Load model at startup
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

@app.get("/")
def read_root():
    return {"message": "Image Classification API"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = classifier(image)
    return {"predictions": results}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**Advanced Dockerfile with GPU Support:**
```dockerfile
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=user . .

# Expose port
EXPOSE 7860

# Run with GPU support
CMD ["python3", "app.py"]
```

From [Docker Spaces Documentation](https://huggingface.co/docs/hub/en/spaces-sdks-docker) (accessed 2025-11-16):
- Container runs with user ID 1000 (security requirement)
- Always use `--chown=user` with COPY and ADD commands
- GPU available at runtime, not during build
- Internal ports configurable, external port fixed at app_port

### Multi-Service Docker Compose Pattern

While Spaces doesn't support docker-compose directly, you can simulate multi-service with supervisor:

```dockerfile
FROM python:3.10

RUN apt-get update && apt-get install -y supervisor redis-server

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . .
RUN pip install --no-cache-dir -r requirements.txt

# Supervisor config
COPY --chown=user supervisord.conf /etc/supervisor/conf.d/

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

## 6. Secrets Management and Environment Variables

### Variables vs Secrets

**Variables** (public, visible):
- Configuration values
- Model identifiers
- Public API endpoints
- Automatically copied when Space is duplicated

**Secrets** (private, hidden):
- API keys (OpenAI, Anthropic)
- Database credentials
- OAuth tokens
- HuggingFace tokens
- NOT copied when Space is duplicated

From [Spaces Overview - Managing Secrets](https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets) (accessed 2025-11-16):
- Configure via Space Settings → Secrets and Variables
- Accessible as environment variables in all SDK types
- Secrets Scanner detects hard-coded secrets in code

### Using Environment Variables

**Python (Gradio/Streamlit):**
```python
import os
import gradio as gr
from huggingface_hub import InferenceClient

# Access secrets
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Access variables
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "default-model")

# Use with HuggingFace Hub
client = InferenceClient(token=HF_TOKEN)

def inference(text):
    # Use private model with token
    response = client.text_generation(
        text,
        model=MODEL_REPO_ID
    )
    return response

demo = gr.Interface(fn=inference, inputs="text", outputs="text")
demo.launch()
```

**Docker Spaces - Runtime Secrets:**
```python
# app.py
import os

# Secrets available as environment variables at runtime
DB_PASSWORD = os.environ.get("DB_PASSWORD")
API_KEY = os.environ.get("API_KEY")
```

**Docker Spaces - Build-time Secrets:**
```dockerfile
# Dockerfile with build-time secret access
FROM python:3.10

# Declare ARG for build-time variable
ARG MODEL_REPO_NAME

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Use build-time variable
RUN echo "Building with model: $MODEL_REPO_NAME"

# Mount secret at build time (read-only)
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    huggingface-cli login --token $(cat /run/secrets/HF_TOKEN)

# Download private model at build time
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('private-org/private-model', \
                                token=open('/run/secrets/HF_TOKEN').read().strip())"

COPY --chown=user . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
```

### Helper Environment Variables

HuggingFace automatically provides these environment variables:

```python
import os

# Space metadata
SPACE_AUTHOR_NAME = os.getenv("SPACE_AUTHOR_NAME")  # e.g., "username"
SPACE_REPO_NAME = os.getenv("SPACE_REPO_NAME")      # e.g., "my-space"
SPACE_TITLE = os.getenv("SPACE_TITLE")              # From README.md
SPACE_ID = os.getenv("SPACE_ID")                    # "username/my-space"
SPACE_HOST = os.getenv("SPACE_HOST")                # "username-my-space.hf.space"

# Hardware info
CPU_CORES = os.getenv("CPU_CORES")      # e.g., "4"
MEMORY = os.getenv("MEMORY")            # e.g., "15Gi"

# OAuth (if enabled)
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
```

### Best Practices for Secrets

```python
# ❌ BAD: Hard-coded secrets
OPENAI_KEY = "sk-proj-abcdef123456"

# ✅ GOOD: Environment variables
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY secret not configured")

# ❌ BAD: Logging secrets
print(f"Using API key: {OPENAI_KEY}")

# ✅ GOOD: Safe logging
print(f"Using API key: {'*' * len(OPENAI_KEY)}")

# ❌ BAD: Exposing secrets in UI
gr.Textbox(value=OPENAI_KEY, label="API Key")

# ✅ GOOD: Input only, never pre-fill
gr.Textbox(label="API Key", type="password")
```

## 7. Spaces SDK and Programmatic Deployment

### HuggingFace Hub Python Library

```python
from huggingface_hub import (
    HfApi,
    create_repo,
    upload_file,
    upload_folder,
    SpaceHardware
)

api = HfApi()

# Create new Space
repo_id = api.create_repo(
    repo_id="username/my-gradio-space",
    repo_type="space",
    space_sdk="gradio",
    private=False
)

# Upload files
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="username/my-gradio-space",
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="username/my-gradio-space",
    repo_type="space"
)

# Upload entire folder
api.upload_folder(
    folder_path="./my_space",
    repo_id="username/my-gradio-space",
    repo_type="space"
)
```

### Hardware Management API

```python
from huggingface_hub import HfApi, SpaceHardware

api = HfApi()

# Check current hardware
space_info = api.space_info("username/my-space")
print(f"Current hardware: {space_info.runtime.hardware}")

# Request hardware upgrade
api.request_space_hardware(
    repo_id="username/my-space",
    hardware=SpaceHardware.A10G_SMALL
)

# Available hardware options
# SpaceHardware.CPU_BASIC
# SpaceHardware.CPU_UPGRADE
# SpaceHardware.T4_SMALL
# SpaceHardware.T4_MEDIUM
# SpaceHardware.A10G_SMALL
# SpaceHardware.A10G_LARGE
# SpaceHardware.A100_LARGE

# Request downgrade to save costs
api.request_space_hardware(
    repo_id="username/my-space",
    hardware=SpaceHardware.CPU_BASIC
)
```

### Secrets Management API

```python
from huggingface_hub import HfApi

api = HfApi()

# Add secret
api.add_space_secret(
    repo_id="username/my-space",
    key="OPENAI_API_KEY",
    value="sk-proj-..."
)

# Add variable (public)
api.add_space_variable(
    repo_id="username/my-space",
    key="MODEL_REPO_ID",
    value="openai/whisper-large-v3"
)

# Delete secret
api.delete_space_secret(
    repo_id="username/my-space",
    key="OPENAI_API_KEY"
)
```

### CI/CD Integration with GitHub Actions

```yaml
# .github/workflows/deploy-space.yml
name: Deploy to HuggingFace Space

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install huggingface_hub

      - name: Deploy to Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python - <<EOF
          from huggingface_hub import HfApi
          import os

          api = HfApi(token=os.environ['HF_TOKEN'])

          api.upload_folder(
              folder_path=".",
              repo_id="username/my-space",
              repo_type="space",
              ignore_patterns=[".git", ".github", "*.md"]
          )
          EOF
```

From [HuggingFace Hub Manage Spaces Guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces) (accessed 2025-11-16):
- Programmatic control over Space lifecycle
- Hardware can be changed dynamically
- Secrets added via API not visible in UI

## 8. arr-coc-0-1 VLM Demo Deployment on Spaces

### arr-coc-0-1 Gradio Space Architecture

The arr-coc-0-1 project deploys its vision-language model demo on HuggingFace Spaces using Gradio. This demonstrates real-world Vervaekean relevance realization with dynamic visual token allocation.

**Space URL**: [NorthHead/arr-coc-0-1](https://huggingface.co/spaces/NorthHead/arr-coc-0-1)

### Deployment Structure

```
arr-coc-0-1/
├── app.py                    # Gradio interface
├── requirements.txt          # Dependencies
├── arr_coc/
│   ├── knowing.py           # Three ways of knowing
│   ├── balancing.py         # Opponent processing
│   ├── attending.py         # Relevance → token budgets
│   ├── realizing.py         # Pipeline orchestrator
│   └── adapter.py           # Quality adapter
└── README.md                # Space metadata
```

### Gradio App Implementation

```python
# app.py - arr-coc-0-1 Gradio Space
import gradio as gr
import torch
from PIL import Image
from arr_coc.realizing import RelevanceRealizationPipeline
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Initialize relevance realization pipeline
pipeline = RelevanceRealizationPipeline(
    vision_encoder=model.visual,
    adapter=model.adapter,  # Quality adapter (4th P: Procedural)
    num_patches=196,        # 14x14 grid
    base_token_budget=200   # K = 200 tokens
)

def process_image_query(image, query, min_lod, max_lod, balance_explore):
    """
    Args:
        image: PIL Image
        query: str - User question about image
        min_lod: int - Minimum tokens per patch (64-128)
        max_lod: int - Maximum tokens per patch (256-400)
        balance_explore: float - Exploration vs exploitation (0.0-1.0)

    Returns:
        answer: str
        relevance_map: Image (heatmap)
        token_allocation: dict
    """

    # Extract visual features
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    vision_features = model.visual(**inputs)

    # Encode query
    text_inputs = processor(text=query, return_tensors="pt").to(model.device)
    query_embedding = model.get_text_features(**text_inputs)

    # Realize relevance (Vervaekean process)
    compressed_features, metadata = pipeline(
        vision_features=vision_features,
        query_embedding=query_embedding,
        min_lod=min_lod,
        max_lod=max_lod,
        balance_params={'explore_weight': balance_explore}
    )

    # Generate answer
    combined = torch.cat([compressed_features, query_embedding], dim=1)
    outputs = model.generate(combined, max_length=100)
    answer = processor.decode(outputs[0], skip_special_tokens=True)

    # Visualize relevance map
    relevance_scores = metadata['relevance_scores'].cpu().numpy()
    relevance_map = create_heatmap(image, relevance_scores)

    # Token allocation breakdown
    allocation = {
        'Total Tokens': metadata['total_tokens'],
        'High Relevance Patches': metadata['high_lod_count'],
        'Medium Relevance Patches': metadata['medium_lod_count'],
        'Low Relevance Patches': metadata['low_lod_count'],
        'Compression Ratio': f"{metadata['compression_ratio']:.2f}x"
    }

    return answer, relevance_map, allocation

# Gradio interface
demo = gr.Interface(
    fn=process_image_query,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question", placeholder="What do you see in this image?"),
        gr.Slider(64, 128, value=64, step=16, label="Min LOD (tokens/patch)"),
        gr.Slider(256, 400, value=400, step=16, label="Max LOD (tokens/patch)"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Exploration Weight")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Image(type="pil", label="Relevance Map"),
        gr.JSON(label="Token Allocation")
    ],
    title="ARR-COC: Vervaekean Relevance Realization",
    description="""
    Adaptive Relevance Realization with Contexts Optical Compression.

    **How it works:**
    1. **Knowing**: Measures relevance through propositional, perspectival, and participatory dimensions
    2. **Balancing**: Navigates tensions (compress↔particularize, exploit↔explore)
    3. **Attending**: Maps relevance scores to variable LOD (64-400 tokens per patch)
    4. **Realizing**: Executes compression and generates focused response

    **Controls:**
    - Min/Max LOD: Token budget range for patches
    - Exploration Weight: Balance between focusing on known relevant areas vs exploring new areas
    """,
    examples=[
        ["examples/city_street.jpg", "What vehicles are visible?", 64, 400, 0.3],
        ["examples/nature.jpg", "Describe the landscape", 64, 256, 0.5],
        ["examples/portrait.jpg", "What is the person's expression?", 128, 400, 0.2]
    ]
)

if __name__ == "__main__":
    demo.launch()
```

### Hardware Requirements

**Model Size**: Qwen2-VL-7B-Instruct (~14 GB in FP16)

**Recommended Hardware**:
- Development: T4 Medium (16 GB VRAM) - $0.90/hour
- Production: A10G Small (24 GB VRAM) - $1.05/hour
- Batch processing: A10G Large (24 GB VRAM) - $3.15/hour

**Space Settings**:
```yaml
# README.md header
---
title: ARR-COC VLM Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
```

### requirements.txt

```txt
# Core dependencies
gradio>=4.44.0
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0

# Vision processing
pillow
opencv-python-headless

# Visualization
matplotlib
numpy

# Model quantization (optional, for memory efficiency)
bitsandbytes>=0.41.0
```

### Deployment Workflow

**Local Development:**
```bash
cd arr-coc-0-1
python app.py  # Test locally at http://localhost:7860
```

**Push to Space:**
```bash
# Initialize space repo
git remote add space https://huggingface.co/spaces/NorthHead/arr-coc-0-1

# Push code
git add app.py requirements.txt arr_coc/ README.md
git commit -m "Deploy Vervaekean VLM demo"
git push space main

# Space automatically rebuilds and deploys
```

**Hardware Upgrade:**
```python
from huggingface_hub import HfApi

api = HfApi(token="hf_...")

# Upgrade to GPU for inference
api.request_space_hardware(
    repo_id="NorthHead/arr-coc-0-1",
    hardware="a10g-small"
)
```

### Monitoring and Logs

**Space Logs Access:**
- Navigate to Space → Settings → Logs
- Real-time logs during build and runtime
- Debug startup issues and model loading

**Performance Metrics:**
```python
import time

def process_image_query(image, query, min_lod, max_lod, balance_explore):
    start_time = time.time()

    # ... processing ...

    processing_time = time.time() - start_time

    # Add to metadata
    allocation['Processing Time'] = f"{processing_time:.2f}s"

    return answer, relevance_map, allocation
```

### Integration with Training Pipeline

The arr-coc-0-1 project demonstrates end-to-end MLOps:

**Training (Vertex AI + W&B)**:
```bash
# Launch training job
python training/cli.py launch

# Model artifacts saved to GCS
# gs://arr-coc-models/checkpoints/run-123/
```

**Deployment (HuggingFace Space)**:
```python
# After training completes, update Space
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])

# Upload new checkpoint
api.upload_folder(
    folder_path="checkpoints/run-123",
    repo_id="NorthHead/arr-coc-model",
    repo_type="model"
)

# Update Space to use new model
# (Modify app.py MODEL_ID variable)
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="NorthHead/arr-coc-0-1",
    repo_type="space"
)
```

From arr-coc-0-1 project (2025-01):
- Gradio Space provides public demo interface
- GitHub repo at https://github.com/djwar42/arr-coc-0-1
- Full training infrastructure on GCP Vertex AI
- Model versioning through HuggingFace Hub

## Sources

**Source Documents:**
- [huggingface-hub/spaces/overview.md](../huggingface-hub/spaces/overview.md)
- [huggingface-hub/spaces/gradio.md](../huggingface-hub/spaces/gradio.md)
- [huggingface-hub/spaces/docker.md](../huggingface-hub/spaces/docker.md)
- [huggingface-hub/spaces/gpu-upgrades.md](../huggingface-hub/spaces/gpu-upgrades.md)

**Web Research:**
- [HuggingFace Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview) (accessed 2025-11-16)
- [HuggingFace Pricing](https://huggingface.co/pricing) (accessed 2025-11-16)
- [Gradio Documentation](https://gradio.app/docs/) (accessed 2025-11-16)
- [Docker Spaces Documentation](https://huggingface.co/docs/hub/en/spaces-sdks-docker) (accessed 2025-11-16)
- [HuggingFace Hub Manage Spaces Guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces) (accessed 2025-11-16)

**Implementation Reference:**
- arr-coc-0-1 Gradio Space deployment (RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/)
- GitHub: https://github.com/djwar42/arr-coc-0-1
- HuggingFace Space: https://huggingface.co/spaces/NorthHead/arr-coc-0-1

**Additional References:**
- [Streamlit Documentation](https://docs.streamlit.io/) (accessed 2025-11-16)
- HuggingFace Spaces deployment best practices (community tutorials, 2024-2025)
