# HuggingFace Spaces Deployment for Research Demos (2024-2025)

## Overview

HuggingFace Spaces provides a platform for deploying and sharing machine learning applications with interactive, browser-based interfaces. This guide focuses on deploying Gradio-based research demos on Spaces, covering setup, GPU configuration, deployment workflows, and best practices for VLM testing and validation.

From [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces-overview) (accessed 2025-01-31):
- Free hosting for ML demos with automatic rebuild on git push
- Support for Gradio, Streamlit, Docker, and static HTML
- GPU upgrades available (T4, A10G, A100, H100)
- Simple git-based deployment workflow

## Spaces Setup

### Creating a New Space

**Step-by-step process** (from [PyImageSearch deployment guide](https://pyimagesearch.com/2024/12/30/deploy-gradio-apps-on-hugging-face-spaces/), accessed 2025-01-31):

1. **Navigate to Spaces creation:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"

2. **Configure Space settings:**
   - **Owner:** Select username or organization
   - **Space name:** Choose descriptive name (e.g., `vqa-paligemma-demo`)
   - **Description:** Brief summary of functionality
   - **License:** Select appropriate license (MIT, Apache-2.0, etc.)
   - **SDK:** Choose `Gradio` for interactive ML demos
   - **Visibility:** Public (shareable) or Private (restricted access)

3. **Select hardware:**
   - CPU Basic (free): 2 vCPU, 16GB RAM, 50GB disk
   - GPU options available for upgrade (see GPU Configuration section)

### Required Files

**Minimal Gradio Space structure:**

```
your-space/
├── README.md          # Space metadata (auto-generated)
├── requirements.txt   # Python dependencies
└── app.py            # Main Gradio application
```

**Example README.md header** (from [HuggingFace Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference), accessed 2025-01-31):

```yaml
---
title: Visual Question Answering Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
```

### Creating Files in Spaces

**Option 1: Web interface**
- Navigate to "Files" tab in Space
- Click "Add File" → "Create new file"
- Enter filename (`requirements.txt` or `app.py`)
- Add content and commit changes

**Option 2: Git workflow**
```bash
# Clone Space repository
git clone https://huggingface.co/spaces/username/space-name

# Add files locally
cd space-name
echo "gradio>=4.0.0" > requirements.txt
echo "transformers" >> requirements.txt

# Push changes
git add .
git commit -m "Add initial files"
git push
```

From [HuggingFace Spaces Overview](https://huggingface.co/docs/hub/spaces-overview) (accessed 2025-01-31):
- Spaces use git repositories (same as models/datasets)
- Automatic rebuild on each commit
- Build logs available in Space settings

## GPU Configuration

### Hardware Options

**CPU Hardware** (from [HuggingFace GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus), accessed 2025-01-31):

| Hardware | CPU | Memory | GPU Memory | Disk | Hourly Price |
|----------|-----|--------|------------|------|--------------|
| CPU Basic | 2 vCPU | 16 GB | - | 50 GB | Free |
| CPU Upgrade | 8 vCPU | 32 GB | - | 50 GB | $0.03 |

**GPU Hardware Options:**

| Hardware | CPU | Memory | GPU Memory | Disk | Hourly Price |
|----------|-----|--------|------------|------|--------------|
| Nvidia T4 - small | 4 vCPU | 15 GB | 16 GB | 50 GB | $0.40 |
| Nvidia T4 - medium | 8 vCPU | 30 GB | 16 GB | 100 GB | $0.60 |
| Nvidia L4 (1x) | 8 vCPU | 30 GB | 24 GB | 400 GB | $0.80 |
| Nvidia L4 (4x) | 48 vCPU | 186 GB | 96 GB | 3200 GB | $3.80 |
| Nvidia A10G - small | 4 vCPU | 14 GB | 24 GB | 110 GB | $1.00 |
| Nvidia A10G - large | 12 vCPU | 46 GB | 24 GB | 200 GB | $1.50 |
| Nvidia A100 - large | 12 vCPU | 142 GB | 80 GB | 1000 GB | $2.50 |
| Nvidia H100 | 23 vCPU | 240 GB | 80 GB | 3000 GB | $4.50 |

### Framework-Specific GPU Setup

**PyTorch with CUDA** (requirements.txt):
```
--extra-index-url https://download.pytorch.org/whl/cu113
torch
transformers
```

**Verification in app.py:**
```python
import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Output: Is CUDA available: True
#         CUDA device: Tesla T4
```

**JAX with CUDA** (requirements.txt):
```
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
jax[cuda11_pip]
jaxlib
```

**TensorFlow** (requirements.txt):
```
tensorflow
```
Auto-detects CUDA; verify with:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Upgrading to GPU

From [HuggingFace GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus) (accessed 2025-01-31):

1. Click "Settings" in Space top navigation
2. Select "Hardware" section
3. Choose desired GPU tier
4. Confirm upgrade (billing starts when Space is running)

**Community GPU Grants:**
- Available for innovative public demos
- Apply via "Settings" → "Sleep time settings" → "Apply for community GPU grant"
- Covers upgrade costs for approved projects

### Sleep Time Configuration

**Default behavior:**
- CPU Basic: Sleeps after 48 hours of inactivity (free)
- Upgraded hardware: Runs indefinitely by default

**Custom sleep time options:**
- Set in Settings → Sleep time
- Space goes to "stopped" state when inactive
- No billing while sleeping
- Auto-wakes on new visitor

**Options:**
- Never sleep (always running)
- Sleep after 15 minutes
- Sleep after 1 hour
- Sleep after 8 hours
- Sleep after 48 hours

## Deployment Workflow

### Basic Gradio App Structure

**requirements.txt:**
```
gradio>=4.0.0
transformers
torch
peft
bitsandbytes
```

**app.py example** (from [PyImageSearch deployment guide](https://pyimagesearch.com/2024/12/30/deploy-gradio-apps-on-hugging-face-spaces/), accessed 2025-01-31):

```python
import gradio as gr
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

# Load model and processor
model_id = "username/finetuned-model"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# Define inference function
def process_image(image, prompt):
    # Process inputs
    inputs = processor(image.convert("RGB"), prompt, return_tensors="pt")

    try:
        # Generate output
        output = model.generate(**inputs, max_new_tokens=20)

        # Decode and return
        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        return decoded_output[len(prompt):]
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during processing."

# Create Gradio interface
inputs = [
    gr.Image(type="pil"),
    gr.Textbox(label="Prompt", placeholder="Enter your question")
]
outputs = gr.Textbox(label="Answer")

# Launch app
demo = gr.Interface(
    fn=process_image,
    inputs=inputs,
    outputs=outputs,
    title="Visual Question Answering Demo",
    description="Upload an image and ask questions to get answers."
)

demo.launch()
```

### Deployment Steps

1. **Create Space** with desired configuration
2. **Add requirements.txt** with all dependencies
3. **Add app.py** with Gradio interface
4. **Commit changes** (via web UI or git push)
5. **Monitor build logs** in Space interface
6. **Test deployment** once build completes

**Automatic rebuild process:**
- Triggered on every git commit
- Installs dependencies from requirements.txt
- Runs app.py
- Makes Space available at `username-space-name.hf.space`

### Build Logs and Debugging

From [HuggingFace Spaces Forums](https://discuss.huggingface.co/t/how-to-debug-spaces-on-hf-co/13191) (accessed 2025-01-31):

**Enable debug mode** in app.py:
```python
demo.launch(debug=True)
```

**Check build status:**
- Click "Running" or "Build error" badge in Space
- View detailed logs in Settings → Logs
- Common errors: missing dependencies, CUDA version mismatches, model loading failures

**Debugging tips:**
- Use `print()` statements in app.py (appears in logs)
- Verify CUDA availability if using GPU
- Check dependency versions match requirements.txt
- Test locally before deploying

## Best Practices for Research Demos

### Simplified Public vs Full Local Version

**Strategy for resource-limited demos:**
- Offer lightweight model on Spaces (smaller checkpoint)
- Provide full model weights for local download
- Link to GitHub repo with complete implementation

**Example approach:**
```python
# Use quantized or smaller model for Spaces
model_id = "username/model-small-quantized"  # For Spaces
# Full model: "username/model-full"  # For local use
```

### Providing Example Inputs

**Best practice from deployed Spaces:**
```python
examples = [
    ["example_image1.jpg", "What is in this image?"],
    ["example_image2.jpg", "How many people are present?"],
    ["example_image3.jpg", "Describe the scene in detail."]
]

demo = gr.Interface(
    fn=process_image,
    inputs=inputs,
    outputs=outputs,
    examples=examples  # Pre-populated examples for users
)
```

### Usage Instructions

**Add clear instructions in README.md:**
```markdown
## How to Use

1. Upload an image using the image upload box
2. Enter your question in the prompt field
3. Click "Submit" to get the model's answer
4. Try the example inputs below to see different capabilities

## Limitations

- Model may struggle with complex scenes
- Best performance on clear, well-lit images
- Maximum 20 token responses for speed
```

### Rate Limiting Considerations

**For popular Spaces:**
- Free CPU: Shared resources, may experience queuing
- Upgraded GPU: Dedicated resources, faster inference
- Consider enabling "Sleep after inactivity" to control costs

**Programmatic rate limiting** (optional):
```python
import time
from functools import lru_cache

last_request_time = {}

def process_image(image, prompt):
    # Simple rate limiting (1 request per 5 seconds per session)
    user_id = gr.get_session_id()
    if user_id in last_request_time:
        elapsed = time.time() - last_request_time[user_id]
        if elapsed < 5:
            return f"Please wait {5-elapsed:.1f} seconds before next request"

    last_request_time[user_id] = time.time()
    # ... rest of processing
```

## Managing Secrets and Environment Variables

From [HuggingFace Spaces Overview](https://huggingface.co/docs/hub/spaces-overview) (accessed 2025-01-31):

### Secrets vs Variables

**Secrets:**
- Private, cannot be viewed after setting
- For API keys, tokens, credentials
- Not copied when Space is duplicated
- Set in Settings → Repository secrets

**Variables:**
- Public, viewable in Settings
- For non-sensitive configuration
- Automatically copied when duplicated
- Set in Settings → Variables

### Adding Secrets/Variables

**Via web interface:**
1. Navigate to Settings in Space
2. Scroll to "Repository secrets" or "Variables"
3. Click "Add a new secret/variable"
4. Enter name and value
5. Save

**Accessing in app.py:**
```python
import os

# Access secret (e.g., HuggingFace token)
hf_token = os.getenv('HF_TOKEN')

# Access variable (e.g., model repo ID)
model_id = os.getenv('MODEL_REPO_ID')

# Use in model loading
model = AutoModel.from_pretrained(
    model_id,
    token=hf_token
)
```

**Example use case for VLM demos:**
```python
# Set as secrets in Space settings
HF_TOKEN=hf_xxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxx

# Set as variables in Space settings
MODEL_REPO_ID=username/vqa-model
MAX_NEW_TOKENS=50
```

## Helper Environment Variables

From [HuggingFace Spaces Overview](https://huggingface.co/docs/hub/spaces-overview) (accessed 2025-01-31):

**Available at runtime:**
```python
import os

# Hardware info
cpu_cores = os.getenv('CPU_CORES')        # "4"
memory = os.getenv('MEMORY')              # "15Gi"

# Space info
author = os.getenv('SPACE_AUTHOR_NAME')   # "username"
repo_name = os.getenv('SPACE_REPO_NAME')  # "space-name"
title = os.getenv('SPACE_TITLE')          # "My Demo Space"
space_id = os.getenv('SPACE_ID')          # "username/space-name"
space_host = os.getenv('SPACE_HOST')      # "username-space-name.hf.space"
```

**Use case: Dynamic model loading based on Space:**
```python
# Load different model based on Space ID
space_id = os.getenv('SPACE_ID', 'default/demo')
if 'research' in space_id:
    model_id = "username/research-model"
else:
    model_id = "username/demo-model"
```

## Persistent Storage

From [HuggingFace Spaces Storage](https://huggingface.co/docs/hub/spaces-storage) (accessed 2025-01-31):

**Storage tiers:**

| Tier | Size | Persistent | Monthly Price |
|------|------|------------|---------------|
| Ephemeral (default) | 50GB | No | Free |
| Small | Ephemeral + 20GB | Yes | $5 |
| Medium | Ephemeral + 150GB | Yes | $25 |
| Large | Ephemeral + 1TB | Yes | $100 |

**Use cases for research demos:**
- Cache downloaded models (avoid re-downloading on rebuild)
- Store user-uploaded datasets
- Save intermediate results
- Log inference statistics

**Accessing persistent storage:**
```python
import os

# Persistent storage path
storage_path = os.getenv('PERSISTENT_STORAGE_PATH', '/data')

# Cache model downloads
cache_dir = os.path.join(storage_path, 'model_cache')
model = AutoModel.from_pretrained(
    model_id,
    cache_dir=cache_dir
)
```

## Embedding Spaces

From [HuggingFace Spaces Embed](https://huggingface.co/docs/hub/spaces-embed) (accessed 2025-01-31):

**Embed Space in external website:**
```html
<!-- Full embed -->
<iframe
    src="https://username-space-name.hf.space"
    frameborder="0"
    width="850"
    height="450"
></iframe>

<!-- Direct URL (no gradio wrapper) -->
<iframe
    src="https://username-space-name.hf.space?__theme=light"
    frameborder="0"
    width="850"
    height="450"
></iframe>
```

**Embedding options:**
- `?__theme=light` or `?__theme=dark`: Set theme
- Direct embedding in research papers (e.g., Papers with Code)
- Integration into personal portfolio websites

## Advanced: Programmatic Space Management

From [HuggingFace Hub Python Library](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage_spaces) (accessed 2025-01-31):

**Using huggingface_hub for dynamic GPU allocation:**
```python
from huggingface_hub import HfApi

api = HfApi(token="your_hf_token")

# Upgrade Space to GPU
api.request_space_hardware(
    repo_id="username/space-name",
    hardware="t4-small"
)

# Downgrade to CPU after task
api.request_space_hardware(
    repo_id="username/space-name",
    hardware="cpu-basic"
)
```

**Use cases:**
- Cost optimization (upgrade only when needed)
- Scheduled GPU usage (upgrade during demos, downgrade overnight)
- Batch processing workflows

## Billing and Cost Management

From [HuggingFace Spaces GPU](https://huggingface.co/docs/hub/spaces-gpus) (accessed 2025-01-31):

### Billing Model

- **Billed by the minute:** Only charged when Space is "Running"
- **No cost during build:** Building/startup is free
- **Auto-suspend on failure:** Failing Spaces stop billing automatically
- **Free hardware never billed:** CPU Basic is always free

### Cost Control Strategies

**1. Use sleep time:**
```
Settings → Sleep time → "Sleep after 15 minutes of inactivity"
```
Saves costs when demo not in active use.

**2. Pause Space when not needed:**
```
Settings → Pause Space
```
Completely stops execution and billing (owner can restart).

**3. Downgrade hardware:**
```
Settings → Hardware → Change to CPU Basic
```
Revert to free tier when GPU not required.

**4. Monitor usage:**
- Check billing dashboard at https://huggingface.co/settings/billing
- Review Space runtime in Settings

### Example Cost Scenarios

**VQA Research Demo (T4-small):**
- Running 24/7: $0.40/hour × 24 × 30 = $288/month
- With 1-hour sleep time: ~$120/month (assuming 50% active usage)
- With 15-min sleep time: ~$60/month (assuming 25% active usage)

**Best practice:** Start with aggressive sleep time, monitor usage, adjust as needed.

## Common Issues and Solutions

### Build Failures

**Issue: Missing dependencies**
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution:** Add to requirements.txt:
```
transformers>=4.35.0
```

**Issue: CUDA version mismatch**
```
RuntimeError: CUDA error: no kernel image is available
```
**Solution:** Specify correct PyTorch CUDA version in requirements.txt:
```
--extra-index-url https://download.pytorch.org/whl/cu118
torch
```

### Runtime Errors

**Issue: Out of memory on GPU**
```
torch.cuda.OutOfMemoryError
```
**Solutions:**
1. Reduce batch size
2. Use model quantization (8-bit, 4-bit)
3. Upgrade to larger GPU tier
4. Enable gradient checkpointing

**Issue: Slow inference**
**Solutions:**
1. Use model optimization (TorchScript, ONNX)
2. Enable half-precision (fp16)
3. Upgrade GPU tier
4. Cache model outputs for common inputs

### Debugging Build Logs

From [HuggingFace Spaces Forums](https://discuss.huggingface.co/t/spaces-runtime-logging/139383) (accessed 2025-01-31):

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)

# In Gradio
demo.launch(debug=True)
```

**Check logs location:**
- Build logs: Visible during Space rebuild
- Runtime logs: Settings → Logs tab
- stderr logged by default (stdout may require explicit logging)

## Real-World Examples

From PyImageSearch and HuggingFace Spaces (accessed 2025-01-31):

**Deployed VLM demos:**
1. [Visual Question Answering](https://huggingface.co/spaces/pyimagesearch/visual-question-answer-finetuned-paligemma)
   - Fine-tuned PaliGemma on VQAv2
   - T4 GPU for inference
   - Example inputs provided

2. [Document Understanding](https://huggingface.co/spaces/pyimagesearch/document-understanding-mix-paligemma)
   - OCR + comprehension
   - Handles document images
   - Text extraction + QA

3. [Image Captioning](https://huggingface.co/spaces/pyimagesearch/image-captioning-mix-paligemma)
   - Generate image descriptions
   - Multiple caption styles
   - CPU-friendly demo

4. [Video Captioning](https://huggingface.co/spaces/pyimagesearch/video-captioning-mix-paligemma)
   - Frame extraction + captioning
   - Temporal aggregation
   - GPU-accelerated

## Best Practices Summary

### For Research Demos

1. **Start with CPU Basic** (free tier), upgrade only if needed
2. **Provide example inputs** to showcase capabilities
3. **Add clear documentation** in README and interface
4. **Use lightweight models** for Spaces, link to full versions
5. **Enable sleep time** to control costs
6. **Test locally first** before deploying
7. **Monitor build logs** for issues
8. **Apply for community GPU grants** if building innovative demos

### For Production Use

1. **Use persistent storage** for model caching
2. **Implement error handling** in inference functions
3. **Add rate limiting** if expecting high traffic
4. **Set up secrets** for API keys and tokens
5. **Use environment variables** for configuration
6. **Enable analytics** to track usage
7. **Consider embedding** in external sites
8. **Monitor costs** via billing dashboard

### For VLM Testing

1. **Start with T4-small GPU** ($0.40/hour) for prototyping
2. **Use quantized models** (8-bit/4-bit) to fit memory
3. **Provide diverse test images** as examples
4. **Show inference time** to set user expectations
5. **Document model limitations** clearly
6. **Link to model card** on HuggingFace Hub
7. **Enable debug mode** during development
8. **Test with different image sizes/types**

## Sources

**Official Documentation:**
- [HuggingFace Spaces Overview](https://huggingface.co/docs/hub/spaces-overview) - Official Spaces documentation (accessed 2025-01-31)
- [HuggingFace GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus) - GPU hardware configuration guide (accessed 2025-01-31)
- [HuggingFace Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference) - Complete configuration reference (accessed 2025-01-31)
- [HuggingFace Spaces Embed](https://huggingface.co/docs/hub/spaces-embed) - Embedding Spaces documentation (accessed 2025-01-31)

**Tutorials:**
- [PyImageSearch: Deploy Gradio Apps on Hugging Face Spaces](https://pyimagesearch.com/2024/12/30/deploy-gradio-apps-on-hugging-face-spaces/) - Step-by-step deployment tutorial (accessed 2025-01-31)

**Community Resources:**
- [HuggingFace Forums: Debugging Spaces](https://discuss.huggingface.co/t/how-to-debug-spaces-on-hf-co/13191) - Debugging guidance (accessed 2025-01-31)
- [HuggingFace Forums: Runtime Logging](https://discuss.huggingface.co/t/spaces-runtime-logging/139383) - Logging best practices (accessed 2025-01-31)

**Example Spaces:**
- [PyImageSearch VQA Demo](https://huggingface.co/spaces/pyimagesearch/visual-question-answer-finetuned-paligemma) - Production VQA demo
- [PyImageSearch Document Understanding](https://huggingface.co/spaces/pyimagesearch/document-understanding-mix-paligemma) - Document analysis demo
- [PyImageSearch Image Captioning](https://huggingface.co/spaces/pyimagesearch/image-captioning-mix-paligemma) - Captioning demo
- [PyImageSearch Video Captioning](https://huggingface.co/spaces/pyimagesearch/video-captioning-mix-paligemma) - Video analysis demo
