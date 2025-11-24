# Vertex AI Model Garden & Foundation Models

**Comprehensive guide to pre-built models, fine-tuning foundation models, and cost optimization for production deployments**

## Overview

Vertex AI Model Garden is Google's centralized catalog of foundation models, offering 150+ pre-trained models for immediate deployment. Model Garden provides access to Google's proprietary models (Gemini, PaLM 2, Imagen), open-source models (Llama, Gemma, Mistral), and third-party models, with built-in fine-tuning, deployment, and cost optimization features.

**Key capabilities:**
- **Foundation model catalog**: Gemini Pro, PaLM 2, Imagen 3, Gemma 2, Llama 3.1, Claude 3.5
- **Fine-tuning options**: Supervised tuning, LoRA, full fine-tuning, RLHF
- **Deployment modes**: Online prediction, batch inference, serverless endpoints
- **Cost models**: Pay-per-token (hosted), self-deployed (GKE), committed use discounts
- **Quota management**: RPM limits, TPM throttling, burst capacity

From [Vertex AI Model Garden Overview](https://cloud.google.com/model-garden) (accessed 2025-11-16):
> "Model Garden on Vertex AI. Jumpstart your ML project with a single place to discover, customize, and deploy a wide variety of models from Google and Google partners."

---

## Section 1: Model Garden Catalog (~150 lines)

### Google Foundation Models

**Gemini Model Family (Multimodal):**

From [Model versions and lifecycle | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) (accessed 2025-11-16):

| Model | Context Window | Capabilities | Best For |
|-------|---------------|--------------|----------|
| **Gemini 2.5 Pro** | 1M tokens | Reasoning, coding, math, STEM | Complex analysis, agentic workflows |
| **Gemini 2.5 Flash** | 1M tokens | Low-latency, thinking, multimodal | High-volume production inference |
| **Gemini 2.0 Flash** | 1M tokens | Native image generation, multimodal | Real-time applications |
| **Gemini 1.5 Pro** | 2M tokens | Long-context, multimodal | Document analysis, video understanding |
| **Gemini 1.5 Flash** | 1M tokens | Cost-efficient, fast | Production workloads |

**PaLM 2 Models (Text-only):**
- **text-bison**: General text generation, classification, summarization
- **chat-bison**: Conversational AI, multi-turn dialogue
- **code-bison**: Code generation, completion, debugging
- **textembedding-gecko**: Text embeddings for similarity search

**Imagen Models (Image Generation):**
- **Imagen 3**: Latest high-quality image generation
- **Imagen 2**: Predecessor with broad creative capabilities
- **Gemini 2.5 Flash Image**: State-of-the-art generation and editing

From [Introducing Gemini 2.5 Flash Image](https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/) (accessed 2025-11-16):
> "Explore Gemini 2.5 Flash Image, a powerful new image generation and editing model with advanced features and creative control."

### Open-Source Models in Model Garden

**Meta Llama Family:**
- **Llama 3.1 405B**: Largest open model (405B parameters)
- **Llama 3.1 70B**: High-performance, production-ready
- **Llama 3.1 8B**: Efficient, edge-deployable
- **Llama 3.2 Vision**: Multimodal capabilities (11B, 90B)

**Google Gemma Family:**
- **Gemma 2 27B**: Efficient, high-quality text generation
- **Gemma 2 9B**: Lightweight, production-optimized
- **CodeGemma**: Code-specialized variant

**Other Open Models:**
- **Mistral 7B, Mixtral 8x7B**: Mixture-of-experts architecture
- **Falcon 40B, 180B**: Performance-optimized text models
- **Stable Diffusion XL**: Image generation
- **CLIP**: Vision-language embedding

### Accessing Model Garden

**Via Vertex AI Console:**
```bash
# Navigate to Model Garden in GCP Console
https://console.cloud.google.com/vertex-ai/model-garden

# Filter by:
# - Task type (text, vision, multimodal)
# - Publisher (Google, Meta, Hugging Face, etc.)
# - Deployment options (1-click deploy, fine-tunable)
```

**Via gcloud CLI:**
```bash
# List available models
gcloud ai models list \
  --region=us-central1 \
  --filter="labels.publisher=google"

# Get model details
gcloud ai models describe MODEL_ID \
  --region=us-central1
```

**Via Python SDK:**
```python
from google.cloud import aiplatform

aiplatform.init(project="PROJECT_ID", location="us-central1")

# List models in Model Garden
models = aiplatform.Model.list(
    filter='labels.model_garden_source=vertex-ai-model-garden'
)

for model in models:
    print(f"{model.display_name}: {model.description}")
```

### Model Deployment Options

**1-Click Deploy:**
- Pre-configured containers for instant deployment
- Autoscaling enabled by default
- Optimized for common use cases

**Customizable Deploy:**
- Modify machine types, GPU counts
- Adjust autoscaling parameters
- Custom serving containers

**Fine-Tune & Deploy:**
- Fine-tune on custom data
- Deploy fine-tuned model to endpoint
- A/B test against base model

From [Vertex AI Model Garden: All of your favorite LLMs in one place](https://medium.com/google-cloud/vertex-ai-model-garden-all-of-your-favorite-llms-in-one-place-a8940ea333c1) (accessed 2025-11-16):
> "Enter Vertex AI Model Garden, a library that helps you discover, test, customize, and deploy select proprietary and open models and assets."

---

## Section 2: Pre-Built Containers & Hosted Models (~140 lines)

### Gemini API on Vertex AI

**Serverless Deployment:**

Gemini models are available via serverless API - no infrastructure management required.

```python
from vertexai.preview.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-pro")

# Text generation
response = model.generate_content("Explain quantum computing")
print(response.text)

# Multimodal prompt (text + image)
import vertexai.preview.vision_models as vision_models

image = vision_models.Image.load_from_file("image.jpg")
response = model.generate_content(["Describe this image:", image])
print(response.text)

# Streaming response
for chunk in model.generate_content("Write a story", stream=True):
    print(chunk.text, end="")
```

**Gemini Vision API:**
```python
# Image understanding
model = GenerativeModel("gemini-1.5-pro-vision")

response = model.generate_content([
    "What objects are in this image?",
    vision_models.Image.load_from_file("scene.jpg")
])

# Video understanding (1M token context)
response = model.generate_content([
    "Summarize this video",
    vision_models.Video.load_from_file("gs://bucket/video.mp4")
])
```

From [Generate content with the Gemini API in Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference) (accessed 2025-11-16):
> "The Gemini model family includes models that work with multimodal prompt requests. The term multimodal indicates that you can use more than one modality, or type, such as combining text, images, and video."

### PaLM 2 API

**Text Generation (text-bison):**
```python
from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison@002")

response = model.predict(
    "Write a product description for a smartwatch",
    temperature=0.7,
    max_output_tokens=256,
    top_p=0.95
)
print(response.text)
```

**Chat API (chat-bison):**
```python
from vertexai.language_models import ChatModel

chat_model = ChatModel.from_pretrained("chat-bison@002")
chat = chat_model.start_chat()

# Multi-turn conversation
response = chat.send_message("What is machine learning?")
print(response.text)

response = chat.send_message("Give me an example")
print(response.text)

# Maintain conversation history
print(chat.message_history)
```

**Code Generation (code-bison):**
```python
from vertexai.language_models import CodeGenerationModel

code_model = CodeGenerationModel.from_pretrained("code-bison@002")

response = code_model.predict(
    prefix="def quicksort(arr):",
    max_output_tokens=512
)
print(response.text)
```

### Imagen API

**Image Generation:**
```python
from vertexai.preview.vision_models import ImageGenerationModel

model = ImageGenerationModel.from_pretrained("imagegeneration@006")

# Generate images
images = model.generate_images(
    prompt="A futuristic city with flying cars, cyberpunk style",
    number_of_images=4,
    aspect_ratio="16:9",
    safety_filter_level="block_some",
    person_generation="allow_all"
)

for i, image in enumerate(images):
    image.save(f"generated_{i}.png")
```

**Image Editing:**
```python
# Inpainting (edit specific regions)
base_image = vision_models.Image.load_from_file("scene.jpg")
mask_image = vision_models.Image.load_from_file("mask.png")

edited_images = model.edit_image(
    base_image=base_image,
    mask=mask_image,
    prompt="Replace with a modern building",
    edit_mode="inpainting-insert"
)

# Outpainting (extend image)
extended_images = model.edit_image(
    base_image=base_image,
    prompt="Extend the scene to show more sky",
    edit_mode="outpainting"
)
```

From [Generate and edit images with Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-generation) (accessed 2025-11-16):
> "The following sections cover how to generate images using either Vertex AI Studio or using the API. For guidance and best practices for prompting, see Design image generation prompts."

### Embeddings API

**Text Embeddings:**
```python
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Single text embedding
embeddings = model.get_embeddings(["Hello world"])
vector = embeddings[0].values  # 768-dim vector

# Batch embeddings
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = model.get_embeddings(texts)

# Task-specific embeddings
embeddings = model.get_embeddings(
    texts=["Query about machine learning"],
    task_type="RETRIEVAL_QUERY"  # vs RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY
)
```

---

## Section 3: Fine-Tuning Foundation Models (~180 lines)

### Supervised Fine-Tuning Overview

From [Introduction to tuning | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models) (accessed 2025-11-16):
> "Vertex AI supports supervised fine-tuning to customize foundational models. Supervised fine-tuning improves the performance of the model on specific tasks."

**Tuning Methods Available:**

| Method | Description | Memory | Training Time | Accuracy |
|--------|-------------|---------|---------------|----------|
| **Full Fine-Tuning** | Update all model weights | High (100% params) | Slow | Best |
| **LoRA** | Low-rank adapter matrices | Low (0.1-1% params) | Fast | Good |
| **Adapter Tuning** | Add trainable layers | Medium (1-5% params) | Medium | Good |
| **Prompt Tuning** | Learn soft prompts | Very Low (<0.01%) | Very Fast | Fair |

### LoRA Fine-Tuning (Low-Rank Adaptation)

**Concept:**
LoRA introduces low-rank decomposition matrices to transformer layers, dramatically reducing trainable parameters.

From [Fine-Tuning Large Language Models: How Vertex AI Takes LLMs to the Next Level](https://medium.com/google-cloud/fine-tuning-large-language-models-how-vertex-ai-takes-llms-to-the-next-level-3c113f4007da) (accessed 2025-11-16):
> "LoRA: LoRA is a technique for efficiently fine-tuning LLMs. It does this by introducing trainable, low-rank decomposition matrices into the model."

**LoRA Mathematics:**
```
Original weight matrix: W ∈ ℝ^(d×k)

LoRA decomposition:
W' = W + ΔW
ΔW = B × A

where:
- B ∈ ℝ^(d×r)  (low-rank matrix, r << d)
- A ∈ ℝ^(r×k)  (low-rank matrix)
- r = rank (typically 4-64)

Trainable parameters: (d + k) × r
vs Full fine-tuning: d × k

Example (Llama 70B):
- Full: 70B parameters
- LoRA (r=16): ~20M parameters (0.03% of original)
```

**Fine-Tuning Gemini with LoRA:**

```python
from vertexai.preview.tuning import sft

# Prepare training data (JSONL format)
training_data = {
    "examples": [
        {
            "input_text": "Summarize: Long document text...",
            "output_text": "Brief summary..."
        },
        # ... more examples
    ]
}

# Upload to GCS
training_data_uri = "gs://bucket/training_data.jsonl"

# Create tuning job
tuning_job = sft.train(
    source_model="gemini-1.5-pro-002",
    train_dataset=training_data_uri,
    tuning_task_addons=["LORA"],  # Use LoRA
    tuning_configs={
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "epochs": 3
    }
)

# Monitor job
print(f"Tuning job: {tuning_job.name}")
tuning_job.wait()

# Deploy tuned model
tuned_model = tuning_job.get_tuned_model()
endpoint = tuned_model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)
```

### Full Fine-Tuning

**When to use:**
- Maximum accuracy required
- Large training datasets (100k+ examples)
- Domain-specific vocabulary/knowledge
- Budget for GPU compute

**Open Model Fine-Tuning (Llama 3.1):**

From [Tune an open model | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning) (accessed 2025-11-16):
> "This page describes how to perform supervised fine-tuning on open models such as Llama 3.1. Supported tuning modes: Full fine-tuning, Low-Rank Adaptation (LoRA)."

```python
from google.cloud import aiplatform

# Full fine-tuning configuration
tuning_job = aiplatform.CustomTrainingJob(
    display_name="llama-3.1-70b-full-tuning",
    script_path="train.py",
    container_uri="gcr.io/project/llama-trainer:latest",
    requirements=["transformers", "torch", "deepspeed"],
    model_serving_container_image_uri="gcr.io/project/llama-serve:latest"
)

# Launch training
model = tuning_job.run(
    dataset=training_dataset,
    replica_count=4,  # Multi-GPU training
    machine_type="n1-highmem-96",
    accelerator_type="NVIDIA_A100",
    accelerator_count=8,  # 8× A100 per replica
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    args=[
        f"--model_name_or_path=meta-llama/Meta-Llama-3.1-70B",
        "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=32",
        "--learning_rate=2e-5",
        "--num_train_epochs=3",
        "--bf16",
        "--deepspeed=ds_config.json"
    ]
)
```

**Training Cost Estimate (Llama 3.1 70B):**
- Machine: n1-highmem-96 + 8× A100 = ~$40/hour
- Training time: 10-20 hours for 100k examples
- Total cost: $400-$800

### Fine-Tuning for Specific Tasks

**Summarization Fine-Tuning:**
```python
# Training data format
{
    "input_text": "context: Article about climate change...",
    "output_text": "summary: Climate change impacts global temperatures..."
}
```

**Question Answering:**
```python
{
    "input_text": "question: What is photosynthesis? context: Plants use...",
    "output_text": "answer: Photosynthesis is the process..."
}
```

**Code Generation:**
```python
{
    "input_text": "# Function to calculate fibonacci\ndef fibonacci(n):",
    "output_text": "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
}
```

### Evaluation & Validation

**Automatic Evaluation Metrics:**
```python
from vertexai.preview.evaluation import EvalTask

# Define evaluation task
eval_task = EvalTask(
    dataset=validation_dataset,
    metrics=["bleu", "rouge", "exact_match"],
    experiment="llama-tuning-v1"
)

# Run evaluation
results = eval_task.evaluate(
    model=tuned_model,
    experiment_run_name="run-1"
)

print(f"BLEU: {results.summary_metrics['bleu']}")
print(f"ROUGE-L: {results.summary_metrics['rouge_l']}")
```

**Human Evaluation:**
```python
from vertexai.preview import rlhf

# Collect human preferences
pairwise_task = rlhf.PairwiseChoiceTask(
    model_a=base_model,
    model_b=tuned_model,
    prompts=test_prompts
)

# Deploy for labelers
pairwise_task.deploy_labeling_task()
```

From [How LoRA Fine-Tunes Gemini Models on Vertex AI](https://www.linkedin.com/posts/walterwlee_lora-gemini-summarization-activity-7363748259342798848-RQOJ) (accessed 2025-11-16):
> "#LoRA (Low-Rank Adaptation) is a technique for fine-tuning large language models like Google's #Gemini on Vertex AI."

---

## Section 4: Deployment Options & Inference (~140 lines)

### Online Prediction (Real-Time Inference)

**Deploying to Endpoints:**

```python
from google.cloud import aiplatform

# Upload model to Model Registry
model = aiplatform.Model.upload(
    display_name="gemini-1.5-pro-tuned",
    artifact_uri="gs://bucket/tuned-model/",
    serving_container_image_uri="gcr.io/vertex-ai/prediction/gemini:latest"
)

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="gemini-tuned-endpoint"
)

# Deploy model
deployed_model = model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="gemini-v1",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100,
    sync=True
)

# Make predictions
prediction = endpoint.predict(
    instances=[{"content": "Summarize this text..."}]
)
print(prediction.predictions[0])
```

**Autoscaling Configuration:**
```python
# GPU autoscaling
deployed_model = model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=2,
    max_replica_count=20,
    target_utilization=0.7,  # Scale at 70% GPU util
    sync=True
)
```

### Batch Prediction

**Large-Scale Offline Inference:**

```python
from google.cloud import aiplatform

# Create batch prediction job
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name="gemini-batch-inference",
    model_name=model.resource_name,
    instances_format="jsonl",
    gcs_source="gs://bucket/input/prompts.jsonl",
    gcs_destination_prefix="gs://bucket/output/",
    machine_type="n1-standard-16",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=2,
    starting_replica_count=10,
    max_replica_count=50
)

# Monitor job
batch_prediction_job.wait()
print(f"Batch prediction complete: {batch_prediction_job.output_info}")
```

**Input format (prompts.jsonl):**
```json
{"content": "Summarize: Article 1..."}
{"content": "Summarize: Article 2..."}
{"content": "Summarize: Article 3..."}
```

**Cost optimization:**
- Batch prediction: 50% cheaper than online prediction
- Use preemptible VMs: Additional 60% savings
- Batch size: 1000-10000 requests per job for efficiency

### Serverless Inference (Gemini API)

**No infrastructure management:**

```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="PROJECT_ID", location="us-central1")

model = GenerativeModel("gemini-1.5-flash")

# Pay only for tokens used
response = model.generate_content("Write a poem about AI")
print(response.text)

# Cost: ~$0.00001 per 1k input tokens
```

**Serverless vs Dedicated Endpoints:**

| Feature | Serverless (Gemini API) | Dedicated Endpoint |
|---------|-------------------------|-------------------|
| **Setup** | Instant (no deploy) | 5-10 min deploy |
| **Cost** | Pay-per-token | Pay-per-hour + tokens |
| **Scaling** | Automatic (unlimited) | Manual (max replicas) |
| **Latency** | 200-500ms | 50-200ms (optimized) |
| **Best For** | Variable workloads | Predictable, high-volume |

### Private Endpoints

**VPC-SC Integration:**

```python
# Create private endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="gemini-private-endpoint",
    network="projects/PROJECT_ID/global/networks/VPC_NAME",
    enable_private_service_connect=True
)

# Deploy model to private endpoint
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    sync=True
)

# Access only from VPC
# No public internet exposure
```

From [karpathy/vertex-ai-production/01-inference-serving-optimization.md](../vertex-ai-production/01-inference-serving-optimization.md):
> "Private endpoint benefits: Traffic never leaves VPC network, No public IP exposure, VPC-SC perimeter enforcement, Lower latency (avoid internet routing)"

---

## Section 5: Quota Management & Rate Limits (~90 lines)

### Understanding Quotas

**Vertex AI Quotas (per project, per region):**

| Resource | Default Quota | Adjustable |
|----------|---------------|------------|
| **API Requests** | 60 req/min | Yes (up to 600) |
| **Tokens Per Minute (TPM)** | 1M TPM | Yes (up to 10M) |
| **Concurrent Requests** | 100 | Yes |
| **Online Prediction Nodes** | 50 nodes | Yes (up to 500) |
| **Batch Prediction Jobs** | 10 concurrent | Yes |
| **Training Jobs** | 5 concurrent | Yes |

**Gemini-Specific Quotas:**

From [Vertex AI Pricing](https://docs.cloud.google.com/vertex-ai/generative-ai/pricing) (accessed 2025-11-16):

```python
# Gemini 1.5 Pro quotas
# - Requests: 360 req/min (6 req/sec)
# - Input tokens: 4M TPM
# - Output tokens: 16k TPM

# Gemini 1.5 Flash quotas
# - Requests: 2000 req/min (33 req/sec)
# - Input tokens: 4M TPM
# - Output tokens: 16k TPM
```

### Monitoring Quota Usage

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{PROJECT_ID}"

# Query quota metrics
query = client.query_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="serviceruntime.googleapis.com/quota/rate/net_usage"',
        "interval": {
            "end_time": {"seconds": int(time.time())},
            "start_time": {"seconds": int(time.time() - 3600)}
        }
    }
)

for result in query:
    print(f"Quota usage: {result.points[0].value.double_value}")
```

### Rate Limit Error Handling

```python
import time
from google.api_core import retry, exceptions

@retry.Retry(
    predicate=retry.if_exception_type(
        exceptions.ResourceExhausted,  # 429 Too Many Requests
        exceptions.ServiceUnavailable   # 503 Service Unavailable
    ),
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=300.0
)
def generate_with_retry(prompt):
    model = GenerativeModel("gemini-1.5-pro")
    return model.generate_content(prompt)

# Usage
try:
    response = generate_with_retry("Write a story")
except exceptions.ResourceExhausted as e:
    print(f"Quota exceeded: {e}")
    # Implement exponential backoff
```

### Requesting Quota Increases

```bash
# Via gcloud CLI
gcloud alpha services quota update \
  --consumer=projects/PROJECT_ID \
  --service=aiplatform.googleapis.com \
  --metric=aiplatform.googleapis.com/online_prediction_requests_per_base_model \
  --value=1000 \
  --unit=1/min/{region}

# Or via GCP Console → IAM & Admin → Quotas
```

**Best practices:**
- Request increases 2-3 days before needed
- Provide justification (expected load, use case)
- Start with 2× current usage, scale incrementally
- Monitor actual usage to avoid waste

---

## Section 6: Cost Analysis & Optimization (~100 lines)

### Vertex AI Hosted Pricing (2024)

**Gemini Models (Pay-per-Token):**

From [Vertex AI Pricing](https://docs.cloud.google.com/vertex-ai/generative-ai/pricing) (accessed 2025-11-16):

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|-------|----------------------|------------------------|----------------|
| **Gemini 1.5 Pro** | $1.25 | $5.00 | 2M tokens |
| **Gemini 1.5 Flash** | $0.075 | $0.30 | 1M tokens |
| **Gemini 2.5 Pro** | $2.50 | $10.00 | 1M tokens |
| **Gemini 2.5 Flash** | $0.15 | $0.60 | 1M tokens |

**PaLM 2 Models:**
| Model | Input (per 1M chars) | Output (per 1M chars) |
|-------|---------------------|----------------------|
| **text-bison** | $0.50 | $0.50 |
| **chat-bison** | $0.50 | $0.50 |
| **code-bison** | $0.50 | $0.50 |

**Imagen Models:**
| Operation | Cost per Image |
|-----------|----------------|
| **Image Generation** | $0.020 |
| **Image Editing** | $0.030 |
| **Upscaling** | $0.010 |

### Self-Deployed Models on GKE

**Infrastructure Costs:**

From [Google Vertex AI vs. Open-Source Models on GKE](https://medium.com/@darkmatter4real/fine-tuning-llms-google-vertex-ai-vs-open-source-models-on-gke-53830c2c0ef3) (accessed 2025-11-16):
> "Cost Control: Preemptible VMs + custom GPU drivers reduce costs by 30–50% vs Vertex."

**GKE Deployment Costs (Llama 3.1 70B):**

```python
# Infrastructure: 2× A100 80GB GPUs
# Machine type: a2-highgpu-2g

# Costs (us-central1):
# - On-demand: $14.68/hour
# - Preemptible: $5.87/hour (60% savings)
# - 1-year commit: $10.28/hour (30% savings)
# - 3-year commit: $7.34/hour (50% savings)

# Monthly costs (24/7 operation):
# - On-demand: $10,570/month
# - Preemptible: $4,228/month
# - 3-year commit: $5,285/month
```

**Self-Hosted vs Vertex AI Comparison:**

| Workload | Vertex AI Hosted | Self-Hosted (GKE) | Savings |
|----------|------------------|-------------------|---------|
| **1M tokens/day** (Gemini 1.5 Flash) | $45/day | $175/day (24/7 A100) | Vertex wins |
| **10M tokens/day** | $450/day | $175/day | GKE wins (61%) |
| **100M tokens/day** | $4,500/day | $350/day (2× A100) | GKE wins (92%) |

**Break-even analysis:**
- Low volume (<5M tokens/day): Use Vertex AI hosted
- High volume (>10M tokens/day): Deploy on GKE
- Preemptible workloads: GKE preemptible VMs (60% cheaper)

### Cost Optimization Strategies

**1. Model Selection:**
```python
# Use Flash models for high-volume workloads
# Gemini 1.5 Flash: 16× cheaper than Pro for similar tasks

# Example: Customer support chatbot
# - 1M conversations/day
# - Avg 500 tokens per conversation
# - Total: 500M tokens/day

# Gemini Pro: $500M × $1.25/1M = $625/day
# Gemini Flash: $500M × $0.075/1M = $37.50/day
# Savings: $587.50/day ($17,625/month)
```

**2. Caching Strategies:**
```python
from vertexai.preview import caching

# Cache system prompts (Gemini 1.5 Pro)
# Regular: $1.25 per 1M input tokens
# Cached: $0.3125 per 1M tokens (75% cheaper)

cached_content = caching.CachedContent.create(
    model_name="gemini-1.5-pro-002",
    system_instruction="You are a helpful assistant...",  # Long prompt
    ttl=3600  # Cache for 1 hour
)

model = GenerativeModel.from_cached_content(cached_content)

# Subsequent requests use cached prompt
for query in user_queries:
    response = model.generate_content(query)  # Only pay for query tokens
```

**3. Batch Processing:**
```python
# Batch prediction: 50% cheaper than online
# Example: 10M predictions/day

# Online: $4,500/day
# Batch: $2,250/day
# Savings: $2,250/day
```

**4. Committed Use Discounts:**
```bash
# 1-year commit: 25% discount
# 3-year commit: 52% discount

# Example: $10,000/month workload
# Pay-as-you-go: $120,000/year
# 1-year commit: $90,000/year (save $30k)
# 3-year commit: $57,600/year (save $62k)
```

From [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-11-16):
> "The costs for Vertex AI remain the same as they are for the legacy AI Platform and AutoML products that Vertex AI supersedes."

---

## Section 7: ARR-COC Integration with Gemini Vision API (~100 lines)

### Gemini Vision for Relevance Scoring

ARR-COC can leverage Gemini Vision API for high-level scene understanding before texture-based relevance realization:

**Architecture:**
```
Image Input
    ↓
[Gemini Vision API] → Scene description, object detection
    ↓
[ARR-COC Texture Extraction] → 13-channel arrays
    ↓
[Relevance Scorers] → Use Gemini output as additional context
    ↓
[Opponent Processing] → LOD allocation
    ↓
[Qwen3-VL Decoder] → Final generation
```

**Implementation:**

```python
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.vision_models as vision_models

# Stage 1: Gemini Vision for scene understanding
gemini_vision = GenerativeModel("gemini-1.5-pro-vision")

image = vision_models.Image.load_from_file("scene.jpg")

# Get high-level scene description
scene_prompt = """Analyze this image and provide:
1. Main objects and their locations
2. Scene type (indoor/outdoor, setting)
3. Important visual features
4. Suggested areas of focus for detailed analysis
"""

scene_analysis = gemini_vision.generate_content([scene_prompt, image])
scene_context = scene_analysis.text

# Stage 2: ARR-COC texture extraction
from arr_coc.knowing import TextureExtractor

extractor = TextureExtractor()
texture_array = extractor.extract(image, patches=196)  # 14×14 grid

# Stage 3: Query-aware relevance with Gemini context
query = "What animals are in this image?"
query_with_context = f"{query}\n\nScene context: {scene_context}"

# Stage 4: Relevance scoring (use Gemini context)
from arr_coc.knowing import PropositionScorer, PerspectivalScorer, ParticipatoryScorer

prop_scorer = PropositionScorer()
prop_scores = prop_scorer.score(texture_array)

persp_scorer = PerspectivalScorer()
persp_scores = persp_scorer.score(texture_array)

parti_scorer = ParticipatoryScorer()
parti_scores = parti_scorer.score(texture_array, query_with_context)

# Stage 5: Opponent processing → LOD allocation
from arr_coc.balancing import TensionBalancer
from arr_coc.attending import LODAllocator

balancer = TensionBalancer()
balanced_scores = balancer.balance(prop_scores, persp_scores, parti_scores)

allocator = LODAllocator(total_budget=5000)  # 5k tokens for 196 patches
lod_allocations = allocator.allocate(balanced_scores)  # 64-400 tokens per patch

# Stage 6: Compress textures based on LOD
compressed_features = []
for i, (texture, lod) in enumerate(zip(texture_array, lod_allocations)):
    compressed = compress_texture(texture, target_tokens=lod)
    compressed_features.append(compressed)

# Stage 7: Final VLM decoding
from arr_coc.model import QwenVLMDecoder

decoder = QwenVLMDecoder()
response = decoder.generate(
    visual_features=compressed_features,
    text_prompt=query,
    scene_context=scene_context  # Include Gemini analysis
)

print(response)
```

### Cost Analysis for ARR-COC + Gemini

**Hybrid Pipeline Costs:**

```python
# Assumptions:
# - 1000 images/day
# - Avg 1024×1024 resolution
# - Gemini Vision: Scene analysis (200 input tokens, 300 output tokens)
# - ARR-COC: Texture extraction + relevance scoring (local)
# - Qwen3-VL: Final generation (self-hosted)

# Daily costs:
# Gemini Vision API:
# - 1000 images × 200 input tokens = 200k tokens
# - 1000 images × 300 output tokens = 300k tokens
# - Input cost: 200k × $1.25/1M = $0.25
# - Output cost: 300k × $5.00/1M = $1.50
# - Total Gemini: $1.75/day

# ARR-COC processing (self-hosted):
# - GKE cluster: 1× A100 = $7.34/hour = $176/day
# - Processes 1000 images in ~2 hours
# - Allocated cost: $14.68/day

# Total pipeline cost: $16.43/day
# Cost per image: $0.016
```

**Benefits of Hybrid Approach:**
- Gemini provides high-quality scene understanding ($0.00175/image)
- ARR-COC focuses compute on relevant regions (saves 60-80% tokens)
- Self-hosted Qwen3-VL for final generation (predictable costs)
- Total: 3-5× cheaper than pure Gemini Vision for complex queries

### Production Deployment Pattern

```python
# Deploy Gemini Vision (serverless) + ARR-COC (GKE)

# 1. Gemini Vision Endpoint (managed)
gemini_model = GenerativeModel("gemini-1.5-flash-vision")  # Use Flash for cost

# 2. ARR-COC on GKE (self-hosted)
# Deploy arr-coc-triton container to GKE
# Use node pools with A100 GPUs

# 3. Orchestrate via Cloud Run
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.post("/analyze")
async def analyze_image(image_url: str, query: str):
    # Stage 1: Gemini Vision (async)
    scene_task = asyncio.create_task(
        get_scene_analysis(image_url)
    )

    # Stage 2: ARR-COC processing (async)
    texture_task = asyncio.create_task(
        extract_textures(image_url)
    )

    # Wait for both
    scene_context, textures = await asyncio.gather(scene_task, texture_task)

    # Stage 3: Relevance scoring + LOD allocation
    relevance_scores = score_relevance(textures, query, scene_context)
    lod_allocations = allocate_lod(relevance_scores)

    # Stage 4: Final generation
    response = generate_response(textures, lod_allocations, query, scene_context)

    return {"response": response, "cost": estimate_cost()}
```

From [karpathy/inference-optimization/00-tensorrt-fundamentals.md](../inference-optimization/00-tensorrt-fundamentals.md):
> "TensorRT achieves 5-40× speedups over CPU-only platforms and 2-10× speedups over naive GPU implementations through graph optimization, kernel fusion, precision calibration, and hardware-specific tuning."

---

## Sources

**Google Cloud Documentation:**
- [Vertex AI Platform](https://cloud.google.com/vertex-ai) - Vertex AI overview (accessed 2025-11-16)
- [Model Garden on Vertex AI](https://cloud.google.com/model-garden) - Model catalog and deployment (accessed 2025-11-16)
- [Model versions and lifecycle | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) - Gemini model versions (accessed 2025-11-16)
- [Introduction to tuning | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models) - Fine-tuning overview (accessed 2025-11-16)
- [Tune an open model | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning) - Llama fine-tuning (accessed 2025-11-16)
- [Vertex AI Pricing](https://docs.cloud.google.com/vertex-ai/generative-ai/pricing) - Pricing documentation (accessed 2025-11-16)
- [Generate and edit images with Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-generation) - Imagen API (accessed 2025-11-16)
- [Generate content with the Gemini API in Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference) - Gemini inference (accessed 2025-11-16)

**Blog Posts & Articles:**
- [Vertex AI Model Garden: All of your favorite LLMs in one place](https://medium.com/google-cloud/vertex-ai-model-garden-all-of-your-favorite-llms-in-one-place-a8940ea333c1) by Nikita Namjoshi - Model Garden overview (accessed 2025-11-16)
- [Fine-Tuning Large Language Models: How Vertex AI Takes LLMs to the Next Level](https://medium.com/google-cloud/fine-tuning-large-language-models-how-vertex-ai-takes-llms-to-the-next-level-3c113f4007da) by Abirami Sukumaran - LoRA fine-tuning (accessed 2025-11-16)
- [Introducing Gemini 2.5 Flash Image](https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/) - Google Developers Blog - Image generation (accessed 2025-11-16)
- [Introducing Gemini 2.0: our new AI model for the agentic era](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/) - Google Blog - Gemini 2.0 announcement (accessed 2025-11-16)
- [How LoRA Fine-Tunes Gemini Models on Vertex AI](https://www.linkedin.com/posts/walterwlee_lora-gemini-summarization-activity-7363748259342798848-RQOJ) by Walter Lee - LoRA on Vertex AI (accessed 2025-11-16)
- [Google Vertex AI vs. Open-Source Models on GKE](https://medium.com/@darkmatter4real/fine-tuning-llms-google-vertex-ai-vs-open-source-models-on-gke-53830c2c0ef3) by DarkMatter - Cost comparison (accessed 2025-11-16)

**Web Research:**
- Google Search: "Vertex AI Model Garden Gemini PaLM 2024 2025" (accessed 2025-11-16)
- Google Search: "foundation model fine-tuning Vertex AI LoRA" (accessed 2025-11-16)
- Google Search: "Imagen Gemini API Vertex AI multimodal 2024" (accessed 2025-11-16)
- Google Search: "Vertex AI hosted vs self-hosted cost pricing 2024" (accessed 2025-11-16)

**Related Knowledge (Internal References):**
- [karpathy/inference-optimization/00-tensorrt-fundamentals.md](../inference-optimization/00-tensorrt-fundamentals.md) - TensorRT optimization for serving
- [karpathy/vertex-ai-production/01-inference-serving-optimization.md](../vertex-ai-production/01-inference-serving-optimization.md) - Production deployment patterns

---

**Created**: 2025-11-16
**Lines**: ~700
**Purpose**: PART 21 of Vertex AI Advanced Integration expansion - Comprehensive guide to Model Garden, foundation models, fine-tuning strategies, deployment options, quota management, and cost optimization for production ML deployments.
