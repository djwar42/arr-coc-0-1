# HuggingFace Multi-Cloud Integration: Vertex AI, SageMaker, Azure ML, and Ray Train

**Comprehensive guide to deploying HuggingFace models across cloud platforms with unified workflows**

**Created**: 2025-11-16

---

## Overview

HuggingFace Transformers integrate seamlessly with major cloud ML platforms, enabling teams to leverage platform-specific capabilities while maintaining consistent model development workflows. This guide covers production deployment patterns across Google Cloud (Vertex AI), AWS (SageMaker), Azure (Azure ML), and distributed training frameworks (Ray Train).

**Key Integration Value:**
- **Unified API**: Train locally with Transformers, deploy anywhere
- **Platform optimization**: Leverage cloud-specific accelerators (TPUs, Trainium, etc.)
- **Cost flexibility**: Choose optimal compute/cost balance per workload
- **Vendor independence**: Avoid lock-in with portable model artifacts

From [HuggingFace + Google Cloud Partnership](https://huggingface.co/blog/google-cloud) (November 2025, accessed 2025-11-16):
> "We want to see a future where every company can build their own AI with open models and host it within their own secure infrastructure, with full control. Our deep collaboration will accelerate this vision, whether you are using Vertex AI Model Garden, Google Kubernetes Engine, Cloud Run or Hugging Face Inference Endpoints."

---

## Section 1: Vertex AI Integration (~150 lines)

### 1.1 Vertex AI Model Garden Deployment

From [Google Cloud Developer Forums - HuggingFace Agent on Vertex AI](https://discuss.google.dev/t/from-smol-to-scaled-deploying-hugging-face-s-agent-on-vertex-ai/181268) (February 2025, accessed 2025-11-16):
> "This blog post walks you through the entire process of building and deploying an agent to Vertex AI, from defining your HuggingFace's smolagent deploying and scaling your agent."

**HuggingFace + Vertex AI Partnership (2025):**

From [HuggingFace Blog - Google Cloud Partnership](https://huggingface.co/blog/google-cloud) (accessed 2025-11-16):
- **CDN Gateway**: Direct caching of HuggingFace models on Google Cloud infrastructure
- **10x growth**: Usage increased 10x over 3 years (tens of petabytes/month)
- **Model Garden integration**: Click-to-deploy for popular open models
- **TPU support**: Native integration for Google's AI accelerators

**Model Garden Deployment Pattern:**

```python
from google.cloud import aiplatform
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Vertex AI
aiplatform.init(
    project='your-project-id',
    location='us-central1',
    staging_bucket='gs://your-bucket'
)

# Deploy HuggingFace model to Vertex AI Endpoint
# Method 1: Pre-built containers (Model Garden)
endpoint = aiplatform.Endpoint.create(
    display_name='hf-t5-endpoint'
)

# Upload model from HuggingFace Hub
model = aiplatform.Model.upload(
    display_name='t5-small-finetuned',
    artifact_uri='gs://your-bucket/model',  # Optional: upload custom fine-tuned model
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest',
    serving_container_environment_variables={
        'MODEL_ID': 't5-small',  # HuggingFace Hub model ID
        'TASK': 'text2text-generation'
    }
)

# Deploy to endpoint
model.deploy(
    endpoint=endpoint,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=3
)

# Predict
instances = [{"inputs": "translate English to French: Hello world"}]
predictions = endpoint.predict(instances=instances)
```

### 1.2 Custom Training Jobs with HuggingFace Trainer

From [vertex-ai-production/02-ray-distributed-integration.md](../karpathy/vertex-ai-production/02-ray-distributed-integration.md):
> "Ray on Vertex AI is a fully managed service that provides scalability for AI and Python applications using Ray. It simplifies distributed computing by eliminating the need to become a DevOps engineer."

**Custom Training Job Pattern:**

```python
from google.cloud import aiplatform

# Define training job with HuggingFace
job = aiplatform.CustomTrainingJob(
    display_name='hf-bert-training',
    container_uri='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest',
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest',
)

# Run training (HuggingFace Trainer script)
model = job.run(
    args=[
        '--model_name_or_path', 'bert-base-uncased',
        '--task_name', 'mrpc',
        '--do_train',
        '--do_eval',
        '--max_seq_length', '128',
        '--per_device_train_batch_size', '32',
        '--learning_rate', '2e-5',
        '--num_train_epochs', '3',
        '--output_dir', '/gcs/your-bucket/output'
    ],
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=2,
    base_output_dir='gs://your-bucket/model-output'
)
```

### 1.3 GKE Deployment for Custom Infrastructure

**HuggingFace on GKE Pattern:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: huggingface-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hf-inference
  template:
    metadata:
      labels:
        app: hf-inference
    spec:
      containers:
      - name: inference
        image: huggingface/transformers-pytorch-gpu:latest
        env:
        - name: MODEL_ID
          value: "distilbert-base-uncased-finetuned-sst-2-english"
        - name: TASK
          value: "text-classification"
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8080
```

**CDN Gateway Benefits (2025 Partnership):**

From [HuggingFace + Google Cloud](https://huggingface.co/blog/google-cloud):
> "This CDN Gateway will cache Hugging Face models and datasets directly on Google Cloud to significantly reduce downloading times, and strengthen model supply chain robustness for Google Cloud customers."

- Cached models on Google Cloud infrastructure
- Reduced time-to-first-token
- Improved model governance
- Works across Vertex, GKE, Cloud Run, Compute Engine

---

## Section 2: AWS SageMaker Integration (~150 lines)

### 2.1 SageMaker Training with HuggingFace Estimator

From [HuggingFace SageMaker Docs](https://huggingface.co/docs/sagemaker/en/getting-started) (accessed 2025-11-16):
> "The get started guide will show you how to quickly use Hugging Face on Amazon SageMaker. Learn how to fine-tune and deploy a pretrained 🤗 Transformers model on SageMaker for a binary text classification task."

**SageMaker Estimator Pattern:**

```python
from sagemaker.huggingface import HuggingFace

# Define hyperparameters
hyperparameters = {
    'epochs': 3,
    'train_batch_size': 32,
    'model_name': 'distilbert-base-uncased',
    'learning_rate': 2e-5
}

# Create HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point='train.py',              # Training script
    source_dir='./scripts',
    instance_type='ml.p3.2xlarge',       # 1x V100 GPU
    instance_count=1,
    role=sagemaker_role,                 # IAM role
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    hyperparameters=hyperparameters
)

# Start training
huggingface_estimator.fit({
    'train': 's3://your-bucket/train',
    'test': 's3://your-bucket/test'
})
```

### 2.2 Distributed Training with SageMaker Model Parallelism

From [aws-sagemaker/00-distributed-inference-optimization.md](../karpathy/aws-sagemaker/00-distributed-inference-optimization.md):
> "Amazon SageMaker model parallel library now accelerates PyTorch FSDP workloads by up to 20% through optimized NCCL communication and custom memory management."

**FSDP on SageMaker:**

```python
from sagemaker.huggingface import HuggingFace

# Multi-node FSDP training
estimator = HuggingFace(
    entry_point='train.py',
    instance_type='ml.p4d.24xlarge',     # 8x A100 GPUs
    instance_count=4,                    # 32 GPUs total
    role=role,
    transformers_version='4.26',
    pytorch_version='2.0',
    py_version='py310',
    distribution={
        'torch_distributed': {
            'enabled': True
        },
        'smdistributed': {
            'modelparallel': {
                'enabled': True,
                'parameters': {
                    'sharding_strategy': 'FULL_SHARD',      # FSDP
                    'backward_prefetch': 'BACKWARD_PRE',
                    'forward_prefetch': True,
                    'cpu_offload': False
                }
            }
        }
    },
    hyperparameters={
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4
    }
)

estimator.fit('s3://your-bucket/data')
```

### 2.3 SageMaker Inference Endpoints

**Real-time Inference Deployment:**

```python
# Deploy to SageMaker endpoint
predictor = huggingface_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',      # 1x T4 GPU
    endpoint_name='hf-sentiment-analysis'
)

# Predict
sentiment_input = {
    "inputs": "This movie was absolutely fantastic!"
}
result = predictor.predict(sentiment_input)

# Multi-model endpoint (cost optimization)
from sagemaker.multidatamodel import MultiDataModel

mme = MultiDataModel(
    name='hf-multi-model-endpoint',
    model_data_prefix='s3://your-bucket/models/',
    image_uri=predictor.image_uri,
    role=role
)

mme_predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge'
)

# Invoke specific model
mme_predictor.predict(
    data=sentiment_input,
    target_model='sentiment-model-v1.tar.gz'
)
```

### 2.4 SageMaker + Spot Instances (Cost Optimization)

```python
# Use Spot instances for training (up to 90% cost savings)
huggingface_estimator = HuggingFace(
    entry_point='train.py',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    use_spot_instances=True,             # Enable Spot
    max_wait=7200,                       # Max wait time (seconds)
    max_run=3600,                        # Max training time
    checkpoint_s3_uri='s3://your-bucket/checkpoints',  # For resuming
    hyperparameters=hyperparameters
)
```

---

## Section 3: Azure ML Integration (~100 lines)

### 3.1 Azure ML Training with HuggingFace

From [azure-ml/00-distributed-training-aks-serving.md](../karpathy/azure-ml/00-distributed-training-aks-serving.md):
> "Azure ML automatically sets distributed training environment variables: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK, NODE_RANK"

**Azure ML Job Configuration:**

```python
from azure.ai.ml import command, Input
from azure.ai.ml.entities import ResourceConfiguration

# Define training job
job = command(
    code="./src",
    command="python train.py --model_name ${{inputs.model_name}} --epochs ${{inputs.epochs}}",
    inputs={
        "model_name": "bert-base-uncased",
        "epochs": 3,
        "learning_rate": 2e-5,
        "train_batch_size": 32
    },
    environment="azureml:AzureML-acpt-pytorch-2.2-cuda12.1@latest",
    instance_count=2,                    # 2 nodes
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 4  # 4 GPUs per node
    },
    resources=ResourceConfiguration(
        instance_type="STANDARD_NC24RS_V3",  # 4x V100, InfiniBand
        instance_count=2
    )
)

# Submit job
ml_client.jobs.create_or_update(job)
```

### 3.2 Azure ML Model Catalog Deployment

From [HuggingFace Forums - Azure ML Model Catalog](https://discuss.huggingface.co/t/about-the-azure-ml-studio-model-catalog-category/40677) (May 2023, accessed 2025-11-16):
> "This category is to ask questions about deploying Hugging Face Hub models for real-time inference in Azure Machine Learning using the new Hugging Face model catalog integration."

**Model Catalog Deployment Pattern:**

```python
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment

# Register HuggingFace model
model = Model(
    path="azureml://registries/HuggingFace/models/bert-base-uncased",
    name="hf-bert-sentiment",
    version="1"
)

registered_model = ml_client.models.create_or_update(model)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="hf-bert-endpoint",
    description="HuggingFace BERT sentiment analysis"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy model
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="hf-bert-endpoint",
    model=registered_model,
    instance_type="Standard_NC6s_v3",    # 1x V100
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment).result()
```

### 3.3 AKS Deployment with Triton Inference Server

From [azure-ml/00-distributed-training-aks-serving.md](../karpathy/azure-ml/00-distributed-training-aks-serving.md):
> "Triton Inference Server integration for high-performance model serving"

**Triton on AKS Pattern:**

```python
from azure.ai.ml.entities import AksCompute, KubernetesOnlineDeployment

# Deploy to AKS with Triton
deployment = KubernetesOnlineDeployment(
    name="triton-hf-deployment",
    endpoint_name="hf-inference",
    model=registered_model,
    compute="aks-gpu-cluster",
    resources={
        "requests": {"nvidia.com/gpu": 1},
        "limits": {"nvidia.com/gpu": 1}
    },
    environment_variables={
        "TRITON_MODEL_REPOSITORY": "/models",
        "HF_MODEL_ID": "distilbert-base-uncased-finetuned-sst-2-english"
    }
)
```

---

## Section 4: Ray Train Integration (~100 lines)

### 4.1 Ray Train with HuggingFace Transformers

From [orchestration/02-ray-distributed-ml.md](../karpathy/orchestration/02-ray-distributed-ml.md):
> "Ray is an open-source unified framework for scaling AI and Python applications from your laptop to clusters across any cloud. Unlike Dask (which focuses on distributed data processing), Ray is designed specifically for compute-intensive ML workloads with strong support for distributed training, hyperparameter tuning, and model serving."

**Ray Train HuggingFace Integration:**

```python
from ray.train.huggingface import TransformersTrainer
from ray.train import ScalingConfig
import transformers

def train_func(config):
    """Training function compatible with Ray Train"""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Prepare Ray datasets
    train_dataset = ray.train.get_dataset_shard("train")
    eval_dataset = ray.train.get_dataset_shard("eval")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/output",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

# Configure Ray Train
trainer = TransformersTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={
        "model_name": "distilbert-base-uncased",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5
    },
    scaling_config=ScalingConfig(
        num_workers=4,               # 4 distributed workers
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    ),
    datasets={
        "train": ray_train_dataset,
        "eval": ray_eval_dataset
    }
)

# Run distributed training
result = trainer.fit()
```

### 4.2 Ray on Vertex AI (Managed)

From [vertex-ai-production/02-ray-distributed-integration.md](../karpathy/vertex-ai-production/02-ray-distributed-integration.md):
> "Ray on Vertex AI is a fully managed service that provides scalability for AI and Python applications using Ray."

**Ray Cluster on Vertex AI:**

```python
import vertex_ray
from vertex_ray import Resources
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project='your-project',
    location='us-central1',
    staging_bucket='gs://your-bucket'
)

# Create Ray cluster
ray_cluster = vertex_ray.create_ray_cluster(
    head_node_type=Resources(
        machine_type="n1-standard-8",
        node_count=1
    ),
    worker_node_types=[Resources(
        machine_type="a2-highgpu-1g",    # A100 GPUs
        node_count=4,
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=2
    )],
    python_version='3_10',
    ray_version='2_9',
    cluster_name='hf-training-cluster'
)

# Submit Ray job to cluster
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient(ray_cluster.dashboard_address)

job_id = client.submit_job(
    entrypoint="python train_hf_model.py",
    runtime_env={
        "pip": ["transformers", "datasets", "ray[train]"]
    }
)
```

---

## Section 5: Multi-Cloud Model Registry (~80 lines)

### 5.1 HuggingFace Hub as Universal Registry

**Hub as Central Model Repository:**

From [HuggingFace + Google Cloud](https://huggingface.co/blog/google-cloud):
> "With this new strategic partnership, we're making it easy to do on Google Cloud."

```python
from huggingface_hub import HfApi, create_repo, upload_folder

# Train on any platform, push to Hub
api = HfApi()

# Create private repo for organization
repo_id = create_repo(
    "your-org/bert-finetuned-custom",
    private=True,
    repo_type="model"
)

# Upload model artifacts
upload_folder(
    folder_path="./model_output",
    repo_id=repo_id,
    commit_message="Training run on Vertex AI - 2025-11-16"
)

# Download to different platform
from transformers import AutoModel

# Deploy same model on SageMaker, Vertex AI, or Azure ML
model = AutoModel.from_pretrained("your-org/bert-finetuned-custom")
```

### 5.2 Cross-Platform Deployment Pattern

**Unified Workflow:**

```python
# Step 1: Train on Platform A (e.g., Vertex AI)
# Step 2: Push to HuggingFace Hub
# Step 3: Deploy on Platform B (e.g., SageMaker)

from sagemaker.huggingface import HuggingFaceModel

# Deploy HuggingFace Hub model to SageMaker
huggingface_model = HuggingFaceModel(
    model_data='s3://your-bucket/model.tar.gz',  # Optional: download from Hub
    env={
        'HF_MODEL_ID': 'your-org/bert-finetuned-custom',  # Direct Hub reference
        'HF_TASK': 'text-classification'
    },
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39'
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge'
)
```

---

## Section 6: Cross-Platform Deployment Strategies (~80 lines)

### 6.1 Multi-Region Strategy

**Geographic Distribution Pattern:**

```python
# Deploy same model to multiple regions/clouds
deployments = {
    'us-east': {
        'platform': 'sagemaker',
        'region': 'us-east-1',
        'instance': 'ml.g4dn.xlarge'
    },
    'europe': {
        'platform': 'vertex-ai',
        'region': 'europe-west4',
        'machine': 'n1-standard-4-gpu'
    },
    'asia': {
        'platform': 'azure-ml',
        'region': 'southeastasia',
        'instance': 'Standard_NC6s_v3'
    }
}

# Deploy to all regions
for region, config in deployments.items():
    if config['platform'] == 'sagemaker':
        deploy_to_sagemaker(region, config)
    elif config['platform'] == 'vertex-ai':
        deploy_to_vertex(region, config)
    elif config['platform'] == 'azure-ml':
        deploy_to_azure(region, config)
```

### 6.2 Cost Optimization Across Platforms

**Platform-Specific Cost Strategies:**

| Platform | GPU Instance | Spot/Preemptible | Hourly Cost | Best For |
|----------|-------------|------------------|-------------|----------|
| **SageMaker** | ml.g4dn.xlarge (T4) | Spot: 90% savings | ~$0.15 (Spot) | Batch inference |
| **Vertex AI** | n1-standard-4 + T4 | Preemptible: 80% savings | ~$0.12 (Preempt) | Training jobs |
| **Azure ML** | Standard_NC6s_v3 (V100) | Spot: 80% savings | ~$0.20 (Spot) | High-throughput |
| **Ray on GKE** | n1-standard-4 + T4 | Preemptible nodes | ~$0.10 (Preempt) | Distributed training |

**Cost-Aware Deployment:**

```python
def get_cheapest_platform(workload_type, region):
    """Select platform based on cost and availability"""

    if workload_type == 'training' and region == 'us':
        # Vertex AI Spot often cheapest for training
        return 'vertex-ai'
    elif workload_type == 'inference' and region == 'eu':
        # Azure ML competitive in EU
        return 'azure-ml'
    elif workload_type == 'batch':
        # SageMaker Spot excellent for batch
        return 'sagemaker'
    else:
        return 'huggingface-endpoints'  # Simplest option

platform = get_cheapest_platform('training', 'us')
```

### 6.3 Failover and High Availability

**Multi-Cloud HA Pattern:**

```python
# Primary: Vertex AI (us-central1)
# Failover: SageMaker (us-east-1)

def predict_with_failover(text):
    """Try primary endpoint, failover to secondary"""
    try:
        # Try Vertex AI
        return vertex_endpoint.predict({"inputs": text})
    except Exception as e:
        print(f"Vertex AI failed: {e}, failing over to SageMaker")
        # Failover to SageMaker
        return sagemaker_predictor.predict({"inputs": text})

# Health check routing
def route_to_healthy_endpoint(text):
    """Route to healthy endpoint with lowest latency"""
    endpoints = [
        ('vertex-ai', vertex_endpoint, check_vertex_health),
        ('sagemaker', sagemaker_predictor, check_sagemaker_health),
        ('azure-ml', azure_endpoint, check_azure_health)
    ]

    for name, endpoint, health_check in endpoints:
        if health_check():
            return endpoint.predict({"inputs": text})

    raise Exception("All endpoints unhealthy")
```

---

## Section 7: Cost Comparison (~70 lines)

### 7.1 HuggingFace Endpoints vs Cloud Providers

**Monthly Cost Comparison (1 GPU inference endpoint, 24/7):**

| Service | Instance Type | Monthly Cost | Features |
|---------|--------------|--------------|----------|
| **HF Endpoints** | 1x A10G GPU | ~$1,000/mo | Fully managed, auto-scaling |
| **Vertex AI** | n1-standard-4 + T4 | ~$300/mo | Google Cloud integration |
| **SageMaker** | ml.g4dn.xlarge | ~$450/mo | AWS ecosystem |
| **Azure ML** | Standard_NC6s_v3 | ~$550/mo | Azure integration |
| **Cloud Run (Vertex)** | GPU serverless | Pay-per-use | Scales to zero |

**Training Cost Comparison (Fine-tuning BERT-base, 3 epochs, GLUE):**

| Platform | Setup | Instance Type | Training Time | Total Cost |
|----------|-------|--------------|---------------|------------|
| **Vertex AI Spot** | Managed | n1-standard-8 + V100 (preempt) | ~2 hours | ~$0.50 |
| **SageMaker Spot** | Managed | ml.p3.2xlarge (spot) | ~2 hours | ~$0.60 |
| **Azure ML Spot** | Managed | NC6s_v3 (spot) | ~2 hours | ~$0.70 |
| **Ray on GKE** | Self-managed | n1-standard-4 + T4 (preempt) | ~3 hours | ~$0.30 |
| **Local Workstation** | Self-managed | 1x RTX 3090 | ~4 hours | $0 (electricity) |

### 7.2 Cost Optimization Strategies

**Best Practices:**

```python
# 1. Use Spot/Preemptible instances for training
# 2. Scale to zero for inference (serverless)
# 3. Multi-model endpoints for low-traffic models
# 4. Cache models locally (reduce egress)

# Example: SageMaker Spot + S3 caching
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point='train.py',
    instance_type='ml.p3.2xlarge',
    use_spot_instances=True,          # 90% cost reduction
    max_wait=7200,
    checkpoint_s3_uri='s3://bucket/checkpoints',  # Resume on interruption
    hyperparameters={
        'model_name': 'bert-base-uncased',
        'cache_dir': '/opt/ml/input/data/cache'  # Cache transformers models
    }
)
```

**Multi-Cloud Cost Arbitrage:**

```python
def select_training_platform(model_size_gb, deadline_hours):
    """Select cheapest platform meeting deadline"""

    if model_size_gb < 1 and deadline_hours > 12:
        # Small model, flexible deadline → cheapest spot
        return 'ray-on-gke-spot'  # $0.10/hr
    elif model_size_gb > 10 and deadline_hours < 4:
        # Large model, urgent → premium instances
        return 'vertex-ai-a100'   # $2.50/hr
    else:
        # Medium model, normal deadline → balanced
        return 'sagemaker-spot'   # $0.50/hr
```

---

## Section 8: arr-coc-0-1 Multi-Cloud Strategy (~70 lines)

### 8.1 Current Architecture

**arr-coc-0-1 Multi-Cloud Pattern:**

From [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/):
- **Training**: Vertex AI Custom Jobs (GPU clusters)
- **Inference Demo**: HuggingFace Spaces (Gradio app)
- **Model Registry**: HuggingFace Hub (private repo)
- **Orchestration**: W&B Launch + Vertex AI agents

**Hub → Vertex AI → Spaces Flow:**

```python
# Step 1: Train on Vertex AI
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name='arr-coc-training',
    container_uri='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest'
)

model = job.run(
    args=[
        '--model_name', 'arr-coc-vlm',
        '--num_patches', '200',
        '--lod_min', '64',
        '--lod_max', '400',
        '--texture_channels', '13'
    ],
    replica_count=1,
    machine_type='n1-standard-16',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=4,
    base_output_dir='gs://arr-coc-bucket/models'
)

# Step 2: Push to HuggingFace Hub
from huggingface_hub import HfApi, upload_folder

api = HfApi()
upload_folder(
    folder_path='gs://arr-coc-bucket/models/arr-coc-vlm',
    repo_id='NorthHead/arr-coc-vlm',
    commit_message='Training run on Vertex AI'
)

# Step 3: Deploy to Spaces (Gradio demo)
# Spaces automatically pulls from Hub
# app.py loads model: AutoModel.from_pretrained('NorthHead/arr-coc-vlm')
```

### 8.2 Multi-Cloud Expansion Strategy

**Future Platform Integration:**

```python
# arr-coc-0-1 multi-cloud deployment matrix
deployment_matrix = {
    'development': {
        'platform': 'local',
        'resources': '1x RTX 3090',
        'use_case': 'Rapid prototyping'
    },
    'training': {
        'platform': 'vertex-ai',
        'resources': '4x V100 (spot)',
        'use_case': 'Distributed training'
    },
    'demo': {
        'platform': 'huggingface-spaces',
        'resources': 'A10G GPU',
        'use_case': 'Public inference demo'
    },
    'production-us': {
        'platform': 'sagemaker',
        'resources': 'ml.g4dn.xlarge (multi-model)',
        'use_case': 'US production inference'
    },
    'production-eu': {
        'platform': 'azure-ml',
        'resources': 'Standard_NC6s_v3',
        'use_case': 'EU production inference'
    }
}
```

### 8.3 Model Governance Across Clouds

**HuggingFace Hub as Source of Truth:**

```python
from huggingface_hub import HfApi
from datetime import datetime

# Tag production-ready models
api = HfApi()

# Development snapshot
api.create_tag(
    repo_id='NorthHead/arr-coc-vlm',
    tag='dev-2025-11-16',
    revision='main',
    tag_message='Development snapshot - vertex training run #42'
)

# Production release
api.create_tag(
    repo_id='NorthHead/arr-coc-vlm',
    tag='v0.1-production',
    revision='abc123def',
    tag_message='Production release - deployed to SageMaker + Azure ML'
)

# Deploy specific version to platforms
def deploy_to_all_clouds(model_tag='v0.1-production'):
    """Deploy same model version to all platforms"""

    # SageMaker
    deploy_sagemaker(
        model_id=f'NorthHead/arr-coc-vlm',
        revision=model_tag
    )

    # Vertex AI
    deploy_vertex(
        model_id=f'NorthHead/arr-coc-vlm',
        revision=model_tag
    )

    # Azure ML
    deploy_azure(
        model_id=f'NorthHead/arr-coc-vlm',
        revision=model_tag
    )
```

**CDN Gateway Benefits for arr-coc-0-1:**

From [HuggingFace + Google Cloud Partnership](https://huggingface.co/blog/google-cloud):
- Faster model downloads on Vertex AI (cached on Google Cloud)
- Reduced time-to-first-token for inference
- Improved supply chain robustness
- Works across all arr-coc-0-1 deployments on Google Cloud

---

## Sources

**Official Documentation:**
- [HuggingFace + Google Cloud Partnership Announcement](https://huggingface.co/blog/google-cloud) (November 2025, accessed 2025-11-16)
- [HuggingFace SageMaker Documentation](https://huggingface.co/docs/sagemaker/en/getting-started) (accessed 2025-11-16)
- [Google Cloud Vertex AI - HuggingFace Models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hugging-face-models) (accessed 2025-11-16)
- [HuggingFace Forums - Azure ML Model Catalog](https://discuss.huggingface.co/t/about-the-azure-ml-studio-model-catalog-category/40677) (May 2023)
- [Google Developer Forums - HuggingFace Agent on Vertex AI](https://discuss.google.dev/t/from-smol-to-scaled-deploying-hugging-face-s-agent-on-vertex-ai/181268) (February 2025)

**Existing Knowledge Base:**
- [karpathy/vertex-ai-production/02-ray-distributed-integration.md](../karpathy/vertex-ai-production/02-ray-distributed-integration.md) - Ray on Vertex AI
- [karpathy/aws-sagemaker/00-distributed-inference-optimization.md](../karpathy/aws-sagemaker/00-distributed-inference-optimization.md) - SageMaker distributed training
- [karpathy/azure-ml/00-distributed-training-aks-serving.md](../karpathy/azure-ml/00-distributed-training-aks-serving.md) - Azure ML integration
- [karpathy/orchestration/02-ray-distributed-ml.md](../karpathy/orchestration/02-ray-distributed-ml.md) - Ray Train fundamentals

**arr-coc-0-1 Reference:**
- [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/) - Multi-cloud deployment example

**Web Research:**
- Google Search: "HuggingFace Vertex AI integration 2024 2025" (accessed 2025-11-16)
- Google Search: "HuggingFace SageMaker training jobs deployment 2024" (accessed 2025-11-16)
- Google Search: "Azure ML HuggingFace transformers integration 2024" (accessed 2025-11-16)
- Google Search: "Ray Train HuggingFace integration distributed training 2024" (accessed 2025-11-16)
