# Ray Distributed ML Integration with Vertex AI and GKE

**Complete guide to running Ray on Google Cloud Platform for distributed ML workloads**

**Created**: 2025-11-14

---

## Overview

Ray provides a unified framework for distributed machine learning that integrates seamlessly with Google Cloud Platform through two main deployment options: **Ray on Vertex AI** (managed service) and **Ray on GKE** (Kubernetes-based). This guide covers deployment patterns, configuration, autoscaling, and production best practices for both approaches.

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md):
> Ray is an open-source unified framework for scaling AI and Python applications from your laptop to clusters across any cloud. Unlike Dask (which focuses on distributed data processing), Ray is designed specifically for compute-intensive ML workloads with strong support for distributed training, hyperparameter tuning, and model serving.

**Key Value Propositions for Google Cloud**:
- **Python-native**: Parallelize existing code with minimal changes
- **Flexible**: General-purpose distributed computing beyond data processing
- **ML-first**: Built-in libraries (Ray Train, Ray Tune, Ray Serve, Ray Data)
- **Production-ready**: Auto-scaling, fault tolerance, cloud integration

---

## Section 1: Ray on Vertex AI - Managed Service (~150 lines)

### What is Ray on Vertex AI

From [Ray on Vertex AI | Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/ray-on-vertex-ai) (accessed 2025-11-14):
> Ray on Vertex AI is a fully managed service that provides scalability for AI and Python applications using Ray. It simplifies distributed computing by eliminating the need to become a DevOps engineer.

**Announced**: Google Cloud Next 2023, Generally Available May 2024

**Architecture Benefits**:
- No manual infrastructure setup (compute, storage, network, security)
- Seamless integration with Vertex AI platform (Feature Store, Model Registry, Pipelines)
- Access to Google Cloud services (BigQuery, Cloud Storage)
- Built-in Ray OSS Dashboard for monitoring
- Simplified cluster provisioning via Vertex AI Python SDK

From [Medium: Scale AI on Ray on Vertex AI](https://medium.com/google-cloud/ray-on-vertex-ai-lets-get-it-started-7a9f8360ea25) by Ivan Nardini (accessed 2025-11-14):
> Ray on Vertex AI is a simpler way to get started with Ray for running your ML distributed computing workloads without needing to become a DevOps engineer. The integration also provides access to well established MLOps components of the Vertex AI platform.

### Creating Ray Clusters on Vertex AI

**Basic Cluster Provisioning**:

```python
import vertex_ray
from vertex_ray import Resources
from google.cloud import aiplatform as vertex_ai

# Initialize Vertex AI
vertex_ai.init(
    project=project_id,
    location=region,
    staging_bucket=bucket_uri
)

# Define head node resources
head_node_type = Resources(
    machine_type="n1-standard-8",
    node_count=1
)

# Define worker node resources (GPU-enabled)
worker_node_types = [Resources(
    machine_type="a2-highgpu-1g",
    node_count=2,
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=2,
)]

# Create Ray cluster
ray_cluster_resource_name = vertex_ray.create_ray_cluster(
    head_node_type=head_node_type,
    worker_node_types=worker_node_types,
    python_version='3_10',
    ray_version='2_9',  # Currently supports Ray 2.9.3
    cluster_name=your_cluster_name,
    network=your_network_name,
)
```

**Cluster Management**:

```python
# List all Ray clusters
ray_clusters = vertex_ray.list_ray_clusters()

# Get specific cluster information
ray_cluster_resource_name = ray_clusters[0].cluster_resource_name
ray_cluster = vertex_ray.get_ray_cluster(ray_cluster_resource_name)

# Check cluster resources
print(ray_cluster.cluster_resources())
# Output:
# {'CPU': xxx,
#  'GPU': xxx,
#  'accelerator_type:A100': xxx,
#  'object_store_memory': xxx}

# Delete cluster when done
vertex_ray.delete_ray_cluster(ray_cluster_resource_name)
```

### Autoscaling on Vertex AI

From [Vertex AI Documentation: Scale Ray clusters](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/scale-clusters) (accessed 2025-11-14):

Ray clusters on Vertex AI offer **two scaling options**:

**1. Autoscaling (Recommended for Dynamic Workloads)**:

```python
# Configure autoscaling worker pool
worker_node_types = [Resources(
    machine_type="a2-highgpu-1g",
    node_count=2,  # Initial count
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    # Autoscaling configuration
    min_replica_count=1,   # Minimum workers
    max_replica_count=10,  # Maximum workers
)]

ray_cluster = vertex_ray.create_ray_cluster(
    head_node_type=head_node_type,
    worker_node_types=worker_node_types,
    cluster_name="autoscaling-cluster"
)
```

**How Autoscaling Works**:
- Ray monitors resource demands from tasks and actors
- Automatically adds workers when tasks are pending
- Removes idle workers after configurable timeout
- Respects min/max replica bounds
- Scales independently per worker pool

**2. Manual Scaling (For Predictable Workloads)**:

```python
# Update worker pool to fixed size
vertex_ray.update_ray_cluster(
    cluster_resource_name=ray_cluster_resource_name,
    worker_node_types=[Resources(
        machine_type="n1-highmem-32",
        node_count=8,  # Fixed to 8 workers
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=4
    )]
)
```

### Submitting Jobs to Ray on Vertex AI

**Development Workflow** from [Medium: Ray on Vertex AI](https://medium.com/google-cloud/ray-on-vertex-ai-lets-get-it-started-7a9f8360ea25):

```python
# train.py - Your ML application
import ray
from ray.runtime_env import RuntimeEnv
from ray.air.config import RunConfig, ScalingConfig
from ray.train.xgboost import XGBoostTrainer

# Initialize Ray with runtime environment
runtime_env = RuntimeEnv(
    pip={"packages": "requirements.txt"}
)
ray.init(runtime_env=runtime_env)

# Configure distributed training
scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=True if 'GPU' in ray.cluster_resources().keys() else False,
    resources_per_worker={
        "CPU": 4,
        "GPU": 1
    }
)

# Run config with checkpointing
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(num_to_keep=None),
    sync_config=SyncConfig(
        upload_dir="gs://your-bucket/checkpoints"
    ),
    name=experiment_run_id,
)

# Create trainer
trainer = XGBoostTrainer(
    scaling_config=scaling_config,
    run_config=run_config,
    label_column="target",
    params=xgboost_config,
    datasets={"train": train_dataset, "valid": valid_dataset},
)

# Execute training
result = trainer.fit()
```

**Job Submission** (from local IDE or notebook):

```python
from ray.job_submission import JobSubmissionClient, JobStatus
import time

# Connect to Vertex AI Ray cluster
ray_cluster_address = ray_cluster.dashboard_address
client = JobSubmissionClient(address=ray_cluster_address)

# Submit job
job_id = client.submit_job(
    submission_id=submission_id,
    entrypoint="python3 train.py",
    runtime_env={"working_dir": working_dir}
)

# Monitor job status
while True:
    job_status = client.get_job_status(job_id)
    if job_status == JobStatus.SUCCEEDED:
        print("Job succeeded!")
        break
    elif job_status == JobStatus.FAILED:
        print("Job failed!")
        logs = client.get_job_logs(job_id)
        print(logs)
        break
    else:
        print("Job is running...")
        time.sleep(10)
```

### Integration with Ray Train

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md) (Ray Train section):

```python
# Distributed PyTorch training on Vertex AI
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    """Training function runs on each worker"""
    model = nn.Linear(10, 1)
    model = train.torch.prepare_model(model)  # Ray handles DDP setup

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = train.get_dataset_shard("train")  # Ray shards data

    for epoch in range(10):
        for batch in dataset.iter_batches(batch_size=32):
            X, y = batch["X"], batch["y"]

            optimizer.zero_grad()
            output = model(X)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()

        # Ray aggregates metrics across workers
        train.report({"loss": loss.item(), "epoch": epoch})

# Scale to 4 GPUs on Vertex AI
scaling_config = ScalingConfig(
    num_workers=4,
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 4}
)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    datasets={"train": ray_dataset}
)

result = trainer.fit()
print(f"Final metrics: {result.metrics}")
```

**Key Advantages for Vertex AI**:
- Automatic DistributedDataParallel setup (no manual `init_process_group`)
- Built-in data sharding across workers
- Metric aggregation and reporting
- Checkpoint management with Cloud Storage integration
- Worker failure recovery

---

## Section 2: Ray on GKE - Kubernetes Deployment (~200 lines)

### KubeRay Operator Overview

From [Ray on GKE: New features for AI scheduling and scaling](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-gke-new-features-for-ai-scheduling-and-scaling) (accessed 2025-11-14):
> Ray is an OSS compute engine that is popular among Google Cloud developers to handle complex distributed AI workloads across CPUs, GPUs, and TPUs. The Ray Operator Add-On in a GKE cluster is hosted by Google and does not run on GKE nodes, meaning no overhead is added to the cluster.

**KubeRay Architecture**:
- **Ray Operator**: Manages Ray cluster lifecycle via Kubernetes CRDs
- **RayCluster**: Custom resource defining cluster specification
- **Head Node**: Single head pod (dashboard, scheduler, driver)
- **Worker Nodes**: Multiple worker pods (scalable, can use GPUs/TPUs)
- **Autoscaling**: KubeRay handles pod scaling based on Ray demands

From [Medium: Running Ray on Google Cloud](https://medium.com/zencore/running-ray-on-google-cloud-e40b369fabfe) by Shaun Keenan (accessed 2025-11-14):
> Ray on Vertex AI is user-friendly and quick to start with, while KubeRay offers greater scalability and control for larger, more complex deployments.

### Deploying Ray on GKE

**Step 1: Enable Ray Operator on GKE**:

```bash
# Create GKE cluster with Ray Operator
gcloud container clusters create ray-cluster \
    --location=us-central1 \
    --machine-type=n1-standard-8 \
    --num-nodes=3 \
    --addons=RayOperator \
    --cluster-version=latest \
    --enable-autoscaling \
    --min-nodes=3 \
    --max-nodes=10

# Or enable on existing cluster
gcloud container clusters update ray-cluster \
    --location=us-central1 \
    --update-addons=RayOperator=ENABLED
```

**Step 2: Create GPU Node Pool**:

```bash
# Add GPU-enabled node pool for Ray workers
gcloud container node-pools create gpu-workers \
    --cluster=ray-cluster \
    --location=us-central1 \
    --machine-type=n1-standard-16 \
    --accelerator=type=nvidia-tesla-a100,count=2 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=8
```

**Step 3: Deploy RayCluster**:

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ml-training-cluster
  namespace: default
spec:
  rayVersion: '2.9.0'

  # Head node configuration
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: '0.0.0.0'
      num-cpus: '0'  # Don't schedule tasks on head
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0-py310
          resources:
            limits:
              cpu: "4"
              memory: "16Gi"
            requests:
              cpu: "2"
              memory: "8Gi"
          ports:
          - containerPort: 6379  # Redis
          - containerPort: 8265  # Dashboard
          - containerPort: 10001 # Client server

  # Worker node configuration
  workerGroupSpecs:
  - replicas: 4
    minReplicas: 1
    maxReplicas: 10
    groupName: gpu-workers
    rayStartParams:
      num-cpus: "8"
      num-gpus: "2"
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0-py310-gpu
          resources:
            limits:
              nvidia.com/gpu: 2
              cpu: "16"
              memory: "64Gi"
            requests:
              nvidia.com/gpu: 2
              cpu: "8"
              memory: "32Gi"
        nodeSelector:
          cloud.google.com/gke-nodepool: gpu-workers
```

**Deploy the cluster**:

```bash
kubectl apply -f ray-cluster.yaml

# Verify deployment
kubectl get rayclusters
kubectl get pods -l ray.io/cluster=ml-training-cluster

# Access Ray Dashboard
kubectl port-forward service/ml-training-cluster-head-svc 8265:8265
# Open http://localhost:8265
```

### Autoscaling on GKE

**Ray-Level Autoscaling** (managed by Ray Autoscaler):

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md):
- Ray monitors pending tasks and actor resource requirements
- Requests additional worker pods from KubeRay when needed
- KubeRay creates/destroys pods based on Ray demands
- Respects `minReplicas` and `maxReplicas` in worker group spec

**Cluster-Level Autoscaling** (GKE Cluster Autoscaler):

```yaml
# Worker group with aggressive autoscaling
workerGroupSpecs:
- replicas: 2
  minReplicas: 0   # Scale to zero when idle
  maxReplicas: 20  # Scale up to 20 workers
  groupName: scalable-gpu-workers
  rayStartParams:
    num-gpus: "1"
  scaleStrategy:
    workersToDelete: []
  template:
    spec:
      containers:
      - name: ray-worker
        image: rayproject/ray:2.9.0-py310-gpu
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: "16Gi"
```

**Combined Autoscaling Flow**:
1. Ray detects pending tasks requiring GPUs
2. Ray Autoscaler requests KubeRay to add worker pods
3. KubeRay creates pod(s) matching worker group spec
4. If no nodes available, GKE Cluster Autoscaler provisions new node
5. Pod scheduled on new node, joins Ray cluster
6. When idle, Ray Autoscaler removes workers → KubeRay deletes pods
7. GKE Cluster Autoscaler removes idle nodes

From [Google Cloud Blog: Ray on GKE Features](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-gke-new-features-for-ai-scheduling-and-scaling) (accessed 2025-11-14):
> November 4, 2025 announcement: Ray on GKE includes new features for AI scheduling and scaling, with optimized support for GPU workloads and integration with Anyscale Platform and Runtime.

### Running Distributed Training on GKE

**Ray Train Example on GKE**:

```python
# Connect to Ray cluster on GKE
import ray

# Option 1: From inside cluster (using service)
ray.init(address="ray://ml-training-cluster-head-svc:10001")

# Option 2: Port-forward from local machine
ray.init(address="ray://localhost:10001")

# Distributed PyTorch training
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_function(config):
    import torch
    import torch.nn as nn
    from ray import train

    # Model setup
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model = train.torch.prepare_model(model)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters())
    dataset = train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        for batch in dataset.iter_batches(batch_size=64):
            # Training step
            optimizer.zero_grad()
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()

        train.report({"loss": loss.item()})

# Scale to 8 GPUs (4 workers × 2 GPUs each)
scaling_config = ScalingConfig(
    num_workers=4,
    use_gpu=True,
    resources_per_worker={"GPU": 2, "CPU": 8}
)

trainer = TorchTrainer(
    train_loop_per_worker=train_function,
    scaling_config=scaling_config,
    train_loop_config={"epochs": 10}
)

result = trainer.fit()
```

### Integration with Google Cloud Services

**Cloud Storage Integration**:

```python
# Configure Ray to use GCS for checkpoints
from ray.train import RunConfig, CheckpointConfig

run_config = RunConfig(
    storage_path="gs://my-bucket/ray-checkpoints",
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,
        checkpoint_frequency=5
    )
)

trainer = TorchTrainer(
    train_loop_per_worker=train_function,
    scaling_config=scaling_config,
    run_config=run_config
)
```

**Workload Identity** (recommended for production):

```yaml
# Service account for Ray pods
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ray-sa
  annotations:
    iam.gke.io/gcp-service-account: ray-training@PROJECT_ID.iam.gserviceaccount.com

---
# Update RayCluster to use service account
spec:
  headGroupSpec:
    template:
      spec:
        serviceAccountName: ray-sa
        # ...
  workerGroupSpecs:
  - template:
      spec:
        serviceAccountName: ray-sa
        # ...
```

**Grant GCS permissions**:

```bash
# Create GCP service account
gcloud iam service-accounts create ray-training

# Bind to Kubernetes service account
gcloud iam service-accounts add-iam-policy-binding \
    ray-training@PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:PROJECT_ID.svc.id.goog[default/ray-sa]"

# Grant GCS access
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ray-training@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

---

## Section 3: Ray Serve for Model Deployment on GCP (~150 lines)

### Ray Serve Overview

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md) (Ray Serve section):
> Ray Serve is Ray's framework for scalable model deployment and serving. It's framework-agnostic (deploy any Python code), supports autoscaling, model composition (ensemble), and production features like zero-downtime updates and request batching.

### Deploying Ray Serve on Vertex AI

**Basic Model Serving**:

```python
import ray
from ray import serve

# Initialize Ray (connect to Vertex AI cluster)
ray.init(address="vertex_ray://your-cluster-resource-name")

# Start Ray Serve
serve.start()

# Define deployment
@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 2, "num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5
    }
)
class VLMPredictor:
    def __init__(self):
        import torch
        # Load model on GPU
        self.model = load_vlm_model()
        self.model = self.model.cuda()
        self.model.eval()

    async def __call__(self, request):
        image = decode_image(request.query_params["image"])
        query = request.query_params["query"]

        with torch.no_grad():
            result = self.model.predict(image, query)

        return {"answer": result}

# Deploy to Vertex AI Ray cluster
serve.run(VLMPredictor.bind())
```

**Autoscaling Behavior**:
- Monitors `target_num_ongoing_requests_per_replica` (default: 1.0)
- Scales up if: `ongoing_requests / num_replicas > target`
- Scales down if: `ongoing_requests / num_replicas < target`
- Respects `min_replicas` and `max_replicas` bounds

From [Ray Docs: Autoscaling Guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html) (accessed 2025-11-14):
> Instead of setting a fixed number of replicas for a deployment and manually updating it, you can configure a deployment to autoscale based on incoming traffic.

### Ray Serve on GKE

**Deployment Pattern**:

```yaml
# ray-serve-deployment.yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vlm-serving
spec:
  serviceUnhealthySecondThreshold: 900
  deploymentUnhealthySecondThreshold: 300

  # Ray Serve application
  serveConfigV2: |
    applications:
    - name: vlm_app
      import_path: serve_app:deployment
      runtime_env:
        working_dir: "gs://my-bucket/serve-app/"
        pip:
          - torch
          - transformers
          - pillow

      deployments:
      - name: VLMPredictor
        num_replicas: 2
        ray_actor_options:
          num_cpus: 2
          num_gpus: 1
        autoscaling_config:
          min_replicas: 1
          max_replicas: 10
          target_num_ongoing_requests_per_replica: 5

  # Underlying Ray cluster
  rayClusterConfig:
    rayVersion: '2.9.0'
    headGroupSpec:
      serviceType: ClusterIP
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-py310
            resources:
              limits:
                cpu: "4"
                memory: "16Gi"

    workerGroupSpecs:
    - replicas: 2
      minReplicas: 1
      maxReplicas: 8
      groupName: serve-workers
      rayStartParams: {}
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.9.0-py310-gpu
            resources:
              limits:
                nvidia.com/gpu: 1
                cpu: "4"
                memory: "16Gi"
```

**Expose Service**:

```yaml
# Ingress for external access
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ray-serve-ingress
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "ray-serve-ip"
spec:
  rules:
  - http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: vlm-serving-serve-svc
            port:
              number: 8000
```

### Request Batching for GPU Efficiency

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md):

```python
@serve.deployment(max_concurrent_queries=100)
class BatchedVLMPredictor:
    def __init__(self):
        self.model = load_vlm_model().cuda()
        self.model.eval()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def handle_batch(self, images_and_queries):
        """Process multiple requests together for GPU efficiency"""
        images = [item["image"] for item in images_and_queries]
        queries = [item["query"] for item in images_and_queries]

        with torch.no_grad():
            # Batch inference on GPU
            predictions = self.model.predict_batch(images, queries)

        return predictions

    async def __call__(self, request):
        image = decode_image(request.query_params["image"])
        query = request.query_params["query"]

        result = await self.handle_batch({"image": image, "query": query})
        return {"answer": result}

serve.run(BatchedVLMPredictor.bind())
```

**Batching Benefits**:
- Higher GPU throughput (process 32 images simultaneously)
- Reduced per-request latency (amortized overhead)
- Better hardware utilization (keeps GPU busy)
- Configurable `max_batch_size` and `batch_wait_timeout_s`

---

## Section 4: Ray AIR - End-to-End ML Workflows (~100 lines)

### Ray AIR on Google Cloud

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md) (Ray AIR section):
> Ray AIR (AI Runtime) unifies Ray Train, Ray Tune, Ray Serve, and Ray Data into a single, cohesive ML platform for end-to-end workflows.

**Complete Workflow Example**:

```python
import ray
from ray import train, tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.tune import Tuner
from ray import serve

# Initialize Ray on Vertex AI
ray.init(address="vertex_ray://your-cluster")

# Step 1: Load data with Ray Data (from Cloud Storage)
dataset = ray.data.read_parquet("gs://my-bucket/training-data/")

# Step 2: Distributed training function
def train_func(config):
    model = build_model(config["lr"], config["hidden_dim"])
    model = train.torch.prepare_model(model)

    dataset = train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        loss = train_epoch(model, dataset)
        train.report({"loss": loss})

    # Save checkpoint to GCS
    checkpoint = train.save_checkpoint(model)
    train.report(checkpoint=checkpoint)

# Step 3: Hyperparameter tuning with Ray Tune
tuner = Tuner(
    TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=4,
            use_gpu=True
        ),
        datasets={"train": dataset},
        run_config=RunConfig(
            storage_path="gs://my-bucket/ray-results",
            checkpoint_config=CheckpointConfig(num_to_keep=3)
        )
    ),
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-4, 1e-1),
            "hidden_dim": tune.choice([128, 256, 512]),
            "epochs": 20
        }
    },
    tune_config=tune.TuneConfig(
        num_samples=10,
        scheduler=tune.schedulers.ASHAScheduler()
    )
)

results = tuner.fit()
best_result = results.get_best_result(metric="loss", mode="min")
best_checkpoint = best_result.checkpoint

# Step 4: Deploy best model with Ray Serve
@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 16,
        "target_num_ongoing_requests_per_replica": 10
    }
)
class BestModelPredictor:
    def __init__(self, checkpoint):
        self.model = load_from_checkpoint(checkpoint)
        self.model.eval()

    async def __call__(self, request):
        return self.model.predict(request.query_params["input"])

# Deploy to production
serve.run(BestModelPredictor.bind(best_checkpoint))
```

**Benefits of Unified Workflow**:
- **Single API**: Same abstractions across data loading, training, tuning, serving
- **Seamless transitions**: Checkpoints flow directly from training → tuning → serving
- **Resource efficiency**: Share compute resources across pipeline stages
- **Cloud Storage integration**: All artifacts stored in GCS automatically

---

## Section 5: Production Best Practices (~100 lines)

### Monitoring and Observability

**Ray Dashboard Integration**:

```python
# Vertex AI: Dashboard available in Ray cluster details
# GKE: Port-forward to access dashboard
kubectl port-forward service/ray-cluster-head-svc 8265:8265
# Open http://localhost:8265

# Dashboard shows:
# - Live cluster metrics (CPU, GPU, memory, network)
# - Task/actor visualization
# - Resource allocation
# - Job logs and profiling
```

**Cloud Monitoring Integration**:

```python
from google.cloud import monitoring_v3
import time

class RayMetricsPublisher:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"

    def publish_ray_metric(self, metric_type, value, labels=None):
        """Publish Ray metrics to Cloud Monitoring"""
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/ray/{metric_type}"

        if labels:
            for key, val in labels.items():
                series.metric.labels[key] = val

        series.resource.type = "generic_task"
        series.resource.labels["project_id"] = self.project_id
        series.resource.labels["location"] = "us-central1"
        series.resource.labels["namespace"] = "ray-training"

        now = time.time()
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": int(now), "nanos": int((now - int(now)) * 10**9)}}
        )

        point = monitoring_v3.Point({
            "interval": interval,
            "value": {"double_value": value}
        })

        series.points = [point]
        self.client.create_time_series(name=self.project_name, time_series=[series])

# Usage in training
metrics = RayMetricsPublisher(project_id="my-project")
metrics.publish_ray_metric(
    "training/loss",
    value=current_loss,
    labels={"job_id": job_id, "epoch": str(epoch)}
)
```

### Cost Optimization Strategies

**Preemptible/Spot Instances on GKE**:

```yaml
# Spot node pool for cost savings (60-91% discount)
gcloud container node-pools create spot-gpu-workers \
    --cluster=ray-cluster \
    --location=us-central1 \
    --spot \
    --machine-type=n1-standard-16 \
    --accelerator=type=nvidia-tesla-v100,count=4 \
    --num-nodes=0 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=10

# RayCluster tolerating spot instances
workerGroupSpecs:
- replicas: 4
  groupName: spot-workers
  template:
    spec:
      tolerations:
      - key: cloud.google.com/gke-spot
        operator: Equal
        value: "true"
        effect: NoSchedule
      nodeSelector:
        cloud.google.com/gke-spot: "true"
      # ...
```

**Autoscaling to Zero**:

```yaml
# Vertex AI: Set min_replica_count=0
worker_node_types = [Resources(
    machine_type="a2-highgpu-1g",
    node_count=0,
    min_replica_count=0,  # Scale to zero when idle
    max_replica_count=10,
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=2
)]

# GKE: Set minReplicas=0
workerGroupSpecs:
- minReplicas: 0  # Scale to zero
  maxReplicas: 10
  groupName: scalable-workers
  # ...
```

**Cost Comparison** (8x A100 GPUs):

| Configuration | Cost/Hour | 100-Hour Job | Savings |
|---------------|-----------|--------------|---------|
| Vertex AI On-Demand | $29.39 | $2,939 | - |
| Vertex AI with Autoscale (50% idle) | $14.70 | $1,470 | 50% |
| GKE Spot Instances | $7.06 | $706 | 76% |
| GKE Spot + Scale-to-Zero (50% idle) | $3.53 | $353 | 88% |

### Fault Tolerance and Checkpointing

**Checkpoint Strategy**:

```python
from ray.train import CheckpointConfig, RunConfig, FailureConfig

run_config = RunConfig(
    name="fault-tolerant-training",
    storage_path="gs://my-bucket/checkpoints",
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,              # Keep last 3 checkpoints
        checkpoint_frequency=5,      # Checkpoint every 5 iterations
        checkpoint_at_end=True       # Final checkpoint
    ),
    failure_config=FailureConfig(
        max_failures=3               # Retry failed workers 3 times
    )
)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    run_config=run_config
)

# Automatic restoration on failure
result = trainer.fit()
```

**Worker Failure Handling**:
- Ray automatically detects worker failures
- Reschedules tasks on healthy workers
- Restores from latest checkpoint
- Continues training without manual intervention

### Security Best Practices

**Network Security (Vertex AI)**:

```python
# Create Ray cluster in VPC
ray_cluster = vertex_ray.create_ray_cluster(
    head_node_type=head_node_type,
    worker_node_types=worker_node_types,
    network="projects/PROJECT_ID/global/networks/my-vpc",
    # Enable Private Service Connect
    enable_private_service_connect=True
)
```

**Network Security (GKE)**:

```yaml
# Network policies for Ray cluster
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ray-cluster-policy
spec:
  podSelector:
    matchLabels:
      ray.io/cluster: ml-training-cluster
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          ray.io/cluster: ml-training-cluster
    ports:
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 10001 # Client
  egress:
  - to:
    - podSelector:
        matchLabels:
          ray.io/cluster: ml-training-cluster
```

---

## Section 6: arr-coc-0-1 VLM Use Cases (~50 lines)

### Distributed Texture Processing

```python
# Extract 13-channel texture arrays in parallel
@ray.remote(num_gpus=1)
def extract_texture_channels(image_batch):
    """Process RGB, LAB, Sobel, spatial, eccentricity on GPU"""
    texture_array = compute_13_channel_array(image_batch)
    return texture_array

# Distribute across 4 GPUs on Vertex AI/GKE
image_batches = split_dataset(images, num_splits=4)
futures = [extract_texture_channels.remote(batch) for batch in image_batches]
texture_arrays = ray.get(futures)  # Parallel execution
```

### Hyperparameter Tuning for Relevance Scorers

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_relevance_scorer(config):
    """Train Vervaekean relevance scorer"""
    scorer = build_arr_coc_scorer(
        propositional_weight=config["prop_weight"],
        perspectival_weight=config["persp_weight"],
        participatory_weight=config["part_weight"]
    )

    for epoch in range(config["epochs"]):
        accuracy = train_epoch(scorer, vqa_dataset)
        tune.report(accuracy=accuracy)

# Optimize three-way balance with ASHA
analysis = tune.run(
    train_relevance_scorer,
    config={
        "prop_weight": tune.uniform(0.2, 0.5),
        "persp_weight": tune.uniform(0.2, 0.5),
        "part_weight": tune.uniform(0.2, 0.5),
        "epochs": 10
    },
    num_samples=50,
    scheduler=ASHAScheduler(metric="accuracy", mode="max")
)

best_weights = analysis.get_best_config(metric="accuracy", mode="max")
```

### VLM Serving with Ray Serve

```python
@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 16,
        "target_num_ongoing_requests_per_replica": 10
    }
)
class ARR_COC_Predictor:
    def __init__(self):
        self.model = load_arr_coc_checkpoint()
        self.model.eval()

    @serve.batch(max_batch_size=32)
    async def handle_batch(self, requests):
        images = [req["image"] for req in requests]
        queries = [req["query"] for req in requests]

        # Batch inference through relevance realization
        with torch.no_grad():
            predictions = self.model.predict_batch(images, queries)

        return predictions

    async def __call__(self, request):
        image = decode_image(request.query_params["image"])
        query = request.query_params["query"]

        result = await self.handle_batch({"image": image, "query": query})
        return {"answer": result, "relevance_scores": result.get("scores")}

serve.run(ARR_COC_Predictor.bind())
```

---

## Key Takeaways

**Ray on Vertex AI**:
- ✅ **Best for**: Quick start, managed infrastructure, MLOps integration
- ✅ **Advantages**: No DevOps required, auto-scaling, Vertex AI ecosystem
- ⚠️ **Limitations**: Less control, fewer customization options
- 💰 **Cost**: Higher per-hour, but no management overhead

**Ray on GKE**:
- ✅ **Best for**: Production scale, custom requirements, cost optimization
- ✅ **Advantages**: Full control, spot instances, multi-tenancy, portability
- ⚠️ **Limitations**: Requires Kubernetes expertise, more operational complexity
- 💰 **Cost**: Lower per-hour (especially with spot), but management overhead

**Choose Ray on Vertex AI when**:
- Getting started with distributed ML
- Need quick prototyping and experimentation
- Want seamless Vertex AI integration (Feature Store, Model Registry, Pipelines)
- Prefer managed services over DIY infrastructure

**Choose Ray on GKE when**:
- Running production-scale workloads (>10 nodes)
- Need cost optimization with spot/preemptible instances
- Require custom configurations or multi-tenant clusters
- Already have Kubernetes expertise and infrastructure

---

## Sources

**Existing Knowledge**:
- [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md) - Ray fundamentals, Ray Train, Ray Tune, Ray Serve, Ray AIR workflows (1,013 lines, accessed 2025-11-14)

**Google Cloud Documentation**:
- [Ray on Vertex AI Overview](https://docs.cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview) - Vertex AI Ray architecture and features (accessed 2025-11-14)
- [Scale Ray clusters on Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/scale-clusters) - Autoscaling and manual scaling (accessed 2025-11-14)
- [Ray on GKE Overview](https://docs.cloud.google.com/kubernetes-engine/docs/add-on/ray-on-gke/concepts/overview) - KubeRay operator and architecture (accessed 2025-11-14)
- [Deploy GPU-accelerated Ray on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/add-on/ray-on-gke/quickstarts/ray-gpu-cluster) - GPU cluster setup guide (accessed 2025-11-14)

**Google Cloud Blog Posts**:
- [Ray on Vertex AI Announcement](https://cloud.google.com/blog/products/ai-machine-learning/ray-on-vertex-ai) - May 15, 2024, GA announcement (accessed 2025-11-14)
- [Ray on GKE: New features for AI scheduling and scaling](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-gke-new-features-for-ai-scheduling-and-scaling) - Nov 4, 2025, latest features (accessed 2025-11-14)
- [Benefits of using GKE for Ray AI workloads](https://cloud.google.com/blog/products/containers-kubernetes/the-benefits-of-using-gke-for-running-ray-ai-workloads) - Mar 18, 2024 (accessed 2025-11-14)

**Community Articles**:
- [Scale AI on Ray on Vertex AI: Let's get it started](https://medium.com/google-cloud/ray-on-vertex-ai-lets-get-it-started-7a9f8360ea25) - Ivan Nardini, Dec 18, 2023 (accessed 2025-11-14)
- [Running Ray on Google Cloud](https://medium.com/zencore/running-ray-on-google-cloud-e40b369fabfe) - Shaun Keenan, Zencore Engineering (accessed 2025-11-14)
- [Simplifying Ray and Distributed Computing](https://medium.com/google-cloud/simplifying-ray-and-distributed-computing-2c4b5ca72ad8) - Imran Roshan, Google Cloud (accessed 2025-11-14)

**Ray Official Documentation**:
- [Ray Serve Autoscaling Guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html) - Request-based autoscaling (accessed 2025-11-14)
- [Launching Ray Clusters on GCP](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html) - GCP deployment guide (accessed 2025-11-14)
- [Configuring Autoscaling](https://docs.ray.io/en/latest/cluster/vms/user-guides/configuring-autoscaling.html) - Ray autoscaler configuration (accessed 2025-11-14)

**GitHub Examples**:
- [GoogleCloudPlatform/ai-on-gke](https://github.com/GoogleCloudPlatform/ai-on-gke) - Official GKE AI examples (accessed 2025-11-14)
- [ray-project/kuberay](https://github.com/ray-project/kuberay) - KubeRay operator source (accessed 2025-11-14)

---

**Total**: ~650 lines covering Ray on Vertex AI (managed service), Ray on GKE (Kubernetes deployment), autoscaling patterns, Ray Serve for model serving, Ray AIR end-to-end workflows, production best practices, and arr-coc-0-1 VLM use cases with complete code examples and proper citations.
