# Ray Distributed ML

**Complete guide to Ray for distributed machine learning, training, tuning, and serving**

**Access date**: 2025-11-13

---

## Overview

Ray is an open-source unified framework for scaling AI and Python applications from your laptop to clusters across any cloud. Unlike Dask (which focuses on distributed data processing), Ray is designed specifically for compute-intensive ML workloads with strong support for distributed training, hyperparameter tuning, and model serving.

**Core Value Proposition**:
- **Python-native**: Parallelize existing code with minimal changes (single decorator)
- **Flexible**: General-purpose distributed computing, not just data processing
- **ML-first**: Built-in libraries for training, tuning, and serving
- **Production-ready**: Auto-scaling, fault tolerance, distributed operations

**Ray vs Dask Quick Comparison**:

| Aspect | Ray | Dask |
|--------|-----|------|
| **Primary Focus** | ML/AI workloads (training, tuning, serving) | Data processing (DataFrames, arrays) |
| **Abstraction** | Remote functions, Actors (stateful) | DataFrames, Arrays, Delayed tasks |
| **Best For** | Distributed ML training, hyperparameter tuning, model serving | Scalable data analysis, batch ETL, feature engineering |
| **Ease of Use** | Moderate (more boilerplate) | High (familiar Pandas/NumPy API) |
| **Scheduler** | Dynamic, actor-based | Work-stealing scheduler |
| **Ecosystem** | Ray Tune, Ray Train, Ray Serve, RLlib | Dask-ML, integrates with scikit-learn/XGBoost |
| **Stateful Tasks** | Yes (Actors for long-running processes) | No (stateless task-based) |

From [KDnuggets Ray or Dask Guide](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists) (accessed 2025-11-13):
> "Ray is good for tasks that need a lot of flexibility, like machine learning projects. Dask is useful if you want to work with big datasets using tools similar to Pandas or NumPy."

---

## Ray Core: Fundamental Primitives

Ray provides simple, flexible Python primitives to distribute existing code with minimal changes.

### Remote Functions (@ray.remote decorator)

Convert any Python function to run distributedly:

```python
import ray

# Initialize Ray (connects to cluster or starts local)
ray.init()

# Regular Python function
def compute_heavy_task(data):
    # Expensive computation
    return result

# Distributed version (single decorator!)
@ray.remote
def compute_heavy_task_distributed(data):
    # Same expensive computation
    return result

# Execute 100 tasks in parallel across cluster
futures = [compute_heavy_task_distributed.remote(data) for data in dataset]
results = ray.get(futures)  # Block until all complete
```

**Key Features**:
- **Minimal code changes**: Single `@ray.remote` decorator
- **Automatic distribution**: Ray handles scheduling, data transfer, fault tolerance
- **Return futures**: `.remote()` returns immediately, `.get()` blocks
- **Parallel execution**: Automatically uses all available CPUs/GPUs

### Actors (Stateful Distributed Objects)

For long-running, stateful processes (unlike Dask's stateless tasks):

```python
@ray.remote
class ParameterServer:
    def __init__(self, learning_rate):
        self.params = initialize_model()
        self.lr = learning_rate

    def get_params(self):
        return self.params

    def update_params(self, gradients):
        self.params -= self.lr * gradients

# Create actor (runs on worker node)
ps = ParameterServer.remote(learning_rate=0.01)

# Multiple workers can interact with same actor
params = ray.get(ps.get_params.remote())
ps.update_params.remote(computed_gradients)
```

**Actor Use Cases**:
- Parameter servers for distributed training
- Long-running model servers (Ray Serve)
- Stateful data pipelines
- Microservice-like patterns in ML

From [Anyscale Ray Product Page](https://www.anyscale.com/product/open-source/ray) (accessed 2025-11-13):
> "Ray translates existing Python concepts to the distributed setting, allowing any serial application to be easily parallelized with minimal code changes."

---

## Ray Train: Distributed Training

**Ray Train** simplifies distributed deep learning training with PyTorch, TensorFlow, and other frameworks.

### Key Features

1. **Seamless Framework Integration**:
   - PyTorch DistributedDataParallel (DDP)
   - PyTorch Lightning
   - HuggingFace Transformers
   - TensorFlow MultiWorkerMirroredStrategy
   - Horovod

2. **Built-in Fault Tolerance**:
   - Automatic checkpointing
   - Worker failure recovery
   - Elastic training (add/remove workers)

3. **Unified API**:
   - Same code for single-GPU → multi-GPU → multi-node
   - No manual rank/world_size management
   - Automatic data sharding

### PyTorch Distributed Training Example

From [Ray Train PyTorch Docs](https://docs.ray.io/en/latest/train/getting-started-pytorch.html) (accessed 2025-11-13):

```python
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
import torch.nn as nn

def train_func(config):
    # This function runs on each worker
    model = nn.Linear(10, 1)
    model = train.torch.prepare_model(model)  # Ray handles DDP setup

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Ray handles data sharding across workers
    dataset = train.get_dataset_shard("train")

    for epoch in range(10):
        for batch in dataset.iter_batches(batch_size=32):
            X, y = batch["X"], batch["y"]

            optimizer.zero_grad()
            output = model(X)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()

        # Report metrics (Ray aggregates across workers)
        train.report({"loss": loss.item(), "epoch": epoch})

# Configure scaling (2 workers, 1 GPU each)
scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=True,
    resources_per_worker={"GPU": 1}
)

# Create trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    datasets={"train": ray_dataset}  # Ray Data integration
)

# Run distributed training
result = trainer.fit()
print(f"Final metrics: {result.metrics}")
```

**Ray Train handles**:
- DistributedDataParallel setup (no manual `init_process_group`)
- Data sharding across workers
- Metric aggregation
- Checkpointing and restoration
- Worker failure recovery

### Multi-Node Training Patterns

```python
# Scale to 4 nodes, 8 GPUs per node = 32 total GPUs
scaling_config = ScalingConfig(
    num_workers=32,
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 4}
)

# Ray automatically handles:
# - NCCL communication across nodes
# - Data distribution
# - Gradient synchronization
# - Failure recovery
```

### Horovod Integration

```python
from ray.train.horovod import HorovodTrainer

trainer = HorovodTrainer(
    train_loop_per_worker=horovod_train_func,
    scaling_config=scaling_config
)
result = trainer.fit()
```

---

## Ray Tune: Hyperparameter Optimization

**Ray Tune** is Ray's library for distributed hyperparameter tuning with advanced scheduling algorithms.

### Key Features

1. **Advanced Schedulers**:
   - ASHA (Asynchronous Successive Halving Algorithm)
   - Population-Based Training (PBT)
   - Median Stopping Rule
   - HyperBand

2. **Search Algorithms**:
   - Grid search, random search
   - Bayesian optimization (Optuna, Ax)
   - Genetic algorithms
   - Tree-Parzen Estimator (TPE)

3. **Framework Integration**:
   - PyTorch, TensorFlow, Keras
   - XGBoost, LightGBM
   - Scikit-learn
   - Any custom training code

### Basic Hyperparameter Tuning Example

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    # Your training function
    model = build_model(
        lr=config["lr"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"]
    )

    for epoch in range(100):
        accuracy = train_epoch(model)

        # Report intermediate results
        tune.report(accuracy=accuracy, epoch=epoch)

# Define search space
search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "hidden_dim": tune.randint(64, 512)
}

# ASHA: Stops bad trials early to save compute
scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=100,           # Max epochs per trial
    grace_period=10,     # Min epochs before stopping
    reduction_factor=2   # Stop bottom 50% every grace_period
)

# Run tuning (32 trials in parallel if resources available)
analysis = tune.run(
    train_model,
    config=search_space,
    scheduler=scheduler,
    num_samples=100,     # Total trials to run
    resources_per_trial={"cpu": 2, "gpu": 1}
)

# Get best hyperparameters
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print(f"Best config: {best_config}")
print(f"Best accuracy: {analysis.best_result['accuracy']}")
```

### Population-Based Training (PBT)

Dynamically adapt hyperparameters during training:

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="accuracy",
    mode="max",
    perturbation_interval=10,  # Mutate every 10 iterations
    hyperparam_mutations={
        "lr": lambda: tune.loguniform(1e-4, 1e-1).sample(),
        "momentum": [0.8, 0.9, 0.95, 0.99]
    }
)

analysis = tune.run(
    train_model,
    scheduler=pbt_scheduler,
    config=search_space,
    num_samples=20  # Population size
)
```

**PBT Benefits**:
- Continuously adapts hyperparameters based on performance
- Exploits best configurations, explores variations
- More efficient than static hyperparameter search

From [KDnuggets Ray or Dask Guide](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists) (accessed 2025-11-13):

Example Ray Tune usage:
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_fn(config):
    # Model training logic here
    ...

tune.run(
    train_fn,
    config={"lr": tune.grid_search([0.01, 0.001, 0.0001])},
    scheduler=ASHAScheduler(metric="accuracy", mode="max")
)
```

### Integration with Ray Train

Combine distributed training + hyperparameter tuning:

```python
from ray.train.torch import TorchTrainer
from ray import tune

def train_func_with_config(config):
    # config contains both Ray Train and Tune parameters
    model = build_model(lr=config["lr"], hidden_dim=config["hidden_dim"])

    # Ray Train handles distributed training
    model = train.torch.prepare_model(model)

    for epoch in range(config["epochs"]):
        loss = train_epoch(model)
        tune.report(loss=loss)  # Report to Tune

# Tuner wraps Trainer for hyperparameter search
from ray.tune import Tuner

tuner = Tuner(
    TorchTrainer(
        train_loop_per_worker=train_func_with_config,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
    ),
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-4, 1e-1),
            "hidden_dim": tune.choice([128, 256, 512])
        }
    },
    tune_config=tune.TuneConfig(
        num_samples=20,
        scheduler=ASHAScheduler()
    )
)

results = tuner.fit()
```

---

## Ray Serve: Model Serving

**Ray Serve** is Ray's framework for scalable model deployment and serving.

### Key Features

1. **Framework-Agnostic**:
   - Deploy any Python code (PyTorch, TensorFlow, scikit-learn, etc.)
   - No special model format required

2. **Autoscaling**:
   - Scale up/down based on request load
   - Min/max replicas configuration

3. **Model Composition**:
   - Chain multiple models (ensemble)
   - Preprocessing → Model → Postprocessing pipelines

4. **Production Features**:
   - Zero-downtime updates
   - Request batching
   - Multi-model serving

### Basic Model Serving Example

From [KDnuggets Ray or Dask Guide](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists) (accessed 2025-11-13):

```python
from ray import serve

@serve.deployment
class ModelDeployment:
    def __init__(self):
        self.model = load_model()  # Load your trained model

    def __call__(self, request_body):
        data = request_body
        prediction = self.model.predict([data])[0]
        return {"prediction": prediction}

# Deploy model (starts HTTP server)
serve.run(ModelDeployment.bind())

# Model now available at http://localhost:8000/
# Send POST requests with input data
```

### Autoscaling Configuration

```python
@serve.deployment(
    num_replicas=2,              # Start with 2 replicas
    ray_actor_options={"num_cpus": 2, "num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5
    }
)
class ScalableModel:
    def __init__(self):
        self.model = load_gpu_model()

    def __call__(self, request):
        return self.model.predict(request)

serve.run(ScalableModel.bind())
```

**Autoscaling behavior**:
- Ray monitors ongoing requests per replica
- Scales up if target exceeded (too many requests)
- Scales down if below target (idle replicas)
- Respects min/max bounds

### Request Batching

Optimize GPU utilization by batching requests:

```python
@serve.deployment(max_concurrent_queries=100)
class BatchedModel:
    def __init__(self):
        self.model = load_model()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def handle_batch(self, requests):
        # Receives list of requests, returns list of predictions
        inputs = [req for req in requests]
        predictions = self.model.predict_batch(inputs)
        return predictions

    async def __call__(self, request):
        return await self.handle_batch(request)

serve.run(BatchedModel.bind())
```

**Batching benefits**:
- Higher GPU throughput (process multiple inputs together)
- Reduced per-request latency (amortized overhead)
- Better hardware utilization

### Model Composition (Ensemble)

Chain multiple models:

```python
@serve.deployment
class Preprocessor:
    def __call__(self, raw_input):
        return preprocess(raw_input)

@serve.deployment
class Model:
    def __call__(self, processed_input):
        return self.model.predict(processed_input)

@serve.deployment
class Postprocessor:
    def __call__(self, prediction):
        return format_output(prediction)

@serve.deployment
class Pipeline:
    def __init__(self, preprocessor, model, postprocessor):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    async def __call__(self, request):
        x = await self.preprocessor.remote(request)
        y = await self.model.remote(x)
        z = await self.postprocessor.remote(y)
        return z

# Deploy pipeline
serve.run(Pipeline.bind(
    Preprocessor.bind(),
    Model.bind(),
    Postprocessor.bind()
))
```

---

## Ray AIR (AI Runtime)

**Ray AIR** unifies Ray Train, Ray Tune, Ray Serve, and Ray Data into a single, cohesive ML platform.

### Complete ML Workflow with Ray AIR

```python
import ray
from ray import train, tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.tune import Tuner
from ray import serve

# Step 1: Load data with Ray Data
dataset = ray.data.read_parquet("s3://my-bucket/data/")

# Step 2: Distributed training with Ray Train
def train_func(config):
    model = build_model(config["lr"], config["hidden_dim"])
    model = train.torch.prepare_model(model)

    dataset = train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        loss = train_epoch(model, dataset)
        train.report({"loss": loss})

    # Save checkpoint
    train.report(checkpoint=train.save_checkpoint(model))

# Step 3: Hyperparameter tuning with Ray Tune
tuner = Tuner(
    TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
        datasets={"train": dataset}
    ),
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-4, 1e-1),
            "hidden_dim": tune.choice([128, 256, 512]),
            "epochs": 20
        }
    },
    tune_config=tune.TuneConfig(num_samples=10)
)

results = tuner.fit()
best_result = results.get_best_result(metric="loss", mode="min")

# Step 4: Deploy best model with Ray Serve
checkpoint = best_result.checkpoint

@serve.deployment
class BestModel:
    def __init__(self, checkpoint):
        self.model = load_from_checkpoint(checkpoint)

    def __call__(self, request):
        return self.model.predict(request)

serve.run(BestModel.bind(checkpoint))
```

**Ray AIR Benefits**:
- **Unified API**: Same abstractions across data loading, training, tuning, serving
- **Seamless transitions**: Checkpoints flow from training → tuning → serving
- **Resource efficiency**: Share compute resources across pipeline stages

---

## Ray vs Dask: When to Use What

### Use Ray When

1. **Machine Learning Workloads**:
   - Distributed deep learning training (multi-GPU, multi-node)
   - Hyperparameter optimization with advanced schedulers
   - Model serving with autoscaling
   - Reinforcement learning (RLlib)

2. **Stateful Applications**:
   - Long-running parameter servers
   - Microservice-like architectures
   - Online learning systems

3. **Custom Distributed Applications**:
   - Need actors (stateful objects)
   - Complex task dependencies
   - Event-driven architectures

4. **Production ML Pipelines**:
   - End-to-end workflows (data → train → tune → serve)
   - Need fault tolerance for long-running jobs
   - Require autoscaling inference

**Ray Example Use Cases**:
```python
# Distributed hyperparameter tuning
tune.run(train_model, config=search_space, num_samples=100)

# Multi-GPU training
TorchTrainer(train_func, scaling_config=ScalingConfig(num_workers=8))

# Model serving with autoscaling
serve.deployment(autoscaling_config={...})(ModelClass)
```

### Use Dask When

1. **Data Processing**:
   - Large-scale ETL (Extract-Transform-Load)
   - Feature engineering on datasets too big for memory
   - Data cleaning and preprocessing

2. **Familiar APIs**:
   - Want Pandas-like DataFrame API at scale
   - Need NumPy-compatible array operations
   - Using scikit-learn on large datasets

3. **Batch Analytics**:
   - Out-of-core computations (data > RAM)
   - SQL-style aggregations on large datasets
   - Statistical analysis at scale

**Dask Example Use Cases**:
```python
# Large DataFrame operations
df = dd.read_csv('s3://data/*.csv')
df = df[df['amount'] > 100].groupby('category').mean()

# Out-of-core array computations
x = da.random.random((100000, 100000), chunks=(1000, 1000))
y = x.mean(axis=0).compute()

# Distributed scikit-learn
from dask_ml.wrappers import ParallelPostFit
estimator = ParallelPostFit(RandomForest()).fit(X, y)
```

### Performance Comparison

From [Emergent Methods Ray vs Dask](https://emergentmethods.medium.com/ray-vs-dask-lessons-learned-serving-240k-models-per-day-in-real-time-7863c8968a1f) (accessed 2025-11-13):
> "The largest gain in performance is observed in the inference time for the 10,300 models, with Ray Cluster Memory outperforming Dask by 64%."

**Benchmark Summary** (240K models/day production workload):
- **Ray Cluster Memory**: 64% faster than Dask for inference
- **Ray advantages**: Better for ML workloads, stateful actors, model serving
- **Dask advantages**: Better for data processing, familiar Pandas API

From [Onehouse Ray vs Dask vs Spark](https://www.onehouse.ai/blog/apache-spark-vs-ray-vs-dask-comparing-data-science-machine-learning-engines) (accessed 2025-11-13):
> "A benchmark comparing Dask Distributed vs. Dask on Ray found Dask on Ray to have 3x better cost:performance than Dask Distributed for a 3 PB dataset."

**Note**: You can run Dask *on top of* Ray for best of both worlds (Dask API + Ray scheduling).

---

## Ray on Kubernetes

Ray integrates seamlessly with Kubernetes via the **Ray Operator** and **KubeRay**.

### Ray Cluster on K8s

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ml-training-cluster
spec:
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
            requests:
              cpu: "2"
              memory: "8Gi"

  workerGroupSpecs:
  - replicas: 4
    minReplicas: 1
    maxReplicas: 10
    groupName: gpu-workers
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0-py310-gpu
          resources:
            limits:
              nvidia.com/gpu: 1
              cpu: "8"
              memory: "32Gi"
```

**Ray on K8s Features**:
- **Autoscaling**: Ray automatically requests/releases K8s pods
- **GPU scheduling**: Ray Operator handles nvidia.com/gpu resources
- **Multi-tenancy**: Multiple Ray clusters on same K8s cluster
- **Integration**: Works with Kubeflow, Argo Workflows

From [CNCF Ray on Kubernetes Talk](https://www.youtube.com/watch?v=M5FAI2kmcdE) (accessed 2025-11-13):
> "Ray is an open source python library that helps you with some of the distributed compute challenges... Here we're going to see how the distributed framework of Ray can be implemented or used on kubernetes for distributed ml models."

---

## Production Best Practices

### 1. Resource Management

```python
# Specify resources explicitly
@ray.remote(num_cpus=4, num_gpus=1, memory=16*1024**3)
def gpu_task(data):
    return process_on_gpu(data)

# Configure placement groups for co-location
from ray.util.placement_group import placement_group

pg = placement_group([
    {"GPU": 1, "CPU": 4},  # Worker 1
    {"GPU": 1, "CPU": 4},  # Worker 2
    {"GPU": 1, "CPU": 4},  # Worker 3
    {"GPU": 1, "CPU": 4}   # Worker 4
], strategy="STRICT_PACK")  # All on same node

ray.get(pg.ready())
```

### 2. Checkpointing and Fault Tolerance

```python
from ray.train import CheckpointConfig, RunConfig

run_config = RunConfig(
    name="my_experiment",
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,                    # Keep last 3 checkpoints
        checkpoint_frequency=5,           # Checkpoint every 5 iterations
        checkpoint_at_end=True            # Final checkpoint
    ),
    failure_config=FailureConfig(
        max_failures=3                    # Retry failed workers 3 times
    )
)

trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config
)
```

### 3. Monitoring with Ray Dashboard

Ray includes built-in dashboard for observability:

```python
# Start Ray with dashboard
ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)

# Access dashboard at http://localhost:8265
```

**Dashboard Features**:
- Live cluster metrics (CPU, GPU, memory, network)
- Task/actor visualization
- Resource allocation
- Job logs and profiling

### 4. Integration with Weights & Biases

```python
from ray.air.integrations.wandb import WandbLoggerCallback

trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=RunConfig(
        callbacks=[WandbLoggerCallback(project="my_project")]
    )
)
```

---

## arr-coc-0-1 VLM Training Use Cases

### Multi-GPU Texture Array Processing

```python
# Distributed texture extraction for 13-channel arrays
@ray.remote(num_gpus=1)
def extract_texture_channels(image_batch):
    # Process RGB, LAB, Sobel on GPU
    return compute_texture_array(image_batch)

# Distribute across 4 GPUs
image_batches = split_dataset(images, num_splits=4)
futures = [extract_texture_channels.remote(batch) for batch in image_batches]
texture_arrays = ray.get(futures)
```

### Hyperparameter Tuning for Relevance Scorers

```python
def train_relevance_scorer(config):
    # Build scorer with config
    scorer = build_scorer(
        propositional_weight=config["prop_weight"],
        perspectival_weight=config["persp_weight"],
        participatory_weight=config["part_weight"]
    )

    # Train on VQA dataset
    for epoch in range(config["epochs"]):
        accuracy = train_epoch(scorer, vqa_dataset)
        tune.report(accuracy=accuracy)

# Ray Tune: optimize three-way balance
analysis = tune.run(
    train_relevance_scorer,
    config={
        "prop_weight": tune.uniform(0.2, 0.5),
        "persp_weight": tune.uniform(0.2, 0.5),
        "part_weight": tune.uniform(0.2, 0.5),
        "epochs": 10
    },
    num_samples=50,
    scheduler=ASHAScheduler()
)

best_weights = analysis.get_best_config(metric="accuracy", mode="max")
```

### Distributed VLM Training with Ray Train

```python
def train_arr_coc_vlm(config):
    # Initialize ARR-COC model
    model = ARR_COC_VLM(
        texture_channels=13,
        relevance_scorers=3,
        token_budget=(64, 400)
    )

    model = train.torch.prepare_model(model)  # Ray DDP setup

    # Ray Data for large-scale vision-language data
    dataset = train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        for batch in dataset.iter_batches(batch_size=32):
            images, queries, answers = batch

            # Forward pass through relevance realization
            loss = model.train_step(images, queries, answers)

        train.report({"loss": loss, "epoch": epoch})

# Scale to 8 GPUs across 2 nodes
trainer = TorchTrainer(
    train_loop_per_worker=train_arr_coc_vlm,
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8}
    ),
    datasets={"train": ray.data.read_images("s3://vlm-data/")}
)

result = trainer.fit()
```

### Model Serving for Real-Time Inference

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
class ARR_COC_Server:
    def __init__(self):
        self.model = load_arr_coc_checkpoint()
        self.model.eval()

    @serve.batch(max_batch_size=32)
    async def handle_batch(self, images_and_queries):
        # Batch inference for efficiency
        images = [iq["image"] for iq in images_and_queries]
        queries = [iq["query"] for iq in images_and_queries]

        with torch.no_grad():
            predictions = self.model.predict_batch(images, queries)

        return predictions

    async def __call__(self, request):
        image = decode_image(request["image"])
        query = request["query"]

        prediction = await self.handle_batch({"image": image, "query": query})
        return {"answer": prediction}

serve.run(ARR_COC_Server.bind())
```

---

## Key Takeaways

**Ray Strengths**:
1. **ML-first design**: Purpose-built for distributed training, tuning, serving
2. **Flexibility**: General-purpose distributed computing, not limited to data processing
3. **Stateful actors**: Long-running processes for parameter servers, model servers
4. **Production features**: Autoscaling, fault tolerance, zero-downtime updates
5. **Unified platform**: Ray AIR connects data → train → tune → serve

**When to Choose Ray Over Dask**:
- Distributed deep learning training (multi-GPU, multi-node)
- Hyperparameter optimization with advanced schedulers (ASHA, PBT)
- Model serving with autoscaling and request batching
- Need stateful distributed objects (actors)
- Building end-to-end ML pipelines

**When to Choose Dask Over Ray**:
- Large-scale data processing (ETL, feature engineering)
- Want familiar Pandas/NumPy API
- Batch analytics and aggregations
- Don't need ML-specific features

**Ray + Kubernetes**:
- Ray Operator for K8s integration
- Autoscaling worker pods
- GPU scheduling and resource management
- Works with Kubeflow, Argo Workflows

---

## Sources

**Official Documentation**:
1. [Ray Documentation](https://docs.ray.io/en/latest/) - Ray project official docs
2. [Ray Train PyTorch Guide](https://docs.ray.io/en/latest/train/getting-started-pytorch.html) - Getting started with distributed training (accessed 2025-11-13)
3. [Anyscale Ray Product Page](https://www.anyscale.com/product/open-source/ray) - Ray features and capabilities (accessed 2025-11-13)

**Web Research**:
4. [KDnuggets: Ray or Dask? A Practical Guide for Data Scientists](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists) - Jayita Gulati, September 9, 2025 (accessed 2025-11-13)
5. [Emergent Methods: Ray vs Dask - Lessons Learned Serving 240K Models](https://emergentmethods.medium.com/ray-vs-dask-lessons-learned-serving-240k-models-per-day-in-real-time-7863c8968a1f) - Production comparison, 2022 (accessed 2025-11-13)
6. [Onehouse: Ray vs Dask vs Apache Spark Comparison](https://www.onehouse.ai/blog/apache-spark-vs-ray-vs-dask-comparing-data-science-machine-learning-engines) - April 17, 2025 (accessed 2025-11-13)

**GitHub Examples**:
7. [Ray Project GitHub](https://github.com/ray-project/ray) - Main Ray repository with examples (accessed 2025-11-13)

**Additional References**:
- [CNCF: Introduction to Distributed ML Workloads with Ray on Kubernetes](https://www.youtube.com/watch?v=M5FAI2kmcdE) - November 16, 2024 (accessed 2025-11-13)
- [Anyscale: Overcoming Distributed ML Challenges with Ray Train](https://www.youtube.com/watch?v=jLsi8-op-lg) - Ray Summit 2024 (accessed 2025-11-13)

---

**Total**: ~530 lines covering Ray fundamentals, Ray Train (distributed training), Ray Tune (hyperparameter optimization), Ray Serve (model serving), Ray AIR (unified platform), Ray vs Dask comparison, Kubernetes integration, production best practices, and arr-coc-0-1 VLM training use cases with complete code examples and proper citations.
