# GKE Training Operators: Kubeflow Training Operator Deep Dive

## Overview

The Kubeflow Training Operator provides Kubernetes-native orchestration for distributed machine learning training jobs on GKE. It abstracts the complexity of distributed training across multiple frameworks (PyTorch, TensorFlow, MPI/Horovod, JAX, XGBoost, PaddlePaddle) with declarative custom resources and automated lifecycle management.

**Core value proposition**: Transform complex distributed training setups into simple YAML declarations, with automatic master/worker coordination, fault tolerance, and GPU scheduling.

From [Kubeflow Training Operator Overview](https://www.kubeflow.org/docs/components/trainer/overview/) (accessed 2025-11-16):
- Kubernetes custom resource definitions (CRDs) for ML training jobs
- Multi-framework support with unified API patterns
- Built-in distributed training primitives (master/worker topology, gang scheduling)
- Integration with GKE GPU scheduling, autoscaling, and monitoring

## Section 1: Kubeflow Training Operator Architecture

### Core Components

**Training Operator Controller**:
- Watches custom resources (PyTorchJob, TFJob, MPIJob, etc.)
- Creates pods based on replicaSpecs (master, worker, parameter server)
- Manages job lifecycle (creation → running → completed/failed)
- Handles pod restarts and failure recovery

**Custom Resource Definitions (CRDs)**:
```yaml
# PyTorchJob structure
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed-mnist
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/kubeflow-ci/pytorch-dist-mnist:1.0
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/kubeflow-ci/pytorch-dist-mnist:1.0
            resources:
              limits:
                nvidia.com/gpu: 1
```

From [Kubeflow PyTorchJob Documentation](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/) (accessed 2025-11-16):
- Master replica coordinates training and hosts rank 0
- Worker replicas participate in distributed training
- Automatic environment variable injection (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE)

**Service Discovery**:
- Operator creates headless services for pod-to-pod communication
- DNS-based discovery: `<job-name>-master-0.<job-name>-master.namespace.svc.cluster.local`
- Port configuration in pod specs exposed as service endpoints

### Installation on GKE

**Standalone Training Operator**:
```bash
# Install Training Operator v1 (legacy)
kubectl apply -k "github.com/kubeflow/manifests/apps/training-operator/upstream/overlays/kubeflow?ref=v1.9.0"

# Verify installation
kubectl get pods -n kubeflow | grep training-operator
# Output: training-operator-xxxxxxxxx-xxxxx   1/1     Running   0          30s

# Check CRDs installed
kubectl get crd | grep kubeflow.org
# pytorchjobs.kubeflow.org
# tfjobs.kubeflow.org
# mpijobs.kubeflow.org
# jaxjobs.kubeflow.org
# xgboostjobs.kubeflow.org
# paddlejobs.kubeflow.org
```

**Full Kubeflow Platform** (includes Training Operator + Pipelines + Katib + Notebooks):
```bash
# Install full Kubeflow 1.9
git clone https://github.com/kubeflow/manifests.git
cd manifests
git checkout v1.9-branch

# GKE-specific configuration
kustomize build distributions/gke/kustomization.yaml | kubectl apply -f -

# Wait for all components
kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=600s
```

From [Kubeflow Installation Guide](https://www.kubeflow.org/docs/started/installing-kubeflow/) (accessed 2025-11-16):
- Training Operator is core component in full Kubeflow platform
- Can be installed standalone for training-only use cases
- GKE distribution includes IAM integration and GPU optimizations

## Section 2: PyTorchJob - Distributed PyTorch Training

### Single-Node Multi-GPU Training

**DDP (DistributedDataParallel) Example**:
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-ddp-cifar10
  namespace: arr-coc-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/pytorch-ddp-cifar10:v1.0
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=8
            - train.py
            - --epochs=100
            - --batch-size=256
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "128Gi"
              requests:
                nvidia.com/gpu: 8
                memory: "128Gi"
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
```

**Automatic Environment Variables**:
```python
# PyTorchJob operator automatically sets:
# MASTER_ADDR=pytorch-ddp-cifar10-master-0
# MASTER_PORT=23456
# RANK=0 (for master)
# WORLD_SIZE=1 (single node)

import torch.distributed as dist

def setup_distributed():
    """No manual setup needed - operator provides env vars"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://'  # Uses MASTER_ADDR, MASTER_PORT from operator
    )

# Training code
def train():
    setup_distributed()
    model = torch.nn.parallel.DistributedDataParallel(model)
    # ... training loop
```

### Multi-Node Distributed Training

**4-Node 32-GPU PyTorchJob**:
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-multinode-bert
  namespace: arr-coc-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/bert-pretraining:v2.0
            args:
            - --model_name_or_path=bert-large-uncased
            - --train_file=/data/wikipedia.txt
            - --per_device_train_batch_size=32
            - --learning_rate=5e-5
            - --num_train_epochs=3
            env:
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_IB_DISABLE
              value: "1"
            - name: NCCL_DEBUG
              value: "INFO"
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "256Gi"
              requests:
                nvidia.com/gpu: 8
                memory: "256Gi"
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/bert-pretraining:v2.0
            args:
            - --model_name_or_path=bert-large-uncased
            - --train_file=/data/wikipedia.txt
            - --per_device_train_batch_size=32
            - --learning_rate=5e-5
            - --num_train_epochs=3
            env:
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_IB_DISABLE
              value: "1"
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "256Gi"
              requests:
                nvidia.com/gpu: 8
                memory: "256Gi"
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
  runPolicy:
    cleanPodPolicy: None  # Keep pods for debugging
    ttlSecondsAfterFinished: 86400  # Clean up after 24 hours
```

**Automatic Rank Assignment**:
```python
# Master pod: RANK=0, WORLD_SIZE=4
# Worker-0 pod: RANK=1, WORLD_SIZE=4
# Worker-1 pod: RANK=2, WORLD_SIZE=4
# Worker-2 pod: RANK=3, WORLD_SIZE=4

import os
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
master_addr = os.environ['MASTER_ADDR']
master_port = os.environ['MASTER_PORT']

print(f"Node {rank}/{world_size} connecting to {master_addr}:{master_port}")
```

From [PyTorch Distributed Training Documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html) (accessed 2025-11-16):
- NCCL backend for GPU communication (best for multi-GPU)
- Gloo backend for CPU communication
- init_method='env://' reads operator-injected environment variables

### Elastic Training (PyTorch Elastic)

**Elastic Training with Dynamic Worker Scaling**:
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-elastic-imagenet
spec:
  elasticPolicy:
    minReplicas: 2
    maxReplicas: 8
    rdzvBackend: c10d
  pytorchReplicaSpecs:
    Worker:
      replicas: 4  # Initial replicas, can scale 2-8
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/elastic-imagenet:v1.0
            command:
            - python
            - -m
            - torch.distributed.run
            - --nnodes=2:8
            - --nproc_per_node=8
            - --rdzv_backend=c10d
            - --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 8
```

From [Elastic Training Blog Post](https://blog.kubeflow.org/elastic%20training/operators/2021/03/15/elastic-training.html) (accessed 2025-11-16):
- Elastic training adapts to node failures and additions
- Workers can join/leave without restarting entire job
- Checkpoint-resume mechanism handles worker changes
- Cost optimization: Scale down during off-peak hours

## Section 3: TFJob - TensorFlow Distributed Strategies

### TensorFlow MultiWorkerMirroredStrategy

**TFJob with Parameter Server Architecture**:
```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: tfjob-resnet50
  namespace: arr-coc-training
spec:
  tfReplicaSpecs:
    Chief:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: gcr.io/arr-coc-project/tf-resnet50:v2.15
            command:
            - python
            - train.py
            - --strategy=MultiWorkerMirroredStrategy
            - --epochs=50
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "32Gi"
    Worker:
      replicas: 7
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: gcr.io/arr-coc-project/tf-resnet50:v2.15
            command:
            - python
            - train.py
            - --strategy=MultiWorkerMirroredStrategy
            - --epochs=50
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "32Gi"
    PS:  # Parameter servers (optional for ParameterServerStrategy)
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: gcr.io/arr-coc-project/tf-resnet50:v2.15
            resources:
              limits:
                memory: "64Gi"
              requests:
                memory: "64Gi"
```

**TF_CONFIG Environment Variable** (auto-generated by operator):
```json
{
  "cluster": {
    "chief": ["tfjob-resnet50-chief-0:2222"],
    "worker": [
      "tfjob-resnet50-worker-0:2222",
      "tfjob-resnet50-worker-1:2222",
      "tfjob-resnet50-worker-2:2222",
      "tfjob-resnet50-worker-3:2222",
      "tfjob-resnet50-worker-4:2222",
      "tfjob-resnet50-worker-5:2222",
      "tfjob-resnet50-worker-6:2222"
    ],
    "ps": [
      "tfjob-resnet50-ps-0:2222",
      "tfjob-resnet50-ps-1:2222"
    ]
  },
  "task": {
    "type": "worker",
    "index": 0
  }
}
```

**Training Code** (TensorFlow 2.x):
```python
import tensorflow as tf
import json
import os

# TFJob operator automatically sets TF_CONFIG
tf_config = json.loads(os.environ['TF_CONFIG'])
print(f"Task type: {tf_config['task']['type']}, index: {tf_config['task']['index']}")

# Strategy automatically reads TF_CONFIG
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_model()  # Model creation under strategy scope
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Distributed training
model.fit(train_dataset, epochs=50, callbacks=[...])
```

From [TensorFlow Distributed Training Guide](https://www.tensorflow.org/guide/distributed_training) (accessed 2025-11-16):
- MultiWorkerMirroredStrategy: All-reduce across workers (like PyTorch DDP)
- ParameterServerStrategy: Centralized parameters on PS nodes
- TF_CONFIG: JSON configuration for cluster topology

## Section 4: MPIJob - Horovod Multi-Node Training

### MPI Operator for Horovod

**8-Node Horovod Training**:
```yaml
apiVersion: kubeflow.org/v1
kind: MPIJob
metadata:
  name: horovod-resnet50
  namespace: arr-coc-training
spec:
  slotsPerWorker: 8  # 8 GPUs per node
  cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: mpi
            image: gcr.io/arr-coc-project/horovod-resnet50:v0.28
            command:
            - mpirun
            - --allow-run-as-root
            - -np
            - "64"  # 8 nodes × 8 GPUs = 64 processes
            - --hostfile
            - /etc/mpi/hostfile
            - --bind-to
            - none
            - -map-by
            - slot
            - -x
            - NCCL_DEBUG=INFO
            - -x
            - LD_LIBRARY_PATH
            - -x
            - PATH
            - python
            - train_resnet50.py
            - --batch-size=256
            - --epochs=90
            resources:
              limits:
                cpu: "4"
                memory: "16Gi"
    Worker:
      replicas: 8
      template:
        spec:
          containers:
          - name: mpi
            image: gcr.io/arr-coc-project/horovod-resnet50:v0.28
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "256Gi"
              requests:
                nvidia.com/gpu: 8
                memory: "256Gi"
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
```

**Horovod Training Code**:
```python
import horovod.torch as hvd
import torch

# Initialize Horovod
hvd.init()

# Pin GPU to local rank
torch.cuda.set_device(hvd.local_rank())

# Create model and optimizer
model = create_model().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Wrap optimizer with Horovod DistributedOptimizer
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    compression=hvd.Compression.fp16  # FP16 gradient compression
)

# Broadcast initial parameters from rank 0
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # Horovod handles allreduce automatically

        if hvd.rank() == 0 and batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
```

From [Kubeflow MPI Operator Documentation](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/mpi/) (accessed 2025-11-16):
- Launcher pod orchestrates mpirun across worker pods
- Automatic SSH key generation for MPI communication
- Hostfile generated with worker pod IPs
- Ring-allreduce algorithm for gradient synchronization

### Elastic Horovod on Kubernetes

**Dynamic Worker Scaling**:
```yaml
apiVersion: kubeflow.org/v1
kind: MPIJob
metadata:
  name: elastic-horovod-bert
spec:
  slotsPerWorker: 8
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: mpi
            image: gcr.io/arr-coc-project/elastic-horovod:v0.28
            command:
            - horovodrun
            - --gloo
            - --elastic
            - --min-np=16
            - --max-np=64
            - --host-discovery-script=/usr/local/bin/discover_hosts.sh
            - python
            - train_bert.py
    Worker:
      replicas: 4  # Can scale 2-8 nodes (16-64 GPUs)
      template:
        spec:
          containers:
          - name: mpi
            image: gcr.io/arr-coc-project/elastic-horovod:v0.28
            resources:
              limits:
                nvidia.com/gpu: 8
```

## Section 5: Integration with Kubeflow Pipelines

### Training Jobs as Pipeline Components

**Kubeflow Pipeline with PyTorchJob**:
```python
from kfp import dsl
from kfp import compiler
from kfp.dsl import PipelineTask
import kfp.kubernetes as k8s

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['kubernetes']
)
def create_pytorchjob(
    job_name: str,
    image: str,
    num_workers: int,
    gpu_per_worker: int
) -> str:
    """Create PyTorchJob for distributed training"""
    from kubernetes import client, config
    import yaml

    config.load_incluster_config()

    pytorchjob = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "pytorch",
                                "image": image,
                                "resources": {
                                    "limits": {"nvidia.com/gpu": gpu_per_worker}
                                }
                            }]
                        }
                    }
                },
                "Worker": {
                    "replicas": num_workers,
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "pytorch",
                                "image": image,
                                "resources": {
                                    "limits": {"nvidia.com/gpu": gpu_per_worker}
                                }
                            }]
                        }
                    }
                }
            }
        }
    }

    api = client.CustomObjectsApi()
    api.create_namespaced_custom_object(
        group="kubeflow.org",
        version="v1",
        namespace="kubeflow",
        plural="pytorchjobs",
        body=pytorchjob
    )

    return job_name

@dsl.component(base_image='python:3.11')
def monitor_pytorchjob(job_name: str) -> str:
    """Monitor PyTorchJob until completion"""
    from kubernetes import client, config
    import time

    config.load_incluster_config()
    api = client.CustomObjectsApi()

    while True:
        job = api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace="kubeflow",
            plural="pytorchjobs",
            name=job_name
        )

        status = job.get('status', {})
        conditions = status.get('conditions', [])

        for condition in conditions:
            if condition['type'] == 'Succeeded' and condition['status'] == 'True':
                return "Training completed successfully"
            elif condition['type'] == 'Failed' and condition['status'] == 'True':
                raise RuntimeError(f"Training failed: {condition.get('message')}")

        time.sleep(30)

@dsl.pipeline(name='Distributed PyTorch Training Pipeline')
def training_pipeline(
    model_name: str = 'resnet50',
    num_workers: int = 4,
    gpu_per_worker: int = 8
):
    # Step 1: Data preprocessing (CPU-only job)
    preprocess_op = dsl.ContainerOp(
        name='preprocess-data',
        image='gcr.io/arr-coc-project/data-preprocessing:v1.0',
        arguments=['--dataset=imagenet', '--output=/data/preprocessed']
    )

    # Step 2: Create distributed training job
    training_op = create_pytorchjob(
        job_name=f'training-{model_name}',
        image=f'gcr.io/arr-coc-project/{model_name}-training:v1.0',
        num_workers=num_workers,
        gpu_per_worker=gpu_per_worker
    )
    training_op.after(preprocess_op)

    # Step 3: Monitor training completion
    monitor_op = monitor_pytorchjob(
        job_name=training_op.output
    )

    # Step 4: Model evaluation (single GPU)
    eval_op = dsl.ContainerOp(
        name='evaluate-model',
        image='gcr.io/arr-coc-project/model-evaluation:v1.0',
        arguments=['--model_path=/models/final', '--test_data=/data/test']
    )
    k8s.use_field_path_as_env(
        eval_op,
        env_name='POD_NAME',
        field_path='metadata.name'
    )
    eval_op.after(monitor_op)
    eval_op.set_gpu_limit(1)

# Compile pipeline
compiler.Compiler().compile(training_pipeline, 'training_pipeline.yaml')
```

**TrainJob (Kubeflow Trainer V2)**:

From [KubeCon 2024 Presentation](https://static.sched.com/hosted_files/kccncna2024/f4/KubeCon%20NA%202024.%20Democratizing%20AI%20Model%20Training%20on%20Kubernetes%20with%20Kubeflow%20TrainJob%20and%20JobSet.%20Andrey%20Velichkevich%20and%20Yuki%20Iwai.pdf) (accessed 2025-11-16):
- TrainJob (V2): Unified API for PyTorchJob, TFJob, JAXJob
- Built on JobSet for better multi-pod coordination
- Improved integration with Kubeflow Pipelines
- Simpler YAML structure and better defaults

## Section 6: Elastic Training and Autoscaling

### Dynamic Worker Scaling

**HorizontalPodAutoscaler for Training Workers**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pytorch-elastic-hpa
  namespace: arr-coc-training
spec:
  scaleTargetRef:
    apiVersion: kubeflow.org/v1
    kind: PyTorchJob
    name: pytorch-elastic-training
  minReplicas: 2
  maxReplicas: 16
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: training_throughput_samples_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scale down
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

**Checkpoint-Resume for Elastic Training**:
```python
import torch
import torch.distributed.elastic.rendezvous as rdzv

class ElasticCheckpointer:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, model, optimizer, epoch, step):
        """Save checkpoint with elastic training metadata"""
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': int(os.environ['WORLD_SIZE']),
            'timestamp': time.time()
        }

        # Only rank 0 saves
        if int(os.environ['RANK']) == 0:
            path = f"{self.checkpoint_dir}/checkpoint_epoch{epoch}_step{step}.pt"
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path}")

    def load_latest_checkpoint(self, model, optimizer):
        """Load latest checkpoint and resume training"""
        checkpoints = glob.glob(f"{self.checkpoint_dir}/checkpoint_*.pt")
        if not checkpoints:
            return 0, 0  # Start from scratch

        latest = max(checkpoints, key=os.path.getctime)
        checkpoint = torch.load(latest, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        print(f"Resumed from {latest}: epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint['epoch'], checkpoint['step']

# Training loop with elastic support
def train_elastic():
    checkpointer = ElasticCheckpointer('/checkpoints')

    # Resume from latest checkpoint
    start_epoch, start_step = checkpointer.load_latest_checkpoint(model, optimizer)

    for epoch in range(start_epoch, total_epochs):
        for step, batch in enumerate(train_loader, start=start_step if epoch == start_epoch else 0):
            # Training step
            loss = train_step(model, batch, optimizer)

            # Save checkpoint every N steps
            if step % 100 == 0:
                checkpointer.save_checkpoint(model, optimizer, epoch, step)
```

From [Elastic Training Blog](https://blog.kubeflow.org/elastic%20training/operators/2021/03/15/elastic-training.html) (accessed 2025-11-16):
- Elastic training reduces cost by 30-50% (scale down off-peak)
- Worker join/leave without full job restart
- Rendezvous mechanism coordinates worker discovery
- Checkpoint frequency critical for minimizing lost work

## Section 7: GPU Scheduling and Resource Management

### Gang Scheduling with Volcano

**PyTorchJob with Volcano Gang Scheduling**:
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-gang-scheduled
spec:
  schedulingPolicy:
    schedulerName: volcano
    queue: high-priority
    priorityClassName: training-high-priority
    minAvailable: 4  # All 4 pods must be schedulable together
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          schedulerName: volcano
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/bert-training:v1.0
            resources:
              limits:
                nvidia.com/gpu: 8
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
    Worker:
      replicas: 3
      template:
        spec:
          schedulerName: volcano
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/bert-training:v1.0
            resources:
              limits:
                nvidia.com/gpu: 8
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
```

**Install Volcano on GKE**:
```bash
# Install Volcano
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.9.0/installer/volcano-development.yaml

# Create priority class
cat <<EOF | kubectl apply -f -
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-high-priority
value: 1000000
globalDefault: false
description: "High priority for multi-GPU training jobs"
EOF

# Create Volcano queue
cat <<EOF | kubectl apply -f -
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: high-priority
spec:
  weight: 100
  capability:
    nvidia.com/gpu: 64  # Total GPU quota for queue
EOF
```

From [Volcano Integration Documentation](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/job-scheduling/) (accessed 2025-11-16):
- Gang scheduling ensures all pods scheduled together (prevents deadlock)
- Queue-based resource allocation (fair sharing, priority)
- Preemption support (high-priority jobs evict low-priority)
- GKE Autopilot compatible (with limitations)

### GPU Node Affinity and Taints

**GPU Type Selection**:
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-a100-only
spec:
  pytorchReplicaSpecs:
    Master:
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
          tolerations:
          - key: nvidia.com/gpu
            operator: Equal
            value: a100
            effect: NoSchedule
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: cloud.google.com/gke-accelerator
                    operator: In
                    values:
                    - nvidia-tesla-a100
    Worker:
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
          tolerations:
          - key: nvidia.com/gpu
            operator: Equal
            value: a100
            effect: NoSchedule
```

**Topology-Aware Scheduling** (A3 Mega GPUs):
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-a3-mega
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 8
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-h100-mega-80gb
            topology.kubernetes.io/zone: us-central1-a
          affinity:
            podAntiAffinity:
              preferredDuringSchedulingIgnoredDuringExecution:
              - weight: 100
                podAffinityTerm:
                  labelSelector:
                    matchLabels:
                      training.kubeflow.org/job-name: pytorch-a3-mega
                  topologyKey: kubernetes.io/hostname  # Spread across nodes
```

## Section 8: arr-coc-0-1 Training Operator Integration

### ARR-COC Vision Model Training Configuration

**PyTorchJob for ARR-COC Vervaekean Relevance Training**:
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: arr-coc-vervaeke-training
  namespace: arr-coc-training
  labels:
    app: arr-coc
    component: training
    model-version: "0.1"
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"  # Disable Istio for training
        spec:
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/arr-coc-training:v0.1.0
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=8
            - --nnodes=4
            - --node_rank=0
            - --master_addr=${MASTER_ADDR}
            - --master_port=${MASTER_PORT}
            - arr_coc/training/train.py
            - --config=configs/vervaeke_relevance.yaml
            - --epochs=100
            - --batch_size=32
            - --learning_rate=1e-4
            - --checkpoint_dir=/checkpoints
            - --data_dir=/data/coco2017
            - --wandb_project=arr-coc-gke-training
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: CUDA_VISIBLE_DEVICES
              value: "0,1,2,3,4,5,6,7"
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-api-key
                  key: api-key
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "256Gi"
                ephemeral-storage: "200Gi"
              requests:
                nvidia.com/gpu: 8
                memory: "256Gi"
                cpu: "32"
            volumeMounts:
            - name: checkpoint-storage
              mountPath: /checkpoints
            - name: dataset-storage
              mountPath: /data
          volumes:
          - name: checkpoint-storage
            persistentVolumeClaim:
              claimName: arr-coc-checkpoints
          - name: dataset-storage
            persistentVolumeClaim:
              claimName: coco2017-dataset
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
          - name: pytorch
            image: gcr.io/arr-coc-project/arr-coc-training:v0.1.0
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=8
            - --nnodes=4
            - --master_addr=${MASTER_ADDR}
            - --master_port=${MASTER_PORT}
            - arr_coc/training/train.py
            - --config=configs/vervaeke_relevance.yaml
            - --epochs=100
            - --batch_size=32
            - --learning_rate=1e-4
            - --checkpoint_dir=/checkpoints
            - --data_dir=/data/coco2017
            - --wandb_project=arr-coc-gke-training
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: CUDA_VISIBLE_DEVICES
              value: "0,1,2,3,4,5,6,7"
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-api-key
                  key: api-key
            resources:
              limits:
                nvidia.com/gpu: 8
                memory: "256Gi"
                ephemeral-storage: "200Gi"
              requests:
                nvidia.com/gpu: 8
                memory: "256Gi"
                cpu: "32"
            volumeMounts:
            - name: checkpoint-storage
              mountPath: /checkpoints
            - name: dataset-storage
              mountPath: /data
          volumes:
          - name: checkpoint-storage
            persistentVolumeClaim:
              claimName: arr-coc-checkpoints
          - name: dataset-storage
            persistentVolumeClaim:
              claimName: coco2017-dataset
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
  runPolicy:
    cleanPodPolicy: None
    ttlSecondsAfterFinished: 604800  # Keep for 7 days
    backoffLimit: 3
```

**ARR-COC Training Script** (distributed training):
```python
# arr_coc/training/train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from arr_coc.model import ARRCOCModel
from arr_coc.knowing import InformationScorer, SalienceMapper, QueryContentCoupler
from arr_coc.balancing import TensionBalancer

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank

def train():
    local_rank = setup_distributed()

    # Initialize model
    model = ARRCOCModel(
        patch_size=64,
        lod_levels=[64, 128, 256, 400],
        scorers={
            'information': InformationScorer(),
            'salience': SalienceMapper(),
            'coupling': QueryContentCoupler()
        },
        balancer=TensionBalancer()
    ).cuda(local_rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Initialize W&B (rank 0 only)
    if dist.get_rank() == 0:
        wandb.init(
            project="arr-coc-gke-training",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size * dist.get_world_size(),
                "learning_rate": args.learning_rate,
                "num_gpus": dist.get_world_size()
            }
        )

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        for batch_idx, (images, queries, targets) in enumerate(train_loader):
            images = images.cuda(local_rank, non_blocking=True)
            queries = queries.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass (ARR-COC relevance realization)
            output = model(images, queries)
            loss = criterion(output, targets)

            # Backward pass (gradients synchronized by DDP)
            loss.backward()
            optimizer.step()

            # Log metrics (rank 0 only)
            if dist.get_rank() == 0 and batch_idx % 10 == 0:
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Checkpoint (rank 0 only)
        if dist.get_rank() == 0 and epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f'/checkpoints/arr-coc-epoch{epoch}.pt')

if __name__ == '__main__':
    train()
```

**Launch Training Job**:
```bash
# Create PyTorchJob
kubectl apply -f arr-coc-training-job.yaml

# Monitor status
kubectl get pytorchjob arr-coc-vervaeke-training -n arr-coc-training -w

# Check logs (master pod)
kubectl logs -f arr-coc-vervaeke-training-master-0 -n arr-coc-training

# Check logs (worker pod)
kubectl logs -f arr-coc-vervaeke-training-worker-0 -n arr-coc-training

# View W&B dashboard
wandb login
wandb online
# Navigate to https://wandb.ai/arr-coc-project/arr-coc-gke-training
```

## Sources

**Kubeflow Documentation:**
- [Kubeflow Training Operator Overview](https://www.kubeflow.org/docs/components/trainer/overview/) - Kubeflow Trainer architecture and components (accessed 2025-11-16)
- [PyTorchJob User Guide](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/) - PyTorchJob custom resource and distributed training (accessed 2025-11-16)
- [TensorFlow Training (TFJob)](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/tensorflow/) - TFJob configuration and TensorFlow strategies (accessed 2025-11-16)
- [MPI Training (MPIJob)](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/mpi/) - MPI Operator and Horovod integration (accessed 2025-11-16)
- [Distributed Training Reference](https://www.kubeflow.org/docs/components/trainer/legacy-v1/reference/distributed-training/) - Distributed training patterns and strategies (accessed 2025-11-16)
- [Job Scheduling Documentation](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/job-scheduling/) - Gang scheduling with Volcano (accessed 2025-11-16)

**Framework-Specific Documentation:**
- [PyTorch Distributed Training Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html) - PyTorch DDP and distributed primitives (accessed 2025-11-16)
- [TensorFlow Distributed Training Guide](https://www.tensorflow.org/guide/distributed_training) - TensorFlow distribution strategies (accessed 2025-11-16)

**Kubeflow Community Resources:**
- [Elastic Training Blog Post](https://blog.kubeflow.org/elastic%20training/operators/2021/03/15/elastic-training.html) - Elastic training with MPI Operator (accessed 2025-11-16)
- [KubeCon 2024 NA: Democratizing AI Training](https://static.sched.com/hosted_files/kccncna2024/f4/KubeCon%20NA%202024.%20Democratizing%20AI%20Model%20Training%20on%20Kubernetes%20with%20Kubeflow%20TrainJob%20and%20JobSet.%20Andrey%20Velichkevich%20and%20Yuki%20Iwai.pdf) - Kubeflow Trainer V2 and TrainJob (accessed 2025-11-16)

**Third-Party Guides:**
- [Collabnix: Distributed Training on Kubernetes](https://collabnix.com/distributed-training-on-kubernetes-best-practices-implementation/) - Best practices for Kubeflow Training Operator (accessed 2025-11-16)
- [Google Summer of Code 2024: Kubeflow](https://sandipanpanda.github.io/gsoc2024/) - JAX distributed training on Kubernetes (accessed 2025-11-16)

**GitHub Repositories:**
- [kubeflow/trainer](https://github.com/kubeflow/trainer) - Kubeflow Training Operator V2 source code
- [kubeflow/training-operator](https://github.com/kubeflow/training-operator) - Training Operator V1 (legacy)
- [kubeflow/mpi-operator](https://github.com/kubeflow/mpi-operator) - MPI Operator for Horovod
- [kubernetes-sigs/jobset#570](https://github.com/kubernetes-sigs/jobset/issues/570) - Elastic training autoscaling discussion

**Related Knowledge:**
- [../gcp-gpu/04-multi-gpu-training-patterns.md](../gcp-gpu/04-multi-gpu-training-patterns.md) - Single-node multi-GPU DDP patterns
- [../gcp-gpu/05-multi-node-distributed-training.md](../gcp-gpu/05-multi-node-distributed-training.md) - Multi-node NCCL configuration
- [../gcp-gpu/06-preemptible-spot-gpu-strategies.md](../gcp-gpu/06-preemptible-spot-gpu-strategies.md) - Preemptible GPU checkpoint strategies
