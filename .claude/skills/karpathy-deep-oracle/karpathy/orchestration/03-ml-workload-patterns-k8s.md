# Kubernetes ML Workload Patterns: Batch Jobs, CronJobs, Gang Scheduling, and Job Queues

**Knowledge File**: ML-specific workload patterns on Kubernetes - batch jobs, recurring tasks, gang scheduling for multi-GPU training, job queues, and priority scheduling

---

## Overview

Machine learning workloads on Kubernetes require specialized scheduling patterns beyond standard web services. This guide covers Kubernetes-native batch processing, gang scheduling for distributed training, and advanced job queue management with tools like Volcano scheduler.

**Key ML Workload Types:**
- One-time batch jobs (model training, batch inference)
- Recurring scheduled tasks (feature generation, model retraining)
- Distributed multi-GPU jobs (requires gang scheduling)
- Priority-based job queues (production vs research workloads)

From [Kubernetes Official Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/) (accessed 2025-11-13):
> "A Job creates one or more Pods and will continue to retry execution of the Pods until a specified number of them successfully terminate."

---

## Section 1: Kubernetes Batch Jobs for ML (~80 lines)

### Job vs Pod: Understanding the Difference

**Pod**: Single unit of compute that runs containers. If a pod fails, it's gone.

**Job**: Ensures a task completes successfully by:
- Creating and retrying pods until success
- Tracking completion status
- Handling node failures gracefully

From [ML in Production](https://mlinproduction.com/k8s-jobs/) (accessed 2025-11-13):
> "Jobs allow us to reliably run batch processes in a fault tolerant way. Even if an underlying node in the cluster fails, Kubernetes will ensure that the Job is rescheduled on a new node."

### Basic Job Pattern for Model Training

**Training Job Example:**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
  labels:
    workload-type: ml-training
    model: resnet50
    experiment: exp-042
spec:
  completions: 1           # Run once to completion
  parallelism: 1           # Single pod (for single-GPU training)
  backoffLimit: 4          # Retry up to 4 times on failure
  template:
    spec:
      restartPolicy: Never  # Critical: Jobs require Never or OnFailure
      containers:
      - name: trainer
        image: pytorch/pytorch:latest
        command: ["python", "train.py"]
        args:
        - --epochs=100
        - --batch-size=64
        - --model=resnet50
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret
              key: api-key
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
            cpu: "6"
        volumeMounts:
        - name: training-data
          mountPath: /data
        - name: checkpoints
          mountPath: /checkpoints
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: imagenet-dataset
      - name: checkpoints
        persistentVolumeClaim:
          claimName: model-checkpoints
```

### Batch Inference Pattern

**Daily Batch Inference Job:**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: batch-inference-daily
spec:
  completions: 1
  parallelism: 1
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: inference
        image: ml-inference:latest
        command: ["python", "inference.py"]
        args:
        - --input=/data/new-samples.csv
        - --output=/results/predictions.csv
        - --model=/models/latest.pt
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
        volumeMounts:
        - name: input-data
          mountPath: /data
          readOnly: true
        - name: model-storage
          mountPath: /models
          readOnly: true
        - name: results
          mountPath: /results
      volumes:
      - name: input-data
        persistentVolumeClaim:
          claimName: inference-data
      - name: model-storage
        persistentVolumeClaim:
          claimName: production-models
      - name: results
        persistentVolumeClaim:
          claimName: inference-results
```

### Key Job Configuration Fields

From [Kubernetes Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/):

**`.spec.completions`**: Number of successful completions required (default: 1)
**`.spec.parallelism`**: Maximum pods running concurrently (default: 1)
**`.spec.backoffLimit`**: Number of retries before marking job as failed (default: 6)
**`.spec.activeDeadlineSeconds`**: Maximum time job can run before termination
**`.spec.ttlSecondsAfterFinished`**: Automatic cleanup after completion (available in Kubernetes 1.23+)

---

## Section 2: CronJobs for Recurring ML Tasks (~100 lines)

### CronJob Fundamentals

CronJobs create Jobs on a time-based schedule, similar to Linux cron. Essential for:
- Periodic feature generation
- Scheduled model retraining
- Regular batch inference
- Dataset refresh workflows

From [ML in Production](https://mlinproduction.com/k8s-cronjobs/) (accessed 2025-11-13):
> "CronJobs are quite useful in machine learning workflows. Suppose you're building a feature store and need to generate features every hour from an operational data store. One way of producing these features is to use an hourly CronJob."

### CronJob Schedule Format

```
# ┌───────────── minute (0 - 59)
# │ ┌───────────── hour (0 - 23)
# │ │ ┌───────────── day of month (1 - 31)
# │ │ │ ┌───────────── month (1 - 12)
# │ │ │ │ ┌───────────── day of week (0 - 6) (Sunday=0)
# │ │ │ │ │
# * * * * *

*/15 * * * *     # Every 15 minutes
0 2 * * *        # Daily at 2 AM
0 0 * * 0        # Weekly on Sunday midnight
0 0 1 * *        # Monthly on the 1st
```

### Feature Generation CronJob

**Hourly Feature Store Update:**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: feature-generation-hourly
spec:
  schedule: "0 * * * *"  # Every hour at minute 0
  concurrencyPolicy: Forbid  # Don't overlap runs
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: feature-generator
            image: feature-store:latest
            command: ["python", "generate_features.py"]
            args:
            - --source=postgres://db:5432/production
            - --target=redis://cache:6379/features
            - --window=1h
            resources:
              limits:
                memory: "8Gi"
                cpu: "4"
              requests:
                memory: "6Gi"
                cpu: "3"
            env:
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: password
```

### Weekly Model Retraining

**Sunday Night Retraining Job:**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-model-retrain
spec:
  schedule: "0 2 * * 0"  # Sunday 2 AM
  concurrencyPolicy: Replace  # Cancel old job if still running
  startingDeadlineSeconds: 3600  # Skip if delayed > 1 hour
  jobTemplate:
    spec:
      backoffLimit: 2
      template:
        metadata:
          labels:
            workload: weekly-retrain
        spec:
          restartPolicy: Never
          containers:
          - name: retrainer
            image: ml-trainer:latest
            command: ["python", "retrain.py"]
            args:
            - --data=/data/weekly
            - --checkpoint=/models/current.pt
            - --output=/models/weekly-retrain.pt
            - --epochs=50
            resources:
              limits:
                nvidia.com/gpu: 2
                memory: "64Gi"
                cpu: "16"
          nodeSelector:
            gpu-type: "A100"
```

### Daily Batch Inference CronJob

From [ML in Production](https://mlinproduction.com/k8s-cronjobs/):

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-batch-inference
spec:
  schedule: "0 0 * * *"  # Midnight daily
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: inference
            image: lpatruno/k8-model:latest
            command: ["python3", "inference.py"]
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
          restartPolicy: Never
      backoffLimit: 0  # Don't retry failed inference
```

### CronJob Concurrency Policies

From [Kubernetes Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/):

**Allow** (default): Multiple jobs can run concurrently
**Forbid**: Skip new run if previous job still running
**Replace**: Cancel current job and start new one

**Best Practice for ML:**
- Use `Forbid` for long-running training jobs
- Use `Replace` for time-sensitive inference
- Use `Allow` only for independent, short tasks

---

## Section 3: Gang Scheduling for Multi-GPU Training (~100 lines)

### The Gang Scheduling Problem

**Problem**: Distributed training requires ALL worker pods to start simultaneously. Default Kubernetes scheduler operates pod-by-pod, leading to:
- Partial resource allocation (deadlock)
- Resource fragmentation
- Wasted GPU time waiting for all replicas

From [InfraCloud Blog](https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/) (accessed 2025-11-13):
> "Distributed tasks like ML model training require all associated pods to start simultaneously. Without coordination, resource wastage becomes inevitable. A good tool provides gang scheduling, ensuring simultaneous resource allocation for all tasks in a job to avoid partial execution."

**Gang Scheduling**: All-or-nothing pod scheduling. Either ALL pods in a job get resources, or NONE do.

### Volcano Gang Scheduling Example

From [Volcano Documentation](https://volcano.sh/en/docs/vcjob/) (accessed 2025-11-13):

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: distributed-training-tf
spec:
  minAvailable: 4        # Gang scheduling: all 4 pods required
  schedulerName: volcano
  queue: ml-training
  plugins:
    ssh: []
    env: []
    svc: []
  policies:
  - event: PodEvicted
    action: RestartJob
  tasks:
  - replicas: 1
    name: ps              # Parameter server
    template:
      spec:
        containers:
        - name: tensorflow
          image: tensorflow/tensorflow:latest-gpu
          command:
          - sh
          - -c
          - |
            PS_HOST=`cat /etc/volcano/ps.host | sed 's/$/&:2222/g'`;
            WORKER_HOST=`cat /etc/volcano/worker.host | sed 's/$/&:2222/g'`;
            export TF_CONFIG="{\"cluster\":{\"ps\":[${PS_HOST}],\"worker\":[${WORKER_HOST}]}}";
            python train_distributed.py
          ports:
          - containerPort: 2222
            name: tf-port
          resources:
            limits:
              nvidia.com/gpu: 1
        restartPolicy: Never
  - replicas: 3
    name: worker          # 3 worker pods
    policies:
    - event: TaskCompleted
      action: CompleteJob
    template:
      spec:
        containers:
        - name: tensorflow
          image: tensorflow/tensorflow:latest-gpu
          command:
          - sh
          - -c
          - |
            PS_HOST=`cat /etc/volcano/ps.host | sed 's/$/&:2222/g'`;
            WORKER_HOST=`cat /etc/volcano/worker.host | sed 's/$/&:2222/g'`;
            export TF_CONFIG="{\"cluster\":{\"ps\":[${PS_HOST}],\"worker\":[${WORKER_HOST}]}}";
            python train_distributed.py
          ports:
          - containerPort: 2222
            name: tf-port
          resources:
            limits:
              nvidia.com/gpu: 2
        restartPolicy: Never
```

**Key Volcano Fields:**

**`minAvailable`**: Minimum pods required before job runs (gang scheduling threshold)
**`schedulerName: volcano`**: Use Volcano scheduler instead of default
**`queue`**: Job queue for priority and resource management
**`tasks`**: Different roles (ps, worker, chief) with separate configurations

### PyTorch DDP with Gang Scheduling

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pytorch-ddp-resnet
spec:
  minAvailable: 8        # All 8 workers must start together
  schedulerName: volcano
  queue: gpu-training
  tasks:
  - replicas: 8
    name: worker
    template:
      spec:
        containers:
        - name: pytorch
          image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
          command:
          - python
          - -m
          - torch.distributed.run
          args:
          - --nproc_per_node=1
          - --nnodes=8
          - --node_rank=$(VK_TASK_INDEX)
          - --master_addr=$(VC_TASK_0_HOSTNAME)
          - --master_port=23456
          - train.py
          env:
          - name: NCCL_DEBUG
            value: "INFO"
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "32Gi"
              cpu: "8"
        restartPolicy: Never
```

### Why Gang Scheduling Matters

From [InfraCloud Blog](https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/):

**Without Gang Scheduling:**
- Job requests 8 GPUs across 8 pods
- Only 6 GPUs available
- Scheduler allocates 6 pods → 2 pods pending
- 6 GPUs idle waiting for remaining 2
- Resource deadlock and waste

**With Gang Scheduling:**
- Job requests 8 GPUs (minAvailable: 8)
- Only 6 GPUs available
- Volcano doesn't schedule ANY pods
- All 6 GPUs remain available for other jobs
- When 8 GPUs free, ALL 8 pods scheduled simultaneously

---

## Section 4: Job Queues and Priority Scheduling (~80 lines)

### Queue-Based Resource Management

From [Volcano Documentation](https://volcano.sh/en/docs/vcjob/):
> "Volcano supports queue-based resource management, where multiple PodGroups are placed in queues and scheduled based on priority, resource requirements, and policies."

### Creating Volcano Queues

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: production-inference
spec:
  weight: 100           # High priority queue
  capability:
    cpu: "200"
    memory: "800Gi"
    nvidia.com/gpu: "20"
  reclaimable: false    # Can't be preempted by other queues
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: research-training
spec:
  weight: 50            # Medium priority
  capability:
    cpu: "400"
    memory: "1600Gi"
    nvidia.com/gpu: "40"
  reclaimable: true     # Can be preempted if production needs resources
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: exploratory
spec:
  weight: 10            # Low priority
  capability:
    cpu: "100"
    memory: "400Gi"
    nvidia.com/gpu: "10"
  reclaimable: true
```

### Assigning Jobs to Queues

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: critical-inference
spec:
  queue: production-inference  # High priority queue
  minAvailable: 1
  schedulerName: volcano
  tasks:
  - replicas: 3
    name: inference-server
    template:
      spec:
        containers:
        - name: server
          image: inference-api:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

### Priority Classes for Preemption

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-training
value: 1000000          # Higher value = higher priority
globalDefault: false
description: "Critical production training jobs"
preemptionPolicy: PreemptLowerPriority
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: medium-priority-research
value: 100000
globalDefault: false
description: "Research training jobs"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority-experiments
value: 10000
globalDefault: true     # Default for unlabeled jobs
description: "Experimental workloads"
```

**Using Priority Classes:**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: production-model-training
spec:
  template:
    spec:
      priorityClassName: high-priority-training
      containers:
      - name: trainer
        image: production-trainer:latest
        resources:
          limits:
            nvidia.com/gpu: 4
```

### Multi-Tenant Queue Management

From [InfraCloud Blog](https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/):

**Scenario**: Company with 100 GPUs shared between teams:
- NLP team: 50% reserved (50 GPUs)
- Data Science: 30% reserved (30 GPUs)
- Analytics: 20% reserved (20 GPUs)

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: nlp-team
spec:
  weight: 50
  capability:
    nvidia.com/gpu: "50"
  reclaimable: false    # NLP always gets their 50 GPUs
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: data-science-team
spec:
  weight: 30
  capability:
    nvidia.com/gpu: "30"
  reclaimable: true     # Can use idle GPUs from other queues
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: analytics-team
spec:
  weight: 20
  capability:
    nvidia.com/gpu: "20"
  reclaimable: true
```

---

## Section 5: Resource Management Patterns (~40 lines)

### Job-Level Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-training-quota
  namespace: ml-workloads
spec:
  hard:
    requests.nvidia.com/gpu: "16"    # Max 16 GPUs in namespace
    limits.nvidia.com/gpu: "16"
    requests.memory: "256Gi"
    requests.cpu: "128"
    count/jobs.batch: "50"           # Max 50 concurrent jobs
    count/cronjobs.batch: "20"       # Max 20 CronJobs
```

### LimitRange for Job Defaults

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-job-limits
  namespace: ml-workloads
spec:
  limits:
  - max:
      nvidia.com/gpu: "8"            # Max 8 GPUs per pod
      memory: "128Gi"
      cpu: "64"
    min:
      nvidia.com/gpu: "1"            # Min 1 GPU required
      memory: "8Gi"
      cpu: "4"
    default:
      memory: "32Gi"                 # Default if not specified
      cpu: "16"
    type: Container
```

### Preventing GPU Node Contamination

**Taint GPU nodes** to prevent non-GPU workloads:

```bash
# Taint all GPU nodes
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
kubectl taint nodes gpu-node-2 nvidia.com/gpu=present:NoSchedule
```

**ML jobs tolerate the taint:**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-training
spec:
  template:
    spec:
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"
      containers:
      - name: trainer
        image: pytorch:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## arr-coc-0-1 Use Cases

### Weekly Model Retraining CronJob

ARR-COC VLM model benefits from periodic retraining on new data:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: arr-coc-weekly-retrain
  namespace: ml-production
spec:
  schedule: "0 3 * * 0"  # Sunday 3 AM
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: trainer
            image: arr-coc-trainer:latest
            command: ["python", "training/retrain.py"]
            args:
            - --checkpoint=/models/current/arr-coc-latest.pt
            - --data=/data/weekly-vqa
            - --output=/models/weekly/arr-coc-retrained.pt
            - --epochs=20
            - --config=configs/weekly-retrain.yaml
            resources:
              limits:
                nvidia.com/gpu: 4
                memory: "128Gi"
                cpu: "32"
          volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: arr-coc-models
          - name: training-data
            persistentVolumeClaim:
              claimName: vqa-dataset-weekly
          nodeSelector:
            gpu-type: "A100"
```

### Distributed Multi-GPU Training with Volcano

Training ARR-COC with 8 GPUs using gang scheduling:

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: arr-coc-distributed-training
spec:
  minAvailable: 8        # All 8 workers required
  schedulerName: volcano
  queue: arr-coc-training
  plugins:
    ssh: []
    svc: []
  policies:
  - event: PodEvicted
    action: RestartJob
  tasks:
  - replicas: 8
    name: worker
    template:
      spec:
        containers:
        - name: pytorch-worker
          image: arr-coc-trainer:ddp
          command:
          - python
          - -m
          - torch.distributed.run
          args:
          - --nproc_per_node=1
          - --nnodes=8
          - --node_rank=$(VK_TASK_INDEX)
          - --master_addr=$(VC_TASK_0_HOSTNAME)
          - --master_port=29500
          - training/train_distributed.py
          - --config=configs/arr-coc-vlm.yaml
          env:
          - name: NCCL_DEBUG
            value: "INFO"
          - name: CUDA_VISIBLE_DEVICES
            value: "0"
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "64Gi"
              cpu: "16"
        restartPolicy: Never
```

---

## Sources

**Official Documentation:**
- [Kubernetes Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/) - Job controller documentation (accessed 2025-11-13)
- [Kubernetes CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/) - CronJob controller documentation (accessed 2025-11-13)
- [Volcano Documentation](https://volcano.sh/en/docs/vcjob/) - VolcanoJob specifications (accessed 2025-11-13)

**Web Research:**
- [ML in Production: Kubernetes Jobs for Machine Learning](https://mlinproduction.com/k8s-jobs/) (accessed 2025-11-13)
  - Practical Job patterns for ML workloads
  - Batch inference examples
- [ML in Production: Kubernetes CronJobs for Machine Learning](https://mlinproduction.com/k8s-cronjobs/) (accessed 2025-11-13)
  - Feature store generation with CronJobs
  - Scheduled batch inference patterns
  - CronJob configuration best practices
- [InfraCloud: Batch Scheduling on Kubernetes](https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/) (accessed 2025-11-13)
  - Gang scheduling fundamentals
  - Comparison of Volcano, YuniKorn, and Kueue
  - Multi-tenant resource management
  - Priority scheduling patterns

**Additional References:**
- NCCL (NVIDIA Collective Communications Library) for distributed training
- PyTorch DistributedDataParallel (DDP) for multi-GPU training
- Volcano scheduler for advanced batch scheduling
