# W&B Launch Cloud Provider Integration

## Overview

W&B Launch integrates with major cloud providers (AWS SageMaker, Google Cloud Vertex AI, Azure ML) to automate training job execution on managed cloud infrastructure. This enables teams to leverage cloud-native compute resources without managing infrastructure directly, while maintaining experiment tracking and orchestration through W&B.

**Key benefits:**
- **No infrastructure management** - Cloud providers handle resource provisioning and scaling
- **Cost optimization** - Pay only for compute used during training
- **Multi-cloud flexibility** - Run jobs across AWS, GCP, or Azure from single Launch interface
- **Unified tracking** - All cloud jobs log to W&B regardless of provider

From [W&B Launch SageMaker Setup](https://docs.wandb.ai/platform/launch/setup-launch-sagemaker) (accessed 2025-01-31):
> "SageMaker takes care of spinning up and releasing compute resources, so it can be a good choice for teams without an EKS cluster."

From [W&B Launch Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
> "Once a launch job is initiated, Vertex AI manages the underlying infrastructure, scaling, and orchestration."

---

## Section 1: AWS SageMaker Integration (~130 lines)

### SageMaker Training Jobs with Launch

W&B Launch executes jobs on AWS SageMaker as SageMaker Training Jobs using the `CreateTrainingJob` API. The Launch agent submits jobs to SageMaker queues, which handle compute provisioning automatically.

**Architecture flow:**
1. Submit job to Launch queue configured for SageMaker
2. Launch agent calls SageMaker `CreateTrainingJob` API
3. SageMaker provisions EC2 instances (ml.m4.xlarge, ml.p3.2xlarge, etc.)
4. Docker container runs on SageMaker-managed infrastructure
5. Training logs/metrics stream to W&B
6. SageMaker releases resources when job completes

From [W&B Launch SageMaker Setup](https://docs.wandb.ai/platform/launch/setup-launch-sagemaker) (accessed 2025-01-31):
> "Launch jobs sent to a W&B Launch queue connected to Amazon SageMaker are executed as SageMaker Training Jobs with the CreateTrainingJob API."

### SageMaker Queue Configuration

**Minimum required configuration:**

```yaml
{
  "RoleArn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 50
  },
  "OutputDataConfig": {
      "S3OutputPath": "s3://my-bucket/sagemaker-outputs"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 86400
  }
}
```

**Key parameters:**
- `RoleArn` - IAM role SageMaker assumes (needs S3, ECR, CloudWatch access)
- `ResourceConfig.InstanceType` - Compute instance type (ml.p3.8xlarge for 4xV100 GPUs)
- `ResourceConfig.VolumeSizeInGB` - EBS volume size for training data
- `OutputDataConfig.S3OutputPath` - Where SageMaker saves model artifacts
- `StoppingCondition.MaxRuntimeInSeconds` - Timeout to prevent runaway costs

### AWS Prerequisites

**Required AWS resources:**

1. **ECR Repository** - Store Docker images for training jobs
   ```bash
   aws ecr create-repository --repository-name ml-training
   ```

2. **S3 Buckets** - Store training data and model outputs
   ```bash
   aws s3 mb s3://company-ml-training-data
   aws s3 mb s3://company-ml-model-outputs
   ```

3. **SageMaker Execution Role** - IAM role with permissions:
   - `sagemaker:CreateTrainingJob`
   - `s3:GetObject` and `s3:PutObject` for data buckets
   - `ecr:GetDownloadUrlForLayer` and `ecr:BatchGetImage` for container pulls
   - `logs:CreateLogStream` and `logs:PutLogEvents` for CloudWatch

4. **Launch Agent IAM Role** - Separate role for Launch agent with:
   - `sagemaker:CreateTrainingJob`
   - `sagemaker:DescribeTrainingJob`
   - `iam:PassRole` (to pass SageMaker execution role)
   - `ecr:*` (if agent builds images)

From [W&B Launch SageMaker Setup](https://docs.wandb.ai/platform/launch/setup-launch-sagemaker) (accessed 2025-01-31):
> "An IAM role for Amazon SageMaker that permits SageMaker to run training jobs and interact with Amazon ECR and Amazon S3."

### Launch Agent Configuration for SageMaker

**~/.config/wandb/launch-config.yaml:**

```yaml
max_jobs: -1  # No limit on concurrent jobs
queues:
  - sagemaker-gpu-queue
environment:
  type: aws
  region: us-west-2
registry:
  type: ecr
  uri: 123456789012.dkr.ecr.us-west-2.amazonaws.com/ml-training
builder:
  type: docker  # Agent builds images and pushes to ECR
```

**Two deployment modes:**

**Mode 1: Agent builds images**
- Agent has Docker daemon access
- Builds images from code, pushes to ECR
- Good for rapid iteration during development

**Mode 2: Pre-built images**
- CI/CD pipeline builds and pushes to ECR
- Agent uses existing images
- Better for production (reproducible builds)

From [W&B Launch SageMaker Setup](https://docs.wandb.ai/platform/launch/setup-launch-sagemaker) (accessed 2025-01-31):
> "There are two options you can choose from: Permit the launch agent build a Docker image, push the image to Amazon ECR, and submit SageMaker Training jobs for you... or the launch agent uses an existing Docker image."

### SageMaker-Specific Features

**Managed Spot Training:**
- Use spot instances for 70-90% cost reduction
- Configure in queue config:
  ```json
  {
    "EnableManagedSpotTraining": true,
    "CheckpointConfig": {
      "S3Uri": "s3://my-bucket/checkpoints",
      "LocalPath": "/opt/ml/checkpoints"
    }
  }
  ```
- SageMaker handles spot interruptions and resumption

**Multi-GPU Training:**
- Single instance: `InstanceType: "ml.p3.8xlarge"` (4xV100)
- Multi-instance distributed: `InstanceCount: 4` for 16 GPUs total
- SageMaker sets up networking between instances

**VolumeKmsKeyId for encryption:**
- Encrypt training data volumes:
  ```json
  {
    "ResourceConfig": {
      "VolumeKmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/abc123"
    }
  }
  ```

---

## Section 2: Google Cloud Vertex AI Integration (~130 lines)

### Vertex AI Custom Jobs with Launch

W&B Launch submits jobs to Vertex AI as Custom Jobs using the `google-cloud-aiplatform` SDK. Vertex AI manages infrastructure orchestration, similar to SageMaker but in GCP's ecosystem.

**Architecture flow:**
1. Submit job to Launch queue configured for Vertex AI
2. Launch agent creates Vertex AI `CustomJob`
3. Vertex AI provisions Compute Engine instances with specified GPUs
4. Container runs on Vertex-managed infrastructure
5. Logs/metrics stream to W&B
6. Vertex AI releases resources when complete

From [W&B Launch Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
> "W&B Launch works with Vertex AI through the CustomJob class in the google-cloud-aiplatform SDK. The parameters of a CustomJob can be controlled with the launch queue configuration."

### Vertex AI Queue Configuration

**Default configuration structure:**

```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
        accelerator_type: NVIDIA_TESLA_V100
        accelerator_count: 1
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}  # Filled by Launch agent
  staging_bucket: gs://my-company-ml-staging
run:
  restart_job_on_worker_restart: false
```

**Key configuration sections:**

**`spec` key** - Maps to `CustomJob` constructor arguments:
- `worker_pool_specs` - List of worker pool configurations
- `staging_bucket` - GCS bucket for Vertex AI metadata (required)
- `service_account` - Custom service account for job execution

**`run` key** - Maps to `CustomJob.run()` method arguments:
- `restart_job_on_worker_restart` - Auto-restart on failure
- `sync` - Wait for job completion (default: false)

From [W&B Launch Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
> "Resource configurations are stored under the spec and run keys: The spec key contains values for the named arguments of the CustomJob constructor... The run key contains values for the named arguments of the run method."

### Worker Pool Specifications

**Single GPU training:**

```yaml
worker_pool_specs:
  - machine_spec:
      machine_type: n1-standard-8
      accelerator_type: NVIDIA_TESLA_T4
      accelerator_count: 1
    replica_count: 1
```

**Multi-GPU single node (4xA100):**

```yaml
worker_pool_specs:
  - machine_spec:
      machine_type: a2-highgpu-4g
      accelerator_type: NVIDIA_TESLA_A100
      accelerator_count: 4
    replica_count: 1
```

**Multi-node distributed (8 nodes, 8 GPUs each):**

```yaml
worker_pool_specs:
  - machine_spec:
      machine_type: a2-highgpu-8g
      accelerator_type: NVIDIA_TESLA_A100
      accelerator_count: 8
    replica_count: 8  # 8 nodes = 64 GPUs total
```

**Available machine types:**
- `n1-standard-{4,8,16,32}` - General purpose CPU
- `n1-highmem-{8,16,32}` - Memory-optimized
- `a2-highgpu-{1,2,4,8}g` - A100 GPU instances
- `g2-standard-{4,8,12,16}` - L4 GPU instances

**Available accelerator types:**
- `NVIDIA_TESLA_K80` - Legacy
- `NVIDIA_TESLA_T4` - Cost-effective inference/training
- `NVIDIA_TESLA_V100` - Training workhorse
- `NVIDIA_TESLA_A100` - Latest, highest performance
- `NVIDIA_L4` - Newer, efficient

### Google Cloud Prerequisites

**Required GCP resources:**

1. **Enable Vertex AI API** in your GCP project
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

2. **Create Artifact Registry repository** (GCP's container registry)
   ```bash
   gcloud artifacts repositories create ml-images \
     --repository-format=docker \
     --location=us-west1
   ```

3. **Create GCS staging bucket** (must be same region as Vertex jobs)
   ```bash
   gcloud storage buckets create gs://company-vertex-staging \
     --location=us-west1
   ```

4. **Service account with permissions:**
   - `aiplatform.customJobs.create`
   - `aiplatform.customJobs.get`
   - `aiplatform.customJobs.list`
   - `storage.buckets.get` and `storage.objects.*` for GCS
   - `artifactregistry.repositories.downloadArtifacts` for container pulls

From [W&B Launch Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
> "Grant your service account permission to manage Vertex jobs" with the specified permissions for custom jobs.

### Container Registry Considerations

**Important constraint:**

From [W&B Launch Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
> "Vertex AI cannot be configured to pull images from a private registry outside of Google Cloud. This means that you must store container images in Google Cloud or in a public registry."

**Options:**
1. **Google Artifact Registry** (recommended) - Native integration
2. **Public registries** (Docker Hub, etc.) - Less secure
3. **Google Container Registry (GCR)** - Deprecated, use Artifact Registry

### Launch Agent Configuration for Vertex AI

**~/.config/wandb/launch-config.yaml:**

```yaml
max_jobs: 5
queues:
  - vertex-ai-gpu-queue
environment:
  type: gcp
  region: us-west1
  project: my-gcp-project-id
registry:
  type: gcr
  uri: us-west1-docker.pkg.dev/my-project/ml-images
```

**Authentication methods:**
1. **Workload Identity** (recommended for GKE-deployed agents)
2. **Service Account JSON** - Download key, set `GOOGLE_APPLICATION_CREDENTIALS`
3. **gcloud CLI** - `gcloud auth application-default login`

### Vertex AI-Specific Features

**Custom service accounts:**
```yaml
spec:
  service_account: training-jobs@my-project.iam.gserviceaccount.com
```

**VPC configuration for private networking:**
```yaml
spec:
  network: projects/123456/global/networks/ml-vpc
  reserved_ip_ranges:
    - vertex-ai-peering-range
```

**Automatic restarts on failure:**
```yaml
run:
  restart_job_on_worker_restart: true
```

---

## Section 3: Multi-Cloud Orchestration & Strategy (~140 lines)

### W&B Dedicated Cloud Multi-Provider Support

W&B offers Dedicated Cloud deployments on AWS, GCP, or Azure - single-tenant instances in W&B's cloud accounts with isolated networking, compute, and storage.

From [W&B Dedicated Cloud](https://docs.wandb.ai/guides/hosting/hosting-options/dedicated_cloud) (accessed 2025-01-31):
> "W&B Dedicated Cloud is a single-tenant, fully-managed platform deployed in W&B's AWS, GCP or Azure cloud accounts. Each Dedicated Cloud instance has its own isolated network, compute and storage from other W&B Dedicated Cloud instances."

**Available configurations:**
- **AWS Dedicated Cloud** - Deploy W&B on AWS, run Launch jobs on SageMaker
- **GCP Dedicated Cloud** - Deploy W&B on GCP, run Launch jobs on Vertex AI
- **Azure Dedicated Cloud** - Deploy W&B on Azure, run Launch jobs on Azure ML

**Key benefit:** Your W&B instance and training compute can live in the same cloud provider for lower latency, simpler networking, and unified billing.

### Cloud Provider Selection Criteria

**Choose AWS SageMaker when:**
- Existing AWS infrastructure and expertise
- Using AWS services (S3, ECR, IAM) already
- Need managed spot training for cost optimization
- Team familiar with SageMaker ecosystem
- 32% market share leader (as of 2024)

From [AWS vs Azure vs GCP Comparison](https://www.datacamp.com/blog/aws-vs-azure-vs-gcp) (accessed 2025-01-31):
> "AWS is the leader in cloud computing with around 32% of the market share as of 2024."

**Choose GCP Vertex AI when:**
- Using Google Cloud Platform services
- Prefer Google's ML ecosystem (TensorFlow, TPUs)
- Need cutting-edge GPU instances (A100, L4)
- Simpler pricing model preferred
- Integration with BigQuery, Cloud Storage

**Choose Azure ML when:**
- Microsoft-centric enterprise environment
- Active Directory/Entra ID integration needed
- Using Azure services (Blob Storage, ACR)
- Enterprise agreements with Microsoft
- Integration with Power BI, Office 365

From [Azure OpenAI W&B Integration](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/weights-and-biases-integration) (accessed 2025-01-31):
> "Azure OpenAI fine-tuning integrates with W&B, allowing you to track metrics, parameters, and visualize your Azure OpenAI fine-tuning training runs."

### Multi-Cloud Job Routing Strategy

**Scenario: Team uses multiple cloud providers**

Example setup with 3 Launch queues:

```yaml
# Queue 1: AWS SageMaker for production LLM training
name: production-llm-aws
resource: sagemaker
config:
  ResourceConfig:
    InstanceType: ml.p4d.24xlarge  # 8xA100 80GB
    InstanceCount: 4  # 32 GPUs total
priority: high

# Queue 2: GCP Vertex AI for experimental VLM training
name: experimental-vlm-gcp
resource: vertex-ai
config:
  spec:
    worker_pool_specs:
      - machine_spec:
          machine_type: a2-highgpu-2g
          accelerator_count: 2  # 2xA100
priority: medium

# Queue 3: Azure ML for compliance-sensitive workloads
name: compliance-azure
resource: azure-ml
priority: high
```

**Routing logic:**
1. **Production workloads** → AWS (most mature, proven at scale)
2. **Experimental research** → GCP (latest GPUs, TPU access)
3. **Compliance/regulated** → Azure (enterprise controls, AD integration)

### Cost Optimization Across Providers

**Spot/Preemptible instances:**

**AWS SageMaker Managed Spot:**
- 70-90% cost reduction vs on-demand
- SageMaker handles interruptions automatically
- Configure checkpointing for resilience:
  ```json
  {
    "EnableManagedSpotTraining": true,
    "MaxWaitTimeInSeconds": 172800,  # Wait 2 days for spot capacity
    "CheckpointConfig": {
      "S3Uri": "s3://bucket/checkpoints"
    }
  }
  ```

**GCP Preemptible VMs on Vertex AI:**
- Up to 80% cost reduction
- Must handle interruptions in code
- Max 24 hour runtime
- Good for fault-tolerant workloads

**Azure Spot VMs:**
- Up to 90% savings
- Priority-based eviction
- Less mature than AWS/GCP spot offerings

**Cost comparison (approximate):**

From general cloud pricing research (2024-2025):

| Resource | AWS SageMaker | GCP Vertex AI | Azure ML |
|----------|---------------|---------------|----------|
| 8xV100 GPUs (on-demand) | ~$24/hr | ~$20/hr | ~$22/hr |
| 8xV100 GPUs (spot) | ~$7/hr | ~$6/hr | ~$8/hr |
| 8xA100 80GB (on-demand) | ~$40/hr | ~$35/hr | ~$38/hr |

**Recommendation:** Use spot/preemptible for long training runs with checkpointing, on-demand for time-sensitive or short jobs.

### Private Connectivity Options

**AWS PrivateLink:**
- Connect on-premises network to SageMaker via private VPC
- No traffic over public internet
- Launch agent runs in customer VPC

**GCP Private Service Connect:**
- Private connection to Vertex AI services
- Traffic stays within Google network
- Agent in customer VPC/GKE cluster

**Azure Private Link:**
- Private endpoints for Azure ML workspace
- No public IP exposure
- Agent in customer VNet

From [W&B Dedicated Cloud](https://docs.wandb.ai/guides/hosting/hosting-options/dedicated_cloud) (accessed 2025-01-31):
> "You can connect privately to your Dedicated Cloud instance using cloud provider's secure connectivity solution."

### Unified Monitoring Across Clouds

**W&B as single pane of glass:**

All cloud provider jobs log to W&B:
- **Metrics** - Training loss, accuracy, GPU utilization
- **System metrics** - Memory, CPU, GPU temperature
- **Artifacts** - Model checkpoints, datasets
- **Job metadata** - Instance type, region, cost estimates

```python
# Training script works identically on all clouds
import wandb

wandb.init(project="multi-cloud-training")
wandb.config.update({"cloud": "aws", "instance": "ml.p3.8xlarge"})

for epoch in range(num_epochs):
    loss = train_epoch()
    wandb.log({"loss": loss, "epoch": epoch})

wandb.save("model.pt")  # Saved to W&B, works on any cloud
```

**Cloud-agnostic training code:**
- No provider-specific code in training scripts
- W&B SDK abstracts cloud differences
- Same `wandb.log()` calls across AWS, GCP, Azure

### Data Locality and Transfer Costs

**Egress charges:**
- **AWS:** $0.09/GB for data leaving AWS regions
- **GCP:** $0.12/GB for data leaving GCP
- **Azure:** $0.087/GB for data leaving Azure

**Strategy to minimize costs:**
1. **Collocate data and compute** - Store training data in same region as jobs
2. **Use Bring Your Own Bucket (BYOB)** - W&B artifacts stored in your cloud
3. **Avoid cross-cloud transfers** - Don't download 1TB dataset from AWS to train on GCP

From [W&B Dedicated Cloud](https://docs.wandb.ai/guides/hosting/hosting-options/dedicated_cloud) (accessed 2025-01-31):
> "You can bring your own bucket (BYOB) using the secure storage connector at the instance and team levels to store your files such as models, datasets, and more."

**BYOB configuration example:**

```yaml
# Team A uses AWS S3
team: team-a-aws
storage:
  type: s3
  bucket: team-a-training-artifacts
  region: us-west-2

# Team B uses GCS
team: team-b-gcp
storage:
  type: gcs
  bucket: team-b-training-artifacts
  region: us-west1
```

### Migration and Portability

**Switching cloud providers:**

W&B Launch makes cloud migrations easier:

1. **Job definitions are cloud-agnostic** - Same Docker image can run on any provider
2. **Queue configurations are swappable** - Change queue target, keep job code
3. **Experiment history preserved** - All past runs stay in W&B regardless of cloud

**Migration workflow:**

```bash
# Step 1: Run job on AWS
wandb launch --queue aws-sagemaker-gpu --project migration-test

# Step 2: Create equivalent GCP queue with same config
# (adjust instance types, but logic identical)

# Step 3: Run same job on GCP
wandb launch --queue gcp-vertex-gpu --project migration-test

# Step 4: Compare results in W&B (same project, different runs)
```

**Challenges:**
- GPU availability varies by cloud (not all have H100s)
- Instance type naming differs (ml.p3.8xlarge vs a2-highgpu-4g)
- IAM/permissions differ significantly

**Best practice:** Test jobs on new cloud provider with small-scale runs before full migration.

### Compliance and Data Residency

**Region selection for compliance:**

**GDPR (EU data):**
- AWS: eu-west-1 (Ireland), eu-central-1 (Frankfurt)
- GCP: europe-west1 (Belgium), europe-west4 (Netherlands)
- Azure: westeurope (Netherlands), northeurope (Ireland)

**HIPAA (US healthcare):**
- AWS: SageMaker HIPAA-eligible in most US regions
- GCP: Vertex AI with HIPAA BAA available
- Azure ML: HIPAA compliance in Azure Government

**Data sovereignty requirements:**
- Configure W&B Dedicated Cloud in required region
- Use BYOB to keep data in compliant storage
- Ensure Launch jobs run in same geography

From [W&B Dedicated Cloud](https://docs.wandb.ai/guides/hosting/hosting-options/dedicated_cloud) (accessed 2025-01-31):
> "W&B Dedicated Cloud is available in multiple global regions for each cloud provider."

**Available Dedicated Cloud regions:**
- AWS: us-east-1, us-west-2, eu-west-1, ap-southeast-1
- GCP: us-central1, europe-west1, asia-northeast1
- Azure: eastus, westeurope, australiaeast

---

## Sources

**W&B Documentation:**
- [W&B Launch SageMaker Setup](https://docs.wandb.ai/platform/launch/setup-launch-sagemaker) - AWS SageMaker integration guide (accessed 2025-01-31)
- [W&B Launch Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) - Google Cloud Vertex AI integration (accessed 2025-01-31)
- [W&B Dedicated Cloud](https://docs.wandb.ai/guides/hosting/hosting-options/dedicated_cloud) - Multi-cloud deployment options (accessed 2025-01-31)

**Cloud Provider Documentation:**
- [Azure OpenAI W&B Integration](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/weights-and-biases-integration) - Microsoft Azure integration (accessed 2025-01-31)
- [AWS SageMaker CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) - SageMaker API reference
- [Vertex AI CustomJob Documentation](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) - GCP Vertex AI jobs

**Cloud Comparison Research:**
- [AWS vs Azure vs GCP Comparison](https://www.datacamp.com/blog/aws-vs-azure-vs-gcp) - DataCamp cloud provider analysis (accessed 2025-01-31, February 2025 publication)

**Additional References:**
- [W&B Launch Walkthrough](https://docs.wandb.ai/platform/launch/walkthrough) - Basic Launch concepts
- [W&B Launch Kubernetes Setup](https://docs.wandb.ai/platform/launch/setup-launch-kubernetes) - Alternative deployment option
- [W&B Secure Storage Connector](https://docs.wandb.ai/platform/hosting/data-security/secure-storage-connector) - BYOB configuration
