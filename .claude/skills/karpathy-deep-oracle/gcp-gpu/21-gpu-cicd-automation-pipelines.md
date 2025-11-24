# GPU CI/CD & Automation Pipelines

Comprehensive guide to building automated CI/CD pipelines for GPU workloads on GCP, covering Cloud Build GPU runners, GitHub Actions integration, automated testing, container optimization, and deployment automation for machine learning models.

## Overview

GPU CI/CD pipelines extend traditional software CI/CD with ML-specific requirements: model training automation, GPU container builds, automated testing on GPU infrastructure, and deployment strategies for models requiring GPU acceleration. Unlike CPU-based pipelines, GPU workflows require specialized infrastructure, container configurations, and resource management.

From [CI/CD for Machine Learning in 2024](https://medium.com/infer-qwak/ci-cd-for-machine-learning-in-2024-best-practices-to-build-test-and-deploy-c4ad869824d2) (accessed 2025-11-16):
> ML systems demand consistent monitoring for performance and data drift. When model accuracy dips below a set baseline or data experiences concept drift, the entire system must undergo another cycle. This means replicating all steps, from data validation to model training and evaluation, testing, and deployment.

**Key Differences from Traditional CI/CD:**
- **GPU Resource Allocation**: Training jobs require GPU-enabled workers
- **Container Image Size**: CUDA base images add 2-4GB overhead
- **Test Duration**: GPU model validation can take hours vs. seconds
- **Cost Management**: GPU runners cost 10-20x more than CPU equivalents

From [gcloud-cicd/00-pipeline-integration.md](../gcloud-cicd/00-pipeline-integration.md):
> Modern ML production systems require seamless integration of CI/CD pipelines with training infrastructure. This integration enables continuous training, GitOps workflows, automated testing, progressive rollouts, and end-to-end pipeline monitoring.

---

## Section 1: Cloud Build with GPU Custom Worker Pools

Cloud Build supports GPU-accelerated builds through custom private worker pools with attached GPUs. This enables building and testing GPU containers, running model validation, and executing integration tests requiring GPU access.

### Worker Pool Configuration

**Create GPU-Enabled Worker Pool:**

```bash
# Create private worker pool with GPU configuration
gcloud builds worker-pools create gpu-builder-pool \
  --region=us-west2 \
  --worker-machine-type=n1-standard-8 \
  --worker-disk-size=100GB \
  --no-public-egress

# Attach GPU to worker pool (requires custom VM configuration)
# Note: Cloud Build doesn't directly support GPU attachment
# Workaround: Use Compute Engine VMs with GPUs as custom workers
```

**Alternative: Compute Engine as Build Workers:**

From [Cloud Build GPU runners custom worker pools 2024 2025](https://docs.cloud.google.com/run/docs/configuring/services/build-worker-pools) (accessed 2025-11-16):
> Custom worker pools allow you to configure the build environment with specific machine types, network settings, and persistent disk configurations.

```bash
# Create GPU VM for build workers
gcloud compute instances create gpu-build-worker \
  --zone=us-west2-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --metadata=install-nvidia-driver=True

# Install Cloud Build worker agent
gcloud compute ssh gpu-build-worker --zone=us-west2-a --command="
  curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  sudo bash add-google-cloud-ops-agent-repo.sh --also-install
  sudo systemctl enable google-cloud-ops-agent
"
```

### Cloud Build Configuration for GPU Builds

**cloudbuild-gpu.yaml:**

```yaml
steps:
  # Build CUDA-enabled training container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/ml-trainer:$COMMIT_SHA'
      - '-f'
      - 'docker/Dockerfile.gpu'
      - '--build-arg'
      - 'CUDA_VERSION=12.1'
      - '.'
    timeout: '1800s'

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ml-trainer:$COMMIT_SHA']

  # Run GPU smoke test (requires GPU worker)
  - name: 'gcr.io/$PROJECT_ID/ml-trainer:$COMMIT_SHA'
    args:
      - 'python'
      - '-c'
      - |
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"GPUs detected: {torch.cuda.device_count()}")
    env:
      - 'NVIDIA_VISIBLE_DEVICES=all'

  # Deploy to Vertex AI for testing
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud ai custom-jobs create \
          --region=us-west2 \
          --display-name=smoke-test-$BUILD_ID \
          --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=gcr.io/$PROJECT_ID/ml-trainer:$COMMIT_SHA \
          --args="--mode=test"

options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY
  pool:
    name: 'projects/$PROJECT_ID/locations/us-west2/workerPools/gpu-builder-pool'

timeout: '3600s'
```

**Trigger Configuration:**

```bash
# Create Cloud Build trigger for GPU builds
gcloud builds triggers create github \
  --name=gpu-model-build \
  --repo-name=ml-models \
  --repo-owner=your-org \
  --branch-pattern='^main$' \
  --build-config=cloudbuild-gpu.yaml \
  --region=us-west2
```

From [Secure CI/CD on Cloudbuild using private worker pools](https://blog.searce.com/secure-ci-cd-on-cloudbuild-using-private-worker-pools-a269bbfaf155) (accessed 2025-11-16):
> Private worker pools provide isolated build environments with custom machine configurations, network access controls, and the ability to attach specialized hardware like GPUs.

**Cost Optimization for GPU Builds:**

```yaml
# Use ephemeral GPU workers
steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        # Create temporary GPU instance for build
        gcloud compute instances create temp-gpu-builder-$BUILD_ID \
          --zone=us-west2-a \
          --machine-type=n1-standard-4 \
          --accelerator=type=nvidia-tesla-t4,count=1 \
          --preemptible \
          --scopes=cloud-platform \
          --metadata=startup-script='#!/bin/bash
            # Install Docker and NVIDIA drivers
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
            sudo apt-get update && sudo apt-get install -y nvidia-docker2
            sudo systemctl restart docker
          '

  # Wait for instance to be ready
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['compute', 'instances', 'describe', 'temp-gpu-builder-$BUILD_ID', '--zone=us-west2-a']
    waitFor: ['-']

  # Execute GPU build on temporary instance
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud compute ssh temp-gpu-builder-$BUILD_ID --zone=us-west2-a --command="
          docker build -t gcr.io/$PROJECT_ID/model:$COMMIT_SHA .
          docker run --gpus all gcr.io/$PROJECT_ID/model:$COMMIT_SHA python test_gpu.py
        "

  # Cleanup
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['compute', 'instances', 'delete', 'temp-gpu-builder-$BUILD_ID', '--zone=us-west2-a', '--quiet']
    waitFor: ['-']
```

---

## Section 2: GitHub Actions Self-Hosted GPU Runners

Self-hosted GitHub Actions runners on GCP enable GPU-accelerated CI/CD with full control over infrastructure, cost optimization through Spot instances, and integration with existing GCP services.

### Deploying Self-Hosted GPU Runners

From [GitHub Actions self-hosted GPU runners GCP deployment](https://github.com/terraform-google-modules/terraform-google-github-actions-runners) (accessed 2025-11-16):
> Self-hosted runners provide control over hardware, software, and network configurations. For GPU workloads, this enables custom CUDA installations, specialized ML frameworks, and cost-effective Spot instance usage.

**Terraform Configuration:**

```hcl
# terraform/github-runners-gpu.tf
resource "google_compute_instance_template" "gpu_runner" {
  name_prefix  = "github-runner-gpu-"
  machine_type = "n1-standard-4"
  region       = "us-west2"

  disk {
    source_image = "ubuntu-os-cloud/ubuntu-2004-lts"
    auto_delete  = true
    boot         = true
    disk_size_gb = 100
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible        = true
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e

    # Install NVIDIA drivers
    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535 nvidia-docker2
    sudo systemctl restart docker

    # Install GitHub Actions runner
    cd /opt
    mkdir actions-runner && cd actions-runner
    curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
      https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
    tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

    # Configure runner (use GCP Secret Manager for token)
    GITHUB_TOKEN=$(gcloud secrets versions access latest --secret="github-runner-token")
    ./config.sh --url https://github.com/${var.github_org}/${var.github_repo} \
      --token $GITHUB_TOKEN \
      --labels gpu,cuda-12.1,spot \
      --unattended \
      --ephemeral

    # Start runner
    ./run.sh
  EOF

  service_account {
    email  = google_service_account.runner.email
    scopes = ["cloud-platform"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_instance_group_manager" "gpu_runners" {
  name               = "github-gpu-runners"
  base_instance_name = "gpu-runner"
  zone               = "us-west2-a"
  target_size        = 2

  version {
    instance_template = google_compute_instance_template.gpu_runner.id
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.runner.id
    initial_delay_sec = 300
  }
}

resource "google_compute_health_check" "runner" {
  name                = "runner-health-check"
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port = 8080
  }
}
```

**GitHub Actions Workflow Using GPU Runners:**

```yaml
# .github/workflows/train-model-gpu.yml
name: Train ML Model on GPU

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      dataset_version:
        description: 'Dataset version'
        required: true
        default: 'v1.0'

env:
  GCP_PROJECT: ${{ secrets.GCP_PROJECT_ID }}
  GCP_REGION: us-west2

jobs:
  build-and-test:
    runs-on: [self-hosted, gpu, cuda-12.1]

    steps:
    - uses: actions/checkout@v4

    - name: Verify GPU availability
      run: |
        nvidia-smi
        python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    - name: Run unit tests with GPU
      run: |
        pytest tests/ -v --gpu --cov=src --cov-report=xml

    - name: Build Docker image
      run: |
        docker build \
          -t gcr.io/${{ env.GCP_PROJECT }}/ml-model:${{ github.sha }} \
          -f docker/Dockerfile.gpu \
          --build-arg CUDA_VERSION=12.1 \
          .

    - name: Test Docker image GPU access
      run: |
        docker run --gpus all \
          gcr.io/${{ env.GCP_PROJECT }}/ml-model:${{ github.sha }} \
          python -c "import torch; assert torch.cuda.is_available()"

    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Configure Docker for GCR
      run: |
        gcloud auth configure-docker gcr.io

    - name: Push Docker image
      run: |
        docker push gcr.io/${{ env.GCP_PROJECT }}/ml-model:${{ github.sha }}

    - name: Smoke test training
      run: |
        python train.py \
          --mode=smoke-test \
          --epochs=1 \
          --batch-size=8 \
          --device=cuda

    - name: Upload training artifacts
      uses: actions/upload-artifact@v4
      with:
        name: training-logs
        path: logs/

  deploy-to-vertex:
    needs: build-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Submit training job to Vertex AI
      run: |
        gcloud ai custom-jobs create \
          --region=${{ env.GCP_REGION }} \
          --display-name=training-${{ github.sha }} \
          --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=gcr.io/${{ env.GCP_PROJECT }}/ml-model:${{ github.sha }} \
          --args="--epochs=50,--batch-size=32"
```

From [Self-Hosted GitHub Runners on GKE](https://dev.to/cloudiepad/self-hosted-github-runners-on-gke-my-800month-mistake-that-led-to-a-better-solution-1nk4) (accessed 2025-11-16):
> Actions Runner Controller (ARC) on GKE enables dynamic scaling of runners based on workload, reducing costs by 70% compared to always-on runners. For GPU workloads, use node pools with autoscaling.

**Actions Runner Controller (ARC) on GKE with GPU:**

```yaml
# k8s/runner-deployment-gpu.yaml
apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: gpu-runner-deployment
  namespace: actions-runner-system
spec:
  replicas: 2
  template:
    spec:
      repository: your-org/ml-repo
      labels:
        - gpu
        - cuda-12.1

      dockerdWithinRunnerContainer: true

      resources:
        limits:
          nvidia.com/gpu: 1
          memory: "16Gi"
          cpu: "4"
        requests:
          nvidia.com/gpu: 1
          memory: "8Gi"
          cpu: "2"

      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4

      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      volumeMounts:
      - name: docker-sock
        mountPath: /var/run/docker.sock

      volumes:
      - name: docker-sock
        hostPath:
          path: /var/run/docker.sock
```

---

## Section 3: Automated GPU Model Testing

Automated testing for GPU models requires specialized infrastructure, test frameworks, and validation strategies beyond traditional software testing.

From [CI/CD for Machine Learning in 2024](https://medium.com/infer-qwak/ci-cd-for-machine-learning-in-2024-best-practices-to-build-test-and-deploy-c4ad869824d2) (accessed 2025-11-16):
> Automated testing typically encompasses code lints and unit tests. These tests are crucial for verifying that the model code functions as intended and does not disrupt existing features. It's important to distinguish this step from model performance testing — which is conducted post-training and involves model validation and evaluation.

### GPU Test Framework

**pytest Configuration for GPU Tests:**

```python
# tests/conftest.py
import pytest
import torch

def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run GPU tests"
    )
    parser.addoption(
        "--gpu-id", action="store", default="0", help="GPU device ID"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        # Run GPU tests
        return

    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

@pytest.fixture(scope="session")
def gpu_device(request):
    """Fixture to provide GPU device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device_id = request.config.getoption("--gpu-id")
    return torch.device(f"cuda:{device_id}")

@pytest.fixture(scope="function")
def cleanup_gpu():
    """Cleanup GPU memory after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**GPU-Specific Tests:**

```python
# tests/test_model_gpu.py
import pytest
import torch
import torch.nn as nn
from src.model import ResNetClassifier

@pytest.mark.gpu
class TestModelGPU:

    def test_model_loads_on_gpu(self, gpu_device, cleanup_gpu):
        """Test model successfully loads on GPU"""
        model = ResNetClassifier(num_classes=10)
        model = model.to(gpu_device)

        assert next(model.parameters()).is_cuda
        assert next(model.parameters()).device == gpu_device

    def test_forward_pass_gpu(self, gpu_device, cleanup_gpu):
        """Test forward pass executes on GPU"""
        model = ResNetClassifier(num_classes=10).to(gpu_device)
        batch = torch.randn(8, 3, 224, 224).to(gpu_device)

        output = model(batch)

        assert output.is_cuda
        assert output.shape == (8, 10)

    def test_training_step_gpu(self, gpu_device, cleanup_gpu):
        """Test training step with gradient computation"""
        model = ResNetClassifier(num_classes=10).to(gpu_device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        batch = torch.randn(8, 3, 224, 224).to(gpu_device)
        labels = torch.randint(0, 10, (8,)).to(gpu_device)

        # Forward pass
        output = model(batch)
        loss = criterion(output, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert all(p.grad is not None for p in model.parameters())

    @pytest.mark.slow
    def test_memory_efficiency(self, gpu_device, cleanup_gpu):
        """Test model memory usage stays within limits"""
        model = ResNetClassifier(num_classes=1000).to(gpu_device)

        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(gpu_device)

        # Run inference
        batch = torch.randn(32, 3, 224, 224).to(gpu_device)
        with torch.no_grad():
            _ = model(batch)

        peak_memory = torch.cuda.max_memory_allocated(gpu_device)

        # Should use less than 4GB for this model
        assert (peak_memory - initial_memory) < 4 * 1024 ** 3

    def test_multi_gpu_dataparallel(self, cleanup_gpu):
        """Test DataParallel works with multiple GPUs"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")

        model = ResNetClassifier(num_classes=10)
        model = nn.DataParallel(model)
        model = model.cuda()

        batch = torch.randn(16, 3, 224, 224).cuda()
        output = model(batch)

        assert output.shape == (16, 10)
```

**Integration Tests:**

```python
# tests/test_training_integration.py
import pytest
import torch
from src.trainer import Trainer
from src.dataloader import create_dataloaders

@pytest.mark.gpu
@pytest.mark.slow
class TestTrainingIntegration:

    def test_full_training_epoch(self, gpu_device, tmp_path, cleanup_gpu):
        """Test complete training epoch"""
        # Setup
        train_loader, val_loader = create_dataloaders(
            data_dir="tests/fixtures/data",
            batch_size=8,
            num_workers=2
        )

        trainer = Trainer(
            model_name="resnet18",
            num_classes=10,
            device=gpu_device,
            checkpoint_dir=tmp_path
        )

        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader, epoch=0)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Assertions
        assert "loss" in train_metrics
        assert "accuracy" in train_metrics
        assert train_metrics["loss"] > 0
        assert 0 <= train_metrics["accuracy"] <= 1

        # Check checkpoint was saved
        checkpoints = list(tmp_path.glob("*.pth"))
        assert len(checkpoints) > 0

    def test_checkpoint_recovery(self, gpu_device, tmp_path, cleanup_gpu):
        """Test training can resume from checkpoint"""
        train_loader, _ = create_dataloaders(
            data_dir="tests/fixtures/data",
            batch_size=8
        )

        # Train and save checkpoint
        trainer1 = Trainer(
            model_name="resnet18",
            num_classes=10,
            device=gpu_device,
            checkpoint_dir=tmp_path
        )
        trainer1.train_epoch(train_loader, epoch=0)
        checkpoint_path = trainer1.save_checkpoint(epoch=0)

        # Load from checkpoint
        trainer2 = Trainer(
            model_name="resnet18",
            num_classes=10,
            device=gpu_device,
            checkpoint_dir=tmp_path
        )
        trainer2.load_checkpoint(checkpoint_path)

        # Verify state matches
        for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2)
```

### CI Pipeline with GPU Tests

```yaml
# .github/workflows/test-gpu.yml
name: GPU Test Suite

on:
  pull_request:
  push:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - run: pip install -r requirements-test.txt
    - run: pytest tests/ -v -m "not gpu" --cov=src

  gpu-tests:
    runs-on: [self-hosted, gpu]
    needs: unit-tests

    steps:
    - uses: actions/checkout@v4

    - name: Run GPU tests
      run: |
        pytest tests/ -v -m gpu --gpu --gpu-id=0 \
          --cov=src --cov-report=xml --cov-report=html

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        flags: gpu-tests

    - name: Run integration tests
      run: |
        pytest tests/ -v -m "gpu and slow" --gpu \
          --timeout=3600

  model-validation:
    runs-on: [self-hosted, gpu]
    needs: gpu-tests

    steps:
    - uses: actions/checkout@v4

    - name: Train smoke test model
      run: |
        python train.py \
          --config configs/smoke_test.yaml \
          --epochs 2 \
          --output-dir /tmp/smoke-test

    - name: Validate model outputs
      run: |
        python scripts/validate_model.py \
          --checkpoint /tmp/smoke-test/checkpoint.pth \
          --test-data tests/fixtures/validation
```

---

## Section 4: GPU Container Image Optimization

Optimizing Docker images for GPU workloads reduces build times, deployment speed, and storage costs while maintaining CUDA compatibility.

From [GPU container image building optimization Docker CUDA](https://blog.roboflow.com/use-the-gpu-in-docker/) (accessed 2025-11-16):
> The NVIDIA Container Toolkit is a docker image that provides support to automatically recognize GPU drivers on your base machine and pass those same drivers to your Docker container when it runs.

### Multi-Stage Build for GPU Images

**Optimized Dockerfile:**

```dockerfile
# docker/Dockerfile.gpu
# Stage 1: Build dependencies
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Avoid interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m pip install --no-cache-dir virtualenv
RUN python3.10 -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Stage 2: Runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

# Default command
CMD ["python3", "src/serve.py"]
```

**Build Optimization Script:**

```bash
#!/bin/bash
# scripts/build-gpu-image.sh

set -e

PROJECT_ID="your-project-id"
IMAGE_NAME="ml-model-gpu"
VERSION="${1:-latest}"

# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Build with cache
docker build \
  --file docker/Dockerfile.gpu \
  --tag "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${VERSION}" \
  --tag "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
  --cache-from "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  .

# Push to registry
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${VERSION}"
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

# Clean up dangling images
docker image prune -f
```

From [Top 12 Docker Container Images for AI Projects](https://code-b.dev/blog/docker-container-images) (accessed 2025-11-16):
> Using official NVIDIA base images ensures compatibility with CUDA libraries and GPU drivers. Multi-stage builds can reduce final image size by 60-70% compared to single-stage builds.

### Layer Caching Strategy

**Cloud Build with Layer Caching:**

```yaml
# cloudbuild-cached.yaml
steps:
  # Pull previous image for caching
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        docker pull gcr.io/$PROJECT_ID/ml-model-gpu:latest || exit 0

  # Build with cache
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--file=docker/Dockerfile.gpu'
      - '--cache-from=gcr.io/$PROJECT_ID/ml-model-gpu:latest'
      - '--tag=gcr.io/$PROJECT_ID/ml-model-gpu:$COMMIT_SHA'
      - '--tag=gcr.io/$PROJECT_ID/ml-model-gpu:latest'
      - '--build-arg=BUILDKIT_INLINE_CACHE=1'
      - '.'

  # Push both tags
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', 'gcr.io/$PROJECT_ID/ml-model-gpu']

images:
  - 'gcr.io/$PROJECT_ID/ml-model-gpu:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/ml-model-gpu:latest'

options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: 200
```

---

## Section 5: Deployment Automation for GPU Models

Automated deployment strategies for GPU models include canary releases, blue-green deployments, and progressive rollouts with monitoring.

### Vertex AI Deployment Automation

**Automated Model Deployment Script:**

```python
# scripts/deploy_model.py
from google.cloud import aiplatform
from datetime import datetime
import argparse

def deploy_model_canary(
    model_id: str,
    endpoint_name: str,
    project_id: str,
    region: str = "us-west2",
    canary_traffic: int = 10
):
    """Deploy model with canary strategy"""
    aiplatform.init(project=project_id, location=region)

    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )

    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            description="GPU model serving endpoint"
        )
        print(f"Created new endpoint: {endpoint.resource_name}")

    # Get model
    model = aiplatform.Model(model_id)

    # Deploy with canary traffic split
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name=f"canary-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=canary_traffic,
        sync=True
    )

    print(f"Deployed model with {canary_traffic}% traffic")
    return deployed_model.id

def promote_canary(
    endpoint_name: str,
    canary_model_id: str,
    project_id: str,
    region: str = "us-west2",
    traffic_percentage: int = 100
):
    """Promote canary to full traffic"""
    aiplatform.init(project=project_id, location=region)

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )
    endpoint = endpoints[0]

    # Get traffic split
    traffic_split = {}
    for deployed_model in endpoint.list_models():
        if deployed_model.id == canary_model_id:
            traffic_split[deployed_model.id] = traffic_percentage
        else:
            traffic_split[deployed_model.id] = 0

    # Update traffic
    endpoint.update(traffic_split=traffic_split)
    print(f"Promoted canary to {traffic_percentage}% traffic")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--action", choices=["deploy", "promote"], required=True)
    parser.add_argument("--traffic", type=int, default=10)

    args = parser.parse_args()

    if args.action == "deploy":
        deploy_model_canary(
            model_id=args.model_id,
            endpoint_name=args.endpoint_name,
            project_id=args.project_id,
            canary_traffic=args.traffic
        )
    elif args.action == "promote":
        promote_canary(
            endpoint_name=args.endpoint_name,
            canary_model_id=args.model_id,
            project_id=args.project_id,
            traffic_percentage=args.traffic
        )
```

**GitHub Actions Deployment Workflow:**

```yaml
# .github/workflows/deploy-model.yml
name: Deploy Model to Production

on:
  workflow_dispatch:
    inputs:
      model_uri:
        description: 'Model artifact URI'
        required: true
      deployment_strategy:
        description: 'Deployment strategy'
        required: true
        type: choice
        options:
          - canary
          - blue-green
          - immediate

env:
  GCP_PROJECT: ${{ secrets.GCP_PROJECT_ID }}
  GCP_REGION: us-west2
  ENDPOINT_NAME: ml-model-production

jobs:
  deploy-canary:
    if: github.event.inputs.deployment_strategy == 'canary'
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v2

    - name: Deploy canary (10% traffic)
      id: deploy_canary
      run: |
        MODEL_ID=$(python scripts/deploy_model.py \
          --model-id ${{ github.event.inputs.model_uri }} \
          --endpoint-name ${{ env.ENDPOINT_NAME }} \
          --project-id ${{ env.GCP_PROJECT }} \
          --action deploy \
          --traffic 10)
        echo "model_id=$MODEL_ID" >> $GITHUB_OUTPUT

    - name: Monitor canary for 30 minutes
      run: |
        python scripts/monitor_deployment.py \
          --endpoint-name ${{ env.ENDPOINT_NAME }} \
          --model-id ${{ steps.deploy_canary.outputs.model_id }} \
          --duration 1800 \
          --error-threshold 0.05 \
          --latency-threshold 1000

    - name: Gradual traffic increase
      run: |
        for traffic in 25 50 75 100; do
          echo "Increasing traffic to ${traffic}%"
          python scripts/deploy_model.py \
            --model-id ${{ steps.deploy_canary.outputs.model_id }} \
            --endpoint-name ${{ env.ENDPOINT_NAME }} \
            --project-id ${{ env.GCP_PROJECT }} \
            --action promote \
            --traffic $traffic

          # Monitor for 15 minutes
          python scripts/monitor_deployment.py \
            --endpoint-name ${{ env.ENDPOINT_NAME }} \
            --model-id ${{ steps.deploy_canary.outputs.model_id }} \
            --duration 900 \
            --error-threshold 0.05

          sleep 60
        done

    - name: Cleanup old models
      if: success()
      run: |
        python scripts/cleanup_old_deployments.py \
          --endpoint-name ${{ env.ENDPOINT_NAME }} \
          --keep-latest 3

  deploy-blue-green:
    if: github.event.inputs.deployment_strategy == 'blue-green'
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Create green endpoint
      run: |
        gcloud ai endpoints create \
          --region=${{ env.GCP_REGION }} \
          --display-name=${{ env.ENDPOINT_NAME }}-green

    - name: Deploy to green
      run: |
        python scripts/deploy_model.py \
          --model-id ${{ github.event.inputs.model_uri }} \
          --endpoint-name ${{ env.ENDPOINT_NAME }}-green \
          --project-id ${{ env.GCP_PROJECT }} \
          --action deploy \
          --traffic 100

    - name: Run smoke tests
      run: |
        pytest tests/smoke/ \
          --endpoint=${{ env.ENDPOINT_NAME }}-green \
          --region=${{ env.GCP_REGION }}

    - name: Switch traffic to green
      run: |
        # Update load balancer or DNS
        gcloud compute url-maps import ml-model-urlmap \
          --source=configs/urlmap-green.yaml \
          --global

    - name: Monitor green endpoint
      run: |
        python scripts/monitor_deployment.py \
          --endpoint-name ${{ env.ENDPOINT_NAME }}-green \
          --duration 1800

    - name: Decommission blue
      if: success()
      run: |
        gcloud ai endpoints delete ${{ env.ENDPOINT_NAME }}-blue \
          --region=${{ env.GCP_REGION }} \
          --quiet
```

From [MLOps on Kubernetes: CI/CD for Machine Learning Models](https://collabnix.com/mlops-on-kubernetes-ci-cd-for-machine-learning-models-in-2024/) (accessed 2025-11-16):
> Implementing canary deployments for ML models allows you to test new versions with minimal risk. Traffic splitting with Istio enables gradual rollout while monitoring performance metrics.

---

## Section 6: Cost Optimization for GPU CI/CD

GPU CI/CD costs can be reduced 70-90% through ephemeral runners, Spot instances, and intelligent resource scheduling.

### Ephemeral GPU Runners

**On-Demand Runner Creation:**

```python
# scripts/create_ephemeral_runner.py
from google.cloud import compute_v1
import time

def create_gpu_runner(
    project_id: str,
    zone: str,
    runner_name: str,
    github_token: str
) -> str:
    """Create ephemeral GPU instance for GitHub Actions"""

    compute_client = compute_v1.InstancesClient()

    # Instance configuration
    instance = compute_v1.Instance()
    instance.name = runner_name
    instance.machine_type = f"zones/{zone}/machineTypes/n1-standard-4"

    # Add GPU
    accelerator = compute_v1.AcceleratorConfig()
    accelerator.accelerator_count = 1
    accelerator.accelerator_type = f"zones/{zone}/acceleratorTypes/nvidia-tesla-t4"
    instance.guest_accelerators = [accelerator]

    # Use Spot instance for cost savings
    instance.scheduling = compute_v1.Scheduling()
    instance.scheduling.on_host_maintenance = "TERMINATE"
    instance.scheduling.provisioning_model = "SPOT"

    # Startup script
    startup_script = f"""#!/bin/bash
    set -e

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh

    # Install NVIDIA Docker
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker

    # Install GitHub Actions runner
    mkdir -p /opt/actions-runner && cd /opt/actions-runner
    curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
      https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
    tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

    # Configure and start runner
    ./config.sh --url https://github.com/your-org/your-repo \
      --token {github_token} \
      --labels gpu,ephemeral \
      --ephemeral \
      --unattended

    # Run and shutdown when done
    ./run.sh &
    RUNNER_PID=$!

    # Wait for runner to complete
    wait $RUNNER_PID

    # Self-destruct
    gcloud compute instances delete {runner_name} --zone={zone} --quiet
    """

    instance.metadata = compute_v1.Metadata()
    instance.metadata.items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Create instance
    operation = compute_client.insert(
        project=project_id,
        zone=zone,
        instance_resource=instance
    )

    # Wait for creation
    operation.result()

    return runner_name

# GitHub Actions workflow to trigger ephemeral runner
```

**Cloud Function for Runner Management:**

```python
# functions/runner_manager.py
import functions_framework
from google.cloud import compute_v1
import secrets

@functions_framework.http
def create_runner(request):
    """HTTP Cloud Function to create ephemeral GPU runner"""

    request_json = request.get_json()

    runner_name = f"gpu-runner-{secrets.token_hex(4)}"
    github_token = request_json.get("github_token")

    # Create instance
    create_gpu_runner(
        project_id="your-project-id",
        zone="us-west2-a",
        runner_name=runner_name,
        github_token=github_token
    )

    return {
        "runner_name": runner_name,
        "status": "created"
    }
```

### Spot Instance Handling

**Preemption Recovery Script:**

```python
# scripts/checkpoint_handler.py
import torch
import signal
import sys
import os

class CheckpointHandler:
    """Handle preemption gracefully with automatic checkpointing"""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.should_stop = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_preemption)
        signal.signal(signal.SIGINT, self.handle_preemption)

    def handle_preemption(self, signum, frame):
        """Save checkpoint when preemption signal received"""
        print("Preemption signal received, saving checkpoint...")
        self.should_stop = True

    def save_checkpoint(self, state_dict, optimizer, epoch, step):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch{epoch}_step{step}.pth"
        )

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Upload to GCS for persistence
        upload_to_gcs(checkpoint_path, f"gs://checkpoints/{checkpoint_path}")

def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload checkpoint to Cloud Storage"""
    from google.cloud import storage

    client = storage.Client()
    bucket_name = gcs_path.split('/')[2]
    blob_path = '/'.join(gcs_path.split('/')[3:])

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
```

---

## Section 7: Monitoring and Observability

Monitoring GPU CI/CD pipelines requires tracking build performance, GPU utilization, cost metrics, and deployment health.

### Pipeline Metrics Collection

**Custom Cloud Build Metrics:**

```python
# scripts/collect_build_metrics.py
from google.cloud import monitoring_v3
from google.cloud import build_v1
import time

def collect_build_metrics(project_id: str, build_id: str):
    """Collect and export build metrics to Cloud Monitoring"""

    client = build_v1.CloudBuildClient()
    build = client.get_build(project_id=project_id, id=build_id)

    # Calculate metrics
    start_time = build.create_time.timestamp()
    finish_time = build.finish_time.timestamp()
    duration = finish_time - start_time

    # Export to Cloud Monitoring
    monitoring_client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/cloudbuild/duration"
    series.resource.type = "global"

    point = monitoring_v3.Point()
    point.value.double_value = duration
    point.interval.end_time.seconds = int(time.time())

    series.points = [point]

    monitoring_client.create_time_series(
        name=project_name,
        time_series=[series]
    )
```

**Prometheus Metrics for Runners:**

```python
# scripts/runner_metrics_exporter.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
job_duration = Histogram(
    'github_actions_job_duration_seconds',
    'Job execution duration',
    ['workflow', 'job', 'status']
)

gpu_utilization = Gauge(
    'runner_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

jobs_completed = Counter(
    'github_actions_jobs_completed_total',
    'Total completed jobs',
    ['workflow', 'status']
)

def monitor_runner():
    """Monitor runner and export metrics"""
    import pynvml

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    while True:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization.labels(gpu_id=str(i)).set(util.gpu)

        time.sleep(10)

if __name__ == "__main__":
    start_http_server(8000)
    monitor_runner()
```

From [gcloud-cicd/00-pipeline-integration.md](../gcloud-cicd/00-pipeline-integration.md):
> Monitoring helps identify when a model begins to drift or underperform due to changing data patterns or other factors. This phase is not the end but a trigger for a new cycle of improvement — initiating retraining, adjustments, or complete redevelopment as needed.

---

## Section 8: ARR-COC-0-1 GPU CI/CD Implementation

Complete CI/CD pipeline for arr-coc-0-1 project with GPU training, testing, and deployment automation.

### Project-Specific Pipeline

**GitHub Actions Workflow:**

```yaml
# .github/workflows/arr-coc-cicd.yml
name: ARR-COC CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
  workflow_dispatch:

env:
  GCP_PROJECT: ${{ secrets.GCP_PROJECT_ID }}
  GCP_REGION: us-west2
  IMAGE_NAME: arr-coc-training

jobs:
  build-and-test:
    runs-on: [self-hosted, gpu, cuda-12.1]

    steps:
    - uses: actions/checkout@v4

    - name: Verify GPU
      run: |
        nvidia-smi
        python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --gpu

    - name: Test texture extraction
      run: |
        python -m arr_coc.texture.extract_textures \
          --image tests/fixtures/sample.jpg \
          --output /tmp/textures

    - name: Build Docker image
      run: |
        docker build \
          -t gcr.io/${{ env.GCP_PROJECT }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -f docker/Dockerfile.gpu \
          .

    - name: Test Docker GPU access
      run: |
        docker run --gpus all \
          gcr.io/${{ env.GCP_PROJECT }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          python -c "import torch; assert torch.cuda.is_available()"

    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Push image
      run: |
        gcloud auth configure-docker gcr.io
        docker push gcr.io/${{ env.GCP_PROJECT }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  train-model:
    needs: build-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Authenticate to GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Submit training job
      run: |
        gcloud ai custom-jobs create \
          --region=${{ env.GCP_REGION }} \
          --display-name=arr-coc-training-${{ github.sha }} \
          --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/${{ env.GCP_PROJECT }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          --args="--config=configs/production.yaml,--output-dir=gs://arr-coc-models/runs/${{ github.sha }}"

  deploy:
    needs: train-model
    runs-on: ubuntu-latest

    steps:
    - name: Deploy to Vertex AI Endpoint
      run: |
        python scripts/deploy_model.py \
          --model-uri gs://arr-coc-models/runs/${{ github.sha }}/model \
          --endpoint-name arr-coc-production \
          --strategy canary
```

**Cost Tracking:**

```python
# scripts/track_pipeline_costs.py
from google.cloud import billing_v1
from google.cloud import build_v1
import pandas as pd

def analyze_pipeline_costs(project_id: str, days: int = 30):
    """Analyze CI/CD pipeline costs over time"""

    # Query Cloud Build costs
    build_client = build_v1.CloudBuildClient()
    builds = build_client.list_builds(
        project_id=project_id,
        filter=f"create_time>-{days}d"
    )

    cost_data = []
    for build in builds:
        # Estimate cost based on build duration and machine type
        duration_minutes = (
            build.finish_time.timestamp() - build.create_time.timestamp()
        ) / 60

        # GPU cost estimation: T4 = $0.35/hour + n1-standard-8 = $0.38/hour
        gpu_cost = (duration_minutes / 60) * 0.73

        cost_data.append({
            'build_id': build.id,
            'duration_minutes': duration_minutes,
            'estimated_cost': gpu_cost,
            'status': build.status.name
        })

    df = pd.DataFrame(cost_data)

    print(f"\nTotal CI/CD GPU costs (last {days} days): ${df['estimated_cost'].sum():.2f}")
    print(f"Average cost per build: ${df['estimated_cost'].mean():.2f}")
    print(f"Success rate: {(df['status'] == 'SUCCESS').mean():.1%}")

    return df
```

---

## Sources

**Source Documents:**
- [gcloud-cicd/00-pipeline-integration.md](../gcloud-cicd/00-pipeline-integration.md) - Cloud Build CI/CD integration patterns and Vertex AI pipeline orchestration

**Web Research (accessed 2025-11-16):**
- [CI/CD for Machine Learning in 2024: Best Practices](https://medium.com/infer-qwak/ci-cd-for-machine-learning-in-2024-best-practices-to-build-test-and-deploy-c4ad869824d2) - Comprehensive ML CI/CD guide covering training automation, deployment strategies, and monitoring
- [MLOps on Kubernetes: CI/CD for Machine Learning Models](https://collabnix.com/mlops-on-kubernetes-ci-cd-for-machine-learning-models-in-2024/) - Kubernetes-based ML pipelines with Tekton, GPU scheduling, and production deployment patterns
- [How to Use Your GPU in a Docker Container](https://blog.roboflow.com/use-the-gpu-in-docker/) - NVIDIA Container Toolkit setup, Docker GPU access, and container optimization for ML workloads
- [Cloud Build GPU runners custom worker pools](https://docs.cloud.google.com/run/docs/configuring/services/build-worker-pools) - GCP Cloud Build worker pool configuration and GPU support
- [GitHub Actions self-hosted GPU runners](https://github.com/terraform-google-modules/terraform-google-github-actions-runners) - Terraform modules for deploying GPU-enabled GitHub Actions runners on GCP
- [Self-Hosted GitHub Runners on GKE](https://dev.to/cloudiepad/self-hosted-github-runners-on-gke-my-800month-mistake-that-led-to-a-better-solution-1nk4) - Cost optimization strategies for GPU runners with Actions Runner Controller
- [Secure CI/CD with Private Worker Pools](https://blog.searce.com/secure-ci-cd-on-cloudbuild-using-private-worker-pools-a269bbfaf155) - Private worker pool security and network isolation
- [Top 12 Docker Container Images for AI](https://code-b.dev/blog/docker-container-images) - NVIDIA CUDA base images and multi-stage build optimization

**Additional References:**
- [Google Cloud Build Documentation](https://cloud.google.com/build/docs) - Official Cloud Build GPU and worker pool configuration
- [GitHub Actions Documentation](https://docs.github.com/en/actions) - Self-hosted runner setup and workflows
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) - GPU container runtime and Docker integration
