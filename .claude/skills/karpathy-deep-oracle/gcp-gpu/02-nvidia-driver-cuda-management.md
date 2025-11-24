# NVIDIA Driver & CUDA Toolkit Management on GCP

## Overview

NVIDIA driver and CUDA toolkit management is critical for GPU workloads on Google Cloud Platform. Proper driver installation, version compatibility, and automated deployment strategies ensure optimal performance, stability, and maintainability of GPU-accelerated applications including deep learning training, inference serving, and HPC workloads.

**Why Driver Management Matters:**
- **Performance**: Correct driver versions unlock full GPU capabilities (Tensor Cores, NVLink, MIG)
- **Stability**: Version mismatches cause cryptic runtime errors and crashes
- **Security**: Outdated drivers contain vulnerabilities exploitable in multi-tenant environments
- **Compatibility**: CUDA toolkit, cuDNN, and framework versions must align precisely
- **Automation**: Manual installation doesn't scale beyond single-instance experiments

**Key Insight from [GCP GPU Driver Documentation](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu) (accessed 2025-11-16):**
> "GPU drivers can be installed automatically using installation scripts, or manually by downloading the driver from NVIDIA. Automated installation is recommended for production deployments."

**Related Knowledge:**
- See [cuda/02-pytorch-build-system-compilation.md](../cuda/02-pytorch-build-system-compilation.md) for CUDA compilation targeting
- See [cuda/03-compute-capabilities-gpu-architectures.md](../cuda/03-compute-capabilities-gpu-architectures.md) for architecture-specific requirements
- See [practical-implementation/32-vertex-ai-gpu-tpu.md](../practical-implementation/32-vertex-ai-gpu-tpu.md) for Vertex AI GPU configurations

---

## Section 1: NVIDIA Driver Versions & Selection (~100 lines)

### Driver Version Compatibility Matrix

**NVIDIA Driver Release Branches (2024-2025):**

From [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) (accessed 2025-11-16):

| Driver Branch | CUDA Support | GPU Architectures | Release Date | GCP Availability |
|--------------|--------------|-------------------|--------------|------------------|
| **535.x** | CUDA 12.2 | Pascal-Hopper (sm_60-sm_90) | Aug 2023 | Production |
| **545.x** | CUDA 12.3 | Pascal-Hopper (sm_60-sm_90) | Nov 2023 | Production |
| **550.x** | CUDA 12.4 | Pascal-Hopper (sm_60-sm_90) | Mar 2024 | Production |
| **560.x** | CUDA 12.6 | Pascal-Blackwell (sm_60-sm_100) | Sep 2024 | Production |
| **565.x** | CUDA 12.8 | Pascal-Blackwell (sm_60-sm_120) | Jan 2025 | Beta |

**CUDA Toolkit Version Requirements:**

From [cuda/03-compute-capabilities-gpu-architectures.md](../cuda/03-compute-capabilities-gpu-architectures.md):

| GPU Model | Compute Capability | Minimum CUDA | Recommended Driver | GCP Machine Type |
|-----------|-------------------|--------------|-------------------|------------------|
| T4 | sm_75 | CUDA 10.0 | 535.x+ | n1-standard-4 (1x T4) |
| A100 40GB | sm_80 | CUDA 11.0 | 550.x+ | a2-highgpu-1g |
| A100 80GB | sm_80 | CUDA 11.0 | 550.x+ | a2-ultragpu-1g |
| L4 | sm_89 | CUDA 11.8 | 550.x+ | g2-standard-4 |
| H100 80GB | sm_90 | CUDA 12.0 | 560.x+ | a3-highgpu-8g |

### Choosing the Right Driver Version

**Production Recommendations:**

```bash
# A100 (sm_80) - Stable, well-tested
Driver: 550.90.07
CUDA: 12.4
cuDNN: 8.9.7

# H100 (sm_90) - Latest features
Driver: 560.35.03
CUDA: 12.6
cuDNN: 9.0.0

# L4 (sm_89) - Cost-effective inference
Driver: 550.90.07
CUDA: 12.4
cuDNN: 8.9.7
```

**Version Selection Strategy:**

From [cuda/02-pytorch-build-system-compilation.md](../cuda/02-pytorch-build-system-compilation.md):
- **Match PyTorch requirements**: PyTorch 2.1+ requires CUDA 12.1+
- **Check framework compatibility**: TensorFlow 2.15 requires CUDA 12.3
- **Consider stability**: LTS branches (535.x) for production, latest for development

**Driver Feature Matrix:**

| Feature | 535.x | 550.x | 560.x | Impact |
|---------|-------|-------|-------|--------|
| TF32 (A100) | ✓ | ✓ | ✓ | 8× training speedup |
| MIG (A100) | ✓ | ✓ | ✓ | GPU partitioning |
| FP8 (H100) | ✗ | ✓ | ✓ | 2× inference throughput |
| Transformer Engine | ✗ | ✓ | ✓ | Auto FP8 for LLMs |
| NVLink Switch | ✗ | ✗ | ✓ | 256-GPU clusters |

### Forward Compatibility (CUDA Minor Version Compatibility)

**CUDA MVC enables newer toolkit with older drivers:**

```bash
# Driver 535.x supports CUDA 12.2
# But can run applications built with CUDA 12.4 via MVC

# Check driver CUDA version
nvidia-smi
# Driver Version: 535.129.03    CUDA Version: 12.2

# Run application built with CUDA 12.4
./my_app_cuda_124
# Works via forward compatibility!
```

From [NVIDIA CUDA Compatibility Documentation](https://docs.nvidia.com/deploy/cuda-compatibility/) (accessed 2025-11-16):
> "CUDA applications compiled with a toolkit version newer than the driver's supported CUDA version can still run, as long as the driver supports the GPU architecture."

**Limitations:**
- New runtime APIs unavailable (e.g., CUDA 12.4 APIs won't work with 12.2 driver)
- Performance may not match native driver version
- Debugging tools require matching versions

---

## Section 2: Automated Driver Installation on GCP (~150 lines)

### Installation Script (Recommended Method)

**Google-Provided Installation Scripts:**

From [GCP GPU Driver Installation Guide](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu) (accessed 2025-11-16):

**For Debian/Ubuntu:**
```bash
# Download and run Google's automated installer
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py

# Install with default settings (latest stable driver)
sudo python3 install_gpu_driver.py

# Install specific version
sudo python3 install_gpu_driver.py --driver-version 550.90.07

# Install with CUDA toolkit
sudo python3 install_gpu_driver.py --cuda-version 12.4
```

**For Container-Optimized OS (COS):**
```bash
# COS uses Google's automatic driver loader
# Driver installed on boot via DaemonSet

# Verify installation
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Script Features:**
- Automatic kernel header detection
- DKMS (Dynamic Kernel Module Support) configuration
- Secure Boot compatibility (uses signed drivers)
- Rollback on installation failure
- Logging to `/var/log/nvidia-installer.log`

### Startup Script Automation

**Metadata-Based Installation:**

```bash
# Create VM with automated driver installation
gcloud compute instances create gpu-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --metadata startup-script='#!/bin/bash
# Install driver on first boot
if ! command -v nvidia-smi &> /dev/null; then
    curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
    python3 install_gpu_driver.py --driver-version 550.90.07
    echo "Driver installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
fi
'
```

**Persistent Startup Script (Cloud Storage):**

```bash
# Store installation script in GCS
gsutil cp install_gpu_driver.py gs://my-bucket/gpu-setup/

# Reference in VM metadata
gcloud compute instances create gpu-vm \
    --metadata startup-script-url=gs://my-bucket/gpu-setup/install_gpu_driver.py
```

### Instance Templates for GPU Fleets

**Repeatable GPU Instance Template:**

```bash
# Create instance template with GPU and driver automation
gcloud compute instance-templates create gpu-training-template \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --metadata-from-file startup-script=gpu-driver-setup.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# Create managed instance group (auto-scaling GPU fleet)
gcloud compute instance-groups managed create gpu-training-group \
    --base-instance-name=gpu-trainer \
    --template=gpu-training-template \
    --size=0 \
    --zone=us-central1-a

# Scale up for training job
gcloud compute instance-groups managed resize gpu-training-group --size=8
```

### Container-Optimized OS (COS) with GPU

**COS Automatic Driver Installation:**

From [GKE GPU Node Pools Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) (accessed 2025-11-16):

```bash
# COS automatically loads NVIDIA drivers via cos-gpu-installer DaemonSet
# No manual installation required

# Verify driver loaded
gcloud compute ssh gpu-vm --zone=us-central1-a --command="docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
```

**COS GPU Installer Configuration:**

```yaml
# Custom COS GPU installer configuration
# /etc/default/nvidia-installer-env

NVIDIA_DRIVER_VERSION=550.90.07
NVIDIA_INSTALL_DIR=/var/lib/nvidia
NVIDIA_DRIVER_DOWNLOAD_URL=https://us.download.nvidia.com/tesla/
```

### Ansible Automation

**Ansible Playbook for GPU Driver Installation:**

```yaml
# gpu_driver_install.yml
---
- name: Install NVIDIA GPU drivers on GCP instances
  hosts: gpu_instances
  become: yes
  tasks:
    - name: Install prerequisites
      apt:
        name:
          - build-essential
          - linux-headers-{{ ansible_kernel }}
          - dkms
        state: present
        update_cache: yes

    - name: Download Google GPU installer
      get_url:
        url: https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
        dest: /tmp/install_gpu_driver.py
        mode: '0755'

    - name: Install NVIDIA driver
      command: python3 /tmp/install_gpu_driver.py --driver-version 550.90.07
      args:
        creates: /usr/bin/nvidia-smi

    - name: Verify installation
      command: nvidia-smi
      register: nvidia_smi_output

    - name: Display GPU info
      debug:
        var: nvidia_smi_output.stdout_lines
```

**Running Ansible Playbook:**

```bash
# Install on all GPU instances
ansible-playbook -i gcp_gpu_inventory.ini gpu_driver_install.yml

# Inventory file (gcp_gpu_inventory.ini)
[gpu_instances]
gpu-vm-1 ansible_host=34.123.45.67
gpu-vm-2 ansible_host=34.123.45.68
gpu-vm-3 ansible_host=34.123.45.69
```

---

## Section 3: CUDA Toolkit Installation & Configuration (~150 lines)

### CUDA Toolkit Version Selection

**CUDA Toolkit Compatibility Matrix:**

From [cuda/02-pytorch-build-system-compilation.md](../cuda/02-pytorch-build-system-compilation.md):

| CUDA Version | Driver Min | cuDNN Compatible | PyTorch Support | TensorFlow Support |
|--------------|-----------|------------------|-----------------|-------------------|
| 12.1 | 530.x | 8.9.x | 2.1, 2.2 | 2.15 |
| 12.2 | 535.x | 8.9.x, 9.0.x | 2.2, 2.3 | 2.16 |
| 12.4 | 550.x | 9.0.x, 9.1.x | 2.3, 2.4 | 2.17 |
| 12.6 | 560.x | 9.2.x, 9.3.x | 2.5+ | 2.18+ |

### CUDA Toolkit Installation Methods

**Method 1: runfile Installer (Full Control)**

```bash
# Download CUDA 12.4 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install with custom options
sudo sh cuda_12.4.0_550.54.14_linux.run \
    --silent \
    --toolkit \
    --override \
    --installpath=/usr/local/cuda-12.4

# Skip driver installation (already installed)
sudo sh cuda_12.4.0_550.54.14_linux.run \
    --silent \
    --toolkit \
    --no-drm \
    --no-man-page \
    --override

# Set environment variables
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.4, V12.4.99
```

**Method 2: Package Manager (Easier Updates)**

```bash
# Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install specific CUDA version
sudo apt-get install cuda-toolkit-12-4

# Install multiple CUDA versions side-by-side
sudo apt-get install cuda-toolkit-12-1 cuda-toolkit-12-4

# Switch between versions using alternatives
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.4 100
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.1 50
sudo update-alternatives --config cuda
```

**Method 3: Conda (Isolated Environments)**

```bash
# Create environment with specific CUDA version
conda create -n pytorch_cuda124 python=3.11
conda activate pytorch_cuda124

# Install CUDA toolkit via conda
conda install -c nvidia cuda-toolkit=12.4

# Verify
nvcc --version
which nvcc
# /home/user/miniconda3/envs/pytorch_cuda124/bin/nvcc
```

### Multi-CUDA Version Management

**Supporting Multiple CUDA Versions:**

```bash
# Install CUDA 12.1, 12.2, 12.4 side-by-side
sudo sh cuda_12.1.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-12.1
sudo sh cuda_12.2.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-12.2
sudo sh cuda_12.4.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-12.4

# Create symbolic link management script
cat > /usr/local/bin/cuda-select <<'EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "Current CUDA: $(readlink /usr/local/cuda)"
    echo "Available versions:"
    ls -1d /usr/local/cuda-* | sed 's/.*cuda-/  /'
    exit 0
fi

VERSION=$1
if [ ! -d "/usr/local/cuda-$VERSION" ]; then
    echo "Error: CUDA $VERSION not installed"
    exit 1
fi

sudo rm -f /usr/local/cuda
sudo ln -s /usr/local/cuda-$VERSION /usr/local/cuda
echo "Switched to CUDA $VERSION"
nvcc --version
EOF

chmod +x /usr/local/bin/cuda-select

# Usage
cuda-select 12.4  # Switch to CUDA 12.4
cuda-select       # Show current version
```

### CUDA Environment Configuration

**Comprehensive CUDA Environment Setup:**

```bash
# ~/.cuda_env
export CUDA_VERSION=12.4
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# CUDA runtime configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Limit visible GPUs
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Consistent GPU ordering

# CUDA compilation flags
export CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"  # Target architectures
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0+PTX"  # PyTorch

# cuDNN configuration
export CUDNN_VERSION=9.0.0
export CUDNN_ROOT=/usr/local/cudnn-${CUDNN_VERSION}
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib:$LD_LIBRARY_PATH

# TensorRT (if installed)
export TRT_RELEASE=/usr/local/tensorrt-8.6
export LD_LIBRARY_PATH=$TRT_RELEASE/lib:$LD_LIBRARY_PATH

# Source in .bashrc
source ~/.cuda_env
```

### CUDA Samples Verification

**Testing CUDA Installation:**

```bash
# Clone CUDA samples
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery

# Compile and run
make
./deviceQuery

# Expected output
# CUDA Device Query (Runtime API) version (CUDART static linking)
# Detected 1 CUDA Capable device(s)
# Device 0: "NVIDIA A100-SXM4-40GB"
#   CUDA Driver Version / Runtime Version          12.4 / 12.4
#   CUDA Capability Major/Minor version number:    8.0
```

**Performance Benchmarking:**

```bash
# Compile and run bandwidth test
cd cuda-samples/Samples/1_Utilities/bandwidthTest
make
./bandwidthTest

# Output
# Device 0: NVIDIA A100-SXM4-40GB
# Host to Device Bandwidth, 1 Device(s)
#  Transfer Size (Bytes)        Bandwidth(GB/s)
#  32000000                     25.3
#
# Device to Host Bandwidth, 1 Device(s)
#  Transfer Size (Bytes)        Bandwidth(GB/s)
#  32000000                     26.1
#
# Device to Device Bandwidth, 1 Device(s)
#  Transfer Size (Bytes)        Bandwidth(GB/s)
#  32000000                     1555.2  # HBM2e bandwidth
```

---

## Section 4: cuDNN Installation & Management (~150 lines)

### cuDNN Version Selection

**cuDNN Release Matrix:**

From [NVIDIA cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) (accessed 2025-11-16):

| cuDNN Version | CUDA Compatibility | Release Date | Key Features |
|--------------|-------------------|--------------|--------------|
| 8.9.7 | 11.x, 12.0-12.2 | Jan 2024 | Stable, widely supported |
| 9.0.0 | 12.0-12.4 | Mar 2024 | FP8 support, graph API |
| 9.1.0 | 12.2-12.6 | Jun 2024 | Improved FP8 performance |
| 9.2.0 | 12.4-12.6 | Sep 2024 | H100 optimizations |
| 9.3.0 | 12.6+ | Jan 2025 | Blackwell support |

### cuDNN Installation Methods

**Method 1: Debian Package (Recommended for GCP)**

```bash
# Download cuDNN from NVIDIA Developer site (requires login)
# Alternative: Direct download (if link available)
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb

# Install repository
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# Install cuDNN
sudo apt-get install libcudnn9-cuda-12
sudo apt-get install libcudnn9-dev-cuda-12  # Development files
sudo apt-get install libcudnn9-samples-cuda-12  # Samples
```

**Method 2: Tarball (Multiple Versions)**

```bash
# Download cuDNN tarball
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz

# Extract to custom location
tar -xf cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
sudo mkdir -p /usr/local/cudnn-9.0.0
sudo cp -r cudnn-linux-x86_64-9.0.0.312_cuda12-archive/* /usr/local/cudnn-9.0.0/

# Set environment variables
export CUDNN_ROOT=/usr/local/cudnn-9.0.0
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDNN_ROOT/include:$CPLUS_INCLUDE_PATH
```

**Method 3: Conda (Isolated)**

```bash
# Install cuDNN via conda
conda install -c conda-forge cudnn=9.0.0

# Verify installation
python -c "import torch; print(torch.backends.cudnn.version())"
# 9000
```

### cuDNN Version Verification

**Check Installed cuDNN Version:**

```bash
# Method 1: Check header file
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
# #define CUDNN_MAJOR 9
# #define CUDNN_MINOR 0
# #define CUDNN_PATCHLEVEL 0

# Method 2: Use Python
python3 -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"

# Method 3: Check library
dpkg -l | grep cudnn
# ii  libcudnn9-cuda-12  9.0.0.312-1  amd64  cuDNN runtime libraries
```

### cuDNN Sample Testing

**Compile and Run cuDNN Samples:**

```bash
# Copy samples to home directory
cp -r /usr/src/cudnn_samples_v9 ~/
cd ~/cudnn_samples_v9/mnistCUDNN

# Compile sample
make clean && make

# Run MNIST test
./mnistCUDNN
# Executing: mnistCUDNN
# cudnnGetVersion() : 9000 , CUDNN_VERSION from cudnn.h : 9000
#
# Test passed!
```

### Multi-cuDNN Version Management

**Supporting Multiple cuDNN Versions:**

```bash
# Install cuDNN 8.9.7, 9.0.0, 9.1.0 side-by-side
sudo mkdir -p /usr/local/cudnn-{8.9.7,9.0.0,9.1.0}

# Extract each version
tar -xf cudnn-8.9.7-cuda12.tar.xz
sudo cp -r cudnn-8.9.7/* /usr/local/cudnn-8.9.7/

# Create cudnn-select script
cat > /usr/local/bin/cudnn-select <<'EOF'
#!/bin/bash
VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: cudnn-select <version>"
    echo "Available: 8.9.7, 9.0.0, 9.1.0"
    exit 1
fi

export CUDNN_ROOT=/usr/local/cudnn-$VERSION
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib:$LD_LIBRARY_PATH
echo "cuDNN $VERSION activated"
python3 -c "import torch; print(f'Torch cuDNN: {torch.backends.cudnn.version()}')"
EOF

chmod +x /usr/local/bin/cudnn-select
```

### Framework-Specific cuDNN Configuration

**PyTorch cuDNN Settings:**

```python
import torch

# Enable cuDNN auto-tuner (finds fastest algorithms)
torch.backends.cudnn.benchmark = True

# Enable TF32 for A100 (8× speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Deterministic mode (reproducible results, slower)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check cuDNN enabled
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
```

**TensorFlow cuDNN Configuration:**

```python
import tensorflow as tf

# Enable TF32 on A100
from tensorflow.python.framework import config
config.enable_tensor_float_32_execution(True)

# Verify cuDNN
print(tf.test.is_built_with_cuda())  # True
print(tf.sysconfig.get_build_info()['cudnn_version'])  # '90'

# Memory growth (avoid OOM)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Section 5: Driver Update Strategies (~100 lines)

### Zero-Downtime Driver Updates

**Challenge: Driver updates traditionally require reboot**

From [MassedCompute FAQ](https://massedcompute.com/faq-answers) (accessed 2025-11-16):
> "Updating NVIDIA drivers on a live cloud instance is possible in many cases, but the process depends on several factors, including the type of workload and GPU isolation capabilities."

**Strategies for Minimal Downtime:**

**1. Rolling Updates with Load Balancing:**

```bash
# Update GPU fleet one instance at a time
for instance in gpu-vm-{1..8}; do
    # Remove from load balancer
    gcloud compute backend-services remove-backend gpu-backend \
        --instance-group=gpu-group \
        --instance-group-zone=us-central1-a \
        --backend-name=$instance

    # Update driver
    gcloud compute ssh $instance --command="
        sudo python3 install_gpu_driver.py --driver-version 560.35.03
        sudo reboot
    "

    # Wait for instance to come back
    while ! gcloud compute ssh $instance --command="nvidia-smi" 2>/dev/null; do
        sleep 10
    done

    # Re-add to load balancer
    gcloud compute backend-services add-backend gpu-backend \
        --instance-group=gpu-group \
        --instance-group-zone=us-central1-a

    echo "Updated $instance"
    sleep 60  # Drain time
done
```

**2. Blue-Green Deployment:**

```bash
# Create new instance group with updated driver
gcloud compute instance-templates create gpu-template-green \
    --machine-type=a2-highgpu-1g \
    --image-family=ubuntu-2204-lts \
    --metadata startup-script='#!/bin/bash
        python3 install_gpu_driver.py --driver-version 560.35.03
    '

# Create green instance group
gcloud compute instance-groups managed create gpu-group-green \
    --template=gpu-template-green \
    --size=8 \
    --zone=us-central1-a

# Wait for instances to be healthy
gcloud compute instance-groups managed wait-until-stable gpu-group-green

# Switch traffic to green group
gcloud compute backend-services update gpu-backend \
    --instance-group=gpu-group-green

# Delete old blue group
gcloud compute instance-groups managed delete gpu-group-blue
```

**3. Canary Deployment:**

```bash
# Update 10% of fleet first
TOTAL_INSTANCES=20
CANARY_SIZE=2

# Update canary instances
for i in $(seq 1 $CANARY_SIZE); do
    gcloud compute ssh gpu-vm-$i --command="
        sudo python3 install_gpu_driver.py --driver-version 560.35.03
        sudo reboot
    "
done

# Monitor canary performance (24 hours)
# Check nvidia-smi, application logs, error rates

# If successful, update remaining instances
if [ $CANARY_SUCCESS -eq 1 ]; then
    for i in $(seq $((CANARY_SIZE+1)) $TOTAL_INSTANCES); do
        # Update remaining instances
    done
fi
```

### Driver Rollback Procedures

**DKMS-Based Rollback:**

```bash
# List installed driver versions
dkms status
# nvidia, 550.90.07, 5.15.0-1053-gcp, x86_64: installed
# nvidia, 560.35.03, 5.15.0-1053-gcp, x86_64: installed

# Remove problematic version
sudo dkms remove nvidia/560.35.03 --all
sudo apt-get remove --purge nvidia-driver-560

# Reinstall previous version
sudo apt-get install nvidia-driver-550

# Reboot required
sudo reboot
```

**Snapshot-Based Rollback:**

```bash
# Before update: Create boot disk snapshot
gcloud compute disks snapshot gpu-vm-boot \
    --snapshot-names=pre-driver-update-$(date +%Y%m%d) \
    --zone=us-central1-a

# If update fails: Restore from snapshot
gcloud compute instances delete gpu-vm --zone=us-central1-a
gcloud compute instances create gpu-vm \
    --source-snapshot=pre-driver-update-20250116 \
    --zone=us-central1-a
```

### Monitoring Driver Health

**Automated Driver Health Checks:**

```bash
#!/bin/bash
# /usr/local/bin/gpu-health-check.sh

# Check driver loaded
if ! lsmod | grep -q nvidia; then
    echo "ERROR: NVIDIA driver not loaded"
    exit 1
fi

# Check nvidia-smi responds
if ! timeout 5 nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi timeout"
    exit 1
fi

# Check GPU count
EXPECTED_GPUS=8
ACTUAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $ACTUAL_GPUS -ne $EXPECTED_GPUS ]; then
    echo "ERROR: Expected $EXPECTED_GPUS GPUs, found $ACTUAL_GPUS"
    exit 1
fi

# Check GPU temperature
MAX_TEMP=85
CURRENT_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -nr | head -1)
if [ $CURRENT_TEMP -gt $MAX_TEMP ]; then
    echo "WARNING: GPU temperature $CURRENT_TEMP°C exceeds $MAX_TEMP°C"
fi

echo "GPU health check passed"
```

**Cloud Monitoring Integration:**

```bash
# Send GPU metrics to Cloud Monitoring
cat > /etc/google-cloud-ops-agent/config.yaml <<EOF
metrics:
  receivers:
    nvidia_smi:
      type: nvidia_smi
      collection_interval: 60s
  service:
    pipelines:
      nvidia:
        receivers: [nvidia_smi]
EOF

sudo service google-cloud-ops-agent restart
```

---

## Section 6: Troubleshooting & Common Issues (~100 lines)

### Driver Installation Failures

**Issue 1: Kernel Header Mismatch**

```bash
# Error: Unable to find the kernel source tree
# Cause: Running kernel != installed headers

# Check running kernel
uname -r
# 5.15.0-1053-gcp

# Check installed headers
dpkg -l | grep linux-headers
# linux-headers-5.15.0-1050-gcp  (MISMATCH!)

# Solution: Install matching headers
sudo apt-get install linux-headers-$(uname -r)
sudo python3 install_gpu_driver.py
```

**Issue 2: Secure Boot Enabled**

```bash
# Error: modprobe: ERROR: could not insert 'nvidia': Required key not available

# Check Secure Boot status
mokutil --sb-state
# SecureBoot enabled

# Solution: Use signed drivers (Google's installer handles this)
sudo python3 install_gpu_driver.py --secure-boot

# Or disable Secure Boot (not recommended for production)
# Requires UEFI settings change + VM recreation
```

**Issue 3: Conflicting Nouveau Driver**

```bash
# Check if nouveau loaded
lsmod | grep nouveau

# Blacklist nouveau
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

# Update initramfs
sudo update-initramfs -u

# Reboot
sudo reboot
```

### Runtime Errors

**Issue 4: CUDA Version Mismatch**

```python
# Error: RuntimeError: CUDA error: no kernel image is available for execution on the device

# Check compiled CUDA version
import torch
print(torch.version.cuda)  # 12.1

# Check driver CUDA version
# nvidia-smi shows 12.4

# Solution: Rebuild PyTorch with matching CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Issue 5: cuDNN Version Incompatibility**

```python
# Error: Could not load dynamic library 'libcudnn.so.9'

# Check expected version
import torch
torch.backends.cudnn.version()  # Returns None (not found)

# Check installed version
ls -la /usr/lib/x86_64-linux-gnu/libcudnn*
# libcudnn.so.8 (WRONG VERSION!)

# Solution: Install correct cuDNN
sudo apt-get install libcudnn9-cuda-12
```

**Issue 6: Out of Memory (OOM)**

```bash
# Check GPU memory usage
nvidia-smi

# Common causes:
# 1. Previous process still holding memory
nvidia-smi | grep python
sudo kill -9 <PID>

# 2. Fragmented memory
# Solution: Restart training process

# 3. Batch size too large
# Solution: Reduce batch size or enable gradient checkpointing
```

### Performance Issues

**Issue 7: Low GPU Utilization**

```bash
# Check GPU utilization
nvidia-smi dmon -s u -c 10
# gpu   sm   mem   enc   dec
#   0    0     0     0     0   # LOW UTILIZATION!

# Common causes:
# 1. CPU bottleneck (data loading)
# Solution: Increase num_workers in DataLoader

# 2. Small batch size
# Solution: Increase batch size

# 3. Synchronization overhead
# Solution: Use asynchronous data transfer
```

**Issue 8: TF32 Not Enabled**

```python
# Check TF32 status
import torch
print(torch.backends.cuda.matmul.allow_tf32)  # False (8× slower!)

# Enable TF32 for A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Verify performance improvement
# Before: 150ms per iteration
# After: 18ms per iteration (8× speedup)
```

---

## Section 7: Production Best Practices (~100 lines)

### Driver Version Pinning

**Lock Driver Versions in Production:**

```bash
# Pin specific driver version (prevent auto-updates)
sudo apt-mark hold nvidia-driver-550
sudo apt-mark hold nvidia-dkms-550

# Verify pinned packages
apt-mark showhold
# nvidia-driver-550
# nvidia-dkms-550

# When ready to update: unhold
sudo apt-mark unhold nvidia-driver-550
sudo apt-get install nvidia-driver-560
sudo apt-mark hold nvidia-driver-560
```

**Docker Image Pinning:**

```dockerfile
# Dockerfile with pinned CUDA and cuDNN
FROM nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04

# Pin specific versions
RUN apt-get update && apt-get install -y \
    nvidia-driver-550=550.90.07-0ubuntu1 \
    && apt-mark hold nvidia-driver-550

# Install PyTorch with specific CUDA version
RUN pip install torch==2.3.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Automated Testing

**Driver Validation Test Suite:**

```bash
#!/bin/bash
# test_gpu_driver.sh

set -e

echo "=== GPU Driver Validation Suite ==="

# Test 1: Driver loaded
echo "Test 1: Driver module loaded"
lsmod | grep nvidia || exit 1

# Test 2: nvidia-smi responds
echo "Test 2: nvidia-smi responds"
timeout 10 nvidia-smi || exit 1

# Test 3: CUDA samples
echo "Test 3: CUDA deviceQuery"
cd /tmp
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery
make
./deviceQuery | grep "Result = PASS" || exit 1

# Test 4: PyTorch GPU test
echo "Test 4: PyTorch GPU availability"
python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Test 5: cuDNN test
echo "Test 5: cuDNN functionality"
python3 -c "import torch; x=torch.randn(10,10).cuda(); torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0), torch.randn(1,1,3,3).cuda())" || exit 1

echo "=== All tests passed ==="
```

### Monitoring & Alerting

**Cloud Monitoring Alerts:**

```bash
# Create alert policy for driver issues
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="GPU Driver Failure" \
    --condition-display-name="nvidia-smi timeout" \
    --condition-expression='
        fetch gce_instance
        | metric "compute.googleapis.com/instance/gpu/utilization"
        | filter resource.instance_id == "gpu-vm-1"
        | absent_for 5m
    '
```

### Documentation Requirements

**Driver Configuration Documentation:**

```yaml
# gpu-driver-config.yaml
driver:
  version: "550.90.07"
  source: "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
  installation_method: "apt"
  secure_boot: true

cuda_toolkit:
  version: "12.4.0"
  installation_path: "/usr/local/cuda-12.4"

cudnn:
  version: "9.0.0"
  installation_method: "deb"

verification:
  - "nvidia-smi"
  - "nvcc --version"
  - "python3 -c 'import torch; print(torch.cuda.is_available())'"

last_updated: "2025-01-16"
updated_by: "infrastructure-team"
rollback_snapshot: "pre-driver-update-20250116"
```

---

## Section 8: arr-coc-0-1 Driver Configuration (~50 lines)

### Project-Specific Requirements

**arr-coc-0-1 GPU Driver Setup:**

From [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/cli.py](../../../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/cli.py):

```bash
# Target: Vertex AI A100 (sm_80)
# CUDA: 12.1 (PyTorch 2.1 compatibility)
# cuDNN: 8.9.7 (stable)
# Driver: 550.90.07 (CUDA 12.4 support, forward compatible)

# Installation script for arr-coc-0-1
cat > install_arr_coc_gpu.sh <<'EOF'
#!/bin/bash
set -e

# Install NVIDIA driver 550.90.07
curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
sudo python3 install_gpu_driver.py --driver-version 550.90.07

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit --override

# Install cuDNN 8.9.7
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp -r cudnn-*-archive/include/* /usr/local/cuda-12.1/include/
sudo cp -r cudnn-*-archive/lib/* /usr/local/cuda-12.1/lib64/

# Set environment
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, cuDNN: {torch.backends.cudnn.version()}')"
EOF
```

**Vertex AI Custom Container:**

```dockerfile
# arr-coc-0-1/Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Verify GPU setup
RUN python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# arr-coc texture processing requires TF32 enabled
ENV TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1
ENV TORCH_BACKENDS_CUDNN_ALLOW_TF32=1

# Copy arr-coc application
COPY arr_coc/ /app/arr_coc/
WORKDIR /app

ENTRYPOINT ["python", "-m", "arr_coc.train"]
```

**Why These Versions for arr-coc-0-1:**

1. **CUDA 12.1**: PyTorch 2.1 officially supports CUDA 12.1 (not 12.4)
2. **cuDNN 8.9.7**: Stable release, proven FlashAttention-2 compatibility
3. **Driver 550.x**: Supports CUDA 12.4 (forward compatible with 12.1), TF32 enabled
4. **A100 sm_80**: Optimal for texture processing (fast matrix ops for LAB conversion)

---

## Sources

**Source Documents:**
- [cuda/02-pytorch-build-system-compilation.md](../cuda/02-pytorch-build-system-compilation.md) - CUDA compilation and architecture targeting
- [cuda/03-compute-capabilities-gpu-architectures.md](../cuda/03-compute-capabilities-gpu-architectures.md) - GPU architecture specifications
- [practical-implementation/32-vertex-ai-gpu-tpu.md](../practical-implementation/32-vertex-ai-gpu-tpu.md) - Vertex AI GPU configurations

**Web Research:**
- [Install GPU drivers | Compute Engine](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu) - Google Cloud official documentation (accessed 2025-11-16)
- [GKE can now automatically install NVIDIA GPU drivers](https://cloud.google.com/blog/products/containers-kubernetes/gke-can-now-automatically-install-nvidia-gpu-drivers) - Google Cloud Blog, March 2024 (accessed 2025-11-16)
- [GoogleCloudPlatform/compute-gpu-installation](https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases) - Official GCP GPU installer (accessed 2025-11-16)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) - NVIDIA official compatibility documentation (accessed 2025-11-16)
- [Support Matrix — NVIDIA cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) - cuDNN compatibility matrix (accessed 2025-11-16)
- [Can I update NVIDIA drivers on a live cloud instance?](https://massedcompute.com/faq-answers) - MassedCompute cloud GPU best practices (accessed 2025-11-16)

**Additional References:**
- NVIDIA Driver Downloads - https://www.nvidia.com/Download/index.aspx
- CUDA Toolkit Archive - https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN Archive - https://developer.nvidia.com/cudnn
- PyTorch CUDA Compatibility - https://pytorch.org/get-started/locally/
