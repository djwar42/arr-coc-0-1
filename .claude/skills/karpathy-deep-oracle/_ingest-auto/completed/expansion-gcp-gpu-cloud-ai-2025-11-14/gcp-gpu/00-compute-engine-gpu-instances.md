# Compute Engine GPU Instances: Complete Architecture and Configuration Guide

**Comprehensive guide to GCP Compute Engine GPU instances, machine families, driver management, and production deployment patterns**

---

## Section 1: GPU Machine Families and Architecture (~100 lines)

### Overview of GPU Machine Families

Google Cloud Compute Engine offers specialized machine families optimized for GPU workloads, each designed for specific use cases from cost-effective inference to high-performance training.

From [GPU machine types | Compute Engine Documentation](https://docs.cloud.google.com/compute/docs/gpus) (accessed 2025-11-16):
> "Compute Engine provides graphics processing units (GPUs) that you can add to your virtual machine instances. You can use these GPUs to accelerate specific workloads on your instances such as machine learning and data processing."

**Machine Family Categories:**

1. **A2 Family** (A100-based)
   - **A2-highgpu** series: 1-16 NVIDIA A100 40GB or 80GB GPUs
   - **A2-megagpu** series: 16 A100 80GB GPUs with NVSwitch
   - **A2-ultragpu** series: 8 A100 80GB GPUs optimized configuration
   - Architecture: AMD EPYC Milan processors, up to 12TB RAM
   - Use case: Large-scale model training, HPC workloads

2. **A3 Family** (H100-based)
   - **A3-highgpu-8g**: 8× NVIDIA H100 80GB GPUs
   - **A3-megagpu-8g**: 8× H100 with enhanced networking (3.2 Tbps)
   - Architecture: Intel Sapphire Rapids, NVLink 4.0, NVSwitch
   - Network: 100 Gbps per VM, up to 3.2 Tbps for GPU-to-GPU
   - Use case: Frontier AI model training, trillion-parameter models

3. **G2 Family** (L4-based)
   - **G2-standard** series: 1-8 NVIDIA L4 GPUs
   - Architecture: Intel Cascade Lake or Ice Lake
   - Performance: Cost-optimized for inference and light training
   - Use case: AI inference, video transcoding, graphics workloads

4. **N1 Family** (Legacy GPU support)
   - **N1-standard** with T4, P4, P100, V100, K80 GPUs
   - Flexible vCPU and memory configurations
   - Use case: General-purpose GPU workloads, cost optimization

### GPU-to-Machine Type Compatibility

**A100 80GB Configuration Options:**
```
a2-highgpu-1g:   1 GPU,  12 vCPUs,   85 GB RAM
a2-highgpu-2g:   2 GPUs, 24 vCPUs,  170 GB RAM
a2-highgpu-4g:   4 GPUs, 48 vCPUs,  340 GB RAM
a2-highgpu-8g:   8 GPUs, 96 vCPUs,  680 GB RAM
a2-megagpu-16g: 16 GPUs, 96 vCPUs, 1360 GB RAM (NVSwitch)
a2-ultragpu-8g:  8 GPUs, 96 vCPUs, 1360 GB RAM (optimized)
```

**H100 80GB Configuration:**
```
a3-highgpu-8g: 8 GPUs, 208 vCPUs, 1872 GB RAM
a3-megagpu-8g: 8 GPUs, 208 vCPUs, 1872 GB RAM (3.2 Tbps networking)
```

**L4 Configuration Options:**
```
g2-standard-4:   1 GPU,   4 vCPUs,  16 GB RAM
g2-standard-8:   1 GPU,   8 vCPUs,  32 GB RAM
g2-standard-12:  1 GPU,  12 vCPUs,  48 GB RAM
g2-standard-16:  1 GPU,  16 vCPUs,  64 GB RAM
g2-standard-24:  2 GPUs, 24 vCPUs,  96 GB RAM
g2-standard-32:  1 GPU,  32 vCPUs, 128 GB RAM
g2-standard-48:  4 GPUs, 48 vCPUs, 192 GB RAM
g2-standard-96:  8 GPUs, 96 vCPUs, 384 GB RAM
```

### NVLink and NVSwitch Architecture

From [NVIDIA H200 Vs H100 Vs A100 Comparison](https://acecloud.ai/blog/nvidia-h200-vs-h100-vs-a100-vs-l40s-vs-l4/) (accessed 2025-11-16):
> "The H100 GPU powered by Hopper architecture offers improved per-SM computational power, delivering up to 9x faster AI training and up to 30x faster AI inference compared to the previous generation A100."

**A2 Megagpu NVSwitch Configuration:**
- 16 A100 GPUs connected via NVSwitch
- 600 GB/s bidirectional bandwidth per GPU
- Total aggregate bandwidth: 9.6 TB/s
- Enables single unified memory space across all GPUs

**A3 NVLink 4.0 Configuration:**
- 8 H100 GPUs with NVLink 4.0
- 900 GB/s bidirectional bandwidth per GPU (NVLink)
- 3.2 Tbps inter-VM GPU-to-GPU networking
- GPUDirect RDMA for zero-copy GPU memory access

---

## Section 2: GPU Types and Specifications (~120 lines)

### NVIDIA H100 (Hopper Architecture)

**Hardware Specifications:**
- **Architecture**: Hopper (5nm TSMC)
- **CUDA Cores**: 14,592 (132 SMs)
- **Tensor Cores**: 456 (4th generation)
- **Memory**: 80GB HBM3 (3.35 TB/s bandwidth)
- **TF32 Performance**: 1,979 TFLOPS (training)
- **FP16/BF16 Performance**: 3,958 TFLOPS
- **INT8 Performance**: 7,916 TOPS (inference)
- **FP8 Support**: Yes (Transformer Engine)
- **TDP**: 700W
- **NVLink**: 900 GB/s bidirectional

**Key Features:**
- **Transformer Engine**: FP8 mixed precision for 6x faster training
- **Multi-Instance GPU (MIG)**: Up to 7 independent GPU instances
- **DPX Instructions**: Dynamic programming acceleration for genomics/routing
- **Secure Boot**: Hardware root of trust

**Ideal Workloads:**
- Large language model training (GPT-4 scale)
- Diffusion models (Stable Diffusion XL, DALL-E)
- Recommender systems with trillion-parameter embeddings
- Scientific computing (molecular dynamics, climate modeling)

### NVIDIA A100 (Ampere Architecture)

**Hardware Specifications:**
- **Architecture**: Ampere (7nm TSMC)
- **CUDA Cores**: 6,912 (108 SMs)
- **Tensor Cores**: 432 (3rd generation)
- **Memory Options**: 40GB or 80GB HBM2e (1.6-2.0 TB/s)
- **TF32 Performance**: 156 TFLOPS (training)
- **FP16/BF16 Performance**: 312 TFLOPS
- **INT8 Performance**: 624 TOPS
- **TDP**: 400W (PCIe) / 500W (SXM)
- **NVLink**: 600 GB/s bidirectional

**Key Features:**
- **Multi-Instance GPU (MIG)**: Up to 7 instances (isolate workloads)
- **Structural Sparsity**: 2:4 sparsity for 2x performance
- **TF32 Precision**: No code changes for 10x speedup over FP32
- **Third-gen Tensor Cores**: BF16 support for stable training

**Ideal Workloads:**
- BERT, GPT-3, T5 training (up to 175B parameters)
- Computer vision (ResNet, EfficientNet, YOLO)
- Recommendation engines (DLRM)
- Conversational AI

### NVIDIA L4 (Ada Lovelace Architecture)

**Hardware Specifications:**
- **Architecture**: Ada Lovelace (5nm TSMC)
- **CUDA Cores**: 7,424
- **Tensor Cores**: 232 (4th generation)
- **Memory**: 24GB GDDR6 (300 GB/s bandwidth)
- **TF32 Performance**: 121 TFLOPS
- **FP16 Performance**: 242 TFLOPS
- **INT8 Performance**: 485 TOPS
- **FP8 Performance**: 242 TFLOPS
- **TDP**: 72W (ultra power-efficient)

**Key Features:**
- **NVIDIA Ada Architecture**: 2x AI performance per watt vs A10
- **Video Encoding**: Dual 5th-gen NVENC, AV1 encoding
- **Ray Tracing**: 3rd gen RT cores for graphics
- **Low Power**: 72W TDP enables high-density deployment

**Ideal Workloads:**
- Real-time inference (LLM serving with vLLM, TensorRT-LLM)
- Video transcoding and streaming (AV1, H.265)
- Graphics rendering (NVIDIA RTX Virtual Workstation)
- Edge AI deployment

### NVIDIA T4 (Turing Architecture)

**Hardware Specifications:**
- **Architecture**: Turing (12nm TSMC)
- **CUDA Cores**: 2,560
- **Tensor Cores**: 320 (2nd generation)
- **Memory**: 16GB GDDR6 (320 GB/s)
- **FP16 Performance**: 65 TFLOPS
- **INT8 Performance**: 130 TOPS
- **INT4 Performance**: 260 TOPS
- **TDP**: 70W

**Ideal Workloads:**
- Cost-effective inference serving
- Small-scale training (BERT-base, ResNet-50)
- Video analytics pipelines
- Development and testing environments

### NVIDIA V100 (Volta Architecture - Legacy)

**Hardware Specifications:**
- **Architecture**: Volta (12nm TSMC)
- **CUDA Cores**: 5,120
- **Tensor Cores**: 640 (1st generation)
- **Memory**: 16GB or 32GB HBM2 (900 GB/s)
- **FP16 Performance**: 112 TFLOPS
- **TDP**: 300W

**Note**: V100 is legacy; prefer A100 or H100 for new workloads.

### Performance Comparison Matrix

| GPU    | TF32 TFLOPS | FP16 TFLOPS | Memory | Bandwidth | TDP  | Best For                |
|--------|-------------|-------------|--------|-----------|------|-------------------------|
| H100   | 1,979       | 3,958       | 80GB   | 3.35 TB/s | 700W | Frontier AI training    |
| A100   | 156         | 312         | 80GB   | 2.0 TB/s  | 500W | Large-scale training    |
| L4     | 121         | 242         | 24GB   | 300 GB/s  | 72W  | Inference + video       |
| T4     | -           | 65          | 16GB   | 320 GB/s  | 70W  | Cost-effective inference|
| V100   | -           | 112         | 32GB   | 900 GB/s  | 300W | Legacy workloads        |

---

## Section 3: Quota Management and Regional Availability (~100 lines)

### Understanding GPU Quota Structure

From [Allocation quotas | Compute Engine Documentation](https://docs.cloud.google.com/compute/resource-usage) (accessed 2025-11-16):
> "New accounts and projects have a global GPU quota that applies to all regions. Similar to virtual CPU quota, your GPU quota refers to the total number of virtual CPUs across all VM instances in a region."

**Quota Types:**

1. **Global GPU Quotas** (applies to all regions)
   - `GPUS_ALL_REGIONS`: Total GPUs across all zones
   - Default for new accounts: 0 (must request increase)
   - Example: 8 GPUs globally means 8 total, not 8 per region

2. **Regional GPU Quotas** (per-region limits)
   - `NVIDIA_A100_GPUS`: A100 GPUs in specific region
   - `NVIDIA_H100_GPUS`: H100 GPUs in specific region
   - `NVIDIA_L4_GPUS`: L4 GPUs in specific region
   - `NVIDIA_T4_GPUS`: T4 GPUs in specific region
   - `NVIDIA_V100_GPUS`: V100 GPUs in specific region

3. **Preemptible GPU Quotas** (separate allocation)
   - `PREEMPTIBLE_GPUS`: Separate quota for preemptible instances
   - Typically higher limits (encourages Spot usage)
   - Does not count against on-demand quota

### GPU Availability by Region and Zone

From [List of GPU available in the different GCP Zones](https://holori.com/list-of-gpu-available-in-the-different-gcp-zones/) (accessed 2025-11-16):
> "The diversity of GPU models offered by GCP, such as the NVIDIA L4, T4, P4, P100, V100, A100, and H100, provides tailored solutions for different computational needs."

**High-Availability Regions (H100 + A100):**
```
us-central1 (Iowa):        H100, A100, L4, T4, V100
us-east4 (Virginia):       H100, A100, L4, T4
europe-west4 (Netherlands): H100, A100, L4, T4
asia-southeast1 (Singapore): A100, L4, T4
```

**A100-Heavy Regions:**
```
us-west1 (Oregon):         A100, L4, T4, V100
us-west4 (Las Vegas):      A100, L4, T4
europe-west1 (Belgium):    A100, L4, T4, V100
asia-northeast1 (Tokyo):   A100, L4, T4
```

**L4-Optimized Regions (inference workloads):**
```
us-south1 (Dallas):        L4, T4
europe-west2 (London):     L4, T4
asia-south1 (Mumbai):      L4, T4
```

**Limited Availability (check current status):**
```
H100: Currently limited to us-central1, us-east4, europe-west4
A3 Mega (3.2 Tbps): Very limited availability
```

### Requesting Quota Increases

**Quota Increase Process:**

1. **Navigate to IAM & Admin → Quotas**
2. **Filter by:**
   - Metric: Select GPU type (e.g., "NVIDIA A100 GPUs")
   - Location: Choose specific region or "Global"
3. **Select quota and click "Edit Quotas"**
4. **Provide justification:**
   - Workload description (training/inference)
   - Expected GPU utilization (hours per day)
   - Business impact if not approved
   - Timeline for usage

**Approval Timeframes:**
- **T4, L4**: Typically approved within 24-48 hours
- **A100**: 2-5 business days
- **H100**: 5-10 business days (requires detailed justification)
- **Large requests (>32 GPUs)**: May require account review

**Best Practices for Quota Requests:**
- Start with smaller quota (8-16 GPUs) and expand
- Request in multiple regions for redundancy
- Include technical details (model size, batch size, training duration)
- Demonstrate cost awareness (preemptible usage plans)

### Quota Monitoring and Alerts

**Cloud Monitoring Metrics:**
```python
# Monitor GPU quota usage
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Query GPU quota utilization
query = """
fetch compute.googleapis.com/quota/allocation/usage
| filter resource.quota_metric == 'NVIDIA_A100_GPUS'
| group_by 1m, [value_usage_mean: mean(value.usage)]
"""
```

**Alert Policy Example:**
```yaml
displayName: "GPU Quota Near Limit"
conditions:
  - displayName: "A100 quota >80%"
    conditionThreshold:
      filter: resource.type="compute.googleapis.com/Quota"
      comparison: COMPARISON_GT
      thresholdValue: 0.8
      duration: 300s
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
```

---

## Section 4: NVIDIA Driver Installation and Management (~100 lines)

### Driver Installation Methods

From [Install GPU drivers | Compute Engine Documentation](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu) (accessed 2025-11-16):
> "After creating a virtual machine (VM) with one or more GPUs, your system requires NVIDIA device drivers for the GPUs to function properly."

**Three Installation Approaches:**

1. **Automated Driver Installation (Recommended)**
2. **Manual Installation via Repository**
3. **Custom Driver Compilation**

### Method 1: Automated Installation

**Using GCP Startup Script:**
```bash
#!/bin/bash
# Add to VM metadata as startup-script

# Detect OS and install drivers
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
```

**Using gcloud CLI:**
```bash
gcloud compute instances create gpu-instance-1 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB
```

**Deep Learning VM Images (Pre-installed drivers):**
```bash
gcloud compute instances create ml-workstation \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB
```

**Pre-installed software:**
- NVIDIA Driver 535.x or 550.x
- CUDA Toolkit 12.1 or 12.2
- cuDNN 8.9
- PyTorch 2.1+ or TensorFlow 2.15+
- JupyterLab

### Method 2: Manual Installation

**Ubuntu 20.04/22.04:**
```bash
# Update package list
sudo apt-get update

# Install kernel headers
sudo apt-get install -y linux-headers-$(uname -r)

# Add NVIDIA driver repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install specific driver version (H100/A100: 535.x+)
sudo apt-get install -y nvidia-driver-550

# Install CUDA Toolkit 12.2
sudo apt-get install -y cuda-12-2

# Install cuDNN 8.9
sudo apt-get install -y libcudnn8=8.9.7.29-1+cuda12.2 \
                         libcudnn8-dev=8.9.7.29-1+cuda12.2

# Reboot to load driver
sudo reboot
```

**Verify Installation:**
```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
# | N/A   32C    P0    45W / 400W |      0MiB / 81920MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Verify CUDA
nvcc --version

# Test CUDA compilation
cat <<EOF > test.cu
#include <stdio.h>
__global__ void hello() { printf("Hello from GPU!\\n"); }
int main() { hello<<<1, 1>>>(); cudaDeviceSynchronize(); return 0; }
EOF
nvcc test.cu -o test && ./test
```

### Driver Version Compatibility

**CUDA Toolkit and Driver Requirements:**

| CUDA Version | Min Driver (Linux) | Min Driver (Windows) | GPU Support              |
|--------------|-------------------|---------------------|--------------------------|
| CUDA 12.4    | 550.54.15         | 551.61              | H100, A100, L4, T4       |
| CUDA 12.2    | 535.54.03         | 536.25              | H100, A100, L4, T4       |
| CUDA 12.1    | 530.30.02         | 531.14              | A100, L4, T4, V100       |
| CUDA 11.8    | 520.61.05         | 520.06              | A100, T4, V100, P100     |

**Compute Capability Requirements:**
- **H100**: sm_90 (Hopper)
- **A100**: sm_80 (Ampere)
- **L4**: sm_89 (Ada Lovelace)
- **T4**: sm_75 (Turing)
- **V100**: sm_70 (Volta)

### Driver Updates and Rollback

**Update to Latest Driver:**
```bash
# Ubuntu
sudo apt-get update
sudo apt-get upgrade nvidia-driver-550

# Reboot required
sudo reboot
```

**Rollback to Previous Version:**
```bash
# List available driver versions
apt list -a nvidia-driver-*

# Install specific version
sudo apt-get install nvidia-driver-535=535.54.03-0ubuntu1

# Pin version to prevent auto-update
sudo apt-mark hold nvidia-driver-535
```

---

## Section 5: Persistent Disk Attachment and Performance (~80 lines)

### Storage Options for GPU Workloads

**Persistent Disk Types:**
1. **Persistent Disk SSD (pd-ssd)**: Balanced performance
2. **Persistent Disk Balanced (pd-balanced)**: Cost-optimized
3. **Persistent Disk Extreme (pd-extreme)**: Provisioned IOPS
4. **Hyperdisk Extreme**: Highest performance (up to 350K IOPS)
5. **Local SSD**: Ultra-low latency (ephemeral)

### Local SSD Performance

From [Persistent Disk performance overview](https://docs.cloud.google.com/compute/docs/disks/performance) (accessed 2025-11-16):
> "Persistent Disks give you the performance described in the disk type chart if the VM drives usage that is sufficient to reach the performance limits."

**Local SSD Specifications:**
- **Interface**: NVMe
- **Capacity**: 375 GB per device (up to 24 devices = 9 TB total)
- **Read IOPS**: 2.4 million IOPS (4KB random reads)
- **Write IOPS**: 1.2 million IOPS (4KB random writes)
- **Read Throughput**: 9.6 GB/s
- **Write Throughput**: 4.8 GB/s
- **Latency**: <1ms

**Attach Local SSD to GPU Instance:**
```bash
gcloud compute instances create gpu-training-vm \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-8g \
  --accelerator=type=nvidia-tesla-a100,count=8 \
  --local-ssd=interface=NVME \
  --local-ssd=interface=NVME \
  --local-ssd=interface=NVME \
  --local-ssd=interface=NVME \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release
```

**Format and Mount Local SSD:**
```bash
# Create RAID 0 for maximum throughput
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 \
  /dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4

# Format with ext4
sudo mkfs.ext4 -F /dev/md0

# Mount
sudo mkdir -p /mnt/localssd
sudo mount /dev/md0 /mnt/localssd
sudo chmod 777 /mnt/localssd

# Verify performance
sudo fio --name=randread --ioengine=libaio --iodepth=64 --rw=randread \
  --bs=4k --direct=1 --size=10G --numjobs=4 --runtime=60 \
  --filename=/mnt/localssd/test
```

### Persistent Disk for Checkpoints

**Persistent Disk SSD Performance:**
- **Read IOPS**: 30 IOPS per GB (max 100K IOPS)
- **Write IOPS**: 30 IOPS per GB (max 100K IOPS)
- **Read Throughput**: 0.48 MB/s per GB (max 1,200 MB/s)
- **Write Throughput**: 0.48 MB/s per GB (max 1,200 MB/s)

**Create Persistent Disk for Checkpoints:**
```bash
# Create 1 TB SSD for checkpoint storage
gcloud compute disks create training-checkpoints \
  --size=1000GB \
  --type=pd-ssd \
  --zone=us-central1-a

# Attach to running instance
gcloud compute instances attach-disk gpu-training-vm \
  --disk=training-checkpoints \
  --zone=us-central1-a
```

**Mount and Configure:**
```bash
# Format and mount
sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/checkpoints
sudo mount /dev/sdb /mnt/checkpoints
sudo chmod 777 /mnt/checkpoints

# Configure automatic snapshots (daily backup)
gcloud compute resource-policies create snapshot-schedule daily-checkpoints \
  --region=us-central1 \
  --max-retention-days=7 \
  --start-time=02:00 \
  --daily-schedule

gcloud compute disks add-resource-policies training-checkpoints \
  --resource-policies=daily-checkpoints \
  --zone=us-central1-a
```

---

## Section 6: Network Performance Optimization (~70 lines)

### GPU Instance Network Architecture

**Standard GPU Instance Networking:**
- **Bandwidth**: 32-100 Gbps (depends on machine type)
- **A2 Family**: Up to 100 Gbps
- **A3 Family**: 200 Gbps standard, 3.2 Tbps for GPU-to-GPU (A3 Mega)
- **Tier 1 Network**: Google's private global network

### A3 Mega GPU-to-GPU Networking

**Architecture:**
- 8 H100 GPUs per VM
- GPUDirect RDMA enabled
- 3.2 Tbps aggregate bandwidth for multi-node training
- NCCL optimized communication

**Enable GPUDirect RDMA:**
```bash
# Load NVIDIA Peer Memory module
sudo modprobe nvidia-peermem

# Verify
lsmod | grep nvidia_peermem

# Test GPUDirect bandwidth
/usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
```

### Compact Placement Policy

**Create Placement Policy:**
```bash
gcloud compute resource-policies create group-placement compact-gpu-cluster \
  --region=us-central1 \
  --collocation=COLLOCATED

# Create instance group with placement policy
gcloud compute instance-groups managed create gpu-training-group \
  --zone=us-central1-a \
  --size=16 \
  --template=gpu-training-template \
  --resource-policies=compact-gpu-cluster
```

**Benefits:**
- 10-30% reduction in inter-node latency
- 15-25% improvement in collective communication (AllReduce)
- Co-locates VMs on same physical rack

### Network Bandwidth Monitoring

**Monitor Network Throughput:**
```bash
# Install monitoring tools
sudo apt-get install -y iftop nethogs

# Real-time network monitoring
sudo iftop -i eth0

# Per-process network usage
sudo nethogs eth0
```

**Cloud Monitoring Metrics:**
```python
from google.cloud import monitoring_v3

# Query network sent bytes
query = """
fetch gce_instance
| metric 'compute.googleapis.com/instance/network/sent_bytes_count'
| group_by 1m, [value_sent_bytes_mean: mean(value.sent_bytes)]
| every 1m
"""
```

---

## Section 7: Cost Analysis and Optimization (~80 lines)

### GPU Pricing Overview (us-central1, as of 2025)

**On-Demand Pricing (per GPU per hour):**
```
H100 80GB:    $31.69
A100 80GB:    $15.73
A100 40GB:    $12.24
L4:           $2.21
T4:           $1.35
V100 16GB:    $2.48
```

**Preemptible/Spot Pricing (60-91% discount):**
```
H100 80GB:    ~$9.51  (70% savings)
A100 80GB:    ~$5.51  (65% savings)
A100 40GB:    ~$4.29  (65% savings)
L4:           ~$0.77  (65% savings)
T4:           ~$0.41  (70% savings)
```

**Committed Use Discounts (1-year commitment):**
```
A100 80GB: $10.18/hour (35% savings)
T4:        $0.88/hour  (35% savings)
```

### Cost Comparison Example: 7B Model Training

**Scenario**: Train LLaMA 7B model (1 epoch, 1T tokens)

**Option 1: 8× A100 80GB (on-demand)**
```
Training time: ~168 hours (1 week)
Cost: 8 GPUs × $15.73/hr × 168 hrs = $21,142
```

**Option 2: 8× A100 80GB (Spot with checkpointing)**
```
Training time: ~200 hours (accounting for preemptions)
Cost: 8 GPUs × $5.51/hr × 200 hrs = $8,816
Savings: $12,326 (58% reduction)
```

**Option 3: 16× L4 (inference optimization)**
```
For inference serving only (not training)
Monthly cost: 16 GPUs × $2.21/hr × 730 hrs = $25,843/month
Handles ~2M requests/day (50ms latency)
```

### Cost Optimization Strategies

**1. Preemptible GPU Strategy:**
```bash
# Create preemptible instance
gcloud compute instances create preemptible-gpu \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-8g \
  --accelerator=type=nvidia-tesla-a100,count=8 \
  --preemptible \
  --metadata=startup-script='#!/bin/bash
    # Resume training from checkpoint
    cd /mnt/localssd/training
    python train.py --resume-from-checkpoint=/mnt/checkpoints/latest
  '
```

**2. Right-Sizing GPU Selection:**
```python
# Decision matrix
def select_gpu(workload_type, model_size):
    if workload_type == "inference":
        if model_size < "7B":
            return "T4"  # $0.41/hr spot
        elif model_size < "70B":
            return "L4"  # $0.77/hr spot
        else:
            return "A100"  # $5.51/hr spot

    elif workload_type == "training":
        if model_size < "7B":
            return "L4"  # Cost-effective
        elif model_size < "70B":
            return "A100"  # Balanced
        else:
            return "H100"  # Maximum performance
```

**3. Scheduled Shutdown:**
```bash
# Shutdown VM after training completes
gcloud compute instances add-metadata gpu-training-vm \
  --zone=us-central1-a \
  --metadata=shutdown-script='#!/bin/bash
    # Check if training complete
    if [ -f /mnt/checkpoints/training_complete ]; then
      gcloud compute instances stop $(hostname) --zone=us-central1-a
    fi
  '
```

---

## Section 8: arr-coc-0-1 Single-GPU Training Configuration (~70 lines)

### Project-Specific GPU Setup

**arr-coc-0-1 Architecture Requirements:**
- **Model**: Qwen3-VL integration with relevance realization
- **Training Data**: Visual reasoning datasets (GQA, VQA, OK-VQA)
- **Batch Size**: 32 (effective batch size with gradient accumulation)
- **Memory Requirements**: ~45 GB GPU memory per training step
- **Recommended GPU**: A100 80GB (single GPU sufficient for MVP)

### Vertex AI Custom Training Job

**Training Container (Dockerfile):**
```dockerfile
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-1.py310

# Install arr-coc dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training code
COPY arr_coc/ /workspace/arr_coc/
COPY training/ /workspace/training/

WORKDIR /workspace

# Training entry point
ENTRYPOINT ["python", "training/train.py"]
```

**Submit Training Job:**
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=arr-coc-training-$(date +%Y%m%d-%H%M%S) \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=gcr.io/PROJECT_ID/arr-coc-trainer:latest \
  --args="--epochs=10,--batch-size=32,--checkpoint-dir=gs://BUCKET/checkpoints"
```

**Training Configuration (`training/config.yaml`):**
```yaml
model:
  base_model: "Qwen/Qwen-VL-Chat"
  freeze_vision_encoder: false
  lora_config:
    r: 64
    lora_alpha: 128
    target_modules: ["q_proj", "v_proj", "k_proj"]

training:
  epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch 32
  learning_rate: 2e-5
  warmup_steps: 500
  fp16: false  # A100 prefers bf16
  bf16: true

gpu_optimization:
  gradient_checkpointing: true  # Save memory
  flash_attention_2: true       # 2x speedup
  compile: false                # Skip torch.compile (stability)

data:
  train_dataset: "gs://arr-coc-data/gqa_train.jsonl"
  val_dataset: "gs://arr-coc-data/gqa_val.jsonl"
  num_workers: 4
  prefetch_factor: 2

checkpointing:
  save_every_n_steps: 500
  checkpoint_dir: "gs://arr-coc-checkpoints/run-{timestamp}"
  keep_last_n: 3
```

**Training Script (`training/train.py`):**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import deepspeed

def main():
    # Load model on A100
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        torch_dtype=torch.bfloat16,  # A100 native precision
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )

    # LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_config)

    # Training loop with gradient accumulation
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Checkpoint every 500 steps
            if (step + 1) % 500 == 0:
                save_checkpoint(model, f"gs://bucket/ckpt-{step}")
```

**Expected Performance:**
- **Training Speed**: ~12 samples/second (A100 80GB)
- **Training Time**: ~24 hours for 10 epochs (GQA dataset)
- **Memory Usage**: ~48 GB GPU memory (with gradient checkpointing)
- **Cost**: $15.73/hr × 24 hrs = $377 (on-demand) or $132 (Spot)

---

## Sources

**Google Cloud Documentation:**
- [GPU machine types | Compute Engine](https://docs.cloud.google.com/compute/docs/gpus) (accessed 2025-11-16)
- [Install GPU drivers | Compute Engine](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu) (accessed 2025-11-16)
- [Allocation quotas | Compute Engine](https://docs.cloud.google.com/compute/resource-usage) (accessed 2025-11-16)
- [Persistent Disk performance overview](https://docs.cloud.google.com/compute/docs/disks/performance) (accessed 2025-11-16)
- [Regions and zones | Compute Engine](https://docs.cloud.google.com/compute/docs/regions-zones) (accessed 2025-11-16)

**Third-Party Resources:**
- [NVIDIA H200 Vs H100 Vs A100 Comparison](https://acecloud.ai/blog/nvidia-h200-vs-h100-vs-a100-vs-l40s-vs-l4/) - AceCloud (accessed 2025-11-16)
- [List of GPU available in the different GCP Zones](https://holori.com/list-of-gpu-available-in-the-different-gcp-zones/) - Holori (accessed 2025-11-16)
- [GCP Cloud GPUs Pricing Comparison](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/) - Economize Cloud (accessed 2025-11-16)

**Related Knowledge:**
- [distributed-training/00-deepspeed-zero-optimizer.md](../../karpathy/distributed-training/00-deepspeed-zero-optimizer.md) - GPU memory optimization patterns
- [inference-optimization/00-tensorrt-fundamentals.md](../../karpathy/inference-optimization/00-tensorrt-fundamentals.md) - GPU inference acceleration
- [alternative-hardware/03-tpu-programming-fundamentals.md](../../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md) - GPU vs TPU comparison
