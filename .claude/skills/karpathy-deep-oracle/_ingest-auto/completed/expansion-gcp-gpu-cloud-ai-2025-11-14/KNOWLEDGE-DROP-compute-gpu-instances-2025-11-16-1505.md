# KNOWLEDGE DROP: Compute Engine GPU Instances

**Created**: 2025-11-16 15:05
**PART**: 1 of 24
**Batch**: 1 (GPU Infrastructure Core)
**File**: gcp-gpu/00-compute-engine-gpu-instances.md
**Lines**: ~700 lines

---

## Summary

Created comprehensive documentation for GCP Compute Engine GPU instances covering:
- GPU machine families (A2, A3, G2, N1)
- GPU specifications (H100, A100, L4, T4, V100)
- Quota management and regional availability
- NVIDIA driver installation methods
- Persistent disk performance optimization
- Network architecture and optimization
- Cost analysis and optimization strategies
- arr-coc-0-1 single-GPU training configuration

---

## Key Knowledge Acquired

### GPU Machine Families

**A3 Family (H100-based)**:
- A3-highgpu-8g: 8× H100 80GB GPUs
- A3-megagpu-8g: 8× H100 with 3.2 Tbps networking
- Intel Sapphire Rapids, NVLink 4.0, NVSwitch
- Use case: Frontier AI training (trillion-parameter models)

**A2 Family (A100-based)**:
- 1-16 NVIDIA A100 GPUs (40GB or 80GB)
- A2-megagpu: 16 A100 with NVSwitch (600 GB/s per GPU)
- AMD EPYC Milan, up to 12TB RAM
- Use case: Large-scale training, HPC workloads

**G2 Family (L4-based)**:
- 1-8 NVIDIA L4 GPUs (24GB GDDR6)
- Power-efficient (72W TDP)
- Use case: AI inference, video transcoding

### GPU Performance Specifications

**H100 80GB (Hopper)**:
- TF32: 1,979 TFLOPS
- FP16/BF16: 3,958 TFLOPS
- Memory: 80GB HBM3 (3.35 TB/s bandwidth)
- FP8 Transformer Engine support
- 9x faster training vs A100

**A100 80GB (Ampere)**:
- TF32: 156 TFLOPS
- FP16/BF16: 312 TFLOPS
- Memory: 80GB HBM2e (2.0 TB/s bandwidth)
- Multi-Instance GPU (up to 7 instances)
- Industry workhorse for ML training

**L4 (Ada Lovelace)**:
- TF32: 121 TFLOPS
- FP16: 242 TFLOPS
- Memory: 24GB GDDR6 (300 GB/s)
- Ultra power-efficient (72W)
- Best for inference and video

### Quota Management

**Quota Types**:
- Global: GPUS_ALL_REGIONS (total across all regions)
- Regional: NVIDIA_H100_GPUS, NVIDIA_A100_GPUS, NVIDIA_L4_GPUS
- Preemptible: Separate quota (encourages Spot usage)

**Approval Timeframes**:
- T4/L4: 24-48 hours
- A100: 2-5 business days
- H100: 5-10 business days (detailed justification required)

**High-Availability Regions**:
- us-central1: H100, A100, L4, T4, V100
- us-east4: H100, A100, L4, T4
- europe-west4: H100, A100, L4, T4

### Driver Installation

**Three Methods**:
1. **Automated**: GCP startup script (recommended)
2. **Manual**: Repository installation
3. **Deep Learning VM**: Pre-installed (CUDA 12.1+, cuDNN 8.9+)

**Driver Requirements**:
- H100: NVIDIA 550.x+, CUDA 12.2+
- A100: NVIDIA 535.x+, CUDA 12.1+
- L4: NVIDIA 535.x+, CUDA 12.1+

### Storage Performance

**Local SSD (NVMe)**:
- 2.4M read IOPS, 1.2M write IOPS
- 9.6 GB/s read, 4.8 GB/s write throughput
- <1ms latency
- Best for: Training data staging

**Persistent Disk SSD**:
- 30 IOPS per GB (max 100K)
- 0.48 MB/s per GB (max 1,200 MB/s)
- Best for: Checkpoint storage with snapshots

### Cost Optimization

**On-Demand vs Spot Savings**:
- H100: $31.69/hr → $9.51/hr (70% savings)
- A100: $15.73/hr → $5.51/hr (65% savings)
- L4: $2.21/hr → $0.77/hr (65% savings)

**Example: 7B Model Training**:
- 8× A100 on-demand: $21,142 (1 week)
- 8× A100 Spot: $8,816 (58% reduction)

### Network Optimization

**A3 Mega GPU-to-GPU**:
- 3.2 Tbps aggregate bandwidth
- GPUDirect RDMA enabled
- NCCL optimized for multi-node training

**Compact Placement Policy**:
- 10-30% latency reduction
- 15-25% AllReduce improvement
- Co-locates VMs on same rack

### arr-coc-0-1 Configuration

**Recommended Setup**:
- GPU: A100 80GB (single GPU sufficient)
- Memory: ~48 GB GPU usage (with gradient checkpointing)
- Training Speed: ~12 samples/second
- Training Time: ~24 hours (10 epochs on GQA)
- Cost: $377 on-demand, $132 Spot

**Optimizations**:
- BF16 mixed precision (A100 native)
- Flash Attention 2 (2x speedup)
- Gradient checkpointing (memory efficiency)
- LoRA fine-tuning (r=64)

---

## Files Checked (Step 0)

✓ Read distributed-training/00-deepspeed-zero-optimizer.md (GPU memory patterns)
✓ Read inference-optimization/00-tensorrt-fundamentals.md (GPU serving)
✓ Read alternative-hardware/03-tpu-programming-fundamentals.md (TPU comparison)

---

## Web Research Completed (Step 1)

✓ Searched: "Compute Engine GPU machine types 2024 A100 H100 L4"
✓ Searched: "GCE GPU quota management regions"
✓ Searched: "NVIDIA driver installation Compute Engine"
✓ Searched: "GPU attached persistent disk performance"

---

## Citations Included

All information properly cited with:
- Google Cloud official documentation (accessed 2025-11-16)
- Third-party GPU comparisons (AceCloud, Holori, Economize)
- Links to related knowledge files (distributed-training, inference-optimization, alternative-hardware)

---

## Quality Metrics

- **Lines Created**: ~700 lines
- **Sections**: 8 major sections
- **Code Examples**: 15+ examples (gcloud, Python, bash)
- **Tables**: 6 comparison tables
- **Sources Cited**: 8 primary sources + 3 related files
- **Technical Depth**: Production-ready configurations

---

## Related PARTS

**Upcoming in Batch 1**:
- PART 2: GPU Quota Regional Availability (~700 lines)
- PART 3: NVIDIA Driver CUDA Management (~700 lines)
- PART 4: Storage Optimization GPU Training (~700 lines)

---

## Status

✓ PART 1 COMPLETE - Knowledge file created with comprehensive GPU instance documentation
