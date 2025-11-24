# GCP Machine Types for Spot Instances: Complete Selection Guide

**Comprehensive machine type selection for cost-optimized ML training with spot instances**

---

## Overview

Choosing the right GCP machine type for spot instances involves balancing CPU, memory, GPU availability, and cost. This guide covers all machine families available with spot pricing, their optimal use cases for ML workloads, and practical selection strategies for production training pipelines.

**Key considerations:**
- General-purpose machines (N1, N2, N2D, N4) for CPU-intensive preprocessing
- Compute-optimized machines (C2, C2D, C3, H3) for inference serving
- Memory-optimized machines (M1, M2, M3, M4) for large dataset handling
- Accelerator-optimized machines (A2, A3, G2) for GPU-attached training
- Spot pricing varies by machine family (60-91% discounts)

From [GCP VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) (accessed 2025-01-31):
- Spot pricing available for most machine families
- Dynamic pricing can change up to once every 30 days
- No sustained use discounts (SUDs) apply to spot instances
- Committed use discounts (CUDs) don't apply to spot VMs

---

## Section 1: General-Purpose Machine Types (N1, N2, N2D, E2, N4)

General-purpose machines provide balanced CPU and memory ratios for diverse ML workloads. These are the workhorses for data preprocessing, experimentation, and CPU-bound training tasks.

### N1 Machine Series (Legacy, Still Widely Available)

**Overview:**
- Original Compute Engine machine type (2014+)
- Intel Skylake, Broadwell, Haswell, Sandy Bridge, Ivy Bridge CPUs
- Predefined types: standard, highmem, highcpu
- Custom machine types: 1-96 vCPUs, 0.9-624 GB memory
- Extensive spot availability across all regions

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
n1-standard-4       4       15 GB    $0.190/hr    $0.076/hr   60%
n1-standard-8       8       30 GB    $0.380/hr    $0.152/hr   60%
n1-standard-16      16      60 GB    $0.760/hr    $0.304/hr   60%
n1-standard-32      32      120 GB   $1.520/hr    $0.608/hr   60%
n1-standard-96      96      360 GB   $4.560/hr    $1.824/hr   60%

n1-highmem-8        8       52 GB    $0.475/hr    $0.190/hr   60%
n1-highmem-32       32      208 GB   $1.900/hr    $0.760/hr   60%
n1-highmem-96       96      624 GB   $5.700/hr    $2.280/hr   60%

n1-highcpu-16       16      14.4 GB  $0.565/hr    $0.226/hr   60%
n1-highcpu-32       32      28.8 GB  $1.130/hr    $0.452/hr   60%
n1-highcpu-96       96      86.4 GB  $3.390/hr    $1.356/hr   60%
```

**ML Use Cases:**
- **Data preprocessing:** n1-highcpu-32 for CPU-bound ETL pipelines
- **Batch inference:** n1-standard-16 for model serving without GPU
- **Experimentation:** n1-standard-4/8 for rapid prototyping
- **Distributed preprocessing:** Fleets of n1-standard-8 on spot

**Spot Considerations:**
- High availability (legacy support = good spot capacity)
- 60% discount across all N1 types
- GPU attachment supported (NVIDIA K80, P4, P100, V100, T4)
- Good fallback option when newer machines unavailable

From [GCP Machine Types Documentation](https://cloud.google.com/compute/docs/machine-types) (accessed 2025-01-31):
- N1 supports up to 8 GPUs per instance
- Custom machine types allow fine-tuned resource allocation
- Maximum network bandwidth: 32 Gbps (n1-standard-32+)

---

### N2 Machine Series (Balanced Performance)

**Overview:**
- 2nd generation Intel Xeon Scalable (Cascade Lake)
- Improved price-performance vs N1 (25-35% better)
- Predefined: standard, highmem, highcpu
- Custom: 2-128 vCPUs, 0.5-864 GB memory
- Excellent spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
n2-standard-4       4       16 GB    $0.195/hr    $0.047/hr   76%
n2-standard-8       8       32 GB    $0.389/hr    $0.093/hr   76%
n2-standard-16      16      64 GB    $0.778/hr    $0.187/hr   76%
n2-standard-32      32      128 GB   $1.556/hr    $0.373/hr   76%
n2-standard-64      64      256 GB   $3.112/hr    $0.747/hr   76%
n2-standard-128     128     512 GB   $6.224/hr    $1.494/hr   76%

n2-highmem-8        8       64 GB    $0.521/hr    $0.125/hr   76%
n2-highmem-32       32      256 GB   $2.083/hr    $0.500/hr   76%
n2-highmem-128      128     864 GB   $8.333/hr    $2.000/hr   76%

n2-highcpu-16       16      16 GB    $0.593/hr    $0.142/hr   76%
n2-highcpu-32       32      32 GB    $1.186/hr    $0.285/hr   76%
n2-highcpu-96       96      96 GB    $3.558/hr    $0.854/hr   76%
```

**ML Use Cases:**
- **Modern preprocessing:** n2-highcpu-32 for fast CPU pipelines
- **Data augmentation:** n2-standard-16 for image/video processing
- **Feature engineering:** n2-highmem-32 for large in-memory datasets
- **Development environments:** n2-standard-4/8 for coding/debugging

**Spot Advantages:**
- **76% discount** (best among general-purpose)
- Better CPU performance per dollar than N1
- GPU attachment (NVIDIA T4, V100, A100, L4)
- Up to 16 GPUs per instance (A100 80GB)

From [CloudBolt GCP Instance Types](https://www.cloudbolt.io/gcp-cost-optimization/gcp-instance-types/) (accessed 2025-01-31):
- N2 recommended over N1 for new workloads
- 25-35% better price-performance
- Cascade Lake architecture improvements

---

### N2D Machine Series (AMD Cost-Effective)

**Overview:**
- AMD EPYC Rome processors (2nd Gen)
- Lower cost than N2 (~15-20% cheaper)
- Predefined: standard, highmem, highcpu
- Custom: 2-224 vCPUs, 0.5-896 GB memory
- Good spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
n2d-standard-4      4       16 GB    $0.174/hr    $0.042/hr   76%
n2d-standard-8      8       32 GB    $0.349/hr    $0.084/hr   76%
n2d-standard-16     16      64 GB    $0.697/hr    $0.167/hr   76%
n2d-standard-32     32      128 GB   $1.395/hr    $0.335/hr   76%
n2d-standard-64     64      256 GB   $2.789/hr    $0.669/hr   76%
n2d-standard-224    224     896 GB   $9.725/hr    $2.334/hr   76%

n2d-highmem-8       8       64 GB    $0.466/hr    $0.112/hr   76%
n2d-highmem-32      32      256 GB   $1.865/hr    $0.448/hr   76%
n2d-highmem-96      96      768 GB   $5.594/hr    $1.343/hr   76%

n2d-highcpu-16      16      16 GB    $0.531/hr    $0.127/hr   76%
n2d-highcpu-32      32      32 GB    $1.062/hr    $0.255/hr   76%
n2d-highcpu-224     224     224 GB   $7.434/hr    $1.784/hr   76%
```

**ML Use Cases:**
- **Cost-sensitive preprocessing:** Best $/vCPU for CPU workloads
- **Distributed data loading:** Large fleets for parallel processing
- **Budget experimentation:** Lowest-cost general-purpose option
- **High-core workloads:** Up to 224 vCPUs for massive parallelism

**Spot Advantages:**
- **76% discount + lower base price** (cheapest general-purpose)
- AMD EPYC Rome competitive with Intel Cascade Lake
- No GPU attachment (CPU-only)
- Best choice for pure CPU spot workloads

---

### N4 Machine Series (Latest Generation)

**Overview:**
- Intel Xeon Scalable processors (4th Gen Sapphire Rapids)
- Latest general-purpose machine type (2024+)
- Improved performance and efficiency
- Standard configurations: 2-64 vCPUs
- Growing spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
n4-standard-2       2       8 GB     $0.101/hr    $0.024/hr   76%
n4-standard-4       4       16 GB    $0.202/hr    $0.049/hr   76%
n4-standard-8       8       32 GB    $0.404/hr    $0.097/hr   76%
n4-standard-16      16      64 GB    $0.808/hr    $0.194/hr   76%
n4-standard-32      32      128 GB   $1.616/hr    $0.388/hr   76%
n4-standard-64      64      256 GB   $3.232/hr    $0.776/hr   76%
```

**ML Use Cases:**
- **Modern ML pipelines:** Latest CPU architecture for preprocessing
- **Production inference:** Improved efficiency for serving
- **Development:** Fast iteration with latest hardware
- **Future-proofing:** Newest architecture with best roadmap

**Spot Considerations:**
- 76% discount (consistent with N2)
- Limited regional availability (newer machine type)
- Better performance per core than N1/N2
- GPU attachment support (NVIDIA L4, A100)

From [GCP N4 Announcement](https://cloud.google.com/blog/products/compute/introducing-n4-compute-engine-vms) (accessed 2025-01-31):
- Sapphire Rapids architecture improvements
- Better memory bandwidth and cache
- Improved AVX-512 support for ML workloads

---

### E2 Machine Series (Lowest Cost, Limited Spot)

**Overview:**
- Cost-optimized shared-core and standard instances
- Intel Skylake, Broadwell, Haswell CPUs
- Lower performance, lowest cost
- Standard: 2-32 vCPUs, 0.5-128 GB memory
- **Limited spot availability**

**Spot Pricing (where available):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
e2-standard-2       2       8 GB     $0.067/hr    ~$0.027/hr  ~60%
e2-standard-4       4       16 GB    $0.134/hr    ~$0.054/hr  ~60%
e2-standard-8       8       32 GB    $0.268/hr    ~$0.107/hr  ~60%
e2-standard-16      16      64 GB    $0.536/hr    ~$0.214/hr  ~60%
```

**ML Use Cases:**
- **Ultra-low-cost experimentation:** Minimal-cost spot testing
- **Light data preprocessing:** Small-scale ETL jobs
- **Development environments:** Basic coding/debugging
- **Queue workers:** Background job processing

**Spot Considerations:**
- Spot availability is **unreliable** (GCP prioritizes other types)
- No GPU attachment
- Shared-core instances (e2-micro, e2-small) have very limited spot
- Use N2D instead for better spot availability + performance

---

### General-Purpose Selection Matrix

**Choose N1 when:**
- ✓ Need proven spot availability (legacy = high capacity)
- ✓ GPU attachment required (K80, P100, V100, T4)
- ✓ Working in regions with limited N2/N2D
- ✓ Fallback option when newer types preempted

**Choose N2 when:**
- ✓ Best price-performance balance (76% discount)
- ✓ GPU attachment needed (T4, V100, A100, L4)
- ✓ Modern CPU features required
- ✓ Production ML preprocessing pipelines

**Choose N2D when:**
- ✓ Absolute lowest cost for CPU workloads
- ✓ CPU-only spot instances (no GPU needed)
- ✓ High vCPU counts (up to 224)
- ✓ Budget-constrained experimentation

**Choose N4 when:**
- ✓ Latest CPU architecture needed
- ✓ Best single-thread performance
- ✓ Future-proofing infrastructure
- ✓ Available in target region

**Avoid E2 for spot:**
- ✗ Unreliable spot availability
- ✗ Use N2D instead for better value

---

## Section 2: Specialized Machine Types (C2, C2D, C3, H3, M1, M2, M3, M4, A2, A3, G2)

Specialized machines optimize for specific workload characteristics: compute-intensive inference, memory-intensive datasets, or GPU-accelerated training.

### C2 Compute-Optimized (Intel Cascade Lake)

**Overview:**
- High-performance compute-intensive workloads
- Intel Xeon Scalable Cascade Lake (3.9 GHz all-core turbo)
- Highest per-core performance in GCP
- Standard: 4-60 vCPUs, 16-240 GB memory
- Good spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
c2-standard-4       4       16 GB    $0.209/hr    $0.050/hr   76%
c2-standard-8       8       32 GB    $0.418/hr    $0.100/hr   76%
c2-standard-16      16      64 GB    $0.835/hr    $0.200/hr   76%
c2-standard-30      30      120 GB   $1.567/hr    $0.376/hr   76%
c2-standard-60      60      240 GB   $3.134/hr    $0.752/hr   76%
```

**ML Use Cases:**
- **Model inference serving:** High throughput CPU inference
- **Real-time prediction:** Low latency serving endpoints
- **Online learning:** Fast model updates
- **Batch scoring:** High-performance batch predictions

**Spot Advantages:**
- **76% discount** on already high-performance CPUs
- Best single-thread performance for inference
- Good spot availability (mature machine type)
- No GPU attachment (CPU-optimized only)

From [GCP Compute-Optimized Machines](https://docs.cloud.google.com/compute/docs/compute-optimized-machines) (accessed 2025-01-31):
- 3.9 GHz sustained all-core turbo
- Optimized for compute-bound workloads
- 50-100 Gbps network bandwidth (30+ vCPUs)

---

### C2D Compute-Optimized (AMD EPYC Milan)

**Overview:**
- AMD EPYC Milan processors (3rd Gen)
- Lower cost than C2 with competitive performance
- Standard: 2-112 vCPUs, 8-448 GB memory
- High-core configurations available
- Good spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
c2d-standard-2      2       8 GB     $0.093/hr    $0.022/hr   76%
c2d-standard-4      4       16 GB    $0.187/hr    $0.045/hr   76%
c2d-standard-8      8       32 GB    $0.373/hr    $0.090/hr   76%
c2d-standard-16     16      64 GB    $0.747/hr    $0.179/hr   76%
c2d-standard-32     32      128 GB   $1.493/hr    $0.358/hr   76%
c2d-standard-56     56      224 GB   $2.613/hr    $0.627/hr   76%
c2d-standard-112    112     448 GB   $5.226/hr    $1.254/hr   76%
```

**ML Use Cases:**
- **Cost-effective inference:** Lower cost than C2
- **High-core serving:** Up to 112 vCPUs for parallelism
- **Batch processing:** Large-scale CPU-based predictions
- **Distributed serving:** Fleet-based inference

**Spot Advantages:**
- 76% discount + lower base price than C2
- AMD EPYC Milan competitive with Intel
- Higher vCPU counts available (112 vs 60)
- Best $/vCPU for compute-optimized spot

---

### C3 Compute-Optimized (Latest Intel Sapphire Rapids)

**Overview:**
- Intel Xeon Scalable processors (4th Gen Sapphire Rapids)
- Latest compute-optimized machine type (2024+)
- Improved performance and efficiency
- Standard: 4-176 vCPUs, 16-704 GB memory
- Growing spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
c3-standard-4       4       16 GB    $0.210/hr    $0.050/hr   76%
c3-standard-8       8       32 GB    $0.420/hr    $0.101/hr   76%
c3-standard-22      22      88 GB    $1.155/hr    $0.277/hr   76%
c3-standard-44      44      176 GB   $2.310/hr    $0.554/hr   76%
c3-standard-88      88      352 GB   $4.620/hr    $1.109/hr   76%
c3-standard-176     176     704 GB   $9.240/hr    $2.218/hr   76%
```

**ML Use Cases:**
- **Modern inference:** Latest CPU architecture for serving
- **High-scale serving:** Up to 176 vCPUs per instance
- **Production endpoints:** Best CPU performance for latency
- **Future-proof infrastructure:** Newest compute-optimized

**Spot Considerations:**
- 76% discount (consistent with C2/C2D)
- Limited regional availability (newer)
- Best single-thread and multi-thread performance
- Advanced Intel features (AMX, AVX-512)

---

### H3 Compute-Optimized (Intel Sapphire Rapids, Network-Focused)

**Overview:**
- Intel Xeon Scalable Sapphire Rapids (optimized for HPC)
- Ultra-high network bandwidth (200 Gbps)
- Standard: 88 vCPUs, 352 GB memory (single configuration)
- Purpose-built for HPC and distributed computing
- Limited spot availability

**Spot Pricing (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
h3-standard-88      88      352 GB   $4.845/hr    $1.163/hr   76%
```

**ML Use Cases:**
- **Distributed training coordination:** High-bandwidth node-to-node
- **Large-scale inference:** Network-intensive serving
- **Data pipeline orchestration:** Fast data movement
- **HPC workloads:** Scientific computing

**Spot Considerations:**
- 76% discount but high base cost
- Single configuration (88 vCPUs only)
- 200 Gbps network (vs 100 Gbps for C3)
- Limited availability (specialized use case)

---

### M1 Memory-Optimized (Up to 4 TB)

**Overview:**
- Memory-intensive workloads (SAP HANA, in-memory DBs)
- Intel Xeon (Skylake, Broadwell E7)
- Configurations: 40-160 vCPUs, 961 GB - 3,844 GB memory
- Ultra-high memory-to-vCPU ratio (24 GB/vCPU)
- Limited spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
m1-ultramem-40      40      961 GB   $6.611/hr    $1.587/hr   76%
m1-ultramem-80      80      1,922 GB $13.222/hr   $3.173/hr   76%
m1-ultramem-160     160     3,844 GB $26.444/hr   $6.347/hr   76%
m1-megamem-96       96      1,433 GB $10.674/hr   $2.562/hr   76%
```

**ML Use Cases:**
- **Large in-memory datasets:** Load entire dataset in RAM
- **Graph neural networks:** Large graph structures in memory
- **Feature engineering:** Complex transformations on massive data
- **Pre-training data loading:** Cache datasets for fast access

**Spot Considerations:**
- 76% discount but **very high absolute cost**
- Limited spot availability (specialized hardware)
- No GPU attachment
- Use only when memory bottleneck confirmed

---

### M2 Memory-Optimized (Up to 12 TB)

**Overview:**
- Ultra-large memory workloads
- Intel Xeon Cascade Lake
- Configurations: 208-416 vCPUs, 5,888 GB - 11,776 GB memory
- Highest memory capacity in GCP
- Very limited spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
m2-ultramem-208     208     5,888 GB $40.555/hr   $9.733/hr   76%
m2-ultramem-416     416     11,776 GB $81.110/hr  $19.466/hr  76%
m2-megamem-416      416     5,888 GB $40.555/hr   $9.733/hr   76%
```

**ML Use Cases:**
- **Extreme-scale datasets:** Massive in-memory workloads
- **Large language models:** Full model + data in memory (inference)
- **Scientific computing:** Genomics, simulations
- **Rarely justified for ML training** (GPUs more cost-effective)

**Spot Considerations:**
- 76% discount but **extremely high absolute cost**
- Very limited spot availability (rare hardware)
- Spot savings: $19-40k/month (still expensive)
- GPU machines + checkpointing usually better value

---

### M3 Memory-Optimized (Latest Generation)

**Overview:**
- Intel Xeon Scalable (Ice Lake)
- Improved price-performance vs M1/M2
- Configurations: 32-128 vCPUs, 512 GB - 1,952 GB memory
- More accessible than M1/M2
- Growing spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs   Memory   On-Demand    Spot        Savings
m3-ultramem-32      32      512 GB   $4.352/hr    $1.045/hr   76%
m3-ultramem-64      64      1,024 GB $8.704/hr    $2.089/hr   76%
m3-ultramem-128     128     1,952 GB $16.640/hr   $3.994/hr   76%
m3-megamem-64       64      976 GB   $7.808/hr    $1.874/hr   76%
m3-megamem-128      128     1,952 GB $15.616/hr   $3.748/hr   76%
```

**ML Use Cases:**
- **Large in-memory datasets:** More accessible than M1/M2
- **Feature stores:** In-memory feature serving
- **Real-time analytics:** Fast in-memory aggregations
- **Development/testing:** M1/M2 workload prototyping

**Spot Advantages:**
- 76% discount + lower base cost than M1/M2
- Better CPU performance (Ice Lake)
- More reasonable absolute costs
- Better spot availability than M1/M2

---

### M4 Memory-Optimized (Latest AMD)

**Overview:**
- AMD EPYC Genoa processors
- Cost-effective memory-optimized option
- Configurations being released (2024+)
- Lower cost than M3
- Emerging spot availability

**Expected Use Cases:**
- Cost-effective large memory workloads
- Alternative to M3 with AMD pricing
- Growing ecosystem support

**Spot Considerations:**
- Pricing not yet widely available
- Expected 76% spot discount
- Lower base cost than M3
- Limited regional availability (new)

From [GCP Memory-Optimized Machines](https://docs.cloud.google.com/compute/docs/memory-optimized-machines) (accessed 2025-01-31):
- M4 series launching with AMD Genoa
- Improved price-performance vs M3
- Growing availability in 2024-2025

---

### A2 Accelerator-Optimized (NVIDIA A100)

**Overview:**
- Purpose-built for GPU-accelerated ML training
- NVIDIA A100 40GB or 80GB GPUs
- Configurations: a2-highgpu, a2-megagpu, a2-ultragpu
- Pre-configured CPU/GPU/memory ratios
- Good spot availability for A100 40GB

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs  Memory  GPUs        On-Demand    Spot         Savings
a2-highgpu-1g       12     85 GB   1x A100-40  $3.938/hr    $1.183/hr    70%
a2-highgpu-2g       24     170 GB  2x A100-40  $7.875/hr    $2.363/hr    70%
a2-highgpu-4g       48     340 GB  4x A100-40  $15.750/hr   $4.725/hr    70%
a2-highgpu-8g       96     680 GB  8x A100-40  $31.500/hr   $9.450/hr    70%

a2-megagpu-16g      96     1,360GB 16x A100-40 $63.000/hr   $18.900/hr   70%
a2-ultragpu-1g      12     170 GB  1x A100-80  $5.563/hr    $1.669/hr    70%
a2-ultragpu-8g      96     1,360GB 8x A100-80  $44.500/hr   $13.350/hr   70%
```

**ML Use Cases:**
- **LLM training:** 7B-70B parameter models
- **VLM training:** Vision-language models
- **Multi-GPU training:** DDP, FSDP across 2-16 GPUs
- **Large batch training:** High GPU memory for big batches

**Spot Advantages:**
- **70% discount** (slightly lower than CPU-only)
- Good A100-40GB availability
- Limited A100-80GB availability (high demand)
- Best value for mid-scale GPU training

From [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing) (accessed 2025-01-31):
- A100-40GB widely available on spot
- A100-80GB spot limited to select regions
- A2 family optimized for ML workloads

---

### A3 Accelerator-Optimized (NVIDIA H100)

**Overview:**
- Latest GPU-optimized machines
- NVIDIA H100 80GB GPUs (Hopper architecture)
- Ultra-high-bandwidth NVLink/NVSwitch
- Configurations: a3-highgpu-8g (8x H100)
- Very limited spot availability

**Spot Pricing (us-central1, when available):**
```
Machine Type        vCPUs  Memory  GPUs        On-Demand    Spot         Savings
a3-highgpu-8g       208    1,872GB 8x H100-80  ~$60/hr      ~$18/hr      ~70%
```

**ML Use Cases:**
- **Frontier LLM training:** 70B+ parameter models
- **Large-scale pretraining:** Multi-node training
- **Research experiments:** Cutting-edge model architectures
- **Benchmark testing:** Latest GPU performance

**Spot Considerations:**
- ~70% discount when available
- **Very limited spot availability** (high demand)
- Restricted to us-central1, europe-west4
- Spot preemption likelihood high
- Use on-demand for critical training runs

---

### G2 Accelerator-Optimized (NVIDIA L4)

**Overview:**
- NVIDIA L4 GPUs (Ada Lovelace architecture)
- Optimized for inference and lightweight training
- Configurations: g2-standard-4 to g2-standard-96
- 1, 2, 4, or 8 L4 GPUs per instance
- Excellent spot availability

**Spot Pricing Examples (us-central1):**
```
Machine Type        vCPUs  Memory  GPUs    On-Demand    Spot         Savings
g2-standard-4       4      16 GB   1x L4   $1.123/hr    $0.337/hr    70%
g2-standard-8       8      32 GB   1x L4   $1.366/hr    $0.410/hr    70%
g2-standard-12      12     48 GB   1x L4   $1.609/hr    $0.483/hr    70%
g2-standard-16      16     64 GB   1x L4   $1.852/hr    $0.556/hr    70%

g2-standard-24      24     96 GB   2x L4   $2.975/hr    $0.892/hr    70%
g2-standard-48      48     192 GB  4x L4   $5.707/hr    $1.712/hr    70%
g2-standard-96      96     384 GB  8x L4   $11.171/hr   $3.351/hr    70%
```

**ML Use Cases:**
- **Inference serving:** Cost-effective GPU inference
- **Lightweight training:** Small models (< 1B params)
- **Fine-tuning:** LoRA, adapter training
- **Video processing:** GPU-accelerated video pipelines

**Spot Advantages:**
- **70% discount + low base cost**
- **Excellent spot availability** (newer, less contention)
- Best $/TFLOPS for inference workloads
- Good balance of compute and cost

From [GCP L4 GPU Documentation](https://cloud.google.com/compute/docs/gpus/l4-gpus) (accessed 2025-01-31):
- L4 optimized for inference and graphics
- Ada Lovelace architecture efficiency
- Lower power consumption than A100

---

### Specialized Machine Selection Matrix

**Compute-Optimized (C2, C2D, C3, H3):**
- Use for: CPU inference, batch scoring, real-time serving
- Best value: **C2D** (lowest cost, good performance)
- Best performance: **C3** (latest architecture)
- Network-intensive: **H3** (200 Gbps bandwidth)

**Memory-Optimized (M1, M2, M3, M4):**
- Use for: Large in-memory datasets, feature stores
- Most accessible: **M3** (reasonable cost, good availability)
- Extreme scale: **M2** (up to 12 TB, very expensive)
- Cost-effective: **M4** (AMD pricing, emerging)
- **Warning:** GPU machines usually better value for ML

**Accelerator-Optimized (A2, A3, G2):**
- Training: **A2** (A100 40GB/80GB, good spot availability)
- Frontier training: **A3** (H100, limited spot availability)
- Inference: **G2** (L4, excellent spot availability + cost)
- **Default choice: A2 highgpu-1g/2g/4g** for most training

---

## Section 3: Machine Type Selection Guide for ML Pipelines

Practical decision frameworks for choosing machine types across ML pipeline stages, optimizing for cost and performance with spot instances.

### Data Preprocessing Stage

**Workload characteristics:**
- CPU-intensive (tokenization, augmentation, feature extraction)
- I/O bound (reading from Cloud Storage)
- Embarrassingly parallel (process batches independently)
- Fault-tolerant (can restart from last checkpoint)

**Recommended machine types:**

**Budget-conscious (best $/throughput):**
```
n2d-highcpu-32    32 vCPUs, 32 GB   $0.255/hr spot   ~76% savings
n2d-highcpu-64    64 vCPUs, 64 GB   $0.510/hr spot   ~76% savings
```
- Lowest cost per vCPU
- Good spot availability
- Scale horizontally with many instances

**Balanced performance:**
```
n2-highcpu-32     32 vCPUs, 32 GB   $0.285/hr spot   ~76% savings
n2-standard-32    32 vCPUs, 128 GB  $0.373/hr spot   ~76% savings
```
- Better single-thread performance than N2D
- More memory if needed for in-memory caching

**High-performance:**
```
c2-standard-30    30 vCPUs, 120 GB  $0.376/hr spot   ~76% savings
c2-standard-60    60 vCPUs, 240 GB  $0.752/hr spot   ~76% savings
```
- Highest CPU frequency for compute-bound tasks
- Good for complex data transformations

**Selection strategy:**
1. Start with **n2d-highcpu-32** (best value)
2. Profile CPU utilization during preprocessing
3. If CPU-bound: Scale to n2d-highcpu-64 or c2-standard-30
4. If memory-bound: Switch to n2-standard-32/64
5. If I/O-bound: Increase instance count, not size

**Example cost comparison (100 hours preprocessing):**
```
n2d-highcpu-32 spot:  $0.255/hr × 100hr = $25.50
n2-highcpu-32 spot:   $0.285/hr × 100hr = $28.50
c2-standard-30 spot:  $0.376/hr × 100hr = $37.60

Savings vs on-demand:
n2d-highcpu-32: $106.20/hr × 100hr = $10,620 → $25.50 spot (99.76% savings)
```

From [GCP VM Pricing](https://cloud.google.com/compute/vm-instance-pricing) (accessed 2025-01-31):
- N2D provides best $/vCPU for CPU workloads
- Spot pricing consistent across machine families
- No performance penalty for spot vs on-demand

---

### GPU Training Stage

**Workload characteristics:**
- GPU-intensive (forward/backward passes)
- Memory-bound (model parameters, activations)
- Long-running (hours to days)
- Checkpoint-friendly (resume from interruption)

**Recommended machine types by model size:**

**Small models (< 1B parameters):**
```
g2-standard-12    12 vCPUs, 48 GB, 1x L4    $0.483/hr spot  ~70% savings
a2-highgpu-1g     12 vCPUs, 85 GB, 1x A100  $1.183/hr spot  ~70% savings
```
- **G2 (L4)** best for inference-focused training
- **A2 (A100)** better for training-focused workloads
- Single GPU sufficient

**Medium models (1B-13B parameters):**
```
a2-highgpu-1g     12 vCPUs, 85 GB, 1x A100-40  $1.183/hr spot  ~70% savings
a2-highgpu-2g     24 vCPUs, 170 GB, 2x A100-40 $2.363/hr spot  ~70% savings
a2-ultragpu-1g    12 vCPUs, 170 GB, 1x A100-80 $1.669/hr spot  ~70% savings
```
- **A100-40GB** sufficient for most 7B models
- **A100-80GB** for 13B models or large batches
- 1-2 GPUs optimal

**Large models (13B-70B parameters):**
```
a2-highgpu-4g     48 vCPUs, 340 GB, 4x A100-40  $4.725/hr spot   ~70% savings
a2-highgpu-8g     96 vCPUs, 680 GB, 8x A100-40  $9.450/hr spot   ~70% savings
a2-ultragpu-8g    96 vCPUs, 1360GB, 8x A100-80  $13.350/hr spot  ~70% savings
```
- **4x A100-40GB** for 30B models with FSDP
- **8x A100-40GB** for 70B models
- **8x A100-80GB** for 70B+ with large batches

**Frontier models (70B+ parameters):**
```
a3-highgpu-8g     208 vCPUs, 1872GB, 8x H100-80  ~$18/hr spot  ~70% savings
Multi-node A2/A3 configurations
```
- **H100** when available on spot (limited)
- Multi-node A100 more reliable on spot
- Consider on-demand for critical runs

**Selection strategy:**
1. **Default: a2-highgpu-1g** (single A100-40GB, $1.18/hr spot)
2. Profile GPU memory usage during training
3. If OOM: Scale to a2-ultragpu-1g (A100-80GB)
4. If training too slow: Scale to a2-highgpu-2g/4g (multi-GPU)
5. If spot unavailable: Use G2 (L4) as fallback

**Cost comparison (100 GPU-hours training):**
```
g2-standard-12 spot:   $0.483/hr × 100hr = $48.30
a2-highgpu-1g spot:    $1.183/hr × 100hr = $118.30
a2-highgpu-4g spot:    $4.725/hr × 100hr = $472.50
a3-highgpu-8g spot:    $18.00/hr × 100hr = $1,800.00

Savings vs on-demand:
a2-highgpu-1g: $3.938/hr × 100hr = $393.80 → $118.30 spot (70% savings)
```

---

### Inference Serving Stage

**Workload characteristics:**
- Low latency requirements (<100ms)
- Batch or real-time serving
- Variable load (auto-scaling)
- High uptime requirements (avoid spot for critical paths)

**Recommended machine types:**

**CPU inference (lightweight models):**
```
c2-standard-16    16 vCPUs, 64 GB   $0.200/hr spot   ~76% savings
c2d-standard-32   32 vCPUs, 128 GB  $0.358/hr spot   ~76% savings
c3-standard-22    22 vCPUs, 88 GB   $0.277/hr spot   ~76% savings
```
- **C2** best single-thread performance
- **C2D** best cost per vCPU
- **C3** latest architecture, best efficiency

**GPU inference (large models):**
```
g2-standard-12    12 vCPUs, 48 GB, 1x L4    $0.483/hr spot  ~70% savings
g2-standard-24    24 vCPUs, 96 GB, 2x L4    $0.892/hr spot  ~70% savings
a2-highgpu-1g     12 vCPUs, 85 GB, 1x A100  $1.183/hr spot  ~70% savings
```
- **G2 (L4)** best $/inference for most models
- **A2 (A100)** for large models requiring GPU memory

**Selection strategy:**
1. Start with **c2d-standard-16** for CPU inference
2. Profile latency and throughput
3. If GPU needed: Use **g2-standard-12** (L4)
4. Scale horizontally for throughput (many instances)
5. **Use on-demand for production** (spot for canary/staging)

**Warning:** Spot instances for inference serving risk:
- Preemption during high-traffic periods
- 30-second graceful shutdown (may drop requests)
- Better for batch inference than real-time serving

---

### Large Dataset Handling Stage

**Workload characteristics:**
- Large in-memory datasets (>100 GB)
- Feature engineering on full dataset
- Graph neural networks (large graphs)
- Minimal compute, maximum memory

**Recommended machine types:**

**Accessible large memory:**
```
n2-highmem-64     64 vCPUs, 512 GB   $1.493/hr spot   ~76% savings
n2-highmem-128    128 vCPUs, 864 GB  $2.000/hr spot   ~76% savings
m3-ultramem-32    32 vCPUs, 512 GB   $1.045/hr spot   ~76% savings
m3-ultramem-64    64 vCPUs, 1024 GB  $2.089/hr spot   ~76% savings
```
- **N2-highmem** for moderate memory needs (512-864 GB)
- **M3** for dedicated memory-optimized (512-1952 GB)

**Extreme memory:**
```
m3-ultramem-128   128 vCPUs, 1952 GB  $3.994/hr spot   ~76% savings
m1-ultramem-160   160 vCPUs, 3844 GB  $6.347/hr spot   ~76% savings
m2-ultramem-208   208 vCPUs, 5888 GB  $9.733/hr spot   ~76% savings
```
- **M3** most cost-effective (up to 1952 GB)
- **M1** for 4 TB workloads
- **M2** for 12 TB extreme cases

**Selection strategy:**
1. **Question memory requirement first:** Can you stream/batch?
2. If truly memory-bound: Start with **n2-highmem-64**
3. Profile actual memory usage during workload
4. Scale to M3 only if N2-highmem insufficient
5. **Avoid M1/M2 unless absolutely necessary** (very expensive)

**Cost reality check:**
```
n2-highmem-64 spot:    $1.493/hr spot × 24hr = $35.83/day
m3-ultramem-64 spot:   $2.089/hr spot × 24hr = $50.14/day
m1-ultramem-160 spot:  $6.347/hr spot × 24hr = $152.33/day

Monthly costs:
n2-highmem-64:  $1,075/month (reasonable)
m3-ultramem-64: $1,504/month (acceptable)
m1-ultramem-160: $4,570/month (expensive)
```

**Alternative approach (usually better):**
- Use A2 GPU machine with large GPU memory
- Stream data from Cloud Storage
- Checkpoint intermediate results
- **A2-highgpu-1g (A100-40GB): $1.183/hr spot = $28.39/day**
- Much cheaper than M1/M2, faster processing

---

### Complete Decision Matrix

```
Stage                 Workload Type           Recommended Machine           Cost/hr (spot)
────────────────────────────────────────────────────────────────────────────────────────
Data Preprocessing    CPU-intensive           n2d-highcpu-32               $0.255
                     CPU-heavy               c2-standard-30               $0.376
                     Memory-bound            n2-highmem-32                $0.500

GPU Training         Small models (<1B)      g2-standard-12 (L4)          $0.483
                     Medium models (1-13B)   a2-highgpu-1g (A100-40)      $1.183
                     Large models (13-70B)   a2-highgpu-4g (4×A100-40)    $4.725
                     Frontier (70B+)         a3-highgpu-8g (8×H100-80)    ~$18.00

Inference Serving    CPU lightweight         c2d-standard-16              $0.200
                     CPU high-perf           c2-standard-16               $0.200
                     GPU inference           g2-standard-12 (L4)          $0.483
                     GPU large models        a2-highgpu-1g (A100)         $1.183

Large Datasets       Moderate memory         n2-highmem-64 (512GB)        $1.493
                     Large memory            m3-ultramem-64 (1TB)         $2.089
                     Extreme memory          m1-ultramem-160 (4TB)        $6.347
```

---

### Spot Availability Considerations

**High availability (recommended for spot):**
- N1, N2, N2D general-purpose
- C2, C2D compute-optimized
- A2-highgpu-1g/2g (A100-40GB)
- G2 (L4 GPUs)

**Medium availability (use with fallback):**
- N4 (newer, limited regions)
- C3 (newer, limited regions)
- M3 memory-optimized
- A2-highgpu-4g/8g (multi-GPU A100)

**Low availability (prefer on-demand):**
- A2-ultragpu (A100-80GB)
- A3-highgpu (H100)
- M1, M2 (specialized hardware)
- E2 (limited spot support)

**Multi-region strategy for spot reliability:**
```python
# Preferred regions for spot availability (in order)
regions_priority = [
    "us-central1",      # Iowa - highest availability
    "us-west1",         # Oregon - good availability
    "europe-west4",     # Netherlands - European hub
    "asia-southeast1",  # Singapore - APAC hub
]

# Fallback machine types if spot unavailable
fallback_chain = [
    "a2-highgpu-2g",    # Primary choice
    "a2-highgpu-1g",    # Smaller if unavailable
    "g2-standard-24",   # L4 GPUs as fallback
    "on-demand",        # Last resort
]
```

---

### Cost Optimization Strategies

**Horizontal scaling (many small instances) vs Vertical scaling (few large instances):**

**Example: 128 vCPU preprocessing workload**

**Horizontal (4× n2d-highcpu-32):**
```
4 × n2d-highcpu-32 @ $0.255/hr = $1.020/hr spot
- Advantages: Better spot availability, fault tolerance
- Disadvantages: More management overhead
```

**Vertical (2× n2d-highcpu-64):**
```
2 × n2d-highcpu-64 @ $0.510/hr = $1.020/hr spot
- Advantages: Simpler management, less network overhead
- Disadvantages: Higher preemption impact
```

**Recommendation:** Horizontal scaling for spot workloads
- If 1/4 instances preempted: 75% capacity remains
- If 1/2 instances preempted: 50% capacity remains
- Better overall reliability

**GPU training: Inverse relationship**
- Prefer fewer large instances (a2-highgpu-4g vs 4× a2-highgpu-1g)
- Reason: Multi-GPU communication within instance faster
- Reason: Checkpointing 1 large instance easier than 4 small

---

### Machine Type Selection Checklist

**Before choosing a machine type, answer:**

1. **What's the bottleneck?**
   - CPU-bound → Compute-optimized (C2, C2D, C3)
   - Memory-bound → Memory-optimized (M3) or highmem variants
   - GPU-bound → Accelerator-optimized (A2, G2)
   - I/O-bound → Scale horizontally, not vertically

2. **What's the workload duration?**
   - <1 hour → Any spot instance fine
   - 1-24 hours → Use checkpoint every 30min
   - >24 hours → Consider on-demand for stability

3. **What's the spot availability?**
   - High (N2, A2-1g/2g, G2) → Use spot confidently
   - Medium (N4, C3, A2-4g) → Use with fallback plan
   - Low (A3, M1/M2) → Prefer on-demand

4. **What's the cost tolerance?**
   - Budget: N2D general-purpose
   - Balanced: N2 general-purpose, A2 single GPU
   - Performance: C2/C3 compute, A2 multi-GPU, A3 H100

5. **Can the workload handle preemption?**
   - Yes → Use spot (60-91% savings)
   - No → Use on-demand (or hybrid strategy)

**Decision tree:**
```
Need GPU?
├─ No → CPU workload
│  ├─ CPU-intensive? → c2d-standard-32 (compute-optimized)
│  ├─ Balanced? → n2-standard-32 (general-purpose)
│  └─ Memory-intensive? → n2-highmem-64 or m3-ultramem-64
│
└─ Yes → GPU workload
   ├─ Model size <1B? → g2-standard-12 (L4)
   ├─ Model size 1-13B? → a2-highgpu-1g (A100-40)
   ├─ Model size 13-70B? → a2-highgpu-4g (4×A100-40)
   └─ Model size >70B? → a3-highgpu-8g (8×H100-80, limited spot)
```

---

## Sources

**GCP Official Documentation:**
- [VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) - Complete pricing tables (accessed 2025-01-31)
- [Machine Types Documentation](https://cloud.google.com/compute/docs/machine-types) - Machine family specifications (accessed 2025-01-31)
- [Compute-Optimized Machines](https://docs.cloud.google.com/compute/docs/compute-optimized-machines) - C2, C2D, C3 details (accessed 2025-01-31)
- [Memory-Optimized Machines](https://docs.cloud.google.com/compute/docs/memory-optimized-machines) - M1, M2, M3, M4 specifications (accessed 2025-01-31)
- [GPU Pricing](https://cloud.google.com/compute/gpus-pricing) - Accelerator-optimized pricing (accessed 2025-01-31)
- [Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) - Current spot pricing (accessed 2025-01-31)

**Third-Party Analysis:**
- [CloudBolt GCP Instance Types](https://www.cloudbolt.io/gcp-cost-optimization/gcp-instance-types/) - Machine type comparison (accessed 2025-01-31)
- [CloudKeeper GCP Instance Types Guide](https://www.cloudkeeper.com/insights/blog/gcp-instance-types-explained-making-right-choice-your-workloads) - Selection strategies (accessed 2025-01-31)
- [CloudZero GCP Instance Types](https://www.cloudzero.com/blog/gcp-instance-types/) - Use case recommendations (accessed 2025-01-31)

**Pricing Tools:**
- [GCloud Compute Machine Type Comparison](https://gcloud-compute.com) - Interactive machine type picker (accessed 2025-01-31)
- [WintelGuy GCP VM Price Compare](https://wintelguy.com/gcp-vm-price-compare.pl) - Price comparison tool (accessed 2025-01-31)

---

## Related Documentation

- [38-gcp-spot-fundamentals.md](38-gcp-spot-fundamentals.md) - Spot instance architecture and mechanics
- [39-gcp-gpu-spot-pricing.md](39-gcp-gpu-spot-pricing.md) - GPU spot pricing details
- [42-gcp-spot-availability.md](42-gcp-spot-availability.md) - Regional availability analysis
- [43-gcp-spot-checkpoint-strategies.md](43-gcp-spot-checkpoint-strategies.md) - Fault-tolerant training patterns
