# GCP GPU Spot Pricing - Complete Regional Analysis

## Overview

Complete GPU spot pricing guide for cost-optimized ML training on Google Cloud Platform. Covers A100 (40GB/80GB), H100 (80GB), L4, T4, and V100 GPUs with regional pricing comparisons, spot discount analysis, and ROI calculations for production workloads.

**Key Savings:** 60-91% cost reduction with spot instances vs on-demand pricing.

---

## NVIDIA A100 Spot Pricing

### A100 40GB Spot Pricing by Region

The NVIDIA A100 40GB is the workhorse GPU for LLM and VLM training, offering 40GB HBM2e memory and 312 TFLOPS FP16 performance.

**Current Spot Pricing (January 2025):**

From [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):

| Region | Location | Spot Price/hour | On-Demand Price/hour | Spot Discount | Monthly Spot Cost |
|--------|----------|-----------------|----------------------|---------------|-------------------|
| us-central1 | Iowa | $1.1472 | $2.933908 | 61% | ~$838 |
| us-west1 | Oregon | $1.1472 | $2.933908 | 61% | ~$838 |
| us-east1 | South Carolina | $1.1472 | $2.933908 | 61% | ~$838 |
| us-west4 | Las Vegas | $1.1472 | $2.933908 | 61% | ~$838 |
| europe-west4 | Netherlands | $1.2619 | $3.227299 | 61% | ~$922 |
| europe-west1 | Belgium | $1.2619 | $3.227299 | 61% | ~$922 |
| asia-southeast1 | Singapore | $1.3766 | $3.520690 | 61% | ~$1,006 |
| asia-northeast1 | Tokyo | $1.3766 | $3.520690 | 61% | ~$1,006 |

**Notes:**
- Prices shown are per GPU per hour
- A2 machine types (a2-highgpu-1g, a2-highgpu-2g, etc.) required for A100 GPUs
- Monthly costs assume 730 hours (24/7 operation)
- Spot instances may be preempted with 30-second notice

### A100 80GB Spot Pricing by Region

The A100 80GB doubles memory capacity for larger models and batch sizes, essential for LLMs >13B parameters.

**Current Spot Pricing (January 2025):**

From [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):

| Region | Location | Spot Price/hour | On-Demand Price/hour | Spot Discount | Monthly Spot Cost |
|--------|----------|-----------------|----------------------|---------------|-------------------|
| us-central1 | Iowa | $1.5712 | $4.020816 | 61% | ~$1,147 |
| us-west1 | Oregon | $1.5712 | $4.020816 | 61% | ~$1,147 |
| us-east1 | South Carolina | $1.5712 | $4.020816 | 61% | ~$1,147 |
| us-west4 | Las Vegas | $1.5712 | $4.020816 | 61% | ~$1,147 |
| europe-west4 | Netherlands | $1.7283 | $4.422897 | 61% | ~$1,262 |
| europe-west1 | Belgium | $1.7283 | $4.422897 | 61% | ~$1,262 |
| asia-southeast1 | Singapore | $1.8854 | $4.824979 | 61% | ~$1,376 |
| asia-northeast1 | Tokyo | $1.8854 | $4.824979 | 61% | ~$1,376 |

**Cost Analysis:**
- A100 80GB costs ~37% more than 40GB variant
- Cost per GB memory: 40GB = $0.029/hour, 80GB = $0.020/hour (better value)
- Recommended for models requiring >30GB activation memory

### A2 Machine Types with A100 GPUs

A2 machine family is purpose-built for A100 GPU workloads with optimized CPU-GPU ratios:

**A2 Machine Type Configurations:**

| Machine Type | GPUs | vCPUs | Memory (GB) | GPU Memory | Spot Price/hour (us-central1) |
|--------------|------|-------|-------------|------------|--------------------------------|
| a2-highgpu-1g | 1x A100 40GB | 12 | 85 | 40 GB | ~$1.35 |
| a2-highgpu-2g | 2x A100 40GB | 24 | 170 | 80 GB | ~$2.70 |
| a2-highgpu-4g | 4x A100 40GB | 48 | 340 | 160 GB | ~$5.40 |
| a2-highgpu-8g | 8x A100 40GB | 96 | 680 | 320 GB | ~$10.80 |
| a2-megagpu-16g | 16x A100 40GB | 96 | 1360 | 640 GB | ~$21.60 |
| a2-ultragpu-1g | 1x A100 80GB | 12 | 170 | 80 GB | ~$1.80 |
| a2-ultragpu-2g | 2x A100 80GB | 24 | 340 | 160 GB | ~$3.60 |
| a2-ultragpu-4g | 4x A100 80GB | 48 | 680 | 320 GB | ~$7.20 |
| a2-ultragpu-8g | 8x A100 80GB | 96 | 1360 | 640 GB | ~$14.40 |

**Machine Type Selection Guide:**
- **a2-highgpu-Xg:** Standard A100 40GB configurations, best for most training
- **a2-megagpu-16g:** Dense GPU packing (16 GPUs), maximum throughput
- **a2-ultragpu-Xg:** A100 80GB configurations, large models (>30B params)
- **CPU-GPU ratio:** 12 vCPUs per GPU, balanced for data loading
- **Memory ratio:** ~85GB RAM per GPU, sufficient for most preprocessing

### Regional Availability Heatmap

**A100 GPU Availability by Region (January 2025):**

```
US Regions:
us-central1 (Iowa)            ███████████ High availability (40GB + 80GB)
us-west1 (Oregon)             ██████████  High availability (40GB + 80GB)
us-east1 (South Carolina)     ██████████  High availability (40GB + 80GB)
us-west4 (Las Vegas)          ████████    Good availability (40GB + 80GB)
us-east4 (N. Virginia)        ██████      Moderate (40GB primarily)

Europe Regions:
europe-west4 (Netherlands)    ███████████ High availability (40GB + 80GB)
europe-west1 (Belgium)        ██████████  High availability (40GB + 80GB)
europe-west2 (London)         ████████    Good availability (40GB)
europe-north1 (Finland)       ██████      Moderate (40GB)

Asia Pacific:
asia-southeast1 (Singapore)   ██████████  High availability (40GB + 80GB)
asia-northeast1 (Tokyo)       ██████████  High availability (40GB + 80GB)
asia-south1 (Mumbai)          ██████      Moderate (40GB)
australia-southeast1 (Sydney) █████       Limited (40GB)
```

**Best Regions for A100 Spot:**
1. **us-central1 (Iowa):** Lowest US pricing, highest availability
2. **europe-west4 (Netherlands):** Best European availability
3. **asia-southeast1 (Singapore):** Strong APAC hub

### Cost Savings Analysis

**Monthly Training Cost Comparison (8x A100 80GB):**

| Scenario | Configuration | On-Demand Cost | Spot Cost | Monthly Savings |
|----------|---------------|----------------|-----------|-----------------|
| LLM Fine-tuning | 8x A100 80GB | $23,512 | $10,570 | $12,942 (55%) |
| VLM Training | 8x A100 80GB | $23,512 | $10,570 | $12,942 (55%) |
| Full Pre-training | 8x A100 80GB 24/7 | $23,512 | $10,570 | $12,942 (55%) |
| Experimentation | 4x A100 40GB (200h/mo) | $2,347 | $1,055 | $1,292 (55%) |

**ROI Calculations:**
- **Break-even point:** Spot instances profitable after ~20 hours of use
- **Preemption overhead:** Average 2-5% training time lost to restarts
- **Net savings:** 50-60% after accounting for preemption overhead
- **Checkpoint cost:** GCS storage ~$0.02/GB/month for checkpoints

From [DataCrunch GPU Pricing Comparison](https://datacrunch.io/blog/cloud-gpu-pricing-comparison) (accessed 2025-01-31):
> GCP A100 spot instances offer consistent 61% savings across all regions, making them the most predictable for budget planning among hyperscalers.

---

## NVIDIA H100 & L4 Spot Pricing

### H100 80GB Spot Pricing

The NVIDIA H100 delivers 3.5x faster training than A100, with 80GB HBM3 memory and Transformer Engine for FP8 acceleration.

**Current H100 Spot Pricing (January 2025):**

From [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):

| Region | Location | Machine Family | Spot Price/hour | On-Demand Price/hour | Spot Discount |
|--------|----------|----------------|-----------------|----------------------|---------------|
| us-central1 | Iowa | A3-HIGH | $2.253 | $5.769231 | 61% |
| us-central1 | Iowa | A3-MEGA | $2.3791 | $6.087563 | 61% |
| europe-west4 | Netherlands | A3-HIGH | $2.4783 | $6.346154 | 61% |
| europe-west4 | Netherlands | A3-MEGA | $2.6170 | $6.696320 | 61% |

**H100 Availability Notes:**
- **Extremely limited availability** - only us-central1 and europe-west4 confirmed
- A3 machine family required (A3-HIGH or A3-MEGA configurations)
- Spot availability varies significantly by time of day
- May require quota increase request to access H100 spot instances

**A3 Machine Types with H100:**

| Machine Type | GPUs | vCPUs | Memory (GB) | GPU Memory | Network | Spot Price/hour (us-central1) |
|--------------|------|-------|-------------|------------|---------|--------------------------------|
| a3-highgpu-1g | 1x H100 80GB | 26 | 208 | 80 GB | 200 Gbps | ~$2.50 |
| a3-highgpu-2g | 2x H100 80GB | 52 | 416 | 160 GB | 200 Gbps | ~$5.00 |
| a3-highgpu-4g | 4x H100 80GB | 104 | 832 | 320 GB | 200 Gbps | ~$10.00 |
| a3-highgpu-8g | 8x H100 80GB | 208 | 1664 | 640 GB | 200 Gbps | ~$20.00 |
| a3-megagpu-8g | 8x H100 80GB | 208 | 1872 | 640 GB | 3200 Gbps | ~$21.00 |

**A3-MEGA vs A3-HIGH:**
- **A3-MEGA:** NVLink + NVSwitch (3.2 Tbps GPU interconnect), for multi-node training
- **A3-HIGH:** PCIe Gen5 interconnect (200 Gbps), for single-node training
- **Price difference:** ~5-10% premium for A3-MEGA
- **Use A3-MEGA when:** Training models >100B params across multiple nodes

### NVIDIA L4 GPU Spot Pricing

The L4 GPU is optimized for inference and light training workloads, with 24GB GDDR6 memory and excellent price-performance for smaller models.

**Current L4 Spot Pricing (January 2025):**

From [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):

| Region | Location | Spot Price/hour | On-Demand Price/hour | Spot Discount | Monthly Spot Cost |
|--------|----------|-----------------|----------------------|---------------|-------------------|
| us-central1 | Iowa | $0.2688 | $0.688 | 61% | ~$196 |
| us-west1 | Oregon | $0.2688 | $0.688 | 61% | ~$196 |
| us-east1 | South Carolina | $0.2688 | $0.688 | 61% | ~$196 |
| europe-west4 | Netherlands | $0.2957 | $0.757 | 61% | ~$216 |
| europe-west1 | Belgium | $0.2957 | $0.757 | 61% | ~$216 |
| asia-southeast1 | Singapore | $0.3226 | $0.826 | 61% | ~$235 |
| asia-northeast1 | Tokyo | $0.3226 | $0.826 | 61% | ~$235 |

**L4 GPU Characteristics:**
- **24GB GDDR6 memory:** Sufficient for models up to 13B params (with quantization)
- **120 TFLOPS FP16:** ~2.5x faster than T4 for training
- **Inference optimized:** INT8/FP8 Tensor Cores for deployment
- **Power efficient:** 72W TDP, lowest cost per inference

**G2 Machine Types with L4:**

| Machine Type | GPUs | vCPUs | Memory (GB) | GPU Memory | Spot Price/hour (us-central1) |
|--------------|------|-------|-------------|------------|--------------------------------|
| g2-standard-4 | 1x L4 24GB | 4 | 16 | 24 GB | ~$0.40 |
| g2-standard-8 | 1x L4 24GB | 8 | 32 | 24 GB | ~$0.50 |
| g2-standard-12 | 1x L4 24GB | 12 | 48 | 24 GB | ~$0.60 |
| g2-standard-16 | 1x L4 24GB | 16 | 64 | 24 GB | ~$0.70 |
| g2-standard-24 | 2x L4 24GB | 24 | 96 | 48 GB | ~$0.90 |
| g2-standard-32 | 1x L4 24GB | 32 | 128 | 24 GB | ~$0.80 |
| g2-standard-48 | 4x L4 24GB | 48 | 192 | 96 GB | ~$1.60 |
| g2-standard-96 | 8x L4 24GB | 96 | 384 | 192 GB | ~$3.00 |

**L4 Use Cases:**
- **Inference deployment:** Production LLM/VLM serving (up to 13B params)
- **Fine-tuning small models:** LoRA/QLoRA for 7B-13B models
- **Experimentation:** Rapid prototyping at low cost
- **Multi-model serving:** 8x L4 can serve multiple models simultaneously

### Cost per TFLOPS Comparison

**GPU Performance vs Cost Analysis (Spot Pricing):**

| GPU Model | Memory | TFLOPS (FP16) | Spot Price/hour | Cost per TFLOPS | Cost per GB Memory |
|-----------|--------|---------------|-----------------|-----------------|---------------------|
| H100 80GB | 80 GB | 989 | $2.25 | $0.0023 | $0.028 |
| A100 80GB | 80 GB | 312 | $1.57 | $0.0050 | $0.020 |
| A100 40GB | 40 GB | 312 | $1.15 | $0.0037 | $0.029 |
| L4 24GB | 24 GB | 120 | $0.27 | $0.0023 | $0.011 |
| T4 16GB | 16 GB | 65 | $0.10 | $0.0015 | $0.006 |

**Best Value Analysis:**
- **Raw performance:** H100 offers best TFLOPS per dollar (tied with L4)
- **Memory capacity:** A100 80GB best value for memory-intensive workloads
- **Budget-conscious:** T4 offers lowest absolute cost per TFLOPS
- **Inference:** L4 best balance of performance and cost for serving

From [GetDeploying GPU Price Comparison](https://getdeploying.com/reference/cloud-gpu) (accessed 2025-01-31):
> L4 GPUs have become the go-to choice for cost-effective inference, offering 120 TFLOPS at $0.27/hour spot pricing - competitive with A100 on TFLOPS-per-dollar basis.

### Regional Availability for H100/L4

**H100 Availability (Very Limited):**
```
us-central1 (Iowa)            ████████ Limited availability (A3-HIGH, A3-MEGA)
europe-west4 (Netherlands)    ██████   Very limited (A3-HIGH, A3-MEGA)
Other regions                 █        Extremely rare or unavailable
```

**L4 Availability (Widespread):**
```
US Regions:
us-central1 (Iowa)            ███████████ Excellent availability
us-west1 (Oregon)             ███████████ Excellent availability
us-east1 (South Carolina)     ██████████  High availability
us-west4 (Las Vegas)          ██████████  High availability

Europe Regions:
europe-west4 (Netherlands)    ███████████ Excellent availability
europe-west1 (Belgium)        ██████████  High availability
europe-west2 (London)         █████████   Good availability

Asia Pacific:
asia-southeast1 (Singapore)   ██████████  High availability
asia-northeast1 (Tokyo)       ██████████  High availability
```

**Recommendation:** L4 GPUs are widely available for spot instances across all major regions, making them reliable for production inference workloads.

---

## Legacy GPUs & Complete Cost Analysis

### NVIDIA T4 GPU Spot Pricing

The T4 GPU remains the most cost-effective option for inference and light training, with 16GB GDDR6 memory and 65 TFLOPS FP16 performance.

**Current T4 Spot Pricing (January 2025):**

From [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):

| Region | Location | Spot Price/hour | On-Demand Price/hour | Spot Discount | Monthly Spot Cost |
|--------|----------|-----------------|----------------------|---------------|-------------------|
| us-central1 | Iowa | $0.0960 | $0.35 | 73% | ~$70 |
| us-west1 | Oregon | $0.0960 | $0.35 | 73% | ~$70 |
| us-east1 | South Carolina | $0.0960 | $0.35 | 73% | ~$70 |
| europe-west4 | Netherlands | $0.1056 | $0.385 | 73% | ~$77 |
| europe-west1 | Belgium | $0.1056 | $0.385 | 73% | ~$77 |
| asia-southeast1 | Singapore | $0.1152 | $0.42 | 73% | ~$84 |
| asia-northeast1 | Tokyo | $0.1152 | $0.42 | 73% | ~$84 |

**T4 GPU Characteristics:**
- **16GB GDDR6 memory:** Adequate for models up to 7B params with quantization
- **65 TFLOPS FP16:** Entry-level training performance
- **INT8 Tensor Cores:** Optimized for quantized inference
- **Lowest cost:** $0.096/hour spot = $70/month for 24/7 operation

**N1 Machine Types with T4:**

| Machine Type | GPUs | vCPUs | Memory (GB) | GPU Memory | Spot Price/hour (us-central1) |
|--------------|------|-------|-------------|------------|--------------------------------|
| n1-standard-4 + 1x T4 | 1x T4 16GB | 4 | 15 | 16 GB | ~$0.15 |
| n1-standard-8 + 1x T4 | 1x T4 16GB | 8 | 30 | 16 GB | ~$0.20 |
| n1-standard-16 + 2x T4 | 2x T4 16GB | 16 | 60 | 32 GB | ~$0.35 |
| n1-standard-32 + 4x T4 | 4x T4 16GB | 32 | 120 | 64 GB | ~$0.65 |

**T4 Use Cases:**
- **Budget inference:** Serving models up to 7B params (GPTQ/AWQ quantized)
- **Experimentation:** Lowest-cost GPU for prototyping
- **Fine-tuning tiny models:** LoRA on models <3B params
- **Batch inference:** Cost-effective batch processing workloads

### NVIDIA V100 GPU Spot Pricing

The V100 GPU is a previous-generation workhorse, still available for training but being phased out in favor of A100.

**Current V100 Spot Pricing (January 2025):**

From [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing) (accessed 2025-01-31):

| Region | Location | Memory | Spot Price/hour | On-Demand Price/hour | Spot Discount |
|--------|----------|--------|-----------------|----------------------|---------------|
| us-central1 | Iowa | 16 GB | $0.6720 | $2.48 | 73% |
| us-west1 | Oregon | 16 GB | $0.6720 | $2.48 | 73% |
| europe-west4 | Netherlands | 16 GB | $0.7392 | $2.728 | 73% |
| asia-southeast1 | Singapore | 16 GB | $0.8064 | $2.976 | 73% |

**V100 Characteristics:**
- **16GB HBM2 memory:** Limited compared to A100's 40GB/80GB
- **125 TFLOPS FP16:** ~2.5x slower than A100 for training
- **Legacy status:** Being phased out, limited spot availability
- **Cost position:** Similar to A100 40GB pricing, but lower performance

**Recommendation:** Avoid V100 for new projects - A100 40GB offers better performance at comparable spot prices. V100 only viable if A100 quota exhausted.

### P100 and P4 GPUs (Legacy, Limited Spot)

**P100 (Pascal generation):**
- 16GB HBM2 memory
- Very limited spot availability (mostly unavailable)
- Not recommended for new ML workloads

**P4 (Pascal generation):**
- 8GB GDDR5 memory
- Primarily for inference
- Being phased out entirely

**Note:** GCP has effectively sunset P100/P4 for spot instances. If these GPUs appear available, they're likely remnant capacity being cleared. Use T4, L4, or A100 instead.

### GPU Memory vs Cost Tradeoff

**Memory Capacity vs Hourly Cost (Spot Pricing):**

| GPU Model | Memory | Spot Price/hour | Cost per GB Memory | Use Case |
|-----------|--------|-----------------|---------------------|----------|
| H100 80GB | 80 GB | $2.25 | $0.028 | Large-scale training (>70B params) |
| A100 80GB | 80 GB | $1.57 | $0.020 | Training 30B-70B param models |
| A100 40GB | 40 GB | $1.15 | $0.029 | Training 7B-30B param models |
| L4 24GB | 24 GB | $0.27 | $0.011 | Inference + fine-tuning <13B |
| T4 16GB | 16 GB | $0.10 | $0.006 | Budget inference <7B params |

**Memory Planning Guidelines:**
- **Training rule of thumb:** Model size × 1.5-2× for activations/gradients
- **13B params × 2 bytes (FP16) = 26GB** → Requires A100 40GB minimum
- **70B params × 2 bytes = 140GB** → Requires 2x A100 80GB or 2x H100
- **Inference with KV cache:** Model size × 1.2-1.5× for context lengths up to 8K tokens

### Training Time vs Cost Optimization

**Example: Fine-tuning 13B Parameter Model**

| Configuration | Time to Complete | Total Spot Cost | Cost per Epoch |
|---------------|------------------|-----------------|----------------|
| 1x H100 80GB | 8 hours | $18.00 | $6.00 |
| 1x A100 80GB | 12 hours | $18.84 | $6.28 |
| 1x A100 40GB | 15 hours | $17.21 | $5.74 |
| 2x L4 24GB | 28 hours | $15.12 | $5.04 |
| 4x T4 16GB | 45 hours | $17.28 | $5.76 |

**Analysis:**
- **Fastest:** H100 completes in 8 hours but at highest absolute cost
- **Best value:** 2x L4 GPUs offer lowest total cost ($15.12)
- **Single-GPU best:** A100 40GB balances speed and cost for single-GPU training
- **Budget option:** 4x T4 viable if time isn't critical (nearly 3× slower than A100)

**Cost Optimization Strategy:**
- Use H100 when time-to-market critical (research deadlines, production launches)
- Use A100 80GB for standard production training runs
- Use L4 for experimentation and cost-sensitive projects
- Avoid T4 for training unless budget extremely constrained

### Complete GPU Pricing Table (All Regions)

**Comprehensive Spot Pricing Matrix (January 2025):**

| GPU | Memory | us-central1 | us-west1 | us-east1 | europe-west4 | asia-southeast1 | Discount |
|-----|--------|-------------|----------|----------|--------------|-----------------|----------|
| H200 | 141 GB | $3.7247 | N/A | N/A | N/A | N/A | 61% |
| H100 (A3-HIGH) | 80 GB | $2.2530 | N/A | N/A | $2.4783 | N/A | 61% |
| H100 (A3-MEGA) | 80 GB | $2.3791 | N/A | N/A | $2.6170 | N/A | 61% |
| A100 80GB | 80 GB | $1.5712 | $1.5712 | $1.5712 | $1.7283 | $1.8854 | 61% |
| A100 40GB | 40 GB | $1.1472 | $1.1472 | $1.1472 | $1.2619 | $1.3766 | 61% |
| L4 | 24 GB | $0.2688 | $0.2688 | $0.2688 | $0.2957 | $0.3226 | 61% |
| T4 | 16 GB | $0.0960 | $0.0960 | $0.0960 | $0.1056 | $0.1152 | 73% |
| V100 | 16 GB | $0.6720 | $0.6720 | N/A | $0.7392 | $0.8064 | 73% |

**Key Observations:**
- **Consistent discounts:** GCP offers stable 61% discount on modern GPUs (A100, H100, L4)
- **Legacy discount:** T4 and V100 get 73% discount (higher than newer GPUs)
- **Regional variance:** 10-15% price increase from US → Europe → Asia regions
- **H100 availability:** Extremely limited, only us-central1 and europe-west4

### ROI Calculations for Spot Training

**Case Study: Production LLM Training on Spot Instances**

**Scenario:** Fine-tune Llama-2-70B on custom dataset (3 epochs, 100K examples)

**Configuration: 8x A100 80GB (a2-ultragpu-8g)**

**On-Demand Cost:**
- Training time: 48 hours
- Cost: 48 hours × 8 GPUs × $4.02/GPU/hour = $1,543
- Checkpoint storage: ~500GB × $0.02/GB/month = $10
- **Total on-demand: $1,553**

**Spot Cost (with preemption):**
- Base training time: 48 hours
- Preemption overhead: +5% (2.4 hours for restarts)
- Total time: 50.4 hours
- Cost: 50.4 hours × 8 GPUs × $1.57/GPU/hour = $633
- Checkpoint storage: $10
- **Total spot: $643**

**Savings:**
- Absolute savings: $1,553 - $643 = $910
- Percentage savings: 58.6% (accounting for preemption overhead)
- Break-even point: 20 hours of training

**Annual Training Budget Projection:**
- 52 training runs per year (weekly iterations)
- On-demand annual cost: $80,756
- Spot annual cost: $33,436
- **Annual savings: $47,320**

**Risk Analysis:**
- Preemption probability: ~2-5% per hour in high-availability regions
- Average restarts per run: 1-2 preemptions (48-hour job)
- Checkpoint overhead: ~5-10 minutes per checkpoint
- Total overhead: 5-8% additional training time

From [Economize Cloud GCP GPU Pricing Analysis](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/) (accessed 2025-01-31):
> Organizations can achieve 50-60% net cost savings using spot instances for LLM training, after accounting for preemption overhead and checkpoint costs. For multi-day training runs, the savings compound significantly.

---

## Sources

**Official GCP Documentation:**
- [GCP Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) - Official spot pricing page (accessed 2025-01-31)
- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing) - On-demand GPU pricing (accessed 2025-01-31)
- [GCP VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) - Machine type pricing (accessed 2025-01-31)
- [GCP GPU Machine Types](https://docs.cloud.google.com/compute/docs/gpus) - GPU configurations and availability (accessed 2025-01-31)

**Pricing Comparisons:**
- [DataCrunch Cloud GPU Pricing Comparison](https://datacrunch.io/blog/cloud-gpu-pricing-comparison) - Multi-cloud GPU pricing analysis (accessed 2025-01-31)
- [GetDeploying GPU Price Comparison](https://getdeploying.com/reference/cloud-gpu) - 2025 GPU pricing database (accessed 2025-01-31)
- [Economize Cloud GCP GPU Pricing Chart](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/) - GCP GPU cost optimization guide (accessed 2025-01-31)
- [Cast AI GPU Price Report 2025](https://cast.ai/reports/gpu-price-2025/) - A100/H100 availability and pricing trends (accessed 2025-01-31)

**Additional References:**
- [Thunder Compute vs GCP GPU Comparison](https://www.thundercompute.com/blog/thunder-compute-vs-gcp-gpu-cloud-comparison) - Alternative provider comparison (accessed 2025-01-31)
- [CloudBolt GCP Spot VMs Guide](https://www.cloudbolt.io/gcp-cost-optimization/google-cloud-spot-vms/) - Spot VM optimization strategies (accessed 2025-01-31)

---

**Related Documentation:**
- See [38-gcp-spot-fundamentals.md](38-gcp-spot-fundamentals.md) for spot instance mechanics and limitations
- See [40-gcp-tpu-spot-pricing.md](40-gcp-tpu-spot-pricing.md) for TPU spot pricing comparison
- See [43-gcp-spot-checkpoint-strategies.md](43-gcp-spot-checkpoint-strategies.md) for fault-tolerant training patterns
