# Vertex AI GPU Quota Metrics - Complete Reference

**Date**: 2025-11-16
**Source**: Compiled from GCP Documentation + Code Analysis
**Purpose**: Master reference for ALL Vertex AI Custom Training GPU quota metrics

---

## The Two Quota Systems (CRITICAL!)

### ❌ COMPUTE ENGINE Quotas (WRONG for Vertex AI!)
```
Metric Format: NVIDIA_*_GPUS, PREEMPTIBLE_NVIDIA_*_GPUS
Command: gcloud compute regions describe us-central1
Service: compute.googleapis.com
```

### ✅ VERTEX AI Custom Training Quotas (CORRECT!)
```
Metric Format: custom_model_training_[preemptible_]nvidia_*_gpus
Service: aiplatform.googleapis.com
Command: gcloud alpha quotas list --service=aiplatform.googleapis.com
```

**YOU MUST USE VERTEX AI QUOTAS FOR CUSTOM TRAINING!**

---

## Complete GPU Quota Metrics List

### Format Pattern
```
aiplatform.googleapis.com/custom_model_training_nvidia_{gpu}_gpus              # Regular (on-demand)
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_{gpu}_gpus  # Spot/Preemptible
```

---

## All Supported GPUs

### 1. T4 GPUs (NVIDIA Tesla T4)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA T4 GPUs"
- Spot: "Custom model training preemptible NVIDIA T4 GPUs"

**Code Mapping:**
```python
# .training file
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_TESLA_T4"

# Maps to
quota_metric_suffix = "nvidia_t4_gpus"
```

---

### 2. L4 GPUs (NVIDIA L4)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_l4_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_l4_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA L4 GPUs"
- Spot: "Custom model training preemptible NVIDIA L4 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_L4"
quota_metric_suffix = "nvidia_l4_gpus"
```

---

### 3. V100 GPUs (NVIDIA Tesla V100)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_v100_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_v100_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA V100 GPUs"
- Spot: "Custom model training preemptible NVIDIA V100 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
quota_metric_suffix = "nvidia_v100_gpus"
```

---

### 4. P4 GPUs (NVIDIA Tesla P4)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_p4_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_p4_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA P4 GPUs"
- Spot: "Custom model training preemptible NVIDIA P4 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_TESLA_P4"
quota_metric_suffix = "nvidia_p4_gpus"
```

---

### 5. P100 GPUs (NVIDIA Tesla P100)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_p100_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_p100_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA P100 GPUs"
- Spot: "Custom model training preemptible NVIDIA P100 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_TESLA_P100"
quota_metric_suffix = "nvidia_p100_gpus"
```

---

### 6. A100 GPUs - 40GB (NVIDIA Tesla A100)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_a100_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_a100_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA A100 GPUs"
- Spot: "Custom model training preemptible NVIDIA A100 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_TESLA_A100"
quota_metric_suffix = "nvidia_a100_gpus"
```

---

### 7. A100 GPUs - 80GB (NVIDIA A100 80GB)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_a100_80gb_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_a100_80gb_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA A100 80GB GPUs"
- Spot: "Custom model training preemptible NVIDIA A100 80GB GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_A100_80GB"
quota_metric_suffix = "nvidia_a100_80gb_gpus"
```

---

### 8. H100 GPUs - 80GB (NVIDIA H100)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA H100 GPUs"
- Spot: "Custom model training preemptible NVIDIA H100 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_H100"
quota_metric_suffix = "nvidia_h100_gpus"
```

---

### 9. H100 GPUs - 80GB Variant (NVIDIA H100 80GB)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_h100_80gb_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_80gb_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA H100 80GB GPUs"
- Spot: "Custom model training preemptible NVIDIA H100 80GB GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_H100_80GB"
quota_metric_suffix = "nvidia_h100_80gb_gpus"
```

---

### 10. H200 GPUs - 141GB (NVIDIA H200)

**On-Demand:**
```
aiplatform.googleapis.com/custom_model_training_nvidia_h200_gpus
```

**Preemptible/Spot:**
```
aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h200_gpus
```

**Console Search Terms:**
- Regular: "Custom model training NVIDIA H200 GPUs"
- Spot: "Custom model training preemptible NVIDIA H200 GPUs"

**Code Mapping:**
```python
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_H200"
quota_metric_suffix = "nvidia_h200_gpus"
```

---

## GPUs Currently in Our Code

**File**: `training/cli/launch/core.py:4114-4122`

```python
gpu_quota_metrics = {
    "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
    "NVIDIA_TESLA_A100": "nvidia_a100_gpus",
    "NVIDIA_A100_80GB": "nvidia_a100_80gb_gpus",
    "NVIDIA_H100": "nvidia_h100_gpus",
    "NVIDIA_H100_80GB": "nvidia_h100_80gb_gpus",
    "NVIDIA_H200": "nvidia_h200_gpus",
    "NVIDIA_L4": "nvidia_l4_gpus",
}
```

**Missing from our code:**
- NVIDIA_TESLA_V100
- NVIDIA_TESLA_P4
- NVIDIA_TESLA_P100

**TODO**: Add these to `gpu_quota_metrics` map!

---

## GCloud Commands Reference

### List All Vertex AI Quotas for a Region

```bash
gcloud alpha quotas list \
    --service=aiplatform.googleapis.com \
    --consumer=projects/PROJECT_ID \
    --location=us-central1 \
    --format=json
```

### Get Specific GPU Quota Value

```bash
# T4 Regular
gcloud alpha quotas describe \
    custom_model_training_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com \
    --consumer=projects/PROJECT_ID \
    --location=us-central1 \
    --format="value(quotaDetails.value)"

# T4 Preemptible
gcloud alpha quotas describe \
    custom_model_training_preemptible_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com \
    --consumer=projects/PROJECT_ID \
    --location=us-central1 \
    --format="value(quotaDetails.value)"
```

### Request Quota Increase

```bash
gcloud alpha quotas update \
    custom_model_training_preemptible_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com \
    --consumer=projects/PROJECT_ID \
    --location=us-central1 \
    --value=1
```

---

## Console URL Templates

### Direct Quota Page Links

**Search for Vertex AI quotas:**
```
https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID&service=aiplatform.googleapis.com
```

**Search for specific GPU:**
```
https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID&service=aiplatform.googleapis.com&metric=custom_model_training_nvidia_t4_gpus
```

**Preemptible GPUs:**
```
https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID&service=aiplatform.googleapis.com&metric=custom_model_training_preemptible_nvidia_t4_gpus
```

---

## Python Code Snippets

### Convert GPU Type to Quota Metric

```python
def get_vertex_ai_quota_metric(gpu_type: str, use_preemptible: bool = False) -> str:
    """
    Convert WANDB_LAUNCH_ACCELERATOR_TYPE to Vertex AI quota metric.

    Args:
        gpu_type: GPU type from .training (e.g., "NVIDIA_TESLA_T4")
        use_preemptible: Whether to use preemptible/spot quota

    Returns:
        Full quota metric name (e.g., "aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus")
    """
    # Complete map (add V100, P4, P100!)
    gpu_quota_metrics = {
        "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
        "NVIDIA_L4": "nvidia_l4_gpus",
        "NVIDIA_TESLA_V100": "nvidia_v100_gpus",
        "NVIDIA_TESLA_P4": "nvidia_p4_gpus",
        "NVIDIA_TESLA_P100": "nvidia_p100_gpus",
        "NVIDIA_TESLA_A100": "nvidia_a100_gpus",
        "NVIDIA_A100_80GB": "nvidia_a100_80gb_gpus",
        "NVIDIA_H100": "nvidia_h100_gpus",
        "NVIDIA_H100_80GB": "nvidia_h100_80gb_gpus",
        "NVIDIA_H200": "nvidia_h200_gpus",
    }

    quota_suffix = gpu_quota_metrics.get(gpu_type, "nvidia_t4_gpus")

    if use_preemptible:
        quota_suffix = f"preemptible_{quota_suffix}"

    return f"aiplatform.googleapis.com/custom_model_training_{quota_suffix}"
```

### Example Usage

```python
# From .training file
gpu_type = "NVIDIA_TESLA_T4"
use_spot = True

# Get quota metric
metric = get_vertex_ai_quota_metric(gpu_type, use_spot)
# Returns: "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus"

# Use in gcloud command
result = subprocess.run([
    "gcloud", "alpha", "quotas", "describe",
    metric.replace("aiplatform.googleapis.com/", ""),  # Strip service prefix
    "--service=aiplatform.googleapis.com",
    "--consumer=projects/PROJECT_ID",
    "--location=us-central1",
    "--format=value(quotaDetails.value)"
])
```

---

## Quick Reference Table

| GPU Type (.training) | Regular Quota Metric | Preemptible Quota Metric |
|---------------------|---------------------|-------------------------|
| NVIDIA_TESLA_T4 | custom_model_training_nvidia_t4_gpus | custom_model_training_preemptible_nvidia_t4_gpus |
| NVIDIA_L4 | custom_model_training_nvidia_l4_gpus | custom_model_training_preemptible_nvidia_l4_gpus |
| NVIDIA_TESLA_V100 | custom_model_training_nvidia_v100_gpus | custom_model_training_preemptible_nvidia_v100_gpus |
| NVIDIA_TESLA_P4 | custom_model_training_nvidia_p4_gpus | custom_model_training_preemptible_nvidia_p4_gpus |
| NVIDIA_TESLA_P100 | custom_model_training_nvidia_p100_gpus | custom_model_training_preemptible_nvidia_p100_gpus |
| NVIDIA_TESLA_A100 | custom_model_training_nvidia_a100_gpus | custom_model_training_preemptible_nvidia_a100_gpus |
| NVIDIA_A100_80GB | custom_model_training_nvidia_a100_80gb_gpus | custom_model_training_preemptible_nvidia_a100_80gb_gpus |
| NVIDIA_H100 | custom_model_training_nvidia_h100_gpus | custom_model_training_preemptible_nvidia_h100_gpus |
| NVIDIA_H100_80GB | custom_model_training_nvidia_h100_80gb_gpus | custom_model_training_preemptible_nvidia_h100_80gb_gpus |
| NVIDIA_H200 | custom_model_training_nvidia_h200_gpus | custom_model_training_preemptible_nvidia_h200_gpus |

---

## Common Mistakes to Avoid

### ❌ Wrong: Using Compute Engine Quota Names
```python
# WRONG - This is Compute Engine!
quota_metric = "PREEMPTIBLE_NVIDIA_T4_GPUS"
command = "gcloud compute regions describe us-central1"
```

### ✅ Correct: Using Vertex AI Quota Names
```python
# CORRECT - This is Vertex AI!
quota_metric = "custom_model_training_preemptible_nvidia_t4_gpus"
command = "gcloud alpha quotas list --service=aiplatform.googleapis.com"
```

### ❌ Wrong: Forgetting Service Prefix
```bash
# WRONG - Missing service prefix
gcloud alpha quotas describe nvidia_t4_gpus
```

### ✅ Correct: Including Service in Command
```bash
# CORRECT - Service specified separately
gcloud alpha quotas describe custom_model_training_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com
```

---

## Testing Your Quota

### Check Current Quota Value

```bash
# Replace PROJECT_ID with your project
# Replace REGION with your region (us-central1, us-west2, etc.)

gcloud alpha quotas describe \
    custom_model_training_preemptible_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com \
    --consumer=projects/weight-and-biases-476906 \
    --location=us-central1 \
    --format=json
```

**Expected Output:**
```json
{
  "name": "projects/weight-and-biases-476906/locations/us-central1/services/aiplatform.googleapis.com/quotas/custom_model_training_preemptible_nvidia_t4_gpus",
  "quotaDetails": {
    "value": "0"  ← If 0, you need to request quota!
  }
}
```

---

## Sources

1. **GCP Official Docs**: https://docs.cloud.google.com/vertex-ai/docs/quotas
2. **Quota Adjuster Docs**: https://docs.cloud.google.com/docs/quotas/quota-adjuster
3. **Our Bug Report**: `VERTEX_AI_GPU_QUOTA_BUG_REPORT.md`
4. **Our Code**: `training/cli/launch/core.py:4114-4122`

---

**Last Updated**: 2025-11-16
**Maintainer**: THE-PATTERN-PERFECTIONIST 🎯
**Status**: Complete reference for quota fix implementation
