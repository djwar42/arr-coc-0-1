# GPU Pricing and Quota System Map - Complete Technical Reference

**Date**: 2025-11-16
**Status**: Comprehensive code flow analysis
**Version**: 2.0 - Expanded with full call chains and data flows

---

## 📊 Table of Contents

1. [System Overview](#system-overview)
2. [GPU Quota System](#1️⃣-gpu-quota-system)
3. [GPU Pricing System](#2️⃣-gpu-pricing-system)
4. [Complete Code Flows](#complete-code-flows)
5. [Data Structures](#data-structures)
6. [Configuration & Constants](#configuration--constants)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Quick Reference Tables](#quick-reference-tables)

---

## System Overview

### Two Distinct Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARR-COC INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────┐  ┌──────────────────────────┐   │
│  │   GPU QUOTA SYSTEM        │  │   GPU PRICING SYSTEM     │   │
│  │   (Infrastructure Mgmt)   │  │   (Cost Tracking)        │   │
│  ├───────────────────────────┤  ├──────────────────────────┤   │
│  │                           │  │                          │   │
│  │ • Check quota availability│  │ • Fetch live prices      │   │
│  │ • Auto-request if zero    │  │ • Store in Artifact Reg  │   │
│  │ • Validate before launch  │  │ • Calculate build costs  │   │
│  │ • Show manual instructions│  │ • Track campaign stats   │   │
│  │                           │  │                          │   │
│  │ Used by: Launch validation│  │ Used by: MECHA, tracking │   │
│  │                           │  │                          │   │
│  └───────────────────────────┘  └──────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: These systems are **independent** - quota checks infrastructure access, pricing tracks costs.

---

## 1️⃣ GPU QUOTA SYSTEM

### Purpose
Ensure sufficient GPU quota exists before launching Vertex AI training jobs.

### Architecture Overview

```
                    GPU QUOTA VALIDATION FLOW

User runs: python training/cli.py launch
                ↓
┌───────────────────────────────────────────────────────────────┐
│ launch_training_job() - training/cli/launch/core.py:750      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1.5 (line 815): Auto-request GPU quota                 │
│     ↓                                                         │
│  _auto_request_gpu_quota(config, region, status)             │
│     │                                                         │
│     ├─→ Returns "EXISTS" → Proceed with launch ✓             │
│     ├─→ Returns "REQUESTED" → HALT launch, wait for approval │
│     └─→ Returns "FAILED" → Proceed (will fail at Vertex AI)  │
│                                                               │
│  [Step 1.6 REMOVED - old verification deleted 2025-11-16]    │
│                                                               │
│  Step 2: Submit to W&B queue                                 │
│  Step 3-7: Setup infrastructure                              │
│  Step 8: Launch training job                                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Detailed Function: `_auto_request_gpu_quota()`

**Location**: `training/cli/launch/core.py:3913-4083`

**Function Signature**:
```python
def _auto_request_gpu_quota(config: Dict, region: str, status) -> str
```

**Returns**:
- `"EXISTS"` - Quota available (>0), proceed
- `"REQUESTED"` - Quota auto-requested, HALT launch
- `"FAILED"` - Auto-request failed, proceed to manual

**Complete Code Flow**:

```python
_auto_request_gpu_quota(config, region, status)
    │
    ├─ Line 3930: Extract config values
    │   • project_id = config["GCP_PROJECT_ID"]
    │   • gpu_type = config["WANDB_LAUNCH_ACCELERATOR_TYPE"]  # "NVIDIA_TESLA_T4"
    │   • gpu_count = config["WANDB_LAUNCH_ACCELERATOR_COUNT"]  # "1"
    │   • use_spot = config["WANDB_LAUNCH_USE_PREEMPTIBLE"]  # "true"
    │
    ├─ Line 3937: Map GPU type to Vertex AI quota metric
    │   gpu_quota_metrics = {
    │       "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
    │       "NVIDIA_TESLA_A100": "nvidia_a100_gpus",
    │       "NVIDIA_A100_80GB": "nvidia_a100_80gb_gpus",
    │       "NVIDIA_H100": "nvidia_h100_gpus",
    │       "NVIDIA_H100_80GB": "nvidia_h100_80gb_gpus",
    │       "NVIDIA_H200": "nvidia_h200_gpus",
    │       "NVIDIA_L4": "nvidia_l4_gpus",
    │   }
    │   quota_metric_suffix = gpu_quota_metrics[gpu_type]
    │   quota_metric = f"aiplatform.googleapis.com/custom_model_training_{suffix}"
    │   # Result: "aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus"
    │
    ├─ Line 3962: STEP 1 - Quick Check (Compute Engine quotas as approximation)
    │   │
    │   ├─ Run: gcloud compute regions describe {region} --format=json
    │   │
    │   ├─ Parse quotas from response
    │   │   region_info = json.loads(result.stdout)
    │   │   quotas = region_info["quotas"]
    │   │
    │   ├─ Line 3996: Map to Compute Engine quota names
    │   │   quota_map = {
    │   │       "NVIDIA_H200": "NVIDIA_H200_GPUS",
    │   │       "NVIDIA_H100": "NVIDIA_H100_GPUS",
    │   │       "NVIDIA_TESLA_T4": "NVIDIA_T4_GPUS",
    │   │       ...
    │   │   }
    │   │   gcp_quota_metric = quota_map[gpu_type]
    │   │   if use_spot:
    │   │       gcp_quota_metric = f"PREEMPTIBLE_{gcp_quota_metric}"
    │   │   # Result: "PREEMPTIBLE_NVIDIA_T4_GPUS"
    │   │
    │   ├─ Line 4010: Search for matching quota
    │   │   for quota in quotas:
    │   │       if quota["metric"] == gcp_quota_metric:
    │   │           limit = quota["limit"]
    │   │           if limit > 0:
    │   │               return "EXISTS" ✓  # Quota available!
    │   │
    │   └─ If limit == 0 or not found → Continue to STEP 2
    │
    ├─ Line 4032: STEP 2 - Auto-request Vertex AI quota
    │   │
    │   ├─ Run: gcloud alpha compute regions update-quota
    │   │        --project={project_id}
    │   │        --region={region}
    │   │        --quota-metric={quota_metric}  # Vertex AI metric!
    │   │        --new-limit={gpu_count}
    │   │
    │   ├─ Line 4049: If successful
    │   │   status("✅ GPU QUOTA REQUEST SUBMITTED!")
    │   │   status(f"   GPU Type: {gpu_type}")
    │   │   status(f"   Requested: {gpu_count} GPU(s)")
    │   │   status("📧 Google will email when approved (1-2 days)")
    │   │   return "REQUESTED"  # HALT launch!
    │   │
    │   └─ Line 4063: If failed
    │       status("⚠️ Auto-request failed")
    │       status(f"Error: {stderr[:200]}")
    │       return "FAILED"  # Proceed to manual verification
    │
    └─ Line 4077: Exception handling
        return "FAILED"  # Any error → proceed to manual
```

**Critical Notes**:

1. **Two Quota Systems**: Line 3991 comment explains this clearly:
   ```python
   # NOTE: This checks COMPUTE ENGINE quotas to see if quota EXISTS
   # But the actual auto-request (line 4034) uses VERTEX AI quotas
   # We check Compute Engine first because it's faster/simpler
   # If Compute Engine quota > 0, we assume Vertex AI quota also exists
   # This is an approximation - the two quota systems are separate!
   ```

2. **Why Approximation Works**:
   - Compute Engine check is fast (single API call)
   - If CE quota exists, VA quota usually exists too
   - If CE quota is zero, VA quota is definitely zero
   - Auto-request uses **correct** Vertex AI quota metric

3. **Quota Namespace Difference**:
   ```
   Compute Engine:  PREEMPTIBLE_NVIDIA_T4_GPUS
   Vertex AI:       aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus
   ```

### Launch Integration

**File**: `training/cli/launch/core.py:815-830`

```python
# Step 1.5: Auto-request GPU quota if needed
quota_status = _auto_request_gpu_quota(config, region, status)

if quota_status == "REQUESTED":
    # Quota just requested - HALT launch!
    status("")
    status("🛑 LAUNCH HALTED - GPU quota requested, awaiting approval")
    status("")
    status("Next steps:")
    status("1. Wait 1-2 business days for Google to approve")
    status("2. Check email for approval notification")
    status("3. Run 'python training/cli.py launch' again")
    status("")
    return False  # Abort launch

# If quota_status == "EXISTS" or "FAILED", continue with launch
# "FAILED" means auto-request didn't work, but job may still succeed
# (user might have manually requested quota earlier)
```

### Deleted System (2025-11-16)

**Old Function**: `_verify_gpu_quota()` - **DELETED**

**Location**: Previously at `training/cli/launch/core.py:4078-4255` (178 lines)

**Why Deleted**:
```python
# _verify_gpu_quota() DELETED (2025-11-16)
#
# This function checked COMPUTE ENGINE quotas (NVIDIA_T4_GPUS, PREEMPTIBLE_NVIDIA_T4_GPUS)
# But Vertex AI Custom Training uses VERTEX AI quotas (custom_model_training_nvidia_t4_gpus)
# These are COMPLETELY DIFFERENT quota namespaces!
#
# Problem: Validation passed on wrong quota → submission failed on correct quota
# Solution: Rely on _auto_request_gpu_quota() (line 3913) which checks CORRECT quotas
#
# See detailed analysis in:
# - VERTEX_AI_GPU_QUOTA_BUG_REPORT.md
# - GPU_QUOTA_SYSTEMS_ANALYSIS.md
```

**What It Did (Historical Reference)**:
1. Checked **Compute Engine** quotas using `gcloud compute regions describe`
2. Validated quota against requested GPU count
3. Showed GO/NO-GO message
4. Called `show_gpu_quota_instructions()` on failure

**The Bug**:
- Checked `PREEMPTIBLE_NVIDIA_T4_GPUS` (Compute Engine) ❌
- Vertex AI uses `custom_model_training_nvidia_t4_gpus` ✓
- Validation passed (CE quota = 1) but Vertex AI failed (VA quota = 0)

### Manual Instructions (Unused Placeholder)

**File**: `training/cli/shared/gpu_quota_instruct.py`

**Status**: Code exists but **NOT CALLED** anywhere

**Function**: `show_gpu_quota_instructions()`

**Lines**: 97-153

**What It Would Do** (if called):
```python
def show_gpu_quota_instructions(
    project_id, region, gpu_type, gpu_count, use_preemptible, status
):
    # Show manual quota request instructions
    status("💡 MANUAL QUOTA REQUEST (if needed):")
    status(f"1. Visit: https://console.cloud.google.com/iam-admin/quotas?project={project_id}")

    if use_preemptible:
        status(f"2. Search for: Preemptible NVIDIA {gpu_name} GPUs")  # ❌ WRONG!
    else:
        status(f"2. Search for: NVIDIA {gpu_name} GPUs")  # ❌ WRONG!

    status(f"3. Select region: {region}")
    status(f"4. Request: {gpu_count} GPU(s)")
```

**BUG in Instructions** (Lines 129-136):
- Tells users to search for **Compute Engine quotas** ❌
- Should search for **Vertex AI quotas** ✓
- Example fix:
  ```python
  # Wrong:
  status("2. Search for: Preemptible NVIDIA T4 GPUs")

  # Correct:
  status("2. Search for: Custom model training preemptible NVIDIA T4 GPUs")
  ```

**Future Enhancement** (Lines 7-95):
- Epic mythological narrative planned (ORDEAL OF DIVINE THUNDER)
- Zeus as GPU quota gatekeeper
- Hermes Trismegistus providing ridiculous alchemical advice
- Enkidu wandering in from C3 quota saga (crossover cameo)
- See file comments for complete mythology design

### Quota Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUOTA CHECK FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  config["WANDB_LAUNCH_ACCELERATOR_TYPE"] = "NVIDIA_TESLA_T4"    │
│  config["WANDB_LAUNCH_ACCELERATOR_COUNT"] = "1"                 │
│  config["WANDB_LAUNCH_USE_PREEMPTIBLE"] = "true"                │
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ _auto_request_gpu_quota()                               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │ Step 1: Map GPU type → CE quota name                   │   │
│  │   "NVIDIA_TESLA_T4" → "NVIDIA_T4_GPUS"                  │   │
│  │   + use_spot → "PREEMPTIBLE_NVIDIA_T4_GPUS"            │   │
│  │                                                         │   │
│  │ Step 2: Check CE quota (approximation)                 │   │
│  │   gcloud compute regions describe                      │   │
│  │   → quotas[metric="PREEMPTIBLE_NVIDIA_T4_GPUS"]        │   │
│  │   → limit = 1.0 ✓ (exists!)                            │   │
│  │   → return "EXISTS"                                     │   │
│  │                                                         │   │
│  │ [OR if limit == 0]                                      │   │
│  │                                                         │   │
│  │ Step 3: Map GPU type → VA quota name                   │   │
│  │   "NVIDIA_TESLA_T4" → "nvidia_t4_gpus"                  │   │
│  │   → "aiplatform.googleapis.com/                        │   │
│  │       custom_model_training_nvidia_t4_gpus"            │   │
│  │                                                         │   │
│  │ Step 4: Auto-request VA quota                          │   │
│  │   gcloud alpha compute regions update-quota            │   │
│  │   --quota-metric=aiplatform.googleapis.com/...         │   │
│  │   → Success: return "REQUESTED" (HALT!)                │   │
│  │   → Failure: return "FAILED" (continue)                │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                      ↓                                          │
│  Launch flow decision:                                          │
│    "EXISTS" → Continue launch ✓                                 │
│    "REQUESTED" → HALT launch, wait for approval ✋              │
│    "FAILED" → Continue anyway (may fail at Vertex AI) ⚠️        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2️⃣ GPU PRICING SYSTEM

### Purpose
Track infrastructure costs for billing, accounting, and campaign statistics. **NOT** for showing user-facing cost estimates.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRICING INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ Cloud        │──→│ Cloud        │──→│ Artifact         │    │
│  │ Scheduler    │   │ Function     │   │ Registry         │    │
│  │              │   │              │   │                  │    │
│  │ Every 20 min │   │ Fetch GCP    │   │ Store pricing    │    │
│  │              │   │ Billing API  │   │ JSON (generic)   │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
│                                                ↓                │
│                         ┌──────────────────────────────────┐    │
│                         │ Consumer Code                    │    │
│                         │ - MECHA (battle pricing display) │    │
│                         │ - Campaign stats (cost tracking) │    │
│                         │ - Launch core (build costs)      │    │
│                         └──────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration & Constants

**File**: `training/cli/shared/pricing_config.py`

```python
# Project config (from .training file)
PROJECT_ID = "weight-and-biases-476906"
REGION = "us-central1"

# Pricing infrastructure names
FUNCTION_NAME = "arr-coc-pricing-runner"
SCHEDULER_JOB = "arr-coc-pricing-scheduler"
SCHEDULER_INTERVAL_MINUTES = 20  # */20 * * * *
REPOSITORY = "arr-coc-pricing"
PACKAGE = "gcp-pricing"

# Pricing data schema (validation)
PRICING_SCHEMA = {
    "updated": {
        "type": "timestamp",
        "required": True,
        "should_have_data": False
    },
    "c3_machines": {
        "type": "dict",
        "required": True,
        "should_have_data": True  # Must have regions!
    },
    "e2_machines": {
        "type": "dict",
        "required": True,
        "should_have_data": True
    },
    "gpus_spot": {
        "type": "dict",
        "required": True,
        "should_have_data": True
    },
    "gpus_ondemand": {
        "type": "dict",
        "required": True,
        "should_have_data": True
    },
}
```

### Setup Flow (Complete Walkthrough)

**Entry Point**: `training/cli/setup/pricing_setup.py:96`

```python
def setup_pricing_infrastructure(status_callback) -> bool
```

**Complete Flow**:

```
setup_pricing_infrastructure(status)
    │
    ├─ Line 101: PHASE 0 - Create Repository
    │   │
    │   └─ create_pricing_repository(status)
    │       │
    │       ├─ Check if exists:
    │       │   gcloud artifacts repositories describe arr-coc-pricing
    │       │   → Exists? Return early (idempotent)
    │       │
    │       └─ Create generic repository:
    │           gcloud artifacts repositories create arr-coc-pricing
    │               --repository-format=generic
    │               --location=us-central1
    │               --description="ARR-COC pricing data storage"
    │
    ├─ Line 105: PHASE 1 - Grant OIDC Permissions
    │   │
    │   └─ grant_actAs_permission(status)  # Silent
    │       │
    │       ├─ Get current user:
    │       │   gcloud config get-value account
    │       │   → user_email
    │       │
    │       ├─ Define service account:
    │       │   {PROJECT_ID}@appspot.gserviceaccount.com
    │       │
    │       ├─ Check if permission already exists (idempotent):
    │       │   gcloud iam service-accounts get-iam-policy {sa} --format=json
    │       │   → Parse bindings
    │       │   → If "roles/iam.serviceAccountUser" + "user:{email}" exists
    │       │      → return (True, None)  # Already granted!
    │       │
    │       ├─ Grant permission:
    │       │   gcloud iam service-accounts add-iam-policy-binding
    │       │       {sa}
    │       │       --member=user:{user_email}
    │       │       --role=roles/iam.serviceAccountUser
    │       │       --condition=expression=resource.service=="cloudscheduler.googleapis.com",
    │       │                    title=OIDCSchedulerOnly
    │       │
    │       └─ Retry with backoff (4 attempts: 0s, 1s, 4s, 8s)
    │           → retry_with_backoff(try_grant_permission, max_attempts=4)
    │
    ├─ Line 109: PHASE 2 - Deploy Cloud Function
    │   │
    │   └─ deploy_cloud_function(status)
    │       │
    │       ├─ Enable APIs:
    │       │   • cloudfunctions.googleapis.com
    │       │   • cloudbuild.googleapis.com
    │       │   • cloudbilling.googleapis.com
    │       │
    │       ├─ Deploy function (with streaming output):
    │       │   gcloud functions deploy arr-coc-pricing-runner
    │       │       --gen2
    │       │       --region=us-central1
    │       │       --runtime=python312
    │       │       --entry-point=fetch_pricing
    │       │       --source={function_dir}
    │       │       --trigger-http
    │       │       --allow-unauthenticated
    │       │       --timeout=540s
    │       │       --memory=512MB
    │       │       --max-instances=1
    │       │
    │       ├─ Timeout: 10 minutes per attempt
    │       │
    │       ├─ Retry logic (4 attempts with 1s, 4s, 8s backoff)
    │       │
    │       ├─ Stream output line-by-line:
    │       │   for line in process.stdout:
    │       │       sys.stdout.write(f"               {line}")
    │       │
    │       └─ Verify deployment:
    │           gcloud functions describe arr-coc-pricing-runner
    │               --format=value(state)
    │           → Must be "ACTIVE"
    │
    ├─ Line 113: PHASE 3 - Bootstrap Pricing Data
    │   │
    │   └─ bootstrap_pricing(status)
    │       │
    │       ├─ Try to fetch existing pricing from Artifact Registry:
    │       │   │
    │       │   └─ artifact_pricing.fetch_pricing_no_save()
    │       │       ├─ Get latest version via HTTP:
    │       │       │   GET https://artifactregistry.googleapis.com/v1/
    │       │       │       projects/{PROJECT}/locations/{LOCATION}/
    │       │       │       repositories/{REPO}/packages/{PACKAGE}/versions
    │       │       │       ?pageSize=1&orderBy=createTime desc
    │       │       │   → version: "1.0.20251116-143052"
    │       │       │
    │       │       └─ Download via gcloud CLI:
    │       │           gcloud artifacts generic download
    │       │               --package=gcp-pricing
    │       │               --version={version}
    │       │               --destination={tmpdir}
    │       │           → Temp file: /var/folders/.../gcp-live-pricing.json
    │       │           → Load JSON → return (pricing_data, version, size_kb)
    │       │
    │       ├─ If pricing found → Validate schema:
    │       │   │
    │       │   ├─ Get expected fields:
    │       │   │   pricing_config.get_required_fields()
    │       │   │   → {"updated": False, "c3_machines": True, ...}
    │       │   │
    │       │   ├─ Check each field:
    │       │   │   for field, should_have_data in expected_fields.items():
    │       │   │       if field not in existing_data:
    │       │   │           missing_fields.append(f"{field} (missing)")
    │       │   │       elif should_have_data and len(existing_data[field]) == 0:
    │       │   │           missing_fields.append(f"{field} (empty)")
    │       │   │
    │       │   ├─ If schema invalid:
    │       │   │   raise ValueError(f"Schema mismatch: {missing_fields}")
    │       │   │   → Triggers fresh fetch
    │       │   │
    │       │   └─ Check age:
    │       │       age_minutes = get_pricing_age_minutes(existing_data)
    │       │       if age_minutes < SCHEDULER_INTERVAL_MINUTES:  # < 20 min
    │       │           → Use existing ✓
    │       │           → Trigger Cloud Function for first run
    │       │           → return
    │       │       else:
    │       │           → Fetch fresh (stale)
    │       │
    │       ├─ If pricing missing or stale → Fetch fresh:
    │       │   │
    │       │   └─ _fetch_pricing_inline(status)
    │       │       │
    │       │       ├─ Get OAuth token:
    │       │       │   gcloud auth print-access-token
    │       │       │
    │       │       ├─ Query GCP Cloud Billing API:
    │       │       │   GET https://cloudbilling.googleapis.com/v1/
    │       │       │       services/6F81-5844-456A/skus
    │       │       │       ?pageSize=500
    │       │       │
    │       │       │   Page through results (nextPageToken)
    │       │       │   → ~30,000 SKUs total
    │       │       │   → Show progress every 5000 SKUs
    │       │       │
    │       │       ├─ For each SKU, extract:
    │       │       │   • Price: units + (nanos / 1e9)
    │       │       │   • Description
    │       │       │   • SKU ID
    │       │       │   • Usage type (Spot, Preemptible, OnDemand, Commit)
    │       │       │   • Regions
    │       │       │
    │       │       ├─ Filter and categorize:
    │       │       │   │
    │       │       │   ├─ C3 machines (spot):
    │       │       │   │   if "c3" in desc and "spot" in desc:
    │       │       │   │       pricing_data["c3_machines"][region]["cpu_per_core_spot"].append(sku)
    │       │       │   │       pricing_data["c3_machines"][region]["ram_per_gb_spot"].append(sku)
    │       │       │   │
    │       │       │   ├─ E2 machines (on-demand):
    │       │       │   │   if "e2" in desc and "instance" in desc:
    │       │       │   │       pricing_data["e2_machines"][region]["cpu_per_core_ondemand"].append(sku)
    │       │       │   │       pricing_data["e2_machines"][region]["ram_per_gb_ondemand"].append(sku)
    │       │       │   │
    │       │       │   └─ GPUs (all types):
    │       │       │       if "gpu" in desc or "tpu" in desc:
    │       │       │           if "Spot" or "Preemptible" in desc:
    │       │       │               pricing_data["gpus_spot"][region].append(sku)
    │       │       │           else:
    │       │       │               pricing_data["gpus_ondemand"][region].append(sku)
    │       │       │               # Includes: OnDemand, 1-Year Commit, 3-Year Commit
    │       │       │
    │       │       └─ Return pricing_data with complete SKU lists
    │       │
    │       ├─ Sort all pricing lists (cheapest first):
    │       │   for region_data in pricing_data["c3_machines"].values():
    │       │       region_data["cpu_per_core_spot"].sort(key=lambda x: x["price"])
    │       │       region_data["ram_per_gb_spot"].sort(key=lambda x: x["price"])
    │       │   # ... same for e2_machines, gpus_spot, gpus_ondemand
    │       │
    │       ├─ Count and display stats:
    │       │   status(f"  • C3 machines (spot): {c3_regions} regions")
    │       │   status(f"  • E2 machines (on-demand): {e2_regions} regions")
    │       │   status(f"  • GPUs (spot): {gpu_spot_regions} regions - T4=45, L4=23, ...")
    │       │   status(f"  • GPUs (on-demand): {gpu_ondemand_regions} regions - ...")
    │       │
    │       ├─ Upload to Artifact Registry:
    │       │   artifact_pricing.upload_pricing_to_artifact_registry(pricing_data)
    │       │   │
    │       │   └─ Generate version:
    │       │       timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    │       │       version = f"1.0.{timestamp}"  # "1.0.20251116-143052"
    │       │
    │       │       Save to temp file → gcp-live-pricing.json
    │       │
    │       │       gcloud artifacts generic upload
    │       │           --package=gcp-pricing
    │       │           --version={version}
    │       │           --source={temp_file}
    │       │
    │       │       Temp file auto-deleted by context manager
    │       │
    │       └─ Trigger Cloud Function (first run):
    │           _trigger_and_verify_function(status)
    │           │
    │           ├─ Trigger (async):
    │           │   subprocess.Popen([
    │           │       "gcloud", "functions", "call",
    │           │       "arr-coc-pricing-runner",
    │           │       "--gen2", "--region=us-central1"
    │           │   ])
    │           │
    │           └─ Watch logs for startup (90s timeout):
    │               while elapsed < 90s:
    │                   gcloud logging read
    │                       'resource.type="cloud_run_revision"
    │                        AND textPayload:"PRICING_RUNNER_STARTED"'
    │
    │                   if found:
    │                       status("✓ Cloud Function verified")
    │                       return
    │
    │                   sleep(2s)
    │
    │               # Timeout OK - function may still be cold starting
    │
    └─ Line 116: PHASE 4 - Create Cloud Scheduler
        │
        └─ create_scheduler(status)
            │
            ├─ Enable API:
            │   gcloud services enable cloudscheduler.googleapis.com
            │
            ├─ Get function URL:
            │   gcloud functions describe arr-coc-pricing-runner
            │       --format=value(serviceConfig.uri)
            │   → function_url
            │
            ├─ Check if scheduler exists and is correctly configured:
            │   gcloud scheduler jobs describe arr-coc-pricing-scheduler
            │       --format=json
            │
            │   Expected config:
            │       schedule: "*/20 * * * *"  (every 20 min)
            │       uri: {function_url}
            │       serviceAccountEmail: {PROJECT}@appspot.gserviceaccount.com
            │       state: "ENABLED"
            │
            │   If config matches → skip creation (idempotent)
            │   If config wrong → delete and recreate
            │
            ├─ Create scheduler job:
            │   gcloud scheduler jobs create http arr-coc-pricing-scheduler
            │       --location=us-central1
            │       --schedule="*/20 * * * *"
            │       --uri={function_url}
            │       --http-method=GET
            │       --oidc-service-account-email={sa}
            │       --oidc-token-audience={function_url}
            │
            └─ Return success
```

### Teardown Flow (Complete Walkthrough)

**Entry Point**: `training/cli/teardown/pricing_teardown.py:95`

```python
def teardown_pricing_infrastructure(status_callback)
```

**Complete Flow**:

```
teardown_pricing_infrastructure(status)
    │
    ├─ Line 103: PHASE 1 - Delete Cloud Scheduler
    │   │
    │   └─ delete_scheduler(status)
    │       │
    │       └─ Retry with backoff (4 attempts):
    │           gcloud scheduler jobs delete arr-coc-pricing-scheduler
    │               --location=us-central1
    │               --quiet
    │
    │           Success or "not found" → (True, None)
    │           Other error → (False, error_msg) → retry
    │
    ├─ Line 107: PHASE 2 - Delete Cloud Function
    │   │
    │   └─ delete_cloud_function(status)
    │       │
    │       └─ Retry with backoff (4 attempts):
    │           gcloud functions delete arr-coc-pricing-runner
    │               --gen2
    │               --region=us-central1
    │               --quiet
    │
    │           Success or "not found" → (True, None)
    │           Other error → (False, error_msg) → retry
    │
    ├─ Line 111: PHASE 3 - Revoke OIDC Permissions
    │   │
    │   └─ revoke_actAs_permission(status)  # Silent
    │       │
    │       ├─ Get current user:
    │       │   gcloud config get-value account
    │       │
    │       └─ Remove permission:
    │           gcloud iam service-accounts remove-iam-policy-binding
    │               {sa}
    │               --member=user:{email}
    │               --role=roles/iam.serviceAccountUser
    │               --condition=expression=resource.service=="cloudscheduler.googleapis.com",
    │                            title=OIDCSchedulerOnly
    │
    │           Ignore "not found" or "no binding" errors
    │
    ├─ Line 115: PHASE 4 - Disable Cloud Billing API
    │   │
    │   └─ disable_cloudbilling_api(status)  # Silent
    │       │
    │       └─ gcloud services disable cloudbilling.googleapis.com
    │           → May fail if still in use (OK to continue)
    │
    └─ Line 119: PRESERVE Artifact Registry Repository ✓
        status("ℹ Pricing repository preserved (historical data intact)")

        # NOTE: Repository NOT deleted!
        # Contains historical pricing data
        # Function cleanup_artifact_registry() exists but NOT called
```

### Pricing Data Access (Consumer Patterns)

**Pattern 1: MECHA Battle Epic** (Display pricing in battle animations)

**File**: `training/cli/launch/mecha/mecha_battle_epic.py:258, 288`

```python
from ...shared.artifact_pricing import get_spot_price

# Get region pricing
region_pricing = pricing["c3_machines"].get(region, {})
cpu_skus = region_pricing.get("cpu_per_core_spot", [])

# Extract cheapest price
cpu_per_core = get_spot_price(cpu_skus)
# → Returns: cpu_skus[0]["price"] if exists (cheapest, since sorted)

# Display in battle text
status(f"⚡ L4 SPOT PRICE: ${price:.2f}/hr ⚡")
```

**Pattern 2: Launch Core** (Calculate build costs for campaign stats)

**File**: `training/cli/shared/pricing/get_live_prices.py:10-58`

```python
def get_live_price_for_launch(machine_type: str, region: str) -> float:
    # Fetch pricing from Artifact Registry
    pricing_data, _, _ = fetch_pricing_no_save()

    # C3 machines (MECHA worker pool - spot pricing)
    if machine_type.startswith("c3-standard-"):
        vcpus = int(machine_type.split("-")[-1])  # "c3-standard-176" → 176
        ram_gb = vcpus * 4  # C3: 4 GB RAM per vCPU

        c3_data = pricing_data["c3_machines"][region]
        cpu_skus = c3_data["cpu_per_core_spot"]
        ram_skus = c3_data["ram_per_gb_spot"]

        cpu_price = get_spot_price(cpu_skus)  # $/core/hour
        ram_price = get_spot_price(ram_skus)  # $/GB/hour

        return (vcpus * cpu_price) + (ram_gb * ram_price)

    # E2 machines (Cloud Build default - on-demand pricing)
    elif machine_type == "E2_HIGHCPU_8":
        vcpus = 8
        ram_gb = 8  # E2_HIGHCPU: 1 GB RAM per vCPU

        e2_data = pricing_data["e2_machines"][region]
        cpu_skus = e2_data["cpu_per_core_ondemand"]
        ram_skus = e2_data["ram_per_gb_ondemand"]

        cpu_price = get_standard_price(cpu_skus)  # $/core/hour (on-demand)
        ram_price = get_standard_price(ram_skus)  # $/GB/hour (on-demand)

        return (vcpus * cpu_price) + (ram_gb * ram_price)

    else:
        return 0.0  # Unknown machine type
```

**Pattern 3: Generic Pricing Extraction**

**File**: `training/cli/shared/artifact_pricing.py:229-397`

```python
# Extract cheapest spot/preemptible price
def get_spot_price(sku_list):
    spot_skus = [s for s in sku_list if s["usage_type"] in ["Preemptible", "Spot"]]
    return spot_skus[0]["price"] if spot_skus else None

# Extract cheapest on-demand price
def get_standard_price(sku_list):
    ondemand_skus = [s for s in sku_list if s["usage_type"] == "OnDemand"]
    return ondemand_skus[0]["price"] if ondemand_skus else None

# Extract 1-year commitment price
def get_commitment_1yr_price(sku_list):
    commit_skus = [s for s in sku_list
                   if "1 Year" in s["description"] or "1yr" in s["description"].lower()]
    return commit_skus[0]["price"] if commit_skus else None

# Extract 3-year commitment price
def get_commitment_3yr_price(sku_list):
    commit_skus = [s for s in sku_list
                   if "3 Year" in s["description"] or "3yr" in s["description"].lower()]
    return commit_skus[0]["price"] if commit_skus else None

# Get all available pricing tiers with names
def all_prices(sku_list):
    options = []

    # Spot/Preemptible
    spot = get_spot_price(sku_list)
    if spot:
        options.append({
            "name": "Spot (Preemptible)",
            "price": spot,
            "description": ...,
            "sku_id": ...,
            "usage_type": ...
        })

    # On-Demand, 1-Year, 3-Year
    # ... (same pattern)

    return options
```

---

## Complete Code Flows

### Flow 1: Full Setup → Launch → Pricing Usage

```
┌─────────────────────────────────────────────────────────────────────┐
│ User: python training/cli.py setup                                  │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ SETUP PHASE (training/cli/setup/pricing_setup.py:96)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Step 1: Create Repository                                          │
│   gcloud artifacts repositories create arr-coc-pricing              │
│   → us-central1-generic.pkg.dev/PROJECT/arr-coc-pricing             │
│                                                                     │
│ Step 2: Grant OIDC Permissions (silent)                            │
│   gcloud iam service-accounts add-iam-policy-binding               │
│   → roles/iam.serviceAccountUser granted                           │
│                                                                     │
│ Step 3: Deploy Cloud Function                                      │
│   gcloud functions deploy arr-coc-pricing-runner                   │
│   → Gen2, Python 3.12, 512MB, 540s timeout                         │
│   → Entry point: fetch_pricing()                                   │
│                                                                     │
│ Step 4: Bootstrap Pricing                                          │
│   Try fetch from Artifact Registry                                 │
│   → Not found                                                       │
│   Fetch from GCP Billing API                                       │
│   → Query ~30,000 SKUs                                              │
│   → Filter: C3, E2, GPUs                                            │
│   → Sort by price (cheapest first)                                 │
│   Upload to Artifact Registry                                      │
│   → Version: 1.0.20251116-143052                                   │
│   Trigger Cloud Function (first run)                               │
│   → Watch logs for "PRICING_RUNNER_STARTED"                        │
│                                                                     │
│ Step 5: Create Scheduler                                           │
│   gcloud scheduler jobs create http arr-coc-pricing-scheduler      │
│   → Schedule: */20 * * * * (every 20 min)                          │
│   → Trigger: Cloud Function HTTP endpoint                          │
│   → Auth: OIDC with service account                                │
│                                                                     │
│ ✓ Pricing infrastructure deployed                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
                   [20 minutes pass]
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ AUTOMATIC PRICING UPDATE (Cloud Scheduler → Function)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Cloud Scheduler triggers:                                          │
│   HTTP GET → arr-coc-pricing-runner function URL                   │
│   Auth: OIDC token from service account                            │
│                                                                     │
│ Cloud Function executes:                                           │
│   fetch_pricing() - same logic as bootstrap                        │
│   → Fetch ~30,000 SKUs from GCP Billing API                        │
│   → Filter, categorize, sort                                       │
│   → Upload to Artifact Registry                                    │
│   → New version: 1.0.20251116-145052                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
                 [User launches training]
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ User: python training/cli.py launch                                │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAUNCH PHASE (training/cli/launch/core.py:750)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ [Line 815] Auto-request GPU quota                                  │
│   _auto_request_gpu_quota(config, region, status)                  │
│   → Check Compute Engine quota (approximation)                     │
│   → If zero, auto-request Vertex AI quota                          │
│   → Return: "EXISTS" (proceed)                                     │
│                                                                     │
│ [Line 804] Handle base image build                                 │
│   _handle_base_image(config, region, status)                       │
│   → Build ML libraries image                                       │
│                                                                     │
│ [Line 808] Build training image                                    │
│   _handle_training_image(config, region, status)                   │
│   → Uses pricing for cost tracking:                                │
│       get_live_price_for_launch("E2_HIGHCPU_8", "us-west2")        │
│       ├─ fetch_pricing_no_save()                                   │
│       │   ├─ HTTP: Get latest version from Artifact Registry       │
│       │   ├─ gcloud: Download pricing JSON to temp                 │
│       │   └─ Return: (pricing_data, version, size_kb)              │
│       ├─ Extract E2 on-demand pricing                              │
│       │   cpu_price = get_standard_price(cpu_skus)                 │
│       │   ram_price = get_standard_price(ram_skus)                 │
│       └─ Calculate: (8 × cpu_price) + (8 × ram_price)              │
│                                                                     │
│ [During MECHA battle animations]                                   │
│   mecha_battle_epic.py uses pricing for display:                   │
│   get_spot_price(cpu_skus)                                         │
│   → Display: "⚡ C3 SPOT PRICE: $0.012/core/hr ⚡"                  │
│                                                                     │
│ [Continue with job submission...]                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Flow 2: Pricing Data Journey (Birth to Usage)

```
                    PRICING DATA LIFECYCLE

┌─────────────────────────────────────────────────────────────────────┐
│ BIRTH: GCP Cloud Billing API                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Source: https://cloudbilling.googleapis.com/v1/                    │
│         services/6F81-5844-456A/skus                                │
│                                                                     │
│ Data volume: ~30,000 SKUs                                          │
│ Pagination: 500 SKUs per page                                      │
│ Fields per SKU:                                                     │
│   {                                                                 │
│     "skuId": "...",                                                 │
│     "description": "Spot Preemptible Nvidia Tesla T4 GPU ...",     │
│     "category": {"usageType": "Preemptible"},                      │
│     "serviceRegions": ["us-central1", ...],                        │
│     "pricingInfo": [{                                              │
│       "pricingExpression": {                                       │
│         "tieredRates": [{                                          │
│           "unitPrice": {                                           │
│             "units": "0",                                          │
│             "nanos": 140000000  # $0.14/hour                       │
│           }                                                         │
│         }]                                                          │
│       }                                                             │
│     }]                                                              │
│   }                                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
            _fetch_pricing_inline() processes
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TRANSFORMATION: Categorize & Structure                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ pricing_data = {                                                    │
│   "updated": "2025-11-16T14:30:52Z",                                │
│   "c3_machines": {                                                  │
│     "us-central1": {                                                │
│       "cpu_per_core_spot": [                                        │
│         {                                                           │
│           "price": 0.01234,                                         │
│           "description": "Compute optimized Core...",               │
│           "sku_id": "...",                                          │
│           "usage_type": "Preemptible"                               │
│         },                                                          │
│         # ... more SKUs, sorted by price (cheapest first)          │
│       ],                                                            │
│       "ram_per_gb_spot": [...]                                      │
│     },                                                              │
│     "us-west2": {...},                                              │
│     # ... all GCP regions                                           │
│   },                                                                │
│   "e2_machines": {                                                  │
│     "us-central1": {                                                │
│       "cpu_per_core_ondemand": [...],                               │
│       "ram_per_gb_ondemand": [...]                                  │
│     },                                                              │
│     # ... all regions                                               │
│   },                                                                │
│   "gpus_spot": {                                                    │
│     "us-central1": [                                                │
│       {                                                             │
│         "price": 0.14,                                              │
│         "description": "Nvidia Tesla T4 GPU attached to...",        │
│         "sku_id": "...",                                            │
│         "usage_type": "Spot"                                        │
│       },                                                            │
│       # ... T4, L4, V100, P4, P100, A100, H100, H200                │
│     ],                                                              │
│     # ... all regions                                               │
│   },                                                                │
│   "gpus_ondemand": {                                                │
│     "us-central1": [                                                │
│       # OnDemand pricing                                            │
│       {"price": 0.35, "usage_type": "OnDemand", ...},               │
│       # 1-Year commitment pricing                                   │
│       {"price": 0.245, "usage_type": "Commit", "description": "... 1 Year Commitment", ...},│
│       # 3-Year commitment pricing                                   │
│       {"price": 0.175, "usage_type": "Commit", "description": "... 3 Year Commitment", ...},│
│     ],                                                              │
│     # ... all regions                                               │
│   }                                                                 │
│ }                                                                   │
│                                                                     │
│ File size: ~180 KB (JSON)                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
        upload_pricing_to_artifact_registry()
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STORAGE: Artifact Registry (Generic Repository)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Repository: us-central1-generic.pkg.dev/                           │
│             weight-and-biases-476906/                               │
│             arr-coc-pricing                                         │
│                                                                     │
│ Package: gcp-pricing                                                │
│                                                                     │
│ Versions: (timestamp-based, immutable)                              │
│   1.0.20251116-143052  (180 KB)  ← Latest                          │
│   1.0.20251116-141052  (180 KB)                                    │
│   1.0.20251116-135052  (180 KB)                                    │
│   1.0.20251116-133052  (179 KB)                                    │
│   # ... historical versions preserved                               │
│                                                                     │
│ Retention: Indefinite (historical data preserved)                   │
│ Access: HTTP API + gcloud CLI                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
            Consumer code fetches
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ CONSUMPTION: Multiple Consumer Patterns                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Consumer 1: MECHA Battle Epic                                      │
│   File: training/cli/launch/mecha/mecha_battle_epic.py:258         │
│   Usage:                                                            │
│     from ...shared.artifact_pricing import get_spot_price          │
│     cpu_price = get_spot_price(pricing["c3_machines"]["us-west2"]["cpu_per_core_spot"])│
│     → Display: "⚡ C3 SPOT: $0.012/core/hr ⚡"                       │
│                                                                     │
│ Consumer 2: Launch Core (Build Cost Tracking)                      │
│   File: training/cli/shared/pricing/get_live_prices.py:10          │
│   Usage:                                                            │
│     price = get_live_price_for_launch("c3-standard-176", "us-west2")│
│     → Calculate: (176 cores × $0.012) + (704 GB × $0.002)          │
│     → Result: $2.11/hour                                            │
│     → Stored in campaign stats JSON                                │
│                                                                     │
│ Consumer 3: Generic Price Extraction                               │
│   File: training/cli/shared/artifact_pricing.py:229-397            │
│   Functions:                                                        │
│     • get_spot_price(sku_list) → Cheapest spot                     │
│     • get_standard_price(sku_list) → Cheapest on-demand            │
│     • get_commitment_1yr_price(sku_list) → 1-year commit           │
│     • get_commitment_3yr_price(sku_list) → 3-year commit           │
│     • all_prices(sku_list) → All tiers with metadata               │
│                                                                     │
│ Access Pattern (all consumers):                                    │
│   1. Call: fetch_pricing_no_save()                                 │
│      ├─ HTTP: Get latest version                                   │
│      ├─ gcloud: Download to temp                                   │
│      └─ Return: (pricing_data, version, size_kb)                   │
│   2. Extract relevant data structure                               │
│   3. Call price helper functions                                   │
│   4. Use price in calculations/display                             │
│                                                                     │
│ Note: NO local caching! Always fetch from Artifact Registry.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Flow 3: Quota Check Decision Tree

```
                    GPU QUOTA DECISION FLOW

User: python training/cli.py launch
                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Read config from .training file                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ WANDB_LAUNCH_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"                  │
│ WANDB_LAUNCH_ACCELERATOR_COUNT = "1"                               │
│ WANDB_LAUNCH_USE_PREEMPTIBLE = "true"                              │
│ GCP_PROJECT_ID = "weight-and-biases-476906"                        │
│ GCP_ROOT_RESOURCE_REGION = "us-central1"                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                ↓
        _auto_request_gpu_quota(config, region, status)
                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Map GPU type to quota metrics                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Compute Engine metric (for quick check):                           │
│   "NVIDIA_TESLA_T4" → "NVIDIA_T4_GPUS"                              │
│   + use_preemptible → "PREEMPTIBLE_NVIDIA_T4_GPUS"                 │
│                                                                     │
│ Vertex AI metric (for auto-request):                               │
│   "NVIDIA_TESLA_T4" → "nvidia_t4_gpus"                              │
│   → "aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus"│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Quick check Compute Engine quota                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ gcloud compute regions describe us-central1 --format=json          │
│                                                                     │
│ Response:                                                           │
│   {                                                                 │
│     "quotas": [                                                     │
│       {                                                             │
│         "metric": "PREEMPTIBLE_NVIDIA_T4_GPUS",                     │
│         "limit": 1.0,                                               │
│         "usage": 0.0                                                │
│       },                                                            │
│       ...                                                           │
│     ]                                                               │
│   }                                                                 │
│                                                                     │
│ Decision:                                                           │
│   if limit > 0:                                                     │
│       status("✓ T4 quota exists (limit: 1.0)")                     │
│       return "EXISTS" ✓                                             │
│   else:                                                             │
│       # Quota is 0 - continue to auto-request                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                ↓ (if limit == 0)
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Auto-request Vertex AI quota                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ status("⚠️ T4 quota is 0 or doesn't exist")                        │
│ status("Attempting automatic quota request...")                    │
│                                                                     │
│ gcloud alpha compute regions update-quota                          │
│     --project=weight-and-biases-476906                             │
│     --region=us-central1                                           │
│     --quota-metric=aiplatform.googleapis.com/                      │
│                    custom_model_training_nvidia_t4_gpus            │
│     --new-limit=1                                                  │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ SUCCESS                                                     │   │
│ ├─────────────────────────────────────────────────────────────┤   │
│ │                                                             │   │
│ │ status("✅ T4 QUOTA REQUEST SUBMITTED!")                   │   │
│ │ status("   GPU Type: NVIDIA_TESLA_T4")                     │   │
│ │ status("   Requested: 1 GPU(s)")                           │   │
│ │ status("   Region: us-central1")                           │   │
│ │ status("📧 Google will email when approved (1-2 days)")    │   │
│ │                                                             │   │
│ │ return "REQUESTED"  → HALT LAUNCH!                         │   │
│ │                                                             │   │
│ └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ FAILURE                                                     │   │
│ ├─────────────────────────────────────────────────────────────┤   │
│ │                                                             │   │
│ │ status("⚠️ Auto-request failed")                            │   │
│ │ status(f"Error: {stderr[:200]}")                           │   │
│ │ status("Proceeding to quota verification...")             │   │
│ │                                                             │   │
│ │ return "FAILED"  → Continue launch                         │   │
│ │                                                             │   │
│ └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAUNCH DECISION (training/cli/launch/core.py:816)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ quota_status = _auto_request_gpu_quota(...)                        │
│                                                                     │
│ if quota_status == "REQUESTED":                                    │
│     status("🛑 LAUNCH HALTED - awaiting quota approval")           │
│     status("Next steps:")                                          │
│     status("1. Wait 1-2 business days")                            │
│     status("2. Check email for approval")                          │
│     status("3. Run launch again")                                  │
│     return False  # ABORT! ✋                                       │
│                                                                     │
│ elif quota_status == "EXISTS":                                     │
│     # Proceed with launch ✓                                        │
│     [Continue to Step 2: Submit to W&B queue...]                   │
│                                                                     │
│ elif quota_status == "FAILED":                                     │
│     # Auto-request didn't work, but continue anyway                │
│     # User might have manually requested quota earlier             │
│     # If quota truly missing, Vertex AI will reject later          │
│     [Continue to Step 2: Submit to W&B queue...]                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                ↓ (if continuing)
┌─────────────────────────────────────────────────────────────────────┐
│ Vertex AI Job Submission (Eventually)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ If quota truly missing:                                             │
│   ERROR: grpc._channel._InactiveRpcError                           │
│   status = StatusCode.RESOURCE_EXHAUSTED                           │
│   details = "The following quota metrics exceed quota limits:      │
│              aiplatform.googleapis.com/                            │
│              custom_model_training_nvidia_t4_gpus"                 │
│                                                                     │
│ Wrapper catches error → Logs to Cloud Logging → Monitor extracts   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Pricing Data Structure (Complete Schema)

```json
{
  "updated": "2025-11-16T14:30:52Z",

  "c3_machines": {
    "us-central1": {
      "cpu_per_core_spot": [
        {
          "price": 0.01234,
          "description": "Compute optimized Core running in Americas",
          "sku_id": "XXXX-YYYY-ZZZZ",
          "usage_type": "Preemptible"
        }
      ],
      "ram_per_gb_spot": [
        {
          "price": 0.00165,
          "description": "Compute optimized Ram running in Americas",
          "sku_id": "AAAA-BBBB-CCCC",
          "usage_type": "Preemptible"
        }
      ]
    },
    "us-west2": { /* same structure */ },
    /* ... all GCP regions ... */
  },

  "e2_machines": {
    "us-central1": {
      "cpu_per_core_ondemand": [
        {
          "price": 0.0218,
          "description": "E2 Instance Core running in Americas",
          "sku_id": "...",
          "usage_type": "OnDemand"
        }
      ],
      "ram_per_gb_ondemand": [
        {
          "price": 0.00292,
          "description": "E2 Instance Ram running in Americas",
          "sku_id": "...",
          "usage_type": "OnDemand"
        }
      ]
    },
    /* ... all regions ... */
  },

  "gpus_spot": {
    "us-central1": [
      {
        "price": 0.14,
        "description": "Nvidia Tesla T4 GPU attached to Spot Preemptible VMs running in Americas",
        "sku_id": "...",
        "usage_type": "Spot"
      },
      {
        "price": 0.22,
        "description": "Nvidia L4 GPU attached to Spot Preemptible VMs running in Americas",
        "sku_id": "...",
        "usage_type": "Preemptible"
      }
      /* T4, L4, V100, P4, P100, A100, H100, H200 */
    ],
    /* ... all regions ... */
  },

  "gpus_ondemand": {
    "us-central1": [
      {
        "price": 0.35,
        "description": "Nvidia Tesla T4 GPU attached to VMs running in Americas",
        "sku_id": "...",
        "usage_type": "OnDemand"
      },
      {
        "price": 0.245,
        "description": "Commitment v1: Nvidia Tesla T4 GPU attached to VMs running in Americas for 1 Year",
        "sku_id": "...",
        "usage_type": "Commit"
      },
      {
        "price": 0.175,
        "description": "Commitment v1: Nvidia Tesla T4 GPU attached to VMs running in Americas for 3 Year",
        "sku_id": "...",
        "usage_type": "Commit"
      }
      /* OnDemand + 1-Year + 3-Year for all GPU types */
    ],
    /* ... all regions ... */
  }
}
```

**Key Properties**:
1. **Sorted**: All SKU lists sorted by price (cheapest first)
2. **Complete**: All pricing tiers included (spot, on-demand, 1yr, 3yr)
3. **Metadata**: Full SKU data preserved (description, ID, usage type)
4. **Immutable**: Each version timestamped, never modified

### Config Data Structure (.training file)

```bash
# GPU Configuration
WANDB_LAUNCH_ACCELERATOR_TYPE=NVIDIA_TESLA_T4
WANDB_LAUNCH_ACCELERATOR_COUNT=1
WANDB_LAUNCH_USE_PREEMPTIBLE=true

# Project Configuration
GCP_PROJECT_ID=weight-and-biases-476906
GCP_ROOT_RESOURCE_REGION=us-central1

# Machine Configuration
WANDB_LAUNCH_MACHINE_TYPE=n1-standard-4
```

---

## Configuration & Constants

### Pricing Configuration

**File**: `training/cli/shared/pricing_config.py`

| Constant | Value | Purpose |
|----------|-------|---------|
| `PROJECT_ID` | `"weight-and-biases-476906"` | GCP project (from .training) |
| `REGION` | `"us-central1"` | Default region (from .training) |
| `FUNCTION_NAME` | `"arr-coc-pricing-runner"` | Cloud Function name |
| `SCHEDULER_JOB` | `"arr-coc-pricing-scheduler"` | Scheduler job name |
| `SCHEDULER_INTERVAL_MINUTES` | `20` | Update frequency (every 20 min) |
| `REPOSITORY` | `"arr-coc-pricing"` | Artifact Registry repo |
| `PACKAGE` | `"gcp-pricing"` | Package name in repo |

### GPU Quota Maps

**File**: `training/cli/launch/core.py:3937-3945, 3996-4004`

**Vertex AI Quota Metrics** (Auto-Request):
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
# → "aiplatform.googleapis.com/custom_model_training_{suffix}"
```

**Compute Engine Quota Metrics** (Quick Check):
```python
quota_map = {
    "NVIDIA_H200": "NVIDIA_H200_GPUS",
    "NVIDIA_H100": "NVIDIA_H100_GPUS",
    "NVIDIA_H100_80GB": "NVIDIA_H100_80GB_GPUS",
    "NVIDIA_A100_80GB": "NVIDIA_A100_80GB_GPUS",
    "NVIDIA_TESLA_A100": "NVIDIA_A100_GPUS",
    "NVIDIA_TESLA_T4": "NVIDIA_T4_GPUS",
    "NVIDIA_L4": "NVIDIA_L4_GPUS",
}
# + "PREEMPTIBLE_" prefix if use_spot == true
```

---

## Error Handling Patterns

### Retry Logic (Shared Pattern)

**File**: `training/cli/shared/retry.py`

```python
# Retry delays (fuck it, restart pattern)
RETRY_DELAYS = [0, 1, 4, 8]  # 0s, 1s, 4s, 8s
MAX_ATTEMPTS = 4

def retry_with_backoff(operation_func, max_attempts=4, operation_name="operation"):
    """
    Retry with fixed backoff: 0s, 1s, 4s, 8s (cloud-optimized).

    Args:
        operation_func: Function returning (success: bool, error_msg: str)
        max_attempts: Max retry attempts
        operation_name: Name for logging

    Returns:
        (success: bool, error_msg: str or None)
    """
    for attempt in range(1, max_attempts + 1):
        success, error_msg = operation_func()

        if success:
            return (True, None)

        if attempt < max_attempts:
            delay = RETRY_DELAYS[attempt]
            time.sleep(delay)
        else:
            return (False, error_msg)

    return (False, "Max retries exceeded")
```

**Used By**:
- Pricing setup: `grant_actAs_permission()` (4 attempts)
- Pricing setup: `deploy_cloud_function()` (4 attempts)
- Pricing teardown: `delete_scheduler()` (4 attempts)
- Pricing teardown: `delete_cloud_function()` (4 attempts)

### Idempotency Patterns

**Pattern 1: Check Before Create**
```python
# Check if resource exists
result = subprocess.run(["gcloud", "...", "describe", resource_name], ...)
if result.returncode == 0:
    return  # Already exists, skip creation (idempotent)

# Create resource
subprocess.run(["gcloud", "...", "create", resource_name], ...)
```

**Pattern 2: Ignore "Already Exists" Errors**
```python
result = subprocess.run(["gcloud", "...", "create", resource_name], ...)

if result.returncode == 0:
    return (True, None)

# Check if error is "already exists"
if "already has" in result.stderr.lower() or "already exists" in result.stderr.lower():
    return (True, None)  # Idempotent success

return (False, error_msg)  # Real error
```

**Pattern 3: Ignore "Not Found" Errors**
```python
result = subprocess.run(["gcloud", "...", "delete", resource_name], ...)

if result.returncode == 0:
    return (True, None)

# Check if error is "not found"
stderr_lower = result.stderr.lower()
if "not found" in stderr_lower or "not_found" in stderr_lower:
    return (True, None)  # Already deleted, idempotent success

return (False, error_msg)  # Real error
```

---

## Quick Reference Tables

### File Locations Quick Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **GPU Quotas** |
| Auto-request function | `training/cli/launch/core.py` | 3913-4083 | Check/request GPU quotas |
| Launch integration | `training/cli/launch/core.py` | 815-830 | Use quota check results |
| Manual instructions (unused) | `training/cli/shared/gpu_quota_instruct.py` | 97-153 | Show manual request steps |
| **GPU Pricing** |
| Pricing config | `training/cli/shared/pricing_config.py` | 1-78 | Constants & schema |
| Setup infrastructure | `training/cli/setup/pricing_setup.py` | 96-1050 | Create pricing system |
| Teardown infrastructure | `training/cli/teardown/pricing_teardown.py` | 95-284 | Delete pricing system |
| Artifact Registry ops | `training/cli/shared/artifact_pricing.py` | 1-398 | Fetch/upload pricing |
| Live price calculator | `training/cli/shared/pricing/get_live_prices.py` | 10-58 | Calculate build costs |
| MECHA pricing display | `training/cli/launch/mecha/mecha_battle_epic.py` | 258, 288 | Battle animations |
| Cloud Function | `training/cli/shared/pricing/cloud_function/main.py` | 1-600+ | Fetch pricing (deployed) |

### Function Call Chains

**Launch → GPU Quota Check**:
```
launch_training_job()  (core.py:750)
└─→ _auto_request_gpu_quota()  (core.py:3913)
    ├─→ gcloud compute regions describe  (check CE quota)
    └─→ gcloud alpha compute regions update-quota  (request VA quota)
```

**Setup → Pricing Infrastructure**:
```
setup_pricing_infrastructure()  (pricing_setup.py:96)
├─→ create_pricing_repository()  (pricing_setup.py:125)
├─→ grant_actAs_permission()  (pricing_setup.py:599)
│   └─→ retry_with_backoff()  (retry.py)
├─→ deploy_cloud_function()  (pricing_setup.py:703)
│   └─→ retry_with_backoff()  (retry.py)
├─→ bootstrap_pricing()  (pricing_setup.py:167)
│   ├─→ fetch_pricing_no_save()  (artifact_pricing.py:96)
│   │   ├─→ _get_latest_version()  (artifact_pricing.py:60)
│   │   └─→ gcloud artifacts generic download
│   ├─→ get_pricing_age_minutes()  (artifact_pricing.py:199)
│   ├─→ _fetch_pricing_inline()  (pricing_setup.py:317)
│   │   └─→ Query GCP Billing API (~30K SKUs)
│   ├─→ upload_pricing_to_artifact_registry()  (artifact_pricing.py:160)
│   │   └─→ gcloud artifacts generic upload
│   └─→ _trigger_and_verify_function()  (pricing_setup.py:527)
└─→ create_scheduler()  (pricing_setup.py:883)
```

**Launch → Pricing Usage**:
```
_handle_training_image()  (core.py:808)
└─→ get_live_price_for_launch()  (get_live_prices.py:10)
    ├─→ fetch_pricing_no_save()  (artifact_pricing.py:96)
    ├─→ get_spot_price()  (artifact_pricing.py:229)
    └─→ get_standard_price()  (artifact_pricing.py:251)
```

### GCloud Commands Used

**GPU Quotas**:
```bash
# Check Compute Engine quota (quick approximation)
gcloud compute regions describe us-central1 --format=json

# Auto-request Vertex AI quota
gcloud alpha compute regions update-quota \
    --project=PROJECT \
    --region=us-central1 \
    --quota-metric=aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus \
    --new-limit=1
```

**Pricing Infrastructure**:
```bash
# Create repository
gcloud artifacts repositories create arr-coc-pricing \
    --repository-format=generic \
    --location=us-central1

# Deploy Cloud Function
gcloud functions deploy arr-coc-pricing-runner \
    --gen2 \
    --region=us-central1 \
    --runtime=python312 \
    --entry-point=fetch_pricing \
    --trigger-http \
    --timeout=540s \
    --memory=512MB

# Create scheduler
gcloud scheduler jobs create http arr-coc-pricing-scheduler \
    --location=us-central1 \
    --schedule="*/20 * * * *" \
    --uri=FUNCTION_URL \
    --oidc-service-account-email=SA_EMAIL

# Upload pricing
gcloud artifacts generic upload \
    --package=gcp-pricing \
    --version=1.0.20251116-143052 \
    --source=pricing.json

# Download pricing
gcloud artifacts generic download \
    --package=gcp-pricing \
    --version=1.0.20251116-143052 \
    --destination=/tmp
```

### API Endpoints Used

**Vertex AI Quotas**:
- Not directly accessed (uses `gcloud alpha compute regions update-quota`)

**Artifact Registry**:
```
GET https://artifactregistry.googleapis.com/v1/
    projects/{PROJECT}/locations/{LOCATION}/repositories/{REPO}/
    packages/{PACKAGE}/versions
    ?pageSize=1&orderBy=createTime desc

→ Returns latest pricing version
```

**GCP Cloud Billing**:
```
GET https://cloudbilling.googleapis.com/v1/
    services/6F81-5844-456A/skus
    ?pageSize=500

→ Returns ~30,000 pricing SKUs (paginated)
```

---

## 🔍 Where Do I Find...?

**Q: Where is GPU quota checking logic?**
→ `training/cli/launch/core.py:3913` (`_auto_request_gpu_quota()`)

**Q: Where does launch use quota check results?**
→ `training/cli/launch/core.py:815-830`

**Q: Where is pricing infrastructure setup?**
→ `training/cli/setup/pricing_setup.py:96` (`setup_pricing_infrastructure()`)

**Q: Where is pricing infrastructure teardown?**
→ `training/cli/teardown/pricing_teardown.py:95` (`teardown_pricing_infrastructure()`)

**Q: Where does MECHA get spot prices?**
→ `training/cli/launch/mecha/mecha_battle_epic.py:258, 288` (uses `get_spot_price()`)

**Q: Where are pricing packages stored?**
→ Artifact Registry: `us-central1-generic.pkg.dev/.../arr-coc-pricing/gcp-pricing`

**Q: Where is the Cloud Function code?**
→ `training/cli/shared/pricing/cloud_function/main.py`

**Q: Where is live price calculation?**
→ `training/cli/shared/pricing/get_live_prices.py:10` (`get_live_price_for_launch()`)

**Q: Where are pricing helper functions?**
→ `training/cli/shared/artifact_pricing.py:229-397`

**Q: Where is pricing schema defined?**
→ `training/cli/shared/pricing_config.py:63-69` (`PRICING_SCHEMA`)

**Q: Where is retry logic?**
→ `training/cli/shared/retry.py` (shared by setup/teardown)

**Q: Is there any user-facing pricing?**
→ **NO** - All user-facing pricing displays deleted 2025-11-16

**Q: What happened to the old GPU quota verification?**
→ **DELETED** - `_verify_gpu_quota()` removed 2025-11-16 (checked wrong quotas)

**Q: What happened to check_and_update_pricing()?**
→ **DELETED** - `mecha_battle_epic.py:275-359` removed 2025-11-16 (manual trigger logic, 24-hour staleness checks)
→ **Replaced with**: THE GOOD PRICING WAY - `fetch_pricing_no_save()` only (Cloud Scheduler handles refresh)

---

## Summary

This system map provides complete technical details for:

1. **GPU Quota System**: Infrastructure validation before Vertex AI launch
2. **GPU Pricing System**: Cost tracking with Cloud Function + Artifact Registry
3. **Complete Code Flows**: Step-by-step execution with line numbers
4. **Data Structures**: Pricing JSON schema, config files
5. **Error Handling**: Retry patterns, idempotency
6. **Quick Reference**: File locations, function chains, gcloud commands

**Key Insights**:
- Two **separate** quota namespaces (Compute Engine vs Vertex AI)
- Pricing stored in Artifact Registry (no local caching)
- Auto-request uses **correct** Vertex AI quotas
- Old verification deleted (checked wrong quotas)
- Infrastructure tracking only (no user-facing cost estimates)

---

**Last Updated**: 2025-11-16
**Version**: 2.0 - Complete code flow analysis
**Prepared by**: THE-PATTERN-PERFECTIONIST 🎯
