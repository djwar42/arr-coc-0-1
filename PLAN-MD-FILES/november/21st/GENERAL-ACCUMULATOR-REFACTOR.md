# General Accumulator Refactor Plan

**Date**: 2025-11-21
**Goal**: Refactor `infra_verify.py` to use ONE GeneralAccumulator with nested accumulators for MAXIMUM PARALLELISM
**Status**: Planning phase - ready for next session
**Expected Speedup**: 2.8× faster (7.0s → 2.5s)

---

## 🎯 Current Problem

`infra_verify.py` runs checks in **sequential groups**:

```python
# Sequential execution (7.0s total)
1. GCP checks (2.5s) - parallel within group
   ↓
2. W&B checks (1.8s) - SEQUENTIAL!
   ↓
3. Worker pools (1.2s) - parallel within group
   ↓
4. Quotas (1.5s) - SEQUENTIAL!

Total: 2.5 + 1.8 + 1.2 + 1.5 = 7.0s
```

---

## 🚀 Solution: ONE GeneralAccumulator

Run **EVERYTHING in parallel** with nested accumulators:

```python
# Parallel execution (2.5s total!)
main_acc = GeneralAccumulator(max_workers=50)

# NEST GCloudAccumulator for GCP checks
gcp_acc = GCloudAccumulator(max_workers=20)
gcp_acc.start("buckets", [...])
gcp_acc.start("registry", [...])
# ... all GCP checks
main_acc.add_accumulator("gcp", gcp_acc)

# NEST GCloudAccumulator for worker pools
pool_acc = GCloudAccumulator(max_workers=10)
pool_acc.start("us-west2", [...])
main_acc.add_accumulator("worker_pools", pool_acc)

# W&B checks as LAMBDAS (parallel!)
main_acc.start("wandb_queue", lambda: check_wandb_queue())
main_acc.start("wandb_project", lambda: check_wandb_project())

# HF checks as LAMBDAS
main_acc.start("hf_repo", lambda: check_hf_repo())

# GPU/C3 quotas as LAMBDAS
main_acc.start("gpu_quotas", lambda: check_gpu_quotas())
main_acc.start("c3_quotas", lambda: check_c3_quotas())

# Get EVERYTHING at once!
all_results = main_acc.get_all()

Total: max(2.5, 1.8, 1.2, 1.5) = 2.5s
Speedup: 7.0 / 2.5 = 2.8×
```

---

## 📋 Implementation Steps

### Step 1: Import GeneralAccumulator ✓
```python
from .api_helpers import GCloudAccumulator, GeneralAccumulator
```
**Already done!**

### Step 2: Create Main Accumulator

```python
# After billing check, if billing enabled:
main_acc = GeneralAccumulator(max_workers=50)
```

### Step 3: Nest GCP GCloudAccumulator

```python
# Create GCP accumulator (keep existing code!)
gcp_acc = GCloudAccumulator(max_workers=20)
gcp_acc.start("buckets", [...])  # All existing GCP checks
gcp_acc.start("registry", [...])
gcp_acc.start("apis", [...])
# ... all others

# Add to main accumulator
main_acc.add_accumulator("gcp", gcp_acc)
```

### Step 4: Create Lambda Wrappers for W&B

```python
def check_wandb_queue():
    """Check W&B queue (parallel-safe wrapper)"""
    try:
        import wandb
        api = wandb.Api()
        queue = api.run_queue(entity, queue_name)
        return {"exists": queue is not None, "name": queue_name}
    except Exception:
        return {"exists": False, "name": queue_name}

def check_wandb_project():
    """Check W&B project (parallel-safe wrapper)"""
    try:
        import wandb
        api = wandb.Api()
        proj = api.project(f"{entity}/{project_name}")
        return {"exists": proj is not None, "name": project_name}
    except Exception:
        return {"exists": False, "name": project_name}

# Add to main accumulator
main_acc.start("wandb_queue", check_wandb_queue)
main_acc.start("wandb_project", check_wandb_project)
```

### Step 5: Create Lambda Wrapper for HF

```python
def check_hf_repo():
    """Check HuggingFace repo (parallel-safe wrapper)"""
    if not hf_repo:
        return {"exists": False, "id": ""}

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.repo_info(repo_id=hf_repo, repo_type="model")
        return {"exists": True, "id": hf_repo}
    except Exception:
        return {"exists": False, "id": hf_repo}

# Add to main accumulator
main_acc.start("hf_repo", check_hf_repo)
```

### Step 6: Nest Worker Pools GCloudAccumulator

```python
# Create pool accumulator (keep existing code!)
pool_acc = GCloudAccumulator(max_workers=10)
for region in mecha_regions:
    pool_acc.start(region, [...])

# Add to main accumulator
main_acc.add_accumulator("worker_pools", pool_acc)
```

### Step 7: Create Lambda Wrappers for Quotas

```python
def check_gpu_quotas():
    """Check GPU quotas (parallel-safe wrapper)"""
    from .quota import get_all_vertex_gpu_quotas

    vertex_gpu = {}
    for region in gpu_check_regions:
        try:
            quotas = get_all_vertex_gpu_quotas(project_id, region)
            granted = [q for q in quotas if q.get("quota_limit", 0) > 0]
            pending = [q for q in quotas if q.get("quota_limit", 0) == 0]
            vertex_gpu[region] = {"granted": granted, "pending": pending}
        except Exception:
            vertex_gpu[region] = {"granted": [], "pending": []}

    return vertex_gpu

def check_c3_quotas():
    """Check C3 quotas (parallel-safe wrapper)"""
    from .quota import get_cloud_build_c3_region_quota

    c3_build = {}
    for region in c3_check_regions:
        try:
            quota_info = get_cloud_build_c3_region_quota(project_id, region)
            c3_build[region] = quota_info
        except Exception:
            c3_build[region] = None

    return c3_build

# Add to main accumulator
main_acc.start("gpu_quotas", check_gpu_quotas)
main_acc.start("c3_quotas", check_c3_quotas)
```

### Step 8: Get All Results (ONE CALL!)

```python
# Get EVERYTHING in parallel!
all_results = main_acc.get_all()
main_acc.shutdown()

# Extract nested results
gcp_results = all_results.get("gcp", {})
worker_pool_results = all_results.get("worker_pools", {})
wandb_queue_result = all_results.get("wandb_queue", {})
wandb_project_result = all_results.get("wandb_project", {})
hf_result = all_results.get("hf_repo", {})
gpu_quotas = all_results.get("gpu_quotas", {})
c3_quotas = all_results.get("c3_quotas", {})
```

### Step 9: Process Results (Same Order as Before!)

```python
# Process GCP results (same code as before)
_status("Checking GCP resources...")
buckets_result = gcp_results.get("buckets")
# ... existing parsing code

# Process W&B results (now from accumulator!)
_status("Checking W&B resources...")
info["wandb"]["queue"] = wandb_queue_result
info["wandb"]["project"] = wandb_project_result
_status(f"  {'✓' if wandb_queue_result['exists'] else '✗'} Queue: {wandb_queue_result['name']}")

# Process HF results (now from accumulator!)
_status("Checking HuggingFace repo...")
info["hf"]["repo"] = hf_result
# ... etc

# Process worker pools (same code)
# Process quotas (now from accumulator!)
```

---

## 🎨 Benefits

✅ **2.8× faster** (7.0s → 2.5s)
✅ **Cleaner code** (one accumulator manages everything)
✅ **Same output** (status messages in same order)
✅ **Type-safe** (GeneralAccumulator already tested)
✅ **Composable** (easy to add more checks)

---

## 📊 Performance Comparison

### Before (Sequential Groups)
```
Billing check.............. 0.5s (sequential)
GCP checks (parallel)...... 2.5s (wait)
W&B checks (sequential).... 1.8s (wait)
Worker pools (parallel).... 1.2s (wait)
Quotas (sequential)........ 1.5s (wait)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total.................... 7.5s
```

### After (Full Parallelism)
```
Billing check.............. 0.5s (sequential - must be first)
Everything else............ 2.5s (ALL IN PARALLEL!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total.................... 3.0s

Speedup: 2.5×
```

---

## 🧪 Testing Checklist

- [ ] All GCP checks still work
- [ ] W&B checks run in parallel
- [ ] HF checks run in parallel
- [ ] Worker pool checks still work
- [ ] GPU quota checks run in parallel
- [ ] C3 quota checks run in parallel
- [ ] Status messages appear in correct order
- [ ] Error handling still works (billing disabled, API failures)
- [ ] Results dict structure unchanged
- [ ] No race conditions

---

## 📝 Notes

- **Key insight**: GeneralAccumulator supports nesting other accumulators!
- **Lambda wrappers**: Ensure W&B/HF checks are thread-safe
- **Result extraction**: Nested results need careful unpacking
- **Status order**: Process results in same order for familiar UX
- **Error handling**: Each lambda must handle its own exceptions

---

**END OF PLAN** - Ready for implementation! ⚡
