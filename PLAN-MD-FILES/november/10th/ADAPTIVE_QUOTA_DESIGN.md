# Adaptive Quota Detection - Design Document

## 🎯 Goal

Dynamically sense BOTH C3 quotas and automatically select the best machine that fits.

## Current Problem

Code only checks Compute Engine C3_CPUS quota (line 259).
Misses Cloud Build C3 quota (the actual limiting factor!).

## Solution Architecture

### 1. Add Cloud Build Quota Checker

```python
def check_cloud_build_c3_quota(project_id: str, region: str) -> tuple[float, float]:
    """
    Check Cloud Build C3 CPUs quota for private pools.

    Uses: gcloud builds quotas list (or parse from IAM quotas)

    Returns:
        (limit, usage) - Cloud Build C3 quota, or (0.0, 0.0) if not found
    """
    # Implementation similar to check_c3_cpu_quota()
    # Parse Cloud Build API quotas
```

### 2. Modify get_best_available_c3_machine()

```python
def get_best_available_c3_machine(
    project_id: str, region: str
) -> tuple[str, int, float, float, str]:
    """
    Returns:
        (machine, vcpus, compute_quota, cloud_build_quota, limiting_factor)
    """
    # Check BOTH quotas
    ce_limit, ce_usage = check_c3_cpu_quota(project_id, region)
    cb_limit, cb_usage = check_cloud_build_c3_quota(project_id, region)

    # Use the MINIMUM as limiting factor
    effective_limit = min(ce_limit, cb_limit) if cb_limit > 0 else ce_limit
    limiting_factor = "cloud_build" if cb_limit < ce_limit else "compute_engine"

    # Select best machine that fits
    available = get_available_c3_machines(effective_limit)

    if available:
        machine, vcpus = available[0]
        return (machine, vcpus, ce_limit, cb_limit, limiting_factor)
    else:
        return ("c3-highcpu-4", 4, ce_limit, cb_limit, limiting_factor)
```

### 3. Update Quota Advice Logic

Show upgrade instructions for BOTH quotas if either is limiting:

```python
if ce_limit < 176 or cb_limit < 176:
    status("⚠️ Quota upgrade available!")

    if ce_limit < 176:
        status("  Compute Engine C3_CPUS: {ce_limit} → 176")

    if cb_limit < 176 or cb_limit == 0:
        status("  Cloud Build C3 CPUs: {cb_limit} → 176")

    # Show unified instructions for BOTH
```

## Implementation Steps

1. Add `check_cloud_build_c3_quota()` function
2. Modify `get_best_available_c3_machine()` to return both quotas
3. Update worker pool creation to use both quotas
4. Update all 3 upgrade advice sections (lines 748, 797, 837)
5. Test with current quota (4) - should select c3-highcpu-4
6. Test after one quota increases - should use MIN of both

## Expected Behavior

### Scenario 1: Current State (CE=144, CB=4)
```
Checking quotas...
  Compute Engine C3_CPUS: 144 (plenty)
  Cloud Build C3 CPUs:    4 (LIMITING!)

Selected: c3-highcpu-4 (4 vCPUs)
Build time: ~2-3 hours

💡 UPGRADE AVAILABLE:
  Cloud Build C3 CPUs: 4 → 176 (CRITICAL!)
  Compute Engine C3_CPUS: 144 → 176 (recommended)

  After upgrade: c3-highcpu-176 available (~20 min builds)
```

### Scenario 2: After Cloud Build Increase (CE=144, CB=176)
```
Checking quotas...
  Compute Engine C3_CPUS: 144 (LIMITING!)
  Cloud Build C3 CPUs:    176 (plenty)

Selected: c3-highcpu-88 (88 vCPUs) - best fit for 144 quota
Build time: ~25 minutes

💡 UPGRADE AVAILABLE:
  Compute Engine C3_CPUS: 144 → 176

  After upgrade: c3-highcpu-176 available (~20 min builds)
```

### Scenario 3: Both at Max (CE=176, CB=176)
```
Checking quotas...
  Compute Engine C3_CPUS: 176 ✅
  Cloud Build C3 CPUs:    176 ✅

Selected: c3-highcpu-176 (176 vCPUs) - MAXIMUM!
Build time: ~20 minutes

✓ Using maximum C3 configuration!
```

## Benefits

1. **Works immediately** - Uses c3-highcpu-4 with current quota (4)
2. **Adaptive** - Automatically upgrades when quota increases
3. **Clear guidance** - Shows which quota is limiting
4. **Smart fallback** - Uses best available, never fails
5. **Future-proof** - Handles any quota configuration

## Files to Modify

- `training/cli/setup/core.py` (lines 175-268, 748-869)
- Add unit tests for quota detection
- Update documentation
