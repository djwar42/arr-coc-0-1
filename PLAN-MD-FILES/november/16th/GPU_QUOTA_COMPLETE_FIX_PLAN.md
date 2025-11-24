# GPU Quota Fix Plan

**Date**: 2025-11-16
**Status**: 2 Phases Remaining
**Goal**: Fix GPU quota display and instructions to show ONLY Vertex AI quotas

---

## The Problem

The CLI shows **Compute Engine GPU quotas**, but Vertex AI Custom Training uses **completely different quotas**.

**Current Situation**:
- CE quota: `PREEMPTIBLE_NVIDIA_T4_GPUS` = 1.0 ✅
- VA quota: `custom_model_training_nvidia_t4_gpus` = 0 ❌
- Infra screen shows quota = 1.0 (CE quota - WRONG!)
- Vertex AI rejects job (VA quota = 0) ❌

**Architecture Note**:
- GPU quota **auto-request** was **intentionally removed** from `launch/core.py` (2025-11-16)
- Previous auto-request checked WRONG quota system (Compute Engine instead of Vertex AI)
- Current design: Users request VA quota manually via infra screen instructions
- If VA quota insufficient → Vertex AI rejects with clear error message
- No automatic quota requests happen during launch (by design)

**The Fix**:
1. Show ONLY Vertex AI quotas in infra screen
2. Show ONLY Vertex AI metric names in instructions
3. Users request correct quota type

---

## ✅ IMPORTANT: Pricing Cloud Function is CORRECT!

**Unlike quotas, GPU pricing is the same for both Compute Engine and Vertex AI Custom Training.**

**Why pricing works (but quotas don't):**

| Aspect | Compute Engine | Vertex AI Custom Training |
|--------|----------------|---------------------------|
| **Quotas** | `NVIDIA_T4_GPUS` | `custom_model_training_nvidia_t4_gpus` ❌ **DIFFERENT!** |
| **Pricing** | $0.35/hr (T4) | $0.35/hr (T4) ✅ **SAME!** |
| **Hardware** | CE VM + GPU | CE VM + GPU (created by Vertex) |

**What the pricing cloud function does** (`training/cli/shared/pricing/cloud_function/main.py`):
- Queries **Cloud Billing API** (service ID `6F81-5844-456A` = Compute Engine)
- Fetches pricing for **ALL GPU types**: T4, L4, V100, P4, P100, A100, H100, H200
- Returns **ALL pricing tiers**: Spot, On-Demand, 1-year commitment, 3-year commitment
- Stores in Artifact Registry for CLI/TUI consumption

**Key insight**: Vertex AI Custom Training uses Compute Engine VMs under the hood! When you launch a training job:
1. Vertex AI creates a CE VM with your requested GPU
2. Runs your training container
3. Tears down the VM when done

**Same GPU hardware → Same underlying price!**

**The ONLY bug is quotas** (showing CE instead of VA). Pricing is correct as-is. No changes needed to pricing cloud function!

---

## The Two Quota Systems

### ❌ Compute Engine Quotas (What We Currently Show - WRONG!)
```bash
Metric: NVIDIA_T4_GPUS
Metric: PREEMPTIBLE_NVIDIA_T4_GPUS
Command: gcloud compute regions describe us-central1
Service: compute.googleapis.com
```

### ✅ Vertex AI Custom Training Quotas (What We NEED - CORRECT!)
```bash
Metric: aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus
Metric: aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus
Command: gcloud alpha quotas list --service=aiplatform.googleapis.com
Service: aiplatform.googleapis.com
```

**THESE ARE COMPLETELY SEPARATE!** You can have CE quota = 1, VA quota = 0.

**Reference**: See `GPU_QUOTA_METRICS_REFERENCE.md` for complete list of all GPU quota metrics.

---

## Solution: Consolidate All Quota Logic Into Canonical Quota Module

**Instead of fixing code scattered across files, we'll create a canonical quota module** (like we did with `pricing/`):

```
training/cli/shared/quota/
├── __init__.py       # Re-exports all functions
├── c3_quota.py       # Cloud Build C3 quota checking (already exists as quota_checker.py)
└── gpu_quota.py      # Vertex AI GPU quota checking (NEW - contains all VA quota logic)
```

**Then all other parts use these canonical functions:**
- `setup/core.py` - Uses `quota.get_all_gpu_quotas()` instead of inline gcloud calls
- `machine_selection.py` - Currently uses `quota_checker.get_cloud_build_c3_region_quota()` → will update to use `quota.get_cloud_build_c3_region_quota()`

**Benefits:**
- ✅ Single source of truth for quota checking
- ✅ DRY - no duplicated quota logic
- ✅ Easy to test quota module independently
- ✅ Parallel structure to `pricing/` module

---

## Quota Module Implementation

### File 1: `quota/c3_quota.py` (Cloud Build Quotas)

**This file already exists as `quota_checker.py` (123 lines) - just move it:**

```python
"""
Cloud Build C3 Quota Checking

Centralized quota checking for Cloud Build worker pools.
Returns confirmed Cloud Build C3 quotas for regions with approved custom quotas.
"""

import subprocess
import json
from typing import Dict


def get_cloud_build_c3_quotas(project_id: str) -> Dict[str, int]:
    """
    Get Cloud Build C3 quota for all regions

    Args:
        project_id: GCP project ID

    Returns:
        Dict mapping region → vCPUs quota
        Example: {"us-central1": 176, "asia-northeast1": 176}
    """
    quotas = _fetch_c3_quotas_from_api(project_id)
    return quotas.copy()


def get_cloud_build_c3_region_quota(project_id: str, region: str) -> int:
    """
    Get Cloud Build C3 quota for single region

    Args:
        project_id: GCP project ID
        region: GCP region (e.g., "us-central1")

    Returns:
        vCPUs quota for that region (0 if no quota)
    """
    quotas = get_cloud_build_c3_quotas(project_id)
    return quotas.get(region, 0)


def has_sufficient_quota(project_id: str, region: str, required_vcpus: int) -> bool:
    """
    Check if region has sufficient Cloud Build C3 quota

    Args:
        project_id: GCP project ID
        region: GCP region
        required_vcpus: Minimum vCPUs needed (e.g., 176)

    Returns:
        True if quota >= required_vcpus
    """
    quota = get_cloud_build_c3_region_quota(project_id, region)
    return quota >= required_vcpus


def _fetch_c3_quotas_from_api(project_id: str) -> Dict[str, int]:
    """
    Internal: Fetch quotas from gcloud API

    Returns only quotas with build_origin=default dimension
    """
    try:
        result = subprocess.run([
            'gcloud', 'alpha', 'services', 'quota', 'list',
            '--service=cloudbuild.googleapis.com',
            f'--consumer=projects/{project_id}',
            '--format=json'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
        quotas = {}

        for service in data:
            for limit in service.get('consumerQuotaLimits', []):
                metric = limit.get('metric', '')

                if 'concurrent_private_pool_c3_build_cpus' not in metric:
                    continue

                for bucket in limit.get('quotaBuckets', []):
                    dims = bucket.get('dimensions', {})

                    if 'build_origin' in dims and dims['build_origin'] == 'default':
                        if 'region' in dims:
                            region = dims['region']
                            effective_limit = bucket.get('effectiveLimit')

                            if effective_limit:
                                try:
                                    quotas[region] = int(effective_limit)
                                except (ValueError, TypeError):
                                    pass

        return quotas

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return {}
```

**Functions:**
- `get_cloud_build_c3_quotas(project_id)` - Get all regions with C3 quota
- `get_cloud_build_c3_region_quota(project_id, region)` - Get single region quota
- `has_sufficient_quota(project_id, region, vcpus)` - Boolean check
- `_fetch_c3_quotas_from_api(project_id)` - Internal API call

---

### File 2: `quota/gpu_quota.py` (Vertex AI GPU Quotas) - NEW!

**This file contains ALL the Vertex AI quota logic from Phase 1 below:**

```python
"""
Vertex AI GPU Quota Checking

Queries Vertex AI Custom Training quotas (NOT Compute Engine quotas!).
Different GPU types use different quota metrics.
"""

import subprocess
import json
from typing import Dict, List, Optional


# GPU type → Vertex AI quota metric mapping
GPU_QUOTA_METRICS = {
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


def get_gpu_quotas(project_id: str, region: str, gpu_type: str, use_spot: bool) -> int:
    """
    Get Vertex AI GPU quota for specific GPU type in region

    Args:
        project_id: GCP project ID
        region: GCP region (e.g., "us-central1")
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4")
        use_spot: True for spot/preemptible, False for on-demand

    Returns:
        GPU quota limit (0 if no quota)

    Example:
        >>> get_gpu_quotas("my-project", "us-west2", "NVIDIA_TESLA_T4", True)
        1  # 1x T4 spot GPU available
    """
    metric_name = get_gpu_quota_metric(gpu_type, use_spot)
    quota = _fetch_gpu_quota_from_api(project_id, region, metric_name)
    return quota


def get_all_gpu_quotas(project_id: str, region: str) -> List[Dict]:
    """
    Get ALL Vertex AI GPU quotas for a region (all GPU types, spot + on-demand)

    Returns list of quota info dicts sorted by quota limit (highest first).
    Used by infra screen to show all available GPUs.

    Args:
        project_id: GCP project ID
        region: GCP region

    Returns:
        List of dicts with keys: gpu_name, quota_limit, metric_name, is_spot
        Example: [
            {"gpu_name": "T4", "quota_limit": 1, "metric_name": "..._nvidia_t4_gpus", "is_spot": True},
            {"gpu_name": "L4", "quota_limit": 0, "metric_name": "..._nvidia_l4_gpus", "is_spot": False},
        ]
    """
    all_quotas = []

    # Query all GPU quotas from Vertex AI API
    try:
        result = subprocess.run([
            'gcloud', 'alpha', 'quotas', 'list',
            '--service=aiplatform.googleapis.com',
            f'--consumer=projects/{project_id}',
            f'--location={region}',
            '--format=json'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)

        # Filter for GPU quotas
        for quota in data:
            quota_id = quota.get('quotaId', '')
            metric = quota.get('metric', '')

            # Only custom_model_training GPU quotas
            if not quota_id.startswith('custom_model_training_'):
                continue
            if 'nvidia' not in quota_id.lower():
                continue

            # Get quota value
            quota_value_str = quota.get('quotaDetails', {}).get('value', '0')
            try:
                quota_limit = int(quota_value_str)
            except (ValueError, TypeError):
                quota_limit = 0

            # Determine GPU name and spot status
            is_spot = 'preemptible' in quota_id
            gpu_name = _extract_gpu_name(quota_id)

            all_quotas.append({
                'gpu_name': gpu_name,
                'quota_limit': quota_limit,
                'metric_name': metric,
                'is_spot': is_spot,
                'quota_id': quota_id,
            })

        # Sort by quota limit (highest first)
        all_quotas.sort(key=lambda x: x['quota_limit'], reverse=True)

        return all_quotas

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return []


def get_gpu_quota_metric(gpu_type: str, use_spot: bool) -> str:
    """
    Convert GPU type to Vertex AI quota metric name

    Args:
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4")
        use_spot: True for spot/preemptible

    Returns:
        Full metric name
        Example: "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus"
    """
    quota_suffix = GPU_QUOTA_METRICS.get(gpu_type, "nvidia_t4_gpus")

    if use_spot:
        quota_suffix = f"preemptible_{quota_suffix}"

    return f"aiplatform.googleapis.com/custom_model_training_{quota_suffix}"


def has_gpu_quota(project_id: str, region: str, gpu_type: str, use_spot: bool, required: int = 1) -> bool:
    """
    Check if region has sufficient GPU quota

    Args:
        project_id: GCP project ID
        region: GCP region
        gpu_type: GPU type
        use_spot: Spot or on-demand
        required: Minimum quota needed (default: 1)

    Returns:
        True if quota >= required
    """
    quota = get_gpu_quotas(project_id, region, gpu_type, use_spot)
    return quota >= required


def _fetch_gpu_quota_from_api(project_id: str, region: str, metric_name: str) -> int:
    """
    Internal: Fetch specific GPU quota from Vertex AI API

    Args:
        project_id: GCP project ID
        region: GCP region
        metric_name: Full metric name

    Returns:
        Quota value (0 if not found)
    """
    # Extract short metric name (without service prefix)
    short_metric = metric_name.replace("aiplatform.googleapis.com/", "")

    try:
        result = subprocess.run([
            'gcloud', 'alpha', 'quotas', 'describe',
            short_metric,
            '--service=aiplatform.googleapis.com',
            f'--consumer=projects/{project_id}',
            f'--location={region}',
            '--format=value(quotaDetails.value)'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return 0

        quota_str = result.stdout.strip()
        return int(quota_str) if quota_str else 0

    except (subprocess.TimeoutExpired, ValueError, TypeError, Exception):
        return 0


def _extract_gpu_name(quota_id: str) -> str:
    """
    Internal: Extract display name from quota ID

    Example:
        "custom_model_training_preemptible_nvidia_t4_gpus" → "T4"
        "custom_model_training_nvidia_h100_80gb_gpus" → "H100 80GB"
    """
    # Remove prefix
    name = quota_id.replace('custom_model_training_', '')
    name = name.replace('preemptible_', '')
    name = name.replace('nvidia_', '')
    name = name.replace('_gpus', '')

    # Map to display names
    name_map = {
        't4': 'T4',
        'l4': 'L4',
        'v100': 'V100',
        'p4': 'P4',
        'p100': 'P100',
        'a100': 'A100',
        'a100_80gb': 'A100 80GB',
        'h100': 'H100',
        'h100_80gb': 'H100 80GB',
        'h200': 'H200',
    }

    return name_map.get(name, name.upper())
```

**Functions:**
- `get_gpu_quotas(project_id, region, gpu_type, use_spot)` - Get single GPU quota
- `get_all_gpu_quotas(project_id, region)` - Get ALL GPU quotas (for infra screen)
- `get_gpu_quota_metric(gpu_type, use_spot)` - GPU type → metric name conversion
- `has_gpu_quota(project_id, region, gpu_type, use_spot, required)` - Boolean check
- `_fetch_gpu_quota_from_api(...)` - Internal API call
- `_extract_gpu_name(quota_id)` - Convert quota ID to display name

---

### File 3: `quota/__init__.py` (Re-exports)

```python
"""
Quota Module - Canonical quota checking for Cloud Build and Vertex AI

Import everything from here:
    from cli.shared.quota import get_cloud_build_c3_quotas, get_all_gpu_quotas, ...
"""

from .c3_quota import (
    get_cloud_build_c3_quotas,
    get_cloud_build_c3_region_quota,
    has_sufficient_quota,
)

from .gpu_quota import (
    get_gpu_quotas,
    get_all_gpu_quotas,
    get_gpu_quota_metric,
    has_gpu_quota,
)

__all__ = [
    # C3 quotas
    'get_cloud_build_c3_quotas',
    'get_cloud_build_c3_region_quota',
    'has_sufficient_quota',

    # GPU quotas
    'get_gpu_quotas',
    'get_all_gpu_quotas',
    'get_gpu_quota_metric',
    'has_gpu_quota',
]
```

---

## Files That Use The Quota Module

| File | What It Does | Which Quota Functions It Uses |
|------|--------------|-------------------------------|
| `setup/core.py` | Infrastructure screen | `get_all_gpu_quotas()` - show all GPUs |
| `setup/core.py` | Console quota request URLs | `get_gpu_quota_metric()` - show correct metric |
| `machine_selection.py` | Cloud Build machine selection | `get_cloud_build_c3_region_quota()` - select best C3 machine |

**Note**: `launch/core.py` does NOT use quota module - GPU quota auto-request was removed by design (2025-11-16)

---

## Phase 1: Update Infra Screen to Use `quota.get_all_gpu_quotas()`

**Goal**: Replace inline Compute Engine quota checking with canonical `quota.get_all_gpu_quotas()` function.

**File**: `training/cli/setup/core.py`
**Function**: `_get_all_gpu_quotas()` (lines 1268-1334)
**What changes**: Replace gcloud CE quota calls with `quota.get_all_gpu_quotas()` import

### Current Code (WRONG - Shows CE Quotas)
```python
# Line 1280-1293
gcloud compute regions describe us-central1  # ❌ CE quotas!

# Line 1302-1304 - COMMENT ALREADY SAYS IT'S WRONG!
# NOTE: These are COMPUTE ENGINE quotas for display purposes only
# Vertex AI Custom Training uses DIFFERENT quotas
```

### New Code (CORRECT - Use Quota Module)

**Simply import and use the quota module:**

```python
from cli.shared.quota import get_all_gpu_quotas

def _get_all_gpu_quotas(project_id: str, region: str) -> list:
    """Get all Vertex AI GPU quotas for region"""
    return get_all_gpu_quotas(project_id, region)
```

**That's it!** The `quota/gpu_quota.py` module (code provided above) handles all the logic:
- Queries Vertex AI API (not Compute Engine!)
- Parses JSON response structure
- Filters for `custom_model_training_*nvidia_*_gpus` quotas
- Converts quota IDs to display names (T4, L4, H100, etc.)
- Sorts by quota limit (highest first)
- Returns list of dicts with `gpu_name`, `quota_limit`, `metric_name`, `is_spot`

### Implementation Checklist

- [ ] Import quota module: `from cli.shared.quota import get_all_gpu_quotas`
- [ ] Replace entire `_get_all_gpu_quotas()` function body with: `return get_all_gpu_quotas(project_id, region)`
- [ ] Remove old gcloud compute calls (lines 1280-1325)
- [ ] Done! The quota module handles everything

**That's it!** All the logic is in `quota/gpu_quota.py` (code provided above).

### Expected Result
```bash
python training/cli.py infra

GPU Quotas (Vertex AI Custom Training):
   • T4 (Spot): 0 GPUs     ← CORRECT! Shows VA quota
```

**Estimated Time**: ~1.5 hours

---

## Phase 2: Update Console Instructions to Use `quota.get_gpu_quota_metric()`

**Goal**: Replace inline metric name building with canonical `quota.get_gpu_quota_metric()` function.

**File**: `training/cli/setup/core.py`

This phase fixes TWO locations where GPU quota instructions show wrong metric names.

**What changes**: Instead of building metric names inline, import and use `quota.get_gpu_quota_metric(gpu_type, use_spot)`

---

### Metric Name Conversion Logic (Already in quota module!)

**The quota module already has this function** - just use it:

```python
# From quota/gpu_quota.py (already implemented above)
def get_gpu_quota_metric(gpu_type: str, use_spot: bool) -> str:
    """
    Convert GPU type to Vertex AI quota metric name

    Args:
        gpu_type: From .training file (e.g., "NVIDIA_TESLA_T4", "NVIDIA_H100")
        use_preemptible: True for spot/preemptible, False for on-demand

    Returns:
        (metric_name, search_term) tuple

    Examples:
        ("NVIDIA_TESLA_T4", True) ->
            ("custom_model_training_preemptible_nvidia_t4_gpus",
             "Custom model training preemptible NVIDIA T4")

        ("NVIDIA_H100", False) ->
            ("custom_model_training_nvidia_h100_gpus",
             "Custom model training NVIDIA H100")
    """
    # Strip prefixes (handle both NVIDIA_TESLA_* and NVIDIA_*)
    gpu_suffix = gpu_type.replace('NVIDIA_TESLA_', '').replace('NVIDIA_', '').lower()

    # Build metric name
    if use_preemptible:
        metric_name = f"custom_model_training_preemptible_nvidia_{gpu_suffix}_gpus"
        search_term = f"Custom model training preemptible NVIDIA {gpu_suffix.upper()}"
    else:
        metric_name = f"custom_model_training_nvidia_{gpu_suffix}_gpus"
        search_term = f"Custom model training NVIDIA {gpu_suffix.upper()}"

    return (metric_name, search_term)
```

**See ADDENDUM for complete GPU type mapping table.**

---

### Location 1: `display_infrastructure_tree()` (Lines 424-457)

**Context**: Shows GPU quota request instructions in the infra tree display

#### Current Code (WRONG - CE Metric Names)
```python
# Lines 424-457 in display_infrastructure_tree()
if config:
    gpu_type = config.get("WANDB_LAUNCH_ACCELERATOR_TYPE", "NVIDIA_H200")
    use_preemptible = (
        config.get("WANDB_LAUNCH_USE_PREEMPTIBLE", "true").lower() == "true"
    )
    gpu_name = gpu_type.replace("NVIDIA_", "").replace("_", " ").lower()

    status(f"  │        [dim]Manual Request:[/dim]")
    status(f"  │        [dim]1. Visit: https://console.cloud.google.com/iam-admin/quotas[/dim]")

    if use_preemptible:
        status(f"  │        [dim]2. Your config uses PREEMPTIBLE/SPOT (60-91% savings!):[/dim]")
        status(f"  │        [dim]   Search: 'preemptible {gpu_name}'[/dim]")
        status(f"  │        [dim]   Metric: PREEMPTIBLE_{gpu_type.replace('NVIDIA_', '')}_GPUS[/dim]")  # ❌ CE!
    else:
        status(f"  │        [dim]2. Your config uses REGULAR GPUs:[/dim]")
        status(f"  │        [dim]   Search: '{gpu_name}'[/dim]")
        status(f"  │        [dim]   Metric: {gpu_type.replace('NVIDIA_', '')}_GPUS[/dim]")  # ❌ CE!
```

#### New Code (CORRECT - Use Quota Module)
```python
# Lines 424-457 in display_infrastructure_tree()
from cli.shared.quota import get_gpu_quota_metric

if config:
    gpu_type = config.get("WANDB_LAUNCH_ACCELERATOR_TYPE", "NVIDIA_H200")
    use_preemptible = (
        config.get("WANDB_LAUNCH_USE_PREEMPTIBLE", "true").lower() == "true"
    )

    # Get correct Vertex AI metric name from quota module
    metric_name = get_gpu_quota_metric(gpu_type, use_preemptible)

    # Build search term for console UI
    gpu_suffix = gpu_type.replace('NVIDIA_TESLA_', '').replace('NVIDIA_', '')
    if use_preemptible:
        search_term = f"Custom model training preemptible NVIDIA {gpu_suffix}"
    else:
        search_term = f"Custom model training NVIDIA {gpu_suffix}"

    status(f"  │        [dim]Manual Request:[/dim]")
    status(f"  │        [dim]1. Visit: https://console.cloud.google.com/iam-admin/quotas[/dim]")
    status(f"  │        [dim]2. Filter by service: Vertex AI API[/dim]")  # ✅ NEW - specify service!

    if use_preemptible:
        status(f"  │        [dim]3. Your config uses SPOT/PREEMPTIBLE (60-91% savings!):[/dim]")
        status(f"  │        [dim]   Search: '{search_term}'[/dim]")  # ✅ VA quota!
        status(f"  │        [dim]   Metric: {metric_name}[/dim]")  # ✅ VA quota!
    else:
        status(f"  │        [dim]3. Your config uses REGULAR (on-demand) GPUs:[/dim]")
        status(f"  │        [dim]   Search: '{search_term}'[/dim]")  # ✅ VA quota!
        status(f"  │        [dim]   Metric: {metric_name}[/dim]")  # ✅ VA quota!

    status(f"  │        [dim]4. Select region: us-central1 (or preferred)[/dim]")
    status(f"  │        [dim]5. Click 'Edit Quotas' → Request 1-2 GPUs[/dim]")
    status(f"  │        [dim]6. Approval: 1-2 business days[/dim]")
```

**Key Changes**:
- ✅ Added "Filter by service: Vertex AI API" step (step 2)
- ✅ Changed search terms to include "Custom model training"
- ✅ Changed metric names from CE format (`PREEMPTIBLE_T4_GPUS`) to VA format (`custom_model_training_preemptible_nvidia_t4_gpus`)
- ✅ Build metric name ONCE at top, use for both spot and on-demand
- ✅ Renumbered steps (now 1-6 instead of 1-5)

---

### Location 2: Setup Inline Instructions (Lines 1562-1570)

**Context**: Shows GPU quota request instructions during setup flow (when quota is missing)

#### Current Code (WRONG - CE Search Terms)
```python
# Lines 1561-1570 in setup flow
if use_preemptible:
    # Show preemptible quota instructions
    gpu_name = gpu_type.replace("NVIDIA_", "").replace("_", " ")
    status(
        f"[yellow]2. Search for:[/yellow] [cyan]Preemptible NVIDIA {gpu_name} GPUs[/cyan]"  # ❌ CE!
    )
else:
    # Show regular quota instructions
    gpu_name = gpu_type.replace("NVIDIA_", "").replace("_", " ")
    status(f"[yellow]2. Search for:[/yellow] [cyan]NVIDIA {gpu_name} GPUs[/cyan]")  # ❌ CE!
```

#### New Code (CORRECT - Use Quota Module)
```python
# Lines 1561-1570 in setup flow
from cli.shared.quota import get_gpu_quota_metric

# Get correct Vertex AI metric name from quota module
metric_name = get_gpu_quota_metric(gpu_type, use_preemptible)

# Build search term for console UI
gpu_suffix = gpu_type.replace('NVIDIA_TESLA_', '').replace('NVIDIA_', '')

if use_preemptible:
    # Show preemptible quota instructions with VA metric
    search_term = f"Custom model training preemptible NVIDIA {gpu_suffix} GPUs"
    status(f"[yellow]2. Filter by service:[/yellow] [cyan]Vertex AI API[/cyan]")  # ✅ NEW!
    status(f"[yellow]3. Search for:[/yellow] [cyan]{search_term}[/cyan]")  # ✅ VA quota!
    status(f"   [dim]Metric: {metric_name}[/dim]")  # ✅ From quota module!
else:
    # Show regular GPU instructions with VA metric
    search_term = f"Custom model training NVIDIA {gpu_suffix} GPUs"
    status(f"[yellow]2. Filter by service:[/yellow] [cyan]Vertex AI API[/cyan]")  # ✅ NEW!
    status(f"[yellow]3. Search for:[/yellow] [cyan]{search_term}[/cyan]")  # ✅ VA quota!
    status(f"   [dim]Metric: {metric_name}[/dim]")  # ✅ From quota module!
```

**Key Changes**:
- ✅ Added "Filter by service: Vertex AI API" step
- ✅ Changed search terms to include "Custom model training"
- ✅ Show exact VA metric name in dim text below search term
- ✅ Build metric name using same logic as Location 1
- ✅ Renumbered steps (step 2 becomes step 2-3)

---

### Phase 2 Checklist

**Location 1 (display_infrastructure_tree, lines 424-457)**:
- [ ] Import: `from cli.shared.quota import get_gpu_quota_metric`
- [ ] Replace inline metric building with: `metric_name = get_gpu_quota_metric(gpu_type, use_preemptible)`
- [ ] Add "Filter by service: Vertex AI API" as step 2
- [ ] Build search term for UI (just for readability - metric comes from quota module)

**Location 2 (Setup inline, lines 1561-1570)**:
- [ ] Import: `from cli.shared.quota import get_gpu_quota_metric`
- [ ] Replace inline metric building with: `metric_name = get_gpu_quota_metric(gpu_type, use_preemptible)`
- [ ] Add "Filter by service: Vertex AI API" as step 2
- [ ] Build search term for UI (just for readability)

---

## Expected Output (After Fix)

### T4 Spot Configuration
```bash
$ python training/cli.py infra

GPU Quotas (Vertex AI Custom Training):
  ⏳ Pending Approval:
     • T4 (Spot): 0 GPUs

     Manual Request:
     1. Visit: https://console.cloud.google.com/iam-admin/quotas
     2. Filter by service: Vertex AI API  ← ✅ NEW!
     3. Your config uses SPOT/PREEMPTIBLE (60-91% savings!):
        Search: 'Custom model training preemptible NVIDIA T4'  ← ✅ VA quota!
        Metric: custom_model_training_preemptible_nvidia_t4_gpus  ← ✅ VA quota!
     4. Select region: us-central1 (or preferred)
     5. Click 'Edit Quotas' → Request 1-2 GPUs
     6. Approval: 1-2 business days
```

### H100 On-Demand Configuration
```bash
$ python training/cli.py infra

GPU Quotas (Vertex AI Custom Training):
  ⏳ Pending Approval:
     • H100: 0 GPUs

     Manual Request:
     1. Visit: https://console.cloud.google.com/iam-admin/quotas
     2. Filter by service: Vertex AI API  ← ✅ NEW!
     3. Your config uses REGULAR (on-demand) GPUs:
        Search: 'Custom model training NVIDIA H100'  ← ✅ VA quota!
        Metric: custom_model_training_nvidia_h100_gpus  ← ✅ VA quota!
     4. Select region: us-central1 (or preferred)
     5. Click 'Edit Quotas' → Request 1-2 GPUs
     6. Approval: 1-2 business days
```

**See ADDENDUM for all supported GPU types and their metric names.**

---

## Implementation Order

### Step 0: Create Quota Module
- [ ] Create `training/cli/shared/quota/` folder
- [ ] Copy code from plan → `quota/c3_quota.py` (move quota_checker.py)
- [ ] Copy code from plan → `quota/gpu_quota.py` (new file)
- [ ] Copy code from plan → `quota/__init__.py`
- [ ] Commit: "Create canonical quota module (c3_quota.py + gpu_quota.py)"

### Step 1: Update Infra Screen (Phase 1)
- [ ] Open `training/cli/setup/core.py`
- [ ] Add import: `from cli.shared.quota import get_all_gpu_quotas`
- [ ] Replace `_get_all_gpu_quotas()` body with: `return get_all_gpu_quotas(project_id, region)`
- [ ] Commit: "Update infra screen to use quota.get_all_gpu_quotas()"

### Step 2: Update Console Instructions (Phase 2)
- [ ] Add import: `from cli.shared.quota import get_gpu_quota_metric`
- [ ] Update `display_infrastructure_tree()` (lines 424-457) - replace inline metric building
- [ ] Update setup inline instructions (lines 1562-1570) - replace inline metric building
- [ ] Commit: "Update console instructions to use quota.get_gpu_quota_metric()"

### Step 3: Update Other Files to Use Quota Module
- [ ] Update `machine_selection.py`: `from cli.shared.quota_checker import` → `from cli.shared.quota import`
- [ ] Update any other files that import `quota_checker`
- [ ] Delete `training/cli/shared/quota_checker.py` (moved to quota/c3_quota.py)
- [ ] Commit: "Complete quota module migration"

---

## Success Criteria

### All Phases Complete

- ✅ `python training/cli.py infra` shows ONLY VA quotas (no CE quotas)
- ✅ Quota = 0 shows clear manual request instructions
- ✅ Instructions mention "Custom model training" and show exact metric names
- ✅ Metric names are `custom_model_training_*` (VA format)
- ✅ Launch with quota=0 fails fast with clear error
- ✅ Launch with quota=1 succeeds and Vertex AI accepts job

### User Flow (After Fix)

```bash
# User with zero VA quota
$ python training/cli.py infra

GPU Quotas (Vertex AI Custom Training):
   • T4 (Spot): 0 GPUs    ← Shows VA quota!

To request quota:
  Metric: custom_model_training_preemptible_nvidia_t4_gpus  ← Correct metric!

# User requests quota manually (1-2 days)
# ...quota approved...

$ python training/cli.py infra

GPU Quotas (Vertex AI Custom Training):
  ✓  T4 (Spot): 1 GPU    ← Shows VA quota!

$ python training/cli.py launch
# Launch succeeds! Vertex AI accepts job ✓
```

---

## Testing

After implementation, test the two main commands:

```bash
# Test infra screen shows correct quotas
python training/cli.py infra

# Test launch works (or fails with correct instructions)
python training/cli.py launch
```

**Expected**: Shows Vertex AI quotas (not Compute Engine quotas).

---

## Git Commit Messages

### Phase 1
```
Fix infra screen to show Vertex AI quotas

Changed from:
  - Compute Engine quotas (NVIDIA_T4_GPUS)

To:
  - Vertex AI Custom Training quotas
    (custom_model_training_nvidia_t4_gpus)

Why: Vertex AI uses separate quota system. Old display showed
wrong quotas, causing confusion when launch failed.

See: GPU_QUOTA_COMPLETE_FIX_PLAN.md
See: GPU_QUOTA_METRICS_REFERENCE.md
```

### Phase 2
```
Fix quota instructions to show Vertex AI metrics

Users were requesting wrong quota type (Compute Engine instead
of Vertex AI Custom Training). Now shows correct metric names
and search terms.

Changes:
- Added "Filter by service: Vertex AI API" step
- Changed metric names: PREEMPTIBLE_T4_GPUS → custom_model_training_preemptible_nvidia_t4_gpus
- Updated search terms to include "Custom model training"
```

---

## Breaking Changes

### User Impact

**Before Fix**:
- Infra shows CE quotas (wrong but familiar)
- Confusing when Vertex AI rejects

**After Fix**:
- Infra shows VA quotas (correct!)
- Numbers are different (usually lower or 0)
- Clear quota request instructions

### Migration Notes

Users will see different quota numbers after this fix. This is **correct** - they need to request Vertex AI quotas if shown as 0.

Add this message to infra screen header:
```
ℹ️  GPU quotas shown are Vertex AI Custom Training quotas
   (Separate from Compute Engine quotas)
```

---

**Prepared by**: karpathy-deep-oracle ¯\_(ツ)_/¯
**Date**: 2025-11-16
**Status**: Ready for Implementation
**Estimated Time**: 2.5 hours total

---
---
---

# ADDENDUM: Complete GPU Quota Metrics Reference

**Source**: `GPU_QUOTA_METRICS_REFERENCE.md`

This addendum provides the complete reference for all Vertex AI GPU quota metrics, gcloud commands, and implementation helpers.

---

## Complete GPU Type Mapping Table

| GPU Type (.training) | Regular Quota Metric | Preemptible Quota Metric | Display Name | Display Name (Spot) |
|---------------------|---------------------|-------------------------|--------------|---------------------|
| NVIDIA_TESLA_T4 | custom_model_training_nvidia_t4_gpus | custom_model_training_preemptible_nvidia_t4_gpus | T4 | T4 (Spot) |
| NVIDIA_L4 | custom_model_training_nvidia_l4_gpus | custom_model_training_preemptible_nvidia_l4_gpus | L4 | L4 (Spot) |
| NVIDIA_TESLA_V100 | custom_model_training_nvidia_v100_gpus | custom_model_training_preemptible_nvidia_v100_gpus | V100 | V100 (Spot) |
| NVIDIA_TESLA_P4 | custom_model_training_nvidia_p4_gpus | custom_model_training_preemptible_nvidia_p4_gpus | P4 | P4 (Spot) |
| NVIDIA_TESLA_P100 | custom_model_training_nvidia_p100_gpus | custom_model_training_preemptible_nvidia_p100_gpus | P100 | P100 (Spot) |
| NVIDIA_TESLA_A100 | custom_model_training_nvidia_a100_gpus | custom_model_training_preemptible_nvidia_a100_gpus | A100 | A100 (Spot) |
| NVIDIA_A100_80GB | custom_model_training_nvidia_a100_80gb_gpus | custom_model_training_preemptible_nvidia_a100_80gb_gpus | A100 80GB | A100 80GB (Spot) |
| NVIDIA_H100 | custom_model_training_nvidia_h100_gpus | custom_model_training_preemptible_nvidia_h100_gpus | H100 | H100 (Spot) |
| NVIDIA_H100_80GB | custom_model_training_nvidia_h100_80gb_gpus | custom_model_training_preemptible_nvidia_h100_80gb_gpus | H100 80GB | H100 80GB (Spot) |
| NVIDIA_H200 | custom_model_training_nvidia_h200_gpus | custom_model_training_preemptible_nvidia_h200_gpus | H200 | H200 (Spot) |

**Total**: 10 GPU types × 2 variants (on-demand + spot) = 20 quota metrics

---

## Vertex AI JSON Response Structure

### Sample Response (gcloud alpha quotas list)

```json
[
  {
    "name": "projects/weight-and-biases-476906/locations/us-central1/services/aiplatform.googleapis.com/quotas/custom_model_training_preemptible_nvidia_t4_gpus",
    "quotaId": "custom_model_training_preemptible_nvidia_t4_gpus",
    "metric": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus",
    "service": "aiplatform.googleapis.com",
    "quotaDetails": {
      "value": "0"  // Current limit - NOTE: This is a STRING not an int!
    },
    "dimensions": ["region=us-central1"]
  },
  {
    "name": "projects/weight-and-biases-476906/locations/us-central1/services/aiplatform.googleapis.com/quotas/custom_model_training_nvidia_h100_gpus",
    "quotaId": "custom_model_training_nvidia_h100_gpus",
    "metric": "aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus",
    "service": "aiplatform.googleapis.com",
    "quotaDetails": {
      "value": "0"
    },
    "dimensions": ["region=us-central1"]
  }
]
```

### Key Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `quotaId` | string | Short quota identifier (use this for filtering) | `"custom_model_training_nvidia_t4_gpus"` |
| `metric` | string | Full metric name with service prefix | `"aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus"` |
| `quotaDetails.value` | **string** | Current quota limit (**NOT an integer!**) | `"0"`, `"1"`, `"8"` |
| `service` | string | API service name | `"aiplatform.googleapis.com"` |
| `dimensions` | array | Region/zone info | `["region=us-central1"]` |

**Critical**: `quotaDetails.value` is a **string**, not a number! You must convert: `int(quota["quotaDetails"]["value"])`

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

**Returns**: Array of all Vertex AI quotas for that region (GPUs + other resources)

### Get Specific GPU Quota Value

```bash
# T4 Spot
gcloud alpha quotas describe \
    custom_model_training_preemptible_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com \
    --consumer=projects/PROJECT_ID \
    --location=us-central1 \
    --format="value(quotaDetails.value)"

# Output: "0" or "1" (string!)
```

### Request Quota Increase (Manual Alternative)

```bash
# OPTIONAL: Can use gcloud instead of console
gcloud alpha quotas update \
    custom_model_training_preemptible_nvidia_t4_gpus \
    --service=aiplatform.googleapis.com \
    --consumer=projects/PROJECT_ID \
    --location=us-central1 \
    --value=1
```

**Note**: Console UI is usually easier for users than this command.

---

## Console URL Templates

### Direct Link to Vertex AI Quotas

```
https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID&service=aiplatform.googleapis.com
```

### Search for Specific GPU Quota

**T4 Spot**:
```
https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID&service=aiplatform.googleapis.com&metric=custom_model_training_preemptible_nvidia_t4_gpus
```

**H100 On-Demand**:
```
https://console.cloud.google.com/iam-admin/quotas?project=PROJECT_ID&service=aiplatform.googleapis.com&metric=custom_model_training_nvidia_h100_gpus
```

---

## Python Helper Functions

### Convert GPU Type to Quota Metric (Complete Implementation)

```python
def get_vertex_ai_quota_metric(gpu_type: str, use_preemptible: bool = False) -> str:
    """
    Convert WANDB_LAUNCH_ACCELERATOR_TYPE to Vertex AI quota metric.

    Args:
        gpu_type: GPU type from .training (e.g., "NVIDIA_TESLA_T4", "NVIDIA_H100")
        use_preemptible: Whether to use preemptible/spot quota

    Returns:
        Full quota metric name (e.g., "aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus")

    Examples:
        get_vertex_ai_quota_metric("NVIDIA_TESLA_T4", True) ->
            "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus"

        get_vertex_ai_quota_metric("NVIDIA_H100", False) ->
            "aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus"
    """
    # Complete map (includes V100, P4, P100 that our code is missing!)
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

    quota_suffix = gpu_quota_metrics.get(gpu_type, "nvidia_t4_gpus")  # Default to T4

    if use_preemptible:
        quota_suffix = f"preemptible_{quota_suffix}"

    return f"aiplatform.googleapis.com/custom_model_training_{quota_suffix}"


def get_quota_metric_short_name(gpu_type: str, use_preemptible: bool = False) -> str:
    """
    Get short quota metric name (without service prefix).

    Use this for gcloud commands that require short name.
    """
    full_metric = get_vertex_ai_quota_metric(gpu_type, use_preemptible)
    return full_metric.replace("aiplatform.googleapis.com/", "")
```

### Example Usage in Code

```python
# From .training file
gpu_type = "NVIDIA_TESLA_T4"
use_spot = True

# Get full metric name
full_metric = get_vertex_ai_quota_metric(gpu_type, use_spot)
# Returns: "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus"

# Get short name (for gcloud commands)
short_metric = get_quota_metric_short_name(gpu_type, use_spot)
# Returns: "custom_model_training_preemptible_nvidia_t4_gpus"

# Use in gcloud command
result = subprocess.run([
    "gcloud", "alpha", "quotas", "describe",
    short_metric,  # Use short name here!
    "--service=aiplatform.googleapis.com",
    "--consumer=projects/PROJECT_ID",
    "--location=us-central1",
    "--format=value(quotaDetails.value)"
], capture_output=True, text=True)

quota_value = int(result.stdout.strip())  # Convert string to int!
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using Compute Engine Quota Names
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

---

### ❌ Mistake 2: Forgetting to Convert String to Int
```python
# WRONG - quotaDetails.value is a string!
quota_value = quota["quotaDetails"]["value"]
if quota_value > 0:  # TypeError: '>' not supported between str and int
```

### ✅ Correct: Convert String to Int
```python
# CORRECT - Convert first!
quota_value = int(quota["quotaDetails"]["value"])
if quota_value > 0:  # Works!
```

---

### ❌ Mistake 3: Using Wrong JSON Path
```python
# WRONG - This is CE quota JSON structure!
limit = quota["limit"]  # KeyError: 'limit' doesn't exist in VA response
```

### ✅ Correct: Using VA JSON Structure
```python
# CORRECT - VA quotas use quotaDetails.value
limit_str = quota.get("quotaDetails", {}).get("value", "0")
limit = int(limit_str)
```

---

### ❌ Mistake 4: Missing Service Prefix in Commands
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

## Testing Your Implementation

### Check Current Quota Value (Full Command)

```bash
# Replace with your values:
PROJECT_ID="weight-and-biases-476906"
REGION="us-central1"
GPU_METRIC="custom_model_training_preemptible_nvidia_t4_gpus"

gcloud alpha quotas describe \
    $GPU_METRIC \
    --service=aiplatform.googleapis.com \
    --consumer=projects/$PROJECT_ID \
    --location=$REGION \
    --format=json
```

**Expected Output**:
```json
{
  "name": "projects/weight-and-biases-476906/locations/us-central1/services/aiplatform.googleapis.com/quotas/custom_model_training_preemptible_nvidia_t4_gpus",
  "quotaDetails": {
    "value": "0"  ← If 0, you need to request quota!
  }
}
```

### Verify All GPU Quotas for a Region

```bash
# List ALL Vertex AI quotas and filter for GPUs
gcloud alpha quotas list \
    --service=aiplatform.googleapis.com \
    --consumer=projects/weight-and-biases-476906 \
    --location=us-central1 \
    --format=json | \
    jq '[.[] | select(.quotaId | contains("nvidia")) | {quotaId, limit: .quotaDetails.value}]'
```

**Expected Output**:
```json
[
  {"quotaId": "custom_model_training_nvidia_t4_gpus", "limit": "0"},
  {"quotaId": "custom_model_training_preemptible_nvidia_t4_gpus", "limit": "0"},
  {"quotaId": "custom_model_training_nvidia_l4_gpus", "limit": "0"},
  ...
]
```

---

## Sources & Documentation

1. **GCP Official Docs**: https://docs.cloud.google.com/vertex-ai/docs/quotas
2. **Quota API Docs**: https://docs.cloud.google.com/docs/quotas/quota-adjuster
3. **Our Metrics Reference**: `GPU_QUOTA_METRICS_REFERENCE.md` (complete file)
4. **Our Bug Report**: Previously documented in `VERTEX_AI_GPU_QUOTA_BUG_REPORT.md`

---

**End of ADDENDUM**

This addendum provides all the detailed reference information needed to implement the GPU quota fixes. Refer back to the main plan sections for implementation steps and checklists.

---
---

# ADDENDUM 2: File Reorganization (Completed 2025-11-16)

**Pricing file structure was reorganized for clarity and consolidation.**

## Changes Made

### 1. Renamed `artifact_pricing.py` → `pricing.py`

**Old name problem**: `artifact_pricing.py` was misleading
- Only 2 of 9 functions dealt with Artifact Registry
- 7 functions were generic pricing helpers (work on ANY pricing data)

**New name benefit**: `pricing.py` better reflects the file's purpose
- Generic pricing operations for GCP resources
- Clear, simple, descriptive

### 2. Consolidated `get_live_prices.py` into `pricing.py`

**What was removed**: `training/cli/shared/pricing/get_live_prices.py`
- Had only 1 function: `get_live_price_for_launch()`
- Was a specialized calculator using pricing.py helpers

**What was added**: Function moved into `pricing.py` as `get_machine_hourly_cost()`
- Same logic, clearer name
- All pricing operations now in ONE file

### 3. Function Rename

**Old**: `get_live_price_for_launch(machine_type, region)`
**New**: `get_machine_hourly_cost(machine_type, region)`

**Why**: More descriptive
- "launch" was vague (launch what?)
- "machine hourly cost" is explicit (machine pricing per hour)

### 4. Updated Import Sites

**Changed in 3 locations** (`training/cli/launch/core.py`):
```python
# OLD
from cli.shared.pricing.get_live_prices import get_live_price_for_launch
prov_price = get_live_price_for_launch(best_machine, region)

# NEW
from cli.shared.pricing import get_machine_hourly_cost
prov_price = get_machine_hourly_cost(best_machine, region)
```

## Final Pricing File Structure

```
training/cli/shared/pricing/          # ✅ ALL pricing code in one directory!
├── __init__.py                       # All 11 pricing functions
│   ├── fetch_pricing_no_save()
│   ├── upload_pricing_to_artifact_registry()
│   ├── get_spot_price()
│   ├── get_standard_price()
│   ├── get_commitment_1yr_price()
│   ├── get_commitment_3yr_price()
│   ├── all_prices()
│   ├── get_pricing_age_minutes()
│   ├── format_pricing_age()
│   ├── list_all_pricing_options()
│   └── get_machine_hourly_cost()     # 🆕 Moved from get_live_prices.py
│
├── pricing_config.py                 # Configuration constants
│
├── cloud_function/
│   └── main.py                       # Cloud Function (fetches from GCP)
│
└── data/                             # Cached pricing data
```

**Import examples:**
```python
# Import pricing functions
from cli.shared.pricing import fetch_pricing_no_save, get_machine_hourly_cost

# Import config
from cli.shared.pricing.pricing_config import FUNCTION_NAME, PRICING_SCHEMA
```

## Benefits

✅ **Single directory**: All pricing code in `training/cli/shared/pricing/`
✅ **Clearer organization**: Functions, config, cloud function all grouped together
✅ **No name conflicts**: Folder structure prevents Python import shadowing issues
✅ **Easier maintenance**: Everything pricing-related in one place

## Git Commits

1. `7eb2525` - Rename artifact_pricing.py → pricing.py
2. `76b79b0` - Move get_live_price_for_launch into pricing.py as get_machine_hourly_cost
3. `7a3a579` - Consolidate all pricing files into pricing/ folder
4. `adc96ee` - Rename config.py → pricing_config.py for clarity

---

**End of ADDENDUM 2**

---
---

# ADDENDUM 3: Quota Consolidation & Standardization (In Preparation)

**Created `training/cli/shared/quota/` folder to consolidate all quota checking into a canonical shared module (analogous to `pricing/` folder).**

**This is NOT a bug fix** - it's a **refactoring to gather quota functions in one place** so mecha, launch, setup, and all other parts can use standardized quota checking.

---

## Current Quota Sprawl (What We Have Now)

Quota checking is currently scattered across multiple files:

### 1. **Cloud Build C3 Quotas**
**File**: `training/cli/shared/quota_checker.py` (149 lines)

**Functions**:
- `get_cloud_build_c3_quotas(project_id)` → Returns all regions with quota
- `get_cloud_build_c3_region_quota(project_id, region)` → Single region quota
- `has_sufficient_quota(project_id, region, required_vcpus)` → Boolean check
- `_fetch_quotas_from_api(project_id)` → Internal API call

**Used by**:
- `machine_selection.py` - Selects best C3 machine within quota
- `launch/core.py` - Validates quota before launch
- `setup/core.py` - Quota verification during setup

### 2. **Vertex AI GPU Quotas**
**Status**: ❌ **NOT YET IMPLEMENTED!**

**Needed for**: GPU training job launches (.training file uses GPU config)

**Will need**:
- GPU type → Vertex AI quota metric mapping
- Region availability checking
- Spot vs On-Demand quota differentiation

### 3. **Lazy Quota Entry Creation**
**File**: `training/cli/launch/mecha/mecha_acquire.py`

**Function**: `lazy_load_quota_entry(project_id, region, output_callback)`
- Submits test build to trigger quota entry (creates 4 vCPU default)
- Used when region has no quota entry yet

**Used by**:
- `launch/core.py` - Creates quota entry if missing

### 4. **MECHA Quota Display** (STAYS SEPARATE)
**File**: `training/cli/launch/mecha/mecha_quota.py` (237 lines)

**Functions**:
- `roll_call_display()` - MECHA roll call with flags (🇺🇸, 🇯🇵)
- `display_sidelined_mecha()` - Show regions without quota

**Keep in mecha/**: This is UI/display logic, not core quota checking

---

## Proposed Consolidation (What We'll Have)

### Final Structure

```
training/cli/shared/quota/
├── __init__.py                   # Re-exports all quota functions
│   # from .c3_quota import *
│   # from .gpu_quota import *
│
├── c3_quota.py                   # Cloud Build C3 quota checking
│   ├── get_cloud_build_c3_quotas()
│   ├── get_cloud_build_c3_region_quota()
│   ├── has_sufficient_c3_quota()
│   └── _fetch_c3_quotas_from_api()
│
└── gpu_quota.py                  # Vertex AI GPU quota checking (NEW!)
    ├── get_gpu_quotas()
    ├── get_gpu_region_quota()
    ├── get_gpu_quota_metric()
    ├── has_gpu_quota()
    └── _fetch_gpu_quotas_from_api()
```

**Note**: Caching removed! Original quota_checker.py had 5-minute cache, but it's unnecessary since quota functions are only called once per CLI run.

### What Gets Moved

| Current Location | New Location | What It Does |
|------------------|--------------|--------------|
| `quota_checker.py` (123 lines) | `quota/c3_quota.py` | C3 quota checking |
| **NEW** | `quota/gpu_quota.py` | GPU quota checking |
| `mecha_quota.py` (237 lines) | **STAYS in mecha/** | UI display logic |
| `mecha_acquire.py::lazy_load_quota_entry()` | **STAYS in mecha/** | MECHA-specific quota creation |

---

## Standardized Quota API (How All Parts Use It)

### Cloud Build C3 Quotas

**Before** (scattered):
```python
# Different files import from different places
from cli.shared.quota_checker import get_cloud_build_c3_region_quota
from cli.shared.quota_checker import has_sufficient_quota
```

**After** (canonical):
```python
# Everyone imports from quota/ module
from cli.shared.quota import (
    get_cloud_build_c3_quotas,      # All regions
    get_cloud_build_c3_region_quota, # Single region
    has_sufficient_c3_quota,         # Boolean check
)
```

### Vertex AI GPU Quotas (NEW!)

```python
from cli.shared.quota import (
    get_gpu_quotas,                   # All regions for GPU type
    get_gpu_region_quota,             # Single region GPU quota
    get_gpu_quota_metric,             # GPU type → metric name
    has_gpu_quota,                    # Boolean check
)

# Example usage in launch/core.py
gpu_type = "NVIDIA_TESLA_T4"
use_spot = True
region = "us-west2"

quota = get_gpu_region_quota(project_id, region, gpu_type, use_spot)
if quota > 0:
    # Launch training job
    pass
else:
    # Show error: "No T4 spot quota in us-west2"
    pass
```

---

## How Different Parts Use Shared Quota Module

### 1. **machine_selection.py** (Cloud Build)

```python
# Import from quota/ module
from cli.shared.quota import get_cloud_build_c3_region_quota

def get_best_c3(project_id, region):
    cb_quota = get_cloud_build_c3_region_quota(project_id, region)

    if cb_quota >= 176:
        return "c3-standard-176", 176
    elif cb_quota >= 88:
        return "c3-standard-88", 88
    else:
        return "c3-standard-44", 44
```

### 2. **mecha/** (Display Only)

```python
# Core quota checking from shared quota module
from cli.shared.quota import get_cloud_build_c3_quotas

# MECHA-specific UI display (stays in mecha/)
from cli.launch.mecha.mecha_quota import roll_call_display, display_sidelined_mecha

# Get quotas using shared module
quotas = get_cloud_build_c3_quotas(project_id)

# Use MECHA-specific display logic
roll_call_display(list(quotas.keys()), status)
display_sidelined_mecha(all_regions - quotas.keys(), status)
```

### 4. **setup/core.py** (Validation)

```python
# Import from quota/ module
from cli.shared.quota import has_sufficient_c3_quota, has_gpu_quota

# Validate Cloud Build quota
if not has_sufficient_c3_quota(project_id, region, 176):
    status("⚠️  Need 176 vCPU quota for optimal builds")

# Validate GPU quota
if not has_gpu_quota(project_id, region, gpu_type, use_spot):
    status("⚠️  No GPU quota - request in Console")
```

---

## Benefits of Consolidation

✅ **Single source of truth**: All quota checking in one canonical module
✅ **DRY principle**: No duplicated quota logic across files
✅ **Consistent API**: Same function signatures everywhere
✅ **Easy testing**: Test quota module once, works everywhere
✅ **Clear separation**: Core quota logic vs UI display logic
✅ **Future-proof**: Easy to add TPU, storage, network quotas

**Parallel to pricing/**: Same organization pattern we used for pricing consolidation!

---

## Implementation Steps (When Ready)

### Phase 1: Move Existing Cloud Build Code
1. Move `quota_checker.py` → `quota/c3_quota.py`
2. Update all imports (machine_selection, launch/core, setup/core)
3. Add re-exports to `quota/__init__.py`

### Phase 2: Add Vertex AI GPU Quota Support
1. Create `quota/gpu_quota.py` with GPU quota functions
2. Implement GPU quota checking (Vertex AI API)
3. Add GPU type → metric mapping
4. Update setup/core.py to use GPU quota functions (infra screen + console instructions)

### Phase 3: Test & Verify
1. Test Cloud Build quota checks still work
2. Test GPU quota checks work for T4/L4/A100/H100
3. Verify mecha display still works
4. Verify setup validation still works

---

**End of ADDENDUM 3**

---
---

# ADDENDUM 4: mecha_quota.py Rename & Import Cleanup (Completed 2025-11-16)

**Renamed `mecha_quota.py` → `mecha_display.py` for accurate naming.**

**Date**: 2025-11-16  
**Commits**: `e5210f3` (quota module), `a1ab751` (mecha rename)

---

## The Problem

**File name was misleading:**
- `mecha_quota.py` implies quota checking/fetching
- Actually contains ONLY display/formatting functions
- No quota checking happens in this file!

**Import bug discovered:**
- Line 88 still imported from old `quota_checker` module
- Should import from new canonical `quota` module

---

## The Solution

### 1. Renamed File
```bash
training/cli/launch/mecha/mecha_quota.py → mecha_display.py
```

**Why "display"?**
- File contains 4 UI/display functions (237 lines)
- No quota checking logic
- Just formats MECHA status with flags/ASCII art

### 2. Fixed Import Bug
**In mecha_display.py (line 88):**
```python
# Before
from ...shared.quota_checker import get_cloud_build_c3_quotas

# After
from ...shared.quota import get_cloud_build_c3_quotas  ✅
```

### 3. Updated All Imports
**In mecha_integration.py (2 fixes):**
```python
# Before
from ...shared.quota_checker import get_cloud_build_c3_quotas
from .mecha_quota import (separate_by_quota, ...)

# After
from ...shared.quota import get_cloud_build_c3_quotas  ✅
from .mecha_display import (separate_by_quota, ...)    ✅
```

---

## What mecha_display.py Actually Does

**4 Functions (all display/formatting):**

1. **`roll_call_display()`** (lines 34-67)
   - Formats regions with flag emojis
   - Groups into lines of 7
   - Example: `🇺🇸 us-west1 ∿ 🇯🇵 asia-northeast1 ∿ ...`

2. **`separate_by_quota()`** (lines 70-102)
   - Splits regions into battle-ready vs sidelined
   - Calls quota module to get quota values
   - Returns: `(battle_ready, sidelined)` tuple

3. **`display_sidelined_mechas()`** (lines 105-212)
   - Shows sidelined MECHAs with instructions
   - Includes MR GODZILLA ASCII art
   - Shows Console URL with pre-applied filters
   - Includes Enkidu's cedar forest advice (humor)

4. **`display_battle_ready_mechas()`** (lines 214-238)
   - Shows battle-ready MECHAs section
   - Roll call display format

**All functions are UI/presentation logic - NO quota checking!**

---

## Final Quota File Structure

```
training/cli/
├── shared/quota/              ← Canonical quota module (data fetching)
│   ├── __init__.py           (32 lines)
│   ├── c3_quota.py           (103 lines) - Cloud Build quota checking
│   └── gpu_quota.py          (225 lines) - Vertex AI GPU quota checking
│
└── launch/mecha/
    └── mecha_display.py      (237 lines) - MECHA UI/display only
```

**Clean separation:**
- **Data fetching**: `shared/quota/` (canonical module)
- **UI display**: `launch/mecha/mecha_display.py` (MECHA-specific)

---

## No Duplicate Functions

**`get_cloud_build_c3_quotas()` exists in only ONE place:**
- `training/cli/shared/quota/c3_quota.py` ✓

**All files import from canonical quota module:**
- ✅ `machine_selection.py` → `from cli.shared.quota import ...`
- ✅ `setup/core.py` → `from ..shared.quota import ...`
- ✅ `mecha_integration.py` → `from ...shared.quota import ...`
- ✅ `mecha_display.py` → `from ...shared.quota import ...`

**Single source of truth achieved!**

---

## Git Commits

### Commit 1: Create Quota Module
```
Commit: e5210f3
Message: Create canonical quota module (Phase 1: Vertex AI quotas)
Files: 5 changed, 341 insertions(+), 143 deletions(-)

Created:
- training/cli/shared/quota/c3_quota.py
- training/cli/shared/quota/gpu_quota.py
- training/cli/shared/quota/__init__.py

Updated:
- training/cli/setup/core.py (uses quota module)
- training/cli/shared/machine_selection.py (imports from quota)

Deleted:
- training/cli/shared/quota_checker.py (migrated to quota/c3_quota.py)
```

### Commit 2: Rename mecha_quota.py
```
Commit: a1ab751
Message: Rename mecha_quota.py → mecha_display.py (accurate naming)
Files: 2 changed, 4 insertions(+), 4 deletions(-)

Renamed:
- mecha_quota.py → mecha_display.py

Updated:
- mecha_display.py (fixed import: quota_checker → quota)
- mecha_integration.py (updated imports)
```

---

**End of ADDENDUM 4**

---

## ADDENDUM 6: Explicit Service Naming (2025-11-16)

**Post-plan completion: All quota functions renamed for explicit service clarity**

After this plan was implemented, quota function names were refined to explicitly indicate which GCP service they query:

### C3 Quota Renames (Cloud Build Service)
```python
# OLD → NEW
has_sufficient_quota()         → has_cloud_build_c3_quota()
```

### GPU Quota Renames (Vertex AI Service)
```python
# OLD → NEW
get_gpu_quotas()               → get_vertex_gpu_quotas()
get_all_gpu_quotas()           → get_all_vertex_gpu_quotas()
get_gpu_quota_metric()         → get_vertex_gpu_quota_metric()
has_gpu_quota()                → has_vertex_gpu_quota()
_fetch_gpu_quota_from_api()    → _fetch_vertex_gpu_quota_from_api()
```

### Why Renamed?
- **Explicit service identification**: Every function name now clearly shows which GCP service (Cloud Build vs Vertex AI)
- **Prevents confusion**: `has_sufficient_quota()` was generic, could be mistaken for any quota type
- **Consistent pattern**: GPU functions had `gpu_` prefix, now have `vertex_gpu_` prefix matching service
- **Future-proof**: New GCP services can be added with clear naming (e.g., `get_dataflow_quotas()`)

### Current File Structure
```
training/cli/shared/quota/
├── __init__.py              # Exports renamed functions
├── c3_quota.py             # Cloud Build quotas (C3 machine types)
└── gpu_quota.py            # Vertex AI quotas (GPU accelerators)
```

### Import Examples
```python
# Modern imports (post-rename)
from cli.shared.quota import (
    # Cloud Build
    get_cloud_build_c3_quotas,
    get_cloud_build_c3_region_quota,
    has_cloud_build_c3_quota,          # ✨ Renamed
    
    # Vertex AI
    get_vertex_gpu_quotas,             # ✨ Renamed
    get_all_vertex_gpu_quotas,         # ✨ Renamed
    get_vertex_gpu_quota_metric,       # ✨ Renamed
    has_vertex_gpu_quota,              # ✨ Renamed
)
```

**All function references in this historical plan document remain unchanged** - they represent the original implementation. Current code uses the renamed functions listed above.

**Commits**:
- 8eeb41c - Rename all quota functions for explicit service clarity
- 0f96ea0 - Fix critical bug: Silent GPU type fallback to T4 (added before rename)
