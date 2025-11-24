# Infrastructure Source of Truth Architecture

**Date**: 2025-11-21
**Status**: PHASE 1 COMPLETE ✅ / PHASE 2 PENDING ⏳
**Goal**: Each system owns its own truth via `_MANIFEST` const, code imports from source

---

## 🎯 The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOURCE OF TRUTH FILES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  zeus_battle.py ──► ZEUS_THUNDER_MANIFEST (5 GPU tiers)        │
│                                                                 │
│  mecha_regions.py ──► MECHA_C3_MANIFEST (18 C3 regions)        │
│                                                                 │
│  gpu_quota.py ──► GPU_TYPES_MANIFEST (10 GPU types)            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  gcp-manifest.json ──► HIGH-LEVEL POINTER (just _source refs)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CODE IMPORTS FROM SOURCE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  from zeus_battle import ZEUS_THUNDER_MANIFEST                  │
│  from mecha_regions import MECHA_C3_MANIFEST                    │
│  from gpu_quota import GPU_TYPES_MANIFEST                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ PHASE 1 COMPLETE: Define Source of Truth

### 1.1 ZEUS_THUNDER_MANIFEST ✅

**File**: `CLI/launch/zeus/zeus_battle.py`

```python
ZEUS_THUNDER_MANIFEST = {
    "system": "zeus",
    "description": "ZEUS Thunder Battle System - GPU tier pricing battles for Vertex AI",
    "registry_file": "CLI/launch/zeus/data/zeus_olympus.json",
    "total_tiers": 5,
    "tiers": [
        {"name": "spark", "emoji": "⚡", "gpu_type": "NVIDIA_TESLA_T4", "display_name": "T4", "memory_gb": 16, "machine_type": "n1-standard-4", "cost_tier": "low"},
        {"name": "bolt", "emoji": "⚡⚡", "gpu_type": "NVIDIA_L4", "display_name": "L4", "memory_gb": 24, "machine_type": "g2-standard-4", "cost_tier": "medium"},
        {"name": "storm", "emoji": "⚡⚡⚡", "gpu_type": "NVIDIA_TESLA_A100", "display_name": "A100", "memory_gb": 40, "machine_type": "a2-highgpu-1g", "cost_tier": "high"},
        {"name": "tempest", "emoji": "⚡⚡⚡⚡", "gpu_type": "NVIDIA_H100_80GB", "display_name": "H100 80GB", "memory_gb": 80, "machine_type": "a3-highgpu-8g", "cost_tier": "premium"},
        {"name": "cataclysm", "emoji": "⚡⚡⚡⚡⚡", "gpu_type": "NVIDIA_H200", "display_name": "H200", "memory_gb": 141, "machine_type": "a3-highgpu-8g", "cost_tier": "premium"},
    ],
}
```

**Status**: ✅ DEFINED

---

### 1.2 MECHA_C3_MANIFEST ✅

**File**: `CLI/launch/mecha/mecha_regions.py`

```python
MECHA_C3_MANIFEST = {
    "system": "mecha",
    "description": "MECHA Battle System - C3 Cloud Build worker pool regions",
    "registry_file": "CLI/launch/mecha/data/mecha_hangar.json",
    "machine_type_default": "c3-standard-176",
    "total_regions": 18,
    "regions": [
        {"code": "us-central1", "location": "Iowa, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-east1", "location": "South Carolina, USA", "continent": "north_america", "latency": "low"},
        # ... all 18 regions
    ],
}
```

**Status**: ✅ DEFINED

---

### 1.3 GPU_TYPES_MANIFEST ✅

**File**: `CLI/shared/quota/gpu_quota.py`

```python
GPU_TYPES_MANIFEST = {
    "description": "All Vertex AI GPU types for quota checking",
    "total_types": 10,
    "zeus_supported_count": 5,
    "non_zeus_count": 5,
    "types": [
        # ZEUS SUPPORTED (5)
        {"internal": "NVIDIA_TESLA_T4", "display": "T4", "metric": "nvidia_t4_gpus", "memory_gb": 16, "zeus_tier": "spark", ...},
        {"internal": "NVIDIA_L4", "display": "L4", "metric": "nvidia_l4_gpus", "memory_gb": 24, "zeus_tier": "bolt", ...},
        {"internal": "NVIDIA_TESLA_A100", "display": "A100", "metric": "nvidia_a100_gpus", "memory_gb": 40, "zeus_tier": "storm", ...},
        {"internal": "NVIDIA_H100_80GB", "display": "H100 80GB", "metric": "nvidia_h100_80gb_gpus", "memory_gb": 80, "zeus_tier": "tempest", ...},
        {"internal": "NVIDIA_H200", "display": "H200", "metric": "nvidia_h200_gpus", "memory_gb": 141, "zeus_tier": "cataclysm", ...},
        # NON-ZEUS (5)
        {"internal": "NVIDIA_A100_80GB", "display": "A100 80GB", ..., "zeus_tier": None, ...},
        {"internal": "NVIDIA_H100", "display": "H100", ..., "zeus_tier": None, ...},
        {"internal": "NVIDIA_TESLA_V100", "display": "V100", ..., "zeus_tier": None, ...},
        {"internal": "NVIDIA_TESLA_P4", "display": "P4", ..., "zeus_tier": None, ...},
        {"internal": "NVIDIA_TESLA_P100", "display": "P100", ..., "zeus_tier": None, ...},
    ],
}
```

**Status**: ✅ DEFINED

---

### 1.4 gcp-manifest.json Updated ✅

**File**: `CLI/config/gcp-manifest.json`

Now v2.0.0 - high-level pointer only:

```json
{
  "version": "2.0.0",
  "description": "GCP infrastructure index - see system files for canonical definitions",

  "source_of_truth": {
    "zeus_gpu_tiers": {
      "file": "CLI/launch/zeus/zeus_battle.py",
      "const": "ZEUS_THUNDER_MANIFEST"
    },
    "mecha_c3_regions": {
      "file": "CLI/launch/mecha/mecha_regions.py",
      "const": "MECHA_C3_MANIFEST"
    },
    "gpu_types_all": {
      "file": "CLI/shared/quota/gpu_quota.py",
      "const": "GPU_TYPES_MANIFEST"
    }
  },

  "worker_pools": {
    "_source": "CLI/launch/mecha/mecha_regions.py:MECHA_C3_MANIFEST"
  },

  // apis, iam_roles, critical_items remain here (manifest-specific)
}
```

**Status**: ✅ UPDATED

---

## ⏳ PHASE 2 PENDING: Wire Up Imports

**Problem**: Code still uses OLD patterns, not the new _MANIFEST consts!

### Current State (NOT WIRED)

```python
# infra_verify.py - STILL loads gcp-manifest.json directly!
manifest_path = Path(__file__).parent.parent / "config" / "gcp-manifest.json"

# Other files - STILL use old dicts, not _MANIFEST:
from .mecha_regions import C3_REGIONS, ALL_MECHA_REGIONS  # OLD
from .zeus_battle import THUNDER_TIERS  # OLD
GPU_QUOTA_METRICS = {...}  # OLD dict, not from manifest
```

### Target State (WIRED UP)

```python
# infra_verify.py - Import from source files!
from CLI.launch.zeus.zeus_battle import ZEUS_THUNDER_MANIFEST
from CLI.launch.mecha.mecha_regions import MECHA_C3_MANIFEST
from CLI.shared.quota.gpu_quota import GPU_TYPES_MANIFEST

# Other files - Import _MANIFEST and derive what they need
from .zeus_battle import ZEUS_THUNDER_MANIFEST
gpu_tiers = {t["name"]: t for t in ZEUS_THUNDER_MANIFEST["tiers"]}
```

---

## 📋 PHASE 2 Implementation Plan

### Step 2.1: Update infra_verify.py

**Current** (loads manifest.json):
```python
manifest_path = Path(__file__).parent.parent / "config" / "gcp-manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)
```

**Target** (import from source):
```python
from ..launch.zeus.zeus_battle import ZEUS_THUNDER_MANIFEST
from ..launch.mecha.mecha_regions import MECHA_C3_MANIFEST
from .quota.gpu_quota import GPU_TYPES_MANIFEST

# Use directly
all_gpu_types = GPU_TYPES_MANIFEST["types"]
all_c3_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
zeus_tiers = ZEUS_THUNDER_MANIFEST["tiers"]
```

**Files to update**:
- [ ] `CLI/shared/infra_verify.py` - Main consumer of manifest
- [ ] `CLI/shared/infra_print.py` - Display logic

---

### Step 2.2: Add Helper Functions to Each Manifest

Add convenience functions alongside each _MANIFEST:

**zeus_battle.py**:
```python
def get_zeus_tier_by_name(name: str) -> dict:
    """Get tier config by name (spark, bolt, etc.)"""
    for tier in ZEUS_THUNDER_MANIFEST["tiers"]:
        if tier["name"] == name:
            return tier
    return None

def get_zeus_tier_by_gpu(gpu_type: str) -> dict:
    """Get tier config by GPU type"""
    for tier in ZEUS_THUNDER_MANIFEST["tiers"]:
        if tier["gpu_type"] == gpu_type:
            return tier
    return None

def get_all_zeus_gpu_types() -> list:
    """Get list of all ZEUS-supported GPU types"""
    return [t["gpu_type"] for t in ZEUS_THUNDER_MANIFEST["tiers"]]
```

**mecha_regions.py**:
```python
def get_all_mecha_region_codes() -> list:
    """Get list of all 18 region codes"""
    return [r["code"] for r in MECHA_C3_MANIFEST["regions"]]

def get_mecha_regions_by_continent(continent: str) -> list:
    """Get regions filtered by continent"""
    return [r for r in MECHA_C3_MANIFEST["regions"] if r["continent"] == continent]
```

**gpu_quota.py**:
```python
def get_gpu_type_by_internal(internal_name: str) -> dict:
    """Get GPU type config by internal name"""
    for gpu in GPU_TYPES_MANIFEST["types"]:
        if gpu["internal"] == internal_name:
            return gpu
    return None

def get_zeus_supported_gpus() -> list:
    """Get only ZEUS-supported GPU types"""
    return [g for g in GPU_TYPES_MANIFEST["types"] if g["zeus_tier"] is not None]

def get_non_zeus_gpus() -> list:
    """Get non-ZEUS GPU types (manual selection)"""
    return [g for g in GPU_TYPES_MANIFEST["types"] if g["zeus_tier"] is None]
```

---

### Step 2.3: Deprecate Old Dicts (Backward Compat)

Keep old dicts but derive them from _MANIFEST:

**zeus_battle.py**:
```python
# DEPRECATED: Use ZEUS_THUNDER_MANIFEST instead
# Kept for backward compatibility
THUNDER_TIERS = {
    tier["name"]: {
        "emoji": tier["emoji"],
        "gpu_types": [tier["gpu_type"]],
        "memory_gb": tier["memory_gb"]
    }
    for tier in ZEUS_THUNDER_MANIFEST["tiers"]
}
```

**mecha_regions.py**:
```python
# DEPRECATED: Use MECHA_C3_MANIFEST instead
# Kept for backward compatibility
C3_REGIONS = {r["code"]: {"location": r["location"], ...} for r in MECHA_C3_MANIFEST["regions"]}
ALL_MECHA_REGIONS = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
```

**gpu_quota.py**:
```python
# DEPRECATED: Use GPU_TYPES_MANIFEST instead
# Kept for backward compatibility
GPU_QUOTA_METRICS = {g["internal"]: g["metric"] for g in GPU_TYPES_MANIFEST["types"]}
```

---

### Step 2.4: Update All Importers

Files that need updating to use _MANIFEST:

```
CLI/shared/infra_verify.py          - Main manifest consumer
CLI/shared/infra_print.py           - Display GPU/region info
CLI/monitor/screen.py               - Uses ALL_MECHA_REGIONS
CLI/monitor/core.py                 - Uses ALL_MECHA_REGIONS
CLI/launch/mecha/mecha_integration.py - Uses ALL_MECHA_REGIONS
CLI/launch/runner_cleanup.py        - Uses ALL_MECHA_REGIONS
CLI/setup/core.py                   - Hardcoded "us-central1"
```

---

## 📊 Progress Summary

| Task | Status |
|------|--------|
| Define ZEUS_THUNDER_MANIFEST | ✅ Done |
| Define MECHA_C3_MANIFEST | ✅ Done |
| Define GPU_TYPES_MANIFEST | ✅ Done |
| Update gcp-manifest.json v2.0 | ✅ Done |
| Add helper functions | ⏳ Pending |
| Derive old dicts from _MANIFEST | ⏳ Pending |
| Update infra_verify.py | ⏳ Pending |
| Update infra_print.py | ⏳ Pending |
| Update monitor files | ⏳ Pending |
| Update launch files | ⏳ Pending |
| Test everything works | ⏳ Pending |

---

## 🎯 Benefits When Complete

1. **Single source of truth** - Each system owns its data
2. **No duplication** - Data defined once, derived elsewhere
3. **Easy updates** - Change ZEUS tier? Edit one place
4. **Self-documenting** - _MANIFEST shows full structure
5. **Type-safe helpers** - Functions to access data safely
6. **Backward compatible** - Old dicts still work (derived)

---

## 📁 File Structure After Phase 2

```
CLI/
├── config/
│   └── gcp-manifest.json          # v2.0 - high-level pointer only
│
├── launch/
│   ├── zeus/
│   │   └── zeus_battle.py         # ZEUS_THUNDER_MANIFEST + helpers
│   └── mecha/
│       └── mecha_regions.py       # MECHA_C3_MANIFEST + helpers
│
└── shared/
    ├── quota/
    │   └── gpu_quota.py           # GPU_TYPES_MANIFEST + helpers
    ├── infra_verify.py            # Imports from _MANIFEST sources
    └── infra_print.py             # Imports from _MANIFEST sources
```

---

**PHASE 1**: ✅ Structure defined
**PHASE 2**: ✅ infra_verify.py WIRED UP!

---

## ✅ Completed Wiring (2025-11-21)

**infra_verify.py** now imports from _MANIFEST sources:

```python
# Import source of truth manifests
from ..launch.mecha.mecha_regions import MECHA_C3_MANIFEST
from .quota.gpu_quota import GPU_TYPES_MANIFEST

# Worker pool regions (18 regions from source)
mecha_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
machine_type = MECHA_C3_MANIFEST["machine_type_default"]

# C3 quota regions (18 regions from source)
c3_check_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
```

**Commit**: 75f0b6b7 - "Wire infra_verify.py to import from _MANIFEST sources"

---

## 📋 Remaining Tasks

| Task | Status |
|------|--------|
| ~~Define ZEUS_THUNDER_MANIFEST~~ | ✅ Done |
| ~~Define MECHA_C3_MANIFEST~~ | ✅ Done |
| ~~Define GPU_TYPES_MANIFEST~~ | ✅ Done |
| ~~Update gcp-manifest.json v2.0~~ | ✅ Done |
| ~~Wire infra_verify.py~~ | ✅ Done |
| Add helper functions | ⏳ Optional |
| Derive old dicts from _MANIFEST | ⏳ Optional |
| Update infra_print.py | ⏳ Optional |
| Update monitor files | ⏳ Optional |

---

**END OF PLAN**
