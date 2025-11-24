# Centralize ALL Infrastructure to gcp-manifest.json

**Date**: 2025-11-21
**Goal**: Make gcp-manifest.json the SINGLE SOURCE OF TRUTH for ALL infrastructure config
**Status**: Planning phase - ready for implementation
**Current Problem**: GPU types and regions scattered across 3+ files, causing inconsistency

---

## 🎯 The Vision: One Canonical Manifest

**Current state** (scattered across codebase):
```
CLI/shared/quota/gpu_quota.py         → GPU_QUOTA_METRICS (10 types)
CLI/launch/mecha/mecha_regions.py     → C3_REGIONS (18 regions)
CLI/config/gcp-manifest.json          → Partial list (4 GPU types, 3 C3 regions)
CLI/setup/core.py                     → Hardcoded "us-central1"
```

**Target state** (one canonical source):
```
CLI/config/gcp-manifest.json          → ALL GPU types (10), ALL regions (18), ALL config
                                      ↓
All code imports from manifest        → Guaranteed consistency!
```

---

## 🎯 KEY INSIGHT: Infrastructure Based on MECHA & ZEUS Systems!

**Critical discovery**: Our GPU/C3 infrastructure is BASED AROUND two battle systems:
- **MECHA** = Cloud Build C3 worker pools (18 regions)
- **ZEUS** = Vertex AI GPU regions (5 tiers, 5 GPU types)

**Manifest should reflect what MECHA and ZEUS actually use**, not all possible options!

---

## 📋 What Needs to Move INTO gcp-manifest.json

### 1. ZEUS Thunder Tiers (5 GPU types)

**Source**: `CLI/launch/zeus/zeus_battle.py:THUNDER_TIERS`

**⚠️ ZEUS ONLY SUPPORTS 5 OF 10 GPU TYPES!**

**ZEUS-supported types** (have pricing battles):
- T4 (spark tier) ✅
- L4 (bolt tier) ✅
- A100 40GB (storm tier) ✅
- H100 80GB (tempest tier) ✅
- H200 (cataclysm tier) ✅

**NOT supported by ZEUS** (no tiers, manual selection):
- A100 80GB ❌
- H100 standard ❌
- V100 ❌ (legacy)
- P4 ❌ (legacy)
- P100 ❌ (legacy)

**Manifest strategy**:
- Primary section: ZEUS tiers (5 types with pricing battles)
- Secondary section: Non-ZEUS GPUs (5 types, manual selection)

**Need to add**: Complete GPU definitions with:
- GPU name (e.g., "NVIDIA_TESLA_A100")
- Display name (e.g., "A100")
- Metric base (e.g., "nvidia_a100_gpus")
- On-demand metric (full path)
- Spot metric (full path)
- Generation (current/previous/budget/legacy)
- Memory (if applicable)
- Cost tier
- Recommended use case

### 2. MECHA C3 Regions (18 regions)

**Source**: `CLI/launch/mecha/mecha_regions.py:C3_REGIONS`

**MECHA tracks ALL 18 C3 regions globally!**

**Current manifest** (3 regions) - INCOMPLETE:
- us-west2 ✓ (MECHA primary)
- us-central1 ✓
- europe-west2 ✓ (MECHA secondary)

**Missing from manifest** (15 regions):
- **US**: us-east1, us-east4, us-east5, us-west1, us-west3, us-west4 (6 regions)
- **North America**: northamerica-northeast1 (1 region)
- **Europe**: europe-west1, europe-west3, europe-west4, europe-west9 (4 regions)
- **Asia**: asia-northeast1, asia-southeast1 (2 regions)
- **Australia**: australia-southeast1 (1 region)
- **South America**: southamerica-east1 (1 region)

**Manifest strategy**:
- Use descriptive naming: `mecha_c3_cloudbuild_regions` (not just "mecha_regions")
- Include ALL 18 regions (MECHA's full fleet)
- Include MECHA status tracking (RUNNING, CREATING, FATIGUED, EXHAUSTED)

**Need to add**: Complete region definitions with:
- Region code (e.g., "us-central1")
- Location name (e.g., "Iowa, USA")
- Zones (e.g., ["a", "b", "c", "f"])
- Continent
- Latency tier (low/medium/high)
- Preferred order
- Cost multiplier (if different)

### 3. Strategic GPU Check Regions

**Current manifest** (1 region):
- us-central1

**Should have** (4-5 strategic regions):
- us-central1 (central US, current)
- us-east4 (east coast)
- us-west1 (west coast)
- europe-west1 (Europe primary)
- asia-northeast1 (Asia, optional)

### 4. Region Groupings

**Need to add**:
- US regions list (8 regions)
- Europe regions list (5 regions)
- Asia regions list (2 regions)
- Preferred regions by continent
- Fallback regions

### 5. Machine Type Mappings

**Currently scattered**:
- C3 machine types
- GPU machine type requirements (A100 → A2, H100 → A3, L4 → G2)
- vCPU counts
- Disk sizes

**Need centralized**:
- C3 machines: c3-standard-4, c3-standard-8, ..., c3-standard-176
- A2 machines: a2-highgpu-1g, a2-ultragpu-1g (for A100)
- A3 machines: a3-highgpu-8g (for H100/H200)
- G2 machines: g2-standard-4, g2-standard-8 (for L4)

---

## 🏗️ Implementation Plan

### Phase 1: Expand gcp-manifest.json (Complete GPU/Region Definitions)

**File**: `CLI/config/gcp-manifest.json`

**Step 1.1: Add ZEUS Thunder Tiers (5 GPU types) + Non-ZEUS types (5 types)**

```json
{
  "zeus_thunder_gpu_tiers": {
    "description": "ZEUS Thunder Battle System - 5 GPU tiers with automated pricing battles",
    "purpose": "ZEUS discovers quotas via snapshot, runs thunder battles to select cheapest region",
    "total_tiers": 5,
    "registry_file": "CLI/launch/zeus/data/zeus_olympus.json",
    "tiers": [
      {
        "tier_name": "spark",
        "tier_emoji": "⚡",
        "internal_name": "NVIDIA_TESLA_T4",
        "display_name": "T4",
        "generation": "budget",
        "metric_base": "nvidia_t4_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus",
        "memory_gb": 16,
        "cost_tier": "low",
        "use_case": "Entry-level, cheapest option, light training",
        "machine_family": "n1",
        "recommended": false
      },
      {
        "tier_name": "bolt",
        "tier_emoji": "⚡⚡",
        "internal_name": "NVIDIA_L4",
        "display_name": "L4",
        "generation": "budget",
        "metric_base": "nvidia_l4_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_l4_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_l4_gpus",
        "memory_gb": 24,
        "cost_tier": "medium",
        "use_case": "Best cost/performance, inference, small training",
        "machine_family": "g2",
        "recommended": true
      },
      {
        "tier_name": "storm",
        "tier_emoji": "⚡⚡⚡",
        "internal_name": "NVIDIA_TESLA_A100",
        "display_name": "A100",
        "generation": "previous",
        "metric_base": "nvidia_a100_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_a100_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_a100_gpus",
        "memory_gb": 40,
        "cost_tier": "high",
        "use_case": "Medium-large training jobs, proven hardware",
        "machine_family": "a2",
        "recommended": true
      },
      {
        "tier_name": "tempest",
        "tier_emoji": "⚡⚡⚡⚡",
        "internal_name": "NVIDIA_H100_80GB",
        "display_name": "H100 80GB",
        "generation": "current",
        "metric_base": "nvidia_h100_80gb_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_h100_80gb_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_80gb_gpus",
        "memory_gb": 80,
        "cost_tier": "premium",
        "use_case": "Large training jobs with high memory requirements",
        "machine_family": "a3",
        "recommended": true
      },
      {
        "tier_name": "cataclysm",
        "tier_emoji": "⚡⚡⚡⚡⚡",
        "internal_name": "NVIDIA_H200",
        "display_name": "H200",
        "generation": "current",
        "metric_base": "nvidia_h200_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_h200_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h200_gpus",
        "memory_gb": 141,
        "cost_tier": "premium",
        "use_case": "Largest training jobs, bleeding edge",
        "machine_family": "a3",
        "recommended": true
      }
    ]
  },

  "non_zeus_gpu_types": {
    "description": "GPU types without ZEUS tier support (manual region selection required)",
    "purpose": "Users can request these quotas but must manually select regions (no pricing battles)",
    "total_types": 5,
    "types": [
      {
        "internal_name": "NVIDIA_A100_80GB",
        "display_name": "A100 80GB",
        "generation": "previous",
        "metric_base": "nvidia_a100_80gb_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_a100_80gb_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_a100_80gb_gpus",
        "memory_gb": 80,
        "cost_tier": "high",
        "use_case": "Medium-large training with high memory (no ZEUS tier)",
        "machine_family": "a2",
        "recommended": false,
        "note": "Not in ZEUS tiers - manual region selection only"
      },
      {
        "internal_name": "NVIDIA_H100",
        "display_name": "H100",
        "generation": "current",
        "metric_base": "nvidia_h100_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus",
        "memory_gb": null,
        "cost_tier": "premium",
        "use_case": "Large training jobs (standard H100, no ZEUS tier)",
        "machine_family": "a3",
        "recommended": false,
        "note": "Not in ZEUS tiers - use H100 80GB (tempest tier) instead"
      },
      {
        "internal_name": "NVIDIA_H200",
        "display_name": "H200",
        "generation": "current",
        "metric_base": "nvidia_h200_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_h200_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h200_gpus",
        "memory_gb": null,
        "cost_tier": "premium",
        "use_case": "Largest training jobs, bleeding edge",
        "machine_family": "a3",
        "recommended": true
      },
      {
        "internal_name": "NVIDIA_H100",
        "display_name": "H100",
        "generation": "current",
        "metric_base": "nvidia_h100_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus",
        "memory_gb": null,
        "cost_tier": "premium",
        "use_case": "Large training jobs, high performance",
        "machine_family": "a3",
        "recommended": true
      },
      {
        "internal_name": "NVIDIA_H100_80GB",
        "display_name": "H100 80GB",
        "generation": "current",
        "metric_base": "nvidia_h100_80gb_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_h100_80gb_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_80gb_gpus",
        "memory_gb": 80,
        "cost_tier": "premium",
        "use_case": "Large training jobs with high memory requirements",
        "machine_family": "a3",
        "recommended": true
      },
      {
        "internal_name": "NVIDIA_TESLA_A100",
        "display_name": "A100",
        "generation": "previous",
        "metric_base": "nvidia_a100_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_a100_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_a100_gpus",
        "memory_gb": 40,
        "cost_tier": "high",
        "use_case": "Medium-large training jobs, proven hardware",
        "machine_family": "a2",
        "recommended": true
      },
      {
        "internal_name": "NVIDIA_A100_80GB",
        "display_name": "A100 80GB",
        "generation": "previous",
        "metric_base": "nvidia_a100_80gb_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_a100_80gb_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_a100_80gb_gpus",
        "memory_gb": 80,
        "cost_tier": "high",
        "use_case": "Medium-large training with high memory",
        "machine_family": "a2",
        "recommended": true
      },
      {
        "internal_name": "NVIDIA_L4",
        "display_name": "L4",
        "generation": "budget",
        "metric_base": "nvidia_l4_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_l4_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_l4_gpus",
        "memory_gb": 24,
        "cost_tier": "medium",
        "use_case": "Best cost/performance, inference, small training",
        "machine_family": "g2",
        "recommended": true
      },
      {
        "internal_name": "NVIDIA_TESLA_T4",
        "display_name": "T4",
        "generation": "budget",
        "metric_base": "nvidia_t4_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus",
        "memory_gb": 16,
        "cost_tier": "low",
        "use_case": "Entry-level, cheapest option, light training",
        "machine_family": "n1",
        "recommended": false
      },
      {
        "internal_name": "NVIDIA_TESLA_V100",
        "display_name": "V100",
        "generation": "legacy",
        "metric_base": "nvidia_v100_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_v100_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_v100_gpus",
        "memory_gb": 16,
        "cost_tier": "medium",
        "use_case": "Legacy workloads, compatibility",
        "machine_family": "n1",
        "recommended": false
      },
      {
        "internal_name": "NVIDIA_TESLA_P4",
        "display_name": "P4",
        "generation": "legacy",
        "metric_base": "nvidia_p4_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_p4_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_p4_gpus",
        "memory_gb": 8,
        "cost_tier": "low",
        "use_case": "Legacy inference",
        "machine_family": "n1",
        "recommended": false
      },
      {
        "internal_name": "NVIDIA_TESLA_P100",
        "display_name": "P100",
        "generation": "legacy",
        "metric_base": "nvidia_p100_gpus",
        "metric_ondemand": "aiplatform.googleapis.com/custom_model_training_nvidia_p100_gpus",
        "metric_spot": "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_p100_gpus",
        "memory_gb": 16,
        "cost_tier": "low",
        "use_case": "Legacy training",
        "machine_family": "n1",
        "recommended": false
      }
    ]
  }
}
```

**Step 1.2: Add ALL 18 MECHA C3 regions**

```json
{
  "mecha_c3_cloudbuild_regions": {
    "description": "MECHA Battle System - All 18 C3 Cloud Build worker pool regions",
    "purpose": "MECHA progressively acquires these regions, runs pricing battles to select cheapest",
    "total_count": 18,
    "machine_type_default": "c3-standard-176",
    "registry_file": "CLI/launch/mecha/data/mecha_hangar.json",
    "regions": [
      {
        "code": "us-central1",
        "location": "Iowa, USA",
        "zones": ["a", "b", "c", "f"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 1
      },
      {
        "code": "us-east1",
        "location": "South Carolina, USA",
        "zones": ["b", "c", "d"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 3
      },
      {
        "code": "us-east4",
        "location": "Northern Virginia, USA",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 2
      },
      {
        "code": "us-east5",
        "location": "Columbus, Ohio, USA",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 4
      },
      {
        "code": "us-west1",
        "location": "Oregon, USA",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 5
      },
      {
        "code": "us-west2",
        "location": "Los Angeles, USA",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 6,
        "notes": "MECHA primary region"
      },
      {
        "code": "us-west3",
        "location": "Salt Lake City, USA",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 7
      },
      {
        "code": "us-west4",
        "location": "Las Vegas, USA",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "USA",
        "latency_tier": "low",
        "preference_order": 8
      },
      {
        "code": "northamerica-northeast1",
        "location": "Montreal, Canada",
        "zones": ["a", "b", "c"],
        "continent": "north_america",
        "country": "Canada",
        "latency_tier": "low",
        "preference_order": 9
      },
      {
        "code": "europe-west1",
        "location": "Belgium",
        "zones": ["b", "c", "d"],
        "continent": "europe",
        "country": "Belgium",
        "latency_tier": "medium",
        "preference_order": 10
      },
      {
        "code": "europe-west2",
        "location": "London, UK",
        "zones": ["a", "b", "c"],
        "continent": "europe",
        "country": "UK",
        "latency_tier": "medium",
        "preference_order": 11,
        "notes": "MECHA secondary region"
      },
      {
        "code": "europe-west3",
        "location": "Frankfurt, Germany",
        "zones": ["a", "b", "c"],
        "continent": "europe",
        "country": "Germany",
        "latency_tier": "medium",
        "preference_order": 12
      },
      {
        "code": "europe-west4",
        "location": "Netherlands",
        "zones": ["a", "b", "c"],
        "continent": "europe",
        "country": "Netherlands",
        "latency_tier": "medium",
        "preference_order": 13
      },
      {
        "code": "europe-west9",
        "location": "Paris, France",
        "zones": ["a", "b", "c"],
        "continent": "europe",
        "country": "France",
        "latency_tier": "medium",
        "preference_order": 14
      },
      {
        "code": "asia-northeast1",
        "location": "Tokyo, Japan",
        "zones": ["a", "b", "c"],
        "continent": "asia",
        "country": "Japan",
        "latency_tier": "high",
        "preference_order": 15
      },
      {
        "code": "asia-southeast1",
        "location": "Singapore",
        "zones": ["a", "b", "c"],
        "continent": "asia",
        "country": "Singapore",
        "latency_tier": "high",
        "preference_order": 16
      },
      {
        "code": "australia-southeast1",
        "location": "Sydney, Australia",
        "zones": ["a", "b", "c"],
        "continent": "australia",
        "country": "Australia",
        "latency_tier": "high",
        "preference_order": 17
      },
      {
        "code": "southamerica-east1",
        "location": "São Paulo, Brazil",
        "zones": ["a", "b", "c"],
        "continent": "south_america",
        "country": "Brazil",
        "latency_tier": "medium-high",
        "preference_order": 18
      }
    ]
  }
}
```

**Step 1.3: Add infrastructure regional groupings**

```json
{
  "infrastructure_regional_groups": {
    "description": "Logical groupings of regions for infrastructure organization and filtering",
    "purpose": "Used by infra display, setup checks, and MECHA/ZEUS systems for region selection",
    "continent_us_regions": [
      "us-central1", "us-east1", "us-east4", "us-east5",
      "us-west1", "us-west2", "us-west3", "us-west4"
    ],
    "continent_north_america_regions": [
      "us-central1", "us-east1", "us-east4", "us-east5",
      "us-west1", "us-west2", "us-west3", "us-west4",
      "northamerica-northeast1"
    ],
    "continent_europe_regions": [
      "europe-west1", "europe-west2", "europe-west3",
      "europe-west4", "europe-west9"
    ],
    "continent_asia_regions": [
      "asia-northeast1", "asia-southeast1"
    ],
    "continent_australia_regions": [
      "australia-southeast1"
    ],
    "continent_south_america_regions": [
      "southamerica-east1"
    ],
    "performance_low_latency_regions": [
      "us-central1", "us-east1", "us-east4", "us-east5",
      "us-west1", "us-west2", "us-west3", "us-west4",
      "northamerica-northeast1"
    ],
    "mecha_preferred_deployment_regions": [
      "us-west2", "us-central1", "europe-west2"
    ],
    "infra_strategic_gpu_check_regions": [
      "us-central1", "us-east4", "us-west1", "europe-west1"
    ]
  }
}
```

**Step 1.4: Add GCP machine family definitions**

```json
{
  "gcp_machine_family_definitions": {
    "description": "Machine type configurations, GPU family mappings, and vCPU/memory specs",
    "purpose": "Used by MECHA for Cloud Build pools, ZEUS for Vertex AI job submissions",
    "mecha_cloudbuild_c3_machines": {
      "family": "c3",
      "system": "MECHA",
      "description": "Compute-optimized C3 machines for Cloud Build worker pools",
      "machines": [
        {"name": "c3-standard-4", "vcpus": 4, "memory_gb": 16},
        {"name": "c3-standard-8", "vcpus": 8, "memory_gb": 32},
        {"name": "c3-standard-22", "vcpus": 22, "memory_gb": 88},
        {"name": "c3-standard-44", "vcpus": 44, "memory_gb": 176},
        {"name": "c3-standard-88", "vcpus": 88, "memory_gb": 352},
        {"name": "c3-standard-176", "vcpus": 176, "memory_gb": 704}
      ],
      "mecha_default_machine": "c3-standard-176"
    },
    "zeus_vertexai_a2_machines": {
      "family": "a2",
      "system": "ZEUS",
      "zeus_tiers": ["storm"],
      "description": "A100 GPU machines for ZEUS storm tier (pre-attached GPUs)",
      "required_for_gpus": ["NVIDIA_TESLA_A100", "NVIDIA_A100_80GB"],
      "machines": [
        {"name": "a2-highgpu-1g", "vcpus": 12, "memory_gb": 85, "gpus": 1, "gpu_type": "A100-40GB"},
        {"name": "a2-highgpu-2g", "vcpus": 24, "memory_gb": 170, "gpus": 2, "gpu_type": "A100-40GB"},
        {"name": "a2-highgpu-4g", "vcpus": 48, "memory_gb": 340, "gpus": 4, "gpu_type": "A100-40GB"},
        {"name": "a2-highgpu-8g", "vcpus": 96, "memory_gb": 680, "gpus": 8, "gpu_type": "A100-40GB"},
        {"name": "a2-megagpu-16g", "vcpus": 96, "memory_gb": 1360, "gpus": 16, "gpu_type": "A100-40GB"},
        {"name": "a2-ultragpu-1g", "vcpus": 12, "memory_gb": 170, "gpus": 1, "gpu_type": "A100-80GB"},
        {"name": "a2-ultragpu-2g", "vcpus": 24, "memory_gb": 340, "gpus": 2, "gpu_type": "A100-80GB"},
        {"name": "a2-ultragpu-4g", "vcpus": 48, "memory_gb": 680, "gpus": 4, "gpu_type": "A100-80GB"},
        {"name": "a2-ultragpu-8g", "vcpus": 96, "memory_gb": 1360, "gpus": 8, "gpu_type": "A100-80GB"}
      ]
    },
    "zeus_vertexai_a3_machines": {
      "family": "a3",
      "system": "ZEUS",
      "zeus_tiers": ["tempest", "cataclysm"],
      "description": "H100/H200 GPU machines for ZEUS tempest/cataclysm tiers (pre-attached GPUs)",
      "required_for_gpus": ["NVIDIA_H100", "NVIDIA_H100_80GB", "NVIDIA_H200"],
      "machines": [
        {"name": "a3-highgpu-8g", "vcpus": 208, "memory_gb": 1872, "gpus": 8, "gpu_type": "H100-80GB"}
      ]
    },
    "zeus_vertexai_g2_machines": {
      "family": "g2",
      "system": "ZEUS",
      "zeus_tiers": ["bolt"],
      "description": "L4 GPU machines for ZEUS bolt tier",
      "required_for_gpus": ["NVIDIA_L4"],
      "machines": [
        {"name": "g2-standard-4", "vcpus": 4, "memory_gb": 16, "gpus": 1},
        {"name": "g2-standard-8", "vcpus": 8, "memory_gb": 32, "gpus": 1},
        {"name": "g2-standard-12", "vcpus": 12, "memory_gb": 48, "gpus": 1},
        {"name": "g2-standard-16", "vcpus": 16, "memory_gb": 64, "gpus": 1},
        {"name": "g2-standard-24", "vcpus": 24, "memory_gb": 96, "gpus": 2},
        {"name": "g2-standard-32", "vcpus": 32, "memory_gb": 128, "gpus": 1},
        {"name": "g2-standard-48", "vcpus": 48, "memory_gb": 192, "gpus": 4},
        {"name": "g2-standard-96", "vcpus": 96, "memory_gb": 384, "gpus": 8}
      ]
    },
    "zeus_vertexai_n1_machines": {
      "family": "n1",
      "system": "ZEUS",
      "zeus_tiers": ["spark"],
      "description": "General purpose N1 machines for ZEUS spark tier (attach T4/V100/P4/P100)",
      "required_for_gpus": ["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_P4", "NVIDIA_TESLA_P100"],
      "notes": "GPUs attached separately, flexible vCPU/memory"
    }
  }
}
```

---

### Phase 2: Create Manifest Loader Module

**File**: `CLI/config/manifest_loader.py` (NEW!)

**Purpose**: Centralized module to load and validate gcp-manifest.json

```python
"""
GCP Manifest Loader

SINGLE SOURCE OF TRUTH for all GCP infrastructure configuration.
All modules import from this loader instead of hardcoded dicts.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

# Path to manifest
MANIFEST_PATH = Path(__file__).parent / "gcp-manifest.json"


class ManifestLoader:
    """Loads and validates gcp-manifest.json"""

    def __init__(self):
        self._manifest = None

    @property
    def manifest(self) -> Dict:
        """Load manifest (cached)"""
        if self._manifest is None:
            with open(MANIFEST_PATH, 'r') as f:
                self._manifest = json.load(f)
        return self._manifest

    # ═══════════════════════════════════════════════════════════════════════
    # GPU TYPES
    # ═══════════════════════════════════════════════════════════════════════

    def get_all_gpu_types(self) -> List[Dict]:
        """Get all GPU type definitions (10 types)"""
        return self.manifest["gpu_types"]["types"]

    def get_gpu_type(self, internal_name: str) -> Optional[Dict]:
        """Get single GPU type by internal name"""
        for gpu in self.get_all_gpu_types():
            if gpu["internal_name"] == internal_name:
                return gpu
        return None

    def get_gpu_quota_metrics(self) -> Dict[str, str]:
        """
        Get GPU_QUOTA_METRICS dict (for backward compatibility)

        Returns:
            Dict mapping internal_name → metric_base
            Example: {"NVIDIA_TESLA_T4": "nvidia_t4_gpus"}
        """
        return {
            gpu["internal_name"]: gpu["metric_base"]
            for gpu in self.get_all_gpu_types()
        }

    def get_recommended_gpus(self) -> List[Dict]:
        """Get only recommended GPU types"""
        return [gpu for gpu in self.get_all_gpu_types() if gpu.get("recommended", False)]

    def get_gpus_by_generation(self, generation: str) -> List[Dict]:
        """Get GPUs by generation (current/previous/budget/legacy)"""
        return [gpu for gpu in self.get_all_gpu_types() if gpu["generation"] == generation]

    # ═══════════════════════════════════════════════════════════════════════
    # C3 REGIONS
    # ═══════════════════════════════════════════════════════════════════════

    def get_all_c3_regions(self) -> List[Dict]:
        """Get all C3 region definitions (18 regions)"""
        return self.manifest["c3_regions"]["regions"]

    def get_c3_region(self, region_code: str) -> Optional[Dict]:
        """Get single C3 region by code"""
        for region in self.get_all_c3_regions():
            if region["code"] == region_code:
                return region
        return None

    def get_all_c3_region_codes(self) -> List[str]:
        """
        Get simple list of C3 region codes (for backward compatibility)

        Returns:
            List of region codes
            Example: ["us-central1", "us-west2", ...]
        """
        return [region["code"] for region in self.get_all_c3_regions()]

    def get_c3_regions_dict(self) -> Dict[str, Dict]:
        """
        Get C3_REGIONS dict (for backward compatibility with mecha_regions.py)

        Returns:
            Dict mapping region_code → region_info
            Example: {"us-central1": {"location": "Iowa, USA", ...}}
        """
        return {
            region["code"]: {
                "location": region["location"],
                "zones": region["zones"],
                "latency_to_us": region["latency_tier"],
            }
            for region in self.get_all_c3_regions()
        }

    # ═══════════════════════════════════════════════════════════════════════
    # REGION GROUPS
    # ═══════════════════════════════════════════════════════════════════════

    def get_region_group(self, group_name: str) -> List[str]:
        """
        Get region group by name

        Args:
            group_name: us_regions, europe_regions, asia_regions,
                       mecha_primary_regions, strategic_gpu_regions, etc.

        Returns:
            List of region codes
        """
        return self.manifest["region_groups"].get(group_name, [])

    def get_mecha_primary_regions(self) -> List[str]:
        """Get MECHA primary regions (3 regions)"""
        return self.get_region_group("mecha_primary_regions")

    def get_strategic_gpu_regions(self) -> List[str]:
        """Get strategic GPU check regions (4 regions)"""
        return self.get_region_group("strategic_gpu_regions")

    # ═══════════════════════════════════════════════════════════════════════
    # MACHINE TYPES
    # ═══════════════════════════════════════════════════════════════════════

    def get_machine_family(self, family: str) -> Dict:
        """Get machine family config (c3, a2, a3, g2, n1)"""
        return self.manifest["machine_types"].get(family, {})

    def get_gpu_machine_family(self, gpu_internal_name: str) -> str:
        """
        Get required machine family for GPU type

        Args:
            gpu_internal_name: e.g., "NVIDIA_TESLA_A100"

        Returns:
            Machine family: "a2", "a3", "g2", "n1"
        """
        gpu = self.get_gpu_type(gpu_internal_name)
        if gpu:
            return gpu.get("machine_family", "n1")
        return "n1"


# Singleton instance
_loader = ManifestLoader()


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API (for easy imports)
# ═══════════════════════════════════════════════════════════════════════

def load_manifest() -> Dict:
    """Load full manifest"""
    return _loader.manifest


def get_all_gpu_types() -> List[Dict]:
    """Get all GPU types"""
    return _loader.get_all_gpu_types()


def get_gpu_quota_metrics() -> Dict[str, str]:
    """Get GPU_QUOTA_METRICS (backward compatibility)"""
    return _loader.get_gpu_quota_metrics()


def get_all_c3_region_codes() -> List[str]:
    """Get all C3 region codes (backward compatibility)"""
    return _loader.get_all_c3_region_codes()


def get_c3_regions_dict() -> Dict[str, Dict]:
    """Get C3_REGIONS dict (backward compatibility)"""
    return _loader.get_c3_regions_dict()


def get_mecha_primary_regions() -> List[str]:
    """Get MECHA primary regions"""
    return _loader.get_mecha_primary_regions()


def get_strategic_gpu_regions() -> List[str]:
    """Get strategic GPU check regions"""
    return _loader.get_strategic_gpu_regions()


# Export singleton for advanced usage
manifest_loader = _loader
```

---

### Phase 3: Update All Importing Code

**Step 3.1: Update `CLI/shared/quota/gpu_quota.py`**

```python
# OLD (lines 53-64):
GPU_QUOTA_METRICS = {
    "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
    "NVIDIA_L4": "nvidia_l4_gpus",
    # ... 8 more ...
}

# NEW (replace with manifest import):
from ...config.manifest_loader import get_gpu_quota_metrics
GPU_QUOTA_METRICS = get_gpu_quota_metrics()
```

**Step 3.2: Update `CLI/launch/mecha/mecha_regions.py`**

```python
# OLD (lines 5-143):
C3_REGIONS = {
    "us-central1": {
        "location": "Iowa, USA",
        # ... 17 more regions ...
    }
}
ALL_MECHA_REGIONS = list(C3_REGIONS.keys())

# NEW (replace with manifest import):
from ...config.manifest_loader import get_c3_regions_dict, get_all_c3_region_codes

C3_REGIONS = get_c3_regions_dict()
ALL_MECHA_REGIONS = get_all_c3_region_codes()
```

**Step 3.3: Update `CLI/shared/infra_verify.py`**

```python
# OLD (lines 600-624):
gpu_check_regions = manifest["regions"]["gpu_quota_regions"]["regions"]
c3_check_regions = manifest["regions"]["mecha_regions"]["regions"]

# NEW (import from loader):
from .config.manifest_loader import get_strategic_gpu_regions, get_all_c3_region_codes

gpu_check_regions = get_strategic_gpu_regions()  # 4 strategic regions
c3_check_regions = get_all_c3_region_codes()     # All 18 regions!
```

**Step 3.4: Update `CLI/shared/infra_print.py`**

```python
# Add filtering for displayed GPUs (only known types)
from ..config.manifest_loader import get_all_gpu_types

# Get known GPU names from manifest
known_gpu_names = {gpu["display_name"] for gpu in get_all_gpu_types()}

# Filter pending GPUs to only known types
pending_filtered = [
    gpu for gpu in pending
    if gpu.get("gpu_name") in known_gpu_names
]
```

**Files to update** (grep results):
- `CLI/shared/quota/gpu_quota.py` ✓
- `CLI/launch/mecha/mecha_regions.py` ✓
- `CLI/shared/infra_verify.py` ✓
- `CLI/shared/infra_print.py` ✓
- `CLI/monitor/screen.py` (uses ALL_MECHA_REGIONS)
- `CLI/monitor/core.py` (uses ALL_MECHA_REGIONS)
- `CLI/launch/mecha/mecha_integration.py` (uses ALL_MECHA_REGIONS)
- `CLI/launch/runner_cleanup.py` (uses ALL_MECHA_REGIONS)
- `CLI/setup/core.py` (hardcoded "us-central1")

---

### Phase 4: Add Manifest Version & Validation

**Step 4.1: Add schema validation**

```python
# CLI/config/manifest_loader.py

def validate_manifest(manifest: Dict) -> None:
    """Validate manifest structure"""
    required_keys = [
        "version",
        "gpu_types",
        "c3_regions",
        "region_groups",
        "machine_types"
    ]

    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"Missing required key in manifest: {key}")

    # Validate GPU types
    gpu_types = manifest["gpu_types"]["types"]
    if len(gpu_types) != 10:
        raise ValueError(f"Expected 10 GPU types, found {len(gpu_types)}")

    # Validate C3 regions
    c3_regions = manifest["c3_regions"]["regions"]
    if len(c3_regions) != 18:
        raise ValueError(f"Expected 18 C3 regions, found {len(c3_regions)}")

    print(f"✓ Manifest v{manifest['version']} validated successfully")
    print(f"  - {len(gpu_types)} GPU types")
    print(f"  - {len(c3_regions)} C3 regions")
```

**Step 4.2: Update manifest version**

```json
{
  "version": "2.0.0",
  "description": "Canonical GCP infrastructure manifest - COMPLETE inventory",
  "last_updated": "2025-11-21",
  "changelog": {
    "2.0.0": "Added ALL 10 GPU types, ALL 18 C3 regions, region groups, machine types",
    "1.0.0": "Initial version with partial inventory (4 GPU types, 3 C3 regions)"
  }
}
```

---

### Phase 5: Testing & Verification

**Step 5.1: Unit tests**

Create `CLI/config/test_manifest_loader.py`:

```python
import unittest
from CLI.config.manifest_loader import (
    load_manifest,
    get_all_gpu_types,
    get_gpu_quota_metrics,
    get_all_c3_region_codes,
    get_mecha_primary_regions,
)

class TestManifestLoader(unittest.TestCase):

    def test_load_manifest(self):
        """Manifest loads successfully"""
        manifest = load_manifest()
        self.assertIsInstance(manifest, dict)
        self.assertIn("version", manifest)

    def test_all_gpu_types_count(self):
        """All 10 GPU types present"""
        gpus = get_all_gpu_types()
        self.assertEqual(len(gpus), 10)

    def test_all_c3_regions_count(self):
        """All 18 C3 regions present"""
        regions = get_all_c3_region_codes()
        self.assertEqual(len(regions), 18)

    def test_gpu_quota_metrics_backward_compat(self):
        """GPU_QUOTA_METRICS format preserved"""
        metrics = get_gpu_quota_metrics()
        self.assertIn("NVIDIA_TESLA_T4", metrics)
        self.assertEqual(metrics["NVIDIA_TESLA_T4"], "nvidia_t4_gpus")

    def test_mecha_primary_regions(self):
        """MECHA primary regions correct"""
        regions = get_mecha_primary_regions()
        self.assertEqual(len(regions), 3)
        self.assertIn("us-west2", regions)
        self.assertIn("us-central1", regions)
        self.assertIn("europe-west2", regions)
```

**Step 5.2: Integration tests**

```bash
# Test infrastructure display
python CLI/cli.py infra

# Should now show:
# - All 10 GPU types in "Need to request"
# - All 18 C3 regions (or at least more than 3!)
# - Strategic GPU regions (4 regions checked)

# Test MECHA
python CLI/cli.py launch --check-only

# Should work without changes (backward compatible!)

# Test monitor
python CLI/cli.py monitor --vertex-runner

# Should work with all 18 regions!
```

---

## 📊 Migration Checklist

### Phase 1: Expand Manifest ✓
- [ ] Add all 10 GPU types to gcp-manifest.json
- [ ] Add all 18 C3 regions to gcp-manifest.json
- [ ] Add region groupings
- [ ] Add machine type mappings
- [ ] Update version to 2.0.0

### Phase 2: Create Loader ✓
- [ ] Create `CLI/config/manifest_loader.py`
- [ ] Add ManifestLoader class
- [ ] Add backward-compatibility functions
- [ ] Add validation

### Phase 3: Update Imports ✓
- [ ] Update `gpu_quota.py` (GPU_QUOTA_METRICS)
- [ ] Update `mecha_regions.py` (C3_REGIONS, ALL_MECHA_REGIONS)
- [ ] Update `infra_verify.py` (region checks)
- [ ] Update `infra_print.py` (display filtering)
- [ ] Update `monitor/screen.py` (ALL_MECHA_REGIONS)
- [ ] Update `monitor/core.py` (ALL_MECHA_REGIONS)
- [ ] Update `mecha_integration.py` (ALL_MECHA_REGIONS)
- [ ] Update `runner_cleanup.py` (ALL_MECHA_REGIONS)
- [ ] Update `setup/core.py` (hardcoded us-central1)

### Phase 4: Testing ✓
- [ ] Write unit tests for manifest_loader
- [ ] Test infrastructure display (all GPU types, all regions)
- [ ] Test MECHA (backward compatibility)
- [ ] Test monitor (multi-region)
- [ ] Test setup (no hardcoded regions)

### Phase 5: Documentation ✓
- [ ] Update CLAUDE.md with manifest usage
- [ ] Add docstrings to manifest_loader
- [ ] Update infrastructure docs
- [ ] Git commit with detailed message

---

## 🎯 Success Criteria

**When complete, infrastructure display should show**:

```
🎮 GPU Quotas (Vertex AI):

  us-central1:
    ✓ T4 (Spot): 1 GPU
    ⚠️ Need to request:
      • H200
      • H100
      • H100 80GB
      • A100            ← FROM MANIFEST!
      • A100 80GB       ← FROM MANIFEST!
      • L4
      • V100            ← FROM MANIFEST!
      • P4              ← FROM MANIFEST!
      • P100            ← FROM MANIFEST!

  us-east4:
    (similar breakdown)

  us-west1:
    (similar breakdown)

  europe-west1:
    (similar breakdown)

☁️ C3 Quotas (Cloud Build):
  ✓ us-west2: 176 vCPUs (c3-standard-176)
  ✓ us-central1: 176 vCPUs (c3-standard-176)
  ⚠️ Need to request (16 more regions):    ← ALL 18 REGIONS!
    • us-east1, us-east4, us-east5
    • us-west1, us-west3, us-west4
    • northamerica-northeast1
    • europe-west1, europe-west2, europe-west3, europe-west4, europe-west9
    • asia-northeast1, asia-southeast1
    • australia-southeast1
    • southamerica-east1
```

**Code should**:
- ✅ Import from manifest_loader (not hardcoded dicts)
- ✅ Show ALL 10 GPU types (not 4)
- ✅ Show ALL 18 C3 regions (not 3)
- ✅ Check 4 strategic GPU regions (not 1)
- ✅ Filter unknown GPU types (no "B200"!)
- ✅ Maintain backward compatibility (existing code works!)

---

## 💡 Benefits

**Single source of truth**:
- One place to add new GPU types
- One place to add new regions
- No duplication or inconsistency

**Easier maintenance**:
- Update manifest JSON → all code updated!
- No grep-and-replace across 10 files
- Version tracked changes

**Better visibility**:
- See ALL options (not partial list)
- Regional comparison
- Strategic region selection

**Backward compatible**:
- Existing code keeps working
- Gradual migration possible
- No breaking changes

---

**END OF PLAN** - Ready for implementation! 🚀
