# Complete Infrastructure Inventory

**Date**: 2025-11-21
**Purpose**: Comprehensive inventory of ALL GPU types and C3 regions supported by arr-coc-0-1
**Status**: DISCOVERED - Our codebase supports WAY more than the manifest!

---

## 🎮 GPU Types (Vertex AI Custom Training)

**Location**: `CLI/shared/quota/gpu_quota.py:53-64`

### Supported GPU Types (10 total!)

Our codebase supports **10 GPU types** across 4 generations:

#### Current Generation (H200, H100)
1. **NVIDIA_H200** - `nvidia_h200_gpus`
   - Newest Hopper GPU (2024)
   - Highest performance

2. **NVIDIA_H100** - `nvidia_h100_gpus`
   - Standard H100

3. **NVIDIA_H100_80GB** - `nvidia_h100_80gb_gpus`
   - High memory variant

#### Previous Generation (A100)
4. **NVIDIA_TESLA_A100** - `nvidia_a100_gpus`
   - Standard A100 (40GB)

5. **NVIDIA_A100_80GB** - `nvidia_a100_80gb_gpus`
   - High memory variant (80GB)

#### Budget Options (L4, T4)
6. **NVIDIA_L4** - `nvidia_l4_gpus`
   - Best cost/performance for inference
   - Good for smaller training jobs

7. **NVIDIA_TESLA_T4** - `nvidia_t4_gpus`
   - Entry-level GPU
   - Cheapest option

#### Legacy Options (V100, P4, P100)
8. **NVIDIA_TESLA_V100** - `nvidia_v100_gpus`
   - Previous generation

9. **NVIDIA_TESLA_P4** - `nvidia_p4_gpus`
   - Older inference GPU

10. **NVIDIA_TESLA_P100** - `nvidia_p100_gpus`
    - Older training GPU

### GPU Quota Metrics

Each GPU type has TWO quota metrics:
- **On-demand**: `custom_model_training_nvidia_<type>_gpus`
- **Spot/Preemptible**: `custom_model_training_preemptible_nvidia_<type>_gpus`

**Total metrics**: 10 types × 2 (on-demand + spot) = **20 separate quotas!**

---

## 🚨 CRITICAL: ZEUS Thunder Tiers DON'T Cover All 10 GPU Types!

**Location**: `CLI/launch/zeus/zeus_battle.py:THUNDER_TIERS`

### ZEUS Tiers (5 of 10 GPU types!)

ZEUS only tracks **5 GPU types** across 5 tiers:

1. **spark** (⚡) - NVIDIA_TESLA_T4 (16GB)
2. **bolt** (⚡⚡) - NVIDIA_L4 (24GB)
3. **storm** (⚡⚡⚡) - NVIDIA_TESLA_A100 (40GB) ⚠️ **Only standard A100!**
4. **tempest** (⚡⚡⚡⚡) - NVIDIA_H100_80GB (80GB) ⚠️ **Only 80GB variant!**
5. **cataclysm** (⚡⚡⚡⚡⚡) - NVIDIA_H200 (141GB)

### Missing from ZEUS Tiers (5 GPU types!)

**NOT tracked by ZEUS battle system**:
- ❌ **NVIDIA_A100_80GB** - No tier! (only standard 40GB A100 in storm tier)
- ❌ **NVIDIA_H100** - No tier! (only H100 80GB in tempest tier)
- ❌ **NVIDIA_TESLA_V100** - No tier! (legacy GPU)
- ❌ **NVIDIA_TESLA_P4** - No tier! (legacy GPU)
- ❌ **NVIDIA_TESLA_P100** - No tier! (legacy GPU)

### What This Means

**Users can request quotas for these 5 GPU types**, but:
- ZEUS won't track quota acquisition
- ZEUS won't run pricing battles
- No divine wrath tracking for preemptions
- Users must manually select regions

**Recommendation**: Add 5 more ZEUS tiers OR use manual region selection for these GPUs!

---

## ☁️ C3 Regions (Cloud Build Worker Pools)

**Location**: `CLI/launch/mecha/mecha_regions.py:5-107`

### All MECHA Regions (18 total!)

Our codebase supports **18 C3 regions** across 6 continents:

#### US Regions (8)
1. **us-central1** - Iowa, USA (current default)
2. **us-east1** - South Carolina, USA
3. **us-east4** - Northern Virginia, USA
4. **us-east5** - Columbus, Ohio, USA
5. **us-west1** - Oregon, USA
6. **us-west2** - Los Angeles, USA ⭐ (MECHA primary)
7. **us-west3** - Salt Lake City, USA
8. **us-west4** - Las Vegas, USA

#### North America (1)
9. **northamerica-northeast1** - Montreal, Canada

#### Europe Regions (5)
10. **europe-west1** - Belgium
11. **europe-west2** - London, UK ⭐ (MECHA secondary)
12. **europe-west3** - Frankfurt, Germany
13. **europe-west4** - Netherlands
14. **europe-west9** - Paris, France

#### Asia Regions (2)
15. **asia-northeast1** - Tokyo, Japan
16. **asia-southeast1** - Singapore

#### Australia/Pacific (1)
17. **australia-southeast1** - Sydney, Australia

#### South America (1)
18. **southamerica-east1** - São Paulo, Brazil

### Region Organization

**By latency to US**:
- **Low**: 8 US regions + Montreal (9 total)
- **Medium**: 5 Europe regions (5 total)
- **Medium-High**: Brazil (1 total)
- **High**: Japan, Singapore, Australia (3 total)

**By continent**:
- **Americas**: 9 regions (8 US + 1 Canada)
- **Europe**: 5 regions
- **Asia**: 2 regions
- **Australia/Pacific**: 1 region
- **South America**: 1 region

---

## 🔴 CRITICAL DISCREPANCY: Manifest vs Reality

### What the Manifest Says (gcp-manifest.json)

**GPU Types** (Line 96-117):
- H200 ✓
- H100 ✓
- L4 ✓
- T4 ✓
- **Missing**: A100, A100 80GB, V100, P4, P100 (5 types!)

**C3 Regions** (Line 70-74):
- us-west2 ✓
- us-central1 ✓
- europe-west2 ✓
- **Missing**: 15 other regions!

**GPU Check Regions** (Line 77-81):
- us-central1 ✓
- **Missing**: 17 other regions!

### What the Code Actually Supports

**GPU Types**: **10 types** (not 4!)
- H200, H100, H100 80GB
- A100, A100 80GB ⭐ **MISSING FROM MANIFEST!**
- L4, T4
- V100, P4, P100 ⭐ **MISSING FROM MANIFEST!**

**C3 Regions**: **18 regions** (not 3!)
- Full global fleet across 6 continents
- Only 3 regions in manifest!

**GPU Check Regions**: **Could check all 18!** (not just 1!)
- Currently only checking us-central1
- Code supports checking ANY region dynamically!

---

## 💡 Why This Matters

### 1. Infrastructure Display Shows Incomplete Data

**Current behavior**:
```
⚠️ Need to request:
  • A100 80GB    ← Actually in code! Just not in manifest!
  • A100         ← Actually in code! Just not in manifest!
  • B200         ← This is a BUG! Not even a real GPU type!
```

**Should show**:
```
⚠️ Need to request:
  • H200
  • H100
  • H100 80GB
  • A100         ← From GPU_QUOTA_METRICS!
  • A100 80GB    ← From GPU_QUOTA_METRICS!
  • L4
  • T4
  • V100         ← Legacy option
  • P4           ← Legacy option
  • P100         ← Legacy option
```

### 2. C3 Quotas Only Check 3 of 18 Regions

**Current**:
- Only checks: us-west2, us-central1, europe-west2
- Misses 15 other valid C3 regions!

**Could check**:
- All 18 MECHA regions
- Show which regions have C3 quota available
- Help user identify best regions for builds

### 3. GPU Quotas Only Check 1 Region

**Current**:
- Only checks us-central1
- Misses all other regions!

**Could check**:
- Multiple strategic regions (us-central1, us-east4, europe-west1)
- Show where user has GPU quota across regions
- Help user find GPUs if one region is exhausted

---

## 📊 Usage Statistics in Codebase

### GPU Types Usage

**Files that reference GPU types**:
```bash
$ grep -rn "H200\|H100\|L4\|T4\|A100\|V100\|P100\|K80" CLI/ | wc -l
192 references!
```

**Most common**:
- A100 mentions: ~80 references (machine selection, setup, docs)
- H100 mentions: ~40 references
- L4 mentions: ~30 references
- T4 mentions: ~25 references

### Region Usage

**Files that reference regions**:
```bash
$ grep -rn "us-west2\|us-central1\|us-east1\|europe-west" CLI/ | wc -l
179 references!
```

**Key files**:
- `CLI/launch/mecha/mecha_regions.py`: C3_REGIONS dict (18 regions)
- `CLI/monitor/screen.py`: ALL_MECHA_REGIONS (monitor all regions!)
- `CLI/monitor/core.py`: Multi-region fetching (4 functions!)

---

## 🎯 Recommendations

### 1. Update gcp-manifest.json

**Add missing GPU types**:
```json
{
  "name": "A100",
  "metric": "Custom model training NVIDIA A100 GPUs per region",
  "spot_metric": "Custom model training preemptible NVIDIA A100 GPUs per region"
},
{
  "name": "A100_80GB",
  "metric": "Custom model training NVIDIA A100 80GB GPUs per region",
  "spot_metric": "Custom model training preemptible NVIDIA A100 80GB GPUs per region"
},
{
  "name": "V100",
  "metric": "Custom model training NVIDIA V100 GPUs per region",
  "spot_metric": "Custom model training preemptible NVIDIA V100 GPUs per region"
},
{
  "name": "P4",
  "metric": "Custom model training NVIDIA P4 GPUs per region",
  "spot_metric": "Custom model training preemptible NVIDIA P4 GPUs per region"
},
{
  "name": "P100",
  "metric": "Custom model training NVIDIA P100 GPUs per region",
  "spot_metric": "Custom model training preemptible NVIDIA P100 GPUs per region"
}
```

**Add ALL 18 C3 regions**:
```json
"mecha_regions": {
  "description": "All 18 MECHA regions for C3 worker pools",
  "regions": [
    "us-central1", "us-east1", "us-east4", "us-east5",
    "us-west1", "us-west2", "us-west3", "us-west4",
    "northamerica-northeast1",
    "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9",
    "asia-northeast1", "asia-southeast1",
    "australia-southeast1",
    "southamerica-east1"
  ]
}
```

**Add strategic GPU check regions**:
```json
"gpu_quota_regions": {
  "description": "Strategic regions to check for GPU quotas",
  "regions": [
    "us-central1",   // Current default
    "us-east4",      // East coast
    "us-west1",      // West coast
    "europe-west1"   // Europe
  ]
}
```

### 2. Use Dynamic Discovery Instead of Manifest

**Current approach**: Hardcoded manifest
**Better approach**: Let the code discover dynamically!

```python
# Import from actual source modules
from CLI.shared.quota.gpu_quota import GPU_QUOTA_METRICS
from CLI.launch.mecha.mecha_regions import ALL_MECHA_REGIONS

# Display ALL supported types/regions
all_gpu_types = list(GPU_QUOTA_METRICS.keys())  # 10 types
all_c3_regions = ALL_MECHA_REGIONS  # 18 regions
```

**Benefits**:
- No duplication!
- Always in sync with code!
- One place to update (the actual implementation)

### 3. Smarter Infrastructure Checks

**GPU quotas**: Check 3-4 strategic regions (not just 1!)
**C3 quotas**: Check all 18 regions (or at least top 10)
**Display**: Show regional breakdown (which regions have quota)

---

## 🔧 Quick Fixes

### Fix 1: Display "B200" Bug

**File**: `CLI/shared/infra_print.py:~290`

**Problem**: Shows "B200" but that's not even a real GPU type!

**Root cause**: `get_all_vertex_gpu_quotas()` dynamically discovers ALL quotas from GCP API, which might include experimental/preview GPUs not in our hardcoded map.

**Fix**: Filter displayed GPUs to only show types in `GPU_QUOTA_METRICS`:
```python
# Filter pending GPUs to only known types
from CLI.shared.quota.gpu_quota import GPU_QUOTA_METRICS
known_gpu_names = {
    "T4", "L4", "V100", "P4", "P100",
    "A100", "A100 80GB",
    "H100", "H100 80GB", "H200"
}

pending_filtered = [
    gpu for gpu in pending
    if gpu.get("gpu_name") in known_gpu_names
]
```

### Fix 2: Check More C3 Regions

**File**: `CLI/config/gcp-manifest.json:70-74`

**Current**: 3 regions
**Should be**: All 18 regions (or at least top 10)

### Fix 3: Check More GPU Regions

**File**: `CLI/config/gcp-manifest.json:77-81`

**Current**: 1 region (us-central1)
**Should be**: 3-4 strategic regions for regional redundancy

---

## 📈 Impact on Infrastructure Display

### Before (Current - 3 C3 regions, 1 GPU region)
```
☁️ C3 Quotas (Cloud Build):
  ✓ us-west2: 176 vCPUs (c3-standard-176)
  ✓ us-central1: 176 vCPUs (c3-standard-176)
  ✗ europe-west2: 0 vCPUs (request quota)

🎮 GPU Quotas (Vertex AI):
  us-central1:
    ✓ T4 (Spot): 1 GPU
    ⚠️ Need to request:
      • A100 80GB    ← BUG: Actually supported but not in manifest!
      • A100         ← BUG: Actually supported but not in manifest!
      • B200         ← BUG: Not even a real GPU type!
```

### After (With Full Inventory - 18 C3 regions, 4 GPU regions)
```
☁️ C3 Quotas (Cloud Build):
  ✓ us-west2: 176 vCPUs (c3-standard-176)
  ✓ us-central1: 176 vCPUs (c3-standard-176)
  ✓ us-east4: 176 vCPUs (c3-standard-176)
  ✓ europe-west1: 176 vCPUs (c3-standard-176)
  ✗ Need to request (14 more regions):
    • us-east1, us-east5, us-west1, us-west3, us-west4
    • northamerica-northeast1
    • europe-west2, europe-west3, europe-west4, europe-west9
    • asia-northeast1, asia-southeast1
    • australia-southeast1
    • southamerica-east1

🎮 GPU Quotas (Vertex AI):
  us-central1:
    ✓ T4 (Spot): 1 GPU
    ⚠️ Need to request (9 types):
      • H200, H100, H100 80GB
      • A100, A100 80GB
      • L4
      • V100, P4, P100

  us-east4:
    ✓ L4 (Spot): 4 GPUs
    ⚠️ Need to request (9 types): ...

  us-west1:
    ⚠️ Need to request (10 types): ...

  europe-west1:
    ⚠️ Need to request (10 types): ...
```

**Better user experience**:
- See ALL GPU types available (not just 4!)
- See ALL C3 regions (not just 3!)
- Regional comparison (which region has best quotas)
- Complete picture of infrastructure options

---

## 🎉 Summary

**What we discovered**:
1. ✅ Codebase supports **10 GPU types** (manifest has 4!)
2. ✅ Codebase supports **18 C3 regions** (manifest has 3!)
3. ✅ Infrastructure checking is LIMITED by manifest (not by code!)
4. ❌ Display shows "B200" which doesn't exist!
5. ❌ Display shows "A100" as "need to request" but it's actually supported!

**Quick wins**:
- Update manifest to include ALL 10 GPU types
- Update manifest to include ALL 18 C3 regions (or top 10)
- Check 3-4 GPU regions instead of 1
- Filter displayed GPUs to known types only

**Better solution**:
- Don't use manifest for types/regions at all!
- Import directly from source modules
- Let code discover dynamically
- Always in sync!

---

## 🤖 MECHA Battle System (Cloud Build C3)

**Purpose**: Progressively acquire & manage 18 C3 worker pools globally

**Location**: `CLI/launch/mecha/`

### Architecture

**Registry**: `mecha/data/mecha_hangar.json` (SafeJSON)
**Tracks**: 18 C3 regions, machine types, statuses, fatigue states

**Key Files**:
- `mecha_integration.py` - Main entry point (called from core.py)
- `mecha_hangar.py` - Registry system + fatigue tracking
- `mecha_acquire.py` - Progressive acquisition (deploy MECHAs)
- `mecha_battle.py` - Pricing battle (select cheapest region)
- `mecha_regions.py` - 18 C3 region definitions
- `campaign_stats.py` - Godzilla incident tracking

### MECHA Lifecycle

**1. CPU Change Detection**:
- Checks if c3-standard-X changed (e.g., 176 → 88)
- If changed: **WIPES ALL 18 POOLS** globally! (nuclear reset)
- Prevents mismatched machine types

**2. Progressive Acquisition**:
- Goal: Collect all 18 MECHAs over time
- Each launch: Try to deploy ONE missing MECHA (background)
- Passive collection strategy (non-blocking)

**3. Pricing Battle**:
- Among deployed MECHAs: Select cheapest region
- Uses pricing data from pricing/pricing_config.py
- Epic battle display with winner banner

**4. Fatigue System** (3-strike):
- **1st timeout** (15 min) → FATIGUED 4 hours 😴
- **2nd timeout** (within 24h) → FATIGUED 4 hours 😴
- **3rd timeout** (within 24h) → EXHAUSTED 24 hours 🛌
- Auto-recovery: MECHAs can battle again after rest

### MECHA Statuses

- **RUNNING** - Pool deployed & ready
- **CREATING** - Pool being created (acquisition in progress)
- **FAILED** - Pool creation failed
- **FATIGUED** - Needs rest (4h cooldown)
- **EXHAUSTED** - Deep rest (24h cooldown)

### Integration Points

**Called by**: `CLI/launch/core.py` (before worker pool creation)
**Returns**: Champion MECHA region (for cloud build submission)
**Side effects**: May wipe pools, deploy new MECHAs, update registry

---

## ⚡ ZEUS Thunder Battle System (Vertex AI GPUs)

**Purpose**: Select cheapest GPU region across 5 tiers

**Location**: `CLI/launch/zeus/`

### Architecture

**Registry**: `zeus/data/zeus_olympus.json` (SafeJSON)
**Tracks**: 5 tiers (spark/bolt/storm/tempest/cataclysm), quota status, divine wrath

**Key Files**:
- `zeus_integration.py` - Main entry point (called from core.py)
- `zeus_olympus.py` - Registry system + divine wrath tracking
- `zeus_acquire.py` - Quota snapshot (discover existing quotas)
- `zeus_battle.py` - Thunder pricing battle (select cheapest region)
- `campaign_stats.py` - Divine incident tracking

### ZEUS Lifecycle

**1. GPU Type Change Detection**:
- Checks if GPU type changed **within a tier**
- Example: storm tier changes T4 → A100 (user config change)
- If changed: **WIPES THAT TIER ONLY** (tier-specific reset)
- Other tiers unaffected (per-tier independence)

**2. Quota Snapshot** (Instant Discovery):
- **ONE API call** discovers ALL regions with quota (15× faster!)
- Runs BEFORE battle (fresh data every launch)
- No passive acquisition - users request quotas via GCP console
- Zeus just discovers what already exists

**3. Thunder Pricing Battle**:
- Among thunder-ready regions: Select cheapest region
- Uses same pricing data as MECHA
- Epic thunder battle display with divine geometry

**4. Divine Wrath System** (3-strike):
- **1st preemption** (spot) → DISFAVORED 4 hours ⚡
- **2nd preemption** (within 24h) → DISFAVORED 4 hours ⚡
- **3rd preemption** (within 24h) → ZEUS'S WRATH 24 hours ⚡⚡⚡
- Auto-recovery: Regions can battle again after wrath period
- **⚠️ NOT YET WIRED!** (system ready, needs job monitoring integration)

### ZEUS Tiers vs GPU Types

**Critical limitation**: ZEUS only supports 5 of 10 GPU types!

| Tier | GPU Type | Memory | Supported |
|------|----------|--------|-----------|
| spark | T4 | 16GB | ✅ |
| bolt | L4 | 24GB | ✅ |
| storm | A100 40GB | 40GB | ✅ |
| tempest | H100 80GB | 80GB | ✅ |
| cataclysm | H200 | 141GB | ✅ |
| - | A100 80GB | 80GB | ❌ No tier! |
| - | H100 | 80GB | ❌ No tier! |
| - | V100 | 16GB | ❌ No tier! |
| - | P4 | 8GB | ❌ No tier! |
| - | P100 | 16GB | ❌ No tier! |

**Users can request quotas for unsupported GPUs**, but must manually select regions!

### ZEUS Statuses

- **THUNDER_READY** - Region has GPU quota (can battle)
- **QUEST_LOCKED** - Region needs quota request (no quota)
- **DISFAVORED** - Under divine wrath (4h cooldown)
- **WRATHFUL** - Under Zeus's wrath (24h cooldown)

### Integration Points

**Called by**: `CLI/launch/core.py` (before Vertex AI job submission)
**Returns**: Champion thunder region (for Vertex AI submission)
**Side effects**: May wipe tier, update quota snapshot, update registry

---

## 🔄 MECHA vs ZEUS Comparison

| Feature | MECHA (C3) | ZEUS (GPU) |
|---------|------------|------------|
| **Regions** | 18 C3 regions | ~15+ GPU regions (tier-specific) |
| **Types tracked** | 1 machine type (c3-standard-176) | 5 GPU types (5 tiers) |
| **Acquisition** | Progressive (passive deploy) | Quota snapshot (instant discovery) |
| **Wipe trigger** | CPU change (wipe ALL 18) | GPU type change (wipe tier only) |
| **Penalty system** | Fatigue (3-strike) | Divine wrath (3-strike) |
| **Registry** | mecha_hangar.json | zeus_olympus.json |
| **Pricing battle** | Among deployed MECHAs | Among thunder-ready regions |
| **API calls** | Multiple (per-region checks) | ONE call (all regions at once!) |

---

## 📊 Why This Matters for Infrastructure Display

### Current Display Gaps

**GPU Types**:
- Shows 4 types from manifest
- Missing 6 types (including unsupported ZEUS types!)
- Should show ALL 10 with ZEUS tier badges

**C3 Regions**:
- Checks 3 regions from manifest
- Missing 15 regions!
- Should show all 18 with MECHA status

### Recommended Display Format

```
🎮 GPU Quotas (Vertex AI):

  ZEUS TIERS (5 GPU types with pricing battles):
    ⚡ spark (T4) - 16GB
    ⚡⚡ bolt (L4) - 24GB
    ⚡⚡⚡ storm (A100 40GB) - 40GB
    ⚡⚡⚡⚡ tempest (H100 80GB) - 80GB
    ⚡⚡⚡⚡⚡ cataclysm (H200) - 141GB

  NOT IN ZEUS TIERS (manual region selection):
    • A100 80GB (80GB) - No pricing battle
    • H100 standard (80GB) - No pricing battle
    • V100 (16GB) - Legacy
    • P4 (8GB) - Legacy
    • P100 (16GB) - Legacy

  us-central1:
    ✓ T4 (Spot): 1 GPU [⚡ spark tier]
    ⚠️ Need to request:
      • H200 [⚡⚡⚡⚡⚡ cataclysm]
      • H100 80GB [⚡⚡⚡⚡ tempest]
      • L4 [⚡⚡ bolt]
      • A100 40GB [⚡⚡⚡ storm]
      • A100 80GB [No tier]
      • H100 [No tier]
      • V100 [No tier - legacy]
      • P4 [No tier - legacy]
      • P100 [No tier - legacy]

☁️ C3 Quotas (Cloud Build):

  MECHA FLEET (18 regions):
    ✓ Deployed (2/18):
      • us-west2: 176 vCPUs (c3-standard-176) [RUNNING]
      • us-central1: 176 vCPUs (c3-standard-176) [RUNNING]

    😴 Fatigued (0/18):
      (none)

    ⚠️ Missing (16/18):
      • us-east1, us-east4, us-east5
      • us-west1, us-west3, us-west4
      • northamerica-northeast1
      • europe-west1, europe-west2, europe-west3, europe-west4, europe-west9
      • asia-northeast1, asia-southeast1
      • australia-southeast1
      • southamerica-east1
```

**Benefits**:
- Shows ZEUS tier associations
- Distinguishes supported vs unsupported GPUs
- Shows MECHA deployment status
- Complete picture of infrastructure

---

**END OF INVENTORY**
