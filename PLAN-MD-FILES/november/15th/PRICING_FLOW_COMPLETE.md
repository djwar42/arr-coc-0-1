# Complete Pricing System Flow Traces

**Date**: 2025-11-15
**Status**: ⚠️ HISTORICAL - Flows describe old implementation with `check_and_update_pricing()`

---

## ⚠️ UPDATE 2025-11-16: Implementation Simplified

**This document describes the ORIGINAL implementation (2025-11-15).**

**Changes made 2025-11-16:**
- ❌ **Removed**: `check_and_update_pricing()` function (85 lines)
- ✅ **Replaced with**: Direct `fetch_pricing_no_save()` calls
- ✅ **Simplified**: No manual triggers, no 24-hour staleness checks
- ✅ **Cloud Scheduler**: Handles all auto-refresh (every 20 min)

**Current implementation**: See `THE_GOOD_PRICING_WAY.md`

**This document is preserved for historical reference only.**

---

---

## Flow 1: MECHA Battle - C3 Machine Pricing (Launch Time)

**File**: `training/cli/launch/core.py` (line 635-707)
**Entry**: User runs `python training/cli.py launch`
**Purpose**: Display MECHA battle animation with live C3 pricing

```
USER RUNS: python training/cli.py launch
    ↓
╔═══════════════════════════════════════════════════════════════════════════
║ FLOW 1: MECHA Battle - C3 Pricing for Animation
╚═══════════════════════════════════════════════════════════════════════════

┌─ training/cli/launch/core.py:635-707 ─────────────────────────────────────
│  launch() function entry
│
├─▶ check_and_update_pricing()
│   │  (from mecha_battle_epic.py)
│   │
│   ├─▶ fetch_pricing_no_save()  ← Artifact Registry
│   │   ├─ Downloads: gcp-pricing package (latest version)
│   │   ├─ Returns: (pricing_data, version, size_kb)
│   │   └─ pricing_data structure:
│   │      {
│   │        "updated": "2025-11-15T12:34:56Z",
│   │        "c3_machines": {
│   │          "us-west2": {
│   │            "cpu_per_core_spot": [
│   │              {"price": 0.00513, "description": "...", "sku_id": "...", "usage_type": "Preemptible"},
│   │              ...
│   │            ],
│   │            "ram_per_gb_spot": [...]
│   │          }
│   │        }
│   │      }
│   │
│   ├─▶ Check age (if >24h, manual trigger warning)
│   │
│   └─▶ Returns: (refreshed, pricing_data)
│
├─▶ get_pricing_age_minutes(pricing_data)
│   └─ Calculates: minutes since pricing_data["updated"]
│
├─▶ Display pricing status:
│   "🪙  Using live pricing (180.7 KB, 12 minutes ago)"
│
└─▶ Pass pricing_data to MECHA battle:
    run_mecha_region_battle(
        ...,
        pricing_data,  ← Full pricing data passed in
        ...
    )
    ↓
    ┌─ training/cli/launch/mecha/mecha_battle_epic.py ──────────────────────
    │  run_mecha_region_battle()
    │
    ├─▶ For each MECHA region battle:
    │   │
    │   ├─▶ calculate_machine_price(machine_type, region, pricing_data)
    │   │   │  (line 230-267)
    │   │   │
    │   │   ├─ Extract vCPUs from machine_type
    │   │   │  (e.g., "c3-standard-176" → 176 vCPUs)
    │   │   │
    │   │   ├─ Calculate RAM based on machine family
    │   │   │  c3-standard: 4 GB RAM per vCPU → 176 * 4 = 704 GB
    │   │   │
    │   │   ├─ Get pricing_data["c3_machines"][region]
    │   │   │
    │   │   ├─▶ get_spot_price(cpu_skus)  ← HELPER FUNCTION
    │   │   │   └─ Returns: cheapest spot CPU price (e.g., 0.00513/core/hr)
    │   │   │
    │   │   ├─▶ get_spot_price(ram_skus)  ← HELPER FUNCTION
    │   │   │   └─ Returns: cheapest spot RAM price (e.g., 0.000687/GB/hr)
    │   │   │
    │   │   └─ Calculate total:
    │   │      price = (176 * 0.00513) + (704 * 0.000687)
    │   │            = 0.90288 + 0.48365
    │   │            = $1.37/hour
    │   │
    │   └─▶ Display in MECHA battle animation:
    │       "💰 PROVISION: $1.37/hr"
    │
    └─▶ Winner selected, battle complete!

RESULT: User sees live C3 pricing in MECHA battle animation
        Pricing used: SPOT (cheapest option for Cloud Build)
```

---

## Flow 2: PyTorch Base Image Build - Provision Quote

**File**: `training/cli/launch/core.py` (line 1819-1826)
**Entry**: During `launch()`, before building `arr-pytorch-base` image
**Purpose**: Show estimated hourly cost for the Cloud Build worker

```
╔═══════════════════════════════════════════════════════════════════════════
║ FLOW 2: PyTorch Build Provision Quote
╚═══════════════════════════════════════════════════════════════════════════

┌─ training/cli/launch/core.py:1819-1826 ───────────────────────────────────
│  _build_pytorch_clean_image() function
│  (Called during launch after MECHA battle selects best_machine)
│
├─▶ get_live_price_for_launch(best_machine, region)
│   │  (from get_live_prices.py)
│   │
│   ├─▶ fetch_pricing_no_save()  ← Artifact Registry
│   │   └─ Returns: (pricing_data, version, size_kb)
│   │
│   ├─▶ Parse machine_type: "c3-standard-176"
│   │   ├─ Detect: C3 machine (starts with "c3-standard-")
│   │   ├─ Extract: 176 vCPUs
│   │   └─ Calculate: 704 GB RAM (176 * 4)
│   │
│   ├─▶ Get pricing_data["c3_machines"][region]
│   │
│   ├─▶ get_spot_price(cpu_skus)  ← HELPER FUNCTION
│   │   └─ Returns: 0.00513/core/hr
│   │
│   ├─▶ get_spot_price(ram_skus)  ← HELPER FUNCTION
│   │   └─ Returns: 0.000687/GB/hr
│   │
│   └─▶ Calculate total:
│       (176 * 0.00513) + (704 * 0.000687) = $1.37/hour
│
├─▶ Store: provision_price_at_start = 1.37
│
├─▶ Submit Cloud Build with worker pool
│   (c3-standard-176 machine for 2-4 hours)
│
└─▶ Track cost in campaign stats
    (provision_price_at_start * build_duration)

RESULT: User sees estimated cost BEFORE build starts
        Used for: Budget planning, campaign stats tracking
        Pricing used: SPOT (Cloud Build worker pool)
```

---

## Flow 3: Campaign Stats - Post-Build Cost Tracking

**File**: `training/cli/launch/core.py` (lines 1591, 1715)
**Entry**: After Cloud Build completes
**Purpose**: Record actual build cost in campaign stats

```
╔═══════════════════════════════════════════════════════════════════════════
║ FLOW 3: Campaign Stats - Post-Build Cost Recording
╚═══════════════════════════════════════════════════════════════════════════

┌─ training/cli/launch/core.py:1591 (after arr-ml-stack build) ────────────
│  AND line 1715 (after arr-trainer build)
│
├─▶ get_live_price_for_launch(machine_type, region)
│   │  (from get_live_prices.py)
│   │
│   ├─▶ For E2_HIGHCPU_8 machines (arr-ml-stack, arr-trainer):
│   │   │
│   │   ├─▶ fetch_pricing_no_save()  ← Artifact Registry
│   │   │
│   │   ├─ Machine: E2_HIGHCPU_8
│   │   │  ├─ vCPUs: 8
│   │   │  └─ RAM: 8 GB (1 GB per vCPU)
│   │   │
│   │   ├─▶ get_standard_price(cpu_skus)  ← HELPER FUNCTION
│   │   │   └─ Returns: ON-DEMAND price (e.g., 0.0218/core/hr)
│   │   │      (NOT spot - E2 uses on-demand for reliability)
│   │   │
│   │   ├─▶ get_standard_price(ram_skus)  ← HELPER FUNCTION
│   │   │   └─ Returns: 0.0029/GB/hr
│   │   │
│   │   └─▶ Calculate:
│   │       (8 * 0.0218) + (8 * 0.0029) = $0.20/hour
│   │
│   └─▶ Returns: hourly_price
│
├─▶ Calculate actual cost:
│   actual_cost = hourly_price * (build_duration / 3600)
│   (e.g., 0.20 * (600s / 3600) = $0.033)
│
└─▶ Record in campaign_stats.json:
    {
      "builds": [
        {
          "image": "arr-ml-stack",
          "machine": "E2_HIGHCPU_8",
          "region": "us-west2",
          "duration_seconds": 600,
          "cost_usd": 0.033
        }
      ]
    }

RESULT: Accurate post-build cost tracking for budget analysis
        Pricing used: ON-DEMAND (E2 for reliability, not spot)
```

---

## Flow 4: Bootstrap - Initial Pricing Fetch

**File**: `training/cli/setup/pricing_setup.py` (line 178-285)
**Entry**: User runs `python training/cli.py setup`
**Purpose**: Fetch initial pricing data during infrastructure setup

```
╔═══════════════════════════════════════════════════════════════════════════
║ FLOW 4: Bootstrap - Initial Pricing Population
╚═══════════════════════════════════════════════════════════════════════════

USER RUNS: python training/cli.py setup
    ↓
┌─ training/cli/setup/pricing_setup.py:178-285 ─────────────────────────────
│  bootstrap_pricing(status) function
│
├─▶ Try to fetch existing pricing:
│   │
│   ├─▶ fetch_pricing_no_save()  ← Artifact Registry
│   │   │
│   │   ├─ IF FOUND:
│   │   │  │
│   │   │  ├─▶ SCHEMA VALIDATION:
│   │   │  │   ├─ get_required_fields()  ← from pricing_config.py
│   │   │  │   │  Returns: {
│   │   │  │   │    "c3_machines": True,      # should have data
│   │   │  │   │    "e2_machines": True,
│   │   │  │   │    "gpus_spot": True,
│   │   │  │   │    "gpus_ondemand": True
│   │   │  │   │  }
│   │   │  │   │
│   │   │  │   ├─ Check each field:
│   │   │  │   │  • Exists? ✓
│   │   │  │   │  • Has data? (len > 0) ✓
│   │   │  │   │
│   │   │  │   └─ IF MISMATCH:
│   │   │  │      "⚠️ Pricing schema mismatch: gpus_spot (empty)"
│   │   │  │      → Force fresh fetch!
│   │   │  │
│   │   │  └─ Check age (if <20 min, use existing)
│   │   │
│   │   └─ IF NOT FOUND → Fresh fetch
│   │
│   └─▶ FileNotFoundError? → Proceed to fresh fetch
│
├─▶ FRESH FETCH:
│   │  _fetch_pricing_inline(status)
│   │
│   ├─▶ Initialize pricing_data structure:
│   │   {
│   │     "updated": "2025-11-15T16:30:45Z",
│   │     "c3_machines": {},  # Will populate with {region: {cpu: [skus], ram: [skus]}}
│   │     "e2_machines": {},
│   │     "gpus_spot": {},    # Will populate with {region: [skus]}
│   │     "gpus_ondemand": {}
│   │   }
│   │
│   ├─▶ Query GCP Billing API:
│   │   │  ~30,000 SKUs scanned
│   │   │
│   │   ├─ Progress updates every 5000 SKUs:
│   │   │  "📄 Checked 5000 SKUs..."
│   │   │  "📄 Checked 10000 SKUs..."
│   │   │  ...
│   │   │
│   │   ├─ For each SKU:
│   │   │  │
│   │   │  ├─ C3 machines (spot):
│   │   │  │  IF "c3" in description AND "preemptible" in description:
│   │   │  │     Add to c3_machines[region]["cpu_per_core_spot"] or ["ram_per_gb_spot"]
│   │   │  │     Store: {price, description, sku_id, usage_type}
│   │   │  │
│   │   │  ├─ E2 machines (on-demand):
│   │   │  │  IF "e2" in description AND NOT preemptible:
│   │   │  │     Add to e2_machines[region]["cpu_per_core_ondemand"] or ["ram_per_gb_ondemand"]
│   │   │  │
│   │   │  └─ GPUs (ALL tiers - spot, on-demand, commitment):
│   │   │     IF "gpu" in description OR "tpu" in description:
│   │   │        IF "Spot" OR "Preemptible" in description:
│   │   │           Add to gpus_spot[region]
│   │   │        ELSE:
│   │   │           Add to gpus_ondemand[region]  ← Includes commitment pricing!
│   │   │
│   │   └─ ALL SKUs collected!
│   │
│   ├─▶ SORT all SKU lists by price (cheapest first):
│   │   for region_data in pricing_data["c3_machines"].values():
│   │       region_data["cpu_per_core_spot"].sort(key=lambda x: x["price"])
│   │       region_data["ram_per_gb_spot"].sort(key=lambda x: x["price"])
│   │   (Same for E2, GPUs spot, GPUs on-demand)
│   │
│   ├─▶ Count and display results:
│   │   "✓ Pricing fetched"
│   │   "   • C3 machines (spot): 43 regions"
│   │   "   • E2 machines (on-demand): 43 regions"
│   │   "   • GPUs (spot): 43 regions - A100=81, H100=53, L4=41, T4=39, V100=28"
│   │   "   • GPUs (on-demand): 47 regions - A100=343, H100=281, H200=105, L4=141, T4=163"
│   │
│   └─▶ Returns: pricing_data (complete, sorted)
│
├─▶ upload_pricing_to_artifact_registry(pricing_data)
│   ├─ Create version: 1.0.YYYYMMDD-HHMMSS
│   ├─ Upload to: arr-coc-pricing repository
│   └─ "Uploading to Artifact Registry (179.4 KB, version 1.0.20251115-163045)..."
│
└─▶ trigger_cloud_function()
    "🚀 Triggering Cloud Function (first run)..."

RESULT: Pricing data populated and stored in Artifact Registry
        Cloud Function will auto-update every 20 minutes
        Schema validation ensures future code changes trigger refetch
```

---

## Flow 5: Cloud Function - Automatic Pricing Updates

**File**: `training/cli/shared/pricing/cloud_function/main.py`
**Entry**: Cloud Scheduler triggers every 20 minutes
**Purpose**: Keep pricing data fresh automatically

```
╔═══════════════════════════════════════════════════════════════════════════
║ FLOW 5: Cloud Function - Auto-Update (Every 20 Minutes)
╚═══════════════════════════════════════════════════════════════════════════

TRIGGER: Cloud Scheduler (*/20 * * * * cron)
    ↓
┌─ training/cli/shared/pricing/cloud_function/main.py ──────────────────────
│  main(request) - Cloud Function entry point
│
├─▶ fetch_gcp_pricing()
│   │
│   ├─▶ get_access_token()
│   │   └─ Metadata server: http://metadata.google.internal/.../token
│   │      Returns: OAuth2 access token (short-lived, secure)
│   │
│   ├─▶ Query GCP Billing API:
│   │   │  (IDENTICAL logic to bootstrap!)
│   │   │
│   │   ├─ Initialize pricing_data structure
│   │   │
│   │   ├─ Scan ~30,000 SKUs:
│   │   │  "📄 Checked 5000 SKUs..."
│   │   │  "📄 Checked 10000 SKUs..."
│   │   │  ...
│   │   │
│   │   ├─ Collect ALL pricing:
│   │   │  • C3 spot
│   │   │  • E2 on-demand
│   │   │  • GPUs spot + on-demand + commitment
│   │   │
│   │   └─ Sort by price (cheapest first)
│   │
│   ├─▶ Display results:
│   │   "✅ Pricing fetched: 30537 SKUs checked"
│   │   "   • C3 machines (spot): 43 regions"
│   │   "   • E2 machines (on-demand): 43 regions"
│   │   "   • GPUs (spot): 43 regions - A100=81, H100=53, ..."
│   │   "   • GPUs (on-demand): 47 regions - A100=343, H100=281, ..."
│   │
│   └─▶ Returns: pricing_data
│
├─▶ upload_to_artifact_registry(pricing_data)
│   ├─ Create new version: 1.0.YYYYMMDD-HHMMSS
│   ├─ Upload via REST API
│   └─ "📦 Uploaded to Artifact Registry (version 1.0.20251115-164523)"
│
└─▶ Return HTTP 200 OK

RESULT: Fresh pricing data every 20 minutes
        No manual intervention needed
        All consumers automatically get updated pricing
```

---

## Helper Functions - How They Work

```
╔═══════════════════════════════════════════════════════════════════════════
║ HELPER FUNCTIONS: Extracting Prices from SKU Lists
╚═══════════════════════════════════════════════════════════════════════════

File: training/cli/shared/artifact_pricing.py

┌─ get_spot_price(sku_list) ────────────────────────────────────────────────
│  Purpose: Get cheapest spot/preemptible price
│
│  Input: [
│    {"price": 0.00513, "usage_type": "Preemptible", ...},
│    {"price": 0.00520, "usage_type": "Preemptible", ...},
│    {"price": 0.0218, "usage_type": "OnDemand", ...}
│  ]
│
│  Logic:
│    1. Filter: usage_type in ["Preemptible", "Spot"]
│       → [0.00513, 0.00520]
│    2. Return first (already sorted, cheapest!)
│       → 0.00513
│
│  Returns: 0.00513  (or None if no spot SKUs)

┌─ get_standard_price(sku_list) ────────────────────────────────────────────
│  Purpose: Get cheapest on-demand (standard) price
│
│  Input: [
│    {"price": 0.00513, "usage_type": "Preemptible", ...},
│    {"price": 0.0218, "usage_type": "OnDemand", ...},
│    {"price": 0.0220, "usage_type": "OnDemand", ...}
│  ]
│
│  Logic:
│    1. Filter: usage_type == "OnDemand"
│       → [0.0218, 0.0220]
│    2. Return first (already sorted!)
│       → 0.0218
│
│  Returns: 0.0218  (or None if no on-demand SKUs)

┌─ get_commitment_1yr_price(sku_list) ──────────────────────────────────────
│  Purpose: Get cheapest 1-year commitment price
│
│  Input: [
│    {"price": 0.0218, "description": "Standard on-demand", "usage_type": "OnDemand"},
│    {"price": 0.015, "description": "1 Year Commitment", "usage_type": "COMMIT"},
│    {"price": 0.010, "description": "3 Year Commitment", "usage_type": "COMMIT"}
│  ]
│
│  Logic:
│    1. Filter: "1 Year" in description OR "1yr" in description
│       → [0.015]
│    2. Return first
│       → 0.015
│
│  Returns: 0.015  (or None if no 1yr commitment SKUs)

┌─ get_commitment_3yr_price(sku_list) ──────────────────────────────────────
│  Purpose: Get cheapest 3-year commitment price
│
│  (Same logic as 1yr, but searches for "3 Year" or "3yr")

┌─ all_prices(sku_list) ────────────────────────────────────────────────────
│  Purpose: Get ALL pricing options with human-readable names
│
│  Input: [sku_list with mixed types]
│
│  Logic:
│    1. Call get_spot_price() → If found, add: {"name": "Spot (Preemptible)", "price": ...}
│    2. Call get_standard_price() → If found, add: {"name": "On-Demand (Standard)", ...}
│    3. Call get_commitment_1yr_price() → If found, add: {"name": "1-Year Commitment", ...}
│    4. Call get_commitment_3yr_price() → If found, add: {"name": "3-Year Commitment", ...}
│
│  Returns: [
│    {"name": "Spot (Preemptible)", "price": 0.00513, "description": "...", ...},
│    {"name": "On-Demand (Standard)", "price": 0.0218, "description": "...", ...},
│    {"name": "1-Year Commitment", "price": 0.015, "description": "...", ...},
│    {"name": "3-Year Commitment", "price": 0.010, "description": "...", ...}
│  ]
│
│  Use case: Display pricing comparison table to user
```

---

## Summary: Complete Pricing Data Flow

```
╔═══════════════════════════════════════════════════════════════════════════
║ COMPLETE SYSTEM OVERVIEW
╚═══════════════════════════════════════════════════════════════════════════

DATA SOURCE (Single Source of Truth):
    │
    │  Artifact Registry: gs://arr-coc-pricing/gcp-pricing
    │  • Package name: gcp-pricing
    │  • Latest version: 1.0.YYYYMMDD-HHMMSS
    │  • Size: ~180 KB
    │  • Updated: Every 20 minutes (Cloud Scheduler)
    │
    ├──▶ Updated by: Cloud Function (every 20 minutes)
    │    Initial fetch: Bootstrap (during setup)
    │
    └──▶ Schema validated: pricing_config.py (PRICING_SCHEMA)

DATA CONSUMERS:
    │
    ├──▶ MECHA Battle (Flow 1)
    │    • When: During launch, before region selection
    │    • Uses: C3 spot pricing (all regions)
    │    • Helper: get_spot_price()
    │    • Purpose: Display live costs in battle animation
    │
    ├──▶ PyTorch Build Provision Quote (Flow 2)
    │    • When: Before Cloud Build submission
    │    • Uses: C3 spot pricing (selected region)
    │    • Helper: get_spot_price()
    │    • Purpose: Show estimated build cost upfront
    │
    ├──▶ Campaign Stats - Post-Build (Flow 3)
    │    • When: After arr-ml-stack and arr-trainer builds
    │    • Uses: E2 on-demand pricing
    │    • Helper: get_standard_price()
    │    • Purpose: Record actual build costs
    │
    └──▶ Future: GPU Training Cost Estimation
         • When: Before Vertex AI job submission
         • Uses: GPU spot/on-demand/commitment pricing
         • Helper: get_spot_price(), all_prices()
         • Purpose: Show estimated training cost, suggest cheaper regions

PRICING TIERS AVAILABLE:
    │
    ├──▶ Spot (Preemptible)
    │    • Cheapest option
    │    • Can be terminated
    │    • Used for: Cloud Build (MECHA worker pool)
    │
    ├──▶ On-Demand (Standard)
    │    • More expensive than spot
    │    • Guaranteed availability
    │    • Used for: E2 builds (arr-ml-stack, arr-trainer)
    │
    ├──▶ 1-Year Commitment
    │    • ~30% cheaper than on-demand
    │    • Requires 1-year commitment
    │    • Available for: Future use (GPU training)
    │
    └──▶ 3-Year Commitment
         • ~50% cheaper than on-demand
         • Requires 3-year commitment
         • Available for: Future use (long-term GPU training)

HELPER FUNCTIONS (All in artifact_pricing.py):
    │
    ├──▶ get_spot_price(sku_list)           → Returns cheapest spot price
    ├──▶ get_standard_price(sku_list)       → Returns cheapest on-demand price
    ├──▶ get_commitment_1yr_price(sku_list) → Returns cheapest 1yr commitment
    ├──▶ get_commitment_3yr_price(sku_list) → Returns cheapest 3yr commitment
    └──▶ all_prices(sku_list)               → Returns all pricing options with names

NO STRAGGLY BITS FOUND! ✅
    • All pricing consumers updated to use helper functions
    • All functions handle None gracefully (fallback to 0.0)
    • Schema validation ensures pricing stays current
    • Cloud Function keeps data fresh automatically
```

---

## Files Modified (Complete List)

1. **training/cli/shared/pricing/cloud_function/main.py**
   - Stores full SKU data (price + metadata)
   - Includes ALL pricing tiers (spot, on-demand, commitment)
   - Sorts by price (cheapest first)

2. **training/cli/setup/pricing_setup.py**
   - Stores full SKU data (same as Cloud Function)
   - Schema validation with auto-refetch
   - E2 pricing added (was missing)

3. **training/cli/shared/pricing_config.py**
   - PRICING_SCHEMA definition (single source of truth)
   - get_required_fields() for validation

4. **training/cli/shared/artifact_pricing.py**
   - Added 5 helper functions
   - get_spot_price(), get_standard_price(), get_commitment_1yr_price(), get_commitment_3yr_price(), all_prices()

5. **training/cli/launch/mecha/mecha_battle_epic.py**
   - Updated calculate_machine_price() to use get_spot_price()

6. **training/cli/shared/pricing/get_live_prices.py**
   - Updated C3 pricing to use get_spot_price()
   - Updated E2 pricing to use get_standard_price()

7. **training/cli/launch/core.py**
   - Uses get_live_price_for_launch() (which uses helpers internally)
   - No changes needed (already uses abstraction layer)

---

**Date**: 2025-11-15
**Status**: ALL FLOWS COMPLETE ✅
**No straggly bits found!** 🎉
