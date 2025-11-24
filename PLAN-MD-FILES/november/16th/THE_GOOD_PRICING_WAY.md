# THE GOOD PRICING WAY

**The canonical pattern for accessing GCP pricing data in ARR-COC**

**Date**: 2025-11-16
**Status**: Production standard
**File**: `training/cli/shared/artifact_pricing.py`

---

## What is THE GOOD PRICING WAY?

**THE GOOD PRICING WAY** = Fetch pricing from **Artifact Registry** with **no local caching**.

```python
# THE GOOD PRICING WAY
from training.cli.shared.artifact_pricing import fetch_pricing_no_save

pricing_data, version, size_kb = fetch_pricing_no_save()
# → Always fetches latest from Artifact Registry
# → No local files
# → No staleness checks needed
# → Cloud Scheduler keeps it fresh (every 20 min)
```

---

## Why is it GOOD?

### ✅ Always Fresh
- Cloud Scheduler updates every 20 minutes
- No manual refresh needed
- No staleness logic required

### ✅ No Local State
- No local JSON files to manage
- No cache invalidation problems
- Works across multiple machines

### ✅ Single Source of Truth
- Artifact Registry = canonical storage
- All consumers get same data
- Version-tracked (immutable history)

### ✅ Fast Enough
- ~1-2 seconds to fetch
- Acceptable for launch flow
- No user-facing delays

---

## The BAD PRICING WAY (What NOT to do)

### ❌ Manual Triggers **(DELETED 2025-11-16)**
```python
# BAD: Manually triggering Cloud Function (THIS CODE WAS REMOVED!)
def check_and_update_pricing():
    if pricing_age > 24_hours:
        trigger_cloud_function()  # DON'T DO THIS
        wait_for_propagation()
        fetch_fresh_pricing()
```

**What was deleted**:
- `mecha_battle_epic.py:275-359` (85 lines) - REMOVED 2025-11-16
- `core.py:702-705` - Now uses `fetch_pricing_no_save()` directly

**Why it was bad**:
- Cloud Scheduler already handles refresh (every 20 min)
- Added unnecessary complexity (24-hour staleness check)
- Required manual Cloud Function trigger
- Could cause race conditions
- **Result**: Deleted entirely, replaced with THE GOOD PRICING WAY

### ❌ Local Caching
```python
# BAD: Saving pricing to local files
with open("pricing_cache.json", "w") as f:
    json.dump(pricing_data, f)  # DON'T DO THIS
```

**Why bad?**
- Stale data risk
- File system dependencies
- Doesn't work across machines
- Cache invalidation problems

---

## THE GOOD PRICING WAY Helper Functions

All helpers live in **`training/cli/shared/artifact_pricing.py`**

### 1. Core Fetching

#### `fetch_pricing_no_save()`
**Lines**: 96-158
**Purpose**: Fetch latest pricing from Artifact Registry

```python
from training.cli.shared.artifact_pricing import fetch_pricing_no_save

pricing_data, version, size_kb = fetch_pricing_no_save()
# Returns:
#   pricing_data: dict - Complete pricing structure
#   version: str - "1.0.20251116-143052"
#   size_kb: float - File size in KB (~180 KB typical)
```

**What it does**:
1. HTTP GET latest version from Artifact Registry
2. `gcloud artifacts generic download` to temp directory
3. Load JSON from temp file
4. Auto-delete temp file (context manager)
5. Return data + metadata

**Example**:
```python
pricing, ver, size = fetch_pricing_no_save()
print(f"Pricing version: {ver}")
print(f"File size: {size:.1f} KB")
print(f"Last updated: {pricing['updated']}")
```

---

### 2. Upload Function

#### `upload_pricing_to_artifact_registry()`
**Lines**: 160-197
**Purpose**: Upload fresh pricing to Artifact Registry

```python
from training.cli.shared.artifact_pricing import upload_pricing_to_artifact_registry

upload_pricing_to_artifact_registry(pricing_data)
# Generates version: "1.0.{timestamp}"
# Uploads to: arr-coc-pricing/gcp-pricing
```

**Used by**:
- Setup bootstrap (after fetching from GCP Billing API)
- Cloud Function (every 20 min refresh)

**Example**:
```python
# After fetching fresh pricing from GCP Billing API
pricing_data = {
    "updated": "2025-11-16T14:30:52Z",
    "c3_machines": {...},
    "e2_machines": {...},
    "gpus_spot": {...},
    "gpus_ondemand": {...}
}

upload_pricing_to_artifact_registry(pricing_data)
# → Creates version: "1.0.20251116-143052"
# → Uploads to Artifact Registry
```

---

### 3. Age Checking

#### `get_pricing_age_minutes()`
**Lines**: 199-217
**Purpose**: Calculate how old pricing data is

```python
from training.cli.shared.artifact_pricing import get_pricing_age_minutes

age_minutes = get_pricing_age_minutes(pricing_data)
# Returns: int - Minutes since pricing["updated"]
```

**Example**:
```python
pricing_data = fetch_pricing_no_save()[0]
age = get_pricing_age_minutes(pricing_data)

if age < 20:
    print("Fresh! (< 20 min)")
elif age < 60:
    print(f"Recent ({age} min old)")
else:
    print(f"Stale ({age // 60} hours old)")
```

#### `format_pricing_age()`
**Lines**: 219-226
**Purpose**: Human-readable age formatting

```python
from training.cli.shared.artifact_pricing import format_pricing_age

age_str = format_pricing_age(age_minutes)
# Returns: "15 minutes ago" or "2 hours ago"
```

**Example**:
```python
age_minutes = 125
print(format_pricing_age(age_minutes))
# Output: "2 hours ago"

age_minutes = 15
print(format_pricing_age(age_minutes))
# Output: "15 minutes ago"
```

---

### 4. Price Extraction Helpers

All helpers work on **SKU lists** (sorted by price, cheapest first).

#### `get_spot_price()`
**Lines**: 229-249
**Purpose**: Extract cheapest spot/preemptible price

```python
from training.cli.shared.artifact_pricing import get_spot_price

cpu_skus = pricing_data["c3_machines"]["us-west2"]["cpu_per_core_spot"]
cpu_price = get_spot_price(cpu_skus)
# Returns: float - Cheapest spot price ($/hour)
# Returns: None - If no spot SKUs found
```

**Filters**: `usage_type in ["Preemptible", "Spot"]`

**Example**:
```python
pricing_data = fetch_pricing_no_save()[0]
region_pricing = pricing_data["c3_machines"]["us-west2"]

cpu_price = get_spot_price(region_pricing["cpu_per_core_spot"])
ram_price = get_spot_price(region_pricing["ram_per_gb_spot"])

print(f"C3 Spot: ${cpu_price:.4f}/core/hr")
print(f"C3 RAM: ${ram_price:.5f}/GB/hr")
# Output:
# C3 Spot: $0.0123/core/hr
# C3 RAM: $0.00165/GB/hr
```

---

#### `get_standard_price()`
**Lines**: 251-270
**Purpose**: Extract cheapest on-demand price

```python
from training.cli.shared.artifact_pricing import get_standard_price

cpu_skus = pricing_data["e2_machines"]["us-central1"]["cpu_per_core_ondemand"]
cpu_price = get_standard_price(cpu_skus)
# Returns: float - Cheapest on-demand price ($/hour)
# Returns: None - If no on-demand SKUs found
```

**Filters**: `usage_type == "OnDemand"`

**Example**:
```python
pricing_data = fetch_pricing_no_save()[0]
e2_pricing = pricing_data["e2_machines"]["us-central1"]

cpu_price = get_standard_price(e2_pricing["cpu_per_core_ondemand"])
ram_price = get_standard_price(e2_pricing["ram_per_gb_ondemand"])

# Calculate E2_HIGHCPU_8 cost
vcpus = 8
ram_gb = 8
hourly_cost = (vcpus * cpu_price) + (ram_gb * ram_price)

print(f"E2_HIGHCPU_8: ${hourly_cost:.3f}/hour")
# Output: E2_HIGHCPU_8: $0.198/hour
```

---

#### `get_commitment_1yr_price()`
**Lines**: 272-292
**Purpose**: Extract 1-year commitment price

```python
from training.cli.shared.artifact_pricing import get_commitment_1yr_price

gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]
price_1yr = get_commitment_1yr_price(gpu_skus)
# Returns: float - 1-year commitment price ($/hour)
# Returns: None - If no 1-year commit SKUs found
```

**Searches description for**: `"1 Year"` or `"1yr"`

**Example**:
```python
pricing_data = fetch_pricing_no_save()[0]
gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]

ondemand = get_standard_price(gpu_skus)
commit_1yr = get_commitment_1yr_price(gpu_skus)
commit_3yr = get_commitment_3yr_price(gpu_skus)

print(f"T4 On-Demand: ${ondemand:.2f}/hr")
print(f"T4 1-Year: ${commit_1yr:.2f}/hr (save {(1 - commit_1yr/ondemand)*100:.0f}%)")
print(f"T4 3-Year: ${commit_3yr:.2f}/hr (save {(1 - commit_3yr/ondemand)*100:.0f}%)")
# Output:
# T4 On-Demand: $0.35/hr
# T4 1-Year: $0.25/hr (save 29%)
# T4 3-Year: $0.18/hr (save 50%)
```

---

#### `get_commitment_3yr_price()`
**Lines**: 294-314
**Purpose**: Extract 3-year commitment price

```python
from training.cli.shared.artifact_pricing import get_commitment_3yr_price

gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]
price_3yr = get_commitment_3yr_price(gpu_skus)
# Returns: float - 3-year commitment price ($/hour)
# Returns: None - If no 3-year commit SKUs found
```

**Searches description for**: `"3 Year"` or `"3yr"`

---

#### `all_prices()`
**Lines**: 316-397
**Purpose**: Get all pricing tiers with full metadata

```python
from training.cli.shared.artifact_pricing import all_prices

gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]
options = all_prices(gpu_skus)
# Returns: list[dict] - All pricing options with metadata
```

**Returns structure**:
```python
[
    {
        "name": "Spot (Preemptible)",
        "price": 0.14,
        "description": "Nvidia Tesla T4 GPU attached to Spot...",
        "sku_id": "XXXX-YYYY-ZZZZ",
        "usage_type": "Spot"
    },
    {
        "name": "On-Demand",
        "price": 0.35,
        "description": "Nvidia Tesla T4 GPU attached to VMs...",
        "sku_id": "AAAA-BBBB-CCCC",
        "usage_type": "OnDemand"
    },
    {
        "name": "1-Year Commitment",
        "price": 0.245,
        "description": "Commitment v1: Nvidia Tesla T4 GPU... 1 Year",
        "sku_id": "DDDD-EEEE-FFFF",
        "usage_type": "Commit"
    },
    {
        "name": "3-Year Commitment",
        "price": 0.175,
        "description": "Commitment v1: Nvidia Tesla T4 GPU... 3 Year",
        "sku_id": "GGGG-HHHH-IIII",
        "usage_type": "Commit"
    }
]
```

**Example**:
```python
pricing_data = fetch_pricing_no_save()[0]
gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]

options = all_prices(gpu_skus)

print("T4 GPU Pricing Options:")
for opt in options:
    print(f"  {opt['name']:20s} ${opt['price']:.3f}/hr")

# Output:
# T4 GPU Pricing Options:
#   Spot (Preemptible)   $0.140/hr
#   On-Demand            $0.350/hr
#   1-Year Commitment    $0.245/hr
#   3-Year Commitment    $0.175/hr
```

---

## Complete Usage Examples

### Example 1: Calculate C3 Spot Cost (MECHA)

```python
from training.cli.shared.artifact_pricing import (
    fetch_pricing_no_save,
    get_spot_price
)

# Fetch pricing (THE GOOD PRICING WAY)
pricing_data, _, _ = fetch_pricing_no_save()

# Get C3 spot pricing for us-west2
region_pricing = pricing_data["c3_machines"]["us-west2"]
cpu_skus = region_pricing["cpu_per_core_spot"]
ram_skus = region_pricing["ram_per_gb_spot"]

cpu_price = get_spot_price(cpu_skus)
ram_price = get_spot_price(ram_skus)

# Calculate c3-standard-176 cost
vcpus = 176
ram_gb = vcpus * 4  # C3: 4 GB per vCPU

hourly_cost = (vcpus * cpu_price) + (ram_gb * ram_price)

print(f"C3-standard-176 Spot (us-west2):")
print(f"  CPU: {vcpus} cores × ${cpu_price:.5f} = ${vcpus * cpu_price:.3f}/hr")
print(f"  RAM: {ram_gb} GB × ${ram_price:.6f} = ${ram_gb * ram_price:.3f}/hr")
print(f"  Total: ${hourly_cost:.2f}/hour")

# Output:
# C3-standard-176 Spot (us-west2):
#   CPU: 176 cores × $0.01234 = $2.172/hr
#   RAM: 704 GB × $0.001650 = $1.162/hr
#   Total: $3.33/hour
```

---

### Example 2: Calculate E2 On-Demand Cost (Cloud Build)

```python
from training.cli.shared.artifact_pricing import (
    fetch_pricing_no_save,
    get_standard_price
)

# Fetch pricing (THE GOOD PRICING WAY)
pricing_data, _, _ = fetch_pricing_no_save()

# Get E2 on-demand pricing for us-central1
region_pricing = pricing_data["e2_machines"]["us-central1"]
cpu_skus = region_pricing["cpu_per_core_ondemand"]
ram_skus = region_pricing["ram_per_gb_ondemand"]

cpu_price = get_standard_price(cpu_skus)
ram_price = get_standard_price(ram_skus)

# Calculate E2_HIGHCPU_8 cost
vcpus = 8
ram_gb = 8  # E2_HIGHCPU: 1 GB per vCPU

hourly_cost = (vcpus * cpu_price) + (ram_gb * ram_price)

print(f"E2_HIGHCPU_8 On-Demand (us-central1):")
print(f"  CPU: {vcpus} cores × ${cpu_price:.5f} = ${vcpus * cpu_price:.3f}/hr")
print(f"  RAM: {ram_gb} GB × ${ram_price:.5f} = ${ram_gb * ram_price:.3f}/hr")
print(f"  Total: ${hourly_cost:.3f}/hour")

# Output:
# E2_HIGHCPU_8 On-Demand (us-central1):
#   CPU: 8 cores × $0.02180 = $0.174/hr
#   RAM: 8 GB × $0.00292 = $0.023/hr
#   Total: $0.198/hour
```

---

### Example 3: GPU Pricing Comparison

```python
from training.cli.shared.artifact_pricing import (
    fetch_pricing_no_save,
    all_prices
)

# Fetch pricing (THE GOOD PRICING WAY)
pricing_data, _, _ = fetch_pricing_no_save()

# Get all T4 GPU pricing options
gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]
options = all_prices(gpu_skus)

# Filter for T4 only
t4_options = [opt for opt in options if "T4" in opt["description"]]

print("NVIDIA T4 GPU Pricing (us-central1):")
print("-" * 50)

for opt in t4_options:
    print(f"{opt['name']:20s} ${opt['price']:.3f}/hr")

    # Calculate savings vs on-demand
    if opt['name'] != "On-Demand":
        ondemand = next(o for o in t4_options if o['name'] == "On-Demand")
        savings = (1 - opt['price'] / ondemand['price']) * 100
        print(f"{'':20s} (save {savings:.0f}%)")

# Output:
# NVIDIA T4 GPU Pricing (us-central1):
# --------------------------------------------------
# Spot (Preemptible)   $0.140/hr
#                      (save 60%)
# On-Demand            $0.350/hr
# 1-Year Commitment    $0.245/hr
#                      (save 30%)
# 3-Year Commitment    $0.175/hr
#                      (save 50%)
```

---

### Example 4: Pricing Age Check

```python
from training.cli.shared.artifact_pricing import (
    fetch_pricing_no_save,
    get_pricing_age_minutes,
    format_pricing_age
)

# Fetch pricing (THE GOOD PRICING WAY)
pricing_data, version, size_kb = fetch_pricing_no_save()

# Check age
age_minutes = get_pricing_age_minutes(pricing_data)
age_str = format_pricing_age(age_minutes)

print(f"Pricing Information:")
print(f"  Version: {version}")
print(f"  File size: {size_kb:.1f} KB")
print(f"  Last updated: {pricing_data['updated']}")
print(f"  Age: {age_str}")
print(f"  Status: {'Fresh ✓' if age_minutes < 20 else 'Stale ⚠️'}")

# Output:
# Pricing Information:
#   Version: 1.0.20251116-143052
#   File size: 180.2 KB
#   Last updated: 2025-11-16T14:30:52Z
#   Age: 15 minutes ago
#   Status: Fresh ✓
```

---

### Example 5: Real-World Launch Flow

```python
# This is how it's used in training/cli/launch/core.py

from training.cli.shared.pricing.get_live_prices import get_live_price_for_launch

# Get cost for Cloud Build machine
machine_type = "E2_HIGHCPU_8"
region = "us-west2"

price_per_hour = get_live_price_for_launch(machine_type, region)
# → Internally calls fetch_pricing_no_save() (THE GOOD PRICING WAY)
# → Returns: 0.198

# Estimate build cost
estimated_build_minutes = 45
estimated_cost = (estimated_build_minutes / 60) * price_per_hour

print(f"Build estimate:")
print(f"  Machine: {machine_type}")
print(f"  Region: {region}")
print(f"  Rate: ${price_per_hour:.3f}/hour")
print(f"  Duration: {estimated_build_minutes} minutes")
print(f"  Estimated cost: ${estimated_cost:.2f}")

# Output:
# Build estimate:
#   Machine: E2_HIGHCPU_8
#   Region: us-west2
#   Rate: $0.198/hour
#   Duration: 45 minutes
#   Estimated cost: $0.15
```

---

## Summary: THE GOOD PRICING WAY Rules

### ✅ DO
1. **Always use `fetch_pricing_no_save()`** - Never cache locally
2. **Use helper functions** - Don't parse SKUs manually
3. **Trust Cloud Scheduler** - No manual refresh needed
4. **Keep it simple** - Fetch → Extract → Calculate

### ❌ DON'T
1. **Don't save to local files** - Artifact Registry is the source
2. **Don't manually trigger Cloud Function** - Scheduler handles it
3. **Don't implement staleness checks** - Not needed (20 min refresh)
4. **Don't parse pricing_data directly** - Use helper functions

---

## Function Reference Quick List

| Function | Purpose | Returns |
|----------|---------|---------|
| `fetch_pricing_no_save()` | Fetch from Artifact Registry | `(dict, str, float)` |
| `upload_pricing_to_artifact_registry()` | Upload to Artifact Registry | `None` |
| `get_pricing_age_minutes()` | Calculate age | `int` |
| `format_pricing_age()` | Format age string | `str` |
| `get_spot_price()` | Cheapest spot price | `float` or `None` |
| `get_standard_price()` | Cheapest on-demand | `float` or `None` |
| `get_commitment_1yr_price()` | 1-year commit price | `float` or `None` |
| `get_commitment_3yr_price()` | 3-year commit price | `float` or `None` |
| `all_prices()` | All tiers with metadata | `list[dict]` |

---

## Where to Find Everything

**Helper functions**: `training/cli/shared/artifact_pricing.py`
**Consumer example**: `training/cli/shared/pricing/get_live_prices.py`
**Infrastructure setup**: `training/cli/setup/pricing_setup.py`
**System map**: `GPU_PRICING_AND_QUOTA_SYSTEM_MAP.md`

---

**Last Updated**: 2025-11-16
**Status**: Production standard
**Philosophy**: Simple, stateless, always fresh ✓
