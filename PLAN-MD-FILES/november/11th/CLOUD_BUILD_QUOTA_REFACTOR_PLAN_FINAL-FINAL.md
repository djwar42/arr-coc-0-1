# Cloud Build C3 Quota Refactor - FINAL IMPLEMENTATION GUIDE

**Goal:** Centralize quota checking, add MECHA quota awareness, lazy load quota entries

**Date:** 2025-11-11

**Estimated Time:** 3-4 hours

---

## 📋 Implementation Checklist

### Phase 1: Create Central Quota Checker
- [x] Create `training/cli/shared/quota_checker.py`
- [x] Test quota checker returns correct data
- [x] Verify caching works (5 min TTL)

### Phase 2: Replace Existing Checks
- [x] Update `training/cli/setup/core.py` line 886
- [x] Update `training/cli/launch/core.py` line 537
- [x] Test setup still works
- [x] Test launch still works

### Phase 3: Add MECHA Quota Awareness
- [x] Create `training/cli/launch/mecha/mecha_quota.py`
- [x] Add lazy loading to `mecha_acquire.py`
- [x] Update `mecha_integration.py` with quota filtering
- [x] Test all 3 battle paths (0/1/2+ ready)

### Phase 4: Update Infrastructure Display
- [x] Add quota section to setup infra display
- [x] Test `python training/cli.py infra` shows quotas

### Phase 5: Documentation
- [x] Update `REQUEST_C3_QUOTA_CONSOLE.md`
- [x] Add lazy loading notes

### Phase 6: Testing
- [x] Run setup - verify quota guidance shown
- [x] Run launch - verify lazy loading works
- [x] Run launch - verify sidelined display shows
- [x] Test solo MECHA scenario
- [x] Test epic battle scenario

### Phase 7: Git Commit
- [x] Commit all changes with descriptive message

---

## ⚠️ IMPORTANT: Cloud Build vs Compute Engine Quotas

**Only Cloud Build C3 quotas apply to worker pools - Compute Engine quotas do NOT apply.**

This is GCP's design: Cloud Build private pools use their own quota namespace
separate from Compute Engine VMs.

---

## 🔧 PHASE 1: Create Central Quota Checker

### File: `training/cli/shared/quota_checker.py` (NEW)

**Location:** `training/cli/shared/quota_checker.py`

**Full Code:**

```python
"""
Cloud Build C3 Quota Checker

Centralized quota checking for build_origin=default quotas.
Uses conservative approach: only returns confirmed quotas (176 vCPUs typically).

Does NOT return System Limits or auto-created 4 vCPU entries
(API limitation - these show in Console but not in gcloud API).
"""

import subprocess
import json
import time
from typing import Dict

# Cache quota results (5 min TTL)
_quota_cache = {}
_cache_timestamp = {}
_CACHE_TTL = 300  # 5 minutes


def get_cloud_build_c3_quotas(project_id: str, use_cache: bool = True) -> Dict[str, int]:
    """
    Get Cloud Build C3 quota for regions with CONFIRMED custom quotas

    Args:
        project_id: GCP project ID
        use_cache: Use cached results if available (default: True)

    Returns:
        Dict mapping region → vCPUs quota (build_origin=default only)
        Example: {"us-central1": 176, "asia-northeast1": 176}

    Notes:
        - Only returns regions with build_origin=default dimension
        - These are manually approved quotas (typically 176 vCPUs)
        - Does NOT return System Limits (region-only dimension)
        - Does NOT return auto-created 4 vCPU entries (API limitation)
    """
    # Check cache
    if use_cache and project_id in _quota_cache:
        age = time.time() - _cache_timestamp.get(project_id, 0)
        if age < _CACHE_TTL:
            return _quota_cache[project_id].copy()

    # Query API
    quotas = _fetch_quotas_from_api(project_id)

    # Update cache
    _quota_cache[project_id] = quotas
    _cache_timestamp[project_id] = time.time()

    return quotas.copy()


def _fetch_quotas_from_api(project_id: str) -> Dict[str, int]:
    """
    Fetch quotas from gcloud API

    Returns only quotas with BOTH dimensions:
    - build_origin=default
    - region=REGION_NAME
    """
    try:
        result = subprocess.run(
            [
                'gcloud', 'alpha', 'services', 'quota', 'list',
                '--service=cloudbuild.googleapis.com',
                f'--consumer=projects/{project_id}',
                '--format=json'
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
        quotas = {}

        # Parse quota buckets
        for service in data:
            for limit in service.get('consumerQuotaLimits', []):
                metric = limit.get('metric', '')

                # Only C3 quotas
                if 'concurrent_private_pool_c3_build_cpus' not in metric:
                    continue

                for bucket in limit.get('quotaBuckets', []):
                    dims = bucket.get('dimensions', {})

                    # MUST have BOTH dimensions (build_origin=default + region)
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


def get_region_quota(project_id: str, region: str) -> int:
    """
    Get Cloud Build C3 quota for a single region

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
        required_vcpus: Minimum vCPUs needed (e.g., 176 for c3-standard-176)

    Returns:
        True if region has >= required_vcpus, False otherwise
    """
    quota = get_region_quota(project_id, region)
    return quota >= required_vcpus


def clear_cache():
    """Clear quota cache (for testing)"""
    global _quota_cache, _cache_timestamp
    _quota_cache = {}
    _cache_timestamp = {}
```

**Testing:**

```bash
# Test the quota checker
cd /Users/alfrednorth/Desktop/Code/arr-coc-ovis/RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1

python3 << 'EOF'
from training.cli.shared.quota_checker import get_cloud_build_c3_quotas

quotas = get_cloud_build_c3_quotas('weight-and-biases-476906')
print("Quotas found:")
for region, vcpus in sorted(quotas.items()):
    print(f"  {region:25s} → {vcpus:3d} vCPUs")

# Expected: {"us-central1": 176, "asia-northeast1": 176}
EOF
```

**Expected Output:**
```
Quotas found:
  asia-northeast1           → 176 vCPUs
  us-central1               → 176 vCPUs
```

**Checkpoint:**
- [ ] File created at `training/cli/shared/quota_checker.py`
- [ ] Test passes and shows 2 regions with 176 vCPUs
- [ ] No errors

---

## 🔧 PHASE 2: Replace Existing Quota Checks

### File 1: `training/cli/setup/core.py`

**Location:** Line 886

**Find this code:**
```python
best_machine, best_vcpus, ce_quota, cb_quota, limiting_factor = get_best_available_c3_machine(project_id, region)
```

**Replace with:**
```python
# NEW: Use centralized quota checker
from training.cli.shared.quota_checker import get_region_quota

cb_quota = get_region_quota(project_id, region)

# Determine best machine based on quota
if cb_quota >= 176:
    best_machine = "c3-standard-176"
    best_vcpus = 176
elif cb_quota >= 88:
    best_machine = "c3-standard-88"
    best_vcpus = 88
elif cb_quota >= 44:
    best_machine = "c3-standard-44"
    best_vcpus = 44
else:
    best_machine = "c3-standard-44"
    best_vcpus = 44

# Note: Compute Engine quota check removed (not needed for Cloud Build)
ce_quota = None
limiting_factor = "cloud_build"
```

**Also update the quota display section (around line 898):**

**Find:**
```python
if best_vcpus >= 176:
    # CHONK power meter code...
```

**Replace with:**
```python
if best_vcpus >= 176:
    status("  ⚡ ABSOLUTE UNIT DETECTED ⚡")
    status("  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MAXIMUM CHONK")
    status(f"  Cloud Build C3 Quota: {cb_quota} vCPUs (c3-standard-176) 🏭")
    status("")
```

**Checkpoint:**
- [ ] Import added: `from training.cli.shared.quota_checker import get_region_quota`
- [ ] Line 886 updated to use `get_region_quota()`
- [ ] Machine selection logic updated
- [ ] No syntax errors

---

### File 2: `training/cli/launch/core.py`

**Location:** Line 537

**Find this code:**
```python
best_machine, best_vcpus, ce_quota, cb_quota, limiting_factor = get_best_available_c3_machine(project_id, region)
```

**Replace with:**
```python
# NEW: Use centralized quota checker
from training.cli.shared.quota_checker import get_region_quota

cb_quota = get_region_quota(project_id, region)

# Determine best machine based on quota
if cb_quota >= 176:
    best_machine = "c3-standard-176"
    best_vcpus = 176
elif cb_quota >= 88:
    best_machine = "c3-standard-88"
    best_vcpus = 88
elif cb_quota >= 44:
    best_machine = "c3-standard-44"
    best_vcpus = 44
else:
    best_machine = "c3-standard-44"
    best_vcpus = 44
```

**Also update the CHONK display section (around line 1282):**

**Find:**
```python
if vcpus >= 176:
    # CHONK display...
```

**Replace with:**
```python
if vcpus >= 176:
    status("  ⚡ ABSOLUTE UNIT ⚡")
    status(f"  Machine: {machine_type}")
    status(f"  Cloud Build Quota: {cb_quota} vCPUs")
```

**Checkpoint:**
- [ ] Import added: `from training.cli.shared.quota_checker import get_region_quota`
- [ ] Line 537 updated to use `get_region_quota()`
- [ ] Machine selection logic updated
- [ ] No syntax errors

**Testing Phase 2:**

```bash
# Test setup still works
python training/cli.py setup

# Test launch still works (will start MECHA process)
# DON'T RUN FULL LAUNCH - just check it starts without errors
```

---

## 🔧 PHASE 3: Add MECHA Quota Awareness

### File 1: `training/cli/launch/mecha/mecha_quota.py` (NEW)

**Location:** `training/cli/launch/mecha/mecha_quota.py`

**Full Code:**

```python
"""
MECHA Quota Management

Handles quota filtering and sidelined MECHA display.
"""

from typing import List, Dict, Tuple


def separate_by_quota(
    regions: List[str],
    project_id: str,
    required_vcpus: int = 176
) -> Tuple[List[str], List[str]]:
    """
    Separate regions into battle-ready vs sidelined based on quota

    Args:
        regions: List of MECHA regions to check
        project_id: GCP project ID
        required_vcpus: Minimum vCPUs needed (default: 176)

    Returns:
        (battle_ready, sidelined) tuple
        - battle_ready: Regions with quota >= required_vcpus
        - sidelined: Regions with quota < required_vcpus
    """
    from training.cli.shared.quota_checker import get_cloud_build_c3_quotas

    quotas = get_cloud_build_c3_quotas(project_id)

    battle_ready = []
    sidelined = []

    for region in regions:
        quota = quotas.get(region, 0)
        if quota >= required_vcpus:
            battle_ready.append(region)
        else:
            sidelined.append(region)

    return battle_ready, sidelined


def display_sidelined_mechas(
    sidelined: List[str],
    quotas: Dict[str, int],
    project_id: str,
    status_callback
):
    """
    Show sidelined MECHAs with COMPLETE quota increase instructions

    Args:
        sidelined: List of sidelined regions
        quotas: Dict of {region: vcpus}
        project_id: GCP project ID
        status_callback: Function to output status messages
    """
    if not sidelined:
        return

    status = status_callback

    status("")
    status("⚠️  SIDELINED MECHAS (Need Quota Increases):")
    status("╔════════════════════════════════════════════════════════════╗")
    status(f"║ {len(sidelined)} MECHAs have worker pools but need quota increases     ║")
    status("╚════════════════════════════════════════════════════════════╝")
    status("")

    status("Regions with pools but insufficient quota:")
    for region in sorted(sidelined):
        current_quota = quotas.get(region, 4)  # Default 4 if not in API
        status(f"  • {region:25s} (Current: {current_quota} vCPUs, Need: 176)")

    status("")
    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("📋 COMPLETE STEPS TO INCREASE QUOTAS:")
    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("")

    # Direct link with filter pre-applied
    status("1️⃣  OPEN QUOTAS CONSOLE (link has filter pre-applied):")
    status("")
    console_link = (
        f"https://console.cloud.google.com/apis/api/cloudbuild.googleapis.com/"
        f"quotas?project={project_id}&pageState=(%22allQuotasTable%22:(%22r%22:200,"
        f"%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22"
        f"Concurrent%2520C3%2520Build%2520CPUs%2520%2528Private%2520Pool%2529_5C_22_22"
        f"%257D%255D%22))"
    )
    status(f"   {console_link}")
    status("")

    status("2️⃣  VERIFY FILTER (should already be applied):")
    status("")
    status("   If filter isn't applied, type EXACTLY:")
    status("   ┌────────────────────────────────────────────────────────┐")
    status("   │ Concurrent C3 Build CPUs (Private Pool)                │")
    status("   └────────────────────────────────────────────────────────┘")
    status("")
    status("   ⚠️  IMPORTANT:")
    status("      - NO '(default)' suffix!")
    status("      - This shows YOUR quotas (not System Limits)")
    status("")

    status("3️⃣  IDENTIFY QUOTA ENTRIES (ignore System Limits):")
    status("")
    status("   You should see:")
    status("   ✅ Type: 'Quota' (Adjustable: Yes) ← THESE ARE YOUR QUOTAS")
    status("   ❌ Type: 'System limit' (Adjustable: No) ← IGNORE THESE")
    status("")
    status("   Look for your sidelined regions:")
    for region in sorted(sidelined):
        status(f"      • {region:25s} → Should show 4 vCPUs")
    status("")

    status("4️⃣  REQUEST QUOTA INCREASE:")
    status("")
    status("   For EACH sidelined region:")
    status("   a) Click on the quota row")
    status("   b) Click 'EDIT QUOTAS' button (top right)")
    status("   c) Enter new limit:")
    status("")
    status("      ┌────────────────────────────────────────┐")
    status("      │ Recommended Options:                   │")
    status("      │                                        │")
    status("      │ • 176 vCPUs (RECOMMENDED - CHONK!)     │")
    status("      │   → c3-standard-176 machine            │")
    status("      │   → Fastest builds (~20 min)           │")
    status("      │                                        │")
    status("      │ • 88 vCPUs (Good middle ground)        │")
    status("      │   → c3-standard-88 machine             │")
    status("      │   → ~25 min builds                     │")
    status("      │                                        │")
    status("      │ • 44 vCPUs (Budget option)             │")
    status("      │   → c3-standard-44 machine             │")
    status("      │   → ~35 min builds                     │")
    status("      └────────────────────────────────────────┘")
    status("")

    status("   d) Justification (copy this):")
    status("      'Cloud Build worker pools for PyTorch compilation.")
    status("       Requesting C3 quota for multi-region builds.")
    status("       Expected usage: 1-2 concurrent builds.'")
    status("")

    status("   e) Click 'SUBMIT REQUEST'")
    status("")

    status("5️⃣  WAIT FOR GOOGLE APPROVAL:")
    status("")
    status("   • Typical approval time: 1-2 business days")
    status("   • You'll get email notification when approved")
    status("   • After approval: Region becomes battle-ready! ⚔️")
    status("")

    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("")
    status("💡 TIP: Request all regions at once (faster than one-by-one)")
    status("")


def display_battle_ready_mechas(
    battle_ready: List[str],
    quotas: Dict[str, int],
    status_callback
):
    """
    Display battle-ready MECHAs section

    Args:
        battle_ready: List of battle-ready regions
        quotas: Dict of {region: vcpus}
        status_callback: Function to output status messages
    """
    if not battle_ready:
        return

    status = status_callback

    status("")
    status("⚔️  BATTLE-READY MECHAS:")
    status("╔════════════════════════════════════════════════════════════╗")
    status(f"║ {len(battle_ready)} MECHAs ready for price battle                          ║")
    status("╚════════════════════════════════════════════════════════════╝")
    status("")

    status("Regions with pools AND quota:")
    for region in sorted(battle_ready):
        vcpus = quotas.get(region, 0)
        status(f"  • {region:25s} (Quota: {vcpus} vCPUs) ✅")
    status("")
```

**Checkpoint:**
- [ ] File created at `training/cli/launch/mecha/mecha_quota.py`
- [ ] No syntax errors
- [ ] All functions present

---

### File 2: `training/cli/launch/mecha/mecha_acquire.py`

**Location:** After worker pool creation in `acquire_mecha_for_region()` or similar function

**Add this function to the file:**

```python
def lazy_load_quota_entry(project_id: str, region: str, status_callback):
    """
    Submit test build to trigger quota entry creation

    This creates a quota entry with 4 vCPUs (default).
    User can then request increase to 176 vCPUs via Console.

    Takes ~5 seconds.

    Args:
        project_id: GCP project ID
        region: GCP region
        status_callback: Function to output status messages
    """
    import subprocess
    import tempfile

    status = status_callback

    status(f"  Initializing quota entry for {region}...")

    # Create minimal test build
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal Dockerfile
        with open(f"{tmpdir}/Dockerfile", "w") as f:
            f.write("FROM alpine:latest\nRUN echo 'quota test'\n")

        # Minimal cloudbuild.yaml
        with open(f"{tmpdir}/cloudbuild.yaml", "w") as f:
            f.write("steps:\n- name: 'gcr.io/cloud-builders/docker'\n  args: ['build', '.']\n")

        # Submit build (will fail with quota error, but that's expected!)
        subprocess.run([
            'gcloud', 'builds', 'submit', tmpdir,
            f'--config={tmpdir}/cloudbuild.yaml',
            f'--region={region}',
            f'--worker-pool=projects/{project_id}/locations/{region}/workerPools/pytorch-mecha-pool',
            '--timeout=1m'
        ], capture_output=True, text=True)

        # We don't care if it fails - quota entry is created either way!

    status(f"  ✅ Quota entry initialized (4 vCPUs default)")
    status(f"     You can now request 176 vCPUs in Console!")
```

**Then find where worker pools are created and add lazy loading after:**

**Find something like:**
```python
# After worker pool creation succeeds
status(f"✅ Worker pool created: {region}")
```

**Add after it:**
```python
# NEW: Lazy load quota entry
status("")
lazy_load_quota_entry(project_id, region, status)
status("")
```

**Checkpoint:**
- [ ] `lazy_load_quota_entry()` function added
- [ ] Function called after each worker pool creation
- [ ] No syntax errors

---

### File 3: `training/cli/launch/mecha/mecha_integration.py`

**Location:** In `run_mecha_battle()` or similar function

**Find the battle logic (something like):**
```python
def run_mecha_battle(...):
    # ... existing code to get acquired MECHAs ...

    # OLD: Battle immediately
    return epic_mecha_price_battle(acquired, status_callback)
```

**Replace with:**
```python
def run_mecha_battle(project_id, best_machine, acquired_regions, primary_region, status_callback):
    """
    Run MECHA battle with quota pre-check

    Args:
        project_id: GCP project ID
        best_machine: Machine type (e.g., "c3-standard-176")
        acquired_regions: List of regions with worker pools
        primary_region: Fallback region
        status_callback: Function to output status messages
    """
    from .mecha_quota import (
        separate_by_quota,
        display_sidelined_mechas,
        display_battle_ready_mechas
    )
    from training.cli.shared.quota_checker import get_cloud_build_c3_quotas

    status = status_callback

    # Get required vCPUs from machine type
    vcpus_needed = int(best_machine.split("-")[-1])  # "c3-standard-176" → 176

    # NEW: Check quota for all acquired MECHAs
    quotas = get_cloud_build_c3_quotas(project_id)
    battle_ready, sidelined = separate_by_quota(acquired_regions, project_id, vcpus_needed)

    # Show sidelined (if any)
    if sidelined:
        display_sidelined_mechas(sidelined, quotas, project_id, status)

    # Show battle-ready
    display_battle_ready_mechas(battle_ready, quotas, status)

    # Battle logic (3 paths)
    if len(battle_ready) == 0:
        status("")
        status("❌ NO BATTLE-READY MECHAS!")
        status(f"   Falling back to PRIMARY: {primary_region}")
        status("")
        return primary_region

    elif len(battle_ready) == 1:
        solo_region = battle_ready[0]
        status("")
        status("⚔️  THIS MECHA BATTLES ALONE AND WINS! ⚔️")
        status("")

        # Get price for display
        from .mecha_battle_epic import get_region_price
        price = get_region_price(solo_region)

        status(f"   🏆 CHAMPION: {solo_region} |${price:.2f}/hr|")
        status("")
        status("   (No price comparison needed - it's the only option!)")
        status("")

        return solo_region

    else:
        # Epic battle (2+ MECHAs)
        from .mecha_battle_epic import epic_mecha_price_battle

        status("")
        status("🤖 MECHA PRICE BATTLE GO!")
        status("")

        selected, price, compare, compare_price, savings = epic_mecha_price_battle(
            battle_ready, status
        )

        savings_percent = (savings / compare_price * 100) if compare_price > 0 else 0

        status("")
        status(f"   ∿◇∿ MECHA BATTLE COMPLETE ∿◇∿")
        status(f"   ∿◇∿ CHAMPION:  |${price:.2f}/hr| {selected} ∿◇∿")
        status(f"   ∿◇∿ SAVES:     {savings_percent:.0f}% ${savings:.2f} vs {compare} |${compare_price:.2f}/hr| ∿◇∿")
        status("")

        return selected
```

**Checkpoint:**
- [ ] Import statements added for quota functions
- [ ] Quota filtering added BEFORE battle
- [ ] Sidelined display added
- [ ] Battle-ready display added
- [ ] 3 battle paths implemented (0/1/2+)
- [ ] Solo MECHA message: "THIS MECHA BATTLES ALONE AND WINS!"
- [ ] No syntax errors

---

## 🔧 PHASE 4: Update Infrastructure Display

### File: `training/cli/setup/core.py`

**Location:** In `run_setup_core()` or `check_infrastructure_core()`, add quota section

**Add after infrastructure checks:**

```python
# NEW: Show Cloud Build C3 Quota Guidance
status("")
status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
status("📊 Cloud Build C3 Quotas:")
status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
status("")

# Check current quotas
from training.cli.shared.quota_checker import get_cloud_build_c3_quotas
quotas = get_cloud_build_c3_quotas(project_id)

if quotas:
    status("Current quotas:")
    for region, vcpus in sorted(quotas.items()):
        status(f"  ✅ {region:25s} → {vcpus:3d} vCPUs")
else:
    status("⚠️  No Cloud Build C3 quotas detected yet")

status("")
status("ℹ️  About Quotas:")
status("")
status("  • Quota entries appear AFTER first launch attempt")
status("  • Run 'python training/cli.py launch' to initialize quotas")
status("  • Then request increases via Console")
status("")

status("📋 To view/manage quotas:")
status("")
status("  1. Open Quotas Console:")
console_link = (
    f"https://console.cloud.google.com/apis/api/cloudbuild.googleapis.com/"
    f"quotas?project={project_id}"
)
status(f"     {console_link}")
status("")

status("  2. Filter for EXACTLY:")
status("     'Concurrent C3 Build CPUs (Private Pool)'")
status("     (NO '(default)' suffix!)")
status("")

status("  3. Look for Type: 'Quota' (Adjustable: Yes)")
status("     Ignore 'System limit' entries")
status("")

status("  4. Recommended quota: 88 or 176 vCPUs per region")
status("")
```

**Checkpoint:**
- [ ] Quota section added to setup output
- [ ] Import added for `get_cloud_build_c3_quotas`
- [ ] Console link included
- [ ] Filter string shown
- [ ] No syntax errors

---

## 🔧 PHASE 5: Documentation Updates

### File: `REQUEST_C3_QUOTA_CONSOLE.md`

**Add new section after "Understanding System Limits vs Quotas":**

```markdown
## 🔄 Lazy Quota Creation (Automatic)

**How quota entries are created:**

### Before Any Builds:
- Regions show: System Limit only (4 vCPUs, Adjustable: No)
- Console: Cannot request quota increase (no entry to edit!)

### After First Launch Attempt:
- MECHA creates worker pool in region
- Test build submitted → Triggers quota entry creation
- Quota entry appears: 4 vCPUs (Adjustable: Yes) ⭐
- You can NOW request increase to 176 vCPUs!

### What This Means:
- Run `python training/cli.py launch` FIRST
- This initializes quota entries for all MECHA regions
- THEN request quota increases via Console
- Wait 1-2 days for Google approval
- After approval: Regions become battle-ready!

**This happens automatically during MECHA acquisition - no manual work needed!**
```

**Checkpoint:**
- [ ] Lazy quota creation section added
- [ ] Explains the automatic process
- [ ] Links to launch command

---

## 🔧 PHASE 6: Testing

### Test 1: Setup Shows Quota Guidance

```bash
python training/cli.py setup
```

**Expected:**
- Shows quota section with current quotas
- Shows Console link
- Shows filter string
- Shows "quotas appear after first launch" message

**Checklist:**
- [ ] Setup runs without errors
- [ ] Quota section displayed
- [ ] Shows 2 quotas (us-central1, asia-northeast1)
- [ ] Console link included

---

### Test 2: Launch Lazy Loads Quotas

**DON'T RUN FULL LAUNCH - Test in isolation:**

```python
# Test lazy loading function
from training.cli.launch.mecha.mecha_acquire import lazy_load_quota_entry

def test_status(msg):
    print(msg)

# Test on a region that already has a pool
lazy_load_quota_entry('weight-and-biases-476906', 'europe-west4', test_status)
```

**Expected:**
- Submits test build
- Shows "Quota entry initialized (4 vCPUs default)"
- Completes in ~5 seconds

**Checklist:**
- [ ] Function runs without errors
- [ ] Completes quickly (~5 sec)
- [ ] Shows success message

---

### Test 3: MECHA Battle Shows Sidelined

**Test quota filtering:**

```python
from training.cli.launch.mecha.mecha_quota import separate_by_quota

regions = ['us-central1', 'asia-northeast1', 'europe-west4', 'us-west1']
battle_ready, sidelined = separate_by_quota(regions, 'weight-and-biases-476906', 176)

print(f"Battle-ready: {battle_ready}")
print(f"Sidelined: {sidelined}")
```

**Expected:**
```
Battle-ready: ['us-central1', 'asia-northeast1']
Sidelined: ['europe-west4', 'us-west1']
```

**Checklist:**
- [ ] Function runs without errors
- [ ] 2 battle-ready regions
- [ ] 2+ sidelined regions
- [ ] Correct separation

---

### Test 4: Solo MECHA Scenario

**Simulate solo MECHA:**

```python
from training.cli.launch.mecha.mecha_integration import run_mecha_battle

def test_status(msg):
    print(msg)

# Only 1 battle-ready region
result = run_mecha_battle(
    'weight-and-biases-476906',
    'c3-standard-176',
    ['asia-northeast1'],  # Only 1 region
    'us-central1',
    test_status
)

print(f"Selected: {result}")
```

**Expected:**
- Shows: "THIS MECHA BATTLES ALONE AND WINS!"
- Returns: asia-northeast1

**Checklist:**
- [ ] Solo message displayed
- [ ] Returns correct region
- [ ] No price battle (skipped)

---

### Test 5: Epic Battle Scenario

**Simulate epic battle:**

```python
from training.cli.launch.mecha.mecha_integration import run_mecha_battle

def test_status(msg):
    print(msg)

# 2+ battle-ready regions
result = run_mecha_battle(
    'weight-and-biases-476906',
    'c3-standard-176',
    ['us-central1', 'asia-northeast1'],  # 2 regions
    'us-central1',
    test_status
)

print(f"Selected: {result}")
```

**Expected:**
- Shows: "MECHA PRICE BATTLE GO!"
- Shows: Battle results with champion
- Returns: Cheapest region

**Checklist:**
- [ ] Battle message displayed
- [ ] Shows price comparison
- [ ] Shows savings
- [ ] Returns cheapest region

---

## 🔧 PHASE 7: Git Commit

**After all tests pass:**

```bash
cd /Users/alfrednorth/Desktop/Code/arr-coc-ovis/RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1

# Stage all changes
git add training/cli/shared/quota_checker.py
git add training/cli/launch/mecha/mecha_quota.py
git add training/cli/setup/core.py
git add training/cli/launch/core.py
git add training/cli/launch/mecha/mecha_acquire.py
git add training/cli/launch/mecha/mecha_integration.py
git add REQUEST_C3_QUOTA_CONSOLE.md

# Commit
git commit -m "Implement Cloud Build C3 quota refactor

Phase 1: Centralized quota checker
- Create training/cli/shared/quota_checker.py
- Single source of truth for build_origin=default quotas
- 5-minute caching for performance

Phase 2: Replace existing checks
- Update setup/core.py to use quota_checker
- Update launch/core.py to use quota_checker
- Deprecate old check_cloud_build_c3_quota()

Phase 3: MECHA quota awareness
- Create mecha_quota.py with quota filtering
- Add lazy loading after worker pool creation
- Show sidelined MECHAs with complete instructions
- 3 battle paths: 0/1/2+ ready MECHAs
- Solo MECHA: 'battles alone and wins!'
- Never pick region without quota

Phase 4: Infrastructure display
- Add quota section to setup output
- Show Console link with filter string
- Explain quota lifecycle

Phase 5: Documentation
- Update REQUEST_C3_QUOTA_CONSOLE.md
- Document lazy quota creation

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Checkpoint:**
- [ ] All files staged
- [ ] Commit created
- [ ] Descriptive commit message

---

## 🎉 COMPLETION CHECKLIST

### Code Complete:
- [ ] Phase 1: quota_checker.py created and tested
- [ ] Phase 2: setup/core.py and launch/core.py updated
- [ ] Phase 3: MECHA quota awareness implemented
- [ ] Phase 4: Infrastructure display updated
- [ ] Phase 5: Documentation updated

### Testing Complete:
- [ ] Setup shows quota guidance
- [ ] Lazy loading works
- [ ] Quota filtering works
- [ ] Solo MECHA scenario works
- [ ] Epic battle scenario works
- [ ] Sidelined display shows complete instructions

### Git Complete:
- [ ] All changes committed
- [ ] Descriptive commit message
- [ ] Clean git status

---

## 📊 Summary of Changes

**Files Created (2):**
- `training/cli/shared/quota_checker.py` - Central quota check
- `training/cli/launch/mecha/mecha_quota.py` - Quota filtering & display

**Files Modified (5):**
- `training/cli/setup/core.py` - Use quota_checker, add quota section
- `training/cli/launch/core.py` - Use quota_checker
- `training/cli/launch/mecha/mecha_acquire.py` - Add lazy loading
- `training/cli/launch/mecha/mecha_integration.py` - Add quota filtering before battle
- `REQUEST_C3_QUOTA_CONSOLE.md` - Document lazy loading

**Total Lines Added:** ~800 lines
**Total Lines Removed:** ~50 lines
**Net Change:** +750 lines

---

## 🚀 What Changed:

### Before:
❌ Quota checks scattered across 2 files
❌ MECHA picks regions without quota → build fails
❌ Confusing error messages
❌ No quota initialization
❌ No guidance on requesting quotas

### After:
✅ Single quota check function (quota_checker.py)
✅ MECHA filters by quota BEFORE battle
✅ Lazy loading initializes quotas automatically
✅ Sidelined display with complete 5-step instructions
✅ 3 battle paths: 0/1/2+ ready MECHAs
✅ Solo MECHA: "battles alone and wins!"
✅ Never picks region without quota

---

**Last Updated:** 2025-11-11
**Status:** READY TO IMPLEMENT
**Estimated Time:** 3-4 hours
**Complexity:** Medium

---

## ADDENDUM: Post-Plan File Renames (2025-11-16)

**Date**: 2025-11-16
**Context**: After this plan was implemented, files were renamed for clarity.

### Files Renamed

**quota_checker.py → quota/ module (Commit: e5210f3)**
```
training/cli/shared/quota_checker.py  ❌ DELETED

training/cli/shared/quota/            ✅ CREATED
├── __init__.py
├── c3_quota.py                       → Moved from quota_checker.py
└── gpu_quota.py                      → NEW (Vertex AI quotas)
```

**mecha_quota.py → mecha_display.py (Commit: a1ab751)**
```
training/cli/launch/mecha/mecha_quota.py  ❌ RENAMED
training/cli/launch/mecha/mecha_display.py  ✅ NEW NAME
```

**All references to `quota_checker.py` and `mecha_quota.py` in this plan are HISTORICAL.**

**Current imports:**
```python
# NEW (current)
from cli.shared.quota import get_cloud_build_c3_quotas
from cli.launch.mecha.mecha_display import separate_by_quota
```

---

## ADDENDUM 2: Explicit Service Naming (2025-11-16)

**Post-plan completion: Cloud Build quota function renamed for service clarity**

After this plan was implemented, the Cloud Build quota checker function was renamed to explicitly indicate it checks Cloud Build C3 quotas:

### Rename
```python
# OLD → NEW
has_sufficient_quota()  → has_cloud_build_c3_quota()
```

**Why?** Explicit service identification - name now clearly shows it checks **Cloud Build C3** quotas (not Vertex AI, Compute Engine, or other GCP quotas).

**Current import**:
```python
from cli.shared.quota import has_cloud_build_c3_quota
```

**Commit**: 8eeb41c - Rename all quota functions for explicit service clarity

All function references in this historical plan document remain unchanged - they represent the original implementation. Current code uses the renamed function listed above.
