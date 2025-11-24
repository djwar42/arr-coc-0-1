# Cloud Build C3 Quota Refactor Plan

**Goal:** Centralize quota checking and add quota-aware MECHA battle system

**Date:** 2025-11-11

---

## 🎯 Problem Statement

**Current Issues:**
1. Quota checks scattered across 2 files (`setup/core.py`, `launch/core.py`)
2. MECHA battle has NO quota checks → picks regions without quota → build fails
3. Misleading error: "quota restrictions" doesn't explain HOW to fix
4. Infrastructure screen doesn't show actual quota per region
5. No way to tell which MECHAs are usable vs sidelined

**What We Learned:**
- Only `build_origin=default` quota matters (what `gcloud builds submit` uses)
- Pools can exist without quota → builds fail with confusing error
- Only 2 regions have 176 vCPU quota: `us-central1`, `asia-northeast1`
- 5+ regions have pools but 0 quota (created by MECHA Fleet Blast)

---

## 📋 PHASE 1: Create Singular Quota Check Function

**File:** `training/cli/shared/quota_checker.py` (NEW)

### Step 1.1: Create quota_checker.py

**Function: `get_cloud_build_c3_quotas(project_id: str) -> Dict[str, int]`**

```python
"""
Get Cloud Build C3 quota for ALL regions (build_origin=default only)

This is THE definitive quota check for Cloud Build worker pools.

Args:
    project_id: GCP project ID

Returns:
    Dict mapping region → vCPUs quota (build_origin=default only)
    Example: {"us-central1": 176, "asia-northeast1": 176, "europe-west4": 0}

Why build_origin=default?
    - `gcloud builds submit` uses build_origin=default
    - This is the ACTUAL quota that matters for our builds
    - System Limits (no build_origin) are irrelevant
"""
```

**Implementation:**
- Call `gcloud alpha services quota list`
- Parse JSON for `concurrent_private_pool_c3_build_cpus`
- Filter for `build_origin=default` dimension ONLY
- Return clean dict: `{region: vcpus}`
- Add 5-minute cache (don't query every time!)

**Dependencies:** None

**Deliverable:**
- [ ] Create `training/cli/shared/quota_checker.py`
- [ ] Add `get_cloud_build_c3_quotas()` function
- [ ] Add docstring explaining `build_origin=default`
- [ ] Add caching (5 min TTL)
- [ ] Test: Returns `{"us-central1": 176, "asia-northeast1": 176}` for current project

---

### Step 1.2: Create convenience wrapper

**Function: `get_region_quota(project_id: str, region: str) -> int`**

```python
"""
Get Cloud Build C3 quota for a single region

Args:
    project_id: GCP project ID
    region: GCP region (e.g., "us-central1")

Returns:
    vCPUs quota for that region (0 if no quota)
"""
```

**Implementation:**
- Calls `get_cloud_build_c3_quotas()`
- Returns `quotas.get(region, 0)`

**Dependencies:** Step 1.1

**Deliverable:**
- [ ] Add `get_region_quota()` helper
- [ ] Test: `get_region_quota("proj", "us-central1")` → 176

---

### Step 1.3: Create quota sufficiency check

**Function: `has_sufficient_quota(project_id: str, region: str, required_vcpus: int) -> bool`**

```python
"""
Check if region has sufficient Cloud Build C3 quota

Args:
    project_id: GCP project ID
    region: GCP region
    required_vcpus: Minimum vCPUs needed (e.g., 176 for c3-standard-176)

Returns:
    True if region has >= required_vcpus, False otherwise
"""
```

**Implementation:**
- Calls `get_region_quota()`
- Returns `quota >= required_vcpus`

**Dependencies:** Step 1.2

**Deliverable:**
- [ ] Add `has_sufficient_quota()` helper
- [ ] Test: `has_sufficient_quota("proj", "us-central1", 176)` → True
- [ ] Test: `has_sufficient_quota("proj", "europe-west4", 176)` → False

---

## 📋 PHASE 2: Replace Existing Quota Checks

### Step 2.1: Update setup/core.py

**File:** `training/cli/setup/core.py`

**Changes:**
- Line 886: Replace `get_best_available_c3_machine()` logic
- Use new `quota_checker.get_region_quota()` instead
- Keep existing UI/messaging
- NO functional changes (just use new function)

**Before:**
```python
best_machine, best_vcpus, ce_quota, cb_quota, limiting_factor = get_best_available_c3_machine(project_id, region)
```

**After:**
```python
from cli.shared.quota_checker import get_region_quota

cb_quota = get_region_quota(project_id, region)
best_machine, best_vcpus = _select_best_c3_machine_for_quota(cb_quota)
```

**Dependencies:** Phase 1 complete

**Deliverable:**
- [ ] Replace quota check in `run_setup_core()`
- [ ] Test: Setup still shows correct quota and machine selection
- [ ] No functional changes

---

### Step 2.2: Update launch/core.py

**File:** `training/cli/launch/core.py`

**Changes:**
- Line 537: Replace `get_best_available_c3_machine()` call
- Use new `quota_checker.get_region_quota()` instead
- Keep existing UI/messaging
- NO functional changes

**Dependencies:** Phase 1 complete

**Deliverable:**
- [ ] Replace quota check in `run_launch_core()`
- [ ] Test: Launch still works with correct machine type
- [ ] No functional changes

---

### Step 2.3: Deprecate old check_cloud_build_c3_quota()

**File:** `training/cli/setup/core.py`

**Changes:**
- Add deprecation comment to `check_cloud_build_c3_quota()` (line 260)
- Add deprecation comment to `get_best_available_c3_machine()` (line 368)
- Keep functions for now (don't break anything)
- Mark for future removal

**Note:** Don't delete yet! May be used elsewhere.

**Dependencies:** Steps 2.1, 2.2 complete

**Deliverable:**
- [ ] Add deprecation comments
- [ ] Document replacement: "Use quota_checker.get_region_quota() instead"

---

## 📋 PHASE 3: Add MECHA Quota Awareness

### Step 3.1: Create MECHA quota filter function

**File:** `training/cli/launch/mecha/mecha_quota.py` (NEW)

**Function: `separate_by_quota(regions: List[str], project_id: str, required_vcpus: int) -> Tuple[List[str], List[str]]`**

```python
"""
Separate regions into battle-ready vs sidelined based on quota

Args:
    regions: List of MECHA regions to check
    project_id: GCP project ID
    required_vcpus: Minimum vCPUs needed (176 for c3-standard-176)

Returns:
    (battle_ready, sidelined) tuple
    - battle_ready: Regions with quota >= required_vcpus
    - sidelined: Regions with quota < required_vcpus
"""
```

**Implementation:**
- Calls `quota_checker.get_cloud_build_c3_quotas()`
- Filters regions by quota
- Returns two lists

**Dependencies:** Phase 1 complete

**Deliverable:**
- [ ] Create `training/cli/launch/mecha/mecha_quota.py`
- [ ] Add `separate_by_quota()` function
- [ ] Test: Returns `(["us-central1", "asia-northeast1"], ["europe-west4", ...])`

---

### Step 3.2: Create sidelined display function

**File:** `training/cli/launch/mecha/mecha_quota.py`

**Function: `display_sidelined_mechas(sidelined: List[str], quotas: Dict[str, int], project_id: str, status_callback)`**

```python
"""
Display sidelined MECHAs section with quota instructions

Shows:
- Count of sidelined MECHAs
- List of regions with quota status
- Exact Console filter string
- Direct link to quota page
- Step-by-step instructions
"""
```

**Implementation:**
- Show formatted sidelined section (see earlier design)
- Include exact filter string: `Concurrent C3 Build CPUs (Private Pool)`
- Include direct link: `https://console.cloud.google.com/iam-admin/quotas?project={project_id}`
- Show step-by-step instructions

**Dependencies:** Step 3.1

**Deliverable:**
- [ ] Add `display_sidelined_mechas()` function
- [ ] Test: Outputs formatted section with instructions

---

### Step 3.3: Create battle-ready display function

**File:** `training/cli/launch/mecha/mecha_quota.py`

**Function: `display_battle_ready_mechas(battle_ready: List[str], quotas: Dict[str, int], status_callback)`**

```python
"""
Display battle-ready MECHAs section

Shows:
- Count of battle-ready MECHAs
- List of regions with quota amounts
"""
```

**Dependencies:** Step 3.1

**Deliverable:**
- [ ] Add `display_battle_ready_mechas()` function
- [ ] Test: Outputs formatted section

---

### Step 3.4: Update MECHA integration with quota checks

**File:** `training/cli/launch/mecha/mecha_integration.py`

**Changes:**
- Import `mecha_quota` functions
- Before battle: Call `separate_by_quota()`
- Display sidelined section (if any)
- Display battle-ready section
- Add 3 battle paths:
  - **0 ready** → PRIMARY fallback
  - **1 ready** → "THIS MECHA BATTLES ALONE AND WINS!" 🏆
  - **2+ ready** → Epic battle

**Implementation:**
```python
def run_mecha_battle(...):
    # ... existing code ...

    # NEW: Check quota
    from .mecha_quota import separate_by_quota, display_sidelined_mechas, display_battle_ready_mechas
    from cli.shared.quota_checker import get_cloud_build_c3_quotas

    quotas = get_cloud_build_c3_quotas(project_id)
    vcpus_needed = int(best_machine.split("-")[-1])  # 176

    battle_ready, sidelined = separate_by_quota(acquired, project_id, vcpus_needed)

    # Show sidelined
    if sidelined:
        display_sidelined_mechas(sidelined, quotas, project_id, status)

    # Show battle-ready
    display_battle_ready_mechas(battle_ready, quotas, status)

    # Battle logic (3 paths)
    if len(battle_ready) == 0:
        status("❌ NO BATTLE-READY MECHAS!")
        status(f"   Falling back to PRIMARY: {primary_region}")
        return primary_region

    elif len(battle_ready) == 1:
        solo_region = battle_ready[0]
        status("")
        status("⚔️  THIS MECHA BATTLES ALONE AND WINS! ⚔️")
        status(f"   🏆 CHAMPION: {solo_region} |${get_region_price(solo_region):.2f}/hr|")
        status("   (No price comparison needed - it's the only option!)")
        return solo_region

    else:
        # Epic battle (2+ MECHAs)
        return epic_mecha_price_battle(battle_ready, status)
```

**Dependencies:** Steps 3.1, 3.2, 3.3

**Deliverable:**
- [ ] Update `run_mecha_battle()` with quota checks
- [ ] Add 3 battle paths (0, 1, 2+ ready)
- [ ] Test all 3 scenarios:
  - [ ] 0 ready → PRIMARY fallback
  - [ ] 1 ready → Solo win message
  - [ ] 2+ ready → Epic battle

---

## 📋 PHASE 4: Update Infrastructure Display

### Step 4.1: Add quota section to infra check

**File:** `training/cli/setup/core.py`

**Function:** `check_infrastructure_core()`

**Changes:**
- Add Cloud Build C3 quota to returned `info` dict
- Structure: `info["cloud_build_quota"] = {region: vcpus}`

**Implementation:**
```python
def check_infrastructure_core(...):
    # ... existing code ...

    # NEW: Check Cloud Build C3 quota
    from cli.shared.quota_checker import get_cloud_build_c3_quotas

    try:
        quotas = get_cloud_build_c3_quotas(project_id)
        info["cloud_build_quota"] = quotas
    except Exception:
        info["cloud_build_quota"] = {}

    return info
```

**Dependencies:** Phase 1 complete

**Deliverable:**
- [ ] Add quota to `check_infrastructure_core()`
- [ ] Test: `info["cloud_build_quota"]` contains quota dict

---

### Step 4.2: Update infrastructure tree display

**File:** `training/cli/setup/core.py`

**Function:** `display_infrastructure_tree()`

**Changes:**
- Add "Cloud Build C3 Quota" section after worker pool
- Show quota per region
- Show which regions are usable (pool + quota)

**Implementation:**
```python
def display_infrastructure_tree(info, status, config):
    # ... existing code ...

    # NEW: Show Cloud Build quota
    quotas = info.get("cloud_build_quota", {})

    if quotas:
        status("")
        status("[bold cyan]Cloud Build C3 Quota (build_origin=default):[/bold cyan]")

        # Show regions with quota
        regions_with_quota = {r: q for r, q in quotas.items() if q > 0}

        if regions_with_quota:
            status("[green]  ✓ Regions with quota:[/green]")
            for region, vcpus in sorted(regions_with_quota.items()):
                status(f"    • {region:25s} → {vcpus:3d} vCPUs")
        else:
            status("[yellow]  ⚠️  No regions have Cloud Build C3 quota![/yellow]")
            status("     Request quota: See REQUEST_C3_QUOTA_CONSOLE.md")
```

**Dependencies:** Step 4.1

**Deliverable:**
- [ ] Add quota display to `display_infrastructure_tree()`
- [ ] Test: Shows quota for regions
- [ ] Test: CLI `python training/cli.py infra` shows quota

---

## 📋 PHASE 5: Documentation Updates

### Step 5.1: Update REQUEST_C3_QUOTA_CONSOLE.md

**File:** `REQUEST_C3_QUOTA_CONSOLE.md`

**Changes:**
- [x] ✅ DONE: Explain System Limits vs Quotas
- [x] ✅ DONE: Show exact filter strings
- [ ] Add "Why `build_origin=default` matters" section
- [ ] Add troubleshooting for quota errors
- [ ] Add link to `quota_checker.py` for programmatic access

**Dependencies:** Phase 1 complete

**Deliverable:**
- [ ] Add `build_origin=default` explanation
- [ ] Add link to quota_checker.py
- [ ] Update troubleshooting section

---

### Step 5.2: Update CLAUDE.md or create QUOTA_GUIDE.md

**File:** `CLAUDE.md` (or new `QUOTA_GUIDE.md`)

**Changes:**
- [ ] Document `quota_checker.py` API
- [ ] Explain pool vs quota distinction
- [ ] Show how to use quota checks in code
- [ ] Link to Console instructions

**Dependencies:** Phase 1 complete

**Deliverable:**
- [ ] Add quota checking guide
- [ ] Include code examples
- [ ] Link to REQUEST_C3_QUOTA_CONSOLE.md

---

## 📋 PHASE 6: Testing & Validation

### Step 6.1: Unit tests for quota_checker.py

**File:** `tests/test_quota_checker.py` (NEW)

**Tests:**
- [ ] `test_get_cloud_build_c3_quotas()` - Returns expected dict
- [ ] `test_get_region_quota()` - Returns correct quota for region
- [ ] `test_has_sufficient_quota()` - True/False logic
- [ ] `test_caching()` - Doesn't re-query within 5 min

**Dependencies:** Phase 1 complete

---

### Step 6.2: Integration tests

**Tests:**
- [ ] Setup with quota → Selects correct machine
- [ ] Launch with quota → Uses correct region
- [ ] MECHA with 0 ready → PRIMARY fallback
- [ ] MECHA with 1 ready → Solo win message
- [ ] MECHA with 2+ ready → Epic battle
- [ ] Infra screen → Shows quota correctly

**Dependencies:** All phases complete

---

### Step 6.3: Manual testing

**Scenarios:**
- [ ] `python training/cli.py infra` → Shows quota
- [ ] `python training/cli.py setup` → Uses quota-aware machine selection
- [ ] `python training/cli.py launch` → Shows sidelined MECHAs
- [ ] Launch with only asia-northeast1 → Solo win message
- [ ] Launch with both regions → Epic battle

**Dependencies:** All phases complete

---

## 📋 PHASE 7: Cleanup

### Step 7.1: Remove deprecated functions

**File:** `training/cli/setup/core.py`

**Remove:**
- [ ] `check_cloud_build_c3_quota()` (line 260) - After confirming no other usage
- [ ] `get_best_available_c3_machine()` (line 368) - After confirming no other usage

**Before removal:**
- [ ] Search entire codebase for usage
- [ ] Confirm only used in setup/launch core (which we replaced)

**Dependencies:** Phases 1-6 complete

---

### Step 7.2: Remove dead code

**Files to check:**
- [ ] `training/cli/shared/c3_quota_instruct.py` - Still needed?
- [ ] Any temp test files

**Dependencies:** Phase 7.1

---

## 🎯 Success Criteria

**Phase 1:**
- [ ] ✅ Single source of truth: `quota_checker.get_cloud_build_c3_quotas()`
- [ ] ✅ Returns quota for `build_origin=default` ONLY
- [ ] ✅ Cached for performance

**Phase 2:**
- [ ] ✅ All existing quota checks use new function
- [ ] ✅ No functional changes (backward compatible)

**Phase 3:**
- [ ] ✅ MECHA never picks region without quota
- [ ] ✅ Sidelined MECHAs shown with instructions
- [ ] ✅ Solo MECHA shows "battles alone and wins!"
- [ ] ✅ Epic battle only with battle-ready MECHAs

**Phase 4:**
- [ ] ✅ Infra screen shows quota per region
- [ ] ✅ Clear which regions are usable

**Phase 5:**
- [ ] ✅ Documentation explains quota system
- [ ] ✅ Users know how to request quota

**Phase 6:**
- [ ] ✅ All tests pass
- [ ] ✅ Manual testing confirms expected behavior

**Phase 7:**
- [ ] ✅ No deprecated code remains
- [ ] ✅ Codebase clean

---

## 📊 Priority Order

**HIGH (Do First):**
1. Phase 1 - Create quota_checker.py (FOUNDATION)
2. Phase 2 - Replace existing checks (COMPATIBILITY)
3. Phase 3 - Add MECHA quota awareness (FIX ERRORS)

**MEDIUM:**
4. Phase 4 - Update infra display (VISIBILITY)
5. Phase 5 - Documentation (USER GUIDANCE)

**LOW (Cleanup):**
6. Phase 6 - Testing (VALIDATION)
7. Phase 7 - Remove deprecated (CLEANUP)

---

## 🚀 Estimated Timeline

- **Phase 1:** 30-45 min (create new function)
- **Phase 2:** 20-30 min (replace existing)
- **Phase 3:** 45-60 min (MECHA integration)
- **Phase 4:** 20-30 min (infra display)
- **Phase 5:** 15-20 min (docs)
- **Phase 6:** 30-45 min (testing)
- **Phase 7:** 10-15 min (cleanup)

**Total:** ~3-4 hours

---

## 🔗 File Dependencies

```
quota_checker.py (NEW)
    ├─→ setup/core.py (replace check)
    ├─→ launch/core.py (replace check)
    ├─→ mecha_quota.py (NEW)
    │       ├─→ mecha_integration.py (use quota)
    │       └─→ mecha_battle_epic.py (battle logic)
    └─→ setup/core.py (infra display)
```

---

## ⚠️ Risks & Mitigation

**Risk 1: Breaking existing quota checks**
- **Mitigation:** Keep old functions until Phase 7
- **Mitigation:** Test thoroughly in Phase 2

**Risk 2: MECHA picks wrong region**
- **Mitigation:** Phase 3 filters BEFORE battle
- **Mitigation:** Fallback to PRIMARY if no quota

**Risk 3: Cache stale quota data**
- **Mitigation:** 5-minute TTL (balance freshness vs API calls)
- **Mitigation:** Users can re-run if quota just approved

**Risk 4: `build_origin=default` assumption wrong**
- **Mitigation:** Tested with real Cloud Build submissions (Phase 6)
- **Mitigation:** Documented why this dimension matters

---

## 📝 Git Commit Strategy

**After Phase 1:**
```
Add centralized Cloud Build C3 quota checker

- Create training/cli/shared/quota_checker.py
- Single source of truth for build_origin=default quota
- 5-minute caching for performance
- Foundation for MECHA quota awareness
```

**After Phase 2:**
```
Replace scattered quota checks with quota_checker

- Update setup/core.py to use quota_checker
- Update launch/core.py to use quota_checker
- Deprecate old check_cloud_build_c3_quota()
- No functional changes (backward compatible)
```

**After Phase 3:**
```
Add MECHA quota awareness with sidelined display

- Create mecha_quota.py with quota filtering
- Show sidelined MECHAs with Console instructions
- 3 battle paths: 0/1/2+ ready MECHAs
- Solo MECHA: "battles alone and wins!"
- Never pick region without quota
```

**After Phase 4:**
```
Add Cloud Build quota to infrastructure display

- Show quota per region in infra screen
- Indicate which regions are usable (pool + quota)
```

**After Phase 5:**
```
Update quota documentation

- Explain build_origin=default importance
- Add quota_checker.py API docs
```

**After Phase 7:**
```
Remove deprecated quota check functions

- Delete check_cloud_build_c3_quota()
- Delete get_best_available_c3_machine()
- Cleanup complete
```

---

## 🔬 ADDENDUM: Critical Quota System Discoveries (2025-11-11)

### Discovery 1: Lazy Quota Entry Creation

**MASSIVE INSIGHT: Quota entries don't exist until you try to use a region!**

#### What We Discovered:

**Test Setup:**
- Submitted Cloud Build test to 3 regions:
  - `asia-northeast1` (known to have 176 vCPUs quota)
  - `europe-west4` (no quota entry before test)
  - `us-west1` (no quota entry before test)

**Test Results:**
```
asia-northeast1:  SUCCESS ✅ (176 vCPUs quota already existed)
europe-west4:     FAILED ❌ (quota restriction error)
us-west1:         FAILED ❌ (quota restriction error)
```

**Console BEFORE test builds:**
```
System Limits (Adjustable: No):
- europe-west4: 4 vCPUs (System Limit only - read-only)
- us-west1: 4 vCPUs (System Limit only - read-only)

Quotas (Adjustable: Yes):
- us-central1: 176 vCPUs (manually requested earlier)
- asia-northeast1: 176 vCPUs (manually requested earlier)
```

**Console AFTER test builds:**
```
System Limits (Adjustable: No):
- (unchanged)

Quotas (Adjustable: Yes):  ⭐ NEW ENTRIES APPEARED!
- us-central1: 176 vCPUs
- asia-northeast1: 176 vCPUs
- europe-west4: 4 vCPUs ⭐ (NEW! Created after failed build attempt)
- us-west1: 4 vCPUs ⭐ (NEW! Created after failed build attempt)
```

#### Key Findings:

✅ **Submitting a build attempt triggers quota entry creation**
✅ **Even FAILED builds create quota entries** (quota restriction error = still creates entry!)
✅ **New entries appear with DEFAULT value (4 vCPUs)**
✅ **After entry exists → you can request increase to 176 vCPUs**

#### Quota Entry Lifecycle:

```
State 1: System Limit Only (read-only)
  ├─ Region has NO quota entry
  ├─ Console shows: System Limit (4 vCPUs, Adjustable: No)
  └─ CANNOT request quota increase (no entry to edit!)

      ↓ Submit first build attempt (even if it fails!)

State 2: Quota Entry Created (editable)
  ├─ Quota entry appears with DEFAULT value (4 vCPUs)
  ├─ Console shows: Quota (4 vCPUs, Adjustable: Yes) ⭐
  └─ CAN NOW request quota increase to 176 vCPUs!

      ↓ Request increase → Google approves

State 3: Quota Approved (usable)
  ├─ Quota entry updated to requested value (176 vCPUs)
  ├─ Console shows: Quota (176 vCPUs, Adjustable: Yes)
  └─ Region is now USABLE for builds!
```

---

### Discovery 2: Dimension Ordering Differences

**CRITICAL: Console displays quota dimensions in DIFFERENT order based on creation method!**

#### Old Entries (Manually Requested):
```
Dimensions: build_origin : default → region : us-central1
Value: 176 vCPUs
```

#### New Entries (Auto-Created on First Build):
```
Dimensions: region : europe-west4 → build_origin : default
Value: 4 vCPUs
```

**Observation:** Google uses different dimension ordering based on HOW the quota entry was created!

| Creation Method | Dimension Order | Example | Value |
|----------------|-----------------|---------|-------|
| Manual Console Request | `build_origin` → `region` | Old entries (us-central1) | 176 vCPUs |
| Auto-Created on First Use | `region` → `build_origin` | New entries (europe-west4) | 4 vCPUs |

**Implication for quota_checker.py:**
- MUST handle BOTH dimension orderings!
- Parse dimensions dict WITHOUT assuming key order
- Extract `region` and `build_origin` by key, not position

---

### Discovery 3: Console vs API Differences

**CRITICAL: Console UI shows different data than API responses!**

#### Console UI Shows (what user sees):
```
Quotas (build_origin=default):
1. us-central1: 176 vCPUs (build_origin=default, region=us-central1)
2. asia-northeast1: 176 vCPUs (build_origin=default, region=asia-northeast1)
3. europe-west4: 4 vCPUs (region=europe-west4, build_origin=default) ⭐
4. us-west1: 4 vCPUs (region=us-west1, build_origin=default) ⭐
```

#### API Response Shows (what gcloud sees):
```json
// System Limits (20 regions) - Only 'region' dimension:
{
  "dimensions": {"region": "europe-west4"},
  "effectiveLimit": "4"
}

// Custom Quotas (2 regions) - BOTH dimensions:
{
  "dimensions": {
    "build_origin": "default",
    "region": "us-central1"
  },
  "effectiveLimit": "176"
}
```

**Key Difference:**
- Console: Shows europe-west4 with BOTH dimensions after test
- API: Shows europe-west4 with ONLY 'region' dimension (no 'build_origin')

**Possible Explanations:**
1. Console uses different API endpoint (quota preferences vs limits)
2. Quota entries exist in different GCP service (not returned by `gcloud alpha services quota list`)
3. Console caches/displays quota request history separately

**Implication:**
- Our quota_checker.py should use the API that Console uses
- May need to investigate Console's actual API calls (Network tab in DevTools)
- Current `gcloud alpha services quota list` may not show auto-created entries

---

### Discovery 4: MECHA Fleet Blast is a Feature!

**Reframe: MECHA creating pools everywhere is actually BRILLIANT!**

#### What We Originally Thought:
- MECHA creates pools in regions without quota → wasteful 😞
- Pools exist but builds fail → confusing errors 😞
- Need to manually request quota for each region → tedious 😞

#### What We Now Know:
- MECHA creates pools → **triggers quota entry creation** → unlocks quota requests! 🎯
- Build failures → **create quota entries** → enables future increases! 🎯
- MECHA Fleet Blast → **initializes quota system for all regions** → strategic! 🎯

**MECHA Fleet Blast = Quota Entry Initialization System!**

#### Benefits of MECHA's Approach:

✅ **Unlocks quota requests:** Before MECHA, regions have System Limits only (can't request)
✅ **After MECHA:** Regions have quota entries (can request 176 vCPUs!)
✅ **One-time setup:** MECHA does the initialization work once
✅ **Future-proof:** New regions immediately requestable after MECHA creates pool

#### Updated Perspective:

```
Old View:
MECHA creates pools everywhere
  → Some regions fail (no quota)
  → Wasted effort

New View:
MECHA creates pools everywhere
  → Triggers quota entry creation for ALL regions
  → Unlocks ability to request 176 vCPUs
  → Strategic infrastructure initialization
  → One-time cost, long-term benefit
```

---

### Discovery 5: Sidelined MECHAs Message Should Be Positive!

**Current plan:** "MECHAs sidelined due to missing quota" (negative framing)

**Better framing:** "MECHAs initialized - ready for quota requests!" (positive framing)

#### Updated Message Template:

```
⚠️  SIDELINED MECHAS (Default Quota - Can Be Increased!):
╔════════════════════════════════════════════════════════════╗
║ 3 MECHAs have pools with DEFAULT quota (4 vCPUs)          ║
║ GOOD NEWS: Quota entries now exist - request increases!   ║
╚════════════════════════════════════════════════════════════╝

Regions initialized by MECHA Fleet Blast:
  • europe-west4       (Quota entry: 4 vCPUs ⚠️  DEFAULT)
  • us-west1           (Quota entry: 4 vCPUs ⚠️  DEFAULT)

💡 WHY THIS IS PROGRESS:

Before MECHA created pools:
  ❌ No quota entries → Cannot request increases
  ❌ System Limits only (read-only)

After MECHA created pools:
  ✅ Quota entries exist (4 vCPUs)
  ✅ Can now request increases to 176 vCPUs!
  ✅ MECHA unlocked quota system for these regions!

[... quota request instructions ...]
```

**Key messaging changes:**
- "Sidelined" → "Initialized"
- "Missing quota" → "Default quota (can be increased!)"
- Emphasize MECHA unlocked quota requests
- Show this as progress, not failure

---

### Implications for Refactor

#### Phase 1 (quota_checker.py):
- **Handle dimension ordering:** Parse dimensions dict by key, not order
- **Filter strategy:** Look for entries with BOTH 'region' AND 'build_origin=default'
- **API investigation:** May need to find correct API endpoint (not just `gcloud alpha services quota list`)
- **Fallback logic:** If API doesn't show auto-created entries, handle gracefully

#### Phase 3 (MECHA quota awareness):
- **Positive framing:** Sidelined = initialized, not broken
- **Show progress:** Quota entries exist = can request increases
- **MECHA value prop:** Fleet Blast unlocked quota system
- **Instructions:** Guide users through requesting 176 vCPUs for initialized regions

#### Phase 5 (Documentation):
- **Lazy quota creation:** Document lifecycle (System Limit → Quota Entry → Approved)
- **MECHA's role:** Explain Fleet Blast as initialization system
- **Dimension ordering:** Document both formats, show examples
- **Console vs API:** Note differences, show both views

#### Testing Considerations:
- **Verify API response:** Check if auto-created entries appear in API
- **Test dimension parsing:** Handle both orderings correctly
- **Quota entry detection:** Distinguish System Limits from Quota Entries
- **Edge cases:** What if region has quota entry but 0 vCPUs? (vs no entry at all)

---

### Updated Success Criteria

**Phase 1:**
- [ ] ✅ Handles BOTH dimension orderings (old vs new)
- [ ] ✅ Correctly identifies quota entries vs System Limits
- [ ] ✅ Filters for `build_origin=default` correctly
- [ ] ✅ Returns quota for regions with actual entries (not System Limits)

**Phase 3:**
- [ ] ✅ Distinguishes 3 states: No entry (System Limit), Entry (4 vCPUs), Entry (176 vCPUs)
- [ ] ✅ Positive messaging for initialized regions
- [ ] ✅ Shows MECHA Fleet Blast value (unlocked quota requests)
- [ ] ✅ Clear instructions for requesting 176 vCPUs

**Phase 5:**
- [ ] ✅ Documents lazy quota creation lifecycle
- [ ] ✅ Explains MECHA's quota initialization role
- [ ] ✅ Shows examples of both dimension orderings
- [ ] ✅ Notes Console vs API differences

---

### Open Questions

1. **API Endpoint:**
   - Which API does Console use to show quota entries with both dimensions?
   - Does `gcloud alpha services quota list` show auto-created entries?
   - Need to investigate Console's actual API calls (Network DevTools)

2. **Dimension Ordering:**
   - Is ordering guaranteed/stable, or just coincidental?
   - Does API always return dimensions in same order for same entry type?
   - Should we rely on ordering or always parse by key?

3. **Quota Entry Types:**
   - Are auto-created entries different object type than manually requested?
   - Do they live in different GCP services/namespaces?
   - Why does Console show them but API doesn't?

4. **MECHA Initialization:**
   - Should we document Fleet Blast as official quota initialization step?
   - Add explicit "initialize quota entries" command?
   - Show quota entry creation as success metric in setup?

---

### Next Steps (Investigation)

Before implementing Phase 1:

1. **Check Console Network tab:**
   ```
   Open Console → Quotas page
   Open DevTools → Network tab
   Refresh page
   Find API call that returns quota entries with both dimensions
   Copy curl command or API endpoint
   ```

2. **Test different gcloud commands:**
   ```bash
   # Try quota preferences API (if it exists)
   gcloud alpha services quota preferences list ...

   # Try different filters
   gcloud alpha services quota list --filter="dimensions.build_origin=default"
   ```

3. **Verify one more region:**
   ```bash
   # Submit test build to asia-southeast1 (hasn't been tested yet)
   # Check if quota entry appears in Console
   # Confirm lazy creation pattern holds
   ```

4. **Document exact API calls:**
   - Record exact gcloud command that shows quota entries correctly
   - Note any discrepancies between Console and CLI
   - Update quota_checker.py implementation plan accordingly

---

**Last Updated:** 2025-11-11 (ADDENDUM 1 ADDED)
**Status:** INVESTIGATING (API discrepancy)
**Next Step:** Add Addendum 2 with API findings → Full lifecycle test

---

## 🔬 ADDENDUM 2: API Investigation & Console Discrepancy (2025-11-11)

### Finding 1: API vs Console Show Different Data

**Confirmed discrepancy between `gcloud` API and Console UI:**

#### gcloud API Response (`gcloud alpha services quota list`):
```python
Custom Quotas (build_origin=default): 2 entries
  - us-central1:      176 vCPUs (dimensions: build_origin → region) ✅
  - asia-northeast1:  176 vCPUs (dimensions: build_origin → region) ✅

System Limits (region only): 20 entries
  - europe-west4:     4 vCPUs (dimensions: region ONLY) ❌
  - us-west1:         4 vCPUs (dimensions: region ONLY) ❌
  - asia-southeast1:  4 vCPUs (dimensions: region ONLY) ❌
  - ... 17 more
```

#### Console UI Shows (after test builds):
```
Quotas (Adjustable: Yes):
  - us-central1:      176 vCPUs (build_origin=default, region) ✅
  - asia-northeast1:  176 vCPUs (build_origin=default, region) ✅
  - europe-west4:     4 vCPUs (region, build_origin=default) ⭐ NEW!
  - us-west1:         4 vCPUs (region, build_origin=default) ⭐ NEW!
```

**Key Difference:**
- Console shows europe-west4 and us-west1 with **BOTH dimensions** (region + build_origin)
- API shows same regions with **ONLY region dimension** (no build_origin)

---

### Finding 2: Quota Bucket Structure Analysis

**From API JSON structure:**

```json
// System Limit bucket (europe-west4):
{
  "dimensions": {"region": "europe-west4"},
  "defaultLimit": "4",
  "effectiveLimit": "4",
  "quotaValue": null  // ⚠️ NULL!
}

// Custom Quota bucket (us-central1):
{
  "dimensions": {
    "build_origin": "default",
    "region": "us-central1"
  },
  "defaultLimit": null,
  "effectiveLimit": "176",
  "quotaValue": {"value": "176"}  // ✅ Has quotaValue object!
}
```

**Observations:**
1. System Limits have `quotaValue: null`
2. Custom Quotas have `quotaValue: {value: "176"}`
3. effectiveLimit shows current enforced value
4. defaultLimit shows Google's default (4 vCPUs for C3)

---

### Finding 3: Quota Override Commands Exist

**Found quota override API commands:**
```bash
gcloud alpha services quota override create
gcloud alpha services quota override delete
gcloud alpha services quota override update
```

**Hypothesis:**
Console might be showing "quota overrides" that are tracked separately from quotaBuckets!

**To test:**
```bash
# Try listing overrides (if this command exists)
gcloud alpha services quota override list \
  --service=cloudbuild.googleapis.com \
  --consumer=projects/weight-and-biases-476906
```

---

### Finding 4: Lazy Creation Confirmed (3rd Test)

**Test 3: asia-southeast1**
- Submitted build → FAILED (quota restriction)
- Expected result: Quota entry appears in Console with 4 vCPUs

**Pattern confirmed for 3 regions:**
1. europe-west4: Test → Failed → Console shows new entry
2. us-west1: Test → Failed → Console shows new entry
3. asia-southeast1: Test → Failed → (check Console to confirm)

---

### Finding 5: Console Uses Different API Endpoint

**Evidence:**
1. gcloud API doesn't show auto-created quota entries
2. Console shows them immediately after build attempt
3. Quota override commands exist (separate system)

**Most likely explanation:**

Console uses a **different API endpoint** that shows:
- Actual quota overrides (what gcloud shows)
- **Pending quota overrides** (what Console shows but gcloud doesn't)
- Quota request history

**To confirm:**
- Open Console → Quotas page
- Open DevTools → Network tab
- Look for API calls to `serviceusage.googleapis.com/v1beta1/...`
- Check if there's a `/quotaPreferences` or `/quotaOverrides` endpoint

---

### Finding 6: Practical Implication for quota_checker.py

**Problem:**
- gcloud API doesn't show auto-created quota entries
- We can't distinguish "no quota entry" from "quota entry with 4 vCPUs"
- Both show as: `{"region": "europe-west4", "effectiveLimit": "4"}`

**Solutions:**

#### Option A: Use gcloud override API (if exists)
```python
def get_quota_overrides(project_id, region):
    """Check if quota override exists for region"""
    # Try: gcloud alpha services quota override list
    # Look for build_origin=default overrides
```

#### Option B: Assume System Limits for non-176 values
```python
def get_cloud_build_c3_quotas(project_id):
    """
    Returns {region: vcpus}

    Only includes regions with CUSTOM quotas (build_origin=default dimension)
    Excludes System Limits (region-only dimension)
    """
    # Filter for buckets with build_origin=default
    # Return only those with effectiveLimit != defaultLimit
```

#### Option C: Conservative approach (current best)
```python
def get_cloud_build_c3_quotas(project_id):
    """
    Returns {region: vcpus} for build_origin=default quotas only

    Rules:
    1. Include: Buckets with BOTH dimensions (build_origin + region)
    2. Exclude: Buckets with ONLY region dimension (System Limits)
    3. Result: Only shows manually approved quotas (176 vCPUs)
    """
    quotas = {}
    for bucket in quota_buckets:
        dims = bucket['dimensions']
        if 'build_origin' in dims and dims['build_origin'] == 'default':
            if 'region' in dims:
                quotas[dims['region']] = int(bucket['effectiveLimit'])

    return quotas
```

**Chosen approach: Option C (Conservative)**

**Why:**
- Works with current API (no guessing)
- Only returns confirmed quotas (176 vCPUs)
- Safe: Won't include unconfirmed System Limits
- Downside: Won't show auto-created 4 vCPU entries

**Result:**
- Battle-ready MECHAs: Regions with 176 vCPUs (confirmed)
- Sidelined MECHAs: ALL other regions (includes both System Limits and auto-created 4 vCPU entries)
- User sees: "These regions need quota increases" (correct guidance)

---

### Finding 7: MECHA Battle Strategy

**Given API limitations, here's the battle strategy:**

```python
def run_mecha_battle(project_id, regions, primary_region):
    # Step 1: Get quotas from API (only returns 176 vCPU regions)
    quotas = get_cloud_build_c3_quotas(project_id)  # {'asia-northeast1': 176, 'us-central1': 176}

    # Step 2: Filter acquired MECHAs
    battle_ready = [r for r in regions if quotas.get(r, 0) >= 176]  # Only 176 vCPU regions
    sidelined = [r for r in regions if r not in battle_ready]       # Everything else

    # Step 3: Show sidelined (includes System Limits AND auto-created)
    display_sidelined_mechas(sidelined, quotas, project_id)

    # Step 4: Battle with battle-ready only
    if len(battle_ready) == 0:
        return primary_region  # Fallback
    elif len(battle_ready) == 1:
        return battle_ready[0]  # Solo win!
    else:
        return epic_price_battle(battle_ready)  # Fight!
```

**Messaging:**
```
⚠️ SIDELINED MECHAS: 3/5

These regions need Cloud Build C3 quota increases:
  • europe-west4
  • us-west1
  • asia-southeast1

Some may already have quota entries (4 vCPUs default).
Request increases to 176 vCPUs in Console.

[... instructions ...]
```

---

### Implications for Refactor Plan

#### Phase 1 (quota_checker.py) - UPDATED:
```python
def get_cloud_build_c3_quotas(project_id: str) -> Dict[str, int]:
    """
    Get Cloud Build C3 quota for regions with CONFIRMED custom quotas

    Returns only regions with build_origin=default dimension
    (These are manually approved quotas, typically 176 vCPUs)

    Does NOT return System Limits or auto-created 4 vCPU entries
    (API limitation - these show up in Console but not in gcloud API)

    Returns:
        {region: vcpus} for confirmed quotas only
        Example: {"us-central1": 176, "asia-northeast1": 176}
    """
```

#### Phase 3 (MECHA) - UPDATED:
```python
def display_sidelined_mechas(sidelined, quotas, project_id):
    """
    Show sidelined MECHAs with nuanced messaging

    Note: Some sidelined regions may have 4 vCPU quota entries
    (auto-created on first build attempt) but API doesn't show them.

    Messaging:
    - "These regions need quota increases"
    - "Some may already have default quota entries (4 vCPUs)"
    - "Request increases to 176 vCPUs in Console"
    """
```

---

### Recommended Next Steps

**Before implementing Phase 1:**

1. **Full Lifecycle Test (pick 1 region):**
   ```bash
   # Pick: australia-southeast1 (not yet tested)

   # Step 1: Submit test build → Fails (quota restriction)
   # Step 2: Check Console → Quota entry appears (4 vCPUs)
   # Step 3: Request increase via Console → 176 vCPUs
   # Step 4: Wait for Google approval → (1-2 days)
   # Step 5: Check API → Quota entry appears with build_origin=default!
   ```

2. **Console Network Investigation:**
   ```
   Open Console → Quotas page
   DevTools → Network tab
   Filter: serviceusage
   Look for API calls showing quota entries
   ```

3. **Try quota override list:**
   ```bash
   gcloud alpha services quota override list \
     --service=cloudbuild.googleapis.com \
     --consumer=projects/weight-and-biases-476906 \
     --format=json
   ```

---

### Key Takeaways

✅ **gcloud API only shows confirmed quotas** (build_origin=default dimension)
✅ **Console shows more** (includes auto-created 4 vCPU entries)
✅ **quota_checker.py will use conservative approach** (only confirmed quotas)
✅ **MECHA sidelined messaging updated** (accounts for API limitation)
✅ **Full lifecycle test needed** (confirm quota entry appears in API after approval)

---

**Last Updated:** 2025-11-11 (ADDENDUM 2 ADDED)
**Status:** READY FOR FULL LIFECYCLE TEST
**Next Step:** Test australia-southeast1 full lifecycle → Then implement Phase 1

---

## 🏗️ ADDENDUM 3: FINAL STRUCTURE - Lazy Loading & Quota Instructions (2025-11-11)

### Design Decision: Where Does Lazy Loading Happen?

**❌ NOT in Setup:**
- Setup is read-only infrastructure check
- Doesn't create worker pools or submit builds
- Just reports current state

**✅ IN MECHA ACQUIRE/BEACON:**
- Worker pool creation already happens here
- Immediately after pool created → Submit test build
- **Triggers lazy quota entry creation** (instant after pool ready)
- Result: Quota entry appears in Console (4 vCPUs, requestable)

**✅ IN BATTLE (Sidelined Display):**
- Show FULL step-by-step instructions to user
- Quota entries should already exist (lazy loaded during acquire)
- User just needs to request increase to 176 vCPUs

---

### Implementation: MECHA Acquire Lazy Loading

**File:** `training/cli/launch/mecha/mecha_acquire.py`

**Add after worker pool creation:**

```python
def acquire_mecha_for_region(project_id, region, machine_type, status_callback):
    """
    Acquire MECHA (create worker pool) and lazy load quota entry
    """
    # Step 1: Create worker pool (existing code)
    status(f"Creating worker pool in {region}...")
    create_worker_pool(project_id, region, machine_type)
    status(f"✅ Worker pool created: {region}")

    # Step 2: NEW - Lazy load quota entry
    status(f"")
    status(f"Initializing quota entry for {region}...")
    lazy_load_quota_entry(project_id, region)
    status(f"✅ Quota entry initialized (4 vCPUs default)")
    status(f"   You can now request 176 vCPUs in Console!")

    return True


def lazy_load_quota_entry(project_id, region):
    """
    Submit test build to trigger quota entry creation

    This creates a quota entry with 4 vCPUs (default).
    User can then request increase to 176 vCPUs via Console.

    Takes ~5 seconds.
    """
    import subprocess
    import tempfile

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
```

**Output during MECHA acquire:**
```
🏭 MECHA FLEET BLAST: Acquiring 5 regions

Region: europe-west4
  Creating worker pool in europe-west4...
  ✅ Worker pool created: europe-west4

  Initializing quota entry for europe-west4...
  ✅ Quota entry initialized (4 vCPUs default)
     You can now request 176 vCPUs in Console!

Region: us-west1
  Creating worker pool in us-west1...
  ✅ Worker pool created: us-west1

  Initializing quota entry for us-west1...
  ✅ Quota entry initialized (4 vCPUs default)
     You can now request 176 vCPUs in Console!

[... etc for all regions ...]
```

---

### Implementation: Sidelined MECHA Instructions

**File:** `training/cli/launch/mecha/mecha_quota.py`

**Full step-by-step instructions with direct link:**

```python
def display_sidelined_mechas(sidelined, quotas, project_id, status):
    """
    Show sidelined MECHAs with COMPLETE quota increase instructions
    """
    if not sidelined:
        return

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
```

**Output during battle:**
```
⚠️  SIDELINED MECHAS (Need Quota Increases):
╔════════════════════════════════════════════════════════════╗
║ 3 MECHAs have worker pools but need quota increases       ║
╚════════════════════════════════════════════════════════════╝

Regions with pools but insufficient quota:
  • europe-west4              (Current: 4 vCPUs, Need: 176)
  • us-west1                  (Current: 4 vCPUs, Need: 176)
  • asia-southeast1           (Current: 4 vCPUs, Need: 176)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 COMPLETE STEPS TO INCREASE QUOTAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  OPEN QUOTAS CONSOLE (link has filter pre-applied):

   https://console.cloud.google.com/apis/api/cloudbuild.googleapis.com/quotas?project=weight-and-biases-476906&pageState=(...)

2️⃣  VERIFY FILTER (should already be applied):

   If filter isn't applied, type EXACTLY:
   ┌────────────────────────────────────────────────────────┐
   │ Concurrent C3 Build CPUs (Private Pool)                │
   └────────────────────────────────────────────────────────┘

   ⚠️  IMPORTANT:
      - NO '(default)' suffix!
      - This shows YOUR quotas (not System Limits)

3️⃣  IDENTIFY QUOTA ENTRIES (ignore System Limits):

   You should see:
   ✅ Type: 'Quota' (Adjustable: Yes) ← THESE ARE YOUR QUOTAS
   ❌ Type: 'System limit' (Adjustable: No) ← IGNORE THESE

   Look for your sidelined regions:
      • europe-west4              → Should show 4 vCPUs
      • us-west1                  → Should show 4 vCPUs
      • asia-southeast1           → Should show 4 vCPUs

4️⃣  REQUEST QUOTA INCREASE:

   For EACH sidelined region:
   a) Click on the quota row
   b) Click 'EDIT QUOTAS' button (top right)
   c) Enter new limit:

      ┌────────────────────────────────────────┐
      │ Recommended Options:                   │
      │                                        │
      │ • 176 vCPUs (RECOMMENDED - CHONK!)     │
      │   → c3-standard-176 machine            │
      │   → Fastest builds (~20 min)           │
      │                                        │
      │ • 88 vCPUs (Good middle ground)        │
      │   → c3-standard-88 machine             │
      │   → ~25 min builds                     │
      │                                        │
      │ • 44 vCPUs (Budget option)             │
      │   → c3-standard-44 machine             │
      │   → ~35 min builds                     │
      └────────────────────────────────────────┘

   d) Justification (copy this):
      'Cloud Build worker pools for PyTorch compilation.
       Requesting C3 quota for multi-region builds.
       Expected usage: 1-2 concurrent builds.'

   e) Click 'SUBMIT REQUEST'

5️⃣  WAIT FOR GOOGLE APPROVAL:

   • Typical approval time: 1-2 business days
   • You'll get email notification when approved
   • After approval: Region becomes battle-ready! ⚔️

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIP: Request all regions at once (faster than one-by-one)
```

---

### Implementation: Setup Instructions

**File:** `training/cli/setup/core.py`

**After infrastructure check, show quota guidance:**

```python
def run_setup_core(...):
    # ... existing infrastructure checks ...

    # NEW: Show quota guidance
    status("")
    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("📊 Cloud Build C3 Quotas:")
    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("")

    # Check current quotas
    from cli.shared.quota_checker import get_cloud_build_c3_quotas
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

**Setup output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Cloud Build C3 Quotas:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current quotas:
  ✅ us-central1              → 176 vCPUs
  ✅ asia-northeast1          → 176 vCPUs

ℹ️  About Quotas:

  • Quota entries appear AFTER first launch attempt
  • Run 'python training/cli.py launch' to initialize quotas
  • Then request increases via Console

📋 To view/manage quotas:

  1. Open Quotas Console:
     https://console.cloud.google.com/apis/api/cloudbuild.googleapis.com/quotas?project=weight-and-biases-476906

  2. Filter for EXACTLY:
     'Concurrent C3 Build CPUs (Private Pool)'
     (NO '(default)' suffix!)

  3. Look for Type: 'Quota' (Adjustable: Yes)
     Ignore 'System limit' entries

  4. Recommended quota: 88 or 176 vCPUs per region
```

---

### Complete Flow Summary

**Setup (Read-Only):**
```
1. Check current quotas (if any)
2. Show quota page link
3. Explain filter string
4. Note: "Quotas appear after first launch"
```

**Launch → MECHA Acquire (Active Lazy Loading):**
```
1. Create worker pool (~5 min)
2. Submit test build → Trigger quota entry (instant)
3. Quota entry appears in Console (4 vCPUs)
4. Show: "Quota initialized - request increase in Console!"
```

**Launch → MECHA Battle (Detailed Instructions):**
```
1. Check quotas via API
2. Identify battle-ready (176 vCPUs)
3. Identify sidelined (< 176 vCPUs)
4. Show COMPLETE step-by-step instructions:
   - Direct Console link (filter pre-applied)
   - Exact filter string
   - How to identify Quota vs System Limit
   - Quota options: 176, 88, 44 vCPUs
   - Justification text to copy
   - Expected approval time
```

---

### Updated Phase Deliverables

**Phase 1 (quota_checker.py):**
- Unchanged (conservative API approach)

**Phase 2 (Replace existing checks):**
- Unchanged

**Phase 3 (MECHA quota awareness):**
- **NEW:** Add `lazy_load_quota_entry()` to mecha_acquire.py
- **NEW:** Call lazy_load after worker pool creation
- **UPDATED:** `display_sidelined_mechas()` with complete 5-step instructions
- **UPDATED:** Direct Console link with pre-applied filter
- **UPDATED:** Quota options: 176, 88, 44 vCPUs with recommendations

**Phase 4 (Infra display):**
- **UPDATED:** Show quota guidance in setup
- **UPDATED:** Link to Console with filter instructions
- **UPDATED:** Note that quotas appear after first launch

**Phase 5 (Documentation):**
- Update REQUEST_C3_QUOTA_CONSOLE.md with complete flow
- Document lazy loading in MECHA acquire
- Show complete sidelined instructions example

---

### Key Design Principles

✅ **Lazy loading happens in MECHA acquire** (after pool creation)
✅ **Battle shows complete instructions** (no guessing needed)
✅ **Setup is informational** ("quotas appear after first launch")
✅ **Direct links with pre-applied filters** (reduce user friction)
✅ **Clear distinction between Quota vs System Limit** (avoid confusion)
✅ **Multiple quota options** (176, 88, 44 with build time trade-offs)
✅ **Copy-paste justification** (streamline request process)

---

**Last Updated:** 2025-11-11 (ADDENDUM 3 ADDED - FINAL STRUCTURE)
**Status:** FINAL DESIGN COMPLETE
**Next Step:** Implement Phase 1 → quota_checker.py

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

**Why?** 
- Generic name `has_sufficient_quota()` could be mistaken for checking ANY quota type
- New name explicitly shows it checks **Cloud Build C3** quotas only
- Matches pattern: `get_cloud_build_c3_quotas()`, `get_cloud_build_c3_region_quota()`, `has_cloud_build_c3_quota()`

**Current import**:
```python
from cli.shared.quota import has_cloud_build_c3_quota
```

**Commit**: 8eeb41c - Rename all quota functions for explicit service clarity

All function references in this historical plan document remain unchanged - they represent the original implementation. Current code uses the renamed function listed above.
