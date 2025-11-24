# Pricing Teardown Error Analysis

**Date**: 2025-11-21
**Status**: Errors are **benign** - pricing infrastructure was never created
**Files Analyzed**: `CLI/teardown/pricing_teardown.py`, `CLI/shared/pricing/pricing_config.py`

---

## 📊 Executive Summary

The pricing teardown errors you're seeing are **NOT FATAL** - they indicate the pricing infrastructure was never created, so teardown is trying to delete things that don't exist. The code handles this gracefully but shows warnings that look scary.

**Errors Observed:**
1. `PERMISSION_DENIED: Cloud Scheduler API has not been used`
2. `PERMISSION_DENIED: Cloud Functions API has not been used`
3. `NOT_FOUND: Unknown service account`

**Root Cause:** Pricing infrastructure setup was **skipped or failed** during initial setup, so these resources don't exist to delete.

**Impact:** ✅ **NONE** - Teardown continues successfully, these are just warnings

---

## 🏗️ Architecture Overview

### What is Pricing Infrastructure?

The pricing infrastructure automatically updates GCP pricing data every 20 minutes:

```
┌─────────────────────────────────────────────────────────────┐
│  PRICING INFRASTRUCTURE (Auto-Update System)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌──────────────────────┐         │
│  │ Cloud Scheduler │ ───> │ Cloud Function Gen2  │         │
│  │                 │ /20m │                      │         │
│  │ "arr-coc-       │      │ "arr-coc-pricing-    │         │
│  │  pricing-       │      │  runner"             │         │
│  │  scheduler"     │      │                      │         │
│  └─────────────────┘      └──────────────────────┘         │
│                                      │                      │
│                                      ↓                      │
│                          ┌────────────────────────┐         │
│                          │ Artifact Registry      │         │
│                          │                        │         │
│                          │ Repository:            │         │
│                          │  arr-coc-pricing       │         │
│                          │                        │         │
│                          │ Package:               │         │
│                          │  gcp-pricing (JSON)    │         │
│                          └────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Purpose: CLI checks pricing before launch (faster than querying GCP API)
```

### Components

**1. Cloud Scheduler (`arr-coc-pricing-scheduler`)**
- Cron job that runs every 20 minutes
- Triggers the Cloud Function via HTTP
- Location: `us-central1`

**2. Cloud Function (`arr-coc-pricing-runner`, Gen2)**
- Queries GCP Billing API for current pricing
- Stores results in Artifact Registry as JSON
- Updates package: `gcp-pricing` in repo `arr-coc-pricing`

**3. Artifact Registry Repository (`arr-coc-pricing`)**
- **PRESERVED during teardown** (contains historical pricing data)
- Stores pricing JSON as versioned artifacts
- Used by CLI to check prices before launch

**4. Service Account (`{PROJECT_ID}@appspot.gserviceaccount.com`)**
- Default App Engine service account
- Granted `roles/iam.serviceAccountUser` with OIDC condition
- Allows Cloud Scheduler to invoke Cloud Function

---

## 🔍 Code Flow Analysis

### File: `CLI/teardown/pricing_teardown.py`

**Entry Point:** `teardown_pricing_infrastructure(status_callback)`

**Line 95-121:** Main teardown orchestrator
```python
def teardown_pricing_infrastructure(status_callback):
    status = status_callback

    # 1. Delete Cloud Scheduler (lines 103-104)
    status("   ⊗  Deleting Cloud Scheduler...")
    delete_scheduler(status)

    # 2. Delete Cloud Function (lines 106-108)
    status("   ⊗  Deleting Cloud Function...")
    delete_cloud_function(status)

    # 3. Revoke OIDC permissions (lines 110-112)
    status("   ⊗  Revoking OIDC permissions...")
    revoke_actAs_permission(status)

    # 4. Disable Cloud Billing API (lines 114-116)
    status("   ⊗  Disabling Cloud Billing API...")
    disable_cloudbilling_api(status)

    # 5. Preserve Artifact Registry (line 119)
    status("   [cyan]ℹ Pricing repository preserved (historical data intact)[/cyan]")

    status("   [cyan]⚡Pricing teardown complete - Roger![/cyan]")
```

**Key Pattern:** Each deletion function handles errors gracefully - warnings shown, but teardown continues.

---

### Function 1: `delete_scheduler()` (Lines 124-162)

**Purpose:** Delete Cloud Scheduler job

**Implementation:**
```python
def delete_scheduler(status):
    def try_delete():
        result = subprocess.run([
            "gcloud", "scheduler", "jobs", "delete", SCHEDULER_JOB,
            f"--location={REGION}",
            f"--project={PROJECT_ID}",
            "--quiet",
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return (True, None)

        # Ignore "not found" errors (lines 143-145)
        stderr_lower = result.stderr.lower()
        if "not found" in stderr_lower or "not_found" in stderr_lower:
            return (True, None)  # Idempotent success!

        # Real error - return for retry
        error_msg = f"Exit code: {result.returncode}, STDERR: {result.stderr[:100]}"
        return (False, error_msg)

    # Retry with backoff: 0s, 1s, 4s, 8s delays (4 attempts)
    success, error_msg = retry_with_backoff(
        try_delete,
        max_attempts=4,
        operation_name="scheduler deletion"
    )

    if not success:
        status(f"    ⚠️  Scheduler deletion failed after 4 attempts")
        if error_msg:
            status(f"    {error_msg}")  # ⬅️ YOUR ERROR SHOWS HERE
```

**Error Handling Logic:**
- ✅ `returncode == 0` → Success
- ✅ `"not found"` in stderr → Success (already deleted)
- ❌ **Any other error** → Retry 4 times, then warn

**Your Error:**
```
PERMISSION_DENIED: Cloud Scheduler API has not been used in project
```

**Why This Happens:**
- GCP APIs are **lazy-loaded** (only enabled when first used)
- If you never created the scheduler, the API was never enabled
- Deletion command triggers API check → sees API never used → returns `PERMISSION_DENIED`
- Code checks for `"not found"` but error says `"has not been used"` → not caught!

**Is This Fatal?** ❌ **NO** - Code shows warning and continues

---

### Function 2: `delete_cloud_function()` (Lines 164-203)

**Purpose:** Delete Cloud Function (Gen2)

**Implementation:** Nearly identical to `delete_scheduler()`, but for Cloud Functions API

**Your Error:**
```
PERMISSION_DENIED: Cloud Functions API has not been used in project
```

**Same Issue:** API never enabled because function never created.

**Is This Fatal?** ❌ **NO** - Code shows warning and continues

---

### Function 3: `revoke_actAs_permission()` (Lines 205-243)

**Purpose:** Revoke Service Account User role from current user

**Implementation:**
```python
def revoke_actAs_permission(status):
    # Get current user email (lines 208-216)
    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        status("    ⚠️  Could not get current user email")
        return

    user_email = result.stdout.strip()
    service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"  # ⬅️ Default App Engine SA

    # Remove Service Account User role with condition (lines 222-234)
    result = subprocess.run([
        "gcloud", "iam", "service-accounts", "remove-iam-policy-binding",
        service_account,  # ⬅️ EXPECTS THIS TO EXIST!
        f"--member=user:{user_email}",
        "--role=roles/iam.serviceAccountUser",
        '--condition=expression=resource.service=="cloudscheduler.googleapis.com",title=OIDCSchedulerOnly',
        f"--project={PROJECT_ID}",
        "--quiet",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        # Ignore "not found" or "no binding" errors (lines 238-242)
        stderr_lower = result.stderr.lower()
        if "not found" not in stderr_lower and "no role binding" not in stderr_lower:
            status(f"    ⚠️  OIDC permission revocation failed (exit code: {result.returncode})")
            status(f"    STDERR: {result.stderr if result.stderr else '(empty)'}")
```

**Your Error:**
```
NOT_FOUND: Unknown service account
```

**Why This Happens:**
- Code assumes **default App Engine service account** exists: `{PROJECT_ID}@appspot.gserviceaccount.com`
- This SA is only created when you **enable App Engine** for the first time
- If you never used App Engine (or pricing scheduler wasn't set up), this SA doesn't exist
- Trying to revoke permissions on non-existent SA → `NOT_FOUND`

**Is This Fatal?** ❌ **NO** - Code shows warning and continues

---

### Function 4: `disable_cloudbilling_api()` (Lines 245-261)

**Purpose:** Disable Cloud Billing API

**Implementation:**
```python
def disable_cloudbilling_api(status):
    result = subprocess.run([
        "gcloud", "services", "disable", "cloudbilling.googleapis.com",
        f"--project={PROJECT_ID}",
        "--quiet",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        # API disable can fail if still in use - that's ok (lines 258-259)
        status(f"    ⚠️  Cloud Billing API disable warning (may still be in use)")
    # Success case: no output (silent success)
```

**Error Handling:** Graceful - warns but continues

---

## 🐛 The Core Issue

### Why These Errors Happen

**Expected Setup Flow:**
1. Setup creates pricing infrastructure (scheduler + function)
2. This enables Cloud Scheduler API and Cloud Functions API
3. Teardown deletes resources using these APIs

**What Actually Happened:**
1. Setup **skipped or failed** to create pricing infrastructure
2. APIs were **never enabled** (lazy-loaded)
3. Teardown tries to delete resources
4. GCP says "API hasn't been used" instead of "resource not found"

### Code Assumption vs Reality

**Code Assumes:**
```python
# Line 143-145: Handles "not found" errors
if "not found" in stderr_lower or "not_found" in stderr_lower:
    return (True, None)  # Idempotent success
```

**But Gets:**
```
PERMISSION_DENIED: Cloud Scheduler API has not been used in project
```

**The Bug:** Error string `"has not been used"` doesn't contain `"not found"`, so retry logic continues instead of treating as "already deleted".

---

## 🔧 Why Pricing Setup Might Have Failed

### Possible Causes

**1. Billing Not Enabled (Most Likely)**
- Cloud Functions Gen2 requires billing
- Cloud Scheduler requires billing
- If billing wasn't linked when setup ran → setup silently failed

**2. API Enable Timeout**
- Setup enables APIs but doesn't wait for propagation
- If APIs weren't fully ready, resource creation failed

**3. Quota Limits**
- Cloud Functions quota exhausted
- Cloud Scheduler quota exhausted

**4. Permission Issues**
- Current user lacks `cloudfunctions.functions.create` permission
- Current user lacks `cloudscheduler.jobs.create` permission

**5. Setup Skipped Pricing Step**
- Check setup logs - was "Building pricing cloud function... (6/8)" shown?
- If not, setup might have early-exited before reaching pricing

---

## ✅ Current Behavior Assessment

### Is This Breaking Anything?

**Short Answer:** ❌ **NO**

**Why It's OK:**

1. **Teardown Continues:** All pricing teardown failures are warnings, not fatal errors
2. **No Resources Leaked:** If resources never existed, there's nothing to clean up
3. **No Cost Impact:** No running scheduler or function = no ongoing costs
4. **Pricing Still Works:** CLI can query GCP Billing API directly (slower but works)

### What You're Missing

**Without Pricing Infrastructure:**
- ✅ Training still works (launch queries GCP API directly)
- ❌ Launch is **slower** (~2-3 seconds to fetch pricing vs instant cache lookup)
- ❌ No automatic pricing updates (must query API every launch)

**Trade-off:** Slightly slower launch vs complexity of maintaining pricing infrastructure

---

## 🛠️ Recommended Fixes

### Option 1: Improve Error Detection (Quick Fix)

**File:** `CLI/teardown/pricing_teardown.py`

**Change:** Add detection for "API has not been used" error

**Lines 143-145 (scheduler) and 184-186 (function):**

```python
# OLD:
if "not found" in stderr_lower or "not_found" in stderr_lower:
    return (True, None)

# NEW:
if ("not found" in stderr_lower
    or "not_found" in stderr_lower
    or "has not been used" in stderr_lower):
    return (True, None)  # API never enabled = resource never created = idempotent success
```

**Impact:** Turns scary warnings into silent success

---

### Option 2: Check If APIs Are Enabled First (Better Fix)

**Add pre-check before deletion:**

```python
def is_api_enabled(api_name):
    """Check if API is enabled (returns False if never used)"""
    result = subprocess.run([
        "gcloud", "services", "list",
        f"--filter=config.name:{api_name}",
        "--format=value(state)",
        f"--project={PROJECT_ID}"
    ], capture_output=True, text=True)

    return result.returncode == 0 and "ENABLED" in result.stdout

def delete_scheduler(status):
    # Skip if API never enabled (resource can't exist)
    if not is_api_enabled("cloudscheduler.googleapis.com"):
        status("    ℹ  Scheduler API not enabled (nothing to delete)")
        return

    # ... rest of deletion logic
```

**Impact:** Skips deletion attempts when APIs aren't even enabled

---

### Option 3: Make Pricing Setup More Robust (Best Fix)

**Ensure pricing setup succeeds or fails loudly:**

1. Check billing is enabled before creating pricing infrastructure
2. Wait for API propagation after enabling
3. Verify resources were created successfully
4. Show clear error if pricing setup fails (don't silently skip)

**Add to `CLI/setup/pricing_setup.py`:**

```python
def verify_pricing_created(status):
    """Verify pricing infrastructure was created successfully"""

    # 1. Check scheduler exists
    scheduler_result = subprocess.run([
        "gcloud", "scheduler", "jobs", "describe", SCHEDULER_JOB,
        f"--location={REGION}",
        f"--project={PROJECT_ID}"
    ], capture_output=True)

    if scheduler_result.returncode != 0:
        status("[red]✗ Pricing scheduler not found - setup incomplete![/red]")
        return False

    # 2. Check function exists
    function_result = subprocess.run([
        "gcloud", "functions", "describe", FUNCTION_NAME,
        "--gen2",
        f"--region={REGION}",
        f"--project={PROJECT_ID}"
    ], capture_output=True)

    if function_result.returncode != 0:
        status("[red]✗ Pricing function not found - setup incomplete![/red]")
        return False

    status("[green]✓ Pricing infrastructure verified[/green]")
    return True
```

---

## 📝 Summary & Action Items

### Current State

✅ **Teardown works** - Warnings are benign
❌ **Pricing infrastructure missing** - Never created during setup
⚠️ **Error messages confusing** - Look fatal but aren't

### Immediate Actions

**For You (User):**
1. ✅ **Ignore the warnings** - They indicate resources that never existed
2. ✅ **Verify billing is enabled** - Required for pricing infrastructure
3. ⚠️ **Run setup again** - Try creating pricing infrastructure now that billing is ready

**For Code (Developer):**
1. 🔧 **Add "has not been used" to error detection** (5-minute fix)
2. 🔧 **Check API status before deletion** (15-minute fix)
3. 🔧 **Make pricing setup verify resources created** (30-minute fix)

### Long-Term

**Decision Point:** Do we even need pricing infrastructure?

**Pros of keeping it:**
- Faster launches (instant pricing lookup vs 2-3s API query)
- Historical pricing data preserved
- Reduces GCP API calls (cost savings at scale)

**Cons:**
- Additional complexity (scheduler, function, service account)
- Can fail during setup (billing, quotas, permissions)
- Teardown shows scary warnings if never created

**Recommendation:** Keep pricing infrastructure, but make setup/teardown more robust so errors are clear and non-scary.

---

## 🔬 Technical Debt Items

### Identified Issues

**1. Silent Setup Failures**
- Pricing setup might fail without clear error message
- User doesn't know pricing is missing until launch is slow

**2. Incomplete Error Detection**
- `"has not been used"` vs `"not found"` - both mean "doesn't exist"
- Code only handles one case

**3. Missing Verification**
- Setup doesn't verify resources were created
- Teardown assumes they exist and fails loudly

**4. Default App Engine SA Assumption**
- Code assumes `{PROJECT_ID}@appspot.gserviceaccount.com` exists
- Only true if App Engine enabled (may not be)

**5. No Billing Check**
- Pricing setup requires billing but doesn't check first
- Results in cryptic failures

---

**END OF REPORT**

**Next Steps:**
1. User can safely ignore these warnings (teardown succeeded)
2. Developer can implement Option 1 fix (5-minute change to improve UX)
3. Consider Option 3 for production-ready pricing infrastructure
