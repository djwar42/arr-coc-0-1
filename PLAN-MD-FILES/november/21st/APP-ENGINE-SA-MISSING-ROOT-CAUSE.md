# Root Cause: App Engine Service Account Missing on Fresh Accounts

**Date**: 2025-11-21
**Status**: 🔴 **BLOCKING FRESH ACCOUNT SETUP**
**Impact**: Setup fails on new projects, works on accounts where setup ran multiple times

---

## 💥 The Core Problem

Both **setup** and **teardown** assume the **App Engine default service account** exists:

```
{PROJECT_ID}@appspot.gserviceaccount.com
```

**On Fresh Accounts:** This SA **doesn't exist** → Setup/teardown crash
**On Mature Accounts:** This SA **exists** → Everything works

---

## 🔍 Evidence

### File: `CLI/setup/pricing_setup.py`

**Line 625:**
```python
service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"  # ⬅️ ASSUMES THIS EXISTS!
```

**Line 630-642:** Tries to get IAM policy of this SA
```python
check_result = subprocess.run([
    "gcloud", "iam", "service-accounts", "get-iam-policy",
    service_account,  # ⬅️ FAILS IF SA DOESN'T EXIST!
    f"--project={PROJECT_ID}",
    "--format=json",
], capture_output=True, text=True)
```

**What Happens on Fresh Account:**
```
ERROR: (gcloud.iam.service-accounts.get-iam-policy) NOT_FOUND: Unknown service account
```
→ Setup crashes at step 6/8 (pricing infrastructure)

---

### File: `CLI/teardown/pricing_teardown.py`

**Line 219:**
```python
service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"  # ⬅️ ASSUMES THIS EXISTS!
```

**Line 222-234:** Tries to remove IAM binding from this SA
```python
result = subprocess.run([
    "gcloud", "iam", "service-accounts", "remove-iam-policy-binding",
    service_account,  # ⬅️ FAILS IF SA DOESN'T EXIST!
    f"--member=user:{user_email}",
    "--role=roles/iam.serviceAccountUser",
    ...
], capture_output=True, text=True)
```

**What Happens on Fresh Account:**
```
ERROR: (gcloud.iam.service-accounts.remove-iam-policy-binding) NOT_FOUND: Unknown service account
```
→ Teardown shows warning but continues

---

## 🤔 Why Does It Work on Mature Accounts?

**App Engine SA Creation Triggers:**

1. **First Cloud Function Deployment**
   - Cloud Functions (Gen1) used to automatically create App Engine SA
   - If you deployed any Gen1 function, SA was created as side effect

2. **Manual App Engine Enable**
   - Running `gcloud app create` creates the SA
   - Or visiting App Engine in console and clicking "Create App"

3. **Some Other GCP Service**
   - Several GCP services create this SA automatically
   - Hard to predict which ones

**On Accounts With Multiple Setups:**
- One of these triggers happened at some point
- SA now exists
- Future setups work fine

**On Fresh Account:**
- Nothing triggered SA creation yet
- SA doesn't exist
- Setup crashes

---

## 📊 What Gets Created vs What's Missing

### What Setup DOES Create

✅ **Artifact Registry:** `arr-coc-pricing`
✅ **Cloud Function:** `arr-coc-pricing-runner` (Gen2)
✅ **Cloud Scheduler:** `arr-coc-pricing-scheduler`
✅ **Custom SA:** `arr-coc-sa@{PROJECT_ID}.iam.gserviceaccount.com` (step 4/8)

### What Setup ASSUMES Exists

❌ **App Engine SA:** `{PROJECT_ID}@appspot.gserviceaccount.com` ⬅️ **THIS IS THE BUG!**

---

## 🔧 The Fix

### Option 1: Check If SA Exists, Create If Missing (Recommended)

**File:** `CLI/setup/pricing_setup.py`

**Add before line 625:**

```python
def ensure_app_engine_sa_exists(status):
    """
    Ensure App Engine default SA exists (creates if missing).

    This SA is used for Cloud Scheduler OIDC authentication.
    """
    service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"

    # Check if SA exists
    result = subprocess.run([
        "gcloud", "iam", "service-accounts", "describe",
        service_account,
        f"--project={PROJECT_ID}"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        return service_account  # Already exists

    # SA doesn't exist - enable App Engine to create it
    status("   ⊕  Enabling App Engine (creates default SA)...")

    # Method 1: Try to create App Engine app (this creates the SA)
    result = subprocess.run([
        "gcloud", "app", "create",
        f"--region={REGION}",  # App Engine region (not same as resource region!)
        f"--project={PROJECT_ID}"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        # Check if error is "app already exists" (idempotent)
        if "already contains an App Engine" not in result.stderr:
            # Real error
            raise RuntimeError(f"Failed to create App Engine app: {result.stderr}")

    # Wait for SA propagation (GCP eventual consistency)
    time.sleep(5)

    status("   ✓    App Engine SA created")
    return service_account
```

**Then modify `grant_actAs_permission()`:**

```python
def grant_actAs_permission(status):
    # Get current user
    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        status("   Could not get current user email")
        raise RuntimeError("Failed to get gcloud account")

    user_email = result.stdout.strip()

    # Ensure App Engine SA exists (creates if missing)
    service_account = ensure_app_engine_sa_exists(status)  # ⬅️ NEW!

    # ... rest of function unchanged
```

---

### Option 2: Use arr-coc-sa Instead (Alternative)

**Instead of using App Engine SA, use the custom SA we already create:**

```python
# OLD:
service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"

# NEW:
service_account = f"arr-coc-sa@{PROJECT_ID}.iam.gserviceaccount.com"
```

**Pros:**
- Uses SA we control
- No App Engine dependency
- Already created in step 4/8

**Cons:**
- Needs to verify arr-coc-sa is created BEFORE pricing setup
- May require reordering setup steps

---

### Option 3: Skip OIDC Grant If SA Missing (Quick Fix)

**Add error handling to skip gracefully:**

```python
def grant_actAs_permission(status):
    # ... get user_email ...

    service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"

    # Check if SA exists first
    check_sa = subprocess.run([
        "gcloud", "iam", "service-accounts", "describe",
        service_account,
        f"--project={PROJECT_ID}"
    ], capture_output=True, text=True)

    if check_sa.returncode != 0:
        status("   ⚠️  App Engine SA not found - skipping OIDC grant")
        status("   ℹ Cloud Scheduler will use default auth instead")
        return  # Skip this step (non-fatal)

    # SA exists - proceed with grant
    # ... rest of function ...
```

**Pros:** Minimal code change, doesn't break fresh accounts
**Cons:** Cloud Scheduler may use less secure auth method

---

## 🎯 Recommended Solution

**Implement Option 1: Ensure SA Exists**

### Why?

1. **Proper Fix:** Creates the SA if missing (no assumptions)
2. **Secure:** Uses proper OIDC authentication
3. **Idempotent:** Safe to run multiple times
4. **Works Everywhere:** Fresh accounts + mature accounts

### Implementation Steps

1. Add `ensure_app_engine_sa_exists()` function to `pricing_setup.py`
2. Call it from `grant_actAs_permission()` before using SA
3. Add same check to `pricing_teardown.py` for `revoke_actAs_permission()`
4. Test on fresh project to verify

---

## 📝 Same Fix Needed in Teardown

**File:** `CLI/teardown/pricing_teardown.py`

**Line 205-243:** `revoke_actAs_permission()` has same issue

**Fix:** Add SA existence check:

```python
def revoke_actAs_permission(status):
    # Get current user email
    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        status("    ⚠️  Could not get current user email")
        return

    user_email = result.stdout.strip()
    service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"

    # Check if SA exists first (NEW!)
    check_sa = subprocess.run([
        "gcloud", "iam", "service-accounts", "describe",
        service_account,
        f"--project={PROJECT_ID}"
    ], capture_output=True, text=True)

    if check_sa.returncode != 0:
        status("    ℹ  App Engine SA not found - nothing to revoke")
        return  # SA doesn't exist = no permissions to revoke (idempotent success)

    # SA exists - proceed with revoke
    # ... rest of function ...
```

---

## ✅ Testing Plan

### Test 1: Fresh Account (Primary Test Case)

```bash
# 1. Create brand new GCP project
gcloud projects create test-fresh-20251121 --name="Fresh Test"

# 2. Configure
gcloud config set project test-fresh-20251121
# Enable billing, edit .training file, etc.

# 3. Run setup
python CLI/cli.py setup

# Expected: Setup succeeds, creates App Engine SA automatically
```

### Test 2: Mature Account (Regression Test)

```bash
# Use existing project where setup works
python CLI/cli.py teardown
python CLI/cli.py setup

# Expected: Still works (idempotent SA creation)
```

### Test 3: Teardown on Fresh Account

```bash
# After Test 1 setup succeeds
python CLI/cli.py teardown

# Expected: Teardown succeeds, revokes OIDC permissions
```

---

## 🚨 Priority Level: HIGH

**Why This is Critical:**

1. **Blocks New Users:** Fresh accounts can't complete setup
2. **Confusing Errors:** "Unknown service account" doesn't explain the issue
3. **Hidden Dependency:** App Engine not mentioned in docs
4. **Inconsistent Behavior:** Works on some accounts, not others

**User Experience:**
- ❌ New user: Setup fails, unclear why
- ✅ Returning user: Setup works (SA already exists)
- 😕 Documentation: Doesn't mention App Engine requirement

---

## 📚 Documentation Updates Needed

**File:** `SETUP.md`

**Add after Step 1 (Create GCP Project):**

```markdown
║  ├─ 📋 IMPORTANT: App Engine Service Account ────────────────────────
║  │  The pricing infrastructure uses App Engine's default SA:
║  │  {PROJECT_ID}@appspot.gserviceaccount.com
║  │
║  │  Setup will automatically create this SA by enabling App Engine.
║  │  No manual action needed - just be aware this happens!
║  │
║  │  Why: Cloud Scheduler needs this SA for OIDC authentication.
```

---

## 🎬 Next Steps

1. **Implement Option 1** in `pricing_setup.py` (15 minutes)
2. **Add SA check** to `pricing_teardown.py` (5 minutes)
3. **Test on fresh project** (10 minutes)
4. **Update SETUP.md** with App Engine note (5 minutes)
5. **Git commit** with clear message explaining fix

**Total Time:** ~35 minutes to completely fix this issue

---

**END OF REPORT**

**Summary:** App Engine SA is assumed to exist but isn't created on fresh accounts. Fix: Check if it exists, create if missing (via `gcloud app create`). This will make setup work on ALL accounts, not just mature ones.
