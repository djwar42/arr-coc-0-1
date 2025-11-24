# How to Request C3 Cloud Build Quota - Google Cloud Console

## 🎯 Goal
Request C3 machine type quota for Cloud Build worker pools through the Google Cloud Console UI.

**DISCOVERY 2025-01-10:** Only ONE quota matters - Cloud Build "Concurrent C3 Build CPUs (Private Pool)"

---

## 🔍 Understanding System Limits vs Quotas

The Console shows two types of entries for C3 Build CPUs:

### 1️⃣ System Limits (Google's Defaults) - Filter: `Concurrent C3 Build CPUs (Private Pool) (default)`

**What they are:**
- Google's default limits for ALL regions (20+ regions)
- Default: **4 vCPUs** per region
- **Adjustable: No** (these are just informational!)
- Shows what you'd get WITHOUT a quota increase

**Example entries you'll see:**
```
Name: Concurrent C3 Build CPUs (Private Pool) (default)
Type: System limit
Dimensions: region : asia-northeast1    Value: 4    Adjustable: No
Dimensions: region : europe-west4       Value: 4    Adjustable: No
Dimensions: region : us-central1        Value: 4    Adjustable: No
... (20+ total regions)
```

**Key Point:** These are READ-ONLY. You cannot edit System Limits directly!

---

### 2️⃣ Quotas (Your Custom Overrides) - Filter: `Concurrent C3 Build CPUs (Private Pool)`

**What they are:**
- Custom quota increases Google approved for YOU
- Dimensions include: **build_origin : default** + **region : [region-name]**
- **Adjustable: Yes** (you can request changes!)
- Shows ONLY regions where Google granted you more than 4 vCPUs

**Example entries you'll see:**
```
Name: Concurrent C3 Build CPUs (Private Pool)
Type: Quota
Dimensions: build_origin : default, region : us-central1      Value: 176    Adjustable: Yes
Dimensions: build_origin : default, region : asia-northeast1  Value: 176    Adjustable: Yes
```

**Key Point:** These are the ACTUAL quotas your builds use! Only these regions have 176 vCPUs.

---

### 🎯 Which Gets Used?

When you submit a Cloud Build with `gcloud builds submit`:
- **build_origin = default** → Uses **Quota** entry (176 vCPUs) ✅
- **No build_origin** → Uses **System Limit** entry (4 vCPUs) ❌

**Your setup uses `gcloud builds submit` → automatically gets `build_origin=default` → 176 vCPUs!**

---

### 📈 Why You Only See 2 Quota Entries (But 20+ System Limits)

**Console filtering logic:**
- **System Limits:** Shows ALL 20+ regions (informational)
- **Quotas:** Shows ONLY regions where Google approved your increase

**To get 176 vCPUs in a NEW region:**
1. Find the region in **System Limit** section (shows default 4 vCPUs)
2. Request increase for that region (instructions below)
3. After Google approves → new **Quota** entry appears (with 176 vCPUs!)
4. The **System Limit** entry stays at 4 vCPUs (doesn't change - it's just the default)

---

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

---

## ✅ Prerequisites

**You must have the Quota Administrator role:**
- Role: `roles/servicemanagement.quotaAdmin`

**Check if you have it:**
```bash
gcloud projects get-iam-policy weight-and-biases-476906 \
  --flatten="bindings[].members" \
  --filter="bindings.members:YOUR_EMAIL" \
  --format="table(bindings.role)"
```

If you don't have it, ask a project owner to grant it.

---

## 📋 Step-by-Step Console Instructions

### Step 1: Open Quotas Page

1. Go to Google Cloud Console: https://console.cloud.google.com
2. Select project: **weight-and-biases-476906**
3. Navigate to: **IAM & Admin** → **Quotas & System Limits**

   **Direct Link:**
   https://console.cloud.google.com/iam-admin/quotas?project=weight-and-biases-476906

---

### Step 2: View Your Current Quotas

**Filter for YOUR custom quotas (the ones that matter!):**

```
Concurrent C3 Build CPUs (Private Pool)
```

**Important:** NO `(default)` at the end!

**You should see** (only regions Google approved):
```
Service: Cloud Build API
Quota: Concurrent C3 Build CPUs (Private Pool)
Dimensions: build_origin : default, region : us-central1      Value: 176
Dimensions: build_origin : default, region : asia-northeast1  Value: 176
```

If you only see 1-2 entries, that's NORMAL! You only have quota in those regions.

---

### Step 3: View Available Regions (System Limits)

**Filter for ALL available regions:**

```
Concurrent C3 Build CPUs (Private Pool) (default)
```

**Important:** WITH `(default)` at the end!

**You should see** (20+ regions):
```
Name: Concurrent C3 Build CPUs (Private Pool) (default)
Type: System limit
Adjustable: No

region : asia-northeast1       Value: 4
region : asia-southeast1       Value: 4
region : europe-west4          Value: 4
region : us-central1           Value: 4
... (20+ total)
```

These show ALL regions where C3 is available. Pick one to request quota for!

**Important:** You CANNOT edit System Limits! They're read-only (Adjustable: No).

To get quota in a new region, use the methods below ↓

---

### Step 4: Request Quota Increase for a NEW Region

**Option A: Console UI (Recommended)**

1. Click **"ALL QUOTAS"** tab at top
2. Click **filter dropdown** → Select **"Cloud Build API"**
3. Search: `Concurrent C3 Build CPUs`
4. Look for the entry **without** `(default)` suffix
5. Click on that row
6. Click **"EDIT QUOTAS"** button at top right
7. In the edit panel:
   - **Select location**: Choose new region (e.g., europe-west4)
   - **New limit**: 176
   - **Justification**:
     ```
     Cloud Build worker pools for PyTorch compilation.
     Requesting C3 quota for multi-region builds.
     Expected usage: 1-2 concurrent builds.
     ```
8. Click **"Submit Request"**

**Option B: gcloud Command**

```bash
gcloud alpha services quota update \
  --service=cloudbuild.googleapis.com \
  --consumer=projects/weight-and-biases-476906 \
  --metric=cloudbuild.googleapis.com/concurrent_private_pool_c3_build_cpus \
  --unit='1/{project}/{region}/{build_origin}' \
  --dimensions=region=europe-west4,build_origin=default \
  --value=176 \
  --force
2. **Click "EDIT QUOTAS"** button at top of page
3. **Enter new limit:**
   - Recommended: **88 CPUs** (good middle ground, ~25 min builds)
   - Maximum: **176 CPUs** (fastest builds, ~15-20 min)
   - Minimum: **44 CPUs** (decent, ~35 min builds)

4. **Provide justification:**
   ```
   Cloud Build worker pools for PyTorch compilation.
   Requesting C3 quota for faster builds.
   Expected usage: 1-2 concurrent builds.
   ```

5. **Click "Done"** → **"Submit Request"**

6. **Check your email** for confirmation

---

## ⏱️ What Happens After Request?

### Expected Timeline:
- ✉️ Email confirmation (immediate)
- 🔍 Google reviews request (1-2 business days)
- ✅ Approval email (quota increased automatically)
- ❌ OR denial email (with reason)

---

## 🧪 How to Test When Approved

Once approved, test immediately:

```bash
# From project root
cd RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1

# Run setup to detect new quota
python training/cli.py teardown
python training/cli.py setup

# System will automatically detect and select best C3 machine!
# Example outputs:
#   88 CPU quota  → selects c3-highcpu-88  (~25 min builds)
#   176 CPU quota → selects c3-highcpu-176 (~15-20 min builds)
```

**Expected:**
- ✅ Setup shows detected quota
- ✅ Setup selects appropriate c3-highcpu-* machine
- ✅ Launch uses new quota for faster builds

---

## 💡 Key Points

**Only ONE quota matters:**
- ✅ Cloud Build "Concurrent C3 Build CPUs (Private Pool)" ← THIS ONE!
- ❌ Compute Engine C3_CPUS ← NOT enforced (discovery 2025-01-10)

**System is fully automatic:**
- Detects your quota at setup time
- Selects best C3 machine that fits
- You can request any amount: 44, 88, or 176 CPUs

**Recommended quota:**
- **88 CPUs** - Good balance of speed vs cost
- **176 CPUs** - Maximum speed (if you need fastest builds)

---

## 📞 Useful Links

- **Quotas Page:** https://console.cloud.google.com/iam-admin/quotas?project=weight-and-biases-476906
- **Support Cases:** https://console.cloud.google.com/support/cases?project=weight-and-biases-476906
- **Worker Pools:** https://console.cloud.google.com/cloud-build/settings/worker-pool?project=weight-and-biases-476906
- **Cloud Build Settings:** https://console.cloud.google.com/cloud-build/settings?project=weight-and-biases-476906

---

## 🚨 If Request is Denied

**Fallback Option: Use N2 Machines Instead**

N2 machines are proven to work with Cloud Build:

1. Update worker pool:
   ```bash
   gcloud builds worker-pools update pytorch-spot-pool \
     --region=us-central1 \
     --worker-machine-type=n2-highcpu-32 \
     --project=weight-and-biases-476906
   ```

2. Test:
   ```bash
   python training/cli.py launch
   ```

N2 specs:
- Machine: n2-highcpu-32
- vCPUs: 32 (vs 44-176 for C3)
- Memory: 32 GB
- Performance: ~85% of C3
- **Quota: Already available!**
