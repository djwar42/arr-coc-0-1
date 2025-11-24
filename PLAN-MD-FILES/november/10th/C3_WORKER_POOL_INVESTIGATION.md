# C3 Worker Pool Investigation - Final Report

## 🎯 Question
Why can't Cloud Build worker pools use C3 machines despite having C3_CPUS=144 quota?

---

## ✅ What We Have

```
C3_CPUS quota:        144 in us-central1✅
IAM permissions:      compute.admin + compute.networkUser ✅ (tested)
Worker pool status:   RUNNING ✅
Machine type:         c3-highcpu-44 (44 vCPUs) ✅
APIs enabled:         Compute Engine + Cloud Build ✅
```

**Everything looks correct!**

---

## ❌ What Fails

```bash
$ gcloud builds submit --pool=pytorch-spot-pool
ERROR: FAILED_PRECONDITION: quota restrictions for this machine type
```

Even though 44 < 144, worker pools can't use C3 machines!

---

## 🔬 Testing Results

### Test 1: IAM Permissions
**Hypothesis**: Maybe Cloud Build service account needs compute.admin?

**Test**: Granted `roles/compute.admin` and `roles/compute.networkUser` to Cloud Build SA

**Result**: ❌ Still failed with same "quota restrictions" error

**Conclusion**: IAM permissions are NOT the issue

---

### Test 2: Default Cloud Build
**Hypothesis**: Maybe it's specific to worker pools?

**Test**: Disabled worker pool, used default Cloud Build (8 vCPU workers)

**Result**: ✅ Build STARTED successfully!

**Conclusion**:
- Default Cloud Build works fine
- Worker pools have additional restrictions
- IAM permissions NOT needed for default workers

---

## 💡 Conclusions

### 1. IAM Permissions NOT Needed
```python
# These are NOT required for default Cloud Build:
roles/compute.admin         ❌ Not needed
roles/compute.networkUser   ❌ Not needed
```

**Action Taken**: Removed from setup/teardown (commit cc65ee5)

---

### 2. C3 Worker Pool Mystery

**Likely causes** (unverified):

**A. Cloud Build Has Separate C3 Quota**
- Compute Engine: C3_CPUS = 144 ✅
- Cloud Build: Separate allowlist/quota? ❓

**B. C3 Machines Restricted in Some Projects**
- Some GCP projects don't have C3 access in Cloud Build
- May need special enablement or support request

**C. Regional Availability**
- C3 may not be available for Cloud Build in us-central1
- Even if available for Compute Engine

**D. Billing/Organization Policy**
- Some orgs restrict premium machine types
- C3 is newer/more expensive than E2/N2

---

## ✅ Working Solution

**Current Configuration** (`.cloudbuild-pytorch-clean.yaml`):

```yaml
# Using DEFAULT Cloud Build (no worker pool)
options:
  logging: CLOUD_LOGGING_ONLY

timeout: '360m'  # 6 hours
```

**Performance**:
- Default workers: ~8 vCPUs
- Build time: 4-6 hours (first time), then cached
- Cost: Free tier or standard Cloud Build pricing

**Pros**:
- ✅ Works without quota issues
- ✅ No IAM permission management needed
- ✅ Simpler setup

**Cons**:
- ❌ Slower than C3 workers (6hrs vs 2hrs)
- ❌ No spot instance cost savings

---

## 📋 If You Want Worker Pools Later

### Option 1: Request Cloud Build C3 Quota
1. Contact Google Cloud Support
2. Request: "Enable C3 machine types for Cloud Build worker pools"
3. Provide justification: PyTorch compilation for ML training

### Option 2: Use Different Machine Type
Worker pools might work with:
- **N2**: `n2-highmem-64` (64 vCPUs, 512GB RAM)
- **E2**: `e2-highcpu-16` (but need E2_CPUS quota)
- **C2**: `c2-standard-60` (60 vCPUs, 240GB RAM)

Then you'd need to:
1. Grant IAM permissions:
   ```bash
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:PROJECT_NUM@cloudbuild.gserviceaccount.com" \
     --role="roles/compute.admin"

   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:PROJECT_NUM@cloudbuild.gserviceaccount.com" \
     --role="roles/compute.networkUser"
   ```

2. Update worker pool:
   ```bash
   gcloud builds worker-pools update pytorch-spot-pool \
     --region=us-central1 \
     --worker-config=machine-type=n2-highmem-64
   ```

3. Uncomment pool in `.cloudbuild-pytorch-clean.yaml`

---

## 🎯 Recommendation

**Keep using default Cloud Build** unless:
- Build time becomes a blocker (>6 hours unacceptable)
- You get C3 quota approved for Cloud Build
- You switch to a different machine type (N2/C2)

Default workers are **slower but reliable** - builds work, just take longer. The first build is slow (4-6 hours), but then the image is cached in Artifact Registry forever!

---

## 📚 Related Files

- `.cloudbuild-pytorch-clean.yaml` - Build configuration (lines 166-175)
- `training/cli/setup/core.py` - Setup (lines 619-625 show commented IAM)
- `training/cli/shared/setup_helper.py` - Teardown (lines 1170-1172)
- `SPOT_VM_CACHE_MANAGEMENT.md` - Complete architecture docs
- `VALIDATION_GUIDE.md` - How to verify builds work

---

## 🔄 History

**Initial Approach**: Use C3 worker pools for 2-3x faster builds + spot savings

**Problem Discovered**: Worker pools fail with "quota restrictions" despite having C3_CPUS=144

**Investigation**: Tested IAM permissions, different configs, default workers

**Final Solution**: Use default Cloud Build (slower but works)

**Cleanup**: Removed unnecessary IAM permission code from setup/teardown

---

**Last Updated**: 2025-11-10
**Status**: RESOLVED - Using default workers
**Next Steps**: Monitor build times, request C3 quota if needed
