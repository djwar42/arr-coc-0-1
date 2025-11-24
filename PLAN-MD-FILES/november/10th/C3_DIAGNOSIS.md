# C3 Worker Pool Diagnosis

## The Error
```
ERROR: (gcloud.builds.submit) FAILED_PRECONDITION: 
failed precondition: due to quota restrictions, 
Cloud Build cannot run builds of this machine type in this region
```

## What We Have Verified

### ✅ Quota Is Available
```bash
gcloud compute regions describe us-central1
```
Shows: `C3_CPUS: 144` (plenty of quota!)

### ✅ IAM Permissions Granted
```bash
gcloud projects get-iam-policy weight-and-biases-476906
```
Shows:
- roles/cloudbuild.builds.builder ✅
- roles/compute.admin ✅
- roles/compute.networkUser ✅

### ✅ Worker Pool Exists
```bash
gcloud builds worker-pools describe pytorch-spot-pool
```
Shows: `state: RUNNING`, `machineType: c3-highcpu-44`

### ✅ Machine Type Exists
```bash
gcloud compute machine-types describe c3-highcpu-44 --zone=us-central1-a
```
Shows: `44 vCPUs, 88 GB RAM`

## The Problem

**Despite having all the correct configuration, C3 machines fail in Cloud Build worker pools!**

## Research Findings from Bright Data

### Key Clue from Stack Overflow
> "Go to cloudbuild settings & under the WORKER POOL tab create a private pool 
> machine that matches the machine type you specified in your cloudbuild.yaml file."

This suggests the worker pool machine type must EXACTLY match what Cloud Build expects.

### Possible Root Causes

#### Theory 1: C3 Not Yet Supported in Worker Pools
- C3 is Google's newest machine series (2024)
- Worker pools may not support C3 yet
- Similar to how new GPU types take time to be added to Cloud Build

#### Theory 2: Allowlist Required for C3
- From Google Groups: "Due to abuse... we changed the criterion for Private Pool creation"
- C3 machines might require special approval/allowlist
- Contact Google Cloud Support to be added to C3 allowlist

#### Theory 3: Region-Specific Restriction
- Some regions may not support C3 in worker pools yet
- us-central1 might not have C3 worker pool support

#### Theory 4: Separate Cloud Build Quota
- Compute Engine quota (C3_CPUS) ≠ Cloud Build quota
- Cloud Build might have its own C3 quota (currently 0)
- Need to request "Cloud Build C3 machine quota" separately

## Next Steps to Solve

### Option A: Contact Google Cloud Support (Recommended)
```bash
# Open support case specifically asking:
"Please enable C3 machine types for Cloud Build worker pools in project 
weight-and-biases-476906, region us-central1. 

We have:
- C3_CPUS quota: 144
- Worker pool created: pytorch-spot-pool
- Machine type requested: c3-highcpu-44
- Error: FAILED_PRECONDITION quota restrictions

Request allowlist for C3 in Cloud Build worker pools."
```

### Option B: Try Different Machine Type (Workaround)
Switch to proven working machine types:
- **N2**: n2-highcpu-32 (32 vCPUs) - widely supported
- **E2**: e2-highcpu-16 (16 vCPUs) - we know this works
- **C2**: c2-standard-30 (30 vCPUs) - if available

### Option C: Use Default Cloud Build (Current)
Already working! Just slow:
- 8 vCPUs (default)
- 6 hours for first build
- No C3 optimization

## Recommendation

**Contact Google Cloud Support with specific request for C3 allowlist.**

C3 is bleeding edge (released 2024), so Cloud Build support likely requires:
1. Special enablement
2. Allowlist addition
3. Quota increase for Cloud Build (separate from Compute)

While waiting, use default Cloud Build workers (already configured).

## Files Modified for C3 Support
- `.cloudbuild-pytorch-clean.yaml` - Worker pool config
- `training/cli/setup/core.py` - IAM permissions
- `training/cli/shared/setup_helper.py` - Teardown IAM
- `training/cli/teardown/core.py` - Resource list

All changes committed: `git log --oneline | head -5`
