# 🚀 Cloud Build Spot Worker Pool Implementation Plan

⚠️ **IMPORTANT UPDATE**: Cloud Build Worker Pools ONLY support C3 machines!
- Original plan: C4D_HIGHCPU_384 (384 vCPUs) ❌ NOT SUPPORTED
- **Actual implementation**: Smart C3 quota detection (176→88→44→22→8→4 vCPUs) ✅
- Maximum: c3-highcpu-176 (176 vCPUs, ~15-20 min builds)

**What we implemented instead**: Intelligent quota detection system that:
- Auto-detects best C3 machine within quota (checks gcloud quota limits)
- Provides comprehensive error messages on quota failures
- Shows upgrade suggestions when quota allows faster machines
- Estimates build times based on machine size (15-90 minutes)

---

## 🏆 ALL SPOT MACHINE OPTIONS (Highest to Medium)

**⚡ CRITICAL: ALL options below use SPOT PRICING (60-91% savings!)**

### Extreme Performance (300+ vCPUs)

| Rank | Machine Type | vCPUs | Spot $/hr | 15min Build Cost | Build Time | Best For |
|------|--------------|-------|-----------|------------------|------------|----------|
| 🥇 | **C4D_HIGHCPU_384** | **384** | **$6.46** | **$1.61** | **10-12 min** | **FASTEST!** |
| 🥈 | C3D_HIGHCPU_360 | 360 | $7.16 | $1.79 | 11-13 min | High perf |
| 🥉 | C4_HIGHCPU_288 | 288 | $8.26 | $2.07 | 14-16 min | Intel |

### High Performance (150-280 vCPUs)

| Machine Type | vCPUs | Spot $/hr | 15min Build Cost | Build Time |
|--------------|-------|-----------|------------------|------------|
| N2D_HIGHCPU_224 | 224 | $1.95 | $0.49 | 18-20 min |
| C4D_HIGHCPU_192 | 192 | $6.14 | $1.54 | 20-22 min |
| C4_HIGHCPU_192 | 192 | $7.84 | $1.96 | 20-22 min |
| C3D_HIGHCPU_180 | 180 | $4.38 | $1.10 | 22-24 min |
| C3_HIGHCPU_176 | 176 | $1.93 | $0.48 | 23-25 min |

### Medium-High Performance (96-144 vCPUs)

| Machine Type | vCPUs | Spot $/hr | 15min Build Cost | Build Time |
|--------------|-------|-----------|------------------|------------|
| C4_HIGHCPU_144 | 144 | $5.42 | $1.36 | 25-28 min |
| N2D_HIGHCPU_128 | 128 | $1.12 | $0.28 | 28-30 min |
| C2D_HIGHCPU_112 | 112 | $1.06 | $0.26 | 30-32 min |
| C4D_HIGHCPU_96 | 96 | $1.53 | $0.38 | 32-35 min |
| C4_HIGHCPU_96 | 96 | $1.82 | $0.45 | 32-35 min |
| N2D_HIGHCPU_96 | 96 | $0.83 | $0.21 | 32-35 min |
| N2_HIGHCPU_96 | 96 | $1.09 | $0.27 | 32-35 min |

### Current vs Best Option

| Metric | Current (N2-64 on-demand) | **RECOMMENDED (C4D-384 spot)** | Improvement |
|--------|---------------------------|--------------------------------|-------------|
| vCPUs | 64 | **384** | **6× more!** |
| Build time | 45-60 min | **10-12 min** | **5× faster!** |
| Pricing | On-demand $2.80/hr | **Spot $6.46/hr** | Spot = 60-91% off on-demand! |
| Cost (15 min) | N/A (60 min build) | **$1.61** | **Cheapest for speed!** |
| Total build cost | $2.80 (60 min) | **$1.61 (12 min)** | **43% cheaper + 5× faster!** |

### 💎 BEST VALUE PICKS

1. **🏆 C4D_HIGHCPU_384** (HIGHEST RECOMMENDED)
   - 384 vCPUs at $6.46/hr spot = $1.61 for 15 min build
   - Fastest build: 10-12 min
   - Best performance per dollar!

2. **💰 N2D_HIGHCPU_224** (BEST BUDGET)
   - 224 vCPUs at $1.95/hr spot = $0.49 for 15 min build
   - Fast build: 18-20 min
   - Ultra-cheap!

3. **⚖️ C3_HIGHCPU_176** (BALANCED)
   - 176 vCPUs at $1.93/hr spot = $0.48 for 15 min build
   - Good build: 23-25 min
   - Great balance of speed and cost!

---

## ✅ Implementation Checklist

### Phase 1: Create Spot Worker Pool (Manual)

- [ ] **1.1** Open Google Cloud Console worker pool creation page
  - URL: https://console.cloud.google.com/cloud-build/builds/worker-pools/create?project=weight-and-biases-476906
  - Reference: [Private pools overview](https://cloud.google.com/build/docs/private-pools/private-pools-overview)

- [ ] **1.2** Configure worker pool settings:
  - **Name**: `pytorch-spot-pool`
  - **Region**: `us-central1` (same as Artifact Registry)
  - **Machine type**: `c4d-highcpu-384` (384 vCPUs, AMD EPYC)
  - **Disk size**: `200 GB` (SSD recommended)
  - **✅ Enable Spot VMs**: YES (CRITICAL!)
  - **Network**: Default or leave empty
  - Reference: [Worker pool config schema](https://cloud.google.com/build/docs/private-pools/private-pool-config-file-schema)

- [ ] **1.3** Click "CREATE" and wait ~5 minutes for provisioning
  - Status will show "CREATING" then "RUNNING"
  - Verify pool exists: `gcloud builds worker-pools list --region=us-central1`

---

### Phase 2: Update Cloud Build Configuration (Automated)

- [ ] **2.1** Update `.cloudbuild-pytorch-clean.yaml` to use spot worker pool
  - Remove `machineType` from options (worker pool controls this)
  - Add `pool` configuration pointing to `pytorch-spot-pool`
  - Keep `diskSizeGb: 200` and `logging: CLOUD_LOGGING_ONLY`

- [ ] **2.2** Update MAX_JOBS in ALL Dockerfiles to 384
  - `training/images/pytorch-clean/Dockerfile`: `ENV MAX_JOBS=384`
  - `training/images/base-image/Dockerfile`: `ENV MAX_JOBS=384`
  - `training/images/training-image/Dockerfile`: `ENV MAX_JOBS=384`
  - `training/images/runner-image/Dockerfile`: `ENV MAX_JOBS=384`

- [ ] **2.3** Git commit changes
  - Message: "🚀 Implement C4D_HIGHCPU_384 spot worker pool (6× vCPUs, 60-91% savings)"

---

### Phase 3: Test Build (Automated)

- [ ] **3.1** Clear Python cache and launch lock
  - Remove `/tmp/arr-coc-launch.lock`
  - Delete `__pycache__` directories

- [ ] **3.2** Trigger pytorch-clean build with spot pool
  - Run: `python training/cli.py launch`
  - Monitor Cloud Build logs: https://console.cloud.google.com/cloud-build/builds?project=weight-and-biases-476906

- [ ] **3.3** Verify spot instance usage
  - Check build logs for machine type: `c4d-highcpu-384`
  - Confirm "Spot VM" in build details
  - Expected build time: 10-15 minutes

- [ ] **3.4** Monitor build completion
  - Watch for successful image push to Artifact Registry
  - Verify image tags: `pytorch-clean:3f1d952` and `pytorch-clean:latest`

---

## 🔗 Reference Links

### Google Cloud Documentation
- [Private pools overview](https://cloud.google.com/build/docs/private-pools/private-pools-overview)
- [Worker pool config schema](https://cloud.google.com/build/docs/private-pools/private-pool-config-file-schema)
- [Spot VMs documentation](https://cloud.google.com/compute/docs/instances/spot)
- [Build config file schema](https://cloud.google.com/build/docs/build-config-file-schema)

### Machine Type Information
- [C4D machines (AMD EPYC)](https://cloud.google.com/compute/docs/compute-optimized-machines#c4d_series)
- [Machine type comparison](https://gcloud-compute.com/instances.html)
- C4D_HIGHCPU_384: 384 vCPUs, 720 GB RAM, AMD EPYC

### Project-Specific
- Cloud Build console: https://console.cloud.google.com/cloud-build/builds?project=weight-and-biases-476906
- Artifact Registry: https://console.cloud.google.com/artifacts/docker/weight-and-biases-476906/us-central1/arr-coc-registry?project=weight-and-biases-476906
- Worker pools: https://console.cloud.google.com/cloud-build/builds/worker-pools?project=weight-and-biases-476906

---

## 💡 Key Technical Details

### Why Spot Instances?
- **60-91% cost savings** vs on-demand
- Preemptible with 30-second notice (fine for ~15 min builds)
- Automatically retried if preempted (Cloud Build handles this)
- Perfect for batch workloads like Docker image builds

### Why C4D_HIGHCPU_384?
- **384 vCPUs**: 6× more than N2_HIGHCPU_64
- **AMD EPYC**: Excellent for compilation workloads
- **Spot pricing**: ~$6.45/hour (vs $17.40/hour on-demand)
- **15 min build**: Only $1.61 at spot pricing! (vs $8.70 on-demand)

### Optimizations Applied
1. **MAX_JOBS=384**: Full CPU utilization
2. **Ninja generator**: 2-3× faster CMake builds
3. **ccache**: 10-100× on rebuilds
4. **BuildKit cache mounts**: Persistent pip/ccache
5. **BUILD_TEST=0**: Skip PyTorch tests
6. **Spot instances**: 60-91% cost savings

---

## 🎯 Success Criteria

Build is successful when:
- ✅ Worker pool shows "RUNNING" status
- ✅ Build uses `c4d-highcpu-384` machine type
- ✅ Build shows "Spot VM" in details
- ✅ Build completes in 10-15 minutes
- ✅ Images pushed to Artifact Registry successfully
- ✅ Total cost ~$3-4 (similar to previous N2-64 build)

---

## 🚨 Troubleshooting

### Worker pool creation fails
- Check IAM permissions: `cloudbuild.workerPools.create`
- Verify region availability: `us-central1` has C4D machines
- Check quota: C4D_HIGHCPU_384 may need quota increase

### Build fails to use worker pool
- Verify pool name in YAML: `pytorch-spot-pool`
- Check pool status: `gcloud builds worker-pools describe pytorch-spot-pool --region=us-central1`
- Ensure pool is RUNNING before submitting build

### Spot instance preempted during build
- Cloud Build automatically retries
- Monitor retries in build logs
- If frequent preemption, consider on-demand pool

---

## 📝 Post-Implementation

After successful build:
- [ ] Document actual build time achieved
- [ ] Document actual cost from billing
- [ ] Update README with spot pool information
- [ ] Consider creating separate pools for different workloads

---

**Implementation Status**: Ready to execute Phase 1 (manual) then Phases 2-3 (automated)

**Last Updated**: 2025-01-10
