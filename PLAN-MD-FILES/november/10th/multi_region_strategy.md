# Multi-Region Worker Pool Strategy

## Discovery (2025-01-11)

**CRITICAL FINDING**: C3 quota is GLOBAL, not regional!

We can create worker pools in **ALL 10 C3 regions** simultaneously for maximum redundancy and availability.

## Why Multi-Region Matters

### Regional Service Outages
- **us-central1** experienced C3 worker pool service issues (2025-01-11)
- Pools stuck "Creating..." for 20+ minutes, eventually failed
- E2 machines worked fine → Issue specific to C3 in that region

### Regional Performance Variance
- **us-east4** (Virginia): Pool created in ~1 minute ✅
- **europe-west1** (Belgium): Pool created in ~1 minute ✅
- **us-central1** (Iowa): C3 pools failed/stuck for 20+ minutes ❌

### Solution: Pre-Create Pools in Multiple Regions
Instead of creating pools on-demand (slow, can fail), we:
1. Pre-create small pools in multiple regions (c3-standard-4 for fast creation)
2. Use whichever region is healthy
3. Delete and recreate with correct machine type when needed
4. Fall back to other regions if primary fails

## All Available C3 Regions

### US Regions (Low Latency - Best for US-based users)
| Region | Location | Zones | Status |
|--------|----------|-------|--------|
| us-central1 | Iowa | a,b,c,f | ⚠️ C3 worker pools broken (2025-01-11) |
| us-east1 | South Carolina | b,c,d | ✅ Available |
| us-east4 | Northern Virginia | a,b,c | ✅ **TESTED - Works great!** |
| us-west1 | Oregon | a,b,c | ✅ Available |

**Recommended Primary: us-east4** (tested, fast, reliable)

### Europe Regions (Medium Latency)
| Region | Location | Zones | Status |
|--------|----------|-------|--------|
| europe-west1 | Belgium | b,c,d | ✅ **TESTED - Works great!** |
| europe-west2 | London, UK | a,b,c | ✅ Available |
| europe-west3 | Frankfurt, Germany | c | ✅ Available |

**Recommended: europe-west1** (tested, reliable)

### Asia/Pacific Regions (High Latency for US users)
| Region | Location | Zones | Status |
|--------|----------|-------|--------|
| asia-northeast1 | Tokyo, Japan | b,c | ✅ Available |
| asia-southeast1 | Singapore | a,b,c | ✅ Available |
| australia-southeast1 | Sydney, Australia | a,b,c | ✅ Available |

**Use for**: Asia-Pacific users or when all US/EU regions unavailable

## Implementation Strategy

### Phase 1: Test Pools (Fast Creation)
Create c3-standard-4 pools in top 3 regions:
- us-east4 (primary)
- europe-west1 (EU backup)
- us-west1 (US West backup)

Rationale:
- c3-standard-4 creates in ~1-2 minutes
- Cheap to keep running (~$0.16/hour)
- Fast validation that region is healthy

### Phase 2: Production Pools (On-Demand)
When launching PyTorch build:
1. Check if region's test pool is RUNNING
2. Delete test pool
3. Create production pool (c3-standard-176)
4. Wait for RUNNING status
5. Build PyTorch image

### Phase 3: Cleanup
After build completes:
- Delete production pool (expensive: ~$6.90/hour)
- Recreate test pool (cheap: ~$0.16/hour)

## Cost Analysis

**Test Pools (3 regions, always running):**
- 3 × c3-standard-4 @ $0.16/hour = $0.48/hour
- Monthly: $0.48 × 24 × 30 = $345.60/month

**Production Pools (on-demand, ~2 hours/month):**
- c3-standard-176 @ $6.90/hour × 2 hours = $13.80/month

**Total**: ~$360/month for multi-region redundancy

**Alternative (single region, on-demand only):**
- $13.80/month but no redundancy, higher failure risk

**Recommendation**: Keep test pools in 2-3 regions for critical builds

## Quota Implications

**CRITICAL**: C3 quota is shared across ALL regions!

If quota = 176 vCPUs:
- Can run ONE c3-standard-176 pool (uses all 176 vCPUs)
- OR THREE c3-standard-4 pools (4 + 4 + 4 = 12 vCPUs) ✅
- OR mix: one c3-standard-88 + two c3-standard-4 (88 + 4 + 4 = 96 vCPUs)

Current quota check showed: 176 vCPUs available globally
Therefore: We can run small test pools in multiple regions safely!

## Failure Recovery

If primary region (us-east4) fails:
1. Check test pools: `gcloud builds worker-pools list --region=<region>`
2. Use first RUNNING pool's region
3. Delete test pool in that region
4. Create production pool
5. Proceed with build

Regions tried in order:
1. us-east4 (Virginia) - Primary
2. us-west1 (Oregon) - US West backup
3. europe-west1 (Belgium) - EU backup
4. us-east1 (South Carolina) - US East backup
5. [All other regions...]

## Implementation Notes

**Worker pool creation is SLOW:**
- c3-standard-4: ~1-2 minutes (tested)
- c3-standard-176: ~10-45 minutes (varies by region/load)

**Worker pool creation can FAIL:**
- Regional service issues (us-central1 on 2025-01-11)
- Capacity constraints
- GCP infrastructure problems

**Solution: Multi-region redundancy eliminates single point of failure!**
