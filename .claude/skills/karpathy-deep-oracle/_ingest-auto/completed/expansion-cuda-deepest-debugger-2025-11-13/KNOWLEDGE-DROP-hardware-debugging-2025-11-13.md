# KNOWLEDGE DROP: Hardware-Level GPU Debugging

**Runner**: PART 2
**Timestamp**: 2025-11-13 (execution time)
**Status**: ✅ COMPLETE

---

## Knowledge File Created

**File**: `cuda/17-hardware-level-debugging-gpu-faults-expert.md`
**Lines**: ~620 lines
**Size**: Expert-level hardware fault debugging guide

---

## Content Breakdown

### Section 1: ECC Error Analysis (~150 lines)
- ECC architecture (SBE vs DBE, DRAM vs SRAM)
- nvidia-smi ECC monitoring commands
- NVML API for automated ECC tracking
- Error pattern analysis (isolated, recurring, burst, DBE)
- Production monitoring scripts

### Section 2: Memory Controller & Row Remapping (~150 lines)
- Row remapping architecture (A100/H100/H200)
- Comparison: Page retirement vs row remapping
- Remapping status monitoring (nvidia-smi, NVML)
- Xid 63/64 handling (remapping events/failures)
- RMA policy and criteria
- RAS features (error containment, dynamic page offlining, HBM channel repair)

### Section 3: Xid Error Codes & GPU Faults (~150 lines)
- Xid error fundamentals and log locations
- Critical datacenter Xids (13, 31, 43, 45, 48, 61-64, 74, 79, 92, 94-95)
- Xid diagnostic workflow
- Automated Xid monitoring script
- GPU reset capabilities and limitations by architecture

### Section 4: Thermal & Power Debugging (~150 lines)
- Thermal throttling detection (slowdown/shutdown temps)
- nvidia-smi thermal monitoring
- NVML thermal/throttle APIs
- Power monitoring and limit adjustment
- Xid 54 (auxiliary power) debugging
- Production health check scripts
- Pre-flight check automation

---

## Sources Used

**Primary NVIDIA Documentation:**
1. **GPU Debug Guidelines** (https://docs.nvidia.com/deploy/gpu-debug-guidelines/)
   - Complete Xid triage flowcharts
   - GPU reset capabilities by architecture
   - Fabric Manager integration

2. **Xid Errors Catalog** (https://docs.nvidia.com/deploy/xid-errors/)
   - Comprehensive Xid reference table
   - Recovery actions for each Xid
   - Xid 144-150 detailed decode

3. **GPU Memory Error Management** (https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/)
   - Row remapping architecture
   - Error containment (Xid 94/95)
   - RMA policy thresholds

4. **Row Remapping Documentation** (https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html)
   - Page retirement vs row remapping comparison
   - Spare row allocation mechanics
   - InfoROM persistence

**Web Research:**
- ECC error monitoring patterns
- NVML API usage examples
- Production health check automation
- Thermal management best practices

---

## Gaps Filled

**Before**:
- Basic ECC error mentions in cuda/11-advanced-troubleshooting-multi-gpu-expert.md
- Brief Xid 79/43 coverage
- Generic thermal throttling warnings

**After**:
- ✅ Complete ECC error classification (SBE/DBE, DRAM/SRAM)
- ✅ Row remapping architecture (A100+) vs page retirement (legacy)
- ✅ Comprehensive Xid catalog (15+ critical Xids with recovery workflows)
- ✅ Memory controller debugging (row remapping, RAS features)
- ✅ NVML API automation examples (ECC, thermal, power)
- ✅ Production health monitoring scripts
- ✅ RMA criteria and Field Diagnostics requirements
- ✅ GPU reset capabilities by architecture (Ampere, Hopper, NVSwitch)
- ✅ Thermal/power debugging with NVML
- ✅ Automated monitoring and pre-flight checks

---

## Key Insights

**ECC Error Management Evolution:**
- Legacy GPUs: 64 page retirements max, software-visible holes
- Ampere+: 512 row remappings, transparent to software, hardware-level fix
- Row remapping can replace correctable with uncorrectable when exhausted

**Critical Xid Patterns:**
- Xid 48 (DBE) + Xid 63/64 → Row remapping event/failure
- Xid 94 (contained) → Only restart affected app
- Xid 95 (uncontained) → Immediate GPU reset/reboot
- Xid 79 → Catastrophic (GPU off bus), often hardware failure

**Production Monitoring:**
- Volatile vs aggregate ECC counters
- SRAM threshold flag for RMA criteria
- Pending remappings require GPU reset
- Thermal/power throttling impacts performance silently

**GPU Reset Complexity:**
- NVSwitch systems need Fabric Manager coordination
- Hopper simplifies reset (FM-independent)
- Cannot reset with active processes
- Reset clears volatile counters but not aggregate

---

## Expert-Level Coverage

**Achieved ABSOLUTE DEEPEST hardware debugging:**
- Hardware fault taxonomy (ECC, memory controller, thermal, power)
- Row remapping internals (spare rows, bank exhaustion, InfoROM)
- Complete Xid catalog with recovery decision trees
- NVML API for production automation
- RAS feature integration (containment, page offlining, channel repair)
- Multi-architecture reset workflows
- Field Diagnostics and RMA criteria

**Real production scenarios:**
- Handling Xid 48 DBE during training run
- Detecting thermal throttling in multi-GPU clusters
- Automating ECC health checks in job prologs
- Diagnosing NVLink failures (Xid 74)
- Recovering from GPU bus failures (Xid 79)

---

## Integration with Existing Knowledge

**Complements:**
- cuda/11-advanced-troubleshooting-multi-gpu-expert.md (adds hardware layer)
- cuda/15-production-deployment-troubleshooting-expert.md (adds RAS monitoring)
- cuda/12-kernel-debugging-internals-expert.md (different debugging layer)

**Extends:**
- ECC monitoring from basic to production-grade automation
- Xid errors from brief mentions to comprehensive catalog
- GPU reset from simple command to architecture-aware workflows
- Health monitoring from manual checks to automated scripts

---

**Completion**: PART 2 successfully delivered hardware-level GPU debugging expertise (ECC, memory controller, Xid errors, thermal/power, RAS)
