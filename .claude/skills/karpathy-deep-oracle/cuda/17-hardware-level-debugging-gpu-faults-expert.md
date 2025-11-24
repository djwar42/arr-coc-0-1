# Hardware-Level GPU Debugging: ECC, Memory Controller, Faults & RAS

**Deep expertise in GPU hardware fault diagnosis, memory error management, and production health monitoring**

---

## Section 1: ECC Error Analysis & Memory Health (~150 lines)

### 1.1 ECC Error Types & Architecture

**ECC (Error Correction Code) Memory** protects GPU DRAM from bit flips:

**Error Classification:**
- **Single-Bit Errors (SBE)**: Correctable by ECC, GPU continues operation
- **Double-Bit Errors (DBE)**: Uncorrectable, requires GPU reset or reboot
- **SRAM Errors**: Separate from framebuffer DRAM, different thresholds
- **DRAM Errors**: Framebuffer memory (HBM on datacenter GPUs)

From [NVIDIA GPU Memory Error Management](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/index.html) (accessed 2025-11-13):
- A100/H100/H200 use **row remapping** instead of page retirement
- Supports up to 512 remappings per GPU (vs 64 retirements on legacy GPUs)
- Remapping occurs at hardware level, no software-visible holes in address space

**ECC Memory Locations:**
```
GPU Memory Hierarchy (ECC Coverage):
├── DRAM/HBM (Frame Buffer)
│   ├── Row-level ECC (Ampere+)
│   └── Page-level retirement (Volta/Turing)
├── L2 Cache
│   └── ECC protected
├── Register Files
│   └── Parity/ECC protected
└── SRAM (Various units)
    └── Separate error tracking
```

### 1.2 Monitoring ECC Errors with nvidia-smi

**Query ECC error counters:**
```bash
# Basic ECC status
nvidia-smi --query-gpu=ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv

# Output example:
# ecc.mode.current, ecc.errors.corrected.volatile.total, ecc.errors.uncorrected.volatile.total
# Enabled, 143, 2  # 2 uncorrectable = GPU likely failing!

# Detailed ECC memory breakdown
nvidia-smi -i 0 -q -d MEMORY

# Shows:
# - FB Data (frame buffer DRAM)
# - FB ECC (frame buffer ECC)
# - Texture Memory
# - Device Memory
# - Register File
# - L1 Cache
# - L2 Cache

# Reset volatile ECC counters (requires root)
nvidia-smi -i 0 --reset-ecc-errors

# Check aggregate ECC errors (persistent across reboots)
nvidia-smi -i 0 --query-gpu=ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv
```

**ECC error thresholds:**
```bash
# High SBE rate (Xid 92)
# Check if SBE interrupt rate exceeds threshold
# nvidia-smi will show increasing corrected errors

# SRAM threshold exceeded flag
# Check via nvidia-smi or NVML
# If SRAM DBE threshold exceeded → Run Field Diagnostics

# DRAM uncorrectable errors
# Any DBE (Xid 48) → Immediate action required
```

### 1.3 NVML API for ECC Monitoring

**Python example using pynvml:**
```python
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

    # Get ECC mode
    current_mode, pending_mode = pynvml.nvmlDeviceGetEccMode(handle)
    print(f"GPU {i} ECC Mode: {current_mode}")

    # Get detailed ECC errors
    # Memory types: NVML_MEMORY_LOCATION_L1_CACHE, L2_CACHE, DEVICE_MEMORY, etc.
    # Error types: NVML_MEMORY_ERROR_TYPE_CORRECTED, UNCORRECTED
    # Counter types: NVML_VOLATILE_ECC (since last reset), NVML_AGGREGATE_ECC (total)

    try:
        # DRAM correctable errors
        dram_sbe = pynvml.nvmlDeviceGetMemoryErrorCounter(
            handle,
            pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
            pynvml.NVML_VOLATILE_ECC,
            pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
        )

        # DRAM uncorrectable errors
        dram_dbe = pynvml.nvmlDeviceGetMemoryErrorCounter(
            handle,
            pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            pynvml.NVML_VOLATILE_ECC,
            pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
        )

        print(f"GPU {i} DRAM: SBE={dram_sbe}, DBE={dram_dbe}")

        # Check if DBE threshold exceeded
        if dram_dbe > 0:
            print(f"⚠️ GPU {i} has {dram_dbe} uncorrectable DRAM errors - ACTION REQUIRED")

    except pynvml.NVMLError as e:
        print(f"Error querying ECC: {e}")

pynvml.nvmlShutdown()
```

**Automated ECC monitoring script:**
```python
import pynvml
import time

def monitor_ecc_errors(interval_seconds=60):
    """Monitor ECC errors and alert on thresholds"""
    pynvml.nvmlInit()

    # Thresholds
    SBE_RATE_THRESHOLD = 100  # SBEs per minute
    DBE_THRESHOLD = 1  # Any DBE is critical

    previous_sbe = {}

    while True:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get current errors
            sbe_total = pynvml.nvmlDeviceGetMemoryErrorCounter(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                pynvml.NVML_VOLATILE_ECC,
                pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
            )

            dbe_total = pynvml.nvmlDeviceGetMemoryErrorCounter(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC,
                pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
            )

            # Check DBE threshold
            if dbe_total >= DBE_THRESHOLD:
                print(f"🚨 CRITICAL: GPU {i} has {dbe_total} uncorrectable errors")
                print(f"   → Drain node, reset GPU")

            # Check SBE rate
            if i in previous_sbe:
                sbe_rate = (sbe_total - previous_sbe[i]) / (interval_seconds / 60.0)
                if sbe_rate > SBE_RATE_THRESHOLD:
                    print(f"⚠️  WARNING: GPU {i} SBE rate: {sbe_rate:.1f}/min (threshold: {SBE_RATE_THRESHOLD})")
                    print(f"   → Monitor closely, may need Field Diagnostics")

            previous_sbe[i] = sbe_total

        time.sleep(interval_seconds)

    pynvml.nvmlShutdown()
```

### 1.4 ECC Error Patterns & Analysis

**Common ECC error patterns:**

1. **Isolated Single-Bit Errors** (random location, low rate)
   - Cause: Cosmic rays, alpha particles, natural radiation
   - Action: Normal operation, monitor aggregate count
   - Concern: If rate increases over time

2. **Recurring Errors in Same Location** (same address/bank)
   - Cause: Degrading memory cell, manufacturing defect
   - Action: Triggers row remapping (A100+) or page retirement (legacy)
   - Concern: If remapping fails

3. **Burst of Single-Bit Errors** (multiple locations, short time)
   - Cause: Thermal event, power supply glitch, marginal DRAM
   - Action: Check thermal/power, monitor for recurrence
   - Concern: May precede DBE

4. **Double-Bit Error** (uncorrectable)
   - Cause: Severe bit flips, failed ECC, memory failure
   - Action: **IMMEDIATE** - Reset GPU, check for Xid 63/64
   - Concern: Data corruption, application crash

**ECC error diagnostic workflow:**
```bash
# Step 1: Identify error type and location
nvidia-smi -i 0 -q -d MEMORY | grep -A 10 "ECC Errors"

# Step 2: Check if errors are increasing
# Run multiple times, compare counts
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv -l 5

# Step 3: Check memory temperature (if supported)
nvidia-smi --query-gpu=temperature.memory --format=csv

# Step 4: Run GPU memory test (if available)
# DCGM diagnostics includes memory stress test
dcgmi diag -r 3  # Long test, ~30 min

# Step 5: Check InfoROM for remapping status (A100+)
nvidia-smi -i 0 --query-gpu=retired_pages.single_bit_ecc.count,retired_pages.double_bit_ecc.count --format=csv
```

---

## Section 2: Memory Controller & Row Remapping (~150 lines)

### 2.1 Row Remapping Architecture (A100/H100/H200)

From [Row Remapping Documentation](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html) (accessed 2025-11-13):

**Row Remapping vs Page Retirement:**

| Feature | Page Retirement (Legacy) | Row Remapping (Ampere+) |
|---------|--------------------------|-------------------------|
| **Capacity** | Max 64 retirements | Up to 512 remappings |
| **Granularity** | 4KB pages | Hardware rows (sub-page) |
| **Address Space** | Software-visible holes | No holes, transparent remap |
| **Policy** | Permanent, cannot unretire | Can replace correctable with uncorrectable |
| **Activation** | Driver reload or GPU reset | GPU reset required |
| **RMA Criteria** | 64 retirements reached | Row remapping failure flag |

**Row remapping process:**
```
1. ECC detects uncorrectable error in memory row
2. GPU firmware identifies failing row
3. Firmware maps row to spare row from reserved pool
4. Remapping recorded in InfoROM (non-volatile)
5. GPU reset applies the remapping
6. GPU continues operation with remapped memory
```

**Spare row allocation:**
- Each DRAM bank has reserved spare rows
- Spares used for both correctable and uncorrectable error remapping
- When bank exhausted: subsequent errors cannot remap
- Triggers "row-remapping failure" flag → RMA criteria

### 2.2 Monitoring Row Remapping Status

**Check remapping counters:**
```bash
# Query remapped rows (A100+)
nvidia-smi --query-gpu=retired_pages.single_bit_ecc.count,retired_pages.double_bit_ecc.count,retired_pages.pending --format=csv

# Output (terminology note: uses "retired pages" but actually row remapping on A100+):
# retired_pages.single_bit_ecc.count, retired_pages.double_bit_ecc.count, retired_pages.pending
# 3, 1, Yes  # 3 SBE remaps, 1 DBE remap, pending=needs GPU reset

# Check if remapping failure flag set
nvidia-smi -i 0 -q | grep -i "retired\|remap"

# Example output:
#     Retired Pages
#         Single Bit ECC             : 3
#         Double Bit ECC             : 1
#         Pending Page Blacklists    : Yes  # GPU reset needed
```

**NVML API for remapping status:**
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Query retired (remapped) pages
# Cause: NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
#        NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR
try:
    retired_pages = pynvml.nvmlDeviceGetRetiredPages(handle,
        pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR)
    print(f"DBE retired pages: {len(retired_pages)}")

    # Check if pending retirement (needs GPU reset)
    is_pending = pynvml.nvmlDeviceGetRetiredPages_isPendingRetirement(handle)
    if is_pending:
        print("⚠️ Pending retirements - GPU reset required to activate")

except pynvml.NVMLError as e:
    print(f"Error: {e}")

pynvml.nvmlShutdown()
```

### 2.3 Row Remapping Xid Events

**Xid 63: Row-Remapping Recording Event**

From [Xid Catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html) (accessed 2025-11-13):

**On A100+ GPUs:**
- Indicates successful row remapping recorded
- If associated with Xid 94: Application that hit error needs restart
- Other applications can continue until convenient GPU reset
- GPU reset activates the remapping

**Example Xid 63 message:**
```
NVRM: Xid (PCI:0000:1b:00): 63, Row remapper: New row marked for remapping
```

**Response to Xid 63:**
```bash
# 1. Check if associated with Xid 94 (contained error)
dmesg | grep -i xid | tail -20

# If Xid 94 present:
#   → Restart affected application
#   → Other apps can continue
#   → Schedule GPU reset at convenient time

# If standalone Xid 63:
#   → Single-bit error, row remapped
#   → Continue operation
#   → Reset GPU at maintenance window

# 2. Verify remapping recorded
nvidia-smi --query-gpu=retired_pages.pending --format=csv
# If "Yes" → reset needed to activate

# 3. At convenient time, reset GPU
nvidia-smi -i 0 -r  # Requires no processes using GPU
```

**Xid 64: Row-Remapping Recording Failure**

**Critical event** - remapping failed to record:
- Spare rows exhausted in affected bank
- InfoROM write failure
- GPU must be reset **immediately**

**Response to Xid 64:**
```bash
# IMMEDIATE ACTION REQUIRED
# 1. Drain node (stop all GPU work)

# 2. Reset GPU
nvidia-smi -i 0 -r

# 3. If errors persist after reset:
#    → Run Field Diagnostics
#    → Prepare for RMA

# 4. Check RMA criteria (see Section 2.4)
```

### 2.4 RMA Policy for Row Remapping

From [RMA Policy Documentation](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html) (accessed 2025-11-13):

**RMA (Return Merchandise Authorization) criteria:**

**Immediate RMA triggers:**
1. **Row-remapping failure flag set** (Xid 64)
   - Indicates bank exhausted spare rows
   - Future errors in that bank cannot be remapped
   - Check with: `nvidia-smi -i 0 -q | grep -i "remapping failure"`

2. **Excessive remapping events**
   - Threshold varies by GPU model and deployment
   - Consult vendor-specific RMA policy
   - Generally: >10 remapping failures or >100 remappings total

3. **SRAM uncorrectable error threshold exceeded**
   - Check SRAM threshold flag: `nvidia-smi` or NVML
   - Run Field Diagnostics if flag set

**Field Diagnostics requirement:**
```bash
# Before RMA, Field Diagnostics usually required
# Contact system vendor for instructions

# NVIDIA Field Diagnostic tool (vendor-provided)
# Comprehensive GPU hardware test
# Usually takes 1-2 hours
# Required for RMA authorization
```

### 2.5 RAS (Reliability, Availability, Serviceability) Features

**GPU RAS capabilities:**

**Error Containment** (A100+):
- **Contained errors** (Xid 94): Isolated to one application
  - Other apps continue running
  - Only affected app needs restart
- **Uncontained errors** (Xid 95): Affects multiple apps
  - GPU reset required before new apps start
  - All running apps must be stopped

**Dynamic Page Offlining** (A100+):
- Software-level page blacklisting
- Complements row remapping
- Prevents reuse of problematic memory regions

**HBM Channel Repair** (H100+):
- Hardware mechanism to repair entire HBM channels
- After 2 row remappings in same bank
- Next uncorrectable error triggers channel repair
- Uses spare HBM channel if available

**RAS monitoring workflow:**
```python
import pynvml

def check_ras_status(gpu_id):
    """Check GPU RAS health indicators"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    # 1. Check ECC errors
    dbe_count = pynvml.nvmlDeviceGetMemoryErrorCounter(
        handle,
        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
        pynvml.NVML_AGGREGATE_ECC,  # Total since manufacturing
        pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
    )

    # 2. Check remapping status
    try:
        is_pending = pynvml.nvmlDeviceGetRetiredPages_isPendingRetirement(handle)
        sbe_remaps = len(pynvml.nvmlDeviceGetRetiredPages(
            handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS))
        dbe_remaps = len(pynvml.nvmlDeviceGetRetiredPages(
            handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR))
    except pynvml.NVMLError:
        is_pending = False
        sbe_remaps = 0
        dbe_remaps = 0

    # 3. Assess health
    health_status = "HEALTHY"
    actions = []

    if dbe_count > 0:
        health_status = "WARNING"
        actions.append("Uncorrectable errors detected - monitor closely")

    if dbe_remaps > 10:
        health_status = "DEGRADED"
        actions.append("Excessive row remappings - check RMA criteria")

    if is_pending:
        actions.append("Pending remappings - reset GPU at convenient time")

    pynvml.nvmlShutdown()

    return {
        'status': health_status,
        'dbe_count': dbe_count,
        'sbe_remaps': sbe_remaps,
        'dbe_remaps': dbe_remaps,
        'pending_reset': is_pending,
        'actions': actions
    }

# Usage
status = check_ras_status(0)
print(f"GPU 0 Status: {status['status']}")
for action in status['actions']:
    print(f"  → {action}")
```

---

## Section 3: Xid Error Codes & GPU Fault Diagnosis (~150 lines)

### 3.1 Xid Error Fundamentals

From [Xid Error Documentation](https://docs.nvidia.com/deploy/xid-errors/) (accessed 2025-11-13):

**Xid messages** are error reports from NVIDIA driver indicating GPU errors. Format:
```
NVRM: Xid (PCI:0000:BB:DD.F): <Xid_Number>, <Description/Details>
```

**Finding Xid errors:**
```bash
# Linux system logs
dmesg | grep -i xid

# Or from syslog
grep -i xid /var/log/syslog

# Or from journalctl
journalctl -k | grep -i xid

# Example Xid errors:
# NVRM: Xid (PCI:0000:1b:00): 48, Double Bit ECC Error
# NVRM: Xid (PCI:0000:1b:00): 79, GPU has fallen off the bus
# NVRM: Xid (PCI:0000:1b:00): 13, Graphics Engine Exception
```

### 3.2 Critical Xid Errors for Production (Data Center)

**Most common Xids in datacenter deployments:**

**Xid 13: Graphics Engine Exception**
- **Cause**: User application fault, illegal memory access, out-of-bounds array
- **Recovery**: Restart application
- **Debug**: Run in `cuda-gdb` or Compute Sanitizer memcheck
```bash
# Debug workflow
CUDA_DEVICE_WAITS_ON_EXCEPTION=1 ./my_app  # GPU waits on exception
# Then attach cuda-gdb to inspect

# Or run with memcheck
compute-sanitizer --tool memcheck ./my_app
```

**Xid 31: GPU Memory Page Fault**
- **Cause**: MMU fault, illegal address access, driver/app bug
- **Recovery**: Restart application
- **Debug**: Same as Xid 13 (cuda-gdb, memcheck)

**Xid 43: GPU Stopped Processing**
- **Cause**: GPU hung, infinite loop, software fault
- **Recovery**: Informational only (GPU remains healthy)
- **Action**: No GPU reset needed, just restart app

**Xid 45: Preemptive Removal**
- **Cause**: Channel cleanup after previous error, multiple apps
- **Recovery**:
  - Solo (alone): Restart Fabric Manager (NVSwitch systems)
  - With other Xid: Follow other Xid's guidance
- **Note**: Usually symptomatic, not root cause

**Xid 48: Double Bit ECC Error** 🚨
- **Cause**: Uncorrectable DRAM/SRAM error
- **Recovery**:
  - If followed by Xid 63/64: Drain node, reset GPU
  - Solo: Run Field Diagnostics
- **Critical**: Check if SRAM error (check threshold flag)
```bash
# Check SRAM threshold via NVML or nvidia-smi
# If SRAM threshold exceeded → RMA candidate

# Reset GPU after DBE
nvidia-smi -i 0 -r
```

**Xid 61/62: PMU (Power Management Unit) Errors**
- **Cause**: Internal microcontroller halt, firmware error
- **Recovery**: Reset GPU, report issue to vendor
- **Rare**: Uncommon in production

**Xid 63: Row-Remapping Event** (A100+)
- **Cause**: Memory row remapped due to errors
- **Recovery**:
  - With Xid 94: Restart affected app
  - Solo: Schedule GPU reset at convenient time
- **Action**: Continue operation until reset window

**Xid 64: Row-Remapping Failure** (A100+) 🚨
- **Cause**: Spare rows exhausted, InfoROM failure
- **Recovery**: **IMMEDIATE** - Reboot node
- **Critical**: Check RMA criteria

**Xid 74: NVLink Error**
- **Cause**: NVLink connection error, remote device failure
- **Recovery**: Complex, depends on error details (see Xid 74 workflow)
- **Debug**: `nvidia-smi nvlink -e` for detailed link error info

**Xid 79: GPU Fallen Off the Bus** 🚨
- **Cause**: PCIe link failure, hardware failure, power loss
- **Recovery**: Drain node, report to vendor
- **Critical**: Often catastrophic hardware failure
```bash
# Check PCIe link status
lspci -vvv -s 1b:00.0 | grep -i "lnksta\|lnkcap"

# Check kernel PCIe logs
dmesg | grep -i pcie | grep -i err
```

**Xid 92: High Single-Bit ECC Error Rate**
- **Cause**: Excessive SBE interrupts
- **Recovery**: Run Field Diagnostics
- **Monitor**: Check if rate continues to increase

**Xid 94: Contained ECC Error** (A100+)
- **Cause**: Uncorrectable error isolated to one app
- **Recovery**: Restart affected application only
- **Other apps**: Can continue, reset GPU when convenient

**Xid 95: Uncontained ECC Error** (A100+) 🚨
- **Cause**: Uncorrectable error affecting multiple apps
- **Recovery**:
  - MIG enabled: Reset GPU after draining GPU instances
  - MIG disabled: **IMMEDIATE** reboot
- **Critical**: Cannot continue without reset

### 3.3 Xid Error Diagnostic Workflow

**Step-by-step Xid triage:**

```bash
# Step 1: Identify Xid and GPU
dmesg | grep -i "xid" | tail -20

# Example: Xid (PCI:0000:1b:00): 48, Double Bit ECC Error

# Step 2: Identify GPU device
lspci | grep -i nvidia | grep 1b:00
# 1b:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)

# Step 3: Check nvidia-smi for GPU status
nvidia-smi -i 0 -q | head -50

# Step 4: Check associated Xids (often appear in clusters)
dmesg | grep "PCI:0000:1b:00" | grep -i xid

# Step 5: Consult Xid catalog (see Section 3.2) for recovery action
```

**Automated Xid monitoring script:**
```python
#!/usr/bin/env python3
import subprocess
import re
import time

def monitor_xid_errors(interval=60):
    """Monitor dmesg for Xid errors"""

    # Critical Xids requiring immediate action
    CRITICAL_XIDS = {
        48: "Double Bit ECC Error - Reset GPU",
        64: "Row Remapping Failure - IMMEDIATE REBOOT",
        79: "GPU Fallen Off Bus - Hardware failure",
        95: "Uncontained ECC Error - IMMEDIATE REBOOT"
    }

    seen_xids = set()

    while True:
        # Parse dmesg for Xid errors
        result = subprocess.run(['dmesg'], capture_output=True, text=True)

        for line in result.stdout.splitlines():
            if 'NVRM: Xid' in line:
                # Parse Xid number and GPU PCI ID
                match = re.search(r'Xid \(PCI:([0-9a-f:]+)\): (\d+)', line)
                if match:
                    pci_id = match.group(1)
                    xid_num = int(match.group(2))

                    xid_key = (pci_id, xid_num, line)
                    if xid_key not in seen_xids:
                        seen_xids.add(xid_key)

                        # Alert on critical Xids
                        if xid_num in CRITICAL_XIDS:
                            print(f"🚨 CRITICAL XID DETECTED!")
                            print(f"   GPU: {pci_id}")
                            print(f"   Xid {xid_num}: {CRITICAL_XIDS[xid_num]}")
                            print(f"   Full message: {line}")
                        else:
                            print(f"⚠️  Xid {xid_num} on GPU {pci_id}")

        time.sleep(interval)

if __name__ == "__main__":
    monitor_xid_errors()
```

### 3.4 GPU Reset Capabilities & Limitations

From [GPU Debug Guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html) (accessed 2025-11-13):

**GPU reset via nvidia-smi:**
```bash
# Reset single GPU
nvidia-smi -i 0 -r

# Requirements:
# - Root access
# - No processes using GPU (CUDA apps, X server, nvidia-smi, etc.)
# - No other instances of nvidia-smi running

# Check for blocking processes
fuser -v /dev/nvidia0

# Kill blocking processes if safe
# (DO NOT kill system processes blindly!)
```

**Reset capabilities by architecture:**

| GPU Architecture | Fabric Manager | Reset Support |
|------------------|----------------|---------------|
| **Ampere + NVLink** (direct) | N/A | Individual GPU reset supported |
| **Ampere + NVSwitch** | Running | Individual GPU reset + auto NVSwitch link reset |
| **Ampere + NVSwitch** | NOT running | Reset all GPUs + NVSwitches together |
| **Hopper + NVSwitch** | Any state | Individual GPU reset supported |

**When GPU reset fails:**
```bash
# If nvidia-smi -r fails:
# 1. Check for blocking processes
lsof | grep nvidia

# 2. Stop CUDA applications
# (Application-specific)

# 3. Stop monitoring tools
systemctl stop dcgm  # If running DCGM

# 4. Try reset again
nvidia-smi -i 0 -r

# If still fails:
# 5. Reload driver module
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia

# 6. Last resort: Reboot node
reboot
```

---

## Section 4: Thermal & Power Debugging (~150 lines)

### 4.1 Thermal Throttling Detection

**GPU thermal limits:**
- **Slowdown Temperature**: GPU begins reducing clocks (typically ~80-85°C)
- **Shutdown Temperature**: GPU shuts down to prevent damage (typically ~90-95°C)
- **Memory Temperature**: Separate sensor on HBM (A100: ~95°C throttle)

**Monitor GPU temperature:**
```bash
# Current temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Continuous monitoring with clocks
nvidia-smi dmon -i 0 -s pucvmet -c 100

# Output columns:
# gpu   pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk
#   0   350     78     -     98     45      0      0   9501   1410  # Normal
#   0    75     82     -      0      0      0      0   9501    300  # Throttled! (low pclk)

# Check throttle reasons
nvidia-smi -i 0 -q -d PERFORMANCE | grep -A 10 "Clocks Throttle Reasons"

# Example output:
#     Clocks Throttle Reasons
#         Idle                        : Not Active
#         Applications Clocks Setting : Not Active
#         SW Power Cap                : Not Active
#         HW Slowdown                 : Active  ← Thermal throttling!
#         Sync Boost                  : Not Active
#         SW Thermal Slowdown         : Not Active
#         HW Thermal Slowdown         : Active  ← Thermal limit reached
#         HW Power Brake Slowdown     : Not Active
```

**NVML API for thermal monitoring:**
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get GPU core temperature
gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
print(f"GPU Temperature: {gpu_temp}°C")

# Get memory temperature (if supported, A100+)
try:
    mem_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)
    print(f"Memory Temperature: {mem_temp}°C")
except pynvml.NVMLError:
    print("Memory temperature not supported on this GPU")

# Check throttle reasons
throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)

if throttle_reasons & pynvml.nvmlClocksThrottleReasonThermal:
    print("⚠️ GPU is thermally throttled!")

if throttle_reasons & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown:
    print("⚠️ Software thermal slowdown active")

if throttle_reasons & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown:
    print("🚨 Hardware thermal slowdown active - cooling issue!")

# Get thermal shutdown limit
try:
    shutdown_temp = pynvml.nvmlDeviceGetTemperatureThreshold(
        handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
    slowdown_temp = pynvml.nvmlDeviceGetTemperatureThreshold(
        handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)

    print(f"Slowdown Threshold: {slowdown_temp}°C")
    print(f"Shutdown Threshold: {shutdown_temp}°C")
except pynvml.NVMLError:
    pass

pynvml.nvmlShutdown()
```

**Thermal throttling mitigation:**
```bash
# 1. Check datacenter cooling
# - Inlet temperature should be 18-27°C
# - Check for blocked air intakes
# - Verify HVAC operational

# 2. Check GPU fan operation (if air-cooled)
nvidia-smi --query-gpu=fan.speed --format=csv

# 3. Reduce power limit to reduce heat
nvidia-smi -i 0 -pl 300  # Limit to 300W (from 400W)

# 4. Enable persistence mode (reduces init thermal spikes)
nvidia-smi -i 0 -pm 1

# 5. Monitor over time
watch -n 5 'nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,clocks_throttle_reasons.hw_slowdown --format=csv'
```

### 4.2 Power Monitoring & Debugging

**Power-related issues:**
- **Power limit throttling**: GPU hitting configured power limit
- **Power brake**: Hardware emergency power reduction
- **Auxiliary power**: External PCIe power connectors not connected (Xid 54)

**Monitor GPU power:**
```bash
# Query power draw and limits
nvidia-smi --query-gpu=power.draw,power.limit,power.max_limit --format=csv

# Example output:
# power.draw [W], power.limit [W], enforced.power.limit [W]
# 387.23 W, 400.00 W, 400.00 W  # Near limit

# Detailed power query
nvidia-smi -i 0 -q -d POWER

# Shows:
# - Current power draw
# - Power limit (configurable)
# - Default power limit
# - Enforced power limit
# - Min/Max power limits

# Check for power throttling
nvidia-smi -i 0 -q -d PERFORMANCE | grep "SW Power Cap"
```

**NVML power monitoring:**
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get power draw
power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
power_w = power_mw / 1000.0
print(f"Current Power: {power_w:.2f} W")

# Get power limit
power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
print(f"Power Limit: {power_limit:.2f} W")

# Check if hitting power limit
if power_w > power_limit * 0.95:
    print("⚠️ GPU approaching power limit - may throttle")

# Get power limit constraints
try:
    min_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[0] / 1000.0
    max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
    print(f"Configurable Power Range: {min_limit:.0f}W - {max_limit:.0f}W")
except pynvml.NVMLError:
    pass

# Check for power throttling
throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
if throttle_reasons & pynvml.nvmlClocksThrottleReasonSwPowerCap:
    print("GPU throttled due to software power cap")
if throttle_reasons & pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown:
    print("🚨 Hardware power brake activated - power delivery issue!")

pynvml.nvmlShutdown()
```

**Power limit adjustment:**
```bash
# Set power limit (requires root, within min/max range)
nvidia-smi -i 0 -pl 350  # Set to 350W

# Check new limit applied
nvidia-smi --query-gpu=power.limit --format=csv

# Reset to default
nvidia-smi -i 0 -pl $(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader)

# Note: Power limit persists until:
# - Driver reload
# - System reboot
# - Manual reset
```

**Xid 54: Auxiliary Power Not Connected**
```bash
# Error: NVRM: Xid (PCI:0000:1b:00): 54, Auxiliary power is not connected

# Cause: External PCIe power cables not fully connected
# Recovery: Power down, reconnect PCIe power cables, power up

# Verify power cable connection:
# 1. Power down system
# 2. Check all GPU power connectors
# 3. Verify cables fully seated
# 4. Power up and verify
nvidia-smi -q -d POWER
```

### 4.3 Production GPU Health Checks

**Automated health monitoring script:**
```python
#!/usr/bin/env python3
import pynvml
import time

def gpu_health_check():
    """Comprehensive GPU health check for production"""
    pynvml.nvmlInit()

    issues = []

    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)

        # 1. Check ECC errors
        try:
            dbe = pynvml.nvmlDeviceGetMemoryErrorCounter(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC,
                pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
            )
            if dbe > 0:
                issues.append(f"GPU {i} ({name}): {dbe} uncorrectable ECC errors")
        except pynvml.NVMLError:
            pass

        # 2. Check temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        if temp > 80:
            issues.append(f"GPU {i} ({name}): High temperature {temp}°C")

        # 3. Check throttling
        throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        if throttle & (pynvml.nvmlClocksThrottleReasonThermal |
                       pynvml.nvmlClocksThrottleReasonSwThermalSlowdown |
                       pynvml.nvmlClocksThrottleReasonHwThermalSlowdown):
            issues.append(f"GPU {i} ({name}): Thermal throttling active")

        # 4. Check power
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        if power > power_limit * 0.98:
            issues.append(f"GPU {i} ({name}): At power limit ({power:.0f}W/{power_limit:.0f}W)")

        # 5. Check pending remappings
        try:
            if pynvml.nvmlDeviceGetRetiredPages_isPendingRetirement(handle):
                issues.append(f"GPU {i} ({name}): Pending row remappings - reset needed")
        except pynvml.NVMLError:
            pass

    pynvml.nvmlShutdown()

    if issues:
        print("🚨 GPU Health Issues Detected:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ All GPUs healthy")
        return True

# Run health check
if __name__ == "__main__":
    while True:
        gpu_health_check()
        time.sleep(300)  # Check every 5 minutes
```

**Pre-flight checks for production jobs:**
```bash
#!/bin/bash
# GPU health check before launching job

echo "=== GPU Pre-Flight Check ==="

# 1. Check GPU driver loaded
if ! lsmod | grep -q nvidia; then
    echo "❌ NVIDIA driver not loaded"
    exit 1
fi

# 2. Check all GPUs visible
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "✓ Found $gpu_count GPUs"

# 3. Check for ECC errors
ecc_errors=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader | awk '{s+=$1} END {print s}')
if [ "$ecc_errors" -gt 0 ]; then
    echo "❌ Uncorrectable ECC errors detected: $ecc_errors"
    exit 1
fi

# 4. Check temperature
max_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)
if [ "$max_temp" -gt 85 ]; then
    echo "⚠️  High GPU temperature: ${max_temp}°C"
fi

# 5. Check for recent Xid errors
xid_errors=$(dmesg | grep "NVRM: Xid" | tail -5)
if [ -n "$xid_errors" ]; then
    echo "⚠️  Recent Xid errors detected:"
    echo "$xid_errors"
fi

# 6. Run quick DCGM diagnostic (optional)
# dcgmi diag -r 1  # Quick diagnostic

echo "✅ GPU pre-flight check passed"
exit 0
```

---

## Sources

**NVIDIA Official Documentation:**
- [GPU Debug Guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html) - GPU error debug and diagnosis (accessed 2025-11-13)
- [Xid Errors](https://docs.nvidia.com/deploy/xid-errors/) - Comprehensive Xid error catalog (accessed 2025-11-13)
- [Xid Catalog Analysis](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html) - Detailed Xid decode workflow (accessed 2025-11-13)
- [NVIDIA GPU Memory Error Management](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/index.html) - A100/H100/H200 memory error features (accessed 2025-11-13)
- [Row Remapping](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html) - Row remapping architecture and RMA policy (accessed 2025-11-13)
- [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/) - NVIDIA Management Library documentation

**Source Documents:**
- None (web research only)

**Web Research:**
- Search: "CUDA ECC error analysis debugging memory controller 2024"
- Search: "GPU hardware faults Xid errors NVIDIA debugging"
- Search: "NVML API GPU health monitoring temperature throttling"
- Search: "GPU memory controller errors row remapping RAS"

**Additional References:**
- [DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-user-guide/dcgm-diagnostics.html) - Production GPU diagnostics
- [Dynamic Page Retirement](https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html) - Legacy GPU ECC handling (pre-Ampere)
- vLLM Xid error handling patterns (production ML deployment experience)
- PyTorch CUDA error recovery strategies (framework-level ECC handling)
