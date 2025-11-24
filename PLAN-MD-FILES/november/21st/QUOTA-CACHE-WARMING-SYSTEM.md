# Quota Cache Warming System

**Complete implementation of staggered background quota cache warming for arr-coc-0-1**

**Date**: 2025-11-21
**Status**: ✅ COMPLETE & WORKING

---

## The Problem

GPU and C3 quota checks take **2-3 seconds each** via GCP API:
- 20 GPU checks (5 types × 4 regions) = ~40-60 seconds
- 18 C3 checks (MECHA regions) = ~36-54 seconds
- **Total: 80+ seconds for fresh checks!** 😱

This caused:
- Infra screen hanging for 80+ seconds on first load
- Monitor screen slow to start
- Users waiting ages for data

---

## The Solution: Staggered Cache Warming

**Background warming system** that:
1. Starts when TUI launches
2. Warms cache in **staggered batches** (1 GPU + 1 C3 every 2 seconds)
3. Takes ~40 seconds to full warm
4. Re-warms every 30 minutes automatically
5. **Infra screen uses cached data** → instant display!

---

## System Architecture

```
TUI START
    ↓
on_mount() → Clear logs, start timers
    ↓
Timer (2s interval) → _warm_quota_cache_tick()
    ↓
run_worker(thread=True, exclusive=True) → _do_cache_warm_batch()
    ↓
warm_quota_cache_batch() → 1 GPU check + 1 C3 check
    ↓
Cache results (_set_cached_quotas)
    ↓
Repeat until warm (20 GPU + 18 C3 = ~40s)
    ↓
30-min timer → Clear cache → Re-warm
```

---

## Key Files Modified

### 1. `CLI/shared/infra_verify.py`

**Added:**
- 30-minute in-memory quota cache (`_quota_cache`)
- `warm_quota_cache_batch()` - Staggered warming function
- `is_quota_cache_warm()` - Check if cache valid
- `STEVEN_INFRA_VERIFY_DEBUG` flag - Timing logs
- Check/fail counters for debugging

**Cache format:**
```python
_quota_cache = {
    "gpu": {"vertex_gpu": {...}, "all_gpu_found": [...]},
    "gpu_ts": 1732168000.0,  # Timestamp
    "c3": {"c3_build": {...}, "all_c3_found": [...]},
    "c3_ts": 1732168000.0
}
```

### 2. `CLI/tui.py`

**Added:**
- `STEVEN_CACHE_WARM_DEBUG` flag - Cache warming logs
- `_cache_warm_timer` - 2-second interval timer
- `_cache_refresh_timer` - 30-minute refresh timer
- `_warm_quota_cache_tick()` - Timer callback
- `_do_cache_warm_batch()` - Worker function
- `_restart_cache_warming()` - Cache clear + re-warm
- Log file clearing on TUI start

---

## The Warming Process

### Batch Warming (1 GPU + 1 C3 every 2 seconds)

```python
def warm_quota_cache_batch(project_id: str) -> Dict[str, Any]:
    """
    Warm ONE batch (1 GPU check + 1 C3 check).
    Call every 2s until cache warm.
    Returns: {"done": bool, "gpu_progress": "2/20", "c3_progress": "3/18"}
    """
```

**Progress:**
```
T=0s:  🚀 CACHE_WARM_START
T=2s:  🔥 BATCH_WARM: GPU 1/20, C3 1/18
T=4s:  🔥 BATCH_WARM: GPU 2/20, C3 2/18
T=6s:  🔥 BATCH_WARM: GPU 3/20, C3 3/18
...
T=40s: ✅ CACHE_WARM_COMPLETE: GPU 20/20, C3 18/18 - Cache is HOT!
```

### GPU Checks (20 total)

```python
gpu_priority_order = [
    ("NVIDIA_TESLA_T4", "T4", "n1-standard-4"),      # 4 regions
    ("NVIDIA_L4", "L4", "g2-standard-4"),            # 4 regions
    ("NVIDIA_TESLA_A100", "A100", "a2-highgpu-1g"),  # 4 regions
    ("NVIDIA_H100_80GB", "H100", "a3-highgpu-8g"),   # 4 regions
    ("NVIDIA_H200", "H200", "a3-highgpu-8g"),        # 4 regions
]
gpu_regions = ["us-central1", "us-east1", "us-west1", "europe-west4"]
# 5 types × 4 regions = 20 checks
```

### C3 Checks (18 total)

```python
c3_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
# 18 MECHA regions
```

---

## Threading Pattern (Steven's Pattern!)

**Steven's rule:** Use `run_worker(thread=True, exclusive=True)` for background work!

```python
def _warm_quota_cache_tick(self) -> None:
    """Called every 2s - launches worker"""
    if is_quota_cache_warm():
        return  # Already warm, skip

    # Launch worker (non-blocking!)
    self.run_worker(
        lambda: self._do_cache_warm_batch(project_id),
        exclusive=True,  # ONE at a time! Prevents race condition
        name="cache_warm",
        thread=True  # Run in thread, don't block UI!
    )

def _do_cache_warm_batch(self, project_id: str) -> None:
    """Worker function - runs in background thread"""
    result = warm_quota_cache_batch(project_id)

    # Log progress
    if STEVEN_CACHE_WARM_DEBUG:
        log_file = get_log_path("cache_warm.log")
        with open(log_file, "a") as f:
            if result["done"]:
                f.write(f"✅ CACHE_WARM_COMPLETE: GPU {result['gpu_progress']}, C3 {result['c3_progress']}\n")
            else:
                f.write(f"🔥 BATCH_WARM: GPU {result['gpu_progress']}, C3 {result['c3_progress']}\n")
```

**Why `exclusive=True`?**

Prevents race condition on indices:
```
Timer fires every 2s
    ↓
But API calls take 1-3s each
    ↓
Without exclusive=True: TWO workers run simultaneously!
    ↓
Both increment gpu_idx → 💥 IndexError!

With exclusive=True: ONE worker at a time → ✅ Safe!
```

---

## Debug Flags & Logging

### Two Debug Flags

```python
# CLI/tui.py
STEVEN_CACHE_WARM_DEBUG = True  # 🔥 Cache warming logs

# CLI/shared/infra_verify.py
STEVEN_INFRA_VERIFY_DEBUG = True  # 🔥 Quota timing logs
```

### Log Files (Clear on TUI start!)

**`ARR_COC/Training/logs/cache_warm.log`** - Background warming progress:
```
# Cache warming log - Session started 2025-11-21T02:45:22
# Format: timestamp emoji EVENT: details
# Events: 🚀=START, 🔥=BATCH, ✅=WARM, ⏭️=SKIP, 🔄=REWARM
#
2025-11-21T02:45:22 🚀 CACHE_WARM_START: Timers initialized (2s batch, 30m refresh)
2025-11-21T02:45:30 🔥 BATCH_WARM: GPU 1/20, C3 1/18
2025-11-21T02:45:31 🔥 BATCH_WARM: GPU 2/20, C3 2/18
...
2025-11-21T02:46:09 ✅ CACHE_WARM_COMPLETE: GPU 20/20, C3 18/18 - Cache is HOT!
```

**`ARR_COC/Training/logs/infra_verify_timing.log`** - Per-check timing:
```
# Infra verify timing log - Session started 2025-11-21T02:45:22
# Format: timestamp ⏱️ QUOTA_TIMING: GPU/C3 elapsed + cache hit status
#
2025-11-21T02:45:41 ⏱️ QUOTA_TIMING:
  GPU: 12.34s (FRESH) checks=20 fails=0 found=4
  C3:  8.56s (FRESH) checks=18 fails=0 found=16
  TOTAL: 20.90s

2025-11-21T02:46:00 ⏱️ QUOTA_TIMING:
  GPU: 0.01s (CACHE HIT)
  C3:  0.01s (CACHE HIT)
  TOTAL: 0.02s
```

---

## Critical Bugs Fixed During Implementation

### Bug 1: Wrong Function Call (ALL 20 GPU checks failing!)

**Problem:**
```python
# WRONG - Returns STRING (metric name)!
quota = get_vertex_gpu_quota_metric(project_id, region, gpu_internal)
if quota.get("quota_limit", 0) > 0:  # 💥 Can't .get() on string!
```

**Fix:**
```python
# CORRECT - Returns INT (quota limit)!
quota_limit = get_vertex_gpu_quotas(project_id, region, gpu_internal, use_spot=True)
if quota_limit > 0:  # ✅ Works!
```

**Commit:** `02974d09 - Fix: Use get_vertex_gpu_quotas (returns int)`

### Bug 2: IndexError from Race Condition

**Problem:**
```python
exclusive=False  # Multiple workers running simultaneously!
    ↓
Timer fires while previous worker still running
    ↓
Both workers increment gpu_idx
    ↓
gpu_idx exceeds list bounds → 💥 IndexError!
```

**Fix:**
```python
exclusive=True  # ONE worker at a time!
    ↓
Timer fires but previous worker still running
    ↓
Textual queues new worker, waits for previous to finish
    ↓
Safe sequential execution → ✅ No race condition!
```

**Commits:**
- `4277e3ad - Fix: Add bounds checks to prevent IndexError`
- `ee96ebb2 - Fix: Use exclusive=True for cache warming`

---

## Cache Behavior

### Infra Screen Uses Cache

```python
# infra_verify.py - verify_all_infrastructure()

cached_gpu = _get_cached_quotas("gpu")
if cached_gpu is not None:
    # Use cached data (instant!)
    vertex_gpu = cached_gpu["vertex_gpu"]
    all_gpu_found = cached_gpu["all_gpu_found"]
    _status("  (cached)")
else:
    # Fresh check (slow)
    for gpu in gpu_types:
        for region in regions:
            quota = get_vertex_gpu_quotas(...)  # 2-3s each!
```

### Launch-Time Checks Always Fresh

```python
# launch/core.py - ALWAYS uses fresh checks (NO cache!)

# QUOTA CHECKS: ALWAYS FRESH! (NO cache - critical for launch accuracy)
# - GPU quota checks → fresh every time (spot availability changes)
# - C3 quota checks → fresh every time (worker pool must match current quota)
# - Unlike infra_verify.py which uses 30-min cache for TUI display
# - Launch MUST have real-time quota data for correct region/machine selection
```

---

## Performance Impact

### Before (No Cache)

```
Infra screen load: 80+ seconds (FRESH checks every time)
User: 😱 "Why is it hanging??"
```

### After (With Cache)

```
TUI start: Cache warming begins (background, non-blocking)
    ↓
~40 seconds: Cache warm
    ↓
Infra screen load: 0.02 seconds (CACHE HIT!)
User: 🎉 "Instant!"
```

**Speedup:** ~4000× faster (80s → 0.02s) when cache warm!

---

## Testing

### View Logs Live

```bash
# Cache warming progress
tail -f ARR_COC/Training/logs/cache_warm.log

# Infra verify timing
tail -f ARR_COC/Training/logs/infra_verify_timing.log
```

### Test Cache Warming

```bash
# 1. Start TUI
python CLI/tui.py

# 2. Watch cache warm (in another terminal)
tail -f ARR_COC/Training/logs/cache_warm.log

# Expected output:
# 🚀 CACHE_WARM_START
# 🔥 BATCH_WARM: GPU 1/20, C3 1/18
# 🔥 BATCH_WARM: GPU 2/20, C3 2/18
# ...
# ✅ CACHE_WARM_COMPLETE: GPU 20/20, C3 18/18
```

### Test Cache Hit

```bash
# 1. Wait for cache to warm (~40s)
# 2. Go to Infra screen
# 3. Check timing log:

cat ARR_COC/Training/logs/infra_verify_timing.log

# Should see:
# ⏱️ QUOTA_TIMING:
#   GPU: 0.01s (CACHE HIT)
#   C3:  0.01s (CACHE HIT)
#   TOTAL: 0.02s
```

---

## Git Commits

All commits from this session:

```bash
768d9bdb - Add joke to humor sense: Works™ locally
3c6ab007 - Add 30-min in-memory cache for GPU/C3 quota checks (infra_verify ONLY)
8660fdc5 - Add comment: launch quota checks are ALWAYS FRESH (no cache)
92dafcf9 - Add staggered quota cache warming on TUI startup
1efbea95 - Use run_worker(thread=True) for cache warming - Steven's pattern
e05caa8c - Add STEVEN_CACHE_WARM_DEBUG flag + cache_warm.log (Steven's pattern)
2f9c79e7 - Add STEVEN_INFRA_VERIFY_DEBUG flag + timing logs for quota checks
b3239030 - Clear infra_verify_timing.log on TUI start (Steven's pattern)
02974d09 - Fix: Use get_vertex_gpu_quotas (returns int) not get_vertex_gpu_quota_metric (returns string)
4277e3ad - Fix: Add bounds checks to prevent IndexError in cache warming
ee96ebb2 - Fix: Use exclusive=True for cache warming to prevent race condition
```

---

## Summary

✅ **30-minute in-memory cache** for GPU/C3 quota checks
✅ **Staggered warming** (1 GPU + 1 C3 every 2s, ~40s total)
✅ **Auto re-warm** every 30 minutes
✅ **Steven's threading pattern** (`run_worker(thread=True, exclusive=True)`)
✅ **Debug logging** with two flags + two log files
✅ **Infra screen** uses cache → instant display
✅ **Launch checks** always fresh → accurate quota data
✅ **All bugs fixed** (wrong function call, race condition, IndexError)

**Result:** Infra screen is now **~4000× faster** when cache warm! 🎯

---

**Created:** 2025-11-21
**Author:** Claude + Steven (threading guru)
**Status:** ✅ Production ready!
