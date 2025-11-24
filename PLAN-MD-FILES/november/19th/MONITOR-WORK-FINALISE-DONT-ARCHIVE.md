# Wednesday Monitor Work - Finalise Session

**Date**: 2025-11-19
**Status**: CRITICAL HANG - 3 workers frozen, TUI unresponsive
**DO NOT ARCHIVE** - Active debugging in progress

---

## 📍 CURRENT POSITION

### What We Accomplished Today

**✅ Phase 2A: Terminal State Memory Dicts**
- Added `_terminal_failures` and `_terminal_successes` dicts (global cache)
- Prevents re-fetching logs for executions that reached terminal state
- Commit: `fa467f8`

**✅ Phase 2B: Error Extraction Function**
- Created `_fetch_and_extract_error(exec_name)` with ALL 20+ error patterns
- Added full inline documentation (source files, line numbers, pattern explanations)
- Distinguishes "our output" vs "external errors" (GCP, W&B, Python)
- Commit: `fa467f8`

**✅ Phase 2C: Smart Remembering Logic**
- Main loop checks `_terminal_failures`/`_terminal_successes` before fetching
- FAILED: Fetch once, remember forever
- FINISHED: Show "✓ Completed: N jobs" (extracts job count from logs)
- RUNNING: Show "Running..." (check again next 30s refresh)
- Commits: `fa467f8`, `2931c58`

**✅ Enhancements**
- Last-known-good cache for all 3 tables (resilient to API failures)
- Success message extraction (`_fetch_and_extract_success()`)
- Job count parsing from "Runs: N" in wrapper logs
- 1-second runtime ticker for all 3 tables (RUNNING/PENDING/QUEUED states)
- Wrapper integration documentation with full references
- Commits: `ee046d7`, `59a2f97`, `9c95cf5`, `a5fcb7c`

### What Remains

**Phase 2D: Testing**
- ❌ NOT TESTED - TUI hangs before we can verify functionality!
- Need to fix hang, then test the complete Phase 2 implementation

---

## 📋 PLAN FILES

### Primary Plan
**PHASE_2_REMEMBER_FAILED_ERRORS.md** (702 lines)
- Complete Phase 2 implementation plan
- All checklists for 2A, 2B, 2C marked complete ✅
- Phase 2D testing section waiting for hang fix

### Supporting Documentation
**PHASE_2_WRAPPER_INTEGRATION.md** (249 lines)
- Documents how monitoring code connects to arr-vertex-launcher wrapper
- Full data flow diagrams
- Pattern dependencies table
- Integration checklist

---

## 🚨 CRITICAL ISSUE: TUI HANG ANALYSIS

### Symptom Summary

**TUI launches but hangs during initial data fetch:**
- Spinners visible for all 5 tables
- No data populates
- No error messages shown
- TUI remains responsive to input but tables stay empty
- Terminal shows "— No executions —" message

### Log Timeline Analysis

```
Session started: 2025-11-19T00:45:43.687925

00:45:45.492 - 🎯 FETCH_ONLY: builds (accumulator will render later)
00:45:45.540 - 🎯 FETCH_ONLY: runner (accumulator will render later)
00:45:45.586 - 🎯 FETCH_ONLY: completed (accumulator will render later)
00:45:45.682 - 🎯 FETCH_ONLY: active (accumulator will render later)
00:45:45.775 - 🎯 FETCH_ONLY: vertex (accumulator will render later)

[Workers execute in parallel...]

00:45:59.714 - ✅ WORKER_COMPLETE: completed (14.14s) ← SUCCESS!
00:46:21.061 - ✅ WORKER_COMPLETE: runner (35.59s) ← TOO SLOW!

[THEN NOTHING - 3 workers never complete!]

00:46:00.116 - 🟦 DISPLAY_NEXT_ENTERED: index=0, batch_size=5
00:46:00.139 - ⏳ WAITING: builds not ready yet, checking again in 50ms...
00:46:06.682 - 🟦 DISPLAY_NEXT_ENTERED: index=0, batch_size=5
00:46:06.695 - ⏳ WAITING: builds not ready yet, checking again in 50ms...

[Polling timer stuck in waiting loop forever...]

00:46:19.441 - ⏭️  AUTO_REFRESH: No tables enabled, skipping!
00:46:21.069 - ✅ ACCUMULATOR_MARKED: runner complete (polling timer will display)
```

### The 3 Frozen Workers

**❌ BUILDS WORKER - NEVER COMPLETED**
- Started: 00:45:45.492
- Expected duration: ~5-10 seconds
- Status: Missing from completion logs
- Likely cause: Hanging in multi-region gcloud API call

**❌ ACTIVE WORKER - NEVER COMPLETED**
- Started: 00:45:45.682
- Expected duration: ~3-5 seconds
- Status: Missing from completion logs
- Likely cause: Hanging in W&B API call (api.runs())

**❌ VERTEX WORKER - NEVER COMPLETED**
- Started: 00:45:45.775
- Expected duration: ~5-10 seconds
- Status: Missing from completion logs
- Likely cause: Hanging in multi-region gcloud AI custom-jobs list

### The Slow Worker Issue

**⚠️ RUNNER WORKER - COMPLETED BUT TOO SLOW (35.59s)**
- Budget: 30 seconds
- Actual: 35.59 seconds (119% of budget!)
- Root cause: `_fetch_and_extract_success()` called for EVERY FINISHED execution

**Why runner is slow:**
```python
# For EACH FINISHED execution (could be 5-10 executions!)
if status == "FINISHED":
    if exec_name not in _terminal_successes:
        # Calls gcloud logging read for EACH execution!
        success_msg, jobs_count = _fetch_and_extract_success(exec_name)
        # Each gcloud call takes ~3-5 seconds
        # 5 FINISHED executions × 5 seconds = 25 seconds just for success logs!
```

### Display Polling Loop Stuck

The display timer tries to render tables progressively but gets stuck waiting:

```
00:46:00 - Check if builds ready → NO
00:46:06 - Check if builds ready → NO
[Stuck in infinite wait because builds worker never completes]
```

---

## 🔍 DETAILED LOG FORENSICS

### Worker Timing Log Analysis

```bash
$ cat training/logs/table_worker_timing.log
# Session started 2025-11-19T00:45:43.687925

2025-11-19T00:45:59.714309 ⏱️  COMPLETED: total=14.136s, fetch+update=14.107s, overhead=0.038s, budget=✅47% (avg=14.14s over last 1)
2025-11-19T00:46:21.061527 ⏱️  RUNNER: total=35.589s, fetch+update=35.546s, overhead=0.035s, budget=🚨119% (avg=35.59s over last 1)
```

**Key findings:**
1. Only 2 workers reported timing (completed, runner)
2. 3 workers missing (builds, active, vertex) - hung and never reported!
3. Runner worker budget exceeded (119% vs 100% limit)

### Auto-Refresh Log Deep Dive

**Last successful operations:**
```
00:45:59.613 - 📸 SNAPSHOT_TAKEN: completed (10 rows)
00:45:59.614 - 💾 DATA_STORED: completed (ready for display)
00:45:59.714 - ✅ WORKER_COMPLETE: completed (14.14s)
00:45:59.715 - ✅ ACCUMULATOR_MARKED: completed complete (polling timer will display)
00:45:59.715 - 🎯 STARTING_POLLING: First table complete, starting display timer

00:46:21.060 - 📸 SNAPSHOT_TAKEN: runner (0 rows)
00:46:21.061 - 💾 DATA_STORED: runner (ready for display)
00:46:21.061 - ✅ WORKER_COMPLETE: runner (35.59s)
00:46:21.069 - ✅ ACCUMULATOR_MARKED: runner complete (polling timer will display)
```

**What's missing:**
- No SNAPSHOT_TAKEN for builds, active, vertex
- No DATA_STORED for builds, active, vertex
- No WORKER_COMPLETE for builds, active, vertex
- No ACCUMULATOR_MARKED for builds, active, vertex

**Conclusion:** Workers hung during fetch phase, never reached snapshot/store/complete!

### 1-Second Timer Still Running

```
00:45:55.652 - ⏱️  1S_TIMER_TICK: Updating active durations...
00:45:55.653 - ✅ 1S_TIMER_COMPLETE: Updated 0 builds, 0 runners, 0 vertex
... [continues every second] ...
00:46:19.440 - ⏱️  1S_TIMER_TICK: Updating active durations...
00:46:19.440 - ✅ 1S_TIMER_COMPLETE: Updated 0 builds, 0 runners, 0 vertex
```

**Observation:** Timer is working fine, but no data to update (all tables empty!)

### Runner Fetch Debug Log

**Last 15 executions fetched successfully:**
```bash
$ tail -100 training/logs/runner_fetch_debug.log | grep "FETCHED.*executions"
2025-11-19T00:22:09.463850 🌍 FETCHED 15 total executions from 18 regions
2025-11-19T00:23:37.893614 🌍 FETCHED 0 total executions from 18 regions  ← Timeout!
2025-11-19T00:30:41.563449 🌍 FETCHED 15 total executions from 18 regions
2025-11-19T00:37:59.893655 🌍 FETCHED 0 total executions from 18 regions  ← Timeout!
2025-11-19T00:46:06.054814 🌍 FETCHED 15 total executions from 18 regions
```

**Key insight:** Runner worker DID fetch executions (15 found at 00:46:06), but took ~20 seconds just to get there!

---

## 🧬 ROOT CAUSE ANALYSIS

### Primary Issue: Missing Timeouts in API Calls

**Hypothesis:** Some gcloud/W&B API calls have NO timeout set, causing infinite hangs!

**Evidence:**
1. 3 workers hung indefinitely (no timeout → no completion)
2. No exceptions logged (hang, not error)
3. Workers stuck in external API calls

**Suspected locations:**

**1. Builds worker (multi-region gcloud):**
```python
# File: training/cli/monitor/core.py
# Function: _list_recent_cloud_builds() or similar

# Potential missing timeout:
result = run_gcloud_with_retry(
    ["gcloud", "builds", "list", ...],
    max_retries=3,
    timeout=60,  # ← This is set, so NOT the issue!
    operation_name="list Cloud Builds"
)
```

**2. Active worker (W&B API):**
```python
# File: training/cli/shared/wandb_helper.py
# Function: get_active_runs()

# Uses ThreadPoolExecutor timeout wrapper:
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(self._get_active_runs_impl)
    try:
        return future.result(timeout=10)  # ← This is set!
```

**3. Vertex worker (multi-region gcloud AI):**
```python
# File: training/cli/monitor/core.py
# Function: _fetch_vertex_ai_jobs()

result = run_gcloud_with_retry(
    ["gcloud", "ai", "custom-jobs", "list", ...],
    max_retries=3,
    timeout=60,  # ← This is set!
    operation_name="list Vertex AI custom jobs"
)
```

**Wait... all these HAVE timeouts!**

So why are they hanging?

### Alternative Hypothesis: ThreadPoolExecutor Deadlock

**The staggered refresh uses threads:**
```python
def _start_staggered_refresh(self):
    # Start 5 workers in parallel
    self.run_worker(self._fetch_and_update_builds_table, exclusive=False)
    self.run_worker(self._fetch_and_update_runner_table, exclusive=False)
    self.run_worker(self._fetch_and_update_completed_table, exclusive=False)
    self.run_worker(self._fetch_and_update_active_table, exclusive=False)
    self.run_worker(self._fetch_and_update_vertex_table, exclusive=False)
```

**Potential deadlock scenarios:**

**Scenario A: Lock contention**
- 5 workers try to access shared state (self.row_data, etc.)
- Some workers acquire lock and never release
- Other workers wait forever

**Scenario B: Nested ThreadPoolExecutor**
- Worker A uses ThreadPoolExecutor internally (e.g., multi-region fetch)
- Worker B does the same
- Python thread pool exhausted
- Deadlock waiting for available threads

**Scenario C: Exception swallowed silently**
- Worker crashes with exception
- Exception handler swallows it silently
- Worker marked as "still running" but actually dead

### Secondary Issue: Runner Performance

**The `_fetch_and_extract_success()` is too slow:**

```python
def _fetch_and_extract_success(exec_name: str) -> tuple[str, int]:
    # Calls gcloud logging read (3-5 seconds per call!)
    log_result = run_gcloud_with_retry(
        ["gcloud", "logging", "read", ...],
        max_retries=1,
        timeout=10,
        operation_name="fetch Cloud Run job logs"
    )
    # ... parse logs ...
```

**If 5 FINISHED executions:**
- 5 executions × 5 seconds/call = 25 seconds
- Plus runner metadata fetch: ~10 seconds
- **Total: 35 seconds** (matches observed 35.59s!)

**Solution:** Parallelize log fetches OR cache more aggressively!

---

## 🔧 PROPOSED FIXES

### Fix 1: Add Comprehensive Timeout Logging

**Goal:** Identify WHERE the hang occurs

**Implementation:**
```python
# Add to each worker function
import datetime

def _fetch_and_update_builds_table(self):
    start = datetime.datetime.now()
    log_file = Path(__file__).parent.parent.parent / "logs" / "worker_debug.log"

    with open(log_file, "a") as f:
        f.write(f"{start.isoformat()} 🚀 BUILDS_WORKER_START\n")

    try:
        # Fetch builds...
        with open(log_file, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} 📡 BUILDS_FETCH_START\n")

        builds = _list_recent_cloud_builds(...)

        with open(log_file, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} ✅ BUILDS_FETCH_DONE: {len(builds)} builds\n")

        # Update table...

    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} ❌ BUILDS_ERROR: {e}\n")
        raise
    finally:
        duration = (datetime.datetime.now() - start).total_seconds()
        with open(log_file, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} 🏁 BUILDS_WORKER_END: {duration}s\n")
```

**This will show us EXACTLY where each worker hangs!**

### Fix 2: Parallelize Success Log Fetches

**Goal:** Speed up runner worker (reduce from 35s to ~10s)

**Implementation:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _fetch_runner_executions_all_regions(...):
    # ... existing code ...

    # Find all FINISHED executions needing log fetch
    finished_to_fetch = []
    for exec in top_execs:
        if exec['status'] == "FINISHED" and exec['name'] not in _terminal_successes:
            finished_to_fetch.append(exec['name'])

    # Fetch ALL logs in parallel!
    if finished_to_fetch:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(_fetch_and_extract_success, name): name
                for name in finished_to_fetch
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    success_msg, jobs_count = future.result(timeout=15)
                    _terminal_successes[name] = (success_msg, jobs_count)
                except Exception:
                    _terminal_successes[name] = ("✓ Completed", 0)

    # Now populate exec['error'] from cache
    for exec in top_execs:
        if exec['status'] == "FINISHED":
            exec['error'], exec['jobs_run'] = _terminal_successes.get(
                exec['name'],
                ("✓ Completed", 0)
            )
```

**Result:** 5 log fetches in parallel instead of sequential = ~5-7 seconds instead of 25 seconds!

### Fix 3: Add Worker Watchdog Timer

**Goal:** Detect hung workers and fail gracefully

**Implementation:**
```python
def _start_staggered_refresh(self):
    # Track worker start times
    self.worker_start_times = {}

    def run_with_watchdog(worker_fn, name):
        self.worker_start_times[name] = datetime.datetime.now()
        try:
            worker_fn()
        except Exception as e:
            log(f"Worker {name} failed: {e}")
        finally:
            elapsed = (datetime.datetime.now() - self.worker_start_times[name]).total_seconds()
            log(f"Worker {name} finished in {elapsed}s")

    # Start workers with watchdog
    self.run_worker(lambda: run_with_watchdog(self._fetch_and_update_builds_table, "builds"))
    # ... etc ...

    # Start watchdog timer
    def check_workers():
        now = datetime.datetime.now()
        for name, start_time in self.worker_start_times.items():
            elapsed = (now - start_time).total_seconds()
            if elapsed > 60:  # 60 second timeout
                log(f"🚨 WATCHDOG: Worker {name} hung for {elapsed}s!")
                # Mark as failed, don't wait for it
                self.accumulator_status[name] = "failed"

    self.set_interval(5.0, check_workers)  # Check every 5 seconds
```

### Fix 4: Increase Logging in run_gcloud_with_retry

**Goal:** See if gcloud commands are actually hanging

**Check if we have verbose logging enabled:**
```python
# File: training/cli/shared/api_helpers.py
# Function: run_gcloud_with_retry()

# Add logging before/after subprocess.run:
log(f"🚀 GCLOUD_START: {' '.join(cmd[:5])}")
result = subprocess.run(...)
log(f"✅ GCLOUD_DONE: returncode={result.returncode}")
```

---

## 📊 NEXT STEPS

### Immediate Actions (Now)

1. **Kill the hung TUI process** (Ctrl+C or kill PID)
2. **Add worker debug logging** (Fix 1 above)
3. **Re-run TUI** and capture worker_debug.log
4. **Analyze where workers hang** (builds, active, vertex)

### Short-term Fixes (Today)

1. **Parallelize success log fetches** (Fix 2) - Speed up runner
2. **Add watchdog timer** (Fix 3) - Fail gracefully on hangs
3. **Test with debug logging** - Verify fixes work

### Medium-term Improvements (This Week)

1. **Increase gcloud/W&B timeout logging** - Better visibility
2. **Add progress indicators** - Show what's happening during fetch
3. **Implement cache warming** - Pre-fetch on app startup
4. **Add retry logic** - If worker fails, retry once

### Testing Plan

**Test 1: Worker Debug Logging**
- Add logging to all 5 workers
- Run TUI
- Check worker_debug.log for hang location
- **Expected:** See exactly which API call hangs

**Test 2: Parallel Success Fetches**
- Implement Fix 2
- Run TUI
- Check runner worker timing
- **Expected:** Runner completes in <15 seconds (vs 35s before)

**Test 3: Watchdog Timer**
- Implement Fix 3
- Simulate hang (add sleep(120) in builds worker)
- Run TUI
- **Expected:** Watchdog detects hang, TUI shows error instead of freezing

---

## 🎯 CRITICAL QUESTIONS

### Q1: Why did 3 workers hang but 2 complete?

**Hypothesis:** Completed/runner workers finished BEFORE the hang trigger occurred

**Test:** Check if all 5 workers hang consistently, or if it's random

### Q2: Are the timeouts actually being enforced?

**Check:**
```bash
# See if subprocess.run with timeout actually works
cd training/cli/shared
grep -A 10 "def run_gcloud_with_retry" api_helpers.py | grep timeout
```

### Q3: Is this a new issue or existing bug?

**Check git history:**
```bash
git log --oneline --since="1 week ago" -- training/cli/monitor/
```

**Look for recent changes that might have introduced the hang**

### Q4: Does the hang happen EVERY time?

**Test:** Run TUI 3 times, see if:
- Always hangs on same 3 workers
- Hangs on different workers each time
- Sometimes works fine

---

## 📝 APPENDIX: Complete File List

### Plan Files
- `PHASE_2_REMEMBER_FAILED_ERRORS.md` (702 lines)
- `PHASE_2_WRAPPER_INTEGRATION.md` (249 lines)

### Log Files (Current Session)
- `training/logs/auto_refresh.log` (Session: 00:45:43)
- `training/logs/table_worker_timing.log` (2 workers reported)
- `training/logs/runner_fetch_debug.log` (15 executions fetched)
- `training/logs/spinner_timing.log`
- `training/logs/performance.json`

### Code Files Modified Today
- `training/cli/monitor/core.py` (multiple commits)
  - Added _terminal_failures/_terminal_successes dicts
  - Added _fetch_and_extract_success() function
  - Added _fetch_and_extract_error() function
  - Added smart remembering logic
  - Added last-known-good caches
- `training/cli/monitor/screen.py` (1-second ticker fix)

### Recent Commits
```
9c95cf5 - Fix 1s timer: Update Runtime (was Lifetime) + Add Vertex AI ticking
59a2f97 - Fix builds exception handler: Return last-known-good on error
ee046d7 - Add last-known-good cache: Prevent 'No executions' flicker on API failures
a5fcb7c - Document wrapper-monitoring integration with detailed references
2931c58 - Extract job count for FINISHED runners: '✓ Completed: 5 jobs'
00a3e3c - Fix FINISHED display: Show '✓ Completed' in green (not red ❌)
fa467f8 - Implement Phase 2C: Smart log fetching with terminal state memory
```

---

## 🚨 STATUS: BLOCKED ON HANG

**Cannot proceed with Phase 2D testing until hang is resolved!**

**Priority:** Debug and fix the 3 frozen workers (builds, active, vertex)

**End of Analysis** - Ready for investigation and fixes!
