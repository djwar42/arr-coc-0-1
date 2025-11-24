# Universal Table Refresh System & Log Standardization

**Date:** 2025-11-18
**Components:** Monitor TUI, Cloud Run Wrapper, Log Parsing
**Impact:** 65% code reduction + guaranteed cleanup + simplified logging

---

## Table of Contents

1. [Universal Table Refresh System](#universal-table-refresh-system)
2. [Log Standardization: "Runs:" Pattern](#log-standardization-runs-pattern)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Benefits & Impact](#benefits--impact)

---

## Universal Table Refresh System

### The Problem (Before)

**21 separate functions with massive duplication (~1000+ lines):**

```python
# For EACH of 5 tables, we had:

# Lazy load (initial page load)
def _lazy_load_runner_executions():
    if self._refreshing_tables.get("runner"):
        return  # Skip if already refreshing
    self._refreshing_tables["runner"] = True
    self._start_spinner("runner-spinner")
    self.run_worker(self._lazy_load_runner_executions_worker)

def _lazy_load_runner_executions_worker():
    try:
        # Fetch data...
        # Update table...
    except Exception as e:
        self.notify(f"Error: {e}")
    finally:
        self._stop_spinner("runner-spinner")
        self._refreshing_tables["runner"] = False

# Refresh (auto-refresh timer)
def _refresh_runner_executions():
    if self._refreshing_tables.get("runner"):
        return  # Skip if already refreshing
    self._refreshing_tables["runner"] = True
    self._start_spinner("runner-spinner")
    self.run_worker(self._refresh_runner_executions_worker)

def _refresh_runner_executions_worker():
    try:
        # Fetch data...
        # Update table...
    except Exception as e:
        self.notify(f"Error: {e}")
    finally:
        self._stop_spinner("runner-spinner")
        self._refreshing_tables["runner"] = False

# × 5 tables = 20 functions + helpers = 21 total functions!
```

**Problems:**
- Massive code duplication (1000+ lines)
- Hard to maintain (change in 1 place = update 21 functions)
- Stuck table bugs (cleanup not guaranteed)
- Inconsistent error handling
- No centralized logging

---

### The Solution (After)

**7 clean functions with universal architecture (~350 lines):**

```python
# 1. Configuration (ONE source of truth)
TABLE_CONFIG = {
    "runner": {
        "name": "W&B Launch Runner",
        "spinner_id": "runner-spinner"
    },
    "builds": {
        "name": "Cloud Builds",
        "spinner_id": "builds-recent-spinner"
    },
    "vertex": {
        "name": "Vertex AI Jobs",
        "spinner_id": "vertex-spinner"
    },
    "active": {
        "name": "Active W&B Runs",
        "spinner_id": "active-spinner"
    },
    "completed": {
        "name": "Completed W&B Runs",
        "spinner_id": "completed-spinner"
    }
}

# 2. Universal launcher (replaces 10 launch functions)
def _universal_refresh_table(self, table_name: str, is_auto_refresh: bool = False):
    """Universal table refresh - works for ALL tables with guaranteed cleanup!

    Args:
        table_name: One of "runner", "builds", "vertex", "active", "completed"
        is_auto_refresh: True if called from auto-refresh timer (adds logging)
    """
    # Validate table name
    if table_name not in self.TABLE_CONFIG:
        self.notify(f"❌ Unknown table: {table_name}", severity="error")
        return

    config = self.TABLE_CONFIG[table_name]

    # Skip if already refreshing
    if table_name in self._refreshing_tables:
        # Auto-refresh skip logging
        if is_auto_refresh:
            log_skip(table_name)
        return

    # Mark as refreshing + record start time
    self._refreshing_tables.add(table_name)
    self._refresh_start_times[table_name] = time.time()

    # Start spinner
    self._start_spinner(config["spinner_id"])

    # Log launch (for auto-refresh debugging)
    if is_auto_refresh:
        log_launch(table_name)

    # Launch worker with table-specific logic
    self.run_worker(
        lambda: self._universal_table_worker(table_name, config),
        exclusive=True,
        name=f"refresh_{table_name}",
        thread=True
    )

# 3. Universal worker (replaces 10 worker functions)
def _universal_table_worker(self, table_name: str, config: dict):
    """Universal worker - handles ANY table with automatic cleanup guarantee!

    This is the CRITICAL function that prevents stuck tables!
    The finally block ALWAYS runs, ensuring cleanup happens even if worker crashes.
    """
    log_worker_entry(table_name)
    start_time = time.time()

    try:
        log_worker_start(table_name)

        # Dispatch to table-specific fetch/update logic
        if table_name == "runner":
            self._fetch_and_update_runner_table()
        elif table_name == "builds":
            self._fetch_and_update_builds_table()
        elif table_name == "vertex":
            self._fetch_and_update_vertex_table()
        elif table_name == "active":
            self._fetch_and_update_active_runs_table()
        elif table_name == "completed":
            self._fetch_and_update_completed_runs_table()

        elapsed = time.time() - start_time
        log_worker_complete(table_name, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        log_worker_failed(table_name, elapsed, e)
        self.notify(f"❌ {config['name']} refresh failed: {str(e)}", severity="error")

    finally:
        # 🎯 GUARANTEED CLEANUP - ALWAYS runs, no matter what!
        self._stop_spinner(config["spinner_id"])
        self._refreshing_tables.discard(table_name)
        if table_name in self._refresh_start_times:
            del self._refresh_start_times[table_name]

        log_cleanup_done(table_name)

# 4-8. Five simplified fetch/update functions (data-only logic)
def _fetch_and_update_runner_table(self):
    """Fetch runner executions data and update table UI"""
    # Just the data logic - no workers, no cleanup, no logging!
    runner_execs = _list_runner_executions(...)
    runner_table.clear()
    for exec in runner_execs:
        runner_table.add_row(...)

def _fetch_and_update_builds_table(self):
    """Fetch cloud builds data and update table UI"""
    builds = _list_cloud_builds_recent(...)
    builds_table.clear()
    for build in builds:
        builds_table.add_row(...)

# ... (3 more similar functions)
```

**Usage Examples:**

```python
# Page load (initial data fetch)
def on_page_load(self):
    # BATCH 1 (immediate): First 2 tables
    self._universal_refresh_table("builds")
    self._universal_refresh_table("runner")

    # BATCH 2 (3s delay): Remaining tables
    def load_remaining_tables():
        self._universal_refresh_table("vertex")
        self._universal_refresh_table("active")
        self._universal_refresh_table("completed")

    self.set_timer(3.0, load_remaining_tables)

# Auto-refresh timers (every 60s)
def _start_staggered_refresh(self):
    interval = AUTO_REFRESH_INTERVAL  # 60s

    # All tables use SAME universal system!
    if self.refresh_enabled["runner"]:
        timer = self.set_interval(
            interval,
            lambda: self._universal_refresh_table("runner", is_auto_refresh=True)
        )
        self.refresh_timers.append(timer)

    if self.refresh_enabled["vertex"]:
        timer = self.set_interval(
            interval,
            lambda: self._universal_refresh_table("vertex", is_auto_refresh=True)
        )
        self.refresh_timers.append(timer)

    # ... (repeat for all enabled tables)
```

---

### Key Architecture Principles

#### 1. Single Responsibility
- **Launcher** (`_universal_refresh_table`): Validation, skip logic, spinner start, worker dispatch
- **Worker** (`_universal_table_worker`): Execution, error handling, guaranteed cleanup
- **Fetch functions** (`_fetch_and_update_*`): Data-only logic, no infrastructure

#### 2. Guaranteed Cleanup (CRITICAL!)
```python
finally:
    # 🎯 ALWAYS runs - even if exception thrown or worker killed!
    self._stop_spinner(config["spinner_id"])
    self._refreshing_tables.discard(table_name)
    del self._refresh_start_times[table_name]
```

**Why this matters:**
- No more stuck spinners
- No more permanently "refreshing" tables
- No more leaked state

#### 3. Comprehensive Logging
```python
# Auto-refresh debug log (training/logs/auto_refresh.log)
2025-11-18T02:15:30 🚀 LAUNCHING_WORKER: runner
2025-11-18T02:15:30 ✓ WORKER_QUEUED: runner
2025-11-18T02:15:30 🔥 FUNCTION_CALLED: _universal_table_worker(runner) ENTRY
2025-11-18T02:15:30 ▶️  WORKER_START: runner
2025-11-18T02:15:32 ✅ WORKER_COMPLETE: runner (2.1s)
2025-11-18T02:15:32 🧹 CLEANUP_DONE: runner
```

**Log levels:**
- 🚀 LAUNCHING_WORKER: Worker dispatch queued
- ✓ WORKER_QUEUED: Worker accepted by Textual
- 🔥 FUNCTION_CALLED: Worker function entry (proves function was called)
- ▶️ WORKER_START: Data fetch begins
- ✅ WORKER_COMPLETE: Data fetch succeeded
- ❌ WORKER_FAILED: Data fetch failed (with error)
- 🧹 CLEANUP_DONE: Cleanup completed (ALWAYS logged!)
- ⏭️ SKIP: Refresh skipped (already running)

---

## Log Standardization: "Runs:" Pattern

### The Problem (Before)

**Multiple confusing patterns in logs:**

```bash
# Wrapper outputs 3 different patterns:
echo "✅ Job submitted to Vertex AI! (Total jobs: 3)"
echo "   • Jobs processed: 3"
echo "[08:22:36] Runner alive: ... Jobs run: 3"

# Monitor searches for all 3:
if 'Jobs processed:' in line or 'Jobs run:' in line or 'Total jobs:' in line:
    # Extract number...

# Monitor THEN replaces text:
error_msg = error_msg.replace('Jobs run:', 'Runs:')
```

**Problems:**
- 3 different patterns to search for
- Post-processing replacements
- Confusing for debugging
- We control the logs - why the confusion?

---

### The Solution (After)

**ONE simple pattern everywhere: `"Runs: N"`**

#### Wrapper Outputs (training/images/arr-vertex-launcher/entrypoint-wrapper.sh)

```bash
# Startup (line 36)
echo "📊 Runs: 0"

# Job submission success (line 111)
echo "✅ Job submitted to Vertex AI! (Runs: $JOBS_RUN)"

# Idle timeout exit (line 128)
echo "   • Runs: $JOBS_RUN"

# Periodic status (line 139, every 5 minutes)
echo "[$(date '+%H:%M:%S')] Runner alive: ${LIFETIME}s lifetime, ${IDLE_TIME}s idle, Runs: $JOBS_RUN"

# Final stats on exit (line 59)
echo "   • Runs: $JOBS_RUN"
```

#### Monitor Parses (training/cli/monitor/core.py)

```python
# We output "Runs: N" in ALL locations (wrapper controls output!)
# NOTE: gcloud logs are in DESCENDING order (newest first)
# So we take the FIRST match = NEWEST value!

for line in log_lines:
    # Extract jobs count (ONE pattern - we control the logs!)
    if 'Runs:' in line:
        # Extract number: "Runs: 3" → 3
        try:
            parts = line.split(':')
            if len(parts) >= 2:
                number_part = parts[-1].strip()
                jobs_run = int(number_part.split()[0])
                break  # Found it!
        except Exception:
            pass

# Post-process: "alive" → "completed" for FINISHED/FAILED
if error_msg and 'Runner alive:' in error_msg:
    if status_str in ['FINISHED', 'FAILED']:
        error_msg = error_msg.replace('Runner alive:', 'Runner completed:')

# NOTE: No "Jobs run:" → "Runs:" replacement needed!
# Wrapper outputs "Runs:" directly (entrypoint-wrapper.sh line 139)
```

#### Expected Log Output

**RUNNING runner (periodic status every 5 min):**
```
[08:17:45] Runner alive: 1204s lifetime, 360s idle, Runs: 2
```

**FINISHED runner (final stats):**
```
📊 Final stats:
   • Runs: 3
   • Runner lifetime: 55m 57s
   • Idle time: 14m 20s
```

**Monitor TUI displays:**
```
◈ W&B LAUNCH AGENT (Cloud Run Launches - All 18 Regions)
   Queue             Region          Status      Runs    Runtime    Created               Note
   vertex-ai-queue   europe-west2    ✓ FINISHED  3       55m57s     Nov 18, 7:42:27 AM    [08:22:36] Runner completed: 2404s lifetime, 860s idle, Runs: 3
```

---

### End-to-End Control

**We own the entire pipeline:**

```
┌──────────────────────────────────────────────────────────────┐
│ 1. WRAPPER OUTPUTS (entrypoint-wrapper.sh)                   │
│    → echo "Runs: $JOBS_RUN"                                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. CLOUD RUN LOGS (Cloud Logging)                            │
│    → Stores: "[08:22:36] Runner alive: ... Runs: 3"         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. MONITOR FETCHES (gcloud logs read)                        │
│    → Searches for: if 'Runs:' in line                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. MONITOR DISPLAYS (TUI table)                              │
│    → Shows: "Runs: 3" column                                │
└──────────────────────────────────────────────────────────────┘
```

**No confusion, no multiple patterns, no post-processing!**

---

## Architecture Deep Dive

### Universal Refresh Flow

```
User loads page or timer fires
    ↓
_universal_refresh_table("runner")
    ├─ Validate table_name in TABLE_CONFIG
    ├─ Check if already refreshing → SKIP if true
    ├─ Mark as refreshing (add to set)
    ├─ Record start time
    ├─ Start spinner
    ├─ Log launch (if auto-refresh)
    └─ Dispatch worker
         ↓
_universal_table_worker("runner", config)
    ├─ Log worker entry
    ├─ try:
    │   ├─ Log worker start
    │   ├─ Dispatch to _fetch_and_update_runner_table()
    │   │    ├─ Fetch data from GCP APIs
    │   │    ├─ Clear table
    │   │    ├─ Add rows
    │   │    └─ Return
    │   ├─ Calculate elapsed time
    │   └─ Log worker complete
    ├─ except Exception as e:
    │   ├─ Calculate elapsed time
    │   ├─ Log worker failed
    │   └─ Show error notification
    └─ finally:  ← 🎯 CRITICAL! ALWAYS runs!
        ├─ Stop spinner
        ├─ Remove from refreshing set
        ├─ Delete start time
        └─ Log cleanup done
```

### State Management

**Before (problematic):**
```python
# Each table had separate state
self._refreshing_runner = False
self._refreshing_builds = False
self._refreshing_vertex = False
self._refreshing_active = False
self._refreshing_completed = False

# Cleanup was scattered across 10 different finally blocks
```

**After (clean):**
```python
# Unified state tracking
self._refreshing_tables = set()  # {"runner", "vertex"}
self._refresh_start_times = {}   # {"runner": 1700000000.0}

# Cleanup is GUARANTEED by 1 finally block in universal worker
```

### Skip Logic (Auto-Refresh)

```python
# Auto-refresh fires every 60s
self.set_interval(60, lambda: self._universal_refresh_table("runner", is_auto_refresh=True))

# If runner is still refreshing from previous cycle:
if table_name in self._refreshing_tables:
    elapsed = time.time() - self._refresh_start_times.get(table_name, 0)
    if is_auto_refresh:
        # Log skip with elapsed time
        log_skip(table_name, elapsed)
    return  # Don't start new refresh

# Example log output:
# 2025-11-18T02:16:30 ⏭️  SKIP: runner (already running for 12.3s)
```

---

## Benefits & Impact

### Code Reduction

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Total functions | 21 | 7 | 67% fewer |
| Lines of code | ~1000 | ~350 | 65% reduction |
| Duplicate cleanup logic | 10 blocks | 1 block | 90% reduction |
| Configuration sources | Scattered | 1 dict | Centralized |

### Maintainability Improvements

**Before:** Change spinner behavior
```diff
# Need to update 10 different finally blocks!
- self._stop_spinner("runner-spinner")
- self._stop_spinner("builds-recent-spinner")
- self._stop_spinner("vertex-spinner")
- ... (7 more times)
```

**After:** Change spinner behavior
```diff
# Update 1 finally block!
- self._stop_spinner(config["spinner_id"])
```

### Reliability Improvements

**No more stuck tables:**
- `finally` block ALWAYS runs (even on crash)
- No leaked "refreshing" state
- No orphaned spinners
- Guaranteed cleanup

**Better debugging:**
- Comprehensive logging (7 log levels)
- Elapsed time tracking
- Auto-refresh skip detection
- Worker entry/exit tracking

### Performance Improvements

**Skip unnecessary refreshes:**
- Auto-refresh detects already-running workers
- Logs skip events for debugging
- Prevents queue buildup

**Parallel execution:**
- All 5 tables can refresh simultaneously
- Workers run in threads (non-blocking)
- Spinners show active state

---

## Testing Checklist

### Universal Refresh System

- [ ] Page load: All 5 tables load data successfully
- [ ] Auto-refresh: Tables refresh every 60s
- [ ] Skip logic: Second refresh skips if first still running
- [ ] Cleanup: Spinners always stop (even on error)
- [ ] Error handling: Failed refresh shows error notification
- [ ] Logging: auto_refresh.log shows complete flow
- [ ] State management: No leaked "refreshing" state

### Log Standardization

- [ ] Wrapper startup: Shows "📊 Runs: 0"
- [ ] Job submission: Shows "Runs: N" (incremented)
- [ ] Periodic status: Shows "[HH:MM:SS] Runner alive: ... Runs: N"
- [ ] Final stats: Shows "Runs: N" on exit
- [ ] Monitor parsing: Extracts correct run count
- [ ] Monitor display: Shows "Runs: N" in table
- [ ] FINISHED status: Shows "Runner completed: ... Runs: N"

---

## Migration Guide

### If Adding New Table Type

**Before (would need 4 new functions):**
```python
def _lazy_load_new_table(): ...
def _lazy_load_new_table_worker(): ...
def _refresh_new_table(): ...
def _refresh_new_table_worker(): ...
```

**After (add 2 entries):**

```python
# 1. Add to TABLE_CONFIG
TABLE_CONFIG = {
    # ... existing tables ...
    "new_table": {
        "name": "New Table Display Name",
        "spinner_id": "new-table-spinner"
    }
}

# 2. Add fetch function
def _fetch_and_update_new_table(self):
    """Fetch new table data and update UI"""
    data = fetch_new_table_data()
    table = self.query_one("#new-table", DataTable)
    table.clear()
    for item in data:
        table.add_row(...)

# 3. Update universal worker dispatch
def _universal_table_worker(self, table_name: str, config: dict):
    # ... existing try/except/finally ...
    if table_name == "new_table":
        self._fetch_and_update_new_table()

# 4. Use universal system
self._universal_refresh_table("new_table")
```

### If Changing Log Pattern

**Update wrapper output:**
```bash
# training/images/arr-vertex-launcher/entrypoint-wrapper.sh
echo "NewPattern: $VALUE"
```

**Update monitor parsing:**
```python
# training/cli/monitor/core.py
if 'NewPattern:' in line:
    value = extract_value(line)
```

**That's it! End-to-end control.**

---

## Commit History

### Universal Refresh System
- `241757c` ADD: Universal table refresh system (2 functions replace 21!)
- `bafd1a2` ADD: 5 simplified fetch/update functions - clean data logic only
- `e8b6e3d` UPDATE: All calls now use universal refresh system

### Log Standardization
- `58696e2` FIX: Output 'Runs:' directly from wrapper - no replacement needed!
- `59f7652` SIMPLIFY: Standardize on 'Runs:' EVERYWHERE - one pattern only!
- `76ff86e` DOC: Update comment - search for 'Runs:' not old patterns
- `b652065` FIX: Change 'Jobs:' → 'Runs:' in CLI output for consistency
- `cc26968` FIX: Last 'Jobs run:' → 'Runs:' (startup message line 36)

---

## Future Improvements

### Potential Enhancements

1. **Table-specific refresh intervals**
   - Active runs: 30s
   - Completed runs: 120s
   - Builds: 60s

2. **Smart refresh pausing**
   - Pause auto-refresh when user is viewing detail popup
   - Resume when popup closes

3. **Error recovery**
   - Automatic retry on transient failures
   - Exponential backoff for repeated failures

4. **Performance monitoring**
   - Track average refresh times
   - Alert if refresh takes > 10s
   - Dashboard for refresh metrics

5. **Log pattern registry**
   - Centralized YAML config for all log patterns
   - Compile-time validation
   - Auto-generate parsing code

---

## Appendix: Full Code Reference

### TABLE_CONFIG Dictionary
```python
TABLE_CONFIG = {
    "runner": {
        "name": "W&B Launch Runner",
        "spinner_id": "runner-spinner"
    },
    "builds": {
        "name": "Cloud Builds",
        "spinner_id": "builds-recent-spinner"
    },
    "vertex": {
        "name": "Vertex AI Jobs",
        "spinner_id": "vertex-spinner"
    },
    "active": {
        "name": "Active W&B Runs",
        "spinner_id": "active-spinner"
    },
    "completed": {
        "name": "Completed W&B Runs",
        "spinner_id": "completed-spinner"
    }
}
```

### State Tracking
```python
# Instance variables
self._refreshing_tables = set()           # Currently refreshing tables
self._refresh_start_times = {}            # Start time for each refresh
self.refresh_timers = []                  # Active auto-refresh timers
self.refresh_enabled = {                  # Which tables have auto-refresh
    "runner": True,
    "vertex": True,
    "active_runs": True,
    "completed_runs": True,
    "recent_builds": True
}
```

### Auto-Refresh Timer Setup
```python
def _start_staggered_refresh(self):
    """Start auto-refresh timers for all enabled tables"""
    interval = AUTO_REFRESH_INTERVAL  # 60s

    # Runner executions
    if self.refresh_enabled["runner"]:
        timer = self.set_interval(
            interval,
            lambda: self._universal_refresh_table("runner", is_auto_refresh=True)
        )
        self.refresh_timers.append(timer)

    # Vertex AI jobs
    if self.refresh_enabled["vertex"]:
        timer = self.set_interval(
            interval,
            lambda: self._universal_refresh_table("vertex", is_auto_refresh=True)
        )
        self.refresh_timers.append(timer)

    # ... repeat for active, completed, builds
```

---

**End of Document**
