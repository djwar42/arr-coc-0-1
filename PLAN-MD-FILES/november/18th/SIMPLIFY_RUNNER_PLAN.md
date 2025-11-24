# 🔧 SIMPLIFY RUNNER TABLE - ACTION PLAN

**Goal**: Remove complex HOT/COLD adaptive system, make runner table fetch like the others (simple parallel), prepare for terminal state caching later.

---

## 📋 CURRENT STATE (Complex Adaptive)

**Runner table uses HOT/COLD adaptive log fetching:**

```python
def _fetch_runner_executions_all_regions():
    """~500 lines of complex adaptive logic"""

    # Step 1: Fetch 90 basic metadata (18 regions × 5 each)
    # Step 2: Classify all 90 as HOT/COLD
    # Step 3: Build target set (25 HOT + 2 COLD rotation)
    # Step 4: Fetch logs for 27 executions (parallel)
    # Step 5: Parse logs, extract error messages
    # Step 6: Format with log details
    # Step 7: Sort by created_at, return top 5
```

**Problems:**
- 500 lines of code for ONE table!
- Complex HOT/COLD classification
- Rotation index state management
- Fetches logs for 27 but displays 5
- Hard to maintain, hard to understand

---

## 🎯 TARGET STATE (Simple Like Others)

**Make runner table simple like builds/vertex:**

```python
def _fetch_runner_executions_all_regions():
    """~50 lines of simple parallel fetch (NO LOGS!)"""

    # Step 1: Fetch 90 basic metadata (18 regions × 5 each)
    # Step 2: Format with "Fetching logs..." placeholder
    # Step 3: Sort by created_at, return top 5
    # Done! No log fetching, no adaptive logic!
```

**Benefits:**
- Simple! Same pattern as builds/vertex
- ~50 lines instead of 500
- No complex state management
- Fast! (2s for metadata vs 13.5s with logs)
- Easy to understand and maintain

---

## 🚀 IMPLEMENTATION STEPS

### Step 1: Comment Out Log Fetching Code
- [ ] Find log fetching section in `_fetch_runner_executions_all_regions()`
- [ ] Comment out entire HOT/COLD classification (lines ~633-652)
- [ ] Comment out target set building (lines ~654-663)
- [ ] Comment out parallel log fetch (lines ~668-710)
- [ ] **DO NOT DELETE!** Keep for Phase 2 (terminal state caching)
- [ ] Add comment: `# TODO Phase 2: Add back with terminal state caching`

### Step 2: Simplify Function to Basic Fetch
- [ ] Keep region parallel fetch (works great!)
- [ ] Remove HOT/COLD logic
- [ ] Remove rotation index management
- [ ] Remove `_adaptive_exec_state` usage
- [ ] Format executions with basic metadata only

**New simplified structure:**
```python
def _fetch_runner_executions_all_regions(target_regions=None):
    """Fetch Cloud Run executions (simple parallel - NO LOGS!)"""

    regions = target_regions or ALL_MECHA_REGIONS

    def fetch_region(region_name):
        # Fetch executions from region (basic metadata only)
        result = run_gcloud_with_retry([
            "gcloud", "run", "jobs", "executions", "list",
            f"--job=vertex-ai-launcher",
            f"--region={region_name}",
            f"--limit=5",
            "--format=json"
        ], timeout=30, max_retries=1)

        # Format executions (NO LOG FETCHING!)
        formatted_execs = []
        for execution in result:
            formatted_execs.append({
                'name': execution['metadata']['name'].split('/')[-1],
                'region': region_name,
                'status': get_status(execution),  # RUNNING/FAILED/FINISHED
                'created_at': execution['metadata']['creationTimestamp'],
                'note': 'Fetching logs...',  # ← Placeholder!
            })

        return formatted_execs

    # Parallel fetch all regions
    all_execs = _parallel_fetch_regions(fetch_region, regions)

    # Sort and limit
    return _sort_and_limit_by_time(all_execs, LIMIT_RUNNER_EXECUTIONS)
```

### Step 3: Update Display Function
- [ ] Update `_update_runner_table()` or equivalent
- [ ] Expect `note: 'Fetching logs...'` for ALL executions
- [ ] Remove special handling for executions without logs
- [ ] All executions display "Fetching logs..." in Note column

**Display example:**
```
┌────────────────────────────────────────────────────────────┐
│ Name   Region   Status    Note                            │
├────────────────────────────────────────────────────────────┤
│ f4hfv  us-w2    FAILED    Fetching logs...                │
│ wf9fp  us-w2    FAILED    Fetching logs...                │
│ abc123 us-w2    FINISHED  Fetching logs...                │
│ def456 eu-w2    FINISHED  Fetching logs...                │
│ ghi789 asia     RUNNING   Fetching logs...                │
└────────────────────────────────────────────────────────────┘
```

### Step 4: Clean Up Module-Level State
- [ ] Check if `_adaptive_exec_state` is used elsewhere
- [ ] If ONLY used by runner table → Comment out
- [ ] Add comment: `# TODO Phase 2: Restore for terminal state caching`
- [ ] Keep code in place (don't delete!)

### Step 5: Test the Simplified System
- [ ] Run TUI: `python training/tui.py`
- [ ] Check runner table displays correctly
- [ ] Verify all 5 executions show "Fetching logs..."
- [ ] Verify status (RUNNING/FAILED/FINISHED) displays correctly
- [ ] Verify region displays correctly
- [ ] Verify timestamp sorting works (newest first)
- [ ] Check logs for any errors

**Expected behavior:**
- Runner table loads FAST! (~2s instead of ~13.5s)
- All executions show "Fetching logs..." in Note
- Status and region display correctly
- Sorted by created_at (newest first)

### Step 6: Update Comments and Docs
- [ ] Add comment at top of function:
  ```python
  """
  Fetch Cloud Run executions (simplified - NO LOGS!)

  Phase 1: Basic metadata only, display "Fetching logs..."
  Phase 2: Add terminal state caching for smart log fetching

  Original adaptive HOT/COLD code preserved below (commented out)
  for Phase 2 implementation.
  """
  ```
- [ ] Update function signature if needed
- [ ] Remove unused parameters (`status`, `region` if present)

### Step 7: Verify Consistency with Other Tables
- [ ] Compare with `_list_vertex_ai_jobs()` structure
- [ ] Compare with `_list_recent_cloud_builds()` structure
- [ ] Ensure same pattern:
  1. Parallel region fetch
  2. Format results
  3. Sort by created_at
  4. Return top N
- [ ] All 3 functions should look similar!

### Step 8: Commit Changes
- [ ] Clean up `__pycache__`
- [ ] Git add changes
- [ ] Commit message:
  ```
  🔧 SIMPLIFY RUNNER TABLE: Remove adaptive HOT/COLD system

  Phase 1: Make runner table simple like builds/vertex
  - Remove complex adaptive log fetching (500 → 50 lines!)
  - Display "Fetching logs..." for all executions
  - Keep original log code commented out (for Phase 2)
  - Fast! ~2s metadata fetch vs ~13.5s with logs

  Phase 2 prep:
  - Original HOT/COLD code preserved (commented)
  - Ready for terminal state caching implementation

  Benefits:
  - Simpler code (easier to maintain)
  - Faster refresh (2s vs 13.5s)
  - Consistent with other tables
  - Sets up for smarter caching in Phase 2

  🤖 Generated with Claude Code

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

---

## 📊 BEFORE vs AFTER

### BEFORE (Complex Adaptive)
```
Lines of code:         ~500
Fetch time:            ~13.5s (27 log fetches)
Complexity:            HIGH (HOT/COLD, rotation, state)
Display:               5 execs (some with logs, some without)
Maintenance:           HARD (complex logic)
```

### AFTER (Simple)
```
Lines of code:         ~50
Fetch time:            ~2s (metadata only)
Complexity:            LOW (same as builds/vertex)
Display:               5 execs (all show "Fetching logs...")
Maintenance:           EASY (simple parallel pattern)
```

**Savings: 450 lines removed, 11.5s faster, much simpler!**

---

## ⚠️ WHAT TO KEEP (Don't Delete!)

**Preserve these sections as comments:**

1. HOT/COLD classification logic (lines ~633-652)
   ```python
   # TODO Phase 2: Restore for terminal state caching
   # for execution in executions:
   #     exec_name = execution['metadata']['name'].split('/')[-1]
   #     status_obj = execution.get('status', {})
   #     for condition in status_obj.get('conditions', []):
   #         if condition.get('type') == 'Completed':
   #             ... (classification logic)
   ```

2. Target set building (lines ~654-663)
   ```python
   # TODO Phase 2: Restore for terminal state caching
   # target_executions_set = set(hot_execs)
   # if cold_execs:
   #     for i in range(min(2, len(cold_execs))):
   #         ... (rotation logic)
   ```

3. Parallel log fetch (lines ~668-710)
   ```python
   # TODO Phase 2: Restore for terminal state caching
   # def fetch_execution_logs(execution):
   #     name = execution['metadata']['name'].split('/')[-1]
   #     if name not in target_executions_set:
   #         return (name, None)
   #     log_result = run_gcloud_with_retry([
   #         "gcloud", "logging", "read",
   #         ... (log fetching logic)
   ```

4. Module-level state
   ```python
   # TODO Phase 2: Restore for terminal state caching
   # _adaptive_exec_state = {
   #     "hot_executions": set(),
   #     "cold_rotation_idx": 0,
   # }
   ```

**Why keep?** Phase 2 will add terminal state caching, which will reuse parts of this code!

---

## 🎯 SUCCESS CRITERIA

✅ Runner table displays 5 executions
✅ All show "Fetching logs..." in Note column
✅ Status (RUNNING/FAILED/FINISHED) displays correctly
✅ Region displays correctly
✅ Sorted by created_at (newest first)
✅ Refresh time ~2s (fast!)
✅ No errors in logs
✅ Function ~50 lines (simple!)
✅ Original log code preserved as comments
✅ Consistent with builds/vertex tables

---

## 📝 PHASE 2 DEEP DIVE: Terminal State Caching & Smart Log Fetching

**After Phase 1 is working, add intelligent log fetching based on discovered insights:**

---

### 🧠 KEY INSIGHT #1: Terminal State Detection

**Discovery**: Some log messages indicate we NEVER need to fetch again!

**Terminal FINISHED States** (success - cache forever):
```python
TERMINAL_SUCCESS_PATTERNS = [
    "Successfully submitted job to Vertex AI",
    "Vertex AI job created:",
    "Job submission complete",
    "Runner completed successfully",
]

# Example log:
# [2024-11-19 15:42:18] Successfully submitted job to Vertex AI: 12345
# [2024-11-19 15:42:19] Vertex AI job created: projects/.../jobs/12345
# [2024-11-19 15:42:20] Runner completed successfully (2404s, 5 jobs)

# ✅ CACHE THIS FOREVER! Status=FINISHED, won't change!
```

**Terminal FAILED States** (known failures - cache forever):
```python
TERMINAL_FAILURE_PATTERNS = [
    "OOM killed",                          # Out of memory - permanent
    "Image pull failed",                   # Image doesn't exist - permanent
    "Permission denied",                   # Auth error - permanent
    "Task failed: exit code",              # Task crashed - permanent
    "Timeout exceeded",                    # Job took too long - permanent
    "Resource exhausted: Quota exceeded",  # Quota limit - permanent
]

# Example log:
# [2024-11-19 14:30:15] Task failed: Container exited with code 137
# [2024-11-19 14:30:16] OOM killed (memory limit: 2048Mi)
# [2024-11-19 14:30:17] Execution failed

# ✅ CACHE THIS FOREVER! Status=FAILED with known error, won't change!
```

**Non-Terminal States** (need monitoring):
```python
# RUNNING - Always fetch (status changes continuously)
# FAILED with unknown error - Fetch once to analyze
# FINISHED without success message - Fetch once to verify
```

---

### 🧠 KEY INSIGHT #2: Only Fetch Logs for Top 5 Displayed

**Discovery**: We fetch logs for 27 executions but only display 5!

**Current Waste**:
```
Fetch metadata: 90 executions (from 18 regions)
Fetch logs:     27 executions (25 HOT + 2 COLD rotation)
Display:         5 executions (top 5 newest)

WASTED: 22 log fetches we never show! ❌
```

**Smart Strategy**:
```python
# Step 1: Fetch metadata for 90 executions (fast! ~2s)
all_execs = fetch_all_regions()  # 90 executions

# Step 2: Sort by created_at, get top 5
top_5 = sorted(all_execs, key=lambda x: x['created_at'], reverse=True)[:5]

# Step 3: ONLY fetch logs for these 5!
for exec in top_5:
    if exec not in terminal_cache:
        fetch_logs(exec)  # Only 5 fetches instead of 27!
    else:
        exec['note'] = terminal_cache[exec]['note']  # From cache!
```

**Savings**: 27 → 5 log fetches = **81% reduction!**

---

### 🧠 KEY INSIGHT #3: Cache Terminal States Forever

**Discovery**: Terminal state logs NEVER change - cache them!

**Cache Structure**:
```python
_terminal_log_cache = {
    # Execution name: Cached log data
    'f4hfv': {
        'status': 'FAILED',
        'note': '❌ OOM killed (memory limit: 2048Mi)',
        'terminal': True,          # ← Won't change!
        'cached_at': '2024-11-19 14:30:17',
        'log_lines': [...],        # Full logs
    },
    'wf9fp': {
        'status': 'FINISHED',
        'note': 'Runner: 2404s, 5 jobs submitted ✅',
        'terminal': True,          # ← Won't change!
        'cached_at': '2024-11-19 15:42:20',
        'log_lines': [...],
    },
}
```

**Cache Logic**:
```python
def fetch_logs_with_cache(exec_name, status):
    """Fetch logs with terminal state caching"""

    # Check cache first
    if exec_name in _terminal_log_cache:
        cached = _terminal_log_cache[exec_name]
        if cached['terminal']:
            return cached['note']  # Return cached! No API call!

    # Not cached - fetch logs
    log_lines = gcloud_logging_read(exec_name)

    # Check if terminal state
    is_terminal, note = detect_terminal_state(log_lines, status)

    if is_terminal:
        # Cache forever!
        _terminal_log_cache[exec_name] = {
            'status': status,
            'note': note,
            'terminal': True,
            'cached_at': datetime.now().isoformat(),
            'log_lines': log_lines,
        }

    return note
```

---

### 🧠 KEY INSIGHT #4: Smart Fetch Decision Tree

**Discovery**: Different executions need different fetch strategies!

```
For EACH of top 5 displayed executions:
                    │
    ┌───────────────▼────────────────┐
    │ Is it in terminal cache?       │
    └───────────────┬────────────────┘
                    │
        ┌───────────┴────────────┐
        │ YES                    │ NO
        ▼                        ▼
┌──────────────────┐    ┌─────────────────┐
│ Return cached    │    │ What's status?  │
│ note (FREE!)     │    └────────┬────────┘
└──────────────────┘             │
                     ┌───────────┴────────────┐
                     │                        │
            ┌────────▼─────────┐    ┌────────▼─────────┐
            │ RUNNING          │    │ FAILED/FINISHED  │
            └────────┬─────────┘    └────────┬─────────┘
                     │                       │
            ┌────────▼─────────┐    ┌────────▼─────────┐
            │ Always fetch     │    │ Fetch once       │
            │ (status changes) │    │ Check if terminal│
            └──────────────────┘    │ Cache if yes     │
                                    └──────────────────┘
```

**Example Decision Flow**:
```python
# Top 5 executions from 90 total:
top_5 = ['f4hfv', 'wf9fp', '2cmqb', 'lgvwr', 'abc123']

# f4hfv - FAILED, in cache (terminal) → Return cached "OOM killed" ✅ FREE!
# wf9fp - FAILED, in cache (terminal) → Return cached "Image pull failed" ✅ FREE!
# 2cmqb - RUNNING → Fetch logs (need updates) → 1 API call
# lgvwr - FINISHED, NOT in cache → Fetch once, check terminal → 1 API call
# abc123 - FINISHED, in cache (terminal) → Return cached "Runner: 2404s" ✅ FREE!

# RESULT: 2 API calls instead of 5! (3 from cache!)
```

---

### 📊 PHASE 2 COMPLETE IMPLEMENTATION

**Step-by-Step Implementation**:

1. **Add Terminal Detection Function**
   ```python
   def detect_terminal_state(log_lines, status):
       """Check if execution is in terminal state (won't change)"""

       # FINISHED with success message = terminal
       if status == 'FINISHED':
           for line in log_lines:
               if any(pattern in line for pattern in TERMINAL_SUCCESS_PATTERNS):
                   return True, extract_success_message(log_lines)

       # FAILED with known error = terminal
       if status == 'FAILED':
           for line in log_lines:
               if any(pattern in line for pattern in TERMINAL_FAILURE_PATTERNS):
                   return True, extract_failure_message(log_lines)

       # Otherwise: not terminal (need monitoring)
       return False, None
   ```

2. **Add Cache Management**
   ```python
   _terminal_log_cache = {}  # Execution name → cached log data

   def get_cached_note(exec_name):
       """Get cached terminal note if available"""
       if exec_name in _terminal_log_cache:
           cached = _terminal_log_cache[exec_name]
           if cached['terminal']:
               return cached['note']
       return None

   def cache_terminal_note(exec_name, status, note, log_lines):
       """Cache terminal state note forever"""
       _terminal_log_cache[exec_name] = {
           'status': status,
           'note': note,
           'terminal': True,
           'cached_at': datetime.now().isoformat(),
           'log_lines': log_lines,
       }
   ```

3. **Update Fetch Logic (Only Top 5)**
   ```python
   def _fetch_runner_executions_all_regions():
       """Fetch executions with smart terminal caching"""

       # Step 1: Fetch metadata (ALL 90 executions)
       all_execs = _parallel_fetch_regions(fetch_region, ALL_MECHA_REGIONS)

       # Step 2: Sort and get top 5
       top_5 = sorted(all_execs, key=lambda x: x['created_at'], reverse=True)[:5]

       # Step 3: Fetch logs ONLY for top 5 (with caching!)
       for exec in top_5:
           exec_name = exec['name']
           status = exec['status']

           # Check cache first
           cached_note = get_cached_note(exec_name)
           if cached_note:
               exec['note'] = cached_note  # FREE! No API call!
               continue

           # Not cached - need to fetch
           if status == 'RUNNING':
               # Always fetch (status changes)
               log_lines = fetch_logs(exec_name)
               exec['note'] = extract_running_status(log_lines)
               # Don't cache (not terminal)

           else:  # FAILED or FINISHED
               # Fetch once, check if terminal
               log_lines = fetch_logs(exec_name)
               is_terminal, note = detect_terminal_state(log_lines, status)

               if is_terminal:
                   # Cache forever!
                   cache_terminal_note(exec_name, status, note, log_lines)

               exec['note'] = note

       return top_5
   ```

---

### 📈 PHASE 2 PERFORMANCE METRICS

**Scenario**: 18 regions, 90 total executions, 5 displayed

**Without Terminal Caching** (Current Phase 1):
```
Metadata fetch: 90 executions × 0.02s = 1.8s
Log fetch:       0 executions (Phase 1 - none!)
Display:         5 executions with "Fetching logs..."
TOTAL:           1.8s
```

**With Terminal Caching** (Phase 2 - First Refresh):
```
Metadata fetch: 90 executions × 0.02s = 1.8s
Log fetch:       5 executions × 0.5s  = 2.5s (top 5 only!)
Display:         5 executions with actual messages
TOTAL:           4.3s
Cache built:     ~60% terminal (3 cached, 2 RUNNING)
```

**With Terminal Caching** (Phase 2 - Subsequent Refreshes):
```
Metadata fetch: 90 executions × 0.02s = 1.8s
Log fetch:       2 executions × 0.5s  = 1.0s (only RUNNING + new!)
  - 3 from cache (terminal FAILED/FINISHED)
  - 2 fetch (RUNNING executions)
Display:         5 executions with actual messages
TOTAL:           2.8s
Cache hits:      60% (3/5 from cache!)
```

**Savings Over Time**:
- First refresh: 4.3s (builds cache)
- Later refreshes: 2.8s (60% cache hits)
- **vs Original Adaptive**: 13.5s → 2.8s = **79% faster!**
- **vs Phase 1**: 1.8s → 2.8s = +1s (but with actual log messages!)

---

### 🎯 PHASE 2 SUCCESS CRITERIA

✅ Terminal state detection working (success + failure patterns)
✅ Cache built after first refresh (60%+ cache rate)
✅ Only fetch logs for top 5 displayed (not all 27!)
✅ RUNNING executions always show current status
✅ FAILED/FINISHED terminal show cached messages
✅ ~2.8s refresh time (vs 13.5s original)
✅ Cache persists across refreshes
✅ Cache invalidates correctly (RUNNING can't be cached)
✅ ~50-100 lines of simple caching logic
✅ Consistent with Phase 1 structure

---

### 🔬 PHASE 2 CACHE EXAMPLES

**Example Cache After 10 Refreshes**:
```python
_terminal_log_cache = {
    # FAILED - permanent errors (cached forever)
    'f4hfv': {'note': '❌ OOM killed (2048Mi)', 'terminal': True},
    'wf9fp': {'note': '❌ Image pull failed: gcr.io/...', 'terminal': True},
    'xyz123': {'note': '❌ Permission denied: Storage', 'terminal': True},

    # FINISHED - successful completions (cached forever)
    'abc456': {'note': 'Runner: 2404s, 5 jobs ✅', 'terminal': True},
    'def789': {'note': 'Runner: 1776s, 3 jobs ✅', 'terminal': True},
    'ghi012': {'note': 'Runner: 3600s, 8 jobs ✅', 'terminal': True},

    # RUNNING - NOT in cache (status changes)
    # These get fetched every time!
}

# Cache size: 6 executions cached
# Cache hit rate: ~60% (3 hits out of 5 displayed per refresh)
# API savings: 3 log fetches saved per refresh!
```

---

### 💡 WHY PHASE 2 IS BRILLIANT

**Three Key Insights Combined**:

1. **Terminal State Detection** → Cache forever (won't change!)
2. **Only Fetch Top 5** → Don't fetch 85 we never show!
3. **Smart Decision Tree** → Different strategies per status!

**Result**: Simple, fast, intelligent log fetching! 🎯

**But first:** Make Phase 1 work! Get runner table simple and consistent! 🚀

---

---

## 📦 APPENDIX: CURRENT IMPLEMENTATION (Full Code)

**This is the CURRENT complex adaptive system we're simplifying in Phase 1!**

All of this code will be:
- **Phase 1**: Commented out (log fetching parts) or simplified (metadata parts)
- **Phase 2**: Restored and enhanced with terminal state caching

---

### Module-Level State (Top of core.py)

```python
# Adaptive execution log fetching state (module-level for persistence across calls)
_adaptive_exec_state = {
    "hot_executions": set(),  # RUNNING/FAILED executions (need frequent log checks)
    "cold_rotation_idx": 0,   # Rotation index for FINISHED executions
}
```

**Used by**: HOT/COLD rotation system (lines 658-663 in function below)

---

### Complete Function: _fetch_runner_executions_all_regions()

**Location**: `training/cli/monitor/core.py` lines 570-1007 (~437 lines!)

**What it does**:
1. Fetches basic metadata from all 18 regions (parallel)
2. Classifies executions as HOT (RUNNING/FAILED) or COLD (FINISHED)
3. Builds target set (all HOT + 2 rotating COLD)
4. Fetches logs in parallel for target set only
5. Parses logs for:
   - Wrapper bailout errors (fatal errors)
   - W&B agent errors (machine type, permissions, etc.)
   - Success patterns (job submission, runner health)
   - Jobs run counter
   - Error context (20-80 lines around error)
6. Formats execution data with rich status display
7. Sorts by created_at, returns top 5

```python
def _fetch_runner_executions_all_regions(status: StatusCallback, region: str = "us-central1", target_regions: List[str] = None) -> List[Dict]:
    """
    Get Cloud Run job executions from specified or all MECHA regions (adaptive monitoring)

    Shows runner status and any errors from W&B Launch → Vertex AI submission.
    Critical for debugging why jobs aren't starting!

    Uses adaptive log fetching:
    - Hot executions (RUNNING/FAILED): Always fetch logs
    - Cold executions (FINISHED): Rotate through 2 per refresh
    - Reduces logging API calls by ~60%!

    Args:
        region: Legacy parameter (ignored)
        target_regions: Optional list of specific regions to query. If None, queries all 18 MECHA regions.

    Returns:
        List of execution dicts with name, status, error (if any), region
    """
    try:
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ALL 18 MECHA regions (default)
        ALL_MECHA_REGIONS = [
            "us-central1", "us-east1", "us-east4", "us-east5",
            "us-west1", "us-west2", "us-west3", "us-west4",
            "northamerica-northeast1",
            "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9",
            "asia-northeast1", "asia-southeast1",
            "australia-southeast1",
            "southamerica-east1"
        ]

        # Use target_regions if provided (adaptive monitoring), otherwise all 18
        MECHA_REGIONS = target_regions if target_regions is not None else ALL_MECHA_REGIONS

        def fetch_region(region_name: str) -> List[Dict]:
            """Fetch runner executions from a single region"""
            try:
                # List recent Cloud Run executions
                # NOTE: This can take ~20 seconds per region - that's normal for gcloud!
                result = run_gcloud_with_retry(
                    [
                        "gcloud", "run", "jobs", "executions", "list",
                        "--job=vertex-ai-launcher",
                        f"--region={region_name}",
                        f"--limit={LIMIT_RUNNER_EXECUTIONS}",  # Show last N executions per region
                        "--format=json",
                    ],
                    max_retries=1,  # Reduced from 3 - one attempt is enough
                    timeout=30,  # Allow 30s per region (gcloud can be slow)
                    operation_name="list Cloud Run job executions",
                )

                if result.returncode != 0:
                    return []

                executions = json.loads(result.stdout) if result.stdout else []

                # ═══════════════════════════════════════════════════════════════
                # 🔥❄️ ADAPTIVE LOG FETCHING: HOT/COLD CLASSIFICATION
                # ═══════════════════════════════════════════════════════════════

                # Determine which executions need log fetches
                # Hot: RUNNING/FAILED (always fetch)
                # Cold: FINISHED (rotate through 2 per refresh)
                exec_statuses = {}
                for execution in executions:
                    name = execution.get('metadata', {}).get('name', '').split('/')[-1]
                    conditions = execution.get('status', {}).get('conditions', [])
                    status_str = "UNKNOWN"
                    for condition in conditions:
                        if condition.get('type') == 'Completed':
                            if condition.get('status') == 'True':
                                status_str = "FINISHED"
                            elif condition.get('status') == 'False':
                                status_str = "FAILED"
                            elif condition.get('status') == 'Unknown':
                                status_str = "RUNNING"
                    exec_statuses[name] = status_str

                # Hot executions: RUNNING or FAILED
                hot_execs = {name for name, status in exec_statuses.items() if status in ["RUNNING", "FAILED"]}

                # Cold executions: FINISHED
                cold_execs = [name for name, status in exec_statuses.items() if status == "FINISHED"]

                # Rotate through 2 cold executions
                target_executions_set = set(hot_execs)
                if cold_execs:
                    for i in range(min(2, len(cold_execs))):
                        idx = (_adaptive_exec_state["cold_rotation_idx"] + i) % len(cold_execs)
                        target_executions_set.add(cold_execs[idx])
                    _adaptive_exec_state["cold_rotation_idx"] = (_adaptive_exec_state["cold_rotation_idx"] + 2) % len(cold_execs)

                # Update hot executions tracking
                _adaptive_exec_state["hot_executions"] = hot_execs

                # ═══════════════════════════════════════════════════════════════
                # 📋 PARALLEL LOG FETCHING (Target executions only!)
                # ═══════════════════════════════════════════════════════════════

                execution_logs = {}  # execution_name -> lines

                def fetch_execution_logs(execution) -> tuple:
                    """Fetch logs for a single execution (runs in parallel thread)"""
                    name = execution.get('metadata', {}).get('name', '').split('/')[-1]

                    # Only fetch logs for target executions (adaptive!)
                    # This is where the optimization happens - we skip FINISHED executions not in rotation
                    if name not in target_executions_set:
                        return (name, None)

                    # Fetch logs (this is the slow gcloud call!)
                    try:
                        log_result = run_gcloud_with_retry(
                            [
                                "gcloud", "logging", "read",
                                f'resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher AND labels."run.googleapis.com/execution_name"={name}',
                                "--limit=300",
                                "--format=value(textPayload)",
                                "--project=weight-and-biases-476906",
                            ],
                            max_retries=1,
                            timeout=10,
                            operation_name="fetch Cloud Run job logs",
                        )

                        if log_result.returncode == 0 and log_result.stdout:
                            lines = log_result.stdout.strip().split('\n')
                            return (name, lines)
                    except Exception:
                        pass

                    return (name, None)

                # Launch parallel log fetches for targeted executions only (adaptive!)
                # Fetches hot executions + 2 rotating cold
                execs_to_fetch = [e for e in executions if e.get('metadata', {}).get('name', '').split('/')[-1] in target_executions_set]

                if execs_to_fetch:
                    with ThreadPoolExecutor(max_workers=min(10, len(execs_to_fetch))) as executor:
                        futures = {executor.submit(fetch_execution_logs, exec): exec for exec in execs_to_fetch}
                        for future in as_completed(futures):
                            exec_name, lines = future.result()
                            if lines:
                                execution_logs[exec_name] = lines

                # ═══════════════════════════════════════════════════════════════
                # 🔍 FORMAT EXECUTIONS (Complex log parsing!)
                # ═══════════════════════════════════════════════════════════════

                formatted_execs = []
                for execution in executions:
                    # Name is in metadata.name (NOT root.name!)
                    name = execution.get('metadata', {}).get('name', '').split('/')[-1]
                    conditions = execution.get('status', {}).get('conditions', [])

                    # Extract status from conditions
                    status_str = "UNKNOWN"
                    error_msg = None

                    for condition in conditions:
                        if condition.get('type') == 'Completed':
                            if condition.get('status') == 'True':
                                status_str = "FINISHED"
                            elif condition.get('status') == 'False':
                                status_str = "FAILED"
                                error_msg = condition.get('message', 'Unknown error')
                            elif condition.get('status') == 'Unknown':
                                status_str = "RUNNING"

                    # Check for retry conditions (indicates errors)
                    for condition in conditions:
                        if condition.get('type') == 'Retry' and condition.get('status') == 'True':
                            error_msg = condition.get('message', 'Retrying due to error')

                    # If there's an error, fetch detailed logs
                    # Exclude normal polling messages ("will retry after" is just normal queue polling)
                    is_polling_retry = error_msg and "will retry after" in error_msg.lower() and "polling interval" in error_msg.lower()

                    # Clear error_msg for polling retries (not real errors!)
                    if is_polling_retry:
                        error_msg = ""

                    # Extract queue name from container args
                    # Args format: ["-q", "vertex-ai-queue", "-e", "newsofpeace2", ...]
                    queue_name = "—"  # Default fallback
                    spec = execution.get('spec', {})
                    if spec:
                        template = spec.get('template', {})
                        if template:
                            template_spec = template.get('spec', {})
                            if template_spec:
                                containers = template_spec.get('containers', [])
                                if containers and len(containers) > 0:
                                    args = containers[0].get('args', [])
                                    if args:
                                        # Find "-q" flag and get next arg
                                        for i, arg in enumerate(args):
                                            if arg == "-q" and i + 1 < len(args):
                                                queue_name = args[i + 1]
                                                break

                    # ═══════════════════════════════════════════════════════════════
                    # 📜 LOG PARSING (Complex pattern matching!)
                    # ═══════════════════════════════════════════════════════════════

                    # Get logs for error extraction AND jobs run counter
                    # Use parallel-fetched logs (already fetched above - no sequential gcloud calls!)
                    full_error_log = None  # Store FULL error log for TUI popup / CLI detail display
                    jobs_run = 0  # Jobs processed by this runner (NEW: Semi-persistent tracking)

                    # Use pre-fetched logs from parallel execution
                    if status_str in ["FAILED", "RUNNING", "FINISHED"] and name in execution_logs:
                        lines = execution_logs.get(name, [])

                        # ───────────────────────────────────────────────────────
                        # 🚨 PATTERN 1: Wrapper Bailout Detection (PRIORITY!)
                        # ───────────────────────────────────────────────────────

                        bailout_lines = []
                        found_bailout = False
                        bailout_error_msg = None
                        for i, line in enumerate(lines):
                            if '🚨 FATAL ERROR DETECTED' in line or '❌ Killing agent' in line:
                                    found_bailout = True
                                    # Capture bailout context (20 lines before + 80 lines after for full stack trace)
                                    start_idx = max(0, i - 20)
                                    end_idx = min(len(lines), i + 80)
                                    bailout_lines = lines[start_idx:end_idx]

                                    # Search for actual GCP error in bailout context (for table display)
                                    for ctx_line in bailout_lines:
                                        if 'Machine type' in ctx_line and 'is not supported' in ctx_line:
                                            # Extract: Machine type "g2-standard-4" is not supported
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif 'is not supported for machine type' in ctx_line:
                                            # Extract: "NVIDIA_SUPER_GPU" is not supported for machine type "n1-standard-4"
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif 'InvalidArgument:' in ctx_line or 'PermissionDenied:' in ctx_line or 'NotFound:' in ctx_line:
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif 'QuotaExceeded' in ctx_line or 'ResourceExhausted' in ctx_line:
                                            # Quota errors
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif any(pattern in ctx_line for pattern in ['503', 'ServiceUnavailable', '500', 'Internal Error', 'Internal error']):
                                            # 500/503 service errors
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif 'HttpError' in ctx_line and any(code in ctx_line for code in ['400', '401', '403', '404', '429', '500', '502', '503']):
                                            # HTTP errors from GCP API
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif 'ImagePullBackOff' in ctx_line or 'ErrImagePull' in ctx_line:
                                            # Container image errors
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break
                                        elif 'Traceback' in ctx_line or 'Exception:' in ctx_line or 'Error:' in ctx_line:
                                            # Python exceptions
                                            bailout_error_msg = ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()
                                            break

                                    # Fallback to FATAL ERROR message if no specific error found
                                    if not bailout_error_msg:
                                        bailout_error_msg = line.replace('🚨 FATAL ERROR DETECTED:', '').replace('❌', '').strip()

                                    error_msg = bailout_error_msg
                                    break

                            # If wrapper bailout found, use those logs
                            if found_bailout:
                                full_error_log = '\n'.join(bailout_lines)
                            else:
                                # ───────────────────────────────────────────────────────
                                # 🔍 PATTERN 2: W&B Agent Errors (Fallback)
                                # ───────────────────────────────────────────────────────

                                error_context = []
                                for i, line in enumerate(lines):
                                    # Look for W&B agent errors (these appear as "wandb: ERROR" in text)
                                    if 'wandb: ERROR' in line:
                                        # Extract the actual error message (skip "wandb: ERROR" prefix)
                                        if 'Machine type' in line and 'is not supported' in line:
                                            # Extract: Machine type "g2-standard-4" is not supported
                                            error_msg = line.split('wandb: ERROR')[-1].strip()
                                            # Capture context (20 lines around error)
                                            start_idx = max(0, i - 10)
                                            end_idx = min(len(lines), i + 10)
                                            error_context = lines[start_idx:end_idx]
                                            break
                                        elif 'InvalidArgument' in line or 'PermissionDenied' in line or 'NotFound' in line:
                                            error_msg = line.split('wandb: ERROR')[-1].strip()
                                            start_idx = max(0, i - 10)
                                            end_idx = min(len(lines), i + 10)
                                            error_context = lines[start_idx:end_idx]
                                            break
                                    # Fallback: any line with error/exception/failed
                                    # BUT skip INFO messages (Monitoring, checking, watching, etc.)
                                    elif any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'notfound', '404', 'timeout']):
                                        # Skip INFO messages that mention errors/timeouts but aren't errors themselves
                                        # Check for info emojis (⏱️ ⏳ 🔍) or common INFO patterns
                                        if any(info_indicator in line for info_indicator in ['⏱️', '⏳', 'ℹ️', '🔍']):
                                            pass  # Skip INFO messages with info emojis
                                        elif any(info_pattern in line.lower() for info_pattern in ['monitoring for', 'checking for', 'watching for', 'looking for']):
                                            pass  # Skip INFO messages
                                        elif not error_msg:  # Only use as fallback if no W&B error found
                                            error_msg = line[:200]  # Truncate table display only

                                # Store full error context if we found W&B errors
                                if error_context:
                                    full_error_log = '\n'.join(error_context)

                            # ───────────────────────────────────────────────────────
                            # ✅ PATTERN 3: Success Detection & Jobs Counter
                            # ───────────────────────────────────────────────────────

                            # Extract jobs run count from wrapper logs (for ALL statuses)
                            # We output "Runs: N" in ALL locations (wrapper controls output!)
                            # NOTE: gcloud logs are in DESCENDING order (newest first)
                            # So we take the FIRST match = NEWEST value!
                            has_recent_success = False  # Track if runner recently submitted jobs
                            for line in lines:
                                # Detect success patterns (healthy runner indicators)
                                if '✅ Job submitted to Vertex AI!' in line:
                                    has_recent_success = True
                                elif 'Runner alive:' in line:
                                    has_recent_success = True

                                # Extract jobs count (ONE pattern - we control the logs!)
                                if 'Runs:' in line:
                                    # Extract number: "Runs: 3" → 3
                                    try:
                                        parts = line.split(':')
                                        if len(parts) >= 2:
                                            # Handle both "Jobs: 3" and "Jobs: 3)"
                                            num_str = parts[-1].strip().rstrip(')')
                                            jobs_run = int(num_str)
                                            break  # STOP! Found newest value (logs are descending order)
                                    except (ValueError, IndexError):
                                        pass

                            # ───────────────────────────────────────────────────────
                            # 🎨 STATUS DISPLAY LOGIC (Nuanced coloring)
                            # ───────────────────────────────────────────────────────

                            # Determine if this is our completion message (not a real error)
                            # Check for BOTH patterns since "Runner alive:" gets replaced with "Runner completed:" later
                            # ALSO treat timeout as completion (expected behavior, not error)
                            is_completion = error_msg and (
                                'Runner completed:' in error_msg or
                                'Runner alive:' in error_msg or
                                'idle timeout' in error_msg.lower()
                            )
                            # Determine if runner is healthy (recent success)
                            is_healthy_runner = has_recent_success

                    # Color-code status (nuanced - show health!)
                    if status_str == "FINISHED":
                        # FINISHED with completion message or no error = success (green ✓)
                        # FINISHED with real error = warning (yellow, no ✓)
                        if is_completion or not (error_msg and error_msg.strip()):
                            status_display = f"[green]✓ {status_str}[/green]"  # Success!
                        else:
                            status_display = f"[yellow]{status_str}[/yellow]"  # Real warning
                    elif status_str == "FAILED":
                        status_display = f"[bold red]✗ {status_str}[/bold red]"  # Critical failure
                    elif status_str == "RUNNING":
                        # RUNNING healthy runner = green
                        # RUNNING with errors = yellow
                        if is_healthy_runner:
                            status_display = f"[green]▶ {status_str}[/green]"  # Healthy runner!
                        elif error_msg and error_msg.strip():
                            status_display = f"[bold yellow]⚠ {status_str}[/bold yellow]"  # Running with REAL errors (BOLD YELLOW!)
                        else:
                            status_display = f"[green]▶ {status_str}[/green]"  # Healthy running (no messages)!
                    else:
                        status_display = status_str

                    # DON'T clear info messages - they display in cyan in Note column
                    # Only the status icon changes (green vs yellow), messages still show

                    # ───────────────────────────────────────────────────────
                    # ⏱️ DURATION CALCULATION
                    # ───────────────────────────────────────────────────────

                    # Calculate duration (completed = finish-start, running = now-start)
                    start_time = execution.get('status', {}).get('startTime')
                    completion_time = execution.get('status', {}).get('completionTime')
                    duration_display = "—"
                    if start_time:
                        try:
                            from datetime import datetime, timezone
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            # Use completion time if available, otherwise use now (for running executions)
                            if completion_time:
                                end = datetime.fromisoformat(completion_time.replace('Z', '+00:00'))
                            else:
                                end = datetime.now(timezone.utc)  # Running execution - show live duration
                            duration_seconds = (end - start).total_seconds()
                            if duration_seconds < 60:
                                duration_display = f"{int(duration_seconds)}s"
                            else:
                                minutes = int(duration_seconds / 60)
                                seconds = int(duration_seconds % 60)
                                duration_display = f"{minutes}m{seconds}s"
                        except Exception:
                            duration_display = "—"

                    # Post-process error_msg: "alive" → "completed" (for FINISHED/FAILED)
                    if error_msg and 'Runner alive:' in error_msg:
                        if status_str in ['FINISHED', 'FAILED']:
                            error_msg = error_msg.replace('Runner alive:', 'Runner completed:')
                    # NOTE: No "Jobs run:" → "Runs:" replacement needed!
                    # Wrapper outputs "Runs:" directly (entrypoint-wrapper.sh line 139)

                    # ───────────────────────────────────────────────────────
                    # 📦 BUILD FINAL EXECUTION DICT
                    # ───────────────────────────────────────────────────────

                    formatted_execs.append({
                        "name": name,
                        "queue_name": queue_name,  # W&B queue being monitored
                        "status": status_str,
                        "status_display": status_display,
                        "start_time": start_time,  # For live duration ticking
                        "duration": duration_display,  # How long execution took (lifetime for semi-persistent runners)
                        "jobs_run": str(jobs_run),  # NEW: Jobs processed by this runner (semi-persistent tracking)
                        "error": error_msg,
                        "full_error_log": full_error_log,  # FULL error context for TUI popup/CLI detail display
                        "created_at": execution.get('status', {}).get('startTime', 'Unknown'),
                        "created_display": _format_date(execution.get('status', {}).get('startTime', 'Unknown')),
                        "region": region_name  # Add region to each execution!
                    })

                return formatted_execs
            except Exception:
                return []  # Fail silently for this region

        # ═══════════════════════════════════════════════════════════════
        # 🌍 PARALLEL REGION FETCH (All 18 regions!)
        # ═══════════════════════════════════════════════════════════════

        # Query all 18 regions in parallel!
        all_execs = []
        with ThreadPoolExecutor(max_workers=18) as executor:
            future_to_region = {executor.submit(fetch_region, region): region for region in MECHA_REGIONS}
            for future in as_completed(future_to_region):
                region_execs = future.result()
                all_execs.extend(region_execs)

        # 🔍 DEBUG: Log fetch results
        from pathlib import Path
        from datetime import datetime
        debug_log = Path(__file__).parent.parent.parent / "logs" / "runner_fetch_debug.log"
        with open(debug_log, "a") as f:
            f.write(f"\n{datetime.now().isoformat()} 🌍 FETCHED {len(all_execs)} total executions from {len(MECHA_REGIONS)} regions\n")
            for exec in all_execs:
                f.write(f"  - {exec['name']} ({exec['region']}, {exec['status']}, {exec['created_at']})\n")

        # ═══════════════════════════════════════════════════════════════
        # 🔢 SORT & LIMIT (Return top 5 newest)
        # ═══════════════════════════════════════════════════════════════

        # Sort by created time ONLY (newest first) - pure chronological order
        all_execs.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # 🔍 DEBUG: Log sorted results
        with open(debug_log, "a") as f:
            f.write(f"{datetime.now().isoformat()} 🔢 SORTED (newest first):\n")
            for i, exec in enumerate(all_execs[:10]):  # Show top 10
                f.write(f"  {i+1}. {exec['name']} ({exec['region']}, {exec['created_at']})\n")
            f.write(f"{datetime.now().isoformat()} ✂️ RETURNING top {LIMIT_RUNNER_EXECUTIONS}\n\n")

        return all_execs[:LIMIT_RUNNER_EXECUTIONS]  # Return top N most recent across all regions

    except Exception:
        # Fail silently - runner executions are supplementary
        return []
```

**END OF FUNCTION** (~437 lines of complex adaptive + parsing logic!)

---

### Summary of Current Complexity

**Lines of Code**: ~437 lines (function only, not counting state)

**What it does**:
1. ✅ Parallel region fetch (18 regions) - **KEEP THIS!**
2. 🔥❄️ HOT/COLD classification - **COMMENT OUT (Phase 1)**
3. 🎯 Target set building with rotation - **COMMENT OUT (Phase 1)**
4. 📋 Parallel log fetch (27 executions) - **COMMENT OUT (Phase 1)**
5. 🚨 Wrapper bailout detection (20+ error patterns) - **COMMENT OUT (Phase 1)**
6. 🔍 W&B agent error parsing - **COMMENT OUT (Phase 1)**
7. ✅ Success pattern detection - **COMMENT OUT (Phase 1)**
8. 🎨 Rich status display logic - **SIMPLIFY (Phase 1)**
9. ⏱️ Duration calculation - **KEEP THIS!**
10. 🔢 Sort & limit - **KEEP THIS!**

**After Phase 1 simplification**:
- Keep: #1, #9, #10 (~50 lines)
- Comment out: #2, #3, #4, #5, #6, #7 (~350 lines)
- Simplify: #8 (all show "Fetching logs...")

**Result**: ~50 lines, simple like builds/vertex! 🎯

---

**END OF PLAN** - Ready to implement! 🚀
