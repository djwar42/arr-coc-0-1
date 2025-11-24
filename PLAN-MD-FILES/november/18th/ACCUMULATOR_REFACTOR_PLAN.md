# 🎯 Accumulator Refactor Plan: Fix Display Order

**Problem:** Tables render immediately when workers complete, not when accumulator says to display them!

**Goal:** Accumulator controls ACTUAL rendering with 200ms delays between tables

---

## 📊 Current System (BROKEN)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INITIAL PAGE LOAD                            │
└─────────────────────────────────────────────────────────────────┘

  _accumulated_start()
       │
       ├─── _start_accumulator(all_tables)  ← Sets up tracking
       │         │
       │         └─── _current_batch = ["builds", "runner", "vertex", "active", "completed"]
       │
       └─── Launch 5 workers in parallel:
                │
                ├─── Worker: builds (is_initial_load=True)
                ├─── Worker: runner (is_initial_load=True)
                ├─── Worker: vertex (is_initial_load=True)
                ├─── Worker: active (is_initial_load=True)
                └─── Worker: completed (is_initial_load=True)

                    ↓ ↓ ↓ ↓ ↓ (parallel async execution)

┌────────────────────────────────────────────────────────────────┐
│  WORKER FLOW (Current - BROKEN!)                               │
└────────────────────────────────────────────────────────────────┘

    _universal_table_worker(table_name, is_initial_load=True)
         │
         ├─── Calls _fetch_and_update_runner_table()  ← IGNORES is_initial_load!
         │         │
         │         ├─── Fetches data from API
         │         └─── table.clear()  ← 🖼️ RENDERS IMMEDIATELY!
         │
         ├─── Marks table complete in accumulator
         │         │
         │         └─── self._accumulator_results[table_name] = True
         │
         └─── Accumulator says "display now"
                   │
                   └─── But already displayed! TOO LATE!


RESULT:
  🖼️ completed renders at T+0ms   (first to finish)
  🖼️ vertex renders at T+6s       (second to finish)
  🖼️ builds renders at T+6.05s    (third, only 50ms after vertex!)
  🖼️ runner renders at T+9s       (fourth)
  🖼️ active renders at T+11s      (fifth)

  ❌ Out of order!
  ❌ No 200ms delays!
  ❌ Accumulator is FAKE - just tracks completion, doesn't control display!
```

---

## ✅ New System (FIXED)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INITIAL PAGE LOAD                            │
└─────────────────────────────────────────────────────────────────┘

  _accumulated_start()
       │
       ├─── _start_accumulator(all_tables)
       │         │
       │         ├─── _current_batch = ["builds", "runner", "vertex", "active", "completed"]
       │         └─── _fetched_data = {}  ← NEW! Storage for fetched data
       │
       └─── Launch 5 workers in parallel:
                │
                └─── All workers: is_initial_load=True

                    ↓ ↓ ↓ ↓ ↓ (parallel async execution)

┌────────────────────────────────────────────────────────────────┐
│  WORKER FLOW (New - FIXED!)                                    │
└────────────────────────────────────────────────────────────────┘

    _universal_table_worker(table_name, is_initial_load=True)
         │
         ├─── if is_initial_load:  ← USE THE FLAG!
         │         │
         │         ├─── data = _fetch_runner_data()  ← Fetch only!
         │         │
         │         ├─── Store data (thread-safe):
         │         │       with self._accumulator_lock:
         │         │           self._fetched_data[table_name] = data
         │         │
         │         └─── Mark complete:
         │                 self._accumulator_results[table_name] = True
         │                 ← NO RENDERING YET!
         │
         └─── else:  (auto-refresh, manual refresh)
                   │
                   └─── _fetch_and_update_runner_table()  ← Fetch + render immediately


┌────────────────────────────────────────────────────────────────┐
│  ACCUMULATOR FLOW (New - ACTUAL CONTROL!)                      │
└────────────────────────────────────────────────────────────────┘

    _display_next_ready_table()  ← Polling every 50ms
         │
         ├─── Check if next table ready:
         │       next_table = _current_batch[_accumulator_next_display]
         │       is_ready = next_table in _accumulator_results
         │
         ├─── If not ready → wait 50ms, check again
         │
         ├─── If ready:
         │       │
         │       ├─── Enforce 200ms minimum delay from last display
         │       │       if time_since_last < 200ms:
         │       │           wait (200ms - time_since_last)
         │       │           return
         │       │
         │       ├─── Get fetched data:
         │       │       data = self._fetched_data[next_table]
         │       │
         │       ├─── Call render function:
         │       │       self._update_runner_table(data)
         │       │           └─── table.clear()  ← 🖼️ RENDERS NOW!
         │       │
         │       ├─── Record display time:
         │       │       self._accumulator_last_display_time = time.time()
         │       │
         │       ├─── Increment index:
         │       │       self._accumulator_next_display += 1
         │       │
         │       └─── Schedule next with 200ms delay:
         │               self.set_timer(0.2, self._display_next_ready_table)


RESULT:
  🖼️ builds renders at T+0ms     (first in batch, ready first)
  ⏸️  200ms enforced delay
  🖼️ runner renders at T+200ms   (second in batch)
  ⏸️  200ms enforced delay
  🖼️ vertex renders at T+400ms   (third in batch)
  ⏸️  200ms enforced delay
  🖼️ active renders at T+600ms   (fourth in batch)
  ⏸️  200ms enforced delay
  🖼️ completed renders at T+800ms (fifth in batch)

  ✅ Perfect order!
  ✅ Guaranteed 200ms delays!
  ✅ Accumulator controls rendering!
```

---

## 🔧 Implementation Steps

### Step 1: Add Data Storage

```python
# In __init__:
self._fetched_data = {}  # Stores fetched data before rendering
```

### Step 2: Split Fetch/Update Functions

For EACH table, split into two functions:

**Before (combined):**
```python
def _fetch_and_update_runner_table(self) -> None:
    # Fetch data
    runner_execs = fetch_runner_executions()

    # Update table
    runner_table = self.query_one("#runner-executions-table", DataTable)
    runner_table.clear()
    # ... add rows ...
```

**After (separated):**
```python
def _fetch_runner_data(self) -> list[dict]:
    """Fetch data ONLY - no rendering!"""
    runner_execs = fetch_runner_executions()
    return runner_execs

def _update_runner_table(self, runner_execs: list[dict]) -> None:
    """Render data ONLY - assumes data already fetched!"""
    runner_table = self.query_one("#runner-executions-table", DataTable)
    runner_table.clear()
    # ... add rows ...
```

### Step 3: Modify Worker Logic

```python
def _universal_table_worker(self, table_name: str, config: dict, is_initial_load: bool = False):
    try:
        if is_initial_load:
            # ✅ NEW: Fetch only, store data, NO rendering!
            if table_name == "runner":
                data = self._fetch_runner_data()
            elif table_name == "builds":
                data = self._fetch_builds_data()
            # ... etc for all tables

            # Store fetched data (thread-safe)
            with self._accumulator_lock:
                self._fetched_data[table_name] = data

            # Mark complete (accumulator will display when ready)
            with self._accumulator_lock:
                self._accumulator_results[table_name] = True
                self._accumulator_completion_times[table_name] = time.time()

            # Start polling if first complete
            if not self._accumulator_active:
                self._accumulator_active = True
                self.app.call_from_thread(self._display_next_ready_table)

        else:
            # ✅ AUTO-REFRESH / MANUAL: Fetch + render immediately (old behavior)
            if table_name == "runner":
                self._fetch_and_update_runner_table()
            # ... etc

    finally:
        # ... cleanup ...
```

### Step 4: Modify Accumulator Display Logic

```python
def _display_next_ready_table(self) -> None:
    # ... check if ready, enforce 200ms delay ...

    # Get fetched data
    with self._accumulator_lock:
        data = self._fetched_data.get(next_table)

    # Call render function with fetched data
    if next_table == "runner":
        self._update_runner_table(data)
    elif next_table == "builds":
        self._update_builds_table(data)
    # ... etc for all tables

    # Record display time, schedule next
    # ... existing logic ...
```

---

## 📋 Checklist

### Phase 1: Split Functions (5 tables × 2 functions = 10 new functions)
- [x] Split `_fetch_and_update_runner_table` → `_fetch_runner_data` + `_update_runner_table`
- [x] Split `_fetch_and_update_builds_table` → `_fetch_builds_data` + `_update_builds_table`
- [x] Split `_fetch_and_update_vertex_table` → `_fetch_vertex_data` + `_update_vertex_table`
- [x] Split `_fetch_and_update_active_runs_table` → `_fetch_active_data` + `_update_active_table`
- [x] Split `_fetch_and_update_completed_runs_table` → `_fetch_completed_data` + `_update_completed_table`

### Phase 2: Update Worker Logic
- [x] Add `self._fetched_data = {}` to `__init__`
- [x] Modify `_universal_table_worker` to check `is_initial_load` flag
- [x] If `is_initial_load=True` → Fetch only, store data
- [x] If `is_initial_load=False` → Fetch + render immediately (existing behavior)

### Phase 3: Update Accumulator Display
- [x] Modify `_display_next_ready_table` to get data from `_fetched_data`
- [x] Call appropriate `_update_*_table(data)` function when ready to display (ALL 5 TABLES!)
- [x] Keep existing 200ms delay enforcement logic

### Phase 4: Test & Verify
- [ ] Run `python training/tui.py`
- [ ] Check logs: `grep "TABLE_RENDER" logs/auto_refresh.log`
- [ ] Verify order: builds → runner → vertex → active → completed
- [ ] Verify timing: 200ms minimum between each render
- [ ] Test auto-refresh still works (should use old immediate path)

---

## 🎯 Expected Logs After Fix

```bash
$ grep "TABLE_RENDER\|DISPLAYING:" logs/auto_refresh.log

2025-11-18T20:XX:00.000 🎯 DISPLAYING: builds (position 1/5, waited 0.123s, 999000ms since last)
2025-11-18T20:XX:00.001 🖼️  TABLE_RENDER: builds (table.clear() called)
2025-11-18T20:XX:00.201 🎯 DISPLAYING: runner (position 2/5, waited 0.045s, 200ms since last)
2025-11-18T20:XX:00.202 🖼️  TABLE_RENDER: runner (table.clear() called)
2025-11-18T20:XX:00.402 🎯 DISPLAYING: vertex (position 3/5, waited 2.456s, 200ms since last)
2025-11-18T20:XX:00.403 🖼️  TABLE_RENDER: vertex (table.clear() called)
2025-11-18T20:XX:00.603 🎯 DISPLAYING: active (position 4/5, waited 0.089s, 200ms since last)
2025-11-18T20:XX:00.604 🖼️  TABLE_RENDER: active (table.clear() called)
2025-11-18T20:XX:00.804 🎯 DISPLAYING: completed (position 5/5, waited 5.123s, 200ms since last)
2025-11-18T20:XX:00.805 🖼️  TABLE_RENDER: completed (table.clear() called)
```

**Notice:**
- `🎯 DISPLAYING` timestamp matches `🖼️ TABLE_RENDER` timestamp (±1ms)!
- Exactly 200ms between each pair!
- Perfect order: 1→2→3→4→5!

---

## ⚠️ Important Notes

1. **Don't break auto-refresh!** The `is_initial_load=False` path must still work for auto-refresh
2. **Thread safety:** Always use `self._accumulator_lock` when accessing `_fetched_data`
3. **Keep old functions:** Don't delete `_fetch_and_update_*` functions - they're used for auto-refresh!
4. **Test thoroughly:** Initial load AND auto-refresh need to work

---

## 🚀 Estimated Work

- **Phase 1 (Split functions):** ~30 minutes (mechanical, copy-paste mostly)
- **Phase 2 (Worker logic):** ~15 minutes (simple if/else)
- **Phase 3 (Accumulator):** ~10 minutes (call right function)
- **Phase 4 (Testing):** ~10 minutes (verify logs)

**Total:** ~65 minutes of focused work

Let's make accumulator REAL! 🔥
