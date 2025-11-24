# THREADING SOLUTION: The Real Fix for arr-coc-0-1 Monitor TUI

**Date**: 2025-11-19
**Status**: Ready to implement after revert

---

## The Core Problem We Identified

**Chat pattern:** Each chunk = FEW CHARACTERS = microseconds to process
**Our pattern:** Each "chunk" = 50 ROWS × 7 COLUMNS = milliseconds to process

**We're flooding the main thread with heavy `call_from_thread()` callbacks!**

---

## The Solution: BATCH UI UPDATES

### Keep These (They're Correct)
- ✅ `thread=True` for workers (API calls ARE blocking)
- ✅ `call_from_thread()` for UI updates from threads
- ✅ Worker groups for organization

### Change This (The Problem)

**BEFORE (100+ callbacks per table):**
```python
@work(thread=True)
def _universal_table_worker(self):
    for row in rows:
        self.app.call_from_thread(table.add_row, *row)  # 50× heavy callbacks!
        time.sleep(0.010)  # Makes it worse!
    self.app.call_from_thread(table.refresh)
```

**AFTER (1 callback per table):**
```python
@work(thread=True)
def _universal_table_worker(self):
    # 1. Fetch all data (in thread)
    rows = self.helper.get_data()

    # 2. ONE callback to update entire table
    def update_table():
        table.clear()
        for row in rows:
            table.add_row(*row)
        table.refresh()

    self.app.call_from_thread(update_table)
```

---

## Why This Fixes Spinners

### Before (Broken)
```
Main Thread Timeline:
|--callback--|--callback--|--callback--|...(50 more)...|--callback--|
   10ms         10ms         10ms                          10ms

Spinner needs to run every 125ms but main thread is BUSY processing callbacks!
Result: 1-2 FPS
```

### After (Fixed)
```
Main Thread Timeline:
|--ONE callback (all rows)--|--free--|--free--|--free--|
         50ms                  75ms    125ms    125ms

Spinner runs at 125ms intervals with plenty of free time!
Result: 8 FPS
```

---

## Complete Implementation Plan

### Step 1: Revert to Last Working Commit

```bash
git log --oneline -10
# Find commit before async conversion (before 12f2c5b)
git revert --no-commit ce3af21 817a248 21a0db0 12f2c5b
# Or: git reset --hard <commit-before-12f2c5b>
```

### Step 2: Fix Toast Notifications

In ALL worker functions, change:
```python
# BROKEN - direct call from thread
self.notify(f"Error: {e}")

# FIXED - use call_from_thread
self.app.call_from_thread(self.notify, f"Error: {e}", severity="error")
```

### Step 3: Batch UI Updates

For each `_fetch_and_update_*_table()` function:

```python
def _fetch_and_update_runner_table(self):
    """Fetch runner data and update table - BATCHED!"""
    try:
        # 1. Fetch data (blocking, in thread - OK!)
        executions = self.helper.get_runner_executions()

        # 2. Prepare all rows
        rows = []
        for exec in executions:
            rows.append((
                exec.get('id', ''),
                exec.get('status', ''),
                # ... all 7 columns
            ))

        # 3. ONE callback to update entire table
        def update_ui():
            table = self.query_one("#runner-table", DataTable)
            table.clear()

            if not rows:
                # Empty state
                table.add_row("—", "—", "No executions", "—", "—", "—", "—")
            else:
                for row in rows:
                    table.add_row(*row)

            table.refresh()
            self._stop_spinner("runner")

        self.app.call_from_thread(update_ui)

    except Exception as e:
        self.app.call_from_thread(
            self.notify, f"Runner error: {e}", severity="error"
        )
        self.app.call_from_thread(self._stop_spinner, "runner")
```

### Step 4: Remove Per-Row Sleeps

Delete ALL of these from workers:
```python
time.sleep(0.010)   # Per-row yield - NOT NEEDED
time.sleep(0.500)   # Visual fill effect - REMOVE
await asyncio.sleep(0.010)  # Async version - REMOVE
```

The batch update IS the visual effect - table appears all at once!

### Step 5: Keep Visual Fill (Optional)

If you want progressive fill effect, use throttled batches:

```python
def _fetch_and_update_runner_table(self):
    executions = self.helper.get_runner_executions()

    # Update in batches of 5 for visual effect
    for i in range(0, len(executions), 5):
        batch = executions[i:i+5]

        def update_batch(batch=batch, is_last=(i+5 >= len(executions))):
            table = self.query_one("#runner-table", DataTable)
            for exec in batch:
                table.add_row(...)
            if is_last:
                table.refresh()
                self._stop_spinner("runner")

        self.app.call_from_thread(update_batch)
        time.sleep(0.100)  # 100ms between batches = 10 FPS update rate
```

---

## Thread Interaction Summary

### All 5 Table Workers
```
Thread 1 (Runner):   [fetch]----[call_from_thread(update_all)]
Thread 2 (Builds):   [fetch]----[call_from_thread(update_all)]
Thread 3 (Vertex):   [fetch]----[call_from_thread(update_all)]
Thread 4 (Active):   [fetch]----[call_from_thread(update_all)]
Thread 5 (Completed):[fetch]----[call_from_thread(update_all)]

Main Thread:         [spinner]--[spinner]--[update1]--[spinner]--[update2]--[spinner]...
                        125ms      125ms      50ms      125ms       50ms      125ms
```

Each table update is ~50ms, spinner runs every 125ms = PLENTY of time!

---

## Spinner Animation

Spinners should work automatically once we stop flooding main thread:

```python
# In MonitorScreen
def _update_spinners(self):
    """Called by set_interval every 125ms"""
    # This will run smoothly because main thread isn't busy!
    for table_name, spinner in self._spinners.items():
        if self._is_loading[table_name]:
            spinner.update(next_char())
```

No changes needed to spinner code - just stop blocking main thread!

---

## Expected Results After Fix

| Metric | Before | After |
|--------|--------|-------|
| Spinner FPS | 1-2 FPS | 8 FPS |
| Callbacks per table | 50-100 | 1-5 |
| Main thread blocking | 500ms+ | 50ms |
| Toast notifications | Broken | Working |
| Tables loading | Working | Working |

---

## Files to Modify

1. **training/cli/monitor/screen.py**
   - `_fetch_and_update_runner_table()`
   - `_fetch_and_update_builds_table()`
   - `_fetch_and_update_vertex_table()`
   - `_fetch_and_update_active_runs_table()`
   - `_fetch_and_update_completed_runs_table()`
   - All error `notify()` calls

2. **Keep unchanged:**
   - `run_worker(..., thread=True)` ✅
   - `_update_spinners()` ✅
   - Worker groups ✅

---

## Summary

**The fix is NOT about async vs threads.**
**The fix is about BATCHING UI updates!**

1. Keep `thread=True` (correct for blocking I/O)
2. Batch all rows into ONE `call_from_thread()` per table
3. Use `call_from_thread(self.notify, ...)` for toasts
4. Remove per-row sleeps
5. Spinners will animate smoothly automatically!

---

**Estimated Implementation Time**: 1-2 hours
**Risk Level**: Low (pattern is well-documented)
**Confidence**: High (matches canonical Textual patterns)

---

*Based on analysis of THREADING_THEORY.md and comparison with Elia chat streaming pattern*
