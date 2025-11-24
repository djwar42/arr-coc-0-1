# WTF-THREADS-WTF-WTF.md

**Our Current Threading Crisis in arr-coc-0-1 Monitor TUI**

*Context for someone studying canonical threading patterns*

---

## What We're Trying to Do

- 5 DataTables that need parallel data fetching (runner, builds, vertex, active, completed)
- Spinners that need to animate at 8 FPS while tables load
- Auto-refresh every N seconds
- Ordered display (builds → runner → vertex → active → completed)

---

## What We Had (Before Breaking It)

- `run_worker(..., thread=True)` - workers in separate threads
- `time.sleep()` in workers (blocking, but in separate thread)
- `call_from_thread()` to update UI from worker threads
- Spinners animated at ~1.7 FPS (too slow!)

---

## What We Tried (And Broke Everything)

- Converted ALL functions to `async def`
- Changed `time.sleep()` → `await asyncio.sleep()`
- Removed `thread=True` from `run_worker()`
- Result: `RuntimeError: call_from_thread must run in different thread from app`

---

## The Core Problem

- **With `thread=True`**: Workers run in separate thread, use `time.sleep()`, call `call_from_thread()` ✅
- **Without `thread=True`**: Workers run in event loop, use `await asyncio.sleep()`, but `call_from_thread()` fails because we're in same thread! ❌

---

## Key Patterns We Need to Understand

1. **When to use `thread=True`** - for blocking I/O (API calls)
2. **When NOT to use `thread=True`** - for async operations
3. **How to yield to event loop** - so spinners animate
4. **How to update UI from threads** - `call_from_thread()` vs `post_message()`

---

## Current Commits (Need to Revert?)

```
ce3af21 Fix ALL async calls - add await everywhere (CRITICAL FIX #2!)
817a248 Fix run_worker call for async function (CRITICAL!)
21a0db0 Fix missing awaits in async table workers (CRITICAL FIX!)
12f2c5b Convert ALL table workers to async (COMPLETE ASYNC TUI!)
99465ed Add joke to humor sense: Context enormous.
```

---

## The Goal

- Tables load in parallel
- Spinners animate smoothly at 8 FPS
- UI stays responsive during data fetching
- Auto-refresh works correctly

---

## Toast Notification Issues

### What Works ✅

- **"Auto Refresh" toggle toasts** - These show correctly when you press the checkbox
- These are triggered from UI event handlers (same thread as app)

### What Doesn't Work ❌

- **Error toasts from workers** - None of them show!
- **API error notifications** - Silent failures
- **Table load error toasts** - Never appear

### The Difference

```python
# ✅ WORKS - Called from UI event handler (main thread)
def on_checkbox_changed(self, event):
    self.notify("Auto-refresh enabled!")  # Shows!

# ❌ BROKEN - Called from worker thread
def _universal_table_worker(self, ...):
    try:
        data = api.fetch()
    except Exception as e:
        self.notify(f"Error: {e}")  # NEVER SHOWS!
```

### Why This Happens

- `self.notify()` likely needs to be called from main thread/event loop
- Workers with `thread=True` run in separate thread
- Same root cause as `call_from_thread()` issue

### Possible Fixes

1. Use `call_from_thread(self.notify, "message")` from workers
2. Post a custom message that triggers notification in main thread
3. Use `app.post_message()` pattern instead of direct `notify()`

---

## TL;DR

Both `call_from_thread()` crashes AND silent toast failures are symptoms of the same problem - we're trying to update UI from worker threads incorrectly.

The "auto refresh" toasts work because they're triggered from checkbox event handlers (main thread), not from workers.

**We converted sync threaded workers to async event-loop workers but broke `call_from_thread()` which requires being in a DIFFERENT thread.**

Need to either:
1. Keep threads + find another way to yield for spinners
2. Go full async + stop using `call_from_thread()`

---

## The Fundamental Question

**How do we get 8 FPS spinner animation while workers do blocking API calls?**

Options:
1. Thread workers with periodic yields? (but how?)
2. Async workers with `await asyncio.sleep(0)`? (but API calls block!)
3. Hybrid approach? (thread for API, async for UI updates?)

---

*This document exists because we broke everything trying to make spinners smooth. lol ¯\_(ツ)_/¯*

**Date**: 2025-11-19
**Status**: Everything is broken, studying THREADING_THEORY.md for answers
