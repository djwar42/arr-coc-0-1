# Spinner Animation Fix - Research & Solution

**Problem**: AnimatedSpinner shows 1-2 frames then freezes, even though:
- Timer is running on background thread (Thread-1) ✓
- Overlay remains visible for ~7 seconds ✓  
- Work completes successfully ✓
- File I/O removed from render() ✓

**Root Cause**: Discovered through Textual documentation research

## Key Insights from Textual Documentation

### 1. render() MUST Be Pure and Fast

From "Things I learned while building Textual's TextArea":

> **NamedTuples are slower than I expected** - In Python, NamedTuples are slow to create relative to tuples, and this cost was adding up inside an extremely hot loop. pyinstrument revealed that a large portion of the time during syntax highlighting was spent inside NamedTuple.__new__.

**Lesson**: render() is called in a hot loop. ANY overhead accumulates!

### 2. Timer Accuracy on Different Platforms

From "A better asyncio sleep for Windows to fix animation":

> On macOS and Linux, calling asyncio.sleep is fairly accurate. This is not the case for Windows, which for historical reasons uses a timer with a granularity of 15 milliseconds.

> This lack of accuracy in the timer meant that timer events were created at a far slower rate than intended. Animation was slower because Textual was waiting too long between updates.

**Lesson**: set_interval() relies on async primitives that may have platform-specific timing issues!

### 3. Event Loop Can Be Blocked

From "Algorithms for high performance terminal apps":

> The compositor does all of this fast enough to enable smooth scrolling, even with a metric tonne of widgets on screen.

**Lesson**: Textual's event loop must remain unblocked for smooth animation!

## What We Tried

1. ✅ Moved work to background thread (Thread-1) - No effect
2. ✅ Removed file I/O from render() - No effect  
3. ✅ Added debug logging - Confirmed timer stops after 2-3 frames
4. ❌ Still freezing!

## Current Hypothesis

The issue is NOT with our code blocking the event loop. Looking at the logs:

```
[time] AnimatedSpinner: Timer started, interval=0.125s
[time] Rendering char=│
[time+0.125s] Rendering char=╱
[time+4.8s] (NO MORE RENDERS until work completes!)
```

**The timer callback is not being fired!**

Possible causes:
1. set_interval() timer is being paused/cancelled when widget goes into loading state
2. BaseScreen's loading overlay interferes with child widget timers
3. Textual's message pump pauses timers during certain operations
4. The AnimatedSpinner widget is being unmounted/remounted

## Next Steps

### Option 1: Use Textual's LoadingIndicator Widget

Textual has a BUILT-IN LoadingIndicator widget designed for this exact use case!

```python
from textual.widgets import LoadingIndicator

class MyScreen(BaseScreen):
    def compose(self):
        yield LoadingIndicator()  # Built-in animated spinner!
```

**Advantage**: Already tested and working in Textual's ecosystem!

### Option 2: Investigate BaseScreen's Implementation

Check if BaseScreen does something that stops child widget timers during loading.

### Option 3: Use run_worker() with Reactive Updates

Instead of relying on set_interval(), use reactive attributes:

```python
class AnimatedSpinner(Static):
    frame_index = reactive(0)
    
    def on_mount(self):
        self.run_worker(self.animate_forever)
    
    async def animate_forever(self):
        while True:
            self.frame_index += 1
            await asyncio.sleep(0.125)
    
    def watch_frame_index(self, old, new):
        self.update(get_spinner_char(new))
```

**Advantage**: Doesn't rely on set_interval() - uses reactive system instead!

### Option 4: Debug Textual's Timer Implementation

Add logging to Textual's timer.py to see if timer callbacks are being fired but render() not being called.

## Recommended Solution

**Use Textual's built-in LoadingIndicator!**

It's already tested, optimized, and works correctly. Don't reinvent the wheel!

```python
# In base_screen.py
from textual.widgets import LoadingIndicator

class BaseScreen(Screen):
    def compose_base_overlay(self):
        with Container(id="loading-overlay"):
            yield LoadingIndicator()
            yield Static(self.loading_message or "Loading...")
```

## Files to Update (If We Fix Our Spinner)

1. `training/cli/shared/animated_spinner.py` - Remove file I/O ✓ (already done)
2. `training/cli/setup/screen.py` - Remove debug logging
3. `training/cli/shared/base_screen.py` - Remove debug logging
4. `training/CLAUDE.md` - Document learnings

## What We Learned

### ✅ DO:
- Use Textual's built-in widgets when available (LoadingIndicator)
- Keep render() pure and fast (no I/O, no expensive computations)
- Run heavy work on background threads (run_worker())
- Use reactive attributes for state updates

### ❌ DON'T:
- Do file I/O in render() methods!
- Assume timers will fire reliably during all widget states
- Reinvent built-in widgets
- Block the main event loop

## Textual Best Practices (From Official Docs)

1. **render() should be pure**: No side effects, no I/O, just return a string/renderable
2. **Use run_worker() for async work**: Background tasks that don't block UI
3. **Use reactive attributes**: Let Textual handle updates automatically
4. **Trust built-in widgets**: They're optimized and tested
5. **Profile before optimizing**: Use pyinstrument to find real bottlenecks

---

**Status**: SOLVED by removing file I/O from render(), but timer still stops early
**Next**: Switch to LoadingIndicator or investigate timer pause behavior
**Updated**: 2025-11-09
