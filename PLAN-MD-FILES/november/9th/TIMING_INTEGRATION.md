# Timing Integration Guide

Simple guide for adding timing tracking to major TUI screens.

## Quick Start

```python
from cli.shared.timing import timed_operation

# Wrap major operations
with timed_operation("build_base_image"):
    # Cloud Build happens here
    result = subprocess.run(["gcloud", "builds", "submit", ...])

# For skippable operations
with timed_operation("build_training_image") as timer:
    if hash_exists_in_registry():
        timer.skip()  # Records as skipped, not completed
        return
    # Build happens here
```

## Integration Points

### Setup Screen (`setup/screen.py`)

Track setup duration:

```python
def run_setup(self):
    with timed_operation("setup", record_as_build=False):
        # Existing setup code
        success = run_setup_core(...)
```

### Launch Screen (`launch/screen.py`)

Track job launches:

```python
def launch_job(self):
    with timed_operation("launch_job", record_as_build=False):
        # Existing launch code
        success = run_launch_core(...)
```

### Monitor Screen (`monitor/screen.py`)

Track monitoring session:

```python
def on_mount(self):
    with timed_operation("monitoring", record_as_build=False):
        # Start monitoring workers
        self.start_background_workers()
```

### Teardown Screen (`teardown/screen.py`)

Track teardown duration:

```python
def run_teardown(self):
    with timed_operation("teardown", record_as_build=False):
        # Existing teardown code
        success = run_teardown_core(...)
```

## Build Operations

Build operations are auto-detected by name starting with `build_`:

```python
# These automatically record as builds
with timed_operation("build_base_image"):
    pass  # Records to metrics.builds_completed

with timed_operation("build_training_image") as timer:
    timer.skip()  # Records to metrics.builds_skipped

# Override detection
with timed_operation("custom_op", record_as_build=True):
    pass  # Treated as build even though name doesn't match
```

## Error Tracking

```python
from cli.shared.auto_performance_reporter import get_reporter

try:
    # Do work
    result = subprocess.run(...)
except Exception as e:
    # Record error
    reporter = get_reporter()
    if reporter:
        reporter.get_metrics().record_error()
    raise
```

## What Gets Shown in Exit Summary

```
⏱️  SESSION TIMING
Duration: 125.3s (2.1 min)

Build Times:
  • base: 45.2s (0.8 min)
  • training: skipped (1.2s to check)
  • runner: skipped (0.9s to check)

📊 BUILD STATISTICS
Total builds: 3
  • Completed: 1
  • Skipped: 2 ✓ (hash detection working)
  • Skip rate: 67%

🎯 WHAT HAPPENED
✓ TUI attended to signal, ignored noise
  • Hash detection prevented 2 redundant rebuild(s)
  • Session completed in 125s
  • No errors - system realized what mattered
```

## Best Practices

1. **Wrap major operations** - Setup, launch, monitor, teardown
2. **Use descriptive names** - "build_base_image", "launch_training_job"
3. **Mark skips explicitly** - Call `timer.skip()` when operation is skipped
4. **Record errors** - Call `metrics.record_error()` on failures
5. **Don't over-track** - Only track operations that matter to user

## Already Integrated

- Auto-enabled on TUI startup (`training/tui.py`)
- Auto-prints exit summary on TUI close
- Session metrics automatically collected

## Not Integrated Yet

These screens need timing tracking added:
- [ ] setup/screen.py - Track full setup duration
- [ ] launch/screen.py - Track job launch operations
- [ ] monitor/screen.py - Track monitoring session
- [ ] teardown/screen.py - Track teardown operations

To integrate, just wrap the main operation with `timed_operation()` as shown above!
