# Per-Table Autorefresh Checkbox Implementation Plan

## Overview
Add individual autorefresh checkboxes under each table in the TUI monitor screen.
Users can selectively enable/disable autorefresh for each table independently.

## Current State (Post-Cleanup)

- **5 tables** in monitor screen (Active Cloud Builds removed!):
  1. Runner Executions (W&B Launch Agent) - `runner-executions-table`
  2. Vertex AI Jobs - `vertex-jobs-table`
  3. Active W&B Runs - `runs-table`
  4. Completed W&B Runs - `completed-runs-table`
  5. Recent Cloud Builds - `builds-recent-table` (combines QUEUED + WORKING + completed)

- **Current refresh timing** (5 tables):
  - Stagger offset: 10s between tables
  - Refresh interval: 60s
  - Total cycle: 50s (10s × 5)
  - Timers: T+0s, T+10s, T+20s, T+30s, T+40s

- **Dead code cleaned**:
  - All `build_active` and `active_builds` references removed
  - Timer bug fixed (was calling deleted `_refresh_active_cloud_builds()`)
  - `claudes_code_comments` updated: 6 tables → 5 tables

## New Behavior

- **Checkbox under each table** with "Auto-refresh" label
- **Default**: All checkboxes UNCHECKED (disabled by default!)
- **State persistence**: Checkboxes remember state when navigating away/returning
- **Adaptive staggering**: Distributes refresh evenly based on checked count
- **Adaptive interval**: Base 60s divided by enabled count (2 checked = 30s interval, faster!)
- **Toast warning**: If user presses 'a' (auto-refresh) with no checkboxes selected, show toast: "Please check at least one table to enable auto-refresh!"

## Adaptive Timing Logic

```python
# Count enabled tables
num_enabled = sum(self.refresh_enabled.values())

if num_enabled == 0:
    # No tables enabled → no refresh at all
    return

elif num_enabled == 1:
    # Single table → no stagger needed
    stagger_offset = 0
    interval = 60  # Full 60s interval for single table

else:
    # Multiple tables → distribute evenly across 50s window
    # Examples:
    #   5 tables: stagger=50/5=10s, interval=60s
    #   3 tables: stagger=50/3=16.7s, interval=60/3=20s
    #   2 tables: stagger=50/2=25s, interval=60/2=30s
    stagger_offset = 50 / num_enabled
    interval = 60 / num_enabled  # Faster refresh with fewer tables!
```

**Key insight**: When you select only 2 tables, they refresh twice as fast (30s instead of 60s)!

## Implementation Changes

### 1. Import Checkbox Widget

**File**: `training/cli/monitor/screen.py`
**Location**: Line ~68 (import section)

```python
from textual.widgets import Header, Footer, Static, DataTable, Button, Label, Checkbox
```

### 2. Add State Tracking in `__init__()`

**File**: `training/cli/monitor/screen.py`
**Location**: After line ~152 (after `self.row_data = {...}`)

```python
# Per-table autorefresh state (persists across navigation)
self.refresh_enabled = {
    "runner": False,  # Default OFF
    "vertex": False,
    "active_runs": False,
    "completed_runs": False,
    "recent_builds": False
}
```

### 3. Add Checkboxes in `compose()`

**File**: `training/cli/monitor/screen.py`
**Location**: After each table in `compose()` method

For each table, add immediately after the table widget:

```python
# After runner-executions-table:
yield Checkbox("Auto-refresh", value=False, id="cb-runner")

# After vertex-jobs-table:
yield Checkbox("Auto-refresh", value=False, id="cb-vertex")

# After runs-table:
yield Checkbox("Auto-refresh", value=False, id="cb-active-runs")

# After completed-runs-table:
yield Checkbox("Auto-refresh", value=False, id="cb-completed-runs")

# After builds-recent-table:
yield Checkbox("Auto-refresh", value=False, id="cb-recent-builds")
```

**Checkbox IDs**:
- `cb-runner` → controls `self.refresh_enabled["runner"]`
- `cb-vertex` → controls `self.refresh_enabled["vertex"]`
- `cb-active-runs` → controls `self.refresh_enabled["active_runs"]`
- `cb-completed-runs` → controls `self.refresh_enabled["completed_runs"]`
- `cb-recent-builds` → controls `self.refresh_enabled["recent_builds"]`

### 4. Modify `_start_staggered_refresh()`

**File**: `training/cli/monitor/screen.py`
**Location**: Lines ~889-917 (replace existing function)

```python
def _start_staggered_refresh(self) -> None:
    """Start staggered refresh with adaptive timing based on enabled tables"""

    # Count enabled tables
    enabled_tables = [k for k, v in self.refresh_enabled.items() if v]
    num_enabled = len(enabled_tables)

    if num_enabled == 0:
        # No tables enabled - don't start any timers
        return

    # Calculate adaptive timing
    if num_enabled == 1:
        # Single table - no stagger needed
        stagger_offset = 0
        interval = 60
    else:
        # Multiple tables - distribute evenly across 50s window
        stagger_offset = 50 / num_enabled
        interval = 60 / num_enabled  # Faster with fewer tables!

    # Start timers only for enabled tables
    delay = 0

    if self.refresh_enabled["runner"]:
        timer = self.set_timer(delay, lambda: self.set_interval(interval, self._refresh_runner_executions))
        self.refresh_timers.append(timer)
        delay += stagger_offset

    if self.refresh_enabled["vertex"]:
        timer = self.set_timer(delay, lambda: self.set_interval(interval, self._refresh_vertex_jobs))
        self.refresh_timers.append(timer)
        delay += stagger_offset

    if self.refresh_enabled["active_runs"]:
        timer = self.set_timer(delay, lambda: self.set_interval(interval, self._refresh_active_runs))
        self.refresh_timers.append(timer)
        delay += stagger_offset

    if self.refresh_enabled["completed_runs"]:
        timer = self.set_timer(delay, lambda: self.set_interval(interval, self._refresh_completed_runs))
        self.refresh_timers.append(timer)
        delay += stagger_offset

    if self.refresh_enabled["recent_builds"]:
        timer = self.set_timer(delay, lambda: self.set_interval(interval, self._refresh_recent_cloud_builds))
        self.refresh_timers.append(timer)
```

### 5. Add Toast Warning to `action_toggle_auto_refresh()`

**File**: `training/cli/monitor/screen.py`
**Location**: Modify existing `action_toggle_auto_refresh()` method

Add check for zero enabled tables and show toast warning:

```python
def action_toggle_auto_refresh(self) -> None:
    """Toggle auto-refresh on/off"""
    if not self.auto_refresh_enabled:
        # Check if any tables are enabled
        num_enabled = sum(self.refresh_enabled.values())
        if num_enabled == 0:
            # Show toast warning
            self.notify("Please check at least one table to enable auto-refresh!", severity="warning", timeout=3)
            return  # Don't enable auto-refresh

        # Enable auto-refresh
        self.auto_refresh_enabled = True
        self._start_staggered_refresh()
        self.notify("Auto-refresh enabled", timeout=2)
    else:
        # Disable auto-refresh
        self.auto_refresh_enabled = False
        self._stop_staggered_refresh()
        self.notify("Auto-refresh disabled", timeout=2)
```

### 6. Add Checkbox Handler

**File**: `training/cli/monitor/screen.py`
**Location**: Add new method after `on_button_pressed()` (~line 1400)

```python
def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
    """Handle checkbox state changes - restart refresh with new settings"""

    # Map checkbox IDs to state keys
    checkbox_mapping = {
        "cb-runner": "runner",
        "cb-vertex": "vertex",
        "cb-active-runs": "active_runs",
        "cb-completed-runs": "completed_runs",
        "cb-recent-builds": "recent_builds"
    }

    # Update state
    if event.checkbox.id in checkbox_mapping:
        state_key = checkbox_mapping[event.checkbox.id]
        self.refresh_enabled[state_key] = event.value

        # Restart refresh with new adaptive timing
        if hasattr(self, 'auto_refresh_active') and self.auto_refresh_active:
            self._stop_staggered_refresh()
            self._start_staggered_refresh()
```

### 6. Update `claudes_code_comments`

**File**: `training/cli/monitor/screen.py`
**Location**: Lines 1-62 (replace existing comments)

Add to function list:
```python
# MonitorScreen.on_checkbox_changed(event) - Handle autorefresh checkbox changes (restart with adaptive timing)
```

Add to technical review:
```
Adaptive Autorefresh Pattern (NEW!):
- Each table has checkbox underneath ("Auto-refresh")
- Default: ALL enabled
- State persists across navigation
- When fewer tables enabled → faster refresh (60/num_enabled)
- Examples:
  * 5 enabled: 10s stagger, 60s interval
  * 2 enabled: 25s stagger, 30s interval (2× faster!)
  * 1 enabled: 0s stagger, 60s interval
```

## Example Use Cases

### All Tables Enabled (Default)
- 5 checkboxes checked
- Stagger: 10s offset
- Interval: 60s
- T+0s, T+10s, T+20s, T+30s, T+40s, then repeat every 60s

### Only 2 Tables Enabled (Active + Completed Runs)
- Uncheck: Runner, Vertex, Builds
- Check: Active Runs, Completed Runs
- Stagger: 25s offset
- Interval: 30s (2× faster!)
- T+0s Active Runs refresh
- T+25s Completed Runs refresh
- T+30s Active Runs refresh again
- T+55s Completed Runs refresh again

### Single Table Enabled (Recent Builds only)
- Check: Recent Builds
- Uncheck: All others
- Stagger: 0s (no stagger needed)
- Interval: 60s
- T+0s refresh, T+60s refresh, T+120s refresh...

## Files Modified
1. `training/cli/monitor/screen.py` - Main implementation
2. No changes needed to `training/cli/monitor/core.py` (data fetching unchanged)

## Testing Notes
- NOT included (as requested)
- Manual testing required: Navigate to TUI, toggle checkboxes, verify refresh behavior

---

## ✅ IMPLEMENTATION COMPLETE (2025-11-16)

**Commit**: `3aaa0f4` - "Implement per-table autorefresh checkboxes with adaptive timing"

**What was implemented**:
1. ✅ Checkbox widget import (line 66)
2. ✅ State tracking: `self.refresh_enabled` dict (lines 154-161, all False by default)
3. ✅ 5 checkboxes in compose() - all `value=False` by default:
   - cb-runner (line 239)
   - cb-vertex (line 255)
   - cb-active-runs (line 270)
   - cb-completed-runs (line 285)
   - cb-recent-builds (line 224)
4. ✅ Adaptive `_start_staggered_refresh()` (lines 891-951)
   - Counts enabled tables
   - Returns early if 0 enabled
   - Calculates adaptive stagger and interval
   - Only starts timers for enabled tables
5. ✅ Toast warning in `action_toggle_auto_refresh()` (lines 1406-1431)
   - Shows warning if no checkboxes selected
   - Prevents enabling autorefresh with 0 tables
6. ✅ `on_checkbox_changed()` handler (lines 1477-1497)
   - Maps checkbox IDs to state keys
   - Updates `self.refresh_enabled`
   - Restarts refresh if autorefresh is active
7. ✅ Updated `claudes_code_comments` (line 18)

**File changes**:
- `training/cli/monitor/screen.py`: +146 lines, -44 lines
- `training/AUTOREFRESH_CHECKBOX_PLAN.md`: Updated with final behavior

**Key behaviors confirmed**:
- ✅ Default: All checkboxes OFF (user must enable)
- ✅ Toast warning when enabling autorefresh with no checkboxes
- ✅ Adaptive interval: 60 / num_enabled
- ✅ Adaptive stagger: 50 / num_enabled
- ✅ State persists (checkboxes remember settings across navigation)
- ✅ Checkbox changes restart refresh immediately (if autorefresh is on)

**Ready for manual testing in TUI!** 🚀
