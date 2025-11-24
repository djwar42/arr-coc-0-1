# Note Column Consolidation Plan

**Date**: 2025-11-20
**Goal**: Bring Note formatting functions close to the source data

---

## Current State (Scattered!)

### Files Involved

```
training/cli/monitor/
├── core.py          ← SOURCE: Fetches data, extracts raw error messages
└── screen.py        ← DISPLAY: Formats for Rich markup (multiple places!)
```

### Data Flow (Current)

```
API Response
    ↓
core.py: Extract raw error/note text
    ↓
screen.py: Format with Rich markup (colors, emojis)
    ↓
DataTable display
```

---

## Problem: Duplicate Formatting Logic

### Runner Table Note - 2 COPIES!

**Location 1**: `screen.py:2489-2506` (_fetch_runner_data path)
**Location 2**: `screen.py:2671-2688` (_update_runner_table path)

Both have same logic:
```python
if full_error == "Running...":
    error_display = f"[cyan]⏳ {full_error}[/cyan]"
elif full_error.startswith('✓'):
    error_display = f"[green]{full_error[:75]}[/green]"
elif full_error.startswith('✗'):
    error_display = f"[yellow]{full_error[:75]}[/yellow]"
else:
    error_display = f"[red]❌ {full_error[:75]}[/red]"
```

### Builds Table Note - 2 COPIES!

**Location 1**: `screen.py:2797-2806` (_fetch_builds_data path)
**Location 2**: `screen.py:2952-2961` (_update_builds_table path)

Both have same logic:
```python
if not full_error:
    error_display = "[dim]—[/dim]"
elif "Build completed" in full_error:
    error_display = f"[green]{full_error[:60]}[/green]"
else:
    error_display = f"[red]{full_error[:60]}[/red]"
```

### Vertex AI Table Note - Simple (OK for now)

**Location**: Inline in table update functions
```python
error_display = f"[red]{error[:50]}[/red]" if error else "[dim]—[/dim]"
```

---

## Source Files for Raw Error Messages

### 1. Runner Executions (`core.py:1112-1145`)

```python
if status == "FAILED":
    exec['error'] = _fetch_and_extract_error(exec_name)  # From Cloud Run logs
elif status == "FINISHED":
    exec['error'] = "✓ Completed: N jobs"  # From success extraction
elif status == "RUNNING":
    exec['error'] = "Running..."  # Static text
else:
    exec['error'] = "—"
```

**Helper**: `_fetch_and_extract_error()` at line 660
**Helper**: `_fetch_and_extract_success()` at line 598

### 2. Cloud Builds (`core.py:1424-1439`)

```python
if status_val in ["FAILURE", "TIMEOUT"]:
    error_msg = failureInfo.get('detail', '')
elif status_val == "SUCCESS":
    error_msg = f"[timestamp] Build completed: runtime"
```

### 3. Vertex AI Jobs (`core.py:537-559`)

```python
if state == "JOB_STATE_FAILED":
    error_msg = job.get('error', {}).get('message', '')
```

---

## Consolidation Plan

### Option A: Format in core.py (Close to Source) ⭐ RECOMMENDED

Move formatting INTO core.py so data comes pre-formatted:

```python
# core.py - format at source!
def _format_runner_note(raw_error: str) -> str:
    """Format runner Note for Rich display"""
    if not raw_error or raw_error == "—":
        return "[dim]—[/dim]"
    elif raw_error == "Running...":
        return f"[cyan]⏳ {raw_error}[/cyan]"
    elif raw_error.startswith('✓'):
        return f"[green]{raw_error[:75]}[/green]"
    elif raw_error.startswith('✗'):
        return f"[yellow]{raw_error[:75]}[/yellow]"
    else:
        return f"[red]❌ {raw_error[:75]}[/red]"

def _format_builds_note(raw_error: str) -> str:
    """Format builds Note for Rich display"""
    if not raw_error:
        return "[dim]—[/dim]"
    elif "Build completed" in raw_error:
        return f"[green]{raw_error[:60]}[/green]"
    else:
        return f"[red]{raw_error[:60]}[/red]"

def _format_vertex_note(raw_error: str) -> str:
    """Format Vertex AI Note for Rich display"""
    if not raw_error:
        return "[dim]—[/dim]"
    else:
        return f"[red]{raw_error[:50]}[/red]"
```

Then in data return:
```python
# In list_runner_executions:
exec['error'] = raw_error
exec['error_display'] = _format_runner_note(raw_error)

# In _list_recent_cloud_builds:
build['error'] = error_msg
build['error_display'] = _format_builds_note(error_msg)
```

**Benefits**:
- ✅ Single source of truth for formatting
- ✅ screen.py just uses pre-formatted `error_display`
- ✅ No duplicate logic
- ✅ Easy to update colors/formatting

### Option B: Helper in screen.py (Shared Functions)

Create shared helpers in screen.py and call from all locations:

```python
# screen.py - top of file
def _format_runner_note(raw: str) -> str: ...
def _format_builds_note(raw: str) -> str: ...
def _format_vertex_note(raw: str) -> str: ...
```

**Downside**: Still separates formatting from data source

### Option C: Companion File (Close & Named Similarly) ⭐ ALSO GOOD

Create `training/cli/monitor/core_formatters.py` alongside `core.py`:

```
training/cli/monitor/
├── core.py              ← Main data fetching
├── core_formatters.py   ← Note/error formatting (NEW!)
└── screen.py            ← UI display
```

```python
# core_formatters.py
"""Note column formatters - companion to core.py"""

def format_runner_note(raw: str) -> str:
    """Format runner Note for Rich display"""
    if not raw or raw == "—":
        return "[dim]—[/dim]"
    elif raw == "Running...":
        return f"[cyan]⏳ {raw}[/cyan]"
    elif raw.startswith('✓'):
        return f"[green]{raw[:75]}[/green]"
    elif raw.startswith('✗'):
        return f"[yellow]{raw[:75]}[/yellow]"
    else:
        return f"[red]❌ {raw[:75]}[/red]"

def format_builds_note(raw: str) -> str: ...
def format_vertex_note(raw: str) -> str: ...
```

**Benefits**:
- ✅ Close to core.py (same folder, similar name)
- ✅ Easy to find: core.py → core_formatters.py
- ✅ Keeps core.py focused on data fetching
- ✅ Formatters can be imported by both core.py and screen.py

---

## Recommendation: Option A or C

**Option A** (formatting IN core.py) or **Option C** (companion file core_formatters.py)

Both are good because:

1. **Close to source**: Data is formatted where it's extracted
2. **Single definition**: Each formatter defined once
3. **Pre-formatted**: screen.py receives ready-to-display strings
4. **Testable**: Can unit test formatters in core.py

### Implementation Steps

1. **Add 3 formatter functions** to `core.py` (top of file, after imports)
2. **Update `list_runner_executions()`** to add `error_display` field
3. **Update `_list_recent_cloud_builds()`** to add `error_display` field
4. **Update `_list_vertex_ai_jobs()`** to add `error_display` field
5. **Update `screen.py`** to use `error_display` instead of formatting inline
6. **Remove duplicate formatting code** from all 6 locations in screen.py
7. **Test all 3 tables** for correct colors

### Estimated Time

- Add formatters to core.py: 15 min
- Update data functions: 15 min
- Update screen.py (6 locations): 30 min
- Testing: 15 min

**Total: ~1.5 hours**

---

## Files to Modify

### core.py
- Add 3 formatter functions
- Update 3 data functions to use them

### screen.py
- Remove formatting logic from 6 locations:
  - Line ~2489 (runner _fetch path)
  - Line ~2671 (runner _update path)
  - Line ~2797 (builds _fetch path)
  - Line ~2952 (builds _update path)
  - Vertex table locations (TBD)
- Use `data.get('error_display')` instead

---

## Additional Cleanup Opportunities

### Other Error Functions Found

```
training/cli/shared/retry.py:
  - format_retry_error_report()
  - is_already_exists_error()
  - is_not_found_error()

training/cli/launch/zeus/campaign_stats.py:
  - _extract_concise_error()

training/cli/launch/mecha/campaign_stats.py:
  - _extract_concise_error()  ← DUPLICATE!
```

**Future work**: Consider consolidating these too, but lower priority.

---

## Success Criteria

- [ ] 3 formatter functions in core.py
- [ ] All Note formatting logic in ONE place per table type
- [ ] screen.py uses pre-formatted `error_display` field
- [ ] No duplicate formatting code
- [ ] All tables display correct colors
- [ ] Can change a color in ONE place and it updates everywhere

---

**Status**: PLANNED
**Priority**: Medium (technical debt cleanup)
**Dependencies**: None
