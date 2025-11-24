# Monitor Screen Refactor Plan - Show More/Less Feature

## Current State Analysis

**File**: `training/cli/monitor/screen.py`
**Size**: 2100+ lines
**Problem**: Massive duplication across 5 tables × 3 load methods = 15 near-identical code blocks

### Duplication Examples

Each table has 3 nearly identical methods:
1. **Initial load** (`_lazy_load_*_worker`) - runs once on page load
2. **Manual refresh** (`_populate_tables`) - triggered by 'r' key
3. **Auto-refresh** (`_refresh_*_worker`) - runs every 30 seconds

**Same logic repeated 15 times:**
- Separate active/completed items
- Apply MAX_* limits
- Add divider rows
- Populate table
- Stop spinner
- Handle errors

---

## Refactoring Strategy

### Phase 1: Extract Table Configuration (Low Risk)

**Create table metadata dict** at class level:

```python
TABLES = {
    "builds": {
        "id": "builds-recent-table",
        "spinner_id": "builds-recent-spinner",
        "max_const": MAX_CLOUD_BUILDS,
        "has_divider": True,
        "active_key": "status",
        "active_values": ["WORKING", "QUEUED"],
        "columns": 7
    },
    "runner": {
        "id": "runner-executions-table",
        "spinner_id": "runner-spinner",
        "max_const": MAX_RUNNER_EXECS,
        "has_divider": True,
        "active_key": "status",
        "active_values": ["RUNNING"],
        "columns": 7
    },
    "vertex": {
        "id": "vertex-jobs-table",
        "spinner_id": "vertex-spinner",
        "max_const": MAX_VERTEX_JOBS,
        "has_divider": True,
        "active_key": "state",
        "active_values": ["JOB_STATE_RUNNING"],
        "columns": 6
    },
    "active": {
        "id": "runs-table",
        "spinner_id": "active-spinner",
        "max_const": MAX_ACTIVE_RUNS,
        "has_divider": False,
        "columns": 5
    },
    "completed": {
        "id": "completed-runs-table",
        "spinner_id": "completed-spinner",
        "max_const": MAX_COMPLETED_RUNS,
        "has_divider": False,
        "columns": 5
    }
}
```

### Phase 2: Create Helper Functions (Medium Risk)

**File**: `training/cli/monitor/table_helpers.py`

```python
def separate_active_completed(items, active_key, active_values):
    """Separate items into active and completed lists"""
    active = [item for item in items if item.get(active_key) in active_values]
    completed = [item for item in items if item.get(active_key) not in active_values]
    return active, completed

def apply_limits(active_items, completed_items, max_limit, extra_count=0):
    """Apply limits: show ALL active, limited completed"""
    items_to_show = active_items

    if max_limit is not None:
        limit = max_limit + extra_count
        items_to_show += completed_items[:limit]
    else:
        items_to_show += completed_items

    return items_to_show

def create_divider_row(num_columns):
    """Create visual divider row for tables"""
    return tuple("[dim blue]" + "━" * 20 + "[/dim blue]" for _ in range(num_columns))

def should_show_divider(active_count, current_item, active_key, active_values):
    """Check if divider should be shown before this item"""
    if active_count == 0:
        return False
    return current_item.get(active_key) not in active_values
```

### Phase 3: Unified Table Update Method (High Risk)

**Create single method to update any table:**

```python
def _update_table_with_items(
    self,
    table_key: str,
    items: list,
    row_builder_func: callable
) -> None:
    """
    Universal table update method

    Args:
        table_key: Key from TABLES dict ("builds", "runner", etc.)
        items: List of items to display
        row_builder_func: Function that builds a row tuple from an item
    """
    config = TABLES[table_key]
    table = self.query_one(f"#{config['id']}", DataTable)
    table.clear()
    self.row_data[table_key].clear()

    # Get extra items count from show more/less
    extra = self._extra_items.get(table_key, 0)

    # Separate and apply limits
    if config.get("has_divider"):
        active, completed = separate_active_completed(
            items,
            config["active_key"],
            config["active_values"]
        )
        items_to_show = apply_limits(active, completed, config["max_const"], extra)

        # Add rows with divider
        added_divider = False
        for idx, item in enumerate(items_to_show):
            if should_show_divider(len(active), item, config["active_key"], config["active_values"]):
                if not added_divider:
                    table.add_row(*create_divider_row(config["columns"]))
                    added_divider = True

            # Build and add row
            row_key, row_tuple = row_builder_func(item, idx)
            table.add_row(*row_tuple, key=row_key)
    else:
        # No divider - just apply limit
        limit = config["max_const"] + extra if config["max_const"] else None
        items_to_show = items[:limit] if limit else items

        for idx, item in enumerate(items_to_show):
            row_key, row_tuple = row_builder_func(item, idx)
            table.add_row(*row_tuple, key=row_key)

    # Move cursor off table
    table.move_cursor(row=-1)

    # Stop spinner
    self._stop_spinner(config["spinner_id"])
```

### Phase 4: Row Builder Functions (Low Risk)

**One function per table type:**

```python
def _build_builds_row(self, build, idx):
    """Build row tuple for cloud builds table"""
    build_key = f"build-recent-{build['build_id']}"

    # Handle error display
    full_error = build.get('error', '')
    error_display = "[dim]—[/dim]"
    if full_error:
        error_display = f"[red]❌ {full_error[:35]}[/red]"
        self.row_data["build_recent"][build_key] = {
            "build_id": build['build_id'],
            "image_name": build['image_name'],
            "region": build['region'],
            "status": build['status'],
            "duration": build['duration_display'],
            "finished": build['finished_display'],
            "note": full_error
        }

    row = (
        f"[dim blue]{build['build_id']}[/dim blue]",
        f"[bright_white]{build['image_name'][:20]}[/bright_white]",
        f"[cyan]{build['region']}[/cyan]",
        build['status_display'],
        f"[cyan]{build['duration_display']}[/cyan]",
        build['finished_display'],
        error_display
    )

    return build_key, row

def _build_runner_row(self, exec_data, idx):
    """Build row tuple for runner executions table"""
    # Similar pattern...

def _build_vertex_row(self, job, idx):
    """Build row tuple for vertex jobs table"""
    # Similar pattern...

def _build_active_run_row(self, run, idx):
    """Build row tuple for active runs table"""
    # Similar pattern...

def _build_completed_run_row(self, run, idx):
    """Build row tuple for completed runs table"""
    # Similar pattern...
```

### Phase 5: Add Show More/Less Buttons (Medium Risk)

**Update compose() to add buttons below each table:**

```python
# In compose() after each table
yield DataTable(id="builds-recent-table", ...)
yield Horizontal(
    Button("▼ Show More", id="show-more-builds", variant="default"),
    Button("▲ Show Less", id="show-less-builds", variant="default"),
    id="builds-buttons",
    classes="table-buttons"
)
```

**Add CSS for button positioning:**

```css
.table-buttons {
    width: 100%;
    height: 1;
    align: right bottom;
    margin-top: 1;
}

.table-buttons Button {
    min-width: 15;
}
```

**Add button handlers:**

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle show more/less buttons"""
    button_id = event.button.id

    if button_id.startswith("show-more-"):
        table_key = button_id.replace("show-more-", "")
        self._extra_items[table_key] += 5  # Show 5 more
        self._refresh_table(table_key)

    elif button_id.startswith("show-less-"):
        table_key = button_id.replace("show-less-", "")
        # Never go below 0
        self._extra_items[table_key] = max(0, self._extra_items[table_key] - 5)
        self._refresh_table(table_key)

def _refresh_table(self, table_key: str):
    """Refresh a specific table with current data + show more state"""
    # Trigger appropriate refresh based on table
    if table_key == "builds":
        self._lazy_load_recent_cloud_builds()
    elif table_key == "runner":
        self._lazy_load_runner_executions()
    # etc...
```

**Dynamic button visibility:**

```python
def _update_show_more_less_buttons(self, table_key: str, total_items: int, active_count: int):
    """Show/hide buttons based on state"""
    config = TABLES[table_key]
    extra = self._extra_items[table_key]

    # Show less button: only if extra > 0
    show_less_btn = self.query_one(f"#show-less-{table_key}", Button)
    show_less_btn.display = extra > 0

    # Show more button: only if there are more items available
    max_limit = config["max_const"] or 0
    current_limit = max_limit + extra
    completed_count = total_items - active_count

    show_more_btn = self.query_one(f"#show-more-{table_key}", Button)
    show_more_btn.display = completed_count > current_limit
```

---

## Implementation Plan

### Step 1: Create Helper Module (1-2 hours)
- Create `training/cli/monitor/table_helpers.py`
- Add helper functions
- Write unit tests

### Step 2: Add Table Configuration (30 mins)
- Add TABLES dict to MonitorScreen
- Verify all metadata is correct

### Step 3: Create Row Builders (1-2 hours)
- Extract row building logic from each table
- Create 5 builder methods
- Test each one

### Step 4: Create Universal Update Method (2-3 hours)
- Implement `_update_table_with_items()`
- Test with one table first
- Gradually migrate all 5 tables

### Step 5: Add Show More/Less UI (1-2 hours)
- Update compose() with button containers
- Add CSS styling
- Add button handlers

### Step 6: Add Dynamic Button Visibility (1 hour)
- Implement `_update_show_more_less_buttons()`
- Call after each table update
- Test edge cases (no items, all items shown, etc.)

### Step 7: Testing & Refinement (2-3 hours)
- Test all tables with show more/less
- Test persistence across navigation
- Test with auto-refresh enabled
- Test edge cases

**Total Estimated Time: 8-14 hours**

---

## Risks & Mitigation

### High Risk Areas

1. **Breaking existing functionality** - Lots of working code to refactor
   - **Mitigation**: Refactor one table at a time, test thoroughly

2. **Auto-refresh conflicts** - Show more state vs new data
   - **Mitigation**: Preserve extra_items across refreshes, only reset on shutdown

3. **Performance** - Extra function calls per table update
   - **Mitigation**: Profile before/after, optimize if needed

### Low Risk Areas

1. **Helper functions** - Pure functions, easy to test
2. **Table metadata** - Static configuration
3. **CSS styling** - Isolated from logic

---

## Alternative Approach: Minimal Refactor

**If full refactor is too risky, do minimal changes:**

1. Keep existing code as-is
2. Just add show more/less buttons
3. Add extra_items tracking
4. Update all 15 locations to use `MAX_* + _extra_items[table]`
5. Add 10 button handlers (one per button)

**Pros**: Less risk, faster implementation
**Cons**: Still have duplication, harder to maintain

---

## Recommendation

**Hybrid approach:**

1. Start with Phase 1-2 (helpers + config) - **LOW RISK**
2. Add show more/less buttons with minimal changes - **QUICK WIN**
3. Gradually refactor to unified method if time permits - **FUTURE IMPROVEMENT**

This gets show more/less working quickly while setting up for future cleanup.

---

## Files to Create/Modify

### New Files
- `training/cli/monitor/table_helpers.py` - Helper functions
- `training/cli/monitor/table_config.py` - TABLES metadata

### Modified Files
- `training/cli/monitor/screen.py` - Main refactor
- `training/cli/monitor/screen.py` (CSS section) - Button styling

### Test Files
- `training/tests/monitor/test_table_helpers.py` - Helper tests
- `training/tests/monitor/test_show_more_less.py` - Feature tests

---

## Success Criteria

✅ Each table has show more/less buttons
✅ Buttons show/hide based on available items
✅ Show more adds 5 items at a time
✅ Show less removes 5 items (never below MAX_* limit)
✅ State persists across navigation
✅ State resets on TUI shutdown
✅ Works with auto-refresh enabled
✅ No breaking changes to existing functionality
✅ Code duplication reduced by >50%
