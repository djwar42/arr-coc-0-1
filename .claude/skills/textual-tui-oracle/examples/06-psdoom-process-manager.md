# PSDoom - Process Manager with Doom-Inspired Notifications

## Overview

PSDoom is a terminal-based process manager with a playful Doom-inspired interface for safely killing processes. Built with Textual, it demonstrates real-time process monitoring, system integration via `psutil`, async notifications, and creative UI feedback patterns.

**Key Features:**
- Real-time process listing with search/filter
- System process integration via `psutil`
- Async notification sequences (Doom-style "KABOOM")
- Responsive column width calculations
- User-filtered process display (safety feature)
- Keyboard-driven navigation

From [psDooM GitHub Repository](https://github.com/koaning/psdoom) by koaning (accessed 2025-11-02)

## Architecture

### Core Components

**Two-Module Design:**
1. **`app.py`** - Textual application (UI, keyboard handling, async notifications)
2. **`process_manager.py`** - Process operations wrapper around `psutil`

**Separation of Concerns:**
- UI logic isolated in `PSDoomApp`
- System operations encapsulated in `ProcessManager`
- Clean interface between presentation and data

### Entry Point

```python
# psdoom/__main__.py
from psdoom.app import main

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Install
uv pip install psdoom

# Run
python -m psdoom
# or
uv run python -m psdoom
```

## Process Management Layer

### ProcessManager Class

**Located:** `psdoom/process_manager.py`

```python
import psutil
import signal
from typing import Dict, List, Optional

class ProcessManager:
    """Manages listing, filtering, and termination of system processes."""

    def __init__(self):
        self.refresh()

    def refresh(self) -> None:
        """Refresh the list of running processes."""
        self.processes = {}
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline', 'status']):
            try:
                self.processes[proc.info['pid']] = proc.info
                # Convert cmdline list to string for display
                if self.processes[proc.info['pid']]['cmdline']:
                    self.processes[proc.info['pid']]['cmdline'] = ' '.join(
                        self.processes[proc.info['pid']]['cmdline']
                    )
                else:
                    self.processes[proc.info['pid']]['cmdline'] = ""
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
```

**Key Pattern: Exception Handling**
- `psutil` can throw exceptions for processes that vanish mid-iteration
- Gracefully handle `NoSuchProcess`, `AccessDenied`, `ZombieProcess`

### Process Filtering

```python
def filter_processes(self, search_term: Optional[str] = None) -> List[Dict]:
    """Filter processes based on search term in name or command line."""
    if not search_term:
        return list(self.processes.values())

    filtered = []
    for pid, process in self.processes.items():
        name = process.get('name', '').lower()
        cmdline = process.get('cmdline', '').lower()

        if (search_term.lower() in name) or (search_term.lower() in cmdline):
            filtered.append(process)

    return filtered
```

**Pattern:** Case-insensitive substring search across both process name and full command line.

### Safe Process Termination

```python
def kill_process(self, pid: int) -> bool:
    """Kill process with graceful fallback to SIGKILL."""
    try:
        proc = psutil.Process(pid)
        # Try SIGTERM first (graceful shutdown)
        proc.terminate()
        # Wait briefly to see if it terminates
        gone, alive = psutil.wait_procs([proc], timeout=0.5)
        if alive:
            # If still alive, use SIGKILL (force)
            proc.kill()
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        print(f"Error killing process {pid}: {str(e)}")
        return False
```

**Safety Pattern:**
1. SIGTERM first (allows cleanup)
2. Wait 0.5 seconds
3. SIGKILL if still alive (force termination)

### Current User Filter

```python
def get_current_username(self) -> str:
    """Get username of current user."""
    import getpass
    return getpass.getuser()
```

**Safety Feature:** PSDoom filters processes to show only current user's processes by default (prevents accidental system process termination).

## UI Application Layer

### PSDoomApp Class

**Located:** `psdoom/app.py`

```python
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, Input, DataTable, Static
from textual.reactive import reactive
from textual.binding import Binding

class PSDoomApp(App):
    """Main PSDoom application."""

    TITLE = "PSDoom - Terminal Process Manager"
    SUB_TITLE = "Kill processes"

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "focus_table", "Table"),
        Binding("s", "focus_search", "Search"),
        Binding("k", "kill", "Kill Process"),
        Binding("enter", "select_item", "Select Process"),
    ]

    search_term = reactive("")
    selected_pid = reactive(None)

    def __init__(self):
        super().__init__()
        self.process_manager = ProcessManager()
        self.last_ctrl_c_time = 0  # Track for double Ctrl+C exit
```

### Layout Composition

```python
def compose(self) -> ComposeResult:
    """Compose the UI."""
    yield Header()

    with Container(id="main"):
        with Horizontal(id="search-container"):
            yield Static("Search:", id="search-label")
            yield Input(placeholder="Type to filter processes...", id="search-input")

        yield DataTable(id="process-table")

    yield Footer()
```

**Layout Structure:**
```
┌─ Header ──────────────────────────┐
│ PSDoom - Terminal Process Manager │
├───────────────────────────────────┤
│ Search: [________________]        │
├───────────────────────────────────┤
│ PID    Name         Command       │
│ 1234   python       app.py        │
│ 5678   chrome       --no-sandbox  │
│ ...                               │
├───────────────────────────────────┤
│ Footer (keybindings)              │
└───────────────────────────────────┘
```

### CSS Styling

```python
DEFAULT_CSS = """
Screen {
    background: #202020;
}

#main {
    height: 100%;
    margin: 0 1;
    layout: vertical;
}

#search-container {
    height: 3;
    margin-bottom: 1;
    align-vertical: middle;
}

#search-input {
    width: 100%;
    background: #333333;
    color: #ffffff;
    border: solid #777777;
}

#process-table {
    height: 1fr;
    width: 100%;
    margin-bottom: 1;
    border: solid #555555;
}

.datatable--header {
    background: #444444;
    color: #ffffff;
}

.datatable--cursor {
    background: #666666;
}

.datatable--hover {
    background: #444444;
}
"""
```

**Dark Theme Design:**
- Background: `#202020` (dark gray)
- Table headers: `#444444` (medium gray)
- Cursor highlight: `#666666` (lighter gray)
- Borders: `#555555` and `#777777` (subtle contrast)

## Responsive Column Width Calculation

### Data-Aware Width Algorithm

**Problem:** Process names and commands vary wildly in length. Fixed widths waste space or truncate too aggressively.

**Solution:** Calculate column widths based on actual visible data.

```python
def _compute_column_widths_for(self, processes: List[Dict]) -> Dict[int, int]:
    """Compute data-aware column widths to minimize whitespace."""
    pid_width = 7
    safety_padding = 6
    total_width = self.size.width or 100
    available = max(20, total_width - (pid_width + safety_padding))

    # Bounds
    min_name, min_cmd = 12, 20
    max_name = 48  # Cap so name doesn't starve command

    if not processes:
        # Fallback proportional split
        name_width = max(min_name, int(available * 0.30))
        cmd_width = max(min_cmd, available - name_width)
        return {0: pid_width, 1: name_width, 2: cmd_width}

    # Find longest raw lengths in current view
    longest_name = 0
    longest_cmd = 0
    for p in processes:
        longest_name = max(longest_name, len(p.get("name", "")))
        longest_cmd = max(longest_cmd, len(p.get("cmdline", "")))

    # Desired widths, bounded
    desired_name = max(min_name, min(longest_name, max_name))
    base_name_share = max(min_name, int(available * 0.30))
    name_width = min(desired_name, base_name_share)
    cmd_width = max(min_cmd, available - name_width)

    # If command would be too tight while name has room to grow, re-balance
    if cmd_width < min_cmd and desired_name > name_width:
        deficit = min_cmd - cmd_width
        grow = min(deficit, desired_name - name_width)
        name_width += grow
        cmd_width = max(min_cmd, available - name_width)

    # If names are very short, shrink name column to actual need
    name_width = min(name_width, longest_name if longest_name >= min_name else min_name)
    cmd_width = max(min_cmd, available - name_width)

    return {0: pid_width, 1: name_width, 2: cmd_width}
```

**Algorithm Steps:**
1. Reserve fixed space for PID (7 chars)
2. Calculate available space for name + command
3. Set min/max bounds (name: 12-48, command: 20+)
4. Measure longest visible name and command in current filtered view
5. Allocate proportionally (~30% name, rest command) within bounds
6. Re-balance if one column starves the other
7. Shrink name column if actual data is shorter than allocation

### Ellipsis Helper

```python
@staticmethod
def _ellipsize(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis if too long."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"
```

**Pattern:** Unicode ellipsis (`…`) for clean truncation.

### Resize Handling

```python
def on_resize(self, event: Resize) -> None:
    """Recompute column widths on terminal resize and refresh table."""
    self.refresh_process_list()  # Recalculates widths from current data
```

**Real-Time Responsiveness:** Every terminal resize triggers width recalculation, ensuring optimal layout at any terminal size.

## Search and Filtering

### Reactive Search

```python
search_term = reactive("")

@on(Input.Changed, "#search-input")
def on_search_input_changed(self, event: Input.Changed) -> None:
    """Update search term when input changes."""
    self.search_term = event.value
    self.refresh_process_list()
```

**Pattern:**
- Reactive variable tracks search term
- Input.Changed event updates reactive variable
- Automatically triggers table refresh

### User-Filtered Process Display

```python
def refresh_process_list(self) -> None:
    """Refresh process list based on current search term."""
    self.process_manager.refresh()

    # Filter to only show current user's processes (SAFETY)
    current_username = self.process_manager.get_current_username()
    processes = [
        p for p in self.process_manager.filter_processes(self.search_term)
        if p.get('username') == current_username
    ]

    table = self.query_one("#process-table", DataTable)
    table.clear()

    # Compute widths and populate
    widths = self._compute_column_widths_for(processes)
    table.column_widths = widths
    name_max = widths[1]
    cmd_max = widths[2]

    for process in processes:
        pid = str(process.get('pid', ''))
        name = self._ellipsize(process.get('name', ''), name_max)
        cmdline = self._ellipize(process.get('cmdline', ''), cmd_max)
        table.add_row(pid, name, cmdline)
```

**Safety Feature:** Only displays processes owned by current user (prevents accidental system process kills).

## Keyboard Navigation

### Focus Management

```python
def action_focus_search(self) -> None:
    """Focus the search input."""
    search_input = self.query_one("#search-input", Input)
    self.set_focus(search_input)

def action_focus_table(self) -> None:
    """Focus the process table."""
    table = self.query_one("#process-table", DataTable)
    table.focus()

@on(Input.Submitted, "#search-input")
def on_search_input_submitted(self, event: Input.Submitted) -> None:
    """When search is submitted, focus back on process table."""
    self.query_one("#process-table").focus()
```

**Workflow:**
1. App starts with search input focused
2. `s` key returns focus to search
3. `Escape` or Enter focuses table
4. Arrow keys navigate table rows

### Row Selection

```python
@on(DataTable.RowSelected)
def on_row_selected(self, event: DataTable.RowSelected) -> None:
    """Handle row selection in table."""
    row = event.data_table.get_row(event.row_key)
    if row:
        self.selected_pid = int(row[0])
        proc_name = row[1]
        pid = row[0]
        self.notify(f"Selected: {proc_name} (PID: {pid}) - Press 'k' to kill", timeout=3)
```

**Feedback:** Notification shows selected process details and hints at kill action.

### Double Ctrl+C Exit Pattern

```python
@on(Key)
def on_key(self, event: Key) -> None:
    """Handle key press events."""
    # Implement double Ctrl+C to exit
    if event.key == "ctrl+c":
        event.prevent_default()
        event.stop()

        current_time = asyncio.get_event_loop().time()
        # If pressed twice within 1 second, exit
        if current_time - self.last_ctrl_c_time < 1.0:
            self.exit()
        else:
            self.notify("Press Ctrl+C again to exit")
            self.last_ctrl_c_time = current_time
```

**Safety Pattern:**
- First Ctrl+C: Notify user
- Second Ctrl+C (within 1 second): Exit app
- Prevents accidental exits

## Async Process Killing with Notifications

### The "Doom" Experience

**Creative Pattern:** Process termination is accompanied by a timed sequence of notifications creating a playful "Doom-style" kill animation.

```python
async def kill_process_with_notifications(self, pid: int, proc_name: str) -> None:
    """Kill process with status notifications (Doom-inspired)."""
    # Initial warning - red
    self.notify(f"[bold red]KILLING PROCESS: {proc_name} (PID: {pid})[/]", timeout=3)
    await asyncio.sleep(0.5)

    # Status updates
    self.notify("[red]Initiating process termination...[/]", timeout=2)
    await asyncio.sleep(0.7)

    self.notify("[red]Sending SIGTERM signal...[/]", timeout=2)
    await asyncio.sleep(0.8)

    # Actually kill the process
    success = self.process_manager.kill_process(pid)

    self.notify("[yellow]Waiting for process to terminate...[/]", timeout=2)
    await asyncio.sleep(1)

    # The "kaboom" animation
    self.notify("[bold red blink]* * * KABOOM * * *[/]", timeout=3)
    await asyncio.sleep(1)

    # Final result - green for success, orange for warnings
    if success:
        self.notify(f"[bold green]Process {proc_name} (PID: {pid}) terminated successfully![/]", timeout=5)
    else:
        self.notify(f"[bold orange]Warning: Process {proc_name} (PID: {pid}) may not have terminated correctly[/]", timeout=5)

    # Refresh process list
    await self.delayed_refresh()

async def delayed_refresh(self) -> None:
    """Refresh process list after short delay."""
    await asyncio.sleep(2)
    self.refresh_process_list()
```

**Notification Sequence:**
1. **0.0s:** "KILLING PROCESS" (bold red)
2. **0.5s:** "Initiating termination"
3. **1.2s:** "Sending SIGTERM signal"
4. **2.0s:** *Actual kill happens*
5. **2.0s:** "Waiting for process to terminate"
6. **3.0s:** "KABOOM" (bold red blink)
7. **4.0s:** Success/failure message (green/orange)
8. **6.0s:** Table refreshes

**Rich Markup Formatting:**
- `[bold red]` - Bold red text
- `[bold red blink]` - Bold red blinking text
- `[yellow]`, `[green]`, `[orange]` - Color coding
- `timeout=N` - Notification display duration

### Kill Action Handler

```python
def action_kill(self) -> None:
    """Kill the selected process."""
    table = self.query_one("#process-table", DataTable)
    if table.cursor_row is not None:
        # Get PID from current cursor row
        pid_str = table.get_cell_at((table.cursor_row, 0))
        if pid_str and pid_str.isdigit():
            self.selected_pid = int(pid_str)
            # Find process name for notifications
            for proc in self.process_manager.filter_processes():
                if proc['pid'] == self.selected_pid:
                    proc_name = proc.get('name', 'Unknown')

                    # Start async kill sequence
                    asyncio.create_task(
                        self.kill_process_with_notifications(self.selected_pid, proc_name)
                    )
                    break
```

**Pattern:** Launch async task for kill sequence, allowing UI to remain responsive during notifications.

## Initialization and Mounting

```python
def on_mount(self) -> None:
    """Initialize app when mounted."""
    # Set up table
    table = self.query_one("#process-table", DataTable)
    table.cursor_type = "row"
    table.add_columns("PID", "Name", "Command")

    # Initial process list (also computes data-aware widths)
    self.refresh_process_list()

    # Make table focusable
    table.can_focus = True

    # Focus search input on startup
    self.query_one("#search-input", Input).focus()
```

**Startup Flow:**
1. Configure DataTable (row cursor, columns)
2. Load initial process list
3. Calculate responsive column widths
4. Focus search input (ready for user filtering)

## Project Structure

```
psdoom/
├── psdoom/
│   ├── __init__.py
│   ├── __main__.py          # Entry point (python -m psdoom)
│   ├── app.py               # Textual app (UI, events, async notifications)
│   └── process_manager.py   # Process operations (psutil wrapper)
├── pyproject.toml           # Project config
├── uv.lock                  # Dependency lock
├── Makefile                 # Build tasks
├── .gitignore
└── README.md
```

## Dependencies

From `pyproject.toml`:

```toml
[project]
name = "psdoom"
version = "0.1.2"
requires-python = ">=3.8"
dependencies = [
    "textual==0.38.1",
    "psutil==5.9.0",
]

[project.scripts]
psdoom = "psdoom.app:main"
```

**Key Libraries:**
- `textual` (0.38.1) - TUI framework
- `psutil` (5.9.0) - System and process utilities

## Key Patterns and Techniques

### 1. System Integration via psutil

**Process Monitoring:**
```python
for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline', 'status']):
    try:
        process_info = proc.info
        # Use process info...
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass  # Handle ephemeral/protected processes
```

### 2. Responsive UI with Dynamic Calculations

**Terminal Resize:**
- `on_resize()` event triggers width recalculation
- Widths computed from actual visible data
- Min/max bounds prevent degenerate cases

### 3. Async Notifications for User Feedback

**Non-Blocking Sequences:**
```python
# UI remains responsive during notification sequence
asyncio.create_task(self.kill_process_with_notifications(pid, name))
```

### 4. Safety Through Filtering

**User-Only Processes:**
```python
current_username = self.process_manager.get_current_username()
processes = [p for p in all_processes if p['username'] == current_username]
```

### 5. Graceful Process Termination

**SIGTERM → SIGKILL Fallback:**
```python
proc.terminate()  # Graceful
gone, alive = psutil.wait_procs([proc], timeout=0.5)
if alive:
    proc.kill()  # Force
```

### 6. Rich Console Markup

**Textual Notification Formatting:**
```python
self.notify("[bold red blink]* * * KABOOM * * *[/]", timeout=3)
```

**Supported Tags:**
- `[bold]`, `[italic]`, `[underline]`
- `[red]`, `[green]`, `[yellow]`, `[blue]`, `[orange]`, etc.
- `[blink]` - Blinking text
- Combine: `[bold red blink]`

### 7. Reactive State Management

**Automatic Updates:**
```python
search_term = reactive("")  # Changes trigger watchers

@on(Input.Changed)
def on_input_changed(self, event):
    self.search_term = event.value  # Reactive update
    # Automatically refreshes dependent UI
```

### 8. Event-Driven Focus Management

**Bidirectional Focus Flow:**
- Search input → Table (on Enter)
- Table → Search input (on 's' key)
- Table → Table (on Escape - refocus)

## Complete Example Application

Here's a minimal process manager inspired by psDooM:

```python
# simple_process_manager.py
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer, Input, DataTable, Static
from textual.reactive import reactive
from textual.binding import Binding
from textual import on
import psutil

class ProcessManagerApp(App):
    """Simple process manager TUI."""

    TITLE = "Process Manager"

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("k", "kill", "Kill"),
    ]

    CSS = """
    #search-input {
        margin: 1;
    }

    #process-table {
        height: 1fr;
        margin: 1;
    }
    """

    search_term = reactive("")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Filter processes...", id="search-input")
        yield DataTable(id="process-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#process-table", DataTable)
        table.add_columns("PID", "Name", "Command")
        table.cursor_type = "row"
        self.refresh_processes()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        self.search_term = event.value
        self.refresh_processes()

    def refresh_processes(self) -> None:
        """Refresh process list."""
        table = self.query_one("#process-table", DataTable)
        table.clear()

        # Get current user's processes only
        import getpass
        current_user = getpass.getuser()

        for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
            try:
                info = proc.info
                if info['username'] != current_user:
                    continue

                name = info['name']
                cmdline = ' '.join(info['cmdline'] or [])

                # Filter by search term
                if self.search_term and self.search_term.lower() not in name.lower():
                    continue

                table.add_row(
                    str(info['pid']),
                    name[:20],  # Truncate name
                    cmdline[:50]  # Truncate command
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def action_refresh(self) -> None:
        self.refresh_processes()
        self.notify("Process list refreshed")

    def action_kill(self) -> None:
        """Kill selected process."""
        table = self.query_one("#process-table", DataTable)
        if table.cursor_row is not None:
            pid_str = table.get_cell_at((table.cursor_row, 0))
            if pid_str and pid_str.isdigit():
                pid = int(pid_str)
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    self.notify(f"Terminated process {pid}", severity="warning")
                    self.refresh_processes()
                except Exception as e:
                    self.notify(f"Error: {e}", severity="error")

if __name__ == "__main__":
    ProcessManagerApp().run()
```

**Run:**
```bash
python simple_process_manager.py
```

## Use Cases

### 1. Interactive Process Management
- Search for processes by name or command
- Navigate with arrow keys
- Kill selected process with visual feedback

### 2. Development Cleanup
- Find and kill hung development servers
- Clean up zombie processes
- Terminate runaway scripts

### 3. Resource Monitoring
- Quick overview of running processes
- Filter to specific application families
- Identify resource-heavy processes

### 4. Learning System Integration
- Understand `psutil` library usage
- Learn async notification patterns
- Study responsive UI calculations

## Advanced Techniques

### Dynamic Column Width Algorithm

**Problem:** Fixed columns waste space or truncate too aggressively.

**Solution:** Measure actual data, allocate proportionally within bounds.

**Implementation:**
1. Measure longest visible name and command
2. Allocate ~30% to name (bounded 12-48 chars)
3. Give remainder to command (min 20 chars)
4. Re-balance if either column starves the other
5. Shrink name column if data is shorter than allocation

**Benefits:**
- Minimal whitespace
- No unnecessary truncation
- Adapts to filtered data
- Responsive to terminal resizes

### Async Notification Sequences

**Pattern:** Create timed feedback sequences without blocking UI.

```python
async def multi_step_operation(self):
    self.notify("Step 1...", timeout=2)
    await asyncio.sleep(1)

    self.notify("Step 2...", timeout=2)
    await asyncio.sleep(1)

    # Do actual work
    result = await some_operation()

    self.notify(f"Complete: {result}", timeout=3)

# Launch without blocking
asyncio.create_task(self.multi_step_operation())
```

**Use Cases:**
- Progress feedback for long operations
- Staged deployment notifications
- Error recovery sequences
- Tutorial/onboarding flows

### Safe System Operations

**Checklist for System Integration:**
1. **Filter to current user** - Don't expose system processes
2. **Graceful → Force** - Try SIGTERM before SIGKILL
3. **Exception handling** - Processes can vanish mid-operation
4. **User confirmation** - Double-tap for destructive actions
5. **Visual feedback** - Show what's happening (notifications)

## Performance Considerations

### Process Iteration

**Efficient `psutil` Usage:**
```python
# Good: Specify needed fields
for proc in psutil.process_iter(['pid', 'name', 'username']):
    info = proc.info  # Already has only requested fields

# Avoid: Getting all fields
for proc in psutil.process_iter():
    info = proc.as_dict()  # Fetches everything
```

### Table Refresh Strategy

**Current Implementation:** Full table clear + repopulate on every search change.

**Optimization Ideas:**
- Incremental updates (add/remove changed rows)
- Debounce search input (wait 200ms after typing stops)
- Cache process list, filter in-memory

**Trade-off:** psDooM chooses simplicity over micro-optimization (process lists are small enough for full refresh).

## Comparison: psDooM vs htop vs GUI Task Managers

| Feature | psDooM | htop | GUI Task Manager |
|---------|--------|------|------------------|
| **Platform** | Cross-platform Python | Unix/Linux | OS-specific |
| **UI Framework** | Textual | ncurses | Native GUI |
| **Search** | Instant filter | Incremental search | Varies |
| **Safety** | User processes only | All processes | All processes |
| **Feedback** | Async notifications | Status line | Dialogs |
| **Customization** | Python code | Config file | Settings UI |
| **Installation** | `pip install` | Package manager | Built-in |

**psDooM's Niche:**
- Learning tool for Textual + psutil
- Playful UI (Doom-inspired notifications)
- Safe by default (user processes only)
- Easily hackable (pure Python, small codebase)

## Learning Takeaways

### For Textual Developers

**Study These Patterns:**
1. **DataTable with dynamic columns** - Responsive width calculations
2. **Reactive search** - Input.Changed → reactive variable → auto-refresh
3. **Async notifications** - Non-blocking feedback sequences
4. **Focus management** - Bidirectional focus flow between widgets
5. **Rich markup** - Colored, bold, blinking text in notifications

### For System Integration

**Key Lessons:**
1. **`psutil` exception handling** - Processes can vanish mid-iteration
2. **SIGTERM → SIGKILL pattern** - Graceful before force
3. **User filtering** - Safety through limiting scope
4. **Command line parsing** - Join `cmdline` list for display

### For UI Design

**Design Principles:**
1. **Data-aware layouts** - Measure, don't assume
2. **Visual feedback** - Tell users what's happening
3. **Safety through UX** - Double-tap, confirmations, filtering
4. **Playful interfaces** - "KABOOM" makes process killing fun

## Extending psDooM

### Add CPU/Memory Columns

```python
# In ProcessManager.refresh():
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
    info = proc.info
    info['cpu'] = f"{proc.cpu_percent():.1f}%"
    info['memory'] = f"{proc.memory_percent():.1f}%"
    self.processes[info['pid']] = info

# In PSDoomApp.on_mount():
table.add_columns("PID", "Name", "CPU", "Memory", "Command")

# Update refresh_process_list() to include new columns
```

### Add Process Tree View

Replace DataTable with Tree widget to show parent-child process relationships:

```python
from textual.widgets import Tree

def build_process_tree(self):
    tree = self.query_one(Tree)
    tree.clear()

    # Build parent-child mapping
    children = {}
    for proc in psutil.process_iter(['pid', 'ppid', 'name']):
        info = proc.info
        ppid = info['ppid']
        if ppid not in children:
            children[ppid] = []
        children[ppid].append(info)

    # Recursive tree building
    def add_children(node, pid):
        for child in children.get(pid, []):
            child_node = node.add(f"{child['name']} ({child['pid']})")
            add_children(child_node, child['pid'])

    root = tree.root
    add_children(root, 0)  # Start from init process
```

### Add Process Sorting

```python
# Add sorting state
sort_column = reactive("pid")
sort_reverse = reactive(False)

def action_sort_by_pid(self):
    self.sort_column = "pid"
    self.refresh_processes()

def action_sort_by_name(self):
    self.sort_column = "name"
    self.refresh_processes()

def refresh_processes(self):
    # ... existing code ...

    # Sort before displaying
    processes.sort(
        key=lambda p: p.get(self.sort_column, ""),
        reverse=self.sort_reverse
    )

    # ... populate table ...
```

### Add Process Details Panel

```python
# In compose():
with Horizontal():
    yield DataTable(id="process-table")
    yield Static(id="process-details")

# On row selection:
def show_process_details(self, pid):
    details_widget = self.query_one("#process-details", Static)
    try:
        proc = psutil.Process(pid)
        details = f"""
        PID: {pid}
        Name: {proc.name()}
        Status: {proc.status()}
        CPU: {proc.cpu_percent()}%
        Memory: {proc.memory_percent():.1f}%
        Threads: {proc.num_threads()}
        Created: {proc.create_time()}
        """
        details_widget.update(details)
    except psutil.NoSuchProcess:
        details_widget.update("Process not found")
```

## Sources

**GitHub Repository:**
- [psDooM](https://github.com/koaning/psdoom) - Main repository by koaning

**Code Files Referenced:**
- [psdoom/__main__.py](https://github.com/koaning/psdoom/blob/main/psdoom/__main__.py) - Entry point
- [psdoom/app.py](https://github.com/koaning/psdoom/blob/main/psdoom/app.py) - Main application (386 lines)
- [psdoom/process_manager.py](https://github.com/koaning/psdoom/blob/main/psdoom/process_manager.py) - Process operations (99 lines)
- [pyproject.toml](https://github.com/koaning/psdoom/blob/main/pyproject.toml) - Project configuration

**Dependencies:**
- Textual 0.38.1 - TUI framework
- psutil 5.9.0 - System and process utilities

**Access Date:** 2025-11-02

## Cross-References

**Related Oracle Content:**
- `widgets/03-datatable-crud.md` - DataTable patterns
- `advanced/04-workers-async.md` - Async task patterns
- `patterns/00-async-chat-ui.md` - Long-running async operations
- `core/00-reactive-state.md` - Reactive state management
- `styling/01-css-theming.md` - Dark theme CSS patterns

**External Resources:**
- [psutil documentation](https://psutil.readthedocs.io/) - System monitoring library
- [Textual DataTable](https://textual.textualize.io/widgets/data_table/) - Official widget docs
- [Textual Notifications](https://textual.textualize.io/guide/notifications/) - Notification system
