# ToDo TUI Application with Textual

## Overview

A comprehensive guide to building a terminal-based ToDo application using Textual, covering task management, list widgets, state persistence, and CRUD operations with an intuitive TUI interface.

**Key Features:**
- Task creation and management
- List-based UI with interactive widgets
- Task completion tracking
- State persistence
- Clean CRUD operations
- Responsive terminal interface

**Complexity**: Intermediate
**Use Case**: Personal task management, project tracking, command-line productivity tools

---

## Architecture Overview

### Core Components

**1. Application Structure:**
```
TodoApp (Main App)
├── TaskList Widget (ListView/OptionList)
├── Input Widget (Input field)
├── Button Actions (Add/Delete/Complete)
└── State Manager (Task data)
```

**2. Data Model:**
```python
# Task representation
class Task:
    def __init__(self, id: int, description: str, completed: bool = False):
        self.id = id
        self.description = description
        self.completed = completed
```

**3. Widget Hierarchy:**
- Container: Vertical layout
- Header: Title and instructions
- TaskList: ScrollableContainer with task items
- Input: New task entry field
- Actions: Button group (add, delete, mark complete)

---

## Implementation Guide

### Step 1: Basic App Structure

**Create the main application class:**
```python
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, Input, Button, ListView, ListItem, Label

class TodoApp(App):
    """A TUI Todo application."""

    CSS_PATH = "todo.tcss"
    TITLE = "Todo TUI"

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.next_id = 1

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Container(
            Static("My Tasks", id="title"),
            Input(placeholder="Enter new task...", id="task-input"),
            Button("Add Task", id="add-btn", variant="primary"),
            ListView(id="task-list"),
            Button("Delete Selected", id="delete-btn", variant="error"),
            Button("Toggle Complete", id="toggle-btn", variant="success"),
            id="main-container"
        )
        yield Footer()
```

**Key Concepts:**
- `CSS_PATH`: External stylesheet for styling
- `tasks`: List to store task data
- `next_id`: Auto-incrementing task identifier
- `compose()`: Defines widget tree

---

### Step 2: Task List Management

**Displaying tasks:**
```python
def update_task_list(self) -> None:
    """Refresh the task list display."""
    task_list = self.query_one("#task-list", ListView)
    task_list.clear()

    for task in self.tasks:
        # Create visual indicator for completion
        status = "✓" if task["completed"] else "○"
        text_style = "strike" if task["completed"] else ""

        # Create list item with styled label
        item = ListItem(
            Label(f"{status} {task['description']}", classes=text_style)
        )
        item.task_id = task["id"]  # Store reference
        task_list.append(item)
```

**Pattern Highlights:**
- Dynamic list updates using `clear()` and `append()`
- Visual completion indicators (✓/○)
- Strike-through styling for completed tasks
- Storing task ID on ListItem for reference

---

### Step 3: Adding Tasks

**Input handling and task creation:**
```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle button press events."""
    if event.button.id == "add-btn":
        self.add_task()

def add_task(self) -> None:
    """Add a new task from input field."""
    input_widget = self.query_one("#task-input", Input)
    description = input_widget.value.strip()

    # Validate input
    if not description:
        self.notify("Task description cannot be empty", severity="warning")
        return

    # Create task
    new_task = {
        "id": self.next_id,
        "description": description,
        "completed": False
    }
    self.tasks.append(new_task)
    self.next_id += 1

    # Update UI
    input_widget.value = ""  # Clear input
    self.update_task_list()
    self.notify(f"Added: {description}", severity="information")
```

**Best Practices:**
- Input validation before task creation
- User feedback via notifications
- Clear input field after adding
- Auto-increment ID generation

---

### Step 4: Deleting Tasks

**Selected task deletion:**
```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle button press events."""
    if event.button.id == "delete-btn":
        self.delete_task()

def delete_task(self) -> None:
    """Delete the currently selected task."""
    task_list = self.query_one("#task-list", ListView)

    # Check if task is selected
    if task_list.index is None:
        self.notify("No task selected", severity="warning")
        return

    # Get selected item
    selected_item = task_list.highlighted_child
    if selected_item is None:
        return

    task_id = selected_item.task_id

    # Remove from data
    self.tasks = [t for t in self.tasks if t["id"] != task_id]

    # Update UI
    self.update_task_list()
    self.notify("Task deleted", severity="information")
```

**Key Techniques:**
- Selection state checking (`index is None`)
- Retrieving highlighted item
- Filter-based deletion (list comprehension)
- UI refresh after data change

---

### Step 5: Toggling Task Completion

**Mark tasks as complete/incomplete:**
```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle button press events."""
    if event.button.id == "toggle-btn":
        self.toggle_task()

def toggle_task(self) -> None:
    """Toggle completion status of selected task."""
    task_list = self.query_one("#task-list", ListView)

    # Check selection
    if task_list.index is None:
        self.notify("No task selected", severity="warning")
        return

    selected_item = task_list.highlighted_child
    if selected_item is None:
        return

    task_id = selected_item.task_id

    # Find and toggle task
    for task in self.tasks:
        if task["id"] == task_id:
            task["completed"] = not task["completed"]
            status = "completed" if task["completed"] else "incomplete"
            self.notify(f"Task marked as {status}", severity="information")
            break

    # Update UI
    self.update_task_list()
```

**Pattern Benefits:**
- Toggle boolean state (True ↔ False)
- Contextual user feedback
- Immediate visual update

---

### Step 6: Keyboard Navigation

**Add keyboard shortcuts for efficiency:**
```python
def on_key(self, event: events.Key) -> None:
    """Handle keyboard events."""
    key = event.key

    if key == "ctrl+a":
        # Add task shortcut
        self.add_task()
        event.prevent_default()

    elif key == "ctrl+d":
        # Delete task shortcut
        self.delete_task()
        event.prevent_default()

    elif key == "ctrl+t" or key == "space":
        # Toggle completion shortcut
        self.toggle_task()
        event.prevent_default()

    elif key == "enter":
        # Add task on Enter key if input focused
        input_widget = self.query_one("#task-input", Input)
        if input_widget.has_focus:
            self.add_task()
            event.prevent_default()
```

**Keyboard Shortcuts:**
- `Ctrl+A`: Add task
- `Ctrl+D`: Delete selected task
- `Ctrl+T` or `Space`: Toggle completion
- `Enter`: Add task (when input focused)

---

### Step 7: State Persistence

**Save and load tasks to/from JSON file:**

```python
import json
from pathlib import Path

class TodoApp(App):
    """A TUI Todo application with persistence."""

    TASKS_FILE = Path.home() / ".todo_tui_tasks.json"

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.next_id = 1
        self.load_tasks()

    def load_tasks(self) -> None:
        """Load tasks from JSON file."""
        if not self.TASKS_FILE.exists():
            return

        try:
            with open(self.TASKS_FILE, 'r') as f:
                data = json.load(f)
                self.tasks = data.get("tasks", [])
                self.next_id = data.get("next_id", 1)
        except Exception as e:
            self.notify(f"Error loading tasks: {e}", severity="error")

    def save_tasks(self) -> None:
        """Save tasks to JSON file."""
        try:
            data = {
                "tasks": self.tasks,
                "next_id": self.next_id
            }
            with open(self.TASKS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.notify(f"Error saving tasks: {e}", severity="error")

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.update_task_list()

    def add_task(self) -> None:
        """Add a new task (with save)."""
        # ... existing add logic ...
        self.save_tasks()  # Persist after adding

    def delete_task(self) -> None:
        """Delete task (with save)."""
        # ... existing delete logic ...
        self.save_tasks()  # Persist after deleting

    def toggle_task(self) -> None:
        """Toggle task (with save)."""
        # ... existing toggle logic ...
        self.save_tasks()  # Persist after toggling
```

**Persistence Strategy:**
- JSON file in user's home directory
- Load on app initialization
- Save after each modification (add/delete/toggle)
- Error handling for file operations

---

### Step 8: Styling with CSS

**Create `todo.tcss` for visual polish:**

```css
/* Main container */
#main-container {
    background: $surface;
    border: solid $primary;
    padding: 1 2;
}

/* Title */
#title {
    text-align: center;
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}

/* Input field */
#task-input {
    margin-bottom: 1;
    border: solid $primary;
}

/* Task list */
#task-list {
    height: 1fr;
    border: solid $primary;
    margin-bottom: 1;
}

/* Completed tasks (strike-through) */
.strike {
    text-style: strike;
    color: $text-muted;
}

/* Buttons */
Button {
    margin-right: 1;
}

#add-btn {
    background: $success;
}

#delete-btn {
    background: $error;
}

#toggle-btn {
    background: $warning;
}

/* List items */
ListItem {
    padding: 0 1;
}

ListItem:hover {
    background: $boost;
}
```

**Styling Features:**
- Color-coded buttons (success/error/warning)
- Bordered containers
- Hover effects on list items
- Strike-through for completed tasks
- Responsive layout with fractional heights

---

## Complete Example

**Full working todo app:**

```python
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, Input, Button, ListView, ListItem, Label
from textual import events
import json
from pathlib import Path

class TodoApp(App):
    """A complete TUI Todo application."""

    CSS_PATH = "todo.tcss"
    TITLE = "Todo TUI"
    TASKS_FILE = Path.home() / ".todo_tui_tasks.json"

    BINDINGS = [
        ("ctrl+a", "add_task", "Add task"),
        ("ctrl+d", "delete_task", "Delete"),
        ("ctrl+t", "toggle_task", "Toggle"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.next_id = 1
        self.load_tasks()

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Container(
            Static("📝 My Tasks", id="title"),
            Input(placeholder="What needs to be done?", id="task-input"),
            Button("➕ Add Task", id="add-btn", variant="primary"),
            ListView(id="task-list"),
            Container(
                Button("🗑️  Delete", id="delete-btn", variant="error"),
                Button("✓ Toggle", id="toggle-btn", variant="success"),
                id="button-group"
            ),
            id="main-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize display on mount."""
        self.update_task_list()
        input_widget = self.query_one("#task-input", Input)
        input_widget.focus()

    def load_tasks(self) -> None:
        """Load tasks from JSON file."""
        if not self.TASKS_FILE.exists():
            return
        try:
            with open(self.TASKS_FILE, 'r') as f:
                data = json.load(f)
                self.tasks = data.get("tasks", [])
                self.next_id = data.get("next_id", 1)
        except Exception as e:
            pass  # Silent fail on first run

    def save_tasks(self) -> None:
        """Save tasks to JSON file."""
        try:
            data = {"tasks": self.tasks, "next_id": self.next_id}
            with open(self.TASKS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.notify(f"Save error: {e}", severity="error")

    def update_task_list(self) -> None:
        """Refresh task list display."""
        task_list = self.query_one("#task-list", ListView)
        task_list.clear()

        if not self.tasks:
            task_list.append(ListItem(Label("No tasks yet. Add one above!", classes="empty")))
            return

        for task in self.tasks:
            status = "✓" if task["completed"] else "○"
            text_style = "strike" if task["completed"] else ""
            item = ListItem(Label(f"{status} {task['description']}", classes=text_style))
            item.task_id = task["id"]
            task_list.append(item)

    def action_add_task(self) -> None:
        """Add task action."""
        self.add_task()

    def add_task(self) -> None:
        """Add new task from input."""
        input_widget = self.query_one("#task-input", Input)
        description = input_widget.value.strip()

        if not description:
            self.notify("Enter a task description", severity="warning")
            return

        new_task = {
            "id": self.next_id,
            "description": description,
            "completed": False
        }
        self.tasks.append(new_task)
        self.next_id += 1

        input_widget.value = ""
        self.update_task_list()
        self.save_tasks()
        self.notify(f"✓ Added: {description}", severity="information")

    def action_delete_task(self) -> None:
        """Delete task action."""
        self.delete_task()

    def delete_task(self) -> None:
        """Delete selected task."""
        task_list = self.query_one("#task-list", ListView)

        if task_list.index is None or not self.tasks:
            self.notify("No task selected", severity="warning")
            return

        selected_item = task_list.highlighted_child
        if selected_item is None:
            return

        task_id = getattr(selected_item, 'task_id', None)
        if task_id is None:
            return

        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        self.update_task_list()
        self.save_tasks()
        self.notify("🗑️  Task deleted", severity="information")

    def action_toggle_task(self) -> None:
        """Toggle task action."""
        self.toggle_task()

    def toggle_task(self) -> None:
        """Toggle selected task completion."""
        task_list = self.query_one("#task-list", ListView)

        if task_list.index is None or not self.tasks:
            self.notify("No task selected", severity="warning")
            return

        selected_item = task_list.highlighted_child
        if selected_item is None:
            return

        task_id = getattr(selected_item, 'task_id', None)
        if task_id is None:
            return

        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = not task["completed"]
                status = "completed" if task["completed"] else "incomplete"
                self.notify(f"✓ Marked as {status}", severity="information")
                break

        self.update_task_list()
        self.save_tasks()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "add-btn":
            self.add_task()
        elif event.button.id == "delete-btn":
            self.delete_task()
        elif event.button.id == "toggle-btn":
            self.toggle_task()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        if event.input.id == "task-input":
            self.add_task()

if __name__ == "__main__":
    app = TodoApp()
    app.run()
```

---

## Advanced Features

### Task Filtering

**Add filter buttons to show All/Active/Completed:**

```python
class TodoApp(App):
    """Todo app with filtering."""

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.next_id = 1
        self.filter_mode = "all"  # all, active, completed
        self.load_tasks()

    def compose(self) -> ComposeResult:
        """Compose with filter buttons."""
        yield Header()
        yield Container(
            # ... existing widgets ...
            Container(
                Button("All", id="filter-all", classes="filter-btn"),
                Button("Active", id="filter-active", classes="filter-btn"),
                Button("Completed", id="filter-completed", classes="filter-btn"),
                id="filter-group"
            ),
            ListView(id="task-list"),
            # ... rest of UI ...
        )
        yield Footer()

    def update_task_list(self) -> None:
        """Update task list with current filter."""
        task_list = self.query_one("#task-list", ListView)
        task_list.clear()

        # Apply filter
        filtered_tasks = self.tasks
        if self.filter_mode == "active":
            filtered_tasks = [t for t in self.tasks if not t["completed"]]
        elif self.filter_mode == "completed":
            filtered_tasks = [t for t in self.tasks if t["completed"]]

        if not filtered_tasks:
            task_list.append(ListItem(Label(f"No {self.filter_mode} tasks")))
            return

        for task in filtered_tasks:
            # ... render task items ...

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle filter buttons."""
        if event.button.id == "filter-all":
            self.filter_mode = "all"
            self.update_task_list()
        elif event.button.id == "filter-active":
            self.filter_mode = "active"
            self.update_task_list()
        elif event.button.id == "filter-completed":
            self.filter_mode = "completed"
            self.update_task_list()
        # ... existing button handlers ...
```

---

### Task Categories

**Add category support for organizing tasks:**

```python
class TodoApp(App):
    """Todo app with categories."""

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.categories = ["Personal", "Work", "Shopping", "Other"]
        self.next_id = 1

    def compose(self) -> ComposeResult:
        """Compose with category selection."""
        yield Container(
            Input(placeholder="Task description...", id="task-input"),
            Select(
                [(cat, cat) for cat in self.categories],
                prompt="Category",
                id="category-select"
            ),
            Button("Add Task", id="add-btn"),
            # ... rest of UI ...
        )

    def add_task(self) -> None:
        """Add task with category."""
        input_widget = self.query_one("#task-input", Input)
        category_select = self.query_one("#category-select", Select)

        description = input_widget.value.strip()
        category = category_select.value

        if not description:
            return

        new_task = {
            "id": self.next_id,
            "description": description,
            "category": category,
            "completed": False
        }
        self.tasks.append(new_task)
        # ... rest of add logic ...

    def update_task_list(self) -> None:
        """Display tasks grouped by category."""
        task_list = self.query_one("#task-list", ListView)
        task_list.clear()

        # Group by category
        for category in self.categories:
            cat_tasks = [t for t in self.tasks if t.get("category") == category]
            if not cat_tasks:
                continue

            # Category header
            task_list.append(ListItem(Label(f"📁 {category}", classes="category-header")))

            # Tasks in category
            for task in cat_tasks:
                # ... render task items ...
```

---

### Task Priority

**Add priority levels (High/Medium/Low):**

```python
class TodoApp(App):
    """Todo app with priority."""

    def add_task(self) -> None:
        """Add task with priority."""
        # ... existing input handling ...

        priority = self.query_one("#priority-select", Select).value

        new_task = {
            "id": self.next_id,
            "description": description,
            "priority": priority,  # "high", "medium", "low"
            "completed": False
        }
        self.tasks.append(new_task)
        # ... rest of add logic ...

    def update_task_list(self) -> None:
        """Display tasks with priority indicators."""
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_tasks = sorted(
            self.tasks,
            key=lambda t: priority_order.get(t.get("priority", "low"), 2)
        )

        for task in sorted_tasks:
            priority = task.get("priority", "low")
            priority_icon = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(priority, "○")

            status = "✓" if task["completed"] else "○"
            label_text = f"{priority_icon} {status} {task['description']}"
            # ... render task item ...
```

---

## Best Practices

### State Management

**Centralize task operations:**
```python
class TaskManager:
    """Encapsulate task logic."""

    def __init__(self):
        self.tasks = []
        self.next_id = 1

    def add(self, description: str, **kwargs) -> dict:
        """Add new task."""
        task = {
            "id": self.next_id,
            "description": description,
            "completed": False,
            **kwargs
        }
        self.tasks.append(task)
        self.next_id += 1
        return task

    def delete(self, task_id: int) -> bool:
        """Delete task by ID."""
        original_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        return len(self.tasks) < original_count

    def toggle(self, task_id: int) -> bool:
        """Toggle task completion."""
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = not task["completed"]
                return task["completed"]
        return False

    def get_filtered(self, filter_mode: str) -> list:
        """Get filtered task list."""
        if filter_mode == "active":
            return [t for t in self.tasks if not t["completed"]]
        elif filter_mode == "completed":
            return [t for t in self.tasks if t["completed"]]
        return self.tasks
```

**Use in app:**
```python
class TodoApp(App):
    def __init__(self):
        super().__init__()
        self.task_manager = TaskManager()

    def add_task(self):
        task = self.task_manager.add(description, category=category)
        self.save_tasks()
        self.update_task_list()
```

---

### Error Handling

**Robust validation and error messages:**
```python
def add_task(self) -> None:
    """Add task with validation."""
    try:
        input_widget = self.query_one("#task-input", Input)
        description = input_widget.value.strip()

        # Validation
        if not description:
            raise ValueError("Task description cannot be empty")

        if len(description) > 200:
            raise ValueError("Task description too long (max 200 chars)")

        if any(t["description"] == description for t in self.tasks):
            raise ValueError("Duplicate task")

        # Add task
        new_task = {
            "id": self.next_id,
            "description": description,
            "completed": False
        }
        self.tasks.append(new_task)
        self.next_id += 1

        input_widget.value = ""
        self.update_task_list()
        self.save_tasks()
        self.notify(f"✓ Added: {description}", severity="information")

    except ValueError as e:
        self.notify(str(e), severity="error")
    except Exception as e:
        self.notify(f"Unexpected error: {e}", severity="error")
```

---

### Performance Optimization

**Efficient list updates:**
```python
def update_task_list(self) -> None:
    """Optimized task list update."""
    task_list = self.query_one("#task-list", ListView)

    # Only update if tasks changed
    if not hasattr(self, '_last_task_hash'):
        self._last_task_hash = None

    current_hash = hash(json.dumps(self.tasks, sort_keys=True))
    if current_hash == self._last_task_hash:
        return  # No changes, skip update

    self._last_task_hash = current_hash

    # Perform update
    task_list.clear()
    for task in self.tasks:
        # ... render tasks ...
```

**Lazy loading for large lists:**
```python
from textual.widgets import DataTable

class TodoApp(App):
    """Todo app with DataTable for large lists."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="task-table")

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("Status", "Description", "Category", "Priority")
        self.update_task_table()

    def update_task_table(self) -> None:
        """Update DataTable efficiently."""
        table = self.query_one("#task-table", DataTable)
        table.clear()

        for task in self.tasks:
            status = "✓" if task["completed"] else "○"
            table.add_row(
                status,
                task["description"],
                task.get("category", ""),
                task.get("priority", "")
            )
```

---

## Testing

### Unit Tests

**Test task operations:**
```python
import pytest
from todo_app import TaskManager

def test_add_task():
    """Test adding task."""
    manager = TaskManager()
    task = manager.add("Test task")

    assert task["id"] == 1
    assert task["description"] == "Test task"
    assert task["completed"] is False
    assert len(manager.tasks) == 1

def test_delete_task():
    """Test deleting task."""
    manager = TaskManager()
    task = manager.add("Test task")
    result = manager.delete(task["id"])

    assert result is True
    assert len(manager.tasks) == 0

def test_toggle_task():
    """Test toggling completion."""
    manager = TaskManager()
    task = manager.add("Test task")

    result = manager.toggle(task["id"])
    assert result is True
    assert manager.tasks[0]["completed"] is True

    result = manager.toggle(task["id"])
    assert result is False
    assert manager.tasks[0]["completed"] is False

def test_filter_tasks():
    """Test task filtering."""
    manager = TaskManager()
    manager.add("Task 1")
    manager.add("Task 2")
    manager.toggle(1)  # Mark first as complete

    active = manager.get_filtered("active")
    completed = manager.get_filtered("completed")

    assert len(active) == 1
    assert len(completed) == 1
```

---

## Sources

**Primary Reference:**
- [How to Build a ToDo TUI Application with Textual](https://pythongui.org/how-to-build-a-todo-tui-application-with-textual-2/) - PythonGUI.org (accessed 2025-11-02)

**Related Documentation:**
- Textual ListView: [widgets/06-list-widgets.md](../widgets/06-list-widgets.md)
- Input widgets: [widgets/02-input-widgets.md](../widgets/02-input-widgets.md)
- State management: [architecture/03-state-management.md](../architecture/03-state-management.md)
- File I/O patterns: [integration/01-file-io-patterns.md](../integration/01-file-io-patterns.md)

---

## See Also

- [Contact Book with SQLite](05-contact-book-sqlite.md) - Database-backed CRUD application
- [Chat UI with Async](../patterns/00-async-chat-ui.md) - Long-running processes and workers
- [DataTable Tutorial](01-datatable-basics.md) - Alternative list display for large datasets
- [CSS Styling Guide](../styling/00-css-fundamentals.md) - Visual polish techniques
