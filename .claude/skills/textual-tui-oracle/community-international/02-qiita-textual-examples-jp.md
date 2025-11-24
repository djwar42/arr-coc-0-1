# Qiita Textual Examples - Japanese Community Tutorial
## Building Modern Terminal Apps with Python Textual

**Source**: [Qiita Article by Tadataka_Takahashi](https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4) (Japanese)
**Accessed**: 2025-11-02
**Translation**: English summary with preserved code examples

---

## Overview

This article from the Japanese developer community on Qiita demonstrates practical Textual usage through a complete task management application. The author explores Textual's capabilities for building beautiful terminal interfaces that rival GUI applications, documenting both successes and common pitfalls encountered during development.

**Key Topics Covered**:
- Basic "Hello World" example with CSS styling
- Complete task manager application with CRUD operations
- DataTable widget for displaying task lists
- Button handling and keyboard shortcuts
- Common development gotchas and solutions

---

## Introduction to Textual (Japanese Perspective)

The author describes Textual as a modern TUI framework developed by the creator of Rich (a popular terminal output library). Key characteristics highlighted:

**Visual Appeal**:
- Rich, colorful interfaces in the terminal
- CSS-like styling system
- Mouse interaction support
- Responsive layouts that adapt to terminal size

**Developer Experience**:
- Intuitive API similar to GUI frameworks
- Async/await support for non-blocking operations
- Modern Python patterns and type hints

---

## Installation

From [Qiita Article](https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4):

```bash
# Basic installation
pip install textual

# Recommended: Include development tools
pip install textual[dev]
```

**Why dev tools?**: The article recommends installing development tools for better debugging and live preview capabilities during development.

---

## Example 1: Hello World with CSS

### Python Code (hello.py)

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static

class HelloApp(App):
    """Simple Hello World app (シンプルなHello Worldアプリ)"""

    CSS_PATH = "hello.tcss"

    def compose(self) -> ComposeResult:
        """Compose the app UI (アプリのUIを構成)"""
        yield Header()
        yield Static("Hello, Textual!", id="hello")
        yield Footer()

if __name__ == "__main__":
    app = HelloApp()
    app.run()
```

### CSS Styling (hello.tcss)

```css
#hello {
    width: 100%;
    height: 100%;
    content-align: center middle;
    text-style: bold;
    color: cyan;
}
```

**Japanese Community Insight**: The author notes that CSS-like styling makes Textual approachable for developers familiar with web development, reducing the learning curve for TUI applications.

---

## Example 2: Task Manager Application

### Complete Implementation

The article presents a fully functional task management app with:
- Task creation with timestamps
- Task completion toggle
- Task deletion
- DataTable display
- Keyboard shortcuts
- Notification system

### Task Model

```python
from datetime import datetime

class Task:
    def __init__(self, title: str):
        self.title = title
        self.created_at = datetime.now()
        self.completed = False

    def toggle_complete(self):
        self.completed = not self.completed
```

**Pattern**: Simple Python class for data modeling - no complex ORM needed for TUI apps.

### Main Application Structure

```python
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, Button, DataTable, Label
from textual.binding import Binding

class TaskManagerApp(App):
    """Simple task management app (シンプルなタスク管理アプリ)"""

    BINDINGS = [
        Binding("q", "quit", "終了"),          # Quit
        Binding("a", "add_task", "タスク追加"),    # Add task
        Binding("d", "delete_task", "タスク削除"), # Delete task
        Binding("space", "toggle_task", "完了切替"), # Toggle completion
    ]

    def __init__(self):
        super().__init__()
        self.tasks = []
```

**Japanese UX Pattern**: Keyboard shortcuts with Japanese descriptions in the footer, demonstrating Textual's excellent Unicode/multi-language support.

### UI Composition

```python
def compose(self) -> ComposeResult:
    """UI composition (UIの構成)"""
    yield Header()

    # Input section (入力部分)
    yield Label("新しいタスクを追加")  # "Add new task"
    yield Input(placeholder="タスクのタイトルを入力...", id="task-input")

    with Horizontal():
        yield Button("追加", id="add-btn", variant="primary")      # Add
        yield Button("削除", id="delete-btn", variant="error")     # Delete
        yield Button("完了切替", id="toggle-btn", variant="success") # Toggle

    # Task list (タスク一覧)
    yield Label("タスク一覧")  # "Task list"
    yield DataTable(id="task-table")

    yield Footer()
```

**Layout Pattern**: Horizontal container for button grouping - common pattern for action buttons.

### DataTable Initialization

```python
def on_mount(self) -> None:
    """Initialize on app startup (アプリ起動時の初期化)"""
    table = self.query_one("#task-table", DataTable)
    table.add_columns("ID", "タイトル", "作成日時", "状態")
    # Columns: ID, Title, Created Date/Time, Status
    table.cursor_type = "row"

    # Add sample tasks (サンプルタスクを追加)
    self.add_sample_tasks()
```

**Japanese Column Headers**: Demonstrates Textual's excellent support for Japanese characters in DataTable.

### Sample Data Population

```python
def add_sample_tasks(self):
    """Add sample tasks (サンプルタスクの追加)"""
    sample_tasks = [
        "Textualの勉強",           # "Study Textual"
        "Qiita記事を書く",         # "Write Qiita article"
        "コードのリファクタリング"  # "Code refactoring"
    ]

    for task_title in sample_tasks:
        task = Task(task_title)
        self.tasks.append(task)

    self.refresh_task_table()
```

**Community Practice**: Providing sample data for immediate visual feedback - helps users understand the app instantly.

### DataTable Refresh Pattern

```python
def refresh_task_table(self):
    """Refresh task table (タスクテーブルの更新)"""
    table = self.query_one("#task-table", DataTable)
    table.clear()

    for i, task in enumerate(self.tasks):
        status = "✅ 完了" if task.completed else "⏳ 未完了"
        # Status: "✅ Completed" or "⏳ Incomplete"
        created_time = task.created_at.strftime("%m/%d %H:%M")

        table.add_row(
            str(i + 1),
            task.title,
            created_time,
            status,
            key=str(i)
        )
```

**Unicode Icons**: Japanese community commonly uses emoji for status indicators - works perfectly in Textual:
- ✅ for completed tasks
- ⏳ for pending tasks

### Button Event Handling

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle button clicks (ボタンクリック時の処理)"""
    if event.button.id == "add-btn":
        self.action_add_task()
    elif event.button.id == "delete-btn":
        self.action_delete_task()
    elif event.button.id == "toggle-btn":
        self.action_toggle_task()
```

**Pattern**: Centralized button handler dispatching to action methods.

### Input Submission Handler

```python
def on_input_submitted(self, event: Input.Submitted) -> None:
    """Handle Enter key in input field (入力フィールドでEnterが押された時)"""
    if event.input.id == "task-input":
        self.action_add_task()
```

**UX Pattern**: Enter key submission for quick task addition without clicking button.

### Task CRUD Operations

```python
def action_add_task(self) -> None:
    """Add task (タスクの追加)"""
    task_input = self.query_one("#task-input", Input)
    title = task_input.value.strip()

    if title:
        task = Task(title)
        self.tasks.append(task)
        task_input.value = ""
        self.refresh_task_table()
        self.notify(f"タスク '{title}' を追加しました")
        # Notification: "Task '{title}' added"

def action_delete_task(self) -> None:
    """Delete selected task (選択されたタスクの削除)"""
    table = self.query_one("#task-table", DataTable)

    if table.cursor_row is not None and self.tasks:
        task_index = table.cursor_row
        if 0 <= task_index < len(self.tasks):
            deleted_task = self.tasks.pop(task_index)
            self.refresh_task_table()
            self.notify(f"タスク '{deleted_task.title}' を削除しました")
            # Notification: "Task '{deleted_task.title}' deleted"

def action_toggle_task(self) -> None:
    """Toggle task completion (選択されたタスクの完了状態を切り替え)"""
    table = self.query_one("#task-table", DataTable)

    if table.cursor_row is not None and self.tasks:
        task_index = table.cursor_row
        if 0 <= task_index < len(self.tasks):
            task = self.tasks[task_index]
            task.toggle_complete()
            self.refresh_task_table()
            status = "完了" if task.completed else "未完了"
            # Status: "Completed" or "Incomplete"
            self.notify(f"タスク '{task.title}' を{status}にしました")
            # Notification: "Task '{task.title}' marked as {status}"
```

**Notification Pattern**: Japanese community emphasizes user feedback through `self.notify()` for every action - enhances UX in terminal environment.

---

## Common Pitfalls (ハマったポイント)

### Issue 1: DataTable Not Displaying

**Problem**: Deep nesting of Vertical containers prevented DataTable from rendering.

```python
# ❌ DON'T: Deep nesting causes display issues
with Vertical(classes="container"):
    with Vertical(classes="input-section"):
        # ...
    with Vertical(classes="task-list"):
        yield DataTable(id="task-table")

# ✅ DO: Keep structure simple
yield Label("タスク一覧")  # "Task list"
yield DataTable(id="task-table")
```

**Japanese Developer Insight**: The author discovered that simpler layouts are more reliable than deeply nested containers.

**Recommendation**: Prioritize basic structure over complex CSS when starting out.

### Issue 2: CSS Complexity Trap

**Problem**: Overly complex CSS caused layout breakage.

```python
# ❌ DON'T: Complex CSS before functionality works
CSS = """
.container { height: 100%; }
.task-list { height: 1fr; padding: 1; }
DataTable { height: 1fr; min-height: 10; }
"""

# ✅ DO: Start without CSS, add styling incrementally
# First: Make it work
# Then: Make it pretty
```

**Development Strategy** (from article):
1. Implement functionality without CSS
2. Verify it works correctly
3. Add styling step-by-step
4. Test after each CSS addition

**Community Wisdom**: "まずはCSSなしで動作確認" (First verify it works without CSS)

---

## Platform Testing

**Testing Environment** (from article):
- **OS**: Windows 11
- **Terminal**: Command Prompt (コマンドプロンプト)
- **Result**: Full functionality with Japanese text rendering

**Japanese Community Note**: The author specifically tested on Windows Command Prompt to verify Unicode support for Japanese characters - excellent compatibility reported.

---

## Author's Impressions (作ってみた感想)

### Positive Aspects (良かった点)

From [Qiita Article](https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4):

**1. Excellent Developer Experience**:
- Intuitive API design
- Fast iteration cycle
- Clear error messages

**2. Responsive Layouts**:
- Automatic terminal size adaptation
- No manual resize handling needed
- Works across different terminal dimensions

**3. Rich UI Features**:
- Keyboard shortcuts are easy to implement
- Notification system built-in
- Beautiful, colorful buttons and tables

**4. Visual Quality**:
- Looks professional, not like typical terminal apps
- Colorful design that "映え" (stands out visually)
- Polished appearance comparable to GUI apps

---

## Japanese Community Patterns

### Unicode Excellence

The article demonstrates Textual's strong Unicode support through:
- Japanese text in all UI elements
- Emoji status indicators (✅ ⏳)
- Mixed English/Japanese in code comments
- Multi-byte character rendering in DataTable

### User Feedback Culture

Japanese developers emphasize notifications:
```python
self.notify(f"タスク '{title}' を追加しました")
```

Every action provides immediate visual feedback through the notification system - cultural emphasis on user confirmation.

### Documentation Style

The code includes bilingual comments:
```python
def compose(self) -> ComposeResult:
    """Compose the app UI (アプリのUIを構成)"""
```

This pattern helps both Japanese and international developers understand the code.

---

## Conclusion (まとめ)

From the author's summary:

**Textual's Position**: Bridges the gap between traditional CLI tools and GUI applications - a fascinating middle ground.

**Key Strength**: Provides intuitive, rich UIs while maintaining terminal-based operation - revolutionary for terminal applications.

**Current State**: Still evolving, but extremely valuable for developers who want to build terminal apps in Python.

**Recommended Use Cases**:
- Personal productivity tools
- Internal developer utilities
- LLM integration apps (mentioned specifically for Japanese developers)
- Quick prototypes requiring richer UI than standard CLI

**Author's Recommendation**: "一度試してみる価値があるでしょう" (Worth trying at least once) - especially for Python developers who frequently build personal tools.

---

## Cross-References to Oracle Content

**Related Oracle Files**:
- [01-core-concepts.md](../fundamentals/01-core-concepts.md) - App, ComposeResult patterns
- [02-widgets-overview.md](../fundamentals/02-widgets-overview.md) - Input, Button, DataTable widgets
- [06-layout-system.md](../core/06-layout-system.md) - Horizontal container usage
- [04-event-handling.md](../core/04-event-handling.md) - Button.Pressed, Input.Submitted events
- [05-notifications.md](../advanced/05-notifications.md) - Notification system usage

**Comparison with Other Tutorials**:
- Similar to [05-contact-book-sqlite.md](../tutorials/05-contact-book-sqlite.md) - DataTable CRUD patterns
- Simpler than [06-todo-app-complete.md](../tutorials/06-todo-app-complete.md) - More basic implementation
- Demonstrates patterns from [00-zenn-textual-intro-jp.md](./00-zenn-textual-intro-jp.md) - Japanese community approach

---

## Key Takeaways for English Developers

**1. Simplicity Wins**: Japanese community emphasizes starting simple, avoiding CSS complexity.

**2. User Feedback**: Heavy use of notifications for action confirmation - good UX practice.

**3. Unicode Testing**: Article confirms excellent Unicode support on Windows Command Prompt.

**4. Practical Focus**: Complete working example over theoretical concepts - learn by building.

**5. Common Pitfalls**: DataTable layout issues are universal - keep nesting shallow.

---

## Sources

**Primary Source**:
- [Qiita Article: Python Textualを試してみた](https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4) - Tadataka_Takahashi (Japanese, accessed 2025-11-02)

**Author Context**:
- Published on Qiita (Japanese developer community platform)
- Tested on Windows 11 + Command Prompt
- Focus on practical task management application

**Translation Notes**:
- Japanese comments preserved in code examples
- Key phrases translated inline with context
- Cultural development patterns highlighted
- Original emoji and Unicode usage maintained

---

## Appendix: Full Application Features

**Implemented Features**:
- ✅ Task creation with timestamp
- ✅ Task deletion by selection
- ✅ Task completion toggle
- ✅ DataTable display with cursor navigation
- ✅ Keyboard shortcuts (q, a, d, space)
- ✅ Button click handling
- ✅ Enter key submission
- ✅ Notification feedback
- ✅ Sample data loading
- ✅ Unicode emoji status icons

**Not Implemented** (potential extensions):
- ❌ Persistence (tasks lost on exit)
- ❌ Task editing
- ❌ Task priority
- ❌ Task filtering/search
- ❌ Data export

**Community Suggestion**: This article serves as a starting point - readers encouraged to extend with persistence (JSON, SQLite) or additional features based on their needs.
