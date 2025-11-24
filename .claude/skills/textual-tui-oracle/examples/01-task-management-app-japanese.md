# Task Management App - Textual TUI Example (Japanese Tutorial)

**Source**: [Python Textualを試してみた - Qiita](https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4)
**Author**: @Tadataka_Takahashi
**Accessed**: 2025-11-02
**Language**: Japanese (code examples are language-agnostic)

## English Summary

This tutorial demonstrates building a complete task management application using Textual. The author explores Textual's capabilities by creating a TUI app with task creation, deletion, completion toggling, and a data table view. The article includes insights on common pitfalls (nested layout issues, CSS complexity) and development best practices.

**Key Features Demonstrated:**
- DataTable widget for task list display
- Input field with Enter key submission
- Button actions (add, delete, toggle completion)
- Keyboard shortcuts (bindings)
- Notification system
- Responsive layout
- Real-time table updates

---

## Original Tutorial Content (Japanese)

### はじめに (Introduction)

Textualは、Rich（リッチなターミナル出力ライブラリ）の開発者が手がけるモダンなTUIフレームワークです。

Textualは、色やレイアウトを自由にカスタマイズできるリッチな見た目が特徴で、CSS風のスタイリングにも対応しています。非同期処理（async/await）やマウス操作も可能で、ターミナルサイズに応じたレスポンシブな表示も実現できます。ターミナル上でありながら、直感的でモダンなUIが構築できるのが魅力です。

### インストール (Installation)

```bash
pip install textual
```

Development tools (recommended):
```bash
pip install textual[dev]
```

---

## Code Examples

### Basic Hello World

**Python code:**

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static

class HelloApp(App):
    """シンプルなHello Worldアプリ"""

    CSS_PATH = "hello.tcss"

    def compose(self) -> ComposeResult:
        """アプリのUIを構成"""
        yield Header()
        yield Static("Hello, Textual!", id="hello")
        yield Footer()

if __name__ == "__main__":
    app = HelloApp()
    app.run()
```

**CSS file (hello.tcss):**

```css
#hello {
    width: 100%;
    height: 100%;
    content-align: center middle;
    text-style: bold;
    color: cyan;
}
```

---

## Task Manager Application

### Complete Implementation

```python
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, Button, DataTable, Label
from textual.binding import Binding

class Task:
    def __init__(self, title: str):
        self.title = title
        self.created_at = datetime.now()
        self.completed = False

    def toggle_complete(self):
        self.completed = not self.completed

class TaskManagerApp(App):
    """シンプルなタスク管理アプリ"""

    BINDINGS = [
        Binding("q", "quit", "終了"),
        Binding("a", "add_task", "タスク追加"),
        Binding("d", "delete_task", "タスク削除"),
        Binding("space", "toggle_task", "完了切替"),
    ]

    def __init__(self):
        super().__init__()
        self.tasks = []

    def compose(self) -> ComposeResult:
        """UIの構成"""
        yield Header()

        # 入力部分
        yield Label("新しいタスクを追加")
        yield Input(placeholder="タスクのタイトルを入力...", id="task-input")
        with Horizontal():
            yield Button("追加", id="add-btn", variant="primary")
            yield Button("削除", id="delete-btn", variant="error")
            yield Button("完了切替", id="toggle-btn", variant="success")

        # タスク一覧
        yield Label("タスク一覧")
        yield DataTable(id="task-table")

        yield Footer()

    def on_mount(self) -> None:
        """アプリ起動時の初期化"""
        table = self.query_one("#task-table", DataTable)
        table.add_columns("ID", "タイトル", "作成日時", "状態")
        table.cursor_type = "row"

        # サンプルタスクを追加
        self.add_sample_tasks()

    def add_sample_tasks(self):
        """サンプルタスクの追加"""
        sample_tasks = ["Textualの勉強", "Qiita記事を書く", "コードのリファクタリング"]

        for task_title in sample_tasks:
            task = Task(task_title)
            self.tasks.append(task)

        self.refresh_task_table()

    def refresh_task_table(self):
        """タスクテーブルの更新"""
        table = self.query_one("#task-table", DataTable)
        table.clear()

        for i, task in enumerate(self.tasks):
            status = "✅ 完了" if task.completed else "⏳ 未完了"
            created_time = task.created_at.strftime("%m/%d %H:%M")

            table.add_row(
                str(i + 1),
                task.title,
                created_time,
                status,
                key=str(i)
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """ボタンクリック時の処理"""
        if event.button.id == "add-btn":
            self.action_add_task()
        elif event.button.id == "delete-btn":
            self.action_delete_task()
        elif event.button.id == "toggle-btn":
            self.action_toggle_task()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """入力フィールドでEnterが押された時"""
        if event.input.id == "task-input":
            self.action_add_task()

    def action_add_task(self) -> None:
        """タスクの追加"""
        task_input = self.query_one("#task-input", Input)
        title = task_input.value.strip()

        if title:
            task = Task(title)
            self.tasks.append(task)
            task_input.value = ""
            self.refresh_task_table()
            self.notify(f"タスク '{title}' を追加しました")

    def action_delete_task(self) -> None:
        """選択されたタスクの削除"""
        table = self.query_one("#task-table", DataTable)

        if table.cursor_row is not None and self.tasks:
            task_index = table.cursor_row
            if 0 <= task_index < len(self.tasks):
                deleted_task = self.tasks.pop(task_index)
                self.refresh_task_table()
                self.notify(f"タスク '{deleted_task.title}' を削除しました")

    def action_toggle_task(self) -> None:
        """選択されたタスクの完了状態を切り替え"""
        table = self.query_one("#task-table", DataTable)

        if table.cursor_row is not None and self.tasks:
            task_index = table.cursor_row
            if 0 <= task_index < len(self.tasks):
                task = self.tasks[task_index]
                task.toggle_complete()
                self.refresh_task_table()
                status = "完了" if task.completed else "未完了"
                self.notify(f"タスク '{task.title}' を{status}にしました")

if __name__ == "__main__":
    app = TaskManagerApp()
    app.run()
```

---

## Key Patterns Demonstrated

### 1. Task Data Model

```python
class Task:
    def __init__(self, title: str):
        self.title = title
        self.created_at = datetime.now()
        self.completed = False

    def toggle_complete(self):
        self.completed = not self.completed
```

**Pattern**: Simple data class with state management

### 2. Keyboard Bindings

```python
BINDINGS = [
    Binding("q", "quit", "終了"),
    Binding("a", "add_task", "タスク追加"),
    Binding("d", "delete_task", "タスク削除"),
    Binding("space", "toggle_task", "完了切替"),
]
```

**Pattern**: Declarative keyboard shortcuts mapped to action methods

### 3. DataTable Configuration

```python
def on_mount(self) -> None:
    """アプリ起動時の初期化"""
    table = self.query_one("#task-table", DataTable)
    table.add_columns("ID", "タイトル", "作成日時", "状態")
    table.cursor_type = "row"
```

**Pattern**: Initialize table structure in `on_mount()` lifecycle hook

### 4. Table Refresh Pattern

```python
def refresh_task_table(self):
    """タスクテーブルの更新"""
    table = self.query_one("#task-table", DataTable)
    table.clear()

    for i, task in enumerate(self.tasks):
        status = "✅ 完了" if task.completed else "⏳ 未完了"
        created_time = task.created_at.strftime("%m/%d %H:%M")

        table.add_row(
            str(i + 1),
            task.title,
            created_time,
            status,
            key=str(i)
        )
```

**Pattern**: Clear and rebuild table - simple but effective for small datasets

### 5. Event Handling

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    """ボタンクリック時の処理"""
    if event.button.id == "add-btn":
        self.action_add_task()
    elif event.button.id == "delete-btn":
        self.action_delete_task()
    elif event.button.id == "toggle-btn":
        self.action_toggle_task()

def on_input_submitted(self, event: Input.Submitted) -> None:
    """入力フィールドでEnterが押された時"""
    if event.input.id == "task-input":
        self.action_add_task()
```

**Pattern**: Event handlers dispatch to action methods

### 6. Action Methods

```python
def action_add_task(self) -> None:
    """タスクの追加"""
    task_input = self.query_one("#task-input", Input)
    title = task_input.value.strip()

    if title:
        task = Task(title)
        self.tasks.append(task)
        task_input.value = ""
        self.refresh_task_table()
        self.notify(f"タスク '{title}' を追加しました")
```

**Pattern**: Actions handle both keyboard shortcuts and button clicks

---

## Common Pitfalls (ハマったポイント)

### 1. DataTable Not Displaying

**Problem**: Deep nesting of Vertical containers can hide DataTable

```python
# ❌ これだと表示されない場合がある
with Vertical(classes="container"):
    with Vertical(classes="input-section"):
        # ...
    with Vertical(classes="task-list"):
        yield DataTable(id="task-table")

# ✅ シンプルな構造にすると安定
yield Label("タスク一覧")
yield DataTable(id="task-table")
```

**Solution**: Keep layout structure simple, avoid deep nesting

### 2. CSS Complexity

**Problem**: Overly complex CSS causes layout issues

```python
# ❌ 複雑すぎるCSS
CSS = """
.container { height: 100%; }
.task-list { height: 1fr; padding: 1; }
DataTable { height: 1fr; min-height: 10; }
"""

# ✅ まずはCSSなしで動作確認
# 動いたら少しずつスタイルを追加
```

**Solution**: Start without CSS, add styling incrementally after functionality works

---

## Author Insights (作ってみた感想)

### Positive Points (良かった点)

From the original article:

1. **優れた開発体験** - Excellent developer experience
2. **レスポンシブレイアウト自動適用** - Automatic responsive layouts
3. **リッチなUI機能が簡単実装** - Easy implementation of rich UI features (keyboard shortcuts, notifications)
4. **美しい見た目** - Beautiful appearance with colorful buttons and tables
5. **洗練されたデザイン** - Polished design that doesn't look like a terminal app

### Testing Environment

- **OS**: Windows 11
- **Terminal**: Command Prompt (コマンドプロンプト)
- **Result**: Works successfully

---

## Conclusion (まとめ)

From the original article:

> Textualは、従来のCLIツールとGUIアプリの中間的な位置づけとして、とても興味深いフレームワークでした。特に、ターミナル上で動作しながらも直感的でリッチなUIを提供できる点は非常に画期的です。

**Translation**: Textual is a very interesting framework positioned between traditional CLI tools and GUI apps. Being able to provide an intuitive, rich UI while running in a terminal is particularly groundbreaking.

**Use Cases Mentioned:**
- Personal tools in Python
- LLM integration apps
- Development utilities

---

## Related Documentation

See also:
- [../architecture/04-reactive-widgets-state.md](../architecture/04-reactive-widgets-state.md) - Reactive state management patterns
- [../architecture/05-event-system-message-flow.md](../architecture/05-event-system-message-flow.md) - Event handling architecture
- [../widgets/05-datatable-widget.md](../widgets/05-datatable-widget.md) - DataTable widget reference
- [../widgets/09-input-widget.md](../widgets/09-input-widget.md) - Input widget patterns

---

## Sources

**Primary Source:**
- [Python Textualを試してみた TUIフレームワークでモダンなターミナルアプリを作る](https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4) - Qiita article by @Tadataka_Takahashi (accessed 2025-11-02)

**Language Note**: Original content in Japanese, code examples are universal
