# GitHub Tutorial: textual_tutorial by KennyVaneetvelde

## Overview

A complete, project-based tutorial demonstrating a menu-driven TUI application with multiple screens. This tutorial uses a practical example (log viewer + component showcase) to teach Textual fundamentals.

**Repository**: [KennyVaneetvelde/textual_tutorial](https://github.com/KennyVaneetvelde/textual_tutorial)

**Tutorial Approach**:
- **Project-based learning** - builds a complete working application
- **Component showcase pattern** - demonstrates widgets in context
- **Real-world use case** - log data visualization with sorting
- **Modern Python practices** - type hints, Poetry, reactive patterns

**Last Updated**: 2024-11-08 (accessed 2025-11-02)

---

## What This Tutorial Teaches

### Core Textual Concepts

1. **Multi-screen navigation** - Screen management with push/pop
2. **Custom widgets** - Building reusable components (MenuWidget)
3. **Reactive programming** - Using `reactive()` for state management
4. **Event handling** - Messages, bindings, and event routing
5. **TCSS styling** - Custom styling with Textual CSS
6. **Data visualization** - DataTable with sorting capabilities
7. **Widget composition** - Containers, layouts, and organization

### Application Features

**Three main screens:**
1. **Main Menu** - Custom menu widget with keyboard navigation
2. **Component Showcase** - Interactive widget demonstrations
3. **Log Viewer** - Sortable data table with color-coded levels

---

## Project Structure

```
textual_tutorial/
├── app.py                     # Main entry point
├── components/
│   └── menu.py               # Custom MenuWidget
├── screens/
│   ├── log_screen.py         # Log data visualization
│   └── showcase_screen.py    # Component demonstrations
├── styles.tcss               # Textual CSS styling
└── logs/
    └── demo_log.csv          # Sample log data
```

**Architecture Pattern**:
- **App** → manages screens
- **MainScreen** → contains menu
- **Feature Screens** → log viewer, component showcase
- **Custom Widgets** → reusable menu component

---

## Key Code Examples

### 1. Main Application Structure

From [app.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/app.py):

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.containers import Container
from textual.screen import Screen

class MainScreen(Screen):
    """Main application screen containing the menu."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header(show_clock=True)
        yield Container(MenuWidget(MENU_OPTIONS), id="menu-container")
        yield Footer()

    def on_menu_widget_item_selected(self, message: MenuWidget.ItemSelected) -> None:
        """Handle menu item selection."""
        selected_option = MENU_OPTIONS[message.index]

        if selected_option.action == "exit":
            self.app.exit()
        elif selected_option.action == "open_logs":
            self.app.push_screen(LogScreen())
        elif selected_option.action == "component_showcase":
            self.app.push_screen(ShowcaseScreen())

class MenuApp(App[None]):
    """A Textual app to manage several screens."""

    CSS_PATH = "styles.tcss"
    TITLE = "Menu Demo"
    SUB_TITLE = "Select an option"

    def on_mount(self) -> None:
        """Mount the main screen when the app starts."""
        self.push_screen(MainScreen())
```

**Teaching Points**:
- Screen-based architecture with `push_screen()` / `pop_screen()`
- Message passing between widgets and screens
- External CSS file loading with `CSS_PATH`
- Application lifecycle (`on_mount()`)

---

### 2. Custom Widget: MenuWidget

From [components/menu.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/components/menu.py):

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from textual.reactive import reactive
from textual.widget import Widget
from textual.message import Message
from textual.binding import Binding

@dataclass
class MenuOption:
    """Dataclass representing a menu option."""
    label: str
    action: str
    params: Optional[Dict[str, Any]] = None

MENU_OPTIONS: List[MenuOption] = [
    MenuOption("Component Showcase", "component_showcase"),
    MenuOption("Open Logs", "open_logs"),
    MenuOption("Quit", "exit"),
]

class MenuWidget(Widget):
    """A widget that displays a selectable menu."""

    class ItemSelected(Message):
        """Emitted when an item is selected."""
        def __init__(self, index: int):
            self.index = index
            super().__init__()

    _selected_index = reactive(0)

    BINDINGS = [
        Binding("enter", "select", "Select item", priority=True),
        Binding("up", "move_up", "Move up"),
        Binding("down", "move_down", "Move down"),
    ]

    def __init__(self, menu_items: List[MenuOption]):
        super().__init__()
        self._menu_items = menu_items
        self.can_focus = True

    def render(self) -> str:
        """Render the menu items with the current selection highlighted."""
        rendered_menu_items = []
        for index, item in enumerate(self._menu_items):
            is_selected = index == self._selected_index
            menu_text = (
                f"[blue bold][ {item.label} ][/blue bold]"
                if is_selected
                else f"  {item.label}  "
            )
            rendered_menu_items.append(f"[center]{menu_text}[/center]")

        return "\n".join(rendered_menu_items)

    def action_move_up(self) -> None:
        """Move the selection up."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move the selection down."""
        self._move_selection(1)

    def action_select(self) -> None:
        """Handle the selection of a menu item."""
        self.post_message(self.ItemSelected(self._selected_index))

    def _move_selection(self, direction: int) -> None:
        """Move the selection up or down, wrapping around if necessary."""
        self._selected_index = (self._selected_index + direction) % len(
            self._menu_items
        )
```

**Teaching Points**:
- Creating custom widgets by subclassing `Widget`
- Using `reactive()` for state management (auto re-renders on change)
- Custom messages for widget-to-parent communication
- Key bindings with `BINDINGS` and action methods
- Rich text markup for styling (`[blue bold]...[/blue bold]`)
- Focus management with `can_focus`
- Modulo arithmetic for wraparound navigation

---

### 3. Component Showcase Screen

From [screens/showcase_screen.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/screens/showcase_screen.py):

```python
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Button, Static, Input,
    Switch, Label, ProgressBar,
)
from textual.reactive import reactive
import asyncio

class ShowcaseScreen(Screen):
    """A screen showcasing various Textual components."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    progress_value = reactive(0)

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header(show_clock=True)

        with Container(id="showcase-container"):
            # Basic widgets section
            yield Static("Basic Widgets", classes="section-title")
            with Horizontal(classes="widget-row"):
                yield Button("Success!", id="success-button", variant="success")
                yield Button("Warning!", id="warning-button", variant="warning")
                yield Button("Error!", id="error-button", variant="error")

            # Input section
            yield Static("Input Components", classes="section-title")
            with Horizontal(classes="widget-row"):
                with Vertical():
                    yield Label("Text Input")
                    yield Input(placeholder="Type something...", id="demo-input")
                with Vertical():
                    yield Label("Toggle Switch")
                    yield Switch(id="demo-switch")

            # Progress section
            yield Static("Progress Bar", classes="section-title")
            with Vertical():
                yield ProgressBar(id="demo-progress", total=100)
                with Horizontal(classes="widget-row"):
                    yield Button("Start Progress", id="start-progress", variant="success")
                    yield Button("Reset Progress", id="reset-progress", variant="warning")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "success-button":
            self.notify("Success notification!", severity="information")
        elif button_id == "warning-button":
            self.notify("Warning notification!", severity="warning")
        elif button_id == "error-button":
            self.notify("Error notification!", severity="error")
        elif button_id == "start-progress":
            self.run_worker(self.animate_progress())
        elif button_id == "reset-progress":
            self.progress_value = 0
            progress_bar = self.query_one("#demo-progress", ProgressBar)
            progress_bar.progress = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "demo-input" and event.value:
            self.notify(f"Input changed: {event.value}", severity="information")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch toggle."""
        if event.switch.id == "demo-switch":
            state = "ON" if event.value else "OFF"
            self.notify(f"Switch turned {state}!", severity="information")

    async def animate_progress(self) -> None:
        """Animate the progress bar."""
        progress_bar = self.query_one("#demo-progress", ProgressBar)

        while self.progress_value < 100:
            await asyncio.sleep(0.05)
            self.progress_value += 1
            progress_bar.progress = self.progress_value

        self.notify("Progress complete!", severity="information")
```

**Teaching Points**:
- Layout containers: `Horizontal`, `Vertical`, `Container`
- Context managers (`with`) for nested composition
- Button variants (`success`, `warning`, `error`)
- Event handlers for specific widget types (`on_button_pressed`, `on_input_changed`)
- Notifications with severity levels
- Query selectors (`query_one("#demo-progress", ProgressBar)`)
- Async workers with `run_worker()` for background tasks
- Progress bar animation

---

### 4. Data Visualization: Log Screen

From [screens/log_screen.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/screens/log_screen.py):

```python
from pathlib import Path
import csv
from datetime import datetime
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Header, Footer
from textual.containers import Container
from rich.text import Text

class LogScreen(Screen):
    """Screen displaying the log data in a sortable table."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("t", "sort_by_timestamp", "Sort by Time"),
        ("l", "sort_by_level", "Sort by Level"),
        ("m", "sort_by_message", "Sort by Message"),
        ("u", "sort_by_user", "Sort by User"),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Track sort direction for each column
        self.sort_reverse = {
            "timestamp": False,
            "level": False,
            "message": False,
            "user": False,
        }

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header(show_clock=True)
        yield Container(DataTable(id="log-table"), id="log-screen-container")
        yield Footer()

    def on_mount(self) -> None:
        """Load and display the log data when the screen is mounted."""
        table = self.query_one(DataTable)

        # Enable row selection
        table.cursor_type = "row"

        # Store column keys for sorting
        self.columns = table.add_columns("Timestamp", "Level", "Message", "User ID")

        # Read and add data from CSV
        log_path = Path("logs/demo_log.csv")
        with open(log_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse timestamp for sorting
                timestamp = datetime.strptime(
                    row["timestamp"].strip('"'), "%Y-%m-%dT%H:%M:%SZ"
                )

                # Style the level column based on log level
                level_style = {"INFO": "green", "WARN": "yellow", "ERROR": "red"}.get(
                    row["level"].strip('"'), "white"
                )

                level = Text(row["level"].strip('"'), style=level_style)

                table.add_row(
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    level,
                    row["message"].strip('"'),
                    row["user_id"].strip('"'),
                )

    def _sort_column(self, column_index: int, data_table: DataTable) -> None:
        """Helper method to sort a column by index."""
        column_keys = ["timestamp", "level", "message", "user"]
        key = column_keys[column_index]

        self.sort_reverse[key] = not self.sort_reverse[key]

        data_table.sort(
            self.columns[column_index],
            key=lambda x: str(x),
            reverse=self.sort_reverse[key]
        )

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Handle column header clicks for sorting."""
        self._sort_column(event.column_index, event.data_table)

    def action_sort_by_timestamp(self) -> None:
        """Handle the 't' key press to sort by timestamp."""
        self._sort_column(0, self.query_one(DataTable))

    def action_sort_by_level(self) -> None:
        """Handle the 'l' key press to sort by level."""
        self._sort_column(1, self.query_one(DataTable))

    def action_sort_by_message(self) -> None:
        """Handle the 'm' key press to sort by message."""
        self._sort_column(2, self.query_one(DataTable))

    def action_sort_by_user(self) -> None:
        """Handle the 'u' key press to sort by user."""
        self._sort_column(3, self.query_one(DataTable))
```

**Teaching Points**:
- `DataTable` widget for tabular data
- CSV file reading and data loading
- Rich `Text()` objects for styled cell content
- Column-based sorting with toggle direction
- Multiple event handlers for same widget type
- Keyboard shortcuts for data operations
- Interactive table features (cursor, row selection, header clicks)
- State tracking across sort operations

---

### 5. TCSS Styling

From [styles.tcss](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/styles.tcss):

```css
#menu-container {
    width: 100%;
    height: 100%;
    align: center middle;
}

MenuWidget {
    width: auto;
    height: auto;
    border: solid $primary;
    padding: 1 2;
}

#log-screen-container {
    width: 100%;
    height: 100%;
    align: center middle;
    padding: 1;
}

.section-title {
    text-style: bold;
    background: $accent;
    color: $text;
    padding: 1 2;
    margin: 1 0;
    text-align: center;
}

.widget-row {
    height: auto;
    align: center middle;
    margin: 1 0;
    padding: 1;
}

Button {
    margin: 1 2;
}

Input {
    width: 30;
}

ProgressBar {
    width: 75%;
    margin: 1 0;
}

DataTable {
    border: solid $primary;
    height: 100%;
    margin: 1;
}
```

**Teaching Points**:
- ID selectors (`#menu-container`)
- Class selectors (`.section-title`, `.widget-row`)
- Widget type selectors (`Button`, `Input`)
- CSS variables (`$primary`, `$accent`, `$text`)
- Layout properties (`align`, `width`, `height`)
- Spacing (`padding`, `margin`)
- Borders and styling
- Text alignment and styling

---

## Installation & Running

**Requirements**: Python 3.9+, Poetry

```bash
# Install poetry if needed
pip install poetry

# Install dependencies
poetry install

# Run the tutorial application
poetry run python -m textual_tutorial.app
```

---

## Navigation Reference

### Main Menu
- **↑/↓ arrows**: Navigate menu
- **Enter**: Select option
- **Options**:
  - Component Showcase
  - Log Viewer
  - Quit

### Component Showcase
- **ESC**: Return to menu
- **Buttons**: Click or tab + enter
- **Input**: Type to see live feedback
- **Switch**: Click or space to toggle
- **Progress**: Start/reset with buttons

### Log Viewer
- **ESC**: Return to menu
- **T**: Sort by timestamp
- **L**: Sort by log level
- **M**: Sort by message
- **U**: Sort by user ID
- **Click headers**: Sort by column

---

## Learning Progression

### Beginner Concepts
1. **App structure** - `App`, `Screen`, `compose()`
2. **Basic widgets** - `Header`, `Footer`, `Static`, `Button`
3. **Layouts** - `Container`, `Horizontal`, `Vertical`
4. **Event handling** - `on_button_pressed()`

### Intermediate Concepts
1. **Custom widgets** - Subclass `Widget`, implement `render()`
2. **Reactive state** - `reactive()` decorator
3. **Messages** - Custom message classes
4. **Bindings** - Keyboard shortcuts
5. **TCSS** - External styling

### Advanced Concepts
1. **Screen management** - `push_screen()`, `pop_screen()`
2. **Data tables** - CSV loading, sorting, styling
3. **Async workers** - Background tasks with `run_worker()`
4. **Query selectors** - `query_one()` for widget lookup
5. **Rich text** - `Text()` objects for inline styling

---

## Patterns Demonstrated

### 1. Menu-Driven Navigation
```
App → MainScreen (Menu) → Feature Screens
                        ↓
                   ESC to return
```

### 2. Reusable Custom Widget
```python
MenuWidget(menu_items) → Generic, data-driven
    ↓
Used in MainScreen with specific options
```

### 3. Event Flow
```
Widget Event → Handler Method → State Change → Re-render
    ↓
Button.Pressed → on_button_pressed() → Update state → UI updates
```

### 4. Separation of Concerns
- **app.py**: Application lifecycle, screen management
- **components/**: Reusable widgets
- **screens/**: Feature-specific screens
- **styles.tcss**: All styling in one place

---

## Comparison to Official Tutorial

### textual_tutorial (This Repository)
- **Project-based**: Complete working application
- **Multiple screens**: Real navigation patterns
- **Data handling**: CSV loading, sorting
- **Custom widgets**: MenuWidget example
- **Best practices**: Type hints, Poetry, modern Python

### Official Textual Tutorial
- **Concept-based**: Step-by-step feature introduction
- **Single screen**: Stopwatch application
- **Interactive**: Live code examples
- **Comprehensive**: Covers more Textual features
- **Documentation**: Integrated with docs site

**Use this tutorial for**:
- Seeing how pieces fit together in a complete app
- Learning multi-screen navigation patterns
- Understanding project structure and organization
- Data visualization examples (DataTable)

**Use official tutorial for**:
- Step-by-step introduction to concepts
- Comprehensive feature coverage
- Interactive learning experience

---

## Key Takeaways

1. **Screens are containers** - Use `push_screen()` / `pop_screen()` for navigation
2. **Reactive is powerful** - `reactive()` automatically triggers re-renders
3. **Messages connect widgets** - Custom messages for parent-child communication
4. **TCSS is CSS-like** - Familiar syntax for styling
5. **Composition over inheritance** - Build complex UIs from simple widgets
6. **Type hints help** - Makes code clearer and catches errors
7. **Async is built-in** - `run_worker()` for background tasks
8. **DataTable is feature-rich** - Sorting, styling, selection out of the box

---

## Sources

**GitHub Repository:**
- [textual_tutorial](https://github.com/KennyVaneetvelde/textual_tutorial) - Main repository

**Accessed Files** (2025-11-02):
- [README.md](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/README.md)
- [app.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/app.py)
- [components/menu.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/components/menu.py)
- [screens/showcase_screen.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/screens/showcase_screen.py)
- [screens/log_screen.py](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/screens/log_screen.py)
- [styles.tcss](https://github.com/KennyVaneetvelde/textual_tutorial/blob/main/textual_tutorial/styles.tcss)

**Repository Stats** (as of 2024-11-08):
- Stars: 4
- Last updated: 2024-11-08
- Python 3.9+, Poetry-based project
