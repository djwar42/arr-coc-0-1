# Python Textual: Build Beautiful UIs in the Terminal

**Source**: [Real Python - Python Textual Tutorial](https://realpython.com/python-textual/)
**Author**: Charles de Villiers
**Date Accessed**: 2025-11-02
**Publication Date**: March 12, 2025
**Level**: Intermediate

## Overview

This comprehensive tutorial covers building text-based user interface (TUI) applications using the Python Textual framework. Textual provides a powerful and flexible framework for creating interactive and visually appealing console applications with widgets, layouts, styling options, and event handling.

## Key Concepts

**Text-Based User Interface (TUI)**: Terminal applications with interactive and visually appealing text interfaces

**Textual Framework**: Built on Rich (Python library for rich text and formatted terminal output), provides:
- Rich widget library (buttons, inputs, checkboxes, switches, etc.)
- Flexible layout management (docking, grid systems)
- Reactive attributes for dynamic interfaces
- CSS-like styling syntax
- Asynchronous event handling (`asyncio`)
- Cross-platform compatibility (Windows, macOS, Linux)
- Remote application support (SSH)

## Installation

Textual requires Python 3.8 or later.

### Setup Steps

1. **Create virtual environment**:
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

2. **Install Textual**:
```bash
pip install textual textual-dev
```

**Two packages installed**:
- `textual`: The library and application framework
- `textual-dev`: Command-line tool for debugging and interactive development (Textual console)

### Checking Installation

```bash
python -m textual
```

This launches the Textual demo app showing colored text, emojis, structured layout, and keyboard shortcuts.

## Creating Your First Textual App

### Minimal Example

**File**: `hello_textual.py`

```python
from textual.app import App
from textual.widgets import Static

class HelloTextualApp(App):
    def compose(self):
        yield Static("Hello, Textual!")

if __name__ == "__main__":
    app = HelloTextualApp()
    app.run()
```

**Key components**:
- `App` class: Represents a generic Textual application
- `Static` widget: Basic UI component managing a rectangular screen area
- `.compose()` method: Builds the application's UI (called by Textual)

**Run**:
```bash
python hello_textual.py
```

**Exit**: Press `Ctrl+Q`

## Exploring Textual Widgets

Widgets are graphical components that represent UI controls and occupy rectangular screen areas. Each widget can have its own style settings and respond to events.

### Static and Label Widgets

**`Static`**: Manages rectangular area with optional text
**`Label`**: Specialized for displaying text (similar to `Static`)

Both support **Rich renderables** - text with foreground/background colors, **bold**, _italic_, flashing effects, and emojis.

**Example**: `static_and_label.py`

```python
from textual.app import App
from textual.widgets import Label, Static

class StaticAndLabelApp(App):
    def compose(self):
        self.static = Static(
            "I am a [bold red]Static[/bold red] widget!",
        )
        yield self.static
        self.label = Label(
            "I am a [yellow italic]Label[/yellow italic] widget!",
        )
        yield self.label

    def on_mount(self):
        # Styling the static
        self.static.styles.background = "blue"
        self.static.styles.border = ("solid", "white")
        self.static.styles.text_align = "center"
        self.static.styles.padding = 1, 1
        self.static.styles.margin = 4, 4
        # Styling the label
        self.label.styles.background = "darkgreen"
        self.label.styles.border = ("double", "red")
        self.label.styles.padding = 1, 1
        self.label.styles.margin = 2, 4

if __name__ == "__main__":
    app = StaticAndLabelApp()
    app.run()
```

**Key method**:
- `.on_mount()`: Event handler called before screen is rendered (DOM is fully constructed)

**Behavior differences**:
- `Static`: Expands horizontally to fill screen by default
- `Label`: Adjusts to content's width by default (`width: auto`)

### Button and Input Widgets

**Example**: `buttons_and_inputs.py`

```python
from textual.app import App
from textual.widgets import Button, Input

class ButtonsAndInputsApp(App):
    def compose(self):
        # Buttons
        yield Button("Click me!")
        yield Button("Primary!", variant="primary")
        yield Button.success("Success!")
        yield Button.warning("Warning!")
        yield Button.error("Error!")
        # Inputs
        yield Input(placeholder="Type your text here")
        yield Input(placeholder="Password", password=True)
        yield Input(
            placeholder="Type a number here",
            type="number",
            tooltip="Digits only please!",
        )

if __name__ == "__main__":
    app = ButtonsAndInputsApp()
    app.run()
```

**Button variants**:
- Default: Drab black-and-white
- `variant="primary"`: Primary color scheme
- `.success()`: Success variant (green)
- `.warning()`: Warning variant (yellow)
- `.error()`: Error variant (red)

**Input parameters**:
- `placeholder`: Prompt text when empty
- `password=True`: Hides typed text
- `type="number"`: Accepts only numeric input
- `tooltip`: Pop-up label on hover

**Themes**: Variants map to colors via Textual themes - users can change colors via command palette (`Ctrl+P`)

### Advanced Widgets

**Available widgets**:
- `Tree`: Collapsible hierarchical display
- `Sparkline`: Compact histogram
- `Switch`: On/off switch
- `Pretty`: Pretty-printed Python object
- `TabbedContent`: Multi-page navigation
- Classic UI: Labels, buttons, radio buttons, text areas

See [Textual widgets documentation](https://textual.textualize.io/widgets/) for complete list.

## Styling with Textual CSS (TCSS)

TCSS implements a simplified subset of CSS for web pages. Styles are specified in a `.tcss` file (separate from Python code).

### Using External Styles

**Python file**: `static_and_label_tcss.py`

```python
from textual.app import App
from textual.widgets import Static, Label

class StaticAndLabelAppWithTCSS(App):
    CSS_PATH = "static_and_label.tcss"
    def compose(self):
        yield Static(
            "I am a [bold red]Static[/bold red] widget!",
            )
        yield Label(
            "I am a [yellow italic]Label[/yellow italic] widget with an id!",
            id="label_id",
        )
        yield Label(
            "I am a [yellow italic]Label[/yellow italic] widget with a CSS class!",
            classes="label_class",
        )

if __name__ == "__main__":
    app = StaticAndLabelAppWithTCSS()
    app.run()
```

**TCSS file**: `static_and_label.tcss`

```css
Static {
    background: blue;
    border: solid white;
    padding: 1 1;
    margin: 2 2;
    text-align: center;
}

#label_id {
    color: black;
    background: red;
    border: solid black;
    padding: 1 1;
    margin: 2 4;
}

.label_class {
    color: black;
    background: green;
    border: dashed purple;
    padding: 1 1;
    margin: 2 6;
}
```

### TCSS Selectors

**Three ways to select widgets**:

1. **Python class name**: `Static { ... }` - Applies to all `Static` widgets
2. **ID**: `#label_id { ... }` - Unique identifier assigned at construction
3. **TCSS class**: `.label_class { ... }` - Can be applied to multiple widgets

**TCSS classes**:
- Not related to Python classes
- Identifiers for grouping and organizing styles
- Can be dynamically added/removed
- Used as search keys in DOM queries

**Benefits**:
- Separates visual design from Python logic
- Compact syntax
- Easier to maintain and modify styles

See [Textual TCSS documentation](https://textual.textualize.io/guide/CSS/) for complete reference.

## Textual Development Tools

The `textual-dev` package provides the Textual console for convenient development workflow.

### Using Textual Console

**Start console** (in one terminal):
```bash
textual console
```

Shows debugging output including:
- Log messages
- Errors
- `print()` call output
- Events flowing in real-time

**Run app** (in another terminal):
```bash
textual run --dev static_and_label_tcss.py
```

**Benefits**:
- Live reload: TCSS changes appear immediately without restart
- Colorized, formatted event logs
- Debug output in separate window
- Click events visible in log

### Web Server Mode

**Serve app as web page**:
```bash
textual serve static_and_label_tcss.py
```

Open browser at `http://localhost:8000`

**Options**:
- `--host=<hostname>`: Specify server host
- `--port=<port>`: Specify server port

## Laying Out UIs with Containers

Containers help arrange widgets in visual hierarchies. Implemented as context managers (use with `with` statement).

### Vertical Containers

**Example**: `vertical_layout.py`

```python
from textual.app import App
from textual.containers import Vertical
from textual.widgets import Static

NUM_BOXES = 4

class VerticalLayoutApp(App):
    def compose(self):
        with Vertical():
            for i in range(NUM_BOXES):
                static = Static(f"Static {i + 1}")
                static.styles.border = ("solid", "green")
                yield static

if __name__ == "__main__":
    app = VerticalLayoutApp()
    app.run()
```

**Note**: `Vertical` is the default container - you could omit `with Vertical()` and get the same result.

### Vertical Scrolling

For many widgets (e.g., `NUM_BOXES = 20`), use `VerticalScroll`:

**Example**: `vertical_scroll.py`

```python
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Static

NUM_BOXES = 20

class VerticalScrollApp(App):
    CSS_PATH="vertical_layout.tcss"

    def compose(self):
        with VerticalScroll():
           for i in range(NUM_BOXES):
                yield Static(f"Static {i + 1}")

if __name__ == "__main__":
    app = VerticalScrollApp()
    app.run()
```

**TCSS file**: `vertical_layout.tcss`

```css
Static {
    border: solid green;
}
```

**Features**:
- Scrollbar on right side
- Scroll with mouse gestures, scroll wheel, or arrow keys
- Accommodates unlimited widgets

### Horizontal Containers

**Example**: `horizontal_layout.py`

```python
from textual.app import App
from textual.containers import Horizontal
from textual.widgets import Static

NUM_BOXES = 4

class HorizontalLayoutApp(App):
    def compose(self):
        with Horizontal():
            for i in range(NUM_BOXES):
                static = Static(f"Static {i + 1}")
                static.styles.border = ("solid", "green")
                static.styles.width = "10%"
                yield static

if __name__ == "__main__":
    app = HorizontalLayoutApp()
    app.run()
```

**Important**: Must specify `width` for each widget, otherwise first widget claims all horizontal space.

### Horizontal Scrolling

For overflow, use `HorizontalScroll`:

**Example**: `horizontal_scroll.py`

```python
from textual.app import App
from textual.containers import HorizontalScroll
from textual.widgets import Static

NUM_BOXES = 20

class HorizontalScrollApp(App):
    def compose(self):
        with HorizontalScroll():
           for i in range(NUM_BOXES):
                static = Static(f"Static {i + 1}")
                static.styles.border = ("solid", "green")
                static.styles.width = "10%"
                yield static

if __name__ == "__main__":
    app = HorizontalScrollApp()
    app.run()
```

**Features**:
- Scrollbar at bottom of screen
- Scroll with mouse or Left/Right arrow keys

### Docked Widgets

**Dock**: Not a container-based layout option - docked widgets ignore parent container layout rules.

**Dock positions**: `top`, `left`, `bottom`, `right` within container

Container's other widgets still follow layout rules (may overlap docked widget).

### Nested Containers and Docked Widgets

**Example**: `layouts.py`

```python
from textual.app import App
from textual.containers import (
    Horizontal,
    HorizontalScroll,
    VerticalScroll,
)
from textual.widgets import Label, Static

NUM_BOXES = 12

class NestedContainersApp(App):
    CSS_PATH = "layouts.tcss"

    def compose(self):
        with Horizontal(id="horizontal"):
           yield Static("Left", classes="box")
            with HorizontalScroll(id="horizontalscroll"):
               for i in range(NUM_BOXES):
                    yield Static(
                        f"Center.{i + 1}",
                        classes="box yellowbox",
                    )
            with VerticalScroll(id="verticalscroll"):
               for i in range(NUM_BOXES):
                    yield Static(
                        f"Right.{i + 1}",
                        classes="box redbox",
                    )
                yield Label(
                   "I am a docked label.\nI don't move!",
                   id="docked-label",
               )

if __name__ == "__main__":
    app = NestedContainersApp()
    app.run()
```

**TCSS file**: `layouts.tcss`

```css
.box {
    height: 1fr;
    width: 1fr;
    background: $panel;
    border: solid white;
}

.redbox {
    border: heavy red;
    height: 5;
}

.yellowbox {
    border: heavy yellow;
    width: 10;
}

#docked-label {
    dock: bottom;
    border: solid dodgerblue;
}
```

**Fractional units (`fr`)**:
- Represents fraction of available space
- All widgets with `1fr` occupy same space

**Structure**:
- Overall: `Horizontal` container
- Left: Single `Static` widget
- Center: `HorizontalScroll` with 12 widgets
- Right: `VerticalScroll` with 12 widgets + docked label

**Docked label**:
- Fixed at bottom of `VerticalScroll` container
- Independent of container's layout
- Other widgets scroll past it

### Grid Container

**Grid**: Specifies rows and columns. Widgets auto-allocated to grid positions (left-to-right, top-to-bottom).

**Example**: `grid.py`

```python
from textual.app import App
from textual.containers import Grid
from textual.widgets import Static

class GridLayoutApp(App):
    def compose(self):
        grid = Grid()
        grid.styles.grid_size_rows = rows = 6
        grid.styles.grid_size_columns = cols = 4
        with grid:
            for row in range(rows):
                for col in range(cols):
                    static = Static(f"Static ({row=}, {col=})")
                    static.styles.border = ("solid", "green")
                    static.styles.width = "1fr"
                    static.styles.height = "1fr"
                    yield static

if __name__ == "__main__":
    app = GridLayoutApp()
    app.run()
```

**Grid customization options**:
- Row heights and column widths
- Cell spanning (multiple rows/columns)
- Gutters between rows/columns
- Individual cell colors, frames, content

### Grid with TCSS

**Python file**: `grid_tcss.py`

```python
from textual.app import App
from textual.containers import Grid
from textual.widgets import Static

class GridLayoutWithTCSS(App):
    CSS_PATH = "grid.tcss"
    def compose(self):
        with Grid():
            for row in range(6):
                for col in range(4):
                    yield Static(f"Static ({row=}, {col=})")

if __name__ == "__main__":
    app = GridLayoutWithTCSS()
    app.run()
```

**TCSS file**: `grid.tcss`

```css
Grid {
    grid_size: 4 6;
}

Static {
    height: 1fr;
    width: 1fr;
    border: solid green;
}
```

**Benefit**: TCSS simplifies layout configuration and styling.

## Handling Events and Actions

### Two Interaction Mechanisms

**1. Events**: Triggered by user interactions (mouse clicks, keystrokes), timer ticks, or network packets
- Processed by **event handlers** (methods that respond to events)

**2. Actions**: Triggered by specific user actions (keypress, hotlink click)
- Processed by methods prefixed with `.action_` + action name

### Event Handler Creation

**Two ways**:
1. Prefix method with `.on_` + event name
2. Use `@on` decorator on handler method

**Example**: `events.py`

```python
# from textual import on
from textual.app import App
from textual.widgets import Button, Digits, Footer

class EventsApp(App):
    CSS_PATH = "events.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("b", "toggle_border", "Toggle border"),
    ]

    presses_count = 0
    double_border = False

    def compose(self):
        yield Button("Click me!", id="button")
        digits = Digits("0", id="digits")
        digits.border_subtitle = "Button presses"
        yield digits
        yield Footer()

    def action_toggle_border(self):
        self.double_border = not self.double_border
        digits = self.query_one("#digits")
        if self.double_border:
            digits.styles.border = ("double", "yellow")
        else:
            digits.styles.border = ("solid", "white")

    def on_button_pressed(self, event):
        if event.button.id == "button":
            self.presses_count += 1
            digits = self.query_one("#digits")
            digits.update(f"{self.presses_count}")

    # Alternative event handler with @on decorator:
    # @on(Button.Pressed, "#button")
    # def button_pressed(self, event):
    #     self.presses_count += 1
    #     digits = self.query_one("#digits")
    #     digits.update(f"{self.presses_count}")

if __name__ == "__main__":
    app = EventsApp()
    app.run()
```

### Key Components

**BINDINGS constant**: List of three-value tuples
- Element 1: Key to bind (e.g., `"q"`, `"b"`)
- Element 2: Action method name (e.g., `"quit"`, `"toggle_border"`)
- Element 3: Visible text in footer

**New widgets**:
- `Digits`: Large-format display of digits using line graphics
- `Footer`: App footer that auto-shows keyboard action prompts

**Actions**:
- `quit`: Built-in Textual action (no need to implement)
- `toggle_border`: Custom action (method `.action_toggle_border()`)

**Event handler**:
- `.on_button_pressed(event)`: Runs when user clicks button
- `event` parameter: Contains event information (e.g., `event.button.id`)

**Alternative syntax**:
- `@on` decorator: `@on(Button.Pressed, "#button")`
- Targets specific event source by ID
- Method can have any name

### DOM Queries

**Purpose**: Locate widgets in DOM hierarchy

**Query methods** (available to all `Widget` subclasses):
- `.query_one()`: Locate single widget
- `.query()`: Locate collection of widgets

**Query by**:
- Python class name
- Widget ID (must be unique)
- TCSS classes
- DOM hierarchy position
- Combination of above

**Example**:
```python
digits = self.query_one("#digits")  # Query by ID
```

### Styling with TCSS

**TCSS file**: `events.tcss`

```css
Button {
    background: $secondary;
    border: solid $primary;
    margin: 2 2;
}

Button:hover {
    border: round white;
}

#digits {
    color: green;
    border: solid white;
    padding: 1;
    width: 30;
}
```

**Pseudoclass**: `:hover`
- Styles applied only when cursor hovers over widget
- Example: Button gets different border on hover

**ID selector**: `#digits`
- Targets specific widget by ID

## Best Practices Summary

### Installation
- Use virtual environments
- Install both `textual` and `textual-dev`

### Development Workflow
- Use `textual console` for debugging
- Use `textual run --dev` for live TCSS reload
- Use `textual serve` for web browser testing

### Styling
- Prefer external TCSS files over Python code
- Use CSS-like selectors (class, ID, pseudoclasses)
- Separate visual design from application logic

### Layout
- Use `Vertical` (default) for vertical stacking
- Use `Horizontal` for horizontal arrangements
- Use scrolling variants (`VerticalScroll`, `HorizontalScroll`) for overflow
- Use `Grid` for complex multi-row/column layouts
- Use `dock` for fixed-position widgets
- Nest containers for complex designs

### Event Handling
- Use `BINDINGS` for keyboard shortcuts
- Implement event handlers with `.on_` prefix or `@on` decorator
- Use DOM queries to locate widgets dynamically
- Access event information via `event` parameter

### Widgets
- `Static`/`Label`: Basic text display
- `Button`: Interactive buttons with variants
- `Input`: Text input with validation
- `Digits`: Large-format number display
- `Footer`: Auto-displays keyboard shortcuts
- See official docs for complete widget list

## Resources

**Official Documentation**:
- [Textual Guide](https://textual.textualize.io/guide/)
- [Textual Widgets](https://textual.textualize.io/widgets/)
- [Textual CSS](https://textual.textualize.io/guide/CSS/)

**Related**:
- [Rich Package Documentation](https://rich.readthedocs.io/en/latest/protocol.html)
- [Build a Contact Book App With Python, Textual, and SQLite](https://realpython.com/contact-book-python-textual/)

## Frequently Asked Questions

**Q: How can I create a text-based user interface (TUI) in Python?**
A: Use the Textual framework, which provides widgets, containers, event-handling, and styling for building visually appealing and interactive TUI apps.

**Q: What are the benefits of using a text-based user interface?**
A: TUIs are efficient, platform-independent, operate over remote connections, work in low-resource environments, ideal for server management and remote development.

**Q: How can I install Textual in Python?**
A: Create a virtual environment and run `python -m pip install textual textual-dev`.

**Q: What are the differences between Textual and Rich in Python?**
A: Textual is a framework for building TUI applications; Rich is a library for adding rich text formatting to terminal output. Textual leverages Rich for enhanced visual appeal.

---

**Tutorial Coverage**:
- ✓ Installation and setup
- ✓ First app creation
- ✓ Widget exploration (Static, Label, Button, Input, Digits, Footer, advanced widgets)
- ✓ Styling with TCSS (selectors, pseudoclasses)
- ✓ Development tools (console, live reload, web server)
- ✓ Layout containers (Vertical, Horizontal, Grid, docking, nesting, scrolling)
- ✓ Event handling (events, actions, bindings)
- ✓ DOM queries
- ✓ Best practices

**Next Steps**:
- Explore advanced widgets (Tree, Sparkline, TabbedContent)
- Build practical applications (contact book, text editor)
- Study Textual's reactive programming model
- Learn command palette integration
- Explore accessibility features
