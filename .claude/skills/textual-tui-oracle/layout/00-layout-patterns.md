# Layout Patterns

## Overview

Textual provides flexible layout systems for arranging widgets on screen. This guide covers container types, nesting strategies, and CSS-based layout approaches.

From [RealPython](https://realpython.com/python-textual/) (accessed 2025-11-02):
- Vertical and horizontal containers with scrolling variants
- Docking widgets for fixed positioning
- Grid layouts for complex arrangements

From [Mouse Vs Python](https://www.blog.pythonlibrary.org/2025/04/01/textual-how-to-add-widgets-to-a-container/) (accessed 2025-11-02):
- Container fundamentals and usage patterns
- Nesting containers for complex layouts
- CSS-based layout configuration

---

## Default Layout Behavior

### Implicit Vertical Layout

By default, Textual arranges widgets vertically (top to bottom):

```python
from textual.app import App
from textual.widgets import Button

class DefaultApp(App):
    def compose(self):
        yield Button("First")
        yield Button("Second")
        yield Button("Third")

# Buttons appear vertically stacked
```

This is equivalent to using an implicit `Vertical` container.

---

## Container Types

### Pattern 1: Vertical Container

Stack widgets vertically (top to bottom):

From RealPython (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Vertical
from textual.widgets import Static

class VerticalApp(App):
    def compose(self):
        with Vertical():
            for i in range(4):
                yield Static(f"Box {i + 1}")
```

**Use cases**:
- Lists of items
- Sequential workflows
- Form layouts (stacked fields)

### Pattern 2: Horizontal Container

Stack widgets horizontally (left to right):

From Mouse Vs Python (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Horizontal
from textual.widgets import Button

class HorizontalApp(App):
    def compose(self):
        with Horizontal():
            yield Button("OK")
            yield Button("Cancel")
            yield Button("Go!")
```

**Use cases**:
- Button groups
- Side-by-side panels
- Toolbar layouts

**Width considerations**:
- Without explicit width, first widget expands to fill space
- Set `width: 1fr` in CSS to distribute space equally

```css
/* Make horizontal items equal width */
Button {
    width: 1fr;
    height: 100%;
}
```

### Pattern 3: Grid Container

Arrange widgets in rows and columns:

From RealPython (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Grid
from textual.widgets import Static

class GridApp(App):
    def compose(self):
        grid = Grid()
        grid.styles.grid_size_rows = 3
        grid.styles.grid_size_columns = 4
        with grid:
            for row in range(3):
                for col in range(4):
                    yield Static(f"({row},{col})")
```

**Grid CSS configuration**:
```css
Grid {
    grid-size: 4 3;  /* 4 columns, 3 rows */
    grid-gutter: 1 2;  /* gutter between items */
}

Grid > Static {
    width: 1fr;
    height: 1fr;
}
```

**Use cases**:
- Dashboard layouts
- Data tables
- Game boards
- Calendar views

### Pattern 4: Scrolling Containers

Automatically add scrollbars when content overflows:

From RealPython (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import VerticalScroll, HorizontalScroll
from textual.widgets import Static

class ScrollingApp(App):
    def compose(self):
        with VerticalScroll():
            for i in range(20):
                yield Static(f"Item {i + 1}")
```

**Scrolling container variants**:
- `VerticalScroll`: Vertical scrolling
- `HorizontalScroll`: Horizontal scrolling
- `ScrollableContainer`: Both axes

**When to use**:
- Lists that may exceed screen height
- Long horizontal content
- Responsive layouts

---

## Nesting Containers

### Pattern 5: Simple Nesting for Rows and Columns

From Mouse Vs Python (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Button

class NestedApp(App):
    CSS_PATH = "nested.tcss"

    def compose(self):
        yield Vertical(
            Horizontal(
                Button("One"),
                Button("Two"),
                classes="row",
            ),
            Horizontal(
                Button("Three"),
                Button("Four"),
                classes="row",
            ),
        )
```

CSS for styling:
```css
Button {
    width: 1fr;
    height: 1fr;
    border: solid green;
    background: $panel;
}

.row {
    height: 1fr;
}
```

**Visual result**:
```
┌─────────────────┐
│  One  │  Two    │
├───────┼─────────┤
│ Three │  Four   │
└─────────────────┘
```

### Pattern 6: Complex Multi-Level Nesting

From RealPython (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static, Label

class ComplexApp(App):
    CSS_PATH = "complex.tcss"

    def compose(self):
        with Horizontal(id="horizontal"):
            # Left panel
            yield Static("Left", classes="box")

            # Center scrollable area
            with VerticalScroll(id="horizontalscroll"):
                for i in range(12):
                    yield Static(f"Center.{i + 1}", classes="box")

            # Right scrollable area
            with VerticalScroll(id="verticalscroll"):
                for i in range(12):
                    yield Static(f"Right.{i + 1}", classes="box")
```

**Nesting depth considerations**:
- Avoid excessive nesting (>3-4 levels)
- Each nesting level adds complexity
- Use CSS classes for styling related elements

---

## Docked Widgets

### Pattern 7: Fixed Position Widgets

Use `dock` style to fix widgets at specific positions:

From RealPython (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Label

class DockedApp(App):
    CSS_PATH = "docked.tcss"

    def compose(self):
        with Vertical():
            yield Static("Header", id="header")

            with VerticalScroll():
                for i in range(20):
                    yield Static(f"Content {i}")

            yield Static("Footer", id="footer")
```

CSS for docking:
```css
#header {
    dock: top;
    height: 3;
    background: $primary;
}

#footer {
    dock: bottom;
    height: 2;
    background: $secondary;
}
```

**Dock positions**:
- `top`: Fixed at top of container
- `bottom`: Fixed at bottom
- `left`: Fixed at left
- `right`: Fixed at right

**Use cases**:
- Headers and footers
- Sidebars
- Fixed toolbars
- Sticky labels (within scrollable containers)

### Pattern 8: Docked Widget with Scrolling Content

```python
from textual.containers import VerticalScroll
from textual.widgets import Label

class MixedLayoutApp(App):
    CSS_PATH = "mixed.tcss"

    def compose(self):
        with VerticalScroll():
            # Docked label stays visible while content scrolls
            yield Label("I am docked", id="docked-label")

            for i in range(30):
                yield Static(f"Scrollable item {i}")
```

CSS:
```css
#docked-label {
    dock: bottom;
    border: solid blue;
}
```

---

## Advanced Layout Patterns

### Pattern 9: Centered Layout

From mathspp (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Container
from textual.widgets import Static, Label

class CenteredApp(App):
    CSS = """
    Container {
        align: center middle;
        border: solid green;
    }
    """

    def compose(self):
        with Container():
            yield Label("Centered content")
```

CSS alignment options:
```css
Container {
    align: center middle;  /* horizontal center, vertical middle */
    align: left top;       /* top-left */
    align: right bottom;   /* bottom-right */
}
```

### Pattern 10: Responsive Layout with Fractional Units

Use `fr` (fractions) for responsive sizing:

From RealPython (accessed 2025-11-02):
```python
from textual.app import App
from textual.containers import Horizontal
from textual.widgets import Static

class ResponsiveApp(App):
    CSS = """
    Horizontal > Static {
        width: 1fr;  /* Equal width distribution */
        height: 100%;
        border: solid;
    }
    """

    def compose(self):
        with Horizontal():
            yield Static("Panel 1")
            yield Static("Panel 2")
            yield Static("Panel 3")
```

**Fractional unit benefits**:
- Automatically adjust to terminal size
- Responsive without media queries
- Equal distribution with `1fr`
- Weighted distribution: `2fr` gets 2x the space

### Pattern 11: Form Layout Pattern

```python
from textual.app import App
from textual.containers import Vertical
from textual.widgets import Input, Label, Button

class FormApp(App):
    CSS = """
    Vertical {
        width: 60;
        height: auto;
        align: center middle;
    }

    Input {
        margin: 1 0;
        width: 100%;
    }

    Button {
        margin: 1 0;
        width: 100%;
    }
    """

    def compose(self):
        with Vertical():
            yield Label("User Form")
            yield Input(placeholder="Username")
            yield Input(placeholder="Password", password=True)
            yield Button("Submit")
            yield Button("Cancel")
```

### Pattern 12: Dashboard Layout

```python
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

class DashboardApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #header {
        dock: top;
        height: 3;
        background: $primary;
    }

    #footer {
        dock: bottom;
        height: 2;
        background: $secondary;
    }

    #content {
        width: 1fr;
        height: 1fr;
    }

    .panel {
        border: solid;
        padding: 1;
    }
    """

    def compose(self):
        yield Static("Dashboard Header", id="header")

        with Horizontal(id="content"):
            with Vertical(classes="panel"):
                yield Static("Sidebar")

            with Vertical(classes="panel"):
                yield Static("Main content area")

            with Vertical(classes="panel"):
                yield Static("Right panel")

        yield Static("Dashboard Footer", id="footer")
```

---

## CSS Layout Properties

### Common CSS Layout Styles

```css
/* Sizing */
width: 100%;           /* Full width */
width: 50;            /* Fixed width */
width: 1fr;           /* One fraction unit */
height: auto;         /* Auto height */
height: 5;            /* Fixed height */

/* Spacing */
margin: 1;            /* All sides */
margin: 1 2;          /* Top/bottom 1, left/right 2 */
padding: 1 2 3 4;     /* Top, right, bottom, left */

/* Alignment */
align: center middle;  /* Horizontal and vertical */
content-align: left middle;  /* Content alignment within widget */

/* Borders and decorations */
border: solid green;
border: double blue;
border: dashed yellow;

/* Display */
display: block;       /* Default */
display: none;        /* Hide widget */

/* Positioning */
dock: top;           /* Fixed position: top, bottom, left, right */
offset: 0 0;         /* Offset from position */
```

---

## Layout Anti-Patterns

1. **Excessive nesting**: More than 3-4 levels becomes hard to manage
2. **Not setting dimensions**: Widgets need explicit height/width
3. **Forgetting scrolling containers**: Content overflow freezes without them
4. **Mixing layout modes**: Inconsistent width/height units
5. **Hard-coded sizes**: Use `1fr` and percentages for responsiveness

---

## Layout Debugging

### Pattern 13: Visibility Debugging

```python
# Add debug styling to see container boundaries
css = """
* {
    border: solid yellow;  /* All widgets visible */
}
"""
```

### Useful properties for debugging:
```css
Container {
    border: solid;
    background: $primary 20%;  /* Semi-transparent background */
}
```

---

## Common Layout Recipes

| Use Case | Pattern | Key Container |
|----------|---------|---|
| Sidebar layout | Horizontal with left/main | `Horizontal` |
| Tab interface | Horizontal buttons + container | `Horizontal` |
| Form | Vertical stack of inputs | `Vertical` |
| Dashboard | Horizontal with vertical panels | `Horizontal` + `Vertical` |
| Header/Footer/Content | Vertical with docked top/bottom | `Vertical` + `dock` |
| Data table | Grid container | `Grid` |
| Popup/Modal | Centered container | `Container` + centered |

---

## Sources

**Web Research:**
- [RealPython - Python Textual Tutorial](https://realpython.com/python-textual/) (accessed 2025-11-02)
  - Container types and properties
  - Docking and positioning
  - Grid layouts
  - Nesting patterns

- [Mouse Vs Python - Textual - How to Add Widgets to a Container](https://www.blog.pythonlibrary.org/2025/04/01/textual-how-to-add-widgets-to-a-container/) (accessed 2025-11-02)
  - Vertical and horizontal containers
  - Nesting containers
  - CSS-based layout configuration

- [mathspp - Textual for Beginners](https://mathspp.com/blog/textual-for-beginners) (accessed 2025-11-02)
  - Container context manager usage
  - Centered layouts
