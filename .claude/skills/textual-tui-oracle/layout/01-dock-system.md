# Dock Layout System in Textual

## Overview

The `dock` style fixes a widget to the edge of a container, removing it from the normal layout flow. Docked widgets remain fixed in position and do not scroll out of view, making them ideal for sticky headers, footers, and sidebars.

From [Textual Dock Documentation](https://textual.textualize.io/styles/dock/) (accessed 2025-11-02):
- Docking removes widgets from layout
- Widgets stick to container edges (top, right, bottom, left)
- Docked widgets don't scroll with content
- Perfect for persistent UI elements

## Dock Syntax

```css
dock: bottom | left | right | top;
```

The option determines which edge the widget is docked to.

## Basic Dock Example

From [Textual Dock Documentation](https://textual.textualize.io/styles/dock/) (accessed 2025-11-02):

**Python code:**
```python
from textual.app import App, ComposeResult
from textual.widgets import Static

TEXT = """\
Docking a widget removes it from the layout and fixes its position,
aligned to either the top, right, bottom, or left edges of a container.

Docked widgets will not scroll out of view, making them ideal for
sticky headers, footers, and sidebars.
"""

class DockLayoutExample(App):
    CSS_PATH = "dock_layout1_sidebar.tcss"

    def compose(self) -> ComposeResult:
        yield Static("Sidebar", id="sidebar")
        yield Static(TEXT * 10, id="body")

if __name__ == "__main__":
    app = DockLayoutExample()
    app.run()
```

**TCSS styling:**
```css
#sidebar {
    dock: left;
    width: 15;
    height: 100%;
    color: #0f2b41;
    background: dodgerblue;
}
```

Result: The sidebar remains fixed on the left even as the body content scrolls.

## Advanced Docking - All Four Edges

From [Textual Dock Documentation](https://textual.textualize.io/styles/dock/) (accessed 2025-11-02):

**Python code:**
```python
from textual.app import App
from textual.containers import Container
from textual.widgets import Label

class DockAllApp(App):
    CSS_PATH = "dock_all.tcss"

    def compose(self):
        yield Container(
            Container(Label("left"), id="left"),
            Container(Label("top"), id="top"),
            Container(Label("right"), id="right"),
            Container(Label("bottom"), id="bottom"),
            id="big_container",
        )

if __name__ == "__main__":
    app = DockAllApp()
    app.run()
```

**TCSS styling:**
```css
#left {
    dock: left;
    height: 100%;
    width: auto;
    align-vertical: middle;
}

#top {
    dock: top;
    height: auto;
    width: 100%;
    align-horizontal: center;
}

#right {
    dock: right;
    height: 100%;
    width: auto;
    align-vertical: middle;
}

#bottom {
    dock: bottom;
    height: auto;
    width: 100%;
    align-horizontal: center;
}

Screen {
    align: center middle;
}

#big_container {
    width: 75%;
    height: 75%;
    border: round white;
}
```

## Docking in Python vs CSS

**CSS approach (recommended):**
```css
#docked-label {
    dock: bottom;
    border: solid dodgerblue;
}
```

**Python approach:**
```python
widget.styles.dock = "bottom"  # Dock bottom
widget.styles.dock = "left"    # Dock left
widget.styles.dock = "right"   # Dock right
widget.styles.dock = "top"     # Dock top
```

## Common Dock Patterns

### Sticky Header and Footer

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

```python
from textual.app import App
from textual.widgets import Header, Footer, Static

class AppWithHeaderFooter(App):
    def compose(self):
        yield Header()  # Auto-docked to top
        yield Static("Main content area")
        yield Footer()  # Auto-docked to bottom
```

**Note:** `Header` and `Footer` widgets are pre-configured to dock automatically.

### Sidebar Navigation

```python
from textual.app import App
from textual.widgets import Static

class SidebarApp(App):
    CSS_PATH = "sidebar.tcss"

    def compose(self):
        yield Static("Navigation", id="sidebar")
        yield Static("Main content", id="content")
```

```css
#sidebar {
    dock: left;
    width: 20;
    background: $panel;
}
```

### Nested Containers with Docked Widgets

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

```python
from textual.containers import VerticalScroll
from textual.widgets import Label

with VerticalScroll(id="verticalscroll"):
    for i in range(NUM_BOXES):
        yield Static(f"Right.{i + 1}", classes="box redbox")
    yield Label(
        "I am a docked label.\nI don't move!",
        id="docked-label",
    )
```

```css
#docked-label {
    dock: bottom;
    border: solid dodgerblue;
}
```

The label stays at the bottom of the `VerticalScroll` container while other widgets scroll past.

## Dock vs Layout Containers

**Docked widgets:**
- Ignore parent container's layout rules
- Fixed to container edges
- Don't scroll with content
- Positioned independently

**Layout containers (Vertical/Horizontal/Grid):**
- Follow layout rules
- Scroll with content
- Positioned relative to siblings
- Affected by container flow

## Dock Positioning Details

From [Textual Dock Documentation](https://textual.textualize.io/styles/dock/) (accessed 2025-11-02):

When multiple widgets dock to the same edge:
- They stack in the order they appear in the DOM
- Later docked widgets appear further from the edge

Example:
```python
yield Static("First", id="header1")   # Docked top
yield Static("Second", id="header2")  # Docked top
```

```css
#header1, #header2 {
    dock: top;
}
```

Result: `header1` appears at the very top, `header2` appears below it.

## Responsive Docking

Docked widgets adapt to container size changes:

```css
#sidebar {
    dock: left;
    width: 20%;      /* Responsive width */
    min-width: 15;   /* Minimum width constraint */
    height: 100%;
}
```

## Dock with Height/Width Auto

From [Textual Dock Documentation](https://textual.textualize.io/styles/dock/) (accessed 2025-11-02):

```css
#top {
    dock: top;
    height: auto;    /* Height determined by content */
    width: 100%;     /* Full container width */
}
```

For left/right docks, use `width: auto` to size by content.

## Best Practices

1. **Use for persistent UI elements:** Headers, footers, toolbars, navigation
2. **Set appropriate dimensions:** Specify width for left/right, height for top/bottom
3. **Combine with layouts:** Dock works alongside Vertical/Horizontal/Grid layouts
4. **Consider z-order:** Docked widgets render above normal layout widgets
5. **Test scrolling behavior:** Verify docked widgets stay visible during scrolling

## Common Use Cases

**Application chrome:**
```python
yield Header()           # Top dock (automatic)
yield Static("Sidebar")  # Left dock
yield Static("Content")  # Normal layout
yield Footer()           # Bottom dock (automatic)
```

**Floating toolbars:**
```css
#toolbar {
    dock: top;
    height: 3;
    background: $accent;
}
```

**Status indicators:**
```css
#status {
    dock: bottom;
    height: 1;
    text-align: right;
}
```

## Sources

**Official Documentation:**
- [Textual Dock Style](https://textual.textualize.io/styles/dock/) - Official dock reference (accessed 2025-11-02)
- [Textual Layout Guide](https://textual.textualize.io/guide/layout/) - Layout systems overview (accessed 2025-11-02)

**Tutorials:**
- [Python Textual: Build Beautiful UIs in the Terminal - Real Python](https://realpython.com/python-textual/) - Docked widgets examples (accessed 2025-11-02)

**Community:**
- [Textual GitHub Discussions](https://github.com/Textualize/textual/discussions) - Community examples and patterns
