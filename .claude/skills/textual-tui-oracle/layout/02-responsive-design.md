# Responsive Layout Design in Textual

## Overview

Textual layouts are inherently responsive and adapt to different terminal sizes. Using flexible units, layout containers, and proper styling, you can create TUIs that work across various screen dimensions.

From [Textual Layout Guide](https://textual.textualize.io/guide/layout/) and [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):
- Layouts automatically adapt to terminal size changes
- Use fractional units (`fr`) for flexible sizing
- Combine layout containers for complex responsive designs
- Test across different terminal dimensions

## Responsive Sizing Units

### Fractional Units (`fr`)

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

The `fr` unit represents a fraction of available space:

```css
Static {
    height: 1fr;  /* Each widget gets equal height */
    width: 1fr;   /* Each widget gets equal width */
}
```

**Mixed fractions:**
```css
Screen {
    grid-columns: 1fr 2fr 1fr;  /* Middle column is 2x wider */
}
```

If total fractions = 4, middle column gets 50% (2/4).

### Percentage Units

```css
#sidebar {
    width: 20%;     /* 20% of parent container */
    min-width: 15;  /* Minimum absolute width */
    height: 100%;   /* Full parent height */
}
```

### Auto Sizing

```css
#header {
    dock: top;
    height: auto;  /* Height determined by content */
    width: 100%;   /* Full width */
}
```

## Vertical Layout Responsiveness

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

**Basic vertical stacking:**
```python
from textual.app import App
from textual.containers import Vertical
from textual.widgets import Static

class VerticalLayoutApp(App):
    def compose(self):
        with Vertical():
            for i in range(4):
                yield Static(f"Static {i + 1}")
```

Widgets automatically stack and resize as terminal height changes.

**Vertical with scrolling for overflow:**
```python
from textual.containers import VerticalScroll

class VerticalScrollApp(App):
    def compose(self):
        with VerticalScroll():
            for i in range(20):  # More widgets than fit on screen
                yield Static(f"Static {i + 1}")
```

## Horizontal Layout Responsiveness

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

**Horizontal with equal widths:**
```python
from textual.containers import Horizontal

class HorizontalLayoutApp(App):
    def compose(self):
        with Horizontal():
            for i in range(4):
                static = Static(f"Static {i + 1}")
                static.styles.width = "1fr"  # Equal widths
                yield static
```

**Horizontal with scrolling:**
```python
from textual.containers import HorizontalScroll

class HorizontalScrollApp(App):
    def compose(self):
        with HorizontalScroll():
            for i in range(20):
                static = Static(f"Static {i + 1}")
                static.styles.width = "10%"
                yield static
```

## Grid Layout Responsiveness

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

**Flexible grid with fr units:**
```css
Grid {
    grid-size: 4 6;
}

Static {
    height: 1fr;  /* Cells expand to fill available space */
    width: 1fr;
    border: solid green;
}
```

**Responsive grid columns:**
```css
Screen {
    layout: grid;
    grid-size: 3 4;
    grid-columns: 1fr 2fr 1fr;  /* Middle column adapts to screen size */
    grid-rows: 1fr;
}
```

## Nested Responsive Containers

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

Complex layouts combine multiple container types:

```python
from textual.containers import Horizontal, HorizontalScroll, VerticalScroll

class NestedContainersApp(App):
    CSS_PATH = "layouts.tcss"

    def compose(self):
        with Horizontal(id="horizontal"):
            yield Static("Left", classes="box")

            with HorizontalScroll(id="horizontalscroll"):
                for i in range(12):
                    yield Static(f"Center.{i + 1}", classes="box yellowbox")

            with VerticalScroll(id="verticalscroll"):
                for i in range(12):
                    yield Static(f"Right.{i + 1}", classes="box redbox")
```

**Responsive styling:**
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
```

## Responsive Docking

From [Textual Dock Documentation](https://textual.textualize.io/styles/dock/) (accessed 2025-11-02):

Docked widgets with responsive sizing:

```css
#sidebar {
    dock: left;
    width: 20%;      /* Scales with terminal width */
    min-width: 15;   /* Maintains minimum usability */
    max-width: 30;   /* Prevents excessive width */
    height: 100%;
}

#header {
    dock: top;
    height: auto;    /* Adapts to content */
    width: 100%;
}
```

## Responsive Width and Height Constraints

**Minimum/maximum constraints:**
```css
#content {
    width: 50%;
    min-width: 40;   /* Don't shrink below 40 columns */
    max-width: 100;  /* Don't expand beyond 100 columns */

    height: 1fr;
    min-height: 10;  /* Minimum 10 rows */
}
```

## Testing Responsive Layouts

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

**Development workflow:**
1. Run app with `textual run --dev`
2. Resize terminal window
3. Observe layout adaptation
4. Adjust CSS as needed
5. Changes to `.tcss` file apply immediately (no restart needed)

**Test different terminal sizes:**
- Small terminals (80x24)
- Medium terminals (120x40)
- Large terminals (200x60+)
- Ultra-wide and tall terminals

## Responsive Design Patterns

### Three-Column Layout

```python
with Horizontal():
    yield Static("Sidebar", id="sidebar")
    yield Static("Main Content", id="main")
    yield Static("Details", id="details")
```

```css
#sidebar {
    width: 15%;
    min-width: 10;
}

#main {
    width: 1fr;  /* Takes remaining space */
}

#details {
    width: 20%;
    min-width: 15;
}
```

### Header-Content-Footer

```python
from textual.widgets import Header, Footer

def compose(self):
    yield Header()               # Auto-docked top
    yield Static("Content")      # Fills remaining space
    yield Footer()               # Auto-docked bottom
```

### Dashboard Grid

```css
Screen {
    layout: grid;
    grid-size: 3 3;
    grid-columns: 1fr 1fr 1fr;   /* Equal columns */
    grid-rows: auto 1fr auto;    /* Middle row flexible */
    grid-gutter: 1;
}
```

## Screen Size Awareness

**Query terminal dimensions in code:**
```python
from textual.app import App

class ResponsiveApp(App):
    def on_mount(self):
        # Terminal dimensions available
        width = self.size.width
        height = self.size.height

        # Adjust layout based on size
        if width < 80:
            # Compact layout for narrow terminals
            self.query_one("#sidebar").display = False
        else:
            # Full layout for wide terminals
            self.query_one("#sidebar").display = True
```

## Overflow Handling

**Vertical overflow:**
```python
from textual.containers import VerticalScroll

with VerticalScroll():
    # Content that exceeds screen height
    # Automatically scrollable
```

**Horizontal overflow:**
```python
from textual.containers import HorizontalScroll

with HorizontalScroll():
    # Content that exceeds screen width
    # Automatically scrollable
```

**Grid overflow:**
```css
Screen {
    overflow: auto;  /* Enable scrolling when grid exceeds screen */
}
```

## Best Practices

From [Textual How-To: Design a Layout](https://textual.textualize.io/how-to/design-a-layout/) (accessed 2025-11-02):

1. **Use fractional units (`fr`):** Enables automatic space distribution
2. **Set minimum constraints:** Prevents layouts from becoming unusable
3. **Test multiple terminal sizes:** Ensures consistent experience
4. **Use scrolling containers:** Handles overflow gracefully
5. **Leverage docking:** Keeps critical UI elements visible
6. **Combine container types:** Nested containers create flexible layouts
7. **Use percentage for relative sizing:** Adapts to parent dimensions
8. **Apply max constraints:** Prevents excessive expansion

## Common Responsive Challenges

**Challenge: Content overflow**
- Solution: Use `VerticalScroll` or `HorizontalScroll`

**Challenge: Widget too narrow**
- Solution: Set `min-width` constraint

**Challenge: Uneven space distribution**
- Solution: Use `fr` units with appropriate ratios

**Challenge: Fixed elements covering content**
- Solution: Use `dock` for fixed elements, normal layout for content

**Challenge: Grid cells inconsistent**
- Solution: Specify `grid-columns` and `grid-rows` explicitly

## Sources

**Official Documentation:**
- [Textual Layout Guide](https://textual.textualize.io/guide/layout/) - Layout systems overview (accessed 2025-11-02)
- [Textual How-To: Design a Layout](https://textual.textualize.io/how-to/design-a-layout/) - Layout design approach (accessed 2025-11-02)
- [Textual Dock Style](https://textual.textualize.io/styles/dock/) - Docking widgets (accessed 2025-11-02)
- [Textual Grid Styles](https://textual.textualize.io/styles/grid/) - Grid layout reference (accessed 2025-11-02)

**Tutorials:**
- [Python Textual: Build Beautiful UIs in the Terminal - Real Python](https://realpython.com/python-textual/) - Comprehensive layout examples (accessed 2025-11-02)

**Additional Resources:**
- [Textual: The Definitive Guide - Part 1 - DEV Community](https://dev.to/wiseai/textual-the-definitive-guide-part-1-1i0p) - Docking and layout techniques (accessed 2025-11-02)
