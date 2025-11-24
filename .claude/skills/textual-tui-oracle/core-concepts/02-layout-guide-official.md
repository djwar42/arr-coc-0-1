# Textual Layout Guide - Official

Comprehensive guide to layout systems in Textual for arranging and positioning widgets on screen.

## Overview

In Textual, the **layout** defines how widgets will be arranged (or *laid out*) inside a container. Textual supports multiple layouts that can be set either via a widget's `styles` object or via CSS. Layouts can be used for both high-level positioning of widgets on screen, and for positioning of nested widgets.

From [Official Textual Layout Guide](https://textual.textualize.io/guide/layout/) (accessed 2025-11-02):

The framework supports four primary layout modes:
1. **Vertical** - arranges child widgets vertically, from top to bottom
2. **Horizontal** - arranges child widgets horizontally, from left to right
3. **Grid** - arranges widgets within a grid with configurable rows/columns
4. **Dock** - fixes widget position to an edge of the container

---

## Vertical Layout

The `vertical` layout arranges child widgets vertically, from top to bottom.

### Key Characteristics

- First widget yielded from `compose` appears at the top
- Subsequent widgets appear below it in order
- Widgets expand to fill parent width by default
- Height must be specified explicitly per widget
- Can use `fr` units to distribute available height equally

### Basic Usage

```python
from textual.app import ComposeResult, Screen
from textual.widgets import Static

class VerticalLayoutScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Box 1", id="box1")
        yield Static("Box 2", id="box2")
        yield Static("Box 3", id="box3")
```

### CSS Configuration

```css
Screen {
    layout: vertical;
}

.box {
    height: 1fr;  /* Equal distribution of available height */
    width: 100%;
}
```

### Height Distribution

- Using `height: 1fr` guarantees children fill available height equally
- If total height exceeds available space, Textual automatically adds scrollbar
- Fixed heights (e.g., `height: 10;`) create fixed-size widgets
- Parent with `overflow-y: auto;` enables automatic vertical scrollbars

### Common Pattern: Setting Layout Programmatically

```python
widget.styles.layout = "vertical"
```

---

## Horizontal Layout

The `horizontal` layout arranges child widgets horizontally, from left to right.

### Key Characteristics

- Widgets are arranged left to right instead of top to bottom
- Widgets do NOT expand to fill parent height by default
- Width must be specified explicitly per widget
- If width not restricted, each widget grows to screen width (others offscreen)
- Textual does NOT automatically add horizontal scrollbars by default

### Basic Usage

```python
from textual.app import ComposeResult, Screen
from textual.widgets import Static

class HorizontalLayoutScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Box 1", id="box1")
        yield Static("Box 2", id="box2")
        yield Static("Box 3", id="box3")
```

### CSS Configuration

```css
Screen {
    layout: horizontal;
}

.box {
    width: 1fr;   /* Equal distribution of available width */
    height: 100%; /* Explicit height needed */
}
```

### Handling Overflow

By default, widgets extending beyond screen width are hidden. Enable horizontal scrolling:

```css
Screen {
    layout: horizontal;
    overflow-x: auto; /* Enables horizontal scrollbar */
}

.box {
    width: 1fr;
    height: 100%;
}
```

With `overflow-x: auto;`, Textual automatically adds horizontal scrollbar when children exceed available space.

---

## Utility Containers

Textual provides helper widgets that simplify layout composition:

- **`Vertical`** - Container with `layout: vertical`
- **`Horizontal`** - Container with `layout: horizontal`
- **`Grid`** - Container with `layout: grid`

### Building Complex Layouts with Containers

Example: Create 2x2 grid using nested containers

```python
from textual.app import ComposeResult, Screen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

class ContainerLayoutScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Box 1", id="box1")
        with Horizontal():
            yield Static("Box 2", id="box2")
            yield Static("Box 3", id="box3")
```

### Composing with Context Managers

Simplified syntax using Python's `with` statement:

```python
from textual.app import ComposeResult, Screen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

class AppWithContextManagers(Screen):
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical():
                yield Static("Box 1")
                yield Static("Box 2")
            with Vertical():
                yield Static("Box 3")
                yield Static("Box 4")
```

Benefits of context managers:
- Easier to read and edit than positional arguments
- Clearer visual nesting structure
- Can be mixed with positional argument approach

---

## Grid Layout

The `grid` layout arranges widgets within a grid, with configurable rows and columns. Widgets can span multiple rows or columns to create complex layouts.

### Important Note

Grid layouts in Textual have little in common with browser-based CSS Grid - it's a distinct system optimized for TUI layouts.

### Basic Grid Setup

```css
Screen {
    layout: grid;
    grid-size: 3 2;  /* 3 columns, 2 rows */
}
```

Widgets are inserted into grid cells from **left-to-right, top-to-bottom** order.

### Creating Grids with Auto-Rows

Omit rows to create rows on demand:

```css
Screen {
    layout: grid;
    grid-size: 3;  /* 3 columns, rows created on demand */
}
```

If you yield more widgets than grid cells, Textual automatically creates new rows.

### Row and Column Sizes

Customize width of columns and height of rows:

```css
Screen {
    layout: grid;
    grid-size: 3 2;
    grid-columns: 2fr 1fr 1fr;  /* First col = 2x width of others */
    grid-rows: 25% 75%;         /* First row = 25%, second = 75% */
}
```

### Size Units Explained

- **`fr` units**: Fractional units distributing available space proportionally
- **`%` units**: Percentage of parent container
- **`auto`**: Content-based sizing (optimal width/height for content)
- **Repeating values**: Values repeat cyclically across columns/rows
  - `grid-columns: 2 4;` on 4-column grid = `2 4 2 4`
  - `grid-columns: 2 4;` on 3-column grid = `2 4 2`

### Auto Sizing Columns/Rows

Use `auto` to calculate optimal size based on content:

```css
#first-column {
    grid-columns: auto 1fr 1fr;  /* First column fits content */
}
```

First column adjusts to fit content width, other columns share remaining space.

### Cell Spanning

Cells can span multiple rows or columns for complex layouts.

#### Column Span

Make a widget span multiple columns:

```css
#header {
    column-span: 2;  /* Spans 2 columns */
    tint: blue 40%;  /* Visual highlight */
}
```

- Spanning widget expands to fill columns to its right
- Following widgets shift to accommodate
- Excess widgets wrap to new rows

#### Row Span

Make a widget span multiple rows:

```css
#sidebar {
    row-span: 2;     /* Spans 2 rows */
}
```

Can combine `row-span` and `column-span` for complex arrangements:

```css
#big-widget {
    row-span: 2;
    column-span: 2;  /* Spans 4 cells total */
}
```

### Grid Gutter (Cell Spacing)

Control spacing between cells:

```css
Screen {
    layout: grid;
    grid-size: 3 2;
    grid-gutter: 1;      /* 1 cell spacing between all cells */
    /* OR: vertical horizontal */
    grid-gutter: 1 2;    /* 1 vertical, 2 horizontal spacing */
}
```

Important notes about gutter:
- Applies spacing **between** cells only (not at container edges)
- Terminal cells are typically 2x taller than wide
- Common pattern: horizontal gutter = 2x vertical gutter for visual consistency

### Complex Grid Example

```python
from textual.app import ComposeResult, Screen
from textual.widgets import Static

class ComplexGridScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Header", id="header")
        yield Static("Widget 1", id="widget1")
        yield Static("Widget 2", id="widget2")
        yield Static("Widget 3", id="widget3")
        yield Static("Widget 4", id="widget4")
        yield Static("Widget 5", id="widget5")
        yield Static("Widget 6", id="widget6")
```

```css
Screen {
    layout: grid;
    grid-size: 3;           /* 3 columns */
    grid-columns: 2fr 1fr 1fr;
    grid-rows: auto 1fr 1fr;
    grid-gutter: 1 2;
}

#header {
    column-span: 3;
    height: 3;
    background: $boost;
}

#widget1 {
    column-span: 2;
}

#widget6 {
    row-span: 2;
}
```

---

## Docking

Widgets may be **docked** to fix their position at an edge while remaining in the layout flow.

### Key Characteristics

- Removes widget from normal layout flow
- Fixes position to top, right, bottom, or left edge
- Docked widgets do NOT scroll out of view (sticky behavior)
- Ideal for headers, footers, sidebars
- Multiple widgets can dock to same edge (with overlap management)

### Basic Docking

```css
#sidebar {
    dock: left;   /* Options: top, right, bottom, left */
    width: 20;
    background: $panel;
}
```

### Single Sidebar Example

```python
from textual.app import ComposeResult, Screen
from textual.widgets import Static

class SidebarScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Sidebar Content", id="sidebar")
        yield Static("Main body content...")

    def on_mount(self):
        self.styles.layout = "vertical"
```

```css
#sidebar {
    dock: left;
    width: 20;
    background: $accent;
}
```

When scrolling body content, sidebar remains fixed (does not scroll out of view).

### Multiple Docked Widgets

Multiple widgets can dock to same edge. Stacking order determined by yield order:

```python
class MultipleSidebarsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Sidebar 1", id="sidebar1")
        yield Static("Sidebar 2", id="sidebar2")
        yield Static("Main content")
```

```css
#sidebar1 {
    dock: left;
    width: 15;
}

#sidebar2 {
    dock: left;
    width: 20;
}
```

- `#sidebar1` yielded first but appears **below** `#sidebar2` visually
- Last yielded widget appears on top when docking to same edge
- For sidebar above content: yield sidebar **after** main content

### Docking Header and Footer

Textual provides built-in `Header` widget with internal CSS docking to top:

```python
from textual.app import ComposeResult, Screen
from textual.widgets import Header, Footer, Static

class HeaderFooterScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Main body content")
        yield Footer()
```

No CSS needed - Header auto-docks to top, Footer to bottom.

### Common Docking Patterns

**Sidebar + Header:**
```css
Screen {
    layout: vertical;
}

Header {
    dock: top;
    height: 3;
}

#sidebar {
    dock: left;
    width: 20;
}
```

**Tab Bar + Content:**
```css
#tabs {
    dock: top;
    height: 1;
}

#content {
    overflow: auto;
}
```

---

## Layers

Textual provides **layers** for fine-grained control over widget drawing order.

### How Layers Work

- Textual draws on **lower** layers first, working up to **higher** layers
- Widgets on higher layers appear on top of lower layers
- Overrides natural yield order

### Defining and Using Layers

```css
Screen {
    layers: background middle foreground;  /* Left = lowest, right = highest */
}

#background-widget {
    layer: background;
}

#foreground-widget {
    layer: foreground;
}
```

### Practical Example

```python
class LayeredScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Widget 1", id="box1")
        yield Static("Widget 2", id="box2")
```

```css
Screen {
    layers: base top;
}

#box1 {
    layer: top;      /* Appears on top despite yielding first */
}

#box2 {
    layer: base;     /* Appears below box1 */
}
```

Result: `#box1` appears above `#box2`, reversing natural yield order.

---

## Offsets

Widgets can have a **relative offset** applied after their position is determined by parent layout.

### How Offsets Work

Offset is applied **after** layout positioning:

1. Layout determines widget position (vertical, horizontal, grid, dock)
2. Offset then adjusts final position relative to layout-determined position
3. Offset does not affect layout (space still reserved for widget)

### Setting Offsets

```css
#widget {
    offset: 2 5;  /* x=2 (right), y=5 (down) */
}
```

Values:
- **First value** (x): Horizontal offset
  - Positive = shift right
  - Negative = shift left
- **Second value** (y): Vertical offset
  - Positive = shift down
  - Negative = shift up

### Practical Offset Example

```css
#floating-widget {
    offset: -1 -2;  /* Shift left 1, up 2 from positioned location */
    tint: cyan 40%;
}
```

---

## Combining Layouts for Advanced UIs

Real applications typically combine multiple layout techniques.

### Complete Layout Example

```python
from textual.app import ComposeResult, Screen
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static

class AdvancedLayoutScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            yield Static("Sidebar", id="sidebar")
            with Vertical():
                yield Static("Top Panel", id="top-panel")
                yield Static("Main Content", id="content")

        yield Footer()
```

```css
Screen {
    layout: vertical;
}

Header {
    dock: top;
    height: 3;
}

Footer {
    dock: bottom;
    height: 1;
}

#sidebar {
    width: 20;
    background: $panel;
}

#content {
    overflow: auto;
}

#top-panel {
    height: 5;
    border: solid $accent;
}
```

### Layout Composition Strategy

1. **Use containers** (Vertical/Horizontal/Grid) for major sections
2. **Dock** fixed headers/footers/sidebars
3. **Grid** for complex widget arrangements
4. **Layers** to control stacking order
5. **Offsets** for fine-tuning positions

---

## Common Layout Issues and Solutions

### Problem: Widgets Not Expanding to Container Width

**Issue**: Horizontal layout children don't fill parent width

**Solution**: Set explicit width or use `width: 1fr`

```css
.box {
    width: 1fr;  /* Distribute available width equally */
    height: 100%;
}
```

### Problem: Content Hidden in Horizontal Layout

**Issue**: Widgets overflow right edge and are not visible

**Solution**: Enable horizontal scrolling

```css
Screen {
    overflow-x: auto;
}
```

### Problem: Grid Doesn't Fit All Widgets

**Issue**: Some widgets don't appear in grid

**Solution**: Omit row count to auto-create rows

```css
Screen {
    grid-size: 3;      /* 3 columns, rows auto-created */
    /* NOT: grid-size: 3 2; which limits to 6 cells */
}
```

### Problem: Sidebar Scrolls Out of View

**Issue**: Sidebar content scrolls with main content

**Solution**: Dock the sidebar

```css
#sidebar {
    dock: left;  /* Stays fixed while content scrolls */
}
```

### Problem: Uneven Column Widths in Grid

**Issue**: Columns not sized as intended

**Solution**: Specify `grid-columns` with explicit units

```css
Screen {
    grid-size: 3;
    grid-columns: 2fr 1fr 1fr;  /* First column twice as wide */
}
```

### Problem: Widgets Overlapping Unintentionally

**Issue**: Docked widgets overlap or hide layout

**Solution**: Use layers to control stacking order

```css
Screen {
    layers: background content overlay;
}

#overlay-widget {
    layer: overlay;  /* Ensure appears on top */
}
```

---

## Troubleshooting Tips

### Debug Layout Issues

1. **Visualize layout**: Add background colors or borders to see widget boundaries
   ```css
   .box {
       background: $boost;
       border: solid $accent;
   }
   ```

2. **Check dimensions**: Use explicit sizes to understand space allocation
   ```css
   .box {
       width: 50%;
       height: 10;
   }
   ```

3. **Verify layout mode**: Ensure parent layout is what you expect
   ```css
   Screen {
       layout: vertical;  /* Explicitly set */
   }
   ```

4. **Test overflow**: Enable scrollbars to see if content exceeds space
   ```css
   Screen {
       overflow-y: auto;
       overflow-x: auto;
   }
   ```

5. **Trace widget order**: Add IDs and inspect yield order in `compose()`
   ```python
   yield Static("First", id="first")
   yield Static("Second", id="second")
   ```

### Common Patterns

**Standard App Layout:**
```css
Screen {
    layout: vertical;
}

Header {
    dock: top;
    height: 3;
}

Footer {
    dock: bottom;
    height: 1;
}
```

**Two-Column Layout:**
```css
Screen {
    layout: horizontal;
}

#sidebar {
    width: 20;
}

#content {
    width: 1fr;
}
```

**Responsive Grid:**
```css
Screen {
    layout: grid;
    grid-size: 3;
    grid-columns: 1fr 1fr 1fr;
    grid-gutter: 1 2;
}
```

---

## Layout Properties Reference

### CSS Properties

| Property | Values | Purpose |
|----------|--------|---------|
| `layout` | `vertical`, `horizontal`, `grid` | Sets layout mode for widget children |
| `grid-size` | `columns [rows]` | Grid dimensions (rows auto if omitted) |
| `grid-columns` | `size ...` | Width of each column (repeats if needed) |
| `grid-rows` | `size ...` | Height of each row (repeats if needed) |
| `grid-gutter` | `v [h]` | Spacing between cells (vertical, horizontal) |
| `dock` | `top`, `right`, `bottom`, `left` | Fixes widget to edge |
| `column-span` | `number` | Cells to span horizontally in grid |
| `row-span` | `number` | Cells to span vertically in grid |
| `layers` | `name ...` | Defines layer order (space-separated) |
| `layer` | `name` | Assigns widget to named layer |
| `offset` | `x y` | Relative position adjustment |

### Size Units

| Unit | Meaning | Example |
|------|---------|---------|
| `fr` | Fractional (proportional) | `1fr` = 1 share, `2fr` = 2 shares |
| `%` | Percentage of parent | `50%` = half width/height |
| `auto` | Content-based sizing | Auto-fits to content |
| Number | Cells/characters | `10` = 10 characters/cells |

---

## Sources

**Official Documentation:**
- [Textual Layout Guide](https://github.com/Textualize/textual/blob/main/docs/guide/layout.md) - Complete source for all layout systems, patterns, and examples (accessed 2025-11-02)

**Web Reference:**
- [Textual Official Layout Guide](https://textual.textualize.io/guide/layout/) - Interactive documentation with live examples (accessed 2025-11-02)

**Related Documentation:**
- [Textual Containers API](https://textual.textualize.io/api/containers/) - Vertical, Horizontal, Grid container widgets
- [Textual Styles Reference](https://textual.textualize.io/styles/) - Complete CSS property documentation
- [Textual Design a Layout How-To](https://textual.textualize.io/how-to/design-a-layout/) - Design patterns and best practices
