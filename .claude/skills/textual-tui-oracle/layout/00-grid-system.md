# Grid Layout System in Textual

## Overview

The grid layout in Textual arranges widgets within a grid structure. Widgets can span multiple rows and columns to create complex layouts. Grid layout is one of three primary layout systems in Textual (along with vertical and horizontal).

From [Textual Grid Documentation](https://textual.textualize.io/styles/grid/) (accessed 2025-11-02):
- Grid layouts organize widgets in rows and columns
- Widgets can span multiple cells
- Grid gutters provide spacing between cells
- Flexible sizing using scalar units and `fr` (fraction) units

## Grid Layout Properties

### Core Grid Styles

**`grid-size`** - Number of columns and rows:
```css
Screen {
    layout: grid;
    grid-size: 3 4;  /* 3 columns, 4 rows */
}
```

**`grid-columns`** - Width of grid columns:
```css
Screen {
    grid-columns: 1fr 2fr 1fr;  /* Middle column twice as wide */
}
```

**`grid-rows`** - Height of grid rows:
```css
Screen {
    grid-rows: 1fr;  /* All rows equal height */
}
```

**`grid-gutter`** - Spacing between cells:
```css
Screen {
    grid-gutter: 1;     /* 1 cell spacing on all sides */
    grid-gutter: 1 2;   /* 1 vertical, 2 horizontal */
}
```

**`column-span`** - Number of columns a cell spans:
```css
#static1 {
    column-span: 2;  /* Widget spans 2 columns */
}
```

**`row-span`** - Number of rows a cell spans:
```css
#static1 {
    row-span: 3;  /* Widget spans 3 rows */
}
```

## Basic Grid Example

From [Textual Grid Documentation](https://textual.textualize.io/styles/grid/) (accessed 2025-11-02):

**Python code:**
```python
from textual.app import App
from textual.widgets import Static

class GridApp(App):
    CSS_PATH = "grid.tcss"

    def compose(self):
        yield Static("Grid cell 1\n\nrow-span: 3;\ncolumn-span: 2;", id="static1")
        yield Static("Grid cell 2", id="static2")
        yield Static("Grid cell 3", id="static3")
        yield Static("Grid cell 4", id="static4")
        yield Static("Grid cell 5", id="static5")
        yield Static("Grid cell 6", id="static6")
        yield Static("Grid cell 7", id="static7")

if __name__ == "__main__":
    app = GridApp()
    app.run()
```

**TCSS styling:**
```css
Screen {
    layout: grid;
    grid-size: 3 4;
    grid-rows: 1fr;
    grid-columns: 1fr;
    grid-gutter: 1;
}

Static {
    color: auto;
    background: lightblue;
    height: 100%;
    padding: 1 2;
}

#static1 {
    tint: magenta 40%;
    row-span: 3;
    column-span: 2;
}
```

## Advanced Grid Example

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):

**Python code:**
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

**Alternative with TCSS:**
```python
class GridLayoutWithTCSS(App):
    CSS_PATH = "grid.tcss"

    def compose(self):
        with Grid():
            for row in range(6):
                for col in range(4):
                    yield Static(f"Static ({row=}, {col=})")
```

```css
Grid {
    grid-size: 4 6;
}

Static {
    height: 1fr;
    width: 1fr;
    border: solid green;
}
```

## Grid Sizing with `fr` Units

The `fr` (fraction) unit represents a fraction of available space:

```css
Screen {
    grid-columns: 1fr 2fr 1fr;  /* Middle column gets 50% (2/4) */
}
```

If all widgets get `1fr`, each occupies equal space. Mixed units work too:

```css
Screen {
    grid-columns: 100 1fr 200;  /* Fixed sidebars, flexible center */
}
```

## Grid Auto-Placement

Widgets are automatically placed in grid cells from left to right, top to bottom:

From [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02):
- First widget fills row 0, column 0
- Second widget fills row 0, column 1
- Continue until row is full, then move to next row
- Spanning widgets occupy multiple cells and affect auto-placement

## Complex Grid Layouts

Grids can be nested with other containers to create sophisticated layouts:

```python
with Grid():
    yield Static("Header", id="header")
    with Horizontal():
        yield Static("Sidebar")
        yield Static("Content")
    yield Static("Footer", id="footer")
```

## Grid vs Other Layouts

**When to use Grid:**
- Multi-dimensional layouts (rows AND columns)
- Widgets need to align both horizontally and vertically
- Complex dashboard-style interfaces
- Spanning cells for varied widget sizes

**When to use Vertical/Horizontal:**
- Single-direction stacking
- Simple linear flows
- Content that needs to scroll in one direction

## Best Practices

From [Textual Grid Documentation](https://textual.textualize.io/styles/grid/) (accessed 2025-11-02):

1. **Set layout explicitly:** Always specify `layout: grid` in CSS
2. **Define grid-size early:** Set columns and rows before adding widgets
3. **Use fr units for flexibility:** Enables responsive layouts
4. **Consider gutters:** Spacing improves visual clarity
5. **Test with different terminal sizes:** Grid adapts to screen dimensions

## Warning

Grid styles only work when `layout: grid` is set on the container.

## Sources

**Official Documentation:**
- [Textual Grid Styles](https://textual.textualize.io/styles/grid/) - Official grid reference (accessed 2025-11-02)
- [Textual Layout Guide](https://textual.textualize.io/guide/layout/) - Layout systems overview (accessed 2025-11-02)

**Tutorials:**
- [Python Textual: Build Beautiful UIs in the Terminal - Real Python](https://realpython.com/python-textual/) - Grid container examples (accessed 2025-11-02)

**Videos:**
- [Demo of the grid layout in Textual - YouTube](https://www.youtube.com/watch?v=zxWwkXjn-UI) - Visual demonstration
- [Advanced grid styling in Textual - YouTube](https://www.youtube.com/watch?v=p6aAiKfAmCI) - Advanced techniques
