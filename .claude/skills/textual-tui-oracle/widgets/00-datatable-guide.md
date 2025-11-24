# DataTable Widget - Comprehensive Guide

## Overview

The DataTable widget is a powerful component for displaying tabular data in Textual TUI applications. It provides features for navigating data with a cursor, responding to mouse clicks, updating cell values, and managing rows and columns dynamically.

**From [DataTable - Textual Documentation](https://textual.textualize.io/widgets/data_table/) (accessed 2025-11-02)**

## Key Features

### Core Capabilities
- Display text in structured table format
- Cursor navigation (keyboard and mouse)
- Dynamic data updates
- Row and column addition/deletion
- Cell-level data manipulation
- Mouse click event handling
- Sorting support
- Fixed column/row headers
- Styling and theming

### Performance Characteristics
- Efficient rendering for large datasets
- Virtual scrolling for memory efficiency
- Optimized for terminal display

## Basic Usage

### Simple DataTable Example

```python
from textual.app import App, ComposeResult
from textual.widgets import DataTable

class TableApp(App):
    def compose(self) -> ComposeResult:
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Name", "Age", "City")
        table.add_row("Alice", 30, "NYC")
        table.add_row("Bob", 25, "LA")
        table.add_row("Charlie", 35, "Chicago")
```

## Working with DataTable

### Adding Columns

```python
# Simple column names
table.add_columns("Col1", "Col2", "Col3")

# With labels (for display)
table.add_column("user_id", label="User ID")
table.add_column("username", label="Username")
```

### Adding Rows

```python
# Add single row
row_key = table.add_row("Value1", "Value2", "Value3")

# Add multiple rows
for item in data:
    table.add_row(item.name, item.value, item.status)

# Add row with specific key
table.add_row("Alice", 30, key="user_1")
```

### Updating Data

```python
# Update cell by coordinate
table.update_cell(row_key, column_key, "New Value")

# Update cell at cursor position
table.update_cell_at(table.cursor_coordinate, "Updated")

# Clear table
table.clear()
```

### Removing Rows/Columns

```python
# Remove specific row
table.remove_row(row_key)

# Remove specific column
table.remove_column(column_key)

# Clear all data (keeps structure)
table.clear()
```

## Navigation and Selection

### Cursor Movement

The DataTable widget provides built-in keyboard navigation:

**Key Bindings:**
- Arrow keys: Move cursor up/down/left/right
- Home/End: Jump to first/last column
- Page Up/Down: Scroll by page
- Enter: Select row (triggers RowSelected event)

### Mouse Interaction

```python
@on(DataTable.CellSelected)
def handle_cell_click(self, event: DataTable.CellSelected) -> None:
    """Handle cell selection via mouse or keyboard."""
    row_key = event.row_key
    column_key = event.column_key
    value = event.value
    self.notify(f"Selected: {value}")
```

## Reactive Attributes

From the official documentation, DataTable provides these reactive attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `cursor_type` | str | "cell" | Type of cursor ("cell", "row", "column", "none") |
| `show_cursor` | bool | True | Whether to show the cursor |
| `show_header` | bool | True | Whether to show column headers |
| `show_row_labels` | bool | True | Whether to show row labels |
| `fixed_rows` | int | 0 | Number of fixed rows at top |
| `fixed_columns` | int | 0 | Number of fixed columns at left |
| `zebra_stripes` | bool | False | Alternate row colors |

## Messages and Events

### Key Events

**CellSelected**
```python
@on(DataTable.CellSelected)
def on_cell_selected(self, event: DataTable.CellSelected) -> None:
    """Fired when a cell is selected."""
    pass
```

**RowSelected**
```python
@on(DataTable.RowSelected)
def on_row_selected(self, event: DataTable.RowSelected) -> None:
    """Fired when a row is selected (Enter key)."""
    pass
```

**CellHighlighted**
```python
@on(DataTable.CellHighlighted)
def on_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
    """Fired when cursor moves to a new cell."""
    pass
```

## Advanced Features

### Sorting

```python
# Sort by column
table.sort(column_key)

# Reverse sort
table.sort(column_key, reverse=True)

# Custom sort function
table.sort(column_key, key=lambda x: int(x))
```

### Fixed Headers

```python
# Fix first row as header
table.fixed_rows = 1

# Fix first column
table.fixed_columns = 1
```

### Styling

```python
# In CSS
DataTable {
    background: $surface;
}

DataTable > .datatable--header {
    background: $primary;
    color: $text;
}

DataTable > .datatable--cursor {
    background: $secondary;
}

DataTable > .datatable--odd-row {
    background: $surface-darken-1;
}
```

### Zebra Stripes

```python
# Enable alternating row colors
table.zebra_stripes = True
```

## Component Classes

The DataTable widget provides these component classes for styling:

| Class | Description |
|-------|-------------|
| `datatable--header` | Target the header row |
| `datatable--cursor` | Target the cursor |
| `datatable--odd-row` | Target odd-numbered rows |
| `datatable--even-row` | Target even-numbered rows |
| `datatable--fixed` | Target fixed rows/columns |
| `datatable--fixed-cursor` | Target cursor in fixed area |

## Practical Examples

### Dynamic Data Loading

```python
class DataApp(App):
    def compose(self) -> ComposeResult:
        yield DataTable()

    async def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("ID", "Status", "Progress")

        # Simulate loading data
        for i in range(100):
            table.add_row(
                f"Task-{i:03d}",
                "Running" if i % 2 else "Complete",
                f"{i}%"
            )
```

### Interactive Table

```python
@on(DataTable.RowSelected)
def handle_row_selection(self, event: DataTable.RowSelected) -> None:
    """Handle row selection and show details."""
    row_data = event.row_data
    self.show_detail_screen(row_data)

@on(DataTable.CellHighlighted)
def update_status_bar(self, event: DataTable.CellHighlighted) -> None:
    """Update status bar with current cell info."""
    self.query_one(Footer).highlight_key = f"Cell: {event.value}"
```

## Performance Tips

From [DataTable-like widget discussion](https://github.com/Textualize/textual/discussions/2780) (accessed 2025-11-02):

1. **Use virtual scrolling**: DataTable only renders visible rows
2. **Batch updates**: Use `update_cell` multiple times before refresh
3. **Key management**: Use meaningful row/column keys for fast lookup
4. **Limit columns**: Too many columns can impact rendering performance
5. **Consider textual-fastdatatable**: For extremely large datasets, see [textual-fastdatatable](https://pypi.org/project/textual-fastdatatable/) for performance-focused reimplementation

## Common Patterns

### CSV Data Display

```python
import csv

def load_csv_to_table(self, filepath: str) -> None:
    table = self.query_one(DataTable)

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        table.add_columns(*headers)

        for row in reader:
            table.add_row(*row)
```

### Search and Filter

```python
def filter_table(self, search_term: str) -> None:
    """Filter table rows based on search term."""
    table = self.query_one(DataTable)
    table.clear()

    # Re-add matching rows
    for row_data in self.original_data:
        if search_term.lower() in str(row_data).lower():
            table.add_row(*row_data)
```

## Sources

**Official Documentation:**
- [DataTable Widget - Textual](https://textual.textualize.io/widgets/data_table/) (accessed 2025-11-02)

**Community Resources:**
- [DataTable-like widget discussion](https://github.com/Textualize/textual/discussions/2780) - GitHub (accessed 2025-11-02)
- [Textual Widget Gallery](https://textual.textualize.io/widget_gallery/) (accessed 2025-11-02)

**Related Widgets:**
- [ListView](https://textual.textualize.io/widgets/list_view/) - For simpler list displays
- [Tree](https://textual.textualize.io/widgets/tree/) - For hierarchical data

## See Also

- [widgets/01-tree-widget-guide.md](01-tree-widget-guide.md) - Tree widget patterns
- [widgets/02-input-validation.md](02-input-validation.md) - Input widget with validation
- [core-concepts/](../core-concepts/) - Understanding Textual widget architecture
