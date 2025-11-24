# GitHub Examples Directory - Index

## Overview

The [Textualize/textual examples directory](https://github.com/Textualize/textual/tree/main/examples) contains official example applications demonstrating Textual framework capabilities. These examples range from simple demonstrations to complex real-world applications.

**Repository Location**: `/examples/` in the main Textual repository

## Key Examples from README

### Clock App (Time Display)

**Complexity**: Beginner

**Demonstrates**:
- Basic App structure
- CSS styling
- Compose method for widget layout
- Interval-based updates
- DateTime widget (Digits)

**Code Highlights**:
```python
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import Digits

class ClockApp(App):
    CSS = """
    Screen { align: center middle; }
    Digits { width: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Digits("")

    def on_ready(self) -> None:
        self.update_clock()
        self.set_interval(1, self.update_clock)

    def update_clock(self) -> None:
        clock = datetime.now().time()
        self.query_one(Digits).update(f"{clock:%T}")
```

**Concepts Covered**:
- App lifecycle (`on_ready()`)
- Periodic updates with `set_interval()`
- CSS-like styling for centering
- Widget querying with `query_one()`
- Reactive UI updates

## Example Categories

Based on the README widget gallery and documentation references, examples likely cover:

### Widget Demonstrations

**Buttons**:
- Basic button creation
- Button variants and styling
- Event handling
- Themed buttons

**Tree Controls**:
- Hierarchical data display
- Expandable/collapsible nodes
- Tree navigation
- Custom node rendering

**DataTables**:
- Tabular data display
- Sorting and filtering
- Cell selection
- Large dataset handling

**Input Widgets**:
- Text input fields
- Validation patterns
- Input masks
- Form handling

**ListView**:
- Scrollable lists
- Item selection
- Dynamic list updates
- Custom list items

**TextArea**:
- Multi-line text editing
- Syntax highlighting
- Code editing features
- Large text handling

### Layout Examples

**Grid Layout**:
- Responsive grid systems
- Auto-sizing cells
- Grid alignment

**Dock Layout**:
- Panel docking
- Resizable panes
- Edge-based positioning

**Vertical/Horizontal Layout**:
- Stack-based layouts
- Dynamic sizing
- Nested layouts

### Advanced Features

**Command Palette**:
- Fuzzy search implementation
- Custom commands
- Keyboard shortcuts
- Command categories

**Testing with Pilot**:
- Automated UI testing
- Interaction simulation
- Test assertions

**DevTools Integration**:
- Debug console usage
- Message logging
- Event inspection

**Web Serving**:
- `textual serve` examples
- Browser-based TUI apps
- Remote access patterns

## Running Examples

### Standard Method

```bash
# Clone repository
git clone https://github.com/Textualize/textual.git
cd textual/examples

# Run specific example
python example_name.py
```

### Built-in Demo

```bash
# Run Textual's built-in demo showcase
python -m textual
```

### With UV (No Installation)

```bash
# Run demo without installing Textual
uvx --python 3.12 textual-demo
```

## Example Structure

Typical example file structure:
```
examples/
├── basic/           # Simple beginner examples
├── widgets/         # Widget-specific demos
├── layouts/         # Layout system examples
├── advanced/        # Complex real-world apps
└── README.md        # Examples documentation
```

## Learning Path

**Recommended progression**:

1. **Start**: Clock app (intervals and updates)
2. **Widgets**: Button, Input, Label examples
3. **Layouts**: Grid and Dock examples
4. **Interactions**: Event handling examples
5. **Advanced**: DataTable, Tree, TextArea
6. **Complex**: Full application examples

## Widget Gallery Reference

The README showcases visual examples of:

- **Buttons**: Multiple button styles and states
- **Tree**: Expandable file/folder tree views
- **DataTables**: Sortable, selectable data grids
- **Inputs**: Text fields with validation
- **ListView**: Scrollable item lists
- **TextArea**: Multi-line text editor with syntax highlighting

Each widget has corresponding examples demonstrating:
- Basic usage
- Styling options
- Event handling
- Advanced features

## Key Patterns Demonstrated

### Async Integration

From README:
> "Textual is an asynchronous framework under the hood. Which means you can integrate your apps with async libraries — if you want to."

Examples show:
- Sync and async app patterns
- Integration with async libraries
- Non-async simple patterns for beginners

### CSS-like Styling

Examples demonstrate:
- Inline CSS strings
- External CSS files
- Theme application
- Responsive design

### Component Composition

Examples show:
- Widget composition patterns
- Custom widget creation
- Widget reusability
- Screen management

## Related Resources

**Examples Directory**: [https://github.com/Textualize/textual/tree/main/examples](https://github.com/Textualize/textual/tree/main/examples)

**Widget Gallery**: [https://textual.textualize.io/widget_gallery/](https://textual.textualize.io/widget_gallery/)

**Tutorial**: [https://textual.textualize.io/tutorial/](https://textual.textualize.io/tutorial/)

**Textual Demo Repository**: [https://github.com/textualize/textual-demo](https://github.com/textualize/textual-demo)

## Community Examples

Beyond the official examples, the community has created:

**Real-world Applications** (8,100+ repositories using Textual):
- Terminal file managers
- System monitoring tools
- Development utilities
- Chat applications
- Database browsers

See [community/01-showcase-applications.md](../community/01-showcase-applications.md) for curated real-world examples.

## Sources

**GitHub Repository**: [https://github.com/Textualize/textual](https://github.com/Textualize/textual) (accessed 2025-11-02)

**README.md**: [https://github.com/Textualize/textual/blob/main/README.md](https://github.com/Textualize/textual/blob/main/README.md) (accessed 2025-11-02)

**Examples Directory**: [https://github.com/Textualize/textual/tree/main/examples](https://github.com/Textualize/textual/tree/main/examples) (referenced 2025-11-02)

**Textual Demo**: [https://github.com/textualize/textual-demo](https://github.com/textualize/textual-demo) (referenced 2025-11-02)

**Search Results**: Google search for "Textual Python TUI framework examples directory overview" (2025-11-02)

---

**Cross-References**:
- See [community/00-github-repository-overview.md](../community/00-github-repository-overview.md) for repository statistics
- See [getting-started/01-official-tutorial.md](../getting-started/01-official-tutorial.md) for step-by-step learning
- See [widgets/](../widgets/) for detailed widget guides
- See [core-concepts/](../core-concepts/) for framework architecture
