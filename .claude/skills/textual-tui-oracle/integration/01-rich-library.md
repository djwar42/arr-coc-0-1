# Rich Library Integration in Textual

## Overview

Rich is the foundational rendering library that powers Textual's visual output. Created by Will McGugan (also the creator of Textual), Rich provides terminal formatting, styling, and rendering capabilities that Textual builds upon for its TUI framework.

**Rich GitHub Repository**: https://github.com/Textualize/rich
**Rich Documentation**: https://rich.readthedocs.io/en/latest/
**Version Coverage**: Rich 14.1.0+

**Key Relationship**: Textual uses Rich's rendering engine under the hood. While Textual provides widgets and application structure, Rich provides the low-level terminal output, styling system, and renderable protocol that makes it all possible.

## Core Rich Components Used in Textual

### 1. Console - Terminal Output Engine

The `Console` class is Rich's core rendering engine. Textual creates Console instances internally to handle all terminal output.

**Features Textual Relies On**:
- ANSI escape sequence generation
- Terminal capability auto-detection (color systems, dimensions)
- Word wrapping and text overflow handling
- Screen buffers and alternate screen mode
- Style and markup rendering

**Example - Direct Console Usage**:
```python
from rich.console import Console

console = Console()

# Basic printing with markup
console.print("Hello [bold magenta]World[/bold magenta]!")

# Print with styles
console.print("Important message", style="bold red")

# Logging with timestamps
console.log("Debug info", log_locals=True)
```

**Console Auto-Detection**:
```python
console = Console()

# Automatically detected properties
print(f"Terminal size: {console.size}")
print(f"Color system: {console.color_system}")  # truecolor, 256, standard, etc.
print(f"Is terminal: {console.is_terminal}")
print(f"Encoding: {console.encoding}")
```

**Color Systems**:
- `None` - No color
- `"standard"` - 16 colors (8 colors + bright variants)
- `"256"` - 256 color palette
- `"truecolor"` - 16.7 million colors (RGB)
- `"windows"` - Legacy Windows terminal (8 colors)

### 2. Text - Styled String Objects

The `Text` class is Rich's mutable styled string type. Textual widgets use Text internally for all styled content.

**Key Features**:
- Mutable string with style regions
- Style application via ranges or regular expressions
- Justify and overflow control
- ANSI code parsing

**Example - Building Styled Text**:
```python
from rich.text import Text
from rich.console import Console

console = Console()

# Method 1: Stylize ranges
text = Text("Hello, World!")
text.stylize("bold magenta", 0, 5)  # "Hello" in bold magenta
console.print(text)

# Method 2: Append with styles
text = Text()
text.append("Error: ", style="bold red")
text.append("File not found", style="italic")
console.print(text)

# Method 3: Assemble from parts
text = Text.assemble(
    ("SUCCESS", "bold green"),
    ": Operation completed",
)
console.print(text)

# From ANSI codes
text = Text.from_ansi("\033[1;35mHello\033[0m, World!")
console.print(text)
```

**Text Attributes**:
```python
text = Text(
    "Right-aligned text",
    justify="right",      # "left", "center", "right", "full"
    overflow="ellipsis",  # "fold", "crop", "ellipsis", "ignore"
    no_wrap=True,         # Prevent wrapping
    tab_size=4            # Characters per tab
)
```

**Textual Usage**: Widgets like `Label`, `Static`, and `Button` use Text objects internally to store and render styled content.

### 3. Style - Color and Formatting System

Rich's `Style` class defines text appearance: colors, bold, italic, underline, etc. Textual's CSS-like styling compiles down to Rich Style objects.

**Style Properties**:
- `color` - Foreground color
- `bgcolor` - Background color
- `bold`, `italic`, `underline`, `strike` - Text styles
- `dim`, `blink` - Terminal effects
- `reverse` - Swap foreground/background
- `link` - Clickable hyperlinks (terminal-dependent)

**Example - Creating Styles**:
```python
from rich.style import Style
from rich.console import Console

console = Console()

# Define styles
error_style = Style(color="red", bold=True)
success_style = Style(color="green", bold=True, italic=True)
link_style = Style(color="blue", underline=True)

# Use styles
console.print("Error message", style=error_style)
console.print("Success!", style=success_style)

# Style from string
console.print("Warning", style="bold yellow on red")

# Combine styles
base = Style(color="cyan")
emphasized = base + Style(bold=True)
console.print("Emphasized", style=emphasized)
```

**Textual CSS Integration**: When you write Textual CSS like `color: red; text-style: bold;`, it gets compiled to Rich Style objects.

### 4. Syntax - Code Highlighting

The `Syntax` class provides syntax highlighting using Pygments. Textual's `TextArea` widget uses this for code editing with highlighting.

**Example - Syntax Highlighting**:
```python
from rich.console import Console
from rich.syntax import Syntax

console = Console()

code = '''
def hello(name: str) -> str:
    return f"Hello, {name}!"
'''

# From string
syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
console.print(syntax)

# From file (auto-detects language)
syntax = Syntax.from_path("example.py", line_numbers=True)
console.print(syntax)

# Custom theme and background
syntax = Syntax(
    code,
    "python",
    theme="vim",
    line_numbers=True,
    background_color="default"  # Use terminal's background
)
console.print(syntax)
```

**Available Themes**: Any Pygments theme (monokai, vim, github, solarized-dark, etc.)

**Textual Usage**: The `TextArea` widget can optionally use Syntax for highlighting code in editors.

### 5. Table - Structured Data Display

Rich's `Table` class creates formatted tables. Textual's `DataTable` widget is built on similar concepts but with interactivity.

**Example - Creating Tables**:
```python
from rich.console import Console
from rich.table import Table

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Budget", justify="right")
table.add_column("Box Office", justify="right")

table.add_row(
    "Dec 20, 2019",
    "Star Wars: Rise of Skywalker",
    "$275,000,000",
    "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347"
)

console.print(table)
```

**Table Features**:
- Auto-sizing columns to fit terminal width
- Text wrapping within cells
- Column alignment (left, center, right)
- Rich markup in cells
- Nested tables
- Border styles (ascii, rounded, double, etc.)

**Textual DataTable vs Rich Table**:
- Rich Table: Static display, auto-layout
- Textual DataTable: Interactive, scrollable, selectable rows

### 6. Panel - Bordered Content

The `Panel` class wraps content in decorative borders. Textual's `Static` widget can display Panels, and many container widgets use panel-like rendering.

**Example - Creating Panels**:
```python
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Simple panel
console.print(Panel("Hello, World!"))

# With title and subtitle
console.print(Panel(
    "Panel content here",
    title="[bold]My Panel[/bold]",
    subtitle="Status: OK",
    border_style="green"
))

# Panel with Text object (justified)
text = Text("Right-aligned", justify="right")
console.print(Panel(text))

# Nested renderables
from rich.table import Table
table = Table.grid()
table.add_row("Field 1:", "Value 1")
table.add_row("Field 2:", "Value 2")
console.print(Panel(table, title="Info"))
```

**Border Styles**: `ascii`, `rounded`, `heavy`, `double`, `none`

### 7. Markdown - Formatted Text Rendering

Rich's `Markdown` class renders markdown content with terminal styling. Textual's `Markdown` widget wraps this for TUI display.

**Example - Rendering Markdown**:
```python
from rich.console import Console
from rich.markdown import Markdown

console = Console()

markdown_text = """
# Heading 1

## Heading 2

This is **bold** and *italic* text.

- List item 1
- List item 2
  - Nested item

```python
def example():
    return "code block"
```

> Blockquote text
"""

md = Markdown(markdown_text)
console.print(md)
```

**Textual Markdown Widget**: Wraps Rich's Markdown class and makes it scrollable/interactive.

### 8. Progress - Progress Bars and Spinners

Rich's `Progress` class creates progress bars and task tracking. Textual can embed Rich progress displays.

**Example - Progress Tracking**:
```python
from rich.progress import track
from time import sleep

# Simple progress bar
for step in track(range(100), description="Processing..."):
    sleep(0.01)

# Advanced progress with multiple bars
from rich.progress import Progress

with Progress() as progress:
    task1 = progress.add_task("[red]Downloading...", total=100)
    task2 = progress.add_task("[green]Processing...", total=100)
    task3 = progress.add_task("[cyan]Uploading...", total=100)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.2)
        sleep(0.02)
```

**Spinners** (for indeterminate progress):
```python
from rich.console import Console
from time import sleep

console = Console()

with console.status("[bold green]Working on tasks...", spinner="dots") as status:
    sleep(2)
    console.log("Task 1 complete")
    sleep(2)
    console.log("Task 2 complete")
```

**Available Spinners**: dots, line, arc, arrow, bouncingBar, bouncingBall, monkey, etc.
- Run `python -m rich.spinner` to see all options

### 9. Tree - Hierarchical Data Display

Rich's `Tree` class displays hierarchical data with guide lines. Useful for file trees, JSON structures, etc.

**Example - Creating Trees**:
```python
from rich.console import Console
from rich.tree import Tree

console = Console()

tree = Tree("📁 Project Root")
tree.add("📄 README.md")
tree.add("📄 LICENSE")

src = tree.add("📁 src")
src.add("📄 __init__.py")
src.add("📄 main.py")
src.add("📄 utils.py")

tests = tree.add("📁 tests")
tests.add("📄 test_main.py")
tests.add("📄 test_utils.py")

console.print(tree)
```

**Textual DirectoryTree Widget**: Built-in widget that extends this concept with interactivity.

### 10. Live Display - Dynamic Updates

Rich's `Live` class enables updating displays without scrolling. Critical for Textual's rendering model.

**Example - Live Updates**:
```python
from rich.live import Live
from rich.table import Table
from time import sleep
import random

def generate_table() -> Table:
    table = Table()
    table.add_column("ID")
    table.add_column("Value")

    for i in range(5):
        table.add_row(str(i), str(random.randint(0, 100)))

    return table

with Live(generate_table(), refresh_per_second=4) as live:
    for _ in range(20):
        sleep(0.5)
        live.update(generate_table())
```

**Textual's Rendering**: Textual apps run in a continuous Live context, updating the screen layout as widgets change.

## Console Protocol - Renderables

The Console Protocol is how Rich (and Textual) determines if an object can be rendered to the terminal.

### Simple Renderables - `__rich__` Method

Any class can become a renderable by implementing `__rich__()`:

```python
from rich.console import Console

class MyObject:
    def __rich__(self) -> str:
        return "[bold cyan]MyObject()[/bold cyan]"

console = Console()
console.print(MyObject())  # Renders with cyan bold style
```

**Return Types**: Can return string (with markup), Text, Table, Panel, or any other Rich renderable.

### Advanced Renderables - `__rich_console__` Method

For multi-line or complex rendering, implement `__rich_console__()`:

```python
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
from dataclasses import dataclass

@dataclass
class Student:
    id: int
    name: str
    age: int

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions
    ) -> RenderResult:
        yield f"[b]Student:[/b] #{self.id}"

        table = Table("Attribute", "Value")
        table.add_row("name", self.name)
        table.add_row("age", str(self.age))
        yield table

console = Console()
student = Student(1, "Alice", 20)
console.print(student)
```

**Generator Pattern**: Use `yield` to return multiple renderables that will be stacked vertically.

### Low-Level Rendering - Segments

For complete control, yield `Segment` objects:

```python
from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style

class ColorfulText:
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions
    ) -> RenderResult:
        yield Segment("Multi", Style(color="magenta"))
        yield Segment("Color", Style(color="green"))
        yield Segment("Text", Style(color="cyan"))

console = Console()
console.print(ColorfulText())
```

**Segments**: The lowest-level renderable - just text + style. All Rich renderables eventually compile to segments.

### Measuring Renderables - `__rich_measure__`

Implement this for custom size calculations:

```python
from rich.console import Console, ConsoleOptions
from rich.measure import Measurement

class ChessBoard:
    def __rich_measure__(
        self,
        console: Console,
        options: ConsoleOptions
    ) -> Measurement:
        # Minimum 8 chars wide, maximum available width
        return Measurement(8, options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        # Render chess board here
        yield "♔♕♖♗♘♙"
```

**When Needed**: Rich uses measurements to calculate table column widths, panel sizing, etc.

## Rich → Textual Rendering Pipeline

Understanding the rendering flow from Textual widgets to terminal output:

### 1. Textual Widget Rendering

```
Textual Widget (Label, Button, etc.)
    ↓
Widget.render() → returns Rich renderable (Text, Panel, Table)
    ↓
Textual compositor combines widget renderables
    ↓
Rich Console renders final output to terminal
```

### 2. Example Flow - Button Widget

```python
# Simplified Textual Button widget
class Button(Widget):
    def render(self) -> RenderResult:
        # Returns a Rich Panel
        return Panel(
            Text(self.label, justify="center"),
            border_style=self.border_style
        )
```

When Textual needs to display this button:
1. Calls `button.render()` → gets Rich Panel object
2. Panel contains Rich Text object
3. Rich Console converts Panel → Segments (text + style)
4. Segments → ANSI escape codes
5. ANSI codes → terminal display

### 3. Style Compilation

```
Textual CSS (color: red; text-style: bold;)
    ↓
Textual Style parser
    ↓
Rich Style object (Style(color="red", bold=True))
    ↓
Applied to Text/Segments during rendering
```

### 4. Screen Updates

```
Textual detects widget state change
    ↓
Marks widget as dirty (needs re-render)
    ↓
Next frame: calls widget.render()
    ↓
Rich Live display updates only changed segments
    ↓
Terminal shows updated content (no flicker)
```

**Rich Live Display**: Textual apps run inside a Rich `Live` context, enabling smooth updates without scrolling.

## When to Use Rich vs Textual Widgets

### Use Rich Directly When:

1. **Static Terminal Output** - CLI scripts, not TUI apps
   ```python
   # Good use of Rich
   from rich.console import Console
   console = Console()
   console.print("[green]Success:[/green] Operation completed")
   ```

2. **One-off Displays** - Tables, syntax highlighting in scripts
   ```python
   # Rich for script output
   from rich.table import Table
   table = Table()
   table.add_column("Name")
   table.add_column("Value")
   # ... populate and print once
   ```

3. **Progress Bars in CLI Tools**
   ```python
   # Rich progress for downloads, processing
   from rich.progress import track
   for item in track(items):
       process(item)
   ```

4. **Debugging/Logging Output**
   ```python
   # Rich for enhanced logging
   from rich.console import Console
   console = Console()
   console.log("Debug info", locals())
   ```

### Use Textual Widgets When:

1. **Interactive TUI Applications** - Apps with user input, navigation
   ```python
   # Textual for full TUI
   from textual.app import App
   from textual.widgets import Button, Input, DataTable

   class MyApp(App):
       def compose(self):
           yield Button("Click me")
           yield Input(placeholder="Enter text")
   ```

2. **Complex Layouts** - Multiple widgets, nested containers
   ```python
   # Textual for layout management
   class MyApp(App):
       def compose(self):
           yield Header()
           with Container():
               yield Sidebar()
               yield MainContent()
           yield Footer()
   ```

3. **Event-Driven UIs** - Responding to clicks, keypresses, timers
   ```python
   # Textual for interactivity
   class MyWidget(Widget):
       def on_click(self, event: Click) -> None:
           self.notify("Clicked!")
   ```

4. **Long-Running Apps** - TUI servers, monitors, dashboards
   ```python
   # Textual for persistent displays
   class Dashboard(App):
       def on_mount(self):
           self.set_interval(1, self.update_stats)
   ```

### Hybrid Approach - Rich Renderables in Textual

You can render Rich objects inside Textual widgets:

```python
from textual.app import App
from textual.widgets import Static
from rich.table import Table
from rich.syntax import Syntax

class RichInTextual(App):
    def compose(self):
        # Rich Table in Textual Static widget
        table = Table()
        table.add_column("Name")
        table.add_column("Value")
        table.add_row("Item 1", "100")
        yield Static(table)

        # Rich Syntax in Textual
        code = 'print("Hello")'
        syntax = Syntax(code, "python")
        yield Static(syntax)
```

**When to Use This Pattern**:
- You need Rich's advanced rendering (Syntax, complex Tables)
- But want Textual's layout/event system
- Displaying static content that doesn't need interaction

## Rich Features NOT Directly Used in Textual

Some Rich features are CLI-focused and not relevant in TUI apps:

### 1. Traceback Handler
Rich can pretty-print Python exceptions, but Textual apps typically handle errors differently (logging, error dialogs).

### 2. Console.input()
Rich has an input prompt, but Textual uses `Input` widget for user input.

### 3. Pager
Rich can page long output, but Textual uses scrollable containers (`VerticalScroll`, `ScrollView`).

### 4. Alternate Screen
Rich provides alternate screen mode, but Textual manages this automatically for all apps.

### 5. Export/Capture
Rich can export output as SVG/HTML. Textual apps are interactive, not exportable.

### 6. Pretty Printing (REPL)
Rich enhances Python REPL, but Textual apps don't run in REPL mode.

## Code Examples - Rich Renderables in Practice

### Example 1: Custom Textual Widget Using Rich

```python
from textual.widget import Widget
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

class StatusWidget(Widget):
    """Custom widget showing system status using Rich renderables"""

    def __init__(self, cpu: int, memory: int):
        super().__init__()
        self.cpu = cpu
        self.memory = memory

    def render(self) -> RenderResult:
        # Create Rich table
        table = Table.grid(padding=1)
        table.add_column(style="cyan", justify="right")
        table.add_column()

        # Add rows with colored values
        cpu_color = "green" if self.cpu < 80 else "red"
        mem_color = "green" if self.memory < 80 else "red"

        table.add_row("CPU:", f"[{cpu_color}]{self.cpu}%[/{cpu_color}]")
        table.add_row("Memory:", f"[{mem_color}]{self.memory}%[/{mem_color}]")

        # Wrap in panel
        return Panel(
            table,
            title="[bold]System Status[/bold]",
            border_style="blue"
        )

# Use in Textual app
from textual.app import App

class MonitorApp(App):
    def compose(self):
        yield StatusWidget(cpu=45, memory=67)
```

### Example 2: Rich Markdown in Textual

```python
from textual.app import App
from textual.widgets import Static
from rich.markdown import Markdown

class DocViewer(App):
    def compose(self):
        markdown_content = """
        # User Guide

        Welcome to the application!

        ## Features
        - Fast rendering
        - **Rich** formatting
        - `Code` highlighting
        """

        md = Markdown(markdown_content)
        yield Static(md)
```

### Example 3: Syntax Highlighting in Textual

```python
from textual.app import App
from textual.widgets import Static
from textual.containers import Container
from rich.syntax import Syntax

class CodeViewer(App):
    CSS = """
    Static {
        height: auto;
    }
    """

    def compose(self):
        code = '''
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

        syntax = Syntax(
            code,
            "python",
            theme="monokai",
            line_numbers=True,
            background_color="default"
        )

        yield Static(syntax)
```

### Example 4: Dynamic Rich Content Update

```python
from textual.app import App
from textual.widgets import Static
from textual.reactive import reactive
from rich.table import Table

class LiveStats(App):
    counter = reactive(0)

    def compose(self):
        yield Static(id="stats")

    def on_mount(self):
        self.set_interval(1, self.update_counter)

    def update_counter(self):
        self.counter += 1

    def watch_counter(self, new_value: int):
        # Create new Rich table on each update
        table = Table(title="Live Counter")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Count", str(new_value))
        table.add_row("Squared", str(new_value ** 2))

        # Update Static widget with new table
        static = self.query_one("#stats", Static)
        static.update(table)
```

## Console Markup vs Textual Markup

Rich and Textual both support markup, but use different syntax:

### Rich Markup (Console Markup)

```python
from rich.console import Console

console = Console()

# Rich markup syntax
console.print("[bold red]Error:[/bold red] File not found")
console.print("[link=https://example.com]Click here[/link]")
console.print(":thumbs_up: [green]Success![/green]")

# Escaped brackets
console.print("Code: \\[1, 2, 3\\]")  # Prints: Code: [1, 2, 3]
```

**Rich Markup Tags**:
- `[bold]`, `[italic]`, `[underline]`, `[strike]`
- `[red]`, `[blue]`, `[green]`, etc. (color names)
- `[#ff0000]` (hex colors)
- `[rgb(255,0,0)]` (RGB colors)
- `[on red]` (background colors)
- `[link=url]` (hyperlinks)
- `:emoji_name:` (emoji codes)

### Textual Markup (Widget Text)

Textual widgets that accept text typically support Rich markup:

```python
from textual.app import App
from textual.widgets import Label, Button

class MarkupDemo(App):
    def compose(self):
        # Rich markup works in Label
        yield Label("[bold cyan]Welcome![/bold cyan]")

        # Rich markup in Button
        yield Button("[green]▶[/green] Start")
```

**Best Practice**: Use Rich markup for dynamic content in Textual widgets that display text.

## Performance Considerations

### Rich Console Rendering Speed

Rich is optimized for speed:
- Segments are cached to avoid re-computation
- Console auto-detects terminal capabilities (once)
- Minimal ANSI escape codes generated
- Efficient string building

**Benchmark** (approximate):
- Simple text: ~10,000 lines/sec
- Complex tables: ~1,000 renders/sec
- Syntax highlighting: ~500 files/sec

### Textual Rendering Optimization

Textual optimizes rendering by:
- Only re-rendering "dirty" widgets
- Using Rich Live display (updates only changed segments)
- Batching screen updates
- Lazy rendering (off-screen widgets not rendered)

**When Using Rich in Textual**:
```python
# DON'T: Create new Rich object every render
class BadWidget(Widget):
    def render(self):
        table = Table()  # Created fresh each frame - slow!
        # ... populate table
        return table

# DO: Cache Rich objects when possible
class GoodWidget(Widget):
    def __init__(self):
        super().__init__()
        self._table = None
        self._data_version = 0

    def render(self):
        if self._table is None or self._data_changed:
            self._table = Table()
            # ... populate table
            self._data_changed = False
        return self._table
```

## Debugging with Rich

Rich is excellent for debugging, even in Textual apps:

### Logging Rich Output to File

```python
from rich.console import Console

# Log Rich output to file while developing Textual app
debug_console = Console(file=open("debug.log", "w"))

class MyApp(App):
    def on_mount(self):
        debug_console.log("App mounted", self.screen.size)

    def on_button_pressed(self, event):
        debug_console.log("Button pressed", event.button.id)
```

### Rich Inspect for Object Introspection

```python
from rich import inspect

# Inspect Textual widget
inspect(my_widget, methods=True)

# Inspect with documentation
inspect(my_widget, help=True, docs=True)
```

### Rich Traceback for Errors

```python
from rich.traceback import install

# Install Rich traceback handler
install(show_locals=True)

# Now all exceptions show with Rich formatting
# (even in Textual apps if errors are raised)
```

## Sources

**Primary Documentation**:
- [Rich GitHub Repository](https://github.com/Textualize/rich) - Main source code and README (accessed 2025-11-02)
- [Rich Documentation - Console Protocol](https://rich.readthedocs.io/en/latest/protocol.html) - Console Protocol specification (accessed 2025-11-02)
- [Rich Documentation - Console API](https://rich.readthedocs.io/en/latest/console.html) - Console class documentation (accessed 2025-11-02)
- [Rich Documentation - Rich Text](https://rich.readthedocs.io/en/latest/text.html) - Text class documentation (accessed 2025-11-02)
- [Rich Documentation - Syntax](https://rich.readthedocs.io/en/latest/syntax.html) - Syntax highlighting documentation (accessed 2025-11-02)

**Key Concepts**:
- Console Protocol: `__rich__()`, `__rich_console__()`, `__rich_measure__()`
- Renderables: Objects that can be displayed by Rich Console
- Segments: Lowest-level rendering primitive (text + style)
- Console: Terminal output engine with auto-detection
- Text: Mutable styled string class
- Live Display: Dynamic screen updates without scrolling

**Textual Integration**:
- Textual widgets return Rich renderables from `render()` method
- Textual CSS compiles to Rich Style objects
- Textual uses Rich Console internally for all terminal output
- Rich markup works in Textual text-displaying widgets
