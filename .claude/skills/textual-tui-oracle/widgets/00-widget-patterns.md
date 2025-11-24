# Widget Patterns

## Overview

Textual provides a rich library of pre-built widgets and patterns for composing them. This guide documents common widget composition patterns, usage examples, and best practices.

From [RealPython](https://realpython.com/python-textual/) (accessed 2025-11-02):
- Comprehensive widget coverage including Static, Label, Button, Input widgets
- Widget lifecycle and styling approaches
- Event handling and action patterns

From [mathspp - Textual for Beginners](https://mathspp.com/blog/textual-for-beginners) (accessed 2025-11-02):
- Widget composition fundamentals
- Compound widget creation patterns
- Custom message handling for widget communication

---

## Basic Widget Categories

### Display Widgets (Read-Only)

**Static Widget**: Manages rectangular area with optional text
- Supports Rich text renderables (colors, bold, italic, flashing)
- Can display emojis and Unicode characters
- Supports rich text markup like `[bold red]text[/bold red]`

**Label Widget**: Specialized for displaying text
- Adjusts width automatically (default behavior)
- Similar to Static but optimized for labels
- Useful for descriptive text and headings

Example from RealPython:
```python
from textual.widgets import Static, Label

static = Static("I am a [bold red]Static[/bold red] widget!")
label = Label("I am a [yellow italic]Label[/yellow italic] widget!")
```

### Input Widgets (Interactive)

**Button Widget**: Clickable button with multiple variants
- Default variant: black and white
- Variants available: primary, success, warning, error
- Convenience constructors: `Button.success()`, `Button.warning()`, `Button.error()`

```python
from textual.widgets import Button

yield Button("Click me!")  # Default
yield Button("Primary!", variant="primary")
yield Button.success("Success!")
yield Button.warning("Warning!")
yield Button.error("Error!")
```

**Input Widget**: Text field for user entry
- `placeholder`: Text shown when empty
- `password=True`: Hides typed text
- `type="number"`: Accepts only numeric input
- `tooltip`: Hover help text

```python
from textual.widgets import Input

yield Input(placeholder="Type your text here")
yield Input(placeholder="Password", password=True)
yield Input(
    placeholder="Type a number here",
    type="number",
    tooltip="Digits only please!"
)
```

### Advanced Widgets

**RichLog**: Vertically aligned log display
- Accepts Rich renderables
- Supports markup when configured
- Useful for message displays and chat interfaces

```python
from textual.widgets import RichLog

richlog = RichLog(markup=True)
richlog.write("Message text")
```

**ProgressBar**: Display task progress
- Range: 0-100 by default
- Updateable during long operations
- Great for showing operation status

```python
from textual.widgets import ProgressBar

progress = ProgressBar(total=100, id="progress")
progress.update(progress=50)  # Update to 50%
```

**Tree**: Hierarchical/collapsible display
- Display nested data structures
- Expandable/collapsible nodes
- Great for file browsers, settings trees

**Switch**: Toggle on/off switch
- Binary state (true/false)
- Clean visual representation
- Better than checkbox for binary choices

**Checkbox**: Boolean selection
- Traditional checkbox style
- Can be grouped with labels
- Event-driven state changes

**TabbedContent**: Multi-page/tabbed interface
- Switch between different content areas
- Tab headers for navigation
- Cleaner than multiple screens for related content

From ArjanCodes (accessed 2025-11-02):
```python
from textual.widgets import Checkbox, Switch, ProgressBar

# Checkbox pattern
yield Checkbox(label="Accept Terms and Conditions", id="checkbox")

# Switch pattern - better for binary toggles
yield Switch(id="toggle")

# Progress bar with async update
progress = ProgressBar(total=100)
for i in range(101):
    progress.update(progress=i)
    await sleep(0.1)
```

---

## Widget Composition Patterns

### Pattern 1: Simple Widget Composition

Widgets are yielded in `compose()` method in order they should appear:

```python
from textual.app import App
from textual.widgets import Header, Button, Input, Footer

class SimpleApp(App):
    def compose(self):
        yield Header(show_clock=True)
        yield Button("Click me!")
        yield Input(placeholder="Name:")
        yield Footer()
```

Widgets stack vertically by default (default Vertical layout).

### Pattern 2: Compound Widget Creation

Create reusable widget combinations by inheriting from Widget:

From mathspp (accessed 2025-11-02):
```python
from textual.widget import Widget
from textual.widgets import Input, Label

class LabelledInput(Widget):
    DEFAULT_CSS = """
    LabelledInput {
        height: 4;
    }
    """

    def __init__(self, label):
        super().__init__()
        self.label = label

    def compose(self):
        yield Label(f"{self.label}:")
        yield Input(placeholder=self.label.lower())

# Usage
class MyApp(App):
    def compose(self):
        yield LabelledInput("Name")
        yield LabelledInput("Email")
```

**Key points for compound widgets**:
- Always call `super().__init__()` in `__init__`
- Use `DEFAULT_CSS` to set default dimensions
- Expose sub-widget customization through constructor parameters

### Pattern 3: Widget Identification and Querying

Use `id` and `classes` for targeting widgets:

```python
from textual.app import App
from textual.widgets import Button, Input

class QueryApp(App):
    def compose(self):
        yield Input(id="username_input")
        yield Button("Submit", id="submit_btn")
        yield Button("Cancel", id="cancel_btn", classes="secondary-btn")

    def on_button_pressed(self, event):
        # Query single widget by ID
        username = self.query_one("#username_input", Input).value

        # Query multiple widgets by class
        secondary_buttons = self.query(".secondary-btn", Button)
```

**Querying methods**:
- `query_one(selector)`: Find single widget (raises error if 0 or 2+ found)
- `query(selector)`: Find multiple widgets (returns iterable)
- Selectors: CSS-style (type, ID with `#`, classes with `.`)

---

## Widget State Management Patterns

### Pattern 4: Reactive Attributes

From mathspp (accessed 2025-11-02):
```python
from textual.app import App
from textual.reactive import reactive
from textual.widgets import Button, Label

class ReactiveApp(App):
    counter = reactive(0)

    def __init__(self):
        self.label = Label()
        super().__init__()

    def compose(self):
        yield self.label
        yield Button("+1")

    def on_button_pressed(self):
        self.counter += 1

    def watch_counter(self):
        self.label.update(str(self.counter))

# Watch methods are called automatically when reactive attribute changes
```

**Reactive pattern benefits**:
- Automatically updates UI when state changes
- Decouples state from UI updates
- Watch methods follow `watch_<attribute_name>` convention

### Pattern 5: Widget Lifecycle Events

```python
from textual.app import App
from textual.widgets import Button

class LifecycleApp(App):
    def on_mount(self):
        # Called after widget is mounted (earliest safe point)
        # Use for initialization that depends on other widgets
        pass

    def on_unmount(self):
        # Called when widget is removed
        # Use for cleanup
        pass
```

---

## Widget Styling Patterns

### Pattern 6: CSS-Based Styling

From RealPython (accessed 2025-11-02):

Create external `.tcss` file:

```css
/* app.tcss */
Static {
    background: blue;
    border: solid white;
    padding: 1 1;
    margin: 2 2;
}

#special_label {
    color: black;
    background: red;
}

.highlight {
    background: yellow;
    color: black;
}
```

Reference in app:
```python
from textual.app import App
from textual.widgets import Static

class StyledApp(App):
    CSS_PATH = "app.tcss"

    def compose(self):
        yield Static("Default style")
        yield Static("Special", id="special_label")
        yield Static("Highlighted", classes="highlight")
```

### Pattern 7: Inline Styling

```python
from textual.app import App
from textual.widgets import Static

class InlineStyleApp(App):
    def compose(self):
        static = Static("Styled widget")
        yield static

    def on_mount(self):
        static = self.query_one(Static)
        static.styles.background = "blue"
        static.styles.border = ("solid", "white")
        static.styles.padding = (1, 1)
```

---

## Common Widget Patterns in Practice

### Pattern 8: Chat Interface

From ArjanCodes (accessed 2025-11-02):
```python
from typing import Literal
from textual.app import App, ComposeResult, on
from textual.widgets import Input, Header, Footer, RichLog
from rich.panel import Panel
from rich.align import Align

class ChatApp(App):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="Chat")
        yield RichLog(markup=True, id="output")
        yield Input(placeholder="Message", id="input")
        yield Footer()

    async def write_message(self, message: str, title: str,
                          side: Literal["left", "right"], colour: str):
        msg = Align(
            Panel(
                message,
                title=f"[{colour}]{title}[/]",
                title_align=side,
                width=max(self.app.console.width // 3, 80)
            ),
            side
        )
        self.query_one(RichLog).write(msg, expand=True)

    @on(Input.Submitted)
    async def submit(self, message: Input.Submitted) -> None:
        await self.write_message(message.value, "You", "left", "blue")
        self.query_one("#input", Input).clear()
```

### Pattern 9: Form with Multiple Inputs

```python
from textual.widget import Widget
from textual.widgets import Input, Label

class FormField(Widget):
    DEFAULT_CSS = """
    FormField {
        height: 3;
    }
    """

    def __init__(self, field_name: str, field_type: str = "text"):
        super().__init__()
        self.field_name = field_name
        self.field_type = field_type

    def compose(self):
        yield Label(f"{self.field_name}:")
        yield Input(
            id=f"{self.field_name.lower()}_input",
            type=self.field_type if self.field_type != "password" else "text",
            placeholder=f"Enter {self.field_name.lower()}",
            password=(self.field_type == "password")
        )

# Usage
class FormApp(App):
    def compose(self):
        yield FormField("Username")
        yield FormField("Email", "email")
        yield FormField("Password", "password")
```

---

## Dynamic Widget Addition/Removal

### Pattern 10: Dynamic Widget Management

From mathspp (accessed 2025-11-02):
```python
from textual.app import App
from textual.widgets import Button, Label

class DynamicApp(App):
    BINDINGS = [("n", "add_label", "Add")]

    def compose(self):
        yield Button("Add Label")

    def on_button_pressed(self):
        self.mount(Label("Dynamically added label"))

    def action_add_label(self):
        self.mount(Label("Added via action"))
```

**mount() method**: Add widget to app at runtime
**remove() method**: Remove widget from app

From mathspp (accessed 2025-11-02):
```python
class DeletableWidget(Widget):
    class DeletionRequest(Message):
        def __init__(self, widget):
            super().__init__()
            self.widget = widget

    def compose(self):
        yield Button("Delete me")

    def on_button_pressed(self):
        self.post_message(self.DeletionRequest(self))

class AppWithDeletion(App):
    def compose(self):
        yield DeletableWidget()

    def on_deletable_widget_deletion_request(self, message):
        message.widget.remove()
```

---

## Widget Communication Patterns

### Pattern 11: Custom Messages for Widget Communication

From mathspp (accessed 2025-11-02):
```python
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

class TodoItem(Widget):
    class Delete(Message):
        def __init__(self, item):
            super().__init__()
            self.item = item

    class Edit(Message):
        def __init__(self, item):
            super().__init__()
            self.item = item

    def compose(self):
        yield Button("Delete", id="delete")
        yield Button("Edit", id="edit")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "delete":
            self.post_message(self.Delete(self))
        elif event.button.id == "edit":
            self.post_message(self.Edit(self))

class TodoApp(App):
    def on_todo_item_delete(self, message):
        message.item.remove()

    def on_todo_item_edit(self, message):
        # Handle edit
        pass
```

---

## Common Pitfalls

1. **Compound widgets too tall**: Set `DEFAULT_CSS` with `height: auto` or fixed height
2. **Not calling `super().__init__()`**: Always required in compound widget constructors
3. **Reactive attributes before mount**: Initialize dependent widgets in `__init__`, not `__init__` before super
4. **Query before mount**: Widgets may not be queryable until `on_mount()`
5. **Widget not appearing**: Check height/width constraints in CSS or styles

---

## Sources

**Web Research:**
- [RealPython - Python Textual Tutorial](https://realpython.com/python-textual/) (accessed 2025-11-02)
- [mathspp - Textual for Beginners](https://mathspp.com/blog/textual-for-beginners) (accessed 2025-11-02)
- [ArjanCodes - Guide to Building Interactive Terminal Apps with Textual](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/) (accessed 2025-11-02)
