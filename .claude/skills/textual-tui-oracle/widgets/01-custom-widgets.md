# Custom Widget Creation

## Overview

Creating custom widgets by extending the Widget class is one of the most powerful patterns in Textual. This guide covers best practices, common patterns, and advanced techniques.

From [mathspp - Textual for Beginners](https://mathspp.com/blog/textual-for-beginners) (accessed 2025-11-02):
- Compound widget fundamentals and inheritance patterns
- Widget composition and lifecycle management
- Custom message systems for widget communication

---

## Basic Custom Widget Structure

### The Minimal Custom Widget

```python
from textual.widget import Widget

class MyCustomWidget(Widget):
    """A basic custom widget."""
    pass
```

This widget does nothing but is composable. For useful widgets, override `compose()`:

```python
from textual.widget import Widget
from textual.widgets import Label, Button

class SimpleCustom(Widget):
    """Custom widget with a label and button."""

    def compose(self):
        yield Label("My custom widget")
        yield Button("Click me")
```

---

## Widget Initialization and Configuration

### Pattern 1: Configurable Custom Widget

From mathspp (accessed 2025-11-02):

```python
from textual.widget import Widget
from textual.widgets import Label, Input

class LabelledInput(Widget):
    """Custom widget combining label and input."""

    DEFAULT_CSS = """
    LabelledInput {
        height: 4;
    }
    """

    def __init__(self, label_text: str, placeholder: str = ""):
        super().__init__()  # CRITICAL: Always call super().__init__()
        self.label_text = label_text
        self.placeholder = placeholder

    def compose(self):
        yield Label(f"{self.label_text}:")
        yield Input(placeholder=self.placeholder or self.label_text.lower())

# Usage
class MyApp(App):
    def compose(self):
        yield LabelledInput("Username", "Enter username")
        yield LabelledInput("Password", "Enter password")
```

**Critical points**:
1. Always call `super().__init__()` first in custom `__init__`
2. Store configuration in instance variables
3. Use configuration in `compose()` method
4. Avoid complex logic in compose - keep it declarative

### Pattern 2: Widget with Default Styling

```python
from textual.widget import Widget
from textual.widgets import Static

class StyledBox(Widget):
    """Custom widget with built-in styling."""

    DEFAULT_CSS = """
    StyledBox {
        border: solid blue;
        padding: 1 2;
        margin: 1 1;
        background: $panel;
        height: auto;
    }
    """

    def __init__(self, content: str, variant: str = "default"):
        super().__init__()
        self.content = content
        self.variant = variant

    def compose(self):
        yield Static(self.content)
```

**CSS Best Practices**:
- Define sensible defaults in `DEFAULT_CSS`
- Use Textual CSS variables like `$panel`, `$primary`, etc.
- Set explicit heights when needed to prevent layout issues
- Can be overridden by external CSS files

---

## Reactive Attributes in Custom Widgets

### Pattern 3: State Management with Reactives

From mathspp (accessed 2025-11-02):

```python
from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Label, Button
from textual.containers import Horizontal

class Counter(Widget):
    """Custom counter widget with reactive state."""

    DEFAULT_CSS = """
    Counter {
        height: 3;
    }
    """

    count = reactive(0)  # Reactive attribute

    def __init__(self):
        super().__init__()
        self.display_label = Label("0")

    def compose(self):
        with Horizontal():
            yield Button("-", id="minus")
            yield self.display_label
            yield Button("+", id="plus")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "plus":
            self.count += 1
        elif event.button.id == "minus":
            self.count -= 1

    def watch_count(self, new_value: int):
        """Called automatically when count changes."""
        self.display_label.update(str(new_value))

# Usage
class CounterApp(App):
    def compose(self):
        yield Counter()
```

**Reactive pattern benefits**:
- Automatic UI updates when state changes
- Clean separation of state and UI
- Watch methods called before UI is in inconsistent state

### Pattern 4: Multiple Reactive Attributes

```python
from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Label, Input
from textual.containers import Vertical

class ValidatedInput(Widget):
    """Input widget with reactive validation."""

    DEFAULT_CSS = """
    ValidatedInput {
        height: 4;
    }
    """

    value = reactive("")
    is_valid = reactive(True)

    def __init__(self, validator=None, label_text="Input"):
        super().__init__()
        self.validator = validator or (lambda x: len(x) > 0)
        self.label_text = label_text
        self.status_label = Label("")

    def compose(self):
        yield Label(self.label_text)
        yield Input(id="input")
        yield self.status_label

    def on_input_changed(self, event: Input.Changed):
        self.value = event.value

    def watch_value(self, new_value: str):
        """Validate on value change."""
        self.is_valid = self.validator(new_value)

    def watch_is_valid(self, is_valid: bool):
        """Update status display."""
        if is_valid:
            self.status_label.update("[green]Valid[/green]")
        else:
            self.status_label.update("[red]Invalid[/red]")
```

---

## Custom Messages and Communication

### Pattern 5: Widget-to-App Communication

From mathspp (accessed 2025-11-02):

```python
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

class DeleteButton(Widget):
    """Button widget that communicates deletion intent."""

    class Deleted(Message):
        """Posted when delete button is pressed."""
        def __init__(self, widget: "DeleteButton"):
            super().__init__()
            self.widget = widget

    def compose(self):
        yield Button("Delete", id="delete_btn")

    def on_button_pressed(self):
        self.post_message(self.Deleted(self))

class AppWithDeletion(App):
    def compose(self):
        yield DeleteButton()

    def on_delete_button_deleted(self, message: DeleteButton.Deleted):
        """Handle deletion message."""
        message.widget.remove()
```

**Message naming convention**: `on_<widget_class_snake_case>_<message_name_snake_case>`

### Pattern 6: Complex Message with Payload

```python
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input

class FormWidget(Widget):
    """Form that communicates submission."""

    class Submitted(Message):
        """Posted when form is submitted."""
        def __init__(self, widget: "FormWidget", data: dict):
            super().__init__()
            self.widget = widget
            self.data = data

    def __init__(self):
        super().__init__()
        self.name_input = Input(id="name")
        self.email_input = Input(id="email")

    def compose(self):
        yield Label("Name:")
        yield self.name_input
        yield Label("Email:")
        yield self.email_input
        yield Button("Submit")

    def on_button_pressed(self):
        data = {
            "name": self.name_input.value,
            "email": self.email_input.value,
        }
        self.post_message(self.Submitted(self, data))

class FormApp(App):
    def on_form_widget_submitted(self, message: FormWidget.Submitted):
        # Access submitted data
        print(f"Form data: {message.data}")
```

---

## Advanced Widget Patterns

### Pattern 7: Self-Removing Widget

From mathspp (accessed 2025-11-02):

```python
from textual.widget import Widget
from textual.widgets import Button

class DismissibleWidget(Widget):
    """Widget that can remove itself."""

    class DismissalRequested(Message):
        def __init__(self, widget):
            super().__init__()
            self.widget = widget

    def __init__(self, content: str):
        super().__init__()
        self.content = content

    def compose(self):
        yield Label(self.content)
        yield Button("Close")

    def on_button_pressed(self):
        # Request removal instead of removing directly
        self.post_message(self.DismissalRequested(self))

class AppWithDismissible(App):
    BINDINGS = [("a", "add_widget", "Add")]

    def action_add_widget(self):
        self.mount(DismissibleWidget("Dismissible item"))

    def on_dismissible_widget_dismissal_requested(self, message):
        message.widget.remove()
```

**Why post a message instead of calling remove()?**
- Allows parent app to perform cleanup
- Enables undo functionality
- Follows event-driven architecture

### Pattern 8: Widget with Lifecycle Hooks

```python
from textual.widget import Widget
from textual.widgets import Label

class LifecycleWidget(Widget):
    """Widget demonstrating lifecycle hooks."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        print(f"{self.name}: __init__ called")

    def on_mount(self):
        """Called when widget is mounted to the DOM."""
        print(f"{self.name}: on_mount called")

    def on_unmount(self):
        """Called when widget is removed from the DOM."""
        print(f"{self.name}: on_unmount called")

    def compose(self):
        yield Label(f"Widget: {self.name}")
```

**Lifecycle order**:
1. `__init__()` - Object creation
2. `compose()` - Define UI structure
3. `on_mount()` - After DOM insertion, safe to query other widgets
4. `on_unmount()` - Before removal, cleanup resources

### Pattern 9: Composite Widget with Sub-widgets

```python
from textual.widget import Widget
from textual.widgets import Label, Button, Input
from textual.containers import Vertical, Horizontal

class TodoItemWidget(Widget):
    """Complex widget composed of multiple sub-widgets."""

    DEFAULT_CSS = """
    TodoItemWidget {
        height: 2;
        border: solid green;
        padding: 1;
    }

    TodoItemWidget > Horizontal {
        height: 100%;
    }

    #item-content {
        width: 1fr;
    }
    """

    def __init__(self, task: str, due_date: str):
        super().__init__()
        self.task = task
        self.due_date = due_date
        # Store references for later access
        self.task_label = Label(task)
        self.date_label = Label(due_date)

    def compose(self):
        with Horizontal():
            yield self.task_label
            yield self.date_label
            yield Button("Edit", id="edit")
            yield Button("Done", id="done")

    def update_task(self, new_task: str):
        """Update task after creation."""
        self.task = new_task
        self.task_label.update(new_task)

    def update_date(self, new_date: str):
        """Update due date after creation."""
        self.due_date = new_date
        self.date_label.update(new_date)

# Usage
class TodoApp(App):
    def compose(self):
        yield TodoItemWidget("Buy groceries", "2025-01-15")
        yield TodoItemWidget("Write documentation", "2025-01-20")
```

---

## Widget Styling Patterns

### Pattern 10: Themeable Custom Widget

```python
from textual.widget import Widget
from textual.widgets import Static

class ThemedBox(Widget):
    """Widget with customizable theme."""

    DEFAULT_CSS = """
    ThemedBox {
        border: solid blue;
        background: $panel;
        padding: 1;
        height: auto;
    }

    ThemedBox.danger {
        border: solid red;
        background: $error;
    }

    ThemedBox.success {
        border: solid green;
        background: $success;
    }
    """

    def __init__(self, content: str, theme: str = "default"):
        super().__init__()
        self.content = content
        self.theme = theme

    def compose(self):
        yield Static(self.content)

    def on_mount(self):
        # Apply theme as CSS class
        if self.theme != "default":
            self.add_class(self.theme)
```

---

## Common Widget Patterns Summary

| Pattern | Use Case | Example |
|---------|----------|---------|
| Simple Display | Show static content | Status display, headings |
| Configurable | Customize on creation | Labeled inputs, themed boxes |
| Reactive State | Dynamic UI updates | Counters, forms, data displays |
| Custom Messages | Widget communication | Delete buttons, form submission |
| Composite | Complex combinations | Todo items, chat bubbles |
| Themeable | Consistent styling | Alerts, notifications |
| Lifecycle | Resource management | Data loading, cleanup |

---

## Best Practices

1. **Always call `super().__init__()` first**: Non-negotiable for widget initialization
2. **Keep compose() simple**: Don't do heavy logic in compose
3. **Store widget references**: Save references in `__init__` for later access
4. **Use DEFAULT_CSS**: Define sensible defaults
5. **Reactive > imperative**: Use reactive attributes instead of manual updates
6. **Post messages**: Don't call methods on parent directly
7. **Test initialization order**: Ensure dependencies are available
8. **Document custom messages**: Clear contracts for widget communication

---

## Sources

**Web Research:**
- [mathspp - Textual for Beginners](https://mathspp.com/blog/textual-for-beginners) (accessed 2025-11-02)
  - Compound widget creation
  - Custom messages
  - Lifecycle management
  - Reactive attributes
