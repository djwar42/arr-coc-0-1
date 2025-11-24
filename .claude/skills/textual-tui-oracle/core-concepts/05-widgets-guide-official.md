# Official Widgets Guide - Textual

**Source**: Official Textual Documentation
**Accessed**: 2025-11-02
**URL**: https://textual.textualize.io/guide/widgets/

## Overview

Widgets are the fundamental building blocks of Textual applications. A widget is a self-contained component responsible for managing a rectangular region of the screen. Widgets may respond to events in much the same way as an app, and can be composed together to create complex user interfaces.

## Widget Composition

### The Compose Method

The preferred way to add widgets to your app is to implement a `compose()` method which returns an iterable of Widget instances. The method should yield widgets, making it a generator function.

```python
from textual.app import App, ComposeResult
from textual.widgets import Welcome

class WelcomeApp(App):
    def compose(self) -> ComposeResult:
        yield Welcome()
```

### Compound Widgets

A general rule of thumb: if you implement a `compose()` method, there is no need for a `render()` method because it is the widgets yielded from `compose()` which define how the custom widget will look.

However, you can mix these two methods. If you implement both:
- The `render()` method sets the custom widget's background
- The `compose()` method adds widgets on top of that background

This enables creating custom widgets with animated backgrounds while composing child widgets:

```python
from time import time
from textual.app import App, ComposeResult, RenderResult
from textual.containers import Container
from textual.renderables.gradient import LinearGradient
from textual.widgets import Static

class Splash(Container):
    DEFAULT_CSS = """
    Splash {
        align: center middle;
    }
    Static {
        width: 40;
        padding: 2 4;
    }
    """

    def on_mount(self) -> None:
        self.auto_refresh = 1 / 30  # Refresh 30 times per second

    def compose(self) -> ComposeResult:
        yield Static("Making a splash with Textual!")

    def render(self) -> RenderResult:
        return LinearGradient(time() * 90, STOPS)
```

## Widget Lifecycle

### Mounting

While composing is the preferred way of adding widgets when your app starts, it is sometimes necessary to add new widgets in response to events. You can do this by calling `mount()`, which will add a new widget to the UI.

```python
class WelcomeApp(App):
    def on_key(self) -> None:
        self.mount(Welcome())
```

### Awaiting Mount

When you mount a widget, Textual will mount everything the widget composes. Textual guarantees that the mounting will be complete by the next message handler, but not immediately after the call to `mount()`.

To ensure a widget is fully mounted before you make changes to it, you can optionally await the result of `mount()`. This requires the handler to be `async`:

```python
class WelcomeApp(App):
    async def on_key(self) -> None:
        await self.mount(Welcome())
        # Widget is now fully mounted and can be modified
        self.query_one(Button).label = "YES!"
```

### Mount Event

The mount event is sent to a widget after it is added to the app. You can respond to the mount event by defining an `on_mount()` handler:

```python
def on_mount(self) -> None:
    # Widget has been mounted, initialize here
    self.screen.styles.background = "darkblue"
```

## Focus Management

Focus determines which widget receives keyboard input. Textual moves focus around focusable widgets using:
- **Tab** - Move focus forward through focusable widgets
- **Shift + Tab** - Move focus backward through focusable widgets

Widgets can be made focusable by:
1. Setting `can_focus = True` in the widget class
2. Implementing input handlers
3. Being interactive widgets (Button, Input, etc.)

## Overflow Handling

### Display and Visibility

Widgets can be hidden or shown using CSS properties:

- **visibility** - Hide/show widget (preserves layout space)
- **display** - Remove/add widget from layout entirely

```python
# Hide a widget (space preserved)
widget.styles.visibility = "hidden"

# Remove widget from layout
widget.styles.display = "none"

# Show a widget
widget.styles.visibility = "visible"
widget.styles.display = "block"
```

### Scrollable Content

Containers can handle overflow with scrolling:

```python
from textual.containers import ScrollableContainer

class MyApp(App):
    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Static("Long content here...")
```

## Widget Communication

### Events and Messages

Widgets communicate through events and messages:

```python
class MyWidget(Static):
    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Handle button press from child widget
        self.post_message(self.Changed(event.button.id))

    class Changed(Message):
        """Posted when widget state changes."""
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()
```

### DOM Queries

You can query widgets in the DOM tree:

```python
# Get a single widget
button = self.query_one(Button)

# Get multiple widgets
buttons = self.query(Button)

# Filter by ID
widget = self.query_one("#my-widget")

# Filter by class
labels = self.query(".important-label")
```

## Container Patterns

### Horizontal and Vertical Layouts

```python
from textual.containers import Horizontal, Vertical
from textual.widgets import Button

class MyApp(App):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Left")
            yield Button("Right")

        with Vertical():
            yield Button("Top")
            yield Button("Bottom")
```

### Grid Layout

```python
from textual.containers import Grid

class MyApp(App):
    CSS = """
    Grid {
        grid-size: 2 3;  /* 2 columns, 3 rows */
        grid-gutter: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Grid():
            yield Button("1")
            yield Button("2")
            yield Button("3")
            yield Button("4")
            yield Button("5")
            yield Button("6")
```

## Default CSS

Widgets can define default CSS that applies when the widget is used:

```python
class MyWidget(Static):
    DEFAULT_CSS = """
    MyWidget {
        width: 100%;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    """
```

## Common Widget Patterns

### Responsive Design

Create responsive layouts that adapt to container size:

```python
class MyApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: auto;
    }

    @media (max-width: 80) {
        /* Adjust layout for narrow screens */
        Screen {
            grid-size: 1;
        }
    }
    """
```

### Dynamic Widget Mounting

Add and remove widgets in response to state changes:

```python
class DynamicApp(App):
    def on_button_pressed(self) -> None:
        # Mount new widget
        self.mount(Static("New widget added!"))

    def remove_old_widget(self) -> None:
        # Remove specific widget
        widget = self.query_one("#old-widget")
        await widget.remove()
```

## Troubleshooting Common Issues

### "NoMatches" Exception When Mounting

**Problem**: Getting `NoMatches` exception when trying to access a mounted widget immediately.

**Solution**: Use `async` and `await` the `mount()` call:

```python
# Wrong - widget not fully mounted yet
self.mount(MyWidget())
self.query_one(MyWidget).label = "New"  # NoMatches error!

# Correct - wait for mount to complete
async def handler(self) -> None:
    await self.mount(MyWidget())
    self.query_one(MyWidget).label = "New"  # Works!
```

### Widget Not Responding to Input

**Problem**: Widget events not being handled.

**Solution**: Ensure the widget is focusable and check event handler names follow `on_<event>` pattern.

### Layout Issues

**Problem**: Widgets not appearing where expected.

**Solution**:
1. Check CSS layout properties (width, height, display, visibility)
2. Use Textual DevTools to inspect layout: `textual run --dev myapp.py`
3. Verify container is using correct layout mode

## Related Topics

- **Render and Compose**: When to use `render()` vs `compose()` methods
- **Layout Guide**: Detailed layout system documentation (vertical, horizontal, grid, dock)
- **Container Guide**: Working with container widgets for organizing layouts
- **Events Guide**: Complete event system documentation

## Sources

**Official Documentation**:
- [Widgets Guide](https://textual.textualize.io/guide/widgets/) - Textual Official
- [App Basics - Widgets Section](https://textual.textualize.io/guide/app/) - Composing and Mounting
- [Render and Compose How-To](https://textual.textualize.io/how-to/render-and-compose/) - Combining methods

**API References**:
- [Widget API](https://textual.textualize.io/api/widget/)
- [ComposeResult](https://textual.textualize.io/api/app/#textual.app.ComposeResult)
- [Containers](https://textual.textualize.io/api/containers/)

**Access Date**: November 2, 2025
