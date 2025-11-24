# Official Events Guide - Textual Event System & Message Handling

## Overview

Textual has a comprehensive event system for responding to user input, widget lifecycle changes, and internal state changes. Events are messages sent to your app or widgets in response to key presses, mouse actions, and other interactions. Understanding the event system is fundamental to building interactive TUI applications.

From [Events and Messages - Textual](https://textual.textualize.io/guide/events/) (accessed 2025-11-02):

> Events are simply messages reserved for use by Textual for signaling internal state changes, user input, and other occurrences. You can create custom messages for your application that may be used in the same way as events.

## Event Handlers

### Handler Naming Convention

Event handlers are methods prefixed with `on_` followed by the name of the event. The framework automatically calls the appropriate handler when an event occurs.

From [App Basics - Textual](https://textual.textualize.io/guide/app/) (accessed 2025-11-02):

```python
from textual import events
from textual.app import App

class EventApp(App):
    def on_mount(self) -> None:
        """Called after app enters application mode"""
        self.screen.styles.background = "darkblue"

    def on_key(self, event: events.Key) -> None:
        """Called when user presses a key"""
        print(f"Key pressed: {event.key}")
```

### Handler Method Naming Rules

Event handler method names follow a specific convention:
- Prefix: `on_`
- Suffix: The event name in snake_case
- Example: `on_button_pressed`, `on_focus`, `on_key`

## Keyboard Input

### Key Events

The most fundamental way to receive input is via `Key` events sent when the user presses a key.

From [Input - Textual](https://textual.textualize.io/guide/input/) (accessed 2025-11-02):

```python
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import RichLog

class InputApp(App):
    """App to display key events."""

    def compose(self) -> ComposeResult:
        yield RichLog()

    def on_key(self, event: events.Key) -> None:
        self.query_one(RichLog).write(event)
```

### Key Event Attributes

The `Key` event contains several useful attributes:

#### `key`
A string identifying the key pressed:
- Single character for letters and numbers: `"a"`, `"1"`, etc.
- Longer identifiers for special keys: `"home"`, `"f1"`, `"escape"`, etc.
- Modifiers prefix the key: `"shift+home"`, `"ctrl+p"`, `"alt+f4"`, etc.

```python
def on_key(self, event: events.Key) -> None:
    if event.key == "ctrl+c":
        self.exit()
```

#### `character`
Contains a printable Unicode character if one exists:
- `character="p"` for the P key
- `character=None` for function keys like F2
- Useful for input widgets where you care about text content

#### `name`
A Python-valid identifier derived from `key`:
- Lowercase version of the key
- Plus signs replaced with underscores
- Uppercase letters prefixed with `upper_`
- Examples: `"ctrl_p"`, `"upper_p"`, `"f1"`, `"home"`

#### `is_printable`
Boolean indicating if the key produces printable output:
- `True` for regular characters and modified letters
- `False` for control keys and function keys
- Useful for filtering input in text widgets

#### `aliases`
List of possible keys that may have produced this event:
- Example: `["tab", "ctrl+i"]` - Tab and Ctrl+I are indistinguishable in terminals
- Allows handling multiple key combos that produce the same event

### Key Methods

For quick experimentation, methods prefixed with `key_` are called directly:

```python
class InputApp(App):
    def key_space(self) -> None:
        """Called when space key is pressed"""
        self.bell()

    def key_ctrl_s(self) -> None:
        """Called when Ctrl+S is pressed"""
        self.save_file()
```

Note: Key methods are convenient for exploration but key bindings and actions are preferred for production code.

## Input Focus

### Focus Behavior

Only a single widget receives keyboard events at a time. The widget actively receiving events has input **focus**.

From [Input - Textual](https://textual.textualize.io/guide/input/) (accessed 2025-11-02):

```python
class InputApp(App):
    def compose(self) -> ComposeResult:
        yield RichLog()  # Will receive key events when focused
        yield RichLog()
        yield RichLog()
```

### Focus Events

Widgets receive focus events when gaining/losing keyboard focus:

- `Focus` event: Sent when widget receives focus
- `Blur` event: Sent when widget loses focus
- `:focus` CSS pseudo-selector: Style focused widgets

```python
class MyWidget(Widget):
    def on_focus(self) -> None:
        """Widget received input focus"""
        self.styles.border = ("thick", "cyan")

    def on_blur(self) -> None:
        """Widget lost input focus"""
        self.styles.border = ("none",)
```

### Controlling Focus

Widgets have a `can_focus` attribute controlling focusability:
- `can_focus=True`: Widget can receive focus
- `can_focus=False`: Widget cannot receive focus
- Override or set in widget classes

Call `widget.focus()` to programmatically move focus:

```python
def on_mount(self) -> None:
    input_widget = self.query_one(Input)
    input_widget.focus()  # Focus the input widget at startup
```

**Default Focus**: Textual focuses the first focusable widget when the app starts.

**Tab Navigation**: Press Tab to focus the next widget, Shift+Tab for previous.

## Key Bindings

### Creating Bindings

Associate keys with actions using the `BINDINGS` class variable:

From [Input - Textual](https://textual.textualize.io/guide/input/) (accessed 2025-11-02):

```python
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static
from textual.color import Color

class BindingApp(App):
    BINDINGS = [
        ("r", "add_bar('red')", "Add Red"),
        ("g", "add_bar('green')", "Add Green"),
        ("b", "add_bar('blue')", "Add Blue"),
    ]

    def compose(self) -> ComposeResult:
        yield Footer()  # Displays bindings

    def action_add_bar(self, color: str) -> None:
        bar = Static(color)
        bar.styles.background = Color.parse(color).with_alpha(0.5)
        self.mount(bar)
```

### Binding Format

Each binding is a tuple of three strings:
- **Key spec**: The key(s) to bind (e.g., `"r"`, `"ctrl+s"`, `"f1"`)
- **Action name**: The action method to call (without `action_` prefix)
- **Description**: Human-readable description for display

### Multiple Keys

Comma-separate keys to bind multiple combinations to one action:

```python
BINDINGS = [
    ("r,t", "add_bar('red')", "Add Red"),  # Both R and T trigger add_bar
]
```

### Priority Bindings

Mark bindings as priority to check them before widget bindings:

```python
from textual.binding import Binding

BINDINGS = [
    Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
    Binding("ctrl+s", "save", "Save", show=True, priority=False),
]
```

Priority bindings:
- Checked before focused widget bindings
- Cannot be disabled by widget bindings
- Useful for app-level hotkeys

### Show Bindings

Control whether bindings appear in the Footer widget:

```python
BINDINGS = [
    Binding("tab", "focus_next", "Focus Next", show=False),  # Hidden
    ("r", "add_bar", "Add Red"),  # Shown (default)
]
```

## Mouse Input

### Mouse Events

Textual sends events for mouse movement and clicks.

From [Input - Textual](https://textual.textualize.io/guide/input/) (accessed 2025-11-02):

**Available mouse events:**
- `MouseMove`: Mouse cursor moved
- `MouseDown`: Mouse button pressed
- `MouseUp`: Mouse button released
- `Click`: Mouse button clicked (preferred)
- `MouseScrollUp`: Scroll wheel up
- `MouseScrollDown`: Scroll wheel down
- `MouseScrollLeft`: Horizontal scroll left (supported terminals)
- `MouseScrollRight`: Horizontal scroll right (supported terminals)
- `Enter`: Mouse cursor entered widget
- `Leave`: Mouse cursor left widget
- `MouseCapture`: Mouse was captured
- `MouseRelease`: Mouse was released

### Coordinate Systems

Mouse events contain coordinates in two systems:
- **Screen coordinates**: `(0, 0)` is top-left of terminal
- **Widget coordinates**: `(0, 0)` is top-left of widget
- Both accessible via `event.screen_offset` and `event.offset`

### Mouse Movements

Track mouse movement over widgets:

```python
from textual import events

class MouseApp(App):
    def on_mouse_move(self, event: events.MouseMove) -> None:
        widget = self.query_one(Ball)
        # Move widget to follow mouse cursor
        widget.offset = event.screen_offset - (8, 2)
```

### Mouse Capture

Capture all mouse events to a widget regardless of cursor position:

```python
class Ball(Widget):
    def on_mount(self) -> None:
        self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        # This widget receives ALL mouse moves
        self.styles.offset = event.screen_offset

    def release_mouse(self) -> None:
        self.release_mouse()
```

Warning: Captured mouse can produce negative coordinates if cursor moves outside widget bounds.

Mouse capture events:
- `MouseCapture`: Sent when mouse is captured
- `MouseRelease`: Sent when mouse is released

### Click Events

Handle mouse button clicks:

```python
from textual import events

class ClickableWidget(Widget):
    def on_click(self, event: events.Click) -> None:
        """User clicked on this widget"""
        print(f"Clicked at {event.x}, {event.y}")
```

**Recommended**: Use `Click` event, not `MouseDown`/`MouseUp`, for better pointer device compatibility.

### Scroll Events

Handle scrollwheel input:

```python
from textual import events

class CustomScroller(Widget):
    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.scroll_down()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.scroll_up()
```

Scrollable containers handle these automatically. Define handlers for custom scrolling behavior.

## Custom Messages

### Creating Custom Messages

Create custom messages for widget-to-widget communication:

From [Events and Messages - Textual](https://textual.textualize.io/guide/events/) (accessed 2025-11-02):

```python
from textual.message import Message

class MyWidget(Widget):
    class Updated(Message):
        """Posted when widget content is updated."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    def update_content(self, content: str) -> None:
        self.post_message(self.Updated(content))
```

### Handling Custom Messages

Create a handler with the pattern `on_<widget>_<message>`:

```python
class MyApp(App):
    def on_my_widget_updated(self, message: MyWidget.Updated) -> None:
        """Handle the Updated message from MyWidget"""
        print(f"Content updated: {message.content}")
```

### Message Benefits

Custom messages enable:
- Decoupled widget communication
- Parent notification of child events
- Integration with worker threads (long-running tasks)
- Clean separation of concerns

## Async Event Handlers

Textual is built on Python's `asyncio` framework, supporting async handlers:

```python
from textual.app import App

class AsyncApp(App):
    async def on_mount(self) -> None:
        """Async event handler"""
        data = await self.load_data()
        self.display_data(data)

    async def load_data(self) -> list:
        # Async operations don't block UI
        import asyncio
        await asyncio.sleep(2)
        return ["item1", "item2"]
```

**Key benefits of async handlers:**
- Non-blocking: UI remains responsive
- Concurrent: Multiple async handlers run simultaneously
- Clean: Use `await` for async libraries (httpx, aiofiles, etc.)

**Important**: When you make an async handler, Textual will await it and allow other events to process concurrently.

## Common Event Types

### Lifecycle Events

- `Mount`: Widget mounted to DOM
- `Unmount`: Widget removed from DOM
- `Show`: Widget made visible
- `Hide`: Widget made invisible

### Focus Events

- `Focus`: Widget received focus
- `Blur`: Widget lost focus
- `DescendantFocus`: Child widget received focus
- `DescendantBlur`: Child widget lost focus
- `AppFocus`: App received focus (from OS)
- `AppBlur`: App lost focus (to OS)

### Widget Events

- `Button.Pressed`: Button was pressed
- `Input.Changed`: Input widget text changed
- `Input.Submitted`: User pressed Enter in input
- Various widget-specific messages

### System Events

- `Load`: Widget tree loaded
- `Resize`: Terminal/widget resized
- `Paste`: Content pasted into app
- `Print`: Output captured from stdout/stderr
- `ScreenSuspend`: Screen suspended
- `ScreenResume`: Screen resumed

## Event Bubbling

From [API Events Reference](https://textual.textualize.io/api/events/) (accessed 2025-11-02):

Some events "bubble" through the DOM:
- Start at focused/targeted widget
- Rise up through parent widgets if not prevented
- Reach app level if not stopped

**Bubbling events** include:
- `Enter`
- `Leave`
- `DescendantFocus`
- `DescendantBlur`

Check `node` attribute to identify which widget triggered the event:

```python
def on_enter(self, event: events.Enter) -> None:
    if event.node == self:
        # Mouse entered this widget
        pass
    else:
        # Mouse entered a child widget
        pass
```

## Event Handler Concurrency

### Concurrent Events

If event handlers are coroutines (async), Textual allows concurrent event processing:

From [App Basics - Textual](https://textual.textualize.io/guide/app/) (accessed 2025-11-02):

```python
class MyApp(App):
    async def on_key(self, event: events.Key) -> None:
        # This handler is async (coroutine)
        await self.process_key(event)

    async def process_key(self, event: events.Key) -> None:
        # Long-running operation doesn't block UI
        import asyncio
        await asyncio.sleep(5)
        self.display_result()
```

### Single Widget Processing

Individual widgets still process events sequentially:
- A single widget won't receive multiple concurrent event handlers
- Prevents race conditions within a widget
- Multiple widgets can process their events concurrently

## Best Practices

### Do's

- Use key bindings and actions instead of key methods in production
- Create custom messages for widget communication
- Use async handlers for non-blocking operations
- Check `event.node` in bubbling events to identify source
- Prefer `Click` events over `MouseDown`/`MouseUp`

### Don'ts

- Don't use key methods for production code (experimental only)
- Don't block event handlers with long operations (use workers instead)
- Don't capture mouse unless absolutely necessary
- Don't assume key combinations are portable (some keys intercepted by OS/terminal)
- Don't ignore the `:focus` pseudo-selector for focus styling

## Common Pitfalls

### Modifier Keys in Terminals

Not all key combinations work in all terminals:
- Some intercepted by OS
- Some not supported by terminal emulator
- Test with `textual keys` CLI command

### Terminal Key Compatibility

From [Input - Textual](https://textual.textualize.io/guide/input/) (accessed 2025-11-02):

```python
# Run this to test which keys your terminal supports
# Command: textual keys
```

### Mount Timing

Widgets aren't immediately available after mounting:

```python
# WRONG: Widget not yet available
def on_key(self) -> None:
    self.mount(MyWidget())
    button = self.query_one(Button)  # Fails! Not mounted yet

# RIGHT: Await mount to complete
async def on_key(self) -> None:
    await self.mount(MyWidget())
    button = self.query_one(Button)  # Works! Mount complete
```

## Event System Architecture

### Message Queue

Textual maintains a message queue:
1. User input or internal event occurs
2. Event added to queue
3. Widget processes events from queue
4. Handlers called, potentially posting new messages
5. Loop continues until queue empty

### Event Propagation

When a key is pressed:
1. Key event created
2. Sent to focused widget
3. Handler (if exists) called
4. Event bubbles up DOM if applicable
5. App receives event if not handled

### Flow Control

Handlers can influence event flow:
- Return value `None` (default): Event continues normally
- Explicit handling: Return from handler to stop processing

## References

**From Official Textual Documentation:**
- [Events and Messages Guide](https://textual.textualize.io/guide/events/) - Complete event system overview
- [Input Guide](https://textual.textualize.io/guide/input/) - Keyboard and mouse input handling
- [App Basics Guide](https://textual.textualize.io/guide/app/) - Event handlers and app structure
- [API Events Reference](https://textual.textualize.io/api/events/) - All event types and properties
- [API on Decorator](https://textual.textualize.io/api/on/) - Modern @on decorator for handlers
- [API Binding](https://textual.textualize.io/api/binding/) - Key binding configuration

**Web Resources:**
- [Textual 0.23.0 Message Handling Blog Post](https://textual.textualize.io/blog/2023/05/03/textual-0230-improves-message-handling/) - Introduction to @on decorator
- [Anatomy of a Textual UI Blog Post](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/) - Event system in context

## Sources

**Official Documentation:**
- [Events and Messages - Textual Guide](https://textual.textualize.io/guide/events/)
- [Input - Textual Guide](https://textual.textualize.io/guide/input/)
- [App Basics - Textual Guide](https://textual.textualize.io/guide/app/)
- [API Events Reference](https://textual.textualize.io/api/events/)

**Access Date:** November 2, 2025
