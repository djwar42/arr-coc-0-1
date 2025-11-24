# ArjanCodes: Interactive CLI Tools with Textual

**Source**: [ArjanCodes Blog - Textual Python Library for Creating Interactive Terminal Applications](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/)
**Accessed**: 2025-11-02
**Author**: ArjanCodes
**Published**: June 24, 2024

## Overview

ArjanCodes' guide to building interactive terminal applications with Textual. Known for clean code practices, this tutorial demonstrates event handling patterns, widget composition, and multi-screen architecture with practical examples.

**Key Insight**: Textual is built on top of the Rich library, offering fully interactive GUI-like experiences within the terminal with cross-platform support (including web).

## Creating a Simple Application

### Basic Setup Pattern

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

```python
from textual.app import App, ComposeResult
from textual.widgets import Placeholder, Header, Footer

class Application(App):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Placeholder("Hello World")
        yield Footer()

def main() -> None:
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
```

**Design Pattern**: The `compose()` function assembles the layout by yielding components:
- **Header**: Adds a header with command palette and optional clock
- **Placeholder**: Placeholder widget for initial layout composition
- **Footer**: Footer that can list hotkeys for the screen if bound

**Important Note**: Textual applications may not function well on IDE-embedded consoles. Run on actual terminals, as most IDEs don't emulate interactive terminals properly.

## Building Interactive Applications

### Event Handling and User Input

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

```python
from typing import Literal
from textual.app import App, ComposeResult, on
from textual.widgets import Input, Header, Footer, RichLog
from rich.panel import Panel
from rich.align import Align

class Application(App):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="Chat")
        yield RichLog(markup=True)
        yield Input(placeholder="Message", id="input")
        yield Footer()

    async def write_message(
        self,
        message: str,
        title: str,
        side: Literal["left", "right"],
        colour: str
    ):
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

def main() -> None:
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
```

### Interactive Interface Patterns

**RichLog Widget**:
- Provides vertically aligned log for writing messages
- `markup=True` enables Rich's markup features (colors, formatting)

**Input Widget**:
- Creates text box for user input
- Can be queried and processed by other functions

**Event Handling with @on Decorator**:
- Assigns functions as message handlers
- Enables CLI to react to user inputs
- Pattern: `@on(Widget.Event)` → handler function

**Query Pattern**:
- `query_one()` queries objects by type or ID
- Uses CSS-style selectors for IDs (e.g., `"#input"`)

## Textual Widgets Overview

### Button Widget

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

```python
from textual.app import App, ComposeResult, on
from textual.widgets import Button, Header, Footer

class Application(App):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Button(label="Click Me", id="button")
        yield Footer()

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        button = self.query_one("#button", Button)
        button.label = "Clicked!"

def main() -> None:
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
```

**Pattern**: Interactive elements that trigger actions when clicked, with dynamic label updates.

### Checkbox Widget

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

```python
from textual.app import App, ComposeResult, on
from textual.widgets import Checkbox, Header, Footer

class Application(App):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Checkbox(label="Accept Terms and Conditions", id="checkbox")
        yield Footer()

    @on(Checkbox.Changed)
    async def checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox = self.query_one("#checkbox", Checkbox)
        checkbox.label = "Accepted!" if event.value else "Accept Terms and Conditions"

def main() -> None:
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
```

**Pattern**: Binary choices with state change handlers, accessing `event.value` for state.

### ProgressBar Widget

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

```python
from textual import events, on
from textual.app import App, ComposeResult
from textual.widgets import ProgressBar, Header, Footer
from asyncio import sleep

class Application(App):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ProgressBar(total=100, id="progress")
        yield Footer()

    @on(events.Ready)
    async def on_startup(self) -> None:
        progress = self.query_one("#progress", ProgressBar)
        for i in range(101):
            progress.update(progress=i)
            await sleep(0.1)

def main() -> None:
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
```

**Pattern**: Display task progress with async updates, using `@on(events.Ready)` for startup initialization.

## Implementing Multiple Screens

### Multi-Screen Architecture

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

```python
from typing import Literal

from textual.app import App, ComposeResult, on
from textual.widgets import Input, Header, Footer, RichLog
from textual.screen import ModalScreen
from rich.panel import Panel
from rich.align import Align

class ChatScreen(ModalScreen):
    app: "Application"
    BINDINGS = [
        ("ctrl+s", "app.switch_mode('settings')", "Settings"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="Logs")
        yield RichLog(markup=True, id="output")
        yield Input(placeholder="Type a log message", id="input")
        yield Footer()

    async def write_log(
        self,
        message: str,
        title: str,
        side: Literal["left", "right"],
        colour: str
    ):
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
    async def submit_handler(self, event: Input.Submitted) -> None:
        await self.write_log(event.value, "Message", "left", "green")
        self.query_one("#input", Input).clear()
        await self.write_log("A response", "Response", "right", "blue")


class SettingsScreen(ModalScreen):
    app: "Application"
    BINDINGS = [
        ("escape", "app.switch_mode('chat')", "Logs"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(value=self.app.config_setting, id="input")
        yield Footer()

    @on(Input.Submitted)
    async def submit_handler(self, event: Input.Submitted) -> None:
        self.app.config_setting = event.value
        self.app.pop_screen()

class Application(App):
    MODES = {"chat": ChatScreen, "settings": SettingsScreen}

    def __init__(self, config_setting="Default Setting"):
        self.config_setting = config_setting
        super().__init__()

    async def on_mount(self) -> None:
        await self.switch_mode("chat")


def main() -> None:
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
```

### Multi-Screen Design Patterns

**ModalScreen**:
- Used to create custom screens
- Allows definition of separate modes within the application

**MODES Dictionary**:
- Assigns names to various modes
- Switch between modes using `app.switch_mode(mode_name)`

**BINDINGS**:
- Define keyboard shortcuts for each screen
- Format: `(key, action, description)`
- Example: `("ctrl+s", "app.switch_mode('settings')", "Settings")`

**Screen Communication**:
- Screens can access parent app via `self.app`
- Share state through app-level attributes
- Use `app.pop_screen()` to return to previous screen

## Event Handling Patterns

### The @on Decorator

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

**Common Event Patterns**:
- `@on(Input.Submitted)` - Text input submission
- `@on(Button.Pressed)` - Button clicks
- `@on(Checkbox.Changed)` - Checkbox state changes
- `@on(events.Ready)` - App startup/ready state

**Event Handler Signature**:
```python
@on(Widget.Event)
async def handler(self, event: Widget.Event) -> None:
    # Access event data via event object
    value = event.value  # For input/checkbox
    # Process event
```

### Query Patterns

**CSS-Style Selectors**:
- By ID: `query_one("#element_id", WidgetType)`
- By type: `query_one(WidgetType)`
- Returns single widget instance

**Widget Manipulation**:
```python
# Get widget
widget = self.query_one("#my_widget", Input)

# Update properties
widget.clear()
widget.value = "new value"
widget.label = "new label"
```

## Clean Code Insights from ArjanCodes

### Separation of Concerns

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

**Screen-Level Organization**:
- Each screen is a separate class
- Screens handle their own events
- Shared state lives in parent Application

**Helper Methods**:
- Extract complex rendering logic (e.g., `write_message()`, `write_log()`)
- Keep event handlers thin - delegate to helpers
- Type hints for clarity (`Literal["left", "right"]`)

### Type Safety

**Strong Typing Throughout**:
```python
from typing import Literal

async def write_message(
    self,
    message: str,
    title: str,
    side: Literal["left", "right"],  # Restrict to valid values
    colour: str
):
    ...
```

**Type Annotations for App Reference**:
```python
class ChatScreen(ModalScreen):
    app: "Application"  # Forward reference for type checking
```

### Async/Await Patterns

**Consistent Async Usage**:
- Event handlers are async: `async def submit_handler(...)`
- Helper methods are async when they update UI: `async def write_message(...)`
- Use `await` for async operations: `await self.write_log(...)`

## Architecture Takeaways

### Component Composition

**Yield-Based Layout**:
```python
def compose(self) -> ComposeResult:
    yield Header(show_clock=True)
    yield RichLog(markup=True, id="output")
    yield Input(placeholder="Message", id="input")
    yield Footer()
```

**Benefits**:
- Declarative layout definition
- Clear visual hierarchy
- Easy to modify and extend

### State Management

**App-Level State**:
```python
class Application(App):
    def __init__(self, config_setting="Default Setting"):
        self.config_setting = config_setting  # Shared across screens
        super().__init__()
```

**Screen Access to App State**:
```python
# In SettingsScreen
yield Input(value=self.app.config_setting, id="input")

# Update app state
self.app.config_setting = event.value
```

### Rich Integration

**Seamless Rich Usage**:
- RichLog accepts Rich renderables (Panel, Align, etc.)
- Markup support for colors and formatting
- Console width awareness: `self.app.console.width`

**Rich Rendering Pattern**:
```python
from rich.panel import Panel
from rich.align import Align

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
```

## Key Concepts Summary

### Textual Fundamentals

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

1. **App Class**: Base for all Textual applications
2. **compose()**: Method to build widget hierarchy
3. **Widgets**: Pre-built UI components (Header, Footer, Input, Button, etc.)
4. **Events**: User interactions handled via @on decorator
5. **Screens**: Multiple views via ModalScreen and MODES
6. **Query System**: CSS-style selectors for widget access
7. **Async**: All UI updates are async

### Best Practices

**From ArjanCodes' Clean Code Approach**:

1. **Type Everything**: Use type hints for all parameters and return values
2. **Async Consistency**: Make event handlers and UI-updating methods async
3. **Separate Concerns**: One screen per logical view, helper methods for complex logic
4. **Descriptive IDs**: Use meaningful IDs for widget queries (`"#input"`, `"#output"`)
5. **Rich Integration**: Leverage Rich for visual polish (colors, panels, alignment)
6. **State Management**: Share state at app level, access via `self.app`

## Final Thoughts

From [ArjanCodes Blog](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/):

> "Textual offers a wide range of widgets and tools for creating visually stunning and highly responsive terminal user interfaces (TUIs). This guide covers basic concepts and examples, but just scratches the surface of textual's potential. With robust features and the ability to create sophisticated, interactive terminal experiences, Textual is a game-changer for developers building CLI applications."

**Official Documentation**: [textual.textualize.io](https://textual.textualize.io/)

## Sources

**Primary Source**:
- [ArjanCodes Blog - Textual Python Library for Creating Interactive Terminal Applications](https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/) - Published June 24, 2024 (accessed 2025-11-02)

**Related ArjanCodes Content**:
- [Rich Python Library for Interactive CLI Tools](https://arjancodes.com/blog/rich-python-library-for-interactive-cli-tools/) - Foundation for Textual

**Official Resources**:
- [Textual Documentation](https://textual.textualize.io/) - Complete API reference
