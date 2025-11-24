# Advanced Input Collection - Textual-Inputs Library

## Overview

**Textual-Inputs** is a third-party widget library that provided enhanced input widgets for early Textual applications. Created by @sirfuzzalot, it offered `TextInput` and `IntegerInput` widgets with advanced features like password masking, syntax highlighting, and custom event handlers.

**Important Historical Note**: This library is **no longer maintained** (marked as "Abandoned" in 2023) because Textual now provides native support for these input patterns. However, it remains valuable for understanding:
- How custom input widgets are built
- Text editing patterns in terminal UIs
- Cursor management and text overflow handling
- Event-driven form validation
- Early Textual widget architecture patterns

From [textual-inputs README](https://github.com/sirfuzzalot/textual-inputs/blob/main/README.md):
> "Thanks to all the developers who contributed or used Textual Inputs. These widgets filled a gap in the early stages of Textual, which is now supported natively."

## Installation

```bash
python -m pip install textual-inputs~=0.2.6
```

**Note**: For new projects, use native Textual `Input` widget instead (see [official docs](https://textual.textualize.io/widgets/input/)).

## Widget Types

### 1. TextInput Widget

Single-line text input with overflow support, password masking, and syntax highlighting.

**Basic Usage:**
```python
from textual_inputs import TextInput

# Simple text input
username = TextInput(
    name="username",
    placeholder="enter your username...",
    title="Username"
)

# Password input (masked with bullets)
password = TextInput(
    name="password",
    title="Password",
    password=True
)

# Code input with syntax highlighting
code = TextInput(
    name="code",
    placeholder="enter some python code...",
    title="Code",
    syntax="python"  # Uses Pygments for highlighting
)
```

**Key Features:**
- **Value**: String content of the input
- **Placeholder**: Text shown when empty and unfocused
- **Title**: Border label
- **Password mode**: Replaces characters with bullets (•)
- **Syntax highlighting**: One-line code highlighting via Rich/Pygments
- **Unicode support**: Full Unicode character handling
- **Text overflow**: Horizontal scrolling for long text

**Keyboard Controls:**
- `←` / `→`: Move cursor left/right
- `Home` / `End`: Jump to start/end
- `Backspace` / `Ctrl+H`: Delete previous character
- `Delete`: Delete next character
- `Escape`: Reset focus

From [text_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/text_input.py) lines 38-84:
```python
class TextInput(Widget):
    """A simple text input widget.

    Args:
        name (Optional[str]): The unique name of the widget.
        value (str, optional): Defaults to "". The starting text value.
        placeholder (str, optional): Defaults to "". Text that appears
            in the widget when value is "" and the widget is not focused.
        title (str, optional): Defaults to "". A title on the top left
            of the widget's border.
        password (bool, optional): Defaults to False. Hides the text
            input, replacing it with bullets.
        syntax (Optional[str]): The name of the language for syntax highlighting.
    """
```

### 2. IntegerInput Widget

Specialized input for integer values with increment/decrement controls.

**Basic Usage:**
```python
from textual_inputs import IntegerInput

# Age input
age = IntegerInput(
    name="age",
    placeholder="enter your age...",
    title="Age"
)

# With initial value and custom step
quantity = IntegerInput(
    name="quantity",
    value=10,
    title="Quantity",
    step=5  # Increment/decrement by 5
)
```

**Key Features:**
- **Value**: Integer or `None`
- **Arrow key increment**: Press `↑`/`↓` to change value
- **Step support**: Custom increment/decrement amount
- **Type validation**: Only accepts numeric input
- **Negative numbers**: Type `-` at start for negatives

**Keyboard Controls:**
- `←` / `→`: Move cursor left/right
- `↑` / `↓`: Increment/decrement by step
- `Home` / `End`: Jump to start/end
- `0-9`: Type digits
- `-`: Type at position 0 for negative
- `Backspace` / `Delete`: Remove digits

From [integer_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/integer_input.py) lines 23-64:
```python
class IntegerInput(Widget):
    """A simple integer input widget.

    Args:
        name (Optional[str]): The unique name of the widget.
        value (Optional[int]): The starting integer value.
        placeholder (Union[str, int], optional): Text that appears
            in the widget when value is "" and not focused.
        title (str, optional): A title on the top left of the border.
        step (int): Increment/decrement step size.

    Events:
        InputOnChange: Emitted when the contents of the input changes.
        InputOnFocus: Emitted when the widget becomes focused.
    """
```

## Advanced Features

### One-Line Syntax Highlighting

Textual-Inputs leverages Rich's `Syntax` class to provide syntax highlighting for code input.

**⚠️ Limitation**: Only supports single-line code (no multi-line editing).

```python
TextInput(
    name="code",
    placeholder="enter some python code...",
    title="Code",
    syntax="python"  # Any language supported by Pygments
)
```

**How it works** (from [text_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/text_input.py) lines 31-36):
```python
def syntax_highlight_text(code: str, syntax: str) -> Text:
    """Produces highlighted text based on the syntax."""
    syntax_obj = Syntax(code, syntax)
    with CONSOLE.capture() as capture:
        CONSOLE.print(syntax_obj)
    return Text.from_ansi(capture.get())
```

The widget renders syntax using Pygments, captures the ANSI output, and converts it to Rich `Text` for display.

### Custom Event Handlers

Textual-Inputs provides a pattern for custom event handler naming:

```python
email = TextInput(name="email", title="Email")
email.on_change_handler_name = "handle_email_on_change"
email.on_focus_handler_name = "handle_email_on_focus"
```

**Event Types:**
- `InputOnChange`: Emitted when value changes (typing, deletion)
- `InputOnFocus`: Emitted when widget gains focus

**Handler Implementation:**
```python
async def handle_email_on_change(self, message: Message) -> None:
    self.log(f"Email Field Contains: {message.sender.value}")

async def handle_email_on_focus(self, message: Message) -> None:
    self.current_index = self.tab_index.index(message.sender.name)
```

**Default handlers** (if not customized):
- `handle_input_on_change`
- `handle_input_on_focus`

From [simple_form.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/examples/simple_form.py) lines 95-111:
```python
self.username = TextInput(
    name="username",
    placeholder="enter your username...",
    title="Username",
)
self.username.on_change_handler_name = "handle_username_on_change"

self.age = IntegerInput(
    name="age",
    placeholder="enter your age...",
    title="Age",
)
self.age.on_change_handler_name = "handle_age_on_change"
```

## Complete Form Example

From [simple_form.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/examples/simple_form.py):

```python
from textual.app import App
from textual.widgets import Footer, Header, Static
from textual_inputs import IntegerInput, TextInput

class SimpleForm(App):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tab_index = ["username", "password", "age", "code"]

    async def on_mount(self) -> None:
        # Create input widgets
        self.username = TextInput(
            name="username",
            placeholder="enter your username...",
            title="Username",
        )
        self.username.on_change_handler_name = "handle_username_on_change"

        self.password = TextInput(
            name="password",
            title="Password",
            password=True,  # Mask with bullets
        )

        self.age = IntegerInput(
            name="age",
            placeholder="enter your age...",
            title="Age",
        )
        self.age.on_change_handler_name = "handle_age_on_change"

        self.code = TextInput(
            name="code",
            placeholder="enter some python code...",
            title="Code",
            syntax="python",  # Syntax highlighting
        )

        # Dock widgets
        await self.view.dock(
            self.username, self.password, self.age, self.code, edge="top"
        )

    async def action_submit(self) -> None:
        """Handle form submission"""
        formatted = f"""
username: {self.username.value}
password: {"".join("•" for _ in self.password.value)}
     age: {self.age.value}
    code: {self.code.value}
        """
        # Display results...
```

## Implementation Architecture

### Text Overflow and Cursor Management

Textual-Inputs implements horizontal scrolling for text that exceeds widget width.

**Key concept**: Text offset window (from [text_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/text_input.py) lines 222-233):

```python
@property
def _visible_width(self):
    """Width in characters of the inside of the input"""
    # remove 2 for border edges
    # remove 1 for cursor
    # remove 2 for padding
    width, _ = self.size
    if self.border:
        width -= 2
    if self._has_focus:
        width -= 1
    width -= 2
    return width

def _text_offset_window(self):
    """Produce start and end indices of visible text portions."""
    return self._text_offset, self._text_offset + self._visible_width
```

**Offset updating** (lines 275-290):
```python
def _update_offset_left(self):
    """Decrease text offset if cursor moves near left edge."""
    visibility_left = 3
    if self._cursor_position < self._text_offset + visibility_left:
        self._text_offset = max(0, self._cursor_position - visibility_left)

def _update_offset_right(self):
    """Increase text offset if cursor moves beyond right edge."""
    _, right = self._text_offset_window()
    if self._cursor_position > right:
        self._text_offset = self._cursor_position - self._visible_width
```

This creates a "viewport" that scrolls left/right as the cursor approaches edges, keeping the cursor visible with 3-character padding.

### Password Concealment

Simple character replacement pattern (from [text_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/text_input.py) lines 26-28):

```python
def conceal_text(segment: str) -> str:
    """Produce the segment concealed like a password."""
    return "•" * len(segment)
```

Applied during rendering:
```python
def _modify_text(self, segment: str) -> Union[str, Text]:
    """Produces text with modifications (password concealing, syntax highlighting)."""
    if self.has_password:
        return conceal_text(segment)
    if self.syntax:
        return syntax_highlight_text(segment, self.syntax)
    return segment
```

### Rendering with Cursor

From [text_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/text_input.py) lines 235-251:

```python
def _render_text_with_cursor(self) -> List[Union[str, Text, Tuple[str, Style]]]:
    """Produces renderable Text combining value and cursor"""
    text = self._modify_text(self.value)

    # Trim string to fit within widget dimensions
    left, right = self._text_offset_window()
    text = text[left:right]

    # Convert cursor position to be relative to this view
    cursor_relative_position = self._cursor_position - self._text_offset
    return [
        text[:cursor_relative_position],
        self.cursor,  # ("|", Style(color="white", blink=True, bold=True))
        text[cursor_relative_position:],
    ]
```

The cursor is a blinking `|` character inserted at the current position.

### Integer Input Validation

From [integer_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/integer_input.py) lines 305-320:

```python
async def on_key(self, event: events.Key) -> None:
    # Arrow up/down increments/decrements
    elif event.key == "up":
        self._increment_value(self.step)
        self._cursor_position = len(self._value_as_str)
        await self._emit_on_change(event)

    elif event.key == "down":
        self._increment_value(-self.step)
        self._cursor_position = len(self._value_as_str)
        await self._emit_on_change(event)

    # Only allow digits
    elif event.key in string.digits:
        # Insert digit at cursor position...

    # Allow negative sign at start only
    elif event.key == "-" and self._cursor_position == 0:
        if value[0] != "-" and value != "0":
            self.value = int(event.key + value)
```

Type safety is enforced by restricting key presses to digits and `-` (at position 0 only).

## Comparison with Native Textual Input

### Textual-Inputs (Historical)

**Strengths:**
- Custom event handler naming
- Built-in syntax highlighting
- Specialized IntegerInput widget
- Password masking

**Limitations:**
- No longer maintained
- Single-line only
- Early Textual architecture (pre-0.10)
- Manual cursor/offset management

### Native Textual Input (Current)

Modern Textual provides `Input` widget with similar capabilities:

```python
from textual.widgets import Input

# Basic input
username = Input(placeholder="Enter username...")

# Password input
password = Input(placeholder="Password", password=True)

# Input with validation
class IntegerInput(Input):
    def validate_value(self, value: str) -> str:
        """Only allow integers."""
        return "".join(c for c in value if c.isdigit())
```

**Native advantages:**
- Active maintenance and updates
- Better performance
- Integrated with modern Textual architecture
- Built-in validators and suggester support
- Multi-line TextArea widget available

See [Textual Input Documentation](https://textual.textualize.io/widgets/input/) for current best practices.

## Key Patterns for Custom Input Widgets

### 1. Cursor Position Management

```python
cursor: Tuple[str, Style] = (
    "|",
    Style(color="white", blink=True, bold=True),
)
_cursor_position: Reactive[int] = Reactive(0)

def _cursor_left(self):
    if self._cursor_position > 0:
        self._cursor_position -= 1
        self._update_offset_left()

def _cursor_right(self):
    if self._cursor_position < len(self.value):
        self._cursor_position += 1
        self._update_offset_right()
```

### 2. Text Editing Operations

```python
def _key_backspace(self):
    """Handle key press Backspace"""
    if self._cursor_position > 0:
        self.value = (
            self.value[:self._cursor_position - 1]
            + self.value[self._cursor_position:]
        )
        self._cursor_position -= 1

def _key_delete(self):
    """Handle key press Delete"""
    if self._cursor_position < len(self.value):
        self.value = (
            self.value[:self._cursor_position]
            + self.value[self._cursor_position + 1:]
        )

def _key_printable(self, event: events.Key):
    """Handle all printable keys"""
    self.value = (
        self.value[:self._cursor_position]
        + event.key
        + self.value[self._cursor_position:]
    )
    self._cursor_position += 1
```

### 3. Custom Event Emission

```python
from textual.message import Message

class InputOnChange(Message):
    """Emitted when input value changes."""
    pass

class InputOnFocus(Message):
    """Emitted when input gains focus."""
    pass

async def on_key(self, event: events.Key) -> None:
    if event.key in printable_keys:
        self._key_printable(event)
        await self._emit_on_change(event)

async def _emit_on_change(self, event: events.Key) -> None:
    event.stop()
    await self.emit(InputOnChange(self))
```

## Migration Guide

### From Textual-Inputs to Native Input

**Before (Textual-Inputs):**
```python
from textual_inputs import TextInput, IntegerInput

username = TextInput(name="username", title="Username")
age = IntegerInput(name="age", title="Age")
```

**After (Native Textual):**
```python
from textual.widgets import Input

username = Input(placeholder="Username", id="username")
age = Input(
    placeholder="Age",
    id="age",
    validators=[Integer()],
    type="integer"
)
```

**Event handling migration:**
```python
# Before: Custom handler names
username.on_change_handler_name = "handle_username_on_change"

# After: Use Input.Changed message
def on_input_changed(self, event: Input.Changed) -> None:
    if event.input.id == "username":
        self.log(f"Username: {event.value}")
```

## Sources

**GitHub Repository:**
- [sirfuzzalot/textual-inputs](https://github.com/sirfuzzalot/textual-inputs) - Main repository (accessed 2025-11-02)

**Source Files:**
- [README.md](https://github.com/sirfuzzalot/textual-inputs/blob/main/README.md) - Documentation and API reference
- [text_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/text_input.py) - TextInput widget implementation
- [integer_input.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/src/textual_inputs/integer_input.py) - IntegerInput widget implementation
- [simple_form.py](https://github.com/sirfuzzalot/textual-inputs/blob/main/examples/simple_form.py) - Complete form example

**Related Documentation:**
- [Textual Input Widget](https://textual.textualize.io/widgets/input/) - Modern native input widget
- [Textual TextArea Widget](https://textual.textualize.io/widgets/textarea/) - Multi-line text editing

**Historical Context:**
This library represents early Textual widget development patterns (2022-2023) before native Input widgets matured. While no longer maintained, it demonstrates important concepts for custom widget authors and provides insight into Textual's evolution.
