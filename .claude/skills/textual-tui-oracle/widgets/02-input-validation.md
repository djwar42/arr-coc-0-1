# Input Widget with Validation - Comprehensive Guide

## Overview

The Input widget provides single-line text input with comprehensive validation capabilities. It supports type restrictions, custom validators, real-time validation feedback, and event handling for user input.

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/) (accessed 2025-11-02)**

## Key Features

### Core Capabilities
- Single-line text input
- Built-in input types (text, integer, number)
- Regex-based character restriction
- Maximum length limits
- Comprehensive validation framework
- Auto-completion suggestions
- Password masking
- Placeholder text
- Real-time validation feedback

### Widget Characteristics
- **Focusable**: Yes
- **Container**: No

## Basic Usage

### Simple Input Example

```python
from textual.app import App, ComposeResult
from textual.widgets import Input

class InputApp(App):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="First Name")
        yield Input(placeholder="Last Name")
```

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#a-simple-example) (accessed 2025-11-02)**

## Input Types

The Input widget supports different types that automatically restrict and validate input:

| Type | Description | Valid Characters |
|------|-------------|------------------|
| `"text"` | Allow all text (default) | All characters |
| `"integer"` | Integers only | 0-9, -, + |
| `"number"` | Floating point numbers | 0-9, -, +, ., e, E |

### Type Examples

```python
# Integer input
Input(placeholder="Age", type="integer")

# Number input
Input(placeholder="Price", type="number")

# Text input (default)
Input(placeholder="Name", type="text")
```

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#input-types) (accessed 2025-11-02)**

## Restricting Input

### Regex Restrictions

Use the `restrict` parameter to limit characters via regular expression:

```python
# Binary input only
Input(restrict=r"[01]*", placeholder="Binary")

# Hexadecimal input
Input(restrict=r"[0-9A-Fa-f]*", placeholder="Hex")

# Email format
Input(restrict=r"[a-zA-Z0-9@._-]*", placeholder="Email")

# Phone number
Input(restrict=r"[0-9\-\(\) ]*", placeholder="Phone")
```

**Note**: The regex is applied to the full value, not just new characters.

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#restricting-input) (accessed 2025-11-02)**

### Maximum Length

```python
# Limit to 10 characters
Input(max_length=10, placeholder="Username")

# Unlimited (default)
Input(max_length=0)
```

## Validation Framework

### Built-in Validators

Textual provides several built-in validators in `textual.validation`:

**From [textual.validation API](https://textual.textualize.io/api/validation/) (accessed 2025-11-02)**

#### Number Validator

```python
from textual.validation import Number

Input(
    placeholder="Age",
    validators=[
        Number(minimum=1, maximum=120)
    ]
)
```

#### Integer Validator

```python
from textual.validation import Integer

Input(
    placeholder="Count",
    validators=[
        Integer(minimum=0, maximum=100)
    ]
)
```

#### Length Validator

```python
from textual.validation import Length

Input(
    placeholder="Password",
    validators=[
        Length(minimum=8, maximum=32)
    ],
    password=True
)
```

#### Regex Validator

```python
from textual.validation import Regex

Input(
    placeholder="Email",
    validators=[
        Regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    ]
)
```

#### URL Validator

```python
from textual.validation import URL

Input(
    placeholder="Website",
    validators=[URL()]
)
```

#### Function Validator

```python
from textual.validation import Function

def is_even(value: str) -> bool:
    try:
        return int(value) % 2 == 0
    except ValueError:
        return False

Input(
    placeholder="Even number",
    validators=[
        Function(is_even, "Value must be even")
    ]
)
```

### Multiple Validators

Combine multiple validators for complex validation:

```python
Input(
    placeholder="Age",
    validators=[
        Number(minimum=1, maximum=100),
        Function(is_even, "Age must be even"),
    ]
)
```

All validators must pass for the input to be considered valid.

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#validating-input) (accessed 2025-11-02)**

### Custom Validators

Create custom validators by extending `Validator`:

```python
from textual.validation import Validator, ValidationResult

class Palindrome(Validator):
    def validate(self, value: str) -> ValidationResult:
        """Check if string is a palindrome."""
        if value == value[::-1]:
            return self.success()
        else:
            return self.failure("Not a palindrome!")

# Use custom validator
Input(
    placeholder="Palindrome text",
    validators=[Palindrome()]
)
```

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#validating-input) (accessed 2025-11-02)**

## Validation Timing

Control when validation occurs using `validate_on`:

```python
# Validate on change (default)
Input(validate_on=["changed"])

# Validate only on submit
Input(validate_on=["submitted"])

# Validate on blur (lost focus)
Input(validate_on=["blur"])

# Validate on multiple events
Input(validate_on=["changed", "submitted", "blur"])
```

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#validating-input) (accessed 2025-11-02)**

## Handling Validation Events

### Changed Event

```python
from textual import on

@on(Input.Changed)
def on_input_changed(self, event: Input.Changed) -> None:
    """Handle input change and check validation."""
    value = event.value
    validation_result = event.validation_result

    if validation_result and not validation_result.is_valid:
        # Show validation errors
        errors = validation_result.failure_descriptions
        self.notify("\n".join(errors), severity="error")
```

### Submitted Event

```python
@on(Input.Submitted)
def on_input_submitted(self, event: Input.Submitted) -> None:
    """Handle input submission (Enter key)."""
    if event.validation_result and event.validation_result.is_valid:
        self.process_input(event.value)
    else:
        self.notify("Please fix validation errors", severity="warning")
```

### Blurred Event

```python
@on(Input.Blurred)
def on_input_blurred(self, event: Input.Blurred) -> None:
    """Handle input losing focus."""
    if event.validation_result and not event.validation_result.is_valid:
        self.show_error_tooltip(event.input)
```

## Validation Result

The `ValidationResult` object provides:

```python
validation_result.is_valid  # bool: True if valid
validation_result.failures  # list[Failure]: List of failures
validation_result.failure_descriptions  # list[str]: Error messages
```

**From [textual.validation API](https://textual.textualize.io/api/validation/#textual.validation.ValidationResult) (accessed 2025-11-02)**

## Reactive Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | str | "" | Current input value |
| `cursor_position` | int | 0 | Cursor position in text |
| `placeholder` | str | "" | Placeholder text |
| `password` | bool | False | Mask input as password |
| `cursor_blink` | bool | True | Enable cursor blinking |
| `restrict` | str | None | Regex restriction pattern |
| `type` | str | "text" | Input type |
| `max_length` | int | None | Maximum input length |
| `valid_empty` | bool | False | Allow empty as valid |

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#reactive-attributes) (accessed 2025-11-02)**

## Messages and Events

| Message | When Fired | Attributes |
|---------|-----------|------------|
| `Input.Changed` | Value changes | `value`, `validation_result` |
| `Input.Submitted` | Enter pressed | `value`, `validation_result` |
| `Input.Blurred` | Lost focus | `value`, `validation_result` |

## Keyboard Bindings

| Key | Action | Description |
|-----|--------|-------------|
| Left Arrow | cursor_left | Move cursor left |
| Right Arrow | cursor_right | Move cursor right |
| Ctrl+Left | cursor_left_word | Jump word left |
| Ctrl+Right | cursor_right_word | Jump word right |
| Home, Ctrl+A | home | Start of input |
| End, Ctrl+E | end | End of input |
| Backspace | delete_left | Delete left character |
| Delete, Ctrl+D | delete_right | Delete right character |
| Ctrl+W | delete_left_word | Delete word left |
| Ctrl+K | delete_right_all | Delete to end |
| Ctrl+U | delete_left_all | Delete to start |
| Enter | submit | Submit input |
| Ctrl+X | cut | Cut selection |
| Ctrl+C | copy | Copy selection |
| Ctrl+V | paste | Paste clipboard |

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#bindings) (accessed 2025-11-02)**

## Styling

### CSS Classes

```python
# In CSS
Input {
    border: tall $primary;
    background: $surface;
}

Input:focus {
    border: tall $accent;
}

Input.-valid {
    border: tall $success;
}

Input.-invalid {
    border: tall $error;
}
```

### Component Classes

| Class | Description |
|-------|-------------|
| `input--cursor` | Target the cursor |
| `input--placeholder` | Target placeholder text |
| `input--suggestion` | Target auto-completion suggestion |
| `input--selection` | Target selected text |

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#component-classes) (accessed 2025-11-02)**

## Practical Examples

### Form Validation

```python
class LoginForm(Container):
    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="Username",
            validators=[
                Length(minimum=3, maximum=20),
                Regex(r"^[a-zA-Z0-9_]+$", failure_description="Alphanumeric only")
            ]
        )
        yield Input(
            placeholder="Password",
            password=True,
            validators=[
                Length(minimum=8),
                Function(has_uppercase, "Must contain uppercase"),
                Function(has_number, "Must contain number")
            ]
        )
        yield Button("Login", id="login")

    @on(Button.Pressed, "#login")
    def handle_login(self) -> None:
        """Validate all inputs before login."""
        inputs = self.query(Input)
        if all(input.is_valid for input in inputs):
            self.submit_login()
        else:
            self.notify("Please fix validation errors")
```

### Real-time Validation Feedback

```python
class ValidatedInput(Static):
    def compose(self) -> ComposeResult:
        yield Input(id="input")
        yield Label("", id="error")

    @on(Input.Changed, "#input")
    def show_errors(self, event: Input.Changed) -> None:
        """Display validation errors in real-time."""
        error_label = self.query_one("#error", Label)

        if event.validation_result and not event.validation_result.is_valid:
            errors = "\n".join(event.validation_result.failure_descriptions)
            error_label.update(errors)
            error_label.styles.color = "red"
        else:
            error_label.update("✓ Valid")
            error_label.styles.color = "green"
```

### Dynamic Validation

```python
class ConditionalInput(Container):
    def __init__(self):
        super().__init__()
        self.enable_validation = False

    def compose(self) -> ComposeResult:
        yield Switch(id="validate_toggle")
        yield Input(id="input")

    @on(Switch.Changed, "#validate_toggle")
    def toggle_validation(self, event: Switch.Changed) -> None:
        """Enable/disable validation dynamically."""
        input_widget = self.query_one("#input", Input)

        if event.value:
            input_widget.validators = [
                Number(minimum=1, maximum=100)
            ]
        else:
            input_widget.validators = []

    @on(Input.Changed, "#input")
    def validate_if_enabled(self, event: Input.Changed) -> None:
        """Only show errors when validation is enabled."""
        if self.query_one("#validate_toggle", Switch).value:
            # Show validation errors
            pass
```

## Password Input

```python
# Simple password input
Input(password=True, placeholder="Enter password")

# With validation
Input(
    password=True,
    placeholder="Strong password required",
    validators=[
        Length(minimum=12),
        Function(has_special_char, "Must contain special character"),
        Function(has_uppercase, "Must contain uppercase letter"),
        Function(has_number, "Must contain number")
    ]
)
```

## Validate Empty

Allow empty values to bypass validation:

```python
Input(
    placeholder="Optional field",
    valid_empty=True,
    validators=[
        Number(minimum=0, maximum=100)
    ]
)
```

When `valid_empty=True`, empty strings are considered valid even if validators would reject them.

**From [Input - Textual Documentation](https://textual.textualize.io/widgets/input/#validate-empty) (accessed 2025-11-02)**

## Common Patterns

### Email Validation

```python
Input(
    placeholder="Email address",
    validators=[
        Regex(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            failure_description="Invalid email format"
        )
    ]
)
```

### Phone Number

```python
Input(
    placeholder="(123) 456-7890",
    restrict=r"[0-9\-\(\) ]*",
    validators=[
        Length(minimum=10),
        Regex(r"^\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}$")
    ]
)
```

### Credit Card

```python
Input(
    placeholder="1234 5678 9012 3456",
    restrict=r"[0-9 ]*",
    max_length=19,  # 16 digits + 3 spaces
    validators=[
        Function(luhn_check, "Invalid card number")
    ]
)
```

## Sources

**Official Documentation:**
- [Input Widget - Textual](https://textual.textualize.io/widgets/input/) (accessed 2025-11-02)
- [textual.validation API - Textual](https://textual.textualize.io/api/validation/) (accessed 2025-11-02)
- [Input Validation Guide - Textual](https://textual.textualize.io/guide/input/) (accessed 2025-11-02)

**Community Resources:**
- [Demo of Input Validation](https://www.youtube.com/watch?v=BqkdWFwNJmY) - Textualize YouTube (accessed 2025-11-02)
- [Custom Input Validation](https://www.youtube.com/watch?v=3EtArlSSOd0) - Textualize YouTube (accessed 2025-11-02)
- [Validating Input in Textual](https://lllama.github.io/posts/textual-input-validation/) - Lllama's Blog (accessed 2025-11-02)

**Blog Posts:**
- [Textual adds Input validation](https://textual.textualize.io/blog/2023/06/01/textual-adds-sparklines-selection-list-input-validation-and-tool-tips/) (accessed 2025-11-02)

## See Also

- [widgets/00-datatable-guide.md](00-datatable-guide.md) - DataTable widget patterns
- [widgets/01-tree-widget-guide.md](01-tree-widget-guide.md) - Tree widget patterns
- [core-concepts/](../core-concepts/) - Understanding widget architecture
