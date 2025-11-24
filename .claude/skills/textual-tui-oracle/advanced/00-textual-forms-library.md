# Textual Forms Library - Dynamic Forms for Textual

## Overview

**textual-forms** is a third-party library that provides dynamic form functionality for Textual TUI applications. It simplifies building web-like forms with validation, field types, and button controls within Textual apps.

From [rhymiz/textual-forms](https://github.com/rhymiz/textual-forms) (accessed 2025-11-02):
- **Status**: WIP (Work In Progress) - breaking changes possible
- **License**: MIT
- **Latest Release**: 0.3.0 (February 18, 2023)
- **GitHub Stars**: 4
- **Requirements**: Python >=3.7,<4, Textual >=0.11.0

**⚠️ Important**: This library is still experimental and may introduce breaking changes. Use with caution in production environments.

## Installation

```bash
pip install textual-forms
```

## Core Components

### Form Container (`textual_forms.forms.Form`)

The main container widget that behaves like a web-based form with validations.

**Key Features**:
- Auto-validates all fields on input change
- Enables/disables buttons based on form validity
- Emits events when buttons are pressed
- Reactive data binding (form data updates automatically)

**API**:

From [forms.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/forms.py):

```python
class Form(Widget):
    """A container widget for inputs and buttons"""

    # Reactive attributes
    data: dict[str, Any] = reactive({})    # Form field values
    valid: bool = reactive(False)          # Overall form validity

    def __init__(
        self,
        *,
        fields: list[Field | IntegerField | NumberField | StringField],
        buttons: list[Button],
        **kwargs,
    ) -> None:
        """
        Args:
            fields: List of form field widgets
            buttons: List of button widgets
        """
```

**Default CSS Styling**:
```css
Form {
    padding: 1
}
Form>Container {
    padding: 0;
    margin: 0;
}
#button_group {
    margin-left: 1;
}
#button_group > Button {
    margin-right: 5;
}
```

### Form Events

**Form.Event Message**:
```python
class Event(Message):
    """Emitted whenever a button is pressed"""

    def __init__(
        self,
        sender: MessageTarget,
        event: str,      # Button ID
        data: dict[str, Any],  # Form data
    ) -> None:
```

**Handling Form Events**:
```python
def on_form_event(self, message: Form.Event) -> None:
    if message.event == 'submit':
        # message.data contains all form field values
        process_form_data(message.data)
```

## Field Types

### Base Field (`textual_forms.fields.Field`)

All field types inherit from the base `Field` class.

From [fields.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/fields.py):

**Reactive Attributes**:
```python
class Field(Widget):
    value: str = reactive("")      # Current field value
    dirty: bool = reactive(False)  # Has user interacted?
    valid: bool = reactive(False)  # Is value valid?
```

**Visual Feedback**:
- Green border: Valid field
- Red border: Invalid field
- Error message displayed below field
- Border cleared when field is empty (not dirty)

**Default CSS**:
```css
Field {
    height: 5;
    margin: 0;
}
Field>Input {
    height: 1;
    margin: 0;
}
FieldError {
    color: red;
}
```

### StringField

Text input with length validation.

**API**:
```python
StringField(
    name: str,                    # Field name (required)
    *,
    value: str | None = None,     # Initial value
    required: bool = False,       # Is field required?
    placeholder: str | None = None,  # Placeholder text
    min_length: int = 0,          # Minimum string length
    max_length: int | None = None,   # Maximum string length
    **kwargs,
)
```

**Example**:
```python
StringField(
    "username",
    required=True,
    min_length=3,
    max_length=20,
    placeholder="Enter username"
)
```

**Validation Rules**:
- Required: Value must not be empty
- Min length: `len(value) >= min_length`
- Max length: `len(value) <= max_length`
- Error message: "value must be between {min} and {max} characters"

### IntegerField

Integer input with range validation.

**API**:
```python
IntegerField(
    name: str,
    *,
    value: int | None = None,
    required: bool = False,
    placeholder: str | None = None,
    min_value: int | None = None,  # Minimum allowed value
    max_value: int | None = None,  # Maximum allowed value
    **kwargs,
)
```

**Example**:
```python
IntegerField(
    "age",
    required=True,
    min_value=21,
    max_value=120,
    placeholder="Enter your age"
)
```

**Validation Rules**:
- Pattern: Must match regex `\d+$` (digits only)
- Min value: `int(value) >= min_value`
- Max value: `int(value) <= max_value`
- Error messages:
  - "Invalid integer" - not a valid integer
  - "value must be between {min} and {max}"
  - "value must be greater than or equal to {min}"
  - "value must be less than or equal to {max}"

### NumberField

Numeric input (integers and decimals).

**API**:
```python
NumberField(
    name: str,
    *,
    value: int | None = None,
    required: bool = False,
    placeholder: str | None = None,
    **kwargs,
)
```

**Validation Rules**:
- Pattern: Must match regex `\d*[.,]?\d+$`
- Accepts: integers, decimals with `.` or `,`
- Error message: "invalid number"

## Buttons

### Button (`textual_forms.buttons.Button`)

Enhanced Textual Button with form integration.

From [buttons.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/buttons.py):

**API**:
```python
Button(
    label: str | Text | None = None,
    disabled: bool = False,
    variant: ButtonVariant = "default",
    *,
    id: str | None = None,
    name: str | None = None,
    classes: str | None = None,
    enabled_on_form_valid: bool = False,  # Enable only when form is valid
)
```

**enabled_on_form_valid**:
When set to `True`, the button will be:
- Disabled initially if form is invalid
- Automatically enabled when form becomes valid
- Automatically disabled when form becomes invalid

**Example**:
```python
Button(
    "Submit",
    variant="primary",
    enabled_on_form_valid=True,  # Only enabled when form is valid
)
```

## Validation System

### Built-in Validators

From [validators.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/validators.py):

**FieldValidator Base Class**:
```python
class FieldValidator:
    """Base field validator class"""

    def validate(self, value: str, rules: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Returns:
            (is_valid, error_message)
        """
        raise NotImplementedError

    def __call__(
        self,
        value: str,
        required: bool,
        rules: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if required and value is None:
            return False, "This value is required"
        return self.validate(value, rules=rules)
```

**Built-in Validators**:
- `StringValidator` - String length validation
- `IntegerFieldValidator` - Integer pattern and range validation
- `NumberFieldValidator` - Numeric pattern validation

### Custom Validators

Create custom field types with specialized validation:

**Example: UUID Field**:
```python
from textual_forms.fields import Field
from textual_forms.validators import FieldValidator
from typing import Any
import re

class UUIDValidator(FieldValidator):
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    def validate(self, value: str, rules: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.UUID_PATTERN.match(value):
            return False, "Invalid UUID format"
        return True, None

class UUIDField(Field):
    validator = UUIDValidator()

    def __init__(
        self,
        name: str,
        *,
        value: str | None = None,
        required: bool = False,
        placeholder: str | None = None,
        **kwargs,
    ):
        data: dict[str, Any] = {
            "name": name,
            "value": value,
            "required": required,
            "placeholder": placeholder,
            "rules": {},
        }
        super().__init__(data, **kwargs)
```

**Usage**:
```python
Form(
    fields=[
        UUIDField("user_id", required=True),
        StringField("name"),
    ],
    buttons=[Button("Submit", enabled_on_form_valid=True)],
)
```

## Complete Example

From [README.md](https://github.com/rhymiz/textual-forms/blob/main/README.md):

**Order Form Application**:
```python
from rich.table import Table
from textual.app import App, ComposeResult
from textual.widgets import Static

from textual_forms.forms import Form
from textual_forms.fields import StringField, IntegerField
from textual_forms.buttons import Button

class BasicTextualForm(App):
    def compose(self) -> ComposeResult:
        yield Static(id="submitted-data")
        yield Static("Order for beers")
        yield Form(
            fields=[
                StringField("name"),
                IntegerField("age", required=True, min_value=21),
            ],
            buttons=[
                Button(
                    "Submit",
                    enabled_on_form_valid=True,
                )
            ],
        )

    def on_form_event(self, message: Form.Event) -> None:
        if message.event == 'submit':
            # Display submitted data in a table
            table = Table(*message.data.keys())
            table.add_row(*message.data.values())
            self.query_one('#submitted-data').update(table)

if __name__ == '__main__':
    BasicTextualForm().run()
```

**How It Works**:

1. **Initial State**: Form displays with two fields
   - "name" field (optional string)
   - "age" field (required integer, min 21)
   - Submit button is **disabled** (form invalid)

2. **User Input**: As user types...
   - Form data updates reactively: `form.data = {"name": "...", "age": "..."}`
   - Each field validates independently
   - Field shows green/red border based on validity
   - Error messages appear below invalid fields

3. **Form Validation**: Form becomes valid when...
   - All required fields have values
   - All field validators pass
   - Submit button automatically **enables**

4. **Submission**: When user clicks Submit...
   - `Form.Event` emitted with `event='submit'`
   - `message.data` contains all field values as dict
   - Application processes the data

## Form Behavior Details

### Reactive Validation Flow

From [forms.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/forms.py):

```python
async def on_field_value_changed(self, message: Field.ValueChanged) -> None:
    """Listens for form field changes and assesses the validity of the form"""
    self.data[getattr(message.sender, "_field_name")] = message.value
    self.valid = all([x.valid for x in self.query(Field)])
```

**Validation Triggers**:
1. User types in any field
2. `Field.ValueChanged` message emitted
3. Form updates `data` dict with new value
4. Form recalculates `valid` by checking ALL fields
5. `watch_valid()` enables/disables buttons

### Button State Management

```python
async def watch_valid(self, valid: bool) -> None:
    """Enable/disable buttons based on the state of the form"""
    for button in self._watching_form_valid:
        button.disabled = not valid
```

**Button Behavior**:
- Buttons with `enabled_on_form_valid=True` are tracked
- When `form.valid` changes, all tracked buttons update
- Buttons with `enabled_on_form_valid=False` remain unaffected

### Field Dirty State

From [fields.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/fields.py):

```python
async def watch_value(self, value: str) -> None:
    self.dirty = bool(value)

    if self.dirty:
        # Validate and show borders/errors
        self.valid, message = self.validator(value, required, rules=self.rules)
        input_widget.styles.border = (
            self.field_success_style if self.valid else self.field_error_style
        )
        error_widget.update(error_text)
    else:
        # Clear borders and errors
        self.valid = not required
        input_widget.styles.border = None
        error_widget.update("")
```

**Dirty State Logic**:
- Field is "dirty" when it has any value
- Empty fields are "not dirty"
- Not dirty + not required = valid
- Not dirty + required = invalid
- Dirty = run validator

## Integration Patterns

### Multi-Step Forms

```python
class MultiStepForm(App):
    def __init__(self):
        super().__init__()
        self.step = 1
        self.form_data = {}

    def compose(self) -> ComposeResult:
        yield Static(id="step-indicator")
        yield Static(id="form-container")
        self.show_step(1)

    def show_step(self, step: int) -> None:
        container = self.query_one("#form-container")

        if step == 1:
            form = Form(
                fields=[
                    StringField("name", required=True),
                    StringField("email", required=True),
                ],
                buttons=[Button("Next", id="next", enabled_on_form_valid=True)],
            )
        elif step == 2:
            form = Form(
                fields=[
                    IntegerField("age", required=True, min_value=18),
                    StringField("phone", required=True),
                ],
                buttons=[
                    Button("Back", id="back"),
                    Button("Submit", id="submit", enabled_on_form_valid=True),
                ],
            )

        container.update(form)
        self.query_one("#step-indicator").update(f"Step {step} of 2")

    def on_form_event(self, message: Form.Event) -> None:
        if message.event == "next":
            self.form_data.update(message.data)
            self.step = 2
            self.show_step(2)
        elif message.event == "back":
            self.step = 1
            self.show_step(1)
        elif message.event == "submit":
            self.form_data.update(message.data)
            self.process_complete_form(self.form_data)
```

### Form with Dynamic Fields

```python
class DynamicFieldsForm(App):
    def compose(self) -> ComposeResult:
        yield Form(
            fields=[
                StringField("category", required=True),
            ],
            buttons=[Button("Add Details", id="add-details")],
        )

    def on_form_event(self, message: Form.Event) -> None:
        if message.event == "add-details":
            category = message.data.get("category")

            # Rebuild form with category-specific fields
            if category == "book":
                additional_fields = [
                    StringField("author", required=True),
                    IntegerField("pages", required=True),
                ]
            elif category == "movie":
                additional_fields = [
                    StringField("director", required=True),
                    IntegerField("runtime", required=True),
                ]

            new_form = Form(
                fields=[
                    StringField("category", value=category, required=True),
                    *additional_fields,
                ],
                buttons=[Button("Submit", enabled_on_form_valid=True)],
            )
            # Replace form in UI...
```

### Form Data Processing

```python
def on_form_event(self, message: Form.Event) -> None:
    if message.event == "submit":
        # message.data is a dict with field names as keys
        data = message.data

        # Validate business logic
        if data["age"] < 18 and data["requires_consent"]:
            self.show_error("Parental consent required for users under 18")
            return

        # Transform data
        user = User(
            name=data["name"],
            email=data["email"],
            age=int(data["age"]),
        )

        # Persist
        self.database.save(user)

        # Show confirmation
        self.query_one("#status").update(f"User {user.name} created successfully")
```

## Limitations and Considerations

**WIP Status** (from README):
> This library is still very much WIP 🧪. This means that breaking changes can be introduced at any point in time.

**Current Limitations**:
1. **Limited Field Types**: Only String, Integer, Number built-in
   - No date, email, password, select, checkbox, radio fields
   - Must create custom fields for specialized inputs

2. **Basic Validation**: Simple regex and range validation only
   - No async validation
   - No cross-field validation (e.g., "password confirm" matching "password")
   - No conditional validation based on other fields

3. **Styling**: Default CSS is minimal
   - Limited customization options
   - May need custom CSS for complex layouts

4. **No File Upload**: No built-in support for file input fields

5. **Form Reset**: No built-in method to clear/reset form to initial state

6. **Error Handling**: Basic error display only
   - Errors appear below fields
   - No global form error summary
   - No custom error placement

**Textual Version Compatibility**:
- Requires Textual >=0.11.0
- May need updates for newer Textual versions
- Last updated February 2023 (may be outdated for Textual 0.50+)

## Workarounds and Extensions

### Password Field

```python
from textual.widgets import Input
from textual_forms.fields import StringField

class PasswordField(StringField):
    def compose(self) -> ComposeResult:
        input_kwargs = {
            "id": self._field_id,
            "name": self._field_name,
            "placeholder": self.placeholder,
            "password": True,  # Hide characters
        }
        if self.data["value"]:
            input_kwargs["value"] = self.data["value"]
        yield Input(**input_kwargs)
        yield FieldError("", id=self._field_error_id)
```

### Form Reset Method

```python
class ResettableForm(Form):
    def reset(self) -> None:
        """Clear all fields and reset form state"""
        for field in self.query(Field):
            input_widget = field.query_one(Input)
            input_widget.value = ""
            field.valid = False
            field.dirty = False
        self.data = {}
        self.valid = False
```

### Cross-Field Validation

```python
class PasswordConfirmForm(App):
    def compose(self) -> ComposeResult:
        yield Form(
            fields=[
                StringField("password", required=True, min_length=8),
                StringField("password_confirm", required=True),
            ],
            buttons=[Button("Submit", id="submit")],
        )
        yield Static(id="form-error")

    def on_form_event(self, message: Form.Event) -> None:
        if message.event == "submit":
            if message.data["password"] != message.data["password_confirm"]:
                self.query_one("#form-error").update(
                    "Passwords do not match",
                    style="red"
                )
                return

            # Passwords match, proceed
            self.create_account(message.data)
```

## Best Practices

### Field Naming
```python
# Use descriptive names that match your data model
Form(
    fields=[
        StringField("user_email"),  # Clear what it represents
        IntegerField("product_quantity"),
        StringField("shipping_address"),
    ]
)

# Avoid generic names
Form(
    fields=[
        StringField("field1"),  # ❌ Unclear
        IntegerField("input"),  # ❌ Too generic
    ]
)
```

### Required vs Optional
```python
# Mark truly required fields
Form(
    fields=[
        StringField("name", required=True),      # Must have
        StringField("email", required=True),     # Must have
        StringField("phone", required=False),    # Optional
        StringField("company", required=False),  # Optional
    ]
)
```

### Button IDs
```python
# Use descriptive button IDs for event handling
Form(
    buttons=[
        Button("Save Draft", id="save-draft"),
        Button("Submit", id="submit", enabled_on_form_valid=True),
        Button("Cancel", id="cancel"),
    ]
)

def on_form_event(self, message: Form.Event) -> None:
    if message.event == "save-draft":
        self.save_draft(message.data)
    elif message.event == "submit":
        self.submit_form(message.data)
    elif message.event == "cancel":
        self.exit()
```

### Validation Feedback
```python
# Provide clear min/max constraints
IntegerField(
    "quantity",
    required=True,
    min_value=1,
    max_value=100,
    placeholder="Enter quantity (1-100)"  # User sees constraints
)

StringField(
    "username",
    required=True,
    min_length=3,
    max_length=20,
    placeholder="Username (3-20 chars)"  # User sees constraints
)
```

## Sources

**GitHub Repository**:
- [rhymiz/textual-forms](https://github.com/rhymiz/textual-forms) - Main repository (accessed 2025-11-02)
- [forms.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/forms.py) - Form container implementation
- [fields.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/fields.py) - Field types implementation
- [validators.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/validators.py) - Validation system
- [buttons.py](https://github.com/rhymiz/textual-forms/blob/main/src/textual_forms/buttons.py) - Button component

**Package Info**:
- [PyPI: textual-forms](https://pypi.org/project/textual-forms/) - Package distribution
- License: MIT
- Latest Release: 0.3.0 (2023-02-18)

**Related Documentation**:
- [Textual Documentation](https://textual.textualize.io/) - Parent framework
- [Textual Widgets](https://textual.textualize.io/widgets/) - Built-in Textual widgets

## See Also

- [core-concepts/01-widgets.md](../core-concepts/01-widgets.md) - Base widget system
- [core-concepts/02-reactive-attributes.md](../core-concepts/02-reactive-attributes.md) - Reactive programming in Textual
- [core-concepts/03-messages-events.md](../core-concepts/03-messages-events.md) - Message passing system
- [widgets/04-input.md](../widgets/04-input.md) - Textual Input widget (used by fields)
- [widgets/03-button.md](../widgets/03-button.md) - Textual Button widget (extended by forms)
