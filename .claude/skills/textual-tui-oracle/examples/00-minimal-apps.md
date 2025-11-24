# Minimal Textual Apps

**Quick reference for common app patterns**

## Absolute Minimum

The simplest possible Textual app:

```python
from textual.app import App

class MinimalApp(App):
    pass

if __name__ == "__main__":
    MinimalApp().run()
```

**Result**: Blank screen. Press Ctrl+Q to exit.

---

## Hello World

Display static text:

```python
from textual.app import App, ComposeResult
from textual.widgets import Static

class HelloApp(App):
    def compose(self) -> ComposeResult:
        yield Static("Hello, Textual!")

if __name__ == "__main__":
    HelloApp().run()
```

---

## Button Click

Handle button events:

```python
from textual.app import App, ComposeResult
from textual.widgets import Button

class ButtonApp(App):
    def compose(self) -> ComposeResult:
        yield Button("Click Me!")

    def on_button_pressed(self) -> None:
        self.notify("Button was clicked!")

if __name__ == "__main__":
    ButtonApp().run()
```

**Features demonstrated**:
- Widget composition
- Event handling
- Notifications

---

## Styled App

Add CSS styling:

```python
from textual.app import App, ComposeResult
from textual.widgets import Label, Button

class StyledApp(App):
    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }

    Label {
        margin: 2;
        text-style: bold;
    }

    Button {
        margin: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Welcome to Textual!")
        yield Button("Exit", variant="error")

    def on_button_pressed(self) -> None:
        self.exit()

if __name__ == "__main__":
    StyledApp().run()
```

**Features demonstrated**:
- CSS styling
- Layout (centering)
- Button variants
- App exit

---

## Question Dialog

App that returns a value:

```python
from textual.app import App, ComposeResult
from textual.widgets import Label, Button

class QuestionApp(App[str]):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-gutter: 2;
        padding: 2;
    }

    #question {
        width: 100%;
        height: 100%;
        column-span: 2;
        content-align: center bottom;
        text-style: bold;
    }

    Button {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Do you love Textual?", id="question")
        yield Button("Yes", id="yes", variant="primary")
        yield Button("No", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.exit(event.button.id)

if __name__ == "__main__":
    app = QuestionApp()
    reply = app.run()
    print(f"You answered: {reply}")
```

**Features demonstrated**:
- Return values from app
- Grid layout
- Widget IDs
- Button variants
- Event object usage

---

## Key Handler

Respond to keyboard input:

```python
from textual.app import App
from textual import events

class KeyApp(App):
    COLORS = ["white", "red", "green", "blue", "yellow"]

    def on_mount(self) -> None:
        self.screen.styles.background = "darkblue"

    def on_key(self, event: events.Key) -> None:
        if event.key.isdecimal():
            index = int(event.key)
            if index < len(self.COLORS):
                self.screen.styles.background = self.COLORS[index]
        elif event.key == "q":
            self.exit()

if __name__ == "__main__":
    KeyApp().run()
```

**Features demonstrated**:
- Key event handling
- on_mount lifecycle hook
- Dynamic styling
- Exiting app

---

## Multiple Widgets

Compose multiple widgets:

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Static

class MultiWidgetApp(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("This app has multiple widgets")
        yield Button("Button 1")
        yield Button("Button 2")
        yield Button("Exit", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "exit":
            self.exit()
        else:
            self.notify(f"{event.button.label} was clicked!")

if __name__ == "__main__":
    MultiWidgetApp().run()
```

**Features demonstrated**:
- Multiple widget composition
- Header and Footer widgets
- Event routing
- Button labels

---

## Common Patterns

### Exit on "q" key
```python
def on_key(self, event: events.Key) -> None:
    if event.key == "q":
        self.exit()
```

### Notification
```python
self.notify("Message here", severity="information")
self.notify("Warning!", severity="warning")
self.notify("Error!", severity="error")
```

### Query widgets
```python
# By ID
button = self.query_one("#my-button", Button)

# By type
all_buttons = self.query(Button)

# By CSS selector
active_buttons = self.query("Button.active")
```

### Mount widget dynamically
```python
def on_key(self) -> None:
    self.mount(Button("New Button"))
```

---

**For more examples**:
- Official examples: https://github.com/Textualize/textual/tree/main/examples
- Tutorial: https://textual.textualize.io/tutorial/
- Widget gallery: https://textual.textualize.io/widget_gallery/
