# Debugging Textual Applications

## Overview

Debugging text-based user interface (TUI) applications created with Textual presents unique challenges since the TUI takes over your terminal, preventing you from seeing standard output. This guide covers the practical debugging tools and techniques that Textual provides to solve this problem.

From [How to Debug Your Textual Application](https://www.blog.pythonlibrary.org/2024/11/19/how-to-debug-your-textual-application/) (Mouse Vs Python, accessed 2025-11-02):

The core issue: IDEs like PyCharm, WingIDE, and Visual Studio Code don't work well with Textual because a TUI requires a fully functional terminal to interact with it. Standard debugging approaches that work for GUI applications don't apply here.

## Key Debugging Approach: Two Terminal Windows

The fundamental debugging strategy for Textual applications requires:

1. **Terminal 1**: Run `textual console` (listens for Textual applications)
2. **Terminal 2**: Run your application with `textual run --dev your_app.py`

This dual-terminal setup allows you to see all logging and print output in the console window while your application runs normally in the other terminal.

## Setting Up Debugging Tools

### Installation

First, install Textual's development tools:

```bash
python -m pip install textual-dev --upgrade
```

The `textual-dev` package provides the console and DevTools needed for debugging.

## Debugging Techniques

### 1. Developer Mode with Textual Console

**Basic Setup**:

Terminal 1 - Start the Textual console:
```bash
textual console
```

Terminal 2 - Run your application in developer mode:
```bash
textual run --dev hello_textual.py
```

This establishes a debugging connection between your application and the console.

**Example Application**:

```python
from textual.app import App, ComposeResult
from textual.widgets import Button

class WelcomeButton(App):
    def compose(self) -> ComposeResult:
        yield Button("Exit")

    def on_button_pressed(self) -> None:
        self.mount(Button("Other"))

if __name__ == "__main__":
    app = WelcomeButton()
    app.run()
```

When you run this in developer mode, all output appears in the Textual console window instead of the terminal.

### 2. Print Statements

Standard `print()` statements work within the Textual console:

```python
from textual.app import App, ComposeResult
from textual.widgets import Button

class WelcomeButton(App):
    def compose(self) -> ComposeResult:
        yield Button("Exit")
        print("The compose() method was called!")

    def on_button_pressed(self) -> None:
        self.mount(Button("Other"))

if __name__ == "__main__":
    app = WelcomeButton()
    app.run()
```

When executed with `textual run --dev`, the print output appears in the Textual console without interfering with your application's terminal display.

### 3. Logging with `self.log.info()`

For more structured logging, use the app's logging method:

```python
def on_button_pressed(self) -> None:
    self.log.info("You pressed a button")
    self.mount(Button("Other"))
```

The `self.log.info()` method provides logging that integrates directly with Textual's console system. Advantages over print statements:

- Structured output with timestamps
- Different log levels (info, debug, warning, etc.)
- Easily filterable output
- Better organization in complex applications

### 4. Filtering Console Output

When debugging complex applications, the Textual console output can become verbose. Use the `-x` or `--exclude` flag to suppress specific log groups:

```bash
textual console -x SYSTEM -x EVENT -x DEBUG -x INFO
```

Common log groups to exclude:
- **SYSTEM**: Framework system messages
- **EVENT**: Widget events and user interactions
- **DEBUG**: Debug-level messages
- **INFO**: Informational messages
- **WARNING**: Warning messages (usually keep)
- **ERROR**: Error messages (always keep)

**Example**: Run the console with only error and warning messages:
```bash
textual console -x SYSTEM -x EVENT -x DEBUG -x INFO
```

This dramatically reduces console noise when focusing on specific issues.

### 5. Notification Method for User-Facing Debugging

The `notify()` method displays messages directly in your application UI:

```python
from textual.app import App, ComposeResult
from textual.widgets import Button

class WelcomeButton(App):
    def compose(self) -> ComposeResult:
        yield Button("Exit")

    def on_button_pressed(self) -> None:
        self.mount(Button("Other"))
        self.notify("You pressed the button!")

if __name__ == "__main__":
    app = WelcomeButton()
    app.run()
```

**Notify Parameters**:

- `message` (required): The notification message
- `title` (optional): A title for the notification
- `severity` (optional): One of "information", "warning", or "error" (controls color)
- `timeout` (optional): How many seconds to display the notification

**Example with full parameters**:

```python
self.notify(
    "You pressed the button!",
    title="Info Message",
    severity="error"  # Shows in red for emphasis
)
```

Notifications appear as overlay messages in your TUI, useful for:
- Confirming user actions during development
- Displaying state changes
- Showing error conditions
- Testing notification styling

### 6. Bell Method for Attention

Add audio feedback during debugging:

```python
def on_button_pressed(self) -> None:
    self.bell()  # Play system bell sound
    self.mount(Button("Other"))
```

The `bell()` method plays the system bell, useful for:
- Getting developer attention when specific events occur
- Testing alert conditions
- Debugging asynchronous operations (bell sounds when event completes)

## Debugging Workflow Summary

1. **Identify Problem**: Application runs but behavior is unexpected
2. **Open Two Terminals**: One for console, one for application
3. **Start Console**: `textual console` in Terminal 1
4. **Run Application**: `textual run --dev your_app.py` in Terminal 2
5. **Add Logging**: Use `print()`, `self.log.info()`, or `self.notify()`
6. **Filter Output**: Use `-x` flags if console is too verbose
7. **Iterate**: Modify code and re-run application in Terminal 2

## IDE Considerations

Unfortunately, standard Python IDEs have limitations with Textual:

**PyCharm**: Doesn't work well with TUI applications due to terminal emulation
**WingIDE**: No terminal emulator built-in
**Visual Studio Code**: Doesn't work out of the box (custom configuration might help, but complex)

**Recommendation**: Use the dual-terminal approach with your preferred editor (VS Code, Sublime, Vim, etc.) rather than attempting IDE debugging integration.

## Best Practices

1. **Use meaningful log messages**: Help future you understand what was happening
   ```python
   self.log.info(f"Button pressed at {datetime.now()}")
   ```

2. **Combine logging methods**: Mix `print()` for simple output and `self.log.info()` for structured logging
   ```python
   print(f"DEBUG: value = {value}")
   self.log.info(f"Processing completed: {result}")
   ```

3. **Use notifications sparingly**: For user-facing information during testing
   ```python
   self.notify(f"Loaded {count} items", title="Success")
   ```

4. **Filter progressively**: Start with full output, then exclude groups as needed
   ```bash
   # Start with everything
   textual console

   # Too verbose? Reduce noise
   textual console -x SYSTEM -x EVENT
   ```

5. **Keep both terminals visible**: Use split screen or multiple monitor setup for maximum efficiency

## Common Debugging Scenarios

### Scenario 1: Method Not Being Called

**Problem**: Button click handler doesn't execute
**Solution**:

```python
def on_button_pressed(self) -> None:
    self.log.info("Button pressed handler called")
    # Rest of handler code
```

Check Textual console for the log message.

### Scenario 2: Widget State Issues

**Problem**: Widget appears but doesn't update
**Solution**:

```python
def update_widget(self):
    self.log.info(f"Widget state before: {self.widget.value}")
    self.widget.value = "new_value"
    self.log.info(f"Widget state after: {self.widget.value}")
```

### Scenario 3: Event Processing Order

**Problem**: Events fire in unexpected order
**Solution**:

```python
def on_mount(self) -> None:
    self.log.info("on_mount called")

def on_show(self) -> None:
    self.log.info("on_show called")

def watch_value(self, old_value, new_value) -> None:
    self.log.info(f"Value changed from {old_value} to {new_value}")
```

The console shows exactly when each event fires.

## References

From [How to Debug Your Textual Application](https://www.blog.pythonlibrary.org/2024/11/19/how-to-debug-your-textual-application/) (Mouse Vs Python, November 19, 2024):
- Two-terminal debugging approach
- Textual console setup and usage
- Print statements and logging methods
- Notification system
- Bell feedback mechanism
- Console output filtering techniques

From [Textual DevTools Documentation](https://textual.textualize.io/guide/devtools/):
- Official DevTools guide
- Console API reference
- Logging configuration
- Advanced debugging features

## Video Tutorial

YouTube: [Debugging a Textual Application](https://www.youtube.com/watch?v=y5mxb9yyBpM) (Mouse Vs Python)
- Duration: 9:00
- Views: 670+
- Content: Step-by-step debugging demonstration with real applications
- Channel: [Mouse Vs Python](https://www.youtube.com/@MouseVsPython)

## Sources

**Web Articles:**
- [How to Debug Your Textual Application](https://www.blog.pythonlibrary.org/2024/11/19/how-to-debug-your-textual-application/) - Mouse Vs Python (accessed 2025-11-02)

**Official Documentation:**
- [Textual DevTools Guide](https://textual.textualize.io/guide/devtools/) - Textual documentation

**Video Resources:**
- [Debugging a Textual Application](https://www.youtube.com/watch?v=y5mxb9yyBpM) - YouTube, Mouse Vs Python channel

**Related Resources:**
- [Mouse Vs Python Blog](https://www.blog.pythonlibrary.org/) - Python tutorials and TUI development
- [Textual Framework](https://textual.textualize.io/) - Official Textual documentation
