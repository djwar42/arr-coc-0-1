# Textual DevTools and Debugging

## Overview

Textual DevTools provide a powerful debugging console that restores `print()` functionality and adds logging capabilities for TUI development. Since TUI apps control the terminal display, normal console output would interfere with the UI - DevTools solves this by routing output to a separate console.

From [Devtools Guide](https://textual.textualize.io/guide/devtools/) (accessed 2025-11-02)

## The Console

### Basic Setup (Two Terminal Windows)

**Terminal 1 - Start the console:**
```bash
textual console
```

You'll see:
```
Textual Development Console v4.0.0
Run a Textual app with textual run --dev my_app.py to connect.
Press Ctrl+C to quit.
```

**Terminal 2 - Run your app in dev mode:**
```bash
textual run --dev my_app.py
```

Now all `print()` statements and logs appear in Terminal 1!

From [Devtools Guide - Console](https://textual.textualize.io/guide/devtools/#console)

## Using Print in Your App

Once DevTools console is running, `print()` works normally:

```python
from textual.app import App

class MyApp(App):
    def on_mount(self):
        print("App mounted!")  # Appears in console
        print(f"Current state: {self.state}")
```

## Textual Log Function

The `log()` function provides enhanced logging with Rich formatting:

```python
from textual import log
from textual.app import App

class MyApp(App):
    def on_mount(self):
        # Simple string
        log("Hello, World")

        # Log local variables
        log(locals())

        # Log with key/values
        log(children=self.children, pi=3.141592)

        # Log Rich renderables
        log(self.tree)
```

**What can you log?**
- Strings and primitives
- Data structures (dicts, lists, objects)
- Rich renderables (Tree, Table, Panel, etc.)
- Local variables via `locals()`

From [Devtools Guide - Textual Log](https://textual.textualize.io/guide/devtools/#textual-log)

## Convenient Log Method

App and Widget have a built-in `log()` method for quick debugging:

```python
from textual.app import App

class LogApp(App):
    def on_load(self):
        self.log("In the load handler!", pi=3.141529)

    def on_mount(self):
        self.log(self.tree)  # Log the widget tree

    def on_key(self, event):
        self.log(f"Key pressed: {event.key}")
```

From [Devtools Guide - Log Method](https://textual.textualize.io/guide/devtools/#log-method)

## Console Verbosity Control

### Increasing Verbosity

Some events are marked "verbose" and hidden by default. Enable with `-v`:

```bash
textual console -v
```

Shows all events including verbose ones like mouse movements.

### Decreasing Verbosity (Filtering)

Exclude specific log groups with `-x`:

```bash
# Log groups: EVENT, DEBUG, INFO, WARNING, ERROR, PRINT, SYSTEM, LOGGING, WORKER

# Exclude system and event logs
textual console -x SYSTEM -x EVENT

# Only show warnings, errors, and prints
textual console -x SYSTEM -x EVENT -x DEBUG -x INFO
```

**Use case:** Focus on your `print()` statements without framework noise.

From [Devtools Guide - Verbosity](https://textual.textualize.io/guide/devtools/#increasing-verbosity)

## Custom Port

If default port conflicts with other software:

```bash
# Terminal 1
textual console --port 7342

# Terminal 2
textual run --dev --port 7342 my_app.py
```

From [Devtools Guide - Custom Port](https://textual.textualize.io/guide/devtools/#custom-port)

## Standard Logging Integration

Integrate Python's standard `logging` module with Textual DevTools:

```python
import logging
from textual.app import App
from textual.logging import TextualHandler

# Configure logging to use TextualHandler
logging.basicConfig(
    level="NOTSET",
    handlers=[TextualHandler()],
)

class LogApp(App):
    """Using logging with Textual."""

    def on_mount(self) -> None:
        logging.debug("Debug message via TextualHandler")
        logging.info("Info message")
        logging.warning("Warning message")
        logging.error("Error message")

if __name__ == "__main__":
    LogApp().run()
```

**Note:** Standard logging only supports strings, not Rich renderables. Use `textual.log()` for rich output.

From [Devtools Guide - Logging Handler](https://textual.textualize.io/guide/devtools/#logging-handler)

## Running Apps with Textual Run

The `textual run` command provides more control than `python my_app.py`:

### From Python File

```bash
textual run my_app.py
textual run --dev my_app.py  # With DevTools
```

### From Python Module

```bash
# Runs app instance named "app" in music.play module
textual run music.play

# Specify different app instance/class name
textual run music.play:MusicPlayerApp
```

Works for both app instances and app classes.

### From Command

```bash
# Run installed command-line script
textual run -c textual colors

# Run built-in command
textual run -c "textual keys"
```

From [Devtools Guide - Run](https://textual.textualize.io/guide/devtools/#run)

## Live CSS Editing

Dev mode enables live CSS reloading - changes appear in milliseconds:

```bash
textual run --dev my_app.py
```

**Workflow:**
1. Open your `.css` or `.tcss` file in editor
2. Run app with `--dev` in terminal
3. Edit and save CSS
4. Changes appear instantly in running app

**Perfect for:**
- Iterating on styles
- Fine-tuning layouts
- Color scheme adjustments
- Responsive design testing

From [Devtools Guide - Live Editing](https://textual.textualize.io/guide/devtools/#live-editing)

## Serving Apps as Web Apps

Transform your TUI into a web application:

### Serve from Python File

```bash
textual serve my_app.py
```

### Serve from Command

```bash
textual serve "textual keys"
textual serve "python -m textual"
```

**Features:**
- Multiple simultaneous instances
- Refresh browser to reload (great during development)
- Share terminal apps as web apps
- No code changes needed

**Options:**
```bash
textual serve --help  # See all options
```

From [Devtools Guide - Serve](https://textual.textualize.io/guide/devtools/#serve)

## Complete Debugging Workflow Example

```python
# my_app.py
from textual import log
from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer
import logging
from textual.logging import TextualHandler

# Setup standard logging
logging.basicConfig(level="DEBUG", handlers=[TextualHandler()])

class DebugApp(App):
    """Demonstrating various debugging techniques."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Button("Click Me", id="demo")
        yield Footer()

    def on_mount(self):
        # Method 1: Simple print
        print("App mounted")

        # Method 2: Textual log with Rich formatting
        log("Mounted with tree:", self.tree)

        # Method 3: Convenient self.log
        self.log("Children:", children=self.children)

        # Method 4: Standard logging
        logging.info("Using standard logging")

    def on_button_pressed(self, event):
        print(f"Button pressed: {event.button.id}")
        self.log(f"Event details:", event=event)
        logging.debug(f"Button {event.button.id} was pressed")

if __name__ == "__main__":
    app = DebugApp()
    app.run()
```

**Running:**
```bash
# Terminal 1
textual console -v

# Terminal 2
textual run --dev my_app.py
```

## Debugging Best Practices

1. **Use log groups wisely** - Filter with `-x` to reduce noise
2. **Rich renderables for complex data** - Use `log()` instead of `print()` for objects
3. **Enable verbosity when stuck** - `textual console -v` shows everything
4. **Live CSS editing saves time** - Iterate on styles without restarting
5. **Use self.log() in event handlers** - Quick access to logging
6. **Standard logging for third-party libs** - TextualHandler integrates external logs
7. **Custom port for multi-project work** - Avoid port conflicts

## DevTools Command Reference

```bash
# Console
textual console                    # Start console
textual console -v                 # Verbose mode
textual console -x EVENT -x DEBUG  # Exclude log groups
textual console --port 7342        # Custom port

# Run
textual run my_app.py              # Run app
textual run --dev my_app.py        # Run with DevTools
textual run my.module:AppClass     # Run from module
textual run -c "textual keys"      # Run command

# Serve
textual serve my_app.py            # Serve as web app
textual serve "python -m textual"  # Serve module as web app

# Help
textual --help                     # Main help
textual run --help                 # Run command help
textual serve --help               # Serve command help
textual console --help             # Console command help
```

## Troubleshooting

### Console Not Connecting

**Problem:** App runs but console shows "waiting for connection"

**Solutions:**
1. Check both terminal windows are running
2. Ensure `--dev` flag is used: `textual run --dev my_app.py`
3. Try custom port: `--port 7342` on both console and run
4. Check firewall settings (console uses localhost connection)

### Print Statements Not Appearing

**Problem:** `print()` doesn't show in console

**Solutions:**
1. Ensure console is running in separate terminal
2. Verify app is run with `--dev` flag
3. Check console verbosity (may be filtered out)
4. Try `textual.log()` instead for Rich output

### Live CSS Not Updating

**Problem:** CSS changes don't reflect in running app

**Solutions:**
1. Ensure `--dev` flag is used
2. Check CSS file is actually being saved
3. Verify CSS file path is correct in app
4. Try restarting app with `--dev`

## Sources

**Official Documentation:**
- [Devtools Guide](https://textual.textualize.io/guide/devtools/) - Official guide (accessed 2025-11-02)
- [textual.logging API](https://textual.textualize.io/api/logging/) - TextualHandler reference

**Video Tutorials:**
- [How to install and use the Textual devtools console](https://www.youtube.com/watch?v=2w1hJPzQCJY) - Official video (YouTube)
- [How to log messages to the devtools console in Textual](https://www.youtube.com/watch?v=b2HRbz3dgxM) - Official video (YouTube)

**Community Resources:**
- [Textual has devtools](https://dev.to/waylonwalker/textual-has-devtools-j37) - DEV Community (Waylon Walker)
- [textual app devtools](https://waylonwalker.com/textual-app-devtools/) - Integration with CLI apps
