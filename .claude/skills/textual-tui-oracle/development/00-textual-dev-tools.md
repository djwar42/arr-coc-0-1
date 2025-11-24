# Textual Development Tools (textual-dev)

## Overview

The `textual-dev` package provides essential development tools for building Textual applications. It includes a command-line interface (`textual` command) with utilities for running apps in development mode, debugging with a separate console, live CSS editing, serving apps in browsers, and built-in preview tools.

**Installation:**
```bash
pip install textual-dev
```

**Package Information:**
- **GitHub**: https://github.com/Textualize/textual-dev
- **Version**: 1.8.0 (as of October 2025)
- **Dependencies**: textual (>=0.86.2), textual-serve (>=1.0.3), aiohttp, click, msgpack
- **Python Support**: 3.9+

---

## Command-Line Interface

### Available Commands

The `textual` command provides several subcommands:

```bash
textual --help          # Show all available commands
textual run             # Run Textual apps with development features
textual console         # Launch the development console for debugging
textual serve           # Serve Textual apps in a web browser
textual borders         # Preview border styles
textual easing          # Preview animation easing functions
textual colors          # Explore the design system
textual keys            # Show key events
textual diagnose        # Print environment information
```

---

## textual run - Running Applications

### Basic Usage

Run Textual apps from Python files or modules:

```bash
# Run from Python file
textual run my_app.py

# Run from Python module (imports "app" by default)
textual run module.foo

# Run specific app instance or class
textual run module.foo:MyApp
```

### Development Mode (--dev)

Enable development mode to activate the debugging console and live CSS editing:

```bash
textual run --dev my_app.py
```

**Development mode enables:**
- Connection to `textual console` for logging
- Live CSS editing (changes reflect immediately)
- Additional debugging features
- Environment variable: `TEXTUAL=debug,devtools`

### Running Commands

Use the `-c` flag to run command-line scripts that launch Textual apps:

```bash
textual run -c "textual colors"
textual run -c "python -m textual"
```

### Connection Options

Specify custom host/port for devtools console connection:

```bash
# Custom host (default: localhost)
textual run --dev --host 192.168.1.100 my_app.py

# Custom port (default: 8081)
textual run --dev --port 7342 my_app.py
```

### Testing Options

```bash
# Simulate key presses (comma-separated)
textual run --press "ctrl+c,q,enter" my_app.py

# Take screenshot after delay (seconds)
textual run --screenshot 5 my_app.py

# Specify screenshot path and filename
textual run --screenshot 5 --screenshot-path ./screenshots --screenshot-filename app.svg my_app.py

# Show return value on exit
textual run --show-return my_app.py
```

### Environment Variables Set by textual run

When using `--dev`, these environment variables are automatically set:

- `TEXTUAL=debug,devtools` - Enable debugging features
- `TEXTUAL_DEVTOOLS_HOST` - Custom devtools host (if `--host` specified)
- `TEXTUAL_DEVTOOLS_PORT` - Custom devtools port (if `--port` specified)
- `TEXTUAL_PRESS` - Keys to simulate (if `--press` specified)
- `TEXTUAL_SCREENSHOT` - Screenshot delay (if `--screenshot` specified)
- `TEXTUAL_SCREENSHOT_LOCATION` - Screenshot directory
- `TEXTUAL_SCREENSHOT_FILENAME` - Screenshot filename
- `TEXTUAL_SHOW_RETURN` - Show return value on exit

---

## textual console - Debug Console

### Overview

The **Textual console** is a separate terminal window that displays logs, print statements, and debug information from your running Textual application. This solves the fundamental problem of terminal apps: you can't use `print()` for debugging because it overwrites your UI.

### Two-Terminal Workflow

**Terminal 1 (Console):**
```bash
textual console
```

Output:
```
Textual Development Console v4.0.0
Run a Textual app with textual run --dev my_app.py to connect.
Press Ctrl+C to quit.
```

**Terminal 2 (App):**
```bash
textual run --dev my_app.py
```

Now all `print()` statements and logs from your app appear in Terminal 1, while Terminal 2 shows the running app UI.

### Console Features

#### 1. Print Statement Restoration

Use standard Python `print()` in your app - output appears in the console:

```python
from textual.app import App

class MyApp(App):
    def on_mount(self):
        print("App mounted!")  # Appears in console
        print(f"Children: {self.children}")
```

#### 2. Textual Logging

Import and use the `log` function for rich output:

```python
from textual import log
from textual.app import App

class MyApp(App):
    def on_mount(self):
        log("Simple message")
        log(locals())  # Log local variables
        log(children=self.children, screen=self.screen)  # Key/value pairs
        log(self.tree)  # Rich renderables
```

#### 3. App/Widget Log Method

Convenient shortcut available on `App` and `Widget` objects:

```python
from textual.app import App

class MyApp(App):
    def on_load(self):
        self.log("In on_load handler!", pi=3.141529)

    def on_mount(self):
        self.log(self.tree)  # Log widget tree
        self.log("Screen size:", self.size)
```

#### 4. Standard Logging Handler

Integrate Python's logging module with Textual console:

```python
import logging
from textual.app import App
from textual.logging import TextualHandler

# Configure logging to use TextualHandler
logging.basicConfig(
    level="NOTSET",
    handlers=[TextualHandler()],
)

class MyApp(App):
    def on_mount(self):
        logging.debug("Debug message")
        logging.info("Info message")
        logging.warning("Warning message")
        logging.error("Error message")
```

**Note**: The logging handler works with strings only, not Rich renderables.

### Console Options

#### Verbose Mode (-v)

Show additional verbose log messages (normally excluded):

```bash
textual console -v
```

#### Exclude Log Groups (-x)

Filter out specific log groups:

```bash
# Available groups:
# EVENT, DEBUG, INFO, WARNING, ERROR, PRINT, SYSTEM, LOGGING, WORKER

# Exclude events and debug messages
textual console -x EVENT -x DEBUG

# Show only warnings, errors, and print statements
textual console -x SYSTEM -x EVENT -x DEBUG -x INFO
```

#### Custom Port

Run console on a different port:

```bash
# Console on custom port
textual console --port 7342

# App connecting to custom port
textual run --dev --port 7342 my_app.py
```

### Log Message Format

Console output includes:

```
[TIMESTAMP] [GROUP] Message content
```

**Example:**
```
[12:34:56] [EVENT] Key pressed: 'q'
[12:34:57] [INFO] Processing request
[12:34:58] [PRINT] User print statement
[12:34:59] [DEBUG] Variable state: {...}
```

---

## Live CSS Editing

### Overview

One of the most powerful features of development mode: edit CSS files and see changes **instantly** in your running app (within milliseconds).

### Workflow

**Step 1: Run app in dev mode**
```bash
textual run --dev my_app.py
```

**Step 2: Open CSS file in editor**
```python
# my_app.py
from textual.app import App

class MyApp(App):
    CSS_PATH = "my_app.tcss"  # External CSS file
```

**Step 3: Edit and save CSS**
```css
/* my_app.tcss */
Button {
    background: blue;  /* Change this... */
}
```

```css
/* Save with changes */
Button {
    background: red;  /* ...see instant update! */
}
```

The app updates **immediately** after saving - no restart required.

### Best Practices

- Keep CSS in external `.tcss` or `.css` files (not inline strings)
- Use dev mode exclusively during styling iteration
- Open terminal and editor side-by-side for instant visual feedback
- Test responsive behavior by resizing terminal while editing

### Supported CSS Properties

All Textual CSS properties support live editing:
- Colors (background, color, border colors)
- Dimensions (width, height, padding, margin)
- Layout (display, align, grid, dock)
- Borders (border, outline, box-sizing)
- Text (text-align, text-style, opacity)
- Scrollbars (scrollbar colors, sizes)

---

## textual serve - Browser Deployment

### Overview

Serve your Textual app as a web application accessible via browser. Multiple users can connect simultaneously!

### Basic Usage

**Serve from Python file:**
```bash
textual serve my_app.py
```

**Serve from module:**
```bash
textual serve "python -m textual"
```

**Serve command:**
```bash
textual serve "textual keys"
```

### Server Options

```bash
# Custom host and port
textual serve -h 0.0.0.0 -p 8080 my_app.py

# Set app title (shown in browser)
textual serve -t "My Amazing App" my_app.py

# Public URL (for sharing)
textual serve -u "https://myapp.example.com" my_app.py

# Enable debug mode (with devtools)
textual serve --dev my_app.py
```

### Complete Example

```bash
# Serve publicly accessible app with debug mode
textual serve \
    --host 0.0.0.0 \
    --port 8000 \
    --title "Task Manager" \
    --dev \
    my_app.py
```

Access at: `http://localhost:8000`

### Use Cases

- **Development**: Refresh browser to reload app (faster than restarting terminal)
- **Demos**: Share apps with team members via URL
- **Testing**: Test on different devices/browsers
- **Deployment**: Serve production apps to users without terminal access

---

## Built-in Preview Tools

### textual borders

Explore all border styles available in Textual:

```bash
textual borders
```

**Shows:**
- All border types (solid, dashed, double, heavy, etc.)
- Border combinations
- Interactive preview with keyboard navigation

### textual easing

Preview animation easing functions:

```bash
textual easing
```

**Shows:**
- All easing functions (linear, ease-in, ease-out, ease-in-out)
- Elastic, bounce, back easing
- Visual curves and animations
- Useful for selecting animation styles

### textual colors

Explore the Textual design system:

```bash
textual colors
```

**Shows:**
- Color palette (all named colors)
- Theme colors
- Color gradients
- RGB/Hex values
- Color accessibility information

### textual keys

Show key events in real-time:

```bash
textual keys
```

**Shows:**
- Key presses as you type
- Key codes and names
- Modifier keys (ctrl, shift, alt)
- Useful for debugging key bindings

---

## textual diagnose

Print diagnostic information about your Textual environment:

```bash
textual diagnose
```

**Output includes:**
- Python version
- Textual version
- Terminal information
- Operating system
- Environment variables
- Dependency versions

**Use cases:**
- Bug reports
- Environment troubleshooting
- Checking installation

---

## Development Workflow Examples

### Workflow 1: Standard Development

```bash
# Terminal 1: Start console
textual console

# Terminal 2: Run app in dev mode
textual run --dev my_app.py

# Edit code and CSS - see changes on save
# Use print() and log() for debugging
# Check console for errors/logs
```

### Workflow 2: Rapid CSS Iteration

```bash
# Terminal 1: Run app in dev mode (no console needed)
textual run --dev my_app.py

# Terminal 2: Edit CSS file
vim my_app.tcss

# Save frequently - see instant visual feedback
# Adjust colors, spacing, layout in real-time
```

### Workflow 3: Browser Testing

```bash
# Start server with dev mode
textual serve --dev --host 0.0.0.0 --port 8000 my_app.py

# Open browser to http://localhost:8000
# Make code changes
# Refresh browser to see updates
# Check console for logs
```

### Workflow 4: Screenshot Documentation

```bash
# Take screenshot after 3 seconds
textual run --screenshot 3 --screenshot-path ./docs/images my_app.py

# Simulate user interaction before screenshot
textual run --press "tab,tab,enter" --screenshot 2 my_app.py
```

### Workflow 5: Testing Key Bindings

```bash
# Terminal 1: Run keys preview
textual keys

# Terminal 2: Press keys to test
# See exact key codes for binding implementation
```

---

## Architecture Notes

### Console Communication

**Protocol**: WebSocket connection over aiohttp
**Default Port**: 8081 (customizable with `--port`)
**Server**: Async aiohttp application with WebSocketResponse
**Message Format**: msgpack serialization

### Live CSS Reloading

**Mechanism**: File system watching (automatic)
**Latency**: <100ms typical
**Scope**: Affects all mounted widgets with updated styles
**Limitations**: Inline CSS strings (not external files) don't live-reload

### Server Architecture

**Backend**: textual-serve package
**Multiple Clients**: Supported (each gets own app instance)
**WebSocket Updates**: Real-time terminal rendering to browser
**Input Handling**: Keyboard/mouse events from browser to app

---

## Common Issues and Solutions

### Issue: Console won't connect

**Symptom**: `textual console` running but app doesn't log to it

**Solutions:**
```bash
# Ensure using --dev flag
textual run --dev my_app.py

# Check firewall (must allow localhost:8081)
# Try custom port
textual console --port 7342
textual run --dev --port 7342 my_app.py

# Check for port conflicts
lsof -i :8081  # See what's using port 8081
```

### Issue: "Couldn't start server"

**Symptom**: Error when running `textual console`

**Cause**: Another console instance already running

**Solution:**
```bash
# Find and kill existing console
ps aux | grep "textual console"
kill <PID>

# Or use different port
textual console --port 7342
```

### Issue: Live CSS editing not working

**Symptom**: CSS changes don't appear in running app

**Checks:**
- Ensure using `--dev` flag: `textual run --dev my_app.py`
- CSS must be in external file (not inline string)
- File must be saved (not just edited)
- Check CSS_PATH is correct in app class

### Issue: macOS Terminal colors limited

**Warning shown**: "limited to 256 colors"

**Solution**: Use a modern terminal emulator:
- iTerm2 (recommended)
- Ghostty
- Kitty
- WezTerm

Reference: https://textual.textualize.io/FAQ/#why-doesnt-textual-look-good-on-macos

---

## Advanced Tips

### 1. Conditional Development Features

Enable features only in dev mode:

```python
from textual.app import App
import os

class MyApp(App):
    def on_mount(self):
        if "devtools" in os.environ.get("TEXTUAL", ""):
            self.log("Development mode active")
            # Enable extra logging, debug widgets, etc.
```

### 2. Custom Log Formatting

```python
from textual import log

def debug_widget(widget):
    """Custom debug function for widgets"""
    log(
        f"Widget: {widget.__class__.__name__}",
        id=widget.id,
        classes=widget.classes,
        size=widget.size,
        styles=widget.styles,
    )

# Usage
debug_widget(self.query_one("#my-button"))
```

### 3. Performance Monitoring

```python
import time
from textual.app import App

class MyApp(App):
    def on_mount(self):
        start = time.time()
        # ... expensive operation ...
        elapsed = time.time() - start
        self.log(f"Mount took {elapsed:.3f}s")
```

### 4. Screenshot Automation

```bash
#!/bin/bash
# Generate screenshots for all examples

for app in examples/*.py; do
    filename=$(basename "$app" .py)
    textual run --screenshot 2 \
                --screenshot-path ./docs/screenshots \
                --screenshot-filename "$filename.svg" \
                "$app"
done
```

### 5. Integration with pytest

```python
# test_app.py
from textual.pilot import Pilot

async def test_button_click():
    from my_app import MyApp

    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.click("#submit-button")
        # No need for textual run - app.run_test() handles it
```

---

## Sources

**Primary Documentation:**
- [Textual Devtools Guide](https://textual.textualize.io/guide/devtools/) (accessed 2025-11-02)
- [Textual Getting Started](https://textual.textualize.io/getting_started/) (accessed 2025-11-02)

**Source Code:**
- [textual-dev GitHub Repository](https://github.com/Textualize/textual-dev) (accessed 2025-11-02)
- [textual-dev/cli.py](https://github.com/Textualize/textual-dev/blob/main/src/textual_dev/cli.py) - Command-line interface implementation
- [textual-dev/server.py](https://github.com/Textualize/textual-dev/blob/main/src/textual_dev/server.py) - Console server implementation
- [textual-dev/pyproject.toml](https://github.com/Textualize/textual-dev/blob/main/pyproject.toml) - Package configuration

**Additional Resources:**
- [YouTube: How to install and use the Textual devtools console](https://www.youtube.com/watch?v=2w1hJPzQCJY) - Official tutorial (Nov 2023)
- [YouTube: How to log messages to the devtools console](https://www.youtube.com/watch?v=b2HRbz3dgxM) - Logging tutorial (Dec 2023)
- [Mouse Vs Python: How to Debug Your Textual Application](https://www.blog.pythonlibrary.org/2024/11/19/how-to-debug-your-textual-application/) - Community guide (Nov 2024)

---

## Quick Reference Card

```bash
# CONSOLE DEBUGGING
textual console                          # Start debug console
textual console -v                       # Verbose mode
textual console -x EVENT -x DEBUG        # Exclude log groups
textual console --port 7342              # Custom port

# RUN APPLICATIONS
textual run app.py                       # Run Python file
textual run module.app                   # Run from module
textual run module.app:MyApp             # Specific app class
textual run --dev app.py                 # With debugging
textual run -c "textual colors"          # Run command

# DEVELOPMENT OPTIONS
textual run --dev --port 7342 app.py     # Custom console port
textual run --press "tab,enter" app.py   # Simulate keys
textual run --screenshot 5 app.py        # Screenshot after 5s
textual run --show-return app.py         # Show exit value

# SERVE IN BROWSER
textual serve app.py                     # Default (localhost:8000)
textual serve -h 0.0.0.0 -p 8080 app.py  # Custom host/port
textual serve --dev -t "My App" app.py   # With debug + title

# PREVIEW TOOLS
textual borders                          # Border style explorer
textual easing                           # Animation easing preview
textual colors                           # Color palette explorer
textual keys                             # Key event viewer
textual diagnose                         # Environment info
```

**Total Lines**: 429 lines of comprehensive development tools documentation.
