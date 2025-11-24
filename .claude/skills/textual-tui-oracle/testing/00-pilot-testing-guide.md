# Textual Pilot Testing Guide

## Overview

Pilot is Textual's built-in testing API that allows you to programmatically operate your app in headless mode. It simulates user interactions like keyboard input and mouse clicks, enabling automated testing of your TUI applications.

From [Official Testing Guide](https://textual.textualize.io/guide/testing/) (accessed 2025-11-02)

## Core Concept: The Pilot Object

The `Pilot` object is returned by `app.run_test()` and provides methods to interact with your app as if you were a real user.

### Basic Test Structure

```python
from textual.app import App

async def test_my_app():
    """All Pilot tests must be async."""
    app = MyApp()
    async with app.run_test() as pilot:  # Returns Pilot instance
        # Test interactions go here
        await pilot.press("enter")
        assert app.some_state == expected_value
```

**Key Points:**
- Tests must be `async` functions
- Use `app.run_test()` instead of `app.run()` for headless execution
- `run_test()` is an async context manager returning a Pilot
- Default terminal size: (80, 24) - can be customized

From [textual.pilot API](https://textual.textualize.io/api/pilot/) (accessed 2025-11-02)

## Testing Frameworks

While Textual doesn't require a specific test framework, it must support asyncio. Recommended setup:

**pytest + pytest-asyncio:**
```bash
pip install pytest pytest-asyncio
```

**Configuration (pytest.ini or pyproject.toml):**
```ini
[tool.pytest.ini_options]
asyncio_mode = "auto"  # No need for @pytest.mark.asyncio on every test
```

Or run with: `pytest --asyncio-mode=auto`

From [Official Testing Guide - Testing Frameworks](https://textual.textualize.io/guide/testing/#testing-frameworks-for-textual)

## Simulating Key Presses

### Press Method

```python
await pilot.press(*keys)
```

**Single key:**
```python
await pilot.press("r")  # Press 'r' key
```

**Multiple keys (typing):**
```python
await pilot.press("h", "e", "l", "l", "o")  # Type "hello"
```

**Special keys:**
```python
await pilot.press("enter")
await pilot.press("escape")
await pilot.press("tab")
await pilot.press("up", "down", "left", "right")
```

**Key combinations:**
```python
await pilot.press("ctrl+c")
await pilot.press("ctrl+s")
```

**Discover key identifiers:**
```bash
textual keys  # Interactive tool to see key names
```

From [Official Testing Guide - Simulating Key Presses](https://textual.textualize.io/guide/testing/#simulating-key-presses)

## Simulating Mouse Clicks

### Click Method

```python
await pilot.click(
    widget=None,      # Widget/selector/type to click
    offset=(0, 0),    # Offset from widget or screen
    shift=False,      # Modifier keys
    meta=False,
    control=False,
    times=1          # Number of clicks (2=double, 3=triple)
)
```

### Click Examples

**Click by selector:**
```python
# Click button with ID "submit"
await pilot.click("#submit")

# Click button by class
await pilot.click(".danger-button")

# Click by widget type
await pilot.click(Button)
```

**Click at screen coordinates:**
```python
# Click at (0, 0) on screen
await pilot.click()

# Click at (10, 5) on screen
await pilot.click(offset=(10, 5))
```

**Click with offset from widget:**
```python
# Click 8 cells right, 1 cell down from button's top-left
await pilot.click(Button, offset=(8, 1))

# Click one line ABOVE a widget
await pilot.click("#my-widget", offset=(0, -1))
```

**Modifier keys:**
```python
# Ctrl+click
await pilot.click("#item", control=True)

# Shift+click
await pilot.click("#item", shift=True)

# Meta+click (Cmd on Mac)
await pilot.click("#item", meta=True)
```

**Multiple clicks:**
```python
# Double-click
await pilot.click(Button, times=2)
# Or use convenience method:
await pilot.double_click(Button)

# Triple-click
await pilot.click(Button, times=3)
# Or use convenience method:
await pilot.triple_click(Button)
```

**Return value:**
- `True` if no selector specified OR selected widget was under mouse
- `False` if selected widget was NOT under the pointer (obscured/hidden)

From [Official Testing Guide - Simulating Clicks](https://textual.textualize.io/guide/testing/#simulating-clicks)

## Other Mouse Events

### Hover

```python
await pilot.hover(widget=None, offset=(0, 0))
```

Simulates moving mouse cursor to a position without clicking.

### Mouse Down/Up

```python
await pilot.mouse_down(widget=None, offset=(0, 0), shift=False, meta=False, control=False)
await pilot.mouse_up(widget=None, offset=(0, 0), shift=False, meta=False, control=False)
```

Simulate individual mouse button press/release events (for drag operations, etc).

From [textual.pilot API - Mouse Methods](https://textual.textualize.io/api/pilot/)

## Changing Screen Size

### At Test Start

```python
async with app.run_test(size=(100, 50)) as pilot:
    # App starts with 100 columns, 50 rows
    ...
```

### During Test Execution

```python
await pilot.resize_terminal(120, 60)
```

From [Official Testing Guide - Changing Screen Size](https://textual.textualize.io/guide/testing/#changing-the-screen-size)

## Pausing and Waiting

### Pause for Messages

```python
await pilot.pause()
```

**Use cases:**
- Wait for all pending messages to be processed
- Solve race conditions when state changes aren't immediate
- Ensure message bubbling completes

**When to use:**
After posting messages or actions that don't update state synchronously.

### Pause with Delay

```python
await pilot.pause(delay=0.1)  # Wait 0.1 seconds, then process messages
```

Inserts a delay before waiting for message processing.

### Wait for Animations

```python
# Wait for currently running animations
await pilot.wait_for_animation()

# Wait for current AND scheduled animations
await pilot.wait_for_scheduled_animations()
```

From [Official Testing Guide - Pausing](https://textual.textualize.io/guide/testing/#pausing-the-pilot)

## Complete Testing Example

```python
# rgb.py - Simple color-switching app
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer

class RGBApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    Horizontal {
        width: auto;
        height: auto;
    }
    """

    BINDINGS = [
        ("r", "switch_color('red')", "Go Red"),
        ("g", "switch_color('green')", "Go Green"),
        ("b", "switch_color('blue')", "Go Blue"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Red", id="red")
            yield Button("Green", id="green")
            yield Button("Blue", id="blue")
        yield Footer()

    @on(Button.Pressed)
    def pressed_button(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.action_switch_color(event.button.id)

    def action_switch_color(self, color: str) -> None:
        self.screen.styles.background = color

# test_rgb.py - Testing the app
from rgb import RGBApp
from textual.color import Color

async def test_keys():
    """Test pressing keys has the desired result."""
    app = RGBApp()
    async with app.run_test() as pilot:
        # Test pressing the R key
        await pilot.press("r")
        assert app.screen.styles.background == Color.parse("red")

        # Test pressing the G key
        await pilot.press("g")
        assert app.screen.styles.background == Color.parse("green")

        # Test pressing the B key
        await pilot.press("b")
        assert app.screen.styles.background == Color.parse("blue")

        # Test pressing unmapped key (no change)
        await pilot.press("x")
        assert app.screen.styles.background == Color.parse("blue")

async def test_buttons():
    """Test clicking buttons has the desired result."""
    app = RGBApp()
    async with app.run_test() as pilot:
        # Test clicking the "red" button
        await pilot.click("#red")
        assert app.screen.styles.background == Color.parse("red")

        # Test clicking the "green" button
        await pilot.click("#green")
        assert app.screen.styles.background == Color.parse("green")

        # Test clicking the "blue" button
        await pilot.click("#blue")
        assert app.screen.styles.background == Color.parse("blue")
```

From [Official Testing Guide - Testing Apps](https://textual.textualize.io/guide/testing/#testing-apps)

## Error Handling

### OutOfBounds Exception

Raised when click/hover target is outside visible screen:

```python
from textual.pilot import OutOfBounds

try:
    await pilot.click(offset=(1000, 1000))  # Way off screen
except OutOfBounds:
    # Handle out of bounds click
    pass
```

### WaitForScreenTimeout

Raised if messages aren't processed quickly enough (indicates potential deadlock):

```python
from textual.pilot import WaitForScreenTimeout
```

From [textual.pilot API - Exceptions](https://textual.textualize.io/api/pilot/)

## Best Practices

1. **Always use async tests** - Pilot requires coroutines
2. **Use pause() for race conditions** - Wait for messages after state changes
3. **Test user flows, not implementation** - Click buttons, press keys as users would
4. **Use specific selectors** - `#id` or `.class` for precise targeting
5. **Check return values** - `click()` returns bool indicating if target was hit
6. **Test different screen sizes** - Ensure layout works at various dimensions
7. **Combine with snapshot testing** - Visual regression testing (see next guide)

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_rgb.py

# Run with verbose output
pytest -v

# Run with asyncio mode auto
pytest --asyncio-mode=auto
```

## Advanced: Exit and Result

```python
async def test_app_exit():
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.exit(result="my_result")

    # After context manager, app has exited with result
```

From [textual.pilot API - Exit Method](https://textual.textualize.io/api/pilot/#textual.pilot.Pilot.exit)

## Sources

**Official Documentation:**
- [Testing Guide](https://textual.textualize.io/guide/testing/) - Textual official docs (accessed 2025-11-02)
- [textual.pilot API](https://textual.textualize.io/api/pilot/) - API reference (accessed 2025-11-02)

**Test Framework:**
- [pytest](https://docs.pytest.org/) - Testing framework
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) - Async support for pytest

**Related:**
- [Textual's Own Tests](https://github.com/Textualize/textual/tree/main/tests/) - Reference implementations
