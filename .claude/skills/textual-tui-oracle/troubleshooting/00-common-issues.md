# Common Issues and Solutions

**Extract from**: [source-documents/01-faq.md](../source-documents/01-faq.md)

## Installation Issues

### ImportError: cannot import name ComposeResult from textual.app

**Cause**: Outdated Textual version

**Solution**:
```bash
pip install textual-dev -U
```

The `-U` flag forces pip to upgrade to the latest version.

---

## Display Issues

### Textual doesn't look good on macOS

**Problem**: Default Terminal.app renders box characters poorly

**Solution 1 - Fix Terminal.app**:
1. Open Settings → Profiles → Text tab
2. Set font: **Menlo Regular**
3. Character spacing: **1**
4. Line spacing: **0.805**

**Solution 2 - Use Better Terminal** (Recommended):
- **iTerm2**: https://iterm2.com/
- **Kitty**: https://sw.kovidgoyal.net/kitty/
- **WezTerm**: https://wezfurlong.org/wezterm/

These terminals support:
- Full color range (16.7M colors)
- Better performance
- Proper box character rendering

### Translucent app background not working

**Cause**: Textual uses 16.7 million colors, not ANSI colors

**Explanation**: Terminal translucency requires ANSI background colors, which Textual doesn't use (by design for consistency across platforms).

**Workaround**: As of Textual 0.80.0, you can set `ansi_color=True` in App constructor, but you'll lose transparency effects.

### Why doesn't Textual support ANSI themes?

**By design**. Reasons:
1. ANSI themes vary widely - what looks good on your system may be unreadable on another
2. Textual needs color manipulation (blending, shades) for readability
3. Guarantees consistent appearance across all platforms
4. Enables future accessibility features

---

## Widget Issues

### How do I center a widget?

Use `align` on the **parent** container, not the widget itself.

**Example - Center single widget:**
```python
from textual.app import App, ComposeResult
from textual.widgets import Button

class ButtonApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Button("PUSH ME!")
```

**Example - Center multiple widgets individually:**
```python
from textual.containers import Center

class ButtonApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Center(Button("PUSH ME!"))
        yield Center(Button("AND ME!"))
        yield Center(Button("ALSO ME!"))
```

See: [How-To Center Things](https://textual.textualize.io/how-to/center-things/)

### How do I select and copy text?

**Method 1 - Built-in** (most widgets):
- Click and drag to select
- Press Ctrl+C to copy

**Method 2 - Terminal modifier**:
For widgets without text selection:
- **iTerm**: Hold OPTION key
- **Gnome Terminal**: Hold SHIFT key
- **Windows Terminal**: Hold SHIFT key

Then click and drag as usual.

---

## Event Issues

### WorkerDeclarationError when using @work decorator

**Cause**: Textual 0.31.0+ requires explicit `thread=True` for threaded workers

**Solution for threaded worker:**
```python
@work(thread=True)
def run_in_background():
    # Runs in separate thread
    ...
```

**Solution for async worker** (not threaded):
```python
@work()
async def run_in_background():
    # Runs as coroutine
    ...
```

### Some key combinations never reach my app

**Cause**: Terminal limitations - not all key combos are passed through

**Universal keys** (work everywhere):
- Letters, numbers
- F1-F10 function keys
- Space, Return
- Arrow keys, Home, End, Page Up/Down
- Ctrl, Shift

**Keys that DON'T work** (terminal blocks them):
- Cmd and Option (macOS)
- Windows key (Windows)

**Test keys**: Run `textual keys` to see what your terminal supports

**Recommendation**: Stick to universal keys for bindings

---

## App Issues

### How do I pass arguments to an app?

Override `__init__` as normal:

```python
from textual.app import App, ComposeResult
from textual.widgets import Static

class Greetings(App[None]):
    def __init__(self, greeting: str="Hello", to_greet: str="World") -> None:
        self.greeting = greeting
        self.to_greet = to_greet
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(f"{self.greeting}, {self.to_greet}")

# Usage:
Greetings().run()                      # Default arguments
Greetings(to_greet="davep").run()      # Keyword argument
Greetings("Well hello", "there").run() # Positional arguments
```

---

## Feature Requests

### Does Textual support images?

Not yet built-in, but it's on the [Roadmap](https://textual.textualize.io/roadmap/).

**Workaround**: Use [rich-pixels](https://github.com/darrenburns/rich-pixels) project for Rich renderable images that work with Textual.

---

## Getting Help

If you can't find your issue here:
- **Discord**: https://discord.gg/Enf6Z3qhVr
- **GitHub Issues**: https://github.com/Textualize/textual/issues
- **Documentation**: https://textual.textualize.io/help/

---

**Source**: [source-documents/01-faq.md](../source-documents/01-faq.md)
