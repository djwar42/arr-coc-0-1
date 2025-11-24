# Textual FAQ

**Source**: https://textual.textualize.io/FAQ/

Frequently Asked Questions about Textual - the TUI framework for Python.

## Does Textual support images?

Textual doesn't have built-in support for images yet, but it is on the Roadmap.

See also the [rich-pixels](https://github.com/darrenburns/rich-pixels) project for a Rich renderable for images that works with Textual.

## How can I fix ImportError cannot import name ComposeResult from textual.app?

You likely have an older version of Textual. You can install the latest version by adding the `-U` switch which will force pip to upgrade.

```bash
pip install textual-dev -U
```

## How can I select and copy text in a Textual app?

Textual supports text selection for most widgets, via click and drag. Press ctrl+c to copy.

For widgets that don't yet support text selection, you can try and use your terminal's builtin support. Most terminal emulators offer a modifier key which you can hold while you click and drag to restore the behavior you may expect from the command line. The exact modifier key depends on the terminal and platform you are running on.

- **iTerm**: Hold the OPTION key
- **Gnome Terminal**: Hold the SHIFT key
- **Windows Terminal**: Hold the SHIFT key

Refer to the documentation for your terminal emulator, if it is not listed above.

## How can I set a translucent app background?

Some terminal emulators have a translucent background feature which allows the desktop underneath to be partially visible.

This feature is unlikely to work with Textual, as the translucency effect requires the use of ANSI background colors, which Textual doesn't use. Textual uses 16.7 million colors where available which enables consistent colors across all platforms and additional effects which aren't possible with ANSI colors.

## How do I center a widget in a screen?

To center a widget within a container use `align`. But remember that `align` works on the _children_ of a container, it isn't something you use on the child you want centered.

Example - Button in the middle of a Screen:

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

if __name__ == "__main__":
    ButtonApp().run()
```

For multiple widgets that should each be centered individually, wrap each widget in a `Center` container:

```python
from textual.app import App, ComposeResult
from textual.containers import Center
from textual.widgets import Button

class ButtonApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Center(Button("PUSH ME!"))
        yield Center(Button("AND ME!"))
        yield Center(Button("ALSO PLEASE PUSH ME!"))
        yield Center(Button("HEY ME ALSO!!"))

if __name__ == "__main__":
    ButtonApp().run()
```

## How do I fix WorkerDeclarationError?

Textual version 0.31.0 requires that you set `thread=True` on the `@work` decorator if you want to run a threaded worker.

For a threaded worker:
```python
@work(thread=True)
def run_in_background():
    ...
```

For an async worker (not threaded):
```python
@work()
async def run_in_background():
    ...
```

## How do I pass arguments to an app?

When creating your `App` class, override `__init__` as you would when inheriting normally:

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

# Usage examples:
Greetings().run()  # Default arguments
Greetings(to_greet="davep").run()  # Keyword argument
Greetings("Well hello", "there").run()  # Positional arguments
```

## Why do some key combinations never make it to my app?

Textual can only ever support key combinations that are passed on by your terminal application. Which keys get passed on can differ from terminal to terminal, and from operating system to operating system.

**Universally-supported keys:**
- Letters
- Numbers
- Numbered function keys (especially F1 through F10)
- Space
- Return
- Arrow, home, end and page keys
- Control
- Shift

**Keys not normally passed through:**
- Cmd and Option on macOS
- Windows key on Windows

Test key combinations with `textual keys`.

## Why doesn't Textual look good on macOS?

The default macOS Terminal.app doesn't render Textual apps very well, particularly when it comes to box characters.

**Fix for Terminal.app:**
- Open settings → profiles → Text tab
- Use Menlo Regular font
- Character spacing: 1
- Line spacing: 0.805

**Better alternatives:**
- [iTerm2](https://iterm2.com/)
- [Kitty](https://sw.kovidgoyal.net/kitty/)
- [WezTerm](https://wezfurlong.org/wezterm/)

## Why doesn't Textual support ANSI themes?

Textual will not generate escape sequences for the 16 themeable ANSI colors.

**Reasons:**
1. Not everyone has a carefully chosen ANSI color theme - colors may be unreadable on other machines
2. ANSI colors can't be manipulated like Textual's 16.7 million colors
3. Textual can blend colors and produce light/dark shades for better readability
4. Color blending powers future accessibility features

Textual has a design system which guarantees apps will be readable on all platforms and terminals.

**Note**: As of version 0.80.0, you can set `ansi_color = True` in your App to prevent ANSI color conversion (but you lose transparency effects).
