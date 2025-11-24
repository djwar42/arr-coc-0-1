# Textual: The Definitive Guide - Part 1

**Source**: [Textual: The Definitive Guide - Part 1](https://dev.to/wiseai/textual-the-definitive-guide-part-1-1i0p)
**Author**: Mahmoud Harmouch (@wiseai)
**Published**: April 11, 2022 (Edited April 23, 2023)
**Accessed**: 2025-11-02
**Series**: Part 1 of 3-part series on Textual

**⚠️ Important Note**: This tutorial is from 2022. As of 2024, Textual has evolved significantly and many examples here are **outdated**. For current documentation, see the [official Textual tutorial](https://textual.textualize.io/tutorial/). This file is preserved for historical context and foundational concepts.

---

## Overview

This article introduces Textual, a Python framework for building text-based user interfaces (TUIs). It covers installation, basic concepts, widgets, views, and event handling through the development of a Wordle clone example.

## What is a TUI?

From Wikipedia:
> In computing, text-based user interfaces (TUI) (alternately terminal user interfaces, to reflect a dependence upon the properties of computer terminals and not just text), is a retronym describing a type of user interface (UI) common as an early form of human–computer interaction, before the advent of graphical user interfaces (GUIs).

**Key characteristics**:
- Relies on text rather than graphics
- Typically command-line based
- May lack mouse support (though Textual supports it)
- Designed for keyboard input

**Example**: Linux `cat` command provides text-based interface for displaying file contents.

## What is Textual?

A text-based user interface toolkit for building sophisticated terminal applications.

**Core features**:
- Elegant API for building terminal UIs
- Highly interactive and complex application support
- Removes boilerplate from earlier TUI frameworks
- No decorators needed for event handling

**Target audience**:
- Beginner to intermediate Python programmers
- Those interested in advanced Python concepts
- Developers wanting to learn programming workflow
- Anyone building terminal applications

## Installation

### Basic Installation

```bash
pip install textual
```

### With Poetry (Recommended)

The article provides extensive coverage of Poetry for dependency management.

#### Poetry Installation

```bash
# Linux installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

#### Python Version Management with pyenv

```bash
# Configure pyenv on zsh
cat << EOF >> ~/.zshrc
# pyenv config
export PATH="${HOME}/.pyenv/bin:${PATH}"
export PYENV_ROOT="${HOME}/.pyenv"
eval "$(pyenv init -)"
EOF

# Install specific Python version
pyenv install 3.10.1

# Set global version
pyenv global system 3.10.1
```

#### Create Project with Poetry

```bash
# Create new project
poetry new deepwordle && cd deepwordle

# Create virtual environment with specific Python version
poetry env use 3.10.1

# Activate virtual environment
poetry shell

# Add Textual to project
poetry add textual
```

**Note**: Poetry creates virtual environments under `~/.cache/pypoetry/virtualenvs/` by default. To create `.venv` in project directory:

```bash
poetry config virtualenvs.in-project true
```

## Basic Textual App

### Minimal Example

```python
from textual.app import App

App.run()
```

This creates a blank window with black background. Press Ctrl+C to exit.

### Basic App Structure

```python
from textual.app import App

class MainApp(App):
    ...

MainApp.run()
```

Using inheritance creates a subclass of `App` that can be extended with custom functionality.

## User Interface Design

**Before coding, design the UI mockup** with these requirements:

1. Render each guessed word on screen with spacing
2. Words are five letters long (Wordle standard)
3. Allow six attempts maximum
4. Display formatted messages about game state
5. Show available keyboard keys

**Required components**:
- 6x5 grid view for letter tiles
- Button for each letter
- Message panel
- Header and footer

## Textual Widgets

Widgets are the base visual components in Textual. They contain a Canvas for terminal drawing and handle events.

### Widget Fundamentals

**Reactive attributes**: Widgets have 12 reactive attributes for visual properties (style, border, padding, size, etc.)

**Watchers**: Each reactive attribute can have a watcher method:

```python
foo = Reactive("")

def watch_foo(self, val):
    if val == "bar":
        do_something()
    # custom logic
```

### Placeholder

Used for prototyping layouts before implementation:

```python
from textual.app import App
from textual.widgets import Placeholder


class MainApp(App):

    async def on_mount(self) -> None:
        await self.view.dock(Placeholder(name="header"), edge="top", size=3)
        await self.view.dock(Placeholder(name="footer"), edge="bottom", size=3)
        await self.view.dock(Placeholder(name="stats"), edge="left", size=40)
        await self.view.dock(Placeholder(name="message"), edge="right", size=40)
        await self.view.dock(Placeholder(name="grid"), edge="top")

MainApp.run()
```

### Button

A Label with click event handling.

**Properties**:
- `label`: Text rendered on button
- `name`: Widget identifier
- `style`: Label style in 'foreground on background' notation

```python
from textual.app import App
from textual.widgets import Button


class MainApp(App):

    async def on_mount(self) -> None:
        button1 = Button(label='Hello', name='button1')
        button2 = Button(label='world', name='button2', style='black on white')
        await self.view.dock(button1, button2, edge="left")

MainApp.run()
```

### Header

Displays app title, time, and icon:

```python
from textual.app import App
from textual.widgets import Header


class MainApp(App):

    async def on_mount(self) -> None:
        header = Header(tall=False)
        await self.view.dock(header)

MainApp.run(title="DeepWordle")
```

### Footer

Shows available keyboard shortcuts:

```python
from textual.app import App
from textual.widgets import Footer


class MainApp(App):

    async def on_load(self) -> None:
        """Bind keys here."""
        await self.bind("q", "quit", "Quit")
        await self.bind("t", "tweet", "Tweet")
        await self.bind("r", "None", "Record")

    async def on_mount(self) -> None:
        footer = Footer()
        await self.view.dock(footer, edge="bottom")

MainApp.run(title="DeepWordle")
```

### ScrollView

Container with scrolling support:

```python
from textual.app import App
from textual.widgets import ScrollView, Button


class MainApp(App):

    async def on_mount(self) -> None:
        scroll_view = ScrollView(contents=Button(label='button'), auto_width=True)
        await self.view.dock(scroll_view)

MainApp.run()
```

### Static

Simple static content display:

```python
from textual.app import App
from textual.widgets import Static, Button


class MainApp(App):

    async def on_mount(self) -> None:
        static = Static(renderable=Button(label='button'), name='')
        await self.view.dock(static)

MainApp.run()
```

### Custom Widgets

Create custom widgets by extending the `Widget` class:

```python
from textual.app import App
from textual.widget import Widget
from textual.reactive import Reactive
from rich.console import RenderableType
from rich.padding import Padding
from rich.align import Align
from rich.text import Text

class Letter(Widget):

    label = Reactive("")

    def render(self) -> RenderableType:
        return Padding(
            Align.center(Text(text=self.label), vertical="middle"),
            (0, 1),
            style="white on rgb(51,51,51)",
        )

class MainApp(App):

    async def on_mount(self) -> None:
        letter = Letter()
        letter.label = "A"
        await self.view.dock(letter)

MainApp.run(title="DeepWordle")
```

**Additional widgets available**: `TreeClick`, `TreeControl`, `TreeNode`, `NodeID`, `ButtonPressed`, `DirectoryTree`, `FileClick`

## Reusable Components

**Best practice**: Organize components in separate files for better structure and reusability.

```
deepwordle/
├── __init__.py
├── app.py
└── components/
    ├── __init__.py
    ├── constants.py
    ├── letter.py
    ├── letters_grid.py
    ├── message.py
    ├── rich_text.py
    └── utils.py
```

## Views: Organizing Widgets

Views arrange widgets on the terminal using a docking technique (similar to CSS grid).

**Default**: Widgets render at center of terminal.

**Docking edges**: `left`, `right`, `top`, `bottom`

### DockView

Default view that groups widgets vertically (default) or horizontally.

**Horizontal layout** (default edge="top"):

```python
from textual.app import App
from textual.widgets import Placeholder
from textual.views import DockView


class SimpleApp(App):

    async def on_mount(self) -> None:
        view: DockView = await self.push_view(DockView())
        await view.dock(Placeholder(), Placeholder(), Placeholder())

SimpleApp.run()
```

**Vertical layout** (edge="left"):

```python
from textual.app import App
from textual.widgets import Placeholder
from textual.views import DockView


class SimpleApp(App):

    async def on_mount(self) -> None:
        view: DockView = await self.push_view(DockView())
        await view.dock(Placeholder(), Placeholder(), Placeholder(), edge='left')

SimpleApp.run()
```

**Fixed size** (in characters):

```python
from textual.app import App
from textual.widgets import Placeholder
from textual.views import DockView


class SimpleApp(App):

    async def on_mount(self) -> None:
        view: DockView = await self.push_view(DockView())
        await view.dock(Placeholder(), Placeholder(), Placeholder(), size=10)

SimpleApp.run()
```

### GridView

Arranges widgets in rectangular/tabular layout.

**Empty grid**:

```python
from textual.app import App
from textual.widgets import Placeholder
from textual.views import GridView


class SimpleApp(App):

    async def on_mount(self) -> None:
        await self.view.dock(GridView(), size=10)
        await self.view.dock(Placeholder(name='sad'), size=10)
        await self.view.dock(GridView(), size=10)

SimpleApp.run(log="textual.log")
```

**6x6 grid with placeholders**:

```python
from textual.app import App
from textual import events
from textual.widgets import Placeholder


class GridView(App):
    async def on_mount(self, event: events.Mount) -> None:
        """Create a grid with auto-arranging cells."""

        grid = await self.view.dock_grid()

        grid.add_column("col", repeat=6, size=7)
        grid.add_row("row", repeat=6, size=7)
        grid.set_align("stretch", "center")

        placeholders = [Placeholder() for _ in range(36)]
        grid.place(*placeholders)


GridView.run(title="Grid View", log="textual.log")
```

### WindowView

Container for single widget:

```python
from textual.app import App
from textual.widgets import Placeholder
from textual.views import WindowView


class SimpleApp(App):

    async def on_mount(self) -> None:
        await self.view.dock(WindowView(widget=Placeholder(name='sad')), size=10)
        await self.view.dock(WindowView(widget=Placeholder(name='sad')), size=10)
        await self.view.dock(Placeholder(name='sad'), size=10)

SimpleApp.run(log="textual.log")
```

## Widget Event Handlers

Textual uses underscore naming convention instead of decorators.

**Key event handler example**:

```python
def on_key(self, event):
    ...
```

**Traditional decorator approach** (NOT used in Textual):

```python
@on(event)
def key(self):
    ...
```

**Rationale**: The underscore convention reduces boilerplate and improves readability, though it may be less "Pythonic" to some developers.

## Architecture Concepts

**Widget class**:
- Base visual component
- Contains Canvas for terminal drawing
- Receives and reacts to events
- Can contain other widgets (nested containers)

**Reactive attributes**:
- Implemented as Python descriptors
- Allow dynamic property updates
- Trigger automatic re-rendering

**12 reactive attributes available**:
- Style, border, padding, size, and more
- Each can have associated watchers

## Summary

This tutorial covered:

1. **TUI fundamentals**: What text user interfaces are and their characteristics
2. **Textual introduction**: Modern TUI framework with elegant API
3. **Development environment**: Poetry and pyenv for dependency management
4. **Basic applications**: From minimal app to structured projects
5. **Widgets**: Built-in components (Placeholder, Button, Header, Footer, ScrollView, Static) and custom widgets
6. **Views**: DockView, GridView, and WindowView for layout
7. **Event handling**: Underscore naming convention for handlers
8. **Best practices**: Component organization and reusability

## Series Navigation

This is **Part 1** of a 3-part series:

- **Part 1**: Introduction, installation, widgets, and views (this document)
- [Part 2](https://dev.to/wiseai/textual-the-definitive-guide-part-2-6h8): Additional concepts
- [Part 3](https://dev.to/wiseai/textual-the-definitive-guide-part-3-2gl): Final topics

## Additional Resources

- [Textual GitHub Repository](https://github.com/Textualize/textual)
- [Textual Official Documentation](https://textual.textualize.io/) (current)
- [Textual Official Tutorial](https://textual.textualize.io/tutorial/) (recommended for 2024+)

---

**Historical Note**: While this tutorial provides excellent foundational concepts, readers should consult the official Textual documentation for current API patterns and best practices as of 2024 and beyond.
