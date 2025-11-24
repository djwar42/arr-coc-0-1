# Fedora Magazine Crash Course on Using Textual

**Source**: [Crash Course On Using Textual - Fedora Magazine](https://fedoramagazine.org/crash-course-on-using-textual/)
**Author**: Jose Nunez
**Date Accessed**: 2025-11-02
**Original Publication**: January 10, 2024

---

## Overview

Quick crash course on building Text User Interfaces (TUIs) in Python using Textual framework. Covers two practical examples: a log scroller and a race results table with detailed screens.

**What is Textual?**
> Textual is a Rapid Application Development framework for Python, built by Textualize.io. Build sophisticated user interfaces with a simple Python API. Run your apps in the terminal or a web browser!

---

## Prerequisites

**Requirements**:
1. Basic programming experience, preferable in Python
2. Understanding basic object oriented concepts like classes and inheritance
3. A machine with Linux and Python 3.9+ installed
4. A good editor (Vim or PyCharm are good choices)

---

## Installation

### Method 1: Virtual Environment + Git Clone

```bash
# Create virtual environment
python3 -m venv ~/virtualenv/Textualize

# Activate and install
. ~/virtualenv/Textualize/bin/activate
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade build
pip install --editable .
```

### Method 2: Install from PyPI

```bash
. ~/virtualenv/Textualize/bin/activate
pip install --upgrade KodegeekTextualize
```

---

## Example 1: Log Scroller Application

A simple application that executes a list of UNIX commands and captures their output.

### Core Application Code

```python
import shutil
from textual import on
from textual.app import ComposeResult, App
from textual.widgets import Footer, Header, Button, SelectionList
from textual.widgets.selection_list import Selection
from textual.screen import ModalScreen

# Operating system commands are hardcoded
OS_COMMANDS = {
    "LSHW": ["lshw", "-json", "-sanitize", "-notime", "-quiet"],
    "LSCPU": ["lscpu", "--all", "--extended", "--json"],
    "LSMEM": ["lsmem", "--json", "--all", "--output-all"],
    "NUMASTAT": ["numastat", "-z"]
}

class LogScreen(ModalScreen):
    # ... Code of the full separate screen omitted, will be explained next
    def __init__(self, name = None, ident = None, classes = None, selections = None):
        super().__init__(name, ident, classes)
        pass

class OsApp(App):
    BINDINGS = [
        ("q", "quit_app", "Quit"),
    ]
    CSS_PATH = "os_app.tcss"
    ENABLE_COMMAND_PALETTE = False  # Do not need the command palette

    def action_quit_app(self):
        self.exit(0)

    def compose(self) -> ComposeResult:
        # Create a list of commands, valid commands are assumed to be on the PATH variable.
        selections = [Selection(name.title(), ' '.join(cmd), True) for name, cmd in OS_COMMANDS.items() if shutil.which(cmd[0].strip())]
        yield Header(show_clock=False)
        sel_list = SelectionList(*selections, id='cmds')
        sel_list.tooltip = "Select one more more command to execute"
        yield sel_list
        yield Button(f"Execute {len(selections)} commands", id="exec", variant="primary")
        yield Footer()

    @on(SelectionList.SelectedChanged)
    def on_selection(self, event: SelectionList.SelectedChanged) -> None:
        button = self.query_one("#exec", Button)
        selections = len(event.selection_list.selected)
        if selections:
            button.disabled = False
        else:
            button.disabled = True
        button.label = f"Execute {selections} commands"

    @on(Button.Pressed)
    def on_button_click(self):
        selection_list = self.query_one('#cmds', SelectionList)
        selections = selection_list.selected
        log_screen = LogScreen(selections=selections)
        self.push_screen(log_screen)

def main():
    app = OsApp()
    app.title = f"Output of multiple well known UNIX commands".title()
    app.sub_title = f"{len(OS_COMMANDS)} commands available"
    app.run()

if __name__ == "__main__":
    main()
```

### Key Concepts

**Application Structure**:
1. **App class** - Extends `App`, implements `compose()` and optionally `mount()`
2. **compose()** - Yields widgets in display order
3. **Bindings** - Single letter keyboard shortcuts (e.g., 'q' to quit)
4. **Event listeners** - Use `@on()` decorator to handle widget events
5. **Screens** - Push/pop screens for multi-screen navigation

**Widget Management**:
- Each widget has customization options
- Query widgets by ID: `self.query_one('#cmds', SelectionList)`
- Enable/disable buttons based on state
- Tooltips provide user guidance

### CSS Styling

**os_app.tcss**:
```css
Screen {
    layout: vertical;
}

Header {
    dock: top;
}

Footer {
    dock: bottom;
}

SelectionList {
    padding: 1;
    border: solid $accent;
    width: 1fr;
    height: 80%;
}

Button {
    width: 1fr
}
```

**Key Points**:
> The dialect of CSS used in Textual is greatly simplified over web based CSS and much easier to learn.

- Control appearance (colors, position, size) without modifying code
- Separate stylesheet for clean code organization

---

## Log Display Screen (Separate Screen Pattern)

```python
import asyncio
from typing import List
from textual import on, work
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Log
from textual.worker import Worker
from textual.app import ComposeResult

class LogScreen(ModalScreen):
    count = reactive(0)
    MAX_LINES = 10_000
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "log_screen.tcss"

    def __init__(
            self,
            name: str | None = None,
            ident: str | None = None,
            classes: str | None = None,
            selections: List = None
    ):
        super().__init__(name, ident, classes)
        self.selections = selections

    def compose(self) -> ComposeResult:
        yield Label(f"Running {len(self.selections)} commands")
        event_log = Log(
            id='event_log',
            max_lines=LogScreen.MAX_LINES,
            highlight=True
        )
        event_log.loading = True
        yield event_log
        button = Button("Close", id="close", variant="success")
        button.disabled = True
        yield button

    async def on_mount(self) -> None:
        event_log = self.query_one('#event_log', Log)
        event_log.loading = False
        event_log.clear()
        lst = '\n'.join(self.selections)
        event_log.write(f"Preparing:\n{lst}")
        event_log.write("\n")

        for command in self.selections:
            self.count += 1
            self.run_process(cmd=command)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self.count == 0:
            button = self.query_one('#close', Button)
            button.disabled = False
        self.log(event)

    @work(exclusive=False)
    async def run_process(self, cmd: str) -> None:
        event_log = self.query_one('#event_log', Log)
        event_log.write_line(f"Running: {cmd}")
        # Combine STDOUT and STDERR output
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise ValueError(f"'{cmd}' finished with errors ({proc.returncode})")
        stdout = stdout.decode(encoding='utf-8', errors='replace')
        if stdout:
            event_log.write(f'\nOutput of "{cmd}":\n')
            event_log.write(stdout)
        self.count -= 1

    @on(Button.Pressed, "#close")
    def on_button_pressed(self, _) -> None:
        self.app.pop_screen()
```

### Screen Pattern Features

**ModalScreen**:
- Extends `ModalScreen` for modal display
- Has own `compose()` and `mount()` methods
- Independent lifecycle from main app

**Async Workers**:
- `@work(exclusive=False)` runs commands asynchronously
- Using [asyncio](https://docs.python.org/3/library/asyncio.html) for non-blocking execution
- Main TUI thread updates UI as results arrive
- Workers manage concurrency without complexity

**Screen Navigation**:
- `push_screen()` - Push new screen onto stack
- `pop_screen()` - Return to previous screen
- Screens stack for nested navigation

---

## Example 2: Race Results Table

Application displaying race results with sortable columns and detailed row views.

### Main Application

```python
#!/usr/bin/env python
"""
Author: Jose Vicente Nunez
"""
from typing import Any, List

from rich.style import Style
from textual import on
from textual.app import ComposeResult, App
from textual.command import Provider
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header

MY_DATA = [
    ("level", "name", "gender", "country", "age"),
    ("Green", "Wai", "M", "MYS", 22),
    ("Red", "Ryoji", "M", "JPN", 30),
    ("Purple", "Fabio", "M", "ITA", 99),
    ("Blue", "Manuela", "F", "VEN", 25)
]

class DetailScreen(ModalScreen):
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "details_screen.tcss"

    def __init__(
            self,
            name: str | None = None,
            ident: str | None = None,
            classes: str | None = None,
            row: List[Any] | None = None,
    ):
        super().__init__(name, ident, classes)
        # Rest of screen code will be show later

class CustomCommand(Provider):

    def __init__(self, screen: Screen[Any], match_style: Style | None = None):
        super().__init__(screen, match_style)
        self.table = None
        # Rest of provider code will be show later

class CompetitorsApp(App):
    BINDINGS = [
        ("q", "quit_app", "Quit"),
    ]
    CSS_PATH = "competitors_app.tcss"
    # Enable the command palette, to add our custom filter commands
    ENABLE_COMMAND_PALETTE = True
    # Add the default commands and the TablePopulateProvider to get a row directly by name
    COMMANDS = App.COMMANDS | {CustomCommand}

    def action_quit_app(self):
        self.exit(0)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        table = DataTable(id=f'competitors_table')
        table.cursor_type = 'row'
        table.zebra_stripes = True
        table.loading = True
        yield table
        yield Footer()

    def on_mount(self) -> None:
        table = self.get_widget_by_id(f'competitors_table', expect_type=DataTable)
        columns = [x.title() for x in MY_DATA[0]]
        table.add_columns(*columns)
        table.add_rows(MY_DATA[1:])
        table.loading = False
        table.tooltip = "Select a row to get more details"

    @on(DataTable.HeaderSelected)
    def on_header_clicked(self, event: DataTable.HeaderSelected):
        table = event.data_table
        table.sort(event.column_key)

    @on(DataTable.RowSelected)
    def on_row_clicked(self, event: DataTable.RowSelected) -> None:
        table = event.data_table
        row = table.get_row(event.row_key)
        runner_detail = DetailScreen(row=row)
        self.show_detail(runner_detail)

    def show_detail(self, detailScreen: DetailScreen):
        self.push_screen(detailScreen)

def main():
    app = CompetitorsApp()
    app.title = f"Summary".title()
    app.sub_title = f"{len(MY_DATA)} users"
    app.run()

if __name__ == "__main__":
    main()
```

### Table Features

**DataTable Widget**:
- `cursor_type = 'row'` - Select entire rows
- `zebra_stripes = True` - Alternating row colors
- `add_columns()` / `add_rows()` - Populate data
- `sort()` - Sort by column

**Command Palette**:
- Enabled by default with Header widget
- `ENABLE_COMMAND_PALETTE = True` - Explicitly enable
- `COMMANDS = App.COMMANDS | {CustomCommand}` - Add custom provider
- Fuzzy search through table data

**Event Handling**:
- `@on(DataTable.HeaderSelected)` - Handle column header clicks
- `@on(DataTable.RowSelected)` - Handle row selection
- Events contain relevant data (row, column keys)

---

## Detail Screen with Markdown

```python
from typing import Any, List
from textual import on
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, MarkdownViewer

MY_DATA = [
    ("level", "name", "gender", "country", "age"),
    ("Green", "Wai", "M", "MYS", 22),
    ("Red", "Ryoji", "M", "JPN", 30),
    ("Purple", "Fabio", "M", "ITA", 99),
    ("Blue", "Manuela", "F", "VEN", 25)
]

class DetailScreen(ModalScreen):
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "details_screen.tcss"

    def __init__(
            self,
            name: str | None = None,
            ident: str | None = None,
            classes: str | None = None,
            row: List[Any] | None = None,
    ):
        super().__init__(name, ident, classes)
        self.row: List[Any] = row

    def compose(self) -> ComposeResult:
        self.log.info(f"Details: {self.row}")
        columns = MY_DATA[0]
        row_markdown = "\n"
        for i in range(0, len(columns)):
            row_markdown += f"* **{columns[i].title()}:** {self.row[i]}\n"
        yield MarkdownViewer(f"""## User details:
        {row_markdown}
        """)
        button = Button("Close", variant="primary", id="close")
        button.tooltip = "Go back to main screen"
        yield button

    @on(Button.Pressed, "#close")
    def on_button_pressed(self, _) -> None:
        self.app.pop_screen()
```

**MarkdownViewer**:
- Renders Markdown in the terminal
- Creates table of contents automatically
- Clean, formatted display

---

## Custom Command Provider (Search)

```python
from functools import partial
from typing import Any, List
from rich.style import Style
from textual.command import Provider, Hit
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable
from textual.app import App

class CustomCommand(Provider):

    def __init__(self, screen: Screen[Any], match_style: Style | None = None):
        super().__init__(screen, match_style)
        self.table = None

    async def startup(self) -> None:
        my_app = self.app
        my_app.log.info(f"Loaded provider: CustomCommand")
        self.table = my_app.query(DataTable).first()

    async def search(self, query: str) -> Hit:
        matcher = self.matcher(query)

        my_app = self.screen.app
        assert isinstance(my_app, CompetitorsApp)

        my_app.log.info(f"Got query: {query}")
        for row_key in self.table.rows:
            row = self.table.get_row(row_key)
            my_app.log.info(f"Searching {row}")
            searchable = row[1]
            score = matcher.match(searchable)
            if score > 0:
                runner_detail = DetailScreen(row=row)
                yield Hit(
                    score,
                    matcher.highlight(f"{searchable}"),
                    partial(my_app.show_detail, runner_detail),
                    help=f"Show details about {searchable}"
                )
```

### Provider Pattern

**Custom Search**:
1. Extend `Provider` class
2. Implement `search()` method (async)
3. Use `matcher.match()` for fuzzy search
4. Return `Hit` objects for matches

**Hit Objects**:
- `score` - Match quality (for sorting)
- `matcher.highlight()` - Highlighted search term
- `partial()` - Callable to execute on selection
- `help` - Help text displayed

**Lifecycle**:
- `startup()` - Called once on provider initialization
- `search()` - Called for each query
- All methods are async (non-blocking UI)

---

## Debugging and Development Tools

### Install Dev Tools

```bash
pip install textual-dev==1.3.0
```

### Key Testing Tool

See what key events are generated:

```bash
textual keys
```

### Screenshot Capture

```bash
textual run --screenshot 5 ./kodegeek_textualize/log_scroller.py
```

### Logging and Console

**Start console**:
```bash
. ~/virtualenv/Textualize/bin/activate
textual console
```

**Run app in dev mode** (separate terminal):
```bash
. ~/virtualenv/Textualize/bin/activate
textual run --dev ./kodegeek_textualize/log_scroller.py
```

**Log from app**:
```python
my_app = self.screen.app
my_app.log.info(f"Loaded provider: CustomCommand")
```

**Dev Mode Benefits**:
- See events and messages in real-time
- Hot reload CSS changes
- No app restart needed

---

## Unit Testing

Using `unittest.IsolatedAsyncioTestCase` for async code:

### Test Log Scroller

```python
import unittest
from textual.widgets import Log, Button
from kodegeek_textualize.log_scroller import OsApp

class LogScrollerTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_log_scroller(self):
        app = OsApp()
        self.assertIsNotNone(app)
        async with app.run_test() as pilot:
            # Execute the default commands
            await pilot.click(Button)
            await pilot.pause()
            event_log = app.screen.query(Log).first()  # We pushed the screen, query nodes from there
            self.assertTrue(event_log.lines)
            await pilot.click("#close")  # Close the new screen, pop the original one
            await pilot.press("q")  # Quit the app by pressing q


if __name__ == '__main__':
    unittest.main()
```

### Test Table App

```python
import unittest
from textual.widgets import DataTable, MarkdownViewer
from kodegeek_textualize.table_with_detail_screen import CompetitorsApp


class TableWithDetailTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_app(self):
        app = CompetitorsApp()
        self.assertIsNotNone(app)
        async with app.run_test() as pilot:

            """
            Test the command palette
            """
            await pilot.press("ctrl+\\\\")
            for char in "manuela".split():
                await pilot.press(char)
            await pilot.press("enter")
            markdown_viewer = app.screen.query(MarkdownViewer).first()
            self.assertTrue(markdown_viewer.document)
            await pilot.click("#close")  # Close the new screen, pop the original one

            """
            Test the table
            """
            table = app.screen.query(DataTable).first()
            coordinate = table.cursor_coordinate
            self.assertTrue(table.is_valid_coordinate(coordinate))
            await pilot.press("enter")
            await pilot.pause()
            markdown_viewer = app.screen.query(MarkdownViewer).first()
            self.assertTrue(markdown_viewer)
            # Quit the app by pressing q
            await pilot.press("q")


if __name__ == '__main__':
    unittest.main()
```

### Run Tests

```bash
python -m unittest tests/*.py
```

**Test Pattern**:
1. Use `app.run_test()` to get `pilot` instance
2. Interact with `pilot.click()`, `pilot.press()`, `pilot.pause()`
3. Query widgets from app/screen
4. Assert expected behavior
5. Clean up with exit actions

---

## Packaging

### pyproject.toml Example

```toml
[build-system]
requires = [
    "setuptools >= 67.8.0",
    "wheel>=0.42.0",
    "build>=1.0.3",
    "twine>=4.0.2",
    "textual-dev>=1.2.1"
]
build-backend = "setuptools.build_meta"

[project]
name = "KodegeekTextualize"
version = "0.0.3"
authors = [
    {name = "Jose Vicente Nunez", email = "kodegeek.com@protonmail.com"},
]
description = "Collection of scripts that show how to use several features of textualize"
readme = "README.md"
requires-python = ">=3.9"
keywords = ["running", "race"]
classifiers = [
    "Environment :: Console",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Intended Audience :: End Users/Desktop",
    "Topic :: Utilities"
]
dynamic = ["dependencies"]

[project.scripts]
log_scroller = "kodegeek_textualize.log_scroller:main"
table_detail = "kodegeek_textualize.table_with_detail_screen:main"

[tool.setuptools]
include-package-data = true

[tool.setuptools.packages.find]
where = ["."]
exclude = ["test*"]

[tool.setuptools.package-data]
kodegeek_textualize = ["*.txt", "*.tcss", "*.csv"]
img = ["*.svg"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}
```

### Build and Install

```bash
. ~/virtualenv/Textualize/bin/activate
python -m build
pip install dist/KodegeekTextualize-*-py3-none-any.whl
```

**Key Points**:
- Include CSS files with `package-data`
- Define entry points with `project.scripts`
- Use `build` for modern packaging

---

## Key Takeaways

**Textual Advantages**:
- Runs on servers without GUI (low resources)
- Modern async support
- Simple CSS for styling
- Rich widget library

**Common Patterns**:
1. **Compose → Mount** - Build UI then populate data
2. **@on() decorators** - Handle widget events
3. **Screen stack** - Push/pop for navigation
4. **Workers** - Async operations without blocking UI
5. **Command palette** - Built-in search/commands

**Development Tools**:
- `textual console` - Real-time logging
- `textual run --dev` - Hot reload CSS
- `textual keys` - Test key bindings
- `textual run --screenshot` - Capture screenshots

**Testing Strategy**:
- Use `IsolatedAsyncioTestCase` for async code
- `run_test()` provides `pilot` for interaction
- Test event flows, not just final state

---

## Further Resources

**Must-Read**:
- [Official Tutorial](https://textual.textualize.io/tutorial/) - Comprehensive guide with examples
- [API Reference](https://textual.textualize.io/api/) - Complete API documentation
- [Widget Gallery](https://textual.textualize.io/widget_gallery/) - All built-in widgets

**Related Projects**:
- [Rich](https://github.com/Textualize/rich) - Terminal formatting (powers Textual)
- [Trogon](https://github.com/Textualize/trogon) - Make CLI self-discoverable
- [Textual-web](https://github.com/Textualize/textual-web) - Run Textual apps in browser

**Advanced Topics**:
- [Layout Design](https://textual.textualize.io/how-to/design-a-layout/) - Plan component alignment
- [Custom Widgets](https://textual.textualize.io/guide/widgets/) - Build your own widgets
- [Asyncio Development](https://docs.python.org/3/library/asyncio-dev.html) - Understanding async patterns

---

## Summary

This crash course covered:
- ✓ Installation and setup
- ✓ Two complete working examples (log scroller, race table)
- ✓ Core patterns (compose, mount, events, screens)
- ✓ CSS styling
- ✓ Async workers
- ✓ Custom command providers
- ✓ Debugging tools
- ✓ Unit testing
- ✓ Packaging

**Total code**: Less than 200 lines per app for full TUI functionality.

**What makes Textual powerful**: Simple API + Rich ecosystem + Modern async + Terminal/web support = Rapid TUI development.
