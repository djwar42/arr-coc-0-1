# Fedora Crash Course - Textual TUI Tutorial

## Overview

This document covers beginner-friendly Textual tutorials, debugging techniques, setup, and practical tips from the Fedora Magazine crash course article. The article provides hands-on examples demonstrating log scrolling and table applications.

From [Crash Course On Using Textual - Fedora Magazine](https://fedoramagazine.org/crash-course-on-using-textual/) (accessed 2025-11-02)

## What You Need

### Prerequisites

1. Basic programming experience, preferably in Python
2. Understanding basic object-oriented concepts (classes, inheritance)
3. A machine with Linux and Python 3.9+ installed
4. A good editor (Vim or PyCharm recommended)

### Installation

**Create virtual environment:**
```bash
python3 -m venv ~/virtualenv/Textualize
```

**Install from Git (editable distribution):**
```bash
. ~/virtualenv/Textualize/bin/activate
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade build
pip install --editable .
```

**Or install from PyPI:**
```bash
. ~/virtualenv/Textualize/bin/activate
pip install --upgrade KodegeekTextualize
```

## Tutorial Application 1: Log Scroller

### Purpose
Execute a list of UNIX commands and capture output as they finish. Demonstrates command selection, asynchronous execution, and result display.

### Key Architecture Decisions

**Main application class:**
```python
class OsApp(App):
    BINDINGS = [
        ("q", "quit_app", "Quit"),
    ]
    CSS_PATH = "os_app.tcss"
    ENABLE_COMMAND_PALETTE = False  # Don't need command palette

    def compose(self) -> ComposeResult:
        selections = [Selection(name.title(), ' '.join(cmd), True)
                     for name, cmd in OS_COMMANDS.items()
                     if shutil.which(cmd[0].strip())]
        yield Header(show_clock=False)
        sel_list = SelectionList(*selections, id='cmds')
        sel_list.tooltip = "Select one more more command to execute"
        yield sel_list
        yield Button(f"Execute {len(selections)} commands",
                    id="exec", variant="primary")
        yield Footer()
```

### Core Patterns Demonstrated

**1. Compose Method:**
- Yield widgets to add them to the main screen
- Widgets added in order of yield
- Each widget has customization options

**2. Event Handling with Annotations:**
```python
@on(SelectionList.SelectedChanged)
def on_selection(self, event: SelectionList.SelectedChanged) -> None:
    button = self.query_one("#exec", Button)
    selections = len(event.selection_list.selected)
    if selections:
        button.disabled = False
    else:
        button.disabled = True
    button.label = f"Execute {selections} commands"
```

**3. Screen Navigation:**
- Use `push_screen()` to show new screens
- Use `pop_screen()` to return to previous screen
- Screens can be modal (`ModalScreen` base class)

**4. CSS Styling:**
```css
Screen {
    layout: vertical;
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

**Key insight from article:** "The dialect of CSS used in Textual is greatly simplified over web based CSS and much easier to learn."

### Asynchronous Command Execution

**Using Workers for Concurrency:**
```python
class LogScreen(ModalScreen):
    count = reactive(0)

    async def on_mount(self) -> None:
        event_log = self.query_one('#event_log', Log)
        for command in self.selections:
            self.count += 1
            self.run_process(cmd=command)

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
            raise ValueError(f"'{cmd}' finished with errors")
        stdout = stdout.decode(encoding='utf-8', errors='replace')
        if stdout:
            event_log.write(f'\nOutput of "{cmd}":\n')
            event_log.write(stdout)
        self.count -= 1
```

**Key patterns:**
- `@work(exclusive=False)` allows multiple commands to run concurrently
- Worker annotation prevents UI freezing
- Use `reactive()` for state management across async operations
- Workers have dedicated section in Textual manual for deeper learning

## Tutorial Application 2: Race Results Table

### Purpose
Display race results in a sortable, searchable table with detail views. Demonstrates DataTable, command palette customization, and modal screens.

### DataTable Features

**Basic setup:**
```python
def compose(self) -> ComposeResult:
    yield Header(show_clock=True)

    table = DataTable(id='competitors_table')
    table.cursor_type = 'row'
    table.zebra_stripes = True
    table.loading = True
    yield table
    yield Footer()

def on_mount(self) -> None:
    table = self.get_widget_by_id('competitors_table', expect_type=DataTable)
    columns = [x.title() for x in MY_DATA[0]]
    table.add_columns(*columns)
    table.add_rows(MY_DATA[1:])
    table.loading = False
    table.tooltip = "Select a row to get more details"
```

### Event Handling Patterns

**Column header sorting:**
```python
@on(DataTable.HeaderSelected)
def on_header_clicked(self, event: DataTable.HeaderSelected):
    table = event.data_table
    table.sort(event.column_key)
```

**Row selection:**
```python
@on(DataTable.RowSelected)
def on_row_clicked(self, event: DataTable.RowSelected) -> None:
    table = event.data_table
    row = table.get_row(event.row_key)
    runner_detail = DetailScreen(row=row)
    self.push_screen(runner_detail)
```

### Using MarkdownViewer for Details

**Display structured data as Markdown:**
```python
class DetailScreen(ModalScreen):
    def compose(self) -> ComposeResult:
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
```

**Benefit:** MarkdownViewer automatically creates table of contents for structured content.

### Custom Command Palette Provider

**Enable search across table data:**
```python
class CompetitorsApp(App):
    ENABLE_COMMAND_PALETTE = True
    COMMANDS = App.COMMANDS | {CustomCommand}  # Add custom provider

class CustomCommand(Provider):
    async def startup(self) -> None:
        my_app = self.app
        self.table = my_app.query(DataTable).first()

    async def search(self, query: str) -> Hit:
        matcher = self.matcher(query)

        for row_key in self.table.rows:
            row = self.table.get_row(row_key)
            searchable = row[1]  # Search by name column
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

**Key aspects:**
- `Provider` requires implementing `search()` method
- Optional `startup()` method runs once at initialization
- `matcher.match()` returns score (>0 indicates match)
- `Hit` object contains: score, highlighted term, callable action, help text
- All Provider methods are async (non-blocking UI)

## Debugging Techniques

### Common Challenges

From article: "Debugging a Python Textual application is a little bit more challenging. This is because some operations can be asynchronous and setting breakpoints may be cumbersome when troubleshooting widgets."

### Install Dev Tools

```bash
pip install textual-dev==1.3.0
```

### Debug Tool 1: Key Capture

**Verify key events being captured:**
```bash
textual keys
```

- Shows what key combinations generate events
- Useful for debugging keyboard shortcuts
- Confirms proper key binding

### Debug Tool 2: Screenshots

**Capture app state visually:**
```bash
textual run --screenshot 5 ./kodegeek_textualize/log_scroller.py
```

- Takes screenshot after 5 seconds
- Useful for documentation
- Helps communicate layout issues to others

### Debug Tool 3: Console Logging

**Setup two-terminal debugging:**

Terminal 1 - Start console:
```bash
. ~/virtualenv/Textualize/bin/activate
textual console
```

Terminal 2 - Run app in dev mode:
```bash
. ~/virtualenv/Textualize/bin/activate
textual run --dev ./kodegeek_textualize/log_scroller.py
```

**Add logging to your app:**
```python
my_app = self.screen.app
my_app.log.info(f"Loaded provider: CustomCommand")
```

**Console output example:**
```
[20:29:43] SYSTEM                                                                                                                                                                 app.py:2188
Connected to devtools ( ws://127.0.0.1:8081 )
[20:29:43] SYSTEM                                                                                                                                                                 app.py:2192
---
[20:29:43] SYSTEM                                                                                                                                                                 app.py:2194
driver=<class 'textual.drivers.linux_driver.LinuxDriver'>
```

**Dev mode bonus:** CSS changes auto-reload without restart.

## Testing Textual Applications

### Using unittest with Async Support

**Test framework:**
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
            event_log = app.screen.query(Log).first()
            self.assertTrue(event_log.lines)
            await pilot.click("#close")
            await pilot.press("q")
```

### Pilot API Patterns

**Key methods:**
- `pilot.click(widget)` - Click a widget
- `pilot.press(key)` - Simulate key press
- `pilot.pause()` - Wait for events to process
- `app.screen.query(Widget).first()` - Query widget from screen

**Testing command palette:**
```python
async def test_app(self):
    async with app.run_test() as pilot:
        # Open command palette
        await pilot.press("ctrl+\\")
        # Type search query
        for char in "manuela".split():
            await pilot.press(char)
        await pilot.press("enter")
        # Verify results
        markdown_viewer = app.screen.query(MarkdownViewer).first()
        self.assertTrue(markdown_viewer.document)
```

**Run tests:**
```bash
python -m unittest tests/*.py
# Output:
# ..
# ----------------------------------------------------------------------
# Ran 2 tests in 2.065s
# OK
```

## Packaging

### Setup Configuration

**Include CSS files in package:**

```toml
[build-system]
requires = [
    "setuptools >= 67.8.0",
    "wheel>=0.42.0",
    "build>=1.0.3",
    "textual-dev>=1.2.1"
]

[project]
name = "KodegeekTextualize"
version = "0.0.3"
requires-python = ">=3.9"
classifiers = [
    "Environment :: Console",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
]

[project.scripts]
log_scroller = "kodegeek_textualize.log_scroller:main"
table_detail = "kodegeek_textualize.table_with_detail_screen:main"

[tool.setuptools.package-data]
kodegeek_textualize = ["*.txt", "*.tcss", "*.csv"]
```

**Build and install:**
```bash
. ~/virtualenv/Textualize/bin/activate
python -m build
pip install dist/KodegeekTextualize-*-py3-none-any.whl
```

## Beginner Tips from Tutorial

### 1. Design on Paper First
From article: "Grab a piece of paper and draw how you picture the components should align together. It will save you time and headaches later."

### 2. Start with Simple Widgets
- Begin with basic widgets (Header, Footer, Button)
- Add complexity incrementally
- Test each addition before moving forward

### 3. Use CSS for Styling
- Keep styling separate from logic
- CSS is simplified vs web CSS
- Easier to maintain and adjust

### 4. Leverage Existing Widgets
- Rich library widgets can be used
- Don't reinvent common UI patterns
- Check widget gallery first

### 5. Understand Compose vs Mount
- `compose()`: Define widget structure
- `mount()`: Customize after composition, load data
- Both are crucial for proper initialization

### 6. Use Annotations for Events
- `@on(Widget.Event)` pattern is clean
- Reduces boilerplate
- Clear event-to-handler mapping

## Common Beginner Mistakes

### Mistake 1: Blocking the Main Thread
**Problem:** Running slow operations in event handlers freezes UI

**Solution:** Use `@work()` decorator for async operations

### Mistake 2: Forgetting to Enable Command Palette
**Problem:** Command palette requires Header widget

**Solution:** Always yield Header if using command palette

### Mistake 3: Not Using Query Selectors
**Problem:** Hard-coding widget references

**Solution:** Use `query_one()` and widget IDs

### Mistake 4: Ignoring CSS Path
**Problem:** Inline styling becomes unmaintainable

**Solution:** Use `CSS_PATH` for external stylesheets

### Mistake 5: Not Testing with Pilot
**Problem:** Manual testing is time-consuming

**Solution:** Write automated tests with `run_test()` and Pilot

## Framework Benefits Highlighted

### 1. Terminal Native
- Runs on servers without GUI
- Low resource requirements
- SSH-friendly

### 2. Modern Python Features
- Embraces asyncio
- Uses type hints
- Object-oriented design

### 3. Rapid Development
- Simple API
- Good documentation
- Active community

### 4. Rich Integration
- Can use Rich library components
- Consistent styling
- Powerful text rendering

### 5. Web Support (Future)
From article: "Textual-web is a promising project that will allow you to run Textual applications on a browser."

## Next Steps from Tutorial

1. **Official Tutorial:** Take the complete Textual tutorial with API reference
2. **Rich Library:** Explore Rich widgets that can be used in Textual
3. **Custom Widgets:** Learn to create your own widget components
4. **Asyncio Deep Dive:** Read asyncio developer documentation
5. **Trogon:** Explore making CLIs self-discoverable
6. **External Projects:** Check Textualize.io portfolio for real-world examples

## Related Resources

**From tutorial:**
- Official Textual tutorial: https://textual.textualize.io/tutorial/
- Textual API reference: https://textual.textualize.io/api/
- Rich library: https://github.com/Textualize/rich
- Layout design guide: https://textual.textualize.io/how-to/design-a-layout/
- Workers documentation: https://textual.textualize.io/guide/workers/
- Testing guide: https://textual.textualize.io/guide/testing/

## Sources

**Primary Source:**
- [Crash Course On Using Textual - Fedora Magazine](https://fedoramagazine.org/crash-course-on-using-textual/) (accessed 2025-11-02)
  - Author: Jose Nunez
  - Published: January 10, 2024
  - Complete code examples available

**Tutorial Code Repository:**
- Referenced but not directly scraped: https://tutorials.kodegeek.com/Textualize/

**Key Topics Covered:**
- Textual installation and setup on Fedora/Linux
- Log scroller application (command execution)
- Race results table application (DataTable usage)
- Debugging with textual console, keys, screenshots
- Testing with unittest.IsolatedAsyncioTestCase
- Packaging Textual applications
- Common beginner mistakes and best practices
