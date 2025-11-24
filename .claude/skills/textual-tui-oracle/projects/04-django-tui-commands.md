# Django-TUI: Interactive Django Command Interface

## Overview

**Django-TUI** is a Textual-based terminal user interface for inspecting and running Django management commands. Built with Textual and Trogon, it provides a graphical interface for Django's command-line tools, complete with auto-discovery of commands, form-based parameter input, and an interactive Python shell for ORM queries.

**Repository**: [anze3db/django-tui](https://github.com/anze3db/django-tui)
**Author**: Anže Pečar ([@anze3db](https://github.com/anze3db))
**License**: MIT
**Key Technologies**: Textual, Trogon, Django

From [GitHub Repository](https://github.com/anze3db/django-tui) (accessed 2025-11-02):
- 285 stars on GitHub
- Provides TUI for all Django management commands
- Includes interactive shell with ORM query support
- Auto-discovers commands from installed Django apps

## Key Features

### 1. Command Discovery & Execution
- Auto-discovers all Django management commands from installed apps
- Groups commands by Django app
- Displays command documentation and parameter schemas
- Generates form-based interfaces for command parameters
- Executes commands with proper argument formatting

### 2. Interactive Python Shell
- Embedded Python REPL with Django context
- Auto-imports Django models, utilities, and common functions
- Live code execution with output capture
- Syntax highlighting and auto-completion
- Comment toggling and clipboard integration

### 3. Command Parameter Forms
- Dynamic form generation from command schemas
- Type-aware input widgets (bool, int, string, choices)
- Required/optional field handling
- Multi-value parameter support
- Real-time command preview with syntax highlighting

## Installation & Setup

**Installation**:
```bash
pip install django-tui
```

**Django Configuration** (settings.py):
```python
INSTALLED_APPS = [
    # ... other apps
    "django_tui",
]
```

**Running**:
```bash
# Launch command interface
python manage.py tui

# Launch directly to interactive shell
python manage.py tui --shell
```

## Architecture

### Command Discovery System

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 42-147):

The `introspect_django_commands()` function discovers all Django commands and converts them to Trogon's `CommandSchema` format:

```python
def introspect_django_commands() -> dict[str, CommandSchema]:
    groups = {}
    for name, app_name in get_commands().items():
        try:
            kls = load_command_class(app_name, name)
        except AttributeError:
            # Skip invalid commands
            continue

        if app_name == "django.core":
            group_name = "django"
        else:
            group_name = app_name.rpartition(".")[-1]

        parser = kls.create_parser(f"django {name}", name)
        options = []
        args = []
        root = []

        # Extract arguments and options from parser
        for action in parser._actions:
            # Convert Django argparse actions to Click-compatible schemas
            # ... (parameter extraction logic)

        command = CommandSchema(
            name=name,
            function=None,
            is_group=False,
            docstring=None,
            options=options,
            arguments=args,
            parent=groups[group_name],
        )

        groups[group_name].subcommands[name] = command

    return groups
```

**Key Integration Pattern**: Uses Django's built-in `get_commands()` and `load_command_class()` to discover commands, then introspects their argparse parsers to extract parameter schemas.

### Main Application Structure

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 322-371):

```python
class DjangoTui(App):
    CSS_PATH = Path(__file__).parent / "trogon.scss"

    BINDINGS = [
        Binding(key="ctrl+z", action="copy_command", description="Copy Command to Clipboard"),
        Binding(key="ctrl+t", action="focus_command_tree", description="Focus Command Tree"),
        Binding(key="ctrl+s", action="focus('search')", description="Search"),
        Binding(key="ctrl+j", action="select_mode('shell')", description="Shell"),
        Binding(key="f1", action="about", description="About"),
    ]

    def __init__(self, *, open_shell: bool = False) -> None:
        super().__init__()
        self.post_run_command: list[str] = []
        self.is_grouped_cli = True
        self.execute_on_exit = False
        self.app_name = "python manage.py"
        self.command_name = "django-tui"
        self.open_shell = open_shell

    def get_default_screen(self) -> DjangoCommandBuilder:
        if self.open_shell:
            return InteractiveShellScreen("Interactive Shell")
        else:
            return DjangoCommandBuilder(self.app_name, self.command_name)
```

**Multi-Screen Architecture**: Uses Textual's screen system to switch between command interface and interactive shell.

### Command Builder Screen

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 169-269):

```python
class DjangoCommandBuilder(Screen):
    BINDINGS = [
        Binding(key="ctrl+r", action="close_and_run", description="Close & Run"),
    ]

    def compose(self) -> ComposeResult:
        tree = CommandTree("Commands", self.command_schemas, self.command_name)

        sidebar = Vertical(
            Label(title, id="home-commands-label"),
            tree,
            id="home-sidebar",
        )

        yield sidebar

        with Vertical(id="home-body"):
            with Horizontal(id="home-command-description-container") as vs:
                vs.can_focus = False
                yield Static(self.click_app_name or "", id="home-command-description")

            scrollable_body = VerticalScroll(
                Static(""),
                id="home-body-scroll",
            )
            scrollable_body.can_focus = False
            yield scrollable_body

            yield Horizontal(
                NonFocusableVerticalScroll(
                    Static("", id="home-exec-preview-static"),
                    id="home-exec-preview-container",
                ),
                id="home-exec-preview",
            )

        yield Footer()
```

**Layout Pattern**: Three-column layout with:
1. **Sidebar**: CommandTree widget showing grouped commands
2. **Body**: Dynamic form generated from selected command's schema
3. **Preview**: Real-time command string with syntax highlighting

### Event Handling for Command Selection

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 271-285):

```python
@on(Tree.NodeHighlighted)
async def selected_command_changed(self, event: Tree.NodeHighlighted[CommandSchema]) -> None:
    """When we highlight a node in the CommandTree, the main body updates
    to display a form specific to the highlighted command."""
    await self._refresh_command_form(event.node)

@on(CommandForm.Changed)
def update_command_data(self, event: CommandForm.Changed) -> None:
    self.command_data = event.command_data
    self._update_execution_string_preview()
```

**Reactive UI Pattern**: Uses Textual's `@on()` decorator to respond to tree navigation and form changes, updating the preview in real-time.

## Interactive Shell Implementation

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 245-290):

### Shell Screen Architecture

```python
class InteractiveShellScreen(Screen):
    def __init__(self, name: str | None = None):
        super().__init__(name)
        self.input_tarea = ExtendedTextArea("", id="input", language="python", theme="vscode_dark")
        self.output_tarea = TextArea(
            "# Output",
            id="output",
            language="python",
            theme="vscode_dark",
            classes="text-area",
        )

    BINDINGS = [
        Binding(key="ctrl+r", action="run_code", description="Run the query"),
        Binding(key="ctrl+z", action="copy_command", description="Copy to Clipboard"),
        Binding(key="f1", action="editor_keys", description="Key Bindings"),
        Binding(key="f2", action="default_imports", description="Default imports"),
        Binding(key="ctrl+j", action="select_mode('commands')", description="Commands"),
        Binding(key="ctrl+underscore", action="toggle_comment", description="Toggle Comment"),
    ]

    def compose(self) -> ComposeResult:
        self.input_tarea.focus()

        yield HorizontalScroll(
            self.input_tarea,
            self.output_tarea,
        )
        yield Label(f"Python: {platform.python_version()}  Django: {django.__version__}")
        yield Footer()
```

**Split-Pane Design**: Uses `HorizontalScroll` with two `TextArea` widgets for input/output separation.

### Auto-Import System

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 30-108):

```python
DEFAULT_IMPORT = {
    "rich": ["print_json", "print"],
    "django.db.models": [
        "Avg", "Case", "Count", "F", "Max", "Min",
        "Prefetch", "Q", "Sum", "When",
    ],
    "django.conf": ["settings"],
    "django.core.cache": ["cache"],
    "django.contrib.auth": ["get_user_model"],
    "django.utils": ["timezone"],
    "django.urls": ["reverse"],
}

@lru_cache
def get_modules():
    """Return list of modules and symbols to import"""
    mods = {}
    for module_name, symbols in DEFAULT_IMPORT.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            warnings.warn(f"django_admin_shell - autoimport warning :: {str(e)}", ImportWarning)
            continue

        mods[module_name] = []
        for symbol_name in symbols:
            if hasattr(module, symbol_name):
                mods[module_name].append(symbol_name)

    # Auto-import all Django models
    for model_class in apps.get_models():
        _mod = model_class.__module__
        classes = mods.get(_mod, [])
        classes.append(model_class.__name__)
        mods[_mod] = classes

    return mods

@lru_cache
def get_scope():
    """Return map with symbols to module/object"""
    scope = {}
    for module_name, symbols in get_modules().items():
        module = importlib.import_module(module_name)
        for symbol_name in symbols:
            scope[symbol_name] = getattr(module, symbol_name)
    return scope
```

**Smart Import Strategy**:
- Pre-imports common Django utilities
- Auto-discovers and imports all installed Django models
- Uses `@lru_cache` for performance
- Creates execution scope with all symbols

### Code Execution

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 119-145):

```python
def run_code(code):
    """Execute code and return result with status = success|error
    Function manipulates stdout to grab output from exec"""
    status = "success"
    out = ""
    tmp_stdout = sys.stdout
    buf = StringIO()

    try:
        sys.stdout = buf
        exec(code, None, get_scope())
    except Exception:
        out = traceback.format_exc()
        status = "error"
    else:
        out = buf.getvalue()
    finally:
        sys.stdout = tmp_stdout

    result = {
        "code": code,
        "out": out,
        "status": status,
    }
    return result
```

**Execution Pattern**: Redirects stdout to capture print statements, executes code in pre-populated scope, handles exceptions gracefully.

### Enhanced TextArea Widget

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 148-164):

```python
class ExtendedTextArea(TextArea):
    """A subclass of TextArea with parenthesis-closing functionality."""

    def _on_key(self, event: events.Key) -> None:
        auto_close_chars = {
            "(": ")",
            "{": "}",
            "[": "]",
            "'": "'",
            '"': '"',
        }
        closing_char = auto_close_chars.get(event.character)
        if closing_char:
            self.insert(f"{event.character}{closing_char}")
            self.move_cursor_relative(columns=-1)
            event.prevent_default()
```

**Widget Extension Pattern**: Subclasses `TextArea` to add IDE-like auto-closing behavior for brackets and quotes.

## Textual Widget Usage

### Widgets Used

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) imports (lines 13-26):

**Core Widgets**:
- `Tree` / `CommandTree` - Command navigation hierarchy
- `Static` - Text display for descriptions and previews
- `Label` - Title and status information
- `Button` - Execute command action
- `Footer` - Keyboard binding display
- `TextArea` - Multi-line code editor (interactive shell)

**Trogon Widgets** (specialized for CLI introspection):
- `CommandTree` - Hierarchical command browser
- `CommandForm` - Dynamic form generation from command schemas
- `CommandInfo` - Command documentation display
- `TextDialog` - Modal dialogs

**Containers**:
- `Vertical` / `VerticalScroll` - Column layouts
- `Horizontal` / `HorizontalScroll` - Row layouts
- `Screen` - Full-screen views
- `ModalScreen` - Modal overlays

### Layout Patterns

**Three-Column Layout** (Command Builder):
```
┌─────────────┬──────────────────────┬─────────────────┐
│ CommandTree │  Dynamic Form        │                 │
│ (Sidebar)   │  (Command Params)    │                 │
│             │                      │                 │
│             ├──────────────────────┤                 │
│             │  Command Preview     │                 │
│             │  (Syntax Highlight)  │                 │
└─────────────┴──────────────────────┴─────────────────┘
```

**Split-Pane Layout** (Interactive Shell):
```
┌─────────────────────────┬─────────────────────────┐
│ Input TextArea          │ Output TextArea         │
│ (Python Code)           │ (Execution Results)     │
│                         │                         │
│                         │                         │
└─────────────────────────┴─────────────────────────┘
```

### Screen Navigation

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 360-366):

```python
def action_select_mode(self, mode_id: Literal["commands", "shell"]) -> None:
    if mode_id == "commands":
        self.app.push_screen(DjangoCommandBuilder("python manage.py", "Test command name"))

    elif mode_id == "shell":
        self.app.push_screen(InteractiveShellScreen("Interactive Shell"))
```

**Screen Stack Pattern**: Uses `push_screen()` to navigate between command interface and shell, maintaining state.

## Django Integration Patterns

### 1. Command Discovery via Django's Management System

```python
from django.core.management import get_commands, load_command_class

# Discovers all commands from INSTALLED_APPS
commands = get_commands()  # Returns {command_name: app_name}

# Loads actual command class
kls = load_command_class(app_name, command_name)

# Creates argparse parser from command
parser = kls.create_parser(f"django {name}", name)
```

**Pattern**: Leverages Django's built-in command discovery instead of manual registration.

### 2. Argparse to Form Schema Conversion

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 61-95):

```python
for action in parser._actions:
    # Normalize nargs
    if action.nargs == "?":
        nargs = 1
    elif action.nargs in ("*", "+"):
        nargs = -1
    elif not action.nargs:
        nargs = 1
    else:
        nargs = action.nargs

    # Map argparse types to Click types
    if hasattr(action, "type"):
        if action.type is bool:
            type_ = click.BOOL
        elif action.type is int:
            type_ = click.INT
        elif action.type is str:
            type_ = click.STRING
        else:
            type_ = click.STRING if action.nargs != 0 else click.BOOL

    # Create option or argument schema
    if not action.option_strings:
        args.append(ArgumentSchema(...))
    else:
        options.append(OptionSchema(...))
```

**Pattern**: Introspects argparse metadata to generate type-aware form fields.

### 3. Django Model Auto-Import

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 87-91):

```python
from django.apps import apps

# Auto-discover all models
for model_class in apps.get_models():
    _mod = model_class.__module__
    classes = mods.get(_mod, [])
    classes.append(model_class.__name__)
    mods[_mod] = classes
```

**Pattern**: Uses Django's app registry to discover and import all models automatically.

### 4. Async-Safe Execution

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 315-318):

```python
def action_run_code(self) -> None:
    # Enable async-unsafe operations (ORM queries)
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()

    result = run_code(code)
    self.output_tarea.load_text(result["out"])
```

**Pattern**: Sets `DJANGO_ALLOW_ASYNC_UNSAFE` to allow ORM queries in Textual's async event loop.

### 5. Command Execution via execvp

From [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) (lines 337-348):

```python
def run(self, *, headless: bool = False, ...) -> None:
    try:
        super().run(headless=headless, size=size, auto_pilot=auto_pilot)
    finally:
        if self.post_run_command and self.execute_on_exit:
            console.print(
                f"Running [b cyan]{self.app_name} {' '.join(shlex.quote(s) for s in self.post_run_command)}[/]"
            )

            split_app_name = shlex.split(self.app_name)
            program_name = shlex.split(self.app_name)[0]
            arguments = [*split_app_name, *self.post_run_command]
            os.execvp(program_name, arguments)
```

**Pattern**: Uses `os.execvp()` to replace the TUI process with the actual Django command, ensuring proper terminal interaction.

## Key Textual Patterns Demonstrated

### 1. Trogon Integration for CLI Introspection

Django-TUI uses **Trogon**, a Textual-based library for creating TUIs from Click CLI apps. The integration pattern:

```python
from trogon.introspect import CommandSchema, OptionSchema, ArgumentSchema
from trogon.widgets.command_tree import CommandTree
from trogon.widgets.form import CommandForm

# Convert Django commands to Trogon schemas
command_schemas = introspect_django_commands()

# Use Trogon widgets with Django data
tree = CommandTree("Commands", command_schemas, command_name)
form = CommandForm(command_schema=command_schema, command_schemas=command_schemas)
```

**Lesson**: Trogon provides reusable CLI introspection widgets for any command-line tool.

### 2. Dynamic Form Generation

From command schemas to live forms:

```python
async def _update_form_body(self, node: TreeNode[CommandSchema]) -> None:
    parent = self.query_one("#home-body-scroll", VerticalScroll)
    for child in parent.children:
        await child.remove()

    command_schema = node.data
    command_form = CommandForm(command_schema=command_schema, ...)
    await parent.mount(command_form)
```

**Pattern**: Clear old widgets, generate new form based on selected command schema, mount dynamically.

### 3. Real-Time Preview Updates

```python
def _update_execution_string_preview(self) -> None:
    if self.command_data is not None:
        prefix = Text(f"{self.click_app_name} ", command_name_syntax_style)
        new_value = self.command_data.to_cli_string(include_root_command=False)
        highlighted_new_value = Text.assemble(prefix, self.highlighter(new_value))
        prompt_style = self.get_component_rich_style("prompt")
        preview_string = Text.assemble(("$ ", prompt_style), highlighted_new_value)
        self.query_one("#home-exec-preview-static", Static).update(preview_string)
```

**Pattern**: Use Rich's syntax highlighting + Textual's reactive updates for live command preview.

### 4. Clipboard Integration (Cross-Platform)

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 321-334):

```python
def action_copy_command(self) -> None:
    if sys.platform == "win32":
        copy_command = ["clip"]
    elif sys.platform == "darwin":
        copy_command = ["pbcopy"]
    else:
        copy_command = ["xclip", "-selection", "clipboard"]

    try:
        run(
            copy_command,
            input=text_to_copy,
            text=True,
            check=False,
        )
        self.notify(msg)
    except FileNotFoundError:
        self.notify(f"Could not copy to clipboard. `{copy_command[0]}` not found.", severity="error")
```

**Pattern**: Platform-specific clipboard handling with graceful degradation.

### 5. Comment Toggle Implementation

From [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) (lines 369-398):

```python
def action_toggle_comment(self) -> None:
    inline_comment_marker = "#"

    lines, first, last = self._get_selected_lines()
    stripped_lines = [line.lstrip() for line in lines]
    indents = [len(line) - len(line.lstrip()) for line in lines]

    # If lines are already commented, remove comments
    if lines and all(not line or line.startswith(inline_comment_marker) for line in stripped_lines):
        offsets = [
            0 if not line else (2 if line[len(inline_comment_marker)].isspace() else 1)
            for line in stripped_lines
        ]
        for lno, indent, offset in zip(range(first[0], last[0] + 1), indents, offsets):
            self.input_tarea.delete(
                start=(lno, indent),
                end=(lno, indent + offset),
                maintain_selection_offset=True,
            )
    # Add comment tokens to all lines
    else:
        indent = min([indent for indent, line in zip(indents, stripped_lines) if line])
        for lno, stripped_line in enumerate(stripped_lines, start=first[0]):
            if stripped_line:
                self.input_tarea.insert(
                    f"{inline_comment_marker} ",
                    location=(lno, indent),
                    maintain_selection_offset=True,
                )
```

**Pattern**: IDE-like comment toggling with smart indentation handling and selection preservation.

## Use Cases

### 1. Django Command Exploration
- Browse all available management commands
- Read command documentation
- Understand parameter requirements
- Test commands before scripting

### 2. Development Workflow
- Quick access to common commands (migrate, runserver, makemigrations)
- Form-based parameter input (no need to remember flags)
- Command history via preview

### 3. Database Queries
- Interactive ORM exploration
- Ad-hoc data analysis
- Model relationship testing
- Quick data inspection

### 4. Learning Tool
- Discover available Django commands
- Learn command parameters and options
- Experiment with ORM queries safely

## Lessons for Textual Developers

### 1. Third-Party Integration
Django-TUI demonstrates how to wrap external systems (Django management) with Textual UIs by:
- Introspecting existing APIs (argparse parsers)
- Converting to standard schemas (CommandSchema)
- Using specialized widget libraries (Trogon)

### 2. Dynamic UI Generation
Shows how to build forms dynamically from schemas:
- Parse metadata (command parameters)
- Generate appropriate widgets (input types)
- Wire up reactive updates (preview)

### 3. Multi-Modal Apps
Uses Textual's screen system for distinct modes:
- Command interface (form-based)
- Interactive shell (editor-based)
- Easy mode switching (Ctrl+J)

### 4. IDE-Like Features in TUI
Implements modern editor features:
- Syntax highlighting (Rich integration)
- Auto-closing brackets
- Comment toggling
- Clipboard operations
- Split-pane editing

### 5. Process Management
Clever use of `os.execvp()` to:
- Let TUI collect parameters
- Exit cleanly
- Replace process with actual command
- Preserve terminal context

## Code Quality Highlights

**Clean Architecture**:
- Separation between discovery (`introspect_django_commands()`), UI (`DjangoTui`), and execution
- Reusable screen components (`DjangoCommandBuilder`, `InteractiveShellScreen`)
- Clear widget hierarchy

**Error Handling**:
- Graceful failures for invalid commands
- Exception capture in shell execution
- Platform-specific fallbacks for clipboard

**Performance**:
- `@lru_cache` for expensive operations (module discovery)
- Lazy loading of command schemas
- Efficient widget mounting/unmounting

## Sources

**GitHub Repository**:
- [anze3db/django-tui](https://github.com/anze3db/django-tui) - Main repository (accessed 2025-11-02)

**Source Files**:
- [tui.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/tui.py) - Main TUI application (417 lines)
- [ish.py](https://github.com/anze3db/django-tui/blob/main/src/django_tui/management/commands/ish.py) - Interactive shell implementation (401 lines)
- [README.md](https://github.com/anze3db/django-tui/blob/main/README.md) - Project documentation

**Dependencies**:
- [Textual](https://github.com/Textualize/textual) - TUI framework
- [Trogon](https://github.com/Textualize/trogon) - CLI introspection widgets
- [Rich](https://github.com/Textualize/rich) - Terminal formatting and syntax highlighting

**Related Projects**:
- [django-admin-tui](https://github.com/valberg/django-admin-tui) - TUI for Django admin interface (by @valberg)
