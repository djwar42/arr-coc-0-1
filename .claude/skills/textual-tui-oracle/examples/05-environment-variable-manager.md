# Environment Variable Manager

## Overview

A practical TUI application for managing `.env` files across multiple projects from a single terminal interface. Solves the common developer problem of opening multiple editors just to change one API key or configuration value across different projects.

**From [GitHub - FyefoxxM/environment-variable-manager](https://github.com/FyefoxxM/environment-variable-manager) (accessed 2025-11-02)**

**Key Features:**
- Recursive scanning for `.env` files (up to 3 levels deep)
- View all environment variables across projects
- Edit variables in place with immediate saving
- Add new variables without opening an editor
- Simple numeric navigation (press 1-9 to edit files)

**Lines of Code:** ~296 lines (under 300-line project constraint)
**Build Time:** 6 hours (including learning Textual's widget system)

## Architecture

### Core Components

**Data Layer:**
- `EnvFile` dataclass - Represents a single `.env` file with path and variables
- `EnvScanner` - Finds and loads `.env` files recursively

**UI Layer:**
- `EnvManagerApp` - Main screen showing list of found files
- `EditScreen` - Edit variables for a specific file
- `AddVariableScreen` - Modal for adding new variables

### Application Structure

```python
from textual.app import App
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input, Label
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
```

## Implementation Details

### 1. EnvFile Dataclass

Handles loading and saving `.env` files with proper quote handling:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass
class EnvFile:
    path: Path
    variables: Dict[str, str]

    def load(self):
        """Load variables from .env file"""
        self.variables = {}
        if not self.path.exists():
            return

        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    self.variables[key] = value

    def save(self):
        """Save variables back to the .env file"""
        with open(self.path, 'w') as f:
            for key, value in sorted(self.variables.items()):
                # Add quotes if value contains spaces
                if ' ' in value:
                    f.write(f'{key}="{value}"\n')
                else:
                    f.write(f'{key}={value}\n')
```

**Key Details:**
- Preserves spaces in values by adding quotes
- Handles both single and double quotes
- Skips comments and empty lines
- Sorts variables alphabetically on save

### 2. Recursive File Scanner

Finds all `.env` files in a directory tree:

```python
class EnvScanner:
    """Scans directories for .env files"""

    @staticmethod
    def find_env_files(root_dir: Path, max_depth: int = 3) -> List[EnvFile]:
        """Find all .env files in directory tree"""
        env_files = []

        def scan_dir(path: Path, depth: int):
            if depth > max_depth:
                return

            try:
                for item in path.iterdir():
                    if item.is_file() and item.name == '.env':
                        env_file = EnvFile(path=item, variables={})
                        env_file.load()
                        env_files.append(env_file)
                    elif item.is_dir() and not item.name.startswith('.'):
                        scan_dir(item, depth + 1)
            except PermissionError:
                pass  # Skip directories we can't access

        scan_dir(root_dir, 0)
        return env_files
```

**Pattern:** Recursive directory traversal with depth limiting
- Skips hidden directories (starting with `.`)
- Handles permission errors gracefully
- Default max depth: 3 levels (prevents scanning entire filesystem)

### 3. Main Application Screen

Lists all found `.env` files with navigation:

```python
class EnvManagerApp(App):
    """Main application for managing environment variables"""

    CSS = """
    #file-list {
        height: 100%;
        border: solid green;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, scan_dir: Path):
        super().__init__()
        self.scan_dir = scan_dir
        self.env_files: List[EnvFile] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(f"Scanning {self.scan_dir} for .env files...", id="file-list"),
        )
        yield Footer()

    def on_mount(self) -> None:
        """Load .env files when app starts"""
        self.load_env_files()

    def load_env_files(self):
        """Scan for and load all .env files"""
        scanner = EnvScanner()
        self.env_files = scanner.find_env_files(self.scan_dir)

        file_list = self.query_one("#file-list", Static)

        if not self.env_files:
            file_list.update("No .env files found!")
        else:
            content = f"Found {len(self.env_files)} .env file(s):\n\n"
            for i, env_file in enumerate(self.env_files, 1):
                var_count = len(env_file.variables)
                content += f"{i}. {env_file.path}\n"
                content += f"   ({var_count} variable{'s' if var_count != 1 else ''})\n\n"

            content += "\nPress 1-9 to edit a file, or 'q' to quit"
            file_list.update(content)

    def on_key(self, event) -> None:
        """Handle number key presses to open files"""
        if event.character and event.character.isdigit():
            index = int(event.character) - 1
            if 0 <= index < len(self.env_files):
                self.push_screen(EditScreen(self.env_files[index]))
```

**Pattern:** Dynamic content loading in `on_mount()`
- Scan happens after UI is ready
- Number keys (1-9) navigate to edit screens
- Displays variable count for each file

### 4. Edit Screen

Edit variables for a specific `.env` file:

```python
class EditScreen(Screen):
    """Screen for editing a specific .env file"""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, env_file: EnvFile):
        super().__init__()
        self.env_file = env_file
        self.inputs: Dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="edit-container"):
            yield Label(f"Editing: {self.env_file.path}", id="file-path")

            with Vertical(id="variables"):
                for key, value in sorted(self.env_file.variables.items()):
                    with Horizontal(classes="var-row"):
                        yield Label(f"{key}:", classes="var-key")
                        input_widget = Input(
                            value=value,
                            placeholder=key,
                            classes="var-value",
                            name=key
                        )
                        self.inputs[key] = input_widget
                        yield input_widget

            with Horizontal(id="button-row"):
                yield Button("Save", variant="success", id="save-btn")
                yield Button("Add Variable", variant="primary", id="add-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.save_changes()
        elif event.button.id == "cancel-btn":
            self.app.pop_screen()
        elif event.button.id == "add-btn":
            self.app.push_screen(AddVariableScreen(self.env_file))

    def save_changes(self):
        """Save all changes to the .env file"""
        for key, input_widget in self.inputs.items():
            self.env_file.variables[key] = input_widget.value

        self.env_file.save()
        self.app.pop_screen()
```

**Pattern:** Input widget tracking
- Store references to Input widgets in `self.inputs` dict
- Retrieve values on save
- Use `Horizontal` containers for key-value rows

**CSS Styling:**

```css
#edit-container {
    padding: 2;
    height: 100%;
}

#file-path {
    color: $accent;
    text-style: bold;
    margin-bottom: 1;
}

.var-row {
    height: auto;
    margin-bottom: 1;
}

.var-key {
    width: 30;
    content-align: right middle;
    color: $success;
    text-style: bold;
}

.var-value {
    width: 1fr;
}

#variables {
    height: 1fr;
    overflow-y: auto;
}

#button-row {
    height: auto;
    dock: bottom;
    padding-top: 1;
}
```

**Key CSS Techniques:**
- Fixed width for labels (`width: 30`)
- Flexible width for inputs (`width: 1fr`)
- Scrollable variable list (`overflow-y: auto`)
- Docked button row at bottom

### 5. Add Variable Screen

Modal screen for adding new variables:

```python
class AddVariableScreen(Screen):
    """Screen for adding a new variable"""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Cancel"),
    ]

    def __init__(self, env_file: EnvFile):
        super().__init__()
        self.env_file = env_file

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="add-container"):
            yield Label("Add New Variable", id="add-title")
            yield Label("Variable Name:")
            yield Input(placeholder="MY_VAR", id="var-name")
            yield Label("Variable Value:")
            yield Input(placeholder="value", id="var-value")

            with Horizontal(id="button-row"):
                yield Button("Add", variant="success", id="add-submit")
                yield Button("Cancel", variant="default", id="add-cancel")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-submit":
            name_input = self.query_one("#var-name", Input)
            value_input = self.query_one("#var-value", Input)

            key = name_input.value.strip()
            value = value_input.value.strip()

            if key:
                self.env_file.variables[key] = value
                self.env_file.save()
                self.app.pop_screen()  # Close add screen
                self.app.pop_screen()  # Close edit screen (return to main)
        elif event.button.id == "add-cancel":
            self.app.pop_screen()
```

**Pattern:** Modal screen navigation
- Use `query_one()` to retrieve input values
- Double `pop_screen()` to return to main screen
- Validate key exists before saving

## Key Patterns

### 1. Multi-Screen Navigation

```python
# Main screen → Edit screen
self.push_screen(EditScreen(self.env_files[index]))

# Edit screen → Add screen (modal)
self.app.push_screen(AddVariableScreen(self.env_file))

# Add screen → Main screen (skip edit screen)
self.app.pop_screen()  # Close add screen
self.app.pop_screen()  # Close edit screen
```

### 2. Dynamic Input Generation

Create Input widgets dynamically from dictionary:

```python
self.inputs: Dict[str, Input] = {}

for key, value in sorted(self.env_file.variables.items()):
    input_widget = Input(value=value, placeholder=key, name=key)
    self.inputs[key] = input_widget
    yield input_widget
```

### 3. File I/O with Quote Handling

```python
# SAVE: Add quotes for values with spaces
if ' ' in value:
    f.write(f'{key}="{value}"\n')
else:
    f.write(f'{key}={value}\n')

# LOAD: Strip matched quotes
if value.startswith('"') and value.endswith('"'):
    value = value[1:-1]
elif value.startswith("'") and value.endswith("'"):
    value = value[1:-1]
```

### 4. Keyboard-Driven Navigation

```python
def on_key(self, event) -> None:
    """Handle number key presses"""
    if event.character and event.character.isdigit():
        index = int(event.character) - 1
        if 0 <= index < len(self.env_files):
            self.push_screen(EditScreen(self.env_files[index]))
```

## Usage

### Installation

```bash
pip install textual
```

### Basic Usage

```bash
# Scan current directory
python envman.py

# Scan specific directory
python envman.py /path/to/projects
```

### Workflow

1. Launch app - scans for `.env` files
2. Press `1-9` to edit a specific file
3. Edit values in input fields
4. Press "Save" to write changes
5. Press "Add Variable" to add new variables
6. Press `Escape` to go back
7. Press `q` to quit

## Development Challenges

From the [GitHub README](https://github.com/FyefoxxM/environment-variable-manager) (accessed 2025-11-02):

### Bug #1: DataTable Drama
**Problem:** Initially tried to use Textual's `DataTable` widget for displaying and editing variables. Spent 45 minutes reading docs before realizing DataTable doesn't support inline editing. You can select cells, but can't edit them directly.

**Solution:** Switched to individual `Input` widgets in `Horizontal` containers. Worked immediately.

### Bug #2: The 303 Line Problem
**Problem:** First version was 303 lines - over the 300 line project limit by 3 lines.

**Solution:** Stripped out docstrings and merged some imports. Final: 296 lines.

### Bug #3: Quote Handling
**Problem:** Forgot that `.env` files often use quotes for values with spaces. First version stripped ALL quotes, which broke values like `NAME="John Doe"`.

**Solution:** Added logic to only remove matched quotes and preserve the spaces inside:

```python
# Only strip if BOTH quotes match
if value.startswith('"') and value.endswith('"'):
    value = value[1:-1]
elif value.startswith("'") and value.endswith("'"):
    value = value[1:-1]
```

### Bug #4: CSS Typo Hell
**Problem:** Spent 20 minutes debugging why CSS wasn't applying to variable rows. Had `class="var-row"` in the widget but `.variable-row` in the CSS.

**Solution:** Changed CSS to `.var-row` to match the class name.

## Design Decisions

### Limitation: Max 9 Files
Only displays up to 9 `.env` files (single-digit keyboard navigation). Author's reasoning: "who has more than 9 projects open anyway?"

**Alternative approach:** Could use `Tree` widget for unlimited files with arrow key navigation.

### No DataTable for Variables
Despite DataTable being perfect for key-value data, it doesn't support inline editing. Individual Input widgets provide better UX for editing.

### Immediate Save
Changes are written to disk immediately on "Save" button press. No undo functionality. Trade-off: Simple implementation vs. safety.

### Max Depth: 3 Levels
Scanner stops after 3 directory levels. Prevents accidentally scanning entire filesystem if run from `/`.

## Practical Applications

**Use Cases:**
- Switching between dev/prod/staging settings
- Managing API keys across microservices
- Checking which environment variables each project uses
- Quick config changes without context switching
- Bulk updates to shared variables (DATABASE_URL, API_KEY)

**Workflow Example:**
```
You have 5 microservices, all using the same API key.
API key changes.
Instead of:
  1. Open VSCode
  2. Find project 1
  3. Open .env
  4. Edit
  5. Save
  6. Repeat 4 more times

Do this:
  1. python envman.py
  2. Press 1, edit, save
  3. Press 2, edit, save
  4. Press 3, edit, save
  5. Done in 30 seconds
```

## Extension Ideas

From the implementation, potential enhancements:

1. **Encryption Support** - Encrypt sensitive values (API keys, passwords)
2. **Git Integration** - Show which `.env` files are gitignored
3. **Template System** - Copy variables from one file to others
4. **Search/Filter** - Find files containing specific variables
5. **History/Undo** - Track changes with rollback
6. **Multi-line Values** - Support for values with newlines
7. **Validation** - Check for required variables per project type
8. **Export/Import** - Backup and restore variable sets

## Related Patterns

See also:
- [architecture/03-application-structure.md](../architecture/03-application-structure.md) - Screen-based navigation
- [widgets/05-input-textarea.md](../widgets/05-input-textarea.md) - Input widget usage
- [examples/04-xml-editor.md](04-xml-editor.md) - Similar file editing pattern

## Sources

**GitHub Repository:**
- [FyefoxxM/environment-variable-manager](https://github.com/FyefoxxM/environment-variable-manager) - Main repository (accessed 2025-11-02)
- [envman.py](https://github.com/FyefoxxM/environment-variable-manager/blob/main/envman.py) - Complete implementation (296 lines)

**Key Implementation Files:**
- `envman.py` - Complete application code
- `requirements.txt` - Just `textual` dependency
- `DEMO.md` - Demo walkthrough
- `PROJECT_STATS.md` - Project statistics

**License:** MIT
