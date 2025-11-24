# FileDrop - Drag and Drop Widget for Textual

**Repository**: [agmmnn/textual-filedrop](https://github.com/agmmnn/textual-filedrop)
**PyPI**: [textual-filedrop](https://pypi.org/project/textual-filedrop/)
**Accessed**: 2025-11-02

## Overview

`textual-filedrop` provides drag-and-drop file handling for Textual TUI applications. It enables users to drag files from their file manager directly into terminal apps, with automatic file parsing, icon display, and path extraction.

**Platform Support**: Windows and macOS (tested)
**Requirement**: Nerd Font for file icon display

## Key Features

- **Drag-and-drop support** - Accept files dragged from file managers into terminal
- **Directory handling** - Automatically walks directories and includes all files
- **File icons** - Displays appropriate icons using Nerd Fonts (based on lsd icon set)
- **Dual API** - Widget-based or function-based usage
- **File metadata** - Returns path, filename, extension, and icon for each file
- **Visual feedback** - Border color changes on focus/blur, file list display

## Installation

```bash
pip install textual-filedrop
```

Or from source:

```bash
git clone https://github.com/agmmnn/textual-filedrop.git
cd textual-filedrop
poetry install
```

## How It Works

### Terminal Drag-and-Drop Mechanism

Since Textual v0.10.0, the framework supports **bubble** events for paste operations. Many terminals treat drag-and-drop as a paste event, inserting the file path(s) as text. `textual-filedrop` intercepts these paste events and parses the file paths.

**Key insight**: Drag-and-drop in terminals is actually a paste event containing file paths.

### File Path Extraction

The widget handles platform-specific path formats:

**Windows**: Uses regex pattern to handle quoted paths and backslashes
```python
pattern = r'(?:[^\s"]|"(?:\\"|[^"])*")+'
```

**Unix/macOS**: Uses `shlex.split()` for proper shell-style parsing

```python
def _extract_filepaths(text: str) -> List[str]:
    split_filepaths = []
    if os.name == "nt":
        pattern = r'(?:[^\s"]|"(?:\\"|[^"])*")+'
        split_filepaths = re.findall(pattern, text)
    else:
        split_filepaths = shlex.split(text)

    # Clean null bytes and quotes
    filepaths = []
    for i in split_filepaths:
        item = i.replace("\x00", "").replace('"', "")
        if os.path.isfile(item):
            filepaths.append(i)
        elif os.path.isdir(item):
            # Walk directory and add all files
            for root, _, files in os.walk(item):
                for file in files:
                    filepaths.append(os.path.join(root, file))

    return filepaths
```

**Directory handling**: If a folder is dropped, the widget recursively walks it and includes all contained files.

## API Reference

### FileDrop Widget

The main focusable widget that displays drop zones and handles file events.

```python
from textual_filedrop import FileDrop

# Basic usage
yield FileDrop(id="filedrop")

# Focus the widget to activate
self.query_one("#filedrop").focus()
```

**Constructor Parameters**:
- `display: bool = True` - Show/hide widget
- `name: str = None` - Widget name
- `id: str = None` - Widget ID
- `classes: str = None` - CSS classes

**Default Styling**:
```css
FileDrop {
    border: round gray;
    height: 100%;
    background: $panel;
    content-align: center middle;
    padding: 0 3;
}
```

**Visual States**:
- **Focused**: Border changes to dodger blue
- **Blurred**: Border returns to gray
- **After drop**: Displays file icons and names in styled format

### FileDrop.Dropped Message

Custom message posted when files are dropped.

```python
class Dropped(Message):
    def __init__(
        self,
        path: str,          # Directory of first file
        filepaths: list,    # List of file paths
        filenames: list,    # List of filenames only
        filesobj: list,     # List of file objects (see below)
        oneline: str,       # Formatted display string
    ) -> None:
        ...
```

**File Object Structure**:
```python
{
    "path": "/full/path/to/file.txt",
    "name": "file.txt",
    "ext": "txt",
    "icon": "\uf15c"  # Nerd Font icon character
}
```

**Handler Example**:
```python
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    path = message.path              # "/home/user/documents"
    filepaths = message.filepaths    # ["/home/user/documents/file.txt", ...]
    filenames = message.filenames    # ["file.txt", ...]
    filesobj = message.filesobj      # [{"path": ..., "name": ..., "ext": ..., "icon": ...}, ...]
    oneline = message.oneline        # Formatted display string with icons
```

### getfiles() Function

Function-based API for use without the FileDrop widget. Processes paste events directly.

```python
from textual_filedrop import getfiles

def on_paste(self, event) -> None:
    files = getfiles(event)
    # Returns list of file objects
    # [{"path": ..., "name": ..., "ext": ..., "icon": ...}, ...]
```

**Parameters**:
- `event: events.Paste` - Textual paste event

**Returns**: `List[Dict[str, Any]]` - List of file objects

## Usage Patterns

### Pattern 1: Widget-Based (Recommended)

Full widget with visual feedback and focus management.

```python
from textual.app import App, ComposeResult
from textual_filedrop import FileDrop

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield FileDrop(id="filedrop")

    def on_mount(self) -> None:
        # Focus widget to activate drop handling
        self.query_one("#filedrop").focus()

    def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
        # Access dropped files
        for fileobj in message.filesobj:
            self.log(f"Dropped: {fileobj['name']} ({fileobj['ext']})")
            self.log(f"Path: {fileobj['path']}")
            self.log(f"Icon: {fileobj['icon']}")
```

### Pattern 2: Function-Based (Lightweight)

No widget required - handles paste events directly.

```python
from textual.app import App
from textual_filedrop import getfiles
from rich.json import JSON

class FileDropApp(App):
    def on_paste(self, event):
        files = getfiles(event)
        # Process files
        for f in files:
            self.log(f"File: {f['name']}, Ext: {f['ext']}")

        # Display as JSON
        files_json = str(files).replace("'", '"')
        self.query_one("#content").update(JSON(files_json))
```

### Pattern 3: Hidden Widget (Background)

Widget is invisible but remains active when focused.

```python
from textual_filedrop import FileDrop

class MyApp(App):
    def compose(self) -> ComposeResult:
        # Hidden widget - display: none
        yield FileDrop(display=False, id="hidden-drop")

    def on_mount(self) -> None:
        # Widget is invisible but will still catch drops when focused
        self.query_one("#hidden-drop").focus()

    def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
        # Handle files in background
        self.process_files(message.filesobj)
```

## Complete Example: Subdomain Lister

This example demonstrates advanced usage: parsing dropped text files, building tree structures, and dynamic UI updates.

**From**: [examples/subdomain_lister.py](https://github.com/agmmnn/textual-filedrop/blob/master/examples/subdomain_lister.py)

```python
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Tree
from textual.widgets.tree import TreeNode
from rich.text import Text
import tldextract
from textual_filedrop import FileDrop

class CombinerApp(App):
    DEFAULT_CSS = """
    Screen {
        align: center middle;
    }
    Tree {
        border: round $panel-lighten-2;
    }
    """

    def compose(self) -> ComposeResult:
        yield FileDrop(id="drop")
        yield Horizontal(classes="root")

    def on_mount(self):
        self.root = self.query_one(".root")
        self.root.styles.display = "none"  # Hidden initially
        self.drop = self.query_one("#drop")
        self.drop.focus()

    def on_file_drop_dropped(self, event: FileDrop.Dropped) -> None:
        # Show tree container
        self.root.styles.display = "block"

        # Remove old tree if exists
        try:
            self.query_one(".tree").remove()
            self.query_one(".url").remove()
        except:
            pass

        # Resize drop zone to top dock
        self.drop.styles.height = 7
        self.drop.styles.dock = "top"

        # Parse subdomain files
        filepaths = event.filepaths
        subs = []
        for filepath in filepaths:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    contents = f.read()
                    subs.extend(filter(lambda x: x != "", contents.split("\n")))
            except:
                print("Error reading file")

        # Build domain hierarchy
        result = {}
        extract = tldextract.TLDExtract()

        for subdomain_url in subs:
            item = extract(subdomain_url)
            domain = item.domain + "." + item.suffix
            sub = item.subdomain.split(".")
            subroot = sub[-1]
            sublevel = ".".join(sub[:-1])

            if domain not in result and domain:
                result[domain] = {}
            if sub != [""] and subroot:
                if subroot not in result[domain]:
                    result[domain][subroot] = []
                elif sublevel not in result[domain][subroot] and sublevel:
                    result[domain][subroot].append(sublevel)

        # Build tree widget
        tree: Tree[dict] = Tree("domain-tree", classes="tree")
        for domain in sorted(result.keys()):
            domain_node = tree.root.add(domain, expand=True)
            self.add_json(domain_node, result[domain], domain)

        tree.root.expand()
        tree.show_root = False
        tree.show_guides = True

        self.root.mount(tree)
        self.drop.mount(Static("", classes="url"))

    @classmethod
    def add_json(cls, node: TreeNode, json_data: object, root_name: str) -> None:
        """Recursively adds JSON data to tree nodes."""
        def add_node(name: str, node: TreeNode, data: object) -> None:
            if isinstance(data, dict):
                node._label = Text(f"🌐{name}")
                for key, value in data.items():
                    new_node = node.add("", expand=True)
                    add_node(key, new_node, value)
            elif isinstance(data, list):
                node._label = Text(f"📋{name}")
                for index, value in enumerate(data):
                    new_node = node.add("", expand=True)
                    add_node(str(index), new_node, value)
            else:
                node._allow_expand = False
                node._label = Text(data)

        add_node(root_name, node, json_data)

if __name__ == "__main__":
    app = CombinerApp()
    app.run()
```

**What This Example Shows**:
- Dynamic UI updates after file drop
- File reading and parsing from dropped files
- Building complex tree structures from data
- Resizing and repositioning widgets programmatically
- Using FileDrop with other Textual widgets (Tree, Static, Horizontal)

## Icon System

The widget includes a comprehensive icon mapping based on [Peltoche/lsd](https://github.com/Peltoche/lsd) icon theme.

**Icon Selection Logic**:
1. Check exact filename match (e.g., "README.md", "Dockerfile")
2. Check file extension (e.g., ".py", ".js", ".md")
3. Fall back to generic file icon

**Partial Icon Mapping**:
```python
ICONS = {
    # Exact filenames
    "dockerfile": "\uf308",
    "readme": "\ue609",
    "license": "\ue60a",
    "package.json": "\ue718",
    ".gitignore": "\uf1d3",

    # File extensions
    "py": "\ue606",   # Python
    "js": "\ue74e",   # JavaScript
    "md": "\ue609",   # Markdown
    "json": "\ue60b", # JSON
    "pdf": "\uf1c1",  # PDF
    "png": "\uf1c5",  # Image
    # ... 300+ more mappings
}
```

**Icon Rendering**:
Icons are rendered using Rich markup:
```python
oneline = " ".join([
    f'[on dodger_blue3] {i["icon"]} [/][on gray27]{i["name"]}[/]'
    for i in filesobj
])
```

## Implementation Details

### Event Flow

```
User drags file into terminal
    ↓
Terminal emits Paste event with file path(s)
    ↓
FileDrop widget catches Paste event (if focused)
    ↓
_extract_filepaths() parses platform-specific paths
    ↓
_build_filesobj() creates file metadata objects
    ↓
Widget posts FileDrop.Dropped message
    ↓
App handles on_file_drop_dropped()
```

### Focus Management

**Critical**: FileDrop widget MUST be focused to receive paste events.

```python
def on_mount(self) -> None:
    self.query_one("#filedrop").focus()
```

Without focus, the widget won't intercept drag-and-drop events.

### Reactive Text Display

The widget uses Textual's reactive system for text updates:

```python
class FileDrop(Widget):
    txt = reactive("Please Drag and Drop the files here...")

    def render(self) -> RenderableType:
        return self.txt
```

When files are dropped, `self.txt` is updated with the formatted file list, triggering automatic re-render.

### Border State Management

Focus states trigger visual feedback:

```python
async def on_event(self, event: events.Event) -> None:
    if isinstance(event, events.Focus):
        self.styles.border = ("round", "dodgerblue")
    elif isinstance(event, events.Blur):
        self.styles.border = ("round", "gray")
```

## Use Cases

### File Processing Applications

```python
# Batch image converter
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    for fileobj in message.filesobj:
        if fileobj['ext'] in ['png', 'jpg', 'jpeg']:
            self.convert_image(fileobj['path'])
```

### Log File Analyzer

```python
# Log aggregation tool
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    logs = []
    for fileobj in message.filesobj:
        if fileobj['ext'] == 'log':
            with open(fileobj['path']) as f:
                logs.extend(self.parse_log(f.read()))
    self.display_analysis(logs)
```

### Configuration Manager

```python
# Config file loader
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    for fileobj in message.filesobj:
        if fileobj['name'] in ['config.yaml', 'config.json']:
            self.load_config(fileobj['path'])
```

## Best Practices

### 1. Always Focus the Widget

```python
def on_mount(self) -> None:
    self.query_one("#filedrop").focus()
```

Without focus, drag-and-drop won't work.

### 2. Handle File Reading Errors

```python
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    for fileobj in message.filesobj:
        try:
            with open(fileobj['path'], 'r') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            self.log.error(f"Failed to read {fileobj['name']}: {e}")
```

### 3. Filter by Extension

```python
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    # Only process JSON files
    json_files = [f for f in message.filesobj if f['ext'] == 'json']
    if not json_files:
        self.notify("Please drop JSON files only")
        return
```

### 4. Provide User Feedback

```python
def on_file_drop_dropped(self, message: FileDrop.Dropped) -> None:
    count = len(message.filesobj)
    self.notify(f"Processing {count} file(s)...")
    # Process files
    self.notify("Complete!", severity="success")
```

## Limitations and Considerations

### Terminal Compatibility

Not all terminals support drag-and-drop or emit paste events. Compatibility varies:

- **Works**: iTerm2 (macOS), Windows Terminal, many modern terminals
- **May not work**: Some SSH sessions, older terminal emulators
- **Workaround**: Provide alternative file selection (e.g., file picker, manual path entry)

### File Path Edge Cases

The path extraction handles most cases, but edge cases exist:

- **Symlinks**: Resolved as absolute paths
- **Special characters**: Generally handled by shlex/regex
- **Network paths**: Platform-dependent (UNC paths on Windows, etc.)

### Performance with Many Files

Dropping folders with thousands of files will walk entire directory tree:

```python
elif os.path.isdir(item):
    for root, _, files in os.walk(item):  # Recursive walk
        for file in files:
            filepaths.append(os.path.join(root, file))
```

For large directories, consider adding progress indicators or file count limits.

## Sources

**GitHub Repository**:
- [agmmnn/textual-filedrop](https://github.com/agmmnn/textual-filedrop) - Main repository (accessed 2025-11-02)
- [textual_filedrop/_filedrop.py](https://github.com/agmmnn/textual-filedrop/blob/master/textual_filedrop/_filedrop.py) - Widget implementation
- [textual_filedrop/_getfiles.py](https://github.com/agmmnn/textual-filedrop/blob/master/textual_filedrop/_getfiles.py) - Function API
- [textual_filedrop/_icons.py](https://github.com/agmmnn/textual-filedrop/blob/master/textual_filedrop/_icons.py) - Icon mappings
- [examples/subdomain_lister.py](https://github.com/agmmnn/textual-filedrop/blob/master/examples/subdomain_lister.py) - Complete example
- [examples/without_widget.py](https://github.com/agmmnn/textual-filedrop/blob/master/examples/without_widget.py) - Function-based example

**Package**:
- [PyPI: textual-filedrop](https://pypi.org/project/textual-filedrop/) - Package repository

**Upstream Dependencies**:
- [Textualize/textual](https://github.com/Textualize/textual) - Textual framework (paste event support since v0.10.0)
- [Peltoche/lsd](https://github.com/Peltoche/lsd) - Icon theme source
