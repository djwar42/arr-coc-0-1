# Writing a Text Editor in 7 Minutes using Textual

**Source**: [Fronkan - Writing a Text Editor in 7 Minutes using Textual](https://fronkan.hashnode.dev/writing-a-text-editor-in-7-minutes-using-textual)
**Author**: Fredrik Sjöstrand
**Published**: June 3, 2025
**Accessed**: November 2, 2025

---

## Overview

A practical demonstration of rapid prototyping with Textual - building a functional text editor from scratch in under 7 minutes. This example showcases Textual's power for quick TUI development with minimal code.

**What You'll Build**:
- Directory tree navigation
- File opening and editing
- Syntax highlighting for Python files
- Keyboard shortcuts (Ctrl+S to save, Ctrl+Q to quit)

**Total Time**: ~7 minutes of live coding

---

## The Challenge

Author's scenario: Preparing a lightning talk, tried to open a Python file in Vim, discovered Vim wasn't installed. Question emerged: _"Could you live-code a text editor during a lightning talk, in 10 minutes or less?"_

**Answer**: Yes, with Textual.

---

## Full Code

Complete working text editor in approximately 60 lines:

```python
from argparse import ArgumentParser
from pathlib import Path

from textual.app import App
from textual.widgets import TextArea, DirectoryTree
from textual.containers import Horizontal

class Editor(App):
    BINDINGS = [
        ("ctrl+s", "save_file"),
        ("ctrl+q", "quit"),
    ]

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.file = None

    def compose(self):
        yield Horizontal(
            DirectoryTree(self.folder),
            TextArea("", id="editor")
        )

    def _on_directory_tree_file_selected(self, event):
        path: Path = event.path
        if not path.is_file():
            return

        self.file = path

        text_editor = self.query_one("#editor")
        text_editor.text = self.file.read_text()
        text_editor.language = "python" if self.file.suffix == ".py" else None

    def action_save_file(self):
        if self.file is None:
            return
        editor = self.query_one("#editor")
        self.file.write_text(editor.text)

if __name__ == "__main__":
    parser = ArgumentParser("My Editor")
    parser.add_argument("folder", type=Path)

    args = parser.parse_args()
    folder: Path = args.folder

    if not folder.exists():
        raise FileNotFoundError(f"No folder found for path: {folder}")
    if not folder.is_dir():
        folder = folder.parent

    app = Editor(folder)
    app.run()
```

**Dependencies**:
```txt
textual[syntax]  # Includes syntax highlighting support
```

**Usage**:
```bash
pip install textual[syntax]
python editor.py /path/to/folder
```

---

## Step-by-Step Breakdown

### Step 1: Scaffolding (30 seconds)

```python
from argparse import ArgumentParser
from pathlib import Path

from textual.app import App
from textual.widgets import TextArea, DirectoryTree
from textual.containers import Horizontal

class Editor(App):
    pass

if __name__ == "__main__":
    parser = ArgumentParser("My Editor")
    parser.add_argument("folder", type=Path)

    args = parser.parse_args()
    folder: Path = args.folder

    if not folder.exists():
        raise FileNotFoundError(f"No folder found for path: {folder}")
    if not folder.is_dir():
        folder = folder.parent

    app = Editor(folder)
    app.run()
```

**What This Does**:
- Empty `App` class
- Argument parsing for folder path
- Basic input validation (folder exists, handle file paths)
- If path is a file, use parent folder

### Step 2: Initialize Editor (1 minute)

```python
class Editor(App):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.file = None
```

**Critical Detail**: `super().__init__()` call is essential - Textual needs this to wire up the application properly.

**State Tracking**:
- `self.folder` - directory to browse
- `self.file` - currently open file (None initially)

### Step 3: Create UI Layout (2 minutes)

```python
class Editor(App):
    # ...
    def compose(self):
        yield Horizontal(
            DirectoryTree(self.folder),
            TextArea("", id="editor")
        )
```

**Layout**:
- `Horizontal` container - splits screen into two columns
- Left column: `DirectoryTree(self.folder)` - file/folder browser
- Right column: `TextArea("", id="editor")` - text editing area

**ID Assignment**: `id="editor"` allows querying this widget later

### Step 4: File Opening (3 minutes)

```python
class Editor(App):
    # ...
    def _on_directory_tree_file_selected(self, event):
        path: Path = event.path
        if not path.is_file():
            return

        self.file = path

        text_editor = self.query_one("#editor")
        text_editor.text = self.file.read_text()
        text_editor.language = "python" if self.file.suffix == ".py" else None
```

**Event Handling Pattern**: `_on_<object_type>_<event>`
- `DirectoryTree` object type
- `file_selected` event
- Combined: `_on_directory_tree_file_selected`

**File Loading**:
1. Verify path is a file (ignore directories)
2. Store path in `self.file` (needed for saving later)
3. Query for editor widget using `#editor` ID
4. Load file contents into text area
5. Enable Python syntax highlighting if `.py` file

**Syntax Highlighting**: Requires `textual[syntax]` extra dependency

### Step 5: Keyboard Shortcuts and Saving (5 minutes)

```python
class Editor(App):
    BINDINGS = [
        ("ctrl+s", "save_file"),
        ("ctrl+q", "quit"),
    ]
    # ...

    def action_save_file(self):
        if self.file is None:
            return
        editor = self.query_one("#editor")
        self.file.write_text(editor.text)
```

**Keyboard Bindings**:
- Class-level `BINDINGS` variable
- List of tuples: `(keyboard_shortcut, action_name)`
- `"quit"` action is built into Textual
- Custom actions use `action_<name>` method naming

**Save Implementation**:
1. Early return if no file open
2. Query editor widget
3. Write text content back to `self.file`

**Fun Note**: Author jokes about "beating Vim in quitability" with simple `Ctrl+Q` binding

---

## Features Demonstrated

### Rapid Development Power
- **60 lines of code** for a functional editor
- **7 minutes** of live coding time
- **Built-in features** from Textual:
  - Directory tree navigation
  - Text editing
  - Syntax highlighting
  - Keyboard shortcuts
  - Event system

### Core Textual Concepts

**1. Event-Driven Architecture**:
```python
def _on_directory_tree_file_selected(self, event):
    # Textual automatically calls this when file selected
    path: Path = event.path
```

**2. Widget Querying**:
```python
text_editor = self.query_one("#editor")  # CSS-like selector
```

**3. Action System**:
```python
BINDINGS = [("ctrl+s", "save_file")]  # Maps to action_save_file()
```

**4. Layout Containers**:
```python
yield Horizontal(widget1, widget2)  # Side-by-side layout
```

---

## Extended Version

Author created an improved version with additional features (not shown in full here):

**Enhancements**:
- Command palette for file opening
- Status bar with keyboard shortcuts
- Clock display in top-right corner
- Searchable, dynamically generated commands

**Command Palette**: Powerful Textual feature allowing searchable commands. In the improved editor, it generates a list of files in the directory that can be opened via command search.

**Full Code**: Available at [this gist](https://gist.github.com/Fronkan/1f2d8abbef05b39aa5766413c1f4ce6e#file-editor_improved-py)

---

## Key Insights: Why This Is Fast

### 1. Built-In Widgets Are Feature-Rich
- `DirectoryTree`: Complete file browser, no configuration needed
- `TextArea`: Full editing capabilities + syntax highlighting
- No need to implement basic functionality from scratch

### 2. Declarative UI Composition
```python
def compose(self):
    yield Horizontal(DirectoryTree(...), TextArea(...))
```
Single method describes entire layout - no manual positioning, sizing, or rendering

### 3. Convention-Based Event Handling
Method names automatically wire to events - no manual event listener registration:
```python
def _on_directory_tree_file_selected(self, event):
    # Textual knows this handles DirectoryTree.FileSelected events
```

### 4. Action System Reduces Boilerplate
```python
BINDINGS = [("ctrl+s", "save_file")]
def action_save_file(self):
    # Automatically bound - no keyboard event handling code
```

### 5. Smart Widget Querying
CSS-like selectors make finding widgets trivial:
```python
self.query_one("#editor")  # Like document.querySelector("#editor")
```

---

## Practical Applications

**Use This Pattern For**:
- Quick internal tools for file editing/viewing
- Configuration file editors
- Log file browsers
- Code snippet managers
- Note-taking applications

**Development Speed**:
- Prototype in minutes
- Add features incrementally
- Textual handles terminal complexity
- Focus on application logic, not TUI plumbing

---

## Challenge: Can You Beat 7 Minutes?

Author's question: Now that you know how it's done, can you:
1. Write it faster than 7 minutes?
2. Extend it with interesting features?
3. Apply this pattern to other terminal tools?

**Recommendation**: Experiment with Textual - "It's a powerful library and really, really fun to play with."

---

## Sources

**Original Article**:
- [Writing a Text Editor in 7 Minutes using Textual](https://fronkan.hashnode.dev/writing-a-text-editor-in-7-minutes-using-textual) - Fredrik Sjöstrand (June 3, 2025)

**Code Gists**:
- [Basic Editor](https://gist.github.com/Fronkan/1f2d8abbef05b39aa5766413c1f4ce6e#file-editor-py)
- [Improved Editor with Command Palette](https://gist.github.com/Fronkan/1f2d8abbef05b39aa5766413c1f4ce6e#file-editor_improved-py)
- [Dependencies](https://gist.github.com/Fronkan/1f2d8abbef05b39aa5766413c1f4ce6e#file-requirements-txt)

**Tags**: Python, TUI, Terminal, Textual, Rapid Prototyping

---

## Related Topics

- Command palette implementation (see improved editor)
- Syntax highlighting configuration
- File system navigation patterns
- Keyboard shortcut best practices
- Terminal application UX design
