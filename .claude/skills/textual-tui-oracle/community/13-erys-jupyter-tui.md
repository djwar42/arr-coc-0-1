# Erys: Terminal Interface for Jupyter Notebooks

## Overview

**Erys** is a sophisticated terminal user interface (TUI) built with Textual that enables users to open, create, edit, run, and save Jupyter Notebooks directly in the terminal. It also functions as a lightweight text editor for Python, Markdown, YAML, and other file formats.

The project bridges the gap between notebook-based development (which typically requires browser access) and terminal-based workflows, making it ideal for remote work, SSH sessions, and developers who prefer keyboard-driven interfaces.

## Project Information

**Repository**: [natibek/erys](https://github.com/natibek/erys)
**License**: Apache-2.0
**Original URL**: https://bit.ly/38DjRvH
**Expanded URL**: https://github.com/natibek/erys (accessed 2025-11-02)
**PyPI Package**: [erys](https://pypi.org/project/erys/)
**Latest Version**: v0.2.10 (October 2025)
**Stars**: 125+ on GitHub

## Key Technologies

**Textual Integration**:
- Uses Textual for the complete terminal UI rendering
- Custom widgets for notebook cells, code editor, directory tree
- Rich text rendering for markdown, syntax highlighting, and output display
- Modal dialogs for file operations (save-as, open)

**Jupyter Integration**:
- `jupyter_client` for kernel management and code execution
- Supports Python code execution with isolated kernel managers per notebook
- Each notebook has its own kernel to prevent environment leakage between notebooks
- Code execution interruption support
- Output rendering for code cells (Markdown, JSON, images, HTML)

**Development Stack**:
- Python 100% (pure Python project)
- Requires `ipykernel` for code execution capability
- Installation via `uv` (recommended) or `pipx`

## Core Features

### Notebook Management
- Create new Jupyter notebooks from scratch
- Open existing notebooks in multiple formats (supports various NB format versions)
- Save notebooks using format version 4.5
- Directory tree widget for easy file navigation
- Save-as functionality with input validation and directory browsing

### Code Execution
- Execute Python code cells with kernel-based execution
- Interrupt running code cells with button control
- Each notebook maintains isolated kernel environment
- Code cells require `ipykernel` in the Python environment

### Cell Operations
**Cell Manipulation**:
- Add cells before/after current cell (`a`, `b` keys)
- Split cells at cursor position (Ctrl+Backslash)
- Join cells with adjacent cells (Ctrl+PageUp/PageDown)
- Merge multiple selected cells (holding Ctrl for multi-select)
- Toggle between code and markdown cells (`t` key)
- Delete cells with undo support (max 20 undo stack)
- Move cells up/down (Ctrl+Up/Down)
- Copy/Cut/Paste cells with unique ID generation

**Cell Presentation**:
- Collapse cells to save screen space
- Collapse code cell outputs separately

### Rendering & Output
- **Markdown**: Rendered with Textual's Markdown widget
- **JSON**: Pretty-printed with Textual's Pretty widget
- **Errors**: Rich text display with ANSI escape conversion
- **Images**: PNG output rendering outside terminal (via Pillow)
- **HTML**: Rendered in default browser via `webbrowser` module
- **Plain Text Output**: Accessible via focus for copying

### Syntax Highlighting
- Python and Markdown syntax highlighting via Textual
- Extended syntax support via `tree-sitter` library for:
  - Python, Markdown, YAML, Bash
  - CSS, Go, HTML, Java, JavaScript, JSON, Rust, TOML, XML

### File Editing
- Edit text files with syntax highlighting
- Lightweight alternative to full-featured editors
- Supports multiple file formats

## Installation & Usage

### Installation Methods

**Using uv (recommended)**:
```bash
uv tool install erys
```

**Using pipx**:
```bash
pipx install erys
```

### Basic Usage

**Launch empty notebook**:
```bash
erys
```

**Open existing notebooks/files**:
```bash
erys notebook.ipynb
erys file1.py file2.md
erys new_notebook.ipynb  # Creates if parent dir exists
```

**SSH with rendering support**:
- Use X11 forwarding: `ssh -X user@host`
- Images and HTML render in separate windows

## Key Bindings

### Application Level
| Binding | Function |
|---------|----------|
| Ctrl+N | New Notebook |
| Ctrl+K | Close Notebook |
| Ctrl+L | Clear All Tabs |
| D | Toggle Directory Tree |
| Ctrl+Q | Quit Application |

### Notebook Level
| Binding | Function |
|---------|----------|
| A | Add Cell After |
| B | Add Cell Before |
| T | Toggle Cell Type |
| Ctrl+D | Delete Cell |
| Ctrl+U | Undo Delete (max 20) |
| Ctrl+Up | Move Cell Up |
| Ctrl+Down | Move Cell Down |
| M | Merge Selected Cells |
| Ctrl+C | Copy Cell |
| Ctrl+X | Cut Cell |
| Ctrl+V | Paste Cell |
| Ctrl+S | Save Notebook |
| Ctrl+W | Save As |

### Cell Level
| Binding | Function |
|---------|----------|
| C | Collapse Cell |
| Ctrl+PageUp | Join with Previous Cell |
| Ctrl+PageDown | Join with Next Cell |
| Ctrl+R | Run Code Cell (in code cells) |
| Ctrl+\ | Split Cell at Cursor |

## Architecture & Design Patterns

### Kernel Architecture
- **Isolated Kernels**: Each notebook spawns its own kernel manager and client
- **Same Environment**: All notebooks execute in the same Python environment (the one running Erys)
- **No Environment Leakage**: Kernel isolation prevents variable/import contamination between notebooks

### UI Architecture
- **Directory Tree Widget**: Left-side file browser with keyboard navigation
- **Tabbed Interface**: Multiple notebooks open as tabs
- **Cell-based Editing**: Modal focus handling (escape to blur text area, enter to focus)
- **Save Dialog Modal**: Custom modal for "Save As" with filename validation

### Output Rendering Strategy
- **Terminal-renderable**: Markdown, JSON, errors, plain text
- **External Rendering**: PNG images (Pillow), HTML (browser)
- **Rich Text Format**: ANSI escape sequences converted to Textual markup

## Real-World Integration: Jupyter + Textual

### Why Textual for Notebooks?
1. **Keyboard-first workflow**: Terminal-native for power users
2. **Remote accessibility**: SSH-friendly (with X11 for rendering)
3. **Lightweight**: No browser overhead
4. **Custom interactions**: Textual enables unique cell manipulation workflows
5. **Rich terminal rendering**: Unicode, colors, widgets

### Code Execution Flow
```
User Input (Ctrl+R)
    ↓
Cell content extracted
    ↓
Kernel client executes code
    ↓
Output captured (stdout, errors, results)
    ↓
Output type detected (markdown, JSON, image, HTML)
    ↓
Rendered appropriately (terminal or external)
    ↓
Display in UI with focus support
```

### Unique Cell Workflows
- **Merge multiple cells**: Select with Ctrl+click, merge with M
- **Split at cursor**: Place cursor, press Ctrl+Backslash
- **Reorder cells**: Arrow keys for navigation, Ctrl+Up/Down to move
- **Copy/Paste with new IDs**: Prevent kernel confusion with duplicate cell IDs

## Coming Features

- Execution time duration display for code cells
- Configuration file support for themes and key bindings
- Attach to user-selected kernels (beyond auto-spawned)
- Raw cell type support
- Saving progress backups
- Open from cached backup
- Save on exit prompt
- Extended MIME output type rendering

## Community & Reception

**Community Response**:
- Featured on Reddit's r/Python and r/commandline communities (July-August 2025)
- Highlighted in PyCoder's Weekly and Python news aggregators
- Recognized by the Textual framework community as exemplar integration
- LinkedIn mentions and discussions about terminal-based notebook interfaces

**Use Cases Identified**:
- Remote development over SSH
- Data science workflows in terminal environments
- Lightweight Jupyter alternative for resource-constrained systems
- Quick notebook prototyping without GUI overhead
- Integration with terminal-based development workflows

## Learning Path

**Getting Started**:
1. Install via `uv tool install erys` or `pipx`
2. Ensure `ipykernel` in Python environment: `pip install ipykernel`
3. Launch: `erys`
4. Use directory tree (D key) to navigate to existing notebooks

**Explore Features**:
1. Create new notebook (Ctrl+N)
2. Add cell (A), write Python code
3. Execute code (Ctrl+R when in code cell)
4. Practice cell manipulation (split, merge, move)
5. Try different output types (code producing images, markdown, errors)

**Advanced Workflows**:
1. Open multiple notebooks simultaneously
2. Use SSH with X11 forwarding for image/HTML rendering
3. Build repetitive analysis notebooks
4. Develop Python scripts inline with text editing
5. Use as lightweight alternative to Jupyter Lab for terminal workflows

## Textual Integration Lessons

**Widget Reusability**:
- Directory tree (complex hierarchical widget)
- Custom cell containers with multiple sub-widgets
- Modal dialogs for user input
- Rich output rendering with Textual's built-in widgets

**Keyboard-Driven Design**:
- Vim-inspired bindings for notebook operations
- Modal focus states for different interaction modes
- Efficient multi-key combinations for frequent operations

**Performance Considerations**:
- Responsive UI during kernel communication
- Non-blocking cell execution (with interrupt capability)
- Efficient rendering of large notebooks with collapsible cells

## Sources

**GitHub Repository**:
- [natibek/erys](https://github.com/natibek/erys) - Main project repository
- README with comprehensive documentation
- 160+ commits tracking development history
- 14 releases from initial concept to v0.2.10

**Community Coverage**:
- [Erys: A Terminal Interface for Jupyter Notebooks](https://www.reddit.com/r/Python/comments/1ma0852/erys_a_terminal_interface_for_jupyter_notebooks/) - Reddit r/Python discussion (20+ comments, July 2025)
- [Erys – A TUI for Jupyter Notebooks](https://www.pythonpapers.com/p/erys-a-tui-for-jupyter-notebooks) - The Python Papers article by Mike Driscoll (September 2025)
- [Introducing Erys: A Terminal Interface for Jupyter Notebooks](https://www.linkedin.com/posts/nati-bekele_python-jupyter-terminalui-activity-7353495897697079296-YBwq) - LinkedIn post by Nathnael B., creator (July 2025)

**Related Resources**:
- [Textual Documentation](https://textual.textualize.io/) - Framework used for UI
- [Jupyter Client Documentation](https://jupyter-client.readthedocs.io/) - Kernel management library
- [PyPI: erys](https://pypi.org/project/erys/) - Package distribution

**Project Metadata**:
- License: Apache-2.0
- Language: Python 100%
- Latest Release: v0.2.10 (October 2025)
- Repository Created: 160 commits, active development

**Access Date**: November 2, 2025
**Original Shortened URL**: https://bit.ly/38DjRvH
