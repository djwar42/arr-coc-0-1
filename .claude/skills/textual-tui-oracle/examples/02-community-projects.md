# Community Projects - Notable Textual Applications

**Date**: 2025-11-02
**Source**: awesome-textualize-projects, transcendent-textual, written-in-textual repositories

---

## Overview

The Textual community has produced numerous production-quality applications demonstrating the framework's versatility. This document profiles the most notable projects across key categories: developers tools, data/file viewers, chat applications, terminal utilities, and specialized domain applications.

---

## Top Production Applications

### Developer Tools & IDEs

#### 1. **Harlequin** - SQL IDE for Terminal

[GitHub](https://github.com/tconbeer/harlequin) | 560+ stars

A terminal-based SQL IDE built for interactive database exploration.

**Key Features**:
- DuckDB, Postgres, SQLite support
- Query execution and result display
- File browser integration
- Syntax highlighting for SQL

**Textual Patterns Used**:
- `DataTable` widget for query results
- `Container` for multi-pane layout (editor + results)
- Custom command palette
- Async query execution with `Worker`

**Use Case**: Database engineers, data analysts working in remote environments

---

#### 2. **Trogon** - Click CLI to TUI Converter

[GitHub](https://github.com/Textualize/trogon) | 2.2k+ stars (Official Textualize)

Automatically converts Click CLI applications into interactive terminal UIs.

**Key Features**:
- Introspects Click command structure
- Generates TUI with form inputs
- Parameter validation
- Maintains CLI functionality

**Textual Patterns Used**:
- Dynamic widget generation from command specs
- Form building with validation
- Dialog boxes for confirmations
- Custom bindings for CLI compatibility

**Use Case**: DevOps engineers wanting to wrap CLI tools with GUI-like interfaces

---

#### 3. **Django-TUI** - Django Management Interface

[GitHub](https://github.com/anze3db/django-tui) | 160+ stars

TUI for inspecting and running Django commands.

**Key Features**:
- Browse Django models
- Execute management commands
- View command output
- Interactive parameter entry

**Textual Patterns Used**:
- Tree view for model hierarchy
- List views with selections
- Modal dialogs for parameters
- Real-time command output display

**Use Case**: Django developers needing quick access to management commands

---

### File & Data Viewers

#### 4. **Browsr** - Terminal File Explorer

[GitHub](https://github.com/juftin/browsr) | 120+ stars

Pleasant file explorer with remote filesystem support.

**Key Features**:
- Local and remote filesystem browsing
- Preview pane
- Search and filtering
- Directory tree navigation

**Textual Patterns Used**:
- `DirectoryTree` widget
- Split layout (tree + preview)
- Custom bindings for navigation
- Async file operations

**Use Case**: System administrators, developers needing quick file navigation

---

#### 5. **Frogmouth** - Markdown Browser

[GitHub](https://github.com/Textualize/frogmouth) | 2k+ stars (Official Textualize)

Interactive Markdown document viewer for the terminal.

**Key Features**:
- Render Markdown with formatting
- Table of contents navigation
- Code block display
- Link handling

**Textual Patterns Used**:
- Custom widget for Markdown rendering
- Scrollable container for large documents
- Dynamic layout with TOC sidebar
- Custom key bindings for navigation

**Use Case**: Documentation readers, blog browsing in terminal

---

#### 6. **Kupo** - Terminal File Browser

[GitHub](https://github.com/darrenburns/kupo) | 150+ stars

Fast, featureful file browser by Darren Burns.

**Key Features**:
- Blazing fast directory traversal
- File preview
- Fuzzy search
- Customizable keybindings

**Textual Patterns Used**:
- Efficient tree rendering
- Async file operations
- Custom CSS for theming
- Command palette pattern

**Use Case**: Power users, system administrators

---

### Data Applications

#### 7. **Dolphie** - MySQL Monitoring TUI

[GitHub](https://github.com/charles-001/dolphie) | 290+ stars

Real-time MySQL monitoring and inspection tool.

**Key Features**:
- Live query statistics
- Connection monitoring
- Performance metrics
- Database inspection

**Textual Patterns Used**:
- Real-time data refresh with Workers
- Multiple DataTable instances
- Tab-based layout for different views
- Color-coded status indicators

**Use Case**: MySQL DBAs, system operators

---

#### 8. **Toolong** - Log File Viewer

[GitHub](https://github.com/Textualize/toolong) | Official Textualize

Terminal application for viewing, tailing, merging log files.

**Key Features**:
- View/tail multiple logs
- Merge timelines
- Search and filter
- JSONL support

**Textual Patterns Used**:
- Multi-window management
- Real-time file tailing
- Custom scrolling
- Merged timeline display

**Use Case**: DevOps, system administrators debugging issues

---

### Chat & Communication

#### 9. **Elia** - Terminal ChatGPT Client

[GitHub](https://github.com/darrenburns/elia) | 250+ stars

ChatGPT client built with Textual.

**Key Features**:
- Interactive chat interface
- Conversation history
- Code block highlighting
- Token counting

**Textual Patterns Used**:
- Text input with multiline support
- Message display with Rich formatting
- Scrollable chat history
- Async API calls with streaming

**Use Case**: AI enthusiasts, developers exploring ChatGPT

---

### Terminal Utilities & Games

#### 10. **Textual Paint** - Paint in Terminal

[GitHub](https://github.com/1j01/textual-paint) | 840+ stars

MS Paint-like drawing application in your terminal.

**Key Features**:
- Drawing tools (pencil, brush, fill)
- Shape tools
- Color palette
- Canvas manipulation

**Textual Patterns Used**:
- Custom canvas widget (character-based)
- Tool palette sidebar
- Real-time render updates
- Mouse/keyboard input handling

**Use Case**: Fun project demonstrating Textual capabilities

---

#### 11. **Dooit** - Todo Manager TUI

[GitHub](https://github.com/kraanzu/dooit) | 1.8k+ stars

Awesome TUI todo manager with powerful features.

**Key Features**:
- Task creation and management
- Nested tasks
- Due dates and priorities
- Search and filtering

**Textual Patterns Used**:
- Tree view for nested tasks
- Modal dialogs for task input
- Rich formatting for display
- Key command handling

**Use Case**: Productivity enthusiasts, developers

---

#### 12. **Termtyper** - Terminal Typing Application

[GitHub](https://github.com/kraanzu/termtyper) | 980+ stars

Typing practice application to improve speed.

**Key Features**:
- Typing tests with random words
- Speed/accuracy metrics
- Theme support
- Statistics tracking

**Textual Patterns Used**:
- Real-time input handling
- Timer management with Workers
- Stats display with Rich tables
- Custom CSS theming

**Use Case**: Developers, typers wanting to improve speed

---

### Specialized Domain Applications

#### 13. **RecoverPy** - File Recovery Tool

[GitHub](https://github.com/PabloLec/RecoverPy) | 1.1k+ stars

Interactively find and recover deleted files from terminal.

**Key Features**:
- Search for deleted files
- Preview before recovery
- Batch recovery
- Hexdump viewer

**Textual Patterns Used**:
- Search results in DataTable
- File preview widget
- Progress indication
- Hexdump display

**Use Case**: System administrators, data recovery specialists

---

#### 14. **NoteSH** - Sticky Notes Application

[GitHub](https://github.com/Cvaniak/NoteSH) | 400+ stars

Fully functional sticky notes app in your terminal.

**Key Features**:
- Create and manage notes
- Search notes
- Rich text formatting
- Save to file

**Textual Patterns Used**:
- List view for notes
- Text editor widget
- Modal dialogs
- File I/O operations

**Use Case**: Terminal enthusiasts, note-taking workflows

---

#### 15. **Baca** - TUI Ebook Reader

[GitHub](https://github.com/wustho/baca) | 260+ stars

Terminal-based ebook reader with formatting support.

**Key Features**:
- Read EPUB/PDF files
- Chapter navigation
- Bookmarks and progress
- Text formatting

**Textual Patterns Used**:
- Scrollable content area
- Sidebar for navigation
- Status bar for progress
- File selection dialog

**Use Case**: Readers preferring terminal environments

---

## Common Patterns Across Projects

### 1. **Data Display Pattern**

Most applications use `DataTable` or custom widgets for displaying structured information:

```
Projects: Harlequin, Dolphie, Toolong, RecoverPy
Pattern: DataTable for tabular results with pagination/scrolling
```

### 2. **Multi-Pane Layout Pattern**

Split views combining navigation + content display:

```
Projects: Browsr, Kupo, Frogmouth, RecoverPy
Pattern: Container with horizontal/vertical split, separate content areas
```

### 3. **Async Operations Pattern**

Long-running operations handled with `Worker` threads:

```
Projects: Elia, Dolphie, Harlequin, Toolong
Pattern: Worker for API calls, file operations, database queries
```

### 4. **Command Palette Pattern**

Quick command/action access via searchable interface:

```
Projects: Kupo, Browsr, Trogon
Pattern: Modal dialog with filterable command list
```

### 5. **Modal Dialog Pattern**

Input collection and confirmations:

```
Projects: Dooit, NoteSH, Trogon, Django-TUI
Pattern: Modal containers with input validation and buttons
```

### 6. **Real-Time Update Pattern**

Live data refresh with periodic updates:

```
Projects: Dolphie, Toolong, Termtyper
Pattern: Set_interval with scheduled updates, avoid blocking UI
```

### 7. **Tree Navigation Pattern**

Hierarchical data exploration:

```
Projects: Browsr, Django-TUI, Dooit
Pattern: TreeControl/DirectoryTree for nested structures
```

### 8. **Rich Text Formatting Pattern**

Terminal-friendly text rendering:

```
Projects: Frogmouth, Elia, Baca, Toolong
Pattern: Use Rich library for formatted output, ANSI support
```

---

## Integration Examples

### API Integration
- **Elia**: OpenAI API with streaming responses
- **Harlequin**: Database APIs (DuckDB, Postgres)
- **Trogon**: Click CLI introspection

### File System Integration
- **Browsr**: Remote filesystem support (SFTP, etc.)
- **RecoverPy**: Low-level file recovery
- **Baca**: EPUB/PDF parsing

### Real-World Data Sources
- **Dolphie**: Live MySQL connections
- **Toolong**: Log file tailing
- **Dooit**: Task persistence (JSON/YAML)

---

## Key Learnings from Community

### 1. **UX Patterns That Work**
- File browsers work well (Browsr, Kupo)
- Data viewers are popular (Harlequin, Dolphie)
- Chat-like interfaces feel natural (Elia)

### 2. **Performance Considerations**
- Large datasets require pagination/lazy loading
- Async operations prevent UI freezing
- File operations should be non-blocking

### 3. **Widget Selection Matters**
- `DataTable` for structured data
- `TreeControl` for hierarchies
- `Input` with custom validation for forms
- `RichLog` for dynamic content

### 4. **User Experience Wins**
- Command palettes for discoverability
- Keyboard shortcuts for power users
- Visual feedback for long operations
- Sensible defaults with customization

### 5. **Architecture Patterns**
- Separate model/view logic
- Use Rich for rendering
- Workers for background tasks
- Event-driven command handling

---

## Creating Production Textual Apps

### Quality Indicators

**From Community Projects**:

1. **CLI Integration** - Wrap existing tools (Trogon pattern)
2. **Real Data Sources** - Connect to actual APIs/databases
3. **Polished UX** - Keyboard shortcuts, mouse support
4. **Testing** - Use pytest-textual-snapshot
5. **Documentation** - Clear README and usage examples

### Getting Started

**Study successful projects**:
1. Start with simple viewer (Browsr, Frogmouth)
2. Move to interactive tool (Harlequin, Dooit)
3. Add API integration (Elia, Trogon)
4. Scale with performance patterns

---

## Resources

### Project Collections

**Related Curated Lists**:
- [awesome-textualize-projects](https://github.com/oleksis/awesome-textualize-projects) - 93 stars, comprehensive list
- [transcendent-textual](https://github.com/davep/transcendent-textual) - 185 stars, archived but comprehensive
- [written-in-textual](https://github.com/matan-h/written-in-textual) - 130+ projects with ratings

### Finding More Projects

- [GitHub topic: textual](https://github.com/topics/textual)
- [PyPI search: textual](https://pypi.org/search/?q=textual)
- [Textualize Discord](https://discord.gg/Enf6Z3qhVr)

---

## Sources

**Source Repositories**:
- [awesome-textualize-projects](https://github.com/oleksis/awesome-textualize-projects) - Curated list (93 stars)
- [davep/transcendent-textual](https://github.com/davep/transcendent-textual) - Comprehensive index (185 stars, archived 2024-06-22)
- [matan-h/written-in-textual](https://github.com/matan-h/written-in-textual) - Best-of ratings (130 projects)

**Project Links**:
- Harlequin: https://github.com/tconbeer/harlequin
- Trogon: https://github.com/Textualize/trogon
- Frogmouth: https://github.com/Textualize/frogmouth
- Browsr: https://github.com/juftin/browsr
- Kupo: https://github.com/darrenburns/kupo
- Dolphie: https://github.com/charles-001/dolphie
- Toolong: https://github.com/Textualize/toolong
- Elia: https://github.com/darrenburns/elia
- Textual Paint: https://github.com/1j01/textual-paint
- Dooit: https://github.com/kraanzu/dooit
- Termtyper: https://github.com/kraanzu/termtyper
- RecoverPy: https://github.com/PabloLec/RecoverPy
- NoteSH: https://github.com/Cvaniak/NoteSH
- Baca: https://github.com/wustho/baca
- Django-TUI: https://github.com/anze3db/django-tui

**Accessed**: 2025-11-02
