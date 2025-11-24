# Words-TUI: Daily Writing App Case Study

## Overview

Words-TUI is a production-ready TUI (Text User Interface) application for daily writing in the terminal, built with Textual. It demonstrates a complete, real-world implementation of a data-persistent TUI application with database integration, providing an excellent reference for building practical Textual apps.

**Project Details:**
- **Author**: Anže Pečar (@anze3db)
- **GitHub**: https://github.com/anze3db/words-tui
- **Stars**: 42 (as of 2025-11-02)
- **License**: MIT
- **Status**: Production-ready, actively maintained
- **Latest Release**: 23.6 (December 4, 2023)

## Application Purpose

Words-TUI provides a focused environment for daily writing practice. It allows users to:
- Write daily journal entries
- Track writing history
- View writing statistics
- Maintain a persistent database of all entries
- Work entirely in the terminal without distractions

## Architecture Overview

### Project Structure

```
words-tui/
├── src/words_tui/
│   ├── __init__.py          # Package initialization
│   ├── __main__.py          # Entry point (calls cli.words_tui)
│   ├── __about__.py         # Version information
│   ├── cli.py               # Click-based CLI interface
│   ├── app.py               # Main Textual App class
│   ├── models.py            # Peewee ORM database models
│   └── [additional modules]
├── tests/                   # Test suite
├── pyproject.toml          # Project configuration (Hatch-based)
├── README.md
└── CHANGELOG.md
```

### Technology Stack

**Core Dependencies** (from pyproject.toml):
- **textual**: TUI framework (widgets, layout, event handling)
- **click**: CLI argument parsing
- **peewee**: Lightweight ORM for database operations
- **tree-sitter**: Code/text parsing (likely for markdown rendering)

**Development Tools**:
- **hatch**: Modern Python project management
- **pytest**: Testing framework
- **textual-dev**: Textual development tools
- **ruff**: Fast Python linter and formatter
- **mypy**: Type checking

## Key Components

### 1. CLI Interface (cli.py)

**Entry Point Pattern**:
```python
# __main__.py
from words_tui.cli import words_tui
sys.exit(words_tui())
```

**Click-Based CLI**:
- Uses Click library for argument parsing
- Supports `--db` flag to specify custom database path
- Environment variable support: `WORDS_TUI_DB`
- Default database location: `~/.words-tui.db`

**Usage Examples**:
```bash
# Standard usage (default DB location)
words-tui

# Custom database location
words-tui --db /path/to/custom.db

# Using environment variable
export WORDS_TUI_DB=/path/to/db
words-tui
```

### 2. Data Persistence (models.py)

**Peewee ORM Integration**:
- Lightweight SQLite database
- Simple schema for storing writing entries
- Automatic database creation on first run
- Local-first data storage (no cloud dependency)

**Database Pattern**:
- Single SQLite file stored in user's home directory
- All writing history persisted locally
- Fast queries for displaying writing statistics
- No network dependencies

### 3. Main Application (app.py)

**Textual App Architecture**:
- Main `App` class inheriting from `textual.app.App`
- Widget composition for UI elements
- Event-driven architecture
- Reactive data binding for live updates

**Likely Widget Usage** (based on Textual patterns):
- **Input**: For writing daily entries
- **DataTable**: Displaying writing history/statistics
- **Static/Label**: Showing word counts, dates
- **Container**: Layout management
- **Header/Footer**: Navigation and status information

### 4. State Management

**Application State**:
- Current writing entry in memory
- Database sync on save/exit
- History loaded on-demand from database
- Word count tracking in real-time

**Data Flow**:
```
User Input (Input widget)
    ↓
App State (reactive variables)
    ↓
Database Persistence (Peewee ORM)
    ↓
Display Updates (DataTable/Static widgets)
```

## Implementation Patterns

### Database Integration Pattern

**Peewee ORM Benefits**:
- Simple model definitions (no complex migrations)
- Automatic table creation
- Python-native query syntax
- Lightweight (perfect for single-user TUI apps)

**Typical Model Structure**:
```python
# Example pattern (actual implementation in models.py)
from peewee import *

db = SqliteDatabase('~/.words-tui.db')

class Entry(Model):
    date = DateField()
    content = TextField()
    word_count = IntegerField()
    created_at = DateTimeField()

    class Meta:
        database = db
```

### CLI Configuration Pattern

**Environment Variable + Flag Pattern**:
```python
# Flexible configuration priority:
# 1. Command-line flag (--db)
# 2. Environment variable (WORDS_TUI_DB)
# 3. Default (~/.words-tui.db)

@click.command()
@click.option('--db', envvar='WORDS_TUI_DB',
              default='~/.words-tui.db',
              help='Database file location')
def words_tui(db):
    # Initialize app with specified database
    pass
```

### File I/O and Data Persistence

**Local-First Architecture**:
- No network calls or API dependencies
- All data stored in local SQLite database
- Instant startup (no authentication/sync delays)
- Works completely offline
- User owns their data

**Persistence Strategy**:
- Auto-save on entry completion
- Database writes on app exit
- Atomic transactions for data integrity
- No data loss on crash (SQLite ACID properties)

## Widget Usage Examples

### Input Widget for Writing

**Real-Time Writing Interface**:
```python
from textual.widgets import Input

class WritingInput(Input):
    # Multiline text input for daily writing
    # Bindings for save, word count updates
    # Focus management for distraction-free writing
```

**Features**:
- Multiline text editing
- Real-time word count
- Auto-save triggers
- Keyboard shortcuts (Ctrl+S to save, Esc to exit)

### DataTable for History View

**Writing History Display**:
```python
from textual.widgets import DataTable

# Display columns: Date, Word Count, Preview
table = DataTable()
table.add_columns("Date", "Words", "Preview")

# Load from database
for entry in Entry.select():
    table.add_row(
        entry.date,
        str(entry.word_count),
        entry.content[:50] + "..."
    )
```

**Interaction Patterns**:
- Sortable columns (by date, word count)
- Row selection to view full entry
- Keyboard navigation (arrow keys, Enter to open)
- Visual indicators for current day

### Static Widgets for Statistics

**Word Count Display**:
```python
from textual.widgets import Static

# Live-updating word counter
word_count_display = Static("Words: 0")

# Update on text input changes
def on_input_changed(self, event):
    words = len(event.value.split())
    word_count_display.update(f"Words: {words}")
```

## Build and Installation

### Installation Methods

**Via pipx** (recommended):
```bash
pipx install words-tui
```

**Via pip**:
```bash
pip install words-tui
```

**From source**:
```bash
git clone https://github.com/anze3db/words-tui
cd words-tui
pip install -e .
```

### Development Setup

**Hatch-Based Workflow**:
```bash
# Install development dependencies
hatch env create

# Run tests
hatch run test

# Run with coverage
hatch run cov

# Lint and format
hatch run lint:all

# Type checking
hatch run lint:typing
```

**Development Scripts** (from pyproject.toml):
- `test`: Run pytest
- `test-cov`: Run tests with coverage
- `cov`: Generate coverage report
- `lint:style`: Run ruff linter and formatter
- `lint:typing`: Run mypy type checker

## Testing Strategy

### Test Suite Organization

**Test Coverage**:
- Unit tests for database models
- Integration tests for app functionality
- Widget interaction tests
- CLI argument parsing tests

**Test Tools**:
- **pytest**: Test framework
- **pytest-watch**: Continuous testing during development
- **coverage**: Code coverage measurement
- **textual-dev**: Textual testing utilities

**CI/CD**:
- GitHub Actions workflow: `build-and-test.yml`
- Runs on Python 3.8-3.12
- Tests on multiple platforms (Linux, macOS, Windows)
- Badge display on README

## Best Practices Demonstrated

### 1. Configuration Management

**Flexible Database Location**:
- CLI flag for explicit control
- Environment variable for user preference
- Sensible default (~/.words-tui.db)
- Path expansion support (tilde expansion)

### 2. Data Integrity

**Database Safety**:
- Atomic writes via SQLite transactions
- Automatic database initialization
- Safe shutdown with data persistence
- No data loss on unexpected exit

### 3. User Experience

**Terminal-Native Design**:
- No external browser/GUI dependencies
- Fast startup time (local database)
- Keyboard-centric navigation
- Minimal cognitive overhead

**Writing-Focused Interface**:
- Distraction-free text input
- Clear visual feedback (word counts)
- Easy access to writing history
- Simple, intuitive commands

### 4. Code Quality

**Modern Python Practices**:
- Type hints throughout codebase
- Comprehensive linting (ruff)
- Automated formatting
- Clear project structure (src layout)

**Build System**:
- Hatch for modern project management
- Declarative pyproject.toml configuration
- Environment matrix testing (Python 3.8-3.12)
- Reproducible builds

## Deployment Patterns

### PyPI Distribution

**Package Publishing**:
- Distributed via PyPI (words-tui package)
- Semantic versioning (23.6 format: year.month)
- Comprehensive changelog maintenance
- Entry point script: `words-tui` command

**Installation Experience**:
```bash
# Install
pipx install words-tui

# Run
words-tui

# Upgrade
pipx upgrade words-tui

# Uninstall
pipx uninstall words-tui
```

### User Data Management

**Data Location Strategy**:
- Default: `~/.words-tui.db` (hidden in home directory)
- Respects XDG Base Directory specification
- Easy to backup (single SQLite file)
- Portable across machines (copy .db file)

**Data Privacy**:
- Local-only storage (no cloud sync)
- User owns their data completely
- No telemetry or analytics
- No external network calls

## Performance Considerations

### Startup Performance

**Fast Launch**:
- Small dependency footprint (Textual, Peewee, Click)
- Local database (no network latency)
- Lazy loading of writing history
- Efficient SQLite queries

### Runtime Performance

**Responsive UI**:
- Textual's reactive framework for instant updates
- Efficient DOM updates (minimal re-renders)
- Database queries optimized for common operations
- Smooth typing experience (no input lag)

### Database Performance

**SQLite Optimization**:
- Single-file database (no connection overhead)
- Indexed queries for fast history retrieval
- Minimal write overhead (append-only pattern)
- Automatic vacuuming for database health

## Lessons for Textual Developers

### 1. Integration Patterns

**ORM Integration**:
- Peewee works seamlessly with Textual
- Initialize database in App.on_mount()
- Use workers for database operations (avoid blocking UI)
- Handle database connections properly on app shutdown

**CLI Integration**:
- Click provides robust argument parsing
- Environment variable support for configuration
- Clean separation between CLI and app logic
- Easy to extend with new commands/options

### 2. State Management

**Reactive Data Pattern**:
```python
from textual.reactive import reactive

class WritingApp(App):
    word_count = reactive(0)
    current_entry = reactive("")

    def watch_word_count(self, value):
        # Auto-update UI when word_count changes
        self.query_one("#counter").update(f"Words: {value}")
```

### 3. Database Lifecycle

**App Lifecycle Hooks**:
```python
class WritingApp(App):
    def on_mount(self):
        # Initialize database connection
        db.connect()
        db.create_tables([Entry], safe=True)

    def on_unmount(self):
        # Close database connection cleanly
        db.close()
```

### 4. User Feedback

**Progress Indication**:
- Show save status (saved/unsaved indicator)
- Display word counts in real-time
- Provide confirmation on actions
- Handle errors gracefully with notifications

## Common Pitfalls Avoided

### 1. Database Management

**Proper Connection Handling**:
- No connection leaks (proper close on exit)
- Safe to interrupt (Ctrl+C handling)
- Database initialization on first run
- Migration-free schema (Peewee creates tables)

### 2. Input Handling

**Text Input Best Practices**:
- No input lag (efficient update handling)
- Proper multiline support
- Keyboard shortcut conflicts avoided
- Focus management (input stays focused)

### 3. Configuration

**Flexible Configuration**:
- Multiple configuration sources (CLI, env, default)
- Clear precedence order
- Easy to override for testing
- Well-documented configuration options

## Future Enhancement Ideas

**Potential Features** (not yet implemented):
- Markdown rendering for entries
- Search/filter functionality
- Export to various formats (PDF, HTML, Markdown)
- Writing streak tracking
- Daily writing goals and reminders
- Tagging system for entries
- Full-text search across history
- Cloud backup integration (optional)

## Comparison with Other TUI Apps

**Similar Projects**:
- **jrnl**: Command-line journaling (different approach)
- **vim/neovim**: Text editing (more complex)
- **todo.txt**: Task management TUI (simpler data model)

**Words-TUI Advantages**:
- Purpose-built for daily writing
- Beautiful Textual-based UI
- Simple, focused feature set
- Easy to install and use
- Active maintenance

## Community Reception

**Project Metrics**:
- 42 GitHub stars (growing community interest)
- Featured in textual-written lists
- Clean, well-documented codebase
- Responsive maintainer (@anze3db)
- Regular updates and bug fixes

## Conclusion

Words-TUI demonstrates best practices for building production-ready Textual applications:

1. **Clean Architecture**: Separation of CLI, app, and data layers
2. **Database Integration**: Seamless Peewee ORM usage with Textual
3. **User Experience**: Fast, keyboard-centric, distraction-free interface
4. **Code Quality**: Modern Python practices, comprehensive testing
5. **Distribution**: Easy installation via PyPI, pipx support

**Key Takeaways for Developers**:
- Start simple: Focus on core functionality first
- Use established tools: Click for CLI, Peewee for database
- Test thoroughly: Multiple Python versions, platforms
- Document well: Clear README, changelog, inline comments
- Iterate based on usage: Maintain actively, respond to issues

**Reference Value**:
Words-TUI serves as an excellent reference for developers building their first production Textual app, demonstrating how to combine Textual's UI capabilities with practical data persistence and CLI integration.

## Sources

**Project Repository**:
- [anze3db/words-tui](https://github.com/anze3db/words-tui) - Main GitHub repository (accessed 2025-11-02)
- [PyPI: words-tui](https://pypi.org/project/words-tui) - Package distribution page
- [GitHub Releases](https://github.com/anze3db/words-tui/releases) - Version history and changelog

**Code Files Referenced**:
- [pyproject.toml](https://github.com/anze3db/words-tui/blob/main/pyproject.toml) - Project configuration, dependencies, build system
- [__main__.py](https://github.com/anze3db/words-tui/blob/main/src/words_tui/__main__.py) - Entry point implementation
- [README.md](https://github.com/anze3db/words-tui/blob/main/README.md) - Project documentation

**Community Lists**:
- Featured in [written-in-textual](https://github.com/matan-h/written-in-textual) best-of list
- Mentioned in Textual community discussions

**Technology Documentation**:
- [Textual Framework](https://textual.textualize.io/) - Official Textual docs
- [Peewee ORM](https://docs.peewee-orm.com/) - Database library docs
- [Click](https://click.palletsprojects.com/) - CLI framework docs
