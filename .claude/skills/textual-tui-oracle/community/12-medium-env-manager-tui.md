# Environment Variable Manager TUI: Real-World Application

## Overview

A practical Textual-based terminal UI application that solves the real problem of managing `.env` files across multiple projects. Instead of opening 12 editors to change one API key, this tool provides a single unified interface.

**Application Purpose**: Scan project directories for `.env` files, display them in a unified view, and allow inline editing with immediate persistence.

**Author**: Jason Dookeran
**Published**: October 2025
**Code Size**: 297 lines of Python (intentionally kept under 300 line limit)
**Development Time**: 6 hours
**Dependencies**: Single dependency - `textual >= 0.40.0`

From [Environment Variable Manager Article](https://jdookeran.medium.com/environment-variable-manager-stop-opening-12-editors-to-change-one-api-key-e6bfbac951db) (Medium, accessed 2025-11-02)

## The Problem Statement

**Real-world workflow inefficiency:**
- Multiple projects, each with `.env` files
- Developers edit these files 10+ times per day
- Current workflow: Open editor → navigate to project → find file → change value → close → repeat
- Alternative solutions (Doppler, Vault, Shell aliases) are either overkill or inefficient
- No solution for: "I just need to change PORT=3000 to PORT=3001"

## Architecture Overview

### Three-Component Design

**1. Scanner Module** - Directory traversal
- Walks directory tree up to configurable depth (default: 3 levels)
- Finds all `.env` files across projects
- Skips hidden directories (`.git`, `.env.local` variants)
- Handles permission errors gracefully
- Returns list of EnvFile objects with loaded variables

From article (Scanner section):
```python
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

**Key Design Decision**: Manual recursive scanning instead of `Path.rglob()` allows depth limit control and prevents scanning into `node_modules` or `.git` (which hung the application for 30 seconds on monorepos).

**2. Parser Module** - `.env` format handling
- Parses key-value pairs with proper handling of edge cases
- Supports comments, quoted values, spaces in values
- Preserves equals signs in values (e.g., connection strings)
- Strips matched quotes only (preserves spaces inside values)
- Re-adds quotes on save when values contain spaces

From article (Parser section):
```python
def load(self):
    """Load variables from the .env file"""
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
                key, value = line.split('=', 1)  # Only split on first =
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                self.variables[key] = value
```

**Critical Detail**: `line.split('=', 1)` limits split to first equals sign only. Values like `DATABASE_URL=postgres://localhost:5432/db?option=value` require this to work correctly.

**3. Textual UI Layer** - Interactive screen management
- Three-screen application: listing, editing, adding variables
- Keyboard-driven navigation (1-9 to select files, ESC/Q to navigate)
- No mouse required (terminal-native interaction model)
- CSS-based styling (Textual's familiar approach)

## Textual Features Used

### Widget Composition Pattern

From article (Part 3 section):
```python
def compose(self) -> ComposeResult:
    yield Header()
    yield Container(
        Static(f"Scanning {self.scan_dir} for .env files...", id="file-list"),
    )
    yield Footer()

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
```

### Input Widget Handling (Edit Screen)

Critical lesson: Don't use `DataTable` for editable content. Use individual `Input` widgets instead.

From article (Part 4 section):
```python
def compose(self) -> ComposeResult:
    yield Header()

    with Container(id="edit-container"):
        yield Label(f"Editing: {self.env_file.path}", id="file-path")

        with Vertical(id="variables"):
            for key, value in sorted(self.env_file.variables.items()):
                with Horizontal(classes="var-row"):
                    yield Label(f"{key}:", classes="var-key")
                    input_widget = Input(value=value, placeholder=key,
                                       classes="var-value", name=key)
                    self.inputs[key] = input_widget
                    yield input_widget

        with Horizontal(id="button-row"):
            yield Button("Save", variant="success", id="save-btn")
            yield Button("Add Variable", variant="primary", id="add-btn")
            yield Button("Cancel", variant="default", id="cancel-btn")

    yield Footer()
```

**Important**: `Input` widget's `name` property is read-only. Must pass in constructor: `Input(..., name=key)`. Cannot set after creation.

### Styling with CSS

From article (Part 6 section):
```css
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
```

**Design Pattern**: Use consistent class naming between Textual code and CSS. Class names are the contract between both.

### Screen Management (Implied)

UI Flow:
1. Main screen lists all `.env` files
2. Press 1-9 to open one (triggers screen push)
3. Edit screen shows all variables as input fields
4. Change values, press Save (triggers screen pop)
5. ESC goes back (screen pop), Q quits (exit)

## Bug Analysis & Solutions

### Bug #1: DataTable Not Editable (45 minutes wasted)

**Problem**: Textual's `DataTable` widget is read-only. You can select and style cells but not edit them inline.

**Wrong Approach**:
- Started with DataTable assumption
- Spent 45 minutes reading docs, trying different approaches
- Realized it simply wasn't designed for inline cell editing

**Solution**: Use individual `Input` widgets per row instead.

**Lesson**: Read widget documentation thoroughly BEFORE building, not after.

### Bug #2: Quote Handling in Parser (30 minutes)

**Problem**: Initial parser stripped ALL quotes from values, losing information:
```
# In .env file:
SECRET_KEY="my secret key with spaces"

# After parsing:
SECRET_KEY=my secret key with spaces  # Lost the quotes!

# After saving:
SECRET_KEY=my secret key with spaces  # Invalid .env format
```

**Solution**:
- Only strip matched quotes (quotes at both start AND end)
- Preserve spaces inside values
- Re-add quotes on save if value contains spaces

**Code**:
```python
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

**Lesson**: `.env` format edge cases will bite you. Test with real-world messy files.

### Bug #3: CSS Class Typo (20 minutes)

**Problem**: Widget code uses `classes="var-row"` but CSS uses `.variable-row` (different name).

**Symptom**: Styles simply don't apply. No error message.

**Solution**: Copy-paste class names exactly. Don't trust character-level accuracy by hand.

**Lesson**: When CSS doesn't work, check for typos first. Always.

### Bug #4: Read-Only Name Property (5 minutes)

**Problem**: Tried to set Input widget's name after creation:
```python
input_widget = Input(...)
input_widget.name = key  # ❌ AttributeError: property 'name' has no setter
```

**Solution**: Pass name in constructor:
```python
input_widget = Input(..., name=key)  # ✓
```

**Lesson**: When a property errors with "no setter," you can't set it after initialization. Check the constructor parameters.

### Bug #5: Code Size Optimization (Annoying)

**Problem**: First complete version was 303 lines, exceeding 300 line limit by 3 lines.

**Solution**:
- Removed docstrings (kept key comments)
- Merged imports
- Condensed whitespace

**Final**: 297 lines

**Lesson**: 300 lines is tighter than you think. Plan accordingly.

## Design Patterns & Decisions

### Why Textual Instead of Alternatives?

**vs Web UI (Flask/Django)**:
- No context switching for terminal-native developers
- No port management, no server startup
- Runs immediately with just Python

**vs Tkinter**:
- Modern async/await architecture
- Better widget composition system
- Native terminal styling

**vs Shell aliases** (`alias editenv="code ~/project/.env"`):
- Doesn't solve the "which file?" problem across 12 projects
- Still requires context switching to editor

### Keyboard-First Interaction

- Numeric selection (1-9) for quick file access
- ESC to go back
- Q to quit
- No mouse interaction required
- Developers live in terminal anyway

### Limitation-Aware Design

**Known Limitations**:
1. Only shows 9 files max (press 1-9 limitation)
   - Counter: Who has 9+ projects in active use simultaneously?
2. No encryption (values stored in plain text)
   - Counter: `.env` files are already plain text on disk
3. No git integration (doesn't auto-commit changes)
   - Counter: This is an editor, not a git client
4. No multi-line value support
   - Counter: Rare edge case, sign to use YAML/JSON instead
5. Only looks for `.env` (not `.env.local`, `.env.production`)
   - Counter: Easy to add in V2, intentional for MVP

**Philosophy**: Solve the core problem (editing multiple `.env` files) without feature creep.

## Testing Approach

**Test Setup**: Three test projects with varying `.env` complexity

**Project A** (basic):
```
DATABASE_URL=postgresql://localhost/projecta
API_KEY=abc123xyz
DEBUG=True
SECRET_KEY="my secret key with spaces"
PORT=3000
```

**Project B** (AWS credentials):
```
DATABASE_URL=mysql://localhost/projectb
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
NODE_ENV=development
```

**Project C** (nested subdirectory):
```
APP_NAME=SubProject
ENVIRONMENT=production
LOG_LEVEL=info
```

**Test Results**:
- Core scanning: ✓ Found all 3 files
- Parser edge cases: ✓ Handled quotes, spaces, special chars
- Save/load functionality: ✓ Changes persisted correctly
- Recursive directory scanning: ✓ Worked with depth limits

**Real-World Usage**: Used tool to update database URLs across 4 actual projects. Saved estimated 5 minutes of directory navigation per session.

## Time Investment & ROI

**Time to Build**: 6 hours breakdown
- 2 hours: Core scanner and parser
- 1.5 hours: Main TUI screens
- 1.5 hours: Edit screen and input handling
- 45 minutes: DataTable experimentation (wasted)
- 30 minutes: Bug fixing (quotes, CSS, name property)

**Usage Frequency**: 10+ times per week

**Time Saved Per Use**: 30-60 seconds (vs opening editor + navigation)

**ROI Timeline**: 2 weeks to break even, massive savings thereafter

**Key Insight**: The best tools are the ones you actually use, not the ones with the most features.

## Implementation Highlights

### Directory Scanning Strategy

Key decision: Manual recursive scanning with depth limit instead of `Path.rglob()`:
- Prevents accidental scanning into `node_modules`, `.git`
- Respects depth limits
- Handles permission errors gracefully
- More efficient for typical use cases

### Value Preservation

Parser maintains formatting information needed for round-trip editing:
- Quotes on values with spaces
- Comment preservation (implicit through line-by-line parsing)
- Alphabetical sorting on save (intentional normalization)
- Edge cases like equals signs in connection strings

### Screen State Management

Textual handles screen push/pop elegantly:
- Main screen persists while editing screen is on top
- ESC automatically returns to main screen
- Input values collected from all fields on save
- Changes applied to in-memory objects, then persisted to disk

## Broader Principles

**Three Key Lessons**:

1. **Terminal UIs Are Underrated**
   - No context switching for terminal-native developers
   - Keyboard-first interaction aligns with dev workflows
   - Textual makes building them stupid easy
   - Fast to ship, runs anywhere

2. **Read Documentation First**
   - Save 45 minutes by reading DataTable docs before building
   - Verify widgets do what you need BEFORE starting
   - Check property setters/getters in constructors

3. **Small Tools Can Be Genuinely Useful**
   - 297 lines solving a real daily problem
   - Not revolutionary, but immediately valuable
   - ROI measured in weeks, not months
   - Best tools solve problems you actually experience

## GitHub Repository

From article: Available at GitHub (referenced as ["GitHub Repository"](https://github.com/FyefoxxM/environment-variable-manager))

## Usage Instructions

```bash
# Install
pip install textual

# Run (scan current directory)
python envman.py

# Run (scan specific directory)
python envman.py ~/projects
```

**Keyboard Controls**:
- `1-9`: Edit a file
- `Type`: Edit input fields
- `Save`: Persist changes
- `Add Variable`: Create new key-value pair
- `ESC`: Go back
- `Q`: Quit

## Files in Project

- `envman.py` - Main application (297 lines)
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- `DEMO.md` - Visual UI representation
- `PROJECT_STATS.md` - Build statistics
- `test_envman.py` - Testing script
- `examples.sh` - Usage examples

## Connection to Textual Concepts

### Textual Features Demonstrated

1. **Widget Composition** - Building UI from composable parts
2. **Container Layouts** - Vertical/Horizontal/Container for structure
3. **CSS Styling** - Professional appearance with minimal effort
4. **Input Widgets** - Collecting and managing user input
5. **Screen Management** - Navigation between UI states (implicit through screen push/pop)
6. **Event Handling** - Button actions, keyboard input
7. **Dynamic Updates** - `Static.update()` for changing content
8. **Layout System** - `1fr` width units, responsive design

### Anti-Patterns Avoided

1. **DataTable for editable content** - Wrong widget for the job
2. **Setting read-only properties** - Understanding API contracts
3. **CSS class name mismatches** - Exact string correspondence required
4. **Over-engineering scope** - MVP approach (no encryption, no git integration)

## Sources

**Primary Source:**
- [Environment Variable Manager Article](https://jdookeran.medium.com/environment-variable-manager-stop-opening-12-editors-to-change-one-api-key-e6bfbac951db) by Jason Dookeran, Medium, October 2025 (accessed 2025-11-02)

**Related Resources:**
- [GitHub Repository](https://github.com/FyefoxxM/environment-variable-manager) - Source code for the application
- Textual Framework (as dependency, `textual >= 0.40.0`)

**Part of Series:**
- "30 Tools in 30 Days" challenge by Jason Dookeran
- Day 3: Environment Variable Manager
- Day 4: Port Killer (upcoming)
