# Environment Variable Manager: Production TUI Case Study

**Last Updated**: 2025-11-02
**Source**: [Environment Variable Manager: Stop Opening 12 Editors to Change One API Key](https://jdookeran.medium.com/environment-variable-manager-stop-opening-12-editors-to-change-one-api-key-e6bfbac951db) by Jason Dookeran (accessed 2025-11-02)

## Overview

Real-world production TUI built with Textual in 297 lines of Python. Solves a common developer pain point: managing `.env` files across multiple projects without context switching between editors.

**Problem Solved**: Developers working on multiple projects need to frequently update environment variables (API keys, database URLs, debug flags) scattered across different `.env` files. Traditional solutions require opening editors, navigating directories, and manual file management.

**Solution**: Single terminal interface that scans project directories, lists all `.env` files, and provides inline editing capabilities with immediate save functionality.

## Architecture

### Three-Layer Design

**1. Scanner Layer** - Directory traversal and `.env` file discovery
**2. Parser Layer** - `.env` format handling (quotes, comments, edge cases)
**3. TUI Layer** - Textual-based interface with multiple screens

### Component Breakdown

```
EnvScanner
├── find_env_files() - Recursive directory scanning (max depth: 3)
└── EnvFile objects with parsed variables

EnvFile
├── load() - Parse .env format into key-value pairs
├── save() - Write back with quote preservation
└── variables: Dict[str, str]

Textual App
├── MainScreen - File listing and selection
├── EditScreen - Variable editing with Input widgets
└── AddVariableScreen - New variable creation
```

## Implementation Deep Dive

### 1. Directory Scanner with Depth Limiting

**Key Pattern**: Manual recursive traversal with depth control to avoid scanning `node_modules`, `.git`, and other deep directories.

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

**Why Not `rglob()`?**

Initial attempt used `Path(root_dir).rglob('.env')` - simple one-liner. Failed because:
- No depth limit control
- Scans into `node_modules` and `.git` (hung for 30+ seconds on monorepos)
- No permission error handling

**Lesson**: Manual approach provides necessary control for production reliability.

### 2. Robust .env Parser

**Critical Edge Cases**:
- Comments (inline and full-line)
- Quoted values with spaces: `SECRET_KEY="my secret key"`
- Equals signs in values: `CONNECTION_STRING=Server=localhost;Database=db`
- Empty values: `DEBUG=`
- Mixed quote styles (single vs double)

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

**Critical Detail**: `line.split('=', 1)` - Split on FIRST equals sign only. Prevents breaking values containing `=` like database connection strings.

**Quote Handling Bug**: Initial implementation stripped ALL quotes, breaking values on save. Fix: Only remove matched quotes (present at both start AND end), then re-add quotes on save for values containing spaces.

### 3. Save with Quote Preservation

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

**Pattern**: Automatic quote insertion for values with spaces ensures valid `.env` format. Sorts keys alphabetically for consistent output.

## Textual Widget Usage Patterns

### Failed Approach: DataTable Widget

**Initial Plan**: Use `DataTable` for spreadsheet-like key-value display.

**Problem Discovered**: DataTable is **read-only**. You can select cells but cannot edit them inline.

**Time Lost**: 45 minutes trying to make DataTable editable before reading documentation thoroughly.

**Lesson**: Read widget documentation BEFORE building, not after encountering limitations.

### Working Solution: Individual Input Widgets

**Pattern**: Dynamic widget composition with dictionary tracking.

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

**Key Implementation Details**:

**1. Widget Tracking**: `self.inputs[key] = input_widget` - Store reference to each Input widget for save operation
**2. Read-Only Name Property**: `Input(..., name=key)` - Must pass `name` to constructor, cannot set after creation
**3. Layout Containers**: `Horizontal` for label-input pairs, `Vertical` for stacking variables
**4. Button Variants**: `success`, `primary`, `default` - Built-in semantic styling

### Critical Bug: Read-Only Properties

```python
# ❌ WRONG - AttributeError: property 'name' has no setter
input_widget = Input(...)
input_widget.name = key

# ✓ CORRECT - Pass to constructor
input_widget = Input(..., name=key)
```

**Lesson**: Textual widgets often have read-only properties that must be set during construction.

## State Management Pattern

### Save Operation Flow

```python
def save_changes(self):
    """Save all changes to the .env file"""
    # Collect values from all input widgets
    for key, input_widget in self.inputs.items():
        self.env_file.variables[key] = input_widget.value

    # Write to disk
    self.env_file.save()

    # Return to file list
    self.app.pop_screen()
```

**Pattern**:
1. Widget state → Application state (`input_widget.value` → `env_file.variables`)
2. Application state → File system (`env_file.save()`)
3. Screen management (`pop_screen()`)

**No Intermediate State**: Changes written directly to disk on Save. No undo, no staging area. Simple and predictable.

## UI Flow Design

### Screen Navigation

```
MainScreen (File Listing)
    ↓ [Press 1-9]
EditScreen (Variable Editing)
    ↓ [Press "Add Variable"]
AddVariableScreen (New Variable Form)
    ↓ [Save]
EditScreen (Updated List)
    ↓ [ESC]
MainScreen
```

**Keyboard-First Navigation**:
- **1-9 keys**: Select files (max 9 files displayed)
- **ESC**: Go back to previous screen
- **Q**: Quit application
- **Tab**: Navigate between input fields
- **Enter**: Activate buttons

### Main Screen Implementation

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

**Pattern**: Use `query_one()` to find widget by ID, then `update()` to change content dynamically.

## CSS Styling (Lessons Learned)

### Working CSS

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

**Semantic Colors**: `$success`, `$warning`, `$error` - Textual provides theme-aware color variables

**Fractional Units**: `width: 1fr` - CSS Grid-style flexible sizing

### Critical CSS Bug: Class Name Typo

**Code**: `Horizontal(classes="var-row")`
**CSS**: `.variable-row { ... }`

**Different names** → Styles didn't apply.

**Time Lost**: 20 minutes debugging before spotting typo.

**Lesson**: Copy-paste class names. Don't trust character-level accuracy when typing.

### CSS Alignment Patterns

```css
content-align: right middle;  /* Horizontal + Vertical */
text-align: center;           /* Text-specific alignment */
align: center middle;         /* Widget alignment in container */
```

**Pattern**: Textual uses multi-axis alignment properties similar to CSS Flexbox/Grid.

## File I/O Patterns

### Safe File Writing

```python
def save(self):
    """Save variables back to the .env file"""
    with open(self.path, 'w') as f:
        for key, value in sorted(self.variables.items()):
            if ' ' in value:
                f.write(f'{key}="{value}"\n')
            else:
                f.write(f'{key}={value}\n')
```

**Pattern**: Complete file overwrite (not append). Ensures consistent formatting.

**Risk**: No backup or atomic write. If process crashes during write, file could be corrupted.

**Production Improvement Needed**:
```python
# Write to temp file first
temp_path = self.path.with_suffix('.env.tmp')
with open(temp_path, 'w') as f:
    # Write content...

# Atomic rename
temp_path.replace(self.path)
```

### Permission Error Handling

```python
try:
    for item in path.iterdir():
        # Process files...
except PermissionError:
    pass  # Skip directories we can't access
```

**Pattern**: Silent failure for permission errors during scanning. Prevents crashes when encountering protected directories.

**Alternative**: Log skipped directories for user awareness.

## Testing Approach

### Test Data Structure

```
test-projects/
├── project-a/
│   └── .env (5 variables including quoted spaces)
├── project-b/
│   └── .env (5 variables including AWS keys)
└── project-c/
    └── subdir/
        └── .env (3 variables, nested discovery test)
```

**Test Cases Covered**:
1. Multi-project scanning
2. Nested directory discovery (depth 3)
3. Quoted values with spaces
4. Values containing special characters (AWS keys)
5. Save/load round-trip preservation

### Manual Testing Flow

```bash
# Scan test directory
$ python envman.py test-projects

# Expected output
Found 3 .env files:
1. test-projects/project-a/.env (5 variables)
2. test-projects/project-c/subdir/.env (3 variables)
3. test-projects/project-b/.env (5 variables)
```

**Validation**: Changes verified by directly inspecting `.env` files after save operations.

## Bugs Encountered and Fixed

### Bug #1: DataTable Not Editable (45 minutes)
**Symptom**: Couldn't enable inline editing in DataTable
**Root Cause**: DataTable widget doesn't support inline editing (design limitation)
**Fix**: Switched to individual Input widgets
**Lesson**: Read widget docs before building

### Bug #2: Quote Handling Broken (30 minutes)
**Symptom**: `SECRET_KEY="my key"` saved as `SECRET_KEY=my key` (invalid format)
**Root Cause**: Parser stripped all quotes, save didn't re-add them
**Fix**: Only strip matched quotes, re-add quotes for values with spaces
**Lesson**: Edge cases in file formats require careful round-trip testing

### Bug #3: CSS Not Applying (20 minutes)
**Symptom**: Styles not rendering
**Root Cause**: Class name mismatch (`var-row` vs `variable-row`)
**Fix**: Corrected class name to match
**Lesson**: Copy-paste class names to avoid typos

### Bug #4: Read-Only Name Property (5 minutes)
**Symptom**: `AttributeError: property 'name' of 'Input' object has no setter`
**Root Cause**: Tried to set `name` after widget creation
**Fix**: Pass `name` to Input constructor
**Lesson**: Check constructor parameters for read-only properties

### Bug #5: Over 300 Lines (3 lines over limit)
**Symptom**: Initial version was 303 lines
**Root Cause**: Verbose docstrings and whitespace
**Fix**: Removed non-essential docstrings, condensed whitespace
**Lesson**: 300 line limits require tight coding from the start

## Performance Characteristics

### Scan Performance

**Test**: Scanned directory with 50+ subdirectories and 12 projects
**Time**: < 1 second
**Bottleneck**: Disk I/O for file reading, not directory traversal

**Depth Limiting Impact**: Max depth of 3 prevents exponential growth in deeply nested monorepos.

### UI Responsiveness

**Input Lag**: None - immediate response to keypresses
**Screen Transitions**: Instant (no loading delays)
**Save Operations**: < 100ms for typical `.env` files (< 50 variables)

## Limitations and Scope

### Known Limitations

**1. Maximum 9 Files**: Only 1-9 keys for selection (10th+ files inaccessible)
**Counter**: Who has 9+ `.env` files open simultaneously?

**2. No Encryption**: Plain text storage
**Counter**: `.env` files are already plain text on disk. Use Vault/AWS Secrets Manager for encrypted secrets.

**3. No Git Integration**: No commit automation or secret detection
**Counter**: Scope creep. This is an editor, not a git client.

**4. No Multi-Line Values**: Doesn't handle backslash-continued values
**Counter**: Multi-line values in `.env` rare. Use JSON/YAML/TOML for complex configs.

**5. No `.env.*` Variants**: Only finds files named exactly `.env`
**Counter**: Easy to add (change file filter), but "ship it" decision.

### Design Philosophy

**"Solve One Problem Well"**: Edit `.env` files quickly without context switching. Everything else is feature creep.

**Not a Secrets Manager**: No encryption, no access control, no audit logs. Use appropriate tools (Vault, AWS Secrets Manager, Doppler) for production secrets management.

## Production Usage Insights

### Real-World Time Savings

**Per Use**: 30-60 seconds saved (vs opening editor, navigating directories)
**Weekly Uses**: 10+ times
**ROI Timeline**: 2 weeks to break even on 6-hour development time

**Actual Usage Pattern**: Primarily for local development environment switching. Updating database URLs, toggling debug flags, rotating API keys during testing.

## Key Takeaways for TUI Development

### 1. Terminal UIs Are Underrated

**Benefits for Developer Tools**:
- No context switching (developers already in terminal)
- Keyboard-first navigation (faster than mouse)
- Runs anywhere (no browser, no ports, no servers)
- Fast to build (Textual provides high-level widgets)

**When to Use Terminal UIs**:
- Internal developer tools
- CLI enhancements
- System administration interfaces
- Quick productivity utilities

### 2. Widget Documentation First

**Anti-Pattern**: Assume widget does what you need → build → hit limitations → rebuild

**Best Practice**: Read widget docs → verify capabilities → choose correct widget → build

**Time Saved**: 45 minutes per architectural mismatch avoided.

### 3. File Format Edge Cases Will Bite You

**Lesson**: Test with messy real-world files, not clean examples.

**Edge Cases for `.env` Format**:
- Quoted vs unquoted values
- Spaces in values
- Special characters in values
- Comments (inline and full-line)
- Empty values
- Equals signs in values

### 4. Small Tools Compound Value

**Pattern**: 300-line scripts that save 30 seconds but get used daily have ROI measured in weeks, not months.

**Philosophy**: Best tools aren't feature-rich, they're actually used.

## Code Quality Metrics

**Lines of Code**: 297 (under 300-line constraint)
**Time to Build**: 6 hours
**Dependencies**: 1 (`textual >= 0.40.0`)
**Files Created**: 7 (app, requirements, README, tests, examples)
**Bugs Fixed**: 5 (documented above)
**Test Coverage**: Core scanning, parser edge cases, save/load, recursive discovery

## Integration Patterns

### Usage with Other Tools

**Docker Integration**: Can scan Docker project directories for `.env` files
**CI/CD Integration**: Could be extended to validate `.env` format in pipelines
**Git Hooks**: Could check for uncommitted secret changes

**Example - Pre-commit Hook**:
```bash
#!/bin/bash
# .git/hooks/pre-commit
python envman.py --validate-only
```

## Repository and Installation

**GitHub**: [https://github.com/FyefoxxM/environment-variable-manager](https://github.com/FyefoxxM/environment-variable-manager)

**Installation**:
```bash
pip install textual
git clone https://github.com/FyefoxxM/environment-variable-manager.git
cd environment-variable-manager
```

**Usage**:
```bash
# Scan current directory
python envman.py

# Scan specific directory
python envman.py ~/projects
```

**Controls**:
- **1-9**: Edit corresponding file
- **ESC**: Go back
- **Q**: Quit
- **Tab**: Navigate inputs
- **Save Button**: Persist changes

## Sources

**Primary Source**:
- [Environment Variable Manager: Stop Opening 12 Editors to Change One API Key](https://jdookeran.medium.com/environment-variable-manager-stop-opening-12-editors-to-change-one-api-key-e6bfbac951db) - Medium article by Jason Dookeran (accessed 2025-11-02)

**Code Repository**:
- [GitHub - environment-variable-manager](https://github.com/FyefoxxM/environment-variable-manager) - Full source code and examples

**Related Documentation**:
- See [architecture/00-core-concepts.md](../architecture/00-core-concepts.md) - Textual widget fundamentals
- See [widgets/01-input-basics.md](../widgets/01-input-basics.md) - Input widget patterns
- See [layout/00-grid-system.md](../layout/00-grid-system.md) - Container layout strategies

---

**Case Study Summary**: Production-ready TUI demonstrating practical patterns for file I/O, dynamic widget composition, state management, and real-world edge case handling. Built in 6 hours, 297 lines, solves genuine developer workflow problem with measurable time savings.
