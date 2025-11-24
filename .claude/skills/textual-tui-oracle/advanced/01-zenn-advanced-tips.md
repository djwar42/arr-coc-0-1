# Textual TUI Advanced Development Tips

**Source**: Japanese Textual community insights from Zenn platform
**Author**: o.m (SecondSelection Corporation)
**Original Title**: textualで作るTUIのTIPS (Textual TUI Tips)

Advanced patterns, debugging techniques, and production tips from real-world Textual development experience. This guide covers common pitfalls, performance considerations, and expert-level widget usage patterns discovered through hands-on TUI development.

---

## Overview

This knowledge file compiles advanced development tips from Japanese Textual community expert o.m (SecondSelection). Focus areas:
- Widget composition patterns (`compose` vs `on_mount`)
- Instance variable vs direct yield strategies
- Event handling patterns (`@on` decorator usage)
- Debugging TUI applications
- Widget-specific tips (DataTable, MarkdownViewer, TabbedContent)
- Platform-specific issues (VSCode terminal, clipboard handling)

---

## 1. Widget Composition: The `compose` Method

### Purpose
The `compose` method defines widget layout using generators. It enables:
- Reusable component patterns
- Dynamic screen construction
- Clean hierarchical widget organization

### Pattern Example
```python
def compose(self) -> ComposeResult:
    yield Header()
    with Vertical():
        yield DataTable(id="note_table")
        yield RichLog(id="logger")
        with Horizontal(id="action"):
            yield Button("Save note", id="save_note_button")
            yield Button("Delete note", id="delete_note_button")
```

**Key Benefits**:
- Intuitive structure representation
- Container widgets (`Vertical`, `Horizontal`) provide natural grouping
- Generator pattern keeps memory efficient

---

## 2. Widget Addition Strategies: Instance Variables vs Direct Yield

### Strategy 1: Instance Variable Approach

Store widget reference in `self` before yielding:

```python
def compose(self) -> ComposeResult:
    self.note_table = DataTable(id="note_table")
    yield self.note_table
```

**Advantages**:
- Direct access from other methods (`self.note_table.add_row(...)`)
- No query lookups required
- Clear object lifecycle

**Disadvantages**:
- Increases instance variable count
- Potential namespace pollution
- Slightly reduced code readability

### Strategy 2: Direct Yield Approach

Yield widget directly without storing:

```python
def compose(self) -> ComposeResult:
    yield DataTable(id="note_table")
```

**Advantages**:
- Cleaner, more concise code
- Appropriate for static/temporary widgets
- Reduced instance variable clutter

**Disadvantages**:
- Requires query lookup for later access:
  ```python
  table = self.query_one("#note_table", DataTable)
  table.add_row("1", "2025-03-24", "Sample note")
  ```
- Slight performance overhead for queries

### Decision Matrix

| Use Case | Recommended Approach |
|----------|---------------------|
| Widget state changes (enable/disable, data updates) | **Instance variable** |
| Fixed layout elements (static buttons, headers) | **Direct yield** |
| Frequent widget access | **Instance variable** |
| Simple one-time rendering | **Direct yield** |

---

## 3. Initialization: The `on_mount` Method

### Purpose
`on_mount` executes immediately after widget layout construction. Critical for:
- Initial data loading
- Column/row configuration
- Event listener registration
- Post-layout setup

### Pattern Example
```python
def on_mount(self) -> None:
    # Configure DataTable columns
    for key in ["id", "date", "contents"]:
        self.note_table.add_column(key, key=key)

    # Load initial data
    self._load_data()
```

**Why Not `compose`?**
- `compose` defines structure (what widgets exist)
- `on_mount` configures behavior (widget initialization)
- Layout must be complete before configuration

**Common Use Cases**:
- Database queries for initial data
- API calls to populate widgets
- Setting default values
- Registering dynamic event handlers

---

## 4. Styling: CSS-Like Stylesheets (TCSS)

### Selector Types
Target widgets using:
- **ID selectors**: `#note_table { ... }`
- **Class selectors**: `.primary-button { ... }`
- **Widget type selectors**: `Button { ... }`, `RichLog { ... }`

### Critical Layout Rules

**Button Height Requirement**:
```tcss
Button {
    height: 3;  /* Minimum 3 lines for proper border/text rendering */
}
```
- Less than 3 lines: borders and text render incorrectly
- TUI layout constraint, not Textual bug

**Input Width Behavior**:
```tcss
Input {
    width: 40;  /* Specify width in characters */
}
```
- Without explicit width: expands to full screen width
- Always set `width` for predictable layout

---

## 5. Program Termination in VSCode Terminal

### The Problem
Standard `Ctrl+Q` doesn't work in VSCode terminal:
- VSCode intercepts `Ctrl+Q` for feature switching
- `Ctrl+C` shows "Press Ctrl+Q to quit" but doesn't exit
- Terminal must be killed manually (frustrating!)

### Solutions

**Option 1: Quit Button**
```python
@on(Button.Pressed, "#quit_button")
def quit_handler(self) -> None:
    self.exit()
```

**Option 2: Footer Command Palette**
- Add `Footer` widget to app
- Displays `^p palette` in terminal
- Click to open command palette
- Select "Quit the application"

**Option 3: ESC + Q Sequence**
- Press `ESC` then `Q` (two-key sequence)
- Works in VSCode terminal

**Option 4: Custom Key Binding (Recommended)**
```python
class DailyNote(App):
    BINDINGS = [
        Binding("q", "quit_process", "QUIT", show=True, priority=True),
    ]
```
- Footer displays "q QUIT"
- Click with mouse or press `q` to exit
- Most user-friendly approach

**Best Practice**: Option 4 provides visible, mouse-clickable exit option.

---

## 6. Debugging TUI Applications

### The Print Problem
Standard `print()` statements don't display in TUI applications:
- Terminal is captured by Textual's rendering loop
- Output goes nowhere visible

### Solution: RichLog Widget

Add `RichLog` to your layout for live debugging:

```python
def compose(self) -> ComposeResult:
    yield RichLog(id="logger")

def some_method(self):
    logger = self.query_one("#logger", RichLog)
    logger.write(f"Debug: variable value = {self.data}")
```

**Advantages**:
- Live output visible during execution
- Supports rich formatting (colors, tables, etc.)
- Can be toggled on/off with visibility controls
- Better than external log files for interactive debugging

**Alternative**: Textual DevTools
- `textual console` command for external debug output
- Requires separate terminal window
- See official docs for setup

---

## 7. Event Handling Patterns

### Two Approaches

**Approach 1: Override Built-in Event Methods**
```python
def on_key(self, event: Key) -> None:
    # Handle keyboard input
    pass

def on_list_view_selected(self, event: ListView.Selected) -> None:
    # Handle list selection
    pass

def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
    # Handle table row selection
    pass
```

**Approach 2: `@on` Decorator (Recommended)**
```python
@on(Button.Pressed, "#load_note_button")
def load_note_button(self) -> None:
    print("Load Note button pressed")

@on(DataTable.RowSelected, "#note_table")
def data_table_row_selected(self, event: DataTable.RowSelected) -> None:
    print(f"Selected row: {event.row_key}")
```

### Why `@on` Decorator is Superior

**Clarity**:
- Event source immediately visible (widget ID in decorator)
- Handler purpose clear from method name

**Specificity**:
- Multiple handlers for same event type on different widgets
- No conditional logic required

**Example Comparison**:
```python
# Without @on - needs conditionals
def on_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == "load_note_button":
        # Load note logic
    elif event.button.id == "save_note_button":
        # Save note logic

# With @on - clean separation
@on(Button.Pressed, "#load_note_button")
def load_note(self) -> None:
    # Load note logic

@on(Button.Pressed, "#save_note_button")
def save_note(self) -> None:
    # Save note logic
```

---

## 8. Clipboard Handling: Platform Issues

### Linux/WSL2 Problems

**Issue**: `pyperclip` library fails on WSL2:
- Errors during clipboard operations
- `Clip.exe` (Windows clipboard bridge) causes text encoding issues
- No reliable clipboard solution found

**Workaround**: TextArea Manual Copy
```python
# Display content in TextArea
yield TextArea(id="copyable_content")

# User manually selects all (Ctrl+A) then copies (Ctrl+C)
```
- Not ideal, but functional
- Awaiting better solutions from community

**Request**: If you've solved WSL2 clipboard issues, please contribute!

### Windows Native

**No Issues**: Clipboard operations work correctly on native Windows:
```python
import pyperclip

@on(Button.Pressed, "#copy_button")
def copy_content(self) -> None:
    content = self.query_one("#markdown_viewer", MarkdownViewer)
    pyperclip.copy(content.document.text)
```

---

## 9. Widget-Specific Tips

### TabbedContent & TabPane

**Use Case**: Limited screen space requires tab-based navigation:
```python
def compose(self) -> ComposeResult:
    with TabbedContent():
        with TabPane("Notes"):
            yield DataTable(id="notes")
        with TabPane("Archive"):
            yield DataTable(id="archive")
```

**Current Limitation**: Dynamic tab addition not working
- Cannot add tabs at runtime reliably
- Must define all tabs in `compose` method
- Workaround: Pre-create hidden tabs, toggle visibility

**Request**: If you've achieved dynamic tab addition, please share!

### MarkdownViewer

**Purpose**: Render formatted markdown in TUI:

![Daily Report Example](visual reference from original)

**Common Pattern**: Display selected data as markdown:
```python
@on(DataTable.RowSelected, "#note_list")
def show_markdown(self, event: DataTable.RowSelected) -> None:
    markdown_content = self.get_note_content(event.row_key)
    viewer = self.query_one("#markdown_viewer", MarkdownViewer)
    # Update content (see below for method)
```

**Documentation Gap**: Official examples only show:
```python
# Initial mount approach (from docs)
await self.mount(MarkdownViewer(markdown_text))
```

**Problem**: How to update MarkdownViewer content after initial mount?
- Not clearly documented
- Community seeks update pattern

**Possible Solutions** (to investigate):
```python
# Approach 1: Update document attribute?
viewer.document = markdown_content

# Approach 2: Remove and re-mount?
await viewer.remove()
await self.mount(MarkdownViewer(markdown_content))

# Approach 3: Use reactive variable?
# (Needs testing)
```

**Request**: Best practice for updating MarkdownViewer content needed!

### DataTable

**Data Management**: Adding and updating rows

**Adding Rows**:
```python
def on_mount(self) -> None:
    table = self.query_one("#note_table", DataTable)

    # Add columns first
    table.add_column("ID", key="id")
    table.add_column("Date", key="date")
    table.add_column("Contents", key="contents")

    # Add rows
    table.add_row("1", "2025-03-24", "Sample note")
```

**Updating Data**: Pattern varies by use case
- Full refresh: Clear and re-add rows
- Single row update: Use row key to target specific row
- Reactive updates: Connect to reactive variables

---

## Performance Considerations

### Widget Query Optimization

**Instance Variable Strategy** (faster for frequent access):
```python
def compose(self) -> ComposeResult:
    self.table = DataTable(id="table")
    yield self.table

def update_data(self):
    self.table.add_row(...)  # Direct access, no query
```

**Query Strategy** (cleaner for infrequent access):
```python
def compose(self) -> ComposeResult:
    yield DataTable(id="table")

def update_data(self):
    table = self.query_one("#table", DataTable)  # Query overhead
    table.add_row(...)
```

**Recommendation**: Use instance variables for widgets accessed > 5 times per user interaction.

---

## Production Deployment Tips

### Terminal Compatibility

**Test on Target Platforms**:
- VSCode terminal (limited key bindings)
- Native terminal emulators (full capabilities)
- SSH sessions (network latency considerations)

**Key Binding Conflicts**:
- Document alternative keys for VSCode users
- Provide mouse-clickable alternatives
- Test `Ctrl+`, `Alt+`, `Shift+` combinations

### User Experience

**Always Provide Exit Options**:
1. Footer with visible quit binding
2. Menu/command palette access
3. Mouse-clickable quit button

**Debug Information**:
- Include `RichLog` in dev mode
- Toggle visibility with key binding
- Log important state changes

---

## Common Pitfalls

### 1. Button Height < 3 Lines
**Symptom**: Malformed button appearance
**Solution**: Always set `height: 3` minimum in TCSS

### 2. Missing Widget Width
**Symptom**: Input widgets span full screen
**Solution**: Explicitly set `width` in TCSS for input elements

### 3. Print Statements Disappear
**Symptom**: Debug output not visible
**Solution**: Use `RichLog` widget instead of `print()`

### 4. VSCode Terminal Won't Quit
**Symptom**: `Ctrl+Q` doesn't exit
**Solution**: Add custom `BINDINGS` with visible quit key

### 5. Dynamic Tab Addition Fails
**Symptom**: Runtime tab creation doesn't work
**Solution**: Pre-define all tabs in `compose`, toggle visibility

---

## Knowledge Gaps & Community Requests

Areas where documentation is lacking or solutions needed:

1. **Dynamic Tab Addition**: Reliable pattern for runtime tab creation
2. **MarkdownViewer Updates**: Best practice for updating content after mount
3. **WSL2 Clipboard**: Working solution for clipboard operations in WSL2
4. **DataTable Updates**: Efficient patterns for large dataset updates

**Contributing**: If you have solutions, please share with the community!

---

## Sources

**Original Article**:
- [textualで作るTUIのTIPS](https://zenn.dev/secondselection/articles/textual_tips) - Zenn article by o.m (accessed 2025-11-02)
- Author: o.m (SecondSelection Corporation)
- Platform: Zenn (Japanese developer community)
- Published: 2025-06-10

**Related Resources**:
- Textual Official Documentation: https://textual.textualize.io/
- GitHub: https://github.com/Textualize/textual

---

## Summary

Key takeaways from Japanese Textual community:

**Architectural Patterns**:
- Use `compose` for structure, `on_mount` for initialization
- Choose instance variables for frequent access, direct yield for simplicity
- Prefer `@on` decorator for event handling clarity

**Debugging**:
- RichLog widget for live TUI debugging
- No `print()` statements (they disappear)
- Test in VSCode terminal for key binding conflicts

**Production Readiness**:
- Provide multiple exit options (keys + mouse)
- Set minimum button height (3 lines)
- Explicitly size input widgets
- Test clipboard operations per platform

**Community Gaps**:
- Dynamic tab addition pattern needed
- MarkdownViewer update pattern unclear
- WSL2 clipboard issues unresolved

This knowledge represents real-world production experience from Japanese Textual developers, filling gaps in official documentation with practical insights.
