# Textual TUI Tips - Japanese Community Insights (Zenn.dev)

**Source**: [textualで作るTUIのTIPS](https://zenn.dev/secondselection/articles/textual_tips) by o.m @ Second Selection (accessed 2025-11-02)

**Language**: Japanese (translated to English)

**Author**: o.m (前野) at Second Selection株式会社

**Published**: 2025-06-10

---

## Overview

Practical tips and lessons learned from building TUI applications with Textual. This article shares trial-and-error experiences and important considerations when creating screens with Textual, from a Japanese developer's perspective.

---

## Table of Contents

1. [compose() Method](#1-compose-method)
2. [Widget Addition Methods](#2-widget-addition-methods)
3. [on_mount() Method](#3-on_mount-method)
4. [CSS-like Stylesheets](#4-css-like-stylesheets)
5. [Program Termination (VSCode Terminal)](#5-program-termination-vscode-terminal)
6. [Debugging](#6-debugging)
7. [Events](#7-events)
8. [Clipboard Operations](#8-clipboard-operations)
9. [Widgets](#9-widgets)

---

## 1. compose() Method

**Purpose**: Defines widget layout using yield to add widgets sequentially.

**Key Insight**: Especially useful for component reuse and dynamic screen construction.

### Example: Building a Screen with Multiple Widgets

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

**Benefit**: Simple and intuitive way to describe widget structure.

---

## 2. Widget Addition Methods

Two approaches to adding widgets in Textual:

### Method 1: Using Instance Variables

Store widget instance in `self`, then yield for layout:

```python
def compose(self) -> ComposeResult:
    self.note_table = DataTable(id="note_table")
    yield self.note_table
```

**Advantages**:
- Easy access from other methods
- Direct widget manipulation: `self.note_table.add_row(...)`

**Disadvantages**:
- More variables reduce code readability

### Method 2: Direct yield

Yield widget directly without storing:

```python
def compose(self) -> ComposeResult:
    yield DataTable(id="note_table")
```

**Advantages**:
- Simpler, cleaner code
- Good for temporary widgets

**Disadvantages**:
- No direct access from other methods
- Must use `query_one()` to access:

```python
table = self.query_one("#note_table", DataTable)
table.add_row("1", "2025-03-24", "Sample note")
```

### Which to Use?

**Use instance variables when**:
- Updating widget state (button disable/enable, data updates)
- Frequent widget manipulation needed

**Use direct yield when**:
- Simple, static layouts
- One-time widget setup (fixed buttons, labels)

---

## 3. on_mount() Method

**Purpose**: Called immediately after screen construction. Ideal for initialization, data loading, and event listener registration.

**Key Timing**: Executes after widgets are added to layout.

### Example: DataTable Initialization

```python
def on_mount(self) -> None:
    # Add columns
    for key in ["id", "date", "contents"]:
        self.note_table.add_column(key, key=key)

    # Load data
    self._load_data()
```

**Use Cases**:
- Adding DataTable columns (must happen after compose)
- Initial data loading
- Setting up event listeners
- Widget configuration that requires layout to be ready

---

## 4. CSS-like Stylesheets

**TCSS (Textual CSS)**: Use IDs, classes, and widget names to style in `.tcss` files.

### Important Constraints

**Button Height**:
- Must be at least 3 rows tall
- Otherwise, borders and text won't display correctly
- This is a TUI layout limitation

**Input Width**:
- Must specify width in characters
- Without width specification, Input stretches to full screen width

### Styling Example

```tcss
/* Target by ID */
#note_table {
    width: 100%;
    height: 20;
}

/* Target by widget type */
Button {
    min-height: 3;
    width: 20;
}

/* Target by class */
.action-button {
    background: blue;
}
```

---

## 5. Program Termination (VSCode Terminal)

**Common Problem**: How to quit Textual apps running in VSCode terminal?

### The VSCode Terminal Challenge

**Standard Exit**: `Ctrl+Q` (doesn't work in VSCode)
- VSCode intercepts `Ctrl+Q` for its own features
- `Ctrl+C` shows "Press Ctrl+Q to quit" but doesn't exit

**Author's Initial Solution**: Kill the terminal (frustrating!)

### Four Solutions

#### 1. Add Quit Button

```python
def compose(self) -> ComposeResult:
    yield Button("Quit", id="quit_button")

@on(Button.Pressed, "#quit_button")
def quit_app(self) -> None:
    self.exit()
```

#### 2. Use Footer Palette

Add Footer widget to show palette in corner:

```python
def compose(self) -> ComposeResult:
    yield Footer()
    # ... other widgets
```

- Shows `^p palette` in corner
- Click → Select "Quit the application"

#### 3. ESC + Q

Press `ESC`, then `Q` to quit.

#### 4. Key Bindings (Author's Preferred Method)

```python
class DailyNote(App):
    BINDINGS = [
        Binding("q", "quit_process", "QUIT", show=True, priority=True),
    ]
```

**Result**: Footer shows "q QUIT" - click to quit.

**Why This Works Best**:
- Mouse-clickable
- Visual indicator in UI
- Simple one-key binding

---

## 6. Debugging

**Challenge**: `print()` statements don't show anywhere in TUI apps.

**Solution**: Use `RichLog` widget as debug output:

```python
def compose(self) -> ComposeResult:
    yield RichLog(id="logger")
    # ... other widgets

def debug_info(self, message: str) -> None:
    logger = self.query_one("#logger", RichLog)
    logger.write(message)
```

**Benefits**:
- Real-time visibility during execution
- Better than external logging for interactive debugging
- Can display variable contents, state changes, etc.

**Alternative**: Textual's built-in logger, but RichLog provides visual feedback in the UI itself.

---

## 7. Events

### Two Event Handling Approaches

#### Method 1: Override Predefined Event Methods

```python
def on_key(self, event: Key) -> None:
    """Key input event"""
    pass

def on_list_view_selected(self, event: ListView.Selected) -> None:
    """Listbox selection event"""
    pass

def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
    """DataTable row selection event"""
    pass
```

#### Method 2: Use @on Decorator (Preferred for Specificity)

**Decorator Syntax**: `@on(EventClass, "#widget_id")`

**Button Example**:

```python
@on(Button.Pressed, "#load_note_button")
def load_note_button(self) -> None:
    print("Load Note button pressed")
```

**DataTable Example**:

```python
@on(DataTable.RowSelected, "#note_table")
def data_table_row_selected(self, event: DataTable.RowSelected) -> None:
    print(f"Selected row: {event.row_key}")
```

**Advantages of @on Decorator**:
- Clear association between event and handler
- Can handle multiple widgets of same type differently
- More explicit than overriding methods

---

## 8. Clipboard Operations

### Linux (WSL2) Issues

**Problem**: Clipboard copy doesn't work reliably.

**Attempted Solutions**:
- `pyperclip` library → Errors
- WSL2 `Clip.exe` → Character encoding issues (mojibake)

**Current Workaround**: Use TextArea widget
1. Display text in TextArea
2. User manually selects all (Ctrl+A)
3. User copies (Ctrl+C)

**Author's Request**: "If anyone has solved this, please share!"

### Windows

**Status**: Clipboard copy works properly on native Windows.

**Copy Button Implementation**:

```python
@on(Button.Pressed, "#copy_button")
def copy_to_clipboard(self) -> None:
    import pyperclip
    content = self.get_content_to_copy()
    pyperclip.copy(content)
```

---

## 9. Widgets

### TabbedContent & TabPane

**Use Case**: Maximize limited TUI screen space using tabs.

**Current Limitation**: Cannot dynamically add tabs.

**Author's Challenge**:
- Tried various approaches to add tabs dynamically
- No success so far
- **Request**: "If anyone knows how, please share!"

**Static Tab Example**:

```python
def compose(self) -> ComposeResult:
    with TabbedContent():
        with TabPane("Overview"):
            yield Label("Overview content")
        with TabPane("Details"):
            yield Label("Details content")
```

---

### MarkdownViewer

**Purpose**: Display formatted markdown text in TUI.

**Example Use Case**: Daily report viewer

```python
def compose(self) -> ComposeResult:
    yield MarkdownViewer(id="markdown_viewer")

def show_markdown(self, markdown_text: str) -> None:
    viewer = self.query_one("#markdown_viewer", MarkdownViewer)
    # How to update content dynamically?
```

**Documentation Gap**: Most examples show mounting MarkdownViewer with initial markdown:

```python
await self.mount(MarkdownViewer(markdown_text))
```

**Author's Question**: "How to update MarkdownViewer content when selecting items from a list? Documentation doesn't cover this."

**Community Need**: More examples of dynamic content updates in MarkdownViewer.

---

## Key Takeaways (Japanese Developer Perspective)

### Common Pain Points

1. **VSCode Terminal Integration**: Standard quit shortcuts conflict with VSCode
2. **Clipboard Access**: Linux/WSL2 clipboard operations problematic
3. **Dynamic Content**: Limited documentation on updating widgets after initial mount
4. **Tab Management**: Cannot add tabs dynamically (frequent request)

### Best Practices Discovered

1. **Use @on Decorator**: More explicit event handling than method overrides
2. **RichLog for Debugging**: Essential for real-time debug output in TUI
3. **Key Bindings with Footer**: Most user-friendly quit method for VSCode
4. **TCSS Constraints**: Remember minimum heights and explicit widths

### Documentation Gaps Identified

- Dynamic MarkdownViewer updates
- Dynamic tab addition in TabbedContent
- WSL2 clipboard integration
- Advanced widget state management

---

## Cross-Cultural Development Insights

**Japanese Developer Challenges**:
- WSL2 is popular in Japanese dev environments → clipboard issues more prominent
- VSCode dominance → terminal integration critical
- Emphasis on documentation completeness → gaps feel more frustrating

**Community Collaboration Request**: Author actively seeking solutions, demonstrating open approach to problem-solving.

---

## Related Oracle Content

**Cross-References**:
- [architecture/02-basic-app-structure.md](../architecture/02-basic-app-structure.md) - compose() and on_mount() lifecycle
- [widgets/09-datatable.md](../widgets/09-datatable.md) - DataTable usage patterns
- [widgets/20-markdownviewer.md](../widgets/20-markdownviewer.md) - MarkdownViewer documentation
- [widgets/24-tabbedcontent.md](../widgets/24-tabbedcontent.md) - TabbedContent and TabPane
- [architecture/08-event-handling.md](../architecture/08-event-handling.md) - Event handling patterns
- [patterns/00-async-chat-ui.md](../patterns/00-async-chat-ui.md) - Advanced async patterns

---

## Sources

**Primary Source**:
- [textualで作るTUIのTIPS](https://zenn.dev/secondselection/articles/textual_tips) - Zenn.dev article by o.m (accessed 2025-11-02)

**Author Information**:
- o.m (前野)
- Second Selection株式会社 (Second Selection Corporation)
- Published: 2025-06-10

**Topics Covered**: Python, TUI, Textual

---

## Translation Notes

- Original article in Japanese
- Translated for English-speaking developers
- Preserved author's questions and community requests
- Cultural context added for cross-cultural understanding
- Technical terms kept consistent with Textual documentation

**Original Title**: textualで作るTUIのTIPS
**Translation**: "Tips for Creating TUI with Textual"
