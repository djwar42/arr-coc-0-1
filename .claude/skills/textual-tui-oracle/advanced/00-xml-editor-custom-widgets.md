# Custom Widget Patterns from BoomslangXML Editor

## Overview

This document examines advanced custom widget development patterns demonstrated in the BoomslangXML TUI application - a real-world XML editor built with Textual. The application showcases practical techniques for creating specialized widgets, managing complex data structures, and implementing sophisticated event handling in terminal user interfaces.

**Key Topics Covered:**
- Custom widget subclassing with data storage
- Multi-screen application architecture
- Event-driven XML tree navigation
- Dynamic UI updates based on data selection
- File browser and modal dialog patterns

From [Creating a Simple XML Editor in Your Terminal with Python and Textual](https://www.blog.pythonlibrary.org/2025/07/30/tui-xml-editor/) (accessed 2025-11-02)

## Custom Widget Pattern: DataInput

### The Problem
Standard Textual `Input` widgets don't inherently store references to application data. When editing structured data (like XML), you need widgets that maintain bidirectional links between UI elements and underlying data objects.

### The Solution: Subclass with Data Storage

```python
from textual.widgets import Input
import lxml.etree as ET

class DataInput(Input):
    """
    Create a variant of the Input widget that stores data
    """
    def __init__(self, xml_obj: ET.Element, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xml_obj = xml_obj
```

**Key Architectural Decision:**
- Store the XML element reference directly on the widget instance
- Pass the XML object during widget construction
- Enables direct data updates when input changes

### Usage Pattern

```python
# Create input widget bound to XML element
text = child.text if child.text else ""
data_input = DataInput(child, text)
data_input.border_title = child.tag
container = Horizontal(data_input)
right_pane.mount(container)

# Later, when input changes:
@on(Input.Changed)
def on_input_changed(self, event: Input.Changed) -> None:
    xml_obj = event.input.xml_obj
    xml_obj.text = event.input.value  # Direct data update
```

**Benefits:**
- Eliminates need for external mapping dictionaries
- Self-contained widget-to-data relationship
- Type-safe with proper type hints
- Reusable pattern for any data-bound input

## Multi-Screen Application Architecture

### Screen Organization Pattern

BoomslangXML uses a modal screen architecture with specialized screens for different tasks:

```python
class BoomslangXML(App):
    # Main application screen

class EditXMLScreen(ModalScreen):
    # Primary editing interface

class AddNodeScreen(ModalScreen):
    # Node creation dialog

class PreviewXMLScreen(ModalScreen):
    # XML preview viewer

class FileBrowser(Screen):
    # File selection interface

class SaveFileDialog(Screen):
    # Save file interface

class WarningScreen(Screen):
    # Error/warning messages
```

### Screen Navigation Patterns

**Push Screen with Callback:**
```python
def action_add_node(self) -> None:
    def add_node(result: tuple[str, str] | None) -> None:
        if result is not None:
            node_name, node_value = result
            self.update_xml_tree(node_name, node_value)

    self.app.push_screen(AddNodeScreen(), add_node)
```

**Push Screen with Message:**
```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    self.post_message(self.Selected(self.selected_file))
    self.dismiss()
```

**Key Pattern**: Screens communicate via callbacks or custom messages, maintaining loose coupling between components.

## Tree Widget with Lazy Loading

### The Challenge
Loading entire XML trees at once can be slow and memory-intensive for large documents. The tree control needs to populate nodes only when expanded.

### Lazy Loading Implementation

```python
class EditXMLScreen(ModalScreen):
    def __init__(self, xml_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xml_tree = ET.parse(xml_path)
        self.expanded = {}  # Track which nodes are expanded
        self.selected_tree_node: None | TreeNode = None

    @on(Tree.NodeExpanded)
    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """
        When a tree node is expanded, parse the newly shown leaves
        and make them expandable, if necessary.
        """
        xml_obj = event.node.data
        if id(xml_obj) not in self.expanded and xml_obj is not None:
            for top_level_item in xml_obj.getchildren():
                child = event.node.add_leaf(
                    top_level_item.tag,
                    data=top_level_item
                )
                if top_level_item.getchildren():
                    child.allow_expand = True
                else:
                    child.allow_expand = False
            self.expanded[id(xml_obj)] = ""
```

**Key Techniques:**
- Track expanded nodes using dictionary with `id(xml_obj)` as key
- Only populate children when parent is expanded
- Set `allow_expand` flag based on whether node has children
- Store XML element reference in `data` attribute

### Initial Tree Population

```python
def load_tree(self) -> None:
    tree = self.query_one("#xml_tree", Tree)
    xml_root = self.xml_tree.getroot()
    self.expanded[id(xml_root)] = ""

    tree.reset(xml_root.tag)
    tree.root.expand()

    # If the root has children, add them
    if xml_root.getchildren():
        for top_level_item in xml_root.getchildren():
            child = tree.root.add(top_level_item.tag, data=top_level_item)
            if top_level_item.getchildren():
                child.allow_expand = True
            else:
                child.allow_expand = False
```

**Pattern**: Initialize with root + first level, then lazy-load subsequent levels on demand.

## Dynamic UI Updates Based on Selection

### Master-Detail Pattern

The XML editor implements a classic master-detail interface:
- **Left pane**: Tree control (master)
- **Right pane**: Detail view with editable inputs

### Selection-Driven UI Updates

```python
@on(Tree.NodeSelected)
def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
    """
    When a node in the tree control is selected, update the right pane
    to show the data in the XML, if any
    """
    xml_obj = event.node.data
    right_pane = self.query_one("#right_pane", VerticalScroll)
    right_pane.remove_children()  # Clear existing content
    self.selected_tree_node = event.node

    if xml_obj is not None:
        for child in xml_obj.getchildren():
            if child.getchildren():
                continue  # Skip nodes with children
            text = child.text if child.text else ""
            data_input = DataInput(child, text)
            data_input.border_title = child.tag
            container = Horizontal(data_input)
            right_pane.mount(container)
        else:
            # XML object has no children, show tag and text
            if getattr(xml_obj, "tag") and getattr(xml_obj, "text"):
                if xml_obj.getchildren() == []:
                    data_input = DataInput(xml_obj, xml_obj.text)
                    data_input.border_title = xml_obj.tag
                    container = Horizontal(data_input)
                    right_pane.mount(container)
```

**Pattern Breakdown:**
1. **Clear previous content**: `right_pane.remove_children()`
2. **Store selection**: `self.selected_tree_node = event.node`
3. **Iterate children**: Only show leaf nodes (no nested children)
4. **Dynamic mounting**: Create and mount widgets on-the-fly
5. **Fallback handling**: Special case for leaf nodes with text

**Key Insight**: Use `mount()` for dynamic widget addition after initial composition.

## Event Handling Patterns

### The @on Decorator Pattern

Textual's `@on` decorator enables precise event targeting:

```python
@on(Button.Pressed, "#open_xml_file")
def on_open_xml_file(self) -> None:
    self.push_screen(FileBrowser())

@on(Button.Pressed, "#open_recent_file")
def on_open_recent_file(self) -> None:
    if self.current_recent_file is not None:
        self.push_screen(EditXMLScreen(self.current_recent_file))

@on(OptionList.OptionSelected, "#recent_files")
def on_recent_files_selected(self, event: OptionList.OptionSelected) -> None:
    self.current_recent_file = Path(event.option.prompt)
```

**Benefits over generic handlers:**
- No need to check event source in handler
- More readable and maintainable
- Type-safe event parameter
- Can target specific widget IDs or classes

### Input Change Tracking

```python
@on(Input.Changed)
def on_input_changed(self, event: Input.Changed) -> None:
    """
    When an XML element changes, update the XML object
    """
    xml_obj = event.input.xml_obj  # Access stored data
    xml_obj.text = event.input.value
```

**Pattern**: Combine custom widget properties with event handlers for seamless data binding.

### Keyboard Bindings with Actions

```python
class EditXMLScreen(ModalScreen):
    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("ctrl+a", "add_node", "Add Node"),
        ("p", "preview", "Preview"),
        ("escape", "esc", "Exit dialog"),
    ]

    def action_save(self) -> None:
        self.xml_tree.write(r"C:\Temp\books.xml")
        self.notify("Saved!")

    def action_add_node(self) -> None:
        # Show dialog with callback
        def add_node(result: tuple[str, str] | None) -> None:
            if result is not None:
                node_name, node_value = result
                self.update_xml_tree(node_name, node_value)
        self.app.push_screen(AddNodeScreen(), add_node)

    def action_preview(self) -> None:
        temp_directory = Path(tempfile.gettempdir())
        xml_path = temp_directory / "temp.xml"
        self.xml_tree.write(xml_path)
        self.app.push_screen(PreviewXMLScreen(xml_path))

    def action_esc(self) -> None:
        self.dismiss()
```

**Pattern**: Define bindings in `BINDINGS` list, implement as `action_*` methods.

## File Management Patterns

### Recent Files List

```python
def __init__(self) -> None:
    super().__init__()
    self.recent_files_path = Path(__file__).absolute().parent / "recent_files.txt"
    self.current_recent_file: Path | None = None

def update_recent_files_ui(self) -> None:
    if self.recent_files_path.exists():
        self.recent_files.clear_options()
        files = self.recent_files_path.read_text()
        for file in files.split("\n"):
            self.recent_files.add_option(file.strip())

def update_recent_files_on_disk(self, path: Path) -> None:
    if path.exists() and self.recent_files_path.exists():
        recent_files = self.recent_files_path.read_text()
        if str(path) in recent_files:
            return  # Don't add duplicates

        with open(self.recent_files_path, mode="a") as f:
            f.write(str(path) + "\n")

        self.update_recent_files_ui()
    elif not self.recent_files_path.exists():
        with open(self.recent_files_path, mode="a") as f:
            f.write(str(path) + "\n")
```

**Pattern**: Persistent state via simple text file + UI sync methods.

### Custom File Browser Message

```python
class FileBrowser(Screen):
    class Selected(Message):
        """File selected message"""
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.selected_file.suffix.lower() != ".xml":
            self.app.push_screen(WarningScreen("ERROR: Must choose XML!"))
            return

        self.post_message(self.Selected(self.selected_file))
        self.dismiss()

# Main app handler:
def on_file_browser_selected(self, message: FileBrowser.Selected) -> None:
    path = message.path
    if path.suffix.lower() == ".xml":
        self.update_recent_files_on_disk(path)
        self.push_screen(EditXMLScreen(path))
```

**Pattern**: Define custom message as nested class, post and handle at app level.

## Widget Composition Techniques

### Container-Based Layouts

```python
def compose(self) -> ComposeResult:
    yield Header()
    yield Horizontal(
        Vertical(Tree("No Data Loaded", id="xml_tree"), id="left_pane"),
        VerticalScroll(id="right_pane"),
        id="main_ui_container",
    )
    yield Footer()
```

**Layout Strategy:**
- `Horizontal`: Split screen left/right
- `Vertical`: Stack elements top/bottom
- `VerticalScroll`: Scrollable container for dynamic content
- Use IDs for querying and CSS targeting

### Modal Dialog Pattern

```python
class AddNodeScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        self.node_name = Input(id="node_name")
        self.node_name.border_title = "Node Name"
        self.node_value = Input(id="node_value")
        self.node_value.border_title = "Node Value"

        yield Vertical(
            Header(),
            self.node_name,
            self.node_value,
            Horizontal(
                Button("Save Node", variant="primary", id="save_node"),
                Button("Cancel", variant="warning", id="cancel_node"),
            ),
            Footer(),
            id="add_node_screen_ui",
        )

    @on(Button.Pressed, "#save_node")
    def on_save(self) -> None:
        self.dismiss((self.node_name.value, self.node_value.value))

    @on(Button.Pressed, "#cancel_node")
    def on_cancel(self) -> None:
        self.dismiss()
```

**Pattern Elements:**
- Store widget references as instance variables for later access
- Return data via `dismiss(result)`
- Handle both save and cancel actions
- Use `ModalScreen` for overlay behavior

## CSS Organization Strategy

### Per-Screen CSS Files

BoomslangXML uses one CSS file per screen:

```
main.tcss               # Main application
edit_xml_screens.tcss   # XML editor screen
add_node_screen.tcss    # Add node dialog
preview_xml_screen.tcss # Preview screen
file_browser_screen.tcss # File browser
save_file_dialog.tcss   # Save dialog
```

### Example CSS Patterns

**Dialog sizing and centering:**
```css
AddNodeScreen {
    align: center middle;
    background: $primary 30%;

    #add_node_screen_ui {
        width: 80%;
        height: 40%;
        border: thick $background 70%;
        content-align: center middle;
        margin: 2;
    }
}
```

**Auto-sizing for dynamic content:**
```css
EditXMLScreen {
    Input {
        border: solid gold;
        margin: 1;
        height: auto;  /* Fit content */
    }
    Horizontal {
        margin: 1;
        height: auto;  /* Fit content */
    }
}
```

**Widget-specific styling:**
```css
BoomslangXML {
    OptionList {
        border: solid green;
    }
    Button {
        margin: 1;
    }
}
```

## Data Synchronization Pattern

### Keeping UI and Data in Sync

**Three-way synchronization:**
1. **XML object** (source of truth)
2. **Tree widget** (structural view)
3. **Input widgets** (editable values)

```python
def update_xml_tree(self, node_name: str, node_value: str) -> None:
    """Update XML object and UI together"""
    # 1. Update data structure
    element = ET.SubElement(self.selected_tree_node.data, node_name)
    element.text = node_value

    # 2. Update UI
    self.update_tree_nodes(node_name, element)

def update_tree_nodes(self, node_name: str, node: ET.SubElement) -> None:
    """Sync tree widget with data"""
    child = self.selected_tree_node.add(node_name, data=node)
    child.allow_expand = False

@on(Input.Changed)
def on_input_changed(self, event: Input.Changed) -> None:
    """Sync data with UI changes"""
    xml_obj = event.input.xml_obj
    xml_obj.text = event.input.value
```

**Pattern**: Always update data structure and UI representation together.

## Best Practices Demonstrated

### 1. Widget Subclassing for Domain Models
**Do**: Create specialized widgets that carry domain data
```python
class DataInput(Input):
    def __init__(self, xml_obj: ET.Element, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xml_obj = xml_obj
```

**Don't**: Maintain separate mappings between widgets and data

### 2. Lazy Loading for Performance
**Do**: Load tree nodes on-demand when expanded
```python
@on(Tree.NodeExpanded)
def on_tree_node_expanded(self, event):
    if id(xml_obj) not in self.expanded:
        # Load children now
```

**Don't**: Load entire tree structure at initialization

### 3. Event-Driven Architecture
**Do**: Use specific event handlers with `@on` decorator
```python
@on(Button.Pressed, "#save_button")
def on_save_button(self):
    pass
```

**Don't**: Use generic handlers that check event sources

### 4. Screen Composition for Modularity
**Do**: Separate concerns into dedicated screens
```python
EditXMLScreen  # Main editing
AddNodeScreen  # Adding nodes
PreviewScreen  # Viewing
```

**Don't**: Cram all functionality into one monolithic screen

### 5. Dynamic UI Updates
**Do**: Use `mount()` and `remove_children()` for dynamic content
```python
right_pane.remove_children()
for child in data:
    right_pane.mount(create_widget(child))
```

**Don't**: Recreate entire UI for small changes

### 6. Callback-Based Screen Communication
**Do**: Use callbacks for parent-child communication
```python
def on_action(self):
    def callback(result):
        self.process(result)
    self.push_screen(ChildScreen(), callback)
```

**Don't**: Access parent screen directly from child

## Reusable Patterns Identified

### Pattern 1: Data-Bound Input Widget
```python
class DataBoundInput(Input):
    """Input widget that stores reference to data object"""
    def __init__(self, data_obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_obj = data_obj
```

**Use cases**: Forms, editors, any UI binding to domain models

### Pattern 2: Master-Detail with Tree
```python
# Master (left): Tree navigation
# Detail (right): Content editing

@on(Tree.NodeSelected)
def on_selection(self, event):
    self.update_detail_pane(event.node.data)

def update_detail_pane(self, data):
    detail_pane.remove_children()
    for item in data:
        detail_pane.mount(create_editor(item))
```

**Use cases**: File browsers, database explorers, document editors

### Pattern 3: Modal Dialog with Result
```python
def show_dialog(self):
    def on_result(result):
        if result:
            self.process(result)
    self.app.push_screen(DialogScreen(), on_result)

# In dialog:
def on_ok(self):
    self.dismiss(self.get_form_data())
```

**Use cases**: Input dialogs, confirmations, settings screens

### Pattern 4: Lazy-Loading Tree
```python
@on(Tree.NodeExpanded)
def on_expand(self, event):
    node_id = id(event.node.data)
    if node_id not in self.loaded:
        for child in get_children(event.node.data):
            event.node.add(child.name, data=child)
        self.loaded.add(node_id)
```

**Use cases**: File systems, large hierarchies, API-driven trees

### Pattern 5: Recent Files Management
```python
# Persist to simple text file
def save_recent(path):
    with open(recent_file, "a") as f:
        f.write(f"{path}\n")

# Load and populate UI
def load_recent():
    files = recent_file.read_text().split("\n")
    for file in files:
        option_list.add_option(file)
```

**Use cases**: MRU lists, user preferences, session state

## Advanced Techniques

### Technique 1: Directory Tree with Type Filtering

```python
def on_file_browser_selected(self, message: FileBrowser.Selected) -> None:
    path = message.path
    if path.suffix.lower() == ".xml":
        self.update_recent_files_on_disk(path)
        self.push_screen(EditXMLScreen(path))
    else:
        self.notify("Please choose an XML File!", severity="error")
```

**Pattern**: Validate file type before processing, notify user of errors.

### Technique 2: Temporary File for Preview

```python
def action_preview(self) -> None:
    temp_directory = Path(tempfile.gettempdir())
    xml_path = temp_directory / "temp.xml"
    self.xml_tree.write(xml_path)
    self.app.push_screen(PreviewXMLScreen(xml_path))
```

**Pattern**: Use temporary files for previews without altering source data.

### Technique 3: TextArea with Language Syntax

```python
def compose(self) -> ComposeResult:
    with open(self.xml_file_path) as xml_file:
        xml = xml_file.read()
    text_area = TextArea(xml)
    text_area.language = "xml"  # Enable syntax highlighting
    yield text_area
```

**Pattern**: Set `language` property for automatic syntax highlighting.

### Technique 4: Focus Management

```python
def on_mount(self) -> None:
    """Focus the input widget so user can name the file"""
    self.query_one("#filename").focus()
```

**Pattern**: Set initial focus in `on_mount()` for better UX.

## Sources

**Primary Source:**
- [Creating a Simple XML Editor in Your Terminal with Python and Textual](https://www.blog.pythonlibrary.org/2025/07/30/tui-xml-editor/) - Mouse Vs Python blog by Mike Driscoll (accessed 2025-11-02)

**GitHub Repository:**
- [BoomslangXML TUI v0.2.0](https://github.com/driscollis/BoomslangXML_TUI/releases/tag/v0.2.0) - Full source code implementation

**Related Documentation:**
- See [getting-started/00-official-homepage.md](../getting-started/00-official-homepage.md) for Textual basics
- See [getting-started/01-official-tutorial.md](../getting-started/01-official-tutorial.md) for fundamental concepts
