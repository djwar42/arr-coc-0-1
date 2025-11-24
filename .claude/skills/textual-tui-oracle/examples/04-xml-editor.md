# XML Editor TUI - BoomslangXML

Complete implementation of a terminal-based XML editor using Textual and lxml. This example demonstrates tree widgets, file I/O, multiple screens, and XML manipulation patterns.

## Overview

BoomslangXML is a full-featured XML viewer and editor that runs in the terminal. Originally created with wxPython, this version was ported to Textual to provide a powerful TUI alternative.

**Key Features:**
- Tree-based XML DOM navigation
- Edit XML elements and text content
- Add new nodes dynamically
- Preview XML before saving
- Recent files tracking
- File browser integration
- Multiple modal screens

## Architecture

The application consists of multiple screens and components:

**Main Components:**
- `BoomslangXML` - Main application with recent files list
- `EditXMLScreen` - Primary editing interface with tree + inputs
- `AddNodeScreen` - Modal dialog for adding new XML nodes
- `PreviewXMLScreen` - XML preview with syntax highlighting
- `FileBrowser` - File selection dialog
- `SaveFileDialog` - File save dialog
- `DataInput` - Custom Input widget that stores XML objects

## Dependencies

```bash
python -m pip install textual lxml
```

**Required packages:**
- `textual` - TUI framework
- `lxml` - Fast XML parsing and manipulation

## Main Application Structure

### boomslang.py

```python
from pathlib import Path

from .edit_xml_screen import EditXMLScreen
from .file_browser_screen import FileBrowser

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Header, Footer, OptionList


class BoomslangXML(App):
    BINDINGS = [
        ("ctrl+o", "open", "Open XML File"),
    ]
    CSS_PATH = "main.tcss"

    def __init__(self) -> None:
        super().__init__()
        self.title = "Boomslang XML"
        self.recent_files_path = Path(__file__).absolute().parent / "recent_files.txt"
        self.app_selected_file: Path | None = None
        self.current_recent_file: Path | None = None

    def compose(self) -> ComposeResult:
        self.recent_files = OptionList("", id="recent_files")
        self.recent_files.border_title = "Recent Files"
        yield Header()
        yield self.recent_files
        yield Vertical(
            Horizontal(
                Button("Open XML File", id="open_xml_file", variant="primary"),
                Button("Open Recent", id="open_recent_file", variant="warning"),
                id="button_row",
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.update_recent_files_ui()

    def action_open(self) -> None:
        self.push_screen(FileBrowser())

    def on_file_browser_selected(self, message: FileBrowser.Selected) -> None:
        path = message.path
        if path.suffix.lower() == ".xml":
            self.update_recent_files_on_disk(path)
            self.push_screen(EditXMLScreen(path))
        else:
            self.notify("Please choose an XML File!", severity="error", title="Error")

    @on(Button.Pressed, "#open_xml_file")
    def on_open_xml_file(self) -> None:
        self.push_screen(FileBrowser())

    @on(Button.Pressed, "#open_recent_file")
    def on_open_recent_file(self) -> None:
        if self.current_recent_file is not None and self.current_recent_file.exists():
            self.push_screen(EditXMLScreen(self.current_recent_file))

    @on(OptionList.OptionSelected, "#recent_files")
    def on_recent_files_selected(self, event: OptionList.OptionSelected) -> None:
        self.current_recent_file = Path(event.option.prompt)

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
                return

            with open(self.recent_files_path, mode="a") as f:
                f.write(str(path) + "\n")

            self.update_recent_files_ui()
        elif not self.recent_files_path.exists():
            with open(self.recent_files_path, mode="a") as f:
                f.write(str(path) + "\n")


def main() -> None:
    app = BoomslangXML()
    app.run()


if __name__ == "__main__":
    main()
```

**Key Patterns:**

1. **Recent Files Tracking**: Persistent text file stores recently opened files
2. **Keyboard Shortcuts**: CTRL+O opens file browser
3. **File Validation**: Ensures only .xml files are processed
4. **Screen Management**: Uses `push_screen()` for modal dialogs

### main.tcss

```css
BoomslangXML {
    #button_row {
        align: center middle;
    }

    Horizontal{
        height: auto;
    }

    OptionList {
        border: solid green;
    }

    Button {
        margin: 1;
    }
}
```

## Edit XML Screen

The core editing interface with tree navigation and input fields.

### edit_xml_screen.py

```python
import lxml.etree as ET
import tempfile
from pathlib import Path

from .add_node_screen import AddNodeScreen
from .preview_xml_screen import PreviewXMLScreen

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Tree
from textual.widgets._tree import TreeNode


class DataInput(Input):
    """
    Create a variant of the Input widget that stores data
    """

    def __init__(self, xml_obj: ET.Element, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xml_obj = xml_obj


class EditXMLScreen(ModalScreen):
    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("ctrl+a", "add_node", "Add Node"),
        ("p", "preview", "Preview"),
        ("escape", "esc", "Exit dialog"),
    ]
    CSS_PATH = "edit_xml_screens.tcss"

    def __init__(self, xml_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xml_tree = ET.parse(xml_path)
        self.expanded = {}
        self.selected_tree_node: None | TreeNode = None

    def compose(self) -> ComposeResult:
        xml_root = self.xml_tree.getroot()
        self.expanded[id(xml_root)] = ""
        yield Header()
        yield Horizontal(
            Vertical(Tree("No Data Loaded", id="xml_tree"), id="left_pane"),
            VerticalScroll(id="right_pane"),
            id="main_ui_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_tree()

    @on(Tree.NodeExpanded)
    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """
        When a tree node is expanded, parse the newly shown leaves and make
        them expandable, if necessary.
        """
        xml_obj = event.node.data
        if id(xml_obj) not in self.expanded and xml_obj is not None:
            for top_level_item in xml_obj.getchildren():
                child = event.node.add_leaf(top_level_item.tag, data=top_level_item)
                if top_level_item.getchildren():
                    child.allow_expand = True
                else:
                    child.allow_expand = False
            self.expanded[id(xml_obj)] = ""

    @on(Tree.NodeSelected)
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """
        When a node in the tree control is selected, update the right pane to show
        the data in the XML, if any
        """
        xml_obj = event.node.data
        right_pane = self.query_one("#right_pane", VerticalScroll)
        right_pane.remove_children()
        self.selected_tree_node = event.node

        if xml_obj is not None:
            for child in xml_obj.getchildren():
                if child.getchildren():
                    continue
                text = child.text if child.text else ""
                data_input = DataInput(child, text)
                data_input.border_title = child.tag
                container = Horizontal(data_input)
                right_pane.mount(container)
            else:
                # XML object has no children, so just show the tag and text
                if getattr(xml_obj, "tag") and getattr(xml_obj, "text"):
                    if xml_obj.getchildren() == []:
                        data_input = DataInput(xml_obj, xml_obj.text)
                        data_input.border_title = xml_obj.tag
                        container = Horizontal(data_input)
                        right_pane.mount(container)

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """
        When an XML element changes, update the XML object
        """
        xml_obj = event.input.xml_obj
        xml_obj.text = event.input.value

    def action_esc(self) -> None:
        """
        Close the dialog when the user presses ESC
        """
        self.dismiss()

    def action_add_node(self) -> None:
        """
        Add another node to the XML tree and the UI
        """

        # Show dialog and use callback to update XML and UI
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

    def action_save(self) -> None:
        self.xml_tree.write(r"C:\Temp\books.xml")
        self.notify("Saved!")

    def load_tree(self) -> None:
        """
        Load the XML tree UI with data parsed from the XML file
        """
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

    def update_tree_nodes(self, node_name: str, node: ET.SubElement) -> None:
        """
        When adding a new node, update the UI Tree element to reflect the new element added
        """
        child = self.selected_tree_node.add(node_name, data=node)
        child.allow_expand = False

    def update_xml_tree(self, node_name: str, node_value: str) -> None:
        """
        When adding a new node, update the XML object with the new element
        """
        element = ET.SubElement(self.selected_tree_node.data, node_name)
        element.text = node_value
        self.update_tree_nodes(node_name, element)
```

**Key Patterns:**

1. **Custom Input Widget**: `DataInput` stores XML element references for live editing
2. **Tree Lazy Loading**: Only expand children when nodes are opened (performance)
3. **Split Pane UI**: Tree on left, editable inputs on right
4. **Live XML Updates**: Input changes immediately update the lxml tree
5. **Callback Pattern**: Screens return data via callbacks (e.g., `add_node()`)

### edit_xml_screens.tcss

```css
EditXMLScreen {
    Input {
        border: solid gold;
        margin: 1;
        height: auto;
    }
    Button {
        align: center middle;
    }
    Horizontal {
        margin: 1;
        height: auto;
    }
}
```

## Add Node Screen

Modal dialog for adding new XML elements.

### add_node_screen.py

```python
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Footer, Input


class AddNodeScreen(ModalScreen):
    BINDINGS = [
        ("escape", "esc", "Exit dialog"),
    ]
    CSS_PATH = "add_node_screen.tcss"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = "Add New Node"

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

    def action_esc(self) -> None:
        """
        Close the dialog when the user presses ESC
        """
        self.dismiss()
```

**Key Pattern**: `dismiss()` returns tuple of values to parent screen callback

### add_node_screen.tcss

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

    Input {
        border: solid gold;
        margin: 1;
        height: auto;
    }

    Button {
        margin: 1;
    }

    Horizontal{
        height: auto;
        align: center middle;
    }
}
```

## Preview Screen

XML preview with syntax highlighting using TextArea.

### preview_xml_screen.py

```python
from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Header, TextArea


class PreviewXMLScreen(ModalScreen):
    CSS_PATH = "preview_xml_screen.tcss"

    def __init__(self, xml_file_path: str, *args: tuple, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.xml_file_path = xml_file_path
        self.title = "Preview XML"

    def compose(self) -> ComposeResult:
        with open(self.xml_file_path) as xml_file:
            xml = xml_file.read()
        text_area = TextArea(xml)
        text_area.language = "xml"
        yield Header()
        yield Vertical(
            text_area,
            Center(Button("Exit Preview", id="exit_preview", variant="primary")),
            id="exit_preview_ui",
        )

    @on(Button.Pressed, "#exit_preview")
    def on_exit_preview(self, event: Button.Pressed) -> None:
        self.dismiss()
```

**Key Pattern**: TextArea with `language="xml"` enables syntax highlighting

### preview_xml_screen.tcss

```css
PreviewXMLScreen {
    Button {
        margin: 1;
    }
}
```

## File Browser and Warning Screens

Reusable file selection and error dialogs.

### file_browser_screen.py

```python
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Grid, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Footer, Label, Header


class WarningScreen(Screen):
    """
    Creates a pop-up Screen that displays a warning message to the user
    """

    def __init__(self, warning_message: str) -> None:
        super().__init__()
        self.warning_message = warning_message

    def compose(self) -> ComposeResult:
        """
        Create the UI in the Warning Screen
        """
        yield Grid(
            Label(self.warning_message, id="warning_msg"),
            Button("OK", variant="primary", id="ok_warning"),
            id="warning_dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Event handler for when the OK button - dismisses the screen
        """
        self.dismiss()
        event.stop()


class FileBrowser(Screen):
    BINDINGS = [
        ("escape", "esc", "Exit dialog"),
    ]

    CSS_PATH = "file_browser_screen.tcss"

    class Selected(Message):
        """
        File selected message
        """

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self.selected_file = Path("")
        self.title = "Load XML Files"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Header(),
            DirectoryTree("/"),
            Center(
                Button("Load File", variant="primary", id="load_file"),
            ),
            id="file_browser_dialog",
        )

    @on(DirectoryTree.FileSelected)
    def on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """
        Called when the FileSelected Message is emitted from the DirectoryTree
        """
        self.selected_file = event.path

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Event handler for when the load file button is pressed
        """
        event.stop()

        if self.selected_file.suffix.lower() != ".xml" and self.selected_file.is_file():
            self.app.push_screen(WarningScreen("ERROR: You must choose a XML file!"))
            return

        self.post_message(self.Selected(self.selected_file))
        self.dismiss()

    def action_esc(self) -> None:
        """
        Close the dialog when the user presses ESC
        """
        self.dismiss()
```

**Key Pattern**: Custom Message class (`Selected`) for cross-screen communication

### file_browser_screen.tcss

```css
FileBrowser {
    #file_browser_dialog {
            width: 80%;
            height: 50%;
            border: thick $background 70%;
            content-align: center middle;
            margin: 2;
            border: solid green;
        }
    Button {
        margin: 1;
        content-align: center middle;
    }
}
```

## Save File Dialog

Directory selection with filename input.

### save_file_dialog.py

```python
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label


class SaveFileDialog(Screen):
    CSS_PATH = "save_file_dialog.tcss"

    def __init__(self) -> None:
        super().__init__()
        self.title = "Save File"
        self.root = "/"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Header(),
            Label(f"Folder name: {self.root}", id="folder"),
            DirectoryTree("/"),
            Input(placeholder="filename.txt", id="filename"),
            Button("Save File", variant="primary", id="save_file"),
            id="save_dialog",
        )

    def on_mount(self) -> None:
        """
        Focus the input widget so the user can name the file
        """
        self.query_one("#filename").focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Event handler for when the load file button is pressed
        """
        event.stop()
        filename = self.query_one("#filename").value
        full_path = Path(self.root) / filename
        self.dismiss(f"{full_path}")

    @on(DirectoryTree.DirectorySelected)
    def on_directory_selection(self, event: DirectoryTree.DirectorySelected) -> None:
        """
        Called when the DirectorySelected message is emitted from the DirectoryTree
        """
        self.root = event.path
        self.query_one("#folder").update(f"Folder name: {event.path}")
```

**Key Pattern**: Dynamic label updates as user navigates directory tree

### save_file_dialog.tcss

```css
SaveFileDialog {
    #save_dialog {
            width: 80%;
            height: 50%;
            border: thick $background 70%;
            content-align: center middle;
            margin: 2;
            border: solid green;
        }
    Button {
        margin: 1;
        content-align: center middle;
    }
}
```

## Implementation Patterns

### XML Parsing with lxml

```python
import lxml.etree as ET

# Parse XML file
xml_tree = ET.parse(xml_path)
xml_root = xml_tree.getroot()

# Access children
for child in xml_root.getchildren():
    print(child.tag, child.text)

# Add new element
element = ET.SubElement(parent_node, "new_tag")
element.text = "new value"

# Save changes
xml_tree.write(output_path)
```

### Tree Widget Lazy Loading

```python
# Only expand children when node is expanded
@on(Tree.NodeExpanded)
def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
    xml_obj = event.node.data
    if id(xml_obj) not in self.expanded:
        for child in xml_obj.getchildren():
            node = event.node.add_leaf(child.tag, data=child)
            node.allow_expand = child.getchildren() != []
        self.expanded[id(xml_obj)] = ""
```

### Custom Widget Data Storage

```python
class DataInput(Input):
    """Input widget that stores associated XML element"""

    def __init__(self, xml_obj: ET.Element, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xml_obj = xml_obj

# Usage
data_input = DataInput(xml_element, initial_text)
data_input.border_title = xml_element.tag
```

### Screen Callbacks

```python
# Parent screen pushes child screen with callback
def show_add_node_dialog(self) -> None:
    def on_result(result: tuple[str, str] | None) -> None:
        if result is not None:
            name, value = result
            self.update_xml(name, value)

    self.app.push_screen(AddNodeScreen(), on_result)

# Child screen returns data
def on_save(self) -> None:
    self.dismiss((self.name.value, self.value.value))
```

### Dynamic UI Updates

```python
# Clear and rebuild right pane when tree selection changes
@on(Tree.NodeSelected)
def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
    right_pane = self.query_one("#right_pane", VerticalScroll)
    right_pane.remove_children()  # Clear existing

    # Add new widgets
    for child in xml_obj.getchildren():
        input_widget = DataInput(child, child.text)
        right_pane.mount(Horizontal(input_widget))
```

### Live Data Binding

```python
# Update XML object immediately when input changes
@on(Input.Changed)
def on_input_changed(self, event: Input.Changed) -> None:
    xml_obj = event.input.xml_obj
    xml_obj.text = event.input.value
```

## Usage Example

```bash
# Install dependencies
python -m pip install textual lxml

# Run the application
python boomslang.py
```

**Keyboard Shortcuts:**
- `CTRL+O` - Open XML file
- `CTRL+S` - Save XML file
- `CTRL+A` - Add new node
- `P` - Preview XML
- `ESC` - Close current screen

**Workflow:**
1. Open XML file via file browser
2. Navigate tree structure (left pane)
3. Edit values in inputs (right pane)
4. Add new nodes with CTRL+A
5. Preview with P key
6. Save with CTRL+S

## Possible Enhancements

From [TUI XML Editor](https://www.blog.pythonlibrary.org/2025/07/30/tui-xml-editor/) tutorial:

**Suggested improvements:**
- Overwrite confirmation dialog for existing files
- Auto-save functionality using periodic workers
- XML validation before saving
- Attribute editing (currently only tag text)
- Search/filter in tree
- Undo/redo support
- Multiple file tabs
- XML schema validation
- Export to different formats

## Related Examples

See also:
- [examples/01-stopwatch.md](01-stopwatch.md) - Simple app structure
- [examples/02-markdown-browser.md](02-markdown-browser.md) - DirectoryTree usage
- [examples/03-calculator.md](03-calculator.md) - Button grids and CSS styling
- [architecture/06-tree.md](../architecture/06-tree.md) - Tree widget documentation

## Sources

**Primary Source:**
- [Creating a Simple XML Editor in Your Terminal with Python and Textual](https://www.blog.pythonlibrary.org/2025/07/30/tui-xml-editor/) - Mouse Vs Python blog (accessed 2025-11-02)

**Code Repository:**
- [BoomslangXML TUI v0.2.0](https://github.com/driscollis/BoomslangXML_TUI/releases/tag/v0.2.0) - GitHub repository

**Related Projects:**
- [BoomslangXML (wxPython)](https://github.com/driscollis/BoomslangXML) - Original wxPython version

**Additional References:**
- [lxml Documentation](https://lxml.de/) - XML parsing library
- [Textual Tree Widget](https://textual.textualize.io/widgets/tree/) - Official documentation
- [Textual TextArea Widget](https://textual.textualize.io/widgets/text_area/) - Syntax highlighting support
