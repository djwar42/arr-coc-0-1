# Tree Widget - Comprehensive Guide

## Overview

The Tree widget provides a hierarchical tree structure for displaying nested data in Textual TUI applications. It supports expandable/collapsible nodes, navigation, custom labels, and dynamic tree construction.

**From [Tree - Textual Documentation](https://textual.textualize.io/widgets/tree/) (accessed 2025-11-02)**

## Key Features

### Core Capabilities
- Hierarchical data display
- Expand/collapse nodes
- Keyboard and mouse navigation
- Custom node labels and icons
- Dynamic node addition/removal
- Guide lines for visual hierarchy
- Data association with nodes
- Event handling for selection

### Widget Characteristics
- **Focusable**: Yes
- **Container**: Yes (contains TreeNode objects)

## Basic Usage

### Simple Tree Example

```python
from textual.app import App, ComposeResult
from textual.widgets import Tree

class TreeApp(App):
    def compose(self) -> ComposeResult:
        tree = Tree("Root")
        tree.root.add("Child 1")
        tree.root.add("Child 2")
        yield tree
```

**From [Tree Widget - Textual Documentation](https://textual.textualize.io/widgets/tree/) (accessed 2025-11-02)**

## Working with TreeNodes

### Creating and Adding Nodes

```python
# Access root node
tree = Tree("My Tree")
root = tree.root

# Add child nodes
node1 = root.add("Parent Node")
node1.add("Child A")
node1.add("Child B")

# Add leaf nodes (non-expandable)
root.add_leaf("Leaf Node")

# Chain additions
root.add("Folder").add("Subfolder").add_leaf("File")
```

### Node Properties

```python
# Set node label
node.label = "New Label"

# Check if node has children
if node.children:
    print("Has children")

# Check if expanded
if node.is_expanded:
    node.collapse()
else:
    node.expand()

# Associate data with node
node.data = {"id": 123, "type": "folder"}
```

## Navigation and Interaction

### Keyboard Bindings

From the official documentation, Tree provides these default bindings:

| Key | Action | Description |
|-----|--------|-------------|
| Enter | Select node | Trigger NodeSelected event |
| Space | Toggle expansion | Expand/collapse node |
| Up/Down | Navigate | Move between visible nodes |
| Right | Expand | Expand current node |
| Left | Collapse | Collapse current node |

### Mouse Interaction

```python
@on(Tree.NodeSelected)
def handle_node_selected(self, event: Tree.NodeSelected) -> None:
    """Handle node selection via mouse or keyboard."""
    node = event.node
    self.notify(f"Selected: {node.label}")

@on(Tree.NodeExpanded)
def handle_node_expanded(self, event: Tree.NodeExpanded) -> None:
    """Handle node expansion."""
    node = event.node
    # Lazy load children here
    self.load_children(node)

@on(Tree.NodeCollapsed)
def handle_node_collapsed(self, event: Tree.NodeCollapsed) -> None:
    """Handle node collapse."""
    pass
```

## Advanced Features

### Custom Labels and Icons

```python
from rich.text import Text

# Text with styling
label = Text("Folder", style="bold blue")
node = tree.root.add(label)

# With icons/emojis
node = tree.root.add("📁 Documents")
node.add("📄 file1.txt")
node.add("📄 file2.txt")

# Rich renderables
from rich.panel import Panel
custom_label = Panel("Complex Label", border_style="green")
tree.root.add(custom_label)
```

**From [Textual Widget Gallery](https://www.reddit.com/r/Python/comments/11kw6ev/textual_widget_gallery/) - Reddit discussion (accessed 2025-11-02)**

### Dynamic Tree Construction

```python
def build_file_tree(self, path: Path, node: TreeNode) -> None:
    """Recursively build tree from filesystem."""
    try:
        for item in sorted(path.iterdir()):
            if item.is_dir():
                folder_node = node.add(f"📁 {item.name}")
                folder_node.data = item
                # Lazy loading - only expand on demand
            else:
                node.add_leaf(f"📄 {item.name}", data=item)
    except PermissionError:
        node.add_leaf("⛔ Access Denied")
```

### Lazy Loading

```python
class LazyTree(Tree):
    def on_mount(self) -> None:
        # Create root with placeholder
        self.root.expand()

    @on(Tree.NodeExpanded)
    def load_children(self, event: Tree.NodeExpanded) -> None:
        """Load children when node is expanded."""
        node = event.node

        # Check if already loaded
        if node.children:
            return

        # Load children from data source
        if hasattr(node, 'data') and node.data:
            self.populate_node(node, node.data)

    def populate_node(self, node: TreeNode, path: Path) -> None:
        """Populate node with children from filesystem."""
        if path.is_dir():
            for item in path.iterdir():
                if item.is_dir():
                    child = node.add(f"📁 {item.name}")
                    child.data = item
                else:
                    node.add_leaf(f"📄 {item.name}", data=item)
```

## DirectoryTree Widget

Textual provides a specialized `DirectoryTree` widget for filesystem navigation:

```python
from textual.widgets import DirectoryTree

class FileExplorer(App):
    def compose(self) -> ComposeResult:
        # Create tree for specific directory
        yield DirectoryTree("./")

    @on(DirectoryTree.FileSelected)
    def handle_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection."""
        self.notify(f"Selected: {event.path}")
```

**From [DirectoryTree - Textual Documentation](https://textual.textualize.io/widgets/directory_tree/) (accessed 2025-11-02)**

## Messages and Events

### Key Events

**NodeSelected**
```python
@on(Tree.NodeSelected)
def on_node_selected(self, event: Tree.NodeSelected) -> None:
    """Fired when a node is selected (Enter key or click)."""
    node = event.node
    label = node.label
    data = node.data
```

**NodeExpanded**
```python
@on(Tree.NodeExpanded)
def on_node_expanded(self, event: Tree.NodeExpanded) -> None:
    """Fired when a node is expanded."""
    node = event.node
    # Ideal place for lazy loading
```

**NodeCollapsed**
```python
@on(Tree.NodeCollapsed)
def on_node_collapsed(self, event: Tree.NodeCollapsed) -> None:
    """Fired when a node is collapsed."""
    pass
```

**NodeHighlighted**
```python
@on(Tree.NodeHighlighted)
def on_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
    """Fired when cursor moves to a node."""
    self.update_status(event.node.label)
```

## Reactive Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_root` | bool | True | Whether to show the root node |
| `show_guides` | bool | True | Whether to show guide lines |
| `guide_depth` | int | 4 | Number of levels to show guides |

## Styling

```python
# In CSS
Tree {
    background: $surface;
}

Tree > .tree--cursor {
    background: $secondary;
}

Tree > .tree--guides {
    color: $primary-lighten-2;
}

Tree > .tree--highlight {
    background: $accent;
}
```

## Component Classes

| Class | Description |
|-------|-------------|
| `tree--cursor` | Target the cursor |
| `tree--guides` | Target the guide lines |
| `tree--highlight` | Target highlighted nodes |
| `tree--label` | Target node labels |

## Practical Examples

### Configuration Tree

```python
class ConfigTree(Tree):
    def __init__(self, config_dict: dict):
        super().__init__("Configuration")
        self.build_tree(self.root, config_dict)

    def build_tree(self, node: TreeNode, data: dict) -> None:
        """Recursively build tree from dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                child = node.add(f"📂 {key}")
                self.build_tree(child, value)
            elif isinstance(value, list):
                child = node.add(f"📋 {key}")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        subchild = child.add(f"[{i}]")
                        self.build_tree(subchild, item)
                    else:
                        child.add_leaf(f"[{i}] = {item}")
            else:
                node.add_leaf(f"{key} = {value}")
```

### JSON Tree Viewer

```python
import json

class JSONTree(Tree):
    def load_json(self, filepath: str) -> None:
        """Load and display JSON data."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.root.label = filepath
        self._build_json_tree(self.root, data)

    def _build_json_tree(self, node: TreeNode, data) -> None:
        """Recursively build tree from JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    child = node.add(f"🔑 {key}")
                    self._build_json_tree(child, value)
                else:
                    node.add_leaf(f"{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    child = node.add(f"[{i}]")
                    self._build_json_tree(child, item)
                else:
                    node.add_leaf(f"[{i}] = {item}")
```

### Interactive Menu Tree

```python
class MenuTree(App):
    def compose(self) -> ComposeResult:
        tree = Tree("Main Menu")
        tree.root.expand()

        file_menu = tree.root.add("File")
        file_menu.add_leaf("New")
        file_menu.add_leaf("Open")
        file_menu.add_leaf("Save")
        file_menu.add_leaf("Exit")

        edit_menu = tree.root.add("Edit")
        edit_menu.add_leaf("Copy")
        edit_menu.add_leaf("Paste")

        yield tree

    @on(Tree.NodeSelected)
    def handle_menu_selection(self, event: Tree.NodeSelected) -> None:
        """Execute menu action."""
        label = str(event.node.label)
        self.execute_action(label)
```

## Common Patterns

### Search/Filter Tree

```python
def filter_tree(self, search_term: str) -> None:
    """Highlight nodes matching search term."""
    def search_nodes(node: TreeNode) -> bool:
        matches = search_term.lower() in str(node.label).lower()
        for child in node.children:
            if search_nodes(child):
                matches = True
                node.expand()
        return matches

    search_nodes(self.tree.root)
```

### Tree State Persistence

```python
def save_tree_state(self) -> dict:
    """Save which nodes are expanded."""
    state = {}

    def save_node(node: TreeNode, path: str) -> None:
        current_path = f"{path}/{node.label}"
        state[current_path] = node.is_expanded
        for child in node.children:
            save_node(child, current_path)

    save_node(self.tree.root, "")
    return state

def restore_tree_state(self, state: dict) -> None:
    """Restore expansion state."""
    def restore_node(node: TreeNode, path: str) -> None:
        current_path = f"{path}/{node.label}"
        if current_path in state and state[current_path]:
            node.expand()
        for child in node.children:
            restore_node(child, current_path)

    restore_node(self.tree.root, "")
```

## Performance Considerations

1. **Lazy Loading**: Only load children when nodes are expanded
2. **Virtual Rendering**: Tree only renders visible nodes
3. **Batch Updates**: Add multiple nodes before refresh
4. **Data Association**: Use `node.data` for quick access instead of searching

**From [Python Textual: Build Beautiful UIs](https://realpython.com/python-textual/) - Real Python (accessed 2025-11-02)**

## Sources

**Official Documentation:**
- [Tree Widget - Textual](https://textual.textualize.io/widgets/tree/) (accessed 2025-11-02)
- [DirectoryTree Widget - Textual](https://textual.textualize.io/widgets/directory_tree/) (accessed 2025-11-02)
- [Widgets Guide - Textual](https://textual.textualize.io/guide/widgets/) (accessed 2025-11-02)

**Community Resources:**
- [Textual Widget Gallery - Reddit](https://www.reddit.com/r/Python/comments/11kw6ev/textual_widget_gallery/) (accessed 2025-11-02)
- [Python Textual Tutorial - Real Python](https://realpython.com/python-textual/) (accessed 2025-11-02)

**GitHub Resources:**
- [Textual GitHub Repository](https://github.com/Textualize/textual) - Official examples and source code

## See Also

- [widgets/00-datatable-guide.md](00-datatable-guide.md) - DataTable widget patterns
- [widgets/02-input-validation.md](02-input-validation.md) - Input widget with validation
- [core-concepts/](../core-concepts/) - Understanding widget composition
