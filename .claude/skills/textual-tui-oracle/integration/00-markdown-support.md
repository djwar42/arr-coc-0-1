# Textual Markdown Support Integration Guide

## Overview

Textual provides first-class Markdown rendering capabilities through the `Markdown` and `MarkdownViewer` widgets. With Textual 5.0+, you can display formatted Markdown documents directly in terminal applications, including tables, syntax-highlighted code blocks, typography, and streaming content.

**Key Features:**
- Native Markdown rendering in terminal
- Syntax highlighting for code blocks
- Table support with alignment
- Streaming content updates
- Interactive links and anchors
- Table of contents generation
- Typography (emphasis, strong, inline code)
- Block quotes and lists
- Customizable styling via CSS

## Basic Markdown Widget Usage

### Simple Markdown Display

The most basic usage displays a Markdown string:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown

EXAMPLE_MARKDOWN = """
# Welcome to Textual

This is a **Markdown** document with *emphasis* and `inline code`.

- Item 1
- Item 2
- Item 3
"""

class MarkdownApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown(EXAMPLE_MARKDOWN)

if __name__ == "__main__":
    app = MarkdownApp()
    app.run()
```

### Loading Markdown from Files

Load external Markdown files asynchronously:

```python
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Markdown

class MarkdownFileApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown()

    async def on_mount(self) -> None:
        markdown = self.query_one(Markdown)
        await markdown.load(Path("README.md"))

if __name__ == "__main__":
    app = MarkdownFileApp()
    app.run()
```

**Error Handling:**
```python
async def load_markdown_safely(self, path: Path) -> None:
    markdown = self.query_one(Markdown)
    try:
        await markdown.load(path)
    except OSError as e:
        self.notify(f"Failed to load {path}: {e}", severity="error")
```

### Dynamic Content Updates

Update Markdown content dynamically with `update()`:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Button
from textual.containers import Container

class DynamicMarkdownApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown("# Initial Content")
        yield Button("Update Content", id="update")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        markdown = self.query_one(Markdown)
        new_content = "# Updated!\n\nContent has been **refreshed**."
        markdown.update(new_content)
```

**Awaiting Updates:**
```python
async def update_markdown(self) -> None:
    markdown = self.query_one(Markdown)
    # Ensure all children are mounted before continuing
    await markdown.update("# New Content\n\nFully rendered.")
    # Safe to perform actions that depend on rendered content
```

### Appending Content

Append Markdown fragments to existing content:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Button

class AppendMarkdownApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown("# Log\n\n")
        yield Button("Add Entry", id="add")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        markdown = self.query_one(Markdown)
        markdown.append("- New log entry\n")
```

**Awaiting Append:**
```python
async def append_and_scroll(self) -> None:
    markdown = self.query_one(Markdown)
    await markdown.append("- Entry appended\n")
    # Now safe to scroll to bottom
    markdown.scroll_end(animate=False)
```

## Advanced Features

### Streaming Markdown Content

For high-frequency updates (>20 appends/second), use `MarkdownStream` to combine updates:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown
from textual.containers import VerticalScroll
from textual.worker import work

class StreamingApp(App):
    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Markdown("# Streaming Log\n\n")

    async def on_mount(self) -> None:
        self.stream_content()

    @work
    async def stream_content(self) -> None:
        """Stream markdown content efficiently."""
        markdown_widget = self.query_one(Markdown)
        container = self.query_one(VerticalScroll)
        container.anchor()  # Auto-scroll to bottom

        stream = Markdown.get_stream(markdown_widget)
        try:
            # Simulate fetching chunks from network/file
            for i in range(1000):
                chunk = f"- Log entry {i}\n"
                await stream.write(chunk)
                await asyncio.sleep(0.01)  # Simulate delay
        finally:
            await stream.stop()
```

**Benefits of Streaming:**
- Combines multiple updates into single render passes
- Maintains UI responsiveness during rapid updates
- Essential for network data, live logs, or real-time feeds

### Syntax Highlighted Code Blocks

Markdown automatically syntax highlights code blocks:

````python
from textual.widgets import Markdown

CODE_EXAMPLE = """
## Python Example

```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## JavaScript Example

```javascript
function factorial(n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
```
"""

class CodeApp(App):
    def compose(self) -> ComposeResult:
        markdown = Markdown(CODE_EXAMPLE)
        markdown.code_indent_guides = False  # Disable indent guides
        yield markdown
````

**Code Indent Guides:**
```python
# Enable/disable indent guides for code blocks
markdown.code_indent_guides = True  # Show guides (default)
markdown.code_indent_guides = False  # Hide guides
```

### Tables with Alignment

Full table support with column alignment:

```python
TABLE_MARKDOWN = """
| Name            | Type   | Default | Description                        |
| --------------- | ------ | ------- | ---------------------------------- |
| `show_header`   | `bool` | `True`  | Show the table header              |
| `fixed_rows`    | `int`  | `0`     | Number of fixed rows               |
| `fixed_columns` | `int`  | `0`     | Number of fixed columns            |
| `cursor_type`   | `str`  | `cell`  | Cursor selection type              |
"""

class TableApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown(TABLE_MARKDOWN)
```

**Table Features:**
- Automatic column width calculation
- Left/center/right alignment via standard Markdown syntax
- Header row styling
- Scrollable content for large tables

### Interactive Links

Handle link clicks with message events:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown

LINK_MARKDOWN = """
# Links

- [Textual Documentation](https://textual.textualize.io)
- [GitHub Repository](https://github.com/Textualize/textual)
- [Internal anchor](#section)

## Section

Content here.
"""

class LinkApp(App):
    def compose(self) -> ComposeResult:
        # Set open_links=False to handle clicks manually
        yield Markdown(LINK_MARKDOWN, open_links=False)

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks."""
        self.notify(f"Clicked: {event.href}")

        # Custom link handling
        if event.href.startswith("#"):
            # Internal anchor
            event.markdown.goto_anchor(event.href[1:])
        elif event.href.startswith("http"):
            # External link - open in browser
            import webbrowser
            webbrowser.open(event.href)
```

**Link Handling Modes:**
- `open_links=True` (default): Automatically open external links in browser
- `open_links=False`: Handle `LinkClicked` events manually

### Anchors and Navigation

Navigate to document sections programmatically:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Button
from textual.containers import Horizontal

LONG_DOCUMENT = """
# Introduction

Content here...

# Section 1

More content...

# Section 2

Even more content...
"""

class AnchorApp(App):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Intro", id="intro")
            yield Button("Section 1", id="section-1")
            yield Button("Section 2", id="section-2")
        yield Markdown(LONG_DOCUMENT)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        markdown = self.query_one(Markdown)

        # goto_anchor uses GitHub-style slugging
        if markdown.goto_anchor(event.button.id):
            self.notify(f"Jumped to {event.button.id}")
        else:
            self.notify(f"Anchor {event.button.id} not found", severity="warning")
```

**Anchor Slugging:**
- Converts headings to lowercase
- Replaces spaces with hyphens
- Removes special characters
- Example: "Section 1" → "section-1"

### Table of Contents

Access and navigate document structure:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Tree
from textual.containers import Horizontal

class TOCApp(App):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("Contents", id="toc")
            yield Markdown(id="content")

    async def on_mount(self) -> None:
        markdown = self.query_one("#content", Markdown)
        await markdown.load(Path("document.md"))
        self.update_toc()

    def on_markdown_table_of_contents_updated(
        self, event: Markdown.TableOfContentsUpdated
    ) -> None:
        """Rebuild TOC tree when document changes."""
        self.update_toc()

    def update_toc(self) -> None:
        markdown = self.query_one("#content", Markdown)
        tree = self.query_one("#toc", Tree)
        tree.clear()

        root = tree.root
        for entry in markdown.table_of_contents:
            root.add_leaf(f"{entry.label} ({entry.level})")

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Jump to selected TOC entry."""
        markdown = self.query_one("#content", Markdown)
        toc = markdown.table_of_contents
        if event.node.data is not None:
            entry = toc[event.node.data]
            markdown.goto_anchor(entry.anchor)
```

**TOC Structure:**
```python
# Each entry contains:
# - label: str (heading text)
# - level: int (1-6 for h1-h6)
# - anchor: str (slugified anchor)
```

### Custom Markdown Parser

Use custom MarkdownIt parser configurations:

```python
from markdown_it import MarkdownIt
from textual.widgets import Markdown

def custom_parser_factory() -> MarkdownIt:
    """Create a custom MarkdownIt parser."""
    parser = MarkdownIt("gfm-like")  # GitHub-flavored Markdown

    # Enable additional features
    parser.enable(["table", "strikethrough"])

    # Customize options
    parser.options["linkify"] = True  # Auto-link URLs
    parser.options["typographer"] = True  # Smart quotes

    return parser

class CustomParserApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown(
            "Content here...",
            parser_factory=custom_parser_factory
        )
```

**Parser Options:**
- `"gfm-like"`: GitHub-flavored Markdown (default)
- `"commonmark"`: Strict CommonMark
- `"zero"`: All features disabled (enable selectively)

### Custom Block Rendering

Extend Markdown with custom block widgets:

```python
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownBlock

class CustomAlert(MarkdownBlock):
    """Custom alert block."""

    DEFAULT_CSS = """
    CustomAlert {
        background: $warning;
        padding: 1;
        border: solid $warning-darken-2;
    }
    """

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(f"⚠️ {self.content}")

class ExtendedMarkdown(Markdown):
    """Markdown with custom blocks."""

    def unhandled_token(self, token) -> MarkdownBlock | None:
        """Handle custom tokens."""
        if token.type == "fence" and token.info == "alert":
            return CustomAlert(token.content)
        return None

# Usage
class CustomBlockApp(App):
    def compose(self) -> ComposeResult:
        yield ExtendedMarkdown("""
# Document

```alert
This is a custom alert block!
```
""")
```

## MarkdownViewer Widget

`MarkdownViewer` extends `Markdown` with browser-like features:

```python
from textual.app import App, ComposeResult
from textual.widgets import MarkdownViewer

class ViewerApp(App):
    def compose(self) -> ComposeResult:
        # MarkdownViewer adds:
        # - Automatic table of contents sidebar
        # - Better navigation
        # - Document history
        yield MarkdownViewer()

    async def on_mount(self) -> None:
        viewer = self.query_one(MarkdownViewer)
        await viewer.go(Path("README.md"))
```

**MarkdownViewer Features:**
- Built-in TOC panel (toggle with key binding)
- Document navigation history
- Better link handling
- Search capabilities

## Tabbed Markdown Content

Combine Markdown with TabbedContent for multi-page layouts:

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown, TabbedContent, TabPane

class TabbedMarkdownApp(App):
    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Introduction", id="intro"):
                yield Markdown("""
# Introduction

Welcome to the documentation.
""")

            with TabPane("Tutorial", id="tutorial"):
                yield Markdown("""
# Tutorial

Step-by-step guide here.
""")

            with TabPane("API Reference", id="api"):
                yield Markdown("""
# API Reference

Complete API documentation.
""")

if __name__ == "__main__":
    app = TabbedMarkdownApp()
    app.run()
```

**Dynamic Tab Loading:**
```python
from textual.widgets import TabbedContent, TabPane, Markdown

class DynamicTabsApp(App):
    async def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Load markdown when tab is activated."""
        tab_id = event.tab.id
        pane = self.query_one(f"#{tab_id}", TabPane)

        # Load content on demand
        if not pane.children:
            markdown = Markdown()
            await pane.mount(markdown)
            await markdown.load(Path(f"docs/{tab_id}.md"))
```

## Styling Markdown Widgets

### Component Classes

Textual provides component classes for styling Markdown elements:

```css
/* Customize code blocks */
Markdown > .code_block {
    background: $surface;
    border: solid $primary;
    padding: 1;
}

/* Style block quotes */
Markdown > .block_quote {
    background: $panel;
    border-left: solid $accent;
    margin: 1;
    padding: 1;
}

/* Headers */
Markdown > .h1 {
    color: $primary;
    text-style: bold;
}

Markdown > .h2 {
    color: $secondary;
}

/* Lists */
Markdown > .bullet_list {
    padding-left: 2;
}

/* Tables */
Markdown > .table {
    border: solid $primary;
}
```

**Available Component Classes:**
- `.h1` - `.h6`: Headers
- `.code_block`: Code blocks
- `.block_quote`: Block quotes
- `.paragraph`: Paragraphs
- `.bullet_list`: Unordered lists
- `.ordered_list`: Ordered lists
- `.table`: Tables
- `.horizontal_rule`: Horizontal rules

### Custom Styling Example

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown

class StyledMarkdownApp(App):
    CSS = """
    Markdown {
        background: $surface;
        padding: 2;
    }

    Markdown > .h1 {
        color: $accent;
        text-style: bold;
        padding: 1 0;
    }

    Markdown > .code_block {
        background: $panel;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
    }

    Markdown > .block_quote {
        background: $surface-darken-1;
        border-left: thick $accent;
        padding: 0 2;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Markdown("""
# Styled Document

This document has custom styling.

> This is a styled quote.

```python
print("Styled code block")
```
""")
```

## Messages and Events

### LinkClicked Message

```python
from textual.widgets import Markdown

@on(Markdown.LinkClicked)
def handle_link(self, event: Markdown.LinkClicked) -> None:
    """Handle link clicks."""
    link = event.href  # The URL/anchor clicked
    markdown = event.markdown  # The Markdown widget

    # Custom handling logic
    if link.startswith("#"):
        event.markdown.goto_anchor(link[1:])
    else:
        self.notify(f"External link: {link}")
```

### TableOfContentsUpdated Message

```python
from textual.widgets import Markdown

@on(Markdown.TableOfContentsUpdated)
def handle_toc_update(self, event: Markdown.TableOfContentsUpdated) -> None:
    """Handle TOC updates."""
    toc = event.table_of_contents
    markdown = event.markdown

    # Update UI with new TOC
    for entry in toc:
        self.log(f"Level {entry.level}: {entry.label}")
```

### TableOfContentsSelected Message

```python
from textual.widgets import Markdown

@on(Markdown.TableOfContentsSelected)
def handle_toc_selection(self, event: Markdown.TableOfContentsSelected) -> None:
    """Handle TOC item selection."""
    block_id = event.block_id  # ID of selected block
    markdown = event.markdown

    # Navigate to selected section
    markdown.goto_anchor(block_id)
```

## Best Practices

### Performance Optimization

**Use Streaming for Rapid Updates:**
```python
# Instead of this (slow for high frequency):
for chunk in chunks:
    await markdown.append(chunk)

# Use this:
stream = Markdown.get_stream(markdown)
try:
    for chunk in chunks:
        await stream.write(chunk)
finally:
    await stream.stop()
```

**Lazy Loading for Large Documents:**
```python
# Load large documents asynchronously
async def load_large_doc(self) -> None:
    markdown = self.query_one(Markdown)

    # Show loading indicator
    self.notify("Loading document...")

    # Load in background
    await markdown.load(Path("large_document.md"))

    self.notify("Document loaded!")
```

### Content Organization

**Structure Multi-Section Content:**
```python
# Use TabbedContent for distinct topics
# Use single Markdown with TOC for related content
# Use MarkdownViewer for documentation-style content
```

**File Management:**
```python
from pathlib import Path

class DocApp(App):
    DOCS_DIR = Path(__file__).parent / "docs"

    async def load_doc(self, name: str) -> None:
        markdown = self.query_one(Markdown)
        doc_path = self.DOCS_DIR / f"{name}.md"

        if doc_path.exists():
            await markdown.load(doc_path)
        else:
            markdown.update(f"# Error\n\nDocument '{name}' not found.")
```

### Accessibility

**Provide Navigation:**
```python
# Always provide TOC for long documents
# Use MarkdownViewer for complex documentation
# Implement keyboard shortcuts for common actions
```

**Handle Link Failures:**
```python
@on(Markdown.LinkClicked)
def handle_link(self, event: Markdown.LinkClicked) -> None:
    if event.href.startswith("#"):
        if not event.markdown.goto_anchor(event.href[1:]):
            self.notify(f"Section not found: {event.href}", severity="warning")
```

### Error Handling

**Graceful Degradation:**
```python
async def safe_load(self, path: Path) -> None:
    markdown = self.query_one(Markdown)

    try:
        await markdown.load(path)
    except FileNotFoundError:
        markdown.update("# Error\n\nFile not found.")
    except PermissionError:
        markdown.update("# Error\n\nPermission denied.")
    except Exception as e:
        markdown.update(f"# Error\n\n```\n{e}\n```")
        self.log.error(f"Failed to load {path}: {e}")
```

## Common Patterns

### Documentation Viewer

```python
from textual.app import App, ComposeResult
from textual.widgets import MarkdownViewer, Tree
from textual.containers import Horizontal
from pathlib import Path

class DocsApp(App):
    """Documentation browser with navigation."""

    DOCS_DIR = Path("docs")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("Documentation", id="nav")
            yield MarkdownViewer(id="viewer")

    def on_mount(self) -> None:
        self.populate_nav()

    def populate_nav(self) -> None:
        tree = self.query_one("#nav", Tree)
        root = tree.root

        for doc in sorted(self.DOCS_DIR.glob("*.md")):
            root.add_leaf(doc.stem, data=doc)

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data:
            viewer = self.query_one("#viewer", MarkdownViewer)
            await viewer.go(event.node.data)
```

### Live Markdown Editor

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown, TextArea
from textual.containers import Horizontal

class MarkdownEditor(App):
    """Split-pane Markdown editor with live preview."""

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield TextArea(id="editor")
            yield Markdown(id="preview")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update preview when text changes."""
        markdown = self.query_one("#preview", Markdown)
        markdown.update(event.text_area.text)
```

### Log Viewer

```python
from textual.app import App, ComposeResult
from textual.widgets import Markdown
from textual.containers import VerticalScroll
from textual.worker import work
import asyncio

class LogViewer(App):
    """Real-time log viewer with Markdown formatting."""

    def compose(self) -> ComposeResult:
        with VerticalScroll() as scroll:
            scroll.can_focus = False
            yield Markdown("# System Log\n\n")

    async def on_mount(self) -> None:
        self.stream_logs()

    @work
    async def stream_logs(self) -> None:
        markdown = self.query_one(Markdown)
        container = self.query_one(VerticalScroll)
        container.anchor()  # Auto-scroll

        stream = Markdown.get_stream(markdown)
        try:
            while True:
                # Simulate log generation
                log_entry = f"- `{asyncio.get_event_loop().time():.2f}`: Log message\n"
                await stream.write(log_entry)
                await asyncio.sleep(0.5)
        finally:
            await stream.stop()
```

## Sources

**Medium Article:**
- [Building Modern Terminal Apps in Python with Textual and Markdown Support](https://medium.com/towardsdev/building-modern-terminal-apps-in-python-with-textual-and-markdown-support-4bb3e25e49db) - Py-Core Python Programming (accessed 2025-11-02)
  - Overview of Textual Markdown capabilities
  - Integration patterns and use cases
  - Textual 5.0 features

**Official Documentation:**
- [Textual Markdown Widget](https://textual.textualize.io/widgets/markdown/) - Textual Documentation (accessed 2025-11-02)
  - Complete API reference
  - Component classes and styling
  - Messages and events
  - Code examples

**Community Tutorials:**
- [Textual 101 - Using the TabbedContent Widget](https://www.blog.pythonlibrary.org/2023/04/25/textual-101-using-the-tabbedcontent-widget/) - Mouse Vs Python (accessed 2025-11-02)
  - TabbedContent + Markdown integration
  - Multi-page Markdown layouts

**Web Research:**
- Google search: "Textual Python Markdown widget examples tutorial 2024 2025" (accessed 2025-11-02)
