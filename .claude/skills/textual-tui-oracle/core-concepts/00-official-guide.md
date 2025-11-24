# Textual Official Guide - In-Depth Reference

**Source**: [https://textual.textualize.io/guide/](https://textual.textualize.io/guide/)
**Accessed**: 2025-11-02

## Overview

The Textual Guide provides in-depth reference documentation on how to build apps with Textual. It covers all major concepts, patterns, and features of the framework.

**Key Quote**: "Welcome to the Textual Guide! An in-depth reference on how to build apps with Textual."

---

## Guide Structure

The guide is organized into 18 major topics:

### 1. [Devtools](https://textual.textualize.io/guide/devtools/)
Development tools for debugging and live coding

### 2. [App Basics](https://textual.textualize.io/guide/app/)
Core application structure and lifecycle

### 3. [Styles](https://textual.textualize.io/guide/styles/)
Styling widgets programmatically

### 4. [Textual CSS](https://textual.textualize.io/guide/CSS/)
CSS-like styling system for terminal UIs

### 5. [DOM Queries](https://textual.textualize.io/guide/queries/)
Querying and selecting widgets in the DOM tree

### 6. [Layout](https://textual.textualize.io/guide/layout/)
Layout systems (grid, dock, vertical, horizontal)

### 7. [Events and Messages](https://textual.textualize.io/guide/events/)
Event handling and message passing

### 8. [Input](https://textual.textualize.io/guide/input/)
Keyboard and mouse input handling

### 9. [Actions](https://textual.textualize.io/guide/actions/)
Keyboard bindings and action system

### 10. [Reactivity](https://textual.textualize.io/guide/reactivity/)
Reactive programming for dynamic UIs

### 11. [Themes](https://textual.textualize.io/guide/design/)
Theme system and design tokens

### 12. [Widgets](https://textual.textualize.io/guide/widgets/)
Creating custom widgets and widget composition

### 13. [Content](https://textual.textualize.io/guide/content/)
Rendering content in widgets

### 14. [Animation](https://textual.textualize.io/guide/animation/)
Animating widget properties

### 15. [Screens](https://textual.textualize.io/guide/screens/)
Multiple screens and navigation

### 16. [Workers](https://textual.textualize.io/guide/workers/)
Background tasks and async operations

### 17. [Command Palette](https://textual.textualize.io/guide/command_palette/)
Searchable command interface

### 18. [Testing](https://textual.textualize.io/guide/testing/)
Testing apps with Pilot

---

## Example Code

**Important Note**: Most code in the guide is fully working and can be cut and pasted.

**Easier approach**: Check out the [Textual repository](https://github.com/Textualize/textual) and navigate to the `docs/examples/guide` directory to run examples directly.

---

## Key Concepts from Guide Topics

### App Basics
- **App class**: The main entry point for Textual applications
- **Lifecycle**: Startup, mount, compose, run, shutdown
- **Configuration**: Title, CSS, bindings, dark mode

### Styles and CSS
- **Inline styles**: Modify widget styles programmatically
- **CSS files**: External stylesheets
- **CSS syntax**: Textual-specific CSS properties
- **Selectors**: Class, ID, widget type, pseudo-classes

### DOM Queries
- **Query syntax**: `self.query(Widget)`, `self.query_one("#id")`
- **Filtering**: By type, class, ID
- **Actions**: `remove()`, `add_class()`, `set()`

### Layout Systems
- **Vertical**: Stack widgets vertically (default)
- **Horizontal**: Arrange widgets side-by-side
- **Grid**: 2D grid layout with rows/columns
- **Dock**: Dock widgets to edges (top, bottom, left, right)

### Events and Messages
- **Event bubbling**: Events propagate up the DOM tree
- **Event handling**: Decorated methods with `@on()`
- **Custom messages**: Create app-specific messages
- **Message posting**: `self.post_message()`

### Input Handling
- **Keyboard events**: `Key`, `KeyDown`, `KeyUp`
- **Mouse events**: `Click`, `MouseMove`, `MouseDown`, `MouseUp`
- **Input widgets**: `Input`, `TextArea` for text entry
- **Focus management**: Tab navigation and focus events

### Actions
- **Action binding**: Map keys to actions
- **Built-in actions**: `quit`, `focus_next`, `focus_previous`
- **Custom actions**: Define `action_*` methods
- **Action string**: `"action_name(param1, param2)"`

### Reactivity
- **Reactive attributes**: `reactive(default_value)`
- **Watchers**: `watch_attribute_name(old, new)`
- **Computed properties**: `@property` with reactive dependencies
- **Reactive updates**: Automatic UI updates on data changes

### Themes and Design
- **Color system**: Named colors, hex colors, RGB
- **Design tokens**: Variables for consistent styling
- **Built-in themes**: Multiple themes included
- **Custom themes**: Define your own color schemes

### Widgets
- **Widget composition**: Nest widgets to build UIs
- **Custom widgets**: Subclass `Widget` or `Static`
- **Widget lifecycle**: Mount, compose, render
- **40+ built-in widgets**: Buttons, inputs, tables, trees, etc.

### Screens
- **Screen stack**: Push/pop screens for navigation
- **Modal screens**: Overlay screens that return values
- **Screen switching**: `push_screen()`, `pop_screen()`, `switch_screen()`
- **Screen isolation**: Each screen has its own widgets

### Workers
- **Background tasks**: Run async code without blocking UI
- **Worker API**: `@work` decorator
- **Progress tracking**: Report progress from workers
- **Cancellation**: Cancel workers on screen exit

### Command Palette
- **Built-in commands**: Search and execute app commands
- **Custom commands**: Add your own command providers
- **Fuzzy search**: Fast command discovery
- **Keyboard shortcut**: Default: Ctrl+Backslash

### Testing
- **Pilot**: Automated testing framework
- **Simulated input**: Keyboard and mouse events
- **Async tests**: Use `async def` for tests
- **Assertions**: Check widget state and behavior

---

## Reference Documentation

The guide links to comprehensive reference documentation:

### CSS Types
Data types used in Textual CSS:
- `<border>`: Border styles (solid, dashed, etc.)
- `<color>`: Color values (named, hex, RGB)
- `<scalar>`: Numeric values with units (%, fr, w, h)
- `<text-align>`: Text alignment (left, center, right)
- And 15+ more types

### Events
27 event types including:
- **App events**: AppBlur, AppFocus
- **Focus events**: Focus, Blur, DescendantFocus, DescendantBlur
- **Keyboard events**: Key
- **Mouse events**: Click, MouseMove, MouseDown, MouseUp, MouseCapture
- **Lifecycle events**: Mount, Unmount, Show, Hide
- **Other events**: Resize, Print, Paste

### Styles
70+ CSS-like style properties:
- **Layout**: display, layout, dock
- **Sizing**: width, height, min-width, max-width, etc.
- **Spacing**: margin, padding
- **Borders**: border, border-title, border-subtitle
- **Colors**: color, background, tint
- **Text**: text-align, text-style, text-opacity
- **Grid**: grid-columns, grid-rows, grid-gutter, column-span, row-span
- **Scrollbars**: scrollbar-colors, scrollbar-size, scrollbar-visibility
- **And many more**

### Widgets (40+ Built-in)
Full reference for all built-in widgets:

**Input Widgets**: Button, Checkbox, Input, MaskedInput, RadioButton, RadioSet, Select, Switch, TextArea

**Display Widgets**: Label, Static, Pretty, Digits, ProgressBar, Sparkline, LoadingIndicator

**Container Widgets**: Container, Horizontal, Vertical, Grid, ScrollableContainer

**Navigation**: Header, Footer, TabbedContent, Tabs, ContentSwitcher

**Data Widgets**: DataTable, Tree, DirectoryTree, ListView, ListItem, OptionList, SelectionList

**Text/Content**: Log, RichLog, Markdown, MarkdownViewer

**Other**: Collapsible, Link, Placeholder, Rule, Toast

---

## API Documentation

Complete Python API docs available at [https://textual.textualize.io/api/](https://textual.textualize.io/api/)

Key modules:
- **textual.app**: App class and application lifecycle
- **textual.widget**: Widget base class
- **textual.screen**: Screen management
- **textual.containers**: Container widgets
- **textual.dom**: DOM manipulation
- **textual.events**: Event classes
- **textual.reactive**: Reactive programming
- **textual.pilot**: Testing framework
- **textual.color**: Color handling
- **textual.geometry**: Layout geometry
- **textual.binding**: Key bindings
- **And 30+ more modules**

---

## How-To Guides

Practical guides for common tasks:

1. **[Center things](https://textual.textualize.io/how-to/center-things/)**: How to center widgets
2. **[Design a Layout](https://textual.textualize.io/how-to/design-a-layout/)**: Layout design patterns
3. **[Package a Textual app with Hatch](https://textual.textualize.io/how-to/package-with-hatch/)**: Distribution and packaging
4. **[Render and compose](https://textual.textualize.io/how-to/render-and-compose/)**: Advanced rendering
5. **[Style Inline Apps](https://textual.textualize.io/how-to/style-inline-apps/)**: Styling techniques
6. **[Save time with Textual containers](https://textual.textualize.io/how-to/work-with-containers/)**: Container patterns

---

## Related Documentation

- **[Tutorial](https://textual.textualize.io/tutorial/)**: Step-by-step beginner guide
- **[Widget Gallery](https://textual.textualize.io/widget_gallery/)**: Visual reference for all widgets
- **[FAQ](https://textual.textualize.io/FAQ/)**: Frequently asked questions
- **[Roadmap](https://textual.textualize.io/roadmap/)**: Future development plans
- **[Blog](https://textual.textualize.io/blog/)**: DevLog, news, and release notes

---

## GitHub Repository

**Official Repository**: [https://github.com/Textualize/textual](https://github.com/Textualize/textual)

**Example Code Location**: `docs/examples/guide/`

All guide examples are runnable Python files in the repository.

---

## Sources

**Official Documentation:**
- [Textual Guide](https://textual.textualize.io/guide/) - accessed 2025-11-02
- [Textual Homepage](https://textual.textualize.io/)
- [Textual GitHub Repository](https://github.com/Textualize/textual)

**Related Pages:**
- [Tutorial](https://textual.textualize.io/tutorial/)
- [Widget Gallery](https://textual.textualize.io/widget_gallery/)
- [Reference Documentation](https://textual.textualize.io/reference/)
- [API Documentation](https://textual.textualize.io/api/)
- [How-To Guides](https://textual.textualize.io/how-to/)

---

## Next Steps

For detailed information on each guide topic, see the individual documentation pages:

- [guide/devtools/](https://textual.textualize.io/guide/devtools/)
- [guide/app/](https://textual.textualize.io/guide/app/)
- [guide/styles/](https://textual.textualize.io/guide/styles/)
- [guide/CSS/](https://textual.textualize.io/guide/CSS/)
- [guide/queries/](https://textual.textualize.io/guide/queries/)
- [guide/layout/](https://textual.textualize.io/guide/layout/)
- [guide/events/](https://textual.textualize.io/guide/events/)
- [guide/input/](https://textual.textualize.io/guide/input/)
- [guide/actions/](https://textual.textualize.io/guide/actions/)
- [guide/reactivity/](https://textual.textualize.io/guide/reactivity/)
- [guide/design/](https://textual.textualize.io/guide/design/)
- [guide/widgets/](https://textual.textualize.io/guide/widgets/)
- [guide/content/](https://textual.textualize.io/guide/content/)
- [guide/animation/](https://textual.textualize.io/guide/animation/)
- [guide/screens/](https://textual.textualize.io/guide/screens/)
- [guide/workers/](https://textual.textualize.io/guide/workers/)
- [guide/command_palette/](https://textual.textualize.io/guide/command_palette/)
- [guide/testing/](https://textual.textualize.io/guide/testing/)

**TODO for future ingestion**: Scrape each individual guide page for detailed content.
