# Textual-TUI-Oracle - Master Index

**Textual Framework Expert** for Python terminal UI development

---

## Overview

textual-tui-oracle is a SEEKING (dynamically expanding) knowledge base for the Textual TUI framework. Build beautiful terminal applications with Python using modern web development concepts.

**What is Textual?**
- Rapid Application Development framework for Python
- Build sophisticated UIs with simple Python API
- Run apps in terminal OR web browser
- Cross-platform (Linux, macOS, Windows)
- 40+ built-in widgets
- CSS-like styling system
- 16.7 million colors

**Oracle Mode**: SEEKING - Can dynamically expand knowledge by researching Textual documentation, widgets, and patterns.

---

## Source Code

**Complete Textual Framework Codebase** (246 Python files, 3.1MB)

The entire Textual source code from https://github.com/Textualize/textual is available in `source-code/`:

### Core Framework Files
- **app.py** - Main App class and lifecycle
- **widget.py** - Base Widget class
- **screen.py** - Screen management
- **message.py** - Message/event system
- **reactive.py** - Reactive attributes
- **css/** - CSS parser and styling engine
- **dom.py** - DOM tree management
- **binding.py** - Key bindings system
- **pilot.py** - Testing framework

### Widget Implementations
**Location**: `source-code/widgets/`

All 40+ built-in widgets with complete source:
- **_data_table.py** (2,835 lines) - DataTable widget
- **_button.py** - Button widget
- **_input.py** - Input widget
- **_text_area.py** - TextArea widget
- **_tree.py** - Tree widget
- **_select.py** - Select dropdown
- **_markdown.py** - Markdown rendering
- **_tabs.py** - TabbedContent
- *[+ 32 more widget implementations]*

### Layout Systems
**Location**: `source-code/layouts/`
- **vertical.py** - Vertical layout
- **horizontal.py** - Horizontal layout
- **grid.py** - Grid layout
- **dock.py** - Dock layout

### Rendering Pipeline
- **_compositor.py** (44KB) - Rendering compositor
- **_animator.py** - Animation system
- **renderables/** - Rich renderable integration
- **drivers/** - Terminal driver backends

### Utility Modules
- **color.py** - Color handling
- **geometry.py** - Coordinate/Region math
- **_segment_tools.py** - Terminal segment operations
- **_two_way_dict.py** - Bidirectional mapping
- **cache.py** - LRU cache implementation

**Total Coverage**: 492 Python source files (complete Textual v0.93+ codebase with full directory structure)

---

## Source Documents

All original scraped documentation from textual.textualize.io:

| File | Topic | Source URL |
|------|-------|------------|
| [00-guide-index.md](source-documents/00-guide-index.md) | Complete guide section index | https://textual.textualize.io/guide/ |
| [01-faq.md](source-documents/01-faq.md) | Frequently asked questions | https://textual.textualize.io/FAQ/ |
| [02-getting-started.md](source-documents/02-getting-started.md) | Installation and first steps | https://textual.textualize.io/getting_started/ |

---

## Core Concepts

### Getting Started
**[source-documents/02-getting-started.md](source-documents/02-getting-started.md)**
- Installation (PyPI, conda-forge)
- Requirements (Python 3.9+)
- Terminal recommendations
- First demo
- Example apps

**Quick Start:**
```python
from textual.app import App

class MyApp(App):
    pass

if __name__ == "__main__":
    MyApp().run()
```

### App Architecture
**Available via dynamic learning:**
- App class and lifecycle
- The `run()` method
- Event handlers (`on_mount`, `on_key`, etc.)
- Composing widgets
- Mounting widgets dynamically
- Exiting and return values
- Suspending apps

**Key Pattern:**
```python
from textual.app import App, ComposeResult
from textual.widgets import Welcome

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield Welcome()

    def on_button_pressed(self) -> None:
        self.exit()
```

### Widgets System
**[source-documents/00-guide-index.md](source-documents/00-guide-index.md)** → Guide section 12

**40+ Built-in Widgets:**
- **Basic**: Static, Label, Button, Link
- **Input**: Input, TextArea, MaskedInput
- **Selection**: Select, OptionList, RadioSet, Checkbox, Switch
- **Containers**: Horizontal, Vertical, Grid, Center, Middle
- **Display**: DataTable, Tree, DirectoryTree, Log, RichLog
- **Navigation**: TabbedContent, ContentSwitcher
- **Feedback**: ProgressBar, LoadingIndicator, Toast, Sparkline
- **Markdown**: Markdown, MarkdownViewer
- **Layout**: Header, Footer, Placeholder

**Widget Lifecycle:**
- compose() - Initial widget creation
- on_mount() - After widget mounted
- on_unmount() - Before widget removed

### Event System
**Available via dynamic learning:**
- Keyboard events (Key, Paste)
- Mouse events (Click, MouseMove, MouseDown, MouseUp)
- Focus events (Focus, Blur, DescendantFocus, DescendantBlur)
- Lifecycle events (Mount, Unmount, Show, Hide)
- Custom messages

**Event Handler Pattern:**
```python
def on_key(self, event: events.Key) -> None:
    if event.key == "q":
        self.exit()

def on_button_pressed(self, event: Button.Pressed) -> None:
    self.notify("Button clicked!")
```

### Styling with CSS
**[source-documents/00-guide-index.md](source-documents/00-guide-index.md)** → Guide sections 3-4

**Two Ways to Style:**
1. **External CSS** (recommended):
```python
class MyApp(App):
    CSS_PATH = "styles.tcss"
```

2. **Inline CSS**:
```python
class MyApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    Button {
        width: 100%;
    }
    """
```

**Key CSS Features:**
- Layout (vertical, horizontal, grid, dock)
- Sizing (width, height, min/max)
- Spacing (margin, padding)
- Colors (16.7M colors, not ANSI)
- Borders and outlines
- Text styling
- Alignment
- Visibility and layers

### Layout Systems
**[source-documents/00-guide-index.md](source-documents/00-guide-index.md)** → Guide section 6

**Four Layout Modes:**
1. **Vertical** - Stack widgets top to bottom (default)
2. **Horizontal** - Arrange widgets left to right
3. **Grid** - Responsive grid layout
4. **Dock** - Dock widgets to edges

**Grid Example:**
```css
Screen {
    layout: grid;
    grid-size: 2 3;  /* 2 columns, 3 rows */
    grid-gutter: 1;
}
```

### DOM Queries
**[source-documents/00-guide-index.md](source-documents/00-guide-index.md)** → Guide section 5

**Finding Widgets:**
```python
# By ID
button = self.query_one("#submit", Button)

# By class
buttons = self.query(".primary-button")

# By type
all_buttons = self.query(Button)

# CSS selectors
nested = self.query("#container Button.active")
```

---

## Troubleshooting

### Common Issues
**[source-documents/01-faq.md](source-documents/01-faq.md)**

**Question** → **Answer**
- Image support? → Not yet (on roadmap), use rich-pixels
- ImportError ComposeResult? → Upgrade: `pip install textual-dev -U`
- Select/copy text? → Click and drag + Ctrl+C (or terminal modifier key)
- Translucent background? → Won't work (Textual uses 16.7M colors, not ANSI)
- Center a widget? → Use `align: center middle` on parent
- WorkerDeclarationError? → Set `thread=True` on @work decorator
- Pass arguments to app? → Override `__init__`
- Key combinations not working? → Terminal limitation, use universal keys
- Bad rendering on macOS? → Use iTerm2/Kitty/WezTerm instead of Terminal.app
- ANSI themes? → Not supported (by design for consistency)

### macOS Terminal Issues
**Problem**: Misaligned box characters

**Solutions:**
1. **Fix Terminal.app**:
   - Settings → Profiles → Text
   - Font: Menlo Regular
   - Character spacing: 1
   - Line spacing: 0.805

2. **Use better terminal** (recommended):
   - iTerm2: https://iterm2.com/
   - Kitty: https://sw.kovidgoyal.net/kitty/
   - WezTerm: https://wezfurlong.org/wezterm/

### Key Binding Best Practices
**Universal Keys** (work everywhere):
- Letters, numbers
- F1-F10 function keys
- Space, Return
- Arrow keys, Home, End, Page Up/Down
- Ctrl, Shift

**Avoid** (terminal-dependent):
- Cmd (macOS)
- Option (macOS)
- Windows key

**Test keys**: `textual keys`

---

## Examples

### Minimal App
```python
from textual.app import App

class MinimalApp(App):
    pass

if __name__ == "__main__":
    MinimalApp().run()
```

### Hello World with Widget
```python
from textual.app import App, ComposeResult
from textual.widgets import Static

class HelloApp(App):
    def compose(self) -> ComposeResult:
        yield Static("Hello, Textual!")

if __name__ == "__main__":
    HelloApp().run()
```

### Button with Event Handler
```python
from textual.app import App, ComposeResult
from textual.widgets import Button

class ButtonApp(App):
    def compose(self) -> ComposeResult:
        yield Button("Click Me!")

    def on_button_pressed(self) -> None:
        self.notify("Button was clicked!")

if __name__ == "__main__":
    ButtonApp().run()
```

### App with CSS Styling
```python
from textual.app import App, ComposeResult
from textual.widgets import Label, Button

class StyledApp(App):
    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }

    Label {
        margin: 2;
        text-style: bold;
    }

    Button {
        margin: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Welcome to Textual!")
        yield Button("Exit", variant="error")

    def on_button_pressed(self) -> None:
        self.exit()

if __name__ == "__main__":
    StyledApp().run()
```

---

## Guide Sections (via Dynamic Learning)

When user asks about these topics, expand knowledge:

1. **Devtools** - Development tools and debugging
2. **App Basics** - Fundamental app structure
3. **Styles** - Styling widgets with Python
4. **Textual CSS** - CSS-like styling system
5. **DOM Queries** - Selecting and finding widgets
6. **Layout** - Layout systems
7. **Events and Messages** - Event system
8. **Input** - Keyboard and mouse input
9. **Actions** - Action binding system
10. **Reactivity** - Reactive programming
11. **Themes** - Design system and theming
12. **Widgets** - Creating custom widgets
13. **Content** - Rendering content
14. **Animation** - Animating styles
15. **Screens** - Screen management
16. **Workers** - Background workers
17. **Command Palette** - Command palette integration
18. **Testing** - Testing Textual apps

---

## Reference Material (via Dynamic Learning)

### CSS Types (15 types)
border, color, hatch, horizontal, integer, keyline, name, number, overflow, position, percentage, scalar, text-align, text-style, vertical

### Events (27 events)
AppBlur, AppFocus, Blur, Click, DescendantBlur, DescendantFocus, Enter, Focus, Hide, Key, Leave, Load, Mount, MouseCapture, MouseDown, MouseMove, MouseRelease, MouseScrollDown, MouseScrollUp, MouseUp, Paste, Print, Resize, ScreenResume, ScreenSuspend, Show, Unmount

### Styles (60+ properties)
align, background, border, color, display, dock, grid-*, height, width, layout, margin, padding, opacity, overflow, position, scrollbar-*, text-align, text-style, visibility, etc.

### Widgets (40+ widgets)
Button, Checkbox, Collapsible, ContentSwitcher, DataTable, Digits, DirectoryTree, Footer, Header, Input, Label, Link, ListItem, ListView, LoadingIndicator, Log, Markdown, MarkdownViewer, MaskedInput, OptionList, Placeholder, Pretty, ProgressBar, RadioButton, RadioSet, RichLog, Rule, Select, SelectionList, Sparkline, Static, Switch, TabbedContent, Tabs, TextArea, Toast, Tree, Welcome

---

## Web Resources

### Official Documentation
- **Homepage**: https://textual.textualize.io/
- **Tutorial**: https://textual.textualize.io/tutorial/
- **Guide**: https://textual.textualize.io/guide/
- **Reference**: https://textual.textualize.io/reference/
- **API**: https://textual.textualize.io/api/
- **Widget Gallery**: https://textual.textualize.io/widget_gallery/
- **How-To**: https://textual.textualize.io/how-to/
- **FAQ**: https://textual.textualize.io/FAQ/

### GitHub
- **Repository**: https://github.com/Textualize/textual/
- **Examples**: https://github.com/Textualize/textual/tree/main/examples
- **Docs**: https://github.com/Textualize/textual/tree/main/docs
- **Issues**: https://github.com/Textualize/textual/issues

### Community
- **Discord**: https://discord.gg/Enf6Z3qhVr
- **Twitter**: https://twitter.com/textualizeio

---

## How to Use This Oracle

**Asking Questions:**
- "How do I create a Button widget?"
- "Show me how to use grid layout"
- "How do I handle keyboard events?"
- "What's the syntax for CSS borders?"

**Requesting Research:**
- "Research DataTable widget and add documentation"
- "Learn about animation system from Textual docs"
- "Add guide for testing with Pilot"

**Getting Code Examples:**
- "Show me a complete app with buttons and inputs"
- "How do I create a file browser?"
- "Example of custom widget"

---

---

## Knowledge Files (Expanded 2025-11-02)

### Getting Started
| File | Topic | Source |
|------|-------|--------|
| [00-official-homepage.md](getting-started/00-official-homepage.md) | Homepage overview, key features, showcase apps | https://textual.textualize.io/ |
| [01-official-tutorial.md](getting-started/01-official-tutorial.md) | Tutorial walkthrough, key concepts | https://textual.textualize.io/tutorial/ |
| [00-installation.md](getting-started/00-installation.md) | Installation guide, prerequisites | Official docs |
| [02-python-prerequisites.md](getting-started/02-python-prerequisites.md) 🆕 | Python prerequisites checklist (3 levels), common mistakes, development tools | learn-python, learnbyexample resources |
| [03-learning-path.md](getting-started/03-learning-path.md) 🆕 | 6-phase structured learning pathway, verification checklists, hands-on projects | Python learning resources, TUI-apps |
| [03-fedora-crash-course.md](getting-started/03-fedora-crash-course.md) | Crash course tutorial with debugging, testing, packaging tips | Fedora Magazine (accessed 2025-11-02) |

### Core Concepts
| File | Topic | Source |
|------|-------|--------|
| [00-official-guide.md](core-concepts/00-official-guide.md) | Comprehensive guide structure (18 sections) | https://textual.textualize.io/guide/ |
| [01-async-comfortable-tuis.md](core-concepts/01-async-comfortable-tuis.md) | Async programming patterns, UX best practices, deployment insights | Zenn (Japanese, accessed 2025-11-02) |
| [02-layout-guide-official.md](core-concepts/02-layout-guide-official.md) 📘 | Official layout guide (vertical, horizontal, grid, dock), alignment patterns, troubleshooting | Textual Official Guide (accessed 2025-11-02) |
| [03-css-guide-official.md](core-concepts/03-css-guide-official.md) 📘 | Official CSS guide (selectors, specificity, design tokens, themes, best practices) | Textual Official Guide (accessed 2025-11-02) |
| [04-events-guide-official.md](core-concepts/04-events-guide-official.md) 📘 | Official events guide (handlers, bubbling, async, key bindings, mouse input, custom messages) | Textual Official Guide (accessed 2025-11-02) |
| [05-widgets-guide-official.md](core-concepts/05-widgets-guide-official.md) 📘 | Official widgets guide (composition, lifecycle, focus management, overflow, containers) | Textual Official Guide (accessed 2025-11-02) |
| [06-reactivity-guide-official.md](core-concepts/06-reactivity-guide-official.md) 📘 | Official reactivity guide (reactive attributes, watchers, compute, validation, state management) | Textual Official Guide (accessed 2025-11-02) |

### Widgets
| File | Topic | Source |
|------|-------|--------|
| [00-datatable-guide.md](widgets/00-datatable-guide.md) | DataTable widget comprehensive guide | Official docs + community |
| [00-widget-patterns.md](widgets/00-widget-patterns.md) 🆕 | Widget composition patterns, styling, dynamic management, common patterns (chat, forms) | Textual tutorial, RealPython, mathspp, ArjanCodes |
| [01-tree-widget-guide.md](widgets/01-tree-widget-guide.md) | Tree and DirectoryTree patterns | Official docs + Real Python |
| [01-custom-widgets.md](widgets/01-custom-widgets.md) 🆕 | Custom widget creation, reactive attributes, messages, lifecycle hooks, composite widgets | Mouse Vs Python, Textual docs |
| [02-input-validation.md](widgets/02-input-validation.md) | Input widget with validation patterns | Official docs + YouTube |
| [03-filedrop-drag-drop.md](widgets/03-filedrop-drag-drop.md) 🆕 | Drag-and-drop file handling widget - FileDrop API, paste event mechanism, icon system | GitHub @agmmnn/textual-filedrop (accessed 2025-11-02) |
| [04-plotext-plotting.md](widgets/04-plotext-plotting.md) 🆕 | Plotext plotting library integration - all plot types, streaming data, theming, weather dashboard example | GitHub @Textualize/textual-plotext (accessed 2025-11-02) |
| [05-advanced-input-collection.md](widgets/05-advanced-input-collection.md) 🆕 | Advanced input widgets (deprecated library) - TextInput, IntegerInput, password masking, syntax highlighting | GitHub @sirfuzzalot/textual-inputs (accessed 2025-11-02) |
| [10-youtube-custom-widgets.md](widgets/10-youtube-custom-widgets.md) 🎥 | Custom widget video tutorial (toggle button example, subclassing, event handling, styling) | Mouse Vs Python YouTube (accessed 2025-11-02) |

### Layout
| File | Topic | Source |
|------|-------|--------|
| [00-grid-system.md](layout/00-grid-system.md) | Grid layout comprehensive guide | Official docs |
| [01-dock-system.md](layout/01-dock-system.md) | Dock layout patterns and examples | Official docs |
| [02-responsive-design.md](layout/02-responsive-design.md) | Responsive TUI design strategies | Official docs + Real Python |

### Testing
| File | Topic | Source |
|------|-------|--------|
| [00-pilot-testing-guide.md](testing/00-pilot-testing-guide.md) | Pilot API testing framework | https://textual.textualize.io/guide/testing/ |
| [01-devtools-debugging.md](testing/01-devtools-debugging.md) | DevTools console and debugging | https://textual.textualize.io/guide/devtools/ |
| [02-testing-best-practices.md](testing/02-testing-best-practices.md) | Testing strategies and patterns | Official docs + community |

### Performance
| File | Topic | Source |
|------|-------|--------|
| [00-optimization-patterns.md](performance/00-optimization-patterns.md) | Terminal rendering, Python patterns, CSS performance, production best practices | Web research (accessed 2025-11-02) |
| [00-high-performance-algorithms.md](performance/00-high-performance-algorithms.md) | High-performance TUI algorithms | General optimization knowledge |
| [01-optimization-techniques.md](performance/01-optimization-techniques.md) | Practical optimization patterns | Textual patterns |

### Advanced
| File | Topic | Source |
|------|-------|--------|
| [00-chatgpt-integration.md](advanced/00-chatgpt-integration.md) 🆕 | ChatGPT + Textual integration (async API, custom widgets, message display, history management) | GitHub ChatGPT_TUI, chatui (accessed 2025-11-02) |
| [01-project-template.md](advanced/01-project-template.md) 🆕 | Professional Python TUI project template (uv, Nuitka, modern tooling, CI/CD) | CodeCurrents Blog (accessed 2025-11-02) |
| [00-xml-editor-custom-widgets.md](advanced/00-xml-editor-custom-widgets.md) | Custom widget development, event handling, widget composition patterns | Python Library Blog (accessed 2025-11-02) |
| [01-zenn-advanced-tips.md](advanced/01-zenn-advanced-tips.md) | Widget composition, debugging, platform-specific issues, optimization | Zenn (Japanese, accessed 2025-11-02) |
| [00-repl-integration.md](advanced/00-repl-integration.md) | REPL integration with Textual TUIs | GitHub Discussion #4326 |
| [01-packaging-apps.md](advanced/01-packaging-apps.md) | Packaging Textual apps (Nuitka, PyInstaller, Shiv) | GitHub Discussion #4512 |
| [10-youtube-screen-management.md](advanced/10-youtube-screen-management.md) 🎥 | Screen management video tutorial (push_screen, pop_screen, switch_screen, modal dialogs) | Textualize YouTube (accessed 2025-11-02) |
| [11-youtube-debugging.md](advanced/11-youtube-debugging.md) 🎥 | Debugging techniques video tutorial (two-terminal setup, DevTools, logging, notifications) | Mouse Vs Python Blog & YouTube (accessed 2025-11-02) |
| [12-responsive-chat-ui-workers.md](advanced/12-responsive-chat-ui-workers.md) 📘 | Responsive chat UI with workers (long-running processes, observer pattern, thread-safe updates) | Medium @oneryalcin (accessed 2025-11-02) |

### Extensions
| File | Topic | Source |
|------|-------|--------|
| [00-tuilwindcss-styling.md](extensions/00-tuilwindcss-styling.md) 🆕 | Tailwind CSS-inspired styling for Textual - utility classes, color palette, practical examples | GitHub @koaning/tuilwindcss (accessed 2025-11-02) |

### Community
| File | Topic | Source |
|------|-------|--------|
| [00-textual-vs-alternatives.md](community/00-textual-vs-alternatives.md) 🆕 | Textual vs 5+ TUI libraries (Rich, Urwid, Pytermgui, Asciimatics) - comparison matrix, when to choose Textual | Medium, DEV, LibHunt, Reddit (accessed 2025-11-02) |
| [01-youtube-interactive-apps.md](community/01-youtube-interactive-apps.md) 🆕 | Real Python Podcast #80 - Will McGugan interview (Rich, Textual, design insights, 21 timestamps) | RealPython Podcast, YouTube (accessed 2025-11-02) |
| [00-academic-textual-analysis.md](community/00-academic-textual-analysis.md) | Academic perspective on tool selection, error avoidance, AI integration | Journal of Business Research (accessed 2025-11-02) |
| [01-developer-ama-insights.md](community/01-developer-ama-insights.md) | Creator insights on design philosophy, performance, community | Sourcery interview (accessed 2025-11-02) |
| [02-tutorial-community-feedback.md](community/02-tutorial-community-feedback.md) | Beginner pain points, documentation feedback, tutorial effectiveness | GitHub discussions (accessed 2025-11-02) |
| [00-github-repository-overview.md](community/00-github-repository-overview.md) | Repository structure, stats, resources | https://github.com/Textualize/textual |
| [01-showcase-applications.md](community/01-showcase-applications.md) | 12 real-world Textual apps | https://www.textualize.io/projects/ |
| [02-future-roadmap.md](community/02-future-roadmap.md) | Textual's future as community project | Blog post (May 2025) |
| [10-youtube-official-channel.md](community/10-youtube-official-channel.md) 🎥 | Official Textualize YouTube channel catalog (playlists, key videos, learning paths) | @Textualize-official YouTube (accessed 2025-11-02) |
| [11-linkedin-learning-note.md](community/11-linkedin-learning-note.md) | Verification note: LinkedIn course is NLP text data, NOT Textual TUI framework | LinkedIn Learning (verified 2025-11-02) |
| [12-medium-env-manager-tui.md](community/12-medium-env-manager-tui.md) | Real-world TUI example: Environment variable manager (architecture, bugs, design patterns) | Medium @jdookeran (Oct 2025) |
| [13-erys-jupyter-tui.md](community/13-erys-jupyter-tui.md) | Jupyter + Textual integration example: Erys project (kernel management, notebook UI, keyboard workflow) | GitHub natibek/erys (accessed 2025-11-02) |
| [14-leanpub-textual-book.md](community/14-leanpub-textual-book.md) 📚 | Comprehensive Textual book: 10 TUI apps, 21 chapters, 455 pages (Calculator, CSV Viewer, Editor, etc.) | Leanpub @Michael Driscoll (accessed 2025-11-02) |

### Releases
| File | Topic | Source |
|------|-------|--------|
| [00-early-textual-introduction.md](releases/00-early-textual-introduction.md) | Framework origins, initial vision, early design decisions | Reddit, community articles (accessed 2025-11-02) |
| [00-version-3-3-0.md](releases/00-version-3-3-0.md) | v3.3.0 community release features | GitHub Release |
| [01-recent-releases.md](releases/01-recent-releases.md) | v5.1.1 - v6.5.0 release history | GitHub Releases |

### Tutorials
| File | Topic | Source |
|------|-------|--------|
| [00-realpython-comprehensive.md](tutorials/00-realpython-comprehensive.md) | Complete beginner to intermediate guide (widgets, styling, events, layouts) | RealPython (March 2025) |
| [01-devto-definitive-guide-pt1.md](tutorials/01-devto-definitive-guide-pt1.md) | Definitive guide Part 1 (historical, 2022 API - outdated) | Dev.to @wiseai (April 2022) |
| [02-fedora-crash-course.md](tutorials/02-fedora-crash-course.md) | Crash course with 2 complete apps (log scroller, race results table) | Fedora Magazine (Jan 2024) |
| [03-developer-service-intro.md](tutorials/03-developer-service-intro.md) | Introduction to Textual (TUI advantages, Hello World) | Developer Service Blog |
| [04-arjancodes-interactive-cli.md](tutorials/04-arjancodes-interactive-cli.md) | Interactive CLI tools (event handling, clean code patterns) | ArjanCodes (June 2024) |
| [05-contact-book-sqlite.md](tutorials/05-contact-book-sqlite.md) 🆕 | Complete contact book app with SQLite (CRUD, DataTable, dialogs, forms) | RealPython (accessed 2025-11-02) |
| [06-todo-app-complete.md](tutorials/06-todo-app-complete.md) 🆕 | Todo TUI application (task management, state persistence, filtering, categories) | PythonGUI.org (accessed 2025-11-02) |
| [07-leanpub-10-apps-overview.md](tutorials/07-leanpub-10-apps-overview.md) 🆕 📚 | Leanpub book overview: 10 apps (Calculator, CSV Viewer, Editor, MP3 Player, etc.) | Leanpub @Michael Driscoll (accessed 2025-11-02) |
| [10-youtube-stopwatch-series.md](tutorials/10-youtube-stopwatch-series.md) 🎥 | Official 12-episode video series building stopwatch app (setup, widgets, CSS, reactivity) | Textualize YouTube (2022-2023) |
| [11-realpython-video-tui.md](tutorials/11-realpython-video-tui.md) 🎥 | Building text-based UI video tutorial (user input, validation, CLI patterns) | RealPython Video (accessed 2025-11-02) |

### Insights
| File | Topic | Source |
|------|-------|--------|
| [00-creator-lessons-7-things.md](insights/00-creator-lessons-7-things.md) | 7 lessons from Will McGugan (creator insights on performance, immutability, unicode) | Textualize Blog (Aug 2022) |
| [01-ui-revolution-2025.md](insights/01-ui-revolution-2025.md) | How Textual is revolutionizing UI development in 2025 | Level Up (Jan 2025) |
| [02-first-impressions-pros-cons.md](insights/02-first-impressions-pros-cons.md) | Honest pros/cons, gotchas, first impressions | Learn By Example |

### Projects
| File | Topic | Source |
|------|-------|--------|
| [00-trogon-cli-to-tui.md](projects/00-trogon-cli-to-tui.md) 🆕 | Official: Auto-generate TUIs for Click CLI apps (decorator pattern, schema extraction) | GitHub @Textualize/trogon (2,728 stars) |
| [01-frogmouth-markdown-browser.md](projects/01-frogmouth-markdown-browser.md) 🆕 | Official: Markdown browser with GitHub integration (navigation stack, bookmarks) | GitHub @Textualize/frogmouth (2,966 stars) |
| [02-dooit-todo-manager.md](projects/02-dooit-todo-manager.md) 🆕 | Feature-rich TODO manager (Python config, plugin system, hierarchical tasks) | GitHub @dooit-org/dooit (2,704 stars) |
| [03-terraform-tui-infrastructure.md](projects/03-terraform-tui-infrastructure.md) 🆕 | Terraform infrastructure viewer (plan/apply/destroy operations, CLI integration) | GitHub @idoavrah/terraform-tui (1.2k stars) |
| [04-django-tui-commands.md](projects/04-django-tui-commands.md) 🆕 | Django command inspector and interactive shell - command discovery, argparse→form, auto-imports | GitHub @anze3db/django-tui (accessed 2025-11-02) |
| [05-wordle-game-tui.md](projects/05-wordle-game-tui.md) 🆕 | Wordle game implementation - grid layout, keyboard input, state management, daily puzzle | GitHub @frostming/wordle-tui (accessed 2025-11-02) |
| [06-upiano-terminal-piano.md](projects/06-upiano-terminal-piano.md) 🆕 | Interactive piano simulator - ASCII rendering, keyboard mapping, MIDI synthesis, mouse interaction | GitHub @eliasdorneles/upiano (accessed 2025-11-02) |
| [07-browsr-file-explorer.md](projects/07-browsr-file-explorer.md) 🆕 | File explorer for local/remote filesystems - S3, GitHub, SSH support, file rendering | GitHub @juftin/browsr (accessed 2025-11-02) |
| [08-spiel-presentations.md](projects/08-spiel-presentations.md) 🆕 | Presentation TUI with Rich styling - slide transitions, triggers, hot reload, REPL integration | GitHub @JoshKarpel/spiel (accessed 2025-11-02) |
| [09-elia-chatgpt-client.md](projects/09-elia-chatgpt-client.md) 🆕 | ChatGPT/Claude terminal client - streaming API, Vim selection, multi-model, SQLite persistence | GitHub @darrenburns/elia (accessed 2025-11-02) |
| [04-dolphie-mysql-monitor.md](projects/04-dolphie-mysql-monitor.md) | MySQL/MariaDB real-time monitor (record/replay, multi-host tabs, daemon mode) | GitHub @charles-001/dolphie (988 stars) |
| [05-oterm-ollama-client.md](projects/05-oterm-ollama-client.md) | Ollama AI terminal client (streaming, MCP tools, multimodal support) | GitHub @ggozad/oterm (2.2k stars) |
| [06-hexabyte-hex-editor.md](projects/06-hexabyte-hex-editor.md) | Modular hex editor (plugin system, split-screen, diff mode) | GitHub @thetacom/hexabyte |
| [07-uproot-browser-particle-physics.md](projects/07-uproot-browser-particle-physics.md) | Particle physics ROOT file browser (CLI+TUI hybrid, terminal plotting) | GitHub @scikit-hep/uproot-browser (75 stars) |
| [08-word-search-generator.md](projects/08-word-search-generator.md) | Word search puzzle generator/solver (game logic, Rich integration, PDF export) | GitHub @joshbduncan/word-search-generator |
| [09-baca-ebook-reader.md](projects/09-baca-ebook-reader.md) | Ebook reader TUI (Epub/Mobi/Azw, ANSI images, scroll animations) | GitHub @wustho/baca (470 stars) |

### Examples
| File | Topic | Source |
|------|-------|--------|
| [00-github-tutorial-examples.md](examples/00-github-tutorial-examples.md) 🆕 | Project-based tutorial: Menu-driven TUI with MenuWidget, ShowcaseScreen, LogScreen, DataTable | GitHub @KennyVaneetvelde/textual_tutorial (accessed 2025-11-02) |
| [01-awesome-projects.md](examples/01-awesome-projects.md) 🆕 | Awesome Textualize projects: 40+ third-party apps across 7 categories (tools, games, widgets) | GitHub @oleksis/awesome-textualize-projects (accessed 2025-11-02) |
| [00-awesome-tui-projects.md](examples/00-awesome-tui-projects.md) | Comprehensive TUI ecosystem survey: 7 Textual projects, 60+ Python TUIs, framework comparisons | awesome-tuis GitHub (accessed 2025-11-02) |
| [00-toad-agentic-coding.md](examples/00-toad-agentic-coding.md) | Toad universal AI coding TUI - architecture, performance, production patterns | Will McGugan blog, Maven talk (accessed 2025-11-02) |
| [00-github-examples-index.md](examples/00-github-examples-index.md) | Official examples catalog | GitHub repository |
| [00-minimal-apps.md](examples/00-minimal-apps.md) | Minimal example applications | Community examples |
| [00-text-editor-7-minutes.md](examples/00-text-editor-7-minutes.md) | Complete text editor in 7 minutes (rapid prototyping demo) | Fronkan (June 2025) |
| [01-production-tuis.md](examples/01-production-tuis.md) | Real-world production TUIs: Janssen server config tool, Doppler CLI secrets manager | Janssen docs, Doppler docs (accessed 2025-11-02) |
| [01-task-management-app-japanese.md](examples/01-task-management-app-japanese.md) | Task management TUI (Japanese guide with English summary) | Qiita @Tadataka_Takahashi |
| [04-xml-editor.md](examples/04-xml-editor.md) 🆕 | BoomslangXML editor: Tree widget, lazy loading, custom inputs, file I/O | Python Library Blog (accessed 2025-11-02) |
| [05-environment-variable-manager.md](examples/05-environment-variable-manager.md) 🆕 | Environment variable manager: .env handling, multi-screen navigation, file scanning | GitHub @FyefoxxM (accessed 2025-11-02) |
| [06-psdoom-process-manager.md](examples/06-psdoom-process-manager.md) 🆕 | psDooM process manager: psutil integration, Doom-inspired UI, async notifications | GitHub @koaning (accessed 2025-11-02) |
| [02-env-manager-case-study.md](examples/02-env-manager-case-study.md) | Environment variable manager: widget patterns, file I/O, state management, bugs & lessons | Medium article (accessed 2025-11-02) |
| [03-meshtui-lora-network.md](examples/03-meshtui-lora-network.md) | LoRa network management TUI: hardware integration, real-time data, network visualization | GitHub meshtui (accessed 2025-11-02) |
| [04-community-showcase.md](examples/04-community-showcase.md) | Community projects showcase: 130+ Textual projects across 10 domains, common patterns | awesome lists, GitHub search (accessed 2025-11-02) |

### Patterns
| File | Topic | Source |
|------|-------|--------|
| [00-async-chat-ui.md](patterns/00-async-chat-ui.md) 🆕 | Async chat UI with long-running processes: Observer pattern, Workers, thread-safe updates | Medium @oneryalcin (accessed 2025-11-02) |

### Community-International
| File | Topic | Source |
|------|-------|--------|
| [00-zenn-textual-intro-jp.md](community-international/00-zenn-textual-intro-jp.md) 🆕 🇯🇵 | Japanese community intro: SSH/Docker focus, pragmatic approach, cultural insights | Zenn.dev @secondselection (accessed 2025-11-02) |
| [01-zenn-textual-tips-jp.md](community-international/01-zenn-textual-tips-jp.md) 🆕 🇯🇵 | Japanese community tips: compose(), CSS, debugging, WSL2 clipboard, community Q&A | Zenn.dev @secondselection (accessed 2025-11-02) |
| [02-qiita-textual-examples-jp.md](community-international/02-qiita-textual-examples-jp.md) 🆕 🇯🇵 | Japanese examples: Task manager with emoji status, notification patterns, Windows CP testing | Qiita @Tadataka_Takahashi (accessed 2025-11-02) |

### Integration
| File | Topic | Source |
|------|-------|--------|
| [00-markdown-support.md](integration/00-markdown-support.md) | Markdown widget comprehensive guide: API, streaming, code highlighting, styling, patterns | Medium article, official docs (accessed 2025-11-02) |
| [01-rich-library.md](integration/01-rich-library.md) | Rich library integration: Console protocol, renderables, Textual rendering pipeline | Rich GitHub, official docs (accessed 2025-11-02) |

### Architecture
| File | Topic | Source |
|------|-------|--------|
| [00-framework-lessons.md](architecture/00-framework-lessons.md) | 7 lessons from building Textual: terminal optimization, DictViews, caching, immutability, Unicode, Fractions | Talk Python podcast, Textualize blog (accessed 2025-11-02) |

### Development
| File | Topic | Source |
|------|-------|--------|
| [00-textual-dev-tools.md](development/00-textual-dev-tools.md) | Textual-dev CLI tools: console debugging, live CSS editing, browser serving, preview tools | textual-dev GitHub, official docs (accessed 2025-11-02) |

### Deployment
| File | Topic | Source |
|------|-------|--------|
| [00-textual-web-browser.md](deployment/00-textual-web-browser.md) | Textual-web browser deployment: server-driven architecture, WebSocket protocol, platform-agnostic APIs | textual-web GitHub, Textualize blog (accessed 2025-11-02) |

---

**Oracle Status**: Active SEEKING mode
**Last Updated**: 2025-11-17 (Source Code Integration: +492 Python files - Complete Textual codebase from GitHub)
**Framework Coverage**: Textual v3.3.0 - v6.5.0+ (full source code available)
**Total Files**: 3 source documents + 97 knowledge files + 492 source code files = 592 files
