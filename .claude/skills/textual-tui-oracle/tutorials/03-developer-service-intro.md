# Developer Service Blog - Introduction to Textual

**Source**: [Introduction to Textual: Building Modern Text User Interfaces in Python](https://developer-service.blog/introduction-to-textual-building-modern-text-user-interfaces-in-python/)
**Author**: Nuno Bispo
**Date Published**: December 17, 2024
**Date Accessed**: 2025-11-02
**Note**: Article is partially paywalled - content extracted from free preview section

---

## What is a TUI?

A **Text User Interface (TUI)** is a type of user interface that allows users to interact with a program through text-based commands and visual elements displayed in a terminal or command-line interface.

Unlike a Graphical User Interface (GUI), which relies on windows, icons, and buttons, a TUI uses characters, symbols, and text-based components to present information and receive input.

### TUI Advantages

TUIs provide several advantages, including:

- **Lightweight**: They require minimal system resources compared to GUIs
- **Cross-Platform**: TUIs work on any system with a terminal, including Linux, macOS, and Windows
- **Fast and Efficient**: Navigating with keyboard shortcuts is often faster than using a mouse in GUIs
- **Accessible**: TUIs can be used on remote servers via SSH, making them ideal for server administration and headless environments

### Classic TUI Examples

Classic examples of TUIs include:
- **vim** - Text editor
- **nano** - Text editor
- **htop** - System monitoring tool
- **ranger** - File manager

Modern TUIs, like those built with **Textual**, incorporate features like animations, custom styles, and interactive elements that go beyond traditional command-line interfaces.

---

## What is Textual?

**Textual** is a Python framework for building interactive, modern TUIs using concepts that are familiar to web developers.

Instead of dealing with low-level console control sequences, Textual allows you to design responsive and interactive layouts using simple Python code.

### Web Development Concepts in Textual

Textual borrows concepts from web development, such as:

- **CSS-like Stylesheets**: Customize the appearance of your TUI components
- **Responsive Design**: Applications that adapt to terminal size changes
- **Widgets and Layouts**: Reusable components like buttons, text areas, and panels

Built on top of **Rich** (another library from the same developers), Textual inherits Rich's capabilities for beautiful text formatting, syntax highlighting, and rendering.

---

## Why Use Textual?

Here's why Textual stands out for building TUIs:

- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Interactive**: Supports buttons, dialogs, and forms
- **Asynchronous**: Built with `asyncio`, enabling non-blocking operations
- **Customizable**: Customize widgets, animations, and styles

If you've ever built a web app using frameworks like React or Vue, you'll find Textual's approach to components and layouts intuitive.

---

## Installing Textual

To get started with Textual, you'll need Python 3.7 or higher. Install it using pip:

```bash
pip install textual
```

Once installed, you're ready to build your first TUI application.

---

## Creating Your First TUI Application

Here's a simple example to create a "Hello, World!" TUI using Textual.

```python
from textual.app import App
from textual.widgets import Header, Footer, Static

class HelloWorldApp(App):
    def compose(self):
        yield Header()
        yield Static("Hello, World!", id="hello-text")
        yield Footer()

if __name__ == "__main__":
    HelloWorldApp().run()
```

### Code Explanation

1. **Header and Footer**: Pre-built widgets for a consistent app layout
2. **Static**: A simple widget to display text
3. **compose()**: This method defines the layout and widgets to display

Run the script and you'll see a clean, interactive "Hello, World!" TUI app.

**Expected Output**: A terminal application with:
- Header bar at top
- "Hello, World!" text in center
- Footer bar at bottom

---

## Key Components and Widgets (Preview)

Textual comes with several pre-built widgets, and you can also create your own.

**Note**: The full widget documentation in this article is behind a paywall. For comprehensive widget documentation, refer to:
- Official Textual documentation: https://textual.textualize.io/
- [00-official-docs.md](00-official-docs.md) - Official documentation knowledge base
- [01-real-python-tutorial.md](01-real-python-tutorial.md) - Comprehensive widget examples

---

## Key Takeaways

From the free preview content:

1. **TUIs are making a comeback** - Modern frameworks like Textual bring web development patterns to terminal UIs
2. **Textual uses familiar concepts** - CSS-like styling, responsive design, component-based architecture
3. **Built on Rich** - Inherits powerful text formatting and rendering capabilities
4. **First app is simple** - Header, Footer, Static widgets get you started quickly
5. **Asynchronous by design** - Uses asyncio for non-blocking operations

---

## Related Resources

**Official Sources:**
- Textual Documentation: https://textual.textualize.io/
- Rich Library: https://github.com/Textualize/rich

**Classic TUI Tools Referenced:**
- vim: https://www.vim.org/
- nano: https://www.nano-editor.org/
- htop: https://htop.dev/
- ranger: https://github.com/ranger/ranger

**Other Tutorials in This Knowledge Base:**
- [00-official-docs.md](00-official-docs.md) - Official Textual documentation
- [01-real-python-tutorial.md](01-real-python-tutorial.md) - Comprehensive Real Python tutorial
- [02-toward-data-science.md](02-toward-data-science.md) - Data science perspective

---

## Limitations

**Paywall Notice**: This article is partially behind a paywall. The extracted content covers:
- ✓ TUI definition and advantages
- ✓ Textual framework overview
- ✓ Installation instructions
- ✓ First "Hello World" example
- ✗ Detailed widget documentation (paywalled)
- ✗ Advanced examples (paywalled)
- ✗ Styling and customization details (paywalled)

For complete widget documentation and advanced examples, refer to the official Textual documentation and other tutorials in this knowledge base.
