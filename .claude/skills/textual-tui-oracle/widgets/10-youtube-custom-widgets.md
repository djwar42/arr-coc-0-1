# Creating Custom Widgets in Textual

## Video Overview

**Video Title**: "How To Create a Custom Textual Widget"
**Creator**: Mouse Vs Python (Mike Driscoll)
**Duration**: 12:54
**Views**: 820+
**Channel**: [@MouseVsPython](https://www.youtube.com/c/MouseVsPython)
**URL**: https://www.youtube.com/watch?v=iHlmTJ9RhVc
**Published**: 11 months ago (approximately November 2024)

From [How To Create a Custom Textual Widget](https://www.youtube.com/watch?v=iHlmTJ9RhVc) (accessed 2025-11-02):
> Learn how to create a custom widget in Textual for your beautiful TUI applications. Create a custom toggle button in Textual by subclassing Button, adding labels, border, and event handlers.

---

## What You'll Learn

The tutorial covers the fundamental process of building custom widgets by:
- Creating a custom widget class by subclassing existing Textual widgets
- Extending the Button widget to create specialized button types
- Adding custom labels and border styling
- Implementing event handlers for user interaction
- Building a custom toggle button as a practical example

---

## Key Concepts Covered

### Widget Subclassing

The core technique is extending built-in Textual widget classes:
- Import the base widget class you want to extend
- Create your custom class inheriting from it
- Override or extend methods to customize behavior

### Example Pattern: Custom Toggle Button

The tutorial demonstrates creating a toggle button by:
1. Subclassing the `Button` widget
2. Adding state tracking (on/off)
3. Customizing the visual label/border based on state
4. Handling button pressed events

### Methods to Override

From the [Textual Widgets Guide](https://textual.textualize.io/guide/widgets/):
- `render()` or `get_content()` - Control what the widget displays
- Message handlers (e.g., `on_button_pressed()`) - Handle widget-specific events
- `compose()` - For container widgets, define child widgets
- CSS-based styling - Customize appearance through TSS

### Event Handling

Custom widgets can respond to events by:
- Defining handler methods with the `on_` prefix (e.g., `on_button_pressed`)
- Using the `@on()` decorator for message handling
- Creating custom message types for widget-to-widget communication

From [Events and Messages](https://textual.textualize.io/guide/events/) documentation:
- Events are messages from Textual in response to input
- Custom messages allow widgets to communicate with their parent
- Message handlers are accessed as properties on widget objects

---

## Implementation Patterns

### Basic Custom Widget Structure

```python
from textual.widgets import Button

class CustomToggleButton(Button):
    """A custom toggle button widget."""

    def __init__(self, label: str = "Toggle"):
        super().__init__(label)
        self.is_on = False

    def on_button_pressed(self) -> None:
        """Handle button press event."""
        self.is_on = not self.is_on
        # Update visual representation
        self.update(self.get_label())

    def get_label(self) -> str:
        """Get label based on toggle state."""
        state = "ON" if self.is_on else "OFF"
        return f"{self.label} [{state}]"
```

### Composition Patterns

For complex widgets, compose simpler widgets:
- Use `compose()` method to return child widgets
- Container widgets (like `Container`, `Horizontal`, `Vertical`) manage layout
- Child widgets emit messages up to parent

### Styling Custom Widgets

From [Textual CSS Guide](https://textual.textualize.io/guide/styles/):
- Use TSS (Textual Style Sheets) for consistent styling
- Target custom widgets by class name
- Combine CSS with Python-based styling for dynamic appearance

---

## Related Resources

### Official Textual Documentation

Reference the complete widget system:
- [Widgets Guide](https://textual.textualize.io/guide/widgets/) - Comprehensive widget creation guide
- [Events and Messages](https://textual.textualize.io/guide/events/) - Event handling patterns
- [Widget API Reference](https://textual.textualize.io/api/widget/) - Technical API details

### Built-in Widgets as Examples

Study Textual's own widgets to understand patterns:
- [Button Widget](https://textual.textualize.io/widgets/button/) - Simple interactive widget
- [Static Widget](https://textual.textualize.io/widgets/static/) - Display-only widget base class
- [Container Widgets](https://textual.textualize.io/guide/layouts/) - Layout composition

### Related Tutorials

From the Textual-TUI-Oracle knowledge base:
- [Widget Composition](../widgets/00-built-in-widgets.md) - Using existing widgets together
- [Widget Styling with CSS](../core-concepts/03-styling-widgets-css.md) - Visual customization
- [Event Handling and Bindings](../core-concepts/05-event-handling.md) - Complete event system

---

## Best Practices

1. **Start Simple**: Subclass existing widgets before creating from scratch
2. **Leverage Composition**: Combine simple widgets rather than building monolithic custom widgets
3. **Clear Naming**: Use descriptive names that reflect widget purpose
4. **Document State**: Track and document any reactive attributes
5. **Handle Events Properly**: Use message handlers following Textual conventions
6. **Test Interactivity**: Ensure custom widgets respond correctly to user input
7. **Style Consistently**: Use TSS for appearance rather than hardcoding styles

---

## Common Use Cases

**Custom Input Widgets**
- Toggle buttons with custom states
- Specialized text inputs with validation
- Multi-select components

**Display Widgets**
- Status indicators with custom appearance
- Progress displays with specialized rendering
- Data visualizations as widgets

**Container Widgets**
- Custom layouts combining multiple widgets
- Sidebar panels with specific organization
- Modal dialogs with custom styling

---

## Video Walkthrough Notes

The tutorial progresses through:
1. **Setup** - Creating a basic custom widget structure (0:00-2:00)
2. **Subclassing** - Extending the Button class (2:00-5:30)
3. **Customization** - Adding labels and styling (5:30-9:00)
4. **Event Handling** - Responding to button presses (9:00-11:30)
5. **Testing** - Running the custom widget in an app (11:30-12:54)

The demonstration shows practical code patterns you can adapt for your own custom widgets.

---

## Sources

**Video Tutorial:**
- [How To Create a Custom Textual Widget - YouTube](https://www.youtube.com/watch?v=iHlmTJ9RhVc) - Mouse Vs Python (accessed 2025-11-02)

**Official Documentation:**
- [Textual Widgets Guide](https://textual.textualize.io/guide/widgets/)
- [Textual Events and Messages](https://textual.textualize.io/guide/events/)
- [Textual API - Widget Class](https://textual.textualize.io/api/widget/)
- [Button Widget Reference](https://textual.textualize.io/widgets/button/)

**Creator:**
- [Mouse Vs Python Blog](https://www.blog.pythonlibrary.org) - Mike Driscoll's Python tutorials and resources
- [Mouse Vs Python YouTube Channel](https://www.youtube.com/c/MouseVsPython) - 4.1K+ followers

**Related Knowledge:**
- Cross-referenced with Textual-TUI-Oracle widget documentation
