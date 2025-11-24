# Textual CSS Guide: Official

Complete guide to styling Textual applications using Textual CSS, covering selectors, specificity rules, design tokens, and best practices for creating consistent, maintainable terminal UIs.

## Overview

Textual CSS is an extension of CSS syntax optimized for terminal applications. Like web CSS, it allows you to separate styling logic from application code, keeping projects organized and maintainable. Textual CSS supports both external stylesheet files (`.tcss`) and inline styling via Python.

From [Textual CSS Guide](https://textual.textualize.io/guide/CSS/) (accessed 2025-11-02):
- Official documentation on Textual CSS syntax and capabilities
- Coverage of stylesheets, DOM structure, selectors, and specificity

## Core Concepts

### Stylesheets

Textual applications reference CSS files to define how the app and widgets appear. This separation keeps display-related code separate from application logic.

**Stylesheet File Format** (`.tcss`):
- Plain text files containing Textual CSS rules
- Referenced in app code via `CSS_PATH` class variable or `load_css()` method
- Support for comments using `/* ... */` syntax

**Inline Styling**:
- Apply styles directly in Python using the `styles` property
- Useful for dynamic styling based on runtime conditions
- Used alongside stylesheet files for complete styling control

### The DOM

Textual organizes widgets in a Document Object Model (DOM), similar to web browsers:

**DOM Structure**:
- Root: `Screen` contains the entire interface
- Containers: Hold groups of widgets
- Widgets: Leaf nodes representing UI elements
- Every widget has an ID and zero or more CSS classes

**Widget Identification**:
- **Widget name**: Automatically derived from widget class (e.g., `Button`, `Input`, `DataTable`)
- **ID attribute**: Unique identifier for specific widget (must be unique within screen)
- **Classes attribute**: Zero or more CSS classes for styling groups of widgets

## CSS Files

### File Organization

**External CSS Files** (`.tcss`):
```css
/* Styles for all buttons */
Button {
    margin: 1;
    padding: 0 2;
}

/* Styles for specific button */
#submit-button {
    background: $primary;
    color: $text;
}

/* Styles for button class */
.action-button {
    width: 20;
    height: 3;
}
```

**Multiple Stylesheets**:
- Can reference multiple CSS files
- Later files override earlier ones
- Useful for organizing by feature or component

### CSS in Python

Direct styling through Python properties:
```python
widget.styles.width = 20
widget.styles.height = 3
widget.styles.margin = (1, 2)
widget.styles.background = "blue"
```

## Selectors

Textual CSS supports various selector types for targeting widgets:

### Type Selectors

Select all widgets of a specific type:
```css
/* All buttons */
Button { ... }

/* All input fields */
Input { ... }

/* All labels */
Label { ... }
```

### ID Selectors

Target a specific widget by ID (unique within screen):
```css
#submit-button { ... }
#status-message { ... }
#main-container { ... }
```

**ID Uniqueness**: Each ID must be unique within a single screen. Reusing IDs across multiple screens is safe.

### Class Selectors

Target widgets with a specific CSS class:
```css
.primary-action { ... }
.error-state { ... }
.hidden-element { ... }
```

**Multiple Classes**: A widget can have multiple classes, all matching selectors apply:
```css
.button-base { width: 20; }
.button-primary { background: blue; }
.button-disabled { opacity: 50%; }
```

### Pseudo-Classes

Target widgets based on state or position:

**Focus State**:
```css
Button:focus {
    border: solid yellow;
}
```

**Hover State** (mouse support):
```css
Button:hover {
    background: lightblue;
}
```

**Child Position**:
```css
Button:first-child { margin-top: 0; }
Button:last-child { margin-bottom: 0; }
```

### Descendant Combinators

Target widgets based on their position in the DOM tree:

**Direct Child**:
```css
/* Select direct children only */
Container > Button { ... }
```

**Any Descendant**:
```css
/* Select any button inside container */
Container Button { ... }
```

### Attribute Selectors

Target widgets with specific attributes:
```css
/* Widgets with data-type attribute */
Button[data-type="primary"] { ... }

/* Widgets with specific attribute value */
Input[type="password"] { ... }
```

## Specificity

When multiple selectors match a widget, Textual needs to determine which styles take precedence. Specificity rules control this.

### Specificity Levels (from lowest to highest)

1. **Element selectors**: `Button { ... }` - 1 point
2. **Class selectors**: `.action { ... }` - 10 points
3. **ID selectors**: `#submit { ... }` - 100 points
4. **Inline styles**: `widget.styles.width = 20` - 1000 points (highest)

### Calculating Specificity

For compound selectors, add points:
```css
Button.primary         /* 1 + 10 = 11 points */
Container Button       /* 1 + 1 = 2 points (descendant not counted) */
#main > Button.action  /* 100 + 1 + 10 = 111 points */
```

### Specificity Rules

- Higher specificity always wins
- If two selectors have equal specificity, the last one in the file wins
- Inline styles (via Python) have highest priority
- Avoid over-specificity; keep selectors simple when possible

**Best Practice**: Use lowest specificity needed for your styling. Overly specific selectors make CSS harder to override later.

## Design Tokens

Textual provides a system of design tokens (CSS variables) for consistent theming:

### Color Tokens

**Primary Colors**:
- `$primary` - Primary brand color
- `$secondary` - Secondary accent color
- `$warning` - Warning/alert color
- `$error` - Error state color
- `$success` - Success state color
- `$info` - Information/help color

**Text Colors**:
- `$text` - Primary text color
- `$text-muted` - Muted/secondary text
- `$text-disabled` - Disabled text color

**Background Colors**:
- `$background` - Default background
- `$panel` - Panel/container background
- `$boost` - Highlighted/boosted background

### Using Design Tokens

```css
Button {
    background: $primary;
    color: $text;
    border: solid $secondary;
}

.warning-message {
    background: $warning;
    color: $text;
}
```

**Benefits**:
- Consistent color usage across app
- Easy theme switching by updating tokens
- Reduced duplication in stylesheets
- Better accessibility through designed color relationships

## Theme System

Textual includes built-in themes that define design tokens:

### Available Themes

Official themes include colors for light and dark modes, with variants for different UI needs:
- Default theme with primary/secondary/accent colors
- Support for custom color definitions
- Theme switching at runtime

### Applying Themes

```python
from textual.app import ComposeResult
from textual.theme import Theme

class MyApp(App):
    THEME = "dark"  # Built-in theme

    # Or custom theme
    THEME = "custom-dark"  # Custom theme file
```

### Custom Theme Definition

Create theme files to customize design token values. Themes define all CSS variables used throughout the application.

## Styling Patterns

### Common Layout Patterns

**Button Groups**:
```css
.button-group Button {
    margin: 0;
    padding: 0 2;
}

.button-group Button:not(:last-child) {
    border-right: none;
}
```

**List Items**:
```css
.list-item {
    width: 100%;
    height: 3;
    border-bottom: solid $panel;
}

.list-item:hover {
    background: $boost;
}
```

**Modal Dialogs**:
```css
#modal-overlay {
    background: transparent;
    border: solid $secondary;
    width: 60%;
    height: auto;
    offset: center;
}
```

### Text Styling

**Font Styles**:
```css
Button {
    text-style: bold;  /* bold, italic, underline, strike */
}

Label.muted {
    color: $text-muted;
    text-opacity: 50%;
}
```

**Text Alignment**:
```css
Header {
    text-align: center;
}

.left-aligned {
    text-align: left;
}
```

### Responsive Patterns

**Size Constraints**:
```css
Container {
    width: 100%;
    max-width: 80;
    height: 1fr;
    min-height: 10;
}
```

**Conditional Display**:
```css
.mobile-only {
    display: block;
}

.desktop-only {
    display: none;
}
```

## Best Practices

### Selector Specificity

- Keep selectors as simple as possible
- Avoid deeply nested selectors
- Prefer classes over IDs for reusable styles
- Use ID selectors only for unique, one-time elements

### Naming Conventions

```css
/* Good: descriptive, component-based */
.button-primary { ... }
.status-bar-text { ... }
.input-error-message { ... }

/* Avoid: vague or too specific */
.red { ... }              /* Too vague */
.button-blue-rounded-10px { ... }  /* Too specific */
```

### Organization

```css
/* 1. Reset/Base styles */
* {
    margin: 0;
    padding: 0;
}

/* 2. Layout components */
.main-container { ... }
.sidebar { ... }

/* 3. Widget styles */
Button { ... }
Input { ... }

/* 4. Utility/State classes */
.hidden { display: none; }
.disabled { opacity: 50%; }
```

### Avoid Anti-Patterns

**Avoid inline styles for reusable styling**:
```python
# Bad - hard to maintain
button.styles.width = 20
button.styles.background = "blue"

# Good - use CSS classes
button.add_class("primary-button")
```

**Avoid over-nesting**:
```css
/* Bad - too deep, hard to override */
Container > Vertical > Static .text-content Label { ... }

/* Good - simpler, easier to override */
.status-label { ... }
```

**Don't repeat values**:
```css
/* Bad - repeated values */
Button { padding: 1 2; }
Input { padding: 1 2; }

/* Good - use classes */
.padded { padding: 1 2; }
Button, Input {
    @extends .padded;
}
```

## CSS Properties Quick Reference

### Dimensions

- `width`, `height` - Size specifications (pixels, fr units, percentages)
- `min-width`, `min-height` - Minimum dimensions
- `max-width`, `max-height` - Maximum dimensions
- `padding` - Internal spacing
- `margin` - External spacing
- `border` - Border definition

### Colors & Appearance

- `color` - Text color
- `background` - Background color
- `border` - Border style/color
- `opacity` - Transparency (0-100%)
- `text-style` - Text decoration (bold, italic, underline, strike)

### Layout

- `layout` - Layout type (vertical, horizontal, grid)
- `display` - Display mode (block, none)
- `offset` - Position offset (relative/absolute positioning)
- `dock` - Dock widget to edge (top, right, bottom, left)

### Text

- `text-align` - Text alignment (left, center, right, justify)
- `text-opacity` - Text transparency
- `text-wrap` - Text wrapping behavior

## Common Issues & Solutions

### Selector Not Matching

**Problem**: CSS rule doesn't apply to expected widget

**Solutions**:
- Check widget ID/class spelling matches selector
- Verify widget is in DOM and visible
- Check specificity - higher specificity selectors may override
- Use browser devtools equivalent (Textual devtools) to inspect DOM

### Specificity Conflicts

**Problem**: Unexpected styles applied or override not working

**Solutions**:
- Increase specificity using IDs or more specific selectors
- Place overriding rule later in stylesheet
- Use inline styles as last resort (highest specificity)
- Review selector specificity calculation

### Layout Issues

**Problem**: Widget sizing/positioning not as expected

**Solutions**:
- Check layout type and alignment settings
- Verify margin/padding values
- Ensure container has defined size
- Use devtools to inspect computed styles

## Performance Considerations

### CSS Optimization

- Keep stylesheets focused (avoid unused rules)
- Use efficient selectors (avoid complex descendant chains)
- Batch style updates in Python when possible
- Consider using CSS classes instead of repeated inline styles

### Dynamic Styling

When styling changes frequently at runtime:

```python
# Efficient - update class
widget.add_class("active")

# Less efficient - multiple style updates
widget.styles.background = "blue"
widget.styles.color = "white"
widget.styles.border = "solid"
```

## Integration with Python

### Loading Stylesheets

```python
from textual.app import App, ComposeResult

class MyApp(App):
    CSS_PATH = "my_app.tcss"

    def compose(self) -> ComposeResult:
        yield Button(id="submit", label="Submit")
```

### Dynamic Styles

```python
# Read computed styles
color = button.styles.background

# Modify styles
button.styles.width = 20
button.styles.add_class("active")

# Conditional styling
if error:
    input_field.add_class("error-state")
else:
    input_field.remove_class("error-state")
```

### Style Watchers

React to style changes:
```python
def watch_styles_color(self, color: Color) -> None:
    """Called when color style changes"""
    self.log(f"Color changed to {color}")
```

## Resources & References

### Official Documentation

From [Textual Documentation Site](https://textual.textualize.io/):
- [Textual CSS Guide](https://textual.textualize.io/guide/CSS/) - Complete CSS reference
- [Styles Reference](https://textual.textualize.io/guide/styles/) - All available style properties
- [CSS Types](https://textual.textualize.io/css_types/) - CSS value type specifications
- [Design Guide](https://textual.textualize.io/guide/design/) - Theme system and design tokens
- [Widgets Reference](https://textual.textualize.io/widgets/) - Widget-specific styling information

### Devtools

Use Textual's built-in devtools for debugging styles:
- Inspect DOM and applied styles
- View specificity calculations
- Test selector matching in real-time
- Debug layout and dimension issues

### Browser Support for Terminal CSS

Unlike web CSS, Textual CSS:
- Works within terminal capabilities (no advanced features)
- Supports colors based on terminal palette
- Adapts to terminal width/height
- Provides consistent appearance across platforms

## Summary

Textual CSS provides a powerful, flexible system for styling terminal applications with familiar CSS-like syntax. Key takeaways:

1. **Separation of Concerns**: Keep styling separate from logic using `.tcss` files
2. **Familiar Syntax**: CSS knowledge transfers to Textual styling
3. **Specificity Matters**: Understand how specificity works to avoid styling conflicts
4. **Design Tokens**: Use provided CSS variables for consistent, maintainable theming
5. **Performance**: Keep selectors efficient and batch style updates in Python
6. **DOM Inspection**: Use devtools to debug styling issues

By following best practices and understanding Textual CSS capabilities, you can create beautiful, maintainable terminal UIs.

## Sources

**Official Documentation:**
- [Textual CSS Guide](https://textual.textualize.io/guide/CSS/) (accessed 2025-11-02)
- [Textual Styles Reference](https://textual.textualize.io/guide/styles/)
- [Textual CSS Types](https://textual.textualize.io/css_types/)
- [Textual Design Guide](https://textual.textualize.io/guide/design/)
- [Textual Documentation Index](https://textual.textualize.io/)
