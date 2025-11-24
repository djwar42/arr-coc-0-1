# Tuilwindcss - Tailwind CSS-inspired Styling for Textual

## Overview

Tuilwindcss is a CSS utility framework for Textual TUI applications, inspired by [Tailwind CSS](https://tailwindcss.com/). It provides pre-built utility classes that can be applied directly to Textual widgets, making it easier to construct styled TUI apps without writing custom CSS.

**Key Philosophy**: Utility-first styling with familiar Tailwind naming conventions adapted for terminal constraints.

From [Tuilwindcss GitHub Repository](https://github.com/koaning/tuilwindcss):
- Provides `tuilwind.css` and `tuilwind.min.css` with hundreds of utility classes
- Mimics Tailwind CSS naming patterns where possible
- Handles terminal-specific constraints (no font sizes, adapted border syntax)
- Experimental project to test utility-first styling in TUIs

## Installation

**Download CSS file directly**:
```bash
wget https://raw.githubusercontent.com/koaning/tuilwindcss/main/tuilwind.css
```

**Download minified version**:
```bash
wget https://raw.githubusercontent.com/koaning/tuilwindcss/main/tuilwind.min.css
```

**Install as Python package (WIP)**:
```bash
python -m pip install "tuilwindcss @ git+https://github.com/koaning/tuilwindcss.git"
```

## Basic Usage

**Step 1: Download CSS file to your project**

**Step 2: Reference in your Textual app**:
```python
from textual.app import App, ComposeResult
from textual.widgets import Static

class MyApp(App):
    CSS_PATH = "tuilwind.css"  # Point to downloaded file

    def compose(self) -> ComposeResult:
        yield Static("Hello World", classes="bg-blue-500 text-white p-2")

if __name__ == "__main__":
    app = MyApp()
    app.run()
```

## Tailwind-Compatible Classes

### Background Colors

**Full Tailwind color palette** (50-900 shades):
- Slate, Gray, Zinc, Neutral, Stone
- Red, Orange, Amber, Yellow, Lime, Green, Emerald, Teal, Cyan
- Sky, Blue, Indigo, Violet, Purple, Fuchsia, Pink, Rose

**Syntax**: `bg-{color}-{shade}`

**Example**:
```python
from textual.app import App, ComposeResult
from textual.widgets import Static

class BackgroundDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static("Light Blue", classes="bg-blue-100")
        yield Static("Blue", classes="bg-blue-500")
        yield Static("Dark Blue", classes="bg-blue-900")
```

**Color reference**: All colors match [Tailwind's default palette](https://tailwindcss.com/docs/customizing-colors)
- `bg-blue-50` → #EFF6FF (lightest)
- `bg-blue-500` → #3B82F6 (base)
- `bg-blue-900` → #1E3A8A (darkest)

### Text Colors

**Syntax**: `text-{color}-{shade}`

**Example**:
```python
class TextColorDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static("Light text", classes="text-gray-300")
        yield Static("Dark text", classes="text-gray-900")
        yield Static("Blue text", classes="text-blue-600")
```

**All Tailwind color families available** (same palette as backgrounds).

### Text Styles

**Available styles**:
- `bold` - Bold text
- `italic` - Italic text
- `underline` - Underlined text
- `strike` - Strikethrough text
- `reverse` - Reversed foreground/background

**Example**:
```python
class TextStyleDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static("Normal text")
        yield Static("Bold text", classes="bold")
        yield Static("Italic text", classes="italic")
        yield Static("Underlined text", classes="underline")
        yield Static("Strikethrough text", classes="strike")
        yield Static("Reversed text", classes="reverse")
```

### Text Alignment

**Available alignments**:
- `text-left` - Left alignment
- `text-center` - Center alignment
- `text-right` - Right alignment
- `text-justify` - Justified alignment
- `text-start` - Start alignment
- `text-end` - End alignment

**Example**:
```python
text = "This is a long text example to demonstrate alignment options."

class TextAlignmentDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static(text, classes="text-left bg-gray-500")
        yield Static(text, classes="text-center bg-gray-600")
        yield Static(text, classes="text-right bg-gray-700")
```

### Padding

**Simple padding (all sides)**:
- `p-1`, `p-2`, `p-3`, `p-4` - Uniform padding

**Directional padding**:
- `pt-{n}` - Padding top
- `pb-{n}` - Padding bottom
- `pl-{n}` - Padding left
- `pr-{n}` - Padding right
- `px-{n}` - Padding horizontal (left + right)
- `py-{n}` - Padding vertical (top + bottom)

**Example**:
```python
class PaddingDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static("Uniform padding", classes="p-2 bg-blue-400")
        yield Static("Top padding", classes="pt-2 bg-gray-200")
        yield Static("Horizontal padding", classes="px-3 bg-gray-300")
        yield Static("Vertical padding", classes="py-2 bg-gray-400")
```

### Margin

**Simple margin (all sides)**:
- `m-1`, `m-2`, `m-3`, `m-4` - Uniform margin

**Directional margin**:
- `mt-{n}` - Margin top
- `mb-{n}` - Margin bottom
- `ml-{n}` - Margin left
- `mr-{n}` - Margin right
- `mx-{n}` - Margin horizontal (left + right)
- `my-{n}` - Margin vertical (top + bottom)

**Example**:
```python
from textual.containers import Vertical

class MarginDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Small margin", classes="m-1 p-1 bg-gray-200"),
            Static("Medium margin", classes="m-2 p-1 bg-gray-300"),
            Static("Large margin", classes="m-3 p-2 bg-gray-400"),
            classes="bg-gray-500"
        )
```

## Tailwind-Inconsistent Classes

### Borders

**Key difference**: Textual requires border type and color together (unlike Tailwind which separates width and color).

**Syntax**: `border-{type}-{color}-{shade}`

**Supported border types**:
- `ascii` - ASCII characters (+-|)
- `blank` - No border (blank space)
- `dashed` - Dashed lines
- `double` - Double lines (═ ║)
- `heavy` - Heavy/thick lines
- `hidden` - Hidden border
- `hkey` - Horizontal key style
- `inner` - Inner border
- `none` - No border
- `outer` - Outer border
- `round` - Rounded corners (╭ ╮ ╰ ╯)
- `solid` - Solid lines (┌ ┐ └ ┘)
- `tall` - Tall border style
- `vkey` - Vertical key style
- `wide` - Wide border style

**Example**:
```python
class BorderDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static("Solid border",
                     classes="p-2 bg-gray-100 border-solid-gray-900")
        yield Static("Rounded border",
                     classes="p-2 bg-gray-200 border-round-gray-900")
        yield Static("Double border",
                     classes="p-2 bg-gray-400 border-double-gray-900")
```

**Directional borders**:
- `border-top-{type}-{color}-{shade}`
- `border-bottom-{type}-{color}-{shade}`
- `border-left-{type}-{color}-{shade}`
- `border-right-{type}-{color}-{shade}`

**Example**:
```python
yield Static("Top border only",
             classes="p-2 border-top-solid-blue-600")
```

## Textual-Specific Classes

### Dock

**Unique to Textual**: Docking controls widget positioning (not in Tailwind CSS).

**Dock left/right**:
- `dock-left` - Dock widget to left edge
- `dock-right` - Dock widget to right edge

**Dock top/bottom**:
- `dock-top` - Dock widget to top edge
- `dock-bottom` - Dock widget to bottom edge

**Example**:
```python
from textual.widgets import Header, Footer

class DockDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Header(classes="dock-top")
        yield Static("Main content")
        yield Footer(classes="dock-bottom")
```

## Combining Classes

**Multiple utilities can be chained**:
```python
yield Static(
    "Styled widget",
    classes="p-2 m-1 bg-blue-500 text-white bold text-center border-round-blue-700"
)
```

**Class combinations**:
- Spacing: `p-2 m-1`
- Colors: `bg-blue-500 text-white`
- Text: `bold text-center`
- Border: `border-round-blue-700`

## Vanilla CSS vs Tuilwindcss

### Vanilla Textual CSS

**Before** (manual CSS file):
```css
/* styles.css */
.my-widget {
    background: #3B82F6;
    color: white;
    padding: 2;
    text-align: center;
    border: solid #1E40AF;
}
```

```python
yield Static("Hello", classes="my-widget")
```

### Tuilwindcss Approach

**After** (utility classes):
```python
yield Static(
    "Hello",
    classes="bg-blue-500 text-white p-2 text-center border-solid-blue-800"
)
```

**Advantages**:
- No custom CSS writing
- Rapid prototyping
- Consistent color palette
- Self-documenting styles (class names describe appearance)
- Easy to modify (change classes instead of editing CSS)

**Trade-offs**:
- Longer class strings
- Less semantic (appearance-based vs meaning-based names)
- Larger CSS file size (prebuilt utilities)

## Practical Examples

### Card-like Widget
```python
from textual.containers import Vertical

class CardDemo(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Card Title", classes="bold text-lg p-1 bg-blue-600 text-white"),
            Static("Card content goes here.", classes="p-2 bg-gray-100 text-gray-800"),
            classes="m-2 border-round-gray-400"
        )
```

### Status Panel
```python
class StatusPanel(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Static("✓ Success", classes="bg-green-500 text-white p-1 m-1 bold")
        yield Static("⚠ Warning", classes="bg-yellow-500 text-black p-1 m-1 bold")
        yield Static("✗ Error", classes="bg-red-500 text-white p-1 m-1 bold")
```

### Dashboard Layout
```python
from textual.containers import Horizontal, Vertical

class Dashboard(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        yield Header(classes="bg-blue-900 text-white")
        yield Horizontal(
            Vertical(
                Static("Sidebar", classes="bg-gray-200 p-2"),
                classes="dock-left"
            ),
            Vertical(
                Static("Main content", classes="bg-white p-3"),
            )
        )
        yield Footer(classes="bg-blue-900 text-white")
```

## Color Palette Reference

**Grayscale families** (neutral backgrounds):
- Slate, Gray, Zinc, Neutral, Stone

**Warm colors** (alerts, highlights):
- Red, Orange, Amber, Yellow

**Cool colors** (success, info):
- Lime, Green, Emerald, Teal, Cyan, Sky, Blue

**Accent colors** (branding, emphasis):
- Indigo, Violet, Purple, Fuchsia, Pink, Rose

**Shade conventions**:
- 50-100: Very light (backgrounds)
- 200-400: Light (borders, secondary text)
- 500-600: Base (primary colors)
- 700-900: Dark (text, emphasis)

## Design Philosophy

From [Tuilwindcss documentation](https://koaning.github.io/tuilwindcss/):

**Utility-first approach**:
- Classes describe what they do, not semantic meaning
- Compose complex designs from simple utilities
- Consistency through constrained design system

**Terminal adaptations**:
- No font size utilities (terminal constraint)
- Border syntax adapted for Textual's border system
- Dock utilities added for TUI-specific layouts

**Tailwind compatibility**:
- Same color palette and names
- Similar padding/margin conventions
- Familiar developer experience for web developers

## Integration Patterns

### Loading CSS Globally
```python
class MyApp(App):
    CSS_PATH = "tuilwind.css"  # All screens inherit utilities

    def compose(self) -> ComposeResult:
        yield Static("Uses tuilwind classes", classes="bg-blue-500 p-2")
```

### Mixing with Custom CSS
```python
class HybridApp(App):
    CSS_PATH = ["tuilwind.css", "custom.css"]  # Combine both

    def compose(self) -> ComposeResult:
        # Use tuilwind utilities
        yield Static("Utility styled", classes="bg-blue-500 p-2")
        # Use custom classes
        yield Static("Custom styled", classes="my-custom-class")
```

### Dynamic Class Application
```python
class DynamicApp(App):
    CSS_PATH = "tuilwind.css"

    def compose(self) -> ComposeResult:
        self.status = Static("Status", classes="bg-gray-500 p-2")
        yield self.status

    def update_status(self, success: bool):
        if success:
            self.status.classes = "bg-green-500 text-white p-2 bold"
        else:
            self.status.classes = "bg-red-500 text-white p-2 bold"
```

## Best Practices

**Keep class lists organized**:
```python
# Group related utilities
classes = " ".join([
    # Layout
    "p-2", "m-1",
    # Colors
    "bg-blue-500", "text-white",
    # Typography
    "bold", "text-center",
    # Border
    "border-round-blue-700"
])
yield Static("Content", classes=classes)
```

**Use consistent color schemes**:
```python
# Define theme constants
PRIMARY_BG = "bg-blue-600"
PRIMARY_TEXT = "text-white"
SECONDARY_BG = "bg-gray-200"
SECONDARY_TEXT = "text-gray-800"

yield Static("Primary", classes=f"{PRIMARY_BG} {PRIMARY_TEXT} p-2")
yield Static("Secondary", classes=f"{SECONDARY_BG} {SECONDARY_TEXT} p-2")
```

**Responsive spacing**:
```python
# Use margin/padding hierarchy
yield Static("Title", classes="p-3 m-2 bold")  # Larger spacing
yield Static("Content", classes="p-2 m-1")     # Medium spacing
yield Static("Footer", classes="p-1")          # Minimal spacing
```

**Semantic color usage**:
```python
# Follow color conventions
yield Static("Success", classes="bg-green-500 text-white")  # Green = success
yield Static("Warning", classes="bg-yellow-500 text-black") # Yellow = warning
yield Static("Error", classes="bg-red-500 text-white")      # Red = error
yield Static("Info", classes="bg-blue-500 text-white")      # Blue = info
```

## Limitations

**No responsive design**: Terminal width is fixed, no breakpoints like `md:`, `lg:` in Tailwind.

**No font sizing**: Terminals use monospace fonts, so `text-sm`, `text-lg` utilities don't exist.

**Border complexity**: Borders require combined type-color classes due to Textual's CSS model:
- Tailwind: `border border-blue-500`
- Tuilwindcss: `border-solid-blue-500`

**No animations**: Textual doesn't support CSS animations/transitions.

**Static color palette**: Unlike Tailwind's JIT mode, colors are pre-generated (no arbitrary values like `bg-[#123456]`).

## When to Use Tuilwindcss

**Good fits**:
- Rapid prototyping of TUI apps
- Teams familiar with Tailwind CSS
- Apps needing consistent color schemes
- Projects with frequent style iterations

**Better alternatives**:
- Custom CSS for highly semantic designs
- Inline styles for one-off widgets
- Pure Python styling for dynamic themes
- Textual's design system for framework consistency

## Sources

**GitHub Repository**:
- [koaning/tuilwindcss](https://github.com/koaning/tuilwindcss) - Main repository (accessed 2025-11-02)
- README.md - Installation and overview
- setup.py - Package configuration

**Documentation**:
- [Tuilwindcss Docs](https://koaning.github.io/tuilwindcss/) - Visual gallery and examples (accessed 2025-11-02)
- Complete utility class demonstrations
- Code examples for all features

**Related**:
- [Tailwind CSS](https://tailwindcss.com/) - Original inspiration
- [Textual CSS Guide](https://textual.textualize.io/guide/CSS/) - Textual's CSS system
