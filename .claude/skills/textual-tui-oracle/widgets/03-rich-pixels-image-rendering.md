# Rich-Pixels - Image & ASCII Art Rendering

## Overview

Rich-Pixels is a Rich-compatible library for writing pixel images and ASCII art to the terminal. Created by Darren Burns (Textualize team member), it enables displaying images directly in terminal applications using half-block characters and RGB color manipulation.

**Key Capabilities:**
- Display PIL/Pillow images in terminal
- Render images using RGB background/foreground colors
- Create ASCII art with styled characters
- Integrate with Rich console output
- Use within Textual TUI applications

**GitHub**: https://github.com/darrenburns/rich-pixels (446 stars, MIT license, accessed 2025-11-02)

---

## Installation

```bash
pip install rich-pixels
```

**Dependencies:**
- `rich` - Terminal formatting library
- `Pillow` (PIL) - Python Image Library for image processing

---

## Core Rendering Technique

Rich-Pixels uses a clever terminal rendering trick:

**Half-Block Character Method:**
- Uses Unicode half-block characters (▀ ▄)
- Sets background color to upper pixel's RGB value
- Sets foreground color to lower pixel's RGB value
- Each terminal character represents 2 vertical pixels
- Achieves pixel-like rendering with terminal character grid

From [Hackaday article](https://hackaday.com/2025/09/13/send-images-to-your-terminal-with-rich-pixels/) (accessed 2025-11-02):
> "The trick is to set the background color of the half-block to the upper pixel's RGB value, and the foreground color of the half-block to the lower pixel's RGB value."

**Technical Details:**
- Terminal characters act as 2-pixel vertical units
- RGB colors provide full-color rendering (24-bit color terminals)
- Resolution depends on terminal size and font
- Image is automatically downsampled to fit terminal grid

---

## Basic Usage Patterns

### 1. Display Image from File Path

From [GitHub README](https://github.com/darrenburns/rich-pixels) (accessed 2025-11-02):

```python
from rich_pixels import Pixels
from rich.console import Console

console = Console()
pixels = Pixels.from_image_path("pokemon/bulbasaur.png")
console.print(pixels)
```

**Method:** `Pixels.from_image_path(path: str)`
- Loads image from file path
- Returns `Pixels` object ready for Rich console printing
- Supports common formats (PNG, JPG, etc.)

### 2. Display PIL Image Object

```python
from rich_pixels import Pixels
from rich.console import Console
from PIL import Image

console = Console()

with Image.open("path/to/image.png") as image:
    pixels = Pixels.from_image(image)

console.print(pixels)
```

**Method:** `Pixels.from_image(image: PIL.Image.Image)`
- Accepts pre-loaded PIL Image object
- Allows image manipulation before display
- Useful for programmatically generated images

**Use Cases:**
- Apply Pillow filters before display
- Draw shapes/text on images
- Create dynamic terminal graphics
- Process images before rendering

From [Mouse Vs Python blog](https://www.blog.pythonlibrary.org/2024/07/15/creating-images-in-your-terminal-with-python-and-rich-pixels/) (accessed 2025-11-02):
> "You can create or draw your images using Pillow. There is some coverage of this topic in my article, Drawing Shapes on Images with Python and Pillow which you could then pass to Rich Pixels to display it."

---

## ASCII Art Creation

Rich-Pixels provides a unique ASCII art workflow with character mapping and styling:

### ASCII Art with Character Mapping

From [GitHub README](https://github.com/darrenburns/rich-pixels) (accessed 2025-11-02):

```python
from rich_pixels import Pixels
from rich.console import Console
from rich.segment import Segment
from rich.style import Style

console = Console()

# Draw your shapes using any character you want
grid = """\\
     xx   xx
     ox   ox
     Ox   Ox
xx             xx
xxxxxxxxxxxxxxxxx
"""

# Map characters to different characters/styles
mapping = {
    "x": Segment(" ", Style.parse("yellow on yellow")),
    "o": Segment(" ", Style.parse("on white")),
    "O": Segment(" ", Style.parse("on blue")),
}

pixels = Pixels.from_ascii(grid, mapping)
console.print(pixels)
```

**Method:** `Pixels.from_ascii(grid: str, mapping: dict)`

**Parameters:**
- `grid`: Multi-line string with ASCII art layout
- `mapping`: Dict mapping characters to Rich Segments with styles

**Workflow:**
1. Sketch shapes using asciiflow or text editor
2. Use any characters as placeholders (x, o, O, etc.)
3. Map each character to styled Segment
4. Segments define character and Rich style (colors, formatting)

**Style Options:**
- Background colors: `"on yellow"`, `"on blue"`, `"on rgb(255,0,0)"`
- Foreground colors: `"red"`, `"green"`, `"rgb(0,255,0)"`
- Combined: `"yellow on blue"`, `"bold white on red"`
- Character replacement: Map 'x' to ' ' (space) for solid color blocks

**Benefits:**
- Rapid prototyping of terminal graphics
- Easy color experimentation
- Separation of layout (ASCII grid) and styling (mapping)
- Can use tools like [asciiflow.com](https://asciiflow.com) for design

---

## Resolution and Display Considerations

From [Mouse Vs Python blog](https://www.blog.pythonlibrary.org/2024/07/15/creating-images-in-your-terminal-with-python-and-rich-pixels/) (accessed 2025-11-02):

### Image Size Guidelines

**200×200 pixels:**
- Image appears pixelated
- May get cut off at bottom (terminal size dependent)
- Too large for typical terminal windows

**80×80 pixels:**
- Fits better in terminal
- Less pixelation due to smaller size
- Recommended starting point for terminal graphics

**Resolution Factors:**
- Monitor resolution affects display quality
- Terminal font size impacts character grid
- Larger images → more terminal lines required
- Off-screen rendering if image exceeds terminal height/width

### Best Practices

**Image Preparation:**
1. Resize images before display (use Pillow)
2. Start with square images (easier sizing)
3. Test with 64×64 or 80×80 for typical terminals
4. Consider terminal window size constraints

**Example Image Resizing:**
```python
from PIL import Image
from rich_pixels import Pixels
from rich.console import Console

# Resize image to fit terminal
with Image.open("large_image.png") as img:
    # Resize to 80x80
    img_resized = img.resize((80, 80))
    pixels = Pixels.from_image(img_resized)

console = Console()
console.print(pixels)
```

---

## Integration with Textual

From [GitHub README](https://github.com/darrenburns/rich-pixels) (accessed 2025-11-02):

Rich-Pixels can be integrated into Textual applications as a standard Rich renderable:

```python
from textual.app import App
from textual.widgets import Static
from rich_pixels import Pixels

class ImageApp(App):
    def compose(self):
        pixels = Pixels.from_image_path("logo.png")
        yield Static(pixels)
```

**Integration Pattern:**
- `Pixels` objects are Rich renderables
- Use with `Static` widget for display
- Can be used in any widget accepting Rich renderables
- Updates via reactive properties trigger re-render

**Use Cases in Textual:**
- Application logos/branding
- Data visualization overlays
- Game graphics (pixel art games)
- Dynamic image displays
- Terminal-based image viewers

---

## Advanced Patterns

### Dynamic Image Generation

```python
from PIL import Image, ImageDraw
from rich_pixels import Pixels
from rich.console import Console

# Create image programmatically
img = Image.new('RGB', (80, 80), color='black')
draw = ImageDraw.Draw(img)

# Draw shapes
draw.rectangle([10, 10, 70, 70], fill='blue', outline='white')
draw.ellipse([20, 20, 60, 60], fill='red')

# Display in terminal
pixels = Pixels.from_image(img)
console = Console()
console.print(pixels)
```

**Capabilities:**
- Programmatic graphics generation
- Real-time visual updates
- Data visualization (charts, graphs)
- Game sprites and animations
- Terminal-based GUIs with graphics

### Webcam to ASCII (Performance Considerations)

From [Stack Overflow discussion](https://stackoverflow.com/questions/78486723/optimization-problem-for-webcam-to-ascii-using-rich-library-in-python) (accessed 2025-11-02):

Real-time video processing possible but requires optimization:
- Resize frames before conversion (reduce pixel count)
- Use `Console.clear()` between frames
- Limit frame rate to terminal refresh capabilities
- Consider using `Live` display for smooth updates

**Performance Tips:**
- Smaller images = faster rendering
- Rich console caching improves performance
- Terminal emulator choice affects speed
- GPU acceleration not available (CPU-bound)

---

## API Reference

### Pixels Class Methods

**`Pixels.from_image_path(path: str) -> Pixels`**
- Load image from file path
- Returns Pixels object ready for printing
- Supports: PNG, JPG, GIF, BMP, etc.

**`Pixels.from_image(image: PIL.Image.Image) -> Pixels`**
- Create from PIL Image object
- Allows pre-processing with Pillow
- Accepts any PIL Image mode (RGB, RGBA, L, etc.)

**`Pixels.from_ascii(grid: str, mapping: dict) -> Pixels`**
- Create from ASCII art grid
- `grid`: Multi-line string with placeholder characters
- `mapping`: Dict of char -> Rich Segment with style
- Returns styled pixel grid

### Rich Console Integration

```python
from rich.console import Console
from rich_pixels import Pixels

console = Console()

# Standard printing
pixels = Pixels.from_image_path("image.png")
console.print(pixels)

# With Live display (for animations)
from rich.live import Live

with Live(pixels, refresh_per_second=10) as live:
    # Update pixels dynamically
    live.update(new_pixels)
```

---

## Common Use Cases

### 1. Terminal Application Branding

```python
# Display logo at app startup
from rich_pixels import Pixels
from rich.console import Console

console = Console()
logo = Pixels.from_image_path("company_logo.png")
console.print(logo)
console.print("[bold]Welcome to MyApp v1.0[/bold]")
```

### 2. Data Visualization

```python
# Create chart with Pillow, display in terminal
from PIL import Image, ImageDraw
from rich_pixels import Pixels

img = Image.new('RGB', (100, 50), 'white')
draw = ImageDraw.Draw(img)

# Draw bar chart
data = [10, 25, 15, 30, 20]
for i, val in enumerate(data):
    x = i * 20
    draw.rectangle([x, 50-val, x+15, 50], fill='blue')

pixels = Pixels.from_image(img)
console.print(pixels)
```

### 3. Game Graphics

```python
# Pixel art game sprite
sprite_grid = """
  xxxx
 xxxxxx
 xoxxox
 xxxxxx
  xxxx
"""

mapping = {
    "x": Segment(" ", Style.parse("green on green")),
    "o": Segment(" ", Style.parse("black on black")),
}

sprite = Pixels.from_ascii(sprite_grid, mapping)
console.print(sprite)
```

### 4. Image Preview in CLI Tools

```python
# CLI image viewer
import sys
from rich_pixels import Pixels
from rich.console import Console

if len(sys.argv) < 2:
    print("Usage: view.py <image_path>")
    sys.exit(1)

console = Console()
pixels = Pixels.from_image_path(sys.argv[1])
console.print(pixels)
console.print(f"[dim]Viewing: {sys.argv[1]}[/dim]")
```

---

## Comparison with Alternatives

### Rich-Pixels vs. ASCII Art Libraries

**Rich-Pixels Advantages:**
- Full RGB color support (not just ASCII chars)
- Integration with Rich ecosystem
- Pillow image processing pipeline
- Styled ASCII art with character mapping
- Textual TUI compatibility

**Traditional ASCII Art Libraries:**
- Character-based only (no RGB backgrounds)
- Limited color palette (ANSI 16/256 colors)
- No Pillow integration
- Text-only output

### Rich-Pixels vs. img2txt / jp2a

**Rich-Pixels:**
- RGB color per pixel
- Rich styling and formatting
- Programmatic image generation
- TUI application integration

**img2txt / jp2a:**
- Character gradients only
- No color or limited ANSI colors
- Command-line tools (not library)
- No GUI/TUI integration

---

## Limitations and Considerations

### Terminal Compatibility

**Requirements:**
- 24-bit color support (true color terminals)
- Unicode character support (half-blocks)
- Modern terminal emulators recommended

**Compatible Terminals:**
- iTerm2 (macOS)
- Windows Terminal
- Alacritty
- Kitty
- Hyper
- VS Code integrated terminal

**Limited/No Support:**
- Old terminal emulators (8/16 color only)
- Basic cmd.exe (Windows)
- Some SSH clients with limited color

### Performance Constraints

**Rendering Speed:**
- CPU-bound (no GPU acceleration)
- Large images slow to render
- Real-time video challenging (need optimization)
- Terminal refresh rate bottleneck

**Memory Usage:**
- PIL images loaded into memory
- Large images consume significant RAM
- Consider image size for embedded systems

### Display Quality

**Factors Affecting Quality:**
- Terminal font (monospace required)
- Character spacing and line height
- Color accuracy (terminal color profile)
- Resolution limitations (character grid)

**Best Results:**
- Small images (64×64 to 100×100)
- High contrast images
- Simple shapes and logos
- Pixel art style graphics

---

## Examples from Community

### Pokémon Sprites

From [GitHub README](https://github.com/darrenburns/rich-pixels):
- Original example uses Pokémon sprites
- Small pixel art images ideal for terminal display
- Demonstrates RGB color fidelity
- Shows character-based pixel rendering

### Image to Colorful ASCII Art

From [Alcyonite blog](https://alcyonite.com/blog/convert-an-image-into-colourful-ascii-art) (accessed 2025-11-02):
- Rich library enables colored ASCII conversion
- Progress bars, tables, markdown support
- Syntax highlighting integration
- Terminal-based image galleries

### Python Show Podcast Logo

From [Mouse Vs Python blog](https://www.blog.pythonlibrary.org/2024/07/15/creating-images-in-your-terminal-with-python-and-rich-pixels/):
- Real-world application logo display
- 200×200 image shows resolution challenges
- 80×80 resize demonstrates better fit
- Practical terminal branding example

---

## Tips and Best Practices

### Image Optimization

**1. Pre-resize Images:**
```python
from PIL import Image

# Resize to terminal-friendly size
with Image.open("large.png") as img:
    target_width = 80
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)
    img_resized = img.resize((target_width, target_height))
    pixels = Pixels.from_image(img_resized)
```

**2. Adjust Image Quality:**
- Reduce color depth if needed
- Apply dithering for better appearance
- Increase contrast for terminal display
- Sharpen images before display

**3. Consider Aspect Ratio:**
- Terminal characters are taller than wide
- Adjust image aspect ratio for square appearance
- Test on target terminal emulator

### ASCII Art Design

**1. Use ASCII Flow for Layout:**
- Visit [asciiflow.com](https://asciiflow.com) for design
- Export as text, use in Rich-Pixels
- Rapid prototyping of terminal UIs

**2. Character Mapping Strategy:**
- Use simple characters (x, o, #, etc.) as placeholders
- Map to Rich Segments with styles
- Iterate on colors without changing layout
- Test with different terminal backgrounds

**3. Layering and Composition:**
- Combine multiple ASCII grids
- Use Rich Panel/Table for structure
- Mix Pixels objects with other Rich renderables

### Textual Integration

**1. Static Display:**
```python
from textual.widgets import Static
from rich_pixels import Pixels

class LogoWidget(Static):
    def on_mount(self):
        pixels = Pixels.from_image_path("logo.png")
        self.update(pixels)
```

**2. Dynamic Updates:**
```python
from textual.reactive import reactive
from textual.widgets import Static

class DynamicImage(Static):
    image_path = reactive("default.png")

    def watch_image_path(self, new_path):
        pixels = Pixels.from_image_path(new_path)
        self.update(pixels)
```

**3. Animation:**
```python
from textual.widgets import Static

class AnimatedSprite(Static):
    def on_mount(self):
        self.set_interval(0.1, self.next_frame)
        self.frame = 0

    def next_frame(self):
        pixels = Pixels.from_image_path(f"frame_{self.frame}.png")
        self.update(pixels)
        self.frame = (self.frame + 1) % 10
```

---

## Sources

**GitHub Repository:**
- [darrenburns/rich-pixels](https://github.com/darrenburns/rich-pixels) - Main repository, README examples (accessed 2025-11-02)

**Blog Posts:**
- [Mouse Vs Python: Creating Images in Your Terminal with Python and Rich Pixels](https://www.blog.pythonlibrary.org/2024/07/15/creating-images-in-your-terminal-with-python-and-rich-pixels/) - Comprehensive tutorial (accessed 2025-11-02)
- [Simon Willison's Weblog: Rich Pixels](https://simonwillison.net/2025/Sep/2/rich-pixels/) - Overview and use cases
- [Hackaday: Send Images To Your Terminal With Rich Pixels](https://hackaday.com/2025/09/13/send-images-to-your-terminal-with-rich-pixels/) - Technical rendering details (accessed 2025-11-02)
- [Alcyonite: Convert an Image into Colourful ASCII Art](https://alcyonite.com/blog/convert-an-image-into-colourful-ascii-art) - Rich library integration patterns (accessed 2025-11-02)

**Community Discussions:**
- [Stack Overflow: Optimization problem for webcam to ASCII using 'rich' library](https://stackoverflow.com/questions/78486723/optimization-problem-for-webcam-to-ascii-using-rich-library-in-python) - Performance optimization discussion (accessed 2025-11-02)
- [GitHub Textualize/rich #384: Rendering images in terminal](https://github.com/Textualize/rich/discussions/384) - Related discussion on image rendering

**Additional Resources:**
- [Rich library documentation](https://rich.readthedocs.io/) - Parent library documentation
- [Textual documentation](https://textual.textualize.io/) - TUI framework integration
- [Pillow documentation](https://pillow.readthedocs.io/) - Image processing library

---

**Created:** 2025-11-02
**Knowledge Domain:** Textual TUI widgets, image rendering, ASCII art, terminal graphics
**Related Topics:** Rich library, Pillow/PIL, terminal color rendering, TUI application development
