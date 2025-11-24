# baca - TUI Ebook Reader

**Source**: [GitHub - wustho/baca](https://github.com/wustho/baca)
**Accessed**: 2025-11-02
**Category**: Productivity
**Stars**: 470 | **Forks**: 12
**License**: GPL-3.0
**Framework**: Textual (Python ≥3.10)

---

## Overview

`baca` is a modern terminal-based ebook reader built with Textual, described as [epy](https://github.com/wustho/epy)'s "lovely sister" with a sleek contemporary appearance. Designed primarily for reading fiction ebooks in the terminal with smooth scroll animations and clean aesthetics.

**Sister Project**: Related to [epy](https://github.com/wustho/epy), but rebuilt with Textual for modern TUI capabilities.

---

## Key Features

### Format Support
- **Epub** - Standard EPUB format
- **Epub3** - Enhanced EPUB with advanced features
- **Mobi** - Amazon Kindle format
- **Azw** - Amazon proprietary format

### Core Capabilities
- **Reading Position Memory** - Automatically remembers last read position
- **ANSI Image Display** - Shows ebook images as ANSI art (clickable for full view)
- **Scroll Animations** - Smooth scrolling experience
- **Text Justification** - Multiple alignment options (justify/center/left/right)
- **Dual Color Schemes** - Dark and light modes with customizable colors
- **Regex Search** - Powerful search with regular expressions
- **Hyperlinks** - Clickable links within ebooks
- **Clean Modern UI** - Contemporary design optimized for terminal reading

### Reading History
- Track all previously read ebooks
- Fuzzy search through reading history by path or title+author
- Quick access to recent books

---

## Installation

### Via pip (Recommended)
```bash
pip install baca
```

### Via git (Development Version)
```bash
pip install git+https://github.com/wustho/baca
```

### Via AUR (Arch Linux)
```bash
yay -S baca-ereader-git
```

---

## Usage Examples

### Basic Reading
```bash
# Read an ebook
baca path/to/your/ebook.epub

# Resume last read ebook
baca

# View reading history
baca -r
```

### History Search (Fuzzy Matching)
```bash
# Search by filename fragments
baca doc ebook.epub

# Search by title and author
baca alice wonder lewis carroll
```

**Pattern**: Searches match against both file paths and metadata (title + author), making it easy to find books without remembering exact paths.

---

## Configuration

**Location**: `~/.config/baca/config.ini` (Linux)

### Complete Default Configuration

```ini
[General]
# Image viewer for full-size image display
PreferredImageViewer = auto

# Maximum text width (int or CSS value like 90%%)
# (escape percent with double percent %%)
MaxTextWidth = 80

# Text alignment options
# 'justify', 'center', 'left', 'right'
TextJustification = justify

# Pretty mode (WARNING: slow and memory intensive)
Pretty = no

# Scroll animation duration (seconds)
PageScrollDuration = 0.2

# Display images as ANSI art
# (affects performance & resource usage)
ShowImageAsANSII = yes

[Color Dark]
Background = #1e1e1e
Foreground = #f5f5f5
Accent = #0178d4

[Color Light]
Background = #f5f5f5
Foreground = #1e1e1e
Accent = #0178d4

[Keymaps]
ToggleLightDark = c
ScrollDown = down,j
ScrollUp = up,k
PageDown = ctrl+f,pagedown,l,space
PageUp = ctrl+b,pageup,h
Home = home,g
End = end,G
OpenToc = tab
OpenMetadata = M
OpenHelp = f1
SearchForward = slash
SearchBackward = question_mark
NextMatch = n
PreviousMatch = N
Confirm = enter
CloseOrQuit = q,escape
Screenshot = f12
```

### Configuration Highlights

**Performance Trade-offs**:
- `Pretty = no` - Default for performance (yes is slow and memory-intensive)
- `ShowImageAsANSII = yes` - Displays images inline (affects performance)

**Text Rendering**:
- `MaxTextWidth` - Supports both absolute (80) and relative (90%%) values
- `TextJustification` - Four alignment modes
- `PageScrollDuration` - Customizable animation speed

**Color Customization**:
- Separate dark/light themes
- Three color properties: Background, Foreground, Accent
- Hex color values

**Vi-style Keybindings**:
- Navigation: `j/k` (scroll), `g/G` (home/end), `h/l` (page up/down)
- Search: `/` (forward), `?` (backward), `n/N` (next/previous)
- Toggle theme: `c`

---

## Image Handling

### ANSI Image Display

When `ShowImageAsANSII=yes`, images are rendered as ANSI art inline with text. Click on any image to open full-size in system image viewer.

When `ShowImageAsANSII=no`, images appear as placeholders:
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                    IMAGE                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why ANSI Images Instead of Direct Terminal Rendering?

From the documentation:

1. **Smooth Scrolling** - Direct terminal image rendering doesn't support partial scrolling (e.g., showing 30% of an image while scrolling), resulting in broken scrolling experience

2. **Fiction Focus** - Primary use case is fiction ebooks which contain few images

3. **Maintenance** - Direct rendering requires different implementations per terminal emulator (high maintenance burden)

**Design Decision**: Prioritizes seamless reading experience over perfect image rendering.

---

## Known Limitations

### Search Behavior

**Cross-Line Search Limitation** (like vi/vim):
- Cannot find phrases spanning two lines
- Example: Searching `"for it"` won't find:
  ```
  ...she went back to the table for
  it, she found she could not...
  ```

**Text Justification Impact**:
- Justification adds variable spacing between words
- `"see it"` may become `"see  it"` or `"see   it"`
- **Solution**: Use regex search `"see +it"` or search individual words

**Best Practice**: Search feature optimized for individual words, not multi-word phrases.

### Missing Features (Compared to epy)

Planned for future implementation:
- **Bookmarks** - Save and navigate to marked positions
- **FictionBook support** - `.fb2` format
- **URL reading** - Read ebooks directly from URLs

**Current Status**: Feature parity with epy is actively being developed.

---

## Architecture Insights

### Technology Stack
- **Framework**: [Textual](https://github.com/Textualize/textual) - Modern Python TUI framework
- **Ebook Parsing**: [Kindle Unpack](https://github.com/kevinhendricks/KindleUnpack) - Mobi/Azw format support
- **Python Version**: ≥3.10 (uses modern Python features)

### Design Philosophy

**Textual Integration**:
- Built entirely on Textual framework (vs. epy which uses curses)
- Leverages Textual's reactive properties and widgets
- Uses Textual's animation system for smooth scrolling

**Performance Considerations**:
- ANSI image rendering trades visual fidelity for performance
- Pretty mode disabled by default (memory/speed trade-off)
- Scroll animations configurable (0.2s default)

**User Experience Focus**:
- Vi-style keybindings for terminal power users
- Fuzzy search through reading history (UX convenience)
- Dual theme support (dark/light)
- Clean, modern aesthetic (contemporary look vs. traditional terminal readers)

---

## Textual Patterns Used

### Configuration Management
- INI-based configuration file (`~/.config/baca/config.ini`)
- Multi-section config: General, Color Dark, Color Light, Keymaps
- Supports multiple keybindings per action (comma-separated)

### Navigation
- Dual input methods: Vi-style + standard arrow keys
- Page-based and line-based scrolling
- Smooth scroll animations with configurable duration

### Color Scheme Switching
- Runtime theme toggle (`c` key)
- Separate color configurations for dark/light modes
- Three-color palette: Background, Foreground, Accent

### Image Interaction
- Mouse-clickable ANSI images
- Fallback to system image viewer for full resolution
- Configurable display mode (ANSI vs placeholder)

### Search Implementation
- Regex search support (forward and backward)
- Match navigation (`n`/`N`)
- Vi-style search triggers (`/` and `?`)

---

## Use Cases

### Primary Use Case: Fiction Reading
- Designed for immersive fiction ebook reading
- Minimal UI distractions
- Smooth scrolling for continuous reading flow
- Reading position memory for session resumption

### Terminal-Based Library Management
- Reading history tracking
- Fuzzy search through previously read books
- Quick access without file manager navigation

### Minimalist Reading Environment
- No GUI overhead
- SSH-friendly (works over remote connections)
- Customizable color schemes for different lighting conditions
- Vi keybindings for keyboard-centric workflow

---

## Key Learnings for Textual Developers

### 1. Performance vs. Features Trade-off
- **Decision**: ANSI images instead of terminal image protocols
- **Rationale**: Prioritizes smooth scrolling over perfect image rendering
- **Lesson**: Choose features that align with primary use case (fiction vs. technical ebooks)

### 2. Configuration Design
- **Pattern**: INI format with multiple sections
- **Benefit**: Familiar format, human-readable, easy to edit
- **Feature**: Multi-key bindings (e.g., `ScrollDown = down,j`)

### 3. Search UX Considerations
- **Challenge**: Text justification breaks phrase search
- **Solution**: Document limitation, recommend regex patterns
- **Lesson**: Be transparent about technical limitations

### 4. Animation Configuration
- **Pattern**: `PageScrollDuration = 0.2` (configurable timing)
- **Benefit**: Users can adjust animation speed to preference
- **Lesson**: Make animation speeds configurable, not hardcoded

### 5. Dual Theme Support
- **Pattern**: Separate `[Color Dark]` and `[Color Light]` sections
- **Implementation**: Runtime toggle with persistent preference
- **Lesson**: Dark/light mode is essential for long reading sessions

---

## Related Projects

- **[epy](https://github.com/wustho/epy)** - Original ebook reader (curses-based, feature-rich)
- **[Frogmouth](https://github.com/Textualize/frogmouth)** - Markdown browser (official Textual project)
- Both are document readers but different approaches:
  - epy: Traditional curses, feature-complete
  - baca: Modern Textual, smooth animations, contemporary UI
  - Frogmouth: Markdown-focused, official Textual showcase

---

## Repository Stats

- **Created**: 2023
- **Last Updated**: 2023-06-08
- **Contributors**: 3
- **Language**: Python 99.1%
- **Topics**: ebook, tui, mobi, epub, reader, terminal-based

---

## Credits

From the README:
- [Textual Project](https://github.com/Textualize/textual) - TUI framework
- [Kindle Unpack](https://github.com/kevinhendricks/KindleUnpack) - Mobi/Azw parsing
- GPL-3.0 License

---

## Summary

`baca` demonstrates how to build a polished, production-ready ebook reader using Textual. Key strengths:

1. **Modern UI/UX** - Smooth animations, clean design, contemporary appearance
2. **Format Support** - Handles multiple ebook formats (Epub, Mobi, Azw)
3. **Performance Optimization** - Makes pragmatic trade-offs (ANSI images, configurable pretty mode)
4. **Configuration Flexibility** - Extensive INI-based customization
5. **Vi Compatibility** - Keyboard-centric navigation for power users

**Architecture Lesson**: Built as a complete rewrite of epy using Textual, showcasing how modern frameworks can improve UX (animations, themes) while maintaining terminal-based simplicity.

**Design Philosophy**: Prioritizes fiction reading experience with smooth scrolling, minimal distractions, and SSH-friendly operation over feature maximalism.
