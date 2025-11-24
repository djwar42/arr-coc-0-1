# Frogmouth - Markdown Browser TUI

**Source**: [Textualize/frogmouth](https://github.com/Textualize/frogmouth)
**Accessed**: 2025-11-02
**Project Type**: Official Textualize Project
**Stars**: 2,966 | **Forks**: 63
**Last Updated**: 2024-03-19

---

## Overview

Frogmouth is a Markdown viewer and browser for your terminal, built with [Textual](https://github.com/Textualize/textual). It provides a familiar browser-like experience for reading Markdown files directly in the terminal, supporting both local files and remote URLs.

**Key Concept**: Terminal-native Markdown browser with navigation stack, history, bookmarks, and table of contents - bringing web browser UX patterns into the terminal.

---

## Features

### Core Capabilities

- **Markdown Rendering**: Full Markdown document rendering in the terminal
- **Local and Remote Files**: Open `*.md` files from disk or via URL
- **Browser-Like Navigation**:
  - Navigation stack (forward/back)
  - History tracking
  - Bookmarks system
  - Table of contents
- **GitHub Integration**: Direct README loading from GitHub repos using `gh` command
- **Interactive Controls**: Mouse and keyboard navigation support

### GitHub Repository Quick Access

Special `gh` command syntax for loading GitHub repository READMEs:

```bash
frogmouth gh textualize/textual
```

This works both from the command line and within the app's address bar, providing instant access to any public GitHub repository's README.

---

## Installation

### pipx (Recommended for Non-Developers)

```bash
pipx install frogmouth
```

### pip (Standard Python Installation)

```bash
pip install frogmouth
```

### Homebrew (macOS/Linux)

```bash
brew tap textualize/homebrew
brew install frogmouth
```

All methods create a `frogmouth` command on your system PATH.

---

## Usage

### Basic Usage

```bash
# Launch with no file (browse from app)
frogmouth

# Open specific local file
frogmouth README.md

# Open GitHub repository README
frogmouth gh textualize/textual
```

### Navigation

- **Tab/Shift+Tab**: Navigate between controls on screen
- **Mouse**: Click to navigate and interact
- **Keyboard**: Full keyboard navigation support
- **F1**: Open help dialog

### Browser-Like Features

From [README.md](https://github.com/Textualize/frogmouth/blob/main/README.md):
- Navigation stack for forward/back browsing
- History tracking for previously visited documents
- Bookmark system for frequently accessed files
- Table of contents for document structure
- Address bar for direct URL/path entry

---

## Compatibility

**Platforms**: Linux, macOS, Windows
**Python Version**: 3.8 or above

---

## Screenshots & Demo

The project includes a video demonstration showing:
- Markdown rendering quality
- Navigation features
- Browser-like interface
- Table of contents sidebar
- Bookmark management

(Video created with [Screen Studio](https://www.screen.studio/))

Screenshots showcase:
1. Clean Markdown rendering with syntax highlighting
2. Split-view navigation with table of contents
3. Bookmark management interface
4. GitHub integration in action

---

## Architecture Insights

While the README doesn't expose internal architecture details, we can infer key patterns from its features:

### Textual Integration Patterns

**Document Viewing**:
- Markdown parser integration (likely using Textual's `Markdown` widget)
- Viewport/scrolling management for large documents
- Link handling and navigation

**Browser-Like State Management**:
- Navigation stack (history implementation)
- Bookmark persistence (configuration/storage)
- URL/path resolution system
- Loading/caching strategy

**Interactive Components**:
- Address bar (input widget)
- Navigation controls (buttons/keybindings)
- Table of contents (tree/list widget)
- Bookmark panel (list/container widget)

**GitHub Integration**:
- `gh` command parser
- GitHub API or raw content fetching
- URL rewriting for GitHub repository paths

### Likely Code Structure

Based on features and Textual patterns:

```
frogmouth/
├── app.py              # Main Textual application
├── widgets/
│   ├── address_bar.py  # URL/path input
│   ├── viewer.py       # Markdown display widget
│   ├── toc.py          # Table of contents
│   └── bookmarks.py    # Bookmark panel
├── navigation/
│   ├── stack.py        # Browser-like nav stack
│   └── history.py      # History tracking
├── loaders/
│   ├── local.py        # Local file loading
│   ├── remote.py       # HTTP/URL loading
│   └── github.py       # GitHub integration
└── config.py           # Settings/bookmarks persistence
```

---

## Key Textual Patterns Demonstrated

### 1. Document Viewer Pattern

Frogmouth demonstrates a complete document viewer implementation:
- Content loading (local + remote)
- Navigation controls
- State persistence (bookmarks, history)
- Interactive table of contents

### 2. Browser-Like Navigation

Navigation stack implementation for terminal apps:
- Forward/back functionality
- History management
- URL/path handling
- Link following

### 3. Multi-Source Content Loading

Unified interface for multiple content sources:
- Local filesystem
- HTTP/HTTPS URLs
- GitHub repository integration
- Custom protocols (`gh` command)

### 4. Configuration Persistence

Bookmark and history persistence patterns for Textual apps.

---

## Use Cases

**For Users**:
- Reading documentation without leaving terminal
- Browsing GitHub READMEs quickly
- Organizing frequently accessed Markdown files
- Terminal-native note-taking/wiki browsing

**For Developers Learning Textual**:
- Study document viewer implementation
- Learn navigation stack patterns
- Understand multi-source content loading
- See browser-like UX in terminal context
- Study GitHub API integration

---

## Community & Support

**Discord**: [Textual Discord Server](https://discord.gg/Enf6Z3qhVr)
**Issues**: 0 open issues (as of last update)
**Pull Requests**: 7 pull requests

Active community support through Textualize's Discord server.

---

## Related Projects

**Official Textualize Projects**:
- [Textual](https://github.com/Textualize/textual) - The framework Frogmouth is built on
- [Trogon](https://github.com/Textualize/trogon) - CLI to TUI converter
- [Rich](https://github.com/Textualize/rich) - Terminal formatting library

**Similar Patterns**:
- Document viewers (see `uproot-browser` for scientific data)
- Browser-like navigation (applicable to any content browsing TUI)
- Multi-source loading (databases, APIs, filesystems)

---

## Learning Value

**What Frogmouth Teaches**:
1. **Document Rendering**: High-quality Markdown display in terminal
2. **Navigation Patterns**: Browser-like UX translated to TUI
3. **State Management**: History, bookmarks, navigation stack
4. **External Integration**: GitHub API usage, URL handling
5. **User Experience**: Familiar desktop patterns in terminal context

**Code Study Focus**:
- How navigation stack is implemented
- Bookmark persistence strategy
- GitHub integration approach
- Link handling and resolution
- Configuration management

---

## Notes

- **Official Project**: Built and maintained by Textualize (Textual framework creators)
- **Production Quality**: Well-maintained, active community, polished UX
- **Open Source**: Full source code available for study
- **Real-World Usage**: Practical tool for daily terminal usage
- **Design Philosophy**: Brings desktop browser UX patterns to terminal seamlessly

**Last Feature Added** (2024-03-19): Homebrew installation instructions

---

## Quick Reference

```bash
# Install
pipx install frogmouth

# Launch
frogmouth README.md

# GitHub integration
frogmouth gh textualize/textual

# Help
# Press F1 in-app for full keyboard shortcuts
```

---

**Cross-References**:
- [INDEX.md](../INDEX.md) - Full oracle documentation index
- [00-trogon-cli-to-tui.md](00-trogon-cli-to-tui.md) - Another official Textualize tool
- [09-baca-ebook-reader.md](09-baca-ebook-reader.md) - Similar document viewing patterns
