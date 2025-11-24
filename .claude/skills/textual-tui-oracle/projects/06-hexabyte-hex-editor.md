# Hexabyte - Modular Hex Editor TUI

**Source**: [https://github.com/thetacom/hexabyte](https://github.com/thetacom/hexabyte)
**Accessed**: 2025-11-02
**Category**: Developer Tools
**Textual Version**: Built with Textual framework

## Overview

Hexabyte is a modern, robust, and extensible commandline hex editor built with Textual. It supports multiple view modes (hexadecimal, binary, UTF-8), split-screen editing, and file comparison (diff mode). The architecture emphasizes modularity through a plugin system that allows extending functionality via custom Python packages.

**Key Innovation**: Built-in plugins have been separated into independent packages to reduce core dependencies while maintaining extensibility.

## Features

### Core Functionality
- **Multiple View Modes**: Hexadecimal, Binary, UTF-8 text display
- **Split Screen Mode**: Single file with dual editor view
- **Diff Mode**: Side-by-side comparison of two files
- **Customizable Interface**: Adjustable column size and count per view mode
- **Command Prompt**: Interactive command interface for navigation and control

### Display Capabilities
- Hexadecimal byte view with ASCII representation
- Binary view showing individual bit patterns
- UTF-8 text view for character data
- Color-coded difference highlighting in diff mode

### Navigation
- Keyboard-driven interface
- Jump to specific byte offsets
- Chunk-based navigation (via entropy plugin)

## Installation

### User Installation

```bash
# Install via pip
pip install hexabyte

# Run the editor
hexabyte [files...] [options]
```

### Command-Line Options

```bash
hexabyte [-h] [-c CONFIG_FILEPATH] [-s] [files ...]

Positional arguments:
  files                 Specify 1 or 2 filenames

Options:
  -h, --help            Show help message
  -c CONFIG_FILEPATH    Specify config location (default: ~/.config/hexabyte/config.toml)
  -s, --split           Display single file in split screen mode
```

### Developer Setup

```bash
git clone https://github.com/thetacom/hexabyte
cd hexabyte
poetry install

# Run tests
make test
```

## Plugin Architecture

### Overview

Hexabyte's core strength is its **modular plugin system**. Built-in functionality has been extracted into separate packages to minimize dependencies while maintaining extensibility.

### Configuration

Plugins are configured in `~/.config/hexabyte/config.toml`:

```toml
plugins = [
    "hexabyte_extended_info",
    "hexabyte_entropy",
]
```

### Official Plugins

#### 1. hexabyte_extended_info

**Purpose**: Provides enhanced file metadata sidebar similar to Unix `file` command.

**Installation**:
```bash
pip install hexabyte-extended-info
```

**Features**:
- Uses `python-magic` package for robust file type detection
- Displays file format details (ELF, Mach-O, etc.)
- Architecture information (x86_64, ARM, etc.)
- File size and metadata

**Example Output** (x86_64 ELF binary):
- File Type: ELF 64-bit executable
- Architecture: x86-64
- Linking: Dynamically linked
- Additional metadata from magic numbers

**GitHub**: [https://github.com/thetacom/hexabyte_extended_info](https://github.com/thetacom/hexabyte_extended_info)

#### 2. hexabyte_entropy

**Purpose**: Visualizes file entropy through color-coded chunk display.

**Installation**:
```bash
pip install hexabyte-entropy
```

**Features**:
- Calculates Shannon entropy for file chunks
- Color-coded visualization (low entropy = blue, high entropy = red)
- Interactive chunk selection (click to jump to location)
- Scrollable entropy sidebar
- Useful for identifying compressed/encrypted sections

**Use Cases**:
- Malware analysis (detecting packed/encrypted sections)
- File format analysis (identifying data vs code sections)
- Compression detection

**GitHub**: [https://github.com/thetacom/hexabyte_entropy](https://github.com/thetacom/hexabyte_entropy)

### Creating Custom Plugins

**Plugin Structure**:

Plugins are Python packages that integrate with hexabyte's extension system. While specific API details aren't documented in the README, the architecture follows these principles:

1. **Separate Package**: Each plugin is an independent Python package
2. **Configuration Integration**: Plugins are loaded via config file
3. **Sidebar Widgets**: Plugins can add custom sidebar panels (as seen in extended_info and entropy)
4. **Event Hooks**: Plugins can respond to file events (loading, navigation, etc.)

**Example Plugin Package Structure**:
```
hexabyte_custom_plugin/
├── hexabyte_custom_plugin/
│   ├── __init__.py
│   └── plugin.py
├── pyproject.toml
├── README.md
└── tests/
```

**Dependencies**:
- Plugins depend on `hexabyte` package
- Use Textual widgets for UI components
- Follow hexabyte's plugin interface (inspect official plugins for API)

**Distribution**:
- Publish to PyPI for easy installation
- Users install via `pip install hexabyte-yourplugin`
- Add plugin name to config file

## Code Examples

### Using hexabyte_extended_info Plugin

**Installation**:
```bash
pip install hexabyte-extended-info
```

**Configuration** (`~/.config/hexabyte/config.toml`):
```toml
plugins = ["hexabyte_extended_info"]
```

**Result**: Extended file information sidebar appears automatically when opening files, showing:
- Magic number identification
- File format details
- Architecture and linking information

### Using hexabyte_entropy Plugin

**Installation**:
```bash
pip install hexabyte-entropy
```

**Configuration** (`~/.config/hexabyte/config.toml`):
```toml
plugins = ["hexabyte_entropy"]
```

**Result**: Entropy visualization sidebar shows:
- Color-coded chunks (blue=low entropy, red=high entropy)
- Interactive navigation (click chunk to jump)
- Visual identification of compressed/encrypted sections

### Combined Plugin Usage

```toml
# ~/.config/hexabyte/config.toml
plugins = [
    "hexabyte_extended_info",
    "hexabyte_entropy",
]
```

Both sidebars appear simultaneously, providing comprehensive file analysis.

## Architecture Insights

### Modular Design Philosophy

**Core Principle**: Minimize dependencies by extracting functionality into plugins.

**Benefits**:
1. **Lightweight Core**: Base hexabyte has minimal dependencies
2. **Optional Features**: Users install only needed plugins
3. **Independent Updates**: Plugins can evolve separately from core
4. **Community Extensions**: Third-party plugins follow same pattern

### View Mode System

Hexabyte implements multiple view modes for the same binary data:
- **Hexadecimal**: Traditional hex editor view
- **Binary**: Bit-level representation
- **UTF-8**: Text interpretation

Each view mode has configurable:
- Column count
- Column width
- Display formatting

### Split Screen Architecture

**Single File Mode**: Two synchronized editors viewing same file
- Independent cursor positions
- Synchronized scrolling option
- Useful for comparing different sections

**Diff Mode**: Two separate files side-by-side
- Byte-level difference highlighting
- Color-coded changes
- Synchronized navigation

## Integration Patterns

### Plugin Sidebar Pattern

Official plugins demonstrate the **sidebar extension pattern**:

1. **Register Plugin**: Plugin name in config file
2. **Create Widget**: Plugin provides Textual widget class
3. **Attach to UI**: Hexabyte mounts widget in sidebar area
4. **Event Integration**: Widget responds to file/navigation events

**Example Flow**:
```
User Opens File
    ↓
Hexabyte Loads File Data
    ↓
Plugin Receives File Event
    ↓
Plugin Analyzes Data (entropy, magic numbers, etc.)
    ↓
Plugin Updates Sidebar Widget
    ↓
User Interacts (click entropy chunk)
    ↓
Plugin Triggers Navigation Event
    ↓
Main Editor Updates View
```

### Configuration System

Uses TOML for configuration (`~/.config/hexabyte/config.toml`):

```toml
# Plugin configuration
plugins = [
    "plugin_name_1",
    "plugin_name_2",
]

# View mode settings (example structure)
[hex_view]
columns = 16
bytes_per_column = 1

[binary_view]
columns = 8
bits_per_column = 8
```

## Use Cases

### Malware Analysis
- **Entropy Plugin**: Detect packed/encrypted sections (high entropy)
- **Diff Mode**: Compare malware variants
- **Extended Info**: Identify file format and architecture

### Reverse Engineering
- **Multiple Views**: Switch between hex and binary representations
- **Split Screen**: Compare different code sections
- **Custom Plugins**: Add disassembly or string extraction

### File Format Analysis
- **Extended Info**: Identify magic numbers and file types
- **Entropy**: Visualize data vs code sections
- **Diff Mode**: Compare file format versions

### Embedded Development
- **Binary View**: Examine bit-level patterns
- **Hex View**: Verify firmware images
- **Split Screen**: Compare memory dumps

## Related Textual Patterns

### Widget Composition
Hexabyte uses complex widget composition:
- Main editor container
- Split/diff layout containers
- Plugin sidebar widgets
- Command prompt overlay

### Event-Driven Architecture
- File load events trigger plugin updates
- Navigation events synchronize views
- User input events (keyboard, mouse) drive editor

### Reactive UI Updates
- Cursor position updates
- Scroll synchronization
- Difference highlighting in diff mode

## Development Resources

**Main Repository**: [https://github.com/thetacom/hexabyte](https://github.com/thetacom/hexabyte)

**Plugin Examples**:
- [hexabyte_extended_info](https://github.com/thetacom/hexabyte_extended_info) - File metadata sidebar
- [hexabyte_entropy](https://github.com/thetacom/hexabyte_entropy) - Entropy visualization

**Testing**: Uses `make test` for test suite execution

**Package Manager**: Poetry for dependency management

**Quality**: Pre-commit hooks, Ruff linting, automated CI/CD

## Performance Considerations

### Large File Handling
- Chunk-based reading for memory efficiency
- Lazy loading of file sections
- Entropy calculations on demand

### Plugin Performance
- Plugins should avoid blocking UI thread
- Use async operations for heavy analysis
- Cache computed values (entropy, magic numbers)

## Community and Ecosystem

**Stars**: 287 (main repository)
**Last Update**: 2023-05-21
**License**: GPL-3.0
**Maturity**: Stable, production-ready

**Extension Ecosystem**:
- Two official plugins (extended_info, entropy)
- Clear path for third-party plugins
- PyPI distribution for easy installation

## Summary

Hexabyte demonstrates **modular TUI architecture** with Textual:

1. **Plugin System**: Core functionality extracted into optional plugins
2. **Sidebar Extensions**: Plugins add custom analysis widgets
3. **Multiple Views**: Same data, different representations
4. **Split/Diff Modes**: Complex layout management
5. **Configuration**: TOML-based plugin and view customization

**Key Takeaway**: Hexabyte shows how to build a **highly extensible TUI** where plugins are first-class citizens, reducing core complexity while enabling community extensions.
