# dooit - Feature-Rich TODO Manager TUI

**Source**: https://github.com/dooit-org/dooit
**Documentation**: https://dooit-org.github.io/dooit/
**Accessed**: 2025-11-02
**Type**: Community Project (Built with Textual)
**Stars**: 2,704 | **Forks**: 113
**Last Update**: 2024-11-25

---

## Overview

dooit is a feature-rich TODO manager built with Textual that provides an interactive, beautiful terminal UI for managing tasks and projects. It emphasizes customization through Python configuration files and extensibility via plugins.

**Tagline**: *"A todo manager that you didn't ask for, but needed!"*

**Key Characteristics**:
- Fully customizable via Python config files
- Vim-like keybindings
- Topic-wise separated TODO lists with branching
- Extensible through plugin system (dooit-extras)
- Beautiful UI with custom theming and CSS support
- Built on Textual framework

---

## Installation

### PyPI (Recommended)
```bash
pip install dooit dooit-extras
```

### Arch Linux (AUR)
```bash
yay -S dooit dooit-extras
```

### Conda/Mamba
```bash
conda install dooit dooit-extras
# or
mamba install dooit dooit-extras
```

### Pixi (Global Access)
```bash
pixi global install dooit
```

### NixOS (Flakes)

**Home Manager Module**:
```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    dooit.url = "github:dooit-org/dooit";
    dooit-extras.url = "github:dooit-org/dooit-extras";
  };

  outputs = inputs @ { nixpkgs, ... }: {
    homeConfigurations."${username}" = nixpkgs.lib.nixosSystem {
      modules = [ ./home-manager/dooit.nix ];
    };
  };
}
```

```nix
# home-manager/dooit.nix
{
  imports = [ inputs.dooit.homeManagerModules.default ];
  nixpkgs.overlays = [inputs.dooit-extras.overlay];

  programs.dooit = {
    enable = true;
    extraPackages = [pkgs.dooit-extras];
  };
}
```

---

## Features

### Core Functionality
- **Interactive & Beautiful UI**: Modern terminal interface with smooth navigation
- **Hierarchical Organization**: Topic-wise separated TODO lists with branching support
- **Task Management**:
  - Add child nodes and siblings
  - Priority/urgency levels
  - Due dates and recurrence
  - Effort estimation
  - Task descriptions
  - Completion tracking

### Customization
- **Python Configuration**: Full control via Python config files
- **Theme Support**: Custom color schemes and CSS theming
- **Layout Configuration**: Customize bar, dashboard, and formatter
- **Keybinding Customization**: Remap all keyboard shortcuts

### Built-in Features
- **Vim-like Navigation**: Familiar keybindings for Vim users
- **Search**: Quick search within TODO lists
- **Sorting**: Flexible sorting options for tasks
- **Clipboard Operations**: Copy/paste tasks and descriptions
- **Help System**: Built-in help screen (press `?`)

---

## Architecture & Configuration

### Configuration File Structure

dooit uses a Python-based configuration file that provides access to the `DooitAPI` object:

```python
from dooit.api import DooitAPI
from dooit.api.events import Startup, subscribe

@subscribe(Startup)
def setup(api: DooitAPI, _):
    # Configuration goes here
    api.vars.some_setting = value
    api.keys.some_action = "key"
```

### DooitAPI Sub-modules

The API exposes several sub-modules for different aspects of configuration:

- **`api.keys`**: Keybinding configuration
- **`api.layout`**: Layout and display settings
- **`api.formatter`**: Format how tasks are displayed
- **`api.bar`**: Status bar configuration
- **`api.vars`**: General variables and settings
- **`api.dashboard`**: Dashboard configuration

### Event System

dooit uses an event subscription pattern:

```python
from dooit.api.events import Startup, subscribe

@subscribe(Startup)
def on_startup(api: DooitAPI, event):
    api.notify("dooit started!", level="info")
```

**Notification Levels**: `info`, `warning`, `error`

---

## DooitAPI Methods

### Core Actions

**Task Management**:
- `add_child_node()` - Add child to highlighted item
- `add_sibling()` - Add sibling to highlighted item
- `remove_node()` - Remove highlighted item
- `toggle_complete()` - Toggle task completion
- `copy_model()` - Copy focused task to clipboard
- `paste_model_above()` - Paste task above focused item
- `paste_model_below()` - Paste task below focused item

**Navigation**:
- `move_up()` / `move_down()` - Cursor movement
- `go_to_top()` / `go_to_bottom()` - Jump to boundaries
- `shift_up()` / `shift_down()` - Reorder tasks
- `switch_focus()` - Switch between workspace and TODO list
- `toggle_expand()` - Expand/collapse item
- `toggle_expand_parent()` - Expand/collapse parent

**Editing**:
- `edit_description()` - Edit task description
- `edit_due()` - Edit due date
- `edit_effort()` - Edit effort estimation
- `edit_recurrence()` - Edit recurrence pattern
- `increase_urgency()` / `decrease_urgency()` - Adjust priority

**Utility**:
- `start_search()` - Search within list
- `start_sort()` - Sort siblings
- `show_help()` - Display help screen
- `copy_description_to_clipboard()` - Copy description text
- `notify(message, level="info")` - Show notification
- `quit()` - Exit application

---

## Backend API (Data Models)

### Workspace
Represents a collection of TODOs (workspace/project):
```python
from dooit.api.workspace import Workspace

workspace = Workspace(description="My Project")
```

### Todo
Represents individual tasks:
```python
from dooit.api.todo import Todo

todo = Todo(
    description="Task description",
    urgency=1,
    due="2025-12-31",
    effort="2h",
    recurrence="weekly"
)
```

---

## Dooit Extras (Plugin System)

**Repository**: https://github.com/dooit-org/dooit-extras

The plugin system extends dooit with additional functionality:

- **Custom widgets**
- **Additional themes**
- **Enhanced formatters**
- **Extra keybinding patterns**

**Installation**: `pip install dooit-extras`

**Note**: For customizability feature requests, open issues at [dooit-extras](https://github.com/dooit-org/dooit-extras/issues) rather than the main repo.

---

## Usage

### Launch Application
```bash
dooit
```

### Built-in Help
Press `?` key while running dooit to see all keybindings and commands.

### Default Keybindings
Most functionality is mapped to sane defaults - check the help screen for the complete list.

---

## Theme Examples

dooit supports extensive theming. The README showcases three example configurations:

### 1. Nord Theme (Icy Configuration)
Cold, minimal aesthetic based on Nord color palette.

### 2. Catppuccin Theme (Colorful Configuration)
Vibrant, colorful theme based on Catppuccin palette.

### 3. Everforest Theme (Calm Configuration)
Calm, nature-inspired theme based on Everforest palette.

**All themes heavily utilize [dooit-extras](https://github.com/dooit-org/dooit-extras) for enhanced customization.**

---

## Key Textual Integration Patterns

### 1. Python-Based Configuration
Unlike many TUI apps using YAML/TOML, dooit uses Python config files:
- Full programming language capabilities
- Dynamic configuration based on runtime conditions
- Direct API access for complex customization

### 2. Event-Driven Architecture
Uses subscription-based event system:
```python
@subscribe(EventType)
def handler(api: DooitAPI, event):
    # Handle event
```

### 3. API-First Design
All functionality exposed through clean API:
- Methods for all actions
- Sub-modules for different configuration domains
- Programmatic access to all features

### 4. Extensibility Through Plugins
Plugin system (dooit-extras) demonstrates:
- How to extend Textual apps
- Separation of core vs extensions
- Community-driven feature additions

---

## Development Insights

### Project Structure
- **Community-driven**: Active Discord community (989186205025464390)
- **Organization**: Moved from single-user repo to dooit-org organization
- **Documentation**: Comprehensive wiki at https://dooit-org.github.io/dooit/
- **Modularity**: Separate core and extras repositories

### Best Practices Demonstrated
1. **Clear API separation**: Backend (Workspace/Todo) vs UI (DooitAPI)
2. **Configuration flexibility**: Python over static config files
3. **Extensibility first**: Plugin system for community contributions
4. **Documentation-driven**: Extensive wiki with examples
5. **Theme support**: CSS and custom theming built-in

---

## Related Projects

**By the same author** (kraanzu):
- [smassh](https://github.com/kraanzu/smassh) - Typing test app for terminal (also Textual-based)

---

## Links & Resources

**Official**:
- GitHub: https://github.com/dooit-org/dooit
- Wiki/Docs: https://dooit-org.github.io/dooit/
- Discord: https://discord.gg/WA2ER9MBWa
- PyPI: https://pypi.org/project/dooit/

**Extension**:
- Dooit Extras: https://github.com/dooit-org/dooit-extras

**Social**:
- Twitter: https://twitter.com/kraanzu

---

## Key Takeaways for Textual Developers

1. **Python Configuration Power**: Shows the advantage of Python config files over static formats for TUI apps
2. **API Design**: Clean separation between data models (Backend API) and UI control (DooitAPI)
3. **Event System**: Demonstrates effective event-driven architecture in Textual
4. **Plugin Architecture**: Example of how to make Textual apps extensible
5. **Theme Support**: Integration with Textual's CSS system for custom themes
6. **Vim-like UX**: Shows how to implement familiar keybinding patterns
7. **Community Growth**: Successfully transitioned from solo project to organization

---

**Last Updated**: 2025-11-02
**dooit Version**: Latest commit 2024-11-25
**Textual Framework**: Built on Textual (version not specified in README)
