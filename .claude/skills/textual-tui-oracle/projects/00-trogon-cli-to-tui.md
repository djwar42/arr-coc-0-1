# Trogon - CLI to TUI Converter

**Source**: [Textualize/trogon](https://github.com/Textualize/trogon) (accessed 2025-11-02)
**Official Textualize Project**: Yes
**Stars**: 2,728 | **Forks**: 66

---

## Overview

Trogon auto-generates friendly terminal user interfaces (TUIs) for command line applications. It inspects your CLI app and creates a Textual-based interactive interface that helps users discover and use commands without memorizing options.

**Core Concept**: "Swagger for CLIs" - extracts a schema describing CLI options/switches/help and builds an interactive TUI for editing and running commands.

**Target Problem**: Command line apps reward repeated use but lack discoverability. Trogon bridges this gap with an interactive interface for CLI exploration.

---

## Key Features

### Schema Extraction
- Inspects CLI apps to extract command structure
- Builds schema describing options, switches, arguments, help text
- Future goal: Formalize schema/protocol for any CLI app (language-agnostic)

### Interactive TUI
- Browse available commands and subcommands
- Edit command parameters through forms
- View help text inline
- Execute commands from TUI
- Built on Textual framework

### Framework Support
- **Click** (Python): Fully supported
- **Typer** (Python): Fully supported
- **Future**: Other libraries and languages planned

---

## Installation

```bash
pip install trogon
```

---

## Usage Examples

### Click Integration

**Only 2 lines of code needed:**

```python
from trogon import tui
import click

@tui()  # 1. Add @tui decorator
@click.group()
def cli():
    """Your CLI application"""
    pass

@cli.command()
@click.option('--name', default='World', help='Name to greet')
def hello(name):
    """Say hello"""
    click.echo(f'Hello {name}!')

if __name__ == '__main__':
    cli()
```

**Result**: Adds a `tui` command to your Click app.

```bash
# Launch TUI interface
python app.py tui
```

### Typer Integration

```python
import typer
from trogon.typer import init_tui

cli = typer.Typer()
init_tui(cli)  # Initialize TUI support

@cli.command()
def greet(name: str = "World"):
    """Greet someone"""
    print(f"Hello {name}!")

if __name__ == '__main__':
    cli()
```

**Result**: Adds a `tui` command to your Typer app.

### Custom Command Name and Help

```python
@tui(command="ui", help="Open terminal UI")
@click.group()
def cli():
    """Your CLI application"""
    pass
```

**Customization**:
- `command=` - Change TUI command name (default: "tui")
- `help=` - Change help text (default: "Open Textual TUI.")

---

## Architecture Insights

### Schema-Based Design

```
CLI App (Click/Typer)
    ↓
Trogon Inspector
    ↓
Schema Extraction
    ↓
Textual TUI Generation
    ↓
Interactive Command Builder
    ↓
Execute Command
```

**Key Pattern**: Introspection → Schema → UI Generation → Execution

### Integration Pattern

**Decorator Pattern (Click)**:
```python
@tui()           # Trogon layer
@click.group()   # Click layer
def cli():
    pass
```

**Function Call Pattern (Typer)**:
```python
cli = typer.Typer()
init_tui(cli)  # Inject TUI capability
```

### Textual Framework Foundation
- Built entirely on Textual
- Inherits Textual's reactive, widget-based architecture
- Uses Textual's CSS styling system
- Leverages Textual's event handling

---

## Real-World Application

**Demonstrated with**: [sqlite-utils](https://github.com/simonw/sqlite-utils)
- SQLite CLI tool by Simon Willison
- Complex command structure with many options
- Trogon makes exploration/discovery easier
- Video demonstration available in README

---

## Code Patterns

### Minimal Integration
```python
# Complete Click + Trogon integration
from trogon import tui
import click

@tui()
@click.group()
def cli():
    pass

# That's it - 2 lines!
```

### Schema Inspection
- Trogon introspects Click command decorators
- Extracts: command names, options, arguments, help text, types
- Builds hierarchical command tree
- Maps to Textual widgets (forms, inputs, buttons)

### Command Execution
- TUI builds command string from user inputs
- Executes via underlying Click/Typer framework
- Displays output in TUI

---

## Project Context

### Birds Theme
Part of Textualize's bird-themed projects:
- **Trogon**: CLI to TUI converter (this project)
- **Frogmouth**: Markdown browser for terminal

**Name Origin**: Trogon is a beautiful bird species (photographed by Will McGugan, Textual creator, in 2017)

### Development Status
- **Usable Now**: Production-ready for Click/Typer apps
- **Early Stage**: Lots of planned improvements
- **Active**: Last updated 2024-09-20
- **Community**: 13 pull requests, active Discord

---

## Key Takeaways

### For Textual Developers

**Schema-driven UI generation**:
- Inspect external library structures
- Generate Textual widgets programmatically
- Create domain-specific TUI builders

**Integration patterns**:
- Decorator-based injection
- Function-based initialization
- Minimal API surface (2 lines of code)

**Introspection techniques**:
- Extract metadata from decorated functions
- Build hierarchical command structures
- Map CLI types to UI widgets

### For CLI Developers

**Instant TUI interface**:
- Add `@tui()` decorator
- Get full interactive interface
- No UI code required

**Discoverability**:
- Help users explore commands
- Reduce memorization burden
- Visual command building

---

## Related Projects

- [Textual](https://github.com/Textualize/textual) - Foundation TUI framework
- [Click](https://click.palletsprojects.com/) - Python CLI library
- [Typer](https://typer.tiangolo.com/) - Python CLI library
- [Frogmouth](https://github.com/Textualize/frogmouth) - Markdown browser TUI

---

## Resources

- **Repository**: https://github.com/Textualize/trogon
- **Discord**: https://discord.gg/Enf6Z3qhVr
- **Video Demo**: Available in README (sqlite-utils demonstration)

---

## Sources

**Primary Source**:
- [Trogon README.md](https://github.com/Textualize/trogon/blob/main/README.md) - GitHub repository (accessed 2025-11-02)

**Additional Context**:
- Official Textualize project
- Built with Textual framework
- Active development (2,728 stars, 66 forks)
- Last updated: 2024-09-20
