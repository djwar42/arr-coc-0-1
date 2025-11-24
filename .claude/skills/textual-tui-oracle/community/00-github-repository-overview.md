# GitHub Repository Overview - Textualize/textual

## Repository Statistics

**Repository**: [Textualize/textual](https://github.com/Textualize/textual)

**Statistics** (as of 2025-11-02):
- **Stars**: 31,907
- **Forks**: 1,005
- **Watchers**: 178
- **Open Issues**: 172
- **Open Pull Requests**: 50
- **Contributors**: 183+
- **Latest Release**: v6.5.0 "The Spooky Trap" (Oct 31, 2025)
- **Total Releases**: 192

**License**: MIT License

**Platform Support**: macOS, Linux, Windows

**Python Versions**: Multiple versions supported (see PyPI badge)

## Repository Structure

The Textual repository is well-organized with the following key directories:

### Core Directories

**`.github/`** - GitHub-specific configuration (workflows, templates)

**`docs/`** - Documentation source files (MkDocs-based)

**`examples/`** - Official example applications demonstrating Textual features

**`src/textual/`** - Main source code for the Textual framework

**`tests/`** - Test suite for framework validation

**`tools/`** - Development and build tools

### Documentation & Reference

**`reference/`** - API reference materials

**`notes/`** - Development notes and design decisions

**`questions/`** - FAQ and common questions

**`.faq/`** - Frequently asked questions content

### Supporting Files

**`imgs/`** - Images and visual assets

**`pyproject.toml`** - Project configuration and dependencies (Poetry-based)

**`Makefile`** - Build automation commands

**`mkdocs-*.yml`** - MkDocs configuration files for documentation

**`poetry.lock`** - Locked dependency versions

## Key Repository Files

### README.md Overview

From [README.md](https://github.com/Textualize/textual/blob/main/README.md):

The README showcases Textual as:
- "The lean application framework for Python"
- Build sophisticated user interfaces with a simple Python API
- Run apps in terminal AND web browser
- 211 lines of well-structured documentation

**Key Example**: Clock App demonstrating:
```python
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import Digits

class ClockApp(App):
    CSS = """
    Screen { align: center middle; }
    Digits { width: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Digits("")

    def on_ready(self) -> None:
        self.update_clock()
        self.set_interval(1, self.update_clock)

    def update_clock(self) -> None:
        clock = datetime.now().time()
        self.query_one(Digits).update(f"{clock:%T}")
```

### Main Features Highlighted

**Widget Library**:
- Buttons, tree controls, data tables
- Input fields, text areas
- ListView, DataTable
- Flexible layout system
- Predefined themes

**Development Tools**:
- Dev console (`textual-dev` package)
- Command palette (fuzzy search via `ctrl+p`)
- Testing framework with Pilot
- DevTools for debugging

**Web Deployment**:
- `textual serve` command for browser access
- Textual Web for firewall-busting technology
- Cross-platform deployment (terminal + browser)

## Community & Support

**Discord Server**: Active community at [discord.gg/Enf6Z3qhVr](https://discord.gg/Enf6Z3qhVr)

**Documentation**: [textual.textualize.io](https://textual.textualize.io/)

**Company**: Built by Textualize.io

**Code of Conduct**: Available in repository

**Contributing Guide**: CONTRIBUTING.md (available in repository)

## Installation

```bash
pip install textual textual-dev
```

**Quick Demo**:
```bash
python -m textual
```

**Or without installation** (using uv):
```bash
uvx --python 3.12 textual-demo
```

## Repository Activity

**Recent Activity** (from commit history):
- Latest feature: "docs(readme): update python versions badge" (2025-10-07)
- Active development with 12,659+ commits
- Regular releases (192 total releases)
- Maintained by core team including @willmcgugan, @davep, @darrenburns, @rodrigogiraoserrao

## Notable Repository Features

**Comprehensive Documentation Setup**:
- MkDocs-based documentation system
- Multiple MkDocs config files (online, offline, common, nav)
- FAQ system with YAML configuration
- Extensive guides and tutorials

**Testing Infrastructure**:
- Full test suite in `tests/` directory
- Coverage configuration (`.coveragerc`)
- Pre-commit hooks (`.pre-commit-config.yaml`)
- Type checking with mypy (`mypy.ini`)

**Build System**:
- Poetry for dependency management
- Automated builds via Makefile
- DeepSource integration (`.deepsource.toml`)

## Topics/Tags

The repository is tagged with:
- python
- cli
- framework
- terminal
- tui
- rich (related to Rich library by same author)

## Used By

**8,100+ repositories** depend on Textual, indicating strong ecosystem adoption.

## Development Philosophy

From the README:
> "Textual's API combines modern Python with the best of developments from the web world, for a lean app development experience. De-coupled components and an advanced testing framework ensure you can maintain your app for the long-term."

> "Textual is an asynchronous framework under the hood. Which means you can integrate your apps with async libraries — if you want to. If you don't want or need to use async, Textual won't force it on you."

## Sources

**GitHub Repository**: [https://github.com/Textualize/textual](https://github.com/Textualize/textual) (accessed 2025-11-02)

**README.md**: [https://github.com/Textualize/textual/blob/main/README.md](https://github.com/Textualize/textual/blob/main/README.md) (accessed 2025-11-02)

**Repository Stats**: Retrieved via GitHub web interface (2025-11-02)

---

**Cross-References**:
- See [examples/00-github-examples-index.md](../examples/00-github-examples-index.md) for detailed examples overview
- See [releases/00-version-3-3-0.md](../releases/00-version-3-3-0.md) for latest release information
- See [getting-started/](../getting-started/) for installation and quickstart guides
