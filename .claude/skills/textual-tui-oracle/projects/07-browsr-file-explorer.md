# Browsr - Universal File Explorer TUI

## Overview

**Browsr** is a production-grade terminal file explorer built with Textual that provides a pleasant browsing experience for both local and remote filesystems. Created by Justin Flannery (juftin), it demonstrates enterprise-level TUI architecture with support for cloud storage, GitHub repositories, SSH/SFTP, and advanced file rendering.

**GitHub**: https://github.com/juftin/browsr (accessed 2025-11-02)

**Key Features**:
- Universal filesystem support (local, S3, GCS, Azure, GitHub, SSH/SFTP)
- Advanced file rendering (syntax highlighting, images, PDFs, JSON, datatables)
- Rich configuration system with environment variables
- Plugin architecture for extensibility
- Production-ready packaging and distribution

## Architecture Overview

### Application Structure

```
browsr/
├── browsr.py          # Main Textual app class
├── cli.py             # Rich-Click command-line interface
├── config.py          # Configuration and constants
├── __main__.py        # Entry point
└── screens.py         # Screen components (404 in current version)
```

### Core Dependencies

From [pyproject.toml](https://github.com/juftin/browsr/blob/main/pyproject.toml):

```toml
dependencies = [
  "art~=6.5",                              # ASCII art headers
  "click~=8.1.7",                          # CLI framework
  "pandas>2,<3",                           # Dataframe support
  "rich>=14,<15",                          # Rich text rendering
  "rich-click~=1.8.9",                     # Rich CLI styling
  "rich-pixels~=3.0.1",                    # Pixel rendering for images
  "textual>=5,<6",                         # TUI framework
  "textual-universal-directorytree~=1.6.0", # Universal filesystem tree widget
  "universal-pathlib~=0.2.6",              # Unified pathlib API
  "Pillow>=11.3.0",                        # Image processing
  "PyMuPDF~=1.26.3",                       # PDF rendering
  "pyperclip~=1.9.0"                       # Clipboard support
]
```

**Remote Filesystem Support**:
```toml
[project.optional-dependencies]
remote = [
  "textual-universal-directorytree[remote]~=1.6.0"
]
```

## Main Application Class

From [browsr.py](https://github.com/juftin/browsr/blob/main/browsr.py):

```python
from textual.app import App
from textual.binding import Binding, BindingType
from browsr.base import TextualAppContext
from browsr.screens import CodeBrowserScreen

class Browsr(App[str]):
    """Textual code browser app."""

    TITLE = __application__
    CSS_PATH = "browsr.css"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="d", action="toggle_dark", description="Dark Mode"),
    ]

    def __init__(
        self,
        config_object: TextualAppContext | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        config_object: Optional[TextualAppContext]
            Configuration object for the application
        """
        super().__init__(*args, **kwargs)
        self.config_object = config_object or TextualAppContext()
        self.code_browser_screen = CodeBrowserScreen(config_object=self.config_object)
        self.install_screen(self.code_browser_screen, name="code-browser")

    @on(Mount)
    async def mount_screen(self) -> None:
        """Mount the screen"""
        await self.push_screen(screen=self.code_browser_screen)

    def action_copy_file_path(self) -> None:
        """Copy the file path to the clipboard"""
        self.code_browser_screen.code_browser.copy_file_path()

    def action_download_file(self) -> None:
        """Copy the file path to the clipboard"""
        self.code_browser_screen.code_browser.download_file_workflow()
```

**Key Patterns**:
- Dependency injection via `config_object` parameter
- Screen-based navigation (code-browser screen)
- Custom actions for file operations (copy, download)
- Centralized configuration object

## Command-Line Interface

From [cli.py](https://github.com/juftin/browsr/blob/main/browsr/cli.py):

```python
import click
import rich_click
from browsr.base import TextualAppContext
from browsr.browsr import Browsr

# Rich-Click styling configuration
rich_click.rich_click.MAX_WIDTH = 100
rich_click.rich_click.STYLE_OPTION = "bold green"
rich_click.rich_click.STYLE_SWITCH = "bold blue"
rich_click.rich_click.STYLE_METAVAR = "bold red"

@click.command(name="browsr", cls=rich_click.rich_command.RichCommand)
@click.argument("path", default=None, required=False, metavar="PATH")
@click.option(
    "-l", "--max-lines",
    default=1000,
    show_default=True,
    type=int,
    help="Maximum number of lines to display in the code browser",
    envvar="BROWSR_MAX_LINES",
    show_envvar=True,
)
@click.option(
    "-m", "--max-file-size",
    default=20,
    show_default=True,
    type=int,
    help="Maximum file size in MB for the application to open",
    envvar="BROWSR_MAX_FILE_SIZE",
    show_envvar=True,
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable extra debugging output",
    type=click.BOOL,
    envvar="BROWSR_DEBUG",
    show_envvar=True,
)
@click.option(
    "-k", "--kwargs",
    multiple=True,
    help="Key=Value pairs to pass to the filesystem",
    envvar="BROWSR_KWARGS",
    show_envvar=True,
)
def browsr(
    path: Optional[str],
    debug: bool,
    max_lines: int,
    max_file_size: int,
    kwargs: Tuple[str, ...],
) -> None:
    """
    browsr 🗂️  a pleasant file explorer in your terminal

    Navigate through directories and peek at files whether they're hosted locally,
    over SSH, in GitHub, AWS S3, Google Cloud Storage, or Azure Blob Storage.
    """
    extra_kwargs = {}
    if kwargs:
        for kwarg in kwargs:
            try:
                key, value = kwarg.split("=")
                extra_kwargs[key] = value
            except ValueError as ve:
                raise click.BadParameter(
                    message=(
                        f"Invalid Key/Value pair: `{kwarg}` "
                        "- must be in the format Key=Value"
                    ),
                    param_hint="kwargs",
                ) from ve

    file_path = path or os.getcwd()
    config = TextualAppContext(
        file_path=file_path,
        debug=debug,
        max_file_size=max_file_size,
        max_lines=max_lines,
        kwargs=extra_kwargs,
    )
    app = Browsr(config_object=config)
    app.run()
```

**CLI Design Patterns**:
- Rich-Click for beautiful terminal output with colors and formatting
- Environment variable support (`BROWSR_MAX_LINES`, `BROWSR_DEBUG`, etc.)
- Flexible configuration via key=value pairs (`-k anon=True`)
- Sensible defaults (1000 max lines, 20MB max file size)
- Type validation and error handling

## Universal Filesystem Support

Browsr leverages **textual-universal-directorytree** for filesystem abstraction:

### Supported Filesystems

From [textual-universal-directorytree README](https://github.com/juftin/textual-universal-directorytree/blob/main/README.md):

| Filesystem | Format | Optional Dependency |
|------------|--------|---------------------|
| Local | `path/to/file` | None |
| Local | `file://path/to/file` | None |
| AWS S3 | `s3://bucket/path` | s3fs |
| AWS S3 | `s3a://bucket/path` | s3fs |
| Google GCS | `gs://bucket/path` | gcsfs |
| Azure Data Lake | `adl://bucket/path` | adlfs |
| Azure Blob | `abfs://bucket/path` | adlfs |
| Azure Blob | `az://bucket/path` | adlfs |
| GitHub | `github://owner:repo@branch` | requests |
| GitHub | `github://owner:repo@branch/path` | requests |
| SSH | `ssh://user@host:port/path` | paramiko |
| SFTP | `sftp://user@host:port/path` | paramiko |

### Usage Examples

**Local filesystem**:
```bash
browsr ~/Downloads/
```

**GitHub repository**:
```bash
browsr github://juftin:browsr
export GITHUB_TOKEN="ghp_1234567890"
browsr github://juftin:browsr-private@main
```

**Cloud storage**:
```bash
browsr s3://my-bucket
browsr gs://my-bucket
browsr az://my-bucket
```

**SSH/SFTP**:
```bash
browsr ssh://username@example.com:22
browsr sftp://user@host:22/path/to/directory
```

**Anonymous S3 bucket**:
```bash
browsr s3://anonymous-bucket -k anon=True
```

## UniversalDirectoryTree Widget

From [textual-universal-directorytree](https://github.com/juftin/textual-universal-directorytree):

```python
from textual_universal_directorytree import UniversalDirectoryTree, UPath
from textual.widgets import DirectoryTree

class UniversalDirectoryTreeApp(App):
    def __init__(self, path: str | UPath, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.universal_path = UPath(path).resolve()
        self.directory_tree = UniversalDirectoryTree(path=self.universal_path)

    @on(DirectoryTree.FileSelected)
    def handle_file_selected(self, message: DirectoryTree.FileSelected) -> None:
        """
        Objects returned by FileSelected event are upath.UPath objects,
        compatible with pathlib.Path API
        """
        self.sub_title = str(message.path)
        try:
            file_content = message.path.read_text()
        except UnicodeDecodeError:
            self.file_content.update("")
            return None

        lexer = Syntax.guess_lexer(path=message.path.name, code=file_content)
        code = Syntax(code=file_content, lexer=lexer)
        self.file_content.update(code)
```

**Key Features**:
- `UPath` objects provide unified API across all filesystems
- Compatible with standard `pathlib.Path` methods
- Automatic filesystem detection from URI scheme
- Events work identically regardless of filesystem type

## Configuration System

From [config.py](https://github.com/juftin/browsr/blob/main/browsr/config.py):

```python
from os import getenv
from typing import List

# Rich syntax highlighting themes
favorite_themes: List[str] = [
    "monokai",
    "material",
    "dracula",
    "solarized-light",
    "one-dark",
    "solarized-dark",
    "emacs",
    "vim",
    "github-dark",
    "native",
    "paraiso-dark",
]

# Respect RICH_THEME environment variable
rich_default_theme = getenv("RICH_THEME", None)
if rich_default_theme in favorite_themes:
    favorite_themes.remove(rich_default_theme)
if rich_default_theme is not None:
    favorite_themes.insert(0, rich_default_theme)

# Image file extensions for rendering
image_file_extensions = [
    ".bmp", ".dib", ".eps", ".ps", ".gif", ".icns", ".ico", ".cur",
    ".im", ".im.gz", ".im.bz2", ".jpg", ".jpe", ".jpeg", ".jfif",
    ".msp", ".pcx", ".png", ".ppm", ".pbm", ".pgm", ".sgi", ".rgb",
    ".bw", ".spi", ".tif", ".tiff", ".webp", ".xbm", ".xv", ".pdf",
]
```

**Configuration Patterns**:
- Environment variable integration (`RICH_THEME`, `BROWSR_*`)
- Sensible defaults with user overrides
- Extensive file type support for image rendering
- Theme cycling support for syntax highlighting

## Key Bindings

From [README.md](https://github.com/juftin/browsr/blob/main/README.md) documentation:

| Key | Action | Description |
|-----|--------|-------------|
| `Q` | Quit | Exit the application |
| `F` | Toggle Sidebar | Show/hide file tree sidebar |
| `T` | Toggle Theme | Cycle through Rich syntax themes |
| `N` | Toggle Line Numbers | Show/hide line numbers in code |
| `D` | Toggle Dark Mode | Switch between dark/light modes |
| `.` | Parent Directory | Navigate up one directory |
| `R` | Reload | Refresh current directory |
| `C` | Copy Path | Copy file/directory path to clipboard |
| `X` | Download | Download file from cloud storage |

## File Rendering Capabilities

Browsr provides advanced file rendering through multiple libraries:

**Text Files**:
- Syntax highlighting via Rich (100+ language lexers)
- Theme cycling (`T` key)
- Line number toggling (`N` key)
- Max line limit (default 1000, configurable)

**Images** (via Pillow + rich-pixels):
- Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP, etc.
- Terminal-based pixel rendering
- Automatic scaling to terminal dimensions

**PDFs** (via PyMuPDF):
- Text extraction from PDF documents
- Page-by-page rendering
- Metadata display

**Data Files** (via pandas):
- CSV, Parquet, Feather files
- Navigable datatables
- Column sorting and filtering

**JSON**:
- Pretty-printed formatting
- Syntax highlighting
- Structure visualization

## Installation and Distribution

**Basic installation**:
```bash
pipx install browsr
```

**With remote filesystem support**:
```bash
pipx install "browsr[remote]"
```

**With data file support**:
```bash
pipx install "browsr[data]"
```

**All features**:
```bash
pipx install "browsr[all]"
```

**Package entry point** (from pyproject.toml):
```toml
[project.scripts]
browsr = "browsr.__main__:browsr"
```

## Production-Ready Features

### Error Handling

From cli.py - filesystem kwargs parsing:
```python
try:
    key, value = kwarg.split("=")
    extra_kwargs[key] = value
except ValueError as ve:
    raise click.BadParameter(
        message=(
            f"Invalid Key/Value pair: `{kwarg}` "
            "- must be in the format Key=Value"
        ),
        param_hint="kwargs",
    ) from ve
```

### File Size Limits

```python
@click.option(
    "-m", "--max-file-size",
    default=20,
    show_default=True,
    type=int,
    help="Maximum file size in MB for the application to open",
)
```

Prevents memory issues when browsing large files.

### Debug Mode

```python
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable extra debugging output",
    envvar="BROWSR_DEBUG",
)
```

Enables developer console and extra logging.

## Architecture Insights

### Screen-Based Navigation

```python
self.code_browser_screen = CodeBrowserScreen(config_object=self.config_object)
self.install_screen(self.code_browser_screen, name="code-browser")

@on(Mount)
async def mount_screen(self) -> None:
    await self.push_screen(screen=self.code_browser_screen)
```

**Pattern**: Separate screen classes for different views, installed and pushed on mount.

### Configuration Dependency Injection

```python
config = TextualAppContext(
    file_path=file_path,
    debug=debug,
    max_file_size=max_file_size,
    max_lines=max_lines,
    kwargs=extra_kwargs,
)
app = Browsr(config_object=config)
```

**Pattern**: CLI builds config object, passes to app constructor (testable, flexible).

### Custom Actions

```python
def action_copy_file_path(self) -> None:
    """Copy the file path to the clipboard"""
    self.code_browser_screen.code_browser.copy_file_path()

def action_download_file(self) -> None:
    """Download file from remote storage"""
    self.code_browser_screen.code_browser.download_file_workflow()
```

**Pattern**: Custom actions delegate to screen/widget methods (separation of concerns).

## Textual Patterns Demonstrated

### 1. Universal Widget Extension

Browsr uses `textual-universal-directorytree`, which extends Textual's built-in `DirectoryTree` widget to support remote filesystems. This demonstrates:
- Widget inheritance and extension
- Filesystem abstraction (fsspec + universal-pathlib)
- Event compatibility (same `FileSelected` event)

### 2. Rich Integration

Heavy use of Rich library alongside Textual:
- `rich-click` for beautiful CLI
- `rich-pixels` for image rendering
- `Syntax` for code highlighting
- Theme management

### 3. Configuration Management

Multiple configuration layers:
- Environment variables (`BROWSR_*`, `RICH_THEME`)
- CLI arguments with defaults
- Runtime options (`-k key=value`)
- Configuration object injection

### 4. Production Packaging

Professional Python packaging:
- `pyproject.toml` with full metadata
- Optional dependencies (`[remote]`, `[data]`, `[all]`)
- Entry point scripts
- Comprehensive dev dependencies (testing, linting, docs)

## Use Cases

### 1. Cloud Storage Exploration

Browse S3/GCS/Azure buckets directly in terminal:
```bash
browsr s3://my-data-bucket/analytics/
browsr gs://ml-models/checkpoints/
```

### 2. GitHub Repository Analysis

Quickly explore repos without cloning:
```bash
browsr github://python:cpython@main/Lib
browsr github://torvalds:linux@master/drivers
```

### 3. Remote Server File Management

Navigate remote filesystems over SSH:
```bash
browsr ssh://deploy@prod-server:22/var/log
browsr sftp://backup@storage:22/backups/daily
```

### 4. Local Development

Enhanced file browsing with syntax highlighting:
```bash
browsr ~/projects/myapp/src
browsr . # Current directory
```

### 5. Data Analysis

Browse and preview data files:
```bash
browsr s3://analytics-bucket/reports/  # View parquet files as tables
browsr ~/datasets/  # Preview CSVs and images
```

## Testing and Development

From pyproject.toml dev dependencies:
```toml
[dependency-groups]
dev = [
  # Testing
  "pytest",
  "pytest-cov",
  "pytest-vcr~=1.0.2",
  "textual-dev~=1.4.0",
  "pytest-textual-snapshot",
  "pytest-asyncio",
  # Linting
  "mypy>=1.9.0",
  "ruff~=0.1.7"
]
```

**Testing stack**:
- pytest with coverage
- pytest-vcr for HTTP mocking (GitHub filesystem tests)
- textual-dev for TUI testing utilities
- pytest-textual-snapshot for visual regression testing
- pytest-asyncio for async test support

**Type safety**:
- mypy with strict configuration
- Type hints throughout codebase

## Lessons for Textual Developers

### 1. Filesystem Abstraction

Use libraries like `fsspec` and `universal-pathlib` for unified filesystem APIs:
- Write once, work everywhere
- Same code for local and remote
- Transparent protocol handling

### 2. CLI Design

Rich-Click provides professional CLI experience:
- Colorful help messages
- Environment variable integration
- Type validation
- Clear error messages

### 3. Configuration Patterns

Multi-layer configuration system:
- Environment variables for defaults
- CLI args for overrides
- Config object for runtime state
- Dependency injection for testability

### 4. Screen Organization

Separate screens for distinct application modes:
- Install screens at app creation
- Push/pop for navigation
- Screen-specific configuration

### 5. Widget Extension

Extend built-in widgets for specialized behavior:
- Inherit from Textual widgets
- Add new capabilities (remote filesystems)
- Maintain event compatibility

### 6. Production Readiness

Enterprise-grade features:
- File size limits
- Debug mode
- Error handling
- Comprehensive testing
- Type safety
- Documentation

## Sources

**GitHub Repositories**:
- [juftin/browsr](https://github.com/juftin/browsr) - Main repository (accessed 2025-11-02)
- [juftin/textual-universal-directorytree](https://github.com/juftin/textual-universal-directorytree) - Universal DirectoryTree widget (accessed 2025-11-02)

**Files Analyzed**:
- [README.md](https://github.com/juftin/browsr/blob/main/README.md) - Project documentation
- [pyproject.toml](https://github.com/juftin/browsr/blob/main/pyproject.toml) - Dependencies and configuration
- [browsr.py](https://github.com/juftin/browsr/blob/main/browsr/browsr.py) - Main application class
- [cli.py](https://github.com/juftin/browsr/blob/main/browsr/cli.py) - Command-line interface
- [config.py](https://github.com/juftin/browsr/blob/main/browsr/config.py) - Configuration constants
- [__main__.py](https://github.com/juftin/browsr/blob/main/browsr/__main__.py) - Entry point

**Package Information**:
- PyPI: `browsr` version 1.22.1
- License: MIT
- Python: 3.9-3.13 support
- Textual: 5.x requirement

## Summary

Browsr demonstrates professional-grade Textual application development with:
- Universal filesystem support (10+ protocols)
- Rich file rendering (code, images, PDFs, data)
- Production-ready CLI with Rich-Click
- Comprehensive configuration system
- Clean architecture with dependency injection
- Enterprise features (limits, debug mode, testing)
- Beautiful terminal UX

It serves as an excellent reference for building complex, production-ready TUI applications that integrate multiple libraries and handle diverse file sources.
