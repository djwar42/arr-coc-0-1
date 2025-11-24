# TUI Project Templates and Structure

## Overview

This document provides production-ready project templates for Textual TUI applications. Based on real-world practices from experienced TUI developers using modern Python tooling (uv, dependency management, packaging).

Sources:
- Textual Definitive Guide Part 1 (DEV.to) - Mahmoud Harmouch
- My Python TUI Project Template (CodeCurrents) - Michel Lavoie
- Official Textual documentation and examples

---

## Template 1: Minimal TUI Application

### Perfect For: Simple utilities, learning projects, quick prototypes

```
minimal-tui/
├── pyproject.toml              # Project metadata
├── README.md                   # Documentation
├── requirements.txt            # Optional: for pip
└── main.py                     # Single-file app
```

### pyproject.toml (Minimal)

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "minimal-tui"
version = "0.1.0"
description = "A simple Textual TUI application"
requires-python = ">=3.10"
dependencies = [
    "textual>=0.40.0",
]

[project.scripts]
minimal-tui = "main:run"
```

### main.py

```python
"""Minimal Textual application."""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static


class MinimalApp(App):
    """A minimal Textual application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    Static {
        width: 100%;
        height: 1fr;
        content-align: center middle;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Static("Hello, Textual!", id="main")
        yield Footer()


def run():
    """Run the application."""
    app = MinimalApp()
    app.run()


if __name__ == "__main__":
    run()
```

### Installation and Running

```bash
# Install for development
pip install -e .

# Run
minimal-tui

# Or run directly
python main.py
```

---

## Template 2: Standard Project Structure

### Perfect For: Medium-sized applications, team projects, maintainability

```
my-tui-app/
├── pyproject.toml              # Project metadata + dependencies
├── uv.lock                      # Lock file (uv package manager)
├── README.md                    # Project documentation
├── CHANGELOG.md                 # Version history
├── LICENSE                      # License (MIT, Apache, etc)
│
├── src/
│   └── my_tui_app/
│       ├── __init__.py
│       ├── __main__.py          # Entry point
│       ├── app.py               # Main App class
│       ├── config.py            # Configuration management
│       ├── constants.py          # App constants
│       │
│       ├── screens/             # Different app screens/views
│       │   ├── __init__.py
│       │   ├── home.py
│       │   ├── details.py
│       │   └── settings.py
│       │
│       ├── widgets/             # Custom widgets
│       │   ├── __init__.py
│       │   ├── header.py
│       │   ├── footer.py
│       │   └── sidebar.py
│       │
│       └── utils/               # Utility functions
│           ├── __init__.py
│           └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_widgets.py
│   ├── test_screens.py
│   └── fixtures.py
│
└── docs/                        # Documentation (optional)
    ├── getting-started.md
    ├── architecture.md
    └── contributing.md
```

### pyproject.toml (Standard)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-tui-app"
version = "0.1.0"
description = "A production-ready Textual TUI application"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Your Name", email = "you@example.com"}]
keywords = ["tui", "terminal", "textual"]

dependencies = [
    "textual>=0.40.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",  # For config management
    "python-dotenv>=0.21.0",  # For .env support
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "pytest-cov>=4.0",
]

[project.scripts]
my-tui-app = "my_tui_app:run"

[tool.hatch.build.targets.wheel]
packages = ["src/my_tui_app"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
select = ["E", "F", "W"]
line-length = 100
```

### src/my_tui_app/__init__.py

```python
"""My TUI Application."""

__version__ = "0.1.0"

from my_tui_app.app import MyApp


def run():
    """Entry point for the application."""
    app = MyApp()
    app.run()


__all__ = ["MyApp", "run"]
```

### src/my_tui_app/__main__.py

```python
"""Allow running module with python -m my_tui_app."""

from my_tui_app import run

if __name__ == "__main__":
    run()
```

### src/my_tui_app/app.py

```python
"""Main application class."""

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer

from my_tui_app.screens.home import HomeScreen
from my_tui_app.config import Config


class MyApp(App):
    """Main application."""

    BINDINGS = [
        ("d", "dark_mode", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    CSS_PATH = "app.css"

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config()

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Container(id="main")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize app on mount."""
        self.push_screen(HomeScreen())

    def action_dark_mode(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark
```

### src/my_tui_app/config.py

```python
"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    """Application configuration."""

    app_name: str = "My TUI App"
    version: str = "0.1.0"
    debug: bool = False
    config_path: Path = field(default_factory=lambda: Path.home() / ".config" / "my-tui-app")
    theme: str = "default"

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        import os
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            theme=os.getenv("THEME", "default"),
        )

    def save(self) -> None:
        """Save config to file."""
        self.config_path.mkdir(parents=True, exist_ok=True)
        # Implementation depends on your needs
```

### tests/test_app.py

```python
"""Test main application."""

import pytest
from textual.pilot import Pilot

from my_tui_app.app import MyApp


@pytest.fixture
async def app():
    """Provide test app."""
    app = MyApp()
    async with app.run_test() as pilot:
        yield pilot


async def test_app_starts(app: Pilot):
    """Test that app starts successfully."""
    assert app.app.title is not None


async def test_dark_mode_toggle(app: Pilot):
    """Test dark mode toggling."""
    initial_dark = app.app.dark
    await app.press("d")
    assert app.app.dark != initial_dark
```

### Installation with uv

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"

# Or with uv sync (preferred)
uv sync --all-extras
```

### Development Workflow

```bash
# Run app
my-tui-app

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/my_tui_app

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

---

## Template 3: Production-Ready with Packaging

### Perfect For: Distributing to end users, commercial applications

```
production-tui/
├── pyproject.toml              # Project config
├── uv.lock
├── build_config.py             # Build settings (for Nuitka)
├── README.md
├── CHANGELOG.md
├── LICENSE
│
├── src/
│   └── my_app/
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── screens/
│       ├── widgets/
│       ├── utils/
│       └── assets/             # Icons, data files
│           ├── logo.txt
│           └── themes/
│
├── tests/
│   └── ...
│
├── scripts/
│   ├── build.py                # Build script
│   └── release.py              # Release automation
│
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions
```

### pyproject.toml (Production)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-app"
version = "1.0.0"
description = "Production TUI application"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Company Name", email = "team@example.com"}
]
keywords = ["tui", "terminal", "app"]
classifiers = [
    "Environment :: Console",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software :: User Interfaces",
]

dependencies = [
    "textual>=0.40.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "python-dotenv>=0.21.0",
    "click>=8.1.0",  # CLI argument parsing
    "requests>=2.31.0",  # HTTP requests
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
build = [
    "nuitka>=1.7.0",  # Compile to binary
    "zstandard",  # For Nuitka
]

[project.scripts]
my-app = "my_app.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/my_app"]

[tool.hatch.build.targets.sdist]
include = ["src/", "tests/", "docs/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src/my_app --cov-report=html"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### scripts/build.py

```python
"""Build script for creating standalone binary."""

import subprocess
import sys
from pathlib import Path

def build_binary():
    """Compile to standalone binary using Nuitka."""
    project_root = Path(__file__).parent.parent

    # Ensure Nuitka is installed
    try:
        import nuitka
    except ImportError:
        print("Nuitka not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka"])

    # Compile to binary
    cmd = [
        sys.executable,
        "-m", "nuitka",
        "--onefile",
        "--output-dir=dist/",
        "--include-package=my_app",
        str(project_root / "src" / "my_app" / "__main__.py"),
    ]

    print(f"Building: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("Build complete! Binary in dist/")

if __name__ == "__main__":
    build_binary()
```

### .github/workflows/ci.yml

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: uv pip install -e ".[dev]"

    - name: Run tests
      run: pytest --cov

    - name: Lint
      run: ruff check src/ tests/

    - name: Type check
      run: mypy src/
```

---

## Dependency Management with uv

### Why uv?

- 10-100x faster than pip
- Single tool for venvs, pip, and pipx
- Lock files for reproducible builds
- Simpler syntax than Poetry

### Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv init my-tui-app
cd my-tui-app

# Add dependencies
uv pip install textual rich pydantic

# Create lock file
uv lock

# Install for development
uv pip install -e .
```

### uv.lock File

```
version = 1
requires-python = ">=3.10"

[[package]]
name = "textual"
version = "0.40.0"
source = { type = "registry", url = "https://pypi.org/simple" }

[[package]]
name = "rich"
version = "13.6.0"
source = { type = "registry", url = "https://pypi.org/simple" }
```

---

## Configuration Management

### Environment-Based Config

```python
# config.py
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    app_name: str = "MyApp"
    debug: bool = False
    config_dir: Path = Path.home() / ".config" / "myapp"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
from config import Settings

settings = Settings()  # Reads from .env or environment
if settings.debug:
    print("Debug mode enabled")
```

### .env File

```bash
# .env
APP_NAME=MyApp
DEBUG=true
CONFIG_DIR=/home/user/.config/myapp
LOG_LEVEL=DEBUG
```

---

## Common Project Pitfalls

### Problem 1: No Virtual Environment
```bash
# ❌ Wrong
pip install textual

# ✓ Right
uv venv
source .venv/bin/activate
uv pip install textual
```

### Problem 2: Mixing Installation Methods
```bash
# ❌ Mixed approaches
pip install textual
uv pip install rich  # Don't mix!

# ✓ Consistent
uv pip install textual rich
```

### Problem 3: No Entry Point
```python
# ❌ Users can't run easily
# They have to do: python -m src.my_app

# ✓ Define entry point in pyproject.toml
[project.scripts]
my-app = "my_app:run"

# Now users can just: my-app
```

### Problem 4: Circular Imports
```python
# ❌ Circular: app.py imports widgets.py imports app.py
# widgets.py
from my_app.app import MyApp

# ✓ Break circular dependency
# widgets.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_app.app import MyApp
```

---

## Testing Project Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_app.py
├── test_screens/
│   ├── __init__.py
│   ├── test_home.py
│   └── test_settings.py
├── test_widgets/
│   ├── __init__.py
│   └── test_custom_button.py
└── fixtures/
    ├── __init__.py
    └── sample_data.py
```

### conftest.py

```python
"""Shared test fixtures."""

import pytest
from textual.pilot import Pilot

from my_tui_app.app import MyApp
from my_tui_app.config import Config


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return Config(debug=True)


@pytest.fixture
async def app(test_config):
    """Provide test app."""
    app = MyApp(config=test_config)
    async with app.run_test() as pilot:
        yield pilot
```

---

## Sources

**Referenced From**:
- [Textual: The Definitive Guide - Part 1](https://dev.to/wiseai/textual-the-definitive-guide-part-1-1i0p) - Mahmoud Harmouch (DEV.to, April 2022)
  - Project structure and organization
  - Dependency management with poetry (now uv preferred)
  - Development workflow

- [My Python TUI Project Template](https://codecurrents.blog/article/2024-10-14) - Michel Lavoie (CodeCurrents, October 2024)
  - Modern uv-based setup
  - Production build configuration
  - Nuitka compilation

**Textual Official Resources**:
- [Textual Project Setup](https://textual.textualize.io/guide/devtools/)
- [Python Packaging Guide](https://packaging.python.org/)

---

## Related Documentation

- [00-long-running-processes.md](./00-long-running-processes.md) - Worker patterns
- [02-lessons-learned.md](../best-practices/02-lessons-learned.md) - Production insights
- [widgets/00-widget-patterns.md](../widgets/00-widget-patterns.md) - Custom widget organization
