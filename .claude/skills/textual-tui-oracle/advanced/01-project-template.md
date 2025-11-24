# Python TUI Project Template (CodeCurrents Blog)

## Source Information

**Original Article**: [My Python TUI Project Template](https://codecurrents.blog/article/2024-10-14)
**Author**: Michel Lavoie
**Date**: October 14, 2024
**Accessed**: 2025-11-02

**Note**: This document is based on search results and article metadata. Direct scraping of the full article encountered technical limitations (27k+ token HTML response exceeding MCP tool limits). The article covers professional Textual project setup using modern Python tooling.

## Overview

Michel Lavoie's approach to starting new Python TUI projects using three key tools:

1. **uv** - Fast Python package installer and resolver
2. **Textual** - Modern TUI framework
3. **Nuitka** - Python-to-C compiler for binary distribution

## Key Topics Covered

Based on article description and search metadata:

### Tooling Stack

**uv (Package Management)**:
- Modern alternative to pip/poetry for faster dependency resolution
- Project initialization and virtual environment management
- Lock file generation for reproducible builds

**Textual (TUI Framework)**:
- Declarative UI development
- Reactive programming patterns
- CSS-like styling for terminal interfaces

**Nuitka (Binary Compilation)**:
- Compile Python TUI apps to standalone binaries
- Distribution without Python interpreter dependency
- Performance optimization through C compilation

### Project Structure

Professional Textual projects typically include:

```
my-tui-project/
├── pyproject.toml          # Project metadata and dependencies
├── src/
│   └── my_app/
│       ├── __init__.py
│       ├── app.py          # Main Textual application
│       └── components/     # Custom widgets
├── tests/                  # Test suite
├── README.md
└── .gitignore
```

### Development Workflow

1. **Initialization**: Use `uv` to create project structure
2. **Development**: Build Textual UI with hot reload
3. **Testing**: pytest for unit/integration tests
4. **Distribution**: Nuitka compilation for end users

## Modern Python Project Best Practices

### Configuration (pyproject.toml)

Modern Python projects consolidate configuration in `pyproject.toml`:

```toml
[project]
name = "my-tui-app"
version = "0.1.0"
description = "Professional TUI application"
requires-python = ">=3.10"
dependencies = [
    "textual>=0.40.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "mypy>=1.0",
]

[tool.black]
line-length = 88

[tool.mypy]
strict = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Code Quality Tools

**Black** - Automatic code formatting:
```bash
black src/ tests/
```

**Mypy** - Static type checking:
```bash
mypy src/
```

**Pytest** - Testing framework:
```bash
pytest tests/ -v
```

## Why This Stack?

### uv Advantages
- 10-100x faster than pip for dependency resolution
- Built in Rust for performance
- Compatible with existing Python packaging standards
- Single tool for project management

### Textual Benefits
- Rich terminal UI with minimal code
- Reactive data binding
- Built-in widgets (DataTable, Tree, Input, etc.)
- CSS-like styling language (TCSS)
- Cross-platform (Windows, macOS, Linux)

### Nuitka Value
- Distribute Python apps without Python installation
- Single executable for end users
- Performance improvements through compilation
- Protect source code (compiled to C)

## Professional Project Checklist

- [ ] **pyproject.toml** with complete metadata
- [ ] **Type hints** throughout codebase
- [ ] **Black** for consistent formatting
- [ ] **Mypy** for type safety
- [ ] **Pytest** for comprehensive testing
- [ ] **README.md** with setup instructions
- [ ] **CI/CD** pipeline (GitHub Actions, etc.)
- [ ] **License** file (MIT, Apache, etc.)
- [ ] **.gitignore** for Python projects
- [ ] **Documentation** (docstrings, wiki, or ReadTheDocs)

## Related Resources

### uv Documentation
- Official site: https://github.com/astral-sh/uv
- Fast Python package management and environment handling

### Textual Resources
- See `../getting-started/01-official-tutorial.md` for complete framework walkthrough
- See `../getting-started/00-official-homepage.md` for core concepts

### Nuitka
- Official site: https://nuitka.net/
- Python compiler documentation

### Alternative Project Templates
- **Poetry**: Traditional dependency management with pyproject.toml
- **PDM**: PEP 582 compliant package manager
- **Hatch**: Modern build system with environment management

## Example: Minimal Textual App Structure

```python
# src/my_app/app.py
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static

class MyTUIApp(App):
    """Professional TUI application template."""

    CSS = """
    Screen {
        align: center middle;
    }

    Static {
        width: 60;
        height: 10;
        border: solid green;
        content-align: center middle;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Hello from Textual!")
        yield Footer()

def main() -> None:
    app = MyTUIApp()
    app.run()

if __name__ == "__main__":
    main()
```

### pyproject.toml Entry Point

```toml
[project.scripts]
my-app = "my_app.app:main"
```

After installation with `uv sync`, run via:
```bash
my-app
```

## Setup Workflow (Typical)

```bash
# 1. Create project with uv
uv init my-tui-project
cd my-tui-project

# 2. Add Textual dependency
uv add textual

# 3. Add dev dependencies
uv add --dev pytest black mypy

# 4. Create source structure
mkdir -p src/my_app
touch src/my_app/__init__.py
touch src/my_app/app.py

# 5. Run in development
uv run python -m my_app.app

# 6. Run tests
uv run pytest

# 7. Format code
uv run black src/

# 8. Type check
uv run mypy src/

# 9. Build with Nuitka (for distribution)
uv run python -m nuitka --standalone src/my_app/app.py
```

## Best Practices Summary

### For Professional Textual Projects

1. **Use modern tooling** (uv, not outdated pip workflows)
2. **Type everything** (mypy strict mode)
3. **Test thoroughly** (pytest with good coverage)
4. **Format consistently** (black, no debates)
5. **Document clearly** (docstrings, README, examples)
6. **Version properly** (semantic versioning)
7. **Distribute easily** (Nuitka for binaries, PyPI for libraries)

### Development Environment

```bash
# .env or .envrc for direnv
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

### Git Ignore Patterns

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# uv
.venv/
uv.lock

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/

# Type checking
.mypy_cache/
.dmypy.json
dmypy.json

# Nuitka
*.build/
*.dist/
*.onefile-build/
```

## Additional Considerations

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --all-extras
      - name: Run tests
        run: uv run pytest
      - name: Type check
        run: uv run mypy src/
      - name: Format check
        run: uv run black --check src/
```

### Performance Tips

1. **Lazy loading**: Import Textual components only when needed
2. **Async workers**: Use workers for CPU-intensive tasks
3. **Reactive design**: Minimize unnecessary re-renders
4. **Profiling**: Use `textual console` for debugging

## Cross-Reference

- **Official Tutorial**: See `../getting-started/01-official-tutorial.md` for Textual basics
- **ChatGPT Integration**: See `00-chatgpt-integration.md` for async API patterns
- **GitHub Examples**: See `../examples/00-github-examples-index.md` for real projects

## Limitations of This Document

**Access Issues**: The original article could not be fully scraped due to technical limitations:
- HTML response exceeded 25,000 token limit for MCP scraping tool
- Connection errors on markdown scraping attempts
- This document reconstructs expected content from search results and Python TUI best practices

**For Complete Information**: Visit the original article directly at https://codecurrents.blog/article/2024-10-14

## Summary

Michel Lavoie's Python TUI project template emphasizes:
- **Modern tooling** (uv > pip, Nuitka for distribution)
- **Professional structure** (src layout, pyproject.toml, type hints)
- **Quality tooling** (black, mypy, pytest from the start)
- **Easy distribution** (compile to standalone binaries)

This approach creates maintainable, professional Textual applications ready for both development and end-user distribution.

---

**Document Status**: Partial reconstruction from search results and metadata
**Original Source**: https://codecurrents.blog/article/2024-10-14 (access date: 2025-11-02)
**Author**: Michel Lavoie
**Topic**: Professional Python TUI project setup with uv, Textual, and Nuitka
