# Packaging Textual Applications

## Overview

Packaging Textual TUI applications into standalone executables allows distribution to end-users without requiring Python installation or dependency management. This document covers proven tools and techniques for creating "1-click" executables from Textual applications.

## Packaging Goals

**Ideal Executable Characteristics:**
- **Single file**: One executable, no separate dependencies
- **No Python required**: End-user doesn't need Python installed
- **No pip install**: All dependencies bundled
- **Cross-platform**: Support for Windows, macOS, Linux
- **Include assets**: Bundle CSS files, images, data files

## Packaging Tools Comparison

### 1. Nuitka (Recommended)

**Status**: ✅ Works out of the box with Textual

From [GitHub Discussion #4512](https://github.com/Textualize/textual/discussions/4512) (accessed 2025-11-02):

> "Another option here would be Nuitka, which can accomplish the same thing using it's `--onefile` mode, it should work out of the box" - KRRT7 (Jun 16, 2024)

**Community Validation**:
- "Thanks man, this worked for me as well" - Devanshi-Crypto (Jan 6, 2025)
- Marked as accepted answer by original poster

**Basic Usage**:
```bash
nuitka --onefile main.py
```

**Advantages**:
- ✅ Works with Textual without special configuration
- ✅ Creates single executable file
- ✅ Better performance than PyInstaller (compiles to C)
- ✅ Cross-platform support

**Platform-Specific Notes**:

**macOS**:
```bash
# For terminal binary
nuitka --onefile main.py

# For .app bundle (experimental)
nuitka --mode=app main.py
nuitka --macos-create-app-bundle main.py
```

**Note on macOS .app bundles**: As of Jun 2025, there are ongoing discussions about creating clickable `.app` bundles that properly open a terminal for TUI apps. The `.bin` output works reliably via terminal, but `.app` integration is still being refined.

**Advanced Options**:
```bash
# With icon
nuitka --onefile --windows-icon=icon.ico main.py

# With company info (Windows)
nuitka --onefile \
       --windows-company-name="MyCompany" \
       --windows-product-name="MyApp" \
       main.py

# Optimize for size
nuitka --onefile --remove-output main.py
```

### 2. PyInstaller

**Status**: ⚠️ Works with manual configuration

**Basic Usage**:
```bash
pyinstaller --onefile --name "MyApp" main.py
```

**With Textual Assets**:
```bash
pyinstaller --onefile \
            --add-data 'app/style.tcss:app/' \
            --name "TUI v0.1.0" \
            main.py
```

**Known Issues and Solutions**:

From jamesreverett (Jun 12, 2024):

> "pyinstaller was missing a Textualize module... ModuleNotFoundError: No module named 'textual.widgets._tab_pane'"

**Solution**: Add hidden imports to PyInstaller spec file

**Using .spec file**:
```python
# myapp.spec
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('app/style.tcss', 'app/')],
    hiddenimports=['textual.widgets._tab_pane'],  # Add missing modules
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MyApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
```

**Build with spec**:
```bash
pyinstaller myapp.spec
```

**Advantages**:
- Mature, widely used
- Good documentation
- Supports many Python libraries

**Disadvantages**:
- May require manual hidden imports for Textual widgets
- Larger executable size than Nuitka
- Slower startup time

### 3. Shiv (Zipapp with Dependencies)

**Status**: ✅ Lightweight alternative

From dontascii (Sep 22, 2024):

> "I've had great success using Shiv, from the LinkedIn devs. It works just like zip apps do (PEP 441) but it includes all your dependencies, like PEX. It performs better than PEX, however"

**Installation**:
```bash
pip install shiv
```

**Basic Usage**:
```bash
shiv -c myapp -o myapp.pyz mypackage
```

**With reproducible builds**:
```bash
shiv -c myapp -o myapp.pyz --reproducible mypackage
```

**With change detection**:
```bash
shiv -c myapp -o myapp.pyz --no-modify mypackage
```

**Features**:
- `--reproducible`: Generate reproducible zipapp (same hash every time)
- `--no-modify`: Detect changes to unzipped source files and throw exception
- Better performance than PEX
- Follows [PEP 441](http://legacy.python.org/dev/peps/pep-0441/) zipapp standard

**Advantages**:
- ✅ Lightweight (Python zipapp format)
- ✅ Fast execution
- ✅ Reproducible builds
- ✅ Still requires Python on target system (but manages dependencies)

**Disadvantages**:
- Requires Python installed on end-user machine
- Not a true "standalone" executable

**Best For**:
- Internal tools where Python is available
- Development/testing distributions
- CI/CD environments

## Project Structure for Packaging

```
my-textual-app/
├── requirements.txt
├── main.py
├── app/
│   ├── __init__.py
│   ├── screens.py
│   ├── widgets.py
│   └── style.tcss
├── assets/
│   ├── logo.png
│   └── icon.ico
└── build/
    └── myapp.spec  # PyInstaller spec
```

## Including Assets

### CSS Files (TCSS)

**PyInstaller**:
```bash
pyinstaller --add-data 'app/style.tcss:app/' main.py
```

**Nuitka**:
```bash
nuitka --onefile --include-data-file=app/style.tcss=app/style.tcss main.py
```

**In code** (accessing bundled files):
```python
import sys
from pathlib import Path

def get_asset_path(relative_path):
    """Get absolute path to resource (works for dev and packaged app)."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = Path(sys._MEIPASS)  # PyInstaller
    else:
        # Running in development
        base_path = Path(__file__).parent

    return base_path / relative_path

# Usage
CSS_PATH = get_asset_path("app/style.tcss")
```

### Data Files and Images

**Pattern**:
```bash
# PyInstaller
--add-data 'assets/*:assets/'

# Nuitka
--include-data-dir=assets=assets
```

## Common Issues and Solutions

### Issue 1: Missing Textual Widgets

**Error**:
```
ModuleNotFoundError: No module named 'textual.widgets._tab_pane'
```

**Solution**:
Add to PyInstaller hiddenimports:
```python
hiddenimports=['textual.widgets._tab_pane']
```

### Issue 2: CSS Files Not Found

**Error**:
```
FileNotFoundError: style.tcss not found
```

**Solution**:
1. Use `--add-data` to include CSS
2. Update code to use `get_asset_path()` helper (see above)

### Issue 3: Large Executable Size

**Solutions**:
- Use Nuitka (produces smaller executables than PyInstaller)
- Exclude unnecessary dependencies
- Use UPX compression (PyInstaller: `--upx`)
- Strip debug symbols (Nuitka: `--remove-output`)

### Issue 4: Slow Startup

**Nuitka optimization**:
```bash
nuitka --onefile --standalone --remove-output main.py
```

**PyInstaller optimization**:
```bash
pyinstaller --onefile --strip main.py
```

## Build Automation

### Using Poetry + Invoke

From jamesreverett's comment:

> "I'm using poetry and invoke for build infrastructure"

**pyproject.toml**:
```toml
[tool.poetry]
name = "my-textual-app"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.10"
textual = "^0.67.0"

[tool.poetry.dev-dependencies]
nuitka = "^2.0"
invoke = "^2.0"
```

**tasks.py** (invoke):
```python
from invoke import task

@task
def build(c):
    """Build standalone executable with Nuitka."""
    c.run("nuitka --onefile --remove-output main.py")

@task
def build_windows(c):
    """Build Windows executable with icon."""
    c.run("nuitka --onefile --windows-icon=icon.ico main.py")

@task
def build_all(c):
    """Build for all platforms."""
    build(c)
    # Add platform-specific builds
```

**Usage**:
```bash
invoke build
invoke build-windows
```

## Distribution Checklist

- [ ] Test executable on clean system (no Python installed)
- [ ] Verify all assets (CSS, images) are bundled
- [ ] Check executable size (optimize if > 50MB)
- [ ] Test on target OS (Windows/macOS/Linux)
- [ ] Include README with usage instructions
- [ ] Verify no hardcoded paths
- [ ] Test with antivirus (PyInstaller can trigger false positives)
- [ ] Consider code signing for macOS/Windows

## Recommendations by Use Case

### Internal Tools
**Recommended**: Shiv
- Fast build times
- Small package size
- Python expected in environment

### End-User Applications
**Recommended**: Nuitka
- True standalone executable
- No Python required
- Best performance

### Quick Prototypes
**Recommended**: PyInstaller
- Fast iteration
- Well-documented
- Good for testing packaging workflow

### Cross-Platform Distribution
**Recommended**: Nuitka or PyInstaller
- Build on each platform
- Use CI/CD for automated builds (GitHub Actions)

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Executables

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    runs-on: ${{ matrix.os }}

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install nuitka

    - name: Build with Nuitka
      run: nuitka --onefile --remove-output main.py

    - name: Upload artifact
      uses: actions/upload-artifact@v3
      with:
        name: myapp-${{ matrix.os }}
        path: main.bin  # or main.exe on Windows
```

## Sources

**GitHub Discussions:**
- [Discussion #4512: "Packaging my text-user-interface and its requirements.txt libraries into a '1-click' executable"](https://github.com/Textualize/textual/discussions/4512) - GitHub (accessed 2025-11-02)
  - Original question by ishaan-ghosh (May 14, 2024)
  - PyInstaller hidden imports solution by jamesreverett (Jun 12, 2024)
  - Nuitka recommendation by KRRT7 (Jun 16, 2024) - **Accepted Answer**
  - Nuitka validation by Devanshi-Crypto (Jan 6, 2025)
  - macOS .app discussion by moonblind & KRRT7 (Jun 2025)
  - Shiv recommendation by dontascii (Sep 22, 2024)

**Tools Documentation:**
- [Nuitka](https://nuitka.net/) - Python compiler
- [PyInstaller](https://pyinstaller.org/) - Freeze Python applications
- [Shiv](https://shiv.readthedocs.io/) - Zipapp builder
- [PEP 441](http://legacy.python.org/dev/peps/pep-0441/) - Python zipapp standard

**Related Resources:**
- Textual documentation on [deployment](https://textual.textualize.io/)
- Poetry for [dependency management](https://python-poetry.org/)
- Invoke for [task automation](https://www.pyinvoke.org/)
