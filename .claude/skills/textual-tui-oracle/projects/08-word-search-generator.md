# word-search-generator - Word Search Puzzle Generator

**Source**: [GitHub - joshbduncan/word-search-generator](https://github.com/joshbduncan/word-search-generator)
**Accessed**: 2025-11-02
**Version**: 5.0.0
**License**: MIT

## Overview

Word-Search-Generator is a Python package for generating customizable Word Search puzzles. While not a Textual TUI, it provides an excellent example of CLI-based puzzle generation with rich terminal output and demonstrates patterns useful for interactive puzzle games.

**Key Quote from Author**:
> 🤦‍♂️ Does the world need this? Probably not.
> ⏰ Did I spend way too much time on it? Yep!
> ✅ Does it come in handy? Heck yeah!
> 👍 Did I have fun making it? Absofreakinglutly!

## Core Features

### Puzzle Generation
- **Dynamic puzzle sizing**: Automatically sizes grid based on word list
- **Difficulty levels**: 3 built-in levels controlling word directions
- **Custom directions**: Specify allowed cardinal directions (N, NE, E, SE, S, SW, W, NW)
- **Random word lists**: Built-in word lists for quick puzzle generation
- **Secret mode**: Hide word list for harder gameplay
- **Mask support**: Shape-based puzzle masking (e.g., heart shapes)

### Word Placement Algorithm
- **Smart placement**: Words placed in specified directions only
- **Fill characters**: Random letters fill empty spaces
- **No overlap conflicts**: Ensures clean word boundaries
- **Answer key generation**: Provides coordinates and directions for all words

### Export Options
- **PDF generation**: Save puzzles as PDF files via fpdf2
- **Terminal display**: Rich-formatted console output
- **Answer keys**: 1-based indexing with (x, y) coordinates and cardinal directions

## Installation

```bash
# Using uv (recommended)
uv add word-search-generator  # For Python projects
uv tool install word-search-generator  # For CLI use

# Using pip
python -m pip install word-search-generator
```

## Dependencies

From [pyproject.toml](https://github.com/joshbduncan/word-search-generator/blob/main/pyproject.toml):

```toml
dependencies = [
  "fpdf2==2.8.4",      # PDF generation
  "Pillow==12.0.0",    # Image processing for masks
  "rich==14.2.0",      # Terminal formatting
  "ordered-set>=4.0.0", # Ordered collections
]
```

**Python Version**: 3.11+

## API Usage Examples

### Basic Puzzle Creation

From [README.md](https://github.com/joshbduncan/word-search-generator/blob/main/README.md):

```python
from word_search_generator import WordSearch

# Create puzzle from word list
puzzle = WordSearch("dog, cat, pig, horse, donkey, turtle, goat, sheep")

# Display puzzle
puzzle.show()  # or print(puzzle)
```

### Difficulty Levels

```python
# Level 1: Easy (fewer directions)
puzzle.level = 1

# Level 3: Hard (all 8 directions)
puzzle.level = 3
puzzle.show()
```

### Custom Directions

```python
# Only allow diagonal words
puzzle.directions = "NW,SW"
puzzle.show()
```

### Expert Mode with Random Words

```python
puzzle = WordSearch(level=3)
puzzle.random_words(10, secret=True, reset_size=True)
```

### Masked Puzzles (Shapes)

```python
from word_search_generator.mask.shapes import Heart

puzzle.apply_mask(Heart())
puzzle.show()
```

**Example Output**:
```
-------------------------
       WORD SEARCH
-------------------------
    C L C       C Y E
  Q S T N S   T E L D K
Z L O Z T P A K T H N M W
C C L Q C O N R S P Z V U
M I X V G O U R H S Z C H
  M H K D T T L E C U Q
  H C T H O R S E B M
    Z I S W R B P E G
      Y P I G B X Q
        C A T N G
        A R W O
          K D Q
            A
```

### PDF Export

```python
puzzle.save(path="~/Desktop/puzzle.pdf")
# Returns: "~/Desktop/puzzle.pdf"
```

## CLI Integration

From [README.md](https://github.com/joshbduncan/word-search-generator/blob/main/README.md):

```bash
# Generate random 10-word puzzle, size 15x15, difficulty level 3
$ word-search -r 10 -s 15 -l 3
```

**CLI Entry Point** (from pyproject.toml):
```toml
[project.scripts]
word-search = "word_search_generator.cli:main"
```

## Answer Key Format

Example answer key from README:
```
Answer Key: CAT NE @ (5, 14), DOG NE @ (7, 13), DONKEY S @ (10, 6),
            GOAT S @ (11, 7), HORSE NE @ (7, 7), PIG S @ (4, 7),
            SHEEP NE @ (2, 7), TURTLE E @ (3, 11)
```

**Key Format**:
- 1-based indexing for user display
- (x, y) coordinate system
- Cardinal directions from first letter to last
- **Note**: Internal API uses 0-based indexing

## Project Structure Highlights

From [GitHub repository](https://github.com/joshbduncan/word-search-generator):

```
word-search-generator/
├── src/word_search_generator/
│   ├── __init__.py           # Main exports: WordSearch, WORD_LISTS
│   ├── word_search/          # Core puzzle logic
│   ├── mask/                 # Shape masking system
│   │   └── shapes.py         # Built-in shapes (Heart, etc.)
│   ├── words.py              # Word list management
│   └── cli/                  # CLI interface
├── tests/                    # Comprehensive test suite
├── pyproject.toml            # Modern Python packaging
└── README.md                 # Extensive documentation
```

## Package Organization

From [__init__.py](https://github.com/joshbduncan/word-search-generator/blob/main/src/word_search_generator/__init__.py):

```python
__all__ = [
    "__version__",
    "WORD_LISTS",
    "WordSearch",
]

from rich.traceback import install
from .word_search.word_search import WordSearch
from .words import WORD_LISTS

install(show_locals=True)  # Enhanced error tracebacks
```

**Key Exports**:
- `WordSearch` - Main puzzle class
- `WORD_LISTS` - Built-in word collections
- Rich traceback integration for better debugging

## Patterns for TUI Development

### 1. Rich Integration
Uses `rich` library for terminal formatting (v14.2.0)
- Enhanced tracebacks with `show_locals=True`
- Formatted puzzle display
- Color-coded output

### 2. CLI Design
- Simple entry point via `word-search` command
- Flag-based configuration (`-r`, `-s`, `-l`)
- Immediate visual output

### 3. Puzzle State Management
- Mutable puzzle properties (`level`, `directions`)
- Dynamic regeneration on property changes
- Separation of puzzle data from display

### 4. Masking System
- Pluggable shape system
- Custom mask support
- Visual puzzle variations

## Testing Infrastructure

From pyproject.toml:
```toml
test = [
  "pdfplumber",      # PDF validation
  "pypdf",           # PDF parsing
  "pytest-asyncio",  # Async testing
  "pytest-cov",      # Coverage reports
  "pytest-repeat",   # Repeated test runs
  "pytest",          # Main framework
]
```

## Build System

Modern Python tooling:
- **Package manager**: uv (Astral's fast Python package manager)
- **Build backend**: uv_build
- **Linting**: ruff
- **Type checking**: mypy
- **Testing**: pytest with coverage
- **Pre-commit**: Automated code quality checks

## Key Learnings for TUI Games

### Game Logic Patterns
1. **State separation**: Game state separate from display logic
2. **Answer tracking**: Comprehensive answer key system with coordinates
3. **Progressive difficulty**: Built-in difficulty levels with custom override
4. **Random generation**: Support for procedural content

### CLI/TUI Considerations
1. **Rich output**: Use Rich library for terminal formatting
2. **Immediate feedback**: Show puzzle immediately after generation
3. **Export options**: Multiple output formats (terminal, PDF)
4. **Error handling**: Enhanced tracebacks for development

### Algorithm Insights
1. **Grid sizing**: Dynamic sizing based on word count
2. **Word placement**: Backtracking algorithm for valid placements
3. **Direction constraints**: Configurable cardinal directions
4. **Fill strategy**: Random character filling with no conflicts

## Documentation

From [README.md](https://github.com/joshbduncan/word-search-generator/blob/main/README.md):

> If you dig deep enough, you'll notice this package is completely overkill...
> There are numerous options and lots of possibilities.

**Documentation Sources**:
- [Project Wiki](https://github.com/joshbduncan/word-search-generator/wiki) - Comprehensive guides
- [Puzzle Masking docs](https://github.com/joshbduncan/word-search-generator/wiki/Puzzle-Masking)
- [CLI docs](https://github.com/joshbduncan/word-search-generator/wiki/Command-Line-Interface-(CLI))

## Stats

- **Stars**: 93
- **Forks**: 33
- **Latest Release**: v5.0.0 (Oct 16, 2025)
- **Active Development**: Regular updates and maintenance

## Relevance to Textual TUI Development

While not a Textual application, word-search-generator demonstrates:

1. **Puzzle game architecture**: Clean separation of logic, display, and state
2. **Rich terminal output**: Professional-looking CLI output
3. **Progressive complexity**: Multiple difficulty modes
4. **Export flexibility**: Terminal display + file export
5. **Modern Python tooling**: uv, ruff, mypy, pytest
6. **User experience**: Balance of simplicity and power

**Potential Textual Integration**:
- Could be wrapped in Textual TUI for interactive puzzle solving
- Game state management patterns applicable to TUI games
- Answer key tracking useful for hint systems
- Masking system could inspire visual variations

## Sources

**GitHub Repository**:
- [Main Repository](https://github.com/joshbduncan/word-search-generator)
- [README.md](https://github.com/joshbduncan/word-search-generator/blob/main/README.md)
- [pyproject.toml](https://github.com/joshbduncan/word-search-generator/blob/main/pyproject.toml)
- [__init__.py](https://github.com/joshbduncan/word-search-generator/blob/main/src/word_search_generator/__init__.py)

**Documentation**:
- [Project Wiki](https://github.com/joshbduncan/word-search-generator/wiki)
- [Changelog](https://github.com/joshbduncan/word-search-generator/blob/main/CHANGLOG.md)

**Package**:
- [PyPI Package](https://pypi.org/project/word-search-generator/)

All content accessed: 2025-11-02
