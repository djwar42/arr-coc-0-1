# uproot-browser - Particle Physics ROOT File Browser

**Source**: [scikit-hep/uproot-browser](https://github.com/scikit-hep/uproot-browser)
**Accessed**: 2025-11-02
**Category**: Scientific Data Visualization | Developer Tools
**Project Type**: Official Scikit-HEP Project
**License**: BSD 3-Clause

---

## Overview

uproot-browser is a terminal-based browser for ROOT files used in particle physics research. It provides a complete TUI (Text User Interface) for exploring and visualizing ROOT file contents without requiring a graphical environment - perfect for remote HPC clusters and SSH sessions.

**Built with**:
- **Textual** - TUI framework
- **Click** - CLI interface
- **plotext** - Terminal plotting
- **Hist** - Histogram handling
- **Part of**: [Scikit-HEP](https://scikit-hep.org/) (Scientific Python ecosystem for High Energy Physics)

**Inspiration**: ROOT object browser (official C++ tool), reimagined for terminal environments.

---

## Key Features

### 1. Three Primary Commands

**`browse`** (default):
- Interactive TUI for navigating ROOT file structure
- Real-time exploration of nested objects
- Visual tree representation with emoji indicators

**`plot`**:
- Display histograms directly in terminal
- plotext-based ASCII plots
- Optional iTerm2 graphics support (macOS)

**`tree`**:
- Print complete file structure
- Shows all branches, leaves, types, and sizes
- Emoji-coded object types (📁 files, 🌴 trees, 🍁 leaves, 📊 histograms)

### 2. Scientific Data Visualization

- Browse complex nested physics data structures
- View histogram statistics (entries, bins, distributions)
- Navigate TClonesArray, TRef, and other ROOT-specific types
- Supports Float16_t, Double32_t, and various ROOT data types

### 3. Designed for Remote Work

- No X11/GUI required
- Works over SSH
- Minimal dependencies for HPC environments
- Fast terminal rendering for large datasets

---

## Installation

### Standard Installation (PyPI)

```bash
python3 -m pip install uproot-browser
```

### With Test Data

```bash
python3 -m pip install uproot-browser[testdata]
```

### iTerm2 Graphics Support (macOS)

```bash
python3 -m pip install uproot-browser[iterm]
```

### Run Without Installing (pipx/uvx)

```bash
# Try it in one line
uvx uproot-browser[testdata] --testdata uproot-Event.root

# Or with pipx
pipx run uproot-browser[iterm]
```

### Conda-Forge

```bash
conda install -c conda-forge uproot-browser
```

---

## Usage Examples

### Browse Command (Interactive TUI)

```bash
uproot-browser browse --testdata uproot-Event.root
```

**Features**:
- Navigate nested ROOT structures interactively
- Keyboard-driven interface
- Real-time data exploration
- Textual-powered responsive UI

### Plot Command (Terminal Histograms)

```bash
uproot-browser plot --testdata uproot-Event.root:hstat
```

**Output**:
```
                        hstat -- Entries: 1000
    ┌───────────────────────────────────────────────────────────────┐
18.0┤▐▌                                                             │
    │▐▌                                                 ▗▖         ▄│
15.6┤▐▌▗▖                                               ▐▌         █│
    │███▌               █                           █   ▐▌        ▐█│
13.1┤████▟▌    ▗▖  ▗▖   █▌▗▖ ▐▌       ▄   █▌   ▄  ▟▌█ ▗▄▐▙▗▖    ▐▌▐█│
10.6┤█████▌    ▐▌  ▐▙▖  █▌▐▌ ▐▙       █▄  █▙   █  █▌█ ▐█▟█▐▌  ▗▄▟▌▐█│
    │█████▌ █▌▐█▌  ████▌█▌▐█ ▐█▐▌ ▐▌  ███▐██  ▐█ ▐████▐███▐▌ ▐███▌▐█│
 8.2┤█████▌▐█▌▐█▌ █████▌██▐█ ██▐█ ▐▌▐████▐███▌▐█ █████▐███▐██▐████▐█│
    │████████▙██▌█████████▟█▖████▖▟██████▟██████▖████████████▟██████│
 5.8┤███████████▙███████████▙████▌██████████████▌███████████████████│
    │████████████████████████████▌██████████████████████████████████│
 3.3┤███████████████████████████████████████████████████████████████│
    └┬───────────────┬──────────────┬───────────────┬──────────────┬┘
     0.00          0.25           0.50            0.75          1.00
                               [x] xaxis
```

**iTerm2 Enhanced Graphics**:
```bash
uproot-browser plot --testdata uproot-Event.root:hstat --iterm
```

### Tree Command (Structure Inspection)

```bash
uproot-browser tree --testdata uproot-Event.root
```

**Output**:
```
📁 uproot-Event.root
┣━━ ❓ <unnamed> TProcessID
┣━━ 🌴 T (1000)
┃   ┗━━ 🌿 event Event
┃       ┣━━ 🌿 TObject (group of fUniqueID:uint32_t, fBits:uint32_t)
┃       ┃   ┣━━ 🍁 fBits uint32_t
┃       ┃   ┗━━ 🍁 fUniqueID uint32_t
┃       ┣━━ 🍁 fClosestDistance unknown[]
┃       ┣━━ 🍁 fEventName char*
┃       ┣━━ 🌿 fEvtHdr EventHeader
┃       ┃   ┣━━ 🍁 fEvtHdr.fDate int32_t
┃       ┃   ┣━━ 🍁 fEvtHdr.fEvtNum int32_t
┃       ┃   ┗━━ 🍁 fEvtHdr.fRun int32_t
┃       ┣━━ 🍁 fFlag uint32_t
┃       ┣━━ 🍁 fH TH1F
┃       ┣━━ 🍁 fHighPt TRefArray*
┃       ┣━━ 🍁 fIsValid bool
┃       ┣━━ 🍁 fLastTrack TRef
┃       ┣━━ 🍁 fMatrix[4][4] float[4][4]
┃       ┣━━ 🌿 fTracks TClonesArray*
┃       ┃   ┣━━ 🍃 fTracks.fBits uint32_t[]
┃       ┃   ┣━━ 🍃 fTracks.fBx Float16_t[]
┃       ┃   ┣━━ 🍃 fTracks.fBy Float16_t[]
┃       ┃   ┣━━ 🍃 fTracks.fCharge Double32_t[]
┃       ┃   ┣━━ 🍃 fTracks.fMass2 Float16_t[]
┃       ┃   ┣━━ 🍃 fTracks.fPx float[]
┃       ┃   ┣━━ 🍃 fTracks.fPy float[]
┃       ┃   ┗━━ 🍃 fTracks.fPz float[]
┃       ┗━━ 🌿 fTriggerBits TBits
┣━━ 📊 hstat TH1F (100)
┗━━ 📊 htime TH1F (10)
```

**Emoji Indicators**:
- 📁 ROOT files
- 🌴 TTrees (event data structures)
- 🌿 Branches (data groups)
- 🍁 Leaves (individual data fields)
- 🍃 Array leaves
- 📊 Histograms
- ❓ Unknown/TProcessID objects

---

## Architecture & Design Patterns

### Scientific Data Integration

**ROOT File Format**:
- High Energy Physics standard format (CERN)
- Stores event data, histograms, trees, branches
- Complex nested structures with references (TRef, TRefArray)
- Custom compression and type systems

**Integration Strategy**:
- Uses uproot library (pure Python ROOT I/O)
- Hist library for histogram manipulation
- Type-aware navigation (Float16_t, Double32_t, TClonesArray)

### CLI Architecture

**Click Framework**:
- Subcommands: `browse`, `plot`, `tree`
- Common flags: `-h/--help`, `--version`, `--testdata`
- Extensible command structure

**Example Command Structure**:
```python
# Pseudocode - actual implementation pattern
@click.group()
@click.version_option()
def cli():
    pass

@cli.command()
@click.argument('filename')
def browse(filename):
    # Launch Textual TUI
    pass

@cli.command()
@click.argument('path')  # file:histogram
def plot(path):
    # plotext rendering
    pass

@cli.command()
@click.argument('filename')
def tree(filename):
    # Print tree structure
    pass
```

### TUI Patterns (Textual)

**Interactive Navigation**:
- Tree widgets for ROOT file structure
- Keyboard bindings for navigation
- Real-time data loading
- Lazy loading for large datasets

**Visual Design**:
- Unicode box-drawing characters
- Emoji-based type indicators
- Color-coded data types
- Responsive layout

### Terminal Plotting

**plotext Integration**:
- ASCII histogram rendering
- Automatic scaling and binning
- Statistics display (entries, mean, std dev)
- X/Y axis labels and grid

**iTerm2 Enhancement**:
- Optional high-quality graphics
- Image protocol support
- Fallback to ASCII for compatibility

---

## Use Cases

### 1. Remote HPC Analysis

**Scenario**: Analyzing particle physics data on computing clusters

```bash
# SSH into cluster
ssh user@cluster.cern.ch

# Browse collision events
uproot-browser browse /data/lhc/run_2024/collision_events.root

# Quick histogram check
uproot-browser plot /data/lhc/run_2024/collision_events.root:energy_distribution
```

### 2. Quick Data Inspection

**Scenario**: Verify ROOT file contents before processing

```bash
# Check file structure
uproot-browser tree experiment_data.root

# Validate histogram
uproot-browser plot experiment_data.root:pt_distribution
```

### 3. Teaching & Demonstrations

**Scenario**: Teaching ROOT file format to students

```bash
# Use test data for learning
uvx uproot-browser[testdata] --testdata uproot-Event.root

# Students can explore interactively without ROOT installation
```

---

## Textual Integration Patterns

### 1. Scientific Data Widgets

**Lessons**: Textual can handle:
- Complex nested tree structures (particle physics events)
- Real-time data visualization (histograms, statistics)
- Type-aware rendering (Float16_t, Double32_t display)
- Large dataset lazy loading

### 2. CLI + TUI Hybrid

**Pattern**:
- Click for argument parsing and subcommands
- Textual for interactive `browse` mode
- plotext for non-interactive `plot` output
- Pure text for `tree` command

**Benefit**: Users can choose interaction level (CLI one-shot vs TUI exploration)

### 3. Domain-Specific Styling

**ROOT Type System**:
- Custom emoji indicators per object type
- Color coding for data types
- Unicode tree drawing for nested structures

**Implementation Idea**:
```python
# Type to emoji mapping
TYPE_ICONS = {
    'TTree': '🌴',
    'TBranch': '🌿',
    'TLeaf': '🍁',
    'TH1F': '📊',
    'TFile': '📁',
}
```

---

## Development & Community

**Repository**: https://github.com/scikit-hep/uproot-browser
**Issues**: Currently 0 open issues
**Pull Requests**: 2 open
**Stars**: 75
**Forks**: 9
**Latest Update**: 2025-07-29 (feat: add --testdata)

**Part of Scikit-HEP**:
- Scientific Python ecosystem for High Energy Physics
- Interoperates with uproot, awkward-array, hist, mplhep
- Active community on Gitter and GitHub Discussions

**Contributing**: See [CONTRIBUTING.md](https://github.com/scikit-hep/uproot-browser/blob/main/.github/CONTRIBUTING.md)

**Community Links**:
- [Scikit-HEP Organization](https://scikit-hep.org/)
- [GitHub Discussions](https://github.com/scikit-hep/uproot-browser/discussions)
- [Gitter Chat](https://gitter.im/Scikit-HEP/community)

---

## Related Textual Patterns

**Similar Projects**:
- **frogmouth** - Markdown browser (Textualize official)
- **hexabyte** - Hex editor (binary data visualization)
- **dolphie** - MySQL monitor (real-time data display)

**Common Patterns**:
- Domain-specific data browsing TUIs
- CLI + TUI hybrid interfaces
- Terminal-based visualization for remote work
- Scientific/technical data exploration

---

## Key Takeaways for Textual Developers

1. **Scientific Computing**: Textual is viable for serious scientific tools (not just demos)
2. **Remote Work**: TUIs excel for SSH/HPC environments where GUI isn't available
3. **Hybrid Interfaces**: Combine CLI (quick operations) + TUI (exploration) effectively
4. **Domain Expertise**: Custom widgets/styling for specialized data types
5. **Lazy Loading**: Handle large scientific datasets efficiently in TUI
6. **Visualization**: Terminal plotting (plotext) + TUI navigation is powerful combo
7. **Ecosystem Integration**: Fits naturally into scientific Python stack (uproot, hist, awkward)

---

## Technical Details

**Python Version**: Compatible with modern Python (3.x)
**Installation Size**: 10.2 KB (README)
**File Count**: 191 lines in README
**Package Extras**:
- `[testdata]` - Include scikit-hep-testdata examples
- `[iterm]` - iTerm2 graphics support

**Dependencies** (core):
- textual - TUI framework
- click - CLI interface
- plotext - Terminal plotting
- uproot - ROOT file I/O
- hist - Histogram handling

**CI/CD**:
- GitHub Actions for continuous integration
- pre-commit.ci for code quality
- Conda-Forge distribution
- PyPI releases

---

## Sources

**Primary Source**:
- [GitHub Repository](https://github.com/scikit-hep/uproot-browser) - Main codebase and documentation
- [README.md](https://github.com/scikit-hep/uproot-browser/blob/main/README.md) - Comprehensive usage guide (accessed 2025-11-02)

**Related Resources**:
- [Scikit-HEP](https://scikit-hep.org/) - Scientific Python for High Energy Physics
- [ROOT](https://root.cern/) - Original C++ framework and file format
- [Textual](https://github.com/Textualize/textual) - TUI framework
- [plotext](https://github.com/piccolomo/plotext) - Terminal plotting library

**Package Repositories**:
- [PyPI: uproot-browser](https://pypi.org/project/uproot-browser/)
- [conda-forge feedstock](https://github.com/conda-forge/uproot-browser-feedstock)
