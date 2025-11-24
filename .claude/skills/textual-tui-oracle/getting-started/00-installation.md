# Installing Textual

**Extract from**: [source-documents/02-getting-started.md](../source-documents/02-getting-started.md)

## Requirements

- **Python 3.9 or later** (recommend latest Python)
- Works on: Linux, macOS, Windows
- Any OS where Python runs

## Platform-Specific Notes

### Linux
All distros include compatible terminal emulators.

### macOS
Default Terminal.app limited to 256 colors. Recommended alternatives:
- [iTerm2](https://iterm2.com/)
- [Ghostty](https://ghostty.org/)
- [Kitty](https://sw.kovidgoyal.net/kitty/)
- [WezTerm](https://wezfurlong.org/wezterm/)

### Windows
Use [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701) for best results.

## Installation Methods

### PyPI (Recommended)

**Basic installation:**
```bash
pip install textual
```

**With developer tools:**
```bash
pip install textual-dev
```

**With syntax highlighting** (for TextArea widget):
```bash
pip install "textual[syntax]"
```

### Conda-forge

Using micromamba (preferred package manager):
```bash
# Install Textual
micromamba install -c conda-forge textual

# Install developer tools
micromamba install -c conda-forge textual-dev
```

## Developer Tools

After installing `textual-dev`, you have access to the `textual` CLI:

```bash
# See all available commands
textual --help

# Common commands
textual run app.py          # Run app in development mode
textual console             # Open console for debugging
textual keys                # Test key combinations
textual borders             # Preview border styles
textual colors              # Preview color palette
```

## Verify Installation

Run the built-in demo:
```bash
python -m textual
```

You should see the Textual demo application showcasing various widgets and features.

## Next Steps

- **Try examples**: Clone the repo and explore `examples/` directory
- **Read tutorial**: https://textual.textualize.io/tutorial/
- **Join Discord**: https://discord.gg/Enf6Z3qhVr

---

**Source**: [source-documents/02-getting-started.md](../source-documents/02-getting-started.md)
