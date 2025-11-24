# Getting Started with Textual

**Source**: https://textual.textualize.io/getting_started/

All you need to get started building Textual apps.

## Requirements

Textual requires **Python 3.9 or later** (if you have a choice, pick the most recent Python). Textual runs on Linux, macOS, Windows and probably any OS where Python also runs.

### Linux (all distros)
All Linux distros come with a terminal emulator that can run Textual apps.

### macOS
The default terminal app is limited to 256 colors. We recommend installing a newer terminal such as:
- [iTerm2](https://iterm2.com/)
- [Ghostty](https://ghostty.org/)
- [Kitty](https://sw.kovidgoyal.net/kitty/)
- [WezTerm](https://wezfurlong.org/wezterm/)

### Windows
The new [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701) runs Textual apps beautifully.

## Installation

### From PyPI

Install Textual via PyPI:
```bash
pip install textual
```

For development tools:
```bash
pip install textual-dev
```

For syntax highlighting in TextArea widget:
```bash
pip install "textual[syntax]"
```

### From conda-forge

Using micromamba (preferred):
```bash
micromamba install -c conda-forge textual
micromamba install -c conda-forge textual-dev
```

### Textual CLI

If you installed developer tools, you have access to the `textual` command:
```bash
textual --help
```

See [devtools](https://textual.textualize.io/guide/devtools/) for more about the `textual` command.

## Demo

Once Textual is installed, try the demo:
```bash
python -m textual
```

## Examples

Clone the Textual repository for examples:
```bash
git clone https://github.com/Textualize/textual.git
cd textual/examples/
python code_browser.py ../
```

### Widget examples
Find code listings used in docs screenshots in the `docs/examples` directory.

## Need Help?

See the [help page](https://textual.textualize.io/help/) for how to get help with Textual or to report bugs.
