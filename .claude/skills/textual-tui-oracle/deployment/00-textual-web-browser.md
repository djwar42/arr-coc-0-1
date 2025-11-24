# Textual-Web Browser Deployment

## Overview

**Critical Distinction: NOT WebAssembly/Pyodide**

Textual-web does **NOT** use WebAssembly or Pyodide. It uses a fundamentally different architecture: **server-driven applications accessed via browser using WebSockets and xterm.js terminal emulation**.

From [Textual-web GitHub](https://github.com/Textualize/textual-web) (accessed 2025-11-02):
> "Textual Web publishes Textual apps and terminals on the web... The Textual app runs on a machine/server under your control, and communicates with the browser via a protocol which runs over websocket."

## Architecture: Server-Driven, Not Client-Side

### How It Actually Works

**Server-Side Execution:**
- Textual app runs as a Python process on your server/machine
- App has full access to server resources (filesystem, databases, network)
- Uses standard CPython interpreter (not compiled to WASM)

**Browser-Side Rendering:**
- Terminal rendered using [xterm.js](https://xtermjs.org/) (same engine as VS Code)
- xterm.js acts as terminal emulator in browser
- Translates user interactions into ANSI escape codes

**WebSocket Communication:**
- Bidirectional communication between browser and server
- Browser sends: ANSI escape codes (keyboard, mouse events)
- Server sends: ANSI escape codes (visual updates, terminal output)
- Additional protocol messages for web-specific features

### Data Flow Diagram

```
┌─────────────────────────────
│ END USER BROWSER
│
│  ┌──────────────────────
│  │ xterm.js Terminal
│  │ (Frontend Emulator)
│  │
│  │ • Renders ANSI output
│  │ • Captures user input
│  │ • Converts to escape codes
│  └──────────────────────
│         │
│         │ WebSocket
│         │ (ANSI + Protocol Messages)
│         ▼
└─────────────────────────────

┌─────────────────────────────
│ SERVER (textual-serve)
│
│  ┌──────────────────────
│  │ WebSocket Handler
│  │
│  │ • Routes messages
│  │ • Manages connections
│  │ • Handles file delivery
│  └──────────────────────
│         │
│         │ stdin/stdout pipes
│         ▼
│  ┌──────────────────────
│  │ Textual App Process
│  │ (Python subprocess)
│  │
│  │ • Runs actual app code
│  │ • Full server access
│  │ • Standard Python
│  └──────────────────────
│
└─────────────────────────────
```

From [Towards Textual Web Applications](https://textual.textualize.io/blog/2024/09/08/towards-textual-web-applications/) (accessed 2025-11-02):
> "The Textual app runs on a machine/server under your control, and communicates with the browser via a protocol which runs over websocket. End-users interacting with the app via their browser do not have access to the machine the application is running on via their browser, only the running Textual app."

## Installation and Setup

### Prerequisites

- Python 3.8+ (server-side)
- pipx (recommended) or pip
- Network access for serving

### Install textual-web

```bash
# Recommended: Install with pipx
pipx install textual-web

# Alternative: Install with pip
pip install textual-web
```

### Verify Installation

```bash
textual-web --version
```

## Basic Usage

### Quick Start: Run Welcome Demo

```bash
# Start textual-web with example apps
textual-web
```

**Output:**
```
╔═══════════════════════════
║ Textual Web Server
╠═══════════════════════════
║ Calculator: http://localhost:8000/abc123/calculator
║ Dictionary: http://localhost:8000/abc123/dictionary
╚═══════════════════════════
```

Click URLs to access apps in browser. Each session gets unique URL.

### Serve a Terminal

```bash
# Serve your shell in browser
textual-web -t

# Terminal URL generated:
# http://localhost:8000/xyz789/terminal
```

**Warning:** Don't share terminal URLs with untrusted users - they get terminal access to your machine!

### Serve Custom Apps

Create configuration file `serve.toml`:

```toml
[app.Calculator]
command = "python calculator.py"

[app.Dictionary]
command = "python dictionary.py"

[terminal.Terminal]
# Serves default shell

[terminal.HTOP]
command = "htop"
```

Run with config:

```bash
textual-web --config serve.toml
```

### Custom URL Slugs

```toml
[app.Calculator]
command = "python calculator.py"
slug = "calc"  # URL becomes: /abc123/calc
```

## Account Management for Permanent URLs

### Create Account

```bash
textual-web --signup
```

**Signup Dialog:**
- Email address
- Choose account slug (appears in URLs)
- Creates `ganglion.toml` with API key

### Account Configuration

```toml
# ganglion.toml
[account]
api_key = "JSKK234LLNWEDSSD"

[app.MyApp]
command = "python myapp.py"
```

**Benefits:**
- Permanent URLs (account slug doesn't change)
- Same URL across runs
- Share stable links with team

### Use Account Config

```bash
textual-web --config ganglion.toml
```

## Driver Architecture: How Input/Output Works

### Terminal Drivers

From [Towards Textual Web Applications](https://textual.textualize.io/blog/2024/09/08/towards-textual-web-applications/) (accessed 2025-11-02):

> "Input in Textual apps is handled, at the lowest level, by 'driver' classes. We have different drivers for Linux and Windows, and also one for handling apps being served via web."

**Linux/Windows Drivers:**
1. Read `stdin` for ANSI escape sequences
2. Parse sequences (mouse movement, keyboard input)
3. Translate to Textual Events
4. Send to app's message queue

**Web Driver:**
1. xterm.js captures browser interactions
2. Converts to ANSI escape codes
3. Sends via WebSocket to textual-serve
4. Pipes to Textual app's `stdin`
5. Web driver parses and creates Events

### Output Rendering

**Terminal:**
- App writes ANSI to `stdout`
- Terminal emulator renders directly

**Browser:**
- App writes ANSI to `stdout`
- textual-serve reads `stdout`
- Sends via WebSocket to browser
- xterm.js renders in browser

## Platform-Agnostic APIs

### Opening Web Links

**Problem:** `webbrowser.open()` opens on server, not user's browser

**Solution:** `App.open_url()`

```python
from textual.app import App

class MyApp(App):
    def on_button_pressed(self):
        # Works correctly in terminal AND browser
        self.open_url("https://example.com")
```

**Behavior:**
- **Terminal:** Uses `webbrowser` module (opens on user's machine)
- **Browser:** Sends WebSocket message → browser opens link in new tab

From [Towards Textual Web Applications](https://textual.textualize.io/blog/2024/09/08/towards-textual-web-applications/) (accessed 2025-11-02):
> "When the Textual app is being served and used via the browser however, the running app will inform textual-serve, which will in turn tell the browser via websocket that the end-user is requesting to open a link, which will then be opened in their browser - just like a normal web link."

### Saving Files to Disk

**Problem:** Writing to server filesystem doesn't help browser users

**Solution:** `App.deliver_text()` and `App.deliver_binary()`

```python
from textual.app import App

class MyApp(App):
    def export_data(self):
        data = "CSV content here"

        # Works in terminal AND browser
        self.deliver_text(
            data,
            filename="export.csv",
            mime_type="text/csv"
        )
```

**Behavior:**
- **Terminal:** Writes file to disk, notifies app when complete
- **Browser:** Initiates download via ephemeral URL, streams file to browser

### File Delivery Protocol

From [Towards Textual Web Applications](https://textual.textualize.io/blog/2024/09/08/towards-textual-web-applications/) (accessed 2025-11-02):

> "To support file delivery we updated our protocol to allow applications to signal that a file is 'ready' for delivery when one of the new 'deliver file' APIs is called. An ephemeral, single-use, download link is then generated and sent to the browser via websocket."

**Streaming Process:**
1. App calls `deliver_text()` or `deliver_binary()`
2. App signals "file ready" via protocol
3. textual-serve generates one-time download URL
4. URL sent to browser via WebSocket
5. Browser opens URL
6. File streamed in chunks (Bencode encoding)
7. Download completes

**API Reference:**

```python
# Deliver text file
await app.deliver_text(
    content: str,
    filename: str,
    mime_type: str = "text/plain",
    open_method: str = "download"  # or "browser"
)

# Deliver binary file
await app.deliver_binary(
    content: bytes,
    filename: str,
    mime_type: str = "application/octet-stream",
    open_method: str = "download"
)
```

## Use Cases

### Example 1: Harlequin SQL IDE

```bash
# Install harlequin on server
pip install harlequin

# Serve it
textual-web harlequin mydb.sqlite

# Share URL with team
# Users query databases from server via browser
```

### Example 2: Posting API Client

From [Towards Textual Web Applications](https://textual.textualize.io/blog/2024/09/08/towards-textual-web-applications/) (accessed 2025-11-02):

> "Or, you could deploy posting (a terminal-based API client) on a server, and provide your colleagues with the URL, allowing them to quickly send HTTP requests from that server, right from within their browser."

```bash
# Install posting on server with API access
pip install posting

# Serve it
textual-web posting

# Team sends requests FROM server (bypasses local firewalls)
```

### Example 3: Internal Tools Dashboard

```toml
# internal-tools.toml
[account]
api_key = "YOUR_API_KEY"

[app.Logs]
command = "tail -f /var/log/app.log"
slug = "logs"

[app.Metrics]
command = "python metrics_dashboard.py"
slug = "metrics"

[app.Deploy]
command = "python deploy_manager.py"
slug = "deploy"
```

```bash
textual-web --config internal-tools.toml

# Share URLs:
# https://yourserver.com/yourslug/logs
# https://yourserver.com/yourslug/metrics
# https://yourserver.com/yourslug/deploy
```

## Development and Debugging

### Debug Mode

```bash
DEBUG=1 textual-web --config myapp.toml
```

**Output:**
- Verbose logging
- Protocol message details
- May slow down apps

### Known Issues

From [Textual-web README](https://github.com/Textualize/textual-web) (accessed 2025-11-02):

**Color Glitches:**
> "You may encounter a glitch with apps that have a lot of colors. This is a bug in an upstream library, which we are expecting a fix for soon."

**Mobile Experience:**
> "The experience on mobile may vary. On iPhone Textual apps are quite usable, but other systems may have a few issues."

## Limitations and Considerations

### Security

**Access Control:**
- URLs contain random tokens (e.g., `/abc123/app`)
- Tokens regenerate each run (unless using account)
- Anyone with URL can access app
- No built-in authentication
- Terminal access = machine access!

**Best Practices:**
- Don't serve terminals to untrusted users
- Use account slugs for stable URLs
- Deploy behind reverse proxy with auth
- Use HTTPS in production

### Performance

**Network Latency:**
- Every keystroke: browser → server → app
- Every frame: app → server → browser
- Works well on LAN/fast connections
- May feel sluggish on slow connections

**Resource Usage:**
- Each browser session = separate Python process
- Memory scales with concurrent users
- CPU for ANSI processing

### Comparison to WebAssembly Approaches

| Feature | Textual-web | Pyodide/WASM |
|---------|-------------|--------------|
| **Execution** | Server (Python process) | Client (browser) |
| **Network** | Constant (WebSocket) | Initial download only |
| **Resources** | Server CPU/memory | Client CPU/memory |
| **Access** | Full server filesystem/network | Sandboxed browser |
| **Scalability** | Limited by server | Unlimited clients |
| **Latency** | Network dependent | No network after load |
| **Python Support** | Full CPython | Limited WASM build |
| **Best For** | Internal tools, shared resources | Public apps, offline use |

## Future Roadmap

From [Textual-web README](https://github.com/Textualize/textual-web) (accessed 2025-11-02):

**Planned Features:**

**Sessions:**
> "Currently, if you close the browser tab it will also close the Textual app. In the future you will be able to close a tab and later resume where you left off. This will also allow us to upgrade servers without kicking anyone off."

**Web APIs:**
> "For example, a Textual app might generate a file (say a CSV with a server report). If you run that in the terminal, the file would be saved in your working directory. But in a Textual app it would be served and saved in your Downloads folder, like a regular web app."

**Custom Protocol:**
> "Currently serving Textual apps and terminals appears very similar... Under the hood, however, Textual apps are served using a custom protocol. This protocol will be used to expose web application features to the Textual app."

## Code Examples

### Basic Server-Side App

```python
# myapp.py
from textual.app import App
from textual.widgets import Button, Label

class ServerApp(App):
    """App that runs on server, accessed via browser"""

    def compose(self):
        yield Label("Running on server, viewed in browser!")
        yield Button("Download Report", id="download")
        yield Button("Open Docs", id="docs")

    async def on_button_pressed(self, event):
        if event.button.id == "download":
            # Works in terminal AND browser
            await self.deliver_text(
                "Server-side generated report",
                filename="report.txt"
            )
        elif event.button.id == "docs":
            # Opens in user's browser (not server browser)
            self.open_url("https://docs.example.com")

if __name__ == "__main__":
    ServerApp().run()
```

### Configuration for Multiple Apps

```toml
# production.toml
[account]
api_key = "YOUR_API_KEY"

[app.Dashboard]
command = "python dashboard.py"
slug = "dash"

[app.Logs]
command = "python log_viewer.py"
slug = "logs"

[terminal.SSH]
command = "ssh production-server"
slug = "ssh"
```

### File Export Example

```python
from textual.app import App
from textual.widgets import Button
import csv
import io

class ExportApp(App):
    def compose(self):
        yield Button("Export CSV")

    async def on_button_pressed(self):
        # Generate CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Name", "Value"])
        writer.writerow(["Metric 1", "100"])
        writer.writerow(["Metric 2", "200"])

        # Deliver to user (works in terminal AND browser)
        await self.deliver_text(
            output.getvalue(),
            filename="metrics.csv",
            mime_type="text/csv"
        )
```

## Version History

From [CHANGELOG.md](https://github.com/Textualize/textual-web/blob/main/CHANGELOG.md) (accessed 2025-11-02):

**v0.7.0 (2024-02-20):**
- Python >= 3.8 required
- Dependency updates

**v0.6.0 (2023-11-28):**
- Added app focus support

## Community and Support

**Discord Server:**
- Join: https://discord.com/invite/Enf6Z3qhVr
- Beta testing coordination
- Help and discussions

**GitHub:**
- Repository: https://github.com/Textualize/textual-web
- Issues: Report bugs and feature requests
- 1.3k stars, active development

**Hacker News Discussion:**
- https://news.ycombinator.com/item?id=37418424

## Sources

**Primary Documentation:**
- [Textual-web GitHub Repository](https://github.com/Textualize/textual-web) - Main README, installation, configuration (accessed 2025-11-02)
- [Towards Textual Web Applications](https://textual.textualize.io/blog/2024/09/08/towards-textual-web-applications/) - Architecture deep dive, driver explanation, file delivery (accessed 2025-11-02)
- [What is Textual Web?](https://textual.textualize.io/blog/2023/09/06/what-is-textual-web/) - Original announcement (accessed 2025-11-02)
- [CHANGELOG.md](https://github.com/Textualize/textual-web/blob/main/CHANGELOG.md) - Version history (accessed 2025-11-02)

**Related Projects:**
- [xterm.js](https://xtermjs.org/) - Browser-side terminal emulator used by textual-web
- [Textual Framework](https://github.com/Textualize/textual) - Core TUI framework
- [Harlequin](https://github.com/tconbeer/harlequin) - SQL IDE example
- [Posting](https://github.com/darrenburns/posting) - API client example

**Technical References:**
- WebSocket protocol for ANSI streaming
- Bencode encoding for file transfer (BitTorrent encoding variant)
- xterm.js terminal emulation (same as VS Code)
