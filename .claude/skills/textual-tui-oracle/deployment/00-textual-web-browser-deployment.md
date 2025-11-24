# Textual-Web: Browser Deployment

## Overview

Textual-Web is an official tool from Textualize that serves Textual TUI applications and terminals directly in web browsers. It transforms terminal-based Textual apps into accessible web applications without code modifications, using WebSocket-based communication and a custom protocol for web API integration.

**Key Capabilities:**
- Run Textual apps in any modern web browser
- Serve terminal sessions with full interactivity
- Generate public URLs for instant sharing
- Support multiple apps/terminals from single configuration
- Account-based persistent URLs
- WebSocket protocol for real-time bidirectional communication

**Project Status:** Beta phase (as of 2024)
- Version: 0.8.0 (latest)
- Python requirement: >= 3.8
- Active development with planned sessions and web API features

## Installation

### Via pipx (Recommended)

```bash
# Install textual-web globally
pipx install textual-web

# Verify installation
textual-web --version
```

**Why pipx?**
- Isolated environment (no dependency conflicts)
- Global CLI access
- Easy updates (`pipx upgrade textual-web`)

### Via pip

```bash
# Install in current environment
pip install textual-web

# Or with specific Python version
python3.8 -m pip install textual-web
```

### Dependencies

From [pyproject.toml](https://github.com/Textualize/textual-web/blob/main/pyproject.toml) (accessed 2025-11-02):

**Core:**
- `textual >= 0.43.0` - Textual framework
- `aiohttp >= 3.9.3` - Async HTTP server/client
- `msgpack >= 1.0.5` - Binary serialization for protocol

**Server Components:**
- `uvloop >= 0.19.0` (Unix/macOS only) - High-performance event loop
- `aiohttp-jinja2 >= 1.5.1` - Template rendering

**Utilities:**
- `click >= 8.1.3` - CLI interface
- `pydantic >= 2.1.1` - Configuration validation
- `httpx >= 0.24.1` - HTTP client
- `xdg >= 6.0.0` - XDG base directory support
- `tomli >= 2.0.1` - TOML parsing

## Quick Start

### Serve Example Apps

```bash
# Launch textual-web with welcome screen
textual-web
```

**Output:**
```
Ganglion server
───────────────────────────────────────────────────────────────────
 https://textual.run/1234567890/calculator
 https://textual.run/1234567890/dictionary
 https://textual.run/1234567890/welcome
───────────────────────────────────────────────────────────────────
Press Ctrl+C to stop
```

Click any link to open the Textual app in your browser. URLs are public and shareable during the session.

### Serve Terminal

```bash
# Quick terminal access
textual-web -t

# Serves your default shell in browser
# WARNING: Public URL with full terminal access!
```

**Security Warning:** Terminal URLs grant complete terminal access to your machine. Only share with trusted individuals.

## Configuration

### TOML Configuration File

Create a `serve.toml` file to define multiple apps and terminals:

```toml
# Textual Apps
[app.Calculator]
command = "python calculator.py"
slug = "calc"  # Optional: custom URL slug

[app.Dictionary]
command = "python dictionary.py"
# slug auto-derived from name: "dictionary"

# Terminals
[terminal.HTOP]
command = "htop"

[terminal.Shell]
# No command = uses default shell
```

**Load configuration:**
```bash
textual-web --config serve.toml
```

### Configuration Options

**App Sections (`[app.NAME]`):**
- `command` (required): Shell command to launch app
- `slug` (optional): Custom URL path segment

**Terminal Sections (`[terminal.NAME]`):**
- `command` (optional): Shell command (defaults to user's shell)

**Slug Derivation:**
- Auto-generated from section name if not specified
- Converts to lowercase with hyphens
- Example: `[app.MyAwesomeApp]` → slug: `myawesomeapp`

### Platform Support

**Textual Apps:** All platforms (Windows, macOS, Linux)

**Terminals:**
- ✅ macOS - Full support
- ✅ Linux - Full support
- ⏳ Windows - Planned for future release

## Account Management

### Creating Accounts

Accounts provide **persistent URLs** that don't change between runs.

```bash
# Launch signup dialog
textual-web --signup
```

**Signup Dialog:**
- Email address (for account recovery)
- Username (becomes part of URL)
- Password

**Output:** Creates `ganglion.toml` with API key:

```toml
[account]
api_key = "JSKK234LLNWEDSSD"

# Add your apps/terminals below
[app.MyApp]
command = "python app.py"
```

### Using Accounts

```bash
# Run with account configuration
textual-web --config ganglion.toml
```

**URL Format with Account:**
```
https://textual.run/{username}/{app-slug}
```

**Benefits:**
- Consistent URLs across sessions
- Bookmarkable links
- Professional sharing (no random digits)
- Account-scoped app management

## Architecture

### WebSocket Protocol

Textual-Web uses a **custom WebSocket protocol** (not standard terminal emulation) for bidirectional communication between browser and Textual app.

**Protocol Layers:**

1. **Transport:** WebSocket (bidirectional, low-latency)
2. **Serialization:** MessagePack (efficient binary format)
3. **Protocol:** Custom Textual-Web messages

**Why Custom Protocol?**
- Exposes web APIs to Textual apps (file downloads, uploads, etc.)
- More efficient than terminal escape sequences
- Enables future web-specific features
- Allows session persistence (planned)

### Server Architecture

**Components:**

1. **HTTP Server** (aiohttp)
   - Serves static HTML/JS/CSS
   - Handles WebSocket upgrades
   - Template rendering (Jinja2)

2. **WebSocket Handler**
   - Manages client connections
   - Routes messages to/from Textual apps
   - Handles protocol serialization

3. **Process Manager**
   - Spawns Textual app processes
   - Manages app lifecycle
   - Handles graceful shutdown

4. **URL Router**
   - Maps slugs to apps/terminals
   - Generates public URLs
   - Account slug resolution

**Data Flow:**

```
Browser Client
    ↓ [WebSocket]
HTTP Server (aiohttp)
    ↓ [MessagePack Protocol]
WebSocket Handler
    ↓ [Process IPC]
Textual App Process
    ↓ [Textual Driver]
App Rendering & Logic
```

### Terminal vs Textual App Serving

**Terminal Mode:**
- Uses PTY (pseudo-terminal) for process I/O
- Sends raw terminal escape sequences over WebSocket
- Browser renders using terminal emulator (xterm.js likely)
- App unaware it's in browser

**Textual App Mode:**
- Uses custom driver (not terminal escape sequences)
- Sends structured protocol messages
- Browser renders using custom logic
- App can access web APIs through protocol

**Unified Interface:** Both modes use same WebSocket transport, different protocols.

## Deployment Workflow

### Development Workflow

**1. Local Development:**
```bash
# Develop Textual app normally
python my_app.py

# Test in terminal
textual run my_app.py

# Preview in browser
textual-web --config dev.toml
```

**2. Share for Testing:**
```bash
# Generate temporary public URL
textual-web -c app.toml

# Share URL with testers
# URL expires when server stops
```

**3. Production Deployment:**
```bash
# Create account for persistent URL
textual-web --signup

# Add apps to ganglion.toml
# Deploy server on cloud VM
# Run with process manager (systemd, supervisor, pm2)
```

### Production Deployment Patterns

**Cloud VM Deployment:**

```bash
# Install on Ubuntu/Debian server
sudo apt update
sudo apt install pipx
pipx install textual-web

# Create config
cat > /etc/textual-web/apps.toml <<EOF
[account]
api_key = "YOUR_API_KEY"

[app.Production]
command = "/usr/local/bin/python /opt/myapp/app.py"
EOF

# Run with systemd
sudo systemctl enable textual-web
sudo systemctl start textual-web
```

**Docker Deployment:**

```dockerfile
FROM python:3.11-slim

# Install textual-web
RUN pip install textual-web

# Copy app and config
COPY app.py /app/
COPY serve.toml /app/

WORKDIR /app

# Run server
CMD ["textual-web", "--config", "serve.toml"]
```

**Reverse Proxy (Nginx):**

```nginx
# Proxy WebSocket connections
location /apps/ {
    proxy_pass http://localhost:8000/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### Security Considerations

**Terminal Access:**
- ⚠️ Terminal mode grants full shell access
- Use authentication/firewall for production
- Consider restricted shells or docker containers

**App Isolation:**
- Each app runs as separate process
- Inherits server user permissions
- Sandbox with containers for untrusted apps

**URL Security:**
- Random URLs provide weak security (obscurity)
- Account URLs are predictable
- Add authentication layer for sensitive apps

## Debugging

### Debug Mode

```bash
# Enable verbose logging
DEBUG=1 textual-web --config app.toml
```

**Debug Output:**
- WebSocket connection events
- Protocol message details
- Process lifecycle events
- Error tracebacks

**Performance Impact:** Debug mode may slow apps significantly due to logging overhead.

### Common Issues

**Color Glitches:**
- **Symptom:** Apps with many colors display incorrectly
- **Cause:** Bug in upstream library (as of 2023-09-06)
- **Status:** Fix expected in future update
- **Workaround:** Reduce color palette or wait for update

**Mobile Experience:**
- **iPhone:** Quite usable
- **Android/Other:** Variable quality, some issues
- **Status:** Improvements planned for future updates

**Connection Drops:**
- **Cause:** Browser tab close terminates app
- **Future:** Session persistence planned (resume apps)

## Features and Limitations

### Current Features (v0.8.0)

✅ **Working:**
- Serve Textual apps in browser
- Serve terminals (macOS/Linux)
- Multiple apps per server
- Public URL generation
- Account-based persistent URLs
- Configuration file support
- WebSocket-based communication
- Custom protocol for web APIs

### Planned Features

🔮 **Roadmap:**

**Sessions (High Priority):**
- Close browser tab without terminating app
- Resume app from where you left off
- Survive server upgrades without disconnect
- Multi-tab access to same app instance

**Web API Integration:**
- File downloads (e.g., generate CSV → save to Downloads)
- File uploads to Textual apps
- Clipboard integration
- Native notifications
- Browser storage access

**Enhanced Mobile:**
- Improved touch handling
- Virtual keyboard optimization
- Responsive layout adjustments

**Windows Terminal Support:**
- Full terminal serving on Windows

### Known Limitations

**Current Constraints:**

1. **No Session Persistence:**
   - Close tab → app terminates
   - Server restart → all apps lost

2. **Mobile Experience:**
   - Variable quality across devices
   - Touch interaction issues

3. **Color Rendering:**
   - Glitches with high color counts
   - Upstream library dependency

4. **Windows Terminals:**
   - Not yet supported
   - Textual apps work fine

5. **Authentication:**
   - No built-in auth system
   - URLs provide security by obscurity only

6. **Single Instance:**
   - One browser tab = one app instance
   - No multi-user access to same app (yet)

## Example: Multi-App Configuration

### Complete Production Setup

```toml
# ganglion.toml - Production configuration

[account]
api_key = "YOUR_PRODUCTION_API_KEY"

# Monitoring Dashboard
[app.Dashboard]
command = "python /opt/monitoring/dashboard.py"
slug = "monitor"

# Log Viewer
[app.Logs]
command = "python /opt/monitoring/logs.py"
slug = "logs"

# System Stats (htop alternative)
[app.SystemStats]
command = "python /opt/monitoring/stats.py"
slug = "stats"

# Admin Terminal (RESTRICTED ACCESS!)
[terminal.Admin]
command = "bash"
slug = "admin-shell"

# Development Workspace
[terminal.DevBox]
command = "tmux attach -t dev || tmux new -s dev"
slug = "dev"
```

**Generated URLs:**
```
https://textual.run/{your-username}/monitor
https://textual.run/{your-username}/logs
https://textual.run/{your-username}/stats
https://textual.run/{your-username}/admin-shell
https://textual.run/{your-username}/dev
```

## Integration Patterns

### Textual App Compatibility

**Zero Code Changes Required:**

Existing Textual apps work in browser without modification:

```python
# app.py - Works in terminal AND browser
from textual.app import App
from textual.widgets import Header, Footer

class MyApp(App):
    def compose(self):
        yield Header()
        yield Footer()

if __name__ == "__main__":
    app = MyApp()
    app.run()
```

**Serve it:**
```toml
[app.MyApp]
command = "python app.py"
```

### Future Web API Integration

**Planned Pattern (not yet available):**

```python
from textual.app import App
from textual.web import download_file  # Future API

class DataExporter(App):
    async def export_csv(self):
        data = self.generate_report()

        # In terminal: saves to current directory
        # In browser: triggers download to Downloads folder
        await download_file("report.csv", data)
```

**Vision:** Same code, context-aware behavior (terminal vs browser).

## Testing and Development

### Local Testing Workflow

```bash
# 1. Clone Textual examples
git clone https://github.com/Textualize/textual.git
cd textual/examples

# 2. Create test config
cat > test.toml <<EOF
[app.Calculator]
command = "python calculator.py"

[app.Dictionary]
command = "python dictionary.py"
EOF

# 3. Serve locally
textual-web --config test.toml

# 4. Open in browser
# Click generated links
```

### Community Testing

Join [Textualize Discord](https://discord.com/invite/Enf6Z3qhVr) for:
- Beta testing coordination
- Bug reports and feedback
- Feature discussions
- Implementation help

## Performance Considerations

### Network Performance

**WebSocket Overhead:**
- Lower latency than HTTP polling
- Efficient binary serialization (MessagePack)
- Bidirectional without handshake overhead

**Bandwidth:**
- Terminal mode: Sends escape sequences (verbose)
- Textual mode: Sends structured protocol (efficient)
- Minimal data for static screens
- Higher bandwidth for animated/updating UIs

### Server Resource Usage

**Per-App Costs:**
- Process overhead: ~10-50MB per Textual app
- WebSocket connection: ~1-5KB per client
- CPU: Minimal when idle, scales with app logic

**Scaling Limits:**
- Depends on app complexity
- Network bandwidth (WebSocket connections)
- Process limits on host system

**Optimization:**
- Use process pooling for identical apps (future)
- Deploy multiple servers with load balancing
- Containerize apps for resource limits

## Sources

**Primary Repository:**
- [Textualize/textual-web](https://github.com/Textualize/textual-web) - Official GitHub repository (accessed 2025-11-02)

**Documentation:**
- [README.md](https://github.com/Textualize/textual-web/blob/main/README.md) - Main documentation (accessed 2025-11-02)
- [pyproject.toml](https://github.com/Textualize/textual-web/blob/main/pyproject.toml) - Dependencies and version info (accessed 2025-11-02)
- [CHANGELOG.md](https://github.com/Textualize/textual-web/blob/main/CHANGELOG.md) - Version history (accessed 2025-11-02)

**Community:**
- [Hacker News Discussion](https://news.ycombinator.com/item?id=37418424) - Initial announcement
- [Textualize Discord](https://discord.com/invite/Enf6Z3qhVr) - Beta testing coordination

**Project Stats (as of 2025-11-02):**
- Stars: 1,300+
- Forks: 28
- Contributors: 4 (Will McGugan, Darren Burns, Rodrigo Girão Serrão, Dave Pearson)
- License: MIT
- Latest Release: v0.7.0 (Feb 20, 2024)
- Current Version: v0.8.0

## Cross-References

**Related Oracle Knowledge:**

- [getting-started/00-official-homepage.md](../getting-started/00-official-homepage.md) - Textual framework overview
- [getting-started/01-official-tutorial.md](../getting-started/01-official-tutorial.md) - Building Textual apps
- [core-concepts/00-application-basics.md](../core-concepts/00-application-basics.md) - App structure fundamentals
- [advanced/00-drivers-and-platforms.md](../advanced/00-drivers-and-platforms.md) - Platform compatibility

**See Also:**
- Community examples (coming in PART 1-3)
- Production app case studies (coming in PART 6-8)
