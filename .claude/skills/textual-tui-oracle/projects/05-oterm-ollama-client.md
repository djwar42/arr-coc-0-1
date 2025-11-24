# oterm - Ollama AI Terminal Client

**Source**: [https://github.com/ggozad/oterm](https://github.com/ggozad/oterm)
**Accessed**: 2025-11-02
**Stars**: 2,228 | **Forks**: 127
**License**: MIT
**Latest Update**: 2025-08-13

---

## Overview

`oterm` is a feature-rich terminal client for [Ollama](https://github.com/ollama/ollama), providing an intuitive TUI for interacting with local AI models. Built with Textual, it offers persistent chat sessions, Model Context Protocol (MCP) integration, and extensive customization - all without requiring separate servers or frontends.

**Key Distinction**: Unlike web-based Ollama interfaces, `oterm` runs entirely in the terminal with full offline capability and native terminal integration.

---

## Features

### Core Capabilities

**Chat & Session Management**:
- Multiple persistent chat sessions stored in SQLite
- Each session preserves:
  - Conversation history
  - Custom system prompts
  - Model-specific parameters
- Session switching without data loss

**Model Integration**:
- Works with any Ollama-pulled model
- Support for custom models
- Model-specific parameter customization
- "Thinking" mode for models that support it (e.g., reasoning models)

**Model Context Protocol (MCP)**:
- Full MCP tools integration
- MCP prompts support
- MCP sampling (allows servers to request completions)
- Multiple transport types: `stdio`, HTTP Streamable, WebSocket
- HTTP bearer token authentication

**Image Support**:
- Include images in conversations
- Visual model interactions (vision-language models)

**Multimodal Streaming**:
- Streaming responses with tools
- Real-time token generation display
- Async response handling

---

## Installation

**Quick Start** (uvx - recommended):
```bash
uvx oterm
```

**Homebrew** (macOS):
```bash
brew install oterm
```

**pip**:
```bash
pip install oterm
```

**AUR** (Arch Linux):
```bash
yay -S oterm
```

**NixOS**:
```bash
nix-env -iA nixpkgs.oterm
```

**FreeBSD**:
```bash
pkg install misc/py-oterm
```

**Prerequisites**: Ollama must be installed and running. See [Ollama Installation Guide](https://github.com/ollama/ollama?tab=readme-ov-file#ollama).

---

## Usage

**Launch**:
```bash
oterm
```

**Basic Workflow**:
1. Start `oterm` - greeted by animated splash screen
2. Select/create chat session
3. Choose model and customize parameters
4. Optional: Attach MCP tools/prompts
5. Start conversing with streaming responses
6. Sessions auto-saved to SQLite

---

## AI Integration Patterns

### Streaming & Async

`oterm` demonstrates production-quality patterns for AI response streaming in Textual TUIs:

**Streaming with Tools**:
- Handles streaming completions while tool calls are active
- Real-time token display as model generates
- Graceful fallback for non-streaming models

**Async Architecture**:
- Non-blocking AI requests
- UI remains responsive during generation
- Clean async/await patterns for Ollama API

**Response Handling**:
- Progressive rendering of streamed tokens
- Message UI styling improvements (recent update)
- Support for multimodal responses (text + images)

### Model Context Protocol Integration

`oterm` bridges MCP servers with Ollama, enabling:

**MCP Tools** → **Ollama Tools**:
- Transforms MCP tool definitions into Ollama-compatible format
- Provides external information sources to models
- Example: Git MCP server allows model to query repository

**MCP Prompts**:
- Interactive forms for parameterized prompts
- Inserts prompt messages directly into chat
- Reusable prompt templates

**MCP Sampling**:
- Acts as gateway between Ollama and MCP servers
- MCP servers can request completions
- Servers declare model preferences and parameters

**Transport Support**:

1. **stdio** (local servers):
```json
{
  "mcpServers": {
    "git": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "--mount", "type=bind,src=/path,dst=/oterm", "mcp/git"]
    }
  }
}
```

2. **Streamable HTTP** (remote servers):
```json
{
  "mcpServers": {
    "my_mcp": {
      "url": "http://remote:port/path"
    }
  }
}
```

3. **WebSocket** (remote servers):
```json
{
  "mcpServers": {
    "my_mcp": {
      "url": "wss://remote:port/path"
    }
  }
}
```

4. **HTTP Bearer Auth**:
```json
{
  "mcpServers": {
    "my_mcp": {
      "url": "http://remote:port/path",
      "auth": {
        "type": "bearer",
        "token": "XXX"
      }
    }
  }
}
```

---

## Textual Integration Examples

### Architecture Insights

**Session Persistence**:
- SQLite backend for chat storage
- Async database operations to keep UI responsive
- Schema stores: messages, system prompts, parameters

**UI Components**:
- Custom chat message widgets with streaming support
- Model selection screens with parameter customization
- Image selection interface for multimodal input
- Theme support (multiple built-in themes)

**Event Handling**:
- Keyboard shortcuts for navigation
- Command palette for power users
- Real-time message updates as tokens stream

**Configuration**:
- JSON-based config file (`config.json`)
- MCP server declarations
- User preferences and defaults

---

## Key Patterns for AI TUIs

**1. Non-Blocking Generation**:
```python
# Pseudo-code pattern
async def stream_response():
    async for token in ollama_client.generate_stream():
        await self.update_message(token)
        # UI stays responsive
```

**2. Tool Integration**:
- Transform external tool schemas → model-compatible format
- Handle tool calls during streaming
- Display tool execution results inline

**3. Session Management**:
- Persistent storage with SQLite
- Async read/write to avoid UI blocking
- Session switching without data loss

**4. Progressive Rendering**:
- Append tokens as they arrive
- Markdown rendering for formatted responses
- Handle multimodal content (text + images)

---

## Platform Support

- **Linux**: Full support
- **macOS**: Full support (Homebrew available)
- **Windows**: Full support
- **FreeBSD**: Full support
- **Most terminal emulators**: Wide compatibility

---

## Recent Updates (What's New)

- RAG example with [haiku.rag](https://github.com/ggozad/haiku.rag) integration
- Official Homebrew package (in `homebrew/core`)
- "Thinking" mode for reasoning models
- Streaming with tools support
- Message UI styling improvements
- MCP Sampling support
- Streamable HTTP & WebSocket transports for MCP

---

## Architecture Highlights

**Technology Stack**:
- **UI Framework**: Textual (TUI framework)
- **AI Backend**: Ollama (local LLM runtime)
- **Storage**: SQLite (chat persistence)
- **Protocol**: Model Context Protocol (tool/prompt integration)
- **Async**: Python asyncio (non-blocking operations)

**Design Philosophy**:
- No external servers required (runs locally)
- Terminal-native experience (no browser)
- Persistent sessions (survives restarts)
- Extensible (MCP tools/prompts)
- Model-agnostic (works with any Ollama model)

---

## Use Cases

**Development**:
- Quick AI assistance without leaving terminal
- Git integration via MCP (query repo, get diffs)
- Code explanation and generation

**Research**:
- RAG (Retrieval-Augmented Generation) workflows
- Document analysis with vision models
- Iterative prompt engineering

**Productivity**:
- Chat sessions organized by topic
- Custom system prompts per session
- Tool-augmented AI for external data

---

## Resources

- **Documentation**: [https://ggozad.github.io/oterm/](https://ggozad.github.io/oterm/)
- **GitHub**: [https://github.com/ggozad/oterm](https://github.com/ggozad/oterm)
- **MCP Documentation**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Ollama**: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

---

## Textual Learning Points

**For TUI Developers**:

1. **Streaming AI Responses**: How to handle progressive token generation in Textual widgets
2. **Async Database**: SQLite integration without blocking UI
3. **Theme System**: Multiple visual themes for user customization
4. **Complex Forms**: Interactive forms for MCP prompts with parameters
5. **Session Management**: Persistent state across application restarts
6. **Tool Integration**: Bridging external protocols (MCP) with AI backends (Ollama)
7. **Multimodal UI**: Handling text + image inputs in chat interface
8. **Configuration**: JSON-based config with hot-reload potential

**Code Quality**: Production-grade Textual application with 2.2k+ stars, active maintenance, and multi-platform support.

---

## Related Projects

- **Ollama**: Local LLM runtime that `oterm` interfaces with
- **Model Context Protocol**: Anthropic's open-source protocol for AI tool integration
- **haiku.rag**: RAG integration example with `oterm`

---

**Project Type**: Community (not official Textualize)
**Maturity**: Production-ready (Homebrew official, AUR, Nix, FreeBSD packages)
**Active Development**: Yes (last update 2025-08-13)
