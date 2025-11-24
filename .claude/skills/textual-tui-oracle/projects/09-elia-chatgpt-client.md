# Elia - ChatGPT/Claude/LLM Terminal Client

**Project**: Elia - A snappy, keyboard-centric terminal user interface for interacting with large language models
**Author**: Darren Burns (@darrenburns)
**GitHub**: https://github.com/darrenburns/elia
**Stars**: 2.3k+ | **Forks**: 149 | **License**: Apache-2.0
**Package**: `elia-chat` (via pipx)

## Overview

Elia is a production-grade TUI application for interacting with LLMs (ChatGPT, Claude, Gemini, local models via Ollama/LocalAI) entirely within the terminal. It demonstrates advanced Textual patterns including streaming API responses, SQLite database integration, screen management, custom themes, and keyboard-centric design.

**Key Features**:
- Multi-model support (OpenAI, Anthropic, Google, Ollama, LocalAI)
- Streaming response rendering with real-time updates
- Local conversation storage (SQLite database)
- Inline mode (under prompt) and full-screen mode
- Custom themes and syntax highlighting
- Import ChatGPT conversation history
- Keyboard-focused navigation and controls

## Installation & Usage

```bash
# Install via pipx
pipx install --python 3.11 elia-chat

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# Launch Elia
elia

# Inline mode (chat appears under your prompt)
elia -i "What is the Zen of Python?"

# Full-screen mode with specific model
elia -m gpt-4o "Tell me a cool fact about lizards!"

# Gemini Flash inline
elia -i -m gemini/gemini-1.5-flash-latest "How do I call Rust code from Python?"
```

## Architecture Overview

### Application Structure

Elia follows a clean screen-based architecture with the Textual app pattern:

```
elia_chat/
├── app.py                 # Main Elia app class, theme management
├── config.py              # Model configuration, LaunchConfig
├── chats_manager.py       # SQLite database operations
├── models.py              # ChatData, ChatMessage models
├── runtime_config.py      # Runtime configuration (model selection, system prompt)
├── screens/
│   ├── home_screen.py     # Main screen with conversation list
│   ├── chat_screen.py     # Active chat screen
│   ├── help_screen.py     # Help/keybindings screen
│   └── chat_details.py    # Chat metadata screen
├── widgets/
│   ├── chat.py            # Main chat widget (conversation flow)
│   ├── chatbox.py         # Individual message display
│   ├── prompt_input.py    # User input widget
│   ├── chat_header.py     # Chat title and metadata
│   └── agent_is_typing.py # Response status indicator
└── themes.py              # Theme system (9 built-in themes)
```

### Core Flow

1. **App Launch** → `HomeScreen` (conversation list)
2. **Select/Create Chat** → `ChatScreen` (push screen)
3. **User Message** → Store in SQLite → Stream API response
4. **Agent Response** → Real-time rendering → Store in SQLite
5. **Navigation** → Screen stack (ESC to pop, F1 for help)

## Key Textual Integration Patterns

### 1. Screen Management with Push/Pop

From `app.py`:
```python
class Elia(App[None]):
    BINDINGS = [
        Binding("q", "app.quit", "Quit", show=False),
        Binding("f1,?", "help", "Help"),
    ]

    async def on_mount(self) -> None:
        await self.push_screen(HomeScreen(self.runtime_config_signal))
        self.theme = self.launch_config.theme
        if self.startup_prompt:
            await self.launch_chat(
                prompt=self.startup_prompt,
                model=self.runtime_config.selected_model,
            )

    async def launch_chat(self, prompt: str, model: EliaChatModel) -> None:
        # Create chat data with system + user messages
        chat = ChatData(...)
        chat.id = await ChatsManager.create_chat(chat_data=chat)
        await self.push_screen(ChatScreen(chat))
```

**Pattern**: Home screen stays in stack, chat screens push on top. ESC pops back to home.

### 2. Streaming API Integration with Worker Threads

From `widgets/chat.py` - the core streaming pattern:

```python
class Chat(Widget):
    @work(thread=True, group="agent_response")
    async def stream_agent_response(self) -> None:
        import litellm
        from litellm import ModelResponse, acompletion

        # Prepare messages (trim to context window)
        raw_messages = [message.message for message in self.chat_data.messages]
        messages = trim_messages(raw_messages, model.name)

        try:
            response = await acompletion(
                messages=messages,
                stream=True,
                model=model.name,
                temperature=model.temperature,
                max_retries=model.max_retries,
                api_key=model.api_key.get_secret_value() if model.api_key else None,
                api_base=model.api_base.unicode_string() if model.api_base else None,
            )
        except Exception as exception:
            self.app.notify(f"{exception}", severity="error")
            self.post_message(self.AgentResponseFailed(self.chat_data.messages[-1]))
            return

        # Create response chatbox
        ai_message = {"content": "", "role": "assistant"}
        message = ChatMessage(message=ai_message, model=model, timestamp=now)
        response_chatbox = Chatbox(message, model, classes="response-in-progress")

        self.post_message(self.AgentResponseStarted())
        self.app.call_from_thread(self.chat_container.mount, response_chatbox)

        # Stream chunks
        async for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if isinstance(chunk_content, str):
                self.app.call_from_thread(
                    response_chatbox.append_chunk, chunk_content
                )

            # Auto-scroll if near bottom
            scroll_y = self.chat_container.scroll_y
            max_scroll_y = self.chat_container.max_scroll_y
            if scroll_y in range(max_scroll_y - 3, max_scroll_y + 1):
                self.app.call_from_thread(
                    self.chat_container.scroll_end, animate=False
                )

        self.post_message(self.AgentResponseComplete(...))
```

**Critical Patterns**:
- `@work(thread=True)` - Run API call in background thread
- `app.call_from_thread()` - Update UI from worker thread safely
- Streaming chunks → `append_chunk()` → Incremental rendering
- Auto-scroll logic: Only scroll if user is near bottom (avoid disruption)
- Error handling with `AgentResponseFailed` message

### 3. Message-Based State Management

Elia uses Textual messages for state coordination between widgets:

```python
class Chat(Widget):
    @dataclass
    class AgentResponseStarted(Message):
        pass

    @dataclass
    class AgentResponseComplete(Message):
        chat_id: int | None
        message: ChatMessage
        chatbox: Chatbox

    @dataclass
    class AgentResponseFailed(Message):
        last_message: ChatMessage

    @dataclass
    class NewUserMessage(Message):
        content: str
```

From `chat_screen.py`:
```python
class ChatScreen(Screen[None]):
    @on(Chat.NewUserMessage)
    def new_user_message(self, event: Chat.NewUserMessage) -> None:
        self.query_one(Chat).allow_input_submit = False
        response_status = self.query_one(ResponseStatus)
        response_status.set_awaiting_response()
        response_status.display = True

    @on(Chat.AgentResponseComplete)
    async def agent_response_complete(self, event: Chat.AgentResponseComplete) -> None:
        self.query_one(ResponseStatus).display = False
        self.query_one(Chat).allow_input_submit = True
        await self.chats_manager.add_message_to_chat(
            chat_id=event.chat_id, message=event.message
        )
```

**Pattern**: Chat widget posts messages → ChatScreen listens → Updates UI state
- Lock input during response (`allow_input_submit = False`)
- Show/hide typing indicator
- Persist to database when complete

### 4. SQLite Integration for Persistence

Elia stores all conversations locally in SQLite. The `ChatsManager` handles database operations:

**Key Operations**:
- `create_chat(chat_data)` - Create new conversation
- `add_message_to_chat(chat_id, message)` - Append message
- `rename_chat(chat_id, new_title)` - Update conversation title
- `load_chats()` - Retrieve conversation list for home screen
- Import ChatGPT history from JSON export

**Database Schema**: Stores chat metadata (title, timestamp, model) and messages (role, content, timestamp).

### 5. Custom Theming System

From `app.py`:
```python
class Elia(App[None]):
    theme: Reactive[str | None] = reactive(None, init=False)

    def __init__(self, config: LaunchConfig, startup_prompt: str = ""):
        available_themes: dict[str, Theme] = BUILTIN_THEMES.copy()
        available_themes |= load_user_themes()
        self.themes = available_themes

    def get_css_variables(self) -> dict[str, str]:
        if self.theme:
            theme = self.themes.get(self.theme)
            if theme:
                color_system = theme.to_color_system().generate()
            else:
                color_system = {}
        else:
            color_system = {}

        return {**super().get_css_variables(), **color_system}

    def watch_theme(self, theme: str | None) -> None:
        self.refresh_css(animate=False)
        self.screen._update_styles()
```

**Built-in Themes** (9 total):
- Nebula, Cobalt, Twilight, Hacker
- Alpine, Galaxy, Nautilus, Monokai, Textual

**Custom Themes**: Users can add YAML files to themes directory:
```yaml
name: example
primary: '#4e78c4'
secondary: '#f39c12'
accent: '#e74c3c'
background: '#0e1726'
surface: '#17202a'
error: '#e74c3c'
success: '#2ecc71'
warning: '#f1c40f'
```

**Pattern**: Reactive theme property → `watch_theme()` → Regenerate CSS variables → Refresh UI

### 6. Model Configuration System

From `config.py`:
```python
class EliaChatModel(BaseModel):
    name: str                    # e.g. "gpt-3.5-turbo"
    id: str | None = None        # Unique identifier for multiple instances
    display_name: str | None     # UI display name
    provider: str | None         # "OpenAI", "Anthropic", etc.
    api_key: SecretStr | None    # Override env var
    api_base: AnyHttpUrl | None  # For LocalAI/Ollama
    organization: str | None     # OpenAI org key
    description: str | None      # UI description
    product: str | None          # "ChatGPT", "Claude", etc.
    temperature: float = 1.0
    max_retries: int = 0

class LaunchConfig(BaseModel):
    default_model: str = "elia-gpt-4o"
    system_prompt: str = "You are a helpful assistant named Elia."
    message_code_theme: str = "monokai"  # Pygments theme
    models: list[EliaChatModel] = []     # User-defined models
    builtin_models: list[EliaChatModel]  # Built-in (GPT, Claude, Gemini)
    theme: str = "nebula"
```

**Built-in Models**:
- OpenAI: GPT-3.5 Turbo, GPT-4o, GPT-4 Turbo
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- Google: Gemini 1.5 Pro/Flash

**Configuration File** (TOML):
```toml
default_model = "gpt-4o"
system_prompt = "You are a helpful assistant who talks like a pirate."
theme = "galaxy"
message_code_theme = "dracula"

# Add local Ollama model
[[models]]
name = "ollama/llama3"

# LocalAI server
[[models]]
name = "openai/some-model"
api_base = "http://localhost:8080/v1"
api_key = "api-key-if-required"

# Multiple instances of same model (work vs personal)
[[models]]
id = "work-gpt-3.5-turbo"
name = "gpt-3.5-turbo"
display_name = "GPT 3.5 Turbo (Work)"

[[models]]
id = "personal-gpt-3.5-turbo"
name = "gpt-3.5-turbo"
display_name = "GPT 3.5 Turbo (Personal)"
```

**Pattern**: Pydantic models for type safety → TOML config → Merge user + builtin models

### 7. Keyboard-Centric Navigation

Elia is designed for keyboard-only usage:

**Global Bindings**:
- `q` - Quit application
- `F1` / `?` - Help screen

**Chat Screen Bindings**:
- `ESC` - Return to home screen / Focus prompt
- `Ctrl+R` - Rename conversation
- `F2` - Chat details/info
- `g` - Jump to first message
- `G` - Jump to latest message
- `Shift+Up/Down` - Scroll chat container

**Prompt Input**:
- `Enter` - Send message (or map `Cmd+Enter` via terminal)
- `Shift+Enter` - New line
- Arrow keys - Navigate input history

### 8. Inline vs Full-Screen Modes

**Full-Screen Mode** (default):
```bash
elia "Tell me about Textual"
```
- Launches chat screen directly
- Full terminal takeover
- Screen stack: HomeScreen → ChatScreen

**Inline Mode** (`-i` flag):
```bash
elia -i "Quick question"
```
- Chat appears under your shell prompt
- Non-intrusive, quick queries
- Returns to shell when done

**Implementation**: Same screen logic, different terminal rendering mode.

## Real-World Integration Examples

### LiteLLM Integration (Multi-Provider Support)

Elia uses [LiteLLM](https://github.com/BerriAI/litellm) for unified API access:

```python
import litellm
from litellm import acompletion

# Works with any provider
response = await acompletion(
    messages=[...],
    stream=True,
    model="gpt-4o",              # or "claude-3-5-sonnet-20240620"
    api_key=api_key,             # or "anthropic/claude-instant-1"
    api_base=api_base,           # or "ollama/llama3"
)
```

**Supported Providers**:
- OpenAI (ChatGPT)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (local models: Llama 3, Phi 3, Mistral, Gemma)
- LocalAI (self-hosted)
- Groq (fast inference)

### ChatGPT History Import

```bash
# Export from ChatGPT UI → conversations.json
elia import 'path/to/conversations.json'
```

**Implementation**: Parse JSON → Map to `ChatData` model → Insert into SQLite

### Response Status Indicator

From `widgets/agent_is_typing.py`:
```python
class ResponseStatus(Widget):
    def set_awaiting_response(self):
        # Show "Waiting for response..." indicator

    def set_agent_responding(self):
        # Show "Agent is responding..." with animation
```

**Pattern**: Toggle display based on message events from Chat widget

### Markdown Rendering with Syntax Highlighting

Chat messages support:
- **Markdown formatting** (Rich library)
- **Code syntax highlighting** (Pygments, configurable theme)
- **Inline code blocks**
- **Multi-line code fences**

Configuration:
```toml
message_code_theme = "dracula"  # or "monokai", "github", etc.
```

## Advanced Patterns Demonstrated

### 1. Worker Thread + UI Updates

**Challenge**: API calls block, but UI must remain responsive
**Solution**: `@work(thread=True)` + `call_from_thread()`

```python
@work(thread=True, group="agent_response")
async def stream_agent_response(self) -> None:
    # Long-running API call
    async for chunk in response:
        # Update UI from background thread
        self.app.call_from_thread(
            response_chatbox.append_chunk, chunk_content
        )
```

### 2. Reactive State with Signals

**Challenge**: Multiple widgets need runtime config updates
**Solution**: Signal pattern for pub/sub

```python
# In app.py
self.runtime_config_signal = Signal[RuntimeConfig](self, "runtime-config-updated")

# Widgets subscribe
def on_mount(self):
    self.app.runtime_config_signal.subscribe(self, self.on_config_change)
```

### 3. Dynamic Screen Stack

**Challenge**: Navigate between home, chat, help, details
**Solution**: Screen stack with push/pop

```
[HomeScreen]
    → push → [ChatScreen]
        → push → [HelpScreen] (F1)
        → pop → [ChatScreen] (ESC)
    → pop → [HomeScreen] (ESC)
```

### 4. Preventing Race Conditions

**Challenge**: User tries to send message while agent responding
**Solution**: Lock input with reactive property

```python
class Chat(Widget):
    allow_input_submit = reactive(True)

    @on(PromptInput.PromptSubmitted)
    async def user_chat_message_submitted(self, event) -> None:
        if self.allow_input_submit is True:
            await self.new_user_message(event.text)
```

Set to `False` when agent starts responding, `True` when complete.

### 5. Auto-Scroll Smart Logic

**Challenge**: Auto-scroll disrupts if user scrolled up to read history
**Solution**: Only auto-scroll if near bottom (within 3 lines)

```python
scroll_y = self.chat_container.scroll_y
max_scroll_y = self.chat_container.max_scroll_y
if scroll_y in range(max_scroll_y - 3, max_scroll_y + 1):
    self.app.call_from_thread(
        self.chat_container.scroll_end, animate=False
    )
```

### 6. Custom Widget Composition

**Chat Widget Hierarchy**:
```
Chat
├── ResponseStatus (typing indicator)
├── ChatHeader (title, model info)
├── VerticalScroll#chat-container
│   ├── Chatbox (message 1)
│   ├── Chatbox (message 2)
│   └── Chatbox (message 3, streaming)
└── ChatPromptInput (user input)
```

**Pattern**: Compose complex UI from simple widgets, use IDs for queries

## Performance Considerations

### 1. Message Trimming

```python
from litellm.utils import trim_messages

messages = trim_messages(raw_messages, model.name)
```

**Why**: Context window limits (8k, 32k, 128k tokens) - trim old messages to fit

### 2. Streaming Response Rendering

**Incremental updates** instead of full re-render:
```python
response_chatbox.append_chunk(chunk_content)
```

Chatbox accumulates chunks, updates only its content area.

### 3. Database Indexing

SQLite database uses indexes on:
- `chat_id` (for message queries)
- `timestamp` (for sorting)
- `model` (for filtering by model)

### 4. Lazy Loading

Conversations loaded on-demand:
- Home screen: Load metadata only (title, timestamp, model)
- Chat screen: Load full messages when opened

## Configuration Management

### Configuration File Location

Press `Ctrl+O` in options menu to see location:
- macOS: `~/Library/Application Support/elia/config.toml`
- Linux: `~/.config/elia/config.toml`
- Windows: `%APPDATA%\elia\config.toml`

### Themes Directory

Custom themes go in themes directory (location shown in Ctrl+O menu):
- macOS: `~/Library/Application Support/elia/themes/`
- Linux: `~/.config/elia/themes/`
- Windows: `%APPDATA%\elia\themes\`

### Database Location

SQLite database stored alongside config:
- macOS: `~/Library/Application Support/elia/conversations.db`
- Linux: `~/.config/elia/conversations.db`
- Windows: `%APPDATA%\elia\conversations.db`

### Wiping Database

```bash
elia reset
```

**Warning**: Deletes all conversations permanently.

## Lessons for Textual TUI Development

### 1. Thread Safety is Critical

When using `@work(thread=True)`, always use `call_from_thread()` to update UI:
```python
# ✅ Safe
self.app.call_from_thread(widget.update, data)

# ❌ Unsafe (race conditions, crashes)
widget.update(data)
```

### 2. Message Pattern for Complex State

Use Textual messages for state coordination:
- Decouple widgets (Chat → ChatScreen)
- Clear event flow (NewUserMessage → AgentResponseComplete)
- Easy to debug (log messages)

### 3. Reactive Properties for UI State

```python
allow_input_submit = reactive(True)
theme = reactive(None)
```

**Benefits**: Automatic UI updates, watchers for side effects

### 4. Screen Stack for Navigation

Push/pop screens instead of showing/hiding widgets:
- Cleaner navigation model
- Automatic focus management
- Memory efficient (screens destroyed on pop)

### 5. Configuration with Pydantic

Use Pydantic for type-safe config:
- Validation built-in
- Environment variable support
- Easy serialization (TOML/JSON)

### 6. Streaming API Pattern

For LLM/API streaming:
1. Create placeholder widget
2. Mount immediately (user sees it)
3. Stream chunks → append incrementally
4. Mark complete when done

**Feels snappy** - user sees immediate feedback.

### 7. Keyboard-First Design

Terminal users expect keyboard navigation:
- Vim-like bindings (g/G for navigation)
- Clear focus indicators
- ESC to go back (universal pattern)
- Help screen (F1) with all bindings

## Testing & Debugging

### Running Ollama Locally

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3

# Start server
ollama serve

# Add to elia config
[[models]]
name = "ollama/llama3"
```

### Debugging API Issues

Elia shows error notifications:
- Connection failures → Check API keys, network
- Model not found → Check model name spelling
- Rate limits → Check provider dashboard
- Context length exceeded → Message history too long

### DevTools

Textual DevTools work with Elia:
```bash
textual console
# In another terminal
elia
```

See logs, widget tree, CSS in real-time.

## Key Takeaways

**What Makes Elia Exceptional**:

1. **Production Quality** - 2.3k stars, actively maintained, real users
2. **Multi-Provider** - Works with OpenAI, Anthropic, Google, Ollama, LocalAI
3. **Streaming Done Right** - Real-time rendering with worker threads
4. **Keyboard-Centric** - Terminal-native navigation patterns
5. **Local Storage** - SQLite persistence, no cloud dependency
6. **Themeable** - 9 built-in themes + custom theme support
7. **Clean Architecture** - Screen-based design, message passing, reactive state

**Textual Patterns to Adopt**:
- Screen stack for navigation
- `@work(thread=True)` + `call_from_thread()` for async operations
- Message pattern for widget communication
- Reactive properties for UI state
- Signal pattern for pub/sub events
- Pydantic for configuration management

**Perfect Reference For**:
- API integration (streaming responses)
- Database integration (SQLite)
- Multi-screen applications
- Keyboard-focused UIs
- Custom theming systems
- Configuration management

## Sources

**GitHub Repository**:
- [Elia Main Repository](https://github.com/darrenburns/elia) (accessed 2025-11-02)
- [README.md](https://github.com/darrenburns/elia/blob/main/README.md)
- [app.py](https://github.com/darrenburns/elia/blob/main/elia_chat/app.py)
- [screens/chat_screen.py](https://github.com/darrenburns/elia/blob/main/elia_chat/screens/chat_screen.py)
- [widgets/chat.py](https://github.com/darrenburns/elia/blob/main/elia_chat/widgets/chat.py)
- [config.py](https://github.com/darrenburns/elia/blob/main/elia_chat/config.py)

**Package**:
- [PyPI: elia-chat](https://pypi.org/project/elia-chat/)

**Dependencies**:
- [Textual](https://github.com/Textualize/textual) - TUI framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Multi-provider LLM API
- [Rich](https://github.com/Textualize/rich) - Markdown rendering
- [Pydantic](https://github.com/pydantic/pydantic) - Configuration
- [Pygments](https://pygments.org/) - Syntax highlighting
