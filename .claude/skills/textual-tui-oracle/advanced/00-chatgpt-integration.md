# ChatGPT Integration with Textual TUI

## Overview

Integrating ChatGPT with Textual creates powerful terminal-based chat interfaces. This guide covers API integration patterns, async handling, and UI structure for building ChatGPT TUIs.

## Key Architecture Components

### 1. Async API Integration with httpx

**Core Pattern**: Use `httpx.AsyncClient` for non-blocking OpenAI API calls.

From [ChatGPT_TUI/chat_api.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/chat_api.py):

```python
import httpx
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
timeout = httpx.Timeout(60.0)  # Long timeout for extended responses

async def get_openai_response(content) -> tuple[bool, str]:
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": content
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return (True, response.json()['choices'][0]['message']['content'])
        except Exception as e:
            return (False, repr(e))
```

**Key Points**:
- **60 second timeout**: ChatGPT responses can take time for long messages
- **Tuple return**: `(success: bool, content: str)` pattern for error handling
- **Environment variable**: Store API key in `OPENAI_API_KEY` env var
- **Exception handling**: Catch all errors and return as error messages

### 2. Custom Input Widget with Send Button

**Pattern**: Combine Input field + Button in a container widget with custom message passing.

From [ChatGPT_TUI/widgets.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/widgets.py):

```python
from textual.widgets import Input, Button, Static
from textual.message import Message, MessageTarget
from textual.binding import Binding

class UserInput(Static):
    """Widget where the user types messages."""

    BINDINGS = [
        Binding('enter', 'send', 'Send', show=True, priority=False),
    ]

    class Utterance(Message):
        """Custom message for user input."""
        def __init__(self, sender: MessageTarget, text: str) -> None:
            self.text = text
            super().__init__(sender)

    def compose(self) -> ComposeResult:
        yield Input(name="input", placeholder="Type here...", classes="user_input_field")
        yield Button("Send", classes="user_send_button")

    async def action_send(self) -> None:
        """Called when user presses Enter."""
        await self.send()

    async def on_button_pressed(self) -> None:
        """Called when user clicks Send button."""
        await self.send()

    async def send(self) -> None:
        input_widget = self.query_one(Input)
        text = input_widget.value
        await self.post_message(self.Utterance(self, text))
        input_widget.value = ""  # Clear input after sending
```

**Key Patterns**:
- **Custom Message class**: `Utterance` carries user input through app
- **Multiple triggers**: Both Enter key and button click call `send()`
- **Input clearing**: Reset input field after sending
- **Async message posting**: Use `post_message()` for event-driven communication

### 3. Message Display Widget

**Pattern**: Render messages with agent labels (User/Assistant/Error) using Markdown.

From [ChatGPT_TUI/widgets.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/widgets.py):

```python
from textual.widgets import Static, Markdown
from chatgpt_tui.structures import Agent

class AgentMessage(Static):
    """Widget that renders a message from an agent."""

    def __init__(self, agent: Agent, message: str = "") -> None:
        super().__init__()
        self.agent = agent
        self.message = message

    def compose(self) -> ComposeResult:
        if self.agent == Agent.User:
            yield Static("You:")
        elif self.agent == Agent.Bot:
            yield Static("Assistant:")
        elif self.agent == Agent.ERROR:
            yield Static("Error:")

        yield Markdown(self.message)
```

**Key Points**:
- **Agent enum**: Distinguish between User, Bot, System, Error messages
- **Markdown rendering**: ChatGPT responses often include code blocks and formatting
- **Label + Content pattern**: Each message shows who sent it + the content

### 4. Main App Structure with Chat History

**Pattern**: Container for messages + async response handling.

From [ChatGPT_TUI/app.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/app.py):

```python
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Input
from textual.binding import Binding

class ChatApp(App):
    CSS_PATH = "style.css"
    history: History = History()  # Maintains conversation context

    BINDINGS = [
        Binding('ctrl+d', 'toggle_dark', 'Toggle dark mode'),
        Binding('ctrl+y', 'yank', 'Yank latest message'),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(*self.compose_history(), id="chat_container")
        yield UserInput()
        yield Static("Enter: Send | Ctrl+y: Yank | Ctrl+d: Dark mode")

    def on_mount(self) -> None:
        """Focus input field on startup."""
        self.query_one(Input).focus()

    def compose_history(self) -> ComposeResult:
        """Render all previous messages."""
        history_widgets = []
        for agent, utterance in self.history.data:
            history_widgets.append(AgentMessage(agent, utterance))
        return history_widgets

    async def on_user_input_utterance(self, message: UserInput.Utterance) -> None:
        """Handle user message and get bot response."""
        # Add user message to history and UI
        self.history.add_utterance(Agent.User, message.text)
        self.query_one("#chat_container").mount(AgentMessage(Agent.User, message.text))

        # Get and display bot response
        await self.add_bot_response()

    async def add_bot_response(self):
        """Call OpenAI API and display response."""
        succeed, bot_response = await get_openai_response(self.history.export())

        if succeed:
            self.history.add_utterance(Agent.Bot, bot_response)
            self.query_one("#chat_container").mount(AgentMessage(Agent.Bot, bot_response))
        else:
            self.history.add_utterance(Agent.ERROR, bot_response)
            self.query_one("#chat_container").mount(AgentMessage(Agent.ERROR, bot_response))
```

**Key Patterns**:
- **History object**: Maintains conversation context for OpenAI API (required for multi-turn chat)
- **Dynamic mounting**: Use `.mount()` to add new messages to chat container
- **Message handler**: `on_user_input_utterance()` catches custom Utterance events
- **Error display**: Show API errors as ERROR agent messages in chat
- **Auto-focus**: Input field gets focus on app startup

### 5. History Management

**Pattern**: Store conversation as list of `(agent, message)` tuples, export to OpenAI format.

```python
from enum import Enum

class Agent(Enum):
    User = "user"
    Bot = "assistant"
    System = "system"
    ERROR = "error"

class History:
    def __init__(self):
        self.data: list[tuple[Agent, str]] = []

    def add_utterance(self, agent: Agent, utterance: str):
        self.data.append((agent, utterance))

    def export(self) -> list[dict]:
        """Convert to OpenAI API message format."""
        return [
            {"role": agent.value, "content": utterance}
            for agent, utterance in self.data
            if agent in (Agent.User, Agent.Bot, Agent.System)
        ]
```

**Key Points**:
- **Enum for agents**: Type-safe agent identification
- **Export method**: Convert internal format to OpenAI API format
- **Filter errors**: Don't send ERROR messages to API

## Best Practices

### API Key Management

```bash
# Set environment variable
export OPENAI_API_KEY=sk-...

# Check in Python
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")
```

### Async/Await Patterns

**DO**:
- Use `async def` for API calls
- Use `await` when calling async functions
- Use `httpx.AsyncClient` (not `requests`)
- Handle long timeouts (60+ seconds)

**DON'T**:
- Block the UI thread with synchronous API calls
- Forget to handle API errors
- Use small timeouts (ChatGPT can take 20-30+ seconds)

### UI Responsiveness

```python
# Good: Async API call doesn't block UI
async def add_bot_response(self):
    succeed, response = await get_openai_response(self.history.export())
    # UI stays responsive during await
    self.query_one("#chat_container").mount(AgentMessage(Agent.Bot, response))

# Bad: Synchronous call blocks entire app
def add_bot_response(self):
    response = requests.post(...)  # UI freezes!
    self.query_one("#chat_container").mount(AgentMessage(Agent.Bot, response))
```

### Message Formatting

- **Use Markdown widget**: ChatGPT often returns formatted text, code blocks
- **Preserve formatting**: Don't strip newlines or markdown syntax
- **Error handling**: Display API errors as distinct ERROR messages

## Additional Features

### Copy Latest Message (Yank)

From [ChatGPT_TUI/app.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/app.py):

```python
import pyperclip

def action_yank(self) -> None:
    """Copy latest message to clipboard."""
    if self.history.data:
        try:
            pyperclip.copy(self.history.data[-1][1])
        except pyperclip.PyperclipException as e:
            self.history.add_utterance(Agent.ERROR, repr(e))
            self.query_one("#chat_container").mount(AgentMessage(Agent.ERROR, repr(e)))
```

### Dark Mode Toggle

```python
def action_toggle_dark(self) -> None:
    self.dark = not self.dark
```

## Installation and Setup

From [ChatGPT_TUI README](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/README.md):

```bash
# Install
pip install chatgpt_tui

# Export API key
export OPENAI_API_KEY=your-api-key

# Run
catui
```

## Alternative Implementations

### chatui (ttyobiwan)

From [chatui README](https://github.com/ttyobiwan/chatui/blob/main/README.md):

- Uses virtual environment setup
- Development-focused (includes dev.txt requirements)
- Similar architecture pattern
- 25 stars, active development (2023)

**Setup**:
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r ./requirements/dev.txt
export OPENAI_KEY=<key>
make run
```

## Common Pitfalls

### 1. Timeout Errors
**Problem**: Default timeout too short for long ChatGPT responses.
**Solution**: Set `httpx.Timeout(60.0)` or higher.

### 2. UI Freezing
**Problem**: Using synchronous HTTP calls.
**Solution**: Use `httpx.AsyncClient` with `async/await`.

### 3. Missing Conversation Context
**Problem**: Each message sent without previous messages.
**Solution**: Maintain `History` object and export full conversation to API.

### 4. Error Display
**Problem**: API errors crash app or disappear.
**Solution**: Catch exceptions, display as ERROR agent messages.

### 5. Input Focus
**Problem**: User must click input field after sending.
**Solution**: Call `self.query_one(Input).focus()` after clearing input.

## Sources

**Primary Implementation**:
- [ChatGPT_TUI Repository](https://github.com/Jiayi-Pan/ChatGPT_TUI) - Complete working implementation (accessed 2025-11-02)
  - [app.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/app.py) - Main app structure
  - [chat_api.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/chat_api.py) - OpenAI API integration
  - [widgets.py](https://github.com/Jiayi-Pan/ChatGPT_TUI/blob/main/chatgpt_tui/widgets.py) - Custom input and message widgets

**Alternative Implementation**:
- [chatui by ttyobiwan](https://github.com/ttyobiwan/chatui) - Similar ChatGPT TUI (accessed 2025-11-02)

**Original Tutorial** (unavailable):
- [Using Textual to Build a ChatGPT TUI App](https://chaoticengineer.hashnode.dev/textual-and-chatgpt) - Hashnode article by The Chaotic Engineer (published 2023-03-15, attempted access 2025-11-02 - site experiencing technical issues)

**Related Projects**:
- [awesome-textual-projects](https://github.com/oleksis/awesome-textualize-projects) - Curated list including ChatGPT TUIs
- [written-in-textual](https://github.com/matan-h/written-in-textual) - Collection of Textual apps including elia (ChatGPT client)

## See Also

- [../getting-started/01-official-tutorial.md](../getting-started/01-official-tutorial.md) - Textual basics
- [../core-concepts/](../core-concepts/) - Async patterns, message passing
- [../widgets/](../widgets/) - Input, Button, Container, Markdown widgets
