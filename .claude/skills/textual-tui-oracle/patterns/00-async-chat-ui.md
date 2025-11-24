# Async Chat UI Pattern: Building Responsive TUIs with Long-Running Processes

## Overview

This pattern demonstrates how to build a responsive chat interface in Textual that handles long-running background processes without blocking the UI. The solution combines the Observer pattern with Textual's Worker system to maintain UI responsiveness while providing real-time feedback during processing.

**Key Achievement**: Transform a frozen, unresponsive interface into a smooth, event-driven system that provides continuous feedback to users during long operations.

**Use Cases**:
- Chat interfaces with AI/NLP backends
- Data processing applications
- API integration interfaces
- File processing tools
- Any long-running operation requiring user feedback

From [Building a Responsive Textual Chat UI with Long-Running Processes](https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224) by Mehmet Öner Yalçın (accessed 2025-11-02)

---

## The Problem

### Before: Blocking Implementation

**Symptoms**:
- UI freezes completely during processing
- No indication whether system is working or stuck
- Keystrokes queue up and appear all at once when processing completes
- User anxiety: "Is it broken?"

**Code Pattern (Anti-Pattern)**:
```python
# DON'T DO THIS - Blocks UI thread
def process_message(self, message: str) -> str:
    result = self.bot.think()  # UI freezes here
    self.update_ui(result)     # Direct UI update
    return result
```

### Three Core Challenges

1. **Keep UI responsive** during long-running operations
2. **Provide real-time feedback** about processing status
3. **Maintain clean separation** between processing logic and UI

---

## The Solution: Observer + Worker Pattern

### Architecture Overview

The solution combines three design patterns:

1. **Observer Pattern**: Subject (bot) broadcasts events; observers (UI, logger) react
2. **Worker Pattern**: Background thread handles processing
3. **Thread-Safe UI Updates**: All UI updates on main thread via `call_from_thread()`

```
┌─────────────────────────────────────────────────
│ USER INPUT
│   ↓
│ WORKER (Background Thread)
│   ↓
│ OBSERVABLE (SimpleBot)
│   ├─→ OBSERVER 1 (TUI Callback)
│   │     └─→ call_from_thread() → UI Update
│   └─→ OBSERVER 2 (File Logger)
│
│ Result flows back to UI via Worker.StateChanged
└─────────────────────────────────────────────────
```

---

## Implementation Details

### 1. Observable Subject (SimpleBot)

The processing logic broadcasts events to observers:

```python
from enum import Enum
from typing import List, Callable
import time
import random

class ChatEvent(Enum):
    START_PROCESSING = "start_processing"
    THINKING = "thinking"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR = "error"

# Type alias for callbacks
ChatCallback = Callable[[ChatEvent, str], None]

class SimpleBot:
    def __init__(self) -> None:
        self.callbacks: List[ChatCallback] = []  # Our observers

    def register_callback(self, callback: ChatCallback) -> None:
        """Register an observer"""
        self.callbacks.append(callback)

    def notify_callbacks(self, event: ChatEvent, message: str) -> None:
        """Notify all observers of an event"""
        for callback in self.callbacks:
            callback.on_event(event, message)

    def process_message(self, message: str) -> str:
        """Process message with status notifications"""
        # Notify: Starting
        self.notify_callbacks(ChatEvent.START_PROCESSING, "Starting...")

        # Simulate processing steps
        time.sleep(random.uniform(0.5, 2.0))
        self.notify_callbacks(ChatEvent.THINKING, "Analyzing input...")

        # More processing
        time.sleep(random.uniform(0.5, 2.0))
        self.notify_callbacks(ChatEvent.THINKING, "Generating response...")

        # Generate result
        result = f"Response to: {message}"

        # Notify: Complete
        self.notify_callbacks(ChatEvent.PROCESSING_COMPLETE, "Done!")

        return result
```

**Key Features**:
- Broadcasts events at each processing stage
- Doesn't care who's listening (loose coupling)
- Easy to add new event types
- Clean separation from UI logic

---

### 2. Observer: TUI Callback

Updates the UI in response to bot events:

```python
from textual.widgets import Static
from textual.containers import Container

class ChatMessage(Static):
    """A single chat message widget"""
    def __init__(self, sender: str, message: str):
        super().__init__(f"{sender}: {message}")
        self.add_class(f"message-{sender.lower()}")

class TuiCallback(ChatCallback):
    """Observer that updates the TUI"""
    def __init__(self, app, message_container: Container):
        self.app = app
        self.message_container = message_container

    def on_event(self, event: ChatEvent, message: str) -> None:
        """Handle bot events - CRUCIAL: Thread-safe UI updates"""
        def update_ui() -> None:
            if event == ChatEvent.START_PROCESSING:
                self.message_container.mount(
                    ChatMessage("System", f"🚀 {message}")
                )
            elif event == ChatEvent.THINKING:
                self.message_container.mount(
                    ChatMessage("Bot", f"💭 {message}")
                )
            elif event == ChatEvent.PROCESSING_COMPLETE:
                self.message_container.mount(
                    ChatMessage("System", f"✅ {message}")
                )
            elif event == ChatEvent.ERROR:
                self.message_container.mount(
                    ChatMessage("System", f"❌ {message}")
                )

        # CRITICAL: Update UI safely from main thread
        self.app.call_from_thread(update_ui)
```

**Key Features**:
- `call_from_thread()` ensures thread safety
- All UI updates happen on main thread
- Different visual indicators for event types
- Non-blocking observer execution

---

### 3. Observer: File Logger

Records events to file for debugging:

```python
import logging
from datetime import datetime

class FileLoggerCallback(ChatCallback):
    """Observer that logs events to file"""
    def __init__(self, log_file: str = "chat_events.log"):
        self.logger = logging.getLogger("ChatLogger")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def on_event(self, event: ChatEvent, message: str) -> None:
        """Log event - safe to run in background thread"""
        self.logger.info(f"{event.value}: {message}")
```

**Key Features**:
- Completely independent from UI
- Safe to run in any thread (file I/O)
- Easy debugging and monitoring
- No impact on UI performance

---

### 4. Textual App with Worker Integration

The main application coordinates everything:

```python
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Header, Footer, Input, Button
from textual.worker import Worker, WorkerState

class ChatApp(App):
    """Responsive chat application with background processing"""

    CSS = """
    #message-container {
        height: 1fr;
        overflow-y: auto;
    }

    #input-container {
        height: auto;
        dock: bottom;
    }

    .message-bot {
        background: $secondary;
        padding: 1;
        margin: 1;
    }

    .message-user {
        background: $primary;
        padding: 1;
        margin: 1;
    }

    .message-system {
        color: $text-muted;
        padding: 1;
    }
    """

    def __init__(self):
        super().__init__()
        # Initialize bot
        self.bot = SimpleBot()

        # Register observers
        # TUI callback will be registered in compose()
        # File logger
        file_logger = FileLoggerCallback()
        self.bot.register_callback(file_logger)

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        # Message display area
        with ScrollableContainer(id="message-container"):
            yield Container(id="messages")

        # Input area
        with Container(id="input-container"):
            yield Input(placeholder="Type a message...", id="user-input")
            yield Button("Send", id="send-button")

        yield Footer()

    def on_mount(self) -> None:
        """App initialization"""
        # Register TUI callback now that widgets exist
        message_container = self.query_one("#messages")
        tui_callback = TuiCallback(self, message_container)
        self.bot.register_callback(tui_callback)

        # Focus input
        self.query_one("#user-input").focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button"""
        if event.button.id == "send-button":
            self.send_message()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input"""
        self.send_message()

    def send_message(self) -> None:
        """Send user message and start background processing"""
        input_widget = self.query_one("#user-input", Input)
        user_input = input_widget.value.strip()

        if not user_input:
            return

        # Display user message
        message_container = self.query_one("#messages")
        message_container.mount(ChatMessage("User", user_input))

        # Clear input
        input_widget.value = ""

        # Start background processing
        self.process_message_in_background(user_input)

    def process_message_in_background(self, user_input: str) -> Worker:
        """
        Start worker for background processing.
        Returns Worker object for tracking.
        """
        worker = self.run_worker(
            lambda: self.bot.process_message(user_input),
            name="bot_processing",
            thread=True,  # Run in background thread
        )
        return worker

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """
        Handle worker completion in main thread.
        CRUCIAL: This runs on main thread, safe for UI updates.
        """
        if event.worker.name == "bot_processing":
            if event.worker.state == WorkerState.SUCCESS:
                # Safe to update UI - we're in the main thread
                message_container = self.query_one("#messages")
                message_container.mount(
                    ChatMessage("Bot", event.worker.result)
                )
            elif event.worker.state == WorkerState.ERROR:
                message_container = self.query_one("#messages")
                message_container.mount(
                    ChatMessage("System", f"Error: {event.worker.error}")
                )

if __name__ == "__main__":
    app = ChatApp()
    app.run()
```

---

## Thread Safety: The Golden Rule

### The Problem

Every UI framework has the same rule: **Don't update UI from background threads**. It causes chaos, like trying to help someone solve a puzzle while they're still arranging pieces.

### The Solution: `call_from_thread()`

Textual provides `call_from_thread()` to safely schedule UI updates on the main thread:

```python
# WRONG - Direct UI update from background thread
def on_event(self, event: ChatEvent, message: str) -> None:
    self.message_container.mount(ChatMessage("Bot", message))  # ❌ CRASH

# RIGHT - Schedule UI update on main thread
def on_event(self, event: ChatEvent, message: str) -> None:
    def update_ui() -> None:
        self.message_container.mount(ChatMessage("Bot", message))

    self.app.call_from_thread(update_ui)  # ✅ SAFE
```

### Worker Lifecycle Events

Workers emit events that run on the main thread - safe for direct UI updates:

```python
def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
    """This method runs on main thread - safe for UI updates"""
    if event.worker.state == WorkerState.SUCCESS:
        # Direct UI update - we're on main thread
        self.display_result(event.worker.result)
```

**Worker States**:
- `PENDING`: Worker created but not started
- `RUNNING`: Processing in background
- `SUCCESS`: Completed successfully (result available)
- `ERROR`: Failed (error available)
- `CANCELLED`: Cancelled before completion

---

## Performance & Benefits

### Measurements

From the article's real-world implementation:

**Before (Blocking)**:
- UI freezes: 2-10 seconds per message
- User feedback: "Feels frozen"
- CPU usage: 100% on main thread during processing

**After (Async)**:
- UI responsiveness: 0ms freeze time
- Status update latency: ~200ms average
- CPU usage: Efficient thread utilization
- Memory overhead: Minimal (observer pattern)

### User Experience Improvements

Key feedback from users:

> "It's like night and day. Before, we'd click and hope. Now we can actually see the system thinking. It's not faster, but it feels faster because we know what's happening."

**Improvements**:
- Clear processing status indicators
- Ability to queue multiple requests
- Confidence in system operation
- Improved debugging through logged events

---

## Pattern Benefits

### 1. Separation of Concerns

```python
# Processing logic (SimpleBot)
- No UI dependencies
- Pure business logic
- Easy to test

# UI logic (TuiCallback)
- No processing logic
- Pure presentation
- Easy to swap UI frameworks

# Logging (FileLoggerCallback)
- Independent monitoring
- No coupling to anything
```

### 2. Extensibility

Adding new observers is trivial:

```python
# Add metrics tracking
class MetricsCallback(ChatCallback):
    def on_event(self, event: ChatEvent, message: str) -> None:
        self.metrics.increment(f"chat.{event.value}")

# Register it
bot.register_callback(MetricsCallback())
```

### 3. Thread Safety

Architecture enforces thread safety:
- Workers run in background threads
- Observers use `call_from_thread()` for UI updates
- Worker lifecycle events on main thread
- Clear boundaries prevent threading bugs

### 4. Real-Time Feedback

Users see processing progress:
- "Starting..." → "Analyzing input..." → "Generating response..." → "Done!"
- Visual indicators (emojis: 🚀 💭 ✅ ❌)
- Confidence that system is working

---

## Broader Applications

This pattern combination works well for:

### File Processing
```python
class FileProcessor:
    def process_files(self, file_list: List[str]) -> None:
        self.notify_callbacks(Event.START, "Processing files...")
        for i, file in enumerate(file_list):
            self.notify_callbacks(Event.PROGRESS, f"File {i+1}/{len(file_list)}")
            self.process_single_file(file)
        self.notify_callbacks(Event.COMPLETE, "All files processed!")
```

### Data Analysis
```python
class DataAnalyzer:
    def analyze(self, dataset: pd.DataFrame) -> Results:
        self.notify_callbacks(Event.START, "Loading data...")
        # ... load ...
        self.notify_callbacks(Event.PROGRESS, "Computing statistics...")
        # ... compute ...
        self.notify_callbacks(Event.PROGRESS, "Generating visualizations...")
        # ... visualize ...
        self.notify_callbacks(Event.COMPLETE, "Analysis complete!")
```

### API Integration
```python
class APIClient:
    def batch_request(self, items: List[Item]) -> None:
        self.notify_callbacks(Event.START, "Starting batch...")
        for i, item in enumerate(items):
            self.notify_callbacks(Event.PROGRESS, f"Request {i+1}/{len(items)}")
            response = self.api_call(item)
            self.notify_callbacks(Event.UPDATE, f"Got response: {response.status}")
        self.notify_callbacks(Event.COMPLETE, "Batch complete!")
```

---

## Practical Tips

### 1. Worker Management

Let Textual handle worker lifecycle:

```python
# Good - Framework manages lifecycle
worker = self.run_worker(
    lambda: self.bot.process_message(user_input),
    name="bot_processing",
    thread=True
)

# Don't manually manage threads
# threading.Thread(target=...).start()  # ❌ Avoid
```

### 2. Event Granularity

Choose appropriate event detail level:

```python
# Too coarse - not useful
class ChatEvent(Enum):
    PROCESSING = "processing"  # What kind?

# Too fine - overwhelming
class ChatEvent(Enum):
    PARSING_INPUT = "parsing"
    TOKENIZING = "tokenizing"
    EMBEDDING = "embedding"
    SEARCHING = "searching"
    # ... 20 more events ...

# Just right - meaningful stages
class ChatEvent(Enum):
    START_PROCESSING = "start"
    THINKING = "thinking"
    PROCESSING_COMPLETE = "complete"
    ERROR = "error"
```

### 3. Error Handling

Always handle worker errors:

```python
def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
    if event.worker.state == WorkerState.ERROR:
        # Log error
        self.log.error(f"Worker failed: {event.worker.error}")

        # Notify user
        self.notify(
            f"Processing failed: {event.worker.error}",
            severity="error"
        )

        # Cleanup if needed
        self.cleanup_failed_operation()
```

### 4. Progress Indicators

Use different visual styles for different event types:

```python
EVENT_STYLES = {
    ChatEvent.START_PROCESSING: ("🚀", "blue"),
    ChatEvent.THINKING: ("💭", "yellow"),
    ChatEvent.PROCESSING_COMPLETE: ("✅", "green"),
    ChatEvent.ERROR: ("❌", "red"),
}

def display_event(self, event: ChatEvent, message: str) -> None:
    emoji, color = EVENT_STYLES[event]
    widget = ChatMessage(f"{emoji} {message}")
    widget.styles.color = color
    self.mount(widget)
```

---

## Common Gotchas

### 1. Forgetting Thread Safety

```python
# ❌ WRONG - Direct UI update from observer
class TuiCallback(ChatCallback):
    def on_event(self, event: ChatEvent, message: str) -> None:
        self.message_container.mount(ChatMessage("Bot", message))

# ✅ RIGHT - Use call_from_thread()
class TuiCallback(ChatCallback):
    def on_event(self, event: ChatEvent, message: str) -> None:
        def update_ui():
            self.message_container.mount(ChatMessage("Bot", message))
        self.app.call_from_thread(update_ui)
```

### 2. Worker Name Collisions

```python
# ❌ WRONG - All workers named "processing"
self.run_worker(task1, name="processing")
self.run_worker(task2, name="processing")  # Collision!

# ✅ RIGHT - Unique names
self.run_worker(task1, name=f"processing_{uuid.uuid4()}")
# Or track by reference
worker = self.run_worker(task1)
self.active_workers[worker] = metadata
```

### 3. Forgetting to Clear Input

```python
# ❌ WRONG - Input stays after sending
def send_message(self):
    user_input = self.query_one("#user-input").value
    self.process(user_input)

# ✅ RIGHT - Clear input after sending
def send_message(self):
    input_widget = self.query_one("#user-input")
    user_input = input_widget.value
    input_widget.value = ""  # Clear
    self.process(user_input)
```

---

## Testing Strategies

### 1. Test Observable in Isolation

```python
def test_bot_notifies_observers():
    bot = SimpleBot()
    events_received = []

    def mock_callback(event, message):
        events_received.append((event, message))

    bot.register_callback(mock_callback)
    bot.process_message("test")

    assert ChatEvent.START_PROCESSING in [e[0] for e in events_received]
    assert ChatEvent.PROCESSING_COMPLETE in [e[0] for e in events_received]
```

### 2. Test Observers Independently

```python
def test_file_logger_callback():
    with tempfile.NamedTemporaryFile(mode='w') as f:
        logger = FileLoggerCallback(f.name)
        logger.on_event(ChatEvent.THINKING, "Test message")

        # Check log file
        with open(f.name) as log:
            content = log.read()
            assert "thinking: Test message" in content
```

### 3. Test UI Updates (Mock App)

```python
def test_tui_callback_calls_main_thread():
    mock_app = Mock()
    mock_container = Mock()
    callback = TuiCallback(mock_app, mock_container)

    callback.on_event(ChatEvent.THINKING, "Test")

    # Verify call_from_thread was called
    mock_app.call_from_thread.assert_called_once()
```

---

## Full Example: Complete Chat Application

See the article's complete working example on GitHub:
[github.com/oneryalcin/blog_textual_observer](https://github.com/oneryalcin/blog_textual_observer)

Key files:
- `simple_bot.py` - Observable bot with event broadcasting
- `callbacks.py` - TUI and file logger observers
- `chat_app.py` - Main Textual application
- `tests/` - Unit tests for each component

---

## Key Learnings Summary

### Architecture
- **Observer Pattern**: Loose coupling between processing and presentation
- **Worker Pattern**: Background processing without blocking UI
- **Thread Safety**: Always use `call_from_thread()` for UI updates

### User Experience
- Real-time feedback transforms user perception
- "Feels faster" is as important as "is faster"
- Visual indicators reduce user anxiety

### Code Quality
- Clean separation of concerns
- Easy to test each component independently
- Easy to add new features (just add observers)
- Clear flow of information

### The Golden Rule
> "It's not about eliminating the waiting, it's about making the waiting informative and keeping your application responsive throughout."

---

## Sources

**Primary Source**:
- [Building a Responsive Textual Chat UI with Long-Running Processes](https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224) - Medium article by Mehmet Öner Yalçın (accessed 2025-11-02)

**Code Repository**:
- [github.com/oneryalcin/blog_textual_observer](https://github.com/oneryalcin/blog_textual_observer) - Complete working example

**Related Concepts**:
- Observer Pattern (design pattern)
- Worker threads (concurrency)
- Thread-safe UI updates (GUI programming)
- Event-driven architecture (software design)

---

## See Also

Within textual-tui-oracle:
- `architecture/03-workers-threads.md` - Detailed Worker documentation
- `core-concepts/02-reactivity.md` - Reactive programming in Textual
- `integration/` - Database and I/O integration patterns (future)
- `tutorials/` - Step-by-step application tutorials (future)
