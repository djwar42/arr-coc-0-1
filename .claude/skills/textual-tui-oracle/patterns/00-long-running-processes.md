# Long-Running Processes Pattern in Textual

## Overview

Building responsive TUI applications requires careful management of long-running operations (processing, API calls, file I/O). This document covers battle-tested patterns for keeping your Textual UI responsive while executing blocking operations in the background.

From [Building a Responsive Textual Chat UI with Long-Running Processes](https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224) by Mehmet Öner Yalçın, accessed 2025-11-02.

---

## The Core Problem

### UI Freezing During Long Operations

**Symptom**: When a user triggers a long-running operation (data processing, API call), the entire UI becomes unresponsive:
- No cursor blinking
- Keystrokes don't register (or appear all at once when done)
- No progress feedback
- User thinks application is broken

**Root Cause**: Long operations run on the main thread, blocking event handling and rendering.

```python
# Problem code: UI freezes during processing
class ChatApp(App):
    def on_button_press(self):
        # This blocks the entire UI
        response = self.bot.process_message(user_input)  # Takes 2-10 seconds
        self.display_response(response)  # Never reached until done
```

---

## Solution Architecture: Observer Pattern + Workers

### The Pattern

Combine three components:

1. **Background Worker**: Executes long operation on separate thread
2. **Observer/Callback System**: Long operation broadcasts status updates
3. **Thread-Safe UI Updates**: Updates UI safely from main thread

### Architecture Diagram

```
User Input Event
    │
    ├─→ [Main Thread]
    │   └─→ Create Worker
    │   └─→ Return immediately (UI responsive)
    │
    └─→ [Background Thread]
        └─→ Long Operation (2-10 seconds)
            └─→ notify_callbacks(START)
            └─→ [Processing...]
            └─→ notify_callbacks(THINKING)
            └─→ [More work...]
            └─→ notify_callbacks(COMPLETE)
            │
            └─→ [Main Thread] (via call_from_thread)
                └─→ Update UI safely
```

---

## Implementation: Chat UI Example

### Step 1: Define Event Types

```python
from enum import Enum

class ChatEvent(Enum):
    """Events broadcast during processing."""
    START_PROCESSING = "start_processing"
    THINKING = "thinking"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR = "error"
```

### Step 2: Observer Pattern - Define Callbacks

```python
from abc import ABC, abstractmethod
from typing import List, Callable

class ChatCallback(ABC):
    """Observer interface for chat events."""

    @abstractmethod
    def on_event(self, event: ChatEvent, message: str) -> None:
        """Called when bot emits an event."""
        pass


class TuiCallback(ChatCallback):
    """Update Textual UI when bot processes messages."""

    def __init__(self, app, message_container):
        self.app = app
        self.message_container = message_container

    def on_event(self, event: ChatEvent, message: str) -> None:
        """Handle chat events, updating UI safely."""

        def update_ui() -> None:
            """UI update function (runs on main thread)."""
            if event == ChatEvent.START_PROCESSING:
                self.message_container.mount(
                    ChatMessage("System", "Processing...")
                )
            elif event == ChatEvent.THINKING:
                self.message_container.mount(
                    ChatMessage("Bot", f"💭 {message}")
                )
            elif event == ChatEvent.PROCESSING_COMPLETE:
                self.message_container.mount(
                    ChatMessage("System", "Done!")
                )

        # CRITICAL: Call from thread ensures UI updates on main thread
        self.app.call_from_thread(update_ui)


class FileLogCallback(ChatCallback):
    """Also log events to file for debugging."""

    def on_event(self, event: ChatEvent, message: str) -> None:
        with open('chat.log', 'a') as f:
            f.write(f"{event.value}: {message}\n")
```

### Step 3: Processing Engine with Callbacks

```python
class SimpleBot:
    """Bot that broadcasts events as it processes."""

    def __init__(self):
        self.callbacks: List[ChatCallback] = []

    def register_callback(self, callback: ChatCallback):
        """Observer registers itself."""
        self.callbacks.append(callback)

    def notify_callbacks(self, event: ChatEvent, message: str):
        """Broadcast event to all observers."""
        for callback in self.callbacks:
            callback.on_event(event, message)

    def process_message(self, user_input: str) -> str:
        """Long-running operation that broadcasts progress."""

        # Notify: starting
        self.notify_callbacks(ChatEvent.START_PROCESSING, "Beginning processing...")

        # Simulate work
        import time, random
        time.sleep(random.uniform(0.5, 2.0))

        # Notify: intermediate step
        self.notify_callbacks(ChatEvent.THINKING, "Analyzing input...")

        # More work
        time.sleep(random.uniform(1.0, 3.0))

        # Generate response
        response = f"Bot response to: {user_input}"

        # Notify: complete
        self.notify_callbacks(ChatEvent.PROCESSING_COMPLETE, "Ready for next message")

        return response
```

### Step 4: Textual App with Worker Management

```python
from textual.app import App, ComposeResult
from textual.widgets import Container, Input
from textual.worker import Worker, WorkerState

class ChatUIApp(App):
    """Responsive chat UI using workers."""

    def __init__(self):
        super().__init__()
        self.bot = SimpleBot()

    async def on_mount(self) -> None:
        """Initialize UI and register observers."""

        # Set up UI
        self.message_container = Container(id="messages")
        self.input_field = Input(placeholder="Type your message...")

        # Register observer
        tui_callback = TuiCallback(self, self.message_container)
        self.bot.register_callback(tui_callback)

        # Also log for debugging
        self.bot.register_callback(FileLogCallback())

    def on_input_submit(self, event) -> None:
        """User submitted a message."""
        user_input = event.control.value
        event.control.value = ""  # Clear input

        # Display user message immediately
        self.message_container.mount(
            ChatMessage("User", user_input)
        )

        # Process in background - return immediately
        self.process_message_in_background(user_input)

    def process_message_in_background(self, user_input: str) -> None:
        """Start background worker for message processing."""

        # run_worker handles thread creation and lifecycle
        worker = self.run_worker(
            lambda: self.bot.process_message(user_input),
            name="bot_processing",
            thread=True  # Run in separate thread
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion (called on main thread)."""

        if event.worker.is_finished:
            # Worker finished successfully
            result = event.worker.result

            # Safe to update UI - we're on main thread
            self.message_container.mount(
                ChatMessage("Bot", result)
            )

    def on_worker_error(self, event: Worker.Error) -> None:
        """Handle worker errors."""

        self.bot.notify_callbacks(
            ChatEvent.ERROR,
            f"Error: {event.exception}"
        )


def compose(self) -> ComposeResult:
    yield self.message_container
    yield self.input_field
```

---

## Key Principles

### 1. Main Thread Only for UI Updates

```python
# WRONG: Update UI from background thread (race conditions!)
def background_work():
    self.label.update("Done!")  # ❌ May crash or corrupt UI

# RIGHT: Use call_from_thread
def background_work():
    self.app.call_from_thread(
        lambda: self.label.update("Done!")  # ✓ Safe
    )
```

### 2. Use Textual's Worker System

```python
# Textual's run_worker handles thread safety
worker = self.run_worker(
    long_function,
    name="my_worker",
    thread=True  # Separate thread
)

# Completion happens safely on main thread
def on_worker_state_changed(self, event: Worker.StateChanged):
    if event.worker.is_finished:
        result = event.worker.result  # Safe to access
        self.update_ui_with_result(result)  # Safe to update UI
```

### 3. Design for Responsiveness

```python
# ❌ Bad: Blocks after 1 second input
def process(self, data):
    time.sleep(1)  # User sees freeze
    return transform(data)

# ✓ Good: User sees feedback immediately
def process(self, data):
    self.notify_callbacks(ChatEvent.START, "Beginning...")
    time.sleep(0.1)

    self.notify_callbacks(ChatEvent.THINKING, "Analyzing...")
    time.sleep(0.5)

    self.notify_callbacks(ChatEvent.STEP2, "Processing...")
    time.sleep(0.4)

    return transform(data)
    # User sees 3 progress updates, not 1 freeze
```

---

## Real-World Patterns

### Pattern 1: File Processing with Progress

```python
class FileProcessor:
    def __init__(self):
        self.callbacks = []

    def process_file(self, filepath: str):
        """Process large file with progress updates."""

        self.notify(ChatEvent.START_PROCESSING, f"Reading {filepath}")
        lines = open(filepath).readlines()
        total = len(lines)

        for i, line in enumerate(lines):
            # Process line
            result = self.transform_line(line)

            # Periodic progress update (not every line - too noisy)
            if i % 100 == 0:
                progress = f"{i}/{total} ({100*i/total:.1f}%)"
                self.notify(ChatEvent.THINKING, progress)

        self.notify(ChatEvent.PROCESSING_COMPLETE, "File processed")
        return results

    def notify(self, event, msg):
        for cb in self.callbacks:
            cb.on_event(event, msg)
```

### Pattern 2: API Integration with Streaming

```python
class APIClient:
    def fetch_data_streaming(self, endpoint: str):
        """Fetch data in chunks with progress."""

        self.notify(ChatEvent.START_PROCESSING, f"Connecting to {endpoint}")
        response = requests.get(endpoint, stream=True)

        chunks_received = 0
        for chunk in response.iter_content(chunk_size=4096):
            chunks_received += 1
            self.notify(
                ChatEvent.THINKING,
                f"Downloaded {chunks_received * 4}KB..."
            )
            process_chunk(chunk)

        self.notify(ChatEvent.PROCESSING_COMPLETE, "Transfer complete")
```

### Pattern 3: Long Computation with Cancellation

```python
from threading import Event

class ComputeWorker:
    def __init__(self):
        self.cancel_flag = Event()
        self.callbacks = []

    def long_computation(self, data):
        """Computation that respects cancellation."""

        self.cancel_flag.clear()
        self.notify(ChatEvent.START_PROCESSING, "Computing...")

        results = []
        for i, item in enumerate(data):
            # Check for cancellation
            if self.cancel_flag.is_set():
                self.notify(ChatEvent.ERROR, "Cancelled by user")
                return None

            result = expensive_compute(item)
            results.append(result)

            if i % 10 == 0:
                self.notify(ChatEvent.THINKING, f"Step {i}/{len(data)}")

        self.notify(ChatEvent.PROCESSING_COMPLETE, "Done")
        return results

    def cancel(self):
        """Signal worker to stop."""
        self.cancel_flag.set()
```

---

## Testing Long-Running Operations

```python
# Test that UI remains responsive during processing
def test_ui_responsive_during_processing():
    """Verify UI doesn't freeze while bot processes."""

    app = ChatUIApp()
    async with app.run_test() as pilot:
        # Send message
        await pilot.press("tab")  # Focus input
        await pilot.press(*"hello")
        await pilot.press("enter")

        # Immediately try to type again (should work)
        await pilot.press("t")

        # Check that input wasn't blocked
        assert app.input_field.value == "t"  # ✓ Responsive

        # Wait for processing to complete
        await pilot.pause(timeout=15)  # Wait for bot

        # Verify response appeared
        messages = app.query(".chat-message")
        assert len(messages) >= 3  # User msg, processing updates, response


# Test observer callbacks
def test_callbacks_fired():
    """Verify all progress events are reported."""

    events_received = []

    class TestCallback(ChatCallback):
        def on_event(self, event, msg):
            events_received.append(event)

    bot = SimpleBot()
    bot.register_callback(TestCallback())

    bot.process_message("test")

    # Verify event sequence
    assert ChatEvent.START_PROCESSING in events_received
    assert ChatEvent.THINKING in events_received
    assert ChatEvent.PROCESSING_COMPLETE in events_received
```

---

## Performance Considerations

### Thread Pool Management

```python
# Textual manages worker threads automatically
# But be aware of constraints:

# ❌ Don't create unlimited workers
for item in huge_list:
    self.run_worker(process_item(item))  # Spawns 1M threads - disaster!

# ✓ Limit concurrent workers
MAX_WORKERS = 5
active_workers = []

for item in huge_list:
    while len(active_workers) >= MAX_WORKERS:
        active_workers = [w for w in active_workers if not w.is_finished]
        time.sleep(0.1)

    worker = self.run_worker(process_item(item))
    active_workers.append(worker)
```

### Memory Usage During Processing

```python
# ❌ Bad: Load entire file in memory
def process_file(filepath):
    all_data = open(filepath).read()  # 1GB file = 1GB memory
    return process_all(all_data)

# ✓ Good: Process in streaming fashion
def process_file(filepath):
    for line in open(filepath):
        process_line(line)  # Constant memory usage
```

---

## Sources

**Primary Source**:
- [Building a Responsive Textual Chat UI with Long-Running Processes](https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224) - Mehmet Öner Yalçın (November 8, 2024)
  - Observer pattern implementation
  - Worker thread management
  - Thread-safe UI updates
  - Real-world chat UI architecture

**Textual Official Resources**:
- [Textual Worker Documentation](https://textual.textualize.io/guide/workers/) - Background task management
- [Textual Threading Safety](https://textual.textualize.io/guide/async/) - Async and threading patterns

---

## Related Documentation

- [02-lessons-learned.md](../best-practices/02-lessons-learned.md) - Terminal performance optimization
- [01-project-templates.md](./01-project-templates.md) - Project structure for TUI apps
- [widgets/00-widget-patterns.md](../widgets/00-widget-patterns.md) - Custom widget patterns
