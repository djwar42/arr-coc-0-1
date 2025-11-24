# THREADING THEORY: Textual Worker System Deep Analysis

**Study of Threading Patterns from Canonical Textual Examples**

*All patterns extracted directly from source code implementation - worker.py, worker_manager.py, _work_decorator.py*

---

## Table of Contents

1. [Core Architecture: How Workers Actually Work](#1-core-architecture-how-workers-actually-work)
2. [Worker Types: Thread vs Async](#2-worker-types-thread-vs-async)
3. [Worker Lifecycle: States and Transitions](#3-worker-lifecycle-states-and-transitions)
4. [Worker Groups: Exclusivity and Cancellation](#4-worker-groups-exclusivity-and-cancellation)
5. [Thread Safety: The call_from_thread() Pattern](#5-thread-safety-the-call_from_thread-pattern)
6. [The @work Decorator: Under the Hood](#6-the-work-decorator-under-the-hood)
7. [Message System: StateChanged Events](#7-message-system-statechanged-events)
8. [Pattern #1: Streaming API Responses (Chat)](#8-pattern-1-streaming-api-responses-chat)
9. [Pattern #2: Async File Loading (DirectoryTree)](#9-pattern-2-async-file-loading-directorytree)
10. [Pattern #3: HTTP Fetching (Demo Home)](#10-pattern-3-http-fetching-demo-home)
11. [Pattern #4: Observer + Worker Combo](#11-pattern-4-observer--worker-combo)
12. [Anti-Patterns: What NOT to Do](#12-anti-patterns-what-not-to-do)
13. [General Form: The Universal Pattern](#13-general-form-the-universal-pattern)

---

## 1. Core Architecture: How Workers Actually Work

### The Worker Class (worker.py)

A Worker is a **wrapper around work** that provides:
- State management (PENDING → RUNNING → SUCCESS/ERROR/CANCELLED)
- Progress tracking (completed_steps, total_steps)
- Result storage
- Cancellation support
- Message emission on state changes

```python
# From worker.py - The Worker constructor
class Worker(Generic[ResultType]):
    def __init__(
        self,
        node: DOMNode,           # Widget/Screen/App that owns this worker
        work: WorkType,          # The actual work (callable/coroutine/awaitable)
        *,
        name: str = "",          # Identifier for debugging
        group: str = "default",  # Group for exclusive workers
        description: str = "",   # Longer description
        exit_on_error: bool = True,  # Exit app on error?
        thread: bool = False,    # Run in thread vs asyncio task?
    ) -> None:
        self._node = node
        self._work = work
        self._state = WorkerState.PENDING
        self.cancelled_event: Event = Event()  # threading.Event for cancellation
        self._thread_worker = thread
        self._result: ResultType | None = None
        self._task: asyncio.Task | None = None

        # IMMEDIATELY post state change message!
        self._node.post_message(self.StateChanged(self, self._state))
```

**Key Insight #1**: A Worker is created with PENDING state and immediately posts a message. This enables UI to know work is queued.

### The Two Execution Paths

```python
# From worker.py - The run() method
async def run(self) -> ResultType:
    return await (
        self._run_threaded() if self._thread_worker else self._run_async()
    )
```

**Thread Worker** (`thread=True`):
- Uses `loop.run_in_executor(None, runner, self._work)`
- Runs in ThreadPoolExecutor (OS thread)
- Can call blocking code (network I/O, file I/O, CPU work)
- Must use `call_from_thread()` for UI updates

**Async Worker** (`thread=False`, default):
- Uses `await self._work()` or `await self._work`
- Runs in asyncio event loop
- Shares thread with UI
- Can update UI directly (but still shouldn't block!)

---

## 2. Worker Types: Thread vs Async

### Thread Worker Execution (Most Common for I/O)

```python
# From worker.py - _run_threaded() method
async def _run_threaded(self) -> ResultType:
    def run_callable(work: Callable[[], ResultType]) -> ResultType:
        """Set the active worker, and call the callable."""
        active_worker.set(self)  # Set context var in thread
        return work()

    def run_coroutine(work: Callable[[], Coroutine[...]]) -> ResultType:
        """Set the active worker and await coroutine."""
        async def do_work() -> ResultType:
            active_worker.set(self)
            return await work
        return asyncio.run(do_work())  # New event loop in thread!

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, runner, self._work)
```

**Key Insight #2**: Thread workers that are async coroutines get their OWN event loop via `asyncio.run()`. This is why you can use `async for` in a thread worker!

### Why Thread Workers for API Calls?

```python
# Example: Elia streaming API call
@work(thread=True, group="agent_response")
async def stream_agent_response(self) -> None:
    response = await acompletion(...)  # This runs in ITS OWN event loop
    async for chunk in response:       # Streaming works!
        self.app.call_from_thread(...)
```

The thread worker:
1. Creates a new thread
2. Creates a new asyncio event loop in that thread
3. Runs your async code in that loop
4. Main UI thread stays responsive

**This is WHY `@work(thread=True)` + async works!**

### Async Worker Execution (For Concurrent Textual Tasks)

```python
# From worker.py - _run_async() method
async def _run_async(self) -> ResultType:
    if inspect.iscoroutinefunction(self._work):
        return await self._work()
    elif inspect.isawaitable(self._work):
        return await self._work
```

Use async workers when:
- Work is already non-blocking (pure Textual operations)
- You want multiple Textual things to happen concurrently
- Work doesn't need separate thread isolation

---

## 3. Worker Lifecycle: States and Transitions

### The Five States

```python
# From worker.py
class WorkerState(enum.Enum):
    PENDING = 1   # Created but not started
    RUNNING = 2   # Actively executing
    CANCELLED = 3 # Was cancelled before completion
    ERROR = 4     # Raised an exception
    SUCCESS = 5   # Completed normally
```

### State Transition Diagram

```
                    ┌───────────────┐
                    │   PENDING     │
                    │  (created)    │
                    └───────┬───────┘
                            │ _start()
                            ▼
                    ┌───────────────┐
              ┌─────│   RUNNING     │─────┐
              │     │  (executing)  │     │
              │     └───────┬───────┘     │
              │             │             │
        cancel()      normal exit    exception
              │             │             │
              ▼             ▼             ▼
      ┌───────────┐ ┌───────────┐ ┌───────────┐
      │ CANCELLED │ │  SUCCESS  │ │   ERROR   │
      └───────────┘ └───────────┘ └───────────┘
```

### State Changes Emit Messages

```python
# From worker.py - state property setter
@state.setter
def state(self, state: WorkerState) -> None:
    changed = state != self._state
    self._state = state
    if changed:
        self._node.post_message(self.StateChanged(self, state))
```

**Key Insight #3**: Every state change posts a message. Your app receives `Worker.StateChanged` events on the main thread, safe for UI updates!

---

## 4. Worker Groups: Exclusivity and Cancellation

### The Exclusive Pattern

```python
# From worker_manager.py - add_worker()
def add_worker(
    self, worker: Worker, start: bool = True, exclusive: bool = True
) -> None:
    if exclusive and worker.group:
        self.cancel_group(worker.node, worker.group)  # Cancel existing!
    self._workers.add(worker)
    if start:
        worker._start(self._app, self._remove_worker)
```

**Key Insight #4**: When `exclusive=True` (default for decorator), starting a new worker in the same group cancels all existing workers in that group!

### Group Cancellation

```python
# From worker_manager.py
def cancel_group(self, node: DOMNode, group: str) -> list[Worker]:
    workers = [
        worker
        for worker in self._workers
        if (worker.group == group and worker.node == node)
    ]
    for worker in workers:
        worker.cancel()
    return workers
```

**Use Case**: User types new query while previous API call is running → cancel old, start new.

```python
# Chat app pattern
@work(thread=True, group="agent_response")  # Same group = exclusive
async def stream_response(self, prompt):
    ...

# When user sends new message:
# 1. Old worker in "agent_response" group is cancelled
# 2. New worker starts
# 3. User sees new response instead of old
```

### Checking for Cancellation

```python
# From worker.py
@property
def is_cancelled(self) -> bool:
    return self._cancelled

# Inside a worker, check if you should stop:
from textual.worker import get_current_worker

worker = get_current_worker()
if worker.is_cancelled:
    return  # Stop processing

# Or use the threading.Event:
if self.cancelled_event.is_set():
    return
```

---

## 5. Thread Safety: The call_from_thread() Pattern

### Why Thread Safety Matters

The UI is single-threaded. All widget updates must happen on the main thread.

When a thread worker tries to update the UI directly:
- Race conditions
- Corrupted state
- Crashes
- Visual glitches

### The Solution: call_from_thread()

```python
# Pattern from Elia chat app
@work(thread=True)
async def stream_response(self):
    async for chunk in response:
        # WRONG - Direct update from background thread
        # self.widget.update(chunk)  # CRASH!

        # CORRECT - Schedule on main thread
        self.app.call_from_thread(
            self.widget.update, chunk
        )
```

### How call_from_thread() Works

```python
# Conceptually:
def call_from_thread(self, callback, *args, **kwargs):
    # Post callback to main event loop
    self._loop.call_soon_threadsafe(
        lambda: callback(*args, **kwargs)
    )
```

It schedules the callback to run on the main asyncio event loop, which is the UI thread.

### Multiple Arguments Pattern

```python
# Mount a widget from thread
self.app.call_from_thread(
    container.mount,     # Method to call
    ChatMessage("Bot", content)  # Arguments
)

# Scroll to end
self.app.call_from_thread(
    container.scroll_end,
    animate=False  # Keyword argument
)
```

---

## 6. The @work Decorator: Under the Hood

### Decorator Flow

```python
# From _work_decorator.py
def work(
    method=None, *,
    name="", group="default", exit_on_error=True,
    exclusive=False, description=None, thread=False,
) -> ...:
    def decorator(method):
        # IMPORTANT: Non-async functions MUST be thread workers!
        if not iscoroutinefunction(method) and not thread:
            raise WorkerDeclarationError(
                "Can not create a worker from a non-async function "
                "unless `thread=True` is set"
            )

        @wraps(method)
        def decorated(*args, **kwargs):
            self = args[0]  # First arg is the widget/app

            # Create description for debugging
            debug_description = f"{method.__name__}(...)"

            # THIS IS THE KEY - calls run_worker!
            worker = self.run_worker(
                partial(method, *args, **kwargs),
                name=name or method.__name__,
                group=group,
                exclusive=exclusive,
                exit_on_error=exit_on_error,
                thread=thread,
            )
            return worker  # Returns Worker object!

        return decorated
    return decorator
```

**Key Insight #5**: The @work decorator transforms your method into one that returns a Worker object. The original method becomes the work.

### Decorator Usage Patterns

```python
# Pattern 1: Simple decorator (async required)
@work
async def fetch_data(self):
    ...

# Pattern 2: With parameters
@work(thread=True, exclusive=True, group="api")
async def fetch_data(self):
    ...

# Pattern 3: Thread worker for sync code
@work(thread=True)
def blocking_operation(self):
    time.sleep(5)  # This is fine in a thread worker
    ...
```

### Return Value

```python
# The decorated method returns a Worker!
worker = self.fetch_data()
print(worker.state)  # WorkerState.RUNNING

# You can await the result
result = await worker.wait()

# Or let it complete in background and handle via message
```

---

## 7. Message System: StateChanged Events

### The StateChanged Message

```python
# From worker.py
class Worker:
    @rich.repr.auto
    class StateChanged(Message, bubble=False, namespace="worker"):
        def __init__(self, worker: Worker, state: WorkerState):
            self.worker = worker
            self.state = state
            super().__init__()
```

**Key Insight #6**: `bubble=False` means the message stays on the node that created the worker. It doesn't bubble up the DOM tree.

### Handling State Changes

```python
# In your App or Screen
def on_worker_state_changed(self, event: Worker.StateChanged):
    """This runs on the MAIN THREAD - safe for UI updates!"""

    if event.worker.name == "api_call":
        if event.worker.state == WorkerState.SUCCESS:
            # Safe to update UI!
            self.display_result(event.worker.result)
        elif event.worker.state == WorkerState.ERROR:
            self.show_error(event.worker.error)
        elif event.worker.state == WorkerState.CANCELLED:
            self.show_cancelled()
```

### Message Flow

```
Thread Worker               Main Thread
─────────────               ───────────
work()
  │
  ├── (work completes)
  │
  └── state = SUCCESS
        │
        └──────────────────► StateChanged posted
                                   │
                                   ▼
                            on_worker_state_changed()
                                   │
                                   └── Update UI (safe!)
```

---

## 8. Pattern #1: Streaming API Responses (Chat)

**Source**: Elia Chat Client - THE MOST THREADED EXAMPLE

### The Pattern

```python
class Chat(Widget):
    @work(thread=True, group="agent_response")
    async def stream_agent_response(self) -> None:
        # 1. Make API call
        response = await acompletion(
            messages=messages,
            stream=True,
            model=model.name,
        )

        # 2. Create placeholder widget IMMEDIATELY
        response_chatbox = Chatbox(...)
        self.app.call_from_thread(
            self.chat_container.mount, response_chatbox
        )

        # 3. Stream chunks incrementally
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                self.app.call_from_thread(
                    response_chatbox.append_chunk, content
                )

                # 4. Smart auto-scroll
                scroll_y = self.chat_container.scroll_y
                max_scroll_y = self.chat_container.max_scroll_y
                if scroll_y >= max_scroll_y - 3:  # Near bottom
                    self.app.call_from_thread(
                        self.chat_container.scroll_end,
                        animate=False
                    )

        # 5. Post completion message
        self.post_message(self.AgentResponseComplete(...))
```

### Why This Works

1. **`thread=True`**: API call gets its own thread + event loop
2. **`group="agent_response"`**: New queries cancel old ones (exclusive)
3. **`call_from_thread()`**: Every UI update is thread-safe
4. **Smart scroll**: Only auto-scroll if user hasn't scrolled up
5. **Incremental append**: Fast updates, user sees progress

### General Form

```python
@work(thread=True, group="<operation_type>")
async def stream_operation(self):
    # 1. Start long-running operation
    response = await external_api_call(stream=True)

    # 2. Create UI placeholder immediately
    self.app.call_from_thread(container.mount, placeholder)

    # 3. Stream updates
    async for chunk in response:
        self.app.call_from_thread(
            placeholder.update, chunk
        )

    # 4. Signal completion
    self.post_message(OperationComplete(...))
```

---

## 9. Pattern #2: Async File Loading (DirectoryTree)

**Source**: widgets/_directory_tree.py

### The Pattern

```python
class DirectoryTree(Tree[DirEntry]):
    def __init__(self, path):
        self._load_queue: Queue[TreeNode[DirEntry]] = Queue()
        super().__init__(...)

    def _add_to_load_queue(self, node: TreeNode[DirEntry]) -> AwaitComplete:
        """Queue a node for loading."""
        if not node.data.loaded:
            node.data.loaded = True
            self._load_queue.put_nowait(node)
        return AwaitComplete(self._load_queue.join())

    @work
    async def _loader(self) -> None:
        """Background loader that processes the queue."""
        while True:
            node = await self._load_queue.get()
            try:
                content = await self._load_directory(node).wait()
                self._populate_node(node, content)
            except (WorkerCancelled, WorkerFailed):
                pass
            finally:
                self._load_queue.task_done()

    @work
    async def _load_directory(self, node: TreeNode[DirEntry]) -> Iterable[Path]:
        """Load a single directory."""
        path = node.data.path
        return sorted(
            path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
```

### Why This Works

1. **Queue-based loading**: Nodes are queued, processed in order
2. **Worker per operation**: Each directory load is a worker
3. **Cancellation handling**: Graceful handling of cancelled loads
4. **AwaitComplete**: Callers can optionally await completion

---

## 10. Pattern #3: HTTP Fetching (Demo Home)

**Source**: source-code/demo/home.py

### The Pattern

```python
class StarCount(Vertical):
    stars = reactive(25251, recompose=True)
    forks = reactive(776, recompose=True)

    @work
    async def get_stars(self):
        """Worker to get stars from GitHub API."""
        self.loading = True
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/repos/textualize/textual"
                )
                data = response.json()

            # Update reactive properties - triggers recompose
            self.stars = data["stargazers_count"]
            self.forks = data["forks"]
        except Exception:
            self.notify("Unable to update", severity="error")
        self.loading = False

    def on_mount(self) -> None:
        self.get_stars()  # Start worker on mount

    def on_click(self) -> None:
        self.get_stars()  # Refresh on click
```

### Why This Works

1. **`@work` (no thread=True)**: Uses async worker because httpx is async
2. **Reactive properties**: `stars = reactive(...)` → UI auto-updates
3. **`recompose=True`**: Entire widget rebuilds when value changes

---

## 11. Pattern #4: Observer + Worker Combo

**Source**: patterns/00-async-chat-ui.md

### The Observer Pattern

```python
class ChatEvent(Enum):
    START_PROCESSING = "start_processing"
    THINKING = "thinking"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR = "error"

class SimpleBot:
    def __init__(self):
        self.callbacks: List[Callable] = []

    def notify_callbacks(self, event: ChatEvent, message: str):
        for callback in self.callbacks:
            callback.on_event(event, message)

    def process_message(self, message: str) -> str:
        self.notify_callbacks(ChatEvent.START_PROCESSING, "Starting...")
        time.sleep(1)
        self.notify_callbacks(ChatEvent.THINKING, "Analyzing...")
        time.sleep(1)
        self.notify_callbacks(ChatEvent.PROCESSING_COMPLETE, "Done!")
        return f"Response to: {message}"
```

### The TUI Callback (Thread-Safe)

```python
class TuiCallback:
    def __init__(self, app, container):
        self.app = app
        self.container = container

    def on_event(self, event: ChatEvent, message: str):
        def update_ui():
            self.container.mount(ChatMessage("Bot", message))

        # CRITICAL: Thread-safe update!
        self.app.call_from_thread(update_ui)
```

### Why This Works

1. **Separation of concerns**: Bot doesn't know about UI
2. **Multiple observers**: TUI, file logger, metrics all listen
3. **Thread-safe callbacks**: Each observer uses `call_from_thread()`
4. **Real-time updates**: UI shows each processing phase

---

## 12. Anti-Patterns: What NOT to Do

### Anti-Pattern #1: Direct UI Updates from Thread

```python
# WRONG - Will crash or corrupt state
@work(thread=True)
def fetch_data(self):
    result = api.call()
    self.result_widget.update(result)  # CRASH!

# CORRECT
@work(thread=True)
def fetch_data(self):
    result = api.call()
    self.app.call_from_thread(self.result_widget.update, result)
```

### Anti-Pattern #2: Blocking in Async Workers

```python
# WRONG - Blocks the UI thread!
@work
def fetch_data(self):
    result = requests.get(url)  # Blocking!
    return result

# CORRECT - Use thread worker
@work(thread=True)
def fetch_data(self):
    result = requests.get(url)
    return result
```

### Anti-Pattern #3: Non-Async Function Without thread=True

```python
# WRONG - Will raise WorkerDeclarationError
@work
def blocking_operation(self):
    time.sleep(5)
    return result

# CORRECT
@work(thread=True)
def blocking_operation(self):
    time.sleep(5)
    return result
```

### Anti-Pattern #4: Ignoring Cancellation

```python
# WRONG - Keeps running after cancellation
@work(thread=True)
def long_operation(self):
    for i in range(1000000):
        do_something()  # Never checks cancellation

# CORRECT - Check cancellation periodically
@work(thread=True)
def long_operation(self):
    worker = get_current_worker()
    for i in range(1000000):
        if worker.is_cancelled:
            return  # Stop processing!
        do_something()
```

---

## 13. General Form: The Universal Pattern

### The Complete Pattern

```python
from textual.app import App
from textual.worker import Worker, WorkerState, get_current_worker
from textual import work

class MyApp(App):

    @work(
        thread=True,        # For blocking/IO work
        group="operation",  # For exclusive execution
        exclusive=True,     # Cancel previous in group
    )
    async def do_background_work(self, input_data):
        worker = get_current_worker()

        # 1. Setup - Create UI placeholder
        placeholder = ResultWidget()
        self.app.call_from_thread(self.container.mount, placeholder)

        try:
            # 2. Do work with progress updates
            for i, item in enumerate(input_data):
                if worker.is_cancelled:
                    return None

                result = await process_item(item)
                self.app.call_from_thread(
                    placeholder.update_progress, i + 1, len(input_data)
                )

            return final_result

        except Exception as e:
            self.app.call_from_thread(
                self.notify, str(e), severity="error"
            )
            return None

    def on_worker_state_changed(self, event: Worker.StateChanged):
        if event.worker.group != "operation":
            return

        match event.worker.state:
            case WorkerState.SUCCESS:
                self.display_result(event.worker.result)
            case WorkerState.ERROR:
                self.notify(f"Error: {event.worker.error}", severity="error")
            case WorkerState.CANCELLED:
                self.notify("Operation cancelled")
```

### Pattern Decision Tree

```
Need background work?
    │
    ├── Is it blocking I/O (network, file, CPU)?
    │       └── YES → @work(thread=True)
    │
    ├── Is it async library (httpx, aiofiles)?
    │       └── YES → @work (no thread flag)
    │
    └── Should new work cancel old work?
            ├── YES → Add group="name", exclusive=True
            └── NO → Default (group="default", exclusive=False)
```

### Thread Safety Checklist

- [ ] All UI updates use `call_from_thread()`
- [ ] Worker checks `is_cancelled` in long loops
- [ ] Errors are handled (try/catch or StateChanged)
- [ ] Thread workers don't wait for other workers
- [ ] Non-async functions use `thread=True`

---

## Summary: The Three Golden Rules

### Rule 1: Thread Safety
**ALL UI updates from thread workers must use `call_from_thread()`**

### Rule 2: Worker Type Selection
- **Blocking code** → `@work(thread=True)`
- **Async code** → `@work`
- **Non-async function** → Must use `thread=True`

### Rule 3: State Management
- **Use Worker groups** for cancellation/exclusivity
- **Handle StateChanged** for completion/errors
- **Check is_cancelled** in long operations

---

*Document generated from direct source code analysis of Textual worker.py, worker_manager.py, _work_decorator.py, _directory_tree.py, demo/home.py, and patterns from Elia chat client.*

**Last Updated**: 2025-11-19
**Source**: textual-tui-oracle/source-code/
