# Textual Reactivity Guide - Official

Comprehensive guide to reactive attributes and their superpowers in Textual applications.

## Overview

Textual's reactive attributes are attributes **with superpowers**. They enable automatic UI updates, validation, computed properties, and intelligent state management without manual DOM manipulation. Reactive attributes use Python's descriptor protocol (same as `property` decorator) to provide declarative, efficient state handling.

From [Reactivity - Textual Official Documentation](https://textual.textualize.io/guide/reactivity/) (accessed 2025-11-02):
> With great power comes great responsibility. - Uncle Ben

## Creating Reactive Attributes

Reactive attributes are defined at the class scope using the `reactive()` function from `textual.reactive`:

```python
from textual.reactive import reactive
from textual.widget import Widget

class MyWidget(Widget):
    name = reactive("Paul")        # String with default "Paul"
    count = reactive(0)            # Integer with default 0
    is_cool = reactive(True)       # Boolean with default True
```

**Key Points:**
- No need to modify `__init__()` method
- Get and set like normal attributes: `self.name = "Jessica"`, `self.count += 1`
- Type hints optional if default value provided
- Type hints useful for optional attributes: `name: reactive[str | None] = reactive("Paul")`

### Dynamic Defaults

Defaults can be callable (functions). Textual calls the function to get the initial value:

```python
from time import time
from textual.reactive import reactive

class Timer(Widget):
    start_time = reactive(time)  # Called when widget created
```

## Superpower 1: Smart Refresh

When you modify a reactive attribute, Textual automatically:
1. Detects the change
2. Checks if value actually changed (prevents unnecessary refreshes)
3. Calls the widget's `render()` method
4. Updates the display

**Important:** If multiple reactive attributes change, Textual batches them into a single refresh to minimize updates.

From official examples (Textual docs/guide/reactivity/refresh01.py):
```python
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Static

class Name(Widget):
    who = reactive("World")

    def render(self) -> str:
        return f"Hello, {self.who}!"

# When who attribute changes, render() is called automatically
```

### Disabling Refresh: Using `var` Instead

If you want reactive superpowers without automatic refresh, use `var()`:

```python
from textual.reactive import var

class MyWidget(Widget):
    count = var(0)  # Changing count won't trigger refresh or layout
```

### Layout Updates

By default, refresh updates content but not widget size. Set `layout=True` for layout recalculation:

```python
class MyWidget(Widget):
    title = reactive("Title", layout=True)  # Updates both content AND size
```

## Superpower 2: Validation

Validation methods check and potentially modify values before assignment. Define methods named `validate_<attribute>`:

```python
class Counter(Widget):
    count = reactive(0)

    def validate_count(self, value: int) -> int:
        """Keep count between 0 and 10"""
        return max(0, min(value, 10))

# When you do: self.count = 15
# Validation runs: validate_count(15) returns 10
# Actual value set: self.count becomes 10
```

**Common Use Cases:**
- Range restrictions (min/max values)
- Type coercion
- Path validation
- Enum verification

## Superpower 3: Watch Methods

Watch methods react to reactive attribute changes. Define methods named `watch_<attribute>`:

**Single Argument Form:**
```python
class ColorWidget(Widget):
    color = reactive("red")

    def watch_color(self, new_value: str) -> None:
        """Called when color changes, receives new value"""
        self.styles.background = new_value
```

**Two Argument Form (Receives Old & New Values):**
```python
def watch_color(self, old_value: str, new_value: str) -> None:
    """Called with both old and new values"""
    print(f"Color changed from {old_value} to {new_value}")
```

### When Watch Methods Are Called

- **Called:** When reactive value **actually changes**
- **NOT called:** When new value equals old value
- **Override behavior:** Pass `always_update=True` to reactive

```python
class Widget(Widget):
    # This watcher will be called even if value doesn't change
    value = reactive(0, always_update=True)

    def watch_value(self, new_value: int) -> None:
        pass
```

### Dynamic Watch Methods

Programmatically add watchers using `watch()` method:

```python
class MyApp(App):
    def on_mount(self) -> None:
        counter = self.query_one(Counter)
        # Add a callback for counter.count changes
        counter.watch(Counter.count, self.on_counter_change)

    def on_counter_change(self, value: int) -> None:
        print(f"Counter changed to {value}")
```

## Superpower 4: Recompose

Alternative to refresh: remove all child widgets and call `compose()` again. Set `recompose=True`:

```python
class MyWidget(Widget):
    content = reactive("default", recompose=True)

    def compose(self):
        yield Label(self.content)  # Re-created each time content changes
```

**Recompose vs Refresh:**
- **Refresh:** Updates content only (more efficient)
- **Recompose:** Removes and recreates all children (simpler code, slightly less efficient)

**When to Use Recompose:**
- When widget structure changes based on state
- When you want cleaner code (no separate watch methods)
- When performance isn't critical

**When NOT to Use Recompose:**
- Stateful widgets (Input, DataTable, TextArea) lose state
- Rapidly changing values (recomposing is more expensive)
- Many child widgets

## Superpower 5: Compute Methods

Compute methods calculate derived values, like properties with caching. Define methods named `compute_<attribute>`:

```python
class ColorMixer(Widget):
    red = reactive(0)
    green = reactive(0)
    blue = reactive(0)

    color = reactive()  # Define computed reactive

    def compute_color(self) -> Color:
        """Combine RGB components into Color object"""
        return Color(self.red, self.green, self.blue)

    def watch_color(self, new_color: Color) -> None:
        """Called when computed color changes"""
        self.styles.background = new_color
```

**How Compute Methods Work:**
1. Textual calls compute method when attribute accessed
2. Result is cached
3. Cached value invalidated when ANY reactive attribute changes
4. Watch methods called if computed result changed

**Important Notes:**
- Avoid slow/CPU-intensive operations in compute methods (called frequently)
- Compute methods called BEFORE validate methods, which called BEFORE watch methods
- Like Python `property` decorator but with dependency tracking and caching

## Advanced Patterns

### Setting Reactives Without Superpowers

Use `set_reactive()` to set values without triggering watchers. Useful in constructors before mounting:

```python
class Greeter(Widget):
    greeting = reactive("Hello")

    def __init__(self, greeting: str):
        super().__init__()
        # Using set_reactive avoids calling watchers before mount
        self.set_reactive(Greeter.greeting, greeting)

    def watch_greeting(self) -> None:
        # This could break if widget not mounted yet
        self.query_one(Label).update(self.greeting)
```

**When to Use:**
- In `__init__()` before widget is mounted
- When you need to set internal state without side effects

### Mutable Reactives

Textual detects reassignment but NOT mutations of collections:

```python
class MyWidget(Widget):
    items = reactive([])

# This triggers reactivity:
self.items = [1, 2, 3]

# This does NOT (mutation, not reassignment):
self.items.append(4)  # Won't trigger refresh!
```

**Fix: Use `mutate_reactive()`:**
```python
from textual.dom import DOMNode

class MyWidget(Widget):
    items = reactive([])

    def add_item(self, item):
        self.items.append(item)
        self.mutate_reactive(MyWidget.items)  # Triggers reactivity
```

### Data Binding

Connect reactive attributes between parent and child widgets. Changes in parent automatically update child:

```python
class MyApp(App):
    time = reactive(datetime.now())

    def compose(self):
        clock = WorldClock()
        # Bind app.time to clock.time
        yield clock.data_bind(time=MyApp.time)

    def on_mount(self):
        # Update app.time every second
        self.set_interval(self.update_time, 1)

    def update_time(self):
        self.time = datetime.now()  # Automatically updates all bound clocks
```

**Data Binding Rules:**
- One-directional only (parent → child)
- Supports same-name binding: `data_bind(MyApp.time)`
- Supports different-name binding: `data_bind(clock_time=MyApp.time)`
- Simplifies state management in compound widgets

## Reactive Parameters

The `reactive()` function accepts several parameters:

```python
name = reactive(
    default="Paul",           # Default value (required, positional)
    init=True,               # Call watchers on initialize post-mount
    always_update=False,     # Call watchers even if value unchanged
    layout=False,            # Update widget size when changed
    recompose=False,         # Recreate compose() when changed
    compute=None,            # Custom compute method reference
)
```

## Common Patterns & Best Practices

### Pattern 1: Counter with Bounds

```python
class Counter(Widget):
    count = reactive(0)
    min_value = 0
    max_value = 10

    def validate_count(self, value: int) -> int:
        return max(self.min_value, min(value, self.max_value))

    def watch_count(self, old: int, new: int) -> None:
        self.update_display()
```

### Pattern 2: Search with Debounce

```python
class SearchBox(Widget):
    query = reactive("")

    def watch_query(self, query: str) -> None:
        # Debounce search (call after user stops typing)
        self.remove_timer("search")
        self.set_timer("search", 0.5, self.perform_search)

    def perform_search(self) -> None:
        results = search_db(self.query)
        self.update_results(results)
```

### Pattern 3: Validated Input

```python
class EmailInput(Widget):
    email = reactive("")

    def validate_email(self, value: str) -> str:
        if "@" not in value and value != "":
            raise ValueError("Invalid email")
        return value

    def watch_email(self, new_email: str) -> None:
        self.styles.border = ("solid", "green" if "@" in new_email else "red")
```

### Pattern 4: Computed Derived State

```python
class Calculator(Widget):
    a = reactive(0)
    b = reactive(0)
    operation = reactive("+")

    result = reactive()

    def compute_result(self):
        if self.operation == "+":
            return self.a + self.b
        elif self.operation == "-":
            return self.a - self.b
        return 0
```

## Performance Considerations

**Efficient Reactivity:**
- Textual batches multiple reactive changes into single refresh
- Computed methods cached until dependencies change
- Watch methods only called if value actually changed
- Use `var()` for non-UI state (no refresh overhead)

**Optimization Tips:**
1. Use `var()` for internal state that doesn't affect UI
2. Keep compute methods simple and fast
3. Use layout updates sparingly (more expensive than refresh)
4. Batch related changes before assignment
5. Avoid expensive operations in watch methods

## Sources

**Official Textual Documentation:**
- [Reactivity - Textual Guide](https://textual.textualize.io/guide/reactivity/) (accessed 2025-11-02)
- GitHub Source: [docs/guide/reactivity.md](https://github.com/Textualize/textual/blob/main/docs/guide/reactivity.md)

**API Reference:**
- [textual.reactive.reactive](https://textual.textualize.io/api/reactive/) - Reactive attribute descriptor
- [textual.reactive.var](https://textual.textualize.io/api/reactive/) - Non-refreshing reactive
- [DOMNode.watch()](https://textual.textualize.io/api/dom_node/) - Dynamic watchers
- [DOMNode.set_reactive()](https://textual.textualize.io/api/dom_node/) - Set without superpowers
- [DOMNode.mutate_reactive()](https://textual.textualize.io/api/dom_node/) - Trigger mutation detection
- [DOMNode.data_bind()](https://textual.textualize.io/api/dom_node/) - Data binding

**Related Documentation:**
- [Textual Tutorial - Reactive Attributes](https://textual.textualize.io/tutorial/)
- [Official Examples](https://github.com/Textualize/textual/tree/main/docs/examples/guide/reactivity)

## See Also

- [core-concepts/01-widgets-and-containers.md](01-widgets-and-containers.md) - Widget architecture
- [core-concepts/02-styling-and-themes.md](02-styling-and-themes.md) - CSS styling with reactivity
- [core-concepts/03-event-system.md](03-event-system.md) - Events and reactive interaction
