# Practical Optimization Techniques for Textual Apps

**Related**: [00-high-performance-algorithms.md](00-high-performance-algorithms.md)

---

## Overview

This guide provides practical, actionable optimization techniques for building high-performance Textual terminal applications. Each technique includes when to use it, how to implement it, and expected performance gains.

---

## 1. Optimize Widget Updates

### Use Reactive Variables

**When**: Widget state changes frequently and UI needs to stay in sync.

**How**:
```python
from textual.reactive import reactive

class MyWidget(Widget):
    count = reactive(0)  # Automatically triggers refresh on change

    def watch_count(self, old_value, new_value):
        """Called automatically when count changes"""
        # Only update what changed
        self.query_one("#counter").update(str(new_value))
```

**Benefits**:
- Automatic, targeted updates
- No manual refresh() calls
- Textual handles dirty region tracking

---

### Batch Updates with batch_update()

**When**: Multiple widget properties change at once.

**How**:
```python
from textual import work

class DataDisplay(Widget):
    @work(exclusive=True)
    async def update_all_data(self, new_data):
        """Batch multiple updates into single repaint"""
        with self.batch_update():
            self.title = new_data['title']
            self.status = new_data['status']
            self.items = new_data['items']
            # Single repaint after context exits
```

**Benefits**:
- Single layout/render cycle instead of multiple
- Reduced flicker and CPU usage

---

## 2. Virtualize Large Lists

### Implement Virtual Scrolling

**When**: Displaying hundreds or thousands of items (logs, data tables, file lists).

**Problem**: Rendering 10,000 widgets tanks performance.

**Solution**: Only render visible items + small buffer.

**Example pattern**:
```python
class VirtualList(Widget):
    items = reactive([])  # Full dataset
    visible_start = reactive(0)
    visible_count = 20  # Items visible at once

    def render_visible_items(self):
        """Only render items in viewport"""
        end = self.visible_start + self.visible_count
        visible = self.items[self.visible_start:end]

        # Render only these items
        for item in visible:
            yield ItemWidget(item)

    def on_scroll(self, event):
        """Update visible window as user scrolls"""
        self.visible_start = event.scroll_offset // item_height
```

**Benefits**:
- Constant rendering time regardless of dataset size
- Smooth scrolling with large datasets
- Memory efficient

**Textual widgets with built-in virtualization**:
- `ListView` - Virtual list with keyboard navigation
- `DataTable` - Virtual table for large datasets

---

## 3. Async Operations with Workers

### Move Heavy Work Off Main Thread

**When**: Loading files, API calls, data processing, computations.

**How**:
```python
from textual import work

class DataLoader(Widget):
    @work(exclusive=True, thread=True)
    async def load_large_file(self, path):
        """Runs in worker thread, won't block UI"""
        data = await self.expensive_io_operation(path)

        # Update UI from worker (thread-safe)
        self.call_from_thread(self.update_display, data)

    def update_display(self, data):
        """Called on main thread to update UI"""
        self.data_table.add_rows(data)
```

**Worker parameters**:
- `thread=True` - Run in thread (for blocking I/O)
- `exclusive=True` - Cancel previous run if new one starts
- `exit_on_error=False` - Don't exit app on worker error

**Benefits**:
- UI remains responsive during heavy operations
- Parallel processing of multiple tasks
- Clean async/await syntax

---

## 4. Optimize Layout Performance

### Avoid Deep Nesting

**Problem**: Deep widget trees slow down layout calculation.

❌ **Avoid**:
```python
Container(
    Container(
        Container(
            Container(
                Container(
                    MyWidget()  # 5 levels deep
                )
            )
        )
    )
)
```

✅ **Better**:
```python
Container(
    MyWidget()  # Flat structure
)
```

**Rule of thumb**: Keep nesting under 5 levels when possible.

---

### Use CSS Grid/Dock Instead of Manual Positioning

**Why**: Textual's layout engine is optimized; manual positioning isn't.

✅ **Efficient**:
```css
#container {
    layout: grid;
    grid-size: 2 2;
    grid-gutter: 1;
}
```

❌ **Slow**:
```python
# Manual positioning in compose()
widget.styles.offset = (10, 20)  # Recalculated often
```

---

## 5. Minimize Repaints

### Debounce Rapid Updates

**When**: Handling rapid user input (typing, mouse movement).

**How**:
```python
from textual import work

class SearchField(Input):
    @work(exclusive=True)
    async def on_input_changed(self, event):
        """Debounce: only process after typing stops"""
        await asyncio.sleep(0.3)  # Wait for pause
        self.perform_search(event.value)
```

**Benefits**:
- Fewer layout/render cycles
- Better perceived performance
- Reduced CPU usage

---

### Use set_interval for Animations

**When**: Periodic updates (clocks, progress bars, live data).

**How**:
```python
class LiveCounter(Widget):
    def on_mount(self):
        """Set up periodic update"""
        self.set_interval(1.0, self.update_count)  # Every 1 second

    def update_count(self):
        """Only updates this widget, not entire screen"""
        self.count += 1
        self.refresh()  # Targeted refresh
```

**Benefits**:
- Predictable update frequency
- Efficient timer management
- Automatic cleanup on widget removal

---

## 6. Optimize DataTable Performance

### Use Cursor Navigation

**When**: User needs to select/navigate large tables.

✅ **Efficient**:
```python
table = DataTable(cursor_type="row")  # Built-in cursor
# User presses up/down, Textual handles efficiently
```

❌ **Inefficient**:
```python
# Manual row highlighting on every keypress
for row in all_rows:
    row.update_style(...)  # Repaints entire table
```

---

### Add Rows in Batches

**When**: Populating table with many rows.

✅ **Batch add**:
```python
# Add many rows at once
table.add_rows([
    ["Row 1", "Data 1"],
    ["Row 2", "Data 2"],
    # ... 1000 more rows
])
# Single layout/render cycle
```

❌ **Individual adds**:
```python
for row_data in large_dataset:
    table.add_row(*row_data)  # Triggers layout each time
```

---

## 7. CSS and Styling Optimization

### Avoid Inline Style Changes

**Problem**: Inline style changes trigger style recalculation.

❌ **Slow**:
```python
widget.styles.background = "red"  # Triggers recalc
widget.styles.border = "solid"    # Another recalc
```

✅ **Fast**:
```python
widget.add_class("error")  # Single class change
```

```css
.error {
    background: red;
    border: solid red;
}
```

---

### Precompile CSS Classes

**When**: Multiple states/themes that switch frequently.

**Pattern**:
```css
/* Define all states upfront */
.status-ok { background: green; }
.status-warn { background: yellow; }
.status-error { background: red; }
```

```python
# Fast class swapping
widget.remove_class("status-ok")
widget.add_class("status-error")
```

---

## 8. Memory Optimization

### Clean Up Unused Widgets

**When**: Dynamically creating/destroying widgets (tabs, screens).

**How**:
```python
class DynamicScreen(Screen):
    async def on_unmount(self):
        """Clean up when screen closes"""
        # Remove event handlers
        # Clear large data structures
        self.data = None
        await super().on_unmount()
```

---

### Use Generators for Large Datasets

**When**: Composing many child widgets.

✅ **Memory efficient**:
```python
def compose(self):
    """Yields widgets one at a time"""
    for item in large_dataset:
        yield ItemWidget(item)
```

❌ **Memory hungry**:
```python
def compose(self):
    """Creates all widgets in memory"""
    return [ItemWidget(item) for item in large_dataset]
```

---

## 9. Profiling and Debugging

### Use Textual DevTools

**Enable**:
```bash
textual run --dev myapp.py
```

**Features**:
- Real-time DOM inspector
- CSS live editing
- Console for debug output
- Performance metrics

---

### Add Timing Logs

**When**: Identifying performance bottlenecks.

**How**:
```python
import time

class MyApp(App):
    def on_mount(self):
        start = time.perf_counter()
        self.expensive_operation()
        elapsed = time.perf_counter() - start
        self.log(f"Operation took {elapsed:.3f}s")
```

**View logs**: Check `textual.log` or DevTools console.

---

## 10. Terminal I/O Optimization

### Minimize Screen Updates

**Principle**: Terminal I/O is often the bottleneck, not CPU.

**Best practices**:
1. Use Textual's built-in dirty region tracking (automatic)
2. Batch updates with `batch_update()`
3. Avoid unnecessary `refresh()` calls
4. Let reactive variables handle updates

---

### Buffer Output

**When**: Writing lots of text (logs, file viewers).

**Pattern**: Textual handles this automatically, but for custom rendering:

```python
from textual.widgets import RichLog

# Efficient built-in buffering
log = RichLog()
log.write_lines([
    "Line 1",
    "Line 2",
    # ... many lines
])  # Buffered and batched
```

---

## Performance Checklist

Before deploying your Textual app:

- [ ] Large lists use virtualization (ListView, DataTable)
- [ ] Heavy operations use `@work` decorators
- [ ] Reactive variables handle state updates
- [ ] CSS classes used instead of inline styles
- [ ] Widget nesting kept under 5 levels
- [ ] Rapid updates debounced/throttled
- [ ] DevTools profiling shows 60+ FPS
- [ ] Memory usage stable over time
- [ ] No blocking operations on main thread

---

## Common Performance Issues

### Issue: App freezes during data load

**Cause**: Blocking I/O on main thread.

**Fix**: Use `@work(thread=True)` for I/O operations.

---

### Issue: Slow scrolling with large lists

**Cause**: Rendering all items.

**Fix**: Use `ListView` or implement virtual scrolling.

---

### Issue: Choppy animations

**Cause**: Too many repaints per frame.

**Fix**: Use `batch_update()`, throttle updates to 60 FPS max.

---

### Issue: High CPU usage when idle

**Cause**: Unnecessary refresh() calls or polling.

**Fix**: Use reactive variables and set_interval, avoid tight loops.

---

## Sources

**Related Documentation**:
- [Textual Reactivity Guide](https://textual.textualize.io/guide/reactivity/)
- [Textual Workers](https://textual.textualize.io/guide/workers/)
- [Textual DevTools](https://textual.textualize.io/guide/devtools/)

**Blog Post**:
- [Algorithms for high performance terminal apps](https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/) (December 12, 2024, accessed 2025-11-02)
