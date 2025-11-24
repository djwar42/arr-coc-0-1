# Lessons Learned from Production Textual Apps

## Overview

This document captures critical lessons learned from building modern TUI applications with Textual. These are discoveries from real-world production experiences that can significantly impact your application's quality, performance, and user experience.

From [7 Things I've learned building a modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework) by Will McGugan (Textualize CEO), accessed 2025-11-02:

---

## Terminal Performance and Animation

### Problem: Flickering and Tearing in Terminal Output

**Challenge**: Many developers assume smooth animation in terminals is impossible, leading to choppy or flickering interfaces.

**Solution**: Use three proven techniques:

**1. Overwrite, Don't Clear**
- Never clear the screen then add new content
- This creates a visible blank frame
- Instead: overwrite content entirely in a single pass

**2. Write in Single Output Call**
```python
# Bad: Multiple write calls risk partial updates
terminal.write(header)
terminal.write(content)
terminal.write(footer)

# Good: Single write operation
output = header + content + footer
terminal.write(output)
```

**3. Use Synchronized Output Protocol**
- Modern terminal emulator feature (widely supported)
- Tell terminal when frame begins and ends
- Enables flicker-free updates
- Textual supports this automatically at 60fps baseline

### Key Learning
Hardware-accelerated terminals can deliver smooth animation if you respect the rendering pipeline. Target 60fps as baseline—anything higher won't be perceptible.

**Application**: Smooth scrolling and UI transitions feel more responsive and professional when implemented correctly.

---

## Performance Optimization

### Problem: Inefficient Cache Usage

**Challenge**: Small functions called thousands of times accumulate CPU overhead, even though they do minimal work.

**Solution**: Use `@lru_cache` from functools

```python
from functools import lru_cache

# This method calculates rectangular region overlaps
# Called 1000s of times during layout
@lru_cache(maxsize=1000)
def calculate_overlap(region1, region2):
    """Find where two rectangular regions overlap."""
    x1_start, y1_start, x1_end, y1_end = region1
    x2_start, y2_start, x2_end, y2_end = region2

    x_start = max(x1_start, x2_start)
    x_end = min(x1_end, x2_end)
    y_start = max(y1_start, y2_start)
    y_end = min(y1_end, y2_end)

    if x_start < x_end and y_start < y_end:
        return (x_start, y_start, x_end, y_end)
    return None
```

### Key Learning
- CPython's `lru_cache` is implemented in C and extremely fast
- For Textual apps, `@lru_cache(maxsize=1000-4000)` handles most scenarios
- Monitor cache effectiveness with `cache_info()`:

```python
# Check if your caching strategy works
info = calculate_overlap.cache_info()
print(f"Hits: {info.hits}, Misses: {info.misses}")
# Healthy ratio: hits >> misses
```

**Application**: Textual uses caching extensively for layout calculations, widget measurements, and CSS property changes. Measurable performance improvement (faster re-renders).

---

## Data Structure Design

### Problem: Mutable State Makes Code Hard to Reason About

**Challenge**: Passing mutable objects between functions creates hidden side effects and makes testing/caching difficult.

**Solution**: Prefer immutable data structures

```python
# Bad: Mutable dict, unpredictable changes
widget_state = {'width': 100, 'height': 50}
update_dimensions(widget_state)  # Widget state modified?

# Good: Use NamedTuple or dataclass
from typing import NamedTuple

class WidgetDimensions(NamedTuple):
    width: int
    height: int

dimensions = WidgetDimensions(100, 50)
new_dimensions = update_dimensions(dimensions)  # Returns new instance
```

### Benefits in Textual Development
- Easier to cache (immutable = cacheable)
- Easier to test (no hidden state mutations)
- Easier to reason about (pure functions)
- Thread-safe by default
- Better reactive property handling

**Application**: Textual's reactive attributes work best with immutable updates.

---

## Precision Arithmetic

### Problem: Floating-Point Rounding Errors in Layout

**Challenge**: When dividing screen real estate proportionally (e.g., 1/3 and 2/3 of width), floating-point rounding creates single-character gaps in your layout.

```python
# Problem demonstration
>>> 0.1 + 0.1 + 0.1 == 0.3
False  # Floating point rounding error!

# Layout scenario
width = 100
first_panel = width / 3      # 33.333...
remaining = width - first_panel
second_panel = remaining / 2  # 33.333...
third_panel = remaining / 2   # 33.333...
# Total: 33 + 33 + 33 = 99 (1 character gap!)
```

**Solution**: Use `fractions.Fraction` for exact division

```python
from fractions import Fraction

def divide_space_proportionally(total_width: int, proportions: list):
    """Divide space exactly without rounding errors."""
    fractions = [Fraction(p) for p in proportions]
    total_proportion = sum(fractions)

    return [
        int(total_width * (f / total_proportion))
        for f in fractions
    ]

# Example
widths = divide_space_proportionally(100, [1, 2, 1])
# Result: [25, 50, 25] - perfect division with no gaps
```

**Application**: Critical for multi-panel layouts where visual alignment matters.

---

## Unicode Handling

### Problem: Emoji and Unicode Character Width

**Challenge**: Terminal width calculation is broken by:
- Double-width characters (CJK: Chinese, Japanese, Korean)
- Multi-codepoint emoji (👨🏻‍🦰 = 4 codepoints = 1 glyph)
- Terminal inconsistency in emoji rendering

```python
# Width problem
text = "Hello 👨🏻‍🦰"
print(len(text))  # Returns 11 (counting codepoints)
# But terminal renders as ~9 characters wide

# Terminal rendering inconsistency
# Same emoji renders differently across terminals:
# - 1 character wide
# - 2 characters wide
# - As 4 separate characters
# - As "?" characters (not supported)
```

### Solutions and Recommendations

**1. Use Unicode Width Database**
```python
from rich.text import Text

# Rich handles character width correctly
text = Text("Hello 👨🏻‍🦰")
width = text.cell_len  # Correct terminal width
```

**2. Stick to Safe Unicode Ranges**
- Unicode version 9 emoji: Reliable across terminals
- Avoid multi-codepoint emoji combinations
- Test on target terminals before shipping

**3. Provide Fallbacks**
```python
def render_with_fallback(primary_char, fallback_char):
    """Attempt primary char, fall back to ASCII if needed."""
    try:
        # Attempt to render
        return primary_char
    except UnicodeError:
        return fallback_char
```

**Key Learning**: Textual uses Rich for character width calculation. Modern terminals handle basic emoji (v9) reliably, but multi-codepoint combinations remain problematic.

---

## Dictionary Views for Optimization

### Problem: Inefficient Set Operations on Dictionaries

**Challenge**: Comparing two dictionaries for changes (e.g., before/after layout) requires manual iteration.

**Solution**: Use `DictView` objects (keys(), values(), items())

```python
# Before: Complex manual comparison
before_layout = {widget_id: position for widget_id, position in old_positions}
after_layout = {widget_id: position for widget_id, position in new_positions}

changed = {}
for widget_id in after_layout:
    if widget_id not in before_layout or before_layout[widget_id] != after_layout[widget_id]:
        changed[widget_id] = after_layout[widget_id]

# After: Elegant set operations on DictViews
before_items = before_layout.items()
after_items = after_layout.items()
changed_items = after_items - before_items  # Set difference at C level!
```

**Textual Application**: When widgets change position/size, Textual compares render maps using:
```python
before_render_map = {widget: location for widget, location in old_renders}
after_render_map = {widget: location for widget, location in new_renders}

# Efficient at C level
modified_regions = set(after_render_map.items()) - set(before_render_map.items())
```

This enables **optimized screen updates** - only changed regions re-render.

---

## Terminal Quirks Summary

### Known Terminal Issues

| Issue | Impact | Workaround |
|-------|--------|-----------|
| Emoji width variation | Layout breaks | Use Rich, test emoji set, provide ASCII fallback |
| Slow scroll rendering | Perceived lag | Use Synchronized Output protocol |
| Mouse support inconsistency | Input bugs | Test on target terminal, provide keyboard alternative |
| Color palette limits | Color bleeding | Stick to 256-color or 24-bit RGB support detection |
| Alt-screen buffer | State loss | Always reset terminal state on exit |

### Best Practices for Terminal Compatibility

1. **Test on multiple terminals**: iTerm2, Alacritty, GNOME Terminal, tmux, Windows Terminal
2. **Provide keyboard-only operation**: Not all terminals support mouse
3. **Detect terminal capabilities** at runtime:
   ```python
   import os
   supports_true_color = os.environ.get('COLORTERM') in ('truecolor', '24bit')
   ```
4. **Fallback gracefully**: Always have ASCII/basic mode
5. **Document terminal requirements** in README

---

## Sources

**Primary Source**:
- [7 Things I've learned building a modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework) - Will McGugan, Textualize CEO (August 3, 2022)
  - Terminal animation optimization
  - Performance tuning (lru_cache)
  - Immutable data structures
  - Fractions for precision
  - Unicode/emoji challenges
  - DictView optimization
  - Practical terminal quirks

---

## Related Documentation

- [00-getting-started.md](../getting-started/00-getting-started.md) - Textual setup and first steps
- [patterns/00-long-running-processes.md](../patterns/00-long-running-processes.md) - Threading and background work
- [widgets/00-widget-patterns.md](../widgets/00-widget-patterns.md) - Widget implementation patterns
