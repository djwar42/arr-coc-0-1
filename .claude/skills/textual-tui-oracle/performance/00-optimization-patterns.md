# Textual Performance Optimization Patterns

## Overview

Performance optimization in Textual TUI applications requires understanding terminal rendering fundamentals, Python optimization techniques, and framework-specific best practices. This guide synthesizes production-tested patterns from the Textual framework creators and community.

From [7 Things I've learned building a modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/) (accessed 2025-11-02):
- Modern terminals support smooth 60fps animation
- Cache-based optimization is critical
- Immutable data structures improve performance

From [Algorithms for high performance terminal apps](https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/) (accessed 2025-11-02):
- Compositor optimization for render maps
- Spatial mapping algorithms
- Grid-based rendering strategies

---

## Terminal Rendering Optimization

### Flicker-Free Animation (60fps Baseline)

**Three Core Techniques:**

1. **Overwrite, Don't Clear**
   - Never clear screen then redraw
   - Overwrite content entirely to avoid blank frames
   - Eliminates intermediate partial updates

2. **Single Write per Frame**
   - Batch all updates into one `file.write()` call
   - Multiple writes risk visible partial frames
   - Synchronize updates at terminal level

3. **Synchronized Output Protocol**
   - Use terminal's synchronized output when available
   - Tell terminal when frame begins/ends
   - Terminal handles flicker-free delivery
   - Relatively new but widely supported

**Target framerate:**
```python
# Textual uses 60fps as baseline
# Higher framerates not perceptible in terminals
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS  # ~16.6ms per frame
```

**Animation Philosophy:**
- **Helpful animation**: Smooth scrolling (maintains context in text)
- **Gratuitous animation**: Slide-in sidebars (eye candy only)
- Always provide mechanism to disable animations

From Will McGugan (Textual creator):
> "Modern terminal emulators use hardware-accelerated rendering and synchronize updates with your display. With three simple tricks you can achieve smooth animation as long as you can deliver updates at regular intervals."

---

## Python Performance Patterns

### 1. lru_cache for Hot Paths

**Why lru_cache is Fast:**
- Implemented in C (CPython)
- Extremely fast for cache hits AND misses
- Don't try to beat it with custom caching

**Effective Use Cases:**

```python
from functools import lru_cache

# Small functions called thousands of times
@lru_cache(maxsize=4000)
def calculate_overlap(region1, region2):
    """Calculate where two rectangular regions overlap.

    Not computationally expensive, but called 1000s of times.
    Cache hit rate > 90% with maxsize=4000.
    """
    # Overlap calculation logic
    pass
```

**Cache Sizing Guidelines:**
- `maxsize=1000-4000` typically sufficient
- Check assumptions with `cache_info()`
- Expect hits growing faster than misses

**Always Verify:**
```python
# Inspect cache effectiveness
print(calculate_overlap.cache_info())
# CacheInfo(hits=45230, misses=1847, maxsize=4000, currsize=1847)
# Hit rate: 96% - excellent!
```

From the Textual codebase:
> "Judicious use of @lru_cache provided a significant win. Typically a maxsize of around 1000-4000 was enough to ensure that the majority calls were cache hits."

### 2. DictViews for Set Operations

**Powerful Pattern:**
```python
# Before and after render maps (Widget -> screen location)
before_map = {widget1: rect1, widget2: rect2}
after_map = {widget1: rect1, widget2: new_rect2, widget3: rect3}

# Get changed/new items using symmetric difference
changed = before_map.items() ^ after_map.items()
# Returns: {(widget2, new_rect2), (widget3, rect3)}
```

**Benefits:**
- `KeysView` and `ItemsView` have set interfaces
- Operations done at C level
- Avoids complex Python loops
- Used in Textual for optimized screen updates

**Real-World Application:**
```python
def get_modified_regions(old_render_map, new_render_map):
    """Get widgets that changed position or are new."""
    # Symmetric difference: items in either set but not both
    modified = old_render_map.items() ^ new_render_map.items()
    return modified
```

### 3. Immutable Objects

**Preferred Types:**
- `tuple`
- `NamedTuple`
- Frozen `dataclass`

**Benefits:**
- Free of side-effects
- Easy to cache (hashable)
- Easy to reason about
- Easy to test

**Example:**
```python
from dataclasses import dataclass
from typing import NamedTuple

# Good: Immutable region
class Region(NamedTuple):
    x: int
    y: int
    width: int
    height: int

# Also good: Frozen dataclass
@dataclass(frozen=True)
class Style:
    color: str
    bold: bool
    italic: bool
```

From Textual development experience:
> "In Textual, the code that uses immutable objects is the easiest to reason about, easiest to cache, and easiest to test. Mainly because you can write code that is free of side-effects."

### 4. Fractions for Accurate Layout

**Problem with Floats:**
```python
>>> 0.1 + 0.1 + 0.1 == 0.3
False  # Floating point rounding error
```

**Solution: Use Fractions:**
```python
from fractions import Fraction as F

>>> F(1, 10) + F(1, 10) + F(1, 10) == F(3, 10)
True  # Exact arithmetic
```

**Layout Example:**
```python
from fractions import Fraction

def split_width_floats(total: int, ratios: list[float]) -> list[int]:
    """Split using floats - may have gaps due to rounding."""
    result = []
    for ratio in ratios:
        result.append(int(total * ratio))
    return result

def split_width_fractions(total: int, ratios: list[Fraction]) -> list[int]:
    """Split using fractions - pixel-perfect layout."""
    result = []
    for ratio in ratios:
        result.append(int(total * ratio))
    return result

# Float version (first row) is 1 character short:
# ------------------------
# 00011122223334444555666  (float - missing char)
# 000111222233344445556666 (fraction - perfect)
```

**When to Use:**
- Screen/panel division calculations
- Proportional layouts
- Any arithmetic where precision matters

---

## Textual-Specific Optimization

### Render Map Optimization

**Concept:**
- Layout process creates "render map"
- Mapping: Widget → screen location
- Compare before/after to update only changed regions

**Naive Approach (Slow):**
```python
# Refresh entire screen if anything changes
if render_map != old_render_map:
    refresh_entire_screen()  # Wasteful!
```

**Optimized Approach:**
```python
# Only update changed regions
modified = old_render_map.items() ^ new_render_map.items()
for widget, region in modified:
    refresh_region(region)  # Targeted updates
```

### CSS Property Change Optimization

**Pattern:**
```python
# Get modified regions when CSS property changes
old_regions = widget.get_regions()
widget.styles.color = "red"  # CSS change
new_regions = widget.get_regions()

# Only update what changed
changed = old_regions.items() ^ new_regions.items()
for region in changed:
    update_region(region)
```

### Widget Rendering Cache

**Caching Strategy:**
```python
from functools import lru_cache

class CustomWidget(Widget):
    @lru_cache(maxsize=128)
    def render_line(self, y: int) -> Strip:
        """Render single line with caching.

        Cache key includes widget state for invalidation.
        """
        # Expensive rendering logic
        return Strip(segments)
```

**Cache Invalidation:**
- Include relevant state in cache key
- Clear cache on state changes
- Monitor `cache_info()` to tune `maxsize`

---

## Memory Optimization

### Widget Lifecycle Management

**Pattern:**
```python
class EfficientApp(App):
    def on_mount(self):
        # Load widgets lazily
        pass

    def on_unmount(self):
        # Clean up resources explicitly
        self.clear_caches()
        self.remove_children()
```

### Lazy Loading

**Example:**
```python
class DataTable(Widget):
    def __init__(self, data_source):
        self.data_source = data_source
        self._loaded_rows = {}  # Cache loaded rows

    def get_row(self, index: int):
        """Load rows on demand, cache results."""
        if index not in self._loaded_rows:
            self._loaded_rows[index] = self.data_source.fetch(index)
        return self._loaded_rows[index]
```

### Memory Profiling

From [2024 Textual blog archive](https://textual.textualize.io/blog/archive/2024/) (accessed 2025-11-02):

**Use Memray:**
```bash
# Install
pip install memray

# Profile your app
memray run --live app.py

# Analyze results
memray flamegraph memray-results.bin
```

**Memray Benefits:**
- Built by Bloomberg engineers
- Identifies memory leaks
- C-level profiling
- Live monitoring mode

---

## Production Deployment Best Practices

### Terminal Detection

**Detect Capabilities:**
```python
import os
import sys

def detect_terminal_features():
    """Detect terminal capabilities for optimization."""
    features = {
        'colors': os.environ.get('COLORTERM', '') == 'truecolor',
        'unicode': sys.stdout.encoding.lower().startswith('utf'),
        'synchronized_output': check_sync_support(),
    }
    return features
```

### Graceful Degradation

**Pattern:**
```python
class App(App):
    def __init__(self):
        self.features = detect_terminal_features()

    def render_widget(self, widget):
        if self.features['colors']:
            return widget.render_rich()
        else:
            return widget.render_simple()
```

### Performance Monitoring

**Built-in Dev Tools:**
```python
# Enable Textual devtools
from textual.devtools import DevConsole

class MyApp(App):
    def on_mount(self):
        if self.debug:
            self.mount(DevConsole())
```

**Key Metrics:**
- Frame render time (target: <16.6ms for 60fps)
- Widget count (minimize active widgets)
- Cache hit rates (monitor all `@lru_cache`)
- Memory usage (use Memray)

---

## CSS Performance

### Selector Efficiency

**Fast Selectors:**
```css
/* ID selector - fastest */
#my-widget { color: red; }

/* Class selector - fast */
.button { background: blue; }
```

**Slow Selectors:**
```css
/* Universal selector - avoid */
* { margin: 0; }

/* Deep nesting - minimize */
.container .panel .button .icon { }
```

### Property Optimization

**Expensive Properties:**
- Layout changes (width, height, border)
- Triggers full reflow

**Cheap Properties:**
- Color changes
- Text styling
- Background (no layout impact)

**Pattern:**
```css
/* Prefer property changes that don't affect layout */
.button:hover {
    /* Good: color change only */
    color: white;
    background: blue;
}

.panel.expanded {
    /* Expensive: triggers layout recalculation */
    width: 100%;  /* Use sparingly */
}
```

### CSS Caching

**Textual CSS Engine:**
- Parses CSS once at startup
- Caches computed styles
- Only recalculates on state changes

**Optimization:**
```python
# Define static styles in CSS file
# Dynamic styles in Python only when needed
class MyWidget(Widget):
    DEFAULT_CSS = """
    MyWidget {
        border: solid blue;  /* Static */
    }
    """

    def update_dynamic(self, highlighted: bool):
        # Only modify when state changes
        if highlighted:
            self.styles.background = "yellow"
```

---

## Common Performance Pitfalls

### 1. Emoji Rendering Issues

**Problem:**
- Emoji width unpredictable across terminals
- Multi-codepoint emoji (👨🏻‍🦰 = 4 codepoints)
- Unicode database version variations

**Safe Strategy:**
```python
# Stick to Unicode 9 emoji for reliability
SAFE_EMOJI = {
    'check': '✓',
    'cross': '✗',
    'arrow': '→',
}

# Avoid newer multi-codepoint emoji
AVOID = ['👨🏻‍🦰', '🧑‍💻']  # Unreliable rendering
```

**Why:**
- Different terminals render differently
- May appear as 1, 2, or 4 characters
- Width calculation fails
- Layout breaks

From Textual experience:
> "Sticking to the emoji in version 9 of the Unicode database seems to be reliable across all platforms. Avoid newer emoji and multi-codepoint characters even if they look okay on your terminal."

### 2. Excessive Widget Updates

**Anti-Pattern:**
```python
# DON'T: Update on every frame
def on_timer(self):
    self.query_one("#status").update("tick")  # Every frame!
```

**Better:**
```python
# DO: Update only when data changes
def on_data_change(self, new_data):
    if new_data != self.current_data:
        self.query_one("#status").update(new_data)
        self.current_data = new_data
```

### 3. Unbounded Caches

**Problem:**
```python
# Dangerous: unlimited cache growth
@lru_cache(maxsize=None)  # Memory leak!
def expensive_calculation(x):
    pass
```

**Solution:**
```python
# Safe: bounded cache with monitoring
@lru_cache(maxsize=4000)
def expensive_calculation(x):
    pass

# Periodic monitoring
if expensive_calculation.cache_info().currsize > 3500:
    expensive_calculation.cache_clear()  # Prevent growth
```

### 4. Synchronous I/O in Event Handlers

**Anti-Pattern:**
```python
def on_button_pressed(self):
    data = requests.get(url)  # Blocks UI!
    self.update(data)
```

**Better:**
```python
async def on_button_pressed(self):
    data = await self.fetch_async(url)  # Non-blocking
    self.update(data)
```

---

## Unicode and Text Rendering

### Width Calculation

**The Problem:**
```python
# len() doesn't account for display width
text = "Hello世界"  # Mix of single/double-width
print(len(text))  # 7 characters
print(display_width(text))  # 9 cells (世界 are double-width)
```

**Textual's Solution:**
- Looks up Unicode database for every character
- Caches width calculations (`@lru_cache`)
- Handles CJK (Chinese, Japanese, Korean) characters

**Pattern:**
```python
from functools import lru_cache
import unicodedata

@lru_cache(maxsize=8192)
def get_char_width(char: str) -> int:
    """Get display width of character.

    Returns 0, 1, or 2 based on Unicode database.
    Cached for performance.
    """
    if unicodedata.east_asian_width(char) in 'WF':
        return 2  # Wide/Fullwidth
    return 1  # Normal width
```

---

## Documentation and Debugging

### Unicode Art in Docstrings

**Benefits:**
- Clarifies complex spatial relationships
- Better than words for geometry

**Example:**
```python
def split_region(region, cut_x, cut_y):
    """Split region into 4 sub-regions.

               cut_x ↓
            ┌────────┐ ┌───┐
            │        │ │   │
            │    0   │ │ 1 │
            │        │ │   │
    cut_y → └────────┘ └───┘
            ┌────────┐ ┌───┐
            │    2   │ │ 3 │
            └────────┘ └───┘
    """
    pass
```

**Tool Recommendation:**
- MacOS: [Monodraw](https://monodraw.helftone.com/)
- Alternatives exist for other platforms

---

## Performance Checklist

**Before Production:**

- [ ] Profile with Memray (identify memory leaks)
- [ ] Check all `cache_info()` (verify hit rates > 80%)
- [ ] Monitor frame render times (< 16.6ms for 60fps)
- [ ] Test on multiple terminal emulators
- [ ] Verify Unicode handling (CJK, emoji)
- [ ] Enable synchronized output protocol
- [ ] Use immutable objects for shared state
- [ ] Replace floats with Fractions for layouts
- [ ] Lazy load heavy widgets
- [ ] Batch terminal writes per frame

**Runtime Monitoring:**

- [ ] Frame rate consistency
- [ ] Memory growth over time
- [ ] Cache effectiveness
- [ ] Widget count trends
- [ ] Render map diff sizes

---

## Sources

**Primary Resources:**

- [7 Things I've learned building a modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/) - Will McGugan, August 3, 2022 (accessed 2025-11-02)
- [Algorithms for high performance terminal apps](https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/) - Textual Documentation, December 12, 2024 (accessed 2025-11-02)
- [2024 Textual Blog Archive](https://textual.textualize.io/blog/archive/2024/) - Textual Documentation (accessed 2025-11-02)

**Additional References:**

- [Real Python - Python Textual](https://realpython.com/python-textual/) - March 12, 2025
- [CSS Performance Optimization - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Performance/CSS) (accessed 2025-11-02)
- Textual GitHub Repository - Performance discussions and issues

**Key Contributors:**

- Will McGugan (Textual creator, CEO Textualize)
- Textual/Textualize engineering team
- Community performance reports

---

## Summary

**Core Performance Principles:**

1. **Terminal rendering**: Overwrite, single write, synchronized output
2. **Python optimization**: `@lru_cache`, DictViews, immutable objects, Fractions
3. **Textual-specific**: Render map diffs, CSS caching, widget lifecycle
4. **Memory**: Memray profiling, lazy loading, bounded caches
5. **Production**: Terminal detection, graceful degradation, monitoring

**Target Metrics:**
- 60fps (16.6ms frame time)
- Cache hit rate > 80%
- Memory stable over time
- Unicode 9 emoji for reliability

**Philosophy:**
> "Modern terminals are fast. With the right techniques, you can achieve smooth 60fps animation. The key is understanding what the terminal can do and optimizing Python code to deliver updates consistently." - Will McGugan
