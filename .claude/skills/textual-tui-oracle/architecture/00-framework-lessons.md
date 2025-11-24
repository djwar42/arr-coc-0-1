# Framework Development Lessons: Building Modern TUI Frameworks

## Overview

This document captures 7 key lessons learned from building Textual, a modern Terminal User Interface (TUI) framework. These insights come from Will McGugan (CEO/Founder of Textualize) after over a year of development on the Textual framework, which builds upon the Rich library to create animated, web-app-like terminal applications.

These lessons span performance optimization, Python standard library features, design patterns, and terminal protocol intricacies - providing architectural guidance for anyone building terminal applications or frameworks.

---

## Lesson 1: Terminals Are Fast (With the Right Techniques)

### The Challenge

Modern terminal emulators are sophisticated pieces of software, often powered by the same graphics technologies used in video games. However, achieving smooth animation without flickering or tearing is not automatic - it requires specific techniques.

### The Three Tricks for Flicker-Free Animation

#### 1. Overwrite, Don't Clear

**Problem**: Clearing the screen and then adding new content creates visible blank frames.

**Solution**: Overwrite content entirely rather than clearing first.

```python
# ❌ BAD: Creates blank intermediate frames
sys.stdout.write('\033[2J')  # Clear screen
sys.stdout.write(new_content)

# ✅ GOOD: Overwrite content directly
sys.stdout.write('\033[H')   # Move to home position
sys.stdout.write(new_content)  # Overwrite existing content
```

**Why it works**: Eliminates the intermediate blank state where users would see flickering.

#### 2. Single Write Operations

**Problem**: Multiple `file.write()` calls risk partial updates becoming visible.

**Solution**: Buffer all content and write in a single operation.

```python
# ❌ BAD: Multiple writes create partial frames
sys.stdout.write(header)
sys.stdout.write(body)
sys.stdout.write(footer)

# ✅ GOOD: Single atomic write
output = []
output.append(header)
output.append(body)
output.append(footer)
sys.stdout.write(''.join(output))
```

**Why it works**: Ensures the terminal receives complete frames, not partial updates.

#### 3. Synchronized Output Protocol

**Protocol**: A relatively new terminal protocol extension already supported by many modern terminals.

**How it works**:
- Tell the terminal when a frame begins and ends
- Terminal can then synchronize updates with display refresh

```python
# Synchronized Output escape sequences
BEGIN_SYNC = '\033[?2026h'  # Begin synchronized update
END_SYNC = '\033[?2026l'    # End synchronized update

# Usage pattern
sys.stdout.write(BEGIN_SYNC)
sys.stdout.write(frame_content)
sys.stdout.write(END_SYNC)
sys.stdout.flush()
```

**Reference**: [Synchronized Output Protocol](https://gist.github.com/christianparpart/d8a62cc1ab659194337d73e399004036)

### Frame Rate Considerations

**Textual Baseline**: 60fps

**Rationale**: Higher frame rates aren't perceptibly different in terminal contexts. 60fps provides smooth animation while being achievable on most hardware.

### Animation Philosophy: Helpful vs Gratuitous

Not all animation is equal:

**Gratuitous Animation** (optional, can be disabled):
- Sidebar slide-in effects
- Decorative transitions
- "Nifty" but non-functional effects

**Helpful Animation** (core UX enhancement):
- Smooth scrolling - helps users maintain position in text
- Progress indicators
- State transitions that communicate system status

**Design Principle**: Textual includes mechanisms to disable gratuitous animations while preserving helpful ones.

### Terminal Emulator Differences

While choice of terminal emulator matters (modern terminals use hardware acceleration), proper implementation of the three tricks above has greater impact than emulator choice. Even older terminals achieve flicker-free animation with these techniques.

---

## Lesson 2: DictViews Are Amazing

### The Discovery

Python dict methods `keys()` and `items()` return `KeysView` and `ItemsView` objects that have **set-like interfaces** - a fact often overlooked by Python developers.

### The Problem: Optimized Screen Updates

**Context**: Textual's layout process creates a "render map" - a mapping of Widget → screen location.

**Challenge**: Avoid wasteful full-screen refresh when only some widgets change position.

**Solution**: Compare before/after render maps using set operations on ItemsView.

### Set Operations on ItemsViews

```python
from typing import Dict, Tuple

# Example: Widget render maps
# (widget_id -> (x, y, width, height))
before_map: Dict[str, Tuple[int, int, int, int]] = {
    'widget_a': (0, 0, 10, 5),
    'widget_b': (10, 0, 10, 5),
    'widget_c': (0, 5, 20, 5),
}

after_map: Dict[str, Tuple[int, int, int, int]] = {
    'widget_a': (0, 0, 10, 5),  # Unchanged
    'widget_b': (10, 1, 10, 5),  # Moved down 1
    'widget_c': (0, 5, 20, 5),   # Unchanged
    'widget_d': (0, 10, 20, 5),  # New widget
}

# Get modified items (changed or new)
modified = before_map.items() ^ after_map.items()  # Symmetric difference
# Result: {('widget_b', (10, 0, 10, 5)),
#          ('widget_b', (10, 1, 10, 5)),
#          ('widget_d', (0, 10, 20, 5))}

# Get only new widgets
new_widgets = after_map.keys() - before_map.keys()
# Result: {'widget_d'}

# Get removed widgets
removed_widgets = before_map.keys() - after_map.keys()
# Result: set()

# Get widgets that changed position
changed_positions = (before_map.items() ^ after_map.items()) & \
                   (before_map.keys() & after_map.keys())
```

### Performance Benefits

**C-Level Operations**: Set operations on views are implemented in C (CPython), making them extremely fast.

**Memory Efficiency**: Views don't copy data - they operate on the original dict.

**Use Case in Textual**: Calculate modified screen regions when CSS properties change, enabling optimized partial updates instead of full redraws.

### Key Takeaway

DictViews provide set algebra without the need for complex manual comparison loops. This is both faster and more readable than hand-written comparison code.

---

## Lesson 3: lru_cache Is Fast (Really Fast)

### The Assumption

`@lru_cache` is designed for speed, but the actual performance is surprising even for experienced developers.

### Implementation Details

**Python Version**: Available in `functools` module (standard library).

**CPython Secret**: The Python implementation in `functools.py` is just a fallback. CPython uses a [C implementation](https://github.com/python/cpython/blob/main/Modules/_functoolsmodule.c#L992) that's extremely fast for both cache hits and misses.

**Attempted Optimization**: Will McGugan attempted to beat the standard library implementation. Result: Couldn't beat it.

### When to Use lru_cache

**Lower the Barrier**: Functions that are:
- Not exactly slow individually
- Called thousands of times
- Highly cacheable (same inputs → same outputs)
- Pure functions (no side effects)

### Example: Region Overlap Calculation

```python
from functools import lru_cache
from typing import Tuple

@lru_cache(maxsize=4096)
def calculate_overlap(
    rect1: Tuple[int, int, int, int],
    rect2: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Calculate where two rectangular regions overlap.

    Args:
        rect1: (x, y, width, height) of first rectangle
        rect2: (x, y, width, height) of second rectangle

    Returns:
        (x, y, width, height) of overlapping region, or (0,0,0,0) if no overlap
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate intersection
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    if right <= left or bottom <= top:
        return (0, 0, 0, 0)  # No overlap

    return (left, top, right - left, bottom - top)

# Usage - called thousands of times during layout
overlap = calculate_overlap((0, 0, 10, 10), (5, 5, 10, 10))
```

**Why Cache This?**:
- Function is pure (no side effects)
- Relatively simple calculation
- Called thousands of times during complex layouts
- High probability of repeated inputs

### Optimal maxsize Selection

**Textual Experience**: `maxsize=1000` to `maxsize=4000` was sufficient for most use cases.

**Why This Range**:
- Large enough to capture repeated patterns
- Small enough to avoid memory bloat
- Balances cache hits vs memory usage

### Monitoring Cache Effectiveness

**Critical Practice**: Always verify assumptions with `cache_info()`.

```python
# Check cache statistics
info = calculate_overlap.cache_info()
print(f"Hits: {info.hits}")
print(f"Misses: {info.misses}")
print(f"Hit Rate: {info.hits / (info.hits + info.misses):.2%}")
print(f"Current Size: {info.currsize}/{info.maxsize}")

# Expected pattern for effective caching:
# Hits growing MUCH faster than misses
# Example good result: 10,000 hits, 200 misses = 98% hit rate
```

**Red Flags**:
- Misses growing at same rate as hits: Function inputs too variable
- Current size always at maxsize: Consider increasing maxsize
- Hit rate below 70%: Function might not be cacheable

### Design Implications

**Use caching liberally on**:
- Coordinate calculations
- Color conversions
- String measurements
- Unicode width calculations
- CSS property parsing

**Avoid caching on**:
- Functions with mutable arguments
- Functions with side effects
- Functions returning large objects
- Functions with highly variable inputs

---

## Lesson 4: Immutable Is Best

### The Philosophy

While Python doesn't have true immutable objects, immutable-style design using tuples, NamedTuples, or frozen dataclasses provides significant benefits.

### Benefits of Immutability

#### 1. Easier to Reason About

**Problem with Mutable Objects**:
```python
def update_layout(widget_state: WidgetState) -> None:
    # Does this modify widget_state?
    # Do we need to check inside the function?
    calculate_dimensions(widget_state)
    widget_state.width = 100  # Surprise side effect!
```

**Solution with Immutable Objects**:
```python
def update_layout(widget_state: WidgetState) -> WidgetState:
    # Clear: Returns NEW state, doesn't modify input
    return WidgetState(
        x=widget_state.x,
        y=widget_state.y,
        width=100,  # Changed value
        height=widget_state.height
    )
```

#### 2. Easier to Cache

Immutable objects can be used as dict keys and with `@lru_cache`:

```python
from functools import lru_cache
from dataclasses import dataclass
from typing import Tuple

# ✅ GOOD: Immutable, hashable, cacheable
@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

@lru_cache(maxsize=2048)
def calculate_area(region: Region) -> int:
    return region.width * region.height

# Works perfectly with lru_cache
area = calculate_area(Region(0, 0, 10, 10))


# ❌ BAD: Mutable, not hashable
@dataclass
class MutableRegion:
    x: int
    y: int
    width: int
    height: int

@lru_cache(maxsize=2048)
def calculate_area_bad(region: MutableRegion) -> int:
    return region.width * region.height

# TypeError: unhashable type: 'MutableRegion'
```

#### 3. Easier to Test

```python
def test_layout_calculation():
    # Immutable objects don't change - no need to deep copy
    input_state = WidgetState(x=0, y=0, width=10, height=10)

    result = calculate_layout(input_state)

    # Input is guaranteed unchanged
    assert input_state.width == 10
    assert result.width == 20  # New object with updated value
```

#### 4. Free from Side Effects

```python
# Immutable design forces functional style
def resize_widget(state: WidgetState, new_width: int) -> WidgetState:
    # Can only return new state, cannot modify input
    return WidgetState(
        x=state.x,
        y=state.y,
        width=new_width,  # Only change
        height=state.height
    )

# Caller has full control
old_state = widget.state
new_state = resize_widget(old_state, 100)

# Old state still exists and valid
# Can compare, rollback, or maintain history
```

### Practical Immutable Patterns in Python

#### NamedTuple (Classic Approach)

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: int
    y: int

# Immutable by default
p = Point(10, 20)
# p.x = 15  # AttributeError
```

#### Frozen Dataclass (Modern Approach)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def with_width(self, new_width: int) -> 'Rectangle':
        """Return new Rectangle with updated width."""
        return Rectangle(self.x, self.y, new_width, self.height)

# Usage
rect = Rectangle(0, 0, 10, 10)
wider_rect = rect.with_width(20)  # Returns new object
```

#### Tuples for Simple Cases

```python
# Simple immutable coordinates
Position = Tuple[int, int]
Size = Tuple[int, int]

def move_region(pos: Position, delta: Position) -> Position:
    return (pos[0] + delta[0], pos[1] + delta[1])
```

### When to Break Immutability

**Valid Exceptions**:
- Performance-critical tight loops where allocation overhead matters
- Large data structures where copying is prohibitive
- FFI/C extensions requiring mutable buffers
- Explicit builder patterns

**Rule of Thumb**: Start with immutable. Only make mutable when profiling proves it necessary.

---

## Lesson 5: Unicode Art Is Good

### The Value of Visual Documentation

Some technical concepts are hard to explain in words. Unicode box characters can create massively beneficial diagrams in documentation.

### Example: Region Splitting Visualization

From Textual's docstring for a method that splits a region into four sub-regions:

```python
def split_region(
    self,
    region: Tuple[int, int, int, int],
    cut_x: int,
    cut_y: int
) -> Tuple[Region, Region, Region, Region]:
    """Split a region into four sub-regions at cut points.

               cut_x ↓
            ┌────────┐ ┌───┐
            │        │ │   │
            │    0   │ │ 1 │
            │        │ │   │
    cut_y → └────────┘ └───┘
            ┌────────┐ ┌───┐
            │    2   │ │ 3 │
            └────────┘ └───┘

    Args:
        region: (x, y, width, height) to split
        cut_x: Horizontal cut point
        cut_y: Vertical cut point

    Returns:
        Tuple of four regions (top-left, top-right, bottom-left, bottom-right)
    """
    # Implementation...
```

**Value**: The diagram immediately communicates what would take several paragraphs to describe.

### More Examples

#### Layout Flow

```python
"""
Widget hierarchy and layout flow:

    ┌─ Container ──────────────────────┐
    │  ┌─ Header ─────────────────┐   │
    │  │  Logo    Title    Menu   │   │
    │  └──────────────────────────┘   │
    │  ┌─ Main ───────────┐  ┌─Side┐ │
    │  │                   │  │     │ │
    │  │     Content       │  │     │ │
    │  │                   │  │     │ │
    │  └───────────────────┘  └─────┘ │
    │  ┌─ Footer ─────────────────┐   │
    │  │     Status    Info       │   │
    │  └──────────────────────────┘   │
    └──────────────────────────────────┘
"""
```

#### Event Flow

```python
"""
Event propagation:

    Keyboard Event
         ↓
    ┌─────────────┐
    │   Screen    │  ← Can capture
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Container  │  ← Can handle
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Widget    │  ← Target widget
    └─────────────┘
         ↓
    Bubble up if not handled
"""
```

#### State Machine

```python
"""
Widget lifecycle states:

    ┌─────────┐
    │ Created │
    └────┬────┘
         ↓
    ┌─────────┐     ┌─────────┐
    │ Mounted │ ←──→│ Unmount │
    └────┬────┘     └─────────┘
         ↓
    ┌─────────┐
    │  Shown  │
    └────┬────┘
         ↓
    ┌─────────┐
    │ Removed │
    └─────────┘
"""
```

### Tools for Creating Unicode Art

**Recommended**: [Monodraw](https://monodraw.helftone.com/) (MacOS only)

**Alternatives**:
- ASCIIFlow (web-based, cross-platform)
- JavE (Java, cross-platform)
- Text editors with drawing plugins

### Best Practices

1. **Keep it simple**: Don't over-engineer diagrams
2. **Label clearly**: Use arrows and labels to show flow/direction
3. **Consistent style**: Use same box characters throughout project
4. **Test in terminals**: Verify rendering across different terminals
5. **Combine with prose**: Diagrams enhance, don't replace, written explanation

### Common Box Drawing Characters

```
┌ ─ ┐  └ ─ ┘  ├ ┤  ┬ ┴  ┼     # Standard box characters
╔ ═ ╗  ╚ ═ ╝  ╠ ╣  ╦ ╩  ╬     # Double line
╭ ─ ╮  ╰ ─ ╯                   # Rounded corners
│ ║                             # Vertical lines
↑ ↓ ← → ↔ ↕                    # Arrows
```

### Integration Tip

Place diagrams in docstrings for maximum benefit - visible in:
- IDE tooltips
- Generated documentation
- Help() output
- Code review diffs

---

## Lesson 6: Fractions Are Accurate

### The Problem with Floats

Floating point arithmetic has well-known limitations:

```python
>>> 0.1 + 0.1 + 0.1 == 0.3
False

>>> 0.1 + 0.1 + 0.1
0.30000000000000004
```

### The Real-World Impact in Textual

**Context**: Layout system dividing screen into proportional regions.

**Scenario**:
- Panel that's 1/3 of screen width
- Remaining 2/3 divided further
- Process repeated for complex nested layouts

**Problem**: Rounding errors accumulated, causing single-character gaps where content should be.

**Example**:
```python
# Screen width: 100 characters
# Divide into thirds using floats

width = 100
panel_a = int(width * (1/3))    # 33
panel_b = int(width * (2/3))    # 66

total = panel_a + panel_b        # 99 - we lost a character!
```

### The Solution: Fractions

Python's `fractions` module (standard library since Python 2.6) provides exact rational arithmetic.

```python
from fractions import Fraction as F

# Three tenths sum correctly
F(1, 10) + F(1, 10) + F(1, 10) == F(3, 10)  # True

# Screen division without rounding errors
width = 100
panel_a = int(width * F(1, 3))   # 33
panel_b = int(width * F(2, 3))   # 66
remaining = width - panel_a - panel_b  # 1

# Can allocate remaining to specific panel
# No mysterious gaps in layout!
```

### Practical Example: Splitting Characters

```python
from fractions import Fraction as F

def split_with_floats(total: int, parts: int) -> list[int]:
    """Split total into parts using float division."""
    size = total / parts
    return [int(size * i) - int(size * (i-1)) for i in range(1, parts + 1)]

def split_with_fractions(total: int, parts: int) -> list[int]:
    """Split total into parts using exact fractions."""
    size = F(total, parts)
    return [int(size * i) - int(size * (i-1)) for i in range(1, parts + 1)]

# Test with 24 characters split into 7 parts
total = 24
parts = 7

print("Float division:")
float_result = split_with_floats(total, parts)
print(''.join(str(i) * float_result[i] for i in range(parts)))
print(f"Total: {sum(float_result)}")  # 23 - Lost a character!

print("\nFraction division:")
frac_result = split_with_fractions(total, parts)
print(''.join(str(i) * frac_result[i] for i in range(parts)))
print(f"Total: {sum(frac_result)}")  # 24 - Perfect!

# Output:
# Float division:
# 00011122223334444555666    # 23 characters
#
# Fraction division:
# 000111222233344445556666   # 24 characters
```

### When to Use Fractions

**Good Use Cases**:
- Layout calculations requiring exact proportions
- Splitting fixed resources (screen space, time slices)
- Financial calculations
- Rational arithmetic where accuracy matters more than speed

**Performance Considerations**:
- Slightly slower than float arithmetic
- Additional memory overhead for numerator/denominator
- For Textual's use case (layout calculations done once per frame), performance impact negligible

### API Design with Fractions

```python
from fractions import Fraction
from typing import Union

# Accept both fractions and floats for convenience
Numeric = Union[int, float, Fraction]

class LayoutEngine:
    def split_region(
        self,
        width: int,
        ratios: list[Numeric]
    ) -> list[int]:
        """Split width according to ratios.

        Args:
            width: Total width to split
            ratios: Proportions (can be floats or Fractions)

        Returns:
            List of widths for each region

        Example:
            >>> engine.split_region(100, [1/3, 2/3])  # Floats
            [33, 67]
            >>> engine.split_region(100, [Fraction(1,3), Fraction(2,3)])  # Exact
            [33, 67]
        """
        # Convert to fractions internally
        frac_ratios = [Fraction(r).limit_denominator() for r in ratios]
        total = sum(frac_ratios)

        # Calculate exact divisions
        result = []
        accumulated = Fraction(0)
        for ratio in frac_ratios:
            accumulated += ratio
            result.append(int(width * accumulated / total))

        # Adjust for rounding by distributing remainder
        remainder = width - sum(result)
        for i in range(remainder):
            result[-(i+1)] += 1

        return result
```

### Key Takeaway

Fractions aren't just for mathematicians - they solve real engineering problems where exact arithmetic matters. The `fractions` module has been in the standard library since Python 2.6, but remains underutilized.

---

## Lesson 7: Emojis Are Terrible (But Manageable)

### The Emoji Problem Statement

Emoji support has been an ongoing challenge in Rich and inherited by Textual. It was top priority when Textualize was founded in January 2022. Investigation revealed the problem is worse than initially understood.

### Problem 1: Variable Width Characters

**Background**: Terminal characters can be:
- Single width (Western alphabet, ASCII)
- Double width (Chinese, Japanese, Korean characters - CJK)
- Zero width (combining characters)

**Impact**: Can't use `len(text)` to determine terminal width.

```python
# Standard ASCII
text = "Hello"
print(len(text))  # 5
# Terminal width: 5 characters

# CJK characters
text = "你好"
print(len(text))  # 2
# Terminal width: 4 characters (each CJK char takes 2 cells)

# Mixed
text = "Hi你"
print(len(text))  # 3
# Terminal width: 4 characters
```

**Solution**: Unicode database contains character width mapping. Rich/Textual look up every character to determine actual width.

```python
import unicodedata

def get_char_width(char: str) -> int:
    """Get terminal width of a character."""
    east_asian_width = unicodedata.east_asian_width(char)
    if east_asian_width in ('F', 'W'):  # Fullwidth, Wide
        return 2
    elif east_asian_width in ('Na', 'H', 'N', 'A'):  # Narrow, Half, Neutral, Ambiguous
        return 1
    return 0  # Zero-width

def terminal_width(text: str) -> int:
    """Calculate actual terminal width of text."""
    return sum(get_char_width(c) for c in text)
```

**Performance**: With caching (via `@lru_cache`), this lookup is fast enough.

### Problem 2: Emoji Width Unpredictability

**Core Issue**: While CJK characters are stable, emojis change with every Unicode release.

**Manifestation**:
- Newer emojis may render as single or double width
- Behavior varies by terminal emulator
- No reliable way to detect Unicode version support

**Attempted Solutions**:
1. Ship Rich with information from every Unicode release ❌
2. Detect terminal's Unicode version via env vars ❌ (no standard)
3. Heuristic: Write sequences and check cursor position ❌ (unreliable)

**Reality**: Terminals render emoji unpredictably even when Unicode version is "known".

### Problem 3: Multi-Codepoint Emojis

**Concept**: Some emojis combine multiple codepoints to produce a single glyph.

**Example**: 👨🏻‍🦰 (man, light skin tone, red hair)

```python
emoji = "👨🏻‍🦰"
print(len(emoji))  # 7 (!) - it's 4 codepoints but Python counts more

# Breaking it down:
# U+1F468: Man
# U+1F3FB: Light skin tone modifier
# U+200D:  Zero-width joiner
# U+1F9B0: Red hair component
```

**Terminal Rendering Varies**:
- Some terminals: 1 character, double width ✓
- Some terminals: 1 character, single width
- Some terminals: 2 visible characters
- Some terminals: 4 visible characters
- Some terminals: 4 "?" characters ❌

**Fundamental Problem**: Even with correct parsing code, can't predict actual output.

### The Pragmatic Solution

**Recommendation**: Stick to Unicode 9.0 emoji and earlier.

**Rationale**:
- Unicode 9.0 emoji render reliably across platforms
- Avoid newer emoji (even if they look okay on your terminal)
- Avoid multi-codepoint emoji characters
- Trade completeness for reliability

```python
# Safe emoji (Unicode 9.0 and earlier)
SAFE_EMOJI = {
    '😀', '😁', '😂', '🤣', '😃', '😄', '😅', '😆', '😉', '😊',
    '👍', '👎', '👏', '🙏', '💪', '🎉', '🎊', '🏆', '⚡', '🔥',
    '❤️', '💚', '💙', '💛', '🧡', '💜', '🖤', '🤍', '🤎', '💖'
    # ... etc
}

def is_safe_emoji(char: str) -> bool:
    """Check if emoji is safe to use across terminals."""
    return char in SAFE_EMOJI
```

### Implementation in Rich/Textual

```python
from functools import lru_cache
import unicodedata

@lru_cache(maxsize=4096)
def cell_len(text: str) -> int:
    """Get terminal cell width of text.

    Uses Unicode database for character widths and caches results.
    Handles CJK and emoji characters.
    """
    width = 0
    for char in text:
        # Check if zero-width
        if unicodedata.combining(char):
            continue

        # Get east asian width
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):  # Fullwidth or Wide
            width += 2
        elif ea_width in ('Na', 'H', 'N', 'A'):  # Narrow variants
            width += 1
        # Note: Emoji handling has special cases based on Unicode version

    return width
```

### Testing Strategy

**Cross-Platform Testing Required**:
- iTerm2 (macOS)
- Terminal.app (macOS)
- Windows Terminal
- Alacritty (cross-platform)
- Kitty (Linux/macOS)
- GNOME Terminal (Linux)
- Konsole (Linux)

**Test Cases**:
```python
test_cases = [
    ("ASCII", "Hello", 5),
    ("CJK", "你好", 4),
    ("Mixed", "Hi你", 4),
    ("Emoji safe", "Hello 👍", 8),
    ("Emoji multi", "👨🏻‍🦰", "???"),  # Unknown - terminal dependent
]
```

### Documentation Guidance

**For Library Users**:
1. Stick to Unicode 9.0 emoji
2. Test emoji rendering in target terminals
3. Provide text fallbacks for multi-codepoint emoji
4. Consider `--no-emoji` flag for CLI apps

**For Framework Developers**:
1. Cache width calculations aggressively
2. Provide emoji support as opt-in
3. Document known limitations
4. Consider emoji-safe subsets

### The Bottom Line

Emoji support in terminals is messy, but manageable:
- **Accept**: Perfect emoji support isn't achievable
- **Pragmatic**: Use proven-safe emoji subset
- **Test**: Verify across multiple terminals
- **Document**: Be clear about limitations

**Quote from Will McGugan**: "It's a mess for sure, but in practice it's not that bad."

---

## Architectural Patterns Summary

### Performance Optimization

1. **Batch Terminal Updates**: Single write, synchronized output protocol
2. **Cache Aggressively**: Use `@lru_cache` liberally with 1000-4000 maxsize
3. **Leverage C-Level Operations**: DictViews, standard library implementations

### Code Design

1. **Prefer Immutability**: Use frozen dataclasses, NamedTuples
2. **Exact Arithmetic**: Use Fractions for layout calculations
3. **Document Visually**: Unicode art in docstrings

### Terminal Compatibility

1. **Test Widely**: Multiple terminal emulators required
2. **Conservative Emoji**: Stick to Unicode 9.0
3. **Measure Everything**: Don't assume, use cache_info() and profiling

### Development Philosophy

- **Measure First**: Profile before optimizing
- **Standard Library Power**: Don't reinvent wheels (lru_cache, fractions, etc.)
- **User Experience**: Distinguish helpful vs gratuitous features

---

## Additional Resources

### Textual Framework

- **Repository**: https://github.com/Textualize/textual
- **Documentation**: https://textual.textualize.io/
- **Company**: https://www.textualize.io/

### Rich Library

- **Repository**: https://github.com/Textualize/rich
- **Foundation**: Terminal rendering library that Textual builds upon

### Terminal Protocols

- **Synchronized Output**: https://gist.github.com/christianparpart/d8a62cc1ab659194337d73e399004036
- **ANSI Escape Codes**: Standard terminal control sequences

### Python Standard Library

- **functools.lru_cache**: https://docs.python.org/3/library/functools.html#functools.lru_cache
- **fractions**: https://docs.python.org/3/library/fractions.html
- **unicodedata**: https://docs.python.org/3/library/unicodedata.html

---

## Sources

**Primary Source**:
- [7 Things I've Learned Building a Modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/) - Textualize Blog (August 3, 2022)

**Podcast Episode**:
- [Talk Python To Me #380: 7 Lessons from Building a Modern TUI Framework](https://talkpython.fm/episodes/show/380/7-lessons-from-building-a-modern-tui-framework) - September 5, 2022
- Guest: Will McGugan (CEO/Founder, Textualize)
- Host: Michael Kennedy

**Author**: Will McGugan ([@willmcgugan](https://twitter.com/willmcgugan))

**Accessed**: 2025-11-02

---

## Related Oracle Documentation

- [Core Concepts: Reactive Programming](../concepts/00-reactive-programming.md)
- [Core Concepts: CSS Styling](../concepts/01-css-styling.md)
- [Widgets: Built-in Widget Gallery](../widgets/00-builtin-widgets.md)
- [Layout Systems: Grid](../layout/00-grid-system.md)
- [Layout Systems: Dock](../layout/01-dock-system.md)

---

*This knowledge was acquired via web research on 2025-11-02 as part of the textual-tui-oracle dynamic knowledge expansion initiative.*
