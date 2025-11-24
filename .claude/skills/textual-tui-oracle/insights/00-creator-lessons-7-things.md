# 7 Things I've Learned Building a Modern TUI Framework

**Source**: [Textualize Blog - 7 Things I've Learned Building a Modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/)
**Author**: Will McGugan (CEO/Founder, Textualize)
**Published**: August 3, 2022
**Accessed**: 2025-11-02

**Context**: Insider knowledge from the creator of Textual after over a year of development. These are hard-won lessons about terminals, Python performance, and TUI framework design.

---

## Overview

Will McGugan shares 7 key discoveries from building Textual - covering terminal performance, Python optimization techniques, design patterns, and the harsh reality of emoji support. Essential reading for understanding both the capabilities and limitations of modern TUI development.

---

## Lesson 1: Terminals Are Fast

### The Reality

Modern terminal emulators are remarkably sophisticated - powered by the same graphics technologies used in video games. Despite ancient protocols (teleprinter heritage), they can achieve smooth 60fps animation **if you know the tricks**.

### The Problem: Flicker and Tearing

Visual effects in terminals often disappoint with flickering or tearing. But smooth animation is possible.

### Three Tricks for Flicker-Free Animation

**Trick 1: Overwrite, Don't Clear**
- **DON'T**: Clear screen, then add new content → blank frame flash
- **DO**: Overwrite content entirely → no intermediate blank frames
- Eliminates the momentary blank screen that causes flicker

**Trick 2: Single Write for Updates**
- **DON'T**: Multiple `file.write()` calls → risk of partial update becoming visible
- **DO**: Write new content in a single write to stdout
- Ensures atomic updates that the terminal can render as one frame

**Trick 3: Use Synchronized Output Protocol**
- Relatively new terminal protocol addition
- Already supported by many modern terminals
- Tell the terminal when you begin/end a frame
- Terminal uses this to deliver flicker-free updates
- Details: [Synchronized Output Spec](https://gist.github.com/christianparpart/d8a62cc1ab659194337d73e399004036)

### Baseline: 60fps

Textual uses 60fps as baseline. More than that isn't noticeable in terminals.

### Animation Philosophy: Helpful vs Gratuitous

**Gratuitous**: Sidebar sliding in from left
- Nifty but doesn't add to UX
- Textual will have mechanism to disable

**Helpful**: Smooth scrolling
- Helps keep your place in walls of text
- Actually improves usability

All animations lie somewhere between helpful and gratuitous. Few people want zero animation, but gratuitous effects should be optional.

---

## Lesson 2: DictViews Are Amazing

### The Discovery

Python dict methods `keys()` and `items()` return `KeysView` and `ItemsView` objects that have **the same interfaces as sets**. Most developers don't know this.

### Real-World Use Case: Render Map Optimization

**Problem**: Textual creates a "render map" (Widget → screen location mapping). Early version wastefully refreshed entire screen if even one widget changed position.

**Solution**: Compare before/after render maps using set operations.

**Key Insight**: Symmetric difference of two `ItemsView` objects gives you:
- Items that are new
- Items that have changed

Exactly what was needed for optimized screen updates, done at C level for speed.

### Code Example Pattern

```python
# From Will's demonstration
# Symmetric difference of ItemsView objects
changed_items = old_map.items() ^ new_map.items()
```

### Application in Textual

Used to get modified regions of the screen when CSS properties change → enables optimized partial updates instead of full screen redraws.

### Takeaway

Don't reinvent set operations - `KeysView` and `ItemsView` already implement them efficiently in C.

---

## Lesson 3: lru_cache Is Fast

### The Surprise

`@lru_cache` is **FAST**. Not just "fast for Python" - legitimately fast enough to beat hand-rolled alternatives.

### The Attempt to Beat It

Will looked at [CPython lru_cache implementation](https://github.com/python/cpython/blob/main/Lib/functools.py#L566) thinking he could beat it.

**Spoiler**: He couldn't.

**Why**: CPython uses [this C version](https://github.com/python/cpython/blob/main/Modules/_functoolsmodule.c#L992) which is extremely fast for both:
- Cache hits
- Cache misses

### Changed Philosophy: Lower the Barrier

This discovery convinced Will to use `@lru_cache` more liberally.

**Target Functions**:
- Small functions
- Not exactly slow individually
- Called thousands of times
- Highly cacheable results

**Sweet Spot**: `maxsize` of 1000-4000 ensures majority of calls are cache hits.

### Real Example: Region Overlap Calculation

```python
# Method that calculates where two rectangular regions overlap
# Doesn't do much work, but called 1000s of times
# Perfect candidate for lru_cache with maxsize=1024-4096
```

### Critical Advice: Always Verify

Check your assumptions by inspecting `cache_info()`:
- **Good caching**: `hits` growing faster than `misses`
- **Bad caching**: `misses` growing as fast or faster than `hits`

Don't assume - measure!

---

## Lesson 4: Immutable Is Best

### The Philosophy

Python doesn't have true immutable objects, but you can get benefits from:
- Tuples
- NamedTuples
- Frozen dataclasses

### Why It Seems Like a Limitation

It **seems** arbitrary that you can't change an object.

### Why It's Actually Powerful

Computer scientists know: many languages are immutable by default for **good reason**.

### Benefits in Textual

Code using immutable objects is:
1. **Easiest to reason about** - no hidden state changes
2. **Easiest to cache** - can use as dict keys, lru_cache works perfectly
3. **Easiest to test** - predictable behavior, no side effects

### The Core Advantage: Side-Effect-Free Code

Difficult to write when passing class instances to functions (mutable state).
Natural when using immutable objects.

### Practical Impact

Immutability reduces cognitive load and bugs. You can trust that objects won't change behind your back.

---

## Lesson 5: Unicode Art Is Good

### The Problem

Some technical concepts are hard to explain in words alone.

### The Solution

Diagrams created from Unicode box characters in documentation and docstrings.

### Real Example from Textual

Method that splits a region into four sub-regions:

```
               cut_x ↓
            ┌────────┐ ┌───┐
            │        │ │   │
            │    0   │ │ 1 │
            │        │ │   │
    cut_y → └────────┘ └───┘
            ┌────────┐ ┌───┐
            │    2   │ │ 3 │
            └────────┘ └───┘
```

### Value Proposition

- Not a substitute for well-written docstrings
- **In combination**: super helpful
- Provides instant visual understanding of spatial/structural concepts

### Tools

Will uses [Monodraw](https://monodraw.helftone.com/) (macOS only).

Alternatives exist for other platforms - the key is having a tool for creating clean Unicode diagrams.

### Recommendation

Add diagrams to docstrings wherever it makes sense, especially for:
- Spatial relationships
- Layout algorithms
- Data structure visualizations
- Flow diagrams

---

## Lesson 6: Fractions Are Accurate

### The Revelation

Python's `fractions` module (in stdlib since Python 2.6) was considered "for mathematicians only" - **wrong assumption**.

It was a "real life saver" for Textual.

### The Floating Point Problem

Classic example that illustrates the issue:

```python
>>> 0.1 + 0.1 + 0.1 == 0.3
False
```

Floating point rounding errors are not unique to Python - they're fundamental to how floats work.

### The Textual Problem: Layout Precision

Layouts require dividing screen based on varying proportions:
- Panel that's 1/3 of screen width
- Remaining 2/3 further divided
- Rounding errors → single character gaps where there should be content

Visual artifacts from float imprecision.

### The Solution: Replace Floats with Fractions

**Fractions don't suffer from rounding errors** in the same way floats do.

```python
>>> from fractions import Fraction as F
>>> F(1, 10) + F(1, 10) + F(1, 10) == F(3, 10)
True
```

Three tenths add up to three tenths - exact arithmetic.

### Real Example: Character Division

```python
# Float version (first row): one character short
# Fraction version (second row): exact division
# ------------------------
# 00011122223334444555666
# 000111222233344445556666
```

The Fraction version gives pixel-perfect layouts with no rounding artifacts.

### Practical Use

Once you have a `Fraction` object, you can use it in place of floats. The benefit: exact division and proportional calculations without cumulative rounding errors.

### Takeaway

Don't dismiss stdlib modules as "too specialized" - they often solve real-world problems elegantly.

---

## Lesson 7: Emojis Are Terrible

### The Hard Truth

Emoji support has been an ongoing problem in Rich since its conception, inherited by Textual. It was **top of the list** when Textualize was founded in January 2022.

**Quote**: "We had big plans, but the more we looked into this issue, the worse it got."

### The Core Problem: Variable Width

**Background**: When writing a character to terminal, it may be one of **three sizes**:
1. Single width (Western alphabet)
2. Double width (Chinese, Japanese, Korean characters)
3. Zero width (combining characters)

**Impact**: Formatting (centering, boxes around text) requires knowing width. Can't use `len(text)` for in-terminal width.

### The Partial Solution: Unicode Database

Unicode database contains mapping of character widths.
Rich/Textual look up **every character** before printing.
- Not cheap operation
- With engineering effort + caching (see lru_cache): fast enough

### Why This Doesn't Solve Emoji

**Problem 1: Unicode Versions Change**
- Asian characters: stable
- Emoji: **every new Unicode release adds new batch**
- Newer emoji render unpredictably: single/double width varies by terminal

**Problem 2: No Reliable Version Detection**
- No standard environment variable
- Heuristic: write sequences, ask for cursor position
- **Unreliable**: terminals still render emoji unpredictably even when you think you know Unicode version

**Problem 3: Multi-Codepoint Emoji**

Example: 👨🏻‍🦰 (man, light skin tone, red hair) = **4 codepoints**

Different terminals render this as:
- 4 individual characters
- 2 characters
- 1 character (single or double width)
- 4 "?" characters

**Fundamental Problem**: Even if you implement code to understand multi-codepoint characters, you **can't tell what the output will be** in a given terminal.

### The Practical Workaround

**Reality**: "It's a mess for sure, but in practice it's not that bad."

**Safe Strategy**:
- Stick to emoji in **Unicode version 9** of the database
- Reliable across all platforms
- Avoid newer emoji
- Avoid multi-codepoint characters (even if they look okay on your terminal)

### Lesson Learned

Emoji in terminals are fundamentally unreliable due to:
- Unpredictable terminal implementations
- No version detection mechanism
- Multi-codepoint complexity
- Evolving Unicode standards

If you need reliability, stick to Unicode 9 emoji set.

---

## Meta: Textualize Was Hiring

Article ends with job posting - looking for Python developers with:
- Very strong Python skills
- Web experience
- Experience with another language
- Good API design skills

**Goal**: "Build a TUI framework that will eat some of the browser's lunch"

This reveals Textualize's ambition: TUIs as serious alternative to web UIs for certain use cases.

---

## Key Takeaways for TUI Developers

1. **60fps is achievable** - use overwrite, single writes, and synchronized output
2. **DictViews have set operations** - don't reinvent the wheel
3. **lru_cache is legitimately fast** - use liberally for small frequently-called functions
4. **Immutability reduces bugs** - easier to reason about, cache, and test
5. **Unicode diagrams in docstrings** - worth the effort for complex concepts
6. **Fractions solve layout precision** - no more rounding artifact gaps
7. **Emoji are unreliable** - stick to Unicode 9 for cross-platform consistency

---

## Cross-References

**Related Topics**:
- Performance optimization patterns → See tutorials on async and caching
- Terminal limitations → See architecture docs on escape sequences and capabilities
- Layout algorithms → See architecture docs on CSS and positioning

**Implementation Examples**:
- Synchronized output usage → Check Textual source for frame rendering
- Fraction-based layouts → Check Textual grid/layout implementation
- DictView set operations → Check widget rendering and diffing code

---

## Sources

**Primary Source**:
- [7 Things I've Learned Building a Modern TUI Framework](https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework/) - Will McGugan, August 3, 2022 (accessed 2025-11-02)

**Referenced Specs**:
- [Synchronized Output Protocol](https://gist.github.com/christianparpart/d8a62cc1ab659194337d73e399004036)
- [CPython lru_cache (Python)](https://github.com/python/cpython/blob/main/Lib/functools.py#L566)
- [CPython lru_cache (C implementation)](https://github.com/python/cpython/blob/main/Modules/_functoolsmodule.c#L992)

**Tools Mentioned**:
- [Monodraw](https://monodraw.helftone.com/) - macOS Unicode diagram tool
