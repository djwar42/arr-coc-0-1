# First Impressions: Honest Pros, Cons, and Gotchas

**Source**: [Building TUIs with textual: first impressions](https://learnbyexample.github.io/textual-first-impressions/) (accessed 2025-11-02)

**Author Context**: Experienced with Python CLI/scripting (20 years), newbie with classes/OOP, bad experience with Android GUI development, successfully built tkinter apps previously.

**Project**: 4x4 Square Tic Tac Toe (similar to regular Tic Tac Toe but form a square instead of a line)

---

## What I Liked ✓

### Live Editing Mode
```bash
textual run --dev script.py
```
**Impact**: "Very helpful while trying out layout combinations, margin, padding, etc."

**Key insight**: CSS experimentation without restart - critical for visual iteration.

### Default Colors & Theming
- Beautiful defaults out-of-box (no color picking needed)
- Easy dark/light theme switching built-in
- Author only set header and status background colors

**Note**: Author struggled to make light theme default (worked around by explicitly calling toggle method)

### Code Simplicity
- "Significantly shorter compared to tkinter version"
- "Much easier to reason about"
- tkinter version required "days shifting through stackoverflow threads and tkdocs"

**Why it matters**: Lower cognitive load despite author's OOP inexperience.

---

## What Gave Me Trouble ✗

### Layout Alignment Struggles
**Problem**: Status text (left), 4x4 board (center), controls (right) wouldn't align properly - "too much spacing around the 4x4 board"

**Attempted approach**: Give board 50-60% width, divide remainder for status/controls.

**Solution that worked**:
- Status: 20% width
- Controls: 25% width
- Board: **no width value assigned** (let it flex)

**Lesson**: Sometimes NOT setting a dimension works better than explicit percentage.

### Widget Discovery Issues

**Problem 1 - Text display widget**:
- Couldn't find "textbox" widget for status display
- Initially used Button widget (wrong choice)
- Found `Static` widget but documentation didn't show `update()` method clearly
- Discovered `update()` only when reading **Widgets guide** (not widget-specific docs)

**Lesson**: Widget-specific docs incomplete - check general Widgets guide for methods.

**Problem 2 - Event handling**:
- Single `on_button_pressed()` method handles ALL button clicks
- Author found it strange - "I'd prefer a way to bind a method to the buttons, like tkinter provides"

**Implication**: Need to use button IDs/logic to route events, not direct method binding per button.

### Documentation Discovery

**Hidden feature found too late**:
> "As I looked up the Devtools page to link here in this blog post, I found that there's a `console` command for `print()` based debugging! That would've been handy while I was working on the game — sigh, I should've been more proactive in exploring the documentation site."

**Lesson**: Read Devtools page FIRST, don't discover debugging tools after finishing project.

---

## Refactoring Patterns

### From tkinter to Textual

**Game logic**: Copied from tkinter version (mostly unchanged)

**UI code**: Complete rewrite but simpler

**Features**: tkinter version had more features, but Textual felt easier to reason about

**Code volume**: Textual = significantly shorter

---

## "What I Wish I Knew" Insights

### Before Starting
1. **Read Devtools page** - Console debugging available via `textual console`
2. **Check Widgets guide** for methods - widget-specific docs may be incomplete
3. **Don't over-specify layout** - sometimes leaving dimensions unset works better
4. **Light theme default** - Not obvious how to set (may need explicit toggle call)

### During Development
5. **Use `Static.update()`** for dynamic text - not immediately obvious from Static widget docs
6. **Button events are centralized** - expect single handler, not per-button callbacks
7. **Live editing is essential** - Use `--dev` mode for any visual work

### Workflow
8. **Incremental feature addition** works well (author tweeted progress after each step)
9. **Copy game logic, rewrite UI** - Sensible separation of concerns

---

## Gotchas

### Layout Percentage Math
❌ Board 50-60%, split remainder → broken alignment
✓ Specific percentages for sides, **no value** for center → works

### Widget Documentation Completeness
❌ Widget page shows widget exists
✓ Widgets guide shows actual methods available

### Event Binding Model
❌ tkinter-style per-button binding
✓ Textual-style centralized `on_button_pressed()` with routing logic

### Feature Discovery
❌ Assuming Getting Started covers all dev tools
✓ Check Devtools page explicitly for debugging utilities

---

## Practical Tips

**Setup**:
```bash
pip install 'textual[dev]'  # Include dev tools from start
textual run --dev script.py # Always use live editing
```

**Development approach**:
1. Start with builtin demo: `python -m textual`
2. Read tutorial (Stopwatch app)
3. **Read Devtools page** (don't skip!)
4. Check GitHub examples for similar patterns
5. Build incrementally, test each feature

**When stuck**:
- Layout issues → Try removing explicit dimensions
- Widget methods → Check Widgets guide, not just widget-specific page
- Debugging → Use `textual console` (read Devtools first!)

---

## Comparison: Textual vs tkinter (Author's Experience)

| Aspect | Textual | tkinter |
|--------|---------|---------|
| Code length | Significantly shorter | Longer |
| Reasoning difficulty | Much easier | Required days of stackoverflow |
| Default aesthetics | Beautiful | Manual color/theme work |
| Documentation flow | Some methods hidden in guides | Centralized in tkdocs |
| Event binding | Centralized handler | Per-widget callbacks |
| Layout approach | CSS-like, percentage-based | Grid/pack managers |

**Author's verdict**: "Textual felt much easier to reason about" (despite having fewer features in v1)

---

## Sources

**Original Article**: [Building TUIs with textual: first impressions](https://learnbyexample.github.io/textual-first-impressions/) (accessed 2025-11-02)

**Code Example**: [Square Tic Tac Toe GitHub repo](https://github.com/learnbyexample/TUI-apps/tree/main/SquareTicTacToe)

**tkinter Comparison**: [Square Tic Tac Toe tkinter version](https://learnbyexample.github.io/practice_python_projects/square_tic_tac_toe/square_tic_tac_toe.html)

**Referenced Textual Docs**:
- [Getting Started](https://textual.textualize.io/getting_started/)
- [Tutorial](https://textual.textualize.io/tutorial/) (Stopwatch app)
- [Devtools](https://textual.textualize.io/guide/devtools/)
- [Widgets guide](https://textual.textualize.io/guide/widgets/)
- [Static widget](https://textual.textualize.io/widgets/static/)

**Author**: learnbyexample (Sundeep Agarwal) - November 15, 2022
