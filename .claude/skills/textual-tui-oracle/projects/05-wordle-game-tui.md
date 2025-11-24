# Wordle TUI - Game Implementation

## Overview

Wordle TUI is a complete terminal implementation of the popular Wordle game using Textual. Created by frostming, it demonstrates game development patterns including grid-based UI, state management, keyboard input handling, statistics tracking, and Rich rendering integration. The implementation stays faithful to the original web version while adapting seamlessly to terminal environments.

**GitHub Repository**: [frostming/wordle-tui](https://github.com/frostming/wordle-tui)

## Key Features

- **Full Wordle gameplay** - 6 attempts to guess a 5-letter word
- **Grid-based letter display** - Custom grid widget for letter tiles
- **Keyboard visualization** - On-screen keyboard with status indicators
- **Statistics tracking** - Persistent game stats (win rate, streaks)
- **Result sharing** - Copy results to clipboard
- **Daily puzzle system** - Matches web version's daily word rotation
- **ETA countdown** - Shows time until next puzzle

## Installation

```bash
# Direct install from GitHub
pip3 install --user "https://github.com/frostming/wordle-tui/archive/main.zip"

# Run
wordle
```

**Dependencies**:
- `textual` - TUI framework
- `pyperclip` - Clipboard operations
- `platformdirs` - Cross-platform data directory

## Architecture

### Main Components

1. **WordleApp** - Main application coordinator
2. **GuessView** - Grid of letter tiles (6 rows × 5 columns)
3. **KeyboardRow** - On-screen keyboard rows
4. **Letter** - Individual letter button/tile widget
5. **GameMessage** - Message display with countdown
6. **GameStats** - Statistics panel

### Widget Hierarchy

```
WordleApp
├── Header (top)
├── DockView (main)
│   ├── GuessView (letter grid)
│   ├── KeyboardRow × 3 (keyboard)
│   └── Right Side DockView
│       ├── GameMessage (status)
│       └── GameStats (statistics)
```

## Core Implementation Patterns

### 1. Custom Grid Widget - GuessView

The `GuessView` class extends `GridView` to create the 5×6 letter grid:

```python
class GuessView(GridView):
    COLUMN_SIZE = 5
    ROW_SIZE = 6

    def __init__(self, layout: Layout = None, name: str | None = None) -> None:
        super().__init__(layout, name)
        self.slots = [Letter("") for _ in range(self.COLUMN_SIZE * self.ROW_SIZE)]
        self.current = 0

    async def on_mount(self) -> None:
        self.grid.set_align("center", "center")
        self.grid.set_gap(1, 1)
        self.grid.add_column("column", repeat=self.COLUMN_SIZE, size=7)
        self.grid.add_row("row", size=3, repeat=self.ROW_SIZE)
        self.grid.place(*self.slots)
```

**Key patterns**:
- Pre-allocate all 30 letter slots
- Track current position with `self.current`
- Use `grid.set_gap(1, 1)` for spacing
- `grid.place(*self.slots)` for batch widget placement

### 2. Letter Widget - Custom Clickable Tiles

Each letter tile is a custom widget with reactive state:

```python
class Letter(Widget):
    label: Reactive[RenderableType] = Reactive("")
    status: Reactive[int | None] = Reactive(None)

    def __init__(self, name: str, clickable: bool = False):
        super().__init__(name)
        self.name = name
        self.label = name
        self.clickable = clickable
        self.style = IDLE if clickable else EMPTY

    def render(self) -> RenderableType:
        return ButtonRenderable(
            self.label,
            self.style if self.status is None else LETTER_STATUS[self.status],
        )

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default().stop()
        if self.clickable:
            await self.emit(ButtonPressed(self))
```

**Status states** (integer enum):
- `ABSENT = 0` - Letter not in word (gray)
- `PRESENT = 1` - Letter in word, wrong position (yellow)
- `CORRECT = 2` - Letter in correct position (green)

**Style mapping**:
```python
LETTER_STATUS = {
    ABSENT: "bold white on rgb(58,58,58)",
    PRESENT: "bold white on rgb(181,159,59)",
    CORRECT: "bold white on rgb(83,141,78)",
}
```

### 3. Keyboard Input Handling

The app handles both physical keyboard and on-screen keyboard clicks:

```python
class WordleApp(App):
    def on_key(self, event: events.Key) -> None:
        if self.result is not None:
            if event.key == "c":
                self.copy_result()
            return
        self.message.content = ""
        if event.key in string.ascii_letters:
            self.guess.input_letter(event.key.upper())
        elif event.key == "enter":
            self.check_input()
        elif event.key == "ctrl+h":
            self.guess.backspace_letter()

    def handle_button_pressed(self, message: ButtonPressed) -> None:
        if self.result is not None:
            return
        self.message.content = ""
        if message.sender.name == "enter":
            self.check_input()
        elif message.sender.name == "backspace":
            self.guess.backspace_letter()
        else:
            self.guess.input_letter(message.sender.name)
```

**Input patterns**:
- Check game state before accepting input
- Clear messages on new input
- Unified logic for keyboard/clicks via `input_letter()` and `backspace_letter()`

### 4. Letter Input Management

```python
def input_letter(self, letter: str) -> None:
    button = self.slots[self.current]
    if button.name:
        if self.current % self.COLUMN_SIZE == self.COLUMN_SIZE - 1:
            # The last letter is filled
            return
        self.current += 1
        button = self.slots[self.current]
    button.name = letter
    button.label = letter

def backspace_letter(self) -> None:
    button = self.slots[self.current]
    if not button.name:
        if self.current % self.COLUMN_SIZE == 0:
            # the first letter
            return
        self.current -= 1
        button = self.slots[self.current]
    button.name = button.label = ""
```

**State tracking**:
- `self.current` tracks cursor position
- Boundary checks prevent overflow
- Direct manipulation of `Letter` widget properties

### 5. Solution Checking Algorithm

```python
def check_solution(self, solution: str) -> bool | None:
    word = self.current_word
    letters = self.current_guess

    if list(solution) == word:
        for b in letters:
            b.status = CORRECT
        return True

    counter = Counter(solution)
    # First pass: mark exact matches
    for i, b in enumerate(letters):
        if solution[i] == b.name:
            counter[b.name] -= 1
            b.status = CORRECT

    # Second pass: mark present/absent
    for b in letters:
        if b.status == CORRECT:
            continue
        if counter.get(b.name, 0) <= 0:
            b.status = ABSENT
        else:
            counter[b.name] -= 1
            b.status = PRESENT

    if self.current < self.COLUMN_SIZE * self.ROW_SIZE - 1:
        self.current += 1
    else:
        return False  # No more attempts
```

**Two-pass algorithm** (matches original Wordle):
1. Mark all exact position matches (CORRECT), decrement counter
2. For remaining letters, check if letter exists (PRESENT) or not (ABSENT)

**Returns**:
- `True` - User won
- `False` - User lost (6 attempts used)
- `None` - Game continues

### 6. Keyboard State Updates

After each guess, update on-screen keyboard to reflect discovered letters:

```python
def check_input(self) -> bool | None:
    current = self.guess.current_guess
    current_word = "".join(self.guess.current_word).lower()

    if "" in self.guess.current_word:
        self.message.content = "Not enough letters"
        return
    if current_word not in Ta and current_word not in La:
        self.message.content = "Not in word list"
        return

    self.result = self.guess.check_solution(self.solution)

    # Update keyboard button states
    for l in current:
        button = self.buttons[l.name]
        button.status = max(button.status or 0, l.status)

    self.save_statistics()
    if self.result is not None:
        self.show_result()
```

**Key pattern**: `max(button.status or 0, l.status)` ensures keyboard shows best status (CORRECT > PRESENT > ABSENT).

### 7. Statistics Tracking

```python
INITIAL_STATS = {
    "played": 0,
    "stats": [0, 0, 0, 0, 0, 0],  # Win distribution (1-6 attempts)
    "current_streak": 0,
    "max_streak": 0,
}

def save_statistics(self) -> None:
    guesses = self.guess.valid_guesses
    if self.result:
        self.stats["stats"][len(guesses) - 1] += 1

    is_streak = (
        "last_played" in self.stats and self.index - self.stats["last_played"] == 1
    )
    current_streak = self.stats.get("current_streak", 0) if is_streak else 0

    if self.result is not None:
        self.stats["played"] += 1
        current_streak += 1

    max_streak = max(current_streak, self.stats.get("max_streak", 0))

    data = {
        "last_played": self.index,
        "last_guesses": (
            "".join("".join(str(l.name) for row in guesses for l in row)),
            "".join("".join(str(l.status) for row in guesses for l in row)),
        ),
        "last_result": self.result,
        "played": self.stats["played"] + 1,
        "stats": self.stats["stats"],
        "current_streak": current_streak,
        "max_streak": max_streak,
    }

    self.stats.update(data)
    self.stats_view.refresh()

    with open(STATS_JSON, "w") as f:
        json.dump(data, f, indent=2)
```

**Persistence**:
- Stored in `platformdirs.user_data_dir("wordle")/.stats.json`
- Tracks win distribution (attempts 1-6)
- Maintains current and max streaks
- Saves current game state for restoration

### 8. GameStats Widget - Rich Rendering

```python
class GameStats(Widget):
    def render(self) -> RenderableType:
        total_played = self.stats["played"]
        total_win = sum(self.stats["stats"])
        num_guesses = (
            len(self.stats["last_guesses"][0]) // 5 if self.stats["last_result"] else 0
        )

        data = {
            "Played": total_played,
            "Win %": round(total_win / total_played * 100, 1) if total_played else 0,
            "Current Streak": self.stats.get("current_streak", 0),
            "Max Streak": self.stats.get("max_streak", 0),
        }

        table = Table(*data.keys())
        table.add_row(*map(str, data.values()))

        bars = Table.grid("idx", "bar", padding=(0, 1))
        for i, value in enumerate(self.stats["stats"], 1):
            bars.add_row(
                str(i),
                Bar(
                    max(self.stats["stats"]),
                    0,
                    value,
                    color="rgb(83,141,78)"
                    if i == num_guesses and self.stats["last_result"]
                    else "rgb(58,58,58)",
                ),
            )

        render_group = Group(table, bars)
        return RichPanel(render_group, title="Stats")
```

**Rich integration**:
- `Table` for statistics grid
- `Bar` for win distribution histogram
- `Group` to combine multiple renderables
- `RichPanel` as container

### 9. Daily Puzzle System

```python
SEED_DATE = datetime.datetime.combine(datetime.datetime(2021, 6, 19), datetime.time())

def get_index(self) -> int:
    this_date = datetime.datetime.combine(datetime.date.today(), datetime.time())
    return (this_date - SEED_DATE).days

async def on_mount(self) -> None:
    self.index = self.get_index()
    self.solution = La[self.index].upper()
    # ...
```

**Puzzle rotation**:
- Fixed seed date matches original Wordle launch
- Daily index calculated from days elapsed
- Solution selected from answer list `La[index]`

### 10. Game State Restoration

```python
def init_game(self) -> None:
    if self.index > self.stats.get("last_played", -1):
        self.stats["last_result"] = None
        return

    slots = self.guess.slots
    for i, (letter, status) in enumerate(zip(*self.stats["last_guesses"])):
        slots[i].name = slots[i].label = letter
        slots[i].status = int(status)
        self.buttons[letter].status = max(
            self.buttons[letter].status or 0, int(status)
        )

    self.result = self.stats["last_result"]
    self.guess.current = i + 1

    if self.result is not None:
        self.show_result()
```

**State restoration**:
- Check if today's puzzle already started
- Restore letter grid from saved guesses
- Update keyboard states
- Show result if game finished

### 11. Result Sharing

```python
BLOCKS = {ABSENT: "⬛", PRESENT: "🟨", CORRECT: "🟩"}

def copy_result(self) -> None:
    guesses = self.guess.valid_guesses
    trials = len(guesses) if self.result else "x"
    result = [f"Wordle {self.index} {trials}/6", ""]
    for row in guesses:
        result.append("".join(BLOCKS[l.status] for l in row))
    text = "\n".join(result)
    pyperclip.copy(text)

    old_content = self.message.content
    self.message.content = "Successfully copied to the clipboard."

    def restore():
        self.message.content = old_content

    self.message.set_timer(2, restore)
```

**Output format** (matches web version):
```
Wordle 237 4/6

⬛🟨⬛⬛⬛
⬛⬛🟩🟨⬛
🟩🟩🟩⬛🟩
🟩🟩🟩🟩🟩
```

### 12. Countdown Timer

```python
class GameMessage(Widget):
    content: Reactive[str] = Reactive("")

    def show_eta(self, target: datetime.datetime) -> None:
        self.target_date = target
        self.timer = self.set_interval(1, self.refresh)

    async def clear_eta(self) -> None:
        if self.timer is not None:
            await self.timer.stop()
            self.timer = None

    def render(self) -> RenderableType:
        renderable = self.content
        if self.timer is not None:
            eta = calculate_eta(self.target_date)
            if eta is None:
                self._child_tasks.add(asyncio.create_task(self.clear_eta()))
            else:
                renderable += f"\n\nNext wordle: {eta}"
        renderable = Align.center(renderable, vertical="middle")
        return RichPanel(renderable, title="Message")

def calculate_eta(target_date: datetime.datetime) -> str | None:
    units = [3600, 60, 1]
    now = datetime.datetime.now()
    dt = (target_date - now).total_seconds()
    if dt <= 0:
        return None
    digits = []
    for unit in units:
        digits.append("%02d" % int(dt // unit))
        dt %= unit
    return f'[green]{":".join(digits)}[/green]'
```

**Timer pattern**:
- `set_interval(1, self.refresh)` triggers updates every second
- Automatic cleanup when countdown reaches zero
- HH:MM:SS format display

## Layout Structure

```python
async def on_mount(self) -> None:
    view = await self.push_view(DockView())
    header = Header()
    await view.dock(header, edge="top")

    subview = DockView()
    self.guess = GuessView()
    await subview.dock(self.guess, size=26)
    await subview.dock(*keyboard_rows, size=4)

    right_side = DockView()
    self.stats_view = GameStats(self.stats)
    await right_side.dock(self.message, self.stats_view)

    await view.dock(right_side, edge="right", size=40)
    await view.dock(subview, edge="right")
```

**Nested DockView pattern**:
- Main view: header on top, content below
- Subview: guess grid + keyboard rows
- Right side: message + statistics
- Fixed sizes for consistent layout

## Word Lists

```python
with BASE_DIR.joinpath("La.gz").open("rb") as laf, BASE_DIR.joinpath("Ta.gz").open("rb") as taf:
    La: list[str] = json.loads(gzip.decompress(laf.read()))  # Answer list
    Ta: list[str] = json.loads(gzip.decompress(taf.read()))  # Valid guess list
```

**Two word lists**:
- `La` - Answers (solutions, ~2,300 words)
- `Ta` - Valid guesses (accepted words, ~12,000 words)
- Gzipped JSON for compact storage

## Key Textual Patterns Demonstrated

### 1. GridView Usage
- Pre-defined grid dimensions (5×6)
- Gap spacing between cells
- Batch widget placement with `grid.place(*widgets)`

### 2. Reactive Properties
- `Reactive[str]` for dynamic content
- `Reactive[int | None]` for state enums
- Automatic re-rendering on changes

### 3. Custom Widgets
- Subclass `Widget` for custom components
- Override `render()` for custom rendering
- Emit messages for parent communication

### 4. Event Handling
- Global `on_key()` for keyboard input
- Custom message handlers (`handle_button_pressed`)
- Click event handling with `on_click()`

### 5. Rich Integration
- Use Rich renderables in widget `render()` methods
- `Panel`, `Table`, `Bar`, `Group` for complex layouts
- Rich markup in strings (`[green]text[/green]`)

### 6. State Management
- Persistent state via JSON files
- State restoration on app restart
- Reactive properties for UI updates

### 7. Timers and Async
- `set_interval()` for periodic updates
- `set_timer()` for delayed actions
- Async lifecycle methods (`on_mount`, `on_click`)

### 8. Layout Composition
- Nested `DockView` for complex layouts
- Fixed sizing for predictable UI
- Edge-based docking (top, right)

## Lessons for Game Development

### State Tracking
- Track cursor position explicitly (`self.current`)
- Distinguish between UI state (labels) and game state (status)
- Save/restore complete game state

### Input Validation
- Check boundaries before state changes
- Validate input against word lists
- Provide immediate feedback via messages

### Visual Feedback
- Update keyboard as user progresses
- Use color coding for status (RGB values)
- Highlight current game in statistics

### Performance
- Pre-allocate all widgets on mount
- Use reactive properties for minimal updates
- Efficient grid indexing (mod arithmetic)

### User Experience
- Support both keyboard and clicks
- Clear error messages
- Countdown timer for next puzzle
- Easy result sharing

## Comparison with Web Version

**Preserved features**:
- 6 attempts, 5-letter words
- Same color scheme (green/yellow/gray)
- Daily puzzle rotation
- Statistics tracking
- Result sharing format

**Adapted for terminal**:
- On-screen keyboard (not just physical)
- Simplified animations (terminal constraints)
- Text-based UI (no hover effects)
- Clipboard copy instead of share dialog

## Sources

**GitHub Repository**:
- [frostming/wordle-tui](https://github.com/frostming/wordle-tui) - Main repository (accessed 2025-11-02)
- [wordle_app.py](https://github.com/frostming/wordle-tui/blob/main/wordle_app.py) - Complete implementation (422 lines)
- [README.md](https://github.com/frostming/wordle-tui/blob/main/README.md) - Installation and usage

**Key Technologies**:
- [Textual](https://github.com/Textualize/textual) - TUI framework
- [Rich](https://github.com/Textualize/rich) - Terminal rendering
- [pyperclip](https://github.com/asweigart/pyperclip) - Clipboard operations

**Original Game**:
- [Wordle](https://www.powerlanguage.co.uk/wordle/) - Original web version (now owned by NYT)
