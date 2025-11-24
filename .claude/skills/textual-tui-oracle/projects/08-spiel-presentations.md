# Spiel - Terminal Presentation Framework

## Overview

Spiel is a Python framework for building and presenting richly-styled presentations directly in the terminal using Textual and Rich. It provides a decorator-based API for defining slides, supports slide transitions, animations through triggers, and includes features like hot-reloading during development and a REPL for interactive debugging.

**GitHub Repository**: https://github.com/JoshKarpel/spiel (accessed 2025-11-02)

**Key Features**:
- Decorator-based slide definition
- Rich content rendering (markdown, code highlighting, Rich renderables)
- Slide transitions (swipe animations)
- Trigger system for progressive reveals and animations
- Hot-reload during development
- Multiple views: slide view, deck grid view, help screen
- Key bindings customization per-slide
- REPL integration for debugging

## Core Architecture

### Deck and Slide Model

Spiel uses a simple hierarchical model:
- **Deck**: Collection of slides (a presentation)
- **Slide**: Individual slide with title, content function, bindings, and transition

From [spiel/deck.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/deck.py):

```python
@dataclass
class Deck(Sequence[Slide]):
    """Represents a "deck" of "slides": a presentation."""

    name: str
    default_transition: Type[Transition] | None = Swipe
    _slides: list[Slide] = field(default_factory=list)

    def slide(
        self,
        title: str = "",
        bindings: Mapping[str, Callable[..., None]] | None = None,
        transition: Type[Transition] | None = None,
    ) -> Callable[[Content], Content]:
        """Decorator that creates a new slide in the deck."""
        def slideify(content: Content) -> Content:
            self.add_slides(
                Slide(
                    title=title,
                    content=content,
                    bindings=bindings or {},
                    transition=transition,
                )
            )
            return content
        return slideify
```

**Design Pattern**: The `Deck` acts as both a container (implements `Sequence[Slide]`) and a decorator factory. This dual role makes the API clean while maintaining proper data structure semantics.

### Slide Content Rendering

From [spiel/slide.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/slide.py):

```python
@dataclass
class Slide:
    """Represents a single slide in the presentation."""

    title: str = ""
    content: Content = Text  # Callable[..., RenderableType]
    bindings: Mapping[str, Callable[..., None]] = field(default_factory=dict)
    transition: Type[Transition] | None = Swipe

    def render(self, triggers: Triggers) -> RenderableType:
        signature = inspect.signature(self.content)

        kwargs: dict[str, object] = {}
        if TRIGGERS in signature.parameters:
            kwargs[TRIGGERS] = triggers

        return self.content(**kwargs)
```

**Content Functions**: Slides are defined as functions returning `RenderableType` (any Rich renderable). The function can optionally accept a `triggers` parameter for animations.

**Example**:
```python
@deck.slide(title="Introduction")
def intro() -> RenderableType:
    return Panel("Welcome to Spiel!", border_style="blue")

@deck.slide(title="Animated")
def animated(triggers: Triggers) -> RenderableType:
    # Progressive reveal based on trigger count
    items = ["First", "Second", "Third"]
    return "\n".join(triggers.take(items))
```

## Textual Integration Patterns

### Application Structure

From [spiel/app.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/app.py):

```python
class SpielApp(App[None]):
    CSS_PATH = "spiel.css"
    BINDINGS: ClassVar[List[Binding | Tuple[str, str, str]]] = [
        Binding("d", "switch_screen('deck')", "Go to the Deck view."),
        Binding("question_mark", "push_screen('help')", "Go to the Help view."),
        Binding("i", "repl", "Switch to the REPL."),
        Binding("p", "screenshot", "Take a screenshot."),
    ]

    deck = reactive(Deck(name="New Deck"))
    current_slide_idx = reactive(0)
    message = reactive(Text(""))

    async def on_mount(self) -> None:
        self.deck = load_deck(self.deck_path)
        self.reloader = asyncio.create_task(self.reload())

        self.install_screen(SlideScreen(), name="slide")
        self.install_screen(DeckScreen(), name="deck")
        self.install_screen(HelpScreen(), name="help")
        await self.push_screen("slide")
```

**Pattern: Screen-Based Architecture**
- **SlideScreen**: Main presentation view (single slide)
- **DeckScreen**: Grid overview of all slides
- **HelpScreen**: Keybinding help
- **SlideTransitionScreen**: Temporary screen for animated transitions

### Reactive State Management

```python
deck = reactive(Deck(name="New Deck"))
current_slide_idx = reactive(0)
message = reactive(Text(""))

def watch_deck(self, new_deck: Deck) -> None:
    self.title = new_deck.name

def watch_current_slide_idx(self, new_current_slide_idx: int) -> None:
    self.query_one(SlideWidget).triggers = Triggers.new()
    self.sub_title = self.deck[new_current_slide_idx].title
```

**Pattern**: Reactive watchers automatically update UI elements (title, subtitle) when deck or slide index changes. This eliminates manual update logic.

### Slide Navigation Actions

```python
async def action_next_slide(self) -> None:
    await self.handle_new_slide(self.current_slide_idx + 1, Direction.Next)

async def action_prev_slide(self) -> None:
    await self.handle_new_slide(self.current_slide_idx - 1, Direction.Previous)

async def handle_new_slide(self, new_slide_idx: int, direction: Direction) -> None:
    new_slide_idx = clamp(new_slide_idx, 0, len(self.deck) - 1)

    current_slide = self.deck[self.current_slide_idx]
    new_slide = self.deck[new_slide_idx]
    transition = new_slide.transition or self.deck.default_transition

    if (
        self.current_slide_idx == new_slide_idx
        or not isinstance(self.screen, SlideScreen)
        or transition is None
        or not self.enable_transitions
    ):
        self.current_slide_idx = new_slide_idx
        return

    # Create transition screen
    transition_screen = SlideTransitionScreen(
        from_slide=current_slide,
        from_triggers=self.query_one(SlideWidget).triggers,
        to_slide=new_slide,
        direction=direction,
        transition=transition,
    )
    await self.switch_screen(transition_screen)

    # Animate transition
    transition_screen.animate(
        "progress",
        value=100,
        duration=0.75,
        on_complete=lambda: self.finalize_transition(new_slide_idx),
    )
```

**Pattern: Conditional Transition Screens**: When transitioning between slides, Spiel temporarily switches to a `SlideTransitionScreen` that renders both the current and next slide, animates the transition, then finalizes by switching back to the main slide screen. This keeps the main slide screen simple.

## Slide Transitions

### Transition Protocol

From [spiel/transitions/swipe.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/transitions/swipe.py):

```python
class Swipe(Transition):
    """
    A transition where the current and incoming slide are placed side-by-side
    and gradually slide across the screen.
    """

    def initialize(
        self,
        from_widget: Widget,
        to_widget: Widget,
        direction: Direction,
    ) -> None:
        match direction:
            case Direction.Next:
                to_widget.styles.offset = ("100%", 0)
            case Direction.Previous:
                to_widget.styles.offset = ("-100%", 0)

    def progress(
        self,
        from_widget: Widget,
        to_widget: Widget,
        direction: Direction,
        progress: float,
    ) -> None:
        match direction:
            case Direction.Next:
                from_widget.styles.offset = (f"-{progress:.2f}%", 0)
                to_widget.styles.offset = (f"{100 - progress:.2f}%", 0)
            case Direction.Previous:
                from_widget.styles.offset = (f"{progress:.2f}%", 0)
                to_widget.styles.offset = (f"-{100 - progress:.2f}%", 0)
```

**Pattern: CSS Offset Animation**: Transitions manipulate `widget.styles.offset` to create smooth slide animations. The `Swipe` transition positions the incoming slide off-screen, then gradually moves both slides across the screen.

**Extensibility**: Custom transitions can be created by implementing the `Transition` protocol with `initialize()` and `progress()` methods.

## Trigger System for Animations

From [spiel/triggers.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/triggers.py):

```python
@dataclass(frozen=True)
class Triggers(Sequence[float]):
    """
    Provides information to slide content about the current slide's "trigger state".

    Triggers is a Sequence of times (from time.monotonic) that the current slide
    was triggered at. The first trigger time is when the slide started being displayed.
    """

    now: float  # Current render time
    _times: tuple[float, ...]

    @cached_property
    def time_since_last_trigger(self) -> float:
        """The elapsed time since the most recent trigger."""
        return self.now - self._times[-1]

    @cached_property
    def time_since_first_trigger(self) -> float:
        """Time since the slide started being displayed."""
        return self.now - self._times[0]

    @cached_property
    def triggered(self) -> bool:
        """Returns whether the slide has been manually triggered."""
        return len(self) > 1

    def take(self, iter: Iterable[T], offset: int = 1) -> Iterator[T]:
        """
        Takes elements from the iterable equal to the number of triggers minus offset.
        Defaults to offset=1 to ignore the automatic initial trigger.
        """
        return islice(iter, len(self) - offset)
```

**Pattern: Time-Based Animation State**: `Triggers` is a sequence of monotonic timestamps. Slide content functions can use this to:
1. **Progressive reveals**: Use `triggers.take(items)` to show more items on each trigger
2. **Time-based animations**: Use `time_since_last_trigger` for smooth animations
3. **Conditional rendering**: Use `triggered` to show/hide elements

**Example Usage**:
```python
@deck.slide(title="Bullet Points")
def bullets(triggers: Triggers) -> RenderableType:
    points = [
        "First point",
        "Second point",
        "Third point",
    ]
    # Show one more point each time user presses space (trigger action)
    visible = list(triggers.take(points))
    return "\n".join(f"• {p}" for p in visible)
```

### Trigger Actions

```python
def action_trigger(self) -> None:
    now = monotonic()
    slide_widget = self.query_one(SlideWidget)
    slide_widget.triggers = Triggers(
        now=now,
        _times=(*slide_widget.triggers._times, now)
    )

def action_reset_trigger(self) -> None:
    slide_widget = self.query_one(SlideWidget)
    slide_widget.triggers = Triggers.new()
```

**Pattern**: Triggers are immutable dataclasses. To add a trigger, create a new `Triggers` instance with the appended timestamp. Assigning to the reactive `triggers` property causes the slide to re-render with updated state.

## Hot Reload System

```python
async def reload(self) -> None:
    if self.watch_path is None:
        return

    log(f"Watching {self.watch_path} for changes")
    async for changes in awatch(self.watch_path):
        change_msg = "\n  ".join([""] + [f"{k.raw_str()}: {v}" for k, v in changes])
        log(f"Reloading deck from {self.deck_path} due to detected file changes:{change_msg}")

        try:
            self.deck = load_deck(self.deck_path)
            self.current_slide_idx = clamp(self.current_slide_idx, 0, len(self.deck))
            self.set_message_temporarily(
                Text(
                    f"Reloaded deck at {datetime.datetime.now().strftime(RELOAD_MESSAGE_TIME_FORMAT)}",
                    style=Style(dim=True),
                ),
                delay=10,
            )
        except Exception as e:
            self.set_message_temporarily(
                Text(
                    f"Failed to reload deck at {datetime.datetime.now().strftime(RELOAD_MESSAGE_TIME_FORMAT)} due to: {e}",
                    style=Style(color="red"),
                ),
                delay=10,
            )
```

**Pattern: Async File Watching**: Uses `watchfiles.awatch()` to monitor the presentation file for changes. On change, reloads the deck module and updates the reactive `deck` property, triggering UI refresh. Error handling shows temporary messages without crashing the app.

**UX Detail**: After reload, clamps `current_slide_idx` to ensure it's still valid if slides were added/removed.

## REPL Integration

```python
@cached_property
def repl(self) -> Callable[[], None]:
    # Lazily enable readline support
    try:
        import readline  # noqa: F401, PLC0415
    except ImportError:
        pass

    self.console.clear()
    sys.stdout.flush()

    repl = code.InteractiveConsole()
    return partial(repl.interact, banner="", exitmsg="")

def action_repl(self) -> None:
    with self.suspend():
        self.repl()

@contextmanager
def suspend(self) -> Iterator[None]:
    driver = self._driver

    if driver is not None:
        driver.stop_application_mode()
        with redirect_stdout(sys.__stdout__), redirect_stderr(sys.__stderr__):
            yield
        driver.start_application_mode()
```

**Pattern: Suspended REPL Access**: Pressing 'i' suspends the Textual app (stops application mode), launches a Python REPL with access to the app context, then resumes the app when exiting. This allows live debugging and inspection during presentations.

**Technical Details**:
- Uses `code.InteractiveConsole()` for REPL
- `@cached_property` ensures REPL is initialized once
- Stdout/stderr redirection ensures REPL output goes to terminal, not Textual console

## Module Loading Pattern

```python
def load_deck(path: Path) -> Deck:
    module_name = "__deck"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None:
        raise NoDeckFound(f"{path.resolve()} does not appear to be an importable Python module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)

    try:
        deck = getattr(module, DECK)
    except AttributeError:
        raise NoDeckFound(f"The module at {path} does not have an attribute named {DECK}.")

    if not isinstance(deck, Deck):
        raise NoDeckFound(
            f"The module at {path} has an attribute named {DECK}, but it is a {type(deck).__name__}, not a {Deck.__name__}."
        )

    return deck
```

**Pattern: Dynamic Module Loading**: Spiel loads presentation files as Python modules at runtime using `importlib`. This allows:
1. Hot reloading without restarting the app
2. User code has full Python capabilities
3. Clear error messages if deck is missing/invalid

**Convention**: Expects a module-level `deck` variable containing the `Deck` instance.

## Minimal Example

From [README.md](https://github.com/JoshKarpel/spiel/blob/main/README.md):

```python
from rich.console import RenderableType

from spiel import Deck, present

deck = Deck(name="Your Deck Name")

@deck.slide(title="Slide 1 Title")
def slide_1() -> RenderableType:
    return "Your content here!"

if __name__ == "__main__":
    present(__file__)
```

**Usage**:
```bash
pip install spiel
python deck.py  # Present the deck
```

**Entry Point**:
```python
def present(deck_path: Path | str, watch_path: Path | str | None = None) -> None:
    """
    Present the deck defined in the given deck_path.

    Args:
        deck_path: The file to look for a deck in.
        watch_path: When filesystem changes are detected below this path (recursively),
                    reload the deck from the deck_path.
    """
    os.environ["TEXTUAL"] = ",".join(sorted({"debug", "devtools"}))

    deck_path = Path(deck_path).resolve()
    watch_path = Path(watch_path or deck_path.parent).resolve()

    SpielApp(deck_path=deck_path, watch_path=watch_path).run()
```

## Key Textual Patterns Used

### 1. Screen Management
- Multiple screens for different views (slide, deck grid, help)
- `switch_screen()` for permanent changes
- `push_screen()` for temporary overlays
- Temporary transition screens for animations

### 2. Reactive Properties
- `reactive()` for deck, slide index, messages
- Watchers (`watch_deck`, `watch_current_slide_idx`) for automatic UI updates
- Eliminates manual synchronization between state and display

### 3. Actions and Bindings
- Global bindings in app (d=deck view, ?=help, i=REPL)
- Per-slide bindings for custom interactions
- Action methods (`action_next_slide`, `action_trigger`) for navigation

### 4. Async Patterns
- Async mount for initialization
- Background task for file watching (`asyncio.create_task`)
- Async screen switching for smooth transitions

### 5. Widget Animation
- `widget.animate()` for smooth property changes
- CSS offset manipulation for slide transitions
- Completion callbacks for sequencing animations

### 6. Message System
- Temporary messages with auto-clear timers
- Reactive message property updates footer
- Non-intrusive status updates (reload success/failure, resize info)

## Use Cases

**Developer Presentations**:
- Code-heavy talks with syntax highlighting
- Live-coded demos with REPL access
- Terminal-native presentations for technical audiences

**Quick Demos**:
- Rapid prototyping of presentation ideas
- No need for GUI presentation software
- Version control friendly (pure Python)

**Animated Explanations**:
- Progressive reveals using triggers
- Time-based animations for visualizations
- Custom transitions for emphasis

**Remote/SSH Presentations**:
- Works over SSH without X11 forwarding
- Lightweight, terminal-only requirement
- Perfect for server demonstrations

## Integration Insights

**Rich Integration**:
- All slide content is `RenderableType` from Rich
- Full access to Rich's rendering capabilities (panels, tables, syntax highlighting, markdown)
- Spiel handles the presentation layer, Rich handles the content rendering

**Textual Integration**:
- Built entirely on Textual's screen/widget architecture
- Uses Textual's CSS for styling and animations
- Leverages Textual's reactive system for state management
- Benefits from Textual's input handling and keybindings

## Performance Considerations

**Slide Rendering**:
- Content functions called on every frame (60 FPS by default)
- Keep content functions fast (cached computations, pre-rendered Rich objects)
- Use `@cached_property` for expensive operations

**Hot Reload**:
- File watching uses efficient OS-level notifications
- Module reloading is fast (pure Python)
- Current slide position preserved across reloads

**Transitions**:
- CSS-based animations are efficient
- 750ms default duration (configurable)
- Can disable transitions for performance

## Documentation and Resources

**Official Site**: https://www.spiel.how
**GitHub**: https://github.com/JoshKarpel/spiel
**PyPI**: https://pypi.org/project/spiel

**Quick Start**: https://www.spiel.how/quickstart
**Contributing**: https://www.spiel.how/contributing/

**Demo Deck**:
```bash
# Without installing
docker run -it --rm ghcr.io/joshkarpel/spiel

# After installing
spiel demo present
```

## Sources

**GitHub Repository**:
- [README.md](https://github.com/JoshKarpel/spiel/blob/main/README.md) - Overview and quick start
- [spiel/deck.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/deck.py) - Deck and slide decorator
- [spiel/slide.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/slide.py) - Slide model and rendering
- [spiel/app.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/app.py) - Main Textual application
- [spiel/triggers.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/triggers.py) - Animation trigger system
- [spiel/transitions/swipe.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/transitions/swipe.py) - Swipe transition implementation
- [spiel/__init__.py](https://github.com/JoshKarpel/spiel/blob/main/spiel/__init__.py) - Public API exports

All accessed 2025-11-02
