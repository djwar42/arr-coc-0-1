# UPiano - Interactive Terminal Piano Simulator

**Project**: Terminal-based piano simulator with MIDI synthesis
**GitHub**: https://github.com/eliasdorneles/upiano
**Technology Stack**: Textual, FluidSynth, pyFluidSynth
**Stars**: 678+ | **Author**: Elias Dorneles

## Overview

UPiano is a fully functional piano simulator running entirely in the terminal. It demonstrates advanced interactive patterns in Textual, including real-time audio integration, complex custom widget composition, mouse-based interaction with visual highlighting, and keyboard mapping for musical input.

**Key Innovation**: Renders a visually authentic piano keyboard using ASCII box-drawing characters, with separate upper (black keys) and lower (white keys) widget layers for realistic appearance and interaction.

From [UPiano README](https://github.com/eliasdorneles/upiano/blob/master/README.md) (accessed 2025-11-02):
- Play notes with mouse clicks or computer keyboard
- 128 General MIDI instruments via FluidSynth synthesizer
- Real-time controls: transpose, octave shift, sustain, volume, reverb, chorus
- Dynamic keyboard width - more terminal space = more keys available
- Mouse "swiping" - drag across keys to play glissando

## Architecture

### Widget Composition Strategy

**Two-Layer Piano Rendering** (from `keyboard_ui.py`):

```python
class KeyboardWidget(Widget):
    def compose(self):
        # Upper layer: black keys portion
        with Horizontal():
            for w in self.note_upper_widgets:
                yield w

        # Lower layer: white keys portion
        with Horizontal():
            for w in self.note_lower_widgets:
                yield w
```

This creates the visual illusion of overlapping black and white piano keys:
- **Upper widgets**: Render black keys (sharps/flats) and gaps between them
- **Lower widgets**: Render white keys base section
- Together they form a cohesive piano keyboard appearance

### Note Rendering System

From [note_render.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/note_render.py) (accessed 2025-11-02):

**Upper Part Rendering** (8 lines tall):
```python
def render_upper_part_key(note, first_corner=False, last_corner=False,
                          highlight=False, use_rich=False):
    normalized_note = re.sub("[0-9]", "", note).upper()

    if normalized_note in ("C#", "D#", "F#", "G#", "A#"):
        # Black key: full 3-char width box
        filling = ("#" if highlight else "█") * 3
        return "\n".join([
            "┬───",
            "│" + filling,
            "│" + filling,
            "│" + filling,
            "│" + filling,
            "│" + filling,
            "│" + filling,
            "└─┬─",
        ])

    if normalized_note in ("D", "G", "A"):
        # Gap between black keys (1 char width)
        return "\n".join(["┬", "│", "│", "│", "│", "│", "│", "╯"])

    if normalized_note in ("C", "E", "F", "B"):
        # White key upper section (2-4 char width)
        fill_char = "#" if highlight else " "
        filling = fill_char * 2
        # ... corner handling for first/last keys ...
```

**Lower Part Rendering** (5 lines tall):
```python
def render_lower_part_key(is_first=False, is_last=False,
                          highlight=False, use_rich=False):
    # Standard white key base (5 chars wide)
    text = """
│
│
│
│
┴────
    """.strip()

    if use_rich:
        color = "red" if highlight else "white"
        text = text.replace("    ", f"[on {color}]    [/]")
```

**Result**: Authentic piano appearance with proper spacing and proportions.

## Interactive Patterns

### Mouse Interaction with Visual Feedback

From [keyboard_ui.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/keyboard_ui.py) (accessed 2025-11-02):

**Mouse Event Handling Mixin**:
```python
class KeyPartMouseMixin:
    def on_mouse_down(self, event):
        MOUSE_STATUS.pressed = True
        MOUSE_STATUS.black_key_pressed = "#" in self.key.note
        if event.button == 1:
            self.post_message(KeyDown(self.key))
            self.highlight = True

    def on_mouse_up(self, event):
        MOUSE_STATUS.pressed = False
        MOUSE_STATUS.black_key_pressed = False
        if event.button == 1:
            self.post_message(KeyUp(self.key))
            self.highlight = False

    def on_enter(self, event):
        # Handle mouse dragging/"swiping" across keys
        if MOUSE_STATUS.pressed:
            # Prevent white key triggering when black key is held
            if MOUSE_STATUS.black_key_pressed and "#" not in self.key.note:
                return
            self.post_message(KeyDown(self.key))
            self.highlight = True

    def on_leave(self, event):
        self.highlight = False
        self.post_message(KeyUp(self.key))
```

**Key Innovation**: Global `MOUSE_STATUS` tracker enables "swiping" - drag mouse across keyboard to play glissando. The black/white key discrimination prevents unwanted white key triggers when dragging over black keys.

### Reactive Highlighting

**Upper Key Widget**:
```python
class KeyUpperPart(Static, KeyPartMouseMixin):
    highlight = reactive(False)

    def watch_highlight(self, value):
        rendered = render_upper_part_key(
            self.key.note,
            first_corner=self.key.position == 0,
            last_corner=self.key.position == len(NOTES) - 1,
            use_rich=True,
            highlight=value,  # Switches color: white → red
        )
        self._content_width = len(rendered.splitlines()[0])
        self.update(rendered)  # Re-render on highlight change
```

**Pattern**: `reactive(False)` attribute automatically triggers `watch_highlight()` when changed, which re-renders the key with new color. Clean separation between state and rendering.

### Computer Keyboard Mapping

From [keyboard_ui.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/keyboard_ui.py) (accessed 2025-11-02):

```python
KEYMAP_CHAR_TO_INDEX = {
    "A": 0,   # C3
    "W": 1,   # C#3
    "S": 2,   # D3
    "E": 3,   # D#3
    "D": 4,   # E3
    "F": 5,   # F3
    "T": 6,   # F#3
    "G": 7,   # G3
    "Y": 8,   # G#3
    "H": 9,   # A3
    "U": 10,  # A#3
    "J": 11,  # B3
    "K": 12,  # C4
    "O": 13,  # C#4
    "L": 14,  # D4
    "P": 15,  # D#4
    ";": 16,  # E4
    "'": 17,  # F4
}
```

Visual keyboard layout (from README):
```
    ┌─┬──┬┬──┬─┬─┬──┬┬──┬┬──┬─┬─┬──┬┬──┬─┬─┬──┬┐
    │ │██││██│ │ │██││██││██│ │ │██││██│ │ │██││
    │ │W█││E█│ │ │T█││Y█││U█│ │ │O█││P█│ │ │██││
    │ └┬─┘└┬─┘ │ └┬─┘└┬─┘└┬─┘ │ └┬─┘└┬─┘ │ └┬─┘│
    │A │ S │ D │F │ G │  H│ J │K │ L │ ; │' │  │
    └──┴───┴───┴──┴───┴───┴───┴──┴───┴───┴──┴──┘
```

**App-Level Key Handler**:
```python
class MyApp(App):
    def on_key(self, event):
        key = event.key.upper()
        note_index = KEYMAP_CHAR_TO_INDEX.get(key)
        if note_index is not None:
            self.keyboard_widget.play_key(note_index)
```

**Keyboard Widget Play Method**:
```python
def play_key(self, key_index):
    virtual_key = self.virtual_keys[key_index]
    self.handle_key_down(virtual_key)
    # Auto-release after 0.3s (terminal doesn't send key-up events)
    self.set_timer(0.3, partial(self.handle_key_up, virtual_key))
```

**Terminal Limitation**: Terminals receive character streams, not key press/release events. Solution: auto-release notes after 300ms timeout.

## Audio Integration

### FluidSynth MIDI Synthesis

From [midi.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/midi.py) (accessed 2025-11-02):

```python
import fluidsynth

class MidiSynth:
    def __init__(self, soundfont_name=None):
        self.synthesizer = fluidsynth.Synth()
        self.synthesizer.start()
        self.soundfont_id = self.load_soundfont(
            soundfont_name or "GeneralUser_GS_v1.471.sf2"
        )
        self.select_midi_program(0)  # Acoustic Grand Piano

    def note_on(self, note_value, channel=0, velocity=100):
        self.synthesizer.noteon(channel, note_value, velocity)

    def note_off(self, note_value, channel=0):
        self.synthesizer.noteoff(channel, note_value)

    def set_sustain(self, value, channel=0):
        self.synthesizer.cc(channel, 64, value)  # MIDI CC 64

    def set_volume(self, value, channel=0):
        self.synthesizer.cc(channel, 7, value)   # MIDI CC 7

    def set_chorus(self, value, channel=0):
        self.synthesizer.cc(channel, 93, value)  # MIDI CC 93

    def set_reverb(self, value, channel=0):
        self.synthesizer.cc(channel, 91, value)  # MIDI CC 91
```

**MIDI Note Calculation**:
```python
def note_to_midi(note: str) -> int:
    """
    Convert note string to MIDI value.
    >>> note_to_midi("C4")
    60  # Middle C
    >>> note_to_midi("C#4")
    61
    """
    octave = int(note[-1])
    is_sharp = note[1] == "#"
    return 12 * (octave + 1) + "C D EF G A B".index(note[:1]) + int(is_sharp)
```

**Integration Pattern**: Global `synthesizer` instance connects keyboard events to audio output via callback functions.

### Note Triggering Architecture

From [app.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/app.py) (accessed 2025-11-02):

```python
@dataclass
class KeyboardPlayingSettings:
    octave: int = 0      # Octave shift: -3 to +3
    transpose: int = 0   # Transpose: -11 to +11 semitones

PLAY_SETTINGS = KeyboardPlayingSettings()

def transpose_note(note_value: int):
    """Apply current transpose and octave settings"""
    return note_value + PLAY_SETTINGS.transpose + PLAY_SETTINGS.octave * 12

def tranposed_note_on(note_value: int):
    synthesizer.note_on(transpose_note(note_value), velocity=100)

def transposed_note_off(note_value: int):
    synthesizer.note_off(transpose_note(note_value))

# Keyboard widget receives these callbacks
self.keyboard_widget = KeyboardWidget(
    note_on=tranposed_note_on,
    note_off=transposed_note_off,
)
```

**Pattern**: Callback functions passed to widget constructor, allowing separation of UI (keyboard widget) from audio logic (synthesizer).

## Custom Controls

### NumericUpDownControl Widget

From [widgets.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/widgets.py) (accessed 2025-11-02):

```python
class NumericUpDownControl(Widget):
    DEFAULT_CSS = """
    NumericUpDownControl {
        height: 5;
    }
    NumericUpDownControl Button {
        min-width: 5;
    }
    NumericUpDownControl .value-holder {
        min-width: 2;
        border: thick $primary;
        text-style: bold;
        padding: 0 1;
        text-align: right;
    }
    NumericUpDownControl .value-holder.modified {
        border: thick $secondary;  # Visual feedback when not at default
    }
    """

    value = reactive(0)

    def __init__(self, label, watch_value=None, min_value=-11, max_value=11):
        super().__init__()
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.watch_value = watch_value  # Callback for value changes

    def compose(self):
        yield Label(self.label)
        with Horizontal():
            yield Button("⬇️ ", id=f"{self.id}-down")
            yield Label("0", id=f"{self.id}-value", classes="value-holder")
            yield Button("⬆️ ", id=f"{self.id}-up")

    def on_button_pressed(self, event):
        if event.button.id == f"{self.id}-down":
            if self.value > self.min_value:
                self.value -= 1
        elif event.button.id == f"{self.id}-up":
            if self.value < self.max_value:
                self.value += 1

        label = self.query_one(f"#{self.id}-value", Label)
        label.update(str(self.value))

        # Visual feedback: change border when modified
        if self.value == 0:
            label.remove_class("modified")
        else:
            label.add_class("modified")
```

**Usage Pattern**:
```python
yield NumericUpDownControl(
    "Transpose",
    lambda value: setattr(PLAY_SETTINGS, "transpose", value),
    min_value=-11,
    max_value=11,
)
```

### Custom Slider Widget

**ASCII Slider Design**:
```python
_NAKED_SLIDER = """
 ╷         ╷         ╷
 ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
 ╵         ╵         ╵
""".strip()

class Slider(Widget):
    position = reactive(0)  # 0-20 (21 positions)
    can_focus = True

    DEFAULT_CSS = """
    Slider {
        height: 3;
    }
    Slider:focus {
        background: $primary-lighten-1;
        max-width: 23;
        layers: base-layer top-layer;  # Key: overlay button on track
    }
    .slider-button {
        width: 3;
        height: 3;
        background: $primary;
        layer: top-layer;  # Render on top of track
        border: none;
        border-top: tall $panel-lighten-2;
        border-bottom: tall $panel-darken-3;
    }
    """

    def compose(self):
        yield Static(_NAKED_SLIDER)  # Base layer: track
        yield Static(classes="slider-button")  # Top layer: button

    def _watch_position(self, position):
        button = self.query_one("Static.slider-button", Static)
        button.styles.margin = (0, 0, 0, position)  # Move left margin
        self.post_message(self.PositionUpdate(position))

    def on_key(self, event):
        if event.key == "left":
            self.position = max(0, self.position - 1)
        elif event.key == "right":
            self.position = min(20, self.position + 1)

    def on_mouse_scroll_up(self, event):
        self.position = max(0, self.position - 1)

    def on_mouse_scroll_down(self, event):
        self.position = min(20, self.position + 1)
```

**Layer System**: Uses Textual's layer feature to overlay slider button on top of track. Button position controlled via left margin (0-20 characters).

**LabeledSlider Wrapper**:
```python
class LabeledSlider(Widget):
    def __init__(self, label, watch_value=None, value=100, value_range=(0, 127)):
        self.label = label
        self.min_value, self.max_value = value_range
        self.watch_value = watch_value
        self.value = value

    def compose(self):
        yield Label(self.label)
        # Map value (0-127) to slider position (0-20)
        initial_position = self.value * 20 // (self.max_value - self.min_value)
        yield Slider(position=initial_position)

    def on_slider_position_update(self, event):
        # Map slider position back to value range
        self.value = event.position * (self.max_value - self.min_value) // 20
```

**Usage**:
```python
yield LabeledSlider(
    "Volume",
    lambda value: synthesizer.set_volume(value),
    value=100,  # Default: full volume
    value_range=(0, 127)  # MIDI standard range
)
```

## Main Application Structure

From [app.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/app.py) (accessed 2025-11-02):

```python
class MyApp(App):
    BINDINGS = [
        ("ctrl-c", "quit", "Quit"),
        ("insert", "toggle_sustain", "Toggle sustain"),
    ]
    CSS_PATH = "style.css"
    TITLE = "UPiano"
    SUB_TITLE = "A piano in your terminal"

    def compose(self):
        yield Header()
        yield Footer()

        self.keyboard_widget = KeyboardWidget(
            note_on=tranposed_note_on,
            note_off=transposed_note_off,
        )

        with Container(id="main"):
            with Container(id="controls"):
                yield InstrumentSelector()
                yield NumericUpDownControl("Transpose", ...)
                yield NumericUpDownControl("Octave", ...)
                yield LabeledSwitch("Sustain", ...)
                yield LabeledSlider("Volume", ...)
                yield LabeledSlider("Reverb", ...)
                yield LabeledSlider("Chorus", ...)

            yield self.keyboard_widget

    def action_toggle_sustain(self):
        self.query_one(LabeledSwitch).toggle()
```

**Layout**: Controls sidebar + keyboard main area. All controls update global synthesizer or settings via callbacks.

## Key Textual Patterns Demonstrated

### 1. Custom Content Size Control

```python
class KeyUpperPart(Static):
    def get_content_height(self, *args, **kwargs):
        return 8  # Fixed height

    def get_content_width(self, *args, **kwargs):
        return self._content_width  # Dynamic based on rendered content
```

### 2. Message-Based Communication

```python
class KeyDown(Message):
    def __init__(self, key: Key):
        super().__init__()
        self.key = key

# Widget posts message
self.post_message(KeyDown(self.key))

# Parent handles message
def on_key_down(self, event):
    self.handle_key_down(event.key)
```

### 3. Reactive Attributes with Watchers

```python
highlight = reactive(False)

def watch_highlight(self, value):
    # Auto-called when highlight changes
    self.update(render_upper_part_key(..., highlight=value))
```

### 4. Mixin Pattern for Shared Behavior

```python
class KeyPartMouseMixin:
    # Shared mouse handling logic
    def on_mouse_down(self, event): ...
    def on_mouse_up(self, event): ...

class KeyUpperPart(Static, KeyPartMouseMixin):
    # Inherits all mouse handling
    pass
```

### 5. CSS Layers for Overlays

```python
DEFAULT_CSS = """
Slider:focus {
    layers: base-layer top-layer;
}
.slider-button {
    layer: top-layer;  # Render on top
}
"""
```

## Installation and Usage

**Dependencies**:
```bash
pip install upiano
```

**External Requirement**: FluidSynth synthesizer must be installed on system:
- macOS: `brew install fluid-synth`
- Ubuntu: `apt-get install fluidsynth`
- See: https://github.com/FluidSynth/fluidsynth/wiki/Download

**Running**:
```bash
upiano
```

**Performance Note**: On Ubuntu with Pipewire, reduce latency with:
```bash
PIPEWIRE_QUANTUM=256/48000 upiano
```

## Project Evolution

From [README](https://github.com/eliasdorneles/upiano/blob/master/README.md) (accessed 2025-11-02):

**Original Version (2017)**:
- Built with urwid library
- Used sox subprocesses for audio
- Available as `python upiano/legacy.py`

**Modern Version (2023)**:
- Rebuilt with Textual after author attended EuroPython
- FluidSynth integration for true MIDI synthesis
- Added synthesizer controls (reverb, chorus, volume)
- Mouse swiping support
- Sustain pedal control

**Release History**:
- v0.1.0: Initial PyPI release with Textual
- v0.1.1: Added sustain, improved mouse handling
- v0.1.2: Volume/reverb/chorus sliders, Python 3.10+ support

## Key Learnings for Textual Developers

**1. Complex Widget Composition**: Two-layer rendering creates illusion of overlapping elements (piano keys).

**2. Real-Time Interaction**: Global state (`MOUSE_STATUS`) enables advanced gestures like swiping across keys.

**3. External Integration**: Demonstrates integrating C libraries (FluidSynth) via Python bindings with Textual UI.

**4. Terminal Limitations Workaround**: Auto-release timer compensates for lack of key-up events.

**5. Custom Widget Patterns**:
- Reactive attributes for automatic UI updates
- Message-based parent-child communication
- Mixins for shared behavior
- CSS layers for visual overlays

**6. Callback Architecture**: Widgets accept callback functions for loose coupling between UI and logic.

**7. Dynamic Sizing**: Terminal width determines number of piano keys rendered - responsive design.

## Sources

**GitHub Repository**:
- [UPiano](https://github.com/eliasdorneles/upiano) - Main repository (accessed 2025-11-02)

**Source Files Analyzed**:
- [README.md](https://github.com/eliasdorneles/upiano/blob/master/README.md) - Project overview
- [app.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/app.py) - Main application
- [keyboard_ui.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/keyboard_ui.py) - Piano keyboard widget
- [widgets.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/widgets.py) - Custom controls
- [midi.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/midi.py) - MIDI synthesis
- [note_render.py](https://github.com/eliasdorneles/upiano/blob/master/upiano/note_render.py) - Key rendering

**Technologies**:
- [Textual](https://textual.textualize.io) - TUI framework
- [FluidSynth](https://www.fluidsynth.org) - Software synthesizer
- [pyFluidSynth](https://github.com/nwhitehead/pyfluidsynth) - Python bindings
