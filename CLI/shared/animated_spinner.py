"""
AnimatedSpinner - Custom loading spinner with random special characters

COOL SPINNER - 42 chars total (4 rotation + 38 random special)
Classic rotation: │ ╱ ─ ╲ (always in order)
Random special chars between rotations: * · • ∘ ○ ~ ∿ ≈ ◇ ◆ × + ÷ - and more!

Pattern: │ ╱ ─ ╲ [random] │ ╱ ─ ╲ [random] ...

GIL & Performance Note:
- Animates at 8 FPS (125ms per frame) using Textual's set_interval()
- May freeze briefly during heavy subprocess work (Python GIL blocks main thread)
- Breathe points (time.sleep) in background work help reduce freeze duration
- render() MUST be pure and fast - NO file I/O! (blocks every frame)
"""

# <claudes_code_comments>
# ** Function List **
# AnimatedSpinner.__init__() - Initialize spinner widget
# AnimatedSpinner.on_mount() - Start animation timer (8 FPS)
# AnimatedSpinner.on_unmount() - Stop animation timer
# AnimatedSpinner.render() - Render next spinner character (MUST be pure and fast!)
#
# ** Technical Review **
# Custom Textual widget that displays an animated loading spinner with random special characters.
# Uses get_next_spinner_char() from cool_spinner module for pattern: │╱─╲[random]│╱─╲[random]...
#
# Animates at 8 FPS (125ms per frame) for smooth rotation without flickering.
# Each frame calls render() which gets the next character from the centralized sequence.
#
# CRITICAL Performance Rule:
# render() MUST be pure and fast! NO file I/O, NO network calls, NO expensive operations!
# Any blocking in render() will freeze the spinner AND the entire event loop.
# We learned this the hard way - file logging in render() blocked every frame!
#
# Timer System:
# - set_interval(0.125, self.refresh) schedules refresh() on main event loop
# - Refresh calls render() which must return immediately
# - Timer may miss callbacks if GIL is blocked (subprocess work on background thread)
# - Breathe points (time.sleep) in background workers help release GIL briefly
#
# Integration: Drop-in replacement for LoadingIndicator() in loading overlays.
# </claudes_code_comments>

from textual.widget import Widget

# COOL SPINNER - Import random char generator (42 chars: 4 rotation + 38 special)
from .cool_spinner import get_next_spinner_char


class AnimatedSpinner(Widget):
    """Custom animated loading spinner with random special characters"""

    DEFAULT_CSS = """
    AnimatedSpinner {
        width: auto;
        height: 3;
        content-align: center middle;
        text-align: center;
        color: $accent;
    }
    """

    def __init__(self):
        super().__init__()
        self._timer = None

    def on_mount(self) -> None:
        """
        Start spinner animation when mounted.

        Animates at 8 FPS (125ms per frame) - fast enough to look smooth,
        slow enough to not consume too many resources.
        """
        # Animate at ~8 FPS (125ms per frame)
        self._timer = self.set_interval(0.125, self.refresh)

    def on_unmount(self) -> None:
        """Stop animation when widget unmounts"""
        if self._timer:
            self._timer.stop()

    def render(self) -> str:
        """Render next spinner character (random special chars between rotations!)"""
        char = get_next_spinner_char()
        return f" {char}"  # Preceding space for alignment
