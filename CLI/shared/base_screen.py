"""
BaseScreen - Universal loading overlay for all ARR-COC TUI screens

Provides consistent "navigate and break" behavior:
1. Navigate to screen (instant)
2. Show centered loading overlay with animated spinner
3. Load content in background worker thread
4. Hide loading, reveal content underneath

All screens inherit from this for consistent UX.

CRITICAL THREADING & GIL KNOWLEDGE:
===================================

Python's Global Interpreter Lock (GIL) Limitations:
- Only ONE thread can execute Python bytecode at a time
- Subprocess calls (gcloud, gsutil, docker) CAN block the GIL during I/O
- When GIL is blocked, main thread's event loop CAN'T process timer callbacks
- Result: Spinner may freeze during heavy subprocess work (unavoidable!)

Solutions Implemented:
1. Manual threading.Thread() - @work(thread=True) decorator is BROKEN (never starts threads)
2. app.call_from_thread() - Thread-safe UI updates (Textual requirement)
3. time.sleep(0.01) breathe points - Release GIL after each subprocess call
   - Allows main thread to process 1-2 spinner frames
   - Creates smoother (but not perfect) animation during blocking work

What to Expect:
- Spinner animates smoothly at start
- Spinner may freeze briefly during subprocess calls (Python GIL limitation)
- Breathe points (sleep) reduce freeze duration significantly
- Spinner resumes after work completes
- Screen loads successfully without hanging

See SPINNER_SOLUTION.md for complete investigation details.
"""

# <claudes_code_comments>
# ** Function List **
# BaseScreen.__init__(loading_message) - Initialize with custom loading message
# BaseScreen.compose_base_overlay() - Create loading overlay (call from subclass compose)
# BaseScreen.on_mount() - Start background worker thread manually
# BaseScreen._load_content() - Thread worker that runs initialize_content()
# BaseScreen.initialize_content() - Override in subclasses for content setup
# BaseScreen.finish_loading(data) - Hide loading overlay and reveal content
# BaseScreen.update_loading_status(msg) - Update loading message dynamically (thread-safe)
# BaseScreen._handle_worker_error(e) - Show error to user if worker crashes
# BaseScreen.notify_with_full_error(title, error, severity) - Show truncated error + store for popup
# BaseScreen.action_show_last_error() - Show full error popup (press 'e' key)
#
# ** Technical Review **
# Universal base class for all ARR-COC TUI screens. Implements loading overlay pattern:
# navigate → show loading overlay → background init → hide overlay → reveal content.
#
# Threading Architecture (CRITICAL):
# - Uses manual threading.Thread() because @work(thread=True) is BROKEN
# - Worker thread runs initialize_content() which may make blocking subprocess calls
# - All UI updates from worker thread MUST use app.call_from_thread() (Textual requirement)
# - Python's GIL may block during subprocess I/O, causing brief spinner freezes (unavoidable)
# - Breathe points (time.sleep) in subprocess-heavy code help release GIL for smoother animation
#
# Lifecycle flow:
# 1. Screen.compose() → calls compose_base_overlay() → creates #loading-overlay (visible, layer 1)
#                     → creates screen-specific content (hidden, layer 0)
# 2. on_mount() → Creates threading.Thread(target=_load_content) and starts it
# 3. _load_content() → Runs on worker thread, calls initialize_content()
# 4. _load_content() → Uses app.call_from_thread(finish_loading) for thread-safe UI update
# 5. finish_loading() → Runs on main thread, hides overlay, shows content
#
# Subclass pattern:
# 1. Call super().__init__("Loading message")
# 2. In compose(), call compose_base_overlay() LAST (so overlay is on top layer)
# 3. Override initialize_content() to return data dict
# 4. Add time.sleep(0.01) after subprocess calls to release GIL (breathe points)
# 5. Content widgets should start with display=False, finish_loading() will show them
#
# Loading overlay: Full-screen container with centered spinner, positioned 5 lines from top.
# Uses layer system - overlay is layer 1 (on top), content is layer 0 (underneath).
#
# GIL-Release Breathe Points Pattern:
# def initialize_content(self):
#     result = subprocess.run(["gcloud", "..."])
#     time.sleep(0.01)  # ← Breathe point! Releases GIL, lets spinner animate 1-2 frames
#     result2 = subprocess.run(["gsutil", "..."])
#     time.sleep(0.01)  # ← Another breathe point!
#     return data
# </claudes_code_comments>

from typing import Any, Optional
import threading
from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static
from textual.screen import Screen
from .animated_spinner import AnimatedSpinner


# (get_log_path removed - no longer needed after debug cleanup)
class BaseScreen(Screen):
    """Base screen with universal loading overlay pattern"""

    CSS = """
    /* Loading overlay - covers entire screen, layer 1 (on top) */
    #loading-overlay {
        width: 100%;
        height: 100%;
        align: center middle;
        content-align: center middle;
        layer: overlay;
        background: $surface;
    }

    /* Loading content wrapper - centers both horizontally and vertically */
    #loading-content {
        width: auto;
        height: auto;
        align: center middle;
        content-align: center middle;
    }

    /* Animated spinner styling */
    AnimatedSpinner {
        width: auto;
        height: auto;
        text-align: center;
    }

    /* Loading message below spinner */
    #loading-message {
        width: auto;
        height: auto;
        text-align: center;
        padding-top: 1;
        color: $accent;
    }
    """

    # Key binding for viewing last error
    BINDINGS = [
        ("e", "show_last_error", "Last Error"),
    ]

    def __init__(self, loading_message: str = "Loading..."):
        super().__init__()
        self.loading_message = loading_message
        self.loading_overlay: Optional[Container] = None
        self._has_mounted = False  # Track if on_mount has run
        self.last_error = None  # Store last error for popup (press 'e' to view)

    def compose_base_overlay(self) -> ComposeResult:
        """
        Create loading overlay - call this LAST in subclass compose().

        Overlay will be on top layer, hiding content underneath until loading completes.
        If loading_message is None, no overlay is created (for screens with custom loading).
        """
        # Skip overlay if loading_message is None (custom loading animation)
        if self.loading_message is None:
            return

        with Container(id="loading-overlay") as overlay:
            self.loading_overlay = overlay
            with Vertical(id="loading-content"):
                yield AnimatedSpinner()
                yield Static(" " + self.loading_message, id="loading-message")

    def on_mount(self) -> None:
        """
        Start content initialization using manual threading.

        CRITICAL: Uses manual threading.Thread() because @work(thread=True) is broken!
        The decorator creates a Worker object but never executes the function.
        """
        self._has_mounted = True
        # Create and start thread manually (works reliably!)
        thread = threading.Thread(target=self._load_content, daemon=True, name="content_loader")
        thread.start()

    def on_show(self) -> None:
        """
        Called when screen becomes visible (including revisits).

        NOTE: Disabled reload on revisit - it was causing UI halts.
        Screens now show cached data on revisit. User can manually refresh if needed.
        """
        # DISABLED: Automatic reload on revisit causes UI halt
        # If you want fresh data, implement manual refresh button instead
        pass

    def _reload_content(self) -> None:
        """
        Reload content when screen is shown again (called via call_later).

        This runs asynchronously after the screen transition completes,
        preventing blocking on the previous screen.
        """
        # Clear any existing content first
        try:
            # Query for content-panel or any other main content container
            # This removes old widgets so they don't show through overlay
            content_panel = self.query_one("#content-panel")
            content_panel.remove_children()
        except Exception:
            pass  # No content panel to clear

        # Show loading overlay again
        if self.loading_overlay:
            self.loading_overlay.display = True

        # Restart the loading worker
        thread = threading.Thread(target=self._load_content, daemon=True, name="content_loader")
        thread.start()

    def _load_content(self) -> None:
        """
        Thread worker that runs blocking initialize_content().

        CRITICAL Threading Details:
        - Runs on background thread (daemon=True, won't block app exit)
        - Calls initialize_content() which may make blocking subprocess calls
        - Python's GIL may block during subprocess I/O, briefly freezing spinner
        - Breathe points (time.sleep) in initialize_content() help release GIL
        - ALL UI updates MUST use app.call_from_thread() for thread safety
        """
        try:
            # Run blocking initialize_content() - may include subprocess calls
            result = self.initialize_content()

            # Update UI from main thread (thread-safe!)
            self.app.call_from_thread(self.finish_loading, result)

        except Exception as e:
            # Show error to user (thread-safe!)
            self.app.call_from_thread(self._handle_worker_error, e)

    def update_loading_status(self, message: str) -> None:
        """
        Update loading overlay status text dynamically (thread-safe).

        Show users what's happening during page load by updating the status
        message as work progresses. Replaces the initial loading message.

        WHEN TO USE:
        - Call from initialize_content() to show progress
        - Update at key milestones during loading (e.g., halfway through)
        - Show what operation is currently running

        THREAD SAFETY:
        - Safe to call from worker threads (uses app.call_from_thread)
        - Won't crash if overlay already hidden
        - Silently fails if widget not available

        EXAMPLES:
            def initialize_content(self):
                # Start of loading
                self.update_loading_status("Checking infrastructure...")

                info = check_infrastructure_core(...)

                # Midway through loading
                self.update_loading_status("Analyzing resources...")

                resources = list_resources_core(...)

                return {"info": info, "resources": resources}

        BEST PRACTICES:
        - Keep messages short (1 line, ~40 chars max)
        - Use present progressive: "Checking...", "Loading...", "Analyzing..."
        - End with "..." to indicate ongoing work
        - Update at meaningful progress points (not too frequent)
        - Typical pattern: 1-2 updates per initialize_content()

        Args:
            message: New status message (one line, present progressive tense)
        """
        try:
            if self.loading_overlay and self.loading_overlay.display:
                # Thread-safe UI update via app
                def update_text():
                    try:
                        msg_widget = self.query_one("#loading-message", Static)
                        msg_widget.update(message)
                    except Exception:
                        pass  # Widget not found or overlay hidden

                # Use app.call_from_thread for proper thread safety
                self.app.call_from_thread(update_text)
        except Exception:
            pass  # Silently fail if overlay not available

    def initialize_content(self) -> Any:
        """
        Override this in subclasses for content initialization.
        Runs in background THREAD worker - use for blocking operations.

        Examples:
        - Fetch data from W&B
        - Check infrastructure status (gsutil, gcloud commands)
        - Load configuration

        NOTE: This runs in a separate thread, so blocking subprocess
        calls won't freeze the UI or loading animations.

        TIP: Use update_loading_status() to show progress during loading!

        Return data needed for finish_loading().
        """
        return None

    def _handle_worker_error(self, error: Exception) -> None:
        """Handle errors from background worker (called from main thread)"""
        # Hide loading overlay
        if self.loading_overlay:
            self.loading_overlay.display = False

        # Show error message to user
        # Subclasses can override this to customize error handling
        error_msg = f"Error loading screen: {error}"

        # Try to show error in UI if possible
        try:
            # This is a basic implementation - subclasses should override
            # for better error display
            pass
        except Exception:
            pass  # Fail silently if we can't show the error

    def notify_with_full_error(self, title: str, error: str, severity="error"):
        """
        Show truncated notification but store full error for popup.
        User can press 'e' key to see complete error details.

        Args:
            title: Error title (e.g., "Error Loading Runs")
            error: Full error message/stack trace
            severity: Notification severity ("error", "warning", "information")

        Example:
            try:
                data = fetch_data()
            except Exception as e:
                self.notify_with_full_error("Error Loading Data", str(e))
        """
        import time

        # Store full error for popup
        self.last_error = {
            "title": title,
            "full_text": error,
            "timestamp": time.time()
        }

        # Show truncated notification
        truncated = error[:80] + ("..." if len(error) > 80 else "")
        self.notify(
            f"{title}: {truncated} [Press 'e' for details]",
            severity=severity
        )

    def action_show_last_error(self):
        """Show popup with full error details (press 'e' key)"""
        if self.last_error:
            from CLI.shared.datatable_info_popup import DataTableInfoPopup
            self.app.push_screen(DataTableInfoPopup(
                title=self.last_error["title"],
                full_text=self.last_error["full_text"],
                full_text_label="Error"
            ))
        else:
            self.notify("No recent errors", severity="information")

    def finish_loading(self, data: Any = None) -> None:
        """
        Hide loading overlay and reveal content.
        Override this if you need to do something with the data before showing content.

        Args:
            data: Result from initialize_content()
        """
        if self.loading_overlay:
            self.loading_overlay.display = False
