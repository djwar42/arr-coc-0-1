"""
Universal DataTable info popup modal for displaying full row details.

<claudes_code_comments>
** Function List **
- DataTableInfoPopup - Modal screen that displays full row information with close button

** Technical Review **
This module implements a reusable modal popup for displaying full row details
from any DataTable in the TUI. When a user clicks a table row, this modal appears
showing the complete, untruncated information (errors, notes, logs, etc.).

Key features:
- Automatically centers on screen
- Shows full text (no truncation)
- Optional dense summary at top (colored field labels)
- Customizable full text label (Error, Note, Details, CVE Report, etc.)
- Close button and click-outside-to-close behavior
- Works with any table (universal pattern)
- Scrollable content for long text
- Uses Textual's Screen and Container widgets

Flow:
1. User clicks table row → MonitorScreen.on_data_table_row_selected()
2. Extract info from stored row data
3. Push DataTableInfoPopup with title, full_text, optional dense_summary and label
4. User closes modal → dismiss, return to table

Usage examples:
- DataTableInfoPopup(title, text)  # Simple, label defaults to "Text"
- DataTableInfoPopup(title, text, full_text_label="Error")  # Custom label
- DataTableInfoPopup(title, text, dense_summary, "Details")  # With summary
</claudes_code_comments>
"""

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class DataTableInfoPopup(ModalScreen):
    """Modal popup to display full row information from DataTable"""

    CSS = """
    DataTableInfoPopup {
        align: center middle;
        background: transparent;
    }

    #info-popup-container {
        background: $surface;
        border: thick $primary;
        width: 80%;
        height: 80%;
        padding: 1 2;
        layout: vertical;
    }

    #info-popup-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        height: auto;
    }

    #info-popup-content {
        height: 1fr;
        overflow-y: auto;
        background: $panel;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #info-popup-close {
        width: 20;
        height: auto;
        dock: bottom;
        align: right middle;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, title: str, full_text: str, dense_summary: str = None, full_text_label: str = None):
        """
        Initialize info popup modal.

        Args:
            title: Title for the popup (e.g., "Error Details", "Note", "Full Log")
            full_text: Full, untruncated text to display
            dense_summary: Optional dense summary line shown at top (already formatted with colors)
            full_text_label: Optional label for full text section (e.g., "Error", "Note", "Details")
                           Defaults to "Text" if not provided
        """
        super().__init__()
        self.popup_title = title
        self.full_text = full_text or "[dim]No additional information[/dim]"
        self.dense_summary = dense_summary
        self.full_text_label = full_text_label or "Text"

    def compose(self) -> ComposeResult:
        """Compose info popup UI"""
        with Container(id="info-popup-container"):
            yield Static(self.popup_title, id="info-popup-title")

            # Build content: optional dense summary + labeled full text
            content_parts = []
            if self.dense_summary:
                content_parts.append(f"[bold]{self.dense_summary}[/bold]\n")

            content_parts.append(f"[yellow]{self.full_text_label}:[/yellow]\n{self.full_text}")

            # Wrap Static in VerticalScroll for proper scrolling of long content
            with VerticalScroll(id="info-popup-content"):
                yield Static("\n".join(content_parts))
            yield Button("Close [Esc/Q]", id="info-popup-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close popup when button clicked"""
        if event.button.id == "info-popup-close":
            self.dismiss()

    def action_dismiss(self) -> None:
        """Close popup (called by Esc key)"""
        self.dismiss()

    def on_click(self, event) -> None:
        """Close popup when clicking transparent background (outside container)"""
        # Only close if clicking directly on ModalScreen background (not children)
        if event.widget == self:
            self.dismiss()
