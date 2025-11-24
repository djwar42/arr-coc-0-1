# <claudes_code_comments>
# ** Function List **
# format_runner_note(raw) - Format runner Note column with color-coded states
# format_builds_note(raw) - Format builds Note column with success/error colors
# format_vertex_note(raw) - Format Vertex AI Note column with state colors
# format_active_note(raw) - Format active runs Note (placeholder for future)
# format_completed_note(raw) - Format completed runs Note (placeholder for future)
#
# ** Technical Review **
# Companion file to core.py - centralizes all Note column formatting.
# Each function takes raw error/status text and returns Rich-formatted string.
# Color scheme:
#   - cyan ⏳  = Active/running (not an error)
#   - green    = Success/completed
#   - yellow   = Warning (✗ prefix)
#   - red ❌   = Actual error
#   - dim —    = No data
# Truncation: Runner 75 chars, Builds 60 chars, Vertex 60 chars
# </claudes_code_comments>

"""
Note Column Formatters - Companion to core.py

Centralizes all Note/error column formatting for monitor tables.
Each function returns Rich-formatted strings ready for display.
"""


def format_runner_note(raw: str) -> str:
    """
    Format runner Note column for Rich display.

    Args:
        raw: Raw error/status string from execution data

    Returns:
        Rich-formatted string with appropriate color

    Color scheme:
        - cyan ⏳  = Running (active, not an error)
        - green    = Success (✓ prefix or Completed)
        - yellow   = Warning (✗ prefix)
        - red ❌   = Error (anything else)
        - dim —    = No data
    """
    if not raw or raw == "—" or raw == "[dim]—[/dim]":
        return "[dim]—[/dim]"

    # 🎨 NOTE COLUMN FORMATTING (organized by priority):
    if raw == "Running...":
        # Active execution - cyan (not an error!)
        return f"[cyan]⏳ {raw}[/cyan]"
    elif raw.startswith('✓'):
        # Success - green (all success msgs start with ✓)
        return f"[green]{raw[:75]}[/green]"
    elif raw.startswith('✗'):
        # Warning - yellow
        return f"[yellow]{raw[:75]}[/yellow]"
    else:
        # Actual error - red
        return f"[red]❌ {raw[:75]}[/red]"


def format_builds_note(raw: str) -> str:
    """
    Format builds Note column for Rich display.

    Args:
        raw: Raw error/status string from build data

    Returns:
        Rich-formatted string with appropriate color

    Color scheme:
        - green = Build completed successfully
        - red   = Build error/failure
        - dim — = No data
    """
    if not raw or raw == "—":
        return "[dim]—[/dim]"

    # 🎨 NOTE COLUMN FORMATTING for builds:
    if "Build completed" in raw:
        # Success message - green
        return f"[green]{raw[:60]}[/green]"
    else:
        # Actual error - red
        return f"[red]{raw[:60]}[/red]"


def format_vertex_note(raw: str) -> str:
    """
    Format Vertex AI Note column for Rich display.

    Args:
        raw: Raw error/status string from Vertex job data
             Note: Only FAILED jobs have error_msg set, others are None

    Returns:
        Rich-formatted string with appropriate color

    Color scheme:
        - red   = Error/failure (only case with actual text)
        - dim — = No data (RUNNING/PENDING/SUCCEEDED jobs)
    """
    if not raw or raw == "—":
        return "[dim]—[/dim]"

    # 🎨 NOTE COLUMN FORMATTING for Vertex AI:
    # Only FAILED jobs have error text - all others return dim dash above
    # Just show the error in red (it's always an error message)
    return f"[red]{raw[:60]}[/red]"


def format_active_note(raw: str) -> str:
    """
    Format active runs Note column for Rich display.

    Placeholder for future - active runs don't currently have Note column.

    Args:
        raw: Raw status string

    Returns:
        Rich-formatted string
    """
    if not raw or raw == "—":
        return "[dim]—[/dim]"
    return f"[cyan]{raw[:60]}[/cyan]"


def format_completed_note(raw: str) -> str:
    """
    Format completed runs Note column for Rich display.

    Placeholder for future - completed runs don't currently have Note column.

    Args:
        raw: Raw status/exit string

    Returns:
        Rich-formatted string
    """
    if not raw or raw == "—":
        return "[dim]—[/dim]"

    if "crashed" in raw.lower() or "failed" in raw.lower():
        return f"[red]{raw[:60]}[/red]"
    else:
        return f"[green]{raw[:60]}[/green]"
