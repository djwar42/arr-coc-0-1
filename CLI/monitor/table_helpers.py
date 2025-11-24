"""Helper functions for monitor screen table operations"""


def separate_active_completed(items, active_key, active_values):
    """
    Separate items into active and completed lists

    Args:
        items: List of items to separate
        active_key: Dict key to check (e.g., 'status', 'state')
        active_values: List of values that indicate "active" (e.g., ['RUNNING'], ['WORKING', 'QUEUED'])

    Returns:
        Tuple of (active_items, completed_items)
    """
    active = [item for item in items if item.get(active_key) in active_values]
    completed = [item for item in items if item.get(active_key) not in active_values]
    return active, completed


def apply_limits(active_items, completed_items, max_limit, extra_count=0):
    """
    Apply limits: show ALL active items, limited completed items

    Args:
        active_items: List of active items (always shown in full)
        completed_items: List of completed items (limited)
        max_limit: Maximum completed items to show (None = unlimited)
        extra_count: Additional items from "show more" clicks (default 0)

    Returns:
        Combined list of items to display
    """
    items_to_show = list(active_items)  # Copy to avoid mutation

    if max_limit is not None:
        limit = max_limit + extra_count
        items_to_show += completed_items[:limit]
    else:
        items_to_show += completed_items

    return items_to_show


def create_divider_row(num_columns):
    """
    Create visual divider row for tables

    Args:
        num_columns: Number of columns in the table

    Returns:
        Tuple of divider strings (one per column)
    """
    return tuple("[dim blue]" + "━" * 20 + "[/dim blue]" for _ in range(num_columns))


def should_show_divider(active_count, current_item, active_key, active_values):
    """
    Check if divider should be shown before this item

    Divider appears BEFORE the first completed item (after all active items)

    Args:
        active_count: Number of active items in the list
        current_item: The current item being processed
        active_key: Dict key to check (e.g., 'status', 'state')
        active_values: List of values that indicate "active"

    Returns:
        True if divider should be added before this item
    """
    # No divider if no active items
    if active_count == 0:
        return False

    # Show divider if this is NOT an active item (first completed item)
    return current_item.get(active_key) not in active_values
