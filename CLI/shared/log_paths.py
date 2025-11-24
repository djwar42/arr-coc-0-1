"""
Centralized log path management - works from any working directory!

All log paths are resolved relative to the ARR_COC/Training/ directory,
regardless of where Python is executed from.
"""
from pathlib import Path


def get_training_dir() -> Path:
    """Get the ARR_COC/Training/ directory path (absolute)."""
    # This file is at: CLI/shared/log_paths.py
    # We need to navigate to project root, then to ARR_COC/Training/
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent.parent  # Go up: shared/ -> CLI/ -> project root
    return project_root / "ARR_COC" / "Training"


def get_log_path(filename: str) -> Path:
    """
    Get absolute path to a log file in ARR_COC/Training/logs/.
    
    Args:
        filename: Name of log file (e.g., "spinner_debug.log")
    
    Returns:
        Absolute path to log file
    
    Example:
        >>> log_path = get_log_path("spinner_debug.log")
        >>> # Always returns: /absolute/path/to/ARR_COC/Training/logs/spinner_debug.log
        >>> # Regardless of current working directory!
    """
    training_dir = get_training_dir()
    log_path = training_dir / "logs" / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path
