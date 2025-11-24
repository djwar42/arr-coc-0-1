"""
Shared Type Definitions for CLI
"""

from typing import Callable

# Status callback type - used throughout setup/launch/teardown
StatusCallback = Callable[[str], None]
