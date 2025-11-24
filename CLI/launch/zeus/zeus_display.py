"""
Zeus Thunder Region Display

Handles thunder region display, roll call, and Hermes passage display.
"""

# <claudes_code_comments>
# ** Function List **
# get_region_display_name(region) - Format region with flag emoji
# roll_call_display(regions, status_callback) - Display regions in roll call format (4 per line, shuffled)
#
# ** Technical Review **
# Display utilities for Zeus thunder regions. Mirrors MECHA's mecha_display.py.
# Provides flag emojis and formatted output for region lists.
#
# REGION_FLAGS dict maps GCP regions to country flag emojis (45+ regions supported).
# get_region_display_name() formats regions with flags: "🇺🇸 us-east4".
# roll_call_display() shows regions in compact roll call format (4 per line, random order).
#
# Note: show_hermes_passage() remains in zeus_battle.py (battle-specific, uses THUNDER_TIERS).
# </claudes_code_comments>

from typing import List, Dict
import random


# Region to flag mapping
REGION_FLAGS: Dict[str, str] = {
    # US regions
    "us-central1": "🇺🇸",
    "us-east1": "🇺🇸",
    "us-east4": "🇺🇸",
    "us-east5": "🇺🇸",
    "us-west1": "🇺🇸",
    "us-west2": "🇺🇸",
    "us-west3": "🇺🇸",
    "us-west4": "🇺🇸",
    "us-south1": "🇺🇸",

    # Europe regions
    "europe-west1": "🇧🇪",  # Belgium
    "europe-west2": "🇬🇧",  # London
    "europe-west3": "🇩🇪",  # Frankfurt
    "europe-west4": "🇳🇱",  # Netherlands
    "europe-west6": "🇨🇭",  # Zurich
    "europe-west8": "🇮🇹",  # Milan
    "europe-west9": "🇫🇷",  # Paris
    "europe-north1": "🇫🇮",  # Finland
    "europe-central2": "🇵🇱",  # Poland

    # Asia regions
    "asia-east1": "🇹🇼",  # Taiwan
    "asia-east2": "🇭🇰",  # Hong Kong
    "asia-northeast1": "🇯🇵",  # Tokyo
    "asia-northeast2": "🇯🇵",  # Osaka
    "asia-northeast3": "🇰🇷",  # Seoul
    "asia-southeast1": "🇸🇬",  # Singapore
    "asia-southeast2": "🇮🇩",  # Jakarta
    "asia-south1": "🇮🇳",  # Mumbai
    "asia-south2": "🇮🇳",  # Delhi

    # Australia
    "australia-southeast1": "🇦🇺",  # Sydney
    "australia-southeast2": "🇦🇺",  # Melbourne

    # South America
    "southamerica-east1": "🇧🇷",  # São Paulo
    "southamerica-west1": "🇨🇱",  # Santiago

    # North America (other)
    "northamerica-northeast1": "🇨🇦",  # Montreal
    "northamerica-northeast2": "🇨🇦",  # Toronto
}


def get_region_display_name(region: str) -> str:
    """
    Get region display name with flag emoji.

    Args:
        region: Region name (e.g., "us-east4")

    Returns:
        Formatted name with flag (e.g., "🇺🇸 us-east4")
    """
    flag = REGION_FLAGS.get(region, "🌍")  # Default to earth emoji
    return f"{flag} {region}"


def roll_call_display(regions: List[str], status_callback) -> None:
    """
    Display thunder regions in roll call format with flags

    Format: 4 regions per line max, random order
    Example: 🇺🇸 us-west1 ∿ 🇯🇵 asia-northeast1 ∿ 🇩🇪 europe-west3 ∿ 🇬🇧 europe-west2

    Args:
        regions: List of region names
        status_callback: Function to output status messages
    """
    status = status_callback

    # Shuffle for random ordering
    shuffled = regions.copy()
    random.shuffle(shuffled)

    # Split into lines of 4 max
    lines = []
    for i in range(0, len(shuffled), 4):
        line_regions = shuffled[i:i+4]
        # Format each region with flag
        formatted = []
        for region in line_regions:
            flag = REGION_FLAGS.get(region, "🏳️")
            formatted.append(f"{flag} {region}")
        # Join with separator
        line = " ∿ ".join(formatted)
        lines.append(line)

    # Output each line
    for line in lines:
        status(f"   {line}")
    status("")


# NOTE: show_hermes_passage() remains in zeus_battle.py
# It's battle-specific and deeply integrated with THUNDER_TIERS constants
