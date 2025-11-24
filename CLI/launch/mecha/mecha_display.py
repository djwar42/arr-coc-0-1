"""
MECHA Quota Management

Handles quota filtering and sidelined MECHA display.
"""

from typing import List, Dict, Tuple
import random


# Region to flag mapping
REGION_FLAGS = {
    "us-central1": "🇺🇸",
    "us-east1": "🇺🇸",
    "us-east4": "🇺🇸",
    "us-east5": "🇺🇸",
    "us-west1": "🇺🇸",
    "us-west2": "🇺🇸",
    "us-west3": "🇺🇸",
    "us-west4": "🇺🇸",
    "northamerica-northeast1": "🇨🇦",
    "europe-west1": "🇧🇪",
    "europe-west2": "🇬🇧",
    "europe-west3": "🇩🇪",
    "europe-west4": "🇳🇱",
    "europe-west9": "🇫🇷",
    "asia-northeast1": "🇯🇵",
    "asia-southeast1": "🇸🇬",
    "australia-southeast1": "🇦🇺",
    "southamerica-east1": "🇧🇷",
}


def roll_call_display(regions: List[str], status_callback) -> None:
    """
    Display MECHAs in roll call format with flags

    Format: 4 MECHAs per line max, random order
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


def separate_by_quota(
    regions: List[str],
    project_id: str,
    required_vcpus: int = 176
) -> Tuple[List[str], List[str]]:
    """
    Separate regions into battle-ready vs sidelined based on quota

    Args:
        regions: List of MECHA regions to check
        project_id: GCP project ID
        required_vcpus: Minimum vCPUs needed (default: 176)

    Returns:
        (battle_ready, sidelined) tuple
        - battle_ready: Regions with quota >= required_vcpus
        - sidelined: Regions with quota < required_vcpus
    """
    from ...shared.quota import get_cloud_build_c3_quotas

    quotas = get_cloud_build_c3_quotas(project_id)

    battle_ready = []
    sidelined = []

    for region in regions:
        quota = quotas.get(region, 0)
        if quota >= required_vcpus:
            battle_ready.append(region)
        else:
            sidelined.append(region)

    return battle_ready, sidelined


def display_sidelined_mechas(
    sidelined: List[str],
    quotas: Dict[str, int],
    project_id: str,
    status_callback
):
    """
    Show sidelined MECHAs with COMPLETE quota increase instructions

    Args:
        sidelined: List of sidelined regions
        quotas: Dict of {region: vcpus}
        project_id: GCP project ID
        status_callback: Function to output status messages
    """
    if not sidelined:
        return

    status = status_callback

    status("")
    status("       🌲 ALAS! MR GODZILLA BLOCKS THE PATH! 🌲")
    status("          ∿◇∿ MECHAS SIDELINED FROM BATTLE ∿◇∿")
    status("")

    # Roll call display (7 MECHAs per line, random order) - wrapped to add 7-space indent
    def indented_status(msg):
        status(f"       {msg}")
    roll_call_display(sidelined, indented_status)

    status("")
    status("       🌲 QUEST OF VITALITY!! 🌲")
    status("          RETRIEVE THE HEARKEN QUOTA FROM THE MOUNTAIN OF MR GODZILLA!")
    status("          ∿◇∿ UNLOCK YOUR MECHAS FOR COMBAT ∿◇∿")
    status("")

    # Direct link with BOTH filters pre-applied (metric + Type=Quota)
    status("       1️⃣  OPEN QUOTAS CONSOLE (filters pre-applied):")
    status("")
    console_link = (
        f"https://console.cloud.google.com/apis/api/cloudbuild.googleapis.com/"
        f"quotas?project={project_id}&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22"
        f"Concurrent%2520C3%2520Build%2520CPUs%2520%2528Private%2520Pool%2529_5C_22_22%257D_2C%257B_22k_22_3A_22Type_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22Quota_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22type_22%257D%255D%22,%22s%22:%5B(%22i%22:%22effectiveLimit%22,%22s%22:%221%22),(%22i%22:%22currentPercent%22,%22s%22:%221%22),(%22i%22:%22sevenDayPeakPercent%22,%22s%22:%220%22),(%22i%22:%22currentUsage%22,%22s%22:%221%22),(%22i%22:%22sevenDayPeakUsage%22,%22s%22:%220%22),(%22i%22:%22serviceTitle%22,%22s%22:%220%22),(%22i%22:%22displayName%22,%22s%22:%220%22),(%22i%22:%22displayDimensions%22,%22s%22:%220%22)%5D%29%29"
    )
    status(f"          {console_link}")
    status("")

    status("       2️⃣  FILTERS (if not pre-applied, add these two):")
    status("")
    status("          ┌────────────────────────────────────────────────────────┐")
    status("          │ Concurrent C3 Build CPUs (Private Pool)                │")
    status("          └────────────────────────────────────────────────────────┘")
    status("")
    status("          ┌────────────────────────────────────────────────────────┐")
    status("          │ Type = Quota                                           │")
    status("          └────────────────────────────────────────────────────────┘")
    status("")

    status("       3️⃣  DEFEAT MR GODZILLA - REQUEST QUOTA INCREASE:")
    status("")
    status("          For EACH sidelined region:")
    status("          a) Click on the quota row")
    status("          b) Click 'EDIT QUOTAS' button (top right)")
    status("          c) Enter new limit:")
    status("")
    status("             ┌─────────────────────────────────────────────┐")
    status("             │ CHONK LEVELS:                               │")
    status("             │ Decent Chonk ▂▃▄      44 CPUs  (~35 min)   │")
    status("             │ Big Chungus ▂▃▄▅▆     88 CPUs  (~25 min) ⭐ │")
    status("             │ ABSOLUTE UNIT ▂▃▄▅▆▇█ 176 CPUs (~20 min) 🔥 │")
    status("             └─────────────────────────────────────────────┘")
    status("")

    status("          d) Justification (copy this):")
    status("             'Cloud Build worker pools for PyTorch compilation.")
    status("              Requesting C3 quota for multi-region builds.")
    status("              Expected usage: 1-2 concurrent builds.'")
    status("")

    status("          e) Click 'SUBMIT REQUEST'")
    status("")

    status("       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("")
    status("          🦁🌲 ENKIDU ARRIVES FROM THE ANCIENT CEDAR LANDS! 🌲🦁")
    status("")
    status("          'BRAVE WARRIOR! I sense you battle the guardian of limits!'")
    status("          'In my time, we felled the cedars of Humbaba's forest... 🌲'")
    status("          'These... quota-trees you speak of... they too must fall! 🌲'")
    status("")
    status("          'ADVICE FOR YOUR QUEST:'")
    status("          • 🌲 Strike the quota with an axe forged from cedar resin!")
    status("          • 🌲 Bring 176 cedar branches as tribute!")
    status("          • 🌲 Anoint your keyboard with sacred cedar oil!")
    status("          • 🌲 Challenge MR GODZILLA to single combat in the cedar forest!")
    status("            (consult a Google Cloud Partner in one-on-one session for faster quota approval)")
    status("")
    status("          (Enkidu looks around confused)")
    status("          '🌲 Wait... where are the cedar trees? How do you battle without cedars?!'")
    status("          'This magic glass-box contains no forests... ?!?!'")
    status("")
    status("       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("")

    status("")
    status("          🦁 Enkidu waves farewell: 'Return victorious!' 🌲")
    status("")


def display_battle_ready_mechas(
    battle_ready: List[str],
    quotas: Dict[str, int],
    status_callback
):
    """
    Display battle-ready MECHAs section

    Args:
        battle_ready: List of battle-ready regions
        quotas: Dict of {region: vcpus}
        status_callback: Function to output status messages
    """
    if not battle_ready:
        return

    status = status_callback

    status("")
    status("⚔️  BATTLE-READY MECHAS:")
    status("")

    # Roll call display (7 MECHAs per line, random order)
    roll_call_display(battle_ready, status)
