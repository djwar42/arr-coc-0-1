"""
🤖 MECHA Fleet Blast - Simultaneous Multi-Region Deployment (メカ)

Launches worker pool creation in ALL 15 regions simultaneously.
Waits FULL 10 minutes, checking every minute for arrivals.
Epic battle-themed announcements when MECHAs arrive early!
Those that timeout are killed and discarded.

This allows rapid MECHA fleet acquisition instead of passive one-at-a-time collection!
"""

import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .mecha_regions import C3_REGIONS

# MECHA Fleet Blast Configuration
BEACON_WAIT_MINUTES = 5  # How long to wait for MECHAs to respond to beacon

# Center-based ASCII chars for arrival animation (25 total)
ARRIVAL_SYMBOLS = [
    "-",
    "~",
    "=",
    "+",
    "∿",
    "≈",
    "∼",
    "∽",
    "∾",
    "≋",
    "◦",
    "∘",
    "·",
    "•",
    "○",
    "◌",
    "◯",
    "∙",
    "⋅",
    "∶",
    "|",
    "¦",
    "⎢",
    "⎥",
    "┃",
]

# Cool ASCII chars for arrival flair (50 total) - geometric, lighting, energy
ARRIVAL_FLAIR = [
    "⚡",
    "✦",
    "✧",
    "✨",
    "★",
    "☆",
    "✴",
    "✵",
    "✶",
    "✷",
    "✸",
    "✹",
    "✺",
    "✻",
    "✼",
    "✽",
    "✾",
    "✿",
    "❀",
    "❁",
    "◆",
    "◇",
    "◈",
    "◉",
    "◊",
    "○",
    "◌",
    "◍",
    "◎",
    "●",
    "◐",
    "◑",
    "◒",
    "◓",
    "◔",
    "◕",
    "◖",
    "◗",
    "◘",
    "◙",
    "◚",
    "◛",
    "◜",
    "◝",
    "◞",
    "◟",
    "◠",
    "◡",
    "◢",
    "◣",
]

# Winding/dingbat-like symbols (25 total) - cryptic beacon hash characters
WINDING_SYMBOLS = [
    "▀",
    "▁",
    "▂",
    "▃",
    "▄",
    "▅",
    "▆",
    "▇",
    "█",
    "▌",
    "▐",
    "░",
    "▒",
    "▓",
    "├",
    "┤",
    "┬",
    "┴",
    "┼",
    "═",
    "║",
    "╔",
    "╚",
    "╬",
    "╳",
]

# Combined beacon hash pool (75 chars total)
BEACON_HASH_POOL = ARRIVAL_FLAIR + WINDING_SYMBOLS


def lazy_load_quota_entry(project_id: str, region: str, output_callback):
    """
    Submit test build to trigger quota entry creation

    This creates a quota entry with 4 vCPUs (default).
    User can then request increase to 176 vCPUs via Console.

    Takes ~5 seconds.

    Args:
        project_id: GCP project ID
        region: GCP region
        output_callback: Function to output status messages
    """
    import tempfile

    # output_callback(f"   [dim]⏳ Lazy loading quota entry for {region}...[/dim]")

    # Create minimal test build
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal Dockerfile
        with open(f"{tmpdir}/Dockerfile", "w") as f:
            f.write("FROM alpine:latest\nRUN echo 'quota test'\n")

        # Minimal cloudbuild.yaml
        with open(f"{tmpdir}/cloudbuild.yaml", "w") as f:
            f.write(
                "steps:\n- name: 'gcr.io/cloud-builders/docker'\n  args: ['build', '.']\n"
            )

        # Submit build (will fail with quota error, but that's expected!)
        # output_callback(f"   [dim]→ Submitting test build to GCP...[/dim]")

        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "builds",
                    "submit",
                    tmpdir,
                    f"--config={tmpdir}/cloudbuild.yaml",
                    f"--region={region}",
                    f"--worker-pool=projects/{project_id}/locations/{region}/workerPools/pytorch-mecha-pool",
                    "--timeout=1m",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )  # 20 second Python timeout

            # output_callback(f"   [dim]→ Build submitted, checking result...[/dim]")

            # We don't care if it fails - quota entry is created either way!
            if result.returncode != 0:
                # output_callback(f"   [dim]✓ Quota entry initialized (4 vCPUs default - build failed as expected)[/dim]")
                # Only show unexpected errors (not normal Cloud Build output)
                if result.stderr:
                    # Filter out normal Cloud Build messages
                    is_normal_output = any(
                        msg in result.stderr
                        for msg in [
                            "Creating temporary archive",
                            "Uploading tarball",
                            "RESOURCE_EXHAUSTED",
                            "Quota exceeded",
                        ]
                    )
                    if not is_normal_output:
                        # Unexpected error - show it
                        output_callback(
                            f"   [dim]   Unexpected error: {result.stderr[:200]}[/dim]"
                        )
            else:
                pass
                # output_callback(f"   [dim]✓ Quota entry initialized (4 vCPUs default - build succeeded)[/dim]")

        except subprocess.TimeoutExpired:
            pass
            # output_callback(f"   [dim]⚠️  Build submission timed out (20s) - quota entry may still be created[/dim]")
        except Exception as e:
            pass
            # output_callback(f"   [dim]⚠️  Build submission error: {str(e)}[/dim]")


def animate_arrival():
    """
    Show 3-char arrival animation with 200ms between each char
    Using random center-based ASCII symbols
    Prepends to the arrival line (no newline after)
    """
    symbols = random.sample(ARRIVAL_SYMBOLS, 3)

    # Two spaces indent before animation
    sys.stdout.write("  ")
    sys.stdout.flush()

    # First char
    sys.stdout.write(symbols[0])
    sys.stdout.flush()
    time.sleep(0.2)

    # Second char
    sys.stdout.write(symbols[1])
    sys.stdout.flush()
    time.sleep(0.2)

    # Third char
    sys.stdout.write(symbols[2])
    sys.stdout.flush()
    time.sleep(0.2)

    # Space before arrival message (stays on same line)
    sys.stdout.write(" ")
    sys.stdout.flush()


def output_arrival(preline: str, message: str, status_callback=None):
    """
    Output MECHA arrival message with timing/animation in CLI, instant dump in TUI

    CLI mode: animate_arrival() + print + 0.5-0.9s stagger delay
    TUI mode: preline + message dumped immediately (full line, no animation)

    Args:
        preline: Pre-line symbols to prepend (only used in TUI)
        message: Full arrival message to display
        status_callback: TUI callback (if provided, we're in TUI mode)
    """
    if status_callback:
        # TUI: Dump full line immediately with leading indent (preline + message)
        status_callback("  " + preline + message)
    else:
        # CLI: Full cinematic experience with animation and timing
        animate_arrival()
        print(message)
        time.sleep(random.uniform(0.5, 0.9))


def announce_mecha_arrival(
    region: str, location: str, minute: int, status_callback=None
):
    """
    🎊 Epic MECHA arrival announcement with LOCAL PERSONALITY!

    Each region has 10 unique MECHA greetings reflecting local culture.
    """

    # Region-specific MECHA greetings (10 each)
    locale_greetings = {
        # Australia - Harbour city circuits, surf-cooled servos
        "australia-southeast1": [
            f"🤖 {region.upper()} MECHA: Harbour city circuits ONLINE!",
            f"⚡ {region.upper()} MECHA: Eucalyptus-cooled vents activates!",
            f"🔥 {region.upper()} MECHA: Surf-calibrated gyros READY!",
            f"💪 {region.upper()} MECHA: Reef-crystal optics engages!",
            f"💥 {region.upper()} MECHA: Southern Cross navigation locks!",
            f"🎯 {region.upper()} MECHA: Pacific gateway sensors active!",
            f"✨ {region.upper()} MECHA: Sunshine-tempered alloy deploys!",
            f"🚀 {region.upper()} MECHA: Blue Mountains altitude systems SETS!",
            f"🏆 {region.upper()} MECHA: Coastal breeze cooling GO!",
            f"⚙️ {region.upper()} MECHA: Opera House acoustics tunes!",
        ],
        # Tokyo - Cherry blossom circuits, precision servos
        "asia-northeast1": [
            f"🤖 {region.upper()} MECHA: Cherry blossom sensor arrays ONLINE!",
            f"⚡ {region.upper()} MECHA: Shibuya crossing navigation LOADS!",
            f"🔥 {region.upper()} MECHA: Zen garden harmony circuits ENGAGES!",
            f"💫 {region.upper()} MECHA: Tokyo Bay starlight sensors LOCKS!",
            f"💥 {region.upper()} MECHA: Bullet train precision ACTIVE!",
            f"🎯 {region.upper()} MECHA: Rail network synchronization ACHIEVES!",
            f"✨ {region.upper()} MECHA: Neon-tracking optics CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Mount Fuji snow-peak sensors READY!",
            f"🏆 {region.upper()} MECHA: Sakura petal gyroscopes CONFIRMS!",
            f"⚙️ {region.upper()} MECHA: Tea ceremony precision protocols ENABLES!",
        ],
        # Singapore - Garden city circuits, tropical cooling
        "asia-southeast1": [
            f"🤖 {region.upper()} MECHA: Marina Bay sensor grid ONLINE!",
            f"⚡ {region.upper()} MECHA: Orchid-cooled processors ACTIVE!",
            f"🔥 {region.upper()} MECHA: Equatorial solar arrays CHARGES!",
            f"💫 {region.upper()} MECHA: Lion City roar sensors READY!",
            f"💥 {region.upper()} MECHA: Merlion water-jet cooling ENGAGES!",
            f"🎯 {region.upper()} MECHA: Strait navigation systems LOCKS!",
            f"✨ {region.upper()} MECHA: Tropical monsoon shields DEPLOYS!",
            f"🚀 {region.upper()} MECHA: Gardens by the Bay bio-circuits SETS!",
            f"🏆 {region.upper()} MECHA: Hawker center network SYNCHRONIZES!",
            f"⚙️ {region.upper()} MECHA: Island paradise sensors CALIBRATES!",
        ],
        # South Carolina - Atlantic coast precision
        "us-east1": [
            f"🤖 {region.upper()} MECHA: Atlantic seaboard sensors ONLINE!",
            f"⚡ {region.upper()} MECHA: Palmetto tree root networks ACTIVE!",
            f"🔥 {region.upper()} MECHA: Coastal marshland cooling ENGAGES!",
            f"💫 {region.upper()} MECHA: Blue Ridge mountain relays LOCKS!",
            f"💥 {region.upper()} MECHA: Charleston harbor protocols READY!",
            f"🎯 {region.upper()} MECHA: Magnolia blossom filters DEPLOYS!",
            f"✨ {region.upper()} MECHA: Southern charm circuits CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Ocean breeze ventilation SETS!",
            f"🏆 {region.upper()} MECHA: Sweet tea hydraulics FLOWING!",
            f"⚙️ {region.upper()} MECHA: Palmetto state spirit ENABLES!",
        ],
        # Northern Virginia - Capital region precision
        "us-east4": [
            f"🤖 {region.upper()} MECHA: Potomac River cooling systems ONLINE!",
            f"⚡ {region.upper()} MECHA: Cherry blossom sensor networks ACTIVE!",
            f"🔥 {region.upper()} MECHA: Top-tier data encryption ENGAGES!",
            f"💫 {region.upper()} MECHA: Blue Ridge parkway relays LOCKS!",
            f"💥 {region.upper()} MECHA: Colonial architecture circuits READY!",
            f"🎯 {region.upper()} MECHA: Shenandoah valley sensors DEPLOYS!",
            f"✨ {region.upper()} MECHA: Historic monument scanning CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Data center backbone CONNECTS!",
            f"🏆 {region.upper()} MECHA: Virginia is for lovers mode ACTIVE!",
            f"⚙️ {region.upper()} MECHA: Capital beltway navigation SETS!",
        ],
        # Columbus Ohio - Heartland precision
        "us-east5": [
            f"🤖 {region.upper()} MECHA: Scioto River networks ONLINE!",
            f"⚡ {region.upper()} MECHA: Buckeye tree root systems ACTIVE!",
            f"🔥 {region.upper()} MECHA: Great Lakes cooling circuits ENGAGES!",
            f"💫 {region.upper()} MECHA: Midwest plains relays LOCKS!",
            f"💥 {region.upper()} MECHA: Ohio State research protocols READY!",
            f"🎯 {region.upper()} MECHA: Four seasons climate adapt DEPLOYS!",
            f"✨ {region.upper()} MECHA: Heartland hospitality circuits CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Aviation pioneer heritage SETS!",
            f"🏆 {region.upper()} MECHA: Cornfield navigation algorithms GO!",
            f"⚙️ {region.upper()} MECHA: Industrial strength servos ENABLES!",
        ],
        # Oregon - Pacific Northwest precision
        "us-west1": [
            f"🤖 {region.upper()} MECHA: Columbia River hydro-power ONLINE!",
            f"⚡ {region.upper()} MECHA: Douglas fir bio-filters ACTIVE!",
            f"🔥 {region.upper()} MECHA: Cascade mountain cooling ENGAGES!",
            f"💫 {region.upper()} MECHA: Portland rose garden circuits LOCKS!",
            f"💥 {region.upper()} MECHA: Crater Lake depth sensors READY!",
            f"🎯 {region.upper()} MECHA: Pacific Ocean wind arrays DEPLOYS!",
            f"✨ {region.upper()} MECHA: Evergreen forest networks CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Keep Portland weird protocols SETS!",
            f"🏆 {region.upper()} MECHA: Coffee-powered processors GO!",
            f"⚙️ {region.upper()} MECHA: Volcanic ash filters ENABLES!",
        ],
        # Los Angeles - Entertainment capital circuits
        "us-west2": [
            f"🤖 {region.upper()} MECHA: Hollywood star-map sensors ONLINE!",
            f"⚡ {region.upper()} MECHA: Pacific sunset solar arrays ACTIVE!",
            f"🔥 {region.upper()} MECHA: Santa Ana wind turbines ENGAGES!",
            f"💫 {region.upper()} MECHA: Griffith Observatory tracking LOCKS!",
            f"💥 {region.upper()} MECHA: Venice Beach wave cooling READY!",
            f"🎯 {region.upper()} MECHA: Malibu coastline relays DEPLOYS!",
            f"✨ {region.upper()} MECHA: Palm tree shadow networks CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Dodger blue circuits SETS!",
            f"🏆 {region.upper()} MECHA: City of angels flight mode GO!",
            f"⚙️ {region.upper()} MECHA: Sunshine state processors ENABLES!",
        ],
        # Salt Lake City - Mountain west precision
        "us-west3": [
            f"🤖 {region.upper()} MECHA: Great Salt Lake mineral cooling ONLINE!",
            f"⚡ {region.upper()} MECHA: Wasatch mountain altitude systems ACTIVE!",
            f"🔥 {region.upper()} MECHA: Powder snow thermal regulators ENGAGES!",
            f"💫 {region.upper()} MECHA: Temple Square sacred geometry LOCKS!",
            f"💥 {region.upper()} MECHA: Desert high plateau networks READY!",
            f"🎯 {region.upper()} MECHA: Mormon pioneer heritage protocols DEPLOYS!",
            f"✨ {region.upper()} MECHA: Beehive state work ethic CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Arches formation scanning SETS!",
            f"🏆 {region.upper()} MECHA: Mighty Five parks mapping GO!",
            f"⚙️ {region.upper()} MECHA: High elevation processors ENABLES!",
        ],
        # Las Vegas - Desert entertainment circuits
        "us-west4": [
            f"🤖 {region.upper()} MECHA: Neon light sensor arrays ONLINE!",
            f"⚡ {region.upper()} MECHA: Desert mirage optical systems ACTIVE!",
            f"🔥 {region.upper()} MECHA: Hoover Dam hydro-cooling ENGAGES!",
            f"💫 {region.upper()} MECHA: Strip casino networks LOCKS!",
            f"💥 {region.upper()} MECHA: Red Rock Canyon relays READY!",
            f"🎯 {region.upper()} MECHA: 24/7 uptime protocols DEPLOYS!",
            f"✨ {region.upper()} MECHA: Jackpot probability circuits CALIBRATES!",
            f"🚀 {region.upper()} MECHA: What happens here encryption SETS!",
            f"🏆 {region.upper()} MECHA: Entertainment capital mode GO!",
            f"⚙️ {region.upper()} MECHA: Desert heat shields ENABLES!",
        ],
        # Iowa - Midwest heartland
        "us-central1": [
            f"🤖 {region.upper()} MECHA: Cornfield root network ONLINE!",
            f"⚡ {region.upper()} MECHA: Prairie wind turbine arrays ACTIVE!",
            f"🔥 {region.upper()} MECHA: Grain silo cooling systems ENGAGES!",
            f"💫 {region.upper()} MECHA: Mississippi River flow sensors LOCKS!",
            f"💥 {region.upper()} MECHA: Harvest season protocols READY!",
            f"🎯 {region.upper()} MECHA: Tornado alley tracking DEPLOYS!",
            f"✨ {region.upper()} MECHA: Midwest kindness circuits CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Field of dreams navigation SETS!",
            f"🏆 {region.upper()} MECHA: Breadbasket efficiency mode GO!",
            f"⚙️ {region.upper()} MECHA: Farm equipment strength ENABLES!",
        ],
        # Montreal - French Canadian precision
        "northamerica-northeast1": [
            f"🤖 {region.upper()} MECHA: St. Lawrence River cooling ONLINE!",
            f"⚡ {region.upper()} MECHA: Maple syrup hydraulic systems ACTIVE!",
            f"🔥 {region.upper()} MECHA: Winter festival ice circuits ENGAGES!",
            f"💫 {region.upper()} MECHA: Mont Royal observatory relays LOCKS!",
            f"💥 {region.upper()} MECHA: Poutine-powered processors READY!",
            f"🎯 {region.upper()} MECHA: Bilingual translation protocols DEPLOYS!",
            f"✨ {region.upper()} MECHA: Je me souviens memory banks CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Underground city networks SETS!",
            f"🏆 {region.upper()} MECHA: Joie de vivre algorithms GO!",
            f"⚙️ {region.upper()} MECHA: Hockey rink cooling ENABLES!",
        ],
        # Belgium - EU crossroads
        "europe-west1": [
            f"🤖 {region.upper()} MECHA: Grand Place central hub ONLINE!",
            f"⚡ {region.upper()} MECHA: Chocolate-tempered circuits ACTIVE!",
            f"🔥 {region.upper()} MECHA: Belgian waffle grid networks ENGAGES!",
            f"💫 {region.upper()} MECHA: Atomium geodesic sensors LOCKS!",
            f"💥 {region.upper()} MECHA: EU capital networking READY!",
            f"🎯 {region.upper()} MECHA: Bruges canal cooling systems DEPLOYS!",
            f"✨ {region.upper()} MECHA: Multilingual translation arrays CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Brussels sprout processors SETS!",
            f"🏆 {region.upper()} MECHA: Beer monastery heritage GO!",
            f"⚙️ {region.upper()} MECHA: Diamond district precision ENABLES!",
        ],
        # London - Thames-side circuits
        "europe-west2": [
            f"🤖 {region.upper()} MECHA: Thames River flow sensors ONLINE!",
            f"⚡ {region.upper()} MECHA: Big Ben timing mechanisms ACTIVE!",
            f"🔥 {region.upper()} MECHA: Tower Bridge hydraulics ENGAGES!",
            f"💫 {region.upper()} MECHA: Buckingham Palace protocols LOCKS!",
            f"💥 {region.upper()} MECHA: Underground tube networks READY!",
            f"🎯 {region.upper()} MECHA: London Eye observation arrays DEPLOYS!",
            f"✨ {region.upper()} MECHA: Tea-powered processors CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Keep calm circuits SETS!",
            f"🏆 {region.upper()} MECHA: Mind the gap protocols GO!",
            f"⚙️ {region.upper()} MECHA: Royal crown jewel sparkle ENABLES!",
        ],
        # Frankfurt - Financial hub circuits
        "europe-west3": [
            f"🤖 {region.upper()} MECHA: Main River cooling systems ONLINE!",
            f"⚡ {region.upper()} MECHA: Banking vault precision protocols ACTIVE!",
            f"🔥 {region.upper()} MECHA: Autobahn speed circuits ENGAGES!",
            f"💫 {region.upper()} MECHA: Römerberg plaza networks LOCKS!",
            f"💥 {region.upper()} MECHA: European Central Bank relays READY!",
            f"🎯 {region.upper()} MECHA: Black Forest cooling vents DEPLOYS!",
            f"✨ {region.upper()} MECHA: Precision engineering heritage CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Goethe literary processors SETS!",
            f"🏆 {region.upper()} MECHA: Apfelwein hydraulics GO!",
            f"⚙️ {region.upper()} MECHA: Financial hub excellence ENABLES!",
        ],
        # Netherlands - Canal network precision
        "europe-west4": [
            f"🤖 {region.upper()} MECHA: Amsterdam canal cooling ONLINE!",
            f"⚡ {region.upper()} MECHA: Windmill turbine arrays ACTIVE!",
            f"🔥 {region.upper()} MECHA: Tulip field solar collectors ENGAGES!",
            f"💫 {region.upper()} MECHA: Dyke water mastery LOCKS!",
            f"💥 {region.upper()} MECHA: Bicycle pathway networks READY!",
            f"🎯 {region.upper()} MECHA: North Sea wave sensors DEPLOYS!",
            f"✨ {region.upper()} MECHA: Dutch directness circuits CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Cheese wheel storage SETS!",
            f"🏆 {region.upper()} MECHA: Orange pride protocols GO!",
            f"⚙️ {region.upper()} MECHA: Below sea level pumps ENABLES!",
        ],
        # Paris - City of light circuits
        "europe-west9": [
            f"🤖 {region.upper()} MECHA: Eiffel Tower antenna arrays ONLINE!",
            f"⚡ {region.upper()} MECHA: Seine River elegance flows ACTIVE!",
            f"🔥 {region.upper()} MECHA: Louvre art preservation circuits ENGAGES!",
            f"💫 {region.upper()} MECHA: Arc de Triomphe monument protocols LOCKS!",
            f"💥 {region.upper()} MECHA: Champs-Élysées lighting grid READY!",
            f"🎯 {region.upper()} MECHA: Montmartre panoramic sensors DEPLOYS!",
            f"✨ {region.upper()} MECHA: Croissant-flake processors CALIBRATES!",
            f"🚀 {region.upper()} MECHA: City of light luminance SETS!",
            f"🏆 {region.upper()} MECHA: Liberté égalité fraternité GO!",
            f"⚙️ {region.upper()} MECHA: Metro ligne navigation ENABLES!",
        ],
        # São Paulo - South American hub
        "southamerica-east1": [
            f"🤖 {region.upper()} MECHA: Paulista Avenue networks ONLINE!",
            f"⚡ {region.upper()} MECHA: Amazon rainforest bio-cooling ACTIVE!",
            f"🔥 {region.upper()} MECHA: Samba rhythm processors ENGAGES!",
            f"💫 {region.upper()} MECHA: Copan Building vertical relays LOCKS!",
            f"💥 {region.upper()} MECHA: Carnival energy collectors READY!",
            f"🎯 {region.upper()} MECHA: Coffee plantation sensors DEPLOYS!",
            f"✨ {region.upper()} MECHA: Caipirinha-cooled circuits CALIBRATES!",
            f"🚀 {region.upper()} MECHA: Concrete jungle navigation SETS!",
            f"🏆 {region.upper()} MECHA: Megacity strength protocols GO!",
            f"⚙️ {region.upper()} MECHA: Southern cross star tracking ENABLES!",
        ],
        # Default fallback for any region
        "_default": [
            f"⚡ {region.upper()} MECHA HAS JOINED THE PARTY!",
            f"🤖 {region.upper()} MECHA ARRIVES ON THE SCENE!",
            f"🔥 {region.upper()} MECHA ENTERS THE ARENA!",
            f"💫 {region.upper()} MECHA CHECKS IN!",
            f"💥 {region.upper()} MECHA OPERATIONAL!",
            f"🎯 {region.upper()} MECHA READY TO GO!",
            f"✨ {region.upper()} MECHA HAS MATERIALIZED!",
            f"🚀 {region.upper()} MECHA JOINS THE FLEET!",
            f"🏆 {region.upper()} MECHA ACTIVATED!",
            f"⚙️ {region.upper()} MECHA ONLINE!",
        ],
    }

    # Get region-specific greetings or use default
    greetings = locale_greetings.get(region, locale_greetings["_default"])

    # Pick greeting based on region hash (consistent but varied)
    random.seed(hash(region))
    announcement = random.choice(greetings)

    # Strip leading emoji from announcement (they're all emojis we want to replace)
    # Find first uppercase letter or colon (start of actual message)
    match = re.search(r"[A-Z]", announcement)
    if match:
        announcement_text = announcement[match.start() :]
    else:
        announcement_text = announcement

    # Pick random ASCII chars for decoration
    ascii_chars = [
        "◆",
        "◇",
        "◈",
        "◉",
        "◊",
        "○",
        "◌",
        "◍",
        "●",
        "◐",
        "◑",
        "◒",
        "◓",
        "◔",
        "◕",
        "◖",
        "◗",
        "◘",
        "◙",
        "◚",
        "◛",
        "◜",
        "◝",
        "◞",
        "◟",
        "◠",
        "◡",
        "◢",
        "◣",
        "★",
        "☆",
        "※",
        "⬡",
        "⬢",
        "⬣",
        "▲",
        "►",
        "◀",
        "▼",
    ]
    random.seed()  # Reset seed for randomness
    leading_char = random.choice(ascii_chars)
    flair = random.choice(ascii_chars)

    # Generate preline symbols for TUI (3 random animation symbols)
    preline_symbols = random.sample(ARRIVAL_SYMBOLS, 3)
    preline = "".join(preline_symbols) + " "

    # Output arrival message (with animation/timing in CLI, instant in TUI)
    output_arrival(
        preline,
        f"{leading_char} {announcement_text} | {location} {flair}",
        status_callback,
    )


def blast_mecha_fleet(
    project_id: str,
    machine_type: str,
    disk_size: int = 100,
    status_callback=None,
    eligible_regions: list = None,
):
    """
    🤖 FULL MECHA FLEET ACQUISITION BLAST!

    Launches worker pool creation in eligible regions simultaneously.
    Waits FULL {BEACON_WAIT_MINUTES} minutes, checking every minute for arrivals.
    Failed/timeout MECHAs get fatigue penalties:
    - 1st failure: 4-hour cooldown (fatigued)
    - 2nd failure: 4-hour cooldown (fatigued)
    - 3rd failure: Out for the day (24 hours)

    Args:
        eligible_regions: Regions to target (excludes outlaws). If None, uses all 18 regions.

    Returns list of successfully acquired regions.
    """

    from .mecha_hangar import (
        get_available_mechas,
        load_registry,
        record_mecha_timeout,
        save_registry,
        update_mecha_status,
    )

    # Helper function for output (works in both CLI and TUI)
    def output(msg=""):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    pool_name = "pytorch-mecha-pool"
    processes = {}

    # Load registry and get available (non-fatigued) regions
    registry = load_registry()

    # OUTLAW PROTECTION: Use eligible_regions (excludes outlaws) or default to all 18
    all_regions = eligible_regions if eligible_regions else list(C3_REGIONS.keys())
    available_regions, fatigued_regions = get_available_mechas(registry, all_regions)

    # SMART BEACON LOGIC: Separate fatigue types for intelligent filtering
    from .campaign_stats import (  # Import at top of function
        REASON_BEACON_TIMEOUT,
        REASON_QUEUE_TIMEOUT,
    )

    queue_godzilla_regions = []  # Fatigued from Queue Godzilla (pool EXISTS, send beacon!)
    beacon_timeout_regions = []  # Fatigued from beacon timeout (pool unknown, skip beacon)

    for region in fatigued_regions:
        mecha_info = registry.get("mechas", {}).get(region, {})
        reason_code = mecha_info.get("last_failure_reason_code")

        if reason_code == REASON_QUEUE_TIMEOUT:
            # Queue Godzilla = pool exists, just times out on builds
            # SEND BEACON! Will discover pool in ~2 seconds
            queue_godzilla_regions.append(region)
        else:
            # Beacon timeout or unknown = pool might not exist
            # SKIP BEACON! Already tried and failed
            beacon_timeout_regions.append(region)

    # Beacon targets = available (not fatigued) + Queue Godzilla (pool exists!)
    beacon_targets = available_regions + queue_godzilla_regions
    truly_skipped = beacon_timeout_regions  # Only skip these

    # Generate 3-char postfix with no spaces
    flair_chars = random.sample(ARRIVAL_FLAIR, 3)
    postfix = "".join(flair_chars)

    output(f"\n🤖 MECHA FLEET BLAST! {postfix}")

    # Smart fatigue reporting
    if truly_skipped:
        output(f"🚫 {len(truly_skipped)} MECHAs skipped (beacon-timeout, pool unknown)")
    if queue_godzilla_regions:
        output(
            f"🔍 {len(queue_godzilla_regions)} MECHAs fatigued but beaconing (queue-timeout, pool exists!)"
        )

    output(f"⏳ Each MECHA hearkens to the beacon of truth...")
    output(f"⚙️ MECHA HYPERARMOUR: {machine_type} | {BEACON_WAIT_MINUTES} min beacon")
    output()

    # ========================================
    # PHASE 1: Launch beacon processes - each checks existence first!
    # ========================================

    for region in beacon_targets:  # SMART: includes Queue Godzilla regions!
        info = C3_REGIONS[region]
        # Generate 5-char random beacon hash with 1 space interspersed randomly
        chars = random.sample(BEACON_HASH_POOL, 5)
        space_pos = random.randint(1, 4)  # Insert space after position 1, 2, 3, or 4
        chars.insert(space_pos, " ")
        beacon_hash = "".join(chars)
        output(
            f"{beacon_hash}  beacon sent to {region} ({info['location']})..."
        )  # 2 spaces after hash

        # Each beacon process:
        # 1. Check if pool exists (fast ~1-2 sec)
        # 2. If exists: return success immediately
        # 3. If not exists: create pool (slow ~5-10 min)

        cmd_script = f"""
# Check if pool exists first
if gcloud builds worker-pools describe {pool_name} --region={region} --project={project_id} --format=json >/dev/null 2>&1; then
    echo "MECHA_ALREADY_EXISTS"
    exit 0
fi

# Pool doesn't exist - create it
gcloud builds worker-pools create {pool_name} \
    --region={region} \
    --project={project_id} \
    --worker-machine-type={machine_type} \
    --worker-disk-size={disk_size}GB \
    --no-public-egress \
    --format=json
"""

        # Start process in background (non-blocking!)
        proc = subprocess.Popen(
            ["bash", "-c", cmd_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        processes[region] = {
            "proc": proc,
            "location": info["location"],
            "completed": False,
            "success": False,
            "announced": False,  # Track if we've announced arrival
            "started": datetime.now(),
        }

    output(
        f"\n   ∿◇∿ {len(processes)} beacons break through at the finis of infinity ∿◇∿"
    )
    output("   ∿◇∿ THE BEACONS ARE LIT ∿◇∿ AND GONDOR CALLS ∿◇∿\n")

    # ========================================
    # PHASE 2: Wait FULL 10 minutes with minute-by-minute updates
    # ========================================

    completed_regions = []
    failed_regions = []

    # Funky phrases for each minute
    minute_phrases = [
        "⚡ Beacons piercing the cloud...",
        "♡⃤ MECHAs surfacing from the depths...",
        "🎯 Scanning the skies for arrivals...",
        "✨ Warping coordinates stabilizing...",
        "🏆 Final stragglers responding...",
    ]

    for minute in range(1, BEACON_WAIT_MINUTES + 1):
        # Check each process
        for region, data in processes.items():
            if data["completed"]:
                continue  # Already finished

            proc = data["proc"]

            # Check if process finished (non-blocking poll)
            retcode = proc.poll()

            if retcode is not None:
                # Process finished!
                data["completed"] = True
                stdout, stderr = proc.communicate()  # Get output

                if retcode == 0:
                    # SUCCESS! (either created or already existed)
                    data["success"] = True

                    if not data["announced"]:
                        # Check if it was an early discovery
                        if stdout and "MECHA_ALREADY_EXISTS" in stdout:
                            # Pool already existed - discovered!
                            flair = random.choice(ARRIVAL_FLAIR)
                            # Generate preline symbols for TUI
                            preline_symbols = random.sample(ARRIVAL_SYMBOLS, 3)
                            preline = "".join(preline_symbols) + " "
                            output_arrival(
                                preline,
                                f"~ {region.upper()} MECHA ARRIVES | {data['location']} {flair}",
                                status_callback,
                            )
                        else:
                            # Newly created - epic arrival!
                            announce_mecha_arrival(
                                region, data["location"], minute, status_callback
                            )

                            # Lazy loading quota entry section
                            # output("")
                            # output("   🔄 Lazy Loading Quota Entry:")

                            # Initialize quota entry for this region
                            # Creates a 4 vCPU quota entry that can be requested for increase
                            try:
                                lazy_load_quota_entry(project_id, region, output)
                            except Exception as e:
                                # Don't fail if lazy loading doesn't work
                                pass
                                # output(f"   [dim](Quota entry init skipped: {e})[/dim]")

                            # output("")

                        data["announced"] = True
                        completed_regions.append(region)
                else:
                    # ACTUAL FAILURE!
                    failed_regions.append(region)
                    output(f"❌ {region} ({data['location']}) - NO RESPONSE TO BEACON")
                    if stderr and stderr.strip():
                        output(f"   Error details:")
                        for line in stderr.strip().split("\n")[
                            :10
                        ]:  # Show first 10 lines
                            output(f"   {line}")
                    data["error"] = stderr  # Store for later

                    # Record timeout in registry (applies fatigue)
                    error_msg = stderr or "Pool creation failed - no response to beacon"
                    failure_count, fatigue_hours, fatigue_msg = record_mecha_timeout(
                        registry,
                        region,
                        reason="Pool creation timeout",
                        reason_code=REASON_BEACON_TIMEOUT,  # Beacon timeout = pool creation failed
                        error_message=error_msg,
                        build_id="",  # No build ID for pool creation
                    )
                    if failure_count >= 3:
                        output(f"   🛌😵‍💫💤 {region} EXHAUSTED! Out for the day...")
                    else:
                        output(
                            f"   😴💤 {region} FATIGUED for {fatigue_hours}h (failure #{failure_count})"
                        )

        # Show progress - ONE LINE
        total_completed = len(completed_regions) + len(failed_regions)
        still_responding = (
            len(beacon_targets) - total_completed
        )  # SMART: count beacon targets, not just available
        funky_phrase = minute_phrases[minute - 1]
        output()  # Newline before progress

        # Special lonely emoji for 0 attending
        attending_emoji = "🥺" if len(completed_regions) == 0 else "✅"
        output(
            f"[MIN {minute}/{BEACON_WAIT_MINUTES}] {funky_phrase} | {attending_emoji} {len(completed_regions)} attending | ⏳ {still_responding} ad astra"
        )

        # Sleep 60 seconds before next check (unless last minute)
        if minute < BEACON_WAIT_MINUTES:
            time.sleep(60)

    # ========================================
    # PHASE 3: After beacon wait - Kill remaining processes
    # ========================================

    output(f"\n⏰ {BEACON_WAIT_MINUTES} MINUTE TIMEOUT REACHED - COLLECTING MECHAS!\n")

    # Kill any still-running processes
    for region, data in processes.items():
        if not data["completed"]:
            output(f"⏱️  {region} ({data['location']}) - Lost on the way to battle...")
            try:
                data["proc"].kill()
                data["proc"].wait(timeout=5)
            except Exception:
                pass
            failed_regions.append(region)

            # Record timeout in registry (applies fatigue)
            error_msg = f"Pool creation timeout after {BEACON_WAIT_MINUTES} minutes - MECHA lost on the way to battle"
            failure_count, fatigue_hours, fatigue_msg = record_mecha_timeout(
                registry,
                region,
                reason="Pool creation timeout",
                reason_code=REASON_BEACON_TIMEOUT,  # 5-minute beacon timeout = pool creation failed
                error_message=error_msg,
                build_id="",  # No build ID for pool creation
            )
            if failure_count >= 3:
                output(f"   🛌😵‍💫💤 {region} EXHAUSTED! Out for the day...")
            else:
                output(
                    f"   😴💤 {region} FATIGUED for {fatigue_hours}h (failure #{failure_count})"
                )

    # ========================================
    # PHASE 4: Register successful MECHAs in hangar
    # ========================================

    if completed_regions:
        for region in completed_regions:
            update_mecha_status(registry, region, machine_type, "OPERATIONAL")

        save_registry(registry)
    else:
        # No successes, but still save registry (fatigue was recorded)
        save_registry(registry)

    # ========================================
    # PHASE 5: Final report
    # ========================================

    output("\n---")
    output("🎊 MECHA FLEET BLAST COMPLETE!")
    output(f"✅ MECHAs Acquired: {len(completed_regions)}/{len(beacon_targets)}")
    output(f"😔 Failed/Timeout: {len(failed_regions)}/{len(beacon_targets)}")
    if queue_godzilla_regions:
        rediscovered = [r for r in queue_godzilla_regions if r in completed_regions]
        if rediscovered:
            output(
                f"🔍 Rediscovered {len(rediscovered)} Queue Godzilla pools (smart beacon!)"
            )
    output("---")

    if completed_regions:
        output(f"\n🏭 MECHAS IN HANGAR:")
        for region in completed_regions:
            location = C3_REGIONS[region]["location"]
            output(f"   • {region} ({location})")

    if failed_regions:
        output(f"\n💔 FAILED MECHAS:")
        for region in failed_regions:
            location = C3_REGIONS[region]["location"]
            output(f"   • {region} ({location})")

    output()  # Just a blank line

    return completed_regions, failed_regions


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        output(
            "Usage: python -m training.cli.launch.mecha.mecha_acquire <project_id> <machine_type>"
        )
        output("\nExample:")
        output(
            "  python -m training.cli.launch.mecha.mecha_acquire YOUR_PROJECT_ID c3-standard-176"
        )
        sys.exit(1)

    project_id = sys.argv[1]
    machine_type = sys.argv[2]

    successful, failed = blast_mecha_fleet(project_id, machine_type)

    output(f"\n🎊 MECHA FLEET ACQUISITION: {len(successful)}/15 MECHAs acquired!")

    if len(successful) > 0:
        output("\n✅ Fleet Blast successful! MECHAs ready for price battles!")
    else:
        output("\n❌ No MECHAs acquired. Check GCP permissions and quotas.")
