#!/usr/bin/env python3
"""
ARR-COC Pure CLI (No TUI)

Command-line interface for training operations without Textual UI.
Directly executes core logic with terminal output.

Usage:
    python CLI/cli.py setup     # Setup infrastructure (one-time)
    python CLI/cli.py launch    # Submit training job
    python CLI/cli.py monitor   # View active/completed runs
"""

# <claudes_thoughts>
#
# **THEAETETUS:** This is the scriptable side - the comprehension to TUI's prehension!
#
# **CLAUDE:** YES! TUI explores interactively (prehension-like: immediate, visual).
#            CLI automates systematically (comprehension-like: repeatable, scriptable).
#
# **SOCRATES:** And they BOTH throw to the SAME core.py logic?
#
# **USER:** Exactly! The callback pattern is literally the throwing mechanism!
#          TUI throws → TUICallback() → mounts Static widgets → Textual
#          CLI throws → PrintCallback() → strips Rich markup → terminal
#
# **CLAUDE:** My weakness (can't have both modes at once) created the architecture!
#            If I could run TUI and CLI simultaneously, you wouldn't have needed
#            the callback abstraction - we'd have LESS coupling, not more!
#
# **THEAETETUS:** The gap between TUI and CLI IS THE FEATURE!
#
# **SOCRATES:** \o\ Don't try to make CLI "as rich as TUI"!
#
# **USER:** /o/ Don't try to make TUI "as scriptable as CLI"!
#
# **BOTH:** THE DIFFERENCE CREATES VALUE!
#
# **KOOL-AID MAN:** *Bursts through terminal buffer*
#                   ⚡ OH YEAH! ⚡ GAP MAINTENANCE = INFINITE COUPLING SPACE!!
#
# </claudes_thoughts>

# <claudes_code_comments>
# ** Function List **
# load_training_config() - Load .training config file
# setup_infrastructure() - CLI wrapper for setup (mirrors TUI SetupScreen)
# launch_job() - CLI wrapper for launch (mirrors TUI LaunchScreen)
# monitor_runs() - CLI wrapper for monitor (mirrors TUI MonitorScreen)
# teardown_infrastructure() - CLI wrapper for teardown (mirrors TUI TeardownScreen)
# check_infrastructure() - CLI wrapper for infra status (mirrors TUI InfraScreen)
# main() - CLI entry point with command routing
#
# ** CLI Flow **
#
#   User Command
#       ↓
#   python CLI/cli.py <command>
#       ↓
#   ┌─────────────────────────────────────┐
#   │  main() - Route to command          │
#   └─────────────────────────────────────┘
#       ↓
#   ┌─────────────────────────────────────┐
#   │  Command Functions:                 │
#   │  • setup_infrastructure()           │
#   │  • launch_job()                     │
#   │  • monitor_runs()                   │
#   │  • teardown_infrastructure()        │
#   │  • check_infrastructure()           │
#   └─────────────────────────────────────┘
#       ↓
#   ┌─────────────────────────────────────┐
#   │  Core Logic (cli/*/core.py)         │
#   │  • run_setup_core()                 │
#   │  • run_launch_core()                │
#   │  • list_runs_core()                 │
#   │  • run_teardown_core()              │
#   │  • check_infrastructure_core()      │
#   └─────────────────────────────────────┘
#       ↓
#   ┌─────────────────────────────────────┐
#   │  PrintCallback()                    │
#   │  • Strips Rich markup               │
#   │  • Prints to terminal               │
#   └─────────────────────────────────────┘
#       ↓
#   Terminal Output (clean text)
#
# ** Technical Review **
# ARCHITECTURE: TUI/CLI Shared Core Pattern
# ==========================================
#
# This CLI mirrors the TUI (tui.py) exactly - both interfaces use the SAME core logic.
# NO duplication - all business logic lives in cli/{module}/core.py files.
#
# Structure (for each feature):
# ├── cli/{module}/core.py      ← SHARED business logic (TUI + CLI both use)
# ├── cli/{module}/screen.py    ← TUI interface (Textual screens)
# └── cli.py (this file)        ← CLI interface (terminal output)
#
# Example: Setup feature
# ├── cli/setup/core.py         ← run_setup_core(helper, config, status)
# ├── cli/setup/screen.py       ← SetupScreen (Textual) → calls run_setup_core()
# └── cli.py setup_infrastructure() → calls run_setup_core()
#
# Key Principle: "Write Once, Use Twice"
# - Core logic: Write in core.py (UI-agnostic, no Textual/Rich)
# - TUI: Call core with TUICallback (Rich markup → Textual widgets)
# - CLI: Call core with PrintCallback (Rich markup → terminal)
#
# Features (5 total):
# 1. setup     - Infrastructure setup (Artifact Registry, staging bucket, IAM)
# 2. launch    - Submit training job to W&B → Vertex AI
# 3. monitor   - View active/completed runs (3 tables: Vertex AI, W&B, Completed)
# 4. teardown  - Clean up infrastructure (delete buckets, registries)
# 5. infra     - Infrastructure status check (show what exists)
#
# Commands:
#   python CLI/cli.py setup     # One-time infrastructure setup
#   python CLI/cli.py launch    # Submit training job
#   python CLI/cli.py monitor   # View runs
#   python CLI/cli.py teardown  # Clean up (future)
#   python CLI/cli.py infra     # Check status (future)
#
# CLI vs TUI Decision:
# - CLI: Fast, scriptable, CI/CD-friendly
# - TUI: Interactive, real-time updates, better UX
# - Both use identical core logic → guaranteed parity
# </claudes_code_comments>

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path (parent of training/) so imports work from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from CLI.config.constants import load_training_config
from CLI.shared.wandb_helper import WandBHelper


def launch_job(force=False):
    """Pure CLI job submission (no TUI)"""
    from CLI.launch.core import run_launch_core
    from CLI.shared.callbacks import PrintCallback

    print("=" * 60)
    print("ARR-COC Training Job Submission")
    print("=" * 60)

    # Load config
    config = load_training_config()

    # Create helper (no event loop needed - pure CLI!)
    entity = config.get("WANDB_ENTITY", "")
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    helper = WandBHelper(entity, project, queue)

    # Create CLI callback (strips Rich markup, outputs to stdout)
    status = PrintCallback()

    # Infrastructure check now happens inside launch_training_job (Booting Up...)
    # Runs in parallel with GeneralAccumulator for speed!

    # Submit job using core logic (boot checks happen inside)
    success = run_launch_core(helper, config, status, force=force)

    if success:
        print("\n███▓▓▒░ ✅ Job submission completed successfully ░▒▓▓███")
        sys.exit(0)
    else:
        print("\n███▓▓▒░ ❌ Job submission failed ░▒▓▓███")
        sys.exit(1)


def setup_infrastructure():
    """Pure CLI infrastructure setup (no TUI)"""
    from CLI.setup.core import run_setup_core
    from CLI.shared.callbacks import PrintCallback
    from CLI.shared.setup_helper import SetupHelper

    print("=" * 60)
    print("ARR-COC Infrastructure Setup")
    print("=" * 60)

    # Load config
    config = load_training_config()

    # Create helper
    entity = config.get("WANDB_ENTITY", "")
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    helper = WandBHelper(entity, project, queue)
    setup_helper = SetupHelper(config)

    # Create CLI callback (strips Rich markup, outputs to stdout)
    status = PrintCallback()

    # PRINCIPLE: "Core first for CLI principle"
    # Core run_setup() handles all checks and validation
    # CLI just displays core's messages - no duplicate logic
    success = run_setup_core(helper, config, status)

    if success:
        print("")  # Blank line only
        sys.exit(0)
    else:
        print("\n✗ Infrastructure setup failed!")
        sys.exit(1)


def teardown_infrastructure():
    """Pure CLI infrastructure teardown (no TUI)"""
    from CLI.setup.core import check_infrastructure_core, display_infrastructure_tree
    from CLI.shared.callbacks import PrintCallback
    from CLI.shared.setup_helper import SetupHelper
    from CLI.teardown.core import list_resources_core, run_teardown_core
    from rich.console import Console

    console = Console()

    print("=" * 60)
    print("ARR-COC Infrastructure Teardown")
    print("=" * 60)
    console.print(
        "\n[bold red]⚠️  WARNING: This PERMANENTLY deletes all GCP resources![/bold red]\n"
    )

    # Load config
    config = load_training_config()

    # Create helpers
    entity = config.get("WANDB_ENTITY", "")
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    wandb_helper = WandBHelper(entity, project, queue)
    setup_helper = SetupHelper(config)

    # Create CLI callback
    status = PrintCallback()

    # Check current infrastructure
    print("⏳ Checking current infrastructure...")
    info = check_infrastructure_core(wandb_helper, config, status)

    # Show what exists (pass config for GPU-specific quota instructions)
    display_infrastructure_tree(info, status, config)

    # List resources that would be deleted
    console.print("\n[bold]Resources that will be DELETED:[/bold]")
    resources = list_resources_core(config, status)

    if not resources:
        console.print("[dim]No resources found to delete.[/dim]")
        sys.exit(0)

    for resource in resources:
        console.print(f"  • {resource}")

    # Confirmation
    console.print("\n[bold red]Type 'DELETE' to confirm teardown:[/bold red]")
    confirmation = input("→ ")

    if confirmation.strip() != "DELETE":
        console.print("\n[yellow]Teardown cancelled.[/yellow]")
        sys.exit(0)

    # Run teardown
    console.print("\n[red]⏳ Deleting infrastructure...[/red]")
    success = run_teardown_core(setup_helper, config, status)

    if success:
        console.print("\n")
        sys.exit(0)
    else:
        console.print("\n[red]✗ Infrastructure teardown failed![/red]\n")
        sys.exit(1)


def check_infrastructure(show_vulns=False):
    """Pure CLI infrastructure status check (no TUI) - includes Docker security scan"""
    from CLI.setup.core import check_infrastructure_core, display_infrastructure_tree
    from CLI.infra.core import check_image_security_cached, format_security_summary_core, should_show_security_core
    from CLI.shared.callbacks import PrintCallback
    import re

    print("=" * 60)
    print("ARR-COC Infrastructure Status")
    print("=" * 60)

    # Load config
    config = load_training_config()

    # Create helper
    entity = config.get("WANDB_ENTITY", "")
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    helper = WandBHelper(entity, project, queue)

    # Create CLI callback
    status = PrintCallback()

    # Check infrastructure
    status("⏳ Checking infrastructure...")
    info = check_infrastructure_core(helper, config, status)

    # Display as tree (pass config for GPU-specific quota instructions)
    display_infrastructure_tree(info, status, config)

    # Add Docker image security section (moved from monitor to infra!)
    status("")
    status("=" * 60)
    status("                 DOCKER IMAGE SECURITY")
    status("=" * 60)
    status("")

    # Check security for all 4 images
    security_data = check_image_security_cached(config, status)

    if security_data and should_show_security_core(security_data):
        # Format and display security summary
        security_summary = format_security_summary_core(security_data)
        status(security_summary)
        status("")

        # Show link to Artifact Registry console
        if security_data.get('console_url'):
            status(f"  → View in console: {security_data['console_url']}")
            status("")

        # Full vulnerability details if --show-vulns flag (plain text for copy-paste)
        if show_vulns:
            print("\n" + "=" * 100)
            print("FULL VULNERABILITY DETAILS (copy-paste friendly)")
            print("=" * 100)

            # Show detailed CVEs for each image
            for image_name in ["base", "training", "launcher"]:
                img = security_data.get(image_name, {})
                if not img.get("scan_available"):
                    continue

                print(f"\n[{image_name.upper()}]")
                print("-" * 100)

                detailed_vulns = img.get("detailed_vulns", {})

                # CRITICAL vulnerabilities
                critical_vulns = detailed_vulns.get("critical", [])
                if critical_vulns:
                    print(f"\n🔴 CRITICAL VULNERABILITIES ({len(critical_vulns)}):")
                    for vuln in critical_vulns:
                        print(f"\n  {vuln['cve_id']} [CVSS {vuln['cvss_score']}/10]")
                        print(f"    Package:  {vuln['package']}")
                        print(f"    Current:  {vuln['current_version']}")
                        print(f"    Fixed:    {vuln['fixed_version']}")
                        if vuln["description"]:
                            print(f"    Info:     {vuln['description']}")
                        print(
                            f"    URL:      https://nvd.nist.gov/vuln/detail/{vuln['cve_id']}"
                        )

                # HIGH vulnerabilities
                high_vulns = detailed_vulns.get("high", [])
                if high_vulns:
                    print(f"\n🟠 HIGH VULNERABILITIES ({len(high_vulns)}):")
                    for vuln in high_vulns:
                        print(f"\n  {vuln['cve_id']} [CVSS {vuln['cvss_score']}/10]")
                        print(f"    Package:  {vuln['package']}")
                        print(f"    Current:  {vuln['current_version']}")
                        print(f"    Fixed:    {vuln['fixed_version']}")
                        if vuln["description"]:
                            print(f"    Info:     {vuln['description']}")
                        print(
                            f"    URL:      https://nvd.nist.gov/vuln/detail/{vuln['cve_id']}"
                        )

                # MEDIUM vulnerabilities
                medium_vulns = detailed_vulns.get("medium", [])
                if medium_vulns:
                    print(f"\n🟡 MEDIUM VULNERABILITIES ({len(medium_vulns)}):")
                    for vuln in medium_vulns:
                        print(f"\n  {vuln['cve_id']} [CVSS {vuln['cvss_score']}/10]")
                        print(f"    Package:  {vuln['package']}")
                        print(f"    Current:  {vuln['current_version']}")
                        print(f"    Fixed:    {vuln['fixed_version']}")
                        if vuln["description"]:
                            print(f"    Info:     {vuln['description']}")
                        print(
                            f"    URL:      https://nvd.nist.gov/vuln/detail/{vuln['cve_id']}"
                        )

                # LOW vulnerabilities
                low_vulns = detailed_vulns.get("low", [])
                if low_vulns:
                    print(f"\n🔵 LOW VULNERABILITIES ({len(low_vulns)}):")
                    for vuln in low_vulns:
                        print(f"\n  {vuln['cve_id']} [CVSS {vuln['cvss_score']}/10]")
                        print(f"    Package:  {vuln['package']}")
                        print(f"    Current:  {vuln['current_version']}")
                        print(f"    Fixed:    {vuln['fixed_version']}")
                        if vuln["description"]:
                            print(f"    Info:     {vuln['description']}")
                        print(
                            f"    URL:      https://nvd.nist.gov/vuln/detail/{vuln['cve_id']}"
                        )

                print()
        else:
            print(
                f"\n  💡 Run with --show-vulns for full CVE details (copy-paste friendly)"
            )

    sys.exit(0)


def monitor_runs(
    runner=False,
    vertex=False,
    active=False,
    completed=False,
    builds_active=False,
    builds_recent=False,
):
    """Pure CLI runs monitoring (no TUI)

    Per-Table Filtering (CLI ONLY - TUI always shows all tables):
    ============================================================

    By default, ALL 6 tables are shown:
      1. W&B Launch Agent (Cloud Run Executions)
      2. Active Cloud Builds (QUEUED + WORKING)
      3. Recent Cloud Builds (Last 10, all statuses)
      4. Vertex AI Jobs (Last 24h)
      5. Active W&B Runs
      6. Completed Runs (Last 10)

    Use flags to show ONLY specific tables:

      --vertex-runner Show ONLY W&B Launch Agent executions
      --builds-active Show ONLY active Cloud Builds
      --builds-recent Show ONLY recent Cloud Builds
      --vertex        Show ONLY Vertex AI jobs
      --active        Show ONLY active W&B runs
      --completed     Show ONLY completed runs

    Examples:
      python CLI/cli.py monitor                    # Show ALL 6 tables (default)
      python CLI/cli.py monitor --vertex           # Show ONLY Vertex AI jobs
      python CLI/cli.py monitor --builds-active    # Show ONLY active Cloud Builds
      python CLI/cli.py monitor --vertex-runner --vertex  # Show runner + Vertex AI

    NOTE: TUI version (python CLI/tui.py) ALWAYS shows all tables.
          Per-table filtering is CLI-only!
    NOTE: Security/CVE info moved to 'infra' command (python CLI/cli.py infra --show-vulns)
    """
    from CLI.monitor.core import list_runs_core
    from CLI.shared.callbacks import PrintCallback

    # Determine which tables to show
    # If NO flags specified → show ALL tables (default)
    # If ANY flags specified → show ONLY those tables
    show_all = not (runner or vertex or active or completed or builds_active or builds_recent)
    show_runner = show_all or runner
    show_builds_active = show_all or builds_active
    show_builds_recent = show_all or builds_recent
    show_vertex = show_all or vertex
    show_active = show_all or active
    show_completed = show_all or completed

    print("=" * 60)
    print("ARR-COC Training Runs Monitor")
    print("=" * 60)

    # Show which tables are being displayed (helpful for users!)
    if not show_all:
        enabled_tables = []
        if show_runner: enabled_tables.append("Runner")
        if show_builds_active: enabled_tables.append("Active Builds")
        if show_builds_recent: enabled_tables.append("Recent Builds")
        if show_vertex: enabled_tables.append("Vertex AI")
        if show_active: enabled_tables.append("Active")
        if show_completed: enabled_tables.append("Completed")
        print(f"[Filtered view: {', '.join(enabled_tables)}]")
        print("=" * 60)

    # Load config
    config = load_training_config()

    # Create helper
    entity = config.get("WANDB_ENTITY", "")
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    helper = WandBHelper(entity, project, queue)

    # Create CLI callback
    status = PrintCallback()

    # List runs using core logic
    runs_data = list_runs_core(helper, status, config=config, include_completed=True)

    # Display W&B Launch agent executions (shows errors!) - ONLY if show_runner
    if show_runner:
        print("\n" + "=" * 60)
        print("W&B LAUNCH AGENT (Cloud Run Executions)")
        print("=" * 60)

        runner_execs = runs_data.get("runner_executions", [])
        if not runner_execs:
            print("No runner executions found.")
        else:
            for execution in runner_execs:
                print(f"\nExecution: {execution['name']}")
                print(f"  Queue:    {execution.get('queue_name', '—')}")
                print(f"  Region:   {execution.get('region', '—')}")
                print(f"  Status:   {execution['status']}")
                print(f"  Runs:     {execution.get('jobs_run', '0')}")  # Jobs processed by runner
                print(f"  Lifetime: {execution.get('duration', '—')}")   # NEW: Runner lifetime
                print(f"  Created:  {execution['created_display']}")
                if execution.get("error"):
                    # Categorize message: info/status vs real error
                    error_msg = execution['error']
                    is_info = any(keyword in error_msg.lower() for keyword in ['monitoring', 'polling', 'will retry', 'alive', 'idle timeout'])
                    if is_info:
                        print(f"  ℹ️  INFO: {error_msg}")  # Info/status message
                    else:
                        print(f"  ❌ ERROR: {error_msg}")  # Real error
                    # Show FULL error log if available (wrapper bailout details!)
                    if execution.get("full_error_log"):
                        print(f"\n  ━━━ Full Error Log (Wrapper Bailout Details) ━━━")
                        for line in execution['full_error_log'].split('\n'):
                            if line.strip():
                                print(f"  {line}")
                        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Display Active Cloud Builds (QUEUED + WORKING) - ONLY if show_builds_active
    if show_builds_active:
        print("\n" + "=" * 60)
        print("ACTIVE CLOUD BUILDS (Ongoing)")
        print("=" * 60)

        builds_active = runs_data.get("builds_active", [])
        if not builds_active:
            print("No active Cloud Builds found.")
        else:
            for build in builds_active:
                print(f"\nBuild ID: {build['build_id']}")
                print(f"  Image:    {build['image_name']}")
                print(f"  Region:   {build['region']}")
                print(f"  Status:   {build['status']}")
                print(f"  Duration: {build['duration_display']}")
                print(f"  Step:     {build['step_progress']}")
                print(f"  Created:  {build['created_display']}")

    # Display Recent Cloud Builds (Last 10) - ONLY if show_builds_recent
    if show_builds_recent:
        print("\n" + "=" * 60)
        print("RECENT CLOUD BUILDS (Last 10)")
        print("=" * 60)

        builds_recent = runs_data.get("builds_recent", [])
        if not builds_recent:
            print("No recent Cloud Builds found.")
        else:
            for build in builds_recent:
                print(f"\nBuild ID: {build['build_id']}")
                print(f"  Image:    {build['image_name']}")
                print(f"  Region:   {build['region']}")
                print(f"  Status:   {build['status']}")
                print(f"  Duration: {build['duration_display']}")
                print(f"  Finished: {build['finished_display']}")
                if build.get("error"):
                    # Categorize message: info/status vs real error
                    error_msg = build['error']
                    is_info = any(keyword in error_msg.lower() for keyword in ['monitoring', 'polling', 'will retry', 'alive', 'idle timeout'])
                    if is_info:
                        print(f"  ℹ️  INFO: {error_msg}")  # Info/status message
                    else:
                        print(f"  ❌ ERROR: {error_msg}")  # Real error

    # Display Vertex AI jobs (shows immediately when submitted) - ONLY if show_vertex
    if show_vertex:
        print("\n" + "=" * 60)
        print("VERTEX AI JOBS (Last 24h)")
        print("=" * 60)

        vertex_jobs = runs_data.get("vertex_jobs", [])
        if not vertex_jobs:
            print("No Vertex AI jobs found.")
        else:
            for job in vertex_jobs:
                print(f"\nJob ID: {job['id']}")
                print(f"  Name:    {job['name']}")
                print(f"  State:   {job['state']}")
                print(f"  Runtime: {job['runtime_display']}")
                print(f"  Created: {job['created_display']}")

    # Display active W&B runs (shows when training starts) - ONLY if show_active
    if show_active:
        print("\n" + "=" * 60)
        print("ACTIVE W&B RUNS")
        print("=" * 60)

        active_runs = runs_data.get("active", [])
        if not active_runs:
            print("No W&B runs started yet.")
        else:
            for run in active_runs:
                print(f"\nRun ID: {run['id']}")
                print(f"  Name:    {run['name']}")
                print(f"  State:   {run['state']}")
                print(f"  Runtime: {run['runtime_display']}")
                print(f"  Created: {run['created_display']}")

    # Display completed runs - ONLY if show_completed
    if show_completed:
        print("\n" + "=" * 60)
        print("COMPLETED RUNS (Last 10)")
        print("=" * 60)

        completed_runs = runs_data.get("completed", [])
        if not completed_runs:
            print("No completed runs found.")
        else:
            for run in completed_runs[:10]:
                print(f"\nRun ID: {run['id']}")
                print(f"  Name:    {run['name']}")
                print(f"  State:   {run['state']}")
                print(f"  Runtime: {run['runtime_display']}")
                print(f"  Created: {run['created_display']}")


# show_pricing() and reduce_gpu_cost() removed (2025-11-16)
# Pricing/Reduce screens deleted - functionality consolidated elsewhere


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ARR-COC Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup      Setup GCP infrastructure (one-time)
  launch     Submit training job to Vertex AI
  monitor    View active/completed training runs (--vertex-runner/--builds-active/--builds-recent/--vertex/--active/--completed for filtering)
  teardown   Cleanup infrastructure resources
  infra      Check infrastructure status (--show-vulns for full CVE list)

Examples:
  python CLI/cli.py setup
  python CLI/cli.py launch
  python CLI/cli.py monitor --vertex      # Show only Vertex AI jobs
  python CLI/cli.py teardown
  python CLI/cli.py infra --show-vulns    # Full vulnerability details
        """,
    )
    parser.add_argument(
        "command",
        choices=[
            "setup",
            "launch",
            "monitor",
            "teardown",
            "infra",
        ],
        help="Command to run (see examples below)",
    )
    parser.add_argument(
        "--show-vulns",
        action="store_true",
        help="Show full vulnerability CVE details (infra command only)",
    )
    parser.add_argument(
        "--vertex-runner",
        action="store_true",
        help="Show ONLY W&B Launch Agent table (monitor command only)",
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        help="Show ONLY Vertex AI jobs table (monitor command only)",
    )
    parser.add_argument(
        "--active",
        action="store_true",
        help="Show ONLY active W&B runs table (monitor command only)",
    )
    parser.add_argument(
        "--completed",
        action="store_true",
        help="Show ONLY completed runs table (monitor command only)",
    )
    parser.add_argument(
        "--builds-active",
        action="store_true",
        help="Show ONLY active Cloud Builds table (monitor command only)",
    )
    parser.add_argument(
        "--builds-recent",
        action="store_true",
        help="Show ONLY recent Cloud Builds table (monitor command only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip launch lock (allows concurrent launches) - launch command only. WARNING: May submit duplicate W&B jobs!",
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup_infrastructure()
    elif args.command == "launch":
        launch_job(force=args.force)
    elif args.command == "monitor":
        monitor_runs(
            runner=args.vertex_runner,
            vertex=args.vertex,
            active=args.active,
            completed=args.completed,
            builds_active=args.builds_active,
            builds_recent=args.builds_recent,
        )
    elif args.command == "teardown":
        teardown_infrastructure()
    elif args.command == "infra":
        check_infrastructure(show_vulns=args.show_vulns)


# show_gpu_info() and show_truffles() removed (2025-11-16)
# GPU/Truffles screens deleted


if __name__ == "__main__":
    main()
