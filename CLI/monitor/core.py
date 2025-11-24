"""Monitor Core Logic - W&B runs and infrastructure monitoring across all 18 MECHA regions"""

# Table size limits (change these to adjust how many items each table shows)
LIMIT_VERTEX_AI_JOBS = 10        # Vertex AI custom jobs table
LIMIT_RUNNER_EXECUTIONS = 5      # Cloud Run runner executions table

# TODO Phase 2: Restore adaptive execution state for terminal caching
# _adaptive_exec_state = {
#     "hot_executions": set(),  # RUNNING/FAILED executions (need frequent log checks)
#     "cold_rotation_idx": 0,   # Rotation index for FINISHED executions
# }
LIMIT_COMPLETED_RUNS = 10        # W&B completed runs table
LIMIT_CLOUD_BUILDS_ACTIVE = 50   # Active Cloud Builds (QUEUED + WORKING only) - high limit to show all
LIMIT_CLOUD_BUILDS_RECENT = 4    # Recent Cloud Builds (all statuses) - compact view

# <claudes_code_comments>
# ** Function List **
# list_runs_core() - Main entry point for listing all runs across 6 tables
# cancel_run_core() - Cancel a specific W&B run by ID
# _list_active_runs() - Get active W&B runs from API
# _list_completed_runs() - Get completed W&B runs from API (limit: LIMIT_COMPLETED_RUNS)
# _format_run_data() - Format run dict for CLI/TUI display
# _format_runtime() - Convert seconds to "Xh Ym" human-readable format
# _format_date() - Extract date from ISO timestamp string
# _list_vertex_ai_jobs() - Get Vertex AI jobs across all 18 regions (limit: LIMIT_VERTEX_AI_JOBS)
# _fetch_runner_executions_all_regions() - Get Cloud Run executions across all 18 regions (limit: LIMIT_RUNNER_EXECUTIONS)
# _list_active_cloud_builds() - Get active Cloud Builds across all 18 regions (limit: LIMIT_CLOUD_BUILDS_ACTIVE)
# _list_recent_cloud_builds() - Get recent Cloud Builds across all 18 regions (limit: LIMIT_CLOUD_BUILDS_RECENT)
#
# Note Column Formatters (companion: core_formatters.py):
# - format_runner_note(raw) - Format runner Note with color-coded states (cyan/green/yellow/red)
# - format_builds_note(raw) - Format builds Note (green success, red error)
# - format_vertex_note(raw) - Format Vertex AI Note (cyan active, green success, red error)
#
# ** Technical Review **
# This module provides infrastructure monitoring across 6 tables: W&B Launch Agent (runner executions),
# Vertex AI Jobs, Active W&B Runs, Completed W&B Runs, Active Cloud Builds, and Recent Cloud Builds.
# All tables use pure chronological ordering (newest first). Table size limits defined as constants at top of file:
# - LIMIT_VERTEX_AI_JOBS=10, LIMIT_RUNNER_EXECUTIONS=10, LIMIT_COMPLETED_RUNS=10
# - LIMIT_CLOUD_BUILDS_ACTIVE=50 (high limit to show all active), LIMIT_CLOUD_BUILDS_RECENT=5
#
# Core flow: list_runs_core() orchestrates 6 parallel fetches → each table queries all 18 MECHA regions
# concurrently using ThreadPoolExecutor → results are combined, sorted by created_at timestamp (newest first),
# and top N returned (per constant). CRITICAL: NO status-priority sorting - pure time-based ordering only!
#
# Runner executions (_fetch_runner_executions_all_regions) implements semi-persistent runner tracking (v2.0):
# Fetches Cloud Run logs (--limit=300) for RUNNING, FINISHED, and FAILED executions to extract:
# 1. Fatal error detection: Searches for "🚨 FATAL ERROR DETECTED" or "❌ Killing agent", extracts
#    100-line context window (20 before + 80 after), parses GCP API errors (machine type, GPU, permissions)
# 2. Jobs run counter: Searches for "Runs: N" to track how many jobs
#    this runner processed before exiting (semi-persistent runners process multiple jobs)
# Stores short error message + jobs_run (for table display) and full error log (for TUI popup).
# Supports GPU accelerator errors ("is not supported for machine type") + all 11 fatal error patterns.
#
# Vertex AI jobs and Cloud Builds use same multi-region pattern: query all 18 regions in parallel,
# combine results, sort by timestamp, return top 10. W&B runs use API ordering (already chronological).
# Constants: 10-item limit per table, 60s gcloud timeout, 300-line log limit for error context.
# </claudes_code_comments>

from typing import Dict, List, Optional
import subprocess
import time
from ..shared.callbacks import StatusCallback
from ..shared.wandb_helper import WandBHelper
from ..shared.api_helpers import run_gcloud_with_retry
from ..shared.log_paths import get_log_path
from .core_formatters import format_runner_note, format_builds_note, format_vertex_note
from ..config.constants import load_training_config


# ═══════════════════════════════════════════════════════════════
# ⭐ PHASE 2: TERMINAL STATE MEMORY (MODULE-LEVEL)
# ═══════════════════════════════════════════════════════════════

# Remember error messages for FAILED executions (terminal state - never change!)
_terminal_failures = {}  # exec_name → error message

# Remember success messages for FINISHED executions (terminal state - never change!)
_terminal_successes = {}  # exec_name → success message (usually "—")

# Last-known-good cache: When gcloud APIs fail, return this instead of empty list!
# Prevents "No executions" flicker when API temporarily fails/times out
_last_known_runner_executions = []  # Last successful runner fetch
_last_known_vertex_jobs = []  # Last successful vertex fetch
_last_known_cloud_builds = []  # Last successful builds fetch


def list_runs_core(
    helper: WandBHelper,
    status: StatusCallback,
    config: dict = None,
    include_completed: bool = True,
) -> Dict[str, List[Dict]]:
    """
    List W&B training runs

    Args:
        helper: WandBHelper instance
        status: Status callback for updates
        config: Training configuration dict (optional)
        include_completed: Whether to include completed runs

    Returns:
        Dict with 'active', 'vertex_jobs', 'runner_executions', 'completed',
        'builds_active', 'builds_recent' run lists.
        Each run is a dict with: id, name, state, runtime, created_at

    Example:
        >>> helper = WandBHelper("entity", "project", "queue")
        >>> status = PrintCallback()
        >>> runs = list_runs_core(helper, status, config)
        >>> print(f"Active: {len(runs['active'])}")
        >>> print(f"Completed: {len(runs['completed'])}")
        >>> print(f"Active Builds: {len(runs['builds_active'])}")
    """
    result = {
        "active": [],
        "vertex_jobs": [],  # Vertex AI jobs
        "runner_executions": [],  # Cloud Run executions (W&B Launch agent)
        "completed": [],
        "builds_active": [],  # Active Cloud Builds (QUEUED + WORKING)
        "builds_recent": []  # Recent Cloud Builds (last 10, all statuses)
    }

    # Get region from config (defaults to us-central1 if not specified)
    region = config.get('GCP_ROOT_RESOURCE_REGION', 'us-central1') if config else 'us-central1'

    try:
        # Fetch Cloud Run executions (W&B Launch agent - shows errors!)
        status("⏳ Checking W&B Launch agent...")
        runner_execs = _fetch_runner_executions_all_regions(status, region)
        result["runner_executions"] = runner_execs

        if len(runner_execs) == 0:
            status("[dim]No runner executions found[/dim]")
        else:
            status(f"[green]✓[/green]  Found {len(runner_execs)} runner execution(s)")

        # Fetch Active Cloud Builds (QUEUED + WORKING only)
        status("⏳ Checking active Cloud Builds...")
        builds_active = _list_active_cloud_builds(status, region)
        result["builds_active"] = builds_active

        if len(builds_active) == 0:
            status("[dim]No active Cloud Builds found[/dim]")
        else:
            status(f"[green]✓[/green]  Found {len(builds_active)} active build(s)")

        # Fetch Recent Cloud Builds (last 10, all statuses)
        status("⏳ Checking recent Cloud Builds...")
        builds_recent = _list_recent_cloud_builds(status, region)
        result["builds_recent"] = builds_recent

        if len(builds_recent) == 0:
            status("[dim]No recent Cloud Builds found[/dim]")
        else:
            status(f"[green]✓[/green]  Found {len(builds_recent)} recent build(s)")

        # Fetch Vertex AI jobs (these show immediately, before W&B runs start)
        status("⏳ Checking Vertex AI jobs...")
        vertex_jobs = _list_vertex_ai_jobs(status, region)
        result["vertex_jobs"] = vertex_jobs

        if len(vertex_jobs) == 0:
            status("[dim]No Vertex AI jobs found[/dim]")
        else:
            status(f"[green]✓[/green]  Found {len(vertex_jobs)} Vertex AI job(s)")

        # Fetch active W&B runs (only appear after job starts training)
        status("⏳ Fetching active runs from W&B...")
        active_runs = _list_active_runs(helper, status)
        result["active"] = active_runs

        if len(active_runs) == 0:
            status("[dim]No active W&B runs found[/dim]")
        else:
            status(f"[green]✓[/green]  Found {len(active_runs)} active run(s)")

        # Fetch completed runs if requested
        if include_completed:
            status("⏳ Fetching completed runs...")
            completed_runs = _list_completed_runs(helper, status)
            result["completed"] = completed_runs

            if len(completed_runs) == 0:
                status("[dim]No completed runs found[/dim]")
            else:
                status(f"[green]✓[/green]  Found {len(completed_runs)} completed run(s)")


        return result

    except ConnectionError:
        status("[red]❌ Cannot connect to W&B API. Check your internet connection.[/red]")
        return result
    except ValueError as e:
        # Project doesn't exist yet
        status("[yellow]ℹ️  W&B project not found. Launch a job to create it.[/yellow]")
        return result
    except Exception as e:
        error_msg = str(e)[:200]
        status(f"[red]❌ Error fetching runs: {error_msg}[/red]")
        return result


def cancel_run_core(
    helper: WandBHelper,
    run_id: str,
    status: StatusCallback,
) -> bool:
    """
    Cancel a W&B training run

    Args:
        helper: WandBHelper instance
        run_id: Run ID to cancel
        status: Status callback for updates

    Returns:
        True if cancellation succeeded
        False if cancellation failed

    Example:
        >>> helper = WandBHelper("entity", "project", "queue")
        >>> status = PrintCallback()
        >>> success = cancel_run_core(helper, "abc123", status)
    """
    try:
        status(f"⏳ Cancelling run: {run_id[:20]}...")

        # Call helper to cancel run
        helper.cancel_run(run_id)

        status(f"[green]✓[/green]  Cancelled run: {run_id[:20]}")
        return True

    except Exception as e:
        error_msg = str(e)[:200]
        status(f"[red]❌ Failed to cancel run: {error_msg}[/red]")
        return False


def _list_active_runs(
    helper: WandBHelper,
    status: StatusCallback,
) -> List[Dict]:
    """
    Get active runs from W&B

    Extracted from: screen.py lines 237-282

    Args:
        helper: WandBHelper instance
        status: Status callback

    Returns:
        List of active run dicts
    """
    try:
        runs = helper.get_active_runs()

        # Format each run for display
        formatted_runs = []
        for run in runs:
            formatted = _format_run_data(run)
            formatted_runs.append(formatted)

        return formatted_runs

    except Exception as e:
        raise  # Re-raise for outer handler


def _list_completed_runs(
    helper: WandBHelper,
    status: StatusCallback,
    limit: int = LIMIT_COMPLETED_RUNS,
) -> List[Dict]:
    """
    Get completed runs from W&B

    Extracted from: screen.py lines 319-363

    Args:
        helper: WandBHelper instance
        status: Status callback
        limit: Max number of runs to return

    Returns:
        List of completed run dicts
    """
    try:
        runs = helper.get_completed_runs(limit=limit)

        # Format each run for display
        formatted_runs = []
        for run in runs:
            formatted = _format_run_data(run)
            formatted_runs.append(formatted)

        return formatted_runs

    except Exception:
        # Fail silently - completed runs are supplementary
        return []


def _format_run_data(run: Dict) -> Dict:
    """
    Format run data for display

    Extracted from: screen.py lines 252-281, 334-362

    Args:
        run: Raw run dict from WandBHelper

    Returns:
        Formatted run dict with display-friendly strings
    """
    # Get runtime
    runtime_seconds = run.get('runtime', 0)
    runtime_str = _format_runtime(runtime_seconds)

    # Color-code runtime based on duration
    if runtime_seconds < 3600:  # < 1 hour - normal
        runtime_display = f"[green]{runtime_str}[/green]"
    elif runtime_seconds < 14400:  # 1-4 hours - getting long
        runtime_display = runtime_str
    elif runtime_seconds < 28800:  # 4-8 hours - quite long
        runtime_display = f"[yellow]{runtime_str}[/yellow]"
    else:  # > 8 hours - possibly stuck?
        runtime_display = f"[red]{runtime_str}[/red]"

    # Format created_at date (dimmed)
    created_str = _format_date(run.get('created_at', 'Unknown'))
    created_display = f"[dim]{created_str}[/dim]"

    # Color-code state with icons for display
    state = run.get('state', 'unknown')
    if state == "running":
        state_display = f"[green]▶ running[/green]"
    elif state == "pending":
        state_display = f"[yellow]⏳ pending[/yellow]"
    elif state == "queued":
        state_display = f"[blue]◆ queued[/blue]"
    elif state == "leased":
        state_display = f"[green]◎ leased[/green]"
    elif state == "preempting":
        state_display = f"[orange]⚠ preempting[/orange]"
    elif state == "retrying":
        state_display = f"[magenta]↻ retrying[/magenta]"
    elif state == "finished":
        state_display = f"[green]✓ finished[/green]"
    elif state == "failed":
        state_display = f"[bold red]✗ failed[/bold red]"
    elif state == "crashed":
        state_display = f"[bold red]💥 crashed[/bold red]"
    elif state == "killed":
        state_display = f"[yellow]⊗ killed[/yellow]"
    else:
        state_display = state

    return {
        "id": run.get('id', ''),
        "name": run.get('name', 'Unknown'),
        "state": state,  # Raw state
        "state_display": state_display,  # Colored state for display
        "runtime": runtime_seconds,  # Raw runtime in seconds
        "runtime_display": runtime_display,  # Color-coded runtime
        "created_at": run.get('created_at', 'Unknown'),  # Raw timestamp
        "created_display": created_display,  # Dimmed date
    }


def _format_runtime(runtime_seconds: int) -> str:
    """
    Format runtime from seconds to "Xh Ym Zs" (always show seconds)

    Extracted from: screen.py line 252, 334

    Args:
        runtime_seconds: Runtime in seconds

    Returns:
        Formatted string like "1h 23m 45s" or "2m 30s" or "15s" or "—"

    Examples:
        >>> _format_runtime(5445)
        "1h 30m 45s"
        >>> _format_runtime(150)
        "2m 30s"
        >>> _format_runtime(15)
        "15s"
        >>> _format_runtime(0)
        "—"
    """
    if runtime_seconds <= 0:
        return "—"

    hours = runtime_seconds // 3600
    minutes = (runtime_seconds % 3600) // 60
    seconds = runtime_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _format_date(created_at: str) -> str:
    """
    Format ISO timestamp to human-friendly datetime string (down to the second)

    Extracted from: screen.py lines 255-259, 337-341

    Args:
        created_at: ISO timestamp like "2025-11-16T06:21:48+00:00" or "Unknown"

    Returns:
        DateTime string like "Nov 16, 10:21:48 PM" or "Unknown"

    Examples:
        >>> _format_date("2025-11-16T06:21:48+00:00")
        "Nov 16, 10:21:48 PM"
        >>> _format_date("Unknown")
        "Unknown"
    """
    if not created_at or created_at == 'Unknown':
        return "Unknown"

    try:
        from datetime import datetime
        # Parse ISO timestamp
        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        # Format: "Nov 16, 10:21:48 PM" (%-I strips leading zero: 7 not 07)
        return dt.strftime("%b %d, %-I:%M:%S %p")
    except Exception:
        # Fallback: extract date part from ISO timestamp
        if 'T' in created_at:
            return created_at.split('T')[0]
        return created_at


def _list_vertex_ai_jobs(status: StatusCallback, region: str = "us-central1", target_regions: List[str] = None) -> List[Dict]:
    """
    Get Vertex AI custom jobs from specified or all MECHA regions (adaptive monitoring)

    Args:
        region: Legacy parameter (ignored)
        target_regions: Optional list of specific regions to query. If None, queries all 18 MECHA regions.

    Returns:
        List of Vertex AI job dicts with id, name, state, created_at, region
    """
    try:
        import json
        from datetime import datetime, timezone
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ALL 18 MECHA regions (default)
        ALL_MECHA_REGIONS = [
            "us-central1", "us-east1", "us-east4", "us-east5",
            "us-west1", "us-west2", "us-west3", "us-west4",
            "northamerica-northeast1",
            "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9",
            "asia-northeast1", "asia-southeast1",
            "australia-southeast1",
            "southamerica-east1"
        ]

        # Use target_regions if provided (adaptive monitoring), otherwise all 18
        MECHA_REGIONS = target_regions if target_regions is not None else ALL_MECHA_REGIONS

        def fetch_region(region_name: str) -> List[Dict]:
            """Fetch Vertex AI jobs from a single region"""
            try:
                # List Vertex AI jobs from last 7 days
                result = run_gcloud_with_retry(
                    [
                        "gcloud", "ai", "custom-jobs", "list",
                        f"--region={region_name}",
                        f"--limit={LIMIT_VERTEX_AI_JOBS}",
                        "--format=json",
                        "--filter=createTime>-P7D",  # Last 7 days
                    ],
                    max_retries=3,
                    timeout=60,
                    operation_name="list Vertex AI custom jobs",
                )

                if result.returncode != 0:
                    # Check for billing error
                    if result.stderr and "BILLING_DISABLED" in result.stderr:
                        # Return special marker for billing error
                        return [{"_billing_error": True, "region": region_name}]
                    # Other errors - likely not configured
                    return []

                jobs = json.loads(result.stdout) if result.stdout else []

                # Format jobs for display
                formatted_jobs = []
                for job in jobs:
                    job_id = job.get('name', '').split('/')[-1]  # Extract numeric ID
                    state = job.get('state', 'UNKNOWN')
                    created = job.get('createTime', 'Unknown')
                    start_time = job.get('startTime')
                    end_time = job.get('endTime')

                    # Calculate runtime (like W&B runs)
                    runtime_seconds = 0
                    if start_time and end_time:
                        # Job finished - calculate total runtime
                        try:
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                            runtime_seconds = int((end - start).total_seconds())
                        except Exception:
                            runtime_seconds = 0
                    elif start_time and state == "JOB_STATE_RUNNING":
                        # Job still running - calculate current runtime
                        try:
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            now = datetime.now(timezone.utc)
                            runtime_seconds = int((now - start).total_seconds())
                        except Exception:
                            runtime_seconds = 0

                    # Color-code Vertex AI states with icons
                    if state == "JOB_STATE_SUCCEEDED":
                        state_display = f"[green]✓ SUCCEEDED[/green]"
                    elif state == "JOB_STATE_FAILED":
                        state_display = f"[bold red]✗ FAILED[/bold red]"
                    elif state == "JOB_STATE_RUNNING":
                        state_display = f"[green]▶ RUNNING[/green]"
                    elif state == "JOB_STATE_PENDING":
                        state_display = f"[yellow]⏳ PENDING[/yellow]"
                    elif state == "JOB_STATE_QUEUED":
                        state_display = f"[blue]◆ QUEUED[/blue]"
                    elif state == "JOB_STATE_CANCELLING":
                        state_display = f"[yellow]⊗ CANCELLING[/yellow]"
                    elif state == "JOB_STATE_CANCELLED":
                        state_display = f"[dim]⊗ CANCELLED[/dim]"
                    else:
                        state_display = state

                    # Extract error message if job failed
                    error_msg = None
                    if state == "JOB_STATE_FAILED":
                        error_obj = job.get('error', {})
                        if error_obj:
                            error_msg = error_obj.get('message', '')
                            # Also check for code
                            code = error_obj.get('code')
                            if code and not error_msg:
                                error_msg = f"Error code: {code}"

                    formatted_jobs.append({
                        "id": job_id,
                        "name": f"vertex-job-{job_id[:8]}",  # Shortened name
                        "state": state,
                        "state_display": state_display,
                        "runtime": runtime_seconds,  # Raw runtime in seconds
                        "runtime_display": _format_runtime(runtime_seconds),  # Formatted "Xh Ym"
                        "created_at": created,
                        "created_display": _format_date(created),
                        "start_time": start_time,  # For ticker updates (RUNNING jobs)
                        "created": created,  # For ticker updates (PENDING/QUEUED jobs)
                        "error": format_vertex_note(error_msg),  # Pre-formatted for display!
                        "region": region_name  # Add region to each job!
                    })

                return formatted_jobs
            except Exception:
                return []  # Fail silently for this region

        # Query all 18 regions in parallel!
        all_jobs = []
        with ThreadPoolExecutor(max_workers=18) as executor:
            future_to_region = {executor.submit(fetch_region, region): region for region in MECHA_REGIONS}
            for future in as_completed(future_to_region):
                region_jobs = future.result()
                all_jobs.extend(region_jobs)

        # Sort by created time ONLY (newest first) - pure chronological order
        all_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # Last-known-good cache: Return cached data if API fails
        global _last_known_vertex_jobs
        top_jobs = all_jobs[:LIMIT_VERTEX_AI_JOBS]

        if len(top_jobs) > 0:
            _last_known_vertex_jobs = top_jobs
            return top_jobs
        else:
            # API returned 0 results - return last known good!
            return _last_known_vertex_jobs

    except Exception:
        # Exception during fetch - return last known good data
        return _last_known_vertex_jobs


# ═══════════════════════════════════════════════════════════════
# ⭐ PHASE 2: ERROR EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════

def _fetch_and_extract_success(exec_name: str) -> tuple[str, int]:
    """
    Fetch logs for a FINISHED execution and extract job count.

    ⚠️ RELIES ON: arr-vertex-launcher wrapper printing "Runs: N" pattern!
    Source: Stack/arr-vertex-launcher/entrypoint-wrapper.sh

    The wrapper script prints "Runs: N" at these points:
    - Line 36: echo "📊 Runs: 0" (initial startup)
    - Line 59: echo "   • Runs: $JOBS_RUN" (final stats)
    - Line 111: echo "✅ Job submitted to Vertex AI! (Runs: $JOBS_RUN)"
    - Line 128: echo "   • Runs: $JOBS_RUN" (bailout stats)
    - Line 139: echo "... Runs: $JOBS_RUN" (alive heartbeat)

    We fetch Cloud Run logs and search for "Runs: N", taking the MAX value
    (wrapper increments this counter each time it submits a job).

    Returns:
        Tuple of (success_message, jobs_run_count)
        Example: ("✓ Completed: 5 jobs", 5)
    """
    try:
        log_result = run_gcloud_with_retry(
            [
                "gcloud", "logging", "read",
                f'resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher AND labels."run.googleapis.com/execution_name"={exec_name}',
                "--limit=300",
                "--format=value(textPayload)",
                f"--project={load_training_config().get('GCP_PROJECT_ID', '')}",
            ],
            max_retries=1,
            timeout=10,
            operation_name="fetch Cloud Run job logs",
        )

        if log_result.returncode != 0 or not log_result.stdout:
            return ("✓ Completed", 0)

        lines = log_result.stdout.strip().split('\n')

        # Extract highest "Runs: N" value (wrapper increments this)
        max_runs = 0
        for line in lines:
            if 'Runs:' in line:
                # Extract number after "Runs:"
                try:
                    runs_str = line.split('Runs:')[1].strip().split()[0]
                    runs_count = int(runs_str)
                    max_runs = max(max_runs, runs_count)
                except (IndexError, ValueError):
                    continue

        # Format success message
        if max_runs > 0:
            return (f"✓ Completed: {max_runs} job{'s' if max_runs != 1 else ''}", max_runs)
        else:
            return ("✓ Completed", 0)

    except Exception:
        return ("✓ Completed", 0)


def _fetch_and_extract_error(exec_name: str) -> str:
    """
    Fetch logs for a FAILED execution and extract concise error message.

    ⚠️ RELIES ON: arr-vertex-launcher wrapper printing "🚨 FATAL ERROR DETECTED"!
    Source: Stack/arr-vertex-launcher/entrypoint-wrapper.sh

    The wrapper script prints bailout messages when detecting fatal errors:
    - Line 144, 153, 162, 170, 180, 192, 202, 212, 228, 238: echo "🚨 FATAL ERROR DETECTED: ..."

    We search for these wrapper markers, then extract the REAL GCP/W&B/Python errors
    from surrounding context (100-line window: 20 before + 80 after).

    Uses ALL our complex error matching patterns (20+ patterns):
    - Wrapper bailout detection (🚨 FATAL ERROR, ❌ Killing agent)
    - Machine type errors
    - Permission errors (InvalidArgument, PermissionDenied, NotFound)
    - Quota errors (QuotaExceeded, ResourceExhausted)
    - Service errors (500, 503)
    - HTTP errors (400-503)
    - Container errors (ImagePullBackOff, ErrImagePull)
    - Python exceptions (Traceback, Exception, Error)
    - W&B agent errors (wandb: ERROR)
    - Info filtering (⏱️ ⏳ ℹ️ 🔍, "monitoring for")

    Args:
        exec_name: Cloud Run execution name (e.g., 'vertex-ai-launcher-f4hfv-0001')

    Returns:
        Concise error message for table display (max 200 chars)

    Example:
        >>> _fetch_and_extract_error('vertex-ai-launcher-f4hfv-0001')
        'QuotaExceeded: Quota NVIDIA_L4_GPUS exceeded. Limit: 0 in us-west2'
    """
    try:
        # ───────────────────────────────────────────────────────────
        # Step 1: Fetch logs from Cloud Logging
        # ───────────────────────────────────────────────────────────
        log_result = run_gcloud_with_retry(
            [
                "gcloud", "logging", "read",
                f'resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher AND labels."run.googleapis.com/execution_name"={exec_name}',
                "--limit=300",
                "--format=value(textPayload)",
                f"--project={load_training_config().get('GCP_PROJECT_ID', '')}",
            ],
            max_retries=1,
            timeout=10,
            operation_name="fetch Cloud Run job logs",
        )

        if log_result.returncode != 0 or not log_result.stdout:
            return "Error fetching logs"

        lines = log_result.stdout.strip().split('\n')

        # ───────────────────────────────────────────────────────────
        # Step 2: Extract error using ALL our complex patterns!
        # ───────────────────────────────────────────────────────────

        # ⭐ Priority 1: Wrapper bailout detection
        # ✅ MATCHING OUR OWN STUFF HERE: entrypoint-wrapper.sh prints these!
        # Source: Stack/arr-vertex-launcher/entrypoint-wrapper.sh lines 144-240
        # What: Wrapper detects fatal errors and prints bailout messages
        # Why: Fast bailout - we capture context around these markers
        for i, line in enumerate(lines):
            if '🚨 FATAL ERROR DETECTED' in line or '❌ Killing agent' in line:
                # Capture context (20 before + 80 after)
                start_idx = max(0, i - 20)
                end_idx = min(len(lines), i + 80)
                bailout_lines = lines[start_idx:end_idx]

                # Search for specific errors in context
                # ❌ NOT our output - REAL GCP/W&B/Python errors!
                for ctx_line in bailout_lines:
                    # ❌ GCP: Machine type incompatibility
                    # Source: GCP API (external)
                    # Example: "InvalidArgument: 400 Machine type 'n2-standard-4' is not supported for GPU 'NVIDIA_TESLA_T4'"
                    if 'Machine type' in ctx_line and 'is not supported' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Machine type (alternate phrasing)
                    # Source: GCP API (external)
                    # Example: "GPU 'NVIDIA_L4' is not supported for machine type 'n1-standard-4'"
                    elif 'is not supported for machine type' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Invalid args, permissions, missing resources
                    # Source: GCP API (external)
                    # Examples: "InvalidArgument: 400", "PermissionDenied: 403", "NotFound: 404"
                    elif 'InvalidArgument:' in ctx_line or 'PermissionDenied:' in ctx_line or 'NotFound:' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Quota exceeded
                    # Source: GCP Quota System (external)
                    # Example: "QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0"
                    elif 'QuotaExceeded' in ctx_line or 'ResourceExhausted' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Service errors (500/503)
                    # Source: GCP APIs (external)
                    # Examples: "503 Service Unavailable", "500 Internal Error"
                    elif any(pattern in ctx_line for pattern in ['503', 'ServiceUnavailable', '500', 'Internal Error', 'Internal error']):
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: HTTP error codes
                    # Source: GCP APIs (external)
                    # Example: "HttpError: <HttpError 429 ... 'Too Many Requests'>"
                    elif 'HttpError' in ctx_line and any(code in ctx_line for code in ['400', '401', '403', '404', '429', '500', '502', '503']):
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ K8s/GCP: Image pull failures
                    # Source: Kubernetes/GCP (external)
                    # Examples: "ImagePullBackOff", "ErrImagePull"
                    elif 'ImagePullBackOff' in ctx_line or 'ErrImagePull' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ Python: Exceptions
                    # Source: Python interpreter (external)
                    # Example: "Traceback (most recent call last)..."
                    elif 'Traceback' in ctx_line or 'Exception:' in ctx_line or 'Error:' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                # Fallback: return wrapper's generic message
                # ✅ Our wrapper message if no specific error found
                return line.replace('🚨 FATAL ERROR DETECTED:', '').replace('❌', '').strip()

        # ⭐ Priority 2: W&B agent errors
        # ❌ NOT our output - W&B Launch agent's error prefix!
        # Source: W&B Launch agent (external)
        # What: W&B prefixes errors with "wandb: ERROR", often re-logging GCP errors
        for i, line in enumerate(lines):
            if 'wandb: ERROR' in line:
                if 'Machine type' in line and 'is not supported' in line:
                    return line.split('wandb: ERROR')[-1].strip()
                elif 'InvalidArgument' in line or 'PermissionDenied' in line or 'NotFound' in line:
                    return line.split('wandb: ERROR')[-1].strip()

        # ⭐ Priority 3: Generic error patterns (skip INFO messages)
        for line in lines:
            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'notfound', '404', 'timeout']):
                # ✅ MATCHING OUR OWN STUFF HERE: Skip our info emojis!
                # Source: Stack/arr-vertex-launcher/entrypoint-wrapper.sh lines 35, 51
                # What: We print "⏱️ Idle timeout...", "⏳ Monitoring for..." - NOT errors!
                # Why: Filter false positives
                if any(info_indicator in line for info_indicator in ['⏱️', '⏳', 'ℹ️', '🔍']):
                    continue

                # ✅ MATCHING OUR OWN STUFF HERE: Skip our status messages!
                # Source: Stack/arr-vertex-launcher/entrypoint-wrapper.sh line 51
                # What: "⏳ Monitoring for fatal errors..." - NOT an error!
                # Why: Filter false positives
                if any(info_pattern in line.lower() for info_pattern in ['monitoring for', 'checking for', 'watching for', 'looking for']):
                    continue

                # ❌ Generic error from logs
                return line[:200]  # Truncate for table display

        # No error found
        return "Unknown error"

    except Exception as e:
        return f"Error parsing logs: {str(e)}"


def _fetch_runner_executions_all_regions(status: StatusCallback, region: str = "us-central1", target_regions: List[str] = None) -> List[Dict]:
    """
    Get Cloud Run job executions from specified or all MECHA regions (simplified - Phase 1)

    Shows runner status from W&B Launch → Vertex AI submission.

    Phase 1: Basic metadata only, displays "Fetching logs..." placeholder
    Phase 2: Add terminal state caching for smart log fetching

    Original adaptive HOT/COLD code preserved below (commented out) for Phase 2 implementation.

    Args:
        region: Legacy parameter (ignored)
        target_regions: Optional list of specific regions to query. If None, queries all 18 MECHA regions.

    Returns:
        List of execution dicts with name, status, region
    """
    try:
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ALL 18 MECHA regions (default)
        ALL_MECHA_REGIONS = [
            "us-central1", "us-east1", "us-east4", "us-east5",
            "us-west1", "us-west2", "us-west3", "us-west4",
            "northamerica-northeast1",
            "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9",
            "asia-northeast1", "asia-southeast1",
            "australia-southeast1",
            "southamerica-east1"
        ]

        # Use target_regions if provided (adaptive monitoring), otherwise all 18
        MECHA_REGIONS = target_regions if target_regions is not None else ALL_MECHA_REGIONS

        def fetch_region(region_name: str) -> List[Dict]:
            """Fetch runner executions from a single region (simplified - Phase 1)"""
            try:
                import time
                from pathlib import Path
                from datetime import datetime

                region_start = time.time()

                # List recent Cloud Run executions (basic metadata only!)
                gcloud_start = time.time()
                result = run_gcloud_with_retry(
                    [
                        "gcloud", "run", "jobs", "executions", "list",
                        "--job=vertex-ai-launcher",
                        f"--region={region_name}",
                        f"--limit={LIMIT_RUNNER_EXECUTIONS}",
                        "--format=json",
                    ],
                    max_retries=1,
                    timeout=30,
                    operation_name="list Cloud Run job executions",
                )
                gcloud_time = time.time() - gcloud_start

                if result.returncode != 0:
                    return []

                # 🦡 JSON PARSING (holds GIL!)
                json_start = time.time()
                executions = json.loads(result.stdout) if result.stdout else []
                json_time = time.time() - json_start

                # 🦡 YIELD after JSON parsing to let spinners breathe!
                time.sleep(0.005)  # 5ms yield after heavy GIL work

                # ═══════════════════════════════════════════════════════════════
                # TODO Phase 2: Restore HOT/COLD adaptive log fetching
                # ═══════════════════════════════════════════════════════════════
                # exec_statuses = {}
                # for execution in executions:
                #     name = execution.get('metadata', {}).get('name', '').split('/')[-1]
                #     conditions = execution.get('status', {}).get('conditions', [])
                #     status_str = "UNKNOWN"
                #     for condition in conditions:
                #         if condition.get('type') == 'Completed':
                #             if condition.get('status') == 'True':
                #                 status_str = "FINISHED"
                #             elif condition.get('status') == 'False':
                #                 status_str = "FAILED"
                #             elif condition.get('status') == 'Unknown':
                #                 status_str = "RUNNING"
                #     exec_statuses[name] = status_str
                #
                # hot_execs = {name for name, status in exec_statuses.items() if status in ["RUNNING", "FAILED"]}
                # cold_execs = [name for name, status in exec_statuses.items() if status == "FINISHED"]
                #
                # target_executions_set = set(hot_execs)
                # if cold_execs:
                #     for i in range(min(2, len(cold_execs))):
                #         idx = (_adaptive_exec_state["cold_rotation_idx"] + i) % len(cold_execs)
                #         target_executions_set.add(cold_execs[idx])
                #     _adaptive_exec_state["cold_rotation_idx"] = (_adaptive_exec_state["cold_rotation_idx"] + 2) % len(cold_execs)
                #
                # _adaptive_exec_state["hot_executions"] = hot_execs
                #
                # ═══════════════════════════════════════════════════════════════
                # TODO Phase 2: Restore parallel log fetching
                # ═══════════════════════════════════════════════════════════════
                # execution_logs = {}
                #
                # def fetch_execution_logs(execution) -> tuple:
                #     name = execution.get('metadata', {}).get('name', '').split('/')[-1]
                #     if name not in target_executions_set:
                #         return (name, None)
                #     try:
                #         log_result = run_gcloud_with_retry(
                #             [
                #                 "gcloud", "logging", "read",
                #                 f'resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher AND labels."run.googleapis.com/execution_name"={name}',
                #                 "--limit=300",
                #                 "--format=value(textPayload)",
                #                 f"--project={load_training_config().get('GCP_PROJECT_ID', '')}",
                #             ],
                #             max_retries=1,
                #             timeout=10,
                #             operation_name="fetch Cloud Run job logs",
                #         )
                #         if log_result.returncode == 0 and log_result.stdout:
                #             lines = log_result.stdout.strip().split('\n')
                #             return (name, lines)
                #     except Exception:
                #         pass
                #     return (name, None)
                #
                # execs_to_fetch = [e for e in executions if e.get('metadata', {}).get('name', '').split('/')[-1] in target_executions_set]
                # if execs_to_fetch:
                #     with ThreadPoolExecutor(max_workers=min(10, len(execs_to_fetch))) as executor:
                #         futures = {executor.submit(fetch_execution_logs, exec): exec for exec in execs_to_fetch}
                #         for future in as_completed(futures):
                #             exec_name, lines = future.result()
                #             if lines:
                #                 execution_logs[exec_name] = lines

                # ═══════════════════════════════════════════════════════════════
                # FORMAT EXECUTIONS (Simplified - Phase 1)
                # ═══════════════════════════════════════════════════════════════
                format_start = time.time()
                formatted_execs = []
                for execution in executions:
                    # Extract basic metadata
                    name = execution.get('metadata', {}).get('name', '').split('/')[-1]
                    conditions = execution.get('status', {}).get('conditions', [])

                    # Extract status from conditions
                    status_str = "UNKNOWN"
                    for condition in conditions:
                        if condition.get('type') == 'Completed':
                            if condition.get('status') == 'True':
                                status_str = "FINISHED"
                            elif condition.get('status') == 'False':
                                status_str = "FAILED"
                            elif condition.get('status') == 'Unknown':
                                status_str = "RUNNING"

                    # Extract queue name from container args
                    queue_name = "—"
                    spec = execution.get('spec', {})
                    if spec:
                        template = spec.get('template', {})
                        if template:
                            template_spec = template.get('spec', {})
                            if template_spec:
                                containers = template_spec.get('containers', [])
                                if containers and len(containers) > 0:
                                    args = containers[0].get('args', [])
                                    if args:
                                        for i, arg in enumerate(args):
                                            if arg == "-q" and i + 1 < len(args):
                                                queue_name = args[i + 1]
                                                break

                    # Simple status display (no nuanced coloring - Phase 1)
                    if status_str == "FINISHED":
                        status_display = f"[green]✓ {status_str}[/green]"
                    elif status_str == "FAILED":
                        status_display = f"[bold red]✗ {status_str}[/bold red]"
                    elif status_str == "RUNNING":
                        status_display = f"[green]▶ {status_str}[/green]"
                    else:
                        status_display = status_str

                    # Calculate duration
                    start_time = execution.get('status', {}).get('startTime')
                    completion_time = execution.get('status', {}).get('completionTime')
                    duration_display = "—"
                    if start_time:
                        try:
                            from datetime import datetime, timezone
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            if completion_time:
                                end = datetime.fromisoformat(completion_time.replace('Z', '+00:00'))
                            else:
                                end = datetime.now(timezone.utc)
                            duration_seconds = (end - start).total_seconds()
                            if duration_seconds < 60:
                                duration_display = f"{int(duration_seconds)}s"
                            else:
                                minutes = int(duration_seconds / 60)
                                seconds = int(duration_seconds % 60)
                                duration_display = f"{minutes}m{seconds}s"
                        except Exception:
                            duration_display = "—"

                    # Phase 1: Show "Fetching logs..." placeholder for all executions
                    formatted_execs.append({
                        "name": name,
                        "queue_name": queue_name,
                        "status": status_str,
                        "status_display": status_display,
                        "start_time": start_time,
                        "duration": duration_display,
                        "jobs_run": "—",  # Phase 2: Extract from logs
                        "error": "[dim]Fetching logs...[/dim]",  # Phase 1: Placeholder (pre-formatted)
                        "full_error_log": None,  # Phase 2: Extract from logs
                        "created_at": execution.get('status', {}).get('startTime', 'Unknown'),
                        "created_display": _format_date(execution.get('status', {}).get('startTime', 'Unknown')),
                        "region": region_name
                    })

                format_time = time.time() - format_start

                # 🦡 YIELD after formatting to let spinners breathe!
                time.sleep(0.005)  # 5ms yield after heavy GIL work

                # 📊 SMART TIMING LOG - Identify slow regions!
                total_time = time.time() - region_start
                gil_hold_time = json_time + format_time  # Time spent holding GIL (no I/O)

                # Log to spinner timing file so we can correlate with FPS drops
                timing_log = get_log_path("region_timing.log")
                with open(timing_log, "a") as f:
                    f.write(f"{datetime.now().isoformat()} 🌍 {region_name}: "
                           f"total={total_time*1000:.0f}ms, "
                           f"gcloud={gcloud_time*1000:.0f}ms (I/O), "
                           f"json={json_time*1000:.1f}ms (GIL), "
                           f"format={format_time*1000:.1f}ms (GIL), "
                           f"⚠️ GIL_HOLD={gil_hold_time*1000:.1f}ms, "
                           f"execs={len(executions)}\n")

                return formatted_execs
            except Exception:
                return []  # Fail silently for this region

        # Query all 18 regions in parallel!
        all_execs = []
        with ThreadPoolExecutor(max_workers=18) as executor:
            future_to_region = {executor.submit(fetch_region, region): region for region in MECHA_REGIONS}
            for future in as_completed(future_to_region):
                region_execs = future.result()
                all_execs.extend(region_execs)

        # 🔍 DEBUG: Log fetch results
        from pathlib import Path
        from datetime import datetime
        debug_log = get_log_path("runner_fetch_debug.log")
        with open(debug_log, "a") as f:
            f.write(f"\n{datetime.now().isoformat()} 🌍 FETCHED {len(all_execs)} total executions from {len(MECHA_REGIONS)} regions\n")
            for exec in all_execs:
                f.write(f"  - {exec['name']} ({exec['region']}, {exec['status']}, {exec['created_at']})\n")

        # Sort by created time ONLY (newest first) - pure chronological order
        all_execs.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # 🔍 DEBUG: Log sorted results
        with open(debug_log, "a") as f:
            f.write(f"{datetime.now().isoformat()} 🔢 SORTED (newest first):\n")
            for i, exec in enumerate(all_execs[:10]):  # Show top 10
                f.write(f"  {i+1}. {exec['name']} ({exec['region']}, {exec['created_at']})\n")
            f.write(f"{datetime.now().isoformat()} ✂️ RETURNING top {LIMIT_RUNNER_EXECUTIONS}\n\n")

        # ═══════════════════════════════════════════════════════════════
        # ⭐ PHASE 2C: SMART LOG FETCHING WITH TERMINAL STATE MEMORY
        # ═══════════════════════════════════════════════════════════════
        # Only fetch logs for NEW FAILED executions!
        # Remember terminal states (FAILED & FINISHED) forever!
        # RUNNING executions: show "Running..." (check again next refresh)

        top_execs = all_execs[:LIMIT_RUNNER_EXECUTIONS]

        for exec in top_execs:
            exec_name = exec['name']
            status = exec['status']

            if status == "FAILED":
                # Terminal FAILED! Check if we already fetched error
                if exec_name in _terminal_failures:
                    # Already fetched before - use remembered error!
                    exec['error'] = _terminal_failures[exec_name]
                else:
                    # New FAILED execution - fetch logs ONCE and remember!
                    error_msg = _fetch_and_extract_error(exec_name)
                    formatted_error = format_runner_note(error_msg)  # Pre-format for display!
                    _terminal_failures[exec_name] = formatted_error  # Remember forever!
                    exec['error'] = formatted_error

            elif status == "FINISHED":
                # Terminal FINISHED! Remember success message
                if exec_name in _terminal_successes:
                    # Already know it succeeded - use remembered message!
                    cached_msg, cached_jobs = _terminal_successes[exec_name]
                    exec['error'] = cached_msg
                    exec['jobs_run'] = str(cached_jobs) if cached_jobs > 0 else "—"
                else:
                    # New FINISHED execution - fetch logs ONCE to extract job count!
                    success_msg, jobs_count = _fetch_and_extract_success(exec_name)
                    formatted_success = format_runner_note(success_msg)  # Pre-format for display!
                    _terminal_successes[exec_name] = (formatted_success, jobs_count)  # Remember forever!
                    exec['error'] = formatted_success
                    exec['jobs_run'] = str(jobs_count) if jobs_count > 0 else "—"

            elif status == "RUNNING":
                # NOT terminal! Keep showing "Running..." and poll metadata again next refresh
                exec['error'] = format_runner_note("Running...")  # Pre-format cyan ⏳

            # Any other status (PENDING, CANCELLED, etc.)
            else:
                exec['error'] = format_runner_note("—")

        # ═══════════════════════════════════════════════════════════════
        # LAST-KNOWN-GOOD CACHE: Save successful results, return cached on failure!
        # ═══════════════════════════════════════════════════════════════
        global _last_known_runner_executions

        if len(top_execs) > 0:
            # Success! Cache these results
            _last_known_runner_executions = top_execs
            return top_execs
        else:
            # API returned 0 results (timeout/failure) - return last known good!
            return _last_known_runner_executions

    except Exception:
        # Exception during fetch - return last known good data
        return _last_known_runner_executions


# REMOVED: _get_latest_image_digest() function
#
# Previously tried to detect stale scans by comparing :latest tag digest
# vs newest-by-upload-time digest. This caused false positives because:
# - gcloud describe :latest (scan data source) != gcloud list --sort-by=~UPDATE_TIME
# - Tag assignments can race with upload times
# - After teardown/rebuild, digests mismatched despite being current
#
# We now trust GCP scan results if they exist. The :latest tag is authoritative.


def _list_active_cloud_builds(status: StatusCallback, region: str = "us-central1", target_regions: List[str] = None) -> List[Dict]:
    """
    Get active Cloud Builds from specified or all MECHA regions (adaptive monitoring)

    Shows builds currently in progress (QUEUED + WORKING) - most important for real-time monitoring.

    Args:
        region: Legacy parameter (ignored)
        status: Status callback for messages
        target_regions: Optional list of specific regions to query. If None, queries all 18 MECHA regions.

    Returns:
        List of active build dicts with id, image_name, status, duration, step, created, region
    """
    try:
        import json
        from datetime import datetime, timezone
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ALL 18 MECHA regions (default)
        ALL_MECHA_REGIONS = [
            "us-central1", "us-east1", "us-east4", "us-east5",
            "us-west1", "us-west2", "us-west3", "us-west4",
            "northamerica-northeast1",
            "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9",
            "asia-northeast1", "asia-southeast1",
            "australia-southeast1",
            "southamerica-east1"
        ]

        # Use target_regions if provided (adaptive monitoring), otherwise all 18
        MECHA_REGIONS = target_regions if target_regions is not None else ALL_MECHA_REGIONS

        def fetch_region(region_name: str) -> List[Dict]:
            """Fetch active builds from a single region"""
            try:
                result = run_gcloud_with_retry([
                    "gcloud", "builds", "list",
                    "--ongoing",  # Only QUEUED + WORKING
                    f"--region={region_name}",
                    f"--limit={LIMIT_CLOUD_BUILDS_ACTIVE}",
                    "--format=json"
                ], max_retries=3, timeout=60, operation_name="list active Cloud Builds")

                if result.returncode != 0:
                    return []

                builds = json.loads(result.stdout) if result.stdout else []

                formatted = []
                for build in builds:
                    build_id = build.get('id', '')[:12]  # First 12 chars
                    status_val = build.get('status', 'UNKNOWN')
                    created = build.get('createTime', '')

                    # Extract image name from images list
                    images = build.get('images', [])
                    image_name = "unknown"
                    if images:
                        full_image = images[0]
                        # Extract just the image name (e.g., "arr-training")
                        if '/' in full_image:
                            image_name = full_image.split('/')[-1].split(':')[0]

                    # Calculate duration (createTime to now for QUEUED/WORKING)
                    runtime_seconds = 0
                    if created:
                        try:
                            start = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            now = datetime.now(timezone.utc)
                            runtime_seconds = int((now - start).total_seconds())
                        except Exception:
                            pass

                    # Calculate step progress (steps completed / total steps)
                    steps = build.get('steps', [])
                    total_steps = len(steps)
                    completed_steps = sum(1 for s in steps if s.get('status') in ['SUCCESS', 'FAILURE'])
                    step_progress = f"{completed_steps}/{total_steps}" if total_steps > 0 else "0/0"

                    # Color-code active build status with icons
                    if status_val == "WORKING":
                        status_display = f"[green]▶ WORKING[/green]"
                    elif status_val == "QUEUED":
                        status_display = f"[blue]◆ QUEUED[/blue]"  # Blue to match all other QUEUED statuses
                    else:
                        status_display = status_val

                    formatted.append({
                        "build_id": build_id,
                        "image_name": image_name,
                        "status": status_val,
                        "status_display": status_display,
                        "duration": runtime_seconds,
                        "duration_display": _format_runtime(runtime_seconds),
                        "step_progress": step_progress,
                        "created_at": created,  # Use consistent field name across all tables
                        "created_display": _format_date(created),
                        "region": region_name  # Add region to each build!
                    })

                return formatted
            except Exception:
                return []  # Fail silently for this region

        # Query all 18 regions in parallel!
        all_builds = []
        with ThreadPoolExecutor(max_workers=18) as executor:
            future_to_region = {executor.submit(fetch_region, region): region for region in MECHA_REGIONS}
            for future in as_completed(future_to_region):
                region_builds = future.result()
                all_builds.extend(region_builds)

        # Sort by created time ONLY (newest first) - pure chronological order
        all_builds.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # Last-known-good cache: Return cached data if API fails
        global _last_known_cloud_builds

        if len(all_builds) > 0:
            _last_known_cloud_builds = all_builds
            return all_builds
        else:
            # API returned 0 results - return last known good!
            return _last_known_cloud_builds

    except Exception:
        # Exception during fetch - return last known good data
        return _last_known_cloud_builds


def _list_recent_cloud_builds(status: StatusCallback, region: str = "us-central1", target_regions: List[str] = None) -> List[Dict]:
    """
    Get recent Cloud Builds from specified or all MECHA regions (adaptive monitoring)

    ACTIVE builds (WORKING/QUEUED) are prepended at the top, then recent completed builds.

    Shows build history for context and debugging.

    Args:
        region: Legacy parameter (ignored)
        status: Status callback for messages
        target_regions: Optional list of specific regions to query. If None, queries all 18 MECHA regions.

    Returns:
        List of build dicts with ACTIVE builds first, then recent builds
        Each dict: id, image_name, status, duration, finished, error, region
    """
    try:
        import json
        from datetime import datetime, timezone
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ALL 18 MECHA regions (default)
        ALL_MECHA_REGIONS = [
            "us-central1", "us-east1", "us-east4", "us-east5",
            "us-west1", "us-west2", "us-west3", "us-west4",
            "northamerica-northeast1",
            "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9",
            "asia-northeast1", "asia-southeast1",
            "australia-southeast1",
            "southamerica-east1"
        ]

        # Use target_regions if provided (adaptive monitoring), otherwise all 18
        MECHA_REGIONS = target_regions if target_regions is not None else ALL_MECHA_REGIONS

        def fetch_region(region_name: str) -> List[Dict]:
            """Fetch recent builds from a single region"""
            try:
                result = run_gcloud_with_retry([
                    "gcloud", "builds", "list",
                    f"--region={region_name}",
                    f"--limit={LIMIT_CLOUD_BUILDS_RECENT}",
                    "--format=json"
                ], max_retries=3, timeout=60, operation_name="list recent Cloud Builds")

                if result.returncode != 0:
                    return []

                builds = json.loads(result.stdout) if result.stdout else []

                formatted = []
                for build in builds:
                    build_id = build.get('id', '')[:12]
                    status_val = build.get('status', 'UNKNOWN')
                    created = build.get('createTime', '')
                    start_time = build.get('startTime')
                    finish_time = build.get('finishTime')

                    # Extract image name
                    images = build.get('images', [])
                    image_name = "unknown"
                    if images:
                        full_image = images[0]
                        if '/' in full_image:
                            image_name = full_image.split('/')[-1].split(':')[0]

                    # Calculate duration (startTime to finishTime, or startTime to NOW for WORKING builds)
                    runtime_seconds = 0
                    if start_time and finish_time:
                        # Completed build - use actual duration
                        try:
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            end = datetime.fromisoformat(finish_time.replace('Z', '+00:00'))
                            runtime_seconds = int((end - start).total_seconds())
                        except Exception:
                            pass
                    elif status_val in ['WORKING', 'QUEUED']:
                        # Active build - calculate elapsed time
                        # WORKING: startTime to NOW (actual build time)
                        # QUEUED: createTime to NOW (time waiting in queue)
                        try:
                            if status_val == 'QUEUED' and created:
                                # Use create time for queued builds (time waiting)
                                start = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            elif start_time:
                                # Use start time for working builds (actual runtime)
                                start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            else:
                                raise ValueError("No time available")

                            now = datetime.now(timezone.utc)
                            runtime_seconds = int((now - start).total_seconds())
                        except Exception:
                            pass

                    # Calculate finished time (how long ago)
                    finished_display = "—"
                    if finish_time:
                        finished_display = _format_date(finish_time)

                    # Color-code recent build status with icons
                    if status_val == "SUCCESS":
                        status_display = f"[green]✓ SUCCESS[/green]"
                    elif status_val == "FAILURE":
                        status_display = f"[bold red]✗ FAILURE[/bold red]"
                    elif status_val == "TIMEOUT":
                        status_display = f"[red]⏱ TIMEOUT[/red]"
                    elif status_val == "CANCELLED":
                        status_display = f"[yellow]⊗ CANCELLED[/yellow]"
                    elif status_val == "WORKING":
                        status_display = f"[green]▶ WORKING[/green]"
                    elif status_val == "QUEUED":
                        status_display = f"[blue]◆ QUEUED[/blue]"
                    else:
                        status_display = status_val

                    # Extract error message if build failed, or completion message if succeeded
                    error_msg = None
                    if status_val in ["FAILURE", "TIMEOUT"]:
                        failure_info = build.get('failureInfo', {})
                        if failure_info:
                            error_msg = failure_info.get('detail', '')
                    elif status_val == "SUCCESS":
                        # Show completion message like runners do (with timestamp)
                        timestamp = ""
                        if finish_time:
                            try:
                                dt = datetime.fromisoformat(finish_time.replace('Z', '+00:00'))
                                timestamp = f"[{dt.strftime('%-I:%M:%S %p')}] "  # %-I strips leading zero
                            except Exception:
                                pass
                        error_msg = f"{timestamp}Build completed: {_format_runtime(runtime_seconds)}"

                    formatted.append({
                        "build_id": build_id,
                        "image_name": image_name,
                        "status": status_val,
                        "status_display": status_display,
                        "duration": runtime_seconds,
                        "duration_display": _format_runtime(runtime_seconds),
                        "created": created,  # createTime for sorting
                        "start_time": start_time,  # For live duration updates
                        "finished": finish_time,
                        "finished_display": finished_display,
                        "error": format_builds_note(error_msg),  # Pre-formatted for display!
                        "region": region_name  # Add region to each build!
                    })

                return formatted
            except Exception:
                return []  # Fail silently for this region

        # Query all 18 regions in parallel!
        all_builds = []
        with ThreadPoolExecutor(max_workers=18) as executor:
            future_to_region = {executor.submit(fetch_region, region): region for region in MECHA_REGIONS}
            for future in as_completed(future_to_region):
                region_builds = future.result()
                all_builds.extend(region_builds)

        # Sort by createTime (most recent first) - matches GCP Web UI and gcloud builds list
        all_builds.sort(key=lambda x: x.get('created') or '', reverse=True)

        # Separate ACTIVE builds (WORKING/QUEUED) from completed builds
        active_builds = [b for b in all_builds if b.get('status') in ['WORKING', 'QUEUED']]
        completed_builds = [b for b in all_builds if b.get('status') not in ['WORKING', 'QUEUED']]

        # ACTIVE builds first (in their own sorted order), then completed builds
        # Limit completed builds to ensure total doesn't exceed LIMIT_CLOUD_BUILDS_RECENT
        max_completed = max(0, LIMIT_CLOUD_BUILDS_RECENT - len(active_builds))
        final_builds = active_builds + completed_builds[:max_completed]

        return final_builds

    except Exception:
        return []  # Fail silently



