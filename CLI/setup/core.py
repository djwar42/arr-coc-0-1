"""
Setup Core Logic - Infrastructure Prerequisites Only

# <claudes_code_comments>
# ** Function List **
# run_setup_core() - Main setup entry point (creates IAM, buckets, VPC, worker pool config)
# check_infrastructure_core() - Check current state (uses infra_verify.verify_all_infrastructure)
# display_infrastructure_tree() - Render infrastructure status as tree (includes GPU quota instructions)
# retry_gcloud_command() - Retry gcloud commands with exponential backoff
# _run_all_setup_steps() - Execute all setup steps sequentially
# _check_prerequisites_core() - Validate CLIs and credentials
# _verify_queue_core() - Check if W&B queue exists (60s polling)
# _retry_subprocess_with_backoff() - Retry subprocess commands with backoff
# _get_all_gpu_quotas() - Get Vertex AI GPU quotas for all types (uses quota module)
# _get_entities_to_try() - Get W&B entities to try for queue/project operations
#
# ** Technical Review **
# Setup creates ONE-TIME infrastructure prerequisites. Worker pools and Docker images
# are created during launch (adaptive to quota changes).
#
# INFRASTRUCTURE VERIFICATION: Uses unified infra_verify.verify_all_infrastructure()
# Checks in order: Billing (FIRST!), GCP resources, W&B, HuggingFace, Local credentials
# Parallel execution with GCloudAccumulator for 9× speedup on GCP checks.
# Returns comprehensive dict with billing/gcp/wandb/hf/local status.
# - Runs EVERY TIME Launch screen opens in TUI (10× per day = 14.5 min daily savings!)
# - Uses api_helpers.GCloudAccumulator + GeneralAccumulator with retry logic
#
# TWO ARTIFACT REGISTRIES:
# 1. arr-coc-registry (deletable) - Fast-building images: ml-stack, trainer, launcher
# 2. arr-coc-registry-persistent (NEVER deleted) - PyTorch base only (~15GB, 2-4hr build)
# Separation prevents expensive PyTorch rebuilds across teardown/setup cycles.
#
# REGIONAL GCS BUCKETS (On-Demand):
# Buckets created during launch when ZEUS selects GPU region (not during setup).
# Format: arr-coc-{region} (e.g., arr-coc-us-west2, arr-coc-europe-west4)
# Infrastructure check lists all existing regional buckets dynamically.
#
# C3 QUOTA DETECTION:
# Uses centralized quota.get_cloud_build_c3_region_quota() to detect Cloud Build C3 quota.
# Only Cloud Build "Concurrent C3 Build CPUs (Private Pool)" quota is enforced by GCP.
#
# Flow: get_cloud_build_c3_region_quota(project_id, region) → cb_quota → select best_machine
# C3 machine tiers: 4, 8, 22, 44, 88, 176 vCPUs (max) - system auto-selects best fit.
#
# Quota entries appear AFTER first launch (lazy loading via MECHA pool creation).
# User requests quota increases via Console, then teardown+setup to apply changes.
#
# GPU QUOTA DISPLAY:
# Uses quota.get_all_vertex_gpu_quotas() to fetch Vertex AI Custom Training quotas.
# Shows GPU quota status with correct VA metric names in console instructions.
# GPU instructions shown in display_infrastructure_tree() when quotas pending.
#
# Setup creates: Two Artifact Registries, Service Account, VPC peering, IAM permissions
# Launch creates: Regional GCS buckets, Worker pools (MECHA multi-region), Docker images, training jobs
# MECHA handles: Region selection, price comparison, quota-aware battles
# </claudes_code_comments>

# ** Setup Flow **
#
#   User → Run Setup
#       ↓
#   run_setup_core(helper, config, status)
#       ↓
#   Execute GCP commands (subprocess)
#       ↓
#   ┌─────────────────────────────────────────────────────┐
#   │  Create infrastructure:                             │
#   │  • Artifact Registry (deletable - SHARED)           │
#   │  • Artifact Registry Persistent (PyTorch - SHARED)  │
#   │  • Service Account (SHARED)                         │
#   │  • W&B Queue (via helper)                           │
#   │  • W&B Project (via helper)                         │
#   │  • HuggingFace Repo (via helper)                    │
#   │  Note: GCS buckets created on-demand during launch  │
#   └─────────────────────────────────────────────────────┘
#       ↓
#   Verify queue exists (60s polling)
#       ↓
#   Return: True (success) | False (failure)
#
# ** Infrastructure Check Flow **
#
#   check_infrastructure_core(helper, config, status)
#       ↓
#   Query GCP/W&B/HF APIs
#       ↓
#   ┌─────────────────────────────────────────────────┐
#   │  Check existence:                               │
#   │  ✓ GCS Regional Buckets (lists all arr-coc-*)   │
#   │  ✓ Artifact Registry (deletable)                │
#   │  ✓ Artifact Registry Persistent (PyTorch)       │
#   │  ✓ Service Account                              │
#   │  ✓ W&B Queue                                    │
#   │  ✓ W&B Project                                  │
#   │  ✓ HuggingFace Repo                             │
#   └─────────────────────────────────────────────────┘
#       ↓
#   Return: Dict with status info
#
# INFRASTRUCTURE ARCHITECTURE:
===========================

SHARED RESOURCES (All ARR-COC prototypes share these):
-------------------------------------------------------
1. Artifact Registry: arr-coc-registry
   - Stores Docker images for ALL prototypes
   - Images tagged by project: arr-coc-0-1:latest, arr-coc-0-2:latest

2. Service Account: arr-coc-sa@{project_id}.iam.gserviceaccount.com
   - Shared credentials with project-level permissions
   - All prototypes use same SA for Vertex AI launches

3. W&B Launch Queue: vertex-ai-queue
   - Handles launch jobs from ALL prototypes
   - Jobs specify which W&B project to log to

PROJECT-SPECIFIC RESOURCES (Each prototype gets its own):
---------------------------------------------------------
1. GCS Buckets (for easy deletion when removing prototypes):
   - {project_id}-{PROJECT_NAME}-staging
   - {project_id}-{PROJECT_NAME}-checkpoints

2. W&B Project: {entity}/{PROJECT_NAME}
   - Separate runs/experiments/dashboards per prototype

3. HuggingFace Repo: {user}/{PROJECT_NAME}
   - Separate model outputs per prototype

CRITICAL UNDERSTANDING: Setup vs Launch Separation
==================================================

SETUP (this file):
    - Creates infrastructure prerequisites ONCE
    - Artifact Registry (for storing Docker images)
    - Staging bucket (required by Vertex AI)
    - Service account (with IAM roles)
    - Does NOT build images!

LAUNCH (cli/launch/core.py):
    - Happens EVERY time you launch a training job
    - Checks Dockerfile hash (SHA256, last 7 chars)
    - If hash changed → rebuilds image on Cloud Build
    - If hash same → reuses existing image (fast!)
    - Image tag: training:{hash} (e.g., training:a1b2c3d)

WHY SEPARATE?
    - Setup: One-time infrastructure (minutes)
    - Launch: Per-job with smart caching (seconds if hash matches)
    - Image rebuilds happen automatically when Dockerfile changes!

CRITICAL: NO Textual dependencies! Uses StatusCallback for all output.

Architecture:
    CLI: cli.py setup → run_setup_core() → PrintCallback
    TUI: screen.py → run_setup_core() → TUICallback
    Both: Reuse _create_staging_bucket() and _create_service_account() from ../launch/core.py

Flow:
    1. Create Artifact Registry (if doesn't exist)
    2. Create staging bucket (reuses launch function)
    3. Create service account (reuses launch function)
    4. Done! Images build during launch with hash checking.

Functions:
    - run_setup_core(): Main entry point - creates infrastructure
    - check_infrastructure_core(): Check current state
    - _check_prerequisites_core(): Validate CLIs and credentials
    - _gather_infrastructure_info_core(): Query GCP/W&B/HF status
    - _verify_queue_core(): Check if W&B queue exists
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import infrastructure setup functions from launch (they're reusable!)
from ..launch.core import _create_service_account
from ..shared.api_helpers import GCloudAccumulator, GeneralAccumulator
from ..shared.callbacks import StatusCallback
from ..shared.wandb_helper import WandBHelper
from ..shared.quota import get_all_vertex_gpu_quotas, get_vertex_gpu_quota_metric
from .steps import (
    _enable_apis,
    _setup_registry,
    _setup_persistent_registry,
    _setup_buckets,
    _create_service_account as _setup_service_account,
    _create_queue,
    _setup_cloudbuild_iam,
    _setup_vpc_peering,
)


def retry_gcloud_command(
    cmd: list[str],
    success_msg: str,
    error_msg: str,
    already_exists_phrases: list[str] = None,
    timeout: int = 60,
    retry_delay: int = 2,
) -> tuple[bool, str]:
    """
    Execute gcloud command with automatic retry (1 original attempt + 1 retry).

    Args:
        cmd: gcloud command as list of strings
        success_msg: Message to log on success
        error_msg: Message prefix for errors
        already_exists_phrases: List of phrases in stderr that indicate resource already exists (OK)
        timeout: Command timeout in seconds
        retry_delay: Seconds to wait before retry

    Returns:
        (success: bool, output: str)
    """
    if already_exists_phrases is None:
        already_exists_phrases = ["already exists"]

    for attempt in range(2):  # 2 attempts: original + 1 retry
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            return True, success_msg

        # Check if resource already exists (this is OK!)
        for phrase in already_exists_phrases:
            if phrase.lower() in result.stderr.lower():
                return True, f"{success_msg} (already exists)"

        # Retry if this was first attempt
        if attempt < 1:
            time.sleep(retry_delay)
            continue

        # Failed after retry
        return False, f"{error_msg}: {result.stderr[:200]}"

    return False, f"{error_msg}: Unknown error"


def display_infrastructure_tree(
    info: Dict, status: StatusCallback, config: Dict[str, str] = None
) -> None:
    """
    Display infrastructure status using unified infra_print

    Args:
        info: Infrastructure dict from verify_all_infrastructure()
        status: Status callback for output
        config: Optional config dict (unused, kept for backwards compatibility)
    """
    # 🎯 USE UNIFIED DISPLAY FUNCTION!
    from CLI.shared.infra_print import display_infrastructure as format_infra

    # Get formatted output (Rich markup enabled for CLI)
    output = format_infra(info, use_rich=True)

    # Print each line using status callback
    for line in output.split("\n"):
        status(line)


def _run_all_setup_steps(
    config: Dict[str, str],
    project_id: str,
    region: str,
    entity: str,
    status: StatusCallback,
) -> bool:
    """
    Execute all 8 setup steps with immediate feedback

    Each step function:
    - Receives status() callback for immediate feedback
    - Returns bool (True = success, False = failure)
    - Handles its own retries and idempotency

    Returns:
        bool: True if all steps succeeded, False otherwise
    """

    # Step 1/9: Enable APIs
    status("")
    status("Enabling GCP APIs... (1/8)")
    status("   [dim]ℹ[/dim] APIs for Vertex AI, Cloud Build, Artifact Registry")
    if not _enable_apis(project_id, status):
        return False

    # Step 2/9: Artifact Registry (SHARED - for all ARR-COC prototypes)
    status("")
    status("Setting up Artifact Registry... (2/8)")
    status("   [dim]ℹ[/dim] Docker images for all ARR-COC prototypes")
    if not _setup_registry(project_id, region, status):
        return False

    # Step 2a/9: Persistent Registry (PyTorch base - NEVER deleted)
    status("   [dim]ℹ[/dim] Persistent registry for PyTorch base (~15GB)")
    if not _setup_persistent_registry(project_id, region, status):
        return False

    # Step 3/9: GCS Buckets (reuse launch function)
    status("")
    status("Creating GCS buckets... (3/8)")
    status("   [dim]ℹ[/dim] W&B Launch staging bucket for training artifacts")
    if not _setup_buckets(config, project_id, region, status):
        return False

    # Step 4/9: Service Account (reuse launch function)
    status("")
    status("Creating Service Account... (4/8)")
    status("   [dim]ℹ[/dim] IAM roles for Storage, Artifact Registry, Cloud Build")
    if not _setup_service_account(config, project_id, region, status):
        return False

    # Step 4.5/8: W&B API Key Secret
    status("")
    status("Creating W&B API Key Secret... (4.5/8)")
    status("   [dim]ℹ[/dim] Securely stores W&B API key for Cloud Run job")
    from ..shared.setup_helper import create_wandb_secret
    if not create_wandb_secret(status):
        return False
    status("   [green]⚡W&B Secret passed - Roger![/green]")

    # Step 5/8: W&B Queue
    status("")
    status("Creating W&B Launch Queue... (5/8)")
    status("   [dim]ℹ[/dim] Vertex AI queue for automated training jobs")
    if not _create_queue(config, entity, status):
        return False

    # Step 7/9: Pricing Cloud Function
    status("")
    status("Building pricing cloud function... (6/8)")
    status("   [dim]ℹ[/dim] Auto-updates GCP pricing every 20 minutes")
    from .pricing_setup import setup_pricing_infrastructure
    if not setup_pricing_infrastructure(status):
        return False

    # Step 8/9: Cloud Build IAM (required for worker pools)
    status("")
    status("Granting Cloud Build permissions... (7/8)")
    status("   [dim]ℹ[/dim] Roles needed: compute.admin + compute.networkUser")
    if not _setup_cloudbuild_iam(project_id, status):
        return False

    # Step 9/9: VPC Peering (required for worker pools)
    status("")
    status("Setting up VPC peering... (8/8)")
    status("   [dim]ℹ[/dim] Required for Service Networking API")
    if not _setup_vpc_peering(project_id, status):
        return False

    return True


def run_setup_core(
    helper: WandBHelper,
    config: Dict[str, str],
    status: StatusCallback,
) -> bool:
    """
    Run infrastructure setup

    Creates:
    - GCP APIs (enabled)
    - Artifact Registry (SHARED)
    - GCS staging bucket
    - Service account
    - Environment file
    - W&B Launch queue
    - Cloud Build worker pool
    - Cloud Build IAM permissions
    - VPC peering

    Returns:
        bool: True if setup succeeded, False otherwise
    """

    try:
        # Extract config values (OLD WAY - direct access)
        region = "us-central1"
        project_id = config.get("GCP_PROJECT_ID", "")
        entity = config.get("WANDB_ENTITY", "arr-coc")

        # Display header
        status("⏳ Starting infrastructure setup...")
        status(f"   Project: {project_id}")
        status(f"   Region: {region}")
        status(f"   Entity: {entity}")

        # Run all setup steps with numbered format (IDEAL WAY)
        # Each step gets status() callback for immediate feedback
        success = _run_all_setup_steps(
            config=config,
            project_id=project_id,
            region=region,
            entity=entity,
            status=status
        )

        if not success:
            return False

        # Success!
        status("")
        status("[green]✅ Infrastructure setup complete![/green]")
        status("")
        status("[dim]🚀 Next: Launch a training job[/dim]")
        return True

    except Exception as e:
        status(f"[red]❌ Setup error: {str(e)[:300]}[/red]")
        return False


def check_infrastructure_core(
    helper: WandBHelper,
    config: Dict[str, str],
    status: StatusCallback,
    app=None,  # Unused - kept for API compatibility
    gpu_progress_callback=None,  # 🦡⚡ Optional callback for GPU quota progress
    c3_progress_callback=None,   # 🦡⚡ Optional callback for C3 quota progress
) -> Dict:
    """
    Check current infrastructure state

    Queries GCP, W&B, and HuggingFace for existing infrastructure.
    Always runs fresh check - no caching (simpler, more reliable).

    Args:
        helper: WandBHelper instance
        config: Training configuration
        status: Status callback
        app: Unused (kept for compatibility)

    Returns:
        Dict with infrastructure state:
        {
            "gcp": {"bucket": {...}, "registry": {...}, "sa": {...}},
            "wandb": {"queue": {...}, "project": {...}},
            "hf": {"repo": {...}}
        }

    Example:
        >>> helper = WandBHelper("entity", "project", "queue")
        >>> config = load_training_config()
        >>> status = PrintCallback()
        >>> info = check_infrastructure_core(helper, config, status)
        >>> print(info['gcp']['buckets']['count'])  # Number of regional buckets
        >>> print(info['gcp']['registry']['exists'])  # Deletable registry
        >>> print(info['gcp']['persistent_registry']['exists'])  # PyTorch registry
    """
    try:
        from CLI.shared.infra_verify import verify_all_infrastructure  # 🦡 FIX: Absolute import (same as tui.py)
        from ..shared.stevens_dance import stevens_log

        stevens_log("infra", "🍞 BREADCRUMB 2.1: About to call verify_all_infrastructure")
        info = verify_all_infrastructure(
            helper,
            config,
            status,
            app=app,
            gpu_progress_callback=gpu_progress_callback,  # 🦡⚡ Pass callbacks!
            c3_progress_callback=c3_progress_callback
        )
        stevens_log("infra", "🍞 BREADCRUMB 2.2: verify_all_infrastructure returned!")
        return info

    except Exception as e:
        stevens_log("infra", f"🚨 BREADCRUMB 2.ERROR: Exception in check_infrastructure_core: {str(e)[:100]}")
        status(f"[red]❌ Error checking infrastructure: {str(e)[:200]}[/red]")
        return {"billing": {}, "gcp": {}, "wandb": {}, "hf": {}, "local": {}}


def _check_prerequisites_core(
    helper: WandBHelper,
    config: Dict[str, str],
    status: StatusCallback,
) -> Dict:
    """
    Check setup prerequisites

    Extracted from: screen.py lines 392-465

    Checks:
    - W&B queue exists

    Args:
        helper: WandBHelper instance
        config: Training configuration
        status: Status callback

    Returns:
        Dict with check results:
        {
            "queue_exists": bool,
            "queue_name": str,
            "entities_checked": List[str]
        }
    """
    queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
    entity = config.get("WANDB_ENTITY", "")

    status(f"⏳ Checking for W&B queue '{queue_name}'...")

    # Check if queue exists
    try:
        import wandb

        api = wandb.Api()

        # Get entities to try
        entities_to_try = _get_entities_to_try(api, entity)

        status(f"[dim]Searching entities: {', '.join(entities_to_try)}[/dim]")

        queue_exists = False
        for ent in entities_to_try:
            try:
                # Use run_queue() to check if queue exists (not launch_queues - doesn't exist!)
                queue = api.run_queue(ent, queue_name)
                if queue:
                    queue_exists = True
                    status(
                        f"[green]✓[/green]  Found queue '{queue_name}' in entity '{ent}'"
                    )
                    break
            except Exception:
                # Queue doesn't exist in this entity, try next
                continue

        if not queue_exists:
            status(f"[yellow]⚠️  Queue '{queue_name}' not found[/yellow]")

        return {
            "queue_exists": queue_exists,
            "queue_name": queue_name,
            "entities_checked": entities_to_try,
        }

    except Exception as e:
        status(f"[red]❌ Error checking queue: {str(e)[:200]}[/red]")
        return {"queue_exists": False, "queue_name": queue_name, "entities_checked": []}


def _retry_subprocess_with_backoff(cmd, timeout=10, max_attempts=3):
    """
    Execute subprocess with retry and backoff

    Retries command up to 3 times with delays: 1s → 4s → 8s
    Total worst-case time: ~10s + 1s + 10s + 4s + 10s = 35s

    Args:
        cmd: Command list (e.g., ["gsutil", "ls", "gs://bucket"])
        timeout: Timeout per attempt in seconds
        max_attempts: Number of attempts (default 3)

    Returns:
        subprocess.CompletedProcess or None if all attempts failed
    """
    import time

    delays = [1, 4, 8]  # Backoff delays between attempts
    result = None

    for attempt in range(max_attempts):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result  # Success! Return immediately
            # Non-zero returncode (e.g., resource doesn't exist) - don't retry
            return result
        except subprocess.TimeoutExpired:
            # Timeout - retry with backoff (unless last attempt)
            if attempt < max_attempts - 1:
                time.sleep(delays[attempt])
                continue
            else:
                # Last attempt failed - return None
                return None

    return result



def _get_all_gpu_quotas(project_id: str, region: str) -> List[Dict]:
    """
    Get ALL Vertex AI GPU quotas for a specific region

    Args:
        project_id: GCP project ID
        region: GCP region (e.g., "us-central1")

    Returns:
        List of dicts with Vertex AI GPU quotas (sorted by quota limit desc)
    """
    return get_all_vertex_gpu_quotas(project_id, region)


def _verify_queue_core(
    helper: WandBHelper,
    config: Dict[str, str],
    status: StatusCallback,
    timeout: int = 60,
) -> bool:
    """
    Verify W&B queue was created

    Extracted from: screen.py lines 1200-1280

    Polls for queue existence for up to timeout seconds.

    Args:
        helper: WandBHelper instance
        config: Training configuration
        status: Status callback
        timeout: Max seconds to wait

    Returns:
        True if queue found
        False if timeout reached

    Side effects:
        - Polls every 3 seconds
        - Updates status with progress
    """
    queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
    entity = config.get("WANDB_ENTITY", "")

    status(f"⏳ Waiting for queue '{queue_name}' to appear...")
    status(f"[dim]Checking every 3s (max {timeout}s)...[/dim]")

    start_time = time.time()
    checks = 0

    while (time.time() - start_time) < timeout:
        checks += 1
        elapsed = int(time.time() - start_time)

        try:
            import wandb

            api = wandb.Api()
            entities_to_try = _get_entities_to_try(api, entity)

            for ent in entities_to_try:
                try:
                    # Use run_queue() to check if queue exists
                    queue = api.run_queue(ent, queue_name)
                    if queue:
                        status(
                            f"[green]✓[/green]  Queue '{queue_name}' found in entity '{ent}'!"
                        )
                        return True
                except Exception:
                    # Queue doesn't exist in this entity, try next
                    continue

        except Exception as e:
            pass  # Ignore transient errors

        # Update status every 10 seconds
        if checks % 3 == 0:  # Every 3rd check (9 seconds)
            status(f"[dim]Still checking... ({elapsed}s elapsed)[/dim]")

        # Wait 3 seconds before next check
        time.sleep(3)

    # Timeout reached
    status(f"[yellow]⚠️  Queue not found after {timeout}s[/yellow]")
    return False


def _get_entities_to_try(api, primary_entity: str) -> List[str]:
    """
    Get list of W&B entities to search

    Extracted from: screen.py lines 320-380

    Returns primary entity + user's default entity if different.

    Args:
        api: W&B API instance
        primary_entity: Primary entity from config

    Returns:
        List of entity names to try
    """
    entities = [primary_entity]

    try:
        # Try to get user's default entity
        viewer = api.viewer
        if viewer and hasattr(viewer, "entity"):
            user_entity = viewer.entity
            if user_entity and user_entity != primary_entity:
                entities.append(user_entity)
    except Exception:
        pass  # If API call fails, just use primary entity

    return entities


