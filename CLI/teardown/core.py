"""
Teardown Core Logic

Refactored to match setup architecture:
- Separate step functions in steps.py
- Coordinator function for orchestration
- Entry point for CLI/TUI integration

Architecture matches setup:
    CLI/teardown/
    ├── steps.py    ← 7 step functions (~400 lines)
    └── core.py     ← Coordinator + entry point (this file)

Pattern:
    ✓ One function per step
    ✓ All return bool (simple)
    ✓ Numbered (1/7 → 6/6)
    ✓ Clean separation of concerns
"""

# <claudes_code_comments>
# ** Function List **
# _run_all_teardown_steps(config, project_id, region, status, dry_run, skip_images) - Coordinator: runs 6 teardown steps sequentially
# run_teardown_core(helper, config, status, dry_run, mode) - Main entry point for teardown (called from CLI/TUI) with granular mode support
# list_resources_core(config, status) - List all resources that would be deleted (dry-run helper)
#
# ** Technical Review **
# This module implements granular teardown with two modes: infrastructure vs individual Docker images.
#
# PERFORMANCE OPTIMIZATION: Bucket checks use GCloudAccumulator for 2× speedup!
# - check_buckets_for_region() checks 2 buckets in PARALLEL (was sequential)
# - Before: 2 × 5s = 10s | After: Both at once = 5s (5s savings per teardown!)
# - Uses api_helpers.GCloudAccumulator with retry logic
# - Note: IAM revocations MUST stay sequential (GCP concurrent policy error)
#
# Flow: run_teardown_core() parses mode string → routes to infrastructure or image deletion:
# - "DELETE" mode: Calls _run_all_teardown_steps(skip_images=True) → deletes buckets/SA/worker pool but preserves all 4 arr- images
# - Image mode ("ARR-PYTORCH-BASE", "ARR-ML-STACK", etc.): Calls _delete_single_image() for each image → deletes specific images, keeps infrastructure
#
# Coordinator (_run_all_teardown_steps) executes 6 steps:
# 1. Worker Pool (pytorch-mecha-pool)
# 2. Artifact Registry (arr-coc-registry) - skips image deletion when skip_images=True
# 3. Staging Buckets (SHARED + project)
# 4. Checkpoints Bucket (project)
# 5. IAM Role Bindings (6 bindings total)
# 6. Pricing Infrastructure (function, scheduler, OIDC)
#
# Mode routing (line 186-239):
# - Parses space-separated keywords from TUI/CLI
# - DELETE in keywords → skip_images=True (preserves images)
# - Image keywords only → loops through _delete_single_image() for each
# - Supports combinations like "ARR-PYTORCH-BASE ARR-ML-STACK" (deletes both images)
#
# Performance monitoring integrated via get_monitor() for both infrastructure and image operations.
# Dry-run mode supported for all operations (shows "Would delete" messages without actual deletion).
# </claudes_code_comments>

from typing import Dict, List
from pathlib import Path

from ..shared.callbacks import StatusCallback
from ..shared.setup_helper import SetupHelper

# Import all step functions
from .steps import (
    _delete_worker_pool,
    _delete_registry,
    _delete_staging_buckets,
    _delete_checkpoints_bucket,
    _remove_iam_bindings,
    _teardown_pricing,
)


def _run_all_teardown_steps(
    config: Dict[str, str],
    project_id: str,
    region: str,
    status: StatusCallback,
    dry_run: bool = False,
    skip_images: bool = False,
) -> bool:
    """
    Coordinator: Run all 6 teardown steps sequentially

    Matches setup coordinator pattern:
    - Calls each step function in order
    - Shows numbered progress (1/7 → 6/6)
    - Returns False on first failure
    - Shows context comments for each step

    Args:
        config: Training configuration
        project_id: GCP project ID
        region: GCP region (us-central1)
        status: Status callback for updates
        dry_run: If True, show what would be deleted (no actual deletion)
        skip_images: If True, skip image deletion (keeps all arr- images)

    Returns:
        True if all steps succeeded
        False if any step failed
    """

    if dry_run:
        status("")
        status("[yellow]DRY RUN MODE: No resources will be deleted[/yellow]")
        status("")

    # Step 1/6: Worker Pool
    status("")
    status("Tearing down Cloud Build Worker Pool... (1/6)")
    status("   [dim]ℹ[/dim] Cloud Build worker pool for training jobs")
    if dry_run:
        status("   Would delete: pytorch-mecha-pool (c3-standard-176)")
    elif not _delete_worker_pool(project_id, region, status):
        return False

    # Step 2/6: Artifact Registry (SHARED)
    status("")
    status("Tearing down Artifact Registry... (2/6)")
    status("   [dim]ℹ[/dim] Docker images for all ARR-COC prototypes")
    if dry_run:
        if skip_images:
            status("   Would skip: Image deletion (keeping all arr- images)")
        else:
            status("   Would delete: arr-coc-registry (including all images)")
    elif not _delete_registry(region, status, delete_images=not skip_images):
        return False

    # Step 3/6: Staging Buckets (SHARED + project)
    status("")
    status("Tearing down GCS Staging Buckets... (3/6)")
    project_name = config.get("PROJECT_NAME", "arr-coc")
    status(f"   [dim]ℹ[/dim] W&B Launch staging + {project_name} staging")
    if dry_run:
        status(f"   Would delete: gs://{project_id}-staging")
        status(f"   Would delete: gs://{project_id}-{project_name}-staging")
    elif not _delete_staging_buckets(config, project_id, status):
        return False

    # Step 4/6: Checkpoints Bucket (project-specific)
    status("")
    status("Tearing down GCS Checkpoints Bucket... (4/6)")
    status(f"   [dim]ℹ[/dim] {project_name} training checkpoints")
    if dry_run:
        status(f"   Would delete: gs://{project_id}-{project_name}-checkpoints")
    elif not _delete_checkpoints_bucket(config, project_id, status):
        return False

    # Step 5/6: IAM Role Bindings
    status("")
    status("Tearing down IAM Role Bindings... (5/6)")
    status("   [dim]ℹ[/dim] 4 SA roles + 2 Cloud Build roles")
    if dry_run:
        status("   Would remove: 4 SA IAM bindings")
        status("   Would remove: 2 Cloud Build IAM bindings")
    elif not _remove_iam_bindings(config, project_id, status):
        return False

    # Step 6/6: Pricing Infrastructure
    status("")
    status("Tearing down pricing infrastructure... (6/6)")
    status("   [dim]ℹ Scheduler, Function, OIDC, APIs (repository preserved)[/dim]")
    if dry_run:
        status("   Would delete: arr-coc-pricing-scheduler")
        status("   Would delete: arr-coc-pricing-runner")
        status("   Would revoke: OIDC permissions")
        status("   Would disable: Cloud Billing API")
        status("   Would preserve: arr-coc-pricing repository (historical data)")
    elif not _teardown_pricing(status):
        return False

    return True


def run_teardown_core(
    helper: SetupHelper,
    config: Dict[str, str],
    status: StatusCallback,
    dry_run: bool = False,
    mode: str = "DELETE",
) -> bool:
    """
    Main entry point for teardown (called from CLI/TUI)

    Matches setup entry point pattern:
    - Extract config values
    - Call coordinator
    - Show final success/failure message

    Args:
        helper: SetupHelper instance (not used in refactored version)
        mode: What to delete (DELETE, ARR-PYTORCH-BASE, ARR-ML-STACK, ARR-TRAINER, ARR-VERTEX-LAUNCHER)
        config: Training configuration dict
        status: Status callback for updates
        dry_run: If True, preview only (no deletion)

    Returns:
        True if teardown succeeded (or dry-run completed)
        False if teardown failed

    Example:
        >>> helper = SetupHelper(config)
        >>> status = PrintCallback()
        >>> success = run_teardown_core(helper, config, status, dry_run=True)
    """
    from CLI.shared.performance_monitor import get_monitor
    monitor = get_monitor()

    try:
        # Extract config values (matches setup pattern)
        region = "us-central1"
        project_id = config.get("GCP_PROJECT_ID", "")

        # Check if GCP project configured
        if not project_id or project_id == "(unset)":
            status("[yellow]⚠️  No GCP project configured - skipping GCP resource cleanup[/yellow]")
            status("")
            return True

        # Parse mode to determine what to delete
        keywords = mode.upper().split()

        # Route based on mode
        if "DELETE" in keywords:
            # Infrastructure teardown (skip image deletion)
            status("")
            status("[bold]Infrastructure Teardown Mode[/bold]")
            status("[dim]Deleting infrastructure, keeping all arr- images[/dim]")
            status("")

            op_id = monitor.start_operation("teardown_infrastructure", category="GCP")
            success = _run_all_teardown_steps(
                config, project_id, region, status, dry_run, skip_images=True
            )
            monitor.end_operation(op_id)

            if success and not dry_run:
                status("")
                status("[green]✓ Infrastructure teardown complete![/green]")
                status("[dim]All 4 arr- images preserved in Artifact Registry[/dim]")

        else:
            # Image deletion mode
            image_keywords = [k for k in keywords if k in ["ARR-PYTORCH-BASE", "ARR-ML-STACK", "ARR-TRAINER", "ARR-VERTEX-LAUNCHER"]]

            if not image_keywords:
                status("[red]✗ No valid image names found in mode[/red]")
                return False

            status("")
            status(f"[bold]Image Deletion Mode[/bold]")
            status(f"[dim]Deleting {len(image_keywords)} image(s)[/dim]")
            status("")

            from .steps import _delete_single_image

            op_id = monitor.start_operation("teardown_images", category="GCP")
            all_success = True
            for image_name in image_keywords:
                status(f"Deleting {image_name}...")
                if not _delete_single_image(project_id, region, image_name, status):
                    all_success = False
                    status(f"[red]✗ Failed to delete {image_name}[/red]")
                else:
                    status(f"[green]✓ Deleted {image_name}[/green]")
                status("")

            monitor.end_operation(op_id)
            success = all_success

            if success:
                status("[green]✓ All image deletions complete![/green]")
            else:
                status("[yellow]⚠ Some image deletions failed[/yellow]")

        if success:
            # Show final success message (mode-specific)
            status("")
            if dry_run:
                status("[green]✅ Dry-run complete. No resources deleted.[/green]")
            elif "DELETE" in keywords:
                # Infrastructure teardown summary
                status("")
                status("   Resources deleted:")
                status("   • Cloud Build Pool, Worker Pool")
                status("   • GCS Buckets (staging + checkpoints)")
                status("   • IAM Bindings, SA Key")
                status("   • Pricing Infrastructure (Cloud Function + Scheduler)")
                status("")
                status("   Preserved (NOT deleted):")
                status("   • All 4 arr- Docker images in Artifact Registry")
                status("   • Artifact Registry repository itself")
                status("   • Service Account")
                status("   • VPC Peering (shared infrastructure)")
                status("")
                status("   Manual deletion required:")
                status("   • W&B Launch Queue (delete at https://wandb.ai)")
            # Image deletion mode summary already shown above

            return True
        else:
            status("[red]❌ Teardown failed![/red]")
            return False

    except Exception as e:
        status(f"[red]❌ Teardown error: {str(e)[:300]}[/red]")
        return False


def check_infrastructure_core(
    config: Dict[str, str],
    status: StatusCallback,
    helper=None,  # Optional: for full cloud checks (teardown only needs local)
) -> Dict:
    """
    Check if infrastructure exists (uses unified verifier)

    Args:
        config: Training configuration
        status: Status callback
        helper: Optional WandBHelper (for full cloud checks, not needed for teardown)

    Returns:
        Dict with:
        {
            "exists": bool,
            "key_path": str
        }
    """
    try:
        from CLI.shared.infra_verify import verify_all_infrastructure  # 🦡 FIX: Absolute import (same as tui.py)

        # Use unified verifier (only local check needed for teardown)
        info = verify_all_infrastructure(helper, config, status)

        exists = info["local"]["key_file_exists"]
        key_path = info["local"]["key_path"]

        if exists:
            status("[green]✓[/green]  Infrastructure detected")
        else:
            status("[dim]○ No infrastructure detected[/dim]")

        return {
            "exists": exists,
            "key_path": key_path
        }

    except Exception as e:
        status(f"[red]❌ Error checking infrastructure: {str(e)[:200]}[/red]")
        return {
            "exists": False,
            "key_path": ""
        }


def list_resources_core(
    config: Dict[str, str],
    status: StatusCallback,
) -> List[str]:
    """
    List resources that would be deleted

    Args:
        config: Training configuration
        status: Status callback

    Returns:
        List of resource descriptions
    """
    project_name = config.get('PROJECT_NAME', 'arr-coc-0-1')
    project_id = config.get('GCP_PROJECT_ID', '')
    queue_name = config.get('WANDB_LAUNCH_QUEUE_NAME', 'vertex-ai-queue')

    resources = [
        f"W&B Launch Queue: {queue_name}",
        f"Cloud Build Worker Pool: pytorch-mecha-pool (c3-standard-176)",
        f"Artifact Registry: arr-coc-registry (SHARED)",
        f"W&B Staging Bucket: gs://{project_id}-staging (SHARED)",
        f"Project Staging Bucket: gs://{project_id}-{project_name}-staging",
        f"Checkpoints Bucket: gs://{project_id}-{project_name}-checkpoints",
        f"Service Account: arr-coc-sa@{project_id}.iam.gserviceaccount.com (SHARED)",
        f"Service Account (kept - using gcloud auth)",
        f"6 IAM Role Bindings (4 for training SA + 2 for Cloud Build)",
        f"Pricing Infrastructure: Cloud Function + Scheduler (arr-coc-pricing-runner)"
    ]

    return resources
