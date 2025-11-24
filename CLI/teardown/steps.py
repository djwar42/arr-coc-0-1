"""
Teardown Step Functions

Each step deletes one type of GCP/W&B infrastructure resource.
Matches setup architecture: one function per step, simple bool returns.

Steps:
1. _delete_worker_pool() - Delete pytorch-mecha-pool
2. _delete_registry() - Delete arr-coc-registry (SHARED)
3. _delete_staging_buckets() - Delete staging buckets (SHARED + project)
4. _delete_checkpoints_bucket() - Delete checkpoints bucket (project)
5. _remove_iam_bindings() - Remove all IAM role bindings
6. _delete_sa_key() - Delete local SA key file
7. _teardown_pricing() - Delete pricing infrastructure

Pattern (matches setup architecture):
- All functions return bool (simple, not tuples)
- All use status: StatusCallback for immediate feedback
- All handle idempotent deletion (already deleted = success)
- All truncate stderr to 4000 chars (full errors)
- All use subprocess with capture_output, text, timeout
"""

# <claudes_code_comments>
# ** Function List **
# _delete_worker_pool(project_id, region, status) - Step 1: Delete Cloud Build worker pool (pytorch-mecha-pool)
# _delete_registry(region, status, delete_images=True) - Step 2: Delete Artifact Registry (with optional image preservation)
# _delete_staging_buckets(config, project_id, status) - Step 3: Delete GCS staging buckets (SHARED + project)
# _delete_checkpoints_bucket(config, project_id, status) - Step 4: Delete GCS checkpoints bucket (project)
# _remove_iam_bindings(config, project_id, status) - Step 5: Remove all IAM role bindings
# _teardown_pricing(status) - Step 6: Delete pricing infrastructure (function, scheduler, OIDC, APIs)
# _delete_single_image(project_id, region, image_name, status) - Delete a single Docker image from Artifact Registry
#
# ** Technical Review **
# This module implements all teardown steps, including granular Docker image deletion.
#
# Infrastructure teardown (6 steps):
# 1. Worker pool deletion - c3-standard-176 machine, us-west2/us-central1
# 2. Artifact Registry - Deletes arr-coc-registry (deletable images only)
#    NOTE: arr-coc-registry-persistent is NEVER deleted (PyTorch base image preservation)
# 3-4. GCS buckets - Staging (SHARED + project-specific) + checkpoints
# 5. IAM bindings - 6 total (4 SA roles + 2 Cloud Build roles)
# 6. Pricing infrastructure - Scheduler, Function, OIDC, APIs (preserves repository)
#
# TWO REGISTRIES ARCHITECTURE:
# - arr-coc-registry (deletable): ml-stack, trainer, launcher - Safe to delete, rebuilds in minutes
# - arr-coc-registry-persistent (permanent): PyTorch base only - NEVER deleted, prevents 2-4hr rebuilds
# User must manually delete persistent registry if needed (gcloud artifacts repositories delete)
#
# Image deletion (_delete_single_image):
# - Maps keyword to actual image (ARR-PYTORCH-BASE → arr-pytorch-base, ARR-ML-STACK → arr-ml-stack, etc.)
# - Lists all tags for the image using gcloud artifacts docker images list
# - Deletes each tag individually with gcloud artifacts docker images delete
# - Idempotent: returns True if image already deleted or not found
# - Error handling: returns False on any tag deletion failure, shows truncated stderr (400 chars)
#
# _delete_registry granular mode (delete_images=False):
# - When skip_images=True passed from coordinator, skips ALL registry/image deletion
# - Returns True immediately with "skipped" message
# - Preserves: Registry, all 4 arr- images, pricing data packages
#
# All functions follow setup pattern: idempotent deletion (already deleted = success), full error display (4000 chars).
# </claudes_code_comments>

# ═══════════════════════════════════════════════════════════════════════
# TWO-SPACE STANDARD SYSTEM (Output Formatting)
# IMPORTANT: Don't remove - consistent formatting across setup/teardown
# ═══════════════════════════════════════════════════════════════════════
# SPACING AFTER ICONS (spaces between icon and text):
#   ℹ shows like: "   ℹ APIs for Storage"       - 1 space  (info/context)
#   ⊕ shows like: "   ⊕  Creating registry..."   - 2 spaces (operation/action)
#   ⚡ shows like: "   ⚡GCP APIs passed - Roger!"  - 0 spaces  (step complete - Lightning Pow Pow!)
#   ✓ shows like: "   ✓    Registry created"    - 4 spaces (success/done)
#   ✗ shows like: "   ✗    Creation failed"     - 4 spaces (failure/error)
#   🪙 shows like: "   🪙  Using pricing (4.1KB)" - 2 spaces (pricing data)
#   ☁️ shows like: "   ☁️  Function deployed"    - 2 spaces (cloud resource)
#   📄 shows like: "   📄  Checked 5000 SKUs"    - 2 spaces (progress update)
#   🔄 shows like: "   🔄  Fetching pricing"     - 2 spaces (fetching/loading)
#   🚀 shows like: "   🚀  Triggering function"  - 2 spaces (triggering/launching)
#   ⏳ shows like: "   ⏳  Deploying..."         - 2 spaces (waiting/in-progress)
#
# OTHER RULES:
#   - All icons have 3 spaces before them (prefix)
#   - Headers: "Creating Service Account... (4/9)" (no prefix)
#   - Indentation levels: L0=0sp | L1=3sp | L2=9sp | L3=15sp
#   - LIGHTNING POW POW FINALE: Every step ends "   ⚡[Step] {text} - Roger!" (0 spaces after ⚡)
#   - Follow when possible, break for clarity when needed
#
# LIGHTNING POW POW RULES:
#   - ✅ Success → ⚡Pow Pow!
#   - ❌ Halting failure (returns False) → NO Pow Pow
#   - ⚠️ Non-halting failure (continues) → ⚡Pow Pow!
# ═══════════════════════════════════════════════════════════════════════

import subprocess
from pathlib import Path
from typing import Dict

from ..shared.callbacks import StatusCallback
from ..shared.api_helpers import run_gcloud_with_retry, GCloudAccumulator


def _delete_worker_pool(
    project_id: str,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Step 1/7: Delete Cloud Build Worker Pool

    Deletes: pytorch-mecha-pool (c3-standard-176 spot instances)
    """
    pool_name = "pytorch-mecha-pool"

    status(f"   ⊗  Deleting worker pool: {pool_name}...")

    try:
        result = subprocess.run(
            [
                "gcloud", "builds", "worker-pools", "delete",
                pool_name,
                f"--region={region}",
                f"--project={project_id}",
                "--quiet",  # No confirmation prompt
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            status(f"   [green]✓    Worker pool deleted: {pool_name}[/green]")
        else:
            # Idempotent: already deleted is success
            if "NOT_FOUND" in result.stderr or "does not exist" in result.stderr:
                status(f"   [green]✓    Worker pool already removed: {pool_name}[/green]")
            else:
                status(f"   [red]✗    Failed to delete worker pool[/red]")
                status(f"     Error: {result.stderr[:4000]}")
                return False  # Halting failure - NO Pow Pow

        # Success - ONE Pow Pow for the whole step
        status(f"   [cyan]⚡Worker pool teardown complete - Roger![/cyan]")
        return True

    except Exception as e:
        status(f"   [red]✗    Error deleting worker pool: {str(e)[:4000]}[/red]")
        return False  # Halting failure - NO Pow Pow


def _delete_registry(
    region: str,
    status: StatusCallback,
    delete_images: bool = True,
) -> bool:
    """
    Step 2/7: Delete Artifact Registry

    Deletes: arr-coc-registry (fast-building images: ml-stack, trainer, launcher)
    Preserves: arr-coc-registry-persistent (PyTorch base image - takes 2-4 hours to rebuild)

    Args:
        region: GCP region
        status: Status callback
        delete_images: If False, skips registry deletion entirely
    """
    # SHARED registry name (matches setup)
    registry_name = "arr-coc-registry"
    persistent_registry_name = "arr-coc-registry-persistent"

    # If delete_images=False, skip deletion entirely
    if not delete_images:
        status("   ℹ Skipping registry/image deletion (keeping all arr- images)")
        status("   [dim]Note: Both registries and pricing data preserved[/dim]")
        status(f"   [cyan]⚡Registry teardown skipped - Roger![/cyan]")
        return True

    status(f"   ⊗  Deleting registry: {registry_name}...")

    try:
        result = subprocess.run(
            [
                "gcloud", "artifacts", "repositories", "delete",
                registry_name,
                f"--location={region}",
                "--quiet",  # No confirmation prompt
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            status(f"   [green]✓    Registry deleted: {registry_name}[/green]")
        else:
            # Idempotent: already deleted is success
            if "NOT_FOUND" in result.stderr or "does not exist" in result.stderr:
                status(f"   [green]✓    Registry already removed: {registry_name}[/green]")
            else:
                status(f"   [red]✗    Failed to delete registry[/red]")
                status(f"     Error: {result.stderr[:4000]}")
                return False  # Halting failure - NO Pow Pow

        # Show persistent registry info
        status("")
        status("   [yellow]⚠️  Persistent Registry NOT Deleted:[/yellow]")
        status(f"      Registry: {persistent_registry_name} (us-central1)")
        status("      Contains: arr-pytorch-base (~15GB)")
        status("      Reason: Prevents 2-4 hour rebuild on next launch")
        status("")
        status("   [dim]To delete manually (if needed):[/dim]")
        status(f"      [dim]gcloud artifacts repositories delete {persistent_registry_name} \\[/dim]")
        status(f"      [dim]  --location=us-central1 \\[/dim]")
        status(f"      [dim]  --project=$(gcloud config get-value project)[/dim]")
        status("")

        # Success - ONE Pow Pow for the whole step
        status(f"   [cyan]⚡Registry teardown complete - Roger![/cyan]")
        return True

    except Exception as e:
        status(f"   [red]✗    Error deleting registry: {str(e)[:4000]}[/red]")
        return False  # Halting failure - NO Pow Pow


def _delete_staging_buckets(
    config: Dict[str, str],
    project_id: str,
    status: StatusCallback,
) -> bool:
    """
    Step 3/7: Delete GCS Staging Buckets

    Deletes:
    - gs://{project_id}-staging (SHARED - W&B Launch)
    - gs://{project_id}-{project_name}-staging (project-specific)
    """
    project_name = config.get("PROJECT_NAME", "arr-coc")

    # W&B staging bucket (SHARED across all W&B Launch jobs)
    wandb_staging = f"gs://{project_id}-staging"

    # Project-specific staging bucket
    project_staging = f"gs://{project_id}-{project_name}-staging"

    buckets = {"wandb": wandb_staging, "project": project_staging}

    # Use accumulator: Delete both buckets in parallel
    # Sequential: 2 × 30s = 60s | Accumulator: Both at once = 30s
    acc = GCloudAccumulator(max_workers=2)

    # Start both deletions (non-blocking!)
    for key, bucket in buckets.items():
        status(f"   ⊗  Deleting bucket: {bucket}...")
        acc.start(
            key=key,
            cmd=["gsutil", "-m", "rm", "-r", bucket],
            max_retries=3,
            timeout=120,
            operation_name=f"delete {key} bucket",
        )

    # Progressive rendering - show each deletion as it completes!
    deletion_failed = False

    def render_deletion_result(key, result):
        nonlocal deletion_failed
        bucket = buckets[key]

        try:
            if result.returncode == 0:
                status(f"   [green]✓    Bucket deleted: {bucket}[/green]")
            else:
                # Idempotent: already deleted is success
                if "BucketNotFoundException" in result.stderr:
                    status(f"   [green]✓    Bucket already removed: {bucket}[/green]")
                else:
                    status(f"   [red]✗    Failed to delete bucket[/red]")
                    status(f"     Error: {result.stderr[:4000]}")
                    deletion_failed = True

        except Exception as e:
            status(f"   [red]✗    Error deleting bucket: {str(e)[:4000]}[/red]")
            deletion_failed = True

    # Automatic progressive rendering!
    acc.wait_and_render(render_deletion_result)
    acc.shutdown()

    if deletion_failed:
        return False  # Halting failure - NO Pow Pow

    # Success - ONE Pow Pow for the whole step
    status(f"   [cyan]⚡Staging buckets teardown complete - Roger![/cyan]")
    return True


def _delete_checkpoints_bucket(
    config: Dict[str, str],
    project_id: str,
    status: StatusCallback,
) -> bool:
    """
    Step 4/7: Delete GCS Checkpoints Bucket

    Deletes: gs://{project_id}-{project_name}-checkpoints (project-specific)
    """
    project_name = config.get("PROJECT_NAME", "arr-coc")
    bucket = f"gs://{project_id}-{project_name}-checkpoints"

    status(f"   ⊗  Deleting bucket: {bucket}...")

    try:
        result = run_gcloud_with_retry(
            ["gsutil", "-m", "rm", "-r", bucket],
            max_retries=3,
            timeout=120,
            operation_name="delete checkpoints bucket",
        )

        if result.returncode == 0:
            status(f"   [green]✓    Bucket deleted: {bucket}[/green]")
        else:
            # Idempotent: already deleted is success
            if "BucketNotFoundException" in result.stderr:
                status(f"   [green]✓    Bucket already removed: {bucket}[/green]")
            else:
                status(f"   [red]✗    Failed to delete bucket[/red]")
                status(f"     Error: {result.stderr[:4000]}")
                return False  # Halting failure - NO Pow Pow

        # Success - ONE Pow Pow for the whole step
        status(f"   [cyan]⚡Checkpoints bucket teardown complete - Roger![/cyan]")
        return True

    except Exception as e:
        status(f"   [red]✗    Error deleting bucket: {str(e)[:4000]}[/red]")
        return False  # Halting failure - NO Pow Pow


def _remove_iam_bindings(
    config: Dict[str, str],
    project_id: str,
    status: StatusCallback,
) -> bool:
    """
    Step 5/7: Remove IAM Role Bindings

    Removes:
    - 4 SA roles (Vertex AI, Storage, AR, Logging)
    - 2 Cloud Build roles (Network, Compute)
    """
    project_name = config.get("PROJECT_NAME", "arr-coc")
    sa_name = f"{project_name}-sa"
    sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"

    status(f"   ⊗  Removing IAM role bindings...")

    # Service Account roles
    sa_roles = [
        "roles/aiplatform.user",
        "roles/storage.objectAdmin",
        "roles/artifactregistry.writer",
        "roles/logging.logWriter",
    ]

    try:
        # Get project number for Cloud Build SA
        project_number = run_gcloud_with_retry(
            [
                "gcloud", "projects", "describe",
                project_id,
                "--format=value(projectNumber)",
            ],
            max_retries=3,
            timeout=10,
            operation_name="get project number",
        )

        # Use accumulator: Remove all IAM roles in parallel
        # IAM *removal* is safe to parallelize (unlike *grants* which have read-modify-write conflicts)
        # Sequential: 6 roles × 10s = 60s | Accumulator: All at once = 10s
        acc = GCloudAccumulator(max_workers=6)

        # Start all SA role removals (non-blocking!)
        for role in sa_roles:
            acc.start(
                key=f"sa-{role}",
                cmd=[
                    "gcloud", "projects", "remove-iam-policy-binding",
                    project_id,
                    "--member", f"serviceAccount:{sa_email}",
                    "--role", role,
                    "--quiet",
                ],
                max_retries=3,
                timeout=30,
                operation_name=f"remove SA role {role}",
            )

        # Start Cloud Build role removals (if project number available)
        if project_number.returncode == 0:
            project_num = project_number.stdout.strip()
            cloudbuild_sa = f"{project_num}@cloudbuild.gserviceaccount.com"

            cloudbuild_roles = [
                "roles/compute.networkUser",
                "roles/compute.admin",
            ]

            for role in cloudbuild_roles:
                acc.start(
                    key=f"cb-{role}",
                    cmd=[
                        "gcloud", "projects", "remove-iam-policy-binding",
                        project_id,
                        "--member", f"serviceAccount:{cloudbuild_sa}",
                        "--role", role,
                        "--quiet",
                    ],
                    max_retries=3,
                    timeout=30,
                    operation_name=f"remove Cloud Build role {role}",
                )

        # Get all results (waits if not ready - but likely already done!)
        for role in sa_roles:
            acc.get(f"sa-{role}")  # Ignoring result - removal is idempotent

        status(f"   [green]✓    Removed 4 SA role bindings[/green]")

        if project_number.returncode == 0:
            for role in cloudbuild_roles:
                acc.get(f"cb-{role}")  # Ignoring result - removal is idempotent
            status(f"   [green]✓    Removed 2 Cloud Build role bindings[/green]")
        else:
            status(f"   [yellow]⚠️  Could not determine project number for Cloud Build SA removal[/yellow]")

        acc.shutdown()

        status(f"   [cyan]⚡IAM bindings removed - Roger![/cyan]")
        return True

    except Exception as e:
        status(f"   [red]✗    Error removing IAM bindings: {str(e)[:4000]}[/red]")
        return False


def _teardown_pricing(status: StatusCallback) -> bool:
    """
    Step 6/6: Teardown Pricing Infrastructure

    Deletes:
    - Cloud Scheduler: arr-coc-pricing-scheduler
    - Cloud Function: arr-coc-pricing-runner
    - OIDC permissions
    - Cloud Billing API

    Preserves:
    - Artifact Registry: arr-coc-pricing repository (historical pricing data)
    """
    # Header already shown by core.py - just execute the teardown steps
    try:
        from CLI.teardown.pricing_teardown import teardown_pricing_infrastructure
        teardown_pricing_infrastructure(status)
        return True

    except Exception as e:
        status(f"   [red]✗    Error tearing down pricing infrastructure: {str(e)[:4000]}[/red]")
        return False


def _delete_single_image(
    project_id: str,
    region: str,
    image_name: str,
    status: StatusCallback,
) -> bool:
    """
    Delete a single Docker image from Artifact Registry.

    Args:
        project_id: GCP project ID
        region: Registry region (us-central1)
        image_name: Image keyword to delete (ARR-PYTORCH-BASE, ARR-ML-STACK, ARR-TRAINER, ARR-VERTEX-LAUNCHER)
        status: Status callback

    Returns:
        True if deletion succeeded (or image didn't exist)
        False if deletion failed
    """
    import subprocess

    # Map keyword to actual image name
    image_map = {
        "ARR-PYTORCH-BASE": "arr-pytorch-base",
        "ARR-ML-STACK": "arr-ml-stack",
        "ARR-TRAINER": "arr-trainer",
        "ARR-VERTEX-LAUNCHER": "arr-vertex-launcher",
    }

    actual_name = image_map.get(image_name, image_name)
    registry_name = "arr-coc-registry"
    image_path = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/{actual_name}"

    status(f"   ⊕  Deleting image: {actual_name}...")

    try:
        # List all tags for this image
        result = subprocess.run(
            ["gcloud", "artifacts", "docker", "images", "list",
             f"{region}-docker.pkg.dev/{project_id}/{registry_name}",
             "--filter", f"package={actual_name}",
             "--format", "value(version)"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            status(f"   ⚠  Image {actual_name} not found (already deleted?)")
            return True  # Not an error

        tags = result.stdout.strip().split('\n') if result.stdout.strip() else []

        if not tags:
            status(f"   ℹ Image {actual_name} has no tags (already deleted?)")
            return True

        # Delete each tag
        for tag in tags:
            tag_path = f"{image_path}:{tag}"
            status(f"   ⊕  Deleting tag: {tag}...")

            delete_result = subprocess.run(
                ["gcloud", "artifacts", "docker", "images", "delete",
                 tag_path, "--quiet"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if delete_result.returncode != 0:
                status(f"   ✗    Failed to delete {tag}: {delete_result.stderr[:400]}")
                return False

        status(f"   ✓    Image deleted: {actual_name} ({len(tags)} tags)")
        return True

    except subprocess.TimeoutExpired:
        status(f"   ✗    Timeout deleting {actual_name}")
        return False
    except Exception as e:
        status(f"   ✗    Error deleting {actual_name}: {str(e)[:400]}")
        return False
