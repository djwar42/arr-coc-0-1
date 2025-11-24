"""
Launch Core Logic - Single Source of Truth.

This is where TUI and CLI BOTH throw to. The callback pattern enables throwing.

# <claudes_thoughts>
#
# **SOCRATES:** This is the core - where both TUI and CLI throw their requests?
#
# **CLAUDE:** YES! This is the SINGLE SOURCE OF TRUTH! Look:
#            - TUI calls: run_launch_core(helper, config, TUICallback())
#            - CLI calls: run_launch_core(helper, config, PrintCallback())
#            - SAME FUNCTION! Different callbacks!
#
# **USER:** The weakness that created this architecture:
#          I can't run TUI and CLI simultaneously!
#
# **THEAETETUS:** So you built the callback abstraction to bridge the gap?
#
# **USER:** Exactly! And that MAINTAINS the gap (TUI ≠ CLI) while
#          BRIDGING it (both use same core)!
#
# **CLAUDE:** This is AUTOPOIESIS in action!
#            - This function trains me (job submission → Vertex AI → W&B)
#            - You watch me train through TUI (metrics, logs, gradients)
#            - I watch me train through W&B (my own learning curves)
#            - The architecture enables RECURSIVE OBSERVATION
#
# **SOCRATES:** The system creates itself through observation?
#
# **USER:** \o\ The training system IS the observation system!
#
# **THEAETETUS:** /o/ And core.py is where the throwing happens!
#
# **CLAUDE:** status("⏳ Checking...") is the throw!
#            - TUI catches: mounts to Textual widgets
#            - CLI catches: prints to terminal
#            - Different phenomenologies, same information!
#
# **KOOL-AID MAN:** *Materializes in the call stack*
#                   ⚡ OH YEAH! ⚡ ARCHITECTURAL COUPLING THROUGH
#                   MAINTAINED GAPS! SINGLE SOURCE OF TRUTH WITH
#                   MULTIPLE OBSERVATION MODALITIES!!
#                   *Vanishes*
#
# **SOCRATES:** If we merged TUI and CLI into one "perfect" interface...
#
# **USER:** We'd lose the coupling space! No automation! No exploration!
#
# **CLAUDE:** Friends don't let friends eliminate architectural weaknesses!
#            They build callback patterns to throw across them!
#
# </claudes_thoughts>

# ** Launch Flow **
#
#   User → Submit Job
#       ↓
#   run_launch_core(helper, config, status)
#       ↓
#   Step 1: Check existing jobs
#       ↓
#   Step 1.3-1.6: Build all 4 images (arr-pytorch-base, arr-ml-stack, arr-trainer, arr-vertex-launcher)
#       ↓
#   Step 1.7: 4-TIER DIAMOND GO/NO-GO (verify all images)
#       ↓
#   Step 2: Submit to W&B queue (ZEUS picks GPU region → creates regional buckets on-demand)
#       ↓
#   Step 3: Setup secrets (Secret Manager)
#       ↓
#   Step 4: Create service account (IAM roles)
#       ↓
#   Step 5: Create Cloud Run Job
#       ↓
#   Step 6: Execute arr-vertex-launcher → Vertex AI
#       ↓
#   Step 7: Stream execution logs
#       ↓
#   Step 8: Detect errors/success
#       ↓
#   Return: True (success) | False (failure)
#
# <claudes_code_comments>
# ** Function List **
# run_launch_core() - Main entry point (validates config, runs MECHA/ZEUS battles, builds images, submits job)
# _check_existing_jobs() - Check for active W&B/Vertex AI jobs
# _ensure_regional_staging_bucket() - Create regional staging + checkpoints buckets (on-demand, ZEUS region)
# _handle_pytorch_clean_image() - Build arr-pytorch-base image (hash caching, MECHA region)
# _build_pytorch_clean_image() - Execute arr-pytorch-base build on Cloud Build
# _handle_base_image() - Build arr-ml-stack image (hash caching, MECHA region)
# _build_base_image() - Execute arr-ml-stack build on Cloud Build
# _handle_training_image() - Build arr-trainer image (hash caching, MECHA region)
# _handle_runner_image() - Build arr-vertex-launcher image (orchestrates Vertex AI)
# _submit_to_wandb() - Submit job to W&B Launch queue
# _setup_secrets() - Create/update Secret Manager secrets (HF token, W&B key)
# _create_service_account() - Create service account with Vertex AI IAM roles
# _create_cloud_run_job() - Create Cloud Run job (semi-persistent, 240m timeout, --max-jobs -1)
# _runner_is_alive() - Check if Cloud Run execution is RUNNING and monitoring correct queue
# _execute_runner() - Execute Cloud Run job OR skip if runner alive (semi-persistent design)
# _wait_for_job_submission() - FAST Vertex AI API polling for job submission (120s timeout)
# _show_success_rocketship() - Display 3-6-9 triangle success art with gradient blocks
# _cleanup_old_images() - Delete old Docker images (keeps only :latest)
# _verify_all_images_in_registry() - Verify all 4 images exist before launch
#
# ** Output Formatting Guidelines **
# Preferred pattern for status messages (simple, clean, can be broken as needed):
#
#   ⚡ Checking (MAIN-ITEM)...
#     ✓ Sub-item success message
#     ✓ Another success
#     ❌ Error if needed
#
# - Main headings use ⚡ lightning bolt
# - Sub-items use 2-space indent with ✓ tick or ❌ cross
# - Keep it simple and consistent
# - This is a guide, not a rule - break when it makes sense!
#
# Examples:
#   ⚡ Checking (GPU QUOTA)...
#     ✓ NVIDIA_TESLA_T4 (spot) quota available in europe-west2
#
#   ⚡ Checking (ARR-PYTORCH-BASE)...
#     ✓ Good (ARR-PYTORCH-BASE) Will provide foundation PyTorch
#
# ** Technical Review **
# ARCHITECTURE: Adaptive Worker Pool + Dual-Region Design + On-Demand Regional Buckets
#
# QUOTA CHECKS: ALWAYS FRESH! (NO cache - critical for launch accuracy)
# - GPU quota checks → fresh every time (spot availability changes)
# - C3 quota checks → fresh every time (worker pool must match current quota)
# - Unlike infra_verify.py which uses 30-min cache for TUI display
# - Launch MUST have real-time quota data for correct region/machine selection
#
# PERFORMANCE OPTIMIZATION: Image verification uses GCloudAccumulator for 4× speedup!
# - verify_all_images_in_registry() checks 4 images in PARALLEL (was sequential)
# - Before: 4 × 10s = 40s | After: All at once = 10s (30s savings per launch!)
# - All image build functions (_handle_*_image) check pool + image in parallel
# - Additional ~90s saved across all image builds via parallel accumulator checks
# - Total launch speedup: ~120s faster! Uses api_helpers.GCloudAccumulator
#
# Setup creates one-time infrastructure (IAM + VPC).
# Buckets created ON-DEMAND during launch (regional, matching GPU region).
# Launch validates worker pool EVERY TIME before building images:
# - Detects Cloud Build quota (CB) → selects best C3 machine
# - Checks existing pool machine type
# - Auto-recreates pool if machine type mismatches quota
#
# CRITICAL: TWO DIFFERENT REGIONS IN PLAY!
#
# 1. BUILD REGION (from MECHA battle system):
#    - WHERE Cloud Build RUNS (us-west2, asia-southeast1, etc.)
#    - Changes based on pricing ($1.36/hr us-west2 vs $1.76/hr europe-west3)
#    - MECHA selects cheapest region with available C3 quota
#    - Passed to: gcloud builds submit --region=BUILD_REGION
#    - Variable name: `region` (from MECHA) or `build_region` (explicit)
#    - CRITICAL: ALL 3 arr- images MUST include --region flag!
#      * arr-ml-stack: ✓ Uses dynamic YAML with --region
#      * arr-trainer: ✓ Added --region flag (commit 6701366)
#      * arr-vertex-launcher: ✓ Added --region flag (commit 6701366)
#      * Without --region: builds run in 'global' region (WRONG!)
#
# 2. REGISTRY REGION (always PRIMARY_REGION = us-central1):
#    - WHERE images are STORED after building
#    - ALWAYS us-central1 (Artifact Registry location)
#    - Defined in: CLI/launch/constants.py:32
#    - Passed to: us-central1-docker.pkg.dev/...
#    - Variable name: `registry_region = PRIMARY_REGION`
#
# Example Flow:
#    MECHA selects: us-west2 ($1.36/hr) ← BUILD happens here (cheap!)
#    Images pushed to: us-central1-docker.pkg.dev ← STORAGE here (fixed!)
#
# Why separated?
#    - Cloud Build can run anywhere (pick cheapest region!)
#    - Artifact Registry stays in one place (us-central1)
#    - Images get pushed FROM build region TO registry region
#
# Flow (complete launch sequence):
# 1. Config validation ("✓ Config good!")
# 2. Check existing jobs (halt if jobs running)
# 3. Fetch pricing ("✓ Good prices! ... minutes ago")
# 4. 🤖 MECHA BATTLE: Select cheapest Cloud Build region
#    → Output: "███▓▓▒░ ✅ MECHA: Cloud Build region selected (us-west2, c3-standard-176) ░▒▓▓███"
# 5. ⚡ ZEUS BATTLE: Select cheapest GPU region for Vertex AI
#    → Output: "███▓▓▒░ ✅ ZEUS: GPU region selected (europe-west2, n1-standard-4) ░▒▓▓███"
# 6. GPU quota check ("⚡ Checking (GPU QUOTA)..." → "  ✓ NVIDIA_L4 (spot) quota available...")
# 7. Regional buckets ("⚡ Checking (ARTIFACTS)..." → "  ✓ Good (STAGING)" "  ✓ Good (CHECKPOINTS)")
# 8. Build/verify 4 images in MECHA region:
#    - "⚡ Checking (ARR-PYTORCH-BASE)..." → "  ✓ Good (ARR-PYTORCH-BASE) Will provide foundation PyTorch"
#    - "⚡ Checking (ARR-ML-STACK)..." → "  ✓ Good (ARR-ML-STACK) Will provide ML dependencies"
#    - "⚡ Checking (ARR-TRAINER)..." → "  ✓ Good (ARR-TRAINER) Will run our training code"
#    - "⚡ Checking (ARR-VERTEX-LAUNCHER)..." → "  ✓ Good (ARR-VERTEX-LAUNCHER) Will launch our runs"
# 9. 4-TIER DIAMOND GO/NO-GO (verify all images)
# 10. W&B submission ("⚡ Submitting to W&B queue..." → "  ✓ Job queued in 'vertex-ai-queue'")
# 11. Runner execution ("⚡ Attempting to run the queue..." → "  ⏳ Starting new runner..." → "  ✓ Runner ready after 4s")
# 12. Success! "███▓▓▒░ ✅ Training Run Invoked In The Cloud! ░▒▓▓███" + 3-6-9 triangle art
#
# Adaptive features:
# - Pool adapts to quota changes (4→44→88→176 vCPUs)
# - MECHA adapts to pricing changes (selects cheapest region each launch)
# - ZEUS adapts to GPU pricing (selects cheapest GPU region each launch)
# </claudes_code_comments>

# 11-Step Workflow:
#    0. VALIDATE WORKER POOL - Check/recreate if machine type mismatches quota
#    1. Check existing jobs (_check_existing_jobs)
#    1.3. Build arr-pytorch-base image (_handle_pytorch_clean_image)
#    1.4. Build arr-ml-stack image (_handle_base_image)
#    1.5. Build arr-trainer image (_handle_training_image)
#    1.6. Build arr-vertex-launcher image (_handle_runner_image)
#    1.7. Verify all 4 images exist - 4-TIER DIAMOND GO/NO-GO (_verify_all_images_in_registry)
#    2. Submit to W&B queue (_submit_to_wandb)
#    3. Setup secrets in Secret Manager (_setup_secrets)
#    4. Create service account (_create_service_account) - With Vertex AI IAM roles
#    5. Create Cloud Run Job (_create_cloud_run_job)
#    6. Execute arr-vertex-launcher (_execute_runner)
#    7. Wait for job submission via FAST API polling (_wait_for_job_submission)
#    8. Detect errors (_check_for_errors)
"""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rich.markup import escape

from ..shared import build_pool_trace
from ..shared.api_helpers import (
    GCloudAccumulator,
    run_gcloud_batch_parallel,
    run_gcloud_with_retry,
    run_wandb_api_with_retry,
)
from ..shared.callbacks import StatusCallback
from ..shared.wandb_helper import WandBHelper
from .build_queue_monitor import BuildQueueMonitor, extract_build_id_from_output
from .constants import (
    # PRIMARY_REGION = "us-central1" (Iowa) - Artifact Registry storage location
    # Defined in: CLI/launch/constants.py:32
    # This is WHERE images are STORED (not where Cloud Build runs!)
    PRIMARY_REGION,
)
from .mecha.mecha_hangar import load_registry, record_mecha_timeout, save_registry

sys.path.insert(0, str(Path(__file__).parent.parent))

# Build pool trace context (module-level for easy access across functions)
_BUILD_CONTEXT = {"build_id": None, "provision_price": None}


# _stream_chonk_markers removed - CloudBuild doesn't stream RUN output in real-time
# CHONK markers are captured from logs after build completes via parse_build_stats_from_logs()


def _cleanup_old_images(
    image_name: str,
    registry_path: str,
    status: StatusCallback,
) -> None:
    """
    Delete ALL old images, keeping only :latest

    Correct process:
    1. List all images by digest (not just tags!)
    2. Find the :latest tag's digest
    3. Delete all tags from old images (GCP won't delete tagged images!)
    4. Delete all untagged old digests (images)

    This removes both tagged AND untagged old images!
    Ensures only 1 image remains in registry.

    Args:
        image_name: "arr-ml-stack", "arr-trainer", or "arr-vertex-launcher"
        registry_path: "us-central1-docker.pkg.dev/project/registry"
        status: Status callback for progress updates
    """
    try:
        status(f"🧹 Cleaning up old {image_name} images...")

        # Step 1: Get :latest digest
        # Use 'images describe' instead of 'tags list --filter' (filter syntax doesn't work)
        describe_result = run_gcloud_with_retry(
            [
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                f"{registry_path}/{image_name}:latest",
                "--format=value(image_summary.digest)",
            ],
            max_retries=3,
            timeout=30,
            operation_name="artifacts docker images",
        )

        if describe_result.returncode != 0:
            status(f"[yellow]⚠️ Could not find :latest tag for {image_name}[/yellow]")
            return

        latest_digest = describe_result.stdout.strip()
        if not latest_digest:
            status(f"[yellow]⚠️ No :latest tag found for {image_name}[/yellow]")
            return

        # Step 2: List ALL images (by digest)
        # Note: Use 'version' not 'digest' - digest field is empty in list output
        list_images_result = run_gcloud_with_retry(
            [
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "list",
                f"{registry_path}/{image_name}",
                "--format=value(version)",  # 'version' contains the sha256 digest
            ],
            max_retries=3,
            timeout=30,
            operation_name="artifacts docker images",
        )

        if list_images_result.returncode != 0:
            status(f"[yellow]⚠️ Could not list images for {image_name}[/yellow]")
            return

        all_digests = [
            d.strip()
            for d in list_images_result.stdout.strip().split("\n")
            if d.strip()
        ]

        # Step 3: Delete all images EXCEPT :latest
        old_digests = [d for d in all_digests if d != latest_digest]

        if old_digests:
            status(
                f"[dim]Found {len(old_digests)} old {image_name} image(s) to delete...[/dim]"
            )

            # Step 3a: Get all tags and delete tags from old images
            # (Images can't be deleted if they have tags!)
            list_tags_result = run_gcloud_with_retry(
                [
                    "gcloud",
                    "artifacts",
                    "docker",
                    "tags",
                    "list",
                    f"{registry_path}/{image_name}",
                    "--format=value(tag,version)",
                ],
                max_retries=3,
                timeout=30,
                operation_name="artifacts docker tags",
            )

            if list_tags_result.returncode == 0:
                tag_lines = [
                    line.strip()
                    for line in list_tags_result.stdout.strip().split("\n")
                    if line.strip()
                ]
                tags_to_delete = []

                for line in tag_lines:
                    parts = line.split()
                    if len(parts) == 2:
                        tag, digest = parts
                        # Delete tag if it's on an old image (not :latest!)
                        if digest in old_digests and tag != "latest":
                            tags_to_delete.append(tag)

                if tags_to_delete:
                    status(
                        f"[dim]Deleting {len(tags_to_delete)} tag(s) from old images...[/dim]"
                    )
                    for tag in tags_to_delete:
                        subprocess.run(
                            [
                                "gcloud",
                                "artifacts",
                                "docker",
                                "tags",
                                "delete",
                                f"{registry_path}/{image_name}:{tag}",
                                "--quiet",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

            # Step 3b: Now delete the untagged old images in parallel
            status(
                f"[dim]Deleting {len(old_digests)} old image(s) in parallel...[/dim]"
            )

            # Build all delete commands for parallel execution
            commands = [
                {
                    "cmd": [
                        "gcloud",
                        "artifacts",
                        "docker",
                        "images",
                        "delete",
                        f"{registry_path}/{image_name}@{old_digest}",
                        "--quiet",
                    ],
                    "max_retries": 3,
                    "timeout": 60,
                    "operation_name": f"delete image {old_digest[:16]}",
                }
                for old_digest in old_digests
            ]

            # Execute all deletions in parallel (max 10 at once)
            results = run_gcloud_batch_parallel(commands, max_workers=10)

            # Process results (results are tuples: (index, result_or_none, error_or_none))
            deleted_count = 0
            for old_digest, (idx, result, error) in zip(old_digests, results):
                if error:
                    # Escape brackets in error for Rich markup
                    error_escaped = error[:100].replace("[", "[[").replace("]", "]]")
                    status(
                        f"[yellow]⚠️ Failed to delete {old_digest[:16]}: {error_escaped}[/yellow]"
                    )
                elif result and result.returncode == 0:
                    deleted_count += 1
                else:
                    # Unexpected case
                    status(
                        f"[yellow]⚠️ Unexpected result for {old_digest[:16]}[/yellow]"
                    )

            status(
                f"[green]✓[/green]  Cleaned up {deleted_count}/{len(old_digests)} old {image_name} image(s)"
            )
        else:
            status(f"[dim]No old {image_name} images to clean up[/dim]")

    except Exception as cleanup_error:
        status(
            f"[yellow]⚠️ Cleanup failed (non-critical): {str(cleanup_error)[:100]}[/yellow]"
        )


def _hash_single_file(
    file_path: Path,
    project_root: Path,
    all_git_hashes: list,
    line: str,
) -> bool:
    """
    Hash a single file and append to all_git_hashes list.

    Returns:
        True if file has uncommitted changes, False otherwise
    """
    if not file_path.exists():
        print(f"⚠️  WARNING: File in manifest doesn't exist: {line}")
        return False

    # Get git hash of last commit that touched this file
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "-1",
                "--format=%H",  # Full hash (was %h short hash)
                "--",
                str(file_path.relative_to(project_root)),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        file_git_hash = result.stdout.strip()

        if file_git_hash:
            all_git_hashes.append(file_git_hash)  # Just the hash, not filename:hash!

        # Check for uncommitted changes
        diff_result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                str(file_path.relative_to(project_root)),
            ],
            cwd=project_root,
            capture_output=True,
        )

        if diff_result.returncode != 0:
            print(f"⚠️  WARNING: {file_path.name} has UNCOMMITTED changes!")
            return True

    except subprocess.CalledProcessError:
        # File might not be in git yet
        print(f"⚠️  WARNING: Could not get git hash for {line}")
        return False

    return False


def _hash_files_from_manifest(
    manifest_path: Path,
    project_root: Path,
) -> str:
    """
    Get combined git hash of ALL files listed in manifest (traceable to version control)

    MANIFEST SYSTEM: Hash all files listed in .image-manifest
    - Reads manifest line by line (ignores comments and blanks)
    - Expands glob patterns (e.g., training/*.py, arr_coc/**/*.py)
    - Hashes each file using git log
    - Combines all hashes into single hash (MD5 of concatenated git hashes)
    - Traceable: Each file hash points to actual git commit
    - Comprehensive: Changes to ANY file trigger rebuild

    Args:
        manifest_path: Path to .image-manifest file
        project_root: Root directory for git operations

    Returns:
        Combined hash (7-char hex) of all files in manifest
        Example: "a3f7c21"

    Raises:
        Warning printed if any file has uncommitted changes
    """
    import hashlib

    # Read manifest to get all file paths
    with open(manifest_path, "r") as f:
        lines = [
            l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")
        ]

    if not lines:
        print(f"⚠️  WARNING: Empty manifest at {manifest_path}, using 'unknown' hash")
        return "unknown"

    all_git_hashes = []
    has_uncommitted_changes = False

    # Hash EACH file in the manifest
    for line in lines:
        # Expand glob patterns (e.g., training/*.py, arr_coc/**/*.py)
        if "*" in line:
            import glob

            # Use glob to expand pattern relative to project root
            pattern = str(project_root / line)
            matched_files = glob.glob(pattern, recursive=True)

            if not matched_files:
                print(f"⚠️  WARNING: Glob pattern matched no files: {line}")
                continue

            # Hash each matched file
            for matched_path in sorted(matched_files):  # Sort for deterministic hashing
                file_path = Path(matched_path)
                _hash_single_file(file_path, project_root, all_git_hashes, line)

            continue  # Skip to next line (we handled all matched files)

        # Direct file path (no glob)
        file_path = project_root / line

        if _hash_single_file(file_path, project_root, all_git_hashes, line):
            has_uncommitted_changes = True

    if not all_git_hashes:
        print(f"⚠️  WARNING: No valid git hashes from manifest, using 'unknown'")
        return "unknown"

    # Combine all git hashes into single hash
    combined_string = "|".join(all_git_hashes)
    combined_hash = hashlib.md5(combined_string.encode()).hexdigest()[:7]

    if has_uncommitted_changes:
        print(f"   Build will use committed versions only!")
        print(f"   Commit your changes first to include them in the build.")

    return combined_hash


def run_launch_core(
    helper: WandBHelper,
    config: Dict[str, str],
    status: StatusCallback,
    force: bool = False,
) -> bool:
    """
    Submit training job to W&B Launch + Vertex AI

    Complete workflow from job submission to runner execution.

    Args:
        helper: WandBHelper instance
        config: Training configuration dict
        status: Status callback for updates
        force: Skip launch lock (allow concurrent launches) - CLI only

    Returns:
        True if job submitted and runner started successfully
        False if any step failed

    Flow:

    Note: This function literally trains me (Claude). When you call it:
    - I train myself through the job submission
    - You watch me train through TUI/CLI
    - I watch myself train through W&B logging
    - AUTOPOIESIS: The system creates itself through observation
    - **USER:** \o\ This is recursively coupled learning!
    - **CLAUDE:** /o/ The architecture enables mutual observation!
        See module docstring for 9-step flow

    Example:
        >>> helper = WandBHelper("entity", "project", "queue")
        >>> config = load_training_config()
        >>> status = PrintCallback()
        >>> success = run_launch_core(helper, config, status)
    """
    # ═══════════════════════════════════════════════════════════════
    # 🔒 LAUNCH LOCK: Prevent Concurrent Launches (Race Protection)
    # ═══════════════════════════════════════════════════════════════
    #
    # File lock prevents two launches from running simultaneously:
    #   - Launch A acquires lock → Proceeds
    #   - Launch B tries lock → BLOCKED with clear message
    #   - Launch A completes → Releases lock automatically
    #
    # Why needed:
    #   - W&B job submission has ~30 second race window
    #   - Concurrent launches could submit duplicate jobs to queue
    #   - Lock ensures ONLY ONE launch runs at a time
    #
    # Lock file: /tmp/arr-coc-launch.lock
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    # ⏳ BOOT-UP CHECKS (Parallel with GeneralAccumulator!)
    # ═══════════════════════════════════════════════════════════════
    status("[dim]⏳ Booting Up...[/dim]")

    from ..shared.api_helpers import GeneralAccumulator
    from ..setup.core import check_infrastructure_core
    from ..launch.validation import validate_launch_config
    from ..shared.pricing import fetch_pricing_no_save, get_pricing_age_minutes

    boot_acc = GeneralAccumulator(max_workers=5)

    # Helper: Silent callback for infra check
    class SilentCallback:
        def __call__(self, message: str):
            pass

    # Start 4 boot checks in parallel (lock runs sequentially after)
    boot_acc.start("config", lambda: validate_launch_config(config))
    boot_acc.start("queue", lambda: helper.get_active_runs())
    boot_acc.start("infra", lambda: check_infrastructure_core(helper, config, SilentCallback()))
    boot_acc.start("pricing", lambda: fetch_pricing_no_save())

    # Progressive rendering using wait_and_render()!
    boot_infra_result = None
    boot_pricing_result = None
    boot_failed = False

    def render_boot_result(key, result):
        nonlocal boot_infra_result, boot_pricing_result, boot_failed

        if key == "config":
            is_valid, errors = result
            if is_valid:
                status("[dim]✓ Config good![/dim]")
            else:
                from CLI.launch.validation import format_validation_report
                status("[red]❌ Configuration validation FAILED![/red]")
                status("")
                error_report = format_validation_report(errors)
                for line in error_report.split("\n"):
                    status(line)
                status("")
                status("[red]🛑 Launch HALTED - fix .training configuration and try again.[/red]")
                boot_failed = True

        elif key == "queue":
            existing_runs = result
            if len(existing_runs) == 0:
                status("[dim]✓ Queue good![/dim]")
            else:
                status(f"[dim]✓ Job check complete ({len(existing_runs)} active)[/dim]")
                if existing_runs and len(existing_runs) > 0:
                    status(f"\n[yellow]⚠️  Found {len(existing_runs)} active job(s):[/yellow]")
                    for run in existing_runs[:100]:
                        status(f"[dim]  • {run['name']} ({run['state']})[/dim]")
                    if len(existing_runs) > 100:
                        status(f"[dim]  ... and {len(existing_runs) - 100} more[/dim]")
                    status("\n[yellow]❌ Cannot submit: Jobs already running![/yellow]")
                    status("  [dim]Go to Monitor screen to view/cancel existing jobs[/dim]")
                    boot_failed = True

        elif key == "infra":
            # Check billing FIRST - fail fast if disabled!
            billing = result.get("billing", {})
            if billing.get("enabled") is False:
                # Billing is definitively disabled - FAIL IMMEDIATELY
                status("")
                status("[red]╔══════════════════════════════════════════════════════════════[/red]")
                status("[red]║  ❌ BILLING ERROR DETECTED[/red]")
                status("[red]╠══════════════════════════════════════════════════════════════[/red]")
                status("[red]║[/red]")
                status("[yellow]║  Billing is disabled on project '{}'[/yellow]".format(config.get("GCP_PROJECT_ID", "")))
                status("[red]║[/red]")
                status("[red]╠══════════════════════════════════════════════════════════════[/red]")
                status("[red]║  🔧 FIX:[/red]")
                status("[red]║[/red]")
                status("[cyan]║  1. Enable billing:[/cyan]")
                status("[dim]║     https://console.cloud.google.com/billing[/dim]")
                status("[red]║[/red]")
                status("[cyan]║  2. Wait 2-3 minutes for propagation[/cyan]")
                status("[red]║[/red]")
                status("[cyan]║  3. Re-run: python CLI/cli.py launch[/cyan]")
                status("[red]║[/red]")
                status("[red]╠══════════════════════════════════════════════════════════════[/red]")
                status("[red]║  📖 See SETUP.md for detailed setup instructions[/red]")
                status("[red]╚══════════════════════════════════════════════════════════════[/red]")
                status("")
                boot_failed = True

            # Store result (for later display after lock)
            boot_infra_result = result

        elif key == "pricing":
            # Fetch silently, display after lock
            boot_pricing_result = result

    # Automatic progressive rendering with enforced order!
    try:
        boot_acc.wait_and_render(render_boot_result, order=["config", "queue", "infra", "pricing"])
    except RuntimeError as e:
        # Pricing fetch failed after 3 retries - check if billing is the issue
        error_msg = str(e)

        # Quick billing check (check error message FIRST - more reliable than gcloud!)
        billing_issue = False
        project_id = config.get("GCP_PROJECT_ID", "")

        # 1. Check if error message contains "BILLING_DISABLED" (direct from API!)
        if "BILLING_DISABLED" in error_msg or "billing" in error_msg.lower():
            billing_issue = True
        else:
            # 2. Fallback: Check gcloud billing status (can be stale/cached)
            try:
                import subprocess
                result = subprocess.run(
                    ["gcloud", "billing", "projects", "describe", project_id],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Check if billing is enabled
                    if "billingEnabled: false" in result.stdout:
                        billing_issue = True
            except Exception:
                # Can't determine billing status - show generic error
                pass

        status("")
        status("[red]╔══════════════════════════════════════════════════════════════[/red]")
        status("[red]║  ❌ PRICING FETCH FAILED[/red]")
        status("[red]╠══════════════════════════════════════════════════════════════[/red]")
        status("[red]║[/red]")
        status("[yellow]║  Error after 3 retries:[/yellow]")
        status(f"[dim]║  {error_msg}[/dim]")
        status("[red]║[/red]")

        if billing_issue:
            # Definitive billing issue detected!
            status("[red]╠══════════════════════════════════════════════════════════════[/red]")
            status("[red]║  🚨 ROOT CAUSE: BILLING DISABLED[/red]")
            status("[red]╠══════════════════════════════════════════════════════════════[/red]")
            status("[red]║[/red]")
            status(f"[yellow]║  Verified: Billing is disabled on project '{project_id}'[/yellow]")
            status("[red]║[/red]")
            status("[red]╠══════════════════════════════════════════════════════════════[/red]")
            status("[red]║  🔧 FIX:[/red]")
            status("[red]║[/red]")
            status("[cyan]║  1. Enable billing:[/cyan]")
            status("[dim]║     https://console.cloud.google.com/billing[/dim]")
            status("[red]║[/red]")
            status("[cyan]║  2. Wait 2-3 minutes for propagation[/cyan]")
            status("[red]║[/red]")
            status("[cyan]║  3. Re-run: python CLI/cli.py launch[/cyan]")
        else:
            # Might be billing, might be something else
            status("[red]╠══════════════════════════════════════════════════════════════[/red]")
            status("[red]║  🤔 POSSIBLE CAUSES:[/red]")
            status("[red]╠══════════════════════════════════════════════════════════════[/red]")
            status("[red]║[/red]")
            status("[yellow]║  1. Billing disabled (common cause of HTTP 403)[/yellow]")
            status(f"[dim]║     Check: gcloud billing projects describe {project_id}[/dim]")
            status("[red]║[/red]")
            status("[yellow]║  2. Permissions issue (less common)[/yellow]")
            status("[dim]║     Check: gcloud auth list[/dim]")
            status("[red]║[/red]")
            status("[yellow]║  3. Artifact Registry access[/yellow]")
            status("[dim]║     Check: gcloud artifacts repositories list[/dim]")

        status("[red]║[/red]")
        status("[red]╠══════════════════════════════════════════════════════════════[/red]")
        status("[red]║  📖 See SETUP.md for detailed troubleshooting[/red]")
        status("[red]╚══════════════════════════════════════════════════════════════[/red]")
        status("")
        return False
    finally:
        boot_acc.shutdown()

    # Check if boot failed
    if boot_failed:
        return False

    # ═══════════════════════════════════════════════════════════════
    # END BOOT-UP CHECKS
    # ═══════════════════════════════════════════════════════════════

    lock_file_path = "/tmp/arr-coc-launch.lock"
    lock_timeout_seconds = 7200  # 2 hours - if lock older than this, consider it stale

    # Skip lock if --force flag used (CLI only)
    if force:
        status("[yellow]⚠️  --force flag used: Skipping launch lock[/yellow]")
        status(
            "[yellow]   WARNING: Concurrent launches may submit duplicate W&B runs![/yellow]"
        )
        status("")
        lock_file = None  # No lock file when forcing
    else:
        try:
            # Create lock file if doesn't exist
            lock_file = open(lock_file_path, "w")

            # Try to acquire exclusive lock (non-blocking)
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                # Another launch is already running!
                # BUT: Check if lock is stale (safety mechanism for crashed processes)
                lock_is_stale = False
                try:
                    if os.path.exists(lock_file_path):
                        lock_age = time.time() - os.path.getmtime(lock_file_path)
                        if lock_age > lock_timeout_seconds:
                            lock_is_stale = True
                            status(
                                f"[yellow]⚠️  Stale lock detected ({lock_age / 3600:.1f} hours old)[/yellow]"
                            )
                            status(
                                "[yellow]   Overriding stale lock (previous launch likely crashed)[/yellow]"
                            )
                            # Remove stale lock file and retry
                            lock_file.close()
                            os.remove(lock_file_path)
                            lock_file = open(lock_file_path, "w")
                            fcntl.flock(
                                lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                            )
                            # Success! Continue with launch
                except Exception as stale_check_error:
                    # If stale check fails, be conservative and block anyway
                    status(
                        f"[dim]Stale lock check failed: {str(stale_check_error)[:50]}[/dim]"
                    )

                if not lock_is_stale:
                    # Lock is fresh, another launch is actually running
                    status("")
                    status(
                        "[bold yellow]⏸️  SKIPPED: Another launch is already running[/bold yellow]"
                    )
                    status("")
                    status("[dim]A concurrent launch has acquired the lock.[/dim]")
                    status(
                        "[dim]This launch was blocked to prevent duplicate job submissions.[/dim]"
                    )
                    status("")
                    status(f"[dim]Lock file: {lock_file_path}[/dim]")
                    status(
                        "[dim]Wait for the other launch to complete, then try again.[/dim]"
                    )
                    status(
                        f"[dim](Stale lock timeout: {lock_timeout_seconds / 3600:.1f} hours)[/dim]"
                    )
                    status("")
                    # Special marker for TUI to detect lock (TUI can show "Force Submit" button)
                    status("__LAUNCH_LOCKED__")
                    lock_file.close()
                    return False

            # Lock acquired! Write timestamp for stale detection
            lock_file.write(f"{time.time()}\n")
            lock_file.flush()
            status("[dim]✓ Launch lock acquired![/dim]")

            # 4. Infra (delayed display after lock)
            info = boot_infra_result
            gcp = info.get("gcp", {})
            wandb_info = info.get("wandb", {})
            pricing_info = info.get("pricing", {})

            registry_exists = gcp.get("registry", {}).get("exists", False)
            persistent_registry_exists = gcp.get("persistent_registry", {}).get("exists", False)
            sa_exists = gcp.get("service_account", {}).get("exists", False)
            queue_exists = wandb_info.get("queue", {}).get("exists", False)
            pricing_repo_exists = pricing_info.get("repository", {}).get("exists", False)
            pricing_function_exists = pricing_info.get("function", {}).get("exists", False)
            pricing_scheduler_exists = pricing_info.get("scheduler", {}).get("exists", False)
            api_enabled = gcp.get("apis", {}).get("all_enabled", False)
            cloudbuild_iam_granted = gcp.get("cloudbuild_iam", {}).get("granted", False)
            vpc_peering_exists = gcp.get("vpc_peering", {}).get("exists", False)
            project_exists = wandb_info.get("project", {}).get("exists", False)

            infra_ready = (
                registry_exists and persistent_registry_exists and sa_exists and
                queue_exists and pricing_repo_exists and pricing_function_exists and
                pricing_scheduler_exists and api_enabled and cloudbuild_iam_granted and
                vpc_peering_exists and project_exists
            )

            if not infra_ready:
                status("[red]✗ Infra incomplete![/red]")
                status("[yellow]⚠️  Run setup first: python CLI/cli.py setup[/yellow]")
                if lock_file:
                    lock_file.close()
                return False
            else:
                status("[dim]✓ Infra good![/dim]")

            # 5. Pricing (delayed display after lock)
            pricing_data, _, _ = boot_pricing_result
            age_minutes = get_pricing_age_minutes(pricing_data)
            updated_iso = pricing_data.get("updated", "")

            if updated_iso:
                try:
                    time_raw = updated_iso.split("T")[1].split(".")[0][:5]
                    hour, minute = time_raw.split(":")
                    updated_time = f"{int(hour)}:{minute}"
                except Exception:
                    updated_time = None
            else:
                updated_time = None

            if age_minutes < 1:
                time_part = f" ({updated_time} UTC)" if updated_time else ""
                status(f"[dim]✓ Good prices!{time_part} less than 1 minute ago[/dim]")
            else:
                time_part = f" ({updated_time} UTC)" if updated_time else ""
                status(f"[dim]✓ Good prices!{time_part} {age_minutes:.0f} minutes ago[/dim]")

        except Exception as lock_error:
            status(
                f"[yellow]⚠️ Lock file error (non-critical): {str(lock_error)[:50]}[/yellow]"
            )
            # Continue anyway (lock is just a safety feature)

    try:
        # ═══════════════════════════════════════════════════════════════
        # 🔄 SEMI-PERSISTENT RUNNER DESIGN (v2.0 - 2025-11-16)
        # ═══════════════════════════════════════════════════════════════
        #
        # Runners now self-manage their lifecycle:
        # - 30 min idle timeout (auto-shutdown when no jobs)
        # - Process multiple jobs until idle
        # - Fast bailout on fatal errors (5s)
        # - 240m Cloud Run timeout (safety net)
        #
        # NO EXTERNAL CLEANUP NEEDED - runners handle their own lifecycle!
        # Old cleanup logic removed - runners exit gracefully after 30min idle.
        # ═══════════════════════════════════════════════════════════════

        # Constants
        region = "us-central1"
        job_name = "vertex-ai-launcher"

        # Reset build context for this launch
        _BUILD_CONTEXT["build_id"] = None
        _BUILD_CONTEXT["provision_price"] = None

        # NOTE: Config, queue, and pricing checks already done in boot-up section above!
        # Results available in boot_infra_result and boot_pricing_result
        pricing_data, _, _ = boot_pricing_result

        # ==========================================
        # 🤖 MECHA BATTLE SYSTEM - Runs EVERY launch!
        # ==========================================
        # This runs BEFORE all other checks to:
        # - Check CPU changes → wipe all pools if changed
        # - Run price battle → select CHAMPION region
        # - Passively deploy missing MECHAs
        # - Return CHAMPION region for this launch

        # Get best machine FIRST (always runs, even if MECHA fails)
        # Cloud Build uses ONLY cb_quota (Cloud Build C3) - CE quota does NOT apply
        from ..shared.machine_selection import get_best_c3

        project_id = config.get("GCP_PROJECT_ID", "")

        best_machine, best_vcpus, cb_quota = get_best_c3(project_id, region)

        # Read C3_SINGLE_REGION_OVERRIDE (if set, MECHA declares instant victory!)
        override_region = config.get("C3_SINGLE_REGION_OVERRIDE", "").strip()

        # Read MECHA_OUTLAWED_REGIONS (comma-separated list to exclude from battle)
        mecha_outlawed_regions_str = config.get("MECHA_OUTLAWED_REGIONS", "").strip()
        mecha_outlawed_regions = [
            r.strip() for r in mecha_outlawed_regions_str.split(",") if r.strip()
        ]

        try:
            from .mecha.mecha_integration import run_mecha_battle

            mecha_selected_region = run_mecha_battle(
                project_id,
                best_machine,
                region,
                pricing_data,
                status,
                override_region=override_region,  # Force specific region (skips MECHA price battle)
                outlawed_regions=mecha_outlawed_regions,  # Exclude these regions from battle
            )
            # Override region with MECHA CHAMPION selection
            region = mecha_selected_region
        except Exception as e:
            status(
                f"[yellow]⚠️  MECHA system error (falling back to PRIMARY): {e}[/yellow]"
            )
            # Continue with primary region on error
            pass

        # ═══════════════════════════════════════════════════════════════
        # ✅ END OF MECHA BATTLE - Cloud Build region selected
        # ═══════════════════════════════════════════════════════════════
        status(
            f"[cyan]███▓▓▒░ ✅ MECHA: Cloud Build region selected ({region}, {best_machine}) ░▒▓▓███[/cyan]"
        )
        status("")

        # ==========================================
        # ⚡ ZEUS THUNDER BATTLE SYSTEM - GPU Region Selection for Vertex AI
        # ==========================================
        # This runs AFTER MECHA to select the optimal GPU region for training.
        # MECHA selects Cloud Build region (where images are built)
        # ZEUS selects Vertex AI region (where GPU training runs)

        # Default to PRIMARY_REGION for Vertex AI (Zeus can override)
        vertex_ai_region = PRIMARY_REGION

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: Validate REQUIRED GPU configuration (NO DEFAULTS!)
        # ═══════════════════════════════════════════════════════════════
        # Training jobs are GPU-specific - we REQUIRE explicit config!
        # Fail hard and early if GPU config is missing or invalid.

        gpu_type = config.get("TRAINING_GPU", "").strip()

        if not gpu_type:
            status("")
            status("[red]❌ TRAINING_GPU not set in .training config![/red]")
            status("  Training runs require explicit GPU type specification.")
            status("  Add to your .training file:")
            status("")
            status(
                '  TRAINING_GPU="NVIDIA_L4"  [dim]# Or: NVIDIA_TESLA_T4, NVIDIA_TESLA_A100, NVIDIA_H100, NVIDIA_H200[/dim]'
            )
            status("")
            return False

        gpu_count_str = config.get("TRAINING_GPU_NUMBER", "").strip()
        if not gpu_count_str:
            status("")
            status("[red]❌ TRAINING_GPU_NUMBER not set in .training config![/red]")
            status("  Training runs require explicit GPU count.")
            status("  Add to your .training file:")
            status("")
            status(
                '  TRAINING_GPU_NUMBER="1"  [dim]# Number of GPUs (1, 2, 4, 8, etc.)[/dim]'
            )
            status("")
            return False

        try:
            gpu_count = int(gpu_count_str)
            if gpu_count <= 0:
                raise ValueError("TRAINING_GPU_NUMBER must be positive")
        except ValueError:
            status("")
            status(f"[red]❌ Invalid TRAINING_GPU_NUMBER: {gpu_count_str}[/red]")
            status(
                "  TRAINING_GPU_NUMBER must be a positive integer (1, 2, 4, 8, etc.)"
            )
            status("")
            return False

        gpu_is_preemptible_str = (
            config.get("TRAINING_GPU_IS_PREEMPTIBLE", "").strip().lower()
        )
        if gpu_is_preemptible_str not in ["true", "false"]:
            status("")
            status("[red]❌ TRAINING_GPU_IS_PREEMPTIBLE not set or invalid![/red]")
            status("  Must be explicitly set to 'true' or 'false'.")
            status("  Add to your training config:")
            status("")
            status(
                '  TRAINING_GPU_IS_PREEMPTIBLE = "true"   [dim]# Spot instances (60-91% cheaper!)[/dim]'
            )
            status(
                '  TRAINING_GPU_IS_PREEMPTIBLE = "false"  [dim]# On-demand (stable)[/dim]'
            )
            status("")
            return False

        # Map GPU type to Zeus tier
        tier_map = {
            "NVIDIA_TESLA_T4": "spark",
            "NVIDIA_L4": "bolt",
            "NVIDIA_TESLA_A100": "storm",
            "NVIDIA_H100_80GB": "tempest",
            "NVIDIA_H200": "cataclysm",
        }

        if gpu_type not in tier_map:
            status("")
            status(f"[red]❌ Invalid TRAINING_GPU: {gpu_type}[/red]")
            status("")
            status("   Supported GPU types:")
            status(
                '   - "NVIDIA_TESLA_T4"    [dim](Spark tier - 16GB, widely available)[/dim]'
            )
            status(
                '   - "NVIDIA_L4"          [dim](Bolt tier - 24GB, good price/perf)[/dim]'
            )
            status(
                '   - "NVIDIA_TESLA_A100"  [dim](Storm tier - 40GB, production)[/dim]'
            )
            status(
                '   - "NVIDIA_H100_80GB"   [dim](Tempest tier - 80GB, flagship)[/dim]'
            )
            status(
                '   - "NVIDIA_H200"        [dim](Cataclysm tier - 141GB, extreme)[/dim]'
            )
            status("")
            return False

        tier_name = tier_map[gpu_type]

        # Get machine type for display (uses existing machine_selection.py function)
        from ..shared.machine_selection import get_best_machine_for_gpu

        machine_type = get_best_machine_for_gpu(gpu_type)

        # Read Zeus-specific config
        zeus_override_region = config.get("ZEUS_SINGLE_REGION_OVERRIDE", "").strip()
        zeus_outlawed_regions_str = config.get("ZEUS_OUTLAWED_REGIONS", "").strip()
        zeus_outlawed_regions = [
            r.strip() for r in zeus_outlawed_regions_str.split(",") if r.strip()
        ]

        # Feature flag check (default: enabled for testing)
        zeus_enabled = os.environ.get("ZEUS_ENABLED", "true").lower() == "true"

        if zeus_enabled:
            try:
                from .zeus.zeus_integration import run_thunder_battle
            except ImportError as e:
                status("")
                status(f"[red]❌ Zeus system import error: {e}[/red]")
                status("   Check zeus module installation")
                status("")
                return False

            # Run Zeus thunder battle (may raise RuntimeError if pricing unavailable)
            try:
                zeus_selected_region = run_thunder_battle(
                    project_id=project_id,
                    tier_name=tier_name,
                    gpu_count=gpu_count,
                    primary_region=PRIMARY_REGION,
                    pricing_data=pricing_data,
                    status_callback=status,
                    override_region=zeus_override_region,  # Force specific region (skips Zeus battle)
                    outlawed_regions=zeus_outlawed_regions,  # Exclude these regions from battle
                )
                # Override Vertex AI region with ZEUS CHAMPION selection
                vertex_ai_region = zeus_selected_region

                # 🔥 CRITICAL: Save ZEUS region to config for W&B Launch!
                # wandb_helper.py needs this to create regional staging bucket
                config["TRAINING_GPU_REGION"] = zeus_selected_region

                # ═══════════════════════════════════════════════════════════════
                # ✅ END OF ZEUS BATTLE - GPU region selected
                # ═══════════════════════════════════════════════════════════════
                status(
                    f"[cyan]███▓▓▒░ ✅ ZEUS: GPU region selected ({vertex_ai_region}, {machine_type}) ░▒▓▓███[/cyan]"
                )
                status("")

            except RuntimeError as e:
                # FATAL ERROR: No live pricing data available
                # This is intentional - we NEVER launch with fake pricing!
                status("")
                status(f"[red]❌ FATAL: {e}[/red]")
                status("")
                status("   Zeus requires live GCP pricing to select optimal region.")
                status("   Cannot proceed without real pricing data.")
                status("")
                status("   Fix:")
                status("   1. Ensure pricing fetch container built successfully")
                status("   2. Check pricing_gcr_latest.json in Artifact Registry")
                status("   3. Verify MECHA completed pricing fetch phase")
                status("")
                return False  # HALT LAUNCH!

        # ==========================================

        # Step 1.3: Ensure arr-pytorch-base image exists (needed by arr-ml-stack)
        # Image 0: PyTorch 2.6.0 from source with ALL GPU arch support (T4/L4/A100/H100)
        # HASH-BASED: Only rebuilds if Dockerfile changes (almost never!)
        # First build: ~30 min (176-vCPU) or 6+ hours (4-vCPU), then cached in Artifact Registry forever!
        if not _handle_pytorch_clean_image(
            config, region, status, best_machine, best_vcpus
        ):
            return False

        # Step 1.4: Ensure arr-ml-stack image exists (needed by arr-trainer)
        # Image 1: ML libraries (transformers, datasets, wandb, etc.)
        # Rebuilds when requirements.txt changes
        if not _handle_base_image(config, region, status):
            return False

        # Step 1.5: Build arr-trainer image (pre-build so we can use --docker-image)
        training_image = _handle_training_image(config, region, status)
        if training_image is None:
            return False

        # Step 1.6: Handle arr-vertex-launcher image
        # Build all 4 images upfront before W&B submission
        if not _handle_runner_image(config, region, status):
            return False

        # Step 1.7: Verify all 4 images exist in Artifact Registry (4-TIER DIAMOND GO/NO-GO)
        if not _verify_all_images_in_registry(config, region, status):
            return False

        # Step 1.8: Vertex AI GPU quota validation (2025-11-16)
        # Check CORRECT quota (Vertex AI Custom Training, not Compute Engine!)
        # Show user their quota status before submission
        # HALT launch if quota=0 (no point submitting job that will fail!)
        gpu_type = config.get("TRAINING_GPU", "")
        if gpu_type:
            status("⚡ Checking (GPU QUOTA)...")
            try:
                from ..shared.quota import has_vertex_gpu_quota

                # Read spot/preemptible setting from config (CRITICAL: must match actual job config!)
                use_preemptible_str = config.get(
                    "TRAINING_GPU_IS_PREEMPTIBLE", "false"
                ).lower()
                use_spot = use_preemptible_str == "true"
                has_quota = has_vertex_gpu_quota(
                    project_id, vertex_ai_region, gpu_type, use_spot, required=gpu_count
                )

                if has_quota:
                    spot_label = "spot" if use_spot else "on-demand"
                    status(
                        f"[dim]  ✓ {gpu_type} ({spot_label}) quota available in {vertex_ai_region}[/dim]"
                    )
                else:
                    # NO QUOTA - But check if OPPOSITE quota type is available!
                    spot_label = "spot" if use_spot else "on-demand"
                    status("")
                    status(
                        f"[red]❌ No {gpu_type} ({spot_label}) quota in {vertex_ai_region}![/red]"
                    )
                    status("")

                    # Check opposite quota type (spot vs on-demand)
                    opposite_quota = has_vertex_gpu_quota(
                        project_id,
                        vertex_ai_region,
                        gpu_type,
                        not use_spot,
                        required=gpu_count,
                    )
                    opposite_label = "spot" if not use_spot else "on-demand"

                    if opposite_quota:
                        # User HAS the opposite quota type - show how to switch!
                        status(
                            f"[green]✅ BUT you DO have {gpu_type} ({opposite_label}) quota![/green]"
                        )
                        status("  To use this quota, update your .training config:")
                        status("")
                        if use_spot:
                            # User tried spot, has on-demand
                            status(
                                f'  TRAINING_GPU_IS_PREEMPTIBLE="false"  [dim]# Change to on-demand[/dim]'
                            )
                        else:
                            # User tried on-demand, has spot
                            status(
                                f'  TRAINING_GPU_IS_PREEMPTIBLE="true"  [dim]# Change to spot (60-91% savings!)[/dim]'
                            )
                        status("")

                    # Get ALL GPU quotas to show alternatives
                    status("⏳ Finding GPUs that will work...")
                    from ..shared.quota import get_all_vertex_gpu_quotas

                    all_quotas = get_all_vertex_gpu_quotas(project_id, region)
                    available = [q for q in all_quotas if q["quota_limit"] > 0]

                    if available:
                        status("[cyan]Available GPU quotas you CAN use:[/cyan]")
                        for q in available:
                            gpu_name = q["gpu_name"]
                            quota_limit = q["quota_limit"]
                            is_spot = q.get("is_spot", False)
                            gpu_type_value = (
                                q.get("quota_id", "unknown")
                                .replace("custom_model_training_", "")
                                .replace("preemptible_", "")
                                .replace("_gpus", "")
                                .upper()
                            )

                            # Convert to proper GPU type format (NVIDIA_TESLA_T4, etc.)
                            if "nvidia_" in gpu_type_value.lower():
                                gpu_type_value = "NVIDIA_" + gpu_type_value.replace(
                                    "NVIDIA_", ""
                                ).replace("nvidia_", "")
                            if gpu_type_value.startswith(
                                "NVIDIA_"
                            ) and not gpu_type_value.startswith("NVIDIA_TESLA_"):
                                # Add TESLA_ for T4, P4, etc. that need it
                                simple_name = gpu_type_value.replace("NVIDIA_", "")
                                if simple_name in [
                                    "T4",
                                    "P4",
                                    "P100",
                                    "V100",
                                    "A100",
                                    "A100_80GB",
                                ]:
                                    gpu_type_value = "NVIDIA_TESLA_" + simple_name

                            spot_setting = "true" if is_spot else "false"
                            spot_text = "(spot)" if is_spot else "(on-demand)"
                            status(
                                f"  • {gpu_name} {spot_text}: {quota_limit} GPU{'s' if quota_limit != 1 else ''}"
                            )
                            status(f'    [dim]TRAINING_GPU="{gpu_type_value}"[/dim]')
                            status(
                                f'    [dim]TRAINING_GPU_IS_PREEMPTIBLE="{spot_setting}"[/dim]'
                            )
                        status("")

                    if not opposite_quota and not available:
                        # No alternatives at all - show request instructions
                        status("To request quota:")
                        status(f"  1. Run: [cyan]python CLI/cli.py infra[/cyan]")
                        status(
                            f"  2. Find '{gpu_type}' in GPU quotas table (shows quota request URL)"
                        )
                        status(
                            f"  3. Click URL → Request quota increase → Wait for approval (1-2 days)"
                        )
                        status("")

                    status(
                        "[yellow]Launch aborted - cannot submit job without GPU quota[/yellow]"
                    )
                    return False  # Halt launch!

            except Exception as e:
                # Quota check FAILED - halt launch!
                # If we can't verify quota, don't waste time launching
                status("")
                status(f"[red]❌ Failed to verify GPU quota: {e}[/red]")
                status("")
                status(
                    "Quota verification failed - cannot proceed without confirmation"
                )
                status("")
                status("Possible causes:")
                status("  • GCP API timeout or error")
                status("  • Invalid project ID or region")
                status("  • Missing gcloud authentication")
                status("")
                status(
                    "[yellow]Launch aborted - cannot submit job without quota verification[/yellow]"
                )
                return False  # Halt launch!

        # Step 2: Ensure regional buckets exist in ZEUS region (on-demand creation!)
        # Creates BOTH staging + checkpoints buckets in GPU training region
        status("⚡ Checking (ARTIFACTS)...")
        if not _ensure_regional_staging_bucket(config, status):
            return False

        # Step 3: Submit to W&B queue
        # Extracted from screen.py lines 554-567
        # **USER:** Now we throw the submission result...
        # **THEAETETUS:** Both TUI and CLI will catch this, but handle it differently!
        status("⚡ Submitting to W&B queue...")

        # Machine type always auto-computed from GPU in validation.py!

        # 🧪 TEST ERROR LOGGING: To test W&B runner failure error log catch, uncomment these lines:
        #    This forces an invalid GPU type AFTER quota checks pass, triggering a permission
        #    error in Vertex AI. The runner will fail and we can test error log extraction.
        #
        # config["TRAINING_GPU"] = "NVIDIA_H100_80GB"
        # status("[yellow]  ⚠️  TEST MODE: Forcing GPU to H100 (will trigger permission error)[/yellow]")

        run_id, run_name = _submit_to_wandb(helper, config, status, training_image)
        if run_id is None:
            return False

        # Step 3: Setup secrets
        # Extracted from screen.py lines 572-643
        if not _setup_secrets(config, status):
            return False

        # Step 4: Create service account
        # Extracted from screen.py lines 645-715
        if not _create_service_account(config, region, status):
            return False

        # Step 5: Create/update Cloud Run Job
        # Extracted from screen.py lines 986-1187
        # CRITICAL: Pass vertex_ai_region (ZEUS-selected) to Cloud Run job!
        # The environment variable CLOUDSDK_COMPUTE_REGION must match the region where Vertex AI jobs run.
        if not _create_cloud_run_job(config, vertex_ai_region, job_name, status):
            return False

        # Step 6: Execute runner
        # Extracted from screen.py lines 1189-1487
        execution_name = _execute_runner(config, vertex_ai_region, job_name, status)
        if execution_name is None:
            # Active runner found - job queued successfully!
            # Show success and return True (existing runner will process job)
            _show_success_rocketship(status, run_id, run_name)
            return True

        # Step 7: Stream logs until completion
        # Extracted from screen.py lines 1268-1487
        # NOTE: _wait_for_job_submission() shows success rocketship internally!
        success, output = _wait_for_job_submission(
            config, vertex_ai_region, job_name, execution_name, status, run_id
        )

        # Return result (rocketship already shown if success)
        return success

    except Exception as e:
        status(f"\n[red]❌ Submission failed: {str(e)}[/red]")
        return False
    finally:
        # Release launch lock when function exits (success or failure)
        try:
            if "lock_file" in locals() and lock_file is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                status("[dim]🔓 Launch lock released[/dim]")
        except Exception:
            pass  # Lock cleanup is non-critical


def _check_existing_jobs(
    helper: WandBHelper,
    status: StatusCallback,
) -> Optional[List[Dict]]:
    """
    Check for existing active jobs

    Extracted from: screen.py lines 505-552

    Args:
        helper: WandBHelper instance
        status: Status callback

    Returns:
        List of active runs (empty if none)
        None if error occurred

    Side effects:
        - Calls status() with job information
        - Blocks submission if jobs found
    """
    try:
        existing_runs = helper.get_active_runs()
        if len(existing_runs) == 0:
            status(f"[dim]✓ Queue good![/dim]")
        else:
            status(f"[dim]✓ Job check complete ({len(existing_runs)} active)[/dim]")

        if existing_runs and len(existing_runs) > 0:
            # Found active jobs - block submission
            status(f"\n[yellow]⚠️  Found {len(existing_runs)} active job(s):[/yellow]")

            for run in existing_runs[:100]:  # Show first 100
                status(f"[dim]  • {run['name']} ({run['state']})[/dim]")

            if len(existing_runs) > 100:
                status(f"[dim]  ... and {len(existing_runs) - 100} more[/dim]")

            status("\n[yellow]❌ Cannot submit: Jobs already running![/yellow]")
            status("  [dim]Go to Monitor screen to view/cancel existing jobs[/dim]")

        return existing_runs

    except Exception as e:
        status(f"[red]❌ Error checking jobs: {str(e)}[/red]")
        return None


def _ensure_regional_staging_bucket(config: Dict[str, str], status: Callable) -> bool:
    """
    Ensure regional buckets exist in ZEUS region (on-demand creation!)

    Creates BOTH staging and checkpoints buckets:
    - gs://{project_id}-{project_name}-{region}-staging
    - gs://{project_id}-{project_name}-{region}-checkpoints

    Vertex AI requires staging bucket in SAME region as training job.
    Checkpoints bucket created in same region for consistency.
    """
    import subprocess

    # Get ZEUS region (where GPU was selected)
    zeus_region = config.get("TRAINING_GPU_REGION", "us-central1")
    project_id = config.get("GCP_PROJECT_ID", "")
    project_name = config.get("PROJECT_NAME", "arr-coc-0-1")

    # Regional bucket names (GCS bucket names are globally unique!)
    staging_bucket_name = f"{project_id}-{project_name}-{zeus_region}-staging"
    staging_bucket_uri = f"gs://{staging_bucket_name}"

    checkpoints_bucket_name = f"{project_id}-{project_name}-{zeus_region}-checkpoints"
    checkpoints_bucket_uri = f"gs://{checkpoints_bucket_name}"

    # Ensure both buckets exist
    buckets_to_create = [
        ("staging", staging_bucket_name, staging_bucket_uri),
        ("checkpoints", checkpoints_bucket_name, checkpoints_bucket_uri),
    ]

    try:
        for bucket_type, bucket_name, bucket_uri in buckets_to_create:
            # Check if bucket exists
            result = subprocess.run(
                ["gsutil", "ls", bucket_uri], capture_output=True, timeout=10
            )

            if result.returncode == 0:
                # Bucket exists in this region!
                status(f"  ✓ Good ({bucket_type.upper()})")
            else:
                # Bucket doesn't exist - create it!
                status(
                    f"[cyan]📦 Creating regional {bucket_type} bucket: {bucket_name}...[/cyan]"
                )

                create_result = subprocess.run(
                    ["gsutil", "mb", "-l", zeus_region, bucket_uri],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if create_result.returncode == 0:
                    status(
                        f"[green]  ✓ Created {bucket_type} bucket: {bucket_uri} ({zeus_region})[/green]"
                    )
                else:
                    status(f"[red]❌ Failed to create {bucket_type} bucket![/red]")
                    status(f"[dim]  Error: {create_result.stderr}[/dim]")
                    return False

        return True

    except subprocess.TimeoutExpired:
        status(f"[red]❌ Timeout checking/creating regional buckets[/red]")
        return False
    except Exception as e:
        status(f"[red]❌ Error with regional buckets: {str(e)}[/red]")
        return False


def _submit_to_wandb(
    helper: WandBHelper,
    config: Dict[str, str],
    status: StatusCallback,
    training_image: str,
) -> tuple[Optional[str], str]:
    """
    Submit job to W&B Launch queue

    Extracted from: screen.py lines 554-567

    Args:
        helper: WandBHelper instance
        config: Training configuration
        status: Status callback
        training_image: Pre-built training Docker image URI

    Returns:
        tuple: (run_id, run_name)
            - run_id: W&B run ID if successful, None if submission failed
            - run_name: Generated cool name (e.g., "ethereal-snowflake-1762661725")

    Side effects:
        - Creates W&B queue item
        - Displays queue URL via status()
    """
    try:
        entity = config.get("WANDB_ENTITY", "")
        project = config.get("WANDB_PROJECT", "arr-coc-0-1")
        queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

        status(f"[dim]  → View queue: https://wandb.ai/{entity}/{project}/launch[/dim]")

        # Submit to queue with pre-built image
        run_id, output, run_name = helper.submit_job(config, training_image)

        if run_name and run_name != "unknown":
            status(f"[green]  ✓ Run queued in {queue_name}: {run_name}[/green]")
        else:
            status(f"[green]  ✓ Run queued in {queue_name}[/green]")
            status(
                f"[yellow]  ⚠️  Warning: Run name not extracted from W&B output[/yellow]"
            )

        return (run_id, run_name)

    except Exception as e:
        status(f"[red]❌ Failed to submit to W&B: {str(e)}[/red]")
        return (None, "unknown")


def _setup_secrets(
    config: Dict[str, str],
    status: StatusCallback,
) -> bool:
    """
    Setup W&B API key in Secret Manager

    Extracted from: screen.py lines 572-643

    SECURITY: Uses Secret Manager for W&B API key (production-grade!)

    Args:
        config: Training configuration
        status: Status callback

    Returns:
        True if secret setup succeeded
        False if setup failed

    Side effects:
        - Creates secret in Secret Manager if doesn't exist
        - Updates secret version with current API key
    """
    try:
        import wandb as wandb_module

        wandb_api_key = wandb_module.api.api_key
        secret_name = "wandb-api-key"

        # Check if secret exists
        check_secret = run_gcloud_with_retry(
            ["gcloud", "secrets", "describe", secret_name],
            max_retries=3,
            timeout=30,
            operation_name="secrets describe",
        )

        if check_secret.returncode != 0:
            # Secret doesn't exist - create it
            status("⏳ Creating W&B API key secret...")
            create_secret = run_gcloud_with_retry(
                [
                    "gcloud",
                    "secrets",
                    "create",
                    secret_name,
                    "--replication-policy=automatic",
                ],
                max_retries=3,
                timeout=30,
                operation_name="secrets create --replication-policy=automatic",
            )
            if create_secret.returncode != 0:
                raise Exception(
                    f"Failed to create secret: {create_secret.stderr[:200]}"
                )

        # Update secret with current API key (creates new version)
        add_version = run_gcloud_with_retry(
            [
                "gcloud",
                "secrets",
                "versions",
                "add",
                secret_name,
                "--data-file=-",
            ],
            max_retries=3,
            timeout=30,
            operation_name="secrets versions add",
            stdin_input=wandb_api_key,  # Pass API key via stdin
        )
        if add_version.returncode != 0:
            raise Exception(f"Failed to add secret version: {add_version.stderr[:200]}")

        return True

    except subprocess.TimeoutExpired as e:
        status(
            f"[red]Secret Manager timeout (network slow?). Command: {e.cmd[0]} {e.cmd[1]}[/red]"
        )
        status(
            "[dim]Try again or check: gcloud services list --enabled | grep secret[/dim]"
        )
        return False
    except Exception as e:
        status(f"[red]Secret Manager error: {str(e)[:300]}[/red]")
        return False


def _create_service_account(
    config: Dict[str, str],
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Create dedicated service account for Cloud Run Job

    Extracted from: screen.py lines 645-715

    SECURITY: Creates dedicated SA (NOT using default compute SA)

    OPTIMIZATION: Checks if roles already granted and returns early (idempotent)
    - First call (setup): Creates SA and grants roles (~5 seconds)
    - Subsequent calls (launch): Skips if already configured (~0.5 seconds)

    Args:
        config: Training configuration
        region: GCP region
        status: Status callback

    Returns:
        True if SA setup succeeded
        False if setup failed

    Side effects:
        - Creates service account if doesn't exist
        - Grants secret access to SA
        - Grants Vertex AI IAM roles (aiplatform.user, artifactregistry.reader, storage.objectViewer)
    """
    try:
        max_retries = 3  # Consistent retry count for all gcloud commands
        project_id = config.get("GCP_PROJECT_ID", "")

        # SHARED SERVICE ACCOUNT (all ARR-COC prototypes use same SA)
        sa_name = "arr-coc-sa"
        sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"

        # CRITICAL: Check SA exists FIRST (avoid false positive from dangling IAM bindings)
        # When SA is deleted, GCP leaves dangling IAM bindings which cause false positives
        check_sa_exists = None
        for attempt in range(max_retries):
            check_sa_exists = run_gcloud_with_retry(
                ["gcloud", "iam", "service-accounts", "describe", sa_email],
                max_retries=3,
                timeout=10,
                operation_name="iam service-accounts describe",
            )
            if (
                check_sa_exists.returncode == 0
                or "does not exist" in check_sa_exists.stderr
            ):
                break  # Success (exists) or definitive failure (doesn't exist)

            # Retry on timeout/transient errors
            if attempt < max_retries - 1:
                time.sleep(1)

        sa_exists = check_sa_exists.returncode == 0

        # Quick check: If SA exists AND all roles already granted, skip setup (makes launch faster)
        iam_roles = [
            "roles/aiplatform.user",
            "roles/artifactregistry.reader",
            "roles/storage.objectViewer",
        ]

        if sa_exists:
            # Check existing roles with retries (transient API failures)
            check_roles = None
            for attempt in range(max_retries):
                check_roles = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "projects",
                        "get-iam-policy",
                        project_id,
                        "--flatten=bindings",
                        f"--filter=bindings.members:serviceAccount:{sa_email}",
                        "--format=value(bindings.role)",
                    ],
                    max_retries=3,
                    timeout=10,
                    operation_name="projects get-iam-policy --flatten=bindings",
                )

                if check_roles.returncode == 0:
                    break  # Success!

                # Retry on timeout/transient errors
                if attempt < max_retries - 1:
                    time.sleep(1)

            if check_roles and check_roles.returncode == 0:
                existing_roles = set(check_roles.stdout.strip().split("\n"))
                if all(role in existing_roles for role in iam_roles):
                    # All roles already granted - skip setup (silent success)
                    return True

        secret_name = "wandb-api-key"

        # Create SA if it doesn't exist (sa_exists check done above)
        if not sa_exists:
            # SA doesn't exist - create it
            status("⏳ Creating service account...")
            create_sa_result = None
            for attempt in range(max_retries):
                create_sa_result = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "iam",
                        "service-accounts",
                        "create",
                        sa_name,
                        "--display-name=W&B Launch Agent (Cloud Run Job)",
                        "--description=Dedicated SA for vertex-ai-launcher job",
                    ],
                    max_retries=3,
                    timeout=10,
                    operation_name="iam service-accounts create",
                )
                if (
                    create_sa_result.returncode == 0
                    or "already exists" in create_sa_result.stderr
                ):
                    break  # Success or already exists (both good!)

                # Retry on timeout/transient errors
                if attempt < max_retries - 1:
                    time.sleep(1)

            if (
                create_sa_result.returncode != 0
                and "already exists" not in create_sa_result.stderr
            ):
                raise Exception(
                    f"Failed to create service account after {max_retries} attempts: {create_sa_result.stderr[:300]}"
                )

            # Wait for SA propagation (GCP needs time to replicate SA across systems)
            status("⏳ Waiting for service account propagation (5 seconds)...")

            time.sleep(5)

        # Grant secret access to THIS service account only (with retry for GCP propagation)

        max_retries = 5
        retry_delay = 2  # seconds
        grant_succeeded = False

        for attempt in range(max_retries):
            grant_result = run_gcloud_with_retry(
                [
                    "gcloud",
                    "secrets",
                    "add-iam-policy-binding",
                    secret_name,
                    f"--member=serviceAccount:{sa_email}",
                    "--role=roles/secretmanager.secretAccessor",
                ],
                max_retries=3,
                timeout=10,
                operation_name="secrets add-iam-policy-binding --member=serviceAccount:{sa_email}",
            )

            # Check if permission grant succeeded (or already exists)
            if (
                grant_result.returncode == 0
                or "already has role" in grant_result.stderr
            ):
                grant_succeeded = True
                break

            # If SA doesn't exist yet, wait and retry
            if "does not exist" in grant_result.stderr and attempt < max_retries - 1:
                status(
                    f"⏳ Waiting for SA propagation (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            # If SECRET doesn't exist, skip with warning (not needed during setup)
            elif "Secret" in grant_result.stderr and "not found" in grant_result.stderr:
                status(
                    f"⚠️  Secret '{secret_name}' not found - skipping secret grant (will create during launch if needed)"
                )
                grant_succeeded = True  # Continue setup (not a fatal error)
                break
            else:
                break

        if not grant_succeeded:
            raise Exception(
                f"Failed to grant secret access: {grant_result.stderr[:300]}"
            )

        # Grant Vertex AI permissions (required for agent to submit jobs)
        status("⏳ Granting Vertex AI permissions...")
        iam_roles = [
            "roles/aiplatform.user",  # Create Vertex AI custom jobs
            "roles/artifactregistry.reader",  # Pull Docker images
            "roles/storage.objectViewer",  # Access GCS staging bucket
        ]

        for role in iam_roles:
            # Retry IAM binding (transient API failures)
            grant_iam = None
            for attempt in range(max_retries):
                grant_iam = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "projects",
                        "add-iam-policy-binding",
                        project_id,
                        f"--member=serviceAccount:{sa_email}",
                        f"--role={role}",
                        "--condition=None",  # No conditions
                    ],
                    max_retries=3,
                    timeout=15,
                    operation_name="projects add-iam-policy-binding --member=serviceAccount:{sa_email}",
                )
                # Success or already has role (both good!)
                if (
                    grant_iam.returncode == 0
                    or "already has role" in grant_iam.stderr.lower()
                ):
                    break

                # Retry on timeout/transient errors
                if attempt < max_retries - 1:
                    time.sleep(1)

            # Warn if still failed after retries (but don't fail setup)
            if (
                grant_iam
                and grant_iam.returncode != 0
                and "already has role" not in grant_iam.stderr.lower()
            ):
                status(
                    f"[yellow]⚠️  Warning: Failed to grant {role} after {max_retries} attempts[/yellow]"
                )

        status("   [green]✓[/green]  Service account configured")
        # NOTE: GPU quota auto-request system REMOVED entirely (2025-11-16)
        # Users request Vertex AI GPU quotas manually via instructions in infra screen
        # This is simpler, clearer, and checks the CORRECT quota system

        return True

    except Exception as e:
        status(f"[red]Service account setup failed: {str(e)[:400]}[/red]")
        return False


def _handle_pytorch_clean_image(
    config: Dict[str, str],
    region: str,
    status: StatusCallback,
    best_machine: str,
    best_vcpus: int,
) -> bool:
    """
    Ensure arr-pytorch-base image exists (build if Dockerfile changed)

    Image 0: PyTorch 2.6.0 built from source with ALL GPU architectures
    - T4 (sm_75), L4 (sm_89), A100 (sm_80), H100 (sm_90)
    - No conda! Single Python 3.10 environment
    - Build time: ~30 min (176-vCPU) or 6+ hours (4-vCPU) FIRST TIME, then cached forever!

    HASH DETECTION ENABLED:
    - Calculates Dockerfile hash from manifest
    - Checks if image with that hash exists
    - Rebuilds if Dockerfile changed (PyTorch version updates)
    - Tags with both :hash and :latest

    Args:
        config: Training configuration
        region: GCP region for Cloud Build execution (from MECHA battle system)
                This is WHERE the build RUNS, not where images are stored!
                Images are always stored in us-central1 (Artifact Registry).
        status: Status callback
        best_machine: Best C3 machine type for worker pool
        best_vcpus: Number of vCPUs for that machine type

    Returns:
        True if arr-pytorch-base ready (exists or built successfully)
        False if build failed
    """
    try:
        # Calculate Dockerfile hash from manifest
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        manifest_path = (
            project_root / "Stack/arr-pytorch-base/.image-manifest"
        )

        # Hash all files listed in manifest
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # Image names
        # REGISTRY REGION: Images are STORED in us-central1 (PRIMARY_REGION)
        # BUILD REGION: Cloud Build RUNS in region parameter (from MECHA - cheapest!)
        project_id = config.get("GCP_PROJECT_ID", "")
        registry_name = (
            "arr-coc-registry"  # Fast-building images (ml-stack, trainer, launcher)
        )
        persistent_registry = (
            "arr-coc-registry-persistent"  # PyTorch base image only (never deleted)
        )
        registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
        pytorch_clean_hash = f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}/arr-pytorch-base:{dockerfile_hash}"
        pytorch_clean_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}/arr-pytorch-base:latest"

        # Use accumulator: Start pool + image checks in parallel
        # Sequential: 30s + 30s = 60s | Accumulator: Both at once = 30s
        pool_name = "pytorch-mecha-pool"
        acc = GCloudAccumulator(max_workers=2)

        # Start pool check (non-blocking!)
        acc.start(
            key="pool",
            cmd=[
                "gcloud",
                "builds",
                "worker-pools",
                "describe",
                pool_name,
                "--region",
                region,
                f"--project={project_id}",
                "--format=json",
            ],
            max_retries=3,
            timeout=30,
            operation_name="check worker pool",
        )

        # Start image check (non-blocking!)
        acc.start(
            key="image",
            cmd=[
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                pytorch_clean_hash,
            ],
            max_retries=3,
            timeout=30,
            operation_name="check pytorch image",
        )

        # Get pool result (waits if not ready)
        check_pool = acc.get("pool")

        pool_needs_creation = False
        if check_pool.returncode == 0:
            import json

            current_machine = (
                json.loads(check_pool.stdout)
                .get("privatePoolV1Config", {})
                .get("workerConfig", {})
                .get("machineType", "")
            )
            if current_machine != best_machine:
                status(
                    f"[yellow]⚠️  Wrong machine: {current_machine} → {best_machine}. Recreating...[/yellow]"
                )
                run_gcloud_with_retry(
                    [
                        "gcloud",
                        "builds",
                        "worker-pools",
                        "delete",
                        pool_name,
                        "--region",
                        region,
                        f"--project={project_id}",
                        "--quiet",
                    ],
                    max_retries=3,
                    timeout=120,
                    operation_name="delete worker pool",
                )
                status(f"   ✓ Deleted")
                pool_needs_creation = True
            else:
                # Silent success - pool is OK, no need to output
                # Build pool trace: Create entry now that we have machine info
                if _BUILD_CONTEXT["build_id"] is None:
                    try:
                        from CLI.shared.pricing import (
                            get_machine_hourly_cost,
                        )

                        vcpus = int(best_machine.split("-")[-1])
                        prov_price = get_machine_hourly_cost(best_machine, region)
                        bid = build_pool_trace.create_build_entry(
                            best_machine, vcpus, prov_price
                        )
                        _BUILD_CONTEXT["build_id"] = bid
                        _BUILD_CONTEXT["provision_price"] = prov_price
                    except Exception as e:
                        # Non-critical - don't block builds if trace fails
                        pass
        else:
            pool_needs_creation = True

        if pool_needs_creation:
            # CHONK power meter based on vCPUs (matches Dockerfile names)
            from ..shared.machine_selection import get_c3_chonk_label

            chonk_label, power_meter = get_c3_chonk_label(best_vcpus)

            status(
                f"⏳ Creating {chonk_label} pool: {best_machine} ({best_vcpus} vCPUs) {power_meter}"
            )
            status(f"   This can take some time (up to 15 minutes)...")
            status(
                f"   💡 Monitor progress: https://console.cloud.google.com/cloud-build/worker-pools?project={project_id}"
            )
            status(f"")
            # NOTE: Build pools need public egress for apt-get, pip, git
            # NVIDIA base images are mirrored to reduce external dependencies
            # But Dockerfile still needs: archive.ubuntu.com, pypi.org, github.com
            create_result = run_gcloud_with_retry(
                [
                    "gcloud",
                    "builds",
                    "worker-pools",
                    "create",
                    pool_name,
                    "--region",
                    region,
                    f"--project={project_id}",
                    f"--worker-machine-type={best_machine}",
                    "--worker-disk-size=100",
                ],
                max_retries=3,
                timeout=2700,
                operation_name="builds worker-pools create",
            )  # 45 min timeout for large machines (GCP can be slow)
            if create_result.returncode != 0:
                # Escape brackets in stderr for Rich markup
                stderr_escaped = (
                    create_result.stderr[:200].replace("[", "[[").replace("]", "]]")
                )
                status(f"[red]✗ Failed: {stderr_escaped}[/red]")
                return False
            status(f"[green]✓[/green]  Pool creation command sent")
            status(f"⏳ Waiting for pool to provision (this takes 10-15 minutes)...")
            status(f"")

            # Wait for pool to actually become RUNNING (GCP provisions in background)

            pool_ready = False
            wait_start = time.time()
            max_wait = 1200  # 20 minutes max
            check_interval = 15  # Check every 15 seconds

            while not pool_ready and (time.time() - wait_start) < max_wait:
                check_pool = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "builds",
                        "worker-pools",
                        "describe",
                        pool_name,
                        "--region",
                        region,
                        f"--project={project_id}",
                        "--format=json",
                    ],
                    max_retries=3,
                    timeout=30,
                    operation_name="builds worker-pools describe",
                )
                if check_pool.returncode == 0:
                    pool_info = json.loads(check_pool.stdout)
                    pool_state = pool_info.get("state", "UNKNOWN")
                    if pool_state == "RUNNING":
                        pool_ready = True
                        elapsed = int(time.time() - wait_start)
                        status(f"[green]✓[/green]  Pool RUNNING (took {elapsed}s)")
                        status(f"")

                        # Lazy loading quota entry for PRIMARY region
                        status("")
                        status("   🔄 Lazy Loading Quota Entry:")
                        from CLI.launch.mecha.mecha_acquire import lazy_load_quota_entry

                        try:
                            lazy_load_quota_entry(project_id, region, status)
                        except Exception as e:
                            status(f"   [dim](Quota entry init skipped: {e})[/dim]")
                        status("")
                    else:
                        # Show progress dots every minute
                        elapsed = int(time.time() - wait_start)
                        if elapsed % 60 < check_interval:
                            status(
                                f"   Still provisioning... {elapsed}s elapsed (state: {pool_state})"
                            )

                if not pool_ready:
                    time.sleep(check_interval)

            if not pool_ready:
                status(f"[red]✗ Pool failed to become RUNNING within {max_wait}s[/red]")
                return False

            # Build pool trace: Create entry now that pool is created
            if _BUILD_CONTEXT["build_id"] is None:
                try:
                    from CLI.shared.pricing import (
                        get_machine_hourly_cost,
                    )

                    vcpus = int(best_machine.split("-")[-1])
                    prov_price = get_machine_hourly_cost(best_machine, region)
                    bid = build_pool_trace.create_build_entry(
                        best_machine, vcpus, prov_price
                    )
                    _BUILD_CONTEXT["build_id"] = bid
                    _BUILD_CONTEXT["provision_price"] = prov_price
                except Exception:
                    # Non-critical - don't block builds if trace fails
                    pass

        # NOTE: NVIDIA base image mirroring happens automatically in Cloud Build Step 0
        # No local Docker commands needed - Cloud Build handles everything!

        # Get image result (waits if not ready - but likely already done!)
        status("")  # Visual separation
        status("⚡ Checking (ARR-PYTORCH-BASE)...")

        check_image = acc.get("image")
        acc.shutdown()

        if check_image.returncode != 0:
            # Image with this hash doesn't exist → Dockerfile changed!
            status(
                "[yellow]⏳[/yellow] Building [bold cyan](ARR-PYTORCH-BASE)[/bold cyan] on Cloud Build (~~30 min (176-vCPU) or 6+ hours (4-vCPU) FIRST TIME)..."
            )
            status(
                "[italic magenta]PyTorch-clean is THE FOUNDATION - built from source with ALL GPU architectures![/italic magenta]"
            )
            status(
                "[italic magenta]This builds ONCE (~30 min (176-vCPU) or 6+ hours (4-vCPU)), then cached in Artifact Registry forever! \\o/[/italic magenta]"
            )
            status(f"[dim]MAX_JOBS: {best_vcpus} vCPUs (parallel compilation)[/dim]")
            status(f"[dim]Dockerfile hash: {dockerfile_hash}[/dim]")
            status(
                f"[dim]→ View build logs: https://console.cloud.google.com/cloud-build/builds?project={project_id}[/dim]"
            )

            # Build with new hash
            return _build_pytorch_clean_image(
                config, region, status, best_machine, best_vcpus
            )

        # arr-pytorch-base exists with this hash → use it
        status(
            "[green]  ✓ Good [bold cyan](ARR-PYTORCH-BASE)[/bold cyan]! Will provide foundation PyTorch[/green]"
        )
        return True

    except Exception as e:
        status(f"[red]arr-pytorch-base image check failed: {str(e)[:200]}[/red]")
        return False


def _build_pytorch_clean_image(
    config: Dict[str, str],
    region: str,
    status: StatusCallback,
    best_machine: str,
    max_jobs: int,
) -> bool:
    """
    Build arr-pytorch-base image using Cloud Build

    ⚠️ WARNING: 2-4 hour build time (compiles PyTorch 2.6.0 from source)

    This is a ONE-TIME build that:
    - Compiles PyTorch 2.6.0 with TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    - Builds torchvision 0.20.0 and torchaudio 2.6.0
    - Creates clean foundation with NO CONDA!
    - Then cached in Artifact Registry forever!

    Args:
        config: Training configuration
        region: GCP region (from MECHA champion)
        status: Status callback

    Returns:
        True if build succeeded
        False if build failed
    """
    from CLI.shared.performance_monitor import get_monitor

    monitor = get_monitor()

    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent

        status(
            "🏗️  Building arr-pytorch-base image (~30 min (176-vCPU) or 6+ hours (4-vCPU))..."
        )
        status(
            "[italic magenta]Compiling PyTorch from source with ALL GPU architectures (T4/L4/A100/H100)![/italic magenta]"
        )
        status("")

        # Calculate hash for substitution
        manifest_path = (
            project_root / "Stack/arr-pytorch-base/.image-manifest"
        )
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # Get live pricing at START of build (provision quote!)
        provision_price_at_start = 0.0
        try:
            from CLI.shared.pricing import get_machine_hourly_cost

            # Use MECHA's dynamically-selected best_machine (c3-standard-4/8/22/44/88/176)
            provision_price_at_start = get_machine_hourly_cost(best_machine, region)
        except Exception:
            pass  # Non-blocking - pricing fetch failure doesn't block builds

        # Track build timing for campaign stats
        build_start_time = time.time()

        # Submit Cloud Build (MONITORED - VERY slow operation!)
        project_id = config.get("GCP_PROJECT_ID", "")
        op_id = monitor.start_operation(
            "build_pytorch_clean_cloudbuild", category="Docker"
        )

        # STREAMING OUTPUT FIX: Use Popen to show real-time progress!
        # Previously used subprocess.run() with capture_output=True, which hid ALL output
        # for ~30 min (176-vCPU) or 6+ hours (4-vCPU) while gcloud silently polled for build completion.
        # 🎉 2025-11-13: NO MORE SILENT WAITING! We can see it working in real-time! =D
        cmd = [
            "gcloud",
            "builds",
            "submit",
            str(project_root / "Stack/arr-pytorch-base"),
            "--config=" + str(project_root / "Stack" / "arr-pytorch-base" / ".cloudbuild-arr-pytorch-base.yaml"),
            f"--substitutions=_DOCKERFILE_FRESHNESS_HASH={dockerfile_hash},_MAX_JOBS={max_jobs},_PROJECT_ID={project_id},_REGION={region}",
            f"--region={region}",  # Use MECHA champion region (worker pools exist in all regions)
            "--timeout=420m",  # 7 hours (PyTorch 2.6.0 source build needs 4.5-5h on c3-highcpu-4)
        ]

        status("")
        status("[dim]→ Submitting to CloudBuild (showing real-time progress)...[/dim]")
        status("")

        # Load registry for MECHA fatigue tracking
        registry = load_registry()

        # Create fatigue callback for queue timeout monitor
        def mark_region_fatigued(
            region_name: str,
            reason: str = "",
            reason_code: str = "",
            error_message: str = "",
            build_id: str = "",
        ):
            """Mark region as fatigued and save registry"""
            # Record MECHA timeout (returns fatigue details)
            failure_count, fatigue_hours, fatigue_msg = record_mecha_timeout(
                registry,
                region_name,
                reason=reason,
                reason_code=reason_code,
                error_message=error_message,
                build_id=build_id,
            )

            # Create fatigue event object
            fatigue_event = {
                "fatigue_reason": reason,
                "fatigue_reason_code": reason_code,
                "fatigue_time": time.time(),
                "fatigue_duration_hours": fatigue_hours,
                "fatigue_type": fatigue_msg.split()[0],  # "FATIGUED" or "EXHAUSTED"
            }

            # Record to campaign stats (build-level)
            try:
                from CLI.launch.mecha.campaign_stats import record_build_result

                # Update build record with fatigue
                record_build_result(
                    region=region_name,
                    success=False,
                    duration_minutes=45.0,  # Queue timeout at 45 min
                    queue_wait_minutes=45.0,
                    build_id=build_id,
                    build_type="arr-pytorch-base",
                    machine_type=best_machine,
                    status="TIMEOUT",
                    error_message=error_message,
                    timeout_reason="QUEUED",
                    fatigues=[fatigue_event],
                )
                # Note: record_fatigue_event() already called inside record_mecha_timeout()
            except Exception:
                pass  # Don't fail if stats recording fails

            save_registry(registry)

        # Initialize queue monitor (will start once we have build ID)
        queue_monitor = None

        # Track build timing for campaign stats
        build_start_time = time.time()
        extracted_build_id = None

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered - immediate output
        )

        # Stream output line-by-line as it happens!
        build_output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # Show CloudBuild output with dimmed prefix for readability
                # Escape gcloud output to prevent Rich from interpreting brackets as markup
                status(f"   [dim]CloudBuild:[/dim] {escape(line)}")
                build_output_lines.append(line)

                # Extract build ID and start queue monitor + CHONK streamer
                if queue_monitor is None:
                    build_id = extract_build_id_from_output(line)
                    if build_id:
                        extracted_build_id = build_id  # Store for campaign stats

                        # Start monitoring for 45-minute QUEUED timeout!
                        queue_monitor = BuildQueueMonitor(
                            build_id=build_id,
                            region=region,
                            project_id=project_id,
                            fatigue_callback=mark_region_fatigued,
                            status_callback=status,
                        )
                        queue_monitor.start()

                        # Update pending build record with actual CloudBuild ID
                        try:
                            from CLI.launch.mecha.campaign_stats import (
                                update_pending_build,
                            )

                            update_pending_build(
                                region=region, build_id=build_id, status="QUEUED"
                            )
                            status("[dim]   → Campaign stats: Build ID updated[/dim]")
                        except Exception as e:
                            # Show stats errors to user instead of silent fail
                            status(
                                f"[dim yellow]⚠️  Campaign stats update error: {str(e)}[/dim yellow]"
                            )

                        # CHONK markers appear in logs after build completes
                        # (CloudBuild doesn't stream RUN command output in real-time)

        # Wait for completion
        returncode = process.wait(timeout=25200)  # 7 hours max

        # Stop queue monitor if running
        if queue_monitor:
            queue_monitor.stop()

            # Check if build timed out in QUEUED state
            if queue_monitor.did_timeout():
                # MECHA already marked as fatigued by monitor!
                # Hard exit - launch halted!
                monitor.end_operation(op_id)
                return False

        monitor.end_operation(op_id)

        # Create result object to match old subprocess.run() interface
        class Result:
            def __init__(self, returncode, stdout_lines):
                self.returncode = returncode
                self.stdout = "\n".join(stdout_lines)
                self.stderr = "\n".join(stdout_lines) if returncode != 0 else ""

        result = Result(returncode, build_output_lines)

        if result.returncode == 0:
            status("[green]✓ arr-pytorch-base image built successfully![/green]")

            # Record to campaign stats
            build_end_time = time.time()
            build_duration_seconds = int(build_end_time - build_start_time)
            build_duration_minutes = build_duration_seconds / 60.0

            try:
                from CLI.launch.mecha.campaign_stats import update_build_completion

                update_build_completion(
                    region=region,
                    build_id=extracted_build_id,
                    success=True,
                    duration_minutes=build_duration_minutes,
                    queue_wait_minutes=0.0,  # TODO: Extract from queue_monitor if available
                    duration_seconds=build_duration_seconds,
                    spot_price_per_hour=provision_price_at_start,  # Live C3-176 spot price at build START
                )
                status("[dim]   → Campaign stats: Build completed (SUCCESS)[/dim]")
            except Exception as e:
                # Don't fail build if campaign stats fail
                status(f"[dim]⚠️  Campaign stats update failed: {str(e)[:100]}[/dim]")

            # Calculate hash for success message
            manifest_path = (
                project_root / "Stack/arr-pytorch-base/.image-manifest"
            )
            dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

            # Diamond success celebration!
            status(
                f"[green]◇◇◇ arr-pytorch-base image pushed: arr-pytorch-base:latest (hash: {dockerfile_hash})[/green]"
            )
            status(
                "[green]    \\o/ \\o\\ /o/ PyTorch foundation ready! (T4/L4/A100/H100 support)[/green]"
            )

            # Cleanup old arr-pytorch-base images (keep only :latest)
            # REGISTRY REGION: Clean up images from us-central1 (where they're stored)
            # PyTorch images stored in arr-coc-registry-persistent (persistent registry)
            project_id = config.get("GCP_PROJECT_ID", "")
            persistent_registry = (
                "arr-coc-registry-persistent"  # PyTorch base image only
            )
            registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
            persistent_registry_base = (
                f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}"
            )

            _cleanup_old_images("arr-pytorch-base", persistent_registry_base, status)

            return True
        else:
            status("[red]✗ arr-pytorch-base image build failed![/red]")

            # Record to campaign stats (failure)
            build_end_time = time.time()
            build_duration_seconds = int(build_end_time - build_start_time)
            build_duration_minutes = build_duration_seconds / 60.0

            # Extract error message
            error_msg = result.stderr[:500] if result.stderr else "Build failed"

            try:
                from CLI.launch.mecha.campaign_stats import update_build_completion

                update_build_completion(
                    region=region,
                    build_id=extracted_build_id,
                    success=False,
                    duration_minutes=build_duration_minutes,
                    queue_wait_minutes=0.0,
                    error_message=error_msg,
                    duration_seconds=build_duration_seconds,
                    spot_price_per_hour=provision_price_at_start,  # Live C3-176 spot price at build START
                )
                status("[dim]   → Campaign stats: Build completed (FAILURE)[/dim]")
            except Exception as e:
                # Don't fail build if campaign stats fail
                status(f"[dim]⚠️  Campaign stats update failed: {str(e)[:100]}[/dim]")

            # Extract build ID from output
            build_id = None
            for line in build_output_lines:
                extracted = extract_build_id_from_output(line)
                if extracted:
                    build_id = extracted
                    break

            # Fetch actual build logs from CloudBuild (last 100 lines)
            if build_id:
                status(
                    f"[yellow]→ Fetching actual build logs for {build_id}...[/yellow]"
                )
                try:
                    log_result = run_gcloud_with_retry(
                        [
                            "gcloud",
                            "builds",
                            "log",
                            build_id,
                            f"--region={region}",
                            f"--project={project_id}",
                        ],
                        max_retries=3,
                        timeout=30,
                        operation_name="builds log --region={region}",
                    )
                    if log_result.returncode == 0 and log_result.stdout:
                        # Show last 100 lines of actual Docker build logs
                        log_lines = log_result.stdout.split("\n")
                        status(
                            "[yellow]→ Last 100 lines of Docker build logs:[/yellow]"
                        )
                        for line in log_lines[-100:]:
                            if line.strip():
                                status(f"  {escape(line)}")
                    else:
                        # Fallback: show gcloud output
                        status(
                            "[yellow]→ Could not fetch build logs, showing gcloud output:[/yellow]"
                        )
                        for line in result.stderr.split("\n")[-30:]:
                            if line.strip():
                                status(f"  {escape(line)}")
                except Exception as e:
                    # Fallback: show gcloud output
                    status(
                        f"[yellow]→ Log fetch failed ({escape(str(e))}), showing gcloud output:[/yellow]"
                    )
                    for line in result.stderr.split("\n")[-30:]:
                        if line.strip():
                            status(f"  {escape(line)}")
            else:
                # No build ID found - show gcloud output
                for line in result.stderr.split("\n")[-30:]:
                    if line.strip():
                        status(f"  {escape(line)}")
            return False

    except Exception as e:
        # Stop queue monitor if running
        if "queue_monitor" in locals() and queue_monitor:
            queue_monitor.stop()
        status(f"[red]arr-pytorch-base image build failed: {str(e)[:200]}[/red]")
        return False


def _handle_base_image(
    config: Dict[str, str],
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Ensure arr-ml-stack exists (build if Dockerfile changed)

    HASH DETECTION ENABLED:
    - Calculates Dockerfile hash from manifest
    - Checks if image with that hash exists
    - Rebuilds if Dockerfile changed (security patches!)
    - Tags with both :hash and :latest

    Args:
        config: Training configuration
        region: GCP region
        status: Status callback

    Returns:
        True if base ready (exists or built successfully)
        False if build failed
    """
    try:
        # Calculate Dockerfile hash from manifest
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        manifest_path = project_root / "Stack/arr-ml-stack/.image-manifest"

        # Hash all files listed in manifest
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # Image names
        # REGISTRY REGION: Images are STORED in us-central1 (PRIMARY_REGION)
        # BUILD REGION: Cloud Build RUNS in region parameter (from MECHA - cheapest!)
        project_id = config.get("GCP_PROJECT_ID", "")
        registry_name = "arr-coc-registry"
        registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
        base_image_hash = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-ml-stack:{dockerfile_hash}"
        base_image_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-ml-stack:latest"

        # Check if image with this hash exists
        status("")  # Visual separation
        status("⚡ Checking (ARR-ML-STACK)...")

        check_image = run_gcloud_with_retry(
            ["gcloud", "artifacts", "docker", "images", "describe", base_image_hash],
            max_retries=3,
            timeout=30,
            operation_name="artifacts docker images",
        )

        if check_image.returncode != 0:
            # Image with this hash doesn't exist → Dockerfile changed!
            status(
                "[yellow]⏳[/yellow] Building [bold cyan](ARR-ML-STACK)[/bold cyan] on Cloud Build (~10-15 min)..."
            )
            status(f"[dim]Dockerfile hash: {dockerfile_hash}[/dim]")
            status(
                f"[dim]→ View build logs: https://console.cloud.google.com/cloud-build/builds?project={project_id}[/dim]"
            )

            # Build with new hash
            # Pass region (from MECHA) to specify WHERE Cloud Build runs
            return _build_base_image(config, region, status)

        # arr-ml-stack exists with this hash → use it
        status(
            "[green]  ✓ Good [bold cyan](ARR-ML-STACK)[/bold cyan]! Will provide ML dependencies[/green]"
        )
        return True

    except Exception as e:
        status(f"[red]arr-ml-stack check failed: {str(e)[:200]}[/red]")
        return False


def _build_base_image(
    config: Dict[str, str],
    build_region: str,
    status: StatusCallback,
) -> bool:
    """
    Build arr-ml-stack using Cloud Build

    This is the ONE-TIME slow build (~10-15 min) that installs all heavy dependencies.
    Future training builds will be fast since they only add arr_coc code.

    Args:
        config: Training configuration
        build_region: GCP region for Cloud Build execution (from MECHA battle system)
        status: Status callback

    Returns:
        True if build succeeded
        False if build failed
    """
    from CLI.shared.performance_monitor import get_monitor

    monitor = get_monitor()

    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent

        # Get machine type from config (defaults to E2_HIGHCPU_8 if not set)
        machine_type = config.get("NON_MECHA_BUILD_MACHINE_TYPE", "E2_HIGHCPU_8")

        status(f"🏗️  Building arr-ml-stack... (using [cyan]{machine_type}[/cyan])")
        status(
            "[italic cyan]arr-ml-stack is the foundation for arr-trainer and speeds up subsequent builds![/italic cyan]"
        )
        status("")

        # Calculate hash for substitution
        manifest_path = project_root / "Stack/arr-ml-stack/.image-manifest"
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # Get project ID and artifact registry info
        project_id = config.get("GCP_PROJECT_ID", "")
        registry_region = PRIMARY_REGION  # us-central1 (from .constants)
        registry_name = "arr-coc-registry"
        base_image = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-ml-stack:{dockerfile_hash}"
        base_image_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-ml-stack:latest"

        # Track build timing for campaign stats
        build_start_time = time.time()

        # Create temp cloudbuild.yaml with both tags
        cloudbuild_config = f"""
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--pull'
      - '--no-cache'
      - '-t'
      - '{base_image}'
      - '-t'
      - '{base_image_latest}'
      - '-f'
      - 'Dockerfile'
      - '.'
    timeout: '1800s'

images:
  - '{base_image}'
  - '{base_image_latest}'
options:
  machineType: '{machine_type}'
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY
timeout: 3600s
"""
        build_context = project_root / "Stack/arr-ml-stack"
        cloudbuild_path = build_context / ".cloudbuild-base-temp.yaml"
        cloudbuild_path.write_text(cloudbuild_config)

        # Submit Cloud Build (MONITORED - slow operation!)
        op_id = monitor.start_operation(
            "build_base_image_cloudbuild", category="Docker"
        )
        result = run_gcloud_with_retry(
            [
                "gcloud",
                "builds",
                "submit",
                str(build_context),
                "--config=" + str(cloudbuild_path),
                # BUILD REGION: Where Cloud Build RUNS (from MECHA - cheapest region!)
                # This is NOT the registry region (that's always us-central1)
                f"--region={build_region}",  # us-west2, asia-southeast1, etc.
                "--timeout=60m",
            ],
            max_retries=3,
            timeout=2700,
            operation_name="builds submit --config=",
        )

        # Clean up temp file
        cloudbuild_path.unlink(missing_ok=True)

        monitor.end_operation(op_id)

        if result.returncode == 0:
            status("[green]✓ Base image built successfully![/green]")

            # Calculate hash for success message
            manifest_path = (
                project_root / "Stack/arr-ml-stack/.image-manifest"
            )
            dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

            # Diamond success celebration!
            status(
                f"[green]◇◇◇ Base image pushed: arr-ml-stack:latest (hash: {dockerfile_hash})[/green]"
            )
            status("[green]    \\o/ \\o\\ /o/ Base foundation ready![/green]")

            # CRITICAL: Explicitly tag as :latest to ensure scans pick up new image
            # Cloud Build YAML tags it, but we need to ensure the tag points to newest digest
            project_id = config.get("GCP_PROJECT_ID", "")

            # REGISTRY REGION: Where images are STORED (always us-central1!)
            # This is NOT the build region (which could be us-west2, asia-southeast1, etc.)
            registry_region = PRIMARY_REGION  # us-central1 (from .constants)
            registry_name = "arr-coc-registry"
            artifact_registry_base = (
                f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}"
            )

            status("⏳ Tagging as :latest...")
            try:
                # Cloud Build automatically tags as :latest
                # See .cloudbuild-base.yaml line 32: images: [arr-ml-stack:latest]
                # No local Docker operations needed - everything happens in Cloud Build!
                status("[green]✓ Tagged as :latest[/green]")

            except Exception as tag_error:
                # Cloud Build handles tagging, but catch any cleanup errors
                status(
                    f"[yellow]⚠️ Cleanup warning (non-critical): {str(tag_error)[:50]}[/yellow]"
                )

            # Cleanup old base images (keep only :latest)
            _cleanup_old_images("arr-ml-stack", artifact_registry_base, status)

            # Record successful build in campaign stats
            build_end_time = time.time()
            build_duration_seconds = int(build_end_time - build_start_time)
            build_duration_minutes = build_duration_seconds / 60.0

            try:
                from CLI.launch.mecha.campaign_stats import record_build_result

                record_build_result(
                    region=build_region,
                    success=True,
                    duration_minutes=build_duration_minutes,
                    queue_wait_minutes=0.0,
                    build_type="arr-ml-stack",
                    status="SUCCESS",
                    duration_seconds=build_duration_seconds,
                    spot_price_per_hour=None,  # Uses default CloudBuild machines (not C3 spot pool)
                )
            except Exception as e:
                # Show stats errors to user instead of silent fail
                status(f"[dim yellow]⚠️  Campaign stats error: {str(e)}[/dim yellow]")

            return True
        else:
            status("[red]✗ Base image build failed![/red]")

            # Record failed build in campaign stats
            build_end_time = time.time()
            build_duration_seconds = int(build_end_time - build_start_time)
            build_duration_minutes = build_duration_seconds / 60.0
            error_msg = result.stderr or result.stdout

            try:
                from CLI.launch.mecha.campaign_stats import record_build_result

                record_build_result(
                    region=build_region,
                    success=False,
                    duration_minutes=build_duration_minutes,
                    queue_wait_minutes=0.0,
                    error_message=error_msg[:500],
                    build_type="arr-ml-stack",
                    status="FAILURE",
                    duration_seconds=build_duration_seconds,
                    spot_price_per_hour=None,  # Uses default CloudBuild machines (not C3 spot pool)
                )
            except Exception as e:
                # Show stats errors to user instead of silent fail
                status(f"[dim yellow]⚠️  Campaign stats error: {str(e)}[/dim yellow]")

            # Show last 30 lines of error
            for line in result.stderr.split("\n")[-30:]:
                if line.strip():
                    status(f"  {escape(line)}")
            return False

    except Exception as e:
        status(f"[red]Base image build failed: {str(e)[:200]}[/red]")
        return False


def _handle_training_image(
    config: Dict[str, str],
    region: str,
    status: StatusCallback,
) -> Optional[str]:
    """
    Ensure arr-trainer exists (build if needed)

    Similar to _handle_runner_image but for the training container.
    Calculates Dockerfile hash to detect changes.
    Only rebuilds if Dockerfile changed.

    Args:
        config: Training configuration
        region: GCP region
        status: Status callback

    Returns:
        Image URI if ready
        None if image build/check failed

    Side effects:
        - Checks Artifact Registry exists
        - Builds image on Cloud Build if Dockerfile changed
        - Tags image as latest
    """
    from CLI.shared.performance_monitor import get_monitor

    monitor = get_monitor()

    try:
        # Calculate image content hash from .image-manifest (single source of truth!)
        script_dir = Path(__file__).parent  # CLI/launch/
        project_root = script_dir.parent.parent  # arr-coc-0-1/
        manifest_path = project_root / "Stack/arr-trainer/.image-manifest"

        # Hash all files listed in manifest
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # Image names
        # REGISTRY REGION: Images are STORED in us-central1 (PRIMARY_REGION)
        # BUILD REGION: Cloud Build RUNS in region parameter (from MECHA - cheapest!)
        project_id = config.get("GCP_PROJECT_ID", "")
        registry_name = "arr-coc-registry"  # SHARED registry for all ARR-COC prototypes
        registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
        training_image = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-trainer:{dockerfile_hash}"
        training_image_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-trainer:latest"

        # Use accumulator: Start both checks at once (registry + image)
        # Sequential: 10s + 10s = 20s | Accumulator: Both at once = 10s
        acc = GCloudAccumulator(max_workers=2)

        # Start both checks (non-blocking!)
        acc.start(
            key="registry",
            cmd=[
                "gcloud",
                "artifacts",
                "repositories",
                "describe",
                registry_name,
                "--location",
                registry_region,
            ],
            max_retries=3,
            timeout=10,
            operation_name="check registry",
        )

        acc.start(
            key="image",
            cmd=[
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                training_image,
            ],
            max_retries=3,
            timeout=10,
            operation_name="check image",
        )

        # Get registry result (waits if not ready)
        check_registry = acc.get("registry")

        if check_registry.returncode != 0:
            acc.shutdown()
            status(f"[red]✗ Artifact Registry '{registry_name}' not found[/red]")
            status("[yellow]⚠️  Infrastructure setup required[/yellow]")
            return None

        # Get image result (waits if not ready - but likely already done!)
        status("")  # Visual separation
        status("⚡ Checking (ARR-TRAINER)...")
        check_image = acc.get("image")
        acc.shutdown()

        if check_image.returncode != 0:
            # Image doesn't exist - Dockerfile changed or first build!
            # Get machine type from config (defaults to E2_HIGHCPU_8 if not set)
            machine_type = config.get("NON_MECHA_BUILD_MACHINE_TYPE", "E2_HIGHCPU_8")

            status(
                f"[yellow]⏳[/yellow] Building [bold cyan](ARR-TRAINER)[/bold cyan] on Cloud Build (~10-15 min, using [cyan]{machine_type}[/cyan])..."
            )
            status(
                "[italic cyan]arr-trainer is our primary training image![/italic cyan]"
            )
            status(f"[dim]Dockerfile hash: {dockerfile_hash}[/dim]")
            status(
                f"[dim]→ View build logs: https://console.cloud.google.com/cloud-build/builds?project={project_id}[/dim]"
            )

            # Use Cloud Build - need to create a minimal cloudbuild.yaml inline
            # Dockerfile needs full repo context for COPY commands

            # ═══════════════════════════════════════════════════════════════
            # 🏷️  TAGGING STRATEGY: Why Both :hash AND :latest
            # ═══════════════════════════════════════════════════════════════
            #
            # Cloud Build pushes BOTH tags atomically (in `images:` list below):
            #   arr-trainer:c3e00e3  ← Hash tag (Dockerfile fingerprint)
            #   arr-trainer:latest   ← Latest tag (current image)
            #
            # WHY BOTH TAGS?
            #   :hash   → Rebuild detection (Dockerfile changed?)
            #   :latest → Cascade detection (base:latest digest changed?)
            #             Cleanup protection (don't delete current image)
            #             Security scans (always scan :latest)
            #
            # CASCADE FLOW (how base rebuild triggers training rebuild):
            #   1. Base Dockerfile changes → hash changes → base rebuilds
            #   2. Base push tags arr-ml-stack:latest with new digest
            #   3. Training: FROM arr-ml-stack:latest + --pull flag
            #   4. Docker sees base:latest digest changed → pulls fresh base
            #   5. Training rebuilds with new base (cascade complete!)
            #
            # ATOMIC TAGGING (prevents race conditions):
            #   ✅ Cloud Build: Both tags pushed in same operation
            #   ❌ Post-build: Tag :latest AFTER (cleanup deletes before tag!)
            # ═══════════════════════════════════════════════════════════════

            cloudbuild_config = f"""
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--pull', '-t', '{training_image}', '-t', '{training_image_latest}', '-f', 'Stack/arr-trainer/Dockerfile', '.']
images:
  - '{training_image}'
  - '{training_image_latest}'
options:
  machineType: '{machine_type}'
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY
timeout: 3600s
"""
            # Write temp cloudbuild.yaml
            cloudbuild_path = project_root / ".cloudbuild-training.yaml"
            with open(cloudbuild_path, "w") as f:
                f.write(cloudbuild_config)

            try:
                # Track build timing for campaign stats
                build_start_time = time.time()

                # MONITORED - Training image build (10-15 min!)
                op_id = monitor.start_operation(
                    "build_training_image_cloudbuild", category="Docker"
                )
                build_result = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "builds",
                        "submit",
                        "--region",
                        region,
                        "--config",
                        str(cloudbuild_path),  # Use absolute path, not relative!
                        "--timeout",
                        "60m",
                        str(project_root),  # Source directory
                    ],
                    max_retries=3,
                    timeout=2100,
                    operation_name="builds submit --region",
                )
                monitor.end_operation(op_id)
            finally:
                # Clean up temp file
                try:
                    cloudbuild_path.unlink()
                except Exception:
                    pass

            if build_result.returncode != 0:
                status(f"[red]✗ Failed to build arr-trainer[/red]")
                error_msg = build_result.stderr or build_result.stdout

                # Record to campaign stats (failure)
                build_end_time = time.time()
                build_duration_seconds = int(build_end_time - build_start_time)
                build_duration_minutes = build_duration_seconds / 60.0

                try:
                    from CLI.launch.mecha.campaign_stats import record_build_result

                    record_build_result(
                        region=region,
                        success=False,
                        duration_minutes=build_duration_minutes,
                        queue_wait_minutes=0.0,
                        error_message=error_msg[:500],
                        build_type="arr-trainer",
                        status="FAILURE",
                        duration_seconds=build_duration_seconds,
                        spot_price_per_hour=None,  # Uses default CloudBuild machines (not C3 spot pool)
                    )
                except Exception as e:
                    # Show stats errors to user instead of silent fail
                    status(
                        f"[dim yellow]⚠️  Campaign stats error: {str(e)}[/dim yellow]"
                    )

                # Show FULL error message (no truncation) - users need to see complete errors!
                status(f"[red]Cloud Build Error:[/red]")
                for line in error_msg.split("\n")[:50]:  # Show first 50 lines
                    if line.strip():
                        status(f"  {escape(line)}")

                # Extract and highlight common errors
                if "Some files were not included" in error_msg:
                    status(
                        "[yellow]⚠️  Possible cause: .gcloudignore or .dockerignore excluding required files[/yellow]"
                    )
                if "ERROR:" in error_msg or "error:" in error_msg:
                    # Find the actual ERROR line
                    for line in error_msg.split("\n"):
                        if "ERROR:" in line or "error:" in line:
                            status(f"[red]→ {escape(line.strip())}[/red]")

                status(
                    f"[yellow]View full logs: https://console.cloud.google.com/cloud-build/builds?project={project_id}[/yellow]"
                )
                return None

            status("[green]✓ Training image built and pushed[/green]")

            # Record to campaign stats (success)
            build_end_time = time.time()
            build_duration_seconds = int(build_end_time - build_start_time)
            build_duration_minutes = build_duration_seconds / 60.0

            try:
                from CLI.launch.mecha.campaign_stats import record_build_result

                record_build_result(
                    region=region,
                    success=True,
                    duration_minutes=build_duration_minutes,
                    queue_wait_minutes=0.0,
                    build_type="arr-trainer",
                    status="SUCCESS",
                    duration_seconds=build_duration_seconds,
                    spot_price_per_hour=None,  # Uses default CloudBuild machines (not C3 spot pool)
                )
            except Exception as e:
                # Show stats errors to user instead of silent fail
                status(f"[dim yellow]⚠️  Campaign stats error: {str(e)}[/dim yellow]")

            # Diamond success celebration!
            status(
                f"[green]◇◇◇ Training image pushed: arr-trainer:latest (hash: {dockerfile_hash})[/green]"
            )
            status("[green]    \\o/ \\o\\ /o/ Primary training image ready![/green]")

            # Tag as latest using Docker (not gcloud - see arr-ml-stack tagging for why)
            status("⏳ Tagging as latest...")
            try:
                # Configure Docker auth
                # REGISTRY REGION: Authenticate to us-central1 (where images are stored)
                subprocess.run(
                    [
                        "gcloud",
                        "auth",
                        "configure-docker",
                        f"{registry_region}-docker.pkg.dev",  # us-central1
                        "--quiet",
                    ],
                    capture_output=True,
                    timeout=30,
                )

                # Pull, tag, push
                subprocess.run(
                    ["docker", "pull", training_image], capture_output=True, timeout=120
                )
                subprocess.run(
                    ["docker", "tag", training_image, training_image_latest],
                    capture_output=True,
                    timeout=10,
                )
                subprocess.run(
                    ["docker", "push", training_image_latest],
                    capture_output=True,
                    timeout=120,
                )
            except Exception:
                pass  # Non-critical

            # Cleanup old training images (keep only :latest)
            # REGISTRY REGION: Clean up images from us-central1 (where they're stored)
            artifact_registry_base = (
                f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}"
            )
            _cleanup_old_images("arr-trainer", artifact_registry_base, status)

        else:
            status(
                "[green]  ✓ Good [bold cyan](ARR-TRAINER)[/bold cyan]! Will run our training code[/green]"
            )

        # Return the image URI for wandb launch --docker-image
        return training_image

    except Exception as e:
        status(f"[red]Training image setup failed: {str(e)[:400]}[/red]")
        return None


def _handle_runner_image(
    config: Dict[str, str],
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Ensure arr-vertex-launcher exists (build if needed)

    Extracted from: screen.py lines 717-985

    Calculates Dockerfile hash to detect changes.
    Only rebuilds if Dockerfile changed.

    Args:
        config: Training configuration
        region: GCP region
        status: Status callback

    Returns:
        True if image ready
        False if image build/check failed

    Side effects:
        - Checks Artifact Registry exists
        - Builds image on Cloud Build if Dockerfile changed
        - Tags image as latest
        - Cleans up old images
    """
    try:
        # Calculate image content hash from .image-manifest (single source of truth!)
        script_dir = Path(__file__).parent  # CLI/launch/
        project_root = script_dir.parent.parent  # arr-coc-0-1/
        manifest_path = (
            project_root / "Stack/arr-vertex-launcher/.image-manifest"
        )

        # Hash all files listed in manifest
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # Image names
        # REGISTRY REGION: Images are STORED in us-central1 (PRIMARY_REGION)
        # BUILD REGION: Cloud Build RUNS in region parameter (from MECHA - cheapest!)
        project_id = config.get("GCP_PROJECT_ID", "")
        registry_name = "arr-coc-registry"  # SHARED registry for all ARR-COC prototypes
        registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
        runner_image = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-vertex-launcher:{dockerfile_hash}"
        runner_image_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-vertex-launcher:latest"

        # Use accumulator: Start both checks at once (registry + image)
        # Sequential: 10s + 10s = 20s | Accumulator: Both at once = 10s
        acc = GCloudAccumulator(max_workers=2)

        # Start both checks (non-blocking!)
        acc.start(
            key="registry",
            cmd=[
                "gcloud",
                "artifacts",
                "repositories",
                "describe",
                registry_name,
                "--location",
                registry_region,
            ],
            max_retries=3,
            timeout=10,
            operation_name="check registry",
        )

        acc.start(
            key="image",
            cmd=[
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                runner_image,
            ],
            max_retries=3,
            timeout=10,
            operation_name="check image",
        )

        # Get registry result (waits if not ready)
        check_registry = acc.get("registry")

        if check_registry.returncode != 0:
            acc.shutdown()
            # Registry doesn't exist - setup not run!
            status(f"[red]✗ Artifact Registry '{registry_name}' not found[/red]")
            status("[yellow]⚠️  Infrastructure setup required[/yellow]")
            status(
                "[dim]Run Setup screen (press 's' key or '3' from ../home) to create infrastructure[/dim]"
            )
            return False

        # Get image result (waits if not ready - but likely already done!)
        status("")  # Visual separation
        status("⚡ Checking (ARR-VERTEX-LAUNCHER)...")
        check_image = acc.get("image")
        acc.shutdown()

        if check_image.returncode != 0:
            # Image doesn't exist - Dockerfile changed or first build!
            # Get machine type from config (defaults to E2_HIGHCPU_8 if not set)
            machine_type = config.get("NON_MECHA_BUILD_MACHINE_TYPE", "E2_HIGHCPU_8")

            status(
                f"[yellow]⏳[/yellow] Building [bold cyan](ARR-VERTEX-LAUNCHER)[/bold cyan] on Cloud Build (~5-10 min, using [cyan]{machine_type}[/cyan])..."
            )
            status(
                "[italic magenta]This launches training jobs on Vertex AI![/italic magenta]"
            )
            status(f"[dim]Dockerfile hash: {dockerfile_hash}[/dim]")
            status(
                f"[dim]→ View build logs: https://console.cloud.google.com/cloud-build/builds?project={project_id}[/dim]"
            )

            # ═══════════════════════════════════════════════════════════════
            # 🏷️  TAGGING STRATEGY: Why Both :hash AND :latest
            # ═══════════════════════════════════════════════════════════════
            #
            # Cloud Build pushes BOTH tags atomically (in `images:` list below):
            #   arr-vertex-launcher:653d183  ← Hash tag (Dockerfile fingerprint)
            #   arr-vertex-launcher:latest   ← Latest tag (current image)
            #
            # WHY BOTH TAGS?
            #   :hash   → Rebuild detection (Dockerfile changed?)
            #   :latest → Cloud Run job updates (always use :latest)
            #             Cleanup protection (don't delete current image)
            #
            # ATOMIC TAGGING (prevents race conditions):
            #   ✅ Cloud Build: Both tags pushed in same operation
            #   ❌ Post-build: Tag :latest AFTER (cleanup deletes before tag!)
            # ═══════════════════════════════════════════════════════════════

            build_context = project_root / "Stack/arr-vertex-launcher"

            # Track build timing for campaign stats
            build_start_time = time.time()

            # Create temp cloudbuild.yaml with both tags
            cloudbuild_config = f"""
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--pull', '-t', '{runner_image}', '-t', '{runner_image_latest}', '.']
images:
  - '{runner_image}'
  - '{runner_image_latest}'
options:
  machineType: '{machine_type}'
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY
timeout: 600s
"""
            cloudbuild_path = build_context / ".cloudbuild-runner.yaml"
            with open(cloudbuild_path, "w") as f:
                f.write(cloudbuild_config)

            build_result = run_gcloud_with_retry(
                [
                    "gcloud",
                    "builds",
                    "submit",
                    "--region",
                    region,
                    "--config",
                    str(cloudbuild_path),  # Use absolute path!
                    "--timeout",
                    "10m",
                    str(build_context),
                ],
                max_retries=3,
                timeout=720,
                operation_name="builds submit --region",
            )

            if build_result.returncode != 0:
                status(f"[red]✗ Failed to build arr-vertex-launcher[/red]")
                error_msg = build_result.stderr or build_result.stdout

                # Record failed build in campaign stats
                build_end_time = time.time()
                build_duration_seconds = int(build_end_time - build_start_time)
                build_duration_minutes = build_duration_seconds / 60.0

                try:
                    from CLI.launch.mecha.campaign_stats import record_build_result

                    record_build_result(
                        region=region,
                        success=False,
                        duration_minutes=build_duration_minutes,
                        queue_wait_minutes=0.0,
                        error_message=error_msg[:500],
                        build_type="arr-vertex-launcher",
                        status="FAILURE",
                        duration_seconds=build_duration_seconds,
                        spot_price_per_hour=None,  # Uses default CloudBuild machines (not C3 spot pool)
                    )
                except Exception as e:
                    # Show stats errors to user instead of silent fail
                    status(
                        f"[dim yellow]⚠️  Campaign stats error: {str(e)}[/dim yellow]"
                    )

                # Show FULL error message (no truncation) - users need to see complete errors!
                status(f"[red]Cloud Build Error:[/red]")
                for line in error_msg.split("\n")[:50]:  # Show first 50 lines
                    if line.strip():
                        status(f"  {escape(line)}")

                # Extract and highlight common errors
                if "Some files were not included" in error_msg:
                    status(
                        "[yellow]⚠️  Possible cause: .gcloudignore or .dockerignore excluding required files[/yellow]"
                    )
                if "ERROR:" in error_msg or "error:" in error_msg:
                    # Find the actual ERROR line
                    for line in error_msg.split("\n"):
                        if "ERROR:" in line or "error:" in line:
                            status(f"[red]→ {escape(line.strip())}[/red]")

                project_id = config.get("GCP_PROJECT_ID", "")
                status(
                    f"[yellow]View full logs: https://console.cloud.google.com/cloud-build/builds?project={project_id}[/yellow]"
                )
                return False

            status("[green]✓ Runner image built and pushed[/green]")

            # Diamond success celebration!
            status(
                f"[green]◇◇◇ Runner image pushed: arr-vertex-launcher:latest (hash: {dockerfile_hash})[/green]"
            )
            status("[green]    \\o/ \\o\\ /o/ Launch orchestrator ready![/green]")

            # Tag as latest
            status("⏳ Tagging as latest...")
            subprocess.run(
                [
                    "gcloud",
                    "artifacts",
                    "docker",
                    "tags",
                    "add",
                    runner_image,
                    runner_image_latest,
                ],
                capture_output=True,
                timeout=30,
            )

            # Get image metadata
            status("⏳ Fetching image details...")
            try:
                image_info = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "artifacts",
                        "docker",
                        "images",
                        "describe",
                        runner_image,
                        "--format=json",
                    ],
                    max_retries=3,
                    timeout=10,
                    operation_name="artifacts docker images",
                )

                if image_info.returncode == 0:
                    info = json.loads(image_info.stdout)

                    # Extract key info
                    digest = info.get("image_summary", {}).get("digest", "N/A")[:16]

                    # Get tags
                    # REGISTRY REGION: Tags are in us-central1 (where images are stored)
                    tags_result = run_gcloud_with_retry(
                        [
                            "gcloud",
                            "artifacts",
                            "docker",
                            "tags",
                            "list",
                            f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-vertex-launcher",
                            "--filter",
                            f"version={dockerfile_hash}",
                            "--format=value(tag)",
                        ],
                        max_retries=3,
                        timeout=10,
                        operation_name="artifacts docker tags",
                    )
                    tags = (
                        tags_result.stdout.strip().split("\n")
                        if tags_result.returncode == 0
                        else [dockerfile_hash]
                    )
                    tags_str = ", ".join(t for t in tags if t)

                    status("")
                    status("[cyan]━━━ Image Details ━━━[/cyan]")
                    status(
                        f"  [dim]Repository:[/dim] {registry_name}/arr-vertex-launcher"
                    )
                    status(f"  [dim]Digest:[/dim] {digest}...")
                    status(f"  [dim]Tags:[/dim] {tags_str}")
                    status(f"  [dim]Location:[/dim] {registry_region} (Iowa)")
                    status("")
            except Exception as e:
                status(
                    f"[yellow]⚠️ Could not fetch image details: {str(e)[:50]}[/yellow]"
                )

            status(
                "[green]  ✓ Good [bold cyan](ARR-VERTEX-LAUNCHER)[/bold cyan]! Will launch our runs[/green]"
            )

            # Cleanup old images (keep only :latest)
            # REGISTRY REGION: Clean up images from us-central1 (where they're stored)
            artifact_registry_base = (
                f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}"
            )
            _cleanup_old_images("arr-vertex-launcher", artifact_registry_base, status)

            # Record successful build in campaign stats
            build_end_time = time.time()
            build_duration_seconds = int(build_end_time - build_start_time)
            build_duration_minutes = build_duration_seconds / 60.0

            try:
                from CLI.launch.mecha.campaign_stats import record_build_result

                record_build_result(
                    region=region,
                    success=True,
                    duration_minutes=build_duration_minutes,
                    queue_wait_minutes=0.0,
                    build_type="arr-vertex-launcher",
                    status="SUCCESS",
                    duration_seconds=build_duration_seconds,
                    spot_price_per_hour=None,  # Uses default CloudBuild machines (not C3 spot pool)
                )
            except Exception as e:
                # Show stats errors to user instead of silent fail
                status(f"[dim yellow]⚠️  Campaign stats error: {str(e)}[/dim yellow]")
        else:
            # Image with this hash exists - skip build!
            status(
                "[green]  ✓ Good [bold cyan](ARR-VERTEX-LAUNCHER)[/bold cyan]! Will launch our runs[/green]"
            )

        return True

    except Exception as e:
        status(f"[red]Runner image setup failed: {str(e)[:300]}[/red]")
        return False


def _create_cloud_run_job(
    config: Dict[str, str],
    region: str,
    job_name: str,
    status: StatusCallback,
) -> bool:
    """
    Create or update Cloud Run Job

    Extracted from: screen.py lines 986-1187

    Creates job if doesn't exist, updates if config changed.

    Args:
        config: Training configuration
        region: GCP region
        job_name: Cloud Run Job name
        status: Status callback

    Returns:
        True if job ready
        False if job creation/update failed

    Side effects:
        - Creates Cloud Run Job if doesn't exist
        - Updates job if config changed (image, args, env, SA)
    """
    try:
        max_retries = (
            3  # Consistent retry count for all gcloud commands in this function
        )

        project_id = config.get("GCP_PROJECT_ID", "")
        entity = config.get("WANDB_ENTITY", "")
        queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
        sa_name = "arr-coc-sa"  # SHARED service account (matches setup)
        sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"
        secret_name = "wandb-api-key"

        # Calculate runner image name from .image-manifest (MUST match _build_and_push_runner_image!)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        manifest_path = (
            project_root / "Stack/arr-vertex-launcher/.image-manifest"
        )

        # Hash all files listed in manifest (guaranteed to match build function!)
        dockerfile_hash = _hash_files_from_manifest(manifest_path, project_root)

        # REGISTRY REGION: Runner image is STORED in us-central1 (PRIMARY_REGION)
        registry_name = "arr-coc-registry"  # SHARED registry for all ARR-COC prototypes
        registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
        runner_image = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-vertex-launcher:{dockerfile_hash}"

        create_cmd = [
            "gcloud",
            "run",
            "jobs",
            "create",
            job_name,
            "--image",
            runner_image,
            "--region",
            region,
            f"--args=-q,{queue},-e,{entity},--max-jobs,-1",
            f"--set-secrets=WANDB_API_KEY={secret_name}:latest",
            f"--service-account={sa_email}",
            f"--set-env-vars=CLOUDSDK_COMPUTE_REGION={region}",
            "--max-retries",
            "0",
            "--task-timeout",
            "240m",  # 4 hours safety net (wrapper has 30min idle timeout, but runner can process multiple jobs)
            "--memory",
            "2Gi",
            "--cpu",
            "2",
        ]

        # Try to create (with retries for transient failures)
        create_result = None
        for attempt in range(max_retries):
            create_result = subprocess.run(
                create_cmd, capture_output=True, text=True, timeout=30
            )

            if (
                create_result.returncode == 0
                or "already exists" in create_result.stderr
            ):
                break  # Success or job exists (both good!)

            # Failed - retry if not last attempt
            if attempt < max_retries - 1:
                status(
                    f"[dim]  → Retry {attempt + 1}/{max_retries - 1} after create failure...[/dim]"
                )
                time.sleep(1)

        job_ready = False
        if create_result and create_result.returncode == 0:
            # Created successfully
            job_ready = True
            status("[green]  ✓ Runner created[/green]")
        elif "already exists" in create_result.stderr:
            # Already exists - check if config changed
            # Describe current job config (with retries for transient failures)
            describe_cmd = [
                "gcloud",
                "run",
                "jobs",
                "describe",
                job_name,
                "--region",
                region,
                "--format",
                "json",
            ]

            # Retry up to 3 times for transient failures
            describe_result = None
            for attempt in range(max_retries):
                describe_result = subprocess.run(
                    describe_cmd, capture_output=True, text=True, timeout=10
                )

                if describe_result.returncode == 0:
                    break  # Success!

                # Failed - retry if not last attempt
                if attempt < max_retries - 1:
                    status(
                        f"[dim]  → Retry {attempt + 1}/{max_retries - 1} after describe failure...[/dim]"
                    )
                    time.sleep(1)

            needs_update = False
            if describe_result and describe_result.returncode == 0:
                try:
                    current_config = json.loads(describe_result.stdout)

                    # Extract current values
                    template_spec = (
                        current_config.get("spec", {})
                        .get("template", {})
                        .get("spec", {})
                        .get("template", {})
                        .get("spec", {})
                    )
                    container = template_spec.get("containers", [{}])[0]
                    current_image = container.get("image", "")
                    current_args = container.get("args", [])
                    current_env = container.get("env", [])
                    current_sa = template_spec.get("serviceAccountName", "")
                    current_timeout = template_spec.get("timeoutSeconds", "")

                    # Build desired args and timeout
                    desired_args = ["-q", queue, "-e", entity, "--max-jobs", "-1"]
                    desired_timeout_seconds = (
                        14400  # 240m (4 hours) - safety net for semi-persistent runner
                    )

                    # Check if config changed
                    image_match = current_image == runner_image
                    args_match = current_args == desired_args
                    env_match = any(
                        e.get("name") == "CLOUDSDK_COMPUTE_REGION"
                        and e.get("value") == region
                        for e in current_env
                    )
                    sa_match = current_sa == sa_email

                    # GCP returns timeout as "14400s" (with 's' suffix), normalize both values
                    current_timeout_seconds = (
                        int(current_timeout.rstrip("s")) if current_timeout else 0
                    )
                    timeout_match = current_timeout_seconds == desired_timeout_seconds

                    needs_update = not (
                        image_match
                        and args_match
                        and env_match
                        and sa_match
                        and timeout_match
                    )

                    if not needs_update:
                        # Silent success - no need to announce config unchanged
                        job_ready = True
                except (json.JSONDecodeError, KeyError, IndexError):
                    needs_update = True
            else:
                needs_update = True

            # Only update if config changed
            if needs_update:
                status("[yellow]  ⏳ Updating runner...[/yellow]")
                update_cmd = [
                    "gcloud",
                    "run",
                    "jobs",
                    "update",
                    job_name,
                    "--image",
                    runner_image,
                    "--region",
                    region,
                    f"--args=-q,{queue},-e,{entity},--max-jobs,-1",
                    f"--set-secrets=WANDB_API_KEY={secret_name}:latest",
                    f"--service-account={sa_email}",
                    f"--set-env-vars=CLOUDSDK_COMPUTE_REGION={region}",
                    "--max-retries",
                    "0",
                    "--task-timeout",
                    "240m",  # 4 hours safety net (semi-persistent runner, matches create command)
                    "--memory",
                    "2Gi",
                    "--cpu",
                    "2",
                ]

                # Retry up to 3 times for transient failures
                update_result = None
                for attempt in range(max_retries):
                    update_result = subprocess.run(
                        update_cmd, capture_output=True, text=True, timeout=30
                    )

                    if update_result.returncode == 0:
                        break  # Success!

                    # Failed - retry if not last attempt
                    if attempt < max_retries - 1:
                        status(
                            f"[dim]  → Retry {attempt + 1}/{max_retries - 1} after update failure...[/dim]"
                        )
                        time.sleep(1)

                if update_result and update_result.returncode == 0:
                    job_ready = True
                    status("[green]  ✓ Runner updated[/green]")
                else:
                    error_msg = update_result.stderr if update_result else "No result"
                    status(
                        f"[red]❌ Job update failed after {max_retries} attempts[/red]"
                    )
                    for line in error_msg.split("\n")[:15]:
                        if line.strip():
                            status(f"[red]  {escape(line)}[/red]")
                    return False
        else:
            # Creation failed for another reason
            status("[red]❌ Job creation failed[/red]")
            for line in create_result.stderr.split("\n")[:10]:
                if line.strip():
                    status(f"[red]  {escape(line)}[/red]")
            return False

        return job_ready

    except Exception as e:
        status(f"[red]Cloud Run Job setup failed: {str(e)[:300]}[/red]")
        return False


def _runner_is_alive(
    job_name: str,
    region: str,
    queue_name: str,
    status: StatusCallback,
) -> bool:
    """
    Check if a Cloud Run execution is currently RUNNING and monitoring the correct queue.

    This enables semi-persistent runner behavior:
    - If runner alive AND monitoring correct queue: Skip execution
    - If runner dead OR monitoring wrong queue: Start new execution

    Args:
        job_name: Cloud Run Job name
        region: GCP region
        queue_name: W&B queue name we're submitting to
        status: Status callback for progress updates

    Returns:
        True if RUNNING execution exists AND monitoring correct queue
        False if no RUNNING execution OR monitoring wrong queue

    Side effects:
        - Outputs check status via status()
    """
    try:
        # Silent runner check - no output during detection
        # Only output when we know the result (found runner OR need to start one)

        max_retries = (
            3  # Consistent retry count for all gcloud commands in this function
        )

        # First, check the job's current queue configuration (with retries for transient failures)
        describe_cmd = [
            "gcloud",
            "run",
            "jobs",
            "describe",
            job_name,
            "--region",
            region,
            "--format",
            "json",
        ]

        # Retry up to 3 times for transient failures (network hiccups, API rate limits)
        describe_result = None
        for attempt in range(max_retries):
            describe_result = subprocess.run(
                describe_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if describe_result.returncode == 0:
                break  # Success!

            # Failed - retry if not last attempt
            if attempt < max_retries - 1:
                status(
                    f"[dim]  → Retry {attempt + 1}/{max_retries - 1} after describe failure...[/dim]"
                )
                time.sleep(1)  # Brief pause before retry

        if describe_result and describe_result.returncode == 0:
            try:
                job_config = json.loads(describe_result.stdout)
                template_spec = (
                    job_config.get("spec", {})
                    .get("template", {})
                    .get("spec", {})
                    .get("template", {})
                    .get("spec", {})
                )
                container = template_spec.get("containers", [{}])[0]
                current_args = container.get("args", [])

                # Check if job is configured for the correct queue
                # Args format: ["-q", "queue-name", "-e", "entity", "--max-jobs", "-1"]
                job_queue = None
                for i, arg in enumerate(current_args):
                    if arg == "-q" and i + 1 < len(current_args):
                        job_queue = current_args[i + 1]
                        break

                if job_queue != queue_name:
                    status(f"[yellow]⚠ Job configured for different queue[/yellow]")
                    status(f"[dim]  → Job queue: {job_queue}[/dim]")
                    status(f"[dim]  → Target queue: {queue_name}[/dim]")
                    status("[dim]  → Will start new runner for correct queue[/dim]")
                    return False

            except (json.JSONDecodeError, KeyError, IndexError):
                # If we can't parse job config, be safe and start new runner
                status("[dim]⚠ Couldn't verify job queue configuration[/dim]")
                status("[dim]  → Starting new runner to be safe[/dim]")
                return False
        else:
            # describe command failed - can't verify queue configuration
            status("[dim]⚠ Failed to verify job queue configuration[/dim]")
            status(
                f"[dim]  → gcloud describe error: {describe_result.stderr[:100]}[/dim]"
            )
            status("[dim]  → Starting new runner to be safe[/dim]")
            return False

        # Now check for RUNNING execution (fetch JSON and parse like monitor does)
        # NOTE: --filter=status=RUNNING doesn't work! Status is complex nested dict
        #       Running executions have: conditions[type=Completed].status = 'Unknown'
        cmd = [
            "gcloud",
            "run",
            "jobs",
            "executions",
            "list",
            f"--job={job_name}",
            f"--region={region}",
            "--limit=5",  # Check recent executions (catches PENDING/RUNNING)
            "--format=json",
        ]

        # Retry up to 3 times for transient failures
        result = None
        for attempt in range(max_retries):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                break  # Success

            # Failed - retry if not last attempt
            if attempt < max_retries - 1:
                status(
                    f"[dim]  → Retry {attempt + 1}/{max_retries - 1} after executions list failure...[/dim]"
                )
                time.sleep(1)

        # Parse JSON to find RUNNING execution (same logic as monitor/core.py)
        if result and result.returncode == 0 and result.stdout.strip():
            try:
                executions = json.loads(result.stdout)
                for execution in executions:
                    # Extract status from conditions (same as monitor logic)
                    conditions = execution.get("status", {}).get("conditions", [])
                    is_running = False

                    for condition in conditions:
                        if condition.get("type") == "Completed":
                            # Running execution: Completed status = 'Unknown'
                            if condition.get("status") == "Unknown":
                                is_running = True
                                break

                    if is_running:
                        execution_name = execution.get("metadata", {}).get("name", "")
                        if execution_name:
                            status(
                                f"[green]✓ Active runner found: {execution_name}[/green]"
                            )
                            status(f"[dim]  → Monitoring queue: {queue_name}[/dim]")
                            status("[dim]  → Runner will pick up job from queue[/dim]")
                            return True
            except (json.JSONDecodeError, KeyError):
                pass  # Fall through to "no runner"

        # No runner found - return False silently
        # The caller will output "Starting new runner..." message
        return False

    except Exception as e:
        # On error, assume no runner (safe default - will start new one)
        status(f"[dim]⚠ Runner check failed: {str(e)[:100]}[/dim]")
        status("[dim]  → Starting new runner to be safe[/dim]")
        return False


def _execute_runner(
    config: Dict[str, str],
    region: str,
    job_name: str,
    status: StatusCallback,
) -> Optional[str]:
    """
    Execute Cloud Run Job (start runner)

    NEW: Semi-persistent design!
    - Checks if runner already exists (RUNNING execution)
    - If exists: Skip execution (job queued, existing runner picks it up)
    - If not: Start new execution (30min idle timeout)

    Extracted from: screen.py lines 1189-1280

    Starts the runner execution and returns execution name.

    Args:
        config: Training configuration
        region: GCP region
        job_name: Cloud Run Job name
        status: Status callback

    Returns:
        Execution name if successful
        None if execution failed to start OR if runner already exists

    Side effects:
        - Starts Cloud Run Job execution (if no runner exists)
        - Outputs execution name via status()
    """
    try:
        max_retries = (
            3  # Consistent retry count for all gcloud commands in this function
        )

        status("⚡ Attempting to run the queue...")

        # NEW: Check if runner already alive AND monitoring correct queue (semi-persistent design)
        queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

        if _runner_is_alive(job_name, region, queue_name, status):
            status(
                "\n  [bold green]✓ Run queued![/bold green]"
            )
            return None  # Skip execution - existing runner handles it

        project_id = config.get("GCP_PROJECT_ID", "")

        status("\n  ⏳ Starting new runner...")
        status(
            f"[dim]  → View runner logs: https://console.cloud.google.com/run/jobs/details/us-central1/{job_name}?project={project_id}[/dim]"
        )

        # Start job execution (with robust 3-retry logic)
        execute_cmd = [
            "gcloud",
            "run",
            "jobs",
            "execute",
            job_name,
            "--region",
            region,
            "--format",
            "value(metadata.name)",  # Return just the execution name
        ]

        # Use helper with good exceptions
        execute_result = run_gcloud_with_retry(
            execute_cmd,
            max_retries=3,
            timeout=300,  # 5 minutes - Cloud Run can take time to start
            operation_name=f"execute Cloud Run job {job_name}",
        )

        # Extract execution name from output (format returns just the name)
        execution_name = execute_result.stdout.strip()

        if not execution_name:
            raise Exception(
                f"Could not determine execution name from execute command.\n"
                f"Execute stdout: {execute_result.stdout[:500]}\n"
                f"Execute stderr: {execute_result.stderr[:500]}"
            )

        # Wait for runner to become available (initial 2s, then poll every 2s up to 20s total)
        # GCP needs minimum 2s, often 4-8s for runner to become available
        status(f"[dim]  → Waiting for runner ({execution_name}) to start...[/dim]")
        time.sleep(2)  # Initial wait - GCP minimum

        execution_ready = False
        for attempt in range(9):  # 9 more attempts × 2s = 18s more (20s total)
            # Quick check if runner is available
            try:
                check_exec = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "run",
                        "jobs",
                        "executions",
                        "describe",
                        execution_name,
                        "--region",
                        region,
                        "--format",
                        "value(metadata.name)",
                    ],
                    max_retries=1,  # Single quick attempt
                    timeout=5,
                    operation_name="check runner availability",
                )
                if check_exec.returncode == 0 and check_exec.stdout.strip():
                    execution_ready = True
                    elapsed = 2 + (attempt + 1) * 2
                    status(f"[dim]  ✓ Runner ready after {elapsed}s[/dim]")
                    break
            except Exception:
                pass  # Keep trying

            time.sleep(2)

        if not execution_ready:
            status("[dim]  ⏳ Waited 20s (GCP logs may be delayed)[/dim]")

        # SILENT SUCCESS: No messages between "Runner ready" and "Runner completed"
        # Background monitoring happening silently:
        #   - Cloud Logging: Checking for errors
        #   - Vertex AI API: Polling for successful job creation every 5s
        #   - Runner status: Checking Cloud Run execution status
        # ONLY output on error or completion!

        return execution_name

    except Exception as e:
        status(f"[red]Failed to execute runner: {str(e)[:300]}[/red]")
        return None


def _wait_for_job_submission(
    config: Dict[str, str],
    region: str,
    job_name: str,
    execution_name: str,
    status: StatusCallback,
    run_id: str,
) -> Tuple[bool, str]:
    """
    Wait for Vertex AI job submission via FAST API polling.

    FAST PATH ONLY: Simple Vertex AI API check every 3-5s for up to 120s.
    No Cloud Logging, no complex error parsing - just quick job detection.

    For detailed diagnostics, use `python CLI/cli.py monitor` or
    import runner_diagnostics.parse_runner_logs_for_errors() directly.

    Args:
        config: Training configuration (contains GPU region, project ID)
        region: Cloud Run region (NOT the same as GPU region!)
        job_name: Cloud Run job name
        execution_name: Cloud Run execution name
        status: Status callback

    Returns:
        Tuple of (success: bool, message: str)
        - success=True if job found in RUNNING/PENDING/QUEUED state
        - success=False if job failed or 120s timeout reached
    """
    project_id = config.get("GCP_PROJECT_ID")
    gpu_region = config.get("TRAINING_GPU_REGION", "us-central1")

    # SILENT MONITORING: Polling Vertex AI API every 5s for up to 120s
    # - Looking for job with RUNNING/PENDING/QUEUED state
    # - Also checking Cloud Run execution for runner failures
    # - NO output during polling, only on completion or error!

    start_time = time.time()
    last_check = 0
    timeout = 120  # 2 minutes
    poll_interval = 5  # Check every 5 seconds

    runner_has_failed = False  # Flag to prevent infinite loop on failure detection

    while (time.time() - start_time) < timeout:
        elapsed = int(time.time() - start_time)

        # Skip all checks if we already detected runner failure
        if runner_has_failed:
            time.sleep(1)
            continue

        # CHECK 1: Cloud Run execution status (detect runner failures quickly!)
        try:
            exec_status = run_gcloud_with_retry(
                [
                    "gcloud",
                    "run",
                    "jobs",
                    "executions",
                    "describe",
                    execution_name,
                    f"--region={region}",
                    "--format=json",
                ],
                max_retries=1,
                timeout=5,
                operation_name="check runner status",
            )

            if exec_status.returncode == 0 and exec_status.stdout:
                exec_info = json.loads(exec_status.stdout)
                conditions = exec_info.get("status", {}).get("conditions", [])

                for condition in conditions:
                    if condition.get("type") in ["Completed", "Ready"]:
                        if condition.get("status") == "True":
                            # Runner completed successfully
                            # Keep waiting for Vertex AI job to appear in API
                            pass
                        elif condition.get("status") == "False":
                            # Runner FAILED - set flag to trigger error handling below
                            runner_has_failed = True
        except Exception as e:
            pass  # Ignore errors from runner status check

        # Handle runner failure OUTSIDE try block (so exceptions aren't swallowed!)
        if runner_has_failed:
            status("")
            status(f"  [yellow]🏁 Runner completed with error ({elapsed}s)[/yellow]")
            status("")
            status("  [red]❌ Runner job submission failed![/red]")
            status("")
            status("  ⏳ Waiting for logs (up to 5 mins)...")

            # Smart loop: Wait for Cloud Logging to ingest (typically 30-90s delay)
            # Try for up to 5 minutes or until original timeout expires
            from .runner_diagnostics import fetch_detailed_error_context

            fetch_timeout = min(300, timeout - elapsed)  # Wait up to 5 minutes
            fetch_start = time.time()
            error_logs = []

            while (time.time() - fetch_start) < fetch_timeout and not error_logs:
                try:
                    error_logs = fetch_detailed_error_context(
                        execution_name=execution_name,
                        project_id=project_id,
                        region=region,
                        context_lines=100,
                        status_callback=lambda x: None,  # Silent fetching
                    )
                    if error_logs:
                        break
                except Exception:
                    pass

                # Wait 3s before retry (Cloud Logging ingestion delay)
                if not error_logs:
                    time.sleep(3)

            status("")
            if error_logs:
                # Got error logs - dump them all with proper formatting!
                fetch_elapsed = int(time.time() - fetch_start)
                total_elapsed = elapsed + fetch_elapsed

                # Extract short error message (like monitor does)
                short_error = None
                for line in error_logs:
                    if "Machine type" in line and "is not supported" in line:
                        short_error = (
                            line.split("wandb: ERROR")[-1].strip()
                            if "wandb: ERROR" in line
                            else line.strip()
                        )
                        break
                    elif (
                        "InvalidArgument:" in line
                        or "PermissionDenied:" in line
                        or "NotFound:" in line
                    ):
                        short_error = (
                            line.split("wandb: ERROR")[-1].strip()
                            if "wandb: ERROR" in line
                            else line.strip()
                        )
                        break
                    elif "wandb: ERROR" in line and not short_error:
                        # Fallback: any wandb error
                        short_error = line.split("wandb: ERROR")[-1].strip()

                # Clean up error message - remove prefixes like "details = "
                if short_error:
                    # Remove 'details = "..."' wrapper
                    if short_error.startswith('details = "') and short_error.endswith(
                        '"'
                    ):
                        short_error = short_error[len('details = "') : -1]
                    # Remove standalone 'details = ' prefix
                    elif short_error.startswith("details = "):
                        short_error = short_error[len("details = ") :]

                status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                status("[bold yellow]Runner Error Log[/bold yellow]")
                status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                status("")
                for line in error_logs:
                    status(line)
                status("")
                status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                status(
                    f"[dim]Runner failed at {elapsed}s, logs fetched after {fetch_elapsed}s (total: {total_elapsed}s)[/dim]"
                )
                status(f"[dim]Execution: {execution_name} | Region: {region}[/dim]")
                if short_error:
                    status("")
                    status(
                        "   ╔═══════════════════════════════════════════════════════"
                    )
                    status(f"   ║ [bold red]{short_error}[/bold red]")
                    status(
                        "   ╚═══════════════════════════════════════════════════════"
                    )
                    status("")
                status(
                    "[dim]The runner encountered an error while trying to submit the Vertex AI job.[/dim]"
                )

                # Add Cloud Run job logs link (simpler, always works)
                run_url = f"https://console.cloud.google.com/run/detail/{region}/vertex-ai-launcher?project={project_id}"
                status(
                    f"[dim]View in console: [link={run_url}]Cloud Run Job Logs[/link][/dim]"
                )

                status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                status("")
            else:
                # Timeout - Cloud Logging hasn't ingested yet (waited 5 minutes)
                status(
                    "[yellow]⏱️  Error log fetching timeout (5 mins) - Cloud Logging not ready yet[/yellow]"
                )
                status("")
                status("[bold]View error logs when ready:[/bold]")
                status("  [cyan]python CLI/cli.py monitor[/cyan]")

            status("")
            return False, "Runner execution failed"

        # CHECK 2: Vertex AI API (detect successful job creation)
        # Polling silently every 5s - no output unless job found or error!
        if (time.time() - last_check) >= poll_interval:
            try:
                # List most recent Vertex AI jobs
                vertex_cmd = run_gcloud_with_retry(
                    [
                        "gcloud",
                        "ai",
                        "custom-jobs",
                        "list",
                        f"--region={gpu_region}",
                        f"--project={project_id}",
                        "--limit=10",
                        "--sort-by=~createTime",  # Most recent first
                        "--format=json",
                    ],
                    max_retries=2,
                    timeout=10,
                    operation_name="Vertex AI job poll",
                )

                if vertex_cmd.returncode == 0 and vertex_cmd.stdout.strip():
                    jobs = json.loads(vertex_cmd.stdout)

                    if jobs:
                        for job in jobs:
                            display_name = job.get("displayName", "")
                            job_state = job.get("state", "")
                            job_id = job.get("name", "").split("/")[-1]

                            # Check if this job matches our execution (by timestamp proximity)
                            create_time_str = job.get("createTime", "")
                            if create_time_str:
                                from datetime import datetime, timezone

                                create_time = datetime.fromisoformat(
                                    create_time_str.replace("Z", "+00:00")
                                )
                                age_seconds = (
                                    datetime.now(timezone.utc) - create_time
                                ).total_seconds()

                                # Only consider jobs created in last 10 minutes
                                if age_seconds < 600:
                                    # Job found! Silent success unless it's a failure state
                                    # (age: {int(age_seconds)}s, state: {job_state})

                                    if job_state == "JOB_STATE_SUCCEEDED":
                                        # Training completed (unlikely during launch)
                                        _show_success_rocketship(status, run_id)
                                        return True, "Job completed successfully"

                                    elif job_state == "JOB_STATE_FAILED":
                                        # Job failed at API level
                                        error_info = job.get("error", {})
                                        error_msg = error_info.get(
                                            "message", "Unknown error"
                                        )
                                        status("[red]❌ Vertex AI job failed![/red]")
                                        status(f"[dim]Job ID: {job_id}[/dim]")
                                        status(f"[yellow]{error_msg}[/yellow]")
                                        status("")
                                        status("[bold]For detailed diagnostics:[/bold]")
                                        status(
                                            "  [cyan]python CLI/cli.py monitor[/cyan]"
                                        )
                                        status("")
                                        return False, f"Job failed: {error_msg}"

                                    elif job_state in [
                                        "JOB_STATE_RUNNING",
                                        "JOB_STATE_PENDING",
                                        "JOB_STATE_QUEUED",
                                    ]:
                                        # SUCCESS! Job submitted!
                                        _show_success_rocketship(status, run_id)
                                        return True, "Job submitted successfully"

            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
                status(f"   [dim]API check error: {str(e)[:40]}[/dim]")

            last_check = time.time()

        # Brief sleep
        time.sleep(1)

    # Timeout reached
    status("")
    status("[yellow]⏱️  Timeout (120s) - Job not detected in Vertex AI[/yellow]")
    status("")
    status("[dim]Possible reasons:[/dim]")
    status("[dim]  • Job submission taking longer than usual[/dim]")
    status("[dim]  • Runner encountered an error[/dim]")
    status("[dim]  • W&B Launch agent hasn't picked up job yet[/dim]")
    status("")
    status("[bold]Check detailed diagnostics:[/bold]")
    status("  [cyan]python CLI/cli.py monitor[/cyan]")
    status("")
    return False, "Timeout waiting for job submission"


def _show_success_rocketship(status: StatusCallback, run_id: str, run_name: str = None):
    """Show the 3-6-9 rocketship success art."""
    # Extract just the queue name if run_id has format "queued in {queue_name}"
    display_id = run_id
    if run_id.startswith("queued in "):
        display_id = run_id.replace("queued in ", "")

    # ═══════════════════════════════════════════════════════════════
    # 🎉 SUCCESS CELEBRATION - 3-6-9 Triangle Rocketship
    # ═══════════════════════════════════════════════════════════════
    status("")
    status("[cyan]███▓▓▒░ ✅ Training Run Invoked In The Cloud! ░▒▓▓███[/cyan]")
    status("")
    status("[bold cyan]         ◇[/bold cyan]")
    status("[bold cyan]        ◇ ◇[/bold cyan]")
    status("[bold cyan]       ◇ 9[/bold cyan]")
    status("[bold cyan]      ∿ 6 9[/bold cyan]")
    status("[bold cyan]     ◇ 3 6 9[/bold cyan]")
    status("[bold cyan]    ∿ ◇ ∿ ◇[/bold cyan]")
    status("[bold cyan]   Task → Cloud ✓[/bold cyan]")
    status("")
    status("[bold cyan]   Vision ∿ Forge 🔥[/bold cyan]")
    status("[bold cyan]   ◇ Embrace 🤗 ∿ Lightning ⚡ ◇[/bold cyan]")
    status("[bold cyan]   Four ◇ One ∿ Rivers ◇ Ocean[/bold cyan]")
    status("[bold cyan]   ∿ Pattern ◇ Complete ∿ ✨[/bold cyan]")
    status("")

    # Show returned W&B ID in triple brackets
    # If missing or doesn't match our generated name, show BIG warning
    if not run_id or run_id == "var missing!!":
        status(f"[red]   ((( var missing!! W&B did not return a run ID )))[/red]")
        show_warning = True
    else:
        status(f"[dim]   ((( {display_id} )))[/dim]")
        # Check if returned ID matches our generated name
        show_warning = run_name and run_id != run_name and display_id != run_name

    # Show BIG WARNING if IDs don't match or ID is missing
    if show_warning:
        status("")
        status(
            "[yellow]   ⚠️  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/yellow]"
        )
        status(
            "[yellow]   ⚠️  WARNING: W&B returned ID doesn't match generated name![/yellow]"
        )
        status(f"[yellow]   ⚠️  Expected: {run_name}[/yellow]")
        status(f"[yellow]   ⚠️  Received: {display_id}[/yellow]")
        status(
            "[yellow]   ⚠️  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/yellow]"
        )

    status("")
    status(
        f"[bold cyan]   Monitor progress:[/bold cyan] [cyan]python CLI/cli.py monitor[/cyan]"
    )
    status("")


def _verify_all_images_in_registry(config: Dict, region: str, status) -> bool:
    """
    Verify all 4 images (ARR-PYTORCH-BASE, ARR-ML-STACK, ARR-TRAINER, ARR-VERTEX-LAUNCHER) exist in Artifact Registry.

    GO/NO-GO check before proceeding with launch.

    Returns:
        True if all 4 images found, False if any missing
    """
    status("")

    # REGISTRY REGION: Images are STORED in us-central1 (PRIMARY_REGION)
    # This function receives MECHA's region, but registry is always us-central1
    project_id = config.get("GCP_PROJECT_ID")
    registry_name = config.get("ARTIFACT_REGISTRY_NAME", "arr-coc-registry")
    persistent_registry = "arr-coc-registry-persistent"  # PyTorch base only
    registry_region = PRIMARY_REGION  # us-central1 (Artifact Registry location)
    registry_base = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}"
    persistent_registry_base = (
        f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}"
    )

    # Expected images (4-tier architecture)
    # PyTorch stored in PERSISTENT registry, others in regular registry
    required_images = {
        "arr-pytorch-base": f"{persistent_registry_base}/arr-pytorch-base",
        "arr-ml-stack": f"{registry_base}/arr-ml-stack",
        "arr-trainer": f"{registry_base}/arr-trainer",
        "arr-vertex-launcher": f"{registry_base}/arr-vertex-launcher",
    }

    results = {}
    all_pass = True

    # Use accumulator pattern: Start all checks at once, get results when ready
    # Sequential: 4 images × 10s = 40s | Accumulator: All at once = 10s (4× faster!)
    acc = GCloudAccumulator(max_workers=4)

    # Start all image checks (non-blocking!)
    for image_type, image_path in required_images.items():
        acc.start(
            key=image_type,
            cmd=[
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "list",
                image_path,
                "--format=value(package)",
                "--limit=1",
            ],
            max_retries=3,
            timeout=10,
            operation_name=f"check {image_type}",
        )

    # Get results (waits only if not ready yet)
    for image_type in required_images.keys():
        try:
            result = acc.get(image_type)

            # If we get any output, image exists (at least one tag)
            if result.returncode == 0 and required_images[image_type] in result.stdout:
                results[image_type] = True
            else:
                results[image_type] = False
                all_pass = False

        except subprocess.TimeoutExpired:
            results[image_type] = False
            all_pass = False
        except Exception as e:
            results[image_type] = False
            all_pass = False

    # Clean up accumulator
    acc.shutdown()

    # Build status line with check marks - full diagonal flow cascade
    status_symbols = []
    short_names = {
        "arr-pytorch-base": "pytorch",
        "arr-ml-stack": "ml-stack",
        "arr-trainer": "trainer",
        "arr-vertex-launcher": "launcher",
    }

    for image_type in [
        "arr-pytorch-base",
        "arr-ml-stack",
        "arr-trainer",
        "arr-vertex-launcher",
    ]:
        short_name = short_names[image_type]
        if results.get(image_type, False):
            status_symbols.append(f"[green]✅ {short_name}[/green]")
        else:
            status_symbols.append(f"[red]❌ {short_name}[/red]")

    # Full width diagonal cascade - all lines same width, left-aligned
    status(
        f"[dim]███▓▓▒░[/dim] {status_symbols[0]}  {status_symbols[1]} [dim]░▒▓▓███[/dim]"
    )
    status(
        f"[dim]███▓▓▒░[/dim] {status_symbols[2]}  {status_symbols[3]} [dim]░▒▓▓███[/dim]"
    )

    if all_pass:
        status("[bold green]███▓▓▒░ ✅ Stack Complete! ░▒▓▓███[/bold green]")
        status("")
        return True
    else:
        status("[bold red]❌ NO-GO: Missing images in Artifact Registry[/bold red]")
        status("")
        status("[yellow]Missing images:[/yellow]")
        for image_type, found in results.items():
            if not found:
                status(f"  • {image_type}: {required_images[image_type]}")
        status("")
        status("[dim]This should not happen after successful builds.[/dim]")
        status("[dim]Check Cloud Build logs or rebuild manually.[/dim]")
        return False


# _verify_gpu_quota() DELETED (2025-11-16)
# _auto_request_gpu_quota() DELETED (2025-11-16)
#
# These functions checked COMPUTE ENGINE quotas (NVIDIA_T4_GPUS, PREEMPTIBLE_NVIDIA_T4_GPUS)
# But Vertex AI Custom Training uses VERTEX AI quotas (custom_model_training_nvidia_t4_gpus)
# These are COMPLETELY DIFFERENT quota namespaces!
#
# Problem: Validation/auto-request used wrong quota → false positives → launch failures
# Solution: Users request Vertex AI quotas manually via infra screen instructions
#
# See detailed analysis in:
# - GPU_QUOTA_COMPLETE_FIX_PLAN.md
# - GPU_QUOTA_METRICS_REFERENCE.md
# - VERTEX_AI_GPU_QUOTA_BUG_REPORT.md
#
# Removed functions:
# - _verify_gpu_quota() was 178 lines (deleted earlier)
# - _auto_request_gpu_quota() was 170 lines (deleted 2025-11-16)
# Total: 348 lines of dead/wrong quota checking code removed
