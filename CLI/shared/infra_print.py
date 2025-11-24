"""Unified Infrastructure Display - Single source of truth for displaying ALL infrastructure"""

# <claudes_code_comments>
# ** Function List **
# display_infrastructure(info, use_rich) - Unified infrastructure display for CLI and TUI
#   ├─ green(text) - Rich markup helper (green = exists)
#   ├─ red(text) - Rich markup helper (red = missing)
#   ├─ dim(text) - Rich markup helper (dim = optional)
#   ├─ bold(text) - Rich markup helper (bold = headers)
#   └─ yellow(text) - Rich markup helper (yellow = warnings)
#
# ** Technical Review **
# This module provides ONE unified function for displaying complete infrastructure status.
# Takes data from verify_all_infrastructure() and formats it for display.
#
# Core Design:
# - Single display function eliminates duplicate logic across CLI/TUI
# - use_rich flag: True = Rich markup (CLI), False = plain text (TUI adds styling)
# - Displays ALL 10 infrastructure items in consistent format:
#   1. Billing (enabled/disabled - checked FIRST to prevent cascade failures)
#   2. GCS Buckets (count + regional locations)
#   3. Artifact Registry (arr-coc-registry)
#   4. Persistent Registry (arr-coc-registry-persistent)
#   5. Service Account (email - truncated if >50 chars)
#   6. W&B Secret (wandb-api-key in GCP Secret Manager)
#   7. W&B Queue (vertex-ai-queue)
#   8. W&B Project (arr-coc-0-1)
#   9. HuggingFace Repo (optional)
#   10. Local Key File (service account JSON)
#
# Color Coding:
# - Green (✓): Item exists and ready
# - Red (✗): Item missing (needs setup)
# - Dim (○): Optional item or created on-demand
# - Yellow: Warnings (billing disabled, etc.)
#
# Smart Features:
# - Shows "100% COMPLETE!" banner when all 6 critical items exist
# - Handles billing disabled gracefully (notes on GCP items)
# - Displays bucket regions for multi-region deployments
# - Truncates long service account emails for display
# - Shows key filename only (security - hides full path)
#
# Usage:
#   # CLI commands (with Rich markup)
#   output = display_infrastructure(info, use_rich=True)
#   print(output)  # Rich renders [bold green] etc.
#
#   # TUI screens (plain text, TUI adds styling)
#   output = display_infrastructure(info, use_rich=False)
#   Static(output)  # TUI parses ✓/✗/○ and adds colors
#
# Used By:
# - CLI/cli.py:253 (setup command)
# - CLI/cli.py:315 (infra command)
# - CLI/setup/screen.py (TUI setup screen)
# - CLI/infra/screen.py (TUI infrastructure screen)
# </claudes_code_comments>

from typing import Dict


def display_infrastructure(info: Dict, use_rich: bool = True) -> str:
    """
    Display complete infrastructure status

    Args:
        info: Dict from verify_all_infrastructure()
        use_rich: If True, uses Rich markup (CLI). If False, plain text (TUI adds its own styling)

    Returns:
        Formatted string with complete infrastructure status
    """

    # Helper functions for colored output
    def green(text: str) -> str:
        """Green for exists"""
        return f"[bold green]{text}[/bold green]" if use_rich else text

    def red(text: str) -> str:
        """Red for missing"""
        return f"[bold red]{text}[/bold red]" if use_rich else text

    def dim(text: str) -> str:
        """Dim for optional/notes"""
        return f"[dim]{text}[/dim]" if use_rich else text

    def bold(text: str) -> str:
        """Bold for section headers"""
        return f"[bold]{text}[/bold]" if use_rich else text

    def yellow(text: str) -> str:
        """Yellow for warnings"""
        return f"[bold yellow]{text}[/bold yellow]" if use_rich else text

    # Build output
    lines = []

    # Add blank line separator (after verify_all_infrastructure status messages)
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # BILLING STATUS (Critical - shows first!)
    # ═══════════════════════════════════════════════════════════════════════

    lines.append(bold("💳 Billing:"))
    billing = info.get("billing", {})
    billing_enabled = billing.get("enabled")
    billing_error = billing.get("error", "")
    billing_note = billing.get("note", "")

    if billing_enabled is True:
        lines.append(green("  ✓ Enabled"))
    elif billing_enabled is False:
        lines.append(red("  ✗ Disabled"))
        if billing_note:
            lines.append(yellow(f"     {billing_note}"))
    else:
        # Unknown status
        lines.append(dim("  ? Unknown"))
        if billing_error:
            lines.append(dim(f"     {billing_error}"))

    # ═══════════════════════════════════════════════════════════════════════
    # GCP INFRASTRUCTURE
    # ═══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append(bold("📦 GCP Infrastructure:"))

    gcp = info.get("gcp", {})

    # GCS Buckets
    buckets = gcp.get("buckets", {})
    bucket_count = buckets.get("count", 0)
    bucket_list = buckets.get("buckets", [])
    bucket_note = buckets.get("note", "")

    if bucket_count == 0:
        if bucket_note:
            lines.append(dim(f"  ○ GCS Buckets: None ({bucket_note})"))
        else:
            lines.append(dim("  ○ GCS Buckets: None (created on-demand)"))
    else:
        lines.append(green(f"  ✓ GCS Buckets: {bucket_count}"))
        for bucket in bucket_list:
            name = bucket.get("name", "")
            location = bucket.get("location", "")
            lines.append(dim(f"      • {name} ({location})"))

    # Artifact Registry
    registry = gcp.get("registry", {})
    registry_exists = registry.get("exists", False)
    registry_name = registry.get("name", "arr-coc-registry")
    registry_note = registry.get("note", "")

    if registry_exists:
        lines.append(green(f"  ✓ Registry: {registry_name}"))
    else:
        if registry_note:
            lines.append(red(f"  ✗ Registry: {registry_name} ({registry_note})"))
        else:
            lines.append(red(f"  ✗ Registry: {registry_name}"))

    # Persistent Registry
    persistent = gcp.get("persistent_registry", {})
    persistent_exists = persistent.get("exists", False)
    persistent_name = persistent.get("name", "arr-coc-registry-persistent")
    persistent_note = persistent.get("note", "")

    if persistent_exists:
        lines.append(green(f"  ✓ Persistent Registry: {persistent_name}"))
    else:
        if persistent_note:
            lines.append(red(f"  ✗ Persistent Registry: {persistent_name} ({persistent_note})"))
        else:
            lines.append(red(f"  ✗ Persistent Registry: {persistent_name}"))

    # Service Account
    sa = gcp.get("service_account", {})
    sa_exists = sa.get("exists", False)
    sa_email = sa.get("email", "")
    sa_note = sa.get("note", "")

    if sa_exists:
        # Truncate email for display
        email_display = sa_email if len(sa_email) < 50 else sa_email[:47] + "..."
        lines.append(green(f"  ✓ Service Account: {email_display}"))
    else:
        if sa_note:
            lines.append(red(f"  ✗ Service Account: Not created ({sa_note})"))
        else:
            lines.append(red("  ✗ Service Account: Not created"))

    # W&B Secret
    wandb_secret = gcp.get("wandb_secret", {})
    secret_exists = wandb_secret.get("exists", False)
    secret_name = wandb_secret.get("name", "wandb-api-key")

    if secret_exists:
        lines.append(green(f"  ✓ W&B Secret: {secret_name}"))
    else:
        lines.append(red(f"  ✗ W&B Secret: {secret_name}"))

    # ═══════════════════════════════════════════════════════════════════════
    # GCP ADVANCED (APIs, IAM, VPC)
    # ═══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append(bold("🔧 GCP Advanced:"))

    # APIs
    apis = gcp.get("apis", {})
    apis_enabled = apis.get("all_enabled", False)
    apis_count = apis.get("count", "0/8")
    apis_missing = apis.get("missing", [])

    if apis_enabled:
        lines.append(green(f"  ✓ APIs: {apis_count} enabled"))
    else:
        lines.append(red(f"  ✗ APIs: {apis_count} enabled"))
        if apis_missing and len(apis_missing) <= 3:
            # Show missing APIs if only a few
            missing_short = [api.split(".")[0] for api in apis_missing]
            lines.append(dim(f"     Missing: {', '.join(missing_short)}"))

    # Cloud Build IAM
    cloudbuild_iam = gcp.get("cloudbuild_iam", {})
    iam_granted = cloudbuild_iam.get("granted", False)
    iam_count = cloudbuild_iam.get("count", "0/2")

    if iam_granted:
        lines.append(green(f"  ✓ Cloud Build IAM: {iam_count} roles"))
    else:
        lines.append(red(f"  ✗ Cloud Build IAM: {iam_count} roles"))

    # VPC Peering (optional)
    vpc_peering = gcp.get("vpc_peering", {})
    vpc_exists = vpc_peering.get("exists", False)
    vpc_count = vpc_peering.get("count", 0)

    if vpc_exists:
        lines.append(green(f"  ✓ VPC Peering: {vpc_count} configured"))
    else:
        lines.append(dim("  ○ VPC Peering: Not needed (using public egress)"))

    # ═══════════════════════════════════════════════════════════════════════
    # CLOUD BUILD INFRASTRUCTURE (Worker Pools)
    # ═══════════════════════════════════════════════════════════════════════

    cloud_build = info.get("cloud_build", {})
    worker_pools = cloud_build.get("worker_pools", {})

    if worker_pools:
        lines.append("")
        lines.append(bold("☁️ Cloud Build:"))
        lines.append("  Worker Pools:")

        for region, pool_data in worker_pools.items():
            if pool_data.get("exists"):
                pool_name = pool_data.get("name", "pytorch-mecha-pool")
                machine = pool_data.get("machine_type", "c3-standard-176")
                lines.append(green(f"    ✓ {region}: {pool_name} ({machine})"))
            else:
                lines.append(red(f"    ✗ {region}: Not created"))

    # ═══════════════════════════════════════════════════════════════════════
    # GPU QUOTAS (Vertex AI)
    # ═══════════════════════════════════════════════════════════════════════

    quotas = info.get("quotas", {})
    gpu_quotas = quotas.get("vertex_gpu", {})

    if gpu_quotas:
        lines.append("")
        lines.append(bold("🎮 GPU Quotas (Vertex AI):"))

        for region, quota_data in gpu_quotas.items():
            lines.append(f"  {region}:")

            granted = quota_data.get("granted", [])
            pending = quota_data.get("pending", [])

            # Show granted GPUs
            if granted:
                for gpu in granted:
                    gpu_name = gpu.get("gpu_name", "Unknown")
                    spot_suffix = " (Spot)" if gpu.get("is_spot") else ""
                    count = gpu.get("quota_limit", 0)
                    lines.append(green(f"    ✓ {gpu_name}{spot_suffix}: {count} GPUs"))
            else:
                lines.append(red("    ✗ No GPU quotas granted"))

            # Show pending GPUs (need to request)
            if pending:
                lines.append(yellow("    ⚠️ Need to request:"))
                for gpu in pending[:3]:  # Show max 3 to avoid clutter
                    gpu_name = gpu.get("gpu_name", "Unknown")
                    spot_suffix = " (Spot)" if gpu.get("is_spot") else ""
                    lines.append(dim(f"      • {gpu_name}{spot_suffix}"))

    # ═══════════════════════════════════════════════════════════════════════
    # C3 QUOTAS (Cloud Build)
    # ═══════════════════════════════════════════════════════════════════════

    c3_quotas = quotas.get("c3_build", {})

    if c3_quotas:
        lines.append("")
        lines.append(bold("☁️ C3 Quotas (Cloud Build):"))

        # Separate granted vs pending (consistent with GPU quotas display)
        granted_regions = [(r, q) for r, q in c3_quotas.items() if q]
        pending_regions = [r for r, q in c3_quotas.items() if not q]

        # Show granted regions
        if granted_regions:
            for region, quota_data in granted_regions:
                vcpus = quota_data.get("vcpus", 0)
                machine = quota_data.get("machine_type", "c3-standard-176")
                lines.append(green(f"  ✓ {region}: {vcpus} vCPUs ({machine})"))
        else:
            lines.append(red("  ✗ No C3 quotas granted"))

        # Show pending regions (need to request)
        if pending_regions:
            lines.append(yellow("  ⚠️ Need to request:"))
            for region in pending_regions:
                lines.append(dim(f"    • {region}"))

    # ═══════════════════════════════════════════════════════════════════════
    # W&B INFRASTRUCTURE
    # ═══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append(bold("🔄 W&B Infrastructure:"))

    wandb = info.get("wandb", {})

    # Queue
    queue = wandb.get("queue", {})
    queue_exists = queue.get("exists", False)
    queue_name = queue.get("name", "vertex-ai-queue")

    if queue_exists:
        lines.append(green(f"  ✓ Queue: {queue_name}"))
    else:
        lines.append(red(f"  ✗ Queue: {queue_name}"))

    # Project
    project = wandb.get("project", {})
    project_exists = project.get("exists", False)
    project_name = project.get("name", "arr-coc-0-1")

    if project_exists:
        lines.append(green(f"  ✓ Project: {project_name}"))
    else:
        lines.append(dim(f"  ○ Project: {project_name} (created on first run)"))

    # ═══════════════════════════════════════════════════════════════════════
    # HUGGINGFACE
    # ═══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append(bold("🤗 HuggingFace:"))

    hf = info.get("hf", {})
    repo = hf.get("repo", {})
    repo_exists = repo.get("exists", False)
    repo_id = repo.get("id", "")

    if repo_id:
        if repo_exists:
            lines.append(green(f"  ✓ Repo: {repo_id}"))
        else:
            lines.append(dim(f"  ○ Repo: {repo_id} (not created yet)"))
    else:
        lines.append(dim("  ○ Repo: Not configured"))

    # ═══════════════════════════════════════════════════════════════════════
    # LOCAL FILES
    # ═══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append(bold("📁 Local Files:"))

    local = info.get("local", {})
    key_exists = local.get("key_file_exists", False)
    key_path = local.get("key_path", "")

    if key_exists:
        # Show just the filename for security
        from pathlib import Path
        filename = Path(key_path).name if key_path else "key.json"
        lines.append(green(f"  ✓ Service Account Key: {filename}"))
    else:
        lines.append(red("  ✗ Service Account Key: Not found"))

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    # Check if ALL critical infrastructure exists (11 items total)
    critical_items = [
        # Original 6 critical items
        billing_enabled is True,
        gcp.get("registry", {}).get("exists", False),
        gcp.get("persistent_registry", {}).get("exists", False),
        gcp.get("service_account", {}).get("exists", False),
        gcp.get("wandb_secret", {}).get("exists", False),
        wandb.get("queue", {}).get("exists", False),

        # NEW: 5 additional critical items (from manifest)
        gcp.get("apis", {}).get("all_enabled", False),           # All 8 APIs enabled
        gcp.get("cloudbuild_iam", {}).get("granted", False),     # Cloud Build IAM roles
        any(p.get("exists", False) for p in worker_pools.values()) if worker_pools else False,  # At least 1 worker pool
        any(len(q.get("granted", [])) > 0 for q in gpu_quotas.values()) if gpu_quotas else False,  # At least 1 GPU quota
        any(c is not None for c in c3_quotas.values()) if c3_quotas else False,  # At least 1 C3 quota
    ]

    all_critical_exists = all(critical_items)

    if all_critical_exists:
        lines.append("")
        lines.append(green("═" * 60))
        lines.append(green("✅ INFRASTRUCTURE 100% COMPLETE!"))
        lines.append(green("═" * 60))

    return "\n".join(lines)
