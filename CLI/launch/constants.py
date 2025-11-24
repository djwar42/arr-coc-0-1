"""
GCP Constants for ARR-COC Training Infrastructure

IMPORTANT REGIONAL DECISION (2025-01-11):
    PRIMARY_REGION remains us-central1 (Iowa) with Regional Blast redundancy

    Background: us-central1 experienced C3 worker pool service issues (2025-01-11)
    - C3 pools in us-central1: Failed/stuck for 20+ minutes ❌
    - C3 pools in us-east4: Created successfully in ~1 minute ✅
    - C3 pools in europe-west1: Created successfully in ~1 minute ✅

    Solution: Regional Worker Pool Blast System
    - Blasts ALL 9 C3 regions simultaneously
    - 15 minute timeout
    - Selects randomly from successful pools
    - Provides automatic failover if us-central1 fails

    Impact:
    - us-central1 remains PRIMARY_REGION for consistency
    - Regional Blast handles transient regional issues automatically
    - Launch succeeds as long as ANY region succeeds (high availability)
    - Full Mecha Fleet goal: All 9 regions RUNNING

Historical context:
    - us-central1 (Iowa) has been primary since launch (2024-2025)
    - Regional Mecha Blast system added 2025-01-11
    - Provides automatic failover across all 9 C3 regions globally
"""

# PRIMARY REGION for all GCP operations
# us-central1 (Iowa) remains primary - Regional Blast provides redundancy
PRIMARY_REGION = "us-central1"  # Iowa, USA

# Note: C3 worker pools in us-central1 experienced issues (2025-01-11)
# Regional Worker Pool Blast handles this by creating pools in ALL regions
# If us-central1 fails, system auto-selects from us-east4, europe-west1, etc.

# Registry paths (region-specific)
def get_registry_path(project_id: str, registry_name: str = "arr-coc-images", region: str = None) -> str:
    """
    Get Artifact Registry path for Docker images.

    Args:
        project_id: GCP project ID
        registry_name: Artifact Registry name (default: arr-coc-images)
        region: Override PRIMARY_REGION if needed

    Returns:
        Full registry path: {region}-docker.pkg.dev/{project}/{registry}

    Example:
        get_registry_path("my-project")
        → "us-east4-docker.pkg.dev/my-project/arr-coc-images"
    """
    region = region or PRIMARY_REGION
    return f"{region}-docker.pkg.dev/{project_id}/{registry_name}"

# Cloud Run paths (region-specific)
def get_cloud_run_console_url(project_id: str, job_name: str, region: str = None) -> str:
    """
    Get Cloud Run console URL for viewing logs.

    Args:
        project_id: GCP project ID
        job_name: Cloud Run job name
        region: Override PRIMARY_REGION if needed

    Returns:
        Console URL for Cloud Run job
    """
    region = region or PRIMARY_REGION
    return f"https://console.cloud.google.com/run/jobs/details/{region}/{job_name}?project={project_id}"
