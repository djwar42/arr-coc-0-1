"""
Pricing Infrastructure Configuration

Shared constants for pricing infrastructure (used by both setup and teardown).

Architecture:
- Project-wide config (PROJECT_ID, REGION) → Loaded from .training file
- Pricing-specific names (FUNCTION_NAME, etc.) → Defined here
- Both setup and teardown import from this single source
"""

# <claudes_code_comments>
# ** Function List **
# get_required_fields() - returns validation dict of required pricing fields
#
# ** Technical Review **
# Centralized config for pricing infrastructure with schema validation support.
# Loads PROJECT_ID and REGION from .training file, defines pricing-specific constants
# (Cloud Function, Scheduler, Artifact Registry names). PRICING_SCHEMA is the canonical
# source of truth for pricing data structure - defines expected fields and validation rules.
# Schema auto-detection: get_required_fields() extracts validation dict for bootstrap/
# Cloud Function to check if existing pricing matches current schema (triggers refetch
# if fields missing/empty). Used by setup, teardown, and pricing validation.
# </claudes_code_comments>

from ...config.constants import load_training_config  # Go up to training.cli.constants (3 levels: pricing -> shared -> cli)

# ============================================================================
# Project-Wide Config (from .training file - canonical source)
# ============================================================================

config = load_training_config()
PROJECT_ID = config["GCP_PROJECT_ID"]
REGION = config["GCP_ROOT_RESOURCE_REGION"]


# ============================================================================
# Pricing Infrastructure Names (domain-specific - OK to define here)
# ============================================================================

# Cloud Function that runs pricing updates
FUNCTION_NAME = "arr-coc-pricing-runner"

# Cloud Scheduler job that triggers the function
SCHEDULER_JOB = "arr-coc-pricing-scheduler"

# How often to update pricing (minutes)
SCHEDULER_INTERVAL_MINUTES = 20  # Matches */20 * * * * cron schedule

# Artifact Registry repository for pricing data storage
REPOSITORY = "arr-coc-pricing"

# Pricing data package name
PACKAGE = "gcp-pricing"

# ============================================================================
# Pricing Data Schema (canonical definition - auto-detected by validation)
# ============================================================================

# Expected pricing data schema
# This MUST match the structure created in Cloud Function and bootstrap code
# Add new fields here when expanding pricing system
PRICING_SCHEMA = {
    "updated": {"type": "timestamp", "required": True, "should_have_data": False},
    "c3_machines": {"type": "dict", "required": True, "should_have_data": True},
    "e2_machines": {"type": "dict", "required": True, "should_have_data": True},
    "gpus_spot": {"type": "dict", "required": True, "should_have_data": True},
    "gpus_ondemand": {"type": "dict", "required": True, "should_have_data": True},
}

# Quick helper to get just field names that should have data
def get_required_fields():
    """Returns dict of {field_name: should_have_data} for validation"""
    return {
        field: spec["should_have_data"]
        for field, spec in PRICING_SCHEMA.items()
    }
