"""
Launch Validation - GO/NO-GO Checks Before Job Submission

Prevents misconfigured jobs from being submitted to Vertex AI.
Validates that .training config matches what will actually be sent to Vertex AI.
"""

# <claudes_code_comments>
# ** Function List **
# validate_launch_config(config) - GO/NO-GO validation before job submission
# get_launch_spec_config(config) - Build Vertex AI launch spec from .training
# format_validation_report(errors) - Format validation errors into readable report
#
# ** Technical Review **
# This module validates .training configuration and builds the Vertex AI CustomJobSpec
# that gets submitted to W&B Launch. Critical flow:
#
# 1. validate_launch_config() checks GPU type, machine type, disk size, compatibility
# 2. get_launch_spec_config() builds workerPoolSpec with machine_spec, disk_spec, scheduling
# 3. format_validation_report() creates user-friendly error messages with fix guidance
#
# CRITICAL VERTEX AI API STRUCTURE (scheduling.strategy):
# The scheduling field MUST use "strategy" with enum values "SPOT" or "STANDARD".
# Common mistake: using scheduling.preemptible (NOT a valid field for workerPoolSpec).
#
# Correct API structure:
#   spec = {
#       "machine_spec": {...},
#       "disk_spec": {...},
#       "scheduling": {
#           "strategy": "SPOT"      # For spot/preemptible instances (60-91% savings)
#           # OR
#           "strategy": "STANDARD"  # For on-demand instances
#       }
#   }
#
# Research Links:
# - Vertex AI CustomJobSpec: https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/CustomJobSpec
# - Scheduling.Strategy enum: https://cloud.google.com/vertex-ai/docs/Training/configure-compute
# - Spot VMs documentation: https://cloud.google.com/vertex-ai/docs/Training/use-spot-vms
#
# Historical Bug (2025-11-16):
# Initial implementation used scheduling.preemptible=True which caused API errors:
#   "Unknown field for WorkerPoolSpec: scheduling"
# Fix: Changed to scheduling.strategy="SPOT" per Vertex AI API spec.
# Commit: 11968d0 - CRITICAL FIX: Use scheduling.strategy (SPOT/STANDARD)
# </claudes_code_comments>

from typing import Dict, Tuple, List


def validate_launch_config(config: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    GO/NO-GO validation before job submission.

    Checks:
    1. GPU configuration is present and valid
    2. Machine type matches GPU requirements
    3. Boot disk size is sufficient (200GB minimum)
    4. Required configuration keys are present

    Args:
        config: Training configuration from .training file

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if all checks pass, False otherwise
        - error_messages: List of validation errors (empty if valid)
    """
    errors = []

    # Check 1: GPU configuration required
    gpu_type = config.get("TRAINING_GPU")
    if not gpu_type:
        errors.append("❌ TRAINING_GPU not set in .training")
        errors.append("   Required GPU types: NVIDIA_TESLA_T4, NVIDIA_TESLA_A100, NVIDIA_H200, NVIDIA_L4")

    gpu_count = config.get("TRAINING_GPU_NUMBER")
    if not gpu_count:
        errors.append("❌ TRAINING_GPU_NUMBER not set in .training")
        errors.append("   Set to number of GPUs (usually 1 or 2)")
    else:
        # Validate it's a valid integer >= 1
        try:
            count = int(gpu_count)
            if count < 1:
                errors.append(f"❌ TRAINING_GPU_NUMBER must be >= 1, got: {count}")
                errors.append("   Set to number of GPUs (usually 1 or 2)")
        except ValueError:
            errors.append(f"❌ TRAINING_GPU_NUMBER must be a number, got: {gpu_count}")
            errors.append("   Examples: 1, 2, 4, 8")

    # Check 2: Machine type (computed on-the-fly from GPU)
    # No stored variable - get_best_machine_for_gpu() called where needed

    # Check 3: Boot disk size recommendation
    # We'll default to 200GB in code, but warn if explicitly set too low
    boot_disk_gb = config.get("WANDB_LAUNCH_BOOT_DISK_GB")
    if boot_disk_gb:
        try:
            disk_size = int(boot_disk_gb)
            if disk_size < 200:
                errors.append(f"⚠️  WANDB_LAUNCH_BOOT_DISK_GB={disk_size}GB is too small")
                errors.append("   Recommended: 200GB minimum for VQAv2 dataset + checkpoints")
        except ValueError:
            errors.append(f"❌ WANDB_LAUNCH_BOOT_DISK_GB must be a number, got: {boot_disk_gb}")

    # Check 4: Machine type validation removed - always auto-computed correctly from GPU!

    is_valid = len(errors) == 0
    return is_valid, errors


def get_launch_spec_config(config: Dict[str, str]) -> Dict:
    """
    Build Vertex AI launch spec from .training configuration.

    This replaces the hardcoded n1-standard-4 with actual GPU config from .training.

    Args:
        config: Training configuration from .training file

    Returns:
        Dict with worker_pool_specs configuration for Vertex AI
    """
    # Get GPU configuration from .training
    gpu_type = config.get("TRAINING_GPU")
    gpu_count = config.get("TRAINING_GPU_NUMBER", "1")
    boot_disk_gb = config.get("WANDB_LAUNCH_BOOT_DISK_GB", "200")  # Default 200GB

    # Compute machine type from GPU on-the-fly (ONLY place this happens!)
    from ..shared.machine_selection import get_best_machine_for_gpu
    machine_type = get_best_machine_for_gpu(gpu_type) if gpu_type else "n1-standard-4"

    # Get preemptible setting (CRITICAL - must match quota check!)
    use_preemptible_str = config.get("TRAINING_GPU_IS_PREEMPTIBLE", "false").lower()
    use_preemptible = use_preemptible_str == "true"

    # Build machine spec
    machine_spec = {
        "machine_type": machine_type
    }

    # Add GPU if configured
    if gpu_type:
        machine_spec["accelerator_type"] = gpu_type
        machine_spec["accelerator_count"] = int(gpu_count)

    # Build disk spec (CRITICAL - fixes "low on disk" errors!)
    disk_spec = {
        "boot_disk_type": "pd-ssd",  # SSD for faster image pull
        "boot_disk_size_gb": int(boot_disk_gb)
    }

    # Build worker pool spec (machine + disk only)
    # NOTE: scheduling goes at CustomJobSpec level, not workerPoolSpec level!
    spec = {
        "machine_spec": machine_spec,
        "disk_spec": disk_spec,
        "replica_count": 1
    }

    return spec


def format_validation_report(errors: List[str]) -> str:
    """
    Format validation errors into readable report for user.

    Args:
        errors: List of validation error messages

    Returns:
        Formatted multi-line error report
    """
    if not errors:
        return ""

    report_lines = [
        "",
        "╔" + "═" * 98 + "╗",
        "║" + " " * 30 + "🚨 LAUNCH VALIDATION FAILED 🚨" + " " * 38 + "║",
        "╠" + "═" * 98 + "╣",
        "║" + " " * 98 + "║",
        "║  Your .training configuration doesn't match what will be sent to Vertex AI!" + " " * 22 + "║",
        "║" + " " * 98 + "║",
    ]

    # Add each error with proper padding
    for error in errors:
        # Wrap long lines
        if len(error) > 94:  # 98 - 4 for padding
            # Split into multiple lines
            words = error.split()
            line = "║  "
            for word in words:
                if len(line) + len(word) + 1 > 96:  # Start new line
                    line += " " * (98 - len(line)) + "║"
                    report_lines.append(line)
                    line = "║     " + word  # Indent continuation
                else:
                    line += word + " "
            line += " " * (98 - len(line)) + "║"
            report_lines.append(line)
        else:
            padded = error + " " * (94 - len(error))
            report_lines.append(f"║  {padded}  ║")

    report_lines.extend([
        "║" + " " * 98 + "║",
        "║  💡 Fix your .training file and try again." + " " * 52 + "║",
        "║" + " " * 98 + "║",
        "╚" + "═" * 98 + "╝",
        ""
    ])

    return "\n".join(report_lines)
