"""
GPU + C3 Machine Selection

Shared logic for selecting machines:
- C3 machines for Cloud Build (quota-aware)
- GPU machines for Vertex AI training (compatibility-aware)
"""

# <claudes_code_comments>
# ** Function List **
# get_best_c3(project_id, region) - Select best C3 machine within Cloud Build quota limits
# get_c3_chonk_label(vcpus) - Get fun CHONK label and power meter for C3 machine size
# get_best_machine_for_gpu(gpu_type) - Auto-select cheapest compatible machine type for GPU
# get_gpu_chonk_label(gpu_type) - Get personality label for GPU type
# get_gpu_machine_family(machine_type) - Extract machine family from GPU machine type
# validate_gpu_machine_compatibility(machine_type, gpu_type) - Validate GPU+machine combination compatibility
#
# ** Technical Review **
# Machine selection logic for two use cases:
#
# 1. Cloud Build C3 (PyTorch compilation):
#    - get_best_c3() queries quota.get_cloud_build_c3_region_quota()
#    - Selects largest C3 machine within quota (176/88/44 vCPUs)
#    - get_c3_chonk_label() provides fun labels (ABSOLUTE UNIT, Big Chungus, etc.)
#
# 2. Vertex AI GPU Training:
#    - get_best_machine_for_gpu() auto-selects machine based on GPU type
#    - Handles GCP pre-attached GPU rules (G2→L4, A2→A100, A3→H100/H200)
#    - validate_gpu_machine_compatibility() validates user overrides
#    - get_gpu_chonk_label() provides personality labels per GPU
#
# GPU Compatibility Rules:
# - L4 GPU → MUST use G2 machines (pre-attached)
# - A100 GPU → MUST use A2 machines (pre-attached)
# - H100/H200 GPU → MUST use A3 machines (pre-attached)
# - T4/V100/P4/P100 → Use N1 machines (4+ vCPUs recommended)
#
# Flow examples:
# - C3: get_cloud_build_c3_region_quota() → get_best_c3() → (machine, vcpus, quota)
# - GPU: user picks T4 → get_best_machine_for_gpu("NVIDIA_TESLA_T4") → "n1-standard-4"
# - GPU validation: validate_gpu_machine_compatibility("n1-standard-4", "NVIDIA_L4") → (False, "L4 requires G2")
# </claudes_code_comments>

from typing import Tuple


# ============================================================================
# C3 Machine Selection (Cloud Build)
# ============================================================================

def get_best_c3(project_id: str, region: str) -> Tuple[str, int, int]:
    """
    Get best C3 machine based on Cloud Build quota.

    Uses centralized quota module to detect Cloud Build C3 quota,
    then selects largest machine within quota limits.

    Args:
        project_id: GCP project ID
        region: GCP region

    Returns:
        Tuple of (best_machine, best_vcpus, cb_quota)

    Example:
        >>> get_best_c3("my-project", "us-central1")
        ("c3-standard-176", 176, 176)
    """
    from CLI.shared.quota import get_cloud_build_c3_region_quota

    cb_quota = get_cloud_build_c3_region_quota(project_id, region)

    if cb_quota >= 176:
        return "c3-standard-176", 176, cb_quota
    elif cb_quota >= 88:
        return "c3-standard-88", 88, cb_quota
    elif cb_quota >= 44:
        return "c3-standard-44", 44, cb_quota
    else:
        return "c3-standard-44", 44, cb_quota


def get_c3_chonk_label(vcpus: int) -> Tuple[str, str]:
    """
    Get CHONK label and power meter for C3 vCPU count.

    Returns fun labels for different C3 machine sizes with visual power meters.

    Args:
        vcpus: Number of vCPUs (4, 8, 22, 44, 88, 176)

    Returns:
        Tuple of (chonk_label, power_meter)

    Examples:
        >>> get_c3_chonk_label(176)
        ("[bold magenta]ABSOLUTE UNIT[/bold magenta]", "▂▃▄▅▆▇█")

        >>> get_c3_chonk_label(88)
        ("[bold cyan]Big Chungus[/bold cyan]", "▂▃▄▅▆")

        >>> get_c3_chonk_label(44)
        ("[bold yellow]Decent Chonk[/bold yellow]", "▂▃▄")

        >>> get_c3_chonk_label(4)
        ("[bold yellow]Smol Boi[/bold yellow]", "▂")
    """
    if vcpus >= 176:
        return "[bold magenta]ABSOLUTE UNIT[/bold magenta]", "▂▃▄▅▆▇█"
    elif vcpus >= 88:
        return "[bold cyan]Big Chungus[/bold cyan]", "▂▃▄▅▆"
    elif vcpus >= 44:
        return "[bold yellow]Decent Chonk[/bold yellow]", "▂▃▄"
    else:
        return "[bold yellow]Smol Boi[/bold yellow]", "▂"


# ============================================================================
# GPU Machine Selection (Vertex AI Training)
# ============================================================================

def get_best_machine_for_gpu(gpu_type: str) -> str:
    """
    Auto-select cheapest compatible machine type for GPU.

    Handles GCP pre-attached GPU rules where certain GPUs only work
    with specific machine families (G2, A2, A3).

    Args:
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4", "NVIDIA_L4")

    Returns:
        Machine type string (e.g., "n1-standard-4", "g2-standard-4")

    Rules:
        - L4 → g2-standard-4 (pre-attached, cheapest G2)
        - A100-80GB → a2-ultragpu-1g (pre-attached)
        - A100-40GB → a2-highgpu-1g (pre-attached, cheapest)
        - H100 → a3-highgpu-1g (pre-attached, cheapest A3)
        - H200 → a3-ultragpu-8g (pre-attached, only option)
        - T4/V100/P4/P100 → n1-standard-4 (recommended 4+ vCPUs)

    Examples:
        >>> get_best_machine_for_gpu("NVIDIA_TESLA_T4")
        "n1-standard-4"

        >>> get_best_machine_for_gpu("NVIDIA_L4")
        "g2-standard-4"

        >>> get_best_machine_for_gpu("NVIDIA_TESLA_A100")
        "a2-highgpu-1g"
    """
    # L4 GPU → MUST use G2 (pre-attached)
    if "L4" in gpu_type:
        return "g2-standard-4"  # Cheapest G2 with L4 built-in

    # A100 GPU → MUST use A2 (pre-attached)
    if "A100" in gpu_type:
        if "80GB" in gpu_type or "80G" in gpu_type:
            return "a2-ultragpu-1g"  # A100-80GB
        return "a2-highgpu-1g"  # A100-40GB (cheapest)

    # H100 GPU → MUST use A3 (pre-attached)
    if "H100" in gpu_type:
        return "a3-highgpu-1g"  # Cheapest A3 with H100

    # H200 GPU → MUST use A3-Ultra (pre-attached)
    if "H200" in gpu_type:
        return "a3-ultragpu-8g"  # Only option for H200

    # T4/V100/P4/P100 → Use N1 (recommended 4+ vCPUs)
    if any(gpu in gpu_type for gpu in ["T4", "V100", "P4", "P100"]):
        return "n1-standard-4"  # Cheapest recommended N1

    # Fallback for unknown GPU types
    return "n1-standard-4"


def get_gpu_chonk_label(gpu_type: str) -> Tuple[str, str]:
    """
    Get personality label for GPU type.

    Returns fun labels for different GPU types with visual flair.

    Args:
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4", "NVIDIA_L4")

    Returns:
        Tuple of (personality_label, visual_meter)

    Examples:
        >>> get_gpu_chonk_label("NVIDIA_H200")
        ("[bold magenta]ABSOLUTE BEAST[/bold magenta]", "🔥🔥🔥")

        >>> get_gpu_chonk_label("NVIDIA_H100_80GB")
        ("[bold cyan]Powerhouse[/bold cyan]", "⚡⚡")

        >>> get_gpu_chonk_label("NVIDIA_TESLA_A100")
        ("[bold yellow]Workhorse[/bold yellow]", "💪")

        >>> get_gpu_chonk_label("NVIDIA_L4")
        ("[bold green]Balanced[/bold green]", "✨")

        >>> get_gpu_chonk_label("NVIDIA_TESLA_T4")
        ("[bold blue]Reliable[/bold blue]", "⭐")
    """
    # H200 - Most powerful
    if "H200" in gpu_type:
        return "[bold magenta]ABSOLUTE BEAST[/bold magenta]", "🔥🔥🔥"

    # H100 - High-end
    if "H100" in gpu_type:
        return "[bold cyan]Powerhouse[/bold cyan]", "⚡⚡"

    # A100 - Workhorse
    if "A100" in gpu_type:
        if "80GB" in gpu_type or "80G" in gpu_type:
            return "[bold yellow]Mega Workhorse[/bold yellow]", "💪💪"
        return "[bold yellow]Workhorse[/bold yellow]", "💪"

    # L4 - Balanced modern GPU
    if "L4" in gpu_type:
        return "[bold green]Balanced[/bold green]", "✨"

    # V100 - Previous gen high-end
    if "V100" in gpu_type:
        return "[bold cyan]Classic Power[/bold cyan]", "⚡"

    # T4 - Reliable workhorse
    if "T4" in gpu_type:
        return "[bold blue]Reliable[/bold blue]", "⭐"

    # P100 - Older gen
    if "P100" in gpu_type:
        return "[bold yellow]Veteran[/bold yellow]", "🛡️"

    # P4 - Entry level
    if "P4" in gpu_type:
        return "[bold green]Budget Friend[/bold green]", "💚"

    # Unknown
    return "[bold white]Unknown GPU[/bold white]", "❓"


def get_gpu_machine_family(machine_type: str) -> str:
    """
    Extract machine family from GPU machine type.

    Args:
        machine_type: Full machine type (e.g., "n1-standard-4")

    Returns:
        Machine family string (e.g., "n1", "g2", "a2", "a3")

    Examples:
        >>> get_gpu_machine_family("n1-standard-4")
        "n1"

        >>> get_gpu_machine_family("g2-standard-8")
        "g2"

        >>> get_gpu_machine_family("a2-highgpu-1g")
        "a2"

        >>> get_gpu_machine_family("a3-ultragpu-8g")
        "a3"
    """
    return machine_type.split("-")[0]


def validate_gpu_machine_compatibility(machine_type: str, gpu_type: str) -> Tuple[bool, str]:
    """
    Validate that GPU+machine combination is compatible.

    Validates GCP pre-attached GPU rules where certain GPU types
    only work with specific machine families.

    Args:
        machine_type: Machine type (e.g., "n1-standard-4")
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4")

    Returns:
        Tuple of (is_compatible, error_message)
        - is_compatible: True if valid combo, False otherwise
        - error_message: Empty if valid, helpful error if invalid

    Examples:
        >>> validate_gpu_machine_compatibility("n1-standard-4", "NVIDIA_TESLA_T4")
        (True, "")

        >>> validate_gpu_machine_compatibility("n1-standard-4", "NVIDIA_L4")
        (False, "L4 GPU requires G2 machines (g2-standard-4 or higher)")

        >>> validate_gpu_machine_compatibility("g2-standard-4", "NVIDIA_L4")
        (True, "")

        >>> validate_gpu_machine_compatibility("g2-standard-4", "NVIDIA_TESLA_T4")
        (False, "G2 machines have L4 built-in (cannot use NVIDIA_TESLA_T4)")
    """
    family = get_gpu_machine_family(machine_type)

    # G2 machines have L4 GPU pre-attached
    if family == "g2":
        if "L4" in gpu_type:
            return True, ""
        return False, f"G2 machines have L4 built-in (cannot use {gpu_type})"

    # A2 machines have A100 GPU pre-attached
    if family == "a2":
        if "A100" in gpu_type:
            return True, ""
        return False, f"A2 machines have A100 built-in (cannot use {gpu_type})"

    # A3 machines have H100/H200 GPU pre-attached
    if family == "a3":
        if "H100" in gpu_type or "H200" in gpu_type:
            return True, ""
        return False, f"A3 machines have H100/H200 built-in (cannot use {gpu_type})"

    # L4 only works on G2
    if "L4" in gpu_type:
        return False, "L4 GPU requires G2 machines (g2-standard-4 or higher)"

    # A100 only works on A2
    if "A100" in gpu_type:
        return False, "A100 GPU requires A2 machines (a2-highgpu-1g or higher)"

    # H100/H200 only work on A3
    if "H100" in gpu_type or "H200" in gpu_type:
        return False, f"{gpu_type} requires A3 machines (a3-highgpu-1g or higher)"

    # T4/V100/P4/P100 work on N1
    if any(gpu in gpu_type for gpu in ["T4", "V100", "P4", "P100"]):
        if family == "n1":
            return True, ""
        return False, f"{gpu_type} works with N1 machines (n1-standard-4 or higher)"

    # Unknown GPU type
    return False, f"Unknown GPU type: {gpu_type}"
