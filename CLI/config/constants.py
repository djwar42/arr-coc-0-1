"""
ARR-COC CLI Constants and Shared Values
"""

# <claudes_code_comments>
# ** Function List **
# load_training_config() - Load configuration from ARR_COC/Training/.training file with GPU machine auto-selection
#
# ** Technical Review **
# Provides shared constants and config loading for ARR-COC CLI. Contains ASCII art headers with Rich markup
# gradients (blue→cyan) and configuration file parsing with intelligent machine type selection.
#
# Constants:
# - ARR_COC_TITLE: Gradient text using Rich markup (#4a9eff → #00d4ff)
# - ARR_COC_WITH_WINGS: Title with ◇ decorations
# - ARR_COC_SUBTITLE/DESCRIPTION: Taglines for UI display
#
# Config flow:
# 1. load_training_config() reads ARR_COC/Training/.training file
# 2. Parses KEY=VALUE lines, strips comments/quotes
# 3. Auto-selects optimal machine type based on GPU via get_best_machine_for_gpu()
# 4. Returns Dict[str, str] with WANDB_ENTITY, WANDB_PROJECT, TRAINING_GPU, auto-selected machine type
#
# Auto-selection rules (from machine_selection.py):
# - L4 → g2-standard-4 (pre-attached to G2)
# - A100 → a2-highgpu-1g (pre-attached to A2)
# - H100 → a3-highgpu-1g (pre-attached to A3)
# - H200 → a3-ultragpu-8g (pre-attached to A3-Ultra)
# - T4/V100/P4/P100 → n1-standard-4 (recommended 4+ vCPUs)
#
# Path resolution: CLI/constants.py → up 1 parent (CLI/ → repo root) →
# finds ARR_COC/Training/.training file
# </claudes_code_comments>

import os
import sys
from pathlib import Path
from typing import Dict
from rich import print as rprint

# ============================================================================
# ASCII Art Header with Gradient & Compact Wings
# ============================================================================

# Create gradient title (blue to cyan, like Gradio) using Rich markup
# Each letter gets a slightly different shade from blue (#4a9eff) to cyan (#00d4ff)
ARR_COC_TITLE = "[#4a9eff]A[/][#3da7ff]R[/][#30afff]R[/][#23b7ff]-[/][#16bfff]C[/][#09c7ff]O[/][#00d4ff]C[/]"

# Compact wing decorations (subtle, minimalist)
ARR_COC_WITH_WINGS = f"[dim]◇[/dim]  {ARR_COC_TITLE}  [dim]◇[/dim]"

ARR_COC_SUBTITLE = "'What you see changes what you see'"
ARR_COC_DESCRIPTION = "Adaptive Relevance Realization [#4a9eff]•[/] Contexts Optical Compression"


# ============================================================================
# Project Paths
# ============================================================================

# Project root (arr-coc-0-1/)
# constants.py location: CLI/config/constants.py
# Navigate up: CLI/config/ → CLI/ → arr-coc-0-1/ (repo root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Logs directory
LOGS_DIR = PROJECT_ROOT / "ARR_COC" / "Training" / "logs"


# ============================================================================
# Configuration Loader
# ============================================================================

def load_training_config() -> Dict[str, str]:
    """
    Load configuration from ARR_COC/Training/.training file with GPU machine auto-selection.

    Reads ARR_COC/Training/.training file and automatically selects optimal machine type based on GPU.
    Always uses get_best_machine_for_gpu() to select the best machine for the specified GPU.

    Returns:
        Dict[str, str]: Configuration with all keys from .training plus auto-selected machine type

    Example:
        .training has: TRAINING_GPU="NVIDIA_TESLA_T4"
        Auto-selects and adds: machine_type="n1-standard-4"
    """
    config_path = PROJECT_ROOT / "ARR_COC" / "Training" / ".training"

    if not config_path.exists():
        rprint("[red]Error: ARR_COC/Training/.training file not found![/red]")
        rprint(f"Expected at: {config_path}")
        sys.exit(1)

    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove inline comments
                if "#" in value:
                    value = value.split("#")[0]
                # Remove quotes and whitespace
                value = value.strip().strip('"').strip("'")
                config[key.strip()] = value

    # Machine type is computed on-the-fly from GPU when needed
    # No longer stored in config - use get_best_machine_for_gpu() directly

    return config


# ============================================================================
# W&B API Helper
# ============================================================================


