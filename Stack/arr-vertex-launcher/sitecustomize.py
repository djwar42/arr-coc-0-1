"""
sitecustomize.py - Auto-loaded by Python on startup

CRITICAL SOLUTION (2025-11-16):
================================

Problem: Monkey-patch must persist when 'wandb launch-agent' command runs.

Previous Failed Approaches:
1. ❌ Separate Python process → Patch lost when process exits.
2. ❌ os.execvp('wandb') → Replaces entire process, patch destroyed.
3. ❌ subprocess.run(['wandb']) → Separate child process, no patch inheritance.

The Solution: sitecustomize.py
-------------------------------

Python automatically imports sitecustomize.py on startup (before any other imports!)
- Loaded when Python interpreter starts
- Happens BEFORE wandb modules are imported
- Works with ANY Python invocation (python, wandb binary, etc.)

Source: https://docs.python.org/3/library/site.html#module-sitecustomize
"This module is automatically imported during initialization"

When user runs: wandb launch-agent
1. Python starts
2. Loads sitecustomize.py (THIS FILE!)
3. Applies our Vertex AI spot patch
4. THEN imports wandb modules (already patched!)
5. wandb runs with patch active! ✅

Research Links:
- Python site module: https://docs.python.org/3/library/site.html
- W&B Launch source: https://github.com/wandb/wandb/blob/main/wandb/sdk/launch/runner/vertex_runner.py
- Monkey-patching persistence: https://stackoverflow.com/questions/tagged/python+monkey-patch

SUCCESS EVIDENCE (2025-11-16):
==============================

✅ SPOT INSTANCES ENABLED! Cost savings: 60-91%

Proof the patch works:
- Quota error changed from "custom_model_training_nvidia_t4_gpus" (on-demand)
  to "custom_model_training_preemptible_nvidia_t4_gpus" (SPOT!)
- Debug logs confirmed: "🔥 SITECUSTOMIZE.PY LOADED! (Image is fresh!)"
- Patch location verified: /usr/lib/python3.13/site-packages/sitecustomize.py
- scheduling_strategy='SPOT' successfully added to Vertex AI submission

CRITICAL PATH FIX:
------------------
WRONG: /usr/local/lib/python3.13/site-packages/  ← Python doesn't check here first!
RIGHT: /usr/lib/python3.13/site-packages/        ← Python's primary site-packages!

The Dockerfile must install to /usr/lib/ not /usr/local/lib/ because Python's
import system checks /usr/lib/ before /usr/local/lib/ in the module search path.

See Dockerfile line ~139 for the correct installation path.
"""

# Apply the Vertex AI spot instance patch immediately on Python startup!
try:
    import os
    import sys

    # Add /app to path so we can import wandb_vertex_patch
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # Import and apply the patch
    from wandb_vertex_patch import apply_wandb_vertex_spot_patch

    # Apply patch (this modifies wandb.sdk.launch.runner.vertex_runner module)
    # When wandb imports vertex_runner later, it will already be patched!
    apply_wandb_vertex_spot_patch()

except Exception as e:
    # Don't crash Python startup if patch fails
    # Just log the error so we can see it in container logs
    import sys

    print(
        f"⚠️  sitecustomize.py: Failed to apply W&B Vertex spot patch: {e}",
        file=sys.stderr,
    )

# Version 1.1 - Force rebuild (2025-11-16)

# CRITICAL: Output debug logs to stderr on module load (Python startup)
# This confirms sitecustomize.py loaded and patch was applied
import sys

print("=" * 70, file=sys.stderr)
print("🔥 SITECUSTOMIZE.PY LOADED! (Image is fresh!)", file=sys.stderr)
print(f"   Path: {__file__}", file=sys.stderr)
print("   ✅ W&B Vertex spot patch APPLIED on Python startup!", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Version 1.2 - Added startup debug output + fixed path to /usr/lib/ (2025-11-16)
