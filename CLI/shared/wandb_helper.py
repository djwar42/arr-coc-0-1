# <claudes_code_comments>
# ** Function List **
# generate_cool_run_name() - Generate W&B-style adjective-noun run names
# WandBHelper.__init__(entity, project, queue) - Initialize with W&B credentials
# WandBHelper.get_active_runs() - Fetch active runs + Launch queue items (two sources combined)
# WandBHelper.submit_job(config) - Submit training job to W&B Launch queue
# WandBHelper.cancel_run(run_id) - Cancel running job or delete queued job (handles both types)
#
# ** Technical Review **
# W&B API wrapper for training job management. Handles run monitoring, job submission via wandb launch CLI,
# and run cancellation. Used by MonitorScreen and LaunchScreen.
#
# Flow: Initialize with entity/project/queue → get_active_runs() for monitoring → submit_job(config) for
# launching → cancel_run(id) for stopping
#
# get_active_runs():
# - Combines TWO sources: (1) Active runs from api.runs() + (2) Queue items from api.run_queue()
# - Source 1: Runs with states: running, pending, preempting (jobs already picked up by agent)
# - Source 2: Queue items with states: pending, leased (jobs waiting for or being processed by agent)
# - Returns combined list of dicts with: id, name, state, created_at, runtime, url
# - Runtime calculated as seconds since creation (for running jobs only, 0 for queue items)
# - Queue items show as "{project} (queue)" to distinguish from actual runs
# - Graceful error handling: returns partial results if either source fails
#
# submit_job(config, docker_image):
# - Builds wandb launch command with config from .training file
# - Creates temp JSON file with environment variables (overrides.run_config) and docker_image in spec
# - Executes: wandb launch --uri . --queue QUEUE --config config.json
# - Parses output to extract run ID (3 formats supported)
# - Returns (run_id, full_output) tuple
#
# Run ID extraction (3 parsing strategies):
# 1. URL format: "https://wandb.ai/entity/project/runs/RUN_ID" (when agent picks up job)
# 2. Queue format: "Added run to queue QUEUE_NAME" (when no agent running)
# 3. Text format: "Queued run RUN_ID" (legacy format)
#
# Environment variables passed to job:
# - BASE_MODEL, NUM_VISUAL_TOKENS, LEARNING_RATE, BATCH_SIZE
# - GRADIENT_ACCUMULATION_STEPS, NUM_EPOCHS, SAVE_EVERY_N_STEPS, SEED
# - WANDB_PROJECT, HF_HUB_REPO_ID, DATASET_NAME, MAX_TRAIN_SAMPLES
#
# cancel_run(run_id):
# - Handles BOTH queue items (pending/leased) and actual runs (running)
# - First tries queue item deletion: Searches queue for matching ID → calls item.delete()
# - Falls back to run cancellation: Fetches run via api.run() → calls run.stop()
# - Queue items have base64 IDs (e.g., "UnVuUXVl..."), runs have alphanumeric IDs
# - Used by MonitorScreen for cancelling/deleting selected jobs
#
# Error handling:
# - get_active_runs(): Returns empty list on ValueError (no project) or TypeError (API None)
# - submit_job(): Raises Exception on nonzero returncode with stderr message
# - Temp config file cleanup in finally block (even on error)
# </claudes_code_comments>

import os
import sys
import subprocess
import threading
import random
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import wandb


def generate_cool_run_name() -> str:
    """
    Generate W&B-style adjective-noun run names

    Returns names like: "spazzy-forest-1762670954", "ethereal-snowflake-1762664669"

    Format: {adjective}-{noun}
    """
    adjectives = [
        "spazzy", "chonky", "bonkers", "zesty", "wonky", "goofy", "quirky", "janky",
        "thicc", "smol", "yeet", "zoomy", "derpy", "snazzy", "funky", "groovy",
        "joyous", "bright", "dazzling", "shiny", "zimmering", "sparkling", "gleaming", "beaming",
        "jubilant", "cheerful", "merry", "zippy", "peppy", "bouncy", "bubbly", "fizzy",
        "blessed", "lucky", "happy", "sunny", "cozy", "warm", "smooth", "sweet",
        "fresh", "crisp", "cool", "neat", "keen", "ace", "stellar", "epic",
        "ethereal", "deep", "major", "cosmic", "lunar", "solar", "quantum", "neural",
        "atomic", "vivid", "crimson", "azure", "golden", "silver", "twilight", "misty",
        "frosty", "blazing", "gentle", "fierce", "swift", "noble", "ancient", "modern",
        "wild", "calm", "radiant", "silent", "wandering", "steady", "flying", "roaming",
        "dancing", "singing", "glowing", "shimmering", "twinkling", "flickering", "proud", "bold"
    ]

    nouns = [
        "forest", "snowflake", "sky", "surf", "mountain", "ocean", "river", "meadow",
        "valley", "peak", "canyon", "desert", "glacier", "volcano", "aurora", "comet",
        "nebula", "galaxy", "star", "moon", "sun", "planet", "asteroid", "meteor",
        "thunder", "lightning", "breeze", "storm", "rain", "snow", "mist", "fog",
        "wave", "tide", "current", "wind", "cloud", "horizon", "dawn", "dusk",
        "blossom", "willow", "birch", "maple", "cedar", "pine", "oak", "fern",
        "orchid", "lotus", "lily", "rose", "daisy", "tulip", "iris", "violet",
        "butterfly", "dragonfly", "firefly", "phoenix", "dragon", "unicorn", "pegasus", "griffin",
        "castle", "tower", "fortress", "palace", "temple", "shrine", "arch", "bridge",
        "crystal", "diamond", "sapphire", "ruby", "emerald", "pearl", "opal", "jade",
        "fountain", "garden", "oasis", "grove", "glade", "sanctuary", "haven", "refuge"
    ]

    adjective = random.choice(adjectives)
    noun = random.choice(nouns)

    return f"{adjective}-{noun}"

class WandBHelper:
    """Helper for W&B Launch API interactions"""

    def __init__(self, entity: str, project: str, queue: str):
        self.entity = entity
        self.project = project
        self.queue = queue
        self._api = None  # Lazy-create API on first use (after event loop exists)

    @property
    def api(self):
        """Lazy-create wandb.Api on first access (ensures event loop exists)"""
        if self._api is None:
            self._api = wandb.Api(timeout=15)
        return self._api

    def get_active_runs(self) -> List[Dict]:
        """Get active training runs + queued jobs from Launch queue (with 10s timeout)"""
        # Wrap the actual API calls with a timeout to prevent hanging
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._get_active_runs_impl)
            try:
                return future.result(timeout=10)  # 10 second timeout
            except FuturesTimeoutError:
                # Silent failure - return empty list if timeout
                # (Monitor screen will show "No active runs" instead of hanging)
                return []
            except Exception as e:
                # Silent failure - return empty list on API errors
                # (Prevents crashing if W&B API is down)
                return []

    def _get_active_runs_impl(self) -> List[Dict]:
        """Internal implementation of get_active_runs (without timeout wrapper)"""
        run_list = []

        # 1. Get actual runs (jobs picked up by agent and running)
        try:
            runs = self.api.runs(
                f"{self.entity}/{self.project}",
                filters={"state": {"$in": ["running", "pending", "preempting"]}}
            )
            for run in runs:
                run_list.append({
                    "id": run.id,  # Keep full ID for unique keys
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "runtime": (datetime.now() - datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))).seconds if run.state == "running" else 0,
                    "url": run.url,
                })
        except (ValueError, TypeError):
            pass  # Project doesn't exist yet or API returned None
        except Exception:
            pass  # Catch any other errors

        # 2. Get queue items (jobs waiting for or being processed by agent)
        try:
            queue = self.api.run_queue(self.entity, self.queue)
            if queue and hasattr(queue, 'items'):
                for item in queue.items:
                    # Only show pending/leased items from THIS project (not other projects using same queue)
                    if item.state in ['pending', 'leased'] and item.project == self.project:
                        run_list.append({
                            "id": item.id,  # Keep full ID for unique keys
                            "name": f"{item.project} (queue)",
                            "state": item.state,
                            "created_at": "Unknown",
                            "runtime": 0,
                            "url": f"https://wandb.ai/{item.entity}/{item.project}",
                        })
        except Exception:
            pass  # Queue doesn't exist or API error

        return run_list

    def get_completed_runs(self, limit: int = 20) -> List[Dict]:
        """Get recently completed runs (finished, failed, crashed, killed)"""
        try:
            runs = self.api.runs(
                f"{self.entity}/{self.project}",
                filters={"state": {"$in": ["finished", "failed", "crashed", "killed"]}},
                order="-created_at"  # Most recent first
            )

            run_list = []
            for run in list(runs)[:limit]:  # Limit results
                # Calculate total runtime
                if run.summary.get("_runtime"):
                    runtime = run.summary["_runtime"]
                else:
                    runtime = 0

                run_list.append({
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "runtime": runtime,
                    "url": run.url,
                })

            return run_list
        except (ValueError, TypeError):
            return []  # Project doesn't exist
        except Exception:
            return []  # Any other error

    def submit_job(self, config: Dict[str, str], docker_image: str = None) -> tuple[str, str, str]:
        """
        Submit training job to W&B Launch queue. Returns (run_id, full_output, run_name)

        Args:
            config: Training configuration dict
            docker_image: Optional pre-built Docker image URI. If provided, adds to launch spec.
                         If None, uses --dockerfile (requires Docker in runner).

        Returns:
            tuple: (run_id, full_output, run_name)
                - run_id: Actual W&B run ID or generated run_name as fallback
                - full_output: Full W&B Launch output
                - run_name: Cool generated name (e.g., "ethereal-snowflake-1762661725")
        """
        import json
        import tempfile

        # Get repo root (up to project root)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

        # W&B Launch will package and upload the local directory
        # Use "." as URI + cwd=repo_root to avoid artifact name length issues
        # (absolute paths create artifact names >128 chars)

        # Build environment variables from .training
        env_vars = {
            "BASE_MODEL": config.get("BASE_MODEL"),
            "NUM_VISUAL_TOKENS": config.get("NUM_VISUAL_TOKENS"),
            "LEARNING_RATE": config.get("LEARNING_RATE"),
            "BATCH_SIZE": config.get("BATCH_SIZE"),
            "GRADIENT_ACCUMULATION_STEPS": config.get("GRADIENT_ACCUMULATION_STEPS"),
            "NUM_EPOCHS": config.get("NUM_EPOCHS"),
            "SAVE_EVERY_N_STEPS": config.get("SAVE_EVERY_N_STEPS"),
            "SEED": config.get("SEED"),
            "WANDB_PROJECT": config.get("WANDB_PROJECT"),
            "HF_HUB_REPO_ID": config.get("HF_HUB_REPO_ID"),
            "DATASET_NAME": config.get("DATASET_NAME"),
            "MAX_TRAIN_SAMPLES": config.get("MAX_TRAIN_SAMPLES"),
        }

        # Filter out None/empty values
        env_vars = {k: v for k, v in env_vars.items() if v}

        # Import validation to get proper GPU + disk config from .training
        from CLI.launch.validation import get_launch_spec_config

        # Get validated launch spec (GPU, machine type, boot disk from .training!)
        # This replaces the old hardcoded n1-standard-4 with no GPU
        worker_pool_spec = get_launch_spec_config(config)

        # Add container spec to worker pool
        worker_pool_spec["container_spec"] = {
            # W&B Launch will use this pre-built image (no building needed!)
            "image_uri": docker_image if docker_image else "python:3.10"
        }

        # Get preemptible setting to add scheduling at CustomJobSpec level
        use_preemptible_str = config.get("TRAINING_GPU_IS_PREEMPTIBLE", "false").lower()
        use_preemptible = use_preemptible_str == "true"

        # Create launch config JSON with environment variables and GCP region
        # Structure follows W&B Launch Vertex AI docs: https://docs.wandb.ai/platform/launch/setup-vertex
        #
        # CRITICAL DISCOVERY: W&B Launch DOES NOT SUPPORT scheduling_strategy! (2025-11-16)
        #
        # After deep research (Bright Data, GitHub source code), we discovered:
        #
        # 1. W&B Launch Vertex runner source code (vertex_runner.py lines 189-199):
        #    https://github.com/wandb/wandb/blob/main/wandb/sdk/launch/runner/vertex_runner.py
        #
        #    execution_kwargs = dict(
        #        timeout=run_args.get("timeout"),
        #        service_account=run_args.get("service_account"),
        #        network=run_args.get("network"),
        #        enable_web_access=run_args.get("enable_web_access", False),
        #        experiment=run_args.get("experiment"),
        #        experiment_run=run_args.get("experiment_run"),
        #        tensorboard=run_args.get("tensorboard"),
        #        restart_job_on_worker_restart=run_args.get("restart_job_on_worker_restart", False),
        #        # ❌ scheduling_strategy NOT EXTRACTED FROM run_args!
        #    )
        #
        # 2. Then W&B calls: job.run(**execution_kwargs) WITHOUT scheduling_strategy!
        #    - This means our "run.scheduling_strategy" config is IGNORED
        #    - Vertex AI defaults to on-demand (not spot)
        #    - We get quota errors for custom_model_training_nvidia_p4_gpus (on-demand quota)
        #
        # 3. Our testing confirmed (execution vertex-ai-launcher-4m55t):
        #    - We send: "run": {"scheduling_strategy": "SPOT"}
        #    - W&B ignores it (not in execution_kwargs)
        #    - Quota error: aiplatform.googleapis.com/custom_model_training_nvidia_p4_gpus
        #                   ^^^^^^^^^^^^^^^^^^^ ON-DEMAND, not spot_model_training!
        #
        # References (comprehensive research 2025-11-16):
        # - https://github.com/wandb/wandb/blob/main/wandb/sdk/launch/runner/vertex_runner.py (source code)
        # - https://docs.wandb.ai/platform/launch/setup-vertex (W&B Launch structure: spec + run keys)
        # - https://docs.cloud.google.com/vertex-ai/docs/Training/use-spot-vms (Vertex AI spot config)
        # - https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec (REST API structure)
        #
        # WORKAROUND NEEDED: W&B Launch missing this feature!
        # Current approach (including scheduling_strategy in "run") is CORRECT per docs,
        # but W&B Launch vertex_runner.py doesn't implement it yet.
        #
        # Options:
        # 1. Patch launcher container (modify vertex_runner.py to extract scheduling_strategy)
        # 2. File W&B bug report (missing feature in W&B Launch)
        # 3. Use queue-level config (if supported)
        # 4. Fork wandb and maintain patched version
        #
        # For now, keeping correct config structure (may work in future W&B versions)
        launch_config = {
            "overrides": {
                "run_config": env_vars
            },
            "resource_args": {
                "gcp-vertex": {
                    "spec": {
                        "worker_pool_specs": [worker_pool_spec],
                        "staging_bucket": f"gs://{config.get('GCP_PROJECT_ID', '')}-{config.get('PROJECT_NAME', 'arr-coc')}-{config.get('TRAINING_GPU_REGION', 'us-central1')}-staging"
                    },
                    "run": {
                        "scheduling_strategy": "SPOT" if use_preemptible else "STANDARD",  # ✅ Correct location for W&B Launch!
                        "restart_job_on_worker_restart": False
                    }
                }
            }
        }

        # Note: For gcp-vertex, image goes in container_spec.image_uri (not as CLI arg)
        # Job artifact created by --uri, agent uses image from container_spec (no building!)

        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(launch_config, f)
            config_file = f.name

        # Temporarily rename git remote to prevent W&B from detecting GitHub URL
        # This forces W&B Launch to package local files without storing remote URL
        HIDDEN_REMOTE_NAME = "origin-changed-for-wandb-launch"
        remote_renamed = False

        try:
            # Check if 'origin' remote exists and rename it
            try:
                check_remote = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True,
                    cwd=repo_root
                )
                if check_remote.returncode == 0:
                    # Rename origin → origin-changed-for-wandb-launch (hide it from W&B)
                    subprocess.run(
                        ["git", "remote", "rename", "origin", HIDDEN_REMOTE_NAME],
                        capture_output=True,
                        cwd=repo_root
                    )
                    remote_renamed = True
            except Exception:
                pass  # No remote or rename failed - continue anyway

            # Generate cool run name (adjective-noun-timestamp)
            import time
            cool_name = generate_cool_run_name()  # "spazzy-forest", "ethereal-snowflake", etc.
            base_timestamp = int(time.time()) % 10000  # Last 4 digits
            run_name = f"{cool_name}-{base_timestamp}"  # "spazzy-forest-9868"

            # Collision detection: check if this name already exists
            # If collision, try incrementing up to 10 times
            attempt = 0
            while attempt < 10:
                if self._run_name_exists(run_name):
                    # Collision! Increment and try again
                    attempt += 1
                    run_name = f"{cool_name}-{(base_timestamp + attempt) % 10000}"
                else:
                    # Name is unique!
                    break

            # Build wandb launch command
            # For gcp-vertex with pre-built image:
            # - Use --docker-image to create DOCKER IMAGE JOB (not code job!)
            # - Docker image jobs don't require builder (noop works)
            # - Image also in container_spec.image_uri for Vertex AI
            # - No --uri (that creates code jobs which need building)

            cmd = [
                "wandb", "launch",
                "--docker-image", docker_image,  # Create docker image job (no building!)
                "--project", config.get("WANDB_PROJECT", self.project),
                "--entity", self.entity,  # Use entity from helper init
                "--queue", config.get("WANDB_LAUNCH_QUEUE_NAME", self.queue),
                "--resource", "gcp-vertex",  # GCP Vertex AI resource
                "--entry-point", "python ARR_COC/Training/train.py",  # Full command needed
                "--config", config_file,  # Config includes image in container_spec
                "--name", run_name,  # Cool name: ethereal-snowflake-1762661725
            ]

            # Docker image specified twice:
            # 1. --docker-image: Creates docker image job (no builder needed)
            # 2. container_spec.image_uri: Used by Vertex AI for actual execution

            # Run from repo root
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

        finally:
            # ALWAYS restore git remote if we renamed it (even on exception!)
            if remote_renamed:
                try:
                    subprocess.run(
                        ["git", "remote", "rename", HIDDEN_REMOTE_NAME, "origin"],
                        capture_output=True,
                        cwd=repo_root
                    )
                except Exception:
                    pass  # Best effort restore

            # Clean up temp file
            try:
                os.unlink(config_file)
            except Exception:
                pass

        if result.returncode != 0:
            # Only show W&B output on error
            print("\n" + "="*80)
            print("W&B Launch Error Output:")
            print("="*80)
            print(result.stdout)
            print(result.stderr)
            print("="*80 + "\n")
            raise Exception(f"Job submission failed: {result.stderr}")

        # Extract run ID or queue item ID from output
        # W&B Launch output formats:
        # - "Run queued: https://wandb.ai/entity/project/runs/RUN_ID"
        # - "View run at https://wandb.ai/entity/project/runs/RUN_ID"
        # - "Queued run RUN_ID"
        # - Queue item IDs: base64-encoded strings starting with capital letters (UnVu...)
        output = result.stdout + result.stderr  # Check both stdout and stderr

        run_id = None
        queue_name = None

        for line in output.split('\n'):
            # Try URL-based extraction first (when agent picks up job)
            if 'wandb.ai' in line and '/runs/' in line:
                parts = line.split('/runs/')
                if len(parts) >= 2:
                    # Extract run ID (may have trailing chars like ')' or '.')
                    extracted_id = parts[-1].split()[0].strip('.,;:()[]')
                    if extracted_id:
                        run_id = extracted_id
                        break

            # Check for "Added run to queue QUEUE_NAME" (when no agent running)
            if 'Added run to queue' in line:
                parts = line.split('Added run to queue')
                if len(parts) >= 2:
                    queue_name = parts[1].strip().rstrip('.')
                    continue

            # Try "Queued run RUN_ID" format
            if 'Queued run' in line or 'queued:' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() in ['run', 'queued:'] and i + 1 < len(parts):
                        potential_id = parts[i + 1].strip('.,;:()[]')
                        # Run IDs are usually alphanumeric with some special chars
                        if len(potential_id) > 5 and potential_id[0].isalnum():
                            run_id = potential_id
                            break
                if run_id:
                    break

        # Return tuple: (run_id, full_output, run_name)
        # Priority: actual run ID > run name we created > "var missing"
        if not run_id:
            if 'run_name' in locals():
                # Use the run name we generated (available in this scope)
                run_id = run_name  # "ethereal-snowflake-1762661725"
            else:
                run_id = "var missing!!"

        # Always return run_name as third element (or "unknown" if not available)
        final_run_name = run_name if 'run_name' in locals() else "unknown"
        return (run_id, output.strip(), final_run_name)

    def _run_name_exists(self, run_name: str) -> bool:
        """Check if a run with this name already exists (3 quick retries, then assume unique)"""
        # Try 3 times - if all fail, assume name is unique
        # Collision detection is not critical (worst case: slightly longer run name)
        for attempt in range(3):
            try:
                # Quick query to check if run name exists
                runs = self.api.runs(
                    f"{self.entity}/{self.project}",
                    filters={"display_name": run_name}
                )
                # Check if any runs exist with this name
                # Just peek at first result (don't iterate all)
                for run in runs:
                    return True  # Found at least one run with this name
                return False  # No runs found
            except Exception:
                # API error - retry (attempts 0, 1, 2)
                if attempt == 2:
                    # All 3 attempts failed - assume name is unique
                    return False
                # Otherwise continue to next retry
                continue

        # Should never reach here, but just in case
        return False

    def cancel_run(self, run_id: str):
        """Cancel a running job or delete a queued job"""
        # Queue item IDs are base64 encoded (start with capital letters like "UnVu...")
        # Regular run IDs are alphanumeric lowercase (like "abc123def")

        # Try queue item deletion first (if it's a queue item)
        try:
            queue = self.api.run_queue(self.entity, self.queue)
            for item in queue.items:
                if item.id == run_id:
                    # Found the queue item - delete it
                    item.delete()
                    return
        except Exception:
            pass  # Not a queue item or queue doesn't exist

        # Fall back to regular run cancellation
        run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
        run.stop()



