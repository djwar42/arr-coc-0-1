"""
API Helpers - Retry logic for gcloud and W&B API calls.

Provides robust retry wrappers for external API calls with good error handling.
"""

# <claudes_code_comments>
# ** Function List **
# run_gcloud_with_retry(cmd, max_retries, timeout, operation_name, stdin_input) - Run gcloud with 3-retry logic
# run_wandb_api_with_retry(api_call, max_retries, operation_name) - Run W&B API call with 3-retry logic
# run_requests_with_retry(method, url, max_retries, timeout, operation_name, **kwargs) - Run HTTP request with 3-retry logic
# run_gcloud_batch_parallel(commands, max_workers) - Run multiple gcloud commands in parallel (max 10 at once)
# run_wandb_batch_parallel(api_calls, max_workers) - Run multiple W&B API calls in parallel (max 10 at once)
# GeneralAccumulator - Async prefetch for ANY Python callable (most flexible, use for mixed workloads)
#   - start(key, callable_fn) - Start async operation
#   - get(key) - Get result (blocks if not ready)
#   - is_done(key) - Check if ready (non-blocking)
#   - wait_and_render(callback, order) - Easy progressive rendering helper
#   - get_all() - Get all results as dict
#   - add_accumulator(key, accumulator) - Nest other accumulators (composition)
# GCloudAccumulator - Async prefetch pattern for gcloud calls (start early, get results when needed)
# RequestsAccumulator - Async prefetch pattern for HTTP requests (start early, get results when needed)
#
# ** Technical Review **
# API helpers provide retry wrappers for external API calls with exponential backoff.
# Single-command helpers (run_gcloud_with_retry, run_wandb_api_with_retry, run_requests_with_retry) provide 3 retries
# with 2-second initial delay, doubling each retry.
#
# Batch parallel helpers (run_gcloud_batch_parallel, run_wandb_batch_parallel) run multiple
# operations simultaneously using ThreadPoolExecutor with max_workers limit (default 10).
# BATCHING BEHAVIOR: If len(commands) > max_workers, commands run in batches.
# Example: 15 commands with max_workers=10 = First batch (10 parallel) + Second batch (5 parallel).
# ThreadPoolExecutor enforces the limit - at most max_workers threads run simultaneously.
#
# Each parallel operation still gets individual retry logic (3 retries per command/API call).
# Results returned as list of tuples: (index, result_or_none, error_or_none) maintaining order.
#
# Accumulator classes provide async prefetch pattern:
# - GeneralAccumulator: For ANY Python callable (most flexible - use for mixed workloads)
# - GCloudAccumulator: For gcloud commands (start(key, cmd, ...), get(key))
# - RequestsAccumulator: For HTTP requests (start(key, method, url, ...), get(key))
# - All enable overlapping API calls with other work for massive speedup
# Example: Check 4 images sequentially = 40s, with accumulator = 10s (4× faster)
#
# PROGRESSIVE RENDERING - Two Ways:
# 1. EASY WAY: wait_and_render() helper (recommended!)
#   acc.start("config", validate_fn)
#   acc.start("queue", check_fn)
#   results = acc.wait_and_render(render_callback, order=["config", "queue"])
#   acc.shutdown()
#   # Automatic progressive rendering with optional ordering!
#
# 2. Manual way: is_done(key) polling
#   acc.start("mecha", mecha_fn)
#   acc.start("zeus", zeus_fn)
#   while not all_done:
#       if acc.is_done("mecha"): render_mecha()
#       if acc.is_done("zeus"): render_zeus()
#   # User sees output as each operation completes, not all at end!
#
# NOTE: All accumulators use the same underlying pattern (ThreadPoolExecutor).
# GeneralAccumulator is the base generic pattern - accepts ANY callable.
# Specific accumulators (GCloud, Requests) provide clearer APIs for common cases.
# Use GeneralAccumulator for new code with mixed workloads!
# </claudes_code_comments>

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple


def run_gcloud_with_retry(
    cmd: List[str],
    max_retries: int = 3,
    timeout: int = 30,
    operation_name: str = "gcloud command",
    stdin_input: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    Run gcloud command with retry logic and exponential backoff.

    Args:
        cmd: Command list (e.g., ["gcloud", "compute", "instances", "list"])
        max_retries: Number of retry attempts (default: 3)
        timeout: Command timeout in seconds (default: 30)
        operation_name: Description for error messages (default: "gcloud command")
        stdin_input: Optional stdin data (for --data-file=- commands like secrets)

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode

    Raises:
        subprocess.TimeoutExpired: If command times out after all retries
        RuntimeError: If command fails after all retries

    Retry schedule:
        Attempt 1: immediate
        Attempt 2: 2s delay
        Attempt 3: 4s delay
        Attempt 4: 8s delay

    Example:
        result = run_gcloud_with_retry(
            ["gcloud", "compute", "instances", "list", "--format=json"],
            max_retries=3,
            timeout=60,
            operation_name="list compute instances",
        )
        if result.returncode == 0:
            instances = json.loads(result.stdout)

    Example with stdin (secrets):
        result = run_gcloud_with_retry(
            ["gcloud", "secrets", "versions", "add", "my-secret", "--data-file=-"],
            operation_name="add secret version",
            stdin_input="secret-value",
        )
    """
    retry_delay = 2  # Initial delay in seconds

    for attempt in range(1, max_retries + 1):
        try:
            if stdin_input is not None:
                # Pass data via stdin (for --data-file=- commands)
                result = subprocess.run(
                    cmd,
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                # Standard command (no stdin)
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )

            # Success (returncode 0) or non-zero but completed
            return result

        except subprocess.TimeoutExpired as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise  # Final attempt failed

        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts: {str(e)}"
                )

    # Should never reach here (all paths return or raise)
    raise RuntimeError(f"Unexpected error in {operation_name}")


def run_wandb_api_with_retry(
    api_call: Callable,
    max_retries: int = 3,
    operation_name: str = "W&B API call",
) -> Any:
    """
    Run W&B API call with retry logic and exponential backoff.

    Args:
        api_call: Lambda or function that makes the W&B API call
        max_retries: Number of retry attempts (default: 3)
        operation_name: Description for error messages

    Returns:
        API call result

    Raises:
        RuntimeError: If API call fails after all retries

    Example:
        import wandb
        api = wandb.Api()
        runs = run_wandb_api_with_retry(
            lambda: api.runs("entity/project", filters={"state": "running"}),
            operation_name="fetch running jobs"
        )
    """
    retry_delay = 2

    for attempt in range(1, max_retries + 1):
        try:
            return api_call()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts: {str(e)}"
                )


def run_gcloud_batch_parallel(
    commands: List[Dict[str, Any]],
    max_workers: int = 10,
) -> List[Tuple[int, Optional[subprocess.CompletedProcess], Optional[str]]]:
    """
    Run multiple gcloud commands in PARALLEL with retry logic.

    Each command automatically gets 3 retries via run_gcloud_with_retry().

    Batching behavior: If len(commands) > max_workers, commands run in batches.
    For example, 15 commands with max_workers=10 runs as: 10 parallel + 5 parallel.

    Args:
        commands: List of command dicts, each containing:
            - cmd: Command list (required)
            - operation_name: Description for errors (required)
            - timeout: Command timeout in seconds (optional, default 30)
            - max_retries: Retry attempts (optional, default 3)
            - stdin_input: Optional stdin data (optional, for --data-file=- commands)
        max_workers: Max parallel threads (default 10). ThreadPoolExecutor enforces
                     this limit - at most max_workers commands run simultaneously.

    Returns:
        List of tuples: (index, result_or_none, error_or_none)
        - index: Position in input list
        - result: subprocess.CompletedProcess if success, None if failed
        - error: Error string if failed, None if success

    Example:
        commands = [
            {
                "cmd": ["gcloud", "compute", "instances", "list", "--format=json"],
                "operation_name": "list instances",
            },
            {
                "cmd": ["gcloud", "compute", "disks", "list", "--format=json"],
                "operation_name": "list disks",
            },
            {
                "cmd": ["gcloud", "secrets", "versions", "add", "my-secret", "--data-file=-"],
                "operation_name": "add secret version",
                "stdin_input": "secret-value",
            },
        ]

        results = run_gcloud_batch_parallel(commands, max_workers=3)

        for idx, result, error in results:
            if error:
                print(f"Command {idx} failed: {error}")
            else:
                print(f"Command {idx} succeeded: {result.stdout}")
    """

    def run_single_command(idx: int, cmd_dict: Dict[str, Any]) -> Tuple[int, Optional[subprocess.CompletedProcess], Optional[str]]:
        """Run a single command with retry logic."""
        try:
            result = run_gcloud_with_retry(
                cmd=cmd_dict["cmd"],
                operation_name=cmd_dict["operation_name"],
                timeout=cmd_dict.get("timeout", 30),
                max_retries=cmd_dict.get("max_retries", 3),
                stdin_input=cmd_dict.get("stdin_input"),
            )
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))

    # Run all commands in parallel (ThreadPoolExecutor enforces max_workers limit)
    # If len(commands) > max_workers, they run in batches automatically
    # Example: 15 commands with max_workers=10 = batch1 (10 parallel) + batch2 (5 parallel)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all commands (executor queues any beyond max_workers)
        futures = {
            executor.submit(run_single_command, idx, cmd_dict): idx
            for idx, cmd_dict in enumerate(commands)
        }

        # Collect results as they complete (maintains batching behavior)
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by original index to maintain order (batching can change completion order)
    results.sort(key=lambda x: x[0])

    return results


def run_wandb_batch_parallel(
    api_calls: List[Dict[str, Any]],
    max_workers: int = 10,
) -> List[Tuple[int, Optional[Any], Optional[str]]]:
    """
    Run multiple W&B API calls in PARALLEL with retry logic.

    Each call automatically gets 3 retries via run_wandb_api_with_retry().

    Batching behavior: If len(api_calls) > max_workers, calls run in batches.
    For example, 15 calls with max_workers=10 runs as: 10 parallel + 5 parallel.

    Args:
        api_calls: List of API call dicts, each containing:
            - api_call: Lambda or function that makes the W&B API call (required)
            - operation_name: Description for errors (required)
            - timeout: Call timeout in seconds (optional, default 30.0)
            - max_retries: Retry attempts (optional, default 3)
        max_workers: Max parallel threads (default 10). ThreadPoolExecutor enforces
                     this limit - at most max_workers calls run simultaneously.

    Returns:
        List of tuples: (index, result_or_none, error_or_none)
        - index: Position in input list
        - result: API result if success, None if failed
        - error: Error string if failed, None if success

    Example:
        import wandb
        api = wandb.Api()

        api_calls = [
            {
                "api_call": lambda: api.runs("entity/project", filters={"state": "running"}),
                "operation_name": "fetch running jobs",
            },
            {
                "api_call": lambda: api.artifact("entity/project/artifact:latest"),
                "operation_name": "fetch artifact",
            },
        ]

        results = run_wandb_batch_parallel(api_calls, max_workers=2)

        for idx, result, error in results:
            if error:
                print(f"API call {idx} failed: {error}")
            else:
                print(f"API call {idx} succeeded: {result}")
    """

    def run_single_api_call(idx: int, call_dict: Dict[str, Any]) -> Tuple[int, Optional[Any], Optional[str]]:
        """Run a single API call with retry logic."""
        try:
            result = run_wandb_api_with_retry(
                api_call=call_dict["api_call"],
                operation_name=call_dict["operation_name"],
                max_retries=call_dict.get("max_retries", 3),
            )
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))

    # Run all API calls in parallel (ThreadPoolExecutor enforces max_workers limit)
    # If len(api_calls) > max_workers, they run in batches automatically
    # Example: 15 calls with max_workers=10 = batch1 (10 parallel) + batch2 (5 parallel)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all API calls (executor queues any beyond max_workers)
        futures = {
            executor.submit(run_single_api_call, idx, call_dict): idx
            for idx, call_dict in enumerate(api_calls)
        }

        # Collect results as they complete (maintains batching behavior)
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by original index to maintain order (batching can change completion order)
    results.sort(key=lambda x: x[0])

    return results


class GeneralAccumulator:
    """
    Async prefetch pattern for ANY Python callable - start early, get results when needed.

    The most flexible accumulator - accepts any callable (functions, lambdas, methods).
    Use when you have a mix of different operation types to run in parallel.

    Allows overlapping arbitrary Python calls with other work:
    1. Start multiple independent callables (non-blocking)
    2. Do other work while they run in background
    3. Get results when needed (blocks only if not ready)

    Example (Mixed Workload):
        acc = GeneralAccumulator(max_workers=20)

        # Start checks we'll need later (returns immediately!)
        # W&B API call
        acc.start("wandb", lambda: wandb.Api().runs("entity/project"))

        # Subprocess (gcloud)
        acc.start("gcloud", lambda: subprocess.run(
            ["gcloud", "compute", "instances", "list"],
            capture_output=True, timeout=30
        ))

        # HTTP request
        acc.start("http", lambda: requests.get("https://api.example.com/status"))

        # File I/O
        acc.start("file", lambda: json.load(open("config.json")))

        # Custom function
        acc.start("custom", lambda: my_complex_calculation(data))

        # Do other work (overlaps with all 5 calls running in parallel!)
        prepare_config()
        validate_settings()

        # Get results when needed (waits only if not ready)
        wandb_runs = acc.get("wandb")      # Maybe already done!
        gcloud_result = acc.get("gcloud")  # Maybe already done!
        http_resp = acc.get("http")        # Maybe already done!
        config = acc.get("file")           # Maybe already done!
        result = acc.get("custom")         # Maybe already done!

        acc.shutdown()  # Clean up

    Example (Simple - All Same Type):
        acc = GeneralAccumulator(max_workers=10)

        # Start multiple database queries
        for i in range(10):
            acc.start(f"query_{i}", lambda i=i: db.execute(f"SELECT * FROM table_{i}"))

        # Get all results
        results = [acc.get(f"query_{i}") for i in range(10)]
        acc.shutdown()

    Example (Progressive Rendering - Show Results as Each Completes):
        import time
        acc = GeneralAccumulator(max_workers=2)

        # Start two long operations (MECHA takes 5s, ZEUS takes 8s)
        acc.start("mecha", lambda: expensive_mecha_calculation())
        acc.start("zeus", lambda: expensive_zeus_calculation())

        # Progressive rendering - show output as each finishes!
        mecha_rendered = False
        zeus_rendered = False

        while not (mecha_rendered and zeus_rendered):
            # Check if MECHA is done (renders at ~5s)
            if acc.is_done("mecha") and not mecha_rendered:
                mecha_result = acc.get("mecha")
                print(f"✅ MECHA: {mecha_result}")  # Shows immediately at 5s!
                mecha_rendered = True

            # Check if ZEUS is done (renders at ~8s, after MECHA)
            if acc.is_done("zeus") and not zeus_rendered:
                zeus_result = acc.get("zeus")
                print(f"✅ ZEUS: {zeus_result}")  # Shows at 8s
                zeus_rendered = True

            time.sleep(0.1)  # Poll interval

        acc.shutdown()

        # Result: User sees MECHA output at 5s, ZEUS at 8s (cascade!)
        # Without progressive rendering: User waits 8s, sees both at once

    Performance:
        Sequential: 5 operations × 5s each = 25s
        Accumulator: Start all 5 at once = 5s (5× faster!)
        Progressive: Show each result as it completes (better UX!)

    When to Use:
        - Mixed workload (subprocess + API + HTTP + custom)
        - Don't want to import specific accumulator classes
        - Need maximum flexibility
        - Prototyping or one-off scripts
        - Progressive rendering (show results as they complete)

    When to Use Specific Accumulators Instead:
        - GCloudAccumulator: If ONLY running gcloud commands (clearer API)
        - RequestsAccumulator: If ONLY making HTTP requests (built-in retry)
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize accumulator.

        Args:
            max_workers: Max concurrent callables (default 10)
        """
        self.futures = {}  # key -> Future (for callables)
        self.accumulators = {}  # key -> Accumulator (nested accumulators)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    def start(
        self,
        key: str,
        callable_fn: Callable,
        max_retries: int = 3,
        operation_name: str = "operation",
    ):
        """
        Start a callable in background (non-blocking).

        Args:
            key: Unique identifier for this callable
            callable_fn: Any Python callable (function, lambda, method)
                        Should take no arguments (use lambda for parameterized calls)
            max_retries: Retry attempts (default 3)
            operation_name: Description for errors

        Returns:
            None (callable starts immediately, returns control to caller)

        Example:
            # Lambda with parameters
            acc.start("fetch", lambda: api.get_data(user_id=123))

            # Function reference
            acc.start("compute", my_function)

            # Method call
            acc.start("validate", obj.validate_data)
        """
        if key in self.futures or key in self.accumulators:
            raise KeyError(f"Key '{key}' already used (callable or accumulator)")

        # Submit to executor (returns immediately!)
        future = self.executor.submit(
            run_wandb_api_with_retry,  # Reuse existing retry wrapper
            callable_fn,
            max_retries,
            operation_name
        )
        self.futures[key] = future

    def add_accumulator(
        self,
        key: str,
        accumulator,  # GCloudAccumulator, RequestsAccumulator, or GeneralAccumulator
    ):
        """
        Add a nested accumulator (composition pattern).

        Allows composing multiple accumulators together. When you get(key),
        it returns accumulator.get_all() with all nested results.

        Args:
            key: Unique identifier for this accumulator
            accumulator: Another accumulator instance (GCloudAccumulator, etc.)

        Returns:
            None (accumulator added, operations already started on it)

        Example:
            # Create specialized accumulators
            gcloud_acc = GCloudAccumulator(max_workers=10)
            gcloud_acc.start("image1", ["gcloud", "artifacts", ...])
            gcloud_acc.start("image2", ["gcloud", "artifacts", ...])
            gcloud_acc.start("pool", ["gcloud", "builds", "worker-pools", ...])

            requests_acc = RequestsAccumulator(max_workers=5)
            requests_acc.start("api1", "GET", "https://...")
            requests_acc.start("api2", "GET", "https://...")

            # Compose them in GeneralAccumulator
            general_acc = GeneralAccumulator(max_workers=20)
            general_acc.add_accumulator("gcloud_ops", gcloud_acc)
            general_acc.add_accumulator("http_ops", requests_acc)

            # Also add some direct callables
            general_acc.start("wandb", lambda: wandb.Api().runs(...))
            general_acc.start("custom", lambda: my_function())

            # Do other work while ALL operations run in parallel!
            prepare_config()

            # Get nested results
            gcloud_results = general_acc.get("gcloud_ops")  # {image1: ..., image2: ..., pool: ...}
            http_results = general_acc.get("http_ops")      # {api1: ..., api2: ...}
            wandb_result = general_acc.get("wandb")         # Direct result
            custom_result = general_acc.get("custom")       # Direct result

            general_acc.shutdown()  # Shuts down all nested accumulators too!

        Benefits:
            - Organize operations by type (gcloud, HTTP, custom)
            - Reuse existing specialized accumulators
            - Hierarchical result structure
            - Clean separation of concerns
        """
        if key in self.futures or key in self.accumulators:
            raise KeyError(f"Key '{key}' already used (callable or accumulator)")

        self.accumulators[key] = accumulator

    def get(self, key: str) -> Any:
        """
        Get result for a callable OR nested accumulator (waits if not ready yet).

        Args:
            key: Identifier used in start() or add_accumulator()

        Returns:
            - For callables: Direct result (any type)
            - For nested accumulators: Dict[str, Any] from accumulator.get_all()

        Raises:
            KeyError: If key not found (callable/accumulator not added)
        """
        if key in self.futures:
            # Callable result (direct)
            future = self.futures[key]
            result = future.result()  # Wait for result (blocks if not ready)
            return result
        elif key in self.accumulators:
            # Nested accumulator result (all results as dict)
            return self.accumulators[key].get_all()
        else:
            raise KeyError(f"No operation with key: {key}")

    def is_done(self, key: str) -> bool:
        """
        Check if callable or nested accumulator is done (non-blocking).

        Args:
            key: Identifier used in start() or add_accumulator()

        Returns:
            - For callables: True if done, False if still running
            - For nested accumulators: True if ALL operations in accumulator are done

        Raises:
            KeyError: If key not found
        """
        if key in self.futures:
            # Callable done?
            return self.futures[key].done()
        elif key in self.accumulators:
            # Nested accumulator - check if ALL its operations are done
            accumulator = self.accumulators[key]
            # Check all futures in nested accumulator
            if hasattr(accumulator, 'futures'):
                return all(f.done() for f in accumulator.futures.values())
            else:
                return True  # Empty accumulator = done
        else:
            raise KeyError(f"No operation with key: {key}")

    def get_all(self) -> Dict[str, Any]:
        """
        Get ALL results as a dict (callables + nested accumulators).

        Returns:
            Dict mapping keys to results:
            - Callable keys → direct results
            - Accumulator keys → nested dicts from accumulator.get_all()

        Example:
            results = acc.get_all()
            # {
            #   "wandb": {...},                    # Direct callable result
            #   "custom": 42,                      # Direct callable result
            #   "gcloud_ops": {                    # Nested accumulator
            #       "image1": {...},
            #       "image2": {...},
            #       "pool": {...}
            #   },
            #   "http_ops": {                      # Nested accumulator
            #       "api1": {...},
            #       "api2": {...}
            #   }
            # }
        """
        results = {}

        # Get all callable results
        for key in self.futures:
            results[key] = self.get(key)

        # Get all nested accumulator results
        for key in self.accumulators:
            results[key] = self.accumulators[key].get_all()

        return results

    def wait_and_render(
        self,
        render_callback: Callable[[str, Any], None],
        order: Optional[List[str]] = None,
        poll_interval: float = 0.05
    ) -> Dict[str, Any]:
        """
        Automatically render results as each completes (progressive rendering helper).

        This is the EASY WAY to do progressive rendering - no manual loop needed!

        Args:
            render_callback: Function called for each result: callback(key, result)
            order: Optional order to enforce (renders only after previous keys done)
                   Example: ["config", "queue", "infra"] ensures config renders first
            poll_interval: How often to check for completion (seconds, default 0.05)

        Returns:
            Dict of all results (same as get_all())

        Example (Simple - No ordering):
            acc = GeneralAccumulator()
            acc.start("config", lambda: validate_config())
            acc.start("queue", lambda: check_queue())
            acc.start("pricing", lambda: fetch_pricing())

            def render(key, result):
                if key == "config":
                    print("✓ Config good!")
                elif key == "queue":
                    print(f"✓ Queue: {len(result)} jobs")
                elif key == "pricing":
                    print("✓ Good prices!")

            # Automatic progressive rendering!
            results = acc.wait_and_render(render)
            acc.shutdown()

        Example (With Ordering - Enforces sequence):
            acc = GeneralAccumulator()
            acc.start("config", lambda: validate_config())
            acc.start("queue", lambda: check_queue())
            acc.start("infra", lambda: check_infra())
            acc.start("pricing", lambda: fetch_pricing())

            def render(key, result):
                print(f"✓ {key} done!")

            # Config renders first, then queue (even if queue finishes before config!)
            results = acc.wait_and_render(
                render,
                order=["config", "queue", "infra", "pricing"]
            )
            acc.shutdown()

        Why use this instead of manual loop?
        - No tracking dict needed
        - No poll loop to write
        - Optional ordering built-in
        - Returns all results like get_all()
        - One-liner for progressive rendering!

        Performance:
        - Same as manual loop (polls every 0.05s)
        - User sees results immediately as they complete
        - Total time = longest operation (parallel execution)
        """
        import time

        # Determine keys to render
        if order:
            keys_to_render = order
        else:
            # No order - render as completed (all futures + accumulators)
            keys_to_render = list(self.futures.keys()) + list(self.accumulators.keys())

        rendered = {key: False for key in keys_to_render}
        results = {}

        while not all(rendered.values()):
            for i, key in enumerate(keys_to_render):
                # Skip if already rendered
                if rendered[key]:
                    continue

                # If order specified, check if previous keys are done
                if order and i > 0:
                    prev_key = keys_to_render[i - 1]
                    if not rendered[prev_key]:
                        continue  # Wait for previous key

                # Check if this key is done
                if self.is_done(key):
                    result = self.get(key)
                    results[key] = result
                    render_callback(key, result)
                    rendered[key] = True

            time.sleep(poll_interval)

        return results

    def shutdown(self):
        """Shutdown executor AND all nested accumulators (waits for completion)."""
        # Shutdown nested accumulators first
        for accumulator in self.accumulators.values():
            accumulator.shutdown()

        # Then shutdown our executor
        self.executor.shutdown(wait=True)


class GCloudAccumulator:
    """
    Async prefetch pattern for gcloud calls - start early, get results when needed.

    Allows overlapping gcloud API calls with other work:
    1. Start multiple independent gcloud calls (non-blocking)
    2. Do other work while they run in background
    3. Get results when needed (blocks only if not ready)

    Example:
        acc = GCloudAccumulator()

        # Start checks we'll need later (returns immediately!)
        acc.start("image1", ["gcloud", "artifacts", "docker", "images", "list", "..."],
                  operation_name="check image1")
        acc.start("image2", ["gcloud", "artifacts", "docker", "images", "list", "..."],
                  operation_name="check image2")
        acc.start("image3", ["gcloud", "artifacts", "docker", "images", "list", "..."],
                  operation_name="check image3")

        # Do other work (overlaps with API calls!)
        prepare_config()
        validate_settings()

        # Get results when needed (waits only if not ready)
        image1_result = acc.get("image1")  # Maybe already done!
        image2_result = acc.get("image2")  # Maybe already done!
        image3_result = acc.get("image3")  # Maybe already done!

        acc.shutdown()  # Clean up

    Performance:
        Sequential: Check 4 images × 10s each = 40s
        Accumulator: Start all 4 at once = 10s (4× faster!)
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize accumulator.

        Args:
            max_workers: Max concurrent gcloud calls (default 10)
        """
        self.futures = {}  # key -> Future
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    def start(
        self,
        key: str,
        cmd: List[str],
        max_retries: int = 3,
        timeout: int = 60,
        operation_name: str = "operation",
        stdin_input: Optional[str] = None
    ):
        """
        Start a gcloud command in background (non-blocking).

        Args:
            key: Unique identifier for this command
            cmd: Command list (e.g., ["gcloud", "compute", "instances", "list"])
            max_retries: Retry attempts (default 3)
            timeout: Timeout in seconds (default 60)
            operation_name: Description for errors
            stdin_input: Optional stdin data (for secrets)

        Returns:
            None (command starts immediately, returns control to caller)
        """
        if key in self.futures:
            raise KeyError(f"Command with key '{key}' already started")

        # Submit to executor (returns immediately!)
        future = self.executor.submit(
            run_gcloud_with_retry,
            cmd,
            max_retries,
            timeout,
            operation_name,
            stdin_input
        )
        self.futures[key] = future

    def get(self, key: str) -> subprocess.CompletedProcess:
        """
        Get result for a command (waits if not ready yet).

        Args:
            key: Identifier used in start()

        Returns:
            subprocess.CompletedProcess result

        Raises:
            KeyError: If key not found (command not started)
        """
        if key not in self.futures:
            raise KeyError(f"No command started with key: {key}")

        future = self.futures[key]

        # Wait for result (blocks if not ready)
        result = future.result()
        return result

    def is_done(self, key: str) -> bool:
        """
        Check if command is done (non-blocking).

        Args:
            key: Identifier used in start()

        Returns:
            True if done, False if still running

        Raises:
            KeyError: If key not found
        """
        if key not in self.futures:
            raise KeyError(f"No command started with key: {key}")

        return self.futures[key].done()

    def get_all(self) -> Dict[str, Any]:
        """
        Get ALL gcloud command results as a dict.

        Returns:
            Dict mapping keys to subprocess.CompletedProcess results

        Example:
            results = acc.get_all()
            # {
            #   "image1": CompletedProcess(...),
            #   "image2": CompletedProcess(...),
            #   "pool": CompletedProcess(...)
            # }
        """
        results = {}
        for key in self.futures:
            results[key] = self.get(key)
        return results

    def wait_and_render(
        self,
        render_callback: Callable[[str, subprocess.CompletedProcess], None],
        order: Optional[List[str]] = None,
        poll_interval: float = 0.05
    ) -> Dict[str, subprocess.CompletedProcess]:
        """
        Automatically render gcloud results as each completes (progressive rendering).

        Args:
            render_callback: Function called for each result: callback(key, result)
            order: Optional order to enforce (renders only after previous keys done)
            poll_interval: How often to check for completion (seconds, default 0.05)

        Returns:
            Dict mapping keys to subprocess.CompletedProcess results

        Example:
            acc = GCloudAccumulator()
            acc.start("image1", ["gcloud", "artifacts", ...])
            acc.start("image2", ["gcloud", "artifacts", ...])

            def render(key, result):
                if result.returncode == 0:
                    print(f"✓ {key} complete!")

            results = acc.wait_and_render(render, order=["image1", "image2"])
            acc.shutdown()
        """
        import time

        keys_to_render = order if order else list(self.futures.keys())
        rendered = {key: False for key in keys_to_render}
        results = {}

        while not all(rendered.values()):
            for i, key in enumerate(keys_to_render):
                if rendered[key]:
                    continue

                if order and i > 0:
                    prev_key = keys_to_render[i - 1]
                    if not rendered[prev_key]:
                        continue

                if self.is_done(key):
                    result = self.get(key)
                    results[key] = result
                    render_callback(key, result)
                    rendered[key] = True

            time.sleep(poll_interval)

        return results

    def shutdown(self):
        """Shutdown executor (waits for all commands to complete)."""
        self.executor.shutdown(wait=True)


def run_requests_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    operation_name: str = "HTTP request",
    **kwargs
) -> Any:
    """
    Run HTTP request with retry logic and exponential backoff.

    Args:
        method: HTTP method ("GET", "POST", "PUT", "DELETE")
        url: URL to request
        max_retries: Number of retry attempts (default: 3)
        timeout: Request timeout in seconds (default: 30)
        operation_name: Description for error messages
        **kwargs: Additional arguments passed to requests (headers, data, json, etc.)

    Returns:
        requests.Response object

    Raises:
        RuntimeError: If request fails after all retries

    Retry schedule:
        Attempt 1: immediate
        Attempt 2: 2s delay
        Attempt 3: 4s delay
        Attempt 4: 8s delay

    Example:
        response = run_requests_with_retry(
            "GET",
            "https://api.example.com/data",
            headers={"Authorization": "Bearer token"},
            timeout=10,
            operation_name="fetch API data",
        )
        if response.status_code == 200:
            data = response.json()
    """
    import requests

    retry_delay = 2

    for attempt in range(1, max_retries + 1):
        try:
            # Make request
            response = requests.request(
                method=method.upper(),
                url=url,
                timeout=timeout,
                **kwargs
            )

            # Success (any response, even 4xx/5xx - caller decides what's success)
            return response

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts: Timeout"
                )

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts: {str(e)}"
                )

    # Should never reach here
    raise RuntimeError(f"Unexpected error in {operation_name}")


class RequestsAccumulator:
    """
    Async prefetch pattern for HTTP requests - start early, get results when needed.

    Allows overlapping HTTP API calls with other work:
    1. Start multiple independent requests (non-blocking)
    2. Do other work while they run in background
    3. Get results when needed (blocks only if not ready)

    Example:
        acc = RequestsAccumulator()

        # Start checks we'll need later (returns immediately!)
        acc.start("pricing", "GET", "https://cloudbilling.googleapis.com/v1/...",
                  headers={"Authorization": "Bearer token"},
                  operation_name="fetch pricing")
        acc.start("metadata", "GET", "http://metadata.google.internal/...",
                  headers={"Metadata-Flavor": "Google"},
                  operation_name="get token")

        # Do other work (overlaps with API calls!)
        prepare_config()
        validate_settings()

        # Get results when needed (waits only if not ready)
        pricing_resp = acc.get("pricing")      # Maybe already done!
        metadata_resp = acc.get("metadata")    # Maybe already done!

        acc.shutdown()  # Clean up

    Performance:
        Sequential: 4 API calls × 5s each = 20s
        Accumulator: Start all 4 at once = 5s (4× faster!)
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize accumulator.

        Args:
            max_workers: Max concurrent requests (default 10)
        """
        self.futures = {}  # key -> Future
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    def start(
        self,
        key: str,
        method: str,
        url: str,
        max_retries: int = 3,
        timeout: int = 30,
        operation_name: str = "HTTP request",
        **kwargs
    ):
        """
        Start an HTTP request in background (non-blocking).

        Args:
            key: Unique identifier for this request
            method: HTTP method ("GET", "POST", etc.)
            url: URL to request
            max_retries: Retry attempts (default 3)
            timeout: Timeout in seconds (default 30)
            operation_name: Description for errors
            **kwargs: Additional arguments for requests (headers, data, json, etc.)

        Returns:
            None (request starts immediately, returns control to caller)
        """
        if key in self.futures:
            raise KeyError(f"Request with key '{key}' already started")

        # Submit to executor (returns immediately!)
        future = self.executor.submit(
            run_requests_with_retry,
            method,
            url,
            max_retries,
            timeout,
            operation_name,
            **kwargs
        )
        self.futures[key] = future

    def get(self, key: str) -> Any:
        """
        Get result for a request (waits if not ready yet).

        Args:
            key: Identifier used in start()

        Returns:
            requests.Response object

        Raises:
            KeyError: If key not found (request not started)
        """
        if key not in self.futures:
            raise KeyError(f"No request started with key: {key}")

        future = self.futures[key]

        # Wait for result (blocks if not ready)
        result = future.result()
        return result

    def is_done(self, key: str) -> bool:
        """
        Check if request is done (non-blocking).

        Args:
            key: Identifier used in start()

        Returns:
            True if done, False if still running

        Raises:
            KeyError: If key not found
        """
        if key not in self.futures:
            raise KeyError(f"No request started with key: {key}")

        return self.futures[key].done()

    def get_all(self) -> Dict[str, Any]:
        """
        Get ALL HTTP request results as a dict.

        Returns:
            Dict mapping keys to requests.Response results

        Example:
            results = acc.get_all()
            # {
            #   "api1": Response(...),
            #   "api2": Response(...)
            # }
        """
        results = {}
        for key in self.futures:
            results[key] = self.get(key)
        return results

    def wait_and_render(
        self,
        render_callback: Callable[[str, Any], None],
        order: Optional[List[str]] = None,
        poll_interval: float = 0.05
    ) -> Dict[str, Any]:
        """
        Automatically render HTTP results as each completes (progressive rendering).

        Args:
            render_callback: Function called for each result: callback(key, result)
            order: Optional order to enforce (renders only after previous keys done)
            poll_interval: How often to check for completion (seconds, default 0.05)

        Returns:
            Dict mapping keys to requests.Response results

        Example:
            acc = RequestsAccumulator()
            acc.start("api1", "GET", "https://api.example.com/status")
            acc.start("api2", "GET", "https://api.example.com/health")

            def render(key, result):
                if result.status_code == 200:
                    print(f"✓ {key}: {result.status_code}")

            results = acc.wait_and_render(render, order=["api1", "api2"])
            acc.shutdown()
        """
        import time

        keys_to_render = order if order else list(self.futures.keys())
        rendered = {key: False for key in keys_to_render}
        results = {}

        while not all(rendered.values()):
            for i, key in enumerate(keys_to_render):
                if rendered[key]:
                    continue

                if order and i > 0:
                    prev_key = keys_to_render[i - 1]
                    if not rendered[prev_key]:
                        continue

                if self.is_done(key):
                    result = self.get(key)
                    results[key] = result
                    render_callback(key, result)
                    rendered[key] = True

            time.sleep(poll_interval)

        return results

    def shutdown(self):
        """Shutdown executor (waits for all requests to complete)."""
        self.executor.shutdown(wait=True)
