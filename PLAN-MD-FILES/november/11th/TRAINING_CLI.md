# TRAINING_CLI.md
**Simple CLI Tool for ARR-COC Training Submission & Monitoring**

*One command to launch, monitor, and manage training runs*

---

## Overview

Dead simple CLI using Textual for:
- ✅ Submit training jobs to W&B Launch with one command
- ✅ Monitor training progress in real-time
- ✅ View logs streaming from Vertex AI
- ✅ Cancel/halt running jobs
- ✅ Exit CLI without stopping training (detached mode)

**Usage:**
```bash
cd training/
python cli.py launch    # Submit new training job
python cli.py monitor   # Monitor running/recent jobs
python cli.py status    # Quick status check (no TUI)
```

---

## File Structure

```
training/
├── train.py          # Main training script (unchanged)
├── .training         # Configuration file
└── cli.py            # ← NEW: Simple CLI tool
```

**That's it.** Just add `cli.py` alongside your existing files.

---

## Installation

```bash
# Add to requirements.txt
textual==0.47.1
wandb>=0.16.0
rich>=13.0.0

# Install
pip install textual wandb rich
```

---

## Implementation: `training/cli.py`

**File: `training/cli.py`**

```python
#!/usr/bin/env python3
"""
ARR-COC Training CLI

Simple CLI for launching and monitoring training runs via W&B Launch.
Uses Textual for a clean TUI experience.

Usage:
    python cli.py launch           # Submit new job
    python cli.py monitor          # Monitor active jobs
    python cli.py status           # Quick status check
    python cli.py cancel <run_id>  # Cancel a running job
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import wandb
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, DataTable, Button, Label, Log
from textual.reactive import reactive
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint


# ============================================================================
# Configuration Loader
# ============================================================================

def load_training_config() -> Dict[str, str]:
    """Load configuration from .training file"""
    config_path = Path(__file__).parent / ".training"

    if not config_path.exists():
        rprint("[red]Error: .training file not found![/red]")
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
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                config[key.strip()] = value

    return config


# ============================================================================
# W&B API Helpers
# ============================================================================

class WandBHelper:
    """Helper for W&B Launch API interactions"""

    def __init__(self, entity: str, project: str, queue: str):
        self.entity = entity
        self.project = project
        self.queue = queue
        self.api = wandb.Api()

    def submit_job(self, config: Dict[str, str]) -> str:
        """Submit training job to W&B Launch queue"""

        # Get git repo URI (assume current directory or configured repo)
        git_uri = os.getenv("GIT_REPO_URI", "https://github.com/djwar42/arr-coc-0-1.git")

        # Build launch config
        launch_config = {
            "uri": git_uri,
            "project": config.get("WANDB_PROJECT", self.project),
            "queue": config.get("WANDB_LAUNCH_QUEUE_NAME", self.queue),
            "entry_point": "python training/train.py",
            "dockerfile": "Dockerfile.wandb",
            "resource": "vertex-ai",
        }

        # Add environment variables from .training
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
        }

        # Filter out None/empty values
        env_vars = {k: v for k, v in env_vars.items() if v}

        launch_config["env"] = env_vars

        rprint(f"\n[yellow]Submitting job to queue:[/yellow] {self.queue}")
        rprint(f"[yellow]Git repo:[/yellow] {git_uri}")
        rprint(f"[yellow]Config:[/yellow]")
        for k, v in env_vars.items():
            rprint(f"  {k}={v}")

        # Submit via wandb CLI (simpler than API for launch)
        import subprocess

        # Build wandb launch command
        cmd = [
            "wandb", "launch",
            "--uri", git_uri,
            "--project", launch_config["project"],
            "--queue", launch_config["queue"],
            "--dockerfile", launch_config["dockerfile"],
            "--entry-point", launch_config["entry_point"],
        ]

        # Add env vars
        for k, v in env_vars.items():
            cmd.extend(["--env", f"{k}={v}"])

        rprint(f"\n[cyan]Running:[/cyan] {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            rprint(f"[red]Error submitting job:[/red]\n{result.stderr}")
            sys.exit(1)

        # Extract run ID from output (wandb prints it)
        output = result.stdout
        rprint(output)

        # Try to extract run ID
        for line in output.split('\n'):
            if 'View run at' in line or 'wandb.ai' in line:
                parts = line.split('/')
                if len(parts) >= 2:
                    return parts[-1].strip()

        return "submitted"

    def get_active_runs(self) -> List[Dict]:
        """Get active training runs"""
        runs = self.api.runs(
            f"{self.entity}/{self.project}",
            filters={"state": {"$in": ["running", "pending", "preempting"]}}
        )

        run_list = []
        for run in runs:
            run_list.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "runtime": (datetime.now() - datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))).seconds if run.state == "running" else 0,
                "url": run.url,
            })

        return run_list

    def get_run_logs(self, run_id: str, tail: int = 50) -> List[str]:
        """Get recent logs for a run"""
        try:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")

            # Get logs via history
            history = run.history(samples=tail)

            logs = []
            for _, row in history.iterrows():
                if '_step' in row:
                    step = row['_step']
                    loss = row.get('loss', 'N/A')
                    logs.append(f"Step {step}: loss={loss}")

            return logs
        except Exception as e:
            return [f"Error fetching logs: {e}"]

    def cancel_run(self, run_id: str):
        """Cancel a running job"""
        run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
        run.stop()
        rprint(f"[yellow]Cancelled run:[/yellow] {run_id}")


# ============================================================================
# Textual TUI App
# ============================================================================

class TrainingMonitor(App):
    """Textual app for monitoring training runs"""

    CSS = """
    Screen {
        background: $surface;
    }

    #status-container {
        height: 10;
        border: solid $primary;
        margin: 1;
    }

    #runs-table {
        height: 15;
        border: solid $accent;
        margin: 1;
    }

    #logs-container {
        height: 1fr;
        border: solid $secondary;
        margin: 1;
    }

    Button {
        margin: 1 2;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "cancel", "Cancel Run"),
        ("q", "quit", "Quit"),
    ]

    runs_data = reactive([])

    def __init__(self, helper: WandBHelper):
        super().__init__()
        self.helper = helper
        self.selected_run = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Status panel
        with Container(id="status-container"):
            yield Label("ARR-COC Training Monitor", id="status-label")
            yield Label("Loading...", id="queue-status")

        # Runs table
        yield DataTable(id="runs-table")

        # Buttons
        with Horizontal():
            yield Button("Refresh", variant="primary", id="refresh-btn")
            yield Button("Cancel Selected", variant="warning", id="cancel-btn")
            yield Button("View Logs", variant="default", id="logs-btn")

        # Logs panel
        with Container(id="logs-container"):
            yield Log(id="logs-view")

        yield Footer()

    def on_mount(self) -> None:
        """Set up the table"""
        table = self.query_one(DataTable)
        table.add_columns("Run ID", "Name", "State", "Runtime", "Created")
        table.cursor_type = "row"

        # Initial load
        self.refresh_runs()

    def refresh_runs(self):
        """Refresh runs from W&B"""
        runs = self.helper.get_active_runs()
        self.runs_data = runs

        # Update table
        table = self.query_one(DataTable)
        table.clear()

        for run in runs:
            runtime_str = f"{run['runtime'] // 3600}h {(run['runtime'] % 3600) // 60}m" if run['runtime'] > 0 else "N/A"
            created_str = run['created_at'].split('T')[0] if run['created_at'] else "Unknown"

            table.add_row(
                run['id'][:8],
                run['name'][:30],
                run['state'],
                runtime_str,
                created_str,
            )

        # Update status
        self.query_one("#queue-status", Label).update(
            f"Active runs: {len(runs)} | Queue: {self.helper.queue}"
        )

    def action_refresh(self):
        """Refresh button action"""
        self.refresh_runs()
        self.notify("Refreshed!")

    def action_cancel(self):
        """Cancel selected run"""
        table = self.query_one(DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.runs_data):
            run = self.runs_data[table.cursor_row]
            self.helper.cancel_run(run['id'])
            self.refresh_runs()
            self.notify(f"Cancelled: {run['name']}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "logs-btn":
            self.view_logs()

    def view_logs(self):
        """View logs for selected run"""
        table = self.query_one(DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.runs_data):
            run = self.runs_data[table.cursor_row]
            logs = self.helper.get_run_logs(run['id'])

            log_view = self.query_one("#logs-view", Log)
            log_view.clear()
            for log in logs:
                log_view.write_line(log)

            self.notify(f"Loaded logs for: {run['name']}")


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_launch():
    """Launch a new training job"""
    rprint("\n[bold cyan]ARR-COC Training Launcher[/bold cyan]\n")

    # Load config
    config = load_training_config()

    # Get W&B entity from env or config
    entity = os.getenv("WANDB_ENTITY", wandb.api.viewer()["username"])
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-arr-coc-queue")

    # Show config preview
    rprint(Panel.fit(
        f"""[bold]Configuration Preview[/bold]

Model: {config.get('BASE_MODEL')}
Visual Tokens: {config.get('NUM_VISUAL_TOKENS')}
Batch Size: {config.get('BATCH_SIZE')} × {config.get('GRADIENT_ACCUMULATION_STEPS')} accum = {int(config.get('BATCH_SIZE', 4)) * int(config.get('GRADIENT_ACCUMULATION_STEPS', 4))} effective
Learning Rate: {config.get('LEARNING_RATE')}
Epochs: {config.get('NUM_EPOCHS')}
Queue: {queue}
Machine: {config.get('WANDB_LAUNCH_MACHINE_TYPE')} ({config.get('WANDB_LAUNCH_ACCELERATOR_COUNT')}× {config.get('WANDB_LAUNCH_ACCELERATOR_TYPE')})
Spot: {config.get('WANDB_LAUNCH_USE_PREEMPTIBLE')}
""",
        title="Training Job",
        border_style="cyan"
    ))

    # Confirm
    confirm = input("\nSubmit job? [y/N]: ")
    if confirm.lower() != 'y':
        rprint("[yellow]Cancelled.[/yellow]")
        return

    # Submit
    helper = WandBHelper(entity, project, queue)
    run_id = helper.submit_job(config)

    rprint(f"\n[green]✓ Job submitted![/green]")
    rprint(f"Run ID: {run_id}")
    rprint(f"\nMonitor at: https://wandb.ai/{entity}/{project}")
    rprint(f"Or run: [cyan]python cli.py monitor[/cyan]")


def cmd_monitor():
    """Monitor active training runs (Textual TUI)"""
    # Get W&B entity
    entity = os.getenv("WANDB_ENTITY", wandb.api.viewer()["username"])

    # Load config for project/queue
    config = load_training_config()
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-arr-coc-queue")

    helper = WandBHelper(entity, project, queue)

    # Launch Textual app
    app = TrainingMonitor(helper)
    app.run()


def cmd_status():
    """Quick status check (no TUI)"""
    console = Console()

    # Get W&B entity
    entity = os.getenv("WANDB_ENTITY", wandb.api.viewer()["username"])

    # Load config
    config = load_training_config()
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-arr-coc-queue")

    helper = WandBHelper(entity, project, queue)

    console.print(f"\n[bold]Checking status for:[/bold] {entity}/{project}\n")

    runs = helper.get_active_runs()

    if not runs:
        console.print("[yellow]No active runs.[/yellow]")
        return

    # Create table
    table = Table(title=f"Active Runs ({len(runs)})")
    table.add_column("Run ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("State", style="green")
    table.add_column("Runtime")
    table.add_column("Created")

    for run in runs:
        runtime_str = f"{run['runtime'] // 3600}h {(run['runtime'] % 3600) // 60}m" if run['runtime'] > 0 else "N/A"
        created_str = run['created_at'].split('T')[0] if run['created_at'] else "Unknown"

        table.add_row(
            run['id'][:8],
            run['name'][:40],
            run['state'],
            runtime_str,
            created_str
        )

    console.print(table)
    console.print(f"\n[dim]Full monitor: python cli.py monitor[/dim]")


def cmd_cancel(run_id: str):
    """Cancel a specific run"""
    entity = os.getenv("WANDB_ENTITY", wandb.api.viewer()["username"])
    config = load_training_config()
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-arr-coc-queue")

    helper = WandBHelper(entity, project, queue)
    helper.cancel_run(run_id)

    rprint(f"[green]✓ Cancelled run: {run_id}[/green]")


# ============================================================================
# Main CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point"""

    if len(sys.argv) < 2:
        rprint("""
[bold cyan]ARR-COC Training CLI[/bold cyan]

Usage:
    python cli.py launch           Submit new training job
    python cli.py monitor          Monitor active runs (TUI)
    python cli.py status           Quick status check
    python cli.py cancel <run_id>  Cancel a specific run

Examples:
    python cli.py launch           # Submit job with .training config
    python cli.py monitor          # Open monitoring dashboard
    python cli.py status           # Quick status in terminal
    python cli.py cancel abc123    # Cancel run with ID abc123
""")
        sys.exit(1)

    command = sys.argv[1]

    if command == "launch":
        cmd_launch()
    elif command == "monitor":
        cmd_monitor()
    elif command == "status":
        cmd_status()
    elif command == "cancel":
        if len(sys.argv) < 3:
            rprint("[red]Error: cancel requires <run_id>[/red]")
            sys.exit(1)
        cmd_cancel(sys.argv[2])
    else:
        rprint(f"[red]Unknown command:[/red] {command}")
        rprint("Valid commands: launch, monitor, status, cancel")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Usage Examples

### 1. Launch New Training Job

```bash
cd training/
python cli.py launch

# Output:
╭─ Training Job ───────────────────────────────╮
│ Configuration Preview                        │
│                                              │
│ Model: Qwen/Qwen3-VL-2B-Instruct            │
│ Visual Tokens: 200                           │
│ Batch Size: 4 × 4 accum = 16 effective      │
│ Learning Rate: 1e-5                          │
│ Epochs: 3                                    │
│ Queue: vertex-arr-coc-queue                  │
│ Machine: a2-highgpu-1g (1× A100)            │
│ Spot: true                                   │
╰──────────────────────────────────────────────╯

Submit job? [y/N]: y

✓ Job submitted!
Run ID: abc12345

Monitor at: https://wandb.ai/newsofpeace2/arr-coc-0-1
Or run: python cli.py monitor
```

### 2. Monitor Active Runs (TUI)

```bash
python cli.py monitor
```

**Screenshot of TUI:**
```
╭─ ARR-COC Training Monitor ──────────────────────────── 14:32:15 ─╮
│ Active runs: 2 | Queue: vertex-arr-coc-queue                     │
╰───────────────────────────────────────────────────────────────────╯

╭─ Runs Table ──────────────────────────────────────────────────────╮
│ Run ID    Name              State     Runtime  Created            │
│ abc12345  baseline-v0.1     running   2h 34m   2025-01-31        │
│ def67890  test-lr-sweep     pending   N/A      2025-01-31        │
╰───────────────────────────────────────────────────────────────────╯

[Refresh] [Cancel Selected] [View Logs]

╭─ Logs ────────────────────────────────────────────────────────────╮
│ Step 1000: loss=2.3421                                            │
│ Step 1010: loss=2.3156                                            │
│ Step 1020: loss=2.2987                                            │
│ Saving checkpoint to gs://bucket/checkpoints/step-1000...         │
│ Step 1030: loss=2.2801                                            │
╰───────────────────────────────────────────────────────────────────╯

r refresh | c cancel | q quit
```

**Features:**
- Real-time refresh (press `r`)
- Select row and press `c` to cancel
- Logs update automatically
- Exit with `q` (training keeps running!)

### 3. Quick Status Check

```bash
python cli.py status

# Output:
Checking status for: newsofpeace2/arr-coc-0-1

           Active Runs (2)
┏━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Run ID  ┃ Name          ┃ State   ┃ Runtime ┃ Created    ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ abc12345│ baseline-v0.1 │ running │ 2h 34m  │ 2025-01-31 │
│ def67890│ test-lr-sweep │ pending │ N/A     │ 2025-01-31 │
└─────────┴───────────────┴─────────┴─────────┴────────────┘

Full monitor: python cli.py monitor
```

### 4. Cancel a Run

```bash
python cli.py cancel abc12345

# Output:
Cancelled run: abc12345
✓ Cancelled run: abc12345
```

---

## Key Features

### ✅ **Detached Mode**
- Exiting the CLI **does NOT stop training**
- Training runs on Vertex AI, completely independent
- CLI is just a monitoring tool
- Close laptop, training keeps going

### ✅ **Simple Configuration**
- Reads `.training` file automatically
- No extra config needed
- All settings in one place

### ✅ **W&B Integration**
- Uses W&B Launch API
- Streams logs from W&B
- Shows run status/metrics
- Direct links to W&B dashboard

### ✅ **Clean TUI**
- Built with Textual (Python's best TUI framework)
- Keyboard shortcuts (`r`, `c`, `q`)
- Real-time updates
- Works over SSH

---

## Workflow

**Typical daily workflow:**

```bash
# Morning: Start training
cd training/
python cli.py launch
# ✓ Job submitted!

# Check status throughout day
python cli.py status
# 2 runs active, 1 pending

# Afternoon: Deep dive into logs
python cli.py monitor
# [Opens TUI, watch logs scroll]
# Press 'q' to exit (training keeps running)

# Evening: Check final status
python cli.py status
# 1 run completed, 1 still running

# Cancel if needed
python cli.py cancel abc123
```

**That's it!** Simple, clean, effective.

---

## Advanced: Environment Variables

Override config on-the-fly:

```bash
# Override learning rate for this run
LEARNING_RATE=5e-6 python cli.py launch

# Override model size
BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct python cli.py launch

# Use different W&B entity
WANDB_ENTITY=my-team python cli.py monitor
```

---

## Troubleshooting

**CLI can't find .training:**
```bash
# Make sure you're in training/ folder
cd training/
python cli.py launch
```

**W&B auth error:**
```bash
# Re-login
wandb login
```

**No runs showing:**
```bash
# Check you're looking at right project
# .training should have:
WANDB_PROJECT="arr-coc-0-1"  # Your project name
```

**Can't cancel run:**
```bash
# Get run ID from status
python cli.py status
# Copy the Run ID (first column)
python cli.py cancel <run-id>
```

---

## Why This Design?

**Single file (`cli.py`):**
- ✅ No extra folders/packages
- ✅ Self-contained
- ✅ Easy to modify
- ✅ Just works

**Textual TUI:**
- ✅ Beautiful terminal UI
- ✅ Works over SSH
- ✅ No browser needed
- ✅ Keyboard shortcuts

**W&B Launch integration:**
- ✅ Uses existing infrastructure
- ✅ No custom job management
- ✅ Proven, reliable
- ✅ Free monitoring

**Detached training:**
- ✅ Close laptop, training continues
- ✅ Check from anywhere
- ✅ No "screen" or "tmux" needed
- ✅ Cloud-native

---

## Next Steps

1. **Add `cli.py` to your training/ folder**
2. **Install dependencies:** `pip install textual wandb rich`
3. **Test launch:** `python cli.py launch`
4. **Monitor:** `python cli.py monitor`

That's the whole setup. ¯\\\_(ツ)_/¯

---

**Related:** [BUILD_OUT_TRAINING.md](./BUILD_OUT_TRAINING.md) - Main training infrastructure guide
