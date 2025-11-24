"""CloudBuild QUEUED Timeout Monitor

Monitors CloudBuild for 45-minute QUEUED timeout.
If timeout occurs, marks MECHA as fatigued for 24 hours!
"""

import subprocess
import threading
import time
from typing import Callable, Dict, Optional

# Import formal reason code for Queue Godzilla
from .mecha.campaign_stats import REASON_QUEUE_TIMEOUT

# 45-minute timeout for QUEUED state → 24 hours fatigue!
QUEUE_TIMEOUT_SECONDS = 45 * 60  # 2700 seconds


class BuildQueueMonitor:
    """Monitor CloudBuild for QUEUED timeout"""

    def __init__(
        self,
        build_id: str,
        region: str,
        project_id: str,
        fatigue_callback: Callable[
            [str, str, str, str, str], None
        ],  # region, reason, reason_code, error_msg, build_id
        status_callback: Callable[[str], None],
    ):
        self.build_id = build_id
        self.region = region
        self.project_id = project_id
        self.fatigue_callback = fatigue_callback
        self.status_callback = status_callback

        self.start_time = time.time()
        self.stopped = False
        self.timed_out = False
        self.thread = None

    def get_build_status(self) -> Optional[str]:
        """Get current build status (QUEUED, WORKING, SUCCESS, FAILURE)"""
        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "builds",
                    "describe",
                    self.build_id,
                    f"--region={self.region}",
                    f"--project={self.project_id}",
                    "--format=value(status)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def monitor_loop(self):
        """Background monitoring loop"""
        while not self.stopped:
            elapsed = time.time() - self.start_time
            status = self.get_build_status()

            # TIMEOUT CHECK: 45 minutes in QUEUED = MECHA FATIGUED FOR 24 HOURS!
            if status == "QUEUED" and elapsed > QUEUE_TIMEOUT_SECONDS:
                self.status_callback("")
                self.status_callback("")
                self.status_callback(
                    "[yellow]═══════════════════════════════════════════════════════════[/yellow]"
                )
                self.status_callback(
                    "[yellow]🌲 MECHA DEFEATED BY QUEUE GODZILLA! 🌲[/yellow]"
                )
                self.status_callback(
                    "[yellow]═══════════════════════════════════════════════════════════[/yellow]"
                )
                self.status_callback("")
                self.status_callback(
                    f"[yellow]   Region: {self.region} stayed in QUEUED for {int(elapsed / 60)} minutes![/yellow]"
                )
                self.status_callback(
                    "[yellow]   The QUEUE GODZILLA has exhausted this MECHA![/yellow]"
                )
                self.status_callback(
                    "[yellow]   MECHA FATIGUED - OUT FOR 24 HOURS! 🛌[/yellow]"
                )
                self.status_callback("")
                self.status_callback(
                    "[dim]   (Technical: Worker pool failed to spin up after 45 minutes.[/dim]"
                )
                self.status_callback(
                    "[dim]    This triggers automatic 24-hour timeout on this region's pool.[/dim]"
                )
                self.status_callback(
                    "[dim]    After 24 hours, pool will auto-recover and rejoin battles.)[/dim]"
                )
                self.status_callback("")
                self.status_callback(
                    "[yellow]🛑 LAUNCH HALTED - MECHA needs rest![/yellow]"
                )
                self.status_callback("")

                # Mark MECHA as fatigued!
                # Pass build_id, reason, reason_code, and error message for Godzilla incident log
                reason = f"Queue timeout - 45 minutes in QUEUED state"
                reason_code = REASON_QUEUE_TIMEOUT  # Formal machine-readable code
                error_msg = f"Worker pool failed to spin up after 45 minutes. Build {self.build_id} stuck in QUEUED state."
                self.fatigue_callback(self.region, reason, reason_code, error_msg, self.build_id)
                self.timed_out = True
                self.stopped = True

                # Cancel the build
                self._cancel_build()
                return

            # If build is WORKING, it's no longer queued - stop monitoring!
            if status and status != "QUEUED":
                if status == "WORKING":
                    # CELEBRATION! MECHA awakened!
                    elapsed_mins = int((time.time() - self.start_time) / 60)
                    elapsed_secs = int((time.time() - self.start_time) % 60)
                    self.status_callback("")
                    self.status_callback(
                        f"[green]🎉 MECHA AWAKENED! Build started after {elapsed_mins}m {elapsed_secs}s[/green]"
                    )
                    self.status_callback("")
                    self.status_callback(
                        f"[yellow]⏳ This can take some serious time (~30 min on 176-vCPU, 6+ hours on 4-vCPU)...[/yellow]"
                    )
                    self.status_callback(
                        f"[cyan]→ Watch build progress:[/cyan] https://console.cloud.google.com/cloud-build/builds;region={self.region}/{self.build_id}?project={self.project_id}"
                    )
                    self.status_callback("")
                self.stopped = True
                return

            # Poll every 10 seconds
            time.sleep(10)

    def _cancel_build(self):
        """Cancel the stuck build"""
        try:
            self.status_callback("[dim]→ Cancelling stuck build...[/dim]")
            subprocess.run(
                [
                    "gcloud",
                    "builds",
                    "cancel",
                    self.build_id,
                    f"--region={self.region}",
                    f"--project={self.project_id}",
                    "--quiet",
                ],
                capture_output=True,
                timeout=30,
            )
            self.status_callback("[dim]   Build cancelled![/dim]")
        except Exception as e:
            self.status_callback(
                f"[dim]   Warning: Could not cancel build: {str(e)[:100]}[/dim]"
            )

    def start(self):
        """Start monitoring in background thread"""
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=5)

    def did_timeout(self) -> bool:
        """Check if timeout occurred"""
        return self.timed_out


def extract_build_id_from_output(output_line: str) -> Optional[str]:
    """
    Extract build ID from gcloud output.

    Example line:
    "Created [https://cloudbuild.googleapis.com/.../builds/39164f58-968b-4bd5-9c7e-6b6a7abc8e15]."
    """
    if "Created [" in output_line and "/builds/" in output_line:
        try:
            # Extract between /builds/ and ]
            start = output_line.find("/builds/") + 8
            end = output_line.find("]", start)
            return output_line[start:end].strip(".")
        except Exception:
            return None
    return None
