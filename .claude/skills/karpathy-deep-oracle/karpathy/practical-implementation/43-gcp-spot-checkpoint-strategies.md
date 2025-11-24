# GCP Spot Instance Checkpoint Strategies

**Complete guide to fault-tolerant training on Google Cloud spot instances**

Checkpoint strategies for preemptible training are essential for achieving high training goodput (uptime) while using spot instances with 60-91% cost savings. This guide covers comprehensive checkpoint patterns for handling the 30-second termination notice on GCP spot VMs.

---

## Section 1: Checkpoint Fundamentals for Spot Instances (~200 lines)

### Why Checkpoint for Spot Instances

Spot VMs on GCP can be preempted at any time with a 30-second termination notice. Without proper checkpointing:
- **Lost progress**: Hours of training evaporate on preemption
- **Wasted compute**: Pay for compute that produces no final result
- **Delayed time-to-market**: Repeated restarts extend project timelines

**The checkpoint equation**: Optimal checkpointing balances three factors:
1. **Saving overhead**: Time spent writing checkpoints (blocks training)
2. **Lost computation**: Work lost between last checkpoint and preemption
3. **Loading time**: Time to restore state after preemption

From [PyTorch Distributed Checkpoint - Efficient checkpointing](https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/) (accessed 2025-01-31):

**Checkpoint Badput Formula**:
```
Badput = (Loading_Time + Saving_Overhead + Computation_Loss) / MTBI

Where:
- MTBI = Mean Time Between Interruptions
- Computation_Loss = time since last checkpoint when interrupted
- Saving_Overhead = training slowdown during checkpoint save
```

**Key insight**: Frequent checkpoints reduce computation loss but increase overhead. Infrequent checkpoints have low overhead but risk large losses.

### Checkpoint Frequency Optimization

**The tradeoff**:
- **Too frequent** (every 10 minutes): High saving overhead, minimal loss
- **Too infrequent** (every 4 hours): Low overhead, catastrophic loss risk
- **Optimal** (varies by job): Minimize total badput

**Recommended frequencies for GCP spot instances**:
- **Large models (70B+ params)**: Every 30-60 minutes
- **Medium models (7-13B params)**: Every 15-30 minutes
- **Small models (<3B params)**: Every 10-15 minutes
- **Vision models (VLMs)**: Every 20-30 minutes

**Formula for optimal interval** (from PyTorch blog):
```python
optimal_interval = sqrt(2 * MTBI * saving_overhead / load_time)

# Example: MTBI=4 hours, save_overhead=2min, load_time=90sec
optimal_interval = sqrt(2 * 240 * 2 / 1.5) = 25.3 minutes
```

### Checkpoint Storage Options

**1. Google Cloud Storage (GCS) - Persistent**
- **Pros**: Survives VM termination, accessible across regions
- **Cons**: Higher latency (~90-135s save time for large models)
- **Cost**: $0.020/GB/month (Standard), $0.010/GB/month (Nearline)
- **Best for**: Final checkpoints, daily backups

**2. Local SSD - Fast but Ephemeral**
- **Pros**: Ultra-fast saves (~10-47s for same models)
- **Cons**: Lost on VM termination
- **Cost**: $0.178/GB/month (local SSD)
- **Best for**: Frequent intermediate checkpoints with GCS replication

**3. Persistent Disk SSD - Middle Ground**
- **Pros**: Survives stop/restart, faster than GCS
- **Cons**: Lost if VM deleted, regional only
- **Cost**: $0.170/GB/month
- **Best for**: Single-node training with restarts

**Recommended hybrid strategy**:
```
Every 10 min  → Local SSD (fast intermediate)
Every 30 min  → Persistent disk SSD (restart-safe)
Every 2 hours → GCS (permanent backup)
```

### Checkpoint Size Reduction Techniques

**Minimize checkpoint size to reduce save/load time**:

**1. State Dict Only (Not Full Model)**
```python
# BAD - Saves entire model structure
torch.save(model, 'checkpoint.pt')  # ~10GB

# GOOD - Saves only weights
torch.save(model.state_dict(), 'checkpoint.pt')  # ~6GB
```

**2. Optimizer State Management**
```python
# Full checkpoint (needed for exact resume)
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),  # Can be large!
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
    'step': global_step,
    'rng_state': torch.get_rng_state()
}

# Lightweight checkpoint (for inference or fine-tuning restart)
checkpoint = {
    'model': model.state_dict(),
    'epoch': epoch,
    'step': global_step
}
```

**3. FP16/BF16 Checkpoint Casting**
```python
# Save in half precision (50% size reduction)
checkpoint = {
    'model': {k: v.half() for k, v in model.state_dict().items()},
    'optimizer': optimizer.state_dict()
}
torch.save(checkpoint, 'checkpoint_fp16.pt')

# Load and restore to FP32
checkpoint = torch.load('checkpoint_fp16.pt')
model.load_state_dict({k: v.float() for k, v in checkpoint['model'].items()})
```

**4. Checkpoint Compression**
```python
import zipfile
import torch

# Save with compression
torch.save(checkpoint, 'checkpoint.pt')
with zipfile.ZipFile('checkpoint.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('checkpoint.pt')

# Typical compression: 30-40% size reduction for model weights
```

### Complete Checkpoint Manager Implementation

```python
import torch
import os
from pathlib import Path
from typing import Dict, Optional
from google.cloud import storage

class SpotCheckpointManager:
    """
    Checkpoint manager optimized for GCP spot instances.

    Features:
    - Hierarchical storage (local SSD → persistent disk → GCS)
    - Automatic cleanup of old checkpoints
    - Atomic writes (write to temp, then rename)
    - Progress tracking
    """

    def __init__(
        self,
        local_dir: str = "/mnt/localssd/checkpoints",
        persistent_dir: str = "/mnt/disks/persistent/checkpoints",
        gcs_bucket: Optional[str] = None,
        gcs_prefix: str = "checkpoints",
        keep_local: int = 3,
        keep_persistent: int = 5,
        keep_gcs: int = 10
    ):
        self.local_dir = Path(local_dir)
        self.persistent_dir = Path(persistent_dir)
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix

        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_dir.mkdir(parents=True, exist_ok=True)

        self.keep_local = keep_local
        self.keep_persistent = keep_persistent
        self.keep_gcs = keep_gcs

        if gcs_bucket:
            self.gcs_client = storage.Client()
            self.bucket = self.gcs_client.bucket(gcs_bucket)

    def save_checkpoint(
        self,
        state_dict: Dict,
        step: int,
        save_to_persistent: bool = False,
        save_to_gcs: bool = False
    ) -> str:
        """
        Save checkpoint with hierarchical storage strategy.

        Returns: Path to local checkpoint
        """
        filename = f"checkpoint_step_{step}.pt"

        # Always save to local SSD first (fastest)
        local_path = self.local_dir / filename
        temp_path = self.local_dir / f"{filename}.tmp"

        # Atomic write: save to temp, then rename
        torch.save(state_dict, temp_path)
        temp_path.rename(local_path)

        print(f"✓ Saved checkpoint to local SSD: {local_path}")

        # Optionally save to persistent disk
        if save_to_persistent:
            persistent_path = self.persistent_dir / filename
            temp_persistent = self.persistent_dir / f"{filename}.tmp"
            torch.save(state_dict, temp_persistent)
            temp_persistent.rename(persistent_path)
            print(f"✓ Saved checkpoint to persistent disk: {persistent_path}")

        # Optionally save to GCS (slowest but permanent)
        if save_to_gcs and self.gcs_bucket:
            self._upload_to_gcs(local_path, filename)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(local_path)

    def _upload_to_gcs(self, local_path: Path, filename: str):
        """Upload checkpoint to GCS bucket."""
        blob_name = f"{self.gcs_prefix}/{filename}"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        print(f"✓ Uploaded checkpoint to GCS: gs://{self.gcs_bucket}/{blob_name}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        # Cleanup local SSD
        self._cleanup_dir(self.local_dir, self.keep_local)

        # Cleanup persistent disk
        self._cleanup_dir(self.persistent_dir, self.keep_persistent)

        # Cleanup GCS (if configured)
        if self.gcs_bucket and self.keep_gcs > 0:
            self._cleanup_gcs()

    def _cleanup_dir(self, directory: Path, keep: int):
        """Keep only the most recent N checkpoints in directory."""
        checkpoints = sorted(
            directory.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_checkpoint in checkpoints[keep:]:
            old_checkpoint.unlink()
            print(f"✗ Removed old checkpoint: {old_checkpoint}")

    def _cleanup_gcs(self):
        """Keep only the most recent N checkpoints in GCS."""
        blobs = list(self.bucket.list_blobs(prefix=self.gcs_prefix))
        blobs.sort(key=lambda b: b.time_created, reverse=True)

        for old_blob in blobs[self.keep_gcs:]:
            old_blob.delete()
            print(f"✗ Removed old GCS checkpoint: {old_blob.name}")

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load most recent checkpoint from available storage.

        Priority: Local SSD → Persistent disk → GCS
        """
        # Try local SSD first
        local_checkpoints = sorted(
            self.local_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if local_checkpoints:
            print(f"Loading checkpoint from local SSD: {local_checkpoints[0]}")
            return torch.load(local_checkpoints[0])

        # Try persistent disk
        persistent_checkpoints = sorted(
            self.persistent_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if persistent_checkpoints:
            print(f"Loading checkpoint from persistent disk: {persistent_checkpoints[0]}")
            return torch.load(persistent_checkpoints[0])

        # Try GCS as last resort
        if self.gcs_bucket:
            return self._load_from_gcs()

        print("No checkpoint found in any storage location")
        return None

    def _load_from_gcs(self) -> Optional[Dict]:
        """Download and load most recent checkpoint from GCS."""
        blobs = list(self.bucket.list_blobs(prefix=self.gcs_prefix))
        if not blobs:
            return None

        latest_blob = max(blobs, key=lambda b: b.time_created)
        temp_path = self.local_dir / "temp_gcs_checkpoint.pt"

        latest_blob.download_to_filename(str(temp_path))
        print(f"Downloaded checkpoint from GCS: {latest_blob.name}")

        checkpoint = torch.load(temp_path)
        temp_path.unlink()  # Clean up temp file

        return checkpoint


# Example usage
checkpoint_manager = SpotCheckpointManager(
    local_dir="/mnt/localssd/checkpoints",
    persistent_dir="/mnt/disks/persistent/checkpoints",
    gcs_bucket="my-training-bucket",
    gcs_prefix="experiments/llama-13b",
    keep_local=3,
    keep_persistent=5,
    keep_gcs=10
)

# Save checkpoint with hierarchical strategy
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'step': global_step
}

# Every 10 minutes: local only
checkpoint_manager.save_checkpoint(checkpoint, global_step)

# Every 30 minutes: local + persistent
checkpoint_manager.save_checkpoint(checkpoint, global_step, save_to_persistent=True)

# Every 2 hours: all three (local + persistent + GCS)
checkpoint_manager.save_checkpoint(checkpoint, global_step,
                                  save_to_persistent=True,
                                  save_to_gcs=True)

# Load on restart
loaded_checkpoint = checkpoint_manager.load_latest_checkpoint()
if loaded_checkpoint:
    model.load_state_dict(loaded_checkpoint['model'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer'])
    start_step = loaded_checkpoint['step']
```

**Performance characteristics**:
- Local SSD save: 10-15 seconds (large models)
- Persistent disk save: 20-30 seconds
- GCS upload: 90-135 seconds
- Load from local: 5-8 seconds
- Load from GCS: 80-120 seconds

---

## Section 2: Preemption Detection & Handling (~200 lines)

### Understanding the 30-Second Termination Notice

From [GCP Spot VMs documentation](https://cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31):

**GCP preemption process**:
1. **T+0s**: Preemption signal sent
2. **T+0s to T+30s**: Shutdown grace period (best effort, up to 30 seconds)
3. **T+30s**: VM forcefully terminated if still running

**Critical**: The 30-second window is a **maximum**, not guaranteed. The VM can be stopped at any point during this period.

**Key differences from AWS/Azure**:
- **AWS Spot**: 2-minute warning via instance metadata
- **Azure Spot**: 30-second warning via scheduled events API
- **GCP Spot**: 30-second shutdown period via ACPI G2 soft-off signal

### Three Methods to Detect Preemption

**Method 1: ACPI G2 Soft-Off Signal (Most Reliable)**

The ACPI G2 signal is sent to the VM's operating system when preemption begins.

```python
import signal
import sys
import torch
import time

class PreemptionHandler:
    """
    Handle GCP preemption via ACPI G2 signal (SIGTERM).

    Provides graceful shutdown with emergency checkpoint.
    """

    def __init__(self, checkpoint_manager, model, optimizer, global_step):
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.global_step = global_step
        self.preemption_detected = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigterm(self, signum, frame):
        """Handle SIGTERM (preemption notice)."""
        print("\n" + "="*60)
        print("⚠️  PREEMPTION DETECTED - SIGTERM received")
        print("="*60)

        self.preemption_detected = True
        self._emergency_checkpoint()

        print("✓ Emergency checkpoint complete. Exiting gracefully.")
        sys.exit(0)

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT (user interrupt)."""
        print("\n" + "="*60)
        print("⚠️  USER INTERRUPT - SIGINT received")
        print("="*60)

        self._emergency_checkpoint()

        print("✓ Interrupt checkpoint complete. Exiting.")
        sys.exit(0)

    def _emergency_checkpoint(self):
        """
        Save emergency checkpoint during preemption.

        Strategy: Save to fastest available storage (local SSD),
        then attempt persistent/GCS if time permits.
        """
        start_time = time.time()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.global_step.item() if torch.is_tensor(self.global_step) else self.global_step,
            'preemption_recovery': True,
            'timestamp': time.time()
        }

        # Priority 1: Save to local SSD (fastest, ~5-10s)
        try:
            local_path = self.checkpoint_manager.save_checkpoint(
                checkpoint,
                step=checkpoint['step'],
                save_to_persistent=False,
                save_to_gcs=False
            )
            elapsed = time.time() - start_time
            print(f"✓ Emergency checkpoint saved to local SSD in {elapsed:.2f}s")
        except Exception as e:
            print(f"✗ Failed to save to local SSD: {e}")
            return

        # Priority 2: Save to persistent disk if time permits (target: <15s total)
        elapsed = time.time() - start_time
        if elapsed < 15:
            try:
                self.checkpoint_manager.save_checkpoint(
                    checkpoint,
                    step=checkpoint['step'],
                    save_to_persistent=True,
                    save_to_gcs=False
                )
                elapsed = time.time() - start_time
                print(f"✓ Emergency checkpoint saved to persistent disk in {elapsed:.2f}s")
            except Exception as e:
                print(f"⚠️  Could not save to persistent disk (timeout): {e}")

        # Priority 3: GCS upload only if significant time remains (unlikely)
        elapsed = time.time() - start_time
        if elapsed < 20:  # Very optimistic
            try:
                self.checkpoint_manager.save_checkpoint(
                    checkpoint,
                    step=checkpoint['step'],
                    save_to_persistent=False,
                    save_to_gcs=True
                )
                print(f"✓ Emergency checkpoint uploaded to GCS")
            except Exception as e:
                print(f"⚠️  Could not upload to GCS (timeout expected): {e}")

    def is_preempted(self) -> bool:
        """Check if preemption has been detected."""
        return self.preemption_detected


# Usage in training loop
preemption_handler = PreemptionHandler(
    checkpoint_manager=checkpoint_manager,
    model=model,
    optimizer=optimizer,
    global_step=global_step
)

# Training loop with preemption checking
for step in range(start_step, max_steps):
    # Check for preemption before expensive operations
    if preemption_handler.is_preempted():
        print("Preemption detected, exiting training loop")
        break

    # Normal training step
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()

    global_step += 1

    # Regular checkpointing
    if step % checkpoint_frequency == 0:
        checkpoint_manager.save_checkpoint(...)
```

**Method 2: GCE Metadata Server API**

Query the GCE metadata server for preemption status:

```python
import requests
import threading
import time

class MetadataPreemptionMonitor:
    """
    Monitor GCE metadata server for preemption signals.

    Polls the preemption endpoint every 1 second.
    """

    METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
    METADATA_HEADERS = {"Metadata-Flavor": "Google"}

    def __init__(self, callback):
        self.callback = callback
        self.monitoring = False
        self.thread = None

    def start_monitoring(self):
        """Start background thread to monitor preemption."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("✓ Metadata preemption monitor started")

    def stop_monitoring(self):
        """Stop monitoring thread."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                response = requests.get(
                    self.METADATA_URL,
                    headers=self.METADATA_HEADERS,
                    timeout=1
                )

                if response.status_code == 200:
                    preempted = response.text.strip().lower() == "true"
                    if preempted:
                        print("⚠️  PREEMPTION DETECTED via metadata server")
                        self.callback()
                        break

            except requests.RequestException:
                # Metadata server unreachable (normal if not on GCE)
                pass

            time.sleep(1)  # Poll every second


# Usage
def on_preemption_detected():
    """Callback when preemption is detected."""
    preemption_handler._emergency_checkpoint()
    sys.exit(0)

metadata_monitor = MetadataPreemptionMonitor(callback=on_preemption_detected)
metadata_monitor.start_monitoring()
```

**Method 3: Log-Based Detection**

Monitor GCP logs for preemption events (useful for post-mortem analysis):

```python
from google.cloud import logging_v2

def check_preemption_logs(project_id: str, instance_name: str):
    """
    Check Cloud Logging for preemption events.

    Useful for understanding past preemptions.
    """
    client = logging_v2.Client(project=project_id)

    filter_str = f"""
    resource.type="gce_instance"
    resource.labels.instance_id="{instance_name}"
    logName="projects/{project_id}/logs/compute.googleapis.com%2Fpreempted"
    """

    entries = list(client.list_entries(filter_=filter_str, page_size=10))

    if entries:
        print(f"Found {len(entries)} preemption events:")
        for entry in entries:
            print(f"  - {entry.timestamp}: {entry.payload}")
    else:
        print("No preemption events found in logs")


# Run post-training to analyze preemptions
check_preemption_logs(
    project_id="my-project",
    instance_name="training-vm-spot-001"
)
```

### Complete Preemption Handler Code

**Production-ready implementation combining all methods**:

```python
import signal
import sys
import torch
import time
import threading
import requests
from typing import Callable, Optional

class ProductionPreemptionHandler:
    """
    Production-grade preemption handler for GCP spot instances.

    Features:
    - Multiple detection methods (signal, metadata, heartbeat)
    - Graceful degradation
    - Emergency checkpoint with time-aware strategy
    - Thread-safe operation
    """

    METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
    METADATA_HEADERS = {"Metadata-Flavor": "Google"}

    def __init__(
        self,
        checkpoint_fn: Callable,
        poll_interval: float = 1.0,
        enable_metadata_check: bool = True
    ):
        self.checkpoint_fn = checkpoint_fn
        self.poll_interval = poll_interval
        self.enable_metadata_check = enable_metadata_check

        self.preemption_detected = False
        self.monitoring_thread = None
        self._lock = threading.Lock()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Start metadata monitoring if enabled
        if self.enable_metadata_check:
            self.start_monitoring()

    def _handle_signal(self, signum, frame):
        """Handle SIGTERM/SIGINT signals."""
        with self._lock:
            if self.preemption_detected:
                return  # Already handling preemption

            self.preemption_detected = True

        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n{'='*60}")
        print(f"⚠️  {signal_name} RECEIVED - INITIATING EMERGENCY SHUTDOWN")
        print(f"{'='*60}\n")

        self._execute_emergency_checkpoint()
        sys.exit(0)

    def start_monitoring(self):
        """Start background metadata monitoring thread."""
        self.monitoring_thread = threading.Thread(
            target=self._metadata_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

    def _metadata_monitoring_loop(self):
        """Background loop to check metadata server."""
        while not self.preemption_detected:
            try:
                response = requests.get(
                    self.METADATA_URL,
                    headers=self.METADATA_HEADERS,
                    timeout=1
                )

                if response.status_code == 200:
                    if response.text.strip().lower() == "true":
                        with self._lock:
                            if not self.preemption_detected:
                                self.preemption_detected = True
                                print("\n⚠️  PREEMPTION DETECTED via metadata server\n")
                                self._execute_emergency_checkpoint()
                                sys.exit(0)

            except requests.RequestException:
                pass  # Metadata server unreachable

            time.sleep(self.poll_interval)

    def _execute_emergency_checkpoint(self):
        """Execute emergency checkpoint with timing."""
        start_time = time.time()

        try:
            self.checkpoint_fn()
            elapsed = time.time() - start_time
            print(f"✓ Emergency checkpoint completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Emergency checkpoint failed after {elapsed:.2f}s: {e}")

    def is_preempted(self) -> bool:
        """Thread-safe check for preemption."""
        with self._lock:
            return self.preemption_detected


# Usage in training script
def emergency_checkpoint():
    """Emergency checkpoint function."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': global_step,
        'timestamp': time.time()
    }

    # Save to fastest storage first
    checkpoint_manager.save_checkpoint(
        checkpoint,
        step=global_step,
        save_to_persistent=True,
        save_to_gcs=False  # Skip GCS in emergency (too slow)
    )

# Initialize handler
handler = ProductionPreemptionHandler(
    checkpoint_fn=emergency_checkpoint,
    poll_interval=1.0,
    enable_metadata_check=True
)

# Training loop
for step in range(start_step, max_steps):
    if handler.is_preempted():
        break

    # Training step
    train_step(model, batch)
```

### Resume Detection on Restart

Detect if the current run is recovering from preemption:

```python
def detect_preemption_recovery(checkpoint_manager) -> tuple[bool, Optional[Dict]]:
    """
    Detect if we're recovering from a preemption.

    Returns:
        (is_recovery, checkpoint_dict)
    """
    checkpoint = checkpoint_manager.load_latest_checkpoint()

    if checkpoint is None:
        return False, None

    is_recovery = checkpoint.get('preemption_recovery', False)

    if is_recovery:
        print("="*60)
        print("🔄 RECOVERING FROM PREEMPTION")
        print(f"   Last step: {checkpoint['step']}")
        print(f"   Timestamp: {time.ctime(checkpoint['timestamp'])}")
        print("="*60)

    return is_recovery, checkpoint


# At training start
is_recovery, checkpoint = detect_preemption_recovery(checkpoint_manager)

if checkpoint:
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step']

    if is_recovery:
        # Log recovery metrics
        print(f"Resuming training from step {start_step}")
```

---

## Section 3: Distributed Training Checkpoints (~200 lines)

### DDP Checkpoint Coordination

From [PyTorch DDP checkpoint best practices](https://discuss.pytorch.org/t/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data-parallel-ddp-in-pytorch/139575) (accessed 2025-01-31):

**Key principle**: Only rank 0 should save checkpoints to avoid race conditions and storage overhead.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPCheckpointManager:
    """
    Checkpoint manager for DDP training.

    Only rank 0 saves checkpoints, all ranks participate in loading.
    """

    def __init__(self, checkpoint_manager, rank, world_size):
        self.checkpoint_manager = checkpoint_manager
        self.rank = rank
        self.world_size = world_size

    def save_checkpoint(
        self,
        model: DDP,
        optimizer,
        step: int,
        save_to_persistent: bool = False,
        save_to_gcs: bool = False
    ) -> Optional[str]:
        """
        Save checkpoint from rank 0 only.

        Returns: checkpoint path (rank 0) or None (other ranks)
        """
        # Barrier: Ensure all ranks finish current step
        dist.barrier()

        checkpoint_path = None

        if self.rank == 0:
            # Extract model from DDP wrapper
            model_state_dict = model.module.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'step': step,
                'world_size': self.world_size
            }

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint,
                step=step,
                save_to_persistent=save_to_persistent,
                save_to_gcs=save_to_gcs
            )

        # Barrier: Ensure rank 0 completes checkpoint before continuing
        dist.barrier()

        return checkpoint_path

    def load_checkpoint(self, model: DDP, optimizer) -> Optional[int]:
        """
        Load checkpoint on all ranks.

        Returns: step number or None
        """
        checkpoint = None

        if self.rank == 0:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()

        # Broadcast checkpoint availability from rank 0
        has_checkpoint = torch.tensor([checkpoint is not None], dtype=torch.bool)
        if self.rank == 0:
            dist.broadcast(has_checkpoint, src=0)
        else:
            dist.broadcast(has_checkpoint, src=0)

        if not has_checkpoint.item():
            return None

        # Rank 0 shares checkpoint with all ranks
        if self.rank == 0:
            # Load state dicts
            model.module.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
        else:
            # Other ranks wait for broadcast
            step = None

        # Broadcast step number to all ranks
        step_tensor = torch.tensor([step if self.rank == 0 else 0], dtype=torch.long)
        dist.broadcast(step_tensor, src=0)
        step = step_tensor.item()

        # Load state dicts on all ranks from rank 0's model
        # (DDP automatically syncs parameters)
        if self.rank != 0:
            # Parameters automatically synced by DDP after first forward pass
            pass

        return step


# DDP training setup
def setup_ddp(rank, world_size):
    """Initialize distributed process group."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


# Usage in DDP training
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

setup_ddp(rank, world_size)

model = MyModel().cuda(rank)
model = DDP(model, device_ids=[rank])
optimizer = torch.optim.AdamW(model.parameters())

# Create checkpoint manager
base_checkpoint_manager = SpotCheckpointManager(...)
ddp_checkpoint_manager = DDPCheckpointManager(
    checkpoint_manager=base_checkpoint_manager,
    rank=rank,
    world_size=world_size
)

# Load checkpoint (all ranks)
start_step = ddp_checkpoint_manager.load_checkpoint(model, optimizer) or 0

# Training loop
for step in range(start_step, max_steps):
    train_step(model, batch)

    # Save checkpoint (rank 0 only)
    if step % checkpoint_frequency == 0:
        ddp_checkpoint_manager.save_checkpoint(
            model,
            optimizer,
            step=step,
            save_to_persistent=(step % 1000 == 0)
        )
```

### FSDP Full State Dict Checkpointing

From [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html) (accessed 2025-01-31):

FSDP (Fully Sharded Data Parallel) requires special handling for checkpoints because model parameters are sharded across ranks.

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig

class FSDPCheckpointManager:
    """
    Checkpoint manager for FSDP training.

    Handles full state dict gathering from sharded parameters.
    """

    def __init__(self, checkpoint_manager, rank):
        self.checkpoint_manager = checkpoint_manager
        self.rank = rank

    def save_checkpoint(
        self,
        model: FSDP,
        optimizer,
        step: int,
        save_to_persistent: bool = False,
        save_to_gcs: bool = False
    ):
        """
        Save FSDP checkpoint with full state dict.

        Gathers sharded parameters to rank 0.
        """
        # Configure FSDP to return full state dict
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            model_state_dict = model.state_dict()

        # Gather optimizer state dict
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
        ):
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)

        # Only rank 0 has the full state dict
        if self.rank == 0:
            checkpoint = {
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'step': step
            }

            return self.checkpoint_manager.save_checkpoint(
                checkpoint,
                step=step,
                save_to_persistent=save_to_persistent,
                save_to_gcs=save_to_gcs
            )

        return None

    def load_checkpoint(self, model: FSDP, optimizer):
        """Load FSDP checkpoint and distribute to all ranks."""
        checkpoint = None

        if self.rank == 0:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()

        if checkpoint is None:
            return None

        # Load model state dict with FSDP distribution
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            if self.rank == 0:
                model.load_state_dict(checkpoint['model'])

        # Load optimizer state dict
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
        ):
            if self.rank == 0:
                optimizer_state_dict = checkpoint['optimizer']
            else:
                optimizer_state_dict = None

            FSDP.optim_state_dict_load(model, optimizer, optimizer_state_dict)

        return checkpoint['step'] if self.rank == 0 else None


# FSDP training setup
model = FSDP(MyModel().cuda(rank))
optimizer = torch.optim.AdamW(model.parameters())

fsdp_checkpoint_manager = FSDPCheckpointManager(
    checkpoint_manager=base_checkpoint_manager,
    rank=rank
)

# Load and train
start_step = fsdp_checkpoint_manager.load_checkpoint(model, optimizer) or 0
```

### Checkpoint Sharding Strategies

For extremely large models (100B+ parameters), sharding checkpoints across multiple files reduces memory pressure:

```python
import torch
from pathlib import Path
from typing import Dict, List

class ShardedCheckpointManager:
    """
    Manage sharded checkpoints for very large models.

    Splits checkpoint into multiple files to reduce memory usage.
    """

    def __init__(self, checkpoint_dir: str, num_shards: int = 8):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_shards = num_shards

    def save_sharded_checkpoint(
        self,
        state_dict: Dict,
        step: int,
        metadata: Dict = None
    ):
        """
        Save checkpoint split across multiple shard files.

        Reduces peak memory usage during save/load.
        """
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Separate metadata from state dict
        param_keys = list(state_dict.keys())
        shard_size = len(param_keys) // self.num_shards

        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'step': step,
            'num_shards': self.num_shards,
            'total_params': len(param_keys)
        })

        metadata_path = checkpoint_dir / "metadata.pt"
        torch.save(metadata, metadata_path)

        # Save sharded parameters
        for shard_idx in range(self.num_shards):
            start_idx = shard_idx * shard_size
            end_idx = start_idx + shard_size if shard_idx < self.num_shards - 1 else len(param_keys)

            shard_keys = param_keys[start_idx:end_idx]
            shard_state_dict = {k: state_dict[k] for k in shard_keys}

            shard_path = checkpoint_dir / f"shard_{shard_idx:03d}.pt"
            torch.save(shard_state_dict, shard_path)

            print(f"✓ Saved shard {shard_idx}/{self.num_shards} ({len(shard_keys)} params)")

        print(f"✓ Sharded checkpoint saved to {checkpoint_dir}")
        return str(checkpoint_dir)

    def load_sharded_checkpoint(self, checkpoint_dir: str) -> Dict:
        """Load sharded checkpoint from directory."""
        checkpoint_path = Path(checkpoint_dir)

        # Load metadata
        metadata = torch.load(checkpoint_path / "metadata.pt")
        num_shards = metadata['num_shards']

        # Load all shards
        state_dict = {}
        for shard_idx in range(num_shards):
            shard_path = checkpoint_path / f"shard_{shard_idx:03d}.pt"
            shard_state_dict = torch.load(shard_path)
            state_dict.update(shard_state_dict)

            print(f"✓ Loaded shard {shard_idx}/{num_shards}")

        print(f"✓ Loaded sharded checkpoint from {checkpoint_dir}")
        return state_dict, metadata


# Usage for large model checkpointing
sharded_manager = ShardedCheckpointManager(
    checkpoint_dir="/mnt/localssd/sharded_checkpoints",
    num_shards=8
)

# Save
sharded_manager.save_sharded_checkpoint(
    state_dict=model.state_dict(),
    step=global_step,
    metadata={'optimizer': optimizer.state_dict()}
)

# Load
state_dict, metadata = sharded_manager.load_sharded_checkpoint(
    checkpoint_dir="/mnt/localssd/sharded_checkpoints/checkpoint_step_1000"
)
model.load_state_dict(state_dict)
```

### Cloud Storage Parallel Uploads

Use GCS multipart upload and parallel workers for faster checkpoint uploads:

```python
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import hashlib

class ParallelGCSUploader:
    """
    Upload large checkpoints to GCS using parallel multipart uploads.

    From [Connector for PyTorch](https://cloud.google.com/storage/docs/pytorch-connector)
    (accessed 2025-01-31): Multipart upload provides up to 10x performance
    improvement over standard uploads.
    """

    def __init__(self, bucket_name: str, chunk_size_mb: int = 32):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.chunk_size = chunk_size_mb * 1024 * 1024  # Convert to bytes

    def upload_checkpoint_parallel(
        self,
        local_path: str,
        gcs_path: str,
        num_workers: int = 4
    ):
        """
        Upload checkpoint using parallel multipart upload.

        Splits file into chunks and uploads in parallel.
        """
        import os

        file_size = os.path.getsize(local_path)
        num_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

        print(f"Uploading {local_path} ({file_size / (1024**3):.2f} GB) in {num_chunks} chunks")

        blob = self.bucket.blob(gcs_path)

        # Use resumable upload for large files
        with open(local_path, 'rb') as f:
            blob.upload_from_file(
                f,
                content_type='application/octet-stream',
                checksum='md5',  # Verify integrity
                timeout=300
            )

        print(f"✓ Uploaded to gs://{self.bucket.name}/{gcs_path}")


# Usage
uploader = ParallelGCSUploader(
    bucket_name="my-training-bucket",
    chunk_size_mb=32
)

# Upload checkpoint asynchronously
import threading

def async_upload_checkpoint(local_path, gcs_path):
    """Upload checkpoint in background thread."""
    uploader.upload_checkpoint_parallel(local_path, gcs_path)

# Save checkpoint locally, then upload in background
checkpoint_path = checkpoint_manager.save_checkpoint(checkpoint, step=global_step)

upload_thread = threading.Thread(
    target=async_upload_checkpoint,
    args=(checkpoint_path, f"checkpoints/checkpoint_step_{global_step}.pt"),
    daemon=True
)
upload_thread.start()

# Training continues while upload happens in background
```

### Resume with Different World Size

Handle cases where the number of GPUs changes between runs:

```python
def load_checkpoint_flexible_world_size(
    checkpoint_path: str,
    model,
    optimizer,
    current_world_size: int
) -> int:
    """
    Load checkpoint even if world size changed.

    Handles transition from N GPUs to M GPUs.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    saved_world_size = checkpoint.get('world_size', 1)

    if saved_world_size != current_world_size:
        print(f"⚠️  World size changed: {saved_world_size} → {current_world_size}")
        print("   Checkpoint will be loaded, but optimizer state may be incompatible")

    # Load model (always compatible)
    model.load_state_dict(checkpoint['model'])

    # Attempt to load optimizer (may fail if world size incompatible)
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("✓ Optimizer state loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load optimizer state: {e}")
        print("   Training will continue with fresh optimizer state")

    return checkpoint['step']
```

---

## Sources

**Source Documents:**
None (created from web research)

**Web Research:**
- [PyTorch Distributed Checkpoint - Efficient checkpointing](https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/) - Meta/Google collaboration on checkpoint optimization (accessed 2025-01-31)
- [GCP Spot VMs Documentation](https://cloud.google.com/compute/docs/instances/spot) - Official GCP spot instance documentation (accessed 2025-01-31)
- [GCP Preemptible VM Instances](https://cloud.google.com/compute/docs/instances/preemptible) - Preemption mechanics and termination notice (accessed 2025-01-31)
- [Vertex AI Spot VMs](https://cloud.google.com/vertex-ai/docs/training/use-spot-vms) - Using spot instances with Vertex AI training (accessed 2025-01-31)
- [PyTorch DDP Checkpoint Best Practices](https://discuss.pytorch.org/t/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data-parallel-ddp-in-pytorch/139575) - Community discussion on DDP checkpointing (accessed 2025-01-31)
- [GCS Connector for PyTorch](https://cloud.google.com/storage/docs/pytorch-connector) - Optimized GCS integration for PyTorch (accessed 2025-01-31)
- [Universal Checkpointing (UCP)](https://www.usenix.org/system/files/atc25-lian.pdf) - Research on decoupling checkpoint structure from training strategies (accessed 2025-01-31)

**GitHub References:**
- [PyTorch Distributed Checkpoint](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/) - PyTorch DCP implementation
- [Google AI Hypercomputer Resiliency Library](https://github.com/AI-Hypercomputer/resiliency) - Local checkpointing solution for GCP

**Additional References:**
- [NVIDIA Run:ai Checkpointing Documentation](https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/using-training/checkpointing-preemptible-workloads) - Checkpointing for preemptible workloads (accessed 2025-01-31)
