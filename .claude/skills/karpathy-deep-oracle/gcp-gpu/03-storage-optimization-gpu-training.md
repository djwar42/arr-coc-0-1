# Storage Optimization for GPU Training: Local SSD, Persistent Disk, and GCS Strategies

**Comprehensive guide to storage architecture for GPU-accelerated ML training on GCP, covering Local SSD performance, Persistent Disk checkpointing, gcsfuse optimization, and cost-effective data loading patterns**

This document provides production-ready storage strategies for maximizing GPU utilization during training by eliminating I/O bottlenecks, implementing efficient checkpoint management, and optimizing data pipelines for high-throughput ML workloads.

---

## Section 1: Storage Options Overview (~80 lines)

### Storage Type Comparison for GPU Workloads

**Three primary storage options for GPU training:**

| Storage Type | IOPS (Read/Write) | Throughput | Latency | Use Case | Cost/GB/Month |
|--------------|-------------------|------------|---------|----------|---------------|
| **Local SSD (NVMe)** | 2.4M / 1.2M | 9.6 GB/s / 4.8 GB/s | <1ms | Hot training data, staging | $0.080 |
| **Persistent Disk SSD** | 100k / 50k | 1.2 GB/s / 400 MB/s | 1-2ms | Checkpoints, warm data | $0.170 |
| **Persistent Disk Balanced** | 6k / 6k | 240 MB/s / 240 MB/s | 2-5ms | Cold checkpoints, archives | $0.100 |
| **Cloud Storage (GCS)** | 5k / 5k | 250 MB/s / 250 MB/s | 10-50ms | Dataset repository, backups | $0.020 |

From [Google Cloud Local SSD documentation](https://cloud.google.com/products/local-ssd) (accessed 2025-11-16):
- Local SSD delivers 10-100x faster performance than Persistent Disk
- NVMe interface provides superior performance over SCSI
- 375 GB per device, up to 24 devices per instance (9 TB total)

From [Persistent Disk performance overview](https://cloud.google.com/compute/docs/disks/performance) (accessed 2025-11-16):
- Persistent Disk performance scales with size
- Balanced Persistent Disk offers cost-performance middle ground
- Regional Persistent Disk provides automatic replication

### Performance Scaling by Disk Size

**Persistent Disk SSD performance tiers:**

| Disk Size | Read IOPS | Write IOPS | Read Throughput | Write Throughput |
|-----------|-----------|------------|-----------------|------------------|
| 100 GB | 3,000 | 3,000 | 48 MB/s | 48 MB/s |
| 500 GB | 15,000 | 15,000 | 240 MB/s | 240 MB/s |
| 1 TB | 30,000 | 30,000 | 480 MB/s | 400 MB/s |
| 10 TB | 100,000 | 50,000 | 1,200 MB/s | 400 MB/s |

**Key insight:** Larger Persistent Disks provide better performance, but Local SSD outperforms at all sizes for training workloads.

### Local SSD Performance Characteristics

**NVMe Local SSD performance (per device):**

```
Single 375 GB Local SSD:
- Read IOPS: 400,000
- Write IOPS: 200,000
- Read throughput: 1,600 MB/s
- Write throughput: 800 MB/s

Maximum configuration (24 devices, 9 TB):
- Read IOPS: 2,400,000
- Write IOPS: 1,200,000
- Read throughput: 9,600 MB/s
- Write throughput: 4,800 MB/s
```

From search results on "NVMe Local SSD IOPS throughput GCP" (accessed 2025-11-16):
- NVMe interface provides lower latency than SCSI Local SSD
- Performance scales linearly with number of devices
- Ideal for high-throughput training data staging

**Limitations:**
- Ephemeral storage (data lost on VM shutdown/deletion)
- Cannot detach and reattach to different instances
- Not suitable for long-term checkpoint storage

---

## Section 2: Local SSD for Training Data Staging (~100 lines)

### Architecture: GCS → Local SSD → GPU Pipeline

**Recommended data flow for GPU training:**

```
GCS Bucket (Dataset Repository)
    ↓ (Bulk download at job start)
Local SSD (Hot Training Data)
    ↓ (High-speed random access)
GPU Memory (Active Batches)
    ↓ (Training computation)
Persistent Disk (Checkpoints)
    ↓ (Periodic snapshots)
GCS (Long-term Checkpoint Storage)
```

**Benefits of Local SSD staging:**
1. **Eliminate GCS latency**: Random reads from GCS (10-50ms) vs Local SSD (<1ms)
2. **Maximize GPU utilization**: GPUs never wait for data
3. **Enable shuffling**: Fast random access for epoch-level data shuffling
4. **Reduce network egress**: Pay once for GCS → VM transfer, unlimited local reads

### Mounting and Formatting Local SSD

**Create VM with Local SSD:**

```bash
# Create GPU instance with 4 Local SSDs (1.5 TB total)
gcloud compute instances create gpu-training-vm \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced
```

**Format and mount Local SSDs (RAID 0 for maximum performance):**

```bash
#!/bin/bash
# Format Local SSDs in RAID 0 for maximum throughput

# Install mdadm for RAID management
sudo apt-get update
sudo apt-get install -y mdadm

# Identify Local SSD devices
lsblk | grep nvme

# Create RAID 0 array from 4 Local SSDs
sudo mdadm --create /dev/md0 \
    --level=0 \
    --raid-devices=4 \
    /dev/nvme0n1 \
    /dev/nvme0n2 \
    /dev/nvme0n3 \
    /dev/nvme0n4

# Format with ext4
sudo mkfs.ext4 -F /dev/md0

# Create mount point
sudo mkdir -p /mnt/localssd

# Mount RAID array
sudo mount /dev/md0 /mnt/localssd

# Set permissions
sudo chmod a+w /mnt/localssd

# Verify performance
sudo hdparm -Tt /dev/md0
```

**Expected RAID 0 performance (4x Local SSD):**

```
/dev/md0:
 Timing cached reads:   38000 MB in  2.00 seconds = 19000.00 MB/sec
 Timing buffered disk reads: 19200 MB in  3.00 seconds = 6400.00 MB/sec
```

### Data Loading Pattern: Pre-stage from GCS

**Download dataset to Local SSD at job start:**

```python
from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_blob(blob, local_path):
    """Download single blob to local path."""
    blob.download_to_filename(local_path)
    return local_path

def stage_dataset_to_local_ssd(bucket_name, gcs_prefix, local_dir):
    """
    Download entire dataset from GCS to Local SSD before training.
    Uses parallel downloads for maximum throughput.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all blobs in dataset
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    print(f"Found {len(blobs)} files in gs://{bucket_name}/{gcs_prefix}")

    # Create local directory structure
    os.makedirs(local_dir, exist_ok=True)

    # Parallel download (32 workers for maximum throughput)
    download_tasks = []
    for blob in blobs:
        rel_path = blob.name.replace(gcs_prefix, "").lstrip("/")
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        download_tasks.append((blob, local_path))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(download_blob, blob, local_path)
            for blob, local_path in download_tasks
        ]

        # Progress bar
        for future in tqdm(futures, desc="Downloading dataset"):
            future.result()

    print(f"Dataset staged to {local_dir}")
    return local_dir

# Usage: Download 500GB dataset to Local SSD before training
dataset_path = stage_dataset_to_local_ssd(
    bucket_name="ml-training-data",
    gcs_prefix="datasets/imagenet/train/",
    local_dir="/mnt/localssd/imagenet-train"
)

# Now train with local data (10-100x faster than GCS)
train_loader = DataLoader(
    ImageFolder(dataset_path),
    batch_size=256,
    shuffle=True,  # Fast random access on Local SSD
    num_workers=8
)
```

**Performance comparison:**

| Source | First Epoch | Subsequent Epochs | Cost |
|--------|-------------|-------------------|------|
| GCS direct (gcsfuse) | 45 min | 45 min | $0 egress (same region) |
| GCS → Local SSD stage | 5 min download + 8 min train = 13 min | 8 min | $0 egress + $0.08/GB/month |
| Persistent Disk SSD | 25 min | 25 min | $0.17/GB/month |

**When to use Local SSD staging:**
- ✅ Datasets fit in Local SSD capacity (up to 9 TB)
- ✅ Multiple training epochs (amortize download cost)
- ✅ Random access patterns (shuffling, data augmentation)
- ✅ Maximum GPU utilization required

**When to use GCS direct:**
- ✅ Single-pass training (no epoch repeats)
- ✅ Sequential access patterns
- ✅ Dataset larger than Local SSD capacity
- ✅ Cost optimization (no Local SSD charges)

---

## Section 3: Persistent Disk for Checkpoint Storage (~100 lines)

### Why Persistent Disk for Checkpoints

**Persistent Disk advantages over Local SSD for checkpoints:**

1. **Durability**: Data persists across VM shutdowns/deletions
2. **Snapshots**: Point-in-time recovery via incremental snapshots
3. **Detachable**: Move checkpoints between VMs
4. **Regional replication**: Automatic redundancy for regional disks

From search results on "Persistent Disk snapshot for checkpoints GCP 2024" (accessed 2025-11-16):
- Snapshots are incremental (only changed blocks)
- Instant snapshots provide near-instantaneous point-in-time recovery
- Snapshots stored in Cloud Storage with geo-redundancy

### Checkpoint Storage Architecture

**Recommended setup:**

```
GPU Training VM:
├── Boot Disk: 200 GB Persistent Disk Balanced ($20/month)
├── Local SSD: 1.5 TB RAID 0 (training data staging) ($120/month)
└── Checkpoint Disk: 1 TB Persistent Disk SSD ($170/month)
```

**Create and attach checkpoint disk:**

```bash
# Create 1 TB SSD Persistent Disk for checkpoints
gcloud compute disks create checkpoint-disk \
    --size=1TB \
    --type=pd-ssd \
    --zone=us-central1-a

# Attach to running VM
gcloud compute instances attach-disk gpu-training-vm \
    --disk=checkpoint-disk \
    --zone=us-central1-a

# Format and mount (on VM)
sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/checkpoints
sudo mount /dev/sdb /mnt/checkpoints
sudo chmod a+w /mnt/checkpoints
```

### Checkpoint Saving Strategy

**Save checkpoints directly to Persistent Disk:**

```python
import torch
import os
from datetime import datetime

class CheckpointManager:
    """Manage training checkpoints on Persistent Disk with snapshot support."""

    def __init__(self, checkpoint_dir="/mnt/checkpoints", experiment_name="exp-001"):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_path = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, is_best=False):
        """Save checkpoint to Persistent Disk."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save regular checkpoint
        checkpoint_file = os.path.join(
            self.checkpoint_path,
            f"checkpoint_epoch_{epoch:03d}_step_{step:06d}.pt"
        )
        torch.save(checkpoint, checkpoint_file)
        print(f"Saved checkpoint: {checkpoint_file}")

        # Save best model separately
        if is_best:
            best_file = os.path.join(self.checkpoint_path, "best_model.pt")
            torch.save(checkpoint, best_file)
            print(f"Updated best model: {best_file}")

        return checkpoint_file

    def create_disk_snapshot(self, snapshot_name=None):
        """
        Create Persistent Disk snapshot for checkpoint backup.
        Requires gcloud SDK installed on VM.
        """
        if snapshot_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            snapshot_name = f"{self.experiment_name}-{timestamp}"

        # Get disk name and zone
        disk_name = "checkpoint-disk"  # From gcloud compute disks create
        zone = "us-central1-a"

        # Create snapshot via gcloud
        import subprocess
        cmd = [
            "gcloud", "compute", "disks", "snapshot", disk_name,
            "--snapshot-names", snapshot_name,
            "--zone", zone,
            "--storage-location", "us"  # Multi-region for durability
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Created snapshot: {snapshot_name}")
        else:
            print(f"Snapshot failed: {result.stderr}")

        return snapshot_name

# Usage in training loop
checkpoint_mgr = CheckpointManager(
    checkpoint_dir="/mnt/checkpoints",
    experiment_name="arr-coc-baseline"
)

best_loss = float('inf')
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        loss = train_step(batch)

        # Save checkpoint every 500 steps
        if step % 500 == 0:
            is_best = loss < best_loss
            if is_best:
                best_loss = loss

            checkpoint_mgr.save_checkpoint(
                model, optimizer, scheduler,
                epoch, step,
                metrics={'loss': loss, 'lr': scheduler.get_last_lr()[0]},
                is_best=is_best
            )

    # Create disk snapshot at end of each epoch
    checkpoint_mgr.create_disk_snapshot()
```

### Snapshot Management and Recovery

**Snapshot lifecycle policy:**

```bash
# List existing snapshots
gcloud compute snapshots list --filter="name~arr-coc"

# Create snapshot retention policy (keep last 7 daily snapshots)
gcloud compute resource-policies create snapshot-schedule daily-snapshots \
    --region=us-central1 \
    --max-retention-days=7 \
    --on-source-disk-delete=keep-auto-snapshots \
    --daily-schedule \
    --start-time=02:00

# Attach policy to checkpoint disk
gcloud compute disks add-resource-policies checkpoint-disk \
    --resource-policies=daily-snapshots \
    --zone=us-central1-a
```

**Restore from snapshot:**

```bash
# Create new disk from snapshot
gcloud compute disks create checkpoint-disk-restored \
    --source-snapshot=arr-coc-baseline-20251116-153000 \
    --size=1TB \
    --type=pd-ssd \
    --zone=us-central1-a

# Attach to VM and mount
gcloud compute instances attach-disk gpu-training-vm \
    --disk=checkpoint-disk-restored \
    --zone=us-central1-a

# Mount on VM
sudo mkdir -p /mnt/checkpoints-restored
sudo mount /dev/sdc /mnt/checkpoints-restored

# Resume training from restored checkpoint
python train.py --resume=/mnt/checkpoints-restored/arr-coc-baseline/best_model.pt
```

---

## Section 4: Cloud Storage (GCS) Integration (~100 lines)

### GCS for Dataset Repository and Long-term Checkpoints

**GCS role in GPU training storage architecture:**

1. **Dataset repository**: Source of truth for training data
2. **Checkpoint backups**: Long-term storage for completed experiments
3. **Multi-region redundancy**: Geo-replicated backup
4. **Cost optimization**: Cheapest storage for cold data

From [Cloud Storage optimization for ML workloads](../gcp-vertex/07-gcs-optimization-ml-workloads.md):
- Standard storage: $0.020/GB/month (same-region access)
- Nearline storage: $0.010/GB/month (30-day retention)
- Coldline storage: $0.004/GB/month (90-day retention)

### gcsfuse for Direct GCS Access

**When to use gcsfuse vs pre-staging:**

| Use Case | gcsfuse Direct | Pre-stage to Local SSD |
|----------|----------------|------------------------|
| Dataset size | >9 TB (doesn't fit Local SSD) | <9 TB |
| Access pattern | Sequential reads | Random access, shuffling |
| Training epochs | Single-pass | Multiple epochs |
| GPU utilization | 60-80% (I/O bound) | 95-99% (compute bound) |

**Optimized gcsfuse mount for GPU training:**

```bash
# Mount GCS bucket with GPU-optimized settings
gcsfuse \
    --implicit-dirs \
    --stat-cache-ttl=3600s \
    --type-cache-ttl=3600s \
    --max-conns-per-host=100 \
    --file-cache-max-size-mb=50000 \
    --temp-dir=/mnt/localssd/gcsfuse-cache \
    --experimental-metadata-prefetch-on-mount=all \
    ml-training-data /mnt/gcs-data
```

From search results on "gcsfuse performance GPU data loading 2024" (accessed 2025-11-16):
- Default max-conns-per-host (10) is too low for ML workloads
- Recommend 50-200 connections for multi-GPU training
- File caching on Local SSD improves subsequent epoch performance

**gcsfuse performance tuning:**

```python
import tensorflow as tf

# TensorFlow dataset with gcsfuse-mounted GCS
def create_tf_dataset_gcsfuse(data_dir, batch_size):
    """
    Create TensorFlow dataset from gcsfuse-mounted GCS bucket.
    Optimized for GPU training throughput.
    """
    files = tf.io.gfile.glob(f"{data_dir}/*.tfrecord")

    dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=tf.data.AUTOTUNE  # Parallel reads from gcsfuse
    )

    dataset = dataset.map(
        parse_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch while GPU trains

    return dataset

# Usage with gcsfuse-mounted GCS
dataset = create_tf_dataset_gcsfuse(
    data_dir="/mnt/gcs-data/tfrecords/train",
    batch_size=256
)
```

### Checkpoint Upload to GCS

**Upload checkpoints to GCS for long-term storage:**

```python
from google.cloud import storage
import os

class GCSCheckpointBackup:
    """Backup Persistent Disk checkpoints to GCS for long-term storage."""

    def __init__(self, bucket_name, experiment_name):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.experiment_name = experiment_name

    def upload_checkpoint(self, local_checkpoint_path):
        """Upload checkpoint from Persistent Disk to GCS."""
        # Construct GCS path
        filename = os.path.basename(local_checkpoint_path)
        gcs_path = f"checkpoints/{self.experiment_name}/{filename}"

        # Upload with parallel composite uploads (automatic for >150MB files)
        blob = self.bucket.blob(gcs_path)
        blob.chunk_size = 50 * 1024 * 1024  # 50MB chunks
        blob.upload_from_filename(local_checkpoint_path)

        print(f"Uploaded to gs://{self.bucket.name}/{gcs_path}")
        return f"gs://{self.bucket.name}/{gcs_path}"

    def sync_checkpoint_dir(self, local_dir):
        """Sync entire checkpoint directory to GCS."""
        import subprocess

        # Use gsutil for efficient directory sync
        gcs_dest = f"gs://{self.bucket.name}/checkpoints/{self.experiment_name}/"
        cmd = ["gsutil", "-m", "rsync", "-r", local_dir, gcs_dest]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Synced {local_dir} to {gcs_dest}")
        else:
            print(f"Sync failed: {result.stderr}")

# Usage: Backup best checkpoint to GCS at end of training
gcs_backup = GCSCheckpointBackup(
    bucket_name="ml-training-checkpoints",
    experiment_name="arr-coc-baseline"
)

# Upload best model to GCS
gcs_backup.upload_checkpoint("/mnt/checkpoints/arr-coc-baseline/best_model.pt")

# Sync all checkpoints to GCS
gcs_backup.sync_checkpoint_dir("/mnt/checkpoints/arr-coc-baseline/")
```

---

## Section 5: Storage Cost Optimization (~80 lines)

### Cost Analysis for GPU Training Storage

**Monthly storage costs for 1TB training job:**

| Storage Type | Cost/Month | Use Case | Performance |
|--------------|------------|----------|-------------|
| Local SSD (1 TB) | $80 | Hot training data | 2.4M IOPS, 9.6 GB/s |
| Persistent Disk SSD (1 TB) | $170 | Active checkpoints | 30k IOPS, 480 MB/s |
| Persistent Disk Balanced (1 TB) | $100 | Warm checkpoints | 6k IOPS, 240 MB/s |
| GCS Standard (1 TB) | $20 | Dataset repository | 5k IOPS, 250 MB/s |
| GCS Nearline (1 TB) | $10 | Recent checkpoints (<30d) | 5k IOPS, 250 MB/s |
| GCS Coldline (1 TB) | $4 | Old checkpoints (30-90d) | 5k IOPS, 250 MB/s |

**Optimized storage architecture for arr-coc-0-1:**

```
Training Job (7-day duration):

Local SSD (1.5 TB):
- Training data staging: $120 × 7/30 = $28

Persistent Disk SSD (500 GB):
- Active checkpoints: $85 × 7/30 = $20

GCS Standard (2 TB):
- Dataset repository: $40 (persistent)
- Checkpoint backups: $40 (persistent)

Total 7-day training cost: $28 + $20 + $80 = $128
```

### Lifecycle Policies for Checkpoint Management

**Automatic transition of checkpoints to cheaper storage:**

```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
          "age": 7,
          "matchesPrefix": ["checkpoints/arr-coc-baseline/"]
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["checkpoints/arr-coc-baseline/"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["checkpoints/arr-coc-ablation/"]
        }
      }
    ]
  }
}
```

**Apply lifecycle policy:**

```bash
# Save policy to file
cat > checkpoint-lifecycle.json << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 7}
      }
    ]
  }
}
EOF

# Apply to bucket
gsutil lifecycle set checkpoint-lifecycle.json gs://ml-training-checkpoints
```

### Cost Savings Calculation

**6-month training project with lifecycle management:**

```
Scenario: 100 TB of checkpoints generated over 6 months

Without lifecycle policy (all Standard storage):
- 100 TB × $0.020/GB × 6 months = $12,000

With lifecycle policy:
- First 7 days (Standard): 100 TB × $0.020 × 7/30 = $467
- Days 8-30 (Nearline): 100 TB × $0.010 × 23/30 = $767
- Days 31-180 (Coldline): 100 TB × $0.004 × 5 months = $2,000

Total with lifecycle: $3,234
Savings: $8,766 (73% cost reduction)
```

---

## Section 6: Data Loading Best Practices for GPU Training (~90 lines)

### Parallel Data Loading Architecture

**CPU-side data loading to keep GPUs fed:**

```python
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

class LocalSSDDataset(Dataset):
    """Dataset reading from Local SSD for maximum throughput."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_file_list()

    def _load_file_list(self):
        """Load list of files from Local SSD."""
        import glob
        return glob.glob(f"{self.data_dir}/**/*.jpg", recursive=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        # Load image from Local SSD (fast random access)
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self._extract_label(img_path)
        return img, label

    def _extract_label(self, path):
        # Extract label from path
        return int(path.split('/')[-2])

# Optimized DataLoader configuration for GPU training
def create_gpu_dataloader(data_dir, batch_size=256):
    """
    Create DataLoader optimized for GPU training throughput.
    Uses Local SSD for maximum I/O performance.
    """
    dataset = LocalSSDDataset(data_dir)

    # CPU count minus 2 (leave cores for OS and monitoring)
    num_workers = max(1, mp.cpu_count() - 2)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Fast on Local SSD
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=4,  # Prefetch 4 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )

    return dataloader

# Usage
train_loader = create_gpu_dataloader(
    data_dir="/mnt/localssd/imagenet-train",
    batch_size=256
)

# Training loop with data loading monitoring
for epoch in range(num_epochs):
    epoch_start = time.time()
    data_time = 0
    compute_time = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()

        # Data loading time
        data_time += batch_start - (compute_end if batch_idx > 0 else epoch_start)

        # Move to GPU
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Training step
        loss = train_step(images, labels)

        compute_end = time.time()
        compute_time += compute_end - batch_start

    # Monitor I/O bottlenecks
    total_time = time.time() - epoch_start
    gpu_util = compute_time / total_time * 100
    io_overhead = data_time / total_time * 100

    print(f"Epoch {epoch}: GPU utilization {gpu_util:.1f}%, I/O overhead {io_overhead:.1f}%")
```

**Target metrics:**
- GPU utilization: >95%
- I/O overhead: <5%
- Data loading time per batch: <100ms

**If GPU utilization <80%, diagnose:**
1. Increase `num_workers` (more parallel data loading)
2. Increase `prefetch_factor` (more batches queued)
3. Check if data is on Local SSD (not Persistent Disk or GCS)
4. Profile data loading pipeline (find slow transforms)

### Sharded TFRecords for Distributed Training

**Pre-shard datasets for efficient multi-GPU loading:**

```bash
# Create 100 sharded TFRecords for distributed training
python create_tfrecords.py \
    --input-dir=/mnt/localssd/imagenet-train/ \
    --output-pattern=/mnt/localssd/imagenet-tfrecords/train-{:05d}-of-00100.tfrecord \
    --num-shards=100
```

**Each GPU worker reads its own shards (no overlap):**

```python
import tensorflow as tf

def create_distributed_dataset(data_dir, global_batch_size, num_replicas, replica_id):
    """
    Create TensorFlow dataset for distributed training.
    Each replica reads non-overlapping shards.
    """
    # List all TFRecord shards
    files = tf.io.gfile.glob(f"{data_dir}/train-*.tfrecord")
    files = sorted(files)  # Deterministic ordering

    # Each replica reads its own subset of files
    files_per_replica = len(files) // num_replicas
    start_idx = replica_id * files_per_replica
    end_idx = start_idx + files_per_replica
    replica_files = files[start_idx:end_idx]

    # Create dataset from replica's files
    dataset = tf.data.TFRecordDataset(
        replica_files,
        num_parallel_reads=tf.data.AUTOTUNE
    )

    # Per-replica batch size
    batch_size = global_batch_size // num_replicas

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Usage with 8 GPUs
dataset = create_distributed_dataset(
    data_dir="/mnt/localssd/imagenet-tfrecords",
    global_batch_size=2048,  # 256 per GPU
    num_replicas=8,
    replica_id=hvd.rank()  # Horovod rank
)
```

---

## Section 7: arr-coc-0-1 Storage Architecture (~100 lines)

### Project-Specific Storage Configuration

**arr-coc-0-1 training requirements:**

```
Model: Qwen3-VL + QualityAdapter
- Checkpoint size: ~14 GB (7B params)
- Training data: 100K image-text pairs (~800 GB processed textures)
- Training duration: 7 days
- Checkpointing: Every 500 steps + every epoch
```

### Storage Provisioning for arr-coc-0-1

**Create VM with optimized storage:**

```bash
# Create A100 GPU VM with Local SSD + Persistent Disk
gcloud compute instances create arr-coc-training-vm \
    --zone=us-west2-b \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --create-disk=name=arr-coc-checkpoints,size=500GB,type=pd-ssd,auto-delete=no

# Estimated monthly costs:
# - A100 GPU: ~$2,900/month
# - Local SSD (750 GB): $60/month
# - Persistent Disk SSD (500 GB): $85/month
# - Persistent Disk Balanced (200 GB): $20/month
# Total storage: $165/month
```

### Data Pipeline Architecture

**Complete data flow for arr-coc-0-1:**

```
GCS Bucket: arr-coc-training-us-west2
├── datasets/
│   ├── raw/coco-captions/ (500 GB)
│   └── processed/texture-arrays/ (800 GB)
│
└── checkpoints/ (archived experiments)

↓ (Download at job start)

Local SSD: /mnt/localssd (750 GB)
├── texture-arrays/ (800 GB active training data)
└── gcsfuse-cache/ (cache for GCS access)

↓ (Training with random access)

GPU: A100 80GB
└── Active batches (64 patches × 13 channels)

↓ (Checkpoint every 500 steps)

Persistent Disk SSD: /mnt/checkpoints (500 GB)
├── arr-coc-baseline/
│   ├── step-000500/ (14 GB)
│   ├── step-001000/ (14 GB)
│   └── best_model.pt (14 GB)
│
└── arr-coc-ablation-no-eccentricity/
    └── step-000500/ (14 GB)

↓ (Snapshot daily)

GCS: Long-term checkpoint storage
└── checkpoints/arr-coc-baseline/ (lifecycle managed)
```

### Complete Training Setup Script

```bash
#!/bin/bash
# Setup storage for arr-coc-0-1 training

set -e

echo "=== Setting up storage for arr-coc-0-1 training ==="

# 1. Format and mount Local SSDs
echo "Configuring Local SSD..."
sudo mdadm --create /dev/md0 --level=0 --raid-devices=2 /dev/nvme0n1 /dev/nvme0n2
sudo mkfs.ext4 -F /dev/md0
sudo mkdir -p /mnt/localssd
sudo mount /dev/md0 /mnt/localssd
sudo chmod a+w /mnt/localssd

# 2. Format and mount checkpoint Persistent Disk
echo "Configuring checkpoint disk..."
sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/checkpoints
sudo mount /dev/sdb /mnt/checkpoints
sudo chmod a+w /mnt/checkpoints

# 3. Download training data from GCS to Local SSD
echo "Downloading training data to Local SSD..."
gsutil -m rsync -r \
    gs://arr-coc-training-us-west2/datasets/processed/texture-arrays/ \
    /mnt/localssd/texture-arrays/

# 4. Verify data
echo "Verifying data..."
ls -lh /mnt/localssd/texture-arrays/ | head -10
df -h /mnt/localssd
df -h /mnt/checkpoints

echo "=== Storage setup complete ==="
echo "Local SSD: /mnt/localssd (training data)"
echo "Persistent Disk: /mnt/checkpoints (checkpoints)"
```

### Training Script with Storage Optimization

```python
#!/usr/bin/env python3
"""arr-coc-0-1 training with optimized storage pipeline."""

import torch
from torch.utils.data import DataLoader
import os

# Storage paths
DATA_DIR = "/mnt/localssd/texture-arrays"  # Local SSD
CHECKPOINT_DIR = "/mnt/checkpoints/arr-coc-baseline"  # Persistent Disk
GCS_BUCKET = "gs://arr-coc-training-us-west2/checkpoints/"

# Import arr-coc modules
from arr_coc.texture import TextureArrayDataset
from arr_coc.knowing import KnowingModule
from arr_coc.realizing import RelevanceRealizationPipeline

def main():
    # Load data from Local SSD (fast random access)
    dataset = TextureArrayDataset(DATA_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,  # Fast on Local SSD
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize model
    model = RelevanceRealizationPipeline().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, "arr-coc-baseline")

    # Training loop
    for epoch in range(10):
        for step, batch in enumerate(train_loader):
            # Training step
            loss = train_step(model, batch, optimizer)

            # Save checkpoint every 500 steps
            if step % 500 == 0:
                checkpoint_file = checkpoint_mgr.save_checkpoint(
                    model, optimizer, None, epoch, step,
                    metrics={'loss': loss},
                    is_best=(loss < best_loss)
                )

                # Upload to GCS for backup
                upload_to_gcs(checkpoint_file, GCS_BUCKET)

        # Snapshot Persistent Disk at end of epoch
        checkpoint_mgr.create_disk_snapshot()

if __name__ == "__main__":
    main()
```

---

## Section 8: Monitoring Storage Performance (~80 lines)

### I/O Metrics for GPU Training

**Key metrics to monitor:**

1. **Disk throughput**: MB/s read/write
2. **IOPS**: Operations per second
3. **Queue depth**: Number of pending I/O operations
4. **CPU iowait**: Percentage of CPU waiting for I/O
5. **GPU utilization**: Should be >95% if I/O is optimized

### Real-time I/O Monitoring

**Monitor Local SSD performance during training:**

```bash
# Monitor disk I/O in real-time
iostat -x 1 /dev/md0

# Output:
# Device   rrqm/s wrqm/s    r/s    w/s  rMB/s  wMB/s avgrq-sz avgqu-sz await  %util
# md0        0.00   0.00 45000    0   1800      0     40.00      2.5   0.05  99.00

# Target metrics:
# - rMB/s: >1000 MB/s (Local SSD should sustain high throughput)
# - await: <1ms (low latency)
# - %util: >80% (disk is being fully utilized)
```

**Monitor GPU utilization:**

```bash
# Watch GPU utilization
nvidia-smi dmon -s u

# Output:
# gpu   sm   mem   enc   dec
#   0   99    95     0     0

# Target: sm (streaming multiprocessor) >95%
# If sm <80%, data loading is bottleneck
```

### Cloud Monitoring Dashboard

**Create custom dashboard for storage metrics:**

```python
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import query

def create_storage_dashboard(project_id, instance_name):
    """Create Cloud Monitoring dashboard for storage metrics."""
    client = monitoring_v3.DashboardsServiceClient()
    project_name = f"projects/{project_id}"

    dashboard = {
        "display_name": "GPU Training Storage Performance",
        "grid_layout": {
            "widgets": [
                {
                    "title": "Local SSD Read Throughput",
                    "xy_chart": {
                        "data_sets": [{
                            "time_series_query": {
                                "time_series_filter": {
                                    "filter": f'resource.type="gce_instance" AND resource.labels.instance_id="{instance_name}" AND metric.type="compute.googleapis.com/instance/disk/read_bytes_count"'
                                }
                            }
                        }]
                    }
                },
                {
                    "title": "Disk IOPS",
                    "xy_chart": {
                        "data_sets": [{
                            "time_series_query": {
                                "time_series_filter": {
                                    "filter": f'resource.type="gce_instance" AND metric.type="compute.googleapis.com/instance/disk/read_ops_count"'
                                }
                            }
                        }]
                    }
                }
            ]
        }
    }

    client.create_dashboard(name=project_name, dashboard=dashboard)
    print(f"Created dashboard: GPU Training Storage Performance")

# Usage
create_storage_dashboard("arr-coc-project", "arr-coc-training-vm")
```

### Alerting for Storage Bottlenecks

**Create alert when GPU utilization drops (data starvation):**

```bash
# Create alerting policy for low GPU utilization
gcloud alpha monitoring policies create \
    --notification-channels=EMAIL_CHANNEL_ID \
    --display-name="GPU Utilization Low - Data Bottleneck" \
    --condition-display-name="GPU SM <80%" \
    --condition-threshold-value=80 \
    --condition-threshold-duration=300s \
    --aggregation-alignment-period=60s \
    --condition-threshold-comparison=COMPARISON_LT \
    --condition-threshold-filter='metric.type="compute.googleapis.com/instance/gpu/utilization" resource.type="gce_instance"'
```

---

## Sources

**Google Cloud Documentation:**
- [Local SSD performance](https://cloud.google.com/products/local-ssd) - NVMe Local SSD specifications (accessed 2025-11-16)
- [Persistent Disk performance overview](https://cloud.google.com/compute/docs/disks/performance) - Persistent Disk IOPS and throughput (accessed 2025-11-16)
- [Create disk snapshots](https://cloud.google.com/compute/docs/disks/create-snapshots) - Snapshot management (accessed 2025-11-16)
- [Cloud Storage FUSE performance tuning](https://cloud.google.com/storage/docs/cloud-storage-fuse/performance) - gcsfuse optimization (accessed 2025-11-16)

**Source Documents:**
- [gcp-vertex/07-gcs-optimization-ml-workloads.md](../gcp-vertex/07-gcs-optimization-ml-workloads.md) - GCS optimization patterns, gcsfuse tuning, lifecycle policies
- [karpathy/practical-implementation/34-vertex-ai-data-integration.md](../karpathy/practical-implementation/34-vertex-ai-data-integration.md) - GCS bucket organization, data transfer patterns

**Web Research (accessed 2025-11-16):**
- Search results for "Local SSD vs Persistent Disk GPU training 2024" - Storage comparison for ML workloads
- Search results for "NVMe Local SSD IOPS throughput GCP" - Local SSD performance specifications
- Search results for "Persistent Disk snapshot for checkpoints GCP 2024" - Checkpoint backup strategies
- Search results for "gcsfuse performance GPU data loading 2024" - gcsfuse optimization for training
- [Why SSDs are a Great Way to Decrease Training Machine Learning Model Times](https://blog-us.kioxia.com/post/2024/11/08/why-ssds-are-a-great-way-to-decrease-training-machine-learning-model-times) - SSD benefits for ML (November 2024)
- [Choosing storage for deep learning: a comprehensive guide](https://medium.com/nebius/choosing-storage-for-deep-learning-a-comprehensive-guide-b415bad207fa) - Storage selection patterns (2023)
- [Scaling ML workloads with gcsfuse](https://medium.com/google-cloud/scaling-new-heights-addressing-ai-ml-workload-scale-challenges-in-gke-gcsfuse-csi-driver-539eb377a660) - gcsfuse optimization (November 2023)

**Additional References:**
- Google Cloud pricing calculator - Storage cost estimations
- Cloud Storage pricing - Current rates (accessed 2025-11-16)
- Compute Engine disk pricing - Local SSD and Persistent Disk costs

---

*This document provides production-ready storage optimization strategies for GPU training on GCP, covering Local SSD staging, Persistent Disk checkpointing, gcsfuse integration, and arr-coc-0-1 implementation patterns for maximum training throughput and cost efficiency.*
