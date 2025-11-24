# SA-1B Storage Requirements: Infrastructure Planning for 10TB Dataset

## Overview

SA-1B is one of the largest publicly available datasets for computer vision, requiring approximately 10-11TB of storage when fully downloaded. Understanding storage requirements, infrastructure planning, and cost optimization strategies is essential for successfully working with this massive dataset. This document provides comprehensive guidance for research teams planning to download, store, and process SA-1B.

---

## 1. Total Dataset Size Breakdown

### Complete Dataset Statistics

| Component | Count | Size |
|-----------|-------|------|
| Tar files | 1,000 | ~10-11TB total |
| Images | 11 million | ~10TB (majority of size) |
| Annotations | 11 million JSON files | ~50-100GB |
| Masks | 1.1 billion | Encoded in JSONs |

### Per-Tar File Breakdown

Each of the 1,000 tar files contains:
- **Average size**: 10-11GB compressed
- **Images per tar**: ~11,000
- **Annotations per tar**: ~11,000 JSONs
- **Total masks per tar**: ~1.1 million (100 per image average)

```python
# Calculate storage per tar
TAR_COUNT = 1000
IMAGES_PER_TAR = 11000
AVG_TAR_SIZE_GB = 10.5

total_gb = TAR_COUNT * AVG_TAR_SIZE_GB
total_tb = total_gb / 1024

print(f"Total dataset size: {total_gb:,.0f} GB ({total_tb:.1f} TB)")
print(f"Images per tar: {IMAGES_PER_TAR:,}")
print(f"Total images: {TAR_COUNT * IMAGES_PER_TAR:,}")
```

### Size Distribution Analysis

```python
# From GitHub Issue #60 - actual measured sizes
# Total size: 11,298,949,953,923 bytes = ~11.3 TB

def analyze_tar_sizes(tar_dir):
    """Analyze size distribution of tar files."""
    import os
    import glob

    tar_files = glob.glob(os.path.join(tar_dir, "sa_*.tar"))
    sizes = []

    for tar in tar_files:
        size_gb = os.path.getsize(tar) / (1024**3)
        sizes.append(size_gb)

    return {
        "count": len(sizes),
        "total_gb": sum(sizes),
        "mean_gb": np.mean(sizes),
        "std_gb": np.std(sizes),
        "min_gb": np.min(sizes),
        "max_gb": np.max(sizes)
    }
```

---

## 2. Storage Medium Considerations

### SSD vs HDD Trade-offs

| Factor | SSD (NVMe) | HDD |
|--------|------------|-----|
| Cost per TB | $80-150 | $15-25 |
| Total cost (11TB) | $880-1,650 | $165-275 |
| Read speed | 3,000-7,000 MB/s | 100-200 MB/s |
| Random access | 50,000+ IOPS | 50-100 IOPS |
| Training impact | Excellent | Significant bottleneck |
| Recommendation | **Preferred** | Archival only |

### Speed Impact on Training

**I/O bottleneck analysis:**

```python
# Training batch loading time comparison

# HDD scenario
hdd_read_speed_mbps = 150
batch_size = 32
images_per_sample = 1
avg_sample_size_mb = 1.5  # Image + annotations

hdd_batch_time = (batch_size * images_per_sample * avg_sample_size_mb) / hdd_read_speed_mbps
print(f"HDD batch load time: {hdd_batch_time*1000:.0f} ms")  # ~320 ms

# SSD scenario
ssd_read_speed_mbps = 3500
ssd_batch_time = (batch_size * images_per_sample * avg_sample_size_mb) / ssd_read_speed_mbps
print(f"SSD batch load time: {ssd_batch_time*1000:.0f} ms")  # ~14 ms

# With 4 workers prefetching, SSD keeps GPU fully utilized
# HDD would require 20+ workers to hide latency
```

### Recommended Storage Configuration

**For active training:**
```
Primary: NVMe SSD (1-2 TB) - Working subset
Secondary: SSD array (4-8 TB) - Additional data
Backup: HDD (12+ TB) - Cold storage
```

**For exploration/research:**
```
Primary: SSD (2-4 TB) - Frequently accessed subsets
Secondary: HDD (12+ TB) - Full dataset
```

---

## 3. Cloud Storage Options and Costs

### AWS S3 Storage

```python
# AWS S3 cost estimation for SA-1B

class S3CostEstimator:
    def __init__(self, size_tb=11):
        self.size_tb = size_tb
        self.size_gb = size_tb * 1024

        # S3 pricing (us-east-1, as of 2024)
        self.standard_per_gb = 0.023
        self.infrequent_per_gb = 0.0125
        self.glacier_per_gb = 0.004

    def monthly_storage_cost(self, tier="standard"):
        rates = {
            "standard": self.standard_per_gb,
            "infrequent": self.infrequent_per_gb,
            "glacier": self.glacier_per_gb
        }
        return self.size_gb * rates[tier]

    def annual_cost(self, tier="standard"):
        return self.monthly_storage_cost(tier) * 12

# Calculate costs
estimator = S3CostEstimator(11)

print("Monthly storage costs:")
print(f"  S3 Standard: ${estimator.monthly_storage_cost('standard'):.2f}")
print(f"  S3 Infrequent: ${estimator.monthly_storage_cost('infrequent'):.2f}")
print(f"  S3 Glacier: ${estimator.monthly_storage_cost('glacier'):.2f}")

print("\nAnnual storage costs:")
print(f"  S3 Standard: ${estimator.annual_cost('standard'):.2f}")
# S3 Standard: ~$260/month, ~$3,100/year
```

### Google Cloud Storage

```python
# GCS cost estimation

class GCSCostEstimator:
    def __init__(self, size_tb=11):
        self.size_gb = size_tb * 1024

        # GCS pricing (us-central1)
        self.standard_per_gb = 0.020
        self.nearline_per_gb = 0.010
        self.coldline_per_gb = 0.004
        self.archive_per_gb = 0.0012

    def monthly_cost(self, tier="standard"):
        rates = {
            "standard": self.standard_per_gb,
            "nearline": self.nearline_per_gb,
            "coldline": self.coldline_per_gb,
            "archive": self.archive_per_gb
        }
        return self.size_gb * rates[tier]

gcs = GCSCostEstimator(11)
print(f"GCS Standard: ${gcs.monthly_cost('standard'):.2f}/month")
print(f"GCS Nearline: ${gcs.monthly_cost('nearline'):.2f}/month")
print(f"GCS Coldline: ${gcs.monthly_cost('coldline'):.2f}/month")
# Standard: ~$225/month, Coldline: ~$45/month
```

### Azure Blob Storage

```python
# Azure Blob cost estimation

class AzureCostEstimator:
    def __init__(self, size_tb=11):
        self.size_gb = size_tb * 1024

        # Azure pricing (US East)
        self.hot_per_gb = 0.0184
        self.cool_per_gb = 0.01
        self.archive_per_gb = 0.00099

    def monthly_cost(self, tier="hot"):
        rates = {
            "hot": self.hot_per_gb,
            "cool": self.cool_per_gb,
            "archive": self.archive_per_gb
        }
        return self.size_gb * rates[tier]

azure = AzureCostEstimator(11)
print(f"Azure Hot: ${azure.monthly_cost('hot'):.2f}/month")
print(f"Azure Cool: ${azure.monthly_cost('cool'):.2f}/month")
# Hot: ~$207/month, Cool: ~$113/month
```

### Cost Comparison Summary

| Provider | Hot Tier | Cool Tier | Archive |
|----------|----------|-----------|---------|
| AWS S3 | $260/mo | $141/mo | $45/mo |
| GCS | $225/mo | $113/mo | $13/mo |
| Azure | $207/mo | $113/mo | $11/mo |

**Recommendation:** GCS Coldline or Azure Archive for long-term storage (~$40-50/month)

---

## 4. Compression Analysis

### Tar File Compression

SA-1B tar files are **uncompressed** archives (pure tar, not tar.gz):

```python
# Check if further compression is worthwhile

def test_compression(tar_path):
    """Test additional compression on tar file."""
    import gzip
    import lzma
    import os

    original_size = os.path.getsize(tar_path)

    # Test gzip
    with open(tar_path, 'rb') as f_in:
        with gzip.open(tar_path + '.gz', 'wb', compresslevel=6) as f_out:
            f_out.write(f_in.read())
    gzip_size = os.path.getsize(tar_path + '.gz')

    # Results for SA-1B (JPEG images already compressed)
    # Typical compression ratio: 0.95-1.02 (almost no benefit)

    return {
        "original_gb": original_size / (1024**3),
        "gzip_gb": gzip_size / (1024**3),
        "ratio": gzip_size / original_size
    }
```

**Why additional compression doesn't help:**
- JPEG images are already compressed
- JSON annotations use RLE (already compressed)
- Compression overhead may increase I/O time

**Recommendation:** Store tar files as-is (uncompressed tar)

### Space-Saving Strategies

```python
def compute_subset_sizes():
    """Calculate storage for common subset strategies."""
    full_size_tb = 11.0

    strategies = {
        "full_dataset": full_size_tb,
        "10_percent": full_size_tb * 0.1,  # 1.1 TB (100 tars)
        "1_percent": full_size_tb * 0.01,   # 110 GB (10 tars)
        "single_tar": 0.011,                 # 11 GB
        "high_quality_only": full_size_tb * 0.3,  # ~3.3 TB (top 30% quality)
    }

    print("Storage requirements by strategy:")
    for name, size in strategies.items():
        print(f"  {name}: {size*1024:.0f} GB ({size:.1f} TB)")

    return strategies
```

---

## 5. Download Infrastructure Requirements

### Bandwidth Planning

```python
def estimate_download_time(size_tb, bandwidth_mbps, parallel_downloads=4):
    """Estimate download time for SA-1B."""
    size_bits = size_tb * 1024 * 1024 * 1024 * 1024 * 8
    effective_bandwidth = bandwidth_mbps * 0.8  # Account for overhead

    # With parallel downloads
    total_bandwidth = effective_bandwidth * parallel_downloads
    seconds = size_bits / (total_bandwidth * 1_000_000)

    hours = seconds / 3600
    days = hours / 24

    return {
        "hours": hours,
        "days": days,
        "effective_mbps": total_bandwidth
    }

# Common scenarios
scenarios = [
    ("Home (100 Mbps)", 100, 4),
    ("Office (500 Mbps)", 500, 8),
    ("Data center (1 Gbps)", 1000, 16),
    ("Cloud instance (10 Gbps)", 10000, 32),
]

print("Download time estimates for 11 TB:")
for name, bandwidth, parallel in scenarios:
    result = estimate_download_time(11, bandwidth, parallel)
    print(f"  {name}: {result['days']:.1f} days ({result['hours']:.0f} hours)")

# Results:
# Home (100 Mbps): ~24 days
# Office (500 Mbps): ~5 days
# Data center (1 Gbps): ~2.5 days
# Cloud (10 Gbps): ~6 hours
```

### Disk Space During Download

**Important:** Need additional space during download:
- Downloaded tar files: 11 TB
- Extraction workspace: +11 TB (if extracting)
- **Total**: 22+ TB recommended

```python
def plan_download_storage():
    """Plan storage for download and extraction."""
    download_tb = 11.0
    extraction_buffer = 0.2  # 20% buffer

    if extracting:
        # Need space for tars + extracted files
        total = download_tb * 2 * (1 + extraction_buffer)
    else:
        # Keep in tar format
        total = download_tb * (1 + extraction_buffer)

    return total

# Recommendation: 25+ TB for full extraction
# Recommendation: 14+ TB for tar-only storage
```

---

## 6. Memory Requirements for Processing

### RAM Planning

```python
def estimate_memory_requirements():
    """Estimate RAM needs for SA-1B processing."""

    # Single image with all masks
    image_size_mb = 2  # 1500x2250 RGB
    masks_per_image = 100
    mask_size_mb = (1500 * 2250) / (1024 * 1024)  # ~3.2 MB per decoded mask
    annotations_mb = 0.01  # JSON annotation

    single_sample_decoded = image_size_mb + (masks_per_image * mask_size_mb) + annotations_mb
    print(f"Single image with decoded masks: {single_sample_decoded:.0f} MB")

    # Batch processing
    batch_sizes = [1, 4, 8, 16, 32]
    for batch in batch_sizes:
        ram_gb = (batch * single_sample_decoded) / 1024
        print(f"  Batch {batch}: {ram_gb:.1f} GB RAM")

    # Recommended configurations
    print("\nRecommended RAM:")
    print("  Exploration: 32 GB")
    print("  Training (small batch): 64 GB")
    print("  Training (large batch): 128+ GB")

    return single_sample_decoded

# Output:
# Single image with decoded masks: ~325 MB
# Batch 8: ~2.6 GB
# Batch 32: ~10.4 GB (just for data, plus model, optimizer, etc.)
```

### GPU Memory

```python
def gpu_memory_planning():
    """Plan GPU memory for SA-1B training."""

    # Typical requirements
    configs = {
        "Exploration (inference)": {
            "vram": 8,
            "batch_size": 1,
            "model": "SAM-base"
        },
        "Fine-tuning (encoder frozen)": {
            "vram": 16,
            "batch_size": 4,
            "model": "SAM-base"
        },
        "Full training": {
            "vram": 40,
            "batch_size": 8,
            "model": "SAM-huge"
        },
        "Multi-GPU training": {
            "vram": "8x40",
            "batch_size": 64,
            "model": "SAM-huge"
        }
    }

    return configs
```

---

## 7. Subset Strategies for Limited Storage

### Strategic Subset Selection

```python
class SA1BSubsetSelector:
    """Select representative subsets of SA-1B."""

    @staticmethod
    def select_by_tar_count(tar_count):
        """Select specific number of tar files."""
        # Uniform sampling across 1000 tars
        step = 1000 // tar_count
        selected = list(range(0, 1000, step))[:tar_count]
        return [f"sa_{i:06d}.tar" for i in selected]

    @staticmethod
    def recommended_subsets():
        """Get recommended subset configurations."""
        return {
            # For quick experiments
            "tiny": {
                "tars": 1,
                "size_gb": 11,
                "images": 11000,
                "masks": 1100000
            },
            # For development
            "small": {
                "tars": 10,
                "size_gb": 110,
                "images": 110000,
                "masks": 11000000
            },
            # For serious experimentation
            "medium": {
                "tars": 100,
                "size_gb": 1100,
                "images": 1100000,
                "masks": 110000000
            },
            # For full research
            "large": {
                "tars": 500,
                "size_gb": 5500,
                "images": 5500000,
                "masks": 550000000
            }
        }

# Usage
subsets = SA1BSubsetSelector.recommended_subsets()
for name, config in subsets.items():
    print(f"{name}: {config['tars']} tars, {config['size_gb']} GB, {config['images']:,} images")
```

### Quality-Based Subset

```python
def create_quality_subset(full_annotations_index, quality_threshold=0.95):
    """
    Create subset of high-quality masks only.

    This can reduce storage while maintaining quality.
    Typical retention: 30-50% of masks at 0.95 threshold.
    """
    high_quality_count = 0
    total_count = 0

    # Would need to scan all annotations
    # Estimate: ~30% of masks have predicted_iou >= 0.95

    estimated_size_reduction = 0.3  # 30% of full size
    reduced_size_tb = 11.0 * estimated_size_reduction

    return {
        "estimated_size_tb": reduced_size_tb,
        "quality_threshold": quality_threshold,
        "estimated_masks": int(1.1e9 * estimated_size_reduction)
    }
```

---

## 8. Infrastructure Planning Checklist

### Hardware Checklist

```markdown
## SA-1B Infrastructure Checklist

### Storage
- [ ] Primary storage: _____ TB (minimum 11 TB, recommended 22+ TB)
- [ ] Storage type: [ ] SSD [ ] HDD [ ] Hybrid
- [ ] RAID configuration: [ ] RAID 0 [ ] RAID 5 [ ] RAID 10
- [ ] Backup solution: _____

### Network
- [ ] Download bandwidth: _____ Mbps
- [ ] Estimated download time: _____ days
- [ ] Multiple download connections: [ ] Yes (_____ parallel) [ ] No

### Compute
- [ ] RAM: _____ GB (minimum 32 GB, recommended 64+ GB)
- [ ] CPU cores: _____ (minimum 8, recommended 16+)
- [ ] GPU VRAM: _____ GB (if training)

### Cloud (if applicable)
- [ ] Provider: [ ] AWS [ ] GCP [ ] Azure
- [ ] Storage tier: _____
- [ ] Monthly budget: $_____
- [ ] Egress costs considered: [ ] Yes [ ] No
```

### Budget Planning

```python
def compute_total_cost(
    local_storage_tb=11,
    ssd_price_per_tb=100,
    cloud_backup=True,
    cloud_tier="coldline",
    cloud_months=12
):
    """Compute total infrastructure cost."""

    # Local storage
    local_cost = local_storage_tb * ssd_price_per_tb

    # Cloud backup (GCS Coldline rates)
    cloud_monthly_rates = {
        "standard": 0.020 * 1024,  # per TB
        "nearline": 0.010 * 1024,
        "coldline": 0.004 * 1024,
        "archive": 0.0012 * 1024
    }

    cloud_cost = 0
    if cloud_backup:
        monthly = local_storage_tb * cloud_monthly_rates.get(cloud_tier, 0.004 * 1024)
        cloud_cost = monthly * cloud_months

    total = local_cost + cloud_cost

    return {
        "local_storage": local_cost,
        "cloud_backup": cloud_cost,
        "total": total,
        "per_month": (local_cost / 24) + (cloud_cost / cloud_months) if cloud_months > 0 else local_cost / 24
    }

# Example budget
costs = compute_total_cost(
    local_storage_tb=11,
    ssd_price_per_tb=100,
    cloud_backup=True,
    cloud_tier="coldline",
    cloud_months=12
)
print(f"Initial setup: ${costs['local_storage']:.0f}")
print(f"Annual cloud backup: ${costs['cloud_backup']:.0f}")
print(f"Total first year: ${costs['total']:.0f}")
```

---

## 9. Data Transfer Optimization

### Optimized Download Script

```python
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class SA1BDownloader:
    """Optimized downloader for SA-1B dataset."""

    def __init__(self, output_dir, max_concurrent=8):
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        os.makedirs(output_dir, exist_ok=True)

    async def download_tar(self, session, url, filename):
        """Download single tar file with resume support."""
        filepath = os.path.join(self.output_dir, filename)

        # Check for partial download
        start_byte = 0
        if os.path.exists(filepath):
            start_byte = os.path.getsize(filepath)

        headers = {}
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'

        async with session.get(url, headers=headers) as response:
            mode = 'ab' if start_byte > 0 else 'wb'
            with open(filepath, mode) as f:
                async for chunk in response.content.iter_chunked(1024*1024):
                    f.write(chunk)

        return filepath

    async def download_all(self, urls):
        """Download all tar files with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession() as session:
            async def bounded_download(url):
                async with semaphore:
                    filename = url.split('/')[-1]
                    return await self.download_tar(session, url, filename)

            tasks = [bounded_download(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

# Usage
# downloader = SA1BDownloader("/data/sa1b", max_concurrent=8)
# urls = load_urls_from_links_file("segment_anything_links.txt")
# asyncio.run(downloader.download_all(urls))
```

### Transfer Between Storage Systems

```bash
# Fast copy between SSDs (Linux)
rsync -avh --progress /source/sa1b/ /destination/sa1b/

# Parallel transfer with GNU Parallel
find /source/sa1b -name "*.tar" | parallel -j 4 cp {} /destination/sa1b/

# Between cloud and local
# AWS
aws s3 sync s3://bucket/sa1b/ /local/sa1b/ --quiet

# GCS
gsutil -m rsync -r gs://bucket/sa1b/ /local/sa1b/
```

---

## 10. Monitoring and Maintenance

### Storage Monitoring

```python
import shutil
import psutil

def monitor_storage(path):
    """Monitor storage usage for SA-1B directory."""
    usage = shutil.disk_usage(path)

    return {
        "total_tb": usage.total / (1024**4),
        "used_tb": usage.used / (1024**4),
        "free_tb": usage.free / (1024**4),
        "percent_used": (usage.used / usage.total) * 100
    }

def check_storage_health(path, warning_threshold=85, critical_threshold=95):
    """Check storage health and alert."""
    status = monitor_storage(path)

    if status["percent_used"] >= critical_threshold:
        return "CRITICAL", f"Storage {status['percent_used']:.1f}% full"
    elif status["percent_used"] >= warning_threshold:
        return "WARNING", f"Storage {status['percent_used']:.1f}% full"
    else:
        return "OK", f"Storage {status['percent_used']:.1f}% used"

# Usage
level, message = check_storage_health("/data/sa1b")
print(f"[{level}] {message}")
```

### Integrity Verification

```python
import hashlib

def verify_tar_integrity(tar_path, expected_checksum=None):
    """Verify tar file integrity."""
    sha256 = hashlib.sha256()

    with open(tar_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    actual_checksum = sha256.hexdigest()

    if expected_checksum:
        is_valid = actual_checksum == expected_checksum
        return is_valid, actual_checksum
    else:
        return True, actual_checksum

def verify_all_tars(tar_dir, checksum_file=None):
    """Verify all tar files in directory."""
    import glob

    tar_files = sorted(glob.glob(os.path.join(tar_dir, "sa_*.tar")))
    results = []

    for tar_path in tar_files:
        is_valid, checksum = verify_tar_integrity(tar_path)
        results.append({
            "file": os.path.basename(tar_path),
            "valid": is_valid,
            "checksum": checksum
        })

    return results
```

---

## 11. ARR-COC-0-1 Integration: Infrastructure Planning for Large-Scale Training

### Storage Strategy for ARR-COC

SA-1B's scale requires careful infrastructure planning for ARR-COC spatial relevance training:

1. **Subset approach**: Start with 1-10% for development
2. **Quality filtering**: Use predicted_iou/stability_score to reduce size
3. **Progressive loading**: Stream from tar files during training
4. **Distributed storage**: Spread across multiple nodes for parallel training

### Infrastructure Recommendations

```python
class ARRCOCInfraPlanner:
    """Plan infrastructure for ARR-COC training on SA-1B."""

    @staticmethod
    def recommend_config(training_scale):
        """Get recommended infrastructure based on training scale."""

        configs = {
            "development": {
                "sa1b_subset": "1%",
                "storage_tb": 1,
                "storage_type": "SSD",
                "ram_gb": 32,
                "gpu": "1x RTX 3090 (24GB)",
                "estimated_cost": "$1,500"
            },
            "research": {
                "sa1b_subset": "10%",
                "storage_tb": 2,
                "storage_type": "NVMe SSD",
                "ram_gb": 64,
                "gpu": "1x A100 (40GB)",
                "estimated_cost": "$5,000"
            },
            "production": {
                "sa1b_subset": "100%",
                "storage_tb": 24,
                "storage_type": "NVMe SSD array",
                "ram_gb": 256,
                "gpu": "8x A100 (40GB)",
                "estimated_cost": "$50,000+"
            }
        }

        return configs.get(training_scale, configs["research"])

    @staticmethod
    def cloud_training_cost(
        hours_per_epoch,
        num_epochs,
        instance_type="a2-highgpu-8g"  # 8x A100
    ):
        """Estimate cloud training costs."""

        # GCP pricing (approximate)
        hourly_rates = {
            "a2-highgpu-1g": 3.67,   # 1x A100
            "a2-highgpu-2g": 7.35,   # 2x A100
            "a2-highgpu-4g": 14.69,  # 4x A100
            "a2-highgpu-8g": 29.39,  # 8x A100
        }

        rate = hourly_rates.get(instance_type, 29.39)
        total_hours = hours_per_epoch * num_epochs
        compute_cost = total_hours * rate

        # Add storage costs (11 TB for 1 month)
        storage_cost = 11 * 1024 * 0.020  # GCS Standard

        return {
            "compute_hours": total_hours,
            "compute_cost": compute_cost,
            "storage_cost": storage_cost,
            "total": compute_cost + storage_cost
        }

# Example: 100 epochs, 10 hours each
costs = ARRCOCInfraPlanner.cloud_training_cost(
    hours_per_epoch=10,
    num_epochs=100,
    instance_type="a2-highgpu-8g"
)
print(f"Training cost estimate: ${costs['total']:,.0f}")
# ~$30,000 for full training run
```

---

## Sources

**Web Research:**
- [GitHub segment-anything Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) - Download discussions and size analysis (accessed 2025-11-20)
- [GitHub segment-anything Issue #60](https://github.com/facebookresearch/segment-anything/issues/60) - Full dataset size measurement (11.3 TB)
- [Hugging Face Forums SA-1B Discussion](https://discuss.huggingface.co/t/problems-about-loading-and-managing-extreme-large-datasets-like-sa1b/46520) - Storage management strategies
- [Meta AI SA-1B Downloads](https://ai.meta.com/datasets/segment-anything-downloads/) - Official download information

**Cloud Pricing References:**
- AWS S3 Pricing: https://aws.amazon.com/s3/pricing/
- GCS Pricing: https://cloud.google.com/storage/pricing
- Azure Blob Pricing: https://azure.microsoft.com/pricing/details/storage/blobs/

**ARR-COC Integration:**
- Infrastructure planning enables large-scale spatial grounding training
- Subset strategies allow progressive scaling of relevance realization training
- Cloud cost optimization supports research budget constraints
