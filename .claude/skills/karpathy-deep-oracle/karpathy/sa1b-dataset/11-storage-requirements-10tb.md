# SA-1B Storage Requirements: ~10TB Uncompressed

## Overview

SA-1B requires substantial storage: **~2.5TB compressed** (as distributed) or **~10TB uncompressed** (after extraction). Planning storage is critical for training at scale.

**Breakdown:**
- Images (JPEG): ~1.5TB compressed → ~4TB uncompressed
- Annotations (JSON): ~1.0TB compressed → ~6TB uncompressed
- Total: ~2.5TB → ~10TB

## Storage Planning

**Minimum requirements:**
- **Download**: 2.5TB free space
- **Extraction**: 10TB free space
- **Training**: 10TB + working space (checkpoints, logs)

**Recommended setup:**
- SSD for frequently accessed data (random reads)
- HDD for full dataset storage (sequential reads)
- Network storage (GCS/S3) with local cache

## Compression Ratios

**Images**: 2.6× compression (JPEG quality ~90)
- 11M images × ~350KB avg = ~3.8TB
- Compressed: ~1.5TB

**Annotations**: 6× compression (gzip on JSON)
- 1.1B masks × RLE + metadata
- Compressed: ~1.0TB

## ARR-COC Implications

**Storage strategy for VLM training:**
- Subset selection: Download only needed categories
- Streaming: Load from cloud on-demand
- Caching: Keep hot data on fast storage

**Cost estimate (GCP)**:
- Standard Storage: $0.020/GB/month × 2500GB = $50/month
- Nearline (infrequent): $0.010/GB/month × 2500GB = $25/month

**Sources**: SA-1B Download Page, GCP Pricing
