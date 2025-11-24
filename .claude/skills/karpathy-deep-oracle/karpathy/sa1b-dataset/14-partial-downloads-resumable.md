# Partial Downloads & Resumable Transfers

## Overview

SA-1B downloads support **resumable transfers** and **partial dataset selection**, critical for unreliable networks and subset experiments.

**Key features:**
- `wget --continue` for resume after interruption
- Select specific tar ranges (e.g., tars 0-99 only)
- Bandwidth throttling to avoid network saturation
- Checksum verification for data integrity

## Resumable Downloads

**wget with --continue flag:**
```bash
# Download with auto-resume on failure
wget --continue \
     --tries=10 \
     --timeout=60 \
     --progress=bar:force \
     https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar

# If interrupted (Ctrl+C or network drop), re-run same command
# wget picks up where it left off
```

**Check partial file:**
```bash
# Partial downloads have .tmp or incomplete extension
ls -lh sa_000000.tar*
# -rw-r--r-- 1 user user 8.5G Nov 20 10:30 sa_000000.tar.1 (partial)
```

## Partial Dataset Selection

**Download specific ranges:**
```bash
# Get links for tars 0-99 only (10% of dataset)
sed -n '1,100p' segment_anything_links.txt > subset_100.txt
wget -i subset_100.txt --continue

# Or use head/tail
head -100 segment_anything_links.txt | wget -i -
```

**Random sampling:**
```bash
# Download 50 random tars for diversity
shuf -n 50 segment_anything_links.txt > random_50.txt
wget -i random_50.txt
```

## Bandwidth Management

**Throttle to avoid network congestion:**
```bash
# Limit to 10MB/s
wget --limit-rate=10m -i segment_anything_links.txt

# Download during off-peak hours only
# (use cron job at night)
```

## Verification After Resume

**Check tar integrity:**
```bash
tar -tzf sa_000000.tar > /dev/null && echo "OK" || echo "CORRUPTED"

# If corrupted, delete and re-download
rm sa_000000.tar
wget --continue <url>
```

## ARR-COC Use Cases

**Iterative scaling:**
1. Start: 10 tars (~110k images) for initial training
2. Validate: Check relevance model quality
3. Scale: Add 40 more tars → 500k images
4. Production: Full dataset if needed

**Cost-efficient cloud downloads:**
- Download to local workstation first
- Transfer to cloud storage (GCS/S3) once
- Training VMs read from cloud (no egress)

**Sources**: wget manual, cloud storage best practices
