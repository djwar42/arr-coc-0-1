# Parallel Tar Extraction for SA-1B

## Overview

Extracting 1,000 tars sequentially takes hours. **Parallel extraction** reduces time from ~10 hours to ~2 hours on multi-core systems.

**Key techniques:**
- GNU parallel for multi-tar extraction
- `pigz` for parallel gzip decompression
- Progress monitoring with `pv`

## Parallel Extraction Script

**Simple parallel extraction:**
```bash
# Extract 4 tars concurrently
ls sa_*.tar | xargs -n 1 -P 4 tar -xf

# Or with GNU parallel (better progress tracking)
ls sa_*.tar | parallel -j 4 'tar -xf {}'
```

**Production script:**
```bash
#!/bin/bash
# extract_sa1b_parallel.sh

NUM_WORKERS=8  # Adjust based on CPU cores
TAR_DIR="./tars"
OUTPUT_DIR="./sa1b_extracted"

mkdir -p "$OUTPUT_DIR"

# Find all tars and extract in parallel
find "$TAR_DIR" -name "sa_*.tar" | \
  parallel -j "$NUM_WORKERS" \
    --bar \
    --eta \
    'tar -xf {} -C '"$OUTPUT_DIR"

echo "Extraction complete!"
```

## Progress Monitoring

**With progress bar:**
```bash
# Count total tars
TOTAL=$(ls sa_*.tar | wc -l)

# Extract with progress
ls sa_*.tar | \
  parallel -j 8 --bar --eta 'tar -xf {} -C ./extracted' 2>&1 | \
  pv -l -s "$TOTAL" > /dev/null
```

**Monitor disk usage during extraction:**
```bash
# Watch disk space in real-time
watch -n 5 'df -h /mnt/data'

# Or with loop
while true; do
  echo "$(date): $(du -sh ./sa1b_extracted)"
  sleep 60
done
```

## Optimizations

**Use pigz for faster decompression:**
```bash
# Install pigz (parallel gzip)
sudo apt-get install pigz

# Extract with pigz (2-3× faster)
tar -I pigz -xf sa_000000.tar
```

**Parallel + pigz combined:**
```bash
ls sa_*.tar | parallel -j 4 'tar -I pigz -xf {} -C ./extracted'
```

## Error Handling

**Check for extraction errors:**
```bash
for tar in sa_*.tar; do
  tar -tzf "$tar" > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "ERROR: $tar is corrupted"
  fi
done
```

**Resume extraction after failure:**
```bash
# Extract only tars that haven't been extracted yet
for tar in sa_*.tar; do
  dir_name=$(basename "$tar" .tar)
  if [ ! -d "./extracted/$dir_name" ]; then
    echo "Extracting $tar"
    tar -xf "$tar" -C ./extracted
  fi
done
```

## ARR-COC Workflow

**Quick subset extraction:**
```bash
# Extract first 10 tars for prototyping
ls sa_00000[0-9].tar | parallel -j 4 'tar -xf {}'
# ~2 minutes on 4-core machine
```

**Production extraction:**
```bash
# Extract all 1,000 tars on 16-core machine
ls sa_*.tar | parallel -j 16 --bar 'tar -I pigz -xf {} -C /mnt/fast-ssd/sa1b'
# ~1.5-2 hours total
```

**Sources**: GNU Parallel documentation, tar manual
