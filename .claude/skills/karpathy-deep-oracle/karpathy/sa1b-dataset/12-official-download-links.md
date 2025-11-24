# SA-1B Official Download: segment_anything_links.txt

## Overview

SA-1B is distributed via a single file `segment_anything_links.txt` containing 1,000 URLs to tar files. Each tar contains ~11,000 images + annotations.

**Download process:**
1. Get links file: `wget https://segment-anything.com/segment_anything_links.txt`
2. Download tars: `wget -i segment_anything_links.txt`
3. Extract: `tar -xf sa_000000.tar`

## Links File Format

**Format**: Plain text, one URL per line

```
https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar
https://dl.fbaipublicfiles.com/segment_anything/sa_000001.tar
...
https://dl.fbaipublicfiles.com/segment_anything/sa_000999.tar
```

**Total**: 1,000 tar files

## Download Commands

**Full dataset** (all 1,000 tars):
```bash
wget https://segment-anything.com/segment_anything_links.txt
wget -i segment_anything_links.txt --continue --progress=bar --tries=10
```

**Partial download** (first 10 tars):
```bash
head -10 segment_anything_links.txt > subset.txt
wget -i subset.txt
```

**Parallel download** (4 concurrent):
```bash
cat segment_anything_links.txt | xargs -n 1 -P 4 wget --continue
```

## Verification

**Check completeness**:
```bash
# Count downloaded tars
ls sa_*.tar | wc -l
# Should be 1,000 for full dataset

# Verify tar integrity
for tar in sa_*.tar; do
  tar -tzf $tar > /dev/null && echo "$tar OK" || echo "$tar CORRUPTED"
done
```

## ARR-COC Application

**Subset selection for quick experimentation:**
- Download 10 tars (~110k images) for prototyping
- Scale to full dataset for production training
- Use GCS/S3 mirror for faster downloads from cloud VMs

**Sources**: [segment-anything.com/dataset](https://segment-anything.com/dataset)
