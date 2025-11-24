# Verification & Checksums for SA-1B

## Overview

**Data integrity verification** is critical for multi-day downloads and distributed training. Checksums detect corrupted files before wasting GPU hours on bad data.

**Verification methods:**
- MD5/SHA256 checksums (if provided by Meta)
- Tar integrity check with `tar -tzf`
- JSON schema validation
- Image file header validation

## Tar Integrity Verification

**Basic tar check:**
```bash
# Verify tar can be read (lists contents without errors)
tar -tzf sa_000000.tar > /dev/null && echo "OK" || echo "CORRUPTED"
```

**Batch verification script:**
```bash
#!/bin/bash
# verify_tars.sh

FAILED=0
TOTAL=0

for tar in sa_*.tar; do
  TOTAL=$((TOTAL + 1))
  if tar -tzf "$tar" > /dev/null 2>&1; then
    echo "✓ $tar"
  else
    echo "✗ $tar CORRUPTED"
    FAILED=$((FAILED + 1))
    echo "$tar" >> failed_tars.txt
  fi
done

echo "Verification complete: $((TOTAL - FAILED))/$TOTAL OK, $FAILED failed"
```

**Parallel verification:**
```bash
ls sa_*.tar | parallel -j 8 \
  'tar -tzf {} > /dev/null 2>&1 && echo "✓ {}" || echo "✗ {} CORRUPTED"'
```

## Checksum Verification

**Compute checksums for future verification:**
```bash
# Generate MD5 checksums for all tars
md5sum sa_*.tar > checksums.md5

# Verify later
md5sum -c checksums.md5
```

**SHA256 (more secure):**
```bash
sha256sum sa_*.tar > checksums.sha256
sha256sum -c checksums.sha256
```

## JSON Schema Validation

**Validate annotation structure:**
```python
import json
from pathlib import Path

def validate_annotation(json_path: Path) -> bool:
    """
    Validate SA-1B annotation JSON structure.

    Expected schema:
    {
      "image": {"image_id": int, "width": int, "height": int, ...},
      "annotations": [
        {
          "id": int,
          "segmentation": {"size": [H, W], "counts": [...]},
          "area": int,
          "bbox": [x, y, w, h],
          "predicted_iou": float,
          "point_coords": [[x, y], ...],
          ...
        }
      ]
    }
    """
    try:
        with open(json_path) as f:
            data = json.load(f)

        # Check required top-level keys
        assert 'image' in data
        assert 'annotations' in data

        # Check image metadata
        img = data['image']
        assert all(k in img for k in ['image_id', 'width', 'height'])

        # Check each annotation
        for ann in data['annotations']:
            assert 'segmentation' in ann
            assert 'predicted_iou' in ann
            assert 'area' in ann
            assert 'bbox' in ann

            # Validate RLE structure
            rle = ann['segmentation']
            assert 'size' in rle and 'counts' in rle
            assert len(rle['size']) == 2
            assert isinstance(rle['counts'], list)

        return True

    except (json.JSONDecodeError, AssertionError, KeyError) as e:
        print(f"Validation failed for {json_path}: {e}")
        return False

# Validate all annotations
corrupted_files = []
for json_file in Path("./sa1b_extracted").rglob("*.json"):
    if not validate_annotation(json_file):
        corrupted_files.append(json_file)

print(f"Found {len(corrupted_files)} corrupted annotation files")
```

## Image File Validation

**Check JPEG headers:**
```python
from PIL import Image
from pathlib import Path

def validate_image(image_path: Path) -> bool:
    """Validate JPEG can be opened."""
    try:
        img = Image.open(image_path)
        img.verify()  # Check integrity
        return True
    except Exception as e:
        print(f"Corrupted: {image_path}: {e}")
        return False

# Validate all images
for img_path in Path("./sa1b_extracted").rglob("*.jpg"):
    if not validate_image(img_path):
        # Re-download or skip
        pass
```

## Re-download Corrupted Files

**Automatic re-download:**
```bash
# Read failed tars from verification
while read tar; do
  echo "Re-downloading $tar"
  rm "$tar"
  tar_num=$(echo "$tar" | grep -oP '\\d+')
  wget "https://dl.fbaipublicfiles.com/segment_anything/sa_${tar_num}.tar"
done < failed_tars.txt
```

## ARR-COC Integration

**Pre-training validation:**
```bash
# Before starting training, verify all data
./verify_tars.sh
./validate_annotations.py
./check_images.py

# Only proceed if all checks pass
if [ $? -eq 0 ]; then
  python train_arr_coc.py --data ./sa1b_extracted
fi
```

**Sources**: Data integrity best practices, tar/gzip documentation
