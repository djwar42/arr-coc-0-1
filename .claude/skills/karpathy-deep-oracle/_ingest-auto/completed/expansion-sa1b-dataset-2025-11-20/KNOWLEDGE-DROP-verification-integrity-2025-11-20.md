# KNOWLEDGE DROP: SA-1B Verification & Integrity Checking

**Date**: 2025-11-20
**Runner**: PART 18 (Verification & Integrity)
**Status**: COMPLETE

---

## Overview

This knowledge drop covers verification and integrity checking strategies for SA-1B downloads, including checksum verification, file count validation, JSON parsing tests, image loading tests, mask decoding tests, and corruption recovery procedures.

---

## 1. Checksum Verification

### Generate and Verify Checksums

```python
"""
Checksum generation and verification for SA-1B files.
"""

import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def calculate_checksum(
    file_path: str,
    algorithm: str = 'md5',
    chunk_size: int = 1024 * 1024
) -> str:
    """
    Calculate checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256, sha1)
        chunk_size: Read chunk size

    Returns:
        Hex digest of hash
    """
    if algorithm == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256()
    elif algorithm == 'sha1':
        hash_obj = hashlib.sha1()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def generate_checksums(
    directory: str,
    output_file: str,
    algorithm: str = 'md5',
    num_workers: int = 4
):
    """
    Generate checksums for all files in directory.

    Args:
        directory: Directory to process
        output_file: Output checksum file
        algorithm: Hash algorithm
        num_workers: Parallel workers
    """
    # Find all tar files
    files = []
    for f in os.listdir(directory):
        if f.endswith('.tar'):
            files.append(os.path.join(directory, f))

    print(f"Generating {algorithm} checksums for {len(files)} files...")

    checksums = {}

    def process_file(file_path):
        filename = os.path.basename(file_path)
        checksum = calculate_checksum(file_path, algorithm)
        return filename, checksum

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_file, files),
            total=len(files),
            desc="Checksums"
        ))

    for filename, checksum in results:
        checksums[filename] = checksum

    # Save checksums
    with open(output_file, 'w') as f:
        for filename in sorted(checksums.keys()):
            f.write(f"{checksums[filename]}  {filename}\n")

    print(f"Saved checksums to {output_file}")


def verify_checksums(
    directory: str,
    checksum_file: str,
    algorithm: str = 'md5'
) -> dict:
    """
    Verify files against checksum file.

    Args:
        directory: Directory with files
        checksum_file: File with expected checksums
        algorithm: Hash algorithm

    Returns:
        Verification results
    """
    results = {
        'verified': [],
        'failed': [],
        'missing': []
    }

    # Load expected checksums
    expected = {}
    with open(checksum_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                checksum = parts[0]
                filename = parts[-1]
                expected[filename] = checksum

    print(f"Verifying {len(expected)} files...")

    for filename, expected_checksum in tqdm(expected.items(), desc="Verifying"):
        file_path = os.path.join(directory, filename)

        if not os.path.exists(file_path):
            results['missing'].append(filename)
            continue

        actual_checksum = calculate_checksum(file_path, algorithm)

        if actual_checksum == expected_checksum:
            results['verified'].append(filename)
        else:
            results['failed'].append({
                'filename': filename,
                'expected': expected_checksum,
                'actual': actual_checksum
            })

    # Report
    print(f"\nVerification Results:")
    print(f"  Verified: {len(results['verified'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Missing: {len(results['missing'])}")

    return results


# Example usage
if __name__ == "__main__":
    # Generate checksums
    generate_checksums('./raw', 'sa1b_checksums.md5', algorithm='md5')

    # Verify
    results = verify_checksums('./raw', 'sa1b_checksums.md5')
```

---

## 2. File Count Validation

### Validate Extracted Files

```python
"""
File count validation for SA-1B dataset.
"""

import os
import json


class FileCountValidator:
    """
    Validate file counts for SA-1B dataset.
    """

    # Expected counts per tar file (approximate)
    EXPECTED_PER_TAR = {
        'images': 11000,  # ~11K images per tar
        'masks': 11000    # 1 mask file per image
    }

    def __init__(self, data_dir: str):
        """
        Initialize validator.

        Args:
            data_dir: Directory with extracted data
        """
        self.data_dir = data_dir

    def count_files_by_type(self) -> dict:
        """
        Count files by type in the dataset.

        Returns:
            Dictionary with file counts
        """
        counts = {
            'jpg': 0,
            'json': 0,
            'other': 0,
            'directories': 0
        }

        for root, dirs, files in os.walk(self.data_dir):
            counts['directories'] += len(dirs)

            for f in files:
                if f.endswith('.jpg'):
                    counts['jpg'] += 1
                elif f.endswith('.json'):
                    counts['json'] += 1
                else:
                    counts['other'] += 1

        return counts

    def validate_tar_extraction(self, tar_name: str) -> dict:
        """
        Validate extraction of a single tar file.

        Args:
            tar_name: Name of tar file (e.g., 'sa_000000')

        Returns:
            Validation results
        """
        results = {
            'tar_name': tar_name,
            'valid': True,
            'images': 0,
            'masks': 0,
            'issues': []
        }

        # Find files from this tar
        prefix = tar_name.replace('.tar', '')

        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.startswith('sa_') and prefix in root:
                    if f.endswith('.jpg'):
                        results['images'] += 1
                    elif f.endswith('.json'):
                        results['masks'] += 1

        # Check counts
        if results['images'] == 0:
            results['valid'] = False
            results['issues'].append('No images found')

        if results['masks'] == 0:
            results['valid'] = False
            results['issues'].append('No mask files found')

        if results['images'] != results['masks']:
            results['issues'].append(
                f"Image/mask count mismatch: {results['images']} vs {results['masks']}"
            )

        return results

    def validate_all(self) -> dict:
        """
        Validate entire dataset.

        Returns:
            Overall validation results
        """
        counts = self.count_files_by_type()

        results = {
            'valid': True,
            'total_images': counts['jpg'],
            'total_masks': counts['json'],
            'issues': []
        }

        # Check image/mask balance
        if counts['jpg'] != counts['json']:
            results['issues'].append(
                f"Global image/mask mismatch: {counts['jpg']} images vs {counts['json']} masks"
            )

        # Check minimum counts (full dataset should have ~11M images)
        if counts['jpg'] < 1000:
            results['issues'].append(
                f"Very few images: {counts['jpg']} (expected thousands)"
            )
            results['valid'] = False

        # Report
        print("Dataset Validation Results")
        print("=" * 40)
        print(f"Total images: {counts['jpg']:,}")
        print(f"Total masks: {counts['json']:,}")

        if results['issues']:
            print("\nIssues found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        else:
            print("\nNo issues found!")

        return results


# Example usage
if __name__ == "__main__":
    validator = FileCountValidator('./extracted')
    results = validator.validate_all()
```

---

## 3. JSON Parsing Tests

### Validate Mask Annotations

```python
"""
JSON parsing and validation for SA-1B mask annotations.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def validate_mask_json(json_path: str) -> dict:
    """
    Validate a single mask annotation JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Validation results
    """
    results = {
        'path': json_path,
        'valid': True,
        'issues': [],
        'num_masks': 0
    }

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Check required fields
        if 'image' not in data:
            results['issues'].append('Missing "image" field')
            results['valid'] = False

        if 'annotations' not in data:
            results['issues'].append('Missing "annotations" field')
            results['valid'] = False
        else:
            results['num_masks'] = len(data['annotations'])

            # Validate each annotation
            for i, ann in enumerate(data['annotations']):
                if 'segmentation' not in ann:
                    results['issues'].append(f'Annotation {i}: missing segmentation')
                    results['valid'] = False

                if 'bbox' not in ann:
                    results['issues'].append(f'Annotation {i}: missing bbox')

        # Check image info
        if 'image' in data:
            img_info = data['image']
            required = ['image_id', 'width', 'height', 'file_name']
            for field in required:
                if field not in img_info:
                    results['issues'].append(f'Image info missing: {field}')

    except json.JSONDecodeError as e:
        results['valid'] = False
        results['issues'].append(f'JSON parse error: {e}')

    except Exception as e:
        results['valid'] = False
        results['issues'].append(f'Error: {e}')

    return results


def validate_all_jsons(
    masks_dir: str,
    num_workers: int = 4,
    sample_size: int = None
) -> dict:
    """
    Validate all mask JSON files.

    Args:
        masks_dir: Directory with JSON files
        num_workers: Parallel workers
        sample_size: Number of files to sample (None = all)

    Returns:
        Validation summary
    """
    # Find all JSON files
    json_files = []
    for root, dirs, files in os.walk(masks_dir):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))

    # Sample if requested
    if sample_size and sample_size < len(json_files):
        import random
        json_files = random.sample(json_files, sample_size)

    print(f"Validating {len(json_files)} JSON files...")

    # Validate in parallel
    valid_count = 0
    invalid_files = []
    total_masks = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(validate_mask_json, json_files),
            total=len(json_files),
            desc="Validating"
        ))

    for result in results:
        if result['valid']:
            valid_count += 1
        else:
            invalid_files.append(result)
        total_masks += result['num_masks']

    # Summary
    summary = {
        'total_files': len(json_files),
        'valid_files': valid_count,
        'invalid_files': len(invalid_files),
        'total_masks': total_masks,
        'invalid_details': invalid_files[:10]  # First 10
    }

    print(f"\nJSON Validation Summary:")
    print(f"  Valid: {valid_count}/{len(json_files)}")
    print(f"  Total masks: {total_masks:,}")

    if invalid_files:
        print(f"\nInvalid files ({len(invalid_files)}):")
        for detail in invalid_files[:5]:
            print(f"  {os.path.basename(detail['path'])}:")
            for issue in detail['issues'][:3]:
                print(f"    - {issue}")

    return summary


# Example usage
if __name__ == "__main__":
    summary = validate_all_jsons('./annotations', sample_size=1000)
```

---

## 4. Image Loading Tests

### Validate Image Files

```python
"""
Image loading and validation for SA-1B.
"""

import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def validate_image(image_path: str) -> dict:
    """
    Validate a single image file.

    Args:
        image_path: Path to image

    Returns:
        Validation results
    """
    results = {
        'path': image_path,
        'valid': True,
        'issues': [],
        'width': 0,
        'height': 0
    }

    try:
        # Open and verify image
        with Image.open(image_path) as img:
            # Verify integrity
            img.verify()

        # Re-open to get dimensions (verify() closes the file)
        with Image.open(image_path) as img:
            results['width'] = img.width
            results['height'] = img.height

            # Check reasonable dimensions
            if img.width < 100 or img.height < 100:
                results['issues'].append(f'Small image: {img.width}x{img.height}')

            if img.width > 10000 or img.height > 10000:
                results['issues'].append(f'Very large image: {img.width}x{img.height}')

            # Check format
            if img.format != 'JPEG':
                results['issues'].append(f'Unexpected format: {img.format}')

    except Exception as e:
        results['valid'] = False
        results['issues'].append(f'Error: {e}')

    return results


def validate_all_images(
    images_dir: str,
    num_workers: int = 4,
    sample_size: int = None
) -> dict:
    """
    Validate all images in directory.

    Args:
        images_dir: Directory with images
        num_workers: Parallel workers
        sample_size: Sample size (None = all)

    Returns:
        Validation summary
    """
    # Find all images
    image_files = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.endswith('.jpg'):
                image_files.append(os.path.join(root, f))

    # Sample if requested
    if sample_size and sample_size < len(image_files):
        import random
        image_files = random.sample(image_files, sample_size)

    print(f"Validating {len(image_files)} images...")

    # Validate in parallel
    valid_count = 0
    corrupt_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(validate_image, image_files),
            total=len(image_files),
            desc="Validating"
        ))

    for result in results:
        if result['valid']:
            valid_count += 1
        else:
            corrupt_files.append(result)

    # Summary
    summary = {
        'total_images': len(image_files),
        'valid_images': valid_count,
        'corrupt_images': len(corrupt_files),
        'corrupt_details': corrupt_files[:10]
    }

    print(f"\nImage Validation Summary:")
    print(f"  Valid: {valid_count}/{len(image_files)}")

    if corrupt_files:
        print(f"\nCorrupt images ({len(corrupt_files)}):")
        for detail in corrupt_files[:5]:
            print(f"  {os.path.basename(detail['path'])}: {detail['issues']}")

    return summary


# Example usage
if __name__ == "__main__":
    summary = validate_all_images('./images', sample_size=1000)
```

---

## 5. Mask Decoding Tests

### Validate RLE Mask Decoding

```python
"""
Mask decoding validation for SA-1B (COCO RLE format).
"""

import os
import json
import numpy as np
from pycocotools import mask as mask_utils


def decode_rle_mask(segmentation: dict, height: int, width: int) -> np.ndarray:
    """
    Decode COCO RLE segmentation to binary mask.

    Args:
        segmentation: RLE segmentation dict
        height: Image height
        width: Image width

    Returns:
        Binary mask as numpy array
    """
    if isinstance(segmentation, dict):
        # Already in RLE format
        rle = segmentation
    else:
        # Convert polygon to RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)

    # Decode to binary mask
    mask = mask_utils.decode(rle)

    return mask


def validate_mask_decoding(json_path: str) -> dict:
    """
    Validate mask decoding for a single annotation file.

    Args:
        json_path: Path to annotation JSON

    Returns:
        Validation results
    """
    results = {
        'path': json_path,
        'valid': True,
        'issues': [],
        'masks_decoded': 0,
        'masks_failed': 0
    }

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_info = data.get('image', {})
        height = image_info.get('height', 0)
        width = image_info.get('width', 0)

        if height == 0 or width == 0:
            results['issues'].append('Invalid image dimensions')
            results['valid'] = False
            return results

        annotations = data.get('annotations', [])

        for i, ann in enumerate(annotations):
            try:
                segmentation = ann.get('segmentation')
                if segmentation is None:
                    results['masks_failed'] += 1
                    continue

                # Decode mask
                mask = decode_rle_mask(segmentation, height, width)

                # Validate mask
                if mask.shape != (height, width):
                    results['issues'].append(
                        f'Mask {i}: wrong shape {mask.shape} vs ({height}, {width})'
                    )
                    results['masks_failed'] += 1
                else:
                    results['masks_decoded'] += 1

                    # Check mask is not empty or full
                    mask_sum = mask.sum()
                    total_pixels = height * width

                    if mask_sum == 0:
                        results['issues'].append(f'Mask {i}: empty mask')

                    if mask_sum == total_pixels:
                        results['issues'].append(f'Mask {i}: full mask')

            except Exception as e:
                results['masks_failed'] += 1
                results['issues'].append(f'Mask {i}: decode error - {e}')

    except Exception as e:
        results['valid'] = False
        results['issues'].append(f'Error: {e}')

    if results['masks_failed'] > 0:
        results['valid'] = False

    return results


def validate_mask_decoding_batch(
    masks_dir: str,
    sample_size: int = 100
) -> dict:
    """
    Validate mask decoding for sample of files.

    Args:
        masks_dir: Directory with mask JSON files
        sample_size: Number of files to test

    Returns:
        Validation summary
    """
    # Find JSON files
    json_files = []
    for root, dirs, files in os.walk(masks_dir):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))

    # Sample
    import random
    if sample_size < len(json_files):
        json_files = random.sample(json_files, sample_size)

    print(f"Testing mask decoding on {len(json_files)} files...")

    valid_count = 0
    total_masks = 0
    failed_masks = 0
    issues = []

    for json_path in json_files:
        result = validate_mask_decoding(json_path)

        if result['valid']:
            valid_count += 1

        total_masks += result['masks_decoded']
        failed_masks += result['masks_failed']

        if result['issues']:
            issues.extend(result['issues'][:3])

    # Summary
    summary = {
        'files_tested': len(json_files),
        'files_valid': valid_count,
        'masks_decoded': total_masks,
        'masks_failed': failed_masks,
        'sample_issues': issues[:10]
    }

    print(f"\nMask Decoding Summary:")
    print(f"  Files: {valid_count}/{len(json_files)} valid")
    print(f"  Masks decoded: {total_masks:,}")
    print(f"  Masks failed: {failed_masks}")

    return summary


# Example usage
if __name__ == "__main__":
    summary = validate_mask_decoding_batch('./annotations', sample_size=100)
```

---

## 6. Corruption Recovery

### Recovery Procedures

```python
"""
Corruption detection and recovery for SA-1B dataset.
"""

import os
import shutil
import json


class CorruptionRecovery:
    """
    Detect and recover from corrupted files in SA-1B.
    """

    def __init__(self, data_dir: str, raw_dir: str = None):
        """
        Initialize recovery manager.

        Args:
            data_dir: Extracted data directory
            raw_dir: Raw tar files directory (for re-extraction)
        """
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.recovery_log = os.path.join(data_dir, '.recovery_log.json')

    def scan_for_corruption(self) -> dict:
        """
        Scan dataset for corrupted files.

        Returns:
            Corruption scan results
        """
        results = {
            'corrupt_images': [],
            'corrupt_masks': [],
            'orphan_images': [],
            'orphan_masks': []
        }

        # Scan images
        print("Scanning for corrupt images...")
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.jpg'):
                    img_path = os.path.join(root, f)

                    # Try to load image
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception:
                        results['corrupt_images'].append(img_path)

                    # Check for matching mask
                    mask_name = f.replace('.jpg', '.json')
                    mask_path = os.path.join(root, mask_name)
                    if not os.path.exists(mask_path):
                        results['orphan_images'].append(img_path)

        # Scan masks
        print("Scanning for corrupt masks...")
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.json'):
                    json_path = os.path.join(root, f)

                    # Try to parse JSON
                    try:
                        with open(json_path, 'r') as fp:
                            json.load(fp)
                    except Exception:
                        results['corrupt_masks'].append(json_path)

                    # Check for matching image
                    img_name = f.replace('.json', '.jpg')
                    img_path = os.path.join(root, img_name)
                    if not os.path.exists(img_path):
                        results['orphan_masks'].append(json_path)

        return results

    def recover_from_tar(self, tar_index: int) -> bool:
        """
        Re-extract files from tar to recover corruption.

        Args:
            tar_index: Index of tar file (0-999)

        Returns:
            True if recovery successful
        """
        if not self.raw_dir:
            print("Cannot recover: raw_dir not specified")
            return False

        tar_name = f"sa_{tar_index:06d}.tar"
        tar_path = os.path.join(self.raw_dir, tar_name)

        if not os.path.exists(tar_path):
            print(f"Cannot recover: {tar_name} not found")
            return False

        print(f"Re-extracting {tar_name}...")

        import tarfile
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(self.data_dir)

        return True

    def remove_orphans(self, dry_run: bool = True) -> dict:
        """
        Remove orphan files (images without masks or vice versa).

        Args:
            dry_run: If True, only report what would be removed

        Returns:
            Removal summary
        """
        scan = self.scan_for_corruption()

        removed = {
            'images': [],
            'masks': []
        }

        for img_path in scan['orphan_images']:
            if dry_run:
                print(f"Would remove orphan image: {img_path}")
            else:
                os.remove(img_path)
                removed['images'].append(img_path)

        for mask_path in scan['orphan_masks']:
            if dry_run:
                print(f"Would remove orphan mask: {mask_path}")
            else:
                os.remove(mask_path)
                removed['masks'].append(mask_path)

        return removed

    def generate_recovery_report(self, output_path: str):
        """
        Generate detailed corruption recovery report.

        Args:
            output_path: Output file path
        """
        scan = self.scan_for_corruption()

        report = {
            'summary': {
                'corrupt_images': len(scan['corrupt_images']),
                'corrupt_masks': len(scan['corrupt_masks']),
                'orphan_images': len(scan['orphan_images']),
                'orphan_masks': len(scan['orphan_masks'])
            },
            'details': scan
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Recovery report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    recovery = CorruptionRecovery('./extracted', './raw')

    # Scan for corruption
    results = recovery.scan_for_corruption()

    print("\nCorruption Scan Results:")
    print(f"  Corrupt images: {len(results['corrupt_images'])}")
    print(f"  Corrupt masks: {len(results['corrupt_masks'])}")
    print(f"  Orphan images: {len(results['orphan_images'])}")
    print(f"  Orphan masks: {len(results['orphan_masks'])}")

    # Generate report
    recovery.generate_recovery_report('recovery_report.json')
```

---

## 7. Complete Verification Script

```python
#!/usr/bin/env python3
"""
Complete SA-1B dataset verification script.
"""

import os
import argparse
import json
from datetime import datetime


def run_full_verification(
    data_dir: str,
    output_report: str,
    sample_size: int = 1000
):
    """
    Run complete verification suite.

    Args:
        data_dir: Dataset directory
        output_report: Output report path
        sample_size: Sample size for statistical tests
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': data_dir,
        'tests': {}
    }

    print("SA-1B Dataset Verification")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Sample size: {sample_size}")
    print()

    # 1. File count validation
    print("1. File Count Validation...")
    validator = FileCountValidator(data_dir)
    count_results = validator.validate_all()
    report['tests']['file_counts'] = count_results
    print()

    # 2. JSON validation
    print("2. JSON Parsing Validation...")
    json_results = validate_all_jsons(
        os.path.join(data_dir, 'annotations'),
        sample_size=sample_size
    )
    report['tests']['json_parsing'] = json_results
    print()

    # 3. Image validation
    print("3. Image Loading Validation...")
    image_results = validate_all_images(
        os.path.join(data_dir, 'images'),
        sample_size=sample_size
    )
    report['tests']['image_loading'] = image_results
    print()

    # 4. Mask decoding
    print("4. Mask Decoding Validation...")
    mask_results = validate_mask_decoding_batch(
        os.path.join(data_dir, 'annotations'),
        sample_size=min(sample_size, 100)
    )
    report['tests']['mask_decoding'] = mask_results
    print()

    # Overall result
    all_passed = (
        count_results.get('valid', False) and
        json_results.get('invalid_files', 0) == 0 and
        image_results.get('corrupt_images', 0) == 0 and
        mask_results.get('masks_failed', 0) == 0
    )

    report['overall_result'] = 'PASS' if all_passed else 'FAIL'

    # Save report
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=2)

    print("=" * 50)
    print(f"Overall Result: {report['overall_result']}")
    print(f"Report saved to: {output_report}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Verify SA-1B dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Dataset directory')
    parser.add_argument('--output', type=str, default='verification_report.json',
                       help='Output report path')
    parser.add_argument('--sample', type=int, default=1000,
                       help='Sample size for statistical tests')

    args = parser.parse_args()

    run_full_verification(
        args.data_dir,
        args.output,
        args.sample
    )


if __name__ == "__main__":
    main()
```

---

## Sources

**Documentation:**
- [Python hashlib](https://docs.python.org/3/library/hashlib.html) - Checksum algorithms
- [Pillow/PIL](https://pillow.readthedocs.io/) - Image loading and verification
- [pycocotools](https://github.com/cocodataset/cocoapi) - COCO RLE decoding

**Best Practices:**
- [Vanderbilt CDS - File Integrity](https://cds.vanderbilt.edu/faq/check-file-integrity-md5) - MD5 verification guide (accessed 2025-11-20)
- [NinjaOne - Checksums](https://www.ninjaone.com/blog/what-is-a-checksum/) - Checksum overview (accessed 2025-11-20)

**Community Resources:**
- [Stack Overflow](https://stackoverflow.com/) - Verification techniques
- [Reddit r/DataHoarder](https://www.reddit.com/r/DataHoarder/) - Data integrity discussions

---

## Summary

This completes the SA-1B Download & Extraction knowledge acquisition series (PARTs 13-18):

- **PART 13**: Official download links and access
- **PART 14**: Parallel downloader tools
- **PART 15**: Download time optimization
- **PART 16**: Partial download and subsets
- **PART 17**: Tar extraction strategies
- **PART 18**: Verification and integrity

All knowledge drops include Python code examples, bash scripts, and citations to sources.
