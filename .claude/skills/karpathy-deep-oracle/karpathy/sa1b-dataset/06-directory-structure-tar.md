# KNOWLEDGE DROP: SA-1B Directory Structure & Tar Organization

**Date**: 2025-11-20 15:55
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 7)
**File Created**: `sa1b-dataset/06-directory-structure-tar.md`

---

## What Was Created

**Knowledge File**: Directory Structure & Tar Organization (~700 lines)

**8 Sections**:
1. Hierarchical Directory Structure (top-level: 1,000 tar files)
2. Individual Tar File Structure (internal: ~11,000 images+JSONs per tar)
3. File Naming Conventions (tar, image, annotation patterns)
4. Tar File Contents Breakdown (~10-11GB per tar)
5. Standalone Tar Design Philosophy (self-contained, parallel-friendly)
6. Extraction and Storage Patterns (full, selective, extract-process-delete)
7. Directory Traversal Patterns (Python code examples)
8. **ARR-COC-0-1** (10%): Efficient tar-based data pipeline for spatial grounding

---

## Key Insights

### 1,000 Tar Files = Perfect Distribution

**Structure**:
- `sa_000000.tar` to `sa_000999.tar`
- Each tar: ~11,000 images + ~11,000 JSON annotations
- Each tar: ~10-11 GB compressed, self-contained
- **Total**: 11M images, 1.1B masks, ~10TB uncompressed

### Standalone Design = Maximum Flexibility

**Each tar is fully independent**:
- Download subset (first 10 tars = 110K images for experiments)
- Parallel processing (different GPUs on different tars)
- Stream from tar without full extraction (saves 10TB disk space!)
- Resumable downloads (only re-download failed tars)

### File Pairing = Simple Access

**Every image has matching JSON**:
- `sa_1.jpg` + `sa_1.json`
- `sa_42.jpg` + `sa_42.json`
- Same base filename = easy programmatic pairing

---

## Research Performed

**Web sources consulted**:
1. GitHub Issue #26 (1000 tar files, 11GB each, standalone confirmation)
2. Hugging Face Forums (10G per tar, 10T total, practical loading advice)
3. Meta AI official downloads (10.5GB per tar confirmation)
4. TensorFlow Datasets catalog (structure documentation)
5. GitHub SA-1B-Downloader (parallel download implementation)

**Source document**:
- SAM_DATASET_SA1B.md (lines 94-123: directory layout, tar organization)

---

## ARR-COC-0-1 Integration (10%)

### Stream-Based Training Pipeline

**Efficient approach**:
```python
# Process tar files sequentially without full extraction
for tar_idx in range(50):  # Start with 50 tars = 550K images
    with tarfile.open(f"sa_{tar_idx:06d}.tar", 'r') as tar:
        for image, annotation in stream_pairs(tar):
            train_spatial_relevance(image, annotation)
```

**Benefits**:
- **Memory efficient**: No 10TB extraction needed
- **Subset training**: 10 tars (110K images) for initial experiments
- **Parallel GPUs**: Different tar files on different GPUs
- **Progressive curriculum**: Start small, scale to 1,000 tars

**Relevance realization**:
- SA-1B's 100+ masks/image = multi-scale spatial attention
- Tar-based subset = validate spatial grounding before scaling
- Class-agnostic masks = pure spatial relevance learning

---

## Why This Structure Matters

### For Research

**Subset experiments**:
- Download first 10 tars (~100 GB) instead of 10 TB
- Validate approach on 1% of data before full-scale training
- Incrementally add more data as experiments progress

### For Storage

**Disk-efficient patterns**:
- Extract-process-delete: Only 11 GB needed at a time
- Stream from tar: Zero extraction, process in-memory
- Selective download: Only download tars you need

### For ARR-COC Training

**Spatial grounding pipeline**:
- Start: 10 tars (110K images, ~100 GB)
- Validate: Spatial relevance realization working
- Scale: Add 50 tars (550K images, ~500 GB)
- Production: Full 1,000 tars (11M images, ~10 TB)

**Curriculum learning enabled by tar structure**.

---

## Statistics

- **Lines**: ~700 lines
- **Code examples**: 7 (directory traversal, extraction patterns, streaming)
- **Sections**: 8 (7 technical + 1 ARR-COC integration at 10%)
- **Web sources**: 5 cited with URLs and access dates
- **Source document**: 1 cited with line numbers
- **Completion time**: ~45 minutes (research + writing + review)

---

## Next Steps

**PART 8** will cover: Image Files (JPG Format & Resolution)
- JPEG specifications for SA-1B
- Variable resolution (1500×2250 average)
- RGB color space details
- Privacy protection (face/plate blurring)

**Estimated**: ~700 lines, 3 web searches, 1 source section
