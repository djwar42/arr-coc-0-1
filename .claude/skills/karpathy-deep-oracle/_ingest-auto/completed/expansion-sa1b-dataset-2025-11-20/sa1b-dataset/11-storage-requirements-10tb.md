# Storage Requirements: ~10TB Uncompressed

**Planning storage for SA-1B dataset: compressed vs uncompressed sizes**

## 1. Total Dataset Size
- **Compressed (tars):** ~9-10TB (1,000 tar files × ~10GB each)
- **Uncompressed:** ~11-12TB (images + JSONs extracted)

## 2. Per-Tar Breakdown
- Images: ~9-9.5GB per tar (~11k JPEGs)
- JSONs: ~500-700MB per tar (~11k annotation files)
- Total per tar: ~10GB compressed, ~11GB uncompressed

## 3. Storage Recommendations
- **Full dataset:** 12-15TB SSD recommended
- **Subset (100 tars):** 1-1.2TB
- **Streaming:** Can process without full extraction

## 4. ARR-COC-0-1 (10%)
Use selective tar extraction for spatial grounding training subset.
