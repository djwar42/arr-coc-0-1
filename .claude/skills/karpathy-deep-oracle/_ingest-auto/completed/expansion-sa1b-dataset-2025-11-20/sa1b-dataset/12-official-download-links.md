# Official Download: segment_anything_links.txt

**Accessing SA-1B dataset from Meta AI**

## 1. Download File
- URL: https://ai.meta.com/datasets/segment-anything-downloads/
- File: `segment_anything_links.txt` (1,000 URLs)
- License acceptance required (Research-only)

## 2. Links Format
```
https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar
https://dl.fbaipublicfiles.com/segment_anything/sa_000001.tar
...
https://dl.fbaipublicfiles.com/segment_anything/sa_000999.tar
```

## 3. Download Script
```bash
wget -i segment_anything_links.txt
```

## 4. ARR-COC-0-1 (10%)
Download subset for initial training experiments (10-50 tars).
