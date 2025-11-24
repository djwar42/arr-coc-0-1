# Parallel Downloader Tools

**Accelerating SA-1B download with parallel connections**

## 1. SA-1B-Downloader (GitHub)
```bash
git clone https://github.com/erow/SA-1B-Downloader
python download.py --processes 4 --start 0 --end 100
```

## 2. aria2c (Fastest)
```bash
aria2c -i segment_anything_links.txt -j 8 -x 16
```
- `-j 8`: 8 parallel downloads
- `-x 16`: 16 connections per file

## 3. Performance
- Single process: 2-3 days
- 4 processes: ~18 hours
- 8 processes: ~12 hours (with good bandwidth)

## 4. ARR-COC-0-1 (10%)
Use 4-8 processes for balanced download speed + bandwidth.
