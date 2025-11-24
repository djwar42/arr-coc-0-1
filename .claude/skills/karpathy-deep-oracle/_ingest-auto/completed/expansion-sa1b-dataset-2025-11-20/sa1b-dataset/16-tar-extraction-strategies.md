# Tar Extraction: Sequential vs Parallel

## 1. Sequential Extraction
```bash
for tar in sa_*.tar; do
  tar -xf "$tar"
done
```
Time: ~30-40 hours for full dataset

## 2. Parallel Extraction
```bash
ls sa_*.tar | parallel -j8 'tar -xf {}'
```
Time: ~6-8 hours with 8 cores

## 3. ARR-COC-0-1 (10%)
Use parallel extraction (4-8 cores) for training subset.
