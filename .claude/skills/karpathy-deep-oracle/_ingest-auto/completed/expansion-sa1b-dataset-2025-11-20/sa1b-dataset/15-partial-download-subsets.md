# Partial Download Strategies: Subsets

## 1. Subset Selection
- **10 tars:** ~100GB (110k images, 11M masks) - Quick experiments
- **50 tars:** ~500GB (550k images, 55M masks) - Development
- **100 tars:** ~1TB (1.1M images, 110M masks) - Small-scale training

## 2. Representative Sampling
```bash
# Every 10th tar (100 total)
for i in $(seq 0 10 990); do
  idx=$(printf "%06d" $i)
  wget https://dl.fbaipublicfiles.com/.../sa_$idx.tar
done
```

## 3. ARR-COC-0-1 (10%)
Use 10-tar subset for ARR-COC spatial grounding experiments.
