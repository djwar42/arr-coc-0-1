# I/O Optimization & Throughput

**Maximizing data loading throughput for SA-1B training**

## DataLoader Optimization
```python
DataLoader(dataset, 
           batch_size=32,
           num_workers=8,      # 8 CPU cores
           pin_memory=True,    # GPU transfer
           prefetch_factor=2)  # Prefetch 2 batches
```

## Storage: SSD vs HDD
- SSD: 500-3000 MB/s (recommended)
- HDD: 100-200 MB/s (too slow)

## ARR-COC-0-1 (10%)
Use SSD storage + 8 workers for SA-1B spatial grounding training.
