# Verification & Integrity Checks

## 1. Checksum Verification
Meta provides checksums in `segment_anything_checksums.txt`
```bash
md5sum -c segment_anything_checksums.txt
```

## 2. Tar Integrity Check
```bash
for tar in sa_*.tar; do
  tar -tzf "$tar" > /dev/null && echo "$tar OK" || echo "$tar CORRUPT"
done
```

## 3. JSON Validation
```python
import json
for json_file in Path("sa_000000").glob("*.json"):
    try:
        json.load(open(json_file))
    except:
        print(f"Corrupt: {json_file}")
```

## 4. ARR-COC-0-1 (10%)
Run integrity checks before starting training runs.
