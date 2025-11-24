# Forgetting Mechanism

## Concept

**Progressive compression** of older contexts to simulate biological memory decay.

## Implementation Strategy

```
Recent context (< 1 min):   Large mode (421 tokens) - Clear memory
Medium-term (1 hour):       Base mode (273 tokens) - Fading
Long-term (1 day):          Small mode (111 tokens) - Distant
Very old (1 week+):         Tiny mode (73 tokens) - Almost gone
```

## Biological Parallel

**Human memory**:
- Recent events: High detail, vivid
- Hours old: Some details lost
- Days old: General gist remains
- Weeks old: Minimal memory

**DeepSeek-OCR**:
- Recent: High resolution (many tokens)
- Older: Progressively lower resolution (fewer tokens)
- Very old: Minimal tokens (compressed heavily)

## Benefits

**Memory efficiency**:
- Don't waste tokens on old content
- Focus budget on recent/important info

**Context window optimization**:
- Fit more history in fixed context
- 10× more documents with progressive compression

## Example

**4-hour conversation with documents**:

```
Document 1 (just now):     421 tokens (Large)
Document 2 (10 min ago):   273 tokens (Base)
Document 3 (1 hour ago):   111 tokens (Small)
Document 4 (3 hours ago):   73 tokens (Tiny)

Total: 878 tokens vs 1684 if all Large (48% savings!)
```

## Implementation

```python
def get_resolution_for_age(document_age_seconds):
    if age_seconds < 60:
        return "large"    # 421 tokens
    elif age_seconds < 3600:
        return "base"     # 273 tokens
    elif age_seconds < 86400:
        return "small"    # 111 tokens
    else:
        return "tiny"     # 73 tokens
```

## Relation to Optical Compression

**Spatial compression**: 16× inside SAM (always)
**Temporal compression**: Progressive resolution reduction (over time)

**Combined**: Massive context efficiency!

## Paper Reference

DeepSeek-OCR paper Figure 13 shows progressive downsampling for older contexts.

**See Also**:
- [optical-compression.md](optical-compression.md) - Spatial compression
- [token-budgets.md](token-budgets.md) - Resolution modes
