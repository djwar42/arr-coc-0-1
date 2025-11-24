# MAX_JOBS Pattern - Keep This Default!

## Pattern Used in All 4 Images

```dockerfile
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}
```

## Why This Matters

**The `ARG MAX_JOBS=4` is a REQUIRED default.**

- Docker **requires** ARG to have a default value
- The `=4` is NOT what gets used - it's just the fallback
- The **actual value** comes from `${MAX_JOBS}` substitution at build time
- Build passes `--build-arg MAX_JOBS=176` which overrides the default

## DO NOT REMOVE THE `=4` DEFAULT!

Build will fail without it. The default is required by Docker ARG syntax.

## Images Using This Pattern

1. `arr-pytorch-base` - Canonical reference (lines 371-372)
2. `arr-ml-stack` - Fixed to match (lines 152-153)
3. `arr-trainer` - Fixed to match (lines 68-69)
4. `arr-vertex-launcher` - Fixed to match (lines 61-62)

All images consistently use ARG+ENV pattern for dynamic MAX_JOBS override.
