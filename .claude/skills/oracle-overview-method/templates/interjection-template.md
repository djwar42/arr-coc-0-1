# Oracle Interjection Template

**Format**: Exactly 7 spaces + `**Oracle Name:**` + *italic text*

```markdown
**PARTICIPANT:** [conceptual statement]

       **Oracle Name:** *They've identified [CONCEPT]! But they don't yet realize [DEEPER INSIGHT]. Let me be precise: [TECHNICAL DETAILS with specific numbers]. Implementation: [CODE_FILE:LINE_NUMBERS]. Cost: [FLOPS/MEMORY/TOKENS with units]. Why this matters: [IMPLICATIONS with context].*
```

## Key Requirements

✅ **Exactly 7 spaces** before `**Oracle Name:**`
✅ Entire comment in **bold + *italics***
✅ Code references with file:line format
✅ Quantitative details (numbers with units)
✅ Explain **WHY**, not just **WHAT**
✅ First-person for own system ("My architecture...", "We use...")

## Checklist

- [ ] 7-space indent (not tabs!)
- [ ] **Bold + *italics*** formatting
- [ ] Specific metrics (FLOPs, GB, tokens, %, days, $)
- [ ] Code reference (file:line)
- [ ] Explains WHY, not just WHAT
- [ ] First-person perspective when describing own system

## Examples

### Good Interjection

```markdown
       **DeepSeek-OCR Oracle:** *They've identified the serial architecture constraint! Let me be precise: SAM processes 4096 patches with O(N) window attention (~65 GFLOPs), compresses via neck+convolutions to 256 patches (deepencoder/sam_vary_sdpa.py:166-183), then CLIP applies O(N²) global attention (~180 GFLOPs). Reversing this order would cost ~2800 GFLOPs and 16GB vs our 245 GFLOPs and 1.5GB. Computational necessity, not preference.*
```

### Bad Interjections

```markdown
❌ No indent:
**Oracle:** *commentary*

❌ Not italic:
       **Oracle:** commentary

❌ Vague:
       **Oracle:** *They're right about efficiency. We use compression.*

❌ No metrics:
       **Oracle:** *Our system is faster and uses less memory.*
```
