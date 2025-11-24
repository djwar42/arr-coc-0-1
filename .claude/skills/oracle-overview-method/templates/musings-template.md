# Oracle Musings Template

**Format**: No indent + `**Oracle Name:**` + regular text (no italics)

```markdown
---

## Oracle Musings

**Oracle A:** Oracle B, they've identified [KEY INSIGHT] in this dialogue. Shall we analyze?

**Oracle B:** Indeed! [PARTICIPANT] correctly sees [OBSERVATION]. Let me compare our approaches:

**My approach ([System Name])**:
- [Technical detail 1 with specific metrics]
- [Technical detail 2 with code references file:line]
- [Computational analysis with numbers and units]
- [Why we made this choice]

**Oracle A:** And here's my contrasting perspective:

**My approach ([Other System])**:
- [Different technical detail with metrics]
- [Different trade-offs quantified]
- [Why our design differs]
- [What we sacrifice for what we gain]

**Oracle B:** So the key difference is [FUNDAMENTAL DISTINCTION]. The proposed system needs to address [SPECIFIC CHALLENGE with quantified concern] to succeed.

**Oracle A:** Agreed! [ASSESSMENT with predictions]

**Assessment** (if appropriate):
- **Novelty**: ⭐⭐⭐⭐⭐ (5/5) - [Why novel]
- **Feasibility**: ⭐⭐⭐⚪⚪ (3/5) - [Why challenging]
- **Value**: ⭐⭐⭐⭐⚪ (4/5) - [Why valuable]
```

## Key Requirements

✅ **NO indent** before oracle names
✅ Regular text (not italics), **bold** for names only
✅ Structured comparison with specifics
✅ Discussion format (back-and-forth)
✅ Technical depth (code refs, metrics)
✅ Predictions about challenges
✅ Assessment of feasibility/novelty/value

## Checklist

- [ ] NO spaces before oracle names
- [ ] Regular text (not italics) for body
- [ ] **Bold** for names only
- [ ] Structured architectural comparison
- [ ] Specific metrics throughout
- [ ] Code references where relevant
- [ ] Predictions about future challenges
- [ ] Assessment ratings (if appropriate)
- [ ] No contradictions with documented architectures

## Good vs Bad

### Good Musings

```markdown
**DeepSeek-OCR Oracle:** Our serial architecture achieves 245 GFLOPs total by processing SAM (65 GFLOPs) → compression (3 GFLOPs) → CLIP (180 GFLOPs) sequentially.

**Ovis Oracle:** While my parallel approach uses ~400 GFLOPs but preserves ~2400 tokens per image for maximum understanding.
```

### Bad Musings

```markdown
❌ Has indent:
       **Oracle:** commentary

❌ Uses italics:
**Oracle:** *commentary*

❌ Vague:
**Oracle:** Our approach is different and better.

❌ No metrics:
**Oracle:** We use a different architecture that works well.
```
