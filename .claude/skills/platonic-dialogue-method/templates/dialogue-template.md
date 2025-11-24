# Platonic Dialogue Template

Copy this template and fill in with your topic:

```markdown
# [Dialogue Title]: [Concept Being Explored]

**Participants**: Socrates, Theaetetus

[Optional: Brief context paragraph setting up the topic]

---

**SOCRATES:** [Opening question about the concept - what, why, or how]

**THEAETETUS:** [Initial understanding or observation with specific details]

**SOCRATES:** [Follow-up question that probes deeper - asks about mechanism or implication]

**THEAETETUS:** [Works through it] ...Ah! [Discovery moment - new insight]

**SOCRATES:** [Question that reveals a potential problem or trade-off]

**THEAETETUS:** [Recognizes the challenge] You're right, if we [do X], then [problem Y] occurs. [Attempts solution or identifies need]

**SOCRATES:** [Guides toward deeper understanding or alternative approach]

**THEAETETUS:** [Builds on previous insight with new understanding]

**SOCRATES:** [Question about comparison or trade-off]

**THEAETETUS:** [Explores options with specific metrics/details] [Option A] gives us [benefit] but costs [trade-off]. While [Option B] achieves [different benefit] at [different cost].

**SOCRATES:** [Synthesis question - "So what you're saying is..." or "How might we..."]

**THEAETETUS:** [Crystallized understanding or path forward - doesn't solve everything, but clear direction]

**SOCRATES:** [Final question about implications or next steps]

**THEAETETUS:** [Final insight - acknowledges what's known and what remains to explore]

```

## Checklist

Before finalizing your dialogue:

- [ ] Opens with Socrates asking a question (not stating)
- [ ] Theaetetus gives specific details (numbers, metrics, architectures)
- [ ] At least one "Ah!" or discovery moment
- [ ] Acknowledges at least one real challenge or trade-off
- [ ] Each exchange advances understanding (no repetition)
- [ ] Socrates guides through questions (doesn't lecture)
- [ ] Theaetetus discovers (doesn't instantly know)
- [ ] Ends with synthesis or path forward (not complete solution)
- [ ] Maintains consistent character voices
- [ ] Includes technical depth appropriate to topic

## Example Filled Template

**Topic**: Query-aware compression trade-offs

```markdown
# The Compression Dilemma

**Participants**: Socrates, Theaetetus

---

**SOCRATES:** What does CLIP need to process in our architecture?

**THEAETETUS:** CLIP processes visual tokens - in DeepSeek's base mode, it handles 257 tokens (256 from compressed grid plus CLS token).

**SOCRATES:** Why not process all 4096 patches directly?

**THEAETETUS:** The computational cost! CLIP uses O(N²) global attention. At 257 tokens that's ~180 GFLOPs. But at 4096 patches... Ah! That would be ~2800 GFLOPs - over 15× more expensive!

**SOCRATES:** So compression is necessary for efficiency. But what do we sacrifice?

**THEAETETUS:** You're right - if we compress too aggressively, we might lose critical details. A 16× compression that works for simple images might destroy intricate formulas or dense text.

**SOCRATES:** How might we balance these constraints?

**THEAETETUS:** Perhaps variable compression? Allocate more tokens to complex regions, fewer to simple backgrounds. Query-aware allocation based on what matters for the specific task.

**SOCRATES:** And what challenges would that create?

**THEAETETUS:** It adds complexity - we'd need to learn which regions are relevant for which queries. The compression itself becomes a learned process, not a fixed operation. But the potential gain... adaptive efficiency with preserved quality where it matters.
```
