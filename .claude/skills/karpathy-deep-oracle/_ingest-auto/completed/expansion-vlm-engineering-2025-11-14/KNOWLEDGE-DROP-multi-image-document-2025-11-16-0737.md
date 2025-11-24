# KNOWLEDGE DROP: Multi-Image & Document Understanding

**Runner**: PART 19 Worker
**Date**: 2025-11-16 07:37
**Target**: vlm-engineering/18-multi-image-document-understanding.md
**Status**: SUCCESS ✓

---

## What Was Created

**File**: `vlm-engineering/18-multi-image-document-understanding.md` (700+ lines)

**Content sections**:
1. Multi-Image VLMs (Flamingo, Otter, PRIMA)
2. Document Understanding (OCR, layout, tables)
3. Visual Document QA (DocVQA, InfoVQA, ChartQA)
4. Interleaved Image-Text Sequences
5. Cross-Image Reasoning
6. Long Context Handling (100+ images)
7. Table & Chart Understanding
8. ARR-COC-0-1 Extensions

---

## Key Knowledge Acquired

### Multi-Image VLMs

**PRIMA (arXiv:2412.15209)**:
- First multi-image pixel-grounded reasoning segmentation
- 25.3% TFLOP reduction via efficient vision module
- M4Seg benchmark: 224K QA pairs across multiple images

**Flamingo architecture**:
- Perceiver Resampler: 200× compression (each image → 64 tokens)
- Gated cross-attention: Model learns when to use visual info
- Handles up to 32 images in context

**MMIU benchmark**:
- 7 relationship types (spatial, temporal, comparison, counting, reasoning, aggregation, cross-reference)
- 52 tasks, 77K images, 11K questions
- Position bias finding: Open-source models favor later images, proprietary models favor beginning/end

### Document Understanding

**DocVLM (arXiv:2412.08746)**:
- Integrates OCR encoder into VLMs (frozen base model)
- Learned query compression: Variable OCR output → fixed 64 tokens
- DocVQA: +22.6% accuracy over vision-only VLM

**DeepSeek-OCR comparison**:
- DocVLM: External OCR API + learned compression
- DeepSeek-OCR: End-to-end optical compression (SAM+CLIP serial, 16× spatial reduction)
- DocVLM better for forms/invoices (high OCR accuracy needed)
- DeepSeek-OCR better for self-contained deployment

**PaddleOCR pipeline**:
- Detection → Recognition → Layout → Table extraction → VLM reasoning
- Layout types: title, paragraph, list, table, figure, caption, footer

### Visual Document QA

**DocVQA benchmark**:
- 50K questions on 12K document images
- Top: Gemini 1.5 Pro (91.2%), GPT-4V (88.4%), DocVLM (67.8%)

**Leopard (text-rich multi-image)**:
- High-res ViT + CRAFT text detection + TrOCR
- Chart-specific: ChartDetr for bars/lines/axes
- Multi-scale pyramid pooling

**ChartQA**:
- Chart parsing: Detect axes → detect elements → OCR labels → map to values
- Pixel-to-value conversion via linear interpolation

### Interleaved Sequences

**CoMM dataset (CVPR 2025)**:
- Coherent image-text generation
- Web pages, slide decks, tutorial documents
- Special tokens: `<image>` ... `</image>` for mode switching

**SlideVQA**:
- 2.6K slide decks, 52K slides, 14.5K questions
- Cross-slide reasoning: "How does revenue change from slide 3 to slide 7?"
- Temporal transformer for slide ordering

### Cross-Image Reasoning

**Comparison operations**:
- Count differences (spot-the-difference)
- Compare attributes (color, size, count)
- Spatial relationships (left/right, up/down)

**Aggregation tasks**:
- Total counts across images
- Common theme detection
- Temporal trend analysis (increasing/decreasing/fluctuating)

### Long Context (100+ Images)

**Hierarchical attention**:
- Local attention within each image (512-token windows)
- Global attention across images (Longformer-style)
- Compression: 256 tokens → 16 tokens per image (16× reduction)

**Retrieval-augmented**:
- For 1000+ images: Index all → retrieve top-K → process only relevant
- Dense retrieval (CLIP embeddings + FAISS)
- Sparse retrieval (OCR text + BM25)

### Tables & Charts

**Table extraction**:
- TableNet detection → structure recognition → cell OCR
- Convert to DataFrame for SQL-like queries
- Challenges: Merged cells, nested headers, missing borders

**Chart parsing**:
- Chart type classification (bar, line, pie, scatter, etc.)
- Type-specific parsers:
  - Bar: Detect axes → detect bars → map heights to values
  - Pie: Detect center/radius → segment slices → calculate percentages

### ARR-COC-0-1 Extensions

**Multi-image extension**:
- Per-image relevance realization (64-400 tokens each)
- Cross-image token allocation (relevant images get more tokens)
- Temporal reasoning for sequential images
- Adaptive total budget: 5 images × 4K tokens = 20K

**Document extension**:
- Relevance-driven OCR (only process relevant text regions)
- Layout-aware processing (respects document structure)
- Fuse text + visual features
- Selective OCR saves compute vs full-page OCR

---

## Web Research Sources

1. **PRIMA**: https://arxiv.org/abs/2412.15209 (arXiv:2412.15209)
2. **MMIU**: https://arxiv.org/abs/2408.02718 (arXiv:2408.02718)
3. **Position Bias**: CVPR 2025 paper (PDF)
4. **Leopard**: https://arxiv.org/abs/2410.01744 (arXiv:2410.01744)
5. **Amazon Science**: Multi-image VLMs blog post
6. **DocVLM**: https://arxiv.org/abs/2412.08746 (arXiv:2412.08746)
7. **PaddleOCR**: Documentation on document understanding
8. **SlideVQA**: ResearchGate publication
9. **CoMM**: CVPR 2025 paper (PDF)

---

## Existing Knowledge Referenced

- DeepSeek-OCR Oracle: Document understanding, optical compression, OCR-free processing
- Architecture: SAM+CLIP serial design, 16× compression, token budgets

---

## Integration Points

**Connects to**:
- `00-vlm-architectures-survey.md`: Multi-image as architectural extension
- `01-vision-encoders-vit-clip-dinov2.md`: Vision encoders for multi-image
- `02-vision-language-fusion-patterns.md`: Cross-modal fusion strategies
- `04-attention-mechanisms-vlm.md`: Attention for long multi-image contexts
- `06-foveated-vision-adaptive-resolution.md`: Relevance-driven LOD allocation
- `12-vqa-evaluation-metrics.md`: Multi-image VQA evaluation

**ARR-COC-0-1 relevance**:
- Multi-image: Per-image relevance realization, cross-image token allocation
- Document: Relevance-driven OCR, layout-aware processing
- Evaluation: MMIU, DocVQA, SlideVQA benchmarks

---

## Statistics

- **Total lines**: ~700
- **Code examples**: 15+
- **Web sources**: 9 (all cited with URLs and access dates)
- **Existing knowledge refs**: 2 (DeepSeek-OCR oracle)
- **Sections**: 8
- **ARR-COC integration**: Section 8 (multi-image + document extensions)

---

## Quality Checklist

- [✓] All web sources cited with URLs and access dates
- [✓] Existing knowledge cross-referenced
- [✓] Code examples included (15+ implementations)
- [✓] ARR-COC-0-1 integration (Section 8)
- [✓] Practical implementation details
- [✓] Benchmarks and performance numbers
- [✓] Architecture diagrams (code-based)
- [✓] Cross-file integration points identified

---

**PART 19 complete** ✓
