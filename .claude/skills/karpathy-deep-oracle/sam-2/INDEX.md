# SAM 2 - Index

**Segment Anything Model 2: Video + Image Segmentation**

Released: July 29, 2024 (Meta AI)
Updated: SAM 2.1 on October 18, 2024 (+2.8 J&F improvement)

## Quick Navigation

### Core Architecture
- `00-overview.md` - SAM 2 introduction and key innovations
- `01-hiera-encoder.md` - Hierarchical encoder (replaces ViT-H)
- `02-streaming-memory.md` - Memory attention mechanism for video
- `03-prompt-encoder-decoder.md` - Lightweight prompt processing

### Video Segmentation
- `04-video-architecture.md` - How SAM 2 handles video sequences
- `05-memory-bank.md` - Temporal memory management
- `06-occlusion-handling.md` - Dealing with objects leaving/entering frame

### SAM 2.1 Updates (October 2024)
- `07-sam21-updates.md` - **Complete SAM 2.1 changelog** (NEW!)
- `KNOWLEDGE-DROP-sam21-jf-improvement-2025-11-21.md` - +2.8 J&F boost
- `KNOWLEDGE-DROP-sam21-occlusion-handling-2025-11-21.md` - Enhanced memory
- `KNOWLEDGE-DROP-sam21-robustness-2025-11-21.md` - 3× fewer interactions
- `KNOWLEDGE-DROP-sam21-data-augmentation-2025-11-21.md` - Training data
- `KNOWLEDGE-DROP-sam21-training-code-2025-11-21.md` - Open training code
- `KNOWLEDGE-DROP-sam21-training-improvements-2025-11-21.md` - Training details
- `KNOWLEDGE-DROP-sam21-developer-suite-2025-11-21.md` - Developer tools
- `KNOWLEDGE-DROP-sam21-web-demo-code-2025-11-21.md` - Web demo (1,532 lines!)
- `KNOWLEDGE-DROP-sam21-architecture-changes-2025-11-21.md` - NO changes (same model!)
- `KNOWLEDGE-DROP-sam21-api-changes-2025-11-21.md` - 100% backward compatible

## What is SAM 2?

SAM 2 extends SAM 1's promptable segmentation to **video**, adding temporal consistency and object tracking.

### Key Features

1. **Unified Image + Video Model**: One model for both tasks
2. **Streaming Memory**: Efficient video processing without full sequence buffering
3. **Hiera Encoder**: 6× faster than ViT-H, better accuracy
4. **Real-time Performance**: 44 FPS on video (A100 GPU)
5. **Promptable**: Click, box, or mask prompts work on any frame

### SAM 2.1 Improvements (October 2024)

- **+2.8 J&F improvement**: 78.2% → 81.0% on SA-V dataset
- **Better occlusion handling**: Enhanced memory persistence
- **Training code released**: Fine-tune on custom data (8× A100)
- **Web demo code**: Full React/TypeScript + Flask stack
- **100% backward compatible**: Drop-in replacement for SAM 2

## Datasets

- **SA-V (Video)**: 50.9K videos, 642.6K masklets (annotations)
- **SA-1B (Image)**: Same as SAM 1 (11M images, 1.1B masks)

## Performance

| Benchmark | SAM 2 (July) | SAM 2.1 (Oct) | Improvement |
|-----------|--------------|---------------|-------------|
| SA-V      | 78.2% J&F    | **81.0% J&F** | **+2.8**    |
| MOSE      | 72.8% J&F    | **77.4% J&F** | **+4.6**    |
| LVOS v2   | 77.3% J&F    | **80.3% J&F** | **+3.0**    |

## Architecture Comparison

```
SAM 1 (Image Only)        SAM 2 (Video + Image)        SAM 2.1 (Better Training)
├─ ViT-H encoder          ├─ Hiera encoder (6× faster) ├─ Same Hiera encoder
├─ Prompt encoder         ├─ Prompt encoder            ├─ Same prompt encoder
├─ Mask decoder           ├─ Mask decoder              ├─ Same mask decoder
└─ [No temporal model]    ├─ Memory attention          ├─ Same memory attention
                          ├─ Memory encoder            ├─ Same memory encoder
                          └─ Memory bank               └─ Same memory bank

NO ARCHITECTURE CHANGES IN SAM 2.1!
Improvements from enhanced training only.
```

## Use Cases

### Video Segmentation
- Object tracking in videos
- Instance segmentation with temporal consistency
- Interactive video editing

### Image Segmentation
- All SAM 1 capabilities preserved
- Faster inference (6× speedup from Hiera)
- Better accuracy on complex scenes

## Developer Resources

- **GitHub**: https://github.com/facebookresearch/sam2
- **Demo**: https://sam2.metademolab.com
- **Paper**: https://arxiv.org/abs/2408.00714
- **Training Code**: Released October 2024 (SAM 2.1)
- **Web Demo Code**: React/TypeScript + Flask (SAM 2.1)

## Cross-References

- Related: `../sam-general/` (SAM 1 comparison)
- Related: `../karpathy/vision-language/` (Multimodal architectures)
- Related: `../implementations/` (GPU optimization)

---

**Last Updated**: November 21, 2025
**SAM 2.1 Integration**: 10 KNOWLEDGE-DROPs (7,257 lines) fully integrated
**Total Files**: 18 (7 core + 1 summary + 10 detailed SAM 2.1 docs)
