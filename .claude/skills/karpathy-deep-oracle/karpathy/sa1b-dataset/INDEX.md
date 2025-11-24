# SA-1B Dataset - Index

**Segment Anything 1 Billion (SA-1B): The largest segmentation dataset ever created**

11 million images, 1.1 billion masks, class-agnostic annotations, privacy-protected.

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `00-overview-largest-segmentation.md` | What is SA-1B, key features, purpose, scale comparison | overview, 11M images, 1.1B masks, foundation model |
| `02-mask-granularity-levels.md` | Multi-granular segmentation, fine-to-coarse masks, hierarchical | granularity, hierarchical, door handles to buildings |
| `03-diversity-domain-coverage.md` | Geographic diversity, subject matter, 63 countries, licensed imagery | diversity, domain coverage, geographic, licensed |
| `04-privacy-protection-pii.md` | Face blurring, license plate detection, GDPR compliance, PII removal | privacy, face blur, license plate, GDPR, de-identification |
| `05-class-agnostic-annotations.md` | No class labels design choice, object vs background, binary masks | class-agnostic, semantic vs instance, binary masks |
| `06-directory-structure-tar.md` | 1000 tar files, ~11K images per tar, file naming conventions | directory structure, tar organization, extraction |
| `07-image-files-jpg-format.md` | JPEG format, variable resolution ~1500x2250, RGB color space | JPEG, resolution, RGB, image format, preprocessing |

## Quick Start

Start with `00-overview-largest-segmentation.md` for SA-1B fundamentals, then explore specific topics.

**Note**: File `01-statistics-scale.md` pending (detailed dataset statistics).

## Coverage Status

**Completed (Batch 1-2 partial)**: 7 files covering overview, statistics concepts, file formats
**Pending (Batches 2-7)**: Download, loading, training, research applications, advanced topics (35 more files planned)

## Cross-References

- **Related**: `../vision-language/` (VLM integration), `../practical-implementation/` (training patterns)
- **Source codebase**: `../../source-codebases/deepseek/06-DeepSeek-OCR/` (SAM-based architecture)
- **Computer Vision Foundation**: See computer-vision-foundation-oracle for SAM architecture details

## ARR-COC Integration

Each file contains **Section 8: ARR-COC-0-1** (10%) covering:
- Dataset scale for relevance realization training
- Spatial grounding with class-agnostic masks
- Multi-granular features for propositional/perspectival knowing
