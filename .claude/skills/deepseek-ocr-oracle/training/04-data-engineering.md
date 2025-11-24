# Data Engineering

**See**: `RESEARCH/DeepSeekOCR/TRAINING.md` lines 550-800

## OCR 1.0 Data (53M samples)

**Document OCR**:
- 30M pages with coarse + fine annotations
- PDF documents, scans, screenshots
- Multiple languages

**Scene OCR**:
- 20M images (Chinese + English)
- Street signs, posters, natural scenes

**Word Documents**:
- 3M converted Word docs
- Preserves layout structure

## OCR 2.0 Data (16M samples)

**Charts** (10M):
- pyecharts generated
- matplotlib plots
- Bar, line, scatter, pie charts

**Chemical Formulas** (5M):
- SMILES notation → images
- Molecular structures

**Geometry** (1M):
- Plane geometry figures
- Annotated with descriptions

## General Vision (100M)

**LAION subset**:
- Image-caption pairs
- Broad visual coverage
- Semantic understanding

## Data Quality

- Filtering: Remove low-quality, duplicates
- Balancing: Oversample difficult examples
- Augmentation: Rotation, scaling, noise

**Total**: ~260M samples (130M × 2 epochs)

**See TRAINING.md** for complete data pipeline details!
