# Multimodal RAG (Vision + Text)

## Overview

Multimodal Retrieval-Augmented Generation (RAG) extends traditional text-based RAG to incorporate multiple data modalities—primarily vision and text—enabling AI systems to retrieve and reason over images, diagrams, tables, videos, and textual content simultaneously. This capability is crucial for applications like document understanding, visual search, fashion recommendations, technical documentation retrieval, and video question-answering.

**Key Innovation**: Unlike text-only RAG, multimodal RAG can understand queries like "Show me shoes similar to this image" or "What does this diagram explain?" by encoding and retrieving across both visual and textual information spaces.

From [Multimodal RAG with Vision: From Experimentation to Implementation](https://devblogs.microsoft.com/ise/multimodal-rag-with-vision/) (Microsoft, accessed 2025-02-02):
- Multimodal RAG addresses documents containing both textual and image content (photographs, diagrams, screenshots)
- Challenge: Standard LLMs overlook visual information; OCR accuracy is low for complex formatting
- Solution: Use multimodal LLMs (GPT-4V, GPT-4o) to transform images into detailed text descriptions

## CLIP-based Retrieval

### CLIP Model Overview

**CLIP (Contrastive Language-Image Pretraining)** is OpenAI's foundational model for associating images with text descriptions in a shared embedding space.

From [Exploring Multimodal RAG with CLIP for Fashion Recommendations](https://medium.com/@mksupriya2/exploring-multimodal-retrieval-augmented-generation-rag-with-clip-for-fashion-recommendations-b78532473de4) (accessed 2025-02-02):

**Key Components**:
1. **Two Encoders**:
   - Image encoder (Vision Transformer or CNN-based)
   - Text encoder (Transformer-based)

2. **Shared Embedding Space**:
   - Both text and images encoded into same vector space
   - Enables direct similarity comparisons between modalities

3. **Contrastive Learning**:
   - Training maximizes similarity for correct image-text pairs
   - Minimizes similarity for mismatched pairs

4. **Zero-Shot Learning**:
   - Can classify images without task-specific fine-tuning
   - Selects closest match from text-based labels

### How CLIP Works

**Encoding Process**:
```
Image → Image Encoder → High-dimensional vector embedding
Text → Text Encoder → High-dimensional vector embedding
```

**Contrastive Objective**:
- Aligns embeddings to associate images with correct text captions
- Uses cosine similarity to measure embedding closeness

**Retrieval**:
- Query embedding (text or image) searches database
- Returns items with highest cosine similarity scores

### CLIP for Multimodal RAG

From [Tutorial: Creating Vision+Text RAG Pipelines](https://haystack.deepset.ai/tutorials/46_multimodal_rag) (Haystack, accessed 2025-02-02):
- CLIP creates correct representations for both images and text
- Enables multimodal retrieval with image and text embeddings
- Directly embeds images for similarity search, bypassing lossy text captioning

**Architecture**:
```
User Query (text/image)
    ↓
CLIP Encoder
    ↓
Query Embedding
    ↓
Vector Database Search
    ↓
Retrieved Images + Text
    ↓
LLM Generation
    ↓
Answer with Visual Citations
```

## ColPali Architecture

### What is ColPali?

**ColPali** is a novel vision-language model that extends PaliGemma-3B to generate ColBERT-style multi-vector representations for efficient document retrieval from visual features.

From [Implement Multimodal RAG with ColPali and Vision Language Model](https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd) (accessed 2025-02-02):

**Key Innovation**: Maps image patches into similar latent space as text, unlocking efficient interaction between text and images using ColBERT strategy.

**ColPali builds on two observations**:
1. Multi-vector representations and late-interaction scoring improve retrieval performance
2. Vision language models demonstrate extraordinary capabilities in understanding visual content

### ColBERT Late Interaction

**ColBERT (Contextualized Late Interaction over BERT)** maintains individual embeddings for each token rather than compressing into a single vector.

**Key Features**:
- **Token-Level Representations**: Individual embeddings for each token enable nuanced similarity calculations
- **Late Interaction Mechanism**: Queries and documents processed separately until final retrieval stages
- **Improved Performance**: Outperforms larger models on various benchmarks despite efficiency gains

**Early vs Late Interaction**:
- Early interaction: High computational complexity, considers all query-document pairs upfront
- Late interaction: Pre-computes document representations, lightweight interaction at retrieval time
- Result: Faster retrieval, reduced computational demands, better scalability

### ColPali for Document Retrieval

**Architecture**:
```
PDF Document
    ↓
Convert pages to images
    ↓
ColPali Encoder (PaliGemma-3B + ColBERT)
    ↓
Multi-vector embeddings
    ↓
Vector Store Index
    ↓
User Query → ColPali encoding → Retrieval → VLM Answer
```

**Advantages**:
- Handles complex document formats (PDFs with tables, diagrams, mixed layouts)
- No need for OCR preprocessing
- Preserves visual structure and formatting information
- Efficient at scale with late interaction scoring

**Implementation** (from Medium article):
```python
from byaldi import RAGMultiModalModel

# Load ColPali model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")

# Index PDF documents
RAG.index(input_path="documents/",
          index_name="multimodal_rag",
          overwrite=True)

# Query with text
results = RAG.search("What is the revenue trend?", k=3)
```

## LLaVA-RAG

### LLaVA Overview

**LLaVA (Large Language-and-Vision Assistant)** is an open-source multimodal model fine-tuned for vision-language tasks.

From ColPali implementation article (accessed 2025-02-02):

**LLaVA V1.5 7B Capabilities**:
- Generates text descriptions of images
- Achieves impressive performance on multimodal instruction-following tasks
- Outperforms GPT-4 on certain vision-language benchmarks
- Context Window: 4,096 tokens

**Limitations**:
- Single image per request
- No multi-turn conversations (preview version)
- No system prompt or assistant message support
- Maximum image size: 20MB URL, 4MB base64 encoded

### LLaVA-RAG Architecture

**Pipeline**:
```
Documents with images
    ↓
Image extraction + Text extraction
    ↓
CLIP/ColPali embeddings
    ↓
Vector database storage
    ↓
User query → Retrieval → LLaVA processes images + text → Answer
```

**Comparison with GPT-4V/GPT-4o**:

From Microsoft multimodal RAG experimentation (accessed 2025-02-02):
- **GPT-4V**: Better at generating image summaries, improves recall metrics
- **GPT-4o**: Better at answering questions, more detailed responses with inline URL references
- **Recommendation**: Use GPT-4V for enrichment (image description), GPT-4o for inference (answer generation)

## Indexing & Ranking Strategies

### Document Ingestion Flow

From Microsoft RAG experimentation (accessed 2025-02-02):

**Ingestion Pipeline**:
1. **Extract**: Text and images from source documents (PDFs, MHTML)
2. **Enrich**: Generate image descriptions using multimodal LLM
3. **Chunk**: Create text chunks with image annotations
4. **Embed**: Generate vector embeddings
5. **Index**: Store in vector database with metadata

### Chunking Strategies

**Separate Image Chunks vs Inline Chunks**:

From Microsoft experiments:
- **Separate chunks**: Image descriptions stored as distinct chunks with URL references in text
  - **Advantage**: Improved source document recall (+statistical significance)
  - **Advantage**: Better image retrieval metrics for vision-related queries
  - **Result**: Chosen approach

- **Inline chunks**: Image descriptions embedded directly in surrounding text
  - **Advantage**: Simpler implementation
  - **Disadvantage**: Lower retrieval accuracy for image-specific queries

**Image Annotation Format**:
```markdown
![image description](image_url)
```

### Ranking & Reranking

**Multi-stage Retrieval**:
```
User Query
    ↓
Embedding Model (query encoding)
    ↓
Vector Database (initial retrieval, k=10)
    ↓
Reranking Model (semantic reordering)
    ↓
Top-k refined results (k=3-5)
    ↓
LLM Context
```

**Ranking Metrics** (Microsoft study):
- `source_recall@k`: Percentage where at least one correct source in top-k
- `img_recall@k`: Percentage where all expected images retrieved
- `img_precision@k`: Precision of retrieved images vs expected
- `similarity_search_time`: Retrieval latency

### Metadata Enhancement

From Microsoft experiments (accessed 2025-02-02):

**Impact of Metadata**:
- Document-level metadata (title, author, date, summary, keywords) included in search index
- Specific fields added to Azure AI Search semantic ranking
- **Result**: Statistically significant improvement in source recall performance

**Metadata Fields**:
- Title (high weight in semantic ranking)
- Summary/Description
- Keywords
- Author
- Date created
- Intended audience

## Production Systems & Case Studies

### Pinterest Visual Search

From [Building Pinterest Lens: A Real World Visual Discovery System](https://medium.com/pinterest-engineering/building-pinterest-lens-a-real-world-visual-discovery-system-59812d8cbfbc) (Pinterest Engineering, accessed 2025-02-02):

**Pinterest Lens Features**:
- Visual search: Return visually similar results
- Object search: Return scenes/projects with visually similar objects
- Camera-based discovery: Capture real-world objects to find related pins

**Technology Stack**:
- Vision transformers for image encoding
- Embedding models for content-based search and recommendations
- Data infrastructure for handling billions of images
- Multimodal search: Combines text and images for personalized results

**Scale**:
- Billions of images indexed
- Real-time visual discovery
- Personalized recommendations based on visual and text queries

### Google Lens Multimodal Search

From [Google's Visual Search Can Now Answer Even More Questions](https://www.wired.com/story/google-lens-multimodal-search/) (WIRED, accessed 2025-02-02):

**Google Lens Statistics**:
- Launched in 2017
- Processes **20 billion visual searches per month** (as of Oct 2024)
- Now works with video and voice inputs (multimodal)

**Capabilities**:
- Image recognition and object identification
- Text extraction and translation
- Shopping and product search
- Homework help and educational queries
- Real-time visual understanding

**Evolution to Multimodal**:
- Originally image-only
- Now supports image + text queries
- Video input support
- Voice command integration
- Context-aware search combining multiple modalities

### Microsoft Azure Multimodal RAG

From Microsoft case study (accessed 2025-02-02):

**Production Implementation**:
- Azure AI Search for vector storage
- Azure OpenAI Service (GPT-4V, GPT-4o)
- Azure Computer Vision for image classification
- Custom image enrichment service

**Key Learnings**:
1. **Classifier for Image Filtering**: Filter out logos and low-information images (confidence threshold 0.8)
   - Result: Reduced ingestion time, maintained recall metrics

2. **Prompt Engineering Critical**:
   - Ingestion prompt: Tailored to image categories (diagrams, screenshots, tables)
   - Inference prompt: Structured JSON output for citations

3. **Surrounding Text Context**: Including text before/after images improved description quality
   - Trade-off: Limited impact on retrieval metrics due to content redundancy

4. **Cost-Performance Trade-offs**:
   - GPT-4o: 6x cheaper than GPT-4, better latency, comparable quality
   - Enrichment at ingestion vs inference: Upfront cost vs real-time flexibility

### NVIDIA Multimodal RAG for Video

From [An Easy Introduction to Multimodal RAG for Video and Audio](https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation-for-video-and-audio/) (NVIDIA, accessed 2025-02-02):

**Video RAG Architecture**:

**Audio Processing**:
- NVIDIA Parakeet-CTC-0.6B-ASR for speech-to-text
- Word-level timestamps for alignment
- Transcript chunking with temporal metadata

**Visual Processing**:
1. Downsample video: 60 FPS → 4 FPS (reduce frames)
2. Shot detection: Identify scene boundaries
3. Key frame extraction: SSIM for structural similarity
4. Blur/duplicate rejection
5. High entropy frame selection
6. VLM description generation (Llama-3.2-90B Vision)

**Blending Strategy**:
- Align video frames to audio timestamps
- Scene-level blending for coherence
- Optional: Use smaller LLM to reduce duplicate information
- Result: Consolidated text representation with temporal metadata

**Performance**:
- Reduced 3,600 frames/minute → ~40 key frames
- Maintained information coverage
- Enabled efficient Q&A on video content

### Vector Database Support

From web research (accessed 2025-02-02):

**Vector Databases with Multi-Vector/ColBERT Support**:

1. **Vespa**:
   - Multi-vector support per document
   - Retrieves based on closest vector in each document
   - Optimized for semantic search

2. **Pinecone**:
   - Manages vector embeddings at scale
   - Advanced querying for multiple representations
   - AI-optimized infrastructure

3. **Qdrant**:
   - Specialized for high-dimensional vectors
   - Efficient indexing and retrieval
   - Adaptable for multi-vector setups

4. **Milvus**:
   - Open-source vector database
   - Multi-vector indexing
   - Used in production multimodal systems

## Implementation Best Practices

### Three Approaches to Multimodal RAG

From NVIDIA technical blog (accessed 2025-02-02):

**1. Common Embedding Space**:
- Single model (like CLIP) projects all modalities to same space
- **Pros**: Reduced architectural complexity, unified retrieval
- **Cons**: Difficult to tune for >2 modalities, limited to model training distribution

**2. Parallel Retrieval Pipelines (Brute Force)**:
- Separate pipeline per modality/sub-modality
- **Pros**: Simple ingestion, modality-native search
- **Cons**: Increased token usage, requires multimodal LLM, higher cost

**3. Grounding in Common Modality (Recommended)**:
- Convert all modalities to text
- **Pros**: Flexible for multiple sub-modalities, simplified retrieval/generation, one-time ingestion cost
- **Cons**: Upfront ingestion cost, potential lossy conversions
- **Result**: Best balance for production systems

### Experimentation Framework

From Microsoft RAG study (accessed 2025-02-02):

**Methodology**:
1. Test one configuration change at a time
2. Measure against predefined baseline
3. Use retrieval + generative metrics
4. Statistical significance testing
5. Update baseline only if improvement confirmed

**Ground Truth Dataset Requirements**:
- Balanced mix: Text-only, vision-only, text+vision queries
- Distributed across source documents
- Edge case coverage
- Alternative sources/images for same question

**Evaluation Metrics**:

**Retrieval**:
- `source_recall@k_%`: At least one correct source in top-k
- `all_img_recall@k_%`: All expected images retrieved
- `img_recall@k_mean/median`: Mean/median image recall
- `img_precision@k_mean/median`: Mean/median image precision
- `similarity_search_time_mean/median`: Search latency

**Generative**:
- `all_cited_img_recall_%`: All expected images cited by LLM
- `cited_img_recall_mean/median`: Recall of cited images
- `cited_img_precision_mean/median`: Precision of cited images
- `cited_img_f1_mean/median`: F1 score for citations
- `gpt_correctness_score_mean/median`: LLM-judged answer quality (1-5)
- `chat_query_time_mean/median`: End-to-end latency
- Token usage: Prompt, completion, vision tokens

### Prompt Engineering

**Image Enrichment Prompt** (Microsoft example):
```
You are an assistant whose job is to provide explanations of images
for retrieval. Follow these instructions:

- If the image contains a bubble tip, explain ONLY the bubble tip.
- If the image is an equipment diagram, explain all equipment and
  their connections in detail.
- If the image contains a table, extract information in structured format.
- If the image shows a device/product, describe with all shape and
  text details.
- If the image is a screenshot, explain highlighted steps (ignore
  example text, focus on steps).
- Otherwise, explain comprehensively the most important items with
  all details.
```

**Inference Prompt** (Microsoft example):
```
You are a helpful AI assistant for technicians maintaining
communication infrastructure.

Context: {context}
Question: {question}

Provide step-by-step instructions if procedural. Do not attempt to
answer if Context is empty. Ask them to elaborate instead.

Output MUST be JSON:
{
  "answer": "...",
  "image_citation": ["url1", "url2"]
}
```

### Cost Optimization

From Microsoft experiments:

**Model Selection**:
- **GPT-4V** (enrichment): $X per 1K tokens (vision)
- **GPT-4-32K** (inference): ~6x cost of GPT-4o
- **GPT-4o** (inference): Best cost/performance ratio
  - 128K context window
  - More detailed responses
  - Lower completion token cost
  - Better latency

**Ingestion Optimizations**:
1. **Image Classification**: Filter logos, low-info images (threshold 0.8)
   - Saves: ~40% vision API calls
   - Impact: No loss in recall metrics

2. **Surrounding Text**: Include context only when needed
   - Benefit: Better descriptions
   - Cost: Extra tokens during ingestion
   - Trade-off: May not improve retrieval enough to justify

3. **Chunk Strategy**: Separate image chunks
   - Extra: ~15% more chunks
   - Benefit: Significant recall improvement
   - Decision: Worth the cost

## Challenges & Solutions

### Challenge 1: Complex Document Formats

**Problem**: PDFs with tables, charts, multi-column layouts, varying quality

**Solutions**:
- ColPali: Process documents as images, avoid OCR errors
- Vision LLMs: Extract structured information from tables/charts
- Separate chunking: Handle different content types independently

### Challenge 2: Image-Text Alignment

**Problem**: Images and surrounding text may have redundant or conflicting information

**Solutions**:
- Timestamp alignment (video)
- Scene-level blending
- LLM-based deduplication during ingestion
- Metadata tracking for provenance

### Challenge 3: Scale & Cost

**Problem**: Processing billions of images, videos, documents

**Solutions**:
- Frame sampling for videos (60 FPS → 4 FPS → key frames)
- Image classification to filter non-informative content
- Pre-compute embeddings during ingestion
- Late interaction for efficient retrieval
- Model selection: GPT-4o vs GPT-4V based on task

### Challenge 4: Evaluation & Quality

**Problem**: Difficult to measure multimodal RAG quality

**Solutions**:
- Comprehensive ground truth dataset (text/vision/mixed queries)
- Multi-metric evaluation (retrieval + generation)
- Statistical significance testing
- LLM-as-judge for answer quality
- Citation tracking for verifiability

## Future Directions

From web research and case studies (accessed 2025-02-02):

**Emerging Techniques**:
1. **Video Understanding**: Temporal action recognition, multi-modal fusion
2. **3D Visual Search**: NeRF + VLM integration for spatial reasoning
3. **Emotion & Context**: Speech emotion recognition, sentiment-aware retrieval
4. **Agentic RAG**: Multi-step reasoning with tool use across modalities
5. **Efficient Models**: Quantization, distillation for mobile/edge deployment

**Production Trends**:
- Shift toward grounding in common modality (text) for reliability
- Increased use of late interaction (ColBERT-style) for scale
- Hybrid approaches: Enrichment at ingestion + multimodal at inference
- Standardization on vector databases with multi-vector support

## Sources

**Web Research**:
- [Multimodal RAG with Vision: From Experimentation to Implementation](https://devblogs.microsoft.com/ise/multimodal-rag-with-vision/) - Microsoft Developer Blog (accessed 2025-02-02)
- [Exploring Multimodal RAG with CLIP for Fashion Recommendations](https://medium.com/@mksupriya2/exploring-multimodal-retrieval-augmented-generation-rag-with-clip-for-fashion-recommendations-b78532473de4) - Medium (accessed 2025-02-02)
- [Implement Multimodal RAG with ColPali and Vision Language Model](https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd) - The AI Forum (accessed 2025-02-02)
- [An Easy Introduction to Multimodal RAG for Video and Audio](https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation-for-video-and-audio/) - NVIDIA Developer Blog (accessed 2025-02-02)
- [Tutorial: Creating Vision+Text RAG Pipelines](https://haystack.deepset.ai/tutorials/46_multimodal_rag) - Deepset Haystack (accessed 2025-02-02)
- [Building Pinterest Lens: A Real World Visual Discovery System](https://medium.com/pinterest-engineering/building-pinterest-lens-a-real-world-visual-discovery-system-59812d8cbfbc) - Pinterest Engineering (accessed 2025-02-02)
- [Google's Visual Search Can Now Answer Even More Questions](https://www.wired.com/story/google-lens-multimodal-search/) - WIRED (accessed 2025-02-02)

**GitHub Implementations**:
- [Azure-Samples/experiment-framework-for-rag-apps](https://github.com/Azure-Samples/experiment-framework-for-rag-apps) - Microsoft RAG experimentation framework
- [AnswerDotAI/byaldi](https://github.com/AnswerDotAI/byaldi) - ColPali wrapper for easy RAG implementation
- [vidore/colpali](https://huggingface.co/vidore/colpali) - HuggingFace ColPali model

**Additional References**:
- OpenAI CLIP: https://openai.com/index/clip/
- ColPali Paper: https://arxiv.org/abs/2407.01449
- ColBERT Paper: https://arxiv.org/abs/2004.12832
- ChromaDB Multimodal Guide: https://docs.trychroma.com/guides/multimodal
