# Multi-Image & Document Understanding

**Vision-language models for processing multiple images and complex documents**

---

## Overview

Multi-image and document understanding extends VLMs beyond single-image tasks to handle:
- **Multiple image inputs**: Comparing, relating, and reasoning across image sets
- **Document understanding**: OCR-free processing of text-rich visual content
- **Interleaved sequences**: Mixed image-text documents (slides, PDFs, web pages)
- **Cross-image reasoning**: Aggregating information from multiple sources
- **Long-context documents**: Charts, tables, forms, scientific papers

**Key challenge**: Traditional VLMs process one image at a time. Multi-image VLMs must understand relationships between images while document VLMs must parse layout, text, and visual elements simultaneously.

From [PRIMA: Multi-Image Vision-Language Models](https://arxiv.org/abs/2412.15209) (arXiv:2412.15209, accessed 2025-11-16):
- First multi-image pixel-grounded reasoning segmentation model
- Reduces TFLOPs by 25.3% through efficient cross-image vision module
- Introduces M4Seg benchmark: 224K question-answer pairs requiring fine-grained visual understanding across multiple images

From [DeepSeek-OCR Oracle](../../deepseek-ocr-oracle/concepts/00-optical-compression.md):
- Document understanding through 16× optical compression (SAM+CLIP serial architecture)
- OCR-free visual text recognition (73-421 tokens per image)
- Three-stage training: DeepEncoder → Full VLM → Gundam (high-res tiling)

---

## Section 1: Multi-Image VLMs (Flamingo, Otter, Multiple Inputs)

### Flamingo Architecture (DeepMind)

**Gated Cross-Attention for Multi-Image**:
```python
# Flamingo-style multi-image processing
class FlamingoMultiImage:
    def __init__(self):
        self.perceiver_resampler = PerceiverResampler(
            num_latents=64,  # Compress each image to 64 tokens
            num_layers=6
        )
        self.gated_xattn = GatedCrossAttention(
            dim=768,
            tanh_gating=True  # Adaptive integration
        )

    def forward(self, images, text_tokens):
        # Process multiple images independently
        image_features = []
        for img in images:
            vis_features = self.vision_encoder(img)  # [H*W, 768]
            compressed = self.perceiver_resampler(vis_features)  # [64, 768]
            image_features.append(compressed)

        # Interleave with text using gated cross-attention
        for layer in self.language_model.layers:
            # Standard self-attention on text
            text_out = layer.self_attn(text_tokens)

            # Cross-attention to relevant images (gated)
            for img_feat in image_features:
                cross_attn_out = self.gated_xattn(
                    query=text_out,
                    key_value=img_feat
                )
                # Tanh gating decides how much visual info to use
                text_out = text_out + cross_attn_out

        return text_out
```

**Key innovations**:
- **Perceiver Resampler**: Compresses each image to fixed 64 tokens (200× compression)
- **Gated cross-attention**: `tanh(alpha) * cross_attn_output` - model learns when to use visual info
- **Frozen vision encoder**: Pre-trained CLIP ViT, never updated
- **Interleaved training**: Image-text-image-text sequences (mimics web documents)

**Performance**:
- Handles up to 32 images in context
- 5-shot in-context learning on VQA (no fine-tuning)
- Flamingo-80B: 56.3% on VQAv2 zero-shot

### Otter (Multi-Image Instruction Following)

From [Amazon Science: Vision-Language Models for Multi-Image Inputs](https://www.amazon.science/blog/vision-language-models-that-can-handle-multi-image-inputs) (accessed 2025-11-16):
- Extends Flamingo with instruction tuning on multi-image tasks
- Attention-based representation improves downstream performance
- Training: MIMIC-IT dataset (2.8M multi-image instruction samples)

**Architecture differences**:
```python
# Otter vs Flamingo
class OtterMultiImage(FlamingoMultiImage):
    def __init__(self):
        super().__init__()
        # Add instruction-following adapter
        self.instruction_adapter = InstructionAdapter(
            num_instructions=1000,  # Common instruction embeddings
            dim=768
        )

    def forward(self, images, instruction, context=""):
        # Process instruction first
        inst_embed = self.instruction_adapter(instruction)

        # Combine with multi-image processing
        img_features = [self.process_image(img) for img in images]

        # Instruction-conditioned cross-attention
        output = self.language_model(
            context + inst_embed,
            cross_attn_inputs=img_features
        )
        return output
```

**Training improvements**:
- Instruction tuning: "Compare the two images and identify differences"
- Multi-turn dialogues: Follow-up questions about image sets
- Visual referring: "In the second image, what color is the car?"

### MMIU Benchmark (Multi-Modal Multi-Image Understanding)

From [MMIU: Multimodal Multi-Image Understanding](https://arxiv.org/abs/2408.02718) (arXiv:2408.02718, accessed 2025-11-16):
- 7 relationship types: spatial, temporal, comparison, counting, reasoning
- 52 tasks across relationships
- 77K images, 11K questions
- Evaluates: BLIP-2, LLaVA, Flamingo, GPT-4V

**Relationship taxonomy**:
1. **Spatial**: "Which object is leftmost across all images?"
2. **Temporal**: "What changes between image 1 and image 3?"
3. **Comparison**: "Which image has more people?"
4. **Counting**: "How many red cars total across all images?"
5. **Reasoning**: "Based on images 1-3, what happens next?"
6. **Aggregation**: "What is the common theme?"
7. **Cross-reference**: "Find the same object in both images"

**Position bias finding**:
From [Identifying Position Bias in Multi-Image VLMs](https://openaccess.thecvf.com/content/CVPR2025/papers/Tian_Identifying_and_Mitigating_Position_Bias_of_Multi-image_Vision-Language_Models_CVPR_2025_paper.pdf) (CVPR 2025, accessed 2025-11-16):
- Open-source models: Better performance on later images (recency bias)
- Proprietary models (GPT-4V): Better on beginning/end (primacy/recency)
- Mitigation: Position-aware training, image shuffling augmentation

---

## Section 2: Document Understanding (OCR, Layout, Tables)

### DocVLM (OCR-Integrated VLM)

From [DocVLM: Make Your VLM an Efficient Reader](https://arxiv.org/abs/2412.08746) (arXiv:2412.08746, accessed 2025-11-16):
- Integrates OCR-based modality into VLMs without retraining base weights
- OCR encoder captures text and layout → compresses into learned queries
- Enhances document processing while preserving original VLM capabilities

**Architecture**:
```python
# DocVLM OCR integration
class DocVLM:
    def __init__(self, base_vlm):
        self.vision_encoder = base_vlm.vision_encoder  # Frozen
        self.language_model = base_vlm.language_model  # Frozen

        # NEW: OCR encoder (trainable)
        self.ocr_encoder = OCREncoder(
            input_dim=768,  # From OCR API (e.g., Tesseract, PaddleOCR)
            output_dim=768
        )

        # NEW: Learned query compression
        self.ocr_queries = nn.Parameter(
            torch.randn(64, 768)  # 64 OCR summary tokens
        )

        self.ocr_cross_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12
        )

    def forward(self, image, ocr_boxes_and_text):
        # Standard vision path
        vis_features = self.vision_encoder(image)  # [HW, 768]

        # OCR path
        ocr_features = []
        for box, text in ocr_boxes_and_text:
            # Encode OCR text + bounding box position
            ocr_embed = self.ocr_encoder(text, box)
            ocr_features.append(ocr_embed)

        ocr_features = torch.stack(ocr_features)  # [N_boxes, 768]

        # Compress OCR with learned queries
        ocr_compressed = self.ocr_cross_attn(
            query=self.ocr_queries,  # [64, 768]
            key=ocr_features,
            value=ocr_features
        )  # [64, 768]

        # Combine visual + OCR
        combined = torch.cat([vis_features, ocr_compressed], dim=0)

        # Feed to language model
        return self.language_model(combined)
```

**Key benefits**:
- OCR text provides precise character recognition (better than pure vision)
- Bounding boxes give layout information (spatial relationships)
- Learned queries compress variable OCR output to fixed 64 tokens
- Frozen base model: No catastrophic forgetting of general vision skills

**Performance (DocVQA benchmark)**:
- Base VLM (no OCR): 45.2% accuracy
- DocVLM (+ OCR): 67.8% accuracy (+22.6 points)
- Human performance: ~90%

### DeepSeek-OCR (Optical Compression for Documents)

From [DeepSeek-OCR Oracle Architecture](../../deepseek-ocr-oracle/architecture/00-overview.md):
- **Serial SAM+CLIP**: 16× spatial compression before LLM
- **Token budgets**: 73 (low-res) to 421 (Gundam mode) tokens
- **Three-stage training**:
  1. DeepEncoder pre-training (frozen LLM)
  2. Full VLM training (end-to-end)
  3. Gundam fine-tuning (high-res 2×2 tiling)

**Comparison: DocVLM vs DeepSeek-OCR**:
| Feature | DocVLM | DeepSeek-OCR |
|---------|--------|--------------|
| **OCR method** | External API → learned compression | Implicit (SAM+CLIP vision) |
| **Token count** | Vision tokens + 64 OCR tokens | 73-421 total tokens |
| **Layout info** | Bounding boxes explicitly | Implicit in SAM masks |
| **Training** | Lightweight (OCR encoder only) | Heavy (3-stage, full model) |
| **Text accuracy** | High (uses OCR API) | Medium (learned from pixels) |
| **Deployment** | Requires OCR API | Self-contained |

**When to use**:
- **DocVLM**: High OCR accuracy needed (forms, invoices, legal docs)
- **DeepSeek-OCR**: End-to-end deployment, no external dependencies

### PaddleOCR Document Understanding Pipeline

From [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/doc_understanding.html) (accessed 2025-11-16):
- Complete document processing: Detection → Recognition → Layout → Understanding
- VLM integration for answering questions about documents
- Supports tables, forms, charts, multi-column layouts

**Pipeline stages**:
```python
# PaddleOCR + VLM pipeline
class DocumentUnderstandingPipeline:
    def __init__(self):
        self.text_detector = PPOCRv3Det()  # Detect text regions
        self.text_recognizer = PPOCRv3Rec()  # Recognize characters
        self.layout_analyzer = PPStructure()  # Parse layout
        self.table_recognizer = PPTableRec()  # Extract tables
        self.vlm = DocVLM()  # VLM for reasoning

    def process_document(self, document_image, question):
        # Stage 1: Text detection
        text_boxes = self.text_detector(document_image)

        # Stage 2: Text recognition
        ocr_results = []
        for box in text_boxes:
            text = self.text_recognizer(document_image[box])
            ocr_results.append((box, text))

        # Stage 3: Layout analysis
        layout = self.layout_analyzer(document_image)
        # Returns: title, paragraph, list, table, figure regions

        # Stage 4: Table extraction (if present)
        tables = []
        for region in layout['tables']:
            table_data = self.table_recognizer(document_image[region])
            tables.append(table_data)

        # Stage 5: VLM reasoning
        answer = self.vlm(
            image=document_image,
            ocr_boxes_and_text=ocr_results,
            layout=layout,
            tables=tables,
            question=question
        )
        return answer
```

**Layout types detected**:
- **Title**: Heading text (large font, bold)
- **Paragraph**: Body text blocks
- **List**: Bullet points, numbered items
- **Table**: Grid structures
- **Figure**: Charts, images, diagrams
- **Caption**: Figure/table descriptions
- **Footer**: Page numbers, metadata

---

## Section 3: Visual Document QA (DocVQA, InfoVQA, ChartQA)

### DocVQA Benchmark

**Dataset statistics**:
- 50K questions on 12K document images
- Documents: Scanned forms, reports, letters, invoices
- Question types: Extractive (85%), abstractive (15%)

**Example task**:
```
Image: Invoice with line items
Question: "What is the total amount due?"
Answer: "$1,247.53"

Requires: OCR + table parsing + math reasoning
```

**Top models (2024)**:
1. **Gemini 1.5 Pro**: 91.2% accuracy
2. **GPT-4V**: 88.4%
3. **DocVLM**: 67.8%
4. **LLaVA-1.5**: 42.1%
5. **BLIP-2**: 38.7%

**Why performance gap?**:
- OCR quality: Gemini/GPT-4V have superior text recognition
- Layout understanding: Proprietary models better at spatial reasoning
- Math reasoning: Extracting numbers and performing calculations

### InfoVQA (Infographics)

From [Leopard: Text-Rich Multi-Image VLM](https://arxiv.org/abs/2410.01744) (arXiv:2410.01744, accessed 2025-11-16):
- InfoVQA: Questions about infographics (charts, diagrams, timelines)
- Requires: Text reading + visual chart parsing + cross-element reasoning
- Leopard model: Tailored for text-rich multi-image tasks

**Infographic challenges**:
1. **Dense text**: Tiny labels, legends, annotations
2. **Visual encoding**: Bar heights, pie slices, line trends
3. **Cross-reference**: "Compare 2020 vs 2022 data"
4. **Multi-chart**: Multiple sub-charts in one infographic

**Leopard architecture**:
```python
# Leopard for text-rich multi-image
class Leopard:
    def __init__(self):
        self.vision_encoder = HighResViT(patch_size=14)  # Finer detail
        self.text_detector = CRAFT()  # Text region detection
        self.ocr_module = TrOCR()  # High-accuracy OCR

        # Multi-scale fusion
        self.pyramid_pooling = PPM(scales=[1, 2, 4, 8])

        # Chart-specific reasoning
        self.chart_parser = ChartDetr()  # Detect bars, lines, axes

    def forward(self, images, question):
        all_features = []
        for img in images:
            # High-res vision features
            vis_feat = self.vision_encoder(img)  # [HW, 768]

            # Detect and recognize text
            text_regions = self.text_detector(img)
            ocr_results = [self.ocr_module(img[r]) for r in text_regions]

            # Parse chart elements
            chart_elements = self.chart_parser(img)
            # Returns: bars, lines, axes, legends

            # Multi-scale fusion
            fused = self.pyramid_pooling(vis_feat)

            all_features.append({
                'visual': fused,
                'ocr': ocr_results,
                'chart': chart_elements
            })

        # Cross-image reasoning
        return self.answer_generator(all_features, question)
```

**Performance (InfoVQA)**:
- Leopard: 42.8% accuracy
- GPT-4V: 75.1%
- Human: 83.4%

### ChartQA (Chart Understanding)

**Question types**:
1. **Extractive**: "What is the value of the red bar in 2020?" → Read from chart
2. **Compositional**: "What is the sum of 2019 and 2020 sales?" → Extract + compute
3. **Comparison**: "Which category has the highest value?" → Compare across elements

**Chart parsing challenges**:
```python
# Chart element extraction
class ChartParser:
    def parse_bar_chart(self, image):
        # 1. Detect axes
        x_axis, y_axis = self.detect_axes(image)

        # 2. Detect bars
        bars = self.detect_bars(image)

        # 3. Read axis labels
        x_labels = [self.ocr(label) for label in x_axis.labels]
        y_values = [self.ocr(label) for label in y_axis.labels]

        # 4. Map bars to values
        bar_values = []
        for bar in bars:
            # Find bar height in pixel space
            bar_height_px = bar.top - y_axis.origin

            # Convert to data space using y-axis scale
            value = self.pixel_to_value(bar_height_px, y_axis)
            category = x_labels[bar.x_position]

            bar_values.append({'category': category, 'value': value})

        return bar_values

    def pixel_to_value(self, pixel_height, y_axis):
        # Linear interpolation between axis min and max
        axis_range = y_axis.max - y_axis.min
        axis_height_px = y_axis.max_px - y_axis.min_px

        value = y_axis.min + (pixel_height / axis_height_px) * axis_range
        return value
```

**Evaluation metrics**:
- **Exact match**: Answer must match ground truth exactly
- **Relaxed accuracy**: ±5% tolerance for numerical answers
- **F1 score**: For multi-part answers

---

## Section 4: Interleaved Image-Text Sequences

### Web Documents (HTML, PDFs, Slides)

From [CoMM: Coherent Interleaved Image-Text Dataset](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_CoMM_A_Coherent_Interleaved_Image-Text_Dataset_for_Multimodal_Understanding_and_CVPR_2025_paper.pdf) (CVPR 2025, accessed 2025-11-16):
- Interleaved image-text generation: Creating sequences given a query
- Dataset: Web pages, slide decks, tutorial documents
- Tasks: Generate coherent multi-modal documents

**Interleaved sequence structure**:
```
Text: "Introduction to Neural Networks"
Image: [Diagram of neuron structure]
Text: "A neural network consists of layers of interconnected neurons..."
Image: [Architecture diagram showing input → hidden → output]
Text: "Training uses backpropagation to update weights..."
Image: [Gradient descent visualization]
```

**Processing strategies**:
```python
# Interleaved sequence processing
class InterleavedVLM:
    def __init__(self):
        self.vision_encoder = CLIPViT()
        self.language_model = LLaMA2()

        # Special tokens for mode switching
        self.img_start_token = "<image>"
        self.img_end_token = "</image>"

    def forward(self, interleaved_sequence):
        """
        interleaved_sequence = [
            {'type': 'text', 'content': "Intro to NNs"},
            {'type': 'image', 'content': neuron_image},
            {'type': 'text', 'content': "A neural network..."},
            ...
        ]
        """
        combined_tokens = []

        for element in interleaved_sequence:
            if element['type'] == 'text':
                # Tokenize text normally
                tokens = self.language_model.tokenizer(element['content'])
                combined_tokens.extend(tokens)

            elif element['type'] == 'image':
                # Add image boundary tokens
                combined_tokens.append(self.img_start_token)

                # Encode image to token sequence
                img_features = self.vision_encoder(element['content'])
                img_tokens = self.vision_to_tokens(img_features)
                combined_tokens.extend(img_tokens)

                combined_tokens.append(self.img_end_token)

        # Process as unified sequence
        return self.language_model(combined_tokens)
```

**Training objectives**:
1. **Next-token prediction**: Predict text after image (or vice versa)
2. **Image-text matching**: Does image relate to surrounding text?
3. **Coherence scoring**: Is sequence logically ordered?

### SlideVQA (Multi-Slide Understanding)

From [SlideVQA: Document VQA on Multiple Images](https://www.researchgate.net/publication/371920580_SlideVQA_A_Dataset_for_Document_Visual_Question_Answering_on_Multiple_Images) (accessed 2025-11-16):
- 2.6K slide decks, 52K slides, 14.5K questions
- Questions span multiple slides: "How does revenue change from slide 3 to slide 7?"
- Requires: Slide ordering + cross-slide reasoning

**Example question types**:
```python
# SlideVQA question taxonomy
questions = {
    'single_slide': {
        'question': "What is the title of slide 5?",
        'type': 'extractive',
        'slides_needed': [5]
    },
    'multi_slide_comparison': {
        'question': "Which quarter had highest sales (check slides 2-4)?",
        'type': 'comparative',
        'slides_needed': [2, 3, 4]
    },
    'temporal_reasoning': {
        'question': "How does the trend change across slides 1-6?",
        'type': 'reasoning',
        'slides_needed': [1, 2, 3, 4, 5, 6]
    },
    'aggregation': {
        'question': "What is the main theme of the presentation?",
        'type': 'abstractive',
        'slides_needed': 'all'
    }
}
```

**Architecture for slide decks**:
```python
# Slide deck processing
class SlideVQA:
    def __init__(self):
        self.slide_encoder = LayoutLMv3()  # Document-aware encoder
        self.temporal_fusion = TemporalTransformer(num_layers=6)
        self.qa_head = QuestionAnsweringHead()

    def forward(self, slides, question):
        # Encode each slide independently
        slide_features = []
        for slide in slides:
            # LayoutLM captures text + layout + visual
            feat = self.slide_encoder(slide)
            slide_features.append(feat)

        # Add positional encoding (slide order matters)
        slide_features = self.add_slide_positions(slide_features)

        # Cross-slide temporal reasoning
        fused = self.temporal_fusion(
            torch.stack(slide_features)
        )  # [num_slides, hidden_dim]

        # Answer question
        answer = self.qa_head(
            question=question,
            context=fused
        )
        return answer
```

### Comics and Sequential Art

**Unique challenges**:
1. **Panel ordering**: Read left-to-right, top-to-bottom (Western) or right-to-left (manga)
2. **Speech bubbles**: Text integrated into images (not separate OCR)
3. **Visual metaphors**: Thought clouds, action lines, emotion symbols
4. **Cross-panel reasoning**: Characters reappear, storyline progresses

**Processing approach**:
```python
# Comic understanding
class ComicVLM:
    def __init__(self):
        self.panel_detector = YOLOv8(task='panel_detection')
        self.bubble_detector = YOLOv8(task='bubble_detection')
        self.ocr = TrOCR()

        # Sequential reasoning
        self.panel_lstm = nn.LSTM(hidden_size=768, num_layers=2)

    def forward(self, comic_page, question):
        # Detect and order panels
        panels = self.panel_detector(comic_page)
        panels = self.order_panels(panels)  # Reading order

        panel_features = []
        for panel in panels:
            # Visual features
            vis_feat = self.vision_encoder(panel)

            # Detect and read speech bubbles
            bubbles = self.bubble_detector(panel)
            text = [self.ocr(bubble) for bubble in bubbles]

            # Combine visual + text for panel
            panel_feat = self.fuse_visual_text(vis_feat, text)
            panel_features.append(panel_feat)

        # Sequential reasoning across panels
        story_context, _ = self.panel_lstm(
            torch.stack(panel_features)
        )

        return self.answer_question(story_context, question)
```

---

## Section 5: Cross-Image Reasoning (Compare, Relate, Aggregate)

### Comparison Tasks

**Visual comparison types**:
```python
# Image comparison operations
class ImageComparison:
    def count_differences(self, img1, img2):
        """Spot-the-difference task"""
        # 1. Align images
        img2_aligned = self.align(img2, to=img1)

        # 2. Extract features
        feat1 = self.vision_encoder(img1)
        feat2 = self.vision_encoder(img2_aligned)

        # 3. Compute difference map
        diff_map = torch.abs(feat1 - feat2)

        # 4. Threshold and count regions
        diff_regions = self.region_proposal(diff_map, threshold=0.5)
        return len(diff_regions)

    def compare_attributes(self, img1, img2, attribute='color'):
        """Compare specific attribute across images"""
        attr1 = self.extract_attribute(img1, attribute)
        attr2 = self.extract_attribute(img2, attribute)

        if attribute == 'color':
            return self.color_similarity(attr1, attr2)
        elif attribute == 'size':
            return attr1['size'] - attr2['size']
        elif attribute == 'count':
            return attr1['count'] - attr2['count']
```

**Comparison question types**:
```
1. Object counting: "Which image has more cars?"
2. Attribute: "Which image is brighter?"
3. Spatial: "Which image has the object on the left?"
4. Temporal: "Which event happened first?"
5. Semantic: "Which scene is more cluttered?"
```

### Relational Reasoning

From [MMIU relationship taxonomy](https://arxiv.org/abs/2408.02718):

**Spatial relationships**:
```python
# Spatial reasoning across images
class SpatialReasoning:
    def find_common_object(self, images):
        """Find object that appears in all images"""
        # Extract objects from each image
        objects = [self.object_detector(img) for img in images]

        # Find intersection
        common = set(objects[0])
        for obj_set in objects[1:]:
            common = common.intersection(obj_set)

        return list(common)

    def relative_position(self, img1, img2, object_type):
        """Where is object_type in img1 vs img2?"""
        obj1 = self.find_object(img1, object_type)
        obj2 = self.find_object(img2, object_type)

        # Compare positions
        if obj1['x'] < obj2['x']:
            return "left of"
        elif obj1['x'] > obj2['x']:
            return "right of"
        # ... up/down logic
```

**Temporal relationships**:
```
Question: "What changes between images 1, 2, and 3?"

Image 1: Empty room
Image 2: Furniture partially moved in
Image 3: Fully furnished room

Answer: "Furniture gradually added"

Requires:
1. Detect objects in each image
2. Track object appearance/disappearance
3. Infer temporal sequence
```

### Aggregation & Summarization

**Cross-image aggregation**:
```python
# Aggregate information from multiple images
class MultiImageAggregation:
    def count_total(self, images, object_type):
        """Total count across all images"""
        total = 0
        for img in images:
            objects = self.object_detector(img)
            total += objects.count(object_type)
        return total

    def find_common_theme(self, images):
        """What theme connects these images?"""
        # Extract scene tags for each image
        tags = [self.scene_tagger(img) for img in images]

        # Find most common tags
        from collections import Counter
        all_tags = [tag for img_tags in tags for tag in img_tags]
        common_tags = Counter(all_tags).most_common(3)

        # Generate theme summary
        return f"Common theme: {common_tags[0][0]}"

    def temporal_trend(self, images):
        """Detect trend across time-ordered images"""
        values = []
        for img in images:
            # Extract quantitative value (e.g., crowd size)
            val = self.extract_value(img)
            values.append(val)

        # Fit trend
        if all(values[i] < values[i+1] for i in range(len(values)-1)):
            return "increasing"
        elif all(values[i] > values[i+1] for i in range(len(values)-1)):
            return "decreasing"
        else:
            return "fluctuating"
```

---

## Section 6: Long Context Handling (100+ Images)

### Efficient Attention for Many Images

**Challenge**: Standard attention is O(n²) in sequence length
- 100 images × 256 tokens/image = 25,600 tokens
- Attention: 25,600² = 655M operations (infeasible)

**Solutions**:
```python
# Long-context multi-image processing
class LongContextMultiImage:
    def __init__(self):
        self.vision_encoder = CLIPViT()

        # Hierarchical attention
        self.local_attn = FlashAttention2(window_size=512)
        self.global_attn = LongformerAttention(
            attention_window=[512, 1024],
            global_attention_indices=[0, -1]  # First and last tokens
        )

    def forward(self, images):
        """Process 100+ images efficiently"""
        # Stage 1: Encode images independently (parallelizable)
        img_features = [self.vision_encoder(img) for img in images]
        # Each image: [256, 768]

        # Stage 2: Compress each image
        compressed = [self.compress_image(feat) for feat in img_features]
        # Each compressed: [16, 768] (16× compression)

        # Stage 3: Concat all compressed images
        all_tokens = torch.cat(compressed, dim=0)  # [100*16, 768] = [1600, 768]

        # Stage 4: Hierarchical attention
        # Local attention within each image
        local_out = self.local_attn(all_tokens)

        # Global attention across images
        global_out = self.global_attn(local_out)

        return global_out

    def compress_image(self, img_features):
        """Compress 256 tokens → 16 tokens"""
        # Learnable compression queries
        queries = self.learned_queries  # [16, 768]

        compressed = F.multi_head_attention_forward(
            query=queries,
            key=img_features,
            value=img_features,
            num_heads=12
        )
        return compressed
```

**Compression strategies**:
1. **Perceiver-style**: 256 tokens → 16 tokens (16× compression)
2. **Pooling**: Average/max pooling over spatial dimensions
3. **Top-K selection**: Keep only most salient tokens
4. **Pyramid**: Multi-scale representations (fewer tokens at higher levels)

### Retrieval-Augmented Multi-Image

**For 1000+ images (e.g., entire photo album)**:
```python
# Retrieval-augmented VLM
class RetrievalMultiImage:
    def __init__(self):
        self.image_index = FAISSIndex(dim=768)
        self.vlm = MultiImageVLM()

    def process_album(self, images, question):
        """Process 1000+ images via retrieval"""
        # Stage 1: Index all images
        for img in images:
            feat = self.vision_encoder(img)
            feat_avg = feat.mean(dim=0)  # [768]
            self.image_index.add(feat_avg)

        # Stage 2: Retrieve relevant images based on question
        question_embed = self.text_encoder(question)

        # Find top-K most relevant images
        scores, indices = self.image_index.search(
            question_embed,
            k=10  # Only process 10 images
        )
        relevant_images = [images[i] for i in indices]

        # Stage 3: Process only relevant images
        answer = self.vlm(relevant_images, question)
        return answer
```

**Retrieval strategies**:
- **Dense retrieval**: CLIP embeddings + FAISS index
- **Sparse retrieval**: OCR text + BM25
- **Hybrid**: Combine visual and text retrieval scores

---

## Section 7: Table & Chart Understanding

### Table Extraction

From [PaddleOCR table recognition](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/doc_understanding.html):

**Table structure recognition**:
```python
# Table parsing pipeline
class TableExtractor:
    def __init__(self):
        self.table_detector = TableNet()  # Detect table region
        self.structure_recognizer = TableStructure()  # Rows/cols
        self.cell_ocr = TrOCR()

    def extract_table(self, document_image):
        # Step 1: Detect table region
        table_bbox = self.table_detector(document_image)
        table_img = document_image[table_bbox]

        # Step 2: Recognize structure (rows, columns, cells)
        structure = self.structure_recognizer(table_img)
        # Returns: grid[row][col] = cell_bbox

        # Step 3: OCR each cell
        table_data = []
        for row in structure.rows:
            row_data = []
            for cell_bbox in row.cells:
                cell_img = table_img[cell_bbox]
                cell_text = self.cell_ocr(cell_img)
                row_data.append(cell_text)
            table_data.append(row_data)

        return table_data

    def convert_to_dataframe(self, table_data):
        """Convert to pandas DataFrame for querying"""
        import pandas as pd

        # First row is header
        headers = table_data[0]
        rows = table_data[1:]

        df = pd.DataFrame(rows, columns=headers)
        return df
```

**Table question answering**:
```python
# TableQA
def answer_table_question(table_df, question):
    """
    Example:
    question = "What is the total sales for Q2?"
    table_df:
        Quarter | Sales
        Q1      | 100K
        Q2      | 150K
        Q3      | 120K
    """
    # Parse question to SQL-like query
    query = question_to_query(question)
    # "SELECT SUM(Sales) WHERE Quarter='Q2'"

    # Execute on DataFrame
    if 'SUM' in query:
        result = table_df['Sales'].sum()
    elif 'WHERE' in query:
        condition = parse_condition(query)
        result = table_df.query(condition)['Sales'].values[0]

    return result
```

**Table structure challenges**:
- Merged cells (rowspan/colspan)
- Nested headers (multi-level columns)
- Missing borders (text-only tables)
- Rotated tables (vertical text)

### Chart Understanding

**Chart type detection**:
```python
# Chart classifier
chart_types = [
    'bar_chart',
    'line_chart',
    'pie_chart',
    'scatter_plot',
    'box_plot',
    'heatmap',
    'area_chart'
]

class ChartClassifier:
    def classify(self, chart_image):
        # Train CNN on chart images
        features = self.cnn(chart_image)
        logits = self.classifier_head(features)
        chart_type = chart_types[logits.argmax()]
        return chart_type
```

**Chart-specific parsers**:
```python
# Bar chart parser
class BarChartParser:
    def parse(self, chart_image):
        # 1. Detect axes
        axes = self.detect_axes(chart_image)

        # 2. Detect bars
        bars = self.detect_bars(chart_image)

        # 3. OCR axis labels
        x_labels = [self.ocr(label) for label in axes['x']['labels']]
        y_range = self.parse_y_axis(axes['y'])

        # 4. Extract data
        data = {}
        for i, bar in enumerate(bars):
            category = x_labels[i]
            value = self.bar_height_to_value(bar, y_range)
            data[category] = value

        return data

# Pie chart parser
class PieChartParser:
    def parse(self, chart_image):
        # 1. Detect center and radius
        center, radius = self.detect_circle(chart_image)

        # 2. Segment slices
        slices = self.segment_slices(chart_image, center)

        # 3. OCR labels
        labels = [self.ocr(s.label_region) for s in slices]

        # 4. Calculate percentages
        total_angle = 360
        data = {}
        for i, slice in enumerate(slices):
            percentage = (slice.angle / total_angle) * 100
            data[labels[i]] = percentage

        return data
```

---

## Section 8: ARR-COC-0-1 Multi-Image & Document Extensions

### Current ARR-COC-0-1 Architecture

**Single-image focus**:
- 13-channel texture array (RGB, LAB, Sobel, spatial, eccentricity)
- 64-400 tokens per patch (relevance-driven LOD)
- K=200 patches (total ~14K tokens at mid-LOD)

**Limitation**: No multi-image or document-specific processing

### Proposed Multi-Image Extension

**Architecture**:
```python
# ARR-COC multi-image extension
class ARRCOCMultiImage:
    def __init__(self):
        # Existing single-image pipeline
        self.texture_extractor = TextureArrayExtractor()
        self.knowing = ThreeWaysOfKnowing()
        self.balancing = OpponentProcessing()
        self.attending = RelevanceAllocation()

        # NEW: Multi-image extensions
        self.image_relator = ImageRelator()  # Cross-image relationships
        self.temporal_reasoner = TemporalTransformer()

    def forward(self, images, query):
        """Process multiple images with relevance realization"""
        # Stage 1: Process each image independently
        image_features = []
        relevance_maps = []

        for img in images:
            # Extract 13-channel texture
            texture = self.texture_extractor(img)

            # Realize relevance for this image
            relevance = self.knowing(texture, query)
            relevance = self.balancing(relevance)
            token_budget = self.attending(relevance)

            # Compress to tokens
            img_tokens = self.compress(texture, token_budget)

            image_features.append(img_tokens)
            relevance_maps.append(relevance)

        # Stage 2: Cross-image relevance realization
        # Question: Which images are most relevant to query?
        image_relevance = self.image_relator(
            image_features,
            relevance_maps,
            query
        )

        # Stage 3: Allocate tokens across images
        # High-relevance images get more tokens
        token_allocation = self.allocate_tokens_across_images(
            image_relevance,
            total_budget=20000  # Total token budget
        )

        # Stage 4: Re-process images with new budgets
        final_features = []
        for i, img in enumerate(images):
            tokens = self.compress(
                self.texture_extractor(img),
                budget=token_allocation[i]
            )
            final_features.append(tokens)

        # Stage 5: Temporal/relational reasoning
        if self.is_temporal_query(query):
            final_features = self.temporal_reasoner(final_features)

        # Stage 6: Feed to LLM
        return self.llm(torch.cat(final_features), query)
```

**Key innovations**:
1. **Per-image relevance realization**: Each image processed with ARR-COC pipeline
2. **Cross-image token allocation**: Relevant images get more tokens (64-400)
3. **Temporal reasoning**: Order-aware processing for sequential images
4. **Adaptive total budget**: 5 images × 4K tokens = 20K (manageable context)

### Document Understanding Extension

**OCR integration**:
```python
# ARR-COC document understanding
class ARRCOCDocument:
    def __init__(self):
        self.arr_coc = ARRCOC()

        # Document-specific modules
        self.text_detector = CRAFT()
        self.ocr = TrOCR()
        self.layout_analyzer = LayoutLMv3()

    def forward(self, document_image, query):
        # Stage 1: Layout analysis
        layout = self.layout_analyzer(document_image)
        # Returns: title, paragraph, table, figure regions

        # Stage 2: Relevance-driven OCR
        # Only OCR text regions relevant to query
        text_regions = layout['text_regions']
        text_relevance = self.compute_text_relevance(
            text_regions,
            query
        )

        # Top-K most relevant regions
        top_regions = text_relevance.topk(k=20)

        # Stage 3: OCR selected regions
        ocr_results = []
        for region in top_regions:
            text = self.ocr(document_image[region])
            ocr_results.append(text)

        # Stage 4: Visual processing (ARR-COC)
        visual_features = self.arr_coc(document_image, query)

        # Stage 5: Fuse text + visual
        combined = self.fuse_text_visual(
            ocr_results,
            visual_features
        )

        return self.llm(combined, query)
```

**Benefits for ARR-COC-0-1**:
- **Selective OCR**: Only process relevant text (saves compute)
- **Visual grounding**: Texture features enhance text understanding
- **Adaptive LOD**: Text-heavy regions get higher resolution
- **Layout-aware**: Respects document structure (tables, headers, captions)

### Evaluation Strategy

**Benchmarks**:
1. **MMIU** (multi-image): Measure cross-image reasoning with relevance allocation
2. **DocVQA** (documents): Compare ARR-COC-Document vs pure OCR approaches
3. **SlideVQA** (presentations): Multi-slide temporal reasoning

**Expected performance**:
- Multi-image: Relevance realization should outperform uniform token allocation
- Documents: Selective OCR + visual features should match full OCR with fewer tokens
- Slides: Temporal reasoning + relevance should handle long decks efficiently

**Ablations**:
1. Fixed vs adaptive token allocation across images
2. With/without OCR integration
3. Uniform LOD vs relevance-driven LOD for documents

---

## Sources

**Source Documents**:
- None (this file created from web research)

**Web Research**:
- [PRIMA: Multi-Image Vision-Language Models](https://arxiv.org/abs/2412.15209) - arXiv:2412.15209 (accessed 2025-11-16)
- [MMIU: Multimodal Multi-Image Understanding](https://arxiv.org/abs/2408.02718) - arXiv:2408.02718 (accessed 2025-11-16)
- [Position Bias in Multi-Image VLMs](https://openaccess.thecvf.com/content/CVPR2025/papers/Tian_Identifying_and_Mitigating_Position_Bias_of_Multi-image_Vision-Language_Models_CVPR_2025_paper.pdf) - CVPR 2025 (accessed 2025-11-16)
- [Leopard: Text-Rich Multi-Image VLM](https://arxiv.org/abs/2410.01744) - arXiv:2410.01744 (accessed 2025-11-16)
- [Amazon Science: Multi-Image VLMs](https://www.amazon.science/blog/vision-language-models-that-can-handle-multi-image-inputs) (accessed 2025-11-16)
- [DocVLM: Efficient Reader](https://arxiv.org/abs/2412.08746) - arXiv:2412.08746 (accessed 2025-11-16)
- [PaddleOCR Document Understanding](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/doc_understanding.html) (accessed 2025-11-16)
- [SlideVQA Dataset](https://www.researchgate.net/publication/371920580_SlideVQA_A_Dataset_for_Document_Visual_Question_Answering_on_Multiple_Images) (accessed 2025-11-16)
- [CoMM: Interleaved Image-Text Dataset](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_CoMM_A_Coherent_Interleaved_Image-Text_Dataset_for_Multimodal_Understanding_and_CVPR_2025_paper.pdf) - CVPR 2025 (accessed 2025-11-16)

**Existing Knowledge**:
- [DeepSeek-OCR Oracle](../../deepseek-ocr-oracle/INDEX.md) - Document understanding, optical compression, OCR-free processing
- [DeepSeek-OCR Architecture](../../deepseek-ocr-oracle/architecture/00-overview.md) - SAM+CLIP serial design, 16× compression

**Additional References**:
- Flamingo (DeepMind) - Gated cross-attention for multi-image VLMs
- Otter - Instruction-tuned multi-image model
- DocVQA benchmark - Document visual question answering
- InfoVQA - Infographic understanding
- ChartQA - Chart question answering
