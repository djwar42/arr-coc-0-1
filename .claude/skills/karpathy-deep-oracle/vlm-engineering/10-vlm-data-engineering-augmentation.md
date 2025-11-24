# VLM Data Engineering and Augmentation

## Overview

Vision-language model (VLM) training requires high-quality, large-scale image-text paired datasets. Data engineering and augmentation are critical for VLM performance, as data quality often matters more than model architecture. This document covers dataset curation, augmentation strategies, synthetic data generation, quality filtering, multi-modal data formats, and data loading optimization for production VLM training.

## Section 1: Dataset Curation and Collection (~100 lines)

### Web-Scale Data Collection

**Large-scale dataset sources:**

From [SAIL-VL paper](https://aclanthology.org/2025.acl-long.1595/) (ACL 2025, accessed 2025-11-16):
- LAION-400M/LAION-5B: 400M-5B image-text pairs from web scraping
- COYO-700M: 700M pairs with improved caption quality
- CC12M/CC3M: Conceptual Captions (12M/3M curated pairs)
- DataComp: 12.8B image-text pairs with quality filtering
- SAIL-Caption: 100M+ high-quality recaptioned pairs (SOTA data quality)

**Collection pipeline:**
```python
# Web scraping for image-text pairs
import requests
from bs4 import BeautifulSoup

def scrape_image_text_pairs(url_list):
    """Scrape images and alt-text from web pages"""
    pairs = []
    for url in url_list:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract images with alt text
        for img in soup.find_all('img'):
            if img.get('alt') and img.get('src'):
                pairs.append({
                    'image_url': img['src'],
                    'text': img['alt'],
                    'source': url
                })

    return pairs
```

From [Vertex AI Data Integration](../karpathy/practical-implementation/34-vertex-ai-data-integration.md):
- GCS bucket organization for multi-modal data
- Parallel uploads with `gsutil -m` for large datasets
- Storage class selection (Standard for active training data)

### Deduplication Strategies

**Image deduplication:**
```python
import imagehash
from PIL import Image

def deduplicate_images(image_paths, threshold=5):
    """Remove duplicate/near-duplicate images using perceptual hashing"""
    hashes = {}
    unique_images = []

    for path in image_paths:
        img = Image.open(path)
        # Perceptual hash (robust to minor edits)
        img_hash = imagehash.phash(img)

        # Check for duplicates
        is_duplicate = False
        for existing_hash in hashes:
            if abs(img_hash - existing_hash) < threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            hashes[img_hash] = path
            unique_images.append(path)

    return unique_images
```

**Text deduplication:**
- Exact match deduplication (remove identical captions)
- Near-duplicate detection (MinHash, SimHash for similar captions)
- Cross-dataset deduplication (avoid test set contamination)

### Data Provenance and Licensing

**Metadata tracking:**
```python
# Track data provenance for compliance
metadata = {
    'image_id': 'img_001234',
    'source': 'CC-BY-2.0',
    'license': 'Creative Commons Attribution 2.0',
    'url': 'https://example.com/image.jpg',
    'collected_date': '2024-01-15',
    'dataset': 'CC12M',
}
```

**Licensing considerations:**
- Prefer CC-BY, CC0, public domain images
- Avoid scraping copyrighted content without permission
- Respect robots.txt and rate limits
- Store license metadata for audit trails

## Section 2: Image Augmentation Techniques (~100 lines)

### Standard Image Augmentations

**Random crop and resize:**
```python
from torchvision import transforms

# Standard VLM image augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        384,
        scale=(0.8, 1.0),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])
```

**RandAugment for VLMs:**
```python
from torchvision.transforms import RandAugment

# Automatically select random augmentations
augmentation = RandAugment(
    num_ops=2,        # Number of augmentations to apply
    magnitude=9,      # Strength of augmentations (0-30)
)

# Apply to image before VLM processing
augmented_image = augmentation(original_image)
```

From web research on image-text data augmentation (accessed 2025-11-16):
- Avoid augmentations that change semantic content (e.g., extreme rotations)
- Use mild color jitter (preserve object colors mentioned in captions)
- Random horizontal flip safe for most VLM tasks
- Avoid vertical flips (changes spatial relationships)

### Multi-Resolution Training

**Dynamic resolution strategy:**
```python
def multi_resolution_augmentation(image, resolutions=[224, 336, 384, 448]):
    """Randomly sample resolution during training"""
    import random
    target_size = random.choice(resolutions)

    # Resize with aspect ratio preservation
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])

    return transform(image)
```

**Benefits:**
- Models learn resolution-invariant features
- Better generalization to different input sizes
- Prepares model for adaptive resolution inference

### Image Quality Filtering

**Blur detection:**
```python
import cv2

def is_image_blurry(image_path, threshold=100):
    """Detect blurry images using Laplacian variance"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var < threshold  # True if blurry
```

**Aesthetic scoring:**
- Use CLIP aesthetic predictor to filter low-quality images
- Remove images with poor composition, lighting, or artifacts
- Keep images with high CLIP aesthetic scores (>5.0/10)

## Section 3: Text Augmentation and Caption Engineering (~100 lines)

### Caption Paraphrasing

**Back-translation for diversity:**
```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate_caption(caption, src_lang='en', pivot_lang='fr'):
    """Generate paraphrases via back-translation"""
    # English -> French
    model_name_en_fr = f'Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}'
    tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
    model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)

    translated = model_en_fr.generate(
        **tokenizer_en_fr(caption, return_tensors="pt")
    )
    french_caption = tokenizer_en_fr.decode(translated[0], skip_special_tokens=True)

    # French -> English
    model_name_fr_en = f'Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}'
    tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
    model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en)

    back_translated = model_fr_en.generate(
        **tokenizer_fr_en(french_caption, return_tensors="pt")
    )
    paraphrased_caption = tokenizer_fr_en.decode(back_translated[0], skip_special_tokens=True)

    return paraphrased_caption

# Example:
original = "A dog playing in the park"
paraphrased = back_translate_caption(original)
# Result: "A dog plays in the park" (slightly different wording)
```

### Template-Based Augmentation

**Caption template expansion:**
```python
import random

templates = [
    "A photo of {}",
    "An image showing {}",
    "{} in the scene",
    "This picture contains {}",
    "Depicting {}",
]

def augment_caption_with_templates(caption):
    """Expand caption with random templates"""
    template = random.choice(templates)
    return template.format(caption)

# Example:
base_caption = "a red car"
augmented = augment_caption_with_templates(base_caption)
# Result: "A photo of a red car"
```

### Caption Length Normalization

**Ensure diverse caption lengths:**
```python
def normalize_caption_length(captions, target_min=10, target_max=50):
    """Filter captions by word count"""
    filtered = []
    for caption in captions:
        word_count = len(caption.split())
        if target_min <= word_count <= target_max:
            filtered.append(caption)

    return filtered
```

From web research (accessed 2025-11-16):
- Short captions (5-15 words): Good for object detection alignment
- Medium captions (15-30 words): Best for general VLM training
- Long captions (30-50 words): Useful for detailed scene understanding

## Section 4: Synthetic Data Generation (~150 lines)

### Synthetic Caption Generation with LLMs

From [Synth^2 paper](https://arxiv.org/abs/2403.07750) (arXiv 2024, accessed 2025-11-16):
- Use LLMs to generate diverse captions from prompts
- Generate novel image-caption pairs without human annotation
- Synthetic data achieves comparable performance to human-labeled data

**LLM-based caption generation:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_synthetic_captions(objects, model_name="gpt2"):
    """Generate captions using LLM"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = f"Generate a detailed caption describing an image with: {', '.join(objects)}"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=5,  # Generate 5 variants
        temperature=0.9,
        top_p=0.95,
    )

    captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# Example:
objects = ["cat", "sofa", "window"]
synthetic_captions = generate_synthetic_captions(objects)
# Results:
# - "A cat resting on a sofa near a window"
# - "A fluffy cat lounging on a comfortable sofa by the window"
# - etc.
```

### BLIP-Based Recaptioning

From [BLIP documentation](https://huggingface.co/docs/transformers/en/model_doc/blip) (HuggingFace, accessed 2025-11-16):
- BLIP Captioner generates synthetic captions for images
- BLIP Filter removes noisy/low-quality captions
- CapFilt (Captioning + Filtering) improves training data quality

**BLIP recaptioning pipeline:**
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def recaption_with_blip(image_path):
    """Generate high-quality captions using BLIP"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    outputs = model.generate(**inputs, max_length=50, num_beams=5)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Apply to noisy dataset
noisy_captions = ["img1.jpg", "img2.jpg", "img3.jpg"]
clean_captions = [recaption_with_blip(img) for img in noisy_captions]
```

### Synthetic Image Generation in Embedding Space

From [Synth^2 paper](https://arxiv.org/abs/2403.07750) (arXiv 2024, accessed 2025-11-16):
- Generate image embeddings directly (skip pixel generation)
- 25% faster than pixel-space generation
- Text-to-image models create novel compositions beyond training data

**Embedding-space synthesis:**
```python
def synthesize_image_embedding(caption, text_to_image_model):
    """Generate image embedding from caption (skip pixels)"""
    # Use text-to-image model's encoder
    text_embedding = text_to_image_model.encode_text(caption)

    # Generate image embedding in latent space
    image_embedding = text_to_image_model.decode_to_embedding(text_embedding)

    return image_embedding

# Use synthesized embeddings directly for VLM training
# No need to decode to pixels and re-encode
```

**Semantic diversity in synthetic captions:**

From [Synth^2 paper](https://arxiv.org/abs/2403.07750):
- Semantic diversity more important than volume
- Balance object categories (avoid dominant classes)
- Use LLMs to generate diverse scenarios

```python
def generate_diverse_scenarios(base_objects, num_scenarios=100):
    """Generate semantically diverse scenarios"""
    scenarios = []

    # Vary: actions, locations, attributes, relationships
    actions = ["running", "sitting", "flying", "swimming", "eating"]
    locations = ["park", "beach", "mountain", "city", "forest"]
    attributes = ["red", "large", "small", "colorful", "shiny"]

    for _ in range(num_scenarios):
        obj = random.choice(base_objects)
        action = random.choice(actions)
        location = random.choice(locations)
        attr = random.choice(attributes)

        scenario = f"A {attr} {obj} {action} in the {location}"
        scenarios.append(scenario)

    return scenarios
```

## Section 5: Data Quality Filtering (~150 lines)

### CLIP Score Filtering

**Image-text alignment scoring:**
```python
import torch
from transformers import CLIPProcessor, CLIPModel

def compute_clip_score(image_path, caption):
    """Compute CLIP similarity between image and caption"""
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.open(image_path)
    inputs = processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    # Compute cosine similarity
    clip_score = outputs.logits_per_image.item()

    return clip_score

def filter_by_clip_score(dataset, threshold=0.25):
    """Remove low-quality image-text pairs"""
    filtered = []
    for image_path, caption in dataset:
        score = compute_clip_score(image_path, caption)
        if score > threshold:
            filtered.append((image_path, caption, score))

    return filtered
```

From web research on data quality filtering (accessed 2025-11-16):
- CLIP score threshold: 0.25-0.30 for general datasets
- Higher thresholds (0.35+) for high-quality curation
- CLIP filtering removes ~30-40% of web-scraped pairs

### BLIP Score Filtering

**Caption quality assessment:**
```python
def blip_filter_quality(image_path, caption, threshold=0.5):
    """Use BLIP to score caption quality"""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path)
    inputs = processor(image, caption, return_tensors="pt")

    # Compute caption likelihood given image
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()

    # Lower loss = better caption quality
    quality_score = 1.0 / (1.0 + loss)

    return quality_score > threshold
```

### Multi-Stage Filtering Pipeline

From [SAIL-VL paper](https://aclanthology.org/2025.acl-long.1595/) (ACL 2025):
- Stage 1: Global filtering (remove low-quality images/captions)
- Stage 2: Pair filtering (select best image-caption matches)
- SAIL-Caption achieves highest data quality vs. open-source datasets

**Complete filtering pipeline:**
```python
def multi_stage_filter(dataset):
    """Apply multiple quality filters"""
    # Stage 1: Global image quality
    filtered = [
        (img, cap) for img, cap in dataset
        if not is_image_blurry(img) and is_image_aesthetic(img)
    ]

    # Stage 2: Caption quality
    filtered = [
        (img, cap) for img, cap in filtered
        if 10 <= len(cap.split()) <= 50  # Length filter
    ]

    # Stage 3: Image-text alignment
    filtered = [
        (img, cap) for img, cap in filtered
        if compute_clip_score(img, cap) > 0.28
    ]

    # Stage 4: BLIP quality filter
    filtered = [
        (img, cap) for img, cap in filtered
        if blip_filter_quality(img, cap, threshold=0.5)
    ]

    return filtered
```

### Toxicity and Safety Filtering

**Remove unsafe content:**
```python
def filter_unsafe_content(caption):
    """Remove toxic or inappropriate captions"""
    from transformers import pipeline

    # Use toxicity classifier
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    result = classifier(caption)[0]

    # Keep only safe content
    return result['label'] == 'non-toxic' and result['score'] > 0.95

def filter_nsfw_images(image_path):
    """Remove NSFW images"""
    from transformers import pipeline

    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    image = Image.open(image_path)
    result = classifier(image)[0]

    return result['label'] == 'normal' and result['score'] > 0.9
```

## Section 6: Multi-Modal Data Formats (~100 lines)

### WebDataset Format

**Efficient streaming format:**
```python
import webdataset as wds

# Create WebDataset
def create_webdataset(output_path, image_caption_pairs):
    """Create WebDataset tar archives"""
    with wds.TarWriter(output_path) as sink:
        for idx, (image_path, caption) in enumerate(image_caption_pairs):
            # Read image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Create sample
            sample = {
                "__key__": f"sample{idx:06d}",
                "jpg": image_data,
                "txt": caption.encode('utf-8'),
            }
            sink.write(sample)

# Load WebDataset for training
dataset = wds.WebDataset("dataset-{000000..000099}.tar")
dataset = dataset.decode("pil")  # Decode images
dataset = dataset.to_tuple("jpg", "txt")  # Extract image, caption
```

From [Vertex AI Data Integration](../karpathy/practical-implementation/34-vertex-ai-data-integration.md):
- WebDataset enables streaming from GCS
- Tar archives support parallel data loading
- Efficient for multi-node distributed training

### TFRecord Format

**TensorFlow/JAX training:**
```python
import tensorflow as tf

def create_tfrecord(output_path, image_caption_pairs):
    """Create TFRecord dataset"""
    with tf.io.TFRecordWriter(output_path) as writer:
        for image_path, caption in image_caption_pairs:
            # Read image
            image_data = tf.io.read_file(image_path)

            # Create example
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data.numpy()])),
                'caption': tf.train.Feature(bytes_list=tf.train.BytesList(value=[caption.encode('utf-8')])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Load TFRecord
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'caption': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_jpeg(parsed['image'])
    caption = parsed['caption']

    return image, caption

dataset = tf.data.TFRecordDataset("dataset.tfrecord")
dataset = dataset.map(parse_tfrecord)
```

### Arrow/Parquet Format

**HuggingFace Datasets format:**
```python
from datasets import Dataset
import pyarrow as pa

def create_arrow_dataset(image_caption_pairs):
    """Create Arrow dataset (HuggingFace format)"""
    data = {
        'image_path': [img for img, _ in image_caption_pairs],
        'caption': [cap for _, cap in image_caption_pairs],
    }

    dataset = Dataset.from_dict(data)

    # Save to disk
    dataset.save_to_disk("vlm_dataset")

    # Or save as Parquet
    dataset.to_parquet("vlm_dataset.parquet")

# Load Arrow dataset
from datasets import load_from_disk
dataset = load_from_disk("vlm_dataset")

# Efficient random access and filtering
filtered = dataset.filter(lambda x: len(x['caption'].split()) > 10)
```

## Section 7: Data Loading Optimization (~100 lines)

### Streaming Data Loaders

**Avoid loading entire dataset into memory:**
```python
import torch
from torch.utils.data import IterableDataset

class StreamingVLMDataset(IterableDataset):
    """Stream data from GCS/S3 without loading everything"""
    def __init__(self, data_urls):
        self.data_urls = data_urls

    def __iter__(self):
        for url in self.data_urls:
            # Stream from cloud storage
            for sample in self.stream_from_url(url):
                yield sample

    def stream_from_url(self, url):
        """Stream WebDataset from URL"""
        import webdataset as wds
        dataset = wds.WebDataset(url)
        for sample in dataset:
            yield sample

# Use with DataLoader
dataloader = torch.utils.data.DataLoader(
    StreamingVLMDataset(["gs://bucket/dataset-{000000..000099}.tar"]),
    batch_size=32,
    num_workers=4,
)
```

### Caching and Prefetching

**Speed up data loading:**
```python
from datasets import load_dataset

# Load dataset with caching
dataset = load_dataset("imagefolder", data_dir="images/", cache_dir="cache/")

# Enable prefetching
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    prefetch_factor=2,  # Prefetch 2 batches per worker
    pin_memory=True,    # Faster GPU transfer
)
```

**Multi-process data loading:**
```python
# Persistent workers (avoid worker restart overhead)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,  # Keep workers alive between epochs
)
```

From [VQAv2 Training Protocols](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
- Dynamic batching reduces padding waste
- Group similar-length captions together
- Use `DataCollatorWithPadding` for variable-length text

### Data Pipeline Best Practices

**Efficient preprocessing:**
```python
def optimized_preprocessing(dataset):
    """Apply efficient preprocessing pipeline"""
    # 1. Filter early (before expensive operations)
    dataset = dataset.filter(lambda x: len(x['caption'].split()) > 5)

    # 2. Cache after expensive operations
    dataset = dataset.map(preprocess_image, cache_file_name="cached_images.arrow")

    # 3. Shard for distributed training
    dataset = dataset.shard(num_shards=8, index=rank)

    # 4. Shuffle with large buffer
    dataset = dataset.shuffle(buffer_size=10000, seed=42)

    # 5. Batch and prefetch
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(buffer_size=2)

    return dataset
```

**Memory-efficient image loading:**
```python
from PIL import Image

def load_image_lazy(image_path):
    """Load image only when needed (lazy loading)"""
    def _load():
        return Image.open(image_path).convert('RGB')

    return _load

# Store lazy loaders instead of loaded images
dataset = [
    {'image_loader': load_image_lazy(path), 'caption': caption}
    for path, caption in pairs
]

# Load images during iteration (saves memory)
for batch in dataset:
    images = [item['image_loader']() for item in batch]
```

## Section 8: ARR-COC-0-1 Data Pipeline (~100 lines)

### VQA Dataset Preparation

From [VQAv2 Training Protocols](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
- VQAv2: 82,783 train images, 443,757 questions
- Multi-label soft encoding (10 answers per question)
- Answer normalization (lowercase, remove articles)

**ARR-COC-0-1 VQA data loader:**
```python
def create_arr_coc_vqa_dataset(vqa_annotations, image_dir):
    """Create VQA dataset for ARR-COC-0-1 training"""
    pairs = []

    for qa in vqa_annotations:
        image_path = f"{image_dir}/{qa['image_id']:012d}.jpg"
        question = qa['question']

        # Soft label encoding
        answers = qa['answers']  # List of 10 human answers
        answer_counts = {}
        for ans in answers:
            ans_normalized = normalize_answer(ans['answer'])
            answer_counts[ans_normalized] = answer_counts.get(ans_normalized, 0) + 1

        # VQA accuracy formula: min(count/3, 1.0)
        answer_scores = {
            ans: min(count / 3.0, 1.0)
            for ans, count in answer_counts.items()
        }

        pairs.append({
            'image_path': image_path,
            'question': question,
            'answer_scores': answer_scores,
        })

    return pairs

def normalize_answer(answer):
    """VQA answer normalization"""
    import re
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)  # Remove articles
    answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
    answer = ' '.join(answer.split())  # Remove extra whitespace
    return answer
```

### Relevance Annotation for ARR-COC

**Ground-truth relevance maps:**
```python
def annotate_relevance_for_question(image_path, question, segmentation_masks):
    """Create relevance annotations for ARR-COC training"""
    # Parse question to identify relevant objects
    relevant_objects = extract_objects_from_question(question)

    # Create relevance map based on segmentation
    relevance_map = np.zeros((14, 14))  # 14x14 patches

    for obj in relevant_objects:
        if obj in segmentation_masks:
            # Mark patches containing relevant objects
            mask = segmentation_masks[obj]
            for i in range(14):
                for j in range(14):
                    patch_region = mask[i*27:(i+1)*27, j*27:(j+1)*27]
                    if patch_region.sum() > 0:
                        relevance_map[i, j] = 1.0

    return relevance_map

# Use relevance annotations to supervise ARR-COC allocation
# Train model to allocate more tokens to relevant patches
```

### Data Augmentation for ARR-COC

**Query-aware augmentation:**
```python
def arr_coc_augmentation(image, question):
    """Augmentation that preserves question-relevant content"""
    # Extract relevant regions from question
    relevant_objects = extract_objects_from_question(question)

    # Apply augmentations that don't distort relevant content
    augmentations = [
        transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),  # Mild crop
        transforms.ColorJitter(brightness=0.1, contrast=0.1),   # Mild color
    ]

    # Avoid: random rotation (changes spatial relationships in question)
    # Avoid: strong crop (may remove objects mentioned in question)

    transform = transforms.Compose(augmentations)
    return transform(image)
```

### Multi-Scale Training for Adaptive LOD

**Train with variable token budgets:**
```python
def arr_coc_multiscale_batch(images, questions, token_budgets):
    """Create batches with different token budgets"""
    batch = []

    for image, question in zip(images, questions):
        # Sample random token budget during training
        budget = random.choice(token_budgets)  # e.g., [64, 144, 256, 400]

        batch.append({
            'image': image,
            'question': question,
            'target_token_budget': budget,
        })

    return batch

# Train ARR-COC to allocate tokens based on target budget
# Enables flexible inference-time token allocation
```

### Data Scaling Strategy

From [SAIL-VL paper](https://aclanthology.org/2025.acl-long.1595/):
- Scale pretraining up to 655B tokens
- Logarithmic scaling laws (2B model benefits from more data)
- Progressive complexity scaling for SFT

**ARR-COC data scaling:**
```python
# Stage 1: Pretraining (large scale, simple captions)
pretrain_data = {
    'CC12M': 12_000_000,
    'LAION-400M-subset': 100_000_000,
    'SAIL-Caption': 50_000_000,
}

# Stage 2: VQA fine-tuning (medium scale, query-aware)
vqa_data = {
    'VQAv2': 443_757,
    'GQA': 943_000,
    'OKVQA': 14_055,
}

# Stage 3: Complex reasoning (small scale, high quality)
reasoning_data = {
    'CLEVR': 699_989,
    'VizWiz': 20_523,
}

# Train progressively: pretrain → VQA → reasoning
# Each stage uses higher-quality, more complex data
```

## Sources

**Source Documents:**
- [50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md) - VQA dataset structure, answer encoding, soft labels
- [34-vertex-ai-data-integration.md](../karpathy/practical-implementation/34-vertex-ai-data-integration.md) - GCS storage, data transfer, pipeline patterns

**Web Research:**
- [Synth^2: Boosting VLMs with Synthetic Captions](https://arxiv.org/abs/2403.07750) - arXiv 2403.07750 (accessed 2025-11-16) - Synthetic caption generation, embedding-space synthesis, semantic diversity
- [SAIL-VL: Scalable VLM Training via High Quality Data Curation](https://aclanthology.org/2025.acl-long.1595/) - ACL 2025 (accessed 2025-11-16) - Data curation pipelines, 655B token scaling, progressive SFT
- [BLIP: Bootstrapping Language-Image Pre-training](https://huggingface.co/docs/transformers/en/model_doc/blip) - HuggingFace docs (accessed 2025-11-16) - CapFilt method, BLIP captioner and filter

**Additional References:**
- Google search: "vision-language dataset curation 2024 2025" (accessed 2025-11-16)
- Google search: "image-text data augmentation VLM training" (accessed 2025-11-16)
- Google search: "synthetic caption generation VLM training 2024" (accessed 2025-11-16)
- Google search: "data quality filtering VLM training CLIP score BLIP" (accessed 2025-11-16)
