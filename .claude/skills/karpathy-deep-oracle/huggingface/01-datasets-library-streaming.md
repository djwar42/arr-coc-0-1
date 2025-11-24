# HuggingFace Datasets Library: Streaming, Processing & Performance

**Comprehensive guide to the HuggingFace Datasets library covering streaming, map/batch processing, Arrow backend, cache management, and custom dataset creation for efficient ML data pipelines**

This document provides production-ready patterns for using the HuggingFace Datasets library with focus on streaming large datasets, efficient batch processing, Arrow memory format optimization, and custom dataset loading.

---

## Section 1: Datasets Library Architecture (~90 lines)

### Apache Arrow Backend

From [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/en/about_arrow) (accessed 2025-11-15):

**Core Architecture**:
- **Columnar memory format**: Arrow stores data in columns rather than rows
- **Zero-copy reads**: Direct memory access without deserialization
- **Memory mapping**: Work with datasets larger than RAM
- **Type system**: Rich type metadata (nested structures, timestamps, etc.)

**Why Arrow?**:
```python
# Traditional Python list approach
data = [{"text": "...", "label": 0} for _ in range(1_000_000)]
# High memory overhead, slow iteration

# Arrow-backed Dataset
from datasets import Dataset
dataset = Dataset.from_dict({"text": [...], "label": [...]})
# Efficient columnar storage, fast batch access
```

**Performance Benefits**:
- **Fast column access**: O(1) access to specific columns
- **Efficient filtering**: Skip irrelevant data without loading
- **Batch operations**: Vectorized operations on columns
- **Interoperability**: Native integration with Pandas, PyArrow, Polars

From [Cloud Storage & BigQuery for ML Data](../gcloud-data/00-storage-bigquery-ml-data.md):
- Similar to BigQuery columnar format for analytics
- Optimized for read-heavy ML workloads

### Dataset vs IterableDataset

From [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable) (accessed 2025-11-15):

**Dataset (Map-style)**:
- Random access by index: `dataset[42]`
- Supports shuffling entire dataset
- Requires downloading full dataset
- Use when: Dataset fits in disk, need random access

**IterableDataset (Streaming)**:
- Sequential iteration only
- No download required
- Constant memory usage
- Use when: Huge datasets (TB scale), quick exploration, limited disk

```python
# Map-style: Full download
dataset = load_dataset("squad", split="train")
print(dataset[100])  # Random access

# Streaming: No download
dataset = load_dataset("squad", split="train", streaming=True)
print(next(iter(dataset)))  # Sequential only
```

### Memory Efficiency

**Arrow Memory Layout**:
```
Traditional Row Format:
[{name: "Alice", age: 30}, {name: "Bob", age: 25}]
Memory: Scattered, cache-unfriendly

Arrow Columnar Format:
names: ["Alice", "Bob"]
ages: [30, 25]
Memory: Contiguous, cache-friendly, vectorized ops
```

**Practical Impact**:
- **2-10x faster** batch processing vs Python lists
- **50% less memory** for numeric columns
- **Zero-copy** to NumPy/PyTorch tensors

---

## Section 2: Loading Datasets (Hub, Local, Remote) (~85 lines)

### Loading from HuggingFace Hub

From [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/en/loading) (accessed 2025-11-15):

**Basic loading**:
```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("squad", split="train")

# Load specific split
train_data = load_dataset("glue", "mrpc", split="train")
test_data = load_dataset("glue", "mrpc", split="test")

# Load subset of data
small_dataset = load_dataset("squad", split="train[:1000]")
```

**Multiple configurations**:
```python
# GLUE has multiple tasks
dataset = load_dataset("glue", "sst2")  # Sentiment analysis
dataset = load_dataset("glue", "qnli")  # Question NLI
```

From [HuggingFace Hub Documentation](../huggingface-hub/datasets/overview.md):
- 100,000+ datasets on Hub
- Community-curated and vetted
- Standardized metadata and licensing

### Loading Local Files

**Supported formats**:
- CSV, JSON, Parquet, Arrow, text files
- Images (JPEG, PNG), audio (WAV, MP3, FLAC)
- Custom loading scripts

```python
# CSV files
dataset = load_dataset("csv", data_files="my_file.csv")

# JSON files
dataset = load_dataset("json", data_files="my_file.json")

# Multiple files with glob patterns
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train-*.jsonl.gz",
        "test": "data/test-*.jsonl.gz"
    }
)

# Parquet (columnar format)
dataset = load_dataset("parquet", data_files="data.parquet")
```

### Streaming Local Files

From [HuggingFace Datasets Streaming Documentation](https://huggingface.co/docs/datasets/en/stream) (accessed 2025-11-15):

**No conversion required**:
```python
# Stream compressed JSONL without extraction
data_files = {"train": "path/to/OSCAR-2201/compressed/en_meta/*.jsonl.gz"}
dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)

# Instantly available - no wait for conversion
print(next(iter(dataset)))
```

**Parquet streaming with column selection**:
```python
from datasets.packaged_modules.parquet import ParquetConfig

# Stream only specific columns
dataset = load_dataset(
    "parquet",
    data_files="large_dataset.parquet",
    split="train",
    streaming=True,
    columns=["text", "label"]  # Only load these columns
)

# Apply filters during streaming
dataset = load_dataset(
    "parquet",
    data_files="large_dataset.parquet",
    split="train",
    streaming=True,
    filters=[("score", ">=", 0.95)]  # Only high-quality samples
)
```

---

## Section 3: Streaming Large Datasets (IterableDataset) (~100 lines)

### When to Use Streaming

From [HuggingFace Datasets Streaming Documentation](https://huggingface.co/docs/datasets/en/stream) (accessed 2025-11-15):

**Use streaming when**:
1. Dataset is extremely large (TB scale like FineWeb: 45TB)
2. Limited disk space (dataset won't fit on disk)
3. Quick exploration (want to see samples immediately)
4. Training from remote data (no local copy needed)

**Example: 45TB dataset instantly available**:
```python
from datasets import load_dataset

# FineWeb English: 45 terabytes
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)

# Use immediately - no download
print(next(iter(dataset)))
# {'text': 'How AP reported in all formats from tornado-stricken regions...',
#  'language_score': 0.972, 'token_count': 717}
```

### IterableDataset Operations

**Shuffling**:
```python
# Shuffle with buffer
shuffled = dataset.shuffle(seed=42, buffer_size=10_000)

# Buffer loads 10k examples, samples randomly, refills
# Trade-off: Larger buffer = better shuffle, more memory
```

**Reshuffle per epoch**:
```python
for epoch in range(num_epochs):
    shuffled_dataset.set_epoch(epoch)  # New shuffle seed
    for example in shuffled_dataset:
        # Training loop
        pass
```

**Split dataset**:
```python
# Take first N examples
dataset_head = dataset.take(1000)

# Skip first N examples
dataset_tail = dataset.skip(1000)

# Train/validation split
train_dataset = shuffled_dataset.skip(1000)
val_dataset = shuffled_dataset.take(1000)
```

**Sharding for distributed training**:
```python
# Split dataset into 4 shards for 4 workers
dataset.shard(num_shards=4, index=0)  # Worker 0 gets shard 0
dataset.shard(num_shards=4, index=1)  # Worker 1 gets shard 1
# etc.
```

### Interleaving Multiple Datasets

From [HuggingFace Datasets Streaming Documentation](https://huggingface.co/docs/datasets/en/stream) (accessed 2025-11-15):

```python
from datasets import interleave_datasets

# Multilingual dataset from multiple sources
es_dataset = load_dataset('allenai/c4', 'es', split='train', streaming=True)
fr_dataset = load_dataset('allenai/c4', 'fr', split='train', streaming=True)

# Alternate between datasets
multilingual = interleave_datasets([es_dataset, fr_dataset])

# Control sampling probabilities
multilingual = interleave_datasets(
    [es_dataset, fr_dataset],
    probabilities=[0.7, 0.3],  # 70% Spanish, 30% French
    seed=42
)

# Stopping strategies
# 'first_exhausted': Stop when first dataset ends (default)
# 'all_exhausted': Cycle through datasets until all seen at least once
multilingual = interleave_datasets(
    [es_dataset, fr_dataset],
    stopping_strategy='all_exhausted'
)
```

### Column Indexing

```python
# Iterate over specific column values
dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")

for text in dataset["text"]:
    print(text)
    break
```

---

## Section 4: Data Processing (.map, .filter, .batch, .shuffle) (~120 lines)

### Map Function Basics

From [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/en/process) (accessed 2025-11-15):

**Single example processing**:
```python
def add_prefix(example):
    example['text'] = 'My text: ' + example['text']
    return example

dataset = dataset.map(add_prefix)
```

**With remove_columns**:
```python
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length')

dataset = dataset.map(
    tokenize,
    remove_columns=['text', 'timestamp', 'url']
)
```

### Batch Processing

From [HuggingFace Datasets Batch Mapping Documentation](https://huggingface.co/docs/datasets/en/about_map_batch) (accessed 2025-11-15):

**Why batch processing?**:
1. **Speed**: Tokenizers are faster with batches (parallelization)
2. **Vectorization**: NumPy/PyTorch operations on batches
3. **Flexibility**: Can change dataset size (split/augment)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def encode_batch(examples):
    # examples is a dict with lists as values
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length'
    )

# Process 1000 examples at a time
dataset = dataset.map(
    encode_batch,
    batched=True,
    batch_size=1000,
    remove_columns=['text']
)
```

**Variable output size**:
```python
# Duplicate each row
def duplicate_rows(batch):
    return {
        "text": [x for x in batch["text"] for _ in range(2)]
    }

duplicated = dataset.map(duplicate_rows, batched=True)
# Input: 1000 rows -> Output: 2000 rows

# Split long documents into chunks
def split_into_chunks(batch):
    chunks = []
    for doc in batch["text"]:
        # Split into 512-token chunks
        words = doc.split()
        for i in range(0, len(words), 512):
            chunks.append(" ".join(words[i:i+512]))
    return {"text": chunks}

chunked = dataset.map(split_into_chunks, batched=True)
```

### Multi-processing

From [GitHub Issue #1992](https://github.com/huggingface/datasets/issues/1992) (accessed 2025-11-15):

```python
# Use multiple CPU cores
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,  # 8 parallel processes
    remove_columns=['text']
)

# Note: Overhead exists - only worth it for:
# - Large datasets (> 10k examples)
# - Expensive operations (tokenization, augmentation)
```

**Performance tips**:
- Small datasets: `num_proc=1` (avoid overhead)
- Large datasets: `num_proc=cpu_count()`
- I/O bound: `num_proc=1` (parallelism doesn't help)
- CPU bound: `num_proc=cpu_count()`

### Filter Operation

```python
# Filter by condition
def is_long_text(example):
    return len(example['text']) > 1000

long_texts = dataset.filter(is_long_text)

# Filter with indices
def every_other(example, idx):
    return idx % 2 == 0

even_dataset = dataset.filter(every_other, with_indices=True)

# Batch filtering (faster for vectorized ops)
def high_quality_batch(batch):
    return [score > 0.9 for score in batch['quality_score']]

filtered = dataset.filter(high_quality_batch, batched=True)
```

### Batching for Training

```python
# Create batches for DataLoader
batched_dataset = dataset.batch(batch_size=32)

for batch in batched_dataset:
    # batch is dict with 32 examples
    print(batch['input_ids'].shape)  # (32, seq_len)
    break

# Drop incomplete final batch
batched_dataset = dataset.batch(batch_size=32, drop_last_batch=True)
```

---

## Section 5: Custom Dataset Scripts (Loading Logic) (~90 lines)

### When to Write Custom Scripts

From [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/en/dataset_script) (accessed 2025-11-15):

**Use custom scripts when**:
- Unsupported file formats
- Complex preprocessing needed before loading
- Multiple data sources need merging
- Custom data validation required

### Basic Custom Dataset Script

```python
import datasets
from pathlib import Path

class MyDataset(datasets.GeneratorBasedBuilder):
    """Custom dataset for my specific format."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="My custom dataset",
            features=datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["neg", "pos"]),
                "metadata": {
                    "author": datasets.Value("string"),
                    "date": datasets.Value("string")
                }
            }),
            homepage="https://example.com",
            citation="...",
        )

    def _split_generators(self, dl_manager):
        # Download or get local files
        data_dir = dl_manager.download_and_extract("https://example.com/data.zip")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(data_dir) / "train.jsonl"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(data_dir) / "test.jsonl"}
            ),
        ]

    def _generate_examples(self, filepath):
        # Yield (key, example) tuples
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                yield idx, {
                    "text": data["text"],
                    "label": data["label"],
                    "metadata": {
                        "author": data.get("author", "unknown"),
                        "date": data.get("date", "")
                    }
                }
```

### Loading Custom Script

```python
# From local file
dataset = load_dataset("path/to/my_dataset.py")

# From Hub (uploaded script)
dataset = load_dataset("username/my_dataset")

# With streaming
dataset = load_dataset("username/my_dataset", streaming=True)
```

### Advanced: Multiple Configurations

```python
class MyDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="small",
            version=VERSION,
            description="Small subset"
        ),
        datasets.BuilderConfig(
            name="full",
            version=VERSION,
            description="Full dataset"
        ),
    ]

    def _split_generators(self, dl_manager):
        if self.config.name == "small":
            data_files = ["train_small.jsonl"]
        else:
            data_files = ["train_full_part*.jsonl"]

        # ... rest of implementation
```

---

## Section 6: Cache Management (Arrow Format, Fingerprinting) (~95 lines)

### Cache Directory Structure

From [HuggingFace Datasets Cache Documentation](https://huggingface.co/docs/datasets/en/cache) (accessed 2025-11-15):

**Default cache locations**:
```bash
~/.cache/huggingface/
├── hub/              # Downloaded files from Hub
│   ├── datasets--username--dataset_name/
│   └── models--username--model_name/
└── datasets/         # Arrow-converted datasets
    └── username___dataset_name/
        ├── default/
        │   └── 0.0.0/
        │       └── dataset.arrow
        └── cache-*.arrow  # Processed versions
```

**Environment variables**:
```bash
# Change all HuggingFace cache
export HF_HOME="/path/to/cache"

# Change only datasets cache
export HF_DATASETS_CACHE="/path/to/datasets_cache"

# Change only Hub downloads
export HF_HUB_CACHE="/path/to/hub_cache"
```

**In code**:
```python
dataset = load_dataset(
    "squad",
    cache_dir="/path/to/custom/cache"
)
```

### Cache Fingerprinting

From [HuggingFace Datasets About Cache Documentation](https://huggingface.co/docs/datasets/en/about_cache) (accessed 2025-11-15):

**How caching works**:
1. Dataset computes fingerprint hash from:
   - Dataset version
   - Applied transformations (map, filter, etc.)
   - Function code hash
   - Parameters

2. Checks if cache file exists for this fingerprint

3. If exists: Load from cache (instant)
   If not: Compute and save to cache

```python
def add_prefix(example):
    example['text'] = 'Prefix: ' + example['text']
    return example

# First call: Computes and caches
dataset1 = dataset.map(add_prefix)

# Second call: Loads from cache (instant)
dataset2 = dataset.map(add_prefix)
```

**Disable caching**:
```python
# Disable for specific operation
dataset = dataset.map(
    add_prefix,
    load_from_cache_file=False  # Always recompute
)

# Disable globally
from datasets import disable_caching
disable_caching()
```

### Cache Cleanup

```python
# Remove cached versions
num_removed = dataset.cleanup_cache_files()
print(f"Removed {num_removed} cache files")

# Force re-download (ignore cache)
from datasets import DownloadMode
dataset = load_dataset(
    "squad",
    download_mode=DownloadMode.FORCE_REDOWNLOAD
)
```

### In-Memory Datasets

From [HuggingFace Datasets Cache Documentation](https://huggingface.co/docs/datasets/en/cache) (accessed 2025-11-15):

```python
# Set in-memory threshold (in bytes)
import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 10 * 1024**3  # 10GB

# Or via environment variable
export HF_DATASETS_IN_MEMORY_MAX_SIZE=10737418240

# Datasets smaller than threshold stay in memory
# Faster operations, no disk I/O
```

---

## Section 7: Multi-processing and Performance Optimization (~100 lines)

### Parallel Map Processing

From [GitHub datasets/issues/1992](https://github.com/huggingface/datasets/issues/1992) (accessed 2025-11-15):

**When parallelization helps**:
```python
# CPU-intensive operations benefit
def expensive_transform(example):
    # Heavy computation (NLP, CV preprocessing)
    return process(example)

# Good: Parallel processing
dataset = dataset.map(
    expensive_transform,
    num_proc=8,
    batched=True,
    batch_size=1000
)

# Bad: Simple operations (overhead dominates)
def simple_transform(example):
    return {"text": example["text"].lower()}

# Better without parallelization
dataset = dataset.map(simple_transform)  # num_proc=1 default
```

**Performance tips**:
```python
# 1. Use batched mode for tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_batch(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

# 10-100x faster than example-by-example
dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    num_proc=4
)

# 2. Remove unnecessary columns early
dataset = dataset.remove_columns(['unnecessary_field'])

# 3. Use streaming for large datasets
dataset = load_dataset("huge_dataset", streaming=True)
```

### DataLoader Integration

From [HuggingFace Datasets PyTorch Documentation](https://huggingface.co/docs/datasets/en/use_with_pytorch) (accessed 2025-11-15):

```python
import torch
from torch.utils.data import DataLoader

# Convert to PyTorch format
dataset = dataset.with_format("torch")

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
```

### Sharding for Distributed Training

```python
from torch.utils.data import DataLoader
from datasets import load_dataset

# Convert to iterable with sharding
dataset = load_dataset("squad")
iterable_dataset = dataset.to_iterable_dataset(num_shards=64)

# Shuffle shards
iterable_dataset = iterable_dataset.shuffle(buffer_size=10_000)

# DataLoader assigns shards to workers
dataloader = DataLoader(
    iterable_dataset,
    batch_size=32,
    num_workers=4  # Each worker gets 64/4 = 16 shards
)
```

### Streaming Performance

```python
# Optimize streaming throughput
dataset = load_dataset("c4", "en", streaming=True, split="train")

# 1. Prefetch batches
batched = dataset.batch(batch_size=32)

# 2. Use multiple workers in DataLoader
dataloader = DataLoader(batched, num_workers=4)

# 3. Shuffle with appropriate buffer
shuffled = dataset.shuffle(buffer_size=10_000)

# 4. Filter early (before expensive ops)
filtered = dataset.filter(lambda x: len(x['text']) > 100)
processed = filtered.map(expensive_tokenization, batched=True)
```

### Memory Optimization

```python
# 1. Clear references after use
del dataset
import gc; gc.collect()

# 2. Use streaming for large data
dataset = load_dataset("large_dataset", streaming=True)

# 3. Process in chunks
for i in range(0, len(dataset), chunk_size):
    chunk = dataset[i:i+chunk_size]
    process(chunk)
    del chunk

# 4. Disable caching for one-off operations
dataset.map(transform, load_from_cache_file=False)
```

---

## Section 8: arr-coc-0-1 Dataset Preparation (VQA, Image-Text Pairs) (~120 lines)

### Context: arr-coc-0-1 Training Requirements

From [arr-coc-0-1 CLAUDE.md](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/CLAUDE.md):

**Dataset needs**:
- Vision-language pairs for Qwen3-VL fine-tuning
- VQA format: question + image → answer
- Image-text pairs: image → caption/description
- Efficient loading for cloud training (Vertex AI)
- Support for streaming large datasets

### VQA Dataset Preparation

```python
from datasets import load_dataset, Features, Value, Image

# Load VQA dataset (e.g., VQAv2, GQA, OKVQA)
dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")

# Expected format:
# {
#   "image": PIL.Image,
#   "question": str,
#   "answers": List[str],
#   "question_id": int
# }

# Preprocess for Qwen3-VL format
def prepare_vqa(example):
    return {
        "image": example["image"],
        "conversations": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answers"][0]}
        ],
        "metadata": {
            "question_id": example["question_id"],
            "dataset": "vqav2"
        }
    }

vqa_dataset = dataset.map(
    prepare_vqa,
    remove_columns=['question', 'answers', 'question_id']
)

# Save to Arrow for efficient loading
vqa_dataset.save_to_disk("data/vqa_prepared")
```

### Image-Text Pairs Dataset

```python
from datasets import load_dataset

# Load captioning datasets
coco = load_dataset("HuggingFaceM4/COCO", split="train")
# {
#   "image": PIL.Image,
#   "captions": List[str]
# }

# Prepare for training
def prepare_captioning(example):
    # Use first caption (or random sample)
    caption = example["captions"][0]

    return {
        "image": example["image"],
        "conversations": [
            {"role": "user", "content": "Describe this image."},
            {"role": "assistant", "content": caption}
        ],
        "metadata": {
            "num_captions": len(example["captions"]),
            "dataset": "coco"
        }
    }

caption_dataset = coco.map(
    prepare_captioning,
    remove_columns=['captions']
)
```

### Combining Multiple Datasets

```python
from datasets import concatenate_datasets, interleave_datasets

# Load multiple VQA datasets
vqav2 = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True)
gqa = load_dataset("HuggingFaceM4/GQA", split="train", streaming=True)
okvqa = load_dataset("HuggingFaceM4/OKVQA", split="train", streaming=True)

# Interleave with sampling probabilities
combined = interleave_datasets(
    [vqav2, gqa, okvqa],
    probabilities=[0.5, 0.3, 0.2],  # More VQAv2
    seed=42,
    stopping_strategy='all_exhausted'
)

# Or concatenate (non-streaming)
vqav2_full = load_dataset("HuggingFaceM4/VQAv2", split="train")
gqa_full = load_dataset("HuggingFaceM4/GQA", split="train")
combined_full = concatenate_datasets([vqav2_full, gqa_full])
```

### Efficient Image Loading

```python
from datasets import Dataset, Features, Image as ImageFeature
from PIL import Image
import io

# Custom dataset with images
def prepare_image_text_dataset(image_paths, captions):
    # Efficient: Store paths, load on-demand
    dataset = Dataset.from_dict({
        "image_path": image_paths,
        "caption": captions
    })

    # Define image feature (loads on access)
    dataset = dataset.cast_column("image_path", ImageFeature())

    return dataset

# Streaming from cloud storage
def load_from_gcs(example):
    from google.cloud import storage
    # Load image from GCS
    client = storage.Client()
    blob = client.bucket("my-bucket").blob(example["image_path"])
    image_bytes = blob.download_as_bytes()
    example["image"] = Image.open(io.BytesIO(image_bytes))
    return example

dataset = dataset.map(load_from_gcs)
```

### Tokenization for VLM Training

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL")

def tokenize_vqa(examples):
    # Process images and text together
    images = examples["image"]
    questions = examples["question"]
    answers = examples["answer"]

    # VLM processor handles both modalities
    encoding = processor(
        images=images,
        text=[f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    return encoding

# Batch tokenization
tokenized_dataset = dataset.map(
    tokenize_vqa,
    batched=True,
    batch_size=8,  # Smaller for images
    remove_columns=dataset.column_names
)
```

### Uploading to HuggingFace Hub

```python
# Push dataset to Hub for Vertex AI access
dataset.push_to_hub(
    "NorthHead/arr-coc-vqa-training",
    private=True,
    token="hf_..."
)

# Load in training script (Vertex AI)
from datasets import load_dataset
training_data = load_dataset(
    "NorthHead/arr-coc-vqa-training",
    split="train",
    streaming=True  # Don't download in cloud VM
)
```

### Vertex AI Integration Pattern

From [Vertex AI Production Documentation](../vertex-ai-production/00-distributed-training-patterns.md):

```python
# In Vertex AI training job
import os
from datasets import load_dataset

# Load from Hub (cached in /gcs/...)
dataset = load_dataset(
    "NorthHead/arr-coc-vqa-training",
    split="train",
    cache_dir=os.environ.get("AIP_STORAGE_URI")  # GCS bucket
)

# Shard for distributed training
from torch.distributed import get_rank, get_world_size
dataset = dataset.shard(
    num_shards=get_world_size(),
    index=get_rank()
)

# Training loop
for batch in dataset:
    # Train
    pass
```

---

## Sources

**HuggingFace Documentation**:
- [Datasets Library Documentation](https://huggingface.co/docs/datasets/en/index) (accessed 2025-11-15)
- [Streaming Documentation](https://huggingface.co/docs/datasets/en/stream) (accessed 2025-11-15)
- [Batch Mapping Documentation](https://huggingface.co/docs/datasets/en/about_map_batch) (accessed 2025-11-15)
- [Cache Management Documentation](https://huggingface.co/docs/datasets/en/cache) (accessed 2025-11-15)
- [Dataset Scripts Documentation](https://huggingface.co/docs/datasets/en/dataset_script) (accessed 2025-11-15)

**GitHub Issues**:
- [datasets#1992: Multi-processing performance](https://github.com/huggingface/datasets/issues/1992) (accessed 2025-11-15)

**Local Documentation**:
- [HuggingFace Hub Datasets Overview](../huggingface-hub/datasets/overview.md)
- [Cloud Storage & BigQuery ML Data](../gcloud-data/00-storage-bigquery-ml-data.md)
- [Vertex AI Production Patterns](../vertex-ai-production/00-distributed-training-patterns.md)
- [arr-coc-0-1 CLAUDE.md](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/CLAUDE.md)

**Additional Resources**:
- [Apache Arrow Documentation](https://arrow.apache.org/docs/python/dataset.html)
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
