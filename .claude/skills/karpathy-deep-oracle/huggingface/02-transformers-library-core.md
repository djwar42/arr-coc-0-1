# HuggingFace Transformers Library Core Architecture

## Overview

The **Transformers library** by HuggingFace is the de facto standard for working with transformer models in Python. It provides a unified API for loading, using, and customizing thousands of pre-trained models across NLP, vision, audio, and multimodal tasks. The library's core architecture centers on three key abstractions: **AutoClasses** (AutoModel, AutoTokenizer, AutoConfig), **Pipeline API** for high-level inference, and **PretrainedConfig/PreTrainedModel** base classes that enable model customization and sharing.

**Key Innovation**: The `Auto*` classes automatically detect model architecture from config files, enabling users to load any model with a single line of code. This abstraction eliminates the need to know specific model class names (BertModel, GPT2Model, etc.) and makes code portable across different architectures.

From [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/en/model_doc/auto) (accessed 2025-11-15):
> "Instantiating one of AutoConfig, AutoModel, and AutoTokenizer will directly create a class of the relevant architecture. For instance, model = AutoModel.from_pretrained('bert-base-uncased') will create a BertModel instance."

From [HuggingFace Custom Models Guide](https://huggingface.co/docs/transformers/en/custom_models) (accessed 2025-11-15):
> "Transformers models are designed to be customizable. A model's code is fully contained in the model subfolder of the Transformers repository. Each folder contains a modeling.py and a configuration.py file."

**Performance**: The library supports multiple backends (PyTorch, TensorFlow, JAX) and integrates seamlessly with quantization, compilation (torch.compile), and distributed training frameworks.

**Related Knowledge**:
- See [../karpathy/gpt-architecture/00-transformer-fundamentals.md](../karpathy/gpt-architecture/00-transformer-fundamentals.md) for transformer architecture basics
- See [../karpathy/inference-optimization/03-torch-compile-aot-inductor.md](../karpathy/inference-optimization/03-torch-compile-aot-inductor.md) for inference optimization with torch.compile
- See [../karpathy/vision-language/](../karpathy/vision-language/) for vision-language model architectures

---

## Section 1: AutoClasses - The Core Abstraction (~100 lines)

### AutoConfig: Configuration Discovery

`AutoConfig` automatically detects and loads the correct configuration class based on the `model_type` field in `config.json`:

```python
from transformers import AutoConfig

# Load config from Hub
config = AutoConfig.from_pretrained("bert-base-uncased")
# Returns: BertConfig instance

# Load from local directory
config = AutoConfig.from_pretrained("./my-model")
# Reads ./my-model/config.json, instantiates correct config class

# Check model type
print(config.model_type)  # "bert"
```

From [HuggingFace AutoConfig Documentation](https://huggingface.co/docs/transformers/en/model_doc/auto) (accessed 2025-11-15):
> "The model is loaded by supplying a local directory as pretrained_model_name_or_path and a configuration JSON file named config.json is found in the directory."

**How it works**:
1. Read `config.json` from Hub or local directory
2. Extract `model_type` field (e.g., "bert", "gpt2", "vit")
3. Look up corresponding config class in registry
4. Instantiate config class with parameters from JSON
5. Return typed config object

**Model type registry**:
```python
AUTO_CONFIG_MAPPING = {
    "bert": BertConfig,
    "gpt2": GPT2Config,
    "vit": ViTConfig,
    # ... hundreds more
}
```

### AutoTokenizer: Tokenizer Discovery

`AutoTokenizer` loads the correct tokenizer class and vocabulary:

```python
from transformers import AutoTokenizer

# Load tokenizer from Hub
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Returns: BertTokenizerFast instance (Rust-backed)

# Tokenize text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
# Returns: {'input_ids': tensor([[...]), 'attention_mask': tensor([[...]])}

# Check tokenizer type
print(type(tokenizer))  # BertTokenizerFast
```

**Fast vs Slow tokenizers**:

From [HuggingFace Tokenizer Documentation](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) (accessed 2025-11-15):
> "Most of the tokenizers are available in two flavors: a full python implementation and a 'Fast' implementation based on the Rust library Tokenizers. The 'Fast' implementations allows for significant speed-ups in batch tokenization."

```python
# Fast tokenizer (Rust backend, recommended)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 10-100× faster for batch tokenization

# Force slow tokenizer (pure Python)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
# Use when Rust backend unavailable or debugging tokenization
```

**Performance comparison**:
- Fast tokenizer: 1M tokens/sec (batch processing)
- Slow tokenizer: 10K-100K tokens/sec (Python loops)

### AutoModel: Model Architecture Discovery

`AutoModel` loads the base model (no task-specific head):

```python
from transformers import AutoModel

# Load model from Hub
model = AutoModel.from_pretrained("bert-base-uncased")
# Returns: BertModel instance

# Forward pass
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # torch.Size([1, seq_len, 768])
```

**Task-specific AutoModel classes**:

```python
from transformers import (
    AutoModelForSequenceClassification,  # Text classification
    AutoModelForTokenClassification,     # NER, POS tagging
    AutoModelForQuestionAnswering,       # QA tasks
    AutoModelForCausalLM,                # GPT-style generation
    AutoModelForMaskedLM,                # BERT-style masking
    AutoModelForSeq2SeqLM,               # T5, BART
    AutoModelForImageClassification,     # Vision models
    AutoModelForVision2Seq,              # VLMs (image → text)
)

# Load model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3  # Override config
)
```

From [HuggingFace Models Documentation](https://huggingface.co/docs/transformers/en/main_classes/model) (accessed 2025-11-15):
> "The model is loaded by supplying a local directory as pretrained_model_name_or_path and a configuration JSON file named config.json is found in the directory."

---

## Section 2: Pipeline Abstraction for High-Level Inference (~90 lines)

### Pipeline: One-Line Inference

The `pipeline` API provides a high-level interface for common tasks:

```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Named entity recognition
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("My name is Wolfgang and I live in Berlin")
# [{'entity': 'B-PER', 'score': 0.999, 'word': 'Wolfgang', ...}, ...]

# Question answering
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
result = qa(question="What is my name?", context="My name is Clara and I live in Berkeley")
# {'score': 0.987, 'start': 11, 'end': 16, 'answer': 'Clara'}

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=30)
# [{'generated_text': 'Once upon a time, there was a little girl...'}]

# Image classification
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = image_classifier("path/to/image.jpg")
# [{'label': 'tabby cat', 'score': 0.945}, ...]

# Visual question answering (VQA)
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
result = vqa(image="path/to/image.jpg", question="What color is the car?")
# {'score': 0.892, 'answer': 'red'}
```

From [HuggingFace Pipeline Documentation](https://huggingface.co/docs/transformers/main/main_classes/pipelines) (accessed 2025-11-15):
> "This pipeline generates an audio file from an input text and optional other conditional inputs. Unless the model you're using explicitly sets these generation parameters, this pipeline uses default generation parameters."

**Supported tasks** (as of 2024-2025):
- **NLP**: text-classification, token-classification, question-answering, fill-mask, summarization, translation, text-generation, conversational
- **Vision**: image-classification, object-detection, image-segmentation, depth-estimation
- **Audio**: automatic-speech-recognition, text-to-speech, audio-classification
- **Multimodal**: visual-question-answering, document-question-answering, image-to-text, zero-shot-image-classification

### Pipeline Internals: Preprocessing → Model → Postprocessing

```python
# Manual equivalent of pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Preprocess
text = "I love this product!"
inputs = tokenizer(text, return_tensors="pt")

# Model forward
with torch.no_grad():
    outputs = model(**inputs)

# Postprocess
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
label_id = predictions.argmax().item()
score = predictions[0, label_id].item()

print(f"Label: {model.config.id2label[label_id]}, Score: {score:.4f}")
```

**Pipeline parameters**:

```python
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    device=0,  # GPU device
    batch_size=32,  # Batch multiple inputs
    truncation=True,  # Truncate long inputs
    max_length=512,  # Max sequence length
)

# Batch processing
texts = ["I love this", "I hate this", "It's okay"]
results = classifier(texts)
```

---

## Section 3: Model Loading - from_pretrained() Deep Dive (~80 lines)

### Loading from HuggingFace Hub

```python
from transformers import AutoModel

# Load from Hub (default: downloads to ~/.cache/huggingface/hub/)
model = AutoModel.from_pretrained("bert-base-uncased")

# Specify revision (branch, tag, or commit)
model = AutoModel.from_pretrained("bert-base-uncased", revision="main")
model = AutoModel.from_pretrained("bert-base-uncased", revision="v1.0")
model = AutoModel.from_pretrained("bert-base-uncased", revision="a9b2c3d4e5f6")

# Use authentication token for private models
model = AutoModel.from_pretrained("my-org/private-model", use_auth_token=True)
# Or specify token explicitly
model = AutoModel.from_pretrained("my-org/private-model", use_auth_token="hf_...")
```

**What gets downloaded**:
- `config.json`: Model configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `tokenizer.json` or `vocab.txt`: Tokenizer vocabulary
- `tokenizer_config.json`: Tokenizer configuration
- `special_tokens_map.json`: Special token mappings

### Loading from Local Directory

```python
# Load from local directory
model = AutoModel.from_pretrained("./my-local-model")

# Directory structure:
# my-local-model/
#   ├── config.json
#   ├── pytorch_model.bin  (or model.safetensors)
#   ├── tokenizer.json
#   ├── tokenizer_config.json
#   └── special_tokens_map.json
```

### Loading Specific Components

```python
# Load only config (no weights)
config = AutoConfig.from_pretrained("bert-base-uncased")

# Initialize model from config (random weights)
model = AutoModel.from_config(config)

# Load only tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load model with custom config overrides
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    num_hidden_layers=6,  # Override default 12 layers
    hidden_size=512,      # Override default 768
)
```

### Memory-Efficient Loading

```python
# Load in 8-bit quantization (requires bitsandbytes)
model = AutoModel.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto"  # Automatically distribute across GPUs
)

# Load in 4-bit quantization (QLoRA-style)
model = AutoModel.from_pretrained(
    "facebook/opt-6.7b",
    load_in_4bit=True,
    device_map="auto"
)

# Load with CPU offloading (fits large models on limited VRAM)
model = AutoModel.from_pretrained(
    "facebook/opt-6.7b",
    device_map="auto",
    offload_folder="offload",  # Offload to disk
    offload_state_dict=True
)
```

---

## Section 4: Tokenization - Fast Tokenizers and Special Tokens (~90 lines)

### Fast Tokenizers: Rust Backend

From [HuggingFace Tokenizers GitHub](https://github.com/huggingface/tokenizers) (accessed 2025-11-15):
> "Train new vocabularies and tokenize, using today's most used tokenizers. Extremely fast (both training and tokenization), thanks to the Rust implementation."

**Fast vs Slow comparison**:

```python
from transformers import AutoTokenizer
import time

# Fast tokenizer (Rust backend)
fast_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Slow tokenizer (Python backend)
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

# Benchmark
texts = ["This is a test sentence."] * 1000

start = time.time()
fast_results = fast_tokenizer(texts, padding=True, truncation=True)
fast_time = time.time() - start

start = time.time()
slow_results = slow_tokenizer(texts, padding=True, truncation=True)
slow_time = time.time() - start

print(f"Fast tokenizer: {fast_time:.3f}s")  # ~0.05s
print(f"Slow tokenizer: {slow_time:.3f}s")  # ~2.5s
print(f"Speedup: {slow_time/fast_time:.1f}×")  # ~50×
```

**Fast tokenizer features**:
- **Parallelization**: Multi-threaded tokenization (uses all CPU cores)
- **Alignment tracking**: Maps tokens back to original character positions
- **Offset mapping**: Returns character start/end positions for each token
- **Truncation strategies**: Smart truncation with multiple strategies

### Special Tokens Management

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Check special tokens
print(tokenizer.special_tokens_map)
# {
#     'unk_token': '[UNK]',
#     'sep_token': '[SEP]',
#     'pad_token': '[PAD]',
#     'cls_token': '[CLS]',
#     'mask_token': '[MASK]'
# }

# Access special token IDs
print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0

# Tokenize with special tokens (default)
result = tokenizer("Hello world", add_special_tokens=True)
print(tokenizer.convert_ids_to_tokens(result['input_ids']))
# ['[CLS]', 'hello', 'world', '[SEP]']

# Tokenize without special tokens
result = tokenizer("Hello world", add_special_tokens=False)
print(tokenizer.convert_ids_to_tokens(result['input_ids']))
# ['hello', 'world']
```

From [HuggingFace Special Tokens Discussion](https://stackoverflow.com/questions/76937361/when-to-set-add-special-tokens-false-in-huggingface-transformers-tokenizer) (Stack Overflow, accessed 2025-11-15):
> "As far as I understand, setting add_special_tokens=True adds the special tokens like [CLS], [SEP], and padding tokens to the input sequences."

**When to use add_special_tokens=False**:
- Manual token sequence construction
- Concatenating multiple sequences with custom separators
- Token-level tasks (NER, POS tagging) where special tokens aren't part of labels

### Tokenizer Alignment and Offsets

```python
# Fast tokenizers track character positions
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"
result = tokenizer(text, return_offsets_mapping=True)

print(result['offset_mapping'])
# [(0, 0), (0, 5), (6, 11), (0, 0)]  # [CLS], "hello", "world", [SEP]

# Map tokens back to words
word_ids = result.word_ids()
print(word_ids)
# [None, 0, 1, None]  # [CLS]=None, "hello"=word 0, "world"=word 1, [SEP]=None
```

**Use cases**:
- Named Entity Recognition: Align predicted labels with original words
- Question Answering: Find answer span in original text
- Token classification: Map subword predictions to words

---

## Section 5: Custom Model Configuration - config.json and PretrainedConfig (~80 lines)

### PretrainedConfig Base Class

From [HuggingFace Custom Models Guide](https://huggingface.co/docs/transformers/en/custom_models) (accessed 2025-11-15):
> "A custom configuration must subclass PretrainedConfig. This ensures a custom model has all the functionality of a Transformers' model such as from_pretrained(), save_pretrained(), and push_to_hub()."

**Creating a custom configuration**:

```python
from transformers import PretrainedConfig

class ResnetConfig(PretrainedConfig):
    model_type = "resnet"  # Required for AutoConfig support

    def __init__(
        self,
        block_type="bottleneck",
        layers=[3, 4, 6, 3],
        num_classes=1000,
        input_channels=3,
        cardinality=1,
        base_width=64,
        stem_width=64,
        stem_type="",
        avg_down=False,
        **kwargs,
    ):
        # Validation
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or 'bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        # Set attributes
        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down

        # MUST call superclass __init__ with **kwargs
        super().__init__(**kwargs)
```

**Saving and loading config**:

```python
# Create config instance
config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)

# Save to directory
config.save_pretrained("./custom-resnet")
# Creates: ./custom-resnet/config.json

# Load from directory
loaded_config = ResnetConfig.from_pretrained("./custom-resnet")

# Load with AutoConfig (after registration)
from transformers import AutoConfig
AutoConfig.register("resnet", ResnetConfig)
auto_loaded = AutoConfig.from_pretrained("./custom-resnet")
```

### config.json Structure

```json
{
  "model_type": "resnet",
  "block_type": "bottleneck",
  "layers": [3, 4, 6, 3],
  "num_classes": 1000,
  "input_channels": 3,
  "cardinality": 1,
  "base_width": 64,
  "stem_width": 32,
  "stem_type": "deep",
  "avg_down": true,
  "architectures": ["ResnetModelForImageClassification"],
  "auto_map": {
    "AutoConfig": "configuration_resnet.ResnetConfig",
    "AutoModel": "modeling_resnet.ResnetModel",
    "AutoModelForImageClassification": "modeling_resnet.ResnetModelForImageClassification"
  }
}
```

**Key fields**:
- `model_type`: Required for AutoConfig registry
- `architectures`: List of model class names (for AutoModel support)
- `auto_map`: Maps Auto* classes to custom implementation files (for custom models)

---

## Section 6: Model Surgery - Layer Freezing and Head Replacement (~90 lines)

### Freezing Layers

**Freeze backbone, train head** (common fine-tuning pattern):

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classification head
for param in model.classifier.parameters():
    param.requires_grad = True

# Check which parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
# Trainable: 2,301 / 109,483,779 (0.0%)
```

**Freeze specific layers** (progressive unfreezing):

```python
# Freeze embeddings and first 6 layers
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False

# Train layers 7-12 and classifier
for layer in model.bert.encoder.layer[6:]:
    for param in layer.parameters():
        param.requires_grad = True
```

From [HuggingFace Forums - Freeze Lower Layers](https://discuss.huggingface.co/t/freeze-lower-layers-with-auto-classification-model/11386) (accessed 2025-11-15):
> "Yes, in PyTorch freezing layers is quite easy. It can be done as follows: from transformers import AutoModelForSequenceClassification model = AutoModelForSequenceClassification.from_pretrained(...)"

**Layer-wise learning rates** (alternative to freezing):

```python
import torch

# Different learning rates for different layers
optimizer = torch.optim.AdamW([
    {'params': model.bert.embeddings.parameters(), 'lr': 1e-5},
    {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 2e-5},
    {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
])
```

### Replacing Classification Heads

**Replace with custom head**:

```python
from transformers import AutoModel
import torch.nn as nn

# Load base model (no head)
base_model = AutoModel.from_pretrained("bert-base-uncased")

# Create custom classifier
class CustomClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.hidden = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        x = self.dropout(pooled)
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

model = CustomClassifier(base_model, num_labels=3)
```

**Modify existing head in-place**:

```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Replace classifier layer
import torch.nn as nn
model.classifier = nn.Linear(768, 5)  # Change from 2 to 5 labels

# Or add additional layers
model.classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 5)
)
```

---

## Section 7: Integration with PyTorch, TensorFlow, and JAX (~70 lines)

### PyTorch (Primary Backend)

```python
from transformers import AutoModel
import torch

# Load PyTorch model
model = AutoModel.from_pretrained("bert-base-uncased")

# Standard PyTorch operations
model.eval()
model.to("cuda")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Access state dict
state_dict = model.state_dict()

# Save/load with PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

### TensorFlow Integration

```python
from transformers import TFAutoModel
import tensorflow as tf

# Load TensorFlow model
model = TFAutoModel.from_pretrained("bert-base-uncased")

# Standard TensorFlow operations
outputs = model(inputs, training=False)

# Save/load with TensorFlow
model.save_pretrained("./tf-model")
loaded_model = TFAutoModel.from_pretrained("./tf-model")

# Convert PyTorch → TensorFlow
from transformers import AutoModel, TFAutoModel

pt_model = AutoModel.from_pretrained("bert-base-uncased")
tf_model = TFAutoModel.from_pretrained("bert-base-uncased", from_pt=True)
```

### JAX/Flax Integration

```python
from transformers import FlaxAutoModel
import jax.numpy as jnp

# Load Flax model
model = FlaxAutoModel.from_pretrained("bert-base-uncased")

# JAX-style forward pass
outputs = model(**inputs)

# Convert PyTorch → Flax
from transformers import AutoModel, FlaxAutoModel

pt_model = AutoModel.from_pretrained("bert-base-uncased")
flax_model = FlaxAutoModel.from_pretrained("bert-base-uncased", from_pt=True)
```

### Backend Selection

```python
# Automatically use available backend
from transformers import AutoModel

# Will use PyTorch if available, else TensorFlow, else error
model = AutoModel.from_pretrained("bert-base-uncased")

# Force specific backend
model = AutoModel.from_pretrained("bert-base-uncased", from_tf=False, from_flax=False)  # PyTorch only
```

**Backend comparison**:

| Feature | PyTorch | TensorFlow | JAX/Flax |
|---------|---------|------------|----------|
| Dynamic graphs | ✅ Native | ✅ Eager mode | ✅ Native |
| XLA compilation | ✅ torch.compile | ✅ tf.function | ✅ jit |
| TPU support | ❌ Limited | ✅ Native | ✅ Native |
| Ecosystem | ✅ Largest | ✅ Large | 🟡 Growing |
| Transformers support | ✅ Primary | ✅ Full | 🟡 Most models |

---

## Section 8: arr-coc-0-1 Transformers Integration (~70 lines)

### Custom VLM Components with Transformers

The **arr-coc-0-1** project uses Transformers library for the Qwen2-VL vision-language backbone, while implementing custom components for relevance-aware compression.

**Architecture integration**:

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig
from arr_coc.knowing import InformationScorer, SalienceScorer, CouplingScorer
from arr_coc.balancing import TensionBalancer
from arr_coc.attending import BudgetCalculator, TokenAllocator
from arr_coc.realizing import RelevanceRealizer

# Load pretrained VLM backbone
config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
base_model = AutoModel.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Freeze vision encoder (Vervaekean approach: focus on compression, not vision features)
for param in base_model.visual.parameters():
    param.requires_grad = False

# Custom relevance-aware compression module
class ArrCocVLM(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model

        # Vervaekean relevance components
        self.info_scorer = InformationScorer(d_model=config.hidden_size)
        self.salience_scorer = SalienceScorer(d_model=config.hidden_size)
        self.coupling_scorer = CouplingScorer(d_model=config.hidden_size)
        self.balancer = TensionBalancer()
        self.allocator = TokenAllocator(min_tokens=64, max_tokens=400)
        self.realizer = RelevanceRealizer(d_model=config.hidden_size)

    def forward(self, images, text_queries):
        # Extract vision features using frozen encoder
        with torch.no_grad():
            vision_features = self.base_model.visual(images)

        # Measure relevance (3 ways of knowing)
        info_scores = self.info_scorer(vision_features)
        salience_scores = self.salience_scorer(vision_features)
        coupling_scores = self.coupling_scorer(vision_features, text_queries)

        # Balance tensions
        balanced_relevance = self.balancer(info_scores, salience_scores, coupling_scores)

        # Allocate tokens (64-400 per patch based on relevance)
        token_budgets = self.allocator(balanced_relevance)

        # Realize compression
        compressed_features = self.realizer(vision_features, token_budgets)

        # Pass to language decoder
        outputs = self.base_model.language_model(
            inputs_embeds=compressed_features,
            # ... other args
        )
        return outputs
```

From [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/model.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/model.py):
> "ARR-COC extends Qwen2-VL with Vervaekean relevance realization, enabling query-aware visual token compression from 576 tokens (fixed) to 64-400 tokens (dynamic based on relevance)."

**Training with Transformers Trainer**:

```python
from transformers import Trainer, TrainingArguments

# Only train custom relevance components (freeze vision + language)
for param in model.base_model.parameters():
    param.requires_grad = False

for module in [model.info_scorer, model.salience_scorer, model.coupling_scorer,
               model.balancer, model.allocator, model.realizer]:
    for param in module.parameters():
        param.requires_grad = True

# Training arguments
training_args = TrainingArguments(
    output_dir="./arr-coc-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size: 32
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_steps=500,
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,  # Mixed precision training
    push_to_hub=False,
)

# Use Transformers Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**Key integration points**:
1. **AutoModel** loads Qwen2-VL backbone
2. **Layer freezing** preserves pretrained vision features
3. **Custom modules** implement Vervaekean relevance realization
4. **Trainer API** handles distributed training, mixed precision, checkpointing
5. **Hub integration** enables model sharing via `push_to_hub()`

---

## Sources

**HuggingFace Official Documentation:**
- [Auto Classes](https://huggingface.co/docs/transformers/en/model_doc/auto) - HuggingFace Docs (accessed 2025-11-15)
- [Custom Models Guide](https://huggingface.co/docs/transformers/en/custom_models) - HuggingFace Docs (accessed 2025-11-15)
- [PretrainedConfig API](https://huggingface.co/docs/transformers/en/main_classes/configuration) - HuggingFace Docs (accessed 2025-11-15)
- [PreTrainedModel API](https://huggingface.co/docs/transformers/en/main_classes/model) - HuggingFace Docs (accessed 2025-11-15)
- [Tokenizer Documentation](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) - HuggingFace Docs (accessed 2025-11-15)
- [Fast Tokenizers Guide](https://huggingface.co/docs/transformers/en/fast_tokenizers) - HuggingFace Docs (accessed 2025-11-15)

**GitHub Repositories:**
- [huggingface/transformers](https://github.com/huggingface/transformers) - Main Transformers repository (accessed 2025-11-15)
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizers (Rust backend) (accessed 2025-11-15)

**Community Resources:**
- [HuggingFace Forums - Freeze Lower Layers](https://discuss.huggingface.co/t/freeze-lower-layers-with-auto-classification-model/11386) - Community discussion on layer freezing (accessed 2025-11-15)
- [Stack Overflow - Special Tokens](https://stackoverflow.com/questions/76937361/when-to-set-add-special-tokens-false-in-huggingface-transformers-tokenizer) - Discussion on special tokens usage (accessed 2025-11-15)
- [Medium - Understanding Tokenizers](https://medium.com/@danushidk507/analysis-of-the-hugging-face-transformers-library-purpose-and-component-classes-1-8f5bdc7a3b17) - Transformers library analysis (accessed 2025-11-15)

**Source Documents:**
- [../karpathy/gpt-architecture/00-transformer-fundamentals.md](../karpathy/gpt-architecture/00-transformer-fundamentals.md) - Transformer architecture basics
- [../karpathy/inference-optimization/03-torch-compile-aot-inductor.md](../karpathy/inference-optimization/03-torch-compile-aot-inductor.md) - Inference optimization strategies
- [../karpathy/vision-language/](../karpathy/vision-language/) - Vision-language model architectures

**Project Implementation:**
- [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/) - ARR-COC-VIS MVP implementation

**Additional References:**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - PyTorch integration reference
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) - TensorFlow integration reference
- [JAX Documentation](https://jax.readthedocs.io/) - JAX/Flax integration reference
