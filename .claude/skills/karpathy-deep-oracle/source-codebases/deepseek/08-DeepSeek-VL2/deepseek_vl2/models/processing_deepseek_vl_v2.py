# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal, Optional
import math

import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin
from PIL import Image, ImageOps

from .conversation import get_conv_template

# <claudes_code_comments>
# ** Function List **
# select_best_resolution(image_size, candidate_resolutions) - Greedy optimization: max effective pixels, min wasted padding
# DictOutput.keys() - Dict-like keys() for dataclass attribute access
# DictOutput.__getitem__(item) - Dict-like getitem for dataclass __dict__ access
# DictOutput.__setitem__(key, value) - Dict-like setitem for dataclass __dict__ modification
# VLChatProcessorOutput.__len__() - Returns len(input_ids) for single sample
# VLChatProcessorOutput.to(device, dtype) - NOT IMPLEMENTED (handled in BatchCollateOutput)
# BatchCollateOutput.to(device, dtype) - Moves all tensor fields to specified device/dtype
# ImageTransform.__init__() - Compose pipeline: ToTensor → Normalize(mean, std)
# ImageTransform.__call__(pil_img) - Applies composed transforms [PIL → torch.Tensor[3,H,W]]
# DeepseekVLV2Processor.__init__() - Initializes tokenizer + image transform + special tokens
# DeepseekVLV2Processor.new_chat_template() - Creates conversation template (deepseek/chatml/llama3 formats)
# DeepseekVLV2Processor.format_messages() - Applies SFT template to multi-turn conversations
# DeepseekVLV2Processor.format_messages_v2() - Processes messages → tokens + images + masks
# DeepseekVLV2Processor.format_prompts() - Wraps single prompt in SFT template (User role)
# DeepseekVLV2Processor.bos_id - Property: tokenizer.bos_token_id (beginning of sequence)
# DeepseekVLV2Processor.eos_id - Property: tokenizer.eos_token_id (end of sequence)
# DeepseekVLV2Processor.pad_id - Property: tokenizer.pad_token_id (padding token)
# DeepseekVLV2Processor.encode(text, bos, eos) - Tokenizes text with optional BOS/EOS wrapping
# DeepseekVLV2Processor.decode(t, kwargs) - Decodes token IDs to text via tokenizer.decode()
# DeepseekVLV2Processor.process_one() - Single sample processing: prompt/conversations → VLChatProcessorOutput
# DeepseekVLV2Processor.__call__() - Main API: process_one() + optional batchify()
# DeepseekVLV2Processor.tokenize_with_images() - Core: text + PIL images → tokens + tile tensors + masks
# DeepseekVLV2Processor.batchify() - Collates list of VLChatProcessorOutput → padded BatchCollateOutput
#
# ** Technical Review **
# DeepseekVLV2Processor handles all input preprocessing for the vision-language model,
# including multi-tile image processing, tokenization, and conversation formatting.
# It implements the dynamic resolution tiling strategy that allows efficient processing
# of high-resolution images by splitting them into tiles.
#
# CORE PROCESSING PIPELINE (detailed flow):
# 1. Resolution Selection: select_best_resolution() → (best_width, best_height)
#    - Input: image.size (e.g., (1000, 800)), candidate_resolutions [(384,384), (768,384), (768,768)]
#    - Algorithm: Greedy search maximizing effective_resolution, minimizing wasted_resolution
#    - Output: (768, 768) → 2×2 grid of 384×384 tiles
#
# 2. Tile Extraction: split image into global + local views
#    - Global: ImageOps.pad(image, (384, 384)) → downsampled full image
#    - Local: ImageOps.pad(image, (768, 768)) → crop 2×2 grid of (384, 384) tiles
#    - Total: 1 global + 4 local = 5 tiles (each 384×384)
#
# 3. Image Transform: each tile → tensor
#    - ToTensor: PIL Image → torch.Tensor [3, 384, 384] with values [0, 1]
#    - Normalize: (x - mean) / std → typical range [-1, 1]
#    - Default mean=std=(0.5, 0.5, 0.5): (x - 0.5) / 0.5 → [-1, 1]
#
# 4. Tokenization: text with <image> placeholders → token IDs
#    - Split text by <image> token: "Hello <image> world" → ["Hello ", " world"]
#    - Tokenize each split + insert image token placeholders
#    - Image tokens: h × (w + 1) + 1 + (th×h) × (tw×w + 1) where h=w=12 (after 4× downsample)
#
# 5. Mask Creation: boolean mask for image token positions
#    - images_seq_mask: True where <image> tokens, False for text tokens
#    - Used in prepare_inputs_embeds() to inject vision embeddings
#
# MULTI-TILE RESOLUTION SELECTION (detailed algorithm):
# select_best_resolution(image_size, candidate_resolutions):
#   Input:
#     - image_size: (original_width, original_height) e.g. (1000, 600)
#     - candidate_resolutions: [(384, 384), (384, 768), (768, 384), (768, 768), (768, 1152), ...]
#
#   Algorithm:
#     For each candidate (width, height):
#       1. Compute scale factor: min(width/original_width, height/original_height)
#          - Example: (768, 384) → scale = min(768/1000, 384/600) = min(0.768, 0.64) = 0.64
#       2. Downscale image: (downscaled_width, downscaled_height) = (original × scale)
#          - Example: (1000×0.64, 600×0.64) = (640, 384)
#       3. Compute effective_resolution: min(downscaled_area, original_area)
#          - Example: min(640×384, 1000×600) = min(245760, 600000) = 245760
#       4. Compute wasted_resolution: candidate_area - effective_resolution
#          - Example: (768×384) - 245760 = 294912 - 245760 = 49152 wasted
#       5. Select candidate maximizing effective_resolution, or if tied, minimizing wasted_resolution
#
#   Output: (best_width, best_height)
#     - Example: For (1000, 600), best might be (768, 384)
#     - This fits image well (640×384 used, only 49k pixels wasted)
#     - Tiling: 2×1 grid (num_width_tiles=2, num_height_tiles=1)
#
# Why this strategy?
# - Maximizes pixel coverage while minimizing padding
# - Avoids excessive downsampling (preserves details)
# - Balances computational cost (fewer tiles) vs quality (higher resolution)
#
# GLOBAL VIEW + LOCAL TILES (detailed extraction):
# tokenize_with_images() creates tiles for each image:
#
# Example: 1000×600 image, best_resolution=(768, 384)
#
# 1. Global View:
#    - ImageOps.pad(image, (384, 384), color=mean×255)
#    - Pads to square: 1000×600 → 384×384 (downsampled + letterboxed)
#    - Purpose: Provides full image context (layout, overall scene)
#
# 2. Local Views:
#    - ImageOps.pad(image, (768, 384), color=mean×255)
#    - Pads to target: 1000×600 → 768×384 (minimal padding)
#    - Crop 2×1 grid: [(0,0,384,384), (384,0,768,384)]
#    - Purpose: Preserves high-resolution details (text, faces, fine features)
#
# 3. Tile Order (global_view_pos="head"):
#    images_list = [global_view, local_tile_0, local_tile_1]
#    Total: 3 tiles × 384×384
#
# 4. Spatial Crop Recording:
#    images_spatial_crop = [[2, 1]]  # 2 width tiles × 1 height tile
#    Used in prepare_inputs_embeds() for 2D position encoding
#
# IMAGE TOKEN COUNT FORMULA (critical for understanding sequence length):
# Variables:
#   - image_size: Base resolution (384 default)
#   - patch_size: Vision encoder patch size (16 default)
#   - downsample_ratio: MlpProjector downsampling (2 default)
#   - num_width_tiles (tw), num_height_tiles (th): From best_resolution
#
# Step 1: Compute patch grid size after downsampling
#   h = w = ceil((image_size / patch_size) / downsample_ratio)
#   h = w = ceil((384 / 16) / 2) = ceil(24 / 2) = ceil(12) = 12
#
# Step 2: Global view tokens (with newline separators for 2D layout)
#   global_tokens = h × (w + 1)  # +1 for newline at end of each row
#   global_tokens = 12 × (12 + 1) = 12 × 13 = 156 tokens
#
# Step 3: Separator token between global and local
#   separator_tokens = 1
#
# Step 4: Local view tokens (2D grid with newlines)
#   local_tokens = (th × h) × (tw × w + 1)  # +1 for newlines
#   Example (2×1 grid): local_tokens = (1 × 12) × (2 × 12 + 1) = 12 × 25 = 300 tokens
#
# Step 5: Total image tokens
#   total_tokens = global_tokens + separator_tokens + local_tokens
#   total_tokens = 156 + 1 + 300 = 457 tokens
#
# Example calculations:
#   - 1×1 grid (384×384): 156 + 1 + (12 × 13) = 156 + 1 + 156 = 313 tokens
#   - 2×1 grid (768×384): 156 + 1 + (12 × 25) = 456 + 1 = 457 tokens
#   - 2×2 grid (768×768): 156 + 1 + (24 × 25) = 156 + 1 + 600 = 757 tokens
#   - 3×2 grid (1152×768): 156 + 1 + (24 × 37) = 156 + 1 + 888 = 1045 tokens
#   - 4×4 grid (1536×1536): 156 + 1 + (48 × 49) = 156 + 1 + 2352 = 2509 tokens
#
# IMAGE TRANSFORM PIPELINE (detailed normalization):
# ImageTransform applies two operations:
#
# 1. ToTensor:
#    - Input: PIL Image (H, W, 3) with values [0, 255]
#    - Output: torch.Tensor (3, H, W) with values [0.0, 1.0]
#    - Conversion: pixel / 255.0
#
# 2. Normalize (if enabled):
#    - Formula: output = (input - mean) / std
#    - Default: mean = std = (0.5, 0.5, 0.5)
#    - Effect: [0, 1] → [−1, 1] range
#    - Per-channel: R, G, B normalized independently
#
# Why normalize?
# - Vision Transformers expect normalized inputs (zero-centered)
# - Matches pretraining normalization (SigLIP trained with similar stats)
# - Improves gradient flow and training stability
#
# CONVERSATION FORMATTING (detailed SFT template):
# format_messages() applies supervised fine-tuning template:
#
# Input:
#   conversations = [
#     {"role": "user", "content": "Describe <image>"},
#     {"role": "assistant", "content": "This is a photo of..."}
#   ]
#
# Output (sft_format="deepseek"):
#   "<|begin▁of▁sentence|><|User|>Describe <image>\n\n<|Assistant|>This is a photo of...<|end▁of▁sentence|>"
#
# Template components:
#   - <|begin▁of▁sentence|>: BOS token (marks start of sequence)
#   - <|User|>: User role marker
#   - <|Assistant|>: Assistant role marker
#   - <|end▁of▁sentence|>: EOS token (marks end of sequence, triggers stop)
#
# Grounding tokens (added in __init__):
#   - <|ref|>text<|/ref|>: Marks referring expression for localization
#     Example: "The <|ref|>red car<|/ref|> is on the left"
#   - <|det|>object<|grounding|>[[x1,y1,x2,y2]]<|/det|>: Detection output format
#     Example: "<|det|>car<|grounding|>[[100,50,300,200]]<|/det|>"
#   - <|grounding|>: Triggers grounded captioning mode (output includes bounding boxes)
#
# TOKENIZATION STRATEGY (tokenize_with_images detailed flow):
# Input:
#   conversation = "Hello <image> world <image> end"
#   images = [pil_image1, pil_image2]
#
# Step 1: Split text by <image> token
#   text_splits = ["Hello ", " world ", " end"]
#
# Step 2: For each (text_split, image) pair:
#   a. Tokenize text_split → add to tokenized_str
#      "Hello " → [18472, 29871] (example token IDs)
#      images_seq_mask += [False, False]
#
#   b. Process image:
#      - select_best_resolution() → (768, 384)
#      - Extract global + local tiles → images_list
#      - Compute token count: 156 + 1 + 300 = 457
#      - Add 457 <image> token IDs to tokenized_str
#      - images_seq_mask += [True] × 457
#      - images_spatial_crop += [[2, 1]]
#
#   c. Repeat for image2
#
# Step 3: Tokenize last text_split (" end")
#   " end" → [1095]
#   images_seq_mask += [False]
#
# Step 4: Add BOS/EOS if requested
#   if bos: prepend BOS token ID
#   if eos: append EOS token ID
#
# Output:
#   tokenized_str = [BOS, ...text_tokens..., ...457×image1_tokens..., ...text_tokens..., ...457×image2_tokens..., ...text_tokens..., EOS]
#   images_seq_mask = [False, False, False, True×457, False, False, True×457, False, False]
#   images_list = [5 tiles from image1, 5 tiles from image2] (10 total tiles)
#   images_spatial_crop = [[2, 1], [2, 1]]
#
# SPATIAL CROP ENCODING (position encoding for tiles):
# images_spatial_crop records grid dimensions for each image:
#
# Format: [[num_width_tiles, num_height_tiles], ...]
#
# Examples:
#   - 384×384 (1×1 grid): [[1, 1]]
#   - 768×384 (2×1 grid): [[2, 1]]
#   - 768×768 (2×2 grid): [[2, 2]]
#   - 1536×1536 (4×4 grid): [[4, 4]]
#
# Usage in prepare_inputs_embeds():
#   1. Extract num_width_tiles, num_height_tiles from images_spatial_crop
#   2. Rearrange local tiles: flatten → 2D grid (th×h, tw×w)
#   3. Add newline tokens at end of each row
#   4. Enables LLM to understand spatial layout (top-left, bottom-right, etc.)
#
# BATCH COLLATION (batchify detailed algorithm):
# Input: sample_list = [VLChatProcessorOutput1, VLChatProcessorOutput2, ...]
#
# Step 1: Pad input_ids to max length (padding="left" default)
#   - Sample 1: [BOS, 10, 20, 30, ..., EOS] length=500
#   - Sample 2: [BOS, 15, 25, ..., EOS] length=300
#   - Padded:
#     - Sample 1: [BOS, 10, 20, 30, ..., EOS] (unchanged)
#     - Sample 2: [PAD, PAD, ..., BOS, 15, 25, ..., EOS] (200 pads prepended)
#
# Why left padding?
#   - Causal attention: model attends to left context
#   - Left padding ensures final token at same position across batch
#   - Critical for generation: next token prediction aligned
#
# Step 2: Pad labels (target_ids)
#   - Same as input_ids, but pad_id → ignore_id (-100)
#   - Loss computation skips ignore_id positions
#
# Step 3: Pad images_seq_mask
#   - Same shape as input_ids
#   - pad_id → False (no image injection in padding positions)
#
# Step 4: Create attention_mask
#   - True for real tokens, False for padding
#   - Used in transformer attention to mask padding
#
# Step 5: Pad images to max tile count
#   - Sample 1: 5 tiles [3, 384, 384]
#   - Sample 2: 9 tiles [3, 384, 384]
#   - Padded:
#     - Sample 1: [5 real, 4 zero] → [9, 3, 384, 384]
#     - Sample 2: [9 real] → [9, 3, 384, 384]
#   - Zero padding: torch.zeros((n_pads, 3, 384, 384))
#
# Step 6: Pad images_spatial_crop
#   - Sample 1: [[2, 1]] → [[2, 1], [0, 0]] (pad with [0, 0])
#   - Sample 2: [[2, 2]] → [[2, 2]] (no padding needed)
#   - Zero tiles ([0, 0]) are ignored in prepare_inputs_embeds()
#
# Output: BatchCollateOutput with all fields batched and padded
#
# MEMORY CONSIDERATIONS:
# Token count scaling:
#   - Text: ~100-500 tokens typical
#   - Images: 313-2509 tokens per image (depends on resolution)
#   - Multi-image: 5 images × 757 tokens = 3785 vision tokens + text
#
# Batch memory:
#   - Batch size 8, avg 1000 tokens/sample → 8000 tokens total
#   - KV cache: layers × heads × seq_len × head_dim
#   - Example: 60 layers × 128 heads × 8000 × 128 = ~62GB (fp16)
#   - Solution: incremental_prefilling() chunks to 512 tokens at a time
#
# Tile memory:
#   - 16 tiles/image × [3, 384, 384] × fp16 = 16 × 3 × 384 × 384 × 2 bytes = ~18MB
#   - Batch 8 × 16 tiles = 128 tiles → ~140MB (manageable)
#   - Vision encoder processes all tiles: may need gradient checkpointing for training
#
# SPECIAL TOKEN IDS:
# Added in __init__():
#   1. Padding: <｜▁pad▁｜> → pad_id (e.g., 100000)
#      - Used for batch padding (left-side for generation)
#
#   2. Image: <image> → image_token_id (e.g., 100001)
#      - Placeholder replaced by vision embeddings
#
#   3. Grounding tokens: <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>
#      - Enable object detection and grounded captioning
#      - IDs: 100002-100006 (example)
#
#   4. Chat tokens: <|User|>, <|Assistant|>
#      - Role markers for conversation formatting
#      - IDs: 100007-100008 (example)
#
# Why add_special_tokens?
#   - Prevents tokenizer from splitting these tokens
#   - Ensures consistent token IDs across runs
#   - Required for proper embedding lookup
#
# DESIGN PHILOSOPHY:
# 1. Flexible: Dynamic resolution via candidate_resolutions
#    - Handles arbitrary image sizes (square, landscape, portrait)
#    - Automatically selects best tiling strategy
#
# 2. Modular: Image processing separate from tokenization
#    - ImageTransform can be swapped (different normalization, augmentations)
#    - Tokenizer independent of vision pipeline
#
# 3. Efficient: Minimal padding and downsampling
#    - select_best_resolution() minimizes wasted pixels
#    - Only processes tiles at required resolution (no over-sampling)
#
# 4. Extensible: Easy to add new capabilities
#    - New special tokens: just add to tokenizer
#    - New image transforms: modify ImageTransform pipeline
#    - New tiling strategies: modify select_best_resolution() or add candidates
#
# INTEGRATION WITH DEEPSEEK-VL2 MODEL:
# Processor output → Model input:
#
# 1. VLChatProcessorOutput fields:
#    - input_ids: [seq_len] token IDs (includes <image> placeholders)
#    - images: [n_tiles, 3, 384, 384] tile tensors
#    - images_seq_mask: [seq_len] bool mask (True = image position)
#    - images_spatial_crop: [n_images, 2] tile grid dimensions
#
# 2. Model processing (prepare_inputs_embeds):
#    - images → vision encoder → [n_tiles, 576, 1024]
#    - vision features → projector → [n_tiles, 144, 2048]
#    - input_ids → text embeddings → [seq_len, 2048]
#    - Replace <image> positions with vision embeddings using images_seq_mask
#    - Add 2D newlines using images_spatial_crop
#    - Final: [seq_len_with_expanded_images, 2048] ready for language model
#
# 3. Example flow (2×2 grid image):
#    Input: "Describe <image>" with 768×768 image
#    Processor:
#      - Tokenize: [BOS, 20355, <image>×757, EOS]
#      - images: [5, 3, 384, 384] (1 global + 4 local tiles)
#      - images_spatial_crop: [[2, 2]]
#    Model:
#      - Vision encode: [5, 576, 1024]
#      - Project + downsample: [5, 144, 2048]
#      - Add newlines: 156 (global) + 1 (sep) + 600 (local) = 757 embeddings
#      - Replace <image>×757 with 757 vision embeddings
#      - Final seq: [BOS, "Describe", vision×757, EOS] → [~760, 2048]
# </claudes_code_comments>


def select_best_resolution(image_size, candidate_resolutions):
    # used for cropping
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


# 对于inference sample也可以维护input_ids，反正最后不会用到
@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: torch.LongTensor
    target_ids: torch.LongTensor
    images: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_spatial_crop: torch.LongTensor
    num_image_tokens: List[int]

    def __len__(self):
        return len(self.input_ids)


@dataclass
class BatchCollateOutput(DictOutput):
    sft_format: List[str]
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    images: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_spatial_crop: torch.LongTensor
    seq_lens: List[int]

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.labels = self.labels.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_spatial_crop = self.images_spatial_crop.to(device)
        self.images = self.images.to(device=device, dtype=dtype)
        return self


class ImageTransform(object):
    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            normalize: bool = True
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines = [
            T.ToTensor()
        ]

        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))

        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        x = self.transform(pil_img)
        return x



class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
            self,
            tokenizer: LlamaTokenizerFast,
            candidate_resolutions: Tuple[Tuple[int, int]],
            patch_size: int,
            downsample_ratio: int,
            image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
            image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
            normalize: bool = True,
            image_token: str = "<image>",
            pad_token: str = "<｜▁pad▁｜>",
            add_special_token: bool = False,
            sft_format: str = "deepseek",
            mask_prompt: bool = True,
            ignore_id: int = -100,
            **kwargs,
    ):

        self.candidate_resolutions = candidate_resolutions
        self.image_size = candidate_resolutions[0][0]
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio

        self.image_transform = ImageTransform(mean=image_mean, std=image_std, normalize=normalize)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'  # must set this，padding side with make a difference in batch inference

        # add the pad_token as special token to use 'tokenizer.pad_token' and 'tokenizer.pad_token_id'
        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
        print(f"Add pad token = ['{pad_token}'] to the tokenizer\n"
              f"{pad_token}:{tokenizer.encode(pad_token, add_special_tokens=False)[0]}")

        # add image token
        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        print(f"Add image token = ['{image_token}'] to the tokenizer\n"
              f"{image_token}:{tokenizer.encode(image_token, add_special_tokens=False)[0]}")

        # add five special tokens for grounding-related tasks
        # <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>
        special_tokens = ['<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>']
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Add grounding-related tokens = {special_tokens} to the tokenizer with input_ids\n"
              f"<|ref|>:{tokenizer.encode('<|ref|>', add_special_tokens=False)[0]}\n"
              f"<|/ref|>:{tokenizer.encode('<|/ref|>', add_special_tokens=False)[0]}\n"
              f"<|det|>:{tokenizer.encode('<|det|>', add_special_tokens=False)[0]}\n"
              f"<|/det|>:{tokenizer.encode('<|/det|>', add_special_tokens=False)[0]}\n"
              f"<|grounding|>:{tokenizer.encode('<|grounding|>', add_special_tokens=False)[0]}")

        # add special tokens for SFT data
        special_tokens = ["<|User|>", "<|Assistant|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Add chat tokens = {special_tokens} to the tokenizer with input_ids\n"
              f"<|User|>:{tokenizer.encode('<|User|>', add_special_tokens=False)[0]}\n"
              f"<|Assistant|>:{tokenizer.encode('<|Assistant|>', add_special_tokens=False)[0]}\n")

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(
            tokenizer,
            **kwargs,
        )

    def new_chat_template(self):
        conv = get_conv_template(self.sft_format)
        return conv

    def format_messages(
            self,
            conversations: List[Dict[str, str]],
            sft_format: str = "deepseek",
            system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        Args:
            conversations (List[Dict]): A List of messages.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    def format_messages_v2(self, messages, pil_images, systems=None):
        """play the role of format_messages_v2 and get_images_info in the last version"""
        tokenized_data = []
        masked_tokenized_data = []  # labels
        images_list = []
        images_seq_mask = []
        images_spatial_crop = []
        num_image_tokens = []

        image_index = 0

        conv = get_conv_template(self.sft_format)
        conv_system_message = conv.system_message

        for idx, message in enumerate(messages):
            if idx == 0:
                tokenized_data += [self.bos_id]
                masked_tokenized_data += [self.bos_id]
                images_seq_mask += [False]
                conv.system_message = conv_system_message
            else:
                conv.system_message = ''

            if message['role'] == conv.roles[0] or message['role'] == "user":
                conv.reset_message()
                conv.append_message(conv.roles[0], str(message['content']).strip())
                conv.append_message(conv.roles[1], '')
                formatted_question = conv.get_prompt()
                tokenized_str, images, seq_mask, spatial_crop, n_image_tokens = self.tokenize_with_images(
                    formatted_question,
                    pil_images[image_index: image_index + formatted_question.count(self.image_token)],
                    bos=False,
                    eos=False,
                    cropping=len(pil_images) <= 2
                )
                image_index += formatted_question.count(self.image_token)

                tokenized_data += tokenized_str
                if self.mask_prompt:
                    masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
                else:
                    masked_tokenized_data += tokenized_str
                images_list += images
                images_seq_mask += seq_mask
                images_spatial_crop += spatial_crop
                num_image_tokens += n_image_tokens

            elif message['role'] == conv.roles[1] or message['role'] == "assistant":
                formatted_answer = message['content'].strip()
                assert formatted_answer.count(
                    self.image_token) == 0, f"there should be no {self.image_token} in the assistant's reply, but got {messages}"
                tokenized_str, images, seq_mask, spatial_crop, n_image_tokens = self.tokenize_with_images(
                    formatted_answer,
                    [],
                    bos=False,
                    eos=True,
                    cropping=len(pil_images) <= 2)

                tokenized_data += tokenized_str
                masked_tokenized_data += tokenized_str
                images_seq_mask += seq_mask

            elif message['role'] == 'system' or message['role'] == 'deepseekapi-sys':
                # 如果message里面有system，那就只允许出现在message的第一句，同时conv原本的system就会失效
                assert idx == 0, 'system information should only exist in the begining of the conversation'
                formatted_system = message['content'].strip()
                tokenized_str = self.encode(formatted_system, bos=False, eos=False)
                tokenized_data += tokenized_str
                if self.mask_prompt:
                    masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
                else:
                    masked_tokenized_data += tokenized_str
                seq_mask = [False] * len(tokenized_str)
                images_seq_mask += seq_mask

            else:
                assert False, f"Unknown role: {message['role']}"

        assert len(tokenized_data) == len(
            images_seq_mask), f"format_messages_v2: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"
        assert len(images_spatial_crop) == len(num_image_tokens), f"image number should be compatible"

        return tokenized_data, masked_tokenized_data, images_list, images_seq_mask, images_spatial_crop, num_image_tokens

    def format_prompts(
            self,
            prompts: str,
            sft_format: str = "deepseek",
            system_prompt: str = "",
    ):
        """
        Applies the SFT template to prompts.

        Args:
            prompts (str): the non-sft formatted prompt;
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], prompts.strip())
        conv.append_message(conv.roles[1], "")

        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
            self,
            prompt: str = None,
            conversations: List[Dict[str, str]] = None,
            images: List[Image.Image] = None,
            apply_sft_format: bool = False,
            inference_mode: bool = True,
            system_prompt: str = "",
            **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
                prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if prompt is None:
            # apply sft format
            sft_format = self.format_messages(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=system_prompt,
            )
            tokenized_str, masked_tokenized_str, images_list, images_seq_mask, images_spatial_crop, num_image_tokens = self.format_messages_v2(
                conversations, images)
        else:
            if apply_sft_format:
                sft_format = self.format_prompts(
                    prompts=prompt,
                    sft_format=self.sft_format,
                    system_prompt=system_prompt
                )
            else:
                sft_format = prompt
            tokenized_str, images_list, images_seq_mask, images_spatial_crop, num_image_tokens = self.tokenize_with_images(
                sft_format, images, bos=True, eos=True, cropping=len(images) <= 2)
            masked_tokenized_str = []
            for token_index in tokenized_str:
                if token_index != self.image_token_id:
                    masked_tokenized_str.append(token_index)
                else:
                    masked_tokenized_str.append(self.ignore_id)

        assert len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str), \
            (f"tokenized_str's length {len(tokenized_str)}, input_ids' length {len(masked_tokenized_str)}, "
             f"imags_seq_mask's length {len(images_seq_mask)}, are not equal")

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = self.ignore_id
        input_ids[input_ids < 0] = self.pad_id

        if inference_mode:
            # 去掉结尾的eos token
            assert input_ids[-1] == self.eos_id
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            images = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        else:
            images = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            target_ids=target_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            num_image_tokens=num_image_tokens
        )

        return prepare

    def __call__(
            self,
            *,
            prompt: str = None,
            conversations: List[Dict[str, str]] = None,
            images: List[Image.Image] = None,
            apply_sft_format: bool = False,
            force_batchify: bool = True,
            inference_mode: bool = True,
            system_prompt: str = "",
            **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            force_batchify (bool): force batchify the inputs;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=prompt,
            conversations=conversations,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    def tokenize_with_images(
            self,
            conversation: str,
            images: List[Image.Image],
            bos: bool = True,
            eos: bool = True,
            cropping: bool = True,
    ):
        """Tokenize text with <image> tags."""
        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        images_list, images_seq_mask, images_spatial_crop = [], [], []
        num_image_tokens = []
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            """encode text_sep"""
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
            else:
                best_width, best_height = self.image_size, self.image_size
            # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

            """process the global view"""
            global_view = ImageOps.pad(image, (self.image_size, self.image_size),
                                       color=tuple(int(x * 255) for x in self.image_transform.mean))
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(image, (best_width, best_height),
                                      color=tuple(int(x * 255) for x in self.image_transform.mean))
            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(local_view.crop((j, i, j + self.image_size, i + self.image_size))))

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = best_width // self.image_size, best_height // self.image_size
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            # global views tokens h * (w + 1), 1 is for line seperator
            tokenized_image = [self.image_token_id] * h * (w + 1)
            # add a seperator between global and local views
            tokenized_image += [self.image_token_id]
            # local views tokens, (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_image += [self.image_token_id] * (num_height_tiles * h) * (num_width_tiles * w + 1)

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))
            # print(width_crop_num, height_crop_num, len(tokenized_image)) # test the correctness of the number of image-related tokens

        """process the last text split"""
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos and eos tokens"""
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(
            images_seq_mask), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return tokenized_str, images_list, images_seq_mask, images_spatial_crop, num_image_tokens

    def batchify(
            self,
            sample_list: List[VLChatProcessorOutput],
            padding: Literal["left", "right"] = "left"
    ) -> BatchCollateOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            sample_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.
            padding (str): The padding method. Defaults to "left".

        Returns:
            BatchCollateOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batched_sft_format = [sample.sft_format for sample in sample_list]
        batched_input_ids = [sample.input_ids for sample in sample_list]
        batched_labels = [sample.target_ids for sample in sample_list]
        batched_images_seq_mask = [sample["images_seq_mask"] for sample in sample_list]
        seq_lens = [len(sample) for sample in sample_list]

        """padding input_ids and images_seq_mask"""
        if padding == "left":
            # the tokenizer is default to pad at left
            ## TODO, You're using a LlamaTokenizerFast tokenizer.
            #   Please note that with a fast tokenizer, using the `__call__` method is faster than
            #   using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
            padded_input_ids = self.tokenizer.pad({"input_ids": batched_input_ids})
            batched_input_ids, batched_attention_mask = padded_input_ids["input_ids"], padded_input_ids[
                "attention_mask"].bool()
            batched_labels = self.tokenizer.pad({"input_ids": batched_labels})["input_ids"]
            batched_labels[batched_labels == self.pad_id] = self.ignore_id  # labels正常不会出现pad_id，无需额外保护
            batched_images_seq_mask = self.tokenizer.pad({"input_ids": batched_images_seq_mask})["input_ids"]
            batched_images_seq_mask[batched_images_seq_mask == self.pad_id] = False
        else:
            batched_input_ids = pad_sequence(batched_input_ids, batch_first=True, padding_value=self.pad_id)
            batched_labels = pad_sequence(batched_labels, batch_first=True, padding_value=self.ignore_id)
            batched_images_seq_mask = pad_sequence(batched_images_seq_mask, batch_first=True, padding_value=0)
            batched_attention_mask = batched_input_ids != self.pad_id

        """padding images to max_patch_num"""
        max_n_patches = max(sample["images"].shape[0] for sample in sample_list)
        batched_images = []
        for sample in sample_list:
            images = sample["images"]
            n_pads = max_n_patches - images.shape[0]
            if n_pads > 0:
                pad_images = torch.zeros((n_pads, *images.shape[1:]), dtype=images.dtype)
                images = torch.cat([images, pad_images], dim=0)
            batched_images.append(images)
        batched_images = torch.stack(batched_images, dim=0)

        """padding images_spatial_crop to max_n_images"""
        max_n_images = max(sample["images_spatial_crop"].shape[0] for sample in sample_list)
        batched_images_spatial_crop = []
        for sample in sample_list:
            images_spatial_crop = sample["images_spatial_crop"]
            n_pads = max_n_images - sample["images_spatial_crop"].shape[0]
            if n_pads > 0:
                pad_images_spatial_crop = torch.full((n_pads, 2), 0, dtype=images_spatial_crop.dtype)
                images_spatial_crop = torch.cat([images_spatial_crop, pad_images_spatial_crop], dim=0)
            batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)

        batched_samples = BatchCollateOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            labels=batched_labels,
            images=batched_images,
            images_seq_mask=batched_images_seq_mask,
            images_spatial_crop=batched_images_spatial_crop,
            sft_format=batched_sft_format,
            seq_lens=seq_lens
        )

        return batched_samples
