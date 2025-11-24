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

# <claudes_code_comments>
# ** Function List **
# expand2square(pil_img, bg_color) - Pad image to square by adding background borders
# VLMImageProcessorConfig.__init__(...) - Configuration for image preprocessing pipeline
# VLMImageProcessor.__init__(...) - Initialize processor with normalization parameters
# VLMImageProcessor.resize(pil_img) - Resize and pad image to target size maintaining aspect ratio
# VLMImageProcessor.preprocess(images) - Full preprocessing: resize, rescale, normalize
# VLMImageProcessor.default_shape - Returns expected output tensor shape
#
# ** Technical Review **
# This module handles image preprocessing for Janus's vision encoder, transforming PIL images
# into normalized tensors ready for the vision tower. Ensures consistent input format regardless
# of source image dimensions.
#
# **Preprocessing Pipeline**:
# Input: PIL Image (any size, RGB) → Resize → Pad to Square → Rescale → Normalize → Output: Tensor [3, H, W]
#
# **Core Operations**:
# 1. **Aspect Ratio Preserving Resize**:
#    - Determine longest dimension (width or height)
#    - Resize so longest dimension matches target size (e.g., 384)
#    - Maintain aspect ratio (shorter dimension scales proportionally)
#    - Enforce minimum size (min_size=14) to prevent degenerate cases
#    - Uses bicubic interpolation with antialiasing for quality
#
# 2. **Padding to Square (expand2square)**:
#    - After resize, one dimension may be shorter than target
#    - Pad shorter dimension symmetrically with background color
#    - Background color derived from normalization mean (typically gray ~127,127,127)
#    - Result: Perfect square matching vision encoder's expected input size
#
# 3. **Rescaling** ([0, 255] → [0, 1]):
#    - Multiply pixel values by rescale_factor (default: 1/255)
#    - Converts uint8 image to float in [0, 1] range
#
# 4. **Normalization** (ImageNet statistics):
#    - Default mean: (0.48145466, 0.4578275, 0.40821073) - CLIP/LAION statistics
#    - Default std: (0.26862954, 0.26130258, 0.27577711)
#    - Alternative: IMAGENET_INCEPTION mean/std (0.5, 0.5, 0.5) for different encoders
#    - Formula: x_norm = (x - mean) / std
#    - Optional: can disable normalization via do_normalize=False
#
# **Design Rationale**:
# - **Preserve aspect ratio**: Avoids distorting image content (stretching faces, objects)
# - **Pad instead of crop**: Retains all visual information, no content loss
# - **Consistent square format**: Vision Transformer expects fixed-size square inputs
# - **Bicubic interpolation**: Higher quality than bilinear, critical for downsampling
# - **Antialiasing**: Prevents aliasing artifacts when downscaling large images
# - **ImageNet normalization**: Matches pretraining distribution of vision encoders
#
# **Typical Configuration**:
# - image_size: 384 (standard for SigLIP Large)
# - min_size: 14 (prevents edge cases with tiny images)
# - image_mean/std: IMAGENET_MEAN/STD (CLIP pretraining statistics)
# - rescale_factor: 1/255 (standard uint8 to float conversion)
# - do_normalize: True (essential for good encoder performance)
#
# **Integration with Janus**:
# VLChatProcessor calls VLMImageProcessor to preprocess PIL images:
#   PIL Image → VLMImageProcessor.preprocess() → [3, 384, 384] tensor → vision_tower
#
# **Edge Cases Handled**:
# - Very wide/tall images: Downsample longest side, pad shorter side
# - Small images: Upsample to min_size, then to target size
# - Invalid dimensions (width/height ≤ 0): Raises ValueError
# - Non-RGB images: Handled by PIL.Image (assumes RGB input)
#
# **Performance Notes**:
# - torchvision.transforms.functional.resize faster than PIL.Image.resize
# - Uses channels_first format ([3, H, W]) for efficient GPU processing
# - BatchFeature wrapper ensures proper tensor conversion and device placement
#
# **Why Padding Over Cropping?**
# Padding preserves all visual context - important for understanding tasks where
# peripheral information matters (e.g., "What's on the left side of the image?").
# Cropping would discard information that might be relevant to the query.
#
# The image processor is the first stage in visual understanding, ensuring images
# are formatted correctly for the vision encoder while preserving content integrity.
# </claudes_code_comments>

from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from transformers import AutoImageProcessor, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array
from transformers.utils import logging

logger = logging.get_logger(__name__)

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VLMImageProcessorConfig(PretrainedConfig):
    model_type = "deepseek_vlm"
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)


class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(self, pil_img: Image) -> np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """

        width, height = pil_img.size
        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size),
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")

        pil_img = torchvision.transforms.functional.resize(
            pil_img,
            size,
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=True,
        )

        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        images: List[np.ndarray] = [self.resize(image) for image in images]

        # resacle from [0, 255] -> [0, 1]
        images = [
            self.rescale(
                image=image,
                scale=self.rescale_factor,
                input_data_format="channels_first",
            )
            for image in images
        ]

        # normalize
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format="channels_first",
                )
                for image in images
            ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]


AutoImageProcessor.register(VLMImageProcessorConfig, VLMImageProcessor)


if __name__ == "__main__":
    image_processor = VLMImageProcessor(
        image_size=1024,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
        do_normalize=True,
    )
