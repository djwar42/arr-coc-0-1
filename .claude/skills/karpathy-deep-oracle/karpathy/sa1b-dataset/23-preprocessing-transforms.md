# KNOWLEDGE DROP: Preprocessing Pipelines & Transforms for SA-1B

**Date**: 2025-11-20
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 24)
**File Created**: `sa1b-dataset/23-preprocessing-transforms.md`

---

## What Was Created

**Knowledge File**: Preprocessing Pipelines & Transforms (~700 lines)

**8 Sections**:
1. Synchronized Transform Fundamentals
2. Resize Strategies for Segmentation
3. Data Augmentation for Image-Mask Pairs
4. Normalization and Standardization
5. Albumentations Integration
6. torchvision.transforms.v2 for Segmentation
7. Custom Transform Pipelines
8. **ARR-COC-0-1** (10%): Augmentation for spatial relevance training

---

## Key Insights

### Synchronized Transforms Requirement

**Critical challenge**: Image and mask must receive identical spatial transforms

From [Albumentations Semantic Segmentation Guide](https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/):

> "This guide demonstrates the practical steps for setting up and applying synchronized augmentations for images and masks using Albumentations."

**Why synchronization matters**:
- Rotation must apply same angle to both
- Crop must use same coordinates
- Flip must flip both or neither

### Basic Synchronized Transforms with PyTorch

From [PyTorch Discuss: Apply same transformations](https://discuss.pytorch.org/t/how-do-i-apply-same-transformations-to-image-and-mask/195665):

```python
import torch
import torchvision.transforms.functional as TF
import random

class SynchronizedTransforms:
    """
    Apply identical random transforms to image and mask.
    """

    def __init__(
        self,
        size=(1024, 1024),
        horizontal_flip_p=0.5,
        vertical_flip_p=0.0,
        rotation_degrees=15,
        brightness_range=0.1,
        contrast_range=0.1
    ):
        self.size = size
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image, mask):
        """
        Apply synchronized transforms.

        Args:
            image: PIL Image or tensor
            mask: PIL Image or tensor (single mask)

        Returns:
            Transformed image and mask
        """
        # Resize both
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip
        if random.random() < self.horizontal_flip_p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() < self.vertical_flip_p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Color augmentations (image only)
        if self.brightness_range > 0:
            brightness = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            image = TF.adjust_brightness(image, brightness)

        if self.contrast_range > 0:
            contrast = 1 + random.uniform(-self.contrast_range, self.contrast_range)
            image = TF.adjust_contrast(image, contrast)

        # Convert to tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
```

### Handling Multiple Masks (SA-1B)

```python
class SA1BTransforms:
    """
    Transforms for SA-1B with multiple masks per image.
    """

    def __init__(self, size=(1024, 1024), augment=True):
        self.size = size
        self.augment = augment

    def __call__(self, image, masks, bboxes=None):
        """
        Transform image and all associated masks.

        Args:
            image: PIL Image (H, W, 3)
            masks: numpy array (N, H, W)
            bboxes: Optional numpy array (N, 4)

        Returns:
            Transformed image, masks, and bboxes
        """
        # Store random state for synchronization
        h_flip = random.random() < 0.5 if self.augment else False
        v_flip = random.random() < 0.5 if self.augment else False
        angle = random.uniform(-15, 15) if self.augment else 0

        # Transform image
        image = TF.resize(image, self.size)
        if h_flip:
            image = TF.hflip(image)
        if v_flip:
            image = TF.vflip(image)
        if angle != 0:
            image = TF.rotate(image, angle)

        # Transform each mask identically
        transformed_masks = []
        for mask in masks:
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = TF.resize(mask_pil, self.size, interpolation=TF.InterpolationMode.NEAREST)

            if h_flip:
                mask_pil = TF.hflip(mask_pil)
            if v_flip:
                mask_pil = TF.vflip(mask_pil)
            if angle != 0:
                mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

            transformed_masks.append(np.array(mask_pil))

        masks = np.stack(transformed_masks)

        # Transform bounding boxes
        if bboxes is not None:
            bboxes = self._transform_bboxes(bboxes, h_flip, v_flip, angle)

        # Convert to tensors
        image = TF.to_tensor(image)
        masks = torch.from_numpy(masks).float()

        return image, masks, bboxes

    def _transform_bboxes(self, bboxes, h_flip, v_flip, angle):
        """Transform bounding boxes to match image transforms."""
        # Implementation for bbox transformation
        # ...
        return bboxes
```

### Albumentations Integration

From [Albumentations PyTorch Semantic Segmentation](https://albumentations.ai/docs/examples/pytorch-semantic-segmentation/):

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms(size=1024):
    """
    Albumentations training transforms for SA-1B.
    """
    return A.Compose([
        # Spatial transforms (applied to both image and mask)
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5,
            border_mode=0
        ),
        A.RandomCrop(size, size, p=0.5),

        # Color transforms (image only)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # Convert to tensor
        ToTensorV2()
    ])


def get_validation_transforms(size=1024):
    """
    Albumentations validation transforms (no augmentation).
    """
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# Usage with SA-1B
class SA1BAlbumentationsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # ... load samples

    def __getitem__(self, idx):
        image = np.array(Image.open(img_path))
        masks = load_masks(ann_path)  # Shape: (N, H, W)

        if self.transform:
            # Albumentations handles list of masks
            transformed = self.transform(
                image=image,
                masks=[m for m in masks]
            )
            image = transformed['image']
            masks = torch.stack([torch.from_numpy(m) for m in transformed['masks']])

        return image, masks
```

### torchvision.transforms.v2 for Segmentation

From [PyTorch Transforms Guide](https://docs.pytorch.org/vision/main/transforms.html):

```python
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

def get_torchvision_v2_transforms(size=1024):
    """
    New torchvision v2 transforms with native mask support.
    """
    return T.Compose([
        T.Resize(size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Usage
class SA1BV2Dataset(Dataset):
    def __getitem__(self, idx):
        image = tv_tensors.Image(load_image(idx))
        masks = tv_tensors.Mask(load_masks(idx))

        if self.transform:
            image, masks = self.transform(image, masks)

        return image, masks
```

### Resize Strategies for Segmentation

```python
def resize_with_aspect_ratio(image, masks, target_size=1024):
    """
    Resize preserving aspect ratio with padding.

    Better than direct resize for maintaining object proportions.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize
    image = cv2.resize(image, (new_w, new_h))
    masks = np.stack([
        cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        for m in masks
    ])

    # Pad to target size
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    image = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant'
    )
    masks = np.pad(
        masks,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode='constant'
    )

    return image, masks


def multi_scale_resize(image, masks, scales=[0.5, 1.0, 2.0]):
    """
    Generate multi-scale versions for training.
    """
    results = []

    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        scaled_image = cv2.resize(image, (new_w, new_h))
        scaled_masks = np.stack([
            cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            for m in masks
        ])

        results.append((scaled_image, scaled_masks))

    return results
```

### Advanced Augmentation Techniques

```python
def random_crop_around_mask(image, masks, crop_size=512):
    """
    Random crop centered on a random mask.

    Ensures at least one mask is visible in crop.
    """
    # Select random mask
    mask_idx = random.randint(0, len(masks) - 1)
    mask = masks[mask_idx]

    # Find mask center
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        # Fallback to random crop
        return random_crop(image, masks, crop_size)

    center_y = int(y_coords.mean())
    center_x = int(x_coords.mean())

    # Calculate crop boundaries
    h, w = image.shape[:2]
    half = crop_size // 2

    # Add randomness around center
    center_y += random.randint(-half//2, half//2)
    center_x += random.randint(-half//2, half//2)

    # Clamp to valid range
    y1 = max(0, min(center_y - half, h - crop_size))
    x1 = max(0, min(center_x - half, w - crop_size))
    y2 = y1 + crop_size
    x2 = x1 + crop_size

    # Crop
    cropped_image = image[y1:y2, x1:x2]
    cropped_masks = masks[:, y1:y2, x1:x2]

    return cropped_image, cropped_masks


def copy_paste_augmentation(image1, masks1, image2, masks2):
    """
    Copy-paste augmentation using SA-1B masks.

    Paste objects from image2 onto image1.
    """
    # Select random mask from image2
    paste_idx = random.randint(0, len(masks2) - 1)
    paste_mask = masks2[paste_idx]

    # Extract object
    y_coords, x_coords = np.where(paste_mask > 0)
    if len(y_coords) == 0:
        return image1, masks1

    y1, y2 = y_coords.min(), y_coords.max() + 1
    x1, x2 = x_coords.min(), x_coords.max() + 1

    object_crop = image2[y1:y2, x1:x2]
    mask_crop = paste_mask[y1:y2, x1:x2]

    # Random position in image1
    h, w = image1.shape[:2]
    obj_h, obj_w = object_crop.shape[:2]

    if obj_h >= h or obj_w >= w:
        return image1, masks1

    paste_y = random.randint(0, h - obj_h)
    paste_x = random.randint(0, w - obj_w)

    # Paste object
    result_image = image1.copy()
    for c in range(3):
        result_image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w, c] = \
            object_crop[:, :, c] * mask_crop + \
            image1[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w, c] * (1 - mask_crop)

    # Add mask
    new_mask = np.zeros((h, w), dtype=np.uint8)
    new_mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = mask_crop

    result_masks = np.concatenate([masks1, new_mask[np.newaxis]], axis=0)

    return result_image, result_masks
```

### Complete Preprocessing Pipeline

```python
class SA1BPreprocessingPipeline:
    """
    Complete preprocessing pipeline for SA-1B training.
    """

    def __init__(
        self,
        size=1024,
        training=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225]
    ):
        self.size = size
        self.training = training
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if training:
            self.spatial_transforms = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomCrop(size, size, p=0.3),
            ])

            self.color_transforms = A.Compose([
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
                A.GaussNoise(p=0.3),
            ])
        else:
            self.spatial_transforms = A.Compose([
                A.Resize(size, size)
            ])
            self.color_transforms = None

    def __call__(self, image, masks):
        """
        Apply preprocessing pipeline.

        Args:
            image: numpy array (H, W, 3)
            masks: numpy array (N, H, W)

        Returns:
            Preprocessed image and masks as tensors
        """
        # Spatial transforms (synced)
        transformed = self.spatial_transforms(
            image=image,
            masks=list(masks)
        )
        image = transformed['image']
        masks = np.stack(transformed['masks'])

        # Color transforms (image only)
        if self.color_transforms:
            image = self.color_transforms(image=image)['image']

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - self.normalize_mean) / self.normalize_std

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        masks = torch.from_numpy(masks).float()

        return image, masks
```

---

## Research Performed

**Web sources consulted**:
1. [Albumentations Semantic Segmentation](https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/)
2. [PyTorch Discuss: Same transformations](https://discuss.pytorch.org/t/how-do-i-apply-same-transformations-to-image-and-mask/195665)
3. [StackOverflow: Transform image and mask](https://stackoverflow.com/questions/77997000)
4. [Roboflow: Albumentations Guide](https://blog.roboflow.com/how-to-use-albumentations/)
5. [PyTorch Transforms v2](https://docs.pytorch.org/vision/main/transforms.html)

**Source document**:
- SAM_DATASET_SA1B.md (lines 200-250: preprocessing requirements)

---

## ARR-COC-0-1 Integration (10%)

### Augmentation for Spatial Relevance Training

```python
class ARRCOCAugmentation(SA1BPreprocessingPipeline):
    """
    Extended augmentation for spatial relevance learning.
    """

    def __init__(self, *args, spatial_perturbation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_perturbation = spatial_perturbation

    def __call__(self, image, masks):
        # Standard preprocessing
        image, masks = super().__call__(image, masks)

        if self.training and self.spatial_perturbation:
            # Perturb mask positions slightly
            # Forces model to learn robust spatial representations
            masks = self._perturb_masks(masks)

        return image, masks

    def _perturb_masks(self, masks, max_shift=5):
        """
        Slightly shift mask positions.

        Encourages learning of robust spatial relationships.
        """
        perturbed = []
        for mask in masks:
            shift_y = random.randint(-max_shift, max_shift)
            shift_x = random.randint(-max_shift, max_shift)
            shifted = torch.roll(mask, shifts=(shift_y, shift_x), dims=(0, 1))
            perturbed.append(shifted)

        return torch.stack(perturbed)
```

**Benefits**:
- **Synchronized transforms**: Maintain spatial relationships
- **Robustness**: Slight perturbations improve generalization
- **Multi-scale**: Learn spatial relevance at all granularities

---

## Statistics

- **Lines**: ~700 lines
- **Code examples**: 12+ (transforms, pipelines, augmentation)
- **Sections**: 8 (7 technical + 1 ARR-COC at 10%)
- **Web sources**: 5 cited with URLs
- **Completion time**: ~45 minutes

---

## Batch 4 Complete

**PARTs 19-24** (Data Loading & Preprocessing) completed:
- PART 19: Python Dataset Class Implementation
- PART 20: PyTorch DataLoader Integration
- PART 21: TensorFlow Dataset Integration
- PART 22: RLE Mask Decoding with pycocotools
- PART 23: Mask Visualization Techniques
- PART 24: Preprocessing Pipelines & Transforms

**Next Batch**: PARTs 25-30 (Training on SA-1B)
