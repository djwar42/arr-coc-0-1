# SA-1B Data Augmentation: Copy-Paste Techniques

## Overview

Copy-paste augmentation leverages SA-1B's high-quality segmentation masks to create synthetic training data by compositing objects into new scenes. This technique, proven effective for instance segmentation, can dramatically increase training data diversity and model robustness.

---

## Section 1: Copy-Paste Augmentation Technique

### Core Concept

From [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) (CVPR 2021, cited by 1,485):

"For composing new objects into an image, we compute the binary mask (α) of pasted objects using ground-truth annotations and compute the new image as I1 × α + I2 × (1-α)"

### Algorithm Overview

```python
def copy_paste_augmentation(source_image, source_masks, target_image, target_masks):
    """
    Basic copy-paste augmentation algorithm.

    Args:
        source_image: Image to copy objects FROM
        source_masks: List of binary masks for objects in source
        target_image: Image to paste objects INTO
        target_masks: Existing masks in target image

    Returns:
        augmented_image: New composite image
        augmented_masks: Combined masks for all objects
    """
    # Select random subset of source objects to paste
    num_to_paste = random.randint(1, len(source_masks))
    selected_indices = random.sample(range(len(source_masks)), num_to_paste)

    augmented_image = target_image.copy()
    augmented_masks = target_masks.copy()

    for idx in selected_indices:
        mask = source_masks[idx]

        # Extract object pixels
        object_pixels = source_image * mask[:, :, np.newaxis]

        # Apply random transformations
        object_pixels, mask = apply_transforms(object_pixels, mask)

        # Composite onto target
        alpha = mask[:, :, np.newaxis].astype(float)
        augmented_image = object_pixels * alpha + augmented_image * (1 - alpha)

        # Update masks (handle occlusions)
        for i, existing_mask in enumerate(augmented_masks):
            augmented_masks[i] = existing_mask * (1 - mask)
        augmented_masks.append(mask)

    return augmented_image, augmented_masks
```

### Key Properties

1. **Simple yet effective** - No complex blending required
2. **Label-preserving** - Masks transfer directly with objects
3. **Scalable** - Can generate unlimited synthetic samples
4. **Domain-agnostic** - Works across image types

---

## Section 2: Using SA-1B Masks for Compositing

### Why SA-1B Masks Excel for Copy-Paste

**Quality Advantages:**
- High-precision boundaries (pixel-accurate)
- Consistent annotation quality across 1.1B masks
- Multi-granular objects (parts to wholes)
- Class-agnostic design (no label bias)

### Loading SA-1B Masks for Augmentation

```python
import json
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image

class SA1BCopyPasteSource:
    """Load SA-1B data optimized for copy-paste augmentation."""

    def __init__(self, sa1b_root, tar_indices=None):
        self.root = sa1b_root
        self.tar_indices = tar_indices or range(1000)
        self.object_bank = []

        self._build_object_bank()

    def _build_object_bank(self):
        """Pre-load objects from SA-1B for fast access."""
        for tar_idx in self.tar_indices:
            tar_dir = f"sa_{tar_idx:06d}"
            annotation_files = list(Path(self.root / tar_dir).glob("*.json"))

            for ann_file in annotation_files:
                # Load annotation
                with open(ann_file) as f:
                    data = json.load(f)

                # Load corresponding image
                img_path = ann_file.with_suffix('.jpg')
                image = np.array(Image.open(img_path))

                # Extract each mask as potential paste object
                for ann in data['annotations']:
                    # Decode RLE mask
                    rle = ann['segmentation']
                    mask = mask_utils.decode(rle)

                    # Calculate object properties
                    area = mask.sum()
                    bbox = self._mask_to_bbox(mask)

                    # Filter by quality
                    if ann.get('predicted_iou', 0) < 0.7:
                        continue
                    if ann.get('stability_score', 0) < 0.8:
                        continue

                    # Extract object pixels
                    y1, x1, y2, x2 = bbox
                    object_crop = image[y1:y2, x1:x2]
                    mask_crop = mask[y1:y2, x1:x2]

                    self.object_bank.append({
                        'pixels': object_crop,
                        'mask': mask_crop,
                        'area': area,
                        'aspect_ratio': (x2-x1) / max(y2-y1, 1),
                        'source_file': str(ann_file)
                    })

    def sample_objects(self, n=5, size_range=None, aspect_range=None):
        """
        Sample objects from bank with optional filtering.

        Args:
            n: Number of objects to sample
            size_range: (min_area, max_area) tuple
            aspect_range: (min_ratio, max_ratio) tuple

        Returns:
            List of (pixels, mask) tuples
        """
        candidates = self.object_bank

        if size_range:
            min_area, max_area = size_range
            candidates = [o for o in candidates
                         if min_area <= o['area'] <= max_area]

        if aspect_range:
            min_ar, max_ar = aspect_range
            candidates = [o for o in candidates
                         if min_ar <= o['aspect_ratio'] <= max_ar]

        selected = random.sample(candidates, min(n, len(candidates)))
        return [(o['pixels'], o['mask']) for o in selected]

    def _mask_to_bbox(self, mask):
        """Extract bounding box from binary mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return y1, x1, y2+1, x2+1
```

### Advanced Mask Selection

```python
def select_compatible_objects(object_bank, target_image, target_masks, n=3):
    """
    Select objects that will blend well with target scene.

    Considers:
    - Scale compatibility
    - Spatial availability (non-overlapping placement)
    - Visual compatibility (optional)
    """
    target_h, target_w = target_image.shape[:2]
    target_area = target_h * target_w

    # Calculate available space
    occupied_mask = np.zeros((target_h, target_w), dtype=bool)
    for mask in target_masks:
        occupied_mask |= mask.astype(bool)

    available_area = (~occupied_mask).sum()

    selected = []
    for obj in random.sample(object_bank, min(len(object_bank), n * 3)):
        obj_h, obj_w = obj['mask'].shape

        # Check scale compatibility (object shouldn't be too large)
        if obj['area'] > available_area * 0.3:
            continue

        # Check if object fits in image
        if obj_h > target_h * 0.8 or obj_w > target_w * 0.8:
            continue

        selected.append(obj)
        if len(selected) >= n:
            break

    return selected
```

---

## Section 3: Synthetic Scene Generation

### Complete Scene Generation Pipeline

```python
import albumentations as A
from copy_paste import CopyPaste

class SyntheticSceneGenerator:
    """Generate synthetic scenes using SA-1B copy-paste."""

    def __init__(self, sa1b_source, background_source):
        self.objects = sa1b_source
        self.backgrounds = background_source

        # Define augmentation pipeline
        self.transform = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),
            A.PadIfNeeded(512, 512, border_mode=0),
            A.RandomCrop(512, 512),
            A.HorizontalFlip(p=0.5),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)
        ], bbox_params=A.BboxParams(format="coco"))

    def generate_scene(self, num_objects=5):
        """
        Generate a new synthetic scene.

        Args:
            num_objects: Number of objects to place in scene

        Returns:
            scene: Generated image
            masks: Instance masks for all objects
            bboxes: Bounding boxes for all objects
        """
        # Sample background
        background = self.backgrounds.sample_one()

        # Sample objects from SA-1B
        objects = self.objects.sample_objects(n=num_objects)

        # Place objects in scene
        scene = background.copy()
        masks = []
        bboxes = []

        for obj_pixels, obj_mask in objects:
            # Random placement
            placement = self._find_placement(scene.shape, obj_mask.shape, masks)
            if placement is None:
                continue

            y_off, x_off = placement

            # Paste object
            scene, new_mask = self._paste_object(
                scene, obj_pixels, obj_mask, y_off, x_off
            )

            # Update existing masks (handle occlusions)
            masks = self._update_masks_for_occlusion(masks, new_mask)
            masks.append(new_mask)

            # Calculate bbox
            bbox = self._mask_to_bbox(new_mask)
            bboxes.append(bbox)

        return scene, masks, bboxes

    def _find_placement(self, scene_shape, obj_shape, existing_masks):
        """Find valid placement for object avoiding excessive overlap."""
        scene_h, scene_w = scene_shape[:2]
        obj_h, obj_w = obj_shape

        # Try random placements
        for _ in range(100):
            y = random.randint(0, scene_h - obj_h)
            x = random.randint(0, scene_w - obj_w)

            # Check overlap with existing objects
            overlap = 0
            for mask in existing_masks:
                region = mask[y:y+obj_h, x:x+obj_w]
                overlap += region.sum()

            if overlap < obj_h * obj_w * 0.3:  # Allow up to 30% overlap
                return y, x

        return None  # No valid placement found

    def _paste_object(self, scene, obj_pixels, obj_mask, y_off, x_off):
        """Paste object into scene with alpha blending."""
        obj_h, obj_w = obj_mask.shape

        # Create full-size mask
        full_mask = np.zeros(scene.shape[:2], dtype=np.uint8)
        full_mask[y_off:y_off+obj_h, x_off:x_off+obj_w] = obj_mask

        # Alpha composite
        alpha = full_mask[:, :, np.newaxis].astype(float) / 255

        # Resize object to placement region
        scene_region = scene[y_off:y_off+obj_h, x_off:x_off+obj_w]
        alpha_region = alpha[y_off:y_off+obj_h, x_off:x_off+obj_w]

        scene[y_off:y_off+obj_h, x_off:x_off+obj_w] = (
            obj_pixels * alpha_region +
            scene_region * (1 - alpha_region)
        ).astype(np.uint8)

        return scene, full_mask

    def _update_masks_for_occlusion(self, masks, new_mask):
        """Update existing masks to account for occlusion by new object."""
        updated = []
        for mask in masks:
            # Remove pixels covered by new object
            updated_mask = mask * (1 - (new_mask > 0).astype(np.uint8))
            updated.append(updated_mask)
        return updated
```

---

## Section 4: Occlusion Handling

### Realistic Occlusion Patterns

```python
class OcclusionHandler:
    """Handle realistic occlusions in copy-paste augmentation."""

    def __init__(self, depth_model=None):
        self.depth_model = depth_model  # Optional for depth-aware placement

    def handle_occlusion(self, existing_masks, new_mask, strategy='front'):
        """
        Handle occlusion between objects.

        Args:
            existing_masks: List of current masks
            new_mask: New object mask being added
            strategy: 'front' (new in front), 'back', or 'depth'

        Returns:
            updated_masks: Masks adjusted for occlusion
        """
        if strategy == 'front':
            # New object occludes existing
            updated = []
            for mask in existing_masks:
                occluded = mask * (1 - (new_mask > 0).astype(mask.dtype))
                updated.append(occluded)
            updated.append(new_mask)
            return updated

        elif strategy == 'back':
            # New object is occluded by existing
            combined_existing = np.zeros_like(new_mask)
            for mask in existing_masks:
                combined_existing = np.logical_or(combined_existing, mask)

            new_visible = new_mask * (1 - combined_existing.astype(new_mask.dtype))
            return existing_masks + [new_visible]

        elif strategy == 'depth':
            # Use depth ordering
            return self._depth_aware_occlusion(existing_masks, new_mask)

    def _depth_aware_occlusion(self, existing_masks, new_mask):
        """Order objects by estimated depth."""
        # This requires depth estimation model
        if self.depth_model is None:
            return self.handle_occlusion(existing_masks, new_mask, 'front')

        # Get depth values for each mask
        # Objects with higher depth (further) are occluded by closer objects
        # Implementation depends on depth model
        pass

    def simulate_partial_occlusion(self, mask, occlusion_ratio=0.3):
        """
        Simulate partial occlusion of an object.

        Useful for training models to recognize partially visible objects.
        """
        h, w = mask.shape

        # Random occlusion rectangle
        occ_h = int(h * occlusion_ratio * random.uniform(0.5, 1.5))
        occ_w = int(w * occlusion_ratio * random.uniform(0.5, 1.5))

        # Random position (biased toward edges)
        if random.random() < 0.5:
            y = random.choice([0, h - occ_h])
            x = random.randint(0, w - occ_w)
        else:
            y = random.randint(0, h - occ_h)
            x = random.choice([0, w - occ_w])

        # Create occlusion mask
        occluded = mask.copy()
        occluded[y:y+occ_h, x:x+occ_w] = 0

        return occluded
```

---

## Section 5: Scale/Rotation Augmentation

### Geometric Transformations for Pasted Objects

```python
import cv2

class GeometricAugmenter:
    """Apply geometric augmentations to objects before pasting."""

    def __init__(self):
        self.scale_range = (0.5, 2.0)
        self.rotation_range = (-30, 30)
        self.flip_prob = 0.5

    def augment_object(self, pixels, mask):
        """
        Apply random geometric transformations.

        Args:
            pixels: Object pixels (H, W, C)
            mask: Binary mask (H, W)

        Returns:
            aug_pixels: Transformed pixels
            aug_mask: Transformed mask
        """
        # Random scale
        scale = random.uniform(*self.scale_range)
        new_h = int(mask.shape[0] * scale)
        new_w = int(mask.shape[1] * scale)

        pixels = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Random rotation
        angle = random.uniform(*self.rotation_range)
        pixels, mask = self._rotate(pixels, mask, angle)

        # Random flip
        if random.random() < self.flip_prob:
            pixels = np.fliplr(pixels)
            mask = np.fliplr(mask)

        return pixels, mask

    def _rotate(self, pixels, mask, angle):
        """Rotate image and mask by angle degrees."""
        h, w = mask.shape
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix for new size
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Apply rotation
        rotated_pixels = cv2.warpAffine(
            pixels, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        rotated_mask = cv2.warpAffine(
            mask, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            flags=cv2.INTER_NEAREST
        )

        return rotated_pixels, rotated_mask

    def large_scale_jitter(self, pixels, mask, scale_range=(0.1, 2.0)):
        """
        Large Scale Jitter (LSJ) augmentation.

        From Copy-Paste paper: scale objects by factor from 0.1x to 2x
        """
        scale = random.uniform(*scale_range)

        if scale < 1.0:
            # Shrink
            new_h = int(mask.shape[0] * scale)
            new_w = int(mask.shape[1] * scale)
            pixels = cv2.resize(pixels, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            # Enlarge
            new_h = int(mask.shape[0] * scale)
            new_w = int(mask.shape[1] * scale)
            pixels = cv2.resize(pixels, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return pixels, mask
```

---

## Section 6: Instance Paste Strategies

### Strategic Object Placement

```python
class InstancePasteStrategy:
    """Different strategies for placing objects in scenes."""

    def random_paste(self, scene_shape, obj_shape, existing_masks):
        """Completely random placement."""
        h, w = scene_shape[:2]
        obj_h, obj_w = obj_shape

        y = random.randint(0, max(0, h - obj_h))
        x = random.randint(0, max(0, w - obj_w))

        return y, x

    def context_aware_paste(self, scene, obj, existing_masks):
        """
        Place objects in contextually appropriate locations.

        Uses heuristics like:
        - Objects on ground plane
        - Consistent scale with scene
        - Semantic compatibility
        """
        # Simple heuristic: larger objects lower in image
        h, w = scene.shape[:2]
        obj_h, obj_w = obj['mask'].shape

        # Estimate "ground" region (lower third of image)
        if obj['area'] > 10000:  # Large object
            y = random.randint(h // 2, max(h // 2, h - obj_h))
        else:  # Small object
            y = random.randint(0, max(0, h - obj_h))

        x = random.randint(0, max(0, w - obj_w))

        return y, x

    def grid_paste(self, scene_shape, objects, grid_size=(3, 3)):
        """
        Place objects in a grid pattern.

        Useful for creating evaluation datasets with controlled layouts.
        """
        h, w = scene_shape[:2]
        rows, cols = grid_size

        cell_h = h // rows
        cell_w = w // cols

        placements = []
        obj_idx = 0

        for r in range(rows):
            for c in range(cols):
                if obj_idx >= len(objects):
                    break

                # Center object in cell
                obj_h, obj_w = objects[obj_idx]['mask'].shape
                y = r * cell_h + (cell_h - obj_h) // 2
                x = c * cell_w + (cell_w - obj_w) // 2

                placements.append((y, x))
                obj_idx += 1

        return placements

    def crowd_paste(self, scene_shape, objects, density=0.3):
        """
        Create crowded scenes with many overlapping objects.

        Useful for training models on challenging occlusion scenarios.
        """
        h, w = scene_shape[:2]
        placements = []

        for obj in objects:
            obj_h, obj_w = obj['mask'].shape

            # Allow objects to overlap
            y = random.randint(-obj_h // 2, h - obj_h // 2)
            x = random.randint(-obj_w // 2, w - obj_w // 2)

            # Clamp to valid range
            y = max(0, min(y, h - obj_h))
            x = max(0, min(x, w - obj_w))

            placements.append((y, x))

        return placements
```

### Albumentations Integration

From [conradry/copy-paste-aug](https://github.com/conradry/copy-paste-aug):

```python
import albumentations as A
from copy_paste import CopyPaste

# Full augmentation pipeline with copy-paste
transform = A.Compose([
    # Large Scale Jitter
    A.RandomScale(scale_limit=(-0.9, 1), p=1),
    A.PadIfNeeded(256, 256, border_mode=0),
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),

    # Copy-paste augmentation
    CopyPaste(
        blend=True,        # Gaussian blending at edges
        sigma=1,           # Blending sigma
        pct_objects_paste=0.5,  # Fraction of objects to paste
        p=1.0              # Always apply
    )
], bbox_params=A.BboxParams(format="coco"))

# Usage requires 6 arguments
output = transform(
    image=image,
    masks=masks,
    bboxes=bboxes,
    paste_image=paste_image,
    paste_masks=paste_masks,
    paste_bboxes=paste_bboxes
)
```

---

## Section 7: Training with Augmented Data

### Complete Training Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CopyPasteDataset(Dataset):
    """Dataset with copy-paste augmentation for instance segmentation."""

    def __init__(self, base_dataset, sa1b_source, transform=None):
        self.base_dataset = base_dataset
        self.sa1b_source = sa1b_source
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Load base sample
        image, masks, bboxes, labels = self.base_dataset[idx]

        # Sample paste source
        paste_idx = random.randint(0, len(self.sa1b_source) - 1)
        paste_image, paste_masks = self.sa1b_source[paste_idx]

        # Create paste bboxes with mask indices
        paste_bboxes = []
        for i, mask in enumerate(paste_masks):
            bbox = self._mask_to_bbox(mask)
            # Format: (x1, y1, x2, y2, class_id, mask_index)
            paste_bboxes.append((*bbox, 0, i))

        # Apply copy-paste transform
        if self.transform:
            result = self.transform(
                image=image,
                masks=masks,
                bboxes=bboxes,
                paste_image=paste_image,
                paste_masks=paste_masks,
                paste_bboxes=paste_bboxes
            )
            image = result['image']
            masks = result['masks']
            bboxes = result['bboxes']

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        masks = torch.stack([torch.from_numpy(m) for m in masks])

        return image, masks, bboxes, labels

    def _mask_to_bbox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return x1, y1, x2, y2

def train_with_copy_paste(model, train_dataset, val_dataset, config):
    """
    Training loop with copy-paste augmentation.

    Args:
        model: Instance segmentation model
        train_dataset: CopyPasteDataset
        val_dataset: Validation dataset (no copy-paste)
        config: Training configuration
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    for epoch in range(config['epochs']):
        model.train()

        for batch_idx, (images, masks, bboxes, labels) in enumerate(train_loader):
            images = images.to(config['device'])

            # Forward pass
            loss = model(images, masks, bboxes, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation
        val_metrics = evaluate(model, val_dataset)
        print(f"Epoch {epoch} - Val mAP: {val_metrics['mAP']:.4f}")
```

### Blending Techniques

```python
def gaussian_blend(object_pixels, mask, sigma=3):
    """
    Apply Gaussian blending at mask boundaries.

    Creates smoother transitions between pasted objects and background.
    """
    from scipy.ndimage import gaussian_filter

    # Create soft mask
    soft_mask = gaussian_filter(mask.astype(float), sigma=sigma)

    # Normalize
    soft_mask = soft_mask / soft_mask.max()

    return soft_mask

def poisson_blend(source, target, mask, center):
    """
    Poisson image editing for seamless blending.

    More computationally expensive but produces better results.
    """
    import cv2

    # OpenCV's seamlessClone
    result = cv2.seamlessClone(
        source, target, mask * 255,
        center, cv2.NORMAL_CLONE
    )

    return result
```

---

## Section 8: ARR-COC-0-1 Integration (Relevance Realization)

### Data Augmentation for Spatial Grounding Diversity

Copy-paste augmentation with SA-1B masks enhances spatial relevance training:

**Augmentation for Relevance Training:**

```python
class RelevanceAugmentationPipeline:
    """Augment training data for spatial relevance realization."""

    def __init__(self, sa1b_source):
        self.sa1b_source = sa1b_source

    def augment_for_relevance(self, image, text_query, relevant_regions):
        """
        Augment image while maintaining relevance labels.

        Args:
            image: Input image
            text_query: Natural language description
            relevant_regions: Masks of relevant regions

        Returns:
            Augmented samples with updated relevance labels
        """
        augmented_samples = []

        # Strategy 1: Add distractors (irrelevant objects)
        distractor_sample = self._add_distractors(
            image, relevant_regions
        )
        augmented_samples.append({
            'image': distractor_sample['image'],
            'query': text_query,
            'relevant': relevant_regions,
            'distractors': distractor_sample['added_masks']
        })

        # Strategy 2: Copy relevant objects to new scenes
        for bg_image in self._sample_backgrounds(3):
            pasted = self._paste_relevant_objects(
                bg_image, image, relevant_regions
            )
            augmented_samples.append({
                'image': pasted['image'],
                'query': text_query,
                'relevant': pasted['masks']
            })

        # Strategy 3: Scale/position variation
        for _ in range(2):
            varied = self._vary_relevant_positions(
                image, relevant_regions
            )
            augmented_samples.append(varied)

        return augmented_samples

    def _add_distractors(self, image, relevant_regions):
        """Add irrelevant objects to test relevance discrimination."""
        # Sample random SA-1B objects as distractors
        distractors = self.sa1b_source.sample_objects(n=3)

        scene = image.copy()
        added_masks = []

        for obj_pixels, obj_mask in distractors:
            # Find placement avoiding relevant regions
            placement = self._find_non_overlapping_placement(
                scene.shape, obj_mask.shape, relevant_regions
            )
            if placement:
                scene = self._paste_at_location(
                    scene, obj_pixels, obj_mask, placement
                )
                added_masks.append(obj_mask)

        return {'image': scene, 'added_masks': added_masks}

    def _paste_relevant_objects(self, background, source, regions):
        """
        Paste relevant objects into new background.

        Tests if model can identify relevant objects in novel contexts.
        """
        result = background.copy()
        pasted_masks = []

        for region_mask in regions:
            # Extract object from source
            obj_pixels = source * region_mask[:, :, np.newaxis]

            # Paste into background
            placement = self._find_valid_placement(
                result.shape, region_mask.shape, pasted_masks
            )
            result, new_mask = self._paste_with_placement(
                result, obj_pixels, region_mask, placement
            )
            pasted_masks.append(new_mask)

        return {'image': result, 'masks': pasted_masks}
```

**Training Relevance Models with Augmentation:**

```python
class RelevanceTrainer:
    """Train spatial relevance models with copy-paste augmentation."""

    def __init__(self, model, augmentation_pipeline):
        self.model = model
        self.augmenter = augmentation_pipeline

    def train_step(self, batch):
        """
        Single training step with augmented data.

        Uses copy-paste to create:
        1. Hard negatives (similar-looking distractors)
        2. Diverse contexts (same objects, different scenes)
        3. Spatial variations (same objects, different positions)
        """
        images, queries, relevant_masks = batch

        # Generate augmented versions
        augmented_batch = []
        for img, query, masks in zip(images, queries, relevant_masks):
            aug_samples = self.augmenter.augment_for_relevance(
                img, query, masks
            )
            augmented_batch.extend(aug_samples)

        # Train on augmented data
        loss = 0
        for sample in augmented_batch:
            # Forward pass
            pred_relevance = self.model(sample['image'], sample['query'])

            # Compute loss against ground truth relevance
            gt_relevance = self._create_relevance_target(
                sample['relevant'],
                sample.get('distractors', [])
            )
            loss += self.relevance_loss(pred_relevance, gt_relevance)

        return loss / len(augmented_batch)

    def _create_relevance_target(self, relevant_masks, distractor_masks):
        """Create ground truth relevance map."""
        h, w = relevant_masks[0].shape

        relevance = np.zeros((h, w), dtype=np.float32)

        # Relevant regions = 1
        for mask in relevant_masks:
            relevance = np.maximum(relevance, mask.astype(np.float32))

        # Distractors = 0 (explicitly not relevant)
        # Background = 0

        return relevance
```

**Benefits for ARR-COC:**

1. **Increased diversity** - Generate unlimited training scenarios
2. **Hard negative mining** - Add similar-looking distractors
3. **Context invariance** - Same objects in different scenes
4. **Scale robustness** - Objects at various scales
5. **Occlusion handling** - Partially visible relevant objects

---

## Sources

**Primary Source:**
- [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) - Ghiasi et al., CVPR 2021, Cited by 1,485

**Implementation Reference:**
- [conradry/copy-paste-aug](https://github.com/conradry/copy-paste-aug) - GitHub implementation with albumentations integration

**SA-1B Dataset:**
- [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/) - Meta AI, 11M images, 1.1B masks

**Additional References:**
- [SAM as Data Augmentation](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12381) - Wu et al., 2025
- [Albumentations Library](https://albumentations.ai/) - Image augmentation framework
- [Medium: Data Augmentation with Copy-Paste](https://medium.com/data-science/data-augmentation-reducing-model-confusion-with-copy-paste-8f41884ca9d1) - James Buban

---

*Document created: 2025-11-20*
*Part of SA-1B Dataset Mastery Expansion - PART 35*
*~700 lines - Copy-paste augmentation techniques with SA-1B*
