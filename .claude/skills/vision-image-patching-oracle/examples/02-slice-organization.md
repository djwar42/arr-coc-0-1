# Slice Organization Example

**Multi-slice coordination and spatial schema implementation**

## LLaVA-UHD Style Slicing

```python
import torch
import math

class ImageSlicing(nn.Module):
    """Variable-sized image slicing like LLaVA-UHD"""
    
    def __init__(self, base_size=336, max_slices=9):
        super().__init__()
        self.base_size = base_size
        self.max_slices = max_slices
    
    def calculate_slice_config(self, h, w):
        """Determine optimal slice configuration"""
        aspect_ratio = w / h
        
        if aspect_ratio > 1.5:  # Wide image
            num_h = 1
            num_w = min(math.ceil(w / self.base_size), self.max_slices)
        elif aspect_ratio < 0.67:  # Tall image
            num_h = min(math.ceil(h / self.base_size), self.max_slices)
            num_w = 1
        else:  # Balanced
            total = min(math.ceil((h * w) / (self.base_size ** 2)), self.max_slices)
            num_h = int(math.sqrt(total / aspect_ratio))
            num_w = math.ceil(total / num_h)
        
        return num_h, num_w
    
    def forward(self, image):
        """
        Slice image into variable-sized regions
        
        Args:
            image: (batch, channels, height, width)
        
        Returns:
            slices: List of slices
            positions: List of (row, col) positions
        """
        batch, channels, h, w = image.shape
        
        # Calculate slice configuration
        num_h, num_w = self.calculate_slice_config(h, w)
        
        # Calculate slice dimensions
        slice_h = h // num_h
        slice_w = w // num_w
        
        slices = []
        positions = []
        
        for i in range(num_h):
            for j in range(num_w):
                # Handle edge slices
                h_start = i * slice_h
                h_end = (i + 1) * slice_h if i < num_h - 1 else h
                w_start = j * slice_w
                w_end = (j + 1) * slice_w if j < num_w - 1 else w
                
                # Extract slice
                slice_img = image[:, :, h_start:h_end, w_start:w_end]
                
                slices.append(slice_img)
                positions.append((i, j))
        
        return slices, positions

# Usage
slicer = ImageSlicing(base_size=336, max_slices=9)
image = torch.randn(1, 3, 672, 1008)  # Wide image

slices, positions = slicer(image)
print(f"Number of slices: {len(slices)}")
print(f"Positions: {positions}")
```

## Spatial Schema Encoding

```python
class SpatialSchemaEncoder(nn.Module):
    """Add spatial position information to slices"""
    
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable position embeddings
        self.row_embed = nn.Embedding(10, embed_dim // 2)
        self.col_embed = nn.Embedding(10, embed_dim // 2)
    
    def create_position_tokens(self, positions):
        """
        Create position tokens for each slice
        
        Args:
            positions: List of (row, col) tuples
        
        Returns:
            pos_tokens: (num_slices, embed_dim)
        """
        rows = torch.tensor([p[0] for p in positions])
        cols = torch.tensor([p[1] for p in positions])
        
        row_embeds = self.row_embed(rows)
        col_embeds = self.col_embed(cols)
        
        # Concatenate row and column embeddings
        pos_tokens = torch.cat([row_embeds, col_embeds], dim=1)
        
        return pos_tokens
    
    def forward(self, slice_tokens, positions):
        """
        Add spatial schema to slice tokens
        
        Args:
            slice_tokens: List of (num_tokens, embed_dim) per slice
            positions: List of (row, col) tuples
        
        Returns:
            organized_tokens: (total_tokens, embed_dim) with position info
        """
        pos_tokens = self.create_position_tokens(positions)
        
        organized = []
        for i, tokens in enumerate(slice_tokens):
            # Prepend position token to each slice
            pos_token = pos_tokens[i].unsqueeze(0)
            slice_with_pos = torch.cat([pos_token, tokens], dim=0)
            organized.append(slice_with_pos)
        
        # Concatenate all slices
        return torch.cat(organized, dim=0)

# Usage
spatial_encoder = SpatialSchemaEncoder(embed_dim=768)

# Example: 6 slices with 64 tokens each
slice_tokens = [torch.randn(64, 768) for _ in range(6)]
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

organized = spatial_encoder(slice_tokens, positions)
print(f"Organized tokens: {organized.shape}")  # (6*65, 768) - 65 = 64 + 1 position token
```

## Complete Pipeline

```python
class SlicedImageProcessor(nn.Module):
    """Complete pipeline: slice → encode → organize"""
    
    def __init__(self):
        super().__init__()
        self.slicer = ImageSlicing()
        self.patch_embed = BasicPatchEmbedding()
        self.spatial_encoder = SpatialSchemaEncoder()
    
    def forward(self, image):
        # 1. Slice image
        slices, positions = self.slicer(image)
        
        # 2. Extract patches from each slice
        slice_tokens = []
        for slice_img in slices:
            tokens = self.patch_embed(slice_img)
            slice_tokens.append(tokens.squeeze(0))
        
        # 3. Add spatial schema
        organized_tokens = self.spatial_encoder(slice_tokens, positions)
        
        return organized_tokens
```

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)

## Related Documents

- [../techniques/02-variable-sized-slices.md](../techniques/02-variable-sized-slices.md)
- [../architecture/05-spatial-encoding.md](../architecture/05-spatial-encoding.md)
- [../models/02-llava-uhd.md](../models/02-llava-uhd.md)
