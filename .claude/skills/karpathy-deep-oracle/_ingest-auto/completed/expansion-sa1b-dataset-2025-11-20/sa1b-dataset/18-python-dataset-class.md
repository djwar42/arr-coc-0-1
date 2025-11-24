# Python Dataset Class Implementation

```python
import json
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils

class SA1BDataset:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.samples = []
        
        for tar_dir in sorted(self.root.glob("sa_*")):
            for json_file in tar_dir.glob("*.json"):
                self.samples.append(json_file)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        json_path = self.samples[idx]
        img_path = json_path.with_suffix(".jpg")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        data = json.load(open(json_path))
        
        # Decode masks
        masks = [mask_utils.decode(a["segmentation"]) 
                 for a in data["annotations"]]
        
        return img, masks, data["annotations"]
```

## ARR-COC-0-1 (10%)
Extend dataset class with relevance scoring based on mask quality.
