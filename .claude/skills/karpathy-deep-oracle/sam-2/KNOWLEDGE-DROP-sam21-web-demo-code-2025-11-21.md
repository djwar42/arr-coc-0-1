# SAM 2.1: Web Demo Code Release

**Repository**: github.com/facebookresearch/sam2/tree/main/demo
**Live Demo**: https://sam2.metademolab.com
**Release Date**: October 2024 (SAM 2.1 + Developer Suite)
**Purpose**: Interactive browser-based video and image segmentation

---

## Overview

Meta released the SAM 2 web demo code as part of the **SAM 2 Developer Suite** in October 2024 alongside SAM 2.1. This includes both frontend and backend code for deploying a local version of the interactive segmentation demo.

From [Meta AI Blog](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/) (accessed 2025-11-21):
- SAM 2.1 includes "a new developer suite with the code for model training and the web demo"
- The demo allows users to "track an object across any video and create fun effects interactively, with as little as a single click on one frame"

---

## Repository Structure

**Location**: `github.com/facebookresearch/sam2/demo/`

From [GitHub repository](https://github.com/facebookresearch/sam2) (accessed 2025-11-21):
- Frontend + backend code released for locally deployable version
- See `demo/README.md` for installation details
- Similar functionality to https://sam2.metademolab.com

**Key Components**:
- Frontend application
- Backend service
- Model inference pipeline
- Video processing utilities

---

## Tech Stack

From [SAM-DETR demo documentation](https://git.wageningenur.nl/strei004/sam-detr/-/blob/master/sam2/demo/README.md) (accessed 2025-11-21):

**Frontend**:
- React TypeScript
- Vite (build tool)
- Interactive video/image interface

**Backend**:
- Python Flask
- Strawberry (GraphQL)
- Model serving infrastructure

**Model Format**:
- PyTorch models (primary)
- ONNX Runtime Web support available (community implementations)

---

## Live Demo Features

**URL**: https://sam2.metademolab.com

From [SAM 2 Demo site](https://sam2.metademolab.com/) (accessed 2025-11-21):
- Track objects across video with single click
- Interactive effect creation
- Real-time segmentation feedback
- Video and image support

**Capabilities**:
- Click-based object selection
- Automatic tracking across frames
- Mask refinement
- Export capabilities

---

## Local Deployment

From [GitHub repository](https://github.com/facebookresearch/sam2) (accessed 2025-11-21):

**Installation**:
```bash
# Install SAM 2 first
pip install -e .

# Navigate to demo directory
cd demo

# See demo/README.md for specific setup
```

**Requirements**:
- SAM 2.1 model checkpoints
- Python environment with Flask
- Node.js for frontend (React/Vite)
- GPU recommended for inference

**Developer Suite** (October 2024 release):
- Model training code
- Web demo code (frontend + backend)
- Inference utilities
- Documentation

---

## ONNX Runtime Web Support

From [Reddit r/computervision](https://www.reddit.com/r/computervision/comments/1gq9so2/sam2_running_in_the_browser_with_onnxruntimeweb/) (accessed 2025-11-21):

**Community Implementations**:
- SAM 2 running in browser with onnxruntime-web
- CPU execution via WebAssembly
- Multi-threading with SharedArrayBuffer
- ONNX model format for web deployment

From [Medium article by Geronimo](https://medium.com/@geronimo7/in-browser-image-segmentation-with-segment-anything-model-2-c72680170d92) (accessed 2025-11-21):
- Integration with JavaScript apps
- WebGPU support for acceleration
- Browser-based inference without backend

**Note**: Official demo uses backend inference, but ONNX exports enable client-side deployment.

---

## Comparison with SAM 1 Demo

**SAM 1 Demo** (segment-anything.com):
- Image-only segmentation
- Static image processing
- Single frame interaction

**SAM 2/2.1 Demo** (sam2.metademolab.com):
- Video + image segmentation
- Object tracking across frames
- Temporal consistency
- Interactive video effects
- Improved performance (SAM 2.1 speed gains)

From [Meta AI announcement](https://ai.meta.com/blog/segment-anything-2/) (accessed 2025-11-21):
- "SAM 2 is a segmentation model that enables fast, precise selection of any object in any video or image"
- Video capability is the major advancement over SAM 1

---

## API Endpoints

**Backend Architecture**:
- Flask server for inference
- GraphQL API via Strawberry
- Video frame processing
- Mask generation endpoints
- Tracking state management

**Inference Flow**:
1. Upload video/image
2. Click to select object
3. Backend runs SAM 2.1 inference
4. Return masks + tracking data
5. Frontend renders results

---

## Developer Resources

From [GitHub repository](https://github.com/facebookresearch/sam2) (accessed 2025-11-21):

**Documentation**:
- `demo/README.md` - Setup instructions
- Model checkpoints download
- API documentation
- Example usage

**Released October 2024**:
- SAM 2.1 improved model
- Complete demo codebase
- Training code
- Developer suite tools

---

## Sources

**Official Meta Resources**:
- [Meta AI SAM 2 Blog](https://ai.meta.com/blog/segment-anything-2/) - Overview and capabilities
- [Meta AI FAIR News](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/) - SAM 2.1 + Developer Suite announcement (accessed 2025-11-21)
- [SAM 2 Live Demo](https://sam2.metademolab.com/) - Interactive web application (accessed 2025-11-21)

**GitHub Repositories**:
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) - Main repository with demo code (accessed 2025-11-21)
- [SAM-DETR demo/README.md](https://git.wageningenur.nl/strei004/sam-detr/-/blob/master/sam2/demo/README.md) - Tech stack documentation (accessed 2025-11-21)

**Community Implementations**:
- [Reddit r/computervision](https://www.reddit.com/r/computervision/comments/1gq9so2/sam2_running_in_the_browser_with_onnxruntimeweb/) - Browser-based SAM 2 with ONNX Runtime Web (accessed 2025-11-21)
- [Medium: Image Segmentation in Browser](https://medium.com/@geronimo7/in-browser-image-segmentation-with-segment-anything-model-2-c72680170d92) - WebGPU integration guide (accessed 2025-11-21)

**Additional References**:
- [Roboflow SAM 2 Overview](https://www.youtube.com/watch?v=Dv003fTyO-Y) - Technical deep dive
- [Encord SAM 2.1 Explained](https://encord.com/blog/sam-2.1-explained/) - Feature analysis (accessed 2025-11-21)
