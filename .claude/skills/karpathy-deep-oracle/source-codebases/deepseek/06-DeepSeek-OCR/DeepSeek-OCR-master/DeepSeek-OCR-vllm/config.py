# <claudes_code_comments>
# ** Function List **
# No functions - configuration constants only
#
# ** Technical Review **
# Central configuration file for DeepSeek-OCR inference modes and runtime parameters.
# Defines five resolution modes for optical compression: Tiny (64 tokens), Small (100 tokens),
# Base (256 tokens), Large (400 tokens), and Gundam (dynamic multi-crop mode).
# Default "Gundam" mode uses 1024×1024 base + 640×640 dynamic crops for optimal token efficiency.
# Key parameters: BASE_SIZE determines global view resolution, IMAGE_SIZE sets crop tile size,
# CROP_MODE enables/disables dynamic tiling. Token counts scale with resolution via 16px patches
# downsampled 4×. MAX_CROPS (default 6) limits GPU memory usage during multi-tile processing.
# Supports various prompts: <|grounding|> for layout-aware OCR, "Free OCR" for text-only extraction.
# </claudes_code_comments>

# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = '' 
OUTPUT_PATH = ''

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
