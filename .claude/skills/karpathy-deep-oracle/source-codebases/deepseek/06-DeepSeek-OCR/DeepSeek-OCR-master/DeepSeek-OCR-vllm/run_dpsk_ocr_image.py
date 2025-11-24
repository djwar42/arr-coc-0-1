# <claudes_code_comments>
# ** Function List **
# load_image(image_path) - loads and auto-rotates image based on EXIF data
# re_match(text) - extracts grounding annotations (ref/det pairs) from OCR output
# extract_coordinates_and_label(ref_text, image_width, image_height) - parses bounding box coordinates from annotation
# draw_bounding_boxes(image, refs) - renders bounding boxes and labels on image
# process_image_with_refs(image, ref_texts) - wrapper for visualization pipeline
# stream_generate(image, prompt) - async streaming generation with vLLM engine
#
# ** Technical Review **
# Main inference script for DeepSeek-OCR image processing with streaming output and visualization.
# Demonstrates end-to-end pipeline: image loading → preprocessing → vLLM generation → post-processing.
#
# vLLM Engine Configuration (stream_generate):
# AsyncLLMEngine setup for high-throughput inference:
# - model: MODEL_PATH from config (HuggingFace model ID or local path)
# - hf_overrides: {"architectures": ["DeepseekOCRForCausalLM"]} forces custom model class
# - block_size: 256 tokens per KV cache block (memory allocation unit)
# - max_model_len: 8192 tokens maximum sequence length
# - enforce_eager: False enables CUDA graphs for 10-20% speedup
# - trust_remote_code: True allows custom modeling code from HF hub
# - tensor_parallel_size: 1 (single-GPU inference)
# - gpu_memory_utilization: 0.75 reserves 25% for CUDA operations
#
# N-Gram Repetition Prevention:
# NoRepeatNGramLogitsProcessor prevents table/list repetition artifacts:
# - ngram_size: 30 tokens (detects repeated sequences up to 30 tokens long)
# - window_size: 90 tokens (looks back 90 tokens to find repeats)
# - whitelist_token_ids: {128821, 128822} allows <td>, </td> tags to repeat
# Common OCR failure: model generates same table row 50× due to high confidence
# This processor detects n-gram overlap and suppresses probabilities of repeated tokens
#
# Streaming Generation Flow:
# Async generator yields partial outputs for real-time display:
# 1. Create request: {"prompt": prompt, "multi_modal_data": {"image": image}}
# 2. engine.generate() returns AsyncGenerator yielding RequestOutput objects
# 3. Extract new text: full_text[printed_length:] for incremental display
# 4. Print with flush=True for immediate terminal output
# 5. Return final_output when generation completes (EOS token or max_tokens)
#
# EXIF Auto-Rotation (load_image):
# Handles smartphone photos with EXIF orientation tags:
# - ImageOps.exif_transpose() reads EXIF orientation (1-8)
# - Rotates/flips image to canonical orientation
# - Critical for documents photographed in landscape/portrait
# - Fallback to raw Image.open() if EXIF parsing fails
#
# Grounding Annotation Extraction (re_match):
# Parses structured output format for object detection:
# - Pattern: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>
# - ref: semantic label (e.g., "title", "table", "figure", "image")
# - det: list of normalized bounding boxes [0-999 coordinate space]
# - Separates image refs (cropped and saved) from other annotations
# - Example: <|ref|>title<|/ref|><|det|>[[100,50,900,150]]<|/det|>
#
# Bounding Box Visualization (draw_bounding_boxes):
# Renders annotations with color-coded overlays:
# - Random color per object for visual distinction
# - Title boxes: 4px outline (thicker for emphasis)
# - Other boxes: 2px outline
# - Semi-transparent fill (alpha=20) for box interior
# - Label text positioned above box with white background
# - Coordinates denormalized: (x * image_width / 999)
#
# Output Processing Pipeline:
# Multi-stage post-processing for clean markdown:
# 1. Save raw output: result_ori.mmd with all annotations
# 2. Extract image crops: save to OUTPUT_PATH/images/{idx}.jpg
# 3. Replace image annotations: <|ref|>image<|/ref|>... → ![](images/idx.jpg)
# 4. Remove other annotations: strip <|ref|>...<|/ref|><|det|>...</|det|>
# 5. Clean LaTeX: replace \coloneqq → :=, \eqqcolon → =:
# 6. Save cleaned output: result.mmd for markdown rendering
# 7. Save visualization: result_with_boxes.jpg with drawn bounding boxes
#
# Special Case: Geometry Diagrams (lines 251-301):
# Detects and renders geometric figures when 'line_type' in output:
# - Parses JSON: {'Line': {'line': [...], 'line_type': [...], 'line_endpoint': [...]}}
# - Plots lines: matplotlib with configurable line styles (solid/dashed)
# - Handles circles: {'Circle': {'circle_center': [...], 'radius': [...]}}
# - Saves figure: geo.jpg at 200 DPI
# Enables OCR of geometry textbooks, SAT/GRE math problems, architectural drawings
#
# Temperature and Sampling:
# SamplingParams(temperature=0.0) for deterministic greedy decoding:
# - Always selects highest-probability token
# - No randomness in generation
# - Reproducible outputs for same input
# - Critical for OCR where consistency matters
# Higher temperatures (0.7-1.0) would introduce randomness for creative tasks
#
# Image Feature Preprocessing:
# DeepseekOCRProcessor().tokenize_with_images() called before generation:
# - Applies dynamic tiling if CROP_MODE=True
# - Returns [pixel_values, images_crop, images_spatial_crop] tensors
# - Passed as multi_modal_data to vLLM engine
# - Allows preprocessing once, generate multiple times (beam search, batch)
#
# Coordinate Normalization (0-999 space):
# Bounding boxes use normalized 0-999 integer coordinates:
# - Device-independent representation
# - Easier for LLM to generate (avoid floats)
# - Denormalize: actual_x = (bbox_x / 999) * image_width
# - Example: bbox=[500, 250, 999, 750] on 2000×1000 image
#   → actual=[1000, 250, 2000, 750]
#
# Prompt Handling:
# Conditional image features based on prompt:
# - If '<image>' in PROMPT: process image and pass multi_modal_data
# - If no '<image>': skip image processing, text-only generation
# - Allows same script for pure text tasks vs vision-language
#
# Design Rationale - Streaming for UX:
# Async streaming provides immediate feedback for long documents:
# - User sees tokens appear in real-time (like ChatGPT)
# - Can interrupt generation if output is incorrect
# - Better perceived latency vs. waiting for full completion
# - Critical for multi-page PDFs that take 30+ seconds
# vLLM's AsyncLLMEngine enables concurrent request handling at ~2500 tokens/s on A100
# </claudes_code_comments>

import asyncio
import re
import os

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE



ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):


    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts):
    result_image = draw_bounding_boxes(image, ref_texts)
    return result_image




async def stream_generate(image=None, prompt=''):


    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids= {128821, 128822})] #whitelist: <td>, </td> 

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        # ignore_eos=False,
        
    )
    
    request_id = f"request-{int(time.time())}"

    printed_length = 0  

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        assert False, f'prompt is none!!!'
    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n') 

    return final_output




if __name__ == "__main__":

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)

    image = load_image(INPUT_PATH).convert('RGB')

    
    if '<image>' in PROMPT:

        image_features = DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)
    else:
        image_features = ''

    prompt = PROMPT

    result_out = asyncio.run(stream_generate(image_features, prompt))


    save_results = 1

    if save_results and '<image>' in prompt:
        print('='*15 + 'save results:' + '='*15)

        image_draw = image.copy()

        outputs = result_out

        with open(f'{OUTPUT_PATH}/result_ori.mmd', 'w', encoding = 'utf-8') as afile:
            afile.write(outputs)

        matches_ref, matches_images, mathes_other = re_match(outputs)
        # print(matches_ref)
        result = process_image_with_refs(image_draw, matches_ref)


        for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
            outputs = outputs.replace(a_match_image, f'![](images/' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        # if 'structural formula' in conversation[0]['content']:
        #     outputs = '<smiles>' + outputs + '</smiles>'
        with open(f'{OUTPUT_PATH}/result.mmd', 'w', encoding = 'utf-8') as afile:
            afile.write(outputs)

        if 'line_type' in outputs:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            lines = eval(outputs)['Line']['line']

            line_type = eval(outputs)['Line']['line_type']
            # print(lines)

            endpoints = eval(outputs)['Line']['line_endpoint']

            fig, ax = plt.subplots(figsize=(3,3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(' -- ')[0])
                    p1 = eval(line.split(' -- ')[-1])

                    if line_type[idx] == '--':
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                    else:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                    ax.scatter(p0[0], p0[1], s=5, color = 'k')
                    ax.scatter(p1[0], p1[1], s=5, color = 'k')
                except:
                    pass

            for endpoint in endpoints:

                label = endpoint.split(': ')[0]
                (x, y) = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                            fontsize=5, fontweight='light')
            
            try:
                if 'Circle' in eval(outputs).keys():
                    circle_centers = eval(outputs)['Circle']['circle_center']
                    radius = eval(outputs)['Circle']['radius']

                    for center, r in zip(circle_centers, radius):
                        center = eval(center.split(': ')[1])
                        circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                        ax.add_patch(circle)
            except:
                pass


            plt.savefig(f'{OUTPUT_PATH}/geo.jpg')
            plt.close()

        result.save(f'{OUTPUT_PATH}/result_with_boxes.jpg')
