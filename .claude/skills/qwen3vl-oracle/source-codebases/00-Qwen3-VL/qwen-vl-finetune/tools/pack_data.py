"""Qwen3-VL Data Packing Tool - Pre-calculate Token Counts and Bin-Pack Sequences"""

# <claudes_code_comments>
# ** Function List **
# read_data: Load JSON or JSONL annotation file
# write_data: Save data to JSON or JSONL file
# DataArguments.__init__: Configuration for pixel budgets and video sampling
# MultimodalProcessor.__init__: Initialize processor with data args
# MultimodalProcessor._configure_processor: Create processor with custom pixel limits
# MultimodalProcessor.process_image: Calculate image tokens from grid_thw
# MultimodalProcessor.process_video: Calculate video tokens from grid_thw
# calculate_tokens: Count total tokens for conversation (text + visual)
# pack_data: Bin-pack samples into groups with constant volume constraint
#
# ** Technical Review **
# OFFLINE PREPROCESSING TOOL for optimizing sequence packing efficiency.
# Runs ONCE per dataset to create pre-calculated token counts and packed groupings.
#
# PURPOSE:
# - Problem: Training time wasted calculating token counts repeatedly
# - Solution: Pre-compute token counts and save to *_count.json
# - Problem: Random batching creates uneven sequence lengths → padding waste
# - Solution: Bin-pack similar-length samples → minimize padding
#
# TWO-STAGE PROCESS:
#
# STAGE 1: TOKEN COUNT CALCULATION
# - For each sample: calculate_tokens(conversation, processor, tokenizer)
# - Text tokens: Apply chat template to each turn, count token IDs
# - Image tokens: Run smart_resize → get image_grid_thw → tokens = thw.prod() // 4
# - Video tokens: Sample frames → get video_grid_thw → tokens = thw.prod() // 4
# - Base overhead: 21 tokens (chat template markers)
# - Saves results to annotation_path_count.json
# - Reuses cached counts if file exists (faster re-runs)
#
# TOKEN CALCULATION DETAILS:
# - Text: tokenizer.apply_chat_template → count token IDs
# - Image: grid_thw.prod() // 4 because:
#   * grid_thw = (1, grid_h, grid_w) for single image
#   * Each grid cell = 1 patch of ViT
#   * spatial_merge_size = 2 → merge 2×2 patches → divide by 4
#   * Example: 1024×1024 image → (1, 37, 37) grid → 37×37÷4 ≈ 342 tokens
# - Video: Same logic but grid_thw = (num_frames, grid_h, grid_w)
#   * temporal_patch_size = 2 → merge 2 frames → factor already in grid_thw
#
# STAGE 2: BIN-PACKING
# - Uses binpacking library (first-fit-decreasing algorithm)
# - Goal: Pack samples into groups where sum(tokens) ≤ pack_length (4096)
# - binpacking.to_constant_volume(items, capacity, weight_pos=1)
#   * items: [(index, num_tokens), ...]
#   * capacity: 4096 (max sequence length)
#   * weight_pos=1: Token count is at position 1 in tuple
# - Returns groups: [[(idx0, tokens0), (idx1, tokens1), ...], ...]
# - Each group fits within 4096 tokens
#
# BATCH PROCESSING:
# - Processes datasets in batches of 256 samples
# - Reason: binpacking complexity is O(n log n), batching improves performance
# - ThreadPoolExecutor for parallel token calculation (I/O bound)
#
# OUTPUT FILES:
# 1. annotation_count.json: Original data + "num_tokens" field
#    - Reusable: Load this file next time to skip recalculation
#    - Example: {"conversations": [...], "image": "a.jpg", "num_tokens": 512}
#
# 2. annotation_pack.json: Packed groups (list of lists)
#    - Format: [[sample1, sample2, ...], [sample3, sample4, ...], ...]
#    - Each inner list = one packed sequence
#    - "num_tokens" removed (no longer needed)
#    - Load with LazySupervisedDataset in packed mode
#
# BIN-PACKING BENEFITS:
# - Without packing: [512, 768, 1024, 256] → pad to 1024 each → 4096 total
# - With packing: Group [512+256], [768], [1024] → ~2536 tokens (38% savings)
# - More efficient GPU utilization
# - Faster training (fewer wasted FLOPs on padding)
#
# INTEGRATION WITH TRAINING:
# - Training code (data_processor.py) loads annotation_count.json
# - If data_packing=True: Uses pack_data structure (list of lists)
# - _get_packed_item: Concatenates samples within each group
# - FlattenedDataCollatorForSupervisedDataset: Creates cu_seqlens attention
#
# CONFIGURATION:
# - DataArguments: Same pixel budgets as training
#   * max_pixels: 2048×28×28 = 1,605,632 (larger than training default)
#   * video_max_frame_pixels: 576×28×28 = 451,584
#   * Ensures counts match actual training token usage
# - pack_length: 4096 (typical context length)
# - batch_size: 256 (binpacking batch size, not training batch)
#
# EXAMPLE USAGE:
# 1. Edit datasets dict to point to your annotation files
# 2. Set model_path to your Qwen3-VL checkpoint
# 3. Run: python pack_data.py
# 4. Use generated *_pack.json in training with data_packing=True
#
# TYPICAL WORKFLOW:
# python pack_data.py  # Pre-process datasets
# ↓
# train_qwen.py --dataset_use my_dataset --data_packing True
# ↓
# Training loads annotation_pack.json automatically
#
# PERFORMANCE NOTES:
# - Token calculation: ~1-5 seconds per sample (depends on image/video size)
# - ThreadPoolExecutor parallelizes I/O (image loading, video decoding)
# - Binpacking: ~0.1 seconds per 1000 samples
# - Total preprocessing: Minutes to hours depending on dataset size
# - One-time cost, amortized over all training runs
#
# LIMITATIONS:
# - Assumes all samples fit in pack_length (4096)
# - Samples > 4096 tokens will form singleton groups (no packing benefit)
# - Binpacking is greedy (first-fit-decreasing), not optimal
# - Better than random, not perfect
# </claudes_code_comments>

import json
import os
import numpy as np
from PIL import Image
from copy import deepcopy
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from torchcodec.decoders import VideoDecoder
import binpacking
from tqdm import tqdm
import concurrent.futures
import time


def read_data(file_path):
    """Read JSON or JSONL file"""
    if file_path.endswith(('.json', '.jsonl')):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            return [json.loads(line) for line in f]
    raise ValueError('Please provide a .json or .jsonl file')


def write_data(file_path, data):
    """Write data to JSON or JSONL file"""
    with open(file_path, 'w') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=4)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item) + '\n')


class DataArguments:
    def __init__(self):
        self.max_pixels = 2048 * 28 * 28
        self.min_pixels = 32 * 28 * 28
        self.video_max_frame_pixels = 576 * 28 * 28
        self.video_min_frame_pixels = 144 * 28 * 28
        self.base_interval = 4
        self.video_min_frames = 4
        self.video_max_frames = 8
        self.data_path = ''


class MultimodalProcessor:
    def __init__(self, data_args, base_processor, device='cpu'):
        self.data_args = data_args
        self.base_processor = base_processor
        self.device = device

    def _configure_processor(self, max_val, min_val):
        processor = deepcopy(self.base_processor)
        processor.max_pixels = max_val
        processor.min_pixels = min_val
        processor.size = {'longest_edge': max_val, 'shortest_edge': min_val}
        return processor

    def process_image(self, image_file):
        image_path = os.path.join(self.data_args.data_path, image_file)
        if not os.path.exists(image_path):
            print(f'Image file does not exist: {image_path}')
            return 0
        processor = self._configure_processor(self.data_args.max_pixels, self.data_args.min_pixels)
        image = Image.open(image_path).convert('RGB')
        visual_processed = processor.preprocess(images=image, return_tensors='pt')
        return visual_processed['image_grid_thw'].prod() // 4

    def process_video(self, video_file):
        video_path = os.path.join(self.data_args.data_path, video_file)
        processor = self._configure_processor(self.data_args.video_max_frame_pixels, self.data_args.video_min_frame_pixels)
        decoder = VideoDecoder(video_path, device=self.device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = self.data_args.base_interval
        num_frames_to_sample = round(video_length / interval)
        target_frames = min(max(num_frames_to_sample, self.data_args.video_min_frames), self.data_args.video_max_frames)
        frame_idx = np.unique(np.linspace(0, total_frames - 1, target_frames, dtype=int)).tolist()
        frame_batch = decoder.get_frames_at(indices=frame_idx)
        video_frames_numpy = frame_batch.data.cpu().numpy()
        visual_processed = processor.preprocess(images=None, videos=video_frames_numpy, return_tensors='pt')
        return visual_processed['video_grid_thw'].prod() // 4


def calculate_tokens(conversation, processor, tokenizer):
    total_tokens = 21
    roles = {'human': 'user', 'gpt': 'assistant'}
    for message in conversation['conversations']:
        role = message['from']
        text = message['value']
        conv = [{'role': roles[role], 'content': text}]
        encode_id = tokenizer.apply_chat_template(conv, return_tensors='pt', add_generation_prompt=False)[0]
        total_tokens += len(encode_id)
    if 'image' in conversation:
        images = conversation['image'] if isinstance(conversation['image'], list) else [conversation['image']]
        for image_file in images:
            total_tokens += processor.process_image(image_file)
    elif 'video' in conversation:
        videos = conversation['video'] if isinstance(conversation['video'], list) else [conversation['video']]
        for video_file in videos:
            total_tokens += processor.process_video(video_file)
    return total_tokens


def pack_data(data_list, pack_length):
    # Extract the length of each data item
    lengths = [data["num_tokens"] for data in data_list]
    grouped_indices = binpacking.to_constant_volume(
        list(enumerate(lengths)),  # Explicitly convert to list
        pack_length,
        weight_pos=1
    )
    packed_data = []
    for group in grouped_indices:
        group_data = []
        for index, _ in group:
            new_data = data_list[index].copy()
            new_data.pop("num_tokens", None)
            group_data.append(new_data)
        packed_data.append(group_data)
    return packed_data


datasets = {
    'dummy_dataset': {
        'data_path': '',
        'annotation_path': 'path/to/your/annotation.json'
    }
}

data_args = DataArguments()
model_path = 'path/to/your/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
base_image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
print(f'Successfully loaded model components from {model_path}')

processor = MultimodalProcessor(data_args, base_image_processor, device='cpu')

for dataset_name, config in datasets.items():
    processor.data_args.data_path = config['data_path']
    annotation_path = os.path.join(processor.data_args.data_path, config['annotation_path'])
    print(f'\n--- Processing dataset: {dataset_name} ---')
    print(f'Annotation file path: {annotation_path}')
    print(f'Image configuration: max_pixels={data_args.max_pixels}, min_pixels={data_args.min_pixels}')
    print(f'Video frame configuration: video_max_frame_pixels={data_args.video_max_frame_pixels}, video_min_frame_pixels={data_args.video_min_frame_pixels}')
    if not os.path.exists(annotation_path):
        print(f'Annotation file not found: {annotation_path}')
        continue
    data = read_data(annotation_path)

    count_file_path = annotation_path.replace('.jsonl', '_count.json').replace('.json', '_count.json')
    if os.path.exists(count_file_path):
        print(f"Found pre - calculated token counts, loading data from {count_file_path}.")
        data_with_tokens = read_data(count_file_path)
    else:
        def calculate_and_update(item):
            item['num_tokens'] = calculate_tokens(item, processor, tokenizer)
            return item

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data_with_tokens = list(tqdm(executor.map(calculate_and_update, data), total=len(data), desc=f"Processing {dataset_name} data"))

        # Save the token count results
        write_data(count_file_path, data_with_tokens)
        print(f"Token counts saved to: {count_file_path}")

    # Assume the packing length is 4096
    pack_length = 4096
    # Define the batch size
    batch_size = 256
    all_packed_results = []

    # Record the start time of binpacking
    start_time = time.time()
    for i in range(0, len(data_with_tokens), batch_size):
        batch_data = data_with_tokens[i: i + batch_size]
        batch_packed_result = pack_data(batch_data, pack_length)
        all_packed_results.extend(batch_packed_result)
    # Record the end time of binpacking
    end_time = time.time()

    # Calculate the time spent on binpacking
    binpack_time = end_time - start_time
    print(f"Time spent on binpacking: {binpack_time:.4f} seconds")

    # Save the packed results as a JSON file
    pack_output_path = annotation_path.replace('.jsonl', '_pack.json').replace('.json', '_pack.json')
    with open(pack_output_path, 'w', encoding='utf-8') as file:
        json.dump(all_packed_results, file, indent=2)
    print(f"Packed results saved to: {pack_output_path}")