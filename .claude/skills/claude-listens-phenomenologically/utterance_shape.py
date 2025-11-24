#!/usr/bin/env python3
"""
utterance_shape.py - Extract temporal structure via energy envelope
Pure amplitude analysis - no models!
"""
import sys
import argparse
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_shape(audio_path, output_dir=None, threshold=0.02):
    """Extract utterance boundaries from energy envelope."""
    audio_path = Path(audio_path)

    print(f"Extracting shape: {audio_path.name}")
    y, sr = librosa.load(audio_path, sr=None)

    # Output dir
    if output_dir is None:
        output_dir = audio_path.stem
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Energy envelope
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Threshold to find speech/silence
    is_speech = rms > threshold

    # Find utterance boundaries
    utterances = []
    in_utterance = False
    start_time = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_utterance:
            start_time = times[i]
            in_utterance = True
        elif not speech and in_utterance:
            utterances.append((start_time, times[i]))
            in_utterance = False

    # Handle final utterance
    if in_utterance:
        utterances.append((start_time, times[-1]))

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, rms, label='RMS Energy')
    ax.axhline(threshold, color='r', linestyle='--', label='Threshold')
    ax.fill_between(times, 0, 1, where=is_speech, alpha=0.3, transform=ax.get_xaxis_transform())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_title('Utterance Shape')
    ax.legend()
    ax.grid(True, alpha=0.3)

    shape_img_path = analysis_dir / "utterance-shape.png"
    plt.savefig(shape_img_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Write results
    output_path = analysis_dir / "utterance-shape.txt"
    with open(output_path, 'w') as f:
        f.write("UTTERANCE SHAPE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total utterances: {len(utterances)}\n\n")
        for i, (start, end) in enumerate(utterances, 1):
            f.write(f"Utterance {i}: {start:.2f}s → {end:.2f}s ({end-start:.2f}s)\n")
        f.write(f"\nVisualization: analysis/utterance-shape.png\n")

    print(f"✓ Saved: {output_path}")
    print(f"✓ Saved: {shape_img_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract utterance shape')
    parser.add_argument('audio_file')
    parser.add_argument('-t', '--threshold', type=float, default=0.02)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    extract_shape(args.audio_file, args.output, args.threshold)
