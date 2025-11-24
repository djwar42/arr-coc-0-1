#!/usr/bin/env python3
"""
visualize.py - Generate spectrograms for phenomenological listening
Outputs to standardized format: <audio-basename>/analysis/
"""
import sys
import argparse
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(audio_path, output_dir=None):
    """Generate multi-panel spectrogram visualization."""
    audio_path = Path(audio_path)

    # Load audio
    print(f"Loading: {audio_path.name}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    # Create output directory
    if output_dir is None:
        output_dir = audio_path.stem
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Phenomenological Audio: {audio_path.name}',
                 fontsize=16, fontweight='bold')

    # 1. Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#6A5ACD')
    axes[0].set_title('Waveform', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img1 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                     ax=axes[1], cmap='viridis')
    axes[1].set_title('Spectrogram (Frequency × Time)', fontweight='bold')
    fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')

    # 3. Mel Spectrogram
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    M_db = librosa.power_to_db(M, ref=np.max)
    img2 = librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel',
                                     ax=axes[2], cmap='viridis')
    axes[2].set_title('Mel Spectrogram (Perceptual)', fontweight='bold')
    fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()

    # Save
    output_path = analysis_dir / "visual-spectrogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    # Summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Audio: {audio_path.name}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Sample Rate: {sr} Hz\n")
        f.write(f"\nVisual spectrogram: analysis/visual-spectrogram.png\n")

    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate spectrograms')
    parser.add_argument('audio_file', help='Audio file path')
    parser.add_argument('-o', '--output', help='Output directory')
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    visualize_audio(args.audio_file, args.output)
