#!/usr/bin/env python3
"""
acoustic_heuristics.py - Extract acoustic patterns using pure heuristics
50 years of signal processing applied to spectrograms
"""
import sys
import argparse
from pathlib import Path
import librosa
import numpy as np

def analyze_heuristics(audio_path, output_dir=None):
    """Extract acoustic patterns from audio."""
    audio_path = Path(audio_path)

    # Load
    print(f"Analyzing: {audio_path.name}")
    y, sr = librosa.load(audio_path, sr=None)

    # Output dir
    if output_dir is None:
        output_dir = audio_path.stem
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Acoustic features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    mean_f0 = np.mean(f0_clean) if len(f0_clean) > 0 else 0

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(spectral_centroids)

    # Temporal features
    rms = librosa.feature.rms(y=y)[0]
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    n_onsets = len(onset_frames)

    # Syllable estimate (rough)
    hop_length = 512
    frames_per_second = sr / hop_length
    syllables_per_second = n_onsets / (len(y) / sr) if len(y) > 0 else 0

    # Write results
    output_path = analysis_dir / "acoustic-heuristics.txt"
    with open(output_path, 'w') as f:
        f.write("ACOUSTIC HEURISTICS ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Mean F0: {mean_f0:.1f} Hz\n")
        f.write(f"Spectral Centroid: {mean_centroid:.1f} Hz\n")
        f.write(f"Stressed syllables: {n_onsets}\n")
        f.write(f"Speaking rate: {syllables_per_second:.2f} syll/sec\n")
        f.write(f"\nDuration: {len(y)/sr:.2f}s\n")
        f.write(f"Sample rate: {sr} Hz\n")

    print(f"✓ Saved: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract acoustic patterns')
    parser.add_argument('audio_file')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    analyze_heuristics(args.audio_file, args.output)
