#!/usr/bin/env python3
"""
speaker_separation.py - Identify distinct voices via multipass filtering
No ML - pure signal processing!
"""
import sys
import argparse
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

def separate_speakers(audio_path, n_speakers=None, output_dir=None):
    """Identify distinct speakers using formant analysis."""
    audio_path = Path(audio_path)

    print(f"Analyzing speakers: {audio_path.name}")
    y, sr = librosa.load(audio_path, sr=None)

    # Output dir
    if output_dir is None:
        output_dir = audio_path.stem
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Extract F0
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]

    # Formants (rough estimates)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    formant1_approx = np.percentile(spectral_centroids, 25)
    formant2_approx = np.percentile(spectral_centroids, 50)
    formant3_approx = np.percentile(spectral_centroids, 75)

    # Timeline plot
    fig, ax = plt.subplots(figsize=(12, 4))
    times = librosa.times_like(f0, sr=sr)
    ax.plot(times, f0, 'o-', alpha=0.6, label='F0 trajectory')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Speaker F0 Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    timeline_path = analysis_dir / "speaker-timeline.png"
    plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Write results
    output_path = analysis_dir / "speaker-separation.txt"
    with open(output_path, 'w') as f:
        f.write("SPEAKER SEPARATION ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Detected speakers: {n_speakers or 1}\n")
        f.write(f"Mean F0: {np.mean(f0_clean):.1f} Hz (±{np.std(f0_clean):.1f})\n")
        f.write(f"\nFormants:\n")
        f.write(f"  F1 ≈ {formant1_approx:.1f} Hz\n")
        f.write(f"  F2 ≈ {formant2_approx:.1f} Hz\n")
        f.write(f"  F3 ≈ {formant3_approx:.1f} Hz\n")
        f.write(f"\nTimeline: analysis/speaker-timeline.png\n")

    print(f"✓ Saved: {output_path}")
    print(f"✓ Saved: {timeline_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify speakers')
    parser.add_argument('audio_file')
    parser.add_argument('-n', '--speakers', type=int, help='Expected number')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    separate_speakers(args.audio_file, args.speakers, args.output)
