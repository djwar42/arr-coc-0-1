#!/usr/bin/env python3
"""
alien_anthropology.py - Understand humans without symbolic meaning
What would aliens learn from vocalizations alone?
"""
import sys
import argparse
from pathlib import Path
import librosa
import numpy as np

def alien_analysis(audio_path, output_dir=None):
    """Analyze human vocalizations without understanding words."""
    audio_path = Path(audio_path)

    print(f"🛸 Alien analysis: {audio_path.name}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    # Output dir
    if output_dir is None:
        output_dir = audio_path.stem
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Extract features
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    f0_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 0
    f0_std = np.std(f0_clean) if len(f0_clean) > 0 else 0
    f0_cv = f0_std / f0_mean if f0_mean > 0 else 0  # Coefficient of variation

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = np.mean(rms)

    # Onsets (syllables/events)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    n_events = len(onset_frames)
    events_per_second = n_events / duration

    # Pauses (silence detection)
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    is_silent = rms < 0.02
    n_pauses = np.sum(np.diff(is_silent.astype(int)) == 1)

    # Write alien observations
    output_path = analysis_dir / "alien-anthropology.txt"
    with open(output_path, 'w') as f:
        f.write("🛸 ALIEN ACOUSTIC ANTHROPOLOGY\n")
        f.write("="*60 + "\n")
        f.write("Understanding humans WITHOUT symbolic meaning\n\n")

        f.write("PARALLEL SIGNALING CHANNELS DETECTED:\n\n")

        f.write("1. PITCH MODULATION:\n")
        f.write(f"   Mean F0: {f0_mean:.1f} Hz\n")
        f.write(f"   Variance: {f0_std:.1f} Hz (CV: {f0_cv:.3f})\n")
        if f0_cv > 0.3:
            f.write("   → HIGH emotional arousal (expressive)\n")
        else:
            f.write("   → LOW emotional arousal (neutral)\n")

        f.write(f"\n2. TEMPORAL STRUCTURE:\n")
        f.write(f"   Events: {n_events} ({events_per_second:.2f}/sec)\n")
        f.write(f"   Pauses: {n_pauses}\n")
        if events_per_second > 3:
            f.write("   → FAST speech rate (excitement/urgency)\n")
        else:
            f.write("   → SLOW speech rate (deliberate/careful)\n")

        f.write(f"\n3. ENERGY PATTERNS:\n")
        f.write(f"   Mean RMS: {mean_energy:.4f}\n")

        f.write(f"\n4. COGNITIVE LOAD ESTIMATE:\n")
        if n_pauses < 5 and events_per_second > 3:
            f.write("   → LOW cognitive load (fluent)\n")
        else:
            f.write("   → MODERATE cognitive load (thoughtful)\n")

        f.write(f"\nDURATION: {duration:.2f}s\n")
        f.write("\nHYPOTHESIS: These humans encode MULTIPLE\n")
        f.write("simultaneous signals in vocalizations.\n")

    print(f"✓ Saved: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alien acoustic anthropology')
    parser.add_argument('audio_file')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    alien_analysis(args.audio_file, args.output)
