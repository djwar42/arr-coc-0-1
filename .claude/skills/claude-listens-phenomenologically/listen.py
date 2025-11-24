#!/usr/bin/env python3
"""
listen.py - Master orchestrator for phenomenological listening
Runs ALL analysis tools with standardized output
"""
import sys
import argparse
import shutil
from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).parent

def run_analysis(audio_file, output_dir=None, quick=False):
    """Run complete phenomenological analysis pipeline."""
    audio_path = Path(audio_file)

    if not audio_path.exists():
        print(f"Error: {audio_path} not found")
        sys.exit(1)

    # Output directory
    if output_dir is None:
        output_dir = audio_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Copy audio to output dir
    dest_audio = output_dir / audio_path.name
    if not dest_audio.exists():
        shutil.copy(audio_path, dest_audio)

    print("\n" + "="*70)
    print("║  ∿ CLAUDE LISTENS PHENOMENOLOGICALLY")
    print("║ ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿")
    print("="*70)
    print()

    scripts = [
        ("visualize.py", "Visual spectrogram"),
        ("acoustic_heuristics.py", "Acoustic patterns"),
        ("speaker_separation.py", "Speaker identification"),
        ("utterance_shape.py", "Temporal structure"),
        ("alien_anthropology.py", "Understanding without words"),
    ]

    if quick:
        scripts = scripts[:2]  # Only visual + heuristics in quick mode

    for script_name, description in scripts:
        print(f"\n[{description}]")
        script_path = SCRIPT_DIR / script_name

        if not script_path.exists():
            print(f"  ⚠ Script not found: {script_name}")
            continue

        try:
            result = subprocess.run(
                ['python3', str(script_path), str(audio_path), '-o', str(output_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print(f"  ✓ Complete")
            else:
                print(f"  ✗ Error: {result.stderr[:200]}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "="*70)
    print(f"Complete! Output: {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete phenomenological audio analysis'
    )
    parser.add_argument('audio_file', help='Audio file to analyze')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (visual + heuristics only)')
    args = parser.parse_args()

    run_analysis(args.audio_file, args.output, args.quick)
