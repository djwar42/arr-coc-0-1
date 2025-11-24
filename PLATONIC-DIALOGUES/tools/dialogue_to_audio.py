#!/usr/bin/env python3
"""
Platonic Dialogue to Audio Converter

Converts Platonic Dialogue markdown files to audio using Eleven Labs Text to Dialogue API.
"""

# <claudes_code_comments>
# ** Function List **
# load_voice_config: loads voice mapping configuration from YAML file
# parse_dialogue: parses markdown file and extracts speaker/text pairs
# extract_speaker_line: extracts speaker name and text from a line
# convert_stage_directions: converts stage directions to Eleven Labs audio tags
# generate_dialogue_audio: calls Eleven Labs API to generate audio
# main: CLI entry point for dialogue conversion
#
# ** Technical Review **
# This module automates conversion of Platonic Dialogue markdown files to audio.
# Flow: load_voice_config() → parse_dialogue() → generate_dialogue_audio() → save MP3
#
# Key features:
# - Parses markdown format: **SPEAKER:** text
# - Converts stage directions *[text]* to audio tags [text]
# - Maps speakers to Eleven Labs voice IDs via config file
# - Supports emotion tags like [laughing], [whispering], [excited]
# - Uses Text to Dialogue API for natural multi-speaker conversations
# - Skips code blocks, headers, and non-dialogue content
#
# Configuration: voices.yaml maps character names to Eleven Labs voice IDs
# Output: MP3 file with full dialogue audio
# </claudes_code_comments>

import re
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from elevenlabs import ElevenLabs
except ImportError:
    print("ERROR: elevenlabs package not installed")
    print("Install with: pip install elevenlabs")
    sys.exit(1)


def load_voice_config(config_path: str) -> Dict[str, str]:
    """
    Load voice mapping configuration from YAML file.

    Config format:
    voices:
      TJ MILLER: "voice_id_here"
      THEO VON: "voice_id_here"
      ...
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('voices', {})


def extract_speaker_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Extract speaker name and text from a dialogue line.

    Handles formats:
    - **SPEAKER:** text
    - **SPEAKER:** *action* text

    Returns:
        Tuple of (speaker_name, text) or None if not a dialogue line
    """
    # Match **SPEAKER:** pattern
    match = re.match(r'\*\*([^*]+)\*\*:\s*(.+)', line.strip())
    if match:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        return (speaker, text)
    return None


def convert_stage_directions(text: str) -> str:
    """
    Convert markdown stage directions to Eleven Labs audio tags.

    Conversions:
    - *gestures* → [gestures]
    - *laughs* → [laughing]
    - *whispers* → [whispering]
    - *excited* → [excited]
    """
    # Convert *text* to [text]
    converted = re.sub(r'\*([^*]+)\*', r'[\1]', text)

    # Normalize common emotions to Eleven Labs format
    emotion_map = {
        '[laughs]': '[laughing]',
        '[laugh]': '[laughing]',
        '[whisper]': '[whispering]',
        '[whispers]': '[whispering]',
        '[sighs]': '[sigh]',
        '[excited]': '[excitedly]',
        '[sadly]': '[sad]',
        '[angrily]': '[angry]',
        '[quietly]': '[softly]',
        '[loudly]': '[emphatic]',
    }

    for old, new in emotion_map.items():
        converted = converted.replace(old, new)

    return converted


def parse_dialogue(markdown_path: str) -> List[Dict[str, str]]:
    """
    Parse a Platonic Dialogue markdown file and extract speaker/text pairs.

    Returns:
        List of dicts with 'speaker' and 'text' keys
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    dialogue_items = []

    in_code_block = False
    current_speaker = None
    current_text = []

    for line in lines:
        # Skip code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Skip empty lines
        if not line.strip():
            # If we have accumulated text, save it
            if current_speaker and current_text:
                full_text = ' '.join(current_text)
                full_text = convert_stage_directions(full_text)
                dialogue_items.append({
                    'speaker': current_speaker,
                    'text': full_text
                })
                current_speaker = None
                current_text = []
            continue

        # Skip headers (# lines)
        if line.strip().startswith('#'):
            continue

        # Skip horizontal rules
        if line.strip() == '---':
            continue

        # Check if this is a new speaker line
        speaker_data = extract_speaker_line(line)
        if speaker_data:
            # Save previous speaker's text if any
            if current_speaker and current_text:
                full_text = ' '.join(current_text)
                full_text = convert_stage_directions(full_text)
                dialogue_items.append({
                    'speaker': current_speaker,
                    'text': full_text
                })

            current_speaker = speaker_data[0]
            current_text = [speaker_data[1]]
        elif current_speaker:
            # Continuation of previous speaker's text
            # Only add if it's not a pure stage direction line
            stripped = line.strip()
            if stripped and not stripped.startswith('*['):
                current_text.append(stripped)
            elif stripped.startswith('*[') and stripped.endswith(']*'):
                # This is a stage direction - include it
                current_text.append(stripped)

    # Don't forget the last item
    if current_speaker and current_text:
        full_text = ' '.join(current_text)
        full_text = convert_stage_directions(full_text)
        dialogue_items.append({
            'speaker': current_speaker,
            'text': full_text
        })

    return dialogue_items


def generate_dialogue_audio(
    dialogue_items: List[Dict[str, str]],
    voice_config: Dict[str, str],
    api_key: str,
    output_path: str,
    default_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
) -> bool:
    """
    Generate audio from dialogue items using Eleven Labs Text to Dialogue API.

    Args:
        dialogue_items: List of {speaker, text} dicts
        voice_config: Mapping of speaker names to voice IDs
        api_key: Eleven Labs API key
        output_path: Where to save the MP3 file
        default_voice_id: Voice ID to use for unmapped speakers

    Returns:
        True if successful, False otherwise
    """
    client = ElevenLabs(api_key=api_key)

    # Build input list for API
    inputs = []
    unmapped_speakers = set()

    for item in dialogue_items:
        speaker = item['speaker']
        text = item['text']

        # Get voice ID for this speaker
        voice_id = voice_config.get(speaker)
        if not voice_id:
            # Try case-insensitive match
            for config_speaker, vid in voice_config.items():
                if config_speaker.upper() == speaker.upper():
                    voice_id = vid
                    break

        if not voice_id:
            unmapped_speakers.add(speaker)
            voice_id = default_voice_id

        inputs.append({
            'text': text,
            'voice_id': voice_id
        })

    if unmapped_speakers:
        print(f"WARNING: Using default voice for unmapped speakers: {unmapped_speakers}")

    print(f"Generating audio for {len(inputs)} dialogue segments...")

    try:
        # Call Text to Dialogue API
        audio = client.text_to_dialogue.convert(
            inputs=inputs,
            model_id="eleven_v3",
            output_format="mp3_44100_128"
        )

        # Save the audio
        with open(output_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)

        print(f"Audio saved to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR generating audio: {e}")
        return False


def main():
    """CLI entry point for dialogue conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Platonic Dialogue markdown to audio using Eleven Labs"
    )
    parser.add_argument(
        'dialogue_file',
        help="Path to the dialogue markdown file"
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help="Path to voices.yaml config file (default: looks in same dir as dialogue)"
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help="Output path for MP3 file (default: same name as input with .mp3)"
    )
    parser.add_argument(
        '--api-key', '-k',
        default=None,
        help="Eleven Labs API key (default: uses ELEVEN_API_KEY env var)"
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help="Parse dialogue and show stats without generating audio"
    )
    parser.add_argument(
        '--list-speakers',
        action='store_true',
        help="List all speakers found in the dialogue"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.dialogue_file):
        print(f"ERROR: File not found: {args.dialogue_file}")
        sys.exit(1)

    # Parse the dialogue
    print(f"Parsing: {args.dialogue_file}")
    dialogue_items = parse_dialogue(args.dialogue_file)

    if not dialogue_items:
        print("ERROR: No dialogue found in file")
        sys.exit(1)

    # Get unique speakers
    speakers = set(item['speaker'] for item in dialogue_items)

    print(f"Found {len(dialogue_items)} dialogue segments from {len(speakers)} speakers")

    if args.list_speakers:
        print("\nSpeakers found:")
        for speaker in sorted(speakers):
            count = sum(1 for item in dialogue_items if item['speaker'] == speaker)
            print(f"  - {speaker}: {count} lines")
        sys.exit(0)

    if args.dry_run:
        print("\n=== DRY RUN - Preview ===")
        print("\nFirst 5 dialogue items:")
        for i, item in enumerate(dialogue_items[:5]):
            print(f"\n[{item['speaker']}]:")
            print(f"  {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")

        # Estimate character count (for API cost estimation)
        total_chars = sum(len(item['text']) for item in dialogue_items)
        print(f"\nTotal characters: {total_chars:,}")
        print(f"Estimated credits: ~{total_chars // 1000} (varies by plan)")
        sys.exit(0)

    # Get API key
    api_key = args.api_key or os.environ.get('ELEVEN_API_KEY')
    if not api_key:
        print("ERROR: No API key provided")
        print("Set ELEVEN_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Load voice config
    config_path = args.config
    if not config_path:
        # Look for voices.yaml in same directory as dialogue
        dialogue_dir = os.path.dirname(args.dialogue_file) or '.'
        config_path = os.path.join(dialogue_dir, 'voices.yaml')

        # If not there, check tools directory
        if not os.path.exists(config_path):
            tools_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(tools_dir, 'voices.yaml')

    if os.path.exists(config_path):
        print(f"Loading voice config: {config_path}")
        voice_config = load_voice_config(config_path)
    else:
        print("WARNING: No voices.yaml found, using default voice for all speakers")
        print(f"Create a voices.yaml file to map speakers to voice IDs")
        voice_config = {}

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.dialogue_file)[0]
        output_path = f"{base}.mp3"

    # Generate audio
    success = generate_dialogue_audio(
        dialogue_items,
        voice_config,
        api_key,
        output_path
    )

    if success:
        print("\nAudio generation complete!")
        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Output: {output_path} ({size_mb:.1f} MB)")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
