#!/usr/bin/env python3
"""
Make Podcast from Platonic Dialogue

Parses a dialogue markdown file, prepares it for Eleven Labs TTS,
and outputs a .txt file ready for manual upload.
"""

# <claudes_code_comments>
# ** Function List **
# get_dialogue_folder: gets/creates folder for output files
# parse_characters: extracts all characters with line counts
# claude_prepare_dialogue: uses Claude CLI to prepare dialogue for TTS
# main: CLI entry point
#
# ** Technical Review **
# Simplified script that prepares Platonic Dialogue markdown for Eleven Labs.
# Flow: parse_characters() → claude_prepare_dialogue() → save .txt → instructions
#
# The prepared .txt file is uploaded manually to Eleven Labs:
# 1. Go to Projects → Create New Audiobook
# 2. Upload the -PREPARED.txt file
# 3. Enable "Auto-assign voices"
# 4. Generate audio
#
# Claude preparation removes special characters, ASCII art, and formats
# as SPEAKER: text for easy TTS processing.
# </claudes_code_comments>

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List


def get_dialogue_folder(dialogue_file: str) -> Path:
    """Get/create the dialogue folder for output files."""
    script_dir = Path(__file__).parent
    dialogue_name = Path(dialogue_file).stem
    folder = script_dir / "dialogues" / dialogue_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def parse_characters(markdown_path: str) -> Dict[str, int]:
    """
    Parse markdown file and extract all characters with line counts.

    Simple format-based parsing: **NAME:** at start of line

    Returns dict of {character_name: line_count}
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    characters = {}

    for line in lines:
        # Match pattern: **NAME:** where colon is INSIDE the bold
        match = re.match(r'^\s*\*\*([^*:]+).*?:\*\*', line)

        if not match:
            continue

        raw_name = match.group(1).strip()

        # Skip if not ALL CAPS (speakers are ALL CAPS)
        name_without_parens = re.sub(r'\s*\([^)]*\)\s*', '', raw_name).strip()
        if not name_without_parens.isupper():
            continue

        name = name_without_parens

        # Clean up multi-word names
        words = name.split()
        if len(words) > 2 and name.isupper():
            if words[0] == 'THE':
                name = ' '.join(words[:4]) if len(words) >= 4 else name
            elif 'ORACLE' in words:
                idx = words.index('ORACLE')
                name = ' '.join(words[:idx+1])
            else:
                name = ' '.join(words[:2])

        name = name.rstrip(',').strip()
        if ', ' in name:
            name = name.split(', ')[0]

        characters[name] = characters.get(name, 0) + 1

    return characters


def claude_prepare_dialogue(dialogue_text: str, output_path: Path) -> bool:
    """
    Use Claude CLI to prepare dialogue for TTS:
    - Remove special chars (only . , - % $ ! ? ' " allowed)
    - Remove ASCII art, explain via Narrator
    - Create parseable format: SPEAKER: text
    """

    prompt = f"""Transform this dialogue for text-to-speech. Output ONLY the transformed dialogue.

RULES:
1. FORMAT: Each line must be exactly: SPEAKER: dialogue text
2. SPECIAL CHARACTERS: Only allow . , - % $ ! ? ' "
   - Remove or replace all others (no special box chars, no unicode symbols)
   - Replace --- or === with just a pause
3. ASCII ART/DIAGRAMS: Remove entirely
   - Have NARRATOR briefly describe what was shown if needed
4. CODE BLOCKS: Remove or have NARRATOR summarize briefly
5. Keep dialogue natural and flowing
6. Add NARRATOR for transitions, descriptions, stage directions
7. Remove any markdown formatting (**, *, #, etc)

DIALOGUE TO TRANSFORM:
{dialogue_text}

OUTPUT FORMAT (one speaker per line):
NARRATOR: Opening description...
SPEAKER_NAME: Their dialogue here.
ANOTHER_SPEAKER: Their response.
NARRATOR: Stage direction or description."""

    try:
        escaped_prompt = prompt.replace("'", "'\\''")
        print("   Calling Claude to prepare dialogue...")
        result = subprocess.run(
            f"ANTHROPIC_AUTH_TOKEN='' ANTHROPIC_BASE_URL='' CLAUDE_CODE_OAUTH_TOKEN=\"$CLAUDE_CODE_OAUTH\" claude --dangerously-skip-permissions -p '{escaped_prompt}'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 min for long dialogues
        )

        if result.returncode != 0:
            print(f"   Claude CLI error: {result.stderr}")
            return False

        # Save prepared dialogue
        prepared_text = result.stdout.strip()
        with open(output_path, 'w') as f:
            f.write(prepared_text)

        # Count lines
        line_count = len([l for l in prepared_text.split('\n') if ':' in l and l.strip()])
        print(f"   Saved {line_count} lines to: {output_path}")
        return True

    except subprocess.TimeoutExpired:
        print("   ERROR: Claude CLI timed out")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Platonic Dialogue for Eleven Labs audiobook"
    )
    parser.add_argument(
        'dialogue_file',
        help="Path to the dialogue markdown file"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.dialogue_file):
        print(f"ERROR: File not found: {args.dialogue_file}")
        sys.exit(1)

    print("=" * 60)
    print("PLATONIC DIALOGUE → AUDIOBOOK PREP")
    print("=" * 60)

    # 1. Parse characters from dialogue
    print(f"\n1. Parsing: {args.dialogue_file}")
    characters = parse_characters(args.dialogue_file)

    if not characters:
        print("   ERROR: No characters found!")
        sys.exit(1)

    # Add NARRATOR (will be used in prepared dialogue)
    if 'NARRATOR' not in characters:
        characters['NARRATOR'] = 0

    print(f"   Found {len(characters)} characters:\n")
    for char, count in sorted(characters.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"   {char}: {count} lines")
        else:
            print(f"   {char}: (added for TTS)")

    # 2. Setup dialogue folder
    dialogue_folder = get_dialogue_folder(args.dialogue_file)
    print(f"\n2. Output folder: {dialogue_folder}")

    # 3. Prepare dialogue for TTS
    print(f"\n3. Preparing dialogue for TTS...")

    # Read original dialogue
    with open(args.dialogue_file, 'r', encoding='utf-8') as f:
        dialogue_text = f.read()

    # Output as .txt (not .md)
    prepared_path = dialogue_folder / f"{Path(args.dialogue_file).stem}-PREPARED.txt"

    # Call Claude to prepare dialogue
    success = claude_prepare_dialogue(dialogue_text, prepared_path)

    if not success:
        print("\nERROR: Failed to prepare dialogue")
        sys.exit(1)

    # 4. Print instructions
    print("\n" + "=" * 60)
    print("DONE! PREPARED FILE READY")
    print("=" * 60)
    print(f"\nOutput: {prepared_path}")
    print("\n" + "-" * 60)
    print("NEXT STEPS:")
    print("-" * 60)
    print("""
1. Go to elevenlabs.io/app/studio
2. Click "Create New Audiobook"
3. Upload the -PREPARED.txt file
4. Enable "Auto-assign voices"
5. Review voice assignments
6. Generate audio!
""")
    print("=" * 60)


if __name__ == '__main__':
    main()
