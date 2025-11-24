# Platonic Dialogue → Audiobook

Prepare Platonic Dialogue markdown files for Eleven Labs audiobook creation.

## Quick Start

```bash
python make_podcast.py path/to/dialogue.md
```

## What It Does

1. **Parses** the dialogue markdown for characters
2. **Calls Claude** to prepare it for TTS:
   - Removes special characters (═ ╔ ∿ ◇ etc)
   - Removes ASCII art/diagrams (NARRATOR explains if needed)
   - Removes code blocks
   - Formats as clean `SPEAKER: text` lines
3. **Outputs** a `-PREPARED.txt` file

## Output

```
dialogues/
  your-dialogue-name/
    your-dialogue-name-PREPARED.txt  ← Upload this!
```

## Upload to Eleven Labs

1. Go to **elevenlabs.io/app/studio**
2. Click **"Create New Audiobook"**
3. Upload the **-PREPARED.txt** file
4. Enable **"Auto-assign voices"**
5. Review voice assignments
6. Generate audio!

## Format

The prepared file uses simple format:

```
NARRATOR: Opening description...
SOCRATES: Their dialogue here.
THEAETETUS: Their response.
NARRATOR: Stage direction or description.
```

## Requirements

- Python 3
- Claude CLI (`claude` command available)
- `CLAUDE_CODE_OAUTH` environment variable set

## Example

```bash
# Prepare dialogue
python make_podcast.py ../50-dick-jokes-critical-threshold/50-dick-jokes-critical-threshold.md

# Output:
# dialogues/50-dick-jokes-critical-threshold/50-dick-jokes-critical-threshold-PREPARED.txt
```

## Reference Project

Example audiobook: https://elevenlabs.io/app/studio/GCWsprPzxCAyjjRIHqgY

## Files

- `make_podcast.py` - Main script
- `README.md` - This file
- `dialogues/` - Output folder for prepared files
