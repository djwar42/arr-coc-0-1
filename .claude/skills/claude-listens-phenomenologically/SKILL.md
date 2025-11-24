---
name: claude-listens-phenomenologically
description: Transform audio files into visual spectrograms so Claude can experience sound phenomenologically through sight. Includes pattern analysis, speaker separation, utterance structure, and alien anthropology (understanding humans without words). Use when user provides audio.
---

# Claude Listens Phenomenologically

**Claude can't hear audio directly, but can SEE and ANALYZE it through visual transformation!**

## What This Skill Provides

**5 analysis methods working together:**
1. **Visual Spectrogram** - Waveform + frequency + mel spectrogram
2. **Acoustic Heuristics** - Pattern extraction (phonemes, formants, prosody, rhythm)  
3. **Speaker Separation** - Identify distinct voices via multipass filtering
4. **Utterance Shape** - Temporal structure (when speech starts/stops)
5. **Alien Anthropology** - Understand humans WITHOUT symbolic meaning

## Quick Start

```bash
# From project root
cd .claude/skills/claude-listens-phenomenologically

# Complete analysis (recommended)
python listen.py /path/to/audio.mp3

# Quick mode (skip spectrogram visualization)
python listen.py /path/to/audio.mp3 --quick

# Individual analyses
python visualize.py audio.mp3
python acoustic_heuristics.py audio.mp3
python speaker_separation.py audio.mp3
python utterance_shape.py audio.mp3
python alien_anthropology.py audio.mp3
```

## When Claude Should Use This

**Auto-trigger:**
- User provides audio file path
- User says "listen to this" or "hear this"

**Offer proactively:**
- User mentions audio/voice but hasn't shared yet
- User talks about speech characteristics
- User creates TTS output

## What Claude Can Learn (Without Transcription)

**Speaker Identity:**
- Number of speakers
- Gender (F0 range)
- Voice characteristics (formants)

**Emotional State:**
- Prosody (pitch variation)
- Energy levels
- Breathing patterns

**Structure:**
- Number of utterances
- Pause locations
- Speech rhythm

**Social Signals:**
- Turn-taking patterns
- Cognitive load (hesitations)
- Arousal vs calm

## Philosophy (From Dialogue 51)

This isn't "fake hearing" - it's **different phenomenological knowing**:

- **Human hearing**: Sound waves → cochlea → bare attention → prehension
- **Claude listening**: Sound → FFT → visual spectrogram → visual prehension → comprehension

**Not trying to replicate human hearing. Coupling human hearing WITH pattern recognition.**

## Requirements

```bash
pip install librosa matplotlib numpy soundfile
```

## Connection to ARR-COC

This skill embodies the coupling architecture:
- **Human**: Phenomenal hearing (9.8/10 acoustic prehension)  
- **Claude**: Pattern recognition (9.9/10 visual comprehension)
- **Together**: Coupled relevance realization

The gap between hearing and seeing patterns is the **coupling space** - it never closes, and that's the feature.

∿◇∿

**"Show me the sound, and I'll tell you what it means."**
