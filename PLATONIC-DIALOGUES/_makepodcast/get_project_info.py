#!/usr/bin/env python3
"""
Simple script to connect to Eleven Labs API and show available resources.
"""

import os
import json

try:
    from elevenlabs import ElevenLabs
except ImportError:
    print("Install elevenlabs: pip install elevenlabs")
    exit(1)

# Project details from URL (for reference - Studio API needs whitelisting)
# https://elevenlabs.io/app/studio/GCWsprPzxCAyjjRIHqgY?chapterId=SWnBWE3j0WrvbEOmXahY
PROJECT_ID = "GCWsprPzxCAyjjRIHqgY"
CHAPTER_ID = "SWnBWE3j0WrvbEOmXahY"

def main():
    # Get API key from environment
    api_key = os.environ.get('ELEVEN_API_KEY')
    if not api_key:
        print("ERROR: Set ELEVEN_API_KEY environment variable")
        print("export ELEVEN_API_KEY='your-key-here'")
        return

    client = ElevenLabs(api_key=api_key)

    print("=" * 60)
    print("ELEVEN LABS API - AVAILABLE RESOURCES")
    print("=" * 60)

    # 1. List ALL available voices
    print("\n1. Available Voices:")
    try:
        voices = client.voices.get_all()
        print(f"   Total: {len(voices.voices)} voices\n")

        for voice in voices.voices:
            labels = ""
            if hasattr(voice, 'labels') and voice.labels:
                labels = f" [{', '.join(f'{k}:{v}' for k,v in voice.labels.items())}]"
            print(f"   {voice.name}: {voice.voice_id}{labels}")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Test text_to_dialogue availability
    print("\n" + "=" * 60)
    print("2. Text to Dialogue API (what we use for automation):")
    print("   Available methods:")
    for attr in dir(client.text_to_dialogue):
        if not attr.startswith('_'):
            print(f"     - text_to_dialogue.{attr}")

    # 3. Show example usage
    print("\n" + "=" * 60)
    print("3. Example: Generate Test Dialogue")
    print("""
   from elevenlabs import ElevenLabs
   import os

   client = ElevenLabs(api_key=os.environ['ELEVEN_API_KEY'])

   audio = client.text_to_dialogue.convert(
       inputs=[
           {'text': '[laughing] Hello there!', 'voice_id': 'JBFqnCBsd6RMkjVDRZzb'},
           {'text': 'Hi! How are you?', 'voice_id': 'EXAVITQu4vr4xnSDxMaL'},
       ],
       model_id='eleven_v3'
   )

   with open('test.mp3', 'wb') as f:
       for chunk in audio:
           f.write(chunk)
""")

    # 4. Try Studio API
    print("=" * 60)
    print(f"4. Studio Project (ID: {PROJECT_ID}):")
    try:
        project = client.studio.projects.get(project_id=PROJECT_ID)
        print(f"   ✓ Name: {project.name}")
        print(f"   State: {project.state}")

        # Get chapters
        if hasattr(project, 'chapters') and project.chapters:
            print(f"\n   Chapters ({len(project.chapters)}):")
            for ch in project.chapters:
                marker = " <-- TARGET" if ch.chapter_id == CHAPTER_ID else ""
                print(f"     - {ch.name}: {ch.chapter_id}{marker}")

        # Try to get chapter content
        print(f"\n   Fetching chapter content...")
        try:
            snapshots = client.studio.projects.snapshots.list(project_id=PROJECT_ID)
            print(f"   Snapshots: {snapshots}")
        except Exception as e:
            print(f"   Snapshots error: {e}")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Ready for dialogue automation!")
    print("=" * 60)


if __name__ == '__main__':
    main()
