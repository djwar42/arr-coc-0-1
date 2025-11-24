#!/usr/bin/env python3
"""
Attempt to add a voice to an existing Eleven Labs Studio project.
"""

import os
import sys

try:
    from elevenlabs import ElevenLabs
except ImportError:
    print("Install: pip install elevenlabs")
    sys.exit(1)

# Project from URL
PROJECT_ID = "qnoq22Dol9ics6Ob37qs"
CHAPTER_ID = "Q2z4BGLFz4gc6fYgOPGx"

def main():
    api_key = os.environ.get('ELEVEN_API_KEY')
    if not api_key:
        print("ERROR: Set ELEVEN_API_KEY")
        return

    client = ElevenLabs(api_key=api_key)

    print("=" * 60)
    print("ATTEMPTING TO ADD VOICE TO PROJECT")
    print("=" * 60)

    # 1. Get project info
    print(f"\n1. Getting project: {PROJECT_ID}")
    try:
        project = client.studio.get_project(project_id=PROJECT_ID)
        print(f"   Name: {project.name}")
        print(f"   State: {project.state}")

        # Check current voices
        if hasattr(project, 'default_voice_id'):
            print(f"   Default voice: {project.default_voice_id}")

    except Exception as e:
        print(f"   Error: {e}")
        return

    # 2. List available voices
    print("\n2. Your available voices:")
    try:
        voices = client.voices.get_all()
        for i, voice in enumerate(voices.voices[:20]):
            print(f"   {i+1}. {voice.name}: {voice.voice_id}")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Try to update project (this might not work)
    print("\n3. Checking API methods for adding voice...")

    # Check what methods are available on studio
    studio_methods = [m for m in dir(client.studio) if not m.startswith('_')]
    print(f"   Studio API methods: {studio_methods}")

    # Look for update/edit methods
    update_methods = [m for m in studio_methods if 'update' in m.lower() or 'edit' in m.lower() or 'add' in m.lower()]
    if update_methods:
        print(f"   Potential update methods: {update_methods}")
    else:
        print("   No update methods found - project modification may not be supported")

    print("\n" + "=" * 60)
    print("RESULT: Check above for available methods")
    print("=" * 60)
    print("\nIf no update methods exist, you MUST recreate the project.")


if __name__ == '__main__':
    main()
