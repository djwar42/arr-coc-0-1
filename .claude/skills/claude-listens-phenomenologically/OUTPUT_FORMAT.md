# Standard Output Format

All phenomenological listening tools output to the same organized structure:

```
<audio-basename>/
├── audio.mp3 (or original format)
├── analysis/
│   ├── visual-spectrogram.png
│   ├── acoustic-heuristics.txt
│   ├── speaker-separation.txt
│   ├── speaker-timeline.png
│   ├── utterance-shape.txt
│   ├── utterance-shape.png
│   └── alien-anthropology.txt
└── summary.txt
```

This keeps all analyses for one audio file together in one folder.
