# Transcribr

> Bulk Audio & Video Transcriber

A 1-click Pinokio launcher that transcribes audio in bulk using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — from one or many
YouTube videos (or entire playlists) **and/or** your own uploaded audio/video
files.

## What it does

- Paste any number of YouTube video **or** playlist URLs (one per line).
- **Or upload your own audio/video files** to transcribe them directly.
- Downloads the best audio with `yt-dlp` (bundled `ffmpeg`, no system install).
- Transcribes each item with a Whisper model (`tiny` → `large-v3`).
- Runs on **GPU** (NVIDIA, automatic) or **CPU**.
- Exports each transcript as **txt**, **srt**, **vtt**, or **json** — a single
  transcript is shown inline and downloadable as-is, several are bundled into a
  `.zip`.

## How to use

1. Click **Install** (installs dependencies; Whisper models download on first run).
2. Click **Start**, then open the Web UI.
3. Paste YouTube URLs and/or upload audio/video files, then choose a model size, device, language, and output format.
4. Click **Transcribe**. Results appear in the table; one transcript is shown
   in the text box, several arrive as a `.zip` download.

Transcripts are also saved under `app/transcripts/run_<timestamp>/`.

## Options

| Option        | Description                                                        |
|---------------|--------------------------------------------------------------------|
| Model size    | `tiny`, `base`, `small`, `medium`, `large-v3` (bigger = better/slower) |
| Device        | `auto` (GPU if available), `cpu`, `cuda`                            |
| Language      | Blank = auto-detect, or an ISO code such as `en`, `fr`, `es`       |
| Output format | `txt`, `srt`, `vtt`, `json`                                        |

## Programmatic access

The transcription engine is plain `faster-whisper` + `yt-dlp`, so you can script
the same pipeline directly.

### Python

```python
import yt_dlp
from faster_whisper import WhisperModel

url = "https://www.youtube.com/watch?v=VIDEO_ID"
opts = {"format": "bestaudio/best", "outtmpl": "audio.%(ext)s", "noplaylist": True}
with yt_dlp.YoutubeDL(opts) as ydl:
    info = ydl.extract_info(url, download=True)
    # prepare_filename() reports the pre-processing name; the finished file is
    # recorded under requested_downloads.
    audio = info["requested_downloads"][0]["filepath"]

model = WhisperModel("small", device="auto", compute_type="int8")
segments, info = model.transcribe(audio)
print("".join(seg.text for seg in segments))
```

### JavaScript (Node)

There is no JavaScript transcription engine here — shell out to the same Python
tools that this app installs into its `env` virtualenv.

```javascript
import { spawnSync } from "node:child_process";

// Download audio
spawnSync("yt-dlp", ["-f", "bestaudio/best", "-o", "audio.%(ext)s",
  "https://www.youtube.com/watch?v=VIDEO_ID"], { stdio: "inherit" });

// Transcribe with faster-whisper
spawnSync("python", ["-c", `
from faster_whisper import WhisperModel
segments, info = WhisperModel("small", compute_type="int8").transcribe("audio.webm")
print("".join(s.text for s in segments))
`], { stdio: "inherit" });
```

### Curl

`yt-dlp` and `faster-whisper` are local tools, not an HTTP API. To fetch a
single video's audio without this app:

```bash
yt-dlp -f bestaudio/best -o "audio.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Requirements

- ~1–3 GB disk for Whisper models (downloaded on demand).
- NVIDIA GPU optional; CPU works for smaller models.
