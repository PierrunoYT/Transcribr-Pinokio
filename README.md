# Transcribr

> Bulk Audio & Video Transcriber

A 1-click Pinokio launcher that transcribes audio in bulk using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — from one or many
YouTube videos (or entire playlists) **and/or** your own uploaded audio/video
files.

## What it does

- Paste any number of YouTube video **or** playlist URLs (one per line).
- **Or upload your own audio/video files** to transcribe them directly.
- Downloads the best audio with `yt-dlp` (bundled `ffmpeg` and Deno JS runtime,
  no system install). A video link that carries a `list=` parameter stays a
  single video; playlist and channel URLs expand to every video.
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

While the app is running it exposes its Transcribe action as a Gradio API
endpoint named `/transcribe`. Replace `http://127.0.0.1:PORT` with the URL
shown in Pinokio. Inputs, in order: URLs (newline-separated), uploaded files,
model size, device, language, output format. Outputs: status message, results
table, download file, transcript text.

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:PORT/")
status, results, download, text = client.predict(
    "https://www.youtube.com/watch?v=VIDEO_ID",  # URLs, one per line
    [],  # or [handle_file("talk.mp3")] to upload local files
    "small",  # model size
    "auto",  # device
    "",  # language ("" = auto-detect)
    "txt",  # output format
    api_name="/transcribe",
)
print(text)
```

### JavaScript

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:PORT/");
const result = await client.predict("/transcribe", [
  "https://www.youtube.com/watch?v=VIDEO_ID", [], "small", "auto", "", "txt",
]);
const [status, results, download, text] = result.data;
console.log(text);
```

### Curl

```bash
EVENT_ID=$(curl -s -X POST http://127.0.0.1:PORT/gradio_api/call/transcribe \
  -H "Content-Type: application/json" \
  -d '{"data": ["https://www.youtube.com/watch?v=VIDEO_ID", [], "small", "auto", "", "txt"]}' \
  | sed 's/.*"event_id": *"\([^"]*\)".*/\1/')
curl -N http://127.0.0.1:PORT/gradio_api/call/transcribe/$EVENT_ID
```

The final `event: complete` line carries the four outputs as JSON.

## Requirements

- ~1–3 GB disk for Whisper models (downloaded on demand).
- NVIDIA GPU optional; CPU works for smaller models.
