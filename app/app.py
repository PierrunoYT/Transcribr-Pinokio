"""
Transcribr — Bulk Audio & Video Transcriber
===========================================
A Gradio app that downloads the audio from one or many YouTube URLs
(including playlists) and transcribes them with faster-whisper.

Usage:
    python app.py [--host 127.0.0.1] [--port 7860] [--share]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime

import gradio as gr
import imageio_ffmpeg
import yt_dlp
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "json"]

# faster-whisper bundles no ffmpeg, and yt-dlp wants one for some streams, so
# point both at the binary shipped by imageio-ffmpeg for a portable install.
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
os.environ.setdefault("PATH", "")
os.environ["PATH"] = os.path.dirname(FFMPEG_EXE) + os.pathsep + os.environ["PATH"]

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def _pick_device(requested: str):
    """Resolve the (device, compute_type) pair for CTranslate2."""
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "float16"
    # auto
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_model(size: str, device_choice: str) -> WhisperModel:
    device, compute_type = _pick_device(device_choice)
    key = (size, device, compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_timestamp(seconds: float, comma: bool = True) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(
            f"{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}"
        )
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{_format_timestamp(seg['start'], comma=False)} --> "
            f"{_format_timestamp(seg['end'], comma=False)}"
        )
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _render(segments, info, title, fmt: str) -> str:
    if fmt == "srt":
        return _to_srt(segments)
    if fmt == "vtt":
        return _to_vtt(segments)
    if fmt == "json":
        return json.dumps(
            {
                "title": title,
                "language": info.language,
                "duration": info.duration,
                "segments": segments,
            },
            indent=2,
            ensure_ascii=False,
        )
    return "\n".join(seg["text"].strip() for seg in segments).strip()


def _safe_name(name: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    cleaned = "".join(c if c in keep else "_" for c in name).strip()
    return (cleaned or "transcript")[:120]


def _expand_urls(raw: str):
    """Turn the textbox into individual entries, expanding playlists."""
    urls = [u.strip() for u in raw.splitlines() if u.strip()]
    entries = []
    flat_opts = {"quiet": True, "skip_download": True, "extract_flat": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(flat_opts) as ydl:
        for url in urls:
            try:
                meta = ydl.extract_info(url, download=False)
            except Exception:
                entries.append(url)
                continue
            if meta and meta.get("entries"):
                for item in meta["entries"]:
                    if not item:
                        continue
                    vid = item.get("url") or item.get("id")
                    if vid and not str(vid).startswith("http"):
                        vid = f"https://www.youtube.com/watch?v={vid}"
                    entries.append(vid or url)
            else:
                entries.append(url)
    return entries


def _download_audio(url: str, dest_dir: str):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ffmpeg_location": FFMPEG_EXE,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    title = info.get("title") or info.get("id") or "audio"
    return path, title


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------

def transcribe_bulk(urls_text, uploads, model_size, device_choice, language, fmt, progress=gr.Progress()):
    language = (language or "").strip() or None

    progress(0, desc="Collecting inputs...")
    # Build a unified job list: YouTube URLs (need downloading) + uploaded files.
    jobs = []  # each: (kind, ref) where kind in {"url", "file"}
    if urls_text and urls_text.strip():
        jobs.extend(("url", u) for u in _expand_urls(urls_text))
    for up in uploads or []:
        path = up if isinstance(up, str) else getattr(up, "name", None)
        if path and os.path.exists(path):
            jobs.append(("file", path))

    if not jobs:
        return "Add at least one YouTube URL or upload an audio/video file.", [], None, ""

    progress(0.05, desc=f"Loading {model_size} model...")
    try:
        model = get_model(model_size, device_choice)
    except Exception as exc:  # noqa: BLE001
        return (
            f"Could not load the {model_size} model on device '{device_choice}': "
            f"{str(exc)[:200]}",
            [],
            None,
            "",
        )

    run_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    log_rows = []
    files = []
    last_content = ""
    last_title = ""
    last_lang = ""
    total = len(jobs)
    tmp_dir = tempfile.mkdtemp(prefix="bt_audio_")
    try:
        for idx, (kind, ref) in enumerate(jobs):
            frac = 0.05 + 0.9 * (idx / total)
            if kind == "url":
                progress(frac, desc=f"[{idx + 1}/{total}] Downloading...")
                try:
                    audio_path, title = _download_audio(ref, tmp_dir)
                    cleanup = True
                except Exception as exc:  # noqa: BLE001
                    log_rows.append([ref, "download failed", str(exc)[:200]])
                    continue
            else:
                audio_path = ref
                title = os.path.splitext(os.path.basename(ref))[0]
                cleanup = False  # keep the user's uploaded file

            progress(frac, desc=f"[{idx + 1}/{total}] Transcribing {title[:40]}...")
            try:
                seg_iter, info = model.transcribe(audio_path, language=language)
                segments = [
                    {"start": s.start, "end": s.end, "text": s.text} for s in seg_iter
                ]
            except Exception as exc:  # noqa: BLE001
                log_rows.append([title, "transcription failed", str(exc)[:200]])
                continue
            finally:
                if cleanup:
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass

            content = _render(segments, info, title, fmt)
            out_path = os.path.join(run_dir, f"{_safe_name(title)}.{fmt}")
            n_dup = 1
            while os.path.exists(out_path):
                n_dup += 1
                out_path = os.path.join(run_dir, f"{_safe_name(title)}_{n_dup}.{fmt}")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            files.append(out_path)
            last_content = content
            last_title = title
            last_lang = info.language
            log_rows.append([title, f"ok ({info.language})", f"{len(segments)} segments"])

        progress(0.97, desc="Packaging results...")
        zip_path = None
        single_file = None
        transcript_text = ""
        if len(files) == 1:
            single_file = files[0]
            transcript_text = last_content
        elif len(files) > 1:
            zip_path = shutil.make_archive(run_dir, "zip", run_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    n = len(files)
    if n == 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        progress(1.0, desc="Complete")
        return (
            f"No transcripts produced (0/{total}). See the results table for errors.",
            log_rows,
            None,
            "",
        )
    if n == 1:
        summary = (
            f"Done. 1/{total} transcribed ({last_title[:60]} — {last_lang}). "
            f"Text shown below, download available."
        )
        download_out = single_file
    else:
        summary = f"Done. {n}/{total} transcribed. Saved to {run_dir}"
        download_out = zip_path
    progress(1.0, desc="Complete")
    return summary, log_rows, download_out, transcript_text


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(title="Transcribr", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🎬 Transcribr\n"
            "### Bulk Audio & Video Transcriber\n"
            "Paste YouTube video or playlist URLs (one per line) **and/or** upload "
            "your own audio/video files, then transcribe them all with faster-whisper."
        )
        with gr.Row():
            with gr.Column(scale=2):
                urls = gr.Textbox(
                    label="YouTube URLs",
                    lines=8,
                    placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...",
                )
                uploads = gr.File(
                    label="Or upload audio/video files",
                    file_count="multiple",
                    file_types=["audio", "video"],
                )
            with gr.Column(scale=1):
                model_size = gr.Dropdown(
                    MODEL_SIZES, value="small", label="Model size"
                )
                device_choice = gr.Dropdown(
                    ["auto", "cpu", "cuda"], value="auto", label="Device"
                )
                language = gr.Textbox(
                    label="Language (blank = auto-detect)", placeholder="en"
                )
                fmt = gr.Dropdown(OUTPUT_FORMATS, value="txt", label="Output format")
                run_btn = gr.Button("Transcribe", variant="primary")

        status = gr.Markdown("")
        results = gr.Dataframe(
            headers=["Title / URL", "Status", "Detail"],
            label="Results",
            wrap=True,
        )
        download = gr.File(label="Download transcript(s)")
        transcript_box = gr.Textbox(
            label="Transcript",
            lines=20,
            max_lines=50,
            show_copy_button=True,
            interactive=False,
        )

        run_btn.click(
            transcribe_bulk,
            inputs=[urls, uploads, model_size, device_choice, language, fmt],
            outputs=[status, results, download, transcript_box],
        )
    return app


def main():
    parser = argparse.ArgumentParser(description="Transcribr — Bulk Audio & Video Transcriber")
    parser.add_argument("--host", type=str, default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860"))),
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Transcribr — Bulk Audio & Video Transcriber")
    print(f"ffmpeg: {FFMPEG_EXE}")
    print("=" * 70)

    app = build_interface()
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
