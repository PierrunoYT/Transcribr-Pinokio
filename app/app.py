"""
Transcribr — Bulk Audio & Video Transcriber
===========================================
A Gradio app that downloads the audio from one or many YouTube URLs
(including playlists) and transcribes them with faster-whisper.

Usage:
    python app.py [--host 127.0.0.1] [--port 7860] [--share]
"""

import argparse
import gc
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import datetime


def _register_nvidia_libs():
    """Make the pip-installed cuBLAS / cuDNN libraries visible to CTranslate2.

    ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12`` unpack their libraries into
    ``site-packages/nvidia/*/{bin,lib}``, which is not on any loader search path.
    Must run before CTranslate2 loads CUDA.
    """
    try:
        import nvidia  # namespace package provided by the nvidia-* wheels
    except ImportError:
        return
    roots = list(getattr(nvidia, "__path__", []))

    if sys.platform == "win32":
        for root in roots:
            for lib_dir in glob.glob(os.path.join(root, "*", "bin")):
                os.add_dll_directory(lib_dir)
                os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
    elif sys.platform.startswith("linux"):
        # The dynamic loader reads LD_LIBRARY_PATH only at process start, so
        # re-exec once with the library directories added.
        lib_dirs = [d for root in roots for d in glob.glob(os.path.join(root, "*", "lib"))]
        current = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        missing = [d for d in lib_dirs if d not in current]
        if missing and not os.environ.get("TRANSCRIBR_LIBS_SET"):
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(missing + [p for p in current if p])
            os.environ["TRANSCRIBR_LIBS_SET"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


_register_nvidia_libs()

import gradio as gr  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import yt_dlp  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from yt_dlp.extractor import gen_extractor_classes  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "json"]

try:
    from faster_whisper.tokenizer import _LANGUAGE_CODES  # noqa: E402
except ImportError:  # private name; skip validation if it ever moves
    _LANGUAGE_CODES = ()

# yt-dlp needs ffmpeg to extract/remux some audio streams; point it at the
# binary shipped by imageio-ffmpeg for a portable install. (faster-whisper
# decodes with PyAV and does not use it.)
try:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # noqa: BLE001 - fall back to whatever is on PATH
    FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"

_FFMPEG_DIR = os.path.dirname(FFMPEG_EXE)
if _FFMPEG_DIR:
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

# faster-whisper models are expensive (GPU/CPU memory), so keep exactly one
# loaded. The Gradio event runs with concurrency_limit=1, so a model is never
# used by two jobs at once.
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()

# CTranslate2 (4.x) loads cuBLAS 12 and cuDNN 9 lazily; if they are missing it
# aborts the whole process on the first CUDA transcription instead of raising.
_CUDA_LIBS = {
    "win32": ("cublas64_12.dll", "cudnn_ops64_9.dll"),
    "linux": ("libcublas.so.12", "libcudnn_ops.so.9"),
}


def _cuda_compute_type() -> str:
    """float16 where the GPU supports it; older cards (e.g. GTX 10xx) reject it."""
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return "float16"
    for candidate in ("float16", "int8_float32", "float32"):
        if candidate in supported:
            return candidate
    return "float32"


def _cuda_libs_missing() -> list:
    """Names of the CUDA runtime libraries CTranslate2 needs that can't be loaded."""
    import ctypes

    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    missing = []
    for name in _CUDA_LIBS.get(platform, ()):
        try:
            ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def _pick_device(requested: str):
    """Resolve the (device, compute_type) pair for CTranslate2."""
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        missing = _cuda_libs_missing()
        if missing:
            raise RuntimeError(
                "CUDA libraries not found: " + ", ".join(missing)
                + ". Reinstall Transcribr on an NVIDIA machine, or pick device 'cpu'."
            )
        return "cuda", _cuda_compute_type()
    # auto: only use the GPU when CTranslate2 can actually run on it.
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0 and not _cuda_libs_missing():
            return "cuda", _cuda_compute_type()
    except Exception:
        pass
    return "cpu", "int8"


def get_model(size: str, device_choice: str) -> WhisperModel:
    device, compute_type = _pick_device(device_choice)
    key = (size, device, compute_type)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            # Drop any previously loaded model *before* loading the new one so
            # that switching sizes/devices never holds two sets of weights at
            # once (which can exhaust GPU memory).
            _MODEL_CACHE.clear()
            gc.collect()
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
                "language": info.language or "unknown",
                "duration": info.duration,
                "segments": segments,
            },
            indent=2,
            ensure_ascii=False,
        )
    return "\n".join(seg["text"].strip() for seg in segments).strip()


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_name(name: str) -> str:
    # Keep letters/digits from any script (so non-Latin titles stay readable);
    # replace everything else outside a small safe punctuation set.
    cleaned = "".join(c if c.isalnum() or c in "-_.() " else "_" for c in name).strip()
    # Windows rejects names ending in a dot or a space.
    cleaned = cleaned[:120].rstrip(". ")
    # Windows also reserves CON, CON.txt, CON.backup.txt, ...
    if not cleaned or cleaned.split(".")[0].rstrip().upper() in _WINDOWS_RESERVED:
        return "transcript"
    return cleaned


# URLs these extractors match are always one video, so expanding them would
# only cost an extra network round-trip.
_SINGLE_VIDEO_IES = {"Youtube"}
# Flat playlist entries from these extractors are themselves playlists
# (channel tabs, sub-playlists, albums, ...) and need expanding again.
_PLAYLIST_IE_RE = re.compile(r"(Tab|Playlist|Channel|User|Album|Set|Collection)$")
_MAX_PLAYLIST_DEPTH = 3

_YDL_BASE_OPTS = {
    "quiet": True,
    "noprogress": True,
    "no_warnings": True,
    "noplaylist": True,
    # Pinokio runs us in a terminal; keep ANSI colour codes out of error text.
    "color": "no_color",
}


def _extractor_key(url: str) -> str:
    """Key of the extractor yt-dlp would pick for ``url`` (regex only, no network)."""
    for ie in gen_extractor_classes():
        if ie.suitable(url):
            return ie.ie_key()
    return ""


def _collect_entries(ydl, url: str, entries: list, depth: int = 0):
    if _extractor_key(url) in _SINGLE_VIDEO_IES:
        entries.append(url)
        return
    try:
        meta = ydl.extract_info(url, download=False)
    except Exception:
        entries.append(url)  # surface it as a failed row later
        return
    if not meta or meta.get("_type") != "playlist":
        entries.append(url)
        return
    for item in meta.get("entries") or []:
        if not item:
            continue
        extractor = item.get("ie_key") or meta.get("extractor_key") or ""
        vid = item.get("url") or item.get("webpage_url") or item.get("id")
        if vid and not str(vid).startswith("http"):
            # A bare id is only resolvable for YouTube entries.
            vid = (
                f"https://www.youtube.com/watch?v={vid}"
                if extractor.startswith("Youtube")
                else None
            )
        if not vid:
            continue
        if item.get("_type") == "playlist" or _PLAYLIST_IE_RE.search(extractor):
            if depth < _MAX_PLAYLIST_DEPTH:
                _collect_entries(ydl, vid, entries, depth + 1)
        else:
            entries.append(vid)


def _expand_urls(raw: str):
    """Turn the textbox into individual entries, expanding (nested) playlists.

    A video URL that merely carries a ``list=`` parameter (e.g. opened from a
    playlist or Mix) stays a single video.
    """
    urls = [u.strip() for u in raw.splitlines() if u.strip()]
    entries = []
    flat_opts = {**_YDL_BASE_OPTS, "skip_download": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(flat_opts) as ydl:
        for url in urls:
            before = len(entries)
            _collect_entries(ydl, url, entries)
            if len(entries) == before:
                entries.append(url)  # surface it as a failed row, not silently drop it
    return entries


def _download_audio(url: str, dest_dir: str):
    opts = {
        **_YDL_BASE_OPTS,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "ffmpeg_location": FFMPEG_EXE,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        # Resolve first so an unexpanded playlist fails fast instead of
        # downloading every entry into the temp folder.
        info = ydl.extract_info(url, download=False)
        if info.get("_type") == "playlist" or "entries" in info:
            raise ValueError("this URL is a playlist that could not be expanded into videos")
        info = ydl.process_ie_result(info, download=True)
        path = _downloaded_path(ydl, info, dest_dir)
    title = info.get("title") or info.get("id") or "audio"
    return path, title


def _downloaded_path(ydl, info, dest_dir: str) -> str:
    """Resolve the file yt-dlp actually wrote.

    ``prepare_filename`` returns the *pre*-processing name, which is wrong as
    soon as yt-dlp merges or remuxes (e.g. ``.webm`` on disk vs ``.mkv``
    reported), so prefer the path yt-dlp records for the finished download.
    """
    for entry in info.get("requested_downloads") or []:
        actual = entry.get("filepath") or entry.get("_filename")
        if actual and os.path.exists(actual):
            return actual

    guess = ydl.prepare_filename(info)
    if os.path.exists(guess):
        return guess

    matches = glob.glob(os.path.splitext(guess)[0] + ".*")
    if matches:
        return matches[0]

    raise FileNotFoundError(f"yt-dlp reported no output file in {dest_dir}")


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------

def transcribe_bulk(urls_text, uploads, model_size, device_choice, language, fmt, progress=gr.Progress()):
    language = (language or "").strip().lower() or None
    # Reject a bad code up front instead of failing every item after downloading it.
    if language and _LANGUAGE_CODES and language not in _LANGUAGE_CODES:
        return (
            f"Unknown language code '{language}'. Use an ISO code such as en, fr, "
            "es, or leave it blank to auto-detect.",
            [["-", "invalid language", language]],
            None,
            "",
        )

    progress(0, desc="Collecting inputs...")
    # Build a unified job list: YouTube URLs (need downloading) + uploaded files.
    jobs = []  # each: (kind, ref) where kind in {"url", "file"}
    seen = set()

    def _add(kind, ref):
        if ref and (kind, ref) not in seen:
            seen.add((kind, ref))
            jobs.append((kind, ref))

    if urls_text and urls_text.strip():
        for u in _expand_urls(urls_text):
            _add("url", u)
    for up in uploads or []:
        path = up if isinstance(up, str) else getattr(up, "name", None)
        if path and os.path.exists(path):
            _add("file", path)

    if not jobs:
        return (
            "Add at least one YouTube URL or upload an audio/video file.",
            [["-", "no input", "Nothing to transcribe"]],
            None,
            "",
        )

    progress(0.05, desc=f"Loading {model_size} model...")
    try:
        model = get_model(model_size, device_choice)
    except Exception as exc:  # noqa: BLE001
        return (
            f"Could not load the {model_size} model on device '{device_choice}': "
            f"{str(exc)[:200]}",
            [["-", "model load failed", str(exc)[:200]]],
            None,
            "",
        )

    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, stamp)
    suffix = 1
    while os.path.exists(run_dir) or os.path.exists(run_dir + ".zip"):
        suffix += 1
        run_dir = os.path.join(OUTPUT_DIR, f"{stamp}_{suffix}")
    os.makedirs(run_dir)

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
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in seg_iter
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

            detected = info.language or "unknown"
            content = _render(segments, info, title, fmt)
            base = _safe_name(title)
            out_path = os.path.join(run_dir, f"{base}.{fmt}")
            n_dup = 1
            while os.path.exists(out_path):
                n_dup += 1
                out_path = os.path.join(run_dir, f"{base}_{n_dup}.{fmt}")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            files.append(out_path)
            last_content = content
            last_title = title
            last_lang = detected
            log_rows.append([title, f"ok ({detected})", f"{len(segments)} segments"])

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
            concurrency_limit=1,
            api_name="transcribe",
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
        # Gradio only serves files from the cwd or temp dir by default; allow the
        # transcripts folder explicitly so downloads work from any cwd.
        allowed_paths=[OUTPUT_DIR],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
