"""media-edit library CLI. The standard-library path has no pip dependencies."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys

from . import __version__
from .highlights import WEIGHTS
from .indexing import Indexer, discover
from .models import SemanticModel
from .review import export, make_server, preview
from .store import get_segment, highlights, model_key, remove, search, status


def parser():
    root = argparse.ArgumentParser(
        prog="media-edit library",
        description="Local video search and explainable highlights",
    )
    root.add_argument("--version", action="version", version=__version__)
    subs = root.add_subparsers(dest="command", required=True)

    def command(name, help_text):
        child = subs.add_parser(name, help=help_text)
        child.add_argument(
            "--db",
            default=os.environ.get("ME_INDEX_DB", ".media-index/library.sqlite3"),
            help="Local SQLite catalog",
        )
        return child

    def model_args(child):
        child.add_argument(
            "--semantic-model",
            help="Local SentenceTransformer model directory (no download)",
        )
        child.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")

    ingest = command(
        "index", "Recursively index videos and sidecars, resuming unchanged files"
    )
    ingest.add_argument("inputs", nargs="+")
    ingest.add_argument(
        "--asr",
        choices=("auto", "none", "whisper-cpp", "mlx", "parakeet", "faster-whisper"),
        default="auto",
    )
    ingest.add_argument("--model-path", help="Existing local ASR model file/directory")
    ingest.add_argument(
        "--language",
        help="Whisper language code; Parakeet v3 automatically detects language",
    )
    ingest.add_argument(
        "--transcript", help="Explicit JSON/SRT/VTT for one input video"
    )
    ingest.add_argument("--events", help="Explicit event JSON for one input video")
    ingest.add_argument("--audio-model", help="Local AudioSet AST classifier directory")
    ingest.add_argument(
        "--emotion-model", help="Local categorical 16 kHz speech-emotion classifier"
    )
    ingest.add_argument("--vision-model", help="Local CLIP model directory")
    ingest.add_argument(
        "--scene-detector",
        choices=("sampled", "adaptive"),
        default="sampled",
        help="Adaptive uses optional PySceneDetect for frame-level boundaries",
    )
    ingest.add_argument("--audio-threshold", type=float, default=0.3)
    ingest.add_argument(
        "--window", type=float, default=12, help="Candidate duration in seconds"
    )
    ingest.add_argument(
        "--stride", type=float, default=6, help="Candidate grid step in seconds"
    )
    ingest.add_argument("--profile", choices=tuple(WEIGHTS), default="balanced")
    ingest.add_argument("--force", action="store_true")
    model_args(ingest)

    find = command(
        "search", "Search words, semantic topics or event/visual tags across videos"
    )
    find.add_argument("query")
    find.add_argument("--mode", choices=("words", "topic", "event"), default="words")
    find.add_argument("--limit", type=int, default=10)
    find.add_argument("--base-url", default="http://127.0.0.1:8765")
    model_args(find)

    rank = command("highlights", "Rank non-redundant candidates across the catalog")
    rank.add_argument("--limit", type=int, default=10)
    rank.add_argument("--video-id", type=int)
    rank.add_argument("--base-url", default="http://127.0.0.1:8765")

    output = command(
        "export", "Export previews, JSON markers and per-video LosslessCut CSV"
    )
    output.add_argument("--out", required=True)
    output.add_argument("--limit", type=int, default=10)
    output.add_argument("--video-id", type=int)
    output.add_argument(
        "--clips",
        action="store_true",
        help="Render browser-compatible MP4 previews too",
    )
    output.add_argument("--base-url", default="http://127.0.0.1:8765")

    image = command("preview", "Render one still or MP4 preview")
    image.add_argument("segment_id", type=int)
    image.add_argument("--out", required=True)
    image.add_argument("--clip", action="store_true")

    server = command("serve", "Review/search with local video playback at 127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    model_args(server)
    command("status", "Inspect index coverage, warnings and missing source drives")
    command(
        "doctor", "Report tools and optional backend versions without loading models"
    )
    forget = command(
        "remove", "Remove a video from the index; leave source media untouched"
    )
    forget.add_argument("video_id", type=int)
    return root


def doctor():
    packages = {}
    for name in (
        "parakeet-mlx",
        "mlx-whisper",
        "faster-whisper",
        "scenedetect",
        "torch",
        "transformers",
        "sentence-transformers",
        "Pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        fts5 = True
    except sqlite3.Error:
        fts5 = False
    finally:
        conn.close()
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "apple_silicon": platform.system() == "Darwin"
        and platform.machine() == "arm64",
        "sqlite": sqlite3.sqlite_version,
        "fts5": fts5,
        "tools": {
            name: shutil.which(name)
            for name in ("ffmpeg", "ffprobe", "whisper-cli", "mpv")
        },
        "optional_packages": packages,
        "automatic_model_downloads": False,
    }


def execute(args):
    if hasattr(args, "limit") and not 1 <= args.limit <= 1000:
        raise ValueError("--limit must be between 1 and 1000")
    if args.command == "doctor":
        return doctor()
    if args.command == "index":
        import math

        if not (
            math.isfinite(args.window)
            and math.isfinite(args.stride)
            and 1 <= args.stride <= args.window
        ):
            raise ValueError("Require finite 1 <= --stride <= --window")
        if not 0 < args.audio_threshold <= 1:
            raise ValueError("--audio-threshold must be in (0,1]")
        if args.asr == "parakeet" and args.language:
            raise ValueError(
                "Parakeet v3 detects language automatically; omit --language and use the multilingual v3 model"
            )
        paths = discover(args.inputs)
        if not paths:
            raise ValueError("No supported videos found")
        if len(paths) > 1 and (args.transcript or args.events):
            raise ValueError(
                "--transcript/--events apply to one video; use matching sidecars for a batch"
            )
        indexer = Indexer(args)
        results = []
        for path in paths:
            print(f"Indexing {path.name}", file=sys.stderr)
            try:
                results.append(indexer.index(path))
            except (
                ValueError,
                OSError,
                RuntimeError,
                ImportError,
                subprocess.SubprocessError,
                sqlite3.Error,
                KeyError,
                TypeError,
            ) as exc:
                results.append(
                    {"path": str(path), "status": "error", "error": str(exc)}
                )
        return {
            "results": results,
            "errors": sum(row["status"] == "error" for row in results),
        }
    if args.command == "status":
        return status(args.db)
    if args.command == "highlights":
        return highlights(args.db, args.limit, args.base_url, args.video_id)
    if args.command == "search":
        semantic = (
            SemanticModel(args.semantic_model, args.device)
            if args.mode == "topic" and args.semantic_model
            else None
        )
        return search(
            args.db,
            args.query,
            args.mode,
            args.limit,
            semantic,
            model_key(args.semantic_model),
            args.base_url,
        )
    if args.command == "export":
        return export(
            args.db, args.out, args.limit, args.clips, args.base_url, args.video_id
        )
    if args.command == "preview":
        path = preview(get_segment(args.db, args.segment_id), args.out, args.clip)
        return {"output": str(path)}
    if args.command == "remove":
        return {
            "removed_from_index": remove(args.db, args.video_id),
            "source_media_untouched": True,
        }
    if args.command == "serve":
        semantic = (
            SemanticModel(args.semantic_model, args.device)
            if args.semantic_model
            else None
        )
        server = make_server(
            args.db, args.port, semantic, model_key(args.semantic_model)
        )
        print(
            f"Open http://127.0.0.1:{server.server_port} (Ctrl-C to stop)",
            file=sys.stderr,
            flush=True,
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and result.get("errors") else 0
    except KeyboardInterrupt:
        return 130
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as exc:
        print(f"media-index: {exc}", file=sys.stderr)
        return 1
