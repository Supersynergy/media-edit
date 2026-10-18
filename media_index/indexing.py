"""Resumable batch ingestion with atomic replacement per video."""

import importlib.metadata
import shutil
import tempfile
import time
from pathlib import Path

from .analysis import (
    audio_features,
    extract_audio,
    merge_events,
    probe,
    run,
    visual_features,
)
from .highlights import candidates
from .models import AudioEvents, SemanticModel, SpeechModel, VisionTags
from .store import current, fingerprint, model_key, save
from .transcripts import load_events, load_transcript

EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mts", ".m2ts"}


def discover(inputs):
    result = set()
    for value in inputs:
        path = Path(value).expanduser().resolve(strict=True)
        if path.is_dir():
            result.update(
                p.resolve()
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in EXTENSIONS
                and not any(part.startswith(".") for part in p.relative_to(path).parts)
            )
        else:
            result.add(path)
    return sorted(result)


def sidecar(path, suffixes):
    for suffix in suffixes:
        for candidate in (Path(str(path) + suffix), path.with_suffix(suffix)):
            if candidate.is_file():
                return candidate
    return None


class Indexer:
    def __init__(self, args):
        self.args = args
        self.models = {}
        self.keys = {
            name: model_key(getattr(args, name))
            for name in (
                "model_path",
                "audio_model",
                "emotion_model",
                "semantic_model",
                "vision_model",
            )
        }
        self.versions = {"ffmpeg": run(["ffmpeg", "-version"]).decode().splitlines()[0]}
        for name in (
            "parakeet-mlx",
            "mlx-whisper",
            "faster-whisper",
            "sentence-transformers",
            "transformers",
            "torch",
            "scenedetect",
        ):
            try:
                self.versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.versions[name] = None

    def model(self, name, factory):
        if name not in self.models:
            self.models[name] = factory()
        return self.models[name]

    def index(self, path):
        started = time.monotonic()
        args = self.args
        transcript_path = (
            Path(args.transcript).expanduser().resolve(strict=True)
            if args.transcript
            else sidecar(path, (".json", ".srt", ".vtt"))
        )
        events_path = (
            Path(args.events).expanduser().resolve(strict=True)
            if args.events
            else sidecar(path, (".events.json",))
        )
        options = {
            name: getattr(args, name)
            for name in (
                "asr",
                "language",
                "window",
                "stride",
                "profile",
                "device",
                "audio_threshold",
                "scene_detector",
            )
        }
        options["models"] = self.keys
        options["versions"] = self.versions
        signature = fingerprint(path, [transcript_path, events_path], options)
        if not args.force and current(args.db, path, signature):
            return {
                "path": str(path),
                "status": "unchanged",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        metadata = probe(path)
        duration = metadata["duration"]
        warnings = []
        transcript = []
        backend = args.asr
        if transcript_path and (backend in ("auto", "none") or args.transcript):
            transcript = load_transcript(transcript_path, duration)
            backend = "sidecar"
        elif backend == "auto":
            backend = (
                "whisper-cpp"
                if args.model_path and shutil.which("whisper-cli")
                else "none"
            )
        events = load_events(events_path, duration) if events_path else []
        audio = []
        with tempfile.TemporaryDirectory(prefix="media-index-") as temp:
            wav = Path(temp) / "audio.wav"
            if metadata["has_audio"]:
                extract_audio(path, wav)
                audio, measured = audio_features(wav)
                events.extend(measured)
                if backend not in ("sidecar", "none"):
                    speech = self.model(
                        "asr",
                        lambda: SpeechModel(backend, args.model_path, args.language),
                    )
                    transcript = speech.transcribe(wav, duration)
                if args.audio_model:
                    classifier = self.model(
                        "audio", lambda: AudioEvents(args.audio_model, args.device)
                    )
                    events.extend(classifier.detect(wav, args.audio_threshold))
                if args.emotion_model:
                    classifier = self.model(
                        "emotion",
                        lambda: AudioEvents(
                            args.emotion_model, args.device, task="emotion"
                        ),
                    )
                    events.extend(classifier.detect(wav, args.audio_threshold))
            else:
                warnings.append(
                    "No audio stream: acoustic analysis and ASR unavailable"
                )
                if backend not in ("sidecar", "none", "auto"):
                    raise ValueError("Requested ASR on a video with no audio stream")
            if not transcript:
                warnings.append(
                    "No transcript: add a JSON/SRT/VTT sidecar or configure a local ASR model"
                )
            if not args.audio_model:
                warnings.append(
                    "No acoustic classifier: laughter/applause only from explicit imported events; RMS is not laughter"
                )
            vision = (
                self.model("vision", lambda: VisionTags(args.vision_model, args.device))
                if args.vision_model
                else None
            )
            visuals = visual_features(path, vision)
            if args.scene_detector == "adaptive":
                from scenedetect import AdaptiveDetector, detect

                scenes = detect(str(path), AdaptiveDetector())
                for scene_start, _ in scenes[1:]:
                    start = scene_start.get_seconds()
                    events.append(
                        {
                            "start": start,
                            "end": min(duration, start + 1 / max(1, metadata["fps"])),
                            "label": "scene_change",
                            "confidence": 1.0,
                            "source": "pyscenedetect:adaptive",
                        }
                    )
                for row in visuals:
                    row["scene_change"] = False
                    row["tags"] = [tag for tag in row["tags"] if tag != "scene_change"]
        events = merge_events(events)
        segments = candidates(
            duration,
            transcript,
            audio,
            events,
            visuals,
            args.window,
            args.stride,
            args.profile,
        )
        embeddings = None
        if args.semantic_model:
            semantic = self.model(
                "semantic", lambda: SemanticModel(args.semantic_model, args.device)
            )
            embeddings = semantic.encode(
                [row["text"] or " ".join(row["tags"]) for row in segments]
            )
        metadata.update(
            {
                "warnings": warnings,
                "asr_backend": backend,
                "model_keys": self.keys,
                "profile": args.profile,
                "window": args.window,
                "stride": args.stride,
                "indexed_seconds": round(time.monotonic() - started, 3),
            }
        )
        stat = path.stat()
        metadata.update(
            {
                "file_size": stat.st_size,
                "file_mtime_ns": stat.st_mtime_ns,
                "backend_versions": self.versions,
            }
        )
        evidence = {
            "transcript": transcript,
            "audio": audio,
            "events": events,
            "visuals": visuals,
            "options": options,
        }
        video_id = save(
            args.db,
            path,
            signature,
            metadata,
            evidence,
            segments,
            embeddings,
            self.keys["semantic_model"],
        )
        return {
            "path": str(path),
            "video_id": video_id,
            "status": "indexed",
            "segments": len(segments),
            "elapsed_seconds": metadata["indexed_seconds"],
            "warnings": warnings,
        }
