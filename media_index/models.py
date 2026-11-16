"""Opt-in model adapters. Only local paths; never download weights on ingest."""

import os
import platform
import subprocess
from pathlib import Path

from .transcripts import normalize_segments


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def local_model(path):
    if not path:
        raise ValueError(
            "Provide --model-path (ASR) or a local model directory; no automatic downloads"
        )
    result = Path(path).expanduser().resolve(strict=True)
    return str(result)


def torch_device(requested):
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    return requested


class SpeechModel:
    """Reuse loaded weights across a batch; chunk Parakeet recordings explicitly."""

    def __init__(self, backend, model_path, language=None):
        offline()
        self.backend, self.path, self.language = (
            backend,
            local_model(model_path),
            language,
        )
        self.model = None
        if backend in ("mlx", "parakeet"):
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise ValueError(f"{backend} requires an Apple Silicon Mac")
        if backend == "parakeet":
            from parakeet_mlx import from_pretrained

            self.model = from_pretrained(self.path)
        elif backend == "faster-whisper":
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.path, device="cpu", compute_type="int8", local_files_only=True
            )

    def transcribe(self, wav, duration):
        if self.backend == "parakeet":
            aligned = self.model.transcribe(
                str(wav), chunk_duration=120.0, overlap_duration=15.0
            )
            # Parakeet alignments are subword tokens, not guaranteed word boundaries.
            rows = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "tokens": [
                        {"start": t.start, "end": t.end, "text": t.text}
                        for t in s.tokens
                        if t.end > t.start
                    ],
                }
                for s in aligned.sentences
                if s.end > s.start
            ]
            result = {"segments": rows}
        elif self.backend == "mlx":
            import mlx_whisper

            result = mlx_whisper.transcribe(
                str(wav),
                path_or_hf_repo=self.path,
                language=self.language,
                word_timestamps=True,
                verbose=False,
            )
        elif self.backend == "faster-whisper":
            import wave

            import numpy as np

            # Reuse normalized PCM: PyAV 19 removed av.open's metadata_errors.
            with wave.open(str(wav)) as audio:
                if (
                    audio.getframerate(),
                    audio.getnchannels(),
                    audio.getsampwidth(),
                ) != (16000, 1, 2):
                    raise ValueError("faster-whisper expects extracted mono 16 kHz PCM")
                samples = (
                    np.frombuffer(
                        audio.readframes(audio.getnframes()), dtype="<i2"
                    ).astype("float32")
                    / 32768
                )
            segments, _ = self.model.transcribe(
                samples, language=self.language, word_timestamps=True, vad_filter=True
            )
            result = {
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "words": [
                            {"start": w.start, "end": w.end, "word": w.word}
                            for w in s.words or []
                            if w.end > w.start
                        ],
                    }
                    for s in segments
                    if s.end > s.start
                ]
            }
        elif self.backend == "whisper-cpp":
            import json

            output = Path(wav).parent / "transcript"
            subprocess.run(
                [
                    "whisper-cli",
                    "-m",
                    self.path,
                    "-f",
                    str(wav),
                    "-l",
                    self.language or "auto",
                    "-oj",
                    "-of",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            result = json.loads(output.with_suffix(".json").read_text())
        else:
            raise ValueError(f"Unsupported ASR backend: {self.backend}")
        return normalize_segments(
            result.get("segments", result.get("transcription", [])),
            duration,
            self.backend + ":" + self.path,
        )


class SemanticModel:
    def __init__(self, path, device="auto"):
        offline()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            local_model(path),
            device=torch_device(device),
            local_files_only=True,
            trust_remote_code=False,
        )

    def encode(self, texts, query=False):
        method = self.model.encode_query if query else self.model.encode_document
        return method(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


class AudioEvents:
    """AudioSet or categorical speech-emotion scores on 5 s windows, 2.5 s hop."""

    def __init__(self, path, device="auto", task="audioset"):
        offline()
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.path = local_model(path)
        self.task = task
        self.device = torch_device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.path, local_files_only=True
        )
        self.model = (
            AutoModelForAudioClassification.from_pretrained(
                self.path, local_files_only=True, trust_remote_code=False
            )
            .eval()
            .to(self.device)
        )
        if getattr(self.model.config, "problem_type", None) == "regression":
            raise ValueError(
                "Use a categorical emotion/audio classifier, not a dimensional regression model"
            )
        if self.extractor.sampling_rate != 16000:
            raise ValueError("This prototype requires a 16 kHz audio/emotion model")

    def detect(self, wav, threshold=0.3):
        import wave

        import numpy as np
        import torch

        from .transcripts import event_name

        result = []
        with wave.open(str(wav)) as audio:
            sr, count = audio.getframerate(), audio.getnframes()
            for start in range(0, count, int(sr * 2.5)):
                audio.setpos(start)
                samples = (
                    np.frombuffer(audio.readframes(sr * 5), dtype="<i2").astype(
                        "float32"
                    )
                    / 32768
                )
                if len(samples) < sr * 0.25:
                    continue
                inputs = self.extractor(samples, sampling_rate=sr, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    logits = self.model(**inputs).logits
                    scores = (
                        (
                            logits.softmax(dim=-1)
                            if self.task == "emotion"
                            else logits.sigmoid()
                        )[0]
                        .cpu()
                        .tolist()
                    )
                for idx, score in enumerate(scores):
                    label = self.model.config.id2label[idx]
                    canonical = event_name(label)
                    # Keep the model's other AudioSet classes searchable as well.
                    if score >= threshold:
                        result.append(
                            {
                                "start": start / sr,
                                "end": min(count, start + sr * 5) / sr,
                                "label": canonical,
                                "confidence": round(score, 4),
                                "source": ("ser:" if self.task == "emotion" else "ast:")
                                + self.path,
                            }
                        )
        return result


class VisionTags:
    """CLIP prompt ranking, not calibrated object or emotion detection."""

    LABELS = {
        "person": "a photo of a person",
        "dog": "a photo of a dog",
        "landscape": "a photo of an outdoor landscape",
        "screen": "a photo of a computer screen",
    }

    def __init__(self, path, device="auto"):
        offline()
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessorPil,
            CLIPModel,
            CLIPProcessor,
        )

        self.path = local_model(path)
        self.device = torch_device(device)
        self.processor = CLIPProcessor(
            image_processor=CLIPImageProcessorPil.from_pretrained(
                self.path, local_files_only=True
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                self.path, local_files_only=True, trust_remote_code=False
            ),
        )
        self.model = (
            CLIPModel.from_pretrained(self.path, local_files_only=True)
            .eval()
            .to(self.device)
        )

    def tag(self, frame, width, height):
        import torch
        from PIL import Image

        inputs = self.processor(
            text=list(self.LABELS.values()),
            images=Image.frombytes("RGB", (width, height), frame),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            scores = (
                self.model(**inputs).logits_per_image.softmax(dim=1)[0].cpu().tolist()
            )
        return [
            {
                "label": label,
                "relative_similarity": round(score, 4),
                "source": "clip:" + self.path,
            }
            for label, score in zip(self.LABELS, scores)
            if score >= 0.4
        ]
