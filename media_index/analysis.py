"""Bounded-memory audio/visual measurements using the existing FFmpeg install."""

import array
import json
import math
import statistics
import subprocess
import sys
import tempfile
import wave


def run(args):
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{args[0]} failed: {result.stderr.decode(errors='replace')[-2000:]}"
        )
    return result.stdout


def probe(path):
    data = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    videos = [
        s
        for s in data["streams"]
        if s["codec_type"] == "video"
        and not s.get("disposition", {}).get("attached_pic")
    ]
    if not videos:
        raise ValueError(f"Not a video: {path}")
    duration = float(data["format"].get("duration", videos[0].get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A finite positive duration is required")
    fps_parts = videos[0].get("avg_frame_rate", "0/1").split("/")
    fps = float(fps_parts[0]) / max(float(fps_parts[-1]), 1)
    return {
        "duration": duration,
        "fps": fps,
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
        "width": videos[0]["width"],
        "height": videos[0]["height"],
    }


def extract_audio(path, wav):
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
    )


def audio_features(wav):
    result = []
    with wave.open(str(wav)) as audio:
        rate = audio.getframerate()
        pos = 0
        while True:
            raw = audio.readframes(rate // 2)
            if not raw:
                break
            values = array.array("h", raw)
            if sys.byteorder != "little":
                values.byteswap()
            rms = math.sqrt(sum(x * x for x in values) / len(values)) / 32768
            peak = max(abs(x) for x in values) / 32768
            result.append(
                {
                    "start": pos / rate,
                    "end": (pos + len(values)) / rate,
                    "dbfs": round(20 * math.log10(max(rms, 1e-6)), 3),
                    "peak": round(peak, 4),
                }
            )
            pos += len(values)
    events = []
    median = statistics.median(row["dbfs"] for row in result) if result else -60
    previous = median
    for row in result:
        db = row["dbfs"]
        if db < -50:
            events.append(
                {**row, "label": "silence", "confidence": 1.0, "source": "rms<-50dBFS"}
            )
        if db > -40 and (db - previous >= 8 or db - median >= 8):
            events.append(
                {
                    **row,
                    "label": "audio_peak",
                    "confidence": 1.0,
                    "source": "rms_delta>=8dB",
                }
            )
        previous = db
    return result, merge_events(events)


def merge_events(events, gap=0.05):
    result = []
    for row in sorted(events, key=lambda e: (e["label"], e["source"], e["start"])):
        if (
            result
            and row["label"] == result[-1]["label"]
            and row["source"] == result[-1]["source"]
            and row["start"] <= result[-1]["end"] + gap
        ):
            result[-1]["end"] = max(result[-1]["end"], row["end"])
            result[-1]["confidence"] = max(result[-1]["confidence"], row["confidence"])
        else:
            result.append(dict(row))
    return sorted(result, key=lambda e: e["start"])


def frames(path, size=(64, 36), sample_fps=2):
    width, height = size
    filters = (
        f"fps={sample_fps}:start_time=0,scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=rgb24"
    )
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-an",
        "-vf",
        filters,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as error:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error)
        try:
            index = 0
            while True:
                frame = proc.stdout.read(width * height * 3)
                if not frame:
                    break
                if len(frame) != width * height * 3:
                    raise RuntimeError("Truncated RGB frame")
                yield index / sample_fps, frame
                index += 1
            if proc.wait():
                error.seek(0)
                raise RuntimeError(error.read().decode(errors="replace")[-2000:])
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def visual_features(path, vision=None):
    result, previous, prev_hist = [], None, None
    last_cut = -1.0
    width, height = (224, 126) if vision else (64, 36)
    for timestamp, frame in frames(path, size=(width, height)):
        channels = [frame[offset::3] for offset in range(3)]
        means = [statistics.mean(channel) / 255 for channel in channels]
        luminance = [
            (r * 0.2126 + g * 0.7152 + b * 0.0722) / 255 for r, g, b in zip(*channels)
        ]
        hist = [0] * 48
        for offset, channel in enumerate(channels):
            for value in channel:
                hist[offset * 16 + min(value // 16, 15)] += 1 / (width * height * 3)
        delta = sum(abs(a - b) for a, b in zip(hist, prev_hist)) / 2 if prev_hist else 0
        motion = (
            sum(abs(a - b) for a, b in zip(frame, previous)) / len(frame) / 255
            if previous
            else 0
        )
        cut = delta >= 0.35 and timestamp - last_cut >= 0.75
        if cut:
            last_cut = timestamp
        brightness = statistics.mean(luminance)
        tags = []
        if brightness < 0.08:
            tags.append("dark")
        if brightness > 0.8:
            tags.append("bright")
        if motion > 0.06:
            tags.append("motion")
        dominant = max(range(3), key=means.__getitem__)
        if means[dominant] > 0.25 and means[dominant] - sorted(means)[-2] > 0.15:
            tags.append(("red", "green", "blue")[dominant])
        if cut:
            tags.append("scene_change")
        # Reuse a small number of higher-resolution samples for optional CLIP.
        model_tags = []
        if vision and round(timestamp * 2) % 4 == 0:
            model_tags = vision.tag(frame, width, height)
        result.append(
            {
                "time": timestamp,
                "brightness": round(brightness, 4),
                "contrast": round(statistics.pstdev(luminance), 4),
                "motion": round(motion, 4),
                "scene_delta": round(delta, 4),
                "scene_change": cut,
                "tags": tags,
                "model_tags": model_tags,
            }
        )
        previous, prev_hist = frame, hist
    return result
