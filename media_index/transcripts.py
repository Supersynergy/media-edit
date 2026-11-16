"""Validate timestamped sidecars without changing the source media."""

import json
import math
import re
from pathlib import Path


def seconds(value):
    if isinstance(value, str) and ":" in value:
        result = 0.0
        for part in value.replace(",", ".").split(":"):
            result = result * 60 + float(part)
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError("Timestamps must be finite")
    return result


def interval(start, end, duration):
    start, end = seconds(start), seconds(end)
    if start < 0 or end <= start or end > duration + 0.25:
        raise ValueError(f"Invalid interval {start}..{end} for duration {duration:.3f}")
    end = min(end, duration)
    if start >= end:
        raise ValueError("Interval starts beyond the media duration")
    return round(start, 6), round(end, 6)


def normalize_segments(rows, duration, source):
    result = []
    for row in rows:
        if "offsets" in row:  # whisper.cpp JSON uses milliseconds
            start, end = (row["offsets"][key] / 1000 for key in ("from", "to"))
        elif "timestamps" in row:
            start, end = (row["timestamps"][key] for key in ("from", "to"))
        else:
            start, end = row["start"], row["end"]
        start, end = interval(start, end, duration)
        text = str(row.get("text", "")).strip()
        words = []
        for word in row.get("words", []):
            ws, we = interval(word["start"], word["end"], duration)
            if ws < start - 0.25 or we > end + 0.25:
                raise ValueError("Word timing must fall inside its transcript segment")
            words.append({"start": ws, "end": we, "word": str(word.get("word", ""))})
        if text:
            tokens = []
            for token in row.get("tokens", []):
                ts, te = interval(token["start"], token["end"], duration)
                if ts < start - 0.25 or te > end + 0.25:
                    raise ValueError(
                        "Token timing must fall inside its transcript segment"
                    )
                tokens.append({"start": ts, "end": te, "text": str(token["text"])})
            result.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": words,
                    "tokens": tokens,
                    "source": source,
                }
            )
    return sorted(result, key=lambda row: row["start"])


def load_transcript(path, duration):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        rows = (
            data
            if isinstance(data, list)
            else data.get("segments", data.get("transcription"))
        )
        if rows is None or not isinstance(rows, list):
            raise ValueError(
                f"Expected segments or whisper.cpp transcription in {path}"
            )
    else:
        # SRT and WebVTT, including cue identifiers and VTT cue settings.
        rows = []
        for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
            lines = block.splitlines()
            for i, line in enumerate(lines):
                match = re.match(r"\s*([\d:.,]+)\s+-->\s+([\d:.,]+)", line)
                if match:
                    rows.append(
                        {
                            "start": match[1],
                            "end": match[2],
                            "text": " ".join(lines[i + 1 :]),
                        }
                    )
                    break
        if not rows:
            raise ValueError(f"No timestamped cues in {path}")
    return normalize_segments(rows, duration, f"sidecar:{path.name}")


ALIASES = {
    "lachen": "laughter",
    "laughter": "laughter",
    "laugh": "laughter",
    "lacher": "laughter",
    "gelächter": "laughter",
    "giggle": "laughter",
    "applaus": "applause",
    "applause": "applause",
    "clapping": "applause",
    "jubel": "cheering",
    "cheer": "cheering",
    "cheering": "cheering",
    "ausruf": "exclamation",
    "ausrufe": "exclamation",
    "wow": "exclamation",
    "exclamation": "exclamation",
    "shout": "shouting",
    "schrei": "shouting",
    "shouting": "shouting",
    "screaming": "shouting",
    "schreien": "shouting",
    "freude": "joy",
    "joy": "joy",
    "happy": "joy",
    "surprise": "surprise",
    "überraschung": "surprise",
    "surprised": "surprise",
    "stille": "silence",
    "silence": "silence",
    "laut": "audio_peak",
    "audio_peak": "audio_peak",
    "bewegung": "motion",
    "motion": "motion",
    "schnitt": "scene_change",
    "szenenwechsel": "scene_change",
    "scene_change": "scene_change",
    "dunkel": "dark",
    "dark": "dark",
    "hell": "bright",
    "bright": "bright",
    "hund": "dog",
    "dog": "dog",
    "person": "person",
    "landschaft": "landscape",
    "landscape": "landscape",
    "bildschirm": "screen",
    "screen": "screen",
    "rot": "red",
    "red": "red",
    "blau": "blue",
    "blue": "blue",
    "grün": "green",
    "green": "green",
    "baby laughter": "laughter",
    "chuckle, chortle": "laughter",
    "snicker": "laughter",
    "belly laugh": "laughter",
    "giggle, titter": "laughter",
    "yell": "shouting",
    "hap": "joy",
    "exc": "excitement",
    "sur": "surprise",
    "ang": "anger",
    "wut": "anger",
    "anger": "anger",
    "sad": "sadness",
    "traurig": "sadness",
    "sadness": "sadness",
    "neu": "neutral",
    "neutral": "neutral",
}


def event_name(label):
    return ALIASES.get(label.strip().casefold(), label.strip().casefold())


def load_events(path, duration):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data["events"]
    result = []
    for row in rows:
        start, end = interval(row["start"], row["end"], duration)
        confidence = float(row.get("confidence", 1.0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Event confidence must be in [0,1]")
        label = event_name(str(row["label"]))
        if not label:
            raise ValueError("Event label must not be empty")
        result.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "confidence": confidence,
                "source": "import:" + str(row.get("source", "unspecified")),
            }
        )
    return result
