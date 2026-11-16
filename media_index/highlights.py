"""Explainable candidate ranking. Scores are editorial heuristics, not probabilities."""

import math
import re
import statistics
from bisect import bisect_left, bisect_right

WEIGHTS = {
    "balanced": {
        "transcript": 0.25,
        "audio_event": 0.30,
        "loudness": 0.15,
        "scene": 0.10,
        "motion": 0.10,
        "visual": 0.10,
    },
    "interview": {
        "transcript": 0.55,
        "audio_event": 0.20,
        "loudness": 0.05,
        "scene": 0.05,
        "motion": 0.05,
        "visual": 0.10,
    },
    "event": {
        "transcript": 0.10,
        "audio_event": 0.40,
        "loudness": 0.20,
        "scene": 0.10,
        "motion": 0.15,
        "visual": 0.05,
    },
}
REACTION = re.compile(
    r"\b(wow|oh|oho|krass|unglaublich|amazing|unbelievable|haha|hahaha)\b|!", re.I
)
POSITIVE_EVENTS = {
    "laughter",
    "applause",
    "cheering",
    "shouting",
    "exclamation",
    "joy",
    "surprise",
}


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


class Timeline:
    """Slice time-sorted evidence without rescanning an entire recording per window."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: row["start"])
        self.starts = [row["start"] for row in self.rows]
        self.prefix_ends = []
        end = 0
        for row in self.rows:
            end = max(end, row["end"])
            self.prefix_ends.append(end)

    def between(self, start, end):
        lo = bisect_right(self.prefix_ends, start)
        hi = bisect_left(self.starts, end)
        return [row for row in self.rows[lo:hi] if row["end"] > start]


def candidates(
    duration,
    transcript,
    audio,
    events,
    visuals,
    window=12.0,
    stride=6.0,
    profile="balanced",
):
    if not 1 <= stride <= window or not math.isfinite(window):
        raise ValueError("Require 1 <= stride <= window and finite times")
    # Grid + sentence/event anchors preserve context around sparse reactions.
    starts = {0.0, max(0.0, duration - window)}
    starts.update(
        i * stride
        for i in range(math.ceil(duration / stride))
        if i * stride + window <= duration
    )
    starts.update(
        max(0.0, min(row["start"] - 1.5, duration - window))
        for row in [*transcript, *events]
        if row.get("label") != "silence"
    )
    dbs = [row["dbfs"] for row in audio]
    baseline = statistics.median(dbs) if dbs else -60
    speech_timeline, event_timeline, audio_timeline = (
        Timeline(rows) for rows in (transcript, events, audio)
    )
    visual_times = [row["time"] for row in visuals]
    result = []
    for start in sorted(starts):
        end = min(duration, start + window)
        region = {"start": start, "end": end}
        speech = speech_timeline.between(start, end)
        selected_events = event_timeline.between(start, end)
        visual = visuals[
            bisect_left(visual_times, start) : bisect_left(visual_times, end)
        ]
        sound = audio_timeline.between(start, end)
        # Word timings, when available, avoid returning a long sentence for every window.
        texts = []
        for row in speech:
            words = [w["word"] for w in row.get("words", []) if overlap(w, region)]
            tokens = [t["text"] for t in row.get("tokens", []) if overlap(t, region)]
            texts.append(
                "".join(words).strip()
                if row.get("words")
                else "".join(tokens).strip()
                if row.get("tokens")
                else row["text"]
            )
        text = " ".join(t for t in texts if t)
        reactions = len(REACTION.findall(text))
        if reactions:
            selected_events.append(
                {
                    "start": start,
                    "end": end,
                    "label": "exclamation",
                    "confidence": 0.5,
                    "source": "transcript_lexicon_hint",
                }
            )
        tags = sorted(
            {row["label"] for row in selected_events}
            | {tag for row in visual for tag in row["tags"]}
            | {tag["label"] for row in visual for tag in row["model_tags"]}
        )
        event_score = max(
            (
                row["confidence"]
                for row in selected_events
                if row["label"] in POSITIVE_EVENTS
                and row["source"] != "transcript_lexicon_hint"
            ),
            default=0,
        )
        peak_db = max((row["dbfs"] for row in sound), default=-60)
        loudness = min(1, max(0, (peak_db - baseline) / 18)) if peak_db > -50 else 0
        adaptive_scenes = [
            row for row in selected_events if row["source"] == "pyscenedetect:adaptive"
        ]
        scene_count = len(adaptive_scenes) or sum(row["scene_change"] for row in visual)
        motion = statistics.mean(row["motion"] for row in visual) if visual else 0
        contrast = statistics.mean(row["contrast"] for row in visual) if visual else 0
        dark = sum(row["brightness"] < 0.08 for row in visual) / max(len(visual), 1)
        silence = sum(overlap(row, region) for row in sound if row["dbfs"] < -50) / (
            end - start
        )
        components = {
            "transcript": min(
                1,
                len(text.split()) / max(1, end - start) / 3
                + min(0.7, reactions * 0.35),
            ),
            "audio_event": event_score,
            "loudness": loudness,
            "scene": min(1, scene_count / 2),
            "motion": min(1, motion / 0.12),
            "visual": min(1, contrast / 0.20) * (1 - dark),
        }
        score = sum(components[key] * value for key, value in WEIGHTS[profile].items())
        penalty = 0.15 * silence + 0.10 * dark
        reasons = []
        if text:
            reasons.append(
                f"{len(text.split())} transcript words; sentence context included"
            )
        if reactions:
            reasons.append(
                f"{reactions} transcript reaction hints; no acoustic emotion claim"
            )
        for row in selected_events:
            if (
                row["label"] in POSITIVE_EVENTS
                and row["source"] != "transcript_lexicon_hint"
            ):
                reasons.append(
                    f"{row['label']}: {row['confidence']:.2f} ({row['source']})"
                )
        if loudness:
            reasons.append(
                f"Peak RMS {peak_db:.1f} dBFS; {peak_db - baseline:+.1f} dB vs video median"
            )
        if scene_count:
            reasons.append(
                f"{scene_count} frame-level scene changes (PySceneDetect)"
                if adaptive_scenes
                else f"{scene_count} sampled scene changes (0.5 s resolution)"
            )
        if motion > 0.02:
            reasons.append(
                f"Visual frame difference {motion:.3f}; includes camera motion/cuts"
            )
        if silence > 0.25:
            reasons.append(f"Silence penalty: {silence:.0%} of window")
        if dark > 0.25:
            reasons.append(f"Dark-frame penalty: {dark:.0%} of samples")
        result.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "tags": tags,
                "events": selected_events,
                "score": round(max(0, score - penalty) * 100, 3),
                "components": {
                    key: round(value, 4) for key, value in components.items()
                },
                "penalty": round(penalty, 4),
                "profile": profile,
                "reasons": list(dict.fromkeys(reasons))
                or ["Low-signal fallback candidate"],
                "transcript_matches": speech,
                "visual_evidence": [
                    row for row in visual if row["tags"] or row["model_tags"]
                ],
                "metrics": {
                    "peak_dbfs": peak_db,
                    "median_dbfs": baseline,
                    "silence_fraction": round(silence, 4),
                    "dark_fraction": round(dark, 4),
                },
            }
        )
    return result


def diverse(rows, limit, max_overlap=0.35):
    """Suppress redundant windows within the same video, retaining cross-video hits."""
    selected = []
    for row in rows:
        if all(
            row["video_id"] != other["video_id"]
            or overlap(row, other)
            / min(row["end"] - row["start"], other["end"] - other["start"])
            <= max_overlap
            for other in selected
        ):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected
