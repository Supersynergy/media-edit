"""Reproducible offline demo: synthetic video + declared annotations, not an ASR benchmark."""

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_index.analysis import run
from media_index.cli import execute, parser
from media_index.review import export
from media_index.store import highlights, search


def make_fixture(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    first = output / "Interview – O'Brien.mp4"
    second = output / "Vortrag & Wissen.mp4"
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x303050:s=320x180:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc='if(between(t,4,7),0.6,0.015)*sin(2*PI*440*t)':s=16000:d=12",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(first),
        ]
    )
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x547659:s=320x180:r=24:d=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=250:sample_rate=16000:duration=8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(second),
        ]
    )
    first.with_suffix(".json").write_text(
        json.dumps(
            {
                "annotation": "Synthetic test transcript, not recognized speech",
                "segments": [
                    {
                        "start": 0.5,
                        "end": 3,
                        "text": "Willkommen zu unserem Interview.",
                    },
                    {
                        "start": 4,
                        "end": 7,
                        "text": "Wow! Automatisierung spart unserem Hotel viel Zeit!",
                        "words": [
                            {"start": 4, "end": 4.3, "word": "Wow!"},
                            {"start": 4.4, "end": 5, "word": " Automatisierung"},
                            {
                                "start": 5.1,
                                "end": 7,
                                "word": " spart unserem Hotel viel Zeit!",
                            },
                        ],
                    },
                    {
                        "start": 8,
                        "end": 11,
                        "text": "Als Nächstes planen wir den nächsten Schritt.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    second.with_suffix(".json").write_text(
        json.dumps(
            {
                "annotation": "Synthetic test transcript, not recognized speech",
                "segments": [
                    {
                        "start": 1,
                        "end": 3,
                        "text": "Lokale Systeme schützen unsere Daten.",
                    },
                    {
                        "start": 4,
                        "end": 7,
                        "text": "Automatisierung hilft auch beim Wissenstransfer.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    first.with_suffix(".events.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "start": 5,
                        "end": 5.8,
                        "label": "laughter",
                        "confidence": 1,
                        "source": "synthetic_annotation_not_detection",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return [first, second]


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--out", default="demo-output")
    args = options.parse_args()
    root = Path(args.out).resolve()
    sources = make_fixture(root / "sources")
    db = root / "library.sqlite3"
    indexed = execute(
        parser().parse_args(
            [
                "index",
                *map(str, sources),
                "--db",
                str(db),
                "--asr",
                "none",
                "--window",
                "4",
                "--stride",
                "2",
            ]
        )
    )
    if indexed["errors"]:
        raise RuntimeError(indexed)
    words = search(db, "Automatisierung")
    assert len({row["video_id"] for row in words}) == 2, "Cross-video search failed"
    assert search(db, "lachen", "event"), "Imported event search failed"
    assert search(db, "szenenwechsel", "event"), "Measured scene search failed"
    assert any(row["start"] < 6 < row["end"] for row in highlights(db, 2)), (
        "Reaction window missing"
    )
    exported = export(db, root / "review", limit=3, clips=True)
    print(
        json.dumps(
            {
                "fixture": "Synthetic tones/video and explicit transcript/event annotations; no real ASR/event detection claim",
                "index": indexed,
                "cross_video_word_hits": len(words),
                "export": exported,
                "review_command": shlex.join(
                    [sys.executable, "-m", "media_index", "serve", "--db", str(db)]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
