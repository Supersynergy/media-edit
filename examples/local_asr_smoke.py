"""Actual ASR + semantic smoke on synthetic English speech, using LOCAL pretrained models.

Needs FFmpeg with the flite filter. It never downloads model weights.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_index.analysis import run
from media_index.cli import execute, parser
from media_index.models import SemanticModel
from media_index.store import model_key, search


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--model-path", required=True)
    args_parser.add_argument(
        "--asr",
        default="faster-whisper",
        choices=("faster-whisper", "parakeet", "mlx", "whisper-cpp"),
    )
    args_parser.add_argument("--semantic-model")
    args_parser.add_argument("--out", default="demo-output/asr-smoke")
    args = args_parser.parse_args()
    root = Path(args.out).resolve()
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for name, text in (
        (
            "hotel",
            "Our hotel saves time with automation. Local video search finds the best moments.",
        ),
        (
            "dogs",
            "The dogs run through the forest. They play outside in the fresh air.",
        ),
    ):
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
                "color=c=0x303050:s=320x180:r=24:d=30",
                "-f",
                "lavfi",
                "-i",
                f"flite=text='{text}':voice=slt",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(sources / (name + ".mp4")),
            ]
        )
    db = root / "library.sqlite3"
    command = [
        "index",
        str(sources),
        "--db",
        str(db),
        "--asr",
        args.asr,
        "--model-path",
        str(Path(args.model_path).resolve()),
        "--window",
        "8",
        "--stride",
        "4",
        "--device",
        "cpu",
        "--force",
    ]
    if args.semantic_model:
        command += ["--semantic-model", str(Path(args.semantic_model).resolve())]
    result = execute(parser().parse_args(command))
    if result["errors"]:
        raise RuntimeError(result)
    words = search(db, "hotel")
    assert words and all(Path(row["path"]).stem == "hotel" for row in words), (
        "ASR hotel search failed"
    )
    semantic_hits = []
    if args.semantic_model:
        semantic_hits = search(
            db,
            "Improving productivity in guest accommodation",
            mode="topic",
            semantic=SemanticModel(args.semantic_model, "cpu"),
            semantic_key=model_key(args.semantic_model),
        )
        assert Path(semantic_hits[0]["path"]).stem == "hotel", (
            "Semantic accommodation query selected the wrong video"
        )
    report = {
        "input": "Two synthetic English flite utterances; no supplied transcript sidecars",
        "index": result,
        "asr_word_hits": [
            {"path": row["path"], "text": row["text"], "matches": row["matches"]}
            for row in words
        ],
        "semantic_hits": [
            {"path": row["path"], "similarity": row["similarity"]}
            for row in semantic_hits
        ],
    }
    (root / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
