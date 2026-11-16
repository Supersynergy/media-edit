"""Transactional SQLite/FTS5 catalog, provenance, and local semantic retrieval."""

import hashlib
import json
import math
import re
import shlex
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .highlights import diverse
from .transcripts import event_name

SCHEMA = 1


@contextmanager
def connect(db):
    db = Path(db).expanduser().resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA):
            raise ValueError(f"Unsupported index schema {version}; expected {SCHEMA}")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE NOT NULL,
                fingerprint TEXT NOT NULL, duration REAL NOT NULL, metadata TEXT NOT NULL,
                evidence TEXT NOT NULL, indexed_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
                tags TEXT NOT NULL, score REAL NOT NULL, data TEXT NOT NULL,
                embedding TEXT, model_key TEXT);
            CREATE INDEX IF NOT EXISTS segment_video ON segments(video_id);
            CREATE INDEX IF NOT EXISTS segment_score ON segments(score DESC);
            CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(text, tags, tokenize='unicode61 remove_diacritics 2');
        """)
        conn.execute(f"PRAGMA user_version={SCHEMA}")
        conn.commit()
        yield conn
    finally:
        conn.close()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_key(path):
    if not path:
        return None
    root = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")) if root.is_dir() else [root]:
        relative = item.relative_to(root) if root.is_dir() else Path(item.name)
        if item.is_file() and ".cache" not in relative.parts:
            # Config + weight metadata identify the local snapshot without reading GB twice.
            stat = item.stat()
            digest.update(
                str(item.relative_to(root) if root.is_dir() else item.name).encode()
            )
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            if item.suffix in (".json", ".txt"):
                digest.update(item.read_bytes())
    return digest.hexdigest()


def fingerprint(path, sidecars, options):
    payload = {
        "version": __version__,
        "media": file_digest(path),
        "sidecars": {str(p): file_digest(p) for p in sidecars if p},
        "options": options,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def current(db, path, signature):
    with connect(db) as conn:
        row = conn.execute(
            "SELECT id, fingerprint FROM videos WHERE path=?", (str(path),)
        ).fetchone()
        return bool(row and row["fingerprint"] == signature)


def save(
    db,
    path,
    signature,
    metadata,
    evidence,
    segments,
    embeddings=None,
    semantic_key=None,
):
    with connect(db) as conn, conn:
        previous = conn.execute(
            "SELECT id FROM videos WHERE path=?", (str(path),)
        ).fetchone()
        if previous:
            video_id = previous["id"]
            conn.execute(
                "DELETE FROM search_fts WHERE rowid IN (SELECT id FROM segments WHERE video_id=?)",
                (video_id,),
            )
            conn.execute("DELETE FROM segments WHERE video_id=?", (video_id,))
            conn.execute(
                "UPDATE videos SET fingerprint=?, duration=?, metadata=?, evidence=?, indexed_at=? WHERE id=?",
                (
                    signature,
                    metadata["duration"],
                    json.dumps(metadata),
                    json.dumps(evidence),
                    datetime.now(timezone.utc).isoformat(),
                    video_id,
                ),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO videos(path,fingerprint,duration,metadata,evidence,indexed_at) VALUES(?,?,?,?,?,?)",
                (
                    str(path),
                    signature,
                    metadata["duration"],
                    json.dumps(metadata),
                    json.dumps(evidence),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            video_id = cursor.lastrowid
        for idx, row in enumerate(segments):
            cursor = conn.execute(
                """INSERT INTO segments(video_id,start,end,text,tags,score,data,embedding,model_key)
                                     VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    video_id,
                    row["start"],
                    row["end"],
                    row["text"],
                    " ".join(row["tags"]),
                    row["score"],
                    json.dumps(row),
                    json.dumps(embeddings[idx]) if embeddings else None,
                    semantic_key,
                ),
            )
            conn.execute(
                "INSERT INTO search_fts(rowid,text,tags) VALUES(?,?,?)",
                (cursor.lastrowid, row["text"], " ".join(row["tags"])),
            )
    return video_id


SELECT = """SELECT s.*, v.path, v.fingerprint, v.duration FROM segments s JOIN videos v ON v.id=s.video_id"""


def result(row, base_url="http://127.0.0.1:8765"):
    data = json.loads(row["data"])
    data.update(
        {
            "id": row["id"],
            "video_id": row["video_id"],
            "path": row["path"],
            "fingerprint": row["fingerprint"],
            "duration": row["duration"],
        }
    )
    start, end = data["start"], data["end"]
    data["play_url"] = f"{base_url.rstrip('/')}/watch/{row['id']}"
    data["thumbnail_url"] = f"{base_url.rstrip('/')}/thumb/{row['id']}"
    data["file_uri"] = Path(row["path"]).as_uri() + f"#t={start:.3f},{end:.3f}"
    data["mpv_command"] = shlex.join(
        ["mpv", f"--start={start:.3f}", f"--end={end:.3f}", "--", row["path"]]
    )
    data["ffmpeg_command"] = shlex.join(
        [
            "ffmpeg",
            "-nostdin",
            "-n",
            "-ss",
            f"{start:.3f}",
            "-i",
            row["path"],
            "-t",
            f"{end - start:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            f"highlight-{row['id']}.mp4",
        ]
    )
    return data


def get_segment(db, segment_id, base_url="http://127.0.0.1:8765"):
    with connect(db) as conn:
        row = conn.execute(SELECT + " WHERE s.id=?", (segment_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown segment {segment_id}")
    return result(row, base_url)


def highlights(db, limit=10, base_url="http://127.0.0.1:8765", video_id=None):
    with connect(db) as conn:
        where, params = (" WHERE s.video_id=?", [video_id]) if video_id else ("", [])
        rows = conn.execute(
            SELECT + where + " ORDER BY s.score DESC,s.id", params
        ).fetchall()
    return diverse([result(row, base_url) for row in rows], limit)


def search(
    db,
    query,
    mode="words",
    limit=10,
    semantic=None,
    semantic_key=None,
    base_url="http://127.0.0.1:8765",
):
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return []
    with connect(db) as conn:
        if mode == "topic":
            if semantic is None:
                raise ValueError(
                    "Topic search needs --semantic-model /local/model, used during indexing too"
                )
            query_vec = semantic.encode([query], query=True)[0]
            rows = conn.execute(
                SELECT + " WHERE s.embedding IS NOT NULL AND s.model_key=?",
                (semantic_key,),
            ).fetchall()
            if not rows:
                raise ValueError(
                    "No embeddings match this local model snapshot; re-index with --semantic-model"
                )
            scored = []
            for row in rows:
                vector = json.loads(row["embedding"])
                if len(vector) != len(query_vec):
                    raise ValueError(
                        "Embedding dimensions differ; re-index with this model"
                    )
                denom = math.sqrt(
                    sum(v * v for v in vector) * sum(v * v for v in query_vec)
                )
                similarity = (
                    sum(a * b for a, b in zip(vector, query_vec)) / denom
                    if denom
                    else 0
                )
                item = result(row, base_url)
                item["similarity"] = round(similarity, 5)
                scored.append(item)
            return diverse(sorted(scored, key=lambda item: -item["similarity"]), limit)
        if mode == "event":
            tokens = [event_name(query)]
        expression = " AND ".join(
            '"' + token.replace('"', '""') + '"' for token in tokens
        )
        field = "tags" if mode == "event" else "text"
        rows = conn.execute(
            SELECT
            + " JOIN search_fts ON search_fts.rowid=s.id WHERE search_fts MATCH ? "
            "ORDER BY bm25(search_fts),s.score DESC,s.id",
            (f"{field}:({expression})",),
        ).fetchall()
    hits = []
    for row in rows:
        item = result(row, base_url)
        if mode == "words":
            # Preserve original sentence/word evidence separately from the contextual clip.
            item["matches"] = [
                {
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": cue["text"],
                    "words": cue.get("words", []),
                    "tokens": cue.get("tokens", []),
                }
                for cue in item["transcript_matches"]
                if any(token.casefold() in cue["text"].casefold() for token in tokens)
            ]
        else:
            item["matches"] = [
                event for event in item["events"] if event["label"] == tokens[0]
            ]
            if not item["matches"]:
                for sample in item["visual_evidence"]:
                    if tokens[0] in sample["tags"]:
                        item["matches"].append(
                            {
                                "start": sample["time"],
                                "end": min(item["duration"], sample["time"] + 0.5),
                                "label": tokens[0],
                                "source": "sampled_visual_features",
                                "resolution_seconds": 0.5,
                            }
                        )
                    item["matches"].extend(
                        {
                            **tag,
                            "start": sample["time"],
                            "end": min(item["duration"], sample["time"] + 0.5),
                        }
                        for tag in sample["model_tags"]
                        if tag["label"] == tokens[0]
                    )
        precise = []
        if mode == "words":
            for cue in item["matches"]:
                precise.extend(
                    word
                    for word in cue["words"]
                    if word["start"] < item["end"]
                    and word["end"] > item["start"]
                    and any(
                        token.casefold() in word["word"].casefold() for token in tokens
                    )
                )
        matches = precise or item["matches"]
        if matches:
            item["matched_start"] = max(item["start"], matches[0]["start"])
            item["matched_end"] = min(item["end"], matches[0]["end"])
            item["match_url"] = item["play_url"] + f"?at={item['matched_start']:.3f}"
        hits.append(item)
    return diverse(hits, limit)


def status(db):
    with connect(db) as conn:
        videos = [
            dict(row)
            for row in conn.execute(
                "SELECT id,path,duration,indexed_at,metadata FROM videos ORDER BY path"
            )
        ]
        count = conn.execute("SELECT count(*) FROM segments").fetchone()[0]
    for video in videos:
        video["available"] = Path(video["path"]).is_file()
        video["metadata"] = json.loads(video["metadata"])
        if video["available"]:
            stat = Path(video["path"]).stat()
            video["changed_since_index"] = stat.st_size != video["metadata"].get(
                "file_size"
            ) or stat.st_mtime_ns != video["metadata"].get("file_mtime_ns")
    return {"schema": SCHEMA, "videos": videos, "segments": count}


def remove(db, video_id):
    with connect(db) as conn, conn:
        conn.execute(
            "DELETE FROM search_fts WHERE rowid IN (SELECT id FROM segments WHERE video_id=?)",
            (video_id,),
        )
        return conn.execute("DELETE FROM videos WHERE id=?", (video_id,)).rowcount
