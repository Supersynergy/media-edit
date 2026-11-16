import csv
import hashlib
import http.client
import json
import shutil
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from examples.local_search_demo import make_fixture
from media_index.analysis import audio_features, extract_audio, probe, run
from media_index.cli import execute, parser
from media_index.models import SpeechModel, local_model
from media_index.review import export, make_server
from media_index.store import (
    connect,
    highlights,
    model_key,
    search,
    status,
)
from media_index.transcripts import load_transcript


def ingest(paths, db, extra=()):
    return execute(
        parser().parse_args(
            [
                "index",
                *map(str, paths),
                "--db",
                str(db),
                "--asr",
                "none",
                "--window",
                "4",
                "--stride",
                "2",
                *extra,
            ]
        )
    )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required"
)
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="media-index-tests-")
        cls.root = Path(cls.temp.name)
        cls.paths = make_fixture(cls.root / "source files")
        cls.db = cls.root / "library.sqlite3"
        cls.initial_hashes = [
            hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths
        ]
        result = ingest(cls.paths, cls.db)
        if result["errors"]:
            raise RuntimeError(result)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cross_video_search_and_original_word_times(self):
        rows = search(self.db, "Automatisierung")
        self.assertEqual(len({row["video_id"] for row in rows}), 2)
        interview = next(row for row in rows if "Interview" in row["path"])
        self.assertTrue(
            any(
                word["start"] == 4.4
                for cue in interview["matches"]
                for word in cue["words"]
            )
        )
        self.assertIn("O'", interview["path"])
        self.assertIn("http://127.0.0.1:8765/watch/", interview["play_url"])
        self.assertIn("-c:v libx264", interview["ffmpeg_command"])
        self.assertEqual(interview["matched_start"], 4.4)
        self.assertTrue(interview["match_url"].endswith("?at=4.400"))

    def test_audio_scene_evidence_and_provenance(self):
        laughter = search(self.db, "lachen", "event")
        self.assertTrue(laughter)
        self.assertTrue(
            all(
                "import:synthetic_annotation_not_detection" == e["source"]
                for row in laughter
                for e in row["matches"]
            )
        )
        self.assertTrue(search(self.db, "szenenwechsel", "event"))
        self.assertTrue(search(self.db, "laut", "event"))
        self.assertFalse(search(self.db, "applaus", "event"))
        ranked = highlights(self.db, 3)
        self.assertTrue(ranked[0]["start"] < 6 < ranked[0]["end"])
        self.assertGreater(ranked[0]["components"]["audio_event"], 0)
        self.assertTrue(
            any("synthetic_annotation" in reason for reason in ranked[0]["reasons"])
        )

    def test_resume_does_not_duplicate_rows_or_fts(self):
        before = status(self.db)["segments"]
        result = ingest(self.paths, self.db)
        self.assertTrue(all(row["status"] == "unchanged" for row in result["results"]))
        self.assertEqual(status(self.db)["segments"], before)
        with connect(self.db) as conn:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM search_fts").fetchone()[0], before
            )

    def test_source_media_unchanged(self):
        self.assertEqual(
            [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths],
            self.initial_hashes,
        )

    def test_export_real_playable_previews_and_per_source_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            result = export(self.db, temp, limit=3, clips=True)
            payload = json.loads((Path(temp) / "candidates.json").read_text())
            self.assertEqual(result["candidates"], len(payload["candidates"]))
            self.assertFalse(payload["score_is_probability"])
            for row in payload["candidates"]:
                self.assertTrue(
                    (Path(temp) / row["preview_image"])
                    .read_bytes()
                    .startswith(b"\xff\xd8")
                )
                metadata = probe(Path(temp) / row["preview_clip"])
                self.assertAlmostEqual(
                    metadata["duration"], row["end"] - row["start"], delta=0.15
                )
            for key, video in payload["videos"].items():
                with (Path(temp) / video["losslesscut_csv"]).open() as stream:
                    rows = list(csv.reader(stream))
                self.assertTrue(all(float(row[0]) < float(row[1]) for row in rows))
                self.assertEqual(
                    len(rows),
                    sum(r["video_id"] == int(key) for r in payload["candidates"]),
                )
            self.assertIn(
                "synthetic_annotation", (Path(temp) / "review.html").read_text()
            )

    def test_local_review_byte_ranges_and_no_arbitrary_file_routes(self):
        server = make_server(self.db, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            client.request("GET", "/api/search?q=Automatisierung&limit=10")
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            rows = json.loads(response.read())
            self.assertEqual(len({r["video_id"] for r in rows}), 2)
            media_route = "/media/" + str(rows[0]["video_id"])
            source = Path(rows[0]["path"]).read_bytes()
            for header, expected in (
                ("bytes=2-17", source[2:18]),
                ("bytes=-12", source[-12:]),
            ):
                client.request("GET", media_route, headers={"Range": header})
                response = client.getresponse()
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)
            client.request("HEAD", media_route)
            response = client.getresponse()
            self.assertEqual(int(response.getheader("Content-Length")), len(source))
            self.assertEqual(response.read(), b"")
            client.request("GET", media_route, headers={"Range": "bytes=999999999-"})
            response = client.getresponse()
            self.assertEqual(response.status, 416)
            response.read()
            client.request("GET", "/thumb/" + str(rows[0]["id"]))
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.read().startswith(b"\xff\xd8"))
            client.request("GET", "/watch/" + str(rows[0]["id"]))
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b"#t=", response.read())
            client.request(
                "GET", "/clip/" + str(rows[0]["id"]), headers={"Range": "bytes=0-31"}
            )
            response = client.getresponse()
            self.assertEqual(response.status, 206)
            self.assertIn(b"ftyp", response.read())
            client.request(
                "GET",
                "/clip/" + str(rows[0]["id"]) + "?format=webm",
                headers={"Range": "bytes=0-31"},
            )
            response = client.getresponse()
            self.assertEqual(response.status, 206)
            self.assertTrue(response.read().startswith(b"\x1a\x45\xdf\xa3"))
            client.request("GET", "/media/../../etc/passwd")
            response = client.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
            client.request("GET", "/api/status", headers={"Host": "attacker.example"})
            response = client.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_punctuation_query_is_safe(self):
        self.assertEqual(search(self.db, '" OR * :'), [])
        self.assertEqual(search(self.db, '" : *'), [])

    def test_delayed_audio_retains_source_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "delayed.mp4"
            run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(self.paths[1]),
                    "-itsoffset",
                    "2",
                    "-i",
                    str(self.paths[1]),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c",
                    "copy",
                    "-t",
                    "8",
                    str(source),
                ]
            )
            wav = Path(temp) / "audio.wav"
            extract_audio(source, wav)
            audio, _ = audio_features(wav)
            self.assertTrue(
                all(row["dbfs"] < -50 for row in audio if row["end"] <= 1.5)
            )
            self.assertTrue(any(row["dbfs"] > -40 for row in audio if row["start"] > 2))

    def test_batch_continues_after_one_corrupt_video(self):
        with tempfile.TemporaryDirectory() as temp:
            broken = Path(temp) / "broken.mp4"
            broken.write_bytes(b"not a video")
            result = ingest([broken, self.paths[1]], Path(temp) / "library.sqlite3")
            self.assertEqual(result["errors"], 1)
            self.assertEqual(
                {r["status"] for r in result["results"]}, {"error", "indexed"}
            )

    def test_semantic_query_uses_matching_model_only(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "semantic.sqlite3"
            shutil.copy(self.db, copied)
            with connect(copied) as conn, conn:
                row_id = conn.execute(
                    "SELECT id FROM segments ORDER BY id LIMIT 1"
                ).fetchone()[0]
                conn.execute(
                    "UPDATE segments SET embedding='[1,0]',model_key='test-model' WHERE id=?",
                    (row_id,),
                )
            semantic = types.SimpleNamespace(encode=lambda texts, query=False: [[1, 0]])
            rows = search(
                copied,
                "ein verwandtes Thema",
                "topic",
                semantic=semantic,
                semantic_key="test-model",
            )
            self.assertEqual(rows[0]["similarity"], 1)
            with self.assertRaisesRegex(ValueError, "No embeddings match"):
                search(
                    copied,
                    "Thema",
                    "topic",
                    semantic=semantic,
                    semantic_key="different-model",
                )
            with self.assertRaisesRegex(ValueError, "Topic search needs"):
                search(copied, "Thema", "topic")

    def test_sidecar_change_replaces_fts_and_invalid_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / self.paths[1].name
            shutil.copy(self.paths[1], source)
            sidecar = source.with_suffix(".json")
            shutil.copy(self.paths[1].with_suffix(".json"), sidecar)
            db = Path(temp) / "catalog.sqlite3"
            self.assertEqual(ingest([source], db)["errors"], 0)
            sidecar.write_text(
                json.dumps(
                    {"segments": [{"start": 1, "end": 3, "text": "NeuerSuchbegriff"}]}
                )
            )
            self.assertEqual(ingest([source], db)["errors"], 0)
            self.assertTrue(search(db, "NeuerSuchbegriff"))
            self.assertFalse(search(db, "Automatisierung"))
            before = status(db)["segments"]
            sidecar.write_text('{"segments":[{"start":-1,"end":50,"text":"broken"}]}')
            self.assertEqual(ingest([source], db)["errors"], 1)
            self.assertEqual(status(db)["segments"], before)
            self.assertTrue(search(db, "NeuerSuchbegriff"))
            with connect(db) as conn:
                self.assertEqual(
                    conn.execute("SELECT count(*) FROM search_fts").fetchone()[0],
                    before,
                )

    def test_no_audio_and_no_transcript_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "silent.mp4"
            run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(self.paths[1]),
                    "-an",
                    "-c:v",
                    "copy",
                    str(source),
                ]
            )
            db = Path(temp) / "catalog.sqlite3"
            result = ingest([source], db)
            self.assertEqual(result["errors"], 0)
            warnings = result["results"][0]["warnings"]
            self.assertTrue(any("No audio" in w for w in warnings))
            self.assertTrue(any("No transcript" in w for w in warnings))
            self.assertFalse(search(db, "lachen", "event"))


class TranscriptAndAdapters(unittest.TestCase):
    def test_whisper_cpp_milliseconds_and_srt_vtt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpp = root / "cpp.json"
            cpp.write_text(
                json.dumps(
                    {
                        "transcription": [
                            {
                                "offsets": {"from": 1200, "to": 2400},
                                "text": "Hallo Welt",
                            }
                        ]
                    }
                )
            )
            self.assertEqual(load_transcript(cpp, 10)[0]["start"], 1.2)
            for extension, text in (
                ("srt", "1\n00:00:01,200 --> 00:00:02,400\nHallo Welt\n"),
                (
                    "vtt",
                    "WEBVTT\n\ncue-1\n00:01.200 --> 00:02.400 align:start\nHallo Welt\n",
                ),
            ):
                path = root / ("sub." + extension)
                path.write_text(text)
                self.assertEqual(load_transcript(path, 10)[0]["end"], 2.4)

    def test_parakeet_local_reuse_chunking_and_token_provenance(self):
        token = types.SimpleNamespace(start=1.1, end=1.4, text="Hallo")
        aligned = types.SimpleNamespace(
            sentences=[
                types.SimpleNamespace(start=1.1, end=1.4, text="Hallo", tokens=[token])
            ]
        )
        model = types.SimpleNamespace(transcribe=mock.Mock(return_value=aligned))
        loader = mock.Mock(return_value=model)
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.dict(
                "sys.modules",
                {"parakeet_mlx": types.SimpleNamespace(from_pretrained=loader)},
            ),
            mock.patch("media_index.models.platform.system", return_value="Darwin"),
            mock.patch("media_index.models.platform.machine", return_value="arm64"),
        ):
            speech = SpeechModel("parakeet", temp)
            first = speech.transcribe(Path(temp) / "audio.wav", 10)
            speech.transcribe(Path(temp) / "audio.wav", 10)
            loader.assert_called_once_with(str(Path(temp).resolve()))
            self.assertEqual(model.transcribe.call_count, 2)
            self.assertEqual(model.transcribe.call_args.kwargs["chunk_duration"], 120)
            self.assertEqual(first[0]["tokens"][0]["start"], 1.1)
            self.assertFalse(first[0]["words"])
            self.assertTrue(first[0]["source"].startswith("parakeet:"))

    def test_local_models_and_invalid_intervals_fail_clearly(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            local_model("not-a-downloaded-model")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            for start, end in ((-1, 1), (2, 1), (float("nan"), 4), (10.1, 10.2)):
                path.write_text(
                    json.dumps(
                        {"segments": [{"start": start, "end": end, "text": "bad"}]}
                    )
                )
                with self.assertRaises(ValueError):
                    load_transcript(path, 10)

    def test_model_config_change_invalidates_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text('{"a":1}')
            before = model_key(temp)
            path.write_text('{"a":2}')
            self.assertNotEqual(model_key(temp), before)


if __name__ == "__main__":
    unittest.main()
