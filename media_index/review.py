"""Local previews, editing interchange, and a loopback-only review interface."""

import csv
import html
import json
import mimetypes
import re
import sqlite3
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .analysis import run
from .store import connect, get_segment, highlights, search, status


def preview(row, output, clip=False):
    output = Path(output).expanduser().resolve()
    source = Path(row["path"])
    if output == source.resolve():
        raise ValueError("Preview output must differ from the source video")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preview-", dir=output.parent) as work:
        temp = Path(work) / output.name
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            str(row["start"]),
            "-i",
            str(source),
        ]
        if clip:
            cmd += [
                "-t",
                str(row["end"] - row["start"]),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "scale=640:-2",
            ]
            if output.suffix.lower() == ".webm":
                cmd += [
                    "-c:v",
                    "libvpx-vp9",
                    "-deadline",
                    "realtime",
                    "-cpu-used",
                    "6",
                    "-b:v",
                    "0",
                    "-crf",
                    "35",
                    "-c:a",
                    "libopus",
                ]
            else:
                cmd += [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "25",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                ]
        else:
            cmd += ["-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "3"]
        run(cmd + [str(temp)])
        if not temp.is_file() or not temp.stat().st_size:
            raise RuntimeError(
                "FFmpeg produced no preview; check the requested timestamps"
            )
        temp.replace(output)
    return output


def export(
    db, output, limit=10, clips=False, base_url="http://127.0.0.1:8765", video_id=None
):
    output = Path(output).expanduser().resolve()
    rows = highlights(db, limit, base_url, video_id)
    output.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["video_id"], []).append(row)
        name = f"segment-{row['id']}"
        preview(row, output / (name + ".jpg"))
        row["preview_image"] = name + ".jpg"
        if clips:
            preview(row, output / (name + ".mp4"), clip=True)
            row["preview_clip"] = name + ".mp4"
    for source_id, segments in grouped.items():
        # LosslessCut: Import project > CSV > seconds. A separate CSV per source.
        with (output / f"video-{source_id}-losslesscut.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            for row in sorted(segments, key=lambda r: r["start"]):
                writer.writerow(
                    [
                        f"{row['start']:.3f}",
                        f"{row['end']:.3f}",
                        f"Highlight {row['id']}: {row['score']:.1f}/100",
                    ]
                )
    payload = {
        "format": "media-edit-highlights-v1",
        "score_is_probability": False,
        "videos": {
            str(key): {
                "path": values[0]["path"],
                "losslesscut_csv": f"video-{key}-losslesscut.csv",
            }
            for key, values in grouped.items()
        },
        "candidates": rows,
    }
    (output / "candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cards = []
    for row in rows:
        playback = row.get("preview_clip", row["play_url"])
        media = (
            f'<video controls preload="none" poster="{row["preview_image"]}" src="{playback}"></video>'
            if clips
            else f'<a href="{html.escape(playback, quote=True)}"><img src="{row["preview_image"]}" alt="Preview"></a>'
        )
        cards.append(
            f"<article>{media}<h2>{html.escape(Path(row['path']).name)}</h2>"
            f"<p>{row['start']:.3f}–{row['end']:.3f}s · {row['score']:.1f}/100</p>"
            f"<p>{html.escape(row['text'])}</p><ul>"
            + "".join(f"<li>{html.escape(reason)}</li>" for reason in row["reasons"])
            + f'</ul><p><a href="{html.escape(row["play_url"], quote=True)}">Open source position</a></p>'
            f"<details><summary>Editing command</summary><pre>{html.escape(row['ffmpeg_command'])}</pre></details></article>"
        )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Local highlight review</title><style>body{background:#10131a;color:#e9edf5;font:16px system-ui;"
        "max-width:1000px;margin:40px auto;padding:0 24px}article{background:#1c2230;padding:24px;margin:24px 0;"
        "border-radius:16px}video,img{width:100%;max-height:400px;object-fit:contain}a{color:#9bc2ff}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:8px 0}</style><h1>Local highlight review</h1>"
        "<p>Candidate scores are heuristics. Review meaning, timing and model evidence before editing.</p>"
        + "".join(cards),
        encoding="utf-8",
    )
    return {"output": str(output), "candidates": len(rows), "clips": clips}


PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>media-edit · Local Library</title><style>
:root{color-scheme:dark;--bg:#0d1018;--panel:#171d29;--line:#2b3448;--muted:#9aa8bf;--text:#edf2fb;--accent:#abd8b5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}
header{padding:26px 5vw;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
header strong{font-size:20px;letter-spacing:-.6px}header span{font-size:12px;color:var(--accent)}main{max-width:1500px;margin:auto;padding:34px 5vw}
h1{font-size:clamp(28px,4vw,42px);letter-spacing:-1.2px;line-height:1.1;margin:4px 0 14px}p{line-height:1.55}.muted{color:var(--muted)}
.toolbar{display:flex;gap:10px;margin:24px 0 12px;flex-wrap:wrap}input,select,button{font:inherit;border:1px solid var(--line);border-radius:9px;padding:12px;background:var(--panel);color:var(--text)}
input{min-width:180px;flex:1}button{cursor:pointer}button:hover,button:focus-visible{border-color:var(--accent)}.primary{background:var(--accent);color:#14221c;font-weight:650}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 24px}.chips button{font-size:12px;padding:7px 12px;color:var(--muted)}
.layout{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(300px,1fr);gap:24px;align-items:start}.stage{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden;position:sticky;top:20px}
video{width:100%;background:#05070b;max-height:56vh}.detail{padding:24px}.detail h2{font-size:19px;margin-top:0;overflow-wrap:anywhere}.detail ul{padding-left:20px;color:var(--muted);line-height:1.6}
#cards{display:grid;gap:12px}.card{display:grid;grid-template-columns:126px minmax(0,1fr);gap:16px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;cursor:pointer;text-align:left;width:100%}
.card[aria-pressed="true"]{border-color:var(--accent)}.card img{width:126px;height:84px;object-fit:cover;border-radius:7px;background:#222a39}.card h3{margin:0 0 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:14px}
.card p{font-size:12px;margin:6px 0 0;color:var(--muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.score{color:var(--accent);font-size:12px}.time{font-variant-numeric:tabular-nums;font-size:12px}
#message{min-height:25px;color:var(--muted);margin:16px 0}#transcript{white-space:pre-wrap}.foot{font-size:12px;margin-top:30px;color:var(--muted)}a{color:var(--accent)}
@media(max-width:820px){.layout{grid-template-columns:1fr}.stage{position:static}.card{grid-template-columns:100px 1fr}.card img{width:100px}header{padding:20px 5vw}}
</style></head><body><header><strong>media-edit <span style="color:var(--muted);font-weight:400">/ library</span></strong><span>● Lokal auf diesem Gerät</span></header>
<main><p class="muted" id="stats">Lokaler Videobestand</p><h1>Den richtigen Moment finden.</h1><p class="muted">Durchsuche Aussagen, Themen und Ereignisse. Prüfe Highlights direkt an der Quelle.</p>
<form class="toolbar" id="search"><input id="query" aria-label="Suchbegriff" placeholder="Was suchst du in deinen Videos?" autocomplete="off"><select id="mode" aria-label="Suchmodus"><option value="words">Wörter</option><option value="topic">Themen</option><option value="event">Ereignisse & Bildmerkmale</option></select><button class="primary">Suchen</button><button type="button" id="all">Highlights</button></form>
<div class="chips"><button type="button" data-event="lachen">Lachen</button><button type="button" data-event="applaus">Applaus</button><button type="button" data-event="ausruf">Ausrufe</button><button type="button" data-event="bewegung">Bewegung</button><button type="button" data-event="hund">Hund</button></div>
<p id="message" role="status" aria-live="polite"></p><div class="layout"><section class="stage"><video id="player" controls preload="metadata"></video><div class="detail"><h2 id="title">Wähle einen Moment</h2><p class="muted" id="position"></p><p id="transcript"></p><ul id="reasons"></ul><button type="button" id="copy" hidden>Schnittbefehl kopieren</button> <a id="direct" hidden target="_blank" rel="noreferrer">Direktlink</a></div></section><section id="cards" aria-label="Ergebnisse"></section></div>
<p class="foot">Bewertungen sind nachvollziehbare Vorschläge. Ereignis- und Modellhinweise bitte vor dem Schnitt prüfen. Themen benötigen ein lokal eingerichtetes Sprachmodell für die Suche.</p></main>
<script>
const $=id=>document.getElementById(id);let current=null;let request=0;
const fmt=t=>new Date(t*1000).toISOString().slice(11,23);
$('player').addEventListener('timeupdate',()=>{if(current&&$('player').currentTime>=($('player').dataset.fallback==='1'?current.end-current.start:current.end))$('player').pause()});
function select(r){current=r;document.querySelectorAll(".card").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.id===String(r.id))));$('title').textContent=r.path.split('/').pop();$('position').textContent=`${fmt(r.start)} → ${fmt(r.end)} · ${r.score.toFixed(1)}/100`;
$('transcript').textContent=r.text||'Kein Transkript in diesem Ausschnitt.';$('reasons').replaceChildren();for(const reason of r.reasons){const li=document.createElement('li');li.textContent=reason;$('reasons').append(li)}
$('player').dataset.fallback='0';$('player').src=`/media/${r.video_id}#t=${r.matched_start??r.start},${r.end}`;$('player').onloadedmetadata=()=>{$('player').currentTime=r.matched_start??r.start};$('copy').hidden=false;$('direct').hidden=false;$('direct').href=r.match_url||`/watch/${r.id}`;}
$('player').addEventListener('error',()=>{if(!current)return;if($('player').dataset.fallback==='0'){$('player').dataset.fallback='1';$('position').textContent+=' · Vorschau wird vorbereitet …';$('player').onloadedmetadata=()=>{$('player').currentTime=Math.max(0,(current.matched_start??current.start)-current.start);$('position').textContent=`${fmt(current.start)} → ${fmt(current.end)} · ${current.score.toFixed(1)}/100 · Vorschau`};$('player').src=`/clip/${current.id}?format=webm`}else $('message').textContent='Wiedergabe fehlgeschlagen. Prüfe die Quelle und die FFmpeg-Codecs.'});
async function load(url){const own=++request;$('message').textContent='Suche läuft …';try{const response=await fetch(url);const data=await response.json();if(own!==request)return;if(!response.ok)throw Error(data.error||'Suche fehlgeschlagen');
$('cards').replaceChildren();$('message').textContent=data.length?`${data.length} passende Momente`:'Keine Treffer. Versuche ein anderes Wort oder ein anderes Ereignis.';
for(const r of data){const b=document.createElement('button');b.type='button';b.className='card';b.dataset.id=r.id;b.setAttribute('aria-pressed','false');const img=document.createElement('img');img.src=`/thumb/${r.id}`;img.loading='lazy';img.alt='Vorschau';const d=document.createElement('div');const h=document.createElement('h3');h.textContent=r.path.split('/').pop();const score=document.createElement('span');score.className='score';score.textContent=`${r.score.toFixed(1)}/100${r.similarity!==undefined?' · Ähnlichkeit '+r.similarity.toFixed(2):''}`;const t=document.createElement('p');t.className='time';t.textContent=`${fmt(r.start)} → ${fmt(r.end)}`;const p=document.createElement('p');p.textContent=r.text||r.tags.join(' · ');d.append(h,score,t,p);b.append(img,d);b.onclick=()=>select(r);$('cards').append(b)}if(data.length)select(data[0]);else{current=null;$('player').removeAttribute('src');$('player').load();$('title').textContent='Keine passenden Momente';$('position').textContent='';$('transcript').textContent='';$('reasons').replaceChildren();$('copy').hidden=true;$('direct').hidden=true;}
}catch(e){if(own===request)$('message').textContent=e.message}}
$('search').onsubmit=e=>{e.preventDefault();load('/api/search?'+new URLSearchParams({q:$('query').value,mode:$('mode').value,limit:20}))};$('all').onclick=()=>{$('query').value='';$('mode').value='words';load('/api/highlights?limit=20')};
document.querySelectorAll('[data-event]').forEach(b=>b.onclick=()=>{$('query').value=b.dataset.event;$('mode').value='event';$('search').requestSubmit()});
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText(current.ffmpeg_command);$('message').textContent='Schnittbefehl kopiert.'}catch{$('message').textContent=current.ffmpeg_command}};
fetch('/api/status').then(r=>r.json()).then(s=>{const stale=s.videos.filter(v=>v.changed_since_index||!v.available).length;$('stats').textContent=`${s.videos.length} Videos · ${s.segments} Kandidaten · Local-First${stale?' · '+stale+' Quellen prüfen / neu indexieren':''}`}).catch(()=>{});load('/api/highlights?limit=20');
</script></body></html>"""


def make_server(db, port=8765, semantic=None, semantic_key=None):
    db = Path(db).expanduser().resolve()
    model_lock = threading.Lock()
    thumb_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        @property
        def base_url(self):
            return f"http://127.0.0.1:{self.server.server_port}"

        def send(self, content, mime="application/json; charset=utf-8", code=200):
            data = content.encode("utf-8") if isinstance(content, str) else content
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def stream_file(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), end) if match[2] else end
                else:
                    start = max(0, size - int(match[2]))
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if range_header else 200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("X-Content-Type-Options", "nosniff")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            # Constrain Host to loopback; no arbitrary file routes or cross-origin API.
            if self.headers.get("Host") not in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            ):
                self.send('{"error":"Use the loopback URL"}', code=403)
                return
            route = urlparse(self.path)
            params = parse_qs(route.query)
            try:
                limit = min(100, max(1, int(params.get("limit", [10])[0])))
                if route.path == "/":
                    self.send(PAGE, "text/html; charset=utf-8")
                elif route.path == "/api/status":
                    self.send(json.dumps(status(db), ensure_ascii=False))
                elif route.path == "/api/highlights":
                    self.send(
                        json.dumps(
                            highlights(db, limit, self.base_url), ensure_ascii=False
                        )
                    )
                elif route.path == "/api/search":
                    mode = params.get("mode", ["words"])[0]
                    if mode not in ("words", "topic", "event"):
                        raise ValueError("Unknown search mode")
                    with model_lock:
                        rows = search(
                            db,
                            params.get("q", [""])[0],
                            mode,
                            limit,
                            semantic,
                            semantic_key,
                            self.base_url,
                        )
                    self.send(json.dumps(rows, ensure_ascii=False))
                elif re.fullmatch(r"/media/\d+", route.path):
                    with connect(db) as conn:
                        row = conn.execute(
                            "SELECT path FROM videos WHERE id=?",
                            (int(route.path.split("/")[-1]),),
                        ).fetchone()
                    if not row:
                        raise FileNotFoundError("Video not found")
                    self.stream_file(Path(row["path"]))
                elif re.fullmatch(r"/(thumb|clip)/\d+", route.path):
                    row = get_segment(db, int(route.path.split("/")[-1]), self.base_url)
                    clip = route.path.startswith("/clip/")
                    clip_format = params.get("format", ["mp4"])[0]
                    if clip_format not in ("mp4", "webm"):
                        raise ValueError("Preview format must be mp4 or webm")
                    thumb = (
                        db.parent
                        / "previews"
                        / row["fingerprint"]
                        / f"{row['id']}{'.' + clip_format if clip else '.jpg'}"
                    )
                    with thumb_lock:
                        if not thumb.exists():
                            preview(row, thumb, clip)
                    self.stream_file(thumb)
                elif re.fullmatch(r"/watch/\d+", route.path):
                    row = get_segment(db, int(route.path.split("/")[-1]), self.base_url)
                    import math

                    at = float(params.get("at", [row["start"]])[0])
                    if not math.isfinite(at) or not 0 <= at < row["end"]:
                        raise ValueError("Playback timestamp outside this candidate")
                    self.send(
                        '<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                        f'<title>{html.escape(Path(row["path"]).name)}</title><body style="background:#10131a;color:white;font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px">'
                        f"<h1>{html.escape(Path(row['path']).name)}</h1>"
                        f'<video id="v" style="width:100%" controls src="/media/{row["video_id"]}#t={at},{row["end"]}"></video>'
                        f"<p>{row['start']:.3f}–{row['end']:.3f}s · {html.escape(row['text'])}</p>"
                        f'<script>const v=document.getElementById("v");let fallback=false;v.onloadedmetadata=()=>v.currentTime={at};'
                        f"v.ontimeupdate=()=>{{if(v.currentTime>=(fallback?{row['end'] - row['start']}:{row['end']}))v.pause()}};"
                        f'v.onerror=()=>{{if(!fallback){{fallback=true;v.onloadedmetadata=()=>v.currentTime={max(0, at - row["start"])};v.src="/clip/{row["id"]}?format=webm"}}}};</script></body></html>',
                        "text/html; charset=utf-8",
                    )
                else:
                    self.send('{"error":"Not found"}', code=404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except FileNotFoundError:
                self.send(
                    '{"error":"Source unavailable; reconnect the drive or re-index its new path"}',
                    code=404,
                )
            except (ValueError, sqlite3.Error) as exc:
                self.send(json.dumps({"error": str(exc)}), code=400)
            except (RuntimeError, OSError) as exc:
                self.send(json.dumps({"error": str(exc)}), code=500)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
