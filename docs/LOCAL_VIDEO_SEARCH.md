# Lokale Videosuche und Highlight-Erkennung

Der Prototyp erschließt einen Videobestand über mehrere Dateien hinweg. Originale bleiben unverändert.
Die Bearbeitungsfunktionen der Shell bleiben verfügbar; `hook` und `shorts` behalten ihren bisherigen Audioenergie-Algorithmus.
Die neue multimodale Auswahl erfolgt mit `index`, `search`, `highlights`, `review` und `export-highlights`.

## In zwei Minuten ausprobieren

Voraussetzung: Python 3.12+, `ffmpeg`, `ffprobe`, SQLite mit FTS5. Der Basispfad braucht keine Python-Pakete.

```bash
source media-edit.sh
me_library doctor
python3 examples/local_search_demo.py --out demo-output
me_review --db demo-output/library.sqlite3
```

Öffne `http://127.0.0.1:8765`. Suche nach `Automatisierung`, wähle „Lachen“ oder sieh die Highlights an.
Unter `demo-output/review/review.html` liegen auch drei abspielbare Vorschauen.
Die Demo erzeugt zwei Videos mit Testbildern/Tönen sowie ausdrücklich synthetischen Transkripten und Ereignissen.
Sie prüft die Verarbeitungskette; sie ist kein Nachweis für Spracherkennung oder Lacherkennung.

Ein zusätzlicher Inferenz-Smoke-Test erzeugt englische Sprache über den FFmpeg-Filter `flite`, erkennt diese mit bereits lokalen
ASR-Gewichten und prüft eine semantische Frage gegen zwei Videos. Er lädt selbst keine Modelle herunter:

```bash
python3 examples/local_asr_smoke.py --model-path /lokal/faster-whisper-tiny \
  --semantic-model /lokal/all-MiniLM-L6-v2
```

Hierfür sind die Extras `whisper` und `semantic`, die vorab bereitgestellten Modelle und FFmpeg mit `flite` erforderlich.
Der Test ist tatsächlich mit faster-whisper ausgeführt worden; auch er ist kein Benchmark für deutsches Alltagsmaterial.

Ohne Shell geht derselbe Weg über `python3 -m media_index …` oder nach Installation über `media-index …`.
Die Datenbankoption steht nach dem Unterbefehl: `python3 -m media_index search "Hotel" --db /pfad/library.sqlite3`.

## Mit eigenen Videos beginnen

```bash
export ME_INDEX_DB="$HOME/.media-edit/library.sqlite3"
me_index "/Volumes/Videos" --asr none
me_search "Hotel"
me_search "applaus" --mode event
me_highlights --limit 10
me_review
me_export_highlights --out "$HOME/Movies/Highlight-Review" --limit 10 --clips
```

Ohne eingerichtete Modelle verwendet der Index vorhandene Transkripte und misst Audio/Bild.
Fehlende Transkripte und Klassifikatoren werden pro Video ausgewiesen. `status` zeigt auch nicht erreichbare oder veränderte Quelldateien.
Verzeichnisse werden rekursiv durchsucht; versteckte Unterverzeichnisse werden ausgelassen. Symlink-Ziele werden dedupliziert.

Transkript-Namen: `aufnahme.mp4.json` oder `aufnahme.json`, alternativ `.srt` und `.vtt`.
Unterstützt werden OpenAI/MLX/faster-whisper-JSON (`segments` mit Sekunden), whisper.cpp-JSON (`transcription.offsets` in Millisekunden),
SRT und WebVTT. Vorhandene Wort- und Token-Zeitmarken bleiben erhalten. Ohne Wortmarken zeigt die Suche die ursprüngliche Satzspanne.
`--transcript /pfad/datei.json` überschreibt die automatische Auswahl für genau ein Video.

Optionale Ereignisdatei: `aufnahme.events.json` oder `aufnahme.mp4.events.json`:

```json
{"events":[{"start":12.4,"end":14.1,"label":"laughter","confidence":0.86,"source":"own-local-model:v1"}]}
```

Importierte Annotationen werden als `import:…` gekennzeichnet. Gib den tatsächlichen Ursprung an; der Import validiert Zeitspannen und Zahlen,
beweist aber nicht die Qualität einer Annotation. Fehlerhafte Eingaben ersetzen keinen bereits gültigen Indexstand.

## Apple Silicon: Parakeet v3 und optionale Modelle

Die geprüften Paketstände sind in `pyproject.toml` und `uv.lock` festgehalten. Installiere nur die benötigten Extras:

```bash
# Einmalig im Checkout; uv und FFmpeg müssen vorhanden sein.
uv sync --frozen --extra parakeet --extra semantic --extra multimodal --extra scenes
source media-edit.sh
export ME_INDEX_PYTHON="$PWD/.venv/bin/python"
me_library doctor
```

Modellgewichte werden bewusst separat bereitgestellt. Dieser explizite Download ist der Online-Schritt:

```bash
.venv/bin/hf download mlx-community/parakeet-tdt-0.6b-v3 --local-dir "$HOME/Models/parakeet-v3"
.venv/bin/hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir "$HOME/Models/multilingual-search"
.venv/bin/hf download MIT/ast-finetuned-audioset-10-10-0.4593 --local-dir "$HOME/Models/audioset-ast"
.venv/bin/hf download openai/clip-vit-base-patch32 --local-dir "$HOME/Models/clip"
```

Für reproduzierbare Bestände kann `hf download` zusätzlich mit einer festgehaltenen Modellrevision verwendet werden.
Die Gewichte sind keine Bestandteile dieses Repositories. Anschließend arbeitet der Prototyp mit lokalen Pfaden und ohne Modell-Downloads:

```bash
me_index "/Volumes/Videos" \
  --asr parakeet --model-path "$HOME/Models/parakeet-v3" \
  --semantic-model "$HOME/Models/multilingual-search" \
  --audio-model "$HOME/Models/audioset-ast" \
  --vision-model "$HOME/Models/clip" \
  --scene-detector adaptive --profile event --window 20 --stride 8

me_search "Wie verbessern wir den Service für Hotelgäste?" --mode topic \
  --semantic-model "$HOME/Models/multilingual-search"
me_review --semantic-model "$HOME/Models/multilingual-search"
```

Parakeet v3 erkennt die Sprache automatisch; `--language` wird deshalb bei diesem Backend nicht akzeptiert.
Die Aufnahme wird mit 120-Sekunden-Chunks und 15 Sekunden Überlappung transkribiert. Tokenmarken werden als Tokenmarken gespeichert,
nicht als künstlich behauptete Wortgrenzen. Kein NeMo/CUDA-Stack ist dafür auf dem Mac nötig.

Alternativen für ASR:

| Backend | Installation | Lokales Modell | Verhalten |
|---|---|---|---|
| `parakeet` | Extra `parakeet` | MLX-Parakeet-v3-Verzeichnis | Apple Silicon, automatische Sprache, Chunking |
| `mlx` | Extra `mlx` | MLX-Whisper-Verzeichnis | Apple Silicon, `--language de`, Wortmarken |
| `whisper-cpp` | `brew install whisper-cpp` | ggml-`.bin` | Vorhandenen Metal-Pfad nutzen, Satzmarken |
| `faster-whisper` | Extra `whisper` | CTranslate2-Whisper-Verzeichnis | CPU/int8, VAD, Wortmarken; kein MPS-Backend |
| `none` | kein Extra | vorhandene Sidecars | Keine ASR-Inferenz |

`auto` bevorzugt Sidecars und nutzt ansonsten whisper.cpp nur bei explizitem Modellpfad und vorhandenem `whisper-cli`.
Für Parakeet/MLX/faster-whisper wähle das Backend ausdrücklich. Eine explizite ASR-Auswahl erzeugt neue Transkripte;
`--transcript` hat Vorrang. `--device auto|cpu|mps` betrifft die Torch-Adapter für Semantik, Audio und CLIP.
MLX und whisper.cpp steuern ihre Beschleunigung selbst. Auf Intel/Linux bleiben Core und CPU-Whisper nutzbar.

Für kategorische Sprach-Emotionen kann zusätzlich `--emotion-model /lokales/modell` verwendet werden.
Ein kompatibles Ausgangsmodell ist `superb/wav2vec2-base-superb-er`; es benötigt ein separat bereitgestelltes Verzeichnis.
Dieses englisch trainierte Modell ist keine belegte Lösung für deutsche Dialekte. Der Adapter verwendet Softmax für Emotionsklassen,
während AudioSet-Ereignisse unabhängige Sigmoid-Scores erhalten. Dimensionsmodelle für Valenz/Arousal werden nicht als Klassen ausgegeben.

## Sinnvolle Arbeitsabläufe

| Einsatz | Einstellungen und Suche | Ergebnis |
|---|---|---|
| Kundeninterviews / Testimonials | `--profile interview --window 30 --stride 10`; Themen wie „konkrete Zeitersparnis“ | Aussagen mit Kontext finden, inhaltlich prüfen, Schnittmarken übergeben |
| Events / Ecstatic Dance | `--profile event --window 20 --stride 8`; `lachen`, `jubel`, `applaus`, `bewegung` | Reaktionen und visuell aktive Momente vorsortieren; Musik/Kameraeffekte prüfen |
| Vorträge / internes Wissen | Semantikmodell; nach einer Frage oder einem Begriff suchen | Fundstelle samt ursprünglicher Satz-/Wortzeit aufrufen |
| Familien- / Hundeaufnahmen | AudioSet + optional CLIP; `lachen`, `hund`, `landschaft` | Kandidaten über Ton und sichtbare Inhalte erschließen; Bildtags verifizieren |
| Social-Clips | Top-Kandidaten als MP4 exportieren; anschließend bei Bedarf `me_shorts clip.mp4` | Bestehende Schnitt-/Caption-Pipeline für bewusst ausgewählte Ausgangsclips nutzen |

Die Profile ändern die nachvollziehbaren Gewichte. Sie erkennen weder eine gute Geschichte noch die sachliche Richtigkeit einer Aussage.
Bei Interviews sind vollständige Aussagen wichtiger als einzelne Reaktionswörter. Bei Events sind Kamera-Schwenks und laute Musik häufige Störsignale.

## Wie die Bewertung entsteht

Kandidaten entstehen aus einem Zeitraster sowie Satz-/Ereignisankern mit 1,5 Sekunden Vorlauf. Die Standardlänge ist 12 Sekunden,
der Rasterschritt 6 Sekunden. Sehr ähnliche überlappende Ausschnitte desselben Videos werden bei der Ausgabe unterdrückt.

| Signal | Balanced | Interview | Event |
|---|---:|---:|---:|
| Sprachdichte + Reaktionswörter | 25 % | 55 % | 10 % |
| Erkannte/importierte Reaktionsereignisse | 30 % | 20 % | 40 % |
| RMS-Abweichung vom Median desselben Videos | 15 % | 5 % | 20 % |
| Szenenwechsel | 10 % | 5 % | 10 % |
| Bilddifferenz / Bewegung | 10 % | 5 % | 15 % |
| Kontrast / Helligkeitsindikatoren | 10 % | 10 % | 5 % |

Stille und sehr dunkle Bilder verursachen Abzüge. Jeder Kandidat enthält Komponenten, Abzüge, Messwerte, Texte, Ereignisherkunft und Gründe.
Ein Wert von 75/100 bedeutet keine 75-prozentige Qualitäts- oder Viralitätswahrscheinlichkeit.
„Wow“, „!“ und ähnliche Textstellen sind Hinweise aus dem Transkript; hohe Lautstärke ist ein eigenes Messsignal.

Audio-Messwerte und Basisbilder haben 0,5-Sekunden-Auflösung; der optionale adaptive Szenendetektor arbeitet auf Video-Frames.
AudioSet-/Emotionsklassifikation verwendet 5-Sekunden-Fenster mit 2,5 Sekunden Schritt und liefert keine exakten Ereignis-Onsets.
CLIP untersucht alle zwei Sekunden ein verkleinertes Bild und vergleicht vier Prompts: Person, Hund, Landschaft, Bildschirm.
Diese relativen Ähnlichkeiten sind Inhaltsvorschläge und keine kalibrierten Objekt-/Emotionserkennungswahrscheinlichkeiten.

## Vorschau und Übergabe an Schnittprogramme

Die Oberfläche bietet Wort-, Themen- und Ereignissuche, Vorschaubilder, Originalwiedergabe und kopierbare Schnittbefehle.
Browser-untaugliche Originalformate erhalten bei Wiedergabefehlern eine lokal erzeugte VP9/Opus-WebM-Vorschau.
MP4-Vorschauen bleiben für Export und den separaten Vorschau-Endpunkt verfügbar.
Der Server bindet ausschließlich an `127.0.0.1`, unterstützt HTTP-Byte-Ranges und gibt nur registrierte Medien aus.
Ein Link wie `/watch/42?at=125.400` startet an der Fundstelle; `/watch/42` zeigt den Kontext des Kandidaten.
Ohne laufenden Server können exportierte MP4-Vorschauen direkt geöffnet werden.

Der Export erzeugt:

- `candidates.json`: Quellen, Sekundenmarken, Gründe, Komponenten, Herkunft und Abspiel-/FFmpeg-/mpv-Verweise.
- `video-ID-losslesscut.csv`: je Quellvideo eine CSV mit Start, Ende und Name, in Sekunden.
- `segment-ID.jpg`, optional `segment-ID.mp4` sowie `review.html`.

Öffne in LosslessCut zuerst das zugehörige Original und importiere dessen CSV als Sekunden-Schnittliste.
Die CSV enthält keine eingebetteten Videodaten. FFmpeg-Befehle kodieren neu und schneiden dadurch genauer als ein Stream-Copy an Keyframes.
Resolve-/Kdenlive-/Premiere-Timelines oder direkte Projektdateien werden in diesem Prototyp nicht erzeugt.

## Betrieb, Wiederaufnahme und Grenzen

```bash
me_library status
me_index "/Volumes/Videos"                 # identische Signatur: Analyse überspringen
me_index "/Volumes/Videos" --force         # nach eigener Modell-/Algorithmusänderung neu analysieren
me_library preview 42 --out /tmp/moment.mp4 --clip
me_library remove 7                       # nur Video-ID 7 aus dem Katalog entfernen
```

Verwende bei der Wiederaufnahme dieselben Modell- und Analyseoptionen. SHA-256 über Quelldatei und Sidecars, Paketversionen,
Modellsignaturen und Optionen bestimmen den Cache. Ein warmes Einlesen liest weiterhin die Quelldatei zum Hashen.
Modellkonfigurationen werden gehasht; für große Gewichte werden Größe und Änderungszeit erfasst. Bei unveränderten Dateimetadaten manuell ersetzter Gewichte ist `--force` nötig.
Jedes Video wird separat atomar gespeichert. Ein Batch verarbeitet weitere Dateien nach einem Einzelfehler und endet bei Fehlern mit Exitcode 1.
Ein Abbruch vor dem Commit lässt den vorherigen gültigen Stand erhalten. Verschobene Videos werden als neue Pfade behandelt;
alte Einträge können mit `remove` entfernt werden. Direktlinks auf neu erzeugte Segment-IDs ändern sich nach einer Reindexierung.

FFmpeg verarbeitet Bilder als begrenzten Stream; Audio wird temporär als Mono-16-kHz-WAV gehalten und danach entfernt.
Zeitliche Auswahl benutzt sortierte Zeitachsen. Semantische Suche vergleicht im Prototyp alle passenden Embeddings;
für sehr große Bestände ist ein gemessener ANN-/Vektorindex der nächste Schritt. Quelldateien, DB, lokale Modelle und Vorschaucache
belegen Speicher auf dem eigenen Gerät. Der Cache `previews/` neben der Datenbank kann bei gestopptem Server neu aufgebaut werden.
Die Anwendung synchronisiert oder verschlüsselt diese Dateien nicht selbst.

Welche Qualitätsprüfungen noch echte Aufnahmen und einen Apple-Silicon-Mac benötigen, steht im [Validierungsbericht](VALIDATION_LOCAL_SEARCH.md).
