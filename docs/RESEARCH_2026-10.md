# Technikstand und Integrationsentscheidungen · 18. Oktober 2026

Die frühere Awesome-Sammlung dokumentierte ihren Stand im Juni. Für diesen Prototyp wurden aktuelle Projektquellen,
Paketmetadaten von PyPI und Hersteller-Modellkarten erneut geprüft. Paketversionen sind keine Qualitätsrangliste.
Direkte optionale Abhängigkeiten stehen in `pyproject.toml`; transitive Versionen und Paket-Hashes in `uv.lock`.

## Integrierte Bausteine

| Baustein | Geprüfter Paketstand | Konkreter Nutzen / Entscheidung | Quelle |
|---|---|---|---|
| Parakeet-MLX | **0.5.3**, 01.10.2026 | Wahlbares ASR-Backend für Apple Silicon. Lokale v3-Gewichte, 120-s-Chunks / 15-s-Overlap, Tokenmarken. | [PyPI](https://pypi.org/project/parakeet-mlx/0.5.3/), [API](https://github.com/senstella/parakeet-mlx) |
| Parakeet-TDT-0.6B-v3 | Modellfamilie v3 | 25 europäische Sprachen mit automatischer Spracherkennung laut NVIDIA; für Deutsch v3 statt englischem v2 verwenden. Modellqualität auf eigenen Dialekten bleibt zu messen. | [NVIDIA-Modellkarte](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| MLX Whisper | **0.4.3** | Alternative auf Apple Silicon; vorhandene lokale MLX-Gewichte und Wortmarken. | [PyPI](https://pypi.org/project/mlx-whisper/0.4.3/), [MLX-API](https://github.com/ml-explore/mlx-examples/tree/main/whisper) |
| faster-whisper | **1.2.1** | CPU/int8-Alternative mit VAD und Wortmarken. Übernimmt bereits normalisierte PCM-Samples statt einer zweiten Mediendekodierung. | [PyPI](https://pypi.org/project/faster-whisper/1.2.1/), [Projekt](https://github.com/SYSTRAN/faster-whisper) |
| PySceneDetect | **0.7.2**, 12.10.2026 auf PyPI | Optionaler AdaptiveDetector liefert Frame-Grenzen; ohne Extra bleibt der günstige 2-fps-Detektor verfügbar. | [PyPI](https://pypi.org/project/scenedetect/0.7.2/), [Projekt](https://github.com/Breakthrough/PySceneDetect) |
| Sentence Transformers | **6.1.0**, 18.09.2026 | Semantische Suche mit separaten `encode_query`-/`encode_document`-Aufrufen; lokale Modell-Snapshots werden abgeglichen. | [PyPI](https://pypi.org/project/sentence-transformers/6.1.0/), [API](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) |
| Transformers | **5.20.1**, 16.10.2026 | Aktuelle lokale AST-, Emotions- und CLIP-Adapter. Keine automatische Cloud-Inferenz oder Remote-Code-Ausführung. | [PyPI](https://pypi.org/project/transformers/5.20.1/), [AST-Dokumentation](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer) |
| PyTorch | **2.14.1** | Torch-Modelle nutzen CPU oder MPS. Prüfung hier mit 2.14.1+cpu; kein gemessener Mac-Speedup behauptet. | [PyPI](https://pypi.org/project/torch/2.14.1/), [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) |
| NumPy / Pillow | **2.5.4 / 13.0.0** | Aktuelle Arrays/Bildverarbeitung für optionale Modelle. NumPy 2.5 erfordert Python 3.12; entsprechend gilt Python 3.12+ für das Paket. | [NumPy-PyPI](https://pypi.org/project/numpy/2.5.4/), [Pillow-PyPI](https://pypi.org/project/Pillow/13.0.0/) |
| SQLite FTS5 / FFmpeg | Laufzeitwerkzeuge | Persistente Wortsuche, Zeitachsen, Vorschau und Schnitt ohne zusätzliche Python-Pflichtpakete. Tatsächlich geprüft: FFmpeg 6.1.1, SQLite 3.53.1. | [FTS5](https://www.sqlite.org/fts5.html), [FFmpeg](https://ffmpeg.org/documentation.html) |

AudioSet-AST liefert akustische Klassen wie Lachen/Applaus. Der separate Emotionsadapter unterstützt kategorische 16-kHz-Klassifikatoren,
etwa [SUPERB ER](https://huggingface.co/superb/wav2vec2-base-superb-er). AudioSet erhält Sigmoid-Scores, exklusive Emotionsklassen Softmax.
Ein Sprach-Emotionslabel ist ein Modellhinweis; daraus folgt keine gesicherte innere Emotion einer Person.
CLIP wird über den PIL-Bildprozessor geladen, sodass keine zusätzliche TorchVision-Installation erforderlich ist.
Die Paket-Lösung verzichtet auch auf eine obligatorische TorchAudio-Abhängigkeit; der getestete AST-Pfad nutzt den verfügbaren Fallback.

## Weitere relevante Neuerungen und ihr nächster praktischer Einsatz

Diese Kandidaten sind recherchiert; sie sind **keine zusätzlich implementierten oder auf dem Mac vermessenen Backends**.

| Baustein / Neuerung | Praktischer Mehrwert | Nächste sinnvolle Integrationsentscheidung |
|---|---|---|
| [Qwen3-ASR + ForcedAligner](https://github.com/QwenLM/Qwen3-ASR) | Seit 26.06.2026 native Transformers-Unterstützung; separates Alignment kann Zeitmarken für Deutsch und andere Sprachen verbessern. | Erst gegen Parakeet/Whisper auf eigenem Material vergleichen: Fehlerrate, Timing, Speicher, Laufzeit. Danach ASR-/Alignment-Adapter ergänzen. |
| [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), 23.09.2026 | „Wer sprach wann?“; Hersteller beschreibt Offline/Streaming mit bis zu acht Sprechern. Nützlich für Interviews, Panels und Meetings. | Sprecher-Spannen von ASR trennen und als eigene Evidenz speichern. Mac-Laufzeit/Portierung zunächst nachweisen; kein stiller CUDA-Zwang. |
| [Qwen3 Embedding / Reranker](https://github.com/QwenLM/Qwen3-Embedding) | Mehrsprachige, instruktionsabhängige Retrieval-Modelle; interessant für „Kundeneinwände“, „Ergebnisbelege“ und thematische Archivsuche. | Ein kompatibles lokales SentenceTransformer-Snapshot kann bereits verwendet werden. Separate Reranker-Stufe und eigene Suchmetriken sind noch nicht implementiert. |
| [SigLIP 2](https://huggingface.co/blog/siglip2) | Mehrsprachige Bild/Text-Repräsentationen können freie deutsche Bildsuchen über vier feste CLIP-Tags hinaus ermöglichen. | Separate visuelle Embeddings und Suchspur evaluieren. Nicht als ungemessener Ersatz für Audio oder Transkripte einsetzen. |
| [WhisperX](https://github.com/m-bain/whisperX) | Forced Alignment und optionale Diarisierung für weitergehende Timing-Anforderungen. | Vorhandene JSON-Wortmarken sind importierbar; eigener schwerer Inferenzstack erst bei nachgewiesenem Bedarf. |
| [StoryToolkitAI](https://github.com/octimot/StoryToolkitAI), [FunClip](https://github.com/modelscope/FunClip), [autoclip](https://github.com/zhouxiaoka/autoclip) | Semantische Sichtung, textbasierter Schnitt und Highlight-Workflows liefern gute UX-/Interchange-Vorbilder. | Schmale Exporte/Adapter nutzen; keine komplette zweite Anwendung in den kleinen Shell-Editor einbetten. |

Priorität für den ersten echten Bestand: (1) ASR und Timing messen, (2) Suchtreffer und Ereignispräzision beurteilen,
(3) Story-/Kontextgrenzen verbessern, (4) Sprecher und freie visuelle Suche ergänzen, (5) editor-native Timelines.
Ein lokales VLM/LLM zum Nachsortieren weniger Kandidaten wäre eine weitere Stufe; der aktuelle Score behauptet noch kein Story-Verständnis.

## Best Practices, die tatsächlich umgesetzt sind

Keine neue Pflicht-Cloud und keine Modell-Downloads im Analysepfad. Kleine reproduzierbare End-to-End-Demos vor Erweiterungen.
Separate Quellen für Messwerte, akustische Labels, importierte Annotationen, Textreaktionen und relative Bildähnlichkeit.
Atomarer Austausch je Video, explizite Fehler/Exitcodes, wiederaufnehmbare Batches und Cache-Signaturen.
Originale bleiben unverändert. Lokale Modelle werden wiederverwendet, Bilder gestreamt und Evidenz über sortierte Zeitachsen ausgewählt.
Core-, Adapter- und echte Inferenzprüfungen werden im [Validierungsbericht](VALIDATION_LOCAL_SEARCH.md) getrennt ausgewiesen.

Die Vorgehensweise folgt auch den bestehenden [Supersynergy AI Best Practices](https://github.com/Supersynergy/AI-Best-Practices):
konkrete Akzeptanz, kurze Feedbackschleifen, aktuelle Primärquellen und überprüfte Nutzung statt Versionsnummern als Qualitätsersatz.
