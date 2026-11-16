<div align="center">

![media-edit](docs/assets/cover.png)

# `media-edit`

### **Polish any video or audio on your Mac. One command.**

**Cut silences. Transcribe. Burn captions. Denoise. Master to broadcast loudness. Separate stems. Detect scenes. Resync subtitles.** All in one adaptive CLI, Metal-accelerated where it matters, 100% local, zero cloud.

```bash
media-edit polish raw_podcast.mp4
# ✂️  silence cut → 🔇 denoise → 🎚 -16 LUFS → 📝 transcribe → 🎨 burn captions
```

[![macOS](https://img.shields.io/badge/macOS-12%2B-000000?logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Metal-555?logo=apple&logoColor=white)](#how-it-adapts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ShellCheck](https://github.com/Supersynergy/media-edit/actions/workflows/shellcheck.yml/badge.svg)](https://github.com/Supersynergy/media-edit/actions/workflows/shellcheck.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

**🧭 Explore:** [60 seconds](#60-seconds) · [Install](#install) · [Recipes](#recipes) · [AI agents](#ai-agent-integration) · [Limits](#honest-limits)

---

## 60 seconds

```bash
media-edit polish lecture.mov              # full pipeline, done.
media-edit cut interview.mp4               # just remove silences
media-edit transcribe talk.mp4 srt         # whisper.cpp Metal → talk.srt
media-edit caption reel.mp4 out.mp4 tiktok # burn yellow outlined captions
media-edit master song.wav "" -14          # master to Spotify -14 LUFS
media-edit stems song.mp3                  # split vocals/drums/bass/other
media-edit scenes movie.mkv 27             # auto-split by scene cuts
media-edit pitch voice.wav 2 1.0           # +2 semitones, keep tempo
media-edit syncsub movie.mkv subs.srt      # fix subtitle drift (alass)
```

Each editing primitive routes to a local tool. `polish` chains the important ones.

## Local video library and multimodal highlights

Index a whole folder, search across recordings, and review scored moments with source timestamps:

```bash
source media-edit.sh
me_index "/Volumes/Videos"               # use existing JSON/SRT/VTT transcripts
me_search "Automatisierung"              # cross-video word search
me_search "lachen" --mode event          # event search with evidence provenance
me_review                               # http://127.0.0.1:8765
me_export_highlights --out highlights --clips
```

The catalog combines transcripts, audio events, RMS trends, scene boundaries, motion and visual features.
Optional local **Parakeet-MLX**, **MLX Whisper**, **faster-whisper**, AudioSet/emotion classifiers,
CLIP and Sentence Transformers add ASR, event/content tagging and semantic topic search.
Core indexing needs Python 3.12+, FFmpeg and SQLite FTS5; no pip dependencies or automatic model downloads.
Candidate scores include reasons and source evidence. They are suggestions to review, not probabilities of a good clip.
Exports include JPEG/MP4 previews, JSON markers, per-source LosslessCut CSV and precise FFmpeg/mpv commands.
Existing `hook`, `shorts`, `cut`, `polish` and other editing commands remain available.

**Try it offline:** `python3 examples/local_search_demo.py --out demo-output`

See [setup and practical workflows](docs/LOCAL_VIDEO_SEARCH.md),
[October 2026 dependency research](docs/RESEARCH_2026-10.md),
[validation and quality limits](docs/VALIDATION_LOCAL_SEARCH.md),
and the [architecture decision](docs/adr/0001-local-video-library.md).

---

## Install

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Supersynergy/media-edit/main/install.sh)"
```

Read [install.sh](install.sh) first, it's ~70 lines. Uninstall any time with the matching `uninstall.sh`.

**What gets installed:**

- Brew: `ffmpeg` `whisper-cpp` `rubberband` `sox` `alass` `mediainfo` `mpv`
- uv tools: `auto-editor` `demucs` `scenedetect` `ffmpeg-normalize`
- The shell: `media-edit.sh` sourced into your `~/.zshrc`
- Skill file: dropped into `~/.gg/skills/` if it exists (optional, for Claude Code / GG Coder / AI agents)

---

## What each command owns

<details>
<summary><strong>📋 Show 11 entries</strong></summary>

- **`cut`**<br>
  Tool underneath: `auto-editor`<br>
  Why it's the right pick: Speech-aware, beats `ffmpeg silenceremove` by miles

- **`trim`**<br>
  Tool underneath: `ffmpeg -c copy`<br>
  Why it's the right pick: Zero re-encode, frame-accurate on keyframes

- **`concat`**<br>
  Tool underneath: `ffmpeg -f concat -c copy`<br>
  Why it's the right pick: Lossless join for compatible streams

- **`transcribe`**<br>
  Tool underneath: `whisper.cpp` (Metal)<br>
  Why it's the right pick: 8–16× faster than CPU on Apple Silicon, no cloud

- **`caption`**<br>
  Tool underneath: `whisper.cpp` + `ffmpeg ASS`<br>
  Why it's the right pick: Generates then burns styled subs in one call

- **`denoise`**<br>
  Tool underneath: `ffmpeg anlmdn`<br>
  Why it's the right pick: Non-local-means, best built-in denoiser

- **`master`**<br>
  Tool underneath: `ffmpeg-normalize`<br>
  Why it's the right pick: Proper 2-pass EBU R128, broadcast-compliant

- **`stems`**<br>
  Tool underneath: `demucs htdemucs`<br>
  Why it's the right pick: SOTA local source separation

- **`scenes`**<br>
  Tool underneath: `PySceneDetect`<br>
  Why it's the right pick: Content-aware, frame-accurate splits

- **`pitch`**<br>
  Tool underneath: `rubberband`<br>
  Why it's the right pick: SOTA independent pitch/tempo

- **`syncsub`**<br>
  Tool underneath: `alass`<br>
  Why it's the right pick: Magical re-sync without needing a transcript

</details>


---

## `polish`: the killer feature

```bash
media-edit polish raw.mp4 polished.mp4 modern
```

Runs 5 steps, each with the right tool:

```
 [1/5] ✂️   auto-editor silence-cut
 [2/5] 🔇  ffmpeg highpass + anlmdn + lowpass
 [3/5] 🎚   ffmpeg-normalize → -16 LUFS (YouTube)
 [4/5] 📝  whisper.cpp Metal → SRT
 [5/5] 🎨  ffmpeg ASS burn-in (modern | tiktok | cinema styles)
```

Result: a captioned, denoised, loudness-correct, silence-trimmed video in one command. Drop it into a `recipe` file and share your workflow.

---

## Recipes

Pipelines as plain text. Ship one per use-case:

```bash
# recipes/podcast.recipe
cut                 # silence removal
denoise             # anlmdn
master -16          # YouTube loudness
transcribe srt
caption modern
```

```bash
media-edit recipe recipes/podcast.recipe my_show.mp4
```

Bundled recipes: **podcast** (-16 LUFS, modern captions), **tiktok** (-14 LUFS, yellow big text), **cinema** (-23 LUFS, subtle serif captions).

---

## How it adapts

`me_info` detects your Mac on first use:

<details>
<summary><strong>📋 Show 5 entries</strong></summary>

- **🏆 Ultra**<br>
  Chip: M*-Ultra<br>
  Whisper default: `large-v3`<br>
  Encoder: `hevc_videotoolbox`

- **⚡ Max**<br>
  Chip: M*-Max<br>
  Whisper default: `large-v3` (≥32 GB)<br>
  Encoder: `hevc_videotoolbox`

- **💪 Pro**<br>
  Chip: M*-Pro<br>
  Whisper default: `medium.en`<br>
  Encoder: `hevc_videotoolbox`

- **🍎 Base**<br>
  Chip: M1/M2/M3/M4<br>
  Whisper default: `base.en`<br>
  Encoder: `hevc_videotoolbox`

- **🖥 Intel**<br>
  Chip: Intel Mac<br>
  Whisper default: `tiny.en` or `base.en`<br>
  Encoder: `libx264`

</details>


Override anytime: `ME_WHISPER_MODEL=large-v3 media-edit transcribe …`

---

## Sister skill

Need *format conversion* (MP4↔MKV, WAV↔MP3, PNG↔AVIF)? Use **[`conv`](https://github.com/Supersynergy/conv)**, the sibling skill. They're designed to chain:

```bash
conv raw.mov working.mp4           # normalize container
media-edit polish working.mp4      # polish it
conv polished.mp4 final.webm       # re-encode for distribution
```

---

## Easter eggs

```bash
media-edit joke      # encoding joke
media-edit zen       # editing koan
media-edit stats     # lifetime ops counter + most-used commands
```

Plus a daily whisper.cpp model download notification, optional completion chime (`ME_SOUND=0` to silence), and you'll find the rest.

---

## AI agent integration

Drops `skill/universal-media-editor.md` into `~/.gg/skills/` on install. Claude Code, GG Coder, and similar agents will route *editing* requests through `media-edit` and *conversion* requests through `conv`. Natural split.

---

## Honest limits

- The original editing shell targets macOS. The new Python library core also runs on Linux;
  Parakeet-MLX and MLX Whisper require Apple Silicon.
- `polish` is opinionated. It assumes spoken content. For music mastering, use `master` + `stems` directly.
- Whisper models download on first transcribe (~150 MB for base.en, ~3 GB for large-v3). Cached to `~/.cache/whisper.cpp/`.
- Subtitle timing from Whisper is segment-level. For word-level karaoke timing, run `faster-whisper` with `--word_timestamps`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Most-wanted: **new recipes** for your workflow, **M1/M2/Ultra benchmarks**, **additional primitives**.

## License

MIT, by [Maxim Supersynergy](https://github.com/Supersynergy).

---

<div align="center">

*If this saved you an hour of editing, star it ⭐ and tell a friend who still does it by hand.*

</div>

## Additional documentation

[docs/reference](docs/reference/)
