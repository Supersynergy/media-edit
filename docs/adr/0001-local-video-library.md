# ADR 0001: Add an independent local video catalog

Status: accepted for the prototype, 2026-10-18.

## Context

`me_hook` selects a loud audio-energy window and `me_shorts` reuses it. The existing editing primitives are useful,
but neither operation stores searchable evidence across recordings. Replacing those functions would change established editing behavior
and require model setup before a first successful run.

## Decision

Add a Python package beside the shell. The mandatory path uses the standard library, existing FFmpeg/ffprobe and SQLite FTS5.
Keep ASR, acoustic events, categorical emotion, semantic retrieval, CLIP and adaptive scene detection behind explicit optional adapters.
Use local model paths, offline hub settings, no remote code and no automatic model download during ingestion.
Parakeet-MLX is an Apple-Silicon ASR option; Whisper implementations remain available for other workloads.

Persist original transcript/timing evidence, measured audio/visual evidence, tool/model provenance and scored windows.
Keep textual reaction hints, acoustic labels, imported annotations and CLIP similarity distinct.
Do not infer laughter from volume or emotional truth from a transcript.
Use configurable, explainable editorial weights and overlap suppression rather than claiming a universal highlight probability.

Replace a video's derived rows and FTS entries in one transaction after successful analysis. Hash source media and sidecars,
include model metadata, options and package versions, and preserve earlier results after a failed re-index.
Stream decoded frames and slice sorted evidence timelines. Load optional models once per batch.

Expose a loopback-only review server with registered video IDs, range requests and cached WebM fallback previews.
Keep MP4 for editing exports; use VP9/Opus for browser fallback so Chromium builds without H.264 can play it too.
Use JSON markers, per-source LosslessCut CSV and quoted FFmpeg/mpv argument lists as the first editing interfaces.
Re-encode precise previews; do not assert that legacy keyframe stream copies are frame accurate.

## Consequences

First success and tests need no learned weights. Model adapters can evolve independently without breaking editing commands.
The synthetic integration demo demonstrates pipeline correctness, not real-world perception quality.
All optional package versions are recorded in a reproducible lock; actual model weights and Apple hardware need separate verification.

The prototype uses local absolute paths and non-stable segment IDs after reindexing. Moved drives need explicit reindex/removal.
Evidence and JSON embeddings grow linearly; semantic retrieval and global ranking currently scan candidates.
A measured vector index, long-recording checkpoints within one video, automatic sentence-boundary refinement and editor-native timelines
can be added once a representative corpus establishes their value.
