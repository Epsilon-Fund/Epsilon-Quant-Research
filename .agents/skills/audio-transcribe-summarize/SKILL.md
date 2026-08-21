---
name: audio-transcribe-summarize
description: >
  Transcribe a local audio file (voice notes, calls, meetings — e.g. WhatsApp
  .opus files) with speaker diarization, then write a structured summary. Use
  whenever the user shares an audio file path and asks to transcribe it,
  summarize it, or turn it into notes — even if they don't say "transcribe"
  explicitly (e.g. "here's the recording from my call with X", "can you go
  through this voice note"). Requires whisperx and ffmpeg installed locally.
---

<!--
Provenance: user-supplied (Justin, 2026-07-10) — no tracked upstream repo or
commit reference is attached to this copy. Treat as unspecified provenance in
the catalog until a source is identified. Justin
noted a copy of this skill may already exist locally under a different macOS
user account on this machine — if found, diff against this copy before trusting
either as canonical.

Runtime note: this needs `ffmpeg` + `whisperx` (+ an accepted HF_TOKEN for the
pyannote diarization models) on PATH. Cowork's sandboxed session cannot install
these without asking, and may not have them at all — this skill is meant to run
from a local Claude Code / Codex CLI session on Justin's actual machine, not
inside a Cowork sandbox. Mirrors the cost-mode / data-contract packaging layout
(symlink in `.claude/skills/`; `~/.codex/skills/` symlink must be made locally —
that path lives outside this repo and outside any sandboxed session's HOME).
-->

# Audio Transcribe & Summarize

Turns a local audio file into a diarized transcript and a structured `summary.md`.

## Requirements (check once, not every run)

- `ffmpeg` on PATH
- `whisperx` installed (`pip install whisperx`) — needs a HuggingFace token accepted
  for the pyannote diarization models (`pyannote/speaker-diarization`,
  `pyannote/segmentation`), set as `HF_TOKEN` env var
- If either is missing, tell the user exactly what to install/configure and stop.
  Don't try to install system-wide dependencies yourself without asking.

## Step 1: Convert to wav

Always convert first, regardless of input format — don't special-case by extension:

```bash
ffmpeg -i "<input_path>" -y "/tmp/audio_transcribe/audio.wav"
```

## Step 2: Transcribe + diarize

Default language is English (`en`). If the user says the audio is in another
language, use that instead — don't run language auto-detection for anything over
a couple minutes. Default model is `medium`. Before running, check audio duration
(`ffprobe`). If duration > 45 minutes, use `small` instead and tell the user you
did so and why.

```bash
whisperx "/tmp/audio_transcribe/audio.wav" \
  --model <medium|small> \
  --language <en|es|...> \
  --compute_type int8 \
  --diarize \
  --output_format srt \
  --output_dir ./transcript
```

If whisperx errors on the compute type for the user's hardware (e.g. no GPU, int8
unsupported), retry once with `--compute_type float32` and note the fallback to
the user.

## Step 3: Read the transcript

Read `./transcript/audio.srt` (has timestamps + `SPEAKER_00` / `SPEAKER_01` /
etc. tags from diarization — real diarization, not inferred from content).

## Step 4: Map speaker tags to identities

The user is almost always one of the speakers. Map `SPEAKER_NN` labels to real
names/roles using context clues: self-introductions, how people address each
other, who's clearly running vs. receiving the conversation. Usually 2 speakers,
but handle 3+ without special-casing. Do not silently override the diarization's
speaker split. If a specific turn's speaker tag looks wrong (content contradicts
who it's attributed to, or a turn is implausibly short/long for that speaker),
flag it in the final summary rather than reassigning it yourself. Do not invent
or fill in content for inaudible/unclear sections — mark those explicitly as
unclear.

## Step 4.5: Where to save output (epsilon-quant-research)

Default landing zone: `meetings/transcripts/<YYYY-MM-DD>_<short-slug>/`, containing
`audio.srt` (from Step 2/3) and `summary.md` (Step 5). This is separate from
`meetings/` root, which holds curated hand-written prep notes (see
`meetings/README.md`) — transcripts are raw derived output, not a prep doc.
Add one line to `meetings/README.md` § Meeting notes pointing at the new
subfolder entry. If the audio clearly belongs to a specific research thread
instead (e.g. a call about a named branch), ask before filing it under a
branch folder instead of `meetings/transcripts/`.

## Step 5: Write summary.md

Structure (consistent skeleton, but content/section count adapts to what's
actually in the conversation — don't force sections that don't apply):

```markdown
# Summary: <short descriptive title>
**Participants:** <names/roles>
**Date:** <if known from context or filename, else omit>
<Only if the user tells you this is part of an ongoing series: **Meeting #:** Nth call/conversation with X>

## Abstract
2–4 sentences: what this conversation was fundamentally about.

## <Theme 1>
## <Theme 2>
## <Theme N>
(Thematic breakdown — headers should reflect actual content, e.g. "Budget,"
"Next steps," "Disagreement over X." Vary freely between runs.)

## Flags / Uncertainties
- Any speaker-attribution ambiguity from Step 4
- Any inaudible/unclear audio sections
- (Omit this section entirely if there's nothing to flag — don't pad it)
```

Keep it tight — length should track content density, not hit a target word count.

## Notes

- Never guess at content the model didn't actually transcribe. If a whole
  stretch is unclear, say so in Flags rather than smoothing it over.
- Clean up `/tmp/audio_transcribe/` after finishing unless the user asks to
  keep the wav.
