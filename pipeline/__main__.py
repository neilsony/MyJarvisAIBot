"""Phase 0 corpus pipeline CLI.

Run the steps in order; each is resumable and each tells you what to do next:

    python -m pipeline status
    python -m pipeline fetch --limit 20
    python -m pipeline fetch-reference <any-video-url>
    python -m pipeline diarize
    python -m pipeline sample <episode-id>
    python -m pipeline reference --from-episode <episode-id> --speaker SPEAKER_02
    python -m pipeline identify
    python -m pipeline voiceprint
    python -m pipeline ingest
    python -m pipeline register-candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from brain.config import Settings
from pipeline.segments import Turn

AUDIO_DIR = "audio"
TURNS_DIR = "turns"
LABELS_FILE = "labels.json"
VOICE_NAME = "dmills"


def _paths(settings: Settings) -> tuple[Path, Path, Path, Path]:
    audio = settings.data_dir / AUDIO_DIR
    turns = settings.data_dir / TURNS_DIR
    labels = settings.data_dir / LABELS_FILE
    voice = settings.voices_dir / VOICE_NAME
    return audio, turns, labels, voice


def cmd_status(settings: Settings, _args: argparse.Namespace) -> int:
    audio_dir, turns_dir, labels_file, voice_dir = _paths(settings)
    episodes = sorted(audio_dir.glob("*.wav")) if audio_dir.is_dir() else []
    diarized = sorted(turns_dir.glob("*.json")) if turns_dir.is_dir() else []
    labels = json.loads(labels_file.read_text()) if labels_file.is_file() else {}
    reference = voice_dir / "reference.npy"
    clips = sorted((voice_dir / "clips").glob("*.wav")) if (voice_dir / "clips").is_dir() else []

    print("Phase 0 — corpus pipeline\n")
    print(f"  model            {settings.model}")
    print(f"  HF_TOKEN         {'set' if settings.hf_token else 'MISSING (SETUP.md step 3)'}")
    print(f"  episodes fetched {len(episodes)}")
    print(f"  episodes diarized{len(diarized):>4}")
    print(f"  reference voice  {'built' if reference.is_file() else 'not built'}")
    print(f"  episodes labeled {len(labels)}")
    print(f"  voiceprint clips {len(clips)}")

    if not episodes:
        print("\nNext: python -m pipeline fetch --limit 20")
    elif len(diarized) < len(episodes):
        print("\nNext: python -m pipeline diarize")
    elif not reference.is_file():
        print("\nNext: identify Darrick by ear (SETUP.md step 5):")
        print("      python -m pipeline sample <episode-id>")
    elif not labels:
        print("\nNext: python -m pipeline identify")
    elif not clips:
        print("\nNext: python -m pipeline voiceprint")
    else:
        print("\nPhase 0 complete. Next: the bake-off (SETUP.md step 8).")
    return 0


def cmd_list(_settings: Settings, args: argparse.Namespace) -> int:
    from pipeline.fetch import list_episodes

    for ep in list_episodes(limit=args.limit):
        mins = f"{ep.duration_seconds / 60:5.1f}m" if ep.duration_seconds else "    ?"
        print(f"{ep.video_id}  {mins}  {ep.title}")
    return 0


def cmd_fetch(settings: Settings, args: argparse.Namespace) -> int:
    from pipeline.fetch import download_audio, list_episodes

    audio_dir, *_ = _paths(settings)
    episodes = list_episodes(limit=args.limit)
    print(f"{len(episodes)} episodes to fetch into {audio_dir}\n")
    for i, ep in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] {ep.title[:70]}")
        try:
            download_audio(ep, audio_dir)
        except Exception as exc:  # one bad video shouldn't kill a 20-episode run
            print(f"    SKIPPED: {exc}", file=sys.stderr)
    return 0


def cmd_fetch_reference(settings: Settings, args: argparse.Namespace) -> int:
    """Download one arbitrary clip as reference audio, outside the show."""
    from pipeline.fetch import fetch_reference_clip

    out_dir = settings.data_dir / "reference_audio"
    try:
        path = fetch_reference_clip(args.url, out_dir)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    print(f"Downloaded -> {path}")
    print(f"\nNext: python -m pipeline reference {path}")
    return 0


def cmd_fetch_solo(settings: Settings, args: argparse.Namespace) -> int:
    """Download videos from Darrick's own channel — the voiceprint source.

    Separate from `fetch` (the NOTB show) on purpose: this is solo talking-head
    content, so it never has the 4-way speaker-separation problem the show
    audio does. Canon can still come from the show; Voiceprint comes from here.
    """
    from pipeline.fetch import download_audio, list_episodes

    out_dir = settings.data_dir / "solo_audio"
    episodes = list_episodes(f"{args.channel}/videos", limit=args.limit)
    if not episodes:
        print("No videos found at that channel URL.", file=sys.stderr)
        return 1

    print(f"{len(episodes)} videos to fetch into {out_dir}\n")
    for i, ep in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] {ep.title[:70]}")
        try:
            download_audio(ep, out_dir)
        except Exception as exc:
            print(f"    SKIPPED: {exc}", file=sys.stderr)
    return 0


def cmd_diarize_solo(settings: Settings, _args: argparse.Namespace) -> int:
    """Diarize solo videos with num_speakers=1 — no separation needed.

    Still runs through the same diarizer as the show, but forcing one speaker
    means it's really just voice-activity detection: find the speech, skip the
    music/silence. Output feeds straight into `voiceprint-solo`.
    """
    from pipeline.diarize import diarize

    audio_dir = settings.data_dir / "solo_audio"
    turns_dir = settings.data_dir / "solo_turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    token = settings.require("hf_token")

    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        print(
            "No solo audio yet. Run: python -m pipeline fetch-solo <channel-url>",
            file=sys.stderr,
        )
        return 1

    for i, audio in enumerate(files, 1):
        cache = turns_dir / f"{audio.stem}.json"
        state = "cached" if cache.is_file() else "diarizing"
        print(f"[{i}/{len(files)}] {audio.stem} — {state}")
        turns = diarize(audio, token, cache_path=cache, num_speakers=1)
        speakers = sorted({t.speaker for t in turns})
        print(f"    {len(turns)} turns, {len(speakers)} speaker: {', '.join(speakers)}")
    return 0


def _solo_episodes(settings: Settings) -> dict[Path, tuple[list[Turn], str]]:
    """Every diarized solo video, keyed by audio path -> (turns, speaker label).

    No identify step needed here — with num_speakers=1 there's only ever one
    label per file, so there's nothing to disambiguate.
    """
    from pipeline.persist import load_turns

    audio_dir = settings.data_dir / "solo_audio"
    turns_dir = settings.data_dir / "solo_turns"

    episodes: dict[Path, tuple[list[Turn], str]] = {}
    for audio in sorted(audio_dir.glob("*.wav")):
        turns = load_turns(turns_dir / f"{audio.stem}.json")
        if not turns:
            continue
        (label,) = {t.speaker for t in turns}
        episodes[audio] = (turns, label)
    return episodes


def _notb_episodes(settings: Settings) -> dict[Path, tuple[list[Turn], str]]:
    """Every NOTB episode with a confirmed label in labels.json.

    Confirmation can come from `identify` (embedding match) or a hand-added
    entry (see `sample` — a human recognizing a voice is stronger evidence
    than a borderline embedding score).
    """
    from pipeline.persist import load_turns

    audio_dir, turns_dir, labels_file, _ = _paths(settings)
    if not labels_file.is_file():
        return {}

    labels = json.loads(labels_file.read_text())
    episodes: dict[Path, tuple[list[Turn], str]] = {}
    for stem, info in labels.items():
        turns = load_turns(turns_dir / f"{stem}.json")
        audio = audio_dir / f"{stem}.wav"
        if turns and audio.is_file():
            episodes[audio] = (turns, str(info["label"]))
    return episodes


def cmd_voiceprint_solo(settings: Settings, args: argparse.Namespace) -> int:
    """Build the voiceprint training set from solo videos only.

    Useful for isolating what the solo source alone contributes. For the real
    combined dataset (solo videos + confirmed NOTB episodes), use `voiceprint`.
    """
    from pipeline.build_voiceprint import DEFAULTS, VoiceprintBuildSettings, build_voiceprint

    _, _, _, voice_dir = _paths(settings)
    episodes = _solo_episodes(settings)
    if not episodes:
        print("No diarized solo audio. Run: python -m pipeline diarize-solo", file=sys.stderr)
        return 1

    settings_ = VoiceprintBuildSettings(
        min_duration=DEFAULTS.min_duration, pad=DEFAULTS.pad, target_minutes=args.minutes
    )
    result = build_voiceprint(episodes, voice_dir, settings=settings_)

    print(f"\n{result.clip_count} clips, {result.total_minutes:.1f} min "
          f"from {result.source_episodes} video(s)")
    print(f"  clips     {result.clips_dir}")
    print(f"  reference {result.reference_clip}")
    if result.total_minutes < args.minutes * 0.6:
        print(f"\nShort of {args.minutes:.0f} min — fetch more videos with fetch-solo.")
    return 0


def cmd_diarize(settings: Settings, _args: argparse.Namespace) -> int:
    from pipeline.diarize import diarize

    token = settings.require("hf_token")
    audio_dir, turns_dir, *_ = _paths(settings)
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        print("No audio yet. Run: python -m pipeline fetch --limit 20", file=sys.stderr)
        return 1

    for i, audio in enumerate(files, 1):
        cache = turns_dir / f"{audio.stem}.json"
        state = "cached" if cache.is_file() else "diarizing (minutes)"
        print(f"[{i}/{len(files)}] {audio.stem} — {state}")
        turns = diarize(audio, token, cache_path=cache)
        speakers = sorted({t.speaker for t in turns})
        print(f"    {len(turns)} turns, {len(speakers)} speakers: {', '.join(speakers)}")
    return 0


def cmd_sample(settings: Settings, args: argparse.Namespace) -> int:
    """Export a few clips per speaker so a human can tell who is who by ear.

    Three tiers, in order, so a quiet speaker still gets something to listen
    to instead of silently coming up empty:

      1. Fully clean — same crosstalk-free filter as the voiceprint itself.
      2. Mostly clean — some overlap exists, but ranked by *how little*
         (a one-word reaction at the edge, not a shouting match), rather than
         by raw duration. The earlier version of this ranked by duration
         alone, which meant a long, heavily-overlapped turn could beat a
         shorter, nearly-clean one — exactly backwards for "can I tell whose
         voice this is."
      3. Whatever's longest, full stop — only reached if a speaker has
         nothing better anywhere in the episode.
    """
    from pipeline.audio import read_audio, write_turn
    from pipeline.persist import load_turns
    from pipeline.segments import clean_turns, overlap_fraction

    audio_dir, turns_dir, _, _ = _paths(settings)
    audio = audio_dir / f"{args.episode}.wav"
    turns = load_turns(turns_dir / f"{args.episode}.json")
    if turns is None or not audio.is_file():
        print(f"No diarized episode '{args.episode}'. Run diarize first.", file=sys.stderr)
        return 1

    out_dir = settings.data_dir / "samples" / args.episode
    data, sample_rate = read_audio(audio)
    for speaker in sorted({t.speaker for t in turns}):
        speaker_turns = [t for t in turns if t.speaker == speaker]
        others = [t for t in turns if t.speaker != speaker]

        clean = sorted(
            clean_turns(turns, speaker=speaker, min_duration=2.0, pad=0.25),
            key=lambda t: t.duration,
            reverse=True,
        )[: args.per_speaker]
        for i, turn in enumerate(clean):
            write_turn(data, sample_rate, turn, out_dir / f"{speaker}_{i}.wav")
        used = {(t.start, t.end) for t in clean}

        needed = args.per_speaker - len(clean)
        mostly_clean: list[Turn] = []
        if needed > 0:
            candidates = [t for t in speaker_turns if t.duration >= 2.0]
            candidates.sort(key=lambda t: overlap_fraction(t, others))
            mostly_clean = candidates[:needed]
            for i, turn in enumerate(mostly_clean):
                write_turn(data, sample_rate, turn, out_dir / f"{speaker}_{i}_mostly.wav")
            used |= {(t.start, t.end) for t in mostly_clean}
            needed -= len(mostly_clean)

        raw_clips: list[Turn] = []
        if needed > 0:
            remaining = [t for t in speaker_turns if (t.start, t.end) not in used]
            raw_clips = sorted(remaining, key=lambda t: t.duration, reverse=True)[:needed]
            for i, turn in enumerate(raw_clips):
                write_turn(data, sample_rate, turn, out_dir / f"{speaker}_{i}_raw.wav")

        parts = [f"{len(clean)} clean"]
        if mostly_clean:
            parts.append(f"{len(mostly_clean)} mostly-clean")
        if raw_clips:
            parts.append(f"{len(raw_clips)} raw (heavy crosstalk)")
        print(f"{speaker}: {', '.join(parts)}")

    print(f"\nListen to {out_dir}")
    print("Then: python -m pipeline reference --from-episode "
          f"{args.episode} --speaker SPEAKER_0X")
    return 0


def cmd_reference(settings: Settings, args: argparse.Namespace) -> int:
    import numpy as np

    from pipeline.identify import build_reference

    token = settings.require("hf_token")

    if args.from_episode:
        # Build straight from a speaker the human identified by ear — no manual
        # audio editing, and the clips are already known to be crosstalk-free.
        from pipeline.audio import read_audio, write_turn
        from pipeline.persist import load_turns
        from pipeline.segments import clean_turns

        audio_dir, turns_dir, _, voice_dir = _paths(settings)
        audio = audio_dir / f"{args.from_episode}.wav"
        turns = load_turns(turns_dir / f"{args.from_episode}.json")
        if turns is None or not audio.is_file():
            print(f"No diarized episode '{args.from_episode}'.", file=sys.stderr)
            return 1

        picked = sorted(
            clean_turns(turns, speaker=args.speaker, min_duration=6.0, pad=0.25),
            key=lambda t: t.duration,
            reverse=True,
        )[:8]
        if not picked:
            print(f"No clean turns for {args.speaker} in that episode.", file=sys.stderr)
            return 1

        refs_dir = voice_dir / "refs"
        data, sample_rate = read_audio(audio)
        clips = [
            write_turn(data, sample_rate, t, refs_dir / f"ref_{i}.wav")
            for i, t in enumerate(picked)
        ]
    else:
        clips = [Path(p) for p in args.clips]

    missing = [p for p in clips if not p.is_file()]
    if missing:
        print(f"Missing clips: {missing}", file=sys.stderr)
        return 1

    *_, voice_dir = _paths(settings)
    voice_dir.mkdir(parents=True, exist_ok=True)
    reference = build_reference(clips, token)
    np.save(voice_dir / "reference.npy", reference)
    print(f"Reference voice built from {len(clips)} clips -> {voice_dir / 'reference.npy'}")
    return 0


def cmd_identify(settings: Settings, _args: argparse.Namespace) -> int:
    import numpy as np

    from pipeline.identify import find_speaker, speaking_time
    from pipeline.persist import load_turns

    token = settings.require("hf_token")
    audio_dir, turns_dir, labels_file, voice_dir = _paths(settings)
    reference_path = voice_dir / "reference.npy"
    if not reference_path.is_file():
        print("No reference voice. See SETUP.md step 5.", file=sys.stderr)
        return 1

    reference = np.load(reference_path)
    labels: dict[str, dict[str, float | str]] = {}
    skipped: list[str] = []

    for audio in sorted(audio_dir.glob("*.wav")):
        turns = load_turns(turns_dir / f"{audio.stem}.json")
        if turns is None:
            continue
        match = find_speaker(audio, turns, reference, token)
        if match is None:
            skipped.append(audio.stem)
            print(f"{audio.stem}  AMBIGUOUS — skipped")
            continue
        share = speaking_time(turns, match.label) / max(
            sum(t.duration for t in turns), 1e-9
        )
        labels[audio.stem] = {"label": match.label, "score": match.score, "margin": match.margin}
        print(
            f"{audio.stem}  {match.label}  score={match.score:.3f} "
            f"margin={match.margin:.3f}  share={share:.0%}"
        )

    labels_file.parent.mkdir(parents=True, exist_ok=True)
    labels_file.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"\n{len(labels)} labeled, {len(skipped)} skipped -> {labels_file}")
    if skipped:
        print("Skipped episodes are ambiguous, not broken. Ignore unless most are skipped.")
    return 0


def cmd_voiceprint(settings: Settings, args: argparse.Namespace) -> int:
    """Build the combined voiceprint training set — solo videos + confirmed
    NOTB episodes, pooled together. This is the real dataset; `voiceprint-solo`
    is for isolating the solo source alone."""
    from pipeline.build_voiceprint import DEFAULTS, VoiceprintBuildSettings, build_voiceprint

    _, _, _, voice_dir = _paths(settings)
    episodes = {**_solo_episodes(settings), **_notb_episodes(settings)}

    if not episodes:
        print(
            "No episodes to build from. Run diarize-solo and/or identify first.",
            file=sys.stderr,
        )
        return 1

    settings_ = VoiceprintBuildSettings(
        min_duration=DEFAULTS.min_duration,
        pad=DEFAULTS.pad,
        target_minutes=args.minutes,
    )
    result = build_voiceprint(episodes, voice_dir, settings=settings_)

    print(f"\n{result.clip_count} clips, {result.total_minutes:.1f} min "
          f"from {result.source_episodes} source(s)")
    print(f"  clips     {result.clips_dir}")
    print(f"  reference {result.reference_clip}")
    if result.total_minutes < args.minutes * 0.6:
        print(f"\nWell short of {args.minutes:.0f} min. Fetch more episodes, or lower")
        print("min_duration in pipeline/build_voiceprint.py (quality cost).")
    return 0


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    """Transcribe episodes, attribute speakers, chunk, embed, store.

    This is the bridge from Phase 0 (audio) to Phase 1 (a bot that knows the
    show). Transcription is the slow part and is cached per episode.
    """
    import json

    from brain.embeddings import embed
    from brain.store.db import Store
    from pipeline.persist import load_turns
    from pipeline.transcribe import transcribe
    from pipeline.transcript import attribute, chunk_transcript

    audio_dir, turns_dir, labels_file, _ = _paths(settings)
    text_dir = settings.data_dir / "transcripts"
    text_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.data_dir / "jarvis.db")

    from pipeline.naming import load_names, load_windows

    name_map = load_names(settings.data_dir / "speaker_names.json")
    darrick_dir = settings.data_dir / "darrick_turns"
    window_map = {
        p.stem: load_windows(p) for p in sorted(darrick_dir.glob("*.json"))
    } if darrick_dir.is_dir() else {}
    audio_files = sorted(audio_dir.glob("*.wav"))
    if args.episode:
        # Ingesting one episode at a time is the deliberate workflow: a label
        # is confirmed by ear before its episode enters the Canon, and a
        # mistake is cheap to wipe while the store holds one episode. The
        # hash-based dedupe in Store.add_chunks is insert-only — it cannot
        # undo a wrongly-attributed ingest — so "ingest everything and fix
        # labels later" is the one path this filter exists to prevent.
        wanted = args.episode
        audio_files = [a for a in audio_files if a.stem == wanted]
        if not audio_files:
            print(
                f"No audio for episode '{wanted}'. Run: python -m pipeline fetch",
                file=sys.stderr,
            )
            return 1
    if not audio_files:
        print("No audio. Run: python -m pipeline fetch --limit 20", file=sys.stderr)
        return 1

    for i, audio in enumerate(audio_files, 1):
        stem = audio.stem
        cache = text_dir / f"{stem}.json"
        print(f"[{i}/{len(audio_files)}] {stem}")

        if cache.is_file():
            raw = [
                (r["text"], r["start"], r["end"]) for r in json.loads(cache.read_text())
            ]
            from pipeline.transcript import TranscriptSegment

            segments = [TranscriptSegment(text=t, start=s_, end=e) for t, s_, e in raw]
        else:
            print("    transcribing (slow)...")
            segments = transcribe(audio)
            cache.write_text(
                json.dumps(
                    [{"text": s_.text, "start": s_.start, "end": s_.end} for s_ in segments],
                    indent=2,
                )
            )

        turns = load_turns(turns_dir / f"{stem}.json") or []
        attributed = attribute(segments, turns)

        # Two naming layers, most precise last:
        #   1. label-level names (data/speaker_names.json) — "SPEAKER_02 is
        #      Pierre in this episode", hand-confirmed by ear;
        #   2. Darrick windows (data/darrick_turns/<ep>.json) — the exact
        #      spans the per-turn verifier scored as Darrick. They win over
        #      label names because they are the finer-grained evidence; this
        #      is how Darrick gets separated out of a label he shares with
        #      Pierre.
        # Unnamed labels keep their raw form: an honest SPEAKER_02 in the
        # Canon beats a guessed name.
        names = name_map.get(stem, {})
        if names:
            from pipeline.naming import apply_names

            attributed = apply_names(attributed, names)
        windows = window_map.get(stem, [])
        if windows:
            from pipeline.naming import apply_darrick_windows

            attributed = apply_darrick_windows(attributed, windows)

        chunks = chunk_transcript(attributed, episode=stem)
        added = store.add_chunks(chunks, embed([c.text for c in chunks]))
        print(f"    {len(segments)} segments -> {len(chunks)} chunks, {added} new")

    print(f"\nStore: {store.stats()}")
    return 0


def cmd_verify_speakers(settings: Settings, args: argparse.Namespace) -> int:
    """Score every turn in an episode against a solo-derived Darrick reference.

    Unlike `identify` — which averages each label into one centroid and can
    hide a merged label — this scores turns individually and reports the
    distribution. A merged label shows up as a bimodal distribution instead
    of a plausible-but-wrong average. The report ends with the top- and
    bottom-scoring turns of the best candidate so a human can confirm by ear
    in two listens.
    """
    from pipeline.identify import build_reference
    from pipeline.persist import load_turns
    from pipeline.verify_speakers import (
        embed_turn_crop,
        format_report,
        label_reports,
        middle_slice,
        score_turns,
        select_slice_turns,
    )

    token = settings.require("hf_token")
    audio_dir, turns_dir, _, voice_dir = _paths(settings)
    audio = audio_dir / f"{args.episode}.wav"
    turns = load_turns(turns_dir / f"{args.episode}.json")
    if turns is None or not audio.is_file():
        print(
            f"No diarized episode '{args.episode}'. Run diarize first.",
            file=sys.stderr,
        )
        return 1

    # Reference from SOLO clips only: solo audio has no 4-way separation
    # problem, so those clips are ground truth by construction. Show-derived
    # clips would assume the very label correctness this command exists to
    # question. Two clean clips per solo video is plenty for a centroid.
    solo_audio_dir = settings.data_dir / "solo_audio"
    solo_stems = {p.stem for p in solo_audio_dir.glob("*.wav")}
    manifest = _voice_manifest(voice_dir)
    solo_sources = sorted({str(c["source"]) for c in manifest
                           if Path(str(c["source"])).stem in solo_stems})
    if not solo_sources:
        print("No solo clips found. Run: python -m pipeline fetch-solo && diarize-solo",
              file=sys.stderr)
        return 1

    reference_clips: list[Path] = []
    for source in solo_sources:
        clips = sorted(
            (voice_dir / "clips").glob(f"{Path(source).stem}_*.wav"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        reference_clips.extend(clips[:2])
    if not reference_clips:
        print("No solo clips in voices/dmills/clips. Run voiceprint-solo first.", file=sys.stderr)
        return 1

    reference = build_reference(reference_clips, token)

    slice_start, slice_end = middle_slice(max(t.end for t in turns), args.minutes)
    to_score = select_slice_turns(turns, slice_start, slice_end)
    if not to_score:
        print("No turns long enough to score in the middle slice.", file=sys.stderr)
        return 1

    print(
        f"{args.episode}: scoring {len(to_score)} turns "
        f"in slice {slice_start / 60:.1f}m - {slice_end / 60:.1f}m "
        f"against {len(reference_clips)} solo reference clips"
    )

    from pipeline.identify import _inference

    embed_turn = embed_turn_crop(_inference(token), str(audio))
    scores = score_turns(embed_turn, to_score, reference)
    reports = label_reports(scores)

    print(format_report(reports, scores))
    return 0


def cmd_export_darrick_turns(settings: Settings, args: argparse.Namespace) -> int:
    """Verify a whole episode and write Darrick's windows for ingest.

    This is the bridge between `verify-speakers` (evidence) and ingest
    (application). It scores every turn in the episode against the solo
    reference — no slice, because attribution accuracy is the point — and
    writes every turn scoring at or above the cutoff to
    data/darrick_turns/<episode>.json. `cmd_ingest` re-attributes any ASR
    segment mostly inside those windows to DARRICK, which is how Darrick is
    separated from a co-host the diarizer merged him with.

    The cutoff sits above the low cluster (co-host bleed, short interjections)
    and at/below the high cluster (Darrick): 0.60 clears both clusters in
    every verified episode so far. Raise it to trade recall for certainty.
    """
    from pipeline.identify import build_reference
    from pipeline.persist import load_turns, save_turns, turns_from_records
    from pipeline.verify_speakers import (
        embed_turn_crop,
        label_reports,
        score_turns,
        select_slice_turns,
    )

    token = settings.require("hf_token")
    audio_dir, turns_dir, _, voice_dir = _paths(settings)
    audio = audio_dir / f"{args.episode}.wav"
    turns = load_turns(turns_dir / f"{args.episode}.json")
    if turns is None or not audio.is_file():
        print(
            f"No diarized episode '{args.episode}'. Run diarize first.",
            file=sys.stderr,
        )
        return 1

    solo_audio_dir = settings.data_dir / "solo_audio"
    solo_stems = {p.stem for p in solo_audio_dir.glob("*.wav")}
    manifest = _voice_manifest(voice_dir)
    solo_sources = sorted({str(c["source"]) for c in manifest
                           if Path(str(c["source"])).stem in solo_stems})
    reference_clips: list[Path] = []
    for source in solo_sources:
        clips = sorted(
            (voice_dir / "clips").glob(f"{Path(source).stem}_*.wav"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        reference_clips.extend(clips[:2])
    if not reference_clips:
        print("No solo clips in voices/dmills/clips. Run voiceprint-solo first.", file=sys.stderr)
        return 1

    reference = build_reference(reference_clips, token)
    episode_end = max((t.end for t in turns), default=0.0)

    # Whole episode: one slice covering everything. select_slice_turns does
    # the real filtering (min duration, inside slice).
    scored_turns = select_slice_turns(turns, 0.0, episode_end)

    print(
        f"{args.episode}: scoring all {len(scored_turns)} turns against "
        f"{len(reference_clips)} solo reference clips (cutoff {args.cutoff})"
    )

    from pipeline.identify import _inference

    embed_turn = embed_turn_crop(_inference(token), str(audio))
    scores = score_turns(embed_turn, scored_turns, reference)
    reports = label_reports(scores)
    print(format_label_summary(reports))

    confirmed = [s for s in scores if s.score >= args.cutoff]
    if not confirmed:
        print(
            f"No turns scored at or above {args.cutoff}. Darrick may be absent "
            f"from this episode, or the cutoff is too high — compare with "
            f"verify-speakers output before forcing it.",
            file=sys.stderr,
        )
        return 1

    # Windows are written as diarization TURN spans (not bare start/end), so
    # ingest can reuse load_turns and the same Turn plumbing everywhere.
    out_dir = settings.data_dir / "darrick_turns"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.episode}.json"
    darrick_turns = turns_from_records(
        [
            {"speaker": "DARRICK", "start": s.start, "end": s.end}
            for s in sorted(confirmed, key=lambda s: s.start)
        ]
    )
    save_turns(out_path, darrick_turns)

    total = sum(t.duration for t in darrick_turns)
    print(f"\n{len(darrick_turns)} turns, {total / 60:.1f} min above cutoff "
          f"-> {out_path}")
    share = total / max(sum(t.duration for t in turns), 1e-9)
    print(f"share of episode speech: {share:.0%}")
    print("Next: ingest this episode. Windows apply at ingest time — diarization")
    print("caches stay untouched.")
    return 0


def format_label_summary(reports: Any) -> str:
    """One line per label: enough to judge the cutoff, small enough to skim."""
    lines = []
    for r in reports:
        flag = "  <-- mixed (bimodal)" if r.bimodal else ""
        lines.append(
            f"  {r.label}: {r.above_threshold}/{r.turn_count} above, "
            f"median {r.median:.3f}{flag}"
        )
    return "\n".join(lines)


def _voice_manifest(voice_dir: Path) -> list[dict[str, Any]]:
    """The voiceprint manifest, or [] when it doesn't exist yet."""
    import json

    manifest_path = voice_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    return list(json.loads(manifest_path.read_text(encoding="utf-8")))


def cmd_register_candidates(settings: Settings, args: argparse.Namespace) -> int:
    """Surface real speech-pattern candidates from ingested Canon, for review.

    Read-only over the Canon store and never touches brain/persona/register.md
    — it writes a report; a human decides what, if anything, to add. See
    pipeline/register_candidates.py for why that split is deliberate.
    """
    from brain.store.db import Store
    from pipeline.register_candidates import candidate_phrases, format_report, turn_openers
    from pipeline.transcript import parse_chunk_lines

    store = Store(settings.data_dir / "jarvis.db")
    texts = store.all_canon_text()
    if not texts:
        print("Canon is empty. Run: python -m pipeline ingest", file=sys.stderr)
        return 1

    lines = [pair for text in texts for pair in parse_chunk_lines(text)]
    speaker_line_count = sum(1 for s, _ in lines if s == args.speaker)
    if speaker_line_count == 0:
        print(
            f"No lines attributed to {args.speaker!r} in Canon. Check the "
            f"--speaker value, or that `pipeline identify` found him.",
            file=sys.stderr,
        )
        return 1

    phrases = candidate_phrases(lines, args.speaker)
    openers = turn_openers(lines, args.speaker)
    report = format_report(args.speaker, phrases, openers)

    out_path = settings.data_dir / "register_candidates.md"
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"(also written to {out_path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="what's done and what's next")

    p_list = sub.add_parser("list", help="list episodes without downloading")
    p_list.add_argument("--limit", type=int, default=20)

    p_fetch = sub.add_parser("fetch", help="download episode audio")
    p_fetch.add_argument("--limit", type=int, default=20)

    p_fetchref = sub.add_parser(
        "fetch-reference",
        help="download one arbitrary clip (any URL) as reference audio, outside the show",
    )
    p_fetchref.add_argument("url")

    p_fetchsolo = sub.add_parser(
        "fetch-solo",
        help="download videos from Darrick's own channel (the voiceprint source)",
    )
    p_fetchsolo.add_argument("channel", help="channel URL, e.g. https://www.youtube.com/@handle")
    p_fetchsolo.add_argument("--limit", type=int, default=3)

    sub.add_parser("diarize", help="diarize fetched episodes (slow, cached)")

    sub.add_parser("diarize-solo", help="diarize solo videos (fast — num_speakers=1)")

    p_sample = sub.add_parser("sample", help="export clips per speaker so you can ID them by ear")
    p_sample.add_argument("episode", help="episode stem, e.g. dQw4w9WgXcQ")
    p_sample.add_argument("--per-speaker", type=int, default=3)

    p_ref = sub.add_parser("reference", help="build Darrick's reference voice")
    p_ref.add_argument("clips", nargs="*", default=[], help="explicit clip paths")
    p_ref.add_argument("--from-episode", help="build from a diarized episode instead")
    p_ref.add_argument("--speaker", help="which label is Darrick, e.g. SPEAKER_02")

    sub.add_parser("identify", help="find Darrick's label in each episode")

    p_ingest = sub.add_parser("ingest", help="transcribe + chunk + embed into the Canon store")
    p_ingest.add_argument(
        "--episode",
        help="ingest only this episode stem (e.g. oocok912TfM). Default: all fetched episodes.",
    )

    p_verify = sub.add_parser(
        "verify-speakers",
        help="score every turn in an episode against Darrick's solo-voice reference",
    )
    p_verify.add_argument("episode", help="episode stem, e.g. oocok912TfM")
    p_verify.add_argument(
        "--minutes",
        type=float,
        default=15.0,
        help="length of the mid-episode slice to score (default: 15)",
    )

    p_export = sub.add_parser(
        "export-darrick-turns",
        help="verify a whole episode and write Darrick's windows for ingest",
    )
    p_export.add_argument("episode", help="episode stem, e.g. ioMY7oYuhA0")
    p_export.add_argument(
        "--cutoff",
        type=float,
        default=0.60,
        help="score above which a turn counts as Darrick (default: 0.60)",
    )

    p_reg = sub.add_parser(
        "register-candidates",
        help="surface real speech-pattern candidates from Canon, for you to review",
    )
    p_reg.add_argument("--speaker", default="DARRICK")

    p_vp = sub.add_parser("voiceprint", help="cut the clean training set")
    p_vp.add_argument("--minutes", type=float, default=45.0)

    p_vps = sub.add_parser(
        "voiceprint-solo", help="build the voiceprint from solo videos instead of the show"
    )
    p_vps.add_argument("--minutes", type=float, default=45.0)

    args = parser.parse_args(argv)
    settings = Settings.load()

    handlers = {
        "status": cmd_status, "list": cmd_list, "fetch": cmd_fetch,
        "fetch-reference": cmd_fetch_reference, "fetch-solo": cmd_fetch_solo,
        "diarize": cmd_diarize, "diarize-solo": cmd_diarize_solo,
        "sample": cmd_sample, "reference": cmd_reference,
        "identify": cmd_identify, "voiceprint": cmd_voiceprint,
        "voiceprint-solo": cmd_voiceprint_solo, "ingest": cmd_ingest,
        "verify-speakers": cmd_verify_speakers,
        "export-darrick-turns": cmd_export_darrick_turns,
        "register-candidates": cmd_register_candidates,
    }
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
