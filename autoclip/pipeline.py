"""High-level job orchestration, independent of the UI so it can be tested."""
from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config
from .captions import CaptionStyle, build_ass, extract_audio, slice_words, transcribe
from .media import VideoInfo, probe
from .render import make_vertical_clip
from .scoring import blend_scores, sample_scores, select_clips, speech_scores

Progress = Callable[[float, str], None]

# How much transcript content counts versus visual motion when choosing moments.
SPEECH_WEIGHT = {"motion": 0.0, "balanced": 0.5, "speech": 0.85}


@dataclass
class ClipResult:
    index: int
    start: float
    length: float
    score: float
    path: Path
    caption_words: int = 0
    caption_note: str = ""


def new_job_dir() -> Path:
    config.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    d = config.WORK_ROOT / f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return d


def cleanup_old_jobs(ttl_minutes: int | None = None) -> None:
    ttl = (ttl_minutes if ttl_minutes is not None else config.JOB_TTL_MINUTES) * 60
    if not config.WORK_ROOT.exists():
        return
    now = time.time()
    for d in config.WORK_ROOT.glob("job_*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > ttl:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def process(src: Path, job_dir: Path, clip_count: int, clip_length: int, blur: int,
            captions: bool, language: str | None, style: CaptionStyle,
            progress: Progress | None = None, highlight_mode: str = "motion") -> tuple[VideoInfo, list[ClipResult]]:
    progress = progress or (lambda f, m: None)
    info = probe(src)
    progress(0.02, "Analyzing video for high-activity moments…")
    samples, _ = sample_scores(src)
    info.timeline = samples
    all_words = None
    weight = SPEECH_WEIGHT.get(highlight_mode, 0.0)
    if weight > 0 and info.has_audio and config.CAPTIONS_ENABLED:
        progress(0.04, "Transcribing the whole video to find the best moments…")
        try:
            wav = extract_audio(src, job_dir / "full.wav", 0.0, info.duration)
            try:
                all_words = transcribe(wav, language)
            finally:
                wav.unlink(missing_ok=True)
            samples = blend_scores(samples, speech_scores(all_words, [t for t, _ in samples]), weight)
            info.timeline = samples
        except Exception:  # fall back to motion-only picking
            all_words = None
    picks = select_clips(samples, info.duration, clip_count, clip_length)
    if not picks:
        return info, []

    out_dir = job_dir / "clips"
    out_dir.mkdir(exist_ok=True)
    want_captions = captions and info.has_audio and config.CAPTIONS_ENABLED
    results: list[ClipResult] = []

    for n, (start, score) in enumerate(picks, 1):
        length = min(float(clip_length), info.duration - start)
        base = 0.10 + 0.88 * (n - 1) / len(picks)
        span = 0.88 / len(picks)
        res = ClipResult(n, start, length, score, out_dir / f"clip_{n:02d}.mp4")
        ass_path = None

        if want_captions or style.hook_text.strip():
            words = []
            if want_captions and all_words is not None:
                words = slice_words(all_words, start, length)  # reuse the full transcript
                res.caption_words = len(words)
                if not words:
                    res.caption_note = "No speech detected."
            elif want_captions:
                progress(base, f"Clip {n}/{len(picks)}: transcribing speech…")
                try:
                    wav = extract_audio(src, out_dir / f"audio_{n:02d}.wav", start, length)
                    words = transcribe(wav, language)
                    wav.unlink(missing_ok=True)
                    res.caption_words = len(words)
                    if not words:
                        res.caption_note = "No speech detected."
                except Exception as exc:  # captions must never kill the whole job
                    res.caption_note = f"Captioning skipped: {exc}"[:300]
            ass_path = out_dir / f"cap_{n:02d}.ass"
            ass_path.write_text(build_ass(words, style, length), encoding="utf-8")
        elif captions and not info.has_audio:
            res.caption_note = "Video has no audio track; captions skipped."

        progress(base + span * 0.4, f"Clip {n}/{len(picks)}: rendering…")
        make_vertical_clip(src, res.path, start, length, blur=blur,
                           ass_file=ass_path, has_audio=info.has_audio)
        results.append(res)
        progress(base + span, f"Clip {n}/{len(picks)} done")

    progress(1.0, "Done")
    return info, results
