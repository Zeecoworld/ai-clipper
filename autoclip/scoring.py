"""Lightweight visual highlight detection (motion + visual activity)."""
import cv2
import numpy as np


def sample_scores(path, sample_every: float = 0.5) -> tuple[list[tuple[float, float]], float]:
    """Return ([(time, score)], duration). Uses grab() to skip decoding unneeded frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0.0
        step = max(1, int(round(fps * sample_every)))

        samples: list[tuple[float, float]] = []
        prev = None
        idx = 0
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
            else:
                ok = cap.grab()
                frame = None
            if not ok:
                break
            if frame is not None:
                gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(gray, prev))) / 255.0
                activity = float(np.std(gray)) / 64.0
                samples.append((idx / fps, motion * 0.75 + min(activity, 1.5) * 0.25))
                prev = gray
            idx += 1
        return samples, duration
    finally:
        cap.release()


def select_clips(
    samples: list[tuple[float, float]],
    duration: float,
    clip_count: int,
    clip_length: float,
    sample_every: float = 0.5,
) -> list[tuple[float, float]]:
    """Pick the strongest non-overlapping windows. Returns [(start, score)] sorted by start."""
    if not samples:
        return []
    clip_length = min(clip_length, duration)
    scores = np.array([s for _, s in samples])
    window = max(1, int(clip_length / sample_every))
    kernel = np.ones(window) / window
    smoothed = np.convolve(scores, kernel, mode="same")

    order = np.argsort(smoothed)[::-1]
    min_gap = clip_length * 0.8
    selected: list[tuple[float, float]] = []
    for i in order:
        center = samples[int(i)][0]
        start = max(0.0, center - clip_length / 2)
        if start + clip_length > duration:
            start = max(0.0, duration - clip_length)
        if all(abs(start - s) >= min_gap for s, _ in selected):
            selected.append((start, float(smoothed[int(i)])))
        if len(selected) >= clip_count:
            break
    selected.sort()
    return selected


# ------------------------------------------------------------ speech-aware scoring
HOOK_WORDS = {
    "secret", "never", "always", "mistake", "mistakes", "truth", "why", "how", "best", "worst",
    "biggest", "free", "stop", "warning", "crazy", "insane", "shocking", "actually", "everyone",
    "nobody", "money", "important", "learn", "lesson", "tip", "tips", "trick", "hack", "wrong",
    "amazing", "incredible", "imagine", "listen", "remember", "wait", "problem", "solution",
}


def speech_scores(words, times: list[float], radius: float = 3.0) -> list[tuple[float, float]]:
    """Score each sample time by speech density, hook words and emphasis within +/-radius seconds."""
    if not words:
        return [(t, 0.0) for t in times]
    starts = np.array([w.start for w in words])
    hits = np.array([
        (w.text.lower().strip(".,!?\"'") in HOOK_WORDS) + 1.5 * (w.text[-1:] in "?!") + 0.5 * any(c.isdigit() for c in w.text)
        for w in words
    ])
    csum = np.concatenate([[0.0], np.cumsum(hits)])
    out = []
    for t in times:
        lo, hi = np.searchsorted(starts, t - radius), np.searchsorted(starts, t + radius)
        density = (hi - lo) / (2 * radius)                      # words per second
        out.append((t, 0.5 * min(density / 3.0, 1.0) + 0.5 * min((csum[hi] - csum[lo]) / 3.0, 1.0)))
    return out


def blend_scores(motion: list[tuple[float, float]], speech: list[tuple[float, float]],
                 speech_weight: float) -> list[tuple[float, float]]:
    """Combine normalised motion and speech scores. speech_weight 0 = motion only, 1 = speech only."""
    peak = max((s for _, s in motion), default=0.0) or 1.0
    return [(t, (1 - speech_weight) * (m / peak) + speech_weight * sp)
            for (t, m), (_, sp) in zip(motion, speech)]
