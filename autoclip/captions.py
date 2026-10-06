"""Speech-to-text (faster-whisper) and ASS caption generation with word highlighting."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .media import run


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class CaptionStyle:
    font: str = "DejaVu Sans"
    font_size: int = 78
    position: str = "bottom"          # bottom | center | top
    words_per_line: int = 3
    uppercase: bool = True
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    outline_color: str = "#000000"
    outline_width: int = 6
    highlight_active_word: bool = True
    hook_text: str = ""
    hook_seconds: float = 3.0         # 0 = show for the whole clip
    hook_font_size: int = 64
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- transcription
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel  # lazy: heavy import

            _model = WhisperModel(
                config.WHISPER_MODEL, device="cpu", compute_type="int8",
                download_root=config.WHISPER_CACHE, cpu_threads=config.WHISPER_THREADS, num_workers=1,
            )
        return _model


def extract_audio(src: Path, out_wav: Path, start: float, length: float) -> Path:
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav),
    ])
    return out_wav


def transcribe(wav: Path, language: str | None = None) -> list[Word]:
    model = _get_model()
    segments, _info = model.transcribe(
        str(wav), language=language or None, word_timestamps=True,
        vad_filter=True, beam_size=1,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(text, float(w.start), float(w.end)))
    return words


def slice_words(words: list[Word], start: float, length: float) -> list[Word]:
    """Words inside [start, start+length], re-timed so the clip begins at 0."""
    end = start + length
    return [Word(w.text, max(0.0, w.start - start), min(length, w.end - start))
            for w in words if w.end > start and w.start < end]


def transcribe_long(src: Path, work_dir: Path, duration: float, language: str | None = None,
                    transcribe_fn=None, progress=None) -> list[Word]:
    """Transcribe a long video in fixed-size chunks so memory stays flat regardless of length."""
    transcribe_fn = transcribe_fn or transcribe
    words: list[Word] = []
    t = 0.0
    while t < duration:
        length = min(float(config.WHISPER_CHUNK_SECONDS), duration - t)
        wav = extract_audio(src, work_dir / "chunk.wav", t, length)
        try:
            words += [Word(w.text, w.start + t, w.end + t) for w in transcribe_fn(wav, language)]
        finally:
            wav.unlink(missing_ok=True)
        t += length
        if progress:
            progress(min(1.0, t / duration))
    return words


# ------------------------------------------------------------------- ASS output
def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    return (text.replace("\\", "").replace("{", "(").replace("}", ")")
            .replace("\r", " ").replace("\n", "\\N"))


_ALIGN = {"bottom": (2, 380), "center": (5, 0), "top": (8, 260)}


def group_words(words: list[Word], per_line: int) -> list[list[Word]]:
    per_line = max(1, per_line)
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        ends_sentence = w.text[-1:] in ".?!"
        if len(cur) >= per_line or ends_sentence:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[Word], style: CaptionStyle, clip_length: float,
              width: int = 1080, height: int = 1920) -> str:
    align, margin_v = _ALIGN.get(style.position, _ALIGN["bottom"])
    primary = _ass_color(style.text_color)
    outline = _ass_color(style.outline_color)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{style.font},{style.font_size},{primary},{primary},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},2,{align},60,60,{margin_v},1
Style: Hook,{style.font},{style.hook_font_size},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,70,70,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []

    if style.hook_text.strip():
        end = clip_length if style.hook_seconds <= 0 else min(style.hook_seconds, clip_length)
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(end)},Hook,,0,0,0,,{_esc(style.hook_text.strip())}")

    hl = _ass_color(style.highlight_color)
    for group in group_words(words, style.words_per_line):
        texts = [(w.text.upper() if style.uppercase else w.text) for w in group]
        for i, w in enumerate(group):
            start = w.start
            if i + 1 < len(group):
                end = group[i + 1].start
            else:
                end = max(w.end, w.start + 0.15)
            end = min(end, clip_length)
            if end <= start:
                continue
            if style.highlight_active_word:
                parts = []
                for j, t in enumerate(texts):
                    t = _esc(t)
                    parts.append(f"{{\\c{hl}}}{t}{{\\c{primary}}}" if j == i else t)
                body = " ".join(parts)
            else:
                body = " ".join(_esc(t) for t in texts)
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Caption,,0,0,0,,{body}")

    return header + "\n".join(lines) + "\n"
