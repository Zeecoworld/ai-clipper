"""Thin, safe wrappers around ffmpeg / ffprobe."""
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"{cmd[0]} is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{cmd[0]} timed out after {timeout}s") from exc
    if result.returncode != 0:
        raise MediaError(result.stderr[-3000:] or f"{cmd[0]} failed")
    return result


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    has_audio: bool
    timeline: list = field(default_factory=list)  # [(seconds, activity score)]


def probe(path: Path) -> VideoInfo:
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("No video stream found in this file.")
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise MediaError("Could not determine video duration.")
    return VideoInfo(
        duration=duration,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )
