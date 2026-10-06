"""ffmpeg rendering of 9:16 clips with blurred background and optional ASS captions."""
from pathlib import Path

from . import config
from .media import run

W, H = 1080, 1920


def make_vertical_clip(src: Path, out: Path, start: float, length: float,
                       blur: int = 22, ass_file: Path | None = None,
                       has_audio: bool = True) -> None:
    graph = (
        "split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma={blur}[blur];"
        f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[main];"
        "[blur][main]overlay=(W-w)/2:(H-h)/2,format=yuv420p"
    )
    cwd = None
    if ass_file is not None:
        # Run inside the ASS file's folder and reference it by bare filename:
        # avoids all ffmpeg filter-path escaping problems.
        cwd = ass_file.parent
        graph += f",ass={ass_file.name}"

    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-vf", graph, "-c:v", "libx264", "-preset", config.X264_PRESET, "-crf", "21",
           "-threads", str(config.FFMPEG_THREADS), "-r", "30"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(out)]
    run(cmd, cwd=cwd)
