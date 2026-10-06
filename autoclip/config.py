"""Environment-driven settings. Everything tunable for deployment lives here."""
import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


WORK_ROOT = Path(os.environ.get("AUTOCLIP_WORK_DIR", "/tmp/autoclip"))
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 500)
MAX_CONCURRENT_JOBS = _int("MAX_CONCURRENT_JOBS", 1)
JOB_TTL_MINUTES = _int("JOB_TTL_MINUTES", 60)
FFMPEG_THREADS = _int("FFMPEG_THREADS", 2)  # each extra thread costs RAM; 0 = all cores
WHISPER_THREADS = _int("WHISPER_THREADS", 2)
WHISPER_CHUNK_SECONDS = _int("WHISPER_CHUNK_SECONDS", 300)  # transcribe long videos in pieces to cap memory
X264_PRESET = os.environ.get("X264_PRESET", "veryfast")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_CACHE = os.environ.get("WHISPER_CACHE", "/opt/whisper-models")
CAPTIONS_ENABLED = os.environ.get("CAPTIONS_ENABLED", "1") != "0"
