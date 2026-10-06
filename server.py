"""AutoClip AI – FastAPI server. Serves the HTML UI and a small JSON API (no auth)."""
import json
import os
import re
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from autoclip import config
from autoclip.captions import CaptionStyle
from autoclip.pipeline import cleanup_old_jobs, new_job_dir, process

STATIC = Path(__file__).parent / "static"
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
LANGS = {"en", "es", "fr", "de", "pt", "it", "hi", "ar", "ja", "ko", "zh"}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

app = FastAPI(title="AutoClip AI", docs_url=None, redoc_url=None)
SEM = threading.BoundedSemaphore(config.MAX_CONCURRENT_JOBS)


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued…"
    error: str = ""
    duration: float = 0.0
    timeline: list = field(default_factory=list)
    results: list = field(default_factory=list)
    updated: float = 0.0


JOBS: dict[str, Job] = {}


def _purge() -> None:
    cleanup_old_jobs()
    for jid in [j for j, job in JOBS.items() if not job.dir.exists()]:
        JOBS.pop(jid, None)


def _options(raw: str) -> dict:
    try:
        o = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "Invalid options.")

    def num(key, lo, hi, default):
        try:
            v = int(float(o.get(key, default)))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    def color(key, default):
        v = str(o.get(key, default))
        return v if HEX.match(v) else default

    pos = o.get("position", "bottom")
    style = CaptionStyle(
        position=pos if pos in ("bottom", "center", "top") else "bottom",
        words_per_line=num("words_per_line", 1, 6, 3),
        font_size=num("font_size", 40, 130, 78),
        uppercase=bool(o.get("uppercase", True)),
        text_color=color("text_color", "#FFFFFF"),
        highlight_color=color("highlight_color", "#FFD400"),
        highlight_active_word=bool(o.get("highlight", True)),
        hook_text=str(o.get("hook", ""))[:80],
        hook_seconds=float(num("hook_seconds", 0, 20, 3)),
    )
    lang = o.get("language") or None
    return dict(
        clip_count=num("clip_count", 1, 10, 3), clip_length=num("clip_length", 5, 60, 20),
        blur=num("blur", 5, 51, 25), captions=bool(o.get("captions", True)) and config.CAPTIONS_ENABLED,
        language=lang if lang in LANGS else None, style=style,
        highlight_mode=o.get("highlight_mode") if o.get("highlight_mode") in ("motion", "balanced", "speech") else "motion")


JOB_ID = re.compile(r"^job_\d+_[0-9a-f]{8}$")
_FIELDS = ("status", "progress", "message", "error", "duration", "timeline", "results", "updated")
STALE_SECONDS = 90  # a running job refreshes job.json every 15 s; silence means its process died


def _save(job: Job) -> None:
    """Persist job state next to its files so any worker (or a restarted server) can serve it."""
    job.updated = time.time()
    tmp = job.dir / "job.json.tmp"
    try:
        tmp.write_text(json.dumps({k: getattr(job, k) for k in _FIELDS}))
        os.replace(tmp, job.dir / "job.json")
    except OSError:
        pass


def _heartbeat(job: Job, stop: threading.Event) -> None:
    while not stop.wait(15):
        _save(job)


def _load(job_id: str) -> "Job | None":
    if not JOB_ID.match(job_id):
        return None
    d = config.WORK_ROOT / job_id
    try:
        data = json.loads((d / "job.json").read_text())
    except (OSError, ValueError):
        return None
    job = Job(id=job_id, dir=d)
    for k in _FIELDS:
        if k in data:
            setattr(job, k, data[k])
    if job.status in ("queued", "running") and time.time() - job.updated > STALE_SECONDS:
        job.status = "error"
        job.error = "The server restarted or ran out of memory while processing. Please upload the video again."
    return job


def _run(job: Job, src: Path, opts: dict) -> None:
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(job, stop), daemon=True).start()
    try:
        if not SEM.acquire(blocking=False):
            job.message = "Another job is rendering — you're in the queue…"
            _save(job)
            SEM.acquire()
        try:
            job.status = "running"
            _save(job)

            def progress(f: float, m: str) -> None:
                job.progress, job.message = min(max(f, 0.0), 1.0), m
                _save(job)

            info, results = process(src, job.dir, opts["clip_count"], opts["clip_length"], opts["blur"],
                                    opts["captions"], opts["language"], opts["style"], progress=progress,
                                    highlight_mode=opts["highlight_mode"])
            job.duration = info.duration
            step = max(1, len(info.timeline) // 240)
            pts = info.timeline[::step]
            peak = max((s for _, s in pts), default=1.0) or 1.0
            job.timeline = [[round(t, 2), round(s / peak, 3)] for t, s in pts]
            job.results = [dict(index=r.index, start=round(r.start, 2), length=round(r.length, 2),
                                score=round(r.score, 3), caption_words=r.caption_words,
                                caption_note=r.caption_note, url=f"/api/jobs/{job.id}/clips/{r.index}")
                           for r in results]
            job.status, job.progress, job.message = "done", 1.0, "Done"
        except Exception as exc:  # surface any failure to the UI
            job.status, job.error = "error", str(exc)[-600:]
        finally:
            SEM.release()
    finally:
        stop.set()
        _save(job)
        src.unlink(missing_ok=True)


def _job(job_id: str) -> Job:
    job = JOBS.get(job_id) or _load(job_id)
    if job is None or not job.dir.exists():
        raise HTTPException(404, "Job not found or expired. Upload the video again.")
    return job


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    return {"max_upload_mb": config.MAX_UPLOAD_MB, "captions": config.CAPTIONS_ENABLED,
            "ttl_minutes": config.JOB_TTL_MINUTES}


@app.post("/api/jobs")
def create_job(request: Request, file: UploadFile = File(...), options: str = Form("{}")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Unsupported file type. Use mp4, mov, mkv, avi or webm.")
    declared = int(request.headers.get("content-length") or 0)
    if declared > (config.MAX_UPLOAD_MB + 5) * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    opts = _options(options)
    _purge()
    job_dir = new_job_dir()
    src = job_dir / f"source{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)
    if src.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    job = Job(id=job_dir.name, dir=job_dir)
    JOBS[job.id] = job
    _save(job)
    threading.Thread(target=_run, args=(job, src, opts), daemon=True).start()
    return {"id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _job(job_id)
    return {"status": job.status, "progress": job.progress, "message": job.message,
            "error": job.error, "duration": job.duration, "timeline": job.timeline,
            "results": job.results}


@app.get("/api/jobs/{job_id}/clips/{n}")
def get_clip(job_id: str, n: int, download: int = 0):
    path = _job(job_id).dir / "clips" / f"clip_{n:02d}.mp4"
    if not path.exists():
        raise HTTPException(404, "Clip not found.")
    return FileResponse(path, media_type="video/mp4",
                        filename=path.name if download else None,
                        content_disposition_type="attachment" if download else "inline")


@app.get("/api/jobs/{job_id}/zip")
def get_zip(job_id: str):
    job = _job(job_id)
    clips = sorted((job.dir / "clips").glob("clip_*.mp4"))
    if not clips:
        raise HTTPException(404, "No clips to download.")
    z = job.dir / "autoclip_clips.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for c in clips:
            zf.write(c, c.name)
    return FileResponse(z, media_type="application/zip", filename="autoclip_clips.zip")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.pop(job_id, None)
    if job:
        shutil.rmtree(job.dir, ignore_errors=True)
    return JSONResponse({"ok": True})
