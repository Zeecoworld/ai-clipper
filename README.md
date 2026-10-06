# AutoClip AI

Upload a long video → get vertical 9:16 clips (blurred background) with **auto-generated,
word-highlighted captions** and an optional **hook/title text**.

## Stack
FastAPI backend (`server.py`) + a single-page HTML/CSS/JS interface (`static/index.html`). No Streamlit, no login.
The UI offers drag-and-drop upload, a live 9:16 caption preview, an activity timeline showing which moments were picked,
progress with queueing, per-clip preview/download and a ZIP of all clips.

## Pipeline features
- **Captions**: speech-to-text via `faster-whisper` (CPU, int8), burned in with ffmpeg/libass. Word-by-word highlight, position, size, colors, uppercase, words-per-line, language.
- **Hook text**: a text box shown at the top of each clip for N seconds (or the whole clip).
- **Fixes**: results persist on the server per job; faster scoring (skips decoding between samples); handles videos with no audio; headless OpenCV (no libGL needed).
- **Production**: Dockerfile (ffmpeg + fonts + Whisper model baked in), `render.yaml`, health check, non-root user, upload size limit, job queue (one render at a time by default), temp-file cleanup, no login screen.

## Deploy to Render
1. Push this folder to a GitHub/GitLab repo.
2. Render Dashboard → **New → Blueprint** → select the repo (it reads `render.yaml`).
3. Deploy. Note the app has no authentication, so anyone with the URL can use your CPU; put it behind a private network or proxy auth if that matters.
4. Wait for the build. First build takes a few minutes (downloads the Whisper model into the image).

### Sizing
| Plan | RAM | Use with |
|------|-----|----------|
| standard (default in render.yaml) | 2 GB | `WHISPER_MODEL=base`, videos up to ~1 hr |
| starter | 512 MB | `WHISPER_MODEL=tiny` (set as Docker build arg / env), short videos, `MAX_UPLOAD_MB=200` |

To change the model, set the build arg `WHISPER_MODEL` (tiny, base, small) in Render → Settings → Docker Build Args
(it must match the runtime env var because the model is baked into the image).

### Notes
- Render's disk is ephemeral: finished clips live in `/tmp` for `JOB_TTL_MINUTES` (default 60) and are deleted. Download what you need.
- Render's proxy may time out very long single HTTP requests, but the UI uploads once and then polls for progress, so long renders are fine. Rendering speed is CPU bound; expect roughly 1–2× real time of the clip length on a standard instance.

## Environment variables
| Var | Default | Meaning |
|-----|---------|---------|
| `MAX_UPLOAD_MB` | 500 | Upload limit |
| `MAX_CONCURRENT_JOBS` | 1 | Simultaneous renders (others queue) |
| `JOB_TTL_MINUTES` | 60 | Auto-delete finished jobs |
| `WHISPER_MODEL` | base | tiny / base / small |
| `CAPTIONS_ENABLED` | 1 | Set 0 to disable captions entirely |
| `X264_PRESET` | veryfast | ffmpeg speed/quality tradeoff |

## Run locally
```bash
# needs ffmpeg on PATH
pip install -r requirements.txt
uvicorn server:app --port 10000   # then open http://localhost:10000
# or: docker build -t autoclip . && docker run -p 10000:10000 autoclip
```

## Tests
```bash
python -m unittest discover tests   # needs ffmpeg + opencv, not Whisper
```

## Picking moments
"Pick moments by" in the UI: **Motion only** (fastest), **Motion and speech**, or **Speech first**. Speech modes transcribe the
whole video once, score each moment by speech density, questions/emphasis, numbers and hook words, blend that with motion,
and reuse the same transcript for captions. If transcription fails it falls back to motion only.

## Limitations
Speech scoring is keyword/density based, not an LLM. Good next steps: LLM-ranked moments, face/subject tracking for smart
cropping, editable transcripts before rendering.
