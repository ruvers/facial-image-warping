# Facial Image Transform

Full-stack app for facial image warping, aging simulation, and expression transformation using computer vision models.

**Stack:** Python, FastAPI, MediaPipe, OpenCV, HTML/CSS/JS

## Features

- Real-time face landmark detection and warping
- Aging and expression transformation pipelines
- Accessory and appearance editing tools
- REST API with integrated web frontend

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or: `chmod +x dev.sh && ./dev.sh`

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notes

- Model weights in `local_models/` must be downloaded separately (see `docs/`)
- `.env` is not committed — copy from project docs for local configuration
