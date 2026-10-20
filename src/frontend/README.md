# src/frontend — React Drag & Drop UI

Stack: React 19 + Vite + react-dropzone + OpenSeadragon + compare slider.

## Quick start

```bash
# Node 24 LTS (installed via nvm on this box)
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && nvm use --lts

cd src/frontend
npm install
npm run dev          # http://localhost:3000 (proxies API to :8000)
```

Backend must be running on port 8000:

```bash
PYTHONPATH=. .venv/bin/uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

## Production build

```bash
npm run build        # -> src/frontend/dist
```

The FastAPI backend serves `src/frontend/dist` automatically when it exists
(`http://localhost:8000/`), and the Docker image builds it in a Node stage.

## Features

- Dropzone: `.png .jpg .tiff .svs .ome.tiff` (max 500 MB upload)
- Direction toggle: H&E → DAPI / DAPI → H&E
- Compare slider (input vs synthetic) + `SYNTHETIC` watermark
- Download PNG
- Whole-slide viewer (OpenSeadragon, lazy-loaded chunk): scroll to zoom, drag to pan
- Backend health badge

## Backend API

`direction` is a query parameter:

```bash
curl -X POST -F "file=@data/examples/he_example.png" \
  "http://localhost:8000/translate?direction=he2dapi" --output /tmp/out.png
```

## Docker Compose

From the repo root:

```bash
docker compose up --build      # backend :8000 (built UI + API), frontend dev :3000
```
