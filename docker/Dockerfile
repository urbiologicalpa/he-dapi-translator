# he-dapi-translator Dockerfile
# Build: docker build -t he-dapi-translator -f docker/Dockerfile .
# Run: docker run -p 8000:8000 -v $PWD/resources/models/checkpoints:/app/resources/models/checkpoints:ro he-dapi-translator

# 1) Frontend build (React + Vite -> static dist served by FastAPI)
FROM node:24-slim AS frontend-build
WORKDIR /app/src/frontend
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci
COPY src/frontend/ ./
RUN npm run build

# 2) Backend runtime
FROM python:3.10-slim
LABEL org.opencontainers.image.title="he-dapi-translator" \
      org.opencontainers.image.description="Bidirectional H&E<->DAPI virtual staining (Pix2Pix + CycleGAN), patch + WSI" \
      org.opencontainers.image.source="https://github.com/urbiologicalpa/he-dapi-translator" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools libopenslide0 libgl1 libglib2.0-0 curl \
    gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend dependencies
COPY docker/requirements.txt /app/docker/requirements.txt
RUN pip install --no-cache-dir -r docker/requirements.txt

# Copy source + built frontend
COPY src/ /app/src/
COPY resources/ /app/resources/
COPY data/examples/ /app/data/examples/
COPY --from=frontend-build /app/src/frontend/dist /app/src/frontend/dist

# Weights are mounted at runtime (not baked in): -v .../checkpoints:/app/resources/models/checkpoints:ro
EXPOSE 8000
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
