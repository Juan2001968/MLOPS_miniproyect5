
# 08. Empaquetado con Docker y CI/CD (GitHub Actions)

## 8.1. Estructura

```
miniproyecto6/
├─ src/
│  └─ app/
│     └─ main.py
├─ models/
│  ├─ preprocessor.joblib
│  ├─ xgb_clf_only.joblib
│  └─ feature_names.json   # opcional
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ .github/
   └─ workflows/
      ├─ docker-publish.yml
      └─ docker-smoke.yml
```

---

## 8.2. Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 8.3. .dockerignore

```gitignore
**/__pycache__/
**/*.pyc
**/*.pyo
**/*.pyd
**/.pytest_cache
**/.mypy_cache
**/.venv
venv
ml_venv
.env
*.ipynb_checkpoints
docs/_build
data/
```

> **Nota:** No ignores `models/`, los artefactos deben ir dentro de la imagen.

---

## 8.4. Probar localmente

```bash
docker build -t churn-api:latest .
docker run --rm -p 8000:8000 churn-api:latest
```

- Swagger: <http://localhost:8000/docs>  
- Salud: <http://localhost:8000/health>

---

## 8.5. Publicación de la imagen

### Secrets necesarios (en GitHub → Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME` = *tu usuario real de Docker Hub* (normalmente en minúsculas).
- `DOCKERHUB_TOKEN` = *access token* creado en Docker Hub (Settings → Security).

*(Si prefieres GHCR, puedes usar `GITHUB_TOKEN` y cambiar el `REGISTRY` a `ghcr.io` en el workflow).*

---

## 8.6. Workflow: build & push

Archivo: `.github/workflows/docker-publish.yml`

```yaml
name: Build & Push Docker Image

on:
  push:
    branches: [ "main" ]
    tags: [ "v*.*.*" ]
  workflow_dispatch:

env:
  IMAGE_NAME: churn-api
  REGISTRY: docker.io
  DOCKERHUB_NAMESPACE: ${{ secrets.DOCKERHUB_USERNAME }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            ${{ env.REGISTRY }}/${{ env.DOCKERHUB_NAMESPACE }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha
            type=ref,event=tag

      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.DOCKERHUB_NAMESPACE }}/${{ env.IMAGE_NAME }}:buildcache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.DOCKERHUB_NAMESPACE }}/${{ env.IMAGE_NAME }}:buildcache,mode=max
```

**Resultado:** cada *push* a `main` (y tags `vX.Y.Z`) construye y sube tu imagen:
- `docker.io/tu_usuario/churn-api:latest`
- `docker.io/tu_usuario/churn-api:sha-<commit>`
- `docker.io/tu_usuario/churn-api:vX.Y.Z` (si creas un tag)

---

## 8.7. Workflow: smoke test

Archivo: `.github/workflows/docker-smoke.yml`

```yaml
name: Docker Smoke Test

on:
  workflow_dispatch:
  push:
    branches: [ "main" ]

env:
  IMAGE_REF: docker.io/${{ secrets.DOCKERHUB_USERNAME }}/churn-api:latest

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - name: Pull image
        run: docker pull $IMAGE_REF

      - name: Run container
        run: docker run -d --rm -p 8000:8000 --name churn-smoke $IMAGE_REF

      - name: Wait for service
        run: |
          for i in {1..30}; do
            if curl -sSf http://localhost:8000/health > /dev/null; then
              echo "Service is up!"
              exit 0
            fi
            echo "Waiting..."
            sleep 2
          done
          echo "Service did not start" && exit 1

      - name: Curl predict (opcional)
        run: |
          curl -sSf -X POST "http://localhost:8000/predict"             -H "Content-Type: application/json"             -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
                 "tenure":5,"PhoneService":"Yes","MultipleLines":"No",
                 "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
                 "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
                 "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
                 "PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":351.75}'
```

---

## 8.8. Versionado / tags

```bash
git tag v1.0.0
git push origin v1.0.0
```

Se publicará `:v1.0.0` además de `:latest` y `:sha-...`.

---

## 8.9. Troubleshooting

- **Auth en Docker Hub**: revisa `DOCKERHUB_USERNAME` (usuario correcto) y `DOCKERHUB_TOKEN` (Access Token, no tu password).
- **Artefactos no encontrados**: checa que `models/` esté en el repo y que `main.py` lea `/app/models/...`.
- **Versiones sklearn**: usa la misma versión con la que entrenaste el preprocesador (ya fijada en `requirements.txt` y funcionó).

---
