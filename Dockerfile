FROM python:3.11-slim

# Evita .pyc y hace flush inmediato de stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Algunas libs (xgboost/lightgbm) usan OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y artefactos
COPY src ./src
COPY models ./models

# Puerto de la API
EXPOSE 8000

# Arranque
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

