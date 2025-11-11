# src/app/main.py
from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Union, Optional

import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException

# =========================================================
# Rutas de proyecto y artefactos
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../miniproyecto6
MODELS_DIR   = PROJECT_ROOT / "models"
PREPROCESSOR = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH   = MODELS_DIR / "xgb_clf_only.joblib"   # <--- CLASIFICADOR SOLO
FEATURE_NAMES_JSON = MODELS_DIR / "feature_names.json"  # nombres transformados (opcional)

# =========================================================
# Globals (se cargan en startup)
# =========================================================
preprocessor = None        # ColumnTransformer / Pipeline de preprocesamiento
model = None               # XGBClassifier (solo el clf)
raw_features: Optional[List[str]] = None         # columnas crudas esperadas por el preprocesador
transformed_features: Optional[List[str]] = None # columnas post-transform (opt)

VERSION = "0.1.0"

# =========================================================
# Ciclo de vida: carga artefactos al arrancar
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model, raw_features, transformed_features

    # Cargar artefactos
    try:
        preprocessor = joblib.load(PREPROCESSOR)
        model        = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error cargando artefactos: {e}")

    # Columnas crudas esperadas por el preprocesador
    try:
        raw_features = list(preprocessor.feature_names_in_)  # sklearn >=1.0
    except Exception:
        raw_features = None

    # Nombres transformados (si existen)
    if FEATURE_NAMES_JSON.exists():
        try:
            with open(FEATURE_NAMES_JSON, "r", encoding="utf-8") as f:
                transformed_features = json.load(f)
        except Exception:
            transformed_features = None

    print("========================================")
    print("[DEBUG] PROJECT_ROOT :", PROJECT_ROOT)
    print("[DEBUG] MODELS_DIR   :", MODELS_DIR)
    print("[DEBUG] PREPROCESSOR :", PREPROCESSOR)
    print("[DEBUG] MODEL_PATH   :", MODEL_PATH)
    print("[DEBUG] RAW FEATURES :", len(raw_features) if raw_features else "N/A")
    print("[DEBUG] XGB loaded   :", model.__class__.__name__)
    print("========================================")

    yield

# =========================================================
# App
# =========================================================
app = FastAPI(
    title="Churn API (FastAPI)",
    description="Servicio de inferencia para churn: preprocesa + predice con XGB.",
    version=VERSION,
    lifespan=lifespan
)

# =========================================================
# Utilidades
# =========================================================
def _ensure_artifacts_loaded():
    if preprocessor is None or model is None:
        raise HTTPException(status_code=503, detail="Artefactos no cargados.")

def _to_dataframe(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Acepta dict (1 registro) o lista de dicts (batch) y retorna DataFrame."""
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="Lista vacía en payload.")
        records = payload
    else:
        raise HTTPException(status_code=400, detail="Formato de payload inválido.")

    df = pd.DataFrame.from_records(records)

    # Si conocemos las columnas crudas esperadas por el preprocesador,
    # completamos faltantes con NaN, reordenamos y descartamos extras
    if raw_features:
        missing = [c for c in raw_features if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df = df[raw_features]

    return df

def _predict_dataframe(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """Transforma con el preprocesador, predice con el clasificador y devuelve proba/label."""
    try:
        Xt = preprocessor.transform(df)
        proba = model.predict_proba(Xt)[:, 1]
        label = (proba >= threshold).astype(int)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante transformación/predicción: {e}")

    return {
        "proba_churn": proba.tolist(),
        "label": label.tolist(),
        "n_items": len(df),
    }

# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    ok = (preprocessor is not None) and (model is not None)
    return {"status": "ok" if ok else "down"}

@app.get("/version")
def version():
    return {"version": VERSION}

@app.get("/schema")
def schema():
    """Devuelve las columnas crudas esperadas y (opcional) nombres transformados."""
    _ensure_artifacts_loaded()
    return {
        "raw_features": raw_features if raw_features else "desconocidas",
        "transformed_features": transformed_features if transformed_features else "desconocidas"
    }

@app.post("/predict")
def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]], threshold: float = 0.5):
    """
    Recibe:
    - Un objeto JSON (1 registro), o
    - Una lista de objetos JSON (batch)
    Devuelve probabilidades y labels.
    """
    _ensure_artifacts_loaded()
    df = _to_dataframe(payload)
    out = _predict_dataframe(df, threshold=threshold)
    return out

# (Opcional) endpoint que solo devuelve probabilidades
@app.post("/predict_proba")
def predict_proba(payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
    _ensure_artifacts_loaded()
    df = _to_dataframe(payload)
    try:
        Xt = preprocessor.transform(df)
        proba = model.predict_proba(Xt)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante transformación/predicción: {e}")
    return {"proba_churn": proba.tolist(), "n_items": len(df)}
