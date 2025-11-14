# 🚀 Proyecto MLOps de Predicción de Churn en Telecomunicaciones  
**FastAPI + Docker + CI/CD + Dashboard + Jupyter Book**

---

## 📌 Descripción General

Este proyecto implementa un **pipeline MLOps completo** para predecir *churn* (abandono de clientes) en una empresa de telecomunicaciones.  
Incluye:

- Entrenamiento del modelo (XGBoost).
- API REST para inferencia usando **FastAPI**.
- Contenerización con **Docker**.
- Pipeline de integración continua (**GitHub Actions**).
- Dashboard interactivo con **Plotly Dash**.
- Documentación generada con **Jupyter Book**.
- Esquemas, validación y versionado del servicio.

Todo el flujo está diseñado para cumplir con las buenas prácticas del ciclo de vida de un modelo en producción.

---

# 📁 Estructura del Proyecto

```plaintext
proyecto-churn-mlops/
├── notebooks/                     # ETL, EDA, modelado, experimentación
│   └── ...                        # Notebooks del proyecto
│
├── models/                        # Artefactos entrenados (Joblib, JSON)
│   ├── xgb_clf_only.joblib        # Modelo final XGBClassifier
│   └── feature_names.json         # (opcional) Nombre de features transformadas
│
├── src/
│   ├── app/                       # API de inferencia (FastAPI)
│   │   ├── main.py                # Endpoints /health, /version, /schema, /predict...
│   │   ├── schemas.py             # Esquemas Pydantic para requests/responses
│   │   └── __init__.py
│   │
│   ├── dashboard/                 # Dashboard interactivo (Plotly Dash)
│   │   ├── app.py                 # Aplicación Dash
│   │   ├── assets/
│   │   │   └── styles.css         # Estilos personalizados
│   │   └── __init__.py
│   │
│   ├── preprocessing/             # Transformaciones del dataset
│   │   └── ...                    # Pipelines, funciones de limpieza, etc.
│   │
│   └── utils/                     # Funciones auxiliares
│       └── ...
│
├── docs/                          # Jupyter Book (documentación)
│   ├── _config.yml
│   ├── _toc.yml
│   ├── 01_contexto_churn.md
│   ├── 02_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_modelado_arboles_gridsearch.ipynb
│   ├── 05_evaluacion_metricas_clasificacion.ipynb
│   ├── 06_interpretabilidad.ipynb
│   ├── 08_docker_ci_cd.md
│   ├── 09_dashboard_dash.ipynb
│   └── 10_monitoreo.md
│
├── tests/                         # Pruebas unitarias (Pytest)
│   ├── test_health.py             # Test del endpoint /health
│   └── test_predict.py            # Test del endpoint /predict
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint + tests + build
│       ├── docker-publish.yml     # Build & push a DockerHub
│       └── docker-smoke.yml       # Smoke tests sobre la imagen publicada
│
├── Dockerfile                     # Imagen para producción
├── docker-compose.yml             # (opcional) Orquestación local
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```
#  Modelo de Machine Learning

- **Algoritmo:** XGBoost Classifier  
- **Objetivo:** Predecir probabilidad de que un cliente abandone el servicio.  
**Features principales:**

- 🔢 **Numéricas**
  - `tenure`
  - `MonthlyCharges`
  - `TotalCharges`
- 🔤 **Categóricas** (one-hot encoding)
  - `gender`, `Partner`, `Dependents`
  - `PhoneService`, `MultipleLines`
  - `InternetService`, `OnlineSecurity`, `OnlineBackup`
  - `DeviceProtection`, `TechSupport`
  - `StreamingTV`, `StreamingMovies`
  - `Contract`, `PaperlessBilling`, `PaymentMethod`

El modelo final se guarda en:
```bash
models/xgb_clf_only.joblib
```
# 🛠 Instalación y Uso en Entorno Local

## 1️⃣ Crear entorno virtual
```bash
conda create -n mlops python=3.11
conda activate mlops
```
##  Arquitectura del Proyecto

```plaintext
        ┌──────────────────────────┐
        │   notebooks/ (EDA + ML)  │
        └────────────┬─────────────┘
                     │
             Entrenamiento y Exportación
                     │
          ┌──────────▼─────────────┐
          │       models/          │
          │ preprocessor.joblib    │
          │ xgb_clf_only.joblib    │
          └──────────┬─────────────┘
                     │
          ┌──────────▼─────────────┐
          │   src/app/main.py      │
          │  → FastAPI Prediction  │
          │  → Dashboard (Dash)    │
          └──────────┬─────────────┘
                     │
         ┌───────────▼──────────────┐
         │ Docker + GitHub Actions  │
         │ CI (lint/tests) + CD     │
         └──────────────────────────┘
```
