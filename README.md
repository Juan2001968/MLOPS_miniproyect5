#  Proyecto MLOps: Predicción de Churn en Telecomunicaciones



##  Descripción General

Este proyecto implementa un **flujo completo de MLOps** para la **predicción de abandono de clientes (churn)** en una empresa de telecomunicaciones.  
El modelo se entrena utilizando **XGBoost** y **técnicas de ingeniería de características**, y se despliega como un **servicio API REST** con **FastAPI**, acompañado de un **Dashboard interactivo en Dash** y **monitoreo del modelo con Evidently AI**.

El proyecto automatiza todo el pipeline de despliegue mediante **Docker** y **GitHub Actions (CI/CD)**.



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
