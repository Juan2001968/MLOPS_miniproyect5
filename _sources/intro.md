# Proyecto Integrador: Predicción de Churn en Telecomunicaciones con Modelos de Árboles e Interpretabilidad

**Autor:** Juan Andrés Ramos Cardona, Mariana Franco Ritiaga y Jeronimo Dominguez  


## Propósito
El proyecto busca entrenar, evaluar e interpretar modelos de **clasificación binaria** para predecir la **pérdida de clientes (churn)** en una empresa de telecomunicaciones.  
Se implementa un flujo completo de trabajo de **aprendizaje automático e ingeniería MLOps**, que incluye:  
- Preprocesamiento automatizado mediante `Pipeline`.  
- Entrenamiento y ajuste de modelos basados en árboles (`Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`) con `GridSearchCV`.  
- Evaluación mediante métricas de clasificación (precision, recall, F1-score, ROC-AUC).  
- Análisis de **importancia de variables** y **explicabilidad** con técnicas de interpretabilidad (`LIME` y `SHAP`).  
- Despliegue del modelo a través de un **servicio FastAPI**, empaquetado con **Docker**, y orquestado con **CI/CD** en **GitHub Actions**.  
- Visualización de resultados en un **dashboard interactivo** construido con Dash/Plotly.  
- Recomendaciones finales para el **monitoreo post-despliegue** y gestión del modelo en producción.



## Dataset (Telco Customer Churn)
- **Observaciones:** ~7.043 clientes  
- **Variables:** 20 atributos (demográficos, de facturación y de uso de servicios)  
- **Target:** `Churn` (Yes/No)  
- **Reto:** Clasificación binaria con desbalance moderado y mezcla de variables categóricas y numéricas.



## Directorio
1. **Contexto & Dataset**: motivación del problema, descripción del conjunto de datos y variables principales.  
2. **EDA**: análisis exploratorio, visualización de distribución del target y correlaciones.  
3. **Preprocesamiento**: limpieza, imputación de valores faltantes, codificación de variables categóricas (`OneHotEncoder`), tratamiento de desbalance (SMOTE / class_weight).  
4. **Modelado (Árboles)**: construcción de `Pipeline` con `GridSearchCV` para modelos de árbol (RF, XGB, LGBM, CatBoost) y comparación de desempeño.  
5. **Evaluación de Métricas**: precisión, recall, F1, matriz de confusión y curva ROC-AUC.  
6. **Interpretabilidad**: análisis de importancia de variables y explicabilidad local/global con LIME y SHAP.  
7. **Servicio de Inferencia (FastAPI)**: diseño del endpoint `/predict`, validación de entrada con `pydantic` y ejemplo de solicitud/response.  
8. **Docker + CI/CD**: construcción del contenedor, automatización del flujo de despliegue con GitHub Actions.  
9. **Dashboard (Dash/Plotly)**: visualización interactiva de métricas y predicciones del modelo.  
10. **Monitoreo**: estrategias para detección de *data drift*, *label drift* y alertas operativas en entorno productivo.



## Checklist de Objetivos
- [ ] Implementar `Pipeline` con preprocesamiento automático.  
- [ ] Realizar `GridSearchCV` para modelos basados en árboles.  
- [ ] Evaluar el rendimiento con métricas de clasificación (ROC-AUC, F1, recall).  
- [ ] Analizar interpretabilidad con *Feature Importance*, *LIME* y *SHAP*.  
- [ ] Crear servicio de inferencia con **FastAPI**.  
- [ ] Contenerizar con **Docker**.  
- [ ] Configurar **CI/CD** con **GitHub Actions**.  
- [ ] Desarrollar un **Dashboard interactivo (Dash/Plotly)**.  
- [ ] Diseñar recomendaciones para el **monitoreo del modelo**.



