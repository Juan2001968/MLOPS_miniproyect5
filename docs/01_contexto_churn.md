# 1. Contexto del Dataset: *Telco Customer Churn*

## 1.1. Introducción

El presente proyecto tiene como propósito desarrollar un modelo predictivo capaz de anticipar la **pérdida de clientes (churn)** en el sector de las telecomunicaciones.  
El fenómeno de *churn* representa un reto estratégico para las empresas del sector, ya que conservar a un cliente existente suele ser mucho más rentable que adquirir uno nuevo.  
Por esta razón, contar con herramientas de predicción temprana permite a las organizaciones diseñar campañas de retención y ofrecer servicios personalizados que disminuyan la tasa de abandono.


## 1.2. Descripción general del dataset

El dataset contiene información de clientes de una compañía de telecomunicaciones, incluyendo características **demográficas, de facturación y de uso de servicios**.  
Cada fila representa un cliente, mientras que la variable objetivo indica si este **abandonó el servicio** (`Churn = Yes`) o **permaneció con la empresa** (`Churn = No`).

**Características principales:**
- **Tamaño aproximado:** 7.043 registros y 21 columnas.  
- **Variable objetivo:** `Churn` (Yes / No).  
- **Tipo de problema:** Clasificación binaria supervisada.  
- **Fuentes de información:** Datos administrativos y de facturación.  
- **Formato:** CSV (valores separados por comas).


## 1.3. Variables más relevantes

Las columnas más representativas del conjunto de datos incluyen:

- **CustomerID:** identificador único del cliente.  
- **Gender:** género del cliente (Male/Female).  
- **SeniorCitizen:** indicador de si el cliente es adulto mayor.  
- **Partner / Dependents:** información familiar.  
- **Tenure:** número de meses que el cliente ha estado con la empresa.  
- **PhoneService, InternetService, StreamingTV, StreamingMovies:** tipo de servicios contratados.  
- **Contract:** tipo de contrato (mensual, anual, etc.).  
- **PaymentMethod:** método de pago utilizado.  
- **MonthlyCharges / TotalCharges:** cargos mensuales y totales.  
- **Churn:** variable objetivo que indica abandono del servicio (Yes/No).


## 1.4. Objetivo analítico

El objetivo principal del análisis es **identificar los factores que influyen en la deserción de clientes** y construir un modelo predictivo que permita anticipar dichos casos.  
Esto implica desarrollar un flujo de trabajo completo que integre:
- **Análisis exploratorio (EDA)** para conocer las relaciones entre variables.  
- **Preprocesamiento de datos** con tratamiento de valores nulos, codificación de variables categóricas y manejo de desbalance.  
- **Modelado predictivo** utilizando algoritmos basados en árboles como:
  - `Random Forest`
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`
- **Optimización de hiperparámetros** mediante `GridSearchCV` y validación cruzada (`cv=5`).  
- **Evaluación de métricas** de clasificación (accuracy, recall, F1-score, ROC-AUC).  
- **Interpretabilidad del modelo** con técnicas como *Feature Importance* y *LIME*.


## 1.5. Aplicación práctica y motivación

Las empresas de telecomunicaciones pueden usar este tipo de modelos para:
- Detectar patrones asociados a la pérdida de clientes.  
- Anticipar qué usuarios tienen mayor probabilidad de abandonar el servicio.  
- Priorizar acciones de retención mediante estrategias personalizadas.  

En el contexto del aprendizaje automático aplicado a negocios, este caso representa un ejemplo clásico de cómo la **analítica predictiva y la interpretabilidad del modelo** pueden transformar datos operativos en decisiones estratégicas.


