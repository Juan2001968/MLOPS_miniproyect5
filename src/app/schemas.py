from typing import Optional, List
from pydantic import BaseModel, Field

# Esquema de entrada (campos originales del Telco Customer Churn)
class TelcoRecord(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # ojo: en el dataset puede venir como string (con blancos)

class BatchRequest(BaseModel):
    records: List[TelcoRecord]

class Prediction(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: str
    model_version: str
    elapsed_ms: float
