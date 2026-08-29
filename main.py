from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Churn Prediction API")

MODEL_PATH = "churn_model.joblib"
bundle = joblib.load(MODEL_PATH)
model, FEATURES, VERSION = bundle['model'], bundle['features'], bundle['version']

class ChurnRequest(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: int
    internet_service: int
    tech_support: int
    online_security: int
    paperless_billing: int
    payment_method: int
    num_support_calls: int
    senior_citizen: int
    partner: int
    dependents: int

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/metadata")
def metadata():
    return {"model_version": VERSION, "features": FEATURES}

@app.post("/predict")
def predict(req: ChurnRequest):
    row = pd.DataFrame([req.dict()])[FEATURES]
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    return {
        "prediction": pred,
        "churn_probability": round(prob, 4),
        "model_version": VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }
