from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import xgboost as xgb
from predict_function import pred_funct
import pandas as pd

app = FastAPI(title="Scoring API")

# cargar modelo UNA única vez
model = xgb.Booster()
model.load_model("model.json")


# ---- Modelo del request ----
class Features(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/score")
def score_endpoint(payload: Features):
    try:
        # llamar a tu función existente (sin modificarla)
        result = pred_funct(model, **payload.features)

        # usamos raw_output porque tu modelo es binary:logistic
        return {
            "probability": result["raw_output"],
            "raw_output": result["raw_output"],
            "prob_sigmoid": result["prob_sigmoid"],
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---- Test usando los casos del archivo 'muestra' ----

@app.get("/test_muestra/{case}")
def test_muestra(case: int):
    df = pd.read_parquet("muestra")

    # Validar rango permitido
    if case < 0 or case > 19:
        return {
            "status": "error",
            "message": "Solo se permiten casos entre 0 y 19."
        }

    # Validar que la muestra tenga suficientes filas
    if case >= len(df):
        return {
            "status": "error",
            "message": f"La muestra solo tiene {len(df)} filas."
        }

    # Seleccionar la fila solicitada
    row = df.iloc[case].to_dict()
    result = pred_funct(model, **row)

    return {
        "status": "success",
        "case": case,
        "probability": result["raw_output"],
        "used_row": row
    }