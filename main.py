# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import xgboost as xgb
import pandas as pd

from predict_function import pred_funct

app = FastAPI(title="Scoring API")

# Cargamos modelo
model = xgb.Booster()
model.load_model("model.json")

if not model.feature_names:
    raise RuntimeError("El modelo no tiene feature_names definidos.")


class Features(BaseModel):
    features: Dict[str, Any] = Field(..., min_items=1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/score")
def score_endpoint(payload: Features):
    """
    Endpoint principal de scoring.
    Maneja:
    - datos faltantes
    - tipos incorrectos
    - valores NaN/Inf
    """
    try:
        result = pred_funct(model, **payload.features)

        raw_output = result["raw_output"]
        prob_sigmoid = result["prob_sigmoid"]

        # Validación simple rango prob
        if not (0.0 <= raw_output <= 1.0):
            raise HTTPException(
                status_code=500,
                detail=f"Salida del modelo fuera de rango [0,1]: {raw_output}"
            )

        return {
            "status": "success",
            "probability": raw_output,     # esta es la que usaría el motor
            "raw_output": raw_output,
            "prob_sigmoid": prob_sigmoid,
        }

    except ValueError as e:
        # Errores de datos: faltantes, tipo, NaN, etc.
        raise HTTPException(status_code=400, detail=str(e))

    except KeyError as e:
        # Por si algo en pred_funct levanta KeyError
        raise HTTPException(
            status_code=400,
            detail=f"Error en nombres de features: {str(e)}"
        )

    except HTTPException:
        # Re-lanzamos si ya construimos una HTTPException arriba
        raise

    except Exception as e:
        # Cualquier otra cosa inesperada
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en scoring: {str(e)}"
        )


# ---- Ejemplo de test con muestra, también con manejo de errores ----

@app.get("/test_muestra/{case}")
def test_muestra(case: int):
    """
    Usa el archivo 'muestra' para probar casos.
    Solo índices válidos.
    """
    try:
        df = pd.read_parquet("muestra")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo leer el archivo 'muestra': {str(e)}"
        )

    if case < 0 or case >= len(df):
        raise HTTPException(
            status_code=400,
            detail=f"Case fuera de rango. Debe estar entre 0 y {len(df)-1}."
        )

    row = df.iloc[case].to_dict()

    try:
        result = pred_funct(model, **row)
        raw_output = result["raw_output"]

        return {
            "status": "success",
            "case": case,
            "probability": raw_output,
            "used_row": row
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno calculando test_muestra: {str(e)}"
        )
# --- END OF FILE ---