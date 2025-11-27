# main.py
import json
import logging
from typing import Dict, Any

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from predict_function import pred_funct


# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")


# -------------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------------
app = FastAPI(
    title="Scoring API",
    description="Servicio de scoring usando un modelo XGBoost.",
    version="1.0.0"
)


# -------------------------------------------------------------------------
# CARGA DE MODELO
# -------------------------------------------------------------------------
model = xgb.Booster()
model.load_model("model.json")

if not model.feature_names:
    raise RuntimeError("El modelo no tiene feature_names definidos.")

logger.info(f"Modelo cargado correctamente. Features: {len(model.feature_names)}")


# -------------------------------------------------------------------------
# SCHEMA DE ENTRADA
# -------------------------------------------------------------------------
class Features(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        min_items=1,
        description="Diccionario de features numéricos requeridos por el modelo."
    )


# -------------------------------------------------------------------------
# HANDLERS GLOBALES DE ERRORES (MEJOR MANEJO DE VALIDACIÓN)
# -------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Maneja errores de validación Pydantic (tipos incorrectos, JSON mal formado, etc.).
    """
    logger.error(f"RequestValidationError: {exc}")

    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error_type": "VALIDATION_ERROR",
            "detail": exc.errors()
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Maneja cualquier excepción NO controlada.
    """
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": "INTERNAL_ERROR",
            "detail": str(exc)
        },
    )


# -------------------------------------------------------------------------
# ENDPOINT HEALTHCHECK
# -------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "message": "Scoring API Alive"}


# -------------------------------------------------------------------------
# ENDPOINT PRINCIPAL DE SCORING
# -------------------------------------------------------------------------

@app.post("/score")
def score_endpoint(payload: Features):
    """
    Endpoint principal para scoring.
    Manejo robusto de:
    - features faltantes
    - tipos incorrectos
    - NaN / None / Inf
    """

    logger.info("Score solicitado")

    try:
        result = pred_funct(model, **payload.features)
    except ValueError as e:
        # Errores de datos (faltantes, tipo incorrecto, NaN...)
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        # Feature inexistente para el modelo
        raise HTTPException(
            status_code=400,
            detail=f"Error en nombre de feature: {str(e)}"
        )

    raw_output = result["raw_output"]
    prob_sigmoid = result["prob_sigmoid"]

    # Validación básica de rango
    if not (0.0 <= raw_output <= 1.0):
        raise HTTPException(
            status_code=500,
            detail=f"Salida del modelo fuera de [0,1]: {raw_output}"
        )

    logger.info(f"Score calculado: {raw_output}")

    return {
        "status": "success",
        "probability": raw_output,  # La que usa el motor
        "raw_output": raw_output,
        "prob_sigmoid": prob_sigmoid
    }


# -------------------------------------------------------------------------
# ENDPOINT DE PRUEBA CON MUESTRA
# -------------------------------------------------------------------------

@app.get("/test_muestra/{case}")
def test_muestra(case: int):
    """
    Ejecuta el modelo con un caso de la muestra (archivo 'muestra' parquet).
    """

    try:
        df = pd.read_parquet("muestra")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo leer 'muestra': {str(e)}"
        )

    if case < 0 or case >= len(df):
        raise HTTPException(
            status_code=400,
            detail=f"Case fuera de rango. Debe estar entre 0 y {len(df)-1}."
        )

    row = df.iloc[case].to_dict()

    try:
        result = pred_funct(model, **row)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno calculando test_muestra: {str(e)}"
        )

    raw_output = result["raw_output"]

    return {
        "status": "success",
        "case": case,
        "probability": raw_output,
        "used_row": row
    }


# -------------------------------------------------------------------------
# END OF FILE
# -------------------------------------------------------------------------
