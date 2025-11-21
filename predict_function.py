import numpy as np
import xgboost as xgb
from typing import Dict, Any


def _build_dmatrix_from_features(
    model: xgb.Booster,
    features: Dict[str, Any]
) -> xgb.DMatrix:
    """
    Construye un xgb.DMatrix a partir de un diccionario de features,
    respetando el orden de model.feature_names.
    """
    feature_names = model.feature_names

    if not feature_names:
        raise ValueError(
            "El modelo XGBoost no tiene feature_names definidos. "
            "Verificar que el model.json los incluya."
        )

    # Verificamos que estén todos los features requeridos
    missing = [f for f in feature_names if f not in features]
    if missing:
        raise KeyError(f"Faltan features requeridos por el modelo: {missing}")

    # Ordenamos los valores en el mismo orden que el modelo
    vector = np.array(
        [features[f] for f in feature_names],
        dtype=float
    ).reshape(1, -1)

    dmatrix = xgb.DMatrix(vector, feature_names=feature_names)
    return dmatrix


def pred_funct(
    model: xgb.Booster,
    **features: Any
) -> Dict[str, float]:
    """
    Calcula la predicción del modelo.

    Parámetros
    ----------
    model : xgboost.Booster
        Modelo XGBoost ya cargado (por ejemplo desde model.json).
    **features :
        Features de entrada, pasados como kwargs, por ejemplo:
        pred_funct(model,
                   REV328=..., BC01S=..., AT104S=..., ...)

    Returns
    -------
    dict
        {
            "raw_output": salida directa de model.predict(),
            "prob_sigmoid": salida transformada con función logística
        }
    """
    dmatrix = _build_dmatrix_from_features(model, features)

    # Salida directa del modelo
    raw_output = float(model.predict(dmatrix)[0])

    # Aplicamos sigmoide por si el modelo está en escala logit.
    # Si el objective del modelo es "binary:logistic", el raw_output YA es probabilidad
    # y esta transformación NO debería usarse, en ese caso usá directamente raw_output.
    prob_sigmoid = 1.0 / (1.0 + np.exp(-raw_output))

    return {
        "raw_output": raw_output,
        "prob_sigmoid": prob_sigmoid,
    }
