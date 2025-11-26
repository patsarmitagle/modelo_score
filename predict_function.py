# predict_function.py
import numpy as np
import xgboost as xgb
from typing import Dict, Any
import math


def _build_dmatrix_from_features(
    model: xgb.Booster,
    features: Dict[str, Any]
) -> xgb.DMatrix:
    """
    Construye un xgb.DMatrix a partir de un diccionario de features,
    respetando el orden de model.feature_names y validando:
    - features faltantes
    - valores None / NaN / Inf
    - tipos no numéricos
    - features extra (se ignoran, pero se pueden loguear)
    """
    feature_names = model.feature_names

    if not feature_names:
        raise ValueError(
            "El modelo XGBoost no tiene feature_names definidos. "
            "Verificar que el model.json los incluya."
        )

    # 1) Features faltantes o None
    missing = []
    for f in feature_names:
        if f not in features:
            missing.append(f)
        elif features[f] is None:
            missing.append(f)

    if missing:
        raise ValueError(
            f"Faltan features requeridos por el modelo o vienen en None: {missing}"
        )

    # 2) Features extra (no los usa el modelo, pero no son error)
    extra = [k for k in features.keys() if k not in feature_names]
    # Por ahora solo los ignoramos, pero se pueden loguear  en el servicio.

    # 3) Conversión a float + chequeo NaN / Inf
    vector_vals = []
    bad_type = []

    for f in feature_names:
        v = features[f]
        try:
            v_float = float(v)
        except Exception:
            bad_type.append((f, v))
            continue

        if math.isnan(v_float) or math.isinf(v_float):
            raise ValueError(
                f"El feature '{f}' tiene un valor no válido (NaN/Inf): {v}"
            )

        vector_vals.append(v_float)

    if bad_type:
        detalle = ", ".join([f"{name}={val}" for name, val in bad_type])
        raise ValueError(
            f"Los siguientes features no se pudieron convertir a float: {detalle}"
        )

    # 4) Armar DMatrix
    vector = np.array(vector_vals, dtype=float).reshape(1, -1)
    dmatrix = xgb.DMatrix(vector, feature_names=feature_names)
    return dmatrix


def pred_funct(
    model: xgb.Booster,
    **features: Any
) -> Dict[str, float]:
    """
    Calcula la predicción del modelo.
    """
    dmatrix = _build_dmatrix_from_features(model, features)

    # Salida directa del modelo
    raw_output = float(model.predict(dmatrix)[0])

    # En este caso (binary:logistic) raw_output ya es probabilidad,
    # pero mantenemos prob_sigmoid por consistencia.
    prob_sigmoid = 1.0 / (1.0 + np.exp(-raw_output))

    return {
        "raw_output": raw_output,
        "prob_sigmoid": prob_sigmoid,
    }
