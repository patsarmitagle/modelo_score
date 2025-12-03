import pandas as pd
import xgboost as xgb
from predict_function import pred_funct

MODEL_PATH = "model.json"
MUESTRA_PATH = "muestra"   # tu archivo parquet

print(f"Cargando modelo desde: {MODEL_PATH}")
model = xgb.Booster()
model.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.\n")


def score_features(features: dict) -> float:
    """
    Recibe un diccionario de features y devuelve la probabilidad
    usando la misma lógica que el servicio /score.
    """
    result = pred_funct(model, **features)
    prob = result["raw_output"]   # en tu modelo: ya es probabilidad
    return prob


def main():
    print(f"Leyendo muestra desde: {MUESTRA_PATH}")
    df = pd.read_parquet(MUESTRA_PATH)
    print(f"Muestra leída. Filas: {len(df)}, Columnas: {len(df.columns)}\n")

    # Tomamos los casos del 0 al 19 (o hasta donde llegue el DF)
    max_case = min(20, len(df))
    print(f"Calculando score para los casos 0 a {max_case - 1}...\n")

    resultados = []

    for idx in range(max_case):
        row = df.iloc[idx]
        features = row.to_dict()  # dict con todas las columnas como features

        prob = score_features(features)

        print(f"Caso {idx}: score = {prob:.6f}")

        # Guardamos para exportar si queremos
        resultados.append({"case": idx, "score": prob})

    # Convertimos a DataFrame y lo guardamos como CSV
    df_res = pd.DataFrame(resultados)
    out_path = "scores_muestra_0_19.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nScores guardados en: {out_path}")


if __name__ == "__main__":
    main()