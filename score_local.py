import sys
import json
import xgboost as xgb
from predict_function import pred_funct

MODEL_PATH = 'model.json'

print("Cargando modelo desde:", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")

def score_features(features: dict) -> float:
    result = pred_funct(model, **features)
    prob = result['raw_output']
    return prob

if __name__ == "__main__":
    if len(sys.argv) == 2:
        #cargar las características desde un archivo JSON
        file_path = sys.argv[1]
        print ("Leyendo características desde el archivo:", file_path)
        with open(file_path, 'r') as f:
            examples_features = json.load(f)

    else:
        # caso por defecto de pasar las características como argumento JSO
        examples_features = { ... }
       
    prob = score_features(examples_features)
    print(f"Probabilidad predicha: {prob:.6f}")
