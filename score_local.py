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
        examples_features = {
        "REV328": 0.55,
        "BC01S": 2,
        "AT104S": 80,
        "REV112": 4,
        "AT34B": 92,
        "BC20S": 15,
        "ALL255": 1.9,
        "G051S": 0,
        "FI33S_LG": -1,
        "AGG1123": 1.8,
        "RI31S": -3,
        "MNPMAG01": 330,
        "RET84": 5,
        "ALL253": 0.9,
        "AGG324": 0,
        "AT09SF": 7,
        "RI201S": 0,
        "BALMAG03_BG": -1,
        "TRANBAL21": 2,
        "BALMAG01": 300,
        "TELCO_AGG9101": -5,
        "AT29S": 4,
        "REV84": 0,
        "AU51A": -1,
        "RLE907": -1,
        "LL34S": -3,
        "FU20S": 0,
        "SE21S": 5,
        "ALL231": -1,
        "BALMAG04": 280,
        "BI01S": 3,
        "BKC235": 7.0,
        "RI201S_BG": -1,
        "RE12S": 1,
        "RET112": -1,
        "OF01S": 3,
        "IN34S": 90,
        "LMD34S": -3,
        "TEL27S": -1,
        "PAYMNT02": 3
}
       
    prob = score_features(examples_features)
    print(f"Probabilidad predicha: {prob:.6f}")
