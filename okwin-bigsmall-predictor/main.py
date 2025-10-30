import joblib
import cv2
import pandas as pd
import numpy as np
from src import 01_ocr_extract as ocr

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_image(path):
    ocr_res = ocr.get_ocr_result(path)
    df = pd.DataFrame(ocr.extract_data_from_ocr_output(ocr_res))
    # feature engineering same as src/02...
    X = scaler.transform(df.drop(columns=["Period","Big_Small"]))
    prob = model.predict_proba(X)[0]
    return "Big" if prob[1] > 0.5 else "Small", prob

print(predict_image("images\batch1 (28).jpeg"))