import pandas as pd
import joblib
import os

def predict_diabetes():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")

    model = joblib.load(model_path)

    # Create input with feature names
    new_patient = pd.DataFrame([{
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.6,
        "Age": 35
    }])

    # Try to load threshold metadata if available
    meta_path = os.path.join(BASE_DIR, "diabetes_model_meta.json")
    if os.path.exists(meta_path):
        try:
            meta = joblib.load(meta_path) if meta_path.endswith('.pkl') else None
        except Exception:
            meta = None
        if meta is None:
            try:
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                meta = None
    else:
        meta = None

    # If threshold exists and model provides predict_proba, use it
    if meta and 'threshold' in meta and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(new_patient)[:, 1]
        pred = (proba >= float(meta['threshold'])).astype(int)
    else:
        pred = model.predict(new_patient)

    if int(pred[0]) == 1:
        print("🟥 Diabetes Detected")
    else:
        print("🟩 No Diabetes Detected")

if __name__ == "__main__":
    predict_diabetes()
