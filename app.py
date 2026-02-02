from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load model once at startup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")

try:
    model = joblib.load(model_path)
except:
    raise Exception("Model file not found. Run train_model.py first!")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form
        patient = pd.DataFrame([{
            "Pregnancies": int(data["Pregnancies"]),
            "Glucose": float(data["Glucose"]),
            "BloodPressure": float(data["BloodPressure"]),
            "SkinThickness": float(data["SkinThickness"]),
            "Insulin": float(data["Insulin"]),
            "BMI": float(data["BMI"]),
            "DiabetesPedigreeFunction": float(data["DiabetesPedigreeFunction"]),
            "Age": int(data["Age"])
        }])

        # Predict
        prediction = model.predict(patient)[0]

        result = "🟥 Diabetes Detected" if prediction == 1 else "🟩 No Diabetes Detected"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    # host=0.0.0.0 works on all interfaces
    app.run(debug=True, host="127.0.0.1", port=5000)
