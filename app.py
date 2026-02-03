from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = "diabetes_secret_key"  # required for session

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

# HOME (GET)
@app.route("/")
def home():
    # Clear result on refresh
    result = session.pop("result", None)
    return render_template("index.html", result=result)

# PREDICT (POST)
@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        prediction = model.predict(patient)[0]

        session["result"] = (
            "🟥 Diabetes Detected" if prediction == 1 else "🟩 No Diabetes Detected"
        )

        # 🔥 Redirect instead of render
        return redirect(url_for("home"))

    except Exception as e:
        session["result"] = f"Error: {e}"
        return redirect(url_for("home"))

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
