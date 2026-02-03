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

        # Read sex & optional metadata (these optional fields are not used by the current model)
        sex = data.get("Sex", "Female")
        family_history = data.get("FamilyHistory", "unknown")
        smoking = data.get("Smoking", "no")
        activity = data.get("ActivityLevel", "moderate")

        # Pregnancies: if mobile/men, ensure value is 0 to avoid confusion
        try:
            if sex == "Male":
                pregnancies = 0
            else:
                pregnancies = int(data.get("Pregnancies", 0))
        except ValueError:
            pregnancies = 0

        # Build dataframe with features the model expects (order doesn't matter for pandas columns but must match training features)
        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": float(data.get("Glucose", 0.0)),
            "BloodPressure": float(data.get("BloodPressure", 0.0)),
            "SkinThickness": float(data.get("SkinThickness", 0.0)),
            "Insulin": float(data.get("Insulin", 0.0)),
            "BMI": float(data.get("BMI", 0.0)),
            "DiabetesPedigreeFunction": float(data.get("DiabetesPedigreeFunction", 0.0)),
            "Age": int(data.get("Age", 0))
        }])

        prediction = model.predict(patient)[0]

        # Make result message slightly more informative
        if prediction == 1:
            session["result"] = "🟥 Diabetes Detected"
        else:
            session["result"] = "🟩 No Diabetes Detected"

        # Optionally store submitted metadata in session for display (not persisted)
        session["last_input"] = {
            "Sex": sex,
            "Pregnancies": int(pregnancies),
            "FamilyHistory": family_history,
            "Smoking": smoking,
            "ActivityLevel": activity
        }

        return redirect(url_for("home"))

    except Exception as e:
        session["result"] = f"Error: {e}"
        return redirect(url_for("home"))

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    # Bind to all interfaces so the app is reachable from other machines on the LAN
    app.run(debug=True, host="0.0.0.0", port=5000)
