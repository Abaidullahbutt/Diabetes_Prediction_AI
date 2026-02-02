from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

from data_preprocessing import preprocess_data

def train():
    X, y = preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained successfully")
    print(f"🎯 Accuracy: {accuracy:.2f}")

    # Save model in project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")

    joblib.dump(model, model_path)
    print("💾 Model saved at:", model_path)

if __name__ == "__main__":
    train()
