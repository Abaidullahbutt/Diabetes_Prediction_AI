import pandas as pd
import os

def preprocess_data():
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build full path to dataset
    data_path = os.path.join(BASE_DIR, "data", "diabetes.csv")

    print("Reading dataset from:", data_path)

    data = pd.read_csv(data_path)

    # Replace 0 values with mean (medical columns)
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols:
        data[col] = data[col].replace(0, data[col].mean())

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    return X, y

if __name__ == "__main__":
    X, y = preprocess_data()
    print("✅ Data preprocessing completed")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
