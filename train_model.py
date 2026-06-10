import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

DATA_PATH = Path("data/crop_dataset.csv")
MODEL_PATH = Path("data/crop_model.joblib")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/crop_dataset.csv not found")

    df = pd.read_csv(DATA_PATH)

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features]
    y = df["label"]

    # KNN is simple and explainable for first-year viva.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=3))
    ])

    # Small demo dataset: train/test split only for demo accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully!")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Demo accuracy: {accuracy:.2f}")
    print("Note: For final submission, replace crop_dataset.csv with a larger Kaggle dataset.")

if __name__ == "__main__":
    main()
