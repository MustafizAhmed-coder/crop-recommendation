from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/crop_dataset.csv")
MODEL_PATH = Path("data/crop_model.joblib")

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


def validate_dataset(dataframe):
    required_columns = FEATURES + ["label"]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    if dataframe.empty:
        raise ValueError("The crop dataset is empty.")

    if dataframe["label"].nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two crops."
        )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/crop_dataset.csv was not found."
        )

    dataframe = pd.read_csv(DATA_PATH)

    validate_dataset(dataframe)

    dataframe = dataframe.dropna(
        subset=FEATURES + ["label"]
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    X = dataframe[FEATURES]
    y = dataframe["label"]

    print("Total records:", len(dataframe))
    print("Total crops:", y.nunique())
    print("Available crops:", sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nModel trained successfully.")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    print(importance)


if __name__ == "__main__":
    main()