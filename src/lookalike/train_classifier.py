from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.lookalike.classifier import (
    FEATURE_COLUMNS,
    create_classifier,
)


DATA_FILE = Path(
    "data/processed/lookalike/demo_classifier_dataset.csv"
)

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "lookalike_xgboost.pkl"


def main():

    print("Loading demo classifier dataset...")

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    print(f"Total samples : {len(df)}")
    print(f"Features      : {len(FEATURE_COLUMNS)}")

    # 80% training, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")

    print("\nTraining XGBoost...")

    model = create_classifier()

    model.fit(X_train, y_train)

    # Validation
    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    print("\n========================================")
    print("XGBOOST RESULTS")
    print("========================================")
    print(f"Accuracy : {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
            target_names=[
                "look-alike",
                "oil",
            ],
        )
    )

    # Save model
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("========================================")
    print(f"Model saved : {MODEL_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()