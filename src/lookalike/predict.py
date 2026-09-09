from pathlib import Path

import joblib
import numpy as np

from src.lookalike.classifier import FEATURE_COLUMNS
from src.lookalike.features import extract_features


MODEL_FILE = Path("models/lookalike_xgboost.pkl")


def predict_lookalike(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Predict whether a candidate region is oil or look-alike.
    """

    model = joblib.load(MODEL_FILE)

    features = extract_features(image, mask)

    feature_vector = np.array(
        [[features[name] for name in FEATURE_COLUMNS]],
        dtype=np.float32,
    )

    prediction = int(model.predict(feature_vector)[0])

    probabilities = model.predict_proba(feature_vector)[0]

    confidence = float(
        probabilities[prediction]
    )

    if prediction == 1:
        class_name = "oil"
    else:
        class_name = "look-alike"

    return {
        "class": class_name,
        "confidence": confidence,
    }