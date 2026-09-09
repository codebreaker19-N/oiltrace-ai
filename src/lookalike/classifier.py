import xgboost as xgb


FEATURE_COLUMNS = [
    "area",
    "perimeter",
    "compactness",
    "aspect_ratio",
    "mean_intensity",
    "intensity_std",
    "intensity_contrast",
]


def create_classifier():
    """
    Create XGBoost binary classifier.

    label:
        0 = look-alike
        1 = oil
    """

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    return model