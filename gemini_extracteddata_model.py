# phish_classifier_from_gemini.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


CSV_PATH = "phishing_dataset_cleaned.csv"
MODEL_PATH = "phish_rf.pkl"
FEATURES_PATH = "phish_rf_features.json"
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_ESTIMATORS = 300


PHISH_KEYS = [
    "brand_impersonation",
    "sensitive_info_request",
    "suspicious_urls_scripts",
    "emotional_language",
    "misspellings_errors",
    "other_indicators",
]
LEGIT_KEYS = [
    "verifiable_details",
    "professional_tone",
    "standard_features",
    "other_indicators",
]

def _safe_json_load(x: Union[str, dict]) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return json.loads(x)
    raise ValueError(f"Unsupported gemini_analysis_json type: {type(x)}")

def _flatten_indicator_block(block: Any, expected_keys: List[str]) -> Dict[str, int]:
    """
    block is expected to be a dict where keys are from expected_keys and values are dicts
    like {'status': bool, 'examples': [...]}. We convert to {key: 0/1}.
    Missing keys -> 0.
    """
    out = {}
    if not isinstance(block, dict):
        # If block missing or malformed, default zeros
        return {k: 0 for k in expected_keys}
    for k in expected_keys:
        v = block.get(k, {})
        status = 0
        if isinstance(v, dict):
            status = 1 if bool(v.get("status", False)) else 0
        elif isinstance(v, bool):
            status = 1 if v else 0
        out[k] = status
    return out

def extract_features_from_row(ga: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a parsed gemini_analysis_json dict, return a flat feature dict.
    """
    features = {}
    # scalar features
    features["confidence_score"] = float(ga.get("confidence_score", 0.0))

    # phishing_indicators
    phish_block = ga.get("phishing_indicators", {})
    phish_flags = _flatten_indicator_block(phish_block, PHISH_KEYS)
    for k, v in phish_flags.items():
        features[f"phish_{k}"] = int(v)

    # legitimate_indicators
    legit_block = ga.get("legitimate_indicators", {})
    legit_flags = _flatten_indicator_block(legit_block, LEGIT_KEYS)
    for k, v in legit_flags.items():
        features[f"legit_{k}"] = int(v)

    return features

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse gemini_analysis_json, expand features, attach label from summary_judgment.
    """
    # Parse JSON column to dicts
    parsed = df["gemini_analysis_json"].apply(_safe_json_load)

    # Extract features row-wise
    feats = parsed.apply(extract_features_from_row).apply(pd.Series)

    # Label: use summary_judgment inside the JSON
    labels = parsed.apply(lambda d: str(d.get("summary_judgment", "")).strip())
    y = labels.map({"Phish": 1, "Not Phish": 0})

    # Some rows may be unknown labels; drop them if any
    mask_known = y.isin([0, 1])
    feats = feats.loc[mask_known].reset_index(drop=True)
    y = y.loc[mask_known].reset_index(drop=True)

    return feats, y

# Training and Evaluation

def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Inverse-frequency sample weights to counter class imbalance.
    """
    counts = y.value_counts().to_dict()
    inv = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    return y.map(inv).values

def train_eval_save(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    # Train/val split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Handle imbalance via sample weights (works without extra libs)
    w_train = compute_sample_weights(y_train)

    clf.fit(X_train, y_train, sample_weight=w_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        auc = None

    print("\n=== Evaluation on Hold-out Set ===")
    print(classification_report(y_test, y_pred, target_names=["Not Phish", "Phish"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")

    # Persist model & feature list
    joblib.dump(clf, MODEL_PATH)
    Path(FEATURES_PATH).write_text(json.dumps({"feature_columns": list(X.columns)}, indent=2))
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved feature schema to: {FEATURES_PATH}")

    return clf

# Prediction helper

def load_model_and_features(model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH):
    clf = joblib.load(model_path)
    feat_cols = json.loads(Path(features_path).read_text())["feature_columns"]
    return clf, feat_cols

def transform_one(ga: Union[str, Dict[str, Any]], feature_columns: List[str]) -> pd.DataFrame:
    """
    Convert a single gemini_analysis_json (str or dict) into a 1-row DataFrame
    matching the feature schema used during training.
    """
    ga_dict = _safe_json_load(ga)
    row = extract_features_from_row(ga_dict)
    X = pd.DataFrame([row])
    # Ensure column order and fill any missing with 0
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    # Drop any unexpected extras and re-order
    X = X[feature_columns]
    return X

def predict_from_gemini(ga: Union[str, Dict[str, Any]],
                        model_path: str = MODEL_PATH,
                        features_path: str = FEATURES_PATH) -> Dict[str, Any]:
    """
    Predict "Phish" / "Not Phish" for a new gemini_analysis_json blob.
    """
    clf, feat_cols = load_model_and_features(model_path, features_path)
    X = transform_one(ga, feat_cols)
    pred = clf.predict(X)[0]
    try:
        proba = float(clf.predict_proba(X)[0, 1])
    except Exception:
        proba = None
    label = "Phish" if pred == 1 else "Not Phish"
    return {"label": label, "prob_phish": proba}

# main module
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # Build features & labels from gemini_analysis_json
    X, y = build_feature_frame(df)

    print("Class distribution:\n", y.value_counts())
    print("Feature columns:", list(X.columns))

    # Train, evaluate, save
    _ = train_eval_save(X, y)

    # Example: predict on the first row's gemini JSON (sanity check)
    sample_ga = df.loc[0, "gemini_analysis_json"]
    pred_out = predict_from_gemini(sample_ga)
    print("\nSample prediction on first row:")
    print(pred_out)
