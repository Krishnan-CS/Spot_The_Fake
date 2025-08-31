# validate_phish_classifier.py
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
import joblib

MODEL_PATH = "phish_rf.pkl"
FEATURES_PATH = "phish_rf_features.json"
CSV_PATH = "phishing_dataset_cleaned.csv"

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
    out = {}
    if not isinstance(block, dict):
        return {k: 0 for k in expected_keys}
    for k in expected_keys:
        v = block.get(k, {})
        if isinstance(v, dict):
            out[k] = 1 if bool(v.get("status", False)) else 0
        elif isinstance(v, bool):
            out[k] = 1 if v else 0
        else:
            out[k] = 0
    return out

def extract_features_from_row(ga: Dict[str, Any]) -> Dict[str, Any]:
    feats = {}
    feats["confidence_score"] = float(ga.get("confidence_score", 0.0))

    phish_flags = _flatten_indicator_block(ga.get("phishing_indicators", {}), PHISH_KEYS)
    feats.update({f"phish_{k}": int(v) for k, v in phish_flags.items()})

    legit_flags = _flatten_indicator_block(ga.get("legitimate_indicators", {}), LEGIT_KEYS)
    feats.update({f"legit_{k}": int(v) for k, v in legit_flags.items()})

    return feats

def transform_one(ga: Union[str, Dict[str, Any]], feature_columns: List[str]) -> pd.DataFrame:
    ga_dict = _safe_json_load(ga)
    row = extract_features_from_row(ga_dict)
    X = pd.DataFrame([row])

    # Add any missing columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Keep only expected columns and order them
    X = X[feature_columns]
    return X

def load_model_and_features(model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH):
    clf = joblib.load(model_path)
    feat_cols = json.loads(Path(features_path).read_text())["feature_columns"]
    return clf, feat_cols

def predict_from_gemini(ga: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    clf, feat_cols = load_model_and_features(MODEL_PATH, FEATURES_PATH)
    X = transform_one(ga, feat_cols)
    pred = int(clf.predict(X)[0])
    label = "Phish" if pred == 1 else "Not Phish"
    try:
        prob = float(clf.predict_proba(X)[0, 1])
    except Exception:
        prob = None
    return {"label": label, "prob_phish": prob}

# --- sample test cases ----
TEST_CASES = [
    {
        "name": "Obvious Phish - brand impersonation + credential ask",
        "json": {
            "summary_judgment": "Phish",
            "confidence_score": 0.97,
            "phishing_indicators": {
                "brand_impersonation": {"status": True},
                "sensitive_info_request": {"status": True},
                "suspicious_urls_scripts": {"status": True},
                "emotional_language": {"status": False},
                "misspellings_errors": {"status": False},
                "other_indicators": {"status": False},
            },
            "legitimate_indicators": {
                "verifiable_details": {"status": False},
                "professional_tone": {"status": False},
                "standard_features": {"status": False},
                "other_indicators": {"status": False},
            },
        },
    },
    {
        "name": "Legit - professional tone + verifiable details",
        "json": {
            "summary_judgment": "Not Phish",
            "confidence_score": 0.92,
            "phishing_indicators": {
                "brand_impersonation": {"status": False},
                "sensitive_info_request": {"status": False},
                "suspicious_urls_scripts": {"status": False},
                "emotional_language": {"status": False},
                "misspellings_errors": {"status": False},
                "other_indicators": {"status": False},
            },
            "legitimate_indicators": {
                "verifiable_details": {"status": True},
                "professional_tone": {"status": True},
                "standard_features": {"status": True},
                "other_indicators": {"status": False},
            },
        },
    },
    {
        "name": "Mixed signals - suspicious script but also standard footer",
        "json": {
            "summary_judgment": "Phish",   # ground-truth isn't used here; model predicts from features
            "confidence_score": 0.70,
            "phishing_indicators": {
                "brand_impersonation": {"status": False},
                "sensitive_info_request": {"status": False},
                "suspicious_urls_scripts": {"status": True},
                "emotional_language": {"status": False},
                "misspellings_errors": {"status": False},
                "other_indicators": {"status": False},
            },
            "legitimate_indicators": {
                "verifiable_details": {"status": False},
                "professional_tone": {"status": True},
                "standard_features": {"status": True},
                "other_indicators": {"status": False},
            },
        },
    },
]

def main():
    assert Path(MODEL_PATH).exists(), f"Missing model: {MODEL_PATH}"
    assert Path(FEATURES_PATH).exists(), f"Missing features schema: {FEATURES_PATH}"

    print("\n=== Handcrafted test cases ===")
    for case in TEST_CASES:
        out = predict_from_gemini(case["json"])
        print(f"- {case['name']}: {out}")

    if Path(CSV_PATH).exists():
        print("\n=== Sample predictions from CSV ===")
        df = pd.read_csv(CSV_PATH)
        for i in range(min(5, len(df))):
            ga = df.loc[i, "gemini_analysis_json"]
            out = predict_from_gemini(ga)
            # If you want to compare to the CSV's summary_judgment, parse it:
            try:
                parsed = _safe_json_load(ga)
                true_label = parsed.get("summary_judgment", None)
            except Exception:
                true_label = None
            print(f"Row {i}: pred={out}, true_summary={true_label}")
    else:
        print(f"\nCSV not found at {CSV_PATH} — skipping CSV-based check.")

if __name__ == "__main__":
    main()
