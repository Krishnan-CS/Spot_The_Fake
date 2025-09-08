from launch_URLs import launch_URLs, get_url_list, stop_server as _stop_server  # keep alias; we won't call it
from utils import capture_html_screenshot, save_html_content
from ui_similarity import check_UI_Similarity
from transformers import CLIPProcessor, CLIPModel
from phishing_data_validation import predict_from_gemini  # ML model predictor

from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS

import os
from ollama_model import OllamaGemmaModel

LOCAL_HOST_PATH = "http://localhost:8000"

# -------------------------
# Crawl / Prepare data
# -------------------------
url_list = get_url_list(5)
launch_URLs(url_list)
file_paths = [os.getcwd() + path for path in url_list]
save_html_content(file_paths)

# -------------------------
# Load Gemma model
# -------------------------
OLLAMA_API_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma3:1b"
gemma_model = OllamaGemmaModel(OLLAMA_MODEL_NAME, OLLAMA_API_BASE_URL)

# -------------------------
# LLM analysis from text
# -------------------------
llm_result = {}
html_text_folder = os.path.join(os.getcwd(), "HTML_Text")
html_text_files = [f for f in os.listdir(html_text_folder) if not f.startswith(".")]

for file in html_text_files:
    key = file.split(".")[0] + ".html.png"
    llm_result[key] = gemma_model.analyze_html_with_gemma(
        os.path.join(html_text_folder, file)
    )

print("LLM RESULT:", llm_result)

# -------------------------
# Screenshot storage location
# -------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
screenshots_folder = '/phishing-web/public/Screenshots'
os.makedirs(screenshots_folder, exist_ok=True)

# Optionally capture screenshots
# url_paths = [LOCAL_HOST_PATH + path for path in url_list]
# capture_html_screenshot(url_paths, output_dir=screenshots_folder)

screenshot_files = [f for f in os.listdir(screenshots_folder) if not f.startswith(".")]

# -------------------------
# Similarity Analysis
# -------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

similarity_result = {}
for screenshot in screenshot_files:
    screenshot_path = os.path.join(screenshots_folder, screenshot)
    similarity_result[screenshot] = check_UI_Similarity(
        screenshot_path, clip_model, clip_processor
    )

# -------------------------
# Build phishing_result from LLM and ML
# -------------------------
phishing_result = {}

for screenshot, (data, status) in llm_result.items():
    try:
        summary = data.get("summary_judgment", "")
        confidence = float(data.get("confidence_score", 0.0))
        indicators = data.get("phishing_indicators", {})
        legitimate = data.get("legitimate_indicators", {})
    except Exception as e:
        print(f"[WARN] Skipping malformed LLM result for {screenshot}: {e}")
        continue

    normalized_indicators = {}
    if isinstance(indicators, list):
        for raw in indicators:
            if ":" in raw:
                key, val = raw.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                normalized_indicators[key] = {
                    "status": True,
                    "extracted_data": [val.strip()],
                }
            else:
                normalized_indicators[raw.strip().lower().replace(" ", "_")] = {
                    "status": True,
                    "extracted_data": [],
                }
    elif isinstance(indicators, dict):
        for key, val in indicators.items():
            if isinstance(val, dict):
                normalized_indicators[key] = {
                    "status": val.get("status", "unknown"),
                    "extracted_data": val.get("extracted_data", []),
                }
            else:
                normalized_indicators[key] = {
                    "status": val,
                    "extracted_data": [],
                }

    # ML fallback
    ml_output = predict_from_gemini(data)
    ml_label = ml_output.get("label", "")
    ml_prob = ml_output.get("prob_phish", 0.0)

    use_ml = summary not in ["Phish", "Not Phish"] or confidence == 0.0

    final_label = ml_label if use_ml else summary
    final_confidence = ml_prob if use_ml else confidence

    phishing_result[screenshot] = {
        "final_label": final_label,
        "final_confidence": final_confidence,
        "summary_judgment": summary,
        "confidence_score": confidence,
        "ml_label": ml_label,
        "ml_confidence": ml_prob,
        "phishing_indicators": normalized_indicators,
        "legitimate_indicators": legitimate if isinstance(legitimate, dict) else {},
    }

print("PHISHING RESULT:", phishing_result)

# -------------------------
# Flask API
# -------------------------
app = Flask(__name__)
CORS(app)

@app.route("/api/results")
def api_results():
    results = []
    for screenshot in screenshot_files:
        if screenshot not in phishing_result:
            continue
        results.append({
            "screenshot": screenshot,
            "final_label": phishing_result[screenshot]["final_label"],
            "final_confidence": phishing_result[screenshot]["final_confidence"],
            "summary_judgment": phishing_result[screenshot]["summary_judgment"],
            "confidence_score": phishing_result[screenshot]["confidence_score"],
            "ml_label": phishing_result[screenshot]["ml_label"],
            "ml_confidence": phishing_result[screenshot]["ml_confidence"],
            "phishing_indicators": phishing_result[screenshot]["phishing_indicators"],
            "legitimate_indicators": phishing_result[screenshot]["legitimate_indicators"],
            "similarity_score": similarity_result.get(screenshot, 0.0),
        })
    return jsonify(results)

@app.route("/screenshots/<path:filename>")
def serve_screenshot(filename):
    try:
        return send_from_directory(screenshots_folder, filename)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    print(" API ready at http://localhost:5000/api/results")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)