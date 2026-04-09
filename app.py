"""
PhishGuard Flask Backend
========================
Serves the GUI and exposes a /predict endpoint that runs the real ML model.

Folder structure expected:
  Phish Detection/
  ├── app.py              ← this file
  ├── predict.py          ← your existing predict script
  ├── train.py            ← required for EmailFeatureBuilder import
  ├── models/
  │   └── phishguard_rf.joblib
  └── templates/
      └── index.html      ← the GUI file (rename phishguard_gui.html → index.html)

Run:
  pip install flask
  python app.py

Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from train import EmailFeatureBuilder   # ← REQUIRED: joblib needs this in scope to deserialize the model
from predict import load_model, predict_email
import os

app = Flask(__name__)

# ── Load model once at startup (not on every request) ──────────────────────
MODEL_PATH = os.path.join("models", "phishguard_rf.joblib")

print("🔄 Loading PhishGuard model...")
model, feature_builder = load_model(MODEL_PATH)
print("✅ Model loaded and ready.\n")


# ── Serve the GUI ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Prediction endpoint ─────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    email_text = data["text"].strip()

    if not email_text:
        return jsonify({"error": "Email text cannot be empty."}), 400

    try:
        label, score, prob = predict_email(email_text, model, feature_builder)

        # Determine risk level from score
        if score >= 70:
            risk_level = "High"
        elif score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "verdict":     label,        # "PHISHING" or "SAFE"
            "score":       score,         # 0–100
            "confidence":  round(float(prob), 4),  # 0.0–1.0
            "risk_level":  risk_level,    # "Low" / "Medium" / "High"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)