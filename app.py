"""
Simple Flask web app that serves the fake news classifier.

Usage:
    python train.py       # train the model first (creates model/*.joblib)
    python app.py          # start the web server, visit http://localhost:5000
"""
import os
from flask import Flask, render_template, request
from predict import load_model, predict

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
vectorizer, clf = (None, None)


def get_model():
    global vectorizer, clf
    if vectorizer is None or clf is None:
        vectorizer, clf = load_model(MODEL_DIR)
    return vectorizer, clf


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            error = "Please paste some article text or a headline."
        else:
            try:
                vec, model = get_model()
                label, score = predict(text, vec, model)
                result = {
                    "label": label,
                    "score": round(float(score), 3),
                    "confidence": min(abs(float(score)) / 3.0, 1.0) * 100,
                }
            except FileNotFoundError as e:
                error = str(e)

    return render_template("index.html", result=result, error=error, text=text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
