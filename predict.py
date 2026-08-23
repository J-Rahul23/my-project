"""
Command-line prediction tool.

Usage:
    python predict.py "Some news headline or article text here"
"""
import sys
import os
import joblib


def load_model(model_dir="model"):
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    clf_path = os.path.join(model_dir, "classifier.joblib")
    if not (os.path.exists(vec_path) and os.path.exists(clf_path)):
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)
    return vectorizer, clf


def predict(text, vectorizer, clf):
    X = vectorizer.transform([text])
    label = clf.predict(X)[0]
    # decision_function gives a confidence-like score (distance from hyperplane)
    score = clf.decision_function(X)[0]
    return label, score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "headline or article text"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    vectorizer, clf = load_model()
    label, score = predict(text, vectorizer, clf)
    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence score: {score:.3f}  (magnitude = confidence, sign = direction)\n")
