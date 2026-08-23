"""
Trains a fake news classifier using TF-IDF features + a
PassiveAggressiveClassifier (a strong, fast baseline for text classification).

Usage:
    python train.py                     # uses data/news.csv
    python train.py --data path/to.csv  # use your own dataset (needs 'text','label' cols)
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/news.csv", help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--out", default="model", help="Directory to save trained model artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"{args.data} not found. Run `python data/generate_data.py` first, "
            "or pass --data with your own dataset."
        )

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, max_iter=50, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
    print(f"Labels order: {list(clf.classes_)}\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(args.out, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(args.out, "classifier.joblib"))
    print(f"Saved model artifacts to {args.out}/")


if __name__ == "__main__":
    main()
