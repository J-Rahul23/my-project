# 📰 Fake News Detector

A machine learning system that classifies news text as **REAL** or **FAKE** using
TF-IDF text features and a Passive-Aggressive linear classifier. Includes a CLI tool
and a Flask web demo.

> ⚠️ **Note on the bundled dataset**: `data/news.csv` is a small **synthetic**
> dataset generated for demo purposes so the project runs out of the box. For
> real-world accuracy, retrain on a proper labeled dataset such as the
> [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
> — see "Using your own data" below.

## Features

- TF-IDF vectorization (unigrams + bigrams, English stop words removed)
- `PassiveAggressiveClassifier` — fast, strong baseline for text classification
- CLI prediction tool (`predict.py`)
- Flask web UI for interactive testing (`app.py`)
- Clean train/test split with accuracy, confusion matrix, and classification report

## Project structure

```
fake-news-detector/
├── app.py                  # Flask web app
├── train.py                 # Training script
├── predict.py                # CLI prediction script
├── requirements.txt
├── data/
│   ├── generate_data.py       # Generates synthetic demo dataset
│   └── news.csv               # (generated) training data
├── model/                    # (generated) saved model artifacts
├── templates/
│   └── index.html             # Web UI
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Generate the demo dataset

```bash
python data/generate_data.py
```

### 2. Train the model

```bash
python train.py
```

This prints accuracy, a confusion matrix, and a classification report, then saves
`model/vectorizer.joblib` and `model/classifier.joblib`.

### 3. Predict from the command line

```bash
python predict.py "Scientists confirm shocking secret the government is hiding!"
```

### 4. Run the web app

```bash
python app.py
```

Visit `http://localhost:5000` and paste in a headline or article.

## Using your own data

Replace `data/news.csv` with any CSV containing `text` and `label` columns
(`label` should be `REAL` or `FAKE`), then run:

```bash
python train.py --data path/to/your_dataset.csv
```

## How it works

1. **Vectorization**: Raw text is converted into TF-IDF vectors, which weigh words
   by how distinctive they are to a document relative to the whole corpus.
2. **Classification**: A Passive-Aggressive linear classifier is trained on these
   vectors — it updates aggressively on misclassified examples and stays passive
   (unchanged) on correct ones, making it fast and effective for text.
3. **Inference**: New text is vectorized the same way and passed through the
   trained classifier, which outputs a label plus a decision score used as a
   rough confidence indicator.

## Limitations

- The bundled dataset is synthetic and small — it demonstrates the pipeline, not
  production-grade accuracy. Retrain on a real, larger dataset before relying on
  this for anything important.
- The model relies on stylistic/lexical patterns, not fact verification — it
  cannot check claims against real-world facts.
- Always pair automated tools like this with human judgment and trusted sources.

## License

MIT
