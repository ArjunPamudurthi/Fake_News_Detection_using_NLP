# Fake News Detection using Natural Language Processing

Detect fake news articles and headlines using **TF-IDF** feature extraction and a **Random Forest** classifier. Includes a **Streamlit** web UI for interactive predictions.

## Project Structure

```
fake news detection/
├── data/                    # Place True.csv and Fake.csv here
├── outputs/                 # Generated after training
│   ├── charts/              # Confusion matrix, ROC, PR curves
│   ├── pipeline.joblib      # Full sklearn pipeline
│   ├── model.joblib         # Classifier only
│   ├── vectorizer.joblib    # TF-IDF vectorizer only
│   └── metrics.json         # Accuracy, AUC, F1, etc.
├── src/
│   ├── train_model.py       # Training script
│   ├── detect_fake_news.py  # CLI prediction
│   ├── streamlit_app.py     # Web UI
│   ├── text_clean.py        # Text preprocessing utilities
│   └── utils.py             # I/O helpers
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the **Fake and Real News Dataset** from Kaggle:
- https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `True.csv` and `Fake.csv` into the `data/` folder.

## Usage

### 1. Train the Model

```bash
cd "fake news detection"
python src/train_model.py --real data/True.csv --fake data/Fake.csv --outdir outputs
```

This will:
- Load and combine title + text features
- Train a TF-IDF (1-3 grams) + Random Forest pipeline
- Run 5-fold cross-validation
- Save model artifacts to `outputs/`
- Generate evaluation charts (confusion matrix, ROC, PR curves)

### 2. CLI Prediction

```bash
python src/detect_fake_news.py --pipeline outputs/pipeline.joblib --text "Breaking: shocking revelation about..."
```

Or using separate model/vectorizer:

```bash
python src/detect_fake_news.py --model outputs/model.joblib --vectorizer outputs/vectorizer.joblib --text "Some headline"
```

### 3. Streamlit Web App

```bash
streamlit run src/streamlit_app.py
```

Open the displayed URL in your browser. Paste any news headline or article and click **Analyze**.

## How It Works

1. **Text Cleaning** — lowercase, remove URLs/emails, strip non-ASCII, collapse whitespace
2. **Feature Extraction** — TF-IDF with unigrams, bigrams, and trigrams (max 20,000 features)
3. **Classification** — Random Forest (400 trees, balanced class weights)
4. **Evaluation** — Accuracy, ROC-AUC, Average Precision, 5-fold CV F1 score
