# Financial News Sentiment Classifier

**Team:** Z3 · Artificial Intelligence lab, Collegium Da Vinci, 2026/2027 · group L2

## What it does

The system is designed for stock market investors and financial analysts who need to quickly assess market sentiment without manually reading dozens of daily news releases. Short financial news headlines in English go into the system as plain text, and the classifier outputs one of three sentiment labels: positive, neutral, or negative

## AI technique

We use classical machine learning combining TF-IDF text vectorization with three binary Perceptrons arranged in a One-vs-Rest (OvR) architecture, where the final class is selected via raw activation. Unlike rigid keyword dictionaries, this linear model learns word weights directly from labeled financial sentences to capture domain-specific context.

## Results

The models are evaluated on a frozen test set (data/processed/test.csv, 690 sentences), created via a stratified 80/20 split (random_state=42) from 3,448 unique sentences of the Financial PhraseBank dataset (Sentences_75Agree.txt: 62.09% neutral, 25.73% positive, 12.18% negative; average length 22.8 words).

## How to run

1. Clone the repo and install dependencies:

```
git clone https://github.com/yehret/si-l2-z3-sentiment-classifier.git
cd si-l2-z3-sentiment-classifier
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

2. Run the baseline evalutaion:

```
python src/baseline.py
```

3. Run Jupyter Notebooks

## Team

Serhii Samoliuk (31070): Core algorithm implementation and technical documentation.

Mariia Minakova (31100): Data preparation and exploration, model evaluation, documentation, and presentation.

## Documentation

Full documentation in Polish: `docs/dokumentacja.md`. Technical: `docs/tech.md`.
