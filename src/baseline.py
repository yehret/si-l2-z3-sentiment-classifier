import re
from pathlib import Path
from typing import Iterable, List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class MajorityClassBaseline:
    """Punkt odniesienia 1: Zawsze zwraca najczęstszą klasę ze zbioru uczącego."""

    def __init__(self) -> None:
        self.majority_label: str = "neutral"

    def fit(self, y_train: Iterable[str]) -> "MajorityClassBaseline":
        series = pd.Series(list(y_train))
        self.majority_label = str(series.mode().iloc[0])
        return self

    def predict(self, sentences: Iterable[str]) -> List[str]:
        return [self.majority_label for _ in sentences]


class KeywordBaseline:
    """Punkt odniesienia 2: Prosty słownik pozytywnych i negatywnych słów finansowych."""

    POSITIVE_WORDS = {
        "profit", "profits", "growth", "increase", "increased", "rose", "up",
        "gain", "gains", "positive", "strong", "higher", "record", "improved",
        "improvement", "surge", "dividend", "win", "won", "agreement", "expand",
        "expansion", "exceeded", "outperform", "success", "successful"
    }

    NEGATIVE_WORDS = {
        "loss", "losses", "drop", "dropped", "fell", "fall", "decrease",
        "decreased", "down", "negative", "weak", "lower", "decline", "declined",
        "cut", "cuts", "layoffs", "debt", "warning", "deficit", "slump",
        "litigation", "fine", "penalty", "bankruptcy", "crisis"
    }

    def fit(self, y_train: Iterable[str] = None) -> "KeywordBaseline":
        return self

    def predict_one(self, sentence: str) -> str:
        tokens = re.findall(r"\b[a-z]+\b", str(sentence).lower())
        pos_hits = sum(1 for token in tokens if token in self.POSITIVE_WORDS)
        neg_hits = sum(1 for token in tokens if token in self.NEGATIVE_WORDS)

        if pos_hits > neg_hits:
            return "positive"
        if neg_hits > pos_hits:
            return "negative"
        return "neutral"

    def predict(self, sentences: Iterable[str]) -> List[str]:
        return [self.predict_one(sentence) for sentence in sentences]


if __name__ == "__main__":
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("Brak plików train.csv / test.csv. Uruchom najpierw notatnik 01_dane_i_eksploracja.ipynb.")
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        models = {
            "Baseline 1: Klasa większościowa (zawsze neutral)": MajorityClassBaseline().fit(train_df["label"]),
            "Baseline 2: Słownik słów kluczowych": KeywordBaseline().fit(),
        }

        print("=== WYNIKI PUNKTÓW ODNIESIENIA (ZBIÓR TESTOWY, N = 690) ===")
        for name, model in models.items():
            preds = model.predict(test_df["sentence"])
            acc = accuracy_score(test_df["label"], preds) * 100
            macro_f1 = f1_score(test_df["label"], preds, average="macro")
            print(f"{name}:")
            print(f"  Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.4f}\n")