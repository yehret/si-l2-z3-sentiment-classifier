"""
Moduł implementujący własny wieloklasowy klasyfikator oparty na trzech
binarnych perceptronach w architekturze One-vs-Rest (OvR) przy użyciu NumPy.

Zawiera:
- podstawowy perceptron z wyborem klasy przez argmax(Z_k),
- rozszerzenie Averaged Perceptron (uśrednianie wag po wszystkich krokach),
- opcjonalne ważenie klas (class_weight="balanced") dla niezbalansowanego zbioru,
- funkcję porównawczą z gotowym modelem sklearn.linear_model.Perceptron.
"""

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score, f1_score


class OneVsRestPerceptron:
    """
    Wieloklasowy klasyfikator One-vs-Rest złożony z K binarnych perceptronów
    (w naszym projekcie K = 3 dla klas: negative, neutral, positive).

    Każdy perceptron uczy się oddzielać jedną klasę (+1) od pozostałych (-1).
    Podczas predykcji wybierana jest klasa o najwyższym surowym pobudzeniu:
        Z_k = w_k * x + b_k  ->  argmax_k(Z_k)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_epochs: int = 15,
        averaged: bool = False,
        class_weight: Optional[str] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("Współczynnik uczenia (learning_rate) musi być dodatni.")
        if n_epochs < 1:
            raise ValueError("Liczba epok (n_epochs) musi wynosić co najmniej 1.")
        if class_weight not in (None, "balanced"):
            raise ValueError("Parametr class_weight musi mieć wartość None lub 'balanced'.")

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.averaged = averaged
        self.class_weight = class_weight
        self.random_state = random_state

        # Atrybuty ustawiane podczas treningu (fit)
        self.classes_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.biases_: Optional[np.ndarray] = None
        self.train_time_sec_: float = 0.0
        self.last_predict_time_sec_: float = 0.0
        self.errors_history_: Dict[str, List[int]] = {}

    @staticmethod
    def _validate_and_convert_X(X: object) -> np.ndarray:
        """
        Waliduje macierz cech X i konwertuje ją do gęstej tablicy NumPy (float64).
        Obsługuje zarówno macierze rzadkie (np. z TfidfVectorizer), jak i tablice NumPy.
        """
        if X is None:
            raise ValueError("Macierz cech X nie może mieć wartości None.")

        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        if X_dense.ndim != 2:
            raise ValueError(f"Oczekiwano macierzy 2D, otrzymano wymiar: {X_dense.ndim}D.")
        if X_dense.shape[0] == 0 or X_dense.shape[1] == 0:
            raise ValueError("Macierz cech X jest pusta (0 próbek lub 0 cech).")

        return np.nan_to_num(X_dense.astype(np.float64), nan=0.0)

    def fit(self, X: object, y: Iterable[str]) -> "OneVsRestPerceptron":
        """
        Trenuje po jednym binarnym perceptronie dla każdej unikalnej klasy w y.
        """
        start_time = perf_counter()

        X_arr = self._validate_and_convert_X(X)
        y_arr = np.asarray(list(y))

        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(
                f"Niezgodna liczba próbek: X ma {X_arr.shape[0]}, a y ma {len(y_arr)}."
            )

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Do klasyfikacji potrzebne są co najmniej 2 różne klasy.")

        n_samples, n_features = X_arr.shape
        n_classes = len(self.classes_)

        self.weights_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.biases_ = np.zeros(n_classes, dtype=np.float64)
        self.errors_history_ = {}

        rng = np.random.default_rng(self.random_state)

        for class_idx, target_class in enumerate(self.classes_):
            y_binary = np.where(y_arr == target_class, 1.0, -1.0)
            w = np.zeros(n_features, dtype=np.float64)
            b = 0.0

            # Zmienne pomocnicze dla wariantu Averaged Perceptron
            w_sum = np.zeros(n_features, dtype=np.float64)
            b_sum = 0.0
            total_steps = 0

            # Współczynnik kompensacji nierównowagi klas (łagodne pierwiastkowe skalowanie)
            if self.class_weight == "balanced":
                n_pos = float(np.sum(y_binary == 1.0))
                n_neg = float(np.sum(y_binary == -1.0))
                pos_multiplier = float(np.sqrt(n_neg / max(n_pos, 1.0)))
            else:
                pos_multiplier = 1.0

            epoch_errors: List[int] = []

            for _ in range(self.n_epochs):
                indices = rng.permutation(n_samples)
                errors = 0

                for idx in indices:
                    x_i = X_arr[idx]
                    y_true = y_binary[idx]

                    z = float(np.dot(w, x_i) + b)
                    y_pred = 1.0 if z >= 0.0 else -1.0

                    if y_pred != y_true:
                        step_lr = self.learning_rate * (pos_multiplier if y_true == 1.0 else 1.0)
                        update = step_lr * y_true
                        w += update * x_i
                        b += update
                        errors += 1

                    if self.averaged:
                        w_sum += w
                        b_sum += b
                        total_steps += 1

                epoch_errors.append(errors)
                if errors == 0:
                    break

            if self.averaged and total_steps > 0:
                self.weights_[class_idx] = w_sum / total_steps
                self.biases_[class_idx] = b_sum / total_steps
            else:
                self.weights_[class_idx] = w
                self.biases_[class_idx] = b

            self.errors_history_[str(target_class)] = epoch_errors

        self.train_time_sec_ = perf_counter() - start_time
        return self

    def decision_function(self, X: object) -> np.ndarray:
        """
        Oblicza macierz surowych pobudzeń Z o wymiarach (n_samples, n_classes),
        gdzie Z[i, k] = w_k * x_i + b_k.
        """
        if self.weights_ is None or self.biases_ is None or self.classes_ is None:
            raise RuntimeError("Model nie został jeszcze wytrenowany. Wywołaj najpierw fit().")

        X_arr = self._validate_and_convert_X(X)
        if X_arr.shape[1] != self.weights_.shape[1]:
            raise ValueError(
                f"Niezgodna liczba cech: model oczekuje {self.weights_.shape[1]}, "
                f"otrzymano {X_arr.shape[1]}."
            )

        return X_arr @ self.weights_.T + self.biases_

    def predict(self, X: object) -> np.ndarray:
        """
        Zwraca przewidywaną etykietę klasy dla każdej próbki na podstawie reguły argmax(Z).
        """
        start_time = perf_counter()

        scores = self.decision_function(X)
        winning_indices = np.argmax(scores, axis=1)
        predictions = self.classes_[winning_indices]

        self.last_predict_time_sec_ = perf_counter() - start_time
        return predictions

    def get_top_features(
        self, feature_names: Iterable[str], top_n: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Zwraca po top_n słów/fraz o najwyższych dodatnich wagach dla każdej klasy.
        """
        if self.weights_ is None or self.classes_ is None:
            raise RuntimeError("Model nie został jeszcze wytrenowany.")

        names = np.asarray(list(feature_names))
        top_features: Dict[str, List[Tuple[str, float]]] = {}

        for class_idx, class_name in enumerate(self.classes_):
            class_weights = self.weights_[class_idx]
            top_indices = np.argsort(class_weights)[::-1][:top_n]
            top_features[str(class_name)] = [
                (str(names[i]), round(float(class_weights[i]), 4)) for i in top_indices
            ]

        return top_features


def evaluate_sklearn_perceptron(
    X_train: object,
    y_train: Iterable[str],
    X_test: object,
    y_test: Iterable[str],
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Trenuje i ocenia referencyjny Perceptron z biblioteki scikit-learn
    na tych samych cechach TF-IDF (punkt 9 karty tematu).
    """
    t0 = perf_counter()
    sk_clf = SklearnPerceptron(
        max_iter=20,
        eta0=0.1,
        class_weight="balanced",
        random_state=random_state,
    )
    sk_clf.fit(X_train, list(y_train))
    train_time = perf_counter() - t0

    preds = sk_clf.predict(X_test)
    y_test_list = list(y_test)
    return {
        "accuracy": round(accuracy_score(y_test_list, preds) * 100, 2),
        "macro_f1": round(f1_score(y_test_list, preds, average="macro"), 4),
        "train_time_sec": round(train_time, 4),
    }


if __name__ == "__main__":
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("Brak plików train.csv / test.csv. Uruchom najpierw 01_dane_i_eksploracja.ipynb.")
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # 1. Wariant bazowy (z zachowaniem stop_words="english")
        vec_base = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
        X_tr_base = vec_base.fit_transform(train_df["sentence"])
        X_te_base = vec_base.transform(test_df["sentence"])

        clf_base = OneVsRestPerceptron(learning_rate=0.1, n_epochs=15, averaged=False, random_state=42)
        clf_base.fit(X_tr_base, train_df["label"])
        preds_base = clf_base.predict(X_te_base)

        # 2. Wariant ulepszony: zachowanie słów kierunkowych (up/down/not) + Averaged Perceptron + balanced
        vec_imp = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words=None,
        )
        X_tr_imp = vec_imp.fit_transform(train_df["sentence"])
        X_te_imp = vec_imp.transform(test_df["sentence"])

        clf_imp = OneVsRestPerceptron(
            learning_rate=0.1,
            n_epochs=20,
            averaged=True,
            class_weight="balanced",
            random_state=42,
        )
        clf_imp.fit(X_tr_imp, train_df["label"])
        preds_imp = clf_imp.predict(X_te_imp)

        # 3. Referencyjny model z biblioteki scikit-learn
        sk_res = evaluate_sklearn_perceptron(X_tr_imp, train_df["label"], X_te_imp, test_df["label"])

        print("=== PORÓWNANIE WARIANTÓW PERCEPTRONU (ZBIÓR TESTOWY, N = 690) ===")
        print(
            f"1. Własny OneVsRestPerceptron (bazowy):    "
            f"Accuracy = {accuracy_score(test_df['label'], preds_base) * 100:.2f}% | "
            f"Macro-F1 = {f1_score(test_df['label'], preds_base, average='macro'):.4f} | "
            f"Czas = {clf_base.train_time_sec_:.4f} s"
        )
        print(
            f"2. Własny Averaged OvR + balanced:         "
            f"Accuracy = {accuracy_score(test_df['label'], preds_imp) * 100:.2f}% | "
            f"Macro-F1 = {f1_score(test_df['label'], preds_imp, average='macro'):.4f} | "
            f"Czas = {clf_imp.train_time_sec_:.4f} s"
        )
        print(
            f"3. Gotowy sklearn.linear_model.Perceptron: "
            f"Accuracy = {sk_res['accuracy']:.2f}% | "
            f"Macro-F1 = {sk_res['macro_f1']:.4f} | "
            f"Czas = {sk_res['train_time_sec']:.4f} s"
        )