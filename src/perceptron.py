from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
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
        random_state: Optional[int] = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("Współczynnik uczenia (learning_rate) musi być dodatni.")
        if n_epochs < 1:
            raise ValueError("Liczba epok (n_epochs) musi wynosić co najmniej 1.")

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

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

        # Inicjalizacja wag i biasów zerami
        self.weights_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.biases_ = np.zeros(n_classes, dtype=np.float64)
        self.errors_history_ = {}

        rng = np.random.default_rng(self.random_state)

        # Trening oddzielnego perceptronu dla każdej klasy k
        for class_idx, target_class in enumerate(self.classes_):
            y_binary = np.where(y_arr == target_class, 1.0, -1.0)
            w = self.weights_[class_idx]
            b = 0.0
            epoch_errors: List[int] = []

            for _ in range(self.n_epochs):
                indices = rng.permutation(n_samples)
                errors = 0

                for idx in indices:
                    x_i = X_arr[idx]
                    y_true = y_binary[idx]

                    # Surowe pobudzenie liniowe: z = w * x + b
                    z = float(np.dot(w, x_i) + b)
                    y_pred = 1.0 if z >= 0.0 else -1.0

                    if y_pred != y_true:
                        update = self.learning_rate * y_true
                        w += update * x_i
                        b += update
                        errors += 1

                epoch_errors.append(errors)
                if errors == 0:
                    break

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
        Rozwiązuje remisy zarówno gdy dwa perceptrony dają wynik dodatni, jak i gdy żaden.
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
        Przydatne do interpretacji decyzji modelu w dokumentacji i prezentacji.
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


if __name__ == "__main__":
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("Brak plików train.csv / test.csv. Uruchom najpierw 01_dane_i_eksploracja.ipynb.")
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X_train = vectorizer.fit_transform(train_df["sentence"])
        X_test = vectorizer.transform(test_df["sentence"])

        clf = OneVsRestPerceptron(learning_rate=0.1, n_epochs=15, random_state=42)
        clf.fit(X_train, train_df["label"])
        preds = clf.predict(X_test)

        acc = accuracy_score(test_df["label"], preds) * 100
        macro_f1 = f1_score(test_df["label"], preds, average="macro")

        print("=== WYNIKI WŁASNEGO PERCEPTRONU ONE-VS-REST (ZBIÓR TESTOWY, N = 690) ===")
        print(f"Accuracy:          {acc:.2f}%")
        print(f"Macro-F1:          {macro_f1:.4f}")
        print(f"Czas treningu:     {clf.train_time_sec_:.4f} s")
        print(f"Czas predykcji:    {clf.last_predict_time_sec_:.6f} s")