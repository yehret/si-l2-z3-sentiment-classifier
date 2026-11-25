import argparse
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score, f1_score


REAL_WORLD_HEADLINES: List[str] = [
    "Nvidia quarterly revenue surges 94% on strong AI data center demand",
    "Boeing reports wider-than-expected net loss amid production delays and strikes",
    "European Central Bank keeps interest rates unchanged at 3.25% as inflation cools",
    "Apple announces $110 billion share buyback program and raises quarterly dividend",
    "Volkswagen plans to close at least three factories in Germany and cut thousands of jobs",
    "Microsoft completes annual shareholder meeting with board of directors re-elected",
    "Tesla operating profit jumps as vehicle delivery costs fall to record low",
    "Intel shares plunge after weak revenue forecast and suspension of dividend",
    "JPMorgan Chase schedules fourth-quarter earnings conference call for January 15",
    "Amazon operating income rises sharply driven by cloud computing growth",
    "Credit Suisse faces heavy regulatory fine over compliance failures",
    "Siemens Energy raises full-year profit outlook following strong grid orders",
]


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

            w_sum = np.zeros(n_features, dtype=np.float64)
            b_sum = 0.0
            total_steps = 0

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
    na tych samych cechach TF-IDF.
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


def classify_headlines(
    vectorizer: TfidfVectorizer,
    model: OneVsRestPerceptron,
    headlines: Iterable[str],
) -> pd.DataFrame:
    """
    Klasyfikuje listę zewnętrznych nagłówków finansowych i zwraca tabelę
    z przewidywaną klasą oraz surowymi pobudzeniami Z_k.
    """
    headline_list = list(headlines)
    X_vec = vectorizer.transform(headline_list)
    preds = model.predict(X_vec)
    scores = model.decision_function(X_vec)

    rows = []
    for idx, text in enumerate(headline_list):
        row = {
            "headline": text,
            "predicted_sentiment": str(preds[idx]),
        }
        for cls_idx, cls_name in enumerate(model.classes_):
            row[f"Z_{cls_name}"] = round(float(scores[idx, cls_idx]), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def benchmark_performance(
    train_sentences: Iterable[str],
    train_labels: Iterable[str],
    test_sentences: Iterable[str],
    n_runs: int = 5,
) -> Dict[str, float]:
    """
    Mierzy wymagania niefunkcjonalne systemu na CPU:
    - czas wektoryzacji TF-IDF oraz treningu modelu,
    - czas inferencji dla całego zbioru testowego,
    - średni czas odpowiedzi dla pojedynczego nagłówka,
    - zużycie pamięci RAM podczas treningu i predykcji.
    """
    tr_sent = list(train_sentences)
    tr_lbl = list(train_labels)
    te_sent = list(test_sentences)

    tracemalloc.start()

    t_vec_start = perf_counter()
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words=None,
    )
    X_tr = vec.fit_transform(tr_sent)
    X_te = vec.transform(te_sent)
    vec_time_sec = perf_counter() - t_vec_start

    clf = OneVsRestPerceptron(
        learning_rate=0.1,
        n_epochs=20,
        averaged=True,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_tr, tr_lbl)

    pred_times = []
    for _ in range(n_runs):
        _ = clf.predict(X_te)
        pred_times.append(clf.last_predict_time_sec_)
    avg_batch_pred_sec = float(np.mean(pred_times))

    single_times_ms = []
    sample_headline = [te_sent[0]]
    for _ in range(50):
        t0 = perf_counter()
        x_one = vec.transform(sample_headline)
        _ = clf.predict(x_one)
        single_times_ms.append((perf_counter() - t0) * 1000.0)
    avg_single_ms = float(np.mean(single_times_ms))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "tfidf_time_sec": round(vec_time_sec, 4),
        "fit_time_sec": round(clf.train_time_sec_, 4),
        "total_train_pipeline_sec": round(vec_time_sec + clf.train_time_sec_, 4),
        "test_batch_predict_ms": round(avg_batch_pred_sec * 1000.0, 2),
        "single_headline_latency_ms": round(avg_single_ms, 3),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
    }


def predict_single_headline(
    vectorizer: TfidfVectorizer,
    model: OneVsRestPerceptron,
    sentence: str,
) -> None:
    """
    Klasyfikuje pojedynczy nagłówek wpisany przez użytkownika i wypisuje
    przewidywaną klasę oraz surowe pobudzenia Z_k dla każdego z trzech perceptronów.
    """
    cleaned = sentence.strip()
    if not cleaned:
        print("Pusty tekst — podaj niepusty nagłówek w języku angielskim.")
        return

    X_custom = vectorizer.transform([cleaned])
    pred_label = str(model.predict(X_custom)[0])
    scores = model.decision_function(X_custom)[0]
    scores_str = ", ".join(
        f"{cls}: {val:.3f}" for cls, val in zip(model.classes_, scores)
    )
    print(f"Zdanie:     {cleaned}")
    print(f"Predykcja:  {pred_label.upper()}")
    print(f"Pobudzenia: [{scores_str}]")


def interactive_headline_prompt(
    vectorizer: TfidfVectorizer,
    model: OneVsRestPerceptron,
) -> None:
    """
    Uruchamia interaktywną pętlę w terminalu, pozwalającą wpisać własne
    nagłówki finansowe na żywo (zakończenie: pusty Enter lub 'q').
    """
    if not (sys.stdin and sys.stdin.isatty()):
        return

    print("=== TRYB INTERAKTYWNY: SPRAWDŹ WŁASNY NAGŁÓWEK ===")
    print("Wpisz nagłówek finansowy po angielsku (lub naciśnij Enter / wpisz 'q', aby zakończyć):")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nZakończono tryb interaktywny.")
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            print("Zakończono.")
            break

        predict_single_headline(vectorizer, model, user_input)


def train_final_pipeline(
    train_df: pd.DataFrame,
) -> Tuple[TfidfVectorizer, OneVsRestPerceptron]:
    vec_imp = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words=None,
    )
    X_tr_imp = vec_imp.fit_transform(train_df["sentence"])

    clf_imp = OneVsRestPerceptron(
        learning_rate=0.1,
        n_epochs=20,
        averaged=True,
        class_weight="balanced",
        random_state=42,
    )
    clf_imp.fit(X_tr_imp, train_df["label"])
    return vec_imp, clf_imp


def run_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    vec_base = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X_tr_base = vec_base.fit_transform(train_df["sentence"])
    X_te_base = vec_base.transform(test_df["sentence"])

    clf_base = OneVsRestPerceptron(learning_rate=0.1, n_epochs=15, averaged=False, random_state=42)
    clf_base.fit(X_tr_base, train_df["label"])
    preds_base = clf_base.predict(X_te_base)

    vec_imp, clf_imp = train_final_pipeline(train_df)
    X_tr_imp = vec_imp.transform(train_df["sentence"])
    X_te_imp = vec_imp.transform(test_df["sentence"])
    preds_imp = clf_imp.predict(X_te_imp)

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


def run_real_headlines_test(vec_imp: TfidfVectorizer, clf_imp: OneVsRestPerceptron) -> None:
    print("=== TEST NA RZECZYWISTYCH NAGŁÓWKACH RYNKOWYCH (N = 12) ===")
    real_df = classify_headlines(vec_imp, clf_imp, REAL_WORLD_HEADLINES)
    for _, row in real_df.iterrows():
        print(f"[{row['predicted_sentiment'].upper():8s}] {row['headline']}")

    balance = real_df["predicted_sentiment"].value_counts(normalize=True) * 100
    balance_str = ", ".join(f"{k}: {v:.1f}%" for k, v in balance.items())
    print(f"Ogólny bilans nastrojów: {balance_str}")


def run_benchmark_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    perf = benchmark_performance(
        train_df["sentence"], train_df["label"], test_df["sentence"]
    )
    print("=== POMIAR WYMAGAN NIEFUNKCJONALNYCH (CPU) ===")
    print(f"Wektoryzacja TF-IDF (train+test):    {perf['tfidf_time_sec']:.4f} s")
    print(f"Trening 3 perceptronów (2758 zdań):  {perf['fit_time_sec']:.4f} s")
    print(f"Predykcja całego testu (690 zdań):   {perf['test_batch_predict_ms']:.2f} ms")
    print(f"Czas odpowiedzi na 1 nagłówek:       {perf['single_headline_latency_ms']:.3f} ms")
    print(f"Szczytowe zużycie pamięci RAM:       {perf['peak_memory_mb']:.2f} MB")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Klasyfikator sentymentu wiadomości finansowych (TF-IDF + OneVsRestPerceptron)."
    )
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("headline", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("Brak plików train.csv / test.csv. Uruchom najpierw 01_dane_i_eksploracja.ipynb.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if args.headline:
        vec_imp, clf_imp = train_final_pipeline(train_df)
        print("=== KLASYFIKACJA WŁASNEGO ZDANIA ===")
        predict_single_headline(vec_imp, clf_imp, " ".join(args.headline))
        sys.exit(0)

    no_flags = not (args.compare or args.real or args.benchmark or args.interactive or args.all)
    do_compare = args.all or args.compare or no_flags
    do_real = args.all or args.real
    do_benchmark = args.all or args.benchmark
    do_interactive = args.all or args.interactive

    sections_printed = 0

    if do_compare:
        run_comparison(train_df, test_df)
        sections_printed += 1

    vec_imp: Optional[TfidfVectorizer] = None
    clf_imp: Optional[OneVsRestPerceptron] = None

    if do_real or do_interactive:
        vec_imp, clf_imp = train_final_pipeline(train_df)

    if do_real and vec_imp is not None and clf_imp is not None:
        if sections_printed > 0:
            print()
        run_real_headlines_test(vec_imp, clf_imp)
        sections_printed += 1

    if do_benchmark:
        if sections_printed > 0:
            print()
        run_benchmark_report(train_df, test_df)
        sections_printed += 1

    if do_interactive and vec_imp is not None and clf_imp is not None:
        interactive_headline_prompt(vec_imp, clf_imp)

# --compare     : porównuje wyniki własnego perceptronu z punktami odniesienia i modelem sklearn
# --real        : testuje model na 12 rzeczywistych nagłówkach rynkowych
# --benchmark   : mierzy wymagania niefunkcjonalne na CPU
# --interactive | --i : uruchamia tryb interaktywny
# --all         : uruchamia wszystkie powyższe tryby po kolei
