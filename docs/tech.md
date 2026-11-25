# Technical Summary: Financial News Sentiment Classifier

**Team:** Z3 — Serhii Samoliuk (`31070`), Mariia Minakova (`31100`) · Artificial Intelligence Lab, Collegium Da Vinci, 2026/2027 · Group L2

## 1. Purpose and Dataset

Financial analysts and retail investors monitor dozens of short corporate news releases daily, where manually assessing market sentiment is slow and inconsistent. This project implements an automated multi-class sentiment classifier that categorizes short English financial headlines into three classes: `positive`, `neutral`, or `negative`, and computes the overall sentiment balance for a batch of headlines.

We use the public **Financial PhraseBank** corpus (`Sentences_75Agree.txt`), keeping sentences with at least $75\%$ inter-annotator agreement. After removing duplicates, the dataset comprises **3,448 unique sentences** (mean length: $22.8$ words; range: $2\text{--}81$ words) with pronounced class imbalance:

- `neutral`: $2\,141$ sentences ($62.09\%$)
- `positive`: $887$ sentences ($25.73\%$)
- `negative`: $420$ sentences ($12.18\%$)

To guarantee strict evaluation hygiene without data leakage, the corpus was partitioned once using stratified sampling (`random_state=42`) in an $80/20$ ratio into **2,758 training sentences** (`data/processed/train.csv`) and a frozen **690-sentence test set** (`data/processed/test.csv`).

## 2. AI Architecture and Implementation

Raw headlines are converted into sparse numerical feature vectors using **TF-IDF** (`max_features=5000`, `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`) fitted exclusively on `train.csv`. Standard English stop-word filtering is intentionally disabled (`stop_words=None`) so that directional financial terms such as _up_, _down_, and _not_ are preserved.

Classification is performed by a custom **One-vs-Rest (`OvR`) Perceptron** implemented from scratch in NumPy (`src/perceptron.py`), consisting of three independent binary linear classifiers trained for `negative`, `neutral`, and `positive`. Given a feature vector $\mathbf{x} \in \mathbb{R}^d$, each perceptron computes its raw linear activation:

$$
Z_k(\mathbf{x}) = \mathbf{w}_k \cdot \mathbf{x} + b_k
$$

Multi-class predictions and ties (including cases where all activations are negative) are resolved via $\hat{y}(\mathbf{x}) = \arg\max_k Z_k(\mathbf{x})$. To stabilize decision boundaries on non-separable text vectors and address class imbalance, we implemented **Averaged Perceptron** weight averaging across all training steps and square-root balanced class weighting (`class_weight="balanced"`).

## 3. Evaluation Results and Runtime Performance

All models were evaluated on the exact same frozen 690-sentence test set using `Accuracy` and `Macro-F1` (the primary metric for imbalanced classes). Our final model exceeded the project success criterion ($\ge 82.00\%$ Accuracy and $\ge 0.75$ Macro-F1) and matched the reference `scikit-learn` implementation:

| Model (Frozen Test Set, $N = 690$)                              |   Accuracy    |   Macro-F1   |
| :-------------------------------------------------------------- | :-----------: | :----------: |
| Baseline 1: Majority Class (`neutral`)                          |   $62.03\%$   |   $0.2552$   |
| Baseline 2: Financial Keyword Dictionary                        |   $76.81\%$   |   $0.6804$   |
| Model 3: Custom `OneVsRestPerceptron` (basic)                   |   $78.26\%$   |   $0.7174$   |
| Model 4: Reference `sklearn.linear_model.Perceptron`            |   $84.64\%$   |   $0.8015$   |
| **Model 5: Custom `Averaged OneVsRestPerceptron` + `balanced`** | **$85.36\%$** | **$0.7976$** |

**Measured CPU Performance (`python src/perceptron.py --benchmark`):**

- **TF-IDF vectorization (`train` + `test`):** $0.5725\text{ s}$
- **Training time (`fit` on 2,758 sentences, 20 epochs):** $0.8126\text{ s}$
- **Batch inference on 690 test headlines:** $36.93\text{ ms}$
- **Single-headline end-to-end latency (TF-IDF + prediction):** $1.132\text{ ms}$
- **Peak traced RAM usage:** $383.70\text{ MB}$

Qualitative testing on $12$ unseen real-world market headlines (`python src/perceptron.py --real`) confirmed practical applicability, while error analysis documented expected limitations of linear bag-of-words representations on complex nested clauses and implicit financial context.
