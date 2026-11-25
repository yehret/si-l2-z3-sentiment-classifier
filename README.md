# Financial News Sentiment Classifier

**Team:** Z3 · Artificial Intelligence lab, Collegium Da Vinci, 2026/2027 · group L2

## What it does

The system is designed for stock market investors and financial analysts who need to quickly assess market sentiment without manually reading dozens of daily news releases. Short financial news headlines in English go into the system as plain text, and the classifier outputs one of three sentiment labels: `positive`, `neutral`, or `negative`

## AI technique

We use classical machine learning combining TF-IDF text vectorization with three binary Perceptrons arranged in a One-vs-Rest (OvR) architecture, where the final class is selected via raw activation. Unlike rigid keyword dictionaries, this linear model learns word weights directly from labeled financial sentences to capture domain-specific context.

## Results

The models are evaluated on a frozen test set (data/processed/test.csv, 690 sentences), created via a stratified 80/20 split (random_state=42) from 3,448 unique sentences of the Financial PhraseBank dataset (Sentences_75Agree.txt: 62.09% neutral, 25.73% positive, 12.18% negative).

| Model                                                       | Accuracy  | Macro-F1 |
| :---------------------------------------------------------- | :-------: | :------: |
| Baseline 1: Majority Class (`neutral`)                      | $62.03\%$ | $0.2552$ |
| Baseline 2: Financial Keyword Dictionary                    | $76.81\%$ | $0.6804$ |
| Model 3: Custom `OneVsRestPerceptron` (basic)               | $78.26\%$ | $0.7174$ |
| Model 4: Reference `sklearn.linear_model.Perceptron`        | $84.64\%$ | $0.8015$ |
| Model 5: Custom `Averaged OneVsRestPerceptron` + `balanced` | $85.36\%$ | $0.7976$ |

## How to run

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/yehret/si-l2-z3-sentiment-classifier.git
cd si-l2-z3-sentiment-classifier
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

_(On Windows PowerShell, activate the environment with `.\venv\Scripts\Activate`; on Linux/macOS, use `source venv/bin/activate`.)_

### 2. Run the baseline models from the terminal

```bash
python src/baseline.py
```

### 3. Run the Perceptron classifier (modular CLI flags)

- **Compare custom Perceptron variants against `scikit-learn` (default mode):**
  ```bash
  python src/perceptron.py --compare
  ```
- **Test on 12 unseen real-world market headlines and compute sentiment balance:**
  ```bash
  python src/perceptron.py --real
  ```
- **Measure CPU runtime performance (training time, single-headline latency, RAM):**
  ```bash
  python src/perceptron.py --benchmark
  ```
- **Launch interactive mode or run all modes:**
  ```bash
  python src/perceptron.py --interactive
  python src/perceptron.py --all
  ```
- **Classify a single headline passed directly as a CLI argument:**
  ```bash
  python src/perceptron.py "Nvidia quarterly revenue surges 94% on strong AI demand"
  ```

### 4. Run Jupyter Notebooks

Open `notebooks/01_dane_i_eksploracja.ipynb`, `notebooks/02_punkt_odniesienia.ipynb`, and `notebooks/03_model_perceptron.ipynb` sequentially in VS Code or Jupyter, select the `venv` Python kernel, and click **Run All**.

**Where the output appears:** Evaluation metrics (`Accuracy` and `Macro-F1`), real-world headline predictions, and CPU latency measurements are printed directly in the terminal or notebook cell outputs, while confusion matrices and feature-weight plots are saved in the `results/` directory.

## Team

Serhii Samoliuk (31070): Core algorithm implementation and technical documentation.

Mariia Minakova (31100): Data preparation and exploration, model evaluation, documentation, and presentation.

## Documentation

Full documentation in Polish: `docs/dokumentacja.md`. Technical: `docs/tech.md`.
