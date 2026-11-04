Financial News Sentiment Classifier

The system automatically classifies short financial news in English into one of three sentiment classes: positive, neutral, or negative.

1. Dataset Description

The project utilizes the public Financial PhraseBank dataset (takala/financial_phrasebank, variant Sentences_75Agree.txt — minimum 75% annotator agreement).

Number of unique sentences (after removing duplicates): 3448

Class distribution:

neutral: 2141 sentences (62.09%)

positive: 887 sentences (25.73%)

negative: 420 sentences (12.18%)

Average message length: 22.8 words (min: 2, max: 81)

Data split: The dataset was split once with stratification (random_state=42) in an 80/20 ratio:

Training set (data/processed/train.csv): 2758 sentences

Test set (data/processed/test.csv): 690 sentences (frozen for evaluation)

2. Getting Started (From Cloning to Results)

All computations run locally on the CPU and do not require a graphics card.

Step 1: Clone the repository

git clone https://github.com/yehret/si-l2-z3-sentiment-classifier.git

cd si-l2-z3-sentiment-classifier

Step 2: Create a virtual environment and install dependencies

python -m venv venv

source venv/Scripts/activate

pip install -r requirements.txt

Step 3: Run notebooks and scripts

Check the baseline models from the terminal:

python src/baseline.py

The script loads data/processed/train.csv and data/processed/test.csv and prints the Accuracy and macro-F1 results for both baseline methods in the console.

Run Jupyter Notebooks (in VS Code or browser):

Open notebooks/01_dane_i_eksploracja.ipynb and notebooks/02_punkt_odniesienia.ipynb sequentially.

Select the Python kernel from the venv environment and click Run All.

Plots and results are automatically saved in the results/ and data/processed/ directories.
