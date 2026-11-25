# Klasyfikator sentymentu wiadomości finansowych

**Zespół:** Z3 · grupa L2 · **Repozytorium:** [<link>](https://github.com/yehret/si-l2-z3-sentiment-classifier)
Sztuczna inteligencja — laboratorium · Collegium Da Vinci · rok akad. 2026/2027

## 1. Problem _(rubryka A)_

Odbiorcą systemu jest inwestor giełdowy lub analityk finansowy, który każdego dnia śledzi dziesiątki krótkich komunikatów prasowych oraz nagłówków rynkowych dotyczących notowanych spółek. Ręczne czytanie i ocenianie wydźwięku każdej wiadomości jest czasochłonne oraz podatne na subiektywną interpretację, natomiast na rynku finansowym liczy się szybka orientacja w ogólnym nastawieniu informacyjnym wokół danej firmy.

System rozwiązuje problem automatycznej kategoryzacji krótkich tekstów finansowych pod kątem ich wydźwięku emocjonalno-rynkowego (analiza sentymentu):

Input: pojedynczy nagłówek lub krótkie zdanie wiadomości finansowej w języku angielskim.

Output: jedna z trzech dyskretnych klas sentymentu: positive , neutral lub negative.

## 2. Dlaczego SI i dlaczego ta technika _(B · 7 pkt)_

Klasyczne podejście algorytmiczne oparte na sztywnych regułach warunkowych (if-else) lub prostym słowniku słów kluczowych okazuje się niewystarczające w domenie finansowej. O wydźwięku zdania decyduje kontekst oraz wzajemne powiązanie słów: przykładowo słowo growth lub increase ma wydźwięk pozytywny w zestawieniu profit growth (wzrost zysku), ale negatywny w frazie debt growth (wzrost zadłużenia). Sztywne reguły nie potrafią również samodzielnie dobrać wag poszczególnych pojęć na podstawie rzeczywistych przykładów rynkowych.

W projekcie zastosowaliśmy klasyczne uczenie maszynowe łączące wektoryzację TF-IDF z układem trzech binarnych perceptronów w architekturze One-vs-Rest:

Wektoryzacja TF-IDF zamienia tekst w wektor liczbowy, nadając wyższą wagę rzadkim, niosącym treść terminom finansowym, a obniżając znaczenie słów powszechnych.

Trzy perceptrony stanowią prosty, liniowy model klasyfikacji wieloklasowej, który dobrze radzi sobie w wielowymiarowej przestrzeni cech tekstowych. W porównaniu z głębokimi drzewami decyzyjnymi model liniowy jest mniej podatny na przeuczenie na rzadkich wektorach tekstu, a w porównaniu z dużymi sieciami neuronowymi uczy się w ułamku sekundy lokalnie na CPU i pozwala bezpośrednio odczytać wagi przypisane konkretnym słowom.

## 3. Dane _(B · 8 pkt)_

Projekt wykorzystuje publiczny zbiór danych Financial PhraseBank (takala/financial_phrasebank), wariant Sentences_75Agree.txt, który zawiera zdania z wiadomości finansowych co do których etykiety zgodziło się co najmniej 75% niezależnych ekspertów z dziedziny finansów i ekonomii.

Podczas wstępnego przetwarzania (notebooks/01_dane_i_eksploracja.ipynb) usunięto zduplikowane wpisy oraz zweryfikowano kompletność rekordów (brak pustych wartości). Po oczyszczeniu zbiór liczy 3 448 unikalnych zdań. Średnia długość komunikatu wynosi 22,8 słowa.

Zbiór danych charakteryzuje się wyraźną nierównowagą klas, zdecydowaną większość stanowią komunikaty neutralne:

neutral - 2141 zdań - 62,09%
positive - 887 - 25,73%
negative - 420 - 12,18%

Zbiór 3 448 zdań został podzielony jednorazowo w proporcji 80/20
Aby całkowicie wyeliminować wyciek danych, wektoryzator jest dopasowywany wyłącznie na zbiorze uczącym train.csv, natomiast na zbiorze testowym test.csv wykonywana jest jedynie transformacja.

## 4. Rozwiązanie _(A · 10 pkt, B · 5 pkt)_

Przepływ danych w systemie wygląda następująco:

1. Zdanie wejściowe przetwarzane jest przez `TfidfVectorizer`.
2. Wektor tekstowy trafia do trzech binarnych perceptronów.
3. Każdy perceptron oblicza wynik dla jednej klasy i porównuje tę klasę z pozostałymi.
4. Spośród wyników wybierana jest klasa o najwyższej wartości i zwracana jako odpowiedź.

```text
Tekst wiadomości
      |
      v
TF-IDF: tekst -> wektor liczb
      |
      v
+-------------------------------+
| negative vs. pozostałe klasy  |
| neutral vs. pozostałe klasy   |   Wyniki pobudzenia
| positive vs. pozostałe klasy  |
+---------------+---------------+
                |
                v
          argmax(Z) -> jedna etykieta
```

Zaletą takiego układu jest to, że każdy perceptron koncentruje się tylko na jednym zadaniu: np. `negative` kontra zbiór pozostałych klas. Tekst reprezentujemy jako wektory TF-IDF. Wiersz `i` wektora odpowiada jednej wiadomości, a kolumna `j` jednej cechy tekstowej — pojedynczemu słowu lub dwuwyrazowej frazie.

Dla klasy `k` perceptron oblicza surowe pobudzenie:

```text
Z(k, i) = w(k) · x(i) + b(k)
```

gdzie `x(i)` jest wektorem TF-IDF wiadomości, `w(k)` i `b(k)` oznaczają wagę cech i wyraz wolny tej klasy. Jeśli `Z(k, i) >= 0`, perceptron uznaje wiadomość za należącą do klasy `k`; w przeciwnym razie klasyfikuje ją jako jedną z pozostałych klas. Na końcu system wykonuje `argmax(Z)` i wybiera klasę o największym pobudzeniu.

Podczas uczenia model sprawdza próbki w losowej kolejności. Gdy przewidywana klasa jest błędna, dokonuje następującej korekty wag:

```text
w := w + eta · y · x
b := b + eta · y
```

Tutaj `eta` jest współczynnikiem uczenia, a `y` przyjmuje wartość `+1` dla odpowiedniej klasy i `-1` dla pozostałych. W modelu użyliśmy współczynnika uczenia `eta = 0.1`. Wariant bazowy trenowano przez maksymalnie 15 epok, bez uśredniania wag i bez specjalnego ważenia klas.

Ostatnia wersja modelu została ulepszona dla danych z nierówną liczbą przykładów w klasach. Zastosowano:

- TF-IDF dla unigramów i bigramów,
- 5000 cech o najwyższej częstości w zbiorze uczącym,
- `min_df = 2`, czyli pominięcie cech występujących tylko raz,
- ważenie podliniowe TF przez `sublinear_tf=True`,
- zachowanie słów kierunkowych, np. `up`, `down` i `not`,
- 20 epok uczenia,
- Averaged Perceptron — uśrednienie wag po krokach uczenia,
- `class_weight="balanced"` — zwiększenie kroku dla klasy rzadkiej; mnożnik wynosi pierwiastek z liczby próbek innych klas podzielonej przez liczbę próbek tej klasy,
- `random_state = 42` dla powtarzalnego podziału i tasowania próbek.

Wektorizer dopasowano wyłącznie do 2758 zdań ze zbioru uczącego. Zbiór testowy przekształcono, ale nie użyto go ponownie do uczenia ani dobierania cech. Dzięki temu wyniki nie są obciążone wyciekiem danych ze zbioru testowego.

## 5. Ewaluacja _(B · 8 pkt)_

Wszystkie modele zostały ocenione na tym samym, zamrożonym zbiorze testowym (`data/processed/test.csv`, N = 690 zdań) przy użyciu trafności (`Accuracy`) oraz głównej miary dla danych niezbalansowanych (`Macro-F1`).

| Model                                                           | Accuracy (%) |  Macro-F1  |
| :-------------------------------------------------------------- | :----------: | :--------: |
| Baseline 1: Klasa większościowa (`neutral`)                     | $62{,}03\%$  | $0{,}2552$ |
| Baseline 2: Słownik słów kluczowych                             | $76{,}81\%$  | $0{,}6804$ |
| Model 3: Własny `OneVsRestPerceptron` (bazowy)                  | $78{,}26\%$  | $0{,}7174$ |
| Model 4: Referencyjny `sklearn.linear_model.Perceptron`         | $84{,}64\%$  | $0{,}8015$ |
| Model 5: Własny `Averaged OneVsRestPerceptron` + `balanced`\*\* | $85{,}36\%$  | $0{,}7976$ |

**Wnioski z ewaluacji:**

Nasz podstawowy perceptron (Model 3) pokonał oba punkty odniesienia (78,26% vs 76,81%), ale zatrzymał się poniżej progu sukcesu (82% / 0,75) przez usuwanie słów kierunkowych (up, down, not) oraz oscylacje wag na niezbalansowanym zbiorze.

Po zachowaniu słów kierunkowych w TF-IDF, dodaniu uśredniania wag (Averaged Perceptron) oraz ważenia klas (Model 5), trafność wzrosła o ponad 7 punktów procentowych do 85,36%, a Macro-F1 do 0,7976.

Model wyraźnie przekroczył nasze kryterium sukcesu (82% Accuracy i 0,75 Macro-F1) i uzyskał trafność nieznacznie wyższą niż gotowa implementacja sklearn.linear_model.Perceptron (84,64%). Macierz pomyłek oraz krzywą błędów w epokach zapisano w results/perceptron_cm.png`, a najważniejsze słowa dla każdej klasy w `results/perceptron_top_features.png`.

## 6. Wymagania niefunkcjonalne _(A · 5 pkt)_

Cały potok (wektoryzacja TF-IDF + trening 3 perceptronów na 2 758 zdaniach + predykcja na 690 zdaniach) wykonuje się w całości na procesorze w czasie poniżej 1,5 sekundy (czas samego treningu ulepszonego perceptronu wynosi 0,95 s, a wariantu bazowego 0,68 s; predykcja całego zbioru testowego zajmuje poniżej 0,04 s).

Metoda `_validate_and_convert_X` w `src/perceptron.py` sprawdza poprawność wymiarów macierzy wejściowej, odrzuca puste zbiory danych oraz automatycznie zastępuje ewentualne wartości `NaN` zerami (`np.nan_to_num`). Nieznane słowa w nowych nagłówkach są bezpiecznie ignorowane przez `TfidfVectorizer`.

Wszystkie operacje losowe — zarówno podział zbioru na część uczącą i testową, jak i tasowanie próbek w każdej epoce uczenia korzystają z ustalonego ziarna losowości "42", co gwarantuje identyczne wyniki przy każdym uruchomieniu.

## 7. Ograniczenia _(E · 4 pkt)_

Mimo wysokiej skuteczności (85,36%), nasz klasyfikator liniowy oparty na reprezentacji worka słów (TF-IDF) błędnie klasyfikuje 101 z 690 zdań ze zbioru testowego. Wynika to z następujących ograniczeń technicznych:

Brak pełnego zrozumienia składni i rzadszych czasowników: Model ocenia obecność pojedynczych słów i bigramów, nie analizując struktury porównawczej zdania ani rzadszych synonimów spadku lub wzrostu.

Wrażliwość na słowa o negatywnym zabarwieniu w kontekście neutralnym: Informacje branżowe zawierające słowa kojarzone z zagrożeniem (np. o cyberatakach w raporcie firmy bezpieczeństwa) mogą zostać błędnie odebrane jako negatywne dla samej spółki.

Przykłady błędnych klasyfikacji ze zbioru testowego:

    „The company said that sales in the three months to the end of March slid to EUR86 .4 m US$ 113.4 m from EUR91 .2 m last year .” — prawdziwa klasa: negative, predykcja: positive (Z = [neg: 0,013, neu: -0,171, pos: 0,057]; rzadsze słowo slid nie przeważyło nad neutralno-pozytywnym kontekstem sprzedaży).

    „F-Secure reported that : - The first half of 2008 has seen a growing number of targeted malware attacks on individuals , companies , and organizations .” — prawdziwa klasa: neutral, predykcja: negative (Z = [neg: 0,078, neu: -0,054, pos: -0,071]; słowa malware i attacks wywołały fałszywy alarm negatywny w czysto informacyjnym raporcie).

    „Fiskars , the World 's 1 Scissors Brand TM , recently won Learning -« Magazine 's 2011 Teachers ' Choice Award for the Classroom .” — prawdziwa klasa: positive, predykcja: neutral (sukces produktowy bez typowego słownictwa finansowego typu profit czy revenue).

Kierunki dalszego rozwoju: Zastąpienie prostych wektorów TF-IDF gęstymi osadzeniami kontekstowymi (np. dostrojonym modelem językowym FinBERT) oraz rozbudowa zbioru o dodatkowe przykłady wiadomości negatywnych.

## 8. Zastosowania w praktyce _(E · 4 pkt)_

Automatyczna klasyfikacja sentymentu nagłówków finansowych znajduje bezpośrednie zastosowanie w narzędziach wspierających decyzje inwestycyjne i analizę ryzyka:

- Automatyczne oznaczanie strumienia depesz rynkowych (np. Yahoo Finance) etykietami pozytywna / neutralna / negatywna
- Filtrowanie dużej liczby komunikatów prasowych w celu natychmiastowego wychwycenia informacji o stratach finansowych, karach regulacyjnych lub obniżkach prognoz zysków
- Szybka ocena wydźwięku komunikatów jako jedna z cech wejściowych w modelach wspomagających handel algorytmiczny

## 9. Zespół _(D · 8 pkt)_

| Osoba           | Za co odpowiadała                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| :-------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Serhii Samoliuk | Przygotowanie karty tematu i pierwotnego `README.md`, implementacja modeli bazowych, implementacja od zera wieloklasowego klasyfikatora `OneVsRestPerceptron` w NumPy wraz z uśrednianiem wag (_Averaged Perceptron_), kompensacją nierównowagi klas, porównaniem ze `scikit-learn`, testem na rzeczywistych nagłówkach, pomiarem wydajności CPU oraz trybem interaktywnym; opracowanie sekcji w dokumentacji oraz streszczenia technicznego po angielsku. |
| Mariia Minakova | Pobranie, czyszczenie i eksploracja zbioru Financial PhraseBank, jednorazowy podział `train/test` ze stratyfikacją, ewaluacja punktów odniesienia, budowa notatnika ewaluacji modelu perceptronowego z wektoryzacją TF-IDF, wizualizacją macierzy pomyłek, wag cech oraz analizą błędów; opracowanie sekcji w dokumentacji, aktualizacja `README.md` wynikami końcowymi oraz przygotowanie prezentacji finałowej.                                          |

## 10. Uruchomienie _(C · 4 pkt)_

1. Sklonowanie repozytorium i instalacja zależności

```
git clone https://github.com/yehret/si-l2-z3-sentiment-classifier.git
cd si-l2-z3-sentiment-classifier
python -m venv venv
# Windows (Git Bash):
source venv/Scripts/activate
# Windows (PowerShell): .\venv\Scripts\Activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

2. Uruchomienie skryptów z terminala
   Ocena punktów odniesienia (Baseline 1 i Baseline 2):

```
python src/baseline.py
```

3. Porównanie wariantów perceptronu ze scikit-learn (tryb domyślny):

```
python src/perceptron.py --compare
```

4. Test na 12 rzeczywistych nagłówkach rynkowych i bilans nastrojów:

```
python src/perceptron.py --real
```

5. Pomiar wymagań niefunkcjonalnych na CPU (czas treningu, opóźnienie 1 zdania, RAM):

```
python src/perceptron.py --benchmark
```

6. Tryb interaktywny (wpisywanie własnych nagłówków na żywo) lub pełny raport:

```
python src/perceptron.py --interactive
python src/perceptron.py --all
```

Szybka klasyfikacja pojedynczego nagłówka przekazanego jako argument:

```
python src/perceptron.py "Nvidia quarterly revenue surges 94% on strong AI demand"
```

(czas treningu, opóźnienie 1 zdania, RAM):

python src/perceptron.py --benchmark

Tryb interaktywny (wpisywanie własnych nagłówków na żywo) lub pełny raport:

python src/perceptron.py --interactive
python src/perceptron.py --all

Szybka klasyfikacja pojedynczego nagłówka przekazanego jako argument:

python src/perceptron.py "Nvidia quarterly revenue surges 94% on strong AI demand"

3. Uruchomienie notatników Jupyter

Otworzyć kolejno w VS Code lub Jupyter Notebook:

notebooks/01_dane_i_eksploracja.ipynb — czyszczenie danych i podział 80/20,

notebooks/02_punkt_odniesienia.ipynb — ocena modeli bazowych,

notebooks/03_model_perceptron.ipynb — pełna ewaluacja perceptronu, wykresy, analiza błędów i test na nagłówkach rynkowych.

Wybrać kernel Python ze środowiska venv i kliknąć Run All.

## 11. Narzędzia AI

Podczas realizacji projektu korzystaliśmy ze wsparcia narzędzi opartych na dużych modelach językowych (Gemini) jako asystenta programistycznego i redakcyjnego:

Zakres użycia: pomoc przy projektowaniu struktury modułów w Pythonie, debugowaniu problemów ze ścieżkami i wektoryzacją macierzy rzadkich scipy.sparse, dopracowaniu formatowania wykresów matplotlib/seaborn oraz korekcie językowej dokumentacji w języku polskim i angielskim.

Weryfikacja przez zespół: cały kod w src/ i notebooks/ został przez nas samodzielnie uruchomiony, przetestowany na zbiorze danych, a wszystkie wyniki liczbowe, wykresy oraz interpretacje merytoryczne zostały zweryfikowane przez członków zespołu.
