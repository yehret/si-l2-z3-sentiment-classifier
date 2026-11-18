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

Wszystkie modele zostały ocenione na tym samym, zamrożonym zbiorze testowym (`data/processed/test.csv`, $N = 690$ zdań) przy użyciu trafności (`Accuracy`) oraz głównej miary dla danych niezbalansowanych (`Macro-F1`).

| Model                                                           | Accuracy (%) |  Macro-F1  |
| :-------------------------------------------------------------- | :----------: | :--------: |
| Baseline 1: Klasa większościowa (`neutral`)                     | $62{,}03\%$  | $0{,}2552$ |
| Baseline 2: Słownik słów kluczowych                             | $76{,}81\%$  | $0{,}6804$ |
| Model 3: Własny `OneVsRestPerceptron` (bazowy)                  | $78{,}26\%$  | $0{,}7174$ |
| Model 4: Referencyjny `sklearn.linear_model.Perceptron`         | $84{,}64\%$  | $0{,}8015$ |
| Model 5: Własny `Averaged OneVsRestPerceptron` + `balanced`\*\* | $85{,}36\%$  | $0{,}7976$ |

**Wnioski z ewaluacji:**

- Nasz podstawowy perceptron (`Model 3`) pokonał oba punkty odniesienia ($78{,}26\%$ vs $76{,}81\%$), ale zatrzymał się poniżej progu sukcesu ($82\%$ / $0{,}75$) przez usuwanie słów kierunkowych (_up_, _down_, _not_) oraz oscylacje wag na niezbalansowanym zbiorze.
- Po zachowaniu słów kierunkowych w TF-IDF, dodaniu uśredniania wag (_Averaged Perceptron_) oraz ważenia klas (`Model 5`), trafność wzrosła o ponad $7$ punktów procentowych do **$85{,}36\%$**, a `Macro-F1` do **$0{,}7976$**.
- Model wyraźnie przekroczył nasze kryterium sukcesu ($82\%$ Accuracy i $0{,}75$ Macro-F1) i uzyskał trafność nieznacznie wyższą niż gotowa implementacja `sklearn.linear_model.Perceptron` ($84{,}64\%$). Macierz pomyłek oraz krzywą błędów w epokach zapisano w `results/perceptron_cm.png`, a najważniejsze słowa dla każdej klasy w `results/perceptron_top_features.png`.

## 6. Wymagania niefunkcjonalne _(A · 5 pkt)_

Cały potok (wektoryzacja TF-IDF + trening 3 perceptronów na 2 758 zdaniach + predykcja na 690 zdaniach) wykonuje się w całości na procesorze w czasie poniżej 1,5 sekundy (czas samego treningu ulepszonego perceptronu wynosi 0,95 s, a wariantu bazowego 0,68 s; predykcja całego zbioru testowego zajmuje poniżej 0,04 s).

Metoda `_validate_and_convert_X` w `src/perceptron.py` sprawdza poprawność wymiarów macierzy wejściowej, odrzuca puste zbiory danych oraz automatycznie zastępuje ewentualne wartości `NaN` zerami (`np.nan_to_num`). Nieznane słowa w nowych nagłówkach są bezpiecznie ignorowane przez `TfidfVectorizer`.

Wszystkie operacje losowe — zarówno podział zbioru na część uczącą i testową, jak i tasowanie próbek w każdej epoce uczenia korzystają z ustalonego ziarna losowości "42", co gwarantuje identyczne wyniki przy każdym uruchomieniu.

## 7. Ograniczenia _(E · 4 pkt)_

Mimo wysokiej skuteczności ($85{,}36\%$), nasz klasyfikator liniowy oparty na reprezentacji worka słów (TF-IDF) błędnie klasyfikuje $101$ z $690$ zdań ze zbioru testowego. Wynika to z następujących ograniczeń technicznych:

- **Brak pełnego zrozumienia składni i rzadszych czasowników:** Model ocenia obecność pojedynczych słów i bigramów, nie analizując struktury porównawczej zdania ani rzadszych synonimów spadku lub wzrostu.
- **Wrażliwość na słowa o negatywnym zabarwieniu w kontekście neutralnym:** Informacje branżowe zawierające słowa kojarzone z zagrożeniem (np. o cyberatakach w raporcie firmy bezpieczeństwa) mogą zostać błędnie odebrane jako negatywne dla samej spółki.
- **Przykłady błędnych klasyfikacji ze zbioru testowego:**
  1. _„The company said that sales in the three months to the end of March slid to EUR86 .4 m US$ 113.4 m from EUR91 .2 m last year .”_ — prawdziwa klasa: `negative`, predykcja: `positive` ($Z = [\text{neg}: 0{,}013,\ \text{neu}: -0{,}171,\ \text{pos}: 0{,}057]$; rzadsze słowo _slid_ nie przeważyło nad neutralno-pozytywnym kontekstem sprzedaży).
  2. _„F-Secure reported that : - The first half of 2008 has seen a growing number of targeted malware attacks on individuals , companies , and organizations .”_ — prawdziwa klasa: `neutral`, predykcja: `negative` ($Z = [\text{neg}: 0{,}078,\ \text{neu}: -0{,}054,\ \text{pos}: -0{,}071]$; słowa _malware_ i _attacks_ wywołały fałszywy alarm negatywny w czysto informacyjnym raporcie).
  3. _„Fiskars , the World 's 1 Scissors Brand TM , recently won Learning -« Magazine 's 2011 Teachers ' Choice Award for the Classroom .”_ — prawdziwa klasa: `positive`, predykcja: `neutral` (sukces produktowy bez typowego słownictwa finansowego typu _profit_ czy _revenue_).

**Kierunki dalszego rozwoju:** Zastąpienie prostych wektorów TF-IDF gęstymi osadzeniami kontekstowymi (np. dostrojonym modelem językowym **FinBERT**) oraz rozbudowa zbioru o dodatkowe przykłady wiadomości negatywnych.

## 8. Zastosowania w praktyce _(E · 4 pkt)_

> Co tu wpisać: gdzie podobne rozwiązania są używane w branży (firma, produkt, dziedzina). Jeśli rozmawialiście z praktykiem — co powiedział i co zmieniliście.

## 9. Zespół _(D · 8 pkt)_

> Co tu wpisać: kto za co odpowiadał. Jeśli role się zmieniały — napiszcie kiedy i dlaczego. Wkład powinien zgadzać się z historią repozytorium.

| Osoba | Za co odpowiadała |
| ----- | ----------------- |
| …     | …                 |

## 10. Uruchomienie _(C · 4 pkt)_

> Co tu wpisać: kroki od sklonowania repozytorium do wyniku, tak żeby obca osoba dała radę bez pytania autorów: wymagania, instalacja, polecenie uruchomienia, gdzie pojawi się wynik.

```
git clone <repozytorium>
cd <katalog>
pip install -r requirements.txt
python <plik>.py
```

## 11. Narzędzia AI

> Co tu wpisać: czy korzystaliście z narzędzi AI (np. ChatGPT, Claude, Copilot) i do czego — kod, tekst, debugowanie. Jeśli nie — napiszcie „nie korzystaliśmy”. Pamiętajcie: każdą linię kodu musicie umieć wyjaśnić na obronie.
