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

> Co tu wpisać:
>
> - schemat działania systemu (lista kroków albo diagram),
> - opis algorytmu SI z jego podstawą matematyczną: wzór lub pojęcie z zajęć (np. funkcja oceny, gradient, entropia) i co oznacza w Waszym problemie,
> - kluczowe parametry i ich wartości (np. rozmiar populacji, liczba epok, głębokość drzewa).

## 5. Ewaluacja _(B · 8 pkt)_

> Co tu wpisać: metryka (ta z karty, pkt 8), punkt odniesienia (prosta metoda z karty), wyniki na kilku przypadkach — w tabeli albo na wykresie. Jedno–dwa zdania wniosku: czy system jest lepszy od punktu odniesienia i o ile.

| Przypadek / zbiór | Punkt odniesienia | Nasz system |
| ----------------- | ----------------- | ----------- |
| …                 | …                 | …           |

## 6. Wymagania niefunkcjonalne _(A · 5 pkt)_

> Co tu wpisać: ile trwa działanie (na czym mierzone), co się dzieje przy złych lub brakujących danych, czy wyniki są powtarzalne (ziarno losowości).

## 7. Ograniczenia _(E · 4 pkt)_

> Co tu wpisać: kiedy system się myli albo nie zadziała — najlepiej na konkretnym przykładzie. Co zrobilibyście dalej, mając więcej czasu. Czy zastosowana technologia jest dziś nadal stosowana, czy zastąpiły ją nowsze.

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
