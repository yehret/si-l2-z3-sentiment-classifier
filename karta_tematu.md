---
share_link: https://share.note.sx/mdxqmhwe#9dalkcKZZG9oTSGt+Dqi/zjv8lvS1SslMGr78DAE6fY
share_updated: 2026-09-27T09:38:22+02:00
---

# Karta zgłoszenia tematu projektu

**Grupa:** L3 (niestacjonarna) / L2 (stacjonarna) — zostawcie właściwe **Data zgłoszenia:**

---

## 1. Skład zespołu

| Imię i nazwisko | Nr albumu | Rola w zespole                                                |
| --------------- | --------- | ------------------------------------------------------------- |
| Serhii Samoliuk | 31070     | Implementacja rdzenia, dokumentacja                           |
| Mariia Minakova | 31100     | Przygotowanie i eksploracja danych, dokumentacja, prezentacja |
|                 |           |                                                               |

Rola to nie stanowisko, tylko deklaracja, za co ta osoba odpowiada: dane, implementacja, ewaluacja, dokumentacja, koordynacja.

## 2. Tytuł projektu

**Po polsku:**
Klasyfikator wiadomości finansowych
**Po angielsku:**
Financial News Sentiment Classifier

## 3. Problem

_Co system ma robić i dla kogo. Opis sytuacji, nie nazwy technologii. 3–5 zdań._
Inwestor giełdowy codziennie przegląda dziesiątki newsów, aby ocenić nastroje wokół wybranych spółek. Ręczne czytanie każdego komunikatu zajmuje godziny i łatwo w nim o pominięcie ważnej informacji. System automatycznie klasyfikuje krótkie wiadomości finansowe jako pozytywne, neutralne lub negatywne. Dzięki temu inwestor od razu widzi ogólny bilans nastrojów rynkowych.

## 4. Dlaczego zwykły algorytm nie wystarczy

_Wskażcie jeden z trzech powodów omawianych na zajęciach i uzasadnijcie:_

- [ x ] nie umiemy zapisać reguły krok po kroku
- [ ] przepis istnieje, ale jest obliczeniowo za drogi
- [ ] reguł jest za dużo i zmieniają się w czasie

_Uzasadnienie, 2–3 zdania:_
Nie da się tu napisać prostych reguł if-else, bo pojedyncze słowa w finansach bywają mylące, „wzrost” może dotyczyć zarówno zysków, jak i długów firmy. Zwykłe zliczanie słów kluczowych nie łapie takiego kontekstu i daje losowe wyniki. Dlatego zamiast pisać reguły ręcznie, wytrenujemy model na oznaczonych wcześniej wiadomościach.

---

> Punkty poniżej wypełniacie do 3.10.

---

## 5. Technologia SI

- [ ] sieć neuronowa
- [ ] system ekspertowy / wnioskowanie regułowe
- [ ] algorytm genetyczny
- [ x ] klasyczne uczenie maszynowe
- [ ] inne:

_Dlaczego akurat ta technologia pasuje do tego problemu, 2–3 zdania:_
Chcemy przyporządkować każdy nagłówek do jednej z trzech klas na podstawie gotowej bazy przykładów, do czego najlepiej nadaje się zwykła klasyfikacja. Proste algorytmy uczenia maszynowego połączone z wektoryzacją tekstu dobrze łapią powtarzające się zwroty w krótkich newsach

## 6. Dane

|                             |                                                                            |
| --------------------------- | -------------------------------------------------------------------------- |
| Źródło                      | https://huggingface.co/datasets/takala/financial_phrasebank                |
| Rozmiar i format            | 4840 zdań o temacie finansów w języku angielskim (699 kB), format json/csv |
| Czy są już dostępne?        | tak                                                                        |
| Jeśli nie — plan pozyskania |                                                                            |

## 7. Stos technologiczny

_Język, biblioteki, środowisko uruchomieniowe. Pamiętajcie: wszystko ma działać na CPU, bez karty graficznej._
Python, numpy, pandas, scikit-learn, matplotlib/seaborn

## 8. Kryterium sukcesu

_Po czym poznamy, że projekt działa. Jaka metryka, jaki próg. Liczba, nie deklaracja._
Chcemy osiągnąć co najmniej 75% trafności (Accuracy) na zbiorze testowym i uzyskać wynik lepszy niż dwa proste podejścia: wybieranie zawsze najczęstszej klasy neutralnej, ktora stanowi 60% zbioru, oraz zwykłe zliczanie słów kluczowych ze słownika. Do tego wygenerujemy macierz pomyłek i opisujemy, które klasy najczęściej się ze sobą mylą

## 9. Zakres minimalny

_Co powstanie na pewno. To jest obietnica, z której będziecie rozliczeni._
Przygotowanie danych i podział na zbiór uczący i testowy (80/20), zamiana tekstu na wektory, napisanie prostego rozwiązania do porównania, implementacja własnego perceptrona i zestawienie go z gotowym modelem z biblioteki, policzenie wyników oraz wygenerowanie macierzy pomyłek.

## 10. Zakres opcjonalny

_Co dorobicie, jeśli starczy czasu. Brak realizacji tej części nie obniża oceny._
Sprawdzenie, które słowa najmocniej wpływają na decyzję modelu, czyli wagi cech. Przetestowanie wytrenowanego modelu na kilkunastu aktualnych nagłówkach i policzenie dla niej ogólnego bilansu nastrojów.

## 11. Repozytorium

_Link:_
https://github.com/yehret/uni-ai-project

## 12. Główne ryzyko i plan awaryjny

_Co najprawdopodobniej pójdzie nie tak i co wtedy zrobicie._
Główne ryzyko to przeciek danych oraz mocno nierówne klasy, bo w danych przeważają wiadomości neutralne.
Plan awaryjny: dzielimy zbiór na uczący i testowy na samym początku pracy i nie ruszamy części testowej

## 13. Podział pracy w czasie

| Etap                     | Kto                         | Szacowany czas           |
| ------------------------ | --------------------------- | ------------------------ |
| Dane i przygotowanie     | Mariia (10 h), Serhii (4 h) | 14 h                     |
| Implementacja            | Serhii (16 h)               | 16 h                     |
| Eksperymenty i ewaluacja | Mariia (8 h), Serhii (4 h)  | 12 h                     |
| Dokumentacja PL          | Mariia (4 h), Serhii (4 h)  | 8 h                      |
| Dokumentacja EN          | Mariia (3 h), Serhii (1 h)  | 4 h                      |
| Prezentacja              | Mariia (4 h)                | 4 h                      |
| **Razem na osobę**       |                             | **29 h / os. (28–30 h)** |

---

## Decyzja prowadzącego

_Wypełnia prowadzący — nie edytujcie tej sekcji._

- [ ] zatwierdzony
- [ ] do poprawy

**Uwagi:**
