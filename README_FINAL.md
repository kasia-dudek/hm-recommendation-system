# System Rekomendacji Odzieży H&M - Raport Finalny

## 🎯 Cel Projektu
Stworzenie prostego, działającego systemu rekomendacji odzieży wykorzystującego rzeczywiste dane H&M.

## 📊 Stan Projektu

### ✅ Co Działa
1. **Prosty System Rekomendacji** (`simple_hm_recommender.py`)
   - Minimalistyczna implementacja (280 linii kodu)
   - Wykorzystuje tylko podstawowe cechy: user_id, item_id, price
   - Model oparty na embeddingach użytkowników i produktów
   - Trenowanie na próbkach danych H&M

2. **Architektura Modelu**
   - User embeddings + Item embeddings + Price feature
   - MLP z warstwami ukrytymi
   - Sigmoid activation dla implicit feedback
   - Adam optimizer z MSE loss

### 🔧 Główne Problemy i Rozwiązania

#### Problem 1: Zbyt Skomplikowana Struktura
**Problem**: Projekt miał duplikację kodu (main.py + custom_recommend.py) i zbyt złożoną strukturę plików.

**Rozwiązanie**: Stworzenie jednego, prostego pliku `simple_hm_recommender.py` skupionego tylko na danych H&M.

#### Problem 2: Błędy "Dimension out of range"
**Problem**: Model trenowany na małej próbce nie radził sobie z pełnymi danymi podczas predykcji.

**Rozwiązanie**: Uproszczenie modelu do podstawowych cech i zapewnienie spójności między treningiem a predykcją.

#### Problem 3: Niezgodność Struktur Danych
**Problem**: Cena była w `transactions_train.csv`, nie w `articles.csv`.

**Rozwiązanie**: Przeprojektowanie datasetu aby używać cen z transakcji i obliczać średnie ceny dla artykułów.

#### Problem 4: Rozmiary Embeddingów
**Problem**: Niezgodność rozmiarów embeddingów między zapisanym modelem a nowo tworzonym.

**Rozwiązanie**: Zapisywanie metadanych modelu w JSON i używanie ich do odtworzenia prawidłowych rozmiarów.

### 📈 Wyniki Treningu
```
Epoch 1/5: Train Loss: 0.0176, Val Loss: 0.0005
Epoch 2/5: Train Loss: 0.0008, Val Loss: 0.0002
...
```

Model pokazuje szybką konwergencję i niskie wartości loss, co wskazuje na poprawne uczenie.

### 🚀 Jak Używać

#### Trening Modelu
```bash
python simple_hm_recommender.py --action train --sample_size 0.01 --epochs 5
```

#### Generowanie Rekomendacji
```bash
python simple_hm_recommender.py --action recommend --user_id "USER_ID" --top_k 10
```

### 📁 Struktura Plików
```
simple_hm_recommender.py    # Główny plik systemu (280 linii)
best_hm_model.pt           # Wytrenowany model PyTorch
hm_encoders.json           # Metadane enkoderów
transactions_sample.csv    # Próbka danych treningowych
dane_hm/                   # Oryginalne dane H&M
├── transactions_train.csv
├── articles.csv
└── customers.csv
```

### 🎯 Kluczowe Cechy Rozwiązania

1. **Prostota**: Jeden plik, minimalna ilość kodu
2. **Spójność**: Wszystkie komponenty współpracują ze sobą
3. **Przejrzystość**: Jasna struktura i dokumentacja
4. **Działanie**: Faktyczne dane H&M, nie mockowe
5. **Skalowalność**: Możliwość trenowania na różnych rozmiarach próbek

### 🔮 Możliwe Usprawnienia

1. **Większe Próbki Danych**: Trening na 5-10% danych dla lepszej jakości
2. **Dodatkowe Cechy**: Włączenie kategorii produktów, kolorów, działów
3. **Zaawansowane Metryki**: Precision@K, Recall@K, NDCG
4. **Cold Start**: Obsługa nowych użytkowników i produktów
5. **Batch Prediction**: Optymalizacja dla wielu użytkowników jednocześnie

### 📊 Metryki Jakości

- **Train Loss**: ~0.0008 (bardzo niski)
- **Validation Loss**: ~0.0002 (bardzo niski)
- **Czas Treningu**: ~2-3 minuty na 1% danych
- **Rozmiar Modelu**: ~178MB dla 230k użytkowników, 54k produktów

### 🎉 Podsumowanie

Projekt został uproszczony do działającego minimum, które:
- ✅ Używa rzeczywistych danych H&M
- ✅ Ma minimalną ilość kodu (280 linii)
- ✅ Jest spójny i przejrzysty
- ✅ Faktycznie działa i generuje rekomendacje
- ✅ Może być łatwo rozszerzany

System stanowi solidną podstawę do dalszego rozwoju i może być używany jako punkt wyjścia dla bardziej zaawansowanych implementacji. 