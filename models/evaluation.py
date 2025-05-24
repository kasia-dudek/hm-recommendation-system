import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Funkcje do obliczania metryk rekomendacji
def calculate_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Oblicza metryki dla modelu rekomendacji.
    
    Args:
        predictions: Przewidywane oceny
        targets: Rzeczywiste oceny
        
    Returns:
        Słownik z metrykami (RMSE, MAE)
    """
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # MAE - Mean Absolute Error
    mae = mean_absolute_error(targets, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae
    }

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Ewaluuje model na podanym zbiorze danych.
    
    Args:
        model: Model do ewaluacji
        data_loader: DataLoader z danymi do ewaluacji
        device: Urządzenie, na którym ma być przeprowadzona ewaluacja
        
    Returns:
        Wartość funkcji straty i słownik z metrykami
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Przenieś dane na urządzenie
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            targets = batch['ocena']
            
            # Obliczenie straty
            loss = criterion(outputs, targets)
            
            # Zapisz predykcje i cele do późniejszej ewaluacji
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Aktualizacja statystyk
            running_loss += loss.item()
            batch_count += 1
    
    # Średnia strata
    avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
    
    # Konwersja list do numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Obliczenie metryk
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics

def generate_top_k_recommendations(
    model: nn.Module,
    user_id: int,
    sukienki_df: pd.DataFrame,
    uzytkownik_encoders: Dict,
    sukienka_encoders: Dict,
    top_k: int = 10,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Generuje top-k rekomendacji dla konkretnego użytkownika.
    
    Args:
        model: Wytrenowany model
        user_id: ID użytkownika
        sukienki_df: DataFrame z danymi sukienek
        uzytkownik_encoders: Encodery cech użytkowników
        sukienka_encoders: Encodery cech sukienek
        top_k: Liczba rekomendacji do wygenerowania
        device: Urządzenie, na którym ma być przeprowadzona ewaluacja
        
    Returns:
        DataFrame z top-k rekomendacjami dla użytkownika
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Przygotuj zakodowane ID użytkownika
    user_id_encoded = uzytkownik_encoders['user_id'].transform([user_id])[0]
    
    # Przygotuj dane do oceny wszystkich sukienek
    all_predictions = []
    
    with torch.no_grad():
        for _, sukienka in sukienki_df.iterrows():
            item_id = sukienka['item_id']
            
            # Przygotuj cechy dla rekomendacji
            batch = {
                'user_id': torch.tensor([[user_id_encoded]], dtype=torch.long).to(device),
                'item_id': torch.tensor([[sukienka_encoders['item_id'].transform([item_id])[0]]], dtype=torch.long).to(device),
                'kolor': torch.tensor([[sukienka_encoders['kolor'].transform([sukienka['kolor']])[0]]], dtype=torch.long).to(device),
                'fason': torch.tensor([[sukienka_encoders['fason'].transform([sukienka['fason']])[0]]], dtype=torch.long).to(device),
                'material': torch.tensor([[sukienka_encoders['material'].transform([sukienka['material']])[0]]], dtype=torch.long).to(device),
                'okazja': torch.tensor([[sukienka_encoders['okazja'].transform([sukienka['okazja']])[0]]], dtype=torch.long).to(device),
                'cena': torch.tensor([[sukienka['cena']]], dtype=torch.float).to(device),
                'popularnosc': torch.tensor([[sukienka['popularnosc']]], dtype=torch.float).to(device)
            }
            
            # Przewidywana ocena
            prediction = model(batch).item()
            
            # Dodaj do wyników
            all_predictions.append({
                'item_id': item_id,
                'kolor': sukienka['kolor'],
                'fason': sukienka['fason'],
                'material': sukienka['material'],
                'okazja': sukienka['okazja'],
                'cena': sukienka['cena'],
                'rozmiar': sukienka['rozmiar'],
                'popularnosc': sukienka['popularnosc'],
                'przewidywana_ocena': prediction
            })
    
    # Konwersja do DataFrame
    recommendations_df = pd.DataFrame(all_predictions)
    
    # Sortowanie po przewidywanej ocenie (malejąco)
    recommendations_df = recommendations_df.sort_values('przewidywana_ocena', ascending=False)
    
    # Zwróć top-k rekomendacji
    return recommendations_df.head(top_k).reset_index(drop=True)

def plot_recommendations(recommendations_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Wizualizuje rekomendacje dla użytkownika.
    
    Args:
        recommendations_df: DataFrame z rekomendacjami
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 8))
    
    # Przygotuj dane do wykresu
    items = recommendations_df['item_id'].astype(str)
    scores = recommendations_df['przewidywana_ocena']
    colors = [plt.cm.viridis(i/float(len(items))) for i in range(len(items))]
    
    # Stwórz wykres słupkowy
    plt.barh(items, scores, color=colors)
    plt.xlabel('Przewidywana ocena')
    plt.ylabel('ID sukienki')
    plt.title('Top rekomendacje sukienek')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 5.5)  # Zakres ocen 0-5
    
    # Dodaj informacje o cechach sukienek
    for i, (_, row) in enumerate(recommendations_df.iterrows()):
        plt.text(
            row['przewidywana_ocena'] + 0.1, 
            i, 
            f"{row['kolor']}, {row['fason']}, {row['material']}, {row['cena']:.0f} zł",
            verticalalignment='center'
        )
    
    plt.tight_layout()
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()

def evaluate_recommendations_for_users(
    model: nn.Module,
    interakcje_df: pd.DataFrame,
    sukienki_df: pd.DataFrame,
    uzytkownik_encoders: Dict,
    sukienka_encoders: Dict,
    k: int = 10,
    num_users: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[int, float]]:
    """
    Ewaluuje rekomendacje dla losowej próbki użytkowników.
    
    Args:
        model: Wytrenowany model
        interakcje_df: DataFrame z interakcjami
        sukienki_df: DataFrame z danymi sukienek
        uzytkownik_encoders: Encodery cech użytkowników
        sukienka_encoders: Encodery cech sukienek
        k: Liczba rekomendacji do wygenerowania dla każdego użytkownika
        num_users: Liczba użytkowników do ewaluacji
        device: Urządzenie, na którym ma być przeprowadzona ewaluacja
        
    Returns:
        Słownik z metrykami dla każdego użytkownika
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Wybierz losową próbkę użytkowników
    unique_users = interakcje_df['user_id'].unique()
    selected_users = np.random.choice(unique_users, min(num_users, len(unique_users)), replace=False)
    
    results = {}
    
    for user_id in selected_users:
        # Pobierz interakcje użytkownika
        user_interactions = interakcje_df[interakcje_df['user_id'] == user_id]
        
        # Wygeneruj rekomendacje dla użytkownika
        recommendations = generate_top_k_recommendations(
            model, user_id, sukienki_df, uzytkownik_encoders, sukienka_encoders, k, device
        )
        
        # Oblicz Precision@K i Recall@K
        # Zakładamy, że interakcje z oceną >= 3.5 są pozytywnymi preferencjami (zmienione z 4.0)
        positive_interactions = set(user_interactions[user_interactions['ocena'] >= 3.5]['item_id'])
        recommended_items = set(recommendations['item_id'])
        
        relevant_and_recommended = len(positive_interactions.intersection(recommended_items))
        
        precision_at_k = relevant_and_recommended / k if k > 0 else 0
        recall_at_k = relevant_and_recommended / len(positive_interactions) if len(positive_interactions) > 0 else 0
        
        # Zapisz wyniki
        results[int(user_id)] = {
            'precision@k': precision_at_k,
            'recall@k': recall_at_k,
            'num_positive_interactions': len(positive_interactions),
            'num_recommendations': len(recommendations)
        }
    
    return results

def plot_recommendation_metrics(results: Dict[int, Dict[str, float]], save_path: Optional[str] = None):
    """
    Wizualizuje metryki rekomendacji dla użytkowników.
    
    Args:
        results: Słownik z metrykami dla każdego użytkownika
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 8))
    
    # Przygotuj dane do wykresu
    users = list(results.keys())
    precision_values = [results[user]['precision@k'] for user in users]
    recall_values = [results[user]['recall@k'] for user in users]
    
    # Oblicz statystyki
    avg_precision = np.mean(precision_values)
    avg_recall = np.mean(recall_values)
    max_precision = max(precision_values) if precision_values else 0
    max_recall = max(recall_values) if recall_values else 0
    
    # Ustaw odpowiedni zakres osi Y, aby wartości były dobrze widoczne
    max_value = max(max_precision, max_recall, 0.1)  # Minimum 0.1 dla lepszej wizualizacji
    y_max = max_value * 1.2  # Dodatkowa przestrzeń dla etykiet
    
    x = np.arange(len(users))
    width = 0.35
    
    # Stwórz wykres słupkowy
    plt.bar(x - width/2, precision_values, width, label='Precision@K')
    plt.bar(x + width/2, recall_values, width, label='Recall@K')
    
    plt.xlabel('ID użytkownika')
    plt.ylabel('Wartość')
    plt.title('Metryki rekomendacji dla użytkowników')
    plt.xticks(x, [str(user) for user in users])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, y_max)
    
    # Dodaj średnie wartości
    plt.axhline(y=avg_precision, color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=avg_recall, color='orange', linestyle='--', alpha=0.5)
    
    # Dodaj etykiety ze statystykami
    plt.text(
        x[-1], avg_precision + 0.02, 
        f'Średni Precision@K: {avg_precision:.3f}', 
        ha='right', va='bottom'
    )
    plt.text(
        x[-1], avg_recall + 0.02, 
        f'Średni Recall@K: {avg_recall:.3f}', 
        ha='right', va='bottom'
    )
    
    # Dodaj etykiety z wartościami nad słupkami
    for i, v in enumerate(precision_values):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(recall_values):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    return avg_precision, avg_recall

if __name__ == "__main__":
    # Przykładowe użycie:
    import sys
    import os
    
    # Dodaj folder nadrzędny do ścieżki, aby móc importować z innych modułów
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.data_loader_hm import przygotuj_dane_hm, zaladuj_dane_hm
    from models.model_hm import create_model_hm
    from models.training import train_model
    
    # Przygotuj dane
    train_loader, val_loader, test_loader, liczba_kategorii, uzytkownik_encoders, sukienka_encoders = przygotuj_dane_hm()
    
    if train_loader:
        # Wczytaj dane do ewaluacji rekomendacji
        sukienki_df, uzytkownicy_df = zaladuj_dane_hm()
        
        # Utwórz i trenuj model (lub wczytaj już wytrenowany)
        # ...
        
        print("Moduł evaluation.py - funkcje ewaluacji dla systemu rekomendacji H&M")
        print("Aby użyć ewaluacji, zaimportuj odpowiednie funkcje w swoim kodzie.")
        print("Przykład: from models.evaluation import evaluate_model")
    else:
        print("Nie udało się przygotować danych do ewaluacji.") 