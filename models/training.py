import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Dodaj folder nadrzędny do ścieżki, aby móc importować z innych modułów
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_hm import create_model_hm
from models.evaluation import evaluate_model, calculate_metrics

class EarlyStopping:
    """
    Klasa do wczesnego zatrzymywania treningu, gdy model przestaje się poprawiać
    """
    def __init__(
        self, 
        patience: int = 10, 
        delta: float = 0.001, 
        path: str = 'checkpoints/best_model.pt'
    ):
        """
        Inicjalizuje mechanizm wczesnego zatrzymywania.
        
        Args:
            patience: Liczba epok bez poprawy, po których nastąpi zatrzymanie
            delta: Minimalna zmiana, aby uznać za poprawę
            path: Ścieżka do zapisania najlepszego modelu
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Sprawdza, czy należy zatrzymać trening i zapisuje model, jeśli jest najlepszy.
        
        Args:
            val_loss: Wartość funkcji straty na zbiorze walidacyjnym
            model: Model do zapisania, jeśli jest najlepszy
            
        Returns:
            True, jeśli trening powinien zostać zatrzymany
        """
        score = -val_loss  # Im mniejsza strata, tym lepszy model
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """
        Zapisuje model do pliku.
        
        Args:
            val_loss: Wartość funkcji straty na zbiorze walidacyjnym
            model: Model do zapisania
        """
        print(f'Walidacyjna strata zmniejszyła się ({self.best_score:.6f} --> {-val_loss:.6f}). Zapisuję model...')
        torch.save(model.state_dict(), self.path)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    checkpoint_dir: str = 'checkpoints'
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trenuje model rekomendacji.
    
    Args:
        model: Model do treningu
        train_loader: DataLoader z danymi treningowymi
        val_loader: DataLoader z danymi walidacyjnymi
        num_epochs: Liczba epok treningu
        learning_rate: Współczynnik uczenia
        weight_decay: Współczynnik regularyzacji L2
        patience: Liczba epok bez poprawy dla wczesnego zatrzymania
        checkpoint_dir: Katalog do zapisywania checkpointów
        
    Returns:
        Wytrenowany model i historię treningu
    """
    # Przygotuj katalog na checkpointy
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    # Urządzenie (CPU lub GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")
    
    # Przenieś model na urządzenie
    model = model.to(device)
    
    # Funkcja straty i optymalizator
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    
    # Historia treningu
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': []
    }
    
    # Główna pętla treningu
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Tryb treningu
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Przenieś dane na urządzenie
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Zerowanie gradientów
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            targets = batch['ocena']
            
            # Obliczenie straty
            loss = criterion(outputs, targets)
            
            # Backward pass i optymalizacja
            loss.backward()
            optimizer.step()
            
            # Aktualizacja statystyk
            train_loss += loss.item()
            batch_count += 1
        
        # Średnia strata na epoce
        train_loss /= batch_count
        history['train_loss'].append(train_loss)
        
        # Ewaluacja na zbiorze walidacyjnym
        val_loss, val_metrics = evaluate_model(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        
        # Wyświetl postęp
        print(f"Epoka {epoch+1}/{num_epochs} | "
              f"Trening: {train_loss:.4f} | "
              f"Walidacja: {val_loss:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | "
              f"MAE: {val_metrics['mae']:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print("Wczesne zatrzymanie treningu.")
            break
    
    # Czas treningu
    elapsed_time = time.time() - start_time
    print(f"Trening ukończony w {elapsed_time:.2f} sekund.")
    
    # Załaduj najlepszy model
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model, history

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Wizualizuje historię treningu.
    
    Args:
        history: Historia treningu z wartościami strat i metryk
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 8))
    
    # Wykres straty treningowej i walidacyjnej
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Strata treningowa')
    plt.plot(history['val_loss'], label='Strata walidacyjna')
    plt.title('Wartości funkcji straty w czasie treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Wykres metryk walidacyjnych
    plt.subplot(2, 1, 2)
    plt.plot(history['val_rmse'], label='RMSE (walidacja)')
    plt.plot(history['val_mae'], label='MAE (walidacja)')
    plt.title('Metryki walidacyjne w czasie treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()

def save_model_info(
    model: nn.Module, 
    history: Dict[str, List[float]],
    liczba_kategorii: Dict[str, int],
    model_params: Dict,
    metrics: Dict[str, float],
    save_dir: str = 'model_info'
):
    """
    Zapisuje informacje o modelu.
    
    Args:
        model: Wytrenowany model
        history: Historia treningu
        liczba_kategorii: Słownik z liczbą kategorii dla każdej cechy
        model_params: Parametry modelu
        metrics: Metryki ewaluacji modelu
        save_dir: Katalog do zapisania informacji
    """
    # Upewnij się, że katalog istnieje
    os.makedirs(save_dir, exist_ok=True)
    
    # Zapisz model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    
    # Zapisz informacje o modelu
    import json
    
    # Konwersja liczby_kategorii do formatu JSON
    liczba_kategorii_json = {k: int(v) for k, v in liczba_kategorii.items()}
    
    # Zapisz informacje jako JSON
    info = {
        'model_type': model.__class__.__name__,
        'model_params': model_params,
        'liczba_kategorii': liczba_kategorii_json,
        'metrics': metrics,
        'history': {k: [float(val) for val in v] for k, v in history.items()}
    }
    
    with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Informacje o modelu zapisane w: {save_dir}")

def load_model(
    model_dir: str, 
    liczba_kategorii: Dict[str, int],
    device: torch.device = None
) -> nn.Module:
    """
    Ładuje zapisany model.
    
    Args:
        model_dir: Katalog z zapisanym modelem
        liczba_kategorii: Słownik z liczbą kategorii dla każdej cechy
        device: Urządzenie, na którym ma być umieszczony model
        
    Returns:
        Załadowany model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Wczytaj informacje o modelu
    with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
        info = json.load(f)
    
    # Utwórz model
    model = create_model_hm(liczba_kategorii, **info['model_params'])
    
    # Załaduj wagi
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'final_model.pt'),
        map_location=device
    ))
    
    # Przenieś na urządzenie
    model = model.to(device)
    
    # Ustaw tryb ewaluacji
    model.eval()
    
    return model

if __name__ == "__main__":
    # Przykładowe użycie:
    from torch.utils.data import DataLoader
    import sys
    import os
    
    # Dodaj folder nadrzędny do ścieżki, aby móc importować z data
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.data_loader_hm import przygotuj_dane_hm
    
    # Przygotuj dane
    train_loader, val_loader, test_loader, liczba_kategorii, _, _ = przygotuj_dane_hm()
    
    if train_loader:
        # Utwórz model
        model = create_model_hm(liczba_kategorii, dim_embeddingu=16, hidden_dims=[64, 32])
        
        # Trenuj model
        model, history = train_model(model, train_loader, val_loader, num_epochs=5)
        
        # Wyświetl historię treningu
        plot_training_history(history)
        
        print("Trening ukończony pomyślnie!")
    else:
        print("Nie udało się przygotować danych do treningu.") 