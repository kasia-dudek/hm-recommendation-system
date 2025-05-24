import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class HMRecommender(nn.Module):
    """
    Model rekomendacji odzieży H&M wykorzystujący embeddingi i MLP.
    """
    def __init__(
        self,
        liczba_kategorii: Dict[str, int],
        dim_embeddingu: int = 32,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        """
        Inicjalizuje model rekomendacji odzieży H&M.
        
        Args:
            liczba_kategorii: Słownik z liczbą kategorii dla każdej cechy
            dim_embeddingu: Wymiar embedingów
            hidden_dims: Lista z liczbą neuronów w warstwach ukrytych
            dropout: Współczynnik dropout
        """
        super().__init__()
        
        # Podstawowe cechy, które zawsze muszą być dostępne
        self.podstawowe_cechy = ['user_id', 'item_id']
        
        # Opcjonalne cechy kategoryczne, które mogą być dostępne
        self.opcjonalne_cechy = [
            'product_type', 'colour', 'colour_master', 
            'department', 'index'
        ]
        
        # Inicjalizacja listy wszystkich cech, które będą używane w modelu
        # Zaczynamy od podstawowych cech
        self.cechy_kategoryczne = self.podstawowe_cechy.copy()
        
        # Dodajemy opcjonalne cechy, jeśli są dostępne w liczbie_kategorii
        for cecha in self.opcjonalne_cechy:
            if cecha in liczba_kategorii:
                self.cechy_kategoryczne.append(cecha)
        
        # Inicjalizacja warstw embeddingów dla wszystkich cech kategorycznych
        self.embeddings = nn.ModuleDict()
        for nazwa in self.cechy_kategoryczne:
            self.embeddings[nazwa] = nn.Embedding(
                num_embeddings=liczba_kategorii[nazwa],
                embedding_dim=dim_embeddingu
            )
        
        # Liczba cech numerycznych (zawsze mamy cenę)
        self.num_cech_numerycznych = 1
        
        # Wymiar całkowitego wektora cech (embeddingi + cechy numeryczne)
        input_dim = len(self.cechy_kategoryczne) * dim_embeddingu + self.num_cech_numerycznych
        
        # Warstwy MLP
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Warstwa wyjściowa
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid na wyjściu, aby mieć oceny w zakresie [0, 1]
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass modelu.
        
        Args:
            batch: Słownik z tensorami wejściowymi
            
        Returns:
            Tensor z przewidywanymi ocenami
        """
        # Lista do przechowywania cech
        all_features = []
        
        # Przetwarzanie cech kategorycznych
        for nazwa in self.cechy_kategoryczne:
            if nazwa in batch:
                # Pobierz embedingi
                emb = self.embeddings[nazwa](batch[nazwa].squeeze(1))
                all_features.append(emb)
            else:
                # Jeśli cecha nie jest dostępna, wypełnij zerami
                dummy_tensor = torch.zeros(
                    batch[self.podstawowe_cechy[0]].shape[0],
                    self.embeddings[nazwa].embedding_dim,
                    device=batch[self.podstawowe_cechy[0]].device
                )
                all_features.append(dummy_tensor)
        
        # Przetwarzanie cech numerycznych - zawsze mamy cenę
        price_feature = batch['price']
        
        # Łączenie wszystkich cech
        all_features.append(price_feature)
        x = torch.cat(all_features, dim=1)
        
        # Przetwarzanie przez MLP
        output = self.mlp(x)
        
        # Skalowanie wyjścia do zakresu [0, 5] dla ocen
        return output.squeeze(1) * 5.0


def create_model_hm(
    liczba_kategorii: Dict[str, int],
    **kwargs
) -> nn.Module:
    """
    Fabryka modeli - tworzy model H&M.
    
    Args:
        liczba_kategorii: Słownik z liczbą kategorii dla każdej cechy
        **kwargs: Dodatkowe argumenty dla modelu
        
    Returns:
        Instancję modelu
    """
    return HMRecommender(liczba_kategorii, **kwargs)


if __name__ == "__main__":
    # Przykładowe użycie:
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
        
        # Wyświetl informacje o modelu
        print(f"Model: {model.__class__.__name__}")
        print(f"Liczba parametrów: {sum(p.numel() for p in model.parameters())}")
        
        # Testowy forward pass
        for batch in train_loader:
            outputs = model(batch)
            print(f"Przykładowe wyjścia modelu: {outputs[:5]}")
            print(f"Kształt wyjścia: {outputs.shape}")
            break 