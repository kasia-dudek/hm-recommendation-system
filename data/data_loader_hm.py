import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class HMDataset(Dataset):
    """
    Dataset PyTorch do obsługi danych H&M (produkty i transakcje).
    """
    def __init__(
        self, 
        transactions_df: pd.DataFrame, 
        articles_df: pd.DataFrame,
        user_encoders: Dict[str, LabelEncoder],
        item_encoders: Dict[str, LabelEncoder]
    ):
        """
        Inicjalizuje dataset.
        
        Args:
            transactions_df: DataFrame z transakcjami
            articles_df: DataFrame z produktami
            user_encoders: Słownik z encoderami cech użytkowników
            item_encoders: Słownik z encoderami cech produktów
        """
        self.transactions = transactions_df
        self.articles = articles_df
        self.user_encoders = user_encoders
        self.item_encoders = item_encoders
        
        # Przygotuj mapowanie article_id do wiersza w article_df dla szybkiego dostępu
        self.article_map = {row['article_id']: i for i, row in articles_df.iterrows()}
    
    def __len__(self) -> int:
        return len(self.transactions)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Zwraca pojedynczą próbkę danych.
        
        Args:
            idx: Indeks próbki
            
        Returns:
            Słownik z tensorami dla modelu
        """
        transaction = self.transactions.iloc[idx]
        customer_id = transaction['customer_id']
        article_id = int(transaction['article_id'])  # Upewniamy się, że to int
        
        # Zakodowane ID użytkownika i produktu
        try:
            user_id_encoded = self.user_encoders['user_id'].transform([customer_id])[0]
            item_id_encoded = self.item_encoders['item_id'].transform([article_id])[0]
            
            # Przygotowanie tensora cech użytkownika
            user_id_tensor = torch.tensor([user_id_encoded], dtype=torch.long)
            
            # Przygotowanie tensora cech produktu
            item_id_tensor = torch.tensor([item_id_encoded], dtype=torch.long)
            
            # Przygotowanie cech kategorycznych produktu
            if article_id in self.article_map:
                article_idx = self.article_map[article_id]
                article = self.articles.iloc[article_idx]
                
                # Cechy kategoryczne produktu - przekształcamy je tylko wtedy gdy istnieją
                features = {}
                if 'product_type_name' in article and 'product_type' in self.item_encoders:
                    prod_type_encoded = self.item_encoders['product_type'].transform([article['product_type_name']])[0]
                    features['product_type'] = torch.tensor([prod_type_encoded], dtype=torch.long)
                
                if 'colour_group_name' in article and 'colour' in self.item_encoders:
                    colour_encoded = self.item_encoders['colour'].transform([article['colour_group_name']])[0]
                    features['colour'] = torch.tensor([colour_encoded], dtype=torch.long)
                
                if 'perceived_colour_master_name' in article and 'colour_master' in self.item_encoders:
                    colour_master_encoded = self.item_encoders['colour_master'].transform([article['perceived_colour_master_name']])[0]
                    features['colour_master'] = torch.tensor([colour_master_encoded], dtype=torch.long)
                
                if 'department_name' in article and 'department' in self.item_encoders:
                    dept_encoded = self.item_encoders['department'].transform([article['department_name']])[0]
                    features['department'] = torch.tensor([dept_encoded], dtype=torch.long)
                
                if 'index_name' in article and 'index' in self.item_encoders:
                    index_encoded = self.item_encoders['index'].transform([article['index_name']])[0]
                    features['index'] = torch.tensor([index_encoded], dtype=torch.long)
                
                # Ocena jako docelowa zmienna (zakładamy że kupno = pozytywna ocena)
                result = {
                    'user_id': user_id_tensor,
                    'item_id': item_id_tensor,
                    'ocena': torch.tensor(1.0, dtype=torch.float),  # Zakładamy, że zakup = pozytywna ocena
                    'price': torch.tensor([transaction['price']], dtype=torch.float)
                }
                
                # Dodajemy cechy kategoryczne produktu
                result.update(features)
                
                return result
        except:
            # W przypadku błędu (np. brakującego artykułu), zwracamy minimalny zestaw danych
            return {
                'user_id': torch.tensor([0], dtype=torch.long),
                'item_id': torch.tensor([0], dtype=torch.long),
                'ocena': torch.tensor(0.0, dtype=torch.float),
                'price': torch.tensor([0.0], dtype=torch.float)
            }
        
        # Minimalny zestaw danych jako zabezpieczenie
        return {
            'user_id': torch.tensor([user_id_encoded], dtype=torch.long),
            'item_id': torch.tensor([item_id_encoded], dtype=torch.long),
            'ocena': torch.tensor(1.0, dtype=torch.float),  # Zakładamy, że zakup = pozytywna ocena
            'price': torch.tensor([transaction['price']], dtype=torch.float)
        }

def zaladuj_dane_hm(katalog_danych: str = 'dane_hm', sample_size: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ładuje dane H&M z plików CSV.
    
    Args:
        katalog_danych: Ścieżka do katalogu z danymi
        sample_size: Współczynnik próbkowania dla transactions (1.0 = pełne dane)
        
    Returns:
        Krotka (articles_df, transactions_df)
    """
    try:
        # Wczytaj dane o produktach
        articles_df = pd.read_csv(f"{katalog_danych}/articles.csv")
        
        # Wczytaj transakcje (potencjalnie próbkując aby zmniejszyć rozmiar)
        if sample_size < 1.0:
            transactions_df = pd.read_csv(
                f"{katalog_danych}/transactions_train.csv", 
                skiprows=lambda i: i > 0 and np.random.random() > sample_size
            )
        else:
            transactions_df = pd.read_csv(f"{katalog_danych}/transactions_train.csv")
        
        # Konwersja daty na format datetime
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
        
        print(f"Załadowano dane: {len(articles_df)} produktów, {len(transactions_df)} transakcji.")
        return articles_df, transactions_df
    
    except FileNotFoundError as e:
        print(f"Błąd podczas ładowania danych: {e}")
        return None, None

def przygotuj_encodery_hm(
    articles_df: pd.DataFrame, 
    transactions_df: pd.DataFrame
) -> Tuple[Dict[str, LabelEncoder], Dict[str, LabelEncoder]]:
    """
    Przygotowuje encodery dla cech kategorycznych.
    
    Args:
        articles_df: DataFrame z danymi produktów
        transactions_df: DataFrame z transakcjami
        
    Returns:
        Krotka (user_encoders, item_encoders)
    """
    # Encodery dla użytkowników
    user_encoders = {
        'user_id': LabelEncoder().fit(transactions_df['customer_id'].unique())
    }
    
    # Encodery dla produktów
    item_encoders = {
        'item_id': LabelEncoder().fit(articles_df['article_id'].unique())
    }
    
    # Encodery dla atrybutów produktów
    if 'product_type_name' in articles_df.columns:
        item_encoders['product_type'] = LabelEncoder().fit(articles_df['product_type_name'].fillna('Unknown'))
    
    if 'colour_group_name' in articles_df.columns:
        item_encoders['colour'] = LabelEncoder().fit(articles_df['colour_group_name'].fillna('Unknown'))
    
    if 'perceived_colour_master_name' in articles_df.columns:
        item_encoders['colour_master'] = LabelEncoder().fit(articles_df['perceived_colour_master_name'].fillna('Unknown'))
    
    if 'department_name' in articles_df.columns:
        item_encoders['department'] = LabelEncoder().fit(articles_df['department_name'].fillna('Unknown'))
    
    if 'index_name' in articles_df.columns:
        item_encoders['index'] = LabelEncoder().fit(articles_df['index_name'].fillna('Unknown'))
    
    return user_encoders, item_encoders

def uzyskaj_liczbe_kategorii_hm(
    user_encoders: Dict[str, LabelEncoder], 
    item_encoders: Dict[str, LabelEncoder]
) -> Dict[str, int]:
    """
    Zwraca informację o liczbie kategorii dla każdej cechy kategorycznej.
    
    Args:
        user_encoders: Słownik z encoderami cech użytkowników
        item_encoders: Słownik z encoderami cech produktów
        
    Returns:
        Słownik z liczbą kategorii dla każdej cechy
    """
    liczba_kategorii = {}
    
    # Dla użytkowników
    for nazwa, encoder in user_encoders.items():
        liczba_kategorii[nazwa] = len(encoder.classes_)
    
    # Dla produktów
    for nazwa, encoder in item_encoders.items():
        liczba_kategorii[nazwa] = len(encoder.classes_)
    
    return liczba_kategorii

def podziel_dane_hm(
    transactions_df: pd.DataFrame, 
    test_size: float = 0.2, 
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dzieli dane transakcji na zestawy treningowe, walidacyjne i testowe.
    
    Args:
        transactions_df: DataFrame z transakcjami
        test_size: Proporcja danych testowych
        val_size: Proporcja danych walidacyjnych
        
    Returns:
        Krotka (train_df, val_df, test_df)
    """
    # Sortuj po dacie transakcji, aby zachować czasowy charakter danych
    transactions_df = transactions_df.sort_values('t_dat')
    
    # Podziel dane po czasie (ostatnie transakcje jako testowe)
    n = len(transactions_df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = transactions_df.iloc[:train_end]
    val_df = transactions_df.iloc[train_end:val_end]
    test_df = transactions_df.iloc[val_end:]
    
    print(f"Podział danych: {len(train_df)} treningowych, {len(val_df)} walidacyjnych, {len(test_df)} testowych.")
    return train_df, val_df, test_df

def utworz_data_loadery_hm(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    articles_df: pd.DataFrame,
    user_encoders: Dict[str, LabelEncoder],
    item_encoders: Dict[str, LabelEncoder],
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tworzy DataLoadery PyTorch dla zbiorów treningowego, walidacyjnego i testowego.
    
    Args:
        train_df: DataFrame z danymi treningowymi
        val_df: DataFrame z danymi walidacyjnymi
        test_df: DataFrame z danymi testowymi
        articles_df: DataFrame z danymi produktów
        user_encoders: Słownik z encoderami cech użytkowników
        item_encoders: Słownik z encoderami cech produktów
        batch_size: Rozmiar batcha
        
    Returns:
        Krotka (train_loader, val_loader, test_loader)
    """
    # Tworzenie datasetów
    train_dataset = HMDataset(train_df, articles_df, user_encoders, item_encoders)
    val_dataset = HMDataset(val_df, articles_df, user_encoders, item_encoders)
    test_dataset = HMDataset(test_df, articles_df, user_encoders, item_encoders)
    
    # Tworzenie dataloaderów
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def przygotuj_dane_hm(
    katalog_danych: str = 'dane_hm', 
    batch_size: int = 64,
    test_size: float = 0.2, 
    val_size: float = 0.1,
    sample_size: float = 0.01
) -> Tuple[
    DataLoader, DataLoader, DataLoader, 
    Dict[str, int], 
    Dict[str, LabelEncoder], Dict[str, LabelEncoder]
]:
    """
    Kompletna funkcja przygotowująca dane do treningu modelu.
    
    Args:
        katalog_danych: Ścieżka do katalogu z danymi
        batch_size: Rozmiar batcha
        test_size: Proporcja danych testowych
        val_size: Proporcja danych walidacyjnych
        sample_size: Współczynnik próbkowania dla transakcji
        
    Returns:
        Krotka (train_loader, val_loader, test_loader, liczba_kategorii, user_encoders, item_encoders)
    """
    # Wczytaj dane
    articles_df, transactions_df = zaladuj_dane_hm(katalog_danych, sample_size)
    
    if articles_df is None or transactions_df is None:
        return None, None, None, None, None, None
    
    # Podziel dane
    train_df, val_df, test_df = podziel_dane_hm(transactions_df, test_size, val_size)
    
    # Przygotuj encodery
    user_encoders, item_encoders = przygotuj_encodery_hm(articles_df, transactions_df)
    
    # Uzyskaj liczby kategorii
    liczba_kategorii = uzyskaj_liczbe_kategorii_hm(user_encoders, item_encoders)
    
    # Utwórz data loadery
    train_loader, val_loader, test_loader = utworz_data_loadery_hm(
        train_df, val_df, test_df, articles_df, 
        user_encoders, item_encoders, batch_size
    )
    
    return train_loader, val_loader, test_loader, liczba_kategorii, user_encoders, item_encoders

if __name__ == "__main__":
    # Przykładowe użycie:
    train_loader, val_loader, test_loader, liczba_kategorii, _, _ = przygotuj_dane_hm()
    
    if train_loader:
        print("Liczby kategorii dla cech:")
        for cecha, liczba in liczba_kategorii.items():
            print(f"  {cecha}: {liczba}") 