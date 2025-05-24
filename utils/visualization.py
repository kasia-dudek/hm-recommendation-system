import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

# Ustawienia dla wykresów
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')
sns.set(style="whitegrid")


def plot_rating_distribution(interakcje_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Wizualizuje rozkład ocen w danych.
    
    Args:
        interakcje_df: DataFrame z interakcjami
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(10, 6))
    
    # Rozkład ocen
    sns.histplot(interakcje_df['ocena'], bins=10, kde=True)
    plt.title('Rozkład ocen sukienek', fontsize=16)
    plt.xlabel('Ocena', fontsize=12)
    plt.ylabel('Liczba interakcji', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_user_activity(interakcje_df: pd.DataFrame, n_users: int = 20, save_path: Optional[str] = None):
    """
    Wizualizuje aktywność użytkowników.
    
    Args:
        interakcje_df: DataFrame z interakcjami
        n_users: Liczba najaktywniejszych użytkowników do pokazania
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 6))
    
    # Liczba interakcji na użytkownika
    user_activity = interakcje_df['user_id'].value_counts().reset_index()
    user_activity.columns = ['user_id', 'liczba_interakcji']
    
    # Pokaż topowych użytkowników
    top_users = user_activity.head(n_users)
    
    # Wykres
    sns.barplot(x='user_id', y='liczba_interakcji', data=top_users)
    plt.title(f'Top {n_users} najaktywniejszych użytkowników', fontsize=16)
    plt.xlabel('ID użytkownika', fontsize=12)
    plt.ylabel('Liczba interakcji', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_item_popularity(interakcje_df: pd.DataFrame, sukienki_df: pd.DataFrame, n_items: int = 20, 
                          add_info: bool = True, save_path: Optional[str] = None):
    """
    Wizualizuje popularność sukienek.
    
    Args:
        interakcje_df: DataFrame z interakcjami
        sukienki_df: DataFrame z danymi sukienek
        n_items: Liczba najpopularniejszych sukienek do pokazania
        add_info: Czy dodać informacje o sukienkach
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(14, 6))
    
    # Liczba interakcji na sukienkę
    item_popularity = interakcje_df['item_id'].value_counts().reset_index()
    item_popularity.columns = ['item_id', 'liczba_interakcji']
    
    # Dodaj średnią ocenę
    item_ratings = interakcje_df.groupby('item_id')['ocena'].mean().reset_index()
    item_ratings.columns = ['item_id', 'srednia_ocena']
    
    # Połącz dane
    item_data = pd.merge(item_popularity, item_ratings, on='item_id')
    
    # Dodaj informacje o sukienkach
    if add_info:
        item_data = pd.merge(
            item_data, 
            sukienki_df[['item_id', 'kolor', 'fason', 'material', 'okazja']], 
            on='item_id'
        )
    
    # Pokaż najpopularniejsze sukienki
    top_items = item_data.head(n_items)
    
    # Wykres
    ax = sns.barplot(x='item_id', y='liczba_interakcji', data=top_items, palette='viridis')
    
    # Dodaj średnią ocenę jako tekst
    if add_info:
        for i, (_, row) in enumerate(top_items.iterrows()):
            plt.text(
                i, row['liczba_interakcji'] + 1, 
                f"{row['srednia_ocena']:.1f}★",
                ha='center', va='bottom', fontweight='bold'
            )
            
            # Dodaj informacje o sukience
            plt.text(
                i, 2, 
                f"{row['kolor']}\n{row['fason']}\n{row['material']}",
                ha='center', va='bottom', fontsize=8, rotation=90
            )
    
    plt.title(f'Top {n_items} najpopularniejszych sukienek', fontsize=16)
    plt.xlabel('ID sukienki', fontsize=12)
    plt.ylabel('Liczba interakcji', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_correlation_matrix(sukienki_df: pd.DataFrame, numerical_only: bool = True, 
                           save_path: Optional[str] = None):
    """
    Wizualizuje macierz korelacji dla cech sukienek.
    
    Args:
        sukienki_df: DataFrame z danymi sukienek
        numerical_only: Czy używać tylko cech numerycznych
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(10, 8))
    
    # Wybierz cechy
    if numerical_only:
        # Tylko cechy numeryczne
        features = sukienki_df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Wszystkie cechy (po enkodowaniu)
        features = sukienki_df.columns.tolist()
        # Pomiń ID
        features = [col for col in features if col != 'item_id']
    
    # Oblicz korelację
    corr_matrix = sukienki_df[features].corr()
    
    # Wyświetl macierz korelacji
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Macierz korelacji cech sukienek', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_feature_importance(sukienki_df: pd.DataFrame, interakcje_df: pd.DataFrame, 
                           top_n: int = 10, save_path: Optional[str] = None):
    """
    Wizualizuje istotność cech sukienek.
    
    Args:
        sukienki_df: DataFrame z danymi sukienek
        interakcje_df: DataFrame z interakcjami
        top_n: Liczba najważniejszych cech do pokazania
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 6))
    
    # Połącz dane sukienek z interakcjami
    merged_data = pd.merge(
        interakcje_df[['item_id', 'ocena']], 
        sukienki_df, 
        on='item_id'
    )
    
    # Zbadaj korelację cech z oceną
    categorical_cols = ['kolor', 'fason', 'material', 'okazja', 'rozmiar']
    feature_importance = {}
    
    # Oblicz średnią ocenę dla każdej wartości cechy kategorycznej
    for col in categorical_cols:
        grouped = merged_data.groupby(col)['ocena'].mean().reset_index()
        for _, row in grouped.iterrows():
            feature_importance[f"{col}_{row[col]}"] = row['ocena']
    
    # Dodaj cechy numeryczne
    numerical_cols = ['cena', 'popularnosc']
    for col in numerical_cols:
        corr = merged_data['ocena'].corr(merged_data[col])
        feature_importance[col] = abs(corr)  # Używamy wartości bezwzględnej jako miary znaczenia
    
    # Posortuj według znaczenia
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Przygotuj dane do wykresu
    feature_names = [item[0] for item in top_features]
    importance_values = [item[1] for item in top_features]
    
    # Wykres
    sns.barplot(x=importance_values, y=feature_names)
    plt.title(f'Top {top_n} najważniejszych cech sukienek', fontsize=16)
    plt.xlabel('Znaczenie (korelacja z oceną)', fontsize=12)
    plt.ylabel('Cecha', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_embeddings_visualization(model, sukienki_df: pd.DataFrame, 
                                 method: str = 'tsne', save_path: Optional[str] = None):
    """
    Wizualizuje embeddingów sukienek z modelu w przestrzeni 2D.
    
    Args:
        model: Wytrenowany model
        sukienki_df: DataFrame z danymi sukienek
        method: Metoda redukcji wymiarowości ('tsne' lub 'pca')
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    # Sprawdź, czy model ma embeddingi
    if not hasattr(model, 'embeddings') or 'item_id' not in model.embeddings:
        print("Model nie posiada dostępnych embeddingów sukienek")
        return
    
    # Ustaw model w trybie ewaluacji
    model.eval()
    
    # Pobierz macierz embeddingów sukienek
    with torch.no_grad():
        item_embeddings = model.embeddings['item_id'].weight.cpu().numpy()
    
    # Redukcja wymiarowości do 2D
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
    
    # Zastosuj redukcję wymiarowości
    embeddings_2d = reducer.fit_transform(item_embeddings)
    
    # Przygotuj dane do wizualizacji
    viz_df = pd.DataFrame({
        'item_id': range(len(item_embeddings)),
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    
    # Dodaj cechy sukienek, jeśli dostępne
    if not sukienki_df.empty:
        viz_df = pd.merge(viz_df, sukienki_df[['item_id', 'kolor', 'fason']], on='item_id', how='left')
    
    # Stwórz wykres
    plt.figure(figsize=(14, 10))
    
    # Koloruj punkty według kategorii, jeśli dostępne
    if 'kolor' in viz_df.columns:
        sns.scatterplot(x='x', y='y', hue='kolor', style='fason', data=viz_df, s=100, alpha=0.7)
    else:
        sns.scatterplot(x='x', y='y', data=viz_df, s=100, alpha=0.7)
    
    plt.title(f'Wizualizacja embeddingów sukienek ({method.upper()})', fontsize=16)
    plt.xlabel('Wymiar 1', fontsize=12)
    plt.ylabel('Wymiar 2', fontsize=12)
    
    # Dodaj etykiety dla kilku losowych punktów
    if len(viz_df) > 0:
        n_labels = min(20, len(viz_df))
        sample_indices = np.random.choice(len(viz_df), size=n_labels, replace=False)
        
        for idx in sample_indices:
            row = viz_df.iloc[idx]
            plt.text(row['x'], row['y'], str(int(row['item_id'])), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_user_behavior_over_time(interakcje_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Wizualizuje zachowanie użytkowników w czasie.
    
    Args:
        interakcje_df: DataFrame z interakcjami
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(14, 8))
    
    # Upewnij się, że timestamp jest datą
    if not pd.api.types.is_datetime64_any_dtype(interakcje_df['timestamp']):
        interakcje_df['timestamp'] = pd.to_datetime(interakcje_df['timestamp'])
    
    # Utwórz kolumnę z datą (bez godziny)
    interakcje_df['date'] = interakcje_df['timestamp'].dt.date
    
    # Agreguj dane dziennie
    daily_interactions = interakcje_df.groupby('date').size().reset_index(name='liczba_interakcji')
    daily_ratings = interakcje_df.groupby('date')['ocena'].mean().reset_index(name='srednia_ocena')
    
    # Przygotuj podwykresy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Wykres 1: Liczba interakcji
    ax1.plot(daily_interactions['date'], daily_interactions['liczba_interakcji'], marker='o', 
            linestyle='-', linewidth=2, markersize=8, color='dodgerblue')
    ax1.set_title('Liczba interakcji w czasie', fontsize=14)
    ax1.set_ylabel('Liczba interakcji', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Średnia ocena
    ax2.plot(daily_ratings['date'], daily_ratings['srednia_ocena'], marker='s', 
            linestyle='-', linewidth=2, markersize=8, color='coral')
    ax2.set_title('Średnia ocena w czasie', fontsize=14)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_ylabel('Średnia ocena', fontsize=12)
    ax2.set_ylim(0, 5.5)  # Zakres ocen 0-5
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def plot_recommendation_comparison(recommendations_df: pd.DataFrame, 
                                  user_history_df: pd.DataFrame,
                                  save_path: Optional[str] = None):
    """
    Porównuje rekomendacje z historycznymi preferencjami użytkownika.
    
    Args:
        recommendations_df: DataFrame z rekomendacjami
        user_history_df: DataFrame z historią interakcji użytkownika
        save_path: Ścieżka do zapisania wykresu (opcjonalna)
    """
    plt.figure(figsize=(16, 9))
    
    # Przygotuj dane do porównania
    # Średnia ocena użytkownika dla każdej cechy
    user_color_ratings = user_history_df.groupby('kolor')['ocena'].mean().to_dict()
    user_style_ratings = user_history_df.groupby('fason')['ocena'].mean().to_dict()
    user_material_ratings = user_history_df.groupby('material')['ocena'].mean().to_dict()
    user_occasion_ratings = user_history_df.groupby('okazja')['ocena'].mean().to_dict()
    
    # Cechy rekomendacji
    rec_colors = recommendations_df['kolor'].value_counts().to_dict()
    rec_styles = recommendations_df['fason'].value_counts().to_dict()
    rec_materials = recommendations_df['material'].value_counts().to_dict()
    rec_occasions = recommendations_df['okazja'].value_counts().to_dict()
    
    # Normalizacja wartości
    def normalize_dict(d):
        total = sum(d.values())
        return {k: v/total for k, v in d.items()} if total > 0 else d
    
    rec_colors = normalize_dict(rec_colors)
    rec_styles = normalize_dict(rec_styles)
    rec_materials = normalize_dict(rec_materials)
    rec_occasions = normalize_dict(rec_occasions)
    
    # Przygotuj podwykresy
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Funkcja pomocnicza do tworzenia wykresów
    def plot_feature_comparison(ax, user_ratings, rec_distribution, title):
        # Przygotuj dane
        features = sorted(set(list(user_ratings.keys()) + list(rec_distribution.keys())))
        x = np.arange(len(features))
        width = 0.35
        
        # Przygotuj wartości
        user_values = [user_ratings.get(feature, 0) for feature in features]
        rec_values = [rec_distribution.get(feature, 0) for feature in features]
        
        # Stwórz wykres
        rects1 = ax.bar(x - width/2, user_values, width, label='Preferencje użytkownika', color='royalblue')
        rects2 = ax.bar(x + width/2, rec_values, width, label='Rekomendacje', color='tomato')
        
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Wykresy dla różnych cech
    plot_feature_comparison(axs[0, 0], user_color_ratings, rec_colors, 'Kolory')
    plot_feature_comparison(axs[0, 1], user_style_ratings, rec_styles, 'Fasony')
    plot_feature_comparison(axs[1, 0], user_material_ratings, rec_materials, 'Materiały')
    plot_feature_comparison(axs[1, 1], user_occasion_ratings, rec_occasions, 'Okazje')
    
    # Dopasuj układ
    plt.suptitle('Porównanie preferencji użytkownika z rekomendacjami', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Wykres zapisany w: {save_path}")
    
    plt.show()


def create_visual_report(sukienki_df: pd.DataFrame, uzytkownicy_df: pd.DataFrame, 
                         interakcje_df: pd.DataFrame, model=None, 
                         recommendations_df=None, user_id=None, 
                         output_dir: str = 'plots'):
    """
    Tworzy kompleksowy raport wizualny dla systemu rekomendacji.
    
    Args:
        sukienki_df: DataFrame z danymi sukienek
        uzytkownicy_df: DataFrame z danymi użytkowników
        interakcje_df: DataFrame z interakcjami
        model: Wytrenowany model (opcjonalny)
        recommendations_df: DataFrame z rekomendacjami (opcjonalny)
        user_id: ID użytkownika do analiz konkretnego przypadku (opcjonalny)
        output_dir: Katalog na wykresy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Podstawowe statystyki
    plot_rating_distribution(interakcje_df, save_path=f"{output_dir}/rating_distribution.png")
    plot_user_activity(interakcje_df, save_path=f"{output_dir}/user_activity.png")
    plot_item_popularity(interakcje_df, sukienki_df, save_path=f"{output_dir}/item_popularity.png")
    plot_user_behavior_over_time(interakcje_df, save_path=f"{output_dir}/user_behavior_time.png")
    
    # Analizy cech
    if len(sukienki_df.select_dtypes(include=np.number).columns) > 1:
        plot_correlation_matrix(sukienki_df, save_path=f"{output_dir}/correlation_matrix.png")
    
    plot_feature_importance(sukienki_df, interakcje_df, save_path=f"{output_dir}/feature_importance.png")
    
    # Wizualizacje modelu
    if model is not None:
        try:
            plot_embeddings_visualization(model, sukienki_df, save_path=f"{output_dir}/embeddings_tsne.png")
        except Exception as e:
            print(f"Nie udało się wizualizować embeddingów: {e}")
    
    # Analiza rekomendacji
    if recommendations_df is not None and user_id is not None:
        # Pobierz historię użytkownika
        user_history = interakcje_df[interakcje_df['user_id'] == user_id]
        
        if not user_history.empty:
            # Połącz historię z danymi sukienek
            user_history_df = pd.merge(user_history, sukienki_df, on='item_id')
            
            # Porównaj rekomendacje z historią
            plot_recommendation_comparison(
                recommendations_df, 
                user_history_df, 
                save_path=f"{output_dir}/user_{user_id}_recommendations_comparison.png"
            )
    
    print(f"Raport wizualny został wygenerowany w katalogu: {output_dir}")


if __name__ == "__main__":
    # Przykład użycia - można uruchomić ten plik bezpośrednio do testowania wizualizacji
    import sys
    import os
    
    # Dodaj folder nadrzędny do ścieżki, aby móc importować z innych modułów
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Moduł visualization.py - funkcje wizualizacji dla systemu rekomendacji H&M")
    print("Aby użyć wizualizacji, zaimportuj odpowiednie funkcje w swoim kodzie.")
    print("Przykład: from utils.visualization import plot_rating_distribution") 