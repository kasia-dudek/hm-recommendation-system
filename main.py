#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
System Rekomendacji H&M wykorzystujący PyTorch

Ten program implementuje system rekomendacji H&M, który wykorzystuje mechanizmy
uczenia maszynowego do personalizowania rekomendacji na podstawie preferencji użytkowników,
cech produktów oraz historii interakcji.

Autor: [Twoje imię i nazwisko]
Data: [Data utworzenia]
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import json
from dotenv import load_dotenv

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('recommender.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Importy modułów do pracy z danymi H&M
try:
    from data.data_loader_hm import przygotuj_dane_hm, zaladuj_dane_hm
    from models.model_hm import create_model_hm
    from models.training import train_model, load_model, plot_training_history, save_model_info
    from models.evaluation import (
        evaluate_model, generate_top_k_recommendations, evaluate_recommendations_for_users,
        plot_recommendations, plot_recommendation_metrics
    )
    HM_AVAILABLE = True
except ImportError:
    logger.error("Moduły do pracy z danymi H&M nie są dostępne.")
    HM_AVAILABLE = False
    sys.exit(1)

# Importy Azure (opcjonalne)
try:
    from azure.azure_integration import (
        AzureStorageHandler, AzureMLHandler, upload_data_to_azure, load_data_from_azure
    )
    AZURE_AVAILABLE = True
except ImportError:
    logger.warning("Moduły Azure nie są dostępne. Funkcje chmurowe będą wyłączone.")
    AZURE_AVAILABLE = False

def parse_arguments():
    """
    Parsuje argumenty wiersza poleceń.
    
    Returns:
        Sparsowane argumenty
    """
    parser = argparse.ArgumentParser(description='System Rekomendacji H&M')
    
    parser.add_argument('--action', type=str, default='train', 
                        choices=['train', 'evaluate', 'recommend', 'upload', 'download'],
                        help='Akcja do wykonania')
    
    parser.add_argument('--data_dir', type=str, default='dane_hm',
                        help='Katalog z danymi H&M')
    
    parser.add_argument('--output_dir', type=str, default='wyniki',
                        help='Katalog na wyniki')
    
    parser.add_argument('--model_dir', type=str, default='model_info',
                        help='Katalog na model')
    
    parser.add_argument('--user_id', type=str, default=None,
                        help='ID użytkownika do generowania rekomendacji')
    
    parser.add_argument('--top_k', type=int, default=10,
                        help='Liczba rekomendacji do wygenerowania')
    
    parser.add_argument('--epochs', type=int, default=30,
                        help='Liczba epok treningu')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Rozmiar batcha')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Współczynnik uczenia')
    
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Wymiar embeddingów')
    
    parser.add_argument('--hidden_dims', type=str, default='128,64',
                        help='Wymiary warstw ukrytych (oddzielone przecinkami)')
    
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Współczynnik dropout')
    
    parser.add_argument('--use_azure', action='store_true',
                        help='Czy używać integracji z Azure')
    
    parser.add_argument('--azure_model_deploy', action='store_true',
                        help='Czy wdrożyć model w Azure ML')
    
    parser.add_argument('--sample_size', type=float, default=0.01,
                        help='Wielkość próbki danych H&M (procent z pełnego zbioru)')
    
    return parser.parse_args()

def train_and_save_model(args):
    """
    Trenuje i zapisuje model H&M.
    
    Args:
        args: Argumenty wiersza poleceń
        
    Returns:
        Wytrenowany model i historia treningu
    """
    logger.info(f"Przygotowanie danych H&M z katalogu: {args.data_dir}")
    
    # Przygotuj dane H&M
    train_loader, val_loader, test_loader, liczba_kategorii, user_encoders, item_encoders = przygotuj_dane_hm(
        katalog_danych=args.data_dir,
        batch_size=args.batch_size,
        sample_size=args.sample_size
    )
    
    if train_loader is None:
        logger.error("Nie udało się przygotować danych H&M. Upewnij się, że dane istnieją w podanym katalogu.")
        return None, None, None, None, None
    
    # Przygotuj model H&M
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    model_params = {
        'dim_embeddingu': args.embedding_dim,
        'hidden_dims': hidden_dims,
        'dropout': args.dropout
    }
    
    logger.info(f"Tworzenie modelu H&M z parametrami: {model_params}")
    model = create_model_hm(liczba_kategorii, **model_params)
    encoders = (user_encoders, item_encoders)
    
    # Trenuj model
    logger.info(f"Rozpoczęcie treningu na {args.epochs} epokach")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=5,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
    )
    
    # Ewaluuj na zbiorze testowym
    logger.info("Ewaluacja modelu na zbiorze testowym")
    test_loss, test_metrics = evaluate_model(model, test_loader)
    logger.info(f"Finalne metryki testowe: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}")
    
    # Zapisz wizualizację historii treningu
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, 'plots', 'training_history.png')
    )
    
    # Zapisz informacje o modelu
    logger.info(f"Zapisywanie modelu do katalogu: {args.model_dir}")
    save_model_info(
        model,
        history,
        liczba_kategorii,
        model_params,
        test_metrics,
        save_dir=args.model_dir
    )
    
    return model, train_loader, val_loader, test_loader, encoders

def evaluate_and_visualize(args, model=None, encoders=None):
    """
    Ewaluuje model H&M i generuje wizualizacje.
    
    Args:
        args: Argumenty wiersza poleceń
        model: Opcjonalny model (jeśli None, zostanie załadowany)
        encoders: Opcjonalne encodery (jeśli None, zostaną załadowane)
    """
    logger.info("Rozpoczęcie ewaluacji i wizualizacji")
    
    # Załaduj model i encodery jeśli nie zostały podane
    if model is None or encoders is None:
        try:
            model, encoders = load_model(args.model_dir)
            logger.info("Model i encodery załadowane z dysku")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu: {e}")
            return
    
    # Załaduj dane H&M
    sukienki_df, uzytkownicy_df, interakcje_df = zaladuj_dane_hm(katalog_danych=args.data_dir)
    
    if sukienki_df is None:
        logger.error("Nie udało się załadować danych H&M")
        return
    
    # Wybierz próbkę użytkowników do ewaluacji
    sample_users = np.random.choice(
        uzytkownicy_df['customer_id'].unique(), 
        min(20, len(uzytkownicy_df)), 
        replace=False
    )
    
    logger.info(f"Ewaluacja rekomendacji dla {len(sample_users)} użytkowników")
    
    # Generuj rekomendacje dla próbki użytkowników
    all_recommendations = []
    for user_id in sample_users:
        try:
            recommendations = generate_top_k_recommendations(
                model, user_id, interakcje_df, sukienki_df, 
                encoders[0], encoders[1], top_k=args.top_k
            )
            all_recommendations.extend(recommendations)
        except Exception as e:
            logger.warning(f"Błąd generowania rekomendacji dla użytkownika {user_id}: {e}")
    
    if all_recommendations:
        recommendations_df = pd.DataFrame(all_recommendations)
        
        # Ewaluuj jakość rekomendacji
        metrics = evaluate_recommendations_for_users(
            recommendations_df, interakcje_df, k_values=[5, 10, 20]
        )
        
        logger.info("Metryki ewaluacji:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Generuj wizualizacje
        os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
        
        plot_recommendations(
            recommendations_df,
            save_path=os.path.join(args.output_dir, 'plots', 'recommendations_analysis.png')
        )
        
        plot_recommendation_metrics(
            metrics,
            save_path=os.path.join(args.output_dir, 'plots', 'recommendation_metrics.png')
        )
        
        logger.info(f"Wizualizacje zapisane w katalogu: {args.output_dir}/plots")

def recommend_hm_for_user(args, model=None, encoders=None):
    """
    Generuje rekomendacje H&M dla konkretnego użytkownika.
    
    Args:
        args: Argumenty wiersza poleceń
        model: Opcjonalny model (jeśli None, zostanie załadowany)
        encoders: Opcjonalne encodery (jeśli None, zostaną załadowane)
    """
    if args.user_id is None:
        logger.error("Nie podano ID użytkownika. Użyj parametru --user_id")
        return
    
    logger.info(f"Generowanie rekomendacji H&M dla użytkownika: {args.user_id}")
    
    # Załaduj model i encodery jeśli nie zostały podane
    if model is None or encoders is None:
        try:
            model, encoders = load_model(args.model_dir)
            logger.info("Model i encodery załadowane z dysku")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu: {e}")
            return
    
    # Załaduj dane H&M
    sukienki_df, uzytkownicy_df, interakcje_df = zaladuj_dane_hm(katalog_danych=args.data_dir)
    
    if sukienki_df is None:
        logger.error("Nie udało się załadować danych H&M")
        return
    
    # Sprawdź czy użytkownik istnieje
    if args.user_id not in uzytkownicy_df['customer_id'].values:
        logger.error(f"Użytkownik {args.user_id} nie istnieje w danych")
        return
    
    try:
        # Wygeneruj rekomendacje
        recommendations = generate_top_k_recommendations(
            model, args.user_id, interakcje_df, sukienki_df, 
            encoders[0], encoders[1], top_k=args.top_k
        )
        
        if recommendations:
            logger.info(f"Wygenerowano {len(recommendations)} rekomendacji dla użytkownika {args.user_id}")
            
            # Wyświetl rekomendacje
            print(f"\n🎯 TOP {args.top_k} REKOMENDACJI H&M dla użytkownika {args.user_id}:")
            print("=" * 80)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2d}. Produkt: {rec['article_id']}")
                print(f"    Nazwa: {rec.get('product_name', 'N/A')}")
                print(f"    Kategoria: {rec.get('product_type_name', 'N/A')}")
                print(f"    Ocena: {rec['predicted_rating']:.3f}")
                print()
            
            # Zapisz rekomendacje do pliku
            recommendations_df = pd.DataFrame(recommendations)
            output_path = os.path.join(args.output_dir, f'recommendations_user_{args.user_id}.csv')
            recommendations_df.to_csv(output_path, index=False)
            logger.info(f"Rekomendacje zapisane do: {output_path}")
            
        else:
            logger.warning(f"Nie udało się wygenerować rekomendacji dla użytkownika {args.user_id}")
            
    except Exception as e:
        logger.error(f"Błąd podczas generowania rekomendacji: {e}")

def upload_to_azure(args):
    """
    Przesyła dane H&M do Azure Blob Storage.
    
    Args:
        args: Argumenty wiersza poleceń
    """
    if not AZURE_AVAILABLE:
        logger.error("Moduły Azure nie są dostępne")
        return
    
    logger.info("Przesyłanie danych H&M do Azure")
    
    # Pobierz connection string z zmiennych środowiskowych
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    if not connection_string:
        logger.error("Brak AZURE_STORAGE_CONNECTION_STRING w zmiennych środowiskowych")
        return
    
    try:
        # Utwórz obsługę Azure Storage
        storage_handler = AzureStorageHandler(connection_string=connection_string)
        
        # Załaduj dane H&M
        sukienki_df, uzytkownicy_df, interakcje_df = zaladuj_dane_hm(katalog_danych=args.data_dir)
        
        if sukienki_df is not None:
            # Prześlij dane do Azure
            urls = upload_data_to_azure(
                sukienki_df, uzytkownicy_df, interakcje_df, storage_handler
            )
            
            logger.info("Dane H&M zostały przesłane do Azure Blob Storage:")
            for key, url in urls.items():
                logger.info(f"  {key}: {url}")
        else:
            logger.error("Nie udało się załadować danych H&M")
            
    except Exception as e:
        logger.error(f"Błąd podczas przesyłania do Azure: {e}")

def download_from_azure(args):
    """
    Pobiera dane H&M z Azure Blob Storage.
    
    Args:
        args: Argumenty wiersza poleceń
    """
    if not AZURE_AVAILABLE:
        logger.error("Moduły Azure nie są dostępne")
        return
    
    logger.info("Pobieranie danych H&M z Azure")
    
    # Pobierz connection string z zmiennych środowiskowych
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    if not connection_string:
        logger.error("Brak AZURE_STORAGE_CONNECTION_STRING w zmiennych środowiskowych")
        return
    
    try:
        # Utwórz obsługę Azure Storage
        storage_handler = AzureStorageHandler(connection_string=connection_string)
        
        # Pobierz dane z Azure
        sukienki_df, uzytkownicy_df, interakcje_df = load_data_from_azure(storage_handler)
        
        if sukienki_df is not None:
            logger.info("Pobrano dane H&M z Azure Blob Storage:")
            logger.info(f"  Produkty: {len(sukienki_df)} rekordów")
            logger.info(f"  Użytkownicy: {len(uzytkownicy_df)} rekordów")
            logger.info(f"  Interakcje: {len(interakcje_df)} rekordów")
            
            # Zapisz dane lokalnie
            os.makedirs(args.data_dir, exist_ok=True)
            sukienki_df.to_csv(os.path.join(args.data_dir, 'articles.csv'), index=False)
            uzytkownicy_df.to_csv(os.path.join(args.data_dir, 'customers.csv'), index=False)
            interakcje_df.to_csv(os.path.join(args.data_dir, 'transactions_train.csv'), index=False)
            
            logger.info(f"Dane zapisane lokalnie w katalogu: {args.data_dir}")
        else:
            logger.error("Nie udało się pobrać danych z Azure")
            
    except Exception as e:
        logger.error(f"Błąd podczas pobierania z Azure: {e}")

def main():
    """
    Główna funkcja programu.
    """
    args = parse_arguments()
    
    if not HM_AVAILABLE:
        logger.error("System wymaga modułów do obsługi danych H&M")
        return
    
    # Utwórz katalogi wynikowe, jeśli nie istnieją
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Wykonaj żądaną akcję
    if args.action == 'train':
        # Trenuj model H&M
        model, train_loader, val_loader, test_loader, encoders = train_and_save_model(args)
        
        # Jeśli podano user_id, wygeneruj też rekomendacje
        if args.user_id is not None and model is not None and encoders is not None:
            recommend_hm_for_user(args, model, encoders)
    
    elif args.action == 'evaluate':
        # Ewaluuj model H&M
        evaluate_and_visualize(args)
    
    elif args.action == 'recommend':
        # Generuj rekomendacje H&M
        recommend_hm_for_user(args)
    
    elif args.action == 'upload' and args.use_azure:
        # Prześlij dane H&M do Azure
        upload_to_azure(args)
    
    elif args.action == 'download' and args.use_azure:
        # Pobierz dane H&M z Azure
        download_from_azure(args)
    
    else:
        logger.warning(f"Nieobsługiwana akcja: {args.action}")
        if args.action in ['upload', 'download'] and not args.use_azure:
            logger.warning("Aby używać funkcji Azure, dodaj parametr --use_azure")

if __name__ == "__main__":
    main() 