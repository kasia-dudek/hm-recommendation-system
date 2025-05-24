#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prosty System Rekomendacji H&M
Minimalistyczna implementacja systemu rekomendacji odzieży H&M
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from typing import Dict, List, Tuple
import json

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HMDataset(Dataset):
    """Dataset dla danych H&M"""
    
    def __init__(self, transactions_df):
        self.transactions = transactions_df
        
    def __len__(self):
        return len(self.transactions)
    
    def __getitem__(self, idx):
        row = self.transactions.iloc[idx]
        
        return {
            'user_id': torch.tensor(row['user_encoded'], dtype=torch.long),
            'item_id': torch.tensor(row['article_encoded'], dtype=torch.long),
            'price': torch.tensor(row['price'], dtype=torch.float),
            'rating': torch.tensor(1.0, dtype=torch.float)  # Implicit feedback - wszystkie interakcje = 1
        }

class SimpleHMModel(nn.Module):
    """Prosty model rekomendacji H&M"""
    
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=16):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),  # +1 dla ceny
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch):
        user_emb = self.user_embedding(batch['user_id'])
        item_emb = self.item_embedding(batch['item_id'])
        price = batch['price'].unsqueeze(1)
        
        x = torch.cat([user_emb, item_emb, price], dim=1)
        return self.fc(x).squeeze()

def load_and_prepare_data(data_dir, sample_size=0.02):
    """Ładuje i przygotowuje dane H&M"""
    
    logger.info(f"Ładowanie danych z {data_dir}")
    
    # Ładowanie danych
    transactions = pd.read_csv(os.path.join(data_dir, 'transactions_train.csv'))
    articles = pd.read_csv(os.path.join(data_dir, 'articles.csv'))
    
    # Próbkowanie danych
    if sample_size < 1.0:
        transactions = transactions.sample(frac=sample_size, random_state=42)
        logger.info(f"Użyto {sample_size*100}% danych: {len(transactions)} transakcji")
    
    # Przygotowanie cen (wypełnienie brakujących wartości)
    transactions['price'] = transactions['price'].fillna(transactions['price'].median())
    
    # Enkodowanie użytkowników i artykułów
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    transactions['user_encoded'] = user_encoder.fit_transform(transactions['customer_id'])
    transactions['article_encoded'] = item_encoder.fit_transform(transactions['article_id'])
    
    # Filtrowanie artykułów do tych, które występują w transakcjach
    articles = articles[articles['article_id'].isin(transactions['article_id'])]
    
    logger.info(f"Przygotowano dane: {len(transactions)} transakcji, {len(articles)} artykułów, {len(user_encoder.classes_)} użytkowników")
    
    return transactions, articles, user_encoder, item_encoder

def train_model(model, train_loader, val_loader, epochs=3, lr=0.001):
    """Trenuje model z early stopping"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 2  # Early stopping po 2 epokach bez poprawy
    patience_counter = 0
    
    for epoch in range(epochs):
        # Trening
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch['rating'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        # Walidacja
        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                predictions = model(batch)
                loss = criterion(predictions, batch['rating'])
                val_loss += loss.item()
                val_batch_count += 1
        
        train_loss /= batch_count
        val_loss /= val_batch_count
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping i zapisywanie najlepszego modelu
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_hm_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping po {epoch+1} epokach")
                break
    
    return model

def generate_recommendations(model, user_id, transactions_df, articles_df, user_encoder, item_encoder, top_k=10):
    """Generuje rekomendacje dla użytkownika - zoptymalizowana wersja"""
    
    model.eval()
    
    # Sprawdź czy użytkownik istnieje
    if user_id not in user_encoder.classes_:
        logger.warning(f"Użytkownik {user_id} nie istnieje w danych treningowych")
        return None
    
    user_encoded = user_encoder.transform([user_id])[0]
    
    # Użyj tylko artykułów z danych treningowych (znacznie szybsze)
    unique_articles = transactions_df['article_id'].unique()
    logger.info(f"Generowanie rekomendacji dla {len(unique_articles)} artykułów z danych treningowych")
    
    # Oblicz średnią cenę dla każdego artykułu
    avg_prices = transactions_df.groupby('article_id')['price'].mean().to_dict()
    
    # Przygotuj batch dla wszystkich artykułów naraz
    valid_articles = []
    valid_encodings = []
    valid_prices = []
    
    for article_id in unique_articles:
        try:
            article_encoded = item_encoder.transform([article_id])[0]
            avg_price = avg_prices[article_id]
            
            valid_articles.append(article_id)
            valid_encodings.append(article_encoded)
            valid_prices.append(avg_price)
        except ValueError:
            # Artykuł nie był w danych treningowych
            continue
    
    if not valid_articles:
        logger.warning("Brak prawidłowych artykułów do rekomendacji")
        return None
    
    # Batch prediction dla wszystkich artykułów naraz
    with torch.no_grad():
        batch = {
            'user_id': torch.tensor([user_encoded] * len(valid_articles), dtype=torch.long),
            'item_id': torch.tensor(valid_encodings, dtype=torch.long),
            'price': torch.tensor(valid_prices, dtype=torch.float)
        }
        
        scores = model(batch).cpu().numpy()
    
    # Tworzenie listy rekomendacji
    recommendations = []
    for i, article_id in enumerate(valid_articles):
        # Pobierz informacje o produkcie z articles_df jeśli dostępne
        article_info = articles_df[articles_df['article_id'] == article_id]
        if not article_info.empty:
            product_name = article_info.iloc[0].get('prod_name', 'Unknown')
            product_type = article_info.iloc[0].get('product_type_name', 'Unknown')
        else:
            product_name = f"Product {article_id}"
            product_type = "Unknown"
        
        recommendations.append({
            'article_id': article_id,
            'product_name': product_name,
            'product_type': product_type,
            'avg_price': valid_prices[i],
            'score': scores[i]
        })
    
    # Sortuj po wyniku
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return recommendations[:top_k]

def main():
    parser = argparse.ArgumentParser(description='Prosty System Rekomendacji H&M')
    parser.add_argument('--action', choices=['train', 'recommend'], default='train')
    parser.add_argument('--data_dir', default='dane_hm')
    parser.add_argument('--sample_size', type=float, default=0.02)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--user_id', type=str, help='ID użytkownika do rekomendacji')
    parser.add_argument('--top_k', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.action == 'train':
        # Ładowanie i przygotowanie danych
        transactions, articles, user_encoder, item_encoder = load_and_prepare_data(
            args.data_dir, args.sample_size
        )
        
        # Podział na train/val
        train_size = int(0.8 * len(transactions))
        train_transactions = transactions[:train_size]
        val_transactions = transactions[train_size:]
        
        # Tworzenie DataLoaderów
        train_dataset = HMDataset(train_transactions)
        val_dataset = HMDataset(val_transactions)
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2)
        
        # Tworzenie modelu
        num_users = len(user_encoder.classes_)
        num_items = len(item_encoder.classes_)
        
        model = SimpleHMModel(num_users, num_items)
        logger.info(f"Model utworzony: {num_users} użytkowników, {num_items} artykułów")
        
        # Trening
        model = train_model(model, train_loader, val_loader, args.epochs)
        
        # Zapisz encodery i dane
        with open('hm_encoders.json', 'w') as f:
            json.dump({
                'user_classes': user_encoder.classes_.tolist(),
                'item_classes': item_encoder.classes_.tolist(),
                'num_users': num_users,
                'num_items': num_items
            }, f)
        
        # Zapisz próbkę transakcji dla rekomendacji
        transactions.to_csv('transactions_sample.csv', index=False)
        
        logger.info("Trening zakończony. Model i encodery zapisane.")
        
    elif args.action == 'recommend':
        if not args.user_id:
            logger.error("Podaj --user_id dla generowania rekomendacji")
            return
        
        # Ładuj encodery
        with open('hm_encoders.json', 'r') as f:
            encoder_data = json.load(f)
        
        user_encoder = LabelEncoder()
        user_encoder.classes_ = np.array(encoder_data['user_classes'])
        
        item_encoder = LabelEncoder()
        item_encoder.classes_ = np.array(encoder_data['item_classes'])
        
        # Ładuj model z prawidłowymi rozmiarami z zapisanych danych
        model = SimpleHMModel(encoder_data['num_users'], encoder_data['num_items'])
        model.load_state_dict(torch.load('best_hm_model.pt'))
        
        # Ładuj dane
        transactions = pd.read_csv('transactions_sample.csv')
        articles = pd.read_csv(os.path.join(args.data_dir, 'articles.csv'))
        articles = articles[articles['article_id'].isin(transactions['article_id'])]
        
        # Generuj rekomendacje
        recommendations = generate_recommendations(
            model, args.user_id, transactions, articles, user_encoder, item_encoder, args.top_k
        )
        
        if recommendations:
            logger.info(f"Top {args.top_k} rekomendacji dla użytkownika {args.user_id}:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec['product_name']} ({rec['product_type']}) - ID: {rec['article_id']}, Średnia cena: {rec['avg_price']:.2f}, Wynik: {rec['score']:.3f}")
        else:
            logger.warning("Nie udało się wygenerować rekomendacji")

if __name__ == "__main__":
    main() 