#!/usr/bin/env python3
"""
Ulepszona Ewaluacja Systemu Rekomendacji H&M
Lepszy podział danych i dodatkowe metryki jakości
"""

import torch
import pandas as pd
import numpy as np
import json
import argparse
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from collections import defaultdict

# Import naszego modelu
from simple_hm_recommender import SimpleHMModel, generate_recommendations

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ImprovedRecommendationEvaluator:
    """Ulepszona klasa do ewaluacji systemu rekomendacji"""
    
    def __init__(self, model_path: str, encoders_path: str, transactions_path: str):
        """Inicjalizuje ewaluator"""
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.transactions_path = transactions_path
        
        # Ładuj model i encodery
        self._load_model_and_encoders()
        
        # Ładuj dane
        self.transactions = pd.read_csv(transactions_path)
        logger.info(f"Załadowano {len(self.transactions)} transakcji")
        
    def _load_model_and_encoders(self):
        """Ładuje model i encodery"""
        # Ładuj encodery
        with open(self.encoders_path, 'r') as f:
            encoder_data = json.load(f)
        
        self.user_encoder = LabelEncoder()
        self.user_encoder.classes_ = np.array(encoder_data['user_classes'])
        
        self.item_encoder = LabelEncoder()
        self.item_encoder.classes_ = np.array(encoder_data['item_classes'])
        
        # Ładuj model
        self.model = SimpleHMModel(encoder_data['num_users'], encoder_data['num_items'])
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
        logger.info(f"Model załadowany: {encoder_data['num_users']} użytkowników, {encoder_data['num_items']} produktów")
    
    def create_improved_test_set(self, min_user_transactions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Tworzy lepszy podział train/test - tylko użytkownicy z wystarczającą liczbą transakcji"""
        
        # Filtruj użytkowników z wystarczającą liczbą transakcji
        user_transaction_counts = self.transactions.groupby('customer_id').size()
        valid_users = user_transaction_counts[user_transaction_counts >= min_user_transactions].index
        
        logger.info(f"Użytkownicy z ≥{min_user_transactions} transakcjami: {len(valid_users)}")
        
        # Filtruj transakcje do valid users
        filtered_transactions = self.transactions[self.transactions['customer_id'].isin(valid_users)]
        
        # Grupuj transakcje po użytkownikach
        user_transactions = filtered_transactions.groupby('customer_id')
        
        train_data = []
        test_data = []
        
        for user_id, user_trans in user_transactions:
            # Sortuj po dacie jeśli dostępna
            if 't_dat' in user_trans.columns:
                user_trans = user_trans.sort_values('t_dat')
            
            # Weź ostatnie 2 transakcje jako test, resztę jako train
            n_transactions = len(user_trans)
            n_test = min(2, max(1, n_transactions // 4))  # 25% ale max 2
            
            test_transactions = user_trans.tail(n_test)
            train_transactions = user_trans.head(n_transactions - n_test)
            
            train_data.append(train_transactions)
            test_data.append(test_transactions)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        logger.info(f"Podział danych: {len(train_df)} transakcji treningowych, {len(test_df)} testowych")
        logger.info(f"Użytkownicy w train: {train_df['customer_id'].nunique()}, w test: {test_df['customer_id'].nunique()}")
        
        return train_df, test_df
    
    def calculate_coverage_and_diversity(self, all_recommendations: List[List[str]]) -> Dict:
        """Oblicza pokrycie katalogu i różnorodność rekomendacji"""
        
        # Wszystkie rekomendowane produkty
        all_recommended_items = set()
        for recs in all_recommendations:
            all_recommended_items.update(recs)
        
        # Pokrycie katalogu
        total_items = len(self.item_encoder.classes_)
        coverage = len(all_recommended_items) / total_items
        
        # Różnorodność (średnia liczba unikalnych produktów na użytkownika)
        diversity_scores = []
        for recs in all_recommendations:
            if len(recs) > 0:
                diversity_scores.append(len(set(recs)) / len(recs))
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        return {
            'catalog_coverage': coverage,
            'avg_diversity': avg_diversity,
            'total_unique_recommendations': len(all_recommended_items)
        }
    
    def calculate_popularity_bias(self, recommendations: List[str], train_data: pd.DataFrame) -> float:
        """Oblicza bias popularności - czy system preferuje popularne produkty"""
        
        # Oblicz popularność produktów w danych treningowych
        item_popularity = train_data['article_id'].value_counts()
        
        # Oblicz średnią popularność rekomendowanych produktów
        rec_popularities = []
        for item in recommendations:
            if item in item_popularity.index:
                rec_popularities.append(item_popularity[item])
            else:
                rec_popularities.append(0)
        
        if not rec_popularities:
            return 0.0
        
        # Normalizuj przez maksymalną popularność
        max_popularity = item_popularity.max()
        normalized_popularity = np.mean(rec_popularities) / max_popularity
        
        return normalized_popularity
    
    def evaluate_user_improved(self, user_id: str, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                              k_values: List[int] = [5, 10, 20]) -> Dict:
        """Ulepszona ewaluacja dla jednego użytkownika"""
        
        # Sprawdź czy użytkownik istnieje w danych treningowych
        if user_id not in self.user_encoder.classes_:
            return None
        
        # Pobierz rzeczywiste preferencje użytkownika z test set
        user_test_items = test_data[test_data['customer_id'] == user_id]['article_id'].tolist()
        
        if len(user_test_items) == 0:
            return None
        
        # Wygeneruj rekomendacje
        articles_df = pd.DataFrame({'article_id': self.transactions['article_id'].unique()})
        
        try:
            recommendations = generate_recommendations(
                self.model, user_id, train_data, articles_df, 
                self.user_encoder, self.item_encoder, top_k=max(k_values)
            )
            
            if not recommendations:
                return None
            
            recommended_items = [rec['article_id'] for rec in recommendations]
            
        except Exception as e:
            logger.warning(f"Błąd generowania rekomendacji dla użytkownika {user_id}: {e}")
            return None
        
        # Oblicz podstawowe metryki
        results = {
            'user_id': user_id, 
            'num_test_items': len(user_test_items),
            'num_train_items': len(train_data[train_data['customer_id'] == user_id])
        }
        
        for k in k_values:
            top_k_recs = recommended_items[:k]
            relevant_recommended = set(top_k_recs).intersection(set(user_test_items))
            
            # Podstawowe metryki
            precision = len(relevant_recommended) / k if k > 0 else 0
            recall = len(relevant_recommended) / len(user_test_items) if len(user_test_items) > 0 else 0
            hit_rate = 1.0 if len(relevant_recommended) > 0 else 0.0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f'precision@{k}'] = precision
            results[f'recall@{k}'] = recall
            results[f'hit_rate@{k}'] = hit_rate
            results[f'f1@{k}'] = f1
        
        # Dodatkowe metryki
        results['popularity_bias'] = self.calculate_popularity_bias(recommended_items, train_data)
        
        return results
    
    def evaluate_system_improved(self, num_users: int = 50, k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Ulepszona ewaluacja całego systemu"""
        
        logger.info(f"Rozpoczęcie ulepszonej ewaluacji systemu na {num_users} użytkownikach")
        
        # Utwórz lepszy podział train/test
        train_data, test_data = self.create_improved_test_set()
        
        # Wybierz użytkowników do ewaluacji
        test_users = test_data['customer_id'].unique()
        available_users = [u for u in test_users if u in self.user_encoder.classes_]
        
        if len(available_users) == 0:
            logger.error("Brak użytkowników dostępnych do ewaluacji")
            return pd.DataFrame()
        
        # Wybierz próbkę użytkowników
        selected_users = np.random.choice(
            available_users, 
            min(num_users, len(available_users)), 
            replace=False
        )
        
        logger.info(f"Ewaluacja {len(selected_users)} użytkowników")
        
        # Ewaluuj każdego użytkownika
        results = []
        all_recommendations = []
        
        for i, user_id in enumerate(selected_users):
            if i % 10 == 0:
                logger.info(f"Postęp: {i}/{len(selected_users)} użytkowników")
            
            user_results = self.evaluate_user_improved(user_id, train_data, test_data, k_values)
            if user_results:
                results.append(user_results)
                
                # Zbierz rekomendacje dla analizy pokrycia
                try:
                    articles_df = pd.DataFrame({'article_id': self.transactions['article_id'].unique()})
                    recommendations = generate_recommendations(
                        self.model, user_id, train_data, articles_df, 
                        self.user_encoder, self.item_encoder, top_k=20
                    )
                    if recommendations:
                        rec_items = [rec['article_id'] for rec in recommendations]
                        all_recommendations.append(rec_items)
                except:
                    pass
        
        if not results:
            logger.error("Brak wyników ewaluacji")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Oblicz metryki pokrycia i różnorodności
        if all_recommendations:
            coverage_metrics = self.calculate_coverage_and_diversity(all_recommendations)
            logger.info(f"Pokrycie katalogu: {coverage_metrics['catalog_coverage']:.4f}")
            logger.info(f"Średnia różnorodność: {coverage_metrics['avg_diversity']:.4f}")
        
        logger.info(f"Ewaluacja zakończona: {len(results_df)} użytkowników")
        
        return results_df
    
    def calculate_aggregate_metrics_improved(self, results_df: pd.DataFrame) -> Dict:
        """Oblicza ulepszone zagregowane metryki"""
        
        if results_df.empty:
            return {}
        
        aggregate_metrics = {}
        
        # Podstawowe metryki
        metric_columns = [col for col in results_df.columns if '@' in col or col == 'popularity_bias']
        
        for metric_col in metric_columns:
            values = results_df[metric_col].dropna()
            if len(values) > 0:
                aggregate_metrics[f'mean_{metric_col}'] = values.mean()
                aggregate_metrics[f'std_{metric_col}'] = values.std()
                aggregate_metrics[f'median_{metric_col}'] = values.median()
                aggregate_metrics[f'min_{metric_col}'] = values.min()
                aggregate_metrics[f'max_{metric_col}'] = values.max()
        
        # Dodatkowe statystyki
        aggregate_metrics['num_evaluated_users'] = len(results_df)
        aggregate_metrics['avg_test_items_per_user'] = results_df['num_test_items'].mean()
        aggregate_metrics['avg_train_items_per_user'] = results_df['num_train_items'].mean()
        
        # Procent użytkowników z przynajmniej jednym trafieniem
        for k in [5, 10, 20]:
            hit_col = f'hit_rate@{k}'
            if hit_col in results_df.columns:
                hit_rate = (results_df[hit_col] > 0).mean()
                aggregate_metrics[f'users_with_hits@{k}'] = hit_rate
        
        return aggregate_metrics
    
    def create_detailed_report(self, results_df: pd.DataFrame, output_dir: str):
        """Tworzy szczegółowy raport z analizą"""
        
        report_path = os.path.join(output_dir, 'detailed_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SZCZEGÓŁOWY RAPORT EWALUACJI SYSTEMU REKOMENDACJI H&M\n")
            f.write("=" * 60 + "\n\n")
            
            # Podstawowe statystyki
            f.write("PODSTAWOWE STATYSTYKI:\n")
            f.write(f"Liczba ewaluowanych użytkowników: {len(results_df)}\n")
            f.write(f"Średnia liczba produktów testowych: {results_df['num_test_items'].mean():.1f}\n")
            f.write(f"Średnia liczba produktów treningowych: {results_df['num_train_items'].mean():.1f}\n\n")
            
            # Analiza metryk
            f.write("ANALIZA METRYK:\n")
            for k in [5, 10, 20]:
                f.write(f"\nDla K={k}:\n")
                
                precision_col = f'precision@{k}'
                recall_col = f'recall@{k}'
                hit_col = f'hit_rate@{k}'
                f1_col = f'f1@{k}'
                
                if precision_col in results_df.columns:
                    f.write(f"  Precision@{k}: {results_df[precision_col].mean():.4f} ± {results_df[precision_col].std():.4f}\n")
                    f.write(f"  Recall@{k}: {results_df[recall_col].mean():.4f} ± {results_df[recall_col].std():.4f}\n")
                    f.write(f"  Hit Rate@{k}: {results_df[hit_col].mean():.4f}\n")
                    f.write(f"  F1@{k}: {results_df[f1_col].mean():.4f} ± {results_df[f1_col].std():.4f}\n")
                    f.write(f"  Użytkownicy z trafieniami: {(results_df[hit_col] > 0).sum()}/{len(results_df)} ({(results_df[hit_col] > 0).mean()*100:.1f}%)\n")
            
            # Analiza bias popularności
            if 'popularity_bias' in results_df.columns:
                f.write(f"\nBIAS POPULARNOŚCI:\n")
                f.write(f"Średni bias popularności: {results_df['popularity_bias'].mean():.4f}\n")
                f.write(f"(0 = tylko niepopularne, 1 = tylko najpopularniejsze)\n")
            
            # Rekomendacje dla najlepszych użytkowników
            f.write(f"\nNAJLEPSI UŻYTKOWNICY (według Hit Rate@10):\n")
            if 'hit_rate@10' in results_df.columns:
                best_users = results_df.nlargest(5, 'hit_rate@10')
                for _, user in best_users.iterrows():
                    f.write(f"User {user['user_id'][:8]}...: Hit Rate@10 = {user['hit_rate@10']:.3f}, "
                           f"Test items = {user['num_test_items']}, Train items = {user['num_train_items']}\n")
        
        logger.info(f"Szczegółowy raport zapisany: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Ulepszona Ewaluacja Systemu Rekomendacji H&M')
    parser.add_argument('--model_path', default='best_hm_model.pt', help='Ścieżka do modelu')
    parser.add_argument('--encoders_path', default='hm_encoders.json', help='Ścieżka do enkoderów')
    parser.add_argument('--transactions_path', default='transactions_sample.csv', help='Ścieżka do transakcji')
    parser.add_argument('--num_users', type=int, default=30, help='Liczba użytkowników do ewaluacji')
    parser.add_argument('--k_values', nargs='+', type=int, default=[5, 10, 20], help='Wartości K do ewaluacji')
    parser.add_argument('--output_dir', default='evaluation_results_improved', help='Katalog wyników')
    
    args = parser.parse_args()
    
    # Sprawdź czy pliki istnieją
    required_files = [args.model_path, args.encoders_path, args.transactions_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Brak pliku: {file_path}")
            return
    
    # Utwórz katalog wyników
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inicjalizuj ewaluator
    evaluator = ImprovedRecommendationEvaluator(
        args.model_path, 
        args.encoders_path, 
        args.transactions_path
    )
    
    # Przeprowadź ewaluację
    results_df = evaluator.evaluate_system_improved(args.num_users, args.k_values)
    
    if results_df.empty:
        logger.error("Ewaluacja nie powiodła się")
        return
    
    # Oblicz zagregowane metryki
    aggregate_metrics = evaluator.calculate_aggregate_metrics_improved(results_df)
    
    # Wyświetl wyniki
    logger.info("\n" + "="*60)
    logger.info("WYNIKI ULEPSZONEJ EWALUACJI SYSTEMU REKOMENDACJI H&M")
    logger.info("="*60)
    
    # Wyświetl główne metryki
    main_metrics = ['precision', 'recall', 'hit_rate', 'f1']
    for metric in main_metrics:
        logger.info(f"\n{metric.upper()}:")
        for k in args.k_values:
            key = f'mean_{metric}@{k}'
            if key in aggregate_metrics:
                logger.info(f"  @{k}: {aggregate_metrics[key]:.4f}")
    
    # Dodatkowe statystyki
    logger.info(f"\nDODATKOWE STATYSTYKI:")
    logger.info(f"Liczba ewaluowanych użytkowników: {aggregate_metrics.get('num_evaluated_users', 0)}")
    logger.info(f"Średnia liczba produktów testowych: {aggregate_metrics.get('avg_test_items_per_user', 0):.1f}")
    logger.info(f"Średnia liczba produktów treningowych: {aggregate_metrics.get('avg_train_items_per_user', 0):.1f}")
    
    for k in args.k_values:
        key = f'users_with_hits@{k}'
        if key in aggregate_metrics:
            logger.info(f"Użytkownicy z trafieniami @{k}: {aggregate_metrics[key]*100:.1f}%")
    
    if 'mean_popularity_bias' in aggregate_metrics:
        logger.info(f"Średni bias popularności: {aggregate_metrics['mean_popularity_bias']:.4f}")
    
    # Zapisz wyniki
    results_path = os.path.join(args.output_dir, 'detailed_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Szczegółowe wyniki zapisane: {results_path}")
    
    metrics_path = os.path.join(args.output_dir, 'aggregate_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    logger.info(f"Zagregowane metryki zapisane: {metrics_path}")
    
    # Utwórz szczegółowy raport
    evaluator.create_detailed_report(results_df, args.output_dir)
    
    logger.info("\n🎉 Ulepszona ewaluacja zakończona pomyślnie!")

if __name__ == "__main__":
    main() 