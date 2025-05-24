#!/usr/bin/env python3
"""
Demo systemu rekomendacji H&M
Pokazuje jak używać prostego systemu rekomendacji odzieży
"""

import subprocess
import pandas as pd
import json
import re

def run_command(cmd):
    """Uruchamia komendę i zwraca wynik"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def get_sample_users(n=5):
    """Pobiera przykładowych użytkowników z danych treningowych"""
    try:
        df = pd.read_csv('transactions_sample.csv')
        sample_users = df['customer_id'].unique()[:n]
        return sample_users
    except FileNotFoundError:
        print("❌ Brak pliku transactions_sample.csv - uruchom najpierw trening!")
        return []

def parse_recommendations(output):
    """Parsuje rekomendacje z outputu"""
    recommendations = []
    
    # Podziel na linie i znajdź te z rekomendacjami
    lines = output.split('\n')
    
    for line in lines:
        # Szukaj linii z numerem rekomendacji (1., 2., 3.)
        if 'INFO]' in line and any(f'{j}.' in line for j in range(1, 11)):
            # Wyciągnij część po INFO]
            if 'INFO]' in line:
                info_part = line.split('INFO]', 1)[1].strip()
                
                # Sprawdź czy zawiera wzorzec "numer. nazwa (typ) - ID:"
                if ' - ID:' in info_part and '(' in info_part and ')' in info_part:
                    try:
                        # Wyciągnij numer
                        num = info_part.split('.', 1)[0].strip()
                        
                        # Wyciągnij nazwę i typ
                        rest = info_part.split('.', 1)[1].strip()
                        name_and_type = rest.split(' - ID:')[0].strip()
                        
                        # Podziel na nazwę i typ
                        if '(' in name_and_type and ')' in name_and_type:
                            name = name_and_type.split('(')[0].strip()
                            type_part = name_and_type.split('(')[1].split(')')[0].strip()
                            
                            # Wyciągnij ID
                            id_part = rest.split(' - ID:')[1].split(',')[0].strip()
                            
                            recommendations.append({
                                'num': num,
                                'name': name,
                                'type': type_part,
                                'id': id_part
                            })
                    except (IndexError, ValueError):
                        continue
    
    return recommendations

def main():
    print("🛍️  Demo Systemu Rekomendacji H&M")
    print("=" * 50)
    
    # Sprawdź czy model istnieje
    try:
        with open('hm_encoders.json', 'r') as f:
            encoder_data = json.load(f)
        print(f"✅ Model załadowany: {encoder_data['num_users']} użytkowników, {encoder_data['num_items']} produktów")
    except FileNotFoundError:
        print("❌ Brak wytrenowanego modelu!")
        print("Uruchom najpierw: python simple_hm_recommender.py --action train --sample_size 0.01 --epochs 5")
        return
    
    # Pobierz przykładowych użytkowników
    sample_users = get_sample_users(3)
    if len(sample_users) == 0:
        return
    
    print(f"\n🎯 Generowanie rekomendacji dla {len(sample_users)} przykładowych użytkowników:")
    print("-" * 50)
    
    for i, user_id in enumerate(sample_users, 1):
        print(f"\n👤 Użytkownik {i}: {user_id[:20]}...")
        
        # Generuj rekomendacje - używaj pełnego ID użytkownika
        cmd = f'python simple_hm_recommender.py --action recommend --user_id "{user_id}" --top_k 3'
        returncode, stdout, stderr = run_command(cmd)
        
        if returncode == 0:
            # Parsuj STDERR zamiast STDOUT, bo tam są logi
            recommendations = parse_recommendations(stderr)
            
            if recommendations:
                print("🏆 Top 3 rekomendacje:")
                for rec in recommendations:
                    print(f"   {rec['num']}. {rec['name']} ({rec['type']}) - ID: {rec['id']}")
            else:
                print("❌ Brak rekomendacji lub błąd parsowania")
                # Debug - pokaż surowy output jeśli nie ma rekomendacji
                if "Top" in stderr:
                    print(f"   Debug: Znaleziono 'Top' w stderr")
                elif "WARNING" in stderr:
                    print("   ⚠️  Użytkownik nie istnieje w danych treningowych")
        else:
            print(f"❌ Błąd: {stderr}")
    
    print("\n" + "=" * 50)
    print("✅ Demo zakończone!")
    print("\n💡 Aby wygenerować rekomendacje dla własnego użytkownika:")
    print('   python simple_hm_recommender.py --action recommend --user_id "YOUR_USER_ID" --top_k 10')

if __name__ == "__main__":
    main() 