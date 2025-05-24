import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import datetime
import uuid
import sys
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext, Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# Dodaj folder nadrzędny do ścieżki, aby móc importować z innych modułów
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AzureStorageHandler:
    """
    Klasa do obsługi Azure Blob Storage do przechowywania danych i modeli.
    """
    def __init__(
        self, 
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        container_name: str = "sukienki-recommender"
    ):
        """
        Inicjalizuje obsługę Azure Blob Storage.
        
        Args:
            connection_string: Connection string do konta Azure Storage
            account_url: URL konta Azure Storage (używane z DefaultAzureCredential)
            container_name: Nazwa kontenera w Azure Storage
        """
        self.container_name = container_name
        
        if connection_string:
            # Użyj connection string, jeśli podany
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_url:
            # Użyj DefaultAzureCredential, jeśli podany account_url
            credential = DefaultAzureCredential()
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError("Musisz podać albo connection_string, albo account_url")
        
        # Upewnij się, że kontener istnieje
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Tworzy kontener, jeśli nie istnieje."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client = self.blob_service_client.create_container(self.container_name)
                print(f"Utworzono kontener: {self.container_name}")
            else:
                print(f"Kontener {self.container_name} już istnieje")
        except Exception as e:
            print(f"Błąd podczas tworzenia kontenera: {e}")
            raise
    
    def upload_dataframe(self, df: pd.DataFrame, blob_name: str) -> str:
        """
        Przesyła DataFrame do Azure Blob Storage jako CSV.
        
        Args:
            df: DataFrame do przesłania
            blob_name: Nazwa pliku docelowego w Azure Blob Storage
            
        Returns:
            URL do przesłanego pliku
        """
        # Dodaj rozszerzenie .csv, jeśli nie ma
        if not blob_name.endswith('.csv'):
            blob_name += '.csv'
        
        # Konwersja DataFrame do CSV
        csv_data = df.to_csv(index=False)
        
        # Prześlij jako blob
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        
        blob_client.upload_blob(csv_data, overwrite=True)
        
        # Zwróć URL do pliku
        return blob_client.url
    
    def download_dataframe(self, blob_name: str) -> pd.DataFrame:
        """
        Pobiera DataFrame z Azure Blob Storage.
        
        Args:
            blob_name: Nazwa pliku w Azure Blob Storage
            
        Returns:
            Pobrany DataFrame
        """
        # Dodaj rozszerzenie .csv, jeśli nie ma
        if not blob_name.endswith('.csv'):
            blob_name += '.csv'
        
        # Pobierz blob
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        
        # Pobierz zawartość
        download_stream = blob_client.download_blob()
        csv_data = download_stream.readall()
        
        # Utwórz DataFrame
        return pd.read_csv(pd.io.common.BytesIO(csv_data))
    
    def upload_model(self, model_dir: str, base_blob_name: str) -> Dict[str, str]:
        """
        Przesyła model PyTorch i powiązane pliki do Azure Blob Storage.
        
        Args:
            model_dir: Katalog zawierający pliki modelu
            base_blob_name: Bazowa nazwa dla plików w Azure Blob Storage
            
        Returns:
            Słownik z URL do przesłanych plików
        """
        # Sprawdź, czy katalog istnieje
        if not os.path.isdir(model_dir):
            raise ValueError(f"Katalog {model_dir} nie istnieje")
        
        # Pliki do przesłania
        file_urls = {}
        
        # Prześlij pliki
        for filename in os.listdir(model_dir):
            # Pełna ścieżka pliku
            file_path = os.path.join(model_dir, filename)
            
            # Pomiń katalogi
            if os.path.isdir(file_path):
                continue
            
            # Nazwa docelowa w Azure Blob Storage
            blob_name = f"{base_blob_name}/{filename}"
            
            # Prześlij plik
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Dodaj URL do słownika
            file_urls[filename] = blob_client.url
        
        return file_urls
    
    def download_model(self, base_blob_name: str, target_dir: str) -> str:
        """
        Pobiera model PyTorch i powiązane pliki z Azure Blob Storage.
        
        Args:
            base_blob_name: Bazowa nazwa dla plików w Azure Blob Storage
            target_dir: Katalog docelowy na pliki
            
        Returns:
            Ścieżka do katalogu z modelem
        """
        # Upewnij się, że katalog docelowy istnieje
        os.makedirs(target_dir, exist_ok=True)
        
        # Pobierz listę blobów z prefiksem
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blobs = container_client.list_blobs(name_starts_with=base_blob_name)
        
        # Pobierz każdy blob
        for blob in blobs:
            # Nazwa pliku (bez prefiksu)
            filename = os.path.basename(blob.name)
            
            # Ścieżka docelowa
            target_path = os.path.join(target_dir, filename)
            
            # Pobierz blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob.name
            )
            
            # Pobierz zawartość
            download_stream = blob_client.download_blob()
            
            # Zapisz plik
            with open(target_path, "wb") as file:
                file.write(download_stream.readall())
            
            print(f"Pobrano {blob.name} do {target_path}")
        
        return target_dir

class AzureMLHandler:
    """
    Klasa do obsługi Azure ML do treningu i wdrożenia modelu.
    """
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str
    ):
        """
        Inicjalizuje obsługę Azure ML.
        
        Args:
            subscription_id: ID subskrypcji Azure
            resource_group: Nazwa grupy zasobów
            workspace_name: Nazwa obszaru roboczego Azure ML
        """
        # Inicjalizacja uwierzytelniania
        self.credential = DefaultAzureCredential()
        
        # Inicjalizacja klienta ML
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
    
    def deploy_model(
        self,
        model_dir: str,
        model_name: str,
        endpoint_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1
    ) -> str:
        """
        Wdraża model PyTorch do Azure ML.
        
        Args:
            model_dir: Katalog zawierający pliki modelu
            model_name: Nazwa modelu w Azure ML
            endpoint_name: Nazwa endpointu (jeśli None, zostanie wygenerowana)
            deployment_name: Nazwa wdrożenia (jeśli None, zostanie wygenerowana)
            instance_type: Typ instancji obliczeniowej
            instance_count: Liczba instancji
            
        Returns:
            URL endpointu
        """
        # Generuj nazwy, jeśli nie podano
        if endpoint_name is None:
            endpoint_name = f"sukienki-endpoint-{str(uuid.uuid4())[:8]}"
        
        if deployment_name is None:
            deployment_name = f"deployment-{str(uuid.uuid4())[:8]}"
        
        # Sprawdź, czy katalog istnieje
        if not os.path.isdir(model_dir):
            raise ValueError(f"Katalog {model_dir} nie istnieje")
        
        # Załaduj informacje o modelu
        with open(os.path.join(model_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)
        
        # Utwórz wersję modelu opartą na czasie
        model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rejestruj model w Azure ML
        model = Model(
            path=model_dir,
            name=model_name,
            version=model_version,
            type=AssetTypes.CUSTOM_MODEL,
            description=f"Model rekomendacji sukienek (v{model_version})"
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        print(f"Zarejestrowano model: {registered_model.name}, wersja: {registered_model.version}")
        
        # Stwórz endpoint online
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Endpoint dla systemu rekomendacji sukienek",
            auth_mode="key"
        )
        
        try:
            # Sprawdź, czy endpoint już istnieje
            existing_endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            print(f"Endpoint {endpoint_name} już istnieje, dodaję nowe wdrożenie")
        except Exception:
            # Jeśli nie istnieje, utwórz nowy
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            print(f"Utworzono endpoint: {endpoint_name}")
        
        # Stwórz wdrożenie
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=registered_model.id,
            instance_type=instance_type,
            instance_count=instance_count,
            environment_variables={
                "PYTORCH_VERSION": "1.12.0",
                "MODEL_NAME": model_name,
                "MODEL_VERSION": model_version
            }
        )
        
        # Wdróż model
        self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"Utworzono wdrożenie: {deployment_name}")
        
        # Ustaw to wdrożenie jako domyślne
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        # Zwróć URL endpointu
        endpoint_url = self.ml_client.online_endpoints.get(endpoint_name).scoring_uri
        print(f"Endpoint URL: {endpoint_url}")
        
        return endpoint_url
    
    def get_recommendation(
        self,
        endpoint_name: str,
        user_id: int,
        sukienki_df: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Pobiera rekomendacje z wdrożonego modelu.
        
        Args:
            endpoint_name: Nazwa endpointu
            user_id: ID użytkownika
            sukienki_df: DataFrame z danymi sukienek
            top_k: Liczba rekomendacji do pobrania
            
        Returns:
            DataFrame z rekomendacjami
        """
        # Przygotuj dane wejściowe
        input_data = {
            "user_id": int(user_id),
            "top_k": top_k
        }
        
        # Wywołaj endpoint
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        response = self.ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=json.dumps(input_data),
            deployment_name=None  # Użyj domyślnego wdrożenia
        )
        
        # Parsuj odpowiedź
        result = json.loads(response)
        
        # Konwersja do DataFrame
        recommendations = pd.DataFrame(result["recommendations"])
        
        # Dodaj informacje o sukienkach
        return pd.merge(
            recommendations,
            sukienki_df,
            on="item_id",
            how="left"
        ).sort_values(by="predicted_rating", ascending=False)

def upload_data_to_azure(
    sukienki_df: pd.DataFrame,
    uzytkownicy_df: pd.DataFrame,
    interakcje_df: pd.DataFrame,
    storage_handler: AzureStorageHandler
) -> Dict[str, str]:
    """
    Przesyła dane do Azure Blob Storage.
    
    Args:
        sukienki_df: DataFrame z danymi sukienek
        uzytkownicy_df: DataFrame z danymi użytkowników
        interakcje_df: DataFrame z interakcjami
        storage_handler: Instancja AzureStorageHandler
        
    Returns:
        Słownik z URL do przesłanych plików
    """
    # Dodaj timestamp, aby uniknąć nadpisywania
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prześlij dane
    sukienki_url = storage_handler.upload_dataframe(
        sukienki_df, f"sukienki_{timestamp}.csv"
    )
    
    uzytkownicy_url = storage_handler.upload_dataframe(
        uzytkownicy_df, f"uzytkownicy_{timestamp}.csv"
    )
    
    interakcje_url = storage_handler.upload_dataframe(
        interakcje_df, f"interakcje_{timestamp}.csv"
    )
    
    print(f"Dane przesłane do Azure Blob Storage z timestampem: {timestamp}")
    
    return {
        "sukienki_url": sukienki_url,
        "uzytkownicy_url": uzytkownicy_url,
        "interakcje_url": interakcje_url,
        "timestamp": timestamp
    }

def load_data_from_azure(
    storage_handler: AzureStorageHandler,
    timestamp: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pobiera dane z Azure Blob Storage.
    
    Args:
        storage_handler: Instancja AzureStorageHandler
        timestamp: Timestamp danych do pobrania (jeśli None, pobiera najnowsze)
        
    Returns:
        Krotka (sukienki_df, uzytkownicy_df, interakcje_df)
    """
    # Pobierz listę blobów
    container_client = storage_handler.blob_service_client.get_container_client(
        storage_handler.container_name
    )
    
    if timestamp is None:
        # Pobierz timestampy
        timestamps = set()
        for blob in container_client.list_blobs():
            if blob.name.startswith('sukienki_') and blob.name.endswith('.csv'):
                # Ekstrahuj timestamp z nazwy
                ts = blob.name.replace('sukienki_', '').replace('.csv', '')
                timestamps.add(ts)
        
        if not timestamps:
            raise ValueError("Nie znaleziono danych w Azure Blob Storage")
        
        # Wybierz najnowszy timestamp
        timestamp = max(timestamps)
        print(f"Używam najnowszego timestampu: {timestamp}")
    
    # Pobierz dane
    sukienki_df = storage_handler.download_dataframe(f"sukienki_{timestamp}.csv")
    uzytkownicy_df = storage_handler.download_dataframe(f"uzytkownicy_{timestamp}.csv")
    interakcje_df = storage_handler.download_dataframe(f"interakcje_{timestamp}.csv")
    
    # Konwersja kolumny timestamp na datę
    interakcje_df['timestamp'] = pd.to_datetime(interakcje_df['timestamp'])
    
    print(f"Pobrano dane z Azure Blob Storage (timestamp: {timestamp})")
    
    return sukienki_df, uzytkownicy_df, interakcje_df

if __name__ == "__main__":
    # Przykład użycia Azure Storage
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    if connection_string:
        print("Testowanie integracji Azure Storage dla systemu rekomendacji H&M")
        print("Aby użyć funkcji Azure, ustaw zmienną środowiskową AZURE_STORAGE_CONNECTION_STRING")
        
        # Utwórz obsługę Azure Storage
        storage_handler = AzureStorageHandler(connection_string=connection_string)
        print("Azure Storage Handler zainicjalizowany pomyślnie")
    else:
        print("Brak connection string do Azure Storage. Ustaw zmienną środowiskową AZURE_STORAGE_CONNECTION_STRING.") 