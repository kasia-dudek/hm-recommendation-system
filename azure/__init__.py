# Moduł azure zawiera funkcje do integracji z usługami Azure
from azure.azure_integration import (
    AzureStorageHandler, 
    AzureMLHandler, 
    upload_data_to_azure, 
    load_data_from_azure
)

__all__ = [
    'AzureStorageHandler',
    'AzureMLHandler',
    'upload_data_to_azure',
    'load_data_from_azure'
] 