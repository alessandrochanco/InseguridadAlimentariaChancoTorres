import os
import pandas as pd

# Define las rutas basadas en el archivo actual (views.py)
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "resultados")

# Crear el directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# Leer los archivos CSV
try:
    enaho = pd.read_csv(os.path.join(DATA_DIR, "Lima_M.csv"), sep=';', encoding='Latin-1')
    index = pd.read_csv(os.path.join(DATA_DIR, "ubigeo_distrito.csv"), sep=';', encoding='Latin-1')
except FileNotFoundError as e:
    print(f"Error al intentar leer los archivos: {e}")
    # Maneja el error si algún archivo no se encuentra
