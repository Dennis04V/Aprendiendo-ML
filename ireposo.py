import pandas as pd

# Cargar datos del Titanic
datos = pd.read_csv("/home/deny-vic/PythonCod/train.csv")

# Ver primeras 5 filas
print("=== PRIMERAS 5 FILAS ===")
print(datos.head())
print()

# Ver cuántos pasajeros hay
print(f"Total de pasajeros: {len(datos)}")
print()

# Ver las columnas
print("=== COLUMNAS ===")
print(datos.columns)
print()

# Ver datos faltantes
print("=== DATOS FALTANTES ===")
print(datos.isnull().sum())