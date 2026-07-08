import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos reales de flores
datos= pd.read_csv("/home/deny-vic/PythonCod/flores.csv")

print(datos.head())
print()

# Separar datos
X = datos[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = datos["species"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

#medir precisión del modelo
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"Precisión del modelo: {precision * 100:.1f}%")

precision_train = accuracy_score(y_train, modelo.predict(X_train))
precision_test = accuracy_score(y_test, modelo.predict(X_test))

print(f"Precisión en entrenamiento: {precision_train * 100:.1f}%")
print(f"Precisión en prueba: {precision_test * 100:.1f}%")