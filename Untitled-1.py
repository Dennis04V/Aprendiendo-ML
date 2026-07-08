import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Carga de archivo CSV
datos= pd.read_csv("/home/deny-vic/PythonCod/flores.csv")

#Ver los primeros 5 registros
print("primeras 5 flores: ")
print(datos.head())
print()

#Ver cuentas flores tenemos
print(f"Total de flores {len(datos)}")
print()

#ver que tipos de flores hay
print("Tipos de flores: ")
print(datos["species"].unique())
print()

from sklearn.model_selection import train_test_split

#separa datos en X (medias) y Y (tipo de flor)
x = datos[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = datos["species"]

#dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Entrenar el modelo
modelo = DecisionTreeClassifier()
modelo.fit(x_train, y_train)

#predecir una flor nueva
flor_nueva = [[6.3, 2.8, 5.1, 1.5]]  # Ejemplo de medidas de una flor
prediccion = modelo.predict(flor_nueva)

print(f"La flor nueva es: {prediccion[0]}")

from sklearn.metrics import accuracy_score

#El modelo predice todas las flores de prueba
predicciones = modelo.predict(x_test)

#Coparamos las predicciones con las respuestas reales
precision = accuracy_score(y_test, predicciones)

print(f"El modelo acierta el {precision*100:.1f}% de las veces")

from sklearn.tree import export_text

#Ver como piensa el modelo
reglas= export_text(modelo, feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"])
print(reglas)