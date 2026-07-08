import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Carga datos de flores
datos= pd.read_csv("/home/deny-vic/PythonCod/flores.csv")

#separar datos 
x = datos[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = datos["species"]

#dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#crear modelo con 100 arboles
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(x_train, y_train)

#predecir y medir precision
predicciones = modelo.predict(x_test)
precision = accuracy_score(y_test, predicciones)

print(f"Precision con 1 arbol (Decision Tree): 96.7%")  
print(f"Precision con 100 arboles (Random Forest): {precision*100:.1f}%")