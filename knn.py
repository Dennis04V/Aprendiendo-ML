import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#CArgar datos de flores
datos= pd.read_csv("/home/deny-vic/PythonCod/flores.csv")
#Separar datos
x = datos[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = datos["species"]
#Dividir en entretenimiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Probar con diferentes valores de K
for k in [1, 3, 5, 7, 10]:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(x_train, y_train)
    predicciones = modelo.predict(x_test)
    precision = accuracy_score(y_test, predicciones)
    print(f"K={k} -> Precision: {precision * 100:.1f}%")
