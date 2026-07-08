import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos del Titanic
datos = pd.read_csv("/home/deny-vic/PythonCod/titanic/train.csv")

#limpiar datos
datos = datos.drop("Cabin", axis=1)
datos["Age"] = datos["Age"].fillna(datos["Age"].median())
datos["Embarked"] = datos["Embarked"].fillna(datos["Embarked"].mode()[0])

#convertir texto a numero
datos["Sex"] = datos["Sex"].map({"male":0, "female":1})
datos["Embarked"] = datos["Embarked"].map({"S":0, "C":1, "Q":2})

#eliminar columnas innecesarias
datos = datos.drop(["Name", "Ticket", "PassengerId"], axis=1)

#Separar datos
X = datos[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = datos["Survived"]

#entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

#medir precision
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"Precision del modelo: {precision * 100:.1f}%")

# Pasajero nuevo
pasajero = [[3, 0, 5, 1, 2, 7.25, 0]]  # Ejemplo de pasajero
prediccion = modelo.predict(pasajero)

if prediccion[0] == 1:
    print("El pasajero sobrevivirá.")
else:
    print("El pasajero no sobrevivirá.")