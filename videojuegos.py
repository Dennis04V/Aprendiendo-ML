import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Datos de videojuegos
#Columnas: [precio, horas_contenido, multijugador, accion, precio_justo]
#Multijugaodr : 1=si, 0=no
#Accion: 1=si, 0=no
#Exitoso : 1=si, 0=no

datos = {
    "juego": [
        "God of War", "Fifa 24", "Minecraft", "Fornite",
        "Cyberpunk 2077", "Amoung Us", "The las of us",
        "Fall Guys", "Elden Ring", "Ea sport FC",
        "GTA V", "Red Dead Redemption 2", "Hallow Knight",
        "Call of Duty", "Spider-Man", "Stardew Valley",
        "Dark Souls", "Rocket League", "Hades", "Overwatch"

    ],
    "Precio": [
        70, 70, 30, 0, 
        60, 5, 70, 0, 60, 70,
        30, 60, 15, 70, 70, 15,
        40, 0, 25, 40
    ],
    "Horas_contenido": [
        50, 100, 1000, 500,
        100, 10, 25, 20, 80, 100,
        200, 150, 40, 80, 30, 200,
        70, 500, 50, 500
    ],
    "Multijugador": [
        0, 1, 1, 1,
        0, 1, 0, 1, 0, 1,
        1, 0, 0, 1, 0, 0, 
        0, 1, 0, 1
    ],
    "Accion": [
        1, 0, 0, 1,
        1, 0, 1, 0, 1, 0,
        1, 1, 1, 1, 1, 0,
        1, 0, 1, 1
    ],
    "Exitoso": [
        1, 1, 1, 1,
        0, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1
    ] 
}

df = pd.DataFrame(datos)

print("=== DATOS DE VIDEOJUEGOS ===")
print(df)
print()

#Separar datos
x = df [["Precio", "Horas_contenido", "Multijugador", "Accion"]]
y = df["Exitoso"]

#Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Entrenar modelo
modelo = DecisionTreeClassifier()
modelo.fit(x_train, y_train)
#Predecir un juego nuevo
juego_nuevo = [[70, 200, 0, 1]] #Precio=70, 60hrs, sin multi, con accion
prediccion = modelo.predict(juego_nuevo)

if prediccion[0] == 1:
    print("El juego nuevo se predice como EXITOSO.")
else:
    print("El juego nuevo se predice como NO EXITOSO.")

    

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Separar datos para prueba
X = df[["Precio", "Horas_contenido", "Multijugador", "Accion"]]
y = df["Exitoso"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree
modelo_dt = DecisionTreeClassifier()
modelo_dt.fit(X_train, y_train)
pred_dt = modelo_dt.predict(X_test)
precision_dt = accuracy_score(y_test, pred_dt)

# Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100)
modelo_rf.fit(X_train, y_train)
pred_rf = modelo_rf.predict(X_test)
precision_rf = accuracy_score(y_test, pred_rf)

print("=== COMPARACION DE ALGORITMOS ===")
print(f"Decision Tree  → {precision_dt * 100:.1f}%")
print(f"Random Forest  → {precision_rf * 100:.1f}%")

if precision_rf > precision_dt:
    print("Random Forest gana! 🌲")
elif precision_dt > precision_rf:
    print("Decision Tree gana! 🌳")
else:
    print("Empate! 🤝")