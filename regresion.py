import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Datos: horas estudiadas vs calificacion 
horas= [1, 2, 3, 4, 5, 6, 7, 8]
calificaciones = [50, 55, 65, 70, 75, 80, 85, 95]

#Preparar datos
x = pd.DataFrame(horas, columns=["horas"])
y = calificaciones

#Crear y entrear modelo
modelo = LinearRegression()
modelo.fit (x, y)

#Predecir con horas nuevas
horas_nuevas = [[5], [9], [10]]
predicciones = modelo.predict(horas_nuevas)

print("Predicciones: ")
print(f"5 horas -> {predicciones[0]} puntos")
print(f"9 horas -> {predicciones[1]} puntos")
print(f"10 horas -> {predicciones[2]} puntos")
