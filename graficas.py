import pandas as pd
import matplotlib.pyplot as plt

#Cargar datos de flores
datos = pd.read_csv("/home/deny-vic/PythonCod/flores.csv")

#graficade cuentasflores hay de cada tipo de flor
datos["species"].value_counts().plot(kind="bar", color =["blue", "green", "red"])

plt.title("Cantidad de flores por tipo")
plt.xlabel("Tipo de flor")
plt.ylabel("Cantidad de flores")
plt.tight_layout()
plt.savefig("/home/deny-vic/PythonCod/grafica_flores.png")
print ("Gráfica guardada.")
