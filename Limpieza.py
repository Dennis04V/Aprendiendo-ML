import pandas as pd

#creamos datos sucios a própoito
datos = {
    "Nombre": ["Juan", "María", "Pedro", None, "Luis"],
    "Edad": [25, 30, None, 22, 28],
    "Salario": [15000, 18000, 999999, 16000, 17000],
    "Ciudad": ["CDMX", "guadalajara", "CDMX", "Monterrey", "cdmx"]
}

df= pd.DataFrame(datos)

print("===DATOS SUCIOS=== ")
print(df)
print()

#dectetcar datos faltantes
print("===DATOS FALTANTES=== ")
print(df.isnull().sum())

#2. Rellenar datos faltantes
df["Edad"] = df["Edad"].fillna(df["Edad"].mean())
df["Nombre"] = df["Nombre"].fillna("Desconocido")

#3.Eliminar salarios raros (mayores a 100000)
df = df[df["Salario"] < 100000]

#4.Estandarizar ciudades a mayusculas
df["Ciudad"] = df["Ciudad"].str.upper()

print("===DATOS LIMPIOS=== ")
print(df)