import pandas as pd
#Datos sucios de pacientes
datos = {
    "Paciente": ["Ana", "Luis", "Carlos", None, "Sofia", "Pedro", "Maria"],
    "Edad": [25, 30, None, 45, 22, 200, 35],
    "Genero": ["F", "M", "M", "F", None, "M", "F"],
    "Peso_Kg": [60, 80, 65, None, 70, 55, 999],
    "Diagnostico": ["ansiedad", "depreciónn", "ANSIEDAD", "depresion", "Ansiedad", None, "Depresión"]
}

df = pd.DataFrame(datos)
#Datos sucios
print("===DATOS SUCIOS=== ")
print(df)
print()
#Datos faltantes
print("===DATOS FALTANTES=== ")
print(df.isnull().sum())

#Estandarizar diagnosticos a minusculas
df["Diagnostico"] = df["Diagnostico"].str.lower()

#Estandarizar genero a mayusculas
df["Genero"] = df["Genero"].str.upper()
df["Genero"] = df["Genero"].fillna("DESCONOCIDO")

#Eliminar edad imposible mayos a 120
df = df[df["Edad"] < 120]

#Eliminar peso imposible mayos a 300
df = df[df["Peso_Kg"] < 300]

#Rellenar nombre faltante
df["Paciente"] = df["Paciente"].fillna("Desconocido")

#Rellenar peso faltante con promedio
df["Peso_Kg"] = df["Peso_Kg"].fillna(df["Peso_Kg"].mean())

print("===DATOS LIMPIOS=== ")
print(df)


