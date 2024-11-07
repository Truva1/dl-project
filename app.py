import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset en un DataFrame
df = pd.read_csv('./data/NTAD_fatalities_2022.csv')

# Mostrar las primeras filas y las columnas
print(df.head())
print(df.info())  # Ver tipos de datos y valores nulos
print(df.describe())  # Resumen estadístico de las columnas numéricas
print(len(df)) #numero de registros

# identificar registro con valores nulos
null_counts = df.isnull().sum()
print(null_counts)
# Eliminar filas con valores nulos en columnas identificadas
df.dropna(subset=['x', 'y'], inplace=True)
null_counts2 = df.isnull().sum()
print(null_counts2)

# Crear una columna de tiempo
df['TIME'] = df['HOUR'].astype(str) + ':' + df['MINUTE'].astype(str)

# Filtrar solo áreas urbanas si es necesario
df_urban = df[df['RUR_URBNAME'] == 'Urban']

# Ejemplo de gráfico de fatalidades por condiciones climáticas
sns.countplot(data=df, x='WEATHERNAME', hue='FATALS')
plt.title("Fatalidades según el clima")
plt.xticks(rotation=45)
plt.show()

# Gráfico de distribución de accidentes fatales por hora del día
sns.histplot(data=df, x='HOUR', kde=True)
plt.title("Distribución de accidentes fatales por hora")
plt.show()