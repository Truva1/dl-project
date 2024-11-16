import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr, chi2_contingency

# Configuraciones generales para los gráficos
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# =========================
# Carga y exploración de datos
# =========================

# Cargar el dataset en un DataFrame
df = pd.read_csv('./data/NTAD_fatalities_2022.csv')

# Visualización inicial del dataset
print(df.head())       # Primeras filas del dataset
print(df.info())       # Información sobre el tipo de datos y valores nulos
print(df.describe())   # Resumen estadístico de las columnas numéricas
print(f"Número total de registros: {len(df)}")  # Número de registros en el dataset

# =========================
# Limpieza y preprocesamiento de datos
# =========================

# Identificar y mostrar el número de valores nulos en cada columna
null_counts = df.isnull().sum()
print(f"Valores nulos antes de limpieza:\n{null_counts}")

# Eliminar filas con valores nulos en columnas críticas (ej. 'x', 'y')
df.dropna(subset=['x', 'y'], inplace=True)
null_counts_after = df.isnull().sum()
print(f"Valores nulos después de limpieza:\n{null_counts_after}")

# Filtrar registros con horas válidas (entre 0 y 24)
df_f = df[df['HOUR'] <= 24]

# Crear una columna de tiempo combinando 'HOUR' y 'MINUTE'
df_f['TIME'] = df_f['HOUR'].astype(str) + ':' + df_f['MINUTE'].astype(str)

# Filtrar solo accidentes en áreas urbanas
df_urban = df_f[df_f['RUR_URBNAME'] == 'Urban']

# =========================
# Visualización de datos
# =========================

# Gráfico de fatalidades según condiciones climáticas
sns.countplot(data=df_f, x='WEATHERNAME', hue='FATALS')
plt.title("Fatalidades según el clima")
plt.xticks(rotation=45)
plt.show()

# Gráfico de barras apiladas para comparar fatalidades en áreas urbanas y rurales
plt.figure(figsize=(10, 6))
sns.countplot(data=df_f, x='RUR_URBNAME', hue='FATALS', palette='viridis')
plt.title('Comparación de Fatalidades en Áreas Urbanas vs Rurales')
plt.xlabel('Tipo de Área')
plt.ylabel('Frecuencia de Fatalidades')
plt.legend(title='Número de Fatalidades', loc='upper right')
plt.show()


# Distribución de accidentes fatales por hora del día
sns.histplot(data=df_f, x='HOUR', kde=True)
plt.title("Distribución de accidentes fatales por hora")
plt.show()

# Distribución de accidentes fatales por día de la semana
sns.countplot(data=df_f, x='DAY_WEEKNAME', hue='FATALS')
plt.title("Distribución de accidentes fatales por día de la semana")
plt.xticks(rotation=45)
plt.show()

# Distribución de accidentes fatales por mes
sns.countplot(data=df_f, x='MONTHNAME', hue='FATALS')
plt.title("Distribución de accidentes fatales por mes")
plt.xticks(rotation=45)
plt.show()

# Accidentes fatales según la función de la carretera
sns.countplot(data=df_f, x='FUNC_SYSNAME', hue='FATALS')
plt.title("Accidentes fatales según la función de la carretera")
plt.xticks(rotation=45)
plt.show()

# Distribución de fatalidades según la condición de luz
sns.countplot(data=df_f, x='LGT_CONDNAME', hue='FATALS')
plt.title("Distribución de fatalidades según la condición de luz")
plt.xticks(rotation=45)
plt.show()

# Tipo de colisión y su relación con fatalidades
sns.countplot(data=df_f, x='MAN_COLLNAME', hue='FATALS')
plt.title("Relación del tipo de colisión con fatalidades")
plt.xticks(rotation=45)
plt.show()

# =========================
# Análisis de correlación
# =========================

# Matriz de correlación entre variables numéricas seleccionadas
corr_matrix = df_f[['HOUR', 'VE_TOTAL', 'FATALS']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de correlación de variables numéricas")
plt.show()

# Función para calcular y almacenar la correlación
def calcular_correlacion(df, variable1, variable2, tipo='num'):
    """
    Calcula la correlación entre dos variables.

    Args:
        df: DataFrame con los datos.
        variable1: Nombre de la primera variable.
        variable2: Nombre de la segunda variable.
        tipo: Tipo de variables ('num' para numéricas, 'cat' para categóricas).

    Returns:
        Tupla con el coeficiente de correlación y el p-valor (si corresponde).
    """

    if tipo == 'num':
        corr, p_value = spearmanr(df[variable1], df[variable2])
        return corr, p_value
    else:
        # Create contingency table for categorical variables
        contingency_table = pd.crosstab(df[variable1], df[variable2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p

# Examples of correlation calculations
corr_hour_fatals, p_value_hour_fatals = calcular_correlacion(df_f, 'HOUR', 'FATALS', 'num')
print(f"Correlación entre HOUR y FATALS: {corr_hour_fatals:.2f}, p-value: {p_value_hour_fatals:.4f}")



# =========================
# Clusterización usando KMeans
# =========================

# Selección de características para la clusterización
features = df_f[['HOUR', 'VE_TOTAL', 'FATALS']]

# Escalado de características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.dropna())

# Aplicación de KMeans con 4 clusters (ajustar según necesidades)
kmeans = KMeans(n_clusters=4, random_state=42)
df_f['cluster'] = kmeans.fit_predict(scaled_features)

# Visualización de los clusters formados
sns.scatterplot(data=df_f, x='HOUR', y='VE_TOTAL', hue='cluster', palette='Set2')
plt.title("Clusterización de accidentes basada en hora y número de vehículos")
plt.show()

# =========================
# Modelado predictivo con Random Forest
# =========================

# Selección de características para el modelo
features = ['FUNC_SYS', 'LGT_CONDNAME']   # Ajustar según las necesidades
X = df_f[features]       # Variables predictoras
y = df_f['FATALS']       # Variable objetivo

# Manejo de datos faltantes
X = X.dropna()
y = y[X.index]  # Alineación de índices

# Codificación de variables categóricas
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X).toarray()
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(features))

# División del dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.3, random_state=42)

# Creación y entrenamiento del modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# =========================
# Evaluación del modelo
# =========================

# Predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Cálculo de métricas de evaluación
mse_rf = mean_squared_error(y_test, y_pred)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)

# Mostrar resultados
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"R² Score: {r2_rf}")

# =========================
# Visualización de resultados del modelo
# =========================

# Importancia de las características en Random Forest
importances = rf_model.feature_importances_
feature_names = encoder.get_feature_names_out(features)

# Gráfico de la importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Importancia de las Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

# Visualización de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Fallecidos Reales')
plt.ylabel('Fallecidos Predichos')
plt.title('Predicción de Fallecidos vs Reales (Random Forest)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()

# Distribución de residuos del modelo
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.title('Distribución de Residuos')
plt.xlabel('Error de Predicción (Fallecidos Reales - Fallecidos Predichos)')
plt.ylabel('Frecuencia')
plt.show()

# Línea de tendencia en la predicción
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={'color':'green'}, scatter_kws={'alpha':0.6})
plt.xlabel('Fallecidos Reales')
plt.ylabel('Fallecidos Predichos')
plt.title('Predicción de Fallecidos vs Reales con Línea de Tendencia')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()
