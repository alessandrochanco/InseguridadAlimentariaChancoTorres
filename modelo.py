from geodatasets import get_path
from highlight_text import fig_text, ax_text
from imblearn.over_sampling import RandomOverSampler
from pypalettes import load_cmap
from shapely import geometry
from shapely.geometry import MultiPolygon, Polygon, LineString, Point, MultiPoint
from shapely.geometry import Point, Polygon
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import Point, polygon, LineString
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,precision_score,recall_score, f1_score,roc_auc_score,roc_curve,balanced_accuracy_score,confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import IPython.display as display
import copy
import csv
import descartes
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import sys
import warnings
import os


 
def ejecutar_modelo(outdir):
    # Lógica del modelo aquí
    print("Ejecutando el modelo...")
    # Ejemplo de lo que la función puede hacer
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig(f"{outdir}/grafico.png")
    plt.close()
    return [f"{outdir}/grafico.png"]  # Retornar las rutas de las imágenes generadas
    

#import contextily as ctx
#import mplleaflet
#import geoplot.crs as gcrs

# import seaborn as sb
# from datetime import datetime
warnings.filterwarnings('ignore')

#from bioinfokit import analys, visuz

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
plt.show()
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)





warnings.filterwarnings('ignore')
#Carga de base de datos 
#enaho = pd.read_csv(r"D:\EPI-SOCIAL 2024\Tesista-ChancoG\Datos\MODELO\Lima_M.csv", sep=';', encoding='Latin-1')
#index = pd.read_csv(r"D:\EPI-SOCIAL 2024\Datos_ubigeo_indx\ubigeo_distrito.csv", sep=';', encoding='Latin-1')
#visualizamos la composición de las bases de datos Enaho2023
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

enaho = pd.read_csv(os.path.join(DATA_DIR, "Lima_M.csv"), sep=';', encoding='Latin-1')
index = pd.read_csv(os.path.join(DATA_DIR, "ubigeo_distrito.csv"), sep=';', encoding='Latin-1')


print(enaho.head())
enaho.head(2)
print(enaho.shape)
print(list(enaho.columns))
enaho.info()
duplicate_rows_enaho = enaho[enaho.duplicated()]
print("number of duplicate rows: ", duplicate_rows_enaho.shape)
#visualizamos la composición de las bases de datos Index 

print(index.head())
index.head(2)
print(index.shape)
print(list(index.columns))
index.info()
duplicate_rows_index = index[index.duplicated()]
print("number of duplicate rows: ", duplicate_rows_index.shape)
index1 = gpd.GeoDataFrame(
    index, geometry=gpd.points_from_xy(index.longitude, index.latitude), crs="EPSG:4326"
)
index1.info()
index1.head(2)
# Antes de realizar el merge, asegurarse de que las columnas 'inei' sean del mismo tipo (str)
enaho['inei'] = enaho['inei'].astype(str)
index1['inei'] = index1['inei'].astype(str)

df = pd.merge(enaho, index1, on=['inei'], how='inner')
df.head(2)
print(df.shape)
print(list(df.columns))
#en base al archivo de códigos, traer las columnas que vamos a trabajar 

df_trabajo = df[['YEAR', 'C_V_H_ID', 'inei', 'ESTRATO', 'P601A', 'P601A_PRODUCTO', 'P601A_CAT', 'P601X_PROD', 'P601B', 'P601B1', 'P601B4', 'FACTOR07', 'reniec', 'departamento', 'provincia', 'distrito', 'region', 'macroregion_inei', 'macroregion_minsa', 'iso_3166_2', 'fips', 'capital', 'superficie', 'pob_densidad_2020', 'altitude', 'latitude', 'longitude', 'indice_vulnerabilidad_alimentaria', 'idh_2019', 'pct_pobreza_total', 'pct_pobreza_extrema', 'geometry']]

df_trabajo.head(5)
print(df_trabajo.shape)
print(list(df_trabajo.columns))
#hacemos una columna count = 1, porque cada fila representa un individuo (aun no usamos el fexp)
df_trabajo['count'] = 1
df_trabajo.head(2)
LM_df = df_trabajo[df_trabajo['macroregion_inei']== "LIMA METROPOLITANA"]
LM_df.head(2)
print(LM_df.shape)
LM_df.info()
duplicate_rows_LM_df = LM_df[LM_df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_LM_df.shape)
print(LM_df.isnull().sum())
# Convertir la columna a tipo float
LM_df['superficie'] = LM_df['superficie'].astype(float)
LM_df['pob_densidad_2020'] = LM_df['pob_densidad_2020'].astype(float)
#SUMAR FEXP DE LOS DISTRITOS DE LIMA PARA CORROBORAR EL TOTAL DE LA POBLACION 
suma_distritos = LM_df['FACTOR07'].sum()
suma_distritos
LM_df.describe()
LM_df.describe(include='O')
print(LM_df.isnull().sum())
LM_df = LM_df.replace(r'^\s*$', np.nan, regex=True)
LM_df
# Convertir la columna a tipo float
LM_df['P601B1'] = LM_df['P601B1'].astype(float)
LM_df['P601B4'] = LM_df['P601B4'].astype(float)
LM_df.info()
LM_df = LM_df[LM_df['P601B1'].notna()]
LM_df = LM_df[LM_df['P601B4'].notna()]
LM_df.info()
df_final = LM_df[['inei', 'P601A', 'P601A_PRODUCTO', 'P601B', 'P601B1', 'P601B4', 'distrito','pob_densidad_2020', 'latitude', 'longitude', 'indice_vulnerabilidad_alimentaria', 'idh_2019', 'pct_pobreza_total', 'pct_pobreza_extrema']]

df_final.head(5)
df_final.groupby('distrito').mean(numeric_only=True)
correlation_matrix = df_final.corr(numeric_only=True, method='pearson')
correlation_matrix['indice_vulnerabilidad_alimentaria'].sort_values(ascending=False)
correlation_matrix['indice_vulnerabilidad_alimentaria'].sort_values(ascending=False).to_csv('summary_corr.csv', sep=',')
df_final1 = pd.get_dummies(df_final, columns=['distrito'], drop_first=True)
df_final1.head(5)
df_final1 = df_final1.rename(columns={'distrito_BREÃA': 'distrito_BRENA'})
df_final1 = df_final1.astype(float)
df_final1 ['distrito_ATE'] = df_final1['distrito_ATE'].astype(int)
df_final1 ['distrito_BARRANCO'] = df_final1['distrito_BARRANCO'].astype(int)
df_final1 ['distrito_BRENA'] = df_final1['distrito_BRENA'].astype(int)
df_final1 ['distrito_CARABAYLLO'] = df_final1['distrito_CARABAYLLO'].astype(int)
df_final1 ['distrito_CIENEGUILLA'] = df_final1['distrito_CIENEGUILLA'].astype(int)
df_final1 ['distrito_COMAS'] = df_final1['distrito_COMAS'].astype(int)
df_final1 ['distrito_INDEPENDENCIA'] = df_final1['distrito_INDEPENDENCIA'].astype(int)
df_final1 ['distrito_LA MOLINA'] = df_final1['distrito_LA MOLINA'].astype(int)
df_final1 ['distrito_LA VICTORIA'] = df_final1['distrito_LA VICTORIA'].astype(int)
df_final1 ['distrito_LIMA'] = df_final1['distrito_LIMA'].astype(int)
df_final1 ['distrito_LINCE'] = df_final1['distrito_LINCE'].astype(int)
df_final1 ['distrito_LOS OLIVOS'] = df_final1['distrito_LOS OLIVOS'].astype(int)
df_final1 ['distrito_LURIGANCHO'] = df_final1['distrito_LURIGANCHO'].astype(int)
df_final1 ['distrito_LURIN'] = df_final1['distrito_LURIN'].astype(int)
df_final1 ['distrito_PUEBLO LIBRE'] = df_final1['distrito_PUEBLO LIBRE'].astype(int)
df_final1 ['distrito_PUENTE PIEDRA'] = df_final1['distrito_PUENTE PIEDRA'].astype(int)
df_final1 ['distrito_RIMAC'] = df_final1['distrito_RIMAC'].astype(int)
df_final1 ['distrito_SAN BORJA'] = df_final1['distrito_SAN BORJA'].astype(int)
df_final1 ['distrito_SAN JUAN DE MIRAFLORES'] = df_final1['distrito_SAN JUAN DE MIRAFLORES'].astype(int)
df_final1 ['distrito_SAN MARTIN DE PORRES'] = df_final1['distrito_SAN MARTIN DE PORRES'].astype(int)
df_final1 ['distrito_SANTA ANITA'] = df_final1['distrito_SANTA ANITA'].astype(int)
df_final1 ['distrito_SANTIAGO DE SURCO'] = df_final1['distrito_SANTIAGO DE SURCO'].astype(int)
df_final1 ['distrito_SURQUILLO'] = df_final1['distrito_SURQUILLO'].astype(int)
df_final1 ['distrito_VILLA EL SALVADOR'] = df_final1['distrito_VILLA EL SALVADOR'].astype(int)
df_final1.head(5)
df_final1.columns.values
selected_features = [ 'inei', 'P601A', 'P601A_PRODUCTO', 'P601B', 'P601B1', 'P601B4',
       'pob_densidad_2020', 'idh_2019',
       'pct_pobreza_extrema'
       ]
X = df_final1[selected_features]
y = df_final1['indice_vulnerabilidad_alimentaria']

print (X)
print (y)

# Initialize the StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the data and transform it
X_scaled = scaler.fit_transform(X)

# Print the scaled data
print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# The 'LinearRegression' model is initialized and fitted to the training data.
model = LinearRegression()
model.fit(X_train, y_train)

# The model is used to predict the target variable for the test set.
y_pred = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)
regressor = LinearRegression().fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")

# Add a constant to the model
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# Q-Q Plot for residuals
sm.qqplot(model_sm.resid, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

vif_data = pd.DataFrame()
vif_data['Feature'] = selected_features
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)

# Bar Plot for VIF Values
vif_data.plot(kind='bar', x='Feature', y='VIF', legend=False)
plt.title('Variance Inflation Factor (VIF) by Feature')
plt.ylabel('VIF Value')
plt.show()
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print("Cross-Validation Scores:", scores)
print("Mean CV R^2:", scores.mean())

# Line Plot for Cross-Validation Scores
plt.plot(range(1, 6), scores, marker='o', linestyle='--')
plt.xlabel('Fold')
plt.ylabel('R-squared')
plt.title('Cross-Validation R-squared Scores')
plt.show()
rfe = RFE(estimator=LinearRegression(), n_features_to_select=3)
rfe.fit(X_scaled, y)
print("Selected Features:", rfe.support_)

# Bar Plot of Feature Rankings
feature_ranking = pd.DataFrame({
   'Feature': selected_features,
   'Ranking': rfe.ranking_
})
feature_ranking.sort_values(by='Ranking').plot(kind='bar', x='Feature', y='Ranking', legend=False)
plt.title('Feature Ranking (Lower is Better)')
plt.ylabel('Ranking')
plt.show()
#Prediction OLS
performance = pd.DataFrame({'PREDIC IVA':y_pred, 'VALORES ACTUALES':y_test})
performance ['ERROR'] = (performance['VALORES ACTUALES'] -performance['PREDIC IVA'])
performance.head(5)