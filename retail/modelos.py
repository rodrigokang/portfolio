# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Descipción >>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

# * Name: "Modelos Cencosud"
# * Owner: Rodrigo J. Kang
# * Descripción: Este script contiene modelos basados en algoritmos de ML
#                desarrollados para la empresa Cencosud S.A.. El módulo 
#                puede reciclarse para satisfacer distintas demandas de 
#                las siguientes unidades de negocio: 
#                    - Supermercado (smk)
#                    - Mejoramiento del hogar (mdh)
#                    - Cencomedia (cm)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# Evitar sugerencias de código
# ----------------------------
import warnings
warnings.filterwarnings('ignore')

# ====================================================
# Módulos para efectuar la conexión a la base de datos
# ====================================================

import psycopg2
from sqlalchemy import create_engine

# =================================
# Importar librerías de uso general
# =================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
import re

# ==================================
# Importar librerías para modelo RFM
# ==================================

# Procesamiento de datos
# ----------------------

from scipy.stats import boxcox
from scipy.stats import skew
import random

# Módulos de aprendizaje estadístico
# ----------------------------------
# RFM
# ----------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# ----------------------------------
# Abandonadores
# ----------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV # GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
from scipy.stats import zscore
# ----------------------------------
# Reglas de Asociación
# ----------------------------------
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import re

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# ==========
# Modelo RFM
# ==========

class RFM:
    """
    Clase para segmentación de clientes utilizando RFM (Recencia, Frecuencia, Monto).

    Args:
        sql_script (str): Ruta al archivo SQL que contiene la consulta para obtener los datos de clientes.
        user (str, opcional): Nombre de usuario para la conexión a la base de datos.
        password (str, opcional): Contraseña para la conexión a la base de datos.

    Attributes:
        sql_script (str): Ruta al archivo SQL que contiene la consulta para obtener los datos de clientes.
        user (str): Nombre de usuario para la conexión a la base de datos.
        password (str): Contraseña para la conexión a la base de datos.
        data (DataFrame): DataFrame que contiene los datos de clientes procesados.
    """

    def __init__(self, sql_script, user, password):
        self.sql_script = sql_script
        self.user = user
        self.password = password
        self.data = self.preprocess_data()
    
    # ********************************************************************

    def preprocess_data(self):
        """
        Realiza la preprocesamiento de los datos de clientes obtenidos de la base de datos.

        Returns:
            DataFrame: DataFrame preprocesado con las variables de RFM y transformaciones necesarias.
        """
        try:
            # Conectar a la base de datos
            conn = psycopg2.connect(
                host='datalake-dw55-prod.ctnjyflrnhjv.us-east-1.redshift.amazonaws.com',
                database='arsuperlake_prod',
                port=5439,
                user=self.user,
                password=self.password
            )

            # Ejecutar el script del query
            cursor = conn.cursor()
            cursor.execute(self.sql_script)

            # Fetch data and convert to DataFrame
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            input_data = pd.DataFrame(data, columns=columns)

            # Cerrar la conexión
            cursor.close()
            conn.close()

            # Convertir los valores de 'recencia', 'frecuencia' y 'monto' a float
            input_data['recencia'] = input_data['recencia'].astype(int)
            input_data['frecuencia'] = input_data['frecuencia'].astype(int)
            input_data['monto'] = input_data['monto'].astype(float)

            # Transformación Box-Cox
            input_data['recencia_boxcox'], LambdaRecencia = boxcox(input_data['recencia'] + 1)
            input_data['frecuencia_boxcox'], LambdaFrecuencia = boxcox(input_data['frecuencia'] + 1)
            input_data['monto_boxcox'], LambdaMonto = boxcox(input_data['monto'] + 1)

            # Tipificación
            scaler = StandardScaler()
            input_data[['s_r', 's_f', 's_m']] = scaler.fit_transform(
                input_data[['recencia_boxcox', 
                            'frecuencia_boxcox', 
                            'monto_boxcox']])

            return input_data
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante el preprocesamiento de los datos: {e}")
            print("=========================================================")
            return pd.DataFrame()
    
    # ********************************************************************
    
    def elbow_method(self):
        """
        Realiza el Método del Codo para determinar el número óptimo de clusters.
        """
        try:
            data = self.data[['s_r', 's_f', 's_m']]

            inertia_values = []

            for k in range(2, 11):  # Probamos con k desde 2 hasta 10 clusters
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(data)

                inertia = kmeans.inertia_
                inertia_values.append(inertia)

            self.elbow_plot = plt.figure(figsize=(8, 5))
            plt.plot(range(2, 11), inertia_values, marker='o')
            plt.xlabel('Número de clusters (k)')
            plt.ylabel('Suma de Cuadrados de las Distancias (Inercia)')
            plt.title('Método del Codo')
            plt.grid(True)

            return self.elbow_plot
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante el método del codo: {e}")
            print("===========================================")
            return pd.DataFrame()
    
    # ********************************************************************
    
    def random_search_kmeans(self, n_iter=10, sample_ratio=0.01, cluster_range=(2, 11)):
        """
        Realiza una búsqueda aleatoria para determinar el número óptimo de clusters para K-Means
        utilizando diferentes métricas de evaluación en subconjuntos aleatorios de los datos.

        Args:
            n_iter (int, opcional): Número de iteraciones de búsqueda aleatoria. Default es 10.
            sample_ratio (float, opcional): Proporción de datos a utilizar en cada muestra. Default es 0.01.
            cluster_range (tuple, opcional): Rango de números de clusters a evaluar. Default es (2, 11).

        Returns:
            None: Imprime los resultados de las métricas para cada iteración.
        """
        try:
            # Definir el tamaño de cada muestra
            sample_size = int(len(self.data) * sample_ratio)

            # Inicializar listas para almacenar los resultados de los índices
            cluster_numbers = list(range(cluster_range[0], cluster_range[1]))
            instance_numbers = list(range(1, n_iter + 1))

            for instance in instance_numbers:
                max_ch = -1
                max_silhouette = -1
                min_db = float('inf')
                max_ch_cluster = -1
                max_silhouette_cluster = -1
                min_db_cluster = -1

                # Tomar una muestra aleatoria del conjunto de datos
                sample_indexes = np.random.choice(len(self.data), sample_size, replace=False)
                data_sample = self.data.iloc[sample_indexes][['s_r', 's_f', 's_m']]

                for k in cluster_numbers:
                    kmeans = KMeans(n_clusters=k, random_state=0)
                    labels = kmeans.fit_predict(data_sample)

                    ch_metric = calinski_harabasz_score(data_sample, labels)
                    db_metric = davies_bouldin_score(data_sample, labels)
                    silhouette = silhouette_score(data_sample, labels)

                    # Actualizar máximos y mínimos
                    if ch_metric > max_ch:
                        max_ch = ch_metric
                        max_ch_cluster = k
                    if silhouette > max_silhouette:
                        max_silhouette = silhouette
                        max_silhouette_cluster = k
                    if db_metric < min_db:
                        min_db = db_metric
                        min_db_cluster = k

                print(f'Instancia {instance}:')
                print('------------')
                print(f'Máximo Calinski-Harabasz: Número óptimo de clusters = {max_ch_cluster}, Métrica = {max_ch}')
                print(f'Máximo Silhouette: Número óptimo de clusters = {max_silhouette_cluster}, Métrica = {max_silhouette}')
                print(f'Mínimo Davies-Bouldin: Número óptimo de clusters = {min_db_cluster}, Métrica = {min_db}')
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante la búsqueda aleatoria de clusters: {e}")
            print("==========================================================")
    
    # ********************************************************************
    
    def train_kmeans(self, n_clusters):
        """
        Entrena el modelo de KMeans para segmentar clientes en grupos.

        Args:
            n_clusters (int): Número de clústeres a crear.
        """
        try:
            # Entrenar el modelo KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(self.data[['s_r', 's_f', 's_m']])
            self.data['cluster'] = cluster_labels
            
        except Exception as e:
            print()
            print(f"Ocurrió un error durante el entrenamiento de KMeans: {e}")
            print("===================================================")
    
    # ********************************************************************

    def segment_customers(self):
        """
        Segmenta a los clientes en grupos y asigna categorías basadas en RFM.
        """
        try:
            # Variables R, F y M
            variables = ['s_r', 's_f', 's_m']

            # Lista para almacenar las varianzas por cada clúster (corresponden a los pesos)
            variances_by_cluster = []

            # Iterar a través de cada clúster
            for cluster in self.data['cluster'].unique():
                cluster_data = self.data[self.data['cluster'] == cluster][variables]
                cluster_variances = cluster_data.var()
                variances_by_cluster.append(cluster_variances)

            # Crear un DataFrame con las varianzas por cada clúster
            variance_df = pd.DataFrame(variances_by_cluster, index=self.data['cluster'].unique(), columns=variables)

            # Normalizar las varianzas por clúster para que sumen 1
            normalized_variances = variance_df.div(variance_df.sum(axis=1), axis=0)

            # Agregar los pesos
            for cluster in self.data['cluster'].unique():
                mask = self.data['cluster'] == cluster
                # Agregar las columnas de pesos a cada cliente en el clúster
                self.data.loc[mask, 'omega_r'] = normalized_variances.loc[cluster, 's_r']
                self.data.loc[mask, 'omega_f'] = normalized_variances.loc[cluster, 's_f']
                self.data.loc[mask, 'omega_m'] = normalized_variances.loc[cluster, 's_m']

            # Agregar la columna s_rfm al DataFrame
            self.data['s_rfm'] = 0.0
            for cluster in self.data['cluster'].unique():
                mask = self.data['cluster'] == cluster
                self.data.loc[mask, 's_rfm'] = (
                    - self.data.loc[mask, 's_r'] * self.data.loc[mask, 'omega_r'] +
                    self.data.loc[mask, 's_f'] * self.data.loc[mask, 'omega_f'] +
                    self.data.loc[mask, 's_m'] * self.data.loc[mask, 'omega_m']
                )

            # Calcular los valores medios de s_r, s_f y s_m por cluster
            means_by_cluster = self.data.groupby('cluster').agg({'s_r': 'mean', 's_f': 'mean', 's_m': 'mean'})

            # Calcular las medias totales de s_r, s_f y s_m
            mean_s_r = self.data['s_r'].mean()
            mean_s_f = self.data['s_f'].mean()
            mean_s_m = self.data['s_m'].mean()

            # Definir una función para asignar la categoría
            def assign_category(row):
                ClusterMeans = means_by_cluster.loc[row['cluster']]  # Obtener las medias del clúster correspondiente
                if (ClusterMeans['s_r'] < mean_s_r and
                    ClusterMeans['s_f'] > mean_s_f and
                    ClusterMeans['s_m'] > mean_s_m):
                    return 'clientes de mayor contribucion'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] > mean_s_f and
                      ClusterMeans['s_m'] > mean_s_m):
                    return 'clientes importantes a recordar'
                elif (ClusterMeans['s_r'] < mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] > mean_s_m):
                    return 'clientes importantes para desarrollo'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] > mean_s_m):
                    return 'clientes importantes para retencion'
                elif (ClusterMeans['s_r'] < mean_s_r and
                      ClusterMeans['s_f'] > mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'clientes potenciales'
                elif (ClusterMeans['s_r'] < mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'clientes recientes'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] > mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'clientes de mantenimiento general'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'clientes con baja actividad'

            # Aplicar la función para asignar la categoría
            self.data['categoria'] = self.data.apply(assign_category, axis=1)

            # Función para asignar valores a la columna 'descripcion'
            def asignar_descripcion(categoria):
                if categoria == 'clientes de mayor contribucion':
                    return 'r ↓ f ↑ m ↑'
                elif categoria == 'clientes importantes a recordar':
                    return 'r ↑ f ↑ m ↑'
                elif categoria == 'clientes importantes para desarrollo':
                    return 'r ↓ f ↓ m ↑'
                elif categoria == 'clientes importantes para retencion':
                    return 'r ↑ f ↓ m ↑'
                elif categoria == 'clientes potenciales':
                    return 'r ↓ f ↑ m ↓'
                elif categoria == 'clientes recientes':
                    return 'r ↓ f ↓ m ↓'
                elif categoria == 'clientes de mantenimiento general':
                    return 'r ↑ f ↑ m ↓'
                elif categoria == 'clientes con baja actividad':
                    return 'r ↑ f ↓ m ↓'
                else:
                    return ''  # Manejar categorías no especificadas

            # Aplicar la función a la columna 'categoría' para crear la nueva columna 'descripcion'
            self.data['descripcion'] = self.data['categoria'].apply(asignar_descripcion)

            # Inicializar el escalador
            scaler = MinMaxScaler()

            # Escalar la columna s_rfm
            self.data['s_rfm'] = scaler.fit_transform(self.data[['s_rfm']])

            # Eliminar las columnas no deseadas
            drop_columns = ['s_r', 's_f', 's_m', 'cluster', 'omega_r', 'omega_f', 'omega_m', 
                            'recencia_boxcox', 'frecuencia_boxcox', 'monto_boxcox']
            self.data = self.data.drop(columns=drop_columns)

            # Devolver el DataFrame con todas las columnas originales más las agregadas
            return self.data
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante la segmentación: {e}")
            print("========================================")
    
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# ====================
# Modelo Abandonadores
# ====================

class ChurnPredictorRfmLr:
    """
    Clase para predecir la tasa de abandono de clientes utilizando un 
    modelo de regresión logística basado en RFM.
    """

    def __init__(self, random_state=42):
        """
        Inicializa el ChurnPredictorRfmLr con la semilla aleatoria 
        para reproducibilidad.

        Args:
            random_state (int, optional): Semilla aleatoria para 
            reproducibilidad. Valor por defecto es 42.
        """
        self.model = None
        self.random_state = random_state
        
    def set_seed(self, seed):
        """
        Fija la semilla aleatoria para numpy y random.

        Args:
            seed (int): Semilla aleatoria para reproducibilidad.
        """
        np.random.seed(seed)
        random.seed(seed)

    def preprocess_data(self, input_data):
        """
        Preprocesa los datos de entrada agregando columnas 'churn' y 'delta_compra_zscore' 
        basada en la comparación de 'recencia' con 'latencia_promedio' para los clientes de baja actividad.

        Args:
            input_data (DataFrame): Datos de entrada.

        Returns:
            DataFrame: Datos preprocesados con las columnas 'churn' y 'delta_compra_zscore' agregadas.
        """
        try:
            # Crear la columna delta_compra
            input_data['delta_compra'] = input_data['recencia'] - input_data['latencia_promedio']

            # Crear la columna churn usando el criterio de Z-Score > 1
            input_data['churn'] = ((input_data['categoria'] == 'clientes con baja actividad') & 
                                   (input_data['delta_compra'] > 0) &
                                   (input_data['s_rfm'] < input_data[input_data['categoria'] == 
                                    'clientes con baja actividad']['s_rfm'].mean())).astype(int)

            return input_data

        except Exception as e:
            print(f"Ocurrió un error durante el preprocesamiento de los datos: {e}")
            return pd.DataFrame()
    
    # ********************************************************************
    
    def train(self, input_data, test_size=0.2, 
              sampling_strategy=1.0, n_iter=30, cv_folds=5):
        """
        Entrena el modelo usando los datos de entrada.

        Args:
            input_data (DataFrame): Datos de entrada para entrenamiento.
            test_size (float): Proporción del conjunto de datos para incluir en la división de prueba.
            sampling_strategy (float): Estrategia de muestreo para RandomUnderSampler.
            n_iter (int): Número de ajustes de parámetros que se muestrean.
            cv_folds (int): Número de pliegues en la validación cruzada.

        Returns:
            tuple: X_test, y_test para evaluación.
        """
        try:
            self.set_seed(self.random_state)

            # Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
            X = input_data[['recencia', 'delta_compra', 'frecuencia', 'monto', 's_rfm']]
            y = input_data['churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

            # Realizar submuestreo de la clase mayoritaria en el conjunto de entrenamiento
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

            # Generar una lista de valores uniformemente espaciados para C
            C_values = np.random.uniform(0.01, 100, 100)

            # Definir el espacio de búsqueda de hiperparámetros
            param_distributions = {
                'C': C_values,  # Lista de valores de C uniformemente espaciados
                'solver': ['liblinear', 'saga']
            }

            # Configurar RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=LogisticRegression(random_state=self.random_state),
                param_distributions=param_distributions,
                n_iter=n_iter,  # Número de combinaciones a probar
                cv=cv_folds,    # Número de folds para validación cruzada
                scoring='accuracy',
                random_state=self.random_state  # Fijar semilla para reproducibilidad
            )

            # Entrenar el modelo
            random_search.fit(X_resampled, y_resampled)

            # Guardar el mejor modelo
            self.model = random_search.best_estimator_

            # Imprimir los mejores parámetros encontrados
            print(f"Best parameters found: {random_search.best_params_}")

            return X_test, y_test
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante el entrenamiento de LogisticRegression: {e}")
            print("===================================================")
            return pd.DataFrame(), pd.Series()
    
    # ********************************************************************
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el rendimiento del modelo en los datos de prueba.

        Args:
            X_test (DataFrame): Las características de prueba.
            y_test (Series): Las etiquetas verdaderas para el conjunto de prueba.

        Returns:
            tuple: Una tupla que contiene las etiquetas predichas y las probabilidades predichas para el conjunto de prueba.
        """
        try:
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]

            # Mostrar la matriz de confusión en el conjunto de prueba
            conf_matrix = confusion_matrix(y_test, y_pred)
            print('Matriz de Confusión:')
            print('====================')
            print(pd.DataFrame(conf_matrix, columns=['Predicted 1', 'Predicted 0'], index=['Actual 1', 'Actual 0']))
            print()
            print('Indicadores de rendimiento:')
            print('===========================')
            # Calcular y mostrar los indicadores
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}\n')

            return y_pred, y_prob
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante la evaluación de modelo: {e}")
            print("===================================================")
            return np.array([]), np.array([])
    
    # ********************************************************************
    
    def plot_roc_curve(self, y_test, y_prob):
        """
        Visualiza la curva ROC.

        Args:
            y_test (Series): Las etiquetas verdaderas para el conjunto de prueba.
            y_prob (array): Las probabilidades predichas para el conjunto de prueba.
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.title('Curva ROC')
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.legend(loc='lower right')
            plt.show()
            
        except Exception as e:
            print()
            print(f"Ocurrió un error durante la curva ROC: {e}")
            print("===========================================")
            return pd.DataFrame()

    def plot_ks_curve(self, y_test, y_prob):
        """
        Visualiza el test KS.

        Args:
            y_test (Series): Las etiquetas verdaderas para el conjunto de prueba.
            y_prob (array): Las probabilidades predichas para el conjunto de prueba.
        """
        try:
            ks_statistic, ks_p_value = ks_2samp(y_prob[y_test == 0], y_prob[y_test == 1])

            plt.figure(figsize=(10, 6))

            sns.histplot(y_prob[y_test == 0], bins=50, label='~ Churn', kde=True, color='skyblue', alpha=0.7)
            sns.histplot(y_prob[y_test == 1], bins=50, label='Churn', kde=True, color='salmon', alpha=0.7)

            plt.title('Distribuciones de Probabilidades para Churn y No Churn')
            plt.xlabel('Probabilidad Predicha de Churn')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.show()

            # CDF Plot
            plt.figure(figsize=(10, 6))

            # Calcular y graficar ECDF para No Churn
            x_no_churn = np.sort(y_prob[y_test == 0])
            y_no_churn = np.arange(1, len(x_no_churn) + 1) / len(x_no_churn)
            plt.plot(x_no_churn, y_no_churn, label='No Churn', color='blue')

            # Calcular y graficar ECDF para Churn
            x_churn = np.sort(y_prob[y_test == 1])
            y_churn = np.arange(1, len(x_churn) + 1) / len(x_churn)
            plt.plot(x_churn, y_churn, label='Churn', color='red')

            plt.title('Empirical Cumulative Distribution Function (ECDF)')
            plt.xlabel('Probabilidad Predicha de Churn')
            plt.ylabel('ECDF')
            plt.legend()

            plt.show()

            print(f'KS Statistic: {ks_statistic:.4f}')
            print(f'KS p-value: {ks_p_value:.4f}')
            print()
            
        except Exception as e:
            print()
            print(f"Ocurrió un error durante el test K-S: {e}")
            print("=========================================")
            return pd.DataFrame()
    
    # ********************************************************************
    
    def predict(self, perfomance_data, churn_threshold=0.5):
        """
        Predice las probabilidades y etiquetas de churn para los datos de entrada.

        Args:
            perfomance_data (DataFrame): Datos de entrada para la predicción.
            churn_threshold (float): Umbral para clasificar churn.

        Returns:
            DataFrame: Datos de entrada con columnas de probabilidad y predicción añadidas.
        """
        try:
            self.set_seed(self.random_state)

            # Calcular las probabilidades de abandono
            probabilities = self.model.predict_proba(perfomance_data[['recencia', 'delta_compra', 
                                                                      'frecuencia', 'monto', 's_rfm']])[:, 1]
            # Clasificar como abandono o no abandono basado en el umbral
            predictions = (probabilities >= churn_threshold).astype(int)

            # Agregar las columnas de probabilidad y predicción al dataframe
            perfomance_data['prob_abandono'] = probabilities.round(5)
            perfomance_data['pred_abandono'] = predictions

            # Eliminar la columna 'churn'
            perfomance_data = perfomance_data.drop(columns=['churn', 'delta_compra'])

            return perfomance_data
        
        except Exception as e:
            print()
            print(f"Ocurrió un error durante la predicción: {e}")
            print("===========================================")
            return pd.DataFrame()

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# ====================
# Reglas de Asociación
# ====================


class AssociationRules:
    """
    Clase para generar reglas de asociación a partir de datos transaccionales
    utilizando el algoritmo Apriori.

    Attributes:
        sql_script (str): Ruta al archivo SQL que contiene la consulta para obtener los datos.
        user (str): Nombre de usuario para la conexión a la base de datos.
        password (str): Contraseña para la conexión a la base de datos.
        level_column (str): Nombre de la columna que contiene el nivel de categorización de productos.
        csv_file_replace (str, optional): Ruta al archivo .csv que contiene los reemplazos.
        csv_file_avoid (str, optional): Ruta al archivo .csv que contiene los registros a evitar.
        data (DataFrame): DataFrame que contiene los datos preprocesados de transacciones.
        rules (DataFrame): DataFrame que contiene las reglas de asociación generadas.
    """
    def __init__(self, sql_script, user, password, level_column, csv_file_replace=None, csv_file_avoid=None):
        self.sql_script = sql_script
        self.user = user
        self.password = password
        self.level_column = level_column
        self.csv_file_replace = csv_file_replace
        self.csv_file_avoid = csv_file_avoid
        self.data = self.preprocess_data()
        self.rules = None

    def preprocess_data(self):
        """
        Realiza el preprocesamiento de los datos de clientes obtenidos de la base de datos.

        Returns:
            DataFrame: DataFrame preprocesado de transacciones.
        """
        try:
            # Conectar a la base de datos
            conn = redshift_connector.connect(
                host='datalake-dw55-prod.ctnjyflrnhjv.us-east-1.redshift.amazonaws.com',
                database='arsuperlake_prod',
                port=5439,
                user=self.user,
                password=self.password
            )

            # Ejecutar el script del query
            cursor = conn.cursor()
            cursor.execute(self.sql_script)
            input_data = cursor.fetch_dataframe()
            conn.close()

            # Procesamiento adicional de la columna nivel
            input_data[self.level_column] = input_data[
                self.level_column].apply(lambda x: ' '.join(x.split('_')))
            input_data[self.level_column] = input_data[
                self.level_column].apply(lambda x: re.sub(r'\d+', '', x))
            input_data[self.level_column] = input_data[
                self.level_column].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

            # Si se proporciona un archivo .csv de reemplazos, realizar los reemplazos
            if self.csv_file_replace:
                reemplazos_df = pd.read_csv(self.csv_file_replace)
                reemplazos = dict(zip(reemplazos_df['original'], 
                                      reemplazos_df['reemplazo']))
                input_data[self.level_column] = input_data[self.level_column].replace(reemplazos)

            # Si se proporciona un archivo .csv con registros a evitar, eliminarlos
            if self.csv_file_avoid:
                evitar_df = pd.read_csv(self.csv_file_avoid)
                registros_a_evitar = evitar_df['evadir'].tolist()
                input_data = input_data[~input_data[self.level_column].isin(registros_a_evitar)]

            # Convertir datos a formato de transacciones
            te = TransactionEncoder()
            te_ary = te.fit_transform(input_data.groupby('dw_ticket')[self.level_column].apply(list).values)
            tx = pd.DataFrame(te_ary, columns=te.columns_)

            tx.insert(0, 'dw_ticket', input_data['dw_ticket'].unique())
            tx = tx.astype(bool)

            return tx

        except Exception as e:
            print(f"Ocurrió un error durante el preprocesamiento de los datos: {e}")
            return pd.DataFrame()

    def generate_rules(self, support_threshold=0.01, 
                       length_threshold=None, 
                       lift_threshold=1.0, 
                       confidence_threshold=0.5):
        """
        Genera reglas de asociación utilizando el algoritmo Apriori.

        Args:
            support_threshold (float, optional): Umbral de soporte mínimo para los conjuntos de elementos. Por defecto es 0.01.
            length_threshold (int, optional): Longitud mínima de los conjuntos de elementos para generar reglas. Por defecto es 2.
            lift_threshold (float, optional): Umbral mínimo de lift para las reglas generadas. Por defecto es 1.0.
            confidence_threshold (float, optional): Umbral mínimo de confianza para las reglas generadas. Por defecto es 0.5.

        Returns:
            DataFrame: DataFrame que contiene las reglas de asociación generadas.
        """
        try:
            frequent_itemsets = apriori(self.data.drop(['dw_ticket'], axis=1),
                                        min_support=support_threshold,
                                        use_colnames=True,
                                        max_len=length_threshold)

            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_threshold)

            rules = rules[rules['confidence'] >= confidence_threshold]

            table_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

            table_rules['antecedents'] = table_rules['antecedents'].apply(lambda x: tuple(x))
            table_rules['consequents'] = table_rules['consequents'].apply(lambda x: tuple(x))

            self.rules = table_rules
            return table_rules

        except Exception as e:
            print(f"Ocurrió un error durante la generación de reglas de asociación: {e}")
            return pd.DataFrame()
    
    def plot_association_graph(self):
        """
        Visualiza un gráfico de asociación ponderado.
        """
        if self.rules is None or self.rules.empty:
            print("No hay reglas para visualizar. Asegúrese de generar reglas primero.")
            return

        try:
            G = nx.DiGraph()

            for _, row in self.rules.iterrows():
                antecedent = tuple(row['antecedents'])
                consequent = tuple(row['consequents'])
                G.add_edge(antecedent, consequent, weight=row['lift'])

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=35)

            node_labels = {k: str(k) for k in G.nodes()}

            nx.draw(G, pos, with_labels=True, labels=node_labels, font_size=10, 
                    node_size=800, node_color='skyblue', font_color='black', 
                    edgecolors='black', font_weight='bold', edge_color='gray',
                    width=1, alpha=0.7, arrowsize=20, arrowstyle='->')

            edge_labels = {(k[0], k[1]): round(v, 2) for k, v in nx.get_edge_attributes(G, 'weight').items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=15)

            plt.title('Weighted Association Graph', fontsize=20)
            plt.show()

        except Exception as e:
            print(f"Ocurrió un error al generar el gráfico de asociación: {e}")

    def plot_top_lift_rules(self, product, top_n):
        """
        Visualiza un gráfico de las reglas de asociación principales con mayor lift para un producto específico.

        Args:
            product (str): Nombre del producto de interés.
            top_n (int): Número de reglas principales a mostrar.
        """
        if self.rules is None or self.rules.empty:
            print("No hay reglas para visualizar. Asegúrese de generar reglas primero.")
            return

        try:
            product_rules = self.rules[
                (self.rules['antecedents'].apply(lambda x: product in x) | 
                 self.rules['consequents'].apply(lambda x: product in x))
            ]

            top_rules = product_rules.sort_values(by='lift', ascending=False).head(top_n)

            top_rules['rule'] = top_rules['antecedents'].astype(str) + ' -> ' + top_rules['consequents'].astype(str)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='lift', y='rule', data=top_rules, palette='viridis', edgecolor='black')
            plt.title(f'Top Rules with Highest Lift for {product}', fontsize=20)
            plt.xlabel('Lift', fontsize=20)
            plt.ylabel('Association Rule', fontsize=20)
            plt.show()

        except Exception as e:
            print(f"Ocurrió un error al generar el gráfico de las reglas: {e}")

    def plot_weighted_association_graph(self, chosen_product, top_n):
        """
        Visualiza un gráfico de asociación ponderado para un producto específico.

        Args:
            chosen_product (str): Producto seleccionado.
            top_n (int): Número de reglas principales a considerar.
        """
        if self.rules is None or self.rules.empty:
            print("No hay reglas para visualizar. Asegúrese de generar reglas primero.")
            return

        try:
            selected_product_rules = self.rules[
                self.rules['antecedents'].apply(lambda x: chosen_product in x) | 
                self.rules['consequents'].apply(lambda x: chosen_product in x)
            ]

            sorted_rules = selected_product_rules.sort_values(by='lift', ascending=False)
            top_rules = sorted_rules.head(top_n)

            G = nx.DiGraph()

            for _, row in top_rules.iterrows():
                antecedent = ', '.join(product for product in row['antecedents'] if product != chosen_product)
                consequent = ', '.join(product for product in row['consequents'] if product != chosen_product)
                G.add_edge(antecedent, consequent, weight=row['lift'])

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=35)

            nx.draw(
                G, 
                pos, 
                with_labels=True, 
                font_size=10, 
                font_color='black', 
                node_size=1000, 
                node_color='orange', 
                font_weight='bold', 
                edge_color='black',
                edgecolors='black'
            )

            edge_labels = {(antecedent, consequent): f"{lift:.2f}" for (antecedent, consequent, lift) in G.edges(data='weight')}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            plt.title(f'Weighted Association Graph for {chosen_product} Relationships', fontsize=20)
            plt.show()

        except Exception as e:
            print(f"Ocurrió un error al generar el gráfico de asociación ponderado: {e}")

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# ========================
# Sistema de Recomendación
# ========================

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# ========================
# Cargar archivos y tablas
# ========================

class Uploader:
    """
    Clase para subir un DataFrame a un bucket de S3 en formato CSV
    y luego cargarlo en Redshift.

    Args:
        bucket_name (str): Nombre del bucket de S3.
        s3_path (str): Ruta en S3 donde se guardará el archivo y su nombre.
        redshift_user (str): Usuario de Redshift.
        redshift_password (str): Contraseña de Redshift.
        redshift_table_name (str): Nombre de la tabla en Redshift.
    """

    def __init__(self, bucket_name, s3_path, redshift_user, redshift_password, redshift_table_name):
        self.bucket_name = bucket_name
        self.s3_path = s3_path
        self.redshift_user = redshift_user
        self.redshift_password = redshift_password
        self.redshift_table_name = redshift_table_name

    def upload_dataframe_to_s3(self, dataframe):
        """
        Guarda el DataFrame como un archivo CSV y lo sube al bucket de S3.

        Args:
            dataframe (pd.DataFrame): DataFrame que se desea guardar.

        Returns:
            None
        """
        file_name = os.path.basename(self.s3_path)
        
        # Guardar DataFrame como CSV
        dataframe.to_csv(file_name, index=False)
        
        # Subir el archivo al bucket de S3
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(file_name, self.bucket_name, self.s3_path)
            print(f'\nArchivo {file_name} subido a S3 en la ruta {self.s3_path}')
        except Exception as e:
            print(f'Error al subir el archivo a S3: {e}')
        finally:
            # Eliminar el archivo temporal
            os.remove(file_name)
    
    def load_to_redshift(self):
        """
        Carga el archivo desde S3 a Redshift.

        Returns:
            None
        """
        # Configuración de la conexión a Redshift
        redshift_config = {
            'host': 'datalake-dw55-prod.ctnjyflrnhjv.us-east-1.redshift.amazonaws.com',
            'database': 'arsuperlake_prod',
            'port': 5439,
            'user': self.redshift_user,
            'password': self.redshift_password
        }

        # Establecer la conexión a Redshift
        try:
            conn = redshift_connector.connect(**redshift_config)

            # Crear el objeto cursor
            cur = conn.cursor()

            # Crear el comando COPY
            copy_query = f""" COPY lk_analytics.{self.redshift_table_name}
                              FROM 's3://{self.bucket_name}/{self.s3_path}'
                              IAM_ROLE 'arn:aws:iam::191690883584:role/datalake-dw55-prod-redshift-service-role'
                              CSV
                              IGNOREHEADER 1
                              DELIMITER ',';"""

            # Ejecutar el comando COPY
            cur.execute(copy_query)

            # Hacer commit para aplicar los cambios
            conn.commit()

            print(f'Carga de datos desde S3 a Redshift completada para la tabla {self.redshift_table_name}')
        except Exception as e:
            print(f'Error al cargar datos en Redshift: {e}')
        finally:
            # Cerrar el cursor y la conexión
            cur.close()
            conn.close()