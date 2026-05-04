# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Description >>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

# * Name: "Clustering for retail"
# * Owner: Rodrigo J. Kang
# * Description: This script contains K-Mean algorithm to perform
#                customer segmentation for retail.

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# Suppress code suggestions
# -------------------------
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# Modules for establishing a database connection
# ==============================================

import psycopg2
from sqlalchemy import create_engine

# =================================
# Import general-purpose libraries
# =================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ==============================
# Import libraries for RFM model
# ==============================

# Data processing
# ---------------
from scipy.stats import boxcox
from scipy.stats import skew
import random

# Machine learning modules
# ------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

class RFM:
    """
    Class for customer segmentation using RFM (Recency, Frequency, Monetary Value).

    Args:
        sql_script (str): Path to the SQL file containing the query to retrieve customer data.
        user (str, optional): Username for the database connection.
        password (str, optional): Password for the database connection.

    Attributes:
        sql_script (str): Path to the SQL file containing the query to retrieve customer data.
        user (str): Username for the database connection.
        password (str): Password for the database connection.
        data (DataFrame): DataFrame containing the processed customer data.
    """

    def __init__(self, sql_script, user, password):
        self.sql_script = sql_script
        self.user = user
        self.password = password
        self.data = self.preprocess_data()
    
    # ********************************************************************

    def preprocess_data(self):
        """
        Performs preprocessing of customer data retrieved from the database.

        Returns:
            DataFrame: Preprocessed DataFrame with RFM variables and necessary transformations.
        """
        try:
            # Connect to the database
            conn = psycopg2.connect(
                host='datalake-dw55-prod.ctnjyflrnhjv.us-east-1.redshift.amazonaws.com',
                database='arsuperlake_prod',
                port=5439,
                user=self.user,
                password=self.password
            )

            # Execute the query script
            cursor = conn.cursor()
            cursor.execute(self.sql_script)

            # Fetch data and convert to DataFrame
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            input_data = pd.DataFrame(data, columns=columns)

            # Close the connection
            cursor.close()
            conn.close()

            # Convert 'recency', 'frequency', and 'monetary' values to float
            input_data['recencia'] = input_data['recencia'].astype(int)
            input_data['frecuencia'] = input_data['frecuencia'].astype(int)
            input_data['monto'] = input_data['monto'].astype(float)

            # Box-Cox Transformation
            input_data['recencia_boxcox'], LambdaRecencia = boxcox(input_data['recencia'] + 1)
            input_data['frecuencia_boxcox'], LambdaFrecuencia = boxcox(input_data['frecuencia'] + 1)
            input_data['monto_boxcox'], LambdaMonto = boxcox(input_data['monto'] + 1)

            # Standardization
            scaler = StandardScaler()
            input_data[['s_r', 's_f', 's_m']] = scaler.fit_transform(
                input_data[['recencia_boxcox', 
                            'frecuencia_boxcox', 
                            'monto_boxcox']])

            return input_data
        
        except Exception as e:
            print()
            print(f"An error occurred during data preprocessing: {e}")
            print("=========================================================")
            return pd.DataFrame()
    
    # ********************************************************************
    
    def elbow_method(self):
        """
        Performs the Elbow Method to determine the optimal number of clusters.
        """
        try:
            data = self.data[['s_r', 's_f', 's_m']]

            inertia_values = []

            for k in range(2, 11):  # Test with k from 2 to 10 clusters
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(data)

                inertia = kmeans.inertia_
                inertia_values.append(inertia)

            self.elbow_plot = plt.figure(figsize=(8, 5))
            plt.plot(range(2, 11), inertia_values, marker='o')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Sum of Squared Distances (Inertia)')
            plt.title('Elbow Method')
            plt.grid(True)

            return self.elbow_plot
        
        except Exception as e:
            print()
            print(f"An error occurred during the elbow method: {e}")
            print("===========================================")
            return pd.DataFrame()
    
    # ********************************************************************

    def random_search_kmeans(self, n_iter=10, sample_ratio=0.01, cluster_range=(2, 11)):
        """
        Performs a random search to determine the optimal number of clusters for K-Means
        using different evaluation metrics on random subsets of the data.

        Args:
            n_iter (int, optional): Number of random search iterations. Default is 10.
            sample_ratio (float, optional): Proportion of data to use in each sample. Default is 0.01.
            cluster_range (tuple, optional): Range of cluster numbers to evaluate. Default is (2, 11).

        Returns:
            None: Prints the metric results for each iteration.
        """
        try:
            # Define the size of each sample
            sample_size = int(len(self.data) * sample_ratio)

            # Initialize lists to store the results of the indices
            cluster_numbers = list(range(cluster_range[0], cluster_range[1]))
            instance_numbers = list(range(1, n_iter + 1))

            for instance in instance_numbers:
                max_ch = -1
                max_silhouette = -1
                min_db = float('inf')
                max_ch_cluster = -1
                max_silhouette_cluster = -1
                min_db_cluster = -1

                # Take a random sample of the dataset
                sample_indexes = np.random.choice(len(self.data), sample_size, replace=False)
                data_sample = self.data.iloc[sample_indexes][['s_r', 's_f', 's_m']]

                for k in cluster_numbers:
                    kmeans = KMeans(n_clusters=k, random_state=0)
                    labels = kmeans.fit_predict(data_sample)

                    ch_metric = calinski_harabasz_score(data_sample, labels)
                    db_metric = davies_bouldin_score(data_sample, labels)
                    silhouette = silhouette_score(data_sample, labels)

                    # Update max and min values
                    if ch_metric > max_ch:
                        max_ch = ch_metric
                        max_ch_cluster = k
                    if silhouette > max_silhouette:
                        max_silhouette = silhouette
                        max_silhouette_cluster = k
                    if db_metric < min_db:
                        min_db = db_metric
                        min_db_cluster = k

                print(f'Instance {instance}:')
                print('------------')
                print(f'Maximum Calinski-Harabasz: Optimal number of clusters = {max_ch_cluster}, Metric = {max_ch}')
                print(f'Maximum Silhouette: Optimal number of clusters = {max_silhouette_cluster}, Metric = {max_silhouette}')
                print(f'Minimum Davies-Bouldin: Optimal number of clusters = {min_db_cluster}, Metric = {min_db}')
        
        except Exception as e:
            print()
            print(f"An error occurred during the random search for clusters: {e}")
            print("==========================================================")
    
    # ********************************************************************
    
    def train_kmeans(self, n_clusters):
        """
        Trains the KMeans model to segment customers into groups.

        Args:
            n_clusters (int): Number of clusters to create.
        """
        try:
            # Train the KMeans model
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(self.data[['s_r', 's_f', 's_m']])
            self.data['cluster'] = cluster_labels
            
        except Exception as e:
            print()
            print(f"An error occurred during KMeans training: {e}")
            print("===================================================")
    
    # ********************************************************************

    def segment_customers(self):
        """
        Segments customers into groups and assigns categories based on RFM.
        """
        try:
            # R, F, and M variables
            variables = ['s_r', 's_f', 's_m']
    
            # List to store variances for each cluster (correspond to weights)
            variances_by_cluster = []
    
            # Iterate through each cluster
            for cluster in self.data['cluster'].unique():
                cluster_data = self.data[self.data['cluster'] == cluster][variables]
                cluster_variances = cluster_data.var()
                variances_by_cluster.append(cluster_variances)
    
            # Create a DataFrame with variances for each cluster
            variance_df = pd.DataFrame(variances_by_cluster, index=self.data['cluster'].unique(), columns=variables)
    
            # Normalize the variances by cluster so that they sum to 1
            normalized_variances = variance_df.div(variance_df.sum(axis=1), axis=0)
    
            # Add the weights
            for cluster in self.data['cluster'].unique():
                mask = self.data['cluster'] == cluster
                # Add the weight columns to each customer in the cluster
                self.data.loc[mask, 'omega_r'] = normalized_variances.loc[cluster, 's_r']
                self.data.loc[mask, 'omega_f'] = normalized_variances.loc[cluster, 's_f']
                self.data.loc[mask, 'omega_m'] = normalized_variances.loc[cluster, 's_m']
    
            # Add the s_rfm column to the DataFrame
            self.data['s_rfm'] = 0.0
            for cluster in self.data['cluster'].unique():
                mask = self.data['cluster'] == cluster
                self.data.loc[mask, 's_rfm'] = (
                    - self.data.loc[mask, 's_r'] * self.data.loc[mask, 'omega_r'] +
                    self.data.loc[mask, 's_f'] * self.data.loc[mask, 'omega_f'] +
                    self.data.loc[mask, 's_m'] * self.data.loc[mask, 'omega_m']
                )
    
            # Calculate the mean values of s_r, s_f, and s_m by cluster
            means_by_cluster = self.data.groupby('cluster').agg({'s_r': 'mean', 's_f': 'mean', 's_m': 'mean'})
    
            # Calculate the overall means of s_r, s_f, and s_m
            mean_s_r = self.data['s_r'].mean()
            mean_s_f = self.data['s_f'].mean()
            mean_s_m = self.data['s_m'].mean()
    
            # Define a function to assign the category
            def assign_category(row):
                ClusterMeans = means_by_cluster.loc[row['cluster']]  # Get the means of the corresponding cluster
                if (ClusterMeans['s_r'] < mean_s_r and
                    ClusterMeans['s_f'] > mean_s_f and
                    ClusterMeans['s_m'] > mean_s_m):
                    return 'high contribution customers'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] > mean_s_f and
                      ClusterMeans['s_m'] > mean_s_m):
                    return 'important customers to remember'
                elif (ClusterMeans['s_r'] < mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] > mean_s_m):
                    return 'important customers for development'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] > mean_s_m):
                    return 'important customers for retention'
                elif (ClusterMeans['s_r'] < mean_s_r and
                      ClusterMeans['s_f'] > mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'potential customers'
                elif (ClusterMeans['s_r'] < mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'recent customers'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] > mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'general maintenance customers'
                elif (ClusterMeans['s_r'] > mean_s_r and
                      ClusterMeans['s_f'] < mean_s_f and
                      ClusterMeans['s_m'] < mean_s_m):
                    return 'low activity customers'
    
            # Apply the function to assign the category
            self.data['category'] = self.data.apply(assign_category, axis=1)
    
            # Function to assign values to the 'description' column
            def assign_description(category):
                if category == 'high contribution customers':
                    return 'r ↓ f ↑ m ↑'
                elif category == 'important customers to remember':
                    return 'r ↑ f ↑ m ↑'
                elif category == 'important customers for development':
                    return 'r ↓ f ↓ m ↑'
                elif category == 'important customers for retention':
                    return 'r ↑ f ↓ m ↑'
                elif category == 'potential customers':
                    return 'r ↓ f ↑ m ↓'
                elif category == 'recent customers':
                    return 'r ↓ f ↓ m ↓'
                elif category == 'general maintenance customers':
                    return 'r ↑ f ↑ m ↓'
                elif category == 'low activity customers':
                    return 'r ↑ f ↓ m ↓'
                else:
                    return ''  # Handle unspecified categories
    
            # Apply the function to the 'category' column to create the new 'description' column
            self.data['description'] = self.data['category'].apply(assign_description)
    
            # Initialize the scaler
            scaler = MinMaxScaler()
    
            # Scale the s_rfm column
            self.data['s_rfm'] = scaler.fit_transform(self.data[['s_rfm']])
    
            # Drop unwanted columns
            drop_columns = ['s_r', 's_f', 's_m', 'cluster', 'omega_r', 'omega_f', 'omega_m', 
                            'recencia_boxcox', 'frecuencia_boxcox', 'monto_boxcox']
            self.data = self.data.drop(columns=drop_columns)
    
            # Return the DataFrame with all original columns plus the added ones
            return self.data
    
        except Exception as e:
            print()
            print(f"An error occurred during segmentation: {e}")
            print("========================================")

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #