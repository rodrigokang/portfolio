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

# =================================
# Import general-purpose libraries
# =================================

import numpy as np
import pandas as pd

# ==============================
# Import libraries for RFM model
# ==============================

# Data processing
# ---------------
from scipy.stats import boxcox
from scipy.stats import skew

# Machine learning modules
# ------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

class RFM:
    """
    Class for customer segmentation using RFM (Recency, Frequency, Monetary Value).

    Args:
        df (DataFrame): DataFrame containing the customer data with Recency, Frequency, and MonetaryValue.

    Attributes:
        data (DataFrame): DataFrame containing the processed customer data.
    """

    def __init__(self, df):
        self.data = df
        self.data = self.preprocess_data()
    
    # ********************************************************************

    def preprocess_data(self):
        """
        Performs preprocessing of customer data.

        Returns:
            DataFrame: Preprocessed DataFrame with RFM variables and necessary transformations.
        """
        try:
            # Convert 'Recency', 'Frequency', and 'MonetaryValue' values to float
            self.data['Recency'] = self.data['Recency'].astype(int)
            self.data['Frequency'] = self.data['Frequency'].astype(int)
            self.data['MonetaryValue'] = self.data['MonetaryValue'].astype(float)

            # Box-Cox Transformation
            self.data['recency_boxcox'], _ = boxcox(self.data['Recency'] + 1)
            self.data['frequency_boxcox'], _ = boxcox(self.data['Frequency'] + 1)
            self.data['monetary_value_boxcox'], _ = boxcox(self.data['MonetaryValue'] + 1)

            # Standardization
            scaler = StandardScaler()
            self.data[['s_r', 's_f', 's_m']] = scaler.fit_transform(
                self.data[['recency_boxcox', 
                           'frequency_boxcox', 
                           'monetary_value_boxcox']])

            return self.data
        
        except Exception as e:
            print(f"An error occurred during data preprocessing: {e}")
            return pd.DataFrame()
    
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
            print(f"An error occurred during KMeans training: {e}")
    
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
            self.data[['s_rfm']] = scaler.fit_transform(self.data[['s_rfm']])
    
        except Exception as e:
            print(f"An error occurred during customer segmentation: {e}")
    
    # ********************************************************************
    
    def get_segments(self):
        """
        Returns a DataFrame with customer segments and their descriptions, 
        renaming specific columns for clarity.
        
        Returns:
            DataFrame: DataFrame with customer segments and their descriptions.
        """
        try:
            # Ensure CustomerID is included in the returned DataFrame
            segments_df = self.data[['CustomerID', 'Recency', 'Frequency', 'MonetaryValue', 
                                     'cluster', 'category', 'description', 's_rfm']]
            
            # Rename columns
            segments_df = segments_df.rename(columns={
                'CustomerID': 'CustomerID',
                'cluster': 'Cluster',
                'category': 'Category',
                'description': 'Description',
                's_rfm': 'Score'
            })
            
            return segments_df
        except Exception as e:
            print(f"An error occurred while retrieving segments: {e}")
            return pd.DataFrame()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #