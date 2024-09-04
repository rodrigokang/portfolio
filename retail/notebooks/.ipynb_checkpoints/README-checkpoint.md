# Retail Customer Churn Prediction Dashboard

## Overview

This project presents a comprehensive solution for predicting customer churn in a retail environment. The dashboard integrates various machine learning models, including Logistic Regression, Gradient Boosting, Random Forest, Support Vector Machines, and KNN, all of which are based on a Recency-Frequency-Monetary (RFM) model. The customer base is segmented into different clusters using K-Means, providing a targeted approach to churn prediction.

The project leverages the **Northwind** database as the data source, with **Flask** serving as the backend for API creation and **Streamlit** for the frontend. HTML and CSS are used in combination to enhance the user interface, offering a sleek and intuitive experience.

## Features

- **Customer Segmentation**: Implemented K-Means clustering to segment customers based on their RFM scores.
- **Churn Prediction Models**:
  - Logistic Regression
  - Gradient Boosting
  - Random Forest
  - Support Vector Machines (SVM)
  - KNN
- **Interactive Dashboard**: Built with Streamlit, featuring real-time visualizations and predictions.
- **Backend API**: Developed using Flask to handle model predictions and data management.
- **User Interface**: Enhanced with HTML and CSS for a polished and responsive design.

## Technologies

- **Backend**: Flask, Python
- **Frontend**: Streamlit, HTML, CSS
- **Database**: Northwind (SQL Server)
- **Machine Learning**: Scikit-learn, TensorFlow
- **Data Visualization**: Matplotlib, Seaborn

## Installation

To run the project locally, follow these steps:

1. Clone the repository: