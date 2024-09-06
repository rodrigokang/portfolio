# Retail Customer Segmentation Dashboard

## Overview

This project presents a comprehensive solution for customer segmentation in a retail environment. The dashboard integrates K-Means clustering to segment customers into different clusters based on their Recency-Frequency-Monetary (RFM) scores. This targeted approach helps in understanding and analyzing customer behavior effectively.

The project leverages the **Northwind** database as the data source, with **Flask** serving as the backend for API creation and **Streamlit** for the frontend. HTML and CSS are used in combination to enhance the user interface, offering a sleek and intuitive experience.

## Features

- **Customer Segmentation**: Implemented K-Means clustering to segment customers based on their RFM scores.
- **Interactive Dashboard**: Built with Streamlit, featuring real-time visualizations of customer segments and insights.
- **Backend API**: Developed using Flask to handle data management and model interactions.
- **User Interface**: Enhanced with HTML and CSS for a polished and responsive design.

## Technologies

- **Backend**: Flask, Python
- **Frontend**: Streamlit, HTML, CSS
- **Database**: Northwind (SQL Server)
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://
2. Navigate into the project directory:
   ```bash
   cd repository
3. Install the required packages and activate de environment:
   ```bash
   conda env create -f environment.yml
   conda activate retail_dashboard_env
4. Run the Flask backend:
   ```bash
   python app.py
5. Run the Streamlit dashboard:
   ```bash
   streamlit run dashboard.py
6. If you make any changes to the database models, you can handle migrations using the following commands:
   ```bash
   flask db init # Initialize migrations (only the first time)
   flask db migrate -m "Migration Message" # Create a migration whenever the models are updated
   flask db upgrade # Apply the migration to the database