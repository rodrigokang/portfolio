# <<<<<<<<<<<<<<<<<<<<<<<<<<< Description >>>>>>>>>>>>>>>>>>>>>>>>>>> #
#
# This Flask application provides a RESTful API to interact with the 
# Northwind database. It utilizes SQLAlchemy as the ORM to manage
# database operations and performs customer segmentation based on 
# country.
#
# =================================================================== #

# Import Flask, SQLAlchemy, and additional libraries
import os
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  # Import CORS for cross-origin requests
from models import db, Customer, Order, OrderDetail, Product, Category  # Import relevant ORM models
import pandas as pd
from segmentation import RFM  # Import the RFM class from the segmentation script

# Create the Flask app and configure the database
app = Flask(__name__)

# Define the base directory for the database
basedir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(basedir, 'northwind.db')

# Configure the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the Flask app
db = SQLAlchemy(app)

# Enable CORS for all domains
CORS(app)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #
# Helper Functions
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

def serialize_combined_data(data):
    """
    Serializes a result from a combined query into a dictionary format.

    This function converts a SQLAlchemy result object containing data from
    multiple tables into a dictionary with keys representing column names.

    Args:
        data (SQLAlchemy Result): The result of a query combining multiple tables.

    Returns:
        dict: A dictionary where each key corresponds to a column name and each value 
              corresponds to the column's data for the current row.
    """
    return {
        'CustomerID': data.CustomerID,
        'CustomerName': data.CustomerName,
        'ContactName': data.ContactName,
        'City': data.City,
        'Country': data.Country,
        'OrderDetailID': data.OrderDetailID,
        'OrderDate': data.OrderDate,
        'ProductName': data.ProductName,
        'CategoryName': data.CategoryName,
        'Quantity': data.Quantity,
        'Unit': data.Unit,
        'Price': data.Price
    }

def preprocess_data(df):
    """
    Preprocesses customer data to calculate RFM metrics: Recency, Frequency, and Monetary Value.

    This function transforms the raw customer data by calculating the recency of their last 
    purchase, the frequency of their purchases, and the total monetary value of their purchases.

    Args:
        df (pd.DataFrame): DataFrame containing raw customer data, including 'OrderDate', 
                           'Quantity', and 'Price' columns.

    Returns:
        pd.DataFrame: DataFrame with additional columns for 'Recency', 'Frequency', and 
                      'MonetaryValue', representing the calculated RFM metrics for each customer.
    
    Raises:
        ValueError: If any 'OrderDate' values cannot be converted to datetime.
    """
    # Convert 'OrderDate' to datetime if it is not already
    if not pd.api.types.is_datetime64_any_dtype(df['OrderDate']):
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
    
    # Check if conversion was successful
    if df['OrderDate'].isnull().any():
        raise ValueError("Some 'OrderDate' values could not be converted to datetime.")

    # Calculate recency, frequency, and monetary value
    snapshot_date = df['OrderDate'].max() + pd.DateOffset(days=1)  # Snapshot date as the day after the latest order

    # Recency: Number of days since last purchase
    df['Recency'] = (snapshot_date - df['OrderDate']).dt.days
    
    # Monetary value: Quantity * Price per order
    df['MonetaryValue'] = df['Quantity'] * df['Price']
    
    # Frequency: Number of orders per customer
    frequency = df.groupby('CustomerID').size().reset_index(name='Frequency')
    
    # Total monetary value per customer
    monetary_value = df.groupby('CustomerID')['MonetaryValue'].sum().reset_index(name='MonetaryValue')

    # Total recency per customer (last purchase date)
    recency = df.groupby('CustomerID')['Recency'].min().reset_index(name='Recency')

    # Combine all features into a single DataFrame
    features = recency.merge(frequency, on='CustomerID')
    features = features.merge(monetary_value, on='CustomerID')
    
    return features

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #
# Routes
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

@app.route('/api/products', methods=['GET'])
def get_customer_products():
    """
    Retrieves combined data from Customers, Orders, OrderDetails, Products, and Categories tables.

    This route fetches data from multiple related tables, combines it into a single query result,
    and returns it in JSON format.

    Returns:
        Response: A JSON response containing a list of serialized data, where each entry
                  represents a combined record from the query.
    """
    combined_data = db.session.query(
        Customer.CustomerID,
        Customer.CustomerName,
        Customer.ContactName,
        Customer.City,
        Customer.Country,
        OrderDetail.OrderDetailID,
        Order.OrderDate,
        Product.ProductName,
        Category.CategoryName,
        OrderDetail.Quantity,
        Product.Unit,
        Product.Price
    ).join(Order, Customer.CustomerID == Order.CustomerID) \
     .join(OrderDetail, Order.OrderID == OrderDetail.OrderID) \
     .join(Product, OrderDetail.ProductID == Product.ProductID) \
     .join(Category, Product.CategoryID == Category.CategoryID).all()

    return jsonify([serialize_combined_data(data) for data in combined_data])

@app.route('/api/segment', methods=['POST'])
def segment_customers():
    """
    Processes customer data based on the selected country filter, preprocesses it, and performs segmentation.

    This route accepts customer data and a country filter from the request, preprocesses the data
    to compute RFM metrics, applies segmentation using the RFM model, and returns the segmented data.

    Returns:
        Response: A JSON response containing a list of segmented customer data, where each entry
                  represents a customer and their assigned segment.
    
    Raises:
        ValueError: If required columns are missing or data processing fails.
    """
    try:
        # Get country filter from request JSON
        data = request.json
        country_filter = data.get('country', None)

        # Get the DataFrame directly from the request JSON
        df = pd.DataFrame(data.get('data', []))

        # Check if 'OrderDate' exists in the DataFrame
        if 'OrderDate' not in df.columns:
            raise ValueError("'OrderDate' column is missing from the data.")

        # Apply country filter if specified
        if country_filter and country_filter != "All countries":
            df = df[df['Country'] == country_filter]

        # Preprocess data to calculate recency, frequency, and monetary_value
        preprocessed_data = preprocess_data(df)

        # Convert datetime columns to string (optional step, depending on your downstream processing)
        for col in preprocessed_data.select_dtypes(include=[pd.Timestamp]).columns:
            preprocessed_data[col] = preprocessed_data[col].astype(str)

        # Pass the preprocessed data into your RFM segmentation model
        rfm = RFM(preprocessed_data)
        rfm.train_kmeans(n_clusters=8)  # Adjust clusters as needed
        rfm.segment_customers()

        # Get the segmented data
        segments = rfm.get_segments()

        # Return the segmented data as JSON
        return jsonify(segments.to_dict(orient='records'))  # Convert DataFrame to JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Host set to '0.0.0.0' to be accessible from outside