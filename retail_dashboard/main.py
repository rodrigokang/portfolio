# <<<<<<<<<<<<<<<<<<<<<<<<<<< Descipction >>>>>>>>>>>>>>>>>>>>>>>>>>> #
#
# This Flask application provides a RESTful API to interact with the 
# Northwind database. It utilizes SQLAlchemy as the ORM to manage }
# database operations.
#
# =================================================================== #

# Import Flask and SQLAlchemy
from flask import Flask, jsonify
from models import db, Customer, Order, OrderDetail, Product, Category  # Import relevant ORM models

# Create the Flask app and configure the database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/RJKANG/Desktop/portfolio/retail_dashboard/northwind.db'
db.init_app(app)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #
# Customer
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

def serialize_customer(data):
    """
    Serializes a customer object into a dictionary format.

    Args:
        data (Customer): The customer object.

    Returns:
        dict: A dictionary representing the customer data.
    """
    return {
        'CustomerID': data.CustomerID,
        'CustomerName': data.CustomerName,
        'ContactName': data.ContactName,
        'Address': data.Address,
        'City': data.City,
        'PostalCode': data.PostalCode,
        'Country': data.Country
    }

@app.route('/api/customers', methods=['GET'])
def get_customers():
    """
    Fetches data from the Customers table.

    Returns:
        Response: A JSON response containing a list of serialized customer data.
    """
    try:
        customers = Customer.query.all()
        customers_list = [serialize_customer(c) for c in customers]
        return jsonify(customers_list)
    except Exception as e:
        print(f"Error fetching customers: {e}")
        return jsonify({'error': str(e)}), 500


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #
# Products
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# Function to serialize SQLAlchemy objects into JSON
def serialize_combined_data(data):
    """
    Serializes a combined query result into a dictionary format.

    Args:
        data (SQLAlchemy Result): The combined query result containing data from multiple tables.

    Returns:
        dict: A dictionary representing the combined data.
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

# Route to get the combined product, order, and customer details
@app.route('/api/products', methods=['GET'])
def get_customer_products():
    """
    Fetches combined data from Customers, Orders, OrderDetails, Products, and Categories tables.

    Returns:
        Response: A JSON response containing a list of serialized combined data.
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

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)