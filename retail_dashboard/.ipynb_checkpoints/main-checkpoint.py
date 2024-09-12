# Import Flask and SQLAlchemy
from flask import Flask
from models import db, Customer, Order, Product  # Import your ORM models

# Create the Flask app and configure the database
app = Flask(__name__)
# Configure the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/RJKANG/Desktop/portfolio/retail_dashboard/northwind.db'
db.init_app(app)

# Create a context to interact with the database
with app.app_context():
    try:
        # Test: Query all customers
        customers = Customer.query.all()
        for customer in customers:
            print(customer.CustomerName)
        
        # Test: Query specific orders
        orders = Order.query.filter_by(CustomerID=1).all()
        for order in orders:
            print(f"Order ID: {order.OrderID}, Date: {order.OrderDate}")
    except Exception as e:
        print(f"An error occurred: {e}")