# <<<<<<<<<<<<<<<<<<<<<<<<<<< Description >>>>>>>>>>>>>>>>>>>>>>>>>>> #
#
# This module defines the SQLAlchemy ORM models for the Northwind 
# database. It sets up the schema and relationships for the database 
# tables used in the application.
#
# =================================================================== #

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Class for the Categories table
class Category(db.Model):
    """
    Represents a category of products in the Northwind database.
    
    Attributes:
        id (int): Primary key for the category.
        category_name (str): Name of the category.
        description (str): Description of the category.
    """
    __tablename__ = 'Categories'
    CategoryID = db.Column(db.Integer, primary_key=True)
    CategoryName = db.Column(db.String(255), nullable=False)
    Description = db.Column(db.Text)


# Class for the Customers table
class Customer(db.Model):
    """
    Represents a customer in the Northwind database.
    
    Attributes:
        id (int): Primary key for the customer.
        company_name (str): Name of the customer's company.
        contact_name (str): Name of the contact person.
        address (str): Customer's address.
        city (str): City of the customer.
        postal_code (str): Postal code of the customer.
        country (str): Country of the customer.
    """
    __tablename__ = 'Customers'
    CustomerID = db.Column(db.Integer, primary_key=True)
    CustomerName = db.Column(db.String(255), nullable=False)
    ContactName = db.Column(db.String(255))
    Address = db.Column(db.String(255))
    City = db.Column(db.String(255))
    PostalCode = db.Column(db.String(10))
    Country = db.Column(db.String(255))


# Class for the Employees table
class Employee(db.Model):
    """
    Represents an employee in the Northwind database.
    
    Attributes:
        id (int): Primary key for the employee.
        last_name (str): Employee's last name.
        first_name (str): Employee's first name.
        birth_date (str): Employee's birth date.
        photo (str): Photo of the employee.
        notes (str): Additional notes about the employee.
    """
    __tablename__ = 'Employees'
    EmployeeID = db.Column(db.Integer, primary_key=True)
    LastName = db.Column(db.String(255), nullable=False)
    FirstName = db.Column(db.String(255), nullable=False)
    BirthDate = db.Column(db.String(255))
    Photo = db.Column(db.Text)
    Notes = db.Column(db.Text)


# Class for the OrderDetails table
class OrderDetail(db.Model):
    """
    Represents the details of an order in the Northwind database.
    
    Attributes:
        id (int): Primary key for the order detail.
        order_id (int): Foreign key to the Orders table.
        product_id (int): Foreign key to the Products table.
        quantity (int): Quantity of the product ordered.
    """
    __tablename__ = 'OrderDetails'
    OrderDetailID = db.Column(db.Integer, primary_key=True)
    OrderID = db.Column(db.Integer, db.ForeignKey('Orders.id'), nullable=False)
    ProductID = db.Column(db.Integer, db.ForeignKey('Products.id'), nullable=False)
    Quantity = db.Column(db.Integer, nullable=False)


# Class for the Orders table
class Order(db.Model):
    """
    Represents an order placed by a customer in the Northwind database.
    
    Attributes:
        id (int): Primary key for the order.
        customer_id (int): Foreign key to the Customers table.
        employee_id (int): Foreign key to the Employees table.
        order_date (str): Date the order was placed.
        shipper_id (int): Foreign key to the Shippers table (shipping company).
    """
    __tablename__ = 'Orders'
    OrderID = db.Column(db.Integer, primary_key=True)
    CustomerID = db.Column(db.Integer, db.ForeignKey('Customers.id'), nullable=False)
    EmployeeID = db.Column(db.Integer, db.ForeignKey('Employees.id'), nullable=False)
    OrderDate = db.Column(db.String(255))
    ShipperID = db.Column(db.Integer, db.ForeignKey('Shippers.id'), nullable=False)


# Class for the Products table
class Product(db.Model):
    """
    Represents a product in the Northwind database.
    
    Attributes:
        id (int): Primary key for the product.
        product_name (str): Name of the product.
        supplier_id (int): Foreign key to the Suppliers table.
        category_id (int): Foreign key to the Categories table.
        unit (str): Unit measurement for the product.
        price (float): Price of the product.
    """
    __tablename__ = 'Products'
    ProductID = db.Column(db.Integer, primary_key=True)
    ProductName = db.Column(db.String(255), nullable=False)
    SupplierID = db.Column(db.Integer, db.ForeignKey('Suppliers.id'), nullable=False)
    CategoryID = db.Column(db.Integer, db.ForeignKey('Categories.id'), nullable=False)
    Unit = db.Column(db.String(255))
    Price = db.Column(db.Float, nullable=False)


# Class for the Shippers table
class Shipper(db.Model):
    """
    Represents a shipping company in the Northwind database.
    
    Attributes:
        id (int): Primary key for the shipper.
        company_name (str): Name of the shipping company.
        phone (str): Phone number of the shipping company.
    """
    __tablename__ = 'Shippers'
    ShipperID = db.Column(db.Integer, primary_key=True)
    ShipperName = db.Column(db.String(255), nullable=False)
    Phone = db.Column(db.String(255))


# Class for the Suppliers table
class Supplier(db.Model):
    """
    Represents a supplier in the Northwind database.
    
    Attributes:
        id (int): Primary key for the supplier.
        company_name (str): Name of the supplier's company.
        contact_name (str): Name of the contact person at the supplier.
        address (str): Address of the supplier.
        city (str): City where the supplier is located.
        postal_code (str): Postal code of the supplier.
        country (str): Country where the supplier is located.
        phone (str): Phone number of the supplier.
    """
    __tablename__ = 'Suppliers'
    SupplierID = db.Column(db.Integer, primary_key=True)
    SupplierName = db.Column(db.String(255), nullable=False)
    ContactName = db.Column(db.String(255))
    Address = db.Column(db.String(255))
    City = db.Column(db.String(255))
    PostalCode = db.Column(db.String(10))
    Country = db.Column(db.String(255))
    Phone = db.Column(db.String(255))