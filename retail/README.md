# CRM with AI Features

This repository contains the development of a CRM application with AI features, including RFM models, Churn prediction, CLV estimation, Recommendation System, chatbot integration using ChatGPT API, Association Rules with graph representations, and many other features.

## Table of Contents

1. [Features](#features)
2. [Data Engineering](#data-engineering)
3. [Machine Learning Engineering](#machine-learning-engineering)
4. [Frontend Development](#frontend-development)
5. [UX/UI Design](#uxui-design)

## Features

- **RFM Models**: Recency, Frequency, and Monetary value analysis to segment customers.
- **Churn Prediction**: Predicting customer churn using logistic regression and other models.
- **CLV Estimation**: Customer Lifetime Value estimation to identify valuable customers.
- **Recommendation System**: Product recommendations using collaborative filtering and other techniques.
- **Chatbot Integration**: Using the ChatGPT API to handle customer queries and provide recommendations.
- **Association Rules**: Identifying patterns and relationships in customer purchase behavior with graphical representations.

## Data Engineering

### Design
The database is designed to handle both CRM and eCommerce functionalities, with the following schema:

1. **Users**: Stores user information.
2. **Customers**: Stores customer details.
3. **Products**: Stores product information.
4. **Categories**: Categorizes products.
5. **Orders**: Records customer orders.
6. **OrderDetails**: Provides detailed information about each order.
7. **Transactions**: Records payment transactions.
8. **Carts**: Stores shopping cart information.
9. **CartDetails**: Details of products in shopping carts.

### Creation
The database is created using SQL scripts, ensuring referential integrity and optimized performance.

### Simulation
Simulated datasets are generated to mimic real-world data, ensuring that models can be trained and tested effectively.

### Implementation
The database is implemented using PostgreSQL, with necessary indexing and performance tuning.

## Machine Learning Engineering

### Backend
The backend is developed using Flask (Python), providing a robust framework for API development and model integration.

### API Design
APIs are designed to handle various functionalities such as:
- Fetching and updating customer data.
- Processing orders and transactions.
- Integrating machine learning models for real-time predictions and recommendations.

## Frontend Development

### Framework
The frontend is developed using React, with JavaScript, HTML, and CSS for a dynamic and responsive user interface.

### Components
Key components include:
- **Dashboard**: Displays key metrics and insights.
- **Customer Management**: Allows viewing and managing customer information.
- **Product Management**: Enables managing product listings.
- **Order Processing**: Handles order creation and management.
- **Chatbot Interface**: Provides a conversational interface for customer interaction.

## UX/UI Design

### Tools
Figma and other design tools are used to create a user-friendly and visually appealing interface.

### Process
The design process involves:
- **User Research**: Understanding user needs and pain points.
- **Wireframing**: Creating low-fidelity wireframes to outline the basic structure.
- **Prototyping**: Developing high-fidelity prototypes to visualize the final design.
- **Testing**: Conducting usability tests to refine the design based on user feedback.

### Usage
- Access the application at `http://localhost:3000`.
- Request a user and password to log in.

For any questions or support, please contact us at [rodrigokang88@gmail.com].